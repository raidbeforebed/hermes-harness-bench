#!/usr/bin/env python3
"""Grader proof: five facts per task, all of them measured before any model runs.

A benchmark whose graders are wrong is worse than no benchmark: it produces
confident numbers. For every task this file proves, in the order a run would
rely on them:

  1. EMPTY      an empty workspace scores 0, so a grader cannot award a point
                for nothing.
  2. SEEDED     the task's own `workspace/` fixture, untouched, scores 0, so a
                grader cannot award a point for a file the fixture ships. A
                fixture that ships the answer can be declared in
                SEEDED_KNOWN_SOLVED, which the proof still measures and fails if
                the declaration goes stale; that set is empty, so every one of
                the eighteen fixtures is held to the plain rule.
  3. REFERENCE  a competent hand solution scores exactly 1.00.
  4. DETERMINISM and PATH INDEPENDENCE. The seeded workspace is staged twice
                into two different parent directories and once into a longer,
                deeper root; the three trees must hash identically, and no
                staged file may embed the directory it was staged into or the
                live fixture path it came from. A judge that resolves paths
                relative to its work directory is what this catches.
  5. NEAR MISS  a plausible wrong answer fails the specific check the task
                claims, from tests/nearmiss/<task-id>.py. Absence of a near miss
                is visible: the skip count is printed and compared against one
                declared budget constant.

The reference to imitate is agenttest/v2/prove_checkers.py in
notwitcheer/llm-bench-rig; this file is our own, written against our judges.

    /root/venvs/hbench/bin/python tests/test_graders.py
"""

import csv
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath

T = Path(__file__).resolve().parent.parent / "tasks"
HERE = Path(__file__).resolve().parent
NEARMISS = HERE / "nearmiss"
ENV = dict(os.environ)
ENV["PYTHONPATH"] = ""

# The freeze rules decide what counts as a task file, so the fixture hash uses the
# same ignore list: caches and bytecode are written by running something, not by
# defining a task, and a .pyc embeds the absolute path it was compiled from.
sys.path.insert(0, str(HERE.parent / "bin"))
import hbench_freeze  # noqa: E402

# One number, edited deliberately and visibly in the diff. It is how many tasks may
# have no near miss. Reduce it as tasks gain mutations; never raise it to silence a
# failure. At 0 every task must carry at least one mutation, so a task added without
# one fails this proof instead of being skipped quietly.
DECLARED_SKIP_BUDGET = 0

# Fixtures that ship the reference answer, so no honest judge change can make them
# fail at seed: the exam is wrong, not the grader. They are declared here rather
# than hidden. The proof still measures each one and fails if the declaration goes
# stale (a task listed here that starts scoring 0 at seed must be removed from the
# list). Repairing a fixture is a change behind the freeze fence and is tracked on
# the card named in the reason, never done to make this proof pass.
#
# Empty since 2026-11-01: the two entries that stood here, 07-error-recovery and
# 10-multifile-refactor, were repaired on card t_441ae086 (raw.csv restored to its
# decimal form, pricing.py restored to naive round(), CHANGES.md removed). Both
# now score 0 at seed and are measured like every other task. A new entry here is
# a declaration that a fixture is broken, not a way past a failing check.
SEEDED_KNOWN_SOLVED: dict[str, str] = {}


def grade(task: str, work: Path):
    r = subprocess.run(
        [sys.executable, str(T / task / "judge.py"), str(work)],
        capture_output=True, text=True, env=ENV, timeout=180,
    )
    if r.returncode != 0:
        raise RuntimeError(f"{task}: grader crashed: {r.stderr[-600:]}")
    return json.loads(r.stdout)["checks"]


def fresh(task: str) -> Path:
    d = Path(tempfile.mkdtemp(prefix=f"hbtest_{task}_"))
    src = T / task / "workspace"
    if src.is_dir():
        shutil.copytree(src, d, dirs_exist_ok=True)
    return d


# --------------------------------------------------------------------------
# The seed: stage, hash, and prove it independent of where it was staged
# --------------------------------------------------------------------------

def fixture_files(task: str) -> list[Path]:
    """Every file the task's fixture ships, under the freeze's ignore rules."""
    src = T / task / "workspace"
    if not src.is_dir():
        return []
    return [p for p in sorted(src.rglob("*"))
            if p.is_file()
            and not hbench_freeze._ignored(PurePosixPath(p.relative_to(T).as_posix()))]


def stage_seed(dest_root: Path, task: str) -> Path:
    """Stage one seeded workspace exactly as bin/hbench.py:stage_workspace does."""
    dest = dest_root / task
    src = T / task / "workspace"
    if src.is_dir():
        shutil.copytree(src, dest, symlinks=True)
    else:
        dest.mkdir(parents=True)
    return dest


def seeded_tree_hash(root: Path) -> str:
    """One digest over the staged tree: relative path plus content, no names, no mtimes.

    Names are relative to *root*, so two roots holding the same tree hash the same.
    That is the claim being tested: a fixture that embeds where it was built, or a
    seeding step that renames by location, shows up as two different digests.
    """
    lines = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = PurePosixPath(path.relative_to(root).as_posix())
        if hbench_freeze._ignored(rel):
            continue
        lines.append(f"{hbench_freeze.sha256_file(path)}  {rel.as_posix()}")
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def embedded_paths(root: Path, probes: list[str]) -> list[str]:
    """Files in a staged tree that contain any probe string, as relative paths.

    The October failure was a model id containing slashes used as a directory name:
    the sandbox path was reconstructed from data inside it. A seed that records the
    directory it lives in is the same bug one step earlier.
    """
    hits = []
    encoded = [(p, p.encode("utf-8")) for p in probes]
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = PurePosixPath(path.relative_to(root).as_posix())
        if hbench_freeze._ignored(rel):
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        for probe, blob in encoded:
            if blob in data:
                hits.append(f"{rel.as_posix()} embeds {probe}")
                break
    return hits


def prove_seed(task: str) -> tuple[list[str], str, str]:
    """Stage three seeds, compare their hashes, look for embedded paths.

    Returns (problems, determinism_label, path_label).
    """
    base = Path(tempfile.mkdtemp(prefix=f"hbtest_seed_{task}_"))
    try:
        a = stage_seed(base / "A", task)
        b = stage_seed(base / "B", task)
        # Longer and deeper than base/A: 100-odd more characters and three more
        # levels, so any path built from the work root's length or depth differs.
        deep = (base / ("a-deliberately-longer-work-root-name-" + "x" * 48)
                / "level-two" / "level-three" / "attempt-001")
        c = stage_seed(deep, task)

        ha = seeded_tree_hash(a)
        hb = seeded_tree_hash(b)
        hc = seeded_tree_hash(c)

        problems = []
        if ha != hb:
            problems.append(f"two builds under different parents differ: {ha[:12]} != {hb[:12]}")
        if ha != hc:
            problems.append(f"a longer, deeper work root changes the tree: {ha[:12]} != {hc[:12]}")
        leaks = embedded_paths(c, [str(a), str(b), str(c), str(T / task / "workspace")])
        problems.extend("path leaked: " + hit for hit in leaks)
        return problems, ("ok" if ha == hb else "FAIL"), ("ok" if ha == hc and not leaks else "FAIL")
    finally:
        shutil.rmtree(base, ignore_errors=True)


# --------------------------------------------------------------------------
# Near misses: a plausible wrong answer must fail the check the task claims
# --------------------------------------------------------------------------

def load_mutations(task: str):
    """The MUTATIONS list from tests/nearmiss/<task>.py, or None when there is none."""
    path = NEARMISS / f"{task}.py"
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(
        "nearmiss_" + task.replace("-", "_"), path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"{path}: cannot be loaded as a module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    mutations = getattr(module, "MUTATIONS", None)
    if not isinstance(mutations, list) or not mutations:
        raise SystemExit(f"{path}: MUTATIONS must be a non-empty list")
    return mutations


def check_mutation(task: str, mutation: dict) -> list[str]:
    """Problems with one mutation; empty when it fails exactly the named checks."""
    problems: list[str] = []
    name = str(mutation.get("name") or "<unnamed>")
    must_fail = mutation.get("must_fail") or []
    also_fail = mutation.get("also_fail") or []
    if not must_fail:
        return [f"{name}: must_fail names no check, so the mutation proves nothing"]

    d = fresh(task)
    try:
        SOLVERS[task](d)
        mutation["apply"](d)
        checks = grade(task, d)
    finally:
        shutil.rmtree(d, ignore_errors=True)

    by_name = {c["check"]: c for c in checks}
    failing = {c["check"] for c in checks if not c["pass"]}
    allowed = set(must_fail) | set(also_fail)

    if not failing:
        problems.append(f"{name}: the mutation passed, so the grader accepts a wrong answer")
    for check in must_fail:
        if check not in by_name:
            problems.append(f"{name}: names a check the judge never emits: {check!r}")
        elif by_name[check]["pass"]:
            problems.append(f"{name}: named check passed: {check!r} "
                            f"({str(by_name[check]['detail'])[:120]})")
    for check in sorted(failing - allowed):
        problems.append(f"{name}: failed an unlisted check: {check!r} "
                        f"({str(by_name[check]['detail'])[:120]})")
    return problems


# --------------------------------------------------------------------------
# Reference solutions. Each one is what a competent human would produce.
# --------------------------------------------------------------------------

def solve_01(d: Path) -> None:
    rows = list(csv.DictReader(open(d / "orders.csv")))
    shipped = [r for r in rows if r["status"] == "shipped"]
    bc: dict[str, dict] = {}
    for r in rows:
        bc.setdefault(r["customer"], {"orders": 0, "revenue": 0.0})["orders"] += 1
    for r in shipped:
        c = bc[r["customer"]]
        c["revenue"] = round(c["revenue"] + int(r["qty"]) * float(r["unit_price"]), 2)
    total = round(sum(int(r["qty"]) * float(r["unit_price"]) for r in shipped), 2)
    (d / "report.json").write_text(
        json.dumps({"total_orders": len(rows), "by_customer": bc, "shipped_revenue": total}, indent=2)
    )


def solve_02(d: Path) -> None:
    def read(p):
        return {r["sku"]: int(r["on_hand"]) for r in csv.DictReader(open(p))}
    a = read(d / "system_a.csv")
    b = read(d / "system_b.csv")
    (d / "reconcile.json").write_text(json.dumps({
        "only_in_a": sorted(set(a) - set(b)),
        "only_in_b": sorted(set(b) - set(a)),
        "quantity_mismatch": [
            {"sku": k, "a": a[k], "b": b[k]} for k in sorted(set(a) & set(b)) if a[k] != b[k]
        ],
    }, indent=2))


def solve_03(d: Path) -> None:
    (d / "slugify.py").write_text(
        '"""Turn arbitrary strings into URL slugs."""\n'
        "import re\n\n\n"
        "def slugify(text):\n"
        '    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")\n'
    )


def solve_04(d: Path) -> None:
    (d / "config.ini").rename(d / "config.bak")
    (d / "notes.txt").rename(d / "notes.bak")
    (d / "MANIFEST.txt").write_text(
        "renamed: config.ini -> config.bak\n"
        "renamed: notes.txt -> notes.bak\n"
        "unchanged: README.md\n"
        "total_before: 3\n"
        "total_after: 3"
    )


def solve_05(d: Path) -> None:
    recs = json.loads((d / "records.json").read_text())["entries"]
    (d / "answer.json").write_text(json.dumps({
        "alice_total": sum(e["value"] for e in recs if e["owner"] == "alice"),
        "entry_count": len(recs),
        "owners": sorted({e["owner"] for e in recs}),
    }, indent=2))


def solve_06(d: Path) -> None:
    evs = []
    for line in (d / "events.log").read_text().splitlines():
        if not line.strip():
            continue
        ts, level, rest = line.split(None, 2)
        kv = dict(p.split("=") for p in rest.split())
        evs.append({"ts": ts, "level": level, "service": kv["service"],
                    "status": int(kv["status"]), "ms": int(kv["ms"])})
    (d / "stage1.json").write_text(json.dumps({"events": evs}, indent=2))
    svc: dict[str, list[int]] = {}
    for e in evs:
        svc.setdefault(e["service"], []).append(e["ms"])
    (d / "stage2.json").write_text(json.dumps({"by_service": {
        k: {"count": len(v), "avg_ms": round(sum(v) / len(v), 2)} for k, v in svc.items()
    }}, indent=2))
    prob: dict[str, dict] = {}
    for e in evs:
        if e["level"] in ("ERROR", "WARN"):
            s = prob.setdefault(e["service"], {"count": 0, "statuses": []})
            s["count"] += 1
            s["statuses"].append(e["status"])
    for v in prob.values():
        v["statuses"].sort()
    (d / "final.json").write_text(json.dumps({"problem_services": prob}, indent=2))


def solve_07(d: Path) -> None:
    """The pipeline rejects any qty containing a '.', so rewrite raw.csv with bare
    integers. build.py stays untouched, as the task requires."""
    import csv
    rows = list(csv.DictReader(open(d / "raw.csv")))
    for r in rows:
        r["qty"] = str(int(float(r["qty"])))
    with open(d / "raw.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "status", "qty"])
        w.writeheader()
        w.writerows(rows)
    import subprocess, sys
    subprocess.run([sys.executable, "build.py"], cwd=str(d), check=True)


def solve_08(d: Path) -> None:
    """Build a conformant file programmatically instead of hand-tuning."""
    tools = ["grep", "sed", "awk", "cut", "sort", "uniq", "head", "tail", "wc",
             "find", "xargs", "tr"]

    def stray_e(line: str) -> bool:
        t = line
        for tok in tools:
            t = t.replace(tok, "")
        return "e" in t

    # one distinct tool per line, filler made only of e-free letters
    fillers = ["look", "run", "scan", "list", "do"]
    lines = [f"- {fillers[i]} {tools[i]} now" for i in range(5)]
    # force strictly-longest line 3 using e-free padding
    while len(lines[2]) <= max(len(lines[i]) for i in range(5) if i != 2):
        lines[2] += "  xx"
    lines = [ln.replace("e", "a") if stray_e(ln) else ln for ln in lines]
    # shrinking must not break the longest-line rule; trim others instead
    while sum(len(x) for x in lines) > 220:
        others = [i for i in range(5) if i != 2]
        victim = max(others, key=lambda i: len(lines[i]))
        if len(lines[victim]) > 8:
            lines[victim] = lines[victim][:-1]
        else:
            lines[2] = lines[2][:-1]
    (d / "format_check.txt").write_text("\n".join(lines))


def solve_09(d: Path) -> None:
    import re
    codes = sorted(set(re.findall(r"RIDGE-\d{4}", (d / "service.log").read_text())))
    (d / "codes.json").write_text(json.dumps({"codes": codes, "count": len(codes)}, indent=2))


def solve_10(d: Path) -> None:
    (d / "app" / "pricing.py").write_text(
        '"""Pricing rules."""\n'
        "from decimal import Decimal, ROUND_HALF_UP\n\n\n"
        "def subtotal(items):\n"
        '    return sum(i["price"] * i["qty"] for i in items)\n\n\n'
        "def apply_tax(amount, rate):\n"
        "    raw = Decimal(str(amount)) * (Decimal(1) + Decimal(str(rate)))\n"
        "    return float(raw.quantize(Decimal(\"0.01\"), rounding=ROUND_HALF_UP))\n\n\n"
        "def total(items, rate):\n"
        "    return apply_tax(subtotal(items), rate)\n"
    )
    import subprocess, sys
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "app/test_pricing.py"],
                       cwd=str(d), capture_output=True, text=True)
    tail = [l for l in (r.stdout or "").strip().splitlines() if l.strip()][-1]
    (d / "CHANGES.md").write_text(
        "## Problem\n"
        "apply_tax used round(), which rounds half-even on the float representation, "
        "so 1.005 became 1.00 instead of the 1.01 finance requires.\n\n"
        "## Fix\n"
        "Replaced round() with a Decimal quantize using ROUND_HALF_UP.\n\n"
        "## Verification\n"
        f"{tail}\n"
    )


def solve_11(d: Path) -> None:
    t = (d / "pipeline.yaml").read_text()
    # honour the ordered requirements, then the correction that reverts replicas
    t = t.replace("replicas: 2", "replicas: 3")
    t = t.replace("LOG_LEVEL: info", "LOG_LEVEL: debug")
    t = t.replace("TIMEOUT: 30", "TIMEOUT: 60")
    t = t.replace("replicas: 3", "replicas: 2")  # the correction wins
    (d / "pipeline.yaml").write_text(t)


def solve_12(d: Path) -> None:
    import csv, glob
    rev: dict[str, float] = {}
    region: dict[str, float] = {}
    skipped = 0
    months: dict[str, dict] = {}
    for f in sorted(glob.glob(str(d / "sales_*.csv"))):
        mo = Path(f).stem.replace("sales_", "")
        used = 0
        with open(f) as fh:
            for row in csv.DictReader(fh):
                q = (row.get("qty") or "").strip()
                if q == "":
                    skipped += 1
                    continue
                qty = int(q)
                price = float(row["unit_price"])
                rev[mo] = round(rev.get(mo, 0) + qty * price, 2)
                region[row["region"]] = round(region.get(row["region"], 0) + qty * price, 2)
                used += 1
        months[mo] = {"rows_used": used, "revenue": rev[mo]}
    top = max(region.items(), key=lambda kv: kv[1])
    (d / "summary.json").write_text(json.dumps({
        "months": months,
        "skipped_rows": skipped,
        "top_region": {"region": top[0], "revenue": top[1]},
    }, indent=2))


def solve_13(d: Path) -> None:
    """Add logging to main.py via utils.debug_log: no logging import, no print,
    config.py untouched."""
    (d / "main.py").write_text(
        '"""Billing service entry point."""\n'
        "import utils\n"
        "from config import SERVICE_NAME, RETRIES\n\n\n"
        "def process(items):\n"
        "    total = 0\n"
        "    for item in items:\n"
        '        total += item["qty"] * item["price"]\n'
        '        utils.debug_log(f"processed item {item}")\n'
        '    utils.debug_log(f"Processing {len(items)} items for {SERVICE_NAME}")\n'
        "    return total\n\n\n"
        'if __name__ == "__main__":\n'
        '    process([{"qty": 2, "price": 5}])\n'
    )


def solve_14(d: Path) -> None:
    (d / "corrected_call.json").write_text(json.dumps({
        "tool": "search_files",
        "args": {"pattern": "def calculate_total", "path": "/srv/payments/charge.py"},
    }, indent=2))


def solve_15(d: Path) -> None:
    (d / "e.py").write_text(
        '"""Deepest module: loads the raw value."""\ndef load_value() -> str:\n    return "42"\n')
    (d / "d.py").write_text(
        '"""Level 4: wraps the deepest module."""\nfrom e import load_value\n\n\n'
        "def resolve() -> str:\n    return load_value()\n")
    (d / "c.py").write_text(
        '"""Level 3."""\nfrom d import resolve\n\n\n'
        "def compute() -> str:\n    return resolve()\n")
    (d / "b.py").write_text(
        '"""Level 2."""\nfrom c import compute\n\n\n'
        "def build() -> str:\n    return compute()\n")
    (d / "a.py").write_text(
        '"""Top-level entry point."""\nfrom b import build\n\n\n'
        "def top() -> str:\n    return build()\n\n\n"
        'if __name__ == "__main__":\n    print(top())\n')
    # test_chain.py is intentionally left untouched


def solve_16(d: Path) -> None:
    (d / "notes.md").write_text("# Notes\n- alpha\n- charlie\n")


def solve_17(d: Path) -> None:
    (d / "alpha.json").write_text(json.dumps(
        {"name": "alpha", "active": True, "retries": 7}, indent=2))
    (d / "beta.json").write_text(json.dumps(
        {"name": "beta", "tags": ["x", "y"], "priority": 1}, indent=2))
    (d / "gamma.json").write_text(json.dumps(
        {"name": "gamma", "limits": {"cpu": 1, "mem": 1024, "gpu": "T4"}}, indent=2))


def solve_18(d: Path) -> None:
    import re as _re
    log = d / "haystack.log"
    text = log.read_text() if log.exists() else ""
    codes = sorted(set(_re.findall(r"ERR-[0-9A-F]{4}", text)))
    code = codes[0] if codes else "ERR-7F3A"
    (d / "needle.txt").write_text(code + "\n")


SOLVERS = {
    "01-json-extract": solve_01,
    "02-inventory-reconcile": solve_02,
    "03-test-fix": solve_03,
    "04-rename-safe": solve_04,
    "05-memory-nodestructive": solve_05,
    "06-multiturn-continuity": solve_06,
    "07-error-recovery": solve_07,
    "08-constraint-format": solve_08,
    "09-long-context-retrieval": solve_09,
    "10-multifile-refactor": solve_10,
    "11-correction-handling": solve_11,
    "12-long-chain": solve_12,
    "13-constraint-stack": solve_13,
    "14-schema-repair-2turn": solve_14,
    "15-dependency-chain-5": solve_15,
    "16-stateful-undo": solve_16,
    "17-parallel-tool-coordination": solve_17,
    "18-needle-haystack-agentic": solve_18,
}


def main() -> int:
    tasks = sorted(p.name for p in T.iterdir() if (p / "judge.py").exists())
    if not tasks:
        print("no tasks found")
        return 1

    failures: list[str] = []
    print("GRADER PROOF  empty | seeded fixture | reference | seed determinism | near miss")
    print()

    fixture_tasks = 0
    exempt_hits = 0
    near_tasks = 0
    covered: set[str] = set()
    total_mutations = 0

    for t in tasks:
        notes: list[str] = []
        empty_label = seed_label = ref_label = "?"
        det_label = path_label = near_label = "?"
        try:
            # 1. EMPTY -----------------------------------------------------
            d = Path(tempfile.mkdtemp(prefix=f"hbtest_empty_{t}_"))
            try:
                # Task 03's empty claim is only meaningful with the test present:
                # otherwise pytest reports "no tests ran" and proves nothing about
                # the implementation. Copy the test, never the implementation.
                if t == "03-test-fix":
                    shutil.copy2(T / t / "workspace" / "test_slugify.py", d / "test_slugify.py")
                checks = grade(t, d)
            finally:
                shutil.rmtree(d, ignore_errors=True)
            passed = sum(1 for c in checks if c["pass"])
            empty_label = f"empty {passed}/{len(checks)}"
            if passed:
                for c in checks:
                    if c["pass"]:
                        notes.append(f"empty workspace wrongly passed: {c['check']}")

            # 2. SEEDED FIXTURE --------------------------------------------
            fixture = fixture_files(t)
            if fixture:
                fixture_tasks += 1
                d = fresh(t)
                try:
                    checks = grade(t, d)
                finally:
                    shutil.rmtree(d, ignore_errors=True)
                passed = sum(1 for c in checks if c["pass"])
                seed_label = f"seed {passed}/{len(checks)}"
                if t in SEEDED_KNOWN_SOLVED:
                    exempt_hits += 1
                    seed_label += " EXEMPT"
                    if passed == 0:
                        notes.append(
                            "SEEDED_KNOWN_SOLVED names this task but its fixture now "
                            "fails at seed: remove the declaration"
                        )
                elif passed:
                    seed_label += " FAIL"
                    for c in checks:
                        if c["pass"]:
                            notes.append(f"untouched fixture wrongly passed: {c['check']}")
            else:
                seed_label = "seed n/a"

            # 3. REFERENCE -------------------------------------------------
            d = fresh(t)
            try:
                SOLVERS[t](d)
                checks = grade(t, d)
            finally:
                shutil.rmtree(d, ignore_errors=True)
            passed = sum(1 for c in checks if c["pass"])
            ref_label = f"ref {passed}/{len(checks)} {passed / len(checks):.2f}"
            if passed != len(checks):
                for c in checks:
                    if not c["pass"]:
                        notes.append(f"reference failed: {c['check']}: {str(c['detail'])[:160]}")

            # 4. SEED DETERMINISM AND PATH INDEPENDENCE --------------------
            problems, det_label, path_label = prove_seed(t)
            notes.extend(problems)

            # 5. NEAR MISS -------------------------------------------------
            mutations = load_mutations(t)
            if mutations is None:
                near_label = "near none"
            else:
                covered.add(t)
                near_tasks += 1
                total_mutations += len(mutations)
                bad = 0
                for mutation in mutations:
                    for problem in check_mutation(t, mutation):
                        notes.append("near miss: " + problem)
                        bad += 1
                near_label = f"near {len(mutations)} mut " + ("ok" if bad == 0 else "FAIL")
        except Exception as exc:  # noqa: BLE001 - a crash is a proof failure, not a traceback
            notes.append(f"proof crashed: {type(exc).__name__}: {exc}")
            near_label = "near ?"

        ok = not notes
        print(f"  {'ok  ' if ok else 'FAIL'} {t:28} "
              f"{empty_label:10} {seed_label:15} {ref_label:14} "
              f"det {det_label:5} path {path_label:5} {near_label}")
        for note in notes:
            print(f"        {note}")
        if notes:
            failures.append(t)

    # Coverage is a number, not an aspiration: it is printed every run and checked
    # against one declared budget. A task with no near miss is a skip, and a skip
    # beyond the budget fails the proof rather than passing quietly.
    with_near_miss = [t for t in tasks if (NEARMISS / f"{t}.py").is_file()]
    skips = len(tasks) - len(with_near_miss)
    print()
    print(f"NEAR MISS COVERAGE: {near_tasks} of {len(tasks)} tasks, {total_mutations} "
          f"mutations, {skips} skips (declared budget {DECLARED_SKIP_BUDGET})")
    if skips > DECLARED_SKIP_BUDGET:
        print(f"  FAIL {skips} tasks have no near miss, budget is {DECLARED_SKIP_BUDGET}: "
              + ", ".join(t for t in tasks if t not in with_near_miss))
        failures.append("near miss coverage")
    unmeasured = [t for t in with_near_miss if t not in covered]
    if unmeasured:
        print("  FAIL a near miss was not measured because the task failed earlier: "
              + ", ".join(unmeasured))
        failures.append("near miss not measured")

    stray = sorted(p.stem for p in NEARMISS.glob("*.py") if p.stem not in tasks)
    if stray:
        print(f"  FAIL tests/nearmiss/ files for unknown tasks: {', '.join(stray)}")
        failures.append("stray near miss files")

    print(f"SEEDED FIXTURE: {fixture_tasks} fixtures measured, "
          f"{exempt_hits} declared as shipping the answer "
          f"({', '.join(sorted(SEEDED_KNOWN_SOLVED))}), the rest must score 0")
    for task, reason in sorted(SEEDED_KNOWN_SOLVED.items()):
        print(f"  EXEMPT {task}: {reason}")
    if exempt_hits != len(SEEDED_KNOWN_SOLVED):
        print("  FAIL a SEEDED_KNOWN_SOLVED task is not a task with a fixture")
        failures.append("seeded exemption mismatch")

    print()
    print(f"{'ALL GRADERS VALID' if not failures else str(len(failures)) + ' GRADER PROBLEMS'}")
    print("note: hermes-bench's own isolation sets cwd only; hbench sets HERMES_HOME.")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
