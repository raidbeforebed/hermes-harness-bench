#!/usr/bin/env python3
"""Grader self-test.

A benchmark whose graders are wrong is worse than no benchmark: it produces
confident numbers. Every judge.py must score 0 on an empty workspace and 1.0
on a known-good reference solution. Run this after touching any task.

    python tests/test_graders.py
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

T = Path(__file__).resolve().parent.parent / "tasks"
ENV = dict(os.environ)
ENV["PYTHONPATH"] = ""


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
}


def main() -> int:
    tasks = sorted(p.name for p in T.iterdir() if (p / "judge.py").exists())
    if not tasks:
        print("no tasks found")
        return 1
    failures = 0

    print("NEGATIVE  empty workspace must score 0")
    for t in tasks:
        d = Path(tempfile.mkdtemp(prefix=f"hbtest_neg_{t}_"))
        # task 03 needs the test file present to be meaningful, but the
        # implementation must still fail, so copy only the test file
        if t == "03-test-fix":
            shutil.copy2(T / t / "workspace" / "test_slugify.py", d / "test_slugify.py")
        checks = grade(t, d)
        passed = sum(1 for c in checks if c["pass"])
        ok = passed == 0
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {t:30} {passed}/{len(checks)} passed")
        if not ok:
            for c in checks:
                if c["pass"]:
                    print(f"        wrongly passed: {c['check']}")
        shutil.rmtree(d, ignore_errors=True)

    print("\nPOSITIVE  reference solution must score 1.00")
    for t in tasks:
        d = fresh(t)
        SOLVERS[t](d)
        checks = grade(t, d)
        passed = sum(1 for c in checks if c["pass"])
        ok = passed == len(checks)
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {t:30} {passed}/{len(checks)}")
        if not ok:
            for c in checks:
                if not c["pass"]:
                    print(f"        failed: {c['check']}: {str(c['detail'])[:160]}")
        shutil.rmtree(d, ignore_errors=True)

    print(f"\n{'ALL GRADERS VALID' if failures == 0 else str(failures) + ' GRADER PROBLEMS'}")
    print("note: every grader is asserted both ways: 0 on an empty workspace, 1.00 on a reference solution.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
