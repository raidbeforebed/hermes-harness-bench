#!/usr/bin/env python3
"""Publish finished hbench runs to the repo, in one command.

The gap this closes: exporting, charting, editing the README table, committing and
pushing were five manual steps. A run could finish, get reported in chat, and never
reach the published repository, which is exactly what happened on 2026-10-22.

    python3 tools/publish.py <run_id> [<run_id> ...]   # publish these runs
    python3 tools/publish.py --latest                   # newest run per model
    python3 tools/publish.py --dry-run <run_id>         # do everything but commit

Order matters and is enforced:

1. export    - a run with harness failures aborts the whole thing. A fault is not a
               model result, and publishing it as 0.00 is worse than not publishing.
2. charts    - regenerated from results/, so they cannot drift from the data.
3. README    - the results table between the RESULTS markers is rewritten from
               results/. Prose outside the markers is left alone, because that part
               is written by a person and should stay that way.
4. commit    - only if something changed.
5. push      - then verify the remote actually points at the new commit, because a
               push that reported success is not the same thing as a push that landed.

Exit codes: 0 published (or nothing to do), 1 aborted, 2 git failure.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as registry  # noqa: E402
import result_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"
DECISION_START = "<!-- DECISION:START -->"
DECISION_END = "<!-- DECISION:END -->"

# Display names come from tools/model_registry.py, the single source shared with
# tools/make_charts.py. They used to be duplicated here and there, and the two copies
# disagreed: the table printed a new model under its raw provider id while the chart
# builder dropped it entirely. A model missing from the registry still publishes, under
# its own name in grey, rather than silently vanishing.


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, **kw)


def head(short: bool = True) -> str:
    r = run(["git", "rev-parse", "--short" if short else "--verify", "HEAD"])
    return r.stdout.strip()


def fmt_duration(seconds: float) -> str:
    """Seconds up to ten minutes, because a mean wall of 106 s reads better than 2 min."""
    if seconds >= 5400:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def published_rows() -> list[dict]:
    """Every published record grouped by model AND configuration, in display order.

    Grouping by model alone is what merged LFM3 thinking on with LFM3 thinking off: one
    weights file, two instruments, one row. The configuration token comes from the record
    itself (tools/result_key.py), so a run from before the field existed lands in the
    unrecorded row rather than being folded into a named one.
    """
    grouped: dict[str, dict] = {}
    for f in glob.glob(str(RESULTS / "*.json")):
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        if not (d.get("model") and d.get("task")):
            continue
        key = result_key.key_of(d)
        row = grouped.get(key)
        if row is None:
            config = result_key.config_of(d)
            row = grouped[key] = {
                "key": key,
                "model": d["model"],
                "config": config,
                "label": registry.label(d["model"], config),
                "records": [],
            }
        row["records"].append(d)
    if not grouped:
        raise SystemExit("no results to build a table from")
    return sorted(grouped.values(),
                  key=lambda r: registry.sort_key(r["model"], r["config"]))


def table_from_results() -> str:
    """Rebuild the results table from results/*.json only."""
    rows: list[dict] = []
    for row in published_rows():
        model, rs = row["model"], row["records"]
        per_task: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            per_task[r["task"]].append(float(r.get("score") or 0.0))
        stable = sum(1 for v in per_task.values() if all(s >= 1.0 for s in v))
        passed = sum(1 for r in rs if float(r.get("score") or 0.0) >= 1.0)
        turns = [r["turns"] for r in rs if isinstance(r.get("turns"), int)]
        toks = [(r.get("tokens_in") or 0) + (r.get("tokens_out") or 0)
                for r in rs if r.get("tokens_in")]
        walls = [r["wall_seconds"] for r in rs if r.get("wall_seconds")]
        # Attempts per task, not the set of repeat labels: a three-repeat row has labels 1,2,3
        # and reads as "3"; the local row has one attempt on nine tasks and four on three, so
        # it reads as "1-4" and cannot be mistaken for four of everything.
        per_task_attempts = defaultdict(int)
        for r in rs:
            per_task_attempts[r["task"]] = max(per_task_attempts[r["task"]],
                                               r.get("repeat") or 1)
        reps = sorted(set(per_task_attempts.values()))
        faults: dict[str, int] = defaultdict(int)
        for r in rs:
            if r.get("harness_failed"):
                faults[(r.get("fault") or {}).get("class") or "unclassified"] += 1
        rows.append({
            "model": model,
            "label": row["label"],
            # A row whose tasks were attempted a different number of times must say so: "4"
            # over a set where nine tasks had one attempt reads as four of everything.
            "repeats": (str(reps[0]) if len(reps) == 1
                        else f"{reps[0]}-{reps[-1]}"),
            "faults": dict(faults),
            "passed": passed, "n": len(rs),
            "stable": stable, "tasks": len(per_task),
            "turns": statistics.mean(turns) if turns else None,
            "tokens": statistics.mean(toks) if toks else None,
            "wall": statistics.mean(walls) if walls else None,
            "total": sum(walls) if walls else None,
        })

    best = {
        "turns": min((r["turns"] for r in rows if r["turns"]), default=None),
        "tokens": min((r["tokens"] for r in rows if r["tokens"]), default=None),
        "wall": min((r["wall"] for r in rows if r["wall"]), default=None),
    }

    def bold(val, bestval, text):
        return f"**{text}**" if val is not None and bestval is not None and abs(val - bestval) < 1e-9 else text

    out = [
        START,
        "| Model | Repeats | Runs passed | Faults | Tasks all-pass | Mean turns | Mean tokens | Mean wall | Total |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        share = 100 * r["passed"] / r["n"]
        pct = (f"{r['passed']}/{r['n']} ({share:.0f}%)" if share == 100
               else f"{r['passed']}/{r['n']} ({share:.1f}%)") if r["n"] else "-"
        if r["passed"] == r["n"] and r["n"]:
            pct = f"**{pct}**"
        turns = bold(r["turns"], best["turns"], f"{r['turns']:.1f}") if r["turns"] else "-"
        toks = bold(r["tokens"], best["tokens"], f"{int(r['tokens']):,}") if r["tokens"] else "-"
        wall = bold(r["wall"], best["wall"], fmt_duration(r["wall"])) if r["wall"] else "-"
        total = fmt_duration(r["total"]) if r["total"] else "-"
        fault_txt = (", ".join(f"{k} {v}" for k, v in sorted(r["faults"].items()))
                     if r["faults"] else "-")
        out.append(f"| {r['label']} | {r['repeats']} | {pct} | {fault_txt} "
                   f"| {r['stable']}/{r['tasks']} "
                   f"| {turns} | {toks} | {wall} | {total} |")
    out.append(END)
    return "\n".join(out)


def rewrite_readme() -> bool:
    text = README.read_text()
    if START not in text or END not in text:
        raise SystemExit(
            f"README is missing the {START} / {END} markers.\n"
            "Add them around the results table before publishing."
        )
    block = table_from_results()
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), block, text, count=1, flags=re.S)
    if new == text:
        return False
    README.write_text(new)
    return True


# Two-sided 95% Student-t, keyed by the number of repeat means going into the interval.
# Three repeats is the house standard, so the table has to admit what three numbers say:
# t=4.303, not the 1.96 a normal approximation would pretend.
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365,
       9: 2.306, 10: 2.262, 15: 2.131, 20: 2.086}


def _t95(k: int) -> float:
    return T95.get(k, 1.96)


def _get(obj, path: str, default=None):
    cur = obj
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def decision_table_from_results() -> str:
    """Per-model decision metrics, recomputed from ``results/*.json``.

    Accuracy is not what separates models at 0.85 to 0.95. Two rows can share a mean and
    differ by three times in wall time, by half in tool-call validity, or by two turns to
    reach the same answer, and the results table above cannot show any of it. Every number
    here is derived from the published records, so it cannot drift from the table it sits
    beside.

    ``Task bundle`` is the comparability column: the hash of every prompt, task file and
    grader fixture the run measured. Two rows carrying the same hash were measured on the
    same instrument and may be compared. A row with no hash predates the field, and a row
    reading MIXED spans two bundles and is not a series.
    """
    rows: list[dict] = []
    for row in published_rows():
        model, rs = row["model"], row["records"]
        per_repeat: dict[int, list[float]] = defaultdict(list)
        for r in rs:
            per_repeat[r.get("repeat") or 1].append(float(r.get("score") or 0.0))
        means = [statistics.mean(v) for _k, v in sorted(per_repeat.items())]
        mean = statistics.mean(means) if means else 0.0
        half = (_t95(len(means)) * statistics.stdev(means) / (len(means) ** 0.5)
                if len(means) > 1 else None)

        dec = [r.get("decision") or {} for r in rs]

        def nums(path: str) -> list[float]:
            vals: list[float] = []
            for x in dec:
                v = _get(x, path)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    vals.append(float(v))
            return vals

        tps = nums("throughput.tokens_per_second")
        walls = sorted(nums("wall_seconds"))
        t2s = nums("turns_to_success")
        tv = nums("tool_validity_pct")
        head = nums("context.headroom_pct")
        vram = nums("vram.peak_server_mib")
        bundles = sorted({r["task_bundle_sha256"] for r in rs if r.get("task_bundle_sha256")})
        bundle_tasks = sorted({r["task_bundle_tasks"] for r in rs if r.get("task_bundle_tasks")})

        rows.append({
            "model": model,
            "label": row["label"],
            "repeats": len(per_repeat),
            "attempts": len(rs),
            "mean": mean,
            "half": half,
            "tps": statistics.mean(tps) if tps else None,
            "wall_p50": walls[len(walls) // 2] if walls else None,
            "t2s": statistics.mean(t2s) if t2s else None,
            "tv": statistics.mean(tv) if tv else None,
            "head": min(head) if head else None,
            "vram": max(vram) if vram else None,
            "bundles": bundles,
            "bundle_tasks": bundle_tasks[0] if len(bundle_tasks) == 1 else None,
            "tasks_seen": len({r["task"] for r in rs}),
        })

    # published_rows() already returns display order, configurations adjacent and in their
    # fixed order, so no second sort that would key on the model alone and separate them.

    def best(key: str, mode: str = "max"):
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            return None
        return (max if mode == "max" else min)(vals)

    top = {
        "tps": best("tps", "max"), "wall_p50": best("wall_p50", "min"),
        "t2s": best("t2s", "min"), "tv": best("tv", "max"),
        "head": best("head", "max"), "vram": best("vram", "min"),
    }

    def cell(val, key: str, text: str):
        if val is None:
            return "-"
        return f"**{text}**" if top[key] is not None and abs(val - top[key]) < 1e-9 else text

    out = [
        DECISION_START,
        "| Model | Repeats | Mean score (95% CI) | Throughput | Wall p50 "
        "| Turns to success | Tool calls valid | Context headroom | Peak VRAM | Task bundle |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        score = f"{r['mean']:.4f}" + (f" ± {r['half']:.4f}" if r["half"] is not None else "")
        if len(r["bundles"]) > 1:
            bundle = f"**MIXED ({len(r['bundles'])} bundles)**"
        elif r["bundles"]:
            n = r["bundle_tasks"] or r["tasks_seen"]
            bundle = f"{n} tasks, `{r['bundles'][0][:12]}`"
        else:
            bundle = "-"
        tps = cell(r["tps"], "tps", f"{r['tps']:.0f} tok/s") if r["tps"] else "-"
        wall = (cell(r["wall_p50"], "wall_p50", fmt_duration(r["wall_p50"]))
                if r["wall_p50"] else "-")
        t2s = cell(r["t2s"], "t2s", f"{r['t2s']:.1f}") if r["t2s"] else "-"
        tv = cell(r["tv"], "tv", f"{r['tv']:.0f}%") if r["tv"] else "-"
        head = cell(r["head"], "head", f"{r['head']:.0f}%") if r["head"] else "-"
        vram = cell(r["vram"], "vram", f"{r['vram']:,.0f} MiB") if r["vram"] else "-"
        out.append(f"| {r['label']} | {r['repeats']} | {score} | {tps} | {wall} "
                   f"| {t2s} | {tv} | {head} | {vram} | {bundle} |")
    out.append(DECISION_END)
    return "\n".join(out)


def rewrite_decision_table() -> bool:
    """Rewrite only the decision block; prose outside the markers is a person's."""
    text = README.read_text()
    if DECISION_START not in text or DECISION_END not in text:
        raise SystemExit(
            f"README is missing the {DECISION_START} / {DECISION_END} markers.\n"
            "Add them around the decision-metrics table before publishing."
        )
    block = decision_table_from_results()
    new = re.sub(re.escape(DECISION_START) + r".*?" + re.escape(DECISION_END), block,
                 text, count=1, flags=re.S)
    if new == text:
        return False
    README.write_text(new)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="*", help="run directories under runs/ to publish")
    ap.add_argument("--latest", action="store_true", help="newest run per model")
    ap.add_argument("--dry-run", action="store_true", help="skip commit and push")
    ap.add_argument("--message", help="commit message")
    a = ap.parse_args()

    if not a.run_ids and not a.latest:
        ap.error("give at least one run_id, or --latest")

    # 1. export -- a fault aborts the publish rather than entering the table as a zero
    export_cmd = [sys.executable, "tools/export_results.py"]
    export_cmd += ["--latest"] if a.latest else a.run_ids
    r = run(export_cmd)
    print(r.stdout.rstrip())
    if r.returncode != 0:
        print(r.stderr.rstrip(), file=sys.stderr)
        # Name the real cause. This used to print "the run has stall-class exclusions" for any
        # non-zero exit, including a bad argument and a run directory with no records, which
        # sent the reader looking for exclusions that did not exist.
        print("\nPUBLISH ABORTED: the export step failed (exit "
              f"{r.returncode}); see the message above. Nothing was committed.",
              file=sys.stderr)
        return 1

    # 2. charts -- every model in results/ must appear in every chart, or this aborts
    c = run([sys.executable, "tools/make_charts.py"])
    if c.returncode != 0:
        print(c.stdout, c.stderr, file=sys.stderr)
        print("\nPUBLISH ABORTED: chart generation failed.", file=sys.stderr)
        return 1
    if c.stdout.strip():
        print(c.stdout.rstrip())
    if c.stderr.strip():
        print(c.stderr.rstrip(), file=sys.stderr)

    # 2b. the local comparison chart, from the head-to-head artifacts rather than results/
    if (ROOT / "results" / "head-to-head").is_dir():
        lc = run([sys.executable, "tools/make_local_chart.py"])
        if lc.returncode != 0:
            print(lc.stdout, lc.stderr, file=sys.stderr)
            print("\nPUBLISH ABORTED: the local comparison chart failed.", file=sys.stderr)
            return 1
        if lc.stdout.strip():
            print(lc.stdout.rstrip())

    # 3. README table
    if rewrite_readme():
        print("README results table regenerated")
    else:
        print("README results table already current")

    # 3b. README decision metrics -- the columns that separate models at equal accuracy
    if rewrite_decision_table():
        print("README decision table regenerated")
    else:
        print("README decision table already current")

    if a.dry_run:
        print("\nDRY RUN: not committing. Files above are written in the working tree.")
        return 0

    # 4. commit only if there is something to commit
    # tools/ is included on purpose: a pipeline change must land with the run it
    # publishes, or the published repo cannot reproduce what produced the numbers.
    # tests/ belongs here for the same reason tools/ does: tests/test_publish.py is the file
    # that checks the README's own claims against the data, so a run whose table needed a prose
    # fix must ship the check that proves the fix. tasks/BUNDLES.md is the record of which
    # instrument each row measured; without it a second bundle hash is indistinguishable from a
    # defect. Both were left out of this path set once, and a commit that changed the claim and
    # its checker shipped only the claim.
    run(["git", "add", "-A", "results", "assets", "README.md", "tools", "tests",
         "tasks/BUNDLES.md"])
    staged = run(["git", "diff", "--cached", "--quiet"])
    if staged.returncode == 0:
        print("\nNothing to publish: results, charts and table are already committed.")
        return 0

    label = "results: " + (", ".join(a.run_ids) if a.run_ids else "newest run per model")
    msg = a.message or f"{label}\n\nPublished {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}."
    cm = run(["git", "commit", "-m", msg])
    print(cm.stdout.strip() or cm.stderr.strip())
    if cm.returncode != 0:
        return 2

    # 5. push, then prove the remote moved
    before = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    ps = run(["git", "push", "origin", "HEAD"])
    print((ps.stdout + ps.stderr).strip())
    if ps.returncode != 0:
        print("\nPUSH FAILED. The commit exists locally but the repository is unchanged.",
              file=sys.stderr)
        return 2

    remote = run(["git", "ls-remote", "origin", "HEAD"]).stdout.split()
    remote_sha = remote[0] if remote else ""
    if remote_sha != before:
        print(f"\nPUSH UNVERIFIED: local {before[:12]}, remote {remote_sha[:12] or 'unknown'}.",
              file=sys.stderr)
        return 2

    print(f"\nPublished and verified: {before[:12]} is on origin/HEAD")
    print(f"https://github.com/raidbeforebed/hermes-harness-bench/commit/{before[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
