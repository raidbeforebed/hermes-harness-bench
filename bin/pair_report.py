#!/usr/bin/env python3
"""Compare two hbench runs on the frozen suite, task by task and metric by metric.

    python3 bin/pair_report.py <run-a> <run-b> [--markdown]

Both runs must carry the same task_bundle hash; if they do not, the pair is not
comparable and this script says so instead of printing a table. Output is a plain
text report by default and a markdown fragment with --markdown, so the numbers in
the published README come from this script rather than from typing.
"""

from __future__ import annotations

import json
import math
import statistics as st
import sys
from pathlib import Path

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
       8: 2.306, 9: 2.262, 10: 2.228}


def load(run_id: str, root: Path = Path("/root/hermes-bench/runs")) -> dict:
    d = root / run_id
    if not d.is_dir():
        raise SystemExit(f"no such run: {d}")
    man = json.loads((d / "manifest.json").read_text())
    res = json.loads((d / "results.json").read_text())
    records = res if isinstance(res, list) else (res.get("attempts") or res.get("records") or [])
    if isinstance(records, dict):
        records = list(records.values())
    bundle = ((man.get("reproducibility") or {}).get("task_bundle") or {})
    models = man.get("models") or []
    model = models[0] if models else (man.get("model") or man.get("model_id") or "")
    return {"id": run_id, "manifest": man, "records": records, "bundle": bundle,
            "model": Path(str(model)).name.replace(".gguf", "") or run_id}


def mean_ci(values: list[float]) -> tuple[float, float, float, int]:
    """(mean, half-width, sd, n) using a two-sided 95 percent Student-t interval."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 0.0, 1
    sd = st.stdev(values)
    t = T95.get(n - 1, 1.96 + 2.4 / max(n, 1))
    return mean, t * sd / math.sqrt(n), sd, n


def passed(record: dict) -> bool:
    """A perfect score and no harness fault: the same rule the leaderboard uses."""
    if record.get("harness_failed"):
        return False
    try:
        return float(record.get("score") or 0.0) >= 0.999
    except (TypeError, ValueError):
        return False


def per_repeat_means(records: list[dict]) -> list[float]:
    by_rep: dict[int, list[float]] = {}
    for r in records:
        by_rep.setdefault(r.get("repeat") or 1, []).append(float(r.get("score") or 0.0))
    return [sum(v) / len(v) for _k, v in sorted(by_rep.items())]


def main(argv: list[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    md = "--markdown" in argv
    if len(args) != 2:
        raise SystemExit(__doc__)
    a, b = load(args[0]), load(args[1])

    if a["bundle"] != b["bundle"]:
        print("NOT COMPARABLE: the two runs measured different task bundles.")
        for name, run in (("a", a), ("b", b)):
            names = sorted(run["bundle"])
            print(f"  {name}: {run['id']}  {len(names)} tasks")
        return 1

    tasks = sorted({r["task"] for r in a["records"]} | {r["task"] for r in b["records"]})
    print(f"run a: {a['id']}  ({len(a['records'])} records, model {a['model']})")
    print(f"run b: {b['id']}  ({len(b['records'])} records, model {b['model']})")
    print(f"bundle: {len(a['bundle'])} tasks, identical in both runs")
    print()

    scores = {}
    for run in (a, b):
        recs = run["records"]
        reps = per_repeat_means(recs)
        mean, half, sd, n = mean_ci(reps)
        scores[run["id"]] = (mean, half, n)
        passes = sum(1 for r in recs if passed(r))
        print(f"{run['model'] or run['id']}: mean {mean:.4f} +/- {half:.4f} over {n} repeats, "
              f"{passes}/{len(recs)} attempts passed")
    print()

    # Per-task: passes per attempt, and where the two models part company.
    rows = []
    for t in tasks:
        cells = []
        for run in (a, b):
            sel = [r for r in run["records"] if r.get("task") == t]
            cells.append((sum(1 for r in sel if passed(r)), len(sel)))
        rows.append((t, cells[0], cells[1]))

    def frac(c):
        return f"{c[0]}/{c[1]}" if c[1] else "-"

    diff = [r for r in rows if r[1] != r[2]]
    print(f"tasks: {len(rows)}  identical: {len(rows) - len(diff)}  differing: {len(diff)}")
    for t, ca, cb in rows:
        flag = "  <-- differs" if ca != cb else ""
        print(f"  {t:32} {frac(ca):>7} {frac(cb):>7}{flag}")

    # Metrics, taken as the median over attempts so one slow attempt cannot steer the number.
    print()
    print("decision metrics (median over attempts)")
    keys = [("throughput", "tokens_per_second", "tok/s"),
            ("wall", "wall_seconds", "s"),
            ("turns", "turns_to_success", "turns"),
            ("tool_validity", "tool_validity_pct", "%"),
            ("headroom", "headroom_pct", "%"),
            ("vram", "peak_server_mib", "MiB")]
    for label, key, unit in keys:
        line = []
        for run in (a, b):
            vals = []
            for r in run["records"]:
                dec = r.get("decision") or {}
                for group in dec.values():
                    if isinstance(group, dict) and isinstance(group.get(key), (int, float)):
                        vals.append(float(group[key]))
            line.append(f"{st.median(vals):.1f}{unit}" if vals else "unmeasured")
        print(f"  {label:14} {line[0]:>16} {line[1]:>16}")

    if md:
        print()
        print("| task | " + " | ".join(
            [a["model"] or a["id"], b["model"] or b["id"]]) + " |")
        print("|---|---|---|")
        for t, ca, cb in rows:
            print(f"| `{t}` | {frac(ca)} | {frac(cb)} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
