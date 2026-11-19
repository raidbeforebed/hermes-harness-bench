#!/usr/bin/env python3
"""Read a set of published rows and say, in advance of any claim, whether the task set spreads.

The twelve-task suite saturated at eight of twelve tasks scoring 1.000 for both candidates, and
that was discovered after three repeats by computing an interval. It should have been stateable
before the runs: a suite whose weakest candidate nearly aces it cannot rank anything, and no
amount of repeats repairs a set that does not move.

This script is a readout against thresholds fixed in advance, not a discovery tool. The
thresholds and the metric definitions live in
`/mnt/agent_shared/hbench-robustness/SPEC.md`, section "Pre-registered spread test", written
2026-11-01 before this script produced its first reading. Change them there first, with a date,
or the verdict is being tuned to the answer.

    python3 bin/hbench_spread.py                  # every published row, grouped by task set
    python3 bin/hbench_spread.py <run-id> ...    # named runs from runs/, one row each
    python3 bin/hbench_spread.py --results-dir DIR

Metric definitions (SPEC.md, fixed 2026-11-01):

  * a row is one published result key (model plus configuration) or one run id;
  * a row's score for a task is the mean of its attempts on that task, harness failures
    excluded and counted;
  * a row's mean is the unweighted mean of its per-task scores, macro over tasks, never over
    attempts, because attempts inside a task are clustered;
  * a task is perfect in every row when every row scored exactly 1.000 on it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("HBENCH_ROOT", "/root/hermes-bench"))
DEFAULT_RESULTS = ROOT / "publish" / "tree" / "results"
DEFAULT_RUNS = ROOT / "runs"

# Pre-registered 2026-11-01, in SPEC.md before this script was run on any result.
# A saturated set: the weakest row still scores 27 of 30, that is 0.90 of any set size.
SATURATED_MIN_MEAN = 27 / 30
# A floor: the best row scores 5 or fewer of 30.
FLOOR_MAX_MEAN = 5 / 30


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def load_published(results_dir: Path) -> list[dict]:
    """Rows from the published record tree. One file per (row, task, repeat).

    The file name is `<row stem>__<task>__r<n>.json`, which is the only stable grouping a
    published record carries: older records were filed before `run_id` reached the record, so
    the stem is what says which tasks ran together.
    """
    rows: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        stem = path.stem
        parts = stem.rsplit("__", 2)
        if len(parts) != 3:
            continue  # not a per-attempt record (a retired note, a head-to-head page)
        row_key, task, _repeat = parts
        bucket = rows.setdefault(row_key, {"label": row_key, "model": record.get("model"),
                                           "records": []})
        bucket["records"].append({"task": task, **record})
    return list(rows.values())


def load_runs(run_ids: list[str], runs_dir: Path) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    missing: list[str] = []
    for run_id in run_ids:
        path = runs_dir / run_id / "results.json"
        if not path.is_file():
            missing.append(run_id)
            continue
        records = json.loads(path.read_text())
        models = {r.get("model") for r in records if r.get("model")}
        label = run_id if len(models) != 1 else f"{run_id} [{models.pop()}]"
        rows.append({"label": label, "records": records})
    return rows, missing


def summarise(row: dict) -> dict:
    """Per-task means for one row, harness failures excluded and counted.

    The set of tasks the row *ran* is kept separately from the tasks it has a score for: a
    task whose every attempt was cut by the harness ran, produced no model result, and must
    not shrink the row out of its task set.
    """
    by_task: dict[str, list[float]] = {}
    ran: set[str] = set()
    faults = 0
    for record in row["records"]:
        ran.add(record["task"])
        if record.get("harness_failed"):
            faults += 1
            continue
        by_task.setdefault(record["task"], []).append(float(record.get("score") or 0.0))
    scores = {task: mean(values) for task, values in by_task.items()}
    return {
        "label": row["label"],
        "scores": scores,
        "ran": ran,
        "mean": mean(list(scores.values())),
        "faults": faults,
        "attempts": len(row["records"]),
    }


def report_group(rows: list[dict]) -> str:
    """The readout for one task set: perfect count, spread, discriminating tasks, verdict."""
    rows = [r for r in rows if r["scores"]]
    if not rows:
        return "  (no scored attempts in this group)"
    ran = set().union(*(r["ran"] for r in rows))
    tasks = sorted(task for task in ran if all(task in r["scores"] for r in rows))
    unscored = sorted(ran - set(tasks))
    perfect = [task for task in tasks
               if all(abs(r["scores"].get(task, 0.0) - 1.0) < 1e-9 for r in rows)]
    discriminating = [task for task in tasks
                      if len({round(r["scores"].get(task, 0.0), 6) for r in rows}) > 1]

    ordered = sorted(rows, key=lambda r: r["mean"])
    low, high = ordered[0], ordered[-1]

    lines = [
        f"\ntask set: {len(tasks)} tasks ({tasks[0]} .. {tasks[-1]})",
        f"rows: {len(rows)}",
    ]
    for row in sorted(rows, key=lambda r: -r["mean"]):
        fault_note = f"  {row['faults']} fault(s) excluded" if row["faults"] else ""
        lines.append(f"  {row['mean']:.4f}  {row['label']}{fault_note}")
    lines.append(f"perfect in every row: {len(perfect)} of {len(tasks)}")
    if len(perfect) < len(tasks):
        lines.append("  " + ", ".join(perfect))
    if unscored:
        lines.append(f"not scored in every row (harness faults, no model result): "
                     f"{len(unscored)} task(s): {', '.join(unscored)}")
    lines.append(f"spread across rows: min mean {low['mean']:.4f} ({low['label']}), "
                 f"max mean {high['mean']:.4f} ({high['label']})")
    lines.append(f"tasks that discriminate (at least two rows differ): {len(discriminating)}")
    for task in discriminating:
        seen = sorted({r["scores"].get(task, 0.0) for r in rows})
        lines.append(f"  {task:34} {' .. '.join(f'{value:.3f}' for value in seen)}")

    if len(rows) < 2:
        verdict = "ONE ROW: nothing to compare"
    elif low["mean"] >= SATURATED_MIN_MEAN:
        verdict = ("SATURATED, cannot rank: the weakest row scores "
                   f"{low['mean']:.4f}, at or above the pre-registered {SATURATED_MIN_MEAN:.2f}")
    elif high["mean"] <= FLOOR_MAX_MEAN:
        verdict = ("FLOOR: the best row scores "
                   f"{high['mean']:.4f}, at or below the pre-registered {FLOOR_MAX_MEAN:.4f}")
    else:
        verdict = (f"SPREADS: min mean {low['mean']:.4f} is below {SATURATED_MIN_MEAN:.2f} and "
                   f"max mean {high['mean']:.4f} is above {FLOOR_MAX_MEAN:.4f}")
    lines.append(f"VERDICT: {verdict}")

    if discriminating:
        n = len(discriminating)
        lines.append(f"task-level separation: {n} discordant task(s), smallest attainable "
                     f"two-sided permutation p = 2/2^{n} = {2 / 2 ** n:.4f}")
        if n < 5:
            lines.append("  at this many discordant tasks no ranking claim can reach "
                         "significance by construction, which is the structural fact in SPEC.md")
    else:
        lines.append("task-level separation: no task differs between the rows, so there is "
                     "nothing to test")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pre-registered readout: does this task set spread enough to rank?")
    parser.add_argument("run_ids", nargs="*",
                        help="run ids from runs/ (each is one row). Omit to read the published "
                             "record tree and group it by task set.")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS),
                        help=f"published per-attempt records (default {DEFAULT_RESULTS})")
    parser.add_argument("--runs-dir", default=str(DEFAULT_RUNS),
                        help=f"where run ids live (default {DEFAULT_RUNS})")
    args = parser.parse_args()

    if args.run_ids:
        rows, missing = load_runs(args.run_ids, Path(args.runs_dir))
        source = f"{len(rows)} named run(s) from {args.runs_dir}"
        if missing:
            print(f"unknown run id(s): {', '.join(missing)}", file=sys.stderr)
            if not rows:
                return 2
    else:
        results_dir = Path(args.results_dir)
        if not results_dir.is_dir():
            print(f"no published records at {results_dir}", file=sys.stderr)
            return 2
        rows = load_published(results_dir)
        source = f"{len(rows)} published row(s) from {results_dir}"

    print(f"hbench spread test: {source}")
    print("thresholds fixed 2026-11-01 in /mnt/agent_shared/hbench-robustness/SPEC.md: "
          f"saturated at min mean >= {SATURATED_MIN_MEAN:.2f}, floor at max mean <= "
          f"{FLOOR_MAX_MEAN:.4f}")

    groups: dict[frozenset[str], list[dict]] = {}
    for row in rows:
        summary = summarise(row)
        if not summary["ran"]:
            continue
        groups.setdefault(frozenset(summary["ran"]), []).append(summary)

    if not groups:
        print("no scored rows to read")
        return 2
    for task_set in sorted(groups, key=lambda ts: (len(ts), sorted(ts)[0] if ts else "")):
        print(report_group(groups[task_set]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
