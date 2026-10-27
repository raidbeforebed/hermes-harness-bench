#!/usr/bin/env python3
"""Compare two or more runs on the tasks they share, one row per task.

Why this exists rather than a comparison of published rows: a published row is a balanced design
(every task attempted the same number of times), and adding targeted repeats on a model's weak
tasks breaks that balance. So a head-to-head is kept as its own artifact under
``results/head-to-head/``, which the publisher and the charts deliberately do not glob, and the
balanced table in the README stays balanced.

    python3 tools/head_to_head.py RUN_A RUN_B [RUN_C ...] --out results/head-to-head/<name>.json

Each run is read from the bench tree (``HBENCH_RUNS`` or ``/root/hermes-bench/runs``), and only
rows whose score is the model's own artifact score are counted: a substitution and a stall are
listed separately, never averaged.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("HBENCH_ROOT") or Path(__file__).resolve().parent.parent)
RUNS = Path(os.environ.get("HBENCH_RUNS") or "/root/hermes-bench/runs")
if not RUNS.is_dir():
    RUNS = ROOT / "runs"


def scorable(rec: dict) -> bool:
    src = rec.get("score_source")
    if src:
        return src == "artifact"
    return not rec.get("harness_failed") and rec.get("exit_code") == 0


def load(run_id: str) -> tuple[str, list[dict]]:
    run_dir = RUNS / run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"no such run: {run_dir}")
    manifest = json.loads(manifest_path.read_text())
    model = (manifest.get("models") or ["(unknown)"])[0]
    recs = [json.loads(p.read_text()) for p in sorted(run_dir.glob("tasks/*/*/r*/result.json"))]
    if not recs:
        agg = run_dir / "results.json"
        if agg.exists():
            recs = json.loads(agg.read_text())
    return model, recs


def summarise(recs: list[dict]) -> dict:
    per: dict[str, list[dict]] = defaultdict(list)
    excluded: list[dict] = []
    for r in recs:
        if not r.get("task"):
            continue
        if scorable(r) and isinstance(r.get("score"), (int, float)):
            per[str(r["task"])].append(r)
        else:
            excluded.append(r)
    tasks = {}
    for task, rows in per.items():
        scores = [r["score"] for r in rows]
        walls = [r["wall_seconds"] for r in rows if isinstance(r.get("wall_seconds"), (int, float))]
        turns = [r["turns"] for r in rows if isinstance(r.get("turns"), int)]
        tasks[task] = {
            "attempts": len(rows),
            "scores": [round(s, 4) for s in sorted(scores)],
            "mean": round(statistics.mean(scores), 4),
            "perfect": sum(1 for s in scores if s >= 1.0),
            "mean_wall_s": round(statistics.mean(walls)) if walls else None,
            "mean_turns": round(statistics.mean(turns), 1) if turns else None,
            "faults": [{"repeat": r.get("repeat"), "class": (r.get("fault") or {}).get("class")}
                       for r in rows if r.get("harness_failed")],
        }
    return {"tasks": tasks, "excluded": [
        {"task": r.get("task"), "repeat": r.get("repeat"),
         "class": (r.get("fault") or {}).get("class"),
         "detail": ((r.get("fault") or {}).get("detail") or "")[:160]}
        for r in excluded]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run ids, two or more")
    ap.add_argument("--tasks", nargs="+", help="restrict to these task keys (default: the overlap)")
    ap.add_argument("--out", help="write the comparison JSON here")
    ap.add_argument("--title", help="human title for the comparison")
    a = ap.parse_args()
    if len(a.runs) < 2:
        sys.exit("give at least two run ids")

    loaded = {}
    for run_id in a.runs:
        model, recs = load(run_id)
        if model in loaded:
            sys.exit(f"two runs report the same model ({model}); compare different models")
        loaded[model] = {"run": run_id, **summarise(recs)}

    models = list(loaded)
    if a.tasks:
        tasks = list(a.tasks)
    else:
        shared = set(loaded[models[0]]["tasks"])
        for m in models[1:]:
            shared &= set(loaded[m]["tasks"])
        tasks = sorted(shared)
    if not tasks:
        sys.exit("the runs share no task with a scorable attempt")
    # a task must appear in every run, or the columns are not comparable rows
    missing = [(m, t) for m in models for t in tasks if t not in loaded[m]["tasks"]]
    if missing:
        print(f"warning: {len(missing)} model-task pair(s) have no scorable attempt and show as '-': "
              + ", ".join(f"{m}/{t}" for m, t in missing[:4]), file=sys.stderr)

    rows = []
    for t in tasks:
        cells = {m: loaded[m]["tasks"].get(t) for m in models}
        rows.append({"task": t, "cells": cells})

    overall = {}
    for m in models:
        cells = [r["cells"][m] for r in rows if r["cells"].get(m)]
        vals = [c["mean"] for c in cells]
        walls = [c["mean_wall_s"] for c in cells if c["mean_wall_s"]]
        turns = [c["mean_turns"] for c in cells if c["mean_turns"]]
        overall[m] = {
            "run": loaded[m]["run"],
            "mean_over_tasks": round(statistics.mean(vals), 4) if vals else None,
            "attempts": sum(c["attempts"] for c in cells),
            "perfect": sum(c["perfect"] for c in cells),
            "mean_wall_s": round(statistics.mean(walls)) if walls else None,
            "mean_turns": round(statistics.mean(turns), 1) if turns else None,
            "excluded": len(loaded[m]["excluded"]),
        }

    doc = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": a.title or f"{len(models)} local models on {len(tasks)} shared tasks",
        "runs": {m: loaded[m]["run"] for m in models},
        "tasks": rows,
        "overall": overall,
        "excluded": {m: loaded[m]["excluded"] for m in models},
        "note": ("Means are over attempts, per task, on the tasks every run attempted. This is not a "
                 "published row: a head-to-head is deliberately unbalanced, because the extra attempts "
                 "target the tasks each model failed."),
    }

    def cell_text(c: dict | None) -> str:
        return "-" if not c else f"{c['mean']:.3f} ({c['attempts']})"

    width = max(len(t) for t in tasks) + 2
    print(f"{'task':{width}}" + "".join(f"{m[:20]:>22}" for m in models))
    for r in rows:
        line = f"{r['task']:{width}}"
        for m in models:
            line += f"{cell_text(r['cells'][m]):>22}"
        print(line)
    print(f"{'MEAN OF TASK MEANS':{width}}" + "".join(f"{overall[m]['mean_over_tasks']:>22}" for m in models))
    print(f"{'attempts':{width}}" + "".join(f"{overall[m]['attempts']:>22}" for m in models))
    print(f"{'perfect':{width}}" + "".join(f"{overall[m]['perfect']:>22}" for m in models))
    print(f"{'mean wall per attempt (s)':{width}}" + "".join(f"{str(overall[m]['mean_wall_s']):>22}" for m in models))
    print(f"{'mean turns':{width}}" + "".join(f"{str(overall[m]['mean_turns']):>22}" for m in models))
    for m in models:
        if loaded[m]["excluded"]:
            print(f"excluded from {m}: " + ", ".join(f"{e['task']} {e['class']}" for e in loaded[m]["excluded"]))

    if a.out:
        out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
