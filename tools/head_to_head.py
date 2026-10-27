#!/usr/bin/env python3
"""Compare two runs on the tasks they share, one row per task and per attempt.

Why this exists rather than a comparison of two published rows: a published row is a balanced
design (every task attempted the same number of times), and adding targeted repeats on a
model's weak tasks breaks that balance. So a head-to-head is kept as its own artifact under
``results/head-to-head/``, which the publisher and the charts deliberately do not glob, and the
balanced table in the README stays balanced.

    python3 tools/head_to_head.py RUN_A RUN_B --out results/head-to-head/<name>.json
    python3 tools/head_to_head.py a b --tasks 01-json-extract 05-memory-nodestructive

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


def summarise(recs: list[dict]) -> dict[str, dict]:
    per: dict[str, list[dict]] = defaultdict(list)
    excluded: list[dict] = []
    for r in recs:
        if not r.get("task"):
            continue
        if scorable(r) and isinstance(r.get("score"), (int, float)):
            per[str(r["task"])].append(r)
        else:
            excluded.append(r)
    out = {}
    for task, rows in per.items():
        scores = [r["score"] for r in rows]
        walls = [r["wall_seconds"] for r in rows if isinstance(r.get("wall_seconds"), (int, float))]
        out[task] = {
            "attempts": len(rows),
            "scores": [round(s, 4) for s in sorted(scores)],
            "mean": round(statistics.mean(scores), 4),
            "perfect": sum(1 for s in scores if s >= 1.0),
            "mean_wall_s": round(statistics.mean(walls)) if walls else None,
            "faults": [{"repeat": r.get("repeat"), "class": (r.get("fault") or {}).get("class")}
                       for r in rows if r.get("harness_failed")],
        }
    return {"tasks": out, "excluded": [
        {"task": r.get("task"), "repeat": r.get("repeat"),
         "class": (r.get("fault") or {}).get("class"),
         "detail": ((r.get("fault") or {}).get("detail") or "")[:160]}
        for r in excluded]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--tasks", nargs="+", help="restrict to these task keys (default: the overlap)")
    ap.add_argument("--out", help="write the comparison JSON here")
    ap.add_argument("--title", help="human title for the comparison")
    a = ap.parse_args()

    model_a, recs_a = load(a.run_a)
    model_b, recs_b = load(a.run_b)
    sum_a, sum_b = summarise(recs_a), summarise(recs_b)
    shared = a.tasks or sorted(set(sum_a["tasks"]) & set(sum_b["tasks"]))
    if not shared:
        sys.exit("the two runs share no task with a scorable attempt")

    rows = []
    for t in shared:
        ra, rb = sum_a["tasks"].get(t), sum_b["tasks"].get(t)
        rows.append({
            "task": t,
            "a": ra, "b": rb,
            "delta": (round(rb["mean"] - ra["mean"], 4) if ra and rb else None),
        })

    def overall(side: str) -> dict:
        vals = [r[side]["mean"] for r in rows if r[side]]
        walls = [r[side]["mean_wall_s"] for r in rows if r[side] and r[side]["mean_wall_s"]]
        attempts = sum(r[side]["attempts"] for r in rows if r[side])
        perfect = sum(r[side]["perfect"] for r in rows if r[side])
        return {
            "mean_over_tasks": round(statistics.mean(vals), 4) if vals else None,
            "attempts": attempts,
            "perfect": perfect,
            "mean_wall_s": round(statistics.mean(walls)) if walls else None,
        }

    doc = {
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "title": a.title or f"{model_a} vs {model_b} on {len(shared)} shared tasks",
        "runs": {model_a: a.run_a, model_b: a.run_b},
        "tasks": rows,
        "overall": {model_a: overall("a"), model_b: overall("b")},
        "excluded": {model_a: sum_a["excluded"], model_b: sum_b["excluded"]},
        "note": ("Means are over attempts, per task, on the tasks both runs attempted. This is not "
                 "a published row: a head-to-head is deliberately unbalanced, because the extra "
                 "attempts target the tasks each model failed."),
    }

    print(f"{'task':30} {model_a[:18]:>19} {model_b[:18]:>19}  delta")
    for r in rows:
        fa = f"{r['a']['mean']:.3f} ({r['a']['attempts']})" if r["a"] else "-"
        fb = f"{r['b']['mean']:.3f} ({r['b']['attempts']})" if r["b"] else "-"
        d = f"{r['delta']:+.3f}" if r["delta"] is not None else "-"
        print(f"{r['task']:30} {fa:>19} {fb:>19}  {d}")
    oa, ob = doc["overall"][model_a], doc["overall"][model_b]
    print(f"{'MEAN OF TASK MEANS':30} {oa['mean_over_tasks']:>19} {ob['mean_over_tasks']:>19}")
    print(f"{'attempts':30} {oa['attempts']:>19} {ob['attempts']:>19}")
    print(f"{'perfect':30} {oa['perfect']:>19} {ob['perfect']:>19}")
    print(f"{'mean wall per attempt (s)':30} {oa['mean_wall_s']:>19} {ob['mean_wall_s']:>19}")
    for m, ex in doc["excluded"].items():
        if ex:
            print(f"excluded from {m}: {len(ex)} row(s) — " +
                  ", ".join(f"{e['task']} {e['class']}" for e in ex))

    if a.out:
        out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=2) + "\n")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
