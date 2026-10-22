#!/usr/bin/env python3
"""Export finished hbench runs into the published ``results/`` format.

``results/`` holds one file per model per task per repeat, and every chart and
table in the README is derived from it. ``runs/`` is gitignored, so a run that is
not exported is not published.

    python3 tools/export_results.py <run_id>        # one run
    python3 tools/export_results.py --latest        # newest run per model
    python3 tools/export_results.py --all           # every run under runs/

A record whose child exited non-zero is NOT a model result: the CLI never got the
command (an argv bug, a fenced install, a bootstrap failure). Those are counted and
reported loudly, never written, because publishing them as 0.00 is the exact error
that made a broken harness report a fake 100% in the other direction.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("HBENCH_ROOT") or Path(__file__).resolve().parent.parent)
RUNS = ROOT / "runs"
RESULTS = ROOT / "results"

# Keys the published schema carries. `passed` and `total_checks` are derived from
# the checks list so a reader does not have to count booleans.
FIELDS = ("checks", "exit_code", "finished", "model", "repeat", "score", "started",
          "task", "tokens_in", "tokens_out", "tool_calls", "turns", "wall_seconds")


def slug(model: str) -> str:
    """Filesystem-safe model name: ``gemma4:cloud`` -> ``gemma4-cloud``."""
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def record_files(run_dir: Path) -> list[Path]:
    """Every per-task record in a run, from either the per-task or aggregate layout."""
    out = sorted(run_dir.glob("tasks/*/*/*/result.json"))
    if out:
        return out
    agg = run_dir / "results.json"
    if agg.exists():
        return [agg]
    return []


def export_run(run_dir: Path) -> tuple[int, int, list[str]]:
    """Write one run's records. Returns (written, skipped_harness, skipped_names)."""
    written = skipped = 0
    names: list[str] = []
    for path in record_files(run_dir):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            print(f"  unreadable {path}: {exc}", file=sys.stderr)
            continue
        records = data if isinstance(data, list) else [data]
        for r in records:
            task, model = r.get("task"), r.get("model")
            if not task or not model:
                continue
            # A harness failure is not a model result.
            if r.get("exit_code") != 0 or r.get("harness_failed"):
                skipped += 1
                names.append(f"{slug(model)}__{task}__r{r.get('repeat', 1)}")
                continue
            checks = r.get("checks") or []
            pub = {k: r.get(k) for k in FIELDS if k in r}
            pub["model"] = model
            pub["task"] = task
            pub["repeat"] = r.get("repeat", 1)
            pub["checks"] = checks
            pub["passed"] = sum(1 for c in checks if c.get("pass"))
            pub["total_checks"] = len(checks)
            out = RESULTS / f"{slug(model)}__{task}__r{pub['repeat']}.json"
            out.write_text(json.dumps(pub, indent=2, sort_keys=True) + "\n")
            written += 1
    return written, skipped, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id", nargs="?", help="run directory name under runs/")
    ap.add_argument("--all", action="store_true", help="export every run")
    ap.add_argument("--latest", action="store_true", help="newest run per model")
    a = ap.parse_args()

    if not a.run_id and not a.all and not a.latest:
        ap.error("give a run_id, or --all, or --latest")

    RESULTS.mkdir(exist_ok=True)

    if a.run_id:
        run_dir = RUNS / a.run_id
        if not run_dir.is_dir():
            sys.exit(f"no such run: {run_dir}")
        targets = [run_dir]
    elif a.all:
        targets = sorted(d for d in RUNS.iterdir() if d.is_dir())
    else:
        newest: dict[str, tuple[float, Path]] = {}
        for d in RUNS.iterdir():
            if not d.is_dir():
                continue
            files = record_files(d)
            if not files:
                continue
            try:
                model = json.loads(files[0].read_text())["model"]
            except (OSError, ValueError, KeyError):
                model = json.loads((d / "results.json").read_text())[0]["model"]
            mtime = max(f.stat().st_mtime for f in files)
            if model not in newest or mtime > newest[model][0]:
                newest[model] = (mtime, d)
        targets = [d for _m, d in newest.values()]

    total_written = total_skipped = 0
    all_skipped: list[str] = []
    for run_dir in targets:
        w, s, names = export_run(run_dir)
        total_written += w
        total_skipped += s
        all_skipped.extend(names)
        print(f"  {run_dir.name}: {w} written, {s} harness failure(s) skipped")

    print(f"\n{total_written} record(s) in {RESULTS}")
    if total_skipped:
        print(f"\nEXCLUDED {total_skipped} record(s): the child exited non-zero, so the")
        print("harness never ran the task. Re-run those before publishing this model:")
        for n in all_skipped:
            print(f"  {n}")
        sys.exit(1)


if __name__ == "__main__":
    main()
