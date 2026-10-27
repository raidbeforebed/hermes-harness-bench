#!/usr/bin/env python3
"""Export finished hbench runs into the published ``results/`` format.

``results/`` holds one file per model per task per repeat, and every chart and
table in the README is derived from it. ``runs/`` is gitignored, so a run that is
not exported is not published.

    python3 tools/export_results.py <run_id>        # one run
    python3 tools/export_results.py --latest        # newest run per model
    python3 tools/export_results.py --all           # every run under runs/

Two classes of non-zero exit exist and they are filed differently.

A record that carries the model's own artifact score (``score_source: artifact``) is
published, with its exit code and fault class beside the score. A run that hit the
agent's own turn ceiling after doing the work is a finding about the model, and
zeroing it reported a harness ceiling as a capability.

A record with no scorable artifact (``score_source: none``: a provider substitution, an
argv refusal, an untouched workspace) is written to ``results/excluded/`` with the
reason, never to ``results/``, and the export exits non-zero when the class is a stall
so the publisher refuses the run. Excluded is not dropped: the marker names the task,
the repeat, the fault class, the exit code and the run it came from, so a reader of the
published tree can see the hole and why it is there.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(os.environ.get("HBENCH_ROOT") or Path(__file__).resolve().parent.parent)
# The publish repo does not contain runs/ (it is gitignored), so fall back to the
# working tree's runs/ when this repo has none. Without this the documented
# command fails from the publish checkout, which is the only place it is run.
RUNS = Path(os.environ["HBENCH_RUNS"]) if os.environ.get("HBENCH_RUNS") else ROOT / "runs"
if not RUNS.is_dir():
    RUNS = Path("/root/hermes-bench/runs")
RESULTS = ROOT / "results"

# Keys the published schema carries. `passed` and `total_checks` are derived from
# the checks list so a reader does not have to count booleans.
FIELDS = ("checks", "exit_code", "finished", "model", "repeat", "score", "started",
          "task", "tokens_in", "tokens_out", "tool_calls", "turns", "wall_seconds",
          # The fault travels with the score, not instead of it: a reader must be able to see
          # that a row ended at a ceiling and which one.
          "artifact_score", "fault", "harness_failed", "score_source",
          "source_run", "source_repeat")
EXCLUDED = RESULTS / "excluded"
STALL_CLASSES = {"no_work", "cli_error", ""}


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


def scorable(record: dict) -> bool:
    """Is this record's score the model's own artifact score?

    ``score_source`` is written by the harness since 2026-10-27. Records from before it have
    no such field, and the only shape they excluded was a harness fault.
    """
    src = record.get("score_source")
    if src:
        return src == "artifact"
    return not record.get("harness_failed") and record.get("exit_code") == 0


def export_run(run_dir: Path) -> tuple[int, int, list[dict]]:
    """Write one run's records. Returns (written, excluded_count, excluded_markers)."""
    written = skipped = 0
    names: list[dict] = []
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
            if not scorable(r):
                # No artifact of this model's work to score: filed as an exclusion with its
                # reason instead of published as a zero.
                skipped += 1
                fault = r.get("fault") or {}
                marker = {
                    "model": model, "task": task, "repeat": r.get("repeat"),
                    "attempt": r.get("source_repeat"),
                    "fault_class": fault.get("class") or "unclassified",
                    "exit_code": r.get("exit_code"),
                    "detail": (fault.get("detail") or "")[:400],
                    "artifact_score": r.get("artifact_score"),
                    "run_id": r.get("run_id"), "source_run": r.get("source_run"),
                    "source_repeat": r.get("source_repeat"),
                    "workdir": r.get("workdir"),
                }
                EXCLUDED.mkdir(parents=True, exist_ok=True)
                stem = (f"{slug(model)}__{task}__{r.get('source_run')}__r{marker['attempt']}"
                        if r.get("source_run") else f"{slug(model)}__{task}__r{marker['repeat']}")
                path_out = EXCLUDED / f"{stem}.json"
                path_out.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
                names.append(marker)
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
    all_skipped: list[dict] = []
    for run_dir in targets:
        w, s, names = export_run(run_dir)
        total_written += w
        total_skipped += s
        all_skipped.extend(names)
        print(f"  {run_dir.name}: {w} written, {s} excluded")

    print(f"\n{total_written} record(s) in {RESULTS}")
    if total_skipped:
        print(f"\nEXCLUDED {total_skipped} record(s), filed under {EXCLUDED}:")
        for m in all_skipped:
            print(f"  {m['model']} {m['task']} attempt {m['attempt']} [{m['fault_class']} "
                  f"exit {m['exit_code']}]: {str(m['detail'])[:110]}")
        stalls = [m for m in all_skipped if m["fault_class"] in STALL_CLASSES]
        if stalls:
            print("\nThese are stalls: the harness never ran the task. Re-run them before "
                  "publishing this model.")
            sys.exit(1)
        print("\nNo stall classes among them: these are routing/ceiling faults, recorded as "
              "exclusions. Publishing continues.")


if __name__ == "__main__":
    main()
