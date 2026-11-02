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
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("HBENCH_ROOT") or Path(__file__).resolve().parent.parent)
# Where the measurements live. The publish repo does not contain runs/ (it is gitignored), and
# the runs/ directory it does contain is not the working tree: the publish gate creates
# runs/<run_id>/ there to hold manifest.json and published.json, and nothing else. So a run
# directory can exist under this checkout and hold no records at all, which is how an export
# once reported "0 written, 0 excluded" and exited clean.
#
# The rule: a run directory counts as a source only when it holds record files. Candidates are
# tried in order, and the first root that holds at least one record-bearing run is used.
def _has_records(d: Path) -> bool:
    """True when a run directory holds something to export.

    Aggregate layout is one results.json; per-task layout is tasks/*/*/*/result.json. This is
    the same pair record_files() reads, kept here so RUNS can be chosen before that runs.
    """
    return (d / "results.json").is_file() or any(d.glob("tasks/*/*/*/result.json"))


def _runs_roots() -> list[Path]:
    env = os.environ.get("HBENCH_RUNS")
    if env:
        return [Path(env)]
    return [ROOT / "runs", Path("/root/hermes-bench/runs")]


_RUNS_ROOTS = [p for p in _runs_roots() if p.is_dir()]
RUNS = next((p for p in _RUNS_ROOTS
             if any(c.is_dir() and _has_records(c) for c in p.iterdir())),
            _RUNS_ROOTS[0] if _RUNS_ROOTS else ROOT / "runs")
RESULTS = ROOT / "results"


def resolve_run(name: str) -> Path | None:
    """The run directory for ``name`` that actually holds records, or None.

    Prefers a root with records; falls back to a bare directory only so the caller can report
    "this run has no records" by name instead of exporting nothing and returning success.
    """
    best: Path | None = None
    for root in _RUNS_ROOTS:
        d = root / name
        if not d.is_dir():
            continue
        if _has_records(d):
            return d
        best = best or d
    return best

sys.path.insert(0, str(Path(__file__).resolve().parent))
import result_key  # noqa: E402


# Keys the published schema carries. `passed` and `total_checks` are derived from
# the checks list so a reader does not have to count booleans.
FIELDS = ("checks", "exit_code", "finished", "model", "repeat", "score", "started",
          "task", "tokens_in", "tokens_out", "tool_calls", "turns", "wall_seconds",
          # The fault travels with the score, not instead of it: a reader must be able to see
          # that a row ended at a ceiling and which one.
          "artifact_score", "fault", "harness_failed", "score_source",
          "source_run", "source_repeat",
          # Comparability, not decoration. A published record that cannot name its run and
          # the task bundle it measured cannot be paired with another model's row: the reader
          # has no way to tell a same-instrument comparison from a cross-suite accident.
          "run_id", "decision", "instrument_fingerprint", "manifest_sha256",
          # The configuration the model was measured under. A run from before 2026-11-01
          # carries none, and that is published as all-unrecorded rather than inferred from
          # its manifest: the record is the unit of publication. See tools/result_key.py.
          "config")
EXCLUDED = RESULTS / "excluded"
STALL_CLASSES = {"no_work", "cli_error", ""}


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def run_bundle(run_dir: Path) -> dict:
    """The task-bundle hash this run measured, taken from its own manifest.

    ``reproducibility.task_bundle`` maps task -> file -> sha256. Hashing that block into one
    string gives the row a single value a reader can compare across models. Two runs that
    hash the same bundle measured the same prompts and the same graders; two that do not
    cannot be presented as a pair, whatever their scores say.
    """
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text())
    except (OSError, ValueError):
        return {}
    bundle = ((manifest.get("reproducibility") or {}).get("task_bundle") or {})
    if not bundle:
        return {}
    return {
        "task_bundle_sha256": hashlib.sha256(canonical_json(bundle).encode()).hexdigest(),
        "task_bundle_tasks": len(bundle),
    }


def slug(model: str) -> str:
    """Filesystem-safe model name: ``gemma4:cloud`` -> ``gemma4-cloud``.

    The rule lives in tools/result_key.py, shared with the publish gate, so one model
    cannot occupy two file sets under two spellings.
    """
    return result_key.model_slug(model)


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
    bundle = run_bundle(run_dir)
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
                # The configuration token travels here too, for the same reason it travels in
                # results/: two configurations of one model are two instruments, and their
                # exclusions must not overwrite each other.
                tag = result_key.token(result_key.config_of(r))
                mid = f"__{tag}" if tag else ""
                stem = (f"{slug(model)}{mid}__{task}__{r.get('source_run')}__r{marker['attempt']}"
                        if r.get("source_run")
                        else f"{slug(model)}{mid}__{task}__r{marker['repeat']}")
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
            # Always present, and always a full six-field block: a record from before the
            # field existed exports as all-unrecorded, never as a missing key that a surface
            # might read as "no configuration dimension applies here".
            pub["config"] = result_key.config_of(r)
            pub["passed"] = sum(1 for c in checks if c.get("pass"))
            pub["total_checks"] = len(checks)
            for k, v in bundle.items():
                pub.setdefault(k, v)
            out = RESULTS / result_key.published_name(pub)
            out.write_text(json.dumps(pub, indent=2, sort_keys=True) + "\n")
            written += 1
    return written, skipped, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="*", metavar="run_id",
                    help="run directory name(s) under runs/")
    ap.add_argument("--all", action="store_true", help="export every run")
    ap.add_argument("--latest", action="store_true", help="newest run per model")
    a = ap.parse_args()

    if not a.run_ids and not a.all and not a.latest:
        ap.error("give at least one run_id, or --all, or --latest")

    RESULTS.mkdir(exist_ok=True)

    if a.run_ids:
        targets = []
        for name in a.run_ids:
            run_dir = resolve_run(name)
            if run_dir is None:
                sys.exit(f"no such run: {name} under any of "
                         + ", ".join(str(p) for p in _RUNS_ROOTS))
            if not record_files(run_dir):
                # A directory with a manifest and no records is a publish marker, not a
                # measurement. Exporting it silently wrote nothing and returned success, which
                # a caller reads as "done".
                sys.exit(f"{name} holds no record files at {run_dir}; "
                         "it is a publish marker, not a run")
            targets.append(run_dir)
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
                data = json.loads(files[0].read_text())
                # A run's first record file is either one per-task record (a dict) or the
                # aggregate results.json (a list). --latest crashed on the list shape with a
                # TypeError that the except clause below did not cover.
                model = (data[0] if isinstance(data, list) else data)["model"]
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                model = json.loads((d / "results.json").read_text())[0]["model"]
            mtime = max(f.stat().st_mtime for f in files)
            if model not in newest or mtime > newest[model][0]:
                newest[model] = (mtime, d)
        targets = [d for _m, d in newest.values()]

    total_written = total_skipped = 0
    all_skipped: list[dict] = []
    for run_dir in targets:
        marker = run_dir / "RETIRED.md"
        if marker.exists():
            lines = [ln.strip() for ln in marker.read_text().splitlines() if ln.strip()]
            reason = next((ln.lstrip("# ").strip() for ln in lines if not ln.startswith("#")), "")
            print(f"  {run_dir.name}: RETIRED, not exported. {reason}")
            continue
        # An abandoned run is not a measurement that finished. Its records are an aborted
        # prefix (a provider outage on task 01, say), and exporting them would file zeros for a
        # series nobody ran to the end. The collector refuses these; so does the exporter.
        marker = run_dir / "ABANDONED.md"
        if marker.exists():
            lines = [ln.strip() for ln in marker.read_text().splitlines() if ln.strip()]
            reason = next((ln.lstrip("# ").strip() for ln in lines if not ln.startswith("#")), "")
            print(f"  {run_dir.name}: ABANDONED, not exported. {reason}")
            continue
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
