"""Binary quality gates (hbench robustness, SPEC workstream 5).

/mnt/agent_shared/hbench-robustness/SPEC.md, workstream 5: a graded rubric scored by
code needs a bespoke static analyzer per task, which is unbounded maintenance. The
lighter mechanism kept here is a small set of EXTRA binary gates attached to a task.
Each gate is a deterministic, code-checkable pass/fail condition. Nothing in a gate
needs a human or an LLM to judge it: the same workspace and the same attempt record
always produce the same gate result.

A task opts in by listing gates in its ``task.json`` under ``quality_gates``. The runner
(``hbench.py``) evaluates them in order and STOPS AT THE FIRST FAILURE, the way a CI
pipeline blocks on its first failing step: the first failing gate is the binding
constraint and its detail says why. The gates are reported SEPARATELY from the pass
rate (the judge score) and folded into one secondary binary number, ``quality_score``
(1.0 when every gate passes, 0.0 otherwise, null when the task declares no gates).

Design rules that keep this honest:

- Every gate is a pure function of ``(spec, work, record, baseline)``. No clock, no
  randomness, no network, no LLM. Same inputs, same verdict.
- A gate reads the FINAL workspace state (the files the agent left) and, for the two
  behavioural gates, the attempt ``record`` (its stdout and nothing more exotic).
- The behavioural gates fail closed: with no usable stdout they fail rather than pass,
  so a gate can never award a point it did not actually check.
- ``baseline`` is the pristine task fixture (``tasks/<t>/workspace``). The two gates
  that compare against it (lines_changed_bound, no_new_dependencies) never mutate it;
  ``stage_workspace`` copies it into the run dir, so the fixture stays pristine.

Gate registry
-------------
- json_schema_strict     : the output file is a JSON object whose key set is exactly the
                           declared allowed keys (required present, no extras).
- no_trailing_blank_line : the output file does not end with a blank line.
- no_markdown_fences     : the attempt's stdout contains no markdown code fence.
- lines_changed_bound    : the edited file differs from the baseline by at most N lines.
- no_new_dependencies    : the edited file imports no module that the baseline did not
                           already import, outside the standard library.
- no_old_names           : none of the declared old names exist in the workspace.
- no_unrelated_files     : the workspace holds exactly the declared set of files.
"""

from __future__ import annotations

import ast
import difflib
import json
import sys
from pathlib import Path
from typing import Callable

# Reused from the uncertainty module so the quality section applies the exact same
# "this row is a real artifact score, not a harness fault" filter the macro score uses.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hbench_uncertainty as uncertlib  # noqa: E402

RUNS_DIR = uncertlib.RUNS_DIR

# Directories that an agent or a test runner drops into the workspace and that have
# nothing to do with the task. Never counted as "files the agent touched".
EXCLUDE_DIRS = {".git", ".hg", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}

# Standard-library top-level module names (Python 3.10+). Used to tell "imported a
# third-party package" apart from "imported the stdlib". Falls back to an empty set on
# the rare interpreter that lacks it, which makes no_new_dependencies conservative.
STDLIB_MODULES = getattr(sys, "stdlib_module_names", frozenset())


def _pass(name: str, detail: str) -> dict:
    return {"gate": name, "pass": True, "detail": detail}


def _fail(name: str, detail: str) -> dict:
    return {"gate": name, "pass": False, "detail": detail}


# --------------------------------------------------------------------------
# Individual gates. Each returns one check dict: {"gate", "pass", "detail"}.
# --------------------------------------------------------------------------

def gate_json_schema_strict(spec: dict, work: Path, record: dict, baseline: Path | None) -> dict:
    """Output validates against a strict schema with no extra keys."""
    out = spec.get("output", "report.json")
    allowed = spec.get("allowed_keys")
    if not allowed:
        return _fail("json_schema_strict", "gate spec missing allowed_keys")
    allowed_set = set(allowed)
    p = work / out
    if not p.exists():
        return _fail("json_schema_strict", f"{out} missing")
    try:
        data = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001 - the detail is the point
        return _fail("json_schema_strict", f"{out} is not valid JSON: {exc}")
    if not isinstance(data, dict):
        return _fail("json_schema_strict", f"{out} is not a JSON object")
    keys = set(data.keys())
    missing = allowed_set - keys
    extra = keys - allowed_set
    if missing or extra:
        return _fail("json_schema_strict",
                     f"keys={sorted(keys)}; missing={sorted(missing)}; extra={sorted(extra)}")
    return _pass("json_schema_strict", f"key set == {sorted(allowed_set)}")


def gate_no_trailing_blank_line(spec: dict, work: Path, record: dict, baseline: Path | None) -> dict:
    """The output file does not end with a trailing blank line.

    A single terminating newline is fine (``abc\\n``); an empty final line is not
    (``abc\\n\\n``). Both a no-newline and a single-newline ending pass, because the
    reference solvers use the first and the prompt only forbids a trailing blank line.
    """
    out = spec.get("output", "format_check.txt")
    p = work / out
    if not p.exists():
        return _fail("no_trailing_blank_line", f"{out} missing")
    text = p.read_text()
    if text == "":
        return _fail("no_trailing_blank_line", f"{out} is empty")
    body = text[:-1] if text.endswith("\n") else text
    if body.endswith("\n"):
        return _fail("no_trailing_blank_line", f"{out} ends with a blank line")
    return _pass("no_trailing_blank_line", f"{out} has no trailing blank line")


def gate_no_markdown_fences(spec: dict, work: Path, record: dict, baseline: Path | None) -> dict:
    """No markdown code fence in the attempt's stdout.

    Behavioural gate: it reads only ``record["stdout_raw"]``. Fails closed when there is
    no stdout, so it never awards a point it did not actually check.
    """
    stdout = (record or {}).get("stdout_raw")
    if stdout is None:
        return _fail("no_markdown_fences", "no stdout in record")
    if "```" in stdout:
        return _fail("no_markdown_fences", "stdout contains a markdown code fence")
    return _pass("no_markdown_fences", "no markdown fences in stdout")


def _module_imports(path: Path) -> set[str]:
    """Top-level module names imported by a Python file, via AST (no execution)."""
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:  # absolute import, not a relative .
                mods.add(node.module.split(".")[0])
    return mods


def gate_lines_changed_bound(spec: dict, work: Path, record: dict, baseline: Path | None) -> dict:
    """The edited file differs from the baseline by at most ``max_lines`` changed lines.

    Keeps a fix small so that deleting the test (already blocked by the judge's own
    test-file-untouched check) or dumping an unrelated file cannot pass. Changed lines
    are added plus removed lines from a unified diff, excluding the hunk headers.
    """
    fname = spec.get("file")
    max_lines = spec.get("max_lines")
    if not fname or max_lines is None:
        return _fail("lines_changed_bound", "gate spec needs file and max_lines")
    edited = work / fname
    base = (baseline / fname) if baseline else None
    if not edited.exists():
        return _fail("lines_changed_bound", f"{fname} missing")
    if base is None or not base.exists():
        return _fail("lines_changed_bound", f"no baseline fixture for {fname}")
    a = base.read_text().splitlines()
    b = edited.read_text().splitlines()
    diff = list(difflib.unified_diff(a, b, lineterm=""))
    changed = sum(
        1 for ln in diff
        if (ln.startswith("+") and not ln.startswith("+++"))
        or (ln.startswith("-") and not ln.startswith("---"))
    )
    if changed > max_lines:
        return _fail("lines_changed_bound", f"{changed} lines changed, bound {max_lines}")
    return _pass("lines_changed_bound", f"{changed} lines changed (bound {max_lines})")


def gate_no_new_dependencies(spec: dict, work: Path, record: dict, baseline: Path | None) -> dict:
    """The edited file imports no module the baseline did not already import, outside stdlib.

    Enforces a "do not add dependencies" constraint without a third-party type checker:
    an agent that papers over a bug by importing numpy fails, one that reaches for the
    stdlib (``re``, ``csv``) passes.
    """
    fname = spec.get("file")
    if not fname:
        return _fail("no_new_dependencies", "gate spec needs file")
    edited = work / fname
    if not edited.exists():
        return _fail("no_new_dependencies", f"{fname} missing")
    try:
        new = _module_imports(edited)
    except SyntaxError as exc:
        return _fail("no_new_dependencies", f"{fname} does not parse: {exc}")
    base_mods = _module_imports(baseline / fname) if (baseline and (baseline / fname).exists()) else set()
    added = new - base_mods
    new_third_party = sorted(m for m in added if m not in STDLIB_MODULES)
    if new_third_party:
        return _fail("no_new_dependencies", f"new third-party import(s): {new_third_party}")
    return _pass("no_new_dependencies", f"no new third-party imports (added: {sorted(added)})")


def gate_no_old_names(spec: dict, work: Path, record: dict, baseline: Path | None) -> dict:
    """None of the declared old names exist in the workspace any more."""
    old = list(spec.get("old_names") or [])
    if not old:
        return _fail("no_old_names", "gate spec needs old_names")
    present = [n for n in old if (work / n).exists()]
    if present:
        return _fail("no_old_names", f"old name(s) still present: {present}")
    return _pass("no_old_names", f"none of {old} remain")


def _workspace_files(work: Path) -> set[str]:
    """Relative POSIX paths of the files the agent left, minus ephemeral dirs and results."""
    out: set[str] = set()
    for p in work.rglob("*"):
        if not p.is_file() or p.name == "result.json":
            continue
        if any(part in EXCLUDE_DIRS for part in p.parts):
            continue
        out.add(p.relative_to(work).as_posix())
    return out


def gate_no_unrelated_files(spec: dict, work: Path, record: dict, baseline: Path | None) -> dict:
    """The workspace holds exactly the declared set of files (no extras, none missing).

    Catches both "touched an unrelated file" and "created scratch junk". The declared
    set is the expected FINAL file set, so a correct rename that drops the old names and
    adds the new ones passes.
    """
    allowed = set(spec.get("allowed_files") or [])
    if not allowed:
        return _fail("no_unrelated_files", "gate spec needs allowed_files")
    got = _workspace_files(work)
    extra = got - allowed
    missing = allowed - got
    if extra or missing:
        return _fail("no_unrelated_files", f"extra={sorted(extra)}; missing={sorted(missing)}")
    return _pass("no_unrelated_files", f"files == {sorted(got)}")


GATE_HELPERS: dict[str, Callable[[dict, Path, dict, Path | None], dict]] = {
    "json_schema_strict": gate_json_schema_strict,
    "no_trailing_blank_line": gate_no_trailing_blank_line,
    "no_markdown_fences": gate_no_markdown_fences,
    "lines_changed_bound": gate_lines_changed_bound,
    "no_new_dependencies": gate_no_new_dependencies,
    "no_old_names": gate_no_old_names,
    "no_unrelated_files": gate_no_unrelated_files,
}


# --------------------------------------------------------------------------
# Dispatcher: evaluate in order, stop at the first failure.
# --------------------------------------------------------------------------

def evaluate_quality_gates(gates: list[dict], work: Path, record: dict,
                           baseline: Path | None) -> dict:
    """Run each gate in order, stopping at the first failure.

    Returns a dict with the per-gate results actually evaluated, the count of gates the
    task declared, and ``all_pass`` (True only when no gate failed and at least one gate
    ran). A gate named in the task but missing from the registry fails immediately and
    stops the chain, so a typo in task.json cannot silently skip a gate.
    """
    results: list[dict] = []
    stopped_at: int | None = None
    for i, spec in enumerate(gates or []):
        name = spec.get("gate")
        helper = GATE_HELPERS.get(name) if name else None
        if helper is None:
            results.append(_fail(name or "?", f"unknown gate type: {name!r}"))
            stopped_at = i
            break
        res = helper(spec, work, record, baseline)
        results.append(res)
        if not res["pass"]:
            stopped_at = i
            break
    passed = sum(1 for r in results if r["pass"])
    return {
        "gates": results,
        "total": len(gates or []),
        "evaluated": len(results),
        "passed": passed,
        "stopped_at": stopped_at,
        "all_pass": stopped_at is None and bool(gates),
    }


def apply_quality_gates(task, work: Path, record: dict) -> dict:
    """Evaluate a task's gates and store the result and secondary score on the record.

    Called by the runner right after the judge grades the workspace, and by the rescore
    path, so every record carries the same gate verdict. ``quality_score`` is null when
    the task declares no gates, so a gate-free task never looks like a perfect gate run.
    """
    gates = list(getattr(task, "quality_gates", None) or [])
    baseline = task.path / "workspace" if (task.path / "workspace").is_dir() else None
    result = evaluate_quality_gates(gates, work, record, baseline)
    # The gates block holds only the per-gate results; the secondary score is a separate
    # top-level record field, so a reader never has to guess which number is which.
    score = None if not gates else (1.0 if result["all_pass"] else 0.0)
    record["quality_gates"] = result
    record["quality_score"] = score
    return record


# --------------------------------------------------------------------------
# Reporting: the secondary binary score, shown separately from the pass rate.
# --------------------------------------------------------------------------

def _fmt_pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _gate_attempts(records: list[dict], model: str | None = None) -> list[tuple[str, int, float]]:
    """Sorable, gate-bearing attempts as (task, repeat, quality_score), per model."""
    out: list[tuple[str, int, float]] = []
    for r in records:
        if model is not None and (r.get("model") or "config-default") != model:
            continue
        if not uncertlib._score_source_ok(r):
            continue
        q = r.get("quality_score")
        if q is None:
            continue
        out.append((r["task"], int(r.get("repeat") or 0), float(q)))
    return out


def format_single_run(run_id: str, records: list[dict], model: str | None = None) -> str:
    attempts = _gate_attempts(records, model=model)
    header = f"{run_id} ({model})" if model else run_id
    if not attempts:
        return f"Quality gates for {header}: no gate-bearing attempts in this run."
    n = len(attempts)
    passed = sum(1 for _, _, q in attempts if q == 1.0)
    bytask: dict[str, list[int]] = {}
    for t, _rep, q in attempts:
        cell = bytask.setdefault(t, [0, 0])
        cell[1] += 1
        if q == 1.0:
            cell[0] += 1
    lines = [
        f"Quality gates for {header}",
        "=" * 78,
        f"  secondary binary score: {passed}/{n} gate-bearing attempt(s) passed ALL their",
        f"  gates ({_fmt_pct(passed / n)}). Reported separately from the pass rate; each gate is a",
        "  deterministic code check and the runner stops at the first failure.",
        "",
        f"  {'task':32} {'pass/total':>12}",
        "  " + "-" * 78,
    ]
    for t in sorted(bytask):
        p, tot = bytask[t]
        lines.append(f"  {t:32} {p:>5}/{tot:<6}")
    return "\n".join(lines)


def report_single_run(run_id: str, runs_dir: Path | None = None) -> str:
    records = uncertlib.load_run_records(run_id, runs_dir)
    models = uncertlib.run_models(records)
    if len(models) == 1:
        return format_single_run(run_id, records, model=None)
    out = [f"Quality gates for {run_id}  ({len(models)} models)", "=" * 78]
    for m in models:
        out.append("")
        out.append(format_single_run(run_id, records, model=m))
    return "\n".join(out)


def format_comparison(run_a: str, run_b: str, records_a: list[dict],
                      records_b: list[dict], model_a: str | None = None,
                      model_b: str | None = None) -> str:
    a = _gate_attempts(records_a, model=model_a)
    b = _gate_attempts(records_b, model=model_b)
    la = f"{model_a}" if model_a else run_a
    lb = f"{model_b}" if model_b else run_b
    lines = [f"Quality gates comparison: {run_a} vs {run_b}", "=" * 78]
    if not a and not b:
        lines.append("  neither run has gate-bearing attempts.")
        return "\n".join(lines)
    pa = sum(1 for _, _, q in a if q == 1.0) / len(a) if a else float("nan")
    pb = sum(1 for _, _, q in b if q == 1.0) / len(b) if b else float("nan")
    lines.append(f"  {'run':28} {'gate pass rate':>16} {'attempts':>10}")
    lines.append("  " + "-" * 60)
    lines.append(f"  {la:28} {_fmt_pct(pa) if a else 'n/a':>16} {len(a):>10}")
    lines.append(f"  {lb:28} {_fmt_pct(pb) if b else 'n/a':>16} {len(b):>10}")
    if a and b:
        gap = pa - pb
        arrow = "higher" if gap > 0 else ("lower" if gap < 0 else "equal")
        lines.append(f"  gap {gap * 100:+.0f} pts ({la} {arrow}). A gate pass rate is a")
        lines.append("  secondary binary score, not a ranking test (see hbench_uncertainty).")
    return "\n".join(lines)


def report_comparison(run_a: str, run_b: str, runs_dir: Path | None = None,
                      model_a: str | None = None, model_b: str | None = None) -> str:
    records_a = uncertlib.load_run_records(run_a, runs_dir)
    records_b = uncertlib.load_run_records(run_b, runs_dir)
    if model_a is None:
        model_a = uncertlib._resolve_model(run_a, records_a, None)
    if model_b is None:
        model_b = uncertlib._resolve_model(run_b, records_b, None)
    return format_comparison(run_a, run_b, records_a, records_b,
                             model_a=model_a, model_b=model_b)


# --------------------------------------------------------------------------
# CLI: report / compare, mirroring hbench_coresmoke so the section is inspectable
# on its own. The main integration is the appended section in hbench.py report/compare.
# --------------------------------------------------------------------------

def cmd_report(args) -> int:
    try:
        print(report_single_run(args.run_id))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


def cmd_compare(args) -> int:
    try:
        print(report_comparison(args.run_a, args.run_b, model_a=args.model_a,
                                model_b=args.model_b))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"comparison refused: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=(__doc__ or "hbench quality gates").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_rep = sub.add_parser("report", help="secondary binary quality score for one run")
    p_rep.add_argument("run_id")
    p_rep.set_defaults(func=cmd_report)
    p_cmp = sub.add_parser("compare", help="quality pass rate for two runs")
    p_cmp.add_argument("run_a")
    p_cmp.add_argument("run_b")
    p_cmp.add_argument("--model-a", help="which model of run_a to use (only if multi-model)")
    p_cmp.add_argument("--model-b", help="which model of run_b to use (only if multi-model)")
    p_cmp.set_defaults(func=cmd_compare)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
