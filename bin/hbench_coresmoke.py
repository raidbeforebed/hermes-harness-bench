#!/usr/bin/env python3
"""Separate the core score from the smoke score; never delete saturated tasks.

Why two scores
--------------
The macro score over the whole suite is dominated by the tasks every candidate already
aces. hbench's 12-task set had 7 or 8 such tasks: both models scored 1.000 on every
repeat, so those tasks added their perfect 1.0 to the macro mean and held it near 0.96 for
a model that actually differs from its rival on only a handful of tasks. The perfect tasks
do two bad things at once: they lift the reported level of the score, and they bury the
tasks where the models do differ, so a real gap reads as noise.

Deleting the saturated tasks would fix the number and break the bench. Saturation is
measured against the current candidates only, so a weaker future model could fail a task
that is trivial for today's two, and an easy task suddenly failing is a genuine regression
signal (a harness or variant change broke something), not a ranking result. The suite must
keep every task.

So report two aggregates instead of one:

  * smoke score: the macro-average over ALL tasks, the whole suite. It is expected at
    1.000. A drop in the smoke score is a harness or variant regression, not a model
    ranking, which is exactly why the task is kept rather than dropped.
  * core score: the macro-average over only the tasks that are not saturated, so
    saturation can neither inflate the reported level nor hide a real difference.

Classification rule (deterministic, fixed in code so it cannot be tuned after the results)
------------------------------------------------------------------------------------------
A task is ``saturated`` when the *lowest* per-candidate mean is at or above
``SATURATION_CEILING`` (1.000 by default): if even the weakest current candidate aced the
task on average, no candidate can be told apart from any other on it, so it carries no
ranking signal. A task is ``discriminating`` the moment at least one candidate's mean falls
below the ceiling, because that candidate can then be separated from the ones still at the
ceiling.

The ceiling is fixed at 1.000, not a fittable high number. A task a candidate solves only
part of the time keeps a sliver of signal and stays in the core set. Crucially the rule
looks only at the ceiling, never at whether model A outscored model B on a task, so the
core set is the same for every candidate scored against it and cannot be quietly widened or
narrowed to move a ranking. The reference set of candidates is named on the command line
(``--reference`` for a single run, and the two runs named to ``compare``), never inferred,
so the same discriminating set is applied to every model in the comparison. When the task
suite changes (harder tasks added, see the SPEC workstream 4) the reference set is re-run
and the classification recomputes itself from the new per-candidate means; nothing is
edited by hand.

Run it directly, or read it through ``hbench.py report`` and ``hbench.py compare``, which
append the split to the uncertainty section:

    python3 bin/hbench_coresmoke.py report <run_id> [--reference RUN ...]
    python3 bin/hbench_coresmoke.py compare <run_a> <run_b>
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hbench_uncertainty as U  # noqa: E402  (record filtering lives in one place)

# A task is saturated when the lowest per-candidate mean reaches this ceiling. Fixed at
# exactly 1.000, not a fittable number: the smoke score is itself "expected at 1.000", so
# the same ceiling defines both aggregates, and a task a candidate only nearly aces keeps
# its sliver of signal in the core set rather than being dropped on an arbitrary threshold.
SATURATION_CEILING = 1.0
# Float slack only: a mean of three exact 1.0 scores is 1.0, but a resampled mean should not
# fall out at 0.9999999 and flip a genuinely perfect task to discriminating.
CEILING_TOL = 1e-9


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

def candidate_task_means(records: list[dict]) -> dict[str, dict[str, float]]:
    """Task -> {model -> mean score} over every candidate model that attempted it.

    The mean is taken per model over that model's scorable attempts, which is the number a
    candidate is judged on. A model that never attempted a task contributes no entry for it,
    so it cannot be counted as a zero against that task. ``records`` already excludes
    substitutions, refusals and harness faults because it comes from
    ``U.records_to_task_scores`` under the hood.
    """
    out: dict[str, dict[str, float]] = {}
    for model in U.run_models(records):
        for task, scores in U.records_to_task_scores(records, model=model).items():
            out.setdefault(task, {})[model] = statistics.fmean(scores)
    return out


def saturated_task_set(
    candidate_means: dict[str, dict[str, float]],
    ceiling: float = SATURATION_CEILING,
) -> set[str]:
    """The tasks every candidate aced: the lowest per-candidate mean is at the ceiling.

    A task with no candidate mean (nothing attempted) is not saturated; it simply carries
    no signal either way and will not appear in any model's score.
    """
    sat: set[str] = set()
    for task, means in candidate_means.items():
        if means and min(means.values()) >= ceiling - CEILING_TOL:
            sat.add(task)
    return sat


@dataclass(frozen=True)
class CoreSmoke:
    """The two aggregates for one model, over one classification of the suite."""

    smoke: float                 # macro over ALL tasks the model attempted
    core: float                  # macro over the discriminating tasks the model attempted
    n_suite: int                 # tasks the model attempted (smoke denominator)
    n_core: int                  # discriminating tasks the model attempted
    n_saturated: int             # saturated tasks in the whole reference suite
    ceiling: float

    @property
    def has_core(self) -> bool:
        return self.n_core > 0


def compute_core_smoke(
    task_scores: dict[str, list[float]],
    saturated: set[str],
    ceiling: float = SATURATION_CEILING,
) -> CoreSmoke:
    """Core and smoke for one model's ``task -> [scores]`` against a fixed saturated set.

    The saturated set is computed once over the whole candidate panel and then applied
    verbatim here, so two models scored against the same panel share one discriminating set
    and their core numbers stay directly comparable. The core set is the discriminating
    subset of the tasks *this* model actually attempted; a saturated task is kept in the
    suite (it feeds the smoke score) but dropped from the core mean.
    """
    core_scores = {t: s for t, s in task_scores.items() if t not in saturated}
    return CoreSmoke(
        smoke=U.macro_score(task_scores),
        core=U.macro_score(core_scores) if core_scores else float("nan"),
        n_suite=len(task_scores),
        n_core=len(core_scores),
        n_saturated=len(saturated),
        ceiling=ceiling,
    )


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------

def _core_smoke_lines(cs: CoreSmoke) -> list[str]:
    """The two aggregate lines for one model: smoke, then core."""
    core_val = f"{cs.core:.3f}" if cs.has_core else "n/a (no discriminating task)"
    return [
        f"smoke {cs.smoke:.3f}  (all {cs.n_suite} tasks; expected 1.000, "
        f"a drop is a regression, not a ranking)",
        f"core  {core_val}  ({cs.n_core} discriminating task(s); "
        f"saturation excluded so it cannot inflate or hide a gap)",
    ]


def format_single_run(
    run_id: str,
    records: list[dict],
    saturated: set[str],
    ceiling: float = SATURATION_CEILING,
) -> str:
    """Core and smoke split for one run, one section per model it measured."""
    models = U.run_models(records)
    if len(models) == 1:
        return _format_single_model(run_id, records, model=None, saturated=saturated,
                                    ceiling=ceiling)
    out: list[str] = [f"Core / smoke split for {run_id}  ({len(models)} models)"]
    out.append("=" * 78)
    for model in models:
        out.append("")
        out.append(_format_single_model(run_id, records, model=model, saturated=saturated,
                                        ceiling=ceiling))
    return "\n".join(out)


def _format_single_model(
    run_id: str,
    records: list[dict],
    model: str | None,
    saturated: set[str],
    ceiling: float,
) -> str:
    task_scores = U.records_to_task_scores(records, model=model)
    if not task_scores:
        label = model or "?"
        return f"{run_id} ({label}): no scorable records to split into core and smoke."
    cs = compute_core_smoke(task_scores, saturated, ceiling=ceiling)
    label = f"{run_id} ({model})" if model else run_id
    lines: list[str] = []
    lines.append(f"Core / smoke split for {label}")
    lines.append("=" * 78)
    for line in _core_smoke_lines(cs):
        lines.append(f"  {line}")
    lines.append(f"  {cs.n_saturated} task(s) in the reference suite are saturated "
                 f"(every candidate at the ceiling {ceiling:.3f}).")
    lines.append("  Rule: a task is saturated when the lowest per-candidate mean reaches "
                 f"{ceiling:.3f}; it then")
    lines.append("  separates no candidate and leaves the core set. The rule reads the "
                 "ceiling only, never")
    lines.append("  which model won a task, so the core set cannot be tuned after the "
                 "results are known.")
    lines.append("")
    lines.append("  Per-task class (mean over repeats):")
    lines.append("  " + "-" * 70)
    lines.append(f"  {'task':32} {'mean':>6}  class")
    lines.append("  " + "-" * 70)
    for t in sorted(task_scores):
        mean = statistics.fmean(task_scores[t])
        cls = "saturated" if t in saturated else "discriminating"
        lines.append(f"  {t:32} {mean:6.3f}  {cls}")
    lines.append("  " + "-" * 70)
    return "\n".join(lines)


def format_comparison(
    run_a: str,
    task_scores_a: dict[str, list[float]],
    run_b: str,
    task_scores_b: dict[str, list[float]],
    saturated: set[str],
    ceiling: float = SATURATION_CEILING,
) -> str:
    """Core and smoke for a paired comparison, both models over the SAME discriminating set."""
    cs_a = compute_core_smoke(task_scores_a, saturated, ceiling=ceiling)
    cs_b = compute_core_smoke(task_scores_b, saturated, ceiling=ceiling)
    lines: list[str] = []
    lines.append(f"Core / smoke split  {run_a} (A) vs {run_b} (B)")
    lines.append("=" * 78)
    lines.append(f"  Reference candidates: {run_a} + {run_b}. A task is saturated when the "
                 f"lowest of the")
    lines.append(f"  two per-candidate means reaches {ceiling:.3f}, so both models share one "
                 "discriminating set.")
    lines.append("")
    la = _core_smoke_lines(cs_a)
    lb = _core_smoke_lines(cs_b)
    lines.append(f"  A  {la[0]}")
    lines.append(f"     {la[1]}")
    lines.append(f"  B  {lb[0]}")
    lines.append(f"     {lb[1]}")
    lines.append("")
    # Core and smoke gaps, so a reader sees whether saturation was masking the size of the gap.
    smoke_gap = cs_a.smoke - cs_b.smoke
    core_gap = cs_a.core - cs_b.core if (cs_a.has_core and cs_b.has_core) else float("nan")
    lines.append(f"  smoke gap (A - B) {smoke_gap:+.3f}")
    if cs_a.has_core and cs_b.has_core:
        lines.append(f"  core   gap (A - B) {core_gap:+.3f}   "
                     f"({'wider' if abs(core_gap) > abs(smoke_gap) else 'narrower'} than the "
                     "smoke gap: this is the gap saturation was hiding)")
    else:
        lines.append("  core   gap (A - B) n/a (one model has no discriminating task)")
    lines.append("")
    n_sat = len(saturated)
    lines.append(f"  {n_sat} of {len(task_scores_a) + len(set(task_scores_b) - set(task_scores_a))}"
                 f"  tasks saturated, {len(task_scores_a) - n_sat}"
                 " discriminating (shared suite).")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# High-level builders
# --------------------------------------------------------------------------

def _merge_records(run_ids: list[str], runs_dir: Path | None) -> list[dict]:
    """Concatenate the records of several runs into one candidate pool."""
    merged: list[dict] = []
    for rid in run_ids:
        merged.extend(U.load_run_records(rid, runs_dir))
    return merged


def report_single_run(
    run_id: str,
    reference: list[str] | None = None,
    runs_dir: Path | None = None,
    ceiling: float = SATURATION_CEILING,
) -> str:
    """Core and smoke split for one run.

    The reference candidate panel is the run itself plus any runs named to ``--reference``.
    The reported run is always in the panel, because its own model is one of the current
    candidates. With no ``--reference`` the panel is just this run, so a task is saturated
    when this model aced it; passing the sibling candidates makes the discriminating set the
    shared one used for a ranking.
    """
    run_records = U.load_run_records(run_id, runs_dir)
    panel = list(run_records)
    for rid in reference or []:
        if rid == run_id:
            continue
        panel.extend(U.load_run_records(rid, runs_dir))
    candidate_means = candidate_task_means(panel)
    saturated = saturated_task_set(candidate_means, ceiling=ceiling)
    if reference:
        ref_desc = "self + " + ", ".join(reference)
    else:
        ref_desc = "self only"
    body = format_single_run(run_id, run_records, saturated, ceiling=ceiling)
    return (f"Reference candidates: {ref_desc}\n"
            f"Saturation ceiling: {ceiling:.3f} (lowest per-candidate mean at or above this "
            f"=> saturated)\n" + body)


def report_comparison(
    run_a: str,
    run_b: str,
    runs_dir: Path | None = None,
    model_a: str | None = None,
    model_b: str | None = None,
    ceiling: float = SATURATION_CEILING,
) -> str:
    """Core and smoke split for a paired comparison of two runs.

    The candidate panel is the two runs named to the comparison, and the classification uses
    the two resolved models, so both core scores are computed over one shared discriminating
    set.
    """
    rec_a = U.load_run_records(run_a, runs_dir)
    rec_b = U.load_run_records(run_b, runs_dir)
    resolved_a = U._resolve_model(run_a, rec_a, model_a)
    resolved_b = U._resolve_model(run_b, rec_b, model_b)
    panel = rec_a + rec_b
    # Restrict the panel to the two models actually being compared, so a third model in a
    # multi-model run cannot reclassify a task out from under the comparison.
    panel_means = candidate_task_means(panel)
    keep = {t: m for t, m in panel_means.items()
            if resolved_a in m or resolved_b in m}
    restricted = {t: {k: v for k, v in m.items() if k in (resolved_a, resolved_b)}
                  for t, m in keep.items()}
    saturated = saturated_task_set(restricted, ceiling=ceiling)
    task_scores_a = U.records_to_task_scores(rec_a, model=resolved_a)
    task_scores_b = U.records_to_task_scores(rec_b, model=resolved_b)
    return format_comparison(run_a, task_scores_a, run_b, task_scores_b,
                             saturated, ceiling=ceiling)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_report(args) -> int:
    try:
        print(report_single_run(args.run_id, reference=args.reference, ceiling=args.ceiling))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"could not split {args.run_id}: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_compare(args) -> int:
    try:
        print(report_comparison(args.run_a, args.run_b, model_a=args.model_a,
                                model_b=args.model_b, ceiling=args.ceiling))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"comparison refused: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "hbench core/smoke split").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rep = sub.add_parser("report", help="core and smoke split for one run")
    p_rep.add_argument("run_id")
    p_rep.add_argument("--reference", action="append", default=[],
                       help="reference candidate run id, repeatable; the reported run is always "
                            "in the panel too")
    p_rep.add_argument("--ceiling", type=float, default=SATURATION_CEILING,
                       help="saturation ceiling, normally left at 1.000 (default %(default)s)")
    p_rep.set_defaults(func=cmd_report)

    p_cmp = sub.add_parser("compare", help="core and smoke split for two runs over one shared set")
    p_cmp.add_argument("run_a")
    p_cmp.add_argument("run_b")
    p_cmp.add_argument("--model-a", help="which model of run_a to use (only if it measured more than one)")
    p_cmp.add_argument("--model-b", help="which model of run_b to use (only if it measured more than one)")
    p_cmp.add_argument("--ceiling", type=float, default=SATURATION_CEILING,
                       help="saturation ceiling, normally left at 1.000 (default %(default)s)")
    p_cmp.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
