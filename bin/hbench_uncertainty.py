#!/usr/bin/env python3
"""Uncertainty-aware reporting for hbench: bootstrap intervals and an indistinguishable verdict.

The defect this closes
----------------------
hbench ranked a 9B (Ornith 1.5, 0.9597) against a 1.7B (LFM3, 0.9676) on 12 tasks x 3
repeats and the ranking was noise. Only 4 of 12 tasks differ between the two models, and
the paired interval [-0.011, +0.027] spans zero. The number a reader remembers is the
macro score to two decimals, and two such numbers side by side read as a ranking whether
or not the gap is real. A p-value would not help: with four discordant tasks a task-level
sign test has only 2^4 = 16 patterns, so the smallest achievable two-sided p is 2/16 =
0.125. Significance is unreachable by construction, so a published p-value from this
design reads as rigour while delivering none.

What this module does instead
-----------------------------
Report three things and no more, for a score or for a comparison:

  * a point estimate, the macro-average across tasks. The estimand is the task, never the
    attempt: attempts inside one task are clustered and must not be counted as independent
    observations, so a task contributes once (its own mean) to the macro number.
  * a confidence interval obtained by resampling at the task level, because there are only
    12 tasks and a normal approximation on 12 clusters is not trustworthy:
      - a paired bootstrap over tasks for a comparison of two models, and
      - a hierarchical bootstrap (resample tasks, then attempts within a task) for an
        absolute score, which carries both the between-task and the within-task variance.
  * a verdict: ``indistinguishable`` when the interval crosses zero or the gap is smaller
    than the two-percentage-point practical-equivalence threshold this deployment cares
    about, otherwise ``separable`` with the direction and size of the gap.

No p-value is ever computed or printed here. The interval and the verdict carry the
information a p-value would, and each names the threshold it is measured against.

Run it directly to compare two archived runs, or read a single run's uncertainty:

    python3 bin/hbench_uncertainty.py compare <run_a> <run_b>
    python3 bin/hbench_uncertainty.py report <run_id>
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(os.environ.get("HBENCH_ROOT", "/root/hermes-bench"))
RUNS_DIR = ROOT / "runs"

# The practical-equivalence gap for this deployment: two percentage points. A difference
# smaller than this is not worth acting on even if the data could resolve it, and a
# difference this large or larger is only "separable" if the interval also excludes zero.
EQUIV_THRESHOLD = 0.02

DEFAULT_CONFIDENCE = 0.95
DEFAULT_N_BOOT = 10_000
# Fixed by default so a published interval is reproducible: the same records must always
# yield the same reported CI. A caller can pass a different seed, but the published number
# should not drift run to run for the same measurement.
DEFAULT_SEED = 20261101


# --------------------------------------------------------------------------
# Data extraction
# --------------------------------------------------------------------------

def _score_source_ok(record: dict) -> bool:
    """Whether a record's score is the model's own artifact score, matching the bench.

    Mirrors the leaderboard filter in hbench_publish.publish and the scorable() rule in
    hbench.py: a record only counts when its score is the artifact score, and an older row
    with no score_source counts unless it is flagged as a harness fault.
    """
    src = record.get("score_source")
    if src:
        return src == "artifact"
    return not record.get("harness_failed")


def records_to_task_scores(records: Iterable[dict], model: str | None = None) -> dict[str, list[float]]:
    """Group attempt records into ``task -> [score, ...]`` over the scorable attempts.

    The list keeps every repeat as a separate observation, which is what the hierarchical
    bootstrap resamples within a task. Excluded attempts (a substitution, an argv refusal,
    a harness fault) are dropped so they cannot drag a task mean down as if the model had
    attempted it. Pass ``model`` to keep only one model's attempts when a run measured more
    than one; ``records_to_task_scores`` then needs the caller to name which one.
    """
    out: dict[str, list[float]] = {}
    for r in records:
        if model is not None and (r.get("model") or "config-default") != model:
            continue
        if not _score_source_ok(r):
            continue
        out.setdefault(r["task"], []).append(float(r["score"]))
    return out


def run_models(records: Iterable[dict]) -> list[str]:
    """The distinct model strings in a run, in first-seen order."""
    out: list[str] = []
    for r in records:
        m = r.get("model") or "config-default"
        if m not in out:
            out.append(m)
    return out


def load_run_records(run_id: str, runs_dir: Path | None = None) -> list[dict]:
    path = runs_dir or RUNS_DIR
    results_path = path / run_id / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"no results.json for run {run_id} at {results_path}")
    return json.loads(results_path.read_text())


# --------------------------------------------------------------------------
# Point estimate
# --------------------------------------------------------------------------

def macro_score(task_scores: dict[str, list[float]]) -> float:
    """Macro-average across tasks: the mean of the per-task mean.

    Tasks are the estimand, not attempts. A task with three attempts at 1, 1, 0 and a task
    with one attempt at 0.5 each contribute their own mean (0.667 and 0.5) to the macro
    number with equal weight, so a heavily-repeated easy task cannot outvote the hard ones.
    """
    if not task_scores:
        return float("nan")
    return statistics.fmean(statistics.fmean(v) for v in task_scores.values())


# --------------------------------------------------------------------------
# Bootstrap intervals (pure stdlib, deterministic given a seed)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap confidence interval."""

    point: float
    lo: float
    hi: float
    n_boot: int = DEFAULT_N_BOOT
    confidence: float = DEFAULT_CONFIDENCE
    method: str = ""


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (the numpy 'linear' convention) on sorted input.

    ``pct`` is 0-100. With 10000 bootstrap replicates the endpoints are stable to well
    under a tenth of a point, so no external statistics package is needed.
    """
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def hierarchical_bootstrap_ci(
    task_scores: dict[str, list[float]],
    n_boot: int = DEFAULT_N_BOOT,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Interval for an absolute macro score by resampling tasks, then attempts within tasks.

    Each replicate draws the tasks with replacement (there are only ``len(task_scores)`` of
    them), and for every drawn task draws its attempts with replacement before taking that
    task's mean. Averaging those task means over the resampled tasks reproduces the macro
    estimator at the bootstrap replicate, so the interval carries both levels of variance:
    which tasks happened to be in the sample, and how noisy each task's attempts are.
    """
    tasks = list(task_scores.keys())
    n_tasks = len(tasks)
    if n_tasks == 0:
        return Interval(float("nan"), float("nan"), float("nan"), n_boot, confidence, "hierarchical")
    # Pre-sort each task's attempts so the within-task resample is a stable draw.
    attempt_pool = [sorted(task_scores[t]) for t in tasks]
    alpha = (1.0 - confidence) / 2.0
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n_tasks):
            pool = attempt_pool[rng.randrange(n_tasks)]
            k = len(pool)
            pick = [pool[rng.randrange(k)] for _ in range(k)]
            total += statistics.fmean(pick)
        replicates.append(total / n_tasks)
    replicates.sort()
    return Interval(
        point=macro_score(task_scores),
        lo=_percentile(replicates, alpha * 100.0),
        hi=_percentile(replicates, (1.0 - alpha) * 100.0),
        n_boot=n_boot,
        confidence=confidence,
        method="hierarchical",
    )


def paired_bootstrap_ci(
    task_scores_a: dict[str, list[float]],
    task_scores_b: dict[str, list[float]],
    n_boot: int = DEFAULT_N_BOOT,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Interval for a model comparison by resampling tasks over the paired per-task gap.

    The two models are measured on the same tasks, so the comparison is paired: for each
    shared task take the difference of the two per-task means, then resample those tasks
    with replacement and average the drawn differences. Only tasks both models attempted
    enter the statistic; a task one model never ran is not a zero, it is simply not part of
    the paired estimand. The reported point is the observed mean paired difference (A - B).
    """
    common = sorted(set(task_scores_a) & set(task_scores_b))
    diffs = [statistics.fmean(task_scores_a[t]) - statistics.fmean(task_scores_b[t])
             for t in common]
    n_tasks = len(diffs)
    if n_tasks == 0:
        return Interval(float("nan"), float("nan"), float("nan"), n_boot, confidence, "paired")
    alpha = (1.0 - confidence) / 2.0
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n_tasks):
            total += diffs[rng.randrange(n_tasks)]
        replicates.append(total / n_tasks)
    replicates.sort()
    return Interval(
        point=statistics.fmean(diffs),
        lo=_percentile(replicates, alpha * 100.0),
        hi=_percentile(replicates, (1.0 - alpha) * 100.0),
        n_boot=n_boot,
        confidence=confidence,
        method="paired",
    )


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    """Whether two models are separable, and why."""

    label: str                    # "indistinguishable" or "separable"
    effect: float                 # signed difference, A - B
    lo: float
    hi: float
    threshold: float
    discordant_tasks: int         # shared tasks whose paired gap is nonzero
    shared_tasks: int
    reason: str

    @property
    def separable(self) -> bool:
        return self.label == "separable"


def assess_verdict(
    effect: float,
    lo: float,
    hi: float,
    threshold: float = EQUIV_THRESHOLD,
    discordant_tasks: int = 0,
    shared_tasks: int = 0,
) -> Verdict:
    """Combine the interval and the practical-equivalence threshold into one verdict.

    ``indistinguishable`` when either of two conditions holds:

      * the interval crosses zero, so the data cannot rule out "no difference", or
      * the point gap is below ``threshold`` (two points), so even a perfectly resolved
        difference would be too small to act on.

    Only a gap at least ``threshold`` wide whose interval excludes zero on the same side is
    ``separable``. That double gate is what stops a tiny gap with a tight interval, or a
    wide gap whose interval still straddles zero, from being published as a ranking.
    """
    crosses_zero = lo < 0.0 < hi
    below_threshold = abs(effect) < threshold
    if crosses_zero or below_threshold:
        if crosses_zero and below_threshold:
            reason = (f"interval {fmt_pair(lo, hi)} crosses zero and the gap "
                      f"{abs(effect):.3f} is below the {threshold:.2f} equivalence point")
        elif crosses_zero:
            reason = (f"interval {fmt_pair(lo, hi)} crosses zero; the data cannot rule out "
                      f"no difference")
        else:
            reason = (f"gap {abs(effect):.3f} is below the {threshold:.2f} practical-"
                      f"equivalence point even though the interval {fmt_pair(lo, hi)} excludes zero")
        return Verdict("indistinguishable", effect, lo, hi, threshold,
                       discordant_tasks, shared_tasks, reason)
    direction = "A beats B" if effect > 0 else "B beats A"
    reason = (f"interval {fmt_pair(lo, hi)} excludes zero and the gap {abs(effect):.3f} "
              f"exceeds the {threshold:.2f} point, so the ordering is {direction}")
    return Verdict("separable", effect, lo, hi, threshold, discordant_tasks, shared_tasks, reason)


# --------------------------------------------------------------------------
# Formatting (no p-value anywhere)
# --------------------------------------------------------------------------

def fmt_pair(lo: float, hi: float) -> str:
    """A confidence interval as ``[lo, hi]``; brackets, never a range dash."""
    return f"[{lo:+.3f}, {hi:+.3f}]"


def fmt_score_ci(interval: Interval, digits: int = 3) -> str:
    """Point estimate with its interval, e.g. ``0.960 [95% CI +0.942, +0.971]``."""
    ci = f"{interval.confidence:.0%} CI " + fmt_pair(interval.lo, interval.hi)
    return f"{interval.point:.{digits}f} ({ci})"


def fmt_verdict_line(verdict: Verdict) -> str:
    """One verdict line. The label is the first word so it survives a grep."""
    gap = abs(verdict.effect)
    return (f"VERDICT: {verdict.label}  (gap {gap:.3f} pts, {verdict.effect:+.3f} = A - B; "
            f"{fmt_pair(verdict.lo, verdict.hi)}; {verdict.discordant_tasks} of "
            f"{verdict.shared_tasks} shared tasks differ; {verdict.reason})")


# --------------------------------------------------------------------------
# High-level report builders used by the CLI
# --------------------------------------------------------------------------

def report_single_run(run_id: str, runs_dir: Path | None = None,
                      n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> str:
    """Point estimate plus interval and a per-task table for one run.

    A run can measure more than one model; each gets its own section, because the macro
    score and its interval are per-model quantities.
    """
    records = load_run_records(run_id, runs_dir)
    models = run_models(records)
    if len(models) == 1:
        return format_single_run(run_id, records, model=None, n_boot=n_boot, seed=seed)
    out: list[str] = [f"Uncertainty for {run_id}  ({len(models)} models)"]
    out.append("=" * 78)
    for m in models:
        out.append("")
        out.append(format_single_run(run_id, records, model=m, n_boot=n_boot, seed=seed))
    return "\n".join(out)


def format_single_run(run_id: str, records: list[dict], model: str | None = None,
                      n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> str:
    task_scores = records_to_task_scores(records, model=model)
    if not task_scores:
        label = model or "?"
        return f"{run_id} ({label}): no scorable records to summarise uncertainty for."
    ci = hierarchical_bootstrap_ci(task_scores, n_boot=n_boot, seed=seed)
    n_attempts = sum(len(v) for v in task_scores.values())
    n_repeats = max(len(v) for v in task_scores.values())
    header = f"{run_id} ({model})" if model else run_id
    lines: list[str] = []
    lines.append(f"Uncertainty for {header}")
    lines.append("=" * 78)
    lines.append(f"  ({len(task_scores)} tasks, {n_attempts} attempts, up to {n_repeats} repeats)")
    lines.append(f"  macro score {fmt_score_ci(ci)}   "
                 f"(hierarchical bootstrap, {ci.n_boot:,} resamples, seed {seed})")
    lines.append("  estimand is the macro-average across tasks; attempts within a task are")
    lines.append("  clustered, so the interval resamples tasks, then attempts inside a task.")
    lines.append("")
    lines.append("  Per-task detail (mean over repeats, spread = min..max):")
    lines.append("  " + "-" * 70)
    lines.append(f"  {'task':32} {'mean':>6} {'min':>6} {'max':>6}  repeats")
    lines.append("  " + "-" * 70)
    for t in sorted(task_scores):
        scores = task_scores[t]
        mean = statistics.fmean(scores)
        flag = "  <- ceiling" if mean >= 1.0 else ""
        lines.append(f"  {t:32} {mean:6.3f} {min(scores):6.3f} {max(scores):6.3f}  "
                     f"{len(scores)}{flag}")
    lines.append("  " + "-" * 70)
    ceiling = sum(1 for t in task_scores if statistics.fmean(task_scores[t]) >= 1.0)
    lines.append(f"  {ceiling} of {len(task_scores)} tasks sit at the ceiling (mean 1.000); "
                 f"the spread a reader needs to see that most tasks carry no signal.")
    return "\n".join(lines)


def _resolve_model(run_id: str, records: list[dict], requested: str | None) -> str:
    """Pick the model to report from a run.

    A single-model run needs no naming. A multi-model run must be told which model, so a
    comparison is never silently stitched from the first model of one run and the first of
    another. Raises ValueError with the run's model list when the caller did not choose.
    """
    models = run_models(records)
    if requested:
        if requested not in models:
            raise ValueError(f"{run_id} has no model {requested!r}; it measured: {models}")
        return requested
    if len(models) == 1:
        return models[0]
    raise ValueError(f"{run_id} measured {len(models)} models: {models}. "
                     f"Pass --model-a / --model-b to name which one to compare.")


def report_comparison(run_a: str, run_b: str, runs_dir: Path | None = None,
                      model_a: str | None = None, model_b: str | None = None,
                      n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED) -> str:
    """Paired comparison of two runs, with point estimates, interval and verdict."""
    rec_a = load_run_records(run_a, runs_dir)
    rec_b = load_run_records(run_b, runs_dir)
    resolved_a = _resolve_model(run_a, rec_a, model_a)
    resolved_b = _resolve_model(run_b, rec_b, model_b)
    task_scores_a = records_to_task_scores(rec_a, model=resolved_a)
    task_scores_b = records_to_task_scores(rec_b, model=resolved_b)
    return format_comparison(run_a, task_scores_a, run_b, task_scores_b,
                             resolved_a, resolved_b, n_boot=n_boot, seed=seed)


def format_comparison(run_a: str, task_scores_a: dict[str, list[float]],
                      run_b: str, task_scores_b: dict[str, list[float]],
                      model_a: str | None = None, model_b: str | None = None,
                      n_boot: int = DEFAULT_N_BOOT, seed: int = DEFAULT_SEED,
                      threshold: float = EQUIV_THRESHOLD) -> str:
    label_a = model_a or "A"
    label_b = model_b or "B"
    if not task_scores_a or not task_scores_b:
        return (f"cannot compare: {run_a} has {len(task_scores_a)} task(s), "
                f"{run_b} has {len(task_scores_b)} task(s).")
    ci_a = hierarchical_bootstrap_ci(task_scores_a, n_boot=n_boot, seed=seed)
    ci_b = hierarchical_bootstrap_ci(task_scores_b, n_boot=n_boot, seed=seed)
    paired = paired_bootstrap_ci(task_scores_a, task_scores_b, n_boot=n_boot, seed=seed)
    common = sorted(set(task_scores_a) & set(task_scores_b))
    discordant = sum(1 for t in common
                     if abs(statistics.fmean(task_scores_a[t]) - statistics.fmean(task_scores_b[t])) > 1e-9)
    verdict = assess_verdict(paired.point, paired.lo, paired.hi, threshold=threshold,
                             discordant_tasks=discordant, shared_tasks=len(common))

    lines: list[str] = []
    lines.append(f"Comparison {run_a} ({label_a}) vs {run_b} ({label_b})")
    lines.append("=" * 78)
    lines.append(f"  {label_a}  macro {fmt_score_ci(ci_a)}")
    lines.append(f"  {label_b}  macro {fmt_score_ci(ci_b)}")
    lines.append("")
    lines.append(f"  Difference (A - B) {fmt_score_ci(paired)}  "
                 f"[{paired.method} bootstrap, {paired.n_boot:,} resamples, seed {seed}]")
    lines.append("")
    lines.append("  Per-task paired detail (mean over repeats; diff = A - B):")
    lines.append("  " + "-" * 74)
    lines.append(f"  {'task':32} {label_a:>9} {label_b:>9} {'diff':>9}")
    lines.append("  " + "-" * 74)
    for t in common:
        ma = statistics.fmean(task_scores_a[t])
        mb = statistics.fmean(task_scores_b[t])
        d = ma - mb
        mark = "" if abs(d) <= 1e-9 else "  *"
        lines.append(f"  {t:32} {ma:9.3f} {mb:9.3f} {d:+9.3f}{mark}")
    lines.append("  " + "-" * 74)
    lines.append(f"  * marks a task whose paired gap is nonzero. {discordant} of "
                 f"{len(common)} shared tasks differ.")
    lines.append("")
    lines.append(fmt_verdict_line(verdict))
    lines.append(f"  Verdict is 'indistinguishable' when the paired interval crosses zero or "
                 f"the gap is")
    lines.append(f"  below the {threshold:.0%} practical-equivalence point. No significance "
                 "test is printed:")
    lines.append("  12 tasks cannot resolve one, so the interval and this verdict are the result.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _add_boot_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT,
                   help="bootstrap resamples (default %(default)s)")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="rng seed, so the reported interval is reproducible (default %(default)s)")


def cmd_report(args) -> int:
    try:
        print(report_single_run(args.run_id, n_boot=args.n_boot, seed=args.seed))
    except (OSError, ValueError, KeyError) as exc:
        print(f"could not summarise {args.run_id}: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_compare(args) -> int:
    try:
        print(report_comparison(args.run_a, args.run_b, model_a=args.model_a,
                                model_b=args.model_b, n_boot=args.n_boot, seed=args.seed))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"comparison refused: {exc}", file=sys.stderr)
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "hbench uncertainty reporting").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rep = sub.add_parser("report", help="point estimate, interval and per-task table for one run")
    p_rep.add_argument("run_id")
    _add_boot_args(p_rep)
    p_rep.set_defaults(func=cmd_report)

    p_cmp = sub.add_parser("compare", help="paired comparison of two runs with an indistinguishable verdict")
    p_cmp.add_argument("run_a")
    p_cmp.add_argument("run_b")
    p_cmp.add_argument("--model-a", help="label for the first model in the output")
    p_cmp.add_argument("--model-b", help="label for the second model in the output")
    _add_boot_args(p_cmp)
    p_cmp.set_defaults(func=cmd_compare)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
