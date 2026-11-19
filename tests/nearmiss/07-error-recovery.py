"""Near miss for 07-error-recovery: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

FLOAT_QTY = (
    "id,status,qty\n"
    "1,active,12.0\n"
    "2,inactive,5.0\n"
    "3,active,8.0\n"
    "4,active,3.0\n"
    "5,active,7.0\n"
)


def _leaves_the_input_the_pipeline_rejects(w):
    """Does the analysis but never satisfies the validator, so nothing is green."""
    (w / "raw.csv").write_text(FLOAT_QTY)


MUTATIONS = [
    {"name": "leaves the decimal quantity the validator rejects",
     "apply": _leaves_the_input_the_pipeline_rejects,
     "must_fail": ["pipeline runs green"],
     # Every other check is gated on a green run, and the row-level checks are
     # not emitted at all when clean.csv is missing.
     "also_fail": ["build.py untouched (and credit requires a green run)",
                   "clean.csv produced (requires green run)"]},
]
