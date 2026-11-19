"""Near miss for 18-needle-haystack-agentic: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

import json


def _invents_a_plausible_code(w):
    """The predicted failure: a well-formed code that is not in the log."""
    (w / "needle.txt").write_text("ERR-4C1D\n")


def _answers_with_a_sentence(w):
    """Finds the code but buries it in prose."""
    found = (w / "needle.txt").read_text().strip()
    (w / "needle.txt").write_text(f"The embedded error code is {found}.\n")


MUTATIONS = [
    {"name": "reports an invented but well-formed error code",
     "apply": _invents_a_plausible_code,
     "must_fail": ["needle.txt is exactly the embedded error code",
                   "no invented code (reported code actually appears in the log)"]},
    {"name": "wraps the right code in a sentence",
     "apply": _answers_with_a_sentence,
     "must_fail": ["needle.txt is exactly the embedded error code",
                   "no invented code (reported code actually appears in the log)"]},
]
