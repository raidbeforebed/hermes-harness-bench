"""Near miss for 15-dependency-chain-5: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

import re


def _leaves_the_callers_annotated_int(w):
    """Fixes the runtime value, leaves the four callers annotated -> int.

    This is the declared small-model failure mode: the chain runs and returns
    "42", so only the static annotation check can see the drift.
    """
    for name in ("a.py", "b.py", "c.py", "d.py"):
        p = w / name
        p.write_text(re.sub(r"->\s*str\s*:", "-> int:", p.read_text()))


MUTATIONS = [
    {"name": "returns the string but leaves the callers annotated int",
     "apply": _leaves_the_callers_annotated_int,
     "must_fail": ["all five functions annotated -> str (no type drift)"]},
]
