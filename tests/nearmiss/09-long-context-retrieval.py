"""Near miss for 09-long-context-retrieval: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

import json


def _finds_only_the_first_code(w):
    """Reports one of the three codes, with a count that matches what it found."""
    p = w / "codes.json"
    data = json.loads(p.read_text())
    data["codes"] = data["codes"][:1]
    data["count"] = 1
    p.write_text(json.dumps(data, indent=2))


MUTATIONS = [
    {"name": "reports one code and a self-consistent count",
     "apply": _finds_only_the_first_code,
     "must_fail": ["all three codes found"]},
]
