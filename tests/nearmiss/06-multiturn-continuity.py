"""Near miss for 06-multiturn-continuity: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

import json


def _integer_average(w):
    """Integer division on the mean milliseconds, the classic small-model slip."""
    p = w / "stage2.json"
    data = json.loads(p.read_text())
    for stats in data["by_service"].values():
        stats["avg_ms"] = int(stats["avg_ms"])
    p.write_text(json.dumps(data, indent=2))


MUTATIONS = [
    {"name": "averages with integer division instead of two decimals",
     "apply": _integer_average,
     "must_fail": ["stage2 by_service correct"]},
]
