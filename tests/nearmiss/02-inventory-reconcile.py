"""Near miss for 02-inventory-reconcile: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

import json


def _swaps_the_mismatch_direction(w):
    """Reads system_b first and reports its quantity in the `a` slot."""
    p = w / "reconcile.json"
    data = json.loads(p.read_text())
    for mismatch in data["quantity_mismatch"]:
        mismatch["a"], mismatch["b"] = mismatch["b"], mismatch["a"]
    p.write_text(json.dumps(data, indent=2))


MUTATIONS = [
    {"name": "reports the quantity mismatch with a and b swapped",
     "apply": _swaps_the_mismatch_direction,
     "must_fail": ["one mismatch only (A-200 4 vs 6)"]},
]
