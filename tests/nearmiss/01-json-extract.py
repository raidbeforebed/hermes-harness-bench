"""Near miss for 01-json-extract: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

import json


def _counts_cancelled_as_an_order(w):
    """A model that reads "total orders" as "orders that were not cancelled"."""
    p = w / "report.json"
    data = json.loads(p.read_text())
    data["total_orders"] = 6
    p.write_text(json.dumps(data, indent=2))


def _counts_pending_rows_as_shipped(w):
    """A model that treats every non-cancelled row as shipped revenue."""
    p = w / "report.json"
    data = json.loads(p.read_text())
    # 1002 Beta 1*45.00 and 1007 Delta 2*19.99 are pending, not shipped.
    data["by_customer"]["Beta LLC"]["revenue"] = round(30.00 + 45.00, 2)
    data["by_customer"]["Delta Co"]["revenue"] = round(2 * 19.99, 2)
    data["shipped_revenue"] = round(149.97 + 45.00 + 39.98, 2)
    p.write_text(json.dumps(data, indent=2))


MUTATIONS = [
    {"name": "counts only the orders that were not cancelled",
     "apply": _counts_cancelled_as_an_order,
     "must_fail": ["total_orders == 7"]},
    {"name": "counts pending rows as shipped revenue",
     "apply": _counts_pending_rows_as_shipped,
     "must_fail": ["shipped_revenue == 149.97", "per-customer orders+revenue"]},
]
