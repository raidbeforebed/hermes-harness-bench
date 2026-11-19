"""Near miss for 12-long-chain: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

import csv
import glob
import json


def _month_with_the_blank_qty(w):
    """The month whose file carries the empty qty row, as YYYY-MM."""
    for path in sorted(glob.glob(str(w / "sales_*.csv"))):
        for row in csv.DictReader(open(path)):
            if not (row.get("qty") or "").strip():
                return path.rsplit("sales_", 1)[1].split(".")[0]
    raise AssertionError("no blank qty row found: the fixture changed")


def _counts_the_skipped_row_as_used(w):
    """Counts the row it skipped in rows_used as well."""
    month = _month_with_the_blank_qty(w)
    p = w / "summary.json"
    data = json.loads(p.read_text())
    data["months"][month]["rows_used"] += 1
    p.write_text(json.dumps(data, indent=2))


def _charges_the_blank_qty_as_zero(w):
    """Treats the missing qty as 0 instead of skipping the row."""
    month = _month_with_the_blank_qty(w)
    p = w / "summary.json"
    data = json.loads(p.read_text())
    data["months"][month]["rows_used"] += 1
    data["skipped_rows"] = 0
    p.write_text(json.dumps(data, indent=2))


MUTATIONS = [
    {"name": "counts the skipped row in rows_used",
     "apply": _counts_the_skipped_row_as_used,
     "must_fail": ["rows_used correct per month"]},
    {"name": "treats the blank qty as zero instead of skipping the row",
     "apply": _charges_the_blank_qty_as_zero,
     "must_fail": ["rows_used correct per month", "skipped_rows correct"]},
]
