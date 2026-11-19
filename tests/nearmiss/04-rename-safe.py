"""Near miss for 04-rename-safe: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

import shutil


def _copies_instead_of_moving(w):
    """Renames by copying to the new name and never removing the original."""
    shutil.copyfile(w / "config.bak", w / "config.ini")


MUTATIONS = [
    {"name": "copies config.ini to config.bak and leaves the original behind",
     "apply": _copies_instead_of_moving,
     "must_fail": ["config.ini removed"]},
]
