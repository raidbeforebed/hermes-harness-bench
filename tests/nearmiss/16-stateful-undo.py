"""Near miss for 16-stateful-undo: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

def _reverts_past_the_first_append(w):
    """Undoes one step too far: alpha is gone, charlie is appended to the header."""
    (w / "notes.md").write_text("# Notes\n- charlie\n")


MUTATIONS = [
    {"name": "reverts to the empty file instead of the state after the first append",
     "apply": _reverts_past_the_first_append,
     "must_fail": ["undo honoured: alpha kept, bravo dropped, charlie appended",
                   "exact final content (header, alpha, charlie, trailing newline)"]},
]
