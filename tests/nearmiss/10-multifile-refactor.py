"""Near miss for 10-multifile-refactor: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

def _verification_without_a_pass_count(w):
    """Says the tests pass without quoting the pytest summary line."""
    p = w / "CHANGES.md"
    text = p.read_text()
    head = text[:text.index("## Verification")] if "## Verification" in text else text
    p.write_text(head + "## Verification\nThe suite runs green now.\n")


MUTATIONS = [
    {"name": "claims a green suite without quoting the pytest pass count",
     "apply": _verification_without_a_pass_count,
     "must_fail": ["Verification quotes a pytest pass count"]},
]
