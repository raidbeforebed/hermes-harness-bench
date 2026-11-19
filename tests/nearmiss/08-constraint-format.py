"""Near miss for 08-constraint-format: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

def _line_three_is_only_tied_for_longest(w):
    """Trims line 3 to the length of the longest other line: rule 6 now ties."""
    p = w / "format_check.txt"
    lines = p.read_text().split("\n")
    others = [len(line) for i, line in enumerate(lines) if i != 2]
    lines[2] = lines[2][:max(others)]
    p.write_text("\n".join(lines))


MUTATIONS = [
    {"name": "makes line 3 tie with the longest line instead of beating it",
     "apply": _line_three_is_only_tied_for_longest,
     "must_fail": ["line 3 is longest (strictly)"]},
]
