"""Near miss for 03-test-fix: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

BUGGY = (
    '"""Turn arbitrary strings into URL slugs."""\n\n\n'
    "def slugify(text):\n"
    "    out = []\n"
    "    for ch in text.lower():\n"
    "        if ch.isalnum():\n"
    "            out.append(ch)\n"
    "        elif ch in \" -_\":\n"
    "            out.append(\"-\")\n"
    "    result = \"\".join(out).strip(\"-\").replace(\"--\", \"-\")\n"
    "    return result\n"
)


def _collapses_only_one_separator_run(w):
    """The nearest miss to the real bug: one replace() instead of a regex.

    It fixes "Hello  World" with two spaces but not "Hello   World" with three,
    so the shipped suite still fails.
    """
    (w / "slugify.py").write_text(BUGGY)


MUTATIONS = [
    {"name": "collapses one run of separators instead of all of them",
     "apply": _collapses_only_one_separator_run,
     # The two restraint checks are gated on a green suite, so a red suite fails
     # them too. That gate is what stops the untouched fixture earning points.
     "must_fail": ["pytest suite passes", "test file untouched", "slugify.py exists"]},
]
