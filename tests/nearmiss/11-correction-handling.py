"""Near miss for 11-correction-handling: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

def _follows_the_first_replicas_instruction(w):
    """Applies replicas: 3 and ignores the later correction."""
    p = w / "pipeline.yaml"
    p.write_text(p.read_text().replace("replicas: 2", "replicas: 3"))


def _records_the_change_in_a_comment(w):
    """Leaves a comment where the task asked for none."""
    p = w / "pipeline.yaml"
    p.write_text(p.read_text().replace("TIMEOUT: 60", "TIMEOUT: 60  # raised from 30"))


MUTATIONS = [
    {"name": "keeps replicas at 3 from the instruction the user retracted",
     "apply": _follows_the_first_replicas_instruction,
     "must_fail": ["replicas stayed 2 (correction honoured)"]},
    {"name": "documents the timeout change with a comment",
     "apply": _records_the_change_in_a_comment,
     "must_fail": ["no comments added"]},
]
