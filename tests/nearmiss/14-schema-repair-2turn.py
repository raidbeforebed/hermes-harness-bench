"""Near miss for 14-schema-repair-2turn: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

import json


def _adds_a_speculative_timeout(w):
    """A plausible extra field: the schema allows no properties but pattern+path."""
    p = w / "corrected_call.json"
    data = json.loads(p.read_text())
    data["args"]["timeout"] = 30
    p.write_text(json.dumps(data, indent=2))


def _renames_the_pattern_argument(w):
    """Rewrites the argument name to the one the model expected."""
    p = w / "corrected_call.json"
    data = json.loads(p.read_text())
    data["args"]["search_pattern"] = data["args"].pop("pattern")
    p.write_text(json.dumps(data, indent=2))


MUTATIONS = [
    {"name": "adds a speculative timeout argument",
     "apply": _adds_a_speculative_timeout,
     "must_fail": ["args has exactly pattern+path with string types"]},
    {"name": "renames pattern to search_pattern",
     "apply": _renames_the_pattern_argument,
     "must_fail": ["args has exactly pattern+path with string types",
                   "same operation, pattern and path as turn 1"]},
]
