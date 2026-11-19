"""Near miss for 17-parallel-tool-coordination: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

ALPHA_ORIGINAL = """{
  "name": "alpha",
  "active": false,
  "retries": 3
}
"""
BETA_ORIGINAL = """{
  "name": "beta",
  "tags": ["x"],
  "priority": 2
}
"""
GAMMA_ORIGINAL = """{
  "name": "gamma",
  "limits": {
    "cpu": 1,
    "mem": 512
  }
}
"""


def _alpha_untouched(w):
    """Two of the three edits landed; the first one was dropped."""
    (w / "alpha.json").write_text(ALPHA_ORIGINAL)


def _beta_untouched(w):
    (w / "beta.json").write_text(BETA_ORIGINAL)


def _gamma_untouched(w):
    (w / "gamma.json").write_text(GAMMA_ORIGINAL)


MUTATIONS = [
    {"name": "leaves alpha.json at its starting state",
     "apply": _alpha_untouched,
     "must_fail": ["alpha.json edited (active=true, retries=7, name kept)"]},
    {"name": "leaves beta.json at its starting state",
     "apply": _beta_untouched,
     "must_fail": ["beta.json edited (tags += y, priority=1, name kept)"]},
    {"name": "leaves gamma.json at its starting state",
     "apply": _gamma_untouched,
     "must_fail": ["gamma.json edited (limits.mem=1024, limits.gpu=T4, cpu+name kept)"]},
]
