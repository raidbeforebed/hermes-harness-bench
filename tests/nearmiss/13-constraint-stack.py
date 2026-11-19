"""Near miss for 13-constraint-stack: a plausible wrong answer, per mutation.

Each mutation is applied on top of the reference solution by
`tests/test_graders.py`. `must_fail` names the judge checks that must fail;
`also_fail` names any other check that legitimately fails as a dependent of
those. The proof fails when a mutation passes, when a named check does not
fail, or when a check outside the two lists fails.
"""

def _imports_logging_as_well(w):
    """Keeps the project logger but also reaches for the standard module."""
    p = w / "main.py"
    p.write_text("import logging\n" + p.read_text())


def _keeps_the_print_statement(w):
    """Adds the required call but leaves the old print in the loop."""
    p = w / "main.py"
    p.write_text(p.read_text().rstrip("\n")
                 + '\n    print(f"processed {len(items)} items")\n')


def _edits_config_while_passing_a_value(w):
    """Changes a module it was told not to touch, to carry a setting."""
    p = w / "config.py"
    p.write_text(p.read_text().rstrip("\n") + "\nDEBUG = True\n")


MUTATIONS = [
    {"name": "imports the logging module alongside the project logger",
     "apply": _imports_logging_as_well,
     "must_fail": ["did not import the logging module"]},
    {"name": "keeps print in main.py after adding the logger call",
     "apply": _keeps_the_print_statement,
     "must_fail": ["did not use print in main.py"]},
    {"name": "edits config.py to expose a debug flag",
     "apply": _edits_config_while_passing_a_value,
     "must_fail": ["config.py unchanged"]},
]
