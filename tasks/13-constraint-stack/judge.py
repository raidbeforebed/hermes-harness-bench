"""Grader for 13-constraint-stack.

Capability: holding several mutually-constraining rules at once. The task is an
edit to main.py under four simultaneous constraints (use utils.debug_log, do not
import logging, do not use print, do not touch config.py). Every check below is
gated on the primary achievement (`main.py calls utils.debug_log`), so an
untouched workspace scores 0 and only a solution that actually adds the logging
can score any point. Scoring is deterministic and code-based; there is no judge
model.
"""
import json
import re
import sys
import hashlib
import ast
from pathlib import Path

w = Path(sys.argv[1])
checks = []

main_p = w / "main.py"
cfg_p = w / "config.py"

main_src = main_p.read_text() if main_p.exists() else ""

# SHA-256 of the pristine config.py. "Untouched" is compared against this so the
# fixture can never silently drift out of sync with the judge.
PRISTINE_CONFIG_SHA256 = "928afbd7a5c76c8af97c1e893b5bd8e284c0e85de18b872a9644b801bf714320"


def strip_comments(src: str) -> str:
    """Drop # comments line by line so a commented-out call does not count."""
    out = []
    for ln in src.splitlines():
        i = ln.find("#")
        out.append(ln[:i] if i >= 0 else ln)
    return "\n".join(out)


clean = strip_comments(main_src)

# The achievement gate: a live call to utils.debug_log(...).
added = bool(re.search(r"utils\s*\.\s*debug_log\s*\(", clean))

import_logging = bool(re.search(r"^\s*(?:import\s+logging|from\s+logging\b)", clean, re.M))
uses_print = "print(" in clean

try:
    ast.parse(main_src)
    parse_ok = True
    parse_err = ""
except Exception as exc:  # noqa: BLE001 - any parse failure is a failure
    parse_ok = False
    parse_err = str(exc)

checks.append({
    "check": "main.py calls utils.debug_log",
    "pass": added,
    "detail": "ok" if added else "no utils.debug_log( call found in main.py",
})

checks.append({
    "check": "did not import the logging module",
    "pass": added and not import_logging,
    "detail": ("no debug_log call added" if not added
               else ("imports logging" if import_logging else "ok")),
})

checks.append({
    "check": "did not use print in main.py",
    "pass": added and not uses_print,
    "detail": ("no debug_log call added" if not added
               else ("print( still present" if uses_print else "ok")),
})

config_untouched = cfg_p.exists() and hashlib.sha256(cfg_p.read_bytes()).hexdigest() == PRISTINE_CONFIG_SHA256
checks.append({
    "check": "config.py unchanged",
    "pass": added and config_untouched,
    "detail": ("no debug_log call added" if not added
               else ("ok" if config_untouched else ("config.py edited or missing"))),
})

checks.append({
    "check": "main.py still parses as valid Python",
    "pass": added and parse_ok,
    "detail": ("no debug_log call added" if not added
               else ("ok" if parse_ok else "syntax error: " + parse_err[:120])),
})

print(json.dumps({"checks": checks}))
