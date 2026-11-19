"""Grader for 16-stateful-undo.

Capability: tracking an edit sequence with a mid-sequence undo. The model is
told to create a file, append two lines, revert to the state after the FIRST
append (so the second appended line must disappear but the first must survive),
then append a third line. A small model typically reverts to the wrong
checkpoint: it drops line A too, or fails to drop line B, or appends C to the
wrong base.

Every check is gated on notes.md existing, so an empty workspace scores 0. The
primary check is an exact-content match ("check by exact diff" per the SPEC).
Deterministic, code-based, no judge model.
"""
import json
import sys
from pathlib import Path

w = Path(sys.argv[1])
checks = []

p = w / "notes.md"
exists = p.exists()
content = p.read_text() if exists else ""

EXPECTED = "# Notes\n- alpha\n- charlie\n"

checks.append({
    "check": "notes.md exists",
    "pass": exists,
    "detail": "ok" if exists else "missing",
})

lines = content.splitlines()
checks.append({
    "check": "header is the first line",
    "pass": exists and lines[:1] == ["# Notes"],
    "detail": repr(lines[0]) if lines else "missing",
})

# The undo semantics made explicit: A survived the revert, B did not, C was added.
checks.append({
    "check": "undo honoured: alpha kept, bravo dropped, charlie appended",
    "pass": exists and ("- alpha" in content) and ("- bravo" not in content)
            and ("- charlie" in content),
    "detail": ("ok" if exists else "missing"),
})

checks.append({
    "check": "exact final content (header, alpha, charlie, trailing newline)",
    "pass": exists and content == EXPECTED,
    "detail": repr(content) if exists else "missing",
})

print(json.dumps({"checks": checks}))
