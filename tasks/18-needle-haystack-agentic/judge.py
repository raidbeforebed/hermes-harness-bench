"""Grader for 18-needle-haystack-agentic.

Capability: reliable retrieval from a long noisy context. A small model tends to
either miss the single embedded code in ~60K tokens of distractor lines, or
invent a plausible-looking one.

The truth is extracted from haystack.log with a regex that matches only the real
code pattern (ERR- followed by four uppercase hex digits); the noise generator
guarantees that pattern appears exactly once. Every check is gated on needle.txt
existing, so an empty workspace scores 0. The primary check is exact match to the
embedded code; a second check rejects an invented code. Deterministic,
code-based, no judge model.
"""
import json
import re
import sys
from pathlib import Path

w = Path(sys.argv[1])
checks = []

CODE_RE = re.compile(r"ERR-[0-9A-F]{4}")

log = w / "haystack.log"
log_text = log.read_text() if log.exists() else ""
log_codes = sorted(set(CODE_RE.findall(log_text)))

p = w / "needle.txt"
exists = p.exists()
reported = p.read_text().strip() if exists else None

truth = log_codes[0] if len(log_codes) == 1 else None

checks.append({
    "check": "needle.txt exists",
    "pass": exists,
    "detail": "ok" if exists else "missing",
})

checks.append({
    "check": "needle.txt is exactly the embedded error code",
    "pass": exists and truth is not None and reported == truth,
    "detail": "reported=%r truth=%r" % (reported, truth),
})

checks.append({
    "check": "no invented code (reported code actually appears in the log)",
    "pass": exists and reported is not None and reported in log_codes,
    "detail": "reported=%r known=%r" % (reported, log_codes),
})

print(json.dumps({"checks": checks}))
