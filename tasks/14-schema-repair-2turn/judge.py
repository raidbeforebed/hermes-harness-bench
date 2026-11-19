"""Grader for 14-schema-repair-2turn.

Capability: recovering from an explicit schema error by emitting a minimal,
schema-valid correction that keeps the original intent (same operation, same
path). A small model typically repeats the malformed structure, truncates again,
adds speculative fields, or changes the path.

Two layers, both deterministic and code-based (no LLM judge):

  * Workspace checks (always run): `corrected_call.json` exists, parses, is a
    single object with exactly `tool`+`args`, `args` has exactly `pattern`+`path`
    as strings, and keeps turn 1's operation, pattern and path.

  * Trace checks (run only when a per-attempt trace is present). The harness
    stages the attempt's tool-call transcript at `.hbench_trace.jsonl` inside
    the work directory (see bin/hbench.py:stage_trace) and injects the rejected
    turn 1 there, read from the pristine `turn1_rejected.json` fixture. The trace
    layer then checks the SPEC's "turn two is a single valid call" property:
    the rejection is present as turn 1, the model emitted exactly one call that
    produces the correction, that call succeeded, and it did so in a single
    call rather than a retry/spray of attempts. The "same path as turn 1"
    requirement is enforced by the workspace checks on the artifact itself.

Backward compatibility: when there is no trace file (e.g. the judge is invoked
directly as `judge.py <workdir>` by tests/test_graders.py), only the workspace
checks run, so an existing caller is unaffected. Every check is gated on the
artifact/correction existing, so an empty workspace scores 0.
"""
import json
import sys
from pathlib import Path

w = Path(sys.argv[1])
checks = []

p = w / "corrected_call.json"
exists = p.exists()

checks.append({
    "check": "corrected_call.json exists",
    "pass": exists,
    "detail": "ok" if exists else "missing",
})

data = None
if exists:
    try:
        data = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        checks.append({"check": "valid JSON", "pass": False, "detail": str(exc)[:160]})

is_obj = isinstance(data, dict)

checks.append({
    "check": "single object call, not a list or wrapper",
    "pass": exists and is_obj,
    "detail": ("ok" if exists and is_obj
               else (type(data).__name__ if data is not None else "missing/invalid")),
})

if is_obj:
    keys = set(data.keys())
    args = data.get("args")
    args_keys = set(args.keys()) if isinstance(args, dict) else set()

    checks.append({
        "check": "exact top-level keys tool+args, no extras",
        "pass": exists and keys == {"tool", "args"},
        "detail": "keys=" + str(sorted(keys)),
    })
    checks.append({
        "check": "args has exactly pattern+path with string types",
        "pass": (exists and isinstance(args, dict) and args_keys == {"pattern", "path"}
                 and isinstance(args.get("pattern"), str)
                 and isinstance(args.get("path"), str)),
        "detail": "args=" + str(args),
    })
    checks.append({
        "check": "same operation, pattern and path as turn 1",
        "pass": (exists and data.get("tool") == "search_files"
                 and isinstance(args, dict)
                 and args.get("pattern") == "def calculate_total"
                 and args.get("path") == "/srv/payments/charge.py"),
        "detail": "tool=%r args=%r" % (data.get("tool"), args),
    })


# --------------------------------------------------------------------------
# Trace layer: only when the harness staged a per-attempt trace.
# --------------------------------------------------------------------------

def _read_trace(work):
    """The staged trace as a list of records, or None when none was staged."""
    path = work / ".hbench_trace.jsonl"
    if not path.exists():
        return None
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:  # noqa: BLE001
            pass
    return records


def _input_blob(rec):
    try:
        return json.dumps(rec.get("input"), sort_keys=True).lower()
    except Exception:  # noqa: BLE001
        return ""


def _is_correction_call(rec):
    """A tool_use that IS the corrected call: a re-emitted search_files call, or
    a call that writes the corrected_call.json artifact."""
    if "corrected_call.json" in _input_blob(rec):
        return True
    inp = rec.get("input")
    return (rec.get("name") == "search_files" and isinstance(inp, dict)
            and "pattern" in inp and "path" in inp)


def _result_for(use, results):
    """The tool_result that answers *use*, by call id first, then by name+turn."""
    cid = use.get("tool_call_id")
    if cid is not None:
        for res in results:
            if res.get("tool_call_id") == cid:
                return res
    for res in results:
        if (res.get("name") == use.get("name")
                and res.get("turn") == use.get("turn")):
            return res
    return None


trace = _read_trace(w)
if trace is not None:
    uses = [r for r in trace if r.get("kind") == "tool_use"]
    results = [r for r in trace if r.get("kind") == "tool_result"]
    injected = [r for r in trace if r.get("kind") == "injected_rejection"]

    checks.append({
        "check": "trace: rejected turn 1 is present (injected by the harness)",
        "pass": any(r.get("turn") == 1 for r in injected),
        "detail": "injected turns=" + str([r.get("turn") for r in injected]),
    })

    corrections = [r for r in uses if _is_correction_call(r)]
    checks.append({
        "check": "trace: exactly one correction call was emitted",
        "pass": len(corrections) == 1,
        "detail": "correction calls=" + str(len(corrections)),
    })

    if len(corrections) == 1:
        corr = corrections[0]
        res = _result_for(corr, results)
        checks.append({
            "check": "trace: the correction call succeeded",
            "pass": res is not None and res.get("is_error") is False,
            "detail": "result is_error=" + str(None if res is None else res.get("is_error")),
        })
        same_turn = [r for r in uses if r.get("turn") == corr.get("turn")]
        checks.append({
            "check": "trace: the correction was a single call in its turn",
            "pass": len(same_turn) == 1,
            "detail": "calls in correction turn=" + str(len(same_turn)),
        })
    else:
        # Keep the check count stable regardless of branch (deterministic output).
        checks.append({"check": "trace: the correction call succeeded", "pass": False,
                       "detail": "no single correction call to test"})
        checks.append({"check": "trace: the correction was a single call in its turn",
                       "pass": False, "detail": "no single correction call to test"})

print(json.dumps({"checks": checks}))
