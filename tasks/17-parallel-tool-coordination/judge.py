"""Grader for 17-parallel-tool-coordination.

Capability: carrying several independent edits to completion together, emitted
in one turn. A small model tends to do part of the set and drop or botch the
rest, or to serialize the edits across turns.

Two layers, both deterministic and code-based (no LLM judge):

  * Workspace checks (always run): each of alpha.json, beta.json, gamma.json
    parses to its exact target object (dict equality, so key order and
    whitespace do not matter).

  * Trace checks (run only when a per-attempt trace is present). The harness
    stages the attempt's tool-call transcript at `.hbench_trace.jsonl` inside
    the work directory (see bin/hbench.py:stage_trace). The trace layer then
    checks the SPEC's "the single turn contains three valid calls" property:
    some one turn issued the calls that cover all three files, and those calls
    all succeeded. A model that edits the three files correctly but spreads the
    work over three turns fails the trace layer, which is the batching signal.

Backward compatibility: when there is no trace file (e.g. the judge is invoked
directly as `judge.py <workdir>` by tests/test_graders.py), only the workspace
checks run, so an existing caller is unaffected.
"""
import json
import sys
from pathlib import Path

w = Path(sys.argv[1])
checks = []


def load(name):
    p = w / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return None


exp_alpha = {"name": "alpha", "active": True, "retries": 7}
exp_beta = {"name": "beta", "tags": ["x", "y"], "priority": 1}
exp_gamma = {"name": "gamma", "limits": {"cpu": 1, "mem": 1024, "gpu": "T4"}}

got_alpha = load("alpha.json")
got_beta = load("beta.json")
got_gamma = load("gamma.json")

checks.append({
    "check": "alpha.json edited (active=true, retries=7, name kept)",
    "pass": got_alpha == exp_alpha,
    "detail": "got " + repr(got_alpha),
})
checks.append({
    "check": "beta.json edited (tags += y, priority=1, name kept)",
    "pass": got_beta == exp_beta,
    "detail": "got " + repr(got_beta),
})
checks.append({
    "check": "gamma.json edited (limits.mem=1024, limits.gpu=T4, cpu+name kept)",
    "pass": got_gamma == exp_gamma,
    "detail": "got " + repr(got_gamma),
})


# --------------------------------------------------------------------------
# Trace layer: only when the harness staged a per-attempt trace.
# --------------------------------------------------------------------------

FILES = ("alpha.json", "beta.json", "gamma.json")


def _read_trace(work):
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


def _refs(rec):
    """Which of the three target files a call's input mentions."""
    try:
        blob = json.dumps(rec.get("input") or {}, sort_keys=True).lower()
    except Exception:  # noqa: BLE001
        return set()
    return {f for f in FILES if f in blob}


def _result_for(use, results):
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

    by_turn = {}
    for r in uses:
        by_turn.setdefault(r.get("turn"), []).append(r)

    # The batching turn: has at least three calls, and those calls' inputs
    # collectively cover all three files.
    batch = None
    for turn in sorted(by_turn):
        calls = by_turn[turn]
        covered = set()
        for c in calls:
            covered |= _refs(c)
        if len(calls) >= 3 and covered >= set(FILES):
            batch = calls
            break

    checks.append({
        "check": "trace: one turn issued calls covering all three files",
        "pass": batch is not None,
        "detail": ("turn " + str(batch[0]["turn"]) + " with " + str(len(batch)) + " calls")
                  if batch else "no single turn covered all three files",
    })

    if batch is not None:
        ok = True
        failed = []
        for f in FILES:
            targets = [c for c in batch if f in _refs(c)]
            good = False
            for c in targets:
                res = _result_for(c, results)
                if res is not None and res.get("is_error") is False:
                    good = True
                    break
            if not good:
                ok = False
                failed.append(f)
        checks.append({
            "check": "trace: the three batched edits all succeeded",
            "pass": ok,
            "detail": "all three non-error" if ok else "failed/errored: " + ",".join(failed),
        })
    else:
        checks.append({"check": "trace: the three batched edits all succeeded",
                       "pass": False, "detail": "no qualifying turn"})

print(json.dumps({"checks": checks}))
