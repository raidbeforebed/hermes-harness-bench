#!/usr/bin/env python3
"""Runner-integrity test: the published runner must be the runner that produced the results.

The defect this exists for, found on 2026-11-19: publish/tree/bin/hbench.py was 1,524 lines
and predated the per-attempt tool-call trace staging, the quality gates, the reproducibility
block, the frozen-instrument fence and `merge`, against 2,841 live. A reader who followed
README.md's own instructions got a runner that could not score the published suite, and a
measured probe proved it: task 01 scored 0.7143 under the published copy and 1.00 under the
live one, because the published copy staged no trace for the grader.

Nothing failed. A stale file in a published tree is invisible to every other test, so this
test reads the two runners as bytes and fails when the published copy is not the live one.
It also checks the plumbing that keeps them equal, because a one-off copy is what produced
this defect the first time (commit 38314c0 synced it by hand).

Run it after touching bin/, before any publish, and from a checkout before trusting the
README:

    python3 tests/test_publish.py     # checks the surfaces, including this
"""

from __future__ import annotations

import hashlib
import os

import subprocess
import sys
from pathlib import Path

# The live bench root, where the runner that measures and the tasks it measures live. A
# published checkout has no such tree, so the comparison is skipped by name rather than
# passing silently: the skip is printed and says what was not checked.
LIVE_ROOT = Path(os.environ.get("HBENCH_LIVE_ROOT") or "/root/hermes-bench")

# The runner and every module it imports by bare name from its own directory. The published
# runner is the live runner only if the whole set matches: a new helper that did not travel
# leaves the runner refusing to start (ModuleNotFoundError) on the very first command.
RUNNER = "hbench.py"
COMPANIONS = (
    "hbench_coresmoke.py",
    "hbench_freeze.py",
    "hbench_metrics.py",
    "hbench_publish.py",
    "hbench_quality_gates.py",
    "hbench_uncertainty.py",
    "hbench_spread.py",
    "pair_report.py",
)
# Imported by the runner, sourced from tools/, so presence is the whole check for it.
TOOLS_SOURCED = ("result_key.py",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(pub_root: Path) -> list[str]:
    """Every way the published runner can fail to be the live one. Empty means equal."""
    problems: list[str] = []
    live_bin, pub_bin = LIVE_ROOT / "bin", pub_root / "bin"

    if not LIVE_ROOT.is_dir():
        return []          # a published checkout: nothing local to compare against
    if not live_bin.is_dir():
        return [f"{LIVE_ROOT} exists but has no bin/, so there is no live runner to compare"]

    # 1. the runner itself, as bytes
    live, pub = live_bin / RUNNER, pub_bin / RUNNER
    if not pub.is_file():
        problems.append(f"bin/{RUNNER} is missing from the published tree")
    elif sha256(live) != sha256(pub):
        problems.append(
            f"bin/{RUNNER} is not the live runner: published {sha256(pub)[:12]} "
            f"({len(pub.read_text().splitlines())} lines) against live {sha256(live)[:12]} "
            f"({len(live.read_text().splitlines())} lines). "
            "Run `python3 tools/publish.py --sync-runner` and commit the result.")

    # 2. the modules it imports, which travel with it or not at all
    for name in COMPANIONS:
        lv, pv = live_bin / name, pub_bin / name
        if not lv.is_file():
            continue       # the live tree does not carry it; not a publication problem
        if not pv.is_file():
            problems.append(f"bin/{name} is missing from the published tree; "
                            f"bin/{RUNNER} imports it by bare name")
        elif sha256(lv) != sha256(pv):
            problems.append(f"bin/{name} is not the live module: "
                            f"published {sha256(pv)[:12]} against live {sha256(lv)[:12]}")

    # 3. the modules it imports that do not come from the live bin/, which must at least exist
    for name in TOOLS_SOURCED:
        if not (pub_bin / name).is_file():
            problems.append(f"bin/{name} is missing from the published tree; "
                            f"bin/{RUNNER} imports it by bare name and would not start")

    # 4. the runner starts. A byte-identical copy is necessary but the reader's first
    #    command is what actually has to work, so it is run: `list` is offline, reads only
    #    tasks/, and touches no model, no lock and no network.
    if not problems and (pub_bin / RUNNER).is_file():
        r = subprocess.run([sys.executable, str(pub_bin / RUNNER), "list"],
                           cwd=pub_root, capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            problems.append(f"bin/{RUNNER} list exits {r.returncode}: "
                            f"{(r.stderr or r.stdout).strip().splitlines()[-1][:200]}")

    # 5. the freeze the runner enforces, by content not by timestamp: a published copy of
    #    FROZEN.sha256 that names files the published tasks/ do not have would refuse every
    #    run with "the task set no longer matches" for a reason the reader cannot see.
    for name in ("FROZEN.sha256", "FROZEN.json"):
        lv, pv = LIVE_ROOT / "tasks" / name, pub_root / "tasks" / name
        if not lv.is_file():
            continue
        if not pv.is_file():
            problems.append(f"tasks/{name} is missing from the published tree; the runner "
                            "refuses to start without a frozen instrument")
        elif sha256(lv) != sha256(pv):
            problems.append(f"tasks/{name} is not the live freeze")
    return problems


def main(pub_root: Path | None = None) -> int:
    pub_root = pub_root or Path(__file__).resolve().parent.parent
    problems = check(pub_root)
    if not LIVE_ROOT.is_dir():
        print(f"skip no live bench tree at {LIVE_ROOT} (published-only checkout); "
              f"the runner comparison cannot run and is not claimed to have run")
        return 0
    for p in problems:
        print(f"FAIL {p}")
    if problems:
        print(f"\n{len(problems)} RUNNER PROBLEM(S): the published runner cannot reproduce "
              f"the published results")
        return 1
    pub = pub_root / "bin" / RUNNER
    print(f"ok   bin/{RUNNER} is the live runner byte for byte, {len(pub.read_text().splitlines())} lines, "
          f"sha256 {sha256(pub)[:12]}")
    print(f"ok   {len(COMPANIONS)} companion module(s) and the frozen instrument match live")
    print("\nRUNNER IN SYNC")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
