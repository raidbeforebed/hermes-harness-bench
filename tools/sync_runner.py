#!/usr/bin/env python3
"""Sync the runner and the frozen instrument into the published tree, in one command.

    python3 tools/publish.py --sync-runner --message "..."     # sync, commit and push
    python3 tools/publish.py --sync-runner --dry-run           # sync the working tree only

Why this exists. `bin/hbench.py` is the instrument: every published row was measured with
one. It reached the repository by hand the first time (commit 38314c0) and then sat at
1,524 lines while the live runner grew to 2,841, so the published README told a reader to
run something that could not score the published suite. A hand copy is what fails; this
command is the repeatable one, and `tests/test_runner_sync.py` (which `tests/test_publish.py`
runs) fails when the two drift again.

What travels, and why each file:

  bin/hbench.py          the runner
  bin/hbench_*.py        the modules it imports by bare name from its own directory
  bin/pair_report.py     imports hbench_publish; the run-level report tool
  tasks/FROZEN.sha256    the freeze `run` refuses to start without
  tasks/FROZEN.json      the per-task digests a reader cites

`bin/hbench.py` also imports `result_key`, and that one is NOT taken from the live tree: a
bare `result_key.py` does not exist there (the live runner finds it because a `PYTHONPATH`
entry supplies it), so a copy taken from the live tree would be a guess. It is taken from
`tools/result_key.py` in this tree instead, which is where the live runner's own comment
points and where the exporter, the charts and the publisher read the same rule. A published
runner that imported a missing module, or a second copy of the row-identity rule, would fail
on its first command.

The task files themselves are not copied here: `tools/publish.py` lands each run's tree
with its results, and the freeze is exactly what proves those bytes are the ones measured.
A --sync-runner that moved tasks/ under a published result would be the defect this
command exists to prevent, so the freeze is copied and compared, never regenerated.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

LIVE_ROOT = Path(os.environ.get("HBENCH_LIVE_ROOT") or "/root/hermes-bench")

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
# Imported by `bin/hbench.py` but sourced from this tree's tools/, not from the live bin/,
# because the live runner resolves it from a PYTHONPATH entry and there is no bare file to
# copy. The check is one-directional for this module: it must exist in the published bin/.
TOOLS_SOURCED = ("result_key.py",)
FROZEN = ("FROZEN.sha256", "FROZEN.json")


def planned(tree: Path) -> list[tuple[Path, Path]]:
    pairs = [(LIVE_ROOT / "bin" / RUNNER, tree / "bin" / RUNNER)]
    pairs += [(LIVE_ROOT / "bin" / n, tree / "bin" / n) for n in COMPANIONS]
    pairs += [(tree / "tools" / n, tree / "bin" / n) for n in TOOLS_SOURCED]
    pairs += [(LIVE_ROOT / "tasks" / n, tree / "tasks" / n) for n in FROZEN]
    return pairs


def sync(tree: Path, dry_run: bool = False) -> list[str]:
    """Copy live runner files over the published tree. Returns the paths that changed."""
    changed: list[str] = []
    for src, dst in planned(tree):
        if not src.is_file():
            print(f"  skip {src} (the live tree does not carry it)")
            continue
        same = dst.is_file() and src.read_bytes() == dst.read_bytes()
        if same:
            continue
        changed.append(str(dst.relative_to(tree)))
        print(f"  {'would write' if dry_run else 'write'} {dst.relative_to(tree)} "
              f"({len(src.read_text().splitlines())} lines)")
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            shutil.copymode(src, dst)
    return changed


def main(argv: list[str]) -> int:
    tree = Path(__file__).resolve().parent.parent
    dry = "--dry-run" in argv
    if not LIVE_ROOT.is_dir():
        print(f"no live bench tree at {LIVE_ROOT}: nothing to sync from. "
              "Set HBENCH_LIVE_ROOT if it lives elsewhere.", file=sys.stderr)
        return 1
    print(f"live runner tree: {LIVE_ROOT}")
    changed = sync(tree, dry_run=dry)
    if not changed:
        print("already in sync: the runner, its modules and the frozen instrument match live")
        return 0
    print(f"\n{len(changed)} file(s) {'would change' if dry else 'updated'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
