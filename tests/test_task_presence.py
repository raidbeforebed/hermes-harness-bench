#!/usr/bin/env python3
"""Task-presence test: every file the freeze names must actually be in the tree.

The defect this exists for, found on 2026-11-19 while re-running the README from a clean
clone: `tasks/FROZEN.sha256` listed 105 files over 18 tasks, but the repository held only
twelve task directories, 19 of the twelve it did hold had older bytes, and `.gitignore`'s
`*.log` rule had silently un-published `06/workspace/events.log` and `09/workspace/service.log`.
`freeze --check` reported all of it, but only the hashes: a reader's first command then
failed with "the task set no longer matches" and, further in, a FileNotFoundError on a
fixture that was never committed.

The hash check cannot catch a file that is ignored, never committed or deleted, because a
missing file is not a file to hash. So this test reads `tasks/FROZEN.sha256` as a manifest
and asserts presence separately from content:

  PRESENT  every path named by the freeze exists as a regular file in the tree
  IGNORED  no frozen path is matched by .gitignore, which is what un-publishes it silently
  CONTENT  every present file hashes to the digest the freeze records
  TASKS    every task directory the freeze names exists, and no un-frozen task dir is present

Run it from a checkout before trusting the README, and after any publish:

    python3 tests/test_task_presence.py
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FREEZE = ROOT / "tasks" / "FROZEN.sha256"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_freeze() -> dict[str, str]:
    if not FREEZE.is_file():
        raise SystemExit(f"no freeze at {FREEZE.relative_to(ROOT)}")
    out: dict[str, str] = {}
    for line in FREEZE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise SystemExit(f"malformed line in {FREEZE.name}: {line!r}")
        out[parts[1].strip()] = parts[0].strip()
    return out


def ignored_by_gitignore(rel: str) -> str | None:
    """The .gitignore rule that matches *rel* and ignores it, or None.

    Use git when it is available, because git's own answer is the one that decides whether a
    file is committed. The `!tasks/*/workspace/*.log` negation is why a hand-rolled matcher is
    the wrong tool here; `check-ignore` applies the real precedence. A matching rule that
    starts with `!` is a re-include, which means the file is NOT ignored: `check-ignore -v`
    still prints it (and exits 0), so the leading `!` is the thing that decides.
    """
    if not (ROOT / ".git").is_dir():
        return None
    r = subprocess.run(["git", "check-ignore", "-v", "--no-index", rel],
                       cwd=ROOT, capture_output=True, text=True)
    line = r.stdout.strip()
    if not line:
        return None
    # "<source>:<linenum>:<pattern>\t<path>"; the pattern is the field after the line number.
    pattern = line.split("\t", 1)[0].split(":", 2)[-1]
    return None if pattern.startswith("!") else line


def check(root: Path = ROOT) -> tuple[list[str], int, int]:
    problems: list[str] = []
    frozen = read_freeze()
    tasks_dir = root / "tasks"

    missing = [p for p in frozen if not (root / p).is_file()]
    for p in missing:
        problems.append(f"{p} is named by tasks/FROZEN.sha256 but is not in the tree")

    ignored = []
    for p in frozen:
        rule = ignored_by_gitignore(p)
        if rule:
            ignored.append(f"{p} -> {rule}")
            problems.append(f"{p} is ignored by .gitignore ({rule}); it will never be committed")

    changed = []
    for p, want in frozen.items():
        f = root / p
        if f.is_file():
            got = sha256(f)
            if got != want:
                changed.append(p)
                problems.append(f"{p} does not match the freeze: {got[:12]} against {want[:12]}")

    frozen_tasks = {p.split("/")[1] for p in frozen}
    if tasks_dir.is_dir():
        present = {d.name for d in tasks_dir.iterdir()
                   if d.is_dir() and not d.name.startswith(".") and d.name[:2].isdigit()}
        for name in sorted(frozen_tasks - present):
            problems.append(f"task directory tasks/{name} is frozen but absent")
        for name in sorted(present - frozen_tasks):
            problems.append(f"task directory tasks/{name} is present but not in the freeze")

    return problems, len(frozen), len(frozen_tasks)


def main() -> int:
    if not FREEZE.is_file():
        print(f"FAIL no tasks/FROZEN.sha256 at {FREEZE}")
        return 1
    problems, n_files, n_tasks = check()
    for p in problems:
        print(f"FAIL {p}")
    if problems:
        print(f"\n{len(problems)} TASK PRESENCE PROBLEM(S): a clean clone cannot measure the "
              f"frozen instrument")
        return 1
    print(f"ok   {n_files} frozen file(s) over {n_tasks} task(s) are present, un-ignored "
          f"and match their recorded digests")
    print("\nTASK SET COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
