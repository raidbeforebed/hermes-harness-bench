#!/usr/bin/env python3
"""Freeze the instrument: the bytes that define every task and its grader.

Why this exists. A manifest records `task_bundle` hashes, which says afterwards that the
ground moved under a measurement. On 2026-11-01 six tasks were added and four `task.json`
files edited between 04:22 and 05:00 while a series was already running: the row had to be
retired and two hours of card time were lost. Recording is not prevention. This module
writes the hash of the task set down as a file, and a run refuses to start when the tree no
longer matches it.

The invariant, borrowed from `agenttest/v2/freeze.py` in notwitcheer/llm-bench-rig (the code
is not borrowed):

  * every file under `tasks/<task>/` is instrument, except editor and interpreter debris.
    That is `task.json`, `prompt.md`, `judge.py`, every fixture under `workspace/`, and any
    other file a judge may read. Hashing the whole directory rather than a named list is
    the point: a fixture that changed and a code file that did not is exactly the drift
    that a score cannot survive.
  * `tasks/FROZEN.sha256` is the `sha256sum` file, one line per instrument file, sorted by
    path, relative to the bench root so it can travel.
  * `tasks/FROZEN.json` is what a reader cites: per task, a tree digest over that task's
    files, a digest of the `task.json` budget fields, and how many files were covered.
  * `freeze` writes both and is a deliberate act. `freeze --check` recomputes both and
    reports what changed, and that is what `run` calls before its first attempt.

Nothing here touches the network, and nothing writes outside `tasks/`.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

FREEZE_SHA = "FROZEN.sha256"
FREEZE_JSON = "FROZEN.json"
SCHEMA = 1
# Prefix of the canonical text the tree digest is taken over. Bumping it is how a future
# change to what a tree digest means is made visible rather than silent.
TREE_PREFIX = "hbench-instrument-tree-v1"

# Directories that are never instrument: interpreter and tool caches are written by running
# something, not by defining a task, and a `.pytest_cache` appearing mid-series must not read
# as drift. Everything else is covered, dotfiles included.
IGNORED_DIRS = frozenset({"__pycache__", ".git", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
IGNORED_SUFFIXES = (".pyc", ".pyo", ".swp", "~")
# `judge.py.bak-20261023-033855` sits in twelve task directories. A rollback copy is not the
# grader, and freezing it would make every freeze noisy without making a run safer.
IGNORED_MARKERS = (".bak",)

# The fields of task.json that decide how much time and how many turns an attempt gets. A
# digest of them travels separately from the file hash, so a reader can see that a re-file
# changed the budget and nothing else.
BUDGET_KEY = re.compile(r"(budget|turn|timeout|limit|scale)", re.I)


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------
# Coverage
# --------------------------------------------------------------------------

def _ignored(rel: PurePosixPath) -> bool:
    for part in rel.parts:
        if part in IGNORED_DIRS:
            return True
        if any(marker in part for marker in IGNORED_MARKERS):
            return True
    name = rel.name
    return name.endswith(IGNORED_SUFFIXES) or name in (FREEZE_SHA, FREEZE_JSON)


def task_dirs(root: Path) -> list[Path]:
    """Every task directory, in name order. Hidden directories are not tasks."""
    tasks = root / "tasks"
    if not tasks.is_dir():
        return []
    return sorted(p for p in tasks.iterdir() if p.is_dir() and not p.name.startswith("."))


def instrument_files(root: Path) -> dict[str, Path]:
    """Relative path -> absolute path, for every file that defines a task or its grader."""
    out: dict[str, Path] = {}
    for task in task_dirs(root):
        for path in sorted(task.rglob("*")):
            if not path.is_file():
                continue
            rel = PurePosixPath(path.relative_to(root).as_posix())
            if _ignored(rel):
                continue
            out[rel.as_posix()] = path
    return out


def digest_map(root: Path) -> dict[str, str]:
    return {rel: sha256_file(path) for rel, path in sorted(instrument_files(root).items())}


def tree_digest(files: dict[str, str]) -> str:
    """One digest over a task's files, so a fixture cannot move without moving the task.

    `files` is relative-path -> file digest, already limited to one task.
    """
    lines = [f"{digest}  {rel}" for rel, digest in sorted(files.items())]
    return sha256_text(TREE_PREFIX + "\n" + "\n".join(lines) + "\n")


def budget_fields(task_json: Path) -> dict:
    """The task.json keys that are a time or turn budget, in canonical order."""
    if not task_json.is_file():
        return {}
    try:
        spec = json.loads(task_json.read_text())
    except (OSError, ValueError):
        return {}
    if not isinstance(spec, dict):
        return {}
    return {k: v for k, v in spec.items() if BUDGET_KEY.search(str(k))}


def budget_digest(task_json: Path) -> str:
    return sha256_text(canonical_json(budget_fields(task_json)))


# --------------------------------------------------------------------------
# Build and write
# --------------------------------------------------------------------------

def build(root: Path) -> tuple[str, dict]:
    """The FROZEN.sha256 text and the FROZEN.json document this tree should have."""
    digests = digest_map(root)
    sha_text = "".join(f"{digest}  {rel}\n" for rel, digest in sorted(digests.items()))

    tasks: dict[str, dict] = {}
    for task in task_dirs(root):
        prefix = f"tasks/{task.name}"
        under = {rel: digest for rel, digest in digests.items()
                 if rel == prefix or rel.startswith(prefix + "/")}
        tasks[task.name] = {
            "tree_sha256": tree_digest(under),
            "budget_sha256": budget_digest(task / "task.json"),
            "files": len(under),
        }
    doc = {
        "schema": SCHEMA,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": "sha256",
        "files": len(digests),
        "tasks": tasks,
        # The digest of the FROZEN.sha256 text. This is the number a run stamps into its
        # manifest, so a published row and this file describe the same instrument.
        "freeze_sha256": sha256_text(sha_text),
    }
    return sha_text, doc


def write(root: Path) -> dict:
    """Write both freeze files. Deliberate: no run ever calls this on its own behalf."""
    sha_text, doc = build(root)
    tasks = root / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    (tasks / FREEZE_SHA).write_text(sha_text)
    (tasks / FREEZE_JSON).write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    return doc


# --------------------------------------------------------------------------
# Check
# --------------------------------------------------------------------------

def read_freeze_sha(text: str) -> dict[str, str]:
    """Parse a `sha256sum` file. A malformed line is kept as an entry with a NULL digest so
    it reads as a change rather than vanishing."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            out[line] = "(malformed line)"
            continue
        out[parts[1].strip()] = parts[0].strip()
    return out


def check(root: Path) -> dict:
    """Recompute both freeze files and report every way the tree differs from them.

    Read-only. No temp directories, no fixture staging, no network: it hashes files and
    compares strings, which is why `local_bench.sh` can call it before it wakes the GPU.
    """
    tasks_dir = root / "tasks"
    sha_path = tasks_dir / FREEZE_SHA
    json_path = tasks_dir / FREEZE_JSON
    current_map = digest_map(root)
    current_sha = sha256_text("".join(f"{d}  {r}\n" for r, d in sorted(current_map.items())))

    report: dict = {
        "frozen": sha_path.is_file(),
        "json_present": json_path.is_file(),
        "ok": False,
        "changed": [],
        "added": [],
        "removed": [],
        "task_changes": [],
        "frozen_sha256": None,
        "current_sha256": current_sha,
        "files": len(current_map),
        "tasks": len({rel.split("/")[1] for rel in current_map}),
        "coverage": sorted(current_map),
    }
    if not report["frozen"]:
        return report

    frozen_bytes = sha_path.read_bytes()
    report["frozen_sha256"] = hashlib.sha256(frozen_bytes).hexdigest()
    frozen_map = read_freeze_sha(frozen_bytes.decode("utf-8", "replace"))

    for rel in sorted(set(frozen_map) | set(current_map)):
        was, now = frozen_map.get(rel), current_map.get(rel)
        if was is None:
            report["added"].append(rel)
        elif now is None:
            report["removed"].append(rel)
        elif was != now:
            report["changed"].append({"path": rel, "frozen": was, "current": now})

    # FROZEN.json: the per-task tree and budget digests, compared to what is on disk now.
    if report["json_present"]:
        try:
            frozen_doc = json.loads(json_path.read_text())
        except (OSError, ValueError) as exc:
            report["task_changes"].append({"task": "-", "field": "document",
                                           "frozen": "(unreadable)", "current": str(exc)})
            frozen_doc = {}
        _current_sha_text, current_doc = build(root)
        if frozen_doc:
            if frozen_doc.get("freeze_sha256") != current_doc["freeze_sha256"]:
                report["task_changes"].append({
                    "task": "-", "field": "freeze_sha256",
                    "frozen": frozen_doc.get("freeze_sha256"),
                    "current": current_doc["freeze_sha256"]})
            frozen_tasks = frozen_doc.get("tasks") or {}
            for name in sorted(set(frozen_tasks) | set(current_doc["tasks"])):
                was = frozen_tasks.get(name) or {}
                now = current_doc["tasks"].get(name) or {}
                for field in ("tree_sha256", "budget_sha256", "files"):
                    if was.get(field) != now.get(field):
                        report["task_changes"].append({"task": name, "field": field,
                                                       "frozen": was.get(field),
                                                       "current": now.get(field)})

    report["ok"] = (not report["changed"] and not report["added"] and not report["removed"]
                    and not report["task_changes"] and report["json_present"])
    return report


def describe(report: dict, refreeze_hint: str = "python3 bin/hbench.py freeze") -> str:
    """What moved, and the two honest responses to it. Neutral: `freeze --check` prints this."""
    if not report["frozen"]:
        return (
            f"no freeze at tasks/{FREEZE_SHA}: this task set has no instrument to measure with.\n"
            "An unfrozen task set is whatever happens to be on disk, and a score taken on it "
            "cannot be compared with anything.\n"
            f"Freeze it deliberately, once, before the run:\n  {refreeze_hint}\n"
            "That writes tasks/FROZEN.sha256 and tasks/FROZEN.json, and every later run cites "
            "the digest. `--allow-drift` starts anyway and stamps the manifest so no reader "
            "can mistake the run for one on a frozen set."
        )

    if report["ok"]:
        return (f"frozen instrument intact: {report['files']} file(s) across "
                f"{report['tasks']} task(s), digest {report['frozen_sha256']}")

    lines = [
        "the task set changed after it was frozen.",
        f"  freeze:  tasks/{FREEZE_SHA}  sha256 {report['frozen_sha256']}",
        f"  current: {report['files']} file(s) across {report['tasks']} task(s)  "
        f"sha256 {report['current_sha256']}",
    ]
    if report["changed"]:
        lines.append(f"changed ({len(report['changed'])}):")
        for item in report["changed"]:
            lines.append(f"  {item['path']}")
            lines.append(f"    frozen  {item['frozen']}")
            lines.append(f"    current {item['current']}")
    if report["added"]:
        lines.append(f"added ({len(report['added'])}):")
        lines += [f"  {path}" for path in report["added"]]
    if report["removed"]:
        lines.append(f"removed ({len(report['removed'])}):")
        lines += [f"  {path}" for path in report["removed"]]
    if not report["json_present"]:
        lines.append(f"missing: tasks/{FREEZE_JSON} (the per-task tree and budget digests)")
    if report["task_changes"]:
        lines.append(f"task digests ({len(report['task_changes'])}):")
        for item in report["task_changes"]:
            where = ("FROZEN.json" if item["task"] == "-" else item["task"])
            lines.append(f"  {where}.{item['field']}: "
                         f"{item['frozen']} -> {item['current']}")
    lines += [
        "A score measured on a moved instrument is not comparable with the series it joins.",
        "Two responses, both deliberate:",
        "  restore the file(s) to what the freeze records, or",
        f"  re-freeze with `{refreeze_hint}`, which rewrites the freeze and makes every run "
        "carrying the old digest visibly older.",
        "`--allow-drift` starts anyway and stamps instrument_drift in the manifest, so the run "
        "reads as one measured on a moved instrument rather than one on a frozen set.",
    ]
    return "\n".join(lines)


def refusal(report: dict, refreeze_hint: str = "python3 bin/hbench.py freeze") -> str:
    """The message a `run` exits with. Same account, prefixed as a refusal."""
    return "refusing to run: " + describe(report, refreeze_hint)
