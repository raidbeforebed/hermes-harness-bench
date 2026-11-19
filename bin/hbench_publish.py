#!/usr/bin/env python3
"""Publication gate for hbench results: reproducibility metadata as an invariant.

The defect this closes
----------------------
A three-repeat series was published where one repeat had run under an unknown
chat-template setting, and no run in that series recorded a sampling temperature at all.
`manifest.json` carried the server name, the context size and the KV cache type, and
`bench_provenance.py` added the variant and the chat-template flags to runs started
through `local_bench.sh`. But every one of those fields was optional: a run started by
calling `bin/hbench.py` directly recorded `"notes": {}` and nothing complained, and the
result went to `publish/tree/results/` because publishing was a `cp`.

Recording is not enforcing. A manifest that records a build does not stop a series from
spanning two builds, so this module turns the recorded fields into a checked invariant:

  * a required-field schema for the run manifest and for each published attempt record,
  * `validate_manifest` / `validate_record`, which report every absent or empty field
    rather than the first one,
  * `instrument_fingerprint`, a hash over every run-affecting field of a manifest with
    the volatile ones (run id, timestamps, lock) removed, so two runs of one instrument
    produce one fingerprint,
  * `series_conflicts`, which fails a publication when a run's instrument disagrees with
    an already-published run of the same model on any recorded field.

`hbench.py run` calls `validate_manifest` before its first model call and exits when the
schema is incomplete, and `hbench.py publish` calls the whole set before it copies a
single file. Both are hard failures: a warning is what the old behaviour amounted to.

Rules the schema encodes
------------------------
- A required field is absent when the key is missing, when the value is an empty
  string/list/dict, or when it carries a sentinel such as `(unset)` or `unknown`.
- A `null` is only allowed where the schema says so, and only when the reason is itself
  recorded: `temperature: null` is legal while `reproducibility.sampler.omitted`
  contains `"temperature"`, because "the client sends no temperature and the provider
  applies its own default" is a fact, while a missing key is an unknown.
- A field whose recorded source says `unresolved` is present but not publishable.

Run it directly for a one-off check of an archived run:

    python3 bin/hbench_publish.py check <run_id>
    python3 bin/hbench_publish.py check --manifest path/to/manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1

# The tree this file is in, so a clone measures its own tasks and writes its own runs. A
# hard-coded default silently measured this workstation's tree from a clone two directories
# away and wrote the results outside the checkout, which made a clean-clone acceptance run
# evidence about the wrong instrument. HBENCH_ROOT still overrides it deliberately.
ROOT = Path(os.environ.get("HBENCH_ROOT") or Path(__file__).resolve().parent.parent)
# Where the run being published is read from. Normally the tree under ROOT, but a series can be
# measured under a different HBENCH_ROOT (the local-model lanes run from /mnt/fast/hbench-local
# so their work trees sit on the NVMe). Publication is always into this gate's own tree, which
# is a git repository with the site in it, so the read side is the only thing that moves.
RUNS_DIR = Path(os.environ.get("HBENCH_RUNS_DIR") or (ROOT / "runs"))
PUBLISH_TREE = ROOT / "publish" / "tree"
PUBLISHED_RUNS = PUBLISH_TREE / "runs"          # manifest store the records reference
PUBLISHED_RESULTS = PUBLISH_TREE / "results"    # leaderboard input, one file per attempt
SENTINELS = {"(unset)", "(unknown)", "unknown", "n/a", "tbd", "todo", "-"}

# The published identity of a result is model plus configuration, not model alone. The rule
# lives in exactly one module, shared with tools/export_results.py and every reader, because
# two copies of it is how one model came to occupy two file sets on 2026-11-01. Resolved from
# this file rather than from HBENCH_ROOT, which a test may point at a scratch tree.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "publish" / "tree" / "tools"))
import result_key  # noqa: E402

# Fields whose value changes between two runs of the same instrument, so they must not
# enter the fingerprint. Everything else in the manifest is run-affecting by definition.
# `grader_proof` carries the clock reading at which the proof ran (`stamp`) and its duration.
# The proof is a pass/fail precondition: a run whose proof failed does not start, and a run
# that skipped it says so. Two members of one series therefore differ in that block while
# nothing about the instrument changed, so it is a measurement of the run like `lock`.
VOLATILE = {"run_id", "started", "lock", "grader_proof"}

# Inside `decision`, the keys that are measurements of the machine rather than configuration of
# the run. Excluded from the instrument fingerprint and from the series comparison, because a
# measurement moves with the machine and the invariant is about the instrument.
MEASURED_DECISION_KEYS = ("throughput_probe",)


@dataclass(frozen=True)
class Field:
    """One required key, addressed by dotted path."""
    path: str
    kind: str                     # str | num | int | bool | list | list_allow_empty | dict
    why: str
    # null is legal only when the field's own name appears in the list at this path
    nullable_in: str | None = None


# --------------------------------------------------------------------------
# The manifest schema
# --------------------------------------------------------------------------

MANIFEST_SCHEMA: tuple[Field, ...] = (
    # --- run identity ------------------------------------------------------
    Field("run_id", "str", "the name every published file is filed under"),
    Field("started", "str", "when the run began"),
    Field("provider", "str", "which provider resolved the model"),
    Field("models", "list", "which model(s) were measured"),
    Field("tasks", "list", "which task keys this run covers"),
    Field("repeat", "int", "attempts per task"),
    Field("budget_scale", "num", "wall-clock budget multiplier, which is part of the measurement"),
    Field("timeout_scale", "num", "hard subprocess timeout multiplier"),
    Field("tasks_dir", "str", "where the task bundle was read from"),

    # --- isolation ---------------------------------------------------------
    Field("isolation.mode", "str", "how the child was isolated"),
    Field("isolation.user_config", "str", "proves user config was ignored"),
    Field("isolation.rules", "str", "proves rules and memory were not injected"),
    Field("isolation.provider", "str", "where the provider block came from"),
    Field("isolation.network", "str", "the network state the child ran under"),
    Field("isolation.toolsets", "list", "the tool set the child had"),
    Field("isolation.cwd", "str", "the working directory the child ran in"),

    # --- reproducibility: the block this module exists for ------------------
    Field("reproducibility.schema_version", "int", "which schema this manifest satisfies"),

    Field("reproducibility.agent.hermes_version", "str", "the harness that drove the run"),
    Field("reproducibility.agent.install_dir", "str", "which checkout that version came from"),
    Field("reproducibility.agent.cli_flags", "list", "the exact child argv, credentials redacted"),
    Field("reproducibility.agent.toolsets", "list", "tools enabled for the child"),
    Field("reproducibility.agent.reasoning", "str", "reasoning level",
          nullable_in="reproducibility.agent.omitted"),
    Field("reproducibility.agent.reasoning_source", "str",
          "whether the level was set explicitly or left to the child default"),
    Field("reproducibility.agent.omitted", "list_allow_empty",
          "what the agent block could not determine, named rather than silently absent"),
    Field("reproducibility.agent.rules_disabled", "str", "proves no AGENTS.md/SOUL.md/memory"),
    Field("reproducibility.agent.user_config_disabled", "str", "proves no config.yaml"),

    # Chat template: the field that was missing when a series silently mixed configurations.
    Field("reproducibility.chat_template.name", "str", "which template the model was fed"),
    Field("reproducibility.chat_template.sha256", "str",
          "the hash of the template text, so an edited template cannot pass as the same one",
          nullable_in="reproducibility.chat_template.omitted"),
    Field("reproducibility.chat_template.source", "str",
          "where the template is decided: llama-server /props, or the provider"),
    Field("reproducibility.chat_template.kwargs", "str",
          "the --chat-template-kwargs JSON, or an explicit statement that none were passed"),
    Field("reproducibility.chat_template.omitted", "list_allow_empty",
          "what the template block could not determine"),

    # Sampling: temperature was absent from every row of the shipped series.
    Field("reproducibility.sampler.temperature", "num",
          "sampling temperature actually applied",
          nullable_in="reproducibility.sampler.omitted"),
    Field("reproducibility.sampler.temperature_source", "str",
          "how the temperature was resolved, so a default cannot pass as a measurement"),
    Field("reproducibility.sampler.chain", "dict", "the resolved sampler chain, not just temperature"),
    Field("reproducibility.sampler.seed", "num", "the seed the server reports",
          nullable_in="reproducibility.sampler.omitted"),
    Field("reproducibility.sampler.omitted", "list_allow_empty",
          "sampler parameters the client does not send, named rather than silently absent"),

    # The server, when this run drove one.
    Field("reproducibility.server.management", "str", "whether this run started the server"),
    Field("reproducibility.server.argv", "str", "the full resolved command line",
          nullable_in="reproducibility.server.omitted"),
    Field("reproducibility.server.build", "str", "llama.cpp version, build number and commit",
          nullable_in="reproducibility.server.omitted"),
    Field("reproducibility.server.variant", "str", "the flag set the server was started with",
          nullable_in="reproducibility.server.omitted"),
    Field("reproducibility.server.ctx", "int", "context the server opened",
          nullable_in="reproducibility.server.omitted"),
    Field("reproducibility.server.kv_cache_k", "str", "KV cache type for K",
          nullable_in="reproducibility.server.omitted"),
    Field("reproducibility.server.kv_cache_v", "str", "KV cache type for V",
          nullable_in="reproducibility.server.omitted"),
    Field("reproducibility.server.omitted", "list_allow_empty",
          "server facts this run does not have, for example because it drove no server"),

    # The model.
    Field("reproducibility.model.identity", "str", "exact id or path that was requested"),
    Field("reproducibility.model.source", "str", "file or provider"),
    Field("reproducibility.model.sha256", "str", "the file hash, so two builds cannot share a row",
          nullable_in="reproducibility.model.omitted"),
    Field("reproducibility.model.architecture", "str", "architecture, when readable",
          nullable_in="reproducibility.model.omitted"),
    Field("reproducibility.model.parameter_count", "str", "parameter count, when readable",
          nullable_in="reproducibility.model.omitted"),
    Field("reproducibility.model.quantization", "str", "quantization type and mix",
          nullable_in="reproducibility.model.omitted"),
    Field("reproducibility.model.omitted", "list_allow_empty",
          "model facts this run does not have, named rather than guessed"),

    # Environment and task bundle.
    Field("reproducibility.environment.os", "str", "the platform the harness ran on"),
    Field("reproducibility.environment.python", "str", "the interpreter"),
    Field("reproducibility.environment.machine", "str", "host and kernel"),
    Field("reproducibility.task_bundle", "dict",
          "the hash per task of prompt.md, task.json, judge.py and fixtures"),
)

# Per-attempt record. A record is what reaches the leaderboard, and one that cannot name
# the manifest behind it cannot be checked against its siblings.
RECORD_SCHEMA: tuple[Field, ...] = (
    Field("run_id", "str", "which run produced this attempt"),
    Field("manifest_sha256", "str", "the manifest this attempt was produced under"),
    Field("instrument_fingerprint", "str", "the fingerprint of the instrument, shared by siblings"),
    Field("task", "str", "which task"),
    Field("model", "str", "which model"),
    Field("repeat", "int", "which attempt"),
    Field("started", "str", "when this attempt began"),
    Field("finished", "str", "when this attempt ended"),
    Field("wall_seconds", "num", "how long it took"),
    Field("score", "num", "the score"),
    Field("checks", "list", "the grader's per-check output"),
    Field("score_source", "str", "artifact, or an exclusion"),
)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Problem:
    field: str
    message: str

    def __str__(self) -> str:
        return f"  {self.field}: {self.message}"


def _dig(obj: Any, path: str, default: Any = KeyError) -> Any:
    cur = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default if default is not KeyError else KeyError
        cur = cur[part]
    return cur


def _bad_value(value: Any) -> str | None:
    """Why *value* does not count as recorded, or None when it does."""
    if value is None:
        return "null"
    if isinstance(value, str):
        if not value.strip():
            return "empty string"
        if value.strip().lower() in SENTINELS:
            return f"placeholder {value!r}"
        if value.strip().lower().startswith("unresolved"):
            return f"explicitly unresolved ({value.strip()[:80]})"
        return None
    if isinstance(value, (list, tuple, dict, set)) and not value:
        return "empty container"
    return None


def _kind_ok(value: Any, kind: str) -> bool:
    if kind == "str":
        return isinstance(value, str)
    if kind == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "num":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == "bool":
        return isinstance(value, bool)
    if kind in ("list", "list_allow_empty"):
        return isinstance(value, (list, tuple))
    if kind == "dict":
        return isinstance(value, dict)
    raise ValueError(f"unknown kind {kind!r}")


def _validate(obj: Any, schema: Iterable[Field], what: str) -> list[Problem]:
    problems: list[Problem] = []
    for f in schema:
        value = _dig(obj, f.path)
        allow_empty = f.kind.endswith("_allow_empty")
        if value is KeyError:
            problems.append(Problem(f.path, f"absent. It records {f.why}, so a run without "
                                            f"it cannot be reproduced or compared."))
            continue
        bad = _bad_value(value)
        if bad == "empty container" and allow_empty:
            bad = None
        if bad and value is not None:
            problems.append(Problem(f.path, f"{bad}. It records {f.why}."))
            continue
        if value is None:
            if f.nullable_in:
                allowed = _dig(obj, f.nullable_in, None)
                name = f.path.rsplit(".", 1)[-1]
                if not isinstance(allowed, (list, tuple)) or name not in allowed:
                    problems.append(Problem(
                        f.path, f"null, and {f.nullable_in} does not list '{name}'. A null is "
                                f"only meaningful when the omission is itself recorded."))
                continue
            problems.append(Problem(f.path, f"null. It records {f.why}, so it must carry a value."))
            continue
        if not _kind_ok(value, f.kind):
            problems.append(Problem(f.path, f"{type(value).__name__} but the schema wants {f.kind}."))
    return problems


def validate_manifest(manifest: dict) -> list[Problem]:
    """Every required field the manifest is missing or mis-records."""
    problems = _validate(manifest, MANIFEST_SCHEMA, "manifest")
    version = _dig(manifest, "reproducibility.schema_version", None)
    if isinstance(version, int) and version != SCHEMA_VERSION:
        problems.append(Problem(
            "reproducibility.schema_version",
            f"{version} but this validator implements {SCHEMA_VERSION}. Re-check the run "
            f"against the schema it was written under before publishing it as current."))
    return problems


def validate_record(record: dict) -> list[Problem]:
    return _validate(record, RECORD_SCHEMA, "record")


def format_problems(problems: list[Problem], subject: str) -> str:
    if not problems:
        return ""
    lines = [f"REPRODUCIBILITY SCHEMA VIOLATION in {subject}: "
             f"{len(problems)} required field(s) missing or unusable."]
    lines += [str(p) for p in problems]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------

def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_sha256(manifest: dict) -> str:
    """Hash of the manifest as written. Stored beside it and copied onto every record."""
    return sha256_text(canonical_json(manifest))


def instrument(manifest: dict) -> dict:
    """The run-affecting part of a manifest, without ids, timestamps or the lock.

    Two runs of one instrument produce the same dictionary. Anything that survives here
    and differs between two members of a series is a reproducibility defect, which is
    exactly what `series_conflicts` reports.

    Measurements are stripped from `decision` and only measurements. The rule is what the run was
    set up to do, not what the machine happened to be doing: the throughput probe's rates and its
    timestamp move with whatever else held the card, so leaving them in would make every series
    look irreproducible while nothing about the run had actually changed. The probe's configuration
    (prompt size, output size, temperature) is stated in its own `basis` string, which stays in the
    manifest and in the report.
    """
    base = {k: v for k, v in manifest.items() if k not in VOLATILE}
    decision = base.get("decision")
    if isinstance(decision, dict):
        kept = {k: v for k, v in decision.items() if k not in MEASURED_DECISION_KEYS}
        if kept:
            base["decision"] = kept
        else:
            base.pop("decision", None)
    return base


def instrument_fingerprint(manifest: dict) -> str:
    return sha256_text(canonical_json(instrument(manifest)))


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k in sorted(obj):
            out.update(_flatten(obj[k], f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        out[prefix] = canonical_json(obj)
    else:
        out[prefix] = obj
    return out


def differing_fields(left: dict, right: dict, ignore: Iterable[str] = ()) -> list[tuple[str, Any, Any]]:
    """Leaf paths where two manifests disagree, model identity excluded.

    Model identity is what makes two series different, not a defect: a 9B and a 1.7B are
    supposed to disagree on `models` and on `reproducibility.model.*`. Everything else
    must agree, the sampler chain and the chat template included, EXCEPT the paths the
    caller passes in `ignore`: those are the leaves that make up the run's configuration,
    and two runs that differ only there are two configurations, which is a difference of
    instrument rather than a reproducibility defect.
    """
    skip = tuple(("models", "reproducibility.model")) + tuple(ignore)
    a, b = _flatten(left), _flatten(right)
    out = []
    for path in sorted(set(a) | set(b)):
        if any(path == s or path.startswith(s + ".") for s in skip):
            continue
        if a.get(path, KeyError) != b.get(path, KeyError):
            out.append((path, a.get(path, "(absent)"), b.get(path, "(absent)")))
    return out


# The manifest leaves the configuration is made of. Two runs of one model that differ only on
# these are the same weights under two flag sets, which since the row key includes the
# configuration is two published rows rather than one series with an unrecorded split.
CONFIG_PATHS = (
    "notes.variant",
    "notes.chat-template-kwargs",
    "notes.ctx",
    "notes.kv-cache-k",
    "notes.kv-cache-v",
    "notes.server-argv",
    "reproducibility.server.variant",
    "reproducibility.server.ctx",
    "reproducibility.server.kv_cache_k",
    "reproducibility.server.kv_cache_v",
    "reproducibility.server.argv",
    "reproducibility.chat_template.kwargs",
)


# Manifest branches that only exist from 2026-11-01, when the task set started being frozen.
# A published run from before that date has none of them, and the absence is not an
# instrument difference; see _split_instrument_provenance.
INSTRUMENT_PROVENANCE_PATHS = ("instrument_frozen", "instrument_drift", "freeze")


def run_config(manifest: dict) -> dict:
    """The configuration a manifest records, as the record's own config block reads it."""
    return result_key.config_block(manifest.get("notes") or {},
                                   manifest.get("reproducibility") or {})


def _without_upstream(version: Any) -> str:
    """A Hermes version string with the upstream-tracking head removed."""
    return re.sub(r"\s*·\s*upstream\s+[0-9a-f]+", "", str(version or "")).strip()


def _split_upstream_tracking(
        diffs: list[tuple[str, Any, Any]]) -> tuple[list[tuple[str, Any, Any]],
                                                    list[tuple[str, Any, Any]]]:
    """Separate a version difference that is only the upstream-tracking head.

    `reproducibility.agent.hermes_version` carries `upstream <sha>`, the head of the branch
    the install tracks. That ref moves on any `git fetch`, with no change to the code being
    run, so two runs of identical local code written either side of a fetch disagree on it.
    On 2026-11-01 a fetch at 13:01 moved it from c419cf7c to 944799ab while HEAD stayed at
    67d47601bc since 2026-10-29, and the gate then refused to publish any new run of a model
    whose earlier run had been recorded before the fetch: a false instrument difference.

    What the field is for is real version drift, so the rest of the string (version, local
    head) still compares, as do every other leaf. Only the tracking ref, when it is the sole
    difference in that leaf, is reported as a note rather than as a conflict.
    """
    tracked: list[tuple[str, Any, Any]] = []
    hard: list[tuple[str, Any, Any]] = []
    for path, x, y in diffs:
        if (path == "reproducibility.agent.hermes_version"
                and _without_upstream(x) == _without_upstream(y)):
            tracked.append((path, x, y))
        else:
            hard.append((path, x, y))
    return tracked, hard


def _split_instrument_provenance(
        diffs: list[tuple[str, Any, Any]]) -> tuple[list[tuple[str, Any, Any]],
                                                    list[tuple[str, Any, Any]]]:
    """Separate "this run predates the freeze fence" from a real instrument difference.

    `instrument_frozen`, `instrument_drift` and `freeze` were added to the manifest on
    2026-11-01, when the task set started being frozen. Every run published before that date
    has no such leaf, so comparing a new run against one of those old rows reports the absence
    itself as a difference: the gate would then refuse to publish any new row of a model whose
    earlier row was recorded before the fence, which is the same false-difference defect that
    `_split_upstream_tracking` fixes for the version string.

    Only "present on one side, absent on the other" is downgraded. Two runs that both carry a
    freeze and disagree on its digest are still a hard conflict: they measured two different
    instruments.
    """
    provenance: list[tuple[str, Any, Any]] = []
    hard: list[tuple[str, Any, Any]] = []
    for path, x, y in diffs:
        frozen_branch = any(path == p or path.startswith(p + ".")
                            for p in INSTRUMENT_PROVENANCE_PATHS)
        if frozen_branch and "(absent)" in (x, y):
            provenance.append((path, x, y))
        else:
            hard.append((path, x, y))
    return provenance, hard


def series_conflicts(manifest: dict, published: list[dict]) -> list[str]:
    """Fail-loud messages describing how this run relates to published runs of its model.

    One message per related run, in two kinds:

    CONFLICT      the two runs claim the same configuration and disagree on something else,
                  so one of them cannot be trusted as the other's sibling.
    DISTINCT      the two runs claim different configurations. They are two instruments that
    CONFIGURATION share a file, and they publish as two rows, which is why the row key carries
                  the configuration token. Reported so the second row is announced rather
                  than appearing silently, and not fatal: refusing it would make the
                  thinking / not-thinking pair unpublishable, which is the defect this
                  module was extended to fix.

    For a pair that differs in configuration, the comparison runs again with the
    configuration leaves removed, so a build or sampler difference is still a conflict.
    """
    messages = []
    new_models = set(manifest.get("models") or [])
    new_config = run_config(manifest)
    for other in published:
        if other.get("run_id") == manifest.get("run_id"):
            continue
        if not (set(other.get("models") or []) & new_models):
            continue
        other_run = other.get("run_id") or "(unnamed published run)"
        if _dig(other, "reproducibility.schema_version", None) is None:
            # Legacy member: it records no reproducibility block, so it cannot be compared
            # and cannot be silently claimed as comparable either. Reported, not fatal, and
            # named in the publish record.
            messages.append(
                f"LEGACY {other_run}: published without reproducibility metadata, so this run "
                f"cannot be shown to share its instrument. Filed as legacy_incomparable.")
            continue
        other_config = run_config(other)
        ignore: Iterable[str] = ()
        if other_config != new_config:
            messages.append(
                f"DISTINCT CONFIGURATION {other_run}: published as "
                f"'{result_key.descriptor(other_config)}', this run is "
                f"'{result_key.descriptor(new_config)}'. Two configurations of one model are "
                f"two rows, not one series; they are compared below with the configuration "
                f"itself excluded.")
            ignore = CONFIG_PATHS
        diffs = differing_fields(instrument(manifest), instrument(other), ignore=ignore)
        tracked, diffs = _split_upstream_tracking(diffs)
        if tracked:
            _path, x, y = tracked[0]
            messages.append(
                f"NOTE {other_run}: the Hermes version strings differ only in the "
                f"upstream-tracking head ({x!r} against {y!r}). That ref moves on any "
                f"`git fetch` without changing the code under test, and the local build is "
                f"the same, so this is not an instrument difference. Not a conflict.")
        provenance, diffs = _split_instrument_provenance(diffs)
        if provenance:
            _path, x, y = provenance[0]
            messages.append(
                f"NOTE {other_run}: recorded before the task set was frozen, so it carries no "
                f"freeze provenance ({_path}: {x!r} against {y!r}). The fence postdates that "
                f"run; the absence is not an instrument difference.")
        if diffs:
            detail = "; ".join(f"{p}: {x!r} != {y!r}" for p, x, y in diffs[:8])
            more = "" if len(diffs) <= 8 else f" (+{len(diffs) - 8} more)"
            messages.append(
                f"CONFLICT {other_run}: {len(diffs)} recorded field(s) disagree with this run. "
                f"{detail}{more}")
    return messages


def hard_conflicts(messages: list[str]) -> list[str]:
    return [m for m in messages if m.startswith("CONFLICT")]


# --------------------------------------------------------------------------
# Loading a run
# --------------------------------------------------------------------------

class PublishRefused(Exception):
    """A publication the schema or the series invariant forbids."""


def load_run(run_id: str, runs_dir: Path | None = None) -> tuple[dict, list[dict], Path]:
    base = (runs_dir or RUNS_DIR) / run_id
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        raise PublishRefused(f"no manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    results_path = base / "results.json"
    records = json.loads(results_path.read_text()) if results_path.exists() else []
    return manifest, records, base


def verify_manifest_hash(manifest: dict, base: Path) -> None:
    """The manifest was written immutably at launch; prove it still is the one written.

    A manifest edited after the fact is how a series gets re-labelled to look uniform.
    `manifest.sha256` is written once, at launch, next to it.
    """
    stamp_path = base / "manifest.sha256"
    if not stamp_path.exists():
        raise PublishRefused(
            f"{base}/manifest.sha256 is missing. This run predates the launch-time hash, so "
            f"there is no proof its manifest was not edited after the model calls. Re-run it "
            f"rather than publish it.")
    written = stamp_path.read_text().strip()
    actual = manifest_sha256(manifest)
    if written != actual:
        raise PublishRefused(
            f"{base}/manifest.json no longer matches its launch hash.\n"
            f"  recorded at launch: {written}\n"
            f"  computed now:      {actual}\n"
            f"A manifest that changed after the run cannot be published as a record of it.")


def check_run(run_id: str, require_hash: bool = True) -> tuple[dict, list[dict], list[str]]:
    """Validate a run end to end. Raises PublishRefused; returns warnings otherwise."""
    manifest, records, base = load_run(run_id)
    if not isinstance(manifest.get("reproducibility"), dict):
        raise PublishRefused(
            f"run {run_id} records no reproducibility block. It predates the schema, or it is a "
            f"`merge` output assembled from such runs, so there is nothing to check it against "
            f"and nothing to compare it with. Only a run recorded by `hbench.py run` can be "
            f"published; re-run the measurement.")
    if require_hash:
        verify_manifest_hash(manifest, base)
    problems = validate_manifest(manifest)
    if problems:
        raise PublishRefused(format_problems(problems, f"run {run_id} (manifest.json)"))

    if not records:
        raise PublishRefused(f"run {run_id} has no results.json records to publish")

    fingerprint = instrument_fingerprint(manifest)
    stamp = manifest_sha256(manifest) if require_hash else _dig(
        manifest, "reproducibility.manifest_sha256", None)
    failures: list[str] = []
    for i, record in enumerate(records):
        problems = validate_record(record)
        if problems:
            failures.append(format_problems(
                problems, f"run {run_id} record {i} ({record.get('task')} r{record.get('repeat')})"))
            continue
        if record.get("instrument_fingerprint") != fingerprint:
            failures.append(
                f"  record {record.get('task')} r{record.get('repeat')}: instrument_fingerprint "
                f"{record.get('instrument_fingerprint')} does not match its manifest's "
                f"{fingerprint}. The record was produced under a different instrument than the "
                f"manifest describes.")
        if stamp and record.get("manifest_sha256") != stamp:
            failures.append(
                f"  record {record.get('task')} r{record.get('repeat')}: manifest_sha256 "
                f"{record.get('manifest_sha256')} is not the manifest it sits beside ({stamp}).")
    if failures:
        raise PublishRefused("REPRODUCIBILITY SCHEMA VIOLATION in run "
                             f"{run_id} (results.json):\n" + "\n".join(failures))
    return manifest, records, []


def published_manifests(runs_dir: Path | None = None) -> list[dict]:
    base = runs_dir or PUBLISHED_RUNS
    if not base.is_dir():
        return []
    out = []
    for path in sorted(base.glob("*/manifest.json")):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return out


def publish(run_id: str, dry_run: bool = False, force: bool = False,
            tree: Path | None = None) -> dict:
    """Validate, check the series invariant, then copy. Raises PublishRefused."""
    tree = tree or PUBLISH_TREE
    manifest, records, _ = check_run(run_id)
    messages = series_conflicts(manifest, published_manifests(tree / "runs"))
    conflicts = hard_conflicts(messages)
    if conflicts:
        raise PublishRefused(
            f"REFUSING TO PUBLISH {run_id}: its instrument disagrees with a published run of "
            f"the same model. A series whose members disagree on a recorded field is not a "
            f"series.\n" + "\n".join("  " + c for c in conflicts)
            + "\nPublish the run under its own model id, or re-run it under the recorded "
              "instrument of the series it belongs to.")
    legacy = [m for m in messages if m.startswith("LEGACY")]

    planned = []
    for record in records:
        if record.get("score_source") not in (None, "artifact"):
            continue   # excluded attempts are not leaderboard input
        name = published_name(record)
        planned.append((name, record))

    # The comparability stamp: one hash over every prompt, task file and grader fixture the
    # run measured, written into each published record. tools/export_results.py writes the
    # same field from the same manifest, so a row keeps it whichever writer lands last, and
    # two models hashing the same bundle are the only pair a reader can compare on scores.
    bundle = ((manifest.get("reproducibility") or {}).get("task_bundle") or {})
    if bundle:
        stamp = {"task_bundle_sha256": sha256_text(canonical_json(bundle)),
                 "task_bundle_tasks": len(bundle)}
        for _name, record in planned:
            for key, value in stamp.items():
                record.setdefault(key, value)

    if dry_run:
        print(f"[dry run] {len(planned)} record(s) would be published to {tree / 'results'}")
        for m in messages:
            print("  " + m)
        return {"published": 0, "planned": len(planned), "legacy": legacy,
                "instrument_fingerprint": instrument_fingerprint(manifest)}

    run_dest = tree / "runs" / run_id
    results_dest = tree / "results"
    run_dest.mkdir(parents=True, exist_ok=True)
    results_dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNS_DIR / run_id / "manifest.json", run_dest / "manifest.json")
    shutil.copy2(RUNS_DIR / run_id / "manifest.sha256", run_dest / "manifest.sha256")

    written = 0
    for name, record in planned:
        dest = results_dest / name
        if dest.exists():
            # Same name can mean two things, and they need different answers. The same
            # configuration filing twice is a re-publication and `--force` is the deliberate
            # answer. Two different configurations landing on one name is the collision this
            # whole change exists to prevent, and overwriting would destroy one of them.
            try:
                existing = json.loads(dest.read_text())
            except (OSError, ValueError):
                existing = {}
            mine = result_key.config_of(record)
            theirs = result_key.config_of(existing if isinstance(existing, dict) else {})
            if mine != theirs:
                raise PublishRefused(
                    f"{dest} already holds a measurement of this model under a different "
                    f"configuration ({result_key.descriptor(theirs)}); this run is "
                    f"{result_key.descriptor(mine)}. The published name is model plus "
                    f"configuration and these two collapsed onto one name, so the variant "
                    f"does not distinguish them. Give the run its own variant name; do not "
                    f"overwrite a published measurement.")
            if not force:
                raise PublishRefused(
                    f"{dest} already exists. Republishing would replace a published "
                    f"measurement; pass --force when that is deliberate.")
        dest.write_text(json.dumps(record, indent=2, sort_keys=True))
        written += 1

    record_of_publication = {
        "run_id": run_id,
        "published": written,
        "instrument_fingerprint": instrument_fingerprint(manifest),
        "manifest_sha256": manifest_sha256(manifest),
        "legacy_incomparable": legacy,
        "records": [name for name, _ in planned],
    }
    (run_dest / "published.json").write_text(json.dumps(record_of_publication, indent=2))
    return record_of_publication


def published_name(record: dict) -> str:
    """The published file name: model, configuration, task, repeat.

    Delegated to ``tools/result_key.py``, which ``tools/export_results.py`` also calls.
    Two spellings for one model split that model into two in ``publish/tree/results``.
    On 2026-11-01 Ornith 1.5 9B had 36 records under the old slug and 54 under the raw
    path spelling of the same id, and the README row added them together to 77/90 across
    two different suites. One rule, one file per (model, configuration, task, repeat).
    """
    return result_key.published_name(record)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def cmd_check(args) -> int:
    if args.manifest:
        manifest = json.loads(Path(args.manifest).read_text())
        problems = validate_manifest(manifest)
        if problems:
            print(format_problems(problems, args.manifest))
            return 3
        print(f"{args.manifest}: schema OK; instrument {instrument_fingerprint(manifest)[:16]}")
        return 0
    if not args.run_id:
        print("check needs a run id or --manifest", file=sys.stderr)
        return 2
    try:
        manifest, records, _ = check_run(args.run_id, require_hash=not args.no_hash)
    except PublishRefused as exc:
        print(str(exc), file=sys.stderr)
        return 3
    print(f"{args.run_id}: manifest OK, {len(records)} record(s) OK; "
          f"instrument {instrument_fingerprint(manifest)[:16]}")
    for message in series_conflicts(manifest, published_manifests()):
        print("  " + message)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "hbench publication gate").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, helptext in (("check", "validate a run or a manifest without publishing"),
                           ("publish", "validate, check the series invariant, then copy")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("run_id", nargs="?")
        p.add_argument("--manifest", help="validate a manifest file instead of a run id")
        p.add_argument("--no-hash", action="store_true",
                       help="skip the launch-hash check (legacy runs and fixtures only)")
        if name == "publish":
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--force", action="store_true",
                           help="replace already-published records of this run")
        p.set_defaults(func=cmd_check if name == "check" else cmd_publish)
    args = ap.parse_args(argv)
    return args.func(args)


def cmd_publish(args) -> int:
    if not args.run_id:
        print("publish needs a run id", file=sys.stderr)
        return 2
    try:
        result = publish(args.run_id, dry_run=args.dry_run, force=args.force)
    except PublishRefused as exc:
        print(str(exc), file=sys.stderr)
        return 3
    for message in result.get("legacy_incomparable", result.get("legacy", ())):
        print("  " + message, file=sys.stderr)
    print(f"published {result['published']} record(s) for {args.run_id} "
          f"(instrument {result['instrument_fingerprint'][:16]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
