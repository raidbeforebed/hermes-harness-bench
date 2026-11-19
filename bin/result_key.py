#!/usr/bin/env python3
"""One rule for the published identity of a result: model plus configuration.

The defect this closes
----------------------
The row key was the model. The instrument is the model plus its server flags. So
LFM3 1.7B with its chat template's thinking left on and LFM3 1.7B with thinking
switched off published into a single leaderboard row, and a reader could not see that
the better number came from the configuration that was then abandoned:

    runs/lfm3-think-20261030            reasoning high    12 records   mean 0.9861
    runs/lfm3-llamacpp-v2-20261030      no reasoning flag 36 records   mean 0.9676

Two instruments, two results, one published row. The same class of defect had already
landed in the neighbouring form: ``hbench_publish.published_name`` and
``tools/export_results.py`` each carried their own slug rule, so one model occupied two
file sets and the README row added 36 and 54 records across two different suites.

So the rule lives here, once, and every writer and reader calls it: the publish gate,
the exporter, the display registry, the charts, the README tables, the head-to-head
matrix and the dashboard.

What a configuration is
-----------------------
Six recorded fields, never inferred:

    variant                the wrapper's variant name, e.g. ``nothink-q8kv``
    thinking               True, False, or "unknown"
    chat_template_kwargs   the --chat-template-kwargs JSON as passed, or a statement
                           that none were passed
    ctx                    the context the server opened
    kv_cache_k, kv_cache_v the KV cache types

A field the manifest did not record reads ``"unrecorded"``. ``thinking`` reads
``"unknown"`` when the record carries template kwargs that do not name
``enable_thinking``: the template's own default then applies and this module will not
guess which way it points. A guessed configuration is worse than a stated unknown,
which is the lesson of the Ornith repeat-1 episode.

Backward compatibility
----------------------
Every run written before 2026-11-01 carries no ``config`` block at all. ``config_of``
returns the all-unrecorded block for those, so they still publish and still appear on
every surface, labelled ``configuration unrecorded`` rather than silently claiming a
setting. Their file names do not change either: a record that states no configuration
has no configuration dimension to encode, so ``published_name`` returns the pre-existing
name. That keeps the historical file set stable and makes this change additive.
"""

from __future__ import annotations

import json
import re

UNRECORDED = "unrecorded"
UNKNOWN = "unknown"

CONFIG_FIELDS = ("variant", "thinking", "chat_template_kwargs",
                 "ctx", "kv_cache_k", "kv_cache_v")

# Fixed order for a model's configurations, so thinking and not-thinking sit together and
# in the same order for every model: a reader comparing two models always finds the same
# pair laid out the same way.
THINKING_RANK = {"False": 0, "True": 1, UNKNOWN: 2, UNRECORDED: 3}


def _slug(text) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")


def _recorded(value) -> bool:
    """A value counts as recorded when it is present and not a placeholder."""
    if value is None or isinstance(value, bool):
        return value is not None
    if isinstance(value, str):
        return bool(value.strip()) and value.strip().lower() not in {
            UNRECORDED, UNKNOWN, "(unset)", "n/a", "-"}
    return True


def _kwargs_thinking(kwargs) -> object:
    """True, False or UNKNOWN, read only from an explicit ``enable_thinking``.

    The chat-template kwargs are the only recorded statement of intent. Absent, empty or
    placeholder kwargs mean the template's default applied, and what that default is
    belongs to the model, not to this record, so it stays unknown.
    """
    if not isinstance(kwargs, str):
        return UNKNOWN
    text = kwargs.strip()
    if not text or text.startswith("("):
        return UNKNOWN
    try:
        parsed = json.loads(text)
    except ValueError:
        return UNKNOWN
    if not isinstance(parsed, dict):
        return UNKNOWN
    value = parsed.get("enable_thinking")
    return value if isinstance(value, bool) else UNKNOWN


def _ctx(value):
    if isinstance(value, bool):
        return UNRECORDED
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return UNRECORDED


def config_block(notes: dict | None, reproducibility: dict | None = None) -> dict:
    """The config block for a record the harness is about to write.

    Reads only what the manifest recorded: ``notes`` carries the wrapper's variant, the
    chat-template kwargs and the KV cache types; ``reproducibility.server`` carries the
    same facts resolved against the live process. Nothing is derived from the model name.
    """
    notes = notes or {}
    server = ((reproducibility or {}).get("server") or {})
    template = ((reproducibility or {}).get("chat_template") or {})

    variant = notes.get("variant") or server.get("variant")
    kwargs = (notes.get("chat-template-kwargs") or template.get("kwargs"))
    return {
        "variant": variant if _recorded(variant) else UNRECORDED,
        "thinking": _kwargs_thinking(kwargs),
        "chat_template_kwargs": kwargs if _recorded(kwargs) else UNRECORDED,
        "ctx": _ctx(server.get("ctx") if server.get("ctx") is not None
                    else notes.get("ctx")),
        "kv_cache_k": server.get("kv_cache_k") or notes.get("kv-cache-k") or UNRECORDED,
        "kv_cache_v": server.get("kv_cache_v") or notes.get("kv-cache-v") or UNRECORDED,
    }


def config_of(record: dict) -> dict:
    """The normalised config block of any record, historical or current.

    A record written before 2026-11-01 has no ``config`` key, and the answer is the
    all-unrecorded block. It is deliberately not back-filled from the run's manifest:
    the record is the unit of publication, and a configuration that is not in the record
    is not part of the published claim.
    """
    raw = record.get("config")
    raw = raw if isinstance(raw, dict) else {}
    out = {}
    for field in CONFIG_FIELDS:
        value = raw.get(field)
        if field == "thinking":
            out[field] = value if value in (True, False) else UNKNOWN
        elif field == "ctx":
            out[field] = _ctx(value)
        else:
            out[field] = value if _recorded(value) else UNRECORDED
    return out


def is_unrecorded(config: dict | None) -> bool:
    """True when the record states nothing about how the model was configured."""
    config = config or config_of({})
    if _recorded(config.get("variant")):
        return False
    if config.get("thinking") in (True, False):
        return False
    return not _recorded(config.get("chat_template_kwargs"))


def thinking_of(config: dict | None):
    value = (config or {}).get("thinking")
    return value if value in (True, False) else UNKNOWN


def token(config: dict | None) -> str | None:
    """The stable configuration token for a file name, or None when nothing is stated.

    The variant name is the head of the token, because it is the operator's own name for the
    flag set: ``q8kv`` and ``nothink-q8kv`` are two tokens, two file sets, two rows. The
    thinking switch is appended when the variant name does not already say it, so two runs
    that share a variant name but differ in whether the model was asked to think still land
    in two rows instead of one.
    """
    config = config or config_of({})
    if is_unrecorded(config):
        return None
    variant = config.get("variant")
    thinking = thinking_of(config)
    head = _slug(str(variant)) if _recorded(variant) else ""
    words = head.split("-") if head else []
    if thinking is False and "nothink" not in words:
        head = f"{head}-nothink" if head else "nothink"
    elif thinking is True and "think" not in words:
        head = f"{head}-think" if head else "think"
    if head:
        return head
    return None


def descriptor(config: dict | None) -> str:
    """The configuration as it reads inside a display label."""
    config = config or config_of({})
    if is_unrecorded(config):
        return "configuration unrecorded"
    thinking = thinking_of(config)
    if thinking is True:
        return "thinking"
    if thinking is False:
        return "not thinking"
    variant = config.get("variant")
    if _recorded(variant):
        return f"{variant}, thinking unstated"
    return "thinking unstated"


def config_rank(config: dict | None) -> int:
    return THINKING_RANK.get(str(thinking_of(config)), THINKING_RANK[UNKNOWN])


def model_slug(model: str | None) -> str:
    return _slug(model) or "config-default"


def published_name(record: dict) -> str:
    """The published file name for one attempt: model, configuration, task, repeat.

    ``LFM3-1.7B-Q4_K_M.gguf`` under ``q8kv`` and under ``nothink-q8kv`` are two distinct
    published sets because the token differs. A record that states no configuration keeps
    the unqualified name it has always had, so the historical file set is not rewritten.
    """
    stem = model_slug(record.get("model"))
    tag = token(config_of(record))
    middle = f"__{tag}" if tag else ""
    return f"{stem}{middle}__{record['task']}__r{record['repeat']}.json"


def key_of(record: dict) -> str:
    """A row key for grouping surfaces: the model and its configuration token.

    A record that states nothing gets ``unrecorded``; one that states a configuration
    without a name for it gets ``unstated``. Both are distinct rows from a named variant,
    and both are distinct from each other.
    """
    config = config_of(record)
    suffix = token(config) or (UNRECORDED if is_unrecorded(config) else "unstated")
    return f"{record.get('model')}|{suffix}"
