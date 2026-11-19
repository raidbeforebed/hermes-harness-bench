#!/usr/bin/env python3
"""Decision metrics beyond pass rate (workstream 7 of the hbench robustness spec).

Why this exists
---------------
Every earlier workstream made the reported number more honest, and the number is still not the
thing the decision turns on. Both headline candidates pass about 96 percent of the suite, so on
task success they are indistinguishable (see bin/hbench_uncertainty.py). What separates them on a
Pascal card is what the run cost: throughput, wall clock, how many turns it took to get there, how
many tool calls the harness had to reject, how close to the context ceiling it ran, and how much
card it held. Those differ by multiples while the score differs by noise.

The six metrics, with the units they are reported in
----------------------------------------------------
  1. throughput (tok/s) ......... two numbers, two bases, because an agent attempt cannot give
     both honestly:
       * attempt-derived: this attempt's compute tokens (uncached prompt plus output) divided by
         its model time, where model time is the stream span minus time spent inside tools. One
         composite rate, not a fake decode rate: an agent stream carries no per-request timings, so
         splitting it would report prompt tokens over total time and call the result "prefill
         tok/s", which understates prefill by an order of magnitude. Available for every archived
         run, so the back catalogue is measurable without re-running a model.
       * server probe: llama.cpp's own `timings.prompt_per_second` (prefill) and
         `predicted_per_second` (decode) from one fixed request per run, which is where the two
         rates genuinely separate, and which does not depend on how much tool work the agent chose
         to do.
  2. wall time (s) .............. per attempt, reported as mean, p50, p90 and max, not only a mean.
  3. turns to success (turns) ... mean turns over attempts that scored 1.000, with the mean turns
     spent on failures beside it. A 9B that solves in 4 turns and a 1.7B that takes 12 is a real
     difference at equal accuracy.
  4. tool-call validity (%) ..... valid tool calls over total tool calls. A call the harness
     rejected (bad arguments, a blocked tool, a refusal) is invisible in a pass/fail score and is
     exactly the failure mode a small model has.
  5. context headroom (%) ....... percent of the model's window still free at the moment of
     failure or success: 1 - peak prompt tokens / n_ctx. It separates "the model forgot" from
     "the agent loop drowned it". Sampled from the server's own /slots while the attempt runs.
  6. peak VRAM (MiB) ............ the largest card footprint and the largest server-process
     footprint seen during the attempt, sampled from the main PC's SENTINEL telemetry. This is
     the measurement that makes "largest usable context per model" a number rather than a guess.

GPU-exclusive measurements
--------------------------
Metrics 5 and 6 are sampled from the machine that holds the card, not from the client, and they
are only meaningful when nothing else is using that card:

  * peak VRAM is the whole-card used figure and the llama-server process figure. If another GPU
    service (qwen, comfyui, voicestudio, luxtts) is up, the card figure includes it. The sampler
    records every co-tenant process it saw and flags the measurement as contaminated when a
    co-tenant held more than VRAM_CO_TENANT_LIMIT_MIB, so a contaminated number is labelled
    rather than quietly reported as the model's own.
  * context headroom comes from the server's single request slot, so it measures the bench's own
    request only when the bench is the only client. The same co-tenant rule applies.
  * the throughput probe issues one real generation, so it is GPU work: it is skipped when the
    endpoint is a cloud provider and it is recorded with its own timestamp so a reader can see
    what order the probes and the attempts ran in.

So: take these numbers with the card to yourself. `python3 local_models.py status` shows which
services are up, and the report prints the co-tenants it observed for exactly this reason.

Nothing here invents a number. A metric that could not be measured is reported as
"unmeasured" with the reason, never as a zero, because a zero reads as a measurement.

Archived runs
-------------
The four client-side metrics (throughput from the stream, wall time, turns, tool validity) are
derived from the record, so they exist for every run on disk including the ones made before this
module. The two sampled metrics need samples, so a run made before the sampler existed reports
them as unmeasured with that reason and is not back-filled.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# The tree this file is in, so a clone measures its own tasks and writes its own runs. A
# hard-coded default silently measured this workstation's tree from a clone two directories
# away and wrote the results outside the checkout, which made a clean-clone acceptance run
# evidence about the wrong instrument. HBENCH_ROOT still overrides it deliberately.
ROOT = Path(os.environ.get("HBENCH_ROOT") or Path(__file__).resolve().parent.parent)
RUNS_DIR = ROOT / "runs"

SCHEMA_VERSION = 1

# Sampling. Two seconds is slow enough to cost nothing and fast enough that a multi-minute agent
# attempt is sampled dozens of times; a peak held for longer than that is not a peak anyone would
# notice. The timeout is short because a sampler that blocks behind a slow endpoint delays nothing
# but its own thread, and a stale sample is worse than no sample.
DEFAULT_INTERVAL = 2.0
# Public alias: hbench.py prints the interval it is about to sample at, and a reader of the run log
# should not have to know which private name it is stored under.
SAMPLE_INTERVAL = DEFAULT_INTERVAL
HTTP_TIMEOUT = 3.0
# The main PC's SENTINEL control API. Overridable so a test (or a run against another machine)
# can point the sampler somewhere else; the default is the box that holds the card.
DEFAULT_SENTINEL = os.environ.get("HBENCH_SENTINEL_URL", "http://192.168.1.15:8123")

# A process on the card that is not the benchmark's model server. Anything holding more than this
# many MiB flags the VRAM and context measurements as contaminated, because the card was shared
# while they were taken. 512 MiB clears a desktop compositor and a loaded whisper model, and is far
# below a second LLM or an image pipeline.
VRAM_CO_TENANT_LIMIT_MIB = 512

# Process names that mean "this is the model server under test". SENTINEL reports a cmdline
# snippet, so the match is a substring test on the lowercased name.
SERVER_PROCESS_PATTERNS = ("llama-server", "llama_server", "llamacpp", "ollama", "vllm")

# The throughput probe is fixed, because a throughput number is only comparable with another
# throughput number when both were measured the same way: same prompt size, same output size,
# same sampling. 128 output tokens is long enough that decode dominates the fixed per-request
# overhead and short enough to cost a second or two on a Pascal card.
PROBE_PROMPT_TOKENS = 1024
PROBE_OUTPUT_TOKENS = 128
PROBE_TEMPERATURE = 0.0

UNITS = {
    "throughput": "tok/s",
    "wall": "s",
    "turns": "turns",
    "tool_validity": "%",
    "context_headroom": "%",
    "vram": "MiB",
}


# --------------------------------------------------------------------------
# Small statistics helpers (deterministic, stdlib only)
# --------------------------------------------------------------------------

def percentile(values: Iterable[float | None], q: float) -> float | None:
    """Linear-interpolated percentile, so two runs of one record set give one answer.

    ``statistics.quantiles`` needs at least two points and its method has changed between Python
    versions; a metric that moves when the interpreter is upgraded is not a metric.
    """
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def mean_or_none(values: Iterable[float | None]) -> float | None:
    vals = [float(v) for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def _fmt(value: float | None, digits: int = 1, unit: str = "", missing: str = "unmeasured") -> str:
    if value is None:
        return missing
    return f"{value:.{digits}f}{(' ' + unit) if unit else ''}"


# --------------------------------------------------------------------------
# Reading an attempt's stream
# --------------------------------------------------------------------------

def _iter_events(raw: str) -> Iterable[dict]:
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:  # noqa: BLE001 - a corrupt line is a parse error, not a crash
            continue
        if isinstance(ev, dict):
            yield ev


def is_tool_call_event(ev: dict) -> bool:
    """A tool call being issued, never its result.

    ``parse_stream`` in hbench.py counted both halves of every pair, which is why archived rows
    show twice as many tool calls as the child made (12 for 6 calls). The validity rate has to be
    over calls, so this counts the issuing side only.
    """
    low = str(ev.get("type") or ev.get("event") or "").lower()
    if not isinstance(ev.get("name"), str):
        return False
    return low in ("tool_use", "tool_call", "tool_use_delta") or ("tool" in low and "result" not in low)


def stream_facts(record: dict) -> dict:
    """Tool calls, their validity, tool time and the stream span, straight from the record.

    Everything here is derived from the attempt's own stream, which is stored in the record, so
    an archived run yields the same numbers today as it did when it was made.
    """
    calls = 0
    invalid = 0
    invalid_names: list[str] = []
    tool_seconds = 0.0
    have_tool_time = False
    have_error_flag = False
    stamps: list[float] = []

    for ev in _iter_events(record.get("stdout_raw") or ""):
        low = str(ev.get("type") or ev.get("event") or "").lower()
        ts = ev.get("timestamp")
        if isinstance(ts, (int, float)):
            stamps.append(float(ts) / 1000.0 if ts > 1e11 else float(ts))
        if is_tool_call_event(ev):
            calls += 1
        if "tool" in low and "result" in low:
            duration = ev.get("duration_ms")
            if isinstance(duration, (int, float)):
                tool_seconds += float(duration) / 1000.0
                have_tool_time = True
            # The harness says so itself: is_error is set when the call was rejected or the tool
            # failed, and Hermes writes the field on every result (405 of 405 in the archived
            # Ornith series). Both cases are invalid calls for this metric, because the model did
            # not get a usable result out of the turn. Nothing is inferred from the payload text:
            # a tool result may legitimately contain the word error.
            if ev.get("is_error") is not None:
                have_error_flag = True
            if ev.get("is_error") is True:
                invalid += 1
                name = ev.get("name")
                if isinstance(name, str):
                    invalid_names.append(name)

    if calls == 0 and invalid:
        # A stream with results but no recognised call events: count the pairs so the rate still
        # has a denominator rather than reading as 0 of 0.
        calls = invalid

    stream_seconds = None
    if len(stamps) >= 2:
        stream_seconds = max(stamps) - min(stamps)

    return {
        "tool_calls": calls,
        "tool_calls_invalid": invalid,
        "tool_error_flag_present": have_error_flag,
        "invalid_tool_names": invalid_names[:20],
        "tool_seconds": round(tool_seconds, 2) if have_tool_time else None,
        "stream_seconds": round(stream_seconds, 2) if stream_seconds is not None else None,
    }


def attempt_decision(record: dict, *, window: dict | None = None,
                     ctx_size: int | None = None) -> dict:
    """The decision block for one attempt: six metrics, each with its unit and its basis.

    Pure function of the record plus whatever the sampler saw, so it can be recomputed for any
    archived run and produces the same block every time.
    """
    facts = stream_facts(record)
    wall = record.get("wall_seconds")
    wall = float(wall) if isinstance(wall, (int, float)) else None

    # Model time: the stream span minus time spent inside tools. Falling back to the wall clock
    # when the stream carries no timestamps keeps a metric rather than losing it, and the basis
    # string says which was used.
    tool_seconds = facts["tool_seconds"] or 0.0
    if facts["stream_seconds"] is not None:
        model_seconds = max(facts["stream_seconds"] - tool_seconds, 0.0)
        basis = "attempt-derived: stream span minus tool execution time"
    elif wall is not None:
        model_seconds = max(wall - tool_seconds, 0.0)
        basis = "attempt-derived: wall clock minus tool execution time (no stream timestamps)"
    else:
        model_seconds = None
        basis = "unmeasured: no wall clock and no stream"

    tokens_in = record.get("tokens_in")
    tokens_out = record.get("tokens_out")
    cache_read = record.get("cache_read")
    prefill_tokens = None
    if isinstance(tokens_in, int):
        prefill_tokens = tokens_in - (cache_read if isinstance(cache_read, int) else 0)
        prefill_tokens = max(prefill_tokens, 0)

    # Why one number and not two here: an agent attempt's stream carries no per-request timings, so
    # the client cannot separate prefill time from decode time inside the attempt. Dividing the
    # prompt tokens by the whole model time would report decode-pace numbers for the prompt and
    # call them prefill tok/s, which is how a 34 tok/s "prefill" appears for a card that prefills
    # at several hundred. So the attempt-level number is the honest composite, compute tokens
    # (uncached prompt plus output) over model time, and the separated prefill and decode rates
    # come from the run's fixed server probe, which is where they can be measured properly.
    compute_tokens = (prefill_tokens + tokens_out
                      if isinstance(prefill_tokens, int) and isinstance(tokens_out, int) else None)
    processed_tps = (compute_tokens / model_seconds
                     if compute_tokens is not None and model_seconds else None)

    score = record.get("score")
    score = float(score) if isinstance(score, (int, float)) else None
    passed = bool(score is not None and score >= 1.0)
    turns = record.get("turns") if isinstance(record.get("turns"), int) else None

    calls = facts["tool_calls"]
    invalid = facts["tool_calls_invalid"]
    validity = (100.0 * (calls - invalid) / calls
                if calls and facts["tool_error_flag_present"] else None)

    # Context: peak prompt tokens over the attempt's window, against the server's own n_ctx.
    window = window or {}
    kv_peak = window.get("kv_peak_tokens")
    n_ctx = window.get("n_ctx") or ctx_size
    headroom = None
    if isinstance(kv_peak, int) and isinstance(n_ctx, int) and n_ctx > 0:
        headroom = 100.0 * max(0.0, 1.0 - kv_peak / n_ctx)
    if window.get("context_source"):
        context_source = window["context_source"]
    else:
        context_source = "unmeasured: no server /slots samples for this attempt"

    vram = {
        "peak_card_mib": window.get("vram_peak_card_mib"),
        "peak_server_mib": window.get("vram_peak_server_mib"),
        "baseline_card_mib": window.get("vram_baseline_card_mib"),
        "co_tenants": window.get("co_tenants") or [],
        "contaminated": bool(window.get("contaminated", False)),
        "source": window.get("vram_source") or "unmeasured: no SENTINEL samples for this attempt",
        "gpu_exclusive_required": True,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "wall_seconds": wall,
        "stream_seconds": facts["stream_seconds"],
        "tool_seconds": facts["tool_seconds"],
        "model_seconds": round(model_seconds, 2) if model_seconds is not None else None,
        "throughput": {
            "tokens_per_second": round(processed_tps, 2) if processed_tps is not None else None,
            "compute_tokens": compute_tokens,
            "basis": basis,
        },
        "turns": turns,
        "turns_to_success": turns if passed else None,
        "turns_on_failure": turns if (turns is not None and not passed) else None,
        "tool_calls": calls,
        "tool_calls_invalid": invalid,
        "invalid_tool_names": facts["invalid_tool_names"],
        "tool_validity_pct": round(validity, 1) if validity is not None else None,
        "tokens": {"in": tokens_in, "out": tokens_out, "cache_read": cache_read,
                   "prefill": prefill_tokens},
        "context": {
            "n_ctx": n_ctx,
            "peak_prompt_tokens": kv_peak,
            "headroom_pct": round(headroom, 1) if headroom is not None else None,
            "source": context_source,
        },
        "vram": vram,
        "units": UNITS,
    }


# --------------------------------------------------------------------------
# The sampler: GPU-exclusive measurements, polled while a run is in flight
# --------------------------------------------------------------------------

def _server_root(base_url: str) -> str:
    """llama-server's root from an OpenAI base_url (``http://host:port/v1`` -> ``http://host:port``)."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root


def _get_json(url: str, api_key: str | None = None, timeout: float = HTTP_TIMEOUT) -> Any:
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8", "replace"))


def _get_text(url: str, api_key: str | None = None, timeout: float = HTTP_TIMEOUT) -> str | None:
    req = urllib.request.Request(url)
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status != 200:
                return None
            return response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError:
        return None


PROMETHEUS_LINE = re.compile(r"^(llamacpp[:][a-z_]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)\s*$")


def _parse_prometheus(text: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in (text or "").splitlines():
        match = PROMETHEUS_LINE.match(line.strip())
        if not match:
            continue
        try:
            out[match.group(1)] = float(match.group(2))
        except ValueError:
            continue
    return out


def summarise_samples(samples: list[dict]) -> dict:
    """Peaks and co-tenants over a window of samples.

    The peak is what matters for VRAM and for context: an attempt that touched 49,422 of 65,536
    tokens once was close to the ceiling, and an average would hide that.
    """
    if not samples:
        return {}
    kv = [s.get("kv_tokens") for s in samples if isinstance(s.get("kv_tokens"), int)]
    card = [s.get("vram_card_mib") for s in samples if isinstance(s.get("vram_card_mib"), (int, float))]
    server = [s.get("vram_server_mib") for s in samples
              if isinstance(s.get("vram_server_mib"), (int, float))]
    n_ctx = next((s.get("n_ctx") for s in samples if isinstance(s.get("n_ctx"), int)), None)

    tenants: dict[str, int] = {}
    for s in samples:
        for tenant in s.get("co_tenants") or []:
            name = str(tenant.get("name") or "?")
            vram = int(tenant.get("vram_mib") or 0)
            tenants[name] = max(tenants.get(name, 0), vram)
    contaminated = any(v >= VRAM_CO_TENANT_LIMIT_MIB for v in tenants.values())

    first_card = next((s.get("vram_card_mib") for s in samples
                       if isinstance(s.get("vram_card_mib"), (int, float))), None)
    return {
        "samples": len(samples),
        "seconds": round(samples[-1]["t"] - samples[0]["t"], 1) if len(samples) > 1 else 0.0,
        "kv_peak_tokens": max(kv) if kv else None,
        "n_ctx": n_ctx,
        "context_source": ("llama-server /slots n_prompt_tokens, sampled"
                           if kv else "unmeasured: /slots returned no prompt token count"),
        "vram_peak_card_mib": int(max(card)) if card else None,
        "vram_peak_server_mib": int(max(server)) if server else None,
        "vram_baseline_card_mib": int(first_card) if first_card is not None else None,
        "co_tenants": [{"name": k, "peak_mib": v} for k, v in sorted(tenants.items())],
        "contaminated": contaminated,
        "vram_source": ("SENTINEL /api/status vitals.gpu, sampled" if card
                        else "unmeasured: SENTINEL telemetry unavailable"),
    }


def sampler_sources(base_url: str | None = None) -> list[str]:
    """Which endpoints a sampler would read for this run, so the log names them before it starts.

    A cloud route has nothing to sample: there is no server slot to read and no card holding the
    model, so the list comes back empty and the run records the metrics as unmeasured rather than
    sampling the local card and attributing it to a remote model.
    """
    sources: list[str] = []
    if base_url:
        sources.append(f"llama-server {_server_root(base_url)} /slots")
        sources.append(f"SENTINEL {DEFAULT_SENTINEL} /api/status")
    return sources


class DecisionSampler:
    """Polls the model server and SENTINEL while a run is in flight.

    Two independent sources, either of which may be absent:

      * the model server's ``/slots`` for the live prompt token count (and ``/metrics`` when the
        server was started with ``--metrics``), which gives context headroom;
      * SENTINEL's ``/api/status`` for the card's VRAM and its per-process attribution, which
        gives peak VRAM and, just as importantly, the list of other processes on the card.

    Both are best effort by design: a sampler failure must never fail a benchmark attempt. What it
    could not measure is recorded as unmeasured, with the error, so the report can say so.
    """

    def __init__(self, *, base_url: str | None = None, api_key: str | None = None,
                 sentinel_url: str | None = DEFAULT_SENTINEL,
                 interval: float = DEFAULT_INTERVAL,
                 server_match: tuple[str, ...] = SERVER_PROCESS_PATTERNS,
                 timeout: float = HTTP_TIMEOUT) -> None:
        self.base_url = base_url
        self.root = _server_root(base_url) if base_url else None
        self.api_key = api_key
        self.sentinel = sentinel_url.rstrip("/") if sentinel_url else None
        self.interval = max(float(interval), 0.25)
        self.server_match = tuple(p.lower() for p in server_match)
        self.timeout = timeout
        self.samples: list[dict] = []
        self.errors: dict[str, str] = {}
        self.n_ctx: int | None = None
        self.started_at: str | None = None
        self.stopped_at: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "DecisionSampler":
        self.started_at = datetime.now(timezone.utc).isoformat()
        # One synchronous sample before the thread starts, so the baseline card figure exists even
        # for a run that finishes in under one interval.
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, name="hbench-decision-sampler",
                                        daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample_once()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2 + self.timeout)
        self._sample_once()
        self.stopped_at = datetime.now(timezone.utc).isoformat()
        return self.summary()

    # -- sampling ----------------------------------------------------------
    def _sample_once(self) -> dict:
        # `t` is taken before the reads, so everything the reads saw belongs to this timestamp and
        # a window that starts at `t` includes this sample. Taken after, a sample could fall just
        # outside the window it was taken for.
        sample: dict = {"t": time.monotonic(), "at": datetime.now(timezone.utc).isoformat()}
        if self.root:
            self._sample_server(sample)
        if self.sentinel:
            self._sample_sentinel(sample)
        self.samples.append(sample)
        return sample

    def _sample_server(self, sample: dict) -> None:
        try:
            slots = _get_json(self.root + "/slots", self.api_key, self.timeout)
        except Exception as exc:  # noqa: BLE001
            self.errors["slots"] = f"{type(exc).__name__}: {exc}"
            slots = None
        if isinstance(slots, list) and slots and isinstance(slots[0], dict):
            slot = slots[0]
            if isinstance(slot.get("n_prompt_tokens"), int):
                sample["kv_tokens"] = slot["n_prompt_tokens"]
            if isinstance(slot.get("n_ctx"), int):
                sample["n_ctx"] = slot["n_ctx"]
                self.n_ctx = slot["n_ctx"]
            sample["processing"] = bool(slot.get("is_processing"))
            self.errors.pop("slots", None)
        elif slots is None:
            self.errors.setdefault("slots", "unreachable or not a slot list")
        # /metrics is only served when llama-server was started with --metrics. When it is, it
        # carries counters that make the throughput probe checkable; when it is not, the response
        # is a 501 that names the flag, which is a better diagnostic than silence.
        try:
            text = _get_text(self.root + "/metrics", self.api_key, self.timeout)
        except Exception as exc:  # noqa: BLE001
            self.errors["metrics"] = f"{type(exc).__name__}: {exc}"
            text = None
        counters = _parse_prometheus(text)
        if counters:
            if counters.get("llamacpp:kv_cache_tokens") is not None:
                sample.setdefault("kv_tokens", int(counters["llamacpp:kv_cache_tokens"]))
            for key, name in (("prompt_tokens_total", "llamacpp:prompt_tokens_total"),
                              ("tokens_predicted_total", "llamacpp:tokens_predicted_total"),
                              ("requests_processing", "llamacpp:requests_processing")):
                if counters.get(name) is not None:
                    sample[key] = counters[name]
            self.errors.pop("metrics", None)
        elif text is None:
            self.errors.setdefault("metrics", "not served (llama-server needs --metrics)")

    def _sample_sentinel(self, sample: dict) -> None:
        try:
            status = _get_json(self.sentinel + "/api/status", None, self.timeout)
        except Exception as exc:  # noqa: BLE001
            self.errors["sentinel"] = f"{type(exc).__name__}: {exc}"
            return
        if not isinstance(status, dict):
            # A payload of the wrong shape is a sampler problem, not a benchmark failure: record
            # it and leave the metric unmeasured rather than taking the attempt down with it.
            self.errors["sentinel"] = (f"unexpected payload type {type(status).__name__}, "
                                       f"expected an object")
            return
        gpu = ((status or {}).get("vitals") or {}).get("gpu") or {}
        if not isinstance(gpu, dict) or not gpu:
            self.errors.setdefault("sentinel", "no vitals.gpu in the payload")
            return
        if isinstance(gpu.get("vram_used_mb"), (int, float)):
            sample["vram_card_mib"] = int(gpu["vram_used_mb"])
        if isinstance(gpu.get("vram_total_mb"), (int, float)):
            sample["vram_total_mib"] = int(gpu["vram_total_mb"])
        if isinstance(gpu.get("temp_c"), (int, float)):
            sample["gpu_temp_c"] = int(gpu["temp_c"])
        if isinstance(gpu.get("util_percent"), (int, float)):
            sample["gpu_util_percent"] = int(gpu["util_percent"])
        matched_mib = 0
        matched: list[dict] = []
        others: list[dict] = []
        for proc in gpu.get("processes") or []:
            if not isinstance(proc, dict):
                continue
            name = str(proc.get("name") or "")
            entry = {"name": name, "pid": proc.get("pid"), "vram_mib": int(proc.get("vram_mb") or 0)}
            if any(pat in name.lower() for pat in self.server_match):
                matched.append(entry)
                matched_mib += entry["vram_mib"]
            elif entry["vram_mib"] > 0:
                others.append(entry)
        if matched:
            sample["vram_server_mib"] = matched_mib
            sample["server_procs"] = matched
        if others:
            sample["co_tenants"] = others
        self.errors.pop("sentinel", None)

    # -- windows -----------------------------------------------------------
    def mark(self) -> float:
        """A cut point in sample order, taken when an attempt starts.

        It samples synchronously here instead of waiting for the next tick, because an attempt can
        be shorter than one interval. Without this a fast task would come back with an empty window
        and both sampled metrics unmeasured, which reads as "the sampler never worked" rather than
        "the attempt was quick".
        """
        try:
            return self._sample_once()["t"]
        except Exception:  # noqa: BLE001
            # A sampler must never fail an attempt. No sample means the two sampled metrics are
            # reported unmeasured for this attempt, which is the honest outcome anyway.
            return time.monotonic()

    def window(self, since: float | None = None) -> dict:
        samples = [s for s in self.samples if since is None or s["t"] >= since]
        return summarise_samples(samples)

    def summary(self) -> dict:
        """The run-level shape of what was sampled, for the manifest."""
        overall = summarise_samples(self.samples)
        sources = []
        if self.root:
            sources.append(f"llama-server {self.root} /slots")
        if self.sentinel:
            sources.append(f"SENTINEL {self.sentinel} /api/status")
        return {
            "schema_version": SCHEMA_VERSION,
            "sources": sources,
            "interval_seconds": self.interval,
            "started": self.started_at,
            "stopped": self.stopped_at,
            "samples": len(self.samples),
            "errors": dict(self.errors),
            "n_ctx": self.n_ctx,
            "gpu_exclusive_required": True,
            "gpu_exclusive_note": (
                "Peak VRAM and context headroom are sampled from the machine holding the card. "
                "Both are the benchmark's own only when no other GPU service (qwen, comfyui, "
                "voicestudio, luxtts) is up. Co-tenants are listed per attempt and a co-tenant "
                f"holding {VRAM_CO_TENANT_LIMIT_MIB} MiB or more marks the attempt contaminated."),
            "peaks": overall,
        }


def probe_throughput(base_url: str, api_key: str | None, *,
                    model: str | None = None,
                    enabled: bool = True,
                    prompt_tokens: int = PROBE_PROMPT_TOKENS,
                    output_tokens: int = PROBE_OUTPUT_TOKENS,
                    salt: str | None = None, timeout: float = 120.0,
                    retries: int = 6, retry_delay: float = 5.0) -> dict:
    """One fixed request to a llama.cpp server, returning its own prefill and decode rates.

    This is the canonical throughput number: every run issues the same prompt size and asks for
    the same output size at temperature 0, so the only variable is the machine. The salt keeps the
    prompt from being a prefix hit in the server's cache (a cached prompt measures nothing), and
    it is recorded so a suspiciously fast probe can be explained.

    A transport error is retried, because the probe fires at run start, which on this rig is
    exactly the window where the previous run's wrapper stops one server and this run starts
    another: one live run lost its probe to a connection refused inside that restart. Only a
    transport error or a 5xx is retried (a restart is transient, a 4xx is not), and the attempt
    count is recorded, so a probe that needed four tries is visible in the manifest rather than
    silently identical to one that needed none.

    llama.cpp returns a ``timings`` block on a non-streaming completion:
        prompt_n, prompt_ms, prompt_per_second, predicted_n, predicted_ms, predicted_per_second
    """
    root = _server_root(base_url)
    # A unique salt per probe, so the prompt is not a prefix hit in the server's prompt cache. A
    # cached prompt is not processed, so it measures nothing: llama.cpp then reports prompt_n as the
    # handful of new tokens and the "prefill rate" is that handful divided by the request time.
    if salt is None:
        salt = uuid.uuid4().hex[:12]
    if not enabled:
        return {"ok": False, "skipped": True, "salt": salt,
                "reason": "disabled (--no-throughput-probe)",
                "at": datetime.now(timezone.utc).isoformat()}
    words = max(int(prompt_tokens * 0.75), 32)
    body_text = f"hbench throughput probe {salt}\n" + " ".join(f"w{i}" for i in range(words))
    payload = {
        "messages": [{"role": "user", "content": body_text}],
        "max_tokens": output_tokens,
        "temperature": PROBE_TEMPERATURE,
        "stream": False,
        "seed": 1234,
    }
    if model:
        payload["model"] = model
    req = urllib.request.Request(
        root + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    attempts = max(int(retries), 1)
    data = None
    last: Exception | None = None
    used = 0
    req_started = time.monotonic()
    for attempt in range(1, attempts + 1):
        used = attempt
        req_started = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8", "replace"))
            break
        except urllib.error.HTTPError as exc:
            # A 5xx is the server restarting; a 4xx (wrong key, unknown model) will not improve.
            if exc.code < 500 or attempt >= attempts:
                return {"ok": False, "reason": f"{type(exc).__name__}: {exc}",
                        "salt": salt, "attempts": attempt,
                        "at": datetime.now(timezone.utc).isoformat()}
            last = exc
        except Exception as exc:  # noqa: BLE001
            if not isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError)) \
                    or attempt >= attempts:
                return {"ok": False, "reason": f"{type(exc).__name__}: {exc}",
                        "salt": salt, "attempts": attempt,
                        "at": datetime.now(timezone.utc).isoformat()}
            last = exc
        time.sleep(retry_delay)
    if data is None:
        return {"ok": False, "reason": f"{type(last).__name__}: {last}",
                "salt": salt, "attempts": used,
                "at": datetime.now(timezone.utc).isoformat()}
    wall = time.monotonic() - req_started
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    result = {
        "ok": bool(timings),
        "salt": salt,
        "attempts": used,
        "at": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(wall, 2),
        "prompt_tokens": timings.get("prompt_n", usage.get("prompt_tokens")),
        "prompt_tok_s": timings.get("prompt_per_second"),
        "decode_tokens": timings.get("predicted_n", usage.get("completion_tokens")),
        "decode_tok_s": timings.get("predicted_per_second"),
        "basis": ("llama.cpp timings from one fixed request: "
                  f"{prompt_tokens} prompt tokens asked for, {output_tokens} output tokens max, "
                  f"temperature {PROBE_TEMPERATURE}"),
    }
    if not timings:
        result["reason"] = ("the endpoint returned no timings block, so it is not a llama.cpp "
                            "server (or it is a cloud route); no throughput was measured")
    elif isinstance(result.get("prompt_tokens"), int) and result["prompt_tokens"] < words // 2:
        # The prompt came back almost entirely cached, so the prefill figure describes the few new
        # tokens and not the prompt. Report the gap instead of a rate that looks measured.
        result["ok"] = False
        result["reason"] = (f"prompt cache hit: llama.cpp processed {result['prompt_tokens']} prompt "
                            f"tokens of the ~{prompt_tokens} asked for, so no prefill rate was "
                            f"measured; a unique salt is what prevents this")
    return result


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def _decision_of(record: dict, *, ctx_size: int | None) -> dict:
    """A record's decision block, recomputed when the record predates the sampler."""
    block = record.get("decision")
    if isinstance(block, dict) and block.get("schema_version") == SCHEMA_VERSION:
        return block
    return attempt_decision(record, window=None, ctx_size=ctx_size)


def _scorable(record: dict) -> bool:
    src = record.get("score_source")
    if src:
        return src == "artifact"
    return not record.get("harness_failed")


def aggregate(records: list[dict], *, ctx_size: int | None = None,
              probe: dict | None = None) -> dict:
    """Per-model decision metrics over a set of attempts.

    ``probe`` is the run's fixed throughput measurement (see ``probe_throughput``). It is the only
    source that separates prefill from decode, so it is carried through rather than recomputed.
    """
    by_model: dict[str, list[dict]] = {}
    for record in records:
        by_model.setdefault(record.get("model") or "config-default", []).append(record)

    out: dict[str, dict] = {}
    for model, rs in by_model.items():
        blocks = [(r, _decision_of(r, ctx_size=ctx_size)) for r in rs]
        # A re-grade can change `score` after collection, so whether an attempt succeeded is read
        # from the record as it stands now, never from the block frozen at collection time.
        for r, b in blocks:
            passed = (r.get("score") or 0) >= 1.0
            turns = b.get("turns")
            b["turns_to_success"] = turns if passed else None
            b["turns_on_failure"] = turns if (turns is not None and not passed) else None
        scored = [(r, b) for r, b in blocks if _scorable(r)]
        wins = [(r, b) for r, b in scored if (r.get("score") or 0) >= 1.0]
        losses = [(r, b) for r, b in scored if (r.get("score") or 0) < 1.0]

        walls = [b.get("wall_seconds") for _, b in blocks]
        processed = [b["throughput"]["tokens_per_second"] for _, b in blocks
                     if isinstance(b.get("throughput"), dict)]
        turns_success = [b.get("turns_to_success") for _, b in wins]
        turns_failure = [b.get("turns_on_failure") for _, b in losses]

        calls = sum(int(b.get("tool_calls") or 0) for _, b in blocks)
        invalid = sum(int(b.get("tool_calls_invalid") or 0) for _, b in blocks)
        validity_measured = [b for _, b in blocks
                             if b.get("tool_validity_pct") is not None]

        headrooms_fail = [b["context"].get("headroom_pct") for _, b in losses
                          if isinstance(b.get("context"), dict)]
        headrooms_ok = [b["context"].get("headroom_pct") for _, b in wins
                        if isinstance(b.get("context"), dict)]
        measured_headroom = [h for h in headrooms_fail + headrooms_ok if h is not None]
        n_ctx_seen = next((b["context"].get("n_ctx") for _, b in blocks
                           if isinstance(b.get("context"), dict) and b["context"].get("n_ctx")),
                          ctx_size)

        vram_server = [b["vram"].get("peak_server_mib") for _, b in blocks
                       if isinstance(b.get("vram"), dict)]
        vram_card = [b["vram"].get("peak_card_mib") for _, b in blocks
                     if isinstance(b.get("vram"), dict)]
        co_tenants: dict[str, int] = {}
        contaminated = 0
        for _, b in blocks:
            vram = b.get("vram") or {}
            if vram.get("contaminated"):
                contaminated += 1
            for tenant in vram.get("co_tenants") or []:
                name = str(tenant.get("name") or "?")
                co_tenants[name] = max(co_tenants.get(name, 0), int(tenant.get("peak_mib") or 0))

        success_walls = [b.get("wall_seconds") for _, b in wins if b.get("wall_seconds") is not None]
        out[model] = {
            "attempts": len(rs),
            "scored": len(scored),
            "successes": len(wins),
            "failures": len(losses),
            "score_macro": (statistics.fmean([r.get("score") or 0.0 for r, _ in scored])
                            if scored else None),
            # 1. throughput
            "tokens_per_second_median": percentile(processed, 0.5),
            "tokens_per_second_mean": mean_or_none(processed),
            "throughput_n": len([p for p in processed if p is not None]),
            "probe": probe or {},
            # 2. wall time
            "wall_mean": mean_or_none(walls),
            "wall_p50": percentile(walls, 0.5),
            "wall_p90": percentile(walls, 0.9),
            "wall_max": max([w for w in walls if w is not None], default=None),
            "wall_n": len([w for w in walls if w is not None]),
            # cost per successful task
            "seconds_per_success": (statistics.fmean(success_walls) if success_walls else None),
            # 3. turns
            "turns_to_success_mean": mean_or_none(turns_success),
            "turns_to_success_median": percentile(turns_success, 0.5),
            "turns_on_failure_mean": mean_or_none(turns_failure),
            "turns_success_n": len([t for t in turns_success if t is not None]),
            "turns_failure_n": len([t for t in turns_failure if t is not None]),
            # 4. tool-call validity
            "tool_calls": calls,
            "tool_calls_invalid": invalid,
            # Pooled over calls, not a mean of per-attempt rates: the printed counts beside it are
            # pooled too, so the percentage has to be the ratio of those two numbers or the line
            # contradicts itself. Per-attempt spread stays available for a reader who wants it.
            "tool_validity_pct": (100.0 * (calls - invalid) / calls
                                  if calls and validity_measured else None),
            "tool_validity_per_attempt_mean": (statistics.fmean(
                [b["tool_validity_pct"] for b in validity_measured])
                if validity_measured else None),
            "tool_validity_n": len(validity_measured),
            "invalid_tool_names": _top_invalid(blocks),
            # 5. context headroom
            "n_ctx": n_ctx_seen,
            "headroom_pct_at_failure_median": percentile(
                [h for h in headrooms_fail if h is not None], 0.5),
            "headroom_pct_at_success_median": percentile(
                [h for h in headrooms_ok if h is not None], 0.5),
            "headroom_pct_all_median": percentile(measured_headroom, 0.5),
            "headroom_n": len(measured_headroom),
            "peak_prompt_tokens": max(
                [b["context"].get("peak_prompt_tokens") for _, b in blocks
                 if isinstance(b.get("context"), dict)
                 and isinstance(b["context"].get("peak_prompt_tokens"), int)], default=None),
            # 6. peak VRAM
            "vram_peak_server_mib": max([v for v in vram_server if v is not None], default=None),
            "vram_peak_card_mib": max([v for v in vram_card if v is not None], default=None),
            "vram_n": len([v for v in vram_card if v is not None]),
            "vram_co_tenants": [{"name": k, "peak_mib": v} for k, v in sorted(co_tenants.items())],
            "vram_contaminated_attempts": contaminated,
            "units": UNITS,
        }
    return out


def _top_invalid(blocks: list[tuple[dict, dict]]) -> list[dict]:
    counts: dict[str, int] = {}
    for _, b in blocks:
        for name in b.get("invalid_tool_names") or []:
            counts[name] = counts.get(name, 0) + 1
    return [{"tool": k, "invalid": v} for k, v in sorted(counts.items(), key=lambda kv: -kv[1])][:5]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _load_run(run_id: str) -> tuple[list[dict], dict]:
    path = RUNS_DIR / run_id / "results.json"
    if not path.exists():
        raise FileNotFoundError(f"no such run: {run_id} ({path})")
    records = json.loads(path.read_text())
    manifest_path = RUNS_DIR / run_id / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception:  # noqa: BLE001
            manifest = {}
    return records, manifest


def ctx_size_of(manifest: dict) -> int | None:
    """The context window this run ran under, from the manifest rather than from a guess."""
    rep = (manifest.get("reproducibility") or {}).get("server") or {}
    ctx = rep.get("ctx")
    if isinstance(ctx, int):
        return ctx
    notes = manifest.get("notes") or {}
    value = str(notes.get("ctx") or "")
    return int(value) if value.isdigit() else None


def format_run(run_id: str, *, records: list[dict] | None = None,
               manifest: dict | None = None) -> str:
    if records is None or manifest is None:
        records, manifest = _load_run(run_id)
    ctx = ctx_size_of(manifest)
    decision = manifest.get("decision") or {}
    probe = decision.get("throughput_probe") or {}
    agg = aggregate(records, ctx_size=ctx, probe=probe)
    lines: list[str] = []
    lines.append("=" * 92)
    lines.append("DECISION METRICS (workstream 7: what the decision turns on, not the pass rate)")
    lines.append("=" * 92)
    for model, a in agg.items():
        lines.append(f"model: {model}")
        lines.append(f"  attempts                 {a['attempts']} "
                     f"({a['successes']} at 1.000, {a['failures']} below)")
        # 1. throughput
        lines.append("  throughput (tok/s)       "
                     f"{_fmt(a['tokens_per_second_median'], 1)} processed tokens/s"
                     f"   (tok/s, uncached prompt plus output over model time, "
                     f"attempt-derived, n={a['throughput_n']})")
        if probe.get("ok"):
            lines.append(f"  throughput (tok/s)       "
                         f"decode {_fmt(probe.get('decode_tok_s'), 1)}, prefill "
                         f"{_fmt(probe.get('prompt_tok_s'), 1)}"
                         f"   (tok/s, server probe: {probe.get('prompt_tokens')} prompt tokens, "
                         f"{probe.get('decode_tokens')} output tokens)")
        elif probe:
            lines.append("  throughput (tok/s)       server probe not measured: "
                         + str(probe.get("reason", "no reason recorded"))[:60])
        else:
            lines.append("  throughput (tok/s)       server probe not measured"
                         "   (this run predates the probe or used a cloud route)")
        # 2. wall time
        lines.append(f"  wall time (s)            "
                     f"{_fmt(a['wall_mean'], 1)} mean, {_fmt(a['wall_p50'], 1)} p50, "
                     f"{_fmt(a['wall_p90'], 1)} p90, {_fmt(a['wall_max'], 1)} max"
                     f"   (s per attempt, n={a['wall_n']})")
        # cost per success
        lines.append(f"  seconds per success (s)  {_fmt(a['seconds_per_success'], 1)}"
                     f"   (s per successful task, n={a['successes']})")
        # 3. turns
        lines.append(f"  turns to success (turns) {_fmt(a['turns_to_success_mean'], 1)} mean, "
                     f"{_fmt(a['turns_to_success_median'], 1)} median"
                     f"   (turns, n={a['turns_success_n']})")
        lines.append(f"  turns spent failing (turns) {_fmt(a['turns_on_failure_mean'], 1)} mean"
                     f"   (turns, n={a['turns_failure_n']})")
        # 4. tool validity
        lines.append(f"  tool-call validity (%)   {_fmt(a['tool_validity_pct'], 1)}"
                     f"   (% valid of {a['tool_calls']} calls, "
                     f"{a['tool_calls_invalid']} invalid)")
        # 5. context headroom
        if a["headroom_n"]:
            ctx_txt = (f"of a {a['n_ctx']}-token window" if a["n_ctx"]
                       else "of an unknown window")
            lines.append(f"  context headroom (%)     "
                         f"{_fmt(a['headroom_pct_at_failure_median'], 1)} at failure, "
                         f"{_fmt(a['headroom_pct_at_success_median'], 1)} at success"
                         f"   (% of the window still free {ctx_txt}, n={a['headroom_n']})")
            if a["peak_prompt_tokens"]:
                lines.append(f"  context used (tokens)    {a['peak_prompt_tokens']} peak"
                             f"   (tokens, largest single prompt seen on the server)")
        else:
            lines.append("  context headroom (%)     unmeasured"
                         f"   (no /slots samples: n=0, the window is "
                         f"{a['n_ctx'] or 'unknown'} tokens)")
        # 6. peak VRAM
        if a["vram_n"]:
            lines.append(f"  peak VRAM (MiB)          "
                         f"{_fmt(a['vram_peak_server_mib'], 0)} server, "
                         f"{_fmt(a['vram_peak_card_mib'], 0)} card"
                         f"   (MiB, n={a['vram_n']} attempts sampled)")
        else:
            lines.append("  peak VRAM (MiB)          unmeasured"
                         "   (no SENTINEL samples: this run predates the sampler)")
        if a["vram_co_tenants"]:
            lines.append("  VRAM co-tenants (MiB)    "
                         + ", ".join(f"{t['name'].split('/')[-1]} {t['peak_mib']} MiB"
                                     for t in a["vram_co_tenants"]))
        if a["invalid_tool_names"]:
            lines.append("  rejected tools           "
                         + ", ".join(f"{t['tool']} x{t['invalid']}"
                                     for t in a["invalid_tool_names"]))
        lines.append("")
    lines.append("-" * 92)
    lines.append("Units: throughput tok/s, wall time s, turns turns, tool validity %, "
                 "context headroom %, VRAM MiB.")
    lines.append("Peak VRAM and context headroom are GPU-EXCLUSIVE measurements: they are the "
                 "bench's own")
    lines.append("only when nothing else holds the card (qwen, comfyui, voicestudio, luxtts). "
                 "Co-tenants seen")
    lines.append(f"while sampling are listed above, and a co-tenant holding "
                 f"{VRAM_CO_TENANT_LIMIT_MIB} MiB or more marks the attempt contaminated.")
    lines.append("An unmeasured metric is printed as unmeasured with its reason, never as zero: a "
                 "run older")
    lines.append("than the sampler has no samples, and that is a gap in the data, not a "
                 "measurement of zero.")
    lines.append("=" * 92)
    return "\n".join(lines)


def _short_model(name: str, limit: int = 26) -> str:
    """A column label a reader can tell apart: the gguf's file name, not two identical paths."""
    base = name.rsplit("/", 1)[-1]
    if base.lower().endswith(".gguf"):
        base = base[:-5]
    return (base[:limit - 1] + "~") if len(base) > limit else base


def format_comparison(run_a: str, run_b: str,
                      model_a: str | None = None, model_b: str | None = None) -> str:
    """The cost column, with accuracy beside it: seconds and turns per successful task."""
    rec_a, man_a = _load_run(run_a)
    rec_b, man_b = _load_run(run_b)
    probe_a = (man_a.get("decision") or {}).get("throughput_probe") or {}
    probe_b = (man_b.get("decision") or {}).get("throughput_probe") or {}
    agg_a = aggregate(rec_a, ctx_size=ctx_size_of(man_a), probe=probe_a)
    agg_b = aggregate(rec_b, ctx_size=ctx_size_of(man_b), probe=probe_b)

    def pick(agg: dict, wanted: str | None, run_id: str) -> tuple[str, dict]:
        if wanted:
            for model, a in agg.items():
                if model == wanted:
                    return model, a
            raise ValueError(f"{run_id} has no model {wanted!r}; it has "
                             + ", ".join(sorted(agg)) )
        if len(agg) != 1:
            raise ValueError(f"{run_id} measured {len(agg)} models; pass --model-a/--model-b "
                             "so a comparison is never stitched from the wrong row")
        return next(iter(agg.items()))

    name_a, a = pick(agg_a, model_a, run_a)
    name_b, b = pick(agg_b, model_b, run_b)
    label_a, label_b = _short_model(name_a), _short_model(name_b)

    lines = ["=" * 92,
             "COST PER SUCCESSFUL TASK (the decision column, with accuracy beside it)",
             "=" * 92,
             f"{'metric (unit)':32} {label_a:>27} {label_b:>27} {'delta':>12}",
             "-" * 92]

    def row(label: str, va, vb, digits: int = 1, unit: str = "") -> None:
        sa = _fmt(va, digits, unit)
        sb = _fmt(vb, digits, unit)
        delta = (f"{vb - va:+.{digits}f}" if (va is not None and vb is not None) else "n/a")
        lines.append(f"{label:32} {sa:>27} {sb:>27} {delta:>12}")

    row("score (macro)", a["score_macro"], b["score_macro"], 3)
    row("seconds per success (s)", a["seconds_per_success"], b["seconds_per_success"], 1)
    row("wall p50 (s)", a["wall_p50"], b["wall_p50"], 1)
    row("wall p90 (s)", a["wall_p90"], b["wall_p90"], 1)
    row("turns to success (turns)", a["turns_to_success_mean"], b["turns_to_success_mean"], 1)
    row("turns spent failing (turns)", a["turns_on_failure_mean"], b["turns_on_failure_mean"], 1)
    row("throughput, processed (tok/s)", a["tokens_per_second_median"],
        b["tokens_per_second_median"], 1)
    row("throughput, decode (tok/s)", (a["probe"] or {}).get("decode_tok_s"),
        (b["probe"] or {}).get("decode_tok_s"), 1)
    row("throughput, prefill (tok/s)", (a["probe"] or {}).get("prompt_tok_s"),
        (b["probe"] or {}).get("prompt_tok_s"), 1)
    row("tool-call validity (%)", a["tool_validity_pct"], b["tool_validity_pct"], 1)
    row("context headroom at failure (%)", a["headroom_pct_at_failure_median"],
        b["headroom_pct_at_failure_median"], 1)
    row("peak VRAM, server (MiB)", a["vram_peak_server_mib"], b["vram_peak_server_mib"], 0)
    row("peak VRAM, card (MiB)", a["vram_peak_card_mib"], b["vram_peak_card_mib"], 0)
    lines.append("-" * 92)
    lines.append("Units are in the metric names. A negative delta means the second run is lower.")
    lines.append("Peak VRAM and context headroom are GPU-EXCLUSIVE: compare them only between runs "
                 "made with")
    lines.append("the card to themselves. The report names any co-tenant it saw.")
    lines.append("=" * 92)
    return "\n".join(lines)


def report_single_run(run_id: str) -> str:
    return format_run(run_id)


def report_comparison(run_a: str, run_b: str, *, model_a: str | None = None,
                      model_b: str | None = None) -> str:
    return format_comparison(run_a, run_b, model_a=model_a, model_b=model_b)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Decision metrics beyond pass rate (hbench ws7)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_rep = sub.add_parser("report", help="decision metrics for one run")
    p_rep.add_argument("run_id")

    p_cmp = sub.add_parser("compare", help="cost comparison of two runs")
    p_cmp.add_argument("run_a")
    p_cmp.add_argument("run_b")
    p_cmp.add_argument("--model-a")
    p_cmp.add_argument("--model-b")

    p_smp = sub.add_parser("sample", help="sample the model server and SENTINEL for N seconds "
                                         "and print what was seen (GPU-exclusive)")
    p_smp.add_argument("--base-url", help="OpenAI base_url of a llama.cpp server, e.g. "
                                          "http://192.168.1.15:8096/v1")
    p_smp.add_argument("--api-key")
    p_smp.add_argument("--sentinel-url", default=DEFAULT_SENTINEL,
                       help=f"SENTINEL telemetry (default {DEFAULT_SENTINEL}); pass '' to disable")
    p_smp.add_argument("--seconds", type=float, default=20.0)
    p_smp.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    p_smp.add_argument("--json", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "report":
        print(format_run(args.run_id))
        return 0
    if args.cmd == "compare":
        print(format_comparison(args.run_a, args.run_b, model_a=args.model_a,
                                model_b=args.model_b))
        return 0

    sampler = DecisionSampler(base_url=args.base_url, api_key=args.api_key,
                             sentinel_url=args.sentinel_url or None,
                             interval=args.interval).start()
    deadline = time.monotonic() + max(args.seconds, args.interval)
    while time.monotonic() < deadline:
        time.sleep(min(args.interval, max(deadline - time.monotonic(), 0.05)))
    summary = sampler.stop()
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0
    peaks = summary["peaks"]
    print(f"sources: {', '.join(summary['sources']) or 'none'}")
    print(f"samples: {summary['samples']} over {peaks.get('seconds')} s "
          f"every {summary['interval_seconds']:g} s")
    print(f"context: {_fmt(peaks.get('kv_peak_tokens'), 0)} peak prompt tokens of "
          f"{peaks.get('n_ctx') or '?'} tokens "
          f"({_fmt(100.0 * (1 - (peaks['kv_peak_tokens'] / peaks['n_ctx']))
                   if peaks.get('kv_peak_tokens') and peaks.get('n_ctx') else None, 1)} % free)")
    print(f"vram:    {peaks.get('vram_peak_server_mib')} MiB server, "
          f"{peaks.get('vram_peak_card_mib')} MiB card, baseline "
          f"{peaks.get('vram_baseline_card_mib')} MiB")
    tenants = peaks.get("co_tenants") or []
    print("co-tenants: " + (", ".join(f"{t['name'].split('/')[-1]} {t['peak_mib']} MiB"
                                      for t in tenants) or "none")
          + ("  CONTAMINATED (card shared)" if peaks.get("contaminated") else ""))
    for source, error in summary["errors"].items():
        print(f"unavailable: {source}: {error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
