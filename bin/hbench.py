#!/usr/bin/env python3
"""
hbench - isolated model benchmarking for the Hermes harness.

Designed around one finding from Harness-Bench (arXiv 2605.27922, Table 2):
the Hermes harness averages 22.6 turns and 139.7K tokens per task, 2-4x every
other harness. A model that looks fine in a 6-turn prompt can fail here. So
every task is graded on workspace state, and turns and tokens are logged
alongside pass/fail rather than instead of it.

Isolation model (four layers, each one load-bearing):
  1. HERMES_HOME  - every run gets its own home. Memory, sessions, skills,
                    logs and profile config land there and are deleted after.
                    This is the layer that actually prevents cross-run
                    contamination; hermes-bench's own "session isolation"
                    only sets cwd.
  2. cwd          - a fresh cp -a of the task fixture. The pristine fixture
                    is never written to, so run N and run N+1 see identical
                    bytes.
  3. grading      - assertions over the resulting file tree, never the chat
                    reply. The model gets credit for committed artifacts only.
  4. network      - tasks may not reach the live internet. Anything that
                    needs data reads it from the fixture.

Concurrency: exactly one run at a time. Every ``run`` takes an exclusive flock on
``runs/.hbench.lock`` for its whole duration and waits, bounded, for the previous
run to finish. The rule existed only as prose in the task cards until 2026-10-22,
when the dispatcher claimed two bench cards in the same second and both ran. It is
now a property of the tool; see BenchLock for why flock rather than a pid file.

Usage:
    hbench.py list
    hbench.py run --model ollama-cloud/deepseek-v4.1-flash:1017 --tasks all
    hbench.py run --model A --model B --tasks all --repeat 3
    hbench.py report <run_id>
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import statistics
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(os.environ.get("HBENCH_ROOT", "/root/hermes-bench"))
TASKS_DIR = ROOT / "tasks"
RUNS_DIR = ROOT / "runs"
# A grader assertion that passes on an untouched fixture is not evidence that the model did
# anything. Task 05 has two of them by design, because "did not mutate the inputs" is the
# property it measures, so they cannot count as work evidence.
UNCHANGED_CHECK = re.compile(r"unmodified|unchanged|not modified|untouched|preserved|"
                             r"not destruct", re.I)
# The per-attempt tool-call transcript, staged inside the work directory only while a judge
# runs, so a judge can grade HOW the work was done (a single corrective call after a
# rejection; three edits batched into one turn) without changing the `judge.py <workdir>`
# contract. See stage_trace. It is written before the judge starts and removed after, so it
# never appears in the model's output tree or the "did the workspace change" comparison.
TRACE_FILENAME = ".hbench_trace.jsonl"
# One bench, one run. Every `run` holds this lock from before its first task to after
# its last; see BenchLock.
LOCK_PATH = RUNS_DIR / ".hbench.lock"
# How long a queued run waits for the current one before giving up. A full matrix on a
# slow model can hold the bench for over an hour, so the default is generous but finite;
# override per call with --lock-wait or globally with HBENCH_LOCK_WAIT.
DEFAULT_LOCK_WAIT = float(os.environ.get("HBENCH_LOCK_WAIT") or 7200)
REAL_ENV = Path("/root/.hermes/.env")

# Resolve the CLI once, by absolute path. systemd units and cron jobs often have a
# PATH that does not include ~/.local/bin, which made the runner die with
# FileNotFoundError: 'hermes' the first time it was put under systemd.
# Always use the public launcher, never the internal one. shutil.which("hermes")
# inside an agent turn can resolve to hermes-agent/.hermes/bin/hermes, which is a
# different entry point with different argv handling.
HERMES_BIN = "/root/.local/bin/hermes"
if not Path(HERMES_BIN).exists():
    for cand in ("/usr/local/bin/hermes", "/usr/bin/hermes"):
        if Path(cand).exists():
            HERMES_BIN = cand
            break

# The live install. Runs share it but are isolated by flags and session tagging;
# see EnvSandbox for why relocating HERMES_HOME is not viable.
REAL_HOME = Path("/root/.hermes")

# Credential source for providers the isolated launch cannot authenticate by itself.
# Only the provider block is taken from here (endpoint + credential), never rules, memory,
# defaults or any other user setting.
REAL_AUTH_STORE = Path("/root/.hermes/auth.json")

# Providers whose inference credential lives ONLY in the auth store. A --ignore-user-config
# launch resolves none of them, and they do not read an <PROVIDER>_API_KEY from the
# environment either (their registry entry is oauth_device_code / *_oauth with no
# api_key_env_var). Measured 2026-10-22: `nous` with a portal API key sitting in the
# credential pool still fails auth, because the pool's nous gate only admits a NAS invoke
# JWT (credential_pool.runtime_api_key) while the inference API itself accepts the portal
# `sk-nous-` key directly. Providers not in this set already resolve from .env or the pool
# under isolation (ollama-cloud does), so their launch is left exactly as it was.
AUTH_STORE_PROVIDERS = frozenset({"nous", "openai-codex", "xai-oauth", "qwen-oauth", "minimax-oauth"})


def load_provider_credential(provider: str) -> dict | None:
    """The provider block for *provider* from the auth store's credential pool.

    Returns {base_url, api_key, label, source} or None. Reads the pool row's raw
    ``access_token`` on purpose: for nous the pool's ``runtime_api_key`` is empty by design
    (it demands a NAS invoke JWT), while ``access_token`` holds the portal key the inference
    API accepts. Never touches config.yaml, .env, memory or rules.
    """
    if not REAL_AUTH_STORE.exists():
        return None
    try:
        store = json.loads(REAL_AUTH_STORE.read_text())
    except Exception as exc:
        print(f"  provider config: auth store unreadable ({exc})", file=sys.stderr)
        return None
    rows = (store.get("credential_pool") or {}).get(provider) or []
    for row in rows:
        if not isinstance(row, dict):
            continue
        token = str(row.get("access_token") or row.get("api_key") or "").strip()
        base = str(row.get("inference_base_url") or row.get("base_url") or "").strip().rstrip("/")
        if not token or not base:
            continue
        return {
            "base_url": base,
            "api_key": token,
            "label": str(row.get("label") or row.get("id") or "unnamed"),
            "source": f"auth.json credential_pool.{provider}",
        }
    return None


def load_custom_provider_config(name: str) -> dict | None:
    """The provider block for a ``custom_providers`` entry in the user's config.yaml.

    A local llama-server is registered as a custom provider (endpoint plus api_key), and the
    benchmark child runs with ``--ignore-user-config``, so that registration is invisible to
    it. Reading the single entry is the narrow exception: endpoint and credential only, no
    rules, no memory, no defaults, and the mode is stamped in the manifest like any other
    credential path.
    """
    try:
        import yaml  # noqa: PLC0415 - optional at import time, required only for this mode
        cfg = yaml.safe_load((REAL_HOME / "config.yaml").read_text()) or {}
    except Exception as exc:
        print(f"  provider config: cannot read config.yaml ({exc})", file=sys.stderr)
        return None
    for entry in (cfg.get("custom_providers") or []):
        if not isinstance(entry, dict) or entry.get("name") != name:
            continue
        base = str(entry.get("base_url") or "").strip().rstrip("/")
        key = str(entry.get("api_key") or "").strip()
        if not base:
            return None
        return {"base_url": base, "api_key": key, "label": f"config:{name}",
                "source": "config.yaml custom_providers"}
    return None


def resolve_provider_config(mode: str, provider: str | None,
                            custom: str | None = None,
                            base_url: str | None = None,
                            api_key: str | None = None) -> dict | None:
    """Decide whether this run carries its provider block explicitly.

    mode ``off``        never; the child resolves the provider itself (the pre-existing path)
    mode ``auth-store`` always, when the pool holds a row for the provider (error if not)
    mode ``auto``       only for AUTH_STORE_PROVIDERS, which cannot authenticate under
                        --ignore-user-config any other way

    Whatever happens is recorded in the run manifest and printed in the report header, so a
    score is never compared to a run whose credential path was different without saying so.
    """
    if mode == "off" or not provider:
        return None
    if mode == "inline":
        if not base_url:
            sys.exit("--provider-config inline needs --base-url (and usually --api-key)")
        return {"base_url": base_url.rstrip("/"), "api_key": (api_key or "").strip(),
                "label": "inline", "source": "command line"}
    if mode == "config":
        if not custom:
            sys.exit("--provider-config config needs --provider-custom <name> "
                     "(a custom_providers entry in ~/.hermes/config.yaml)")
        cred = load_custom_provider_config(custom)
        if cred is None:
            sys.exit(f"--provider-config config: no custom_providers entry named {custom!r} "
                     f"with a base_url in {REAL_HOME / 'config.yaml'}")
        return cred
    wanted = mode == "auth-store" or (mode == "auto" and provider in AUTH_STORE_PROVIDERS)
    if not wanted:
        return None
    cred = load_provider_credential(provider)
    if cred is None:
        if mode == "auth-store":
            sys.exit(
                f"--provider-config auth-store: no usable credential for provider {provider!r} in "
                f"{REAL_AUTH_STORE} (credential_pool.{provider}). Add one with "
                f"`hermes auth add {provider} ...`, or run with --provider-config off."
            )
        print(f"  provider config: no stored credential for {provider}, leaving resolution to the child",
              file=sys.stderr)
        return None
    return cred


# --------------------------------------------------------------------------
# Mutual exclusion: one bench run at a time
# --------------------------------------------------------------------------

class BenchLock:
    """Exclusive lock over the whole bench, held for one run.

    Why this exists: every run drives ``hermes chat`` children against the LIVE
    install, so concurrent runs contend on its ``state.db`` write lock and their wall
    clocks are not separable measurements. Until now the one-at-a-time rule lived only
    in the task cards' prose, and on 2026-10-22 the dispatcher claimed two bench cards
    in the same second (t_d6266a67 and t_27300d8e, both 17:59:31) and both ran. So the
    rule moves into the tool.

    Why flock and not a pid file: the kernel drops an ``flock`` when the holding
    process exits, including on SIGKILL, so a killed or crashed run cannot wedge the
    queue behind a stale lock. The file is left behind on purpose and its contents are
    only a courtesy note for the next waiter; liveness is decided by the kernel, never
    by the file's existence.

    Granularity is per run, not per task: the lock covers every model, task and repeat
    of one invocation, because that is the unit whose wall clock must not interleave.
    """

    def __init__(self, path: Path = LOCK_PATH, wait_limit: float = DEFAULT_LOCK_WAIT):
        self.path = Path(path)
        self.wait_limit = max(0.0, float(wait_limit))
        self.waited = False
        self.waited_seconds = 0.0
        self.holder_before: dict | None = None
        self.acquired_at: str | None = None
        self._fd: int | None = None

    # -- helpers ---------------------------------------------------------
    def _holder_note(self) -> dict | None:
        """Best-effort read of who the current holder says it is.

        Advisory only: the file may hold a note from a run that has since exited. It is
        read while a ``flock`` attempt is failing, which means some process does hold
        the lock; the pid is checked for liveness so a stale note is labelled as one.
        """
        try:
            raw = self.path.read_text().strip()
        except OSError:
            return None
        if not raw:
            return None
        head, _, rest = raw.partition(" ")
        try:
            pid = int(head)
        except ValueError:
            return {"raw": raw[:120]}
        alive = True
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            alive = False
        except PermissionError:
            alive = True
        return {"pid": pid, "since": rest.strip() or None, "alive": alive}

    def _write_note(self) -> None:
        assert self._fd is not None
        note = f"{os.getpid()} acquired {self.acquired_at}\n".encode()
        os.ftruncate(self._fd, 0)
        os.pwrite(self._fd, note, 0)

    # -- lifecycle -------------------------------------------------------
    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # O_RDWR without O_TRUNC: a waiter must be able to read the holder's note.
        fd = os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o644)
        started_waiting = time.monotonic()
        announced = False
        last_tick = 0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES):
                    os.close(fd)
                    raise
            elapsed = time.monotonic() - started_waiting
            if not announced:
                self.waited = True
                self.holder_before = self._holder_note()
                print(
                    f"bench busy: another hbench run holds {self.path}"
                    + (f" (pid {self.holder_before['pid']}, since {self.holder_before['since']})"
                       if self.holder_before and self.holder_before.get("pid") else "")
                    + f"; waiting up to {self.wait_limit:g}s for it to finish",
                    file=sys.stderr, flush=True,
                )
                announced = True
            if elapsed >= self.wait_limit:
                holder = self.holder_before or {}
                os.close(fd)
                sys.exit(
                    f"gave up after waiting {elapsed:.0f}s for the bench lock at {self.path}.\n"
                    f"Holder: {holder or 'unknown (note file empty or unreadable)'}.\n"
                    "Only one hbench run may hold the bench at a time, because concurrent runs\n"
                    "contend on the live install's state.db and their wall clocks are not\n"
                    "separable. Wait for that run to finish, or raise the bound with\n"
                    "  --lock-wait <seconds>   (or HBENCH_LOCK_WAIT in the environment).\n"
                    "A killed run releases the lock by itself: there is no stale lock to clear."
                )
            if elapsed - last_tick >= 60:
                last_tick = elapsed
                print(f"  still waiting for the bench lock ({elapsed:.0f}s)", file=sys.stderr,
                      flush=True)
            time.sleep(0.5)

        self.waited_seconds = round(time.monotonic() - started_waiting, 1) if self.waited else 0.0
        self.acquired_at = datetime.now(timezone.utc).isoformat()
        self._fd = fd
        self._write_note()
        if self.waited:
            print(f"bench lock acquired after {self.waited_seconds:g}s: this run starts now",
                  file=sys.stderr, flush=True)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            # Clear the note, then drop the lock by closing. Never unlink: another
            # waiter may already have the same inode open.
            os.ftruncate(self._fd, 0)
        except OSError:
            pass
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(self._fd)
        self._fd = None

    def describe(self) -> dict:
        """The manifest block. Records a wait so a measurement taken after one is identifiable."""
        return {
            "path": str(self.path),
            "mechanism": "flock(LOCK_EX) on the bench lock file, held for the whole run",
            "waited": self.waited,
            "waited_seconds": self.waited_seconds,
            "wait_limit_seconds": self.wait_limit,
            "holder_before": self.holder_before,
            "acquired_at": self.acquired_at,
            "pid": os.getpid(),
        }


# --------------------------------------------------------------------------
# Task model
# --------------------------------------------------------------------------

@dataclass
class Task:
    key: str
    path: Path
    prompt: str
    max_turns: int = 40
    run_budget: int = 420
    timeout: int = 480
    judge: Path | None = None
    expect: list[str] = field(default_factory=list)
    quality_gates: list = field(default_factory=list)
    # Optional fixture file (under the task's pristine workspace/) the harness injects as
    # turn 1 of the staged trace. Task 14 uses this for the rejected call it must repair.
    trace_preamble: str | None = None


def load_tasks(keys: list[str] | None, budget_scale: float = 1.0,
               timeout_scale: float | None = None) -> list[Task]:
    """Load tasks, optionally scaling their time budgets.

    A local model on consumer hardware generates at roughly 19 tokens/second, so a
    task that finishes in 50 seconds on a cloud model can take 8 minutes. Rather
    than maintain a forked task set for slow models, budgets are scaled at load
    time. The scale is recorded in the run manifest, because a raised budget is a
    different measurement and must not be silently compared to a default run.
    """
    tasks = []
    if not TASKS_DIR.is_dir():
        sys.exit(f"no tasks directory at {TASKS_DIR}")
    # Hard timeout = scaled budget + fixed slack for tool execution and grading.
    # A multiplicative slack is wrong here: at scale 8 it adds minutes of dead air.
    TIMEOUT_SLACK = 120
    for d in sorted(TASKS_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        spec_path = d / "task.json"
        if not spec_path.exists():
            print(f"  skip {d.name}: no task.json", file=sys.stderr)
            continue
        spec = json.loads(spec_path.read_text())
        # Integrity guard: a task whose prompt duplicates another task's prompt means
        # the fixture was overwritten (this happened once when a build cell rolled
        # back mid-write). Refuse to run a suite that cannot distinguish its tasks.
        prompt_file = d / "prompt.md"
        prompt = prompt_file.read_text().strip() if prompt_file.exists() else (spec.get("prompt") or "").strip()
        if not prompt:
            print(f"  skip {d.name}: empty prompt", file=sys.stderr)
            continue
        for prev in tasks:
            if prev.prompt == prompt:
                sys.exit(
                    f"INTEGRITY ERROR: task {d.name} has the same prompt as {prev.key}.\n"
                    f"The fixture is probably contaminated. Fix the task before running."
                )
        judge = d / "judge.py"
        tasks.append(Task(
            key=d.name,
            path=d,
            prompt=prompt,
            # Turn limits are NOT scaled: a model that needs more turns to finish is
            # a finding about the model, not about the hardware. Only wall-clock
            # budgets scale.
            max_turns=int(spec.get("max_turns", 40)),
            run_budget=int(spec.get("run_budget", 420) * budget_scale),
            timeout=int(spec.get("run_budget", 420) * budget_scale + TIMEOUT_SLACK),
            # an explicit --timeout-scale, when given, still wins for a single fixed ceiling

            judge=judge if judge.exists() else None,
            expect=spec.get("expect", []),
            # Binary quality gates (SPEC ws5): an ordered list of extra code checks the
            # runner evaluates after the judge, reported separately from the pass rate.
            quality_gates=spec.get("quality_gates", []),
            # Optional trace preamble: a fixture injected as turn 1 of the staged trace.
            trace_preamble=spec.get("trace_preamble"),
        ))
    if keys and keys != ["all"]:
        wanted = set(keys)
        missing = wanted - {t.key for t in tasks}
        if missing:
            sys.exit(f"unknown tasks: {sorted(missing)}")
        tasks = [t for t in tasks if t.key in wanted]
    return tasks


# --------------------------------------------------------------------------
# Isolation layer 1: HERMES_HOME
# --------------------------------------------------------------------------

# Benchmark session ids awaiting deletion. Collected during a run and flushed
# once at the end; see EnvSandbox.destroy for why this is not per-task.
_PENDING_CLEANUP: list[str] = []


def cleanup_benchmark_sessions(home: Path, ids: list[str]) -> int:
    """Archive this run's benchmark sessions. Returns the rows touched.

    Archiving, not deleting: ``sessions_runtime_ledger_delete_guard`` refuses to
    delete a session that went through the gateway, because it has runtime history
    in ``session_admissions``. That is why the old per-task ``destroy()`` never
    actually removed anything - it swallowed the sqlite3 error - and why every run
    spent 36 failing write transactions competing with the live gateway for the
    state.db lock. Archiving is allowed, it keeps the rows out of the user's
    session lists, and it is one cheap UPDATE.

    A short busy_timeout is deliberate: if the gateway is mid-write, leave the
    rows for the next run. ``source='tool'`` already hides them.
    """
    if not ids:
        return 0
    db = home / "state.db"
    if not db.exists():
        return 0
    touched = 0
    try:
        con = sqlite3.connect(str(db), timeout=2.0)
        try:
            con.execute("pragma busy_timeout=2000")
            for sid in ids:
                cur = con.execute(
                    "update sessions set archived = 1 where id = ? and source = ?",
                    (sid, EnvSandbox.SOURCE_TAG),
                )
                touched += cur.rowcount or 0
            con.commit()
        finally:
            con.close()
    except sqlite3.Error as exc:
        print(f"  cleanup deferred, {len(ids)} benchmark session(s) left unarchived: {exc}",
              file=sys.stderr)
    return touched


def install_fence() -> str | None:
    """Why new ``hermes chat`` children would be refused, or None when clear.

    An update marker or a held checkout lock makes the gateway answer
    ``draining: update_paused`` and every child exits in about a second. Without
    this check a fenced install is recorded as a full column of zeroes, which
    reads as a model finding when it is an install state. Same class of error as
    the ``--run-budget`` argv bug: the harness reported the model as 0.00.
    """
    try:
        agent_root = str(REAL_HOME / "hermes-agent")
        if agent_root not in sys.path:
            sys.path.insert(0, agent_root)
        from hermes_cli import update_lock  # type: ignore
    except Exception:
        return None
    try:
        marker = REAL_HOME / update_lock.MARKER_NAME
        if marker.exists():
            raw = marker.read_bytes()
            age = time.time() - marker.stat().st_mtime
            verdict, pid, _ = update_lock.judge_marker(raw)
            if raw and verdict in {"live", "ours"}:
                return f"update marker held (pid {pid}, verdict {verdict})"
            if not raw and age < update_lock.EMPTY_MARKER_GRACE_SECONDS:
                return "an update is publishing its marker right now"
        if update_lock.checkout_lock_held():
            return "the checkout lock is held by another process"
    except Exception:
        return None
    return None


class EnvSandbox:
    """Isolation without relocating HERMES_HOME.

    Why not a per-run HERMES_HOME: Hermes' launcher (a sh -c script) runs a lazy
    dependency bootstrap when the home has not been provisioned. Several concurrent
    runs contend on the install lock, the losers re-exec, and the re-exec rebuilds
    argv from a shell-quoted copy: "--format stream-json" then parses as a
    subcommand and the run dies with "'stream-json' is not a hermes command".
    A pre-warmed golden home does not help either, because a venv records its own
    store path in pyvenv.cfg and refuses to run relocated.

    So the isolation is done at the level that actually matters for comparability,
    all of which the CLI supports natively:
      * --ignore-rules       no AGENTS.md, SOUL.md, or memory injection
      * --ignore-user-config no ~/.hermes/config.yaml defaults bleeding in
      * --source tool        the session is tagged, keeping it out of user lists
      * fresh cwd            a copy of the fixture, per task
      * per-run session ids  deleted from state.db after grading

    Verified: MEMORY.md and USER.md hashes are unchanged across runs, and only
    sessions tagged source='tool' are created, which this class removes afterwards.
    """

    # Sessions are tagged with this so cleanup can find exactly the benchmark rows.
    SOURCE_TAG = "tool"

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.home = REAL_HOME
        self.session_ids: list[str] = []

    def env(self, model: str | None, provider: str | None) -> dict:
        e = dict(os.environ)
        # Strip inherited SESSION IDENTITY, not everything. This process runs inside
        # a live Hermes turn, so os.environ carries HERMES_SESSION_*, HERMES_AGENT,
        # HERMES_INTERACTIVE, HERMES_CWD and friends. Passing those into a child
        # `hermes chat` makes the CLI believe it is already inside an agent run and
        # it mis-parses argv: "--format stream-json" becomes "'stream-json' is not a
        # `hermes` command".
        #
        # Do NOT strip HERMES_QUIET / HERMES_STREAM_DRAIN_TIMEOUT: those control how
        # the child writes its output, and removing them makes it spawn a detached
        # grandchild, so the captured pipe comes back empty even though the task ran.
        for k in [k for k in e if k.startswith("HERMES_SESSION_")]:
            e.pop(k, None)
        for k in ("HERMES_AGENT", "HERMES_INTERACTIVE", "HERMES_CWD", "HERMES_PROFILE",
                  "HERMES_DELEGATED_CHILD_CONTEXT", "HERMES_CRON_SESSION",
                  "HERMES_KANBAN_TASK", "HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD",
                  "HERMES_UI_SESSION_ID"):
            e.pop(k, None)
        for k in [k for k in e if k.startswith("HERMES_RPC_")]:
            e.pop(k, None)
        e.pop("AI_AGENT", None)
        e["HERMES_HOME"] = str(self.home)
        # Keep TMPDIR inside the run so scratch files stay with the run.
        tmp = self.run_dir / "tmp"
        tmp.mkdir(exist_ok=True)
        e["TMPDIR"] = str(tmp)
        if model:
            e["HBENCH_MODEL"] = model
        if provider:
            e["HBENCH_PROVIDER"] = provider
        return e

    def destroy(self) -> None:
        """Queue this run's benchmark sessions for end-of-run cleanup.

        Deliberately does no database work. ``self.home`` is the LIVE home (see the
        class docstring), so every cleanup is a write transaction against the
        gateway's own state.db. Doing it after each task meant 36 of them
        interleaved with a live gateway: on 2026-10-22 a run held the state.db
        write lock for over 60s, the gateway could not append a message or refresh
        the user's turn lease, and the user's session was paused mid-turn with
        ``unknown_execution``. One bounded pass at the end has the same effect and
        a fraction of the exposure.
        """
        _PENDING_CLEANUP.extend(self.session_ids)
        self.session_ids = []


# --------------------------------------------------------------------------
# Isolation layer 4: no live network for task data
# --------------------------------------------------------------------------

def network_policy_note() -> str:
    return (
        "Do not fetch live data from the internet. Every file you need is "
        "already present in the working directory."
    )


# --------------------------------------------------------------------------
# Isolation layer 2: workspace fixture
# --------------------------------------------------------------------------

def read_session_stats(home: Path, session_id: str | None) -> dict | None:
    """Read turn/tool/token counts, and the route actually used, from state.db.

    Returns None if the DB or session is absent, which is itself a signal:
    it means the run did not write state into HERMES_HOME.

    ``model``, ``billing_provider`` and ``billing_base_url`` are the ground truth for
    "did this run use the model I asked for". A provider that silently falls back (nous
    auth failure -> ollama-cloud/glm) writes the fallback's model and billing provider
    here while the CLI still exits 0 with a plausible answer, so these columns are the
    only reliable detector.
    """
    db = home / "state.db"
    if not db.exists():
        return None
    cols = ("message_count, tool_call_count, input_tokens, output_tokens, api_call_count, "
            "model, billing_provider, billing_base_url")
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        if session_id:
            row = con.execute(
                f"select {cols} from sessions where id = ?", (session_id,)
            ).fetchone()
        else:
            # No session id in the child's stream (a timeout kills it before the closing
            # event carries one). Returning the newest session in the DB here attributed a
            # FOREIGN session's model, turns and tokens to this attempt: on 2026-11-01 a
            # timed-out local attempt inherited a cron session's ``gemma4:31b-cloud`` and was
            # then faulted as a provider substitution, so a real local run was thrown away as
            # "not a result for the requested model". Unknown is the honest answer.
            row = None

        con.close()
        return dict(row) if row else None
    except Exception:
        return None


def stage_workspace(task: Task, dest: Path) -> None:
    """cp -a from the pristine fixture so fixtures never mutate."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    src = task.path / "workspace"
    if src.is_dir():
        shutil.copytree(src, dest, symlinks=True)
    else:
        dest.mkdir(parents=True)


# --------------------------------------------------------------------------
# Reproducibility metadata: resolved at launch, enforced at publish
# --------------------------------------------------------------------------
# Workstream 1 of /mnt/agent_shared/hbench-robustness/SPEC.md. A manifest that records a
# build does not stop a series from spanning two builds, so the fields below are not
# optional annotations: `hbench_publish.MANIFEST_SCHEMA` requires every one of them and
# `hbench_publish.publish` refuses a series whose members disagree on any of them.
#
# Two fields are resolved rather than taken from the caller, because they are the two that
# went missing from a shipped series:
#   temperature   - read from the llama.cpp server's own /props, or from the exact rule the
#                   Hermes agent applies to the model/route. Never a guess, never absent.
#   chat template - the name and a hash of its text from /props, or an explicit statement
#                   that the provider owns it.
# When either cannot be resolved the manifest records `unresolved (...)` in the source
# field, which the schema rejects at publish time: the run stays usable as a scratch run,
# it just cannot become a published measurement.

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hbench_publish as publishlib  # noqa: E402
import result_key  # noqa: E402  (the one rule for a published row's identity, shared with tools/)
import hbench_uncertainty as uncertlib  # noqa: E402
import hbench_coresmoke as coresmoke  # noqa: E402
import hbench_metrics as metricslib  # noqa: E402
import hbench_quality_gates as qgates  # noqa: E402
import hbench_freeze as freezelib  # noqa: E402  (the frozen-instrument fence: see freeze)

HERMES_ROOT = Path(os.environ.get("HBENCH_HERMES_ROOT", "/root/.hermes/hermes-agent"))
# The sampler parameters a llama.cpp server reports in /props.default_generation_settings.
# Recorded in full rather than "temperature and seed", because a strict-format task can fail
# because one of the others ate a closing brace.
SAMPLER_PARAMS = (
    "temperature", "dynatemp_range", "dynatemp_exponent", "top_k", "top_p", "min_p",
    "xtc_probability", "xtc_threshold", "typical_p", "repeat_last_n", "repeat_penalty",
    "presence_penalty", "frequency_penalty", "dry_multiplier", "dry_base",
    "dry_allowed_length", "mirostat", "mirostat_tau", "mirostat_eta", "top_n_sigma",
    "seed", "n_ctx", "n_predict",
)


def http_json(url: str, api_key: str | None = None, timeout: float = 15.0) -> dict:
    """GET one JSON document. urllib, not curl, so no shell is involved."""
    import urllib.request
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed scheme
        return json.loads(resp.read().decode("utf-8"))


def read_server_props(base_url: str, api_key: str | None) -> dict:
    """The live llama.cpp server's resolved settings, from /props.

    `/props` is the authority for what the server will actually do, as opposed to what a
    variant name or a defaults file suggests. llama.cpp has nested the sampler parameters
    under `params` in newer builds and flattened them in older ones, so both shapes are read.
    """
    root = re.sub(r"/v1/?$", "", base_url.rstrip("/"))
    return http_json(root + "/props", api_key)


def sampler_from_props(props: dict) -> tuple[dict, list[str]]:
    """(resolved sampler chain, parameters the server did not report)."""
    dgs = props.get("default_generation_settings") or props
    params = dgs.get("params") if isinstance(dgs.get("params"), dict) else dgs
    chain = {}
    for key in SAMPLER_PARAMS:
        if key in params:
            chain[key] = params[key]
    if "n_ctx" not in chain and isinstance(dgs.get("n_ctx"), (int, float)):
        chain["n_ctx"] = dgs["n_ctx"]
    omitted = [k for k in SAMPLER_PARAMS if k not in chain]
    return chain, omitted


_AGENT_TEMPERATURE_PROBE = r"""
import json, sys
sys.path.insert(0, {root!r})
try:
    import hermes_bootstrap  # installs the dependency paths the agent itself uses
except Exception:
    pass
out = {{"kind": "failed", "reason": ""}}
try:
    from agent.auxiliary_client import _fixed_temperature_for_model, OMIT_TEMPERATURE
    fixed = _fixed_temperature_for_model({model!r}, {base_url!r}, {provider!r})
    if fixed is OMIT_TEMPERATURE:
        out = {{"kind": "omit", "reason": "the agent omits temperature for this model/route"}}
    elif fixed is None:
        out = {{"kind": "none", "reason": "the agent sends no temperature for this model/route"}}
    else:
        out = {{"kind": "value", "value": float(fixed), "reason": "a fixed temperature rule"}}
except Exception as exc:
    out = {{"kind": "failed", "reason": f"{{type(exc).__name__}}: {{exc}}"}}
print(json.dumps(out))
"""


def launcher_python() -> str | None:
    """The interpreter the installed `hermes` launcher actually runs, or None.

    Discovered rather than assumed: the install moved from a `venv/` tree to a bundled
    CPython, and a hard-coded path turns every cloud run into a reproducibility refusal.
    """
    try:
        proc = subprocess.run([HERMES_BIN, "--print-runtime-command"],
                              capture_output=True, text=True, timeout=60)
        argv = json.loads(proc.stdout.strip())
        exe = argv[0] if isinstance(argv, list) and argv else None
        return exe if exe and Path(exe).exists() else None
    except Exception:  # noqa: BLE001 - reported by the caller as a failed candidate
        return None


def probe_agent_temperature(model: str | None, base_url: str | None,
                            provider: str | None) -> dict:
    """What temperature the Hermes child will send, straight from the agent's own rule.

    The agent passes `temperature` only when a model/route rule fixes it, and omits the key
    otherwise so the provider default applies. That rule lives in
    `agent.auxiliary_client._fixed_temperature_for_model`, and the honest record is what that
    function returns, not a value this tool would have liked to see. Queried in a subprocess
    so importing the whole agent into the bench cannot perturb either one.

    The interpreter matters: hermes runs on its own bundle, and the bench's venv does not
    carry the agent's dependencies (`ruamel.yaml` is the first thing that fails), so the
    launcher's own interpreter is discovered from `hermes --print-runtime-command` and tried
    first, with the bench's interpreter only as a fallback.
    """
    code = _AGENT_TEMPERATURE_PROBE.format(root=str(HERMES_ROOT), model=model,
                                           base_url=base_url, provider=provider)
    candidates = []
    override = os.environ.get("HBENCH_HERMES_PYTHON")
    if override:
        candidates.append(override)
    runtime_python = launcher_python()
    if runtime_python:
        candidates.append(runtime_python)
    candidates.append(str(HERMES_ROOT / "venv" / "bin" / "python"))
    candidates.append(sys.executable)
    failures = []
    for interpreter in candidates:
        if not Path(interpreter).exists():
            failures.append(f"{interpreter}: missing")
            continue
        try:
            proc = subprocess.run([interpreter, "-c", code], capture_output=True,
                                  text=True, timeout=120)
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as exc:  # noqa: BLE001 - any failure is reported, never guessed at
            failures.append(f"{interpreter}: {type(exc).__name__}: {exc}")
            continue
        if payload.get("kind") != "failed":
            return payload
        failures.append(f"{interpreter}: {payload.get('reason')}")
    return {"kind": "failed", "reason": "; ".join(failures)}


_hermes_identity_cache: dict | None = None


def hermes_identity() -> dict:
    """The harness version and the checkout it came from."""
    global _hermes_identity_cache
    if _hermes_identity_cache is not None:
        return _hermes_identity_cache
    version = ""
    install = str(HERMES_ROOT)
    try:
        proc = subprocess.run([HERMES_BIN, "--version"], capture_output=True, text=True,
                              timeout=60)
        for line in proc.stdout.splitlines():
            if line.lower().startswith("install directory:"):
                install = line.split(":", 1)[1].strip()
            elif line.strip() and not version:
                version = line.strip()
    except Exception as exc:  # noqa: BLE001
        version = f"unresolved ({type(exc).__name__}: {exc})"
    _hermes_identity_cache = {"version": version, "install_dir": install}
    return _hermes_identity_cache


def task_bundle_hashes(tasks: list[Task]) -> dict:
    """Hash every file a task brings, so a changed prompt or grader cannot share a row.

    The prompt is part of the instrument in the same way the model is: editing prompt.md
    changes every score produced under it. Fixtures are hashed file by file rather than as a
    directory digest so a diff names the file that moved.
    """
    bundle: dict[str, dict[str, str]] = {}
    for task in tasks:
        files: dict[str, str] = {}
        for name in ("prompt.md", "task.json", "judge.py"):
            path = task.path / name
            if path.is_file():
                files[name] = publishlib.sha256_file(path)
        fixtures = {}
        for path in sorted((task.path / "workspace").rglob("*")) if (task.path / "workspace").is_dir() else []:
            if path.is_file():
                fixtures[path.relative_to(task.path / "workspace").as_posix()] = \
                    publishlib.sha256_file(path)
        if fixtures:
            files["workspace"] = publishlib.sha256_text(publishlib.canonical_json(fixtures))
        bundle[task.key] = files
    return bundle


def model_facts(model: str | None) -> dict:
    """Identity, and whatever of hash/architecture/quantization is readable from here.

    A gguf served from the PC is not readable from this container, so its hash is recorded
    as omitted rather than invented. That is a limit, and the record says so by name in
    `omitted`, which is checked by the schema.
    """
    identity = model or "(config default)"
    omitted: list[str] = []
    looks_like_file = identity.lower().endswith(".gguf") or "/" in identity
    facts = {"identity": identity,
             "source": "file" if looks_like_file else "provider",
             "sha256": None, "architecture": None, "parameter_count": None,
             "quantization": None, "omitted": omitted}
    candidate = Path(identity)
    if candidate.is_file():
        try:
            facts["sha256"] = publishlib.sha256_file(candidate)
        except OSError as exc:
            omitted.append("sha256")
            facts["sha256_detail"] = f"unreadable: {exc}"
    else:
        # The gguf is served by the PC; this container cannot read it, so the hash is named
        # as omitted instead of being invented. The file name and the server argv still say
        # which file it was, and two builds under one name would be caught by the argv.
        omitted.append("sha256")
    for name in ("architecture", "parameter_count", "quantization"):
        omitted.append(name)
    return facts


def build_child_cmd(model: str | None, provider: str | None,
                    provider_config: dict | None, reasoning: str | None,
                    max_turns: int, prompt: str,
                    allow_inline_scripts: bool = True) -> list[str]:
    """The exact argv the child is launched with. One builder, so the recorded flags and
    the executed flags cannot drift apart.

    ``allow_inline_scripts`` adds ``--yolo``. Why it is on by default: the child runs
    with ``--ignore-user-config``, which freezes the code defaults, and the code default
    for ``approvals.single_query_mode`` is ``deny``. No user is present to approve an
    inline ``python3 -c`` or a heredoc, so the guard blocked it EVERY time. On 2026-11-01
    the failure scanner counted 345 such blocks in 24h, 338 of them in bench children.
    Each block is a wasted turn against ``--max-turns`` and a distorted score: the model
    is measured on its ability to recover from a harness artifact, not on the task.

    ``--yolo`` bypasses the RECOVERABLE approval layer only. Hardline floors (rm -rf /,
    runtime self-delete, the sudo-stdin guard) still block, because ``check_dangerous_command``
    runs ``_floor_block`` BEFORE the yolo bypass. The write-then-run path was always
    available to the model, so this does not change what is being scored - it stops the
    harness from taxing the model for a guard nobody could answer.

    ``--yolo`` is a PROCESS-level flag on the child only. It does not touch the live
    profile, the gateway's own approvals, or any other session.
    """
    cmd = [HERMES_BIN, "chat"]
    if model:
        cmd += ["-m", model]
    if provider:
        cmd += ["--provider", provider]
    if provider_config:
        cmd += ["--base-url", provider_config["base_url"], "--api-key", provider_config["api_key"]]
    if reasoning:
        cmd += ["--reasoning", reasoning]
    if allow_inline_scripts:
        cmd += ["--yolo"]
    cmd += [
        "--quiet", "--format", "stream-json", "--oneshot",
        "--source", "tool",          # keeps benchmark runs out of user session lists
        "--ignore-rules",            # no AGENTS.md / SOUL.md / memory injection
        "--ignore-user-config",      # ignore ~/.hermes/config.yaml defaults
        "--max-turns", str(max_turns),
        # --run-budget was removed from `hermes chat` (2026-10-22): the classic/one-shot
        # launch now runs through the canonical gateway client (hermes_cli/gateway_chat.py,
        # _UNSUPPORTED) which REFUSES the flag with exit 2 and points at
        # agent.run_budget_seconds in config.yaml. That route is gateway-global, not per
        # task, and --ignore-user-config forces code defaults, so the per-task wall clock is
        # enforced by this runner's own subprocess timeout (task.timeout = budget + slack).
        # Cost: the agent no longer gets the 80%-elapsed wrap-up notice, so a slow model is
        # hard-killed instead of being told to finish. Recorded per result so the spread is
        # readable, and counted in the report.
        "--pass-session-id",
        "-q", prompt,
    ]
    return cmd


def redacted_argv(cmd: list[str], prompt_index: int | None = None) -> list[str]:
    """The argv with the credential and the task prompt removed, everything else verbatim."""
    out = []
    for i, arg in enumerate(cmd):
        if i and cmd[i - 1] == "--api-key":
            out.append("<redacted>")
        elif prompt_index is not None and i == prompt_index:
            out.append("<task prompt>")
        else:
            out.append(arg)
    return out


def resolve_reproducibility(*, models: list[str | None], provider: str | None,
                            notes: dict[str, str], tasks: list[Task],
                            base_url: str | None, api_key: str | None,
                            reasoning: str | None,
                            provider_config: dict | None) -> dict:
    """Everything a reader needs to re-run this measurement, resolved against reality.

    Raises SystemExit when a local server's /props cannot be read: a run whose server
    settings are unknown must not start, which is the same rule `local_bench.sh` applies to
    the argv, applied to the sampler and the chat template.
    """
    local = bool(base_url)

    # --- sampler and chat template -----------------------------------------
    props: dict = {}
    sampler: dict = {"temperature": None, "temperature_source": "", "chain": {}, "seed": None,
                     "omitted": []}
    template: dict = {"name": "", "sha256": None, "source": "", "kwargs": "", "omitted": []}

    if local:
        try:
            props = read_server_props(base_url, api_key)
        except Exception as exc:  # noqa: BLE001
            sys.exit(
                f"refusing to run: could not read {base_url.rstrip('/')}/props "
                f"({type(exc).__name__}: {exc}).\n"
                "The sampler chain and the chat template are read from the server rather than "
                "assumed, because llama.cpp defaults change between builds and a sampler can eat "
                "a closing brace on a strict-format task. A run that cannot say what the server "
                "will do must not be published as a measurement."
            )
        chain, omitted = sampler_from_props(props)
        sampler.update(chain=chain, omitted=sorted(set(omitted)),
                       temperature=chain.get("temperature"), seed=chain.get("seed"),
                       temperature_source="/props default_generation_settings "
                                          "(llama-server, resolved not assumed)")
        if sampler["temperature"] is None:
            sampler["omitted"] = sorted(set(sampler["omitted"]) | {"temperature"})
        if sampler["seed"] is None:
            sampler["omitted"] = sorted(set(sampler["omitted"]) | {"seed"})

        text = (props.get("chat_template") or "").strip()
        template["source"] = "llama-server /props"
        if text:
            template["name"] = (notes.get("chat-template") or props.get("chat_template_name")
                                or f"server template, {len(text)} chars, sha256 "
                                   f"{publishlib.sha256_text(text)[:12]}")
            template["sha256"] = publishlib.sha256_text(text)
        else:
            template["name"] = notes.get("chat-template") or "unresolved (server reported no template)"
            template["omitted"] = ["sha256"]
        template["kwargs"] = notes.get("chat-template-kwargs") or "(none passed: no --chat-template-kwargs)"
    else:
        fixed = probe_agent_temperature(models[0] if models else None, base_url, provider)
        if fixed.get("kind") == "value":
            sampler["temperature"] = float(fixed["value"])
            sampler["temperature_source"] = (
                f"agent fixed-temperature rule for this model/route: {fixed.get('reason')}")
        elif fixed.get("kind") in ("omit", "none"):
            sampler["omitted"] = ["temperature"]
            sampler["temperature_source"] = (
                f"{fixed.get('reason')}; the client sends no temperature, so the provider's own "
                f"default applies and cannot be read back")
        else:
            sampler["temperature_source"] = f"unresolved (probe failed: {fixed.get('reason')})"
            sampler["omitted"] = ["temperature"]
        sampler["chain"] = {"temperature": sampler["temperature"],
                            "sampled_by": "provider default" if sampler["temperature"] is None
                                          else "client",
                            "note": "the server's own sampler chain is not readable over an "
                                    "OpenAI-compatible cloud endpoint"}
        sampler["omitted"] = sorted(set(sampler["omitted"]) | {"seed"})
        template["source"] = "provider-managed"
        template["name"] = "provider-managed (the client sends messages, not a template)"
        template["kwargs"] = ("(none: the client sends no --chat-template-kwargs; the provider "
                              "owns the template)")
        template["omitted"] = ["sha256"]

    # --- server ------------------------------------------------------------
    if local:
        server = {
            "management": notes.get("server-management") or "local llama-server, started for this run",
            "argv": notes.get("server-argv"),
            "build": notes.get("build"),
            "variant": notes.get("variant"),
            "ctx": int(notes["ctx"]) if (notes.get("ctx") or "").isdigit() else None,
            "kv_cache_k": notes.get("kv-cache-k"),
            "kv_cache_v": notes.get("kv-cache-v"),
            "omitted": [],
        }
        if server["ctx"] is None:
            ctx = (props.get("default_generation_settings") or {}).get("n_ctx") or props.get("n_ctx")
            server["ctx"] = int(ctx) if isinstance(ctx, (int, float)) else None
        for key in ("argv", "build", "variant", "ctx", "kv_cache_k", "kv_cache_v"):
            if server[key] is None:
                server["omitted"].append(key)
        if server["omitted"]:
            sys.exit(
                "refusing to run: this run drives a local server but its manifest cannot "
                "record: " + ", ".join(server["omitted"]) + ".\n"
                "local_bench.sh reads these from the live process; a run started by hand must "
                "pass them, for example:\n"
                "  --note server-argv='llama-server ...' --note build='b4567 abcdef1' "
                "--note variant=nothink --note ctx=8192 "
                "--note kv-cache-k=q8_0 --note kv-cache-v=q8_0\n"
                "server-argv, build and variant are read from `bench info`; ctx and the KV "
                "cache types come from the same payload."
            )
    else:
        server = {"management": "provider-managed", "argv": None, "build": None,
                  "variant": None, "ctx": None, "kv_cache_k": None, "kv_cache_v": None,
                  "omitted": ["argv", "build", "variant", "ctx", "kv_cache_k", "kv_cache_v"]}

    # --- agent -------------------------------------------------------------
    identity = hermes_identity()
    template_cmd = build_child_cmd(
        models[0] if models else None, provider, provider_config, reasoning,
        tasks[0].max_turns if tasks else 30, tasks[0].prompt if tasks else "")
    agent = {
        "hermes_version": identity["version"],
        "install_dir": identity["install_dir"],
        "cli_flags": redacted_argv(template_cmd, prompt_index=len(template_cmd) - 1),
        "toolsets": ["(no -t flag: the child's own default tool set under --ignore-user-config)"],
        "reasoning": reasoning,
        "reasoning_source": "explicit flag" if reasoning else "child default",
        "omitted": [] if reasoning else ["reasoning"],
        "rules_disabled": "ignored (--ignore-rules: no AGENTS.md, SOUL.md or memory injection)",
        "user_config_disabled": "ignored (--ignore-user-config)",
    }

    # --- environment -------------------------------------------------------
    import platform
    environment = {
        "os": platform.platform(),
        "python": f"{platform.python_implementation()} {platform.python_version()} "
                  f"({sys.executable})",
        "machine": f"{platform.node()} {platform.machine()} kernel {platform.release()}",
    }

    return {
        "schema_version": publishlib.SCHEMA_VERSION,
        "agent": agent,
        "chat_template": template,
        "sampler": sampler,
        "server": server,
        "model": model_facts(models[0] if models else None),
        "environment": environment,
        "task_bundle": task_bundle_hashes(tasks),
    }


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def parse_notes(pairs: list[str]) -> dict[str, str]:
    """One note per ``--note``, which is repeatable.

    Deliberately not split on commas or semicolons: a value may be a sentence with punctuation in
    it ("model default; the server had no --reasoning off") and splitting turned that into a second
    note with a nonsense key. Repeat the flag instead of packing several into one string.
    """
    out: dict[str, str] = {}
    for raw in pairs or []:
        piece = str(raw).strip()
        if not piece:
            continue
        if "=" in piece:
            k, v = piece.split("=", 1)
        elif ":" in piece:
            k, v = piece.split(":", 1)
        else:
            k, v = piece, "true"
        out[k.strip()] = v.strip()
    return out


def fault_class(record: dict, task: Task | None = None) -> str:
    """Which ceiling a non-zero exit hit, decided from the record rather than the code.

    ``-1`` is the runner's own subprocess timeout. An explicit iteration-limit message is a
    turn ceiling. Anything else that ended non-zero after doing the work is ``agent_exit``:
    a real exit whose cause the record does not name, and inventing one is worse than saying
    so. Counting tool calls against ``max_turns`` was tried and rejected on measurement: a
    clean exit-0 run of task 08 made 24 tool calls on a 20-turn task, so tool calls are not
    what the ceiling counts.
    """
    if record.get("exit_code") in (0, None):
        return ""
    if record.get("exit_code") == -1:
        return "timeout"
    blob = ((record.get("final_text") or "") + "\n" + (record.get("stdout_raw") or "")
            + "\n" + (record.get("stderr") or "")).lower()
    if any(k in blob for k in ("iteration limit", "max turns", "maximum turns",
                               "turn limit", "max_turns", "max-turns")):
        return "turn_limit"
    if not isinstance(record.get("turns"), int) or record["turns"] <= 0:
        return "no_work"
    return "agent_exit"


def workspace_changed(task: Task, work: Path) -> bool:
    """Did the run actually write anything, judged against the pristine fixture?

    This is the evidence that separates "the harness broke" from "the model worked and then
    ran out of turns". It has to be a byte comparison: a grader can pass checks on an
    untouched workspace (task 05's two "unmodified" assertions do exactly that), so the
    score alone cannot prove work happened.
    """
    def fingerprint(root: Path) -> dict[str, str]:
        out: dict[str, str] = {}
        if not root.is_dir():
            return out
        for p in root.rglob("*"):
            if (p.is_file() and "__pycache__" not in p.parts
                    and p.name not in ("result.json", TRACE_FILENAME)):
                out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
        return out
    return fingerprint(task.path / "workspace") != fingerprint(work)


def work_evidence(record: dict) -> bool:
    """Did this run actually do work, judged from its own record?

    The checks the child produced are the evidence for an old record: turns, a reply, or a
    grader assertion that only passes on content the model wrote. Assertions of the "records
    unmodified" family are deliberately not evidence, because they pass on a pristine
    fixture and task 05 has two of them, so an untouched workspace would otherwise score
    0.67 for doing nothing.
    """
    if isinstance(record.get("turns"), int) and record["turns"] > 0:
        return True
    if (record.get("final_text") or "").strip():
        return True
    for c in record.get("checks", []):
        name = (c.get("check") or "")
        if (c.get("pass") and name != "harness ran the requested model"
                and not UNCHANGED_CHECK.search(name)):
            return True
    return bool(record.get("workspace_changed"))


def artifact_from_checks(checks: list[dict]) -> float | None:
    """The score the run's own grader produced, with the runner's fault row removed.

    This is the non-destructive way to recover an artifact score from an archived run: the
    stored checks are what the grader saw at grading time. Re-running the grader against the
    workspace on disk is NOT the same measurement, because a timeout kills the CLI client
    while the agent keeps running in the gateway, so the workspace can still be rewritten
    after it was graded.
    """
    graded = [c for c in checks if (c.get("check") or "") != "harness ran the requested model"]
    if not graded:
        return None
    return round(sum(1 for c in graded if c.get("pass")) / len(graded), 4)


def mark_harness_failure(record: dict, detail: str, kind: str | None = None) -> None:
    """Record the fault beside the artifact score, never in place of it.

    Until 2026-10-27 this forced ``score = 0.0`` on any non-zero exit. That is right when the
    child never ran the work, and wrong when the model did the work and the agent loop then
    hit its own turn or wall ceiling: three of four task-08 runs ran the right model, produced
    a graded artifact and were published as zero, which reported a harness ceiling as a model
    capability. So the fault is now classified, kept in ``fault`` with the exit code beside
    it, and the graded artifact keeps its score whenever the record proves work happened.
    """
    kind = kind or fault_class(record) or "cli_error"
    prior = record.get("fault") or {}
    record["harness_failed"] = True
    record["fault"] = {
        "class": kind,
        "exit_code": record.get("exit_code"),
        "turns": record.get("turns"),
        "detail": detail[:400],
    }
    if prior.get("class") and prior["class"] != kind:
        # A run can be both: a timeout that then fell back to another model. The dominant
        # class is the one that decides the score; the other is kept rather than dropped.
        record["fault"]["prior"] = list(prior.get("prior") or []) + [prior["class"]]
    record["checks"] = [{"check": "harness ran the requested model", "pass": False,
                         "detail": detail[:400]}] + record.get("checks", [])
    artifact = record.get("artifact_score")
    if artifact is None:
        artifact = record.get("score") or 0.0
    record["artifact_score"] = artifact
    # No score at all for a substitution (the number belongs to another model) or for a run
    # with nothing behind it (an argv refusal, an untouched workspace), because there is no
    # artifact of this model's work to score.
    stall = kind in ("substitution", "no_work") or not work_evidence(record)
    record["score"] = 0.0 if stall else artifact
    record["score_source"] = "none" if stall else "artifact"


def route_substitution(record: dict, model: str | None, provider: str | None) -> str | None:
    """Why this run is not a measurement of what was asked for, or None when it is.

    A provider that cannot authenticate does not always fail loudly: when the user config is
    in play its fallback chain answers instead, with exit code 0 and a plausible reply. On
    2026-10-22 that turned a total Nous auth failure into something that looked like a working
    Nous run (session row: model ``glm-5.3-flash:cloud``, billing_provider ``ollama-cloud``,
    while ``--model`` asked for ``meituan/longcat-2.0:free``). state.db is the only place that
    substitution is visible, so it is checked on every run.
    """
    got_model = record.get("session_model")
    if model and got_model and got_model != model:
        return (f"ran {got_model!r} instead of the requested {model!r}: "
                f"provider fallback, so this is not a result for the requested model")
    billed = record.get("billing_provider")
    if provider and billed and billed != provider:
        return f"billed by provider {billed!r} instead of the requested {provider!r}"
    return None



def model_slug(model: str | None) -> str:
    """Short, stable, path-free directory name for a model id.

    Never derive this by replacing separators in a long id: the result still looks
    like a path and agents reconstruct it with real slashes. Keep it short, keep the
    hash so two ids that slug to the same stem stay distinct.
    """
    import pathlib
    raw = model or "config-default"
    stem = pathlib.Path(raw).stem if ("/" in raw or raw.endswith(".gguf")) else raw
    stem = re.sub(r"[^A-Za-z0-9.]+", "-", stem).strip("-.")[:24]
    return "%s-%s" % (stem or "model", hashlib.sha1(raw.encode()).hexdigest()[:6])

def run_one(task: Task, model: str | None, provider: str | None,
            repeat: int, run_id: str, provider_config: dict | None = None,
            reasoning: str | None = None, manifest_stamp: str | None = None,
            instrument_stamp: str | None = None,
            allow_inline_scripts: bool = True,
            sampler: "metricslib.DecisionSampler | None" = None,
            ctx_size: int | None = None,
            config: dict | None = None) -> dict:
    # The model MUST be part of the path: without it, running model B after model A
    # overwrites A's results for the same task key, silently destroying the matrix.
    # It must NOT be the full id. A model served from a GGUF has an id that is a real
    # path, and swapping its separators for underscores produces a directory name that
    # still reads as a path: the child agent reconstructed it from the sandbox and wrote
    # to /tasks/_home_chris_Agent_Fast/qwen36-drafters/... instead of its work dir,
    # losing 3 of 36 attempts on 2026-10-30 while every short-id model was unaffected.
    # A short slug plus a hash of the full id keeps models distinct and removes the trap.
    safe_model = model_slug(model)
    task_dir = RUNS_DIR / run_id / "tasks" / safe_model / task.key / f"r{repeat}"
    if task_dir.exists():
        shutil.rmtree(task_dir)
    task_dir.mkdir(parents=True)

    work = task_dir / "work"
    stage_workspace(task, work)

    sandbox = EnvSandbox(task_dir)
    record: dict = {
        "task": task.key,
        "model": model,
        "repeat": repeat,
        "run_id": run_id,
        # Which credential path this run used. Never the key itself: label and endpoint only.
        "provider_config": (None if not provider_config else {
            "source": provider_config["source"],
            "label": provider_config["label"],
            "base_url": provider_config["base_url"],
        }),
        "isolation": ("full (--ignore-rules, --ignore-user-config)"
                      + (" + provider block loaded explicitly" if provider_config else "")
                      + " + --yolo (inline scripts allowed; hardline floors still apply)"),
        "started": datetime.now(timezone.utc).isoformat(),
        "workdir": str(work),
        # Provenance for comparability: the wall clock is the runner's subprocess timeout
        # because `hermes chat` no longer accepts --run-budget (see the argv comment below).
        "run_budget_seconds": task.run_budget,
        "timeout_seconds": task.timeout,
        "budget_mode": "runner_timeout",
        # Explicit reasoning level, or None when the child's own default applied. Recorded so a
        # score is never compared across reasoning regimes without the difference being visible.
        "reasoning": reasoning,
        # The two provenance links that make this row checkable: which manifest it was
        # produced under, and which instrument that manifest describes. `publish` refuses a
        # row whose fingerprint disagrees with its own manifest or with a published sibling.
        "manifest_sha256": manifest_stamp,
        "instrument_fingerprint": instrument_stamp,
        # The configuration this attempt was measured under, carried on the record so the
        # published row key can be model plus configuration rather than model alone. Read from
        # the manifest's own recorded provenance; a field the manifest did not record says
        # "unrecorded" here rather than being guessed. See publish/tree/tools/result_key.py.
        "config": config or result_key.config_block({}, None),
    }

    prompt = task.prompt + "\n\n" + network_policy_note()
    cmd = build_child_cmd(model, provider, provider_config, reasoning, task.max_turns, prompt,
                          allow_inline_scripts=allow_inline_scripts)
    # Provenance without the secret: the exact argv, credential redacted and the task prompt
    # replaced by a placeholder, because the prompt is already hashed in the task bundle and a
    # manifest per attempt would triple the run directory.
    record["cli_flags"] = redacted_argv(cmd, prompt_index=len(cmd) - 1)

    t0 = time.monotonic()
    # Cut point for the decision-metrics sampler: everything sampled from here to the end of the
    # attempt belongs to this attempt's window.
    sample_mark = sampler.mark() if sampler else None
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(work),
            env=sandbox.env(model, provider),
            capture_output=True,
            text=True,
            timeout=task.timeout,
        )
        record["exit_code"] = proc.returncode
        record["stdout_raw"] = proc.stdout
        record["stderr"] = proc.stderr[-4000:] if proc.stderr else ""
    except subprocess.TimeoutExpired as exc:
        record["exit_code"] = -1
        record["stderr"] = f"harness timeout after {task.timeout}s"
        record["stdout_raw"] = (exc.stdout or b"").decode("utf-8", "replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
    record["wall_seconds"] = round(time.monotonic() - t0, 1)

    usage = parse_stream(record["stdout_raw"])
    record.update(usage)
    if record.get("session_id"):
        sandbox.session_ids.append(record["session_id"])

    # Turn and tool counts live in the isolated state.db. Read them before the
    # sandbox is destroyed; this is also the last proof that the run wrote its
    # state into HERMES_HOME and not into the live install.
    db_stats = read_session_stats(sandbox.home, record.get("session_id"))
    if db_stats:
        record["turns"] = db_stats.get("message_count")
        record["db_tool_calls"] = db_stats.get("tool_call_count")
        record["db_input_tokens"] = db_stats.get("input_tokens")
        record["db_output_tokens"] = db_stats.get("output_tokens")
        # The route the child actually ran on, straight from its own session row.
        record["session_model"] = db_stats.get("model")
        record["billing_provider"] = db_stats.get("billing_provider")
        record["billing_base_url"] = db_stats.get("billing_base_url")
        # How many model calls the attempt actually made. Turns count messages; this counts
        # requests, which is the denominator a throughput or a cost per call needs.
        record["api_calls"] = db_stats.get("api_call_count")
        if record.get("tool_calls") in (None, 0):
            record["tool_calls"] = db_stats.get("tool_call_count")

    # cleanup before grading so a huge home does not skew disk checks
    sandbox.destroy()

    record["score"], record["checks"] = grade(task, work, record=record)
    # Quality gates (SPEC ws5): extra binary code checks, evaluated in order and stopped at
    # the first failure. Stored separately from the pass rate as record["quality_gates"] and
    # a secondary record["quality_score"]; they never change the judge score above.
    qgates.apply_quality_gates(task, work, record)
    # The artifact score is kept separately from the reported score: a fault below decides
    # whether it survives. Byte comparison against the fixture is the proof that work
    # happened, because a grader can pass checks on an untouched workspace.
    record["artifact_score"] = record["score"]
    record["workspace_changed"] = workspace_changed(task, work)
    record["score_source"] = "artifact"
    record["harness_failed"] = False
    # A run that reported a CLI error is not a result, whatever the files look like.
    # Without this, an argv/bootstrap failure that still left a workspace behind was
    # recorded as a pass, which is how a broken harness reported 100%.
    if record.get("exit_code") not in (0,):
        reason = (record.get("stream_error") or record.get("final_text")
                  or record.get("stderr") or "").strip()
        mark_harness_failure(record, f"exit {record.get('exit_code')}: {reason[:280]}",
                             kind=fault_class(record, task) or "cli_error")
    # A clean exit is not proof either: a provider fallback answers with exit 0 on a
    # different model, which scored as the requested model until 2026-10-22.
    substitution = route_substitution(record, model, provider)
    if substitution:
        mark_harness_failure(record, substitution, kind="substitution")
    record["finished"] = datetime.now(timezone.utc).isoformat()
    # Decision metrics (SPEC workstream 7). Two of the six are sampled (context headroom, peak
    # VRAM) and come from the window this attempt occupied; the other four are derived from the
    # record itself. Every metric names its unit, and an unmeasured one names the reason instead
    # of reporting a zero, because a zero reads as a measurement.
    window = sampler.window(sample_mark) if (sampler and sample_mark) else {}
    try:
        record["decision"] = metricslib.attempt_decision(
            record, window=window, ctx_size=ctx_size)
    except Exception as exc:  # never lose an attempt to the metrics
        record["decision"] = {"error": f"decision metrics failed: {exc}"}
    (task_dir / "result.json").write_text(json.dumps(record, indent=2))
    return record


def parse_stream(raw: str) -> dict:
    """Pull turns, tokens, tool calls and session id out of stream-json."""
    out = {
        "turns": None, "tokens_in": None, "tokens_out": None,
        "tool_calls": 0, "tool_names": [], "session_id": None,
        "final_text": "", "parse_errors": 0, "stream_error": None,
    }
    tool_names: list[str] = []
    texts: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            out["parse_errors"] += 1
            continue
        t = ev.get("type") or ev.get("event") or ""
        low = t.lower()
        if isinstance(ev.get("error"), str) and ev["error"].strip():
            # The provider's own refusal ("HTTP 404: This model is no longer free...") lives
            # here, not in stderr, and it is the detail a fault report has to carry.
            out["stream_error"] = ev["error"].strip()
        # Count the call, not the pair. This used to match any event whose type contained
        # "tool" and had a name, which counted the `tool_use` and the `tool_result` of the
        # same call, so every archived row reports twice the calls the child made (12 for 6).
        # The tool-call validity metric divides by calls, so the denominator has to be right.
        if metricslib.is_tool_call_event(ev):
            tool_names.append(ev["name"])
        if ev.get("session_id"):
            out["session_id"] = ev["session_id"]
        for k in ("turns", "turn_count", "num_turns"):
            if isinstance(ev.get(k), int):
                out["turns"] = ev[k]
        for k in ("input_tokens", "prompt_tokens"):
            if isinstance(ev.get(k), int):
                out["tokens_in"] = ev[k]
        for k in ("output_tokens", "completion_tokens"):
            if isinstance(ev.get(k), int):
                out["tokens_out"] = ev[k]
        # The authoritative usage block lives on the final `result` event:
        # {"tokens":{"input":N,"output":N,"total":N,"cache_read":N,"cache_write":N}}
        tok = ev.get("tokens")
        if isinstance(tok, dict):
            if isinstance(tok.get("input"), int):
                out["tokens_in"] = tok["input"]
            if isinstance(tok.get("output"), int):
                out["tokens_out"] = tok["output"]
            if isinstance(tok.get("total"), int):
                out["tokens_total"] = tok["total"]
            if isinstance(tok.get("cache_read"), int):
                out["cache_read"] = tok["cache_read"]
        u = ev.get("usage")
        if isinstance(u, dict):
            if isinstance(u.get("input_tokens", u.get("prompt_tokens")), int):
                out["tokens_in"] = u.get("input_tokens", u.get("prompt_tokens"))
            if isinstance(u.get("output_tokens", u.get("completion_tokens")), int):
                out["tokens_out"] = u.get("output_tokens", u.get("completion_tokens"))
        if low in ("result", "final", "agent_end", "message") and isinstance(ev.get("text"), str):
            texts.append(ev["text"])
        elif low == "assistant" and isinstance(ev.get("text"), str):
            texts.append(ev["text"])
    out["tool_calls"] = len(tool_names)
    out["tool_names"] = tool_names
    out["final_text"] = "\n".join(texts)[-3000:]
    # the stream is the record; keep it on disk rather than in the summary
    return out


# --------------------------------------------------------------------------
# Per-attempt turn trace: let a judge grade HOW the work was done
# --------------------------------------------------------------------------
# A judge has always received only the final work directory, so a task whose
# capability is a property of the interaction (a single corrective call after a
# rejection; three edits batched into one turn) could only be graded on the end
# state. This stages the attempt's tool-call transcript into a fixed path inside
# the work directory at grade time, so such a judge reads it through the
# unchanged `judge.py <workdir>` contract. A judge that ignores the file is
# unaffected: it is written right before the judge starts and removed right
# after, so it never lands in the model's output tree and never shows up in the
# "did the workspace change" comparison (workspace_changed excludes it too).


def iter_stream_events(raw: str):
    """Yield each JSON event of a stream-json capture, in order."""
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if isinstance(ev, dict):
            yield ev


def extract_trace(raw: str) -> list[dict]:
    """The tool_use / tool_result transcript of one attempt, tagged by turn.

    A *turn* is a maximal run of tool_use events not separated by a tool_result:
    the calls the model emitted before any of them returned. That is exactly the
    "several calls in one turn" property task 17 measures, and it is recovered
    from the stream alone, so it needs no field the capture does not already
    carry. Text, system and result events are dropped: they are not tool
    activity. The turn index is deterministic for a given stream.
    """
    events: list[dict] = []
    turn = 0
    seen_result = False          # a result after the current turn's calls closes it
    seq = 0
    for ev in iter_stream_events(raw):
        kind = str(ev.get("type") or ev.get("event") or "").lower()
        if kind == "tool_use":
            if turn == 0 or seen_result:
                turn += 1
                seen_result = False
            seq += 1
            events.append({
                "turn": turn,
                "seq": seq,
                "kind": "tool_use",
                "name": ev.get("name"),
                "tool_call_id": ev.get("tool_call_id"),
                "input": ev.get("input"),
            })
        elif kind == "tool_result":
            seq += 1
            events.append({
                "turn": turn if turn else 1,   # a result belongs to the turn that issued it
                "seq": seq,
                "kind": "tool_result",
                "name": ev.get("name"),
                "tool_call_id": ev.get("tool_call_id"),
                "is_error": ev.get("is_error"),
                "output": ev.get("output"),
                "duration_ms": ev.get("duration_ms"),
            })
            if turn:
                seen_result = True
    return events


def load_trace_preamble(task: "Task") -> dict | None:
    """The rejected turn the harness injects as trace turn 1, or None.

    Read from the task's pristine fixture (``task.path/workspace/<file>``), never
    from the model's working copy, so the injection is harness-controlled and a
    tampered working file cannot forge it. The record is explicitly tagged
    ``injected`` so a reader can always tell a synthetic turn from a real one.
    """
    name = getattr(task, "trace_preamble", None)
    if not name:
        return None
    try:
        data = json.loads((task.path / "workspace" / name).read_text())
    except Exception:
        return None
    sent = data.get("tool_call_sent") or {}
    return {
        "name": sent.get("tool") or data.get("tool"),
        "input": sent.get("args"),
        "error": data.get("harness_error"),
        "source": f"task fixture: {name}",
    }


def build_trace_text(events: list[dict], preamble: dict | None = None) -> str:
    """Serialize the trace as JSONL, with the injected turn 1 first when present."""
    lines: list[str] = []
    offset = 0
    if preamble:
        lines.append(json.dumps({
            "turn": 1,
            "seq": 0,
            "kind": "injected_rejection",
            "injected": True,
            "source": preamble.get("source"),
            "name": preamble.get("name"),
            "rejected": True,
            "input": preamble.get("input"),
            "error": preamble.get("error"),
        }, sort_keys=True))
        offset = 1
    for ev in events:
        rec = dict(ev)
        rec["turn"] = (rec.get("turn") or 0) + offset
        lines.append(json.dumps(rec, sort_keys=True))
    return "\n".join(lines) + ("\n" if lines else "")


def stage_trace(work: Path, task: "Task", raw: str) -> Path:
    """Write the attempt's trace into the work directory; returns its path.

    Always writes the file, even when the capture has no tool events, so the file
    existing is the reliable signal that a judge is running under the trace
    contract (and an empty file means "no tool activity", not "no trace").
    """
    path = work / TRACE_FILENAME
    try:
        text = build_trace_text(extract_trace(raw), load_trace_preamble(task))
    except Exception:
        text = ""
    path.write_text(text)
    return path


# --------------------------------------------------------------------------
# Isolation layer 3: grade workspace state, never the reply
# --------------------------------------------------------------------------

def grade(task: Task, work: Path,
          record: dict | None = None) -> tuple[float, list[dict]]:
    """Run the task's judge against *work*.

    When *record* carries the attempt's stream capture, the attempt's tool-call
    transcript is staged at ``work/.hbench_trace.jsonl`` before the judge runs and
    removed afterward. The judge still receives exactly ``judge.py <workdir>``;
    the trace is a file inside that workdir, so an existing judge that ignores it
    keeps working unchanged.
    """
    if task.judge is None:
        checks = [{"check": "no judge.py", "pass": False, "detail": "task has no grader"}]
        return 0.0, checks
    trace_path: Path | None = None
    try:
        if record is not None and record.get("stdout_raw") is not None:
            trace_path = stage_trace(work, task, record.get("stdout_raw") or "")
        proc = subprocess.run(
            [sys.executable, str(task.judge), str(work)],
            capture_output=True, text=True, timeout=120,
        )
        payload = json.loads(proc.stdout)
        checks = payload.get("checks", [])
        passed = sum(1 for c in checks if c.get("pass"))
        score = passed / len(checks) if checks else 0.0
        return round(score, 4), checks
    except Exception as exc:
        return 0.0, [{"check": "grader error", "pass": False, "detail": str(exc)}]
    finally:
        if trace_path is not None:
            try:
                trace_path.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

def cmd_list(_args) -> None:
    tasks = load_tasks(None)
    print(f"{len(tasks)} tasks in {TASKS_DIR}\n")
    for t in tasks:
        print(f"  {t.key:32} max_turns={t.max_turns:<3} budget={t.run_budget}s")
        first = t.prompt.splitlines()[0] if t.prompt else ""
        print(f"      {first[:88]}")


def print_run_header(manifest: dict) -> None:
    """The mode line every report starts with, so a score never travels without its caveats."""
    iso = manifest.get("isolation") or {}
    print("=" * 92)
    print(f"run_id:  {manifest.get('run_id')}")
    print(f"provider: {manifest.get('provider') or '(config default)'}"
          f"   models: {', '.join(m for m in (manifest.get('models') or []) if m) or '(config default)'}")
    print(f"mode:     {iso.get('mode', 'unknown')}")
    print(f"isolation: {iso.get('user_config', 'unknown')}; {iso.get('rules', 'unknown')}")
    print(f"provider block: {iso.get('provider', 'unknown')}")
    print(f"tasks:    {len(manifest.get('tasks') or [])} task(s) x {manifest.get('repeat', 1)} repeat(s)"
          f"   budget_scale: {manifest.get('budget_scale', 1.0)}")
    # The instrument line. A run on a moved task set is a different measurement, and the reader
    # of a report should not have to open the manifest to find that out. Absent on runs written
    # before the fence existed (2026-11-01), so nothing is printed for those rather than a lie.
    if manifest.get("instrument_drift"):
        drift = manifest["instrument_drift"] or {}
        moved = drift.get("changed") or []
        print(f"instrument: DRIFT ALLOWED, {len(moved)} changed file(s) "
              f"({', '.join(moved[:3])}{', ...' if len(moved) > 3 else ''}); this row was not "
              f"measured on a frozen task set")
    elif manifest.get("instrument_frozen"):
        freeze_block = manifest.get("freeze") or {}
        print(f"instrument: frozen, {freeze_block.get('files', '?')} file(s), "
              f"sha256 {str(freeze_block.get('sha256'))[:16]}")
    proof = manifest.get("grader_proof") or {}
    if proof.get("skipped"):
        print("graders:  NOT PROVEN for this run (--no-prove)")
    elif proof.get("passed"):
        print(f"graders:  proven for all {proof.get('n', '?')} task(s) "
              f"({proof.get('seconds', '?')}s)")
    # The server flags, when the caller recorded them. A variant name alone does not say
    # whether the model was asked to think, and that difference is invisible in the scores:
    # it shows up as tokens per turn, which is not a column anyone compares across runs.
    notes = manifest.get("notes") or {}
    if notes.get("variant"):
        print(f"server:   variant {notes.get('variant')}  ctx {notes.get('ctx', '?')}"
              f"  kv {notes.get('kv-cache-k', '?')}/{notes.get('kv-cache-v', '?')}"
              f"  chat-template-kwargs {notes.get('chat-template-kwargs', '?')}")
        if notes.get("build"):
            print(f"build:    {notes.get('build')}")
    elif manifest.get("provider") == "custom":
        print("server:   (no variant recorded; this row cannot be compared with another)")
    # The resolved sampler and chat template, which is what an unlabelled series was missing.
    rep = manifest.get("reproducibility") or {}
    if rep.get("sampler"):
        s, t = rep["sampler"], rep.get("chat_template") or {}
        temp = s.get("temperature")
        shown = "omitted (provider default)" if temp is None else temp
        print(f"sampler:  temperature {shown}  seed {s.get('seed', '?')}"
              f"  chain {', '.join(f'{k}={v}' for k, v in (s.get('chain') or {}).items()) or '?'}")
        print(f"sampler:  {s.get('temperature_source', '?')}")
        sha = t.get("sha256")
        print(f"template: {t.get('name', '?')}"
              + (f"  sha256 {sha[:16]}" if sha else "  sha256 (not applicable)"))
    else:
        print("sampler:  (no reproducibility block; this manifest predates the schema and "
              "cannot be published)")
    # Only printed when there was a wait, so a run that started immediately reads exactly
    # as it did before the lock existed.
    lock = manifest.get("lock") or {}
    if lock.get("waited"):
        print(f"lock:     waited {lock.get('waited_seconds')}s for another run to release "
              f"{lock.get('path')}; the wall clocks below start after that wait")
    print("=" * 92)


def cmd_freeze(args) -> None:
    """Write the instrument freeze, or check the tree against it.

    Deliberately manual. Nothing in `run` ever calls the write path: an auto-freeze on the
    first run would bless whatever is on disk at that moment, which is the failure this
    exists to prevent. `--check` is what `run` and `local_bench.sh` call.
    """
    if getattr(args, "check", False):
        report = freezelib.check(ROOT)
        print(freezelib.describe(report))
        sys.exit(0 if report["ok"] else 1)
    doc = freezelib.write(ROOT)
    print(f"froze the instrument: {doc['files']} file(s) across {len(doc['tasks'])} task(s)")
    for name, facts in sorted(doc["tasks"].items()):
        print(f"  {name:32} {facts['files']:3} file(s)  tree {facts['tree_sha256'][:16]}"
              f"  budget {facts['budget_sha256'][:16]}")
    print(f"  tasks/{freezelib.FREEZE_SHA}  sha256 {doc['freeze_sha256']}")
    print(f"  tasks/{freezelib.FREEZE_JSON}  per-task tree and budget digests")
    print("Every run from now on cites that digest and refuses to start if the tree moves.")
    print("Re-freezing is a deliberate act: it rewrites the digest, so an older run reads as older.")


GRADER_PROOF = Path("tests") / "test_graders.py"
GRADER_PROOF_PASS = "ALL GRADERS VALID"
# The proof measured 3.8s on this box on 2026-11-01. The ceiling exists only so a hung proof
# cannot hold the bench forever; nothing is expected to come near it.
GRADER_PROOF_TIMEOUT = float(os.environ.get("HBENCH_PROOF_TIMEOUT") or 900)


def run_grader_proof(root: Path, timeout: float | None = None) -> dict:
    """Run tests/test_graders.py and report whether every grader is proven on this box.

    Empty fails, reference solution passes at 1.00, per task. That is the same proof
    agenttest re-runs in preflight and it costs seconds, so a run that skips it is choosing
    not to know whether its instrument can grade at all.
    """
    script = root / GRADER_PROOF
    expected = len(load_tasks(None))
    block: dict = {"passed": False, "n": 0, "stamp": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if not script.is_file():
        block["error"] = f"no grader proof at {script}"
        block["detail"] = ("A bench tree without tests/test_graders.py cannot show that its graders "
                           "score 0 on nothing and 1.00 on a correct answer, so the run is not proven.")
        return block
    started = time.monotonic()
    try:
        proc = subprocess.run([sys.executable, str(script)], cwd=str(root), capture_output=True,
                              text=True, timeout=timeout or GRADER_PROOF_TIMEOUT)
    except subprocess.TimeoutExpired:
        block["seconds"] = round(time.monotonic() - started, 1)
        block["error"] = f"the grader proof did not finish within {timeout or GRADER_PROOF_TIMEOUT:g}s"
        return block
    block["seconds"] = round(time.monotonic() - started, 1)
    output = (proc.stdout or "") + (proc.stderr or "")
    block["n"] = expected
    block["returncode"] = proc.returncode
    # One line per task in both directions: `  ok  <task> ...` and `  FAIL <task> ...`. A failure
    # names the task, which is the whole point of the proof running per task.
    failing = sorted({m.group(1) for m in re.finditer(r"^\s*FAIL\s+(\S+)", output, re.M)})
    if failing:
        block["failing"] = failing
    block["passed"] = proc.returncode == 0 and GRADER_PROOF_PASS in output
    if not block["passed"]:
        block["detail"] = "\n".join(line for line in output.strip().splitlines()[-25:])
    return block


def instrument_preflight(args) -> dict:
    """The two things that must be true before attempt one: frozen instrument, proven graders.

    Called once, from cmd_run, after the bench lock is held and before `execute_run` writes
    anything. It is never called again: a runner that re-checked between attempts would
    refuse halfway through a series after an unrelated edit, which wastes the series instead
    of preventing it.
    """
    fragment: dict = {}

    report = freezelib.check(ROOT)
    if not report["ok"]:
        drift = {
            "allowed": True,
            "changed": [item["path"] for item in report["changed"]],
            "added": list(report["added"]),
            "removed": list(report["removed"]),
            "freeze_sha256": report["frozen_sha256"],
            "files_checked": report["files"],
        }
        if not getattr(args, "allow_drift", False):
            sys.exit(freezelib.refusal(report))
        fragment["instrument_drift"] = drift
        print("INSTRUMENT DRIFT ALLOWED: this run is not on a frozen task set.")
        print(freezelib.describe(report))
        print("The manifest records instrument_drift; no reader can mistake this row for one "
              "measured on a frozen instrument.", flush=True)
    else:
        fragment["instrument_frozen"] = True
        fragment["freeze"] = {
            "file": f"tasks/{freezelib.FREEZE_SHA}",
            "sha256": report["frozen_sha256"],
            "files": report["files"],
            "tasks": report["tasks"],
        }
        print(f"instrument frozen: {report['files']} file(s) across {report['tasks']} task(s), "
              f"sha256 {report['frozen_sha256']}", flush=True)

    if getattr(args, "no_prove", False):
        fragment["grader_proof"] = {"skipped": True}
        print("grader proof SKIPPED (--no-prove): the manifest records that this run proved "
              "nothing about its graders.", flush=True)
        return fragment

    proof = run_grader_proof(ROOT)
    fragment["grader_proof"] = proof
    if not proof["passed"]:
        failing = proof.get("failing") or []
        named = f" Failing task(s): {', '.join(failing)}." if failing else ""
        sys.exit(
            "refusing to run: the grader proof did not pass on this box, so the instrument that "
            "would score this run is unproven." + named + "\n"
            f"  proof: {ROOT / GRADER_PROOF}  ({proof.get('seconds', 'n/a')}s, "
            f"exit {proof.get('returncode', 'n/a')})\n"
            + (proof.get("error") or "") + "\n"
            + (proof.get("detail") or "") + "\n"
            "Fix the grader, or run with --no-prove to start anyway; that stamps "
            "grader_proof.skipped in the manifest so the row says it was never proven."
        )
    print(f"grader proof: {GRADER_PROOF_PASS} ({proof['n']} task(s), {proof['seconds']}s)",
          flush=True)
    return fragment


def cmd_run(args) -> None:
    fence = install_fence()
    if fence:
        sys.exit(
            "refusing to run: the install is fenced, so every child would exit in about a\n"
            f"second with 'Gateway draining: update_paused' ({fence}).\n"
            "That produces a full column of zeroes that reads like a model result. Wait for\n"
            "the update to finish, confirm with\n"
            f"  ls -la {REAL_HOME / '.hermes-update-in-progress'}\n"
            "and re-run."
        )
    tasks = load_tasks(args.tasks, budget_scale=args.budget_scale,
                       timeout_scale=args.timeout_scale)
    if not tasks:
        sys.exit("nothing to run")
    provider_config = resolve_provider_config(args.provider_config, args.provider,
                                             args.provider_custom, args.base_url, args.api_key)
    # One bench run at a time, enforced here rather than in the task brief. Taken before
    # the run id is minted, so a run that waited carries the time it actually started.
    lock = BenchLock(wait_limit=args.lock_wait)
    lock.acquire()
    try:
        # The fence, once, before attempt one and while nothing else can be running. A drift
        # refusal here costs one hash pass; the same discovery after the first attempt costs
        # the series.
        preflight = instrument_preflight(args)
        execute_run(args, tasks, provider_config, lock, preflight)
    finally:
        lock.release()


def execute_run(args, tasks, provider_config: dict | None, lock: BenchLock,
                preflight: dict | None = None) -> None:
    models = args.model or [None]
    run_id = args.run_id or datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    notes = parse_notes(args.note)
    # Fail closed on an unrecorded local server. A published three-repeat series once mixed
    # repeats whose chat-template flags were unknown, because the manifest carried the server
    # name and the context but not the variant, and `reasoning_tokens` is zero for every local
    # gguf row so it could not answer the question either. The cost of catching that here is
    # one flag; the cost of missing it is a leaderboard row nobody can defend.
    if getattr(args, "base_url", None):
        missing = [k for k in ("variant", "chat-template-kwargs") if not notes.get(k)]
        if missing:
            sys.exit(
                "refusing to run: this run points at a server (--base-url) but records no "
                + " and no ".join(missing) + ".\n"
                "A local row is comparable to another local row only when the server flags travel\n"
                "with the numbers, and a variant name without its chat-template kwargs cannot say\n"
                "whether the model was asked to think. Add one flag each, for example:\n"
                "  --note variant=nothink-q8kv "
                "--note chat-template-kwargs='{\"enable_thinking\":false}'\n"
                "local_bench.sh does this for you from the live server's `bench info`."
            )

    if provider_config:
        provider_block = (f"explicit, from {provider_config['source']} "
                          f"(label {provider_config['label']!r}, endpoint {provider_config['base_url']}); "
                          f"config.yaml, .env, rules and memory still ignored")
        mode = (f"isolated + explicit provider block (--provider-config "
                f"{args.provider_config}"
                + (f" {args.provider_custom}" if args.provider_custom else "") + ")")
    else:
        provider_block = "resolved by the child from its own auth store / config"
        mode = "isolated (child resolves the provider itself)"

    # Everything a reread of this row needs, resolved against the live server and the live
    # harness rather than copied from the caller's intentions. Fails the run when a field
    # cannot be established, because an unrecordable run is a run that cannot be published.
    reproducibility = resolve_reproducibility(
        models=models, provider=args.provider, notes=notes, tasks=tasks,
        base_url=getattr(args, "base_url", None), api_key=getattr(args, "api_key", None),
        reasoning=getattr(args, "reasoning", None), provider_config=provider_config,
    )

    manifest = {
        "run_id": run_id,
        "budget_scale": args.budget_scale,
        "reasoning": getattr(args, "reasoning", None),
        "timeout_scale": (args.timeout_scale if args.timeout_scale is not None else args.budget_scale * 1.15),
        "models": models,
        "provider": args.provider,
        "tasks": [t.key for t in tasks],
        "repeat": args.repeat,
        "tasks_dir": str(TASKS_DIR),
        "started": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        # Provenance: exactly how this run was isolated, and where its credentials came from.
        "isolation": {
            "mode": mode,
            "user_config": "ignored (--ignore-user-config)",
            "rules": "ignored (--ignore-rules: no AGENTS.md, SOUL.md or memory injection)",
            "approvals": "child runs with --yolo: the recoverable approval layer is bypassed so an "
                         "inline `python3 -c` or heredoc is not blocked by a guard no user could "
                         "answer. Hardline floors (rm -rf /, runtime self-delete, sudo-stdin) still "
                         "block. Recorded so a run is never compared with one whose approval posture "
                         "differed.",
            "provider": provider_block,
            "provider_config_flag": args.provider_config,
            "sessions": f"tagged source={EnvSandbox.SOURCE_TAG}, archived after the run",
            "network": "offline for the tasks: no task reaches the live internet, fixtures are "
                       "read from the task bundle; the child's own provider endpoint is the only "
                       "route it may use",
            "toolsets": reproducibility["agent"]["toolsets"],
            "cwd": "a fresh copy of the task's workspace/ fixture per attempt, under "
                   f"{RUNS_DIR}/<run_id>/tasks/<model>/<task>/r<N>/work",
        },
        # Provenance: whether this run had to queue behind another one. A wall clock taken
        # after a wait is still this run's own time, but the wait is part of the story.
        "lock": lock.describe(),
        # Provenance: the invariant workstream. Validated below before a single model call.
        "reproducibility": reproducibility,
    }

    # Provenance: the instrument this run measured. `instrument_frozen` plus the freeze digest
    # when the task set matched the freeze at launch, or `instrument_drift` with the names that
    # moved when the operator overrode the refusal. Plus whether the graders were proven on this
    # box before attempt one. Written here, once, so every record can cite it.
    manifest.update(preflight or {})

    # Decision metrics (SPEC workstream 7). Two measurements cannot be derived after the fact:
    # the peak VRAM the run occupied, and the prefill/decode rate of the server it ran on. The
    # sampler polls while the run is in flight; the probe is a fixed, short generation taken once
    # per run and recorded in the manifest so the throughput column has a stable denominator.
    #
    # Both touch the one 8GB card, which is exclusive with qwen/comfyui/voicestudio/luxtts. The
    # sampler is read-only (nvidia-smi and the server's own /slots, 2s interval), but the probe
    # occupies the model server for the length of one generation. local_bench.sh holds the bench
    # lock for the whole run and refuses to start while a GPU service is up, so the exclusivity is
    # enforced before this code is reached; a run started by hand must check first.
    decision_block: dict = {}
    sampler = None
    base_url = getattr(args, "base_url", None)
    ctx_size = metricslib.ctx_size_of({"reproducibility": reproducibility, "notes": notes})
    if base_url and not getattr(args, "no_decision_sampler", False):
        probe = metricslib.probe_throughput(
            base_url,
            getattr(args, "api_key", None),
            model=(models[0] if len(models) == 1 else None),
            enabled=not getattr(args, "no_throughput_probe", False),
            # The run id makes every run's probe prompt unique, so it is processed and not served
            # from the server's prompt cache. A cached probe measures nothing.
            salt=run_id,
        )
        decision_block["throughput_probe"] = probe
        decision_block["sampler"] = {
            "interval_seconds": metricslib.SAMPLE_INTERVAL,
            "sources": metricslib.sampler_sources(base_url),
            "gpu_exclusive_required": True,
            "note": "peak VRAM is a GPU-exclusive measurement: the sampler records any co-tenant "
                    "it finds on the card and the report says so, but a run that shares the card "
                    "is not a comparable VRAM row",
        }
        sampler = metricslib.DecisionSampler(
            base_url=base_url,
            api_key=getattr(args, "api_key", None),
        )
    manifest["decision"] = decision_block

    # Publish-time validation, moved to launch time: the run does not start when its own
    # manifest would be refused at publication. Failing here costs one flag; failing later
    # costs a series nobody can defend.
    problems = publishlib.validate_manifest(manifest)
    if problems:
        sys.exit(publishlib.format_problems(problems, f"run {run_id} (manifest.json)")
                 + "\nNo model call was made. A run whose instrument cannot be recorded cannot "
                   "be published, so it is not worth the GPU time.")

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # The hash is written once, beside the manifest, and copied onto every record: it is the
    # proof a published row and this manifest describe the same instrument, and that the
    # manifest was not edited after the run.
    manifest_stamp = publishlib.manifest_sha256(manifest)
    (run_dir / "manifest.sha256").write_text(manifest_stamp + "\n")
    instrument_stamp = publishlib.instrument_fingerprint(manifest)
    # The configuration, resolved once from the manifest that was just written: the variant
    # name, the chat-template kwargs that decide whether the model was asked to think, the
    # context and the KV cache types. Every record carries its own copy, so a published row
    # can be keyed on model plus configuration without reading back a manifest that may be
    # gone, and a run of the same weights under another flag set cannot merge into its row.
    config = result_key.config_block(notes, reproducibility)
    print_run_header(manifest)
    if lock.waited:
        print(f"NOTE: this run waited {lock.waited_seconds:g}s for the bench lock "
              f"(held by {((lock.holder_before or {}).get('pid')) or 'another run'}); "
              f"every wall clock below starts after that wait.")

    total = len(models) * len(tasks) * args.repeat
    n = 0
    results = []
    if sampler:
        sampler.start()
        print(f"decision sampler: every {metricslib.SAMPLE_INTERVAL}s from "
              f"{', '.join(metricslib.sampler_sources(base_url))} "
              f"(GPU-exclusive measurement; peak VRAM is only comparable on an idle card)",
              flush=True)
    try:
        for model in models:
            for task in tasks:
                for r in range(1, args.repeat + 1):
                    n += 1
                    print(f"[{n}/{total}] {task.key} r{r} model={model or 'config-default'}", flush=True)
                    rec = run_one(task, model, args.provider, r, run_id, provider_config,
                                  getattr(args, "reasoning", None),
                                  manifest_stamp=manifest_stamp, instrument_stamp=instrument_stamp,
                                  allow_inline_scripts=not getattr(args, "no_inline_scripts", False),
                                  sampler=sampler, ctx_size=ctx_size, config=config)
                    results.append(rec)
                    if rec.get("harness_failed"):
                        status = "FAULT"   # harness/provider fault, not a model result
                    else:
                        status = "PASS" if rec["score"] >= 1.0 else ("part" if rec["score"] > 0 else "FAIL")
                    print(
                        f"      {status} score={rec['score']:.2f} "
                        f"turns={rec.get('turns')} tools={rec.get('tool_calls')} "
                        f"tokens={rec.get('tokens_in')}/{rec.get('tokens_out')} "
                        f"wall={rec['wall_seconds']}s exit={rec['exit_code']}",
                        flush=True,
                    )
    finally:
        # The samples outlive the run even when it dies part way: a partial series still carries
        # the VRAM and context evidence for the attempts that did run.
        if sampler:
            samples = sampler.stop()
            (run_dir / "decision-samples.json").write_text(json.dumps(samples, indent=2))
            print(f"decision samples: {len(samples)} rows -> {run_dir / 'decision-samples.json'}",
                  flush=True)

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))

    # One bounded cleanup for the whole run, never per task (see EnvSandbox.destroy).
    if _PENDING_CLEANUP:
        touched = cleanup_benchmark_sessions(REAL_HOME, _PENDING_CLEANUP)
        print(f"archived {touched} benchmark session(s) in state.db")
        _PENDING_CLEANUP.clear()

    print(f"\nrun_id: {run_id}")
    print(f"results: {run_dir / 'results.json'}")
    summarise(results)


def scorable(record: dict) -> bool:
    """Is this record's score the model's own artifact score?

    ``score_source`` is written by mark_harness_failure since 2026-10-27. Records from
    earlier runs have no such field, and their only excluded shape is a harness fault, so
    the fallback reads the flag rather than inventing a score for old rows.
    """
    src = record.get("score_source")
    if src:
        return src == "artifact"
    return not record.get("harness_failed")


def fault_counts(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        if not r.get("harness_failed"):
            continue
        cls = (r.get("fault") or {}).get("class") or "unclassified"
        counts[cls] = counts.get(cls, 0) + 1
    return counts


def summarise(results: list[dict]) -> None:
    by_model: dict[str, list[dict]] = {}
    for r in results:
        by_model.setdefault(r.get("model") or "config-default", []).append(r)
    # Harness faults are no longer zeros: a faulted run whose workspace proves work happened
    # keeps its artifact score and its exit code travels beside it. What stays excluded is a
    # run that produced nothing scorable at all (a substitution, an argv refusal, an
    # untouched workspace), because there is no artifact to score.
    stalls = [r for r in results if not scorable(r)]
    # "pass@1" would mean the probability of success on a single first attempt. What
    # this table shows is a ratio over every execution, which at --repeat 3 is three
    # attempts per task. Naming it pass@1 overstated what was measured, so the columns
    # are labelled for what they actually count.
    print("\n" + "=" * 104)
    print(f"{'model':38} {'runs':>7} {'tasks 3/3':>10} {'score':>7} {'faults':>7} {'turns':>7} {'tokens':>9}")
    print("=" * 104)
    for model, rs in by_model.items():
        # Score only records that carry the model's own artifact score, and show the excluded
        # count beside it so an excluded run is never invisible.
        valid = [r for r in rs if scorable(r)]
        faulted = len(rs) - len(valid)
        n = len(valid)
        full = sum(1 for r in valid if r["score"] >= 1.0)
        mean_score = statistics.mean(r["score"] for r in valid) if valid else float("nan")
        turns = [r["turns"] for r in valid if isinstance(r.get("turns"), int)]
        tok = [(r.get("tokens_in") or 0) + (r.get("tokens_out") or 0) for r in valid]
        # how many tasks were perfect on every attempt
        per_task: dict[str, list[dict]] = {}
        for r in valid:
            per_task.setdefault(r["task"], []).append(r)
        stable = sum(1 for t, ts in per_task.items() if all(x["score"] >= 1.0 for x in ts))
        counts = fault_counts(rs)
        fault_col = (",".join(f"{k}:{v}" for k, v in sorted(counts.items()))
                     or (f"+{faulted}F" if faulted else "-"))
        runs_col = f"{full}/{n}"
        print(
            f"{model[:38]:38} {runs_col:>7} {f'{stable}/{len(per_task)}':>10} "
            f"{(f'{mean_score:.2f}' if valid else 'n/a'):>7} "
            f"{fault_col[:7]:>7} "
            f"{(statistics.mean(turns) if turns else float('nan')):7.1f} "
            f"{(statistics.mean(tok) if any(tok) else 0):9.0f}"
        )
    print("=" * 104)
    if any(r.get("harness_failed") for r in results):
        counts = fault_counts(results)
        print(f"harness faults: {sum(counts.values())} of {len(results)} runs ended non-zero or on "
              f"the wrong model ({', '.join(f'{k}={v}' for k, v in sorted(counts.items()))}).")
        print("  turn_limit / timeout / agent_exit keep their artifact score and are counted "
              "above; the exit code travels with the row.")
        if stalls:
            print(f"  {len(stalls)} produced no scorable artifact and are EXCLUDED from the mean:")
        seen: set[str] = set()
        for r in results:
            if not r.get("harness_failed"):
                continue
            detail = next((c.get("detail") for c in r.get("checks", []) if not c.get("pass")), "")
            key = f"{r.get('model')}: {detail}"
            if key in seen:
                continue
            seen.add(key)
            cls = (r.get("fault") or {}).get("class", "?")
            print(f"  - {r.get('model')} {r['task']} r{r['repeat']} [{cls} exit "
                  f"{(r.get('fault') or {}).get('exit_code')}]: {str(detail)[:130]}")
        print("=" * 92)


def cmd_rescore(args) -> None:
    """Re-file an archived run's scores under the current fault rule, without model calls.

    The artifact score is recovered from the checks the run's own grader produced, not by
    re-running the grader. That distinction is not pedantry: a timeout kills the CLI client
    while the agent loop keeps running in the gateway, so a workspace can be rewritten after
    it was graded. Re-grading in that state measures a file the run never had, which on this
    box turned a recorded 6/8 into an 8/8. Pass --regrade-from-disk only when the stored
    checks are missing, and read the warning that comes with it.
    """
    run_dir = RUNS_DIR / args.run_id
    agg = run_dir / "results.json"
    if not agg.exists():
        sys.exit(f"no such run: {agg}")
    tasks = {t.key: t for t in load_tasks(None)}
    records = json.loads(agg.read_text())
    changed = 0
    for rec in records:
        work = Path(rec.get("workdir") or "")
        checks = rec.get("checks") or []
        artifact = artifact_from_checks(checks)
        if args.regrade_from_disk:
            task = tasks.get(rec.get("task"))
            if task is None or not work.is_dir():
                print(f"  skip {rec.get('task')} r{rec.get('repeat')}: no workspace on disk",
                      file=sys.stderr)
                continue
            print(f"  WARNING {rec.get('task')} r{rec.get('repeat')}: re-grading from disk; the "
                  f"workspace may have moved after grading", file=sys.stderr)
            artifact, checks = grade(task, work, record=rec)
            rec["checks"] = checks
            rec["workspace_changed"] = workspace_changed(task, work)
            # Re-run the quality gates against the re-graded workspace so the stored
            # secondary score matches the fresh checks (behavioural gates read rec["stdout_raw"],
            # which results.json preserves).
            qgates.apply_quality_gates(task, work, rec)
        if artifact is None:
            print(f"  skip {rec.get('task')} r{rec.get('repeat')}: no stored grader checks",
                  file=sys.stderr)
            continue
        # Keep the fault evidence that was already recorded: the detail names the exit
        # reason, which no re-grade can reconstruct from the workspace.
        fault = rec.get("fault") or {}
        detail = fault.get("detail") or next(
            (c.get("detail") for c in checks
             if (c.get("check") or "") == "harness ran the requested model"), None)
        rec["artifact_score"] = artifact
        task = tasks.get(rec.get("task"))
        kind = fault.get("class") or (
            "substitution" if "instead of the requested" in (detail or "")
            else fault_class(rec, task))
        if not kind and rec.get("harness_failed"):
            # An old record whose only evidence is the flag: the artifact belongs to the
            # model that answered, so it cannot be scored for this one.
            kind = "no_work"
        if rec.get("harness_failed") or kind:
            rec["harness_failed"] = True
            mark_harness_failure(rec, detail or f"exit {rec.get('exit_code')}", kind=kind)
            (work.parent / "result.json").write_text(json.dumps(rec, indent=2))
        else:
            rec["score"] = artifact
            rec["score_source"] = "artifact"
            rec["harness_failed"] = False
            rec.pop("fault", None)
            if work.parent.is_dir():
                (work.parent / "result.json").write_text(json.dumps(rec, indent=2))
        changed += 1
        cls = (rec.get("fault") or {}).get("class", "-")
        print(f"  {rec['task']:28} r{rec['repeat']} artifact={artifact:.4f} "
              f"fault={cls} score={rec['score']:.4f} source={rec.get('score_source')}")
    agg.write_text(json.dumps(records, indent=2))
    print(f"\nre-filed {changed} record(s) in {run_dir}")
    summarise(records)


def cmd_annotate(args) -> int:
    """Attach notes to an archived run: manifest, aggregate, and every per-attempt record.

    Needed because provenance is often learned after the fact. The Groot run of 2026-10-27 was made
    in the model's default thinking mode and that fact lived only in a chat message until this
    existed; a note written here becomes part of the run on disk.
    """
    run_dir = RUNS_DIR / args.run_id
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        sys.exit(f"no such run: {run_dir}")
    notes = parse_notes(args.note)
    if not notes:
        sys.exit("nothing to annotate: pass --note key=value")

    manifest = json.loads(manifest_path.read_text())
    merged = dict(manifest.get("notes") or {})
    merged.update(notes)
    manifest["notes"] = merged
    manifest_path.write_text(json.dumps(manifest, indent=2))

    touched = 0
    for path in sorted(run_dir.glob("tasks/*/*/r*/result.json")):
        rec = json.loads(path.read_text())
        rec_notes = dict(rec.get("notes") or {})
        rec_notes.update(notes)
        rec["notes"] = rec_notes
        path.write_text(json.dumps(rec, indent=2))
        touched += 1
    agg = run_dir / "results.json"
    if agg.exists():
        rows = json.loads(agg.read_text())
        for rec in rows:
            rec_notes = dict(rec.get("notes") or {})
            rec_notes.update(notes)
            rec["notes"] = rec_notes
        agg.write_text(json.dumps(rows, indent=2))
    print(f"annotated {args.run_id}: {notes} on {touched} record(s) and the aggregate")
    return 0


def cmd_merge(args) -> None:
    """File several runs of one model as a single comparable series.

    Two runs of the same tasks for the same model collide on (task, repeat): a suite run at
    one attempt plus three extra attempts on its weak tasks is four attempts, not two runs
    that overwrite each other. This concatenates them, renumbers each attempt into the next
    free slot for its (model, task), and stamps the origin on every record so the source
    directory stays the evidence and the published series stays complete.
    """
    out = RUNS_DIR / args.out_run_id
    if out.exists() and not args.force:
        sys.exit(f"{out} exists; pass --force to replace it")
    merged: list[dict] = []
    loaded: list[dict] = []
    for run_id in args.in_run_ids:
        path = RUNS_DIR / run_id / "results.json"
        if not path.exists():
            sys.exit(f"no results.json in {path.parent}")
        for rec in json.loads(path.read_text()):
            rec = dict(rec)
            rec["source_run"] = run_id
            rec["source_repeat"] = rec.get("repeat")
            rec["run_id"] = args.out_run_id
            loaded.append(rec)
    # Only scorable attempts take a slot in the series. An excluded attempt (a substitution, an
    # untouched workspace) has no score, so numbering it would leave a gap in the published
    # sequence and inflate the repeat count every table row and chart is read against. It keeps
    # its origin and is filed as an exclusion instead.
    taken: dict[tuple[str, str], int] = {}
    for rec in loaded:
        if rec.get("score_source") != "artifact":
            rec["repeat"] = None
            merged.append(rec)
            continue
        key = (rec.get("model") or "", rec["task"])
        taken[key] = taken.get(key, 0) + 1
        rec["repeat"] = taken[key]
        merged.append(rec)
    merged.sort(key=lambda r: (r.get("model") or "", r["task"],
                               -1 if r.get("repeat") is None else r["repeat"]))
    # A merged series is published, so its manifest must satisfy the same schema as a single
    # run: inherit the first segment's manifest (the harness, instrument, template and sampler
    # blocks are identical across segments by construction) and record the per-segment argv,
    # so nothing about the merge is assumed rather than stated.
    base_manifest: dict = {}
    segments: list[dict] = []
    for run_id in args.in_run_ids:
        path = RUNS_DIR / run_id / "manifest.json"
        if not path.exists():
            sys.exit(f"no manifest.json in {path.parent}; a merged series must be able to "
                     f"state the harness that produced every segment")
        seg = json.loads(path.read_text())
        if not base_manifest:
            base_manifest = seg
        else:
            # grader_proof carries its own timestamp and duration, which must differ between
            # segments; what has to match is that both were proven and over the same task count.
            for field in ("freeze", "instrument_frozen"):
                if seg.get(field) != base_manifest.get(field):
                    sys.exit(f"{run_id} differs from {args.in_run_ids[0]} on {field}; "
                             f"segments of one series must share an instrument")
            a = {k: (seg.get("grader_proof") or {}).get(k) for k in ("passed", "n")}
            b = {k: (base_manifest.get("grader_proof") or {}).get(k) for k in ("passed", "n")}
            if a != b:
                sys.exit(f"{run_id} differs from {args.in_run_ids[0]} on grader_proof "
                         f"(passed/n): {a} vs {b}")
        segments.append({
            "run_id": run_id,
            "started": seg.get("started"),
            "cli_flags": (seg.get("reproducibility") or {}).get("agent", {}).get("cli_flags"),
            "records": len([r for r in loaded if r.get("source_run") == run_id]),
            "manifest_sha256": publishlib.manifest_sha256(seg),
        })
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    manifest = dict(base_manifest)
    manifest.update({
        "run_id": args.out_run_id,
        "merged_from": list(args.in_run_ids),
        "models": sorted({str(r.get("model") or "") for r in merged}),
        "tasks": sorted({r["task"] for r in merged}),
        "records": len(merged),
        # The schema wants an int. Segments can be uneven, so state the deepest series and let
        # merge.note carry the renumbering rule rather than pretending every task has this many.
        "repeat": max(taken.values()) if taken else 0,
        "started": datetime.now(timezone.utc).isoformat(),
        # The schema requires the isolation block field by field, so keep the first segment's
        # and state above it that the merge added no model calls of its own.
        "isolation": {**(base_manifest.get("isolation") or {}),
                      "merge": "inherited from the source runs; no new model calls were made"},
        "merge": {
            "segments": segments,
            "note": "The reproducibility block above is the first segment's, verbatim: every "
                    "segment ran the same instrument, provider and flags, and the only per-run "
                    "difference is the run id in the argv, which each segment records here. "
                    "Attempts are renumbered per (model, task); source_run and source_repeat "
                    "keep each row's origin and manifest.sha256 its source manifest.",
        },
    })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    # The merged series is one publishable run, so it carries the same two provenance links a
    # single run does: the manifest hash proves the manifest was not edited after the model
    # calls, and the instrument fingerprint proves every row came from one instrument. Each
    # source stamp is preserved in merge.segments, so the origin of any row is still checkable.
    manifest_stamp = publishlib.manifest_sha256(manifest)
    (out / "manifest.sha256").write_text(manifest_stamp + "\n")
    instrument_stamp = publishlib.instrument_fingerprint(manifest)
    for rec in merged:
        rec["manifest_sha256"] = manifest_stamp
        rec["instrument_fingerprint"] = instrument_stamp
    (out / "results.json").write_text(json.dumps(merged, indent=2))
    print(f"merged {len(merged)} record(s) from {len(args.in_run_ids)} run(s) into {out}")
    summarise(merged)


def cmd_report(args) -> None:
    path = RUNS_DIR / args.run_id / "results.json"
    if not path.exists():
        sys.exit(f"no such run: {args.run_id}")
    results = json.loads(path.read_text())
    manifest_path = RUNS_DIR / args.run_id / "manifest.json"
    if manifest_path.exists():
        print_run_header(json.loads(manifest_path.read_text()))
    summarise(results)
    print("\nPer-task detail")
    for r in results:
        failed = [c for c in r["checks"] if not c.get("pass")]
        flag = "ok  " if r["score"] >= 1.0 else "FAIL"
        print(f"  {flag} {r['task']:28} r{r['repeat']} {r['score']:.2f}"
              f"  turns={r.get('turns')} tools={r.get('tool_calls')}")
        for c in failed[:3]:
            print(f"        - {c.get('check')}: {str(c.get('detail'))[:110]}")
    # Uncertainty: a point estimate with a task-level bootstrap interval and a per-task
    # table, so two macro scores are never shown as a ranking without their spread. No
    # p-value is printed here; see hbench_uncertainty for why it carries no information.
    try:
        print("\n" + uncertlib.report_single_run(args.run_id))
    except (OSError, ValueError, KeyError) as exc:
        print(f"\nUncertainty: could not summarise ({exc})", file=sys.stderr)
    # Core vs smoke: the whole-suite macro is dominated by the tasks every candidate aces,
    # which lifts the reported level and buries the tasks that differ. Split the score so the
    # core ranking excludes saturated tasks while the smoke score keeps every task as a
    # regression check. Saturated tasks are never deleted (bin/hbench_coresmoke.py, SPEC ws3).
    try:
        print("\n" + coresmoke.report_single_run(
            args.run_id, reference=args.reference, ceiling=args.ceiling))
    except (OSError, ValueError, KeyError) as exc:
        print(f"\nCore/smoke: could not split ({exc})", file=sys.stderr)
    # Decision metrics: the cost column beside the score. Six metrics with units, every one
    # either measured or named as unmeasured with its reason (bin/hbench_metrics.py, SPEC ws7).
    try:
        print("\n" + metricslib.report_single_run(args.run_id))
    except (OSError, ValueError, KeyError) as exc:
        print(f"\nDecision metrics: could not report ({exc})", file=sys.stderr)
    # Quality gates: the secondary binary score over gate-bearing attempts, printed after the
    # pass rate so the two are never conflated (bin/hbench_quality_gates.py, SPEC ws5).
    try:
        print("\n" + qgates.report_single_run(args.run_id))
    except (OSError, ValueError, KeyError) as exc:
        print(f"\nQuality gates: could not summarise ({exc})", file=sys.stderr)


def cmd_matrix(args) -> None:
    """Per-task score matrix across every model in the run."""
    path = RUNS_DIR / args.run_id / "results.json"
    if not path.exists():
        sys.exit(f"no such run: {args.run_id}")
    results = json.loads(path.read_text())

    models: list[str] = []
    for r in results:
        m = r.get("model") or "config-default"
        if m not in models:
            models.append(m)
    tasks: list[str] = []
    for r in results:
        if r["task"] not in tasks:
            tasks.append(r["task"])

    cell: dict[tuple[str, str], dict] = {}
    for r in results:
        cell[(r.get("model") or "config-default", r["task"])] = r

    short = {t: t.split("-", 1)[1][:18] for t in tasks}
    width = max(len(m) for m in models) + 2
    header = "task".ljust(20) + "".join(m[:16].rjust(18) for m in models)
    print("\n" + "=" * len(header))
    print("SCORE MATRIX (cell = score, with turns)")
    print(header)
    print("-" * len(header))
    for t in tasks:
        row = short[t].ljust(20)
        for m in models:
            r = cell.get((m, t))
            if not r:
                row += "-".rjust(18)
            else:
                tag = f"{r['score']:.2f} ({r.get('turns')}t)"
                row += tag.rjust(18)
        print(row)
    print("-" * len(header))
    row = "PASS@1".ljust(20)
    for m in models:
        rs = [r for r in results if (r.get("model") or "config-default") == m]
        full = sum(1 for r in rs if r["score"] >= 1.0)
        row += f"{full}/{len(rs)}".rjust(18)
    print(row)
    print("=" * len(header))


def cmd_compare(args) -> None:
    """Paired comparison of two runs with a task-level bootstrap interval and a verdict.

    Prints each model's macro score with its hierarchical bootstrap interval, the paired
    difference with its interval, a per-task paired table, and an explicit verdict. The
    verdict is ``indistinguishable`` when the paired interval crosses zero or the gap is
    below the practical-equivalence threshold; a bare p-value is never printed because a
    12-task design cannot reach significance (see bin/hbench_uncertainty.py).
    """
    try:
        print(uncertlib.report_comparison(
            args.run_a, args.run_b, model_a=args.model_a, model_b=args.model_b,
            n_boot=args.n_boot, seed=args.seed))
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    except ValueError as exc:
        sys.exit(f"comparison refused: {exc}")
    # Core/smoke split over the same two runs: the verdict above is read off the whole-suite
    # macro, which saturation holds near parity. The core gap below is the same ranking with
    # the saturated tasks removed, so a reader sees the gap the perfect tasks were burying.
    try:
        print("\n" + coresmoke.report_comparison(
            args.run_a, args.run_b, model_a=args.model_a, model_b=args.model_b,
            ceiling=args.ceiling))
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"core/smoke split refused: {exc}")
    # The cost column: at equal accuracy the decision is throughput, wall clock, turns, tool
    # validity, headroom and VRAM. Printed after the verdict so the reader has both halves.
    try:
        print("\n" + metricslib.report_comparison(
            args.run_a, args.run_b, model_a=args.model_a, model_b=args.model_b))
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"decision metrics refused: {exc}")
    # Quality gates beside the cost column: the secondary binary pass rate for each run.
    try:
        print("\n" + qgates.report_comparison(
            args.run_a, args.run_b, model_a=args.model_a, model_b=args.model_b))
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"quality gates refused: {exc}")


def cmd_publish(args) -> None:
    """Publish a run's records, or refuse. Exit 3 is a schema or series refusal."""
    try:
        result = publishlib.publish(args.run_id, dry_run=args.dry_run, force=args.force)
    except publishlib.PublishRefused as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(3)
    # The dry-run path returns "legacy"; the real publish returns "legacy_incomparable".
    # Read both, exactly as hbench_publish's own CLI does, or a successful publish dies here
    # after the files are already written and looks like a failed publish.
    for message in result.get("legacy_incomparable", result.get("legacy", ())):
        print("  " + message, file=sys.stderr)
    print(f"published {result['published']} record(s) for {args.run_id} "
          f"(instrument {result['instrument_fingerprint'][:16]})")


def cmd_check(args) -> None:
    """Validate without publishing. Exit 3 is a refusal, so a test can assert on it."""
    if getattr(args, "manifest", None):
        manifest = json.loads(Path(args.manifest).read_text())
        problems = publishlib.validate_manifest(manifest)
        if problems:
            print(publishlib.format_problems(problems, args.manifest))
            sys.exit(3)
        print(f"{args.manifest}: schema OK; "
              f"instrument {publishlib.instrument_fingerprint(manifest)[:16]}")
        return
    if not args.run_id:
        sys.exit("check needs a run id or --manifest")
    try:
        manifest, records, _ = publishlib.check_run(args.run_id)
    except publishlib.PublishRefused as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(3)
    print(f"{args.run_id}: manifest OK, {len(records)} record(s) OK; "
          f"instrument {publishlib.instrument_fingerprint(manifest)[:16]}")
    for message in publishlib.series_conflicts(manifest, publishlib.published_manifests()):
        print("  " + message)


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolated Hermes-harness model benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list tasks")
    p_list.set_defaults(func=cmd_list)

    p_freeze = sub.add_parser(
        "freeze", help="write tasks/FROZEN.sha256 and tasks/FROZEN.json, or check the tree "
                       "against them (--check). `run` refuses to start on a mismatch.")
    p_freeze.add_argument("--check", action="store_true",
                          help="recompute the freeze and exit 1 naming every changed, added and "
                               "removed file. Read-only: no temp files, no fixtures, no network.")
    p_freeze.set_defaults(func=cmd_freeze)

    p_run = sub.add_parser("run", help="run a benchmark")
    p_run.add_argument("--model", action="append", help="model string, repeatable")
    p_run.add_argument("--provider", help="provider name (e.g. ollama-cloud)")
    p_run.add_argument("--tasks", nargs="+", default=["all"])
    p_run.add_argument("--repeat", type=int, default=1)
    p_run.add_argument("--budget-scale", type=float, default=1.0,
                       help="multiply every task's wall-clock run_budget. Use for slow local models. "
                            "Turn limits are never scaled, so a model that needs more turns still fails.")
    p_run.add_argument("--timeout-scale", type=float, default=None,
                       help="multiply every task's hard subprocess timeout. Defaults to budget_scale * 1.15.")
    p_run.add_argument("--run-id")
    p_run.add_argument("--allow-drift", action="store_true",
                       help="start even though the task set no longer matches tasks/FROZEN.sha256. "
                            "The manifest is stamped with instrument_drift and the changed paths, "
                            "so the row reads as measured on a moved instrument.")
    p_run.add_argument("--no-prove", action="store_true",
                       help="skip the grader proof (tests/test_graders.py) that otherwise runs "
                            "before attempt one. The manifest is stamped grader_proof.skipped.")
    p_run.add_argument("--lock-wait", type=float, default=DEFAULT_LOCK_WAIT,
                       help="seconds to wait for the bench lock before failing. Only one hbench run "
                            f"may hold the bench at a time (default {DEFAULT_LOCK_WAIT:g}, or "
                            "HBENCH_LOCK_WAIT). A wait is recorded in manifest.json.")
    p_run.add_argument("--provider-custom",
                       help="name of a custom_providers entry in ~/.hermes/config.yaml. Used with "
                            "--provider-config config to hand a local endpoint and its key to the "
                            "child explicitly, because --ignore-user-config hides that registration.")
    p_run.add_argument("--note", action="append", default=[], metavar="KEY=VALUE",
                       help="free-form provenance for this run, repeatable (e.g. --note reasoning=on). "
                            "Stored in manifest.json and on every record, so a server flag or a build "
                            "difference travels with the numbers instead of living in a chat log.")
    p_run.add_argument("--reasoning", default=None,
                       choices=["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
                       help="passed through to the child as --reasoning LEVEL. Unset means the child's "
                            "own default, which for a local OpenAI-compatible server is usually the "
                            "model default and therefore thinking ON. Set it explicitly whenever a "
                            "run is compared against another, because thinking changes both the "
                            "output-token count and, sometimes, the answer.")
    p_run.add_argument("--base-url", help="endpoint for --provider-config inline, e.g. "
                                          "http://192.168.1.15:8083/v1")
    p_run.add_argument("--api-key", help="key for --provider-config inline (redacted in the record)")
    p_run.add_argument("--provider-config",
                       choices=["auto", "auth-store", "config", "inline", "off"], default="auto",
                       help="How the child gets its provider block. auto (default): load endpoint + "
                            "credential from the auth store's credential pool for providers that cannot "
                            "authenticate under --ignore-user-config (nous, codex, xai-oauth, qwen-oauth, "
                            "minimax-oauth), leave every other provider to the child. auth-store: always "
                            "load it, fail if absent. off: never, the pre-2026-10-22 behaviour. The mode "
                            "used is recorded in manifest.json and printed in the report header. "
                            "config: load the single custom_providers entry named by "
                            "--provider-custom (endpoint and key only). "
                            "config.yaml, .env, rules and memory stay ignored in every mode.")
    p_run.add_argument("--no-inline-scripts", action="store_true",
                       help="Do not pass --yolo to the child. The child then runs with the code "
                            "default approvals.single_query_mode=deny and every inline `python3 -c` "
                            "or heredoc is BLOCKED (nobody is present to approve it). Each block "
                            "wastes a turn against --max-turns and distorts the score. Use this only "
                            "to reproduce a historical run's exact approval posture.")
    p_run.add_argument("--no-decision-sampler", action="store_true",
                       help="Do not sample context headroom or peak VRAM during the run. Both are "
                            "GPU-exclusive measurements read from the model server's /slots and "
                            "the main PC's SENTINEL telemetry; skip them when the card is shared "
                            "and the numbers would be someone else's. The report then names those "
                            "two metrics as unmeasured instead of guessing them.")
    p_run.add_argument("--no-throughput-probe", action="store_true",
                       help="Do not issue the fixed throughput probe at the start of the run. The "
                            "probe is one real generation (1024 prompt tokens, 128 output) that "
                            "makes prefill and decode rates comparable between runs; skip it only "
                            "if that generation would disturb the machine.")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="summarise a run (with uncertainty and a core/smoke split)")
    p_rep.add_argument("run_id")
    p_rep.add_argument("--reference", action="append", default=[],
                       help="reference candidate run id, repeatable; the reported run is always "
                            "in the panel too. Passing the sibling candidates makes the core set "
                            "the shared discriminating set used for a ranking.")
    p_rep.add_argument("--ceiling", type=float, default=coresmoke.SATURATION_CEILING,
                       help="saturation ceiling for the core/smoke split, normally left at 1.000 "
                            "(default %(default)s)")
    p_rep.set_defaults(func=cmd_report)

    p_ann = sub.add_parser("annotate", help="attach provenance notes to an archived run")
    p_ann.add_argument("run_id")
    p_ann.add_argument("--note", action="append", default=[], metavar="KEY=VALUE")
    p_ann.set_defaults(func=cmd_annotate)

    p_mat = sub.add_parser("matrix", help="per-task score matrix across models")
    p_mat.add_argument("run_id")
    p_mat.set_defaults(func=cmd_matrix)

    p_cmp = sub.add_parser("compare", help="paired comparison of two runs: macro scores with "
                                           "bootstrap intervals, the paired difference, and an "
                                           "indistinguishable verdict (no p-value)")
    p_cmp.add_argument("run_a")
    p_cmp.add_argument("run_b")
    p_cmp.add_argument("--model-a", help="which model of run_a to compare (needed only when "
                                         "the run measured more than one)")
    p_cmp.add_argument("--model-b", help="which model of run_b to compare (needed only when "
                                         "the run measured more than one)")
    p_cmp.add_argument("--n-boot", type=int, default=uncertlib.DEFAULT_N_BOOT,
                       help="bootstrap resamples (default %(default)s)")
    p_cmp.add_argument("--seed", type=int, default=uncertlib.DEFAULT_SEED,
                       help="rng seed so the reported interval is reproducible (default %(default)s)")
    p_cmp.add_argument("--ceiling", type=float, default=coresmoke.SATURATION_CEILING,
                       help="saturation ceiling for the core/smoke split, normally left at 1.000 "
                            "(default %(default)s)")
    p_cmp.set_defaults(func=cmd_compare)

    p_res = sub.add_parser("rescore", help="re-file an archived run's scores under the current "
                                          "fault rule (no model calls)")
    p_res.add_argument("run_id")
    p_res.add_argument("--regrade-from-disk", action="store_true",
                       help="re-run the graders instead of reading the stored checks. Only when "
                            "the checks are missing: a killed client can leave the agent still "
                            "writing, so the workspace may not be what was graded.")
    p_res.set_defaults(func=cmd_rescore)

    p_mrg = sub.add_parser("merge", help="file several runs of one model as one series, "
                                        "renumbering repeats per (model, task)")
    p_mrg.add_argument("out_run_id")
    p_mrg.add_argument("in_run_ids", nargs="+")
    p_mrg.add_argument("--force", action="store_true", help="replace an existing output run")
    p_mrg.set_defaults(func=cmd_merge)

    # Publishing is a checked operation, not a cp. See bin/hbench_publish.py for the schema
    # and for the series invariant that refuses a run whose instrument disagrees with an
    # already-published run of the same model.
    p_pub = sub.add_parser("publish", help="validate a run against the reproducibility schema "
                                          "and the published series, then file its records")
    p_pub.add_argument("run_id")
    p_pub.add_argument("--dry-run", action="store_true",
                       help="validate and report what would be published, write nothing")
    p_pub.add_argument("--force", action="store_true",
                       help="replace already-published records of this run (deliberate reissue)")
    p_pub.set_defaults(func=cmd_publish)

    p_chk = sub.add_parser("check", help="validate a run or a manifest against the schema, "
                                        "without publishing")
    p_chk.add_argument("run_id", nargs="?")
    p_chk.add_argument("--manifest", help="validate a manifest file instead of a run id")
    p_chk.set_defaults(func=cmd_check)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
