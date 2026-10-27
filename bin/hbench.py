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
                            custom: str | None = None) -> dict | None:
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
    cols = ("message_count, tool_call_count, input_tokens, output_tokens, "
            "model, billing_provider, billing_base_url")
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        if session_id:
            row = con.execute(
                f"select {cols} from sessions where id = ?", (session_id,)
            ).fetchone()
        else:
            row = con.execute(
                f"select {cols} from sessions order by started_at desc limit 1"
            ).fetchone()
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
# Runner
# --------------------------------------------------------------------------

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
            if p.is_file() and "__pycache__" not in p.parts and p.name != "result.json":
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


def run_one(task: Task, model: str | None, provider: str | None,
            repeat: int, run_id: str, provider_config: dict | None = None) -> dict:
    # The model MUST be part of the path: without it, running model B after model A
    # overwrites A's results for the same task key, silently destroying the matrix.
    safe_model = (model or "config-default").replace("/", "_").replace(":", "_")
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
                      + (" + provider block loaded explicitly" if provider_config else "")),
        "started": datetime.now(timezone.utc).isoformat(),
        "workdir": str(work),
        # Provenance for comparability: the wall clock is the runner's subprocess timeout
        # because `hermes chat` no longer accepts --run-budget (see the argv comment below).
        "run_budget_seconds": task.run_budget,
        "timeout_seconds": task.timeout,
        "budget_mode": "runner_timeout",
    }

    cmd = [HERMES_BIN, "chat"]
    if model:
        cmd += ["-m", model]
    if provider:
        cmd += ["--provider", provider]
    if provider_config:
        # The provider block only: endpoint + credential. Not config.yaml, not .env,
        # not rules or memory. --ignore-user-config stays on.
        cmd += ["--base-url", provider_config["base_url"], "--api-key", provider_config["api_key"]]

    # Provenance without the secret: the exact argv, credential value redacted. This is the
    # audit trail that shows the isolation flags were still on when a credential was supplied.
    record["cli_flags"] = [
        ("<redacted>" if i and cmd[i - 1] == "--api-key" else a)
        for i, a in enumerate(cmd)
    ]
    cmd += [
        "--quiet", "--format", "stream-json", "--oneshot",
        "--source", "tool",          # keeps benchmark runs out of user session lists
        "--ignore-rules",            # no AGENTS.md / SOUL.md / memory injection
        "--ignore-user-config",      # ignore ~/.hermes/config.yaml defaults
        "--max-turns", str(task.max_turns),
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
        "-q", task.prompt + "\n\n" + network_policy_note(),
    ]

    t0 = time.monotonic()
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
        if record.get("tool_calls") in (None, 0):
            record["tool_calls"] = db_stats.get("tool_call_count")

    # cleanup before grading so a huge home does not skew disk checks
    sandbox.destroy()

    record["score"], record["checks"] = grade(task, work)
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
        if "tool" in low and isinstance(ev.get("name"), str):
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
# Isolation layer 3: grade workspace state, never the reply
# --------------------------------------------------------------------------

def grade(task: Task, work: Path) -> tuple[float, list[dict]]:
    if task.judge is None:
        checks = [{"check": "no judge.py", "pass": False, "detail": "task has no grader"}]
        return 0.0, checks
    try:
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
    # Only printed when there was a wait, so a run that started immediately reads exactly
    # as it did before the lock existed.
    lock = manifest.get("lock") or {}
    if lock.get("waited"):
        print(f"lock:     waited {lock.get('waited_seconds')}s for another run to release "
              f"{lock.get('path')}; the wall clocks below start after that wait")
    print("=" * 92)


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
                                             args.provider_custom)
    # One bench run at a time, enforced here rather than in the task brief. Taken before
    # the run id is minted, so a run that waited carries the time it actually started.
    lock = BenchLock(wait_limit=args.lock_wait)
    lock.acquire()
    try:
        execute_run(args, tasks, provider_config, lock)
    finally:
        lock.release()


def execute_run(args, tasks, provider_config: dict | None, lock: BenchLock) -> None:
    models = args.model or [None]
    run_id = args.run_id or datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

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

    manifest = {
        "run_id": run_id,
        "budget_scale": args.budget_scale,
        "timeout_scale": (args.timeout_scale if args.timeout_scale is not None else args.budget_scale * 1.15),
        "models": models,
        "provider": args.provider,
        "tasks": [t.key for t in tasks],
        "repeat": args.repeat,
        "tasks_dir": str(TASKS_DIR),
        "started": datetime.now(timezone.utc).isoformat(),
        # Provenance: exactly how this run was isolated, and where its credentials came from.
        "isolation": {
            "mode": mode,
            "user_config": "ignored (--ignore-user-config)",
            "rules": "ignored (--ignore-rules: no AGENTS.md, SOUL.md or memory injection)",
            "provider": provider_block,
            "provider_config_flag": args.provider_config,
            "sessions": f"tagged source={EnvSandbox.SOURCE_TAG}, archived after the run",
        },
        # Provenance: whether this run had to queue behind another one. A wall clock taken
        # after a wait is still this run's own time, but the wait is part of the story.
        "lock": lock.describe(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print_run_header(manifest)
    if lock.waited:
        print(f"NOTE: this run waited {lock.waited_seconds:g}s for the bench lock "
              f"(held by {((lock.holder_before or {}).get('pid')) or 'another run'}); "
              f"every wall clock below starts after that wait.")

    total = len(models) * len(tasks) * args.repeat
    n = 0
    results = []
    for model in models:
        for task in tasks:
            for r in range(1, args.repeat + 1):
                n += 1
                print(f"[{n}/{total}] {task.key} r{r} model={model or 'config-default'}", flush=True)
                rec = run_one(task, model, args.provider, r, run_id, provider_config)
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
            artifact, checks = grade(task, work)
            rec["checks"] = checks
            rec["workspace_changed"] = workspace_changed(task, work)
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
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    manifest = {
        "run_id": args.out_run_id,
        "merged_from": list(args.in_run_ids),
        "models": sorted({str(r.get("model") or "") for r in merged}),
        "tasks": sorted({r["task"] for r in merged}),
        "records": len(merged),
        "repeat": "renumbered per (model, task); source_run and source_repeat keep the origin",
        "started": datetime.now(timezone.utc).isoformat(),
        "isolation": {"mode": "inherited from the source runs; no new model calls were made"},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Isolated Hermes-harness model benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="list tasks")
    p_list.set_defaults(func=cmd_list)

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
    p_run.add_argument("--lock-wait", type=float, default=DEFAULT_LOCK_WAIT,
                       help="seconds to wait for the bench lock before failing. Only one hbench run "
                            f"may hold the bench at a time (default {DEFAULT_LOCK_WAIT:g}, or "
                            "HBENCH_LOCK_WAIT). A wait is recorded in manifest.json.")
    p_run.add_argument("--provider-custom",
                       help="name of a custom_providers entry in ~/.hermes/config.yaml. Used with "
                            "--provider-config config to hand a local endpoint and its key to the "
                            "child explicitly, because --ignore-user-config hides that registration.")
    p_run.add_argument("--provider-config",
                       choices=["auto", "auth-store", "config", "off"], default="auto",
                       help="How the child gets its provider block. auto (default): load endpoint + "
                            "credential from the auth store's credential pool for providers that cannot "
                            "authenticate under --ignore-user-config (nous, codex, xai-oauth, qwen-oauth, "
                            "minimax-oauth), leave every other provider to the child. auth-store: always "
                            "load it, fail if absent. off: never, the pre-2026-10-22 behaviour. The mode "
                            "used is recorded in manifest.json and printed in the report header. "
                            "config: load the single custom_providers entry named by "
                            "--provider-custom (endpoint and key only). "
                            "config.yaml, .env, rules and memory stay ignored in every mode.")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="summarise a run")
    p_rep.add_argument("run_id")
    p_rep.set_defaults(func=cmd_report)

    p_mat = sub.add_parser("matrix", help="per-task score matrix across models")
    p_mat.add_argument("run_id")
    p_mat.set_defaults(func=cmd_matrix)

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

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
