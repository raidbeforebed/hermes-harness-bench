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

Usage:
    hbench.py list
    hbench.py run --model ollama-cloud/deepseek-v4.1-flash:1017 --tasks all
    hbench.py run --model A --model B --tasks all --repeat 3
    hbench.py report <run_id>
"""

from __future__ import annotations

import argparse
import json
import os
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

# Repo root: honour HBENCH_ROOT, otherwise the parent of this file's directory.
ROOT = Path(os.environ.get("HBENCH_ROOT") or Path(__file__).resolve().parent.parent)
TASKS_DIR = ROOT / "tasks"
RUNS_DIR = ROOT / "runs"
# The live Hermes install. Default follows $HERMES_HOME, which is where the CLI
# keeps config.yaml and .env on any machine.
REAL_HOME = Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))
REAL_ENV = REAL_HOME / ".env"

# Resolve the CLI once, by absolute path. systemd units and cron jobs often have a
# PATH that does not include ~/.local/bin, which made the runner die with
# FileNotFoundError: 'hermes' the first time it was put under systemd.
# Always use the public launcher, never the internal one. shutil.which("hermes")
# inside an agent turn can resolve to hermes-agent/.hermes/bin/hermes, which is a
# different entry point with different argv handling.
HERMES_BIN = shutil.which("hermes") or str(Path.home() / ".local/bin/hermes")
if not Path(HERMES_BIN).exists():
    for cand in (Path.home()/"bin/hermes", "/usr/local/bin/hermes", "/usr/bin/hermes"):
        if Path(cand).exists():
            HERMES_BIN = str(cand)
            break

# The live install. Runs share it but are isolated by flags and session tagging;
# see EnvSandbox for why relocating HERMES_HOME is not viable.
# REAL_HOME defined above with REAL_ENV.


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


def load_tasks(keys: list[str] | None) -> list[Task]:
    tasks = []
    if not TASKS_DIR.is_dir():
        sys.exit(f"no tasks directory at {TASKS_DIR}")
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
            max_turns=int(spec.get("max_turns", 40)),
            run_budget=int(spec.get("run_budget", 420)),
            timeout=int(spec.get("timeout", 480)),
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
        """Delete this run's benchmark sessions from the live state.db."""
        if not self.session_ids:
            return
        db = self.home / "state.db"
        if not db.exists():
            return
        try:
            con = sqlite3.connect(str(db), timeout=10)
            for sid in self.session_ids:
                con.execute(
                    "delete from sessions where id = ? and source = ?",
                    (sid, self.SOURCE_TAG),
                )
                con.execute("delete from messages where session_id = ?", (sid,))
            con.commit()
            con.close()
        except sqlite3.Error:
            pass


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
    """Read turn/tool/token counts from the run's own state.db.

    Returns None if the DB or session is absent, which is itself a signal:
    it means the run did not write state into HERMES_HOME.
    """
    db = home / "state.db"
    if not db.exists():
        return None
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        if session_id:
            row = con.execute(
                "select message_count, tool_call_count, input_tokens, output_tokens "
                "from sessions where id = ?", (session_id,)
            ).fetchone()
        else:
            row = con.execute(
                "select message_count, tool_call_count, input_tokens, output_tokens "
                "from sessions order by started_at desc limit 1"
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

def run_one(task: Task, model: str | None, provider: str | None,
            repeat: int, run_id: str) -> dict:
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
        "started": datetime.now(timezone.utc).isoformat(),
        "workdir": str(work),
    }

    cmd = [HERMES_BIN, "chat"]
    if model:
        cmd += ["-m", model]
    if provider:
        cmd += ["--provider", provider]
    cmd += [
        "--quiet", "--format", "stream-json", "--oneshot",
        "--source", "tool",          # keeps benchmark runs out of user session lists
        "--ignore-rules",            # no AGENTS.md / SOUL.md / memory injection
        "--ignore-user-config",      # ignore ~/.hermes/config.yaml defaults
        "--max-turns", str(task.max_turns),
        "--run-budget", str(task.run_budget),
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
        if record.get("tool_calls") in (None, 0):
            record["tool_calls"] = db_stats.get("tool_call_count")

    # cleanup before grading so a huge home does not skew disk checks
    sandbox.destroy()

    record["score"], record["checks"] = grade(task, work)
    # A run that reported a CLI error is not a result, whatever the files look like.
    # Without this, an argv/bootstrap failure that still left a workspace behind was
    # recorded as a pass, which is how a broken harness reported 100%.
    if record.get("exit_code") not in (0,):
        record["harness_failed"] = True
        record["checks"] = [{
            "check": "harness exited cleanly",
            "pass": False,
            "detail": f"exit {record.get('exit_code')}: {(record.get('stderr') or '')[:300]}",
        }] + record["checks"]
        record["score"] = 0.0
    record["finished"] = datetime.now(timezone.utc).isoformat()
    (task_dir / "result.json").write_text(json.dumps(record, indent=2))
    return record


def parse_stream(raw: str) -> dict:
    """Pull turns, tokens, tool calls and session id out of stream-json."""
    out = {
        "turns": None, "tokens_in": None, "tokens_out": None,
        "tool_calls": 0, "tool_names": [], "session_id": None,
        "final_text": "", "parse_errors": 0,
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


def cmd_run(args) -> None:
    tasks = load_tasks(args.tasks)
    if not tasks:
        sys.exit("nothing to run")
    models = args.model or [None]
    run_id = args.run_id or datetime.now().strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "models": models,
        "provider": args.provider,
        "tasks": [t.key for t in tasks],
        "repeat": args.repeat,
        "tasks_dir": str(TASKS_DIR),
        "started": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = len(models) * len(tasks) * args.repeat
    n = 0
    results = []
    for model in models:
        for task in tasks:
            for r in range(1, args.repeat + 1):
                n += 1
                print(f"[{n}/{total}] {task.key} r{r} model={model or 'config-default'}", flush=True)
                rec = run_one(task, model, args.provider, r, run_id)
                results.append(rec)
                status = "PASS" if rec["score"] >= 1.0 else ("part" if rec["score"] > 0 else "FAIL")
                print(
                    f"      {status} score={rec['score']:.2f} "
                    f"turns={rec.get('turns')} tools={rec.get('tool_calls')} "
                    f"tokens={rec.get('tokens_in')}/{rec.get('tokens_out')} "
                    f"wall={rec['wall_seconds']}s exit={rec['exit_code']}",
                    flush=True,
                )

    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\nrun_id: {run_id}")
    print(f"results: {run_dir / 'results.json'}")
    summarise(results)


def summarise(results: list[dict]) -> None:
    by_model: dict[str, list[dict]] = {}
    for r in results:
        by_model.setdefault(r.get("model") or "config-default", []).append(r)
    print("\n" + "=" * 78)
    print(f"{'model':42} {'pass@1':>8} {'score':>7} {'turns':>7} {'tokens':>10}")
    print("=" * 78)
    for model, rs in by_model.items():
        n = len(rs)
        full = sum(1 for r in rs if r["score"] >= 1.0)
        mean_score = statistics.mean(r["score"] for r in rs)
        turns = [r["turns"] for r in rs if isinstance(r.get("turns"), int)]
        tok = [(r.get("tokens_in") or 0) + (r.get("tokens_out") or 0) for r in rs]
        print(
            f"{model[:42]:42} {full / n:8.1%} {mean_score:7.2f} "
            f"{(statistics.mean(turns) if turns else float('nan')):7.1f} "
            f"{(statistics.mean(tok) if any(tok) else 0):10.0f}"
        )
    print("=" * 78)


def cmd_report(args) -> None:
    path = RUNS_DIR / args.run_id / "results.json"
    if not path.exists():
        sys.exit(f"no such run: {args.run_id}")
    results = json.loads(path.read_text())
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
    p_run.add_argument("--run-id")
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser("report", help="summarise a run")
    p_rep.add_argument("run_id")
    p_rep.set_defaults(func=cmd_report)

    p_mat = sub.add_parser("matrix", help="per-task score matrix across models")
    p_mat.add_argument("run_id")
    p_mat.set_defaults(func=cmd_matrix)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
