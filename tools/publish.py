#!/usr/bin/env python3
"""Publish finished hbench runs to the repo, in one command.

The gap this closes: exporting, charting, editing the README table, committing and
pushing were five manual steps. A run could finish, get reported in chat, and never
reach the published repository, which is exactly what happened on 2026-10-22.

    python3 tools/publish.py <run_id> [<run_id> ...]   # publish these runs
    python3 tools/publish.py --latest                   # newest run per model
    python3 tools/publish.py --dry-run <run_id>         # do everything but commit

Order matters and is enforced:

1. export    - a run with harness failures aborts the whole thing. A fault is not a
               model result, and publishing it as 0.00 is worse than not publishing.
2. charts    - regenerated from results/, so they cannot drift from the data.
3. README    - the results table between the RESULTS markers is rewritten from
               results/. Prose outside the markers is left alone, because that part
               is written by a person and should stay that way.
4. commit    - only if something changed.
5. push      - then verify the remote actually points at the new commit, because a
               push that reported success is not the same thing as a push that landed.

Exit codes: 0 published (or nothing to do), 1 aborted, 2 git failure.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
README = ROOT / "README.md"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

# Display names, kept in step with tools/make_charts.py. A model missing here still
# publishes, under its own name in grey, rather than silently vanishing.
LABEL = {
    "deepseek-v4.1-flash:1017": "DeepSeek V4.1 Flash",
    "glm-5.3-flash:cloud": "GLM-5.3 Flash",
    "minimax-m3.1-flash:cloud": "MiniMax M3.1 Flash",
    "beam:cloud": "Beam",
    "Qwen3.6-35B-A3B-UD-Q4_K_M": "Qwen3.6 35B (local)",
    "gemma4:cloud": "Gemma 4 (cloud)",
    "poolside/laguna-s-2.1:free": "Laguna S 2.1 (free)",
    "poolside/laguna-xs-2.1:free": "Laguna XS 2.1 (free)",
    "poolside/laguna-s-2.1": "Laguna S 2.1",
    "poolside/laguna-xs-2.1": "Laguna XS 2.1",
}
ORDER = [
    "deepseek-v4.1-flash:1017",
    "glm-5.3-flash:cloud",
    "minimax-m3.1-flash:cloud",
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "beam:cloud",
    "gemma4:cloud",
    "Qwen3.6-35B-A3B-UD-Q4_K_M",
]


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, **kw)


def head(short: bool = True) -> str:
    r = run(["git", "rev-parse", "--short" if short else "--verify", "HEAD"])
    return r.stdout.strip()


def fmt_duration(seconds: float) -> str:
    """Seconds up to ten minutes, because a mean wall of 106 s reads better than 2 min."""
    if seconds >= 5400:
        return f"{seconds / 3600:.2f} h"
    if seconds >= 600:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


def table_from_results() -> str:
    """Rebuild the results table from results/*.json only."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for f in glob.glob(str(RESULTS / "*.json")):
        try:
            d = json.loads(Path(f).read_text())
        except (OSError, ValueError):
            continue
        if d.get("model") and d.get("task"):
            by_model[d["model"]].append(d)
    if not by_model:
        raise SystemExit("no results to build a table from")

    rows: list[dict] = []
    for model, rs in by_model.items():
        per_task: dict[str, list[float]] = defaultdict(list)
        for r in rs:
            per_task[r["task"]].append(float(r.get("score") or 0.0))
        stable = sum(1 for v in per_task.values() if all(s >= 1.0 for s in v))
        passed = sum(1 for r in rs if float(r.get("score") or 0.0) >= 1.0)
        turns = [r["turns"] for r in rs if isinstance(r.get("turns"), int)]
        toks = [(r.get("tokens_in") or 0) + (r.get("tokens_out") or 0)
                for r in rs if r.get("tokens_in")]
        walls = [r["wall_seconds"] for r in rs if r.get("wall_seconds")]
        rows.append({
            "model": model,
            "label": LABEL.get(model, model),
            "repeats": max((r.get("repeat") or 1) for r in rs),
            "passed": passed, "n": len(rs),
            "stable": stable, "tasks": len(per_task),
            "turns": statistics.mean(turns) if turns else None,
            "tokens": statistics.mean(toks) if toks else None,
            "wall": statistics.mean(walls) if walls else None,
            "total": sum(walls) if walls else None,
        })

    rows.sort(key=lambda r: (ORDER.index(r["model"]) if r["model"] in ORDER else 99,
                             r["label"]))
    best = {
        "turns": min((r["turns"] for r in rows if r["turns"]), default=None),
        "tokens": min((r["tokens"] for r in rows if r["tokens"]), default=None),
        "wall": min((r["wall"] for r in rows if r["wall"]), default=None),
    }

    def bold(val, bestval, text):
        return f"**{text}**" if val is not None and bestval is not None and abs(val - bestval) < 1e-9 else text

    out = [
        START,
        "| Model | Repeats | Runs passed | Tasks 3/3 | Mean turns | Mean tokens | Mean wall | Total |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        share = 100 * r["passed"] / r["n"]
        pct = (f"{r['passed']}/{r['n']} ({share:.0f}%)" if share == 100
               else f"{r['passed']}/{r['n']} ({share:.1f}%)") if r["n"] else "-"
        if r["passed"] == r["n"] and r["n"]:
            pct = f"**{pct}**"
        turns = bold(r["turns"], best["turns"], f"{r['turns']:.1f}") if r["turns"] else "-"
        toks = bold(r["tokens"], best["tokens"], f"{int(r['tokens']):,}") if r["tokens"] else "-"
        wall = bold(r["wall"], best["wall"], fmt_duration(r["wall"])) if r["wall"] else "-"
        total = fmt_duration(r["total"]) if r["total"] else "-"
        out.append(f"| {r['label']} | {r['repeats']} | {pct} | {r['stable']}/{r['tasks']} "
                   f"| {turns} | {toks} | {wall} | {total} |")
    out.append(END)
    return "\n".join(out)


def rewrite_readme() -> bool:
    text = README.read_text()
    if START not in text or END not in text:
        raise SystemExit(
            f"README is missing the {START} / {END} markers.\n"
            "Add them around the results table before publishing."
        )
    block = table_from_results()
    new = re.sub(re.escape(START) + r".*?" + re.escape(END), block, text, count=1, flags=re.S)
    if new == text:
        return False
    README.write_text(new)
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="*", help="run directories under runs/ to publish")
    ap.add_argument("--latest", action="store_true", help="newest run per model")
    ap.add_argument("--dry-run", action="store_true", help="skip commit and push")
    ap.add_argument("--message", help="commit message")
    a = ap.parse_args()

    if not a.run_ids and not a.latest:
        ap.error("give at least one run_id, or --latest")

    # 1. export -- a fault aborts the publish rather than entering the table as a zero
    export_cmd = [sys.executable, "tools/export_results.py"]
    export_cmd += ["--latest"] if a.latest else a.run_ids
    r = run(export_cmd)
    print(r.stdout.rstrip())
    if r.returncode != 0:
        print(r.stderr.rstrip(), file=sys.stderr)
        print("\nPUBLISH ABORTED: the run has harness failures. Nothing was committed.",
              file=sys.stderr)
        return 1

    # 2. charts
    c = run([sys.executable, "tools/make_charts.py"])
    if c.returncode != 0:
        print(c.stdout, c.stderr, file=sys.stderr)
        print("\nPUBLISH ABORTED: chart generation failed.", file=sys.stderr)
        return 1
    print("charts regenerated")

    # 3. README table
    if rewrite_readme():
        print("README results table regenerated")
    else:
        print("README results table already current")

    if a.dry_run:
        print("\nDRY RUN: not committing. Files above are written in the working tree.")
        return 0

    # 4. commit only if there is something to commit
    # tools/ is included on purpose: a pipeline change must land with the run it
    # publishes, or the published repo cannot reproduce what produced the numbers.
    run(["git", "add", "-A", "results", "assets", "README.md", "tools"])
    staged = run(["git", "diff", "--cached", "--quiet"])
    if staged.returncode == 0:
        print("\nNothing to publish: results, charts and table are already committed.")
        return 0

    label = "results: " + (", ".join(a.run_ids) if a.run_ids else "newest run per model")
    msg = a.message or f"{label}\n\nPublished {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}."
    cm = run(["git", "commit", "-m", msg])
    print(cm.stdout.strip() or cm.stderr.strip())
    if cm.returncode != 0:
        return 2

    # 5. push, then prove the remote moved
    before = run(["git", "rev-parse", "HEAD"]).stdout.strip()
    ps = run(["git", "push", "origin", "HEAD"])
    print((ps.stdout + ps.stderr).strip())
    if ps.returncode != 0:
        print("\nPUSH FAILED. The commit exists locally but the repository is unchanged.",
              file=sys.stderr)
        return 2

    remote = run(["git", "ls-remote", "origin", "HEAD"]).stdout.split()
    remote_sha = remote[0] if remote else ""
    if remote_sha != before:
        print(f"\nPUSH UNVERIFIED: local {before[:12]}, remote {remote_sha[:12] or 'unknown'}.",
              file=sys.stderr)
        return 2

    print(f"\nPublished and verified: {before[:12]} is on origin/HEAD")
    print(f"https://github.com/raidbeforebed/hermes-harness-bench/commit/{before[:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
