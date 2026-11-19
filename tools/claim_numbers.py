#!/usr/bin/env python3
"""Print every count and superlative the README asserts, recomputed from results/.

The README's table is generated; its prose is not, and prose is where a wrong number ships.
Adding a row or a repeat count invalidates the counts, so this prints the figures a sentence
must agree with instead of leaving them to memory. Read it before editing the narrative.

    python3 tools/claim_numbers.py
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry  # noqa: E402
import result_key  # noqa: E402

CORE_TWELVE = {f"{n:02d}" for n in range(1, 13)}


def load():
    per: dict[tuple, list[dict]] = defaultdict(list)
    for path in sorted(glob.glob(str(ROOT / "results" / "*.json"))):
        rec = json.load(open(path))
        key = (rec.get("model"), json.dumps(rec.get("config"), sort_keys=True))
        per[key].append(rec)
    return per


def task_number(task: str) -> str:
    return task.split("-", 1)[0]


def main() -> int:
    per = load()
    rows = len(per)
    perfect = {k: rs for k, rs in per.items() if all((r.get("score") or 0) == 1.0 for r in rs)}
    perfect_twelve = {
        k: rs for k, rs in per.items()
        if all((r.get("score") or 0) == 1.0 for r in rs if task_number(r["task"]) in CORE_TWELVE)
        and any(task_number(r["task"]) in CORE_TWELVE for r in rs)
    }
    print(f"rows (model + configuration): {rows}")
    print(f"perfect on everything they sat: {len(perfect)} of {rows}")
    print(f"perfect on the twelve core tasks: {len(perfect_twelve)} of {rows}")

    by_repeat: dict[int, list[tuple]] = defaultdict(list)
    for key, rs in perfect.items():
        by_repeat[len(rs)].append(key)
    for n in sorted(by_repeat):
        print(f"perfect rows with exactly {n} graded attempts: {len(by_repeat[n])}")

    # 'N models passed all R of R runs'
    plain_models = {k[0] for k, rs in perfect.items() if len(rs) == 36 or len(rs) == 54}
    print(f"models with every attempt perfect at their own repeat count: {len(plain_models)}")

    print("\nperfect rows, by mean wall and mean tokens:")
    table = []
    for key, rs in perfect.items():
        name = model_registry.label(key[0], json.loads(key[1]))
        table.append((statistics.mean(r["wall_seconds"] for r in rs), name,
                      statistics.mean(r.get("tokens_in", 0) + r.get("tokens_out", 0) for r in rs),
                      statistics.mean(r.get("turns") or 0 for r in rs), len(rs)))
    for wall, name, tokens, turns, n in sorted(table):
        print(f"  {wall:7.1f}s  {tokens:9.0f} tok  {turns:5.1f} turns  {n:2d} rows  {name}")

    print("\nrows that are not perfect, worst first:")
    for key, rs in sorted(per.items(),
                          key=lambda kv: statistics.mean(r["score"] for r in kv[1])):
        if key in perfect:
            continue
        scored = [r["score"] for r in rs]
        print(f"  {statistics.mean(scored):.4f}  {len(rs):2d} rows  {model_registry.label(key[0], json.loads(key[1]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
