#!/usr/bin/env python3
"""Publish-integrity test: every published result reaches every published surface.

The defect this exists for: Ornith 1.5 122B was exported, scored 36/36 in the README
table, and appeared in none of the four charts, because make_charts.py iterated a
hardcoded model list and skipped anything not in it. Nothing failed. The publish
"worked". A reader comparing the table with the chart would have seen a missing row and
had no way to tell whether the model was omitted on purpose.

So this test asserts the two directions of the same rule:

  POSITIVE  every model in results/ is named in every chart, and in the README table.
  NEGATIVE  a chart that omits a model is reported, not silently accepted.

Run it after any change to tools/, and before trusting a publish:

    python3 tests/test_publish.py
"""

from __future__ import annotations

import glob
import importlib.util
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
TOOLS = ROOT / "tools"
README = ROOT / "README.md"
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None, name
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_charts = load_tool("make_charts")
registry = load_tool("model_registry")


def models_in_results() -> list[str]:
    seen = set()
    for f in glob.glob(str(RESULTS / "*.json")):
        d = json.loads(Path(f).read_text())
        if d.get("model"):
            seen.add(d["model"])
    return sorted(seen)


def readme_table_rows() -> list[str]:
    text = README.read_text()
    if START not in text or END not in text:
        return []
    block = text.split(START, 1)[1].split(END, 1)[0]
    rows = [ln for ln in block.splitlines() if ln.startswith("|")]
    rows = [r for r in rows if not set(r) <= set("|- ")]
    return rows[1:] if rows else []      # drop the header row


def main() -> int:
    failures = 0
    models = models_in_results()
    print(f"results/ holds {len(models)} models, {len(glob.glob(str(RESULTS / '*.json')))} records")

    # 1. display registry coverage
    print("\nREGISTRY  every model in results/ must have a display name and colour")
    for m in models:
        named = m in registry.LABEL
        coloured = m in registry.COLOUR
        ok = named and coloured
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {m:40} {registry.label(m)}")
        if not ok:
            print(f"        add it to {TOOLS / 'model_registry.py'}")

    # 2. charts, built into a throwaway directory
    print("\nCHARTS  every chart must name every model")
    with tempfile.TemporaryDirectory(prefix="hbcharts_") as tmp:
        r = subprocess.run([sys.executable, str(TOOLS / "make_charts.py"), "--outdir", tmp],
                           capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            failures += 1
            print(f"  FAIL make_charts.py exited {r.returncode}\n{r.stdout}{r.stderr}")
        svgs = sorted(glob.glob(str(Path(tmp) / "*.svg")))
        if len(svgs) != 4:
            failures += 1
            print(f"  FAIL expected 4 charts, got {len(svgs)}")
        for svg in svgs:
            body = Path(svg).read_text()
            missing = [m for m in models if make_charts.esc(registry.label(m)) not in body]
            failures += 1 if missing else 0
            print(f"  {'ok  ' if not missing else 'FAIL'} {Path(svg).name:12} "
                  f"{'names all ' + str(len(models)) + ' models' if not missing else 'missing ' + ', '.join(missing)}")

    # 3. README table
    print("\nREADME  one row per model, display names only, numbers recomputed")
    rows = readme_table_rows()
    failures += 0 if len(rows) == len(models) else 1
    print(f"  {'ok  ' if len(rows) == len(models) else 'FAIL'} table has {len(rows)} rows "
          f"for {len(models)} models")
    table = "\n".join(rows)
    for m in models:
        present = registry.label(m) in table
        failures += 0 if present else 1
        print(f"  {'ok  ' if present else 'FAIL'} {registry.label(m)}")
    raw = [line for line in rows if "/" in line.split("|")[1]]
    failures += 0 if not raw else 1
    print(f"  {'ok  ' if not raw else 'FAIL'} no raw provider ids in the Model column"
          + ("" if not raw else f": {raw}"))

    # 4. NEGATIVE: a chart that drops a model must be reported
    print("\nNEGATIVE  an incomplete chart must be reported, not accepted")
    with tempfile.TemporaryDirectory(prefix="hbneg_") as tmp:
        fake = [{"model": "x/y:free", "label": "Ghost Model", "colour": "#888",
                 "runs": 3, "passed": 3, "tasks": 1, "stable": 1, "tokens": 1,
                 "tokens_known": 3, "turns": 1.0, "wall": 1.0, "repeats": 3}]
        Path(tmp, "partial.svg").write_text('<svg><text>DeepSeek V4.1 Flash</text></svg>')
        problems = make_charts.audit(Path(tmp), fake)
        ok = any("Ghost Model" in p for p in problems)
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} audit reported: {problems}")

    unknown = registry.unregistered(["brand/new:free"])
    ok = unknown == ["brand/new:free"]
    failures += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'} unregistered model is flagged: {unknown}")

    # 5. PROSE: hand-written counts in the README must agree with results/
    print("\nPROSE  count claims in the README must match the data they describe")
    per: dict[str, list[dict]] = {}
    for f in glob.glob(str(RESULTS / "*.json")):
        d = json.loads(Path(f).read_text())
        per.setdefault(d["model"], []).append(d)
    perfect = sorted(m for m, rs in per.items() if all((r.get("score") or 0) == 1.0 for r in rs))
    max_repeat = max((r.get("repeat") or 1) for rs in per.values() for r in rs)
    graded = {m for m, rs in per.items() if max((r.get("repeat") or 1) for r in rs) >= max_repeat}
    perfect_graded = [m for m in perfect if m in graded]
    words = {w: i for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split())}
    words.update({str(i): i for i in range(0, 21)})
    text = README.read_text()

    patterns = [
        (rf"\b(\w+) of (?:the )?(?:{'|'.join(w for w in words if not w.isdigit())}|\d+) "
         rf"(?:models|rows) are perfect", len(perfect), "rows at 100%"),
        (r"\b(\w+) models passed all \*{0,2}(\d+) of (\d+)\*{0,2} runs", len(perfect_graded),
         f"models perfect over {max_repeat} runs"),
        (r"\b(\w+) models passed all (\d+) runs", len(perfect_graded),
         f"models perfect over {max_repeat} runs"),
        (r"fastest of the (\w+) perfect", len(perfect), "perfect rows"),
    ]
    claim_reported = False
    for pattern, want, desc in patterns:
        for hit in re.finditer(pattern, text):
            claim_reported = True
            got = words.get(hit.group(1).lower())
            ok = got == want
            failures += 0 if ok else 1
            print(f"  {'ok  ' if ok else 'FAIL'} \"{hit.group(0)}\" claims "
                  f"{hit.group(1)} = {got}; data says {want} {desc}")
    if not claim_reported:
        failures += 1
        print("  FAIL no count claim found in the README; the prose check cannot run")

    # denominator: "N of the M rows are perfect" must also match the model count
    for hit in re.finditer(r"(\w+) of (?:the )?(\w+) (?:models|rows) are perfect", text):
        got = words.get(hit.group(2).lower())
        ok = got == len(per)
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} \"{hit.group(0)}\" denominator = {got}; "
              f"data says {len(per)} models")

    # a named per-model count in prose must match that model's record
    beam = [m for m in per if "beam" in m.lower()]
    if beam:
        rs = per[beam[0]]
        passed = sum(1 for r in rs if (r.get("score") or 0) == 1.0)
        for hit in re.finditer(r"Beam passes (\d+)\s+of\s+(\d+)\s+runs", text):
            n, d = int(hit.group(1)), int(hit.group(2))
            ok = n == passed and d == len(rs)
            failures += 0 if ok else 1
            print(f"  {'ok  ' if ok else 'FAIL'} \"{hit.group(0)}\" vs data "
                  f"{passed}/{len(rs)} for {registry.label(beam[0])}")

    print(f"\n{'PUBLISH SURFACES CONSISTENT' if failures == 0 else str(failures) + ' PUBLISH PROBLEMS'}")
    print("note: asserted both ways: every result reaches every surface, and a surface "
          "that drops one is caught.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
