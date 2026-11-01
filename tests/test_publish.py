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

A row is a (model, configuration) pair, not a model. The same weights run with thinking on
and with thinking off are two instruments and must reach every surface twice, which is why
every check below compares labels rather than model ids, and why KEY asserts that every
published file name is exactly what the shared rule produces.

Run it after any change to tools/, and before trusting a publish:

    python3 tests/test_publish.py
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
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
DECISION_START = "<!-- DECISION:START -->"
DECISION_END = "<!-- DECISION:END -->"

# Rows that were published on the twelve-task suite and retired on 2026-11-01, when six tasks
# were added and four originals rewritten. An eighteen-task number cannot be substituted for a
# twelve-task one, so the old row must not sit in results/ waiting to be compared with it.
# Removing it from the data is the only way that cannot be undone by a later --all or --latest.
RETIRED_MODELS = {"lfm3-64k"}


def load_tool(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec is not None and spec.loader is not None, name
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sys.path.insert(0, str(TOOLS))
make_charts = load_tool("make_charts")
registry = load_tool("model_registry")
result_key = load_tool("result_key")


def rows_in_results() -> list[dict]:
    """One row per (model, configuration), in the order the tables use.

    Grouping by model alone would silently merge two configurations of one model back into
    the single row this whole change exists to split.
    """
    grouped: dict[str, dict] = {}
    for f in glob.glob(str(RESULTS / "*.json")):
        d = json.loads(Path(f).read_text())
        if not d.get("model"):
            continue
        key = result_key.key_of(d)
        row = grouped.get(key)
        if row is None:
            config = result_key.config_of(d)
            row = grouped[key] = {
                "key": key,
                "model": d["model"],
                "config": config,
                "label": registry.label(d["model"], config),
                "records": [],
            }
        row["records"].append(d)
    rows = list(grouped.values())
    rows.sort(key=lambda r: registry.sort_key(r["model"], r["config"]))
    return rows


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
    rows = rows_in_results()
    models = sorted({r["model"] for r in rows})
    print(f"results/ holds {len(rows)} rows over {len(models)} models, "
          f"{len(glob.glob(str(RESULTS / '*.json')))} records")

    # 1. display registry coverage
    print("\nREGISTRY  every model in results/ must have a display name and colour")
    for m in models:
        named = m in registry.LABEL
        coloured = m in registry.COLOUR
        ok = named and coloured
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {m:40} {registry.base_label(m)}")
        if not ok:
            print(f"        add it to {TOOLS / 'model_registry.py'}")

    # 2. labels: every row says which configuration it was measured under
    print("\nCONFIG  every published row is labelled with its configuration")
    for r in rows:
        qualified = r["label"] != registry.base_label(r["model"])
        named = result_key.descriptor(r["config"]) in r["label"]
        ok = qualified and named
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {r['label']}")
    duplicate = {lab for lab in (r["label"] for r in rows)
                 if sum(1 for x in rows if x["label"] == lab) > 1}
    failures += 0 if not duplicate else 1
    print(f"  {'ok  ' if not duplicate else 'FAIL'} every row has its own label"
          + (f": {sorted(duplicate)}" if duplicate else ""))

    # 3. the row key: one model must not occupy two file sets, and no row may be filed twice.
    #    Both halves of the 2026-11-01 defect are checked here: two spellings for one model
    #    (which split Ornith into 36 records under one stem and 54 under another), and two
    #    files claiming the same (row, task, repeat), which is what a re-export under a
    #    changed name produces.
    print("\nKEY  one file set per model, one file per row, new records filed by the rule")
    stems: dict[str, set[str]] = {}
    seen: dict[tuple, str] = {}
    duplicated: list[str] = []
    unfiled: list[str] = []
    legacy_names = 0
    files = sorted(glob.glob(str(RESULTS / "*.json")))
    for f in files:
        d = json.loads(Path(f).read_text())
        name = Path(f).name
        stem = name.rsplit("__", 2)[0]
        stems.setdefault(d.get("model"), set()).add(stem)
        slot = (result_key.key_of(d), d.get("task"), d.get("repeat"))
        if slot in seen:
            duplicated.append(f"{seen[slot]} and {name}")
        seen[slot] = name
        expected = result_key.published_name(d)
        if name != expected:
            if isinstance(d.get("config"), dict):
                # A record that states its configuration was written under this rule, so a
                # name it does not produce means a writer bypassed the rule. Hard failure.
                unfiled.append(f"{name} != {expected}")
            else:
                # A record from before the configuration existed keeps the name it was
                # published under: renaming it would file the same attempts twice, which is
                # the worse defect. Counted and reported rather than silently accepted.
                legacy_names += 1
    split = {m: sorted(s) for m, s in stems.items() if len(s) > 1}
    failures += 0 if not split else 1
    print(f"  {'ok  ' if not split else 'FAIL'} each model occupies exactly one file-set stem"
          + ("" if not split else f": {split}"))
    failures += 0 if not duplicated else 1
    print(f"  {'ok  ' if not duplicated else 'FAIL'} no (row, task, repeat) is filed twice"
          + ("" if not duplicated else f": {duplicated[:5]}"))
    failures += 0 if not unfiled else 1
    print(f"  {'ok  ' if not unfiled else 'FAIL'} every record that states a configuration is "
          f"filed under the shared rule's name"
          + ("" if not unfiled else f": {unfiled[:5]}"))
    print(f"  ok   {len(files)} file name(s) checked; {legacy_names} historical name(s) kept "
          f"as published, {len(files) - legacy_names} filed by the rule")

    # 4. charts, built into a throwaway directory
    print("\nCHARTS  every chart must name every row")
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
            missing = [row["label"] for row in rows
                       if make_charts.esc(row["label"]) not in body]
            failures += 1 if missing else 0
            print(f"  {'ok  ' if not missing else 'FAIL'} {Path(svg).name:12} "
                  f"{'names all ' + str(len(rows)) + ' rows' if not missing else 'missing ' + ', '.join(missing)}")

    # 5. README table
    print("\nREADME  one row per model and configuration, display names only, numbers recomputed")
    table_rows = readme_table_rows()
    failures += 0 if len(table_rows) == len(rows) else 1
    print(f"  {'ok  ' if len(table_rows) == len(rows) else 'FAIL'} table has {len(table_rows)} rows "
          f"for {len(rows)} published rows")
    table = "\n".join(table_rows)
    for row in rows:
        present = row["label"] in table
        failures += 0 if present else 1
        print(f"  {'ok  ' if present else 'FAIL'} {row['label']}")
    raw = [line for line in table_rows if "/" in line.split("|")[1]]
    failures += 0 if not raw else 1
    print(f"  {'ok  ' if not raw else 'FAIL'} no raw provider ids in the Model column"
          + ("" if not raw else f": {raw}"))

    # 6. NEGATIVE: a chart that drops a model must be reported
    print("\nNEGATIVE  an incomplete chart must be reported, not accepted")
    with tempfile.TemporaryDirectory(prefix="hbneg_") as tmp:
        fake = [{"model": "x/y:free", "label": "Ghost Model (cloud, configuration unrecorded)",
                 "colour": "#888", "runs": 3, "passed": 3, "tasks": 1, "stable": 1,
                 "tokens": 1, "tokens_known": 3, "turns": 1.0, "wall": 1.0, "repeats": 3}]
        Path(tmp, "partial.svg").write_text('<svg><text>DeepSeek V4.1 Flash</text></svg>')
        problems = make_charts.audit(Path(tmp), fake)
        ok = any("Ghost Model" in p for p in problems)
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} audit reported: {problems}")

    unknown = registry.unregistered(["brand/new:free"])
    ok = unknown == ["brand/new:free"]
    failures += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'} unregistered model is flagged: {unknown}")

    # 7. PROSE: hand-written counts in the README must agree with results/
    print("\nPROSE  count claims in the README must match the data they describe")
    per: dict[str, list[dict]] = {r["key"]: r["records"] for r in rows}
    perfect = sorted(k for k, rs in per.items() if all((r.get("score") or 0) == 1.0 for r in rs))

    def perfect_with_runs(runs_n: int) -> list[str]:
        """Rows that are perfect and hold exactly `runs_n` graded attempts.

        Keyed on the denominator the sentence states, not on the largest repeat count in the
        tree: "passed all 36 of 36 runs" means 12 tasks x 3 repeats for that model, and a row
        carrying four attempts on three tasks must not move what the sentence is checked
        against.
        """
        return [k for k, rs in per.items()
                if len(rs) == runs_n and all((r.get("score") or 0) == 1.0 for r in rs)]
    words = {w: i for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve".split())}
    words.update({str(i): i for i in range(0, 21)})
    text = README.read_text()

    patterns = [
        (rf"\b(\w+) of (?:the )?(?:{'|'.join(w for w in words if not w.isdigit())}|\d+) "
         rf"(?:models|rows) are perfect", len(perfect), "rows at 100%"),
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

    # "N models passed all R of R runs": both halves recomputed against the stated denominator
    for hit in re.finditer(r"\b(\w+) models passed all \*{0,2}(\d+) of (\d+)\*{0,2} runs", text):
        n, d = int(hit.group(2)), int(hit.group(3))
        want = len(perfect_with_runs(d))
        got = words.get(hit.group(1).lower())
        ok = got == want and n == d
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} \"{hit.group(0)}\" claims {hit.group(1)} = {got}; "
              f"data says {want} models perfect with {d} graded runs")

    # denominator: "N of the M rows are perfect" must also match the row count
    for hit in re.finditer(r"(\w+) of (?:the )?(\w+) (?:models|rows) are perfect", text):
        got = words.get(hit.group(2).lower())
        ok = got == len(per)
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} \"{hit.group(0)}\" denominator = {got}; "
              f"data says {len(per)} rows")

    # a named per-row count in prose must match that row's records
    beam = [k for k in per if "beam" in k.lower()]
    if beam:
        rs = per[beam[0]]
        passed = sum(1 for r in rs if (r.get("score") or 0) == 1.0)
        for hit in re.finditer(r"Beam passes (\d+)\s+of\s+(\d+)\s+runs", text):
            n, d = int(hit.group(1)), int(hit.group(2))
            ok = n == passed and d == len(rs)
            failures += 0 if ok else 1
            print(f"  {'ok  ' if ok else 'FAIL'} \"{hit.group(0)}\" vs data "
                  f"{passed}/{len(rs)} for {beam[0]}")

    # 8. DECISION table: one row per model and configuration, numbers recomputed
    print("\nDECISION  metrics table must cover every row, match the data, and name one bundle")
    text = README.read_text()
    if DECISION_START not in text or DECISION_END not in text:
        failures += 1
        print(f"  FAIL no {DECISION_START} / {DECISION_END} block in the README")
        decision_rows = []
    else:
        block = text.split(DECISION_START, 1)[1].split(DECISION_END, 1)[0]
        lines = [ln for ln in block.splitlines() if ln.startswith("|")]
        lines = [ln for ln in lines if not set(ln) <= set("|- ")]
        decision_rows = lines[1:] if lines else []
        ok = len(decision_rows) == len(rows)
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} decision table has {len(decision_rows)} rows "
              f"for {len(rows)} published rows")

    per_row_cells: dict[str, list[str]] = {}
    for row in decision_rows:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) >= 10:
            per_row_cells[cells[0]] = cells

    for r in rows:
        label = r["label"]
        cells = per_row_cells.get(label)
        if cells is None:
            failures += 1
            print(f"  FAIL {label} has no decision row")
            continue
        rs = r["records"]
        per_repeat: dict[int, list[float]] = {}
        for x in rs:
            per_repeat.setdefault(x.get("repeat") or 1, []).append(float(x.get("score") or 0.0))
        want_mean = sum(sum(v) / len(v) for v in per_repeat.values()) / len(per_repeat)
        printed = re.match(r"\*{0,2}([0-9.]+)", cells[2])
        got_mean = float(printed.group(1)) if printed else None
        ok_mean = got_mean is not None and abs(got_mean - want_mean) < 5e-5
        ok_reps = cells[1] == str(len(per_repeat))
        # A 3-column row is incomplete; anything else is a build error in the table itself.
        ok = ok_mean and ok_reps
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {label:48} mean {cells[2]!r} vs data "
              f"{want_mean:.4f}; repeats {cells[1]} vs {len(per_repeat)}")

    # The comparability rule: every row that carries a bundle hash must carry the *same* one,
    # because there is one frozen suite. Two different hashes mean two instruments presented
    # as one table; a row whose bundle covers fewer tasks than it has records is mismatched.
    bundles: dict[str, tuple[int, str]] = {}
    for row, cells in per_row_cells.items():
        hit = re.search(r"(\d+) tasks, `([0-9a-f]{12})`", cells[-1])
        if hit:
            bundles[row] = (int(hit.group(1)), hit.group(2))
        elif "MIXED" in cells[-1]:
            failures += 1
            print(f"  FAIL {row} spans more than one task bundle")
    hashes = {h for _n, h in bundles.values()}
    ok = len(hashes) <= 1
    failures += 0 if ok else 1
    print(f"  {'ok  ' if ok else 'FAIL'} {len(bundles)} row(s) carry a bundle hash, "
          f"{len(hashes)} distinct: {sorted(hashes)}")
    for row, (n, h) in sorted(bundles.items()):
        key = next((r["key"] for r in rows if r["label"] == row), None)
        rs = per.get(key, []) if key else []
        own = {x["task_bundle_sha256"] for x in rs
               if isinstance(x.get("task_bundle_sha256"), str)}
        tasks_seen = len({x["task"] for x in rs})
        ok = n == tasks_seen and any(x.startswith(h) for x in own)
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {row:48} {n} tasks over {tasks_seen} in results, "
              f"hash `{h}` matches its records")
    if not bundles:
        print("  note: no row carries a bundle hash; the comparability claim cannot be checked")

    # 9. RETIRED: a retired run and a retired twelve-task row must not be restorable
    print("\nRETIRED  twelve-task rows stay retired, not silently comparable")
    leaked_models = sorted(m for m in models if m in RETIRED_MODELS)
    failures += 0 if not leaked_models else 1
    print(f"  {'ok  ' if not leaked_models else 'FAIL'} no retired model id in results/"
          + (f": {leaked_models}" if leaked_models else f" ({', '.join(sorted(RETIRED_MODELS))} absent)"))
    runs_root = Path(os.environ.get("HBENCH_RUNS") or "/root/hermes-bench/runs")
    if runs_root.is_dir():
        marked = {d.name for d in runs_root.iterdir() if (d / "RETIRED.md").exists()}
        run_ids = {r.get("run_id") for rs in per.values() for r in rs if r.get("run_id")}
        leaked = sorted(marked & run_ids)
        ok = not leaked
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'FAIL'} {len(marked)} run(s) carry RETIRED.md; "
              f"{len(leaked)} of them reach results/"
              + (f": {leaked}" if leaked else ""))
    else:
        print(f"  skip no runs/ tree at {runs_root} (published-only checkout)")

    print(f"\n{'PUBLISH SURFACES CONSISTENT' if failures == 0 else str(failures) + ' PUBLISH PROBLEMS'}")
    print("note: asserted both ways: every result reaches every surface, and a surface "
          "that drops one is caught.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
