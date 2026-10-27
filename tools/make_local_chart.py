#!/usr/bin/env python3
"""Render the local-model comparison as an SVG chart.

The four suite charts come from results/*.json: the balanced twelve-task design. The local
models were not measured on that design, they were measured on the three tasks where the local
models fail, recorded under results/head-to-head/. Drawing them on the suite charts would
compare a nine-attempt sample against a thirty-six-attempt one on a different task mix, so they
get their own chart here, with their own denominator stated on every bar.

    python3 tools/make_local_chart.py [--outdir assets]

Reads every artifact in results/head-to-head/ and draws one series per run, grouped by task.
Nothing is typed in by hand: add an artifact and its series appears.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_charts as base  # noqa: E402  (shared SVG helpers and escaping)
import model_registry as registry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
H2H = ROOT / "results" / "head-to-head"




def load() -> tuple[list[str], list[dict], list[str]]:
    """Series (one per run) and per-task values, from the committed artifacts."""
    series: dict[str, dict] = {}
    for f in sorted(glob.glob(str(H2H / "*.json"))):
        doc = json.loads(Path(f).read_text())
        for key, overall in (doc.get("overall") or {}).items():
            notes = overall.get("notes") or {}
            mode = notes.get("reasoning")
            base_model = key.split(" [")[0]
            label = registry.label(base_model)
            if mode:
                label = f"{label} · reasoning {mode}"
            sid = f"{key}"
            if sid in series:
                continue
            series[sid] = {
                "id": sid,
                "base": base_model,
                "label": label,
                "colour": registry.colour(base_model),
                "mode": mode,
                "mean": overall.get("mean_over_tasks"),
                "wall": overall.get("mean_wall_s"),
                "turns": overall.get("mean_turns"),
                "attempts": overall.get("attempts"),
                "tasks": {},
            }
        for row in doc.get("tasks", []):
            for key, cell in (row.get("cells") or {}).items():
                if key in series and cell:
                    series[key]["tasks"][row["task"]] = cell

    def rank(sid: str) -> tuple:
        """Registry order first, then best mean first.

        No model names here on purpose: two runs of one model differ only by their notes, so the
        tiebreak has to be data. The "on" arm has no suffix in its key, which is why sorting on the
        key alone put it after the other models.
        """
        s = series[sid]
        return (registry.sort_key(s["base"]), -(s["mean"] or 0))

    ids = sorted(series, key=rank)
    tasks = sorted({t for s in series.values() for t in s["tasks"]})
    return ids, [series[i] for i in ids], tasks


def render(series: list[dict], tasks: list[str]) -> str:
    width = 980
    label_w, plot_x, plot_w = 250, 280, 560
    group_h = 46 * len(series) + 34
    subs = base.wrap_sub(
        "Mean score per task at three attempts each, on the same llama.cpp build. This is the "
        "targeted local comparison, not the twelve-task suite: every bar states its own attempt "
        "count, and the two Groot series differ only in reasoning mode.", width)
    top = max(110, base.sub_end(subs) + 30)
    height = top + group_h * len(tasks) + 96
    out = base.svg_open(width, height, "Local models on three tasks", subs)
    # Declare the scope so the suite-chart audit in tools/make_charts.py, which requires every
    # model in results/ to appear in every chart, knows this chart is a different surface.
    out.insert(1, "<desc>scope: targeted - results/head-to-head, three tasks, not the twelve-task "
                  "suite. Models absent here were simply not run on these tasks.</desc>")

    # axis: score 0 to 1
    for frac in (0.25, 0.5, 0.75, 1.0):
        x = plot_x + plot_w * frac
        out.append(f'<line x1="{x:.1f}" y1="{top-12}" x2="{x:.1f}" '
                   f'y2="{top + group_h*len(tasks) - 12}" stroke="#21262d" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{top-18}" fill="#484f58" font-size="10" '
                   f'text-anchor="middle">{frac:.2f}</text>')

    for gi, task in enumerate(tasks):
        gy = top + gi * group_h
        out.append(f'<text x="{plot_x-18}" y="{gy+18}" fill="#c9d1d9" font-size="12.5" '
                   f'text-anchor="end">{base.esc(task)}</text>')
        for si, s in enumerate(series):
            cell = s["tasks"].get(task)
            y = gy + 26 + si * 42
            if not cell:
                continue
            mean = cell["mean"]
            n = cell["attempts"]
            w = max(2.0, plot_w * mean)
            opacity = 0.45 if s["mode"] == "off" else 0.9
            out.append(f'<rect x="{plot_x}" y="{y}" width="{w:.1f}" height="16" rx="3" '
                       f'fill="{s["colour"]}" opacity="{opacity}"/>')
            out.append(f'<text x="{plot_x+w+9:.1f}" y="{y+12.5}" fill="#e6edf3" font-size="11">'
                       f'{mean:.3f} over {n}</text>')

    # legend
    ly = top + group_h * len(tasks) - 4
    lx = plot_x
    for s in series:
        out.append(f'<rect x="{lx:.1f}" y="{ly-9}" width="10" height="10" rx="2" '
                   f'fill="{s["colour"]}" opacity="{0.45 if s["mode"] == "off" else 0.9}"/>')
        txt = f'{s["label"]} · {s["mean"]:.3f} mean · {s["wall"]}s · {s["turns"]} turns'
        out.append(f'<text x="{lx+15:.1f}" y="{ly}" fill="#9aa4b2" font-size="11">'
                   f'{base.esc(txt)}</text>')
        lx += 15 + len(txt) * 6.0 + 18
        if lx > width - 240:
            lx = plot_x
            ly += 16
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="assets")
    a = ap.parse_args()
    ids, series, tasks = load()
    if not series:
        sys.exit("no head-to-head artifacts under results/head-to-head/")
    outdir = ROOT / a.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "local-three-task.svg").write_text(render(series, tasks))
    print(f"wrote {outdir / 'local-three-task.svg'}  ({len(series)} series, {len(tasks)} tasks): "
          + ", ".join(s["label"] for s in series))


if __name__ == "__main__":
    main()
