#!/usr/bin/env python3
"""Render the benchmark results as SVG charts for the GitHub README.

GitHub renders committed SVG natively and it stays sharp at any zoom, which is why
these are plain SVG files rather than PNG screenshots or a Mermaid graph.

No third-party imports: the point of this repository is that it runs anywhere.

    python3 tools/make_charts.py [--outdir assets]

Charts produced:
  passes.svg      runs passed and tasks stable, per model
  tokens.svg      mean tokens per task on a linear axis
  turns.svg       mean turns per task
  pareto.svg      quality against token cost, the trade-off view

Every figure is read from results/*.json. Nothing is typed in by hand, so the charts
cannot drift from the data.
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Display order and colours. Chosen so the two perfect-scoring cloud models are the
# warm colours and the local model is distinct, since it is the one readers care about
# most in a cost chart.
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
COLOUR = {
    "deepseek-v4.1-flash:1017": "#4fd1a5",
    "glm-5.3-flash:cloud": "#7aa2f7",
    "minimax-m3.1-flash:cloud": "#bb9af7",
    "beam:cloud": "#e0af68",
    "Qwen3.6-35B-A3B-UD-Q4_K_M": "#f7768e",
    "gemma4:cloud": "#9ece6a",
    "poolside/laguna-s-2.1:free": "#2ac3de",
    "poolside/laguna-xs-2.1:free": "#c0caf5",
    "poolside/laguna-s-2.1": "#2ac3de",
    "poolside/laguna-xs-2.1": "#c0caf5",
}
ORDER = list(LABEL)


def load() -> dict[str, dict]:
    """Aggregate results/*.json into one summary row per model."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for f in glob.glob(str(RESULTS / "*.json")):
        d = json.loads(Path(f).read_text())
        by_model[d["model"]].append(d)

    out: dict[str, dict] = {}
    for model, rows in by_model.items():
        per_task: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            per_task[r["task"]].append(r["score"])
        stable = sum(1 for v in per_task.values() if all(s >= 1.0 for s in v))
        passed = sum(1 for r in rows if r["score"] >= 1.0)
        # Only runs that reported usage. A timeout kills the process before the usage
        # block is emitted, so those runs are unknown rather than zero. Averaging the
        # known ones is the least wrong choice and the charts say so.
        tok = [(r.get("tokens_in") or 0) + (r.get("tokens_out") or 0)
               for r in rows if r.get("tokens_in")]
        turns = [r["turns"] for r in rows if isinstance(r.get("turns"), int)]
        wall = [r["wall_seconds"] for r in rows if r.get("wall_seconds")]
        out[model] = {
            "label": LABEL.get(model, model),
            "colour": COLOUR.get(model, "#888"),
            "runs": len(rows),
            "passed": passed,
            "tasks": len(per_task),
            "stable": stable,
            "tokens": int(statistics.mean(tok)) if tok else 0,
            "tokens_known": len(tok),
            "turns": statistics.mean(turns) if turns else 0,
            "wall": statistics.mean(wall) if wall else 0,
        }
    return out


def esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def svg_open(w: int, h: int, title: str, sub: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="ui-sans-serif, -apple-system, '
        f'Segoe UI, Helvetica, Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        f'<text x="28" y="36" fill="#e6edf3" font-size="17" font-weight="600">{esc(title)}</text>',
        f'<text x="28" y="58" fill="#7d8590" font-size="12">{esc(sub)}</text>',
    ]


def bar_chart(rows: list[dict], key: str, unit: str, title: str, sub: str,
              fmt, width: int = 760) -> str:
    """Horizontal bars: easier to read model names than a vertical chart."""
    label_w, bar_x, bar_w = 190, 210, 470
    row_h, top = 34, 88
    h = top + row_h * len(rows) + 44
    out = svg_open(width, h, title, sub)
    mx = max(r[key] for r in rows) or 1

    # faint vertical guides at 25/50/75/100 percent of max
    for frac in (0.25, 0.5, 0.75, 1.0):
        x = bar_x + bar_w * frac
        out.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" '
                   f'y2="{top + row_h*len(rows) - 6}" stroke="#21262d" stroke-width="1"/>')

    for i, r in enumerate(rows):
        y = top + i * row_h
        bw = max(2.0, bar_w * (r[key] / mx))
        out.append(f'<text x="{bar_x-14}" y="{y+15}" fill="#c9d1d9" font-size="13" '
                   f'text-anchor="end">{esc(r["label"])}</text>')
        out.append(f'<rect x="{bar_x}" y="{y+2}" width="{bw:.1f}" height="20" rx="4" '
                   f'fill="{r["colour"]}" opacity="0.85"/>')
        out.append(f'<text x="{bar_x+bw+10:.1f}" y="{y+16}" fill="#e6edf3" '
                   f'font-size="12.5" font-weight="600">{esc(fmt(r))}</text>')
    out.append("</svg>")
    return "\n".join(out)


def grouped_chart(rows: list[dict], title: str, sub: str) -> str:
    """Two bars per model: runs passed (of 36) and tasks stable (of 12).

    Both bars are drawn as a FRACTION of their own denominator, so bar length always
    means "proportion passed". An earlier version scaled the 12-task bar against the
    36-run axis, which made 12/12 and 36/36 render as identical lengths and implied
    the two numbers measured the same thing.
    """
    width = 900
    label_w, bar_x, bar_w = 210, 230, 500
    right_pad = 150          # room for the value labels, which were previously clipped
    row_h, top = 48, 96
    h = top + row_h * len(rows) + 56
    out = svg_open(width, h, title, sub)
    bar_h = 15

    # guides at 25/50/75/100 percent of the proportion axis
    for frac in (0.25, 0.5, 0.75, 1.0):
        x = bar_x + bar_w * frac
        out.append(f'<line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" '
                   f'y2="{top + row_h*len(rows) - 8}" stroke="#21262d" stroke-width="1"/>')
        out.append(f'<text x="{x:.1f}" y="{top-16}" fill="#484f58" font-size="10" '
                   f'text-anchor="middle">{int(frac*100)}%</text>')

    for i, r in enumerate(rows):
        y = top + i * row_h
        out.append(f'<text x="{bar_x-16}" y="{y+24}" fill="#c9d1d9" font-size="13" '
                   f'text-anchor="end">{esc(r["label"])}</text>')
        # solid: proportion of runs passed
        f1 = r["passed"] / r["runs"]
        w1 = max(2.0, bar_w * f1)
        out.append(f'<rect x="{bar_x}" y="{y+4}" width="{w1:.1f}" height="{bar_h}" rx="3" '
                   f'fill="{r["colour"]}" opacity="0.9"/>')
        out.append(f'<text x="{bar_x+w1+10:.1f}" y="{y+16}" fill="#e6edf3" font-size="11.5">'
                   f'{r["passed"]}/{r["runs"]} runs</text>')
        # faded: proportion of tasks passed on every attempt
        f2 = r["stable"] / r["tasks"]
        w2 = max(2.0, bar_w * f2)
        out.append(f'<rect x="{bar_x}" y="{y+24}" width="{w2:.1f}" height="{bar_h}" rx="3" '
                   f'fill="{r["colour"]}" opacity="0.4"/>')
        out.append(f'<text x="{bar_x+w2+10:.1f}" y="{y+36}" fill="#9aa4b2" font-size="11.5">'
                   f'{r["stable"]}/{r["tasks"]} tasks 3/3</text>')

    ly = top + row_h * len(rows) + 16
    out.append(f'<text x="{bar_x}" y="{ly}" fill="#7d8590" font-size="11.5">'
               f'solid = share of the 36 runs that passed   ·   '
               f'faded = share of the 12 tasks that passed on all 3 attempts</text>')
    out.append("</svg>")
    return "\n".join(out)


def pareto(rows: list[dict], title: str, sub: str) -> str:
    """Quality against mean tokens per task.

    Points are close together in the x range (17k to 25k) and three of them share
    y = 100%, so naive centre-anchored labels collide. Each point gets an explicit
    offset and anchor so the labels stay apart at any zoom.
    """
    w, h = 900, 470
    left, right, top, bottom = 128, 96, 92, 66
    pw, ph = w - left - right, h - top - bottom
    out = svg_open(w, h, title, sub)

    xs = [r["tokens"] for r in rows]
    ys = [r["passed"] / r["runs"] for r in rows]
    x1 = max(xs) * 1.12
    y0, y1 = 0.84, 1.015

    def X(v): return left + pw * (v / x1)
    def Y(v): return top + ph * (1 - (v - y0) / (y1 - y0))

    for frac in (0.85, 0.90, 0.95, 1.00):
        yy = Y(frac)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+pw}" y2="{yy:.1f}" '
                   f'stroke="#21262d" stroke-width="1"/>')
        out.append(f'<text x="{left-12}" y="{yy+4:.1f}" fill="#7d8590" font-size="11" '
                   f'text-anchor="end">{frac:.0%}</text>')
    for v in range(0, int(x1) + 1, 5000):
        xx = X(v)
        out.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+ph}" '
                   f'stroke="#161b22" stroke-width="1"/>')
        out.append(f'<text x="{xx:.1f}" y="{top+ph+22}" fill="#7d8590" font-size="11" '
                   f'text-anchor="middle">{v//1000}k</text>')

    out.append(f'<text x="{left+pw/2:.0f}" y="{h-18}" fill="#7d8590" font-size="12" '
               f'text-anchor="middle">mean tokens per task</text>')
    yax = 30
    out.append(f'<text x="{yax}" y="{top+ph/2:.0f}" fill="#7d8590" font-size="12" '
               f'text-anchor="middle" '
               f'transform="rotate(-90 {yax} {top+ph/2:.0f})">share of runs passed</text>')

    # Explicit offsets per model: (dx, dy, text-anchor). Hand-tuned because the
    # three perfect models sit within a few thousand tokens of each other.
    OFFSET = {
        "deepseek-v4.1-flash:1017":  (0, -20, "end"),
        "glm-5.3-flash:cloud":       (-14, 26, "end"),
        "minimax-m3.1-flash:cloud":  (14, 26, "end"),
        "beam:cloud":                (0, -20, "middle"),
        "Qwen3.6-35B-A3B-UD-Q4_K_M": (0, 28, "middle"),
    }
    data = {}
    for f in glob.glob(str(RESULTS / "*.json")):
        d = json.loads(Path(f).read_text())
        data.setdefault(d["model"], []).append(d)

    for r in rows:
        model = next((m for m in ORDER if LABEL.get(m) == r["label"]), None)
        cx, cy = X(r["tokens"]), Y(r["passed"] / r["runs"])
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{r["colour"]}" '
                   f'opacity="0.95"/>')
        dx, dy, anchor = OFFSET.get(model, (0, -18, "middle"))
        out.append(f'<text x="{cx+dx:.1f}" y="{cy+dy:.1f}" fill="#c9d1d9" font-size="11.5" '
                   f'text-anchor="{anchor}">{esc(r["label"])}</text>')

    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="assets")
    args = ap.parse_args()

    data = load()
    rows = [data[m] for m in ORDER if m in data]
    if not rows:
        sys.exit("no results found in results/")

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Passes, ordered by runs passed then stability
    rows_p = sorted(rows, key=lambda r: (-r["passed"], -r["stable"]))
    (outdir / "passes.svg").write_text(grouped_chart(
        rows_p,
        "Reliability: 3 repeats per task",
        "Twelve tasks, each attempted three times. 36 runs per model."))

    # Tokens, cheapest first
    rows_t = sorted(rows, key=lambda r: r["tokens"])
    (outdir / "tokens.svg").write_text(bar_chart(
        rows_t, "tokens", "tokens",
        "Token cost per task",
        "Mean tokens per task, over runs that reported usage. A timed-out run is killed "
        "before usage is emitted, so it is unknown rather than zero.",
        lambda r: f"{r['tokens']:,}" + ("" if r["tokens_known"] == r["runs"]
                                       else f"  (from {r['tokens_known']} of {r['runs']} runs)")))

    # Turns, fewest first
    rows_u = sorted(rows, key=lambda r: r["turns"])
    (outdir / "turns.svg").write_text(bar_chart(
        rows_u, "turns", "turns",
        "Turns per task",
        "Mean tool-calling turns. The Hermes harness averages 22.6 turns on "
        "Harness-Bench, so fewer turns per task is less context pressure.",
        lambda r: f"{r['turns']:.1f}"))

    # Pareto
    (outdir / "pareto.svg").write_text(pareto(
        rows,
        "Quality against token cost",
        "Upper left is best: reliable and cheap. The local model sits far left and "
        "lower, which is the trade-off it offers."))

    for f in sorted(outdir.glob("*.svg")):
        print(f"wrote {f.relative_to(ROOT)}  ({f.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
