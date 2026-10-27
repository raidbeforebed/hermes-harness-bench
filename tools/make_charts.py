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

Every model present in results/ is drawn. An earlier version iterated a hardcoded
ORDER and so silently omitted any model missing from it, which is how a 36/36 result
stayed off every chart while appearing in the README table. Display names now come
from tools/model_registry.py; an unregistered model is drawn under its raw id in grey
and reported on stderr.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model_registry as registry  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


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
            "model": model,
            "label": registry.label(model),
            "colour": registry.colour(model),
            "repeats": max((r.get("repeat") or 1) for r in rows),
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


def wrap_sub(sub: str, width: int, fs: float = 12) -> list[str]:
    """Break a subtitle into lines that fit inside the chart width.

    Subtitles are written for meaning, not for length, and two of them ran off the right
    edge of a 760 px chart and were simply cut on GitHub. There is no text-measuring
    engine in an SVG, so this estimates 0.52 em per character, which is close enough to
    choose a break point and deliberately conservative.
    """
    budget = max(20, int((width - 56) / (fs * 0.52)))
    if len(sub) <= budget:
        return [sub]
    lines: list[str] = []
    cur = ""
    for word in sub.split():
        trial = f"{cur} {word}".strip()
        if len(trial) <= budget or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def sub_end(subs: list[str]) -> int:
    """Baseline y of the last subtitle line."""
    return 58 + 16 * (len(subs) - 1)


def svg_open(w: int, h: int, title: str, subs: list[str]) -> list[str]:
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
        f'viewBox="0 0 {w} {h}" font-family="ui-sans-serif, -apple-system, '
        f'Segoe UI, Helvetica, Arial, sans-serif">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        f'<text x="28" y="36" fill="#e6edf3" font-size="17" font-weight="600">{esc(title)}</text>',
    ]
    for i, line in enumerate(subs):
        out.append(f'<text x="28" y="{58 + 16 * i}" fill="#7d8590" font-size="12">'
                   f'{esc(line)}</text>')
    return out


def bar_chart(rows: list[dict], key: str, unit: str, title: str, sub: str,
              fmt, width: int = 760) -> str:
    """Horizontal bars: easier to read model names than a vertical chart."""
    label_w, bar_x, bar_w = 190, 210, 470
    row_h = 34
    subs = wrap_sub(sub, width)
    top = max(88, sub_end(subs) + 22)
    h = top + row_h * len(rows) + 44
    out = svg_open(width, h, title, subs)
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
    """Two bars per model: runs passed (of however many were run) and tasks stable.

    Both bars are drawn as a FRACTION of their own denominator, so bar length always
    means "proportion passed". An earlier version scaled the 12-task bar against the
    36-run axis, which made 12/12 and 36/36 render as identical lengths and implied
    the two numbers measured the same thing.

    The denominators are per-model because they are not all the same: a screening row
    has one attempt per task and a graded row has three. The value labels carry the
    real counts so a one-repeat row cannot be read as a stable one.
    """
    width = 900
    label_w, bar_x, bar_w = 210, 230, 500
    right_pad = 150          # room for the value labels, which were previously clipped
    row_h = 48
    subs = wrap_sub(sub, width)
    top = max(96, sub_end(subs) + 26)
    h = top + row_h * len(rows) + 56
    out = svg_open(width, h, title, subs)
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
                   f'{r["stable"]}/{r["tasks"]} tasks all-pass</text>')

    ly = top + row_h * len(rows) + 16
    out.append(f'<text x="{bar_x}" y="{ly}" fill="#7d8590" font-size="11.5">'
               f'solid = share of the runs that passed   ·   '
               f'faded = share of the 12 tasks that passed on every attempt</text>')
    out.append("</svg>")
    return "\n".join(out)


def text_box(x: float, y: float, text: str, fs: float, anchor: str):
    """Approximate bounding box for one <text> element: (x0, y0, x1, y1).

    An SVG carries no font metrics at build time, so widths are estimated at 0.55 em per
    character. That over-estimates for most strings, which is the safe direction for
    deciding whether two labels touch.
    """
    width = len(text) * fs * 0.55
    x0 = x - width / 2 if anchor == "middle" else (x - width if anchor == "end" else x)
    return (x0, y - fs * 0.8, x0 + width, y + fs * 0.25)


def overlaps(a, b, pad: float = 2.0) -> bool:
    return (a[0] - pad < b[2] and b[0] - pad < a[2]
            and a[1] - pad < b[3] and b[1] - pad < a[3])


def place_labels(candidates: list[dict], obstacles: list[tuple],
                 top: float, bottom: float) -> list[dict]:
    """Nudge each label vertically until it touches nothing already placed.

    Eight models on one scatter put several labels within 20 px of each other, and
    hand-tuned offsets stop working as soon as a model is added or a token count moves.
    This resolves collisions mechanically, top to bottom, with a four pixel gap so the
    result reads as separate lines rather than a paragraph, and flags anything it had to
    move so the caller can draw a leader line back to the point.
    """
    placed = list(obstacles)
    result: list[dict] = []
    for c in sorted(candidates, key=lambda c: (c["y"], c["x"])):
        y = c["y"]
        box = text_box(c["x"], y, c["text"], c["fs"], c["anchor"])
        shift = 0
        while any(overlaps(box, p, pad=4.0) for p in placed) and shift < 120:
            shift += 4
            y = c["y"] + shift
            box = text_box(c["x"], y, c["text"], c["fs"], c["anchor"])
            if box[3] > bottom:
                y = c["y"] - shift
                box = text_box(c["x"], y, c["text"], c["fs"], c["anchor"])
                if box[1] < top:
                    break
        placed.append(box)
        result.append({**c, "y": y, "box": box, "moved": shift > 0})
    return result


def pareto(rows: list[dict], title: str, sub: str) -> str:
    """Quality against mean tokens per task, on a log cost axis.

    The spread runs from 7k to 242k tokens, a factor of 34, and the perfect models sit
    between 17k and 24k where a linear axis squeezes four labels into 20 px. Log cost is
    the standard answer and it is honest about what "twice the tokens" means at either end.
    Offsets come from the registry; a collision pass then moves whatever still overlaps and
    draws a leader line, because hand-tuned offsets break the moment a model is added.
    """
    w, h = 900, 470
    left, right, bottom = 128, 96, 66
    subs = wrap_sub(sub, w)
    top = max(92, sub_end(subs) + 26)
    h += max(0, top - 92)
    pw, ph = w - left - right, h - top - bottom
    out = svg_open(w, h, title, subs)

    xs = [r["tokens"] for r in rows]
    lo, hi = math.log10(min(xs) * 0.85), math.log10(max(xs) * 1.08)
    y0, y1 = 0.84, 1.015

    def X(v): return left + pw * (math.log10(v) - lo) / (hi - lo)
    def Y(v): return top + ph * (1 - (v - y0) / (y1 - y0))

    obstacles: list[tuple] = []
    for frac in (0.85, 0.90, 0.95, 1.00):
        yy = Y(frac)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+pw}" y2="{yy:.1f}" '
                   f'stroke="#21262d" stroke-width="1"/>')
        out.append(f'<text x="{left-12}" y="{yy+4:.1f}" fill="#7d8590" font-size="11" '
                   f'text-anchor="end">{frac:.0%}</text>')
        obstacles.append(text_box(left - 12, yy + 4, f"{frac:.0%}", 11, "end"))

    # Decade-style ticks. A linear 5k step produced 55 gridlines once the spread reached
    # 270k, so the axis is log and the ticks are the round numbers on it.
    for v in (5000, 10000, 20000, 50000, 100000, 200000, 500000):
        if not (min(xs) * 0.85 <= v <= max(xs) * 1.08):
            continue
        xx = X(v)
        out.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{top+ph}" '
                   f'stroke="#161b22" stroke-width="1"/>')
        out.append(f'<text x="{xx:.1f}" y="{top+ph+22}" fill="#7d8590" font-size="11" '
                   f'text-anchor="middle">{v//1000}k</text>')
        obstacles.append(text_box(xx, top + ph + 22, f"{v//1000}k", 11, "middle"))

    out.append(f'<text x="{left+pw/2:.0f}" y="{h-18}" fill="#7d8590" font-size="12" '
               f'text-anchor="middle">mean tokens per task (log scale)</text>')
    yax = 30
    out.append(f'<text x="{yax}" y="{top+ph/2:.0f}" fill="#7d8590" font-size="12" '
               f'text-anchor="middle" '
               f'transform="rotate(-90 {yax} {top+ph/2:.0f})">share of runs passed</text>')

    # Offsets live with the display registry, keyed by model id rather than by label:
    # the earlier label-based lookup meant a renamed model silently lost its placement.
    OFFSET = registry.PARETO_OFFSET
    fs = 11.5
    candidates = []
    points = []
    for r in rows:
        cx, cy = X(r["tokens"]), Y(r["passed"] / r["runs"])
        points.append((cx, cy, r))
        # A label must not sit on any dot, including another model's.
        obstacles.append((cx - 9, cy - 9, cx + 9, cy + 9))
        dx, dy, anchor = OFFSET.get(r["model"], (0, -18, "middle"))
        candidates.append({"x": cx + dx, "y": cy + dy, "anchor": anchor,
                           "text": r["label"], "fs": fs, "point": (cx, cy),
                           "colour": r["colour"]})

    for cx, cy, r in points:
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{r["colour"]}" '
                   f'stroke="#0d1117" stroke-width="2" opacity="0.98"/>')

    for lab in place_labels(candidates, obstacles, top + 4, top + ph - 2):
        bx = lab["box"]
        px, py = lab["point"]
        # Nearest point on the label box to the dot. A leader line is drawn whenever the
        # label sat more than 12 px away, so a nudged label can never be mistaken for its
        # neighbour's.
        lx = min(max(px, bx[0]), bx[2])
        ly = min(max(py, bx[1]), bx[3])
        if ((lx - px) ** 2 + (ly - py) ** 2) ** 0.5 > 4:
            # Solid, muted grey and started at the dot's edge: a leader that looks like a
            # hairline in the model's own colour reads as part of the marker.
            d = max(1.0, ((lx - px) ** 2 + (ly - py) ** 2) ** 0.5)
            ux, uy = (lx - px) / d, (ly - py) / d
            out.append(f'<line x1="{px + ux*9:.1f}" y1="{py + uy*9:.1f}" '
                       f'x2="{lx:.1f}" y2="{ly:.1f}" stroke="#6e7681" stroke-width="1"/>')
        out.append(f'<text x="{lab["x"]:.1f}" y="{lab["y"]:.1f}" fill="#c9d1d9" '
                   f'font-size="{fs}" text-anchor="{lab["anchor"]}">{esc(lab["text"])}</text>')

    out.append("</svg>")
    return "\n".join(out)


def audit(outdir: Path, rows: list[dict]) -> list[str]:
    """Every suite chart must name every model in results/.

    This is the guard for the defect that let Ornith 1.5 122B publish at 36/36 and
    appear on no chart at all. A chart that quietly omits a row is worse than a chart
    that fails to build, so the check returns hard errors and main() exits non-zero,
    which is what stops publish.py from committing an incomplete figure.

    A chart may declare itself out of scope by carrying ``scope: targeted`` in its
    description, which is how assets/local-three-task.svg says it draws the targeted
    comparison rather than results/. Without that escape hatch the audit must see every
    model, so a new suite chart cannot quietly drop a row, and the negative test that
    feeds it a deliberately incomplete chart still has to fire.
    """
    problems: list[str] = []
    for svg in sorted(outdir.glob("*.svg")):
        body = svg.read_text()
        if "scope: targeted" in body:
            continue
        for r in rows:
            if esc(r["label"]) not in body:
                problems.append(f"{svg.name}: {r['label']} is missing")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="assets")
    args = ap.parse_args()

    data = load()
    if not data:
        sys.exit("no results found in results/")

    # Every model in results/ is charted. Sort is display order, then alphabetical, so
    # an unregistered model still appears instead of being filtered away.
    models = sorted(data, key=registry.sort_key)
    rows = [data[m] for m in models]

    unknown = registry.unregistered(models)
    if unknown:
        print("WARNING: no display entry in tools/model_registry.py for: " +
              ", ".join(unknown) + "\n         drawn under the raw model id in grey.",
              file=sys.stderr)

    outdir = ROOT / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Passes, best share first. Sorting on raw counts put the one-repeat model at the
    # bottom of a chart whose bars are proportions, which reads as a poor result.
    rows_p = sorted(rows, key=lambda r: (-(r["passed"] / r["runs"]),
                                         -(r["stable"] / r["tasks"]), -r["passed"]))
    (outdir / "passes.svg").write_text(grouped_chart(
        rows_p,
        f"Reliability: one to {max(r['repeats'] for r in rows)} repeats per task",
        f"Twelve tasks across {len(rows)} models. Runs per model are labelled; "
        "a one-repeat row is a screen, not a graded result."))

    # Tokens, cheapest first
    rows_t = sorted(rows, key=lambda r: r["tokens"])
    (outdir / "tokens.svg").write_text(bar_chart(
        rows_t, "tokens", "tokens",
        "Token cost per task",
        "Mean tokens per task, over runs that reported usage. A timed-out run is killed "
        "before usage is emitted, so it is unknown rather than zero. These are each "
        "endpoint's own accounting: two llama-server models on the same build report 711 "
        "and 10,710 prompt tokens a turn, so compare within an endpoint, not across providers.",
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
        "Upper left is best: reliable and cheap. The horizontal axis is each endpoint's own "
        "token accounting, not a common meter: Qwen and Groot are both llama-server on the same "
        "build and differ by fifteen times per turn, so read this as cost within a provider."))

    problems = audit(outdir, rows)
    if problems:
        for p in problems:
            print(f"CHART INCOMPLETE: {p}", file=sys.stderr)
        sys.exit(1)

    for f in sorted(outdir.glob("*.svg")):
        try:
            shown = f.relative_to(ROOT)
        except ValueError:      # --outdir outside the repo, e.g. from the test harness
            shown = f
        print(f"wrote {shown}  ({f.stat().st_size:,} bytes)")
    print(f"charts cover {len(rows)} models: {', '.join(r['label'] for r in rows)}")


if __name__ == "__main__":
    main()
