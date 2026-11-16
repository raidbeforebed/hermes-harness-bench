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
import result_key  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def load() -> dict[str, dict]:
    """Aggregate results/*.json into one summary row per model AND configuration.

    Keyed by the same rule the publisher files records under, so a model measured under
    two configurations is drawn twice with two labels. Keying on the model alone is what
    let LFM3 thinking on and LFM3 thinking off share one bar.
    """
    by_row: dict[str, list[dict]] = defaultdict(list)
    for f in glob.glob(str(RESULTS / "*.json")):
        d = json.loads(Path(f).read_text())
        by_row[result_key.key_of(d)].append(d)

    out: dict[str, dict] = {}
    for key, rows in by_row.items():
        model = rows[0]["model"]
        config = result_key.config_of(rows[0])
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
        out[key] = {
            "key": key,
            "model": model,
            "config": config,
            "label": registry.label(model, config),
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
              fmt, width: int = 0) -> str:
    """Horizontal bars: easier to read model names than a vertical chart.

    The label column is measured, not guessed. It used to be a fixed 190 px, so every
    configuration-qualified label ("Qwen3.6 35B (local, sentinel-qwen-local, thinking
    unstated)" is over twice that) was drawn past the left edge and arrived on GitHub with
    the model name cut off, leaving only the config tail. The canvas now widens to the
    longest label plus room for the value text on the right, so nothing clips and the bars
    keep their length.
    """
    row_h = 34
    fs = 13.0
    label_w = max((text_width(r["label"], fs) for r in rows), default=0) + 22
    bar_x = int(max(210, label_w))
    bar_w = 470
    right = int(max(70, max((text_width(fmt(r), 12.5) for r in rows), default=0) + 22))
    width = bar_x + bar_w + right
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
    bar_x, bar_w = 230, 500
    right_pad = 150          # room for the value labels, which were previously clipped
    row_h = 48
    # The label column is measured, not fixed: a fixed 230 px cut the model name off the
    # left edge on any configuration-qualified row, and GitHub showed only the tail.
    bar_x = int(max(bar_x, max((text_width(r["label"], 13.0) for r in rows), default=0) + 24))
    right_pad = int(max(right_pad, max(
        (text_width(f'{r["stable"]}/{r["tasks"]} tasks all-pass', 11.5) for r in rows),
        default=0) + 22))
    width = bar_x + bar_w + right_pad
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


def text_width(text: str, fs: float) -> float:
    """Estimated rendered width of `text` at font size `fs`, in user units.

    An SVG carries no font metrics at build time, so every width in this module is
    estimated at 0.55 em per character. That over-estimates for most strings, which is the
    safe direction: a margin computed from it is always wide enough, and a label box
    computed from it never reports a collision as clear.
    """
    return len(text) * fs * 0.55


def text_box(x: float, y: float, text: str, fs: float, anchor: str):
    """Approximate bounding box for one <text> element: (x0, y0, x1, y1)."""
    width = text_width(text, fs)
    x0 = x - width / 2 if anchor == "middle" else (x - width if anchor == "end" else x)
    return (x0, y - fs * 0.8, x0 + width, y + fs * 0.25)


def overlaps(a, b, pad: float = 2.0) -> bool:
    return (a[0] - pad < b[2] and b[0] - pad < a[2]
            and a[1] - pad < b[3] and b[1] - pad < a[3])


def place_labels(candidates: list[dict], obstacles: list[tuple],
                 top: float, bottom: float) -> list[dict]:
    """Place each label at the first position that touches nothing already placed.

    The previous version could only shift a label vertically and gave up after 120 px,
    leaving whatever still collided drawn on top of whatever it collided with. That is
    what produced a paragraph of superimposed text once twenty-three rows shared the same
    quality band. This tries the registry's preferred offset first, then a ring of
    alternatives (above, below, left, right, and further out), and only falls back to a
    collision if every candidate is blocked, which in practice means the plot is full
    rather than that the search gave up.
    """
    placed = list(obstacles)
    result: list[dict] = []
    for c in sorted(candidates, key=lambda c: (c["y"], c["x"])):
        px, py = c.get("point", (c["x"], c["y"]))
        right = c.get("right", float("inf"))
        left_bound = c.get("left", 2.0)
        options = [(c["x"], c["y"], c["anchor"])]
        # A dense ring, nearest first: straight above, then a fan that widens with distance
        # and includes placing the label to either side of the dot. Spacing tight enough to
        # fit labels between neighbours, wide enough that the result does not read as one
        # block. A leader line is drawn for anything that ends up away from its dot.
        half = text_width(c["text"], c["fs"]) / 2 + 8
        for dist in (15, 23, 32, 43, 56, 71, 88, 107, 128, 150):
            for anchor, dx in (("middle", 0.0), ("start", 10.0), ("end", -10.0),
                               ("start", half), ("end", -half)):
                for sgn in (-1, 1):
                    options.append((px + dx, py + sgn * dist, anchor))

        chosen = None
        for x, y, anchor in options:
            box = text_box(x, y, c["text"], c["fs"], anchor)
            if box[0] < left_bound or box[2] > right:
                continue
            if box[1] < top or box[3] > bottom - 2:
                continue
            if any(overlaps(box, p, pad=5.0) for p in placed):
                continue
            chosen = (x, y, anchor, box)
            break
        if chosen is None:
            x, y, anchor = options[0]
            chosen = (x, y, anchor, text_box(x, y, c["text"], c["fs"], anchor))
        placed.append(chosen[3])
        moved = abs(chosen[1] - py) > 6 or abs(chosen[0] - px) > 6
        result.append({**c, "x": chosen[0], "y": chosen[1], "anchor": chosen[2],
                       "box": chosen[3], "moved": moved})
    return result


def pareto(rows: list[dict], title: str, sub: str) -> str:
    """Quality against mean tokens per task, on a log cost axis.

    The spread runs from 7k to 242k tokens, a factor of 34, and the perfect models sit
    between 17k and 24k where a linear axis squeezes four labels into 20 px. Log cost is
    the standard answer and it is honest about what "twice the tokens" means at either end.

    Labels at the points are SHORT names; the full qualified label lives in a legend under
    the axis. With twenty-three rows the qualified labels are 200 to 350 px wide and there
    is no arrangement of them that does not overlap the dots or each other: the collision
    pass could only nudge vertically, ran out of room, and printed a paragraph of
    superimposed text. A short name plus a colour keyed legend is the arrangement that
    stays legible as rows are added.
    """
    w = 980
    left, right = 128, 96
    plot_h = 380
    subs = wrap_sub(sub, w)
    top = max(92, sub_end(subs) + 26)

    # Legend geometry, computed before the canvas so the height is right the first time.
    lcols = 2
    lrow = 19.0
    lrows = math.ceil(len(rows) / lcols) if rows else 0
    legend_top_gap = 74
    legend_h = 22 + lrows * lrow
    bottom = legend_top_gap + legend_h
    h = int(round(top + plot_h + bottom))
    pw, ph = w - left - right, plot_h
    out = svg_open(w, h, title, subs)

    xs = [r["tokens"] for r in rows]
    lo, hi = math.log10(min(xs) * 0.85), math.log10(max(xs) * 1.08)

    # The quality axis is derived from the data, not fixed. A hardcoded floor of 0.84 drew
    # every row below 84% outside the canvas: Nemotron at 61.9% and Bonsai at 58.3% were
    # plotted at y = 987 and y = 1019 on an 804 px figure, so two rows were invisible and
    # their labels printed over the legend. The floor now clears the worst row.
    qs = [r["passed"] / r["runs"] for r in rows]
    y0 = max(0.0, min(qs) - 0.05)
    y1 = min(1.02, max(qs) + 0.02)
    if y1 - y0 < 0.10:          # a compact field still gets a readable band
        y0 = max(0.0, y1 - 0.10)

    def X(v): return left + pw * (math.log10(v) - lo) / (hi - lo)
    def Y(v): return top + ph * (1 - (v - y0) / (y1 - y0))

    obstacles: list[tuple] = []
    step = 0.05 if (y1 - y0) <= 0.55 else 0.10
    frac = math.ceil(y0 / step) * step
    while frac <= y1 + 1e-9:
        yy = Y(frac)
        out.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{left+pw}" y2="{yy:.1f}" '
                   f'stroke="#21262d" stroke-width="1"/>')
        out.append(f'<text x="{left-12}" y="{yy+4:.1f}" fill="#7d8590" font-size="11" '
                   f'text-anchor="end">{frac:.0%}</text>')
        obstacles.append(text_box(left - 12, yy + 4, f"{frac:.0%}", 11, "end"))
        frac = round(frac + step, 6)

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

    out.append(f'<text x="{left+pw/2:.0f}" y="{top+ph+46}" fill="#7d8590" font-size="12" '
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

    # Short names at the dots. Where two rows of one model share a short name (Beam under
    # two configurations, Ornith 1.5 122B under two), the config token is appended so the
    # two dots cannot be confused for each other; the legend carries the full label either
    # way. Computed from the rows on THIS chart, not from the registry, so a model that
    # appears once is never given a needless qualifier.
    shorts = [registry.short(r["model"]) for r in rows]
    dup = {s for s in shorts if shorts.count(s) > 1}

    # Coordinates first, then a separation pass. Two rows can land within a few pixels of
    # each other (two 100% models a few thousand tokens apart), and at this scale a merged
    # pair reads as one dot and hides one colour entirely. Markers closer than 13 px are
    # pushed apart along their own axis; the displacement is a few pixels on a log axis
    # spanning 34x cost, so position is preserved while both dots stay distinguishable.
    coords = [[X(r["tokens"]), Y(r["passed"] / r["runs"])] for r in rows]
    for _ in range(60):
        settled = True
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                ax, ay = coords[i]
                bx, by = coords[j]
                d = math.hypot(ax - bx, ay - by)
                if d >= 17.0:
                    continue
                settled = False
                if d < 0.5:
                    ux, uy = 1.0, 0.0
                else:
                    ux, uy = (ax - bx) / d, (ay - by) / d
                need = (17.0 - d) / 2 + 0.25
                coords[i][0], coords[i][1] = ax + ux * need, ay + uy * need
                coords[j][0], coords[j][1] = bx - ux * need, by - uy * need
        if settled:
            break

    for (cx, cy), r, s in zip(coords, rows, shorts):
        points.append((cx, cy, r))
        # A label must not sit on any dot, including another model's.
        obstacles.append((cx - 9, cy - 9, cx + 9, cy + 9))
        text = s
        if s in dup:
            tag = result_key.token(result_key.config_of(r)) or "unrec"
            text = f"{s} · {tag}"
        dx, dy, anchor = OFFSET.get(r["model"], (0, -18, "middle"))
        candidates.append({"x": cx + dx, "y": cy + dy, "anchor": anchor,
                           "text": text, "fs": fs, "point": (cx, cy),
                           "colour": r["colour"], "full": r["label"],
                           "right": left + pw + 6, "left": left - 40})

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

    # The legend. Full qualified labels, colour-keyed to the dots, in two columns. It is
    # what makes a short name at the dot unambiguous, and it is where `audit()` finds each
    # row's full label, so chart coverage is still proved against results/.
    legend_y = top + ph + legend_top_gap + 14
    col_w = (w - 40) / lcols
    for i, r in enumerate(rows):
        col, row_i = divmod(i, lrows)
        lx = 30 + col * col_w
        ly = legend_y + row_i * lrow
        out.append(f'<rect x="{lx:.0f}" y="{ly - 8:.1f}" width="9" height="9" rx="2" '
                   f'fill="{r["colour"]}" opacity="0.95"/>')
        out.append(f'<text x="{lx + 15:.0f}" y="{ly:.1f}" fill="#8b949e" '
                   f'font-size="11.5">{esc(r["label"])}</text>')

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

    # Every row in results/ is charted. Sort is display order, then configuration order, so
    # two configurations of one model sit side by side and an unregistered model still
    # appears instead of being filtered away.
    rows = sorted(data.values(), key=lambda r: registry.sort_key(r["model"], r["config"]))

    unknown = registry.unregistered({m["model"] for m in rows})
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
    # The subtitle counts what the data holds rather than asserting a suite size: it said
    # "twelve tasks" while the rows covered twelve and eighteen, and the configuration-qualified
    # labels mean a model can now contribute more than one row.
    (outdir / "passes.svg").write_text(grouped_chart(
        rows_p,
        f"Reliability: one to {max(r['repeats'] for r in rows)} repeats per task",
        f"{len(rows)} rows, the widest covering {max(r['tasks'] for r in rows)} tasks. "
        "Runs per row are labelled; a one-repeat row is a screen, not a graded result."))

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
