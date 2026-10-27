#!/usr/bin/env python3
"""Display registry for the published benchmark results.

One place, because there were two. `publish.py` and `make_charts.py` each carried
their own copy of the display names, and the copies disagreed: the table fell back to
the raw model id and still printed the row, while the chart builder filtered to its
own list and dropped anything it did not recognise. Ornith 1.5 122B published as
36/36 and never appeared in a single chart as a result.

Rules for adding a model:

1. Add it here with a label and a colour.
2. Position in ORDER is the table order; unlisted models sort last, alphabetically,
   so an unregistered model still renders under its own name in grey instead of
   vanishing. `tests/test_publish.py` fails until the entry exists, which is the
   prompt to give it a proper name rather than the raw provider id.
"""

from __future__ import annotations

# model id -> human label. The id is exactly what hbench recorded.
LABEL = {
    "deepseek-v4.1-flash:1017": "DeepSeek V4.1 Flash",
    "glm-5.3-flash:cloud": "GLM-5.3 Flash",
    "minimax-m3.1-flash:cloud": "MiniMax M3.1 Flash",
    "ornith/ornith-1.5-122b:free": "Ornith 1.5 122B (free)",
    "poolside/laguna-s-2.1:free": "Laguna S 2.1 (free)",
    "poolside/laguna-xs-2.1:free": "Laguna XS 2.1 (free)",
    "beam:cloud": "Beam",
    "gemma4:cloud": "Gemma 4 (cloud)",
    "Qwen3.6-35B-A3B-UD-Q4_K_M": "Qwen3.6 35B (local)",
    "nemotron-3.5-lightning-30b-a3b": "Nemotron 3.5 Lightning 30B (local)",
    "poolside/laguna-s-2.1": "Laguna S 2.1",
    "poolside/laguna-xs-2.1": "Laguna XS 2.1",
}

# Distinct hues, dark-background legible. The local model is red on purpose: it is
# the row a reader compares against everything else on the cost chart.
COLOUR = {
    "deepseek-v4.1-flash:1017": "#4fd1a5",
    "glm-5.3-flash:cloud": "#7aa2f7",
    "minimax-m3.1-flash:cloud": "#bb9af7",
    "ornith/ornith-1.5-122b:free": "#7dcfff",
    "poolside/laguna-s-2.1:free": "#2ac3de",
    "poolside/laguna-xs-2.1:free": "#c0caf5",
    "beam:cloud": "#e0af68",
    "gemma4:cloud": "#9ece6a",
    "Qwen3.6-35B-A3B-UD-Q4_K_M": "#f7768e",
    "nemotron-3.5-lightning-30b-a3b": "#ff9e64",
    "poolside/laguna-s-2.1": "#2ac3de",
    "poolside/laguna-xs-2.1": "#c0caf5",
}
FALLBACK_COLOUR = "#888888"

# Table and legend order: perfect-scoring models first, then the rest.
ORDER = [
    "deepseek-v4.1-flash:1017",
    "glm-5.3-flash:cloud",
    "minimax-m3.1-flash:cloud",
    "ornith/ornith-1.5-122b:free",
    "poolside/laguna-s-2.1:free",
    "poolside/laguna-xs-2.1:free",
    "beam:cloud",
    "gemma4:cloud",
    "Qwen3.6-35B-A3B-UD-Q4_K_M",
    "nemotron-3.5-lightning-30b-a3b",
]

# Pareto label placement: (dx, dy, text-anchor) per model id. Hand-tuned because the
# perfect-scoring models sit within a few thousand tokens of each other on the x axis
# and three of them share y = 100%.
PARETO_OFFSET = {
    "deepseek-v4.1-flash:1017": (0, -20, "end"),
    "glm-5.3-flash:cloud": (-14, 26, "end"),
    "minimax-m3.1-flash:cloud": (14, 26, "end"),
    "ornith/ornith-1.5-122b:free": (0, -20, "middle"),
    "beam:cloud": (0, -20, "middle"),
    "Qwen3.6-35B-A3B-UD-Q4_K_M": (0, 28, "middle"),
    "nemotron-3.5-lightning-30b-a3b": (0, 28, "middle"),
}


def label(model: str) -> str:
    """Human name for a model id, falling back to the id itself."""
    return LABEL.get(model, model)


def colour(model: str) -> str:
    return COLOUR.get(model, FALLBACK_COLOUR)


def sort_key(model: str) -> tuple[int, str]:
    """Registry order first, then anything unregistered alphabetically."""
    return (ORDER.index(model) if model in ORDER else len(ORDER), label(model))


def unregistered(models) -> list[str]:
    """Model ids present in the data with no display name yet."""
    return sorted(m for m in models if m not in LABEL)
