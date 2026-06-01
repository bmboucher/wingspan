"""The dashboard's visual constants — the "wetland dawn" palette and glyphs.

Every color, glyph ramp, and per-role hue the dashboard uses lives here so the
look can be retuned in one place (the house rule: public constants in a single
file). Colors are truecolor hex strings consumed directly by ``rich.style.Style``.

Small pure helpers (:func:`lerp_color`, :func:`gradient_stops`,
:func:`gradient_text`, :func:`hero_color`) also live here because they are part
of the palette's definition — how the wordmark gradient and the cinematic eval
hero-number recolor as a value changes. :func:`gradient_text` renders the
gradient wordmark and is shared by both full-screen UIs (the dashboard header
and the configurator header).
"""

from __future__ import annotations

from rich import text

from wingspan.training import runstate

# ---------------------------------------------------------------------------
# Structure

CANVAS = "#0C100E"  # reference dark base (the design assumes a dark profile)
BORDER_DEFAULT = "#3A4F5C"  # all ROUNDED panels
BORDER_HEADLINE = "#C9A24B"  # the one HEAVY/gold panel — marks THE chart
BORDER_EVAL = "#58A6FF"  # the docked eval inset

TEXT_PRIMARY = "#E8EEF2"  # values
TEXT_BRIGHT = "#F2F4F6"  # the TOTAL / headline numbers
TEXT_MUTED = "#6B7E8C"  # units, axis labels, suffixes
TEXT_DIM2 = "#9FB3C8"  # secondary numbers (timers, throughput, counts)

# ---------------------------------------------------------------------------
# Wordmark gradient (the three Wingspan habitats, left -> right)

WORDMARK_STOPS: tuple[str, str, str] = ("#5FB37A", "#C9A24B", "#3FB4A6")

# ---------------------------------------------------------------------------
# Phase accents (LED pill bg, status rule, active progress fill)

PHASE_COLOR: dict[runstate.Phase, str] = {
    runstate.Phase.STARTING: "#6B7E8C",
    runstate.Phase.COLLECTING: "#3FB4A6",  # wetland teal
    runstate.Phase.UPDATING: "#C9A24B",  # grassland gold
    runstate.Phase.EVALUATING: "#58A6FF",  # sky blue
    runstate.Phase.CHECKPOINTING: "#B08CD9",  # soft violet
    runstate.Phase.FINAL_EVALUATING: "#58A6FF",  # sky blue (same as eval)
    runstate.Phase.PAUSED_AT_TARGET: "#5BB98C",  # success green — waiting for input
    runstate.Phase.DONE: "#5BB98C",  # success green
    runstate.Phase.STOPPED: "#C9A24B",
    runstate.Phase.ERROR: "#C0564E",
}

# ---------------------------------------------------------------------------
# Six score components — one fixed hue each, reused everywhere (the stacked
# score bar and its legend tell the sources apart by color alone)

SCORE_COLOR: dict[str, str] = {
    "birds": "#6FB37A",  # forest green
    "eggs": "#E6E2C3",  # egg cream
    "food": "#D98C5F",  # warm seed/berry (cached-food pts)
    "tucked": "#7FA9C9",  # wetland blue
    "rounds": "#C9A24B",  # grassland gold (round-goal pts)
    "bonus": "#B08CD9",  # bonus violet
}

# ---------------------------------------------------------------------------
# Family histogram — frequency tiers (also a gentle data-health signal)

HIST_TOP = "#5BB98C"  # share >= 10%
HIST_TOP_BRIGHT = "#8CCBA8"  # the single busiest family
HIST_MID = "#C9A24B"  # 1% .. 10%
HIST_LOW = "#C46B6B"  # < 1% — flags data-starved heads
HIST_COUNT = "#9FB3C8"

HIST_TOP_SHARE = 0.10
HIST_LOW_SHARE = 0.01

# ---------------------------------------------------------------------------
# Convergence chart series

WIN_COLOR = "#5BB98C"  # win-rate EWMA (primary, dominant)
WIN_RAW = "#3E7A57"  # raw per-eval win-rate (dim, plotted dotted under the EWMA)
BEACON_A = "#BFF0D2"  # leading-edge beacon, frame A
BEACON_B = "#FFFFFF"  # leading-edge beacon, frame B
MARGIN_COLOR = "#3FB4A6"  # eval margin (secondary, dim teal)
POINTS_COLOR = "#D9B26A"  # average self-play points (warm amber, dominant)
TARGET_GRID = "#2F5D4A"  # faint gridline (points zero line)
WIN_THRESHOLD = "#C9A24B"  # the yellow opponent-advance win-rate threshold line
CHALLENGER_MARK = "#5A6B86"  # vertical marker where the reference opponent advanced
AXIS = "#6B7E8C"

# ---------------------------------------------------------------------------
# Training-health verdicts

GOOD = "#5BB98C"
CAUTION = "#C9A24B"
BAD = "#C46B6B"
SPARK_COLOR = "#7FA9C9"

# ---------------------------------------------------------------------------
# System telemetry band — host CPU / RAM gauges
#
# CPU utilization: a busy box is what you *want*, so the fill is a calm teal
# that only brightens once it pegs near saturation. RAM: a full pool risks an
# OOM, so it warms from blue -> gold -> clay as it climbs past the caution and
# alarm thresholds.

SYSTEM_LABEL = "#9FB3C8"  # the CPU / RAM row labels
GAUGE_BRACKET = "#6B7E8C"  # the ▕ ▏ gauge end-caps
GAUGE_TRACK = "#2A3A44"  # the unfilled remainder of a gauge
GAUGE_UTIL = "#3FB4A6"  # CPU utilization fill (busy is good)
GAUGE_UTIL_PEAK = "#8CCBA8"  # utilization at / near saturation
GAUGE_MEM = "#7FA9C9"  # RAM fill, comfortable
GAUGE_MEM_HIGH = "#C9A24B"  # RAM past the caution threshold
GAUGE_MEM_FULL = "#C46B6B"  # RAM past the alarm threshold
GAUGE_MEM_PROC = "#B08CD9"  # this process's resident slice within the RAM bar

GAUGE_UTIL_PEAK_PCT = 95.0  # brighten the utilization fill above this
GAUGE_MEM_HIGH_PCT = 80.0  # RAM caution color above this
GAUGE_MEM_FULL_PCT = 92.0  # RAM alarm color above this

# ---------------------------------------------------------------------------
# Events — glyph + color per kind

EVENT_GLYPH: dict[runstate.EventKind, str] = {
    runstate.EventKind.INFO: "·",
    runstate.EventKind.EVAL: "◎",
    runstate.EventKind.CHECKPOINT: "✓",
    runstate.EventKind.BEST: "★",
    runstate.EventKind.ALARM: "⚠",
}
EVENT_COLOR: dict[runstate.EventKind, str] = {
    runstate.EventKind.INFO: "#C8D2DA",
    runstate.EventKind.EVAL: "#5BB98C",
    runstate.EventKind.CHECKPOINT: "#5BB98C",
    runstate.EventKind.BEST: "#C9A24B",
    runstate.EventKind.ALARM: "#C0564E",
}

# ---------------------------------------------------------------------------
# Glyph ramps

SPARK_RAMP = " ▁▂▃▄▅▆▇█"  # 9-step block ramp for sparklines
BAR8_RAMP = " ▁▂▃▄▅▆▇█"  # vertical ramp (sparkline reuse)
BAR8_H_RAMP = " ▏▎▍▌▋▊▉█"  # horizontal eighth-block ramp for bars

# Hero number value -> color ramp breakpoints.
_HERO_RED = "#C0564E"
_HERO_GOLD = "#C9A24B"
_HERO_GREEN_LO = "#5FB37A"
_HERO_GREEN_HI = "#9FE6B0"


def lerp_color(start: str, end: str, t: float) -> str:
    """Interpolate between two ``#rrggbb`` colors. ``t`` clamps to ``[0, 1]``."""
    t = max(0.0, min(1.0, t))
    start_rgb = _parse_hex(start)
    end_rgb = _parse_hex(end)
    mixed = tuple(round(a + (b - a) * t) for a, b in zip(start_rgb, end_rgb))
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"


def gradient_stops(stops: tuple[str, ...], n: int) -> list[str]:
    """Sample ``n`` evenly-spaced colors along a multi-stop gradient."""
    if n <= 1:
        return [stops[0]]
    segments = len(stops) - 1
    out: list[str] = []
    for i in range(n):
        pos = i / (n - 1) * segments
        lo = min(int(pos), segments - 1)
        out.append(lerp_color(stops[lo], stops[lo + 1], pos - lo))
    return out


def gradient_text(content: str) -> text.Text:
    """Render ``content`` as a bold, left-to-right wordmark gradient (the
    :data:`WORDMARK_STOPS` habitats). Used by both full-screen UI headers."""
    colors = gradient_stops(WORDMARK_STOPS, len(content))
    out = text.Text(no_wrap=True, end="")
    for char, color in zip(content, colors):
        out.append(char, style=f"bold {color}")
    return out


def hero_color(win_pct: float) -> str:
    """The eval hero-number color: red < 50%, gold 50-80%, green ramp > 80%."""
    if win_pct < 50.0:
        return _HERO_RED
    if win_pct < 80.0:
        return _HERO_GOLD
    return lerp_color(_HERO_GREEN_LO, _HERO_GREEN_HI, (win_pct - 80.0) / 20.0)


def _parse_hex(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return (int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16))
