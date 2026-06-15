"""Self-contained HTML viewer for a single ``wingspan play`` game log.

Where :mod:`wingspan.reporting.html` documents a trained *network*, this module
renders one *game*: a navigable, phase-by-phase replay of the detailed log the
engine already produces. The page shows exactly one phase (a setup block, a
round banner, or a single player turn) at a time, with prev/next arrows; a
``P0 / P1 / both`` toggle at the top controls navigation only (all boards
always show both players); the current game state (3x5 board grids, hands,
tray, birdfeeder, scores, round goals, bonus cards) is shown at the top of the
page, and the turn's decision narration sits in a collapsible panel beneath.

The data model here holds **primitives only** — every field is a string, int,
or a small nested model of strings/ints — so this module depends on nothing in
the engine or PyTorch and can be rendered from a plain JSON dump. The
:class:`~wingspan.instrumentation.handlers.game_log_html.GameLogHtmlHandler`
walks the live ``GameState`` and fills these models in; the rendering itself is
a static HTML shell plus an inline script that draws each phase from the
embedded JSON.

Public API: :func:`render_game_log_html` (report -> HTML string) and
:func:`write_game_log_html` (report -> file).
"""

from __future__ import annotations

import pathlib
import typing

import pydantic

# Number of board columns per habitat row — the viewer always draws this many
# cells so the board reads as a fixed 3x5 grid (empty slots shown hollow).
BOARD_COLUMNS = 5


# ---------------------------------------------------------------------------
# Data model — primitives only (no engine / torch types).


class BirdCellInfo(pydantic.BaseModel):
    """One played (or tray/hand) bird, flattened to display primitives."""

    name: str
    vp: int
    nest: str
    wingspan_cm: int
    habitats: str
    food_cost: str
    food_cost_slots: list[str] = []
    egg_limit: int
    eggs: int
    tucked: int
    cached: int
    power_color: str
    power_text: str


class BoardCell(pydantic.BaseModel):
    """A single board slot: a bird, or an empty slot when ``bird is None``."""

    bird: BirdCellInfo | None = None


class HabitatRow(pydantic.BaseModel):
    """One habitat row of the board: a label and exactly ``BOARD_COLUMNS`` cells."""

    label: str
    cells: list[BoardCell]


class FoodCount(pydantic.BaseModel):
    """One food type and the player's current count of it."""

    label: str
    count: int


class ScoreBreakdown(pydantic.BaseModel):
    """The seven score columns shown in the log's score table."""

    birds: int
    eggs: int
    tucked: int
    cached: int
    bonus: int
    goals: int
    total: int


class BonusCardInfo(pydantic.BaseModel):
    """A held bonus card with its scoring text, current VP, and qualifying count."""

    name: str
    text: str
    vp_now: int
    count: int = 0
    pending: bool = False


class PlayerPanel(pydantic.BaseModel):
    """One seat's full visible state at a phase boundary."""

    player_id: int
    name: str
    action_cubes_left: int
    rows: list[HabitatRow]
    hand: list[BirdCellInfo]
    food: list[FoodCount]
    score: ScoreBreakdown
    bonus_cards: list[BonusCardInfo]


class RoundGoalInfo(pydantic.BaseModel):
    """One of the four round goals with its 2-player payout, scored flag,
    projected per-player VPs, and qualifying counts (counts are shown in the bar
    chart; VP payouts are shown as a sub-label)."""

    round_num: int
    description: str
    first_vp: int
    second_vp: int
    scored: bool
    p0_vp: int = 0
    p1_vp: int = 0
    p0_count: int = 0
    p1_count: int = 0


class DecisionOption(pydantic.BaseModel):
    """One offered option within a decision box in the decision log.

    ``prob`` is the policy's softmax probability (``None`` when unavailable);
    ``score`` is the raw logit used for ranking (``None`` for the setup-net
    value-only mode); ``selected`` marks the option that was actually played."""

    label: str
    prob: float | None = None
    score: float | None = None
    selected: bool = False


class LogItem(pydantic.BaseModel):
    """One item in the phase's decision log: a decision, a forced move, or a note.

    ``kind`` controls rendering: ``"decision"`` renders as a collapsible box
    with option bars; ``"forced"`` renders as a non-collapsible outcome box;
    ``"note"`` renders as a muted standalone notification."""

    kind: typing.Literal["decision", "forced", "note"]
    player_id: int | None
    text: str
    options: list[DecisionOption] = []
    forced: bool = False


class PhaseRecord(pydantic.BaseModel):
    """One navigable phase: its header, both seats' state, and its log items."""

    index: int
    title: str
    kind: str
    round_idx: int | None
    active_player_id: int | None
    panels: list[PlayerPanel]
    tray: list[BirdCellInfo | None]
    feeder_text: str
    feeder_slots: list[str | None] = []
    round_goals: list[RoundGoalInfo]
    log_items: list[LogItem] = []
    setup_bonus_options: list[BonusCardInfo] = []


class TimelinePoint(pydantic.BaseModel):
    """One decision's snapshot for the modal timeline chart.

    Coordinates are primitives so the instance serialises cleanly into the
    page's embedded JSON. ``value_return_p0`` and ``target_return_p0`` are the
    P0-relative discounted future return (the P0−P1 margin change still to come)
    in victory-point units:

    * ``value_return_p0`` — the critic's predicted return at this decision.
    * ``target_return_p0`` — the actual discounted return the critic is trained
      to match (equals ``terminal − current`` at γ=1).

    Both are ``None`` when the deciding seat has no trained net (random/human).
    """

    timestamp: float
    player_id: int
    score_p0: int
    score_p1: int
    phase_index: int
    value_return_p0: float | None = None
    target_return_p0: float | None = None


class GameLogReport(pydantic.BaseModel):
    """The whole game: matchup metadata plus every phase in play order."""

    seed: int | None = None
    matchup: tuple[str, str] | None = None
    player_names: list[str]
    final_scores: list[int] | None = None
    phases: list[PhaseRecord]
    timeline: list[TimelinePoint] = []


# ---------------------------------------------------------------------------
# Public API


def render_game_log_html(report: GameLogReport) -> str:
    """Render ``report`` to a single self-contained HTML document (no assets).

    The report is embedded as JSON and drawn entirely client-side by the inline
    script, so the same string is valid whether saved to disk or served."""
    title = _page_title(report)
    payload = _embed_json(report)
    return _DOCUMENT.format(
        title=title,
        css=_CSS,
        payload=payload,
        script=_SCRIPT,
    )


def write_game_log_html(report: GameLogReport, out_path: pathlib.Path) -> None:
    """Render ``report`` and write it to ``out_path`` (UTF-8, parents created)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_game_log_html(report), encoding="utf-8")


###### PRIVATE #######


def _page_title(report: GameLogReport) -> str:
    """A short ``<title>`` / header line: the matchup and seed when known."""
    if report.matchup is not None:
        matchup = f"{report.matchup[0]} vs {report.matchup[1]}"
    else:
        matchup = " vs ".join(report.player_names)
    seed = "" if report.seed is None else f" — seed {report.seed}"
    return f"Wingspan game log: {matchup}{seed}"


def _embed_json(report: GameLogReport) -> str:
    """Serialize the report for safe embedding inside a ``<script>`` element.

    ``model_dump_json`` cannot emit a literal ``</script>`` (no raw strings hold
    one), but escaping ``<`` defensively keeps the payload inert regardless of
    future field content."""
    return report.model_dump_json().replace("<", "\\u003c")


# ---------------------------------------------------------------------------
# Static document shell, CSS, and the inline rendering script.
#
# _CSS and _SCRIPT are substituted as VALUES into _DOCUMENT via .format(), not
# as format templates themselves — their { } braces are plain CSS/JS syntax and
# require no escaping.

_DOCUMENT = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
{css}
</style>
</head>
<body>
<header id="topbar">
  <div class="brand">{title}</div>
  <div class="controls">
    <button id="prev" title="Previous phase (←)">◀</button>
    <span id="counter" class="counter"></span>
    <button id="next" title="Next phase (→)">▶</button>
    <button id="prev-round" title="Previous round">◀ Round</button>
    <button id="next-round" title="Next round">Round ▶</button>
    <span class="spacer"></span>
    <button id="timeline-btn" title="Open game timeline chart">Timeline</button>
    <div class="toggle" id="view-toggle" role="group" aria-label="Seat view">
      <button data-view="p0">Just P0</button>
      <button data-view="both" class="active">Both</button>
      <button data-view="p1">Just P1</button>
    </div>
  </div>
  <div id="phase-title" class="phase-title"></div>
</header>
<div id="chart-modal" role="dialog" aria-modal="true" aria-label="Score timeline">
  <div id="chart-dialog">
    <div id="chart-header">
      <span id="chart-title">Game Timeline</span>
      <button id="chart-close" title="Close (Esc)">✕</button>
    </div>
    <div id="chart-body">
      <svg id="chart-svg-top"></svg>
      <svg id="chart-svg-bottom"></svg>
    </div>
  </div>
</div>
<main id="content-layout">
  <section id="state-panel"></section>
  <aside id="decisions-panel">
    <div class="decisions-title">Decision Log</div>
    <div id="decision-log"></div>
  </aside>
</main>
<script type="application/json" id="game-log-data">{payload}</script>
<script>
{script}
</script>
</body>
</html>
"""

_CSS = """\
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f1f5f9; color: #1e293b; font-size: 17px; line-height: 1.45;
}
#topbar {
  position: sticky; top: 0; z-index: 50; background: #0f3d2e; color: #ecfdf5;
  padding: 10px 18px; box-shadow: 0 2px 10px rgba(0,0,0,.25);
}
#topbar .brand { font-weight: 700; font-size: 15px; margin-bottom: 8px; }
.controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.controls .spacer { flex: 1 1 auto; }
.controls button {
  background: #14532d; color: #ecfdf5; border: 1px solid #166534;
  border-radius: 6px; padding: 5px 11px; font-size: 14px; cursor: pointer;
}
.controls button:hover { background: #166534; }
.controls button:disabled { opacity: .4; cursor: default; }
.counter { font-variant-numeric: tabular-nums; min-width: 120px; text-align: center; }
.toggle { display: inline-flex; border: 1px solid #166534; border-radius: 6px; overflow: hidden; }
.toggle button { border: none; border-radius: 0; }
.toggle button.active { background: #4ade80; color: #052e16; font-weight: 700; }
.phase-title { margin-top: 8px; font-size: 13px; color: #a7f3d0; font-family: monospace; }
main {
  display: flex; gap: 14px; padding: 12px 14px;
  height: calc(100vh - 130px); overflow: hidden;
}
#state-panel {
  flex: 1 1 0; min-width: 0; overflow: hidden;
  background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px;
  box-shadow: 0 2px 8px rgba(0,0,0,.08);
}
#state-scaler { width: 100%; }
#decisions-panel {
  width: 320px; flex-shrink: 0; overflow-y: auto;
  background: #0f172a; border-radius: 8px; padding: 10px 12px; color: #e2e8f0;
}
.decisions-title {
  font-size: 11px; font-weight: 700; color: #94a3b8;
  text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;
}

/* === Player boards row === */
.boards-row { display: flex; gap: 12px; }
.player-section { flex: 1 1 0; min-width: 0; border: 2px solid #e2e8f0; border-radius: 8px; padding: 4px; }
.player-section.active-player { border-color: #4ade80; }
.player-header {
  display: flex; align-items: baseline; gap: 8px; margin-bottom: 4px;
  padding-bottom: 4px; border-bottom: 2px solid #0f3d2e;
}
.player-name { font-weight: 700; font-size: 17px; flex-shrink: 0; }
.player-cubes { font-size: 12px; color: #334155; }
.player-vp { margin-left: auto; font-weight: 800; font-size: 19px; color: #0f3d2e; flex-shrink: 0; }
.player-food { font-size: 15px; margin: 3px 0 5px; letter-spacing: 1px; min-height: 22px; }
.board-row { display: flex; align-items: stretch; gap: 4px; margin-bottom: 4px; }
.hab-label {
  display: flex; align-items: center; justify-content: center; text-align: center;
  font-weight: 700; font-size: 12px; color: #fff; border-radius: 4px;
  width: 56px; flex-shrink: 0;
}
.hab-forest    { background: #166534; }
.hab-grassland { background: #ca8a04; }
.hab-wetland   { background: #0369a1; }

/* === Habitat squares shown on each bird card === */
.hab-sq { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin: 0 1px; vertical-align: middle; }
.hs-forest    { background: #166534; }
.hs-grassland { background: #ca8a04; }
.hs-wetland   { background: #0369a1; }

/* === Card cell — fixed size, identical in board / tray / hand === */
.card-cell {
  position: relative;
  width: 132px; height: 154px; flex-shrink: 0;
  border: 1px solid #cbd5e1; border-radius: 5px; overflow: hidden;
  display: flex; flex-direction: column; background: #fff;
}
.card-cell.empty { border-style: dashed; background: #f1f5f9; }
.vp-badge {
  position: absolute; top: 4px; right: 4px; width: 22px; height: 22px;
  border-radius: 50%; background: #0f3d2e; color: #ecfdf5;
  font-size: 9px; font-weight: 800; z-index: 5;
  display: flex; align-items: center; justify-content: center;
}
.card-hdr {
  padding: 3px 5px; background: #f8fafc;
  border-bottom: 1px solid #e2e8f0; flex-shrink: 0;
}
.card-name {
  font-weight: 700; font-size: 11.5px; line-height: 1.15;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  padding-right: 20px;
}
.card-meta { color: #475569; font-size: 9px; line-height: 1.3; }
.card-power {
  flex: 1; padding: 3px 5px; font-size: 9px; line-height: 1.35; overflow-y: auto; overflow-x: hidden;
}
.card-cell.pw-brown  .card-power { background: #f3e9dd; }
.card-cell.pw-white  .card-power { background: #ffffff; }
.card-cell.pw-pink   .card-power { background: #fce7f3; }
.card-cell.pw-yellow .card-power { background: #fef9c3; }
.card-eggs {
  height: 16px; padding: 0 5px; letter-spacing: 1px;
  font-size: 12px; border-top: 1px solid #e2e8f0; flex-shrink: 0;
  color: #64748b; display: flex; align-items: center; justify-content: space-between;
}
.card-status  { font-size: 9px; letter-spacing: 0; color: #475569;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; min-width: 0; }
.card-egglist { flex-shrink: 0; }

/* === Action cube spans === */
.cube {
  display: inline-block; width: 10px; height: 10px; margin: 1px;
  border-radius: 2px; vertical-align: middle;
}
.cube.filled { background: #4ade80; }
.cube.empty  { border: 1px solid #94a3b8; }

/* === Middle row: tray | birdfeeder | player hand === */
.middle-row { display: flex; gap: 8px; align-items: stretch; margin-top: 8px; }
.panel {
  border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 8px; background: #fff;
  display: flex; flex-direction: column;
}
.panel-title {
  font-size: 8.5px; font-weight: 700; color: #94a3b8; text-align: center;
  letter-spacing: 1px; text-transform: uppercase; margin-top: 5px; flex-shrink: 0;
}
.tray-panel .card-row { display: flex; gap: 4px; flex: 1; }
.feeder-panel { min-width: 80px; align-items: center; }
.feeder-slots {
  flex: 1; display: flex; flex-direction: column;
  justify-content: space-evenly; align-items: center; width: 100%;
}
.feeder-slot { font-size: 22px; padding: 2px 0; line-height: 1; }
.feeder-slot.empty { opacity: 0.2; font-size: 16px; }
.hand-panel { flex: 1; min-width: 0; }
.hand-scroll { flex: 1; display: flex; gap: 4px; overflow-x: auto; padding-bottom: 4px; }
.hand-scroll::-webkit-scrollbar { height: 5px; }
.hand-scroll::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }

/* === Bottom row: round goals | point sources | bonus cards === */
.bottom-row { display: flex; gap: 8px; margin-top: 8px; align-items: stretch; }
.goals-panel  { flex: 1 1 0; }
.scores-panel { flex: 1 1 0; }
.bonus-panel  { flex: 1 1 0; }

/* === Bar chart (goals + scores) === */
.goal-cols  { display: flex; gap: 8px; flex: 1; }
.goal-col   { flex: 1 1 0; display: flex; flex-direction: column; align-items: center; gap: 2px; }
.score-cats { display: flex; gap: 6px; flex: 1; }
.score-cat  { flex: 1 1 0; display: flex; flex-direction: column; align-items: center; gap: 2px; }
.bar-pair   { display: flex; gap: 2px; align-items: flex-end; flex: 1; min-height: 30px; }
.bar        { min-height: 2px; width: 14px; border-radius: 2px 2px 0 0; }
.bar.p0     { background: #93c5fd; }
.bar.p1     { background: #fca5a5; }
.bar-vals   { display: flex; gap: 2px; }
.bar-vals span { width: 16px; text-align: center; font-size: 9px; color: #475569; }
.bar-axis   { width: 100%; border-bottom: 1px solid #cbd5e1; }
.goal-desc  { font-size: 8.5px; text-align: center; color: #334155; line-height: 1.3; }
.goal-pay-sm { font-size: 8px; color: #94a3b8; }
.score-cat-lbl { font-size: 9px; color: #475569; text-align: center; }
.score-totals { text-align: center; font-size: 11px; margin-top: 4px; flex-shrink: 0; }

/* === Bonus cards panel === */
.bonus-player-lbl {
  font-size: 9px; font-weight: 700; padding: 4px 6px 2px;
  text-transform: uppercase; letter-spacing: 1px;
}
.bonus-card {
  padding: 4px 6px; border-bottom: 1px solid #f1f5f9; font-size: 10px; line-height: 1.3;
}
.bonus-card:last-child { border-bottom: none; }
.bonus-card.pending {
  opacity: 0.55; border: 1px dashed #94a3b8; border-radius: 4px;
  background: #1e293b; margin-bottom: 3px; padding: 4px 6px;
}
.bonus-card.pending .bonus-card-name::after {
  content: ' (not yet chosen)'; font-size: 8px; color: #94a3b8; font-weight: 400;
}
.bonus-card-name { font-weight: 700; font-size: 11px; }
.bonus-card-text { color: #475569; font-size: 9px; margin-top: 1px; }
.bonus-vp {
  display: inline-block; background: #0f3d2e; color: #ecfdf5;
  border-radius: 10px; padding: 0 5px; font-size: 9px; font-weight: 700;
}
.bonus-count { display: inline-block; color: #64748b; font-size: 9px; }

/* === Decision-log items === */
.event-empty { color: #64748b; font-style: italic; font-size: 11px; padding: 8px; }

/* Shared header style for decision + forced boxes */
.di {
  margin-bottom: 4px; border-radius: 5px; overflow: hidden;
  border: 1px solid transparent;
}
.di summary, .di.forced {
  padding: 5px 9px; font-size: 11px; font-weight: 600; line-height: 1.35;
  cursor: pointer; list-style: none; user-select: none; display: block;
}
.di summary::-webkit-details-marker { display: none; }
.di.forced { cursor: default; }
.di-tag { opacity: 0.65; margin-right: 4px; }
.di.p0 summary, .di.p0.forced { background: #1e3a5f; color: #93c5fd; border-color: #1e4a7f; }
.di.p1 summary, .di.p1.forced { background: #3f1515; color: #fca5a5; border-color: #5a1f1f; }
.di.global summary, .di.global.forced { background: #1e293b; color: #cbd5e1; border-color: #2d3f55; }

/* Option rows inside an expanded decision */
.di-body { padding: 4px 6px 6px; background: #111827; }
.di-opt {
  display: flex; align-items: center; gap: 6px;
  padding: 3px 4px; border-radius: 3px; margin-bottom: 2px;
  border: 1px solid transparent;
}
.di-opt.selected {
  border-color: #4ade80; background: rgba(74,222,128,.08);
}
.di-opt.selected::before {
  content: '▸'; color: #4ade80; font-size: 10px; flex-shrink: 0;
}
.di-opt-label {
  flex: 1; font-size: 10px; color: #94a3b8; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}
.di-opt.selected .di-opt-label { color: #e2e8f0; }
.di-opt-right { display: flex; align-items: center; gap: 5px; flex-shrink: 0; }
.di-bar {
  width: 60px; height: 6px; border-radius: 3px;
  background: #1e293b; overflow: hidden;
}
.di-bar-fill { height: 100%; border-radius: 3px; background: #475569; }
.di.p0 .di-bar-fill { background: #3b82f6; }
.di.p1 .di-bar-fill { background: #ef4444; }
.di-score {
  font-size: 9px; color: #64748b; font-family: 'Fira Code', Consolas, monospace;
  min-width: 34px; text-align: right;
}
.di-opt.selected .di-score { color: #a7f3d0; }

/* Note boxes: muted standalone notifications */
.note {
  padding: 4px 9px; margin-bottom: 3px; border-radius: 4px;
  font-size: 10px; color: #64748b; background: #0f172a;
  border-left: 2px solid #1e293b;
}
.note.p0 { border-left-color: #1e3a5f; }
.note.p1 { border-left-color: #3f1515; }

/* === Timeline modal === */
#chart-modal {
  display: none; position: fixed; inset: 0; z-index: 200;
  background: rgba(0,0,0,.6); align-items: center; justify-content: center;
}
#chart-modal.open { display: flex; }
#chart-dialog {
  background: #0f172a; border-radius: 10px; padding: 0;
  width: min(900px, 95vw); max-height: 90vh;
  display: flex; flex-direction: column; overflow: hidden;
  box-shadow: 0 8px 40px rgba(0,0,0,.6);
}
#chart-header {
  display: flex; align-items: center; padding: 10px 16px;
  border-bottom: 1px solid #1e293b; flex-shrink: 0;
}
#chart-title { color: #a7f3d0; font-weight: 700; font-size: 14px; flex: 1; }
#chart-close {
  background: none; border: none; color: #94a3b8; font-size: 18px;
  cursor: pointer; padding: 2px 6px; border-radius: 4px;
}
#chart-close:hover { background: #1e293b; color: #e2e8f0; }
#chart-body {
  padding: 12px 16px 16px; overflow-y: auto; flex: 1;
  display: flex; flex-direction: column; gap: 10px;
}
#chart-svg-top, #chart-svg-bottom {
  width: 100%; height: 200px; display: block;
  background: #111827; border-radius: 6px;
}
.chart-label { font: 10px/1 sans-serif; fill: #94a3b8; }
.chart-axis { stroke: #1e293b; stroke-width: 1; }
.chart-gridline { stroke: #1e293b; stroke-width: 0.5; stroke-dasharray: 3,3; }
.chart-line-p0 { fill: none; stroke: #93c5fd; stroke-width: 2; }
.chart-line-p1 { fill: none; stroke: #fca5a5; stroke-width: 2; }
.chart-line-realized { fill: none; stroke: #64748b; stroke-width: 1.5; stroke-dasharray: 4,2; }
.chart-line-value   { fill: none; stroke: #4ade80; stroke-width: 2; }
.chart-line-value-p1 { fill: none; stroke: #22d3ee; stroke-width: 2; }
.chart-line-target  { fill: none; stroke: #facc15; stroke-width: 1.5; stroke-dasharray: 5,3; }
.chart-line-target-p1 { fill: none; stroke: #fb923c; stroke-width: 1.5; stroke-dasharray: 5,3; }
.chart-hit { fill: transparent; cursor: pointer; }
.chart-hit:hover { fill: rgba(255,255,255,.1); }
.chart-legend { font: 10px/1.4 sans-serif; fill: #94a3b8; }
"""

_SCRIPT = r"""
'use strict';
const DATA = JSON.parse(document.getElementById('game-log-data').textContent);
const HAB_CLASS = {'Forest':'hab-forest','Grassland':'hab-grassland','Wetland':'hab-wetland'};
const FOOD_EMOJI = {
  invertebrate: '\u{1F41B}', seed: '\u{1F33E}', fish: '\u{1F41F}',
  fruit: '\u{1F352}', rodent: '\u{1F400}', wild: '\u{1F308}', choice: '\u{1F41B}/\u{1F33E}'
};
const HAB_ICON = { Forest: '\u{1F7E9}', Grassland: '\u{1F7E8}', Wetland: '\u{1F7E6}' };
let phaseIdx = 0;
let view = 'both';

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function eggGlyphs(eggs, limit) {
  if (limit <= 0) return '';
  const laid = Math.max(0, Math.min(eggs, limit));
  return '●'.repeat(laid) + '○'.repeat(limit - laid);
}

function cubeGlyphs(n, total) {
  const tot = total || 8;
  const filled = Math.max(0, Math.min(n, tot));
  let html = '';
  for (let i = 0; i < tot; i++) {
    html += '<span class="cube ' + (i < filled ? 'filled' : 'empty') + '"></span>';
  }
  return html;
}

function foodCostHtml(slots) {
  if (!slots || !slots.length) return 'free';
  return slots.map(f => FOOD_EMOJI[f] || esc(f)).join('');
}

function habIconsHtml(habitats) {
  if (!habitats) return '';
  return habitats.split('/').map(h => HAB_ICON[h] || esc(h)).join('');
}

function habSquaresHtml(habitats) {
  if (!habitats) return '';
  return habitats.split('/').map(h => {
    const cls = 'hs-' + h.toLowerCase();
    return '<span class="hab-sq ' + cls + '" title="' + esc(h) + '"></span>';
  }).join('');
}

function applyFoodEmoji(text) {
  return text
    .replace(/\binvertebrate\b/gi, '\u{1F41B}')
    .replace(/\bseed\b/gi, '\u{1F33E}')
    .replace(/\bfish\b/gi, '\u{1F41F}')
    .replace(/\bfruit\b/gi, '\u{1F352}')
    .replace(/\brodent\b/gi, '\u{1F400}');
}

function cardStatusText(bird) {
  const parts = [];
  if (bird.cached > 0) parts.push(bird.cached + ' cached');
  if (bird.tucked > 0) parts.push(bird.tucked + ' tucked');
  return parts.join(', ');
}

function cardCellHtml(bird) {
  if (!bird) return '<div class="card-cell empty"></div>';
  const pwCls = 'pw-' + esc(bird.power_color || 'none');
  const eggs = eggGlyphs(bird.eggs, bird.egg_limit);
  const cost = foodCostHtml(bird.food_cost_slots);
  const habSq = habSquaresHtml(bird.habitats);
  const status = cardStatusText(bird);
  return '<div class="card-cell ' + pwCls + '">'
    + '<div class="vp-badge">' + bird.vp + '</div>'
    + '<div class="card-hdr">'
    +   '<div class="card-name">' + esc(bird.name) + '</div>'
    +   '<div class="card-meta">' + cost + ' ' + habSq + '</div>'
    +   '<div class="card-meta card-traits">' + esc(bird.nest) + ' · ' + bird.wingspan_cm + 'cm</div>'
    + '</div>'
    + '<div class="card-power">' + esc(bird.power_text) + '</div>'
    + '<div class="card-eggs">'
    +   '<span class="card-status">' + esc(status) + '</span>'
    +   '<span class="card-egglist">' + eggs + '</span>'
    + '</div>'
    + '</div>';
}

function boardHtml(panel) {
  let html = '';
  for (const row of panel.rows) {
    const cls = HAB_CLASS[row.label] || '';
    html += '<div class="board-row"><div class="hab-label ' + cls + '">' + esc(row.label) + '</div>';
    for (let i = 0; i < 5; i++) {
      html += cardCellHtml((row.cells[i] || {bird: null}).bird);
    }
    html += '</div>';
  }
  return html;
}

function foodStripHtml(panel) {
  const parts = panel.food
    .filter(f => f.count > 0)
    .map(f => (FOOD_EMOJI[f.label] || f.label).repeat(f.count));
  if (!parts.length) return '<div class="player-food" style="color:#94a3b8;font-size:12px">Ø no food</div>';
  return '<div class="player-food">' + parts.join(' ') + '</div>';
}

function playerSectionHtml(panel, isActive, roundTotal) {
  const cubes = cubeGlyphs(panel.action_cubes_left, roundTotal);
  return '<div class="player-section' + (isActive ? ' active-player' : '') + '">'
    + '<div class="player-header">'
    +   '<span class="player-name">P' + panel.player_id + '</span>'
    +   '<span class="player-cubes">' + cubes + '</span>'
    +   '<span class="player-vp">' + panel.score.total + ' VP</span>'
    + '</div>'
    + foodStripHtml(panel)
    + boardHtml(panel)
    + '</div>';
}

function trayPanelHtml(phase) {
  const cards = phase.tray.map(b => cardCellHtml(b)).join('');
  return '<div class="panel tray-panel">'
    + '<div class="card-row">' + cards + '</div>'
    + '<div class="panel-title">Tray</div>'
    + '</div>';
}

function feederPanelHtml(phase) {
  const slots = phase.feeder_slots || [];
  let inner = '<div class="feeder-slots">';
  for (let i = 0; i < 5; i++) {
    const slot = slots[i];
    if (slot) {
      inner += '<div class="feeder-slot">' + (FOOD_EMOJI[slot] || esc(slot)) + '</div>';
    } else {
      inner += '<div class="feeder-slot empty">○</div>';
    }
  }
  inner += '</div>';
  return '<div class="panel feeder-panel">'
    + inner
    + '<div class="panel-title">Birdfeeder</div>'
    + '</div>';
}

function handPanelHtml(phase) {
  const activePanel = phase.panels.find(p => p.player_id === phase.active_player_id) || phase.panels[0];
  let inner = '';
  if (activePanel) {
    inner = activePanel.hand.map(b => cardCellHtml(b)).join('');
  }
  if (!inner) {
    inner = '<span style="color:#94a3b8;font-style:italic;font-size:11px;padding:4px;">(empty)</span>';
  }
  return '<div class="panel hand-panel">'
    + '<div class="hand-scroll">' + inner + '</div>'
    + '<div class="panel-title">Player Hand</div>'
    + '</div>';
}

function goalsPanelHtml(phase) {
  const cols = phase.round_goals.map(g => {
    const max = Math.max(g.p0_count, g.p1_count, 1);
    const h0 = Math.round(g.p0_count / max * 60);
    const h1 = Math.round(g.p1_count / max * 60);
    const check = g.scored ? ' ✓' : '';
    return '<div class="goal-col">'
      + '<div class="bar-vals"><span>' + g.p0_count + '</span><span>' + g.p1_count + '</span></div>'
      + '<div class="bar-pair">'
      +   '<div class="bar p0" style="height:' + h0 + 'px"></div>'
      +   '<div class="bar p1" style="height:' + h1 + 'px"></div>'
      + '</div>'
      + '<div class="bar-axis"></div>'
      + '<div class="goal-desc">R' + g.round_num + check + '<br>' + esc(g.description) + '</div>'
      + '<div class="goal-pay-sm">(' + g.first_vp + '/' + g.second_vp + ' VP)</div>'
      + '</div>';
  });
  return '<div class="panel goals-panel">'
    + '<div class="goal-cols">' + cols.join('') + '</div>'
    + '<div class="panel-title">Round-End Goals</div>'
    + '</div>';
}

function scoresPanelHtml(seats) {
  const cats = ['birds','eggs','tucked','cached','bonus','goals'];
  const lbls = ['Birds','Eggs','Tuck','Cache','Bonus','Goals'];
  const allVals = seats.flatMap(p => cats.map(c => p.score[c]));
  const maxVal = Math.max(...allVals, 1);
  const cols = cats.map((cat, i) => {
    const bars = seats.map(p => {
      const h = Math.round(p.score[cat] / maxVal * 60);
      return '<div class="bar p' + p.player_id + '" style="height:' + h + 'px" title="P' + p.player_id + ': ' + p.score[cat] + '"></div>';
    }).join('');
    const vals = seats.map(p => '<span>' + p.score[cat] + '</span>').join('');
    return '<div class="score-cat">'
      + '<div class="bar-vals">' + vals + '</div>'
      + '<div class="bar-pair">' + bars + '</div>'
      + '<div class="bar-axis"></div>'
      + '<div class="score-cat-lbl">' + lbls[i] + '</div>'
      + '</div>';
  });
  const totals = seats.map(p => 'P' + p.player_id + ': ' + p.score.total).join(' · ');
  return '<div class="panel scores-panel">'
    + '<div class="score-cats">' + cols.join('') + '</div>'
    + '<div class="score-totals">' + esc(totals) + ' VP</div>'
    + '<div class="panel-title">Point Sources</div>'
    + '</div>';
}

function bonusPanelHtml(phase) {
  // For setup_start phases, show the offered-but-not-yet-chosen bonus options.
  if (phase.kind === 'setup_start' && phase.setup_bonus_options && phase.setup_bonus_options.length) {
    const inner = phase.setup_bonus_options.map(bc =>
      '<div class="bonus-card pending">'
      + '<div class="bonus-card-name">' + esc(bc.name) + '</div>'
      + '<div class="bonus-card-text">' + esc(bc.text) + '</div>'
      + '</div>'
    ).join('');
    return '<div class="panel bonus-panel">'
      + inner
      + '<div class="panel-title">Bonus Options (choosing...)</div>'
      + '</div>';
  }
  const activeId = phase.active_player_id;
  const panel = phase.panels.find(p => p.player_id === activeId) || phase.panels[0];
  let inner = '';
  if (!panel) {
    inner = '<div class="bonus-card" style="color:#94a3b8;font-size:9px;font-style:italic">(no player)</div>';
  } else if (!panel.bonus_cards.length) {
    inner = '<div class="bonus-card" style="color:#94a3b8;font-size:9px;font-style:italic">(none)</div>';
  } else {
    for (const bc of panel.bonus_cards) {
      inner += '<div class="bonus-card">'
        + '<div class="bonus-card-name">' + esc(bc.name)
        +   ' <span class="bonus-vp">' + bc.vp_now + ' VP</span>'
        +   ' <span class="bonus-count">' + bc.count + ' qualifying</span>'
        + '</div>'
        + '<div class="bonus-card-text">' + esc(bc.text) + '</div>'
        + '</div>';
    }
  }
  return '<div class="panel bonus-panel">'
    + inner
    + '<div class="panel-title">Bonus Cards (Active Player)</div>'
    + '</div>';
}
function nextMatchingPhase(from, delta) {
  let idx = from + delta;
  while (idx >= 0 && idx < DATA.phases.length) {
    const phase = DATA.phases[idx];
    if (phase.kind === 'game_start') { idx += delta; continue; }
    if (phase.kind !== 'turn') return idx;
    if (view === 'both') return idx;
    if (view === 'p0' && phase.active_player_id === 0) return idx;
    if (view === 'p1' && phase.active_player_id === 1) return idx;
    idx += delta;
  }
  return from;
}

function renderState(phase) {
  const roundTotal = phase.round_idx != null ? 8 - phase.round_idx : 8;
  const boardsRow = '<div class="boards-row">'
    + phase.panels.map(p => playerSectionHtml(p, p.player_id === phase.active_player_id, roundTotal)).join('')
    + '</div>';
  const middleRow = '<div class="middle-row">'
    + trayPanelHtml(phase) + feederPanelHtml(phase) + handPanelHtml(phase)
    + '</div>';
  const bottomRow = '<div class="bottom-row">'
    + goalsPanelHtml(phase) + scoresPanelHtml(phase.panels) + bonusPanelHtml(phase)
    + '</div>';
  document.getElementById('state-panel').innerHTML =
    '<div id="state-scaler">' + boardsRow + middleRow + bottomRow + '</div>';
  fitStatePanel();
}

function fitStatePanel() {
  const panel = document.getElementById('state-panel');
  const scaler = document.getElementById('state-scaler');
  if (!panel || !scaler) return;

  // Measure at natural (un-zoomed) size first.
  scaler.style.zoom = '1';
  const cs = getComputedStyle(panel);
  const availW = panel.clientWidth  - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
  const availH = panel.clientHeight - parseFloat(cs.paddingTop)  - parseFloat(cs.paddingBottom);
  const naturalH = scaler.scrollHeight;

  // Width budget = boards row at its un-grown (max-content) width so the
  // intentionally-wide, horizontally-scrollable hand never drags the factor down.
  let naturalW = availW;
  const boards = scaler.querySelector('.boards-row');
  if (boards) {
    const prev = boards.style.width;
    boards.style.width = 'max-content';
    naturalW = boards.scrollWidth;
    boards.style.width = prev;
  }

  // Grow AND shrink to fill both dimensions, bounded by the boards width.
  const factor = Math.min(availH / naturalH, availW / naturalW);
  scaler.style.zoom = (isFinite(factor) && factor > 0) ? String(factor) : '1';
}

function renderLog(phase) {
  const log = document.getElementById('decision-log');
  const items = phase.log_items || [];
  if (!items.length) {
    log.innerHTML = '<div class="event-empty">(no decisions for this phase)</div>';
    return;
  }
  log.innerHTML = items.map(item => {
    const seat = item.player_id === 0 ? 'p0' : item.player_id === 1 ? 'p1' : 'global';
    const tag = item.player_id != null ? '<span class="di-tag">[P' + item.player_id + ']</span> ' : '';
    const headerText = tag + applyFoodEmoji(esc(item.text));

    if (item.kind === 'note') {
      return '<div class="note ' + seat + '">' + headerText + '</div>';
    }
    if (item.kind === 'forced') {
      return '<div class="di forced ' + seat + '">' + headerText + '</div>';
    }

    // decision: collapsible box with option rows
    const opts = item.options || [];
    const maxProb = Math.max(...opts.map(o => o.prob != null ? o.prob : 0), 1e-6);
    const optHtml = opts.map(o => {
      const barWidth = o.prob != null ? Math.round(o.prob / maxProb * 100) : 0;
      const rawScore = o.score != null ? o.score : null;
      const scoreText = rawScore != null ? (rawScore >= 0 ? '+' : '') + rawScore.toFixed(1) : '';
      const selCls = o.selected ? ' selected' : '';
      return '<div class="di-opt' + selCls + '">'
        + '<span class="di-opt-label">' + applyFoodEmoji(esc(o.label)) + '</span>'
        + '<span class="di-opt-right">'
        +   '<span class="di-bar"><span class="di-bar-fill" style="width:' + barWidth + '%"></span></span>'
        +   (scoreText ? '<span class="di-score">' + esc(scoreText) + '</span>' : '')
        + '</span>'
        + '</div>';
    }).join('');
    return '<details class="di ' + seat + '">'
      + '<summary>' + headerText + '</summary>'
      + '<div class="di-body">' + optHtml + '</div>'
      + '</details>';
  }).join('');
}

function hasRound(delta) {
  let idx = phaseIdx + delta;
  while (idx >= 0 && idx < DATA.phases.length) {
    if (DATA.phases[idx].kind === 'round') return true;
    idx += delta;
  }
  return false;
}

function render() {
  const phase = DATA.phases[phaseIdx];
  document.getElementById('counter').textContent = 'Phase ' + (phaseIdx + 1) + ' / ' + DATA.phases.length;
  document.getElementById('phase-title').textContent = phase.title;
  document.getElementById('prev').disabled = nextMatchingPhase(phaseIdx, -1) === phaseIdx;
  document.getElementById('next').disabled = nextMatchingPhase(phaseIdx, 1) === phaseIdx;
  document.getElementById('prev-round').disabled = !hasRound(-1);
  document.getElementById('next-round').disabled = !hasRound(1);
  renderState(phase);
  renderLog(phase);
}

function step(delta) {
  phaseIdx = nextMatchingPhase(phaseIdx, delta);
  render();
}

function stepRound(delta) {
  let idx = phaseIdx + delta;
  while (idx >= 0 && idx < DATA.phases.length) {
    if (DATA.phases[idx].kind === 'round') { phaseIdx = idx; render(); return; }
    idx += delta;
  }
}

document.getElementById('prev').addEventListener('click', () => step(-1));
document.getElementById('next').addEventListener('click', () => step(1));
document.getElementById('prev-round').addEventListener('click', () => stepRound(-1));
document.getElementById('next-round').addEventListener('click', () => stepRound(1));
document.querySelectorAll('#view-toggle button').forEach(btn => {
  btn.addEventListener('click', () => {
    view = btn.dataset.view;
    document.querySelectorAll('#view-toggle button').forEach(b => b.classList.toggle('active', b === btn));
    render();
  });
});
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') step(-1);
  else if (e.key === 'ArrowRight') step(1);
  else if (e.key === 'Escape') closeChart();
});
// Start on the first non-game_start phase so the empty pre-deal board is skipped.
if (DATA.phases.length && DATA.phases[0].kind === 'game_start') {
  phaseIdx = nextMatchingPhase(-1, 1);
}
render();

let _fitRaf = 0;
window.addEventListener('resize', () => {
  cancelAnimationFrame(_fitRaf);
  _fitRaf = requestAnimationFrame(fitStatePanel);
});

// ---- Timeline chart ----

function openChart() {
  document.getElementById('chart-modal').classList.add('open');
  renderChart();
}

function closeChart() {
  document.getElementById('chart-modal').classList.remove('open');
}

document.getElementById('timeline-btn').addEventListener('click', openChart);
document.getElementById('chart-close').addEventListener('click', closeChart);
document.getElementById('chart-modal').addEventListener('click', e => {
  if (e.target === document.getElementById('chart-modal')) closeChart();
});

function svgEl(tag, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function polyline(points, cls) {
  if (!points.length) return null;
  const el = svgEl('polyline', {
    points: points.map(([x, y]) => x + ',' + y).join(' '),
    class: cls,
  });
  return el;
}

function mkSvg(tag, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function renderPanel(svg, pts, xMin, xRange, yMin, yRange, lines, yZero, labels) {
  const W = svg.clientWidth || 840;
  const H = svg.clientHeight || 200;
  const PAD = {top: 22, right: 16, bottom: 28, left: 44};
  const cw = W - PAD.left - PAD.right;
  const ch = H - PAD.top - PAD.bottom;

  svg.innerHTML = '';
  svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);

  const toX = t => PAD.left + ((t - xMin) / xRange) * cw;
  const toY = v => PAD.top + ch - ((v - yMin) / yRange) * ch;

  // Horizontal grid lines + y-axis labels
  const yMax = yMin + yRange;
  for (let i = 0; i <= 5; i++) {
    const v = yMin + yRange * i / 5;
    const y = toY(v);
    svg.appendChild(mkSvg('line', {x1: PAD.left, x2: PAD.left + cw, y1: y, y2: y, class: 'chart-gridline'}));
    const lbl = mkSvg('text', {x: PAD.left - 4, y: y + 3, 'text-anchor': 'end', class: 'chart-label'});
    lbl.textContent = Math.round(v);
    svg.appendChild(lbl);
  }

  // Zero line (bottom panel only)
  if (yZero !== null && yZero >= yMin && yZero <= yMax) {
    svg.appendChild(mkSvg('line', {
      x1: PAD.left, x2: PAD.left + cw, y1: toY(yZero), y2: toY(yZero),
      stroke: '#475569', 'stroke-width': '1',
    }));
  }

  // Data lines — each series may contain nulls which split into segments
  for (const {data, cls} of lines) {
    let seg = [];
    const flush = () => {
      if (seg.length > 1) {
        svg.appendChild(mkSvg('polyline', {
          points: seg.map(([t, v]) => toX(t) + ',' + toY(v)).join(' '),
          class: cls,
        }));
      }
      seg = [];
    };
    for (const item of data) {
      if (item === null) flush();
      else seg.push(item);
    }
    flush();
  }

  // Transparent hit strips — one per unique timestamp, full chart height
  const seen = new Set();
  for (const pt of pts) {
    if (seen.has(pt.timestamp)) continue;
    seen.add(pt.timestamp);
    const hitX = toX(pt.timestamp) - 5;
    const hit = mkSvg('rect', {
      x: hitX, y: PAD.top, width: 10, height: ch,
      class: 'chart-hit', 'data-phase': pt.phase_index,
    });
    hit.addEventListener('click', () => {
      phaseIdx = parseInt(hit.getAttribute('data-phase'));
      render(); closeChart();
    });
    svg.appendChild(hit);
  }

  // Legend
  let lx = PAD.left;
  for (const {cls, label} of labels) {
    svg.appendChild(mkSvg('line', {
      x1: lx, x2: lx + 16, y1: PAD.top - 7, y2: PAD.top - 7, class: cls,
    }));
    const ltxt = mkSvg('text', {x: lx + 20, y: PAD.top - 4, class: 'chart-legend'});
    ltxt.textContent = label;
    svg.appendChild(ltxt);
    lx += label.length * 6.5 + 28;
  }
}

function renderChart() {
  const tl = DATA.timeline || [];
  if (!tl.length) return;

  const tMin = Math.min(...tl.map(p => p.timestamp));
  const tMax = Math.max(...tl.map(p => p.timestamp));
  const tRange = tMax - tMin || 1;

  // --- Top panel: P0 / P1 scores ---
  const topSvg = document.getElementById('chart-svg-top');
  const allScores = tl.flatMap(p => [p.score_p0, p.score_p1]);
  const sMin = Math.max(0, Math.min(...allScores) - 2);
  const sMax = Math.max(...allScores) + 2;
  const sRange = sMax - sMin || 1;

  renderPanel(topSvg, tl, tMin, tRange, sMin, sRange, [
    {data: tl.map(p => [p.timestamp, p.score_p0]), cls: 'chart-line-p0'},
    {data: tl.map(p => [p.timestamp, p.score_p1]), cls: 'chart-line-p1'},
  ], null, [
    {cls: 'chart-line-p0', label: 'P0 score (VP)'},
    {cls: 'chart-line-p1', label: 'P1 score (VP)'},
  ]);

  // --- Bottom panel: P0-relative future return (critic prediction vs target) ---
  const bottomSvg = document.getElementById('chart-svg-bottom');
  const realized = tl.map(p => [p.timestamp, p.score_p0 - p.score_p1]);
  const valueP0 = tl.filter(p => p.player_id === 0 && p.value_return_p0 !== null)
                    .map(p => [p.timestamp, p.value_return_p0]);
  const valueP1 = tl.filter(p => p.player_id === 1 && p.value_return_p0 !== null)
                    .map(p => [p.timestamp, p.value_return_p0]);
  const targetPts = tl.filter(p => p.target_return_p0 !== null)
                      .map(p => [p.timestamp, p.target_return_p0]);

  const allMargins = [
    ...realized.map(([, v]) => v),
    ...valueP0.map(([, v]) => v),
    ...valueP1.map(([, v]) => v),
    ...targetPts.map(([, v]) => v),
  ];
  const mMin = Math.min(...allMargins) - 2;
  const mMax = Math.max(...allMargins) + 2;
  const mRange = mMax - mMin || 1;

  const marginLines = [{data: realized, cls: 'chart-line-realized'}];
  if (valueP0.length) marginLines.push({data: valueP0, cls: 'chart-line-value'});
  if (valueP1.length) marginLines.push({data: valueP1, cls: 'chart-line-value-p1'});
  if (targetPts.length) marginLines.push({data: targetPts, cls: 'chart-line-target'});

  const marginLabels = [{cls: 'chart-line-realized', label: 'Realized P0−P1 (VP)'}];
  if (valueP0.length) marginLabels.push({cls: 'chart-line-value', label: 'P0 critic return'});
  if (valueP1.length) marginLabels.push({cls: 'chart-line-value-p1', label: 'P1 critic return'});
  if (targetPts.length) marginLabels.push({cls: 'chart-line-target', label: 'Target return'});

  renderPanel(bottomSvg, tl, tMin, tRange, mMin, mRange, marginLines, 0, marginLabels);
}
"""
