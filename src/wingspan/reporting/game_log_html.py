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
    """A held bonus card with its scoring text and current VP."""

    name: str
    text: str
    vp_now: int


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
    """One of the four round goals with its 2-player payout, scored flag, and
    projected per-player VPs (locked once the round is actually scored)."""

    round_num: int
    description: str
    first_vp: int
    second_vp: int
    scored: bool
    p0_vp: int = 0
    p1_vp: int = 0


class NarrationLine(pydantic.BaseModel):
    """A single decision-log line, tagged with the deciding seat (None = global)."""

    player_id: int | None
    text: str


class PhaseRecord(pydantic.BaseModel):
    """One navigable phase: its header, both seats' state, and its narration."""

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
    narration: list[NarrationLine]


class GameLogReport(pydantic.BaseModel):
    """The whole game: matchup metadata plus every phase in play order."""

    seed: int | None = None
    matchup: tuple[str, str] | None = None
    player_names: list[str]
    final_scores: list[int] | None = None
    phases: list[PhaseRecord]


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
    <span class="spacer"></span>
    <div class="toggle" id="view-toggle" role="group" aria-label="Seat view">
      <button data-view="p0">Just P0</button>
      <button data-view="both" class="active">Both</button>
      <button data-view="p1">Just P1</button>
    </div>
  </div>
  <div id="phase-title" class="phase-title"></div>
</header>
<main>
  <section id="state-panel"></section>
  <section id="decision-panel">
    <button id="log-toggle" class="log-toggle">▾ Decision log</button>
    <div id="decision-log" class="decision-log"></div>
  </section>
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
main { max-width: 1440px; margin: 0 auto; padding: 16px 18px 60px; }
#state-panel {
  background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px;
  box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-bottom: 16px;
}

/* === Player boards row === */
.boards-row { display: flex; gap: 12px; }
.player-section { flex: 1 1 0; min-width: 0; overflow-x: auto; }
.player-section.active-player {
  border: 2px solid #4ade80; border-radius: 8px; padding: 4px;
}
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
  flex: 1; padding: 3px 5px; font-size: 9px; line-height: 1.35; overflow: hidden;
}
.card-cell.pw-brown  .card-power { background: #f3e9dd; }
.card-cell.pw-white  .card-power { background: #ffffff; }
.card-cell.pw-pink   .card-power { background: #fce7f3; }
.card-cell.pw-yellow .card-power { background: #fef9c3; }
.card-eggs {
  height: 16px; padding: 0 5px; text-align: right; letter-spacing: 1px;
  font-size: 12px; border-top: 1px solid #e2e8f0; flex-shrink: 0;
  color: #64748b; display: flex; align-items: center; justify-content: flex-end;
}

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
.bar-pair   { display: flex; gap: 2px; align-items: flex-end; height: 60px; }
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
.bonus-card-name { font-weight: 700; font-size: 11px; }
.bonus-card-text { color: #475569; font-size: 9px; margin-top: 1px; }
.bonus-vp {
  display: inline-block; background: #0f3d2e; color: #ecfdf5;
  border-radius: 10px; padding: 0 5px; font-size: 9px; font-weight: 700;
}

/* === Decision log === */
.log-toggle {
  background: #e2e8f0; border: 1px solid #cbd5e1; border-radius: 6px;
  padding: 6px 12px; font-size: 13px; cursor: pointer; font-weight: 600;
  width: 100%; text-align: left;
}
.decision-log {
  background: #0f172a; color: #e2e8f0;
  font-family: 'Fira Code', Consolas, monospace;
  font-size: 12px; padding: 12px 14px; border-radius: 0 0 8px 8px;
  white-space: pre-wrap; overflow-x: auto;
}
.decision-log.hidden { display: none; }
.decision-log .ln { display: block; }
.decision-log .ln.p0 { color: #93c5fd; }
.decision-log .ln.p1 { color: #fca5a5; }
.decision-log .ln.global { color: #cbd5e1; }
.decision-log .empty { color: #64748b; font-style: italic; }
"""

_SCRIPT = r"""
'use strict';
const DATA = JSON.parse(document.getElementById('game-log-data').textContent);
const HAB_CLASS = {'Forest':'hab-forest','Grassland':'hab-grassland','Wetland':'hab-wetland'};
const FOOD_EMOJI = {
  invertebrate: '\u{1FAB1}', seed: '\u{1F33E}', fish: '\u{1F41F}',
  fruit: '\u{1F347}', rodent: '\u{1F400}', wild: '\u{1F308}', choice: '\u{1F308}'
};
const HAB_ICON = { Forest: '\u{1F7E9}', Grassland: '\u{1F7E8}', Wetland: '\u{1F7E6}' };
let phaseIdx = 0;
let view = 'both';
let logOpen = true;

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

function applyFoodEmoji(text) {
  return text
    .replace(/\binvertebrate\b/gi, '\u{1FAB1}')
    .replace(/\bseed\b/gi, '\u{1F33E}')
    .replace(/\bfish\b/gi, '\u{1F41F}')
    .replace(/\bfruit\b/gi, '\u{1F347}')
    .replace(/\brodent\b/gi, '\u{1F400}');
}

function cardCellHtml(bird) {
  if (!bird) return '<div class="card-cell empty"></div>';
  const pwCls = 'pw-' + esc(bird.power_color || 'none');
  const eggs = eggGlyphs(bird.eggs, bird.egg_limit);
  const cost = foodCostHtml(bird.food_cost_slots);
  const habs = habIconsHtml(bird.habitats);
  return '<div class="card-cell ' + pwCls + '">'
    + '<div class="vp-badge">' + bird.vp + '</div>'
    + '<div class="card-hdr">'
    +   '<div class="card-name">' + esc(bird.name) + '</div>'
    +   '<div class="card-meta">' + cost + ' · ' + esc(bird.nest) + '</div>'
    +   '<div class="card-meta">' + habs + '</div>'
    + '</div>'
    + '<div class="card-power">' + esc(bird.power_text) + '</div>'
    + '<div class="card-eggs">' + eggs + '</div>'
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
    const max = Math.max(g.p0_vp, g.p1_vp, 1);
    const h0 = Math.round(g.p0_vp / max * 60);
    const h1 = Math.round(g.p1_vp / max * 60);
    const check = g.scored ? ' ✓' : '';
    return '<div class="goal-col">'
      + '<div class="bar-vals"><span>' + g.p0_vp + '</span><span>' + g.p1_vp + '</span></div>'
      + '<div class="bar-pair">'
      +   '<div class="bar p0" style="height:' + h0 + 'px"></div>'
      +   '<div class="bar p1" style="height:' + h1 + 'px"></div>'
      + '</div>'
      + '<div class="bar-axis"></div>'
      + '<div class="goal-desc">R' + g.round_num + check + '<br>' + esc(g.description) + '</div>'
      + '<div class="goal-pay-sm">(' + g.first_vp + '/' + g.second_vp + ')</div>'
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

function bonusPanelHtml(seats) {
  let inner = '';
  for (const panel of seats) {
    const lbl = panel.player_id === 0 ? '#93c5fd' : '#fca5a5';
    inner += '<div class="bonus-player-lbl" style="color:' + lbl + '">P' + panel.player_id + ': ' + esc(panel.name) + '</div>';
    if (!panel.bonus_cards.length) {
      inner += '<div class="bonus-card" style="color:#94a3b8;font-size:9px;font-style:italic">(none)</div>';
      continue;
    }
    for (const bc of panel.bonus_cards) {
      const shortText = bc.text.length > 55 ? bc.text.slice(0, 52) + '...' : bc.text;
      inner += '<div class="bonus-card">'
        + '<div class="bonus-card-name">' + esc(bc.name) + ' <span class="bonus-vp">' + bc.vp_now + ' VP</span></div>'
        + '<div class="bonus-card-text">' + esc(shortText) + '</div>'
        + '</div>';
    }
  }
  return '<div class="panel bonus-panel">'
    + inner
    + '<div class="panel-title">Bonus Cards</div>'
    + '</div>';
}

function nextMatchingPhase(from, delta) {
  let idx = from + delta;
  while (idx >= 0 && idx < DATA.phases.length) {
    const phase = DATA.phases[idx];
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
    + goalsPanelHtml(phase) + scoresPanelHtml(phase.panels) + bonusPanelHtml(phase.panels)
    + '</div>';
  document.getElementById('state-panel').innerHTML = boardsRow + middleRow + bottomRow;
}

function renderLog(phase) {
  const log = document.getElementById('decision-log');
  if (!phase.narration.length) {
    log.innerHTML = '<span class="empty">(no decisions for this phase)</span>';
  } else {
    log.innerHTML = phase.narration.map(l => {
      const who = l.player_id === 0 ? 'p0' : l.player_id === 1 ? 'p1' : 'global';
      return '<span class="ln ' + who + '">' + applyFoodEmoji(esc(l.text || ' ')) + '</span>';
    }).join('');
  }
  log.classList.toggle('hidden', !logOpen);
  document.getElementById('log-toggle').textContent = (logOpen ? '▾' : '▸') + ' Decision log';
}

function render() {
  const phase = DATA.phases[phaseIdx];
  document.getElementById('counter').textContent = 'Phase ' + (phaseIdx + 1) + ' / ' + DATA.phases.length;
  document.getElementById('phase-title').textContent = phase.title;
  document.getElementById('prev').disabled = nextMatchingPhase(phaseIdx, -1) === phaseIdx;
  document.getElementById('next').disabled = nextMatchingPhase(phaseIdx, 1) === phaseIdx;
  renderState(phase);
  renderLog(phase);
}

function step(delta) {
  phaseIdx = nextMatchingPhase(phaseIdx, delta);
  render();
}

document.getElementById('prev').addEventListener('click', () => step(-1));
document.getElementById('next').addEventListener('click', () => step(1));
document.getElementById('log-toggle').addEventListener('click', () => { logOpen = !logOpen; render(); });
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
});
render();
"""
