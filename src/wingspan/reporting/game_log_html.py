"""Self-contained HTML viewer for a single ``wingspan play`` game log.

Where :mod:`wingspan.reporting.html` documents a trained *network*, this module
renders one *game*: a navigable, phase-by-phase replay of the detailed log the
engine already produces. The page shows exactly one phase (a setup block, a
round banner, or a single player turn) at a time, with prev/next arrows; a
``P0 / P1 / both`` toggle at the top controls which seat's board is pinned and
which decision lines are shown; the current game state (3x5 board grids, hands,
tray, food, scores, bonus cards, round goals) is pinned at the top of the
window, and the turn's decision narration sits in a collapsible panel beneath.

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
    """One played (or tray) bird, flattened to display primitives."""

    name: str
    vp: int
    nest: str
    habitats: str
    food_cost: str
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
    hand_names: list[str]
    food: list[FoodCount]
    score: ScoreBreakdown
    bonus_cards: list[BonusCardInfo]


class RoundGoalInfo(pydantic.BaseModel):
    """One of the four round goals with its 2-player payout and scored flag."""

    round_num: int
    description: str
    first_vp: int
    second_vp: int
    scored: bool


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
  background: #f1f5f9; color: #1e293b; font-size: 14px; line-height: 1.45;
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
main { max-width: 1280px; margin: 0 auto; padding: 16px 18px 60px; }
#state-panel {
  position: sticky; top: 96px; z-index: 30; background: #f8fafc;
  border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px;
  box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-bottom: 16px;
}
.seats { display: flex; gap: 16px; flex-wrap: wrap; }
.seat { flex: 1 1 420px; min-width: 360px; }
.seat h2 {
  font-size: 14px; margin-bottom: 6px; display: flex; gap: 8px;
  align-items: baseline; border-bottom: 2px solid #0f3d2e; padding-bottom: 3px;
}
.seat h2 .total { margin-left: auto; font-size: 16px; font-weight: 800; color: #0f3d2e; }
.seat h2 .cubes { font-size: 11px; color: #64748b; font-weight: 400; }
.board { display: grid; grid-template-columns: 74px repeat(5, 1fr); gap: 4px; margin: 6px 0; }
.hab-label {
  display: flex; align-items: center; font-weight: 700; font-size: 11px;
  padding: 2px 4px; border-radius: 4px; color: #fff;
}
.hab-forest { background: #166534; }
.hab-grassland { background: #ca8a04; }
.hab-wetland { background: #0369a1; }
.cell {
  min-height: 52px; border: 1px solid #cbd5e1; border-radius: 5px;
  padding: 3px 4px; font-size: 10.5px; background: #fff; overflow: hidden;
}
.cell.empty { border-style: dashed; background: #f1f5f9; }
.cell .cn { font-weight: 700; font-size: 11px; line-height: 1.1; display: block; }
.cell .cs { color: #475569; display: block; }
.cell .ce { letter-spacing: 1px; }
.cell.pw-brown { background: #f3e9dd; }
.cell.pw-white { background: #ffffff; }
.cell.pw-pink  { background: #fce7f3; }
.cell.pw-yellow { background: #fef9c3; }
.cell .extras { color: #b45309; font-size: 9.5px; }
.subline { font-size: 11px; color: #475569; margin: 3px 0; }
.subline b { color: #1e293b; }
.scoretable { border-collapse: collapse; font-size: 11px; margin-top: 4px; width: 100%; }
.scoretable th, .scoretable td { border: 1px solid #e2e8f0; padding: 1px 5px; text-align: right; }
.scoretable th { background: #e2e8f0; }
.shared { margin-top: 10px; border-top: 1px dashed #cbd5e1; padding-top: 8px; font-size: 11.5px; }
.shared .tray-cards { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 3px; }
.tray-card { border: 1px solid #cbd5e1; border-radius: 4px; padding: 2px 6px; background: #fff; }
.goals { margin-top: 6px; }
.goals .goal { font-size: 11px; }
.goals .goal.scored { color: #15803d; }
.log-toggle {
  background: #e2e8f0; border: 1px solid #cbd5e1; border-radius: 6px;
  padding: 6px 12px; font-size: 13px; cursor: pointer; font-weight: 600; width: 100%;
  text-align: left;
}
.decision-log {
  background: #0f172a; color: #e2e8f0; font-family: 'Fira Code', Consolas, monospace;
  font-size: 12px; padding: 12px 14px; border-radius: 0 0 8px 8px; white-space: pre-wrap;
  overflow-x: auto;
}
.decision-log.hidden { display: none; }
.decision-log .ln { display: block; }
.decision-log .ln.p0 { color: #93c5fd; }
.decision-log .ln.p1 { color: #fca5a5; }
.decision-log .ln.global { color: #cbd5e1; }
.decision-log .empty { color: #64748b; font-style: italic; }
"""

_SCRIPT = """\
'use strict';
const DATA = JSON.parse(document.getElementById('game-log-data').textContent);
const HAB_CLASS = { 'Forest': 'hab-forest', 'Grassland': 'hab-grassland', 'Wetland': 'hab-wetland' };
let phaseIdx = 0;
let view = 'both';
let logOpen = true;

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function eggGlyphs(eggs, limit) {
  if (limit <= 0) return '';
  const laid = Math.max(0, Math.min(eggs, limit));
  return '\\u25cf'.repeat(laid) + '\\u25cb'.repeat(limit - laid);
}

function cellHtml(cell) {
  const bird = cell.bird;
  if (!bird) return '<div class="cell empty"></div>';
  const extras = [];
  if (bird.tucked) extras.push('tuck ' + bird.tucked);
  if (bird.cached) extras.push('cache ' + bird.cached);
  const extraHtml = extras.length ? '<span class="extras">[' + esc(extras.join(', ')) + ']</span>' : '';
  const eggs = bird.egg_limit ? '<span class="ce">' + eggGlyphs(bird.eggs, bird.egg_limit) + '</span>' : '';
  return '<div class="cell pw-' + esc(bird.power_color) + '" title="' + esc(bird.power_text) + '">'
    + '<span class="cn">' + esc(bird.name) + '</span>'
    + '<span class="cs">' + bird.vp + 'VP ' + esc(bird.food_cost) + ' ' + esc(bird.nest) + '</span>'
    + '<span class="cs">' + eggs + ' ' + extraHtml + '</span>'
    + '</div>';
}

function boardHtml(panel) {
  let html = '<div class="board">';
  for (const row of panel.rows) {
    const cls = HAB_CLASS[row.label] || '';
    html += '<div class="hab-label ' + cls + '">' + esc(row.label) + '</div>';
    for (let i = 0; i < 5; i++) {
      html += cellHtml(row.cells[i] || { bird: null });
    }
  }
  html += '</div>';
  return html;
}

function scoreTableHtml(panel) {
  const s = panel.score;
  return '<table class="scoretable"><tr>'
    + '<th>Birds</th><th>Eggs</th><th>Tuck</th><th>Cache</th><th>Bonus</th><th>Goals</th><th>Total</th></tr><tr>'
    + '<td>' + s.birds + '</td><td>' + s.eggs + '</td><td>' + s.tucked + '</td><td>' + s.cached
    + '</td><td>' + s.bonus + '</td><td>' + s.goals + '</td><td><b>' + s.total + '</b></td></tr></table>';
}

function foodHtml(panel) {
  const parts = panel.food.filter(f => f.count > 0).map(f => f.count + ' ' + esc(f.label));
  return '<div class="subline"><b>Food:</b> ' + (parts.length ? parts.join(', ') : '\\u2014') + '</div>';
}

function bonusHtml(panel) {
  if (!panel.bonus_cards.length) return '';
  const items = panel.bonus_cards.map(b =>
    '<div>' + esc(b.name) + ' \\u2014 ' + esc(b.text) + ' <b>(' + b.vp_now + ' VP)</b></div>');
  return '<div class="subline"><b>Bonus:</b>' + items.join('') + '</div>';
}

function seatHtml(panel) {
  const hand = panel.hand_names.length
    ? panel.hand_names.map(esc).join(', ') : '\\u2014';
  return '<div class="seat">'
    + '<h2>' + esc(panel.name)
    + ' <span class="cubes">' + panel.action_cubes_left + ' cubes left</span>'
    + ' <span class="total">' + panel.score.total + ' VP</span></h2>'
    + boardHtml(panel)
    + scoreTableHtml(panel)
    + foodHtml(panel)
    + '<div class="subline"><b>Hand (' + panel.hand_names.length + '):</b> ' + hand + '</div>'
    + bonusHtml(panel)
    + '</div>';
}

function sharedHtml(phase) {
  const trayParts = phase.tray.map(b =>
    b ? '<span class="tray-card" title="' + esc(b.power_text) + '">' + esc(b.name)
        + ' <small>(' + b.vp + 'VP ' + esc(b.food_cost) + ')</small></span>'
      : '<span class="tray-card empty">(empty)</span>');
  const goals = phase.round_goals.map(g =>
    '<div class="goal' + (g.scored ? ' scored' : '') + '">R' + g.round_num + ' ('
    + g.first_vp + '/' + g.second_vp + ' VP): ' + esc(g.description)
    + (g.scored ? ' \\u2713' : '') + '</div>');
  return '<div class="shared">'
    + '<div><b>Tray:</b><div class="tray-cards">' + trayParts.join('') + '</div></div>'
    + '<div class="subline"><b>Birdfeeder:</b> ' + esc(phase.feeder_text) + '</div>'
    + '<div class="goals"><b>Round goals:</b>' + goals.join('') + '</div>'
    + '</div>';
}

function visibleSeats(phase) {
  if (view === 'p0') return phase.panels.filter(p => p.player_id === 0);
  if (view === 'p1') return phase.panels.filter(p => p.player_id === 1);
  return phase.panels;
}

function lineVisible(line) {
  if (view === 'both') return true;
  if (line.player_id === null || line.player_id === undefined) return true;
  return (view === 'p0' && line.player_id === 0) || (view === 'p1' && line.player_id === 1);
}

function renderState(phase) {
  const seats = visibleSeats(phase).map(seatHtml).join('');
  document.getElementById('state-panel').innerHTML =
    '<div class="seats">' + seats + '</div>' + sharedHtml(phase);
}

function renderLog(phase) {
  const lines = phase.narration.filter(lineVisible);
  const log = document.getElementById('decision-log');
  if (!lines.length) {
    log.innerHTML = '<span class="empty">(no decisions for this view)</span>';
  } else {
    log.innerHTML = lines.map(l => {
      const who = (l.player_id === 0) ? 'p0' : (l.player_id === 1) ? 'p1' : 'global';
      return '<span class="ln ' + who + '">' + esc(l.text || ' ') + '</span>';
    }).join('');
  }
  log.classList.toggle('hidden', !logOpen);
  document.getElementById('log-toggle').textContent = (logOpen ? '\\u25be' : '\\u25b8') + ' Decision log';
}

function render() {
  const phase = DATA.phases[phaseIdx];
  document.getElementById('counter').textContent = 'Phase ' + (phaseIdx + 1) + ' / ' + DATA.phases.length;
  document.getElementById('phase-title').textContent = phase.title;
  document.getElementById('prev').disabled = (phaseIdx === 0);
  document.getElementById('next').disabled = (phaseIdx === DATA.phases.length - 1);
  renderState(phase);
  renderLog(phase);
}

function step(delta) {
  phaseIdx = Math.max(0, Math.min(DATA.phases.length - 1, phaseIdx + delta));
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
