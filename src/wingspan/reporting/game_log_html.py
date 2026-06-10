"""Self-contained HTML viewer for a single ``wingspan play`` game log.

Where :mod:`wingspan.reporting.html` documents a trained *network*, this module
renders one *game*: a navigable, phase-by-phase replay of the detailed log the
engine already produces. The page shows exactly one phase (a setup block, a
round banner, or a single player turn) at a time, with prev/next arrows; a
``P0 / P1 / both`` toggle at the top controls which seat's board is pinned and
which decision lines are shown; the current game state (3x5 board grids, hands,
tray, birdfeeder, scores, round goals) is pinned at the top of the window, and
the turn's decision narration sits in a collapsible panel beneath.

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
main { max-width: 1440px; margin: 0 auto; padding: 16px 18px 60px; }
#state-panel {
  position: sticky; top: 96px; z-index: 30; background: #f8fafc;
  border: 1px solid #cbd5e1; border-radius: 10px; padding: 14px;
  box-shadow: 0 2px 8px rgba(0,0,0,.08); margin-bottom: 16px;
  max-height: calc(100vh - 112px); overflow-y: auto;
}

/* === Player boards row === */
.boards-row { display: flex; gap: 12px; }
.player-section { flex: 1 1 0; min-width: 0; overflow-x: auto; }
.player-header {
  display: flex; align-items: baseline; gap: 8px; margin-bottom: 6px;
  padding-bottom: 4px; border-bottom: 2px solid #0f3d2e;
}
.player-name { font-weight: 700; font-size: 14px; flex-shrink: 0; }
.player-cubes { font-size: 12px; letter-spacing: 2px; color: #334155; }
.player-vp { margin-left: auto; font-weight: 800; font-size: 16px; color: #0f3d2e; flex-shrink: 0; }
.board-row { display: flex; align-items: stretch; gap: 4px; margin-bottom: 4px; }
.hab-label {
  display: flex; align-items: center; justify-content: center; text-align: center;
  font-weight: 700; font-size: 10px; color: #fff; border-radius: 4px;
  width: 56px; flex-shrink: 0;
}
.hab-forest    { background: #166534; }
.hab-grassland { background: #ca8a04; }
.hab-wetland   { background: #0369a1; }

/* === Card cell — fixed size, identical in board / tray / hand === */
.card-cell {
  width: 110px; height: 128px; flex-shrink: 0;
  border: 1px solid #cbd5e1; border-radius: 5px; overflow: hidden;
  display: flex; flex-direction: column; background: #fff;
}
.card-cell.empty { border-style: dashed; background: #f1f5f9; }
.card-hdr {
  padding: 3px 5px; background: #f8fafc;
  border-bottom: 1px solid #e2e8f0; flex-shrink: 0;
}
.card-name {
  font-weight: 700; font-size: 9.5px; line-height: 1.15;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.card-meta { color: #475569; font-size: 7.5px; line-height: 1.3; }
.card-power {
  flex: 1; padding: 3px 5px; font-size: 7.5px; line-height: 1.35; overflow: hidden;
}
.card-cell.pw-brown  .card-power { background: #f3e9dd; }
.card-cell.pw-white  .card-power { background: #ffffff; }
.card-cell.pw-pink   .card-power { background: #fce7f3; }
.card-cell.pw-yellow .card-power { background: #fef9c3; }
.card-eggs {
  height: 16px; padding: 0 5px; text-align: right; letter-spacing: 1px;
  font-size: 10px; border-top: 1px solid #e2e8f0; flex-shrink: 0;
  color: #64748b; display: flex; align-items: center; justify-content: flex-end;
}

/* === Middle row: tray | birdfeeder | player hand === */
.middle-row { display: flex; gap: 8px; align-items: flex-start; margin-top: 8px; }
.panel { border: 1px solid #cbd5e1; border-radius: 6px; padding: 6px 8px; background: #fff; }
.panel-title {
  font-size: 8.5px; font-weight: 700; color: #94a3b8; text-align: center;
  letter-spacing: 1px; text-transform: uppercase; margin-top: 5px;
}
.tray-panel .card-row { display: flex; gap: 4px; }
.feeder-panel { min-width: 100px; font-size: 10.5px; line-height: 1.7; white-space: pre-line; }
.hand-panel { flex: 1; min-width: 0; }
.hand-scroll { display: flex; gap: 4px; overflow-x: auto; padding-bottom: 4px; }
.hand-scroll::-webkit-scrollbar { height: 5px; }
.hand-scroll::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
.hand-player-lbl {
  writing-mode: vertical-rl; font-size: 8.5px; font-weight: 700;
  color: #64748b; align-self: stretch; display: flex; align-items: center;
  padding: 0 3px; flex-shrink: 0;
}

/* === Bottom row: round goals | point sources === */
.bottom-row { display: flex; gap: 8px; margin-top: 8px; }
.goals-panel { flex: 1 1 0; }
.scores-panel { flex: 1 1 0; }
.goal-item {
  font-size: 10.5px; padding: 3px 2px; border-bottom: 1px solid #f1f5f9;
  display: flex; align-items: baseline; gap: 6px; line-height: 1.3;
}
.goal-item:last-child { border-bottom: none; }
.goal-item.scored { color: #15803d; }
.goal-rnum { font-weight: 700; color: #64748b; font-size: 9.5px; flex-shrink: 0; }
.goal-pay { color: #94a3b8; font-size: 9px; flex-shrink: 0; }
.goal-check { color: #15803d; font-weight: 700; }
.score-tbl { border-collapse: collapse; font-size: 10px; width: 100%; }
.score-tbl th {
  background: #e2e8f0; padding: 2px 5px; text-align: right;
  font-size: 9px; white-space: nowrap;
}
.score-tbl th:first-child { text-align: left; }
.score-tbl td { padding: 2px 5px; text-align: right; border-bottom: 1px solid #f1f5f9; }
.score-tbl td:first-child { text-align: left; font-weight: 700; }
.score-tbl .total-col { font-weight: 800; color: #0f3d2e; }

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

_SCRIPT = """\
'use strict';
const DATA = JSON.parse(document.getElementById('game-log-data').textContent);
const HAB_CLASS = {'Forest':'hab-forest','Grassland':'hab-grassland','Wetland':'hab-wetland'};
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

function cubeGlyphs(n) {
  const MAX = 8;
  const filled = Math.max(0, Math.min(n, MAX));
  return '\\u25a0'.repeat(filled) + '\\u25a1'.repeat(MAX - filled);
}

function cardCellHtml(bird) {
  if (!bird) return '<div class="card-cell empty"></div>';
  const pwCls = 'pw-' + esc(bird.power_color || 'none');
  const eggs = eggGlyphs(bird.eggs, bird.egg_limit);
  const meta = bird.vp + 'VP \\u00b7 ' + esc(bird.food_cost) + ' \\u00b7 ' + esc(bird.nest);
  return '<div class="card-cell ' + pwCls + '">'
    + '<div class="card-hdr">'
    +   '<div class="card-name">' + esc(bird.name) + '</div>'
    +   '<div class="card-meta">' + meta + '</div>'
    +   '<div class="card-meta">' + esc(bird.habitats) + '</div>'
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

function playerSectionHtml(panel) {
  const cubes = cubeGlyphs(panel.action_cubes_left);
  return '<div class="player-section">'
    + '<div class="player-header">'
    +   '<span class="player-name">P' + panel.player_id + '</span>'
    +   '<span class="player-cubes">' + esc(cubes) + '</span>'
    +   '<span class="player-vp">' + panel.score.total + ' VP</span>'
    + '</div>'
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
  return '<div class="panel feeder-panel">'
    + esc(phase.feeder_text)
    + '<div class="panel-title">Birdfeeder</div>'
    + '</div>';
}

function handPanelHtml(seats) {
  let inner = '';
  for (const panel of seats) {
    if (!panel.hand.length) continue;
    if (seats.length > 1) {
      inner += '<div class="hand-player-lbl">P' + panel.player_id + '</div>';
    }
    inner += panel.hand.map(b => cardCellHtml(b)).join('');
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
  const items = phase.round_goals.map(g => {
    const cls = g.scored ? ' scored' : '';
    const check = g.scored ? ' <span class="goal-check">\\u2713</span>' : '';
    return '<div class="goal-item' + cls + '">'
      + '<span class="goal-rnum">R' + g.round_num + '</span>'
      + '<span>' + esc(g.description) + '</span>'
      + '<span class="goal-pay">(' + g.first_vp + '/' + g.second_vp + 'VP)</span>'
      + check
      + '</div>';
  });
  return '<div class="panel goals-panel">'
    + items.join('')
    + '<div class="panel-title">Round-End Goals</div>'
    + '</div>';
}

function scoresPanelHtml(seats) {
  const cats = ['birds','eggs','tucked','cached','bonus','goals','total'];
  const hdrs = ['Birds','Eggs','Tuck','Cache','Bonus','Goals','Total'];
  let html = '<div class="panel scores-panel"><table class="score-tbl"><tr>'
    + '<th>Player</th>'
    + hdrs.map(h => '<th>' + h + '</th>').join('')
    + '</tr>';
  for (const panel of seats) {
    const s = panel.score;
    html += '<tr><td>P' + panel.player_id + '</td>'
      + cats.map((c, i) => '<td' + (i === cats.length - 1 ? ' class="total-col"' : '') + '>' + s[c] + '</td>').join('')
      + '</tr>';
  }
  html += '</table><div class="panel-title">Point Sources</div></div>';
  return html;
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
  const seats = visibleSeats(phase);
  const boardsRow = '<div class="boards-row">' + seats.map(playerSectionHtml).join('') + '</div>';
  const middleRow = '<div class="middle-row">'
    + trayPanelHtml(phase) + feederPanelHtml(phase) + handPanelHtml(seats)
    + '</div>';
  const bottomRow = '<div class="bottom-row">'
    + goalsPanelHtml(phase) + scoresPanelHtml(seats)
    + '</div>';
  document.getElementById('state-panel').innerHTML = boardsRow + middleRow + bottomRow;
}

function renderLog(phase) {
  const lines = phase.narration.filter(lineVisible);
  const log = document.getElementById('decision-log');
  if (!lines.length) {
    log.innerHTML = '<span class="empty">(no decisions for this view)</span>';
  } else {
    log.innerHTML = lines.map(l => {
      const who = l.player_id === 0 ? 'p0' : l.player_id === 1 ? 'p1' : 'global';
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
  document.getElementById('prev').disabled = phaseIdx === 0;
  document.getElementById('next').disabled = phaseIdx === DATA.phases.length - 1;
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
