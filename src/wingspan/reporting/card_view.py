"""Shared HTML display assets reused by the game-log and model-summary reports.

Holds the CSS and JavaScript that both ``game_log_html`` and ``html`` need for
bird-card rendering and the encoding-viewer modal, plus the ``bird_cell_info``
converter that flattens a ``cards.Bird`` to display primitives without any
per-game state.  ``BirdCatalogEntry`` and ``BirdCatalog`` Pydantic models live
in ``game_log_html`` (beside the ``BirdCellInfo`` and ``EncodedStripe`` types
they reference) so this module stays importable before ``game_log_html``
finishes loading, breaking the potential circular import.

Public exports
--------------
``CARD_CSS``, ``CARD_JS`` — bird-card renderer styles and functions.
``STRIPE_VIEWER_CSS``, ``STRIPE_VIEWER_JS`` — encoding-viewer modal styles
    and ``renderStripes`` / ``renderSubField`` functions.
``bird_cell_info(bird)`` — build a static (no played-state) ``BirdCellInfo``.
"""

from __future__ import annotations

import typing

from wingspan import cards

if typing.TYPE_CHECKING:
    from wingspan.reporting import game_log_html as _game_log_types

# ---------------------------------------------------------------------------
# Shared CSS

CARD_CSS: str = """\
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
.card-cell.selected { border: 3px solid #4ade80; box-shadow: 0 0 6px rgba(74,222,128,.35); }
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
"""

STRIPE_VIEWER_CSS: str = """\
/* === Encoding viewer modal === */
#enc-modal {
  display: none; position: fixed; inset: 0; z-index: 300;
  background: rgba(0,0,0,.6); align-items: center; justify-content: center;
}
#enc-modal.open { display: flex; }
#enc-dialog {
  background: #0f172a; border-radius: 10px; padding: 0;
  width: min(700px, 95vw); max-height: 90vh;
  display: flex; flex-direction: column; overflow: hidden;
  box-shadow: 0 8px 40px rgba(0,0,0,.6);
}
#enc-header {
  display: flex; align-items: center; padding: 10px 16px;
  border-bottom: 1px solid #1e293b; flex-shrink: 0; gap: 8px;
}
#enc-opt-label { color: #a7f3d0; font-weight: 700; font-size: 13px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#enc-close {
  background: none; border: none; color: #94a3b8; font-size: 18px;
  cursor: pointer; padding: 2px 6px; border-radius: 4px; flex-shrink: 0;
}
#enc-close:hover { background: #1e293b; color: #e2e8f0; }
#enc-body { padding: 10px 16px 16px; overflow-y: auto; flex: 1; }
.enc-section-title {
  font-size: 12px; font-weight: 700; color: #94a3b8;
  text-transform: uppercase; letter-spacing: 1px; margin: 10px 0 5px;
}
.enc-section-title:first-child { margin-top: 2px; }
.enc-none { color: #475569; font-size: 12px; font-style: italic; padding: 4px 0; }
.enc-stripe {
  margin-bottom: 6px; border: 1px solid #1e293b; border-radius: 5px; overflow: hidden;
}
.enc-stripe > summary {
  padding: 5px 9px; font-size: 11px; font-weight: 600; color: #e2e8f0;
  background: #1e293b; cursor: pointer; list-style: none; user-select: none;
}
.enc-stripe > summary::-webkit-details-marker { display: none; }
.enc-stripe > summary:hover { background: #334155; }
.enc-table { width: 100%; border-collapse: collapse; font-size: 10px; }
.enc-table td { padding: 3px 8px; border-bottom: 1px solid #0f172a; vertical-align: top; }
.enc-table tr:last-child td { border-bottom: none; }
.enc-name { color: #93c5fd; font-family: 'Fira Code', Consolas, monospace; white-space: nowrap; width: 1px; }
.enc-desc { color: #94a3b8; }
.enc-notes { color: #475569; font-size: 9px; display: block; }
.enc-val { color: #4ade80; font-family: 'Fira Code', Consolas, monospace; white-space: nowrap; width: 1px; text-align: right; padding-right: 12px; }
.enc-range { color: #475569; white-space: nowrap; width: 1px; }
.di-opt { cursor: pointer; }
.di-opt:hover { background: rgba(255,255,255,.05); }
"""

# ---------------------------------------------------------------------------
# Shared JavaScript

CARD_JS: str = r"""
const FOOD_EMOJI = {
  invertebrate: '\u{1F41B}', seed: '\u{1F33E}', fish: '\u{1F41F}',
  fruit: '\u{1F352}', rodent: '\u{1F400}', wild: '\u{1F308}', choice: '\u{1F41B}/\u{1F33E}'
};
const HAB_ICON = { Forest: '\u{1F7E9}', Grassland: '\u{1F7E8}', Wetland: '\u{1F7E6}' };

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}

function eggGlyphs(eggs, limit) {
  if (limit <= 0) return '';
  const laid = Math.max(0, Math.min(eggs, limit));
  return '●'.repeat(laid) + '○'.repeat(limit - laid);
}

function foodCostHtml(slots, isOr) {
  if (!slots || !slots.length) return 'free';
  const emojis = slots.map(f => FOOD_EMOJI[f] || esc(f));
  if (isOr) return '(' + emojis.join('/') + ')';
  return emojis.join('');
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
  const selCls = bird.selected ? ' selected' : '';
  const eggs = eggGlyphs(bird.eggs, bird.egg_limit);
  const cost = foodCostHtml(bird.food_cost_slots, bird.food_cost_is_or);
  const habSq = habSquaresHtml(bird.habitats);
  const status = cardStatusText(bird);
  return '<div class="card-cell ' + pwCls + selCls + '">'
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
"""

STRIPE_VIEWER_JS: str = r"""
function renderStripes(stripes) {
  return stripes.map(s =>
    '<details class="enc-stripe" open>'
    + '<summary><b>' + esc(s.name) + '</b> — ' + esc(s.description) + '</summary>'
    + '<table class="enc-table">'
    + (s.sub_fields || []).map(renderSubField).join('')
    + '</table>'
    + '</details>'
  ).join('');
}

function renderSubField(field) {
  let val = '';
  if (field.decoded_label != null) {
    val = field.decoded_label;
  } else if (field.active_index != null) {
    val = 'index ' + field.active_index;
  } else if (field.raw_value != null) {
    val = Number.isInteger(field.raw_value) ? String(field.raw_value) : field.raw_value.toFixed(4);
  } else if (field.raw_values && field.raw_values.length) {
    val = field.raw_values.map(v => Number.isInteger(v) ? String(v) : v.toFixed(4)).join(', ');
  }
  const notes = field.notes ? '<span class="enc-notes">' + esc(field.notes) + '</span>' : '';
  return '<tr>'
    + '<td class="enc-name">' + esc(field.name) + '</td>'
    + '<td class="enc-desc">' + esc(field.description) + notes + '</td>'
    + '<td class="enc-val">' + esc(val) + '</td>'
    + '<td class="enc-range">' + esc(field.value_range) + '</td>'
    + '</tr>';
}
"""

# ---------------------------------------------------------------------------
# Bird cell builder (static, no per-game state)


def bird_cell_info(bird: cards.Bird) -> _game_log_types.BirdCellInfo:
    """Build a ``BirdCellInfo`` from a bird's static card attributes only.

    The played-state fields (``eggs``, ``tucked``, ``cached``) default to 0
    since this is a catalog view with no active game context.  Callers that
    have a ``PlayedBird`` should call this first and then
    ``model_copy(update=...)`` to fill those fields.
    """
    from wingspan.agents import display
    from wingspan.reporting import game_log_html

    # Build food-cost slot list for emoji rendering.
    slots: list[str] = []
    for food, specific_count in zip(cards.ALL_FOODS, bird.food_cost.specific):
        repeat = 1 if (bird.food_cost.is_or_cost and specific_count) else specific_count
        slots.extend([food.value] * repeat)
    slots.extend(["wild"] * bird.food_cost.wild)

    return game_log_html.BirdCellInfo(
        name=bird.name,
        vp=bird.points,
        nest=bird.nest.value,
        wingspan_cm=bird.wingspan_cm,
        habitats="/".join(habitat.value for habitat in bird.habitats),
        food_cost=display.format_cost(bird.food_cost),
        food_cost_slots=slots,
        food_cost_is_or=bird.food_cost.is_or_cost,
        egg_limit=bird.egg_limit,
        eggs=0,
        tucked=0,
        cached=0,
        power_color=bird.power.color.value,
        power_text=bird.plain_power_text,
    )
