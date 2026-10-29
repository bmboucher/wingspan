"""Compact fixed-size HTML cards for every bird and bonus card.

Each card is one self-contained HTML file (inline CSS, no external assets) of
``constants.CARD_WIDTH_PX`` x ``constants.CARD_HEIGHT_PX`` pixels, meant to be
shown in an iframe popover. The layout is this project's own: a name / points
header, habitat cells, food-cost chips, a nest / egg / wingspan row, and the
power text in a box tinted by its color. Every field comes from the parsed card
catalog; no card art or flavor text is used.

Files are named by the id the stats CSVs carry: ``bird/<bird_id>.html`` and
``bonus/<slug>.html`` (the slug of the bonus card's name).
"""

from __future__ import annotations

import html
import pathlib
import re

from wingspan.cards import schema
from wingspan.cards.parse import catalog
from wingspan.research import constants

_TAG_PATTERN = re.compile(r"\[([^\]]+)\]")
_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")

_POWER_LABELS = {
    schema.PowerColor.BROWN: "When activated",
    schema.PowerColor.WHITE: "When played",
    schema.PowerColor.PINK: "Once between turns",
    schema.PowerColor.YELLOW: "End of round",
    schema.PowerColor.NONE: "No power",
}
_HABITAT_LABELS = {
    schema.Habitat.FOREST: "Forest",
    schema.Habitat.GRASSLAND: "Grassland",
    schema.Habitat.WETLAND: "Wetland",
}
_WILD_LABEL = "wild"
_POINT_TAG = "point"
_TIER_SEPARATOR = ";"
_TIER_LABEL_SEPARATOR = ":"
_COST_OR_SEPARATOR = '<span class="sep">/</span>'

_CSS = """
:root{color-scheme:light dark;--bg:#fbf8f1;--fg:#1d1b16;--muted:#6b665a;--line:#cfc8b6;
--box:#efe9da;--forest:#3f7d4e;--grassland:#b89a2e;--wetland:#3b79a6;--off:#e4dfd0;
--brown:#c9b08a;--white:#ffffff;--pink:#f1b8cf;--none:#e4dfd0;
--invertebrate:#c0583a;--seed:#a8873a;--fish:#3b79a6;--fruit:#a8467a;--rodent:#7b6a58;--wild:#6b665a}
@media (prefers-color-scheme:dark){:root{--bg:#23211c;--fg:#ece7da;--muted:#a39d8d;--line:#4a463b;
--box:#2f2c25;--off:#34312a;--brown:#6b5a3f;--white:#4a4740;--pink:#7a3f58;--none:#34312a}}
*{box-sizing:border-box}
body{margin:0;background:transparent;font:12px/1.3 system-ui,sans-serif;color:var(--fg)}
.frame{width:%(width)dpx;height:%(height)dpx;padding:10px;border:1px solid var(--line);
border-radius:10px;background:var(--bg);display:flex;flex-direction:column;gap:7px;overflow:hidden}
.head{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
.name{font-size:16px;font-weight:700;line-height:1.15}
.sci{font-style:italic;color:var(--muted);font-size:11px}
.pts{flex:none;min-width:30px;height:30px;border-radius:15px;border:2px solid var(--fg);
display:flex;align-items:center;justify-content:center;font-weight:700;font-size:15px}
.habs{display:flex;gap:4px}
.hab{flex:1;text-align:center;padding:2px 0;border-radius:4px;font-size:11px;
background:var(--off);color:var(--muted)}
.hab.on{color:#fff;font-weight:600}
.hab.forest.on{background:var(--forest)}.hab.grassland.on{background:var(--grassland)}
.hab.wetland.on{background:var(--wetland)}
.row{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.lbl{color:var(--muted);font-size:11px;min-width:34px}
.chip,.ic{display:inline-block;padding:0 6px;border-radius:9px;color:#fff;font-size:11px;
line-height:17px;white-space:nowrap}
.ic{padding:0 4px;font-weight:600}
.invertebrate{background:var(--invertebrate)}.seed{background:var(--seed)}
.fish{background:var(--fish)}.fruit{background:var(--fruit)}.rodent{background:var(--rodent)}
.wild{background:var(--wild)}
.ic.egg,.ic.card,.ic.die,.ic.point,.ic.flocking,.ic.predator{background:var(--muted)}
.ic.forest{background:var(--forest)}.ic.grassland{background:var(--grassland)}
.ic.wetland{background:var(--wetland)}
.ic.bowl,.ic.cavity,.ic.ground,.ic.platform{background:var(--muted)}
.sep{color:var(--muted)}
.stats{display:flex;justify-content:space-between;color:var(--muted);font-size:11px}
.stats b{color:var(--fg)}
.power{flex:1;min-height:0;overflow:auto;padding:6px 8px;border-radius:6px;background:var(--box);
border-left:5px solid var(--none)}
.power.brown{border-color:var(--brown)}.power.white{border-color:var(--white)}
.power.pink{border-color:var(--pink)}
.power.fit{flex:none}
.when{font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.tags{margin-top:4px;display:flex;gap:4px}
.tiers{width:100%%;border-collapse:collapse}
.tiers td{padding:3px 0;border-bottom:1px solid var(--line)}
.tiers td:last-child{text-align:right;font-weight:700}
.kind{font-size:10px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
""" % {
    "width": constants.CARD_WIDTH_PX,
    "height": constants.CARD_HEIGHT_PX,
}


def write_cards(out_dir: pathlib.Path) -> list[pathlib.Path]:
    """Write one HTML file per core bird (``bird/<id>.html``) and bonus card
    (``bonus/<slug>.html``) under ``out_dir`` and return the written paths in
    catalog order."""
    bird_dir = out_dir / constants.CARD_HTML_BIRD_DIR
    bonus_dir = out_dir / constants.CARD_HTML_BONUS_DIR
    bird_dir.mkdir(parents=True, exist_ok=True)
    bonus_dir.mkdir(parents=True, exist_ok=True)

    written: list[pathlib.Path] = []
    for bird in catalog.birds_ordered():
        path = bird_dir / f"{bird.id}.html"
        path.write_text(bird_card_html(bird), encoding="utf-8")
        written.append(path)
    for bonus in catalog.bonus_cards_ordered():
        path = bonus_dir / f"{bonus_slug(bonus.name)}.html"
        path.write_text(bonus_card_html(bonus), encoding="utf-8")
        written.append(path)
    return written


def bonus_slug(name: str) -> str:
    """The file stem of a bonus card: its lowercased name with runs of
    non-alphanumerics collapsed to ``-``."""
    return _SLUG_PATTERN.sub("-", name.lower()).strip("-")


def bird_card_html(bird: schema.Bird) -> str:
    """The full HTML document of one bird card."""
    body = (
        _header(bird.name, str(bird.points), bird.scientific_name)
        + _habitats(bird)
        + f'<div class="row"><span class="lbl">Cost</span>{_cost(bird.food_cost)}</div>'
        + _stats(bird)
        + _power(bird)
    )
    return _document(bird.name, body)


def bonus_card_html(bonus: schema.BonusCard) -> str:
    """The full HTML document of one bonus card: name, qualifying condition and
    its payout (tier table or per-bird rate)."""
    body = (
        f'<div class="head"><div><div class="kind">Bonus card</div>'
        f'<div class="name">{html.escape(bonus.name)}</div></div></div>'
        f'<div class="power fit">{_tags_to_icons(bonus.condition)}</div>'
        f"{_payout(bonus)}"
    )
    return _document(bonus.name, body)


###### PRIVATE #######


def _document(title: str, body: str) -> str:
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f'<body><div class="frame">{body}</div></body></html>\n'
    )


def _header(name: str, points: str, scientific_name: str) -> str:
    return (
        f'<div class="head"><div><div class="name">{html.escape(name)}</div>'
        f'<div class="sci">{html.escape(scientific_name)}</div></div>'
        f'<div class="pts">{html.escape(points)}</div></div>'
    )


def _habitats(bird: schema.Bird) -> str:
    cells = "".join(
        f'<div class="hab {habitat.value}'
        f'{" on" if habitat in bird.habitats else ""}">{label}</div>'
        for habitat, label in _HABITAT_LABELS.items()
    )
    return f'<div class="habs">{cells}</div>'


def _cost(cost: schema.BirdCost) -> str:
    if cost.is_free():
        return '<span class="chip wild">free</span>'
    chips: list[str] = []
    for food in schema.ALL_FOODS:
        count = cost.specific_of(food)
        if count:
            chips.append(_chip(food.value, count, show_count=not cost.is_or_cost))
    if cost.wild:
        chips.append(_chip(_WILD_LABEL, cost.wild, show_count=not cost.is_or_cost))
    separator = _COST_OR_SEPARATOR if cost.is_or_cost else ""
    return separator.join(chips)


def _chip(label: str, count: int, *, show_count: bool) -> str:
    text = f"{count} {label}" if show_count else label
    return f'<span class="chip {label}">{text}</span>'


def _stats(bird: schema.Bird) -> str:
    nest = "no nest" if bird.nest == schema.NestType.NONE else bird.nest.value
    return (
        '<div class="stats">'
        f"<span><b>{html.escape(nest)}</b> nest</span>"
        f"<span><b>{bird.egg_limit}</b> eggs</span>"
        f"<span><b>{bird.wingspan_cm}</b> cm</span></div>"
    )


def _power(bird: schema.Bird) -> str:
    label = _POWER_LABELS[bird.color]
    text = _tags_to_icons(bird.raw_power_text)
    tags = "".join(
        f'<span class="ic {name}">{name}</span>'
        for name, active in (("flocking", bird.flocking), ("predator", bird.predator))
        if active
    )
    tag_row = f'<div class="tags">{tags}</div>' if tags else ""
    return (
        f'<div class="power {bird.color.value}"><div class="when">{label}</div>'
        f"{text}{tag_row}</div>"
    )


def _payout(bonus: schema.BonusCard) -> str:
    """The printed payout: one table row per ``label: points`` tier, or the
    single per-bird rate line."""
    if bonus.per_bird_vp is not None:
        body = _tags_to_icons(bonus.vp_text)
    else:
        rows = "".join(_tier_row(tier) for tier in bonus.vp_text.split(_TIER_SEPARATOR))
        body = f'<table class="tiers">{rows}</table>'
    return f'<div class="power fit"><div class="when">Scoring</div>{body}</div>'


def _tier_row(tier: str) -> str:
    label, _, points = tier.partition(_TIER_LABEL_SEPARATOR)
    return (
        f"<tr><td>{_tags_to_icons(label.strip())}</td>"
        f"<td>{_tags_to_icons(points.strip())}</td></tr>"
    )


def _tags_to_icons(text: str) -> str:
    """Escape ``text`` and render each ``[tag]`` as a small colored label (the
    ``[point]`` tag reads as the plain word ``pts``)."""
    escaped = html.escape(text)

    def icon(match: re.Match[str]) -> str:
        name = match.group(1)
        if name == _POINT_TAG:
            return " pts"
        return f'<span class="ic {name}">{name}</span>'

    return _TAG_PATTERN.sub(icon, escaped)
