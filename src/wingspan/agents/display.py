"""Human-readable formatters for cards and game state.

These helpers exist so the interactive CLI agent can present birds, bonus
cards, and board state with enough detail for a human to make an informed
choice -- the canonical engine-side ``Choice.label`` strings stay short for
the log, while this module produces the verbose rendering shown only to
the human at decision time.

All functions are pure (state-in, string-out) and have no side effects.
"""

from __future__ import annotations

import functools
import re
import sys
import typing

import pydantic

from wingspan import cards, state
from wingspan.engine import helpers, scoring

# Inline-icon glyphs for the printed ``[card]`` / ``[egg]`` tags wingsearch
# uses in power, goal, and bonus text. Both are emoji-presentation (width 2) so
# columns stay aligned; swap either for a different glyph if your terminal font
# renders it poorly.
_EGG = "🥚"
_CARD = "🃏"

# Egg-capacity glyphs for a bird in play: a filled circle marks each egg laid,
# a hollow circle each empty slot. These are text-presentation (width 1), so --
# unlike the ``🥚`` emoji, which terminals render in their own colour and ignore
# ANSI foreground codes for -- they honour the yellow tint below; the
# filled/hollow shape also tells laid from empty in plain logs with no colour.
_EGG_FILLED = "●"
_EGG_EMPTY = "○"

# 24-bit yellow tinting the filled (laid) egg glyphs; same hue as the yellow
# power color so the board reads consistently. Only emitted on a real terminal.
_LAID_YELLOW = "\x1b[38;2;245;221;60m"

# Truecolor styling for the bird power-text line: each power color renders the
# text on its printed card color rather than a literal "[color]" tag. Values
# are (foreground, background) RGB; emitted as 24-bit ANSI, which modern
# terminals (Windows Terminal included) support. Power colors absent here
# (NONE) — and any output that is not a real terminal — fall back to plain
# text. The core set has no blue/teal power, so none is mapped.
type _Rgb = tuple[int, int, int]

_INK_BLACK: _Rgb = (0, 0, 0)
_INK_WHITE: _Rgb = (255, 255, 255)
_POWER_COLORS: dict[cards.PowerColor, tuple[_Rgb, _Rgb]] = {
    cards.PowerColor.BROWN: (_INK_WHITE, (124, 80, 48)),
    cards.PowerColor.WHITE: (_INK_BLACK, _INK_WHITE),
    cards.PowerColor.PINK: (_INK_BLACK, (255, 182, 193)),
    cards.PowerColor.YELLOW: (_INK_BLACK, (245, 221, 60)),
}
_ANSI_RESET = "\x1b[0m"
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")

# Short column headers for the board's two-row food tables, keyed in
# ``cards.ALL_FOODS`` order. Kept terse so the counter table stays narrow.
_FOOD_ABBR: dict[cards.Food, str] = {
    cards.Food.INVERTEBRATE: "inv",
    cards.Food.SEED: "seed",
    cards.Food.FISH: "fish",
    cards.Food.FRUIT: "berry",
    cards.Food.RODENT: "rod",
}

# Patterns for turning a printed bonus VP string into compact natural text.
# ``per`` flags a per-bird payout ("2[point] per bird" -> "2VP each"); the
# three count-band patterns turn one "A to B / A+ / A birds: N[point]" clause
# into "(A-B) NVP" / "(A+) NVP" / "(A) NVP". Range must be tried before exact.
_VP_PER_RE = re.compile(r"\bper\b", re.I)
_VP_RANGE_RE = re.compile(r"(\d+)\s*to\s*(\d+)\s*birds?\s*:\s*(\d+)\s*\[point\]", re.I)
_VP_PLUS_RE = re.compile(r"(\d+)\s*\+\s*birds?\s*:\s*(\d+)\s*\[point\]", re.I)
_VP_EXACT_RE = re.compile(r"(\d+)\s*birds?\s*:\s*(\d+)\s*\[point\]", re.I)


def _icons(text: str) -> str:
    """Replace the printed ``[card]`` / ``[egg]`` icon tags with glyphs."""
    return text.replace("[card]", _CARD).replace("[egg]", _EGG)


# ---------------------------------------------------------------------------
# Public formatters


def strip_ansi(text: str) -> str:
    """Remove ANSI SGR escape sequences from ``text`` (for plain-text file output)."""
    return _ANSI_ESCAPE_RE.sub("", text)


def format_cost(cost: cards.BirdCost) -> str:
    """Compact rendering of a printed food cost.

    A count of one is dropped (``invertebrate`` rather than ``1invertebrate``);
    higher counts keep the prefix (``2seed``). The trailing ``*`` denotes wild
    slots (any food).

    Examples: ``seed``, ``invertebrate+fish``, ``2*``, ``free``.
    """
    parts: list[str] = [
        _count_food(count, food)
        for food, count in zip(cards.ALL_FOODS, cost.specific)
        if count > 0
    ]
    if cost.wild:
        parts.append("*" if cost.wild == 1 else f"{cost.wild}*")
    return "+".join(parts) if parts else "free"


def _count_food(count: int, food: cards.Food) -> str:
    """``food`` for a single item, ``Nfood`` for more than one."""
    return food.value if count == 1 else f"{count}{food.value}"


def format_food_pool(pool: state.FoodPool) -> str:
    """Compact rendering of a food pool. Lists only non-zero foods."""
    parts = [f"{count}{food.value}" for food, count in pool.items() if count > 0]
    return "+".join(parts) if parts else "(empty)"


def format_can_play(
    birds: typing.Sequence[cards.Bird],
    foods: typing.Sequence[cards.Food],
) -> str:
    """Which of ``birds`` could be played immediately given a ``foods`` supply.

    A bird is playable when its printed food cost can be met from a pool of one
    of each food in ``foods``; at setup every habitat is empty and the first
    slot costs no eggs, so food is the only gate. Reads ``Can Play: None`` or
    ``Can Play: American Kestrel, White-breasted Nuthatch``.
    """
    pool = state.FoodPool.from_dict({food: 1 for food in foods})
    playable = [
        bird.name for bird in birds if helpers.enumerate_payments(pool, bird.food_cost)
    ]
    return "Can Play: " + (", ".join(playable) if playable else "None")


def format_bird(bird: cards.Bird, name_width: int = 0) -> str:
    """One-line bird summary with a fixed-width name then compact stats.

    Example (``name_width`` left-justifies the name so a column of birds
    lines up at the ``|`` separator)::

        Forster's Tern        | 4VP invertebrate+fish wetland star 🥚🥚 79cm

    Stats run points, food cost, habitat(s), nest type, egg limit (one 🥚 per
    slot), and wingspan; ``flocking``/``predator`` flags trail in parentheses
    when set. Pass the longest name in the group as ``name_width`` (0 = no
    padding). Use :func:`format_bird_full` to add the power text on a second
    line.
    """
    return _bird_head(bird, name_width)


def format_bird_full(
    bird: cards.Bird, indent: str = "      ", name_width: int = 0
) -> str:
    """Multi-line bird summary: the compact stat line then the power text.

    The power text is rendered on its card-color background (brown / white /
    pink / yellow) instead of a leading ``[color]`` tag; see
    :data:`_POWER_COLORS`.
    """
    return _with_power_text(_bird_head(bird, name_width), bird, indent)


def format_played_bird_full(
    pb: state.PlayedBird, indent: str = "      ", name_width: int = 0
) -> str:
    """Board rendering of a bird in play: the full card display from
    :func:`format_bird_full` plus its per-game mutable state.

    Laid eggs show as filled yellow circles in the egg-capacity display and
    empty slots as hollow circles; tucked cards and cached food, when present,
    trail in brackets (e.g. ``[tucked 2, food 1]``).
    """
    head = _bird_head(pb.bird, name_width, eggs_laid=pb.eggs)
    extras = [
        f"{label} {n}"
        for label, n in (("tucked", pb.tucked_cards), ("food", pb.cached_food.total()))
        if n
    ]
    if extras:
        head = f"{head} [{', '.join(extras)}]"
    return _with_power_text(head, pb.bird, indent)


def _bird_head(bird: cards.Bird, name_width: int, eggs_laid: int = 0) -> str:
    """The fixed-width name + compact stat line shared by every bird display.

    ``eggs_laid`` shows that many egg-capacity slots as filled yellow circles
    (used for birds in play); 0 leaves the whole capacity hollow (cards not in
    play).
    """
    habs = "/".join(habitat.value for habitat in bird.habitats)
    flags = [
        label
        for label, enabled in (("flocking", bird.flocking), ("predator", bird.predator))
        if enabled
    ]
    stats = " ".join(
        part
        for part in (
            f"{bird.points}VP",
            format_cost(bird.food_cost),
            habs,
            bird.nest.value,
            _egg_capacity(bird.egg_limit, eggs_laid),
            f"{bird.wingspan_cm}cm",
        )
        if part
    )
    flag_str = f" ({', '.join(flags)})" if flags else ""
    return f"{bird.name.ljust(name_width)} | {stats}{flag_str}"


def _with_power_text(head: str, bird: cards.Bird, indent: str) -> str:
    """Append the bird's colour-styled power text under ``head`` (if any)."""
    power_text = bird.raw_power_text.strip()
    if not power_text:
        return head
    styled = _style_power_text(bird.power.color, _icons(power_text))
    return f"{head}\n{indent}{styled}"


def _egg_capacity(egg_limit: int, eggs_laid: int = 0) -> str:
    """Egg-capacity glyphs: a filled circle per egg laid, a hollow circle per
    empty slot. Laid eggs are tinted yellow on a real terminal; off one the
    filled/hollow shape still tells them apart. An empty capacity yields ``""``
    so the stat line drops the field."""
    if egg_limit <= 0:
        return ""
    laid = min(max(eggs_laid, 0), egg_limit)
    filled, empty = _EGG_FILLED * laid, _EGG_EMPTY * (egg_limit - laid)
    if laid <= 0 or not sys.stdout.isatty():
        return f"{filled}{empty}"
    return f"{_LAID_YELLOW}{filled}{_ANSI_RESET}{empty}"


def _style_power_text(color: cards.PowerColor, text: str) -> str:
    """Wrap ``text`` in its power-color ANSI styling; plain on non-terminals."""
    style = _POWER_COLORS.get(color)
    if style is None or not sys.stdout.isatty():
        return text
    (fr, fg, fb), (br, bg, bb) = style
    return f"\x1b[38;2;{fr};{fg};{fb};48;2;{br};{bg};{bb}m{text}{_ANSI_RESET}"


def format_played_bird(pb: state.PlayedBird) -> str:
    """Compact rendering of a bird in play (name + per-bird mutable state)."""
    parts = [f"eggs={pb.eggs}/{pb.bird.egg_limit}"]
    if pb.cached_food.total():
        parts.append(f"cached={pb.cached_food.format()}")
    if pb.tucked_cards:
        parts.append(f"tucked={pb.tucked_cards}")
    return f"{pb.bird.name} ({', '.join(parts)})"


def format_bonus(bc: cards.BonusCard, name_width: int = 0) -> str:
    """One-line bonus-card summary: fixed-width name then natural scoring text.

    Examples (``name_width`` left-justifies the name so a column lines up at
    the ``|`` separator)::

        Omnivore Specialist | Birds that eat [wild] - 2VP each
        Forester            | Birds that can only live in [forest] | (3-4) 4VP (5) 8VP

    A per-bird payout reads ``... - NVP each``; a tiered payout lists each
    ``(count) NVP`` band. Use :func:`format_bonus_score_now` (in play) or
    :func:`format_bonus_with_setup_help` (pre-game) to add the usefulness line.
    """
    return f"{bc.name.ljust(name_width)} | {_bonus_detail(bc)}"


def format_bonus_score_now(bc: cards.BonusCard, player: state.Player) -> str:
    """In-play usefulness: qualifying birds on ``player``'s board and the VP
    they score right now, e.g. ``Currently 2 = 3VP``."""
    count = _qualifying(bc, (pb.bird for row in player.board.values() for pb in row))
    return f"Currently {count} = {scoring.bonus_score(player, bc)}VP"


def format_bonus_with_setup_help(
    bc: cards.BonusCard,
    hand_birds: typing.Sequence[cards.Bird],
    tray_birds: typing.Sequence[cards.Bird],
    selected_hand_birds: typing.Sequence[cards.Bird],
    name_width: int = 0,
) -> str:
    """:func:`format_bonus` plus, for a type-counting card, a second line with
    how many of the qualifying birds in hand you've *selected* so far, out of
    the total qualifying in hand — plus how many sit in the display (tray) and
    the share of the whole core-set deck that qualifies.

    Reads ``0/1 in hand, 2 in the display, 17% of all birds`` before the
    matching bird is kept, ``1/1 in hand, 2 in the display, 17% of all birds``
    once it is. Dynamic cards — whose payout depends on eggs, hand size, or
    board layout rather than a fixed bird property — can't be assessed before
    the game and get no second line.
    """
    head = format_bonus(bc, name_width)
    help_line = _bonus_setup_help(bc, hand_birds, tray_birds, selected_hand_birds)
    return head if help_line is None else f"{head}\n  {help_line}"


def format_board(game_state: state.GameState, player: state.Player) -> str:
    """Multi-line summary of ``player``'s board and resources plus round context.

    The layout, top to bottom: a ``Round R : Cube N/T`` header; a two-row
    counter table (food on hand, eggs laid/capacity, live VP, round-goal
    points); the round goal; a two-row birdfeeder table; the three habitat
    rows (each led by its current action reward, then every bird in play
    rendered in full); the tray and hand (also rendered in full); and the
    bonus cards with their current value.
    """
    lines = [_board_header(game_state, player), ""]
    lines.extend(_counter_rows(game_state, player))
    lines.append("")
    goal = (
        game_state.round_goals[game_state.round_idx] if game_state.round_goals else None
    )
    if goal is not None:
        lines.append(f"  round goal: {_icons(goal.description)}")
    lines.append("  birdfeeder:")
    lines.extend(_food_rows(game_state.birdfeeder.counts))
    if game_state.birdfeeder.choice_dice:
        lines.append(
            f"    invertebrate/seed (choice): {game_state.birdfeeder.choice_dice}"
        )
    lines.append("")
    lines.extend(_habitat_rows(player))
    lines.append("")
    lines.extend(_card_list_rows("tray", [b for b in game_state.tray if b is not None]))
    lines.extend(_card_list_rows("hand", player.hand))
    lines.extend(_bonus_rows(player))
    return "\n".join(lines)


###### PRIVATE #######

#### Board layout ####


def _board_header(game_state: state.GameState, player: state.Player) -> str:
    """The ``=== [P0] Round 1 : Cube 1/8 ===`` banner.

    The cube counter is which of this round's action cubes is about to be
    placed (1-based) out of the round's total."""
    total_cubes = state.ROUND_CUBES[game_state.round_idx]
    cube_num = total_cubes - player.action_cubes_left + 1
    return (
        f"=== [{player.name}] Round {game_state.round_idx + 1} : "
        f"Cube {cube_num}/{total_cubes} ==="
    )


def _counter_rows(game_state: state.GameState, player: state.Player) -> list[str]:
    """The two-row table of single-number resources: the five foods on hand,
    eggs laid / total egg capacity, live VP, and the live round-goal standing.

    The round-goal column shows this player's current count toward the round
    goal, the place that count earns versus the opponent, and the VP it would
    award if the round ended now (e.g. ``3 (1st place = 4 VP)``)."""
    capacity = sum(pb.bird.egg_limit for row in player.board.values() for pb in row)
    pairs = [(str(player.food[food]), _FOOD_ABBR[food]) for food in cards.ALL_FOODS]
    pairs.append((f"{player.total_eggs}/{capacity}", "eggs"))
    pairs.append((str(scoring.running_score(player)), "VP"))
    pairs.append((_round_goal_cell(game_state, player), "round pts"))
    return _aligned_rows(pairs)


def _round_goal_cell(game_state: state.GameState, player: state.Player) -> str:
    """The round-pts cell: ``count (Nth place = V VP)`` for the current round
    goal, or the accumulated round-goal points when no goal is in play."""
    if not game_state.round_goals:
        return str(player.round_goal_points)
    standing = scoring.round_goal_standing(game_state, player)
    return f"{standing.count} ({_ordinal(standing.place)} place =" f" {standing.vp} VP)"


def _ordinal(number: int) -> str:
    """Ordinal string for a small place number (``1`` -> ``1st``)."""
    suffix = {1: "st", 2: "nd", 3: "rd"}.get(number, "th")
    return f"{number}{suffix}"


def _food_rows(pool: state.FoodPool) -> list[str]:
    """The two-row table for a food pool (the birdfeeder)."""
    pairs = [(str(pool[food]), _FOOD_ABBR[food]) for food in cards.ALL_FOODS]
    return _aligned_rows(pairs)


def _aligned_rows(pairs: typing.Sequence[tuple[str, str]]) -> list[str]:
    """Render ``(value, label)`` columns as a value row over a label row, each
    column centred to the wider of its two cells. Both rows share a 2-space
    left margin so they line up under the rest of the board."""
    values: list[str] = []
    labels: list[str] = []
    for value, label in pairs:
        width = max(len(value), len(label))
        values.append(value.center(width))
        labels.append(label.center(width))
    gap = "   "
    return [f"  {gap.join(values)}", f"  {gap.join(labels)}"]


def _habitat_rows(player: state.Player) -> list[str]:
    """The three habitat rows. Each lists every bird in play there (rendered
    in full, the power line indented under the stat line for readability),
    then closes with the action reward and its optional resource trade."""
    lines: list[str] = []
    for habitat in cards.ALL_HABITATS:
        lines.append(f"  {habitat.value}:")
        row = player.board[habitat]
        name_width = max((len(pb.bird.name) for pb in row), default=0)
        for pb in row:
            block = format_played_bird_full(pb, indent="    ", name_width=name_width)
            lines.extend(_indent_block(block, "      "))
        lines.append(f"      {_habitat_action_line(player.board, habitat)}")
    return lines


def _habitat_action_line(board: state.Board, habitat: cards.Habitat) -> str:
    """The habitat action's reward, plus its one-step resource trade when the
    cube lands on a trade space.

    On a trade space (odd bird count) the grassland line reads
    ``+3 🥚 / -1 food -> +4 🥚``; off one it reads just ``+2 🥚``. Forest trades a
    card for a food, grassland a food for an egg, wetland an egg for a card."""
    if habitat is cards.Habitat.FOREST:
        count, reward, cost = board.gain_food_count(), "food", _CARD
    elif habitat is cards.Habitat.GRASSLAND:
        count, reward, cost = board.lay_eggs_count(), str(_EGG), "food"
    else:
        count, reward, cost = board.draw_cards_count(), str(_CARD), _EGG
    base = f"+{count} {reward}"
    if not board.action_offers_convert(habitat):
        return base
    return f"{base} / -1 {cost} -> +{count + 1} {reward}"


def _card_list_rows(title: str, birds: typing.Sequence[cards.Bird]) -> list[str]:
    """A titled list of birds (tray / hand) rendered in full, one per entry."""
    lines = [f"  {title}:"]
    if not birds:
        lines.append("      (empty)")
        return lines
    name_width = max(len(bird.name) for bird in birds)
    for bird in birds:
        block = format_bird_full(bird, indent="", name_width=name_width)
        lines.extend(_indent_block(block, "      "))
    return lines


def _bonus_rows(player: state.Player) -> list[str]:
    """The player's bonus cards, each with its current in-play value."""
    if not player.bonus_cards:
        return ["  bonus cards: (none)"]
    lines = ["  bonus cards:"]
    name_width = max(len(bc.name) for bc in player.bonus_cards)
    for bc in player.bonus_cards:
        lines.append(f"    {format_bonus(bc, name_width)}")
        lines.append(f"      {format_bonus_score_now(bc, player)}")
    return lines


def _indent_block(text: str, prefix: str) -> list[str]:
    """Split a possibly multi-line rendering and prefix every physical line,
    so a bird's stat line and its power line indent together."""
    return [prefix + line for line in text.split("\n")]


#### Bonus-card scoring text ####


def _bonus_detail(bc: cards.BonusCard) -> str:
    """The natural ``condition [joiner] payout`` text for a bonus card.

    A per-bird card joins with `` - `` (``... [forest] - 2VP each``); a tiered
    card joins with `` | `` (``... [forest] | (3-4) 4VP (5) 8VP``).
    """
    condition = _icons(bc.condition.strip().rstrip("."))
    payout = _natural_vp(bc.vp_text)
    if not payout:
        return condition
    if not condition:
        return payout
    joiner = " - " if _VP_PER_RE.search(bc.vp_text) else " | "
    return f"{condition}{joiner}{payout}"


def _natural_vp(vp_text: str) -> str:
    """Compact natural rendering of a printed bonus VP string ("" if blank)."""
    text = vp_text.strip()
    if not text:
        return ""
    if _VP_PER_RE.search(text):
        return text.replace("[point]", "VP").replace("per bird", "each")
    bands = [
        band for chunk in text.split(";") if (band := _natural_vp_band(chunk.strip()))
    ]
    return " ".join(bands)


def _natural_vp_band(chunk: str) -> str:
    """One ``count: payout`` clause as ``(range) NVP``; "" if unrecognised."""
    match = _VP_RANGE_RE.search(chunk)
    if match:
        return f"({match.group(1)}-{match.group(2)}) {match.group(3)}VP"
    match = _VP_PLUS_RE.search(chunk)
    if match:
        return f"({match.group(1)}+) {match.group(2)}VP"
    match = _VP_EXACT_RE.search(chunk)
    if match:
        return f"({match.group(1)}) {match.group(2)}VP"
    return ""


#### Bonus-card usefulness ####


class _BonusCatalogSummary(pydantic.BaseModel):
    """Catalog-wide facts about the type-counting bonus cards, tallied once
    from the bundled card data: the total number of core-set birds, and how
    many of them qualify for each bonus card (keyed by ``bonus.json`` card
    name). Used to annotate the pre-game help line with the share of the whole
    deck a card can draw on."""

    total_birds: int
    qualifying_by_bonus: dict[str, int]


def _bonus_setup_help(
    bc: cards.BonusCard,
    hand_birds: typing.Sequence[cards.Bird],
    tray_birds: typing.Sequence[cards.Bird],
    selected_hand_birds: typing.Sequence[cards.Bird],
) -> str | None:
    """Pre-game usefulness line for a type-counting card; ``None`` otherwise.

    The hand figure is ``selected-qualifying / total-qualifying`` — how many
    matching birds you've kept so far out of how many you could — followed by
    the count sitting in the display and the share of the whole core-set deck
    that qualifies (e.g. ``17% of all birds``), a hint at how readily you can
    find more during the game.
    """
    summary = _bonus_catalog_summary()
    qualifying_total = summary.qualifying_by_bonus.get(bc.name)
    if qualifying_total is None:
        return None  # dynamic card (eggs / hand size / board) — can't assess
    matching_in_hand = _qualifying(bc, hand_birds)
    selected_matching = _qualifying(bc, selected_hand_birds)
    in_display = _qualifying(bc, tray_birds)
    pct = round(100 * qualifying_total / summary.total_birds)
    return (
        f"{selected_matching}/{matching_in_hand} in hand, "
        f"{in_display} in the display, {pct}% of all birds"
    )


def _qualifying(bc: cards.BonusCard, birds: typing.Iterable[cards.Bird]) -> int:
    """How many of ``birds`` count toward ``bc`` by fixed bird property."""
    return sum(1 for bird in birds if bc.name in bird.bonus_categories)


@functools.cache
def _bonus_catalog_summary() -> _BonusCatalogSummary:
    """Tally, once from the bundled catalog, the total core-set bird count and
    how many birds qualify for each type-counting bonus card.

    Dynamic cards — counting eggs laid, end-game hand size, or board layout —
    name no birds, so they never become a key and get no pre-game help line.
    """
    birds, _, _ = cards.load_all()
    qualifying: dict[str, int] = {}
    for bird in birds:
        for bonus_name in bird.bonus_categories:
            qualifying[bonus_name] = qualifying.get(bonus_name, 0) + 1
    return _BonusCatalogSummary(total_birds=len(birds), qualifying_by_bonus=qualifying)
