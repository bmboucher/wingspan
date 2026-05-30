"""Card data loaders and parsers.

This module is responsible for turning the raw wingsearch JSON files in
``wingspan.data`` into the parsed card models defined in
:mod:`wingspan.cards.schema`. It owns two related responsibilities:

- :func:`load_all` reads the bundled JSON files, validates each core-set
  row into an input record (:class:`schema.BirdRecord`,
  :class:`schema.BonusRecord`, :class:`schema.GoalRecord`), and converts
  each record into its parsed counterpart by calling ``record.load()``.
- The various ``parse_*`` helpers turn individual raw fields (power text,
  food costs, habitat columns, VP thresholds, goal descriptions, ...)
  into typed values. These are called from the ``.load()`` methods on the
  input records.

Wingspan's printed power text uses a small inline-icon language
(``[seed]``, ``[bird]``, ``[forest]`` etc.). :func:`parse_power`
recognises the patterns we model and emits a structured ``Power``
consisting of one or more ``Effect`` records. Anything we don't
recognise becomes a single ``EffectKind.UNIMPLEMENTED`` effect so the
simulator can still run.
"""

from __future__ import annotations

import functools
import json
import re
from importlib import resources
from importlib.resources import abc

import pydantic

from wingspan.cards import schema

# ---------------------------------------------------------------------------
# Tag tables. Tag conventions used by wingsearch:
#   [seed]       seed icon
#   [invertebrate] [fish] [fruit] [rodent] [nectar] [wild]
#   [card]       bird card
#   [egg]        egg
#   [die]        birdfeeder die
#   [forest] [grassland] [wetland]
#   [bird] [bowl] [cavity] [ground] [platform]

FOOD_TAGS = {
    "[invertebrate]": schema.Food.INVERTEBRATE,
    "[seed]": schema.Food.SEED,
    "[fish]": schema.Food.FISH,
    "[fruit]": schema.Food.FRUIT,
    "[rodent]": schema.Food.RODENT,
}
HABITAT_TAGS = {
    "[forest]": schema.Habitat.FOREST,
    "[grassland]": schema.Habitat.GRASSLAND,
    "[wetland]": schema.Habitat.WETLAND,
}
NEST_TAGS = {
    "bowl": schema.NestType.BOWL,
    "cavity": schema.NestType.CAVITY,
    "ground": schema.NestType.GROUND,
    "platform": schema.NestType.PLATFORM,
}

_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

# A handful of fan-made bonus cards in the wingsearch data are mistakenly
# tagged ``"Set": "core"`` and prefixed with this marker in their name. They
# are not part of the published base game, so they are excluded at load time.
_FAN_MADE_PREFIX = "[Fan Made]"

# A few bonus cards are named differently in ``bonus.json`` than in the
# per-bird qualification columns of ``master.json`` — wingsearch sourced the
# two files from different printings, and Wingspan renamed several bonus cards
# between them. Map the bonus-card name to the ``master.json`` column that
# marks its qualifying birds so the lookup in ``bonus_categories_for_bird``
# resolves. Without this, "Omnivore Specialist" (column "Omnivore Expert")
# would tag zero birds and silently score 0 VP for the rest of the game.
_BONUS_COLUMN_ALIASES = {
    "Omnivore Specialist": "Omnivore Expert",
}


def parse_power(color: schema.PowerColor, text: str) -> schema.Power:
    """Best-effort parser. Recognises a small set of common patterns and
    returns ``UNIMPLEMENTED`` for everything else. Idempotent and safe to
    call once per bird at load time."""
    text = (text or "").strip()
    if not text:
        return schema.Power(color=color, effects=(), raw_text="")
    normalized = _normalize(text)
    effects = _extract_effects(normalized)
    if not effects:
        effects.append(
            schema.Effect(kind=schema.EffectKind.UNIMPLEMENTED, raw_text=text)
        )
    return schema.Power(color=color, effects=tuple(effects), raw_text=text)


###### PRIVATE #######

#### Text normalization ####


def _normalize(text: str) -> str:
    return text.replace("—", "-").replace("“", '"').replace("”", '"')


def _to_int(tok: str) -> int | None:
    if tok.isdigit():
        return int(tok)
    return _NUM_WORDS.get(tok.lower())


#### Pattern dispatch ####


def _extract_effects(text: str) -> list[schema.Effect]:
    """Apply each recognized pattern in turn, accumulating matched effects.

    Order matters: more specific patterns must run before less specific
    overlapping patterns (e.g. the "or"-disjunction birdfeeder pattern runs
    before the generic ``Gain N [food] from the birdfeeder``)."""
    effects: list[schema.Effect] = []
    for matcher in _PATTERN_MATCHERS:
        eff = matcher(text)
        if eff is not None:
            effects.append(eff)
    return effects


#### Pattern matchers ####

# Each matcher takes the normalized power text and returns an Effect (or None
# if the pattern does not match). They are independent — multiple may fire on
# the same text (no early return). Ordering is significant when patterns
# overlap (see _extract_effects).


def _m_gain_food_supply(text: str) -> schema.Effect | None:
    match = re.search(
        r"Gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply",
        text,
        re.I,
    )
    if match and match.group(2) in FOOD_TAGS:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.GAIN_FOOD_SUPPLY,
            amount=amount,
            food=FOOD_TAGS[match.group(2)],
            raw_text=match.group(0),
        )
    return None


def _m_gain_food_from_feeder_choice(text: str) -> schema.Effect | None:
    # "Gain 1 [foodA] or [foodB] from the birdfeeder" -- Indigo Bunting etc.
    # Matched before the more permissive birdfeeder pattern.
    match = re.search(
        r"Gain 1\s+(\[\w+\])\s+or\s+(\[\w+\])\s+from the birdfeeder",
        text,
        re.I,
    )
    if match and match.group(1) in FOOD_TAGS and match.group(2) in FOOD_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.GAIN_FOOD_FROM_FEEDER_CHOICE,
            amount=1,
            food_a=FOOD_TAGS[match.group(1)],
            food_b=FOOD_TAGS[match.group(2)],
            raw_text=match.group(0),
        )
    return None


def _m_gain_die_any(text: str) -> schema.Effect | None:
    # American Redstart. Anchored so it only matches the unqualified wording.
    match = re.match(r"^Gain 1 \[die\] from the birdfeeder\.?$", text, re.I)
    if match:
        return schema.Effect(
            kind=schema.EffectKind.GAIN_DIE_ANY,
            amount=1,
            raw_text=match.group(0),
        )
    return None


def _m_gain_food_birdfeeder(text: str) -> schema.Effect | None:
    match = re.search(
        r"Gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the birdfeeder",
        text,
        re.I,
    )
    if match and match.group(2) in FOOD_TAGS:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.GAIN_FOOD_BIRDFEEDER,
            amount=amount,
            food=FOOD_TAGS[match.group(2)],
            raw_text=match.group(0),
        )
    return None


def _m_lay_egg_on_this(text: str) -> schema.Effect | None:
    match = re.search(
        r"Lay\s+(\d+|a|an|one|two|three)\s+\[egg\] on this bird",
        text,
        re.I,
    )
    if match:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.LAY_EGG_ON_THIS,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


def _m_lay_egg_any(text: str) -> schema.Effect | None:
    match = re.search(
        r"Lay\s+(\d+|a|an|one|two|three)\s+\[egg\] on any bird",
        text,
        re.I,
    )
    if match:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.LAY_EGG_ANY,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


def _m_draw_cards(text: str) -> schema.Effect | None:
    # Excludes "All players draw" (handled separately).
    if re.search(r"All players draw", text, re.I):
        return None
    match = re.search(r"Draw\s+(\d+|a|an|one|two|three)\s+\[card\]", text, re.I)
    if match:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.DRAW_CARDS,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


def _m_cache_food(text: str) -> schema.Effect | None:
    match = re.search(
        r"Cache\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply on this bird",
        text,
        re.I,
    )
    if match and match.group(2) in FOOD_TAGS:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.CACHE_FOOD,
            amount=amount,
            food=FOOD_TAGS[match.group(2)],
            raw_text=match.group(0),
        )
    return None


def _m_tuck_from_hand(text: str) -> schema.Effect | None:
    match = re.search(
        r"Tuck\s+(\d+|a|an|one|two|three)\s+\[card\] from your hand behind this",
        text,
        re.I,
    )
    if match:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.TUCK_FROM_HAND,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


def _m_play_additional_bird(text: str) -> schema.Effect | None:
    match = re.search(r"Play an additional bird in your (\[\w+\])", text, re.I)
    if match and match.group(1) in HABITAT_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.PLAY_ADDITIONAL_BIRD,
            habitat=HABITAT_TAGS[match.group(1)],
            raw_text=match.group(0),
        )
    return None


def _m_all_players_gain_food(text: str) -> schema.Effect | None:
    match = re.search(
        r"All players gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply",
        text,
        re.I,
    )
    if match and match.group(2) in FOOD_TAGS:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.ALL_PLAYERS_GAIN_FOOD,
            amount=amount,
            food=FOOD_TAGS[match.group(2)],
            raw_text=match.group(0),
        )
    return None


def _m_all_players_draw(text: str) -> schema.Effect | None:
    match = re.search(
        r"All players draw\s+(\d+|a|an|one|two|three)\s+\[card\]",
        text,
        re.I,
    )
    if match:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.ALL_PLAYERS_DRAW,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


def _m_draw_bonus(text: str) -> schema.Effect | None:
    match = re.search(
        r"Draw\s+(\d+|a|an|one|two|three)\s+bonus cards",
        text,
        re.I,
    )
    if match:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.DRAW_BONUS,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


def _m_discard_egg_for_wild(text: str) -> schema.Effect | None:
    match = re.search(
        r"Discard 1 \[egg\] from any of your other birds to gain"
        r"\s+(\d+|a|an|one|two|three)\s+\[wild\] from the supply",
        text,
        re.I,
    )
    if match:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.DISCARD_EGG_FOR_WILD,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


def _m_each_player_gains_die(text: str) -> schema.Effect | None:
    # Anna's / Ruby-Throated Hummingbird.
    match = re.search(
        r"Each player gains\s+(\d+|a|an|one|two|three)\s+\[die\]"
        r"\s+from the birdfeeder, starting with the player of your choice",
        text,
        re.I,
    )
    if match:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


def _m_all_players_lay_egg_on_nest(text: str) -> schema.Effect | None:
    # "All players lay 1 [egg] on any 1 [<nest>] bird.
    #  (You may lay 1 [egg] on 1 additional [<nest>] bird.)"
    # ``amount`` encodes the optional second-sentence extra for the active
    # player (0 when absent, 1 when present).
    match = re.search(
        r"All players lay\s+\d+\s+\[egg\]\s+on any\s+\d+\s+\[(bowl|cavity|ground|platform)\] bird\."
        r"(?:\s+You may lay\s+\d+\s+\[egg\]\s+on\s+\d+\s+additional)?",
        text,
        re.I,
    )
    if match:
        nest = NEST_TAGS[match.group(1).lower()]
        extra_for_self = 1 if "additional" in match.group(0).lower() else 0
        return schema.Effect(
            kind=schema.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST,
            nest=nest,
            amount=extra_for_self,
            raw_text=match.group(0),
        )
    return None


def _m_draw_from_tray_all(text: str) -> schema.Effect | None:
    # Brant.
    match = re.search(r"Draw the (\d+) face-up \[card\] in the bird tray", text, re.I)
    if match:
        amount = int(match.group(1))
        return schema.Effect(
            kind=schema.EffectKind.DRAW_FROM_TRAY_ALL,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


def _m_trade_wild(text: str) -> schema.Effect | None:
    # Green Heron.
    match = re.search(
        r"Trade 1 \[wild\] for any other type from the supply", text, re.I
    )
    if match:
        return schema.Effect(
            kind=schema.EffectKind.TRADE_WILD_FOOD,
            raw_text=match.group(0),
        )
    return None


def _m_fewest_forest_gains_die(text: str) -> schema.Effect | None:
    # Hermit Thrush.
    match = re.search(
        r"Player\(s\) with the fewest birds in their \[forest\] gain 1 \[die\] from birdfeeder",
        text,
        re.I,
    )
    if match:
        return schema.Effect(
            kind=schema.EffectKind.FEWEST_FOREST_GAINS_DIE,
            raw_text=match.group(0),
        )
    return None


def _m_play_additional_bird_here(text: str) -> schema.Effect | None:
    # House Wren.
    match = re.search(
        r"Play an additional bird in this bird.{1,4}s habitat", text, re.I
    )
    if match:
        return schema.Effect(
            kind=schema.EffectKind.PLAY_ADDITIONAL_BIRD_HERE,
            raw_text=match.group(0),
        )
    return None


def _m_draw_n_plus_one_draft(text: str) -> schema.Effect | None:
    # American Oystercatcher.
    match = re.search(r"Draw \[card\] equal to the number of players \+1", text, re.I)
    if match:
        return schema.Effect(
            kind=schema.EffectKind.DRAW_N_PLUS_ONE_DRAFT,
            raw_text=match.group(0),
        )
    return None


def _m_draw_bonus_keep(text: str) -> schema.Effect | None:
    match = re.search(
        r"Draw\s+(\d+|a|an|one|two|three)\s+new bonus cards and keep\s+(\d+|a|an|one|two|three)",
        text,
        re.I,
    )
    if match:
        amount = _to_int(match.group(1)) or 1
        keep = _to_int(match.group(2)) or 1
        return schema.Effect(
            kind=schema.EffectKind.DRAW_BONUS_KEEP,
            amount=amount,
            keep_count=keep,
            raw_text=match.group(0),
        )
    return None


def _m_lay_egg_all_nest(text: str) -> schema.Effect | None:
    # Bobolink etc.
    match = re.search(
        r"Lay\s+(\d+|a|an|one|two|three)\s+\[egg\]"
        r" on each of your birds with a \[(bowl|cavity|ground|platform)\] nest",
        text,
        re.I,
    )
    if match:
        amount = _to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.LAY_EGG_ALL_NEST,
            amount=amount,
            nest=NEST_TAGS[match.group(2).lower()],
            raw_text=match.group(0),
        )
    return None


def _m_gain_all_food_feeder(text: str) -> schema.Effect | None:
    # Bald Eagle, Northern Flicker.
    match = re.search(r"Gain all (\[\w+\]) that are in the birdfeeder", text, re.I)
    if match and match.group(1) in FOOD_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.GAIN_ALL_FOOD_FEEDER,
            food=FOOD_TAGS[match.group(1)],
            raw_text=match.group(0),
        )
    return None


def _m_tuck_from_deck_paid(text: str) -> schema.Effect | None:
    match = re.search(
        r"Discard 1 (\[\w+\]) to tuck\s+(\d+|a|an|one|two|three)\s+\[card\] from the deck behind this bird",
        text,
        re.I,
    )
    if match and match.group(1) in FOOD_TAGS:
        amount = _to_int(match.group(2)) or 1
        return schema.Effect(
            kind=schema.EffectKind.TUCK_FROM_DECK_PAID,
            amount=amount,
            food=FOOD_TAGS[match.group(1)],
            raw_text=match.group(0),
        )
    return None


def _m_predator_hunt(text: str) -> schema.Effect | None:
    match = re.search(
        r"Look at a \[card\] from the deck\. If less than\s+(\d+)\s*cm,"
        r" tuck it behind this bird\. If not, discard it",
        text,
        re.I,
    )
    if match:
        return schema.Effect(
            kind=schema.EffectKind.PREDATOR_HUNT,
            max_wingspan_cm=int(match.group(1)),
            raw_text=match.group(0),
        )
    return None


def _m_move_rightmost(text: str) -> schema.Effect | None:
    match = re.search(
        r"If this bird is to the right of all other birds in its habitat,"
        r" move it to another habitat",
        text,
        re.I,
    )
    if match:
        return schema.Effect(
            kind=schema.EffectKind.MOVE_BIRD_IF_RIGHTMOST,
            raw_text=match.group(0),
        )
    return None


def _m_repeat_brown(text: str) -> schema.Effect | None:
    # Gray Catbird, Northern Mockingbird.
    match = re.search(
        r"Repeat a brown power on another bird in this habitat", text, re.I
    )
    if match:
        return schema.Effect(
            kind=schema.EffectKind.REPEAT_BROWN_POWER,
            raw_text=match.group(0),
        )
    return None


def _m_repeat_predator(text: str) -> schema.Effect | None:
    # Hooded Merganser.
    match = re.search(r"Repeat 1 \[predator\] power in this habitat", text, re.I)
    if match:
        return schema.Effect(
            kind=schema.EffectKind.REPEAT_PREDATOR_POWER,
            raw_text=match.group(0),
        )
    return None


def _m_pink_lay_egg_on_nest(text: str) -> schema.Effect | None:
    match = re.search(
        r"When another player takes the .lay eggs. action,"
        r"\s*lay 1 \[egg\] on \w+ bird with a \[(bowl|cavity|ground|platform)\] nest",
        text,
        re.I,
    )
    if match:
        return schema.Effect(
            kind=schema.EffectKind.PINK_LAY_EGG_ON_NEST,
            nest=NEST_TAGS[match.group(1).lower()],
            raw_text=match.group(0),
        )
    return None


def _m_pink_predator_feeder(text: str) -> schema.Effect | None:
    match = re.search(
        r"When another player's \[predator\] succeeds, gain 1 \[die\] from the birdfeeder",
        text,
        re.I,
    )
    if match:
        return schema.Effect(
            kind=schema.EffectKind.PINK_PREDATOR_FEEDER,
            raw_text=match.group(0),
        )
    return None


_PATTERN_MATCHERS = (
    _m_gain_food_supply,
    _m_gain_food_from_feeder_choice,
    _m_gain_die_any,
    _m_gain_food_birdfeeder,
    _m_lay_egg_on_this,
    _m_lay_egg_any,
    _m_draw_cards,
    _m_cache_food,
    _m_tuck_from_hand,
    _m_play_additional_bird,
    _m_all_players_gain_food,
    _m_all_players_draw,
    _m_draw_bonus,
    _m_discard_egg_for_wild,
    _m_each_player_gains_die,
    _m_all_players_lay_egg_on_nest,
    _m_draw_from_tray_all,
    _m_trade_wild,
    _m_fewest_forest_gains_die,
    _m_play_additional_bird_here,
    _m_draw_n_plus_one_draft,
    _m_draw_bonus_keep,
    _m_lay_egg_all_nest,
    _m_gain_all_food_feeder,
    _m_tuck_from_deck_paid,
    _m_predator_hunt,
    _m_move_rightmost,
    _m_repeat_brown,
    _m_repeat_predator,
    _m_pink_lay_egg_on_nest,
    _m_pink_predator_feeder,
)


# ---------------------------------------------------------------------------
# Top-level loader


def load_all() -> (
    tuple[list[schema.Bird], list[schema.BonusCard], list[schema.EndRoundGoal]]
):
    """Read every core-set bird, bonus card, and end-of-round goal from the
    bundled JSON data. Returns three parallel lists in source order."""
    base = resources.files("wingspan.data")
    bird_records = _load_core_records(base / "master.json", schema.BirdRecord)
    bonus_records = [
        bonus_record
        for bonus_record in _load_core_records(base / "bonus.json", schema.BonusRecord)
        if not bonus_record.bonus_card.startswith(_FAN_MADE_PREFIX)
    ]
    goal_records = _load_core_records(base / "goals.json", schema.GoalRecord)
    birds = [
        bird
        for bird in (record.load(bonus_records) for record in bird_records)
        if bird is not None
    ]
    bonuses = [bonus_record.load() for bonus_record in bonus_records]
    goals = [goal_record.load() for goal_record in goal_records]
    return birds, bonuses, goals


def power_coverage(birds: list[schema.Bird]) -> tuple[int, int]:
    """Return ``(implemented, total)``. Birds with no power text are counted
    as implemented (there is nothing to model)."""
    impl = sum(
        1
        for bird in birds
        if not any(
            effect.kind == schema.EffectKind.UNIMPLEMENTED
            for effect in bird.power.effects
        )
    )
    return impl, len(birds)


# ---------------------------------------------------------------------------
# Canonical catalog index (stable card -> dense-index maps for RL encoding)
#
# The RL encoder represents a card as a one-hot over all core-set cards, and a
# *set* of cards (a hand, a kept-set) as a multi-hot. That needs a stable dense
# index per card, independent of any single game's shuffle. These helpers
# derive one from the full loaded catalog and cache it, so the first call parses
# the bundled JSON once and every later lookup is a dict hit. Keyed on the card
# id so a trained per-card embedding stays aligned across runs.


@functools.lru_cache(maxsize=1)
def _canonical_cards() -> tuple[tuple[schema.Bird, ...], tuple[schema.BonusCard, ...]]:
    birds, bonuses, _ = load_all()
    return tuple(birds), tuple(bonuses)


@functools.lru_cache(maxsize=1)
def _bird_index_by_id() -> dict[int, int]:
    birds, _ = _canonical_cards()
    return {bird.id: i for i, bird in enumerate(birds)}


@functools.lru_cache(maxsize=1)
def _bonus_index_by_id() -> dict[int, int]:
    _, bonuses = _canonical_cards()
    return {bonus_card.id: i for i, bonus_card in enumerate(bonuses)}


def n_birds() -> int:
    """Number of distinct core-set birds — the length of the bird-identity
    one-hot (and the kept-set / hand multi-hot) stripe in the RL encoder."""
    return len(_canonical_cards()[0])


def n_bonus_cards() -> int:
    """Number of distinct core-set bonus cards — the length of the bonus-card
    identity one-hot stripe in the RL encoder."""
    return len(_canonical_cards()[1])


def bird_index(bird: schema.Bird) -> int:
    """Stable dense index of ``bird`` in the core-set catalog, used for the
    bird-identity one-hot. Keyed on the card id, so it is identical across
    games and a trained per-card embedding stays aligned."""
    return _bird_index_by_id()[bird.id]


def bonus_index(bonus_card: schema.BonusCard) -> int:
    """Stable dense index of ``bonus_card`` in the core-set catalog, used for
    the bonus-card identity one-hot."""
    return _bonus_index_by_id()[bonus_card.id]


# ---------------------------------------------------------------------------
# Record-field parsers (called from the ``.load()`` methods on
# ``schema.BirdRecord`` / ``schema.BonusRecord`` / ``schema.GoalRecord``)


def parse_power_color(raw: str | None) -> schema.PowerColor:
    """Map a raw color string (e.g. ``"brown"``) to a ``PowerColor`` enum,
    defaulting to ``NONE`` for unknown or missing values."""
    lowered = (raw or "none").lower()
    try:
        return schema.PowerColor(lowered)
    except ValueError:
        return schema.PowerColor.NONE


def parse_habitats(record: schema.BirdRecord) -> list[schema.Habitat]:
    """Return the habitats the bird may live in, in canonical order."""
    out: list[schema.Habitat] = []
    for habitat, marker in [
        (schema.Habitat.FOREST, record.forest),
        (schema.Habitat.GRASSLAND, record.grassland),
        (schema.Habitat.WETLAND, record.wetland),
    ]:
        if marker == "X":
            out.append(habitat)
    return out


def parse_food_cost(record: schema.BirdRecord) -> schema.BirdCost:
    """Return the :class:`schema.BirdCost` printed on a bird record."""
    vec: list[int] = [0] * schema.N_FOODS
    for amount, food in [
        (record.invertebrate, schema.Food.INVERTEBRATE),
        (record.seed, schema.Food.SEED),
        (record.fish, schema.Food.FISH),
        (record.fruit, schema.Food.FRUIT),
        (record.rodent, schema.Food.RODENT),
    ]:
        if amount is not None and amount > 0:
            vec[schema.food_index(food)] = int(amount)
    wild = record.wild_food
    wild_n = int(wild) if wild is not None and wild > 0 else 0
    return schema.BirdCost(counts=(vec[0], vec[1], vec[2], vec[3], vec[4], wild_n))


def parse_nest(raw: str | None) -> schema.NestType:
    """Map a raw nest-type string to a :class:`NestType` enum."""
    if not raw:
        return schema.NestType.NONE
    normalized = raw.lower().strip()
    for nest_type in schema.NestType:
        if nest_type.value == normalized:
            return nest_type
    if normalized == "wild":
        return schema.NestType.STAR
    return schema.NestType.NONE


def bonus_categories_for_bird(
    record: schema.BirdRecord, bonuses: list[schema.BonusRecord]
) -> tuple[str, ...]:
    """Return the names of all core-set bonus cards whose category column
    is marked ``"X"`` on this bird record. Bonus-card column names are
    dynamic (one per bonus card) so they live in :attr:`model_extra`; a
    card renamed between the two source files is resolved through
    :data:`_BONUS_COLUMN_ALIASES`. The returned name is always the
    ``bonus.json`` card name, so it matches ``BonusCard.name`` downstream."""
    out: list[str] = []
    extras = record.model_extra or {}
    for bonus in bonuses:
        if bonus.card_set != "core":
            continue
        column = _BONUS_COLUMN_ALIASES.get(bonus.bonus_card, bonus.bonus_card)
        if extras.get(column) == "X":
            out.append(bonus.bonus_card)
    return tuple(out)


def parse_bonus_per_bird(vp_text: str) -> int | None:
    """Parse a per-bird payout like ``'2[point] per bird'`` into the VP each
    qualifying bird earns, or ``None`` for a tiered card.

    Per-bird and tiered payouts are mutually exclusive in the core set; a
    tiered string (``'... birds: N[point]'``) never matches this pattern."""
    match = re.search(r"(\d+)\s*\[point\]\s*per bird", vp_text, re.I)
    return int(match.group(1)) if match else None


def parse_bonus_thresholds(vp_text: str) -> tuple[tuple[int, int], ...]:
    """Parse strings like ``'2 to 3 birds: 3[point]; 4+ birds: 7[point]'``
    into ascending ``(min_count, vp)`` pairs. Per-bird cards (handled by
    :func:`parse_bonus_per_bird`) yield no thresholds."""
    out: list[tuple[int, int]] = []
    for chunk in vp_text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        match = re.match(r"(\d+)\s*\+\s*birds?\s*:\s*(\d+)", chunk, re.I)
        if match:
            out.append((int(match.group(1)), int(match.group(2))))
            continue
        match = re.match(r"(\d+)\s*to\s*(\d+)\s*birds?\s*:\s*(\d+)", chunk, re.I)
        if match:
            out.append((int(match.group(1)), int(match.group(3))))
            continue
        match = re.match(r"(\d+)\s*birds?\s*:\s*(\d+)", chunk, re.I)
        if match:
            out.append((int(match.group(1)), int(match.group(2))))
    out.sort(key=lambda pair: pair[0])
    return tuple(out)


# Exact-match table from a goal's raw description to the scoring-engine
# tag it dispatches on. Keys are the verbatim ``"Goal"`` strings from
# ``goals.json`` for the core set.
_GOAL_CATEGORIES: dict[str, str] = {
    "[bird] in [forest]": "birds_forest",
    "[bird] in [grassland]": "birds_grassland",
    "[bird] in [wetland]": "birds_wetland",
    "[egg] in [forest]": "eggs_forest",
    "[egg] in [grassland]": "eggs_grassland",
    "[egg] in [wetland]": "eggs_wetland",
    "[egg] in [bowl]": "eggs_bowl",
    "[egg] in [cavity]": "eggs_cavity",
    "[egg] in [ground]": "eggs_ground",
    "[egg] in [platform]": "eggs_platform",
    "[bowl] [bird] with [egg]": "bowl_birds_with_eggs",
    "[cavity] [bird] with [egg]": "cavity_birds_with_eggs",
    "[ground] [bird] with [egg]": "ground_birds_with_eggs",
    "[platform] [bird] with [egg]": "platform_birds_with_eggs",
}


def goal_category(desc: str) -> str:
    """Look up the scoring-engine tag for a goal description.
    Unknown descriptions return a synthetic ``"unknown:..."`` tag that
    scoring treats as zero points."""
    return _GOAL_CATEGORIES.get(desc, "unknown:" + desc[:30].lower())


###### PRIVATE #######


def _load_core_records[R: pydantic.BaseModel](
    path: abc.Traversable, model: type[R]
) -> list[R]:
    """Load and validate only the core-set rows from a wingsearch JSON file.

    Non-core rows are skipped before validation: other expansions use values
    (e.g. ``"Wingspan": "*"`` for variable-wingspan birds) that fall outside
    the core-set schema and would otherwise fail validation here."""
    return [
        model.model_validate(row)
        for row in json.loads(path.read_text(encoding="utf-8"))
        if row.get("Set") == "core"
    ]
