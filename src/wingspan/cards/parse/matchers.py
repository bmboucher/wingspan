# pyright: reportUnusedFunction=false
# (every function here is a power matcher registered via @registry.pattern /
# @registry.pink_pattern; none is called by name, so the unused check is wrong)
"""General (when-played / when-activated) bird power-text matchers.

Each matcher takes the normalized power text and returns an ``Effect`` (or
``None`` if it does not match). They are independent — several may fire on the
same text. Registration order (source order here) is the trial order; keep
more-specific patterns above overlapping generic ones.
"""

from __future__ import annotations

import re

from wingspan.cards import schema
from wingspan.cards.parse import registry, tags

# ---------------------------------------------------------------------------
# Helpers for "Tuck N from hand. If you do, [X]." conditional patterns

# Regex that matches any covered tuck-conditional consequence. Used to suppress
# the standalone tuck-from-hand matcher when a combined matcher already handles
# the whole pattern.
_TUCK_IF_DO_COVERED = re.compile(
    r"from your hand behind this bird\.\s+If you do,\s+"
    r"(?:"
    r"draw\s+(?:\d+|a|an|one|two|three)\s+\[card\]"
    r"|you may also lay\s+(?:\d+|a|an|one|two|three)\s+\[egg\] on this bird"
    r"|lay\s+(?:\d+|a|an|one|two|three)\s+\[egg\] on any bird"
    r"|gain\s+(?:\d+|a|an|one|two|three)\s+\[\w+\]\s+from the supply(?!\s+or)"
    r")",
    re.I,
)

# Regex that matches the "Roll all dice not in birdfeeder … cache" pattern.
# Used to suppress the standalone _m_cache_food matcher when the more specific
# roll-conditional matcher already handles the whole pattern.
_ROLL_NOT_IN_FEEDER_COVERED = re.compile(
    r"Roll all dice not in birdfeeder\.\s+If any are \[\w+\],",
    re.I,
)


def _is_if_you_do_consequence(text: str, match_start: int) -> bool:
    """True when the matched clause at ``match_start`` is the consequence of an
    'If you do,' clause that appears earlier in the text."""
    return re.search(r"if you do,", text[:match_start], re.I) is not None


# ---------------------------------------------------------------------------
# Combined tuck-conditional matchers (registered before the individual-clause
# matchers they supersede so they get priority in the trial order).


@registry.pattern
def _m_tuck_then_draw(text: str) -> schema.Effect | None:
    """'Tuck N from hand. If you do, draw N [card].' — tuck first, draw only on success."""
    match = re.search(
        r"Tuck\s+(\d+|a|an|one|two|three)\s+\[card\] from your hand behind this bird\."
        r"\s+If you do, draw\s+(\d+|a|an|one|two|three)\s+\[card\]",
        text,
        re.I,
    )
    if match:
        return schema.Effect(
            kind=schema.EffectKind.TUCK_FROM_HAND_THEN_DRAW,
            amount=tags.to_int(match.group(1)) or 1,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_tuck_then_lay_on_this(text: str) -> schema.Effect | None:
    """'Tuck N from hand. If you do, you may also lay N [egg] on this bird.' — optional lay on success."""
    match = re.search(
        r"Tuck\s+(?:\d+|a|an|one|two|three)\s+\[card\] from your hand behind this bird\."
        r"\s+If you do, you may also lay\s+(\d+|a|an|one|two|three)\s+\[egg\] on this bird",
        text,
        re.I,
    )
    if match:
        return schema.Effect(
            kind=schema.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS,
            amount=tags.to_int(match.group(1)) or 1,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_tuck_then_lay_any(text: str) -> schema.Effect | None:
    """'Tuck N from hand. If you do, lay N [egg] on any bird.' — lay any on success."""
    match = re.search(
        r"Tuck\s+(?:\d+|a|an|one|two|three)\s+\[card\] from your hand behind this bird\."
        r"\s+If you do, lay\s+(\d+|a|an|one|two|three)\s+\[egg\] on any bird",
        text,
        re.I,
    )
    if match:
        return schema.Effect(
            kind=schema.EffectKind.TUCK_FROM_HAND_THEN_LAY_ANY,
            amount=tags.to_int(match.group(1)) or 1,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_tuck_then_gain_food_supply(text: str) -> schema.Effect | None:
    """'Tuck N from hand. If you do, gain N [food] from the supply.' — food gain on success."""
    match = re.search(
        r"Tuck\s+(?:\d+|a|an|one|two|three)\s+\[card\] from your hand behind this bird\."
        r"\s+If you do, gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply(?!\s+or)",
        text,
        re.I,
    )
    if match and match.group(2) in tags.FOOD_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY,
            amount=tags.to_int(match.group(1)) or 1,
            food=tags.FOOD_TAGS[match.group(2)],
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_gain_food_supply(text: str) -> schema.Effect | None:
    match = re.search(
        r"Gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply",
        text,
        re.I,
    )
    if match and match.group(2) in tags.FOOD_TAGS:
        if _is_if_you_do_consequence(text, match.start()):
            return None
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.GAIN_FOOD_SUPPLY,
            amount=amount,
            food=tags.FOOD_TAGS[match.group(2)],
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_gain_food_from_feeder_choice(text: str) -> schema.Effect | None:
    # "Gain 1 [foodA] or [foodB] from the birdfeeder" -- Indigo Bunting etc.
    # Matched before the more permissive birdfeeder pattern.
    match = re.search(
        r"Gain 1\s+(\[\w+\])\s+or\s+(\[\w+\])\s+from the birdfeeder",
        text,
        re.I,
    )
    if match and match.group(1) in tags.FOOD_TAGS and match.group(2) in tags.FOOD_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.GAIN_FOOD_FROM_FEEDER_CHOICE,
            amount=1,
            food_a=tags.FOOD_TAGS[match.group(1)],
            food_b=tags.FOOD_TAGS[match.group(2)],
            raw_text=match.group(0),
        )
    return None


@registry.pattern
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


@registry.pattern
def _m_gain_food_birdfeeder(text: str) -> schema.Effect | None:
    match = re.search(
        r"Gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the birdfeeder",
        text,
        re.I,
    )
    if match and match.group(2) in tags.FOOD_TAGS:
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.GAIN_FOOD_BIRDFEEDER,
            amount=amount,
            food=tags.FOOD_TAGS[match.group(2)],
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_lay_egg_on_this(text: str) -> schema.Effect | None:
    match = re.search(
        r"Lay\s+(\d+|a|an|one|two|three)\s+\[egg\] on this bird",
        text,
        re.I,
    )
    if match:
        if _is_if_you_do_consequence(text, match.start()):
            return None
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.LAY_EGG_ON_THIS,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_lay_egg_any(text: str) -> schema.Effect | None:
    match = re.search(
        r"Lay\s+(\d+|a|an|one|two|three)\s+\[egg\] on any bird",
        text,
        re.I,
    )
    if match:
        if _is_if_you_do_consequence(text, match.start()):
            return None
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.LAY_EGG_ANY,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_draw_cards(text: str) -> schema.Effect | None:
    # Excludes "All players draw" (handled separately).
    if re.search(r"All players draw", text, re.I):
        return None
    match = re.search(r"Draw\s+(\d+|a|an|one|two|three)\s+\[card\]", text, re.I)
    if match:
        if _is_if_you_do_consequence(text, match.start()):
            return None
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.DRAW_CARDS,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_roll_not_in_feeder_cache(text: str) -> schema.Effect | None:
    """'Roll all dice not in birdfeeder. If any are [food], cache N [food].'

    Rolls the dice currently outside the feeder and caches only when the target
    food face appears. Registered before _m_cache_food so the full conditional
    pattern takes priority over the bare cache clause."""
    match = re.search(
        r"Roll all dice not in birdfeeder\.\s+If any are (\[\w+\]),"
        r"\s+cache\s+(\d+|a|an|one|two|three)\s+\1\s+from the supply on this bird",
        text,
        re.I,
    )
    if match and match.group(1) in tags.FOOD_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.ROLL_NOT_IN_FEEDER_CACHE,
            amount=tags.to_int(match.group(2)) or 1,
            food=tags.FOOD_TAGS[match.group(1)],
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_cache_food(text: str) -> schema.Effect | None:
    # Skip when the roll-conditional combined matcher already handles this text.
    if _ROLL_NOT_IN_FEEDER_COVERED.search(text):
        return None
    match = re.search(
        r"Cache\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply on this bird",
        text,
        re.I,
    )
    if match and match.group(2) in tags.FOOD_TAGS:
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.CACHE_FOOD,
            amount=amount,
            food=tags.FOOD_TAGS[match.group(2)],
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_tuck_from_hand(text: str) -> schema.Effect | None:
    # Skip patterns handled by a combined tuck-conditional matcher.
    if _TUCK_IF_DO_COVERED.search(text):
        return None
    match = re.search(
        r"Tuck\s+(\d+|a|an|one|two|three)\s+\[card\] from your hand behind this",
        text,
        re.I,
    )
    if match:
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.TUCK_FROM_HAND,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_play_additional_bird(text: str) -> schema.Effect | None:
    match = re.search(r"Play an additional bird in your (\[\w+\])", text, re.I)
    if match and match.group(1) in tags.HABITAT_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.PLAY_ADDITIONAL_BIRD,
            habitat=tags.HABITAT_TAGS[match.group(1)],
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_all_players_gain_food(text: str) -> schema.Effect | None:
    match = re.search(
        r"All players gain\s+(\d+|a|an|one|two|three)\s+(\[\w+\])\s+from the supply",
        text,
        re.I,
    )
    if match and match.group(2) in tags.FOOD_TAGS:
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.ALL_PLAYERS_GAIN_FOOD,
            amount=amount,
            food=tags.FOOD_TAGS[match.group(2)],
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_all_players_draw(text: str) -> schema.Effect | None:
    match = re.search(
        r"All players draw\s+(\d+|a|an|one|two|three)\s+\[card\]",
        text,
        re.I,
    )
    if match:
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.ALL_PLAYERS_DRAW,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_draw_bonus(text: str) -> schema.Effect | None:
    match = re.search(
        r"Draw\s+(\d+|a|an|one|two|three)\s+bonus cards",
        text,
        re.I,
    )
    if match:
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.DRAW_BONUS,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_discard_egg_for_wild(text: str) -> schema.Effect | None:
    match = re.search(
        r"Discard 1 \[egg\] from any of your other birds to gain"
        r"\s+(\d+|a|an|one|two|three)\s+\[wild\] from the supply",
        text,
        re.I,
    )
    if match:
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.DISCARD_EGG_FOR_WILD,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_each_player_gains_die(text: str) -> schema.Effect | None:
    # Anna's / Ruby-Throated Hummingbird.
    match = re.search(
        r"Each player gains\s+(\d+|a|an|one|two|three)\s+\[die\]"
        r"\s+from the birdfeeder, starting with the player of your choice",
        text,
        re.I,
    )
    if match:
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER,
            amount=amount,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
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
        nest = tags.NEST_TAGS[match.group(1).lower()]
        extra_for_self = 1 if "additional" in match.group(0).lower() else 0
        return schema.Effect(
            kind=schema.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST,
            nest=nest,
            amount=extra_for_self,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
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


@registry.pattern
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


@registry.pattern
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


@registry.pattern
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


@registry.pattern
def _m_draw_n_plus_one_draft(text: str) -> schema.Effect | None:
    # American Oystercatcher.
    match = re.search(r"Draw \[card\] equal to the number of players \+1", text, re.I)
    if match:
        return schema.Effect(
            kind=schema.EffectKind.DRAW_N_PLUS_ONE_DRAFT,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_draw_bonus_keep(text: str) -> schema.Effect | None:
    match = re.search(
        r"Draw\s+(\d+|a|an|one|two|three)\s+new bonus cards and keep\s+(\d+|a|an|one|two|three)",
        text,
        re.I,
    )
    if match:
        amount = tags.to_int(match.group(1)) or 1
        keep = tags.to_int(match.group(2)) or 1
        return schema.Effect(
            kind=schema.EffectKind.DRAW_BONUS_KEEP,
            amount=amount,
            keep_count=keep,
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_lay_egg_all_nest(text: str) -> schema.Effect | None:
    # Bobolink etc.
    match = re.search(
        r"Lay\s+(\d+|a|an|one|two|three)\s+\[egg\]"
        r" on each of your birds with a \[(bowl|cavity|ground|platform)\] nest",
        text,
        re.I,
    )
    if match:
        amount = tags.to_int(match.group(1)) or 1
        return schema.Effect(
            kind=schema.EffectKind.LAY_EGG_ALL_NEST,
            amount=amount,
            nest=tags.NEST_TAGS[match.group(2).lower()],
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_gain_all_food_feeder(text: str) -> schema.Effect | None:
    # Bald Eagle, Northern Flicker.
    match = re.search(r"Gain all (\[\w+\]) that are in the birdfeeder", text, re.I)
    if match and match.group(1) in tags.FOOD_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.GAIN_ALL_FOOD_FEEDER,
            food=tags.FOOD_TAGS[match.group(1)],
            raw_text=match.group(0),
        )
    return None


@registry.pattern
def _m_tuck_from_deck_paid(text: str) -> schema.Effect | None:
    match = re.search(
        r"Discard 1 (\[\w+\]) to tuck\s+(\d+|a|an|one|two|three)\s+\[card\] from the deck behind this bird",
        text,
        re.I,
    )
    if match and match.group(1) in tags.FOOD_TAGS:
        amount = tags.to_int(match.group(2)) or 1
        return schema.Effect(
            kind=schema.EffectKind.TUCK_FROM_DECK_PAID,
            amount=amount,
            food=tags.FOOD_TAGS[match.group(1)],
            raw_text=match.group(0),
        )
    return None


@registry.pattern
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


@registry.pattern
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


@registry.pattern
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


@registry.pattern
def _m_repeat_predator(text: str) -> schema.Effect | None:
    # Hooded Merganser.
    match = re.search(r"Repeat 1 \[predator\] power in this habitat", text, re.I)
    if match:
        return schema.Effect(
            kind=schema.EffectKind.REPEAT_PREDATOR_POWER,
            raw_text=match.group(0),
        )
    return None
