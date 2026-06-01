# pyright: reportUnusedFunction=false
# (every function here is a power matcher registered via @registry.pattern /
# @registry.pink_pattern; none is called by name, so the unused check is wrong)
"""Pink (between-turn reactor) bird power-text matchers.

These match a reactive power's consequent clause. The first two also belong
to the general pattern list (their printed text is matched both when scanning
a normal power and when scanning a pink one), so they carry both decorators.
"""

from __future__ import annotations

import re

from wingspan.cards import schema
from wingspan.cards.parse import registry, tags


@registry.pattern
@registry.pink_pattern
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
            nest=tags.NEST_TAGS[match.group(1).lower()],
            raw_text=match.group(0),
        )
    return None


@registry.pattern
@registry.pink_pattern
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


@registry.pink_pattern
def _m_pink_play_bird_gain(text: str) -> schema.Effect | None:
    # Belted Kingfisher / Eastern Kingbird: "When another player plays a bird in
    # their [habitat], gain N [food] from the supply."
    match = re.search(
        r"plays a bird in (?:their )?(\[\w+\]).*?gain\s+(\d+|a|an|one|two|three)\s+"
        r"(\[\w+\])\s+from the supply",
        text,
        re.I,
    )
    if (
        match
        and match.group(1) in tags.HABITAT_TAGS
        and match.group(3) in tags.FOOD_TAGS
    ):
        return schema.Effect(
            kind=schema.EffectKind.PINK_PLAY_BIRD_GAIN,
            habitat=tags.HABITAT_TAGS[match.group(1)],
            food=tags.FOOD_TAGS[match.group(3)],
            amount=tags.to_int(match.group(2)) or 1,
            raw_text=match.group(0),
        )
    return None


@registry.pink_pattern
def _m_pink_play_bird_tuck(text: str) -> schema.Effect | None:
    # Horned Lark: "When another player plays a bird in their [habitat], tuck N
    # [card] from your hand behind this bird."
    match = re.search(
        r"plays a bird in (?:their )?(\[\w+\]).*?tuck\s+(\d+|a|an|one|two|three)\s+\[card\]",
        text,
        re.I,
    )
    if match and match.group(1) in tags.HABITAT_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.PINK_PLAY_BIRD_TUCK,
            habitat=tags.HABITAT_TAGS[match.group(1)],
            amount=tags.to_int(match.group(2)) or 1,
            raw_text=match.group(0),
        )
    return None


@registry.pink_pattern
def _m_pink_gain_food_cache(text: str) -> schema.Effect | None:
    # Loggerhead Shrike: "When another player takes the [gain food] action, if
    # they gain any number of [food], cache N [food] from the supply on this bird."
    match = re.search(
        r"gain food.*?cache\s+(\d+|a|an|one|two|three)\s+(\[\w+\])",
        text,
        re.I,
    )
    if match and match.group(2) in tags.FOOD_TAGS:
        return schema.Effect(
            kind=schema.EffectKind.PINK_GAIN_FOOD_CACHE,
            food=tags.FOOD_TAGS[match.group(2)],
            amount=tags.to_int(match.group(1)) or 1,
            raw_text=match.group(0),
        )
    return None
