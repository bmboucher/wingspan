# pyright: reportPrivateUsage=false
# (reads the shared, package-private layout constants — deliberate intra-package
# coupling identical to state_encode.py's convention)
"""Card-feature and hand-encoder-input stripe layouts.

``card_feature_stripe_layout`` documents the single-card encoder's raw input
vector; ``hand_encoder_input_stripe_layout`` documents the multi-card set
encoder's raw input vector.  Neither applies a post-embedding rewrite — these
vectors are the encoder inputs as-is.
"""

from __future__ import annotations

from wingspan import cards
from wingspan.encode import layout
from wingspan.encode.stripes import descriptors


def card_feature_stripe_layout() -> descriptors.VectorLayout:
    """Build the stripe registry for the single-card encoder's input vector.

    Documents one row of ``state_encode.card_feature_matrix``: the bird's static
    normalized attribute vector concatenated with its identity one-hot
    (``layout.CARD_FEATURE_DIM`` raw dims — this *is* the encoder's input, so
    there is no post-embedding rewrite). Row 0 of the feature table is all-zero
    (the empty-slot / padding row); every other row is one core bird.
    """
    stripes = (
        descriptors.StripeDescriptor(
            name="bird_attrs",
            description=(
                "Dense, normalized view of the bird's immutable card attributes."
            ),
            offset=0,
            size=layout._BIRD_ATTR_DIM,
            encoding="complex",
            value_range="[0, ~1]",
            notes=(
                "Static printed-card facts only — mutable per-slot state (eggs, "
                "cached food, …) lives in the board stripes, not here."
            ),
            sub_fields=_card_attr_sub_fields(),
        ),
        descriptors.StripeDescriptor(
            name="bird_identity",
            description="One-hot identity of the bird over all core birds.",
            offset=layout._BIRD_ATTR_DIM,
            size=layout._BIRD_ID_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=(
                "Indexed by cards.bird_index; gives the encoder a learned "
                "per-card component beyond the shared attributes."
            ),
        ),
    )
    total = layout._BIRD_ATTR_DIM + layout._BIRD_ID_DIM
    assert (
        total == layout.CARD_FEATURE_DIM
    ), f"card stripes end at {total} but CARD_FEATURE_DIM = {layout.CARD_FEATURE_DIM}"
    return descriptors.VectorLayout(total_size=total, stripes=stripes)


def hand_encoder_input_stripe_layout() -> descriptors.VectorLayout:
    """Build the stripe registry for the multi-card encoder's input vector.

    Documents the raw ``[multi-hot ⊕ summary]`` concat ``hand_model.embed_card_set``
    feeds the encoder for any card set (the own hand, a setup pick's kept set, the
    tray set) — ``layout.HAND_ENCODER_INPUT_DIM`` raw dims, no embedding rewrite.
    """
    stripes = (
        descriptors.StripeDescriptor(
            name="hand_multihot",
            description="Multi-hot identity of every bird in the card set.",
            offset=0,
            size=layout.HAND_MULTIHOT_DIM,
            encoding="multi-hot",
            value_range="{0, 1}",
            notes=(
                "One bit per core bird, indexed by cards.bird_index. The same "
                "encoder embeds the own hand, a setup pick's kept set, and the "
                "tray set."
            ),
        ),
        descriptors.StripeDescriptor(
            name="hand_summary",
            description="Aggregate statistics summarizing the card set.",
            offset=layout.HAND_MULTIHOT_DIM,
            size=layout.HAND_SUMMARY_DIM,
            encoding="vector",
            value_range="[0, ~1]",
            notes=(
                "Set size, per-habitat membership counts, and per-food "
                "any-bird-costs-it flags."
            ),
            sub_fields=descriptors.hand_summary_sub_fields(),
        ),
    )
    total = layout.HAND_MULTIHOT_DIM + layout.HAND_SUMMARY_DIM
    assert total == layout.HAND_ENCODER_INPUT_DIM, (
        f"hand-encoder stripes end at {total} but HAND_ENCODER_INPUT_DIM = "
        f"{layout.HAND_ENCODER_INPUT_DIM}"
    )
    return descriptors.VectorLayout(total_size=total, stripes=stripes)


###### PRIVATE #######


def _card_attr_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """11 sub-fields for the bird-attribute stripe, keyed to the same
    ``layout._OFF_ATTR_*`` offsets ``state_encode._bird_attr_vector`` writes."""
    food_names = ", ".join(food.value for food in cards.ALL_FOODS)
    habitat_names = ", ".join(habitat.value for habitat in cards.ALL_HABITATS)
    nest_names = ", ".join(nest.value for nest in layout._NEST_BASE_TYPES)
    color_names = ", ".join(color.value for color in layout._COLORS)

    # (name, relative_offset, size, encoding, value_range, description, notes)
    entries: list[tuple[str, int, int, str, str, str, str | None]] = [
        (
            "points",
            layout._OFF_ATTR_POINTS,
            1,
            "scalar",
            "[0, 1]",
            "Printed victory-point value.",
            f"Normalized ÷ {int(layout._POINTS_SCALE)}.",
        ),
        (
            "food_cost",
            layout._OFF_ATTR_FOOD_COST,
            layout._FOOD_COST_VEC_DIM,
            "vector",
            "[0, ~1]",
            "Printed food cost, one element per food type plus wild.",
            f"Order: {food_names}, wild. Normalized ÷ "
            f"{int(layout._PER_FOOD_COST_SCALE)}.",
        ),
        (
            "nest",
            layout._OFF_ATTR_NEST,
            len(layout._NEST_BASE_TYPES),
            "multi-hot",
            "{0, 1}",
            "Nest type over the concrete nests.",
            f"Order: {nest_names}. A STAR nest is a wildcard (all ones); "
            "a bird with no nest is all zeros.",
        ),
        (
            "habitats",
            layout._OFF_ATTR_HAB,
            len(cards.ALL_HABITATS),
            "multi-hot",
            "{0, 1}",
            "Habitats the bird can be played in.",
            f"Order: {habitat_names}. Dual-habitat birds set two bits.",
        ),
        (
            "flocking",
            layout._OFF_ATTR_FLOCK,
            1,
            "scalar",
            "{0, 1}",
            "Whether the bird has a flocking (tuck) power.",
            None,
        ),
        (
            "predator",
            layout._OFF_ATTR_PRED,
            1,
            "scalar",
            "{0, 1}",
            "Whether the bird has a predator power.",
            None,
        ),
        (
            "wingspan",
            layout._OFF_ATTR_WINGSPAN,
            1,
            "scalar",
            "[0, ~1]",
            "Printed wingspan in centimeters.",
            f"Normalized ÷ {int(layout._WINGSPAN_SCALE)}.",
        ),
        (
            "egg_limit",
            layout._OFF_ATTR_EGG_LIMIT,
            1,
            "scalar",
            "[0, 1]",
            "Printed egg capacity.",
            f"Normalized ÷ {int(layout._EGG_LIMIT_SCALE)}.",
        ),
        (
            "color",
            layout._OFF_ATTR_COLOR,
            len(layout._COLORS),
            "one-hot",
            "{0, 1}",
            "Power color of the bird's ability.",
            f"Order: {color_names}. A power-less bird is all zeros.",
        ),
        (
            "plays_another_bird",
            layout._OFF_ATTR_PLAYS_BIRD,
            1,
            "scalar",
            "{0, 1}",
            "Whether the bird's white power grants an extra bird play.",
            "Set for PLAY_ADDITIONAL_BIRD and PLAY_ADDITIONAL_BIRD_HERE effects.",
        ),
        (
            "caches_food",
            layout._OFF_ATTR_CACHES_FOOD,
            1,
            "scalar",
            "{0, 1}",
            "Whether the bird's power includes a food-caching effect.",
            "Set for CACHE_FOOD, GAIN_FOOD_FEEDER_MAY_CACHE, "
            "ROLL_NOT_IN_FEEDER_CACHE, and PINK_GAIN_FOOD_CACHE.",
        ),
        (
            "bonus_categories",
            layout._OFF_ATTR_BONUS_CATS,
            layout._BONUS_CATS_DIM,
            "multi-hot",
            "{0, 1}",
            "Curated bonus cards the bird statically qualifies for.",
            "7 intrinsic-property categories (lexical or numeric threshold — "
            "not state-dependent, not covered by another stripe). "
            "Order: Anatomist, Backyard Birder, Cartographer, Historian, "
            "Large Bird Specialist, Passerine Specialist, Photographer. "
            "Dense 0..6 indices (independent of cards.bonus_index).",
        ),
        (
            "power_exchange",
            layout._OFF_ATTR_POWER_EX,
            layout._EXCHANGE_DIM,
            "vector",
            "[0, ~1]",
            "Resource exchange encoding what the bird's power does.",
            "13 slots matching the choice-row exchange stripe semantics: "
            "[cards_to_discard, food_to_pay, eggs_to_pay, food_to_gain, "
            "eggs_to_gain, cards_to_draw, cards_to_tuck, opp_food_to_gain, "
            "opp_eggs_to_gain, opp_cards_to_draw, opp_cards_to_tuck, "
            f"plays_to_gain, cache_to_gain]. Normalized ÷ "
            f"{int(layout._EXCHANGE_SCALE)}. Zero for UNIMPLEMENTED powers.",
        ),
    ]
    return tuple(
        descriptors.SubFieldDescriptor(
            name=name,
            description=description,
            relative_offset=relative_offset,
            size=size,
            encoding=encoding,
            value_range=value_range,
            notes=notes,
        )
        for name, relative_offset, size, encoding, value_range, description, notes in entries
    )
