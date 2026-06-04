# pyright: reportPrivateUsage=false
# (reads the sibling encode module's package-private layout constants —
# deliberate intra-package coupling identical to encode/stripes.py's convention)
"""Programmatic stripe registry for the setup model's input vector.

The setup analogue of :mod:`wingspan.encode.stripes`: :func:`setup_stripe_layout`
returns a :class:`wingspan.encode.stripes.VectorLayout` naming every block of the
:func:`wingspan.setup_model.encode.encode_setup_candidate` feature vector, with
offsets and sizes derived from the same constants the encoder uses. The registry
documents the *raw* vector — the layout's ``total_size`` equals
``SETUP_FEATURE_DIM`` — while the in-net embedding rewrite (the kept-cards
multi-hot through the frozen set encoder, the tray index columns through the
frozen card table) is noted per stripe rather than expanded, since the setup
net's readout width also depends on the main net's embed dims.
"""

from __future__ import annotations

from wingspan import cards, encode
from wingspan.encode import stripes as encode_stripes
from wingspan.setup_model import encode as setup_encode


def setup_stripe_layout() -> encode_stripes.VectorLayout:
    """Build the stripe registry for the setup net's input vector.

    Lists the six fixed blocks in offset order. Two deliberate contrasts with
    the in-game encoder are called out in the notes: the birdfeeder block
    carries *raw* die-face counts (the state vector's is normalized ÷ 5), and
    each round goal is a bare category one-hot (no count / VP scalars).
    """
    food_names = ", ".join(food.value for food in cards.ALL_FOODS)

    stripes: list[encode_stripes.StripeDescriptor] = []
    off = 0

    # ---- candidate blocks: the keep being scored ----
    stripes.append(
        encode_stripes.StripeDescriptor(
            name="kept_cards",
            description=(
                f"The bird cards this candidate keeps, as a multi-hot over all "
                f"{setup_encode._KEPT_CARDS_DIM} core-set birds."
            ),
            offset=off,
            size=setup_encode._KEPT_CARDS_DIM,
            encoding="multi-hot",
            value_range="{0, 1}",
            notes=(
                "Indexed by stable bird order from cards.bird_index(). Embedded "
                "in-net as one card *set* through the frozen copy of the main "
                "net's multi-card set encoder (multi-hot ⊕ derived 10-dim set "
                "summary -> one set vector)."
            ),
        )
    )
    off += setup_encode._KEPT_CARDS_DIM

    stripes.append(
        encode_stripes.StripeDescriptor(
            name="kept_foods",
            description="The starting food tokens this candidate keeps.",
            offset=off,
            size=setup_encode._KEPT_FOODS_DIM,
            encoding="multi-hot",
            value_range="{0, 1}",
            notes=f"Food types in order: {food_names}.",
            sub_fields=_kept_food_sub_fields(),
        )
    )
    off += setup_encode._KEPT_FOODS_DIM

    stripes.append(
        encode_stripes.StripeDescriptor(
            name="kept_bonus",
            description=(
                f"The bonus card this candidate keeps, as a one-hot over all "
                f"{setup_encode._BONUS_DIM} core-set bonus cards."
            ),
            offset=off,
            size=setup_encode._BONUS_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=(
                "Indexed by stable bonus-card order from cards.bonus_index(). "
                "All-zero when no bonus is kept (split_setup_bonus defers the "
                "bonus pick to the in-game CHOOSE_BONUS head)."
            ),
        )
    )
    off += setup_encode._BONUS_DIM

    # ---- context blocks: the shared per-deal view ----
    stripes.append(
        encode_stripes.StripeDescriptor(
            name="tray",
            description=(
                f"The face-up tray birds (context), as {setup_encode._TRAY_DIM} "
                "positional integer card indices."
            ),
            offset=off,
            size=setup_encode._TRAY_DIM,
            encoding="integer-index",
            value_range=f"int 0–{cards.n_birds()}",
            notes=(
                f"{setup_encode._TRAY_DIM} slot-order indices (bird_index + 1; "
                "0 = empty slot), matching the state vector's tray block. Embedded "
                "in-net through the frozen copy of the main net's card table (one "
                "card vector per slot) plus one derived tray-*set* embedding "
                "through the frozen set encoder."
            ),
        )
    )
    off += setup_encode._TRAY_DIM

    stripes.append(
        encode_stripes.StripeDescriptor(
            name="birdfeeder",
            description="Birdfeeder die-face counts: single-food faces and choice-wild dice.",
            offset=off,
            size=setup_encode._FEEDER_DIM,
            encoding="vector",
            value_range="int 0–5",
            notes=(
                f"6 values: one per food type ({food_names}) for single-food faces, "
                "then the count of choice-die (wild) faces. Raw counts — NOT "
                "normalized, unlike the state vector's birdfeeder stripe (÷ 5)."
            ),
            sub_fields=_birdfeeder_sub_fields(),
        )
    )
    off += setup_encode._FEEDER_DIM

    stripes.append(
        encode_stripes.StripeDescriptor(
            name="round_goals",
            description="The four rounds' end-of-round goals (context), one one-hot each.",
            offset=off,
            size=setup_encode._GOALS_DIM,
            encoding="complex",
            value_range="{0, 1}",
            notes=(
                f"4 rounds × {setup_encode.SETUP_GOAL_DIM}-wide category one-hot, in "
                "the shared goal-category order the in-game encoder pins. Category "
                "only — no count / placement-VP scalars (nothing has been scored at "
                "setup time)."
            ),
            sub_fields=_round_goal_sub_fields(),
        )
    )
    off += setup_encode._GOALS_DIM

    assert off == setup_encode.SETUP_FEATURE_DIM, (
        f"stripe offsets sum to {off} but SETUP_FEATURE_DIM is "
        f"{setup_encode.SETUP_FEATURE_DIM} — setup_model encode.py and stripes.py "
        "are out of sync"
    )
    return encode_stripes.VectorLayout(
        total_size=setup_encode.SETUP_FEATURE_DIM, stripes=tuple(stripes)
    )


###### PRIVATE #######


def _kept_food_sub_fields() -> tuple[encode_stripes.SubFieldDescriptor, ...]:
    """One sub-field per food type in the kept-food multi-hot."""
    return tuple(
        encode_stripes.SubFieldDescriptor(
            name=f"kept_{food.value}",
            description=f"1.0 if this candidate keeps a {food.value} token.",
            relative_offset=idx,
            size=1,
            encoding="multi-hot bit",
            value_range="{0, 1}",
        )
        for idx, food in enumerate(cards.ALL_FOODS)
    )


def _birdfeeder_sub_fields() -> tuple[encode_stripes.SubFieldDescriptor, ...]:
    """6 sub-fields for the birdfeeder stripe (one per food face + choice die)."""
    sub_fields: list[encode_stripes.SubFieldDescriptor] = []
    for idx, food in enumerate(cards.ALL_FOODS):
        sub_fields.append(
            encode_stripes.SubFieldDescriptor(
                name=f"face_{food.value}",
                description=f"Dice showing a {food.value} face in the birdfeeder.",
                relative_offset=idx,
                size=1,
                encoding="scalar",
                value_range="int 0–5",
                notes="Raw count, not normalized.",
            )
        )
    sub_fields.append(
        encode_stripes.SubFieldDescriptor(
            name="face_choice_die",
            description="Dice showing a choice-wild (invertebrate/seed) face.",
            relative_offset=len(sub_fields),
            size=1,
            encoding="scalar",
            value_range="int 0–5",
            notes="Raw count, not normalized.",
        )
    )
    return tuple(sub_fields)


def _round_goal_sub_fields() -> tuple[encode_stripes.SubFieldDescriptor, ...]:
    """One category one-hot block per round for the round-goals stripe."""
    return tuple(
        encode_stripes.SubFieldDescriptor(
            name=f"round_{round_idx}.category",
            description=(
                f"Round {round_idx} goal category "
                f"(one-hot over {setup_encode.SETUP_GOAL_DIM} categories)."
            ),
            relative_offset=round_idx * setup_encode.SETUP_GOAL_DIM,
            size=setup_encode.SETUP_GOAL_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=f"Categories in index order: {', '.join(encode.GOAL_CATEGORIES)}.",
            group=f"round_{round_idx}",
        )
        for round_idx in range(setup_encode._NUM_SETUP_GOALS)
    )
