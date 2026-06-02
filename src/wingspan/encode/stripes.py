# pyright: reportPrivateUsage=false
# (reads the shared, package-private layout constants — deliberate intra-package
# coupling identical to state_encode.py's convention)
"""Programmatic stripe registry for the state and choice vectors.

Each public function returns a :class:`VectorLayout` that lists every stripe in
the order they appear in the flat vector, with a short reference name, a human
description, size, encoding kind, value range, and optional sub-field notes.
All sizes are derived from the same ``layout`` constants the encoders use, so a
change to ``layout.py`` automatically flows through to this registry.
"""

from __future__ import annotations

import pydantic

from wingspan import cards, decisions, state
from wingspan.encode import layout


class StripeDescriptor(pydantic.BaseModel):
    """One named region of a flat feature vector."""

    name: str
    """Short reference name (snake_case, suitable for indexing or labelling)."""

    description: str
    """Human-readable sentence describing what this stripe encodes."""

    offset: int
    """Index of the first element in the flat vector."""

    size: int
    """Number of elements."""

    encoding: str
    """Encoding kind: ``scalar``, ``vector``, ``one-hot``, ``multi-hot``,
    ``integer-index``, or ``complex`` (structured block, see notes)."""

    value_range: str
    """Typical element values, e.g. ``[0, 1]``, ``{0, 1}``, ``int 0–180``."""

    notes: str | None = None
    """Sub-field layout, normalization constants, or other caveats."""


class VectorLayout(pydantic.BaseModel):
    """The complete named stripe breakdown of a flat feature vector."""

    total_size: int
    """Total element count (equals ``len(stripes[i].size)`` summed)."""

    stripes: tuple[StripeDescriptor, ...]


def state_stripe_layout() -> VectorLayout:
    """Build the stripe registry for the state vector produced by ``encode_state``.

    Stripes are listed in offset order — the same order ``encode_state``
    concatenates them — and sizes are computed from ``layout`` constants so
    changes to the encoding propagate automatically.
    """
    from wingspan.encode import state_encode

    total = state_encode.state_size()
    food_names = ", ".join(f.value for f in cards.ALL_FOODS)
    habitat_names = ", ".join(h.value for h in cards.ALL_HABITATS)

    stripes: list[StripeDescriptor] = []
    off = 0

    # ---- food inventory ----
    stripes.append(
        StripeDescriptor(
            name="food_me",
            description="My food inventory, one element per food type.",
            offset=off,
            size=cards.N_FOODS,
            encoding="vector",
            value_range="[0, ~1.7]",
            notes=f"Food types in order: {food_names}. Normalized ÷ 6.",
        )
    )
    off += cards.N_FOODS

    stripes.append(
        StripeDescriptor(
            name="food_opp",
            description="Opponent food inventory, one element per food type.",
            offset=off,
            size=cards.N_FOODS,
            encoding="vector",
            value_range="[0, ~1.7]",
            notes=f"Food types in order: {food_names}. Normalized ÷ 6.",
        )
    )
    off += cards.N_FOODS

    # ---- board continuous (mutable per-slot state) ----
    n_slots = state.N_HABITATS * state.ROW_SLOTS
    slot_dim = layout._SLOT_MUT_DIM  # 9: eggs, egg_cap, cached×5, tucked, activations
    board_dim = layout._BOARD_CONT_STRIPE_DIM  # n_slots * slot_dim

    _board_notes = (
        f"{n_slots} slots ({state.N_HABITATS} habitats × {state.ROW_SLOTS} positions). "
        f"Per slot ({slot_dim} values): eggs[0], egg_cap[1], "
        f"cached_food_by_type[2:{2 + cards.N_FOODS}] ({food_names}), "
        f"tucked[{layout._SLOT_MUT_TUCKED}], activations[{layout._SLOT_MUT_ACTIVATIONS}]. "
        "Eggs and cached food normalized ÷ 6; activations ÷ 4."
    )
    stripes.append(
        StripeDescriptor(
            name="board_me",
            description="Mutable per-slot board state for my board.",
            offset=off,
            size=board_dim,
            encoding="complex",
            value_range="[0, ~1]",
            notes=_board_notes,
        )
    )
    off += board_dim

    stripes.append(
        StripeDescriptor(
            name="board_opp",
            description="Mutable per-slot board state for the opponent's board.",
            offset=off,
            size=board_dim,
            encoding="complex",
            value_range="[0, ~1]",
            notes=_board_notes,
        )
    )
    off += board_dim

    # ---- board summary (aggregate per-habitat stats) ----
    _board_summary_notes = (
        f"3 habitats ({habitat_names}) × 6 stats: "
        "row_length (filled slots), total_eggs, total_points, "
        "total_tucked, total_cached_food, brown_bird_count. "
        "All normalized to approx [0, 1]."
    )
    _board_summary_size = state.N_HABITATS * 6  # 3 × 6 = 18

    stripes.append(
        StripeDescriptor(
            name="board_summary_me",
            description="Aggregate per-habitat row statistics for my board.",
            offset=off,
            size=_board_summary_size,
            encoding="vector",
            value_range="[0, ~1]",
            notes=_board_summary_notes,
        )
    )
    off += _board_summary_size

    stripes.append(
        StripeDescriptor(
            name="board_summary_opp",
            description="Aggregate per-habitat row statistics for the opponent's board.",
            offset=off,
            size=_board_summary_size,
            encoding="vector",
            value_range="[0, ~1]",
            notes=_board_summary_notes,
        )
    )
    off += _board_summary_size

    # ---- hand summary ----
    stripes.append(
        StripeDescriptor(
            name="hand_summary_me",
            description="Aggregated statistics about my current hand.",
            offset=off,
            size=8,
            encoding="vector",
            value_range="[0, ~1]",
            notes=(
                "8 stats: hand_size (÷10), mean_points (÷9), max_points (÷9), "
                "mean_food_cost (÷7), min_food_cost (÷7), mean_egg_limit (÷6), "
                "forest_bird_count (÷hand_size), wetland_bird_count (÷hand_size)."
            ),
        )
    )
    off += 8

    # ---- bonus progress (POV player only; opponent identity hidden) ----
    bonus_dim = layout._BONUS_ID_DIM  # 26 bonus cards
    stripes.append(
        StripeDescriptor(
            name="bonus_progress_held",
            description=(
                f"Which of the {bonus_dim} bonus cards I am holding (multi-hot)."
            ),
            offset=off,
            size=bonus_dim,
            encoding="multi-hot",
            value_range="{0, 1}",
            notes="Indexed by stable bonus-card order from cards.bonus_index().",
        )
    )
    off += bonus_dim

    stripes.append(
        StripeDescriptor(
            name="bonus_progress_count",
            description="Number of my birds that qualify for each bonus card.",
            offset=off,
            size=bonus_dim,
            encoding="vector",
            value_range="[0, ~1]",
            notes=f"One value per bonus card ({bonus_dim} total). Normalized ÷ 5.",
        )
    )
    off += bonus_dim

    stripes.append(
        StripeDescriptor(
            name="bonus_progress_stepped",
            description="Current stepped VP for each bonus card I hold.",
            offset=off,
            size=bonus_dim,
            encoding="vector",
            value_range="[0, ~1]",
            notes=f"One value per bonus card ({bonus_dim} total). Normalized ÷ 7 (max single-card VP).",
        )
    )
    off += bonus_dim

    stripes.append(
        StripeDescriptor(
            name="bonus_progress_linear",
            description="Linear (fractional) VP for each bonus card I hold.",
            offset=off,
            size=bonus_dim,
            encoding="vector",
            value_range="[0, ~1]",
            notes=f"One value per bonus card ({bonus_dim} total). Normalized ÷ 7.",
        )
    )
    off += bonus_dim

    # ---- opponent aggregate counts ----
    stripes.append(
        StripeDescriptor(
            name="bonus_count_opp",
            description="Number of bonus cards the opponent holds (identity hidden).",
            offset=off,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes="Normalized ÷ 5.",
        )
    )
    off += 1

    stripes.append(
        StripeDescriptor(
            name="hand_size_opp",
            description="Number of bird cards in the opponent's hand (contents hidden).",
            offset=off,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes="Normalized ÷ 10.",
        )
    )
    off += 1

    # ---- birdfeeder ----
    stripes.append(
        StripeDescriptor(
            name="birdfeeder",
            description="Birdfeeder die face counts: single-food faces and choice-wild dice.",
            offset=off,
            size=6,
            encoding="vector",
            value_range="[0, 1]",
            notes=(
                f"6 values: one per food type ({food_names}) for single-food faces, "
                "then the count of choice-die (wild) faces. "
                "Each normalized ÷ 5 (max dice showing that face)."
            ),
        )
    )
    off += 6

    # ---- miscellaneous scalars ----
    stripes.append(
        StripeDescriptor(
            name="misc_scalars",
            description="Miscellaneous scalar game state (round, cubes, scores, deck).",
            offset=off,
            size=7,
            encoding="vector",
            value_range="[0, ~1]",
            notes=(
                "7 values in order: round_index (÷4), my_action_cubes (÷8), "
                "opp_action_cubes (÷8), my_round_goal_pts (÷10), "
                "opp_round_goal_pts (÷10), tray_size (÷3), deck_size (÷100)."
            ),
        )
    )
    off += 7

    # ---- round-goal state (all four rounds) ----
    rounds_dim = layout._ROUND_GOALS_STRIPE_DIM
    goal_slot = layout._ROUND_GOAL_SLOT_DIM  # MAX_GOAL_CATEGORIES + 3 = 23
    stripes.append(
        StripeDescriptor(
            name="round_goals",
            description="State of all four round goals (category, counts, VP placement).",
            offset=off,
            size=rounds_dim,
            encoding="complex",
            value_range="varies",
            notes=(
                f"4 rounds × {goal_slot} values. Per round: "
                f"category_one_hot[0:{layout.MAX_GOAL_CATEGORIES}] ({layout.MAX_GOAL_CATEGORIES} dims), "
                "my_count (normalized ÷ 5), opp_count (normalized ÷ 5), "
                "placement_vp (normalized ÷ 10)."
            ),
        )
    )
    off += rounds_dim

    # ---- card-identity index block ----
    n_board_idx = layout.N_BOARD_INDEX_SLOTS  # 2 * 15 = 30
    stripes.append(
        StripeDescriptor(
            name="card_idx_board",
            description=(
                "Bird indices for all board slots (my board then opponent's), "
                "looked up in the shared card embedding table."
            ),
            offset=off,
            size=n_board_idx,
            encoding="integer-index",
            value_range=f"int 0–{cards.n_birds()}",
            notes=(
                f"{n_board_idx} integer indices ({state.N_HABITATS * state.ROW_SLOTS} me + "
                f"{state.N_HABITATS * state.ROW_SLOTS} opp). "
                "bird_index + 1; 0 = empty slot."
            ),
        )
    )
    off += n_board_idx

    stripes.append(
        StripeDescriptor(
            name="card_idx_tray",
            description=(
                "Bird indices for the three face-up tray slots, "
                "looked up in the shared card embedding table."
            ),
            offset=off,
            size=state.TRAY_SIZE,
            encoding="integer-index",
            value_range=f"int 0–{cards.n_birds()}",
            notes=f"{state.TRAY_SIZE} integer indices. bird_index + 1; 0 = empty slot.",
        )
    )
    off += state.TRAY_SIZE

    # ---- hand identity (multi-hot) ----
    hand_dim = layout.HAND_MULTIHOT_DIM  # n_birds = 180
    stripes.append(
        StripeDescriptor(
            name="hand_multihot",
            description=(
                f"My hand encoded as a multi-hot over all {hand_dim} core birds. "
                "Opponent hand is hidden."
            ),
            offset=off,
            size=hand_dim,
            encoding="multi-hot",
            value_range="{0, 1}",
            notes=(
                "Indexed by stable bird order from cards.bird_index(). "
                "Mean-pooled through the shared card embedding inside the model."
            ),
        )
    )
    off += hand_dim

    # ---- decision-type one-hot (always last) ----
    decision_dim = layout.DECISION_TYPE_DIM
    decision_names = ", ".join(cls.__name__ for cls in decisions.ALL_DECISION_CLASSES)
    stripes.append(
        StripeDescriptor(
            name="decision_type",
            description=(
                f"One-hot encoding of which Decision subclass is being resolved "
                f"({decision_dim} classes)."
            ),
            offset=off,
            size=decision_dim,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=f"Indexed by ALL_DECISION_CLASSES order: {decision_names}.",
        )
    )
    off += decision_dim

    assert off == total, (
        f"stripe offsets sum to {off} but state_size() returns {total} — "
        "layout.py and stripes.py are out of sync"
    )
    return VectorLayout(total_size=total, stripes=tuple(stripes))


def choice_stripe_layout() -> VectorLayout:
    """Build the stripe registry for the choice vector produced by ``encode_choices``.

    Each stripe corresponds to a type-specific feature group in the flat
    ``CHOICE_FEATURE_DIM``-wide vector that every candidate is encoded into.
    """
    total = layout.CHOICE_FEATURE_DIM
    food_names = ", ".join(f.value for f in cards.ALL_FOODS)
    habitat_names = ", ".join(h.value for h in cards.ALL_HABITATS)

    stripes: list[StripeDescriptor] = []

    # Stripe size constants (from layout private constants, stable by convention)
    kind_dim = layout._KIND_DIM  # 6
    food_dim = layout._FOOD_DIM  # 5
    hab_dim = layout._HABITAT_DIM  # 3
    pay_dim = layout._PAYMENT_DIM  # 5
    board_dim = layout._BOARD_TARGET_DIM  # 8
    special_dim = layout._SPECIAL_DIM  # 3
    exchange_dim = layout._EXCHANGE_DIM  # 3
    setup_dim = layout._SETUP_DIM  # 4
    bird_id_dim = layout._BIRD_ID_DIM  # 180
    bonus_id_dim = layout._BONUS_ID_DIM  # 26

    kind_labels = (
        "bird(0), food(1), habitat(2), payment(3), board_target(4), special(5)"
    )

    stripes.append(
        StripeDescriptor(
            name="kind",
            description="One-hot encoding of the choice's type.",
            offset=layout._OFF_KIND,
            size=kind_dim,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=f"Indices: {kind_labels}.",
        )
    )

    stripes.append(
        StripeDescriptor(
            name="food",
            description="One-hot encoding of a food-type choice.",
            offset=layout._OFF_FOOD,
            size=food_dim,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=f"Food types in order: {food_names}. Zero for non-food choices.",
        )
    )

    stripes.append(
        StripeDescriptor(
            name="habitat",
            description="One-hot encoding of a habitat choice.",
            offset=layout._OFF_HAB,
            size=hab_dim,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=f"Habitats in order: {habitat_names}. Zero for non-habitat choices.",
        )
    )

    stripes.append(
        StripeDescriptor(
            name="payment",
            description="Food payment vector: normalized count per food type.",
            offset=layout._OFF_PAY,
            size=pay_dim,
            encoding="vector",
            value_range="[0, 1]",
            notes=(
                f"One value per food type ({food_names}), normalized ÷ 4. "
                "Used for FoodPaymentChoice."
            ),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="board_target",
            description="Features of a board-slot target choice.",
            offset=layout._OFF_BOARD,
            size=board_dim,
            encoding="complex",
            value_range="[0, 1]",
            notes=(
                f"{board_dim} values: habitat_one_hot[0:3] ({habitat_names}), "
                "slot_position (normalized), eggs (normalized ÷ 6), "
                "egg_cap_remaining (normalized ÷ 6), "
                "total_cached_food (normalized ÷ 6), tucked_cards (normalized ÷ 6). "
                "Zero for non-board-target choices."
            ),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="special",
            description="Special-case flags for skip, main-action, and setup choices.",
            offset=layout._OFF_SPECIAL,
            size=special_dim,
            encoding="vector",
            value_range="mixed",
            notes=(
                f"{special_dim} values: is_skip[0] {{0,1}}, "
                "encoded_main_action_slot[1] (÷4, for MainAction choices), "
                "setup_is_keep[2] {0,1} (for SetupChoice)."
            ),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="exchange",
            description="Accept-exchange trade terms for a PayCostChoice.",
            offset=layout._OFF_EXCHANGE,
            size=exchange_dim,
            encoding="vector",
            value_range="[0, 1]",
            notes=(
                f"{exchange_dim} values: eggs_paid (÷3), cards_gained (÷3), "
                "tucks_gained (÷3). "
                "Food paid reuses the FOOD stripe. Zero for non-exchange choices."
            ),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="setup_agg",
            description=(
                "Aggregate statistics of the kept-card subset for a SetupChoice."
            ),
            offset=layout._OFF_SETUP,
            size=setup_dim,
            encoding="vector",
            value_range="[0, ~1]",
            notes=(
                f"{setup_dim} values: summed_points (÷9), summed_food_cost (÷7), "
                "summed_egg_limit (÷6), kept_count (÷5). "
                "Zero for non-setup choices."
            ),
        )
    )

    bird_id_off = layout._OFF_BIRD_ID
    stripes.append(
        StripeDescriptor(
            name="bird_id",
            description=(
                f"Bird identity: one-hot (single bird) or multi-hot (kept set) "
                f"over all {bird_id_dim} core-set birds."
            ),
            offset=bird_id_off,
            size=bird_id_dim,
            encoding="one-hot / multi-hot",
            value_range="{0, 1}",
            notes=(
                "Embedded through the shared card table (same weights as state board/tray slots). "
                "For SetupChoice the kept-set multi-hot is summed through the embedding. "
                "Zero for non-bird choices."
            ),
        )
    )

    bonus_id_off = layout._OFF_BONUS_ID
    stripes.append(
        StripeDescriptor(
            name="bonus_id",
            description=(
                f"Bonus-card identity one-hot over all {bonus_id_dim} core-set bonus cards."
            ),
            offset=bonus_id_off,
            size=bonus_id_dim,
            encoding="one-hot",
            value_range="{0, 1}",
            notes="Used for BonusCardChoice. Zero for non-bonus choices.",
        )
    )

    # Verify offsets cover the full vector
    assert bonus_id_off + bonus_id_dim == total, (
        f"choice stripe offsets end at {bonus_id_off + bonus_id_dim} "
        f"but CHOICE_FEATURE_DIM = {total}"
    )

    return VectorLayout(total_size=total, stripes=tuple(stripes))
