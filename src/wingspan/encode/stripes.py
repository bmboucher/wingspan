# pyright: reportPrivateUsage=false
# (reads the shared, package-private layout constants — deliberate intra-package
# coupling identical to state_encode.py's convention)
"""Programmatic stripe registry for the state and choice vectors.

Each public function returns a :class:`VectorLayout` that lists every stripe in
the order they appear in the flat vector, with a short reference name, a human
description, size, encoding kind, value range, and optional sub-field notes.
All sizes are derived from the same ``layout`` constants the encoders use, so a
change to ``layout.py`` automatically flows through to this registry.

Both layouts take an :class:`layout.EncodingSpec`; the config-driven setup pieces
(the choice ``setup_agg`` stripe and the decision-type one-hot's setup column)
are present only when ``spec.include_setup``. ``wingspan-inspect`` passes the
run's spec so the report shows exactly the fields that run encodes.
"""

from __future__ import annotations

import pydantic

from wingspan import architecture, cards, decisions, state
from wingspan.encode import layout

# Default card-embedding width for the report's post-embedding view, used when a
# caller doesn't pass the run's own (a bare ModelArchitecture baseline).
_DEFAULT_CARD_EMBED_DIM = architecture.ModelArchitecture().card_embed_dim


class SubFieldDescriptor(pydantic.BaseModel):
    """One named element or logical sub-group within a complex stripe.

    Used by :class:`StripeDescriptor` to expose drill-down detail for stripes
    whose elements are semantically distinct from each other (e.g. the 7 scalars
    in ``misc_scalars``, the per-slot features in a board slot, …). Homogeneous
    stripes where every element has the same meaning (``hand_multihot``, bonus
    one-hots) do not carry sub-fields — the parent stripe's ``notes`` are
    sufficient there.
    """

    name: str
    """Dot-qualified sub-field name, e.g. ``forest_0.eggs``."""

    description: str
    """Human-readable sentence describing this specific element or block."""

    relative_offset: int
    """Index of the first element *within the parent stripe* (0-based)."""

    size: int = 1
    """Element count (1 for a scalar; >1 for a one-hot block treated as a unit)."""

    encoding: str
    """Encoding kind matching the parent stripe's vocabulary."""

    value_range: str
    """Typical element values."""

    notes: str | None = None
    """Additional normalization or sub-structure details."""

    group: str | None = None
    """Optional grouping label used to nest sub-fields in the HTML report
    (e.g. ``"slot_forest_0"`` groups a slot's per-slot elements together)."""


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

    sub_fields: tuple[SubFieldDescriptor, ...] = ()
    """Per-element drill-down for semantically distinct stripes. Empty for
    homogeneous stripes where every element has the same meaning."""


class VectorLayout(pydantic.BaseModel):
    """The complete named stripe breakdown of a flat feature vector."""

    total_size: int
    """Total element count (equals ``sum(stripes[i].size)``)."""

    stripes: tuple[StripeDescriptor, ...]


def state_stripe_layout(
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
    card_embed_dim: int = _DEFAULT_CARD_EMBED_DIM,
    *,
    use_distinct_hand_model: bool = False,
    hand_embed_dim: int | None = None,
    tray_set_embedding: bool = False,
) -> VectorLayout:
    """Build the stripe registry for the state trunk's input vector.

    Lists every stripe in offset order with sizes from the ``layout`` constants.
    The card-index block and hand multi-hot are shown at their *post-embedding*
    width — each board / tray slot index as one ``card_embed_dim`` vector, the
    hand as one embedding (mean-pooled at ``card_embed_dim``, or the dedicated
    hand encoder's resolved ``hand_embed_dim`` under
    ``use_distinct_hand_model``, which also folds the 10-dim hand-summary stripe
    into the encoder's input) — so the breakdown sums to the trunk's
    first-``Linear`` input (``layout.trunk_input_dim``): what the network
    actually sees, not the raw encoder output. ``tray_set_embedding`` widens the
    tray stripe by one derived set embedding (3·M + N). (The model concatenates
    the embeddings after the continuous features; here they keep their
    encoding-order position.) Only the trailing decision-type one-hot's width
    depends on ``spec``.
    """
    from wingspan.encode import state_encode

    total = state_encode.state_size(spec)
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
            sub_fields=_food_sub_fields(),
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
            sub_fields=_food_sub_fields(),
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
            sub_fields=_board_slot_sub_fields(),
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
            sub_fields=_board_slot_sub_fields(),
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
            sub_fields=_board_summary_sub_fields(),
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
            sub_fields=_board_summary_sub_fields(),
        )
    )
    off += _board_summary_size

    # ---- hand summary ----
    stripes.append(
        StripeDescriptor(
            name="hand_summary_me",
            description="Compact summary of my current hand.",
            offset=off,
            size=10,
            encoding="vector",
            value_range="[0, ~1]",
            notes=(
                "10 values: hand_size[0] (÷10), per-habitat bird counts[1:4] "
                f"({habitat_names}; a bird counted once per habitat it lives in, "
                "÷10), then a food+wild multi-hot[4:10] — 1.0 if any hand bird has "
                f"that token in its food cost ({food_names}, wild)."
            ),
            sub_fields=_hand_summary_sub_fields(),
        )
    )
    off += 10

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
            sub_fields=_birdfeeder_sub_fields(),
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
                "7 values in order: round_index (÷3, ordinal), my_action_cubes (÷8), "
                "opp_action_cubes (÷8), my_round_goal_pts (÷10), "
                "opp_round_goal_pts (÷10), tray_size (÷3), deck_size (÷100)."
            ),
            sub_fields=_misc_scalars_sub_fields(),
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
            sub_fields=_round_goals_sub_fields(),
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

    # ---- decision-type one-hot (always last; setup column present iff include_setup) ----
    decision_dim = layout.decision_type_dim(spec)
    active_classes = decisions.active_decision_classes(spec.include_setup)
    decision_names = ", ".join(cls.__name__ for cls in active_classes)
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
            notes=(
                f"Indexed by active decision classes: {decision_names}. "
                f"{'Includes' if spec.include_setup else 'Excludes'} the SetupDecision "
                "column (config-driven by use_setup_model)."
            ),
        )
    )
    off += decision_dim

    assert off == total, (
        f"stripe offsets sum to {off} but state_size(spec) returns {total} — "
        "layout.py and stripes.py are out of sync"
    )
    raw = VectorLayout(total_size=total, stripes=tuple(stripes))
    return _embed_layout(
        raw,
        _state_embed_rules(
            card_embed_dim,
            use_distinct_hand_model=use_distinct_hand_model,
            hand_embed_dim=hand_embed_dim,
            tray_set_embedding=tray_set_embedding,
        ),
        layout.trunk_input_dim(
            total,
            card_embed_dim,
            use_distinct_hand_model=use_distinct_hand_model,
            hand_embed_dim=hand_embed_dim,
            tray_set_embedding=tray_set_embedding,
        ),
    )


def choice_stripe_layout(
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
    card_embed_dim: int = _DEFAULT_CARD_EMBED_DIM,
) -> VectorLayout:
    """Build the stripe registry for the per-choice encoder's input vector.

    Each stripe is a type-specific feature group every candidate is encoded into.
    The board-index block and bird-identity one-hot are shown at their
    *post-embedding* width — each board slot as one ``card_embed_dim`` vector, the
    candidate as one — so the breakdown sums to the choice encoder's first-``Linear``
    input (``layout.choice_input_dim``), what the network actually sees. The trailing
    ``setup_agg`` stripe is present only when ``spec.include_setup``.
    """
    total = layout.choice_feature_dim(spec)
    food_names = ", ".join(f.value for f in cards.ALL_FOODS)
    habitat_names = ", ".join(h.value for h in cards.ALL_HABITATS)

    stripes: list[StripeDescriptor] = []

    main_action_names = ", ".join(a.value for a in layout._MAIN_ACTION_ORDER)
    kind_labels = (
        "bird(0), food(1), habitat(2), payment(3), board_target(4), special(5)"
    )

    stripes.append(
        StripeDescriptor(
            name="kind",
            description="One-hot encoding of the choice's data-shape kind.",
            offset=layout._OFF_KIND,
            size=layout._KIND_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=f"Indices: {kind_labels}.",
            sub_fields=_kind_sub_fields(),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="gain_food",
            description="Food selection for a gain (and food choice for spend decisions).",
            offset=layout._OFF_GAIN_FOOD,
            size=layout._GAIN_FOOD_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=(
                f"7 values: the five plain-die foods ({food_names}) then "
                "take-choice-die-as-invertebrate[5] and take-choice-die-as-seed[6]. "
                "Zero for non-food choices."
            ),
            sub_fields=_gain_food_sub_fields(),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="habitat",
            description="One-hot encoding of a habitat choice.",
            offset=layout._OFF_HAB,
            size=layout._HABITAT_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=f"Habitats in order: {habitat_names}. Zero for non-habitat choices.",
            sub_fields=_choice_habitat_sub_fields(),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="pay_food",
            description="Food payment vector: normalized count per food type.",
            offset=layout._OFF_PAY,
            size=layout._PAY_FOOD_DIM,
            encoding="vector",
            value_range="[0, 1]",
            notes=(
                f"One value per food type ({food_names}), normalized ÷ 4. "
                "Used for a bird play's payment and a PayCostChoice's paid food."
            ),
            sub_fields=_choice_payment_sub_fields(),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="board_target",
            description="Per-board-slot features for a board-target (lay/remove egg) choice.",
            offset=layout._OFF_BOARD,
            size=layout._BOARD_TARGET_DIM,
            encoding="complex",
            value_range="[0, ~1]",
            notes=(
                f"{layout._SLOTS_PER_BOARD} board slots × {layout._BT_SLOT_SCALARS} "
                "scalars each: lay_eggs[0], pay_eggs[1] (set on the targeted slot "
                "for a lay-egg vs remove-egg decision), cached food per type[2:7] "
                f"({food_names}, ÷6), tucked[7] (÷6). The occupying bird ids ride the "
                "parallel board_idx block. Zero for non-board-target choices."
            ),
            sub_fields=_board_target_sub_fields(),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="main_action",
            description="One-hot encoding of which top-level action a MainActionChoice picks.",
            offset=layout._OFF_MAIN_ACTION,
            size=layout._MAIN_ACTION_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=f"Actions in order: {main_action_names}. Zero for non-main-action choices.",
            sub_fields=_main_action_sub_fields(),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="special",
            description="Special-case flags for skip and player-id choices.",
            offset=layout._OFF_SPECIAL,
            size=layout._SPECIAL_DIM,
            encoding="binary",
            value_range="{0, 1}",
            notes=(
                "2 flags: is_skip[0] (declines the decision), is_self[1] (the "
                "PlayerIdChoice option that is the active player — the Hummingbird "
                "food-gain order pick)."
            ),
            sub_fields=_special_sub_fields(),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="exchange",
            description="Symmetric pay->gain trade terms for a PayCostChoice.",
            offset=layout._OFF_EXCHANGE,
            size=layout._EXCHANGE_DIM,
            encoding="vector",
            value_range="[0, ~1]",
            notes=(
                f"{layout._EXCHANGE_DIM} resource-flow magnitudes (÷3): a 7-field self "
                "block (cards/food/eggs paid -> food/eggs/cards-drawn/cards-tucked "
                "gained) then a 4-field opponent-gain block (food/eggs/cards/tucks a "
                "shared-benefit power also grants the opponent). The food *type* paid "
                "rides the pay_food stripe. Zero for non-exchange choices."
            ),
            sub_fields=_exchange_sub_fields(),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="board_idx",
            description=(
                "Bird indices for the deciding player's 15 board slots — the "
                "board_target stripe's occupants, looked up in the shared card table."
            ),
            offset=layout._OFF_BOARD_IDX,
            size=layout._BOARD_IDX_SLOTS,
            encoding="integer-index",
            value_range=f"int 0–{cards.n_birds()}",
            notes=(
                f"{layout._BOARD_IDX_SLOTS} integer indices (positional, ALL_HABITATS × "
                "ROW_SLOTS). bird_index + 1; 0 = empty slot. Zero for non-board choices."
            ),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="bird_id",
            description=(
                f"Bird identity: one-hot (single bird) or multi-hot (kept set) "
                f"over all {layout._BIRD_ID_DIM} core-set birds."
            ),
            offset=layout._OFF_BIRD_ID,
            size=layout._BIRD_ID_DIM,
            encoding="one-hot / multi-hot",
            value_range="{0, 1}",
            notes=(
                "Embedded through the shared card table (same weights as state "
                "board/tray slots). For a setup pick the kept-set multi-hot is summed "
                "through the embedding. Zero for non-bird choices."
            ),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="bonus_id",
            description=(
                f"Bonus-card identity one-hot over all {layout._BONUS_ID_DIM} "
                "core-set bonus cards."
            ),
            offset=layout._OFF_BONUS_ID,
            size=layout._BONUS_ID_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes="Used for BonusCardChoice / a setup pick's kept bonus. Zero otherwise.",
        )
    )

    stripes.append(
        StripeDescriptor(
            name="bonus_delta",
            description=(
                "Per-candidate bonus contribution: how much this bird advances "
                "the deciding player's held bonus cards."
            ),
            offset=layout._OFF_BONUS_DELTA,
            size=layout._BONUS_DELTA_DIM,
            encoding="vector",
            value_range="[0, ~1]",
            notes=(
                f"{layout._BONUS_DELTA_DIM} values: qual_count (held bonus cards "
                "this bird qualifies for, ÷5), stepped_delta (summed stepped-VP "
                "gain from the +1 qualifying bird, ÷7), linear_delta (same, "
                "piecewise-linear, ÷7). Filled for play / keep-bird / tray "
                "draw-source candidates; zero otherwise."
            ),
            sub_fields=_bonus_delta_sub_fields(),
        )
    )

    stripes.append(
        StripeDescriptor(
            name="goal_delta",
            description=(
                "Per-candidate round-goal contribution: for each of the 4 round "
                "goals, how much this bird would change the deciding player's "
                "category count and placement VP."
            ),
            offset=layout._OFF_GOAL_DELTA,
            size=layout._GOAL_DELTA_DIM,
            encoding="vector",
            value_range="[0, ~1]",
            notes=(
                f"{layout._GOAL_DELTA_DIM} values: 4 goal slots × 2 scalars. "
                "Per slot: count_delta (÷5, always 0 or 0.2), vp_delta "
                "(÷10, marginal placement VP swing). Filled for play / keep-bird "
                "/ tray draw-source candidates; zero otherwise."
            ),
            sub_fields=_goal_delta_sub_fields(),
        )
    )

    end = layout._OFF_GOAL_DELTA + layout._GOAL_DELTA_DIM

    # ---- setup_agg (trailing; present only when the main model carries setup) ----
    if spec.include_setup:
        stripes.append(
            StripeDescriptor(
                name="setup_agg",
                description=(
                    "Aggregate statistics of the kept-card subset for a SetupChoice."
                ),
                offset=layout._OFF_SETUP,
                size=layout._SETUP_DIM,
                encoding="vector",
                value_range="[0, ~1]",
                notes=(
                    f"{layout._SETUP_DIM} values: summed_points (÷45), summed_food_cost "
                    "(÷35), summed_egg_limit (÷30), kept_count (÷5). Present only when "
                    "use_setup_model is off (the main net scores the opening); zero for "
                    "non-setup choices."
                ),
                sub_fields=_setup_agg_sub_fields(),
            )
        )
        end += layout._SETUP_DIM

    assert (
        end == total
    ), f"choice stripe offsets end at {end} but choice_feature_dim(spec) = {total}"

    raw = VectorLayout(total_size=total, stripes=tuple(stripes))
    return _embed_layout(
        raw,
        _choice_embed_rules(card_embed_dim),
        layout.choice_input_dim(total, card_embed_dim),
    )


###### PRIVATE #######

#### Post-embedding view ####


class _EmbedRule(pydantic.BaseModel):
    """How a raw card-index / identity stripe is shown at its post-embedding width."""

    new_size: int
    encoding: str
    value_range: str
    notes: str


def _embed_layout(
    raw: VectorLayout, rules: dict[str, _EmbedRule], expected_total: int
) -> VectorLayout:
    """Rewrite a raw vector layout into the network's post-embedding input view.

    Every card-index / identity stripe named in ``rules`` is replaced by its
    embedded-width stripe and all offsets are recomputed cumulatively (sizes change,
    so downstream offsets shift). A rule with ``new_size == 0`` *removes* its
    stripe — the raw dims were folded into another block (the hand summary
    redirected into the hand encoder). The result's total must equal
    ``expected_total`` — the trunk / choice-encoder first-``Linear`` input width.
    """
    stripes: list[StripeDescriptor] = []
    off = 0
    for stripe in raw.stripes:
        rule = rules.get(stripe.name)
        if rule is None:
            stripes.append(stripe.model_copy(update={"offset": off}))
            off += stripe.size
            continue
        if rule.new_size == 0:
            continue
        stripes.append(
            stripe.model_copy(
                update={
                    "offset": off,
                    "size": rule.new_size,
                    "encoding": rule.encoding,
                    "value_range": rule.value_range,
                    "notes": rule.notes,
                }
            )
        )
        off += rule.new_size
    assert off == expected_total, (
        f"embedded stripe offsets sum to {off} but expected {expected_total} — "
        "stripes.py expansion is out of sync with layout.trunk/choice_input_dim"
    )
    return VectorLayout(total_size=expected_total, stripes=tuple(stripes))


def _state_embed_rules(
    card_embed_dim: int,
    *,
    use_distinct_hand_model: bool = False,
    hand_embed_dim: int | None = None,
    tray_set_embedding: bool = False,
) -> dict[str, _EmbedRule]:
    """The card-index / hand stripes of the state vector, at embedded width."""
    n_board = layout.N_BOARD_INDEX_SLOTS
    tray = state.TRAY_SIZE
    hand = layout.HAND_MULTIHOT_DIM
    hand_width = (
        (hand_embed_dim if hand_embed_dim is not None else card_embed_dim)
        if use_distinct_hand_model
        else card_embed_dim
    )
    rules = {
        "card_idx_board": _EmbedRule(
            new_size=n_board * card_embed_dim,
            encoding="card-embedding",
            value_range="learned",
            notes=(
                f"{n_board} board slots (15 me + 15 opp) -> one {card_embed_dim}-dim "
                f"shared card embedding each ({n_board}x{card_embed_dim}). Raw encoding "
                "stores 30 integer indices (bird_index + 1; 0 = empty)."
            ),
        ),
        "card_idx_tray": _EmbedRule(
            new_size=tray * card_embed_dim,
            encoding="card-embedding",
            value_range="learned",
            notes=(
                f"{tray} tray slots -> one {card_embed_dim}-dim shared card embedding "
                f"each ({tray}x{card_embed_dim}). Raw encoding stores {tray} indices."
            ),
        ),
        "hand_multihot": _EmbedRule(
            new_size=card_embed_dim,
            encoding="card-embedding (mean-pooled)",
            value_range="learned",
            notes=(
                f"My hand -> one {card_embed_dim}-dim embedding, mean-pooled over the "
                f"held cards' shared card vectors. Raw encoding is a {hand}-wide "
                "multi-hot over all core birds."
            ),
        ),
    }
    if use_distinct_hand_model:
        # The dedicated hand encoder consumes [multi-hot ⊕ hand summary]: the
        # hand stripe becomes the encoder's N-wide output and the 10-dim
        # hand-summary stripe folds into its input (dropped from the trunk view).
        rules["hand_multihot"] = _EmbedRule(
            new_size=hand_width,
            encoding="card-set-embedding (hand encoder)",
            value_range="learned",
            notes=(
                f"My hand -> one {hand_width}-dim set embedding from the dedicated "
                f"hand encoder over [multi-hot ({hand}) ⊕ the redirected 10-dim "
                "hand summary]. Raw encoding is the multi-hot plus the (separate) "
                "hand_summary_me stripe."
            ),
        )
        rules["hand_summary_me"] = _EmbedRule(
            new_size=0,
            encoding="folded",
            value_range="-",
            notes=(
                "Redirected into the hand encoder's input (see hand_multihot); "
                "no longer a direct trunk input."
            ),
        )
    if tray_set_embedding:
        rules["card_idx_tray"] = _EmbedRule(
            new_size=tray * card_embed_dim + hand_width,
            encoding="card-embedding + card-set-embedding",
            value_range="learned",
            notes=(
                f"{tray} tray slots -> one {card_embed_dim}-dim shared card embedding "
                f"each ({tray}x{card_embed_dim}) plus one {hand_width}-dim tray-*set* "
                "embedding from the hand encoder (multi-hot + summary derived "
                f"in-model from the index columns). Raw encoding stores {tray} "
                "indices."
            ),
        )
    return rules


def _choice_embed_rules(card_embed_dim: int) -> dict[str, _EmbedRule]:
    """The board-index / bird-identity stripes of the choice vector, embedded."""
    slots = layout.CHOICE_BOARD_IDX_SLOTS
    birds = layout.CHOICE_BIRD_ID_DIM
    return {
        "board_idx": _EmbedRule(
            new_size=slots * card_embed_dim,
            encoding="card-embedding",
            value_range="learned",
            notes=(
                f"{slots} board slots -> one {card_embed_dim}-dim shared card embedding "
                f"each ({slots}x{card_embed_dim}). Raw encoding stores {slots} integer "
                "indices (bird_index + 1; 0 = empty)."
            ),
        ),
        "bird_id": _EmbedRule(
            new_size=card_embed_dim,
            encoding="card-embedding (candidate)",
            value_range="learned",
            notes=(
                f"Candidate bird -> one {card_embed_dim}-dim shared card embedding (a "
                f"setup pick's kept set sums their vectors). Raw encoding is a {birds}-"
                "wide one-hot / multi-hot over all core birds."
            ),
        ),
    }


#### State sub-field builders ####


def _food_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """One sub-field per food type (inventory or birdfeeder face count)."""
    return tuple(
        SubFieldDescriptor(
            name=f"food_{food.value}",
            description=f"Count of {food.value}.",
            relative_offset=idx,
            size=1,
            encoding="scalar",
            value_range="[0, ~1.7]",
            notes="Normalized ÷ 6.",
        )
        for idx, food in enumerate(cards.ALL_FOODS)
    )


def _board_slot_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """All per-element sub-fields for a board continuous stripe.

    Iterates all 15 slots (3 habitats × 5 positions) in the same order the
    encoder writes them. Each slot contributes 9 elements; the ``group`` field
    names the slot so the HTML report can nest them.
    """
    food_names = [food.value for food in cards.ALL_FOODS]

    slot_dim_meta = [
        ("eggs", "Eggs currently on this bird.", "Normalized ÷ 6."),
        (
            "egg_cap_remaining",
            "Remaining egg capacity of this bird.",
            "Normalized ÷ 6.",
        ),
        *[
            (
                f"cached_{food}",
                f"Cached {food} food count.",
                "Normalized ÷ 6.",
            )
            for food in food_names
        ],
        ("tucked", "Number of tucked cards under this bird.", "Normalized ÷ 6."),
        (
            "activations",
            "Times this bird has been activated this round.",
            "Normalized ÷ 4.",
        ),
    ]

    sub_fields: list[SubFieldDescriptor] = []
    slot_number = 0
    for habitat in cards.ALL_HABITATS:
        for position in range(state.ROW_SLOTS):
            group = f"slot_{habitat.value}_{position}"
            slot_base = slot_number * layout._SLOT_MUT_DIM
            for dim_idx, (dim_name, dim_desc, dim_notes) in enumerate(slot_dim_meta):
                sub_fields.append(
                    SubFieldDescriptor(
                        name=f"{habitat.value}_{position}.{dim_name}",
                        description=dim_desc,
                        relative_offset=slot_base + dim_idx,
                        size=1,
                        encoding="scalar",
                        value_range="[0, ~1]",
                        notes=dim_notes,
                        group=group,
                    )
                )
            slot_number += 1
    return tuple(sub_fields)


def _board_summary_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """18 sub-fields for a board-summary stripe (3 habitats × 6 named stats)."""
    stats = [
        (
            "row_length",
            "Number of occupied slots in this habitat row.",
            "Normalized ÷ 5 (max slots).",
        ),
        ("total_eggs", "Total eggs on all birds in this habitat.", "Normalized ÷ 6."),
        (
            "total_points",
            "Total point value of birds in this habitat.",
            "Normalized ÷ 45 (9 pts × 5 slots).",
        ),
        (
            "total_tucked",
            "Total tucked cards across all birds in this habitat.",
            "Normalized ÷ 6.",
        ),
        (
            "total_cached_food",
            "Total cached food units across all birds in this habitat.",
            "Normalized ÷ 6.",
        ),
        (
            "brown_bird_count",
            "Number of brown-power birds in this habitat.",
            "Normalized ÷ 5 (max slots).",
        ),
    ]
    sub_fields: list[SubFieldDescriptor] = []
    for hab_idx, habitat in enumerate(cards.ALL_HABITATS):
        base = hab_idx * len(stats)
        for stat_idx, (stat_name, stat_desc, stat_notes) in enumerate(stats):
            sub_fields.append(
                SubFieldDescriptor(
                    name=f"{habitat.value}.{stat_name}",
                    description=f"{stat_desc}",
                    relative_offset=base + stat_idx,
                    size=1,
                    encoding="scalar",
                    value_range="[0, ~1]",
                    notes=stat_notes,
                    group=habitat.value,
                )
            )
    return tuple(sub_fields)


def _hand_summary_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """10 sub-fields for the hand-summary stripe: size, habitat counts, food multi-hot."""
    entries: list[tuple[str, str, str]] = [
        ("hand_size", "Total cards currently in hand.", "Normalized ÷ 10."),
        *[
            (
                f"{habitat.value}_count",
                f"Number of hand birds that live in {habitat.value} "
                "(a dual-habitat bird counts in each).",
                "Normalized ÷ 10.",
            )
            for habitat in cards.ALL_HABITATS
        ],
        *[
            (
                f"has_{food.value}_cost",
                f"1.0 if any hand bird has {food.value} in its food cost.",
                "{0, 1}.",
            )
            for food in cards.ALL_FOODS
        ],
        (
            "has_wild_cost",
            "1.0 if any hand bird has a wild token in its cost.",
            "{0, 1}.",
        ),
    ]
    return tuple(
        SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes=notes,
        )
        for idx, (name, desc, notes) in enumerate(entries)
    )


def _birdfeeder_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """6 sub-fields for the birdfeeder stripe (one per food face + choice die)."""
    sub_fields: list[SubFieldDescriptor] = []
    for idx, food in enumerate(cards.ALL_FOODS):
        sub_fields.append(
            SubFieldDescriptor(
                name=f"face_{food.value}",
                description=f"Dice showing a {food.value} face in the birdfeeder.",
                relative_offset=idx,
                size=1,
                encoding="scalar",
                value_range="[0, 1]",
                notes="Normalized ÷ 5 (max dice showing that face).",
            )
        )
    sub_fields.append(
        SubFieldDescriptor(
            name="face_choice_die",
            description="Dice showing a choice-wild (invertebrate/seed) face.",
            relative_offset=len(sub_fields),
            size=1,
            encoding="scalar",
            value_range="[0, 1]",
            notes="Normalized ÷ 5.",
        )
    )
    return tuple(sub_fields)


def _misc_scalars_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """7 sub-fields for the misc-scalars stripe."""
    entries = [
        ("round_index", "Current round number (0–3), ordinal.", "Normalized ÷ 3."),
        ("my_action_cubes", "My remaining action cubes this round.", "Normalized ÷ 8."),
        (
            "opp_action_cubes",
            "Opponent remaining action cubes this round.",
            "Normalized ÷ 8.",
        ),
        (
            "my_round_goal_pts",
            "My accumulated round-goal VP so far.",
            "Normalized ÷ 10.",
        ),
        (
            "opp_round_goal_pts",
            "Opponent accumulated round-goal VP so far.",
            "Normalized ÷ 10.",
        ),
        (
            "tray_size",
            "Number of face-up cards currently in the tray.",
            "Normalized ÷ 3.",
        ),
        (
            "deck_size",
            "Number of cards remaining in the draw deck.",
            "Normalized ÷ 100.",
        ),
    ]
    return tuple(
        SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes=notes,
        )
        for idx, (name, desc, notes) in enumerate(entries)
    )


def _round_goals_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """16 logical sub-field groups for the round-goals stripe (4 rounds × 4 groups).

    Each round contributes a category one-hot block (size 20) plus three
    individual scalars for my count, opponent count, and current placement VP.
    """
    sub_fields: list[SubFieldDescriptor] = []
    for round_idx in range(layout._NUM_ROUNDS):
        base = round_idx * layout._ROUND_GOAL_SLOT_DIM
        group = f"round_{round_idx}"
        sub_fields.append(
            SubFieldDescriptor(
                name=f"round_{round_idx}.category",
                description=(
                    f"Round {round_idx} goal category "
                    f"(one-hot over {layout.MAX_GOAL_CATEGORIES} categories)."
                ),
                relative_offset=base,
                size=layout.MAX_GOAL_CATEGORIES,
                encoding="one-hot",
                value_range="{0, 1}",
                notes=(
                    f"Categories in index order: "
                    f"{', '.join(layout.GOAL_CATEGORIES)}."
                ),
                group=group,
            )
        )
        sub_fields.append(
            SubFieldDescriptor(
                name=f"round_{round_idx}.my_count",
                description=f"Round {round_idx}: my current count toward the goal.",
                relative_offset=base + layout._ROUND_GOAL_MY_COUNT,
                size=1,
                encoding="scalar",
                value_range="[0, ~1]",
                notes="Normalized ÷ 5.",
                group=group,
            )
        )
        sub_fields.append(
            SubFieldDescriptor(
                name=f"round_{round_idx}.opp_count",
                description=f"Round {round_idx}: opponent's current count toward the goal.",
                relative_offset=base + layout._ROUND_GOAL_OPP_COUNT,
                size=1,
                encoding="scalar",
                value_range="[0, ~1]",
                notes="Normalized ÷ 5.",
                group=group,
            )
        )
        sub_fields.append(
            SubFieldDescriptor(
                name=f"round_{round_idx}.placement_vp",
                description=f"Round {round_idx}: VP I would receive at my current standing.",
                relative_offset=base + layout._ROUND_GOAL_VP,
                size=1,
                encoding="scalar",
                value_range="[0, ~1]",
                notes="Normalized ÷ 10.",
                group=group,
            )
        )
    return tuple(sub_fields)


#### Choice sub-field builders ####


def _kind_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """6 sub-fields for the choice-kind one-hot."""
    kind_meta = [
        ("bird", "This choice is a bird card pick."),
        ("food", "This choice is a food type pick."),
        ("habitat", "This choice is a habitat pick."),
        ("payment", "This choice is a food payment specification."),
        ("board_target", "This choice targets a specific board slot."),
        (
            "special",
            "This choice is a special action (skip, main action, bonus, or setup).",
        ),
    ]
    return tuple(
        SubFieldDescriptor(
            name=f"kind_{name}",
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="one-hot bit",
            value_range="{0, 1}",
        )
        for idx, (name, desc) in enumerate(kind_meta)
    )


def _gain_food_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """7 sub-fields for the gain_food stripe: 5 plain foods + 2 choice-die options."""
    entries: list[tuple[str, str]] = [
        *[
            (f"gain_{food.value}", f"Take a plain {food.value} die.")
            for food in cards.ALL_FOODS
        ],
        (
            "choice_die_invertebrate",
            "Take the invertebrate/seed choice die as invertebrate.",
        ),
        ("choice_die_seed", "Take the invertebrate/seed choice die as seed."),
    ]
    return tuple(
        SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="one-hot bit",
            value_range="{0, 1}",
        )
        for idx, (name, desc) in enumerate(entries)
    )


def _choice_habitat_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """3 sub-fields for the habitat one-hot in a choice vector."""
    return tuple(
        SubFieldDescriptor(
            name=f"habitat_{hab.value}",
            description=f"This choice targets the {hab.value} habitat.",
            relative_offset=idx,
            size=1,
            encoding="one-hot bit",
            value_range="{0, 1}",
        )
        for idx, hab in enumerate(cards.ALL_HABITATS)
    )


def _choice_payment_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """5 sub-fields for the pay_food vector stripe in a choice vector."""
    return tuple(
        SubFieldDescriptor(
            name=f"pay_{food.value}",
            description=f"Units of {food.value} food paid in this choice.",
            relative_offset=idx,
            size=1,
            encoding="scalar",
            value_range="[0, 1]",
            notes="Normalized ÷ 4.",
        )
        for idx, food in enumerate(cards.ALL_FOODS)
    )


def _board_target_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """The per-slot sub-fields for the board_target stripe (15 slots × 8 scalars)."""
    food_names = [food.value for food in cards.ALL_FOODS]
    slot_meta: list[tuple[str, str]] = [
        ("lay_eggs", "Set when this choice would lay an egg on this slot."),
        ("pay_eggs", "Set when this choice would remove (pay) an egg from this slot."),
        *[
            (f"cached_{food}", f"Cached {food} on the bird in this slot.")
            for food in food_names
        ],
        ("tucked", "Tucked cards under the bird in this slot."),
    ]
    sub_fields: list[SubFieldDescriptor] = []
    slot_number = 0
    for habitat in cards.ALL_HABITATS:
        for position in range(state.ROW_SLOTS):
            group = f"slot_{habitat.value}_{position}"
            slot_base = slot_number * layout._BT_SLOT_SCALARS
            for dim_idx, (dim_name, dim_desc) in enumerate(slot_meta):
                sub_fields.append(
                    SubFieldDescriptor(
                        name=f"{habitat.value}_{position}.{dim_name}",
                        description=dim_desc,
                        relative_offset=slot_base + dim_idx,
                        size=1,
                        encoding="scalar",
                        value_range="[0, ~1]",
                        notes="Cached food / tucked normalized ÷ 6; flags {0, 1}.",
                        group=group,
                    )
                )
            slot_number += 1
    return tuple(sub_fields)


def _main_action_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """4 sub-fields for the main-action one-hot in a choice vector."""
    return tuple(
        SubFieldDescriptor(
            name=f"action_{action.value}",
            description=f"This choice picks the {action.value} main action.",
            relative_offset=idx,
            size=1,
            encoding="one-hot bit",
            value_range="{0, 1}",
        )
        for idx, action in enumerate(layout._MAIN_ACTION_ORDER)
    )


def _special_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """2 sub-fields for the special-flags stripe in a choice vector."""
    entries = [
        ("is_skip", "Set when this choice declines the current decision."),
        (
            "is_self",
            "Set on the PlayerIdChoice option that is the active player "
            "(the Hummingbird food-gain order pick).",
        ),
    ]
    return tuple(
        SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="binary",
            value_range="{0, 1}",
        )
        for idx, (name, desc) in enumerate(entries)
    )


def _exchange_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """11 sub-fields for the symmetric exchange stripe: a 7-field self block (what
    the deciding player pays / gains) then a 4-field opponent-gain block."""
    entries = [
        ("cards_to_discard", "Cards discarded from hand as payment."),
        ("food_to_pay", "Food paid (magnitude; the type rides the pay_food stripe)."),
        ("eggs_to_pay", "Eggs removed as payment."),
        ("food_to_gain", "Food gained from the supply."),
        ("eggs_to_gain", "Eggs laid."),
        ("cards_to_draw", "Cards drawn into hand."),
        ("cards_to_tuck", "Cards tucked under a bird."),
        ("opp_food_to_gain", "Food the opponent also gains (shared-benefit power)."),
        ("opp_eggs_to_gain", "Eggs the opponent also lays."),
        ("opp_cards_to_draw", "Cards the opponent also draws."),
        ("opp_cards_to_tuck", "Cards the opponent also tucks."),
    ]
    return tuple(
        SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes="Normalized ÷ 3.",
        )
        for idx, (name, desc) in enumerate(entries)
    )


def _bonus_delta_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """3 sub-fields for the per-candidate bonus-contribution stripe."""
    entries = [
        (
            "qual_count",
            "Held bonus cards this candidate bird qualifies for.",
            "Normalized ÷ 5.",
        ),
        (
            "stepped_delta",
            "Summed stepped-VP gain from the +1 qualifying bird.",
            "Normalized ÷ 7.",
        ),
        (
            "linear_delta",
            "Summed piecewise-linear-VP gain from the +1 qualifying bird.",
            "Normalized ÷ 7.",
        ),
    ]
    return tuple(
        SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes=notes,
        )
        for idx, (name, desc, notes) in enumerate(entries)
    )


def _goal_delta_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """8 sub-fields for the per-candidate round-goal delta stripe (4 slots × 2)."""
    entries: list[tuple[str, str, str]] = []
    for goal_idx in range(4):
        entries.append(
            (
                f"goal_{goal_idx}_count_delta",
                f"Count change on round-{goal_idx + 1} goal from playing this bird.",
                "Normalized ÷ 5. Always 0 or 0.2.",
            )
        )
        entries.append(
            (
                f"goal_{goal_idx}_vp_delta",
                f"Placement VP swing on round-{goal_idx + 1} goal from playing this bird.",
                "Normalized ÷ 10.",
            )
        )
    return tuple(
        SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes=notes,
        )
        for idx, (name, desc, notes) in enumerate(entries)
    )


def _setup_agg_sub_fields() -> tuple[SubFieldDescriptor, ...]:
    """4 sub-fields for the setup-aggregate stripe in a choice vector."""
    entries = [
        (
            "summed_points",
            "Total point value of the kept setup cards.",
            "Normalized ÷ 45.",
        ),
        (
            "summed_food_cost",
            "Total food cost of the kept setup cards.",
            "Normalized ÷ 35.",
        ),
        (
            "summed_egg_limit",
            "Total egg capacity of the kept setup cards.",
            "Normalized ÷ 30.",
        ),
        ("kept_count", "Number of cards kept in this setup choice.", "Normalized ÷ 5."),
    ]
    return tuple(
        SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes=notes,
        )
        for idx, (name, desc, notes) in enumerate(entries)
    )
