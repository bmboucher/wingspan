# pyright: reportPrivateUsage=false
# (reads the shared, package-private layout constants — deliberate intra-package
# coupling identical to state_encode.py's convention)
"""Choice-vector stripe layout and its sub-field builders.

``choice_stripe_layout`` returns a :class:`~descriptors.VectorLayout` listing
every stripe in the per-choice encoder's input vector in offset order, with a
post-embedding rewrite applied so the totals match the choice encoder's
first-``Linear`` input width. ``raw_choice_stripe_layout`` is the pre-rewrite
registry at the encoder-output widths — the compat layout shims reuse its
era-shared stripes.
"""

from __future__ import annotations

from wingspan import architecture, cards, state
from wingspan.encode import layout
from wingspan.encode.stripes import descriptors, embed_rules

_DEFAULT_CARD_EMBED_DIM = architecture.ModelArchitecture().card_embed_dim


def choice_stripe_layout(
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
    card_embed_dim: int = _DEFAULT_CARD_EMBED_DIM,
    *,
    has_becomes_playable: bool = True,
) -> descriptors.VectorLayout:
    """Build the stripe registry for the per-choice encoder's input vector.

    Each stripe is a type-specific feature group every candidate is encoded into.
    The board-index block, the bird-index column, and the kept-set multi-hot are
    shown at their *post-embedding* width — each board slot as one
    ``card_embed_dim`` vector, the candidate as one, the kept set as one — so the
    breakdown sums to the choice encoder's first-``Linear`` input
    (``layout.choice_input_dim``), what the network actually sees. The trailing
    ``setup_agg`` / ``kept_multihot`` stripes are present only when
    ``spec.include_setup``.  ``has_becomes_playable`` controls whether the
    ``becomes_playable`` stripe is included; pass ``False`` for pre-0.6 artifacts
    that predate that stripe.
    """
    raw = raw_choice_stripe_layout(spec, has_becomes_playable=has_becomes_playable)
    return embed_rules.embed_layout(
        raw,
        embed_rules.choice_embed_rules(card_embed_dim),
        layout.choice_input_dim(
            raw.total_size,
            card_embed_dim,
            include_setup=spec.include_setup,
            has_becomes_playable=has_becomes_playable,
        ),
    )


def raw_choice_stripe_layout(
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
    *,
    has_becomes_playable: bool = True,
) -> descriptors.VectorLayout:
    """Build the *raw* (pre-embedding) stripe registry for the choice vector.

    Stripes appear at the exact offsets and widths ``encode_choices`` writes —
    the encoder-output view, before the card-index / kept-set stripes are
    rewritten to their embedded widths. :func:`choice_stripe_layout` applies
    that rewrite; the compat layout shims (``wingspan.compat``) instead reuse
    the era-shared stripes of this raw registry at their frozen offsets.
    ``has_becomes_playable`` controls whether the ``becomes_playable`` stripe is
    included; pass ``False`` for pre-0.6 artifacts that predate that stripe.
    """
    total = layout.choice_feature_dim(spec)
    if not has_becomes_playable:
        total -= layout.CHOICE_BECOMES_PLAYABLE_DIM
    food_names = ", ".join(f.value for f in cards.ALL_FOODS)

    stripes: list[descriptors.StripeDescriptor] = []

    main_action_names = ", ".join(a.value for a in layout._MAIN_ACTION_ORDER)
    kind_labels = (
        "bird(0), food(1), habitat(2), payment(3), board_target(4), special(5)"
    )

    stripes.append(
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
            name="board_target",
            description="Per-board-slot features for a board-target (lay/remove egg) choice.",
            offset=layout._OFF_BOARD,
            size=layout._BOARD_TARGET_DIM,
            encoding="complex",
            value_range="[0, ~1]",
            notes=(
                f"{layout._SLOTS_PER_BOARD} board slots × {layout._BT_SLOT_SCALARS} "
                "scalars each: lay_eggs[0], pay_eggs[1] (set on the targeted slot "
                "for a lay-egg vs remove-egg decision), cached_total[2] (summed "
                "cached food ÷6), tucked[3] (÷6). The targeted slot's occupant "
                "rides bird_id; location rides board_hab/board_col. Zero for "
                "non-board-target choices."
            ),
            sub_fields=_board_target_sub_fields(),
        )
    )

    stripes.append(
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
            name="board_hab",
            description=(
                "One-hot habitat of the single board slot relevant to this choice "
                "(landing slot for placements, targeted slot for lay/remove-egg, "
                "current slot for move-bird)."
            ),
            offset=layout._OFF_BOARD_HAB,
            size=layout._BOARD_HAB_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=(
                f"{layout._BOARD_HAB_DIM} dims, indexed by cards.ALL_HABITATS order. "
                "Zero for choices with no board-slot signal."
            ),
            sub_fields=_board_hab_sub_fields(),
        )
    )

    stripes.append(
        descriptors.StripeDescriptor(
            name="board_col",
            description=(
                "One-hot column (0–4) within the habitat row of the single board slot "
                "relevant to this choice."
            ),
            offset=layout._OFF_BOARD_COL,
            size=layout._BOARD_COL_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=(
                f"{layout._BOARD_COL_DIM} dims, indexed by column within the habitat "
                "row (0 = leftmost occupied slot). Zero for choices with no board-slot "
                "signal."
            ),
            sub_fields=_board_col_sub_fields(),
        )
    )

    stripes.append(
        descriptors.StripeDescriptor(
            name="bird_id",
            description=(
                "The candidate or board-target occupant's bird index, looked up in "
                "the shared card table."
            ),
            offset=layout._OFF_BIRD_ID,
            size=layout._CHOICE_BIRD_ID_DIM,
            encoding="integer-index",
            value_range=f"int 0–{cards.n_birds()}",
            notes=(
                "bird_index + 1; 0 = no bird (the model zeroes the embedding so "
                "non-bird rows contribute nothing). For board-target choices the "
                "targeted slot's occupant rides this column; for placement rows the "
                "candidate bird does. Same embedding weights as state board/tray "
                "slots. A setup pick's kept *set* rides the trailing kept_multihot "
                "stripe instead."
            ),
        )
    )

    stripes.append(
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
            name="bonus_delta",
            description=(
                "Per-candidate bonus contribution: how much this choice moves "
                "the deciding player's held bonus cards (static categories and "
                "the dynamic egg / hand / habitat-spread cards alike)."
            ),
            offset=layout._OFF_BONUS_DELTA,
            size=layout._BONUS_DELTA_DIM,
            encoding="vector",
            value_range="[-~1, ~1]",
            notes=(
                f"{layout._BONUS_DELTA_DIM} values: qual_count (held bonus cards "
                "this choice moves, ÷5), stepped_delta (summed stepped-VP "
                "swing, ÷7, signed), linear_delta (same, piecewise-linear, ÷7). "
                "Filled for play / keep-bird / tray draw-source candidates "
                "(+1 board or hand qualifier), egg lay / removal board targets "
                "(egg-threshold crossings), move-bird habitat rows (habitat "
                "spread), and accept / main-action rows committing a net hand "
                "change; zero otherwise."
            ),
            sub_fields=_bonus_delta_sub_fields(),
        )
    )

    stripes.append(
        descriptors.StripeDescriptor(
            name="goal_delta",
            description=(
                "Per-candidate round-goal contribution: for each of the 4 round "
                "goals, how much this choice would change the deciding player's "
                "category count and placement VP. Already-scored rounds stay "
                "zero — their payouts are frozen."
            ),
            offset=layout._OFF_GOAL_DELTA,
            size=layout._GOAL_DELTA_DIM,
            encoding="vector",
            value_range="[-~1, ~1]",
            notes=(
                f"{layout._GOAL_DELTA_DIM} values: 4 goal slots × 2 scalars. "
                "Per slot: count_delta (÷5, signed), vp_delta (÷10, marginal "
                "placement VP swing). Filled for play / keep-bird / tray "
                "draw-source candidates (exact bird delta), egg lay / removal "
                "board targets (exact egg delta), move-bird habitat rows "
                "(exact move delta), and lay/draw commitment rows (accept "
                "trades, the LAY_EGGS main action: capacity-capped optimistic "
                "bound); zero otherwise and for scored rounds."
            ),
            sub_fields=_goal_delta_sub_fields(),
        )
    )

    stripes.append(
        descriptors.StripeDescriptor(
            name="bonus_value",
            description=(
                "Per-candidate bonus-CARD value: what the offered bonus card is "
                "worth to the deciding player, now and in potential."
            ),
            offset=layout._OFF_BONUS_VALUE,
            size=layout._BONUS_VALUE_DIM,
            encoding="vector",
            value_range="[0, ~1]",
            notes=(
                f"{layout._BONUS_VALUE_DIM} values: qual_count (board birds "
                "qualifying for this bonus, ÷5), stepped_vp (VP the card pays at "
                "that count, ÷7), linear_vp (same, piecewise-linear, ÷7), "
                "hand_potential (hand/kept birds qualifying, ÷5), tray_potential "
                "(tray birds qualifying, ÷5). Filled for BonusCardChoice and a "
                "setup pick's kept bonus; zero otherwise."
            ),
            sub_fields=_bonus_value_sub_fields(),
        )
    )

    end = layout._OFF_BONUS_VALUE + layout._BONUS_VALUE_DIM

    # ---- becomes_playable (v0.6+: 180-dim multi-hot in the base spec) ----
    if has_becomes_playable:
        stripes.append(
            descriptors.StripeDescriptor(
                name="becomes_playable",
                description=(
                    "Multi-hot of hand birds that would become playable as a consequence "
                    "of accepting this choice (e.g. gaining the food or eggs that unlock "
                    "a bird from hand). Zero when the choice has no such effect."
                ),
                offset=layout.CHOICE_BECOMES_PLAYABLE_OFFSET,
                size=layout.CHOICE_BECOMES_PLAYABLE_DIM,
                encoding="multi-hot",
                value_range="{0, 1}",
                notes=(
                    f"Indexed by stable bird order from cards.bird_index() "
                    f"({layout.CHOICE_BECOMES_PLAYABLE_DIM} dims). Filled for "
                    "FoodChoice (GainFoodDecision context), MainActionChoice (GAIN_FOOD / "
                    "LAY_EGGS), and PayCostChoice rows that include a food or egg gain."
                ),
            )
        )
        end += layout.CHOICE_BECOMES_PLAYABLE_DIM

    # ---- setup stripes (trailing; present only when the main model carries setup) ----
    if spec.include_setup:
        stripes.append(
            descriptors.StripeDescriptor(
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
        stripes.append(
            descriptors.StripeDescriptor(
                name="kept_multihot",
                description=(
                    "Multi-hot of the specific birds a SetupChoice keeps, over "
                    f"all {layout._KEPT_MULTIHOT_DIM} core-set birds."
                ),
                offset=layout._OFF_KEPT_MULTIHOT,
                size=layout._KEPT_MULTIHOT_DIM,
                encoding="multi-hot",
                value_range="{0, 1}",
                notes=(
                    "Summed through the shared card table into one embedding (the "
                    "kept set is unordered). Present only when use_setup_model is "
                    "off (the main net scores the opening); zero for non-setup "
                    "choices."
                ),
            )
        )
        end += layout._KEPT_MULTIHOT_DIM

    assert (
        end == total
    ), f"choice stripe offsets end at {end} but choice_feature_dim(spec) = {total}"

    return descriptors.VectorLayout(total_size=total, stripes=tuple(stripes))


###### PRIVATE #######

#### Choice sub-field builders ####


def _kind_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
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
        descriptors.SubFieldDescriptor(
            name=f"kind_{name}",
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="one-hot bit",
            value_range="{0, 1}",
        )
        for idx, (name, desc) in enumerate(kind_meta)
    )


def _gain_food_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
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
        descriptors.SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="one-hot bit",
            value_range="{0, 1}",
        )
        for idx, (name, desc) in enumerate(entries)
    )


def _choice_payment_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """5 sub-fields for the pay_food vector stripe in a choice vector."""
    return tuple(
        descriptors.SubFieldDescriptor(
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


def _board_target_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """The per-slot sub-fields for the board_target stripe (15 slots × 4 scalars)."""
    slot_meta: list[tuple[str, str]] = [
        ("lay_eggs", "Set when this choice would lay an egg on this slot."),
        ("pay_eggs", "Set when this choice would remove (pay) an egg from this slot."),
        (
            "cached_total",
            "Total cached food on the bird in this slot (all types summed).",
        ),
        ("tucked", "Tucked cards under the bird in this slot."),
    ]
    sub_fields: list[descriptors.SubFieldDescriptor] = []
    slot_number = 0
    for habitat in cards.ALL_HABITATS:
        for position in range(state.ROW_SLOTS):
            group = f"slot_{habitat.value}_{position}"
            slot_base = slot_number * layout._BT_SLOT_SCALARS
            for dim_idx, (dim_name, dim_desc) in enumerate(slot_meta):
                sub_fields.append(
                    descriptors.SubFieldDescriptor(
                        name=f"{habitat.value}_{position}.{dim_name}",
                        description=dim_desc,
                        relative_offset=slot_base + dim_idx,
                        size=1,
                        encoding="scalar",
                        value_range="[0, ~1]",
                        notes="Cached total / tucked normalized ÷ 6; flags {0, 1}.",
                        group=group,
                    )
                )
            slot_number += 1
    return tuple(sub_fields)


def _board_hab_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """3 sub-fields for the board_hab one-hot stripe (habitat of the relevant slot)."""
    return tuple(
        descriptors.SubFieldDescriptor(
            name=f"hab_{habitat.value}",
            description=f"The relevant board slot is in the {habitat.value} row.",
            relative_offset=idx,
            size=1,
            encoding="one-hot bit",
            value_range="{0, 1}",
        )
        for idx, habitat in enumerate(cards.ALL_HABITATS)
    )


def _board_col_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """5 sub-fields for the board_col one-hot stripe (column of the relevant slot)."""
    return tuple(
        descriptors.SubFieldDescriptor(
            name=f"col_{col}",
            description=f"The relevant board slot is at column {col} in its habitat row.",
            relative_offset=col,
            size=1,
            encoding="one-hot bit",
            value_range="{0, 1}",
        )
        for col in range(state.ROW_SLOTS)
    )


def _main_action_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """4 sub-fields for the main-action one-hot in a choice vector."""
    return tuple(
        descriptors.SubFieldDescriptor(
            name=f"action_{action.value}",
            description=f"This choice picks the {action.value} main action.",
            relative_offset=idx,
            size=1,
            encoding="one-hot bit",
            value_range="{0, 1}",
        )
        for idx, action in enumerate(layout._MAIN_ACTION_ORDER)
    )


def _special_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
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
        descriptors.SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="binary",
            value_range="{0, 1}",
        )
        for idx, (name, desc) in enumerate(entries)
    )


def _exchange_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
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
        descriptors.SubFieldDescriptor(
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


def _bonus_delta_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """3 sub-fields for the per-candidate bonus-contribution stripe."""
    entries = [
        (
            "qual_count",
            "Held bonus cards whose qualifying count this choice moves.",
            "Normalized ÷ 5.",
        ),
        (
            "stepped_delta",
            "Summed stepped-VP swing from the qualifying-count change.",
            "Normalized ÷ 7. Signed.",
        ),
        (
            "linear_delta",
            "Summed piecewise-linear-VP swing from the qualifying-count change.",
            "Normalized ÷ 7. Signed.",
        ),
    ]
    return tuple(
        descriptors.SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="scalar",
            value_range="[-~1, ~1]",
            notes=notes,
        )
        for idx, (name, desc, notes) in enumerate(entries)
    )


def _goal_delta_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """8 sub-fields for the per-candidate round-goal delta stripe (4 slots × 2)."""
    entries: list[tuple[str, str, str]] = []
    for goal_idx in range(4):
        entries.append(
            (
                f"goal_{goal_idx}_count_delta",
                f"Count change on the round-{goal_idx + 1} goal from this choice.",
                "Normalized ÷ 5. Signed; zero once the round is scored.",
            )
        )
        entries.append(
            (
                f"goal_{goal_idx}_vp_delta",
                f"Placement VP swing on the round-{goal_idx + 1} goal from this choice.",
                "Normalized ÷ 10. Signed; zero once the round is scored.",
            )
        )
    return tuple(
        descriptors.SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="scalar",
            value_range="[-~1, ~1]",
            notes=notes,
        )
        for idx, (name, desc, notes) in enumerate(entries)
    )


def _bonus_value_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """5 sub-fields for the per-candidate bonus-card-value stripe."""
    entries = [
        (
            "qual_count",
            "Board birds qualifying for this candidate bonus card.",
            "Normalized ÷ 5.",
        ),
        (
            "stepped_vp",
            "Stepped VP the candidate bonus pays at the current board count.",
            "Normalized ÷ 7.",
        ),
        (
            "linear_vp",
            "Piecewise-linear VP of the candidate bonus at the current board count.",
            "Normalized ÷ 7.",
        ),
        (
            "hand_potential",
            "Hand (or setup kept-subset) birds qualifying for the candidate bonus.",
            "Normalized ÷ 5.",
        ),
        (
            "tray_potential",
            "Face-up tray birds qualifying for the candidate bonus.",
            "Normalized ÷ 5.",
        ),
    ]
    return tuple(
        descriptors.SubFieldDescriptor(
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


def _setup_agg_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
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
        descriptors.SubFieldDescriptor(
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
