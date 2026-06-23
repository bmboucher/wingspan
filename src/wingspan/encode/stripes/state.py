# pyright: reportPrivateUsage=false
# (reads the shared, package-private layout constants — deliberate intra-package
# coupling identical to state_encode.py's convention)
"""State-vector stripe layout and its sub-field builders.

``state_stripe_layout`` returns a :class:`~descriptors.VectorLayout` listing
every stripe in the state trunk's input vector in offset order, with sizes from
the ``layout`` constants and a post-embedding rewrite applied so the totals
match the trunk's first-``Linear`` input width.

``raw_state_stripe_layout`` returns the same layout without the post-embedding
rewrite — sizes and offsets match the flat vector that ``encode_state`` produces
(integer-index stripes at their raw widths, not ``card_embed_dim``).
"""

from __future__ import annotations

from wingspan import architecture, cards, decisions, state
from wingspan.encode import layout
from wingspan.encode.stripes import descriptors, embed_rules

_DEFAULT_CARD_EMBED_DIM = architecture.ModelArchitecture().card_embed_dim


def raw_state_stripe_layout(
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
) -> descriptors.VectorLayout:
    """Build the raw (pre-embedding) stripe registry for the state trunk's input vector.

    Like :func:`state_stripe_layout` but returns the encoder's *raw* output widths:
    card-index stripes appear as ``integer-index`` vectors (one slot per position)
    and the hand multi-hot appears at its full ``HAND_MULTIHOT_DIM`` width — the
    sizes and offsets the ``encode_state`` output actually has, not the post-embedding
    trunk view.  Use this when indexing into the vector that
    :meth:`~wingspan.model.PolicyValueNet.encode_state` returns directly."""
    return _build_raw_state_stripes(spec)


def state_stripe_layout(
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
    card_embed_dim: int = _DEFAULT_CARD_EMBED_DIM,
    *,
    use_distinct_hand_model: bool = False,
    hand_embed_dim: int | None = None,
    tray_set_embedding: bool = False,
) -> descriptors.VectorLayout:
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
    raw = _build_raw_state_stripes(spec)
    return embed_rules.embed_layout(
        raw,
        embed_rules.state_embed_rules(
            card_embed_dim,
            use_distinct_hand_model=use_distinct_hand_model,
            hand_embed_dim=hand_embed_dim,
            tray_set_embedding=tray_set_embedding,
        ),
        layout.trunk_input_dim(
            raw.total_size,
            card_embed_dim,
            use_distinct_hand_model=use_distinct_hand_model,
            hand_embed_dim=hand_embed_dim,
            tray_set_embedding=tray_set_embedding,
        ),
    )


###### PRIVATE #######

#### State stripe builder ####


def _build_raw_state_stripes(
    spec: layout.EncodingSpec,
) -> descriptors.VectorLayout:
    """Build all state stripes for ``spec`` without post-embedding rewrite.

    Called by both :func:`state_stripe_layout` (which then applies
    ``embed_rules``) and :func:`raw_state_stripe_layout` (which returns the raw
    view directly).  The returned layout's offsets and sizes match the flat
    vector that ``encode_state`` produces."""
    from wingspan.encode import state_encode

    total = state_encode.state_size(spec)
    food_names = ", ".join(f.value for f in cards.ALL_FOODS)
    habitat_names = ", ".join(h.value for h in cards.ALL_HABITATS)

    stripes: list[descriptors.StripeDescriptor] = []
    off = 0

    # ---- turn state (first stripe) ----
    turn_dim = layout.N_PLAYER_TURNS + 1  # 26 turn positions + is_first_player flag
    stripes.append(
        descriptors.StripeDescriptor(
            name="turn_state",
            description=(
                "Which of the player's 26 personal turns they are on, plus "
                "whether they go first in the current round."
            ),
            offset=off,
            size=turn_dim,
            encoding="complex",
            value_range="varies",
            notes=(
                f"{layout.N_PLAYER_TURNS + 1} values: "
                f"player_turn_one_hot[0:{layout.N_PLAYER_TURNS}] "
                f"({layout.N_PLAYER_TURNS} dims, all-zeros during setup), "
                f"is_first_player[{layout.N_PLAYER_TURNS}] "
                "(1.0 when me goes first in this round, 0.0 when second). "
                "Turn index = cumulative_cubes_offset[round] + "
                "(ROUND_CUBES[round] - action_cubes_left)."
            ),
            sub_fields=_turn_state_sub_fields(),
        )
    )
    off += turn_dim

    # ---- food inventory ----
    stripes.append(
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
            sub_fields=descriptors.hand_summary_sub_fields(),
        )
    )
    off += 10

    # ---- bonus progress (POV player only; opponent identity hidden) ----
    bonus_dim = layout._BONUS_ID_DIM  # 26 bonus cards
    stripes.append(
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
            name="birdfeeder",
            description=(
                "Birdfeeder die face counts (single-food faces and choice-wild "
                "dice) plus the reset-availability flag."
            ),
            offset=off,
            size=7,
            encoding="vector",
            value_range="[0, 1]",
            notes=(
                f"7 values: one per food type ({food_names}) for single-food faces, "
                "then the count of choice-die (wild) faces — each normalized ÷ 5 "
                "(max dice showing that face) — then a 0/1 flag set when every "
                "die shows the same face (the optional pre-gain reset is on offer)."
            ),
            sub_fields=_birdfeeder_sub_fields(),
        )
    )
    off += 7

    # ---- miscellaneous scalars ----
    misc_dim = 4  # goal pts ×2 + tray size + deck size
    stripes.append(
        descriptors.StripeDescriptor(
            name="misc_scalars",
            description="Miscellaneous game state (scores, deck).",
            offset=off,
            size=misc_dim,
            encoding="vector",
            value_range="[0, ~1]",
            notes=(
                "4 scalars: my_round_goal_pts (÷10), opp_round_goal_pts (÷10), "
                "tray_size (÷3), deck_size (÷100). "
                "Round and cube info moved to the leading turn_state stripe."
            ),
            sub_fields=_misc_scalars_sub_fields(),
        )
    )
    off += misc_dim

    # ---- round-goal state (all four rounds) ----
    rounds_dim = layout._ROUND_GOALS_STRIPE_DIM
    goal_slot = layout._ROUND_GOAL_SLOT_DIM  # MAX_GOAL_CATEGORIES + 3 = 23
    stripes.append(
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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
        descriptors.StripeDescriptor(
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

    # ---- hand-playability multi-hots (v0.6+: two 180-dim stripes) ----
    stripes.append(
        descriptors.StripeDescriptor(
            name="hand_playable_me",
            description=(
                f"Multi-hot of my hand birds that are playable right now "
                "(food affordable, at least one open habitat slot, egg cost met)."
            ),
            offset=off,
            size=hand_dim,
            encoding="multi-hot",
            value_range="{0, 1}",
            notes=(
                "Indexed by stable bird order from cards.bird_index(). "
                "Embedded through the shared card embedder (same as hand_multihot)."
            ),
        )
    )
    off += hand_dim

    stripes.append(
        descriptors.StripeDescriptor(
            name="hand_playable_eggs_me",
            description=(
                f"Multi-hot of my hand birds where food is affordable and a habitat "
                "slot is open, but the egg cost is not yet met."
            ),
            offset=off,
            size=hand_dim,
            encoding="multi-hot",
            value_range="{0, 1}",
            notes=(
                "Indexed by stable bird order from cards.bird_index(). "
                "Embedded through the shared card embedder (same as hand_multihot)."
            ),
        )
    )
    off += hand_dim

    # ---- decision-type one-hot (always last; setup column present iff include_setup) ----
    decision_dim = layout.decision_type_dim(spec)
    active_classes = decisions.active_decision_classes(spec.include_setup)
    decision_names = ", ".join(cls.__name__ for cls in active_classes)
    stripes.append(
        descriptors.StripeDescriptor(
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
    return descriptors.VectorLayout(total_size=total, stripes=tuple(stripes))


#### State sub-field builders ####


def _food_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """One sub-field per food type (inventory or birdfeeder face count)."""
    return tuple(
        descriptors.SubFieldDescriptor(
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


def _board_slot_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
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

    sub_fields: list[descriptors.SubFieldDescriptor] = []
    slot_number = 0
    for habitat in cards.ALL_HABITATS:
        for position in range(state.ROW_SLOTS):
            group = f"slot_{habitat.value}_{position}"
            slot_base = slot_number * layout._SLOT_MUT_DIM
            for dim_idx, (dim_name, dim_desc, dim_notes) in enumerate(slot_dim_meta):
                sub_fields.append(
                    descriptors.SubFieldDescriptor(
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


def _board_summary_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
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
    sub_fields: list[descriptors.SubFieldDescriptor] = []
    for hab_idx, habitat in enumerate(cards.ALL_HABITATS):
        base = hab_idx * len(stats)
        for stat_idx, (stat_name, stat_desc, stat_notes) in enumerate(stats):
            sub_fields.append(
                descriptors.SubFieldDescriptor(
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


def _birdfeeder_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """7 sub-fields for the birdfeeder stripe (one per food face + choice die
    + the reset-availability flag)."""
    sub_fields: list[descriptors.SubFieldDescriptor] = []
    for idx, food in enumerate(cards.ALL_FOODS):
        sub_fields.append(
            descriptors.SubFieldDescriptor(
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
        descriptors.SubFieldDescriptor(
            name="face_choice_die",
            description="Dice showing a choice-wild (invertebrate/seed) face.",
            relative_offset=len(sub_fields),
            size=1,
            encoding="scalar",
            value_range="[0, 1]",
            notes="Normalized ÷ 5.",
        )
    )
    sub_fields.append(
        descriptors.SubFieldDescriptor(
            name="reset_available",
            description=(
                "Every die shows the same face — the optional pre-gain "
                "birdfeeder reset would be offered."
            ),
            relative_offset=len(sub_fields),
            size=1,
            encoding="binary",
            value_range="{0, 1}",
            notes="Mirrors ``Birdfeeder.reset_available()`` (Rule 2).",
        )
    )
    return tuple(sub_fields)


def _turn_state_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """Sub-fields for the leading turn_state stripe (26-dim one-hot + flag)."""
    from wingspan.encode import layout

    return (
        descriptors.SubFieldDescriptor(
            name="player_turn",
            description=(
                f"Which of the player's {layout.N_PLAYER_TURNS} personal turns "
                "they are currently on (0-indexed). All-zeros during setup."
            ),
            relative_offset=0,
            size=layout.N_PLAYER_TURNS,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=(
                f"{layout.N_PLAYER_TURNS} positions. "
                "Turn index = cumulative_offset[round_idx] + "
                "(ROUND_CUBES[round_idx] - action_cubes_left). "
                "Offsets by round: 0, 8, 15, 21."
            ),
        ),
        descriptors.SubFieldDescriptor(
            name="is_first_player",
            description=(
                "1.0 when this player goes first in the current round, 0.0 when second."
            ),
            relative_offset=layout.N_PLAYER_TURNS,
            size=1,
            encoding="binary",
            value_range="{0, 1}",
            notes=(
                "Derived as: me.id == (start_player + round_idx) % 2. "
                "Flips every round."
            ),
        ),
    )


def _misc_scalars_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """4 sub-fields for the misc-scalars stripe (trailing goal/deck scalars)."""
    scalar_entries = [
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
        descriptors.SubFieldDescriptor(
            name=name,
            description=desc,
            relative_offset=idx,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes=notes,
        )
        for idx, (name, desc, notes) in enumerate(scalar_entries)
    )


def _round_goals_sub_fields() -> tuple[descriptors.SubFieldDescriptor, ...]:
    """16 logical sub-field groups for the round-goals stripe (4 rounds × 4 groups).

    Each round contributes a category one-hot block (size 20) plus three
    individual scalars for my count, opponent count, and current placement VP.
    """
    sub_fields: list[descriptors.SubFieldDescriptor] = []
    for round_idx in range(layout._NUM_ROUNDS):
        base = round_idx * layout._ROUND_GOAL_SLOT_DIM
        group = f"round_{round_idx}"
        sub_fields.append(
            descriptors.SubFieldDescriptor(
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
            descriptors.SubFieldDescriptor(
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
            descriptors.SubFieldDescriptor(
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
            descriptors.SubFieldDescriptor(
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
