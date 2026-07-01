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
    use_board_attention: bool = False,
    hand_embed_dim: int | None = None,
    pooled_hand_width: int | None = None,
    tray_set_embedding: bool = False,
    n_playable_multihots: int = 0,
) -> descriptors.VectorLayout:
    """Build the stripe registry for the state trunk's input vector.

    Lists every stripe in offset order with sizes from the ``layout`` constants.
    The card-index block and hand multi-hot are shown at their *post-embedding*
    width — each board / tray slot index as one ``card_embed_dim`` vector, the
    hand as one embedding (pooled at ``pooled_hand_width`` when
    ``use_distinct_hand_model`` is False, or the dedicated hand encoder's resolved
    ``hand_embed_dim`` otherwise) — so the breakdown sums to the trunk's
    first-``Linear`` input (``layout.trunk_input_dim``): what the network
    actually sees, not the raw encoder output. ``tray_set_embedding`` widens the
    tray stripe by one derived set embedding (3·M + N). When ``use_board_attention``
    is True, ``board_me`` / ``board_opp`` each show as their attention-output width
    and ``card_idx_board`` is folded into them (see :func:`~embed_rules.state_embed_rules`).
    ``n_playable_multihots`` is the number of extra playability multi-hot stripes
    (``hand_playable_me``, ``hand_playable_eggs_me``) that follow ``hand_multihot``
    in the v0.6+ state vector; each is embedded at the same width as the hand
    embedding. (The model concatenates the embeddings after the continuous features;
    here they keep their encoding-order position.) Only the trailing decision-type one-hot's
    width depends on ``spec``.
    """
    raw = _build_raw_state_stripes(spec)
    return embed_rules.embed_layout(
        raw,
        embed_rules.state_embed_rules(
            card_embed_dim,
            use_distinct_hand_model=use_distinct_hand_model,
            use_board_attention=use_board_attention,
            hand_embed_dim=hand_embed_dim,
            pooled_hand_width=pooled_hand_width,
            tray_set_embedding=tray_set_embedding,
            n_playable_multihots=n_playable_multihots,
        ),
        layout.trunk_input_dim(
            raw.total_size,
            card_embed_dim,
            use_distinct_hand_model=use_distinct_hand_model,
            hand_embed_dim=hand_embed_dim,
            pooled_hand_width=pooled_hand_width,
            tray_set_embedding=tray_set_embedding,
            n_playable_multihots=n_playable_multihots,
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
    vector that ``encode_state`` produces.

    Offsets and sizes for all continuous stripes are derived from
    :data:`~wingspan.encode.layout.STATE_CONT_LAYOUT` — the single authoritative
    source.  Only the spec-dependent ``decision_type`` stripe at the end requires
    its own size computation."""
    from wingspan.encode import state_encode

    total = state_encode.state_size(spec)
    food_names = ", ".join(f.value for f in cards.ALL_FOODS)
    habitat_names = ", ".join(h.value for h in cards.ALL_HABITATS)

    # Derive offset and size from the authoritative continuous layout by stripe name.
    def _at(name: str) -> tuple[int, int]:
        return (
            layout.STATE_CONT_LAYOUT.offset_of(name),
            layout.STATE_CONT_LAYOUT.size_of(name),
        )

    stripes: list[descriptors.StripeDescriptor] = []

    # ---- turn state (first stripe) ----
    turn_off, turn_size = _at("turn_state")
    stripes.append(
        descriptors.StripeDescriptor(
            name="turn_state",
            description=(
                "Which of the player's 26 personal turns they are on, plus "
                "whether they go first in the current round."
            ),
            offset=turn_off,
            size=turn_size,
            encoding="vector",
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

    # ---- food inventory ----
    food_off_me, food_size = _at("food_me")
    stripes.append(
        descriptors.StripeDescriptor(
            name="food_me",
            description="My food inventory, one element per food type.",
            offset=food_off_me,
            size=food_size,
            encoding="vector",
            value_range="[0, ~1.7]",
            notes=f"Food types in order: {food_names}. Normalized ÷ 6.",
            sub_fields=_food_sub_fields(),
        )
    )

    food_off_opp, _ = _at("food_opp")
    stripes.append(
        descriptors.StripeDescriptor(
            name="food_opp",
            description="Opponent food inventory, one element per food type.",
            offset=food_off_opp,
            size=food_size,
            encoding="vector",
            value_range="[0, ~1.7]",
            notes=f"Food types in order: {food_names}. Normalized ÷ 6.",
            sub_fields=_food_sub_fields(),
        )
    )

    # ---- food distance to playable (hand / tray) ----
    hand_unlock_off, unlock_size = _at("hand_food_unlock_me")
    stripes.append(
        descriptors.StripeDescriptor(
            name="hand_food_unlock_me",
            description=(
                "Per food, the smallest count that would newly unlock one of my "
                "hand birds (0 when none is unlockable by that food)."
            ),
            offset=hand_unlock_off,
            size=unlock_size,
            encoding="vector",
            value_range="[0, ~1.3]",
            notes=(
                f"Food types in order: {food_names}. Open matching slot required, "
                "egg cost ignored; full 2-for-1 affordability. Normalized ÷ 6."
            ),
            sub_fields=_food_sub_fields(),
        )
    )

    tray_unlock_off, _ = _at("tray_food_unlock_me")
    stripes.append(
        descriptors.StripeDescriptor(
            name="tray_food_unlock_me",
            description=(
                "Per food, the smallest count that would unlock a face-up tray "
                "bird as if it were in my hand (0 when none is unlockable)."
            ),
            offset=tray_unlock_off,
            size=unlock_size,
            encoding="vector",
            value_range="[0, ~1.3]",
            notes=(
                f"Food types in order: {food_names}. Same rule as "
                "hand_food_unlock_me, scored against my own food + board. "
                "Normalized ÷ 6."
            ),
            sub_fields=_food_sub_fields(),
        )
    )

    # ---- board continuous (mutable per-slot state) ----
    n_slots = state.N_HABITATS * state.ROW_SLOTS
    slot_dim = layout._SLOT_MUT_DIM  # 9: eggs, egg_cap, cached×5, tucked, activations

    _board_notes = (
        f"{n_slots} slots ({state.N_HABITATS} habitats × {state.ROW_SLOTS} positions). "
        f"Per slot ({slot_dim} values): eggs[0], egg_cap[1], "
        f"cached_food_by_type[2:{2 + cards.N_FOODS}] ({food_names}), "
        f"tucked[{layout._SLOT_MUT_TUCKED}], activations[{layout._SLOT_MUT_ACTIVATIONS}]. "
        "Eggs and cached food normalized ÷ 6; activations ÷ 4."
    )
    board_off_me, board_size = _at("board_me")
    stripes.append(
        descriptors.StripeDescriptor(
            name="board_me",
            description="Mutable per-slot board state for my board.",
            offset=board_off_me,
            size=board_size,
            encoding="complex",
            value_range="[0, ~1]",
            notes=_board_notes,
            sub_fields=_board_slot_sub_fields(),
        )
    )

    board_off_opp, _ = _at("board_opp")
    stripes.append(
        descriptors.StripeDescriptor(
            name="board_opp",
            description="Mutable per-slot board state for the opponent's board.",
            offset=board_off_opp,
            size=board_size,
            encoding="complex",
            value_range="[0, ~1]",
            notes=_board_notes,
            sub_fields=_board_slot_sub_fields(),
        )
    )

    # ---- board summary (aggregate per-habitat stats, compacted in v0.9) ----
    _board_summary_notes = (
        f"3 habitats ({habitat_names}) × 2 stats: "
        "row_length (filled slots ÷ 5), total_eggs (÷ 6). "
        "Compacted in v0.9 from 6→2 stats per habitat (points/tucked/cached/brown "
        "dropped — redundant with the per-slot continuous stripe and card table)."
    )

    bs_off_me, bs_size = _at("board_summary_me")
    stripes.append(
        descriptors.StripeDescriptor(
            name="board_summary_me",
            description="Aggregate per-habitat row statistics for my board (row_length, total_eggs).",
            offset=bs_off_me,
            size=bs_size,
            encoding="vector",
            value_range="[0, ~1]",
            notes=_board_summary_notes,
            sub_fields=_board_summary_sub_fields(),
        )
    )

    bs_off_opp, _ = _at("board_summary_opp")
    stripes.append(
        descriptors.StripeDescriptor(
            name="board_summary_opp",
            description="Aggregate per-habitat row statistics for the opponent's board (row_length, total_eggs).",
            offset=bs_off_opp,
            size=bs_size,
            encoding="vector",
            value_range="[0, ~1]",
            notes=_board_summary_notes,
            sub_fields=_board_summary_sub_fields(),
        )
    )

    # hand_summary_me removed at the 0.9 compaction (the 1.0 baseline): the distinct
    # hand encoder derives the 10-dim summary in-model from the hand multi-hot via
    # set_summary_from_multihot. No pre-1.0 shim reinstates the inline stripe.

    # ---- bonus progress (POV player only; opponent identity hidden) ----
    # "bonus_progress" in layout is one 4×bonus_dim block; split here for viewer clarity.
    bonus_dim = layout._BONUS_ID_DIM  # 26 bonus cards
    bonus_base, _ = _at("bonus_progress")
    stripes.append(
        descriptors.StripeDescriptor(
            name="bonus_progress_held",
            description=(
                f"Which of the {bonus_dim} bonus cards I am holding (multi-hot)."
            ),
            offset=bonus_base,
            size=bonus_dim,
            encoding="multi-hot",
            value_range="{0, 1}",
            notes="Indexed by stable bonus-card order from cards.bonus_index().",
        )
    )

    stripes.append(
        descriptors.StripeDescriptor(
            name="bonus_progress_count",
            description="Number of my birds that qualify for each bonus card.",
            offset=bonus_base + bonus_dim,
            size=bonus_dim,
            encoding="vector",
            value_range="[0, ~1]",
            notes=f"One value per bonus card ({bonus_dim} total). Normalized ÷ 5.",
        )
    )

    stripes.append(
        descriptors.StripeDescriptor(
            name="bonus_progress_stepped",
            description="Current stepped VP for each bonus card I hold.",
            offset=bonus_base + 2 * bonus_dim,
            size=bonus_dim,
            encoding="vector",
            value_range="[0, ~1]",
            notes=f"One value per bonus card ({bonus_dim} total). Normalized ÷ 7 (max single-card VP).",
        )
    )

    stripes.append(
        descriptors.StripeDescriptor(
            name="bonus_progress_linear",
            description="Linear (fractional) VP for each bonus card I hold.",
            offset=bonus_base + 3 * bonus_dim,
            size=bonus_dim,
            encoding="vector",
            value_range="[0, ~1]",
            notes=f"One value per bonus card ({bonus_dim} total). Normalized ÷ 7.",
        )
    )

    # ---- opponent aggregate counts ----
    # "opp_bonus_count" / "opp_hand_size" in layout; renamed here for display consistency.
    opp_bonus_off, opp_bonus_size = _at("opp_bonus_count")
    stripes.append(
        descriptors.StripeDescriptor(
            name="bonus_count_opp",
            description="Number of bonus cards the opponent holds (identity hidden).",
            offset=opp_bonus_off,
            size=opp_bonus_size,
            encoding="scalar",
            value_range="[0, ~1]",
            notes="Normalized ÷ 5.",
        )
    )

    opp_hand_off, opp_hand_size = _at("opp_hand_size")
    stripes.append(
        descriptors.StripeDescriptor(
            name="hand_size_opp",
            description="Number of bird cards in the opponent's hand (contents hidden).",
            offset=opp_hand_off,
            size=opp_hand_size,
            encoding="scalar",
            value_range="[0, ~1]",
            notes="Normalized ÷ 10.",
        )
    )

    # ---- birdfeeder ----
    bf_off, bf_size = _at("birdfeeder")
    stripes.append(
        descriptors.StripeDescriptor(
            name="birdfeeder",
            description=(
                "Birdfeeder die face counts (single-food faces and choice-wild "
                "dice) plus the reset-availability flag."
            ),
            offset=bf_off,
            size=bf_size,
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

    # ---- miscellaneous scalars (compacted in v0.9) ----
    misc_off, misc_size = _at("misc_scalars")
    stripes.append(
        descriptors.StripeDescriptor(
            name="misc_scalars",
            description="Miscellaneous game state scalars (tray size, deck size).",
            offset=misc_off,
            size=misc_size,
            encoding="vector",
            value_range="[0, ~1]",
            notes=(
                "2 scalars (v0.9+): tray_size (÷3), deck_size (÷100). "
                "my_round_goal_pts and opp_round_goal_pts removed in v0.9 — "
                "goal standings are fully captured by the round_goals stripe."
            ),
            sub_fields=_misc_scalars_sub_fields(),
        )
    )

    # ---- round-goal state (all four rounds) ----
    goal_slot = layout._ROUND_GOAL_SLOT_DIM  # MAX_GOAL_CATEGORIES + 3 = 23
    rg_off, rg_size = _at("round_goals")
    stripes.append(
        descriptors.StripeDescriptor(
            name="round_goals",
            description="State of all four round goals (category, counts, VP placement).",
            offset=rg_off,
            size=rg_size,
            encoding="vector",
            value_range="varies",
            notes=(
                f"4 rounds × {goal_slot} values. Per round: "
                f"category_one_hot[0:{layout.MAX_GOAL_CATEGORIES}] ({layout.MAX_GOAL_CATEGORIES} dims), "
                "my_count (normalized ÷ 5), opp_count (normalized ÷ 5), "
                "placement_vp (normalized ÷ 10). "
                "Scored (passed) rounds are zeroed in v0.9+ — their counts/VP are "
                "frozen history no longer relevant to future decisions."
            ),
            sub_fields=_round_goals_sub_fields(),
        )
    )

    # ---- card-identity index block ----
    # "card_idx_block" in layout spans board then tray slots; split here for viewer clarity.
    card_base, _ = _at("card_idx_block")
    n_board_idx = layout.N_BOARD_INDEX_SLOTS  # 2 * 15 = 30
    stripes.append(
        descriptors.StripeDescriptor(
            name="card_idx_board",
            description=(
                "Bird indices for all board slots (my board then opponent's), "
                "looked up in the shared card embedding table."
            ),
            offset=card_base,
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

    stripes.append(
        descriptors.StripeDescriptor(
            name="card_idx_tray",
            description=(
                "Bird indices for the three face-up tray slots, "
                "looked up in the shared card embedding table."
            ),
            offset=card_base + n_board_idx,
            size=state.TRAY_SIZE,
            encoding="integer-index",
            value_range=f"int 0–{cards.n_birds()}",
            notes=f"{state.TRAY_SIZE} integer indices. bird_index + 1; 0 = empty slot.",
        )
    )

    # ---- hand identity (multi-hot) ----
    hand_dim = layout.HAND_MULTIHOT_DIM  # n_birds = 180
    hand_off, hand_size = _at("hand_multihot")
    stripes.append(
        descriptors.StripeDescriptor(
            name="hand_multihot",
            description=(
                f"My hand encoded as a multi-hot over all {hand_dim} core birds. "
                "Opponent hand is hidden."
            ),
            offset=hand_off,
            size=hand_size,
            encoding="multi-hot",
            value_range="{0, 1}",
            notes=(
                "Indexed by stable bird order from cards.bird_index(). "
                "Mean-pooled through the shared card embedding inside the model."
            ),
        )
    )

    # ---- hand-playability multi-hots (v0.6+: two 180-dim stripes) ----
    hp_me_off, hp_size = _at("hand_playable_me")
    stripes.append(
        descriptors.StripeDescriptor(
            name="hand_playable_me",
            description=(
                f"Multi-hot of my hand birds that are playable right now "
                "(food affordable, at least one open habitat slot, egg cost met)."
            ),
            offset=hp_me_off,
            size=hp_size,
            encoding="multi-hot",
            value_range="{0, 1}",
            notes=(
                "Indexed by stable bird order from cards.bird_index(). "
                "Embedded through the shared card embedder (same as hand_multihot)."
            ),
        )
    )

    hp_eggs_off, _ = _at("hand_playable_eggs_me")
    stripes.append(
        descriptors.StripeDescriptor(
            name="hand_playable_eggs_me",
            description=(
                f"Multi-hot of my hand birds where food is affordable and a habitat "
                "slot is open, but the egg cost is not yet met."
            ),
            offset=hp_eggs_off,
            size=hp_size,
            encoding="multi-hot",
            value_range="{0, 1}",
            notes=(
                "Indexed by stable bird order from cards.bird_index(). "
                "Embedded through the shared card embedder (same as hand_multihot)."
            ),
        )
    )

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
            offset=layout.STATE_CONT_LAYOUT.total_size,
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

    assert layout.STATE_CONT_LAYOUT.total_size + decision_dim == total, (
        f"STATE_CONT_LAYOUT.total_size ({layout.STATE_CONT_LAYOUT.total_size}) + "
        f"decision_dim ({decision_dim}) = "
        f"{layout.STATE_CONT_LAYOUT.total_size + decision_dim} "
        f"but state_size(spec) returns {total} — layout.py is out of sync"
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
    """6 sub-fields for a board-summary stripe (3 habitats × 2 named stats, v0.9+).

    Compacted from 6 stats per habitat in v0.8 to 2 in v0.9 (total_points,
    total_tucked, total_cached_food, and brown_bird_count dropped as redundant
    with the per-slot continuous stripe and card table)."""
    stats = [
        (
            "row_length",
            "Number of occupied slots in this habitat row.",
            "Normalized ÷ 5 (max slots).",
        ),
        ("total_eggs", "Total eggs on all birds in this habitat.", "Normalized ÷ 6."),
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
    """2 sub-fields for the misc-scalars stripe (tray size and deck size, v0.9+)."""
    scalar_entries = [
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
