# pyright: reportPrivateUsage=false
# (reads the sibling encode module's package-private layout constants —
# deliberate intra-package coupling identical to encode/stripes.py's convention)
"""Programmatic stripe registry for the setup model's input vector.

Three layouts are available:

* :func:`setup_stripe_layout` — the **raw** pre-embedding vector (``total_dim``
  elements, the bytes the encoder actually writes).  Use this when you need to
  document or inspect the encoder output itself.

* :func:`setup_state_stripe_layout` — the **post-embedding state vector** the
  setup net's STATE trunk receives: the action-independent stripes (tray as
  per-slot card-table rows, birdfeeder, round goals, and the bonus-cards-on-offer
  multi-hot in split-bonus mode), in the order ``SetupNet._embed_state``
  concatenates them.  Sums to :func:`setup_state_input_dim`.

* :func:`setup_choice_stripe_layout` — the **post-embedding choice vector** the
  setup net's CHOICE trunk receives: the action stripes (kept-card / playability
  sets, kept foods, the bonus action, kept-bonus pricing and goal affinity), in
  the order ``SetupNet._embed_choice`` concatenates them.  Sums to
  :func:`setup_choice_input_dim`.

The two post-embedding layouts are the setup analogue of
:func:`wingspan.encode.stripes.state_stripe_layout` / ``choice_stripe_layout`` and
are what the HTML model-summary report displays. The default (no args) reproduces
the all-splits-off layout.
"""

from __future__ import annotations

from wingspan import architecture, cards, encode
from wingspan.encode import stripes as encode_stripes
from wingspan.encode.stripes import embed_rules
from wingspan.setup_model import architecture as arch_module
from wingspan.setup_model import encode as setup_encode


def setup_stripe_layout(
    encoding: arch_module.SetupEncoding | None = None,
) -> encode_stripes.VectorLayout:
    """Build the stripe registry for the setup net's input vector.

    ``encoding`` selects the active layout; the default ``SetupEncoding()``
    reproduces the 308-dim pre-0.2 layout (both splits off). Two deliberate
    contrasts with the in-game encoder are called out in the notes: the
    birdfeeder block carries *raw* die-face counts (the state vector's is
    normalized ÷ 5), and each round goal is a bare category one-hot (no count /
    VP scalars — though the trailing affinity block prices the keep against each
    goal).
    """
    if encoding is None:
        encoding = arch_module.SetupEncoding()

    food_names = ", ".join(food.value for food in cards.ALL_FOODS)
    stripes: list[encode_stripes.StripeDescriptor] = []
    off = 0

    # ---- candidate blocks: the keep being scored ----

    stripes.append(
        encode_stripes.StripeDescriptor(
            name="kept_cards",
            description=(
                f"The bird cards this candidate keeps, as a multi-hot over all "
                f"{arch_module._KEPT_CARDS_DIM} core-set birds."
            ),
            offset=off,
            size=arch_module._KEPT_CARDS_DIM,
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
    off += arch_module._KEPT_CARDS_DIM

    if not encoding.split_food:
        stripes.append(
            encode_stripes.StripeDescriptor(
                name="kept_foods",
                description="The starting food tokens this candidate keeps.",
                offset=off,
                size=arch_module._KEPT_FOODS_DIM,
                encoding="multi-hot",
                value_range="{0, 1}",
                notes=f"Food types in order: {food_names}.",
                sub_fields=_kept_food_sub_fields(),
            )
        )
        off += arch_module._KEPT_FOODS_DIM

    if not encoding.split_bonus:
        stripes.append(
            encode_stripes.StripeDescriptor(
                name="kept_bonus",
                description=(
                    f"The bonus card this candidate keeps, as a one-hot over all "
                    f"{arch_module._BONUS_DIM} core-set bonus cards."
                ),
                offset=off,
                size=arch_module._BONUS_DIM,
                encoding="one-hot",
                value_range="{0, 1}",
                notes=(
                    "Indexed by stable bonus-card order from cards.bonus_index(). "
                    "All-zero when no bonus is kept."
                ),
            )
        )
        off += arch_module._BONUS_DIM
    else:
        stripes.append(
            encode_stripes.StripeDescriptor(
                name="bonus_cards",
                description=(
                    f"The bonus cards available in this deal, as a multi-hot over "
                    f"all {arch_module._BONUS_DIM} core-set bonus cards."
                ),
                offset=off,
                size=arch_module._BONUS_DIM,
                encoding="multi-hot",
                value_range="{0, 1}",
                notes=(
                    "Indexed by cards.bonus_index(). Present only when "
                    "split_setup_bonus is active — encodes which bonuses are on "
                    "offer for this deal (context), since the bonus pick is deferred "
                    "to the in-game CHOOSE_BONUS head."
                ),
            )
        )
        off += arch_module._BONUS_DIM

        stripes.append(
            encode_stripes.StripeDescriptor(
                name="bonus_card_affinity",
                description=(
                    "Min and max qualifier counts for the dealt bonus cards "
                    "against the kept cards."
                ),
                offset=off,
                size=arch_module._BONUS_AFF_DIM,
                encoding="vector",
                value_range="[0, ~1]",
                notes=(
                    "2 values: min_affinity and max_affinity — for each dealt bonus "
                    "card, count how many kept cards qualify it (same logic as "
                    "kept_bonus_value's qual_count), then take the min and max of "
                    "the two counts, normalized ÷ 5."
                ),
                sub_fields=_bonus_affinity_sub_fields(),
            )
        )
        off += arch_module._BONUS_AFF_DIM

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

    # ---- candidate pricing blocks: keep valued against bonus and round goals ----

    if not encoding.split_bonus:
        stripes.append(
            encode_stripes.StripeDescriptor(
                name="kept_bonus_value",
                description=(
                    "The kept bonus card priced against this candidate's keep: "
                    "kept-card qualifiers, the stepped / linear VP they would pay, "
                    "and tray potential."
                ),
                offset=off,
                size=arch_module._KEPT_BONUS_VALUE_DIM,
                encoding="vector",
                value_range="[0, ~1]",
                notes=(
                    f"{arch_module._KEPT_BONUS_VALUE_DIM} values: qual_count (kept "
                    "cards passing the bonus test — every kept card for the "
                    "hand-counting dynamic card, ÷5), stepped_vp / linear_vp (what "
                    "the card pays if they all reach the board, ÷7), tray_potential "
                    "(tray birds that could still qualify it, ÷5). All-zero when no "
                    "bonus is kept."
                ),
                sub_fields=_kept_bonus_value_sub_fields(),
            )
        )
        off += arch_module._KEPT_BONUS_VALUE_DIM

    stripes.append(
        encode_stripes.StripeDescriptor(
            name="goal_affinity",
            description=(
                "Per round goal, how many kept cards would advance the goal's "
                "category if played."
            ),
            offset=off,
            size=setup_encode._GOAL_AFFINITY_DIM,
            encoding="vector",
            value_range="[0, ~1]",
            notes=(
                "One scalar per round (÷5): the summed static category affinity "
                "of the kept cards (e.g. forest-capable birds toward a "
                "birds_forest goal). Egg-driven goals are rightly 0 — nothing "
                "has eggs at setup time."
            ),
            sub_fields=_goal_affinity_sub_fields(),
        )
    )
    off += setup_encode._GOAL_AFFINITY_DIM

    if encoding.include_turn1_playable:
        stripes.append(
            encode_stripes.StripeDescriptor(
                name="turn1_playable",
                description=(
                    f"Birds playable on turn 1 given the candidate's kept food and "
                    f"habitat, as a multi-hot over all {arch_module._KEPT_CARDS_DIM} "
                    "core-set birds."
                ),
                offset=off,
                size=arch_module._KEPT_CARDS_DIM,
                encoding="multi-hot",
                value_range="{0, 1}",
                notes=(
                    "Indexed by cards.bird_index(). Embedded in-net as one extra card "
                    "set through the frozen copy of the main net's multi-card set "
                    "encoder, sharing the hand_embed_width output."
                ),
            )
        )
        off += arch_module._KEPT_CARDS_DIM

    if encoding.include_playable_kept_cards:
        stripes.append(
            encode_stripes.StripeDescriptor(
                name="playable_kept_cards",
                description=(
                    f"Kept birds for which some keepable food set would allow "
                    f"turn-1 play, as a multi-hot over all "
                    f"{arch_module._KEPT_CARDS_DIM} core-set birds."
                ),
                offset=off,
                size=arch_module._KEPT_CARDS_DIM,
                encoding="multi-hot",
                value_range="{0, 1}",
                notes=(
                    "Food-agnostic: a bird is set iff some (5−bird_count)-subset "
                    "of the 5 food types pays its printed cost. Unlike turn1_playable "
                    "this does not require a concrete kept_foods tuple, so it is "
                    "non-trivial in the split_setup_food=True regime. "
                    "Indexed by cards.bird_index(). Embedded in-net as one extra "
                    "card set through the frozen copy of the main net's multi-card "
                    "set encoder."
                ),
            )
        )
        off += arch_module._KEPT_CARDS_DIM

    assert off == encoding.total_dim, (
        f"stripe offsets sum to {off} but encoding.total_dim is "
        f"{encoding.total_dim} — setup_model architecture.py and stripes.py "
        "are out of sync"
    )
    return encode_stripes.VectorLayout(
        total_size=encoding.total_dim, stripes=tuple(stripes)
    )


_DEFAULT_ARCH = architecture.ModelArchitecture()


def setup_state_stripe_layout(
    encoding: arch_module.SetupEncoding | None = None,
    main_arch: architecture.ModelArchitecture | None = None,
) -> encode_stripes.VectorLayout:
    """The post-embedding STATE vector the setup net's state trunk receives.

    Gathers the action-independent stripes — the tray (per-slot card-table rows),
    the birdfeeder, the round goals, and (in ``split_bonus`` mode) the
    bonus-cards-on-offer multi-hot — in the order ``SetupNet._embed_state``
    concatenates them, embedding the tray through the frozen card table. Sums to
    ``setup_state_input_dim`` — the state trunk's first-``Linear`` input width.

    ``main_arch`` determines the card-embedding geometry (``card_embed_dim``).
    Defaults reproduce the default :class:`~wingspan.architecture.ModelArchitecture`.
    """
    if encoding is None:
        encoding = arch_module.SetupEncoding()
    if main_arch is None:
        main_arch = _DEFAULT_ARCH
    names = ["tray", "birdfeeder", "round_goals"]
    if encoding.split_bonus:
        names.append("bonus_cards")
    return _embed_setup_sub_layout(
        encoding,
        main_arch,
        names,
        arch_module.setup_state_input_dim(encoding, main_arch),
    )


def setup_choice_stripe_layout(
    encoding: arch_module.SetupEncoding | None = None,
    main_arch: architecture.ModelArchitecture | None = None,
) -> encode_stripes.VectorLayout:
    """The post-embedding CHOICE vector the setup net's choice trunk receives.

    Gathers the action (keep-dependent) stripes — the kept-cards set, kept foods,
    the bonus action (folded ``kept_bonus`` one-hot or split ``bonus_card_affinity``),
    ``kept_bonus_value`` (folded mode only), the per-round ``goal_affinity``, and
    each appended playability set — in the order ``SetupNet._embed_choice``
    concatenates them, embedding the kept-card and playability multi-hots as set
    vectors. Sums to ``setup_choice_input_dim`` — the choice trunk's first-``Linear``
    input width.

    ``main_arch`` determines the set-embedding geometry. Defaults reproduce the
    default :class:`~wingspan.architecture.ModelArchitecture`.
    """
    if encoding is None:
        encoding = arch_module.SetupEncoding()
    if main_arch is None:
        main_arch = _DEFAULT_ARCH
    names = [
        "kept_cards",
        "kept_foods",
        "kept_bonus",
        "bonus_card_affinity",
        "kept_bonus_value",
        "goal_affinity",
        "turn1_playable",
        "playable_kept_cards",
    ]
    return _embed_setup_sub_layout(
        encoding,
        main_arch,
        names,
        arch_module.setup_choice_input_dim(encoding, main_arch),
    )


###### PRIVATE #######


def _embed_setup_sub_layout(
    encoding: arch_module.SetupEncoding,
    main_arch: architecture.ModelArchitecture,
    names: list[str],
    expected_total: int,
) -> encode_stripes.VectorLayout:
    """Select the named raw stripes (in ``names`` order, skipping absent ones),
    re-offset them contiguously, then apply the setup card-embedding rewrites so
    the result sums to ``expected_total`` — the matching trunk's first-``Linear``
    input width. ``names`` must follow the order the net concatenates the stripes
    so the displayed offsets match the real tensor positions."""
    set_width = (
        main_arch.hand_embed_width
        if main_arch.use_distinct_hand_model
        else main_arch.pooled_hand_width
    )
    by_name = {stripe.name: stripe for stripe in setup_stripe_layout(encoding).stripes}
    selected: list[encode_stripes.StripeDescriptor] = []
    off = 0
    for name in names:
        stripe = by_name.get(name)
        if stripe is None:
            continue
        selected.append(stripe.model_copy(update={"offset": off}))
        off += stripe.size
    sub = encode_stripes.VectorLayout(total_size=off, stripes=tuple(selected))
    return embed_rules.embed_layout(
        sub,
        embed_rules.setup_embed_rules(
            main_arch.card_embed_dim,
            set_width,
            use_distinct=main_arch.use_distinct_hand_model,
        ),
        expected_total,
    )


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


def _kept_bonus_value_sub_fields() -> tuple[encode_stripes.SubFieldDescriptor, ...]:
    """4 sub-fields for the kept-bonus pricing block."""
    entries = [
        (
            "qual_count",
            "Kept cards passing the kept bonus card's test.",
            "Normalized ÷ 5. Every kept card for the hand-counting dynamic card.",
        ),
        (
            "stepped_vp",
            "Stepped VP the kept bonus pays at the kept-qualifier count.",
            "Normalized ÷ 7.",
        ),
        (
            "linear_vp",
            "Piecewise-linear VP the kept bonus pays at the kept-qualifier count.",
            "Normalized ÷ 7.",
        ),
        (
            "tray_potential",
            "Tray birds that could still qualify the kept bonus.",
            "Normalized ÷ 5.",
        ),
    ]
    return tuple(
        encode_stripes.SubFieldDescriptor(
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


def _bonus_affinity_sub_fields() -> tuple[encode_stripes.SubFieldDescriptor, ...]:
    """2 sub-fields: min and max bonus-card affinity against the kept cards."""
    return (
        encode_stripes.SubFieldDescriptor(
            name="min_affinity",
            description="Qualifier count for the weaker-matching dealt bonus card.",
            relative_offset=0,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes="Normalized ÷ 5.",
        ),
        encode_stripes.SubFieldDescriptor(
            name="max_affinity",
            description="Qualifier count for the stronger-matching dealt bonus card.",
            relative_offset=1,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes="Normalized ÷ 5.",
        ),
    )


def _goal_affinity_sub_fields() -> tuple[encode_stripes.SubFieldDescriptor, ...]:
    """One kept-card affinity scalar per round goal."""
    return tuple(
        encode_stripes.SubFieldDescriptor(
            name=f"round_{round_idx}.kept_affinity",
            description=(
                f"Kept cards that would advance the round-{round_idx} goal's "
                "category if played."
            ),
            relative_offset=round_idx,
            size=1,
            encoding="scalar",
            value_range="[0, ~1]",
            notes="Normalized ÷ 5.",
            group=f"round_{round_idx}",
        )
        for round_idx in range(setup_encode._NUM_SETUP_GOALS)
    )


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
