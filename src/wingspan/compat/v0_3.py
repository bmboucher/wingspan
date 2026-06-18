# pyright: reportPrivateUsage=false
# (the v0.3 -> v0.4 shim reads the live layout's package-private stripe offsets
# and calls the frozen v0.3 misc-scalar encoder -- a deliberate compat coupling,
# pinned by the import-time layout-contract assertions below)
"""Frozen v0.3 state encoding: the shim that keeps pre-0.4 artifacts playable.

Artifact version 0.4 added a new leading ``turn_state`` stripe (27 dims) and
shrank ``misc_scalars`` from 26 dims to 4, changing the state vector by +5 dims
(790 → 795):

* Added: 26-dim player-turn one-hot + 1-bit is_first_player flag as the first
  stripe (``turn_state``).
* Dropped: 4-dim round one-hot, both 9-dim cube one-hots from ``misc_scalars``.
* Retained: the 4 trailing scalars (goal pts ×2, tray size, deck size) remain
  in a now-4-dim ``misc_scalars``.

Nets trained before 0.4 have a 790-dim state trunk input; this module keeps
them loadable:

* :func:`encode_misc_scalars_v03` rebuilds the 26-element one-hot misc stripe
  using the frozen v0.3 geometry (round one-hot, my cubes one-hot, opp cubes
  one-hot, 4 scalars) and can be spliced in via :func:`encode_state_v03`.
* :func:`encode_state_v03` produces the complete 790-dim state vector (no
  turn_state stripe, frozen 26-dim misc stripe, everything else live) for a
  v0.3 checkpoint.
* :class:`PolicyValueNetV03` overrides :meth:`encode_state` so inference call
  sites drive the net with the 790-dim input its weights expect, and
  :meth:`_state_embed_offsets` so the trunk *slices* that input at the v0.3
  offsets it was written with (the live 795-dim offsets sit 5 columns too far
  right; the widths coincide, so a live slice corrupts silently rather than
  crashing).
* :func:`state_embed_offsets_v03` is the frozen-geometry slice offsets the pre-0.4
  nets feed to ``_embed_state``.
* :func:`state_feature_dim_v03` is the frozen 790-dim state width itself — what
  the era-dims router (``compat.encoding_dims_for_era``) hands an era-pinned
  ``TrainConfig`` so a resumed pre-0.4 run keeps training at its own geometry.
* :func:`state_stripe_layout_v03` produces the frozen stripe registry matching
  the 790-dim v0.3 state vector, for consistent reporting of old checkpoints.

Per the compatibility policy (``CLAUDE.md``), this shim lives until a MAJOR
``MODEL_VERSION`` bump deletes it together with the v0.3 fixture set.
"""

from __future__ import annotations

import numpy as np

from wingspan import decisions, encode, state, version
from wingspan.encode import layout, state_encode
from wingspan.encode.stripes import descriptors as stripe_descriptors
from wingspan.model import core

MISC_SCALARS_CHANGED_IN = "0.4"
"""The artifact version whose misc-scalar / turn-state reshape this module undoes."""

# ---------------------------------------------------------------------------
# Frozen v0.3 misc-scalar geometry.  These mirror the exact arithmetic that
# _summary_misc_scalars applied in v0.3: a 4-dim round one-hot, two 9-dim cube
# one-hots, and 4 scalars — captured as literals so the shim does not depend on
# any future refactor of those values.

_V03_MISC_DIM = 26  # 4-dim round + 9-dim cubes×2 + 4 scalars
_V03_ROUND_DIM = 4  # rounds 0–3
_V03_CUBES_DIM = 9  # cubes 0–8 per player
_V03_GOAL_PTS_SCALE = 10.0
_V03_TRAY_SCALE = 3.0
_V03_DECK_SCALE = 100.0

# Width deltas for computing v0.3 offsets relative to the live (v0.6) layout.
#
# The v0.3 state vector differs from live v0.6 in two ways:
# 1. v0.3 → v0.4 (+5 dims): v0.4 added 27-dim turn_state and shrank misc 26→4.
#    This shift affects card_index, hand_multihot, AND decision_type (all sit
#    after the affected turn_state / misc stripes).
#    Formula: _V03_MISC_DIM - (N_PLAYER_TURNS+1 + 4) = 26 - 31 = -5.
# 2. v0.4 → v0.6 (+360 dims): v0.6 added N_HAND_PLAYABLE_MULTIHOTS playability
#    stripes BETWEEN hand_multihot and decision_type.
#    This additional shift affects ONLY decision_type (it sits after the stripes).
#
# Per-offset deltas:
# - card_index, hand_multihot, hand_summary: only the v0.3→v0.4 delta applies.
# - decision_type: both deltas apply.
_V04_ENCODING_DELTA = _V03_MISC_DIM - (layout.N_PLAYER_TURNS + 1 + 4)
"""The v0.3 → v0.4 turn_state/misc width delta (−5). Applied to card_index and
hand_multihot (which sit between the turn_state/misc block and the v0.6
playability stripes). decision_type uses the full :data:`_TOTAL_DIM_DELTA`."""

_PLAYABILITY_DELTA = -(layout.N_HAND_PLAYABLE_MULTIHOTS * layout.HAND_MULTIHOT_DIM)
"""The v0.4 → v0.6 playability-stripe width delta (−360). Applies only to
decision_type, which sits after the playability stripes in the live layout."""

# _TOTAL_DIM_DELTA is kept for the import-time assertion and for
# state_feature_dim_v03 / state_embed_offsets_v03 (where it measures the full
# delta from live v0.6 to v0.3 for the state vector total width, applying to
# decision_type which is the last stripe before the decision-type one-hot).
_TOTAL_DIM_DELTA = _V04_ENCODING_DELTA + _PLAYABILITY_DELTA

# The hand-summary stripe (10 dims) precedes misc_scalars but follows the leading
# turn_state stripe, so a v0.3 vector places it back by exactly the turn_state
# width — not the misc-inclusive _V04_ENCODING_DELTA. Pinned by
# _assert_live_layout_contract.
_HAND_SUMMARY_DIM_DELTA = -(layout.N_PLAYER_TURNS + 1)


def uses_v0_3_state_encoding(artifact_version: str) -> bool:
    """Whether ``artifact_version`` uses the v0.3 (790-dim) state encoding and
    therefore needs this module's frozen geometry to load and play."""
    parsed = version.parse_version(artifact_version)
    changed = version.parse_version(MISC_SCALARS_CHANGED_IN)
    return (parsed.major, parsed.minor) >= (0, 3) and (
        parsed.major,
        parsed.minor,
    ) < (changed.major, changed.minor)


def encode_misc_scalars_v03(
    game_state: state.GameState, me: state.Player, opp: state.Player
) -> np.ndarray:
    """The frozen v0.3 26-element misc-scalar stripe.

    Reproduces the exact pre-0.4 ``_summary_misc_scalars`` output: a 4-dim
    round one-hot, a 9-dim cube one-hot for me, a 9-dim cube one-hot for opp,
    then 4 scalars (goal pts ×2, tray size, deck size)."""
    out = np.zeros(_V03_MISC_DIM, dtype=np.float32)
    offset = 0

    # 4-dim one-hot for current round (0..3)
    out[offset + game_state.round_idx] = 1.0
    offset += _V03_ROUND_DIM

    # 9-dim one-hot for my remaining action cubes (0..8)
    out[offset + me.action_cubes_left] = 1.0
    offset += _V03_CUBES_DIM

    # 9-dim one-hot for opponent remaining action cubes (0..8)
    out[offset + opp.action_cubes_left] = 1.0
    offset += _V03_CUBES_DIM

    # 4 trailing scalars (unchanged from v0.2 → v0.3 → frozen here)
    out[offset] = me.round_goal_points / _V03_GOAL_PTS_SCALE
    out[offset + 1] = opp.round_goal_points / _V03_GOAL_PTS_SCALE
    out[offset + 2] = (
        sum(1 for bird in game_state.tray if bird is not None) / _V03_TRAY_SCALE
    )
    out[offset + 3] = len(game_state.bird_deck) / _V03_DECK_SCALE
    return out


def encode_state_v03(
    game_state: state.GameState,
    decision: decisions.Decision[decisions.Choice] | None = None,
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
) -> np.ndarray:
    """Produce the complete 790-dim v0.3 state vector.

    Identical to the live ``encode_state`` except that the leading
    ``_summary_turn_state`` stripe is omitted and ``_summary_misc_scalars`` is
    replaced with the frozen 26-dim one-hot stripe, keeping ``state_dim`` at
    790.  Everything else (board, tray, hand, goals, decision-type) uses the
    live encoders unchanged."""
    pov = decision.player_id if decision is not None else game_state.current_player
    me = game_state.players[pov]
    opp = game_state.players[1 - pov] if len(game_state.players) > 1 else me

    # Collect stripes in the same order as encode_state, omitting turn_state
    # and swapping only misc_scalars.
    parts: list[np.ndarray] = [
        state_encode._summary_food(me),
        state_encode._summary_food(opp),
        state_encode._board_slots_continuous(me),
        state_encode._board_slots_continuous(opp),
        state_encode._summary_board(me),
        state_encode._summary_board(opp),
        state_encode._summary_hand(me),
        state_encode._bonus_progress(me),
        state_encode._opp_bonus_count(opp),
        np.array([len(opp.hand) / layout._HAND_SIZE_SCALE], dtype=np.float32),
        state_encode._summary_birdfeeder(game_state),
        encode_misc_scalars_v03(game_state, me, opp),  # frozen 26-dim v0.3 stripe
        state_encode._round_goals_all_rounds(game_state, me),
        state_encode._card_index_block(me, opp, game_state),
        state_encode._hand_identity(me),
        state_encode._encode_decision_type(decision, spec),
    ]
    return np.concatenate(parts).astype(np.float32)


def state_embed_offsets_v03() -> core.StateEmbedOffsets:
    """The frozen slice offsets ``_embed_state`` uses for the 790-dim v0.3 state
    vector.

    Three deltas are applied selectively by offset position:

    * ``card_index``, ``hand_multihot``: shifted by :data:`_V04_ENCODING_DELTA`
      (-5) — these sit after the v0.4 turn_state/misc change but BEFORE the v0.6
      playability stripes, so only the v0.3→v0.4 delta moves them.
    * ``decision_type``: shifted by :data:`_TOTAL_DIM_DELTA` (-365) — this sits
      after BOTH the turn_state/misc change AND the v0.6 playability stripes, so
      the combined delta moves it.
    * ``hand_summary``: shifted by :data:`_HAND_SUMMARY_DIM_DELTA` (-27, the
      absent turn_state stripe) — it sits before misc_scalars and thus before both
      the turn_state stripe and the playability stripes.

    ``PolicyValueNetV03`` overrides ``_embed_state``'s offsets with this so an old
    checkpoint's state vector is sliced at the columns it was written with rather
    than the live ones (the widths coincide, so a live slice would corrupt silently
    rather than crashing — see ``compat/INDEX.md``)."""
    return core.StateEmbedOffsets(
        card_index=layout.OFF_CARD_INDEX + _V04_ENCODING_DELTA,
        hand_multihot=layout.OFF_HAND_MULTIHOT + _V04_ENCODING_DELTA,
        decision_type=layout.OFF_DECISION_TYPE + _TOTAL_DIM_DELTA,
        hand_summary=layout.HAND_SUMMARY_OFFSET + _HAND_SUMMARY_DIM_DELTA,
    )


def state_feature_dim_v03(spec: layout.EncodingSpec = layout.DEFAULT_SPEC) -> int:
    """The frozen v0.3 state-vector width (790 under the default spec).

    The live width shifted by :data:`_TOTAL_DIM_DELTA` (-365) — the size of
    :func:`encode_state_v03`'s output and the ``state_dim`` every pre-0.4 net
    was built with. The era-dims router (``compat.encoding_dims_for_era``) uses
    this so an era-pinned ``TrainConfig`` derives the dims its checkpoints
    actually carry."""
    return layout.state_feature_dim(spec) + _TOTAL_DIM_DELTA


class PolicyValueNetV03(core.PolicyValueNet):
    """A :class:`~wingspan.model.core.PolicyValueNet` frozen to the v0.3 state
    geometry, for checkpoints written before artifact version 0.4.

    The state trunk was trained against 790-dim inputs (one-hot round + cubes
    in misc_scalars, no turn_state stripe); this subclass overrides
    :meth:`encode_state` to keep that width and feed the frozen v0.3 state
    vector. The choice encoder also uses pre-0.6 geometry:
    :meth:`_choice_embed_offsets` returns ``becomes_playable=None`` so the
    encoder width matches the checkpoint (the ``becomes_playable`` stripe was
    added in v0.6, after these checkpoints). The card encoder is identical to
    the live era. Constructed by the version-routing loaders
    (``PolicyValueNet.from_model_config``, ``players.loaders.load_policy_net``)
    — never by the training pipeline.
    """

    def encode_state(
        self,
        game_state: state.GameState,
        decision: decisions.Decision[decisions.Choice],
    ) -> np.ndarray:
        """Featurize ``game_state`` using the frozen v0.3 26-dim one-hot misc
        stripe (no turn_state), yielding a 790-dim vector this checkpoint's
        trunk expects."""
        return encode_state_v03(game_state, decision, self.spec)

    def encode_choices(
        self,
        decision: decisions.Decision[decisions.Choice],
        game_state: state.GameState,
    ) -> np.ndarray:
        """Produce the pre-0.6 choice rows for ``decision`` — the live encoding
        without the ``becomes_playable`` stripe (added in v0.6, absent from v0.3
        checkpoints). The 215-dim format this net was trained on."""
        return encode.encode_choices(
            decision, game_state, self.spec, has_becomes_playable=False
        )

    def _state_embed_offsets(self) -> core.StateEmbedOffsets:
        """Slice the 790-dim v0.3 state vector at its own frozen offsets rather
        than the live 1155-dim ones — without this the trunk reads the
        card-index / hand stripes 5 columns too far right and decision_type 365
        columns too far right (the widths coincide, so it would corrupt silently,
        not crash)."""
        return state_embed_offsets_v03()

    def _choice_embed_offsets(self) -> core.ChoiceEmbedOffsets:
        """The frozen pre-0.6 choice embed offsets — no ``becomes_playable`` column.

        The v0.3 choice row predates the ``becomes_playable`` stripe added in v0.6;
        returning ``becomes_playable=None`` here keeps ``_build_choice_encoder``
        from adding that embedding to the input width, matching the checkpoint's
        actual first-linear shape."""
        return core.ChoiceEmbedOffsets(
            board_idx=layout.CHOICE_BOARD_IDX_OFFSET,
            bird_id=layout.CHOICE_BIRD_ID_OFFSET,
            becomes_playable=None,
            kept_multihot=(
                layout.CHOICE_KEPT_MULTIHOT_OFFSET if self.include_setup else None
            ),
        )


def state_stripe_layout_v03(
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
    card_embed_dim: int = 64,
    *,
    use_distinct_hand_model: bool = False,
    hand_embed_dim: int | None = None,
    tray_set_embedding: bool = False,
) -> stripe_descriptors.VectorLayout:
    """The frozen v0.3 state stripe registry — the live v0.4 registry with the
    turn_state stripe omitted and misc_scalars replaced by the frozen 26-dim
    one-hot version.

    Used by the reporting seam in ``runmeta.state_layout_for`` when the
    descriptor's version is 0.3, so the report matches the checkpoint's actual
    790-dim state vector."""
    from wingspan.encode.stripes import state as live_state_stripes

    # Obtain the live layout (which has the 27-dim turn_state and 4-dim misc)
    # and rebuild it for the frozen 790-dim v0.3 geometry.
    live_layout = live_state_stripes.state_stripe_layout(
        spec,
        card_embed_dim,
        use_distinct_hand_model=use_distinct_hand_model,
        hand_embed_dim=hand_embed_dim,
        tray_set_embedding=tray_set_embedding,
    )

    # Stripes present in the live v0.6 layout that did not exist in v0.3:
    # - "turn_state": added in v0.4 (27 dims).
    # - "hand_playable_me", "hand_playable_eggs_me": added in v0.6 (180 dims each).
    _V03_OMITTED_STRIPES = frozenset(
        {"turn_state", "hand_playable_me", "hand_playable_eggs_me"}
    )

    # Rebuild the stripe list for the 790-dim v0.3 vector:
    # - Skip the leading turn_state stripe (added in v0.4).
    # - Skip the two hand-playability stripes (added in v0.6).
    # - Replace misc_scalars with the frozen 26-dim one-hot version.
    # - Recompute offsets from scratch so they reflect the patched order.
    patched: list[stripe_descriptors.StripeDescriptor] = []
    running_offset = 0
    for stripe in live_layout.stripes:
        if stripe.name in _V03_OMITTED_STRIPES:
            # Omit entirely — not present in v0.3.
            continue
        elif stripe.name == "misc_scalars":
            patched.append(
                stripe_descriptors.StripeDescriptor(
                    name="misc_scalars",
                    description=(
                        "Miscellaneous game state (round, cubes, scores, deck) "
                        "— v0.3 one-hot encoding."
                    ),
                    offset=running_offset,
                    size=_V03_MISC_DIM,
                    encoding="complex",
                    value_range="varies",
                    notes=(
                        f"26 values: round_index one-hot[0:{_V03_ROUND_DIM}] ({_V03_ROUND_DIM} dims, "
                        f"rounds 0–3), my_action_cubes one-hot[{_V03_ROUND_DIM}:"
                        f"{_V03_ROUND_DIM + _V03_CUBES_DIM}] ({_V03_CUBES_DIM} dims, 0–8 cubes), "
                        f"opp_action_cubes one-hot[{_V03_ROUND_DIM + _V03_CUBES_DIM}:"
                        f"{_V03_ROUND_DIM + _V03_CUBES_DIM * 2}] ({_V03_CUBES_DIM} dims), "
                        "then 4 scalars: my_round_goal_pts (÷10), opp_round_goal_pts (÷10), "
                        "tray_size (÷3), deck_size (÷100). "
                        "(v0.3 one-hot encoding — replaced by turn_state stripe in v0.4)"
                    ),
                    sub_fields=_misc_scalars_v03_sub_fields(running_offset),
                )
            )
            running_offset += _V03_MISC_DIM
        else:
            patched.append(
                stripe_descriptors.StripeDescriptor(
                    name=stripe.name,
                    description=stripe.description,
                    offset=running_offset,
                    size=stripe.size,
                    encoding=stripe.encoding,
                    value_range=stripe.value_range,
                    notes=stripe.notes,
                    sub_fields=stripe.sub_fields,
                )
            )
            running_offset += stripe.size

    total = sum(stripe.size for stripe in patched)
    return stripe_descriptors.VectorLayout(total_size=total, stripes=tuple(patched))


###### PRIVATE #######


def _misc_scalars_v03_sub_fields(
    stripe_offset: int,
) -> tuple[stripe_descriptors.SubFieldDescriptor, ...]:
    """26 frozen sub-fields for the v0.3 misc-scalars stripe (one-hots + scalars)."""
    sub_fields: list[stripe_descriptors.SubFieldDescriptor] = []

    sub_fields.append(
        stripe_descriptors.SubFieldDescriptor(
            name="round_index",
            description="Current round number (0–3), encoded as a 4-dim one-hot.",
            relative_offset=0,
            size=_V03_ROUND_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=f"{_V03_ROUND_DIM} positions for rounds 0–3.",
        )
    )
    sub_fields.append(
        stripe_descriptors.SubFieldDescriptor(
            name="my_action_cubes",
            description="My remaining action cubes this round (0–8), encoded as a 9-dim one-hot.",
            relative_offset=_V03_ROUND_DIM,
            size=_V03_CUBES_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=f"{_V03_CUBES_DIM} positions for 0–{_V03_CUBES_DIM - 1} cubes.",
        )
    )
    sub_fields.append(
        stripe_descriptors.SubFieldDescriptor(
            name="opp_action_cubes",
            description="Opponent remaining action cubes this round (0–8), encoded as a 9-dim one-hot.",
            relative_offset=_V03_ROUND_DIM + _V03_CUBES_DIM,
            size=_V03_CUBES_DIM,
            encoding="one-hot",
            value_range="{0, 1}",
            notes=f"{_V03_CUBES_DIM} positions for 0–{_V03_CUBES_DIM - 1} cubes.",
        )
    )

    scalar_base = _V03_ROUND_DIM + _V03_CUBES_DIM * 2
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
    for idx, (name, desc, notes) in enumerate(scalar_entries):
        sub_fields.append(
            stripe_descriptors.SubFieldDescriptor(
                name=name,
                description=desc,
                relative_offset=scalar_base + idx,
                size=1,
                encoding="scalar",
                value_range="[0, ~1]",
                notes=notes,
            )
        )

    return tuple(sub_fields)


def _assert_live_layout_contract() -> None:
    """Import-time pins for the invariants the shim relies on.

    The shim omits the live turn_state stripe, splices a frozen 26-dim misc
    stripe in place of the live 4-dim one, and omits the v0.6 playability
    multi-hot stripes, producing a 790-dim v0.3 vector. The live vector is
    1155 dims (v0.6); the total gap is -365 (_TOTAL_DIM_DELTA).
    """
    actual_misc_dim = layout.STATE_CONT_LAYOUT.size_of("misc_scalars")
    assert actual_misc_dim == 4, (
        f"v0.3 shim expects live misc-scalars stripe to be 4 dims, "
        f"but found {actual_misc_dim}; update the shim"
    )
    actual_turn_dim = layout.STATE_CONT_LAYOUT.size_of("turn_state")
    assert actual_turn_dim == layout.N_PLAYER_TURNS + 1, (
        f"v0.3 shim expects live turn_state stripe to be {layout.N_PLAYER_TURNS + 1} dims, "
        f"but found {actual_turn_dim}; update the shim"
    )
    assert _TOTAL_DIM_DELTA == -365, (
        f"v0.3 shim expects -365 dim delta between live v0.6 and v0.3 vectors "
        f"(-5 for the v0.3→v0.4 turn_state/misc change, -360 for the v0.4→v0.6 "
        f"playability stripes), but computed {_TOTAL_DIM_DELTA}; update the shim"
    )
    # The hand-summary stripe sits after turn_state but before misc_scalars in
    # the live layout, so dropping turn_state shifts it back by exactly the
    # turn_state width (not _TOTAL_DIM_DELTA). Pin the ordering and the delta so a
    # stripe inserted ahead of hand_summary can't silently desync the frozen
    # offset (the 2026-06-14 regression read it at the live column, 27 too far).
    assert (
        layout.STATE_CONT_LAYOUT.offset_of("turn_state")
        < layout.STATE_CONT_LAYOUT.offset_of("hand_summary_me")
        < layout.STATE_CONT_LAYOUT.offset_of("misc_scalars")
    ), "v0.3 shim assumes turn_state < hand_summary_me < misc_scalars in the live layout"
    assert _HAND_SUMMARY_DIM_DELTA == -layout.STATE_CONT_LAYOUT.size_of("turn_state"), (
        f"v0.3 shim expects hand_summary to shift back by the turn_state width "
        f"({layout.STATE_CONT_LAYOUT.size_of('turn_state')}), but computed "
        f"{_HAND_SUMMARY_DIM_DELTA}; update the shim"
    )


_assert_live_layout_contract()
