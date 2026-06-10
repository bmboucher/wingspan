# pyright: reportPrivateUsage=false
# (the v0.2 -> v0.3 shim reads the live layout's package-private stripe offsets
# and calls the frozen v0.2 misc-scalar encoder -- a deliberate compat coupling,
# pinned by the import-time layout-contract assertions below)
"""Frozen v0.2 misc-scalar encoding: the shim that keeps pre-0.3 artifacts playable.

Artifact version 0.3 replaced three raw scalars in ``_summary_misc_scalars``
with one-hot vectors, growing the state vector by 19 dims (771 → 790):

* round_idx scalar (1 dim) → 4-dim one-hot for rounds 0–3.
* my action_cubes_left scalar (1 dim) → 9-dim one-hot for 0–8 cubes.
* opp action_cubes_left scalar (1 dim) → 9-dim one-hot for 0–8 cubes.

Nets trained before 0.3 have a 771-dim state trunk input; this module keeps
them loadable:

* :func:`encode_misc_scalars_v02` rebuilds the 7-element scalar stripe using
  the frozen v0.2 geometry (round ÷ 3, cubes ÷ 8) and can be spliced in via
  :func:`encode_state_v02`.
* :func:`encode_state_v02` produces the complete 771-dim state vector (the old
  scalar misc stripe, everything else live) for a v0.2 checkpoint.
* :class:`PolicyValueNetV02` overrides :meth:`encode_state` so inference call
  sites drive the net with the 771-dim input its weights expect, and
  :meth:`_state_embed_offsets` so the trunk *slices* that input at the v0.2
  offsets it was written with (the live 790-dim offsets sit 19 columns too far
  right; the widths coincide, so a live slice corrupts silently rather than
  crashing).
* :func:`state_embed_offsets_v02` is the frozen-geometry slice offsets the pre-0.3
  nets feed to ``_embed_state`` (also used by ``PolicyValueNetV01``/``V00``).
* :func:`state_feature_dim_v02` is the frozen 771-dim state width itself — what
  the era-dims router (``compat.encoding_dims_for_era``) hands an era-pinned
  ``TrainConfig`` so a resumed pre-0.3 run keeps training at its own geometry.
* :func:`state_stripe_layout_v02` produces the frozen stripe registry matching
  the 771-dim v0.2 state vector, for consistent reporting of old checkpoints.

Per the compatibility policy (``CLAUDE.md``), this shim lives until a MAJOR
``MODEL_VERSION`` bump deletes it together with the v0.2 fixture set.
"""

from __future__ import annotations

import numpy as np

from wingspan import decisions, state, version
from wingspan.encode import layout, state_encode
from wingspan.encode.stripes import descriptors as stripe_descriptors
from wingspan.model import core

MISC_SCALARS_CHANGED_IN = "0.3"
"""The artifact version whose misc-scalar reshape this module undoes."""

# ---------------------------------------------------------------------------
# Frozen v0.2 misc-scalar geometry.  These mirror the exact arithmetic that
# _summary_misc_scalars applied before the one-hot change, deliberately
# captured as literals so the shim does not depend on live _ACTION_CUBES_SCALE
# or a hypothetical future refactor of those scalars.

_V02_MISC_DIM = 7  # 1 round scalar + 2 cube scalars + 2 goal-pts + 1 tray + 1 deck
_V02_ROUND_SCALE = 3.0  # round_idx / 3.0 → value in [0, 1]
_V02_CUBES_SCALE = 8.0  # action_cubes_left / 8.0 → value in [0, 1]
_V02_GOAL_PTS_SCALE = 10.0  # round_goal_points / 10.0
_V02_TRAY_SCALE = 3.0  # tray occupancy / 3.0
_V02_DECK_SCALE = 100.0  # deck size / 100.0

# The signed width change the 0.3 misc-scalar reshape made to the continuous
# prefix: the frozen 7-dim v0.2 stripe minus the live 26-dim one-hot stripe,
# i.e. -19.  Every stripe after misc_scalars — the card-index block, the hand
# multi-hot, and the decision-type tail — sits 19 columns earlier in the 771-dim
# v0.2 vector than in the live one.  Pinned to -19 by _assert_live_layout_contract.
_MISC_DIM_DELTA = _V02_MISC_DIM - (
    layout.N_ROUNDS + (layout.MAX_ACTION_CUBES + 1) * 2 + 4
)


def uses_v0_2_state_encoding(artifact_version: str) -> bool:
    """Whether ``artifact_version`` predates the 0.3 misc-scalar one-hot reshape
    and therefore needs this module's frozen geometry to load and play."""
    parsed = version.parse_version(artifact_version)
    changed = version.parse_version(MISC_SCALARS_CHANGED_IN)
    return (parsed.major, parsed.minor) < (changed.major, changed.minor)


def encode_misc_scalars_v02(
    game_state: state.GameState, me: state.Player, opp: state.Player
) -> np.ndarray:
    """The frozen v0.2 7-element misc-scalar stripe (scalar round + scalar cubes).

    Reproduces the exact pre-0.3 ``_summary_misc_scalars`` output: round_idx
    divided by 3, action_cubes_left divided by 8 for each player, then goal pts,
    tray size, and deck size as before."""
    return np.array(
        [
            game_state.round_idx / _V02_ROUND_SCALE,
            me.action_cubes_left / _V02_CUBES_SCALE,
            opp.action_cubes_left / _V02_CUBES_SCALE,
            me.round_goal_points / _V02_GOAL_PTS_SCALE,
            opp.round_goal_points / _V02_GOAL_PTS_SCALE,
            sum(1 for bird in game_state.tray if bird is not None) / _V02_TRAY_SCALE,
            len(game_state.bird_deck) / _V02_DECK_SCALE,
        ],
        dtype=np.float32,
    )


def encode_state_v02(
    game_state: state.GameState,
    decision: decisions.Decision[decisions.Choice] | None = None,
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
) -> np.ndarray:
    """Produce the complete 771-dim v0.2 state vector.

    Identical to the live ``encode_state`` except that ``_summary_misc_scalars``
    is replaced with the frozen 7-dim scalar stripe, keeping ``state_dim``
    at 771.  Everything else (board, tray, hand, goals, decision-type) uses
    the live encoders unchanged."""
    pov = decision.player_id if decision is not None else game_state.current_player
    me = game_state.players[pov]
    opp = game_state.players[1 - pov] if len(game_state.players) > 1 else me

    # Collect stripes in the same order as encode_state, swapping only misc_scalars.
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
        encode_misc_scalars_v02(game_state, me, opp),  # frozen 7-dim v0.2 stripe
        state_encode._round_goals_all_rounds(game_state, me),
        state_encode._card_index_block(me, opp, game_state),
        state_encode._hand_identity(me),
        state_encode._encode_decision_type(decision, spec),
    ]
    return np.concatenate(parts).astype(np.float32)


def state_embed_offsets_v02() -> tuple[int, int, int]:
    """The ``(card-index, hand-multi-hot, decision-type)`` slice offsets for the
    frozen 771-dim v0.2 state vector.

    These are the live ``encode.layout`` offsets shifted back by
    :data:`_MISC_DIM_DELTA` (-19), because the 0.3 reshape grew the misc-scalar
    stripe — which precedes all three — by 19 dims. ``PolicyValueNetV02`` (and
    the pre-0.2 nets, which also feed the 771-dim vector) override
    ``_embed_state``'s offsets with this, so an old checkpoint's state vector is
    sliced at the columns it was written with rather than the live ones (the
    widths coincide, so a live slice would corrupt silently — see
    ``compat/INDEX.md``)."""
    return (
        layout.OFF_CARD_INDEX + _MISC_DIM_DELTA,
        layout.OFF_HAND_MULTIHOT + _MISC_DIM_DELTA,
        layout.OFF_DECISION_TYPE + _MISC_DIM_DELTA,
    )


def state_feature_dim_v02(spec: layout.EncodingSpec = layout.DEFAULT_SPEC) -> int:
    """The frozen v0.2 state-vector width (771 under the default spec).

    The live width shifted by :data:`_MISC_DIM_DELTA` (-19) — the size of
    :func:`encode_state_v02`'s output and the ``state_dim`` every pre-0.3 net
    was built with. The era-dims router (``compat.encoding_dims_for_era``) uses
    this so an era-pinned ``TrainConfig`` derives the dims its checkpoints
    actually carry."""
    return layout.state_feature_dim(spec) + _MISC_DIM_DELTA


class PolicyValueNetV02(core.PolicyValueNet):
    """A :class:`~wingspan.model.core.PolicyValueNet` frozen to the v0.2
    misc-scalar state geometry, for checkpoints written before artifact version 0.3.

    The state trunk was trained against 771-dim inputs (scalar round + scalar
    cubes); this subclass overrides :meth:`encode_state` to keep that width
    and feed the frozen v0.2 state vector.  Choice encoding and the card
    encoder are identical to the live era.  Constructed by the version-routing
    loaders (``PolicyValueNet.from_model_config``,
    ``players.loaders.load_policy_net``) — never by the training pipeline.
    """

    def encode_state(
        self,
        game_state: state.GameState,
        decision: decisions.Decision[decisions.Choice],
    ) -> np.ndarray:
        """Featurize ``game_state`` using the frozen v0.2 7-scalar misc stripe,
        yielding a 771-dim vector this checkpoint's trunk expects."""
        return encode_state_v02(game_state, decision, self.spec)

    def _state_embed_offsets(self) -> tuple[int, int, int]:
        """Slice the 771-dim v0.2 state vector at its own frozen offsets rather
        than the live 790-dim ones — without this the trunk reads the
        card-index / hand / decision stripes 19 columns too far right (the
        widths coincide, so it would corrupt silently, not crash)."""
        return state_embed_offsets_v02()


def state_stripe_layout_v02(
    spec: layout.EncodingSpec = layout.DEFAULT_SPEC,
    card_embed_dim: int = 64,
    *,
    use_distinct_hand_model: bool = False,
    hand_embed_dim: int | None = None,
    tray_set_embedding: bool = False,
) -> stripe_descriptors.VectorLayout:
    """The frozen v0.2 state stripe registry — identical to the live registry
    except the misc_scalars stripe is 7 dims (scalar round + scalar cubes)
    rather than the v0.3 26-dim one-hot encoding.

    Used by the reporting seam in ``runmeta.state_layout_for`` when the
    descriptor's version predates 0.3, so the report matches the checkpoint's
    actual 771-dim state vector."""
    from wingspan.encode.stripes import state as live_state_stripes

    # Obtain the live layout (which has the 26-dim misc stripe) and patch it.
    live_layout = live_state_stripes.state_stripe_layout(
        spec,
        card_embed_dim,
        use_distinct_hand_model=use_distinct_hand_model,
        hand_embed_dim=hand_embed_dim,
        tray_set_embedding=tray_set_embedding,
    )

    # Rebuild the stripe list, replacing misc_scalars with the frozen 7-dim version
    # and adjusting every subsequent stripe's offset by the -19 delta (the same
    # shift state_embed_offsets_v02 applies to the model's slice offsets).
    patched: list[stripe_descriptors.StripeDescriptor] = []
    seen_misc = False
    for stripe in live_layout.stripes:
        if stripe.name == "misc_scalars" and not seen_misc:
            seen_misc = True
            patched.append(
                stripe_descriptors.StripeDescriptor(
                    name="misc_scalars",
                    description=(
                        "Miscellaneous scalar game state (round, cubes, scores, deck) "
                        "— v0.2 scalar encoding."
                    ),
                    offset=stripe.offset,
                    size=_V02_MISC_DIM,
                    encoding="vector",
                    value_range="[0, ~1]",
                    notes=(
                        "7 values in order: round_index (÷3, ordinal), my_action_cubes (÷8), "
                        "opp_action_cubes (÷8), my_round_goal_pts (÷10), "
                        "opp_round_goal_pts (÷10), tray_size (÷3), deck_size (÷100). "
                        "(v0.2 scalar encoding — replaced by one-hots in v0.3)"
                    ),
                    sub_fields=_misc_scalars_v02_sub_fields(stripe.offset),
                )
            )
        elif seen_misc:
            # Shift all subsequent stripes back by 19 dims (the delta).
            patched.append(
                stripe_descriptors.StripeDescriptor(
                    name=stripe.name,
                    description=stripe.description,
                    offset=stripe.offset + _MISC_DIM_DELTA,
                    size=stripe.size,
                    encoding=stripe.encoding,
                    value_range=stripe.value_range,
                    notes=stripe.notes,
                    sub_fields=stripe.sub_fields,
                )
            )
        else:
            patched.append(stripe)

    total = sum(s.size for s in patched)
    return stripe_descriptors.VectorLayout(total_size=total, stripes=tuple(patched))


###### PRIVATE #######


def _misc_scalars_v02_sub_fields(
    stripe_offset: int,
) -> tuple[stripe_descriptors.SubFieldDescriptor, ...]:
    """7 frozen sub-fields for the v0.2 misc-scalars stripe (scalar round + cubes)."""
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
        stripe_descriptors.SubFieldDescriptor(
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


def _assert_live_layout_contract() -> None:
    """Import-time pins for the invariants the shim relies on.

    The shim splices its frozen 7-dim misc stripe into a live state vector,
    which requires the live ``_CONT_PREFIX_DIM`` to account for 26 dims in
    that position (N_ROUNDS + 2*(MAX_ACTION_CUBES+1) + 4 = 4 + 18 + 4 = 26)
    and the v0.2 state_dim to be 771 (live minus the 19 added dims).
    """
    live_misc_dim = layout.N_ROUNDS + (layout.MAX_ACTION_CUBES + 1) * 2 + 4
    assert live_misc_dim == 26, (
        f"v0.2 shim expects live misc-scalars stripe to be 26 dims, "
        f"but computed {live_misc_dim}; update the shim"
    )
    assert layout.N_ROUNDS == 4, (
        f"v0.2 shim freezes N_ROUNDS at 4, but live value is {layout.N_ROUNDS}; "
        "update the shim"
    )
    assert layout.MAX_ACTION_CUBES == 8, (
        f"v0.2 shim freezes MAX_ACTION_CUBES at 8, but live value is "
        f"{layout.MAX_ACTION_CUBES}; update the shim"
    )
    # The model's frozen slice offsets (state_embed_offsets_v02) and the
    # reporting registry both shift by this; -19 is the pre-0.3 vector's lag.
    assert _MISC_DIM_DELTA == -19, (
        f"v0.2 shim expects the misc-scalar reshape to shift later stripes by "
        f"-19, but computed {_MISC_DIM_DELTA}; update the shim"
    )


_assert_live_layout_contract()
