# pyright: reportPrivateUsage=false
# (tests read the goal_delta stripe via the private layout offsets, matching
# the test_encode.py convention)
"""Tests for the v1.4 -> v1.5 compat shim.

v1.5 was a behavior-only era of its own: no tensor shape changed, but the live
``PlayBirdChoice`` featurizer now prices the ``goal_delta`` stripe at the row's
committed landing habitat, where pre-1.5 rows priced the bird's *card* habitats
(a two-habitat bird advanced a ``birds_<habitat>`` goal on both of its rows).
``compat.v1_4.PolicyValueNetV1_4`` freezes the old pricing by re-filling each
play-bird row's ``goal_delta`` after its parent's encoding — a value-level
change on top of whatever geometry the parent produces, so the shim overrides
``encode_choices`` only.

Since the v1.6 bump, that parent is ``compat.v1_5.PolicyValueNetV1_5`` (was the
live net directly): era 1.4 now inherits the ``goal_delta_ignoring_eggs``
tail-strip too, so its choice geometry is live minus 8, not dims-equal-live.
The refill still targets ``layout._OFF_GOAL_DELTA``, an offset well before the
stripped tail, so it is unaffected by the re-chain.

As for v1.0 / v1.3, a committed LFS checkpoint fixture is deferred:
``test_v1_4_stamped_checkpoint_round_trips`` builds a v1.4-era net, saves it
with a v1.4 stamp, and reloads it through the production ``load_policy_net``
path, then forward-passes a play-bird decision through the frozen encoder.
"""

from __future__ import annotations

import pathlib
import typing

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wingspan import (
    architecture,
    cards,
    compat,
    decisions,
    encode,
    engine,
    model,
    state,
    version,
)
from wingspan.compat import v1_0 as compat_v1_0
from wingspan.compat import v1_3 as compat_v1_3
from wingspan.compat import v1_4 as compat_v1_4
from wingspan.encode import layout
from wingspan.model import core
from wingspan.players import loaders
from wingspan.training import config

_GOAL_DELTA_SLICE = slice(
    layout._OFF_GOAL_DELTA, layout._OFF_GOAL_DELTA + layout._GOAL_DELTA_DIM
)


def _small_arch() -> architecture.ModelArchitecture:
    return architecture.ModelArchitecture(
        trunk_layers=(8, 8),
        choice_layers=(8, 8),
        head_layers=(),
        value_layers=(),
        card_embed_dim=4,
    )


def _era_shim() -> compat_v1_4.PolicyValueNetV1_4:
    """A v1_4 shim built at era 1.4's (v1.6-narrowed) dims."""
    state_dim, choice_dim = compat.encoding_dims_for_era("1.4", encode.DEFAULT_SPEC)
    return compat_v1_4.PolicyValueNetV1_4(
        state_dim=state_dim, choice_dim=choice_dim, arch=_small_arch()
    )


def _live_rows_without_v1_6_tail(
    decision: decisions.Decision[typing.Any], game_state: state.GameState
) -> np.ndarray:
    """Live-encoded choice rows with the v1.6 ``goal_delta_ignoring_eggs`` tail
    stripped — what era 1.4 (inheriting the v1_5 tail-strip) should match
    outside its own ``goal_delta`` refill."""
    live_rows = encode.encode_choices(decision, game_state)
    start = encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_OFFSET
    end = start + encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
    return np.delete(live_rows, slice(start, end), axis=1)


def _wetland_goal_state() -> state.GameState:
    """A fresh game whose round-1 goal is [bird] in [wetland] (unscored)."""
    eng, *_ = engine.Engine.create(seed=100)
    _, _, all_goals = cards.load_all()
    wetland_goal = next(goal for goal in all_goals if goal.category == "birds_wetland")
    eng.state.round_goals = [wetland_goal, *eng.state.round_goals[1:]]
    return eng.state


def _dual_bird() -> cards.Bird:
    """A grassland-or-wetland bird (the Peregrine Falcon shape)."""
    all_birds, *_ = cards.load_all()
    return next(
        bird
        for bird in all_birds
        if set(bird.habitats) == {cards.Habitat.GRASSLAND, cards.Habitat.WETLAND}
    )


def _play_decision(bird: cards.Bird) -> decisions.PlayBirdDecision:
    return decisions.PlayBirdDecision(
        player_id=0,
        prompt="play",
        choices=[
            decisions.PlayBirdChoice(
                label=bird.name, bird=bird, habitat=cards.Habitat.GRASSLAND
            ),
            decisions.PlayBirdChoice(
                label=bird.name, bird=bird, habitat=cards.Habitat.WETLAND
            ),
        ],
    )


def _forward(
    net: core.PolicyValueNet,
    decision: decisions.Decision[typing.Any],
    game_state: state.GameState,
) -> None:
    state_vec = net.encode_state(game_state, decision)
    choice_feats = net.encode_choices(decision, game_state)
    family_idx = decisions.family_index_for(type(decision))
    logits, value = net(
        torch.from_numpy(state_vec).unsqueeze(0),
        torch.from_numpy(choice_feats).unsqueeze(0),
        torch.ones(1, choice_feats.shape[0]),
        torch.tensor([family_idx], dtype=torch.long),
    )
    assert logits.shape == (1, choice_feats.shape[0])
    assert value.shape == (1,)


# ---------------------------------------------------------------------------
# class_for_version routing


class TestClassForVersionRouting:
    def test_v1_4_routes_to_shim(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version("1.4")
            is compat_v1_4.PolicyValueNetV1_4
        )

    def test_earlier_eras_keep_their_shims(self) -> None:
        for era in ("1.1", "1.2", "1.3"):
            assert (
                core.PolicyValueNet.class_for_version(era)
                is compat_v1_3.PolicyValueNetV1_3
            )
        assert (
            core.PolicyValueNet.class_for_version("1.0")
            is compat_v1_0.PolicyValueNetV1_0
        )

    def test_current_version_returns_live_class(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version(version.MODEL_VERSION)
            is core.PolicyValueNet
        )

    def test_v1_3_inherits_the_goal_delta_freeze(self) -> None:
        """Pre-1.4 eras must freeze the old pricing too: their shims subclass
        the v1_4 shim, so the refill runs inside their ``super()`` chain."""
        assert issubclass(
            compat_v1_3.PolicyValueNetV1_3, compat_v1_4.PolicyValueNetV1_4
        )
        assert issubclass(
            compat_v1_0.PolicyValueNetV1_0, compat_v1_4.PolicyValueNetV1_4
        )


# ---------------------------------------------------------------------------
# Dims: era 1.4 state equals live; choice narrower by the inherited v1.6 tail


def test_encoding_dims_for_era_1_4_choice_narrower_by_goal_delta_ignoring_eggs() -> (
    None
):
    """v1.5 itself changed no shape, so era 1.4's choice narrowing stops at the
    inherited v1.6 tail stripe; but the v1.8 bump additionally narrows every
    pre-1.8 era's state_dim by the known_hand_opp stripe, era 1.4 included."""
    spec = encode.DEFAULT_SPEC
    state_dim, choice_dim = compat.encoding_dims_for_era("1.4", spec)
    assert state_dim == encode.state_size(spec) - encode.STATE_KNOWN_HAND_OPP_DIM
    assert choice_dim == (
        encode.choice_feature_dim(spec) - encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
    )


# ---------------------------------------------------------------------------
# The frozen behavior: play-bird rows price goal_delta habitat-agnostically


class TestGoalDeltaFreeze:
    def test_live_rows_differ_by_landing_habitat(self) -> None:
        """The live encoder prices the wetland goal only on the wetland row —
        the v1.5 change the shim exists to reverse."""
        game_state = _wetland_goal_state()
        rows = encode.encode_choices(_play_decision(_dual_bird()), game_state)
        grassland_row, wetland_row = rows
        slot0_count = layout._OFF_GOAL_DELTA + layout._GOAL_DELTA_COUNT
        assert grassland_row[slot0_count] == 0.0
        assert wetland_row[slot0_count] > 0.0

    def test_shim_rows_price_both_habitats(self) -> None:
        """The shim re-fills play-bird rows with the pre-1.5 pricing: both rows
        of a two-habitat bird claim the wetland goal's count and VP delta."""
        game_state = _wetland_goal_state()
        shim = _era_shim()
        grassland_row, wetland_row = shim.encode_choices(
            _play_decision(_dual_bird()), game_state
        )
        slot0_count = layout._OFF_GOAL_DELTA + layout._GOAL_DELTA_COUNT
        assert grassland_row[slot0_count] > 0.0, "agnostic pricing claims both rows"
        assert np.array_equal(
            grassland_row[_GOAL_DELTA_SLICE], wetland_row[_GOAL_DELTA_SLICE]
        )

    def test_shim_touches_only_the_goal_delta_stripe(self) -> None:
        """Outside goal_delta, shim rows are byte-identical to live rows once
        the inherited v1.6 tail stripe is stripped from the live rows — the
        refill must not disturb any other (still-present) stripe."""
        game_state = _wetland_goal_state()
        decision = _play_decision(_dual_bird())
        live_rows = _live_rows_without_v1_6_tail(decision, game_state)
        shim_rows = _era_shim().encode_choices(decision, game_state)
        assert shim_rows.shape == live_rows.shape
        mask = np.ones(live_rows.shape[1], dtype=bool)
        mask[_GOAL_DELTA_SLICE] = False
        assert np.array_equal(shim_rows[:, mask], live_rows[:, mask])

    def test_non_play_decisions_encode_identically_to_live(self) -> None:
        """Decisions without play-bird rows are untouched (candidate rows kept
        the optimistic bound in both eras), once the inherited v1.6 tail
        stripe is stripped from the live rows — a BirdChoice row still fills
        goal_delta_ignoring_eggs stripe, which the era 1.4 shim lacks."""
        game_state = _wetland_goal_state()
        candidate = _dual_bird()
        decision = decisions.BirdPowerTuckFromHandDecision(
            player_id=0,
            prompt="t",
            choices=[decisions.BirdChoice(label=candidate.name, bird=candidate)],
        )
        live_rows = _live_rows_without_v1_6_tail(decision, game_state)
        shim_rows = _era_shim().encode_choices(decision, game_state)
        assert np.array_equal(shim_rows, live_rows)


# ---------------------------------------------------------------------------
# Real load-path round-trip (fixture-equivalent)


def test_v1_4_stamped_checkpoint_round_trips(tmp_path: pathlib.Path) -> None:
    """A v1.4-stamped checkpoint loads under live code via ``load_policy_net``
    as the shim class (at era 1.4's dims — live minus the inherited v1.6
    ``goal_delta_ignoring_eggs`` tail) and forward-passes a play-bird decision
    through the frozen encoder."""
    base = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(
            run_name="v14-roundtrip",
            checkpoint_dir=str(tmp_path),
            games_per_iter=2,
            eval_games=2,
        ),
        architecture=config.ArchitectureConfig(
            main=config.MainNetArchitecture(
                trunk_layers=(8, 8),
                choice_layers=(8, 8),
                head_layers=(),
                value_layers=(),
                card_embed_dim=4,
                card_encoder_layers=(),
                hand_encoder_layers=(8,),
            ),
        ),
    )
    cfg = config.with_encoding_version(base, "1.4")
    assert cfg.encoding_version == "1.4"
    assert cfg.state_dim == (
        encode.state_size(cfg.encoding_spec) - encode.STATE_KNOWN_HAND_OPP_DIM
    )
    assert cfg.choice_dim == (
        encode.choice_feature_dim(cfg.encoding_spec)
        - encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
    )

    net_cls = model.PolicyValueNet.class_for_version(cfg.encoding_version)
    assert net_cls is compat_v1_4.PolicyValueNetV1_4
    net = net_cls(
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        num_families=len(cfg.family_order),
        arch=cfg.arch,
        spec=cfg.encoding_spec,
    )

    ckpt = tmp_path / "v14.pt"
    torch.save(
        {"config": cfg.model_dump(), "model": net.state_dict(), "version": "1.4"},
        ckpt,
    )

    loaded, saved_cfg = loaders.load_policy_net(ckpt, torch.device("cpu"))
    assert isinstance(loaded, compat_v1_4.PolicyValueNetV1_4)
    assert saved_cfg.encoding_version == "1.4"

    game_state = _wetland_goal_state()
    _forward(loaded, _play_decision(_dual_bird()), game_state)
