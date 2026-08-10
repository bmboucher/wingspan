# pyright: reportPrivateUsage=false
# (tests access _state_embed_offsets / _choice_embed_offsets and the package-
# private layout offset constants to pin the frozen seams, matching the
# test_compat_v1_3.py convention)
"""Tests for the pre-1.5 -> v1.5 compat shim (``wingspan.compat.v1_4``): the
merged main-net and setup-net freezes for era 1.4.

v1.5 folds four changes that landed on main as provisionally-numbered eras
1.5, 1.6, 1.7, and 1.8 — none of which ever trained a run — into one era.
:class:`wingspan.compat.v1_4.PolicyValueNetV1_4` /
:class:`wingspan.compat.v1_4.SetupNetV1_4` freeze all four for a pre-1.5
checkpoint:

1. the per-opponent ``known_hand_opp`` 180-wide **state** stripe, stripped
   from ``encode_state`` (only ``decision_type`` shifts in
   ``_state_embed_offsets``);
2. the 8-dim ``goal_delta_ignoring_eggs`` **choice** tail stripe, stripped
   from ``encode_choices`` (only ``kept_multihot`` shifts);
3. the habitat-conditioned play-bird ``goal_delta`` pricing (main net) and
   the egg-aware setup ``goal_affinity`` pricing (setup net), both frozen at
   their pre-1.5 values via refill after live encoding;
4. the egg-optimistic bonus *potential* pricing (both nets) and the
   spend-decision ``pay_food`` routing (main net only), both frozen at their
   pre-1.7 values via refill after live encoding.

This file supersedes the four now-deleted per-era files
(``test_compat_v1_5.py``, ``test_compat_v1_6.py``, ``test_compat_v1_7.py``,
and the old narrower ``test_compat_v1_4.py``) — every test below is either
ported unchanged, retargeted at the merged class, or adapted for the now-8-
narrower choice geometry at era 1.4 (previously the bonus-potential and
spend-food freezes were tested at live-equal choice width; the tail strip
now always applies alongside them). As for every prior era, a committed LFS
checkpoint fixture is deferred: the round-trip tests build v1.4-era nets
(main and setup), save them with a v1.4 stamp, and reload them through the
production ``load_policy_net`` / ``load_setup_net`` paths.
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
    setup_model,
    state,
    version,
)
from wingspan.compat import v1_0 as compat_v1_0
from wingspan.compat import v1_3 as compat_v1_3
from wingspan.compat import v1_4 as compat_v1_4
from wingspan.encode import choice_encode, layout
from wingspan.engine import scoring
from wingspan.model import core
from wingspan.players import loaders
from wingspan.setup_model import architecture as setup_arch_module
from wingspan.training import artifacts, config, loop_checkpoint, runmeta
from wingspan.training import setup_net as setup_net_module
from wingspan.training import setup_runmeta

_GOAL_DELTA_SLICE = slice(
    layout._OFF_GOAL_DELTA, layout._OFF_GOAL_DELTA + layout._GOAL_DELTA_DIM
)
_HAND_IDX = layout._OFF_BONUS_VALUE + layout._BONUS_VALUE_HAND
_TRAY_IDX = layout._OFF_BONUS_VALUE + layout._BONUS_VALUE_TRAY

_BIRDS, _BONUSES, _GOALS = cards.load_all()
_BONUS_BY_NAME = {bonus_card.name: bonus_card for bonus_card in _BONUSES}
_BIG_NEST = [bird for bird in _BIRDS if bird.egg_limit >= 4]


def _eot_bird_rows() -> set[int]:
    """Card-table row indices (``bird_index + 1``) of every catalog bird
    carrying a ``DRAW_CARDS_THEN_DISCARD_EOT`` power effect — the birds whose
    power_ex block differs between the live (post-v1.5) and era-1.4 (frozen,
    pre-amend) card tables."""
    return {
        cards.bird_index(bird) + 1
        for bird in _BIRDS
        if any(
            effect.kind == cards.EffectKind.DRAW_CARDS_THEN_DISCARD_EOT
            for effect in bird.power.effects
        )
    }


def _small_arch() -> architecture.ModelArchitecture:
    return architecture.ModelArchitecture(
        trunk_layers=(8, 8),
        choice_layers=(8, 8),
        head_layers=(),
        value_layers=(),
        card_embed_dim=4,
    )


def _era_shim(
    era: str = "1.4",
    arch: architecture.ModelArchitecture | None = None,
    spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
) -> compat_v1_4.PolicyValueNetV1_4:
    """A v1_4 shim built at ``era``'s (narrow) dims — exactly how the load path
    (``encoding_dims_for_era`` -> constructor) builds it."""
    arch = arch or _small_arch()
    state_dim, choice_dim = compat.encoding_dims_for_era(era, spec)
    return compat_v1_4.PolicyValueNetV1_4(
        state_dim=state_dim, choice_dim=choice_dim, arch=arch, spec=spec
    )


def _v1_3_era_net(
    spec: encode.EncodingSpec = encode.DEFAULT_SPEC,
) -> compat_v1_3.PolicyValueNetV1_3:
    """A v1_3 shim built at era 1.3's dims (both v1.4 stripes plus the
    inherited v1.5 known_hand_opp / goal_delta_ignoring_eggs stripes
    stripped)."""
    state_dim, choice_dim = compat.encoding_dims_for_era("1.3", spec)
    return compat_v1_3.PolicyValueNetV1_3(
        state_dim=state_dim, choice_dim=choice_dim, arch=_small_arch(), spec=spec
    )


def _decision() -> decisions.MainActionDecision:
    return decisions.MainActionDecision(
        player_id=0,
        prompt="action",
        choices=[
            decisions.MainActionChoice(
                label="food", action=decisions.MainAction.GAIN_FOOD
            ),
            decisions.MainActionChoice(
                label="eggs", action=decisions.MainAction.LAY_EGGS
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


def _live_rows_without_v1_5_tail(
    decision: decisions.Decision[typing.Any], game_state: state.GameState
) -> np.ndarray:
    """Live-encoded choice rows with the v1.5 ``goal_delta_ignoring_eggs``
    tail stripped — what era 1.4's shim should match outside its own value
    refills (all of which target offsets before the stripped tail)."""
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


def _eggy_hand_state() -> state.GameState:
    """A game state whose deciding player holds three 4+-egg-capacity birds and
    whose tray shows one more — nonzero Breeding Manager potentials under the
    live pricing, zero under the frozen static pricing."""
    eng, *_ = engine.Engine.create(seed=110)
    eng.state.players[0].hand = list(_BIG_NEST[:3])
    eng.state.tray = [_BIG_NEST[3], None, None]
    return eng.state


def _pick_bonus_decision(
    bonus_card: cards.BonusCard,
) -> decisions.BirdPowerPickBonusCardDecision:
    return decisions.BirdPowerPickBonusCardDecision(
        player_id=0,
        prompt="x",
        choices=[
            decisions.BonusCardChoice(label=bonus_card.name, bonus_card=bonus_card)
        ],
    )


def _small_main_arch_for_setup() -> architecture.ModelArchitecture:
    """A tiny main architecture — shapes the setup net's frozen embedder
    copies."""
    return architecture.ModelArchitecture(
        trunk_layers=(8, 8),
        choice_layers=(8, 8),
        head_layers=(),
        value_layers=(),
        card_embed_dim=4,
        card_encoder_layers=(),
        hand_encoder_layers=(8,),
    )


def _small_setup_arch() -> setup_model.SetupArchitecture:
    return setup_model.SetupArchitecture(
        trunk_layers=(8,), choice_layers=(8,), head_layers=(8,), value_layers=()
    )


def _setup_nets(
    encoding: setup_model.SetupEncoding,
) -> tuple[setup_net_module.SetupNet, compat_v1_4.SetupNetV1_4]:
    live_net = setup_net_module.SetupNet(
        encoding=encoding,
        arch=_small_setup_arch(),
        main_arch=_small_main_arch_for_setup(),
    )
    shim_net = compat_v1_4.SetupNetV1_4(
        encoding=encoding,
        arch=_small_setup_arch(),
        main_arch=_small_main_arch_for_setup(),
    )
    return live_net, shim_net


def _setup_context(goal_categories: tuple[str, ...]) -> setup_model.SetupContext:
    return setup_model.SetupContext(
        tray_birds=(None, None, None),
        birdfeeder_counts=(0, 0, 0, 0, 0, 0),
        round_goal_categories=goal_categories,
    )


def _star_nest_candidate() -> (
    tuple[setup_model.SetupCandidate, setup_model.SetupContext]
):
    """A single-bird keep whose bird is star-nested (wild) against a
    bowl-with-eggs (nest) goal: nonzero under the live played-and-
    egg-populated pricing, zero under the frozen pre-1.5 play-instant
    pricing (a freshly played bird has no eggs)."""
    star_bird = _BIRDS[0].model_copy(
        update={"nest": cards.NestType.STAR, "egg_limit": 4}
    )
    candidate = setup_model.SetupCandidate(
        kept_cards=(star_bird,), kept_foods=(), bonus_card=None
    )
    context = _setup_context(("bowl_birds_with_eggs",) + ("birds_forest",) * 3)
    return candidate, context


# ---------------------------------------------------------------------------
# (1) class_for_version routing + chain shape


class TestClassForVersionRouting:
    def test_v1_4_routes_to_shim(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version("1.4")
            is compat_v1_4.PolicyValueNetV1_4
        )
        assert (
            setup_net_module.SetupNet.class_for_version("1.4")
            is compat_v1_4.SetupNetV1_4
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

    def test_setup_era_1_3_also_routes_to_the_v1_4_shim(self) -> None:
        """Harmless: a real pre-1.4 ``setup.pt`` differs in shape (the v1.3
        two-tower restructure) and would fail at ``load_state_dict``
        regardless of which class builds it; ``load_setup_net`` already
        turns that into a clear "retrain the setup model" error."""
        assert (
            setup_net_module.SetupNet.class_for_version("1.3")
            is compat_v1_4.SetupNetV1_4
        )

    def test_current_version_returns_live_classes(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version(version.MODEL_VERSION)
            is core.PolicyValueNet
        )
        assert (
            setup_net_module.SetupNet.class_for_version(version.MODEL_VERSION)
            is setup_net_module.SetupNet
        )

    def test_inheritance_chain_v1_0_through_v1_4_to_core(self) -> None:
        """v1_0 subclass v1_3 subclass v1_4 subclass core.PolicyValueNet —
        the short chain this collapse leaves behind (was v1_0 -> v1_3 -> v1_4
        -> v1_5 -> v1_6 -> v1_7 -> core before the merge)."""
        assert issubclass(
            compat_v1_0.PolicyValueNetV1_0, compat_v1_3.PolicyValueNetV1_3
        )
        assert issubclass(
            compat_v1_3.PolicyValueNetV1_3, compat_v1_4.PolicyValueNetV1_4
        )
        assert issubclass(compat_v1_4.PolicyValueNetV1_4, core.PolicyValueNet)


# ---------------------------------------------------------------------------
# (2) Composed dims table


class TestComposedDimsAcrossEras:
    def test_composed_dims_across_every_era(self) -> None:
        """Every era's dims, composed bottom-up from the known stripe widths
        (each individually documented on ``compat.encoding_dims_for_era``)
        rather than hardcoded, so a future stripe addition cannot silently
        desync this test from the real branches."""
        spec = encode.DEFAULT_SPEC
        live_state = encode.state_size(spec)
        live_choice = encode.choice_feature_dim(spec)
        assert (live_state, live_choice) == (1309, 517)

        known_hand_opp = encode.STATE_KNOWN_HAND_OPP_DIM
        goal_delta_tail = encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        food_unlock = 2 * encode.STATE_FOOD_UNLOCK_DIM
        resets_feeder = encode.CHOICE_RESETS_FEEDER_DIM
        becomes_unplayable = encode.CHOICE_BECOMES_UNPLAYABLE_DIM

        expected: dict[str, tuple[int, int]] = {
            version.MODEL_VERSION: (live_state, live_choice),
            "1.4": (live_state - known_hand_opp, live_choice - goal_delta_tail),
            "1.3": (
                live_state - known_hand_opp - food_unlock,
                live_choice - goal_delta_tail - resets_feeder,
            ),
            "1.0": (
                live_state - known_hand_opp - food_unlock,
                live_choice - goal_delta_tail - resets_feeder - becomes_unplayable,
            ),
        }
        for era, dims in expected.items():
            assert compat.encoding_dims_for_era(era, spec) == dims, era


# ---------------------------------------------------------------------------
# (3) State geometry: known_hand_opp stripped, only decision_type shifts


class TestStateGeometry:
    def test_encode_state_matches_live_with_stripe_deleted(self) -> None:
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live = encode.encode_state(eng.state, decision)
        start = encode.STATE_KNOWN_HAND_OPP_OFFSET
        end = start + encode.STATE_KNOWN_HAND_OPP_DIM
        live_stripped = np.delete(live, slice(start, end), axis=0)
        shim_out = shim.encode_state(eng.state, decision)
        assert shim_out.shape == live_stripped.shape
        assert np.array_equal(shim_out, live_stripped)

    def test_width_is_live_minus_known_hand_opp_dim(self) -> None:
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live_len = encode.encode_state(eng.state, decision).shape[0]
        shim_len = shim.encode_state(eng.state, decision).shape[0]
        assert live_len - shim_len == encode.STATE_KNOWN_HAND_OPP_DIM

    def test_card_index_and_hand_multihot_unchanged(self) -> None:
        """The stripe is appended after BOTH playability multi-hots, so
        card_index / hand_multihot never move (the opposite shape from
        v1_3's food-unlock strip, which precedes card_index and shifts it)."""
        arch = _small_arch()
        live_off = core.PolicyValueNet(arch=arch)._state_embed_offsets()
        shim_off = _era_shim(arch=arch)._state_embed_offsets()
        assert shim_off.card_index == live_off.card_index
        assert shim_off.hand_multihot == live_off.hand_multihot
        assert shim_off.hand_summary == live_off.hand_summary
        assert shim_off.hand_summary_end == live_off.hand_summary_end

    def test_decision_type_shifted_left_by_known_hand_opp_dim(self) -> None:
        arch = _small_arch()
        live_off = core.PolicyValueNet(arch=arch)._state_embed_offsets()
        shim_off = _era_shim(arch=arch)._state_embed_offsets()
        assert (
            live_off.decision_type - shim_off.decision_type
            == encode.STATE_KNOWN_HAND_OPP_DIM
        )

    def test_v1_3_composes_its_own_shift_on_top_of_v1_4s(self) -> None:
        """v1_3's card_index / hand_multihot shift by its own food-unlock
        width only (known_hand_opp never touches them); decision_type
        carries BOTH shifts — v1_3's own food-unlock width plus the
        inherited v1_4 known_hand_opp width."""
        arch = _small_arch()
        live_off = core.PolicyValueNet(arch=arch)._state_embed_offsets()
        v1_3_off = compat_v1_3.PolicyValueNetV1_3(arch=arch)._state_embed_offsets()
        food_unlock = 2 * encode.STATE_FOOD_UNLOCK_DIM
        assert live_off.card_index - v1_3_off.card_index == food_unlock
        assert live_off.hand_multihot - v1_3_off.hand_multihot == food_unlock
        assert live_off.decision_type - v1_3_off.decision_type == (
            food_unlock + encode.STATE_KNOWN_HAND_OPP_DIM
        )


# ---------------------------------------------------------------------------
# (4) Choice geometry: goal_delta_ignoring_eggs tail stripped


class TestChoiceGeometry:
    def test_encode_choices_matches_live_with_tail_stripped(self) -> None:
        """Beyond the tail strip, the shim also zeroes each MainActionChoice
        row's v1.5-amended forecast cells (module docstring change 5) — apply
        the same refill to the tail-stripped live rows before comparing, so
        this test still isolates pure geometry/refill equivalence rather than
        asserting an invariant the amend intentionally broke."""
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live_stripped = _live_rows_without_v1_5_tail(decision, eng.state)
        for row, choice in zip(live_stripped, decision.choices):
            choice_encode.refill_main_action_forecast_zeros(row, choice.action)
        shim_out = shim.encode_choices(decision, eng.state)
        assert shim_out.shape == live_stripped.shape
        assert np.array_equal(shim_out, live_stripped)

    def test_encode_choices_narrower_than_live_by_tail_dim(self) -> None:
        eng, *_ = engine.Engine.create(seed=100)
        shim = _era_shim()
        decision = _decision()
        live_cols = encode.encode_choices(decision, eng.state).shape[1]
        shim_cols = shim.encode_choices(decision, eng.state).shape[1]
        assert live_cols - shim_cols == encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM

    def test_bird_id_and_becomes_offsets_unshifted(self) -> None:
        """bird_id / becomes_playable / becomes_unplayable precede the tail
        stripe, so their offsets are identical between live and the shim."""
        arch = _small_arch()
        live = core.PolicyValueNet(arch=arch)._choice_embed_offsets()
        shim = _era_shim(arch=arch)._choice_embed_offsets()
        assert shim.bird_id == live.bird_id
        assert shim.becomes_playable == live.becomes_playable
        assert shim.becomes_unplayable == live.becomes_unplayable

    def test_kept_multihot_shifted_left_with_include_setup(self) -> None:
        arch = _small_arch()
        spec = encode.EncodingSpec(include_setup=True)
        live = core.PolicyValueNet(spec=spec, arch=arch)._choice_embed_offsets()
        shim = _era_shim(arch=arch, spec=spec)._choice_embed_offsets()
        assert live.kept_multihot is not None
        assert shim.kept_multihot is not None
        assert (
            live.kept_multihot - shim.kept_multihot
            == encode.CHOICE_GOAL_DELTA_IGNORING_EGGS_DIM
        )


# ---------------------------------------------------------------------------
# (5) goal_delta freeze: play-bird rows price habitat-agnostically


class TestGoalDeltaFreeze:
    def test_live_rows_differ_by_landing_habitat(self) -> None:
        """The live encoder prices the wetland goal only on the wetland row —
        the change the shim exists to reverse."""
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
        the goal_delta_ignoring_eggs tail is stripped — the refill must not
        disturb any other (still-present) stripe."""
        game_state = _wetland_goal_state()
        decision = _play_decision(_dual_bird())
        live_rows = _live_rows_without_v1_5_tail(decision, game_state)
        shim_rows = _era_shim().encode_choices(decision, game_state)
        assert shim_rows.shape == live_rows.shape
        mask = np.ones(live_rows.shape[1], dtype=bool)
        mask[_GOAL_DELTA_SLICE] = False
        assert np.array_equal(shim_rows[:, mask], live_rows[:, mask])

    def test_non_play_decisions_encode_identically_to_live(self) -> None:
        """Decisions without play-bird rows are untouched (candidate rows kept
        the optimistic bound in both eras), once the goal_delta_ignoring_eggs
        tail is stripped from the live rows — a BirdChoice row still fills
        that stripe, which the era 1.4 shim lacks."""
        game_state = _wetland_goal_state()
        candidate = _dual_bird()
        decision = decisions.BirdPowerTuckFromHandDecision(
            player_id=0,
            prompt="t",
            choices=[decisions.BirdChoice(label=candidate.name, bird=candidate)],
        )
        live_rows = _live_rows_without_v1_5_tail(decision, game_state)
        shim_rows = _era_shim().encode_choices(decision, game_state)
        assert np.array_equal(shim_rows, live_rows)


# ---------------------------------------------------------------------------
# (6) Bonus-potential freeze: static (egg-blind) pricing on the main net


class TestMainNetBonusPotentialFreeze:
    def test_shim_freezes_static_potentials_live_encoder_does_not(self) -> None:
        game_state = _eggy_hand_state()
        decision = _pick_bonus_decision(_BONUS_BY_NAME["Breeding Manager"])
        live_stripped = _live_rows_without_v1_5_tail(decision, game_state)
        shim_rows = _era_shim().encode_choices(decision, game_state)
        assert shim_rows.shape == live_stripped.shape  # both narrowed by the tail

        assert np.isclose(live_stripped[0][_HAND_IDX], 3.0 / layout._BONUS_COUNT_SCALE)
        assert np.isclose(live_stripped[0][_TRAY_IDX], 1.0 / layout._BONUS_COUNT_SCALE)
        assert shim_rows[0][_HAND_IDX] == 0.0
        assert shim_rows[0][_TRAY_IDX] == 0.0

        outside = np.ones(live_stripped.shape[1], dtype=bool)
        outside[[_HAND_IDX, _TRAY_IDX]] = False
        assert np.array_equal(live_stripped[0][outside], shim_rows[0][outside])

    def test_shim_preserves_hand_counting_card_full_source(self) -> None:
        """The refill re-runs the generic static predicate, not a zeroing:
        Visionary Leader's full-hand ``hand_potential`` — unchanged across
        eras — survives the shim byte-identically."""
        game_state = _eggy_hand_state()
        decision = _pick_bonus_decision(_BONUS_BY_NAME["Visionary Leader"])
        live_stripped = _live_rows_without_v1_5_tail(decision, game_state)
        shim_rows = _era_shim().encode_choices(decision, game_state)
        assert np.isclose(live_stripped[0][_HAND_IDX], 3.0 / layout._BONUS_COUNT_SCALE)
        assert np.array_equal(live_stripped, shim_rows)

    def test_shim_matches_live_on_static_card_rows(self) -> None:
        """A static type-counting card's potentials never changed, so the shim
        output equals the (tail-stripped) live encoding byte-for-byte."""
        game_state = _eggy_hand_state()
        decision = _pick_bonus_decision(_BONUS_BY_NAME["Bird Feeder"])
        live_stripped = _live_rows_without_v1_5_tail(decision, game_state)
        shim_rows = _era_shim().encode_choices(decision, game_state)
        assert np.array_equal(live_stripped, shim_rows)

    def test_forward_pass_runs_at_era_dims(self) -> None:
        game_state = _eggy_hand_state()
        _forward(
            _era_shim(),
            _pick_bonus_decision(_BONUS_BY_NAME["Breeding Manager"]),
            game_state,
        )


# ---------------------------------------------------------------------------
# (7) Spend-food routing freeze: spend-decision rows read as gain_food


class TestMainNetSpendFoodRoutingFreeze:
    def test_shim_freezes_spend_food_decision_in_gain_food(self) -> None:
        eng, *_ = engine.Engine.create(seed=3)
        game_state = eng.state
        decision = decisions.SpendFoodDecision(
            player_id=0,
            prompt="x",
            choices=[decisions.FoodChoice(label="fish", food=cards.Food.FISH)],
        )
        fish = cards.food_index(cards.Food.FISH)
        live_stripped = _live_rows_without_v1_5_tail(decision, game_state)
        shim_rows = _era_shim().encode_choices(decision, game_state)
        assert shim_rows.shape == live_stripped.shape

        gain_idx = layout._OFF_GAIN_FOOD + fish
        pay_idx = layout._OFF_PAY + fish
        assert np.isclose(live_stripped[0][pay_idx], 1.0 / layout._PAYMENT_COUNT_SCALE)
        assert live_stripped[0][gain_idx] == 0.0
        assert shim_rows[0][gain_idx] == 1.0
        assert shim_rows[0][pay_idx] == 0.0

        outside = np.ones(live_stripped.shape[1], dtype=bool)
        outside[[gain_idx, pay_idx]] = False
        assert np.array_equal(live_stripped[0][outside], shim_rows[0][outside])

    def test_shim_freezes_spend_food_for_egg_decision_in_gain_food(self) -> None:
        eng, *_ = engine.Engine.create(seed=3)
        game_state = eng.state
        decision = decisions.SpendFoodForEggDecision(
            player_id=0,
            prompt="x",
            choices=[decisions.FoodChoice(label="seed", food=cards.Food.SEED)],
        )
        seed = cards.food_index(cards.Food.SEED)
        live_stripped = _live_rows_without_v1_5_tail(decision, game_state)
        shim_rows = _era_shim().encode_choices(decision, game_state)
        assert shim_rows.shape == live_stripped.shape

        gain_idx = layout._OFF_GAIN_FOOD + seed
        pay_idx = layout._OFF_PAY + seed
        assert np.isclose(live_stripped[0][pay_idx], 1.0 / layout._PAYMENT_COUNT_SCALE)
        assert live_stripped[0][gain_idx] == 0.0
        assert shim_rows[0][gain_idx] == 1.0
        assert shim_rows[0][pay_idx] == 0.0

        outside = np.ones(live_stripped.shape[1], dtype=bool)
        outside[[gain_idx, pay_idx]] = False
        assert np.array_equal(live_stripped[0][outside], shim_rows[0][outside])


# ---------------------------------------------------------------------------
# (8) Setup freezes: goal_affinity, folded kept_bonus_value, split affinity


class TestSetupGoalAffinityFreeze:
    def test_shim_freezes_egg_blind_pricing_live_net_does_not(self) -> None:
        candidate, context = _star_nest_candidate()
        encoding = setup_model.SetupEncoding()
        live_net, shim_net = _setup_nets(encoding)

        live_vec = live_net.encode_candidate(candidate, context)
        shim_vec = shim_net.encode_candidate(candidate, context)

        base = encoding.off_goal_affinity
        assert live_vec[base] != 0.0  # played-and-egg-populated: star wild
        assert shim_vec[base] == 0.0  # frozen pre-1.5 play-instant pricing

        # Every other position — including the other 3 goal_affinity scalars,
        # which are all-zero either way for the padding goal categories — is
        # byte-identical between the shim and the live net.
        stripe_width = encoding.off_turn1_playable - base
        outside_stripe = np.ones(live_vec.shape[0], dtype=bool)
        outside_stripe[base : base + stripe_width] = False
        assert np.array_equal(live_vec[outside_stripe], shim_vec[outside_stripe])
        assert np.array_equal(
            live_vec[base + 1 : base + stripe_width],
            shim_vec[base + 1 : base + stripe_width],
        )

    def test_shim_output_width_matches_live_encoding(self) -> None:
        candidate, context = _star_nest_candidate()
        encoding = setup_model.SetupEncoding()
        _, shim_net = _setup_nets(encoding)
        vec = shim_net.encode_candidate(candidate, context)
        assert vec.shape == (encoding.total_dim,)


class TestSetupBonusPricingFreeze:
    def test_folded_kept_bonus_value_freezes_all_four_scalars(self) -> None:
        """Breeding Manager kept with two 4+-egg-capacity birds: the live
        encoder prices qual/stepped/linear at the optimistic count and the
        tray potential at the tray's egg-capable bird; the shim regenerates
        the static zeros. Every other position is byte-identical."""
        breeding_manager = _BONUS_BY_NAME["Breeding Manager"]
        candidate = setup_model.SetupCandidate(
            kept_cards=(_BIG_NEST[0], _BIG_NEST[1]),
            kept_foods=(cards.Food.SEED, cards.Food.FISH, cards.Food.FRUIT),
            bonus_card=breeding_manager,
        )
        context = setup_model.SetupContext(
            tray_birds=(_BIG_NEST[2], None, None),
            birdfeeder_counts=(0, 0, 0, 0, 0, 0),
            round_goal_categories=("birds_forest",) * 4,
        )
        encoding = setup_model.SetupEncoding()
        live_net, shim_net = _setup_nets(encoding)
        live_vec = live_net.encode_candidate(candidate, context)
        shim_vec = shim_net.encode_candidate(candidate, context)

        base = encoding.off_bonus_value
        assert np.isclose(live_vec[base], 2.0 / layout._BONUS_COUNT_SCALE)
        assert np.isclose(
            live_vec[base + 1],
            scoring.bonus_score_for_count(breeding_manager, 2)
            / layout._BONUS_VALUE_SCALE,
        )
        assert live_vec[base + 2] > 0.0
        assert np.isclose(live_vec[base + 3], 1.0 / layout._BONUS_COUNT_SCALE)
        assert np.all(shim_vec[base : base + 4] == 0.0)

        outside = np.ones(live_vec.shape[0], dtype=bool)
        outside[base : base + 4] = False
        assert np.array_equal(live_vec[outside], shim_vec[outside])

    def test_split_bonus_card_affinity_freezes_min_max(self) -> None:
        """Split mode: with Breeding Manager and a static tagged card dealt,
        the live affinity pair prices the egg card by egg capacity while the
        shim regenerates the static counts."""
        breeding_manager = _BONUS_BY_NAME["Breeding Manager"]
        bird_feeder = _BONUS_BY_NAME["Bird Feeder"]
        tagged = next(
            bird for bird in _BIRDS if bird_feeder.name in bird.bonus_categories
        )
        high_a = _BIRDS[0].model_copy(update={"egg_limit": 4, "bonus_categories": ()})
        high_b = _BIRDS[1].model_copy(update={"egg_limit": 5, "bonus_categories": ()})
        tagged_low = tagged.model_copy(update={"egg_limit": 2})
        candidate = setup_model.SetupCandidate(
            kept_cards=(high_a, high_b, tagged_low),
            kept_foods=(cards.Food.SEED, cards.Food.FISH),
            bonus_card=None,
        )
        context = setup_model.SetupContext(
            tray_birds=(None, None, None),
            birdfeeder_counts=(0, 0, 0, 0, 0, 0),
            round_goal_categories=("birds_forest",) * 4,
            dealt_bonus_cards=(breeding_manager, bird_feeder),
        )
        encoding = setup_model.SetupEncoding(split_bonus=True)
        live_net, shim_net = _setup_nets(encoding)
        live_vec = live_net.encode_candidate(candidate, context)
        shim_vec = shim_net.encode_candidate(candidate, context)

        base = encoding.off_bonus_block + setup_arch_module._BONUS_DIM
        # Live: Bird Feeder counts its tagged keep (1), Breeding Manager both
        # 4+-egg keeps (2). Shim: the egg card falls back to its (empty) tag.
        assert np.isclose(live_vec[base + 0], 1.0 / layout._BONUS_COUNT_SCALE)
        assert np.isclose(live_vec[base + 1], 2.0 / layout._BONUS_COUNT_SCALE)
        assert shim_vec[base + 0] == 0.0
        assert np.isclose(shim_vec[base + 1], 1.0 / layout._BONUS_COUNT_SCALE)

        outside = np.ones(live_vec.shape[0], dtype=bool)
        outside[base : base + 2] = False
        assert np.array_equal(live_vec[outside], shim_vec[outside])


# ---------------------------------------------------------------------------
# (9) Forward passes at era dims and at live default dims


class TestForwardAtEraDims:
    def test_forward_pass_runs_at_era_dims(self) -> None:
        eng, *_ = engine.Engine.create(seed=101)
        _forward(_era_shim(), _decision(), eng.state)

    def test_forward_pass_runs_at_live_default_dims(self) -> None:
        """Constructing with default (live) dims still works — the shim
        derives its true encoder width from ``self.spec``, so the live-dim
        test style remains valid alongside the era-dim load path."""
        eng, *_ = engine.Engine.create(seed=101)
        _forward(
            compat_v1_4.PolicyValueNetV1_4(arch=_small_arch()), _decision(), eng.state
        )


# ---------------------------------------------------------------------------
# (10) Era-owned raw stripe layouts


class TestLayouts:
    def test_raw_state_stripe_layout_lacks_known_hand_opp(self) -> None:
        names = {s.name for s in _era_shim().raw_state_stripe_layout().stripes}
        assert "known_hand_opp" not in names

    def test_raw_state_stripe_layout_total_matches_era_state_dim(self) -> None:
        state_dim, _ = compat.encoding_dims_for_era("1.4", encode.DEFAULT_SPEC)
        assert _era_shim().raw_state_stripe_layout().total_size == state_dim

    def test_every_older_shim_state_layout_also_lacks_known_hand_opp(self) -> None:
        for net in (
            compat_v1_3.PolicyValueNetV1_3(arch=_small_arch()),
            compat_v1_0.PolicyValueNetV1_0(arch=_small_arch()),
        ):
            names = {s.name for s in net.raw_state_stripe_layout().stripes}
            assert "known_hand_opp" not in names, type(net).__name__

    def test_state_layout_without_stripes_on_unknown_name_raises(self) -> None:
        with pytest.raises(KeyError):
            _era_shim().raw_state_stripe_layout().without_stripes(("nope",))

    def test_raw_choice_stripe_layout_total_matches_encoding_dims_for_era(
        self,
    ) -> None:
        cases: list[tuple[str, core.PolicyValueNet]] = [
            ("1.4", _era_shim()),
            ("1.3", _v1_3_era_net()),
        ]
        for era, net in cases:
            _, choice_dim = compat.encoding_dims_for_era(era, encode.DEFAULT_SPEC)
            assert net.raw_choice_stripe_layout().total_size == choice_dim, era

    def test_goal_delta_ignoring_eggs_absent_from_every_pre_1_5_choice_layout(
        self,
    ) -> None:
        for net in (_era_shim(), _v1_3_era_net()):
            names = {s.name for s in net.raw_choice_stripe_layout().stripes}
            assert "goal_delta_ignoring_eggs" not in names


# ---------------------------------------------------------------------------
# (11) Real load-path round-trips (fixture-equivalent), both nets


def test_v1_4_stamped_checkpoint_round_trips(tmp_path: pathlib.Path) -> None:
    """A v1.4-stamped checkpoint loads under live code via ``load_policy_net``
    as the shim class (at era 1.4's dims — live minus the ``known_hand_opp``
    state stripe and the ``goal_delta_ignoring_eggs`` choice tail) and
    forward-passes a play-bird decision through the frozen encoder."""
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
    assert not isinstance(loaded, compat_v1_3.PolicyValueNetV1_3)
    assert saved_cfg.encoding_version == "1.4"

    game_state = _wetland_goal_state()
    _forward(loaded, _play_decision(_dual_bird()), game_state)


def test_v1_4_stamped_setup_checkpoint_round_trips(tmp_path: pathlib.Path) -> None:
    """A v1.4-stamped setup checkpoint loads under live code via
    ``load_setup_net`` as ``SetupNetV1_4`` and freezes both pre-1.5 setup
    value freezes (``goal_affinity`` and the bonus pricing) — the setup-side
    twin of ``test_v1_4_stamped_checkpoint_round_trips``."""
    base_cfg = config.RunConfig(
        misc=config.MiscConfig(device="cpu"),
        run=config.RunSettings(
            run_name="v14-setup-roundtrip", checkpoint_dir=str(tmp_path)
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
            setup=config.SetupNetArchitecture(head_layers=(8,)),
        ),
    )
    cfg = config.with_encoding_version(base_cfg, "1.4")
    assert cfg.encoding_version == "1.4"

    runmeta.write_run_config(
        str(tmp_path),
        cfg,
        stamp="t0",
        started_at="t0",
        git_sha=None,
        resumed_from_iteration=0,
    )
    descriptor = setup_runmeta.read_setup_config(str(tmp_path))
    assert descriptor.version == "1.4"

    net_cls = setup_net_module.SetupNet.class_for_version(cfg.encoding_version)
    assert net_cls is compat_v1_4.SetupNetV1_4
    net = net_cls.from_setup_config(descriptor)

    setup_payload: dict[str, object] = {
        "setup_model": net.state_dict(),
        "version": "1.4",
    }
    loop_checkpoint.atomic_save(setup_payload, tmp_path / artifacts.SETUP_CKPT)

    loaded = loaders.load_setup_net(tmp_path, torch.device("cpu"))
    assert isinstance(loaded, compat_v1_4.SetupNetV1_4)

    # goal_affinity freeze.
    candidate, context = _star_nest_candidate()
    vec = loaded.encode_candidate(candidate, context)
    assert vec[loaded.encoding.off_goal_affinity] == 0.0

    # Bonus pricing freeze, whichever bonus-block shape the production config
    # carries (split_bonus defers the kept bonus, so the candidate keeps none
    # and the dealt Breeding Manager is priced by the affinity pair).
    breeding_manager = _BONUS_BY_NAME["Breeding Manager"]
    bonus_candidate = setup_model.SetupCandidate(
        kept_cards=(_BIG_NEST[0], _BIG_NEST[1]),
        kept_foods=(cards.Food.SEED, cards.Food.FISH, cards.Food.FRUIT),
        bonus_card=None if loaded.encoding.split_bonus else breeding_manager,
    )
    bonus_context = setup_model.SetupContext(
        tray_birds=(None, None, None),
        birdfeeder_counts=(0, 0, 0, 0, 0, 0),
        round_goal_categories=("birds_forest",) * 4,
        dealt_bonus_cards=(breeding_manager,),
    )
    bonus_vec = loaded.encode_candidate(bonus_candidate, bonus_context)
    if loaded.encoding.split_bonus:
        base = loaded.encoding.off_bonus_block + setup_arch_module._BONUS_DIM
        assert bonus_vec[base] == 0.0  # live pricing would read 2/5 (both 4+ eggs)
        assert bonus_vec[base + 1] == 0.0
    else:
        assert bonus_vec[loaded.encoding.off_bonus_value] == 0.0


# ---------------------------------------------------------------------------
# (12) Architecture keys: lead with the era, so era 1.4 differs from live


class TestArchitectureKeyEra:
    def test_keys_differ_between_1_4_and_live(self) -> None:
        base_cfg = config.RunConfig()
        era_cfg = config.with_encoding_version(base_cfg, "1.4")
        live_cfg = config.with_encoding_version(base_cfg, version.MODEL_VERSION)

        assert era_cfg.setup_architecture_key[0] == "1.4"
        assert live_cfg.setup_architecture_key[0] == version.MODEL_VERSION
        assert era_cfg.setup_architecture_key != live_cfg.setup_architecture_key
        assert era_cfg.architecture_key != live_cfg.architecture_key


# ---------------------------------------------------------------------------
# (13) MAIN_ACTION forecast zeroing (module docstring change 5)


class TestMainActionForecastZeroing:
    def test_shim_zeroes_forecast_cells_except_regenerated_scalars(self) -> None:
        """era-1.4 zeroes ``exchange`` on every row, ``bonus_delta`` on every
        row except DRAW_CARDS (equal to live — its hand-growth pricing
        predates the amend), ``goal_delta`` on every row except LAY_EGGS
        (equal to live — its capacity-capped bound likewise predates the
        amend). The row also stays at the narrow era width."""
        eng, *_ = engine.Engine.create(seed=105)
        game_state = eng.state
        decision = decisions.MainActionDecision(
            player_id=0,
            prompt="action",
            choices=[
                decisions.MainActionChoice(label=action.value, action=action)
                for action in decisions.MainAction
            ],
        )
        live = encode.encode_choices(decision, game_state)
        shim = _era_shim()
        shim_out = shim.encode_choices(decision, game_state)

        _, choice_dim = compat.encoding_dims_for_era("1.4", encode.DEFAULT_SPEC)
        assert shim_out.shape[1] == choice_dim

        exchange = slice(
            layout._OFF_EXCHANGE, layout._OFF_EXCHANGE + layout._EXCHANGE_DIM
        )
        bonus = slice(
            layout._OFF_BONUS_DELTA, layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_DIM
        )
        goal = slice(
            layout._OFF_GOAL_DELTA, layout._OFF_GOAL_DELTA + layout._GOAL_DELTA_DIM
        )

        assert np.all(shim_out[:, exchange] == 0.0)
        for row_idx, choice in enumerate(decision.choices):
            if choice.action == decisions.MainAction.DRAW_CARDS:
                assert np.array_equal(shim_out[row_idx, bonus], live[row_idx, bonus])
            else:
                assert np.all(shim_out[row_idx, bonus] == 0.0)
            if choice.action == decisions.MainAction.LAY_EGGS:
                assert np.array_equal(shim_out[row_idx, goal], live[row_idx, goal])
            else:
                assert np.all(shim_out[row_idx, goal] == 0.0)


# ---------------------------------------------------------------------------
# (14) Card-table freeze: power_ex frozen at the pre-v1.5-amend values
# (module docstring change 6)


class TestCardTablePowerExchangeFreeze:
    def test_v1_4_card_table_freezes_pre_eot_power_exchange(self) -> None:
        """``PolicyValueNetV1_4``'s frozen ``card_features`` buffer carries
        the pre-v1.5-amend (EOT-discard-omitted) power_ex block for every
        bird, differing from the live net's block exactly on the
        cards_to_discard column of the DRAW_CARDS_THEN_DISCARD_EOT birds."""
        arch = _small_arch()
        live_net = core.PolicyValueNet(arch=arch)
        shim = _era_shim(arch=arch)

        power_ex = slice(
            layout._OFF_ATTR_POWER_EX, layout._OFF_ATTR_POWER_EX + layout._EXCHANGE_DIM
        )
        live_block = live_net.card_features.numpy()[:, power_ex]
        shim_block = shim.card_features.numpy()[:, power_ex]

        eot_rows = _eot_bird_rows()
        assert eot_rows

        discard_col = layout._EXCHANGE_CARDS_TO_DISCARD
        outside_discard = np.ones(layout._EXCHANGE_DIM, dtype=bool)
        outside_discard[discard_col] = False
        for row in eot_rows:
            assert live_block[row, discard_col] > shim_block[row, discard_col]
            assert np.array_equal(
                live_block[row, outside_discard], shim_block[row, outside_discard]
            )

        non_eot_rows = [
            row for row in range(live_block.shape[0]) if row not in eot_rows
        ]
        assert np.array_equal(live_block[non_eot_rows], shim_block[non_eot_rows])

    def test_setup_net_card_table_freezes_pre_eot_power_exchange(self) -> None:
        """``SetupNetV1_4`` joins the same card-table freeze — it builds its
        own copy of the shared ``card_feature_matrix()`` table."""
        encoding = setup_model.SetupEncoding()
        live_setup, shim_setup = _setup_nets(encoding)

        power_ex = slice(
            layout._OFF_ATTR_POWER_EX, layout._OFF_ATTR_POWER_EX + layout._EXCHANGE_DIM
        )
        live_block = live_setup.card_features.numpy()[:, power_ex]
        shim_block = shim_setup.card_features.numpy()[:, power_ex]

        eot_rows = _eot_bird_rows()
        discard_col = layout._EXCHANGE_CARDS_TO_DISCARD
        for row in eot_rows:
            assert live_block[row, discard_col] > shim_block[row, discard_col]

        non_eot_rows = [
            row for row in range(live_block.shape[0]) if row not in eot_rows
        ]
        assert np.array_equal(live_block[non_eot_rows], shim_block[non_eot_rows])
