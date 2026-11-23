# pyright: reportPrivateUsage=false
# (the encoding tests read the package-private layout offsets to pin stripes)
"""Behavior and encoding tests for the Cedar Waxwing power.

Cedar Waxwing: "Tuck 1 [card] from your hand behind this bird. If you do, gain
1 [fruit] from the supply." It parses to a single
``EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY`` effect: an optional gate
(``ActivateTuckDecision``), then a mandatory card pick
(``BirdPowerTuckFromHandDecision``), then the food gain. The first tests pin the
engine behavior and decision sequence; the last section pins how the gate's
yes row (a ``TuckActivateChoice`` ledger) encodes, live and through the era-1.5
shim.
"""

from __future__ import annotations

import random
import typing

import numpy as np

from wingspan import (
    architecture,
    cards,
    compat,
    decisions,
    encode,
    engine,
    model,
    state,
)
from wingspan.encode import layout
from wingspan.engine import powers, scoring

# ---------------------------------------------------------------------------
# Shared fixtures

_HABITAT = cards.Habitat.FOREST
_CEDAR_WAXWING = "Cedar Waxwing"
_ONLY_ONE_CHOICE_LOG = "only 1 choice"
_NO_CHOICES_LOG = "skipping decision, no choices"

type DecisionType = type[decisions.Decision[typing.Any]]


def _find(birds: list[cards.Bird], name: str) -> cards.Bird:
    return next(bird for bird in birds if bird.name == name)


def _no_agent[C: decisions.Choice](
    _eng: engine.Engine, decision: decisions.Decision[C]
) -> C:
    raise AssertionError(f"unexpected agent consultation: {type(decision).__name__}")


def _staged(
    hand_size: int, seed: int = 0
) -> tuple[engine.Engine, state.Player, state.PlayedBird, list[cards.Bird]]:
    """Engine with a Cedar Waxwing in the forest and ``hand_size`` distinct
    non-waxwing cards in player 0's hand. Returns the hand cards as well."""
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(seed), birds, bonuses, goals)
    eng = engine.Engine(gs, agents=[_no_agent, _no_agent])
    waxwing = _find(birds, _CEDAR_WAXWING)
    player = eng.state.players[0]
    pb = state.PlayedBird(bird=waxwing)
    player.board[_HABITAT] = [pb]
    hand = [bird for bird in birds if bird.name != _CEDAR_WAXWING][:hand_size]
    player.hand = list(hand)
    player.known_hand = []
    return eng, player, pb, hand


class _Recorder:
    """Scripted agent that records the decisions it is consulted on.

    ``accept`` answers the gate with the tuck row (else the skip row);
    ``pick_last`` makes the card selection take the final row (else the first).
    """

    def __init__(self, accept: bool, pick_last: bool = False) -> None:
        self.accept = accept
        self.pick_last = pick_last
        self.decision_types: list[DecisionType] = []
        self.choice_rows: list[list[decisions.Choice]] = []

    def __call__[C: decisions.Choice](
        self, _eng: engine.Engine, decision: decisions.Decision[C], /
    ) -> C:
        self.decision_types.append(type(decision))
        self.choice_rows.append(list(decision.choices))
        if isinstance(decision, decisions.ActivateTuckDecision):
            if self.accept:
                return typing.cast(
                    C,
                    next(
                        ch
                        for ch in decision.choices
                        if isinstance(ch, decisions.TuckActivateChoice)
                    ),
                )
            return typing.cast(
                C,
                next(
                    ch
                    for ch in decision.choices
                    if isinstance(ch, decisions.SkipChoice)
                ),
            )
        if isinstance(decision, decisions.BirdPowerTuckFromHandDecision):
            index = -1 if self.pick_last else 0
            return typing.cast(C, decision.choices[index])
        raise AssertionError(f"unexpected decision: {type(decision).__name__}")


def _activate(
    eng: engine.Engine,
    agent: engine.Agent,
    player: state.Player,
    pb: state.PlayedBird,
) -> None:
    powers.dispatch_power(eng, agent, player, pb, _HABITAT, "activate")


def _food_snapshot(player: state.Player) -> dict[cards.Food, int]:
    return {food: player.food[food] for food in cards.ALL_FOODS}


def _log_text(eng: engine.Engine) -> str:
    return "\n".join(eng.state.log)


# ---------------------------------------------------------------------------
# 1. Card data


def test_cedar_waxwing_card_data():
    birds, _, _ = cards.load_all()
    bird = _find(birds, _CEDAR_WAXWING)

    assert bird.points == 3
    assert bird.nest == cards.NestType.BOWL
    assert bird.egg_limit == 3
    assert bird.wingspan_cm == 25
    assert bird.habitats == (cards.Habitat.FOREST, cards.Habitat.GRASSLAND)
    assert bird.color == cards.PowerColor.BROWN
    assert bird.flocking is True

    assert bird.food_cost.is_or_cost is False
    assert bird.food_cost.specific_of(cards.Food.FRUIT) == 2
    assert bird.food_cost.total == 2
    assert bird.food_cost.wild == 0

    assert len(bird.power.effects) == 1
    effect = bird.power.effects[0]
    assert effect.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY
    assert effect.food == cards.Food.FRUIT
    assert effect.amount == 1


# ---------------------------------------------------------------------------
# 2. Empty hand


def test_empty_hand_never_consults_agent():
    eng, player, pb, _ = _staged(0)
    food_before = _food_snapshot(player)

    _activate(eng, _no_agent, player, pb)

    assert pb.tucked_cards == 0
    assert player.hand == []
    assert _food_snapshot(player) == food_before
    assert _NO_CHOICES_LOG in _log_text(eng)


# ---------------------------------------------------------------------------
# 3 / 4. Exactly one card in hand


def test_single_card_accept_forces_tuck_selection():
    eng, player, pb, hand = _staged(1)
    food_before = _food_snapshot(player)
    agent = _Recorder(accept=True)

    _activate(eng, agent, player, pb)

    # Only the gate reaches the agent; the lone-card pick is auto-forced.
    assert agent.decision_types == [decisions.ActivateTuckDecision]
    gate_rows = agent.choice_rows[0]
    assert len(gate_rows) == 2
    assert isinstance(gate_rows[0], decisions.TuckActivateChoice)
    assert isinstance(gate_rows[1], decisions.SkipChoice)
    log_text = _log_text(eng)
    assert _ONLY_ONE_CHOICE_LOG in log_text
    assert f"tucked {hand[0].name}" in log_text

    assert player.hand == []
    assert pb.tucked_cards == 1
    expected_food = dict(food_before)
    expected_food[cards.Food.FRUIT] += 1
    assert _food_snapshot(player) == expected_food


def test_single_card_skip_changes_nothing():
    eng, player, pb, hand = _staged(1)
    food_before = _food_snapshot(player)
    agent = _Recorder(accept=False)

    _activate(eng, agent, player, pb)

    assert agent.decision_types == [decisions.ActivateTuckDecision]
    assert player.hand == hand
    assert pb.tucked_cards == 0
    assert _food_snapshot(player) == food_before


# ---------------------------------------------------------------------------
# 5 / 6. Three cards in hand


def test_three_cards_accept_agent_picks_last():
    eng, player, pb, hand = _staged(3)
    food_before = _food_snapshot(player)
    agent = _Recorder(accept=True, pick_last=True)

    _activate(eng, agent, player, pb)

    assert agent.decision_types == [
        decisions.ActivateTuckDecision,
        decisions.BirdPowerTuckFromHandDecision,
    ]
    pick_rows = agent.choice_rows[1]
    assert len(pick_rows) == 3
    assert all(isinstance(row, decisions.BirdChoice) for row in pick_rows)
    assert not any(isinstance(row, decisions.SkipChoice) for row in pick_rows)
    offered = [row.bird for row in pick_rows if isinstance(row, decisions.BirdChoice)]
    assert offered == hand

    assert player.hand == [hand[0], hand[1]]
    assert pb.tucked_cards == 1
    assert f"tucked {hand[2].name}" in _log_text(eng)
    expected_food = dict(food_before)
    expected_food[cards.Food.FRUIT] += 1
    assert _food_snapshot(player) == expected_food


def test_three_cards_skip_only_gate_consulted():
    eng, player, pb, hand = _staged(3)
    food_before = _food_snapshot(player)
    agent = _Recorder(accept=False)

    _activate(eng, agent, player, pb)

    assert agent.decision_types == [decisions.ActivateTuckDecision]
    assert player.hand == hand
    assert pb.tucked_cards == 0
    assert _food_snapshot(player) == food_before


# ---------------------------------------------------------------------------
# 7. Decision-family routing


def test_decision_family_routing():
    assert (
        decisions.family_for(decisions.ActivateTuckDecision)
        == decisions.DecisionFamily.SKIP_OPTIONAL
    )
    assert (
        decisions.family_for(decisions.BirdPowerTuckFromHandDecision)
        == decisions.DecisionFamily.DISCARD_BIRD
    )


# ---------------------------------------------------------------------------
# 8. Fruit specifically, accumulation over repeated activations


def test_two_activations_accumulate_tucks_and_fruit_only():
    eng, player, pb, hand = _staged(3)
    food_before = _food_snapshot(player)
    agent = _Recorder(accept=True)

    _activate(eng, agent, player, pb)
    _activate(eng, agent, player, pb)

    # First pick takes hand[0], second takes hand[1] (agent picks the first row).
    assert player.hand == [hand[2]]
    assert pb.tucked_cards == 2
    assert agent.decision_types == [
        decisions.ActivateTuckDecision,
        decisions.BirdPowerTuckFromHandDecision,
        decisions.ActivateTuckDecision,
        decisions.BirdPowerTuckFromHandDecision,
    ]
    # The second pick offered only the two remaining cards.
    second_pick_rows = agent.choice_rows[3]
    assert [
        row.bird for row in second_pick_rows if isinstance(row, decisions.BirdChoice)
    ] == [hand[1], hand[2]]

    expected_food = dict(food_before)
    expected_food[cards.Food.FRUIT] += 2
    assert _food_snapshot(player) == expected_food
    # Tucked cards never come back to the hand.
    assert hand[0] not in player.hand
    assert hand[1] not in player.hand


# ---------------------------------------------------------------------------
# 9. Scoring


def test_tucked_cards_score_one_point_each():
    _, player, pb, _ = _staged(0)
    player.bonus_cards = []
    score_without_tucks = scoring.running_score(player)

    pb.tucked_cards = 2

    assert player.total_tucked == 2
    assert scoring.running_score(player) == score_without_tucks + 2


# ---------------------------------------------------------------------------
# 10+. Encoding of the gate decision
#
# The real handler builds the ``ActivateTuckDecision``; a probe agent encodes it
# at consultation time (live, and optionally through an era net), so the rows
# are exactly what a network would see.

_JUNCO = "Dark-Eyed Junco"
_SWIFT = "Vaux's Swift"
_COOT = "American Coot"
_SIBLINGS = (_CEDAR_WAXWING, _JUNCO, _SWIFT, _COOT)
_VISIONARY_LEADER = "Visionary Leader"
_LIVE_ERA = "1.6"
_ERA_1_5 = "1.5"
_PROBE_HAND_SIZE = 3
_YES, _SKIP = 0, 1

_EXCHANGE = slice(layout._OFF_EXCHANGE, layout._OFF_EXCHANGE + layout._EXCHANGE_DIM)
_GAIN_FOOD = slice(layout._OFF_GAIN_FOOD, layout._OFF_GAIN_FOOD + cards.N_FOODS)
_BONUS_DELTA = slice(
    layout._OFF_BONUS_DELTA, layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_DIM
)
_BECOMES_PLAYABLE = range(
    layout.CHOICE_BECOMES_PLAYABLE_OFFSET,
    layout.CHOICE_BECOMES_PLAYABLE_OFFSET + layout.CHOICE_BECOMES_PLAYABLE_DIM,
)
_LEDGER_FIELDS_THAT_STAY_EMPTY = (
    "gained_egg_count",
    "gained_card_count",
    "gained_cache_count",
    "gained_play_count",
    "opp_gained_food_count",
    "opp_gained_egg_count",
    "opp_gained_card_count",
    "opp_gained_tuck_count",
    "paid_food_count",
    "paid_egg_count",
)


class _GateProbe:
    """Agent that answers the tuck gate and encodes the decision it was shown.

    The decision is encoded while the engine is still at the gate (the accept
    path mutates the state afterwards). ``era_net`` additionally encodes the
    same decision through that net's ``encode_choices``.
    """

    def __init__(
        self, accept: bool, era_net: model.PolicyValueNet | None = None
    ) -> None:
        self.accept = accept
        self.era_net = era_net
        self.gate: decisions.ActivateTuckDecision | None = None
        self.live_rows: np.ndarray | None = None
        self.era_rows: np.ndarray | None = None

    def __call__[C: decisions.Choice](
        self, eng: engine.Engine, decision: decisions.Decision[C], /
    ) -> C:
        if isinstance(decision, decisions.ActivateTuckDecision):
            self.gate = decision
            self.live_rows = encode.encode_choices(decision, eng.state)
            if self.era_net is not None:
                self.era_rows = self.era_net.encode_choices(decision, eng.state)
            # Row order (yes, then skip) is pinned by the ledger test.
            return typing.cast(C, decision.choices[_YES if self.accept else _SKIP])
        if isinstance(decision, decisions.BirdPowerTuckFromHandDecision):
            return typing.cast(C, decision.choices[0])
        raise AssertionError(f"unexpected decision: {type(decision).__name__}")

    @property
    def gate_decision(self) -> decisions.ActivateTuckDecision:
        assert self.gate is not None
        return self.gate

    @property
    def live(self) -> np.ndarray:
        assert self.live_rows is not None
        return self.live_rows

    @property
    def era(self) -> np.ndarray:
        assert self.era_rows is not None
        return self.era_rows


def _probe_gate(
    bird_name: str,
    *,
    accept: bool = False,
    bonus_name: str | None = None,
    era_net: model.PolicyValueNet | None = None,
) -> tuple[_GateProbe, engine.Engine, state.Player, state.PlayedBird]:
    """Activate ``bird_name`` in the forest with a fixed hand and run the gate."""
    birds, bonuses, goals = cards.load_all()
    game_state = state.new_game(random.Random(0), birds, bonuses, goals)
    probe = _GateProbe(accept, era_net)
    eng = engine.Engine(game_state, agents=[probe, probe])
    player = eng.state.players[0]
    pb = state.PlayedBird(bird=_find(birds, bird_name))
    player.board[_HABITAT] = [pb]
    player.hand = [bird for bird in birds if bird.name not in _SIBLINGS][
        :_PROBE_HAND_SIZE
    ]
    player.known_hand = []
    if bonus_name is not None:
        player.bonus_cards = [_find_bonus(bonuses, bonus_name)]
    powers.dispatch_power(eng, probe, player, pb, _HABITAT, "activate")
    return probe, eng, player, pb


def _find_bonus(bonuses: list[cards.BonusCard], name: str) -> cards.BonusCard:
    return next(bonus for bonus in bonuses if bonus.name == name)


def _era_net(era: str) -> model.PolicyValueNet:
    """A small net of the class routed for ``era``, at that era's dims."""
    spec = encode.DEFAULT_SPEC
    state_dim, choice_dim = compat.encoding_dims_for_era(era, spec)
    arch = architecture.ModelArchitecture(
        trunk_layers=(8, 8),
        choice_layers=(8, 8),
        head_layers=(),
        value_layers=(),
        card_embed_dim=4,
    )
    net_cls = model.PolicyValueNet.class_for_version(era)
    return net_cls(state_dim=state_dim, choice_dim=choice_dim, arch=arch, spec=spec)


def _exchange_slot(row: np.ndarray, slot: int) -> float:
    return float(row[layout._OFF_EXCHANGE + slot])


def _food_slot(food: cards.Food) -> int:
    return layout._OFF_GAIN_FOOD + cards.food_index(food)


def test_gate_ledger_rows_for_cedar_waxwing():
    probe, _, _, _ = _probe_gate(_CEDAR_WAXWING)
    rows = probe.gate_decision.choices
    assert len(rows) == 2

    yes = rows[_YES]
    assert isinstance(yes, decisions.TuckActivateChoice)
    assert yes.paid_card_count == 1
    assert yes.gained_tuck_count == 1
    assert yes.gained_food_count == 1
    assert yes.gained_food == cards.Food.FRUIT
    assert yes.paid_food is None
    for field in _LEDGER_FIELDS_THAT_STAY_EMPTY:
        assert getattr(yes, field) == 0, field

    skip = rows[_SKIP]
    assert type(skip) is decisions.SkipChoice


def test_yes_row_exchange_stripe_carries_card_tuck_and_food():
    probe, _, _, _ = _probe_gate(_CEDAR_WAXWING)
    yes, skip = probe.live[_YES], probe.live[_SKIP]

    scale = layout._EXCHANGE_SCALE
    expected = np.zeros(layout._EXCHANGE_DIM, dtype=np.float32)
    expected[layout._EXCHANGE_CARDS_TO_DISCARD] = 1 / scale
    expected[layout._EXCHANGE_CARDS_TO_TUCK] = 1 / scale
    expected[layout._EXCHANGE_FOOD_TO_GAIN] = 1 / scale
    np.testing.assert_allclose(yes[_EXCHANGE], expected)
    assert np.count_nonzero(yes[_EXCHANGE]) == 3
    assert yes[layout._OFF_KIND + layout._KIND_SPECIAL] == 1.0

    assert skip[layout._OFF_SPECIAL + layout._SPECIAL_IS_SKIP] == 1.0
    assert not skip[_EXCHANGE].any()


def test_gain_food_stripe_is_fruit_only_on_the_yes_row():
    probe, _, _, _ = _probe_gate(_CEDAR_WAXWING)

    expected = np.zeros(cards.N_FOODS, dtype=np.float32)
    expected[cards.food_index(cards.Food.FRUIT)] = 1.0
    np.testing.assert_array_equal(probe.live[_YES][_GAIN_FOOD], expected)
    assert not probe.live[_SKIP][_GAIN_FOOD].any()


def test_yes_row_bonus_delta_reflects_the_hand_shrink():
    probe, _, _, _ = _probe_gate(_CEDAR_WAXWING, bonus_name=_VISIONARY_LEADER)
    yes, skip = probe.live[_YES], probe.live[_SKIP]

    assert yes[layout._OFF_BONUS_DELTA + layout._BONUS_DELTA_QUAL] > 0
    assert not skip[_BONUS_DELTA].any()

    # With no hand-counting bonus card the same row has no bonus delta.
    plain, _, _, _ = _probe_gate(_CEDAR_WAXWING)
    assert not plain.live[_YES][_BONUS_DELTA].any()


def test_yes_row_matches_a_pay_cost_row_with_the_same_ledger():
    # The probe skips, so the engine state is still the gate-time state.
    probe, eng, _, _ = _probe_gate(_CEDAR_WAXWING, bonus_name=_VISIONARY_LEADER)
    yes_choice = probe.gate_decision.choices[_YES]
    assert isinstance(yes_choice, decisions.TuckActivateChoice)
    ledger = {
        name: getattr(yes_choice, name)
        for name in decisions.ExchangeLedger.model_fields
    }
    pay_decision = decisions.AcceptExchangeDecision(
        player_id=0,
        prompt="accept?",
        choices=[
            decisions.PayCostChoice(label="pay", **ledger),
            decisions.SkipChoice(label="skip"),
        ],
    )
    pay_row = encode.encode_choices(pay_decision, eng.state)[_YES]
    tuck_row = probe.live[_YES]

    for stripe in (_EXCHANGE, _GAIN_FOOD, _BONUS_DELTA):
        np.testing.assert_array_equal(tuck_row[stripe], pay_row[stripe])
    goal_delta = slice(
        layout._OFF_GOAL_DELTA, layout._OFF_GOAL_DELTA + layout._GOAL_DELTA_DIM
    )
    np.testing.assert_array_equal(tuck_row[goal_delta], pay_row[goal_delta])


def test_cedar_waxwing_yes_row_is_distinct_from_its_siblings():
    rows = {name: _probe_gate(name)[0].live[_YES] for name in _SIBLINGS}

    for first in _SIBLINGS:
        for second in _SIBLINGS:
            if first < second:
                assert not np.array_equal(rows[first], rows[second]), (first, second)

    cedar, junco = rows[_CEDAR_WAXWING], rows[_JUNCO]
    # Same exchange counts: only the food type differs.
    np.testing.assert_array_equal(cedar[_EXCHANGE], junco[_EXCHANGE])
    differing = {
        int(index)
        for index in np.flatnonzero(cedar != junco)
        if index not in _BECOMES_PLAYABLE
    }
    assert differing == {
        _food_slot(cards.Food.FRUIT),
        _food_slot(cards.Food.SEED),
    }

    # The draw sibling differs in the exchange stripe (card draw, no food).
    coot = rows[_COOT]
    assert _exchange_slot(coot, layout._EXCHANGE_CARDS_TO_DRAW) > 0
    assert _exchange_slot(cedar, layout._EXCHANGE_CARDS_TO_DRAW) == 0
    assert not coot[_GAIN_FOOD].any()


def test_era_1_5_net_keeps_the_old_bare_tuck_row():
    probe, _, _, _ = _probe_gate(
        _CEDAR_WAXWING,
        bonus_name=_VISIONARY_LEADER,
        era_net=_era_net(_ERA_1_5),
    )
    live, era = probe.live, probe.era
    assert model.PolicyValueNet.class_for_version(_LIVE_ERA) is model.PolicyValueNet

    # Era 1.5: only the special-kind bit and the tuck slot.
    old = np.zeros(era.shape[1], dtype=np.float32)
    old[layout._OFF_KIND + layout._KIND_SPECIAL] = 1.0
    old[layout._OFF_EXCHANGE + layout._EXCHANGE_CARDS_TO_TUCK] = (
        1 / layout._EXCHANGE_SCALE
    )
    np.testing.assert_array_equal(era[_YES], old)
    assert _exchange_slot(era[_YES], layout._EXCHANGE_CARDS_TO_DISCARD) == 0
    assert not era[_YES][_GAIN_FOOD].any()
    assert not era[_YES][_BONUS_DELTA].any()

    # Live (1.6): the full row.
    assert _exchange_slot(live[_YES], layout._EXCHANGE_CARDS_TO_DISCARD) > 0
    assert live[_YES][_food_slot(cards.Food.FRUIT)] == 1.0
    assert live[_YES][_BONUS_DELTA].any()

    # The skip row is identical in both eras.
    np.testing.assert_array_equal(era[_SKIP], live[_SKIP])


def test_accepting_the_richer_row_leaves_engine_behavior_unchanged():
    birds, bonuses, goals = cards.load_all()
    fresh = state.new_game(random.Random(0), birds, bonuses, goals)
    starting_food = _food_snapshot(fresh.players[0])

    probe, _, player, pb = _probe_gate(_CEDAR_WAXWING, accept=True)
    assert probe.gate is not None

    assert pb.tucked_cards == 1
    assert len(player.hand) == _PROBE_HAND_SIZE - 1
    expected_food = dict(starting_food)
    expected_food[cards.Food.FRUIT] += 1
    assert _food_snapshot(player) == expected_food
