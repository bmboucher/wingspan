"""Unit tests for the five misc-unique bird powers.

Each test builds a minimal engine, patches a played bird with the target
power text, and invokes ``powers.dispatch_power`` directly so we can
assert engine behaviour for the new EffectKinds without driving a full
turn sequence.
"""

from __future__ import annotations

import os
import random
import sys
import typing

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers


def _no_agent[C: decisions.Choice](
    _engine: engine.Engine,
    _decision: decisions.Decision[C],
) -> C:
    """An ``Agent``-typed stub for powers that resolve without a decision; it
    raises if a power unexpectedly consults it."""
    raise AssertionError(
        f"agent should not be consulted (got {type(_decision).__name__})"
    )


def _make_engine_with_bird(
    power_text: str,
    color: cards.PowerColor = cards.PowerColor.WHITE,
    agents: list[engine.Agent] | None = None,
) -> tuple[engine.Engine, state.PlayedBird]:
    """Build a near-empty engine and stage a played bird carrying ``power_text``."""
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(0)
    gs = state.new_game(rng, birds, bonuses, goals)
    eng = engine.Engine(gs, agents=agents) if agents is not None else engine.Engine(gs)
    template = next(candidate for candidate in birds if candidate.color == color)
    bird = template.model_copy(
        update={
            "color": color,
            "raw_power_text": power_text,
            "power": cards.parse_power(color, power_text),
        }
    )
    pb = state.PlayedBird(bird=bird)
    return eng, pb


# ---------------------------------------------------------------------------
# Parser tests


def test_parser_draw_from_tray_all():
    power = cards.parse_power(
        cards.PowerColor.WHITE, "Draw the 3 face-up [card] in the bird tray."
    )
    assert [effect.kind for effect in power.effects] == [
        cards.EffectKind.DRAW_FROM_TRAY_ALL
    ]


def test_parser_trade_wild_food():
    power = cards.parse_power(
        cards.PowerColor.BROWN, "Trade 1 [wild] for any other type from the supply."
    )
    assert [effect.kind for effect in power.effects] == [
        cards.EffectKind.TRADE_WILD_FOOD
    ]


def test_parser_fewest_forest_gains_die():
    power = cards.parse_power(
        cards.PowerColor.BROWN,
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
    )
    assert [effect.kind for effect in power.effects] == [
        cards.EffectKind.FEWEST_FOREST_GAINS_DIE
    ]


def test_parser_play_additional_bird_here():
    power = cards.parse_power(
        cards.PowerColor.WHITE,
        "Play an additional bird in this bird's habitat. Pay its normal cost.",
    )
    assert [effect.kind for effect in power.effects] == [
        cards.EffectKind.PLAY_ADDITIONAL_BIRD_HERE
    ]


def test_parser_draw_n_plus_one_draft():
    power = cards.parse_power(
        cards.PowerColor.WHITE,
        "Draw [card] equal to the number of players +1. Starting with you and "
        "proceeding clockwise, each player selects 1 of those cards and places "
        "it in their hand. You keep the extra card.",
    )
    assert [effect.kind for effect in power.effects] == [
        cards.EffectKind.DRAW_N_PLUS_ONE_DRAFT
    ]


# ---------------------------------------------------------------------------
# Brant — DRAW_FROM_TRAY_ALL


def test_draw_from_tray_all_takes_all_three_no_mid_turn_refill():
    eng, pb = _make_engine_with_bird("Draw the 3 face-up [card] in the bird tray.")
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    player.hand = []
    # Force a deterministic tray of 3 known birds.
    original_tray_names = [bird.name for bird in gs.tray if bird is not None]
    deck_before = len(gs.bird_deck)
    powers.dispatch_power(eng, _no_agent, player, pb, cards.Habitat.WETLAND, "play")
    assert [bird.name for bird in player.hand] == original_tray_names
    # No mid-turn refill — the tray stays empty until end-of-turn.
    assert all(b is None for b in gs.tray)
    assert len(gs.bird_deck) == deck_before


# ---------------------------------------------------------------------------
# Green Heron — TRADE_WILD_FOOD


def test_trade_wild_food_swaps_one_food():
    eng, pb = _make_engine_with_bird(
        "Trade 1 [wild] for any other type from the supply.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    for food in cards.ALL_FOODS:
        player.food[food] = 0
    player.food[cards.Food.SEED] = 2

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        # New flow: activate (AcceptExchangeDecision) → discard SEED
        # (SpendFoodDecision) → gain FRUIT (GainFoodDecision) — a net swap.
        if isinstance(decision, decisions.AcceptExchangeDecision):
            return typing.cast(
                C,
                next(
                    choice
                    for choice in decision.choices
                    if isinstance(choice, decisions.PayCostChoice)
                ),
            )
        elif isinstance(decision, decisions.SpendFoodDecision):
            want = cards.Food.SEED
        elif isinstance(decision, decisions.GainFoodDecision):
            want = cards.Food.FRUIT
        else:
            raise AssertionError(f"unexpected decision: {type(decision).__name__}")
        return typing.cast(
            C,
            next(
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.FoodChoice) and choice.food == want
            ),
        )

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert player.food[cards.Food.SEED] == 1
    assert player.food[cards.Food.FRUIT] == 1


def test_trade_wild_food_skip_does_nothing():
    eng, pb = _make_engine_with_bird(
        "Trade 1 [wild] for any other type from the supply.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    for food in cards.ALL_FOODS:
        player.food[food] = 0
    player.food[cards.Food.SEED] = 1
    food_before = player.food.as_dict()

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        return typing.cast(
            C,
            next(
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.SkipChoice)
            ),
        )

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert player.food.as_dict() == food_before


def test_trade_wild_food_no_food_no_op():
    eng, pb = _make_engine_with_bird(
        "Trade 1 [wild] for any other type from the supply.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    for food in cards.ALL_FOODS:
        player.food[food] = 0

    def agent[C: decisions.Choice](  # pragma: no cover
        _engine: engine.Engine,
        _decision: decisions.Decision[C],
    ) -> C:
        pytest.fail("should not be consulted when player has no food")

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")


def test_trade_wild_food_is_an_activate_then_lose_then_gain_chain():
    """Green Heron decomposes into three atomic decisions, in order: activate
    (SKIP_OPTIONAL head), discard a food (SPEND_FOOD head), gain a food
    (GAIN_FOOD head)."""
    eng, pb = _make_engine_with_bird(
        "Trade 1 [wild] for any other type from the supply.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    for food in cards.ALL_FOODS:
        player.food[food] = 0
    # Two hand foods → SpendFoodDecision has >1 choice, so the engine consults
    # the agent (a single-choice decision is auto-resolved without an agent call).
    player.food[cards.Food.SEED] = 1
    player.food[cards.Food.FISH] = 1
    # Supply is infinite — GainFoodDecision always offers all 5 food types.

    seen: list[type[decisions.Decision[typing.Any]]] = []

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        seen.append(type(decision))
        # Accept the activation gate; pick the first FoodChoice for the two
        # food steps.
        if isinstance(decision, decisions.AcceptExchangeDecision):
            return typing.cast(
                C,
                next(
                    choice
                    for choice in decision.choices
                    if isinstance(choice, decisions.PayCostChoice)
                ),
            )
        return typing.cast(
            C,
            next(
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.FoodChoice)
            ),
        )

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert seen == [
        decisions.AcceptExchangeDecision,
        decisions.SpendFoodDecision,
        decisions.GainFoodDecision,
    ]
    assert (
        decisions.family_for(decisions.AcceptExchangeDecision)
        == decisions.DecisionFamily.SKIP_OPTIONAL
    )
    assert (
        decisions.family_for(decisions.SpendFoodDecision)
        == decisions.DecisionFamily.SPEND_FOOD
    )
    assert (
        decisions.family_for(decisions.GainFoodDecision)
        == decisions.DecisionFamily.GAIN_FOOD
    )


# ---------------------------------------------------------------------------
# Hermit Thrush — FEWEST_FOREST_GAINS_DIE


def test_fewest_forest_gains_die_only_min_player_gets_food():
    eng, pb = _make_engine_with_bird(
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    # Stash agents so the engine can ask non-active players.
    p0, p1 = gs.players
    # Give P0 zero forest birds, P1 one forest bird (so P0 is "fewest").
    p1.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=pb.bird))
    # Explicitly set feeder to 3 SEED dice (and nothing else) — single face,
    # so the engine will offer the optional reset before the gain.
    for food in cards.ALL_FOODS:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.counts[cards.Food.SEED] = 3
    gs.birdfeeder.choice_dice = 0
    food_before = {
        other_player.id: other_player.food.as_dict() for other_player in gs.players
    }

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        # Single-face rule: skip the optional reset and take the seed as-is.
        if isinstance(decision, decisions.ResetBirdfeederDecision):
            return typing.cast(
                C,
                next(
                    ch
                    for ch in decision.choices
                    if isinstance(ch, decisions.SkipChoice)
                ),
            )
        return typing.cast(
            C,
            next(
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.FoodChoice)
                and choice.food == cards.Food.SEED
            ),
        )

    eng.agents = [agent, agent]
    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.FOREST, "activate")
    assert p0.food[cards.Food.SEED] == food_before[0][cards.Food.SEED] + 1
    assert p1.food[cards.Food.SEED] == food_before[1][cards.Food.SEED]
    assert gs.birdfeeder.counts[cards.Food.SEED] == 2


def test_fewest_forest_gains_die_ties_each_gets_one():
    eng, pb = _make_engine_with_bird(
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    p0, p1 = gs.players
    # Both have zero forest birds -> tie -> both gain a die.
    # Explicitly single-face feeder so the optional reset is offered before each gain.
    for food in cards.ALL_FOODS:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.counts[cards.Food.SEED] = 5
    gs.birdfeeder.choice_dice = 0
    food_before = {
        other_player.id: other_player.food.as_dict() for other_player in gs.players
    }

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        # Single-face rule: skip the optional reset and take the seed as-is.
        if isinstance(decision, decisions.ResetBirdfeederDecision):
            return typing.cast(
                C,
                next(
                    ch
                    for ch in decision.choices
                    if isinstance(ch, decisions.SkipChoice)
                ),
            )
        return typing.cast(
            C,
            next(
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.FoodChoice)
                and choice.food == cards.Food.SEED
            ),
        )

    eng.agents = [agent, agent]
    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.FOREST, "activate")
    assert p0.food[cards.Food.SEED] == food_before[0][cards.Food.SEED] + 1
    assert p1.food[cards.Food.SEED] == food_before[1][cards.Food.SEED] + 1


def test_fewest_forest_gains_die_refills_empty_feeder():
    """An empty feeder is auto-rerolled (Rule 1) before the gain, so the
    fewest-forest player still takes a die rather than the power no-opping."""
    eng, pb = _make_engine_with_bird(
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    p0, p1 = gs.players
    # P1 has a forest bird, so only P0 is "fewest" and gains the die.
    p1.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=pb.bird))
    for food in cards.ALL_FOODS:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.choice_dice = 0  # truly empty feeder: clear the choice face too
    food_before = sum(p0.food.values())

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        # Decline any optional reset of the freshly rerolled feeder; take the
        # first die offered.
        if isinstance(decision, decisions.ResetBirdfeederDecision):
            for choice in decision.choices:
                if isinstance(choice, decisions.SkipChoice):
                    return typing.cast(C, choice)
        return typing.cast(C, decision.choices[0])

    eng.agents = [agent, agent]
    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.FOREST, "activate")
    assert gs.birdfeeder.total() > 0  # rerolled, not left empty
    assert sum(p0.food.values()) == food_before + 1  # one die was taken


def test_fewest_forest_auto_skips_when_active_player_not_fewest():
    """If the active player has MORE forest birds than the opponent, activating
    would only give the opponent a free die.  The handler auto-skips without
    consulting any agent."""
    eng, pb = _make_engine_with_bird(
        "Player(s) with the fewest birds in their [forest] gain 1 [die] from birdfeeder.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    p0, p1 = gs.players
    # P0 (active) has 2 forest birds; P1 has 0 — only P1 would qualify.
    p0.board[cards.Habitat.FOREST].extend(
        [state.PlayedBird(bird=pb.bird), state.PlayedBird(bird=pb.bird)]
    )
    for food in cards.ALL_FOODS:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.counts[cards.Food.SEED] = 5
    food_before_p1 = p1.food.as_dict().copy()

    # No agent should be consulted — use the strict _no_agent stub.
    eng.agents = [_no_agent, _no_agent]
    powers.dispatch_power(eng, _no_agent, p0, pb, cards.Habitat.FOREST, "activate")

    # P1 must NOT have received any food.
    assert p1.food.as_dict() == food_before_p1


# ---------------------------------------------------------------------------
# House Wren — PLAY_ADDITIONAL_BIRD_HERE


def test_play_additional_bird_here_grants_extra_play():
    eng, pb = _make_engine_with_bird(
        "Play an additional bird in this bird's habitat. Pay its normal cost.",
    )
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    before = len(gs.turn_extra_plays)
    powers.dispatch_power(eng, _no_agent, player, pb, cards.Habitat.FOREST, "play")
    assert len(gs.turn_extra_plays) == before + 1


# ---------------------------------------------------------------------------
# American Oystercatcher — DRAW_N_PLUS_ONE_DRAFT


def test_draw_n_plus_one_draft_each_player_picks_one_active_keeps_rest():
    eng, pb = _make_engine_with_bird(
        "Draw [card] equal to the number of players +1. Starting with you and "
        "proceeding clockwise, each player selects 1 of those cards and places "
        "it in their hand. You keep the extra card.",
    )
    gs = eng.state
    gs.current_player = 0
    p0, p1 = gs.players
    p0.hand = []
    p1.hand = []
    deck_before = len(gs.bird_deck)

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        # Always pick the first offered card.
        return decision.choices[0]

    eng.agents = [agent, agent]
    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.WETLAND, "play")

    # P0 accepts, draws 3, passes first 2 to P1, P1 returns first of those.
    # Net: p0 ends with 2 cards, p1 with 1.
    assert len(p0.hand) == 2
    assert len(p1.hand) == 1
    assert len(gs.bird_deck) == deck_before - 3


def test_draw_n_plus_one_draft_empty_deck_no_op():
    eng, pb = _make_engine_with_bird(
        "Draw [card] equal to the number of players +1. Starting with you and "
        "proceeding clockwise, each player selects 1 of those cards and places "
        "it in their hand. You keep the extra card.",
    )
    gs = eng.state
    gs.current_player = 0
    p0 = gs.me()
    p0.hand = []
    gs.players[1].hand = []
    gs.bird_deck = []
    gs.bird_discard = []

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        # Only the skip-optional fires before the deck-empty guard aborts.
        assert isinstance(
            decision, decisions.AcceptExchangeDecision
        ), f"unexpected decision: {type(decision).__name__}"
        return typing.cast(C, decision.choices[0])  # accept

    eng.agents = [agent, agent]
    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.WETLAND, "play")
    assert p0.hand == []
    assert gs.players[1].hand == []


# ---------------------------------------------------------------------------
# ALL_PLAYERS_DRAW — gap #11


def test_all_players_draw_each_player_gets_card_from_deck():
    """Each player draws from the deck (no tray menu, no decision); the tray
    is unchanged (gap #11)."""
    eng, pb = _make_engine_with_bird(
        "All players draw 1 [card] from the deck.",
        color=cards.PowerColor.WHITE,
    )
    gs = eng.state
    gs.current_player = 0
    p0, p1 = gs.players
    p0.hand = []
    p1.hand = []
    tray_before = list(gs.tray)
    deck_before = len(gs.bird_deck)

    powers.dispatch_power(eng, _no_agent, p0, pb, cards.Habitat.WETLAND, "play")

    # Each player gains 1 card.
    assert len(p0.hand) == 1
    assert len(p1.hand) == 1
    # Cards came from deck, not tray.
    assert len(gs.bird_deck) == deck_before - 2
    # Tray is unchanged (no mid-power refill).
    assert gs.tray == tray_before


def test_all_players_draw_does_not_consult_agent():
    """Deck-only draw needs no decision — the agent must never be called."""
    eng, pb = _make_engine_with_bird(
        "All players draw 1 [card] from the deck.",
        color=cards.PowerColor.WHITE,
    )
    gs = eng.state
    gs.current_player = 0
    p0 = gs.me()
    p0.hand = []

    # _no_agent raises on any consultation.
    powers.dispatch_power(eng, _no_agent, p0, pb, cards.Habitat.WETLAND, "play")


# ---------------------------------------------------------------------------
# Specific bird wiring


def test_target_birds_parse_to_expected_kinds():
    from wingspan import cards

    birds, _, _ = cards.load_all()
    by_name = {bird.name: bird for bird in birds}
    expected = {
        "Brant": cards.EffectKind.DRAW_FROM_TRAY_ALL,
        "Green Heron": cards.EffectKind.TRADE_WILD_FOOD,
        "Hermit Thrush": cards.EffectKind.FEWEST_FOREST_GAINS_DIE,
        "House Wren": cards.EffectKind.PLAY_ADDITIONAL_BIRD_HERE,
        "American Oystercatcher": cards.EffectKind.DRAW_N_PLUS_ONE_DRAFT,
    }
    for name, kind in expected.items():
        kinds = [effect.kind for effect in by_name[name].power.effects]
        assert kind in kinds, f"{name}: {kinds}"
