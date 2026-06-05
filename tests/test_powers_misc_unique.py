"""Tests for the misc-unique bird powers (off-family single-card effects).

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
    """Green Heron forces the trade — no optional gate (gap #18): discard a food,
    then gain any food from supply."""
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
        # Forced flow (gap #18): discard SEED → gain FRUIT.
        if isinstance(decision, decisions.SpendFoodDecision):
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


def test_trade_wild_food_is_a_forced_lose_then_gain_chain():
    """Green Heron (gap #18): no optional gate — forced discard then forced gain.
    Decomposes into exactly two decisions: discard (SPEND_FOOD) then gain
    (GAIN_FOOD)."""
    eng, pb = _make_engine_with_bird(
        "Trade 1 [wild] for any other type from the supply.",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    for food in cards.ALL_FOODS:
        player.food[food] = 0
    # Two foods → SpendFoodDecision has >1 choice, so the engine consults the
    # agent (a single-choice decision is auto-resolved without a call).
    player.food[cards.Food.SEED] = 1
    player.food[cards.Food.FISH] = 1
    # Supply is infinite — GainFoodDecision always offers all 5 food types.

    seen: list[type[decisions.Decision[typing.Any]]] = []

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        seen.append(type(decision))
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
        decisions.SpendFoodDecision,
        decisions.GainFoodDecision,
    ]
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
        # Accept the tied fewest-forest veto gate (gap #16).
        if isinstance(decision, decisions.AcceptExchangeDecision):
            return typing.cast(
                C,
                next(
                    c
                    for c in decision.choices
                    if isinstance(c, decisions.PayCostChoice)
                ),
            )
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
# American Bittern / Common Loon — FEWEST_WETLAND_DRAWS_CARD


_WETLAND_POWER = "Player(s) with the fewest birds in their [wetland] draw 1 [card]."


def _draw_agent[C: decisions.Choice](
    _engine: engine.Engine,
    decision: decisions.Decision[C],
) -> C:
    """Agent that accepts any AcceptExchangeDecision and picks the first draw source."""
    if isinstance(decision, decisions.AcceptExchangeDecision):
        return typing.cast(
            C,
            next(
                ch for ch in decision.choices if isinstance(ch, decisions.PayCostChoice)
            ),
        )
    return decision.choices[0]


def test_fewest_wetland_draws_card_strict_min():
    """Active player is the sole fewest-wetland player → draws 1 card, no veto asked."""
    eng, pb = _make_engine_with_bird(_WETLAND_POWER, color=cards.PowerColor.BROWN)
    gs = eng.state
    gs.current_player = 0
    p0, p1 = gs.players
    # Give P1 a wetland bird so P0 (0 birds) is strictly fewest.
    p1.board[cards.Habitat.WETLAND].append(state.PlayedBird(bird=pb.bird))
    hand_before_p0 = len(p0.hand)
    hand_before_p1 = len(p1.hand)

    eng.agents = [_draw_agent, _draw_agent]
    powers.dispatch_power(eng, _draw_agent, p0, pb, cards.Habitat.WETLAND, "activate")

    assert len(p0.hand) == hand_before_p0 + 1
    assert len(p1.hand) == hand_before_p1


def test_fewest_wetland_draws_card_tied_accepted():
    """Both players tied for fewest wetland birds → veto offered → accepted → both draw."""
    eng, pb = _make_engine_with_bird(_WETLAND_POWER, color=cards.PowerColor.BROWN)
    gs = eng.state
    gs.current_player = 0
    p0, p1 = gs.players
    # Both have 0 wetland birds — tied.
    hand_before_p0 = len(p0.hand)
    hand_before_p1 = len(p1.hand)

    # Verify the veto choice carries the correct card-draw ledger.
    seen_veto: list[decisions.AcceptExchangeDecision] = []

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            seen_veto.append(decision)
            return typing.cast(
                C,
                next(
                    ch
                    for ch in decision.choices
                    if isinstance(ch, decisions.PayCostChoice)
                ),
            )
        return decision.choices[0]

    eng.agents = [agent, agent]
    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.WETLAND, "activate")

    assert len(seen_veto) == 1
    accept_ch = next(
        ch for ch in seen_veto[0].choices if isinstance(ch, decisions.PayCostChoice)
    )
    assert accept_ch.gained_card_count == 1
    assert accept_ch.opp_gained_card_count == 1
    assert len(p0.hand) == hand_before_p0 + 1
    assert len(p1.hand) == hand_before_p1 + 1


def test_fewest_wetland_draws_card_tied_skipped():
    """Both players tied for fewest wetland birds → veto offered → skipped → neither draws."""
    eng, pb = _make_engine_with_bird(_WETLAND_POWER, color=cards.PowerColor.BROWN)
    gs = eng.state
    gs.current_player = 0
    p0, p1 = gs.players
    hand_before_p0 = len(p0.hand)
    hand_before_p1 = len(p1.hand)

    def skip_agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            return typing.cast(
                C,
                next(
                    ch
                    for ch in decision.choices
                    if isinstance(ch, decisions.SkipChoice)
                ),
            )
        return decision.choices[0]

    eng.agents = [skip_agent, skip_agent]
    powers.dispatch_power(eng, skip_agent, p0, pb, cards.Habitat.WETLAND, "activate")

    assert len(p0.hand) == hand_before_p0
    assert len(p1.hand) == hand_before_p1


def test_fewest_wetland_auto_skips_when_active_not_fewest():
    """Active player has MORE wetland birds than the opponent → power auto-skipped."""
    eng, pb = _make_engine_with_bird(_WETLAND_POWER, color=cards.PowerColor.BROWN)
    gs = eng.state
    gs.current_player = 0
    p0, p1 = gs.players
    # P0 (active) has 1 wetland bird; P1 has 0 — only P1 would qualify.
    p0.board[cards.Habitat.WETLAND].append(state.PlayedBird(bird=pb.bird))
    hand_before_p1 = len(p1.hand)

    eng.agents = [_no_agent, _no_agent]
    powers.dispatch_power(eng, _no_agent, p0, pb, cards.Habitat.WETLAND, "activate")

    assert len(p1.hand) == hand_before_p1


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
    """Each player draws from the deck (no tray menu, no draw decision); the
    tray is unchanged (gap #11). A veto gate is offered first (gap #16)."""
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

    def agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        # Accept the veto gate; no other decision should follow.
        assert isinstance(decision, decisions.AcceptExchangeDecision)
        return typing.cast(
            C,
            next(c for c in decision.choices if isinstance(c, decisions.PayCostChoice)),
        )

    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.WETLAND, "play")

    # Each player gains 1 card.
    assert len(p0.hand) == 1
    assert len(p1.hand) == 1
    # Cards came from deck, not tray.
    assert len(gs.bird_deck) == deck_before - 2
    # Tray is unchanged (no mid-power refill).
    assert gs.tray == tray_before


def test_all_players_draw_veto_skips_entire_draw():
    """Declining the veto gate (gap #16) leaves both hands untouched."""
    eng, pb = _make_engine_with_bird(
        "All players draw 1 [card] from the deck.",
        color=cards.PowerColor.WHITE,
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
        assert isinstance(decision, decisions.AcceptExchangeDecision)
        return typing.cast(
            C, next(c for c in decision.choices if isinstance(c, decisions.SkipChoice))
        )

    powers.dispatch_power(eng, agent, p0, pb, cards.Habitat.WETLAND, "play")
    assert p0.hand == []
    assert p1.hand == []
    assert len(gs.bird_deck) == deck_before


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


# ---------------------------------------------------------------------------
# PLAY_ADDITIONAL_BIRD wrong-habitat skip


def test_play_additional_bird_wrong_habitat_does_not_grant_extra_play():
    """A PLAY_ADDITIONAL_BIRD effect restricted to WETLAND, fired while
    activating in FOREST, should skip the grant silently."""
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(0)
    gs = state.new_game(rng, birds, bonuses, goals)
    eng = engine.Engine(gs)
    gs.current_player = 0
    player = gs.me()

    # Synthesize a bird with PLAY_ADDITIONAL_BIRD restricted to WETLAND.
    template = next(b for b in birds if b.color == cards.PowerColor.WHITE)
    restricted_power = cards.Power(
        color=cards.PowerColor.WHITE,
        effects=(
            cards.Effect(
                kind=cards.EffectKind.PLAY_ADDITIONAL_BIRD,
                habitat=cards.Habitat.WETLAND,
            ),
        ),
    )
    bird = template.model_copy(update={"power": restricted_power})
    pb = state.PlayedBird(bird=bird)
    player.board[cards.Habitat.FOREST] = [pb]

    extra_plays_before = len(gs.turn_extra_plays)
    powers.dispatch_power(eng, _no_agent, player, pb, cards.Habitat.FOREST, "play")
    assert len(gs.turn_extra_plays) == extra_plays_before


# ---------------------------------------------------------------------------
# Predator hunt, move-if-rightmost, repeat-predator edge cases


def test_predator_hunt_empty_deck_logs_and_skips():
    """PREDATOR_HUNT when the deck is empty logs 'deck empty' and leaves the
    tuck count at zero."""
    eng, pb = _make_engine_with_bird(
        "Look at a [card] from the deck."
        " If less than 75 cm, tuck it behind this bird. If not, discard it",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    player = gs.me()
    player.board[cards.Habitat.FOREST] = [pb]
    gs.bird_deck = []
    gs.bird_discard = []

    log_lines: list[str] = []
    eng.log = lambda msg: log_lines.append(msg)  # type: ignore[method-assign]
    powers.dispatch_power(eng, _no_agent, player, pb, cards.Habitat.FOREST, "activate")
    assert pb.tucked_cards == 0
    assert any("deck empty" in line for line in log_lines)


def test_move_bird_if_rightmost_no_other_habitat_space_skips():
    """MOVE_BIRD_IF_RIGHTMOST when all other habitats are full logs the skip
    and leaves the board unchanged."""
    eng, pb = _make_engine_with_bird(
        "If this bird is to the right of all other birds in its habitat,"
        " move it to another habitat",
        color=cards.PowerColor.BROWN,
    )
    gs = eng.state
    gs.current_player = 0
    player = gs.me()

    # pb is rightmost (only bird) in WETLAND.
    player.board[cards.Habitat.WETLAND] = [pb]

    # Fill FOREST and GRASSLAND to capacity so can_play_in returns False.
    filler = next(b for b in gs.bird_deck)
    for habitat in (cards.Habitat.FOREST, cards.Habitat.GRASSLAND):
        player.board[habitat] = [
            state.PlayedBird(bird=filler) for _ in range(state.ROW_SLOTS)
        ]
    board_snapshot = {h: list(row) for h, row in player.board.items()}

    powers.dispatch_power(eng, _no_agent, player, pb, cards.Habitat.WETLAND, "activate")

    # Board is unchanged.
    for habitat, row in board_snapshot.items():
        assert player.board[habitat] == row


def test_repeat_predator_power_fires_the_target_predators_hunt():
    """REPEAT_PREDATOR_POWER picks another predator in the habitat and repeats
    its PREDATOR_HUNT (lines 185-200 in predator_repeat.py)."""
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(0)
    gs = state.new_game(rng, birds, bonuses, goals)
    gs.current_player = 0
    eng = engine.Engine(gs)
    player = gs.me()

    # A real catalog predator with PREDATOR_HUNT.
    predator_bird = next(
        bird
        for bird in birds
        if bird.predator
        and any(
            eff.kind == cards.EffectKind.PREDATOR_HUNT for eff in bird.power.effects
        )
    )
    # A bird with REPEAT_PREDATOR_POWER (Hooded Merganser and similar).
    repeat_bird = next(
        bird
        for bird in birds
        if any(
            eff.kind == cards.EffectKind.REPEAT_PREDATOR_POWER
            for eff in bird.power.effects
        )
    )
    predator_pb = state.PlayedBird(bird=predator_bird)
    repeat_pb = state.PlayedBird(bird=repeat_bird)
    player.board[cards.Habitat.WETLAND] = [predator_pb, repeat_pb]

    # Provide a hunt target; predator may or may not succeed — we only care
    # that the repeat *dispatched* the hunt (lines 185-200 covered).
    hunt_target = next(b for b in birds if b not in (predator_bird, repeat_bird))
    gs.bird_deck = [hunt_target]

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        # BirdPowerPickPlayedBirdDecision: pick the predator.
        if isinstance(decision, decisions.BirdPowerPickPlayedBirdDecision):
            return typing.cast(
                C,
                next(ch for ch in decision.choices if ch.played_bird is predator_pb),
            )
        raise AssertionError(f"unexpected decision: {type(decision).__name__}")

    log_lines: list[str] = []
    eng.log = lambda msg: log_lines.append(msg)  # type: ignore[method-assign]
    powers.dispatch_power(
        eng, agent, player, repeat_pb, cards.Habitat.WETLAND, "activate"
    )

    # The repeat log line proves execution reached lines 185-200.
    assert any(
        "repeat" in line.lower() for line in log_lines
    ), f"Expected repeat log line; got: {log_lines}"
