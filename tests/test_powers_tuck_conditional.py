"""Tests for the four 'Tuck N from hand. If you do, [X].' combined powers.

These powers were previously broken: the secondary effect (draw / lay / gain food)
fired *before* the tuck decision and always fired even when the tuck was skipped.
The fix models each pattern as a combined ``EffectKind`` whose single handler offers
tuck-or-skip first and only applies the secondary effect when the tuck is accepted.
"""

from __future__ import annotations

import os
import random
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers

# ---------------------------------------------------------------------------
# Shared fixtures


def _find(birds: list[cards.Bird], name: str) -> cards.Bird:
    return next(bird for bird in birds if bird.name == name)


def _engine_with_agents(
    seed: int = 0,
) -> tuple[engine.Engine, list[cards.Bird]]:
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(seed)

    def _no_agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        raise AssertionError(
            f"unexpected agent consultation: {type(decision).__name__}"
        )

    gs = state.new_game(rng, birds, bonuses, goals)
    return engine.Engine(gs, agents=[_no_agent, _no_agent]), birds


# ---------------------------------------------------------------------------
# Parser tests: each combined pattern produces the right EffectKind only


def test_parser_tuck_then_draw_correct_kind():
    power = cards.parse_power(
        cards.PowerColor.BROWN,
        "Tuck 1 [card] from your hand behind this bird. If you do, draw 1 [card].",
    )
    kinds = {eff.kind for eff in power.effects}
    assert cards.EffectKind.TUCK_FROM_HAND_THEN_DRAW in kinds
    assert (
        cards.EffectKind.DRAW_CARDS not in kinds
    ), "standalone DRAW_CARDS must not appear"
    assert (
        cards.EffectKind.TUCK_FROM_HAND not in kinds
    ), "standalone TUCK_FROM_HAND must not appear"
    assert len(power.effects) == 1


def test_parser_tuck_then_lay_on_this_correct_kind():
    power = cards.parse_power(
        cards.PowerColor.BROWN,
        "Tuck 1 [card] from your hand behind this bird."
        " If you do, you may also lay 1 [egg] on this bird.",
    )
    kinds = {eff.kind for eff in power.effects}
    assert cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS in kinds
    assert cards.EffectKind.LAY_EGG_ON_THIS not in kinds
    assert cards.EffectKind.TUCK_FROM_HAND not in kinds
    assert len(power.effects) == 1


def test_parser_tuck_then_lay_any_correct_kind():
    power = cards.parse_power(
        cards.PowerColor.BROWN,
        "Tuck 1 [card] from your hand behind this bird. If you do, lay 1 [egg] on any bird.",
    )
    kinds = {eff.kind for eff in power.effects}
    assert cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ANY in kinds
    assert cards.EffectKind.LAY_EGG_ANY not in kinds
    assert cards.EffectKind.TUCK_FROM_HAND not in kinds
    assert len(power.effects) == 1


def test_parser_tuck_then_gain_food_supply_correct_kind():
    power = cards.parse_power(
        cards.PowerColor.BROWN,
        "Tuck 1 [card] from your hand behind this bird. If you do, gain 1 [fruit] from the supply.",
    )
    kinds = {eff.kind for eff in power.effects}
    assert cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY in kinds
    assert cards.EffectKind.GAIN_FOOD_SUPPLY not in kinds
    assert cards.EffectKind.TUCK_FROM_HAND not in kinds
    eff = next(
        e
        for e in power.effects
        if e.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY
    )
    assert eff.food == cards.Food.FRUIT


def test_parser_standalone_tuck_unaffected():
    """A plain 'Tuck N from hand' with no 'If you do' clause still maps to TUCK_FROM_HAND."""
    power = cards.parse_power(
        cards.PowerColor.BROWN,
        "Tuck 1 [card] from your hand behind this bird.",
    )
    kinds = {eff.kind for eff in power.effects}
    assert cards.EffectKind.TUCK_FROM_HAND in kinds
    assert cards.EffectKind.TUCK_FROM_HAND_THEN_DRAW not in kinds


def test_parser_standalone_draw_unaffected():
    """A plain 'Draw N [card]' with no tuck prefix still maps to DRAW_CARDS."""
    power = cards.parse_power(cards.PowerColor.BROWN, "Draw 1 [card].")
    kinds = {eff.kind for eff in power.effects}
    assert cards.EffectKind.DRAW_CARDS in kinds
    assert cards.EffectKind.TUCK_FROM_HAND_THEN_DRAW not in kinds


# ---------------------------------------------------------------------------
# American Coot (tuck-then-draw) engine behaviour tests


def _setup_tuck_then_draw(
    eng: engine.Engine, birds: list[cards.Bird]
) -> tuple[state.Player, state.PlayedBird, list[cards.Bird]]:
    """Place the American Coot in the wetland and give the player 2 hand cards."""
    coot = _find(birds, "American Coot")
    player = eng.state.players[0]
    pb = state.PlayedBird(bird=coot)
    player.board[cards.Habitat.WETLAND] = [pb]

    # Give the player two known hand cards (any two birds that aren't the Coot).
    hand_birds = [bird for bird in birds if bird is not coot][:2]
    player.hand = list(hand_birds)
    return player, pb, hand_birds


def test_tuck_then_draw_tuck_accepted_order_and_state():
    """When tuck is accepted: hand shrinks by 1 (tuck), grows by 1 (draw),
    and the draw decision is offered AFTER the tuck, not before."""
    eng, birds = _engine_with_agents(seed=0)
    player, pb, hand_birds = _setup_tuck_then_draw(eng, birds)
    eng.state.bird_deck = [
        bird for bird in birds if bird not in (pb.bird, *hand_birds)
    ][:5]

    decision_types: list[type[decisions.Decision[typing.Any]]] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        decision_types.append(type(decision))
        if isinstance(decision, decisions.ActivateTuckDecision):
            # Accept the tuck gate.
            return typing.cast(
                C,
                next(
                    ch
                    for ch in decision.choices
                    if isinstance(ch, decisions.TuckActivateChoice)
                ),
            )
        if isinstance(decision, decisions.BirdPowerTuckFromHandDecision):
            # Mandatory card selection — all choices are BirdChoice; pick the first.
            return typing.cast(C, decision.choices[0])
        if isinstance(decision, decisions.DrawCardsPickSourceDecision):
            # Draw from the deck.
            return typing.cast(
                C,
                next(ch for ch in decision.choices if ch.source == "deck"),
            )
        raise AssertionError(f"unexpected decision: {type(decision).__name__}")

    hand_before = len(player.hand)
    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")

    # Gate, then tuck selection, then draw — in that order.
    assert decision_types == [
        decisions.ActivateTuckDecision,
        decisions.BirdPowerTuckFromHandDecision,
        decisions.DrawCardsPickSourceDecision,
    ], f"wrong decision order: {[t.__name__ for t in decision_types]}"

    assert pb.tucked_cards == 1
    assert len(player.hand) == hand_before  # -1 tuck +1 draw = net 0


def test_tuck_then_draw_tuck_skipped_no_draw():
    """When the player skips the tuck, no draw decision is offered and state is unchanged."""
    eng, birds = _engine_with_agents(seed=1)
    player, pb, _ = _setup_tuck_then_draw(eng, birds)

    draw_offered = False

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        nonlocal draw_offered
        if isinstance(decision, decisions.DrawCardsPickSourceDecision):
            draw_offered = True
        # Always skip the tuck.
        return typing.cast(
            C,
            next(ch for ch in decision.choices if isinstance(ch, decisions.SkipChoice)),
        )

    hand_before = list(player.hand)
    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")

    assert not draw_offered, "draw must not be offered when tuck is skipped"
    assert pb.tucked_cards == 0
    assert player.hand == hand_before


def test_tuck_then_draw_empty_hand_skips_silently():
    """With an empty hand, no decision is offered at all and state is unchanged."""
    eng, birds = _engine_with_agents(seed=2)
    coot = _find(birds, "American Coot")
    player = eng.state.players[0]
    pb = state.PlayedBird(bird=coot)
    player.board[cards.Habitat.WETLAND] = [pb]
    player.hand = []

    decision_offered = False

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        nonlocal decision_offered
        decision_offered = True
        raise AssertionError("should not be called with empty hand")

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert not decision_offered
    assert pb.tucked_cards == 0


# ---------------------------------------------------------------------------
# Tuck-then-lay-on-this engine behaviour tests


def test_tuck_then_lay_on_this_tuck_accepted_lay_accepted():
    """After tucking, the player is offered an optional lay on the triggering bird."""
    eng, birds = _engine_with_agents(seed=3)
    # Find a bird with the TUCK_FROM_HAND_THEN_LAY_ON_THIS power.
    source_bird = next(
        bird
        for bird in birds
        if any(
            eff.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS
            for eff in bird.power.effects
        )
    )
    player = eng.state.players[0]
    pb = state.PlayedBird(bird=source_bird)
    player.board[cards.Habitat.WETLAND] = [pb]
    hand_card = next(b for b in birds if b is not source_bird)
    player.hand = [hand_card]

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        if isinstance(decision, decisions.ActivateTuckDecision):
            return typing.cast(
                C,
                next(
                    ch
                    for ch in decision.choices
                    if isinstance(ch, decisions.TuckActivateChoice)
                ),
            )
        if isinstance(decision, decisions.BirdPowerTuckFromHandDecision):
            # Mandatory card selection — pick the first (and only) BirdChoice.
            return typing.cast(C, decision.choices[0])
        if isinstance(decision, decisions.LayEggDecision):
            # Accept the optional lay.
            return typing.cast(
                C,
                next(
                    ch
                    for ch in decision.choices
                    if isinstance(ch, decisions.BoardTargetChoice)
                ),
            )
        raise AssertionError(f"unexpected decision: {type(decision).__name__}")

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert pb.tucked_cards == 1
    assert pb.eggs == 1


def test_tuck_then_lay_on_this_tuck_skipped_no_lay():
    """Skipping the tuck prevents the lay from being offered."""
    eng, birds = _engine_with_agents(seed=4)
    source_bird = next(
        bird
        for bird in birds
        if any(
            eff.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS
            for eff in bird.power.effects
        )
    )
    player = eng.state.players[0]
    pb = state.PlayedBird(bird=source_bird)
    player.board[cards.Habitat.WETLAND] = [pb]
    player.hand = [next(b for b in birds if b is not source_bird)]

    lay_offered = False

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        nonlocal lay_offered
        if isinstance(decision, decisions.LayEggDecision):
            lay_offered = True
        return typing.cast(
            C,
            next(ch for ch in decision.choices if isinstance(ch, decisions.SkipChoice)),
        )

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert not lay_offered
    assert pb.tucked_cards == 0
    assert pb.eggs == 0


# ---------------------------------------------------------------------------
# Tuck-then-gain-food-supply engine behaviour tests


def test_tuck_then_gain_food_tuck_accepted_gains_food():
    """After tucking, the player gains the specified food from supply."""
    eng, birds = _engine_with_agents(seed=5)
    source_bird = next(
        bird
        for bird in birds
        if any(
            eff.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY
            for eff in bird.power.effects
        )
    )
    food_to_gain = next(
        eff.food
        for eff in source_bird.power.effects
        if eff.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY
    )
    assert food_to_gain is not None

    player = eng.state.players[0]
    pb = state.PlayedBird(bird=source_bird)
    player.board[cards.Habitat.WETLAND] = [pb]
    player.hand = [next(b for b in birds if b is not source_bird)]
    food_before = player.food[food_to_gain]

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        if isinstance(decision, decisions.ActivateTuckDecision):
            return typing.cast(
                C,
                next(
                    ch
                    for ch in decision.choices
                    if isinstance(ch, decisions.TuckActivateChoice)
                ),
            )
        if isinstance(decision, decisions.BirdPowerTuckFromHandDecision):
            # Mandatory card selection — pick the first BirdChoice.
            return typing.cast(C, decision.choices[0])
        raise AssertionError(f"unexpected decision: {type(decision).__name__}")

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert pb.tucked_cards == 1
    assert player.food[food_to_gain] == food_before + 1


def test_tuck_then_gain_food_tuck_skipped_no_food():
    """Skipping the tuck prevents the food gain."""
    eng, birds = _engine_with_agents(seed=6)
    source_bird = next(
        bird
        for bird in birds
        if any(
            eff.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY
            for eff in bird.power.effects
        )
    )
    food_to_gain = next(
        eff.food
        for eff in source_bird.power.effects
        if eff.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_SUPPLY
    )
    assert food_to_gain is not None

    player = eng.state.players[0]
    pb = state.PlayedBird(bird=source_bird)
    player.board[cards.Habitat.WETLAND] = [pb]
    player.hand = [next(b for b in birds if b is not source_bird)]
    food_before = player.food[food_to_gain]

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        return typing.cast(
            C,
            next(ch for ch in decision.choices if isinstance(ch, decisions.SkipChoice)),
        )

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert pb.tucked_cards == 0
    assert player.food[food_to_gain] == food_before


# ---------------------------------------------------------------------------
# American Coot is correctly parsed from live card data


def test_american_coot_has_tuck_then_draw_effect():
    """American Coot's loaded power must map to TUCK_FROM_HAND_THEN_DRAW only."""
    birds, _, _ = cards.load_all()
    coot = _find(birds, "American Coot")
    kinds = {eff.kind for eff in coot.power.effects}
    assert kinds == {
        cards.EffectKind.TUCK_FROM_HAND_THEN_DRAW
    }, f"American Coot effect kinds: {kinds}"


# ---------------------------------------------------------------------------
# TUCK_FROM_HAND handler (plain opt-in tuck with no secondary effect)


def _setup_tuck_from_hand(
    eng: engine.Engine, birds: list[cards.Bird]
) -> tuple[state.Player, state.PlayedBird]:
    """Stage a synthetic bird with plain TUCK_FROM_HAND in the wetland."""
    # No catalog bird parses to the standalone TUCK_FROM_HAND kind — all real
    # catalog birds have a combined "tuck then [secondary]" form.  Build a
    # synthetic bird by grafting the plain-tuck power onto an arbitrary template.
    template = next(bird for bird in birds if bird.color == cards.PowerColor.BROWN)
    tuck_power = cards.parse_power(
        cards.PowerColor.BROWN,
        "Tuck 1 [card] from your hand behind this bird.",
    )
    source_bird = template.model_copy(update={"power": tuck_power})
    player = eng.state.players[0]
    pb = state.PlayedBird(bird=source_bird)
    player.board[cards.Habitat.WETLAND] = [pb]
    player.hand = [next(b for b in birds if b is not template)]
    return player, pb


def test_tuck_from_hand_accept_tucks_card_and_shrinks_hand():
    """Accepting the tuck gate and selecting a card moves it from hand to the
    bird's tuck pile."""
    eng, birds = _engine_with_agents(seed=7)
    player, pb = _setup_tuck_from_hand(eng, birds)
    hand_before = len(player.hand)

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        if isinstance(decision, decisions.ActivateTuckDecision):
            return typing.cast(
                C,
                next(
                    ch
                    for ch in decision.choices
                    if isinstance(ch, decisions.TuckActivateChoice)
                ),
            )
        # BirdPowerTuckFromHandDecision — pick the first available card.
        if isinstance(decision, decisions.BirdPowerTuckFromHandDecision):
            return typing.cast(C, decision.choices[0])
        raise AssertionError(f"unexpected decision: {type(decision).__name__}")

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert pb.tucked_cards == 1
    assert len(player.hand) == hand_before - 1


def test_tuck_from_hand_skip_gate_leaves_state_unchanged():
    """Answering the tuck gate with SkipChoice leaves hand and tuck count
    untouched."""
    eng, birds = _engine_with_agents(seed=8)
    player, pb = _setup_tuck_from_hand(eng, birds)
    hand_before = list(player.hand)

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        return typing.cast(
            C,
            next(ch for ch in decision.choices if isinstance(ch, decisions.SkipChoice)),
        )

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert pb.tucked_cards == 0
    assert player.hand == hand_before


# ---------------------------------------------------------------------------
# TUCK_FROM_HAND_THEN_GAIN_FOOD_CHOICE (e.g. Pygmy Nuthatch)


def test_tuck_from_hand_then_gain_food_choice_gains_selected_food():
    """After tucking, the agent picks one of the two supply options; the player
    receives exactly +1 of that food type."""
    eng, birds = _engine_with_agents(seed=9)
    source_bird = next(
        bird
        for bird in birds
        if any(
            eff.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_CHOICE
            for eff in bird.power.effects
        )
    )
    food_eff = next(
        eff
        for eff in source_bird.power.effects
        if eff.kind == cards.EffectKind.TUCK_FROM_HAND_THEN_GAIN_FOOD_CHOICE
    )
    assert food_eff.food_a is not None and food_eff.food_b is not None

    player = eng.state.players[0]
    pb = state.PlayedBird(bird=source_bird)
    player.board[cards.Habitat.WETLAND] = [pb]
    player.hand = [next(b for b in birds if b is not source_bird)]
    food_before_a = player.food[food_eff.food_a]

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        if isinstance(decision, decisions.ActivateTuckDecision):
            return typing.cast(
                C,
                next(
                    ch
                    for ch in decision.choices
                    if isinstance(ch, decisions.TuckActivateChoice)
                ),
            )
        if isinstance(decision, decisions.BirdPowerTuckFromHandDecision):
            return typing.cast(C, decision.choices[0])
        if isinstance(decision, decisions.GainFoodDecision):
            # Always pick food_a for a deterministic assertion.
            return typing.cast(
                C,
                next(
                    ch
                    for ch in decision.choices
                    if isinstance(ch, decisions.FoodChoice)
                    and ch.food == food_eff.food_a
                ),
            )
        raise AssertionError(f"unexpected decision: {type(decision).__name__}")

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "activate")
    assert pb.tucked_cards == 1
    assert len(player.hand) == 0
    assert player.food[food_eff.food_a] == food_before_a + 1
