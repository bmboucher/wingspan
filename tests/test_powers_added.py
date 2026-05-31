"""Tests for the bird powers added to reach 100% coverage."""

from __future__ import annotations

import os
import random
import sys
import typing

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state
from wingspan.engine import actions, powers


def _no_agent[C: decisions.Choice](
    _engine: engine.Engine,
    _decision: decisions.Decision[C],
) -> C:
    """An ``Agent``-typed stub for powers that resolve without a decision; it
    raises if a power unexpectedly consults it."""
    raise AssertionError(
        f"agent should not be consulted (got {type(_decision).__name__})"
    )


def _find(birds: list[cards.Bird], name: str) -> cards.Bird:
    return next(bird for bird in birds if bird.name == name)


def _engine(seed: int = 0) -> tuple[engine.Engine, list[cards.Bird]]:
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(seed)
    gs = state.new_game(rng, birds, bonuses, goals)
    return (
        engine.Engine(gs, agents=[_no_agent, _no_agent]),
        birds,
    )


# --- Coverage sanity -----------------------------------------------------


def test_full_coverage_no_unimplemented():
    birds, _, _ = cards.load_all()
    bad = [
        bird.name
        for bird in birds
        if any(
            effect.kind == cards.EffectKind.UNIMPLEMENTED
            for effect in bird.power.effects
        )
    ]
    assert bad == [], f"unimplemented: {bad}"


# --- Parser tests --------------------------------------------------------


def test_parser_draw_bonus_keep():
    power = cards.parse_power(
        cards.PowerColor.WHITE, "Draw 2 new bonus cards and keep 1."
    )
    assert any(
        effect.kind == cards.EffectKind.DRAW_BONUS_KEEP
        and effect.amount == 2
        and effect.keep_count == 1
        for effect in power.effects
    )


def test_parser_lay_egg_all_nest():
    power = cards.parse_power(
        cards.PowerColor.WHITE,
        "Lay 1 [egg] on each of your birds with a [cavity] nest.",
    )
    eff = next(
        effect
        for effect in power.effects
        if effect.kind == cards.EffectKind.LAY_EGG_ALL_NEST
    )
    assert eff.nest == cards.NestType.CAVITY
    assert eff.amount == 1


def test_parser_gain_all_food_feeder():
    power = cards.parse_power(
        cards.PowerColor.WHITE, "Gain all [fish] that are in the birdfeeder."
    )
    eff = next(
        effect
        for effect in power.effects
        if effect.kind == cards.EffectKind.GAIN_ALL_FOOD_FEEDER
    )
    assert eff.food == cards.Food.FISH


def test_parser_tuck_from_deck_paid():
    power = cards.parse_power(
        cards.PowerColor.BROWN,
        "Discard 1 [seed] to tuck 2 [card] from the deck behind this bird.",
    )
    eff = next(
        effect
        for effect in power.effects
        if effect.kind == cards.EffectKind.TUCK_FROM_DECK_PAID
    )
    assert eff.food == cards.Food.SEED and eff.amount == 2


def test_parser_predator_hunt():
    power = cards.parse_power(
        cards.PowerColor.BROWN,
        "Look at a [card] from the deck. If less than 75cm, tuck it behind this bird. If not, discard it.",
    )
    eff = next(
        effect
        for effect in power.effects
        if effect.kind == cards.EffectKind.PREDATOR_HUNT
    )
    assert eff.max_wingspan_cm == 75


def test_parser_move_rightmost():
    power = cards.parse_power(
        cards.PowerColor.BROWN,
        "If this bird is to the right of all other birds in its habitat, move it to another habitat.",
    )
    assert any(
        effect.kind == cards.EffectKind.MOVE_BIRD_IF_RIGHTMOST
        for effect in power.effects
    )


def test_parser_repeat_brown():
    power = cards.parse_power(
        cards.PowerColor.BROWN, "Repeat a brown power on another bird in this habitat."
    )
    assert any(
        effect.kind == cards.EffectKind.REPEAT_BROWN_POWER for effect in power.effects
    )


def test_parser_repeat_predator():
    power = cards.parse_power(
        cards.PowerColor.BROWN, "Repeat 1 [predator] power in this habitat."
    )
    assert any(
        effect.kind == cards.EffectKind.REPEAT_PREDATOR_POWER
        for effect in power.effects
    )


def test_parser_pink_lay_egg_on_nest():
    power = cards.parse_power(
        cards.PowerColor.PINK,
        "When another player takes the 'lay eggs' action, lay 1 [egg] on a bird with a [bowl] nest.",
    )
    eff = next(
        effect
        for effect in power.effects
        if effect.kind == cards.EffectKind.PINK_LAY_EGG_ON_NEST
    )
    assert eff.nest == cards.NestType.BOWL


def test_parser_pink_predator_feeder():
    power = cards.parse_power(
        cards.PowerColor.PINK,
        "When another player's [predator] succeeds, gain 1 [die] from the birdfeeder.",
    )
    assert any(
        effect.kind == cards.EffectKind.PINK_PREDATOR_FEEDER for effect in power.effects
    )


# --- Engine behavior tests ----------------------------------------------


def test_draw_bonus_keep_keeps_one_discards_one():
    eng, birds = _engine(seed=1)
    bird = _find(birds, "Atlantic Puffin")
    player = eng.state.me()
    pb = state.PlayedBird(bird=bird)
    player.board[cards.Habitat.WETLAND].append(pb)

    # Force a known top of bonus deck.
    eng.state.bonus_deck = list(eng.state.bonus_deck[:5])  # at least 2
    before_have = len(player.bonus_cards)
    before_deck = len(eng.state.bonus_deck)

    def agent[C: decisions.Choice](
        _engine: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        assert isinstance(decision, decisions.BirdPowerPickBonusCardDecision)
        return typing.cast(C, decision.choices[0])

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.WETLAND, "play")
    assert len(player.bonus_cards) == before_have + 1
    assert len(eng.state.bonus_deck) == before_deck - 2
    assert len(eng.state.bonus_discard) >= 1


def test_lay_egg_all_nest_adds_one_to_each_matching():
    eng, birds = _engine(seed=2)
    bobolink = _find(birds, "Bobolink")  # ground
    player = eng.state.me()
    # 3 birds with ground nests + 1 non-ground
    ground = [
        bird
        for bird in birds
        if bird.nest == cards.NestType.GROUND and bird is not bobolink
    ][:3]
    non_ground = next(
        bird
        for bird in birds
        if bird.nest != cards.NestType.GROUND and bird.egg_limit > 0
    )
    targets = [state.PlayedBird(bird=bird) for bird in ground]
    other = state.PlayedBird(bird=non_ground)
    pb = state.PlayedBird(bird=bobolink)
    player.board[cards.Habitat.GRASSLAND].extend(targets + [other, pb])

    powers.dispatch_power(eng, _no_agent, player, pb, cards.Habitat.GRASSLAND, "play")

    for target in targets:
        assert target.eggs == 1
    assert other.eggs == 0


def test_gain_all_food_feeder_drains_one_face():
    eng, birds = _engine(seed=3)
    bird = _find(birds, "Bald Eagle")  # fish
    player = eng.state.me()
    pb = state.PlayedBird(bird=bird)
    player.board[cards.Habitat.WETLAND].append(pb)
    eng.state.birdfeeder.counts.zero()
    eng.state.birdfeeder.counts[cards.Food.FISH] = 3
    eng.state.birdfeeder.counts[cards.Food.SEED] = 2
    for food in player.food:
        player.food[food] = 0

    powers.dispatch_power(eng, _no_agent, player, pb, cards.Habitat.WETLAND, "play")
    assert eng.state.birdfeeder.counts[cards.Food.FISH] == 0
    assert player.food[cards.Food.FISH] == 3
    assert eng.state.birdfeeder.counts[cards.Food.SEED] == 2  # untouched


def test_tuck_from_deck_paid_spends_food_and_tucks():
    eng, birds = _engine(seed=4)
    bird = _find(birds, "Sandhill Crane")  # discard 1 seed -> tuck 2
    player = eng.state.me()
    for food in player.food:
        player.food[food] = 0
    player.food[cards.Food.SEED] = 1
    pb = state.PlayedBird(bird=bird)
    player.board[cards.Habitat.GRASSLAND].append(pb)
    deck_before = len(eng.state.bird_deck)

    def agent[C: decisions.Choice](
        _engine: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        return typing.cast(
            C,
            next(
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.PayCostChoice)
            ),
        )

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.GRASSLAND, "activate")
    assert player.food[cards.Food.SEED] == 0
    assert pb.tucked_cards == 2
    assert len(eng.state.bird_deck) == deck_before - 2


def test_tuck_from_deck_paid_uses_accept_exchange_with_trade_terms():
    """The discard-food-to-tuck commit is a unified ``AcceptExchangeDecision``
    (commit-to-cost head) whose ``PayCostChoice`` carries the trade terms."""
    eng, birds = _engine(seed=14)
    bird = _find(birds, "Sandhill Crane")  # discard 1 seed -> tuck 2
    player = eng.state.me()
    for food in player.food:
        player.food[food] = 0
    player.food[cards.Food.SEED] = 1
    pb = state.PlayedBird(bird=bird)
    player.board[cards.Habitat.GRASSLAND].append(pb)

    captured: list[decisions.Decision[typing.Any]] = []

    def agent[C: decisions.Choice](
        _engine: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        captured.append(decision)
        return typing.cast(
            C,
            next(
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.PayCostChoice)
            ),
        )

    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.GRASSLAND, "activate")
    assert len(captured) == 1
    decision = captured[0]
    assert isinstance(decision, decisions.AcceptExchangeDecision)
    pay = next(
        choice
        for choice in decision.choices
        if isinstance(choice, decisions.PayCostChoice)
    )
    assert pay.paid_food == cards.Food.SEED
    assert pay.gained_tuck_count == 2
    assert pay.paid_egg_count == 0
    assert (
        decisions.family_for(decisions.AcceptExchangeDecision)
        == decisions.DecisionFamily.COMMIT_TO_COST
    )


def test_tuck_from_deck_paid_skip_when_no_food():
    eng, birds = _engine(seed=5)
    bird = _find(birds, "Canada Goose")
    player = eng.state.me()
    for food in player.food:
        player.food[food] = 0
    pb = state.PlayedBird(bird=bird)
    player.board[cards.Habitat.GRASSLAND].append(pb)
    deck_before = len(eng.state.bird_deck)

    powers.dispatch_power(
        eng, _no_agent, player, pb, cards.Habitat.GRASSLAND, "activate"
    )
    assert pb.tucked_cards == 0
    assert len(eng.state.bird_deck) == deck_before


def test_predator_hunt_tucks_small_bird():
    eng, birds = _engine(seed=6)
    hawk = _find(birds, "Cooper's Hawk")  # <75cm
    player = eng.state.me()
    pb = state.PlayedBird(bird=hawk)
    player.board[cards.Habitat.FOREST].append(pb)
    # Force a small bird on top of the deck.
    small = next(
        bird
        for bird in birds
        if bird.wingspan_cm and bird.wingspan_cm < 30 and bird is not hawk
    )
    eng.state.bird_deck.append(small)

    powers.dispatch_power(eng, _no_agent, player, pb, cards.Habitat.FOREST, "activate")
    assert pb.tucked_cards == 1


def test_predator_hunt_discards_large_bird():
    eng, birds = _engine(seed=7)
    hawk = _find(birds, "Cooper's Hawk")  # <75cm
    player = eng.state.me()
    pb = state.PlayedBird(bird=hawk)
    player.board[cards.Habitat.FOREST].append(pb)
    big = next(bird for bird in birds if bird.wingspan_cm and bird.wingspan_cm >= 75)
    eng.state.bird_deck.append(big)
    discard_before = len(eng.state.bird_discard)

    powers.dispatch_power(eng, _no_agent, player, pb, cards.Habitat.FOREST, "activate")
    assert pb.tucked_cards == 0
    assert len(eng.state.bird_discard) == discard_before + 1


def test_move_rightmost_moves_when_rightmost():
    eng, birds = _engine(seed=8)
    mover = _find(birds, "Bewick's Wren")
    other = next(
        bird
        for bird in birds
        if bird.name != mover.name and cards.Habitat.FOREST in bird.habitats
    )
    player = eng.state.me()
    pb_other = state.PlayedBird(bird=other)
    pb = state.PlayedBird(bird=mover)
    player.board[cards.Habitat.FOREST].extend([pb_other, pb])

    def agent[C: decisions.Choice](
        _engine: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        assert isinstance(decision, decisions.BirdPowerPickHabitatDecision)
        grass = [
            choice
            for choice in decision.choices
            if choice.habitat == cards.Habitat.GRASSLAND
        ]
        return typing.cast(C, grass[0] if grass else decision.choices[0])

    # Choose grassland if available among mover's legal habitats; else any.
    powers.dispatch_power(eng, agent, player, pb, cards.Habitat.FOREST, "activate")
    assert pb not in player.board[cards.Habitat.FOREST]
    assert any(
        pb in player.board[habitat]
        for habitat in (cards.Habitat.GRASSLAND, cards.Habitat.WETLAND)
    )


def test_move_rightmost_skipped_when_not_rightmost():
    eng, birds = _engine(seed=9)
    mover = _find(birds, "Bewick's Wren")
    other = next(
        bird
        for bird in birds
        if bird.name != mover.name and cards.Habitat.FOREST in bird.habitats
    )
    player = eng.state.me()
    pb = state.PlayedBird(bird=mover)
    pb_other = state.PlayedBird(bird=other)
    player.board[cards.Habitat.FOREST].extend([pb, pb_other])  # mover NOT rightmost

    powers.dispatch_power(eng, _no_agent, player, pb, cards.Habitat.FOREST, "activate")
    assert pb in player.board[cards.Habitat.FOREST]
    assert player.board[cards.Habitat.FOREST][0] is pb


def test_repeat_brown_replays_neighbor_power():
    eng, birds = _engine(seed=10)
    catbird = _find(birds, "Gray Catbird")
    player = eng.state.me()
    # Use a simple brown bird whose effect is easy to verify (GAIN_FOOD_BIRDFEEDER).
    target = next(
        bird
        for bird in birds
        if bird.color == cards.PowerColor.BROWN
        and any(
            effect.kind == cards.EffectKind.GAIN_FOOD_BIRDFEEDER
            for effect in bird.power.effects
        )
        and cards.Habitat.FOREST in bird.habitats
    )
    pb_target = state.PlayedBird(bird=target)
    pb_cat = state.PlayedBird(bird=catbird)
    player.board[cards.Habitat.FOREST].extend([pb_target, pb_cat])
    # Ensure feeder has the food the target wants.
    eff_food = next(
        effect.food
        for effect in target.power.effects
        if effect.kind == cards.EffectKind.GAIN_FOOD_BIRDFEEDER
    )
    assert eff_food is not None
    eng.state.birdfeeder.counts.zero()
    eng.state.birdfeeder.counts[eff_food] = 5
    for food in player.food:
        player.food[food] = 0

    powers.dispatch_power(
        eng, _no_agent, player, pb_cat, cards.Habitat.FOREST, "activate"
    )
    assert player.food[eff_food] >= 1


def test_pink_lay_eggs_reactor_fires_on_opponent_lay_eggs():
    eng, birds = _engine(seed=11)
    cowbird = _find(birds, "Brown-Headed Cowbird")  # pink, lay on bowl
    bowl_bird = next(
        bird
        for bird in birds
        if bird.nest == cards.NestType.BOWL
        and bird.egg_limit > 0
        and bird.name != cowbird.name
    )
    # P1 owns the cowbird and a bowl bird with room.
    p1 = eng.state.players[1]
    p1.board[cards.Habitat.GRASSLAND].append(state.PlayedBird(bird=cowbird))
    bowl_pb = state.PlayedBird(bird=bowl_bird)
    p1.board[cards.Habitat.GRASSLAND].append(bowl_pb)
    # P0 takes lay-eggs.
    eng.state.current_player = 0

    # Replace P1's agent with one that picks the bowl bird.
    def p1_agent[C: decisions.Choice](
        _engine: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        if isinstance(decision, decisions.LayEggDecision):
            return typing.cast(
                C,
                next(
                    choice
                    for choice in decision.choices
                    if not isinstance(choice, decisions.SkipChoice)
                ),
            )
        return decision.choices[0]

    eng.agents[1] = p1_agent
    # P0 lays eggs (active row empty so 2 eggs to nowhere — that's fine).
    actions.do_lay_eggs(eng, lambda _engine, decision: decision.choices[0])
    assert bowl_pb.eggs >= 1


def test_pink_predator_feeder_fires_when_predator_succeeds():
    eng, birds = _engine(seed=12)
    vulture = _find(birds, "Turkey Vulture")  # pink reactor
    hawk = _find(birds, "Cooper's Hawk")
    p0 = eng.state.players[0]
    p1 = eng.state.players[1]
    p0.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=hawk))
    pb_hawk = p0.board[cards.Habitat.FOREST][-1]
    p1.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=vulture))

    # Force a successful hunt: small bird on top of deck.
    small = next(bird for bird in birds if bird.wingspan_cm and bird.wingspan_cm < 30)
    eng.state.bird_deck.append(small)
    # Give feeder some food.
    eng.state.birdfeeder.counts.zero()
    eng.state.birdfeeder.choice_dice = 0  # controlled feeder: clear the choice face
    eng.state.birdfeeder.counts[cards.Food.SEED] = 3
    for food in p1.food:
        p1.food[food] = 0

    eng.state.current_player = 0
    powers.dispatch_power(eng, _no_agent, p0, pb_hawk, cards.Habitat.FOREST, "activate")
    assert pb_hawk.tucked_cards == 1
    assert p1.food[cards.Food.SEED] == 1
