"""Tests for the bird powers added to reach 100% coverage."""
from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards
from wingspan.actions import Choice, Decision, DecisionType
from wingspan.cards import EffectKind, Food, Habitat, NestType, PowerColor, parse_power
from wingspan.game import Engine
from wingspan.state import PlayedBird, new_game


def _find(birds, name):
    return next(b for b in birds if b.name == name)


def _engine(seed: int = 0) -> tuple[Engine, list]:
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(seed)
    state = new_game(rng, birds, bonuses, goals)
    return Engine(state, agents=[lambda *_: None, lambda *_: None]), birds


# --- Coverage sanity -----------------------------------------------------

def test_full_coverage_no_unimplemented():
    birds, _, _ = cards.load_all()
    bad = [b.name for b in birds
           if any(e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects)]
    assert bad == [], f"unimplemented: {bad}"


# --- Parser tests --------------------------------------------------------

def test_parser_draw_bonus_keep():
    p = parse_power(PowerColor.WHITE, "Draw 2 new bonus cards and keep 1.")
    assert any(e.kind == EffectKind.DRAW_BONUS_KEEP and e.amount == 2 and e.keep_count == 1
               for e in p.effects)


def test_parser_lay_egg_all_nest():
    p = parse_power(PowerColor.WHITE, "Lay 1 [egg] on each of your birds with a [cavity] nest.")
    eff = next(e for e in p.effects if e.kind == EffectKind.LAY_EGG_ALL_NEST)
    assert eff.nest == NestType.CAVITY
    assert eff.amount == 1


def test_parser_gain_all_food_feeder():
    p = parse_power(PowerColor.WHITE, "Gain all [fish] that are in the birdfeeder.")
    eff = next(e for e in p.effects if e.kind == EffectKind.GAIN_ALL_FOOD_FEEDER)
    assert eff.food == Food.FISH


def test_parser_tuck_from_deck_paid():
    p = parse_power(PowerColor.BROWN,
                    "Discard 1 [seed] to tuck 2 [card] from the deck behind this bird.")
    eff = next(e for e in p.effects if e.kind == EffectKind.TUCK_FROM_DECK_PAID)
    assert eff.food == Food.SEED and eff.amount == 2


def test_parser_predator_hunt():
    p = parse_power(PowerColor.BROWN,
                    "Look at a [card] from the deck. If less than 75cm, tuck it behind this bird. If not, discard it.")
    eff = next(e for e in p.effects if e.kind == EffectKind.PREDATOR_HUNT)
    assert eff.max_wingspan_cm == 75


def test_parser_move_rightmost():
    p = parse_power(PowerColor.BROWN,
                    "If this bird is to the right of all other birds in its habitat, move it to another habitat.")
    assert any(e.kind == EffectKind.MOVE_BIRD_IF_RIGHTMOST for e in p.effects)


def test_parser_repeat_brown():
    p = parse_power(PowerColor.BROWN, "Repeat a brown power on another bird in this habitat.")
    assert any(e.kind == EffectKind.REPEAT_BROWN_POWER for e in p.effects)


def test_parser_repeat_predator():
    p = parse_power(PowerColor.BROWN, "Repeat 1 [predator] power in this habitat.")
    assert any(e.kind == EffectKind.REPEAT_PREDATOR_POWER for e in p.effects)


def test_parser_pink_lay_egg_on_nest():
    p = parse_power(PowerColor.PINK,
                    "When another player takes the 'lay eggs' action, lay 1 [egg] on a bird with a [bowl] nest.")
    eff = next(e for e in p.effects if e.kind == EffectKind.PINK_LAY_EGG_ON_NEST)
    assert eff.nest == NestType.BOWL


def test_parser_pink_predator_feeder():
    p = parse_power(PowerColor.PINK,
                    "When another player's [predator] succeeds, gain 1 [die] from the birdfeeder.")
    assert any(e.kind == EffectKind.PINK_PREDATOR_FEEDER for e in p.effects)


# --- Engine behavior tests ----------------------------------------------

def test_draw_bonus_keep_keeps_one_discards_one():
    eng, birds = _engine(seed=1)
    bird = _find(birds, "Atlantic Puffin")
    p = eng.state.me()
    pb = PlayedBird(bird=bird)
    p.board[Habitat.WETLAND].append(pb)

    # Force a known top of bonus deck.
    eng.state.bonus_deck = list(eng.state.bonus_deck[:5])  # at least 2
    before_have = len(p.bonus_cards)
    before_deck = len(eng.state.bonus_deck)

    def agent(_e, d):
        assert d.type == DecisionType.BIRD_POWER_PICK_BIRD
        return d.choices[0]

    eng._dispatch_power(agent, p, pb, Habitat.WETLAND, "play")
    assert len(p.bonus_cards) == before_have + 1
    assert len(eng.state.bonus_deck) == before_deck - 2
    assert len(eng.state.bonus_discard) >= 1


def test_lay_egg_all_nest_adds_one_to_each_matching():
    eng, birds = _engine(seed=2)
    bobolink = _find(birds, "Bobolink")  # ground
    p = eng.state.me()
    # 3 birds with ground nests + 1 non-ground
    ground = [b for b in birds if b.nest == NestType.GROUND and b is not bobolink][:3]
    non_ground = next(b for b in birds if b.nest != NestType.GROUND and b.egg_limit > 0)
    targets = [PlayedBird(bird=b) for b in ground]
    other = PlayedBird(bird=non_ground)
    pb = PlayedBird(bird=bobolink)
    p.board[Habitat.GRASSLAND].extend(targets + [other, pb])

    eng._dispatch_power(lambda *_: None, p, pb, Habitat.GRASSLAND, "play")

    for t in targets:
        assert t.eggs == 1
    assert other.eggs == 0


def test_gain_all_food_feeder_drains_one_face():
    eng, birds = _engine(seed=3)
    bird = _find(birds, "Bald Eagle")  # fish
    p = eng.state.me()
    pb = PlayedBird(bird=bird)
    p.board[Habitat.WETLAND].append(pb)
    eng.state.birdfeeder.counts = {f: 0 for f in eng.state.birdfeeder.counts}
    eng.state.birdfeeder.counts[Food.FISH] = 3
    eng.state.birdfeeder.counts[Food.SEED] = 2
    for f in p.food: p.food[f] = 0

    eng._dispatch_power(lambda *_: None, p, pb, Habitat.WETLAND, "play")
    assert eng.state.birdfeeder.counts[Food.FISH] == 0
    assert p.food[Food.FISH] == 3
    assert eng.state.birdfeeder.counts[Food.SEED] == 2  # untouched


def test_tuck_from_deck_paid_spends_food_and_tucks():
    eng, birds = _engine(seed=4)
    bird = _find(birds, "Sandhill Crane")  # discard 1 seed -> tuck 2
    p = eng.state.me()
    for f in p.food: p.food[f] = 0
    p.food[Food.SEED] = 1
    pb = PlayedBird(bird=bird)
    p.board[Habitat.GRASSLAND].append(pb)
    deck_before = len(eng.state.bird_deck)

    def agent(_e, d):
        return next(c for c in d.choices if c.payload == "pay")

    eng._dispatch_power(agent, p, pb, Habitat.GRASSLAND, "activate")
    assert p.food[Food.SEED] == 0
    assert pb.tucked_cards == 2
    assert len(eng.state.bird_deck) == deck_before - 2


def test_tuck_from_deck_paid_skip_when_no_food():
    eng, birds = _engine(seed=5)
    bird = _find(birds, "Canada Goose")
    p = eng.state.me()
    for f in p.food: p.food[f] = 0
    pb = PlayedBird(bird=bird)
    p.board[Habitat.GRASSLAND].append(pb)
    deck_before = len(eng.state.bird_deck)

    eng._dispatch_power(lambda *_: None, p, pb, Habitat.GRASSLAND, "activate")
    assert pb.tucked_cards == 0
    assert len(eng.state.bird_deck) == deck_before


def test_predator_hunt_tucks_small_bird():
    eng, birds = _engine(seed=6)
    hawk = _find(birds, "Cooper's Hawk")  # <75cm
    p = eng.state.me()
    pb = PlayedBird(bird=hawk)
    p.board[Habitat.FOREST].append(pb)
    # Force a small bird on top of the deck.
    small = next(b for b in birds if b.wingspan_cm and b.wingspan_cm < 30 and b is not hawk)
    eng.state.bird_deck.append(small)

    eng._dispatch_power(lambda *_: None, p, pb, Habitat.FOREST, "activate")
    assert pb.tucked_cards == 1


def test_predator_hunt_discards_large_bird():
    eng, birds = _engine(seed=7)
    hawk = _find(birds, "Cooper's Hawk")  # <75cm
    p = eng.state.me()
    pb = PlayedBird(bird=hawk)
    p.board[Habitat.FOREST].append(pb)
    big = next(b for b in birds if b.wingspan_cm and b.wingspan_cm >= 75)
    eng.state.bird_deck.append(big)
    discard_before = len(eng.state.bird_discard)

    eng._dispatch_power(lambda *_: None, p, pb, Habitat.FOREST, "activate")
    assert pb.tucked_cards == 0
    assert len(eng.state.bird_discard) == discard_before + 1


def test_move_rightmost_moves_when_rightmost():
    eng, birds = _engine(seed=8)
    mover = _find(birds, "Bewick's Wren")
    other = next(b for b in birds if b.name != mover.name and Habitat.FOREST in b.habitats)
    p = eng.state.me()
    pb_other = PlayedBird(bird=other)
    pb = PlayedBird(bird=mover)
    p.board[Habitat.FOREST].extend([pb_other, pb])

    def agent(_e, d):
        assert d.type == DecisionType.BIRD_POWER_PICK_HABITAT
        return next(c for c in d.choices if c.payload == Habitat.GRASSLAND) \
            if any(c.payload == Habitat.GRASSLAND for c in d.choices) else d.choices[0]

    # Choose grassland if available among mover's legal habitats; else any.
    eng._dispatch_power(agent, p, pb, Habitat.FOREST, "activate")
    assert pb not in p.board[Habitat.FOREST]
    assert any(pb in p.board[h] for h in (Habitat.GRASSLAND, Habitat.WETLAND))


def test_move_rightmost_skipped_when_not_rightmost():
    eng, birds = _engine(seed=9)
    mover = _find(birds, "Bewick's Wren")
    other = next(b for b in birds if b.name != mover.name and Habitat.FOREST in b.habitats)
    p = eng.state.me()
    pb = PlayedBird(bird=mover)
    pb_other = PlayedBird(bird=other)
    p.board[Habitat.FOREST].extend([pb, pb_other])  # mover NOT rightmost

    eng._dispatch_power(lambda *_: None, p, pb, Habitat.FOREST, "activate")
    assert pb in p.board[Habitat.FOREST]
    assert p.board[Habitat.FOREST][0] is pb


def test_repeat_brown_replays_neighbor_power():
    eng, birds = _engine(seed=10)
    catbird = _find(birds, "Gray Catbird")
    p = eng.state.me()
    # Use a simple brown bird whose effect is easy to verify (GAIN_FOOD_BIRDFEEDER).
    target = next(b for b in birds
                  if b.color == PowerColor.BROWN
                  and any(e.kind == EffectKind.GAIN_FOOD_BIRDFEEDER for e in b.power.effects)
                  and Habitat.FOREST in b.habitats)
    pb_target = PlayedBird(bird=target)
    pb_cat = PlayedBird(bird=catbird)
    p.board[Habitat.FOREST].extend([pb_target, pb_cat])
    # Ensure feeder has the food the target wants.
    eff_food = next(e.food for e in target.power.effects
                    if e.kind == EffectKind.GAIN_FOOD_BIRDFEEDER)
    eng.state.birdfeeder.counts = {f: 0 for f in eng.state.birdfeeder.counts}
    eng.state.birdfeeder.counts[eff_food] = 5
    for f in p.food: p.food[f] = 0

    eng._dispatch_power(lambda *_: None, p, pb_cat, Habitat.FOREST, "activate")
    assert p.food[eff_food] >= 1


def test_pink_lay_eggs_reactor_fires_on_opponent_lay_eggs():
    eng, birds = _engine(seed=11)
    cowbird = _find(birds, "Brown-Headed Cowbird")  # pink, lay on bowl
    bowl_bird = next(b for b in birds
                     if b.nest == NestType.BOWL and b.egg_limit > 0
                     and b.name != cowbird.name)
    # P1 owns the cowbird and a bowl bird with room.
    p1 = eng.state.players[1]
    p1.board[Habitat.GRASSLAND].append(PlayedBird(bird=cowbird))
    bowl_pb = PlayedBird(bird=bowl_bird)
    p1.board[Habitat.GRASSLAND].append(bowl_pb)
    # P0 takes lay-eggs.
    eng.state.current_player = 0
    p0 = eng.state.me()
    # Replace P1's agent with one that picks the bowl bird.
    def p1_agent(_e, d):
        if d.type == DecisionType.LAY_EGG_PICK_BIRD:
            return next(c for c in d.choices if c.payload != None)
        return d.choices[0]
    eng.agents[1] = p1_agent
    # P0 lays eggs (active row empty so 2 eggs to nowhere — that's fine).
    eng._do_lay_eggs(lambda _e, d: d.choices[0], Habitat.GRASSLAND)
    assert bowl_pb.eggs >= 1


def test_pink_predator_feeder_fires_when_predator_succeeds():
    eng, birds = _engine(seed=12)
    vulture = _find(birds, "Turkey Vulture")  # pink reactor
    hawk = _find(birds, "Cooper's Hawk")
    p0 = eng.state.players[0]
    p1 = eng.state.players[1]
    p0.board[Habitat.FOREST].append(PlayedBird(bird=hawk))
    pb_hawk = p0.board[Habitat.FOREST][-1]
    p1.board[Habitat.FOREST].append(PlayedBird(bird=vulture))

    # Force a successful hunt: small bird on top of deck.
    small = next(b for b in birds if b.wingspan_cm and b.wingspan_cm < 30)
    eng.state.bird_deck.append(small)
    # Give feeder some food.
    eng.state.birdfeeder.counts = {f: 0 for f in eng.state.birdfeeder.counts}
    eng.state.birdfeeder.counts[Food.SEED] = 3
    for f in p1.food: p1.food[f] = 0

    eng.state.current_player = 0
    eng._dispatch_power(lambda *_: None, p0, pb_hawk, Habitat.FOREST, "activate")
    assert pb_hawk.tucked_cards == 1
    assert p1.food[Food.SEED] == 1
