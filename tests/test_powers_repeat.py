"""Tests for REPEAT_BROWN_OWN_HABITAT and REPEAT_PREDATOR_OWN_HABITAT.

Three birds in core carry these powers:

* Gray Catbird          -- "Repeat a brown power on another bird in this habitat."
* Northern Mockingbird  -- same
* Hooded Merganser      -- "Repeat 1 [predator] power in this habitat."

The implementation needs to:

* parse both phrasings into the right EffectKind,
* dispatch them as a follow-up activation of one other brown (or predator)
  bird in the same habitat,
* avoid infinite recursion when the chosen bird's own power is also a repeat.
"""
from __future__ import annotations

import random

from wingspan.actions import Choice, Decision, DecisionType
from wingspan.cards import (
    Bird, Effect, EffectKind, Food, Habitat, NestType, PowerColor,
    Power, load_all, parse_power,
)
from wingspan.game import Engine
from wingspan.state import PlayedBird, new_game


# ---------------------------------------------------------------------------
# Parser-level checks

def test_parse_repeat_brown():
    power = parse_power(
        PowerColor.BROWN, "Repeat a brown power on another bird in this habitat."
    )
    kinds = [e.kind for e in power.effects]
    assert EffectKind.REPEAT_BROWN_OWN_HABITAT in kinds
    assert EffectKind.UNIMPLEMENTED not in kinds


def test_parse_repeat_predator():
    power = parse_power(
        PowerColor.BROWN, "Repeat 1 [predator] power in this habitat."
    )
    kinds = [e.kind for e in power.effects]
    assert EffectKind.REPEAT_PREDATOR_OWN_HABITAT in kinds
    eff = next(e for e in power.effects if e.kind == EffectKind.REPEAT_PREDATOR_OWN_HABITAT)
    assert eff.amount == 1
    assert EffectKind.UNIMPLEMENTED not in kinds


def test_three_target_birds_implemented():
    birds, _, _ = load_all()
    targets = {"Gray Catbird", "Northern Mockingbird", "Hooded Merganser"}
    found = {b.name: b for b in birds if b.name in targets}
    assert set(found) == targets, f"missing: {targets - set(found)}"
    for name, bird in found.items():
        kinds = [e.kind for e in bird.power.effects]
        assert EffectKind.UNIMPLEMENTED not in kinds, (name, bird.raw_power_text)
        if name == "Hooded Merganser":
            assert EffectKind.REPEAT_PREDATOR_OWN_HABITAT in kinds
        else:
            assert EffectKind.REPEAT_BROWN_OWN_HABITAT in kinds


# ---------------------------------------------------------------------------
# Engine-level dispatch

def _fresh_engine(seed: int = 0) -> tuple[Engine, list[Bird]]:
    birds, bonuses, goals = load_all()
    rng = random.Random(seed)
    state = new_game(rng, birds, bonuses, goals)
    return Engine(state), birds


def _stub_brown_bird(name: str, kind: EffectKind, **eff_kwargs) -> Bird:
    """Build a minimal brown-power bird whose only effect is ``kind``."""
    power = Power(color=PowerColor.BROWN, effects=[Effect(kind, **eff_kwargs)])
    return Bird(
        id=-abs(hash(name)) % 100000,
        name=name,
        scientific_name="",
        color=PowerColor.BROWN,
        points=1,
        nest=NestType.BOWL,
        egg_limit=2,
        wingspan_cm=20,
        habitats=(Habitat.FOREST,),
        food_cost={},
        wild_food_cost=0,
        total_food_cost=0,
        flocking=False,
        predator=False,
        is_swift_start=False,
        raw_power_text="<stub>",
        power=power,
    )


def test_repeat_brown_copies_target_power():
    eng, _birds = _fresh_engine(0)
    p = eng.state.players[0]
    eng.state.current_player = p.id

    catbird = next(b for b in _birds if b.name == "Gray Catbird")
    # Build a brown target bird whose power lays 1 egg on itself.
    target_bird = _stub_brown_bird("EggLayer", EffectKind.LAY_EGG_ON_THIS, amount=1)

    catbird_pb = PlayedBird(bird=catbird)
    target_pb = PlayedBird(bird=target_bird)
    p.board[Habitat.FOREST] = [target_pb, catbird_pb]

    def pick_first_non_skip(_eng: Engine, d: Decision) -> Choice:
        assert d.type == DecisionType.BIRD_POWER_PICK_BIRD
        # First non-skip choice points at target_pb.
        assert d.choices[0].payload is target_pb
        return d.choices[0]

    eff = next(e for e in catbird.power.effects
               if e.kind == EffectKind.REPEAT_BROWN_OWN_HABITAT)
    eng._apply_effect(pick_first_non_skip, p, catbird_pb, Habitat.FOREST, eff, "activate")
    assert target_pb.eggs == 1


def test_repeat_brown_no_eligible_logs_and_returns():
    eng, _birds = _fresh_engine(0)
    p = eng.state.players[0]
    eng.state.current_player = p.id

    catbird = next(b for b in _birds if b.name == "Gray Catbird")
    catbird_pb = PlayedBird(bird=catbird)
    # No other birds in the habitat.
    p.board[Habitat.FOREST] = [catbird_pb]

    def no_decisions_expected(_eng: Engine, d: Decision) -> Choice:
        raise AssertionError(f"no decision expected, got {d.type}")

    eff = next(e for e in catbird.power.effects
               if e.kind == EffectKind.REPEAT_BROWN_OWN_HABITAT)
    before_log = len(eng.state.log)
    eng._apply_effect(no_decisions_expected, p, catbird_pb, Habitat.FOREST, eff, "activate")
    # No state changes, but the engine should log the skip.
    assert any("no eligible" in line for line in eng.state.log[before_log:])


def test_repeat_brown_skip_choice_is_a_no_op():
    eng, _birds = _fresh_engine(0)
    p = eng.state.players[0]
    eng.state.current_player = p.id

    catbird = next(b for b in _birds if b.name == "Gray Catbird")
    target_bird = _stub_brown_bird("EggLayer2", EffectKind.LAY_EGG_ON_THIS, amount=1)
    catbird_pb = PlayedBird(bird=catbird)
    target_pb = PlayedBird(bird=target_bird)
    p.board[Habitat.FOREST] = [target_pb, catbird_pb]

    def skip_agent(_eng: Engine, d: Decision) -> Choice:
        # The "skip" choice has payload=None.
        return next(c for c in d.choices if c.payload is None)

    eff = next(e for e in catbird.power.effects
               if e.kind == EffectKind.REPEAT_BROWN_OWN_HABITAT)
    eng._apply_effect(skip_agent, p, catbird_pb, Habitat.FOREST, eff, "activate")
    assert target_pb.eggs == 0


def test_repeat_brown_filters_repeat_recursion():
    """If the user picks another repeater, the copied power must not in turn
    fire its own repeat — guarded by filtering REPEAT_* effects out of the
    copied effect list."""
    eng, _birds = _fresh_engine(0)
    p = eng.state.players[0]
    eng.state.current_player = p.id

    catbird = next(b for b in _birds if b.name == "Gray Catbird")
    mockingbird = next(b for b in _birds if b.name == "Northern Mockingbird")
    catbird_pb = PlayedBird(bird=catbird)
    mocker_pb = PlayedBird(bird=mockingbird)
    p.board[Habitat.FOREST] = [mocker_pb, catbird_pb]

    decisions_made = []

    def pick_mocker(_eng: Engine, d: Decision) -> Choice:
        decisions_made.append(d.type)
        # Pick the mockingbird (its only effect is REPEAT_BROWN_OWN_HABITAT).
        return next(c for c in d.choices if c.payload is mocker_pb)

    eff = next(e for e in catbird.power.effects
               if e.kind == EffectKind.REPEAT_BROWN_OWN_HABITAT)
    eng._apply_effect(pick_mocker, p, catbird_pb, Habitat.FOREST, eff, "activate")

    # Exactly one BIRD_POWER_PICK_BIRD decision — the mockingbird's own repeat
    # was filtered out so the engine never re-prompted.
    assert decisions_made == [DecisionType.BIRD_POWER_PICK_BIRD]


def test_repeat_predator_only_lists_predators():
    eng, _birds = _fresh_engine(0)
    p = eng.state.players[0]
    eng.state.current_player = p.id

    merganser = next(b for b in _birds if b.name == "Hooded Merganser")

    # Build a brown non-predator and a brown predator.
    non_pred = _stub_brown_bird("Songbird", EffectKind.LAY_EGG_ON_THIS, amount=1)
    pred_bird = Bird(
        id=-12345,
        name="StubHawk",
        scientific_name="",
        color=PowerColor.BROWN,
        points=2,
        nest=NestType.BOWL,
        egg_limit=2,
        wingspan_cm=40,
        habitats=(Habitat.FOREST,),
        food_cost={Food.RODENT: 1},
        wild_food_cost=0,
        total_food_cost=1,
        flocking=False,
        predator=True,
        is_swift_start=False,
        raw_power_text="<stub-pred>",
        power=Power(color=PowerColor.BROWN, effects=[
            Effect(EffectKind.GAIN_FOOD_SUPPLY, amount=1, food=Food.RODENT)
        ]),
    )

    merg_pb = PlayedBird(bird=merganser)
    nonpred_pb = PlayedBird(bird=non_pred)
    pred_pb = PlayedBird(bird=pred_bird)
    p.board[Habitat.FOREST] = [nonpred_pb, pred_pb, merg_pb]

    seen_payloads = []

    def pick_predator(_eng: Engine, d: Decision) -> Choice:
        assert d.type == DecisionType.BIRD_POWER_PICK_BIRD
        seen_payloads.extend(c.payload for c in d.choices)
        return next(c for c in d.choices if c.payload is pred_pb)

    eff = next(e for e in merganser.power.effects
               if e.kind == EffectKind.REPEAT_PREDATOR_OWN_HABITAT)
    rodent_before = eng.state.food_supply[Food.RODENT]
    eng._apply_effect(pick_predator, p, merg_pb, Habitat.FOREST, eff, "activate")

    # Only the predator and "skip" should have been offered.
    assert nonpred_pb not in seen_payloads
    assert pred_pb in seen_payloads
    # Effect of the copied predator should have run: +1 rodent to the player.
    assert p.food[Food.RODENT] == 1
    assert eng.state.food_supply[Food.RODENT] == rodent_before - 1


def test_repeat_predator_no_eligible_is_skip():
    eng, _birds = _fresh_engine(0)
    p = eng.state.players[0]
    eng.state.current_player = p.id

    merganser = next(b for b in _birds if b.name == "Hooded Merganser")
    merg_pb = PlayedBird(bird=merganser)
    # Only the merganser in the habitat.
    p.board[Habitat.FOREST] = [merg_pb]

    def no_decisions(_eng: Engine, d: Decision) -> Choice:
        raise AssertionError(f"unexpected decision {d.type}")

    eff = next(e for e in merganser.power.effects
               if e.kind == EffectKind.REPEAT_PREDATOR_OWN_HABITAT)
    before = len(eng.state.log)
    eng._apply_effect(no_decisions, p, merg_pb, Habitat.FOREST, eff, "activate")
    assert any("no eligible" in line for line in eng.state.log[before:])
