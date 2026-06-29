"""Engine-level tests for the ``combine_gain_food`` regime.

When the flag is on, a run of single-food gains is collapsed into one
``GainFoodDecision`` whose options are multi-food subsets
(``decisions.FoodSubsetChoice``):

* the Forest multi-die feeder gain (``actions.combined_feeder_gain``), with the
  path-dependent reset folded in (partial subset → committed reroll → recurse);
* the ravens' two-wild supply gain and the opening setup keep
  (``actions.combined_supply_gain``).

``Birdfeeder.subset_options`` (the pure enumeration) is unit-tested in
``test_birdfeeder.py``; this file drives the engine wiring with scripted agents.
"""

from __future__ import annotations

import random
import typing

from wingspan import agents, cards, decisions, engine, state
from wingspan.engine import actions


def _new_game(seed: int = 0) -> state.GameState:
    birds, bonuses, goals = cards.load_all()
    return state.new_game(random.Random(seed), birds, bonuses, goals)


def _staged_feeder(gs: state.GameState, **faces: int) -> state.Birdfeeder:
    """Zero the feeder, then set the named single-food faces (and ``choice``)."""
    feeder = gs.birdfeeder
    feeder.counts.zero()
    feeder.choice_dice = faces.pop("choice", 0)
    for name, count in faces.items():
        feeder.counts[cards.Food[name.upper()]] = count
    return feeder


def _decline_resets[C: decisions.Choice](decision: decisions.Decision[C]) -> C | None:
    """Skip an offered reset; return ``None`` for anything else."""
    if isinstance(decision, decisions.ResetBirdfeederDecision):
        for choice in decision.choices:
            if isinstance(choice, decisions.SkipChoice):
                return typing.cast(C, choice)
    return None


# ---------------------------------------------------------------------------
# Feeder gain — the path-dependent combined decision


def test_combined_feeder_gain_n1_delegates_to_single_die_path() -> None:
    """N==1 is the unchanged algorithm: the agent sees a FoodChoice menu (not a
    FoodSubsetChoice), so the encoding stays byte-identical to the off path."""
    gs = _new_game()
    _staged_feeder(gs, fish=2, seed=1)  # two distinct faces — no reset offered
    player = gs.players[0]
    gs.current_player = 0
    seen = {"food_choice": False, "subset": False}

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        assert isinstance(decision, decisions.GainFoodDecision)
        for choice in decision.choices:
            if isinstance(choice, decisions.FoodSubsetChoice):
                seen["subset"] = True
            elif isinstance(choice, decisions.FoodChoice):
                seen["food_choice"] = True
        return typing.cast(C, decision.choices[0])

    eng = engine.Engine(gs, combine_gain_food=True)
    actions.combined_feeder_gain(eng, agent, player, 1)

    assert seen["food_choice"] and not seen["subset"]
    assert player.food.total() == 1


def test_combined_feeder_gain_offers_one_subset_decision() -> None:
    """A 2-die gain over a two-face feeder is ONE GainFoodDecision over
    FoodSubsetChoice options; the size-2 subset gains both foods at once."""
    gs = _new_game()
    _staged_feeder(gs, fish=1, seed=1)
    player = gs.players[0]
    gs.current_player = 0
    asked = {"n": 0}

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        assert isinstance(decision, decisions.GainFoodDecision)
        asked["n"] += 1
        for choice in decision.choices:
            if (
                isinstance(choice, decisions.FoodSubsetChoice)
                and choice.total_units() == 2
            ):
                return typing.cast(C, choice)
        raise AssertionError("size-2 subset not offered")

    eng = engine.Engine(gs, combine_gain_food=True)
    actions.combined_feeder_gain(eng, agent, player, 2)

    assert asked["n"] == 1  # a single combined decision, not two single-die asks
    assert player.food[cards.Food.FISH] == 1
    assert player.food[cards.Food.SEED] == 1


def test_combined_feeder_gain_choice_die_split_options() -> None:
    """A choice die can be taken as invertebrate or as seed — distinct subsets."""
    gs = _new_game()
    _staged_feeder(gs, fish=1, choice=1)  # one fish face + one choice die
    player = gs.players[0]
    gs.current_player = 0

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        skip = _decline_resets(decision)
        if skip is not None:
            return skip
        assert isinstance(decision, decisions.GainFoodDecision)
        # The full subset: fish + the choice die taken as seed.
        for choice in decision.choices:
            if (
                isinstance(choice, decisions.FoodSubsetChoice)
                and choice.plain[cards.Food.FISH] == 1
                and choice.choice_seed == 1
            ):
                return typing.cast(C, choice)
        raise AssertionError("fish + choice-as-seed subset not offered")

    eng = engine.Engine(gs, combine_gain_food=True)
    actions.combined_feeder_gain(eng, agent, player, 2)

    assert player.food[cards.Food.FISH] == 1
    assert player.food[cards.Food.SEED] == 1


def test_combined_feeder_gain_partial_triggers_reroll_and_recursion() -> None:
    """Choosing a partial subset commits to a reroll and recurses for the rest,
    so the player still ends up with exactly N food across the two decisions."""
    gs = _new_game(seed=7)
    _staged_feeder(gs, fish=1, seed=2)  # 3 dice, two faces
    player = gs.players[0]
    gs.current_player = 0
    gain_decisions = {"n": 0}

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        skip = _decline_resets(decision)
        if skip is not None:
            return skip
        assert isinstance(decision, decisions.GainFoodDecision)
        gain_decisions["n"] += 1
        # First (combined) decision: take the partial {fish} subset to force the
        # committed reroll; its leftover (seed×2) is a single face.
        for choice in decision.choices:
            if (
                isinstance(choice, decisions.FoodSubsetChoice)
                and choice.total_units() == 1
                and choice.plain[cards.Food.FISH] == 1
            ):
                return typing.cast(C, choice)
        # Recursion tail (N==1): the single-die FoodChoice menu — take the first.
        return typing.cast(C, decision.choices[0])

    eng = engine.Engine(gs, combine_gain_food=True)
    actions.combined_feeder_gain(eng, agent, player, 2)

    assert player.food.total() == 2  # exactly N across the reroll boundary
    assert gain_decisions["n"] >= 2  # the combined subset + the post-reroll gain


def test_do_gain_food_uses_combined_decision_when_flag_on() -> None:
    """The Forest action's base dice route through the combined builder when the
    flag is on: a 3-bird Forest row pulls 2 dice as one subset decision."""
    gs = _new_game(seed=4)
    player = gs.players[0]
    gs.current_player = player.id
    player.hand = []  # suppress the Forest trade-space convert
    forest_birds = [
        bird
        for bird in gs.bird_deck
        if cards.Habitat.FOREST in bird.habitats
        and bird.color != cards.PowerColor.BROWN
    ][:3]
    for bird in forest_birds:
        player.board[cards.Habitat.FOREST].append(state.PlayedBird(bird=bird))
    _staged_feeder(gs, fish=2, seed=2)  # two faces; 2 dice pulled
    saw_subset = {"yes": False}

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        skip = _decline_resets(decision)
        if skip is not None:
            return skip
        if isinstance(decision, decisions.GainFoodDecision):
            for choice in decision.choices:
                if isinstance(choice, decisions.FoodSubsetChoice):
                    saw_subset["yes"] = True
                    if choice.total_units() == 2:
                        return typing.cast(C, choice)
        return typing.cast(C, decision.choices[0])

    eng = engine.Engine(gs, agents=[agent, agent], combine_gain_food=True)
    actions.do_gain_food(eng, agent)

    assert saw_subset["yes"]
    assert player.food.total() == 2


# ---------------------------------------------------------------------------
# Supply gain — the ravens' two-wild gain (repeats allowed)


def test_combined_supply_gain_allows_repeats() -> None:
    """An open supply gain (capacity == n) offers multisets with repeats, so
    'gain 2 fish' is a single legal subset."""
    gs = _new_game()
    player = gs.players[0]
    gs.current_player = 0
    for food in cards.ALL_FOODS:
        player.food[food] = 0

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        assert isinstance(decision, decisions.GainFoodDecision)
        assert all(
            isinstance(choice, decisions.FoodSubsetChoice) and choice.total_units() == 2
            for choice in decision.choices
        )
        for choice in decision.choices:
            if (
                isinstance(choice, decisions.FoodSubsetChoice)
                and choice.plain[cards.Food.FISH] == 2
            ):
                return typing.cast(C, choice)
        raise AssertionError("2-fish subset not offered")

    eng = engine.Engine(gs, combine_gain_food=True)
    actions.combined_supply_gain(eng, agent, player, 2, per_food_capacity=2, prompt="x")

    assert player.food[cards.Food.FISH] == 2


def test_combined_supply_gain_setup_capacity_is_distinct() -> None:
    """Capacity 1 (the setup keep) offers only distinct-food subsets — no repeat
    of any food, since only one die of each is on offer."""
    gs = _new_game()
    player = gs.players[0]
    gs.current_player = 0
    for food in cards.ALL_FOODS:
        player.food[food] = 0

    captured: list[decisions.FoodSubsetChoice] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        assert isinstance(decision, decisions.GainFoodDecision)
        for choice in decision.choices:
            assert isinstance(choice, decisions.FoodSubsetChoice)
            captured.append(choice)
        return typing.cast(C, decision.choices[0])

    eng = engine.Engine(gs, combine_gain_food=True)
    actions.combined_supply_gain(eng, agent, player, 2, per_food_capacity=1, prompt="x")

    # 2-of-5 distinct foods → C(5,2) == 10 options, none with a repeat.
    assert len(captured) == 10
    assert all(all(count <= 1 for count in choice.plain.counts) for choice in captured)
    assert player.food.total() == 2


# ---------------------------------------------------------------------------
# Full-game integration


def test_full_game_completes_with_combine_gain_food() -> None:
    """A full random-agent game with combine_gain_food on completes — exercising
    the Forest combined gain (and setup, with split_setup_food) end to end."""
    gs = _new_game(seed=11)
    rand_rng = random.Random(11)
    agent_a = agents.random_agent(rand_rng)
    agent_b = agents.random_agent(rand_rng)

    eng = engine.Engine.play_one_game(
        gs,
        (agent_a, agent_b),
        split_setup_food=True,
        combine_gain_food=True,
    )

    assert eng.state.game_over
    for player in eng.state.players:
        assert player.final_score is not None
        assert player.food.total() >= 0
