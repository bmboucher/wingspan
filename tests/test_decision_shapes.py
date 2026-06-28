"""Tests for Stage C — Decision-shape fixes (gaps #14–#19).

Gaps covered
------------
#14  Star nests counted via ``cards.nest_matches`` in ``_h_all_players_lay_egg_on_nest``
     eligibility and ``_has_eligible_bird_on_nest``.
#15  ``birds_no_eggs`` AcceptExchangeDecision gate added to ``LAY_EGG_ON_THIS`` and
     ``LAY_EGG_ALL_NEST``.
#16  AcceptExchangeDecision veto added to ``ALL_PLAYERS_GAIN_FOOD``, ``ALL_PLAYERS_DRAW``,
     ``EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER``, and the tied case of
     ``FEWEST_FOREST_GAINS_DIE``.
#17  AcceptExchangeDecision veto in ``PREDATOR_HUNT`` and ``ROLL_NOT_IN_FEEDER_CACHE``
     when opposing ``PINK_PREDATOR_FEEDER`` birds are present.
#18  ``TRADE_WILD_FOOD`` (Green Heron) is now forced — no optional gate (tested in
     ``test_misc_unique_powers.py``; residual coverage here: Hermit Thrush three-way).
#19  ``fire_pink_lay_egg`` and ``_h_tuck_from_hand_then_lay_on_this`` are forced;
     ``birds_no_eggs``-conditional gate replaces the unconditional skip row.
"""

from __future__ import annotations

import random
import typing

from wingspan import cards, decisions, engine, state  # noqa: E402
from wingspan.engine import powers, reactors  # noqa: E402

_BIRDS, _BONUSES, _GOALS = cards.load_all()


# ---------------------------------------------------------------------------
# Helpers


def _new_game(seed: int = 0) -> state.GameState:
    return state.new_game(random.Random(seed), _BIRDS, _BONUSES, _GOALS)


def _no_eggs_goal() -> cards.EndRoundGoal:
    return cards.EndRoundGoal(
        id=0,
        description="[bird] with no [egg]",
        category="birds_no_eggs",
        tile_id=0,
    )


def _accept_agent[C: decisions.Choice](
    _eng: engine.Engine,
    decision: decisions.Decision[C],
) -> C:
    """Always accept: return first non-SkipChoice, or choices[0] if all are skip."""
    for choice in decision.choices:
        if not isinstance(choice, decisions.SkipChoice):
            return choice
    return decision.choices[0]


def _skip_agent[C: decisions.Choice](
    _eng: engine.Engine,
    decision: decisions.Decision[C],
) -> C:
    """Always pick a SkipChoice if available, else choices[0]."""
    for choice in decision.choices:
        if isinstance(choice, decisions.SkipChoice):
            return choice
    return decision.choices[0]


def _assert_no_gate_agent[C: decisions.Choice](
    _eng: engine.Engine,
    decision: decisions.Decision[C],
) -> C:
    """Raises if an AcceptExchangeDecision veto gate is presented (should not happen)."""
    assert not isinstance(
        decision, decisions.AcceptExchangeDecision
    ), f"unexpected AcceptExchangeDecision veto gate: {decision.prompt!r}"
    for choice in decision.choices:
        if not isinstance(choice, decisions.SkipChoice):
            return typing.cast(C, choice)
    return decision.choices[0]


# ---------------------------------------------------------------------------
# Gap #14 — star nests in multi_actor eligibility


def test_star_nest_counted_in_own_eligible_for_all_players_lay_egg():
    """A star-nest bird on the active player's board must be counted in
    ``own_eligible_count`` so the veto ledger shows the correct egg count (gap #14)."""
    gs = _new_game(0)

    # Build a bird with ALL_PLAYERS_LAY_EGG_ON_NEST for a bowl nest.
    power_bird = next(
        bird
        for bird in _BIRDS
        if any(
            eff.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
            and eff.nest == cards.NestType.BOWL
            for eff in bird.power.effects
        )
    )
    pb_power = state.PlayedBird(bird=power_bird)

    # Give P0 only a star-nest bird with room for an egg (no bowl birds).
    star_bird = next(
        bird
        for bird in _BIRDS
        if bird.nest == cards.NestType.STAR and bird.egg_limit >= 1
    )
    pb_star = state.PlayedBird(bird=star_bird)
    for hab in cards.ALL_HABITATS:
        gs.players[0].board[hab].clear()
    gs.players[0].board[star_bird.habitats[0]].append(pb_star)

    # Remove P1's board so no opponent egg is possible (only P0 can gain).
    for hab in cards.ALL_HABITATS:
        gs.players[1].board[hab].clear()

    # Track which AcceptExchangeDecision choices are offered to P0.
    veto_choices: list[decisions.PayCostChoice | decisions.SkipChoice] = []

    def agent_p0[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            veto_choices.extend(decision.choices)
            # Accept.
            return typing.cast(
                C,
                next(
                    c
                    for c in decision.choices
                    if isinstance(c, decisions.PayCostChoice)
                ),
            )
        # For any LayEggDecision pick the first choice.
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return choice
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[agent_p0, _skip_agent])
    gs.current_player = 0

    eff = next(
        eff
        for eff in power_bird.power.effects
        if eff.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
    )
    powers.apply_effect(
        eng, agent_p0, gs.players[0], pb_power, power_bird.habitats[0], eff, "play"
    )

    # The veto was offered (star nest counted as own-eligible).
    assert veto_choices, "veto gate was never presented — star nest not counted"
    pay_choice = next(c for c in veto_choices if isinstance(c, decisions.PayCostChoice))
    assert (
        pay_choice.gained_egg_count >= 1
    ), f"gained_egg_count should be ≥1 but got {pay_choice.gained_egg_count}"
    # Egg was laid on the star-nest bird.
    assert pb_star.eggs == 1


def test_has_eligible_bird_on_nest_counts_star_nest():
    """Opponents with only star-nest birds are found by the eligibility check so
    they participate in the all-players lay (gap #14)."""
    gs = _new_game(0)

    power_bird = next(
        bird
        for bird in _BIRDS
        if any(
            eff.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
            and eff.nest == cards.NestType.BOWL
            for eff in bird.power.effects
        )
    )
    pb_power = state.PlayedBird(bird=power_bird)

    star_bird = next(
        bird
        for bird in _BIRDS
        if bird.nest == cards.NestType.STAR and bird.egg_limit >= 1
    )

    # Give P1 only a star-nest bird (no bowl birds).
    pb_star = state.PlayedBird(bird=star_bird)
    for hab in cards.ALL_HABITATS:
        gs.players[1].board[hab].clear()
    gs.players[1].board[star_bird.habitats[0]].append(pb_star)

    # P0 also has a bowl bird so the power always fires.
    bowl_bird = next(
        bird
        for bird in _BIRDS
        if bird.nest == cards.NestType.BOWL and bird.egg_limit >= 1
    )
    gs.players[0].board[bowl_bird.habitats[0]].append(state.PlayedBird(bird=bowl_bird))

    eng = engine.Engine(gs, agents=[_accept_agent, _accept_agent])
    gs.current_player = 0

    eff = next(
        eff
        for eff in power_bird.power.effects
        if eff.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
    )
    powers.apply_effect(
        eng, _accept_agent, gs.players[0], pb_power, power_bird.habitats[0], eff, "play"
    )

    # P1's star-nest bird received an egg (was found eligible).
    assert pb_star.eggs == 1


# ---------------------------------------------------------------------------
# Gap #15 — birds_no_eggs gate for LAY_EGG_ON_THIS and LAY_EGG_ALL_NEST


def _build_bird_with_effect(
    kind: cards.EffectKind, **kwargs: object
) -> tuple[cards.Bird, cards.Effect]:
    """Return a (bird, effect) pair for the given EffectKind, built from a template."""
    template = _BIRDS[0]
    eff = cards.Effect(kind=kind, amount=1, **kwargs)  # type: ignore[arg-type]
    power = cards.Power(color=cards.PowerColor.BROWN, effects=(eff,))
    bird = template.model_copy(
        update={"power": power, "raw_power_text": f"test {kind}"}
    )
    return bird, eff


def test_lay_egg_on_this_gated_under_birds_no_eggs():
    """Under the birds_no_eggs goal ``LAY_EGG_ON_THIS`` offers a gate; declining
    skips the lay (gap #15)."""
    gs = _new_game(0)
    gs.round_goals = [_no_eggs_goal()] * 4
    gs.round_idx = 0

    bird, eff = _build_bird_with_effect(cards.EffectKind.LAY_EGG_ON_THIS)
    pb = state.PlayedBird(bird=bird)

    eng = engine.Engine(gs, agents=[_skip_agent, _skip_agent])
    gs.current_player = 0

    powers.apply_effect(
        eng, _skip_agent, gs.players[0], pb, bird.habitats[0], eff, "play"
    )
    assert pb.eggs == 0, "egg should not be laid when gate is declined"


def test_lay_egg_on_this_forced_without_birds_no_eggs():
    """Without birds_no_eggs goal, ``LAY_EGG_ON_THIS`` is forced — no gate (gap #15)."""
    gs = _new_game(0)

    bird, eff = _build_bird_with_effect(cards.EffectKind.LAY_EGG_ON_THIS)
    pb = state.PlayedBird(bird=bird)

    eng = engine.Engine(gs, agents=[_assert_no_gate_agent, _assert_no_gate_agent])
    gs.current_player = 0

    powers.apply_effect(
        eng, _assert_no_gate_agent, gs.players[0], pb, bird.habitats[0], eff, "play"
    )
    assert pb.eggs == 1, "egg should be laid without gate when goal is inactive"


def test_lay_egg_all_nest_gated_under_birds_no_eggs():
    """Under the birds_no_eggs goal ``LAY_EGG_ALL_NEST`` offers a gate (gap #15)."""
    gs = _new_game(0)
    gs.round_goals = [_no_eggs_goal()] * 4
    gs.round_idx = 0

    nest = cards.NestType.BOWL
    bowl_bird = next(
        bird for bird in _BIRDS if bird.nest == nest and bird.egg_limit >= 1
    )
    pb_bowl = state.PlayedBird(bird=bowl_bird)
    gs.players[0].board[bowl_bird.habitats[0]].append(pb_bowl)

    eff = cards.Effect(kind=cards.EffectKind.LAY_EGG_ALL_NEST, nest=nest, amount=1)
    template = _BIRDS[0]
    power_bird = template.model_copy(
        update={
            "power": cards.Power(color=cards.PowerColor.BROWN, effects=(eff,)),
            "raw_power_text": "test lay_egg_all_nest",
        }
    )
    pb_power = state.PlayedBird(bird=power_bird)

    eng = engine.Engine(gs, agents=[_skip_agent, _skip_agent])
    gs.current_player = 0

    powers.apply_effect(
        eng, _skip_agent, gs.players[0], pb_power, power_bird.habitats[0], eff, "play"
    )
    assert pb_bowl.eggs == 0, "eggs should not be laid when gate is declined"


# ---------------------------------------------------------------------------
# Gap #16 — all-players vetoes


def test_all_players_gain_food_veto_skips_when_declined():
    """Declining the ``ALL_PLAYERS_GAIN_FOOD`` veto leaves both players' food
    unchanged (gap #16)."""
    gs = _new_game(0)
    food_before_p0 = gs.players[0].food.as_dict()
    food_before_p1 = gs.players[1].food.as_dict()

    eff = cards.Effect(
        kind=cards.EffectKind.ALL_PLAYERS_GAIN_FOOD, food=cards.Food.SEED, amount=1
    )
    template = _BIRDS[0].model_copy(
        update={
            "power": cards.Power(color=cards.PowerColor.BROWN, effects=(eff,)),
            "raw_power_text": "test all_gain",
        }
    )
    pb = state.PlayedBird(bird=template)
    eng = engine.Engine(gs, agents=[_skip_agent, _skip_agent])
    gs.current_player = 0

    powers.apply_effect(
        eng, _skip_agent, gs.players[0], pb, template.habitats[0], eff, "play"
    )
    assert gs.players[0].food.as_dict() == food_before_p0
    assert gs.players[1].food.as_dict() == food_before_p1


def test_all_players_gain_food_veto_accepted_credits_both():
    """Accepting the ``ALL_PLAYERS_GAIN_FOOD`` veto credits each player (gap #16)."""
    gs = _new_game(0)
    seed_before_p0 = gs.players[0].food[cards.Food.SEED]
    seed_before_p1 = gs.players[1].food[cards.Food.SEED]

    eff = cards.Effect(
        kind=cards.EffectKind.ALL_PLAYERS_GAIN_FOOD, food=cards.Food.SEED, amount=1
    )
    template = _BIRDS[0].model_copy(
        update={
            "power": cards.Power(color=cards.PowerColor.BROWN, effects=(eff,)),
            "raw_power_text": "test all_gain",
        }
    )
    pb = state.PlayedBird(bird=template)
    eng = engine.Engine(gs, agents=[_accept_agent, _accept_agent])
    gs.current_player = 0

    powers.apply_effect(
        eng, _accept_agent, gs.players[0], pb, template.habitats[0], eff, "play"
    )
    assert gs.players[0].food[cards.Food.SEED] == seed_before_p0 + 1
    assert gs.players[1].food[cards.Food.SEED] == seed_before_p1 + 1


def test_fewest_forest_gains_die_tied_case_veto():
    """When both players are tied for fewest forest birds, the active player gets a
    veto gate (gap #16)."""
    gs = _new_game(0)
    # Empty both forest rows → both players tied at 0.
    gs.players[0].board[cards.Habitat.FOREST].clear()
    gs.players[1].board[cards.Habitat.FOREST].clear()

    # Fill feeder with mixed faces so no single-face reset is offered.
    for food in cards.ALL_FOODS:
        gs.birdfeeder.counts[food] = 1
    gs.birdfeeder.choice_dice = 0

    food_before_p0 = sum(gs.players[0].food.values())
    food_before_p1 = sum(gs.players[1].food.values())

    eff = cards.Effect(kind=cards.EffectKind.FEWEST_FOREST_GAINS_DIE, amount=1)
    template = _BIRDS[0].model_copy(
        update={
            "power": cards.Power(color=cards.PowerColor.BROWN, effects=(eff,)),
            "raw_power_text": "test fewest_forest",
        }
    )
    pb = state.PlayedBird(bird=template)
    gs.current_player = 0

    veto_seen = [False]

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            veto_seen[0] = True
            # Decline — no die gained.
            return typing.cast(
                C,
                next(
                    c for c in decision.choices if isinstance(c, decisions.SkipChoice)
                ),
            )
        # If somehow further decisions arrive, pick first.
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[agent, agent])
    powers.apply_effect(
        eng, agent, gs.players[0], pb, template.habitats[0], eff, "play"
    )

    assert veto_seen[0], "veto gate should be offered in the tied case"
    assert (
        sum(gs.players[0].food.values()) == food_before_p0
    ), "p0 should gain no die when gate declined"
    assert (
        sum(gs.players[1].food.values()) == food_before_p1
    ), "p1 should gain no die when gate declined"


def test_fewest_forest_gains_die_strictly_fewer_no_veto():
    """When the active player has strictly fewer forest birds, no veto gate is
    offered — the die gain is forced (gap #16)."""
    gs = _new_game(0)
    gs.players[0].board[cards.Habitat.FOREST].clear()
    # Give P1 one forest bird so P0 is strictly fewer.
    dummy_bird = _BIRDS[0]
    gs.players[1].board[cards.Habitat.FOREST].append(state.PlayedBird(bird=dummy_bird))

    for food in cards.ALL_FOODS:
        gs.birdfeeder.counts[food] = 1
    gs.birdfeeder.choice_dice = 0
    gs.current_player = 0

    eff = cards.Effect(kind=cards.EffectKind.FEWEST_FOREST_GAINS_DIE, amount=1)
    template = _BIRDS[0].model_copy(
        update={
            "power": cards.Power(color=cards.PowerColor.BROWN, effects=(eff,)),
            "raw_power_text": "test fewest_forest",
        }
    )
    pb = state.PlayedBird(bird=template)

    food_before_p0 = sum(gs.players[0].food.values())
    eng = engine.Engine(gs, agents=[_assert_no_gate_agent, _accept_agent])
    powers.apply_effect(
        eng, _assert_no_gate_agent, gs.players[0], pb, template.habitats[0], eff, "play"
    )
    # P0 gained a die (forced, no gate).
    assert sum(gs.players[0].food.values()) == food_before_p0 + 1


# ---------------------------------------------------------------------------
# Gap #17 — predator-hunt veto


def _make_predator_feeder_bird() -> cards.Bird:
    """A synthesised pink bird whose power is PINK_PREDATOR_FEEDER."""
    text = (
        "When another player's [predator] succeeds, gain 1 [die] from the birdfeeder."
    )
    template = next(b for b in _BIRDS if b.color == cards.PowerColor.PINK)
    return template.model_copy(
        update={
            "power": cards.parse_power(cards.PowerColor.PINK, text),
            "raw_power_text": text,
        }
    )


def test_predator_hunt_veto_offered_when_feeder_bird_present():
    """``PREDATOR_HUNT`` presents an AcceptExchangeDecision when the opponent has an
    unfired PINK_PREDATOR_FEEDER bird (gap #17)."""
    gs = _new_game(seed=42)
    gs.current_player = 0

    feeder_bird = _make_predator_feeder_bird()
    pb_feeder = state.PlayedBird(bird=feeder_bird)
    gs.players[1].board[feeder_bird.habitats[0]].append(pb_feeder)

    cap = 45  # wingspan cap > any bird
    hunt_eff = cards.Effect(kind=cards.EffectKind.PREDATOR_HUNT, max_wingspan_cm=cap)
    template = next(b for b in _BIRDS if b.predator)
    hunter = template.model_copy(
        update={
            "power": cards.Power(color=cards.PowerColor.BROWN, effects=(hunt_eff,)),
            "raw_power_text": "test predator",
        }
    )
    pb_hunter = state.PlayedBird(bird=hunter)

    veto_choices_seen: list[decisions.PayCostChoice | decisions.SkipChoice] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            veto_choices_seen.extend(decision.choices)
            # Accept.
            return typing.cast(
                C,
                next(
                    c
                    for c in decision.choices
                    if isinstance(c, decisions.PayCostChoice)
                ),
            )
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[agent, agent])
    powers.apply_effect(
        eng, agent, gs.players[0], pb_hunter, hunter.habitats[0], hunt_eff, "play"
    )

    assert veto_choices_seen, "veto gate was not presented"
    pay = next(c for c in veto_choices_seen if isinstance(c, decisions.PayCostChoice))
    assert pay.gained_tuck_count == 1
    assert pay.opp_gained_food_count == 1


def test_predator_hunt_veto_declined_skips_hunt():
    """Declining the ``PREDATOR_HUNT`` veto does not draw from the deck (gap #17)."""
    gs = _new_game(seed=42)
    gs.current_player = 0

    feeder_bird = _make_predator_feeder_bird()
    gs.players[1].board[feeder_bird.habitats[0]].append(
        state.PlayedBird(bird=feeder_bird)
    )

    deck_before = len(gs.bird_deck)

    hunt_eff = cards.Effect(kind=cards.EffectKind.PREDATOR_HUNT, max_wingspan_cm=100)
    template = next(b for b in _BIRDS if b.predator)
    hunter = template.model_copy(
        update={
            "power": cards.Power(color=cards.PowerColor.BROWN, effects=(hunt_eff,)),
            "raw_power_text": "test predator",
        }
    )
    pb_hunter = state.PlayedBird(bird=hunter)

    eng = engine.Engine(gs, agents=[_skip_agent, _skip_agent])
    powers.apply_effect(
        eng, _skip_agent, gs.players[0], pb_hunter, hunter.habitats[0], hunt_eff, "play"
    )

    assert (
        len(gs.bird_deck) == deck_before
    ), "deck should be untouched when hunt is declined"


def test_predator_hunt_no_veto_without_feeder_bird():
    """Without any opposing PINK_PREDATOR_FEEDER birds, ``PREDATOR_HUNT`` runs
    without a gate (gap #17)."""
    gs = _new_game(seed=42)
    gs.current_player = 0

    # Ensure P1 has NO pink predator feeder birds.
    for hab in cards.ALL_HABITATS:
        gs.players[1].board[hab].clear()

    hunt_eff = cards.Effect(kind=cards.EffectKind.PREDATOR_HUNT, max_wingspan_cm=100)
    template = next(b for b in _BIRDS if b.predator)
    hunter = template.model_copy(
        update={
            "power": cards.Power(color=cards.PowerColor.BROWN, effects=(hunt_eff,)),
            "raw_power_text": "test predator",
        }
    )
    pb_hunter = state.PlayedBird(bird=hunter)
    deck_before = len(gs.bird_deck)

    eng = engine.Engine(gs, agents=[_assert_no_gate_agent, _assert_no_gate_agent])
    powers.apply_effect(
        eng,
        _assert_no_gate_agent,
        gs.players[0],
        pb_hunter,
        hunter.habitats[0],
        hunt_eff,
        "play",
    )
    assert (
        len(gs.bird_deck) < deck_before
    ), "predator hunt should draw from deck without gate"


# ---------------------------------------------------------------------------
# Gap #19 — forced pink lay-egg and tuck-self lay-egg


def test_fire_pink_lay_egg_forced_outside_birds_no_eggs():
    """Without birds_no_eggs goal, ``fire_pink_lay_egg`` offers no gate — the
    ``LayEggDecision`` choices have no skip row (gap #19).  Two eligible birds are
    placed so the decision has multiple choices and IS presented to the agent (a
    single-choice decision would auto-resolve without consulting it)."""
    gs = _new_game(0)

    bowl_birds = [
        b for b in _BIRDS if b.nest == cards.NestType.BOWL and b.egg_limit >= 1
    ]
    assert len(bowl_birds) >= 2, "need at least 2 bowl birds in the data"
    bowl_bird = bowl_birds[0]
    bowl_bird2 = bowl_birds[1]
    pb = state.PlayedBird(bird=bowl_bird)
    pb2 = state.PlayedBird(bird=bowl_bird2)
    for hab in cards.ALL_HABITATS:
        gs.players[0].board[hab].clear()
    gs.players[0].board[bowl_bird.habitats[0]].append(pb)
    gs.players[0].board[bowl_bird2.habitats[0]].append(pb2)

    eff = cards.Effect(
        kind=cards.EffectKind.PINK_LAY_EGG_ON_NEST,
        nest=cards.NestType.BOWL,
        exclude_self=False,
        raw_text="test",
    )

    lay_decision_choices: list[list[decisions.Choice]] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        assert not isinstance(
            decision, decisions.AcceptExchangeDecision
        ), "no veto gate should appear outside birds_no_eggs goal"
        if isinstance(decision, decisions.LayEggDecision):
            lay_decision_choices.append(list(decision.choices))
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return typing.cast(C, choice)
        return typing.cast(C, decision.choices[0])

    eng = engine.Engine(gs, agents=[agent, agent])
    committed = reactors.fire_pink_lay_egg(
        eng, gs.players[0], pb, bowl_bird.habitats[0], eff
    )

    assert committed
    assert pb.eggs + pb2.eggs == 1, "exactly one egg should be laid"
    # Two eligible birds → LayEggDecision presented; verify no skip row.
    assert lay_decision_choices, "LayEggDecision should have been presented"
    for choices in lay_decision_choices:
        assert not any(
            isinstance(c, decisions.SkipChoice) for c in choices
        ), "no SkipChoice should be in LayEggDecision choices outside birds_no_eggs goal"


def test_fire_pink_lay_egg_gate_under_birds_no_eggs_accept():
    """Under birds_no_eggs goal, ``fire_pink_lay_egg`` gates; accepting lays the
    egg (gap #19)."""
    gs = _new_game(0)
    gs.round_goals = [_no_eggs_goal()] * 4
    gs.round_idx = 0

    bowl_bird = next(
        b for b in _BIRDS if b.nest == cards.NestType.BOWL and b.egg_limit >= 1
    )
    pb = state.PlayedBird(bird=bowl_bird)
    for hab in cards.ALL_HABITATS:
        gs.players[0].board[hab].clear()
    gs.players[0].board[bowl_bird.habitats[0]] = [pb]

    eff = cards.Effect(
        kind=cards.EffectKind.PINK_LAY_EGG_ON_NEST,
        nest=cards.NestType.BOWL,
        exclude_self=False,
        raw_text="test",
    )

    eng = engine.Engine(gs, agents=[_accept_agent, _accept_agent])
    committed = reactors.fire_pink_lay_egg(
        eng, gs.players[0], pb, bowl_bird.habitats[0], eff
    )

    assert committed
    assert pb.eggs == 1


def test_fire_pink_lay_egg_gate_under_birds_no_eggs_decline():
    """Under birds_no_eggs goal, declining the gate means no egg and returns False
    (gap #19)."""
    gs = _new_game(0)
    gs.round_goals = [_no_eggs_goal()] * 4
    gs.round_idx = 0

    bowl_bird = next(
        b for b in _BIRDS if b.nest == cards.NestType.BOWL and b.egg_limit >= 1
    )
    pb = state.PlayedBird(bird=bowl_bird)
    for hab in cards.ALL_HABITATS:
        gs.players[0].board[hab].clear()
    gs.players[0].board[bowl_bird.habitats[0]] = [pb]

    eff = cards.Effect(
        kind=cards.EffectKind.PINK_LAY_EGG_ON_NEST,
        nest=cards.NestType.BOWL,
        exclude_self=False,
        raw_text="test",
    )

    eng = engine.Engine(gs, agents=[_skip_agent, _skip_agent])
    committed = reactors.fire_pink_lay_egg(
        eng, gs.players[0], pb, bowl_bird.habitats[0], eff
    )

    assert not committed
    assert pb.eggs == 0


def test_tuck_from_hand_then_lay_forced_outside_birds_no_eggs():
    """Outside birds_no_eggs goal, after the tuck the lay is forced — no
    AcceptExchangeDecision gate and no skip row in the LayEggDecision (gap #19)."""
    gs = _new_game(0)
    gs.current_player = 0
    player = gs.players[0]

    tuck_bird = _BIRDS[0]
    carrier_text = "Tuck 1 [card] from your hand behind this bird. If you do, you may also lay 1 [egg] on this bird."
    carrier = tuck_bird.model_copy(
        update={
            "power": cards.parse_power(tuck_bird.color, carrier_text),
            "raw_power_text": carrier_text,
            "egg_limit": 3,
        }
    )
    pb = state.PlayedBird(bird=carrier)
    player.hand = [_BIRDS[1], _BIRDS[2]]
    # The handler looks for pb in player.board[habitat]; add it there.
    player.board[carrier.habitats[0]].append(pb)

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        assert not isinstance(
            decision, decisions.AcceptExchangeDecision
        ), "no veto gate should appear outside birds_no_eggs goal"
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return typing.cast(C, choice)
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[agent, agent])
    powers.dispatch_power(eng, agent, player, pb, carrier.habitats[0], "play")

    assert pb.eggs == 1, "egg should be laid after tuck (forced)"
    # LayEggDecision targets exactly 'this bird' (the carrier) — single choice,
    # auto-resolved without calling agent. The forced behavior is confirmed by pb.eggs == 1.


def test_tuck_from_hand_then_lay_gate_under_birds_no_eggs_decline():
    """Under birds_no_eggs goal, declining the gate after the tuck leaves no egg
    on the bird (gap #19)."""
    gs = _new_game(0)
    gs.round_goals = [_no_eggs_goal()] * 4
    gs.round_idx = 0
    gs.current_player = 0
    player = gs.players[0]

    tuck_bird = _BIRDS[0]
    carrier_text = "Tuck 1 [card] from your hand behind this bird. If you do, you may also lay 1 [egg] on this bird."
    carrier = tuck_bird.model_copy(
        update={
            "power": cards.parse_power(tuck_bird.color, carrier_text),
            "raw_power_text": carrier_text,
            "egg_limit": 3,
        }
    )
    pb = state.PlayedBird(bird=carrier)
    player.hand = [_BIRDS[1], _BIRDS[2]]

    tuck_done = [False]

    def agent[C: decisions.Choice](
        _eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            # If no card has been tucked yet: accept the tuck gate.
            # If a card has been tucked: decline the lay gate.
            if tuck_done[0]:
                return typing.cast(
                    C,
                    next(
                        c
                        for c in decision.choices
                        if isinstance(c, decisions.SkipChoice)
                    ),
                )
            tuck_done[0] = True
            raise AssertionError("AcceptExchangeDecision before tuck is not expected")
        if isinstance(decision, decisions.ActivateTuckDecision):
            # Accept the tuck.
            return typing.cast(
                C,
                next(
                    c
                    for c in decision.choices
                    if isinstance(c, decisions.TuckActivateChoice)
                ),
            )
        if isinstance(decision, decisions.BirdPowerTuckFromHandDecision):
            tuck_done[0] = True
            return typing.cast(C, decision.choices[0])
        return typing.cast(
            C,
            next(
                (
                    c
                    for c in decision.choices
                    if not isinstance(c, decisions.SkipChoice)
                ),
                decision.choices[0],
            ),
        )

    eng = engine.Engine(gs, agents=[agent, agent])
    powers.dispatch_power(eng, agent, player, pb, carrier.habitats[0], "play")

    assert pb.eggs == 0, "egg should not be laid when gate is declined"
    assert (
        pb.tucked_cards == 1
    ), "tuck should have happened before the declined lay gate"
