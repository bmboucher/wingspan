"""Tests for the EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER power.

Two birds carry this in the core set:
    Anna's Hummingbird
    Ruby-Throated Hummingbird

Power text: "Each player gains 1 [die] from the birdfeeder, starting with
the player of your choice."
"""

from __future__ import annotations

import random
import typing

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers


def test_parse_each_player_gains_die():
    power = cards.parse_power(
        cards.PowerColor.WHITE,
        "Each player gains 1 [die] from the birdfeeder, starting with the player of your choice.",
    )
    assert len(power.effects) == 1
    eff = power.effects[0]
    assert eff.kind == cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER
    assert eff.amount == 1


def test_both_hummingbirds_are_implemented():
    birds, _, _ = cards.load_all()
    target_names = {"Anna's Hummingbird", "Ruby-Throated Hummingbird"}
    found = {bird.name: bird for bird in birds if bird.name in target_names}
    missing = target_names - set(found)
    assert not missing, f"birds not present in data: {missing}"
    for name, bird in found.items():
        kinds = [effect.kind for effect in bird.power.effects]
        assert (
            cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER in kinds
        ), f"{name} parsed as {kinds}; raw_power_text={bird.raw_power_text!r}"
        assert cards.EffectKind.UNIMPLEMENTED not in kinds, name


def _setup_state(seed: int = 0, num_players: int = 2):
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(seed)
    return state.new_game(rng, birds, bonuses, goals, num_players=num_players)


def _take_first_decline_reset[C: decisions.Choice](
    decision: decisions.Decision[C],
) -> C:
    """Decline any optional single-face reset; otherwise take the first
    offered choice. Shared scripted-agent building block for the tests below
    that don't care which specific die / order / food is picked."""
    if isinstance(decision, decisions.ResetBirdfeederDecision):
        for choice in decision.choices:
            if isinstance(choice, decisions.SkipChoice):
                return typing.cast(C, choice)
    return typing.cast(C, decision.choices[0])


def test_each_player_gains_die_credits_both_players_in_chosen_order():
    """Active player picks P1 to start, so the dice are credited P1 then P0.

    The feeder holds exactly two distinct faces (2 SEED + 2 FRUIT) so the order
    decision *is* asked — this is the genuinely ambiguous case. Two dice of each
    face type means neither take reduces the feeder to a single face, so no reset
    is offered mid-power."""
    gs = _setup_state(seed=0)

    # Exactly two faces, two dice each: order decision is asked; no reset offered.
    for food in gs.birdfeeder.counts:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.choice_dice = 0  # controlled feeder: clear the choice face
    gs.birdfeeder.counts[cards.Food.SEED] = 2
    gs.birdfeeder.counts[cards.Food.FRUIT] = 2

    p0, p1 = gs.players
    gs.current_player = p0.id
    food_before_p0 = p0.food.as_dict()
    food_before_p1 = p1.food.as_dict()

    carrier = next(
        bird for bird in cards.load_all()[0] if bird.name == "Anna's Hummingbird"
    )
    pb = state.PlayedBird(bird=carrier)
    eff = cards.Effect(
        kind=cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1
    )

    # Scripted agents:
    #   - active player (p0) accepts the veto gate, picks the starting player
    #     as P1, then picks the first die available
    #   - each player picks the first die available
    def agent_p0[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            # Accept the all-players veto gate (gap #16).
            return typing.cast(
                C,
                next(
                    c
                    for c in decision.choices
                    if isinstance(c, decisions.PayCostChoice)
                ),
            )
        if isinstance(decision, decisions.BirdPowerPickGainOrderDecision):
            assert decision.player_id == p0.id
            for choice in decision.choices:
                if choice.player_id == p1.id:
                    return typing.cast(C, choice)
            raise AssertionError("p1 not in choices")
        if isinstance(decision, decisions.GainFoodDecision):
            assert decision.player_id == p0.id
            return typing.cast(C, decision.choices[0])
        raise AssertionError(f"unexpected decision for p0: {type(decision).__name__}")

    def agent_p1[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        assert isinstance(decision, decisions.GainFoodDecision)
        assert decision.player_id == p1.id
        return typing.cast(C, decision.choices[0])

    eng = engine.Engine(gs, agents=[agent_p0, agent_p1])
    powers.apply_effect(eng, agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    # Total food gained: each player got exactly 1 die.
    assert sum(p0.food.values()) - sum(food_before_p0.values()) == 1
    assert sum(p1.food.values()) - sum(food_before_p1.values()) == 1
    assert gs.birdfeeder.total() == 2  # four dice, two taken


def test_each_player_gains_die_refills_empty_feeder():
    """Only 1 die is in the feeder (1 face), but the starter taking it empties the
    feeder, which is immediately rerolled (Rule 1) — so the second player still
    gains a die rather than the power stopping early.

    With a single-face feeder, the order decision is auto-resolved (going first is
    strictly optimal), so no BirdPowerPickGainOrderDecision is ever presented."""
    gs = _setup_state(seed=1)

    for food in gs.birdfeeder.counts:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.choice_dice = 0  # controlled feeder: clear the choice face
    gs.birdfeeder.counts[cards.Food.SEED] = 1

    p0, p1 = gs.players
    gs.current_player = p0.id
    food_before_p0 = p0.food.as_dict()
    food_before_p1 = p1.food.as_dict()

    carrier = next(
        bird for bird in cards.load_all()[0] if bird.name == "Ruby-Throated Hummingbird"
    )
    pb = state.PlayedBird(bird=carrier)
    eff = cards.Effect(
        kind=cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1
    )

    order_decisions_seen: list[decisions.BirdPowerPickGainOrderDecision] = []

    def agent_p0[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.BirdPowerPickGainOrderDecision):
            order_decisions_seen.append(decision)
            return typing.cast(C, decision.choices[0])
        return _take_first_decline_reset(decision)

    def agent_p1[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        return _take_first_decline_reset(decision)

    eng = engine.Engine(gs, agents=[agent_p0, agent_p1])
    powers.apply_effect(eng, agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    # 1-face feeder: order pick is auto-resolved, agent never sees it.
    assert not order_decisions_seen, (
        f"BirdPowerPickGainOrderDecision was presented for a 1-face feeder: "
        f"{order_decisions_seen}"
    )

    # Both players gained a die: the feeder refilled after the starter emptied it.
    assert sum(p0.food.values()) - sum(food_before_p0.values()) == 1
    assert sum(p1.food.values()) - sum(food_before_p1.values()) == 1
    assert gs.birdfeeder.total() > 0  # rerolled, not left empty


def test_each_player_gains_die_routes_decisions_to_correct_agent():
    """When the active player chooses themselves to start, ensure each agent is
    queried for its own die pick (and only for its own).

    The feeder holds exactly two distinct faces with 2 dice each (2 SEED + 2 RODENT)
    so the order decision *is* asked (2-face feeder = genuinely ambiguous case) and
    *both* gain-food picks stay a genuine 2-option fork: after the starter takes one
    die the second player still sees two food types. A single-option pick is forced
    and would be auto-resolved by ``Engine.ask`` without consulting the agent, so
    it would never appear in the routing log."""
    gs = _setup_state(seed=2)

    for food in gs.birdfeeder.counts:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.choice_dice = 0  # controlled feeder: clear the choice face
    gs.birdfeeder.counts[cards.Food.SEED] = 2
    gs.birdfeeder.counts[cards.Food.RODENT] = 2

    p0, p1 = gs.players
    gs.current_player = p0.id

    carrier = next(
        bird for bird in cards.load_all()[0] if bird.name == "Anna's Hummingbird"
    )
    pb = state.PlayedBird(bird=carrier)
    eff = cards.Effect(
        kind=cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1
    )

    queried: list[tuple[str, type, int]] = []

    def agent_p0[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        queried.append(("p0", type(decision), decision.player_id))
        if isinstance(decision, decisions.AcceptExchangeDecision):
            # Accept the all-players veto gate (gap #16).
            return typing.cast(
                C,
                next(
                    c
                    for c in decision.choices
                    if isinstance(c, decisions.PayCostChoice)
                ),
            )
        if isinstance(decision, decisions.BirdPowerPickGainOrderDecision):
            for choice in decision.choices:
                if choice.player_id == p0.id:
                    return typing.cast(C, choice)
        assert decision.player_id == p0.id
        return typing.cast(C, decision.choices[0])

    def agent_p1[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        queried.append(("p1", type(decision), decision.player_id))
        assert decision.player_id == p1.id
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[agent_p0, agent_p1])
    powers.apply_effect(eng, agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    # Expected query log: p0 accepts the veto gate, picks the starting player,
    # p0 picks a die, then p1 picks a die.
    kinds = [(who, dtype) for (who, dtype, _pid) in queried]
    assert kinds == [
        ("p0", decisions.AcceptExchangeDecision),
        ("p0", decisions.BirdPowerPickGainOrderDecision),
        ("p0", decisions.GainFoodDecision),
        ("p1", decisions.GainFoodDecision),
    ]


def test_each_player_gains_die_skips_order_pick_when_uncontested():
    """With more than 2 distinct faces in the feeder, going first is strictly
    optimal (neither player can reset), so the order decision is auto-resolved
    and the active player (P0) always gains before the opponent (P1)."""
    gs = _setup_state(seed=3)

    # Three distinct faces: >2, so the order pick is skipped.
    for food in gs.birdfeeder.counts:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.choice_dice = 0  # controlled feeder: clear the choice face
    gs.birdfeeder.counts[cards.Food.SEED] = 1
    gs.birdfeeder.counts[cards.Food.RODENT] = 1
    gs.birdfeeder.counts[cards.Food.FRUIT] = 1

    p0 = gs.players[0]
    gs.current_player = p0.id

    carrier = next(
        bird for bird in cards.load_all()[0] if bird.name == "Anna's Hummingbird"
    )
    pb = state.PlayedBird(bird=carrier)
    eff = cards.Effect(
        kind=cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1
    )

    queried: list[tuple[str, type]] = []

    def agent_p0[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        queried.append(("p0", type(decision)))
        if isinstance(decision, decisions.AcceptExchangeDecision):
            return typing.cast(
                C,
                next(
                    c
                    for c in decision.choices
                    if isinstance(c, decisions.PayCostChoice)
                ),
            )
        return decision.choices[0]

    def agent_p1[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        queried.append(("p1", type(decision)))
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[agent_p0, agent_p1])
    powers.apply_effect(eng, agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    # Order decision is never shown to any agent.
    order_queries = [
        (who, dt)
        for (who, dt) in queried
        if dt is decisions.BirdPowerPickGainOrderDecision
    ]
    assert (
        not order_queries
    ), f"Order decision was presented with >2 faces: {order_queries}"

    # Active player (P0) gains before the opponent (P1).
    gain_order = [
        (who, dt) for (who, dt) in queried if dt is decisions.GainFoodDecision
    ]
    assert gain_order == [
        ("p0", decisions.GainFoodDecision),
        ("p1", decisions.GainFoodDecision),
    ]


# ---------------------------------------------------------------------------
# N-player gate (3 players): distinct_faces() >= 2, or dice scarcer than
# seats, widens the order pick to every seat; otherwise the active player
# auto-starts exactly as at 2 players.


def _accept_veto[C: decisions.Choice](decision: decisions.Decision[C]) -> C:
    """Accept the all-players veto gate (gap #16) when offered; otherwise
    decline any reset and take the first available choice."""
    if isinstance(decision, decisions.AcceptExchangeDecision):
        return typing.cast(
            C,
            next(c for c in decision.choices if isinstance(c, decisions.PayCostChoice)),
        )
    return _take_first_decline_reset(decision)


def _accept_veto_agent[C: decisions.Choice](
    _engine: engine.Engine, decision: decisions.Decision[C]
) -> C:
    """An ``Agent``-shaped wrapper over :func:`_accept_veto`, for seats with
    no test-specific behavior beyond accepting the veto / declining resets /
    taking the first offered choice."""
    return _accept_veto(decision)


def _record_order_choices(
    decision: decisions.Decision[typing.Any], sink: list[int]
) -> None:
    """If ``decision`` is the order pick, record the candidate player ids
    into ``sink`` (mutated in place). Takes the untyped ``Decision[Any]``
    shape — rather than narrowing a generic caller's own ``decision: Decision[C]``
    in place — so the isinstance check here can't widen that caller's return
    type into a union at its post-call join point."""
    if isinstance(decision, decisions.BirdPowerPickGainOrderDecision):
        sink.extend(choice.player_id for choice in decision.choices)


def test_each_player_gains_die_n3_two_or_more_faces_offers_all_seats():
    """At 3 players, 2 distinct feeder faces already widen the order pick to
    every seat (matching the 2-player ``distinct_faces() == 2`` case)."""
    gs = _setup_state(seed=0, num_players=3)
    for food in gs.birdfeeder.counts:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.choice_dice = 0
    gs.birdfeeder.counts[cards.Food.SEED] = 2
    gs.birdfeeder.counts[cards.Food.FRUIT] = 2

    p0, p1, p2 = gs.players
    gs.current_player = p0.id
    carrier = next(
        bird for bird in cards.load_all()[0] if bird.name == "Anna's Hummingbird"
    )
    pb = state.PlayedBird(bird=carrier)
    eff = cards.Effect(
        kind=cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1
    )

    order_choices_seen: list[int] = []

    def agent_p0[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        _record_order_choices(decision, order_choices_seen)
        return _accept_veto(decision)

    eng = engine.Engine(gs, agents=[agent_p0, _accept_veto_agent, _accept_veto_agent])
    powers.apply_effect(eng, agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    assert order_choices_seen == [p0.id, p1.id, p2.id]


def test_each_player_gains_die_n3_same_face_scarce_dice_offers_all_seats():
    """At 3 players, a single-face feeder with FEWER dice than seats (2 dice,
    3 players) still widens the order pick — someone is guaranteed to gain
    nothing, so who goes last matters even though there is nothing to reset
    over (the 2-player gate would auto-resolve this: 1 face is <= 1)."""
    gs = _setup_state(seed=0, num_players=3)
    for food in gs.birdfeeder.counts:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.choice_dice = 0
    gs.birdfeeder.counts[cards.Food.SEED] = 2  # 1 face, 2 dice < 3 players

    p0, p1, p2 = gs.players
    gs.current_player = p0.id
    carrier = next(
        bird for bird in cards.load_all()[0] if bird.name == "Anna's Hummingbird"
    )
    pb = state.PlayedBird(bird=carrier)
    eff = cards.Effect(
        kind=cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1
    )

    order_choices_seen: list[int] = []

    def agent_p0[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        _record_order_choices(decision, order_choices_seen)
        return _accept_veto(decision)

    eng = engine.Engine(gs, agents=[agent_p0, _accept_veto_agent, _accept_veto_agent])
    powers.apply_effect(eng, agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    assert order_choices_seen == [p0.id, p1.id, p2.id]


def test_each_player_gains_die_n3_same_face_plentiful_dice_auto_starts():
    """At 3 players, a single-face feeder with at least as many dice as seats
    (4 dice, 3 players) auto-starts the active player, exactly like the
    2-player 1-face case — no order decision is ever presented.

    Uses the invertebrate/seed *choice* face rather than a single plain food:
    a choice face is still exactly 1 distinct face (``distinct_faces() ==
    1``, so the order gate sees the uncontested case), but it offers 2
    gainable foods, so each seat's ``GainFoodDecision`` stays a genuine fork
    instead of being auto-resolved by ``Engine.ask``'s single-choice guard —
    otherwise this test could not tell "auto-started, not asked" apart from
    "asked, but forced" for the gain-order assertion below."""
    gs = _setup_state(seed=0, num_players=3)
    for food in gs.birdfeeder.counts:
        gs.birdfeeder.counts[food] = 0
    gs.birdfeeder.choice_dice = 4  # 1 face (the choice face), 4 dice >= 3 players

    p0, p1, p2 = gs.players
    gs.current_player = p0.id
    carrier = next(
        bird for bird in cards.load_all()[0] if bird.name == "Anna's Hummingbird"
    )
    pb = state.PlayedBird(bird=carrier)
    eff = cards.Effect(
        kind=cards.EffectKind.EACH_PLAYER_GAINS_DIE_CHOOSE_ORDER, amount=1
    )

    queried: list[tuple[int, type]] = []

    def agent_p0[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        queried.append((p0.id, type(decision)))
        return _accept_veto(decision)

    def agent_p1[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        queried.append((p1.id, type(decision)))
        return _take_first_decline_reset(decision)

    def agent_p2[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        queried.append((p2.id, type(decision)))
        return _take_first_decline_reset(decision)

    eng = engine.Engine(gs, agents=[agent_p0, agent_p1, agent_p2])
    powers.apply_effect(eng, agent_p0, p0, pb, carrier.habitats[0], eff, "play")

    order_queries = [
        entry
        for entry in queried
        if entry[1] is decisions.BirdPowerPickGainOrderDecision
    ]
    assert (
        not order_queries
    ), f"order decision presented with scarce/uncontested dice: {order_queries}"

    # Active player gains before either opponent, in clockwise order.
    gain_order = [
        seat for seat, dtype in queried if dtype is decisions.GainFoodDecision
    ]
    assert gain_order == [p0.id, p1.id, p2.id]
