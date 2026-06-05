"""Tests for the ALL_PLAYERS_LAY_EGG_ON_NEST bird power (3 birds in core).

Birds:
- Lazuli Bunting       -- bowl
- Pileated Woodpecker  -- cavity
- Western Meadowlark   -- ground

Power text: "All players lay 1 [egg] on any 1 [<nest>] bird. You may lay 1
[egg] on 1 additional [<nest>] bird."
"""

from __future__ import annotations

import os
import random
import sys
import typing

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state  # noqa: E402
from wingspan.engine import powers  # noqa: E402

TARGET_BIRDS = {
    "Lazuli Bunting": cards.NestType.BOWL,
    "Pileated Woodpecker": cards.NestType.CAVITY,
    "Western Meadowlark": cards.NestType.GROUND,
}


def _by_name(birds: list[cards.Bird], name: str) -> cards.Bird:
    for bird in birds:
        if bird.name == name:
            return bird
    raise KeyError(name)


def test_parse_all_players_lay_egg_on_nest():
    """Each printed sentence variant should parse to the new effect kind."""
    for nest_word, expected in [
        ("bowl", cards.NestType.BOWL),
        ("cavity", cards.NestType.CAVITY),
        ("ground", cards.NestType.GROUND),
    ]:
        text = (
            f"All players lay 1 [egg] on any 1 [{nest_word}] bird. "
            f"You may lay 1 [egg] on 1 additional [{nest_word}] bird."
        )
        power = cards.parse_power(cards.PowerColor.WHITE, text)
        kinds = [effect.kind for effect in power.effects]
        assert (
            cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST in kinds
        ), f"failed to parse for nest={nest_word}: {kinds}"
        eff = next(
            effect
            for effect in power.effects
            if effect.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
        )
        assert eff.nest == expected
        assert eff.amount == 1  # optional second sentence present -> 1 extra for self
        assert cards.EffectKind.UNIMPLEMENTED not in kinds

    # Variant without the optional second sentence: amount should be 0.
    text = "All players lay 1 [egg] on any 1 [bowl] bird."
    power = cards.parse_power(cards.PowerColor.WHITE, text)
    eff = next(
        effect
        for effect in power.effects
        if effect.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
    )
    assert eff.nest == cards.NestType.BOWL
    assert eff.amount == 0


def test_all_three_target_birds_implemented():
    birds, _, _ = cards.load_all()
    for name, expected_nest in TARGET_BIRDS.items():
        bird = _by_name(birds, name)
        kinds = [effect.kind for effect in bird.power.effects]
        assert (
            cards.EffectKind.UNIMPLEMENTED not in kinds
        ), f"{name} still UNIMPLEMENTED; raw={bird.raw_power_text!r}"
        assert (
            cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST in kinds
        ), f"{name} parsed as {kinds}; raw={bird.raw_power_text!r}"
        eff = next(
            effect
            for effect in bird.power.effects
            if effect.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
        )
        assert eff.nest == expected_nest
        assert eff.amount == 1


@pytest.mark.parametrize("bird_name,nest", list(TARGET_BIRDS.items()))
def test_power_every_player_lays_one_egg_on_matching_nest(
    bird_name: str, nest: cards.NestType
):
    """Give each player a matching-nest bird with room; expect each gets +1 egg."""
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(0)
    gs = state.new_game(rng, birds, bonuses, goals)

    power_bird = _by_name(birds, bird_name)
    # Pick any bird with the right nest type and an egg_limit >= 2 so it has room.
    target = next(
        bird
        for bird in birds
        if bird.nest == nest and bird.egg_limit >= 2 and bird.name != bird_name
    )

    # Each player gets one target-nest bird (empty) plus a non-matching bird so
    # we can confirm the egg lands on the matching bird, not the other.
    decoy = next(
        bird
        for bird in birds
        if bird.nest != nest
        and bird.nest != cards.NestType.STAR
        and bird.nest != cards.NestType.NONE
        and bird.egg_limit >= 1
        and bird.name != bird_name
    )
    pbs: list[tuple[state.PlayedBird, state.PlayedBird]] = []
    pb_p0_extra: state.PlayedBird | None = None
    for player_idx, player in enumerate(gs.players):
        habitat = target.habitats[0]
        decoy_habitat = decoy.habitats[0]
        pb_target = state.PlayedBird(bird=target)
        pb_decoy = state.PlayedBird(bird=decoy)
        # Place decoy in a different column slot if same habitat to avoid clobber.
        player.board[habitat].append(pb_target)
        if decoy_habitat == habitat:
            player.board[habitat].append(pb_decoy)
        else:
            player.board[decoy_habitat].append(pb_decoy)
        pbs.append((pb_target, pb_decoy))
        # Active player (P0) gets a second matching-nest bird so the extra egg
        # (gap #13a) has a distinct target to land on after the base egg is
        # placed on pb_target.
        if player_idx == 0:
            pb_p0_extra = state.PlayedBird(bird=target)
            player.board[habitat].append(pb_p0_extra)

    # The power bird is held off-board so its own (matching) nest doesn't
    # confuse choice resolution — we trigger the effect directly.
    pb_power = state.PlayedBird(bird=power_bird)

    # Scripted agent: prefer the `target` bird when present, otherwise pick
    # the first non-skip choice.
    target_label_substr = f"{target.name}@"

    def script_agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        for choice in decision.choices:
            if (
                not isinstance(choice, decisions.SkipChoice)
                and target_label_substr in choice.label
            ):
                return choice
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return choice
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[script_agent, script_agent])
    gs.current_player = 0

    eff = next(
        effect
        for effect in power_bird.power.effects
        if effect.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
    )
    powers.apply_effect(
        eng,
        script_agent,
        gs.players[0],
        pb_power,
        power_bird.habitats[0],
        eff,
        trigger="play",
    )

    # Each player's (first) matching-nest bird should have at least 1 egg.
    for pb_target, pb_decoy in pbs:
        assert pb_target.eggs >= 1, (
            f"every player should lay >=1 egg on matching-nest bird; "
            f"target {pb_target.bird.name} has {pb_target.eggs}"
        )
        assert (
            pb_decoy.eggs == 0
        ), f"non-matching-nest bird {pb_decoy.bird.name} must not receive eggs"
    # The active player (P0) has two matching-nest birds; the base egg lands on
    # pb_target and the extra egg (gap #13a — exclude prevents retargeting) on
    # pb_p0_extra. Total must be 2. The opponent has exactly 1.
    p0_target, _ = pbs[0]
    p1_target, _ = pbs[1]
    p0_extra_eggs = pb_p0_extra.eggs if pb_p0_extra is not None else 0
    assert p0_target.eggs + p0_extra_eggs >= 2, (
        f"active player should lay 2 eggs total (1 base + 1 extra on 2nd bird); "
        f"got {p0_target.eggs}+{p0_extra_eggs}"
    )
    assert (
        p1_target.eggs == 1
    ), f"opponent should lay exactly 1 egg; got {p1_target.eggs}"


def test_power_skipped_when_no_matching_nest_bird():
    """If neither player has a matching-nest bird, the power is a silent no-op."""
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(1)
    gs = state.new_game(rng, birds, bonuses, goals)

    power_bird = _by_name(birds, "Lazuli Bunting")  # bowl
    # Put only non-bowl birds on each player's board. Do NOT place the power
    # bird itself (which has a bowl nest) — that would make it eligible.
    non_bowl = next(
        bird
        for bird in birds
        if bird.nest
        not in (cards.NestType.BOWL, cards.NestType.STAR, cards.NestType.NONE)
        and bird.name != power_bird.name
    )
    for player in gs.players:
        player.board[non_bowl.habitats[0]].append(state.PlayedBird(bird=non_bowl))

    pb_power = state.PlayedBird(bird=power_bird)  # off-board; we invoke directly

    def script_agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[script_agent, script_agent])
    gs.current_player = 0

    eff = next(
        effect
        for effect in power_bird.power.effects
        if effect.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
    )
    total_before = sum(
        pb.eggs
        for other_player in gs.players
        for row in other_player.board.values()
        for pb in row
    )
    powers.apply_effect(
        eng,
        script_agent,
        gs.players[0],
        pb_power,
        power_bird.habitats[0],
        eff,
        trigger="play",
    )
    total_after = sum(
        pb.eggs
        for other_player in gs.players
        for row in other_player.board.values()
        for pb in row
    )
    assert (
        total_after == total_before
    ), "no eggs should be laid when no eligible nests exist"


def test_egg_limit_respected():
    """A matching-nest bird already at egg_limit must not receive an egg."""
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(2)
    gs = state.new_game(rng, birds, bonuses, goals)

    power_bird = _by_name(birds, "Pileated Woodpecker")  # cavity
    cavity = next(
        bird
        for bird in birds
        if bird.nest == cards.NestType.CAVITY
        and bird.egg_limit >= 1
        and bird.name != power_bird.name
    )

    # P0 has the cavity bird full; P1 has a fresh cavity bird with room.
    gs.players[0].board[cavity.habitats[0]].append(
        state.PlayedBird(bird=cavity, eggs=cavity.egg_limit)
    )
    gs.players[1].board[cavity.habitats[0]].append(
        state.PlayedBird(bird=cavity, eggs=0)
    )
    pb_power = state.PlayedBird(bird=power_bird)
    gs.players[0].board[power_bird.habitats[0]].append(pb_power)

    def script_agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return choice
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[script_agent, script_agent])
    gs.current_player = 0
    eff = next(
        effect
        for effect in power_bird.power.effects
        if effect.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
    )
    powers.apply_effect(
        eng,
        script_agent,
        gs.players[0],
        pb_power,
        power_bird.habitats[0],
        eff,
        trigger="play",
    )

    # P0's only cavity bird was full -- no change.
    p0_cavity = gs.players[0].board[cavity.habitats[0]][0]
    assert p0_cavity.eggs == cavity.egg_limit
    # P1's cavity bird received exactly 1 egg.
    p1_cavity = gs.players[1].board[cavity.habitats[0]][0]
    assert p1_cavity.eggs == 1


def test_star_nest_bird_is_eligible_for_nest_lay():
    """Star nests are wild: a star-nest bird is a legal target for "lay 1 egg
    on a [bowl] bird" even though its printed nest is not a bowl."""
    birds, bonuses, goals = cards.load_all()
    gs = state.new_game(random.Random(3), birds, bonuses, goals)
    star = next(
        bird
        for bird in birds
        if bird.nest == cards.NestType.STAR and bird.egg_limit >= 1
    )
    pb_star = state.PlayedBird(bird=star)
    gs.players[0].board[star.habitats[0]].append(pb_star)

    def script_agent[C: decisions.Choice](
        _engine: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return choice
        return decision.choices[0]

    eng = engine.Engine(gs, agents=[script_agent, script_agent])
    gs.current_player = 0
    powers.lay_one_egg_on_nest(eng, gs.players[0], cards.NestType.BOWL, label="test")
    assert pb_star.eggs == 1, "a star-nest bird must count as a [bowl] bird"


# ---------------------------------------------------------------------------
# Gap #13 — veto ledger, exclude param, birds_no_eggs gate


def test_veto_ledger_shows_min_2_own_eligible_count() -> None:
    """``gained_egg_count`` in the AcceptExchangeDecision must be
    ``min(2, own_eligible_count)`` (gap #13b)."""
    birds, bonuses, goals = cards.load_all()

    power_bird = _by_name(birds, "Lazuli Bunting")  # bowl, amount=1
    bowl_bird = next(
        bird
        for bird in birds
        if bird.nest == cards.NestType.BOWL
        and bird.egg_limit >= 2
        and bird.name != power_bird.name
    )
    eff = next(
        e
        for e in power_bird.power.effects
        if e.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
    )

    # ── Case A: 1 eligible bird → ledger should show gained_egg_count=1 ──
    gs_a = state.new_game(random.Random(10), birds, bonuses, goals)
    gs_a.current_player = 0
    pb_a = state.PlayedBird(bird=bowl_bird, eggs=0)
    gs_a.players[0].board[bowl_bird.habitats[0]] = [pb_a]
    for h in cards.ALL_HABITATS:
        gs_a.players[1].board[h].clear()

    captured_a: list[decisions.AcceptExchangeDecision] = []

    def agent_a[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            captured_a.append(decision)
            return typing.cast(C, decision.choices[0])  # accept
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return typing.cast(C, choice)
        return decision.choices[0]

    pb_power = state.PlayedBird(bird=power_bird)
    eng_a = engine.Engine(gs_a, agents=[agent_a, agent_a])
    powers.apply_effect(
        eng_a, agent_a, gs_a.players[0], pb_power, power_bird.habitats[0], eff, "play"
    )
    accept_a = next(
        c for c in captured_a[0].choices if isinstance(c, decisions.PayCostChoice)
    )
    assert (
        accept_a.gained_egg_count == 1
    ), "1 eligible bird → ledger must show gained_egg_count=1"

    # ── Case B: 3 eligible birds → ledger should show gained_egg_count=2 (capped) ──
    gs_b = state.new_game(random.Random(11), birds, bonuses, goals)
    gs_b.current_player = 0
    # Give P0 three bowl birds with room.
    for _ in range(3):
        gs_b.players[0].board[bowl_bird.habitats[0]].append(
            state.PlayedBird(bird=bowl_bird, eggs=0)
        )
    for h in cards.ALL_HABITATS:
        gs_b.players[1].board[h].clear()

    captured_b: list[decisions.AcceptExchangeDecision] = []

    def agent_b[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            captured_b.append(decision)
            return typing.cast(C, decision.choices[0])  # accept
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return typing.cast(C, choice)
        return decision.choices[0]

    pb_power_b = state.PlayedBird(bird=power_bird)
    eng_b = engine.Engine(gs_b, agents=[agent_b, agent_b])
    powers.apply_effect(
        eng_b,
        agent_b,
        gs_b.players[0],
        pb_power_b,
        power_bird.habitats[0],
        eff,
        "play",
    )
    accept_b = next(
        c for c in captured_b[0].choices if isinstance(c, decisions.PayCostChoice)
    )
    assert (
        accept_b.gained_egg_count == 2
    ), "3 eligible birds → ledger must be capped at 2"


def test_extra_egg_cannot_target_base_egg_bird_when_only_one_eligible() -> None:
    """When the active player has exactly one matching-nest bird, the base egg
    goes on it. The extra egg has no other eligible target: it silently does
    nothing (gap #13a — ``exclude`` param)."""
    birds, bonuses, goals = cards.load_all()
    power_bird = _by_name(birds, "Lazuli Bunting")  # bowl, amount=1 extra
    bowl_bird = next(
        bird
        for bird in birds
        if bird.nest == cards.NestType.BOWL
        and bird.egg_limit >= 2
        and bird.name != power_bird.name
    )
    eff = next(
        e
        for e in power_bird.power.effects
        if e.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
    )

    gs = state.new_game(random.Random(20), birds, bonuses, goals)
    gs.current_player = 0
    pb_only = state.PlayedBird(bird=bowl_bird, eggs=0)
    gs.players[0].board[bowl_bird.habitats[0]] = [pb_only]
    for h in cards.ALL_HABITATS:
        gs.players[1].board[h].clear()

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return typing.cast(C, choice)
        return decision.choices[0]

    pb_power = state.PlayedBird(bird=power_bird)
    eng = engine.Engine(gs, agents=[agent, agent])
    powers.apply_effect(
        eng, agent, gs.players[0], pb_power, power_bird.habitats[0], eff, "play"
    )

    # The base egg is laid on pb_only (the only eligible bird, auto-resolved by
    # Engine.ask since it is the sole choice). The extra egg has no distinct
    # target — the exclude param prevents retargeting pb_only — so it is silently
    # dropped. Total eggs must be exactly 1.
    assert pb_only.eggs == 1, (
        "extra-egg target is excluded (only one bird); "
        "total eggs must be exactly 1 (base only)"
    )


def test_extra_egg_is_optional_under_birds_no_eggs_goal() -> None:
    """Under the birds_no_eggs round goal, the extra egg must be presented as
    an optional AcceptExchangeDecision (gap #13c)."""
    birds, bonuses, goals = cards.load_all()
    power_bird = _by_name(birds, "Lazuli Bunting")  # bowl, amount=1 extra
    bowl_bird = next(
        bird
        for bird in birds
        if bird.nest == cards.NestType.BOWL
        and bird.egg_limit >= 3
        and bird.name != power_bird.name
    )
    eff = next(
        e
        for e in power_bird.power.effects
        if e.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
    )

    gs = state.new_game(random.Random(30), birds, bonuses, goals)
    gs.current_player = 0

    # Two matching birds so the extra egg has a different target.
    pb_1 = state.PlayedBird(bird=bowl_bird, eggs=0)
    pb_2 = state.PlayedBird(bird=bowl_bird, eggs=0)
    gs.players[0].board[bowl_bird.habitats[0]] = [pb_1, pb_2]
    for h in cards.ALL_HABITATS:
        gs.players[1].board[h].clear()

    # Patch the active round goal to birds_no_eggs.
    anti_egg_goal = cards.EndRoundGoal(
        id=99, description="birds without eggs", category="birds_no_eggs", tile_id=99
    )
    gs.round_goals[gs.round_idx] = anti_egg_goal

    accept_decisions: list[decisions.AcceptExchangeDecision] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            accept_decisions.append(decision)
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return typing.cast(C, choice)
        return typing.cast(C, decision.choices[0])

    pb_power = state.PlayedBird(bird=power_bird)
    eng = engine.Engine(gs, agents=[agent, agent])
    powers.apply_effect(
        eng, agent, gs.players[0], pb_power, power_bird.habitats[0], eff, "play"
    )

    # There must be an AcceptExchangeDecision for the extra egg (with a skip row).
    extra_egg_accepts = [
        d
        for d in accept_decisions
        if any(
            isinstance(c, decisions.SkipChoice) and c.label == "skip" for c in d.choices
        )
        and any(
            isinstance(c, decisions.PayCostChoice) and c.label == "lay extra egg"
            for c in d.choices
        )
    ]
    assert (
        len(extra_egg_accepts) >= 1
    ), "under birds_no_eggs goal an extra-egg AcceptExchangeDecision must appear"
