"""Under the ``birds_no_eggs`` round goal each player is asked at most once per
egg, and the first accept/skip decides whether the egg is laid.

Also pins the ``opp_gained_egg_count`` convention of the egg-granting gates
(total across eligible opponents; zero for self-only powers).
"""

from __future__ import annotations

import random
import typing

import pytest

from wingspan import cards, decisions, engine, state
from wingspan.engine import powers, reactors

NEST = cards.NestType.BOWL
EGG_ROOM_MIN = 3

# ---------------------------------------------------------------------------
# Helpers


class _Script:
    """Agent that answers every ``AcceptExchangeDecision`` with accept or skip
    (per ``accept``) and records the prompts per player; other decisions take
    the first non-skip row."""

    def __init__(self, accept: bool | dict[int, bool]) -> None:
        self._accept = accept
        self.accepts_by_player: dict[int, list[decisions.AcceptExchangeDecision]] = {}

    def __call__[C: decisions.Choice](
        self, _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            self.accepts_by_player.setdefault(decision.player_id, []).append(decision)
            accept = (
                self._accept
                if isinstance(self._accept, bool)
                else self._accept[decision.player_id]
            )
            wanted = [
                choice
                for choice in decision.choices
                if isinstance(choice, decisions.SkipChoice) != accept
            ]
            return typing.cast(C, wanted[0])
        for choice in decision.choices:
            if not isinstance(choice, decisions.SkipChoice):
                return typing.cast(C, choice)
        return decision.choices[0]

    def prompt_count(self, player_id: int) -> int:
        return len(self.accepts_by_player.get(player_id, []))


def _bowl_bird() -> cards.Bird:
    birds, _, _ = cards.load_all()
    return next(
        bird for bird in birds if bird.nest == NEST and bird.egg_limit >= EGG_ROOM_MIN
    )


def _game(*, goal: bool, num_players: int = 2) -> state.GameState:
    birds, bonuses, goals = cards.load_all()
    game = state.new_game(
        random.Random(5), birds, bonuses, goals, num_players=num_players
    )
    game.current_player = 0
    for player in game.players:
        for habitat in cards.ALL_HABITATS:
            player.board[habitat].clear()
    if goal:
        game.round_goals[game.round_idx] = cards.EndRoundGoal(
            id=99,
            description="birds without eggs",
            category="birds_no_eggs",
            tile_id=99,
        )
    return game


def _give_birds(player: state.Player, count: int) -> list[state.PlayedBird]:
    bird = _bowl_bird()
    played = [state.PlayedBird(bird=bird, eggs=0) for _ in range(count)]
    player.board[bird.habitats[0]] = played
    return played


def _power_bird(name: str = "Lazuli Bunting") -> tuple[cards.Bird, cards.Effect]:
    birds, _, _ = cards.load_all()
    bird = next(candidate for candidate in birds if candidate.name == name)
    eff = next(
        effect
        for effect in bird.power.effects
        if effect.kind == cards.EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST
    )
    return bird, eff


def _run_all_players(game: state.GameState, agent: _Script) -> None:
    bird, eff = _power_bird()
    pb_power = state.PlayedBird(bird=bird)
    eng = engine.Engine(game, agents=[agent] * len(game.players))
    powers.apply_effect(
        eng, agent, game.players[0], pb_power, bird.habitats[0], eff, "play"
    )


def _eggs(player: state.Player) -> int:
    return sum(pb.eggs for row in player.board.values() for pb in row)


# ---------------------------------------------------------------------------
# (a) ALL_PLAYERS_LAY_EGG_ON_NEST


def test_all_players_accept_each_player_asked_once_per_egg() -> None:
    game = _game(goal=True)
    _give_birds(game.players[0], 2)
    _give_birds(game.players[1], 1)
    agent = _Script(accept=True)
    _run_all_players(game, agent)

    # Active player: activation veto (covers the base egg) + one gate for the
    # extra egg = 2 prompts for 2 eggs. Opponent: 1 prompt for 1 egg.
    assert agent.prompt_count(0) == 2
    assert agent.prompt_count(1) == 1
    assert _eggs(game.players[0]) == 2
    assert _eggs(game.players[1]) == 1


def test_all_players_opponent_skip_means_no_egg_and_no_second_prompt() -> None:
    game = _game(goal=True)
    _give_birds(game.players[0], 2)
    _give_birds(game.players[1], 1)
    agent = _Script(accept={0: True, 1: False})
    _run_all_players(game, agent)

    assert agent.prompt_count(1) == 1
    assert _eggs(game.players[1]) == 0
    assert _eggs(game.players[0]) == 2


def test_all_players_extra_egg_skip_means_base_egg_only() -> None:
    game = _game(goal=True)
    _give_birds(game.players[0], 2)
    seen: list[int] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        if isinstance(decision, decisions.AcceptExchangeDecision):
            seen.append(decision.player_id)
            # Accept the activation veto (first), skip every later gate.
            index = 0 if len(seen) == 1 else 1
            return typing.cast(C, decision.choices[index])
        return decision.choices[0]

    bird, eff = _power_bird()
    eng = engine.Engine(game, agents=[agent, agent])
    powers.apply_effect(
        eng,
        agent,
        game.players[0],
        state.PlayedBird(bird=bird),
        bird.habitats[0],
        eff,
        "play",
    )
    assert seen == [0, 0]
    assert _eggs(game.players[0]) == 1


def test_all_players_no_goal_asks_only_the_activation_veto() -> None:
    game = _game(goal=False)
    _give_birds(game.players[0], 2)
    _give_birds(game.players[1], 1)
    agent = _Script(accept=True)
    _run_all_players(game, agent)

    assert agent.prompt_count(0) == 1
    assert agent.prompt_count(1) == 0
    assert _eggs(game.players[0]) == 2
    assert _eggs(game.players[1]) == 1


def test_all_players_three_players_each_opponent_asked_once() -> None:
    game = _game(goal=True, num_players=3)
    for player in game.players:
        _give_birds(player, 2)
    agent = _Script(accept=True)
    _run_all_players(game, agent)

    assert agent.prompt_count(1) == 1
    assert agent.prompt_count(2) == 1
    assert [_eggs(player) for player in game.players] == [2, 1, 1]


def test_lay_one_egg_on_nest_already_consented_skips_gate() -> None:
    game = _game(goal=True)
    (pb,) = _give_birds(game.players[0], 1)
    agent = _Script(accept=False)
    eng = engine.Engine(game, agents=[agent, agent])

    laid = powers.lay_one_egg_on_nest(
        eng, game.players[0], NEST, label="t", already_consented=True
    )
    assert laid is pb and pb.eggs == 1
    assert agent.prompt_count(0) == 0

    declined = powers.lay_one_egg_on_nest(eng, game.players[0], NEST, label="t")
    assert declined is None and pb.eggs == 1
    assert agent.prompt_count(0) == 1


# ---------------------------------------------------------------------------
# (b) every other egg-laying site: exactly one gate, first answer decides


def _single_site_game(goal: bool) -> tuple[state.GameState, state.PlayedBird]:
    game = _game(goal=goal)
    (pb,) = _give_birds(game.players[0], 1)
    return game, pb


def _apply(
    game: state.GameState, agent: _Script, pb: state.PlayedBird, eff: cards.Effect
) -> None:
    eng = engine.Engine(game, agents=[agent, agent])
    powers.apply_effect(
        eng, agent, game.players[0], pb, pb.bird.habitats[0], eff, "activate"
    )


@pytest.mark.parametrize("accept", [True, False])
def test_lay_egg_on_this_single_gate(accept: bool) -> None:
    game, pb = _single_site_game(goal=True)
    agent = _Script(accept=accept)
    eff = cards.Effect(kind=cards.EffectKind.LAY_EGG_ON_THIS, amount=1)
    _apply(game, agent, pb, eff)
    assert agent.prompt_count(0) == 1
    assert pb.eggs == (1 if accept else 0)


@pytest.mark.parametrize("accept", [True, False])
def test_lay_egg_any_single_gate(accept: bool) -> None:
    game, pb = _single_site_game(goal=True)
    agent = _Script(accept=accept)
    _apply(game, agent, pb, cards.Effect(kind=cards.EffectKind.LAY_EGG_ANY, amount=1))
    assert agent.prompt_count(0) == 1
    assert pb.eggs == (1 if accept else 0)


@pytest.mark.parametrize("accept", [True, False])
def test_lay_egg_all_nest_single_gate(accept: bool) -> None:
    game, pb = _single_site_game(goal=True)
    agent = _Script(accept=accept)
    eff = cards.Effect(kind=cards.EffectKind.LAY_EGG_ALL_NEST, amount=1, nest=NEST)
    _apply(game, agent, pb, eff)
    assert agent.prompt_count(0) == 1
    assert pb.eggs == (1 if accept else 0)


def _pink_effect() -> cards.Effect:
    return cards.Effect(
        kind=cards.EffectKind.PINK_LAY_EGG_ON_NEST,
        amount=1,
        nest=NEST,
        exclude_self=False,
    )


@pytest.mark.parametrize("accept", [True, False])
def test_pink_lay_egg_reactor_single_gate(accept: bool) -> None:
    game, pb = _single_site_game(goal=True)
    agent = _Script(accept=accept)
    eng = engine.Engine(game, agents=[agent, agent])
    laid = reactors.fire_pink_lay_egg(
        eng, game.players[0], pb, pb.bird.habitats[0], _pink_effect()
    )
    assert laid is accept
    assert agent.prompt_count(0) == 1
    assert pb.eggs == (1 if accept else 0)


@pytest.mark.parametrize(
    ("accept_tuck", "accept_egg"), [(True, True), (True, False), (False, False)]
)
def test_tuck_then_lay_on_this_keeps_its_distinct_egg_gate(
    accept_tuck: bool, accept_egg: bool
) -> None:
    """The egg gate is not a duplicate of the tuck gate: tucking does not decide
    the optional egg, so under birds_no_eggs each is asked exactly once, and
    the egg gate is never reached when the tuck is declined."""
    game, pb = _single_site_game(goal=True)
    player = game.players[0]
    birds, _, _ = cards.load_all()
    player.hand = [birds[0]]
    tuck_gates: list[decisions.ActivateTuckDecision] = []
    egg_gates: list[decisions.AcceptExchangeDecision] = []

    def agent[C: decisions.Choice](
        _eng: engine.Engine, decision: decisions.Decision[C]
    ) -> C:
        if isinstance(decision, decisions.ActivateTuckDecision):
            tuck_gates.append(decision)
            return typing.cast(C, decision.choices[0 if accept_tuck else 1])
        if isinstance(decision, decisions.AcceptExchangeDecision):
            egg_gates.append(decision)
            return typing.cast(C, decision.choices[0 if accept_egg else 1])
        return decision.choices[0]

    eng = engine.Engine(game, agents=[agent, agent])
    eff = cards.Effect(kind=cards.EffectKind.TUCK_FROM_HAND_THEN_LAY_ON_THIS, amount=1)
    powers.apply_effect(eng, agent, player, pb, pb.bird.habitats[0], eff, "activate")
    assert len(tuck_gates) == 1
    assert len(egg_gates) == (1 if accept_tuck else 0)
    assert pb.tucked_cards == (1 if accept_tuck else 0)
    assert pb.eggs == (1 if accept_tuck and accept_egg else 0)


@pytest.mark.parametrize(
    "kind",
    [
        cards.EffectKind.LAY_EGG_ON_THIS,
        cards.EffectKind.LAY_EGG_ANY,
        cards.EffectKind.LAY_EGG_ALL_NEST,
    ],
)
def test_no_goal_no_prompts_and_egg_laid(kind: cards.EffectKind) -> None:
    game, pb = _single_site_game(goal=False)
    agent = _Script(accept=False)
    _apply(game, agent, pb, cards.Effect(kind=kind, amount=1, nest=NEST))
    assert agent.prompt_count(0) == 0
    assert pb.eggs == 1


def test_no_goal_pink_reactor_no_prompt() -> None:
    game, pb = _single_site_game(goal=False)
    agent = _Script(accept=False)
    eng = engine.Engine(game, agents=[agent, agent])
    assert reactors.fire_pink_lay_egg(
        eng, game.players[0], pb, pb.bird.habitats[0], _pink_effect()
    )
    assert agent.prompt_count(0) == 0


# ---------------------------------------------------------------------------
# opp_gained_egg_count convention


@pytest.mark.parametrize("opponents_with_room, expected", [(0, 0), (1, 1), (2, 2)])
def test_all_players_activation_row_counts_eligible_opponents(
    opponents_with_room: int, expected: int
) -> None:
    game = _game(goal=False, num_players=3)
    _give_birds(game.players[0], 1)
    for player in game.players[1 : 1 + opponents_with_room]:
        _give_birds(player, 1)
    agent = _Script(accept=False)
    _run_all_players(game, agent)
    row = agent.accepts_by_player[0][0].choices[0]
    assert isinstance(row, decisions.PayCostChoice)
    assert row.opp_gained_egg_count == expected
    assert row.gained_egg_count == 1


def test_lay_egg_all_nest_row_is_self_only() -> None:
    game, pb = _single_site_game(goal=True)
    agent = _Script(accept=False)
    eff = cards.Effect(kind=cards.EffectKind.LAY_EGG_ALL_NEST, amount=1, nest=NEST)
    _apply(game, agent, pb, eff)
    row = agent.accepts_by_player[0][0].choices[0]
    assert isinstance(row, decisions.PayCostChoice)
    assert row.gained_egg_count == 1
    assert row.opp_gained_egg_count == 0
