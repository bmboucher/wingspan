"""Tests for row-activation visibility in the detailed game log.

Every bird in an activated habitat row must leave a log header (its power text
for brown powers, a "no brown power" note otherwise), and any decision resolved
without consulting the agent must leave a "skipping decision, ..." line —
"no choices" when a handler's choice list is empty, "only 1 choice: <label>"
when ``Engine.ask`` auto-resolves a forced decision.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import cards, decisions, engine, state
from wingspan.engine import actions, powers

# ---------------------------------------------------------------------------
# Shared fixtures


def _raising_agent[C: decisions.Choice](
    _eng: engine.Engine, decision: decisions.Decision[C]
) -> C:
    raise AssertionError(f"unexpected agent consultation: {type(decision).__name__}")


def _engine_with_agents(seed: int = 0) -> tuple[engine.Engine, list[cards.Bird]]:
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(seed)
    gs = state.new_game(rng, birds, bonuses, goals)
    return engine.Engine(gs, agents=[_raising_agent, _raising_agent]), birds


def _find_by_effect(
    birds: list[cards.Bird], kind: cards.EffectKind, color: cards.PowerColor
) -> cards.Bird:
    return next(
        bird
        for bird in birds
        if bird.color == color and any(eff.kind == kind for eff in bird.power.effects)
    )


# ---------------------------------------------------------------------------
# plain_power_text


def test_plain_power_text_strips_icon_brackets():
    """``Bird.plain_power_text`` renders ``[card]``-style tags to plain words."""
    _, birds = _engine_with_agents()
    coot = next(bird for bird in birds if bird.name == "American Coot")
    assert (
        coot.plain_power_text
        == "Tuck 1 card from your hand behind this bird. If you do, draw 1 card."
    )
    assert "[" not in coot.plain_power_text


# ---------------------------------------------------------------------------
# Per-bird activation headers


def test_row_activation_logs_header_for_every_bird_rightmost_first():
    """Activating a row logs one ``@`` header per bird, right-to-left: the
    power text for a brown bird, "no brown power" otherwise."""
    eng, birds = _engine_with_agents()
    player = eng.state.players[0]
    # A brown power that resolves without any decision (lays on itself), so the
    # raising agent proves no consultation happens.
    brown = _find_by_effect(
        birds, cards.EffectKind.LAY_EGG_ON_THIS, cards.PowerColor.BROWN
    )
    white = next(bird for bird in birds if bird.color == cards.PowerColor.WHITE)
    brown_pb = state.PlayedBird(bird=brown)
    white_pb = state.PlayedBird(bird=white)
    player.board[cards.Habitat.FOREST] = [brown_pb, white_pb]

    actions.activate_row_powers(eng, _raising_agent, player, cards.Habitat.FOREST)

    headers = [line for line in eng.state.log if "] @ " in line]
    assert headers == [
        f"[P0] @ {white.name} - no brown power",
        f'[P0] @ {brown.name} - "{brown.plain_power_text}"',
    ], f"unexpected activation headers: {headers}"
    # Only the brown bird counts as activated.
    assert brown_pb.activations == 1
    assert white_pb.activations == 0


# ---------------------------------------------------------------------------
# "skipping decision, no choices"


def test_tuck_power_with_empty_hand_logs_no_choices():
    """A tuck power activated with an empty hand logs the no-choices skip
    instead of fizzling silently."""
    eng, birds = _engine_with_agents()
    player = eng.state.players[0]
    coot = next(bird for bird in birds if bird.name == "American Coot")
    pb = state.PlayedBird(bird=coot)
    player.board[cards.Habitat.WETLAND] = [pb]
    player.hand = []

    powers.dispatch_power(
        eng, _raising_agent, player, pb, cards.Habitat.WETLAND, "activate"
    )

    assert "[P0] skipping decision, no choices" in eng.state.log
    assert pb.tucked_cards == 0


def test_lay_one_egg_with_no_room_logs_no_choices():
    """``lay_one_egg`` on a board with no open egg slot logs the skip."""
    eng, birds = _engine_with_agents()
    player = eng.state.players[0]
    bird = birds[0]
    pb = state.PlayedBird(bird=bird, eggs=bird.egg_limit)
    player.board[cards.Habitat.GRASSLAND] = [pb]

    actions.lay_one_egg(eng, _raising_agent, player)

    assert "[P0] skipping decision, no choices" in eng.state.log
    assert pb.eggs == bird.egg_limit


# ---------------------------------------------------------------------------
# "skipping decision, only 1 choice"


def test_ask_single_choice_logs_label_and_skips_agent():
    """``Engine.ask`` resolves a one-choice decision without the agent and logs
    the auto-picked choice's label."""
    eng, _ = _engine_with_agents()
    only = decisions.BoardTargetChoice(
        label="Purple Martin@grassland[3](2/3)",
        habitat=cards.Habitat.GRASSLAND,
        slot=3,
    )
    decision = decisions.LayEggDecision(
        player_id=0,
        prompt="[P0] lay 1 egg",
        choices=[only],
    )

    chosen = eng.ask(_raising_agent, decision)

    assert chosen is only
    assert (
        "[P0] skipping decision, only 1 choice: Purple Martin@grassland[3](2/3)"
        in eng.state.log
    )


def test_forced_feeder_take_logs_only_one_choice():
    """``take_one_from_feeder`` with a single allowed option routes through
    ``ask`` and logs the forced pick. The feeder shows two faces so the
    entry point's internal reset offer stays silent."""
    eng, _ = _engine_with_agents()
    player = eng.state.players[0]
    eng.state.birdfeeder.counts = state.FoodPool.from_dict(
        {cards.Food.RODENT: 2, cards.Food.FISH: 1}
    )
    eng.state.birdfeeder.choice_dice = 0

    gained = actions.take_one_from_feeder(
        eng,
        _raising_agent,
        player,
        prompt=f"[{player.name}] pick 1 from birdfeeder",
        allowed=[cards.Food.RODENT],
    )

    assert gained == cards.Food.RODENT
    skip_lines = [
        line for line in eng.state.log if "skipping decision, only 1 choice" in line
    ]
    assert len(skip_lines) == 1
    assert player.food[cards.Food.RODENT] == 1
