"""Equivalence test for the fixed-setup engine path.

``Engine.play_one_game_with_setups`` (the setup-model path, which applies
pre-decided keeps via a chooser without asking) must be equivalent to
``Engine.play_one_game`` driven by an agent that picks the same setup keep: with
the same deal and the same in-game RNG, both produce an identical game (same
final scores), confirming the fixed path deals and applies setup exactly like the
ask path.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import agents, cards, decisions, engine  # noqa: E402
from wingspan.setup_model import candidates  # noqa: E402
from wingspan.training import collect  # noqa: E402

_PICK_INDEX = 137  # an arbitrary candidate (< 504), the same in both paths
_SEED = 99
_SEAT_SEEDS = (100, 101)  # per-seat in-game RNG, identical across both runs


def _final_scores(eng: engine.Engine) -> tuple[int, int]:
    return (
        eng.state.players[0].final_score or 0,
        eng.state.players[1].final_score or 0,
    )


def test_fixed_setup_matches_ask_path():
    # Fixed path: a chooser returns the _PICK_INDEX candidate for each seat; both
    # seats play with seeded random agents for the in-game decisions.
    fixed = collect.new_engine(_SEED)
    fixed_agents = (
        agents.random_agent(random.Random(_SEAT_SEEDS[0])),
        agents.random_agent(random.Random(_SEAT_SEEDS[1])),
    )

    def chooser(
        eng: engine.Engine,
        dealt: tuple[tuple[list[cards.Bird], list[cards.BonusCard]], ...],
    ) -> list[candidates.SetupCandidate]:
        return [
            candidates.enumerate_setup_candidates(dealt[s][0], dealt[s][1])[_PICK_INDEX]
            for s in (0, 1)
        ]

    engine.Engine.play_one_game_with_setups(fixed.state, fixed_agents, chooser)

    # Ask path: each seat's agent picks the equally-indexed SetupChoice on its
    # first (setup) call without drawing from its RNG, then delegates in-game
    # decisions to a random agent on the same per-seat seed — so the in-game RNG
    # streams match the fixed run game-for-game.
    asked = collect.new_engine(_SEED)

    def seat_agent(seat_seed: int) -> engine.Agent:
        inner = agents.random_agent(random.Random(seat_seed))
        seen_setup = {"done": False}

        def agent[C: decisions.Choice](
            eng: engine.Engine, decision: decisions.Decision[C]
        ) -> C:
            if not seen_setup["done"]:
                seen_setup["done"] = True
                return decision.choices[_PICK_INDEX]
            return inner(eng, decision)

        return agent

    asked_agents = (seat_agent(_SEAT_SEEDS[0]), seat_agent(_SEAT_SEEDS[1]))
    engine.Engine.play_one_game(asked.state, asked_agents)

    assert _final_scores(fixed) == _final_scores(asked)
