"""Self-play data collection.

Plays one full self-play game where both seats consult the same network, and
records every multi-option decision as a :class:`wingspan.train.Step` (state
features, candidate features, chosen index, player id, judgment-family head
index). After the game it reads each player's final board into a
:class:`metrics.ScoreBreakdown`, so the loop can both train on the trajectory
and report the score's six-way split live.

Single-option forced moves are not recorded — the trainable surface is the
moments with a genuine fork (DECISIONS.md §1.4).

The bundled card catalog is parsed once and reused across games (the card
models are immutable and ``state.new_game`` copies the deck lists before
shuffling), which avoids re-reading the JSON on every game — the dominant
fixed cost of ``Engine.create``.
"""

from __future__ import annotations

import functools
import random

import pydantic
import torch

from wingspan import cards, decisions, encode, engine, model, state, train
from wingspan.engine import scoring
from wingspan.training import metrics, policy


class GameRecord(pydantic.BaseModel):
    """One finished self-play game: its recorded steps plus the per-player
    final score breakdown, the winner (0, 1, or -1 for a tie), and the
    board-shuffle ``seed`` that produced it (carried so the persisted per-game
    history row stays independently reproducible)."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    steps: list[train.Step]
    breakdowns: tuple[metrics.ScoreBreakdown, metrics.ScoreBreakdown]
    winner: int
    seed: int

    @property
    def scores(self) -> tuple[int, int]:
        return (round(self.breakdowns[0].total), round(self.breakdowns[1].total))


def play_game(
    net: model.PolicyValueNet,
    device: torch.device,
    rng: random.Random,
    seed: int,
    opponent_agent: engine.Agent | None = None,
) -> GameRecord:
    """Play one game and return its recorded transitions + scores.

    With ``opponent_agent`` omitted this is ordinary self-play: both seats
    consult the policy and every multi-option decision is recorded. With an
    ``opponent_agent`` (the random-opponent bootstrap phase), the net plays
    seat 0 and ``opponent_agent`` plays seat 1; only the net's decisions are
    recorded, since the opponent's off-policy moves are not trained on.
    """
    eng = new_engine(seed)
    recorded: list[train.Step] = []
    net_agent = _recording_agent(net, device, rng, recorded)
    agent_a, agent_b = (
        (net_agent, net_agent)
        if opponent_agent is None
        else (net_agent, opponent_agent)
    )
    engine.Engine.play_one_game(eng.state, (agent_a, agent_b))

    breakdowns = (
        player_breakdown(eng.state.players[0]),
        player_breakdown(eng.state.players[1]),
    )
    score_0, score_1 = breakdowns[0].total, breakdowns[1].total
    winner = 0 if score_0 > score_1 else (1 if score_1 > score_0 else -1)
    return GameRecord(steps=recorded, breakdowns=breakdowns, winner=winner, seed=seed)


def player_breakdown(player: state.Player) -> metrics.ScoreBreakdown:
    """Split ``player``'s final score into its six sources — the exact terms
    ``engine.scoring.final_scoring`` sums (birds + bonus + eggs + tucked +
    cached-food + round-goal)."""
    bird_pts = sum(pb.bird.points for row in player.board.values() for pb in row)
    bonus_pts = sum(scoring.bonus_score(player, bc) for bc in player.bonus_cards)
    return metrics.ScoreBreakdown(
        birds=float(bird_pts),
        eggs=float(player.total_eggs),
        food=float(player.total_cached),
        tucked=float(player.total_tucked),
        rounds=float(player.round_goal_points),
        bonus=float(bonus_pts),
    )


def new_engine(seed: int) -> engine.Engine:
    """Construct a fresh game engine on a seeded shuffle of the cached catalog."""
    birds, bonuses, goals = _catalog()
    game = state.new_game(random.Random(seed), list(birds), list(bonuses), list(goals))
    return engine.Engine(game)


###### PRIVATE #######


@functools.lru_cache(maxsize=1)
def _catalog() -> (
    tuple[list[cards.Bird], list[cards.BonusCard], list[cards.EndRoundGoal]]
):
    """Parse the bundled card catalog once and reuse it across every game."""
    return cards.load_all()


def _recording_agent(
    net: model.PolicyValueNet,
    device: torch.device,
    rng: random.Random,
    record_into: list[train.Step],
) -> engine.Agent:
    """An agent that samples from the policy and appends every multi-option
    decision it makes to ``record_into`` (both seats share the buffer, tagged
    by ``player_id``)."""

    def agent[C: decisions.Choice](
        eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if len(decision.choices) == 1:
            return decision.choices[0]
        family_idx = decisions.family_index_for(type(decision))
        state_vec = encode.encode_state(eng.state, decision)
        choice_feats = encode.encode_choices(decision, eng.state)
        chosen_idx = policy.sample_action(
            net, device, state_vec, choice_feats, family_idx, rng
        )
        record_into.append(
            train.Step(
                state=state_vec,
                choices=choice_feats,
                chosen_idx=chosen_idx,
                player_id=decision.player_id,
                family_idx=family_idx,
            )
        )
        return decision.choices[chosen_idx]

    return agent
