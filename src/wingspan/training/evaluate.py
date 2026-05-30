"""Out-of-sample evaluation against a fixed reference opponent (TRAINING.md §7).

Self-play win rate is ~50% by symmetry and measures nothing, so strength is
measured against a *fixed* reference opponent with the policy in **greedy** mode
(argmax, no sampling: we are measuring strength, not exploring). The reference
opponent is the random agent at first, and a frozen past self once the policy
has learned to crush the random agent (the opponent is advanced by the training
loop, see ``config.opponent_reset_win_rate``); ``opponent_net=None`` selects the
random agent, any other net plays its own greedy policy.

Variance is controlled with **paired (mirror) games**: each deal is played
twice on the same seed, once with the policy as player 0 and once with the
seats swapped, which cancels Wingspan's real first-player / deal advantage. The
result carries a 95% confidence interval (normal approximation, ties counting
as half a win) so a 55% rate is not mistaken for 50%.
"""

from __future__ import annotations

import math
import random
import typing

import torch

from wingspan import agents, decisions, encode, engine, model
from wingspan.training import collect, metrics, policy

_Z_95 = 1.96

# Called after each held-out eval game with (games_done, total_games) so a
# caller can drive a live progress bar; the eval result itself is unaffected.
type EvalProgress = typing.Callable[[int, int], None]


def evaluate_vs_opponent(
    net: model.PolicyValueNet,
    opponent_net: model.PolicyValueNet | None,
    device: torch.device,
    n_pairs: int,
    seed: int,
    opponent_generation: int = 0,
    on_progress: EvalProgress | None = None,
) -> metrics.EvalResult:
    """Play ``n_pairs`` mirrored deals against the reference opponent and
    summarize. ``opponent_net=None`` plays against the random agent; any other
    net plays its own greedy policy (a frozen past self).

    Returns an :class:`metrics.EvalResult` with the greedy policy's win rate,
    its 95% CI half-width, mean score margin over ``2 * n_pairs`` games, and the
    ``opponent_generation`` it was played against. ``on_progress``, if given, is
    called after every game with the running ``(games_done, total_games)`` so
    the dashboard can track eval progress.
    """
    n_games = 2 * n_pairs
    margins: list[int] = []
    wins = 0.0
    games_done = 0
    for pair in range(n_pairs):
        pair_seed = seed + pair * 2
        for net_seat in (0, 1):
            margin = _play_eval_game(net, opponent_net, device, pair_seed, net_seat)
            margins.append(margin)
            wins += 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
            games_done += 1
            if on_progress is not None:
                on_progress(games_done, n_games)

    if n_games == 0:
        return metrics.EvalResult(
            n_games=0,
            win_rate=0.0,
            ci95=0.0,
            mean_margin=0.0,
            opponent_generation=opponent_generation,
        )
    win_rate = wins / n_games
    ci95 = _Z_95 * math.sqrt(max(win_rate * (1.0 - win_rate), 0.0) / n_games)
    return metrics.EvalResult(
        n_games=n_games,
        win_rate=win_rate,
        ci95=ci95,
        mean_margin=sum(margins) / len(margins),
        opponent_generation=opponent_generation,
    )


###### PRIVATE #######


def _play_eval_game(
    net: model.PolicyValueNet,
    opponent_net: model.PolicyValueNet | None,
    device: torch.device,
    seed: int,
    net_seat: int,
) -> int:
    """Play one greedy-policy-vs-opponent game on ``seed`` with the policy in
    ``net_seat``; return the policy's score margin (its score − opponent's). The
    opponent is the random agent when ``opponent_net is None``, otherwise that
    net's own greedy policy."""
    eng = collect.new_engine(seed)
    net_agent = _greedy_agent(net, device)
    if opponent_net is None:
        opponent_agent: engine.Agent = agents.random_agent(
            random.Random(seed * 2 + net_seat + 1)
        )
    else:
        opponent_agent = _greedy_agent(opponent_net, device)
    seats: list[engine.Agent] = [opponent_agent, opponent_agent]
    seats[net_seat] = net_agent
    engine.Engine.play_one_game(eng.state, (seats[0], seats[1]))

    net_score = eng.state.players[net_seat].final_score or 0
    opp_score = eng.state.players[1 - net_seat].final_score or 0
    return net_score - opp_score


def _greedy_agent(net: model.PolicyValueNet, device: torch.device) -> engine.Agent:
    """A non-recording agent that plays the argmax of the current policy."""

    def agent[C: decisions.Choice](
        eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if len(decision.choices) == 1:
            return decision.choices[0]
        family_idx = decisions.family_index_for(type(decision))
        state_vec = encode.encode_state(eng.state, decision)
        choice_feats = encode.encode_choices(decision, eng.state)
        idx = policy.greedy_action(net, device, state_vec, choice_feats, family_idx)
        return decision.choices[idx]

    return agent
