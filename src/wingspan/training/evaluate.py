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

The per-game unit (:func:`play_eval_game`) and the summarizer
(:func:`summarize_eval`) are public so the process-parallel eval path
(:meth:`mp_collect.ProcessCollector.evaluate_games`) can reuse the *exact* same
game and statistics, guaranteeing the parallel result is identical to the
sequential one game-for-game.
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
    for pair in range(n_pairs):
        pair_seed = seed + pair * 2
        for net_seat in (0, 1):
            margins.append(
                play_eval_game(net, opponent_net, device, pair_seed, net_seat)
            )
            if on_progress is not None:
                on_progress(len(margins), n_games)
    return summarize_eval(margins, opponent_generation)


def play_eval_game(
    net: model.PolicyValueNet,
    opponent_net: model.PolicyValueNet | None,
    device: torch.device,
    seed: int,
    net_seat: int,
) -> int:
    """Play one greedy-policy-vs-opponent game on ``seed`` with the policy in
    ``net_seat``; return the policy's score margin (its score − opponent's). The
    opponent is the random agent when ``opponent_net is None``, otherwise that
    net's own greedy policy. Deterministic in ``(seed, net_seat)`` and the
    weights, so it returns the same margin in any process."""
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


def run_final_self_play_eval(
    net: model.PolicyValueNet,
    device: torch.device,
    n_games: int,
    seed: int,
    at_iteration: int,
    on_progress: EvalProgress | None = None,
) -> metrics.FinalEvalStats:
    """Play ``n_games`` of model-vs-itself (both seats greedy, model fixed).

    Pairs each seed so the same deal is played from both seat perspectives,
    cancelling first-player advantage in the breakdown averages. Returns a
    :class:`metrics.FinalEvalStats` with averaged score breakdowns and game
    stats for the IN-GAME PERFORMANCE pin — no EWMA, a clean snapshot of
    the model we "landed on".
    """
    n_pairs = n_games // 2
    actual_games = 2 * n_pairs

    breakdown_sum = metrics.ScoreBreakdown()
    winner_breakdown_sum = metrics.ScoreBreakdown()
    total_decisions = 0
    total_decided_games = 0
    margin_sum = 0.0
    seat0_wins = 0

    for pair in range(n_pairs):
        pair_seed = seed + pair * 2
        for flip in (0, 1):
            game_seed = pair_seed + flip
            decision_count: list[int] = [0]
            greedy = _counting_greedy_agent(net, device, decision_count)
            eng = collect.new_engine(game_seed)
            engine.Engine.play_one_game(eng.state, (greedy, greedy))
            bd0 = collect.player_breakdown(eng.state.players[0])
            bd1 = collect.player_breakdown(eng.state.players[1])
            breakdown_sum = breakdown_sum + bd0 + bd1
            score0, score1 = round(bd0.total), round(bd1.total)
            if score0 > score1:
                seat0_wins += 1
                winner_breakdown_sum = winner_breakdown_sum + bd0
                total_decided_games += 1
            elif score1 > score0:
                winner_breakdown_sum = winner_breakdown_sum + bd1
                total_decided_games += 1
            margin_sum += abs(score0 - score1)
            total_decisions += decision_count[0]
            if on_progress is not None:
                on_progress(pair * 2 + flip + 1, actual_games)

    player_games = max(actual_games * 2, 1)
    return metrics.FinalEvalStats(
        n_games=actual_games,
        avg_breakdown=breakdown_sum.scaled(1.0 / player_games),
        avg_winner_breakdown=winner_breakdown_sum.scaled(
            1.0 / max(total_decided_games, 1)
        ),
        decisions_per_game=total_decisions / max(actual_games, 1),
        mean_margin=margin_sum / max(actual_games, 1),
        self_play_win_rate=seat0_wins / max(actual_games, 1),
        at_iteration=at_iteration,
    )


def summarize_eval(
    margins: typing.Sequence[int], opponent_generation: int
) -> metrics.EvalResult:
    """Roll per-game score margins (policy − opponent) into an
    :class:`metrics.EvalResult`: win rate with ties counting as half a win, the
    95% CI half-width (normal approximation ``p ± 1.96·√(p(1−p)/n)``), and the
    mean margin. Shared by the sequential and process-parallel eval paths so the
    two report identical statistics from the same games."""
    n_games = len(margins)
    if n_games == 0:
        return metrics.EvalResult(
            n_games=0,
            win_rate=0.0,
            ci95=0.0,
            mean_margin=0.0,
            opponent_generation=opponent_generation,
        )
    wins = sum(
        1.0 if margin > 0 else (0.5 if margin == 0 else 0.0) for margin in margins
    )
    win_rate = wins / n_games
    ci95 = _Z_95 * math.sqrt(max(win_rate * (1.0 - win_rate), 0.0) / n_games)
    return metrics.EvalResult(
        n_games=n_games,
        win_rate=win_rate,
        ci95=ci95,
        mean_margin=sum(margins) / n_games,
        opponent_generation=opponent_generation,
    )


###### PRIVATE #######


def _greedy_agent(net: model.PolicyValueNet, device: torch.device) -> engine.Agent:
    """A non-recording agent that plays the argmax of the current policy."""

    def agent[C: decisions.Choice](
        eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if len(decision.choices) == 1:
            return decision.choices[0]
        if not net.include_setup and decisions.is_setup_decision(decision):
            return decisions.random_choice(decision, eng.state.rng)
        family_idx = decisions.family_index_for(type(decision))
        state_vec = encode.encode_state(eng.state, decision, net.spec)
        choice_feats = encode.encode_choices(decision, eng.state, net.spec)
        idx = policy.greedy_action(net, device, state_vec, choice_feats, family_idx)
        return decision.choices[idx]

    return agent


def _counting_greedy_agent(
    net: model.PolicyValueNet,
    device: torch.device,
    counter: list[int],
) -> engine.Agent:
    """A greedy agent that increments ``counter[0]`` for every multi-option decision."""

    def agent[C: decisions.Choice](
        eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if len(decision.choices) == 1:
            return decision.choices[0]
        if not net.include_setup and decisions.is_setup_decision(decision):
            return decisions.random_choice(decision, eng.state.rng)
        counter[0] += 1
        family_idx = decisions.family_index_for(type(decision))
        state_vec = encode.encode_state(eng.state, decision, net.spec)
        choice_feats = encode.encode_choices(decision, eng.state, net.spec)
        idx = policy.greedy_action(net, device, state_vec, choice_feats, family_idx)
        return decision.choices[idx]

    return agent
