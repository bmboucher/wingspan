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

from wingspan import agents, decisions, engine, model
from wingspan.engine import scoring
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
    num_players: int = 2,
    opponent_generation: int = 0,
    on_progress: EvalProgress | None = None,
    split_setup_bonus: bool = False,
    split_setup_food: bool = False,
    combine_gain_food: bool = False,
) -> metrics.EvalResult:
    """Play ``n_pairs`` mirrored deals against the reference opponent and
    summarize. ``opponent_net=None`` plays against the random agent; any other
    net plays its own greedy policy (a frozen past self).

    The net rotates through all ``num_players`` seats of the same deal (2 at
    the default), so one "pair" produces ``num_players`` games. Returns an
    :class:`metrics.EvalResult` with the greedy policy's win rate (mean
    :class:`metrics.EvalGameOutcome.win_credit`), its 95% CI half-width, mean
    score margin over ``num_players * n_pairs`` games, and the
    ``opponent_generation`` it was played against. ``on_progress``, if given, is
    called after every game with the running ``(games_done, total_games)`` so
    the dashboard can track eval progress. ``split_setup_bonus`` mirrors the
    collection regime — the opening bonus is picked by the in-game greedy head
    rather than baked into the random opening. ``split_setup_food`` analogously
    defers the opening food pick to sequential in-game food decisions.
    ``combine_gain_food`` mirrors the collection regime's collapsed food gains.
    """
    n_games = num_players * n_pairs
    outcomes: list[metrics.EvalGameOutcome] = []
    for pair in range(n_pairs):
        pair_seed = seed + pair * 2
        for net_seat in range(num_players):
            outcomes.append(
                play_eval_game(
                    net,
                    opponent_net,
                    device,
                    pair_seed,
                    net_seat,
                    num_players=num_players,
                    split_setup_bonus=split_setup_bonus,
                    split_setup_food=split_setup_food,
                    combine_gain_food=combine_gain_food,
                )
            )
            if on_progress is not None:
                on_progress(len(outcomes), n_games)
    return summarize_eval(outcomes, opponent_generation)


def play_eval_game(
    net: model.PolicyValueNet,
    opponent_net: model.PolicyValueNet | None,
    device: torch.device,
    seed: int,
    net_seat: int,
    num_players: int = 2,
    split_setup_bonus: bool = False,
    split_setup_food: bool = False,
    combine_gain_food: bool = False,
) -> metrics.EvalGameOutcome:
    """Play one greedy-policy-vs-opponent game on ``seed`` with the policy in
    ``net_seat`` (every other seat plays the opponent); return the policy's
    :class:`metrics.EvalGameOutcome` — its margin (score − best other seat's
    score) and win credit. The opponent is the random agent when
    ``opponent_net is None``, otherwise that net's own greedy policy.
    Deterministic in ``(seed, net_seat)`` and the weights, so it returns the
    same result in any process. ``split_setup_bonus`` defers the opening bonus
    pick to the in-game ``CHOOSE_BONUS`` head. ``split_setup_food`` defers the
    opening food pick to in-game food decisions. ``combine_gain_food``
    collapses multi-token food gains into one combined subset decision (all
    three mirror the collection regime).

    ``win_credit`` is computed from :func:`wingspan.engine.scoring.winners`:
    ``1.0`` when the net's seat is the sole winner, ``1/k`` on a ``k``-way
    shared victory that includes it, else ``0.0`` — at 2 players this matches
    the legacy "tie counts as half a win" convention, except a score tie can
    now resolve to a sole winner via the official food-supply tiebreak (the
    accepted 2P engine drift), so a ``margin == 0`` game is no longer
    automatically ``0.5`` credit."""
    eng = collect.new_engine(seed, num_players)
    net_agent = _greedy_agent(net, device)
    if opponent_net is None:
        opponent_agent: engine.Agent = agents.random_agent(
            random.Random(seed * 2 + net_seat + 1)
        )
    else:
        opponent_agent = _greedy_agent(opponent_net, device)
    seats: list[engine.Agent] = [opponent_agent] * num_players
    seats[net_seat] = net_agent
    engine.Engine.play_one_game(
        eng.state,
        seats,
        split_setup_bonus=split_setup_bonus,
        split_setup_food=split_setup_food,
        combine_gain_food=combine_gain_food,
    )

    net_score = eng.state.players[net_seat].final_score or 0
    best_other_score = max(
        player.final_score or 0 for player in eng.state.opponents_clockwise(net_seat)
    )
    winner_ids = scoring.winners(eng.state.players)
    win_credit = (1.0 / len(winner_ids)) if net_seat in winner_ids else 0.0
    return metrics.EvalGameOutcome(
        margin=float(net_score - best_other_score), win_credit=win_credit
    )


def run_final_self_play_eval(
    net: model.PolicyValueNet,
    device: torch.device,
    n_games: int,
    seed: int,
    at_iteration: int,
    num_players: int = 2,
    on_progress: EvalProgress | None = None,
    split_setup_bonus: bool = False,
    split_setup_food: bool = False,
    combine_gain_food: bool = False,
) -> metrics.FinalEvalStats:
    """Play ``n_games`` of model-vs-itself (every seat greedy, model fixed).

    Groups games into batches of ``num_players`` off consecutive seeds
    (``pair_seed + 0 .. num_players - 1``) so the reported stats sample
    ``num_players`` independent deals per batch — the direct generalization of
    the legacy 2-seat "pair" grouping (mirroring seats provides no variance
    reduction here, since every seat already runs the identical greedy
    policy). Returns a :class:`metrics.FinalEvalStats` with averaged score
    breakdowns and game stats for the IN-GAME PERFORMANCE pin — no EWMA, a
    clean snapshot of the model we "landed on". ``mean_margin`` is the average
    (top score − runner-up score) — exactly ``|s0 − s1|`` at 2 players.
    ``self_play_win_rate`` is seat 0's win share via
    :func:`wingspan.engine.scoring.determine_winner` (~0.5 at 2 players, ~1/N
    generally). ``split_setup_bonus`` defers the opening bonus pick to the
    in-game greedy ``CHOOSE_BONUS`` head, mirroring the collection regime.
    ``split_setup_food`` analogously defers the opening food pick.
    ``combine_gain_food`` collapses multi-token food gains into one combined
    subset decision.
    """
    n_pairs = n_games // num_players
    actual_games = num_players * n_pairs

    breakdown_sum = metrics.ScoreBreakdown()
    winner_breakdown_sum = metrics.ScoreBreakdown()
    total_decisions = 0
    total_decided_games = 0
    margin_sum = 0.0
    seat0_wins = 0

    for pair in range(n_pairs):
        pair_seed = seed + pair * num_players
        for rotation in range(num_players):
            game_seed = pair_seed + rotation
            decision_count: list[int] = [0]
            greedy = _counting_greedy_agent(net, device, decision_count)
            eng = collect.new_engine(game_seed, num_players)
            engine.Engine.play_one_game(
                eng.state,
                [greedy] * num_players,
                split_setup_bonus=split_setup_bonus,
                split_setup_food=split_setup_food,
                combine_gain_food=combine_gain_food,
            )
            breakdowns = [
                collect.player_breakdown(player) for player in eng.state.players
            ]
            for breakdown in breakdowns:
                breakdown_sum = breakdown_sum + breakdown
            scores = sorted((round(bd.total) for bd in breakdowns), reverse=True)
            margin_sum += scores[0] - scores[1]
            winner = scoring.determine_winner(eng.state.players)
            if winner >= 0:
                winner_breakdown_sum = winner_breakdown_sum + breakdowns[winner]
                total_decided_games += 1
            if winner == 0:
                seat0_wins += 1
            total_decisions += decision_count[0]
            if on_progress is not None:
                on_progress(pair * num_players + rotation + 1, actual_games)

    player_games = max(actual_games * num_players, 1)
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
    outcomes: typing.Sequence[metrics.EvalGameOutcome], opponent_generation: int
) -> metrics.EvalResult:
    """Roll per-game :class:`metrics.EvalGameOutcome`\\ s into an
    :class:`metrics.EvalResult`: win rate is the mean ``win_credit`` (already
    fractional per-game for ties / shared victories), with the 95% CI
    half-width (normal approximation ``p ± 1.96·√(p(1−p)/n)``), and the mean
    margin. Shared by the sequential and process-parallel eval paths so the
    two report identical statistics from the same games."""
    n_games = len(outcomes)
    if n_games == 0:
        return metrics.EvalResult(
            n_games=0,
            win_rate=0.0,
            ci95=0.0,
            mean_margin=0.0,
            opponent_generation=opponent_generation,
        )
    wins = sum(outcome.win_credit for outcome in outcomes)
    win_rate = wins / n_games
    ci95 = _Z_95 * math.sqrt(max(win_rate * (1.0 - win_rate), 0.0) / n_games)
    mean_margin = sum(outcome.margin for outcome in outcomes) / n_games
    return metrics.EvalResult(
        n_games=n_games,
        win_rate=win_rate,
        ci95=ci95,
        mean_margin=mean_margin,
        opponent_generation=opponent_generation,
    )


###### PRIVATE #######


def _greedy_agent(net: model.PolicyValueNet, device: torch.device) -> engine.Agent:
    """A non-recording agent that plays the argmax of the current policy.

    Delegates to :func:`policy.greedy_agent` (the single source of truth, shared
    with the tournament); kept as a thin private alias so this module's existing
    call sites read unchanged."""
    return policy.greedy_agent(net, device)


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
        state_vec = net.encode_state(eng.state, decision)
        choice_feats = net.encode_choices(decision, eng.state)
        idx = policy.greedy_action(net, device, state_vec, choice_feats, family_idx)
        return decision.choices[idx]

    return agent
