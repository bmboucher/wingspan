"""Metrics aggregation helpers for the training loop.

Pure free functions (no ``TrainingLoop`` state needed) for computing
per-iteration and per-game statistics from a collected batch.  Callers in
``loop.py`` and the sibling ``loop_*.py`` modules import this module and
call its functions directly.
"""

from __future__ import annotations

from wingspan import setup_model
from wingspan.training import collect, learner, metrics


def family_counts(record: collect.GameRecord) -> metrics.FamilyCounts:
    """Accumulate per-decision-family hit counts for one game."""
    counts = metrics.FamilyCounts()
    for step in record.steps:
        counts.bump(step.family_idx)
    return counts


def build_game_outcomes(
    records: list[collect.GameRecord], iteration: int
) -> list[metrics.GameOutcome]:
    """One persisted :class:`~metrics.GameOutcome` per finished game, tagged with
    the ``iteration`` that produced it — the rows appended to ``games.jsonl``."""
    return [
        metrics.GameOutcome(
            iteration=iteration,
            seed=record.seed,
            winner=record.winner,
            decisions=len(record.steps),
            breakdowns=record.breakdowns,
            family_counts=family_counts(record),
        )
        for record in records
    ]


def pop_std(sum_sq: float, mean: float, n: int) -> float:
    """Population σ from a Σx², a mean, and a sample count (clamped at 0)."""
    var = sum_sq / max(n, 1) - mean * mean
    return var**0.5 if var > 0.0 else 0.0


def avg_points(records: list[collect.GameRecord]) -> float:
    """Mean final score across both seats of every game in a collected batch."""
    if not records:
        return 0.0
    total = sum(rec.breakdowns[0].total + rec.breakdowns[1].total for rec in records)
    return total / (2 * len(records))


def collection_win_rate(records: list[collect.GameRecord]) -> float:
    """Win fraction for the net over a bootstrap-phase batch, ties as half.

    The net always plays seat 0 against the random agent, so ``winner == 0``
    is a net win and ``winner == -1`` is a tie.
    """
    if not records:
        return 0.0
    wins = sum(1 for record in records if record.winner == 0)
    ties = sum(1 for record in records if record.winner < 0)
    return (wins + 0.5 * ties) / len(records)


def build_iteration_metrics(
    iteration: int,
    total_games: int,
    records: list[collect.GameRecord],
    stats: learner.UpdateStats,
    eval_result: metrics.EvalResult | None,
    collect_seconds: float,
    update_seconds: float,
    eval_seconds: float,
    win_rate: float | None,
    setup_phase: collect.SetupPhase | None,
    setup_stats: metrics.SetupUpdateStats | None,
    imitation_phase: bool = False,
) -> metrics.IterationMetrics:
    """Aggregate one iteration's records + update stats into an :class:`~metrics.IterationMetrics` row."""
    n_games = len(records)
    sum_breakdown = metrics.ScoreBreakdown()
    winner_breakdown = metrics.ScoreBreakdown()
    decided_games = 0
    family = metrics.FamilyCounts()
    total_steps = 0
    total_steps_sq = 0
    margin_sum = 0.0
    margin_sq_sum = 0.0
    abs_margin_sum = 0.0
    self_score_sum = 0.0

    # Accumulate per-game stats into totals for normalization below.
    for record in records:
        sum_breakdown = sum_breakdown + record.breakdowns[0] + record.breakdowns[1]
        self_score_sum += record.breakdowns[0].total + record.breakdowns[1].total
        margin = record.breakdowns[0].total - record.breakdowns[1].total
        margin_sum += margin
        margin_sq_sum += margin * margin
        abs_margin_sum += abs(margin)
        if record.winner >= 0:
            winner_breakdown = winner_breakdown + record.breakdowns[record.winner]
            decided_games += 1
        steps = len(record.steps)
        total_steps += steps
        total_steps_sq += steps * steps
        family = family + family_counts(record)

    player_games = max(2 * n_games, 1)
    games = max(n_games, 1)
    margin_mean = margin_sum / games
    abs_margin_mean = abs_margin_sum / games

    # Per-cycle population σ over this iteration's games.
    return metrics.IterationMetrics(
        iteration=iteration,
        total_games=total_games,
        games_this_iter=n_games,
        loss=stats.loss,
        policy_loss=stats.policy_loss,
        value_loss=stats.value_loss,
        entropy=stats.entropy,
        grad_norm=stats.grad_norm,
        advantage_mean=stats.advantage_mean,
        advantage_std=stats.advantage_std,
        avg_self_score=self_score_sum / player_games,
        avg_margin=margin_mean,
        avg_breakdown=sum_breakdown.scaled(1.0 / player_games),
        avg_decisions=total_steps / games,
        avg_winner_breakdown=winner_breakdown.scaled(1.0 / max(decided_games, 1)),
        avg_abs_margin=abs_margin_mean,
        margin_std=pop_std(margin_sq_sum, margin_mean, games),
        abs_margin_std=pop_std(margin_sq_sum, abs_margin_mean, games),
        decisions_std=pop_std(total_steps_sq, total_steps / games, games),
        family_counts=family,
        collect_seconds=collect_seconds,
        update_seconds=update_seconds,
        eval_seconds=eval_seconds,
        games_per_sec=n_games / collect_seconds if collect_seconds > 0 else 0.0,
        eval=eval_result,
        collection_win_rate=win_rate,
        setup_phase=setup_phase.name if setup_phase is not None else None,
        setup_loss=setup_stats.loss if setup_stats is not None else None,
        setup_pred_margin_mean=(
            setup_stats.pred_margin_mean if setup_stats is not None else None
        ),
        setup_realized_margin_mean=(
            setup_stats.realized_margin_mean if setup_stats is not None else None
        ),
        setup_samples_recorded=(
            setup_stats.n_samples if setup_stats is not None else None
        ),
        imitation_loss=stats.imitation_loss if imitation_phase else None,
        clip_fraction=stats.clip_fraction,
        approx_kl=stats.approx_kl,
    )


def mean_setup_margin(samples: list[setup_model.SetupSample]) -> float:
    """Mean realized margin across a list of setup samples (0 if empty).

    The recording phase's readout, since it runs no optimizer step.
    """
    if not samples:
        return 0.0
    return sum(sample.margin for sample in samples) / len(samples)
