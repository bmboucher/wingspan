"""Metric records produced by the training loop and consumed by the dashboard.

Every metric is a Pydantic model so it serializes cleanly into ``metrics.jsonl``
and into the live snapshot the dashboard renders. Two value-objects do most of
the work:

* :class:`ScoreBreakdown` — a Wingspan final score split into its six scoring
  sources (birds / eggs / food / tucked / rounds / bonus). It supports ``+`` and
  ``scaled`` so the loop can keep a running sum and divide for an average.
* :class:`FamilyCounts` — a fixed-length vector of decision counts aligned to
  ``decisions.ALL_DECISION_FAMILIES``, with a dict-like surface (mirroring the
  ``FoodPool`` / ``BirdCost`` two-layer pattern: internal vector, named access).

:class:`IterationMetrics` is the per-iteration row (losses, averaged outcomes,
timings, optional eval block) appended to history and the metrics log.

:class:`SystemStats` is the live host telemetry (CPU / RAM) the SYSTEM band
renders — sampled ~once a second and held only on the live snapshot, never
folded into the per-iteration metrics log.
"""

from __future__ import annotations

import pydantic

from wingspan import decisions

# The six scoring sources, in the order the dashboard lists them. Maps the
# user-facing labels onto ``ScoreBreakdown`` field names.
SCORE_COMPONENTS: tuple[str, ...] = (
    "birds",
    "eggs",
    "food",
    "tucked",
    "rounds",
    "bonus",
)

_N_FAMILIES = len(decisions.ALL_DECISION_FAMILIES)


class ScoreBreakdown(pydantic.BaseModel):
    """A Wingspan score split into its six sources.

    The fields sum to the game total exactly as ``engine.scoring.final_scoring``
    computes it: birds + bonus + eggs + tucked + cached-food + round-goal. Fields
    are floats so the same model carries both a single integral game score and a
    fractional running average.
    """

    birds: float = 0.0
    eggs: float = 0.0
    food: float = 0.0  # cached-food points
    tucked: float = 0.0  # tucked-card points
    rounds: float = 0.0  # end-of-round-goal points
    bonus: float = 0.0  # bonus-card points

    @property
    def total(self) -> float:
        return (
            self.birds + self.eggs + self.food + self.tucked + self.rounds + self.bonus
        )

    def __add__(self, other: "ScoreBreakdown") -> "ScoreBreakdown":
        return ScoreBreakdown(
            birds=self.birds + other.birds,
            eggs=self.eggs + other.eggs,
            food=self.food + other.food,
            tucked=self.tucked + other.tucked,
            rounds=self.rounds + other.rounds,
            bonus=self.bonus + other.bonus,
        )

    def scaled(self, factor: float) -> "ScoreBreakdown":
        return ScoreBreakdown(
            birds=self.birds * factor,
            eggs=self.eggs * factor,
            food=self.food * factor,
            tucked=self.tucked * factor,
            rounds=self.rounds * factor,
            bonus=self.bonus * factor,
        )

    def components(self) -> list[tuple[str, float]]:
        """``(label, value)`` pairs in :data:`SCORE_COMPONENTS` order."""
        return [(name, getattr(self, name)) for name in SCORE_COMPONENTS]


class FamilyCounts(pydantic.BaseModel):
    """Decision counts per judgment family, aligned to
    ``decisions.ALL_DECISION_FAMILIES``.

    Internal storage is a fixed-length ``list[int]`` indexed by the family's
    position in that tuple (the same index the model routes a decision's scoring
    head through), with a small dict-like surface so call sites read naturally.
    """

    counts: list[int] = pydantic.Field(
        default_factory=lambda: [0] * _N_FAMILIES,
        min_length=_N_FAMILIES,
        max_length=_N_FAMILIES,
    )

    def bump(self, family_index: int) -> None:
        """Record one more decision routed to head ``family_index``."""
        self.counts[family_index] += 1

    def __add__(self, other: "FamilyCounts") -> "FamilyCounts":
        return FamilyCounts(counts=[a + b for a, b in zip(self.counts, other.counts)])

    def total(self) -> int:
        return sum(self.counts)

    def items(self) -> list[tuple[decisions.DecisionFamily, int]]:
        """``(family, count)`` pairs in stable head order."""
        return list(zip(decisions.ALL_DECISION_FAMILIES, self.counts))


class EvalResult(pydantic.BaseModel):
    """Outcome of a paired-game evaluation against the reference opponent (§7.3).

    ``win_rate`` and its ``ci95`` half-width use the normal approximation
    ``p ± 1.96·√(p(1−p)/n)``; ties count as half a win. ``mean_margin`` is the
    average score margin (policy − opponent) across the held-out games.
    ``opponent_generation`` tags which reference opponent the eval was played
    against (0 = the random agent; >0 = a frozen past self), so the dashboard's
    EWMA can reset to a fresh trend each time the opponent is advanced.
    """

    n_games: int
    win_rate: float
    ci95: float
    mean_margin: float
    opponent_generation: int = 0


class EvalEwma(pydantic.BaseModel):
    """Exponentially-weighted moving average of the eval win-rate and margin
    across successive evaluations.

    Each evaluation is only an ``eval_games`` sample, so a single win-rate
    bounces from one eval to the next; folding the series with a fixed decay
    (``config.eval_ewma_alpha``) gives the dashboard a steadier trend than any
    one eval estimate. ``win_rate`` is a 0..1 fraction; ``mean_margin`` is in
    points (policy − opponent).
    """

    win_rate: float
    mean_margin: float


class ProduceStats(pydantic.BaseModel):
    """The smoothed "what the AI is producing" readouts the PRODUCING band shows.

    Every field is an exponentially-weighted moving average folded once per
    finished iteration (``config.produce_ewma_alpha``), so the panel tracks the
    *current* policy rather than the diluted since-start average that barely
    moves once a long run has accumulated millions of games. ``breakdown`` is
    the average score split across *all* player-games; ``winner_breakdown`` is
    the same split conditioned on just the winning seat of each decided game, so
    the panel can show all-game and winners-only sources side by side.
    """

    breakdown: ScoreBreakdown  # avg six-way score split, all player-games
    winner_breakdown: ScoreBreakdown  # avg split of the winning seat only
    decisions: float  # avg trainable decisions per game
    margin: float  # avg signed margin (player0 − player1); ~0 by symmetry
    margin_std: float  # σ of the signed margin
    abs_margin: float  # avg winning margin |player0 − player1|
    abs_margin_std: float  # σ of the winning margin


class IterationMetrics(pydantic.BaseModel):
    """One collect-then-update cycle's metrics, logged and charted."""

    iteration: int
    total_games: int
    games_this_iter: int

    # loss components (TRAINING.md §3.3)
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    grad_norm: float
    advantage_mean: float
    advantage_std: float

    # averaged outcomes over this iteration's player-games
    avg_self_score: float  # mean final score per player-game
    avg_margin: float  # mean (player0 − player1); ~0 by self-play symmetry
    avg_breakdown: ScoreBreakdown
    avg_decisions: float  # mean trainable decisions per game

    # Winner-conditioned + dispersion outcomes (default for backward-compatible
    # loading of pre-existing metrics rows / checkpoints).
    avg_winner_breakdown: ScoreBreakdown = pydantic.Field(
        default_factory=ScoreBreakdown
    )  # mean score split of the winning seat, over decided (non-tie) games
    avg_abs_margin: float = 0.0  # mean |player0 − player1| (winning margin)
    avg_margin_sq: float = 0.0  # mean (player0 − player1)^2, for an EWMA σ

    family_counts: FamilyCounts  # decisions seen this iteration, per family

    # wall-clock breakdown (seconds)
    collect_seconds: float
    update_seconds: float
    eval_seconds: float
    games_per_sec: float

    eval: EvalResult | None = None


class SystemStats(pydantic.BaseModel):
    """A point-in-time snapshot of host CPU / RAM utilization.

    Refreshed ~once a second by the monitor thread for the dashboard's SYSTEM
    band; it is live-only telemetry, so unlike :class:`IterationMetrics` it is
    never serialized into the metrics log.
    """

    cpu_percent: float  # system-wide CPU utilization, 0..100
    ram_used_gb: float
    ram_total_gb: float
    proc_rss_gb: float  # resident memory of this training process

    @property
    def ram_percent(self) -> float:
        """System RAM in use as a percentage of total (0 if unknown)."""
        if self.ram_total_gb <= 0:
            return 0.0
        return 100.0 * self.ram_used_gb / self.ram_total_gb
