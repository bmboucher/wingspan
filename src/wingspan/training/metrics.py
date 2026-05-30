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
    """Outcome of a paired-game evaluation against the random agent (§7.3).

    ``win_rate`` and its ``ci95`` half-width use the normal approximation
    ``p ± 1.96·√(p(1−p)/n)``; ties count as half a win. ``mean_margin`` is the
    average score margin (policy − opponent) across the held-out games.
    """

    n_games: int
    win_rate: float
    ci95: float
    mean_margin: float


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

    family_counts: FamilyCounts  # decisions seen this iteration, per family

    # wall-clock breakdown (seconds)
    collect_seconds: float
    update_seconds: float
    eval_seconds: float
    games_per_sec: float

    eval: EvalResult | None = None
