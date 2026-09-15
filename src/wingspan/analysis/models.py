"""Pydantic models and enums for the architecture-probe package.

Every data shape the ``analysis`` package produces or consumes lives here —
operational modules (``attention_probe``, ``layer_probe``, ``probe_set``,
``representation``, ``head_to_head``, ``cli``) import from this module and
never redefine a shape locally. This module depends only on ``pydantic`` and
the standard library so it stays importable without pulling in torch or the
rest of the project.

Three families of model:

* **Ablation identity** — :class:`AblationMode` names *how* a board-attention
  block was perturbed for one measurement.
* **Policy-delta statistics** — :class:`PolicyDeltaStats` and its subclasses
  (:class:`AblationEffect`, :class:`ReferenceComparison`) describe how much an
  ablation (or a reference checkpoint) moved the policy and value output,
  sliced overall, per judgment family (:class:`FamilyEffect`), or per
  board-fill band (:class:`BoardFillEffect`).
* **Structural readouts** — parameter census, per-layer capacity
  (:class:`LayerSummary` / :class:`LayerStats`), attention statistics
  (:class:`AttentionHeadStats` / :class:`AttentionBlockStats`), and trunk
  input-energy shares (:class:`InputGroupShare`), rolled up into one
  :class:`RepresentationReport` per checkpoint.
"""

from __future__ import annotations

import enum

import pydantic

BOARD_FILL_BANDS: tuple[tuple[int, int], ...] = (
    (0, 0),
    (1, 3),
    (4, 6),
    (7, 9),
    (10, 15),
)
"""``(min_birds, max_birds)`` bands the zero/uniform ablation effect is sliced
by, keyed on the deciding player's own board-fill count at decision time."""


class AblationMode(enum.StrEnum):
    """How a board-attention block was perturbed for one measurement.

    ``FULL`` is the unperturbed network (the baseline every other mode is
    compared against). ``ZERO`` drops the block's contribution entirely.
    ``UNIFORM`` replaces its attention weights with a uniform average over the
    unmasked (filled) keys. ``HEAD_KNOCKOUT`` zeroes exactly one attention
    head's output, identified separately by ``AblationEffect.head``.
    """

    FULL = "full"
    ZERO = "zero"
    UNIFORM = "uniform"
    HEAD_KNOCKOUT = "head_knockout"


class ParamCensusEntry(pydantic.BaseModel):
    """One top-level network block's share of the trainable-parameter count."""

    block: str
    parameters: int
    share: float


class LayerSummary(pydantic.BaseModel):
    """The lightweight per-layer capacity readout — the subset small enough to
    fold into a training-loop metrics row (:func:`representation.summarize_for_loop`).
    """

    name: str
    in_features: int
    out_features: int
    rank95: int
    """Number of eigen-directions of the activation covariance needed to reach
    95% of its variance — the layer's effective output rank."""
    rank95_over_width: float
    """``rank95 / out_features`` — near 1 means the layer uses its full width;
    much less than 1 means it is collapsed onto a lower-dimensional subspace."""
    linear_r2: float
    """R^2 of a ridge-regularized linear fit of this layer's post-ReLU output
    from its input — 1.0 means the layer's nonlinearity contributes nothing
    predictively beyond a linear map."""
    dead_fraction: float
    """Fraction of the layer's units that are never active (> 0) over the
    probed decisions."""


class LayerStats(LayerSummary):
    """The full per-layer capacity readout — :class:`LayerSummary` plus the
    fields only the offline report needs (row count, participation ratio,
    rare-unit fraction, and the weight matrix's own stable rank)."""

    rows: int
    """Decisions this layer's statistics were computed over (after the row cap)."""
    participation_ratio: float
    """Participation ratio of the activation covariance — a continuous,
    rotation-invariant effective-dimensionality estimate that complements
    ``rank95``."""
    rare_fraction: float
    """Fraction of units active on fewer than 1% of the probed rows."""
    weight_stable_rank: float
    """``||W||_F^2 / ||W||_2^2`` of this layer's weight matrix — a
    data-independent capacity-use estimate from the weights alone."""

    @property
    def summary(self) -> LayerSummary:
        """Project down to the lighter :class:`LayerSummary` shape consumed by
        the training-loop seam (:func:`representation.summarize_for_loop`)."""
        return LayerSummary(
            name=self.name,
            in_features=self.in_features,
            out_features=self.out_features,
            rank95=self.rank95,
            rank95_over_width=self.rank95_over_width,
            linear_r2=self.linear_r2,
            dead_fraction=self.dead_fraction,
        )


class AttentionHeadStats(pydantic.BaseModel):
    """One board-attention head's behavior over the probed decisions."""

    head: int
    entropy_median: float
    """Median attention entropy, normalized by ``log(n_filled)`` so 1.0 always
    means "uniform over the filled board slots", regardless of how many are
    filled. Computed only over rows with >= 2 filled slots."""
    entropy_p10: float
    self_weight_median: float
    """Median attention weight a filled slot places on itself (the diagonal);
    ``1 / n_filled`` is what a uniform head would produce."""


class AttentionBlockStats(pydantic.BaseModel):
    """The board-attention block's behavior in FULL mode, aggregated over
    every board (own and opponents') the probe set exercised."""

    contribution_ratio_median: float
    """Median ``||attn_out|| / ||token||`` over filled slots — how large the
    attention block's contribution is relative to the token it augments."""
    contribution_ratio_p10: float
    contribution_ratio_p90: float
    cosine_median: float
    """Median cosine similarity between the attention block's output and the
    original token — near 1 means the block mostly re-emits its input; near 0
    or negative means it computes something substantially different."""
    cosine_p10: float
    cosine_p90: float
    heads: list[AttentionHeadStats]


class PolicyDeltaStats(pydantic.BaseModel):
    """How much a perturbation moved the policy and value output, over some
    set of decisions.

    ``mean_kl`` / ``p90_kl`` are ``KL(full‖perturbed)`` in nats, per decision;
    ``flip_rate`` is the fraction of decisions whose greedy (argmax) choice
    changed; ``value_shift_points`` is the mean absolute value-head shift,
    converted to score points via ``score_norm``.
    """

    n: int
    mean_kl: float
    p90_kl: float
    flip_rate: float
    value_shift_points: float


class AblationEffect(PolicyDeltaStats):
    """A :class:`PolicyDeltaStats` for one ablation mode, over every probed
    decision. ``head`` is set only for :attr:`AblationMode.HEAD_KNOCKOUT`."""

    mode: AblationMode
    head: int | None = None


class ReferenceComparison(PolicyDeltaStats):
    """A :class:`PolicyDeltaStats` between the unperturbed network and a
    second, reference checkpoint played through the same probe-set batches."""

    checkpoint: str


class FamilyEffect(pydantic.BaseModel):
    """Zero/uniform ablation effect sliced to one judgment family
    (``decisions.DecisionFamily``)."""

    family: str
    n: int
    zero: PolicyDeltaStats
    uniform: PolicyDeltaStats
    reference: PolicyDeltaStats | None = None


class BoardFillEffect(pydantic.BaseModel):
    """Zero/uniform ablation effect sliced to one :data:`BOARD_FILL_BANDS`
    band of the deciding player's own board-fill count."""

    min_birds: int
    max_birds: int
    n: int
    zero: PolicyDeltaStats
    uniform: PolicyDeltaStats


class InputGroupShare(pydantic.BaseModel):
    """One named group of trunk-input columns' share of the first trunk
    layer's pre-activation output variance (its "energy")."""

    group: str
    dims: int
    energy_share: float


class HeadToHeadResult(pydantic.BaseModel):
    """Paired-game win rate of the full network against a copy with its
    board-attention block substituted (see ``head_to_head.evaluate_substitution``)."""

    mode: AblationMode
    n_games: int
    win_rate: float
    ci95: float
    mean_margin: float


class RepresentationMetrics(pydantic.BaseModel):
    """The lightweight subset of a :class:`RepresentationReport` sized for a
    training-loop metrics row (Stage 2; not populated by this package)."""

    probe_decisions: int
    layers: list[LayerSummary]
    attention_entropy_median: list[float]
    uniform_kl: float | None = None
    uniform_flip_rate: float | None = None
    zero_kl: float | None = None
    zero_flip_rate: float | None = None


class RepresentationReport(pydantic.BaseModel):
    """The full architecture-probe report for one checkpoint — the return
    value of :func:`representation.measure` and the payload the CLI prints /
    writes as JSON."""

    checkpoint: str
    n_games: int
    n_decisions: int
    param_census: list[ParamCensusEntry]
    total_parameters: int
    layers: list[LayerStats]
    attention: AttentionBlockStats | None = None
    ablations: list[AblationEffect]
    reference: ReferenceComparison | None = None
    family_effects: list[FamilyEffect]
    board_fill_effects: list[BoardFillEffect]
    trunk_input_shares: list[InputGroupShare]
    head_to_head: list[HeadToHeadResult] = pydantic.Field(
        default_factory=list[HeadToHeadResult]
    )

    def effect(self, mode: AblationMode) -> AblationEffect | None:
        """The block-level (not per-head) :class:`AblationEffect` for ``mode``,
        or ``None`` when board attention is off (no ablations were measured)."""
        for ablation in self.ablations:
            if ablation.mode == mode and ablation.head is None:
                return ablation
        return None
