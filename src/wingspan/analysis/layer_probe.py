"""Forward-hook capacity diagnostics for one ``nn.Sequential`` MLP.

:class:`LinearLayerProbe` registers a forward hook on every ``nn.Linear`` in a
given block (the state trunk, the choice encoder) and, once a probe pass has
run, reports each layer's effective rank, linear predictability, and
dead/rare-unit fractions from the captured activations — the "how much of
this layer's capacity is actually used" readout ``docs/TRAINING.md``'s
Representation diagnostics section describes. The three statistics functions
(:func:`participation_ratio_and_rank95`, :func:`ridge_r2`,
:func:`weight_stable_rank`) are public because they are independently useful
(and independently testable) linear-algebra primitives, not because callers
outside this module are expected to reach for them directly.
"""

from __future__ import annotations

import typing

import torch
import torch.utils.hooks as hooks
from torch import nn

from wingspan.analysis import models

# Row cap for the per-layer statistics: activations are seeded-subsampled down
# to this many rows before any downstream computation, so a large probe set
# costs no more than a fixed budget of eigendecompositions / ridge solves.
_MAX_ROWS = 20_000
_ROW_SAMPLE_SEED = 0

# A unit active on fewer than this fraction of (capped) rows counts as "rare".
_RARE_ACTIVATION_THRESHOLD = 0.01

# Ridge regularization strength (relative to the Gram matrix trace) for
# ridge_r2's linear fit, and the cumulative-variance threshold rank95 reaches.
_RIDGE_LAMBDA_REL = 1e-3
_RANK95_VARIANCE_THRESHOLD = 0.95

# Numerical floor so a degenerate (zero-variance) denominator never divides.
_VARIANCE_EPS = 1e-12


class _CapturedLayer(typing.NamedTuple):
    """One hooked ``nn.Linear`` plus its accumulated per-pass captures."""

    name: str
    module: nn.Linear
    inputs: list[torch.Tensor]
    pre_activations: list[torch.Tensor]


class LinearLayerProbe:
    """Captures every ``nn.Linear`` input/output in a given ``nn.Sequential``
    block over however many forward passes run while the hooks are installed.

    Layers are named ``f"{prefix}.L{index}"`` in traversal order (``index``
    counts only ``nn.Linear`` submodules, skipping activations, dropout, and
    LayerNorm). Call :meth:`remove` once the probe pass(es) are done, then
    :meth:`stats` for the per-layer :class:`models.LayerStats`.
    """

    def __init__(self, prefix: str, sequential: nn.Sequential) -> None:
        self._layers: list[_CapturedLayer] = []
        self._handles: list[hooks.RemovableHandle] = []
        index = 0
        for module in sequential:
            if not isinstance(module, nn.Linear):
                continue
            captured = _CapturedLayer(
                name=f"{prefix}.L{index}", module=module, inputs=[], pre_activations=[]
            )
            self._layers.append(captured)
            self._handles.append(module.register_forward_hook(_make_hook(captured)))
            index += 1

    def remove(self) -> None:
        """Detach every registered hook. Idempotent."""
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def stats(self) -> list[models.LayerStats]:
        """Per-layer :class:`models.LayerStats` computed on post-ReLU
        activations (the trunk and choice encoder both use ReLU between
        layers) over every row captured since construction, row-capped to
        :data:`_MAX_ROWS` with a fixed seed."""
        generator = torch.Generator().manual_seed(_ROW_SAMPLE_SEED)
        results: list[models.LayerStats] = []
        for captured in self._layers:
            inputs = torch.cat(captured.inputs)
            activations = torch.relu(torch.cat(captured.pre_activations))
            inputs, activations = _row_capped(inputs, activations, generator)
            results.append(
                _layer_stats(captured.name, captured.module, inputs, activations)
            )
        return results


def participation_ratio_and_rank95(activations: torch.Tensor) -> tuple[float, int]:
    """The activation covariance's participation ratio and 95%-variance rank.

    Participation ratio (``(Σλ)² / Σλ²`` of the covariance eigenvalues) is a
    continuous, rotation-invariant effective-dimensionality estimate;
    ``rank95`` is the number of eigen-directions (largest first) needed to
    reach :data:`_RANK95_VARIANCE_THRESHOLD` of the total variance — a more
    interpretable, if coarser, companion statistic."""
    centered = activations - activations.mean(0, keepdim=True)
    rows = centered.shape[0]
    covariance = centered.T @ centered / max(rows - 1, 1)
    eigenvalues = _eigvalsh(covariance).clamp(min=0)
    total_variance = eigenvalues.sum()
    participation_ratio = (
        total_variance**2 / eigenvalues.pow(2).sum().clamp(min=_VARIANCE_EPS)
    ).item()
    cumulative_share = eigenvalues.flip(0).cumsum(0) / total_variance.clamp(
        min=_VARIANCE_EPS
    )
    # A degenerate (zero-variance) layer clamps every share to 0/eps = 0, which
    # would otherwise count past the last eigen-direction; cap at the true
    # dimensionality so rank95 always lands in [1, out_features].
    rank95 = min(
        int((cumulative_share < _RANK95_VARIANCE_THRESHOLD).sum().item()) + 1,
        eigenvalues.shape[0],
    )
    return participation_ratio, rank95


def ridge_r2(
    inputs: torch.Tensor, outputs: torch.Tensor, lam_rel: float = _RIDGE_LAMBDA_REL
) -> float:
    """R^2 of a ridge-regularized linear fit ``outputs ~ inputs``: 1.0 means
    ``outputs`` is (near-)exactly a linear map of ``inputs``; 0.0 means the
    best linear map explains none of ``outputs``'s variance. ``lam_rel``
    scales the ridge penalty relative to the input Gram matrix's trace, so the
    fit stays well-posed even when ``inputs`` is wider than it is tall."""
    centered_in = inputs - inputs.mean(0, keepdim=True)
    centered_out = outputs - outputs.mean(0, keepdim=True)
    dims = centered_in.shape[1]
    gram = centered_in.T @ centered_in
    ridge_penalty = lam_rel * gram.trace() / dims
    weights = _solve(
        gram + ridge_penalty * torch.eye(dims), centered_in.T @ centered_out
    )
    residual = centered_out - centered_in @ weights
    total_variance = centered_out.pow(2).sum().clamp(min=_VARIANCE_EPS)
    return 1.0 - (residual**2).sum().item() / total_variance.item()


def weight_stable_rank(weight: torch.Tensor) -> float:
    """``||W||_F^2 / ||W||_2^2`` — the weight matrix's stable rank, a
    data-independent capacity-use estimate from the learned weights alone
    (as opposed to the activation-based statistics above)."""
    singular_values = _svdvals(weight)
    return ((singular_values**2).sum() / (singular_values[0] ** 2)).item()


###### PRIVATE #######


# The three wrappers below pin their line length by hand (black's own
# reformatting would relocate the trailing ignore comment away from the line
# pyright anchors its diagnostic to — see each docstring); `# fmt: off` keeps
# black from ever re-exploding them.
# fmt: off
def _eigvalsh(matrix: torch.Tensor) -> torch.Tensor:
    """torch.linalg.eigvalsh, pinned to Tensor (the stub under-specifies it)."""
    return torch.linalg.eigvalsh(matrix)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def _svdvals(matrix: torch.Tensor) -> torch.Tensor:
    """torch.linalg.svdvals, pinned to Tensor (the stub under-specifies it)."""
    return torch.linalg.svdvals(matrix)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]


def _solve(matrix: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """torch.linalg.solve, pinned to Tensor (the stub under-specifies it)."""
    return torch.linalg.solve(matrix, rhs)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
# fmt: on


def _make_hook(
    captured: _CapturedLayer,
) -> typing.Callable[[nn.Module, tuple[torch.Tensor, ...], torch.Tensor], None]:
    """A forward hook that appends this call's flattened input and
    pre-activation output onto ``captured``'s accumulator lists."""

    def hook(
        _module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        captured.inputs.append(inputs[0].detach().reshape(-1, inputs[0].shape[-1]))
        captured.pre_activations.append(output.detach().reshape(-1, output.shape[-1]))

    return hook


def _row_capped(
    inputs: torch.Tensor, activations: torch.Tensor, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Subsample ``(inputs, activations)`` to at most :data:`_MAX_ROWS` rows,
    using the same random row selection for both so they stay paired."""
    rows = inputs.shape[0]
    if rows <= _MAX_ROWS:
        return inputs, activations
    index = torch.randperm(rows, generator=generator)[:_MAX_ROWS]
    return inputs[index], activations[index]


def _layer_stats(
    name: str, module: nn.Linear, inputs: torch.Tensor, activations: torch.Tensor
) -> models.LayerStats:
    """Assemble one layer's :class:`models.LayerStats` from its (already
    row-capped) captured input and post-ReLU activation matrices."""
    active_rate = (activations > 0).float().mean(0)
    dead_fraction = float((active_rate == 0).float().mean())
    rare_fraction = float((active_rate < _RARE_ACTIVATION_THRESHOLD).float().mean())
    participation_ratio, rank95 = participation_ratio_and_rank95(activations)
    return models.LayerStats(
        name=name,
        in_features=module.in_features,
        out_features=module.out_features,
        rank95=rank95,
        rank95_over_width=rank95 / module.out_features,
        linear_r2=ridge_r2(inputs, activations),
        dead_fraction=dead_fraction,
        rows=int(activations.shape[0]),
        participation_ratio=participation_ratio,
        rare_fraction=rare_fraction,
        weight_stable_rank=weight_stable_rank(module.weight.detach()),
    )
