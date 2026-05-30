"""The gradient update: length-bucketed REINFORCE with a value baseline.

One :func:`update` call performs a single on-policy REINFORCE step over all the
steps of an iteration's games, with the two TRAINING.md fixes that make the
baseline honest and the memory sane:

* **Length-bucketing (§4.2a).** Stacking every decision into one tensor padded
  to the widest decision in the batch (the 504-option opening draft, or a
  food-rich 600+-option play) wastes ~97% of the tensor on padding and peaks
  GPU memory near 11 GB on a default batch. Instead steps are grouped into
  option-count buckets and padded only to each bucket's own width — a ~40×
  memory reduction — then their losses are summed over one shared backward.

* **Advantage normalization (§3.3).** Advantages are centered and scaled to unit
  std across the whole batch before the policy loss, so the gradient magnitude
  stays stable from the first iteration to the last regardless of how good the
  critic has become.

The loss is the standard actor-critic sum
``policy_loss + VALUE_COEF·value_loss − ENTROPY_COEF·entropy`` (§3.3).
"""

from __future__ import annotations

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
from torch import optim

from wingspan import encode, model, train
from wingspan.training import collect, config

# Option-count bucket edges. A step with ``n`` candidates pads up to the
# smallest edge ``>= n``; the 89.5% of decisions with <=4 options pad to 4 (not
# 504), and the rare wide decisions get their own narrow bucket.
_BUCKET_EDGES: tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128, 256, 512, 2048)

_ADV_STD_EPS = 1e-6


class UpdateStats(pydantic.BaseModel):
    """Summary metrics from one optimizer update."""

    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    grad_norm: float
    advantage_mean: float
    advantage_std: float
    n_steps: int


def update(
    net: model.PolicyValueNet,
    optimizer: optim.Optimizer,
    records: list[collect.GameRecord],
    cfg: config.TrainConfig,
    device: torch.device,
) -> UpdateStats:
    """Run one length-bucketed REINFORCE update over ``records``' steps."""
    steps, returns = _flatten(records, cfg.score_norm)
    if not steps:
        return UpdateStats(
            loss=0.0,
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            grad_norm=0.0,
            advantage_mean=0.0,
            advantage_std=0.0,
            n_steps=0,
        )

    # Forward each bucket separately (padding stays narrow) and keep the
    # graph-carrying tensors so a single backward covers the whole batch.
    chosen_logps: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    returns_parts: list[torch.Tensor] = []
    for bucket in _bucketize(steps):
        logp, value, entropy = _forward_bucket(net, device, steps, bucket)
        chosen_logps.append(logp)
        values.append(value)
        entropies.append(entropy)
        returns_parts.append(
            torch.tensor(
                [returns[i] for i in bucket], dtype=torch.float32, device=device
            )
        )

    logp_all = torch.cat(chosen_logps)
    value_all = torch.cat(values)
    entropy_all = torch.cat(entropies)
    return_all = torch.cat(returns_parts)

    # Advantage = return − baseline, normalized across the batch (§3.3).
    advantage = return_all - value_all.detach()
    adv_mean = advantage.mean()
    adv_std = advantage.std()
    norm_advantage = (advantage - adv_mean) / (adv_std + _ADV_STD_EPS)

    policy_loss = -(logp_all * norm_advantage).mean()
    value_loss = F.mse_loss(value_all, return_all)
    entropy = entropy_all.mean()
    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

    optimizer.zero_grad()
    # torch's stub types Tensor.backward with unknown parameters; the precise
    # typing of the loss chain (unlike the Any-typed Module-call path) surfaces
    # that stub gap, so the narrow suppression is on the stub, not our logic.
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg.grad_clip)
    optimizer.step()

    return UpdateStats(
        loss=float(loss.detach()),
        policy_loss=float(policy_loss.detach()),
        value_loss=float(value_loss.detach()),
        entropy=float(entropy.detach()),
        grad_norm=float(grad_norm),
        advantage_mean=float(adv_mean.detach()),
        advantage_std=float(adv_std.detach()),
        n_steps=len(steps),
    )


###### PRIVATE #######


def _flatten(
    records: list[collect.GameRecord], score_norm: float
) -> tuple[list[train.Step], list[float]]:
    """Flatten every game's steps and pair each with its REINFORCE return —
    the terminal score margin from that step's player POV, scaled by
    ``score_norm`` (so player 0 and player 1 get opposite signs in a decisive
    game; DECISIONS.md §5)."""
    steps: list[train.Step] = []
    returns: list[float] = []
    for record in records:
        score_0, score_1 = record.breakdowns[0].total, record.breakdowns[1].total
        per_pov = (
            (score_0 - score_1) / score_norm,
            (score_1 - score_0) / score_norm,
        )
        for step in record.steps:
            steps.append(step)
            returns.append(per_pov[step.player_id])
    return steps, returns


def _bucketize(steps: list[train.Step]) -> list[list[int]]:
    """Group step indices by option-count bucket (smallest edge >= n_choices)."""
    by_edge: dict[int, list[int]] = {}
    for i, step in enumerate(steps):
        edge = _bucket_edge(step.choices.shape[0])
        by_edge.setdefault(edge, []).append(i)
    return [by_edge[edge] for edge in sorted(by_edge)]


def _bucket_edge(n_choices: int) -> int:
    for edge in _BUCKET_EDGES:
        if n_choices <= edge:
            return edge
    return _BUCKET_EDGES[-1]


def _forward_bucket(
    net: model.PolicyValueNet,
    device: torch.device,
    steps: list[train.Step],
    bucket: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward one bucket (padded to its own width) and return per-step
    ``(chosen_logp, value, entropy)`` tensors with grad attached."""
    width = max(steps[i].choices.shape[0] for i in bucket)
    batch = len(bucket)
    state_batch = np.stack([steps[i].state for i in bucket])
    choice_batch = np.zeros((batch, width, encode.CHOICE_FEATURE_DIM), dtype=np.float32)
    mask_batch = np.zeros((batch, width), dtype=np.float32)
    for row, i in enumerate(bucket):
        count = steps[i].choices.shape[0]
        choice_batch[row, :count] = steps[i].choices
        mask_batch[row, :count] = 1.0

    state_t = torch.tensor(state_batch, dtype=torch.float32, device=device)
    choice_t = torch.tensor(choice_batch, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask_batch, dtype=torch.float32, device=device)
    idx_t = torch.tensor(
        [steps[i].chosen_idx for i in bucket], dtype=torch.long, device=device
    )
    family_t = torch.tensor(
        [steps[i].family_idx for i in bucket], dtype=torch.long, device=device
    )

    logits, value = net(state_t, choice_t, mask_t, family_t)
    logp = F.log_softmax(logits, dim=-1)
    chosen_logp = logp.gather(1, idx_t.unsqueeze(1)).squeeze(1)

    # Entropy over legal slots only; torch.where avoids 0*-inf=NaN on padding.
    zeros = torch.zeros_like(logp)
    legal_logp = torch.where(mask_t > 0.5, logp, zeros)
    legal_p = torch.where(mask_t > 0.5, logp.exp(), zeros)
    entropy = -(legal_p * legal_logp).sum(dim=-1)
    return chosen_logp, value, entropy
