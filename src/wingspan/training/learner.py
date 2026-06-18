"""The gradient update: length-bucketed REINFORCE with a value baseline.

One :func:`update` call performs a single on-policy REINFORCE step over all the
steps of an iteration's games, with the two TRAINING.md fixes that make the
baseline honest and the memory sane:

* **Length-bucketing (TRAINING.md §4.2a).** Stacking every decision into one tensor padded
  to the widest decision in the batch (the 504-option opening draft, or a
  food-rich 600+-option play) wastes ~97% of the tensor on padding and peaks
  GPU memory near 11 GB on a default batch. Instead steps are grouped into
  option-count buckets and padded only to each bucket's own width — a ~40×
  memory reduction — then their losses are summed over one shared backward.

* **Advantage normalization (TRAINING.md §3.3).** Advantages are centered and scaled to unit
  std across the whole batch before the policy loss, so the gradient magnitude
  stays stable from the first iteration to the last regardless of how good the
  critic has become.

The loss is the standard actor-critic sum
``policy_loss + VALUE_COEF·value_loss − ENTROPY_COEF·entropy`` (TRAINING.md §3.3),
or, in the DAgger imitation phase, ``imitation_loss + VALUE_COEF·value_loss``
where ``imitation_loss`` is the mask-weighted cross-entropy to the expert's soft
targets (the value head is kept to warm the critic for the RL handoff).
"""

from __future__ import annotations

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
from torch import optim

from wingspan import model
from wingspan.training import collect, config, steps, timestamps

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
    # Non-zero only during DAgger clone iterations; 0.0 in RL mode.
    imitation_loss: float = 0.0
    n_steps: int


def update(
    net: model.PolicyValueNet,
    optimizer: optim.Optimizer,
    records: list[collect.GameRecord],
    cfg: config.RunConfig,
    device: torch.device,
    imitation_phase: bool = False,
) -> UpdateStats:
    """Run one length-bucketed update over ``records``' steps.

    In the normal RL mode (``imitation_phase=False``) this is a standard
    actor-critic REINFORCE step.  In the DAgger imitation phase
    (``imitation_phase=True``) the policy-gradient and entropy terms are
    replaced by cross-entropy to each step's recorded ``expert_probs``; the
    value-head MSE loss is kept to warm the critic for the RL handoff.
    """
    flat_steps, returns = _flatten(records, cfg)
    if not flat_steps:
        return UpdateStats(
            loss=0.0,
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            grad_norm=0.0,
            advantage_mean=0.0,
            advantage_std=0.0,
            imitation_loss=0.0,
            n_steps=0,
        )

    # Forward each bucket separately (padding stays narrow) and keep the
    # graph-carrying tensors so a single backward covers the whole batch.
    chosen_logps: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []
    returns_parts: list[torch.Tensor] = []
    imitation_ces: list[torch.Tensor] = []
    has_experts: list[torch.Tensor] = []
    for bucket in _bucketize(flat_steps):
        logp, value, entropy, imitation_ce, has_expert = _forward_bucket(
            net, device, flat_steps, bucket
        )
        chosen_logps.append(logp)
        values.append(value)
        entropies.append(entropy)
        returns_parts.append(
            torch.tensor(
                [returns[i] for i in bucket], dtype=torch.float32, device=device
            )
        )
        imitation_ces.append(imitation_ce)
        has_experts.append(has_expert)

    logp_all = torch.cat(chosen_logps)
    value_all = torch.cat(values)
    entropy_all = torch.cat(entropies)
    return_all = torch.cat(returns_parts)
    imitation_ce_all = torch.cat(imitation_ces)
    has_expert_all = torch.cat(has_experts)

    value_loss = F.mse_loss(value_all, return_all)

    if imitation_phase:
        # Pure imitation: minimize cross-entropy to expert's soft targets.
        # Mask-weighted mean over labeled steps only; clamp(min=1) guards the
        # all-unlabeled edge (family_idx >= expert_net.num_families, i.e.
        # SETUP steps when the expert was trained without the SETUP head).
        imitation_loss_t = (imitation_ce_all * has_expert_all).sum() / (
            has_expert_all.sum().clamp(min=1)
        )
        loss = imitation_loss_t + cfg.training.value_coef * value_loss
        # No policy-gradient or entropy in imitation mode.
        policy_loss_t = torch.zeros(1, device=device)
        entropy_t = torch.zeros(1, device=device)
        adv_mean = torch.zeros(1, device=device)
        adv_std = torch.zeros(1, device=device)
    else:
        # Advantage = return − baseline, normalized across the batch (TRAINING.md §3.3).
        advantage = return_all - value_all.detach()
        adv_mean = advantage.mean()
        adv_std = advantage.std()
        norm_advantage = (advantage - adv_mean) / (adv_std + _ADV_STD_EPS)

        policy_loss_t = -(logp_all * norm_advantage).mean()
        entropy_t = entropy_all.mean()
        loss = (
            policy_loss_t
            + cfg.training.value_coef * value_loss
            - cfg.training.entropy_coef * entropy_t
        )
        imitation_loss_t = torch.zeros(1, device=device)

    optimizer.zero_grad()
    # torch's stub types Tensor.backward with unknown parameters; the precise
    # typing of the loss chain (unlike the Any-typed Module-call path) surfaces
    # that stub gap, so the narrow suppression is on the stub, not our logic.
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    grad_norm = torch.nn.utils.clip_grad_norm_(
        net.parameters(), max_norm=cfg.training.grad_clip
    )
    optimizer.step()

    return UpdateStats(
        loss=float(loss.detach()),
        policy_loss=float(policy_loss_t.detach()),
        value_loss=float(value_loss.detach()),
        entropy=float(entropy_t.detach()),
        grad_norm=float(grad_norm),
        advantage_mean=float(adv_mean.detach()),
        advantage_std=float(adv_std.detach()),
        imitation_loss=float(imitation_loss_t.detach()),
        n_steps=len(flat_steps),
    )


###### PRIVATE #######


def _flatten(
    records: list[collect.GameRecord], cfg: config.RunConfig
) -> tuple[list[steps.Step], list[float]]:
    """Flatten every game's steps and pair each with its REINFORCE return.

    Two orthogonal axes (``cfg.training``):

    * ``reward_mode`` — *how* credit spreads across steps:
      ``terminal_margin`` broadcasts the end-of-game value to every step;
      ``decision_delta`` credits each step with only the change in value from
      that step onward, discounted by ``reward_discount`` per game-clock unit.
    * ``reward_basis`` — *what* quantity is used as the value:
      ``margin`` uses own − opponent score (seats get opposite signs);
      ``own_score`` uses each player's absolute final score (both positive).
    """
    flat_steps: list[steps.Step] = []
    returns: list[float] = []
    basis = cfg.training.reward_basis
    for record in records:
        if cfg.training.reward_mode is config.RewardMode.DECISION_DELTA:
            record_returns = _decision_delta_returns(
                record,
                cfg.training.reward_discount,
                cfg.training.score_norm,
                cfg.training.end_game_bonus,
                basis,
            )
        else:
            record_returns = _terminal_margin_returns(
                record, cfg.training.score_norm, cfg.training.end_game_bonus, basis
            )
        flat_steps.extend(record.steps)
        returns.extend(record_returns)
    return flat_steps, returns


def _terminal_margin_returns(
    record: collect.GameRecord,
    score_norm: float,
    end_game_bonus: float,
    basis: config.RewardBasis,
) -> list[float]:
    """The end-of-game value from each step's player POV, broadcast to every step.

    With ``MARGIN`` basis, value = own − opponent score; seats get opposite signs.
    ``end_game_bonus`` is added/subtracted symmetrically (``_winner_bonus``).
    With ``OWN_SCORE`` basis, value = player's own absolute score; both seats
    are positive and ``end_game_bonus`` is added only to the winner's score."""
    score_0, score_1 = record.breakdowns[0].total, record.breakdowns[1].total
    if basis is config.RewardBasis.OWN_SCORE:
        bonus_0 = end_game_bonus if record.winner == 0 else 0.0
        bonus_1 = end_game_bonus if record.winner == 1 else 0.0
        per_pov = (
            (score_0 + bonus_0) / score_norm,
            (score_1 + bonus_1) / score_norm,
        )
    else:
        bonus_0 = _winner_bonus(record.winner, end_game_bonus)
        per_pov = (
            (score_0 - score_1 + bonus_0) / score_norm,
            (score_1 - score_0 - bonus_0) / score_norm,
        )
    return [per_pov[step.player_id] for step in record.steps]


def _decision_delta_returns(
    record: collect.GameRecord,
    discount: float,
    score_norm: float,
    end_game_bonus: float,
    basis: config.RewardBasis,
) -> list[float]:
    """Per-decision discounted returns aligned to ``record.steps`` order.

    For each player the recorded value checkpoints (``margin_before`` with
    ``MARGIN`` basis; ``score_before`` with ``OWN_SCORE`` basis) plus the
    terminal value form a sequence ``v`` with game-clock times ``t``; the
    per-step reward is ``v[k+1] - v[k]`` and the return is the backward
    discounted sum ``G[k] = r[k] + γ^(t[k+1]−t[k])·G[k+1]``, scaled by
    ``score_norm``. With ``MARGIN`` basis the two seats get opposite signs;
    with ``OWN_SCORE`` basis both are positive.

    ``end_game_bonus`` is folded into the terminal checkpoint before the
    backward sweep so it discounts back through all prior decisions."""
    score_0, score_1 = record.breakdowns[0].total, record.breakdowns[1].total

    # Terminal value for each seat (the final ``v`` in the checkpoint sequence).
    if basis is config.RewardBasis.OWN_SCORE:
        bonus_0 = end_game_bonus if record.winner == 0 else 0.0
        bonus_1 = end_game_bonus if record.winner == 1 else 0.0
        terminal = (score_0 + bonus_0, score_1 + bonus_1)
    else:
        bonus_0 = _winner_bonus(record.winner, end_game_bonus)
        terminal = (score_0 - score_1 + bonus_0, score_1 - score_0 - bonus_0)

    # Returns land back in record order, so route each player's discounted
    # returns through the global step index they were computed from.
    out: list[float] = [0.0] * len(record.steps)
    for player_id in (0, 1):
        indices = [
            i for i, step in enumerate(record.steps) if step.player_id == player_id
        ]
        if not indices:
            continue
        if basis is config.RewardBasis.OWN_SCORE:
            checkpoints = [record.steps[i].score_before for i in indices]
        else:
            checkpoints = [record.steps[i].margin_before for i in indices]
        checkpoints.append(terminal[player_id])
        times = [record.steps[i].timestamp for i in indices]
        times.append(record.final_timestamp)
        raw_returns = timestamps.discounted_future_returns(checkpoints, times, discount)
        for position, idx in enumerate(indices):
            out[idx] = raw_returns[position] / score_norm
    return out


def _winner_bonus(winner: int, end_game_bonus: float) -> float:
    """Seat-0-POV bonus delta: ``+bonus`` when seat 0 wins, ``-bonus`` when seat 1
    wins, ``0`` on a tie (``winner == -1``)."""
    if winner == 0:
        return end_game_bonus
    if winner == 1:
        return -end_game_bonus
    return 0.0


def _bucketize(flat_steps: list[steps.Step]) -> list[list[int]]:
    """Group step indices by option-count bucket (smallest edge >= n_choices)."""
    by_edge: dict[int, list[int]] = {}
    for i, step in enumerate(flat_steps):
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
    flat_steps: list[steps.Step],
    bucket: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward one bucket (padded to its own width) and return per-step
    ``(chosen_logp, value, entropy, imitation_ce, has_expert)`` tensors with
    grad attached.

    ``imitation_ce[i]`` is the cross-entropy between the student's distribution
    and the expert's soft target for step ``i``; ``has_expert[i]`` is 1.0 when
    that step carries a DAgger label, 0.0 otherwise (family-skip or RL mode).
    Both are always computed — the caller uses ``has_expert`` as a mask so the
    RL path is zero-cost (imitation_ce is valid but multiplied by 0)."""
    width = max(flat_steps[i].choices.shape[0] for i in bucket)
    batch = len(bucket)
    state_batch = np.stack([flat_steps[i].state for i in bucket])
    choice_batch = np.zeros((batch, width, net.choice_dim), dtype=np.float32)
    mask_batch = np.zeros((batch, width), dtype=np.float32)
    # Zero-initialized: padded columns contribute nothing to imitation CE even
    # without explicit masking (expert_t[pad] == 0, so -0 * legal_logp = 0).
    expert_batch = np.zeros((batch, width), dtype=np.float32)
    has_expert_batch = np.zeros(batch, dtype=np.float32)
    for row, i in enumerate(bucket):
        count = flat_steps[i].choices.shape[0]
        choice_batch[row, :count] = flat_steps[i].choices
        mask_batch[row, :count] = 1.0
        if flat_steps[i].expert_probs is not None:
            expert_batch[row, :count] = flat_steps[i].expert_probs
            has_expert_batch[row] = 1.0

    state_t = torch.tensor(state_batch, dtype=torch.float32, device=device)
    choice_t = torch.tensor(choice_batch, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask_batch, dtype=torch.float32, device=device)
    expert_t = torch.tensor(expert_batch, dtype=torch.float32, device=device)
    has_expert_t = torch.tensor(has_expert_batch, dtype=torch.float32, device=device)
    idx_t = torch.tensor(
        [flat_steps[i].chosen_idx for i in bucket], dtype=torch.long, device=device
    )
    family_t = torch.tensor(
        [flat_steps[i].family_idx for i in bucket], dtype=torch.long, device=device
    )

    logits, value = net(state_t, choice_t, mask_t, family_t)
    logp = F.log_softmax(logits, dim=-1)
    chosen_logp = logp.gather(1, idx_t.unsqueeze(1)).squeeze(1)

    # Entropy + imitation CE over legal slots; torch.where avoids 0*-inf=NaN on padding.
    zeros = torch.zeros_like(logp)
    legal_logp = torch.where(mask_t > 0.5, logp, zeros)
    legal_p = torch.where(mask_t > 0.5, logp.exp(), zeros)
    entropy = -(legal_p * legal_logp).sum(dim=-1)
    # CE to expert soft targets: sum over the legal window.  Padded columns in
    # expert_t are 0 by construction so they contribute nothing to the sum.
    imitation_ce = -(expert_t * legal_logp).sum(dim=-1)
    return chosen_logp, value, entropy, imitation_ce, has_expert_t
