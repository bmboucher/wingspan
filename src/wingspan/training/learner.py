"""The gradient update: length-bucketed actor-critic with optional PPO / GAE.

:func:`update` dispatches between two update paths:

* **Single-pass (default)** — one length-bucketed REINFORCE step with advantage
  normalization and an optional DAgger imitation loss (``imitation_phase=True``).
  This is today's path; defaults reproduce it byte-for-byte.

* **Reuse path** — activated when ``cfg.training.policy_loss`` is ``PPO`` or
  ``cfg.training.reward_mode`` is ``GAE``. Advantages are computed **once** from
  captured ``Step.behavior_logp`` / ``Step.value_pred``, normalized, then reused
  across ``ppo_reuse_epochs`` backward passes (one epoch when only GAE is on).
  PPO uses the clipped surrogate ``−min(ratio·A, clip(ratio,1±ε)·A)``; REINFORCE
  with GAE uses ``−(logp·A)``. Both share the length-bucketed forward pass.

Both paths also have a **gradient-accumulation** variant, activated when
``cfg.training.update_minibatch_steps > 0``.  The batch is split into
sequential minibatches of that many flattened steps; gradients are accumulated
across them and the optimizer takes **one step per epoch** — reproducing
today's gradient up to float summation order while capping peak memory at the
minibatch size (``docs/TRAINING.md §3.3``).  The default (``0``) leaves the
existing paths byte-identical.

Common to both paths:

* **Length-bucketing (TRAINING.md §4.2a):** steps grouped into option-count
  buckets padded only to each bucket's own width (≈40× memory reduction vs. a
  single padded tensor).

* **Advantage normalization (TRAINING.md §3.3):** advantages centered and scaled
  to unit std before the policy loss.

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
    # PPO diagnostics — 0.0 on the single-pass (REINFORCE / DAgger) path.
    clip_fraction: float = 0.0
    approx_kl: float = 0.0


def update(
    net: model.PolicyValueNet,
    optimizer: optim.Optimizer,
    records: list[collect.GameRecord],
    cfg: config.RunConfig,
    device: torch.device,
    imitation_phase: bool = False,
) -> UpdateStats:
    """Run one length-bucketed update over ``records``' steps.

    Dispatches to the single-pass REINFORCE path (today's algorithm, including
    DAgger imitation mode) or the PPO / GAE reuse path based on
    ``cfg.training.policy_loss`` and ``cfg.training.reward_mode``.  The default
    config always dispatches to single-pass, preserving existing behaviour.
    """
    ppo = cfg.training.policy_loss is config.PolicyLoss.PPO
    gae = cfg.training.reward_mode is config.RewardMode.GAE
    minibatch_steps = cfg.training.update_minibatch_steps
    single_pass = imitation_phase or (not ppo and not gae)
    if minibatch_steps > 0:
        if single_pass:
            return _update_single_pass_minibatched(
                net, optimizer, records, cfg, device, imitation_phase, minibatch_steps
            )
        return _update_reuse_minibatched(
            net,
            optimizer,
            records,
            cfg,
            device,
            ppo=ppo,
            minibatch_steps=minibatch_steps,
        )
    if single_pass:
        return _update_single_pass(
            net, optimizer, records, cfg, device, imitation_phase
        )
    return _update_reuse(net, optimizer, records, cfg, device, ppo=ppo)


###### PRIVATE #######


def _update_single_pass(
    net: model.PolicyValueNet,
    optimizer: optim.Optimizer,
    records: list[collect.GameRecord],
    cfg: config.RunConfig,
    device: torch.device,
    imitation_phase: bool = False,
) -> UpdateStats:
    """One length-bucketed REINFORCE / DAgger update (single backward pass).

    This is today's ``update()`` body extracted verbatim — the canonical path
    for the default config (REINFORCE + MC returns) and all DAgger clone iters.
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


def _update_reuse(
    net: model.PolicyValueNet,
    optimizer: optim.Optimizer,
    records: list[collect.GameRecord],
    cfg: config.RunConfig,
    device: torch.device,
    ppo: bool,
) -> UpdateStats:
    """PPO / GAE update with fixed advantages and optional epoch reuse.

    Advantages are computed once from captured ``Step.behavior_logp`` /
    ``Step.value_pred``, normalized, then reused across ``ppo_reuse_epochs``
    backward passes (one epoch when only GAE is on). The PPO clipped surrogate
    is used when ``ppo=True``; otherwise REINFORCE with GAE advantages.
    """
    # Pre-compute fixed advantages from captured data (never re-estimated).
    flat_steps, advantages, value_targets = _flatten_with_advantages(records, cfg)
    if not flat_steps:
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

    # Normalize advantages once across the whole batch.
    adv_arr = np.array(advantages, dtype=np.float32)
    adv_mean = float(adv_arr.mean())
    adv_std = float(adv_arr.std())
    norm_adv = ((adv_arr - adv_mean) / (adv_std + _ADV_STD_EPS)).tolist()

    # Fixed per-step scalars (stable across epochs; only the forward graph changes).
    old_logp_list = [step.behavior_logp for step in flat_steps]
    buckets = _bucketize(flat_steps)
    n_epochs = cfg.training.ppo_reuse_epochs if ppo else 1
    eps = cfg.training.ppo_clip_eps

    last_loss = last_ploss = last_vloss = last_ent = last_gnorm = 0.0
    last_clip = last_kl = 0.0

    for _ in range(n_epochs):
        chosen_logps_e: list[torch.Tensor] = []
        values_e: list[torch.Tensor] = []
        entropies_e: list[torch.Tensor] = []
        adv_parts: list[torch.Tensor] = []
        vt_parts: list[torch.Tensor] = []
        old_logp_parts: list[torch.Tensor] = []

        for bucket in buckets:
            logp_b, value_b, entropy_b, _, _ = _forward_bucket(
                net, device, flat_steps, bucket
            )
            chosen_logps_e.append(logp_b)
            values_e.append(value_b)
            entropies_e.append(entropy_b)
            adv_parts.append(
                torch.tensor(
                    [norm_adv[i] for i in bucket], dtype=torch.float32, device=device
                )
            )
            vt_parts.append(
                torch.tensor(
                    [value_targets[i] for i in bucket],
                    dtype=torch.float32,
                    device=device,
                )
            )
            if ppo:
                old_logp_parts.append(
                    torch.tensor(
                        [old_logp_list[i] for i in bucket],
                        dtype=torch.float32,
                        device=device,
                    )
                )

        logp_all = torch.cat(chosen_logps_e)
        value_flat = torch.cat(values_e)
        entropy_t = torch.cat(entropies_e).mean()
        adv_all = torch.cat(adv_parts)
        vt_all = torch.cat(vt_parts)
        value_loss = F.mse_loss(value_flat, vt_all)

        if ppo:
            old_logp_all = torch.cat(old_logp_parts)
            ratio = (logp_all - old_logp_all).exp()
            surr1 = ratio * adv_all
            surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv_all
            policy_loss_t = -torch.min(surr1, surr2).mean()
            last_clip = float(((ratio - 1.0).abs() > eps).float().mean().detach())
            last_kl = float((old_logp_all - logp_all).mean().detach())
        else:
            policy_loss_t = -(logp_all * adv_all).mean()
            last_clip = 0.0
            last_kl = 0.0

        loss = (
            policy_loss_t
            + cfg.training.value_coef * value_loss
            - cfg.training.entropy_coef * entropy_t
        )

        optimizer.zero_grad()
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=cfg.training.grad_clip
        )
        optimizer.step()

        last_loss = float(loss.detach())
        last_ploss = float(policy_loss_t.detach())
        last_vloss = float(value_loss.detach())
        last_ent = float(entropy_t.detach())
        last_gnorm = float(grad_norm)

    return UpdateStats(
        loss=last_loss,
        policy_loss=last_ploss,
        value_loss=last_vloss,
        entropy=last_ent,
        grad_norm=last_gnorm,
        advantage_mean=adv_mean,
        advantage_std=adv_std,
        n_steps=len(flat_steps),
        clip_fraction=last_clip,
        approx_kl=last_kl,
    )


def _flatten_with_advantages(
    records: list[collect.GameRecord], cfg: config.RunConfig
) -> tuple[list[steps.Step], list[float], list[float]]:
    """Flatten records and compute ``(flat_steps, advantages, value_targets)``
    for the PPO/GAE reuse path.

    With ``GAE`` reward mode the GAE kernel runs per-player using captured
    ``value_pred`` checkpoints.  With MC returns (other modes) the advantage is
    ``G − step.value_pred`` (fixed from capture time) and the value target is
    ``G`` (the MC return).
    """
    if cfg.training.reward_mode is config.RewardMode.GAE:
        return _gae_flatten(records, cfg)

    # PPO with MC returns: fix advantages pre-epoch so each reuse epoch sees the
    # same targets (value_pred captured at collection, not re-estimated each pass).
    flat_steps, returns = _flatten(records, cfg)
    advantages = [ret - step.value_pred for ret, step in zip(returns, flat_steps)]
    return flat_steps, advantages, returns


def _gae_flatten(
    records: list[collect.GameRecord], cfg: config.RunConfig
) -> tuple[list[steps.Step], list[float], list[float]]:
    """GAE advantages and value targets aligned to flattened record steps.

    Mirrors ``_decision_delta_returns``'s per-player checkpoint routing but
    calls ``timestamps.gae_advantages`` instead of ``discounted_future_returns``.
    """
    flat_steps: list[steps.Step] = []
    advantages: list[float] = []
    value_targets: list[float] = []

    basis = cfg.training.reward_basis
    discount = cfg.training.reward_discount
    lam = cfg.training.gae_lambda
    score_norm = cfg.training.score_norm
    end_bonus = cfg.training.end_game_bonus

    for record in records:
        score_0, score_1 = record.breakdowns[0].total, record.breakdowns[1].total

        # Terminal values (same logic as _decision_delta_returns).
        if basis is config.RewardBasis.OWN_SCORE:
            bonus_0 = end_bonus if record.winner == 0 else 0.0
            bonus_1 = end_bonus if record.winner == 1 else 0.0
            terminal = (score_0 + bonus_0, score_1 + bonus_1)
        else:
            bonus_0 = _winner_bonus(record.winner, end_bonus)
            terminal = (score_0 - score_1 + bonus_0, score_1 - score_0 - bonus_0)

        n_record_steps = len(record.steps)
        out_adv = [0.0] * n_record_steps
        out_vt = [0.0] * n_record_steps

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
            values = [record.steps[i].value_pred for i in indices]
            adv, vt = timestamps.gae_advantages(
                checkpoints, times, values, score_norm, discount, lam
            )
            for position, idx in enumerate(indices):
                out_adv[idx] = adv[position]
                out_vt[idx] = vt[position]

        flat_steps.extend(record.steps)
        advantages.extend(out_adv)
        value_targets.extend(out_vt)

    return flat_steps, advantages, value_targets


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


#### Minibatch helpers ####


def _minibatch_chunks(total: int, chunk_size: int) -> list[list[int]]:
    """Partition ``range(total)`` into sequential chunks of at most ``chunk_size``."""
    return [
        list(range(start, min(start + chunk_size, total)))
        for start in range(0, total, chunk_size)
    ]


def _bucketize_indices(
    flat_steps: list[steps.Step],
    indices: list[int],
) -> list[list[int]]:
    """Group a subset of global step indices into option-count buckets.

    Like ``_bucketize`` but operates on a pre-specified subset; the returned
    inner lists hold the same global indices and can be passed directly to
    ``_forward_bucket``."""
    by_edge: dict[int, list[int]] = {}
    for global_idx in indices:
        edge = _bucket_edge(flat_steps[global_idx].choices.shape[0])
        by_edge.setdefault(edge, []).append(global_idx)
    return [by_edge[edge] for edge in sorted(by_edge)]


#### Minibatch update paths ####


def _update_single_pass_minibatched(
    net: model.PolicyValueNet,
    optimizer: optim.Optimizer,
    records: list[collect.GameRecord],
    cfg: config.RunConfig,
    device: torch.device,
    imitation_phase: bool,
    minibatch_steps: int,
) -> UpdateStats:
    """Gradient-accumulation variant of ``_update_single_pass``.

    Splits the flattened batch into minibatches of ``minibatch_steps`` steps,
    accumulates gradients across them, and takes one ``optimizer.step()``.
    Reproduces the full-batch gradient up to float summation order.
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

    N = len(flat_steps)

    # Pre-compute global advantage normalization before the grad loop.
    if imitation_phase:
        # No advantages; count expert-labeled steps for the CE denominator.
        K_expert_total = float(
            sum(1.0 for step in flat_steps if step.expert_probs is not None)
        )
        adv_mean = adv_std = 0.0
        # Sentinel tensors — never indexed in imitation mode.
        norm_adv_flat = torch.empty(0, device=device)
        returns_t_pre = torch.empty(0, device=device)
    else:
        # No-grad prepass: build advantages in BUCKET ORDER, reproducing the
        # exact same float operations as _update_single_pass so mean/std are
        # bitwise identical to the full-batch path.  Scatter to flat order so
        # the grad loop can index by global step index.
        adv_parts_pre: list[torch.Tensor] = []
        bucket_global_order: list[int] = []
        with torch.no_grad():
            for bucket in _bucketize(flat_steps):
                _, value_b, _, _, _ = _forward_bucket(net, device, flat_steps, bucket)
                ret_b = torch.tensor(
                    [returns[i] for i in bucket], dtype=torch.float32, device=device
                )
                adv_parts_pre.append(ret_b - value_b)
                bucket_global_order.extend(bucket)
        adv_t = torch.cat(
            adv_parts_pre
        )  # bucket order — same as `advantage` in full-batch
        adv_mean = float(adv_t.mean().item())
        adv_std = float(adv_t.std().item())
        norm_adv_bucket = (adv_t - adv_t.mean()) / (adv_t.std() + _ADV_STD_EPS)
        # Scatter normalized advantages and flat-order returns for the grad loop.
        norm_adv_flat = torch.empty(N, dtype=torch.float32, device=device)
        for pos, global_idx in enumerate(bucket_global_order):
            norm_adv_flat[global_idx] = norm_adv_bucket[pos]
        returns_t_pre = torch.tensor(returns, dtype=torch.float32, device=device)
        K_expert_total = 0.0

    # Gradient accumulation: forward each minibatch, backward, repeat.
    optimizer.zero_grad()
    acc_loss = acc_ploss = acc_vloss = acc_ent = acc_imit = 0.0

    for mb_indices in _minibatch_chunks(N, minibatch_steps):
        mb_size = len(mb_indices)

        chosen_logps: list[torch.Tensor] = []
        values_mb: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        imit_ces: list[torch.Tensor] = []
        has_experts: list[torch.Tensor] = []
        adv_parts: list[torch.Tensor] = []
        return_parts: list[torch.Tensor] = []

        for bucket in _bucketize_indices(flat_steps, mb_indices):
            logp_b, value_b, entropy_b, imit_ce_b, has_expert_b = _forward_bucket(
                net, device, flat_steps, bucket
            )
            chosen_logps.append(logp_b)
            values_mb.append(value_b)
            entropies.append(entropy_b)
            imit_ces.append(imit_ce_b)
            has_experts.append(has_expert_b)
            if not imitation_phase:
                # Index pre-computed tensors by global step index — avoids
                # Python-float round-trips and keeps ordering consistent with
                # the forward tensors above.
                bucket_t = torch.tensor(bucket, dtype=torch.long, device=device)
                adv_parts.append(norm_adv_flat[bucket_t])
                return_parts.append(returns_t_pre[bucket_t])

        logp_mb = torch.cat(chosen_logps)
        value_mb = torch.cat(values_mb)
        entropy_mb = torch.cat(entropies)
        imit_ce_mb = torch.cat(imit_ces)
        has_expert_mb = torch.cat(has_experts)

        if imitation_phase:
            return_mb = torch.tensor(
                [returns[i] for i in mb_indices], dtype=torch.float32, device=device
            )
            value_loss_mb = F.mse_loss(value_mb, return_mb)
            # Scale CE sum by 1/K_expert_total (global denominator) so the
            # accumulated backward sums to the full-batch imitation loss.
            imit_loss_mb = (imit_ce_mb * has_expert_mb).sum() / max(K_expert_total, 1.0)
            mb_loss = (
                imit_loss_mb + (mb_size / N) * cfg.training.value_coef * value_loss_mb
            )
            policy_loss_mb = torch.zeros(1, device=device)
            entropy_t_mb = torch.zeros(1, device=device)
        else:
            norm_adv_mb = torch.cat(adv_parts)
            return_mb = torch.cat(return_parts)
            value_loss_mb = F.mse_loss(value_mb, return_mb)
            policy_loss_mb = -(logp_mb * norm_adv_mb).mean()
            entropy_t_mb = entropy_mb.mean()
            mb_loss = (mb_size / N) * (
                policy_loss_mb
                + cfg.training.value_coef * value_loss_mb
                - cfg.training.entropy_coef * entropy_t_mb
            )

        mb_loss.backward()  # pyright: ignore[reportUnknownMemberType]

        # Accumulate weighted stats for the final report.
        weight = mb_size / N
        acc_vloss += weight * float(value_loss_mb.detach())
        if imitation_phase:
            acc_imit += float((imit_ce_mb * has_expert_mb).sum().detach()) / max(
                K_expert_total, 1.0
            )
        else:
            acc_ploss += weight * float(policy_loss_mb.detach())
            acc_ent += weight * float(entropy_t_mb.detach())
        acc_loss += float(mb_loss.detach())

    grad_norm = torch.nn.utils.clip_grad_norm_(
        net.parameters(), max_norm=cfg.training.grad_clip
    )
    optimizer.step()

    return UpdateStats(
        loss=acc_loss,
        policy_loss=acc_ploss,
        value_loss=acc_vloss,
        entropy=acc_ent,
        grad_norm=float(grad_norm),
        advantage_mean=adv_mean,
        advantage_std=adv_std,
        imitation_loss=acc_imit,
        n_steps=N,
    )


def _update_reuse_minibatched(
    net: model.PolicyValueNet,
    optimizer: optim.Optimizer,
    records: list[collect.GameRecord],
    cfg: config.RunConfig,
    device: torch.device,
    ppo: bool,
    minibatch_steps: int,
) -> UpdateStats:
    """Gradient-accumulation variant of ``_update_reuse`` for PPO / GAE.

    Splits each epoch's batch into minibatches of ``minibatch_steps`` steps,
    accumulates gradients, and takes one ``optimizer.step()`` per epoch —
    preserving today's per-epoch step count and reproducing the full-batch
    gradient up to float summation order.
    """
    flat_steps, advantages, value_targets = _flatten_with_advantages(records, cfg)
    if not flat_steps:
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

    N = len(flat_steps)
    adv_arr = np.array(advantages, dtype=np.float32)
    adv_mean = float(adv_arr.mean())
    adv_std = float(adv_arr.std())
    norm_adv = ((adv_arr - adv_mean) / (adv_std + _ADV_STD_EPS)).tolist()

    old_logp_list = [step.behavior_logp for step in flat_steps]
    n_epochs = cfg.training.ppo_reuse_epochs if ppo else 1
    eps = cfg.training.ppo_clip_eps
    chunks = _minibatch_chunks(N, minibatch_steps)

    last_loss = last_ploss = last_vloss = last_ent = last_gnorm = 0.0
    last_clip = last_kl = 0.0

    for _ in range(n_epochs):
        optimizer.zero_grad()
        acc_loss_e = acc_ploss_e = acc_vloss_e = acc_ent_e = 0.0
        acc_clip_e = acc_kl_e = 0.0

        for mb_indices in chunks:
            mb_size = len(mb_indices)

            chosen_logps_e: list[torch.Tensor] = []
            values_e: list[torch.Tensor] = []
            entropies_e: list[torch.Tensor] = []
            adv_parts: list[torch.Tensor] = []
            vt_parts: list[torch.Tensor] = []
            old_logp_parts: list[torch.Tensor] = []

            for bucket in _bucketize_indices(flat_steps, mb_indices):
                logp_b, value_b, entropy_b, _, _ = _forward_bucket(
                    net, device, flat_steps, bucket
                )
                chosen_logps_e.append(logp_b)
                values_e.append(value_b)
                entropies_e.append(entropy_b)
                adv_parts.append(
                    torch.tensor(
                        [norm_adv[i] for i in bucket],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                vt_parts.append(
                    torch.tensor(
                        [value_targets[i] for i in bucket],
                        dtype=torch.float32,
                        device=device,
                    )
                )
                if ppo:
                    old_logp_parts.append(
                        torch.tensor(
                            [old_logp_list[i] for i in bucket],
                            dtype=torch.float32,
                            device=device,
                        )
                    )

            logp_mb = torch.cat(chosen_logps_e)
            value_mb = torch.cat(values_e)
            entropy_t_mb = torch.cat(entropies_e).mean()
            adv_mb = torch.cat(adv_parts)
            vt_mb = torch.cat(vt_parts)
            value_loss_mb = F.mse_loss(value_mb, vt_mb)

            if ppo:
                old_logp_mb = torch.cat(old_logp_parts)
                ratio = (logp_mb - old_logp_mb).exp()
                surr1 = ratio * adv_mb
                surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv_mb
                policy_loss_mb = -torch.min(surr1, surr2).mean()
                acc_clip_e += (mb_size / N) * float(
                    ((ratio - 1.0).abs() > eps).float().mean().detach()
                )
                acc_kl_e += (mb_size / N) * float(
                    (old_logp_mb - logp_mb).mean().detach()
                )
            else:
                policy_loss_mb = -(logp_mb * adv_mb).mean()

            mb_loss = (mb_size / N) * (
                policy_loss_mb
                + cfg.training.value_coef * value_loss_mb
                - cfg.training.entropy_coef * entropy_t_mb
            )

            mb_loss.backward()  # pyright: ignore[reportUnknownMemberType]

            # Accumulate weighted stats for the final report.
            weight = mb_size / N
            acc_ploss_e += weight * float(policy_loss_mb.detach())
            acc_vloss_e += weight * float(value_loss_mb.detach())
            acc_ent_e += weight * float(entropy_t_mb.detach())
            acc_loss_e += float(mb_loss.detach())

        grad_norm = torch.nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=cfg.training.grad_clip
        )
        optimizer.step()

        last_loss = acc_loss_e
        last_ploss = acc_ploss_e
        last_vloss = acc_vloss_e
        last_ent = acc_ent_e
        last_gnorm = float(grad_norm)
        last_clip = acc_clip_e
        last_kl = acc_kl_e

    return UpdateStats(
        loss=last_loss,
        policy_loss=last_ploss,
        value_loss=last_vloss,
        entropy=last_ent,
        grad_norm=last_gnorm,
        advantage_mean=adv_mean,
        advantage_std=adv_std,
        n_steps=N,
        clip_fraction=last_clip,
        approx_kl=last_kl,
    )


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
