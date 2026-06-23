"""The setup model's actor-critic training update.

:func:`actor_critic_update` runs one on-policy REINFORCE + value-MSE + entropy
pass over a single iteration's freshly collected samples. Each sample carries
``chosen_idx`` and ``all_candidates`` so the learner can compute a REINFORCE
gradient. Computes:

.. code-block:: text

    loss = pg_coef * (−log_softmax(policy_logits)[chosen_idx] * advantage)
         + value_coef * MSE(value_pred[chosen_idx], margin/score_norm)
         − entropy_coef * H(softmax(policy_logits))

where ``advantage = margin/score_norm − value_pred[chosen_idx].detach()``.

Returns a :class:`SetupUpdateStats` whose margin readouts are in points so the
dashboard can compare predicted against realized margin directly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from wingspan.setup_model import record
from wingspan.training import config, metrics, setup_net


def actor_critic_update(
    net: setup_net.SetupNet,
    optimizer: optim.Optimizer,
    samples: list[record.SetupSample],
    cfg: config.RunConfig,
    device: torch.device,
) -> metrics.SetupUpdateStats:
    """One on-policy REINFORCE + value-MSE + entropy pass over this iteration's
    freshly collected setup samples (actor-critic mode).

    Each sample must carry ``chosen_idx`` and ``all_candidates`` — the index of
    the selected candidate and the full ``(K, feature_dim)`` feature matrix for
    all candidates in that deal. Samples without these fields (e.g. from an
    earlier random-phase iteration) are skipped.

    Samples are grouped by candidate count so each group can be forwarded as one
    ``(batch, K, feature_dim)`` tensor: K = 504 (bonus included) or 252
    (split-bonus deferred)."""
    # Filter to samples that carry actor-critic data.
    valid = [
        sample
        for sample in samples
        if sample.chosen_idx is not None and sample.all_candidates is not None
    ]
    if not valid:
        return _empty_stats()

    net.train()
    total_loss = 0.0
    n_batches = 0
    realized_margins: list[float] = []
    predicted_chosen: list[float] = []

    # Group by candidate count so each group fits in one batched forward.
    groups: dict[int, list[record.SetupSample]] = {}
    for sample in valid:
        assert sample.all_candidates is not None
        k = sample.all_candidates.shape[0]
        groups.setdefault(k, []).append(sample)

    for group_samples in groups.values():
        loss = _ac_group_loss(group_samples, net, cfg, device)
        optimizer.zero_grad()
        loss.backward()  # pyright: ignore[reportUnknownMemberType]
        torch.nn.utils.clip_grad_norm_(
            net.parameters(), max_norm=cfg.training.grad_clip
        )
        optimizer.step()
        total_loss += float(loss.detach())
        n_batches += 1

        # Accumulate stats for the summary readout.
        for sample in group_samples:
            realized_margins.append(sample.margin)
        with torch.no_grad():
            for sample in group_samples:
                assert sample.all_candidates is not None
                assert sample.chosen_idx is not None
                feats = torch.tensor(
                    sample.all_candidates[
                        sample.chosen_idx : sample.chosen_idx + 1
                    ].astype(np.float32),
                    device=device,
                )
                predicted_chosen.append(float(net(feats)[0]) * cfg.training.score_norm)

    net.eval()
    return metrics.SetupUpdateStats(
        loss=total_loss / max(n_batches, 1),
        pred_margin_mean=float(np.mean(predicted_chosen)) if predicted_chosen else 0.0,
        realized_margin_mean=(
            float(np.mean(realized_margins)) if realized_margins else 0.0
        ),
        n_samples=len(valid),
        n_epochs=1,
    )


###### PRIVATE #######


def _ac_group_loss(
    group_samples: list[record.SetupSample],
    net: setup_net.SetupNet,
    cfg: config.RunConfig,
    device: torch.device,
) -> torch.Tensor:
    """Compute the combined actor-critic loss for one group of samples that share
    the same candidate count ``K``.

    Stacks all candidates into ``(batch, K, feature_dim)``, forwards once through
    both heads, then accumulates per-sample losses for clarity."""
    # Stack all candidates into (batch, K, feature_dim) and reshape for one forward.
    batch_size = len(group_samples)
    assert group_samples[0].all_candidates is not None
    k_size = group_samples[0].all_candidates.shape[0]
    all_cands = np.stack(
        [
            sample.all_candidates.astype(np.float32)  # type: ignore[union-attr]
            for sample in group_samples
        ]
    )  # (batch, K, feature_dim)
    feats_flat = torch.tensor(
        all_cands.reshape(batch_size * k_size, -1), dtype=torch.float32, device=device
    )
    policy_flat, value_flat = net.policy_and_value(feats_flat)
    # Reshape back to (batch, K).
    policy_logits = policy_flat.view(batch_size, k_size)
    value_preds = value_flat.view(batch_size, k_size)

    # Accumulate per-sample loss terms.
    total_pg_loss = torch.zeros(1, device=device)
    total_val_loss = torch.zeros(1, device=device)
    total_entropy = torch.zeros(1, device=device)
    for idx, sample in enumerate(group_samples):
        assert sample.chosen_idx is not None
        chosen = sample.chosen_idx
        target = torch.tensor(
            sample.margin / cfg.training.score_norm, dtype=torch.float32, device=device
        )

        # Policy gradient: advantage weighted log-prob of chosen action.
        advantage = (target - value_preds[idx, chosen].detach()).clamp(-10.0, 10.0)
        log_probs = F.log_softmax(policy_logits[idx], dim=0)
        total_pg_loss = total_pg_loss + (-log_probs[chosen] * advantage)

        # Value head: MSE against realized margin.
        total_val_loss = total_val_loss + F.mse_loss(value_preds[idx, chosen], target)

        # Entropy bonus: encourage exploration.
        probs = F.softmax(policy_logits[idx], dim=0)
        total_entropy = total_entropy + (-(probs * log_probs).sum())

    scale = 1.0 / batch_size
    return (
        scale * cfg.training.setup.pg_coef * total_pg_loss
        + scale * cfg.training.setup.value_coef * total_val_loss
        - scale * cfg.training.setup.entropy_coef * total_entropy
    )


def _empty_stats() -> metrics.SetupUpdateStats:
    return metrics.SetupUpdateStats(
        loss=0.0,
        pred_margin_mean=0.0,
        realized_margin_mean=0.0,
        n_samples=0,
        n_epochs=0,
    )
