"""The setup model's training updates: offline fit + on-policy MSE or actor-critic.

Three update functions are exposed:

* :func:`offline_fit` — the one-time pass at ``setup_train_iter`` over every
  recorded ``(features, margin)`` sample from the record window (``setup_offline_epochs``
  epochs, shuffled minibatches). Always targets the **value head** via MSE,
  regardless of actor-critic mode. Warm-starts the net from the random-setup data.
* :func:`online_update` — one MSE epoch over a single iteration's freshly
  collected samples, used in the non-actor-critic MODEL_DRIVEN phase.
* :func:`actor_critic_update` — one on-policy REINFORCE + value-MSE + entropy
  pass over a single iteration's freshly collected samples. Requires that each
  sample carries ``chosen_idx`` and ``all_candidates`` (populated when
  ``setup_use_actor_critic=True``). Computes:

  .. code-block:: text

      loss = pg_coef * (−log_softmax(policy_logits)[chosen_idx] * advantage)
           + value_coef * MSE(value_pred[chosen_idx], margin/score_norm)
           − entropy_coef * H(softmax(policy_logits))

  where ``advantage = margin/score_norm − value_pred[chosen_idx].detach()``.

All three return a :class:`SetupUpdateStats` whose margin readouts are in points
so the dashboard can compare predicted against realized margin directly.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from wingspan import setup_model
from wingspan.setup_model import record
from wingspan.training import config, metrics, setup_net

# Cap on the rows forwarded once at the end of a fit purely to report a mean
# predicted margin — keeps the summary forward bounded when the offline window
# holds hundreds of thousands of samples.
_SUMMARY_SAMPLE_CAP = 4096


def offline_fit(
    net: setup_net.SetupNet,
    optimizer: optim.Optimizer,
    store: record.SetupDataStore,
    cfg: config.TrainConfig,
    device: torch.device,
) -> metrics.SetupUpdateStats:
    """Fit the setup net to every recorded sample (the one-time pass at
    ``setup_train_iter``). Materializes the whole window in memory — a one-time
    cost; the window can be shortened via ``setup_record_start_iter`` if it grows
    too large."""
    features, margins = _materialize(store)
    if features.shape[0] == 0:
        return _empty_stats()
    return _fit(
        net, optimizer, features, margins, cfg, device, epochs=cfg.setup_offline_epochs
    )


def online_update(
    net: setup_net.SetupNet,
    optimizer: optim.Optimizer,
    samples: list[record.SetupSample],
    cfg: config.TrainConfig,
    device: torch.device,
) -> metrics.SetupUpdateStats:
    """One MSE epoch over this iteration's freshly collected setup samples."""
    if not samples:
        return _empty_stats()
    features = np.stack([sample.features.astype(np.float32) for sample in samples])
    margins = np.array([sample.margin for sample in samples], dtype=np.float32)
    return _fit(net, optimizer, features, margins, cfg, device, epochs=1)


def actor_critic_update(
    net: setup_net.SetupNet,
    optimizer: optim.Optimizer,
    samples: list[record.SetupSample],
    cfg: config.TrainConfig,
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
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg.grad_clip)
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
                predicted_chosen.append(float(net(feats)[0]) * cfg.score_norm)

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


def _materialize(store: record.SetupDataStore) -> tuple[np.ndarray, np.ndarray]:
    """Load every stored sample into a ``(N, feature_dim)`` feature matrix and an
    ``(N,)`` margin vector."""
    feats: list[np.ndarray] = []
    margins: list[float] = []
    for sample in store.iter_samples():
        feats.append(sample.features.astype(np.float32))
        margins.append(sample.margin)
    if not feats:
        empty = np.zeros((0, setup_model.SETUP_FEATURE_DIM), dtype=np.float32)
        return empty, np.zeros((0,), dtype=np.float32)
    return np.stack(feats), np.array(margins, dtype=np.float32)


def _fit(
    net: setup_net.SetupNet,
    optimizer: optim.Optimizer,
    features: np.ndarray,
    margins: np.ndarray,
    cfg: config.TrainConfig,
    device: torch.device,
    *,
    epochs: int,
) -> metrics.SetupUpdateStats:
    """Run ``epochs`` shuffled-minibatch MSE passes of ``net(features)`` against
    ``margins / score_norm`` and return the averaged stats."""
    n_samples = features.shape[0]
    targets = margins / cfg.score_norm
    generator = np.random.default_rng(cfg.seed)
    batch_size = cfg.setup_offline_batch_size

    net.train()
    total_loss = 0.0
    n_batches = 0
    for _ in range(epochs):
        order = generator.permutation(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = order[start : start + batch_size]
            feats_t = torch.tensor(features[idx], dtype=torch.float32, device=device)
            target_t = torch.tensor(targets[idx], dtype=torch.float32, device=device)
            pred = net(feats_t)
            loss = F.mse_loss(pred, target_t)
            optimizer.zero_grad()
            loss.backward()  # pyright: ignore[reportUnknownMemberType]
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg.grad_clip)
            optimizer.step()
            total_loss += float(loss.detach())
            n_batches += 1
    net.eval()

    pred_margin_mean = _mean_predicted_margin(net, features, cfg, device)
    return metrics.SetupUpdateStats(
        loss=total_loss / max(n_batches, 1),
        pred_margin_mean=pred_margin_mean,
        realized_margin_mean=float(margins.mean()),
        n_samples=n_samples,
        n_epochs=epochs,
    )


def _mean_predicted_margin(
    net: setup_net.SetupNet,
    features: np.ndarray,
    cfg: config.TrainConfig,
    device: torch.device,
) -> float:
    """Mean predicted margin (in points) over a capped subsample of ``features``,
    for the dashboard's predicted-vs-realized readout."""
    sample = features[:_SUMMARY_SAMPLE_CAP]
    with torch.no_grad():
        feats_t = torch.tensor(sample, dtype=torch.float32, device=device)
        pred = net(feats_t).cpu().numpy()
    return float(pred.mean()) * cfg.score_norm


def _ac_group_loss(
    group_samples: list[record.SetupSample],
    net: setup_net.SetupNet,
    cfg: config.TrainConfig,
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
            sample.margin / cfg.score_norm, dtype=torch.float32, device=device
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
        scale * cfg.setup_pg_coef * total_pg_loss
        + scale * cfg.setup_value_coef * total_val_loss
        - scale * cfg.setup_entropy_coef * total_entropy
    )


def _empty_stats() -> metrics.SetupUpdateStats:
    return metrics.SetupUpdateStats(
        loss=0.0,
        pred_margin_mean=0.0,
        realized_margin_mean=0.0,
        n_samples=0,
        n_epochs=0,
    )
