"""The setup model's training updates: offline fit + on-policy MSE.

The setup model is a value-regression contextual bandit, so both its updates are
plain mean-squared-error regression of the net's scalar output against the
realized score margin (scaled by ``score_norm``, matching the in-game return):

* :func:`offline_fit` — the one-time pass at ``setup_train_iter`` over every
  recorded ``(features, margin)`` sample from the record window (``setup_offline_epochs``
  epochs, shuffled minibatches). Warm-starts the net from the random-setup data.
* :func:`online_update` — one epoch over a single iteration's freshly collected
  samples, run every iteration once the net is driving setup selection, so it
  keeps tracking the (moving) value of setups under the current in-game policy.

Both return a :class:`SetupUpdateStats` whose margin readouts are in points (the
normalized target multiplied back by ``score_norm``) so the dashboard can compare
predicted against realized margin directly.
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


def _empty_stats() -> metrics.SetupUpdateStats:
    return metrics.SetupUpdateStats(
        loss=0.0,
        pred_margin_mean=0.0,
        realized_margin_mean=0.0,
        n_samples=0,
        n_epochs=0,
    )
