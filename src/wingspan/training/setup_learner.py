"""The setup model's actor-critic training update.

:func:`actor_critic_update` runs one on-policy REINFORCE + value-MSE + entropy
pass over a single iteration's freshly collected samples.

The critic is the **state-only** value head ``V(s)`` — a function of the deal
context alone (tray / birdfeeder / round goals / bonus-on-offer), not the chosen
keep. So the advantage ``target − V(s)`` carries a real policy gradient, unlike
the former per-candidate ``Q(s, a_chosen)`` whose advantage self-cancelled (its
conditional mean given the chosen action was ≈ 0, leaving the entropy bonus to
collapse the logits toward uniform — ``docs/TRAINING.md §6.5``).

The ``target`` is the in-game return at this seat's ``t=0`` setup decision
(:func:`wingspan.training.returns.setup_return`), so the setup and in-game value
functions are trained on the *same* return definition under any ``reward_mode``
/ discount / basis / bonus. The policy-gradient advantage is whitened per-batch
exactly as the in-game learner does (``returns.ADV_STD_EPS``).

.. code-block:: text

    A      = setup_return − V(s).detach()            (whitened across the batch)
    loss   = pg_coef · mean(−log_softmax(policy_logits)[chosen] · Â)
           + value_coef · MSE(V(s), setup_return)
           − entropy_coef · mean(H(softmax(policy_logits)))

``V(s)`` is computed once per deal from the state stripes (identical across a
deal's K candidates, so read from row 0); the policy logits are forwarded per
candidate-count group. A single combined backward/step covers the whole
iteration, mirroring the in-game learner's single update.

Returns a :class:`SetupUpdateStats` whose margin readouts are in points so the
dashboard can compare predicted (``V(s) × score_norm``) against realized margin.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from wingspan.setup_model import record
from wingspan.training import config, metrics, returns, setup_net


def actor_critic_update(
    net: setup_net.SetupNet,
    optimizer: optim.Optimizer,
    samples: list[record.SetupSample],
    cfg: config.RunConfig,
    device: torch.device,
) -> metrics.SetupUpdateStats:
    """One on-policy REINFORCE + value-MSE + entropy pass over this iteration's
    freshly collected setup samples.

    Each sample must carry ``chosen_idx`` and ``all_candidates`` — the index of
    the selected candidate and the full ``(K, feature_dim)`` feature matrix for
    all candidates in that deal (K = 504 bonus-included, or 252 split-bonus).
    Samples without these (e.g. from an earlier random-phase iteration) are
    skipped. A single combined optimizer step covers the whole batch."""
    # Filter to samples that carry actor-critic data.
    valid = [
        sample
        for sample in samples
        if sample.chosen_idx is not None and sample.all_candidates is not None
    ]
    if not valid:
        return _empty_stats()

    net.train()

    # Critic V(s): one forward per deal over the state stripes. Row 0 of each
    # candidate matrix is canonical — its tray / feeder / goals / bonus-on-offer
    # stripes are byte-identical across the deal's K candidates.
    state_rows = np.stack([_state_row(sample) for sample in valid])  # (N, feature_dim)
    values = net(torch.tensor(state_rows, dtype=torch.float32, device=device))  # (N,)
    targets = torch.tensor(
        [_setup_target(sample, cfg) for sample in valid],
        dtype=torch.float32,
        device=device,
    )  # (N,)
    value_loss = F.mse_loss(values, targets)
    # Dashboard readout: mean predicted margin V(s) in points.
    pred_margin_mean = float(values.detach().mean()) * cfg.training.score_norm

    # Globally-whitened advantage (the in-game learner's §3.3 normalization).
    # Biased std (``unbiased=False``) so a single-sample batch yields 0, not NaN.
    advantage = targets - values.detach()
    norm_advantage = (advantage - advantage.mean()) / (
        advantage.std(unbiased=False) + returns.ADV_STD_EPS
    )

    # Actor: per-candidate-count-group policy logits; pg + entropy weighted by
    # the globally-whitened advantage.
    pg_terms, entropy_terms = _policy_terms(net, valid, norm_advantage, device)
    pg_loss = torch.stack(pg_terms).mean()
    entropy = torch.stack(entropy_terms).mean()

    loss = (
        cfg.training.setup.pg_coef * pg_loss
        + cfg.training.setup.value_coef * value_loss
        - cfg.training.setup.entropy_coef * entropy
    )
    optimizer.zero_grad()
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=cfg.training.grad_clip)
    optimizer.step()

    net.eval()
    return metrics.SetupUpdateStats(
        loss=float(loss.detach()),
        pred_margin_mean=pred_margin_mean,
        realized_margin_mean=float(np.mean([sample.margin for sample in valid])),
        n_samples=len(valid),
        n_epochs=1,
    )


###### PRIVATE #######


def _state_row(sample: record.SetupSample) -> np.ndarray:
    """The deal's state-input row for ``V(s)``: row 0 of ``all_candidates``.

    Every candidate of a deal shares the same state stripes, so the value head
    (which reads only those) gives the same answer for any row; row 0 is the
    canonical choice."""
    assert sample.all_candidates is not None
    return sample.all_candidates[0].astype(np.float32)


def _setup_target(sample: record.SetupSample, cfg: config.RunConfig) -> float:
    """The in-game return at this seat's ``t=0`` setup decision — the actor-critic
    target, consistent with the main learner (:func:`returns.setup_return`)."""
    return returns.setup_return(
        sample.own_total,
        sample.opp_total,
        sample.won,
        sample.margin_checkpoints,
        sample.score_checkpoints,
        sample.decision_times,
        sample.final_timestamp,
        cfg.training,
    )


def _policy_terms(
    net: setup_net.SetupNet,
    valid: list[record.SetupSample],
    norm_advantage: torch.Tensor,
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-sample policy-gradient and entropy terms.

    Samples are grouped by candidate count ``K`` so each group forwards as one
    ``(batch·K, feature_dim)`` tensor through the policy head; each sample's
    pg term is weighted by its (already globally-whitened) advantage."""
    groups: dict[int, list[int]] = {}
    for global_idx, sample in enumerate(valid):
        assert sample.all_candidates is not None
        groups.setdefault(sample.all_candidates.shape[0], []).append(global_idx)

    pg_terms: list[torch.Tensor] = []
    entropy_terms: list[torch.Tensor] = []
    for indices in groups.values():
        candidates = np.stack([_candidates(valid[i]) for i in indices])  # (G, K, D)
        group_size, k_size = candidates.shape[0], candidates.shape[1]
        feats_flat = torch.tensor(
            candidates.reshape(group_size * k_size, -1),
            dtype=torch.float32,
            device=device,
        )
        policy_logits = net.policy_logits(feats_flat).view(group_size, k_size)
        for row, global_idx in enumerate(indices):
            chosen = valid[global_idx].chosen_idx
            assert chosen is not None
            log_probs = F.log_softmax(policy_logits[row], dim=0)
            pg_terms.append(-log_probs[chosen] * norm_advantage[global_idx])
            entropy_terms.append(-(log_probs.exp() * log_probs).sum())
    return pg_terms, entropy_terms


def _candidates(sample: record.SetupSample) -> np.ndarray:
    """The sample's ``(K, feature_dim)`` candidate matrix as float32."""
    assert sample.all_candidates is not None
    return sample.all_candidates.astype(np.float32)


def _empty_stats() -> metrics.SetupUpdateStats:
    return metrics.SetupUpdateStats(
        loss=0.0,
        pred_margin_mean=0.0,
        realized_margin_mean=0.0,
        n_samples=0,
        n_epochs=0,
    )
