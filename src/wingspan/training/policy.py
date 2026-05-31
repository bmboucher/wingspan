"""Single-decision policy inference: sample (collection) and greedy (eval).

Both helpers run one batch-of-one forward pass through ``PolicyValueNet`` for a
single decision and turn its per-candidate logits into a chosen index. They
differ only in selection rule:

* :func:`sample_action` draws from the softmax — the on-policy behaviour the
  REINFORCE update assumes (no epsilon-greedy; exploration is controlled by the
  entropy bonus, TRAINING.md §3.3).
* :func:`greedy_action` takes the argmax — used by the evaluation harness, which
  measures *strength*, not exploration (TRAINING.md §7.3).

:func:`policy_probs` exposes the underlying softmax distribution itself, for
callers that want every option's probability (e.g. the selfplay log annotator)
rather than just the selected index.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from wingspan import model


def sample_action(
    net: model.PolicyValueNet,
    device: torch.device,
    state_vec: np.ndarray,
    choice_feats: np.ndarray,
    family_idx: int,
    rng: random.Random,
) -> int:
    """Sample a choice index from the current policy (on-policy)."""
    probs = policy_probs(net, device, state_vec, choice_feats, family_idx)
    return sample_index_from_probs(probs, choice_feats.shape[0], rng)


def greedy_action(
    net: model.PolicyValueNet,
    device: torch.device,
    state_vec: np.ndarray,
    choice_feats: np.ndarray,
    family_idx: int,
) -> int:
    """Pick the argmax choice index — deterministic strength play."""
    probs = policy_probs(net, device, state_vec, choice_feats, family_idx)
    return int(np.argmax(probs))


def sample_index_from_probs(
    probs: np.ndarray, n_choices: int, rng: random.Random
) -> int:
    """Sample a choice index from a 1-D probability vector with the seeded
    ``rng``. Falls back to a uniform pick when the policy row is degenerate
    (non-finite or all-zero). Factored out so batched collection can sample
    from server-computed probabilities with exactly the single-decision rule."""
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return rng.randrange(n_choices)
    return _weighted_index(rng, (probs / total).tolist())


def policy_probs(
    net: model.PolicyValueNet,
    device: torch.device,
    state_vec: np.ndarray,
    choice_feats: np.ndarray,
    family_idx: int,
) -> np.ndarray:
    """Softmax over the candidate logits for one decision (no padding needed —
    every row of ``choice_feats`` is a real, legal choice). Public so callers
    that want the full distribution (e.g. the selfplay log annotator) can read
    it directly rather than only the sampled / argmax index."""
    n_choices = choice_feats.shape[0]
    with torch.no_grad():
        state_t = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(
            0
        )
        choice_t = torch.tensor(
            choice_feats, dtype=torch.float32, device=device
        ).unsqueeze(0)
        mask_t = torch.ones((1, n_choices), dtype=torch.float32, device=device)
        family_t = torch.tensor([family_idx], dtype=torch.long, device=device)
        logits, _ = net(state_t, choice_t, mask_t, family_t)
        return F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()


###### PRIVATE #######


def _weighted_index(rng: random.Random, weights: list[float]) -> int:
    """Return an index sampled in proportion to ``weights`` using the seeded
    ``rng`` (not numpy's global state) so episodes stay reproducible."""
    roll = rng.random()
    acc = 0.0
    for i, weight in enumerate(weights):
        acc += weight
        if roll < acc:
            return i
    return len(weights) - 1
