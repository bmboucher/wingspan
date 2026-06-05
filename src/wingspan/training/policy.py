"""Single-decision policy inference: sample (collection) and greedy (eval).

Both helpers run one batch-of-one forward pass through ``PolicyValueNet`` for a
single decision and turn its per-candidate logits into a chosen index. They
differ only in selection rule:

* :func:`sample_action` draws from the softmax — the on-policy behaviour the
  REINFORCE update assumes (no epsilon-greedy; exploration is controlled by the
  entropy bonus, TRAINING.md §3.3).
* :func:`greedy_action` takes the argmax — used by the evaluation harness, which
  measures *strength*, not exploration (TRAINING.md §7.3).

:func:`policy_logits_and_probs` exposes the underlying logits and the softmax
distribution itself, for callers that want every option's raw score and probability
(e.g. the selfplay log annotator). :func:`policy_probs` is a thin wrapper that
returns only the probabilities for callers that don't need the raw logits.

:func:`greedy_agent` wraps a net into an :class:`engine.Agent` that plays the
argmax at every decision — the strength-play agent shared by the evaluation
harness and the tournament so both measure strength with identical play.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn.functional as F

from wingspan import decisions, encode, engine, model


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


def greedy_agent(net: model.PolicyValueNet, device: torch.device) -> engine.Agent:
    """Wrap ``net`` into a non-recording agent that plays the argmax of the
    current policy.

    Single-option decisions short-circuit (no inference needed), and a
    setup-excluded net (``include_setup=False``) resolves the combined opening
    ``SetupDecision`` with a uniform-random legal choice rather than scoring a
    decision it was never trained to encode. Shared by the eval harness
    (``evaluate._greedy_agent``) and the tournament so both measure strength with
    the exact same play.
    """

    def agent[C: decisions.Choice](
        eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if len(decision.choices) == 1:
            return decision.choices[0]
        if not net.include_setup and decisions.is_setup_decision(decision):
            return decisions.random_choice(decision, eng.state.rng)
        family_idx = decisions.family_index_for(type(decision))
        state_vec = encode.encode_state(eng.state, decision, net.spec)
        choice_feats = encode.encode_choices(decision, eng.state, net.spec)
        idx = greedy_action(net, device, state_vec, choice_feats, family_idx)
        return decision.choices[idx]

    return agent


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


def policy_logits_and_probs(
    net: model.PolicyValueNet,
    device: torch.device,
    state_vec: np.ndarray,
    choice_feats: np.ndarray,
    family_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(logits, probs)`` for one decision.

    ``logits`` is the raw 1-D pre-softmax score vector; ``probs`` is the
    corresponding softmax distribution. No padding is needed — every row of
    ``choice_feats`` is a real, legal choice."""
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
        logits_1d = logits.squeeze(0).cpu().numpy()
        probs_1d = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return logits_1d, probs_1d


def policy_probs(
    net: model.PolicyValueNet,
    device: torch.device,
    state_vec: np.ndarray,
    choice_feats: np.ndarray,
    family_idx: int,
) -> np.ndarray:
    """Softmax over the candidate logits for one decision. Thin wrapper around
    :func:`policy_logits_and_probs` for callers that need only the probabilities."""
    return policy_logits_and_probs(net, device, state_vec, choice_feats, family_idx)[1]


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
