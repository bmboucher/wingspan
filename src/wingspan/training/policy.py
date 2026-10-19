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
(e.g. the ``players.factory`` log annotator). :func:`policy_probs` is a thin
wrapper that returns only the probabilities for callers that don't need the raw
logits.

:func:`greedy_agent` wraps a net into an :class:`engine.Agent` that plays the
argmax at every decision — the strength-play agent shared by the evaluation
harness and the tournament so both measure strength with identical play.
"""

from __future__ import annotations

import random
import typing

import numpy as np
import torch
import torch.nn.functional as F

from wingspan import decisions, engine, model, sampling, setup_model, state
from wingspan.training import setup_net as setup_net_module


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


def setup_scores_and_probs(
    setup_net: setup_net_module.SetupNet,
    decision: decisions.SetupDecision,
    game_state: state.GameState,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score every choice in ``decision`` through the setup net.

    Returns ``(scores, probs, vecs)`` where ``scores`` is the per-choice policy
    logit vector (actor-critic mode, the only mode used in practice), ``probs``
    is the softmax distribution, and ``vecs`` is the ``(N, D)`` matrix of raw
    candidate feature vectors, all aligned to ``decision.choices``. The value
    head is the state-only critic ``V(s)`` — identical across candidates — so a
    value-only net cannot rank keeps; that fallback scores them uniformly."""
    context = setup_model.SetupContext.from_state(game_state, decision.dealt_bonus)

    # Encode each choice through the net's own encode_candidate seam (not the
    # free function paired with the net's encoding by hand), so a compat-era net
    # (e.g. SetupNetV1_4) carries its own frozen pricing — the "encode through
    # the net" rule (docs/VERSIONING.md).
    vecs = np.stack(
        [
            setup_net.encode_candidate(
                setup_model.SetupCandidate.from_setup_choice(choice), context
            )
            for choice in decision.choices
        ]
    )
    feats = torch.tensor(vecs, dtype=torch.float32, device=device)

    # Mirror collect.py: actor-critic nets rank candidates by policy logits. The
    # value-only fallback reads the state-only V(s), identical for every
    # candidate of a deal — a uniform (degenerate) ranking, never used in practice.
    with torch.no_grad():
        if setup_net.arch.use_policy_head:
            scores = setup_net.policy_logits(feats).cpu().numpy()
        else:
            scores = setup_net(feats).cpu().numpy()

    shifted = scores - scores.max()
    weights = np.exp(shifted)
    return scores, weights / weights.sum(), vecs


def greedy_setup_choice(
    setup_net: setup_net_module.SetupNet,
    decision: decisions.SetupDecision,
    game_state: state.GameState,
    device: torch.device,
) -> decisions.SetupChoice:
    """The setup net's argmax keep for ``decision`` — the canonical greedy setup
    resolution shared by :func:`greedy_agent` and the ``players.factory`` agent."""
    _, probs, _ = setup_scores_and_probs(setup_net, decision, game_state, device)
    return decision.choices[int(np.argmax(probs))]


def greedy_agent(
    net: model.PolicyValueNet,
    device: torch.device,
    *,
    setup_net: setup_net_module.SetupNet | None = None,
) -> engine.Agent:
    """Wrap ``net`` into a non-recording agent that plays the argmax of the
    current policy.

    Single-option decisions short-circuit (no inference needed), and a
    setup-excluded net (``include_setup=False``) resolves the combined opening
    ``SetupDecision`` with ``setup_net``'s argmax keep when one is supplied
    (:func:`greedy_setup_choice`), otherwise with a uniform-random legal choice
    rather than scoring a decision it was never trained to encode. Shared by the
    eval harness (``evaluate._greedy_agent``) and the tournament so both measure
    strength with the exact same play.
    """

    def agent[C: decisions.Choice](
        eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if len(decision.choices) == 1:
            return decision.choices[0]
        if not net.include_setup and decisions.is_setup_decision(decision):
            if setup_net is not None:
                setup_decision = typing.cast(decisions.SetupDecision, decision)
                return typing.cast(
                    C,
                    greedy_setup_choice(setup_net, setup_decision, eng.state, device),
                )
            return decisions.random_choice(decision, eng.state.rng)
        family_idx = decisions.family_index_for(type(decision))
        state_vec = net.encode_state(eng.state, decision)
        choice_feats = net.encode_choices(decision, eng.state)
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
    return sampling.weighted_index(rng, (probs / total).tolist())


def behavior_logp(probs: np.ndarray, chosen_idx: int, n_choices: int) -> float:
    """Log probability of ``chosen_idx`` under the collection-time softmax.

    Mirrors the degeneracy fallback in :func:`sample_index_from_probs`: returns
    ``−log(n_choices)`` (the uniform log-prob) when the distribution is
    degenerate (non-finite or all-zero), so the PPO ratio is 1 for those steps.
    Clamps the non-degenerate case to avoid ``log(0)``."""
    total = float(probs.sum())
    if not np.isfinite(total) or total <= 0.0:
        return -float(np.log(max(n_choices, 1)))
    p_chosen = float(probs[chosen_idx]) / total
    return float(np.log(max(p_chosen, 1e-45)))


def policy_value_and_probs(
    net: model.PolicyValueNet,
    device: torch.device,
    state_vec: np.ndarray,
    choice_feats: np.ndarray,
    family_idx: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(logits, probs, value)`` for one decision.

    Like :func:`policy_logits_and_probs` but also surfaces the critic's raw
    value scalar (deciding-player POV, not divided by ``score_norm``). The
    caller decides whether to use or discard the value."""
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
        logits, value = net(state_t, choice_t, mask_t, family_t)
        logits_1d = logits.squeeze(0).cpu().numpy()
        probs_1d = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        value_scalar = float(value.squeeze(0))
        return logits_1d, probs_1d, value_scalar


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
    ``choice_feats`` is a real, legal choice. Thin wrapper around
    :func:`policy_value_and_probs` that discards the critic value."""
    logits, probs, _ = policy_value_and_probs(
        net, device, state_vec, choice_feats, family_idx
    )
    return logits, probs


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
