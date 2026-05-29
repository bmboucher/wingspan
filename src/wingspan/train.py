"""Self-play data collection + a single REINFORCE training cycle.

The agent is a masked-policy network that scores each candidate at a
decision point via per-choice features (see ``wingspan.encode``). We collect
episodes by sampling actions from the softmax-of-masked-logits policy,
recording ``(state, choice_features, chosen_idx, player_id)`` transitions
for *both* seats so the same network learns from self-play. At the end of
the cycle we run a small REINFORCE update with a value baseline; the
Monte-Carlo return for each step is the terminal score advantage from that
step's POV (so player 0's and player 1's steps receive opposite-signed
signals when the game's outcome is asymmetric).

This is deliberately minimal — the goal is criterion 3: *"complete a single
training cycle starting from random weights"*. The infrastructure scales
naturally to more sophisticated algorithms (PPO/GAE, AlphaZero) later.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import random

import numpy as np
import pydantic
import torch
import torch.nn.functional as F
from torch import optim

from wingspan import decisions, encode, engine, model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public constants

SCORE_ADVANTAGE_NORM = (
    50.0  # advantage = (my_score - their_score) / SCORE_ADVANTAGE_NORM
)
DEFAULT_HIDDEN = 128
DEFAULT_LR = 3e-4
DEFAULT_EPSILON = 0.05
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
GRAD_CLIP = 5.0


# ---------------------------------------------------------------------------
# Trajectory recorder


class Step(pydantic.BaseModel):
    """One recorded transition during self-play.

    ``choices`` is variable-length per step: shape ``(n_choices, F)``. The
    training loop pads across the batch when stacking.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    state: np.ndarray  # (state_dim,)
    choices: np.ndarray  # (n_choices, choice_dim)
    chosen_idx: int  # 0..n_choices-1
    player_id: int


class Trajectory(pydantic.BaseModel):
    """A full episode's recorded transitions plus the terminal scores."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    steps: list[Step]
    scores: tuple[int, int]
    winner: int  # 0 or 1; -1 for tie


class TrainStepStats(pydantic.BaseModel):
    """Summary metrics from one optimizer update."""

    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    n_steps: int


# ---------------------------------------------------------------------------
# Policy agent that records its decisions


def make_policy_agent(
    net: model.PolicyValueNet,
    device: torch.device,
    rng: random.Random,
    record_into: list[Step],
    epsilon: float = DEFAULT_EPSILON,
) -> engine.Agent:
    """Build an agent that consults ``net`` and appends each decision it
    makes to ``record_into``. Every decision the agent sees is recorded —
    in self-play this means both seats contribute steps, tagged by
    ``player_id``."""

    def agent[C: decisions.Choice](
        eng: engine.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        n_choices = len(decision.choices)
        if n_choices == 1:
            # No degrees of freedom — don't record (loss contribution is 0
            # anyway and skipping keeps the buffer smaller).
            return decision.choices[0]

        state_vec = encode.encode_state(eng.state, decision)
        choice_feats = encode.encode_choices(decision, eng.state)
        chosen_idx = _sample_choice(net, device, state_vec, choice_feats, rng, epsilon)
        record_into.append(
            Step(
                state=state_vec,
                choices=choice_feats,
                chosen_idx=chosen_idx,
                player_id=decision.player_id,
            )
        )
        return decision.choices[chosen_idx]

    return agent


# ---------------------------------------------------------------------------
# Data collection


def collect_episode(
    net: model.PolicyValueNet,
    device: torch.device,
    rng: random.Random,
    epsilon: float,
    seed: int,
) -> Trajectory:
    """Play one self-play game where both seats consult ``net`` and record
    their decisions into a shared step buffer."""
    eng, _, _, _ = engine.Engine.create(seed=seed)
    recorded: list[Step] = []
    agent_a = make_policy_agent(net, device, rng, recorded, epsilon=epsilon)
    agent_b = make_policy_agent(net, device, rng, recorded, epsilon=epsilon)
    engine.Engine.play_one_game(eng.state, (agent_a, agent_b))
    s0 = eng.state.players[0].final_score or 0
    s1 = eng.state.players[1].final_score or 0
    winner = 0 if s0 > s1 else (1 if s1 > s0 else -1)
    return Trajectory(steps=recorded, scores=(s0, s1), winner=winner)


# ---------------------------------------------------------------------------
# Training step


def train_step(
    net: model.PolicyValueNet,
    optimizer: optim.Optimizer,
    trajectories: list[Trajectory],
    device: torch.device,
) -> TrainStepStats:
    """One REINFORCE update over the flattened steps of ``trajectories``.

    Each step's advantage is computed from its own player's POV — so a game
    in which player 0 wins by 10 produces ``+10/SCORE_ADVANTAGE_NORM`` for
    every step recorded by player 0 and ``-10/SCORE_ADVANTAGE_NORM`` for
    every step recorded by player 1. The same network learns from both
    sides; symmetry comes from the per-step POV plus POV-aware state
    encoding."""
    if not trajectories:
        return TrainStepStats(
            loss=0.0,
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            n_steps=0,
        )

    flat_states: list[np.ndarray] = []
    flat_choices: list[np.ndarray] = []
    flat_idx: list[int] = []
    flat_returns: list[float] = []
    flat_n_choices: list[int] = []
    for tr in trajectories:
        score_self_minus_other = [
            (tr.scores[0] - tr.scores[1]) / SCORE_ADVANTAGE_NORM,
            (tr.scores[1] - tr.scores[0]) / SCORE_ADVANTAGE_NORM,
        ]
        for st in tr.steps:
            flat_states.append(st.state)
            flat_choices.append(st.choices)
            flat_idx.append(st.chosen_idx)
            flat_returns.append(score_self_minus_other[st.player_id])
            flat_n_choices.append(st.choices.shape[0])

    if not flat_states:
        return TrainStepStats(
            loss=0.0,
            policy_loss=0.0,
            value_loss=0.0,
            entropy=0.0,
            n_steps=0,
        )

    # Pad choice tensors across the batch so a single forward pass handles
    # all of them. Mask carries the variable cardinality.
    batch_size = len(flat_states)
    max_k = max(flat_n_choices)
    state_batch = np.stack(flat_states)
    choice_batch = np.zeros(
        (batch_size, max_k, encode.CHOICE_FEATURE_DIM), dtype=np.float32
    )
    mask_batch = np.zeros((batch_size, max_k), dtype=np.float32)
    for i, (choice_feats, count) in enumerate(zip(flat_choices, flat_n_choices)):
        choice_batch[i, :count] = choice_feats
        mask_batch[i, :count] = 1.0

    state_t = torch.tensor(state_batch, dtype=torch.float32, device=device)
    choice_t = torch.tensor(choice_batch, dtype=torch.float32, device=device)
    mask_t = torch.tensor(mask_batch, dtype=torch.float32, device=device)
    idx_t = torch.tensor(flat_idx, dtype=torch.long, device=device)
    ret_t = torch.tensor(flat_returns, dtype=torch.float32, device=device)

    logits, value = net(state_t, choice_t, mask_t)
    logp = F.log_softmax(logits, dim=-1)

    # Policy loss: REINFORCE w/ value baseline, gather log-prob at chosen
    # index. Padding rows have -inf there but the chosen index is always a
    # real position by construction.
    chosen_logp = logp.gather(1, idx_t.unsqueeze(1)).squeeze(1)
    advantages = ret_t - value.detach()
    policy_loss = -(chosen_logp * advantages).mean()

    value_loss = F.mse_loss(value, ret_t)

    # Entropy regularizer over legal slots only — torch.where prevents NaN
    # from 0 * -inf when summing over padding columns.
    zeros = torch.zeros_like(logp)
    legal_logp = torch.where(mask_t > 0.5, logp, zeros)
    legal_p = torch.where(mask_t > 0.5, logp.exp(), zeros)
    entropy = -(legal_p * legal_logp).sum(dim=-1).mean()

    loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP)
    optimizer.step()
    return TrainStepStats(
        loss=float(loss.detach()),
        policy_loss=float(policy_loss.detach()),
        value_loss=float(value_loss.detach()),
        entropy=float(entropy.detach()),
        n_steps=batch_size,
    )


# ---------------------------------------------------------------------------
# CLI


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Wingspan training cycle.")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/wingspan_cycle0.pt"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    device = torch.device(args.device)
    logger.info("Device: %s", device)
    rng = random.Random(args.seed)

    net = model.PolicyValueNet().to(device)
    optimizer: optim.Optimizer = optim.Adam(net.parameters(), lr=args.lr)
    logger.info(
        "Model: %s parameters", sum(param.numel() for param in net.parameters())
    )

    for epoch in range(args.epochs):
        trajs: list[Trajectory] = []
        wins = 0
        score_diffs: list[int] = []
        for ep in range(args.episodes):
            tr = collect_episode(
                net,
                device,
                rng,
                args.epsilon,
                seed=args.seed * 10000 + epoch * 1000 + ep,
            )
            trajs.append(tr)
            if tr.winner == 0:
                wins += 1
            score_diffs.append(tr.scores[0] - tr.scores[1])
        stats = train_step(net, optimizer, trajs, device)
        logger.info(
            "Epoch %d: P0 wins=%d/%d, avg_score_diff=%+.2f, "
            "loss=%.4f (policy=%.4f value=%.4f entropy=%.4f), n_steps=%d",
            epoch,
            wins,
            args.episodes,
            sum(score_diffs) / len(score_diffs),
            stats.loss,
            stats.policy_loss,
            stats.value_loss,
            stats.entropy,
            stats.n_steps,
        )

    ckpt = pathlib.Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "args": vars(args)}, ckpt)
    logger.info("Saved checkpoint to %s", ckpt)
    return 0


###### PRIVATE #######


def _sample_choice(
    net: model.PolicyValueNet,
    device: torch.device,
    state_vec: np.ndarray,
    choice_feats: np.ndarray,
    rng: random.Random,
    epsilon: float,
) -> int:
    """Pick a choice via epsilon-greedy on the policy."""
    n_choices = choice_feats.shape[0]
    if rng.random() < epsilon:
        return rng.randrange(n_choices)

    with torch.no_grad():
        state_t = torch.tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(
            0
        )
        choice_t = torch.tensor(
            choice_feats, dtype=torch.float32, device=device
        ).unsqueeze(0)
        mask_t = torch.ones((1, n_choices), dtype=torch.float32, device=device)
        logits, _ = net(state_t, choice_t, mask_t)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    total = probs.sum()
    if not np.isfinite(total) or total <= 0:
        return rng.randrange(n_choices)
    probs = (probs / total).tolist()
    # Use the seeded ``rng`` (not numpy's global state) so episodes stay
    # reproducible from the user-supplied seed.
    return _weighted_choice(rng, probs)


def _weighted_choice(rng: random.Random, weights: list[float]) -> int:
    """``random.Random.choices``-equivalent that returns the picked index."""
    roll = rng.random()
    acc = 0.0
    for i, weight in enumerate(weights):
        acc += weight
        if roll < acc:
            return i
    return len(weights) - 1


if __name__ == "__main__":
    raise SystemExit(main())
