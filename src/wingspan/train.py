"""Self-play data collection + a single training cycle.

The agent is a masked-policy network. We collect episodes by sampling actions
from the (softmax-of-masked-logits) policy, recording (state, mask, action,
reward, value) transitions. At the end of the cycle we run a small REINFORCE
update with a value baseline (Monte-Carlo returns = final score difference,
since Wingspan is episodic and only the terminal score matters).

This is deliberately minimal — the goal is criterion 3: *"complete a single
training cycle starting from random weights"*. The infrastructure scales
naturally to more sophisticated algorithms later.
"""
from __future__ import annotations

import argparse
import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .actions import Choice, Decision
from .agents import random_agent
from .cards import load_all
from .encode import encode_decision, encode_state
from .game import Engine, make_engine
from .model import PolicyValueNet
from .state import new_game

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Trajectory recorder

@dataclass
class Step:
    state: np.ndarray
    mask: np.ndarray
    action_slot: int          # which slot was chosen
    choice_index: int         # index into the legal-choice list (== action_slot - offset)
    player_id: int


@dataclass
class Trajectory:
    steps: list[Step]
    scores: tuple[int, int]   # final scores
    winner: int               # 0 or 1; -1 for tie


# ---------------------------------------------------------------------------
# Policy agent that records its decisions

def make_policy_agent(net: PolicyValueNet, device: torch.device, rng: random.Random,
                      record_into: list[Step], me_id: int, epsilon: float = 0.05) -> Callable:
    def agent(engine: Engine, decision: Decision) -> Choice:
        s = encode_state(engine.state)
        mask, slots = encode_decision(decision)
        if not slots:
            return decision.choices[0]
        # Forward pass
        with torch.no_grad():
            st = torch.from_numpy(s).unsqueeze(0).to(device)
            mk = torch.from_numpy(mask).unsqueeze(0).to(device)
            logits, _ = net(st, mk)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        # Epsilon-greedy exploration on top of policy
        if rng.random() < epsilon:
            idx = rng.randrange(len(slots))
        else:
            slot_probs = np.array([probs[s] for s in slots])
            tot = slot_probs.sum()
            if tot <= 0 or not np.isfinite(tot):
                idx = rng.randrange(len(slots))
            else:
                slot_probs = slot_probs / tot
                idx = int(np.random.choice(len(slots), p=slot_probs))
        chosen_slot = slots[idx]
        if decision.player_id == me_id:
            record_into.append(Step(
                state=s, mask=mask,
                action_slot=chosen_slot, choice_index=idx,
                player_id=decision.player_id,
            ))
        return decision.choices[idx]
    return agent


# ---------------------------------------------------------------------------
# Data collection

def collect_episode(net: PolicyValueNet, device: torch.device, rng: random.Random,
                    epsilon: float, seed: int) -> Trajectory:
    eng, _, _, _ = make_engine(seed=seed)
    recorded: list[Step] = []
    # Learning player is player 0; opponent is random (curriculum: introduce
    # self-play after the first cycle works end-to-end).
    a = make_policy_agent(net, device, rng, recorded, me_id=0, epsilon=epsilon)
    b = random_agent(rng)
    eng.play_one_game((a, b))
    s0 = getattr(eng.state.players[0], "final_score", 0)
    s1 = getattr(eng.state.players[1], "final_score", 0)
    winner = 0 if s0 > s1 else (1 if s1 > s0 else -1)
    return Trajectory(steps=recorded, scores=(s0, s1), winner=winner)


# ---------------------------------------------------------------------------
# Training step

def train_step(net: PolicyValueNet, optimizer: Adam, trajectories: list[Trajectory], device: torch.device) -> dict:
    states, masks, slots, returns = [], [], [], []
    for tr in trajectories:
        # Return = scaled final-score advantage of the learning player.
        # Normalize so values stay O(1).
        adv = (tr.scores[0] - tr.scores[1]) / 50.0
        for st in tr.steps:
            states.append(st.state)
            masks.append(st.mask)
            slots.append(st.action_slot)
            returns.append(adv)
    if not states:
        return {"loss": 0.0, "n_steps": 0}
    state_t = torch.from_numpy(np.stack(states)).to(device)
    mask_t = torch.from_numpy(np.stack(masks)).to(device)
    slot_t = torch.tensor(slots, dtype=torch.long, device=device)
    ret_t = torch.tensor(returns, dtype=torch.float32, device=device)

    logits, value = net(state_t, mask_t)
    logp = F.log_softmax(logits, dim=-1)
    chosen = logp.gather(1, slot_t.unsqueeze(1)).squeeze(1)
    advantages = ret_t - value.detach()
    policy_loss = -(chosen * advantages).mean()
    value_loss = F.mse_loss(value, ret_t)
    # Entropy: only sum over legal slots so masked -inf logits don't poison the gradient.
    legal_logp = torch.where(mask_t > 0.5, logp, torch.zeros_like(logp))
    legal_p = torch.where(mask_t > 0.5, logp.exp(), torch.zeros_like(logp))
    entropy = -(legal_p * legal_logp).sum(dim=-1).mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "n_steps": len(states),
    }


# ---------------------------------------------------------------------------
# CLI

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Wingspan training cycle.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--episodes", type=int, default=32, help="Episodes per epoch.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epsilon", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default="checkpoints/wingspan_cycle0.pt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    device = torch.device(args.device)
    logger.info("Device: %s", device)
    rng = random.Random(args.seed)

    net = PolicyValueNet().to(device)
    optimizer = Adam(net.parameters(), lr=args.lr)
    logger.info("Model: %s parameters", sum(p.numel() for p in net.parameters()))

    for epoch in range(args.epochs):
        trajs: list[Trajectory] = []
        wins = 0
        score_diffs = []
        for ep in range(args.episodes):
            tr = collect_episode(net, device, rng, args.epsilon, seed=args.seed * 10000 + epoch * 1000 + ep)
            trajs.append(tr)
            if tr.winner == 0: wins += 1
            score_diffs.append(tr.scores[0] - tr.scores[1])
        stats = train_step(net, optimizer, trajs, device)
        logger.info(
            "Epoch %d: wins=%d/%d, avg_score_diff=%+.2f, loss=%.4f (policy=%.4f value=%.4f entropy=%.4f), n_steps=%d",
            epoch, wins, args.episodes, sum(score_diffs)/len(score_diffs),
            stats["loss"], stats["policy_loss"], stats["value_loss"], stats["entropy"], stats["n_steps"],
        )

    ckpt = Path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": net.state_dict(), "args": vars(args)}, ckpt)
    logger.info("Saved checkpoint to %s", ckpt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
