"""Batched decision tensors for offline architecture probing.

``ProbeSet`` is the fixed sample of on-distribution decisions
``representation.measure`` runs every ablation mode over: state / choice /
mask / family tensors grouped into :class:`ProbeBatch`\\ es by exact legal-
option count (no padding — every batch's rows are all-real, all-legal
choices, so a batch is built once and reused unmodified across every mode
instead of being backpropagated through, unlike ``training.learner``'s
length-bucketed training batches).
"""

from __future__ import annotations

import collections
import random
import typing

import numpy as np
import pydantic
import torch

from wingspan import encode, model

# ``steps`` is aliased because ``from_steps``'s ``steps`` parameter (and the
# local accumulator lists below) would shadow the bare module name — the same
# rule ``training.collect`` documents for its own ``training_steps`` alias.
from wingspan.training import collect, config
from wingspan.training import steps as training_steps


class ProbeBatch(pydantic.BaseModel):
    """One exact-choice-count group of decisions, batched for one forward pass.

    ``state`` is ``(B, state_dim)``; ``choices`` is ``(B, K, choice_dim)``;
    ``mask`` is ``(B, K)`` (all ``1.0`` — batches are grouped by exact ``K``,
    so there is no padding to mask out); ``family_idx`` is ``(B,)`` long; and
    ``own_bird_count`` is ``(B,)`` long — the number of birds on the deciding
    player's own board at that decision, used to slice the board-fill
    ablation effect.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    state: torch.Tensor
    choices: torch.Tensor
    mask: torch.Tensor
    family_idx: torch.Tensor
    own_bird_count: torch.Tensor


class ProbeSet(pydantic.BaseModel):
    """A fixed sample of decisions to measure every ablation mode over — one
    forward pass per :class:`ProbeBatch`, repeated per mode."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    batches: list[ProbeBatch]
    n_decisions: int
    n_games: int


def from_steps(
    net: model.PolicyValueNet,
    steps: typing.Sequence[training_steps.Step],
    n_games: int,
) -> ProbeSet:
    """Build a :class:`ProbeSet` from already-collected decision steps,
    grouping them by exact legal-option count (no padding). ``n_games`` is
    carried through unchanged (the caller's own game count — it cannot be
    recovered from ``steps`` alone)."""
    groups: dict[int, list[training_steps.Step]] = collections.defaultdict(list)
    for step in steps:
        groups[step.choices.shape[0]].append(step)

    own_board_offset = net.raw_state_stripe_layout().offset_of("card_idx_board")
    batches = [
        _batch_from_group(group, own_board_offset)
        for _, group in sorted(groups.items())
    ]
    return ProbeSet(batches=batches, n_decisions=len(steps), n_games=n_games)


def subsample_steps(
    records: typing.Sequence[collect.GameRecord],
    max_decisions: int,
    rng: random.Random,
) -> list[training_steps.Step]:
    """Flatten every record's steps and, if there are more than
    ``max_decisions``, take a reproducible random subsample of exactly that
    size (every step is returned untouched when already at or under the cap)."""
    all_steps = [step for record in records for step in record.steps]
    if len(all_steps) <= max_decisions:
        return all_steps
    return rng.sample(all_steps, max_decisions)


def from_self_play(
    net: model.PolicyValueNet,
    run_config: config.RunConfig,
    n_games: int,
    seed: int,
    device: torch.device,
) -> ProbeSet:
    """Play ``n_games`` of self-play with ``net`` (mirroring the run's own
    collection regime — seat count and the ``combine_gain_food`` toggle) and
    build a :class:`ProbeSet` from every recorded decision."""
    rng = random.Random(seed)
    records = [
        collect.play_game(
            net,
            device,
            rng,
            seed=seed + game_index,
            combine_gain_food=run_config.engine.combine_gain_food,
            num_players=run_config.num_players,
        )
        for game_index in range(n_games)
    ]
    all_steps = [step for record in records for step in record.steps]
    return from_steps(net, all_steps, n_games)


###### PRIVATE #######


def _batch_from_group(
    group: typing.Sequence[training_steps.Step], own_board_offset: int
) -> ProbeBatch:
    """Stack one exact-choice-count group of steps into a :class:`ProbeBatch`."""
    state = torch.tensor(np.stack([step.state for step in group]), dtype=torch.float32)
    choices = torch.tensor(
        np.stack([step.choices for step in group]), dtype=torch.float32
    )
    mask = torch.ones(choices.shape[:2], dtype=torch.float32)
    family_idx = torch.tensor([step.family_idx for step in group], dtype=torch.long)
    own_board = state[:, own_board_offset : own_board_offset + encode.SLOTS_PER_BOARD]
    own_bird_count = (own_board != 0).sum(dim=1)
    return ProbeBatch(
        state=state,
        choices=choices,
        mask=mask,
        family_idx=family_idx,
        own_bird_count=own_bird_count,
    )
