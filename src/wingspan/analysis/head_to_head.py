"""Paired-game head-to-head evaluation of a full network against a copy of
itself with its board attention substituted.

Where :mod:`representation`'s ablation effects measure *dependence* on the
policy/value output (KL, greedy-flip, value shift over a fixed probe set),
:func:`evaluate_substitution` measures the same substitution's effect on
*actual game strength*: it plays the two networks against each other with
``training.evaluate``'s paired-deal harness, the same measurement the
training loop uses to judge opponent strength.
"""

from __future__ import annotations

import pathlib

import torch

from wingspan.analysis import attention_probe, models
from wingspan.players import loaders
from wingspan.training import evaluate


def evaluate_substitution(
    checkpoint_path: pathlib.Path,
    mode: models.AblationMode,
    n_pairs: int,
    seed: int,
    device: torch.device,
) -> models.HeadToHeadResult:
    """Load ``checkpoint_path`` twice, install board-attention wrappers on the
    second copy in ``mode`` (``ZERO`` / ``UNIFORM`` are the meaningful
    choices; ``FULL`` degenerates to a self-play sanity check), and play
    ``n_pairs`` mirrored deals of the untouched net against the substituted
    one. Returns the untouched net's win rate as a :class:`models.HeadToHeadResult`."""
    full_net, _ = loaders.load_policy_net(checkpoint_path, device)
    substituted_net, run_config = loaders.load_policy_net(checkpoint_path, device)
    wrappers = attention_probe.install(substituted_net)
    attention_probe.set_mode(wrappers, mode)
    result = evaluate.evaluate_vs_opponent(
        full_net,
        substituted_net,
        device,
        n_pairs,
        seed=seed,
        num_players=run_config.num_players,
        split_setup_bonus=run_config.split_setup_bonus_active,
        split_setup_food=run_config.split_setup_food_active,
        combine_gain_food=run_config.engine.combine_gain_food,
    )
    return models.HeadToHeadResult(
        mode=mode,
        n_games=result.n_games,
        win_rate=result.win_rate,
        ci95=result.ci95,
        mean_margin=result.mean_margin,
    )
