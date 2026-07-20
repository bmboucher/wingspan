# pyright: reportPrivateUsage=false
# (accesses TrainingLoop's private fields — deliberate intra-package coupling)
"""Dropout-anneal sweep for ``TrainingLoop``.

Entropy-coefficient annealing threads through the learners as a plain float
(``RunConfig.entropy_coef_at`` / ``setup_entropy_coef_at``, consumed in
``learner.update`` / ``setup_learner.actor_critic_update``); dropout has no
such call-site hook because ``nn.Dropout.p`` is read fresh on every forward
pass, so annealing it only requires mutating the module in place once per
iteration. :func:`apply_dropout_schedules` is that mutation, called once at
the top of each training iteration.
"""

from __future__ import annotations

import typing

from torch import nn

if typing.TYPE_CHECKING:
    from wingspan.training import loop


def set_dropout(net: nn.Module, p: float) -> int:
    """Set ``.p`` on every ``nn.Dropout`` submodule of ``net``; return the count changed."""
    count = 0
    for module in net.modules():
        if isinstance(module, nn.Dropout):
            module.p = p
            count += 1
    return count


def apply_dropout_schedules(training_loop: "loop.TrainingLoop", iteration: int) -> None:
    """Sweep the main and (if active) setup net's dropout to this iteration's value.

    No-op for a net whose ``training.dropout_final`` / ``training.setup.dropout_final``
    is unset — non-annealing runs never mutate ``nn.Dropout.p`` after construction,
    so they stay byte-identical to today's behaviour. Uniform sweep is exact for
    the main net because ``validate_launchable`` forbids per-block dropout
    overrides whenever the anneal is active; for the setup net it anneals every
    Dropout module belonging to the setup-owned blocks (the frozen card/hand
    embedder copies stay pinned to ``eval()`` by ``SetupNet.train()`` and carry
    no Dropout of their own on that path).
    """
    cfg = training_loop.config
    if cfg.training.dropout_final is not None:
        set_dropout(training_loop.net, cfg.dropout_p_at(iteration))
    setup_net = training_loop._setup_net
    if cfg.training.setup.dropout_final is not None and setup_net is not None:
        set_dropout(setup_net, cfg.setup_dropout_p_at(iteration))
