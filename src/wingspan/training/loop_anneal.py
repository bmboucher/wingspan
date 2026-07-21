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

from wingspan.training import config

if typing.TYPE_CHECKING:
    from wingspan import model
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
    so they stay byte-identical to today's behaviour.
    """
    cfg = training_loop.config
    if cfg.training.dropout_final is not None:
        _anneal_main_net_dropout(training_loop.net, cfg, iteration)
    setup_net = training_loop._setup_net
    if cfg.training.setup.dropout_final is not None and setup_net is not None:
        # The setup net has a single global dropout knob (no per-block
        # overrides), so one uniform sweep is exact.
        set_dropout(setup_net, cfg.setup_dropout_p_at(iteration))


def _anneal_main_net_dropout(
    net: "model.PolicyValueNet", cfg: config.RunConfig, iteration: int
) -> None:
    """Sweep each of the main net's dropout-bearing blocks independently.

    A per-block override (``architecture.main.card_dropout`` etc.) can start
    from a different build-time value than the global ``architecture.main.dropout``,
    so each block anneals from its *own* resolved initial toward the shared
    ``training.dropout_final`` target instead of being snapped to one
    globally-computed value. A block whose resolved initial is 0.0 never had
    an ``nn.Dropout`` module built for it (``mlp.build_body`` / ``build_readout``
    only construct one when ``p > 0``), so ``set_dropout`` silently finds
    nothing there — the anneal is inert for that block rather than needing to
    be forbidden at launch.
    """
    arch = cfg.arch
    final = cfg.training.dropout_final
    target_iterations = cfg.run.target_iterations

    set_dropout(
        net.card_encoder,
        config.anneal(arch.card_dropout_resolved, final, target_iterations, iteration),
    )
    if arch.use_distinct_hand_model:
        set_dropout(
            net.hand_encoder,
            config.anneal(
                arch.hand_dropout_resolved, final, target_iterations, iteration
            ),
        )
    set_dropout(
        net.state_trunk,
        config.anneal(arch.trunk_dropout_resolved, final, target_iterations, iteration),
    )
    set_dropout(
        net.choice_encoder,
        config.anneal(
            arch.choice_dropout_resolved, final, target_iterations, iteration
        ),
    )
    # Scorers and the value head have no per-block override — both always
    # build at the global dropout.
    global_p = config.anneal(arch.dropout, final, target_iterations, iteration)
    set_dropout(net.scorers, global_p)
    set_dropout(net.value_head, global_p)
