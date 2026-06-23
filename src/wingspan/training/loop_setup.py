# pyright: reportPrivateUsage=false
# (accesses TrainingLoop's private fields — deliberate intra-package coupling)
"""Setup-model helpers for ``TrainingLoop``.

Free functions whose first argument is a ``TrainingLoop`` instance manage the
separate setup-model lifecycle: actor-critic update, net construction, embedder
sync, resume, architecture match, and checkpoint save.
"""

from __future__ import annotations

import typing

import pydantic
import torch
from torch import optim

from wingspan import setup_model, version
from wingspan.training import (
    artifacts,
    collect,
)
from wingspan.training import config as training_config
from wingspan.training import (
    loop_checkpoint,
    metrics,
    runstate,
    setup_learner,
    setup_net,
)

if typing.TYPE_CHECKING:
    from wingspan.training import loop


def update_setup(
    training_loop: "loop.TrainingLoop",
    records: list[collect.GameRecord],
) -> metrics.SetupUpdateStats:
    """Run one actor-critic pass over this iteration's setup samples.

    Returns a :class:`~metrics.SetupUpdateStats` with the loss and margin stats
    for the dashboard. Returns an empty stats object when there are no samples.
    """
    assert training_loop._setup_net is not None
    assert training_loop._setup_optimizer is not None
    samples = [sample for record in records for sample in record.setup_samples]
    stats = setup_learner.actor_critic_update(
        training_loop._setup_net,
        training_loop._setup_optimizer,
        samples,
        training_loop.config,
        training_loop.device,
    )
    with training_loop.lock:
        training_loop.state.push_event(
            runstate.EventKind.INFO,
            f"SETUP AC {stats.loss:.4f} · pred "
            f"{stats.pred_margin_mean:+.1f} vs real "
            f"{stats.realized_margin_mean:+.1f} ({stats.n_samples} samples)",
        )
        training_loop.state.last_setup = stats
        training_loop.state.record_setup_trained(stats.n_samples)
    return stats


def build_setup_net(
    training_loop: "loop.TrainingLoop",
) -> tuple[setup_net.SetupNet, optim.Optimizer]:
    """A fresh setup net and its optimizer over the *trainable* parameters only.

    The frozen embedder copies are shaped by the main architecture; they are
    synced from the main net and never stepped by this optimizer.
    """
    net = setup_net.SetupNet(
        encoding=training_loop.config.setup_encoding,
        arch=training_loop.config.setup_arch,
        main_arch=training_loop.config.arch,
    ).to(training_loop.device)
    optimizer = optim.Adam(
        [param for param in net.parameters() if param.requires_grad],
        lr=training_loop.config.training.setup.lr,
    )
    return net, optimizer


def sync_setup_embedders(training_loop: "loop.TrainingLoop") -> None:
    """Copy the main net's shared embedder weights into the setup net's frozen copies.

    Syncs the card encoder always, and the hand encoder when the main
    architecture has one; then drops the setup net to ``eval()`` (the
    cache-invalidation contract, so its memoized card table is rebuilt from
    the freshly synced weights).  Called once after resume and once per
    iteration right after the main update.  No-op when the setup model is off.
    """
    if training_loop._setup_net is None:
        return
    training_loop._setup_net.card_encoder.load_state_dict(
        training_loop.net.card_encoder.state_dict()
    )
    if training_loop.config.architecture.main.use_distinct_hand_model:
        training_loop._setup_net.hand_encoder.load_state_dict(
            training_loop.net.hand_encoder.state_dict()
        )
    training_loop._setup_net.eval()


def maybe_resume_setup(training_loop: "loop.TrainingLoop") -> None:
    """Restore the setup net, its optimizer, and the offline-fit-done flag from
    ``setup.pt`` so a resumed run continues the setup model where it left off.

    No-ops when the feature is off, resuming is disabled, or there is no setup
    checkpoint; a mismatched / unreadable / unloadable one starts the setup
    net fresh with an alarm — and clears the recorded sample store, whose rows
    belong to the incompatible layout.
    """
    if training_loop._setup_net is None or training_loop._setup_optimizer is None:
        return
    if not training_loop.config.run.resume:
        return
    path = training_loop._ckpt_dir / artifacts.SETUP_CKPT
    if not path.exists():
        return
    try:
        payload = typing.cast(
            "dict[str, typing.Any]",
            torch.load(path, map_location=training_loop.device, weights_only=False),
        )
    except Exception:  # noqa: BLE001 — a corrupt setup checkpoint starts fresh
        training_loop.state.push_event(
            runstate.EventKind.ALARM,
            f"could not read {artifacts.SETUP_CKPT} — setup net starting fresh",
        )
        return
    if not setup_architecture_matches(training_loop, payload):
        reset_setup_store(training_loop)
        training_loop.state.push_event(
            runstate.EventKind.ALARM,
            f"{artifacts.SETUP_CKPT} architecture differs — setup net fresh "
            "(recorded setup samples cleared)",
        )
        return
    # Belt-and-suspenders: any load failure the key comparison did not
    # foresee rebuilds the setup net fresh rather than crashing the run.
    try:
        training_loop._setup_net.load_state_dict(payload["setup_model"])
        training_loop._setup_optimizer.load_state_dict(payload["setup_optimizer"])
    except Exception:  # noqa: BLE001 — incompatible weights start fresh
        training_loop._setup_net, training_loop._setup_optimizer = build_setup_net(
            training_loop
        )
        reset_setup_store(training_loop)
        training_loop.state.push_event(
            runstate.EventKind.ALARM,
            f"{artifacts.SETUP_CKPT} weights incompatible — setup net fresh "
            "(recorded setup samples cleared)",
        )
        return
    for group in training_loop._setup_optimizer.param_groups:
        group["lr"] = training_loop.config.training.setup.lr
    training_loop.state.push_event(
        runstate.EventKind.INFO,
        f"resumed {artifacts.SETUP_CKPT}",
    )


def setup_architecture_matches(
    training_loop: "loop.TrainingLoop", payload: dict[str, typing.Any]
) -> bool:
    """Whether a ``setup.pt`` payload's setup-net shape matches this run's.

    The persisted ``setup_encoding`` is the encoding-layout discriminator:
    the config-derived key alone cannot see a layout change (both sides
    recompute it from *current* code), so the payload records the encoding it
    actually trained against.  Pre-0.2 payloads carry ``setup_feature_dim``
    instead; those match only when this run's encoding is also the legacy
    308-dim layout.
    """
    current_encoding = training_loop.config.setup_encoding
    raw_encoding = payload.get("setup_encoding")
    if raw_encoding is None:
        # Legacy checkpoint without setup_encoding — use the old feature_dim discriminator.
        saved_dim = payload.get("setup_feature_dim")
        if saved_dim != current_encoding.total_dim:
            return False
    else:
        try:
            saved_encoding = setup_model.SetupEncoding.model_validate(raw_encoding)
        except pydantic.ValidationError:
            return False
        if saved_encoding != current_encoding:
            return False
    raw_config = payload.get("setup_config")
    if raw_config is None:
        return False  # not a self-describing setup checkpoint — refuse
    artifact_version = str(payload.get("version", version.PRE_VERSIONING_VERSION))
    try:
        # Rehydrated at the payload's own era for uniformity with the main
        # gate; the setup key itself is era-independent (no setup encoding has
        # changed shape between eras the dims router distinguishes).
        saved = training_config.run_config_from_artifact(raw_config, artifact_version)
    except pydantic.ValidationError:
        return False
    return saved.setup_architecture_key == training_loop.config.setup_architecture_key


def reset_setup_store(_training_loop: "loop.TrainingLoop") -> None:
    """No-op: the offline JSONL store was removed; kept as a call-site stub."""


def save_setup_checkpoint(training_loop: "loop.TrainingLoop") -> None:
    """Persist the setup net + optimizer + offline-fit flag to ``setup.pt``.

    Stamps with the encoder feature width the weights were trained against
    (the resume gate's layout discriminator).
    """
    if training_loop._setup_net is None or training_loop._setup_optimizer is None:
        return
    payload: dict[str, object] = {
        "setup_config": training_loop.config.model_dump(),
        "setup_encoding": training_loop.config.setup_encoding.model_dump(),
        "setup_model": training_loop._setup_net.state_dict(),
        "setup_optimizer": training_loop._setup_optimizer.state_dict(),
        "git_sha": loop_checkpoint.git_sha(),
        # The run's era, matching every other artifact the run writes.
        "version": training_loop.config.encoding_version,
    }
    loop_checkpoint.atomic_save(payload, training_loop._ckpt_dir / artifacts.SETUP_CKPT)
