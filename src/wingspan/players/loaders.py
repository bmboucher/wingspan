"""Checkpoint loaders shared by the ``play`` and ``tournament`` CLIs.

Two sanctioned, self-describing load paths exist and both live here:

- :func:`load_policy_net` rebuilds a net from the ``TrainConfig`` embedded in
  the ``.pt`` payload itself (the ``play`` CLI path; the returned config also
  carries the regime flags the caller mirrors, e.g. ``split_setup_bonus``).
- :func:`load_policy_net_from_run_dir` rebuilds a net from a run directory's
  ``model_config.json`` descriptor (the tournament path; the descriptor is
  also what run discovery checks without unpickling torch payloads).

Both enforce the hard artifact-version check and construct the era's net class
via ``model.PolicyValueNet.class_for_version`` at the era's frozen dims
(``config.train_config_from_artifact`` / the descriptor), so a pre-0.3
checkpoint's encoders and slice geometry match its weights automatically.
:func:`load_setup_net` loads the optional separately-trained setup model from a
run directory. The encoding-key helpers (:func:`encoding_key`,
:func:`descriptor_encoding_key`, :func:`expected_encoding_key`) define the
compatibility signature both paths verify before seating a net.
"""

from __future__ import annotations

import pathlib
import typing

import torch

from wingspan import compat, decisions, encode, model, version
from wingspan.training import artifacts, config, runmeta
from wingspan.training import setup_net as setup_net_module
from wingspan.training import setup_runmeta


def load_policy_net(
    checkpoint_path: pathlib.Path, device: torch.device
) -> tuple[model.PolicyValueNet, config.RunConfig]:
    """Load a ``PolicyValueNet`` from a training checkpoint, rebuilding it from
    the ``TrainConfig`` stored alongside the weights so the caller need not know
    the network's layer widths; the parsed config is returned with the net so
    regime flags (e.g. ``split_setup_bonus``) can mirror the training run. Raises
    with a clear message when the file is missing, lacks a config, or was trained
    against an incompatible encoding layout."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train a model with "
            "`wingspan dashboard` first, or pass a direct .pt path."
        )

    # Our own trusted checkpoint carries a config dict + metrics, not just
    # tensors, so the full (non weights-only) unpickler is required.
    payload = typing.cast(
        "dict[str, typing.Any]",
        torch.load(checkpoint_path, map_location=device, weights_only=False),
    )
    if "config" not in payload:
        raise ValueError(
            f"Checkpoint at {checkpoint_path} has no embedded 'config' — it is "
            "not a valid self-describing training checkpoint."
        )
    # Payloads that predate the version stamp read as the pre-versioning era.
    artifact_version = str(payload.get("version", version.PRE_VERSIONING_VERSION))
    version.check_artifact_compatible(
        artifact_version, what=f"checkpoint at {checkpoint_path}"
    )

    # The net is rebuilt from the checkpoint's own topology, so its layer widths
    # always match its weights; what must match the *current* code is the
    # encoding layout (state/choice feature dims and the family head order),
    # since freshly-encoded states are fed into the net at inference. A net
    # trained with a different topology is still perfectly usable here.
    # The config is rehydrated at the payload's own era, so its derived dims are
    # the frozen widths the weights actually carry, and the era routing
    # (``class_for_version``) selects the compat subclass whose encoders and
    # slice geometry match them.
    saved = config.run_config_from_artifact(payload["config"], artifact_version)
    net_cls = model.PolicyValueNet.class_for_version(artifact_version)
    if net_cls is model.PolicyValueNet:
        current = config.RunConfig()
        if encoding_key(saved) != encoding_key(current):
            raise ValueError(
                "Checkpoint encoding layout is incompatible with the current code:\n"
                f"  saved:   {encoding_key(saved)}\n"
                f"  current: {encoding_key(current)}\n"
                "It was trained against a different encode.py / decisions.py layout."
            )
    net = net_cls(
        state_dim=saved.state_dim,
        choice_dim=saved.choice_dim,
        num_families=len(saved.family_order),
        arch=saved.arch,
        spec=saved.encoding_spec,
    ).to(device)
    net.load_state_dict(payload["model"])
    net.eval()
    return net, saved


def load_policy_net_from_run_dir(
    checkpoint_dir: str, device: torch.device
) -> model.PolicyValueNet:
    """Load a run directory's ``last.pt`` weights into the net shape recorded by
    its ``model_config.json`` descriptor (every run dir carries one; a run
    without it is not seatable). Raises ``ValueError`` when the saved encoding
    no longer matches the live encoder — freshly-encoded states could not feed
    the net — so an explicitly-named stale run fails at seating with a clear
    message instead of a mid-game tensor-shape error."""
    payload = typing.cast(
        "dict[str, typing.Any]",
        torch.load(
            pathlib.Path(checkpoint_dir) / artifacts.LAST_CKPT,
            map_location=device,
            weights_only=False,
        ),
    )
    # Payloads that predate the version stamp read as the pre-versioning era.
    version.check_artifact_compatible(
        str(payload.get("version", version.PRE_VERSIONING_VERSION)),
        what=f"{artifacts.LAST_CKPT} at {checkpoint_dir}",
    )
    net = _reconstruct_net(checkpoint_dir)
    state_dict = typing.cast("dict[str, torch.Tensor]", payload["model"])
    net.load_state_dict(state_dict)
    net.to(device).eval()
    return net


def load_setup_net(
    checkpoint_dir: pathlib.Path, device: torch.device
) -> setup_net_module.SetupNet | None:
    """Load the separately-trained ``SetupNet`` from ``checkpoint_dir``.

    Returns ``None`` — degrading to random setup picks — only when the setup
    artifacts are absent (the run trained without a setup model). Artifacts that
    exist but fail to load raise: a present-but-broken ``setup.pt`` is an error,
    not something to silently paper over.

    The descriptor is resolvable from either the legacy ``setup_config.json``
    (≤0.4 runs) or the unified ``run_config_<stamp>.json`` (≥0.5 runs); both
    are accepted here so ``wingspan play`` works correctly with current runs."""
    ckpt_path = checkpoint_dir / artifacts.SETUP_CKPT
    has_config = bool(
        (checkpoint_dir / artifacts.SETUP_CONFIG_JSON).exists()
        or list(checkpoint_dir.glob(artifacts.RUN_CONFIG_GLOB))
    )
    if not ckpt_path.exists() or not has_config:
        return None
    descriptor = setup_runmeta.read_setup_config(str(checkpoint_dir))
    payload = typing.cast(
        "dict[str, typing.Any]",
        torch.load(ckpt_path, map_location=device, weights_only=False),
    )
    # Payloads that predate the version stamp read as the pre-versioning era.
    artifact_version = str(payload.get("version", version.PRE_VERSIONING_VERSION))
    version.check_artifact_compatible(
        artifact_version,
        what=f"setup checkpoint at {ckpt_path}",
    )
    # No pre-1.0 shims remain: every loadable artifact uses the live setup net.
    net_instance = setup_net_module.SetupNet.from_setup_config(descriptor)
    try:
        net_instance.load_state_dict(payload["setup_model"])
    except RuntimeError as error:
        # A shape mismatch means the weights predate a setup-net topology change
        # (e.g. the v1.2 state-only V(s) value head, whose Linear is narrower
        # than the old fused Q head). Setup checkpoints are not migrated — refuse
        # explicitly rather than surfacing an opaque torch size error.
        raise version.IncompatibleArtifactError(
            f"setup checkpoint at {ckpt_path} does not fit the live SetupNet "
            f"architecture (artifact version {artifact_version}, code version "
            f"{version.MODEL_VERSION}). Old setup models are discarded, not "
            f"migrated — retrain the setup model."
        ) from error
    net_instance.eval()
    return net_instance.to(device)


def encoding_key(cfg: config.RunConfig) -> tuple[int, int, tuple[str, ...]]:
    """The encoding-compatibility signature: the parts of the architecture that
    must agree with the live ``encode`` / ``decisions`` modules for a checkpoint
    to consume freshly-encoded inputs (the layer widths are excluded — they are
    self-consistent with the loaded weights)."""
    return (cfg.state_dim, cfg.choice_dim, cfg.family_order)


def descriptor_encoding_key(
    descriptor: runmeta.ModelConfig,
) -> tuple[int, int, tuple[str, ...]]:
    """The encoding-compatibility signature a run was trained against."""
    return (descriptor.state_dim, descriptor.choice_dim, descriptor.family_order)


def expected_encoding_key(
    descriptor: runmeta.ModelConfig,
) -> tuple[int, int, tuple[str, ...]]:
    """The encoding-compatibility signature ``descriptor``'s artifact era
    promises: the era's encoder dims (``compat.encoding_dims_for_era`` — the
    same router ``config.TrainConfig._sync_encoding_dims`` derives from), built
    over the descriptor's own spec. The family-head order is unchanged between
    eras, so it is live-derived."""
    spec = encode.EncodingSpec(include_setup=descriptor.include_setup)
    family_order = tuple(
        family.value
        for family in decisions.active_decision_families(spec.include_setup)
    )
    state_dim, choice_dim = compat.encoding_dims_for_era(descriptor.version, spec)
    return (state_dim, choice_dim, family_order)


###### PRIVATE #######


def _reconstruct_net(checkpoint_dir: str) -> model.PolicyValueNet:
    """Rebuild the (fresh-weight) net in a run's saved shape, ready for
    ``load_state_dict``, from the run's ``model_config.json`` descriptor (every
    run dir carries one; a run without it is not seatable). Raises ``ValueError``
    when the saved encoding no longer matches the live encoder — freshly-encoded
    states could not feed the net — so an explicitly-named stale run fails at
    seating with a clear message instead of a mid-game tensor-shape error."""
    descriptor = runmeta.read_model_config(checkpoint_dir)
    saved = descriptor_encoding_key(descriptor)
    expected = expected_encoding_key(descriptor)
    if saved != expected:
        raise ValueError(
            f"Run {checkpoint_dir!r} was trained against an encoding layout "
            "incompatible with the current code:\n"
            f"  saved:    {saved}\n"
            f"  expected: {expected}\n"
            "It cannot consume freshly-encoded states and is not seatable."
        )
    return model.PolicyValueNet.from_model_config(descriptor)
