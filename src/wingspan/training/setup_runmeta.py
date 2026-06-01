"""The setup model's JSON descriptor sidecar (``setup_config.json``).

The setup-model analogue of ``runmeta.write_model_config``: it records the setup
net's weight-compatibility shape (the encoder feature width + the MLP topology)
in a human-readable file beside ``setup.pt``, so a run's setup network reads at a
glance and reconstitutes via ``setup_net.SetupNet.from_setup_config``. Kept
torch-free (only ``pydantic``) so it can be written / read without a training run,
mirroring the ``runmeta`` split.
"""

from __future__ import annotations

import pathlib

import pydantic

from wingspan import setup_model
from wingspan.training import artifacts, config


class SetupConfig(pydantic.BaseModel):
    """The setup network descriptor written to ``setup_config.json``.

    Carries the encoder feature width the net was trained against plus its
    complete :class:`wingspan.setup_model.SetupArchitecture`, so the file both
    summarizes the setup network and fully reconstitutes it. The setup
    weight-compatibility signature ``TrainConfig.setup_architecture_key`` is
    derived from exactly these fields."""

    run_name: str
    setup_feature_dim: int
    setup_arch: setup_model.SetupArchitecture


def write_setup_config(checkpoint_dir: str, cfg: config.TrainConfig) -> pathlib.Path:
    """Write (overwriting) ``setup_config.json`` for ``cfg`` and return its path."""
    descriptor = SetupConfig(
        run_name=cfg.run_name,
        setup_feature_dim=setup_model.SETUP_FEATURE_DIM,
        setup_arch=cfg.setup_arch,
    )
    path = pathlib.Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    descriptor_path = path / artifacts.SETUP_CONFIG_JSON
    descriptor_path.write_text(descriptor.model_dump_json(indent=2), encoding="utf-8")
    return descriptor_path


def read_setup_config(checkpoint_dir: str) -> SetupConfig:
    """Read the ``setup_config.json`` descriptor from ``checkpoint_dir`` (pairs
    with :func:`write_setup_config`). Raises ``FileNotFoundError`` if absent."""
    path = pathlib.Path(checkpoint_dir) / artifacts.SETUP_CONFIG_JSON
    return SetupConfig.model_validate_json(path.read_text(encoding="utf-8"))
