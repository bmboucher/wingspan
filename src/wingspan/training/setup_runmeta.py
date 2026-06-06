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

from wingspan import architecture, setup_model, version
from wingspan.training import artifacts, config


class SetupConfig(pydantic.BaseModel):
    """The setup network descriptor written to ``setup_config.json``.

    Carries the encoder feature width the net was trained against, its complete
    :class:`wingspan.setup_model.SetupArchitecture`, and the main
    :class:`wingspan.architecture.ModelArchitecture` whose embedder copies it
    reads — so the file both summarizes the setup network and fully
    reconstitutes it (``SetupNet.from_setup_config``). ``main_arch`` defaults to
    a bare descriptor so JSON files that predate the shared embedders still
    deserialize. The setup weight-compatibility signature
    ``TrainConfig.setup_architecture_key`` is derived from exactly these
    fields."""

    run_name: str
    setup_feature_dim: int
    setup_arch: setup_model.SetupArchitecture
    main_arch: architecture.ModelArchitecture = architecture.ModelArchitecture()
    # The artifact-compatibility version the run was written at; defaults so
    # files that predate the field read as the pre-versioning era ("0.0").
    version: str = version.PRE_VERSIONING_VERSION


def write_setup_config(checkpoint_dir: str, cfg: config.TrainConfig) -> pathlib.Path:
    """Write (overwriting) ``setup_config.json`` for ``cfg`` and return its path."""
    descriptor = SetupConfig(
        run_name=cfg.run_name,
        setup_feature_dim=setup_model.SETUP_FEATURE_DIM,
        setup_arch=cfg.setup_arch,
        main_arch=cfg.arch,
        version=version.MODEL_VERSION,
    )
    path = pathlib.Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    descriptor_path = path / artifacts.SETUP_CONFIG_JSON
    descriptor_path.write_text(descriptor.model_dump_json(indent=2), encoding="utf-8")
    return descriptor_path


def read_setup_config(checkpoint_dir: str) -> SetupConfig:
    """Read the ``setup_config.json`` descriptor from ``checkpoint_dir`` (pairs
    with :func:`write_setup_config`). Raises ``FileNotFoundError`` if absent and
    ``version.IncompatibleArtifactError`` when the descriptor's artifact version
    is outside the current code's load guarantee."""
    path = pathlib.Path(checkpoint_dir) / artifacts.SETUP_CONFIG_JSON
    descriptor = SetupConfig.model_validate_json(path.read_text(encoding="utf-8"))
    version.check_artifact_compatible(
        descriptor.version, what=f"{artifacts.SETUP_CONFIG_JSON} at {checkpoint_dir}"
    )
    return descriptor
