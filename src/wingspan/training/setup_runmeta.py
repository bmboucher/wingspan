"""The setup model's JSON descriptor sidecar.

For **≥0.5 run directories**, the setup-net descriptor is derived from the
unified ``run_config_<stamp>.json`` file written by :func:`runmeta.write_run_config`
rather than from a standalone ``setup_config.json``.

For **≤0.4 run directories**, the legacy ``setup_config.json`` is read directly
so compat tests and older run dirs continue to work.

:func:`read_setup_config` dispatches on presence: if a unified file is found,
it derives the :class:`SetupConfig` from it; otherwise it falls back to
``setup_config.json``. :func:`write_setup_config` is kept for tooling /
manual use but is no longer called by the training loop.
"""

from __future__ import annotations

import pathlib

import pydantic

from wingspan import architecture, setup_model, version
from wingspan.training import artifacts, config


class SetupConfig(pydantic.BaseModel):
    """The setup network descriptor written to ``setup_config.json`` (legacy).

    Carries the encoding layout the net was trained against, its complete
    :class:`wingspan.setup_model.SetupArchitecture`, and the main
    :class:`wingspan.architecture.ModelArchitecture` whose embedder copies it
    reads — so the file both summarizes the setup network and fully
    reconstitutes it (``SetupNet.from_setup_config``). ``main_arch`` defaults to
    a bare descriptor so JSON files that predate the shared embedders still
    deserialize. The setup weight-compatibility signature
    ``RunConfig.setup_architecture_key`` is derived from exactly these fields.

    ``setup_encoding`` defaults to ``SetupEncoding()`` (both splits off, 308
    dims) so pre-0.2 ``setup_config.json`` files that lack the field deserialize
    correctly — the old 308-dim layout is reproduced by construction."""

    run_name: str
    setup_encoding: setup_model.SetupEncoding = setup_model.SetupEncoding()
    setup_arch: setup_model.SetupArchitecture
    main_arch: architecture.ModelArchitecture = architecture.ModelArchitecture()
    # The artifact-compatibility version the run was written at; defaults so
    # files that predate the field read as the pre-versioning era ("0.0").
    version: str = version.PRE_VERSIONING_VERSION


def read_setup_config(checkpoint_dir: str) -> SetupConfig:
    """Read the setup-net descriptor from ``checkpoint_dir``.

    Dispatches by presence:

    * **≥0.5** run dirs: derive :class:`SetupConfig` from the newest
      ``run_config_<stamp>.json``.
    * **≤0.4** run dirs: read ``setup_config.json`` directly (unchanged legacy
      path, so compat tests pass).

    Raises ``FileNotFoundError`` if neither artifact is present, and
    ``version.IncompatibleArtifactError`` when the version is outside the
    load guarantee.
    """
    directory = pathlib.Path(checkpoint_dir)

    # ≥0.5 path: derive from unified file.
    unified_matches = sorted(directory.glob(artifacts.RUN_CONFIG_GLOB))
    if unified_matches:
        file = config.RunConfigFile.model_validate_json(
            unified_matches[-1].read_text(encoding="utf-8")
        )
        version.check_artifact_compatible(
            file.version,
            what=f"{unified_matches[-1].name} at {checkpoint_dir}",
        )
        return _setup_config_from_run_config(file.config)

    # ≤0.4 legacy path: read setup_config.json directly.
    path = directory / artifacts.SETUP_CONFIG_JSON
    descriptor = SetupConfig.model_validate_json(path.read_text(encoding="utf-8"))
    version.check_artifact_compatible(
        descriptor.version, what=f"{artifacts.SETUP_CONFIG_JSON} at {checkpoint_dir}"
    )
    return descriptor


def write_setup_config(checkpoint_dir: str, cfg: config.RunConfig) -> pathlib.Path:
    """Write (overwriting) ``setup_config.json`` for ``cfg`` and return its path.

    Not called by the training loop for ≥0.5 runs; kept for tooling / tests."""
    descriptor = SetupConfig(
        run_name=cfg.run.run_name,
        setup_encoding=cfg.setup_encoding,
        setup_arch=cfg.setup_arch,
        main_arch=cfg.arch,
        version=cfg.architecture.encoding_version,
    )
    path = pathlib.Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    descriptor_path = path / artifacts.SETUP_CONFIG_JSON
    descriptor_path.write_text(descriptor.model_dump_json(indent=2), encoding="utf-8")
    return descriptor_path


###### PRIVATE #######


def _setup_config_from_run_config(cfg: config.RunConfig) -> SetupConfig:
    """Derive a :class:`SetupConfig` descriptor from a :class:`RunConfig`."""
    return SetupConfig(
        run_name=cfg.run.run_name,
        setup_encoding=cfg.setup_encoding,
        setup_arch=cfg.setup_arch,
        main_arch=cfg.arch,
        version=cfg.architecture.encoding_version,
    )
