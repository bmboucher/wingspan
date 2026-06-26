"""Per-run JSON metadata sidecars: the model descriptor and the session log.

These complement the binary checkpoints with the human-readable JSON artifacts
a training run leaves in its checkpoint dir alongside ``last.pt`` and the
``metrics.jsonl`` / ``games.jsonl`` history logs.

**≥0.5 runs** write a single dated ``run_config_<stamp>.json``
(:class:`config.RunConfigFile`) per session.  All the data that used to be
spread across three files (``model_config.json``, ``setup_config.json``,
``process_<stamp>.json``) lives in one place; ``model_config.json`` and
``setup_config.json`` are no longer written.

**≤0.4 run directories** still carry the legacy three-file layout.  The reader
functions (:func:`read_model_config`, :func:`read_run_config`) dispatch on
presence so old run dirs load without any migration.

This module also owns the descriptor-derived reporting seam (the ``*_for`` /
``build_*`` functions): every layout, input width, and parameter count shown by
``wingspan inspect`` or written to ``model_inspect.json`` / ``model_summary.html``
derives from a :class:`ModelConfig` descriptor and is **era-routed by its
artifact version** (pre-0.1 descriptors get the frozen
``wingspan.compat.v0_0`` geometry). The run-start writers funnel through the
same seam via :func:`_descriptor_for`, so a run's reports are consistent with
the saved config by construction.
"""

from __future__ import annotations

import pathlib

import pydantic

from wingspan import architecture, encode, setup_model, version
from wingspan.encode import stripes as encode_stripes
from wingspan.reporting import html as report
from wingspan.training import artifacts, config

# Up to this many same-second restarts get a unique ``run_config_<stamp>-N.json``
# before the writer gives up de-duplicating and overwrites the base name.
_MAX_SESSION_SUFFIX = 1000


class ModelConfig(pydantic.BaseModel):
    """The full network descriptor written to ``model_config.json`` (legacy) or
    derived from ``run_config_<stamp>.json`` (≥0.5).

    Carries the encoding dims and family-head order the net was trained against
    plus its complete :class:`architecture.ModelArchitecture` topology, so the
    file both reads as a one-glance summary of the run's network *and* fully
    reconstitutes it (``model.PolicyValueNet.from_model_config``). The
    weight-compatibility signature ``RunConfig.architecture_key`` is derived
    from exactly these fields, so a change to any of them invalidates
    previously-trained weights.
    """

    run_name: str
    state_dim: int
    choice_dim: int
    family_order: tuple[str, ...]
    architecture: architecture.ModelArchitecture
    # Whether the main net carries the opening (``encode.EncodingSpec.include_setup``).
    include_setup: bool
    # The artifact-compatibility version the run was written at; defaults so
    # files that predate the field read as the pre-versioning era ("0.0").
    version: str = version.PRE_VERSIONING_VERSION


class SessionRecord(pydantic.BaseModel):
    """One training session's process record (legacy ``process_<stamp>.json``).

    Kept for backward-compat deserialization of ≤0.4 run dirs only.
    ≥0.5 runs use :class:`config.RunConfigFile` instead.
    """

    run_name: str
    started_at: str
    git_sha: str | None
    resumed: bool
    resumed_from_iteration: int
    config: config.RunConfig


class InspectReport(pydantic.BaseModel):
    """The encoding + parameter breakdown saved as ``model_inspect.json``."""

    state_layout: encode_stripes.VectorLayout
    choice_layout: encode_stripes.VectorLayout
    param_report: architecture.ParamReport
    total_params: int


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_run_config(
    checkpoint_dir: str,
    cfg: config.RunConfig,
    *,
    stamp: str,
    started_at: str,
    git_sha: str | None,
    resumed_from_iteration: int,
) -> pathlib.Path:
    """Write a fresh dated ``run_config_<stamp>.json`` for this session (≥0.5).

    Replaces the three legacy writers (model_config + setup_config + process).
    ``stamp`` is a filesystem-safe timestamp supplied by the caller (pure /
    testable); same-second collisions get a ``-N`` suffix."""
    file = config.RunConfigFile(
        version=cfg.architecture.encoding_version,
        saved_at=started_at,
        started_at=started_at,
        git_sha=git_sha,
        resumed=resumed_from_iteration > 0,
        resumed_from_iteration=resumed_from_iteration,
        config=cfg,
    )
    path = _unique_session_path(
        _ensure_dir(checkpoint_dir), stamp, artifacts.RUN_CONFIG_PREFIX
    )
    path.write_text(file.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_inspect_report(checkpoint_dir: str, cfg: config.RunConfig) -> pathlib.Path:
    """Write (overwriting) ``model_inspect.json`` for ``cfg`` and return its path."""
    inspect_report = build_inspect_report(_descriptor_for(cfg))
    path = _ensure_dir(checkpoint_dir) / artifacts.INSPECT_REPORT_JSON
    path.write_text(inspect_report.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_model_summary_html(
    checkpoint_dir: str, cfg: config.RunConfig
) -> pathlib.Path:
    """Write (overwriting) ``model_summary.html`` for ``cfg`` and return its path."""
    html_content = build_model_summary_html(
        _descriptor_for(cfg), cfg.setup_arch, cfg.setup_encoding
    )
    path = _ensure_dir(checkpoint_dir) / artifacts.MODEL_SUMMARY_HTML
    path.write_text(html_content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_run_config(checkpoint_dir: str) -> config.RunConfigFile:
    """Read the newest ``run_config_<stamp>.json`` from ``checkpoint_dir``.

    Only present in ≥0.5 run directories. Raises ``FileNotFoundError`` when
    none exists (callers that need backward compat should check first)."""
    directory = pathlib.Path(checkpoint_dir)
    matches = sorted(directory.glob(artifacts.RUN_CONFIG_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No {artifacts.RUN_CONFIG_GLOB} found in {checkpoint_dir}"
        )
    newest = matches[-1]
    file = config.RunConfigFile.model_validate_json(newest.read_text(encoding="utf-8"))
    version.check_artifact_compatible(
        file.version, what=f"{newest.name} at {checkpoint_dir}"
    )
    return file


def read_model_config(checkpoint_dir: str) -> ModelConfig:
    """Read the weight-compatibility descriptor from ``checkpoint_dir``.

    Dispatches by presence:

    * **≥0.5** run dirs contain ``run_config_<stamp>.json`` — the descriptor is
      derived from the newest file's ``config`` field.
    * **≤0.4** run dirs contain ``model_config.json`` — read directly (legacy
      path, unchanged so compat tests pass).

    Raises ``FileNotFoundError`` if neither artifact is present, and
    ``version.IncompatibleArtifactError`` when the version is outside the load
    guarantee.
    """
    directory = pathlib.Path(checkpoint_dir)

    # ≥0.5 path: derive ModelConfig from the unified file.
    unified_matches = sorted(directory.glob(artifacts.RUN_CONFIG_GLOB))
    if unified_matches:
        file = config.RunConfigFile.model_validate_json(
            unified_matches[-1].read_text(encoding="utf-8")
        )
        version.check_artifact_compatible(
            file.version,
            what=f"{unified_matches[-1].name} at {checkpoint_dir}",
        )
        return _descriptor_for(file.config)

    # ≤0.4 legacy path: read model_config.json directly.
    path = directory / artifacts.MODEL_CONFIG_JSON
    descriptor = ModelConfig.model_validate_json(path.read_text(encoding="utf-8"))
    version.check_artifact_compatible(
        descriptor.version, what=f"{artifacts.MODEL_CONFIG_JSON} at {checkpoint_dir}"
    )
    return descriptor


# ---------------------------------------------------------------------------
# Legacy writer (kept for tests / manual tooling; not called by the loop)
# ---------------------------------------------------------------------------


def write_model_config(checkpoint_dir: str, cfg: config.RunConfig) -> pathlib.Path:
    """Write (overwriting) ``model_config.json`` for ``cfg`` and return its path.

    Called only by tooling that targets ≤0.4 run directories; the training loop
    uses :func:`write_run_config` for ≥0.5 runs."""
    descriptor = _descriptor_for(cfg)
    path = _ensure_dir(checkpoint_dir) / artifacts.MODEL_CONFIG_JSON
    path.write_text(descriptor.model_dump_json(indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Descriptor-derived reporting seam
# ---------------------------------------------------------------------------


def build_inspect_report(descriptor: ModelConfig) -> InspectReport:
    """The encoding + parameter breakdown for ``descriptor``, era-routed."""
    param_report = param_report_for(descriptor)
    return InspectReport(
        state_layout=state_layout_for(descriptor),
        choice_layout=choice_layout_for(descriptor),
        param_report=param_report,
        total_params=param_report.total,
    )


def build_model_summary_html(
    descriptor: ModelConfig,
    setup_arch: setup_model.SetupArchitecture,
    setup_encoding: setup_model.SetupEncoding | None = None,
) -> str:
    """The standalone ``model_summary.html`` document for ``descriptor``."""
    enc = setup_encoding if setup_encoding is not None else setup_model.SetupEncoding()
    return report.generate_html_report(
        state_layout_for(descriptor),
        choice_layout_for(descriptor),
        param_report_for(descriptor),
        descriptor.architecture,
        setup_encoding=enc,
        setup_arch=setup_arch,
        use_setup_model=not descriptor.include_setup,
        state_dim=descriptor.state_dim,
        choice_dim=descriptor.choice_dim,
        family_order=descriptor.family_order,
        run_name=descriptor.run_name,
        model_version=descriptor.version,
    )


def param_report_for(descriptor: ModelConfig) -> architecture.ParamReport:
    """The per-layer / per-block parameter accounting for ``descriptor``'s net,
    era-routed so totals match the actual checkpoint."""
    from wingspan.compat import (  # local: compat imports the model package
        v0_1,
        v0_4,
        v0_6,
    )

    arch = descriptor.architecture
    card_feat_in = (
        v0_1.CARD_FEATURE_DIM_V01
        if v0_1.uses_v0_1_card_feature_encoding(descriptor.version)
        else (
            v0_6.CARD_FEATURE_DIM_V06
            if v0_6.uses_v0_6_card_feature_encoding(descriptor.version)
            else encode.CARD_FEATURE_DIM
        )
    )
    # Pre-0.6 artifacts (0.0–0.5) have 0 playability multi-hots; v0.6+ have
    # N_HAND_PLAYABLE_MULTIHOTS.  The playability stripes were added in 0.6, so
    # any same-major version below 0.6 must use n_playable=0.
    parsed_ver = version.parse_version(descriptor.version)
    playability_ver = version.parse_version(v0_4.PLAYABILITY_STRIPES_ADDED_IN)
    n_playable = (
        0
        if (parsed_ver.major, parsed_ver.minor)
        < (playability_ver.major, playability_ver.minor)
        else encode.N_HAND_PLAYABLE_MULTIHOTS
    )
    # Pre-0.9 artifacts carry the 10-dim hand_summary stripe in the state vector;
    # the distinct-hand path excises it from the continuous feed (trunk_in is 10
    # narrower than state_dim would otherwise suggest).
    compaction_ver = version.parse_version("0.9")
    hand_summary_in_state = arch.use_distinct_hand_model and (
        parsed_ver.major,
        parsed_ver.minor,
    ) < (compaction_ver.major, compaction_ver.minor)
    return architecture.count_parameters(
        arch,
        card_feat_in=card_feat_in,
        trunk_in=encode.trunk_input_dim(
            descriptor.state_dim,
            arch.card_embed_dim,
            use_distinct_hand_model=arch.use_distinct_hand_model,
            hand_summary_in_state=hand_summary_in_state,
            hand_embed_dim=arch.hand_embed_dim,
            pooled_hand_width=arch.pooled_hand_width,
            tray_set_embedding=arch.tray_set_embedding,
            n_playable_multihots=n_playable,
        ),
        choice_in=choice_input_dim_for(descriptor),
        num_families=len(descriptor.family_order),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
    )


def state_layout_for(descriptor: ModelConfig) -> encode_stripes.VectorLayout:
    """The post-embedding state stripe registry for ``descriptor``, era-routed."""
    from wingspan.compat import (  # local: compat imports the model package
        v0_2,
        v0_3,
        v0_4,
    )

    arch = descriptor.architecture
    spec = encode.EncodingSpec(include_setup=descriptor.include_setup)
    if v0_2.uses_v0_2_state_encoding(descriptor.version):
        return v0_2.state_stripe_layout_v02(
            spec,
            arch.card_embed_dim,
            use_distinct_hand_model=arch.use_distinct_hand_model,
            hand_embed_dim=arch.hand_embed_dim,
            tray_set_embedding=arch.tray_set_embedding,
        )
    if v0_3.uses_v0_3_state_encoding(descriptor.version):
        return v0_3.state_stripe_layout_v03(
            spec,
            arch.card_embed_dim,
            use_distinct_hand_model=arch.use_distinct_hand_model,
            hand_embed_dim=arch.hand_embed_dim,
            tray_set_embedding=arch.tray_set_embedding,
        )
    parsed_ver = version.parse_version(descriptor.version)
    playability_ver = version.parse_version(v0_4.PLAYABILITY_STRIPES_ADDED_IN)
    n_playable = (
        0
        if (parsed_ver.major, parsed_ver.minor)
        < (playability_ver.major, playability_ver.minor)
        else encode.N_HAND_PLAYABLE_MULTIHOTS
    )
    return encode_stripes.state_stripe_layout(
        spec,
        arch.card_embed_dim,
        use_distinct_hand_model=arch.use_distinct_hand_model,
        use_board_attention=arch.use_board_attention,
        hand_embed_dim=arch.hand_embed_dim,
        pooled_hand_width=arch.pooled_hand_width,
        tray_set_embedding=arch.tray_set_embedding,
        n_playable_multihots=n_playable,
    )


def choice_layout_for(descriptor: ModelConfig) -> encode_stripes.VectorLayout:
    """The post-embedding choice stripe registry for ``descriptor``, era-routed."""
    from wingspan.compat import v0_0, v0_4  # local: compat imports the model package

    spec = encode.EncodingSpec(include_setup=descriptor.include_setup)
    if v0_0.uses_v0_0_choice_encoding(descriptor.version):
        return v0_0.choice_stripe_layout(spec, descriptor.architecture.card_embed_dim)
    parsed_ver = version.parse_version(descriptor.version)
    playability_ver = version.parse_version(v0_4.PLAYABILITY_STRIPES_ADDED_IN)
    has_becomes = (parsed_ver.major, parsed_ver.minor) >= (
        playability_ver.major,
        playability_ver.minor,
    )
    return encode_stripes.choice_stripe_layout(
        spec, descriptor.architecture.card_embed_dim, has_becomes_playable=has_becomes
    )


def choice_input_dim_for(descriptor: ModelConfig) -> int:
    """The choice encoder's first-``Linear`` input width for ``descriptor``.

    Era-routed: v0.0 artifacts use the v0.0 formula (180-wide bird one-hot +
    board-idx embedding); v0.1–0.8 artifacts use the v0.8 board-bearing formula
    (board_idx 15 embedded + single bird_id); pre-0.6 additionally pass
    ``has_becomes_playable=False``. v0.9+ use the live formula (no board_idx)."""
    from wingspan.compat import v0_0, v0_4, v0_8  # local: compat imports model package

    if v0_0.uses_v0_0_choice_encoding(descriptor.version):
        return v0_0.choice_input_dim(
            descriptor.choice_dim, descriptor.architecture.card_embed_dim
        )
    if v0_8.uses_v0_8_choice_encoding(descriptor.version):
        parsed_ver = version.parse_version(descriptor.version)
        playability_ver = version.parse_version(v0_4.PLAYABILITY_STRIPES_ADDED_IN)
        has_becomes = (parsed_ver.major, parsed_ver.minor) >= (
            playability_ver.major,
            playability_ver.minor,
        )
        return v0_8.choice_input_dim_v08(
            descriptor.choice_dim,
            descriptor.architecture.card_embed_dim,
            include_setup=descriptor.include_setup,
            has_becomes_playable=has_becomes,
        )
    parsed_ver = version.parse_version(descriptor.version)
    playability_ver = version.parse_version(v0_4.PLAYABILITY_STRIPES_ADDED_IN)
    has_becomes = (parsed_ver.major, parsed_ver.minor) >= (
        playability_ver.major,
        playability_ver.minor,
    )
    return encode.choice_input_dim(
        descriptor.choice_dim,
        descriptor.architecture.card_embed_dim,
        include_setup=descriptor.include_setup,
        has_becomes_playable=has_becomes,
    )


def choice_extra_for(descriptor: ModelConfig) -> int:
    """The choice encoder's passthrough width for ``descriptor``, era-routed."""
    from wingspan.compat import v0_0, v0_4, v0_8  # local: compat imports model package

    if v0_0.uses_v0_0_choice_encoding(descriptor.version):
        return v0_0.choice_passthrough_dim(descriptor.choice_dim)
    if v0_8.uses_v0_8_choice_encoding(descriptor.version):
        parsed_ver = version.parse_version(descriptor.version)
        playability_ver = version.parse_version(v0_4.PLAYABILITY_STRIPES_ADDED_IN)
        has_becomes = (parsed_ver.major, parsed_ver.minor) >= (
            playability_ver.major,
            playability_ver.minor,
        )
        return v0_8.choice_passthrough_dim_v08(
            descriptor.choice_dim,
            include_setup=descriptor.include_setup,
            has_becomes_playable=has_becomes,
        )
    parsed_ver = version.parse_version(descriptor.version)
    playability_ver = version.parse_version(v0_4.PLAYABILITY_STRIPES_ADDED_IN)
    has_becomes = (parsed_ver.major, parsed_ver.minor) >= (
        playability_ver.major,
        playability_ver.minor,
    )
    return encode.choice_passthrough_dim(
        descriptor.choice_dim,
        include_setup=descriptor.include_setup,
        has_becomes_playable=has_becomes,
    )


# ---------------------------------------------------------------------------
# Legacy session-record writer (≤0.4; kept for any tooling that needs it)
# ---------------------------------------------------------------------------


def write_session_record(
    checkpoint_dir: str,
    cfg: config.RunConfig,
    *,
    stamp: str,
    started_at: str,
    git_sha: str | None,
    resumed_from_iteration: int,
) -> pathlib.Path:
    """Write a legacy ``process_<stamp>.json`` and return its path.

    Not called by the training loop for ≥0.5 runs; kept for tooling / tests."""
    record = SessionRecord(
        run_name=cfg.run.run_name,
        started_at=started_at,
        git_sha=git_sha,
        resumed=resumed_from_iteration > 0,
        resumed_from_iteration=resumed_from_iteration,
        config=cfg,
    )
    path = _unique_session_path(
        _ensure_dir(checkpoint_dir), stamp, artifacts.PROCESS_PREFIX
    )
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


###### PRIVATE #######


def _descriptor_for(cfg: config.RunConfig) -> ModelConfig:
    """The :class:`ModelConfig` descriptor for ``cfg``, stamped at the run's
    artifact era — the single construction every writer and reporter shares."""
    arch_cfg = cfg.architecture
    return ModelConfig(
        run_name=cfg.run.run_name,
        state_dim=arch_cfg.state_dim,
        choice_dim=arch_cfg.choice_dim,
        family_order=arch_cfg.family_order,
        architecture=cfg.arch,
        include_setup=cfg.encoding_spec.include_setup,
        version=arch_cfg.encoding_version,
    )


def _ensure_dir(checkpoint_dir: str) -> pathlib.Path:
    path = pathlib.Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unique_session_path(
    directory: pathlib.Path, stamp: str, prefix: str
) -> pathlib.Path:
    """``<prefix><stamp>.json`` if free, else ``…-1`` / ``…-2`` / … so two
    same-second startups never overwrite each other's session record."""
    base = directory / f"{prefix}{stamp}.json"
    if not base.exists():
        return base
    for index in range(1, _MAX_SESSION_SUFFIX):
        candidate = directory / f"{prefix}{stamp}-{index}.json"
        if not candidate.exists():
            return candidate
    return base
