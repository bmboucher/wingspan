"""Per-run JSON metadata sidecars: the model descriptor and the session log.

These complement the binary checkpoints with the two human-readable JSON
artifacts a long run leaves behind in its checkpoint dir, alongside ``last.pt``
and the ``metrics.jsonl`` / ``games.jsonl`` history logs:

* ``model_config.json`` — the weight-compatibility descriptor (network shape +
  judgment-family head order). One per run, rewritten on every startup. If any
  field here changes the previously-saved weights can no longer be trusted, and
  the resume gate (``loop._architecture_matches``) refuses to load them.

* ``process_<stamp>.json`` — one *session's* meta-configuration (the batch size,
  learning rate, eval cadence, device, … — the full :class:`config.TrainConfig`
  in force) plus the runtime context it started in (git SHA, the iteration it
  resumed from, the wall-clock start). A run is stopped and restarted many
  times; each startup drops a fresh dated file, so the sequence of these files
  is the run's session ledger.

The writers are split out of ``loop`` (and kept torch-free) so they can be unit
tested without a training run, mirroring the ``artifacts`` / ``runs`` split. The
caller supplies the timestamp strings so these functions stay pure.

This module also owns the descriptor-derived reporting seam (the ``*_for`` /
``build_*`` functions): every layout, input width, and parameter count shown by
``wingspan inspect`` or written to ``model_inspect.json`` /
``model_summary.html`` derives from a :class:`ModelConfig` descriptor and is
**era-routed by its artifact version** (pre-0.1 descriptors get the frozen
``wingspan.compat.v0_0`` geometry). The run-start writers funnel through the
same seam via :func:`_descriptor_for`, so a run's reports are consistent with
its ``model_config.json`` by construction.
"""

from __future__ import annotations

import pathlib

import pydantic

from wingspan import architecture, encode, setup_model, version
from wingspan.encode import stripes as encode_stripes
from wingspan.reporting import html as report
from wingspan.training import artifacts, config

# Up to this many same-second restarts get a unique ``process_<stamp>-N.json``
# before the writer gives up de-duplicating and overwrites the base name.
_MAX_SESSION_SUFFIX = 1000


class ModelConfig(pydantic.BaseModel):
    """The full network descriptor written to ``model_config.json``.

    Carries the encoding dims and family-head order the net was trained against
    plus its complete :class:`architecture.ModelArchitecture` topology, so the
    file both reads as a one-glance summary of the run's network *and* fully
    reconstitutes it (``model.PolicyValueNet.from_model_config``). The
    weight-compatibility signature ``TrainConfig.architecture_key`` is derived
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
    """One training session's process record (``process_<stamp>.json``).

    Embeds the full :class:`config.TrainConfig` in force (so the batch sizes and
    every other knob are captured without field drift) plus the runtime context
    the session began in.
    """

    run_name: str
    started_at: str  # ISO-8601 local start time
    git_sha: str | None
    resumed: bool  # whether this session continued an existing run
    resumed_from_iteration: int  # 0 for a fresh start
    config: config.TrainConfig


class InspectReport(pydantic.BaseModel):
    """The encoding + parameter breakdown saved as ``model_inspect.json``.

    Written alongside ``model_config.json`` at the start of every fresh run so
    the checkpoint directory is fully self-documenting: the state and choice
    stripe registry explains every input element, and the parameter report gives
    the per-layer accounting for the network's trainable weight count.
    """

    state_layout: encode_stripes.VectorLayout
    choice_layout: encode_stripes.VectorLayout
    param_report: architecture.ParamReport
    total_params: int


def write_model_config(checkpoint_dir: str, cfg: config.TrainConfig) -> pathlib.Path:
    """Write (overwriting) ``model_config.json`` for ``cfg`` and return its path."""
    descriptor = _descriptor_for(cfg)
    path = _ensure_dir(checkpoint_dir) / artifacts.MODEL_CONFIG_JSON
    path.write_text(descriptor.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_inspect_report(checkpoint_dir: str, cfg: config.TrainConfig) -> pathlib.Path:
    """Write (overwriting) ``model_inspect.json`` for ``cfg`` and return its path.

    Derives everything from the same descriptor :func:`write_model_config`
    writes, via :func:`build_inspect_report`, so the JSON breakdown is
    consistent with the run's ``model_config.json`` by construction. The file
    is self-documenting: every input element is named and described, and the
    parameter count matches ``sum(p.numel())`` of the equivalent
    :class:`~wingspan.model.PolicyValueNet`.
    """
    inspect_report = build_inspect_report(_descriptor_for(cfg))
    path = _ensure_dir(checkpoint_dir) / artifacts.INSPECT_REPORT_JSON
    path.write_text(inspect_report.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_model_summary_html(
    checkpoint_dir: str, cfg: config.TrainConfig
) -> pathlib.Path:
    """Write (overwriting) ``model_summary.html`` for ``cfg`` and return its path.

    Produces a self-contained browser-readable summary covering the full
    state/choice vector layouts (with per-element drill-down), the network
    architecture diagram, and the per-layer parameter accounting.  Regenerated
    on every startup, matching the contract of :func:`write_model_config`, and
    built by the same :func:`build_model_summary_html` that ``wingspan inspect
    --html`` uses — the two surfaces are identical by construction.
    """
    html_content = build_model_summary_html(
        _descriptor_for(cfg), cfg.setup_arch, cfg.setup_encoding
    )
    path = _ensure_dir(checkpoint_dir) / artifacts.MODEL_SUMMARY_HTML
    path.write_text(html_content, encoding="utf-8")
    return path


def read_model_config(checkpoint_dir: str) -> ModelConfig:
    """Read the ``model_config.json`` topology descriptor from ``checkpoint_dir``.

    Pairs with :func:`write_model_config`: the returned descriptor reconstitutes
    the run's network via ``model.PolicyValueNet.from_model_config``. Raises
    ``FileNotFoundError`` if the run has no descriptor on disk, and
    ``version.IncompatibleArtifactError`` when the descriptor's artifact version
    is outside the current code's load guarantee."""
    path = pathlib.Path(checkpoint_dir) / artifacts.MODEL_CONFIG_JSON
    descriptor = ModelConfig.model_validate_json(path.read_text(encoding="utf-8"))
    version.check_artifact_compatible(
        descriptor.version, what=f"{artifacts.MODEL_CONFIG_JSON} at {checkpoint_dir}"
    )
    return descriptor


def build_inspect_report(descriptor: ModelConfig) -> InspectReport:
    """The encoding + parameter breakdown for ``descriptor``, era-routed.

    The single source behind ``model_inspect.json`` and the ``wingspan
    inspect`` tables: every layout and count derives from the run's own
    descriptor — never the live encoder alone — so the report matches the
    checkpoint the descriptor describes regardless of its artifact era.
    """
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
    """The standalone ``model_summary.html`` document for ``descriptor``.

    The single builder behind both the run-start writer
    (:func:`write_model_summary_html`) and ``wingspan inspect --html``, so the
    two surfaces are identical by construction. ``setup_arch`` is the one
    datum not on the descriptor (it lives in ``setup_config.json``); the
    separate setup model is active exactly when the main net does not carry
    setup (``include_setup`` off). ``setup_encoding`` controls which stripes
    appear in the setup vector breakdown — defaults to ``SetupEncoding()`` (the
    pre-0.2 all-splits-off layout) when not supplied.
    """
    enc = setup_encoding if setup_encoding is not None else setup_model.SetupEncoding()
    return report.generate_html_report(
        state_layout_for(descriptor),
        choice_layout_for(descriptor),
        param_report_for(descriptor),
        descriptor.architecture,
        setup_layout=setup_model.setup_readout_stripe_layout(
            enc,
            descriptor.architecture.card_embed_dim,
            descriptor.architecture.hand_embed_width,
        ),
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
    with both the choice-encoder and card-encoder input widths era-routed so the
    totals match the actual checkpoint — ``sum(p.numel())`` of the net
    ``model.PolicyValueNet.from_model_config`` reconstitutes."""
    from wingspan.compat import v0_1  # local: compat imports the model package

    arch = descriptor.architecture
    # Pre-0.2 checkpoints have a 229-wide card encoder; use the frozen dim.
    card_feat_in = (
        v0_1.CARD_FEATURE_DIM_V01
        if v0_1.uses_v0_1_card_feature_encoding(descriptor.version)
        else encode.CARD_FEATURE_DIM
    )
    return architecture.count_parameters(
        arch,
        card_feat_in=card_feat_in,
        trunk_in=encode.trunk_input_dim(
            descriptor.state_dim,
            arch.card_embed_dim,
            use_distinct_hand_model=arch.use_distinct_hand_model,
            hand_embed_dim=arch.hand_embed_dim,
            tray_set_embedding=arch.tray_set_embedding,
        ),
        choice_in=choice_input_dim_for(descriptor),
        num_families=len(descriptor.family_order),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
    )


def state_layout_for(descriptor: ModelConfig) -> encode_stripes.VectorLayout:
    """The post-embedding state stripe registry for ``descriptor``.

    Always the live registry: the state encoding is era-invariant through
    artifact version 0.1 (only the choice vector reshaped). A future era that
    reshapes the state vector must extend this with version routing, exactly
    like :func:`choice_layout_for`.
    """
    arch = descriptor.architecture
    return encode_stripes.state_stripe_layout(
        encode.EncodingSpec(include_setup=descriptor.include_setup),
        arch.card_embed_dim,
        use_distinct_hand_model=arch.use_distinct_hand_model,
        hand_embed_dim=arch.hand_embed_dim,
        tray_set_embedding=arch.tray_set_embedding,
    )


def choice_layout_for(descriptor: ModelConfig) -> encode_stripes.VectorLayout:
    """The post-embedding choice stripe registry for ``descriptor``,
    era-routed: pre-0.1 artifacts get the frozen v0.0 registry (habitat
    stripe, 180-wide bird one-hot), current ones the live registry."""
    from wingspan.compat import v0_0  # local: compat imports the model package

    spec = encode.EncodingSpec(include_setup=descriptor.include_setup)
    if v0_0.uses_v0_0_choice_encoding(descriptor.version):
        return v0_0.choice_stripe_layout(spec, descriptor.architecture.card_embed_dim)
    return encode_stripes.choice_stripe_layout(
        spec, descriptor.architecture.card_embed_dim
    )


def choice_input_dim_for(descriptor: ModelConfig) -> int:
    """The choice encoder's first-``Linear`` input width for ``descriptor``,
    era-routed: the pre-0.1 formula has no ``include_setup`` axis (the keep
    multi-hot rode the bird stripe)."""
    from wingspan.compat import v0_0  # local: compat imports the model package

    if v0_0.uses_v0_0_choice_encoding(descriptor.version):
        return v0_0.choice_input_dim(
            descriptor.choice_dim, descriptor.architecture.card_embed_dim
        )
    return encode.choice_input_dim(
        descriptor.choice_dim,
        descriptor.architecture.card_embed_dim,
        include_setup=descriptor.include_setup,
    )


def choice_extra_for(descriptor: ModelConfig) -> int:
    """The choice encoder's passthrough width for ``descriptor`` — the row
    columns that are not card-region embedding lookups (the architecture
    diagram's "additional inputs" count), era-routed like
    :func:`choice_input_dim_for`."""
    from wingspan.compat import v0_0  # local: compat imports the model package

    if v0_0.uses_v0_0_choice_encoding(descriptor.version):
        return v0_0.choice_passthrough_dim(descriptor.choice_dim)
    return encode.choice_passthrough_dim(
        descriptor.choice_dim, include_setup=descriptor.include_setup
    )


def write_session_record(
    checkpoint_dir: str,
    cfg: config.TrainConfig,
    *,
    stamp: str,
    started_at: str,
    git_sha: str | None,
    resumed_from_iteration: int,
) -> pathlib.Path:
    """Write a fresh dated ``process_<stamp>.json`` for this session and return
    its path. ``stamp`` is a filesystem-safe timestamp supplied by the caller (so
    this stays pure / testable); a same-second collision gets a ``-N`` suffix so
    rapid restarts never clobber a prior session's record."""
    record = SessionRecord(
        run_name=cfg.run_name,
        started_at=started_at,
        git_sha=git_sha,
        resumed=resumed_from_iteration > 0,
        resumed_from_iteration=resumed_from_iteration,
        config=cfg,
    )
    path = _unique_session_path(_ensure_dir(checkpoint_dir), stamp)
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


###### PRIVATE #######


def _descriptor_for(cfg: config.TrainConfig) -> ModelConfig:
    """The :class:`ModelConfig` descriptor for ``cfg``, stamped at the current
    artifact version — the single construction every run-start writer shares,
    so the JSON / HTML reports describe exactly what ``model_config.json``
    records."""
    return ModelConfig(
        run_name=cfg.run_name,
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        family_order=cfg.family_order,
        architecture=cfg.arch,
        include_setup=cfg.encoding_spec.include_setup,
        version=version.MODEL_VERSION,
    )


def _ensure_dir(checkpoint_dir: str) -> pathlib.Path:
    path = pathlib.Path(checkpoint_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unique_session_path(directory: pathlib.Path, stamp: str) -> pathlib.Path:
    """``process_<stamp>.json`` if free, else ``…-1`` / ``…-2`` / … so two
    same-second startups never overwrite each other's session record."""
    base = directory / f"{artifacts.PROCESS_PREFIX}{stamp}.json"
    if not base.exists():
        return base
    for index in range(1, _MAX_SESSION_SUFFIX):
        candidate = directory / f"{artifacts.PROCESS_PREFIX}{stamp}-{index}.json"
        if not candidate.exists():
            return candidate
    return base
