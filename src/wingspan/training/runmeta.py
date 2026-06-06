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
    descriptor = ModelConfig(
        run_name=cfg.run_name,
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        family_order=cfg.family_order,
        architecture=cfg.arch,
        include_setup=cfg.encoding_spec.include_setup,
        version=version.MODEL_VERSION,
    )
    path = _ensure_dir(checkpoint_dir) / artifacts.MODEL_CONFIG_JSON
    path.write_text(descriptor.model_dump_json(indent=2), encoding="utf-8")
    return path


def write_inspect_report(checkpoint_dir: str, cfg: config.TrainConfig) -> pathlib.Path:
    """Write (overwriting) ``model_inspect.json`` for ``cfg`` and return its path.

    Builds the state and choice stripe registries from the live encoding
    constants and the per-block parameter accounting from the architecture, then
    writes the result as :class:`InspectReport` JSON.  The file is
    self-documenting: every input element is named and described, and the
    parameter count matches ``sum(p.numel())`` of the equivalent
    :class:`~wingspan.model.PolicyValueNet`.
    """
    param_report = architecture.count_parameters(
        cfg.arch,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=_trunk_input_dim(cfg),
        choice_in=encode.choice_input_dim(
            cfg.choice_dim,
            cfg.card_embed_dim,
            include_setup=cfg.encoding_spec.include_setup,
        ),
        num_families=len(cfg.family_order),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
    )
    inspect_report = InspectReport(
        state_layout=_state_layout(cfg),
        choice_layout=encode_stripes.choice_stripe_layout(
            cfg.encoding_spec, cfg.card_embed_dim
        ),
        param_report=param_report,
        total_params=param_report.total,
    )
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
    on every startup, matching the contract of :func:`write_model_config`.
    """
    param_report = architecture.count_parameters(
        cfg.arch,
        card_feat_in=encode.CARD_FEATURE_DIM,
        trunk_in=_trunk_input_dim(cfg),
        choice_in=encode.choice_input_dim(
            cfg.choice_dim,
            cfg.card_embed_dim,
            include_setup=cfg.encoding_spec.include_setup,
        ),
        num_families=len(cfg.family_order),
        hand_feat_in=encode.HAND_ENCODER_INPUT_DIM,
    )
    html_content = report.generate_html_report(
        _state_layout(cfg),
        encode_stripes.choice_stripe_layout(cfg.encoding_spec, cfg.card_embed_dim),
        param_report,
        cfg.arch,
        setup_layout=setup_model.setup_stripe_layout(),
        setup_arch=cfg.setup_arch,
        use_setup_model=cfg.use_setup_model,
        state_dim=cfg.state_dim,
        choice_dim=cfg.choice_dim,
        family_order=cfg.family_order,
        run_name=cfg.run_name,
        model_version=version.MODEL_VERSION,
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


def _trunk_input_dim(cfg: config.TrainConfig) -> int:
    """The run's post-embedding trunk input width, with every embedding knob
    (distinct hand encoder, hand embed width, tray-set embedding) threaded."""
    return encode.trunk_input_dim(
        cfg.state_dim,
        cfg.card_embed_dim,
        use_distinct_hand_model=cfg.use_distinct_hand_model,
        hand_embed_dim=cfg.hand_embed_dim,
        tray_set_embedding=cfg.tray_set_embedding,
    )


def _state_layout(cfg: config.TrainConfig) -> encode_stripes.VectorLayout:
    """The run's post-embedding state stripe registry, embedding knobs threaded
    (the choice layout takes none of them, so it is built inline)."""
    return encode_stripes.state_stripe_layout(
        cfg.encoding_spec,
        cfg.card_embed_dim,
        use_distinct_hand_model=cfg.use_distinct_hand_model,
        hand_embed_dim=cfg.hand_embed_dim,
        tray_set_embedding=cfg.tray_set_embedding,
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
