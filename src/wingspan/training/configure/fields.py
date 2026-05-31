"""The editable ``TrainConfig`` fields: per-kind display metadata and the pure
read / format / commit / nudge helpers the configurator builds on.

There is one :class:`FieldSpec` subclass per *kind* of value (mirroring the
``decisions.Choice`` hierarchy) so consumers branch via ``isinstance`` and read
typed, always-present attributes instead of ``None``-checking a flat spec. The
bounds a value must satisfy are deliberately NOT duplicated here — they live as
declarative ``Annotated[..., Field(...)]`` constraints on ``TrainConfig`` itself,
and every edit is committed through :func:`commit` / :func:`nudge`, which route
the candidate value through ``TrainConfig.model_validate`` so an out-of-range
value is rejected by the model. This module carries only presentation metadata,
the nudge step, and the parse layer.

``resume`` is intentionally absent: it is a launch-time decision (Start vs New
run), not a hyperparameter, so the controller sets it, not the form. The
architecture descriptors (``state_dim`` / ``choice_dim`` / ``family_order``) are
derived and never editable.
"""

from __future__ import annotations

import enum
import typing

import pydantic

from wingspan.training import config

# A single editable scalar value. Every editable field is one of these; ``bool``
# is intentionally excluded (the only bool, ``resume``, is an action not an edit)
# so the union never has to disambiguate ``bool`` from ``int`` under isinstance.
type FieldValue = int | float | str


class ConfigSection(enum.StrEnum):
    """The form's field groups, shown as headers in display order."""

    LOOP = "loop shape"
    OPTIM = "optimization"
    EVAL = "evaluation"
    MODEL = "model & seed"
    RUNTIME = "runtime"
    CHECKPOINT = "checkpoints"


# The order sections appear in the form.
SECTION_ORDER: tuple[ConfigSection, ...] = (
    ConfigSection.LOOP,
    ConfigSection.OPTIM,
    ConfigSection.EVAL,
    ConfigSection.MODEL,
    ConfigSection.RUNTIME,
    ConfigSection.CHECKPOINT,
)


class ChangeImpact(enum.StrEnum):
    """How changing a field relative to a *resumed* run affects continuity.

    Distinct from architecture compatibility: a ``REGIME`` change still resumes
    the weights, but shifts the run's metrics/semantics so the live charts blend
    two configs across the boundary; only ``FRESH`` changes the network shape so
    the checkpoint cannot be loaded at all.
    """

    NONE = "none"  # applied live; safe to change on a resumed run
    REGIME = "regime"  # resumes, but reinterprets the in-progress run
    FRESH = "fresh"  # changes the network shape — weights can't load


class FieldSpec(pydantic.BaseModel):
    """Shared display metadata for one editable ``TrainConfig`` field."""

    attr: str
    label: str
    section: ConfigSection
    help: str
    unit: str = ""
    impact: ChangeImpact = ChangeImpact.NONE


class IntField(FieldSpec):
    """An integer field; ``step`` is the +/- nudge increment."""

    step: int = 1


class FloatField(FieldSpec):
    """A real-valued field. ``scientific`` renders it in ``1e-04`` form."""

    step: float
    scientific: bool = False


class ChoiceField(FieldSpec):
    """A field cycled through a fixed set of string options."""

    choices: list[str]


class TextField(FieldSpec):
    """A free-text field (e.g. the run name)."""


class PathField(FieldSpec):
    """A filesystem-path field (e.g. the checkpoint directory)."""


# The full editable surface, grouped by section. Help text is distilled from the
# field comments in ``config.py``; steps are the LEFT/RIGHT nudge increments.
FIELD_SPECS: list[FieldSpec] = [
    IntField(
        attr="games_per_iter",
        label="games / iter",
        section=ConfigSection.LOOP,
        unit="games",
        step=8,
        help="Self-play games collected and learned from per collect→update cycle. "
        "More games = a less noisy gradient but a slower iteration.",
    ),
    IntField(
        attr="max_iterations",
        label="max iterations",
        section=ConfigSection.LOOP,
        unit="iters",
        step=1,
        help="Additional iterations to run THIS session (0 = until you press Stop). "
        "On a resumed run this is counted from the resume point, not from zero.",
    ),
    FloatField(
        attr="lr",
        label="learning rate",
        section=ConfigSection.OPTIM,
        step=1e-4,
        scientific=True,
        help="Adam step size for the REINFORCE + value-baseline update. Honored "
        "immediately on resume (the optimizer momentum is kept).",
    ),
    FloatField(
        attr="value_coef",
        label="value coef",
        section=ConfigSection.OPTIM,
        step=0.05,
        help="Weight of the value-head (critic) loss term in the total loss.",
    ),
    FloatField(
        attr="entropy_coef",
        label="entropy coef",
        section=ConfigSection.OPTIM,
        step=0.005,
        help="Entropy bonus that keeps the policy exploring; 0 disables it.",
    ),
    FloatField(
        attr="grad_clip",
        label="grad clip",
        section=ConfigSection.OPTIM,
        step=0.5,
        help="Global gradient-norm clip — the TRAINING HEALTH band flags norms "
        "above this.",
    ),
    FloatField(
        attr="score_norm",
        label="score norm",
        section=ConfigSection.OPTIM,
        step=5.0,
        impact=ChangeImpact.REGIME,
        help="Advantage / value-target scale. Changing it rescales the loss and "
        "points curves across a resume boundary, so the charts will step.",
    ),
    IntField(
        attr="eval_every",
        label="eval every",
        section=ConfigSection.EVAL,
        unit="iters",
        step=1,
        help="Run a paired eval block every N iterations; 0 disables evaluation "
        "entirely.",
    ),
    IntField(
        attr="eval_games",
        label="eval games",
        section=ConfigSection.EVAL,
        unit="games",
        step=8,
        help="Held-out games per eval, played as mirrored pairs to cancel the "
        "first-player edge (an odd value rounds down to the nearest pair).",
    ),
    FloatField(
        attr="eval_ewma_alpha",
        label="eval ewma α",
        section=ConfigSection.EVAL,
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Smoothing for the eval win-rate / margin trend (higher = more "
        "responsive). Re-smooths the restored history on resume.",
    ),
    FloatField(
        attr="opponent_reset_win_rate",
        label="opp. advance @",
        section=ConfigSection.EVAL,
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Freeze the current policy as the new opponent once smoothed win-rate "
        "clears this (0 disables). Lowering it below the current win-rate triggers "
        "an immediate advance + win-rate reset.",
    ),
    ChoiceField(
        attr="initial_vs_random",
        label="bootstrap vs random",
        section=ConfigSection.EVAL,
        choices=["True", "False"],
        help="Fresh runs only: start by collecting against the random agent "
        "(net at seat 0, eval paused) before switching to self-play. A resumed "
        "run keeps the phase stored in its checkpoint.",
    ),
    FloatField(
        attr="random_phase_win_rate",
        label="graduate @",
        section=ConfigSection.EVAL,
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Smoothed collection win-rate (vs random) at which the bootstrap "
        "phase freezes self·gen1 and switches to self-play. Lowering it below "
        "the current win-rate graduates immediately.",
    ),
    FloatField(
        attr="produce_ewma_alpha",
        label="produce ewma α",
        section=ConfigSection.EVAL,
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Smoothing for the PRODUCING band's score / margin readouts.",
    ),
    IntField(
        attr="hidden",
        label="hidden width",
        section=ConfigSection.MODEL,
        unit="units",
        step=16,
        impact=ChangeImpact.FRESH,
        help="Network hidden width. Changing it changes the architecture, so the "
        "existing checkpoint cannot be resumed — a fresh run is required.",
    ),
    IntField(
        attr="seed",
        label="seed",
        section=ConfigSection.MODEL,
        step=1,
        impact=ChangeImpact.REGIME,
        help="RNG seed for Python / NumPy / torch and the per-game deal. Changing "
        "it on resume re-randomizes the game stream (reproducibility resets).",
    ),
    ChoiceField(
        attr="device",
        label="device",
        section=ConfigSection.RUNTIME,
        choices=["cpu", "cuda"],
        impact=ChangeImpact.REGIME,
        help="Compute device. Self-play collection is typically fastest on cpu "
        "(TRAINING.md §1.4); cuda mainly helps the update step.",
    ),
    PathField(
        attr="checkpoint_dir",
        label="checkpoint dir",
        section=ConfigSection.CHECKPOINT,
        help="Where last.pt / best.pt / the metrics log / the archive folder live. "
        "Editing this re-inspects the new directory for an existing run.",
    ),
    TextField(
        attr="run_name",
        label="run name",
        section=ConfigSection.CHECKPOINT,
        help="Names this run's log file and the default archive label.",
    ),
    IntField(
        attr="history_len",
        label="history length",
        section=ConfigSection.CHECKPOINT,
        unit="iters",
        step=128,
        help="In-memory iterations retained for the live convergence charts.",
    ),
]

_BY_ATTR: dict[str, FieldSpec] = {spec.attr: spec for spec in FIELD_SPECS}
_DEFAULTS = config.TrainConfig()
_FLOAT_ROUND = 6  # decimal places a nudged float is rounded to (kills FP crud)


def spec_for(attr: str) -> FieldSpec:
    """The :class:`FieldSpec` for an editable attribute name."""
    return _BY_ATTR[attr]


def editable_attrs() -> list[str]:
    """All editable attribute names, in form display order."""
    return [spec.attr for spec in FIELD_SPECS]


def read_field(cfg: config.TrainConfig, spec: FieldSpec) -> FieldValue:
    """The current value of ``spec``'s field. The single localized cast bridges
    the dynamic attribute-name access — every editable field is a ``FieldValue``,
    so re-typing the ``getattr`` result here keeps ``Any`` out of every caller."""
    return typing.cast("FieldValue", getattr(cfg, spec.attr))


def format_value(cfg: config.TrainConfig, spec: FieldSpec) -> str:
    """The display string for ``spec``'s current value."""
    value = read_field(cfg, spec)
    if isinstance(spec, FloatField):
        if spec.scientific:
            return f"{value:.0e}"
        return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
    return str(value)


def default_string(spec: FieldSpec) -> str:
    """The factory-default value of ``spec``, formatted for the detail panel."""
    return format_value(_DEFAULTS, spec)


def is_changed(
    working: config.TrainConfig, saved: config.TrainConfig | None, spec: FieldSpec
) -> bool:
    """Whether ``spec`` differs from the saved run's value (False if no saved
    run) — drives the changed-field marker."""
    if saved is None:
        return False
    return read_field(working, spec) != read_field(saved, spec)


def commit(
    cfg: config.TrainConfig, spec: FieldSpec, raw: str
) -> tuple[config.TrainConfig, str | None]:
    """Parse ``raw`` per ``spec``'s kind and return ``(new_cfg, None)`` on
    success, or ``(cfg, error)`` if it cannot parse or the model rejects it."""
    parsed, parse_error = _parse(spec, raw)
    if parsed is None:
        return cfg, parse_error or "invalid value"
    return _validated_update(cfg, spec, parsed)


def nudge(
    cfg: config.TrainConfig, spec: FieldSpec, direction: int
) -> tuple[config.TrainConfig, str | None]:
    """Apply a LEFT (``direction == -1``) / RIGHT (``+1``) step to ``spec``.

    Numeric fields step by ``spec.step`` and are validated; a choice cycles; a
    text / path field has no step and is returned unchanged.
    """
    if isinstance(spec, ChoiceField):
        return _cycle_choice(cfg, spec, direction), None
    if isinstance(spec, IntField):
        return _validated_update(
            cfg, spec, _read_int(cfg, spec) + direction * spec.step
        )
    if isinstance(spec, FloatField):
        stepped = _read_float(cfg, spec) + direction * spec.step
        return _validated_update(cfg, spec, round(stepped, _FLOAT_ROUND))
    return cfg, None


###### PRIVATE #######


def _parse(spec: FieldSpec, raw: str) -> tuple[FieldValue | None, str | None]:
    """Parse an edit-buffer string into the field's Python type, or an error."""
    text = raw.strip()
    if isinstance(spec, IntField):
        try:
            return int(text), None
        except ValueError:
            return None, f"{spec.label}: expects a whole number"
    if isinstance(spec, FloatField):
        try:
            return float(text), None
        except ValueError:
            return None, f"{spec.label}: expects a number"
    if isinstance(spec, ChoiceField):
        if text in spec.choices:
            return text, None
        return None, f"{spec.label}: choose one of {', '.join(spec.choices)}"
    if not text:
        return None, f"{spec.label}: cannot be empty"
    return text, None


def _validated_update(
    cfg: config.TrainConfig, spec: FieldSpec, value: FieldValue
) -> tuple[config.TrainConfig, str | None]:
    """Apply ``value`` to ``spec``'s field through ``model_validate`` so the
    model's declarative bounds reject anything out of range."""
    candidate: dict[str, object] = {**cfg.model_dump(), spec.attr: value}
    try:
        return config.TrainConfig.model_validate(candidate), None
    except pydantic.ValidationError as error:
        return cfg, _friendly_error(spec, error)


def _friendly_error(spec: FieldSpec, error: pydantic.ValidationError) -> str:
    """Turn a pydantic ValidationError into a one-line field-scoped message."""
    details = error.errors()
    message = details[0]["msg"] if details else "invalid value"
    return f"{spec.label}: {message}"


def _cycle_choice(
    cfg: config.TrainConfig, spec: ChoiceField, direction: int
) -> config.TrainConfig:
    current = format_value(cfg, spec)
    index = spec.choices.index(current) if current in spec.choices else 0
    chosen = spec.choices[(index + direction) % len(spec.choices)]
    updated, _ = _validated_update(cfg, spec, chosen)
    return updated


def _read_int(cfg: config.TrainConfig, spec: FieldSpec) -> int:
    value = read_field(cfg, spec)
    return value if isinstance(value, int) else 0


def _read_float(cfg: config.TrainConfig, spec: FieldSpec) -> float:
    value = read_field(cfg, spec)
    return float(value) if isinstance(value, (int, float)) else 0.0
