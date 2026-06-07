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

from wingspan import architecture
from wingspan.training import config

# A single editable value. Scalars cover the int/float/choice/text fields;
# ``bool`` is intentionally excluded (the only bools, ``resume`` and the
# choice-rendered ``layernorm``, are handled as an action / a choice) so the
# union never has to disambiguate ``bool`` from ``int`` under isinstance. The
# ``tuple[int, ...]`` is the per-layer width list edited by a
# :class:`LayersField`; ``None`` is an :class:`OptionalIntField`'s unset state
# (the field tracks its fallback).
type FieldValue = int | float | str | tuple[int, ...] | None


class ConfigSection(enum.StrEnum):
    """The form's field groups, shown as headers in display order."""

    LOOP = "loop shape"
    OPTIM = "optimization"
    EVAL = "evaluation"
    MODEL = "model & seed"
    SETUP = "setup model"
    RUNTIME = "runtime"
    CHECKPOINT = "checkpoints"


# The order sections appear in the form.
SECTION_ORDER: tuple[ConfigSection, ...] = (
    ConfigSection.LOOP,
    ConfigSection.OPTIM,
    ConfigSection.EVAL,
    ConfigSection.MODEL,
    ConfigSection.SETUP,
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

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    attr: str
    label: str
    section: ConfigSection
    help: str
    unit: str = ""
    impact: ChangeImpact = ChangeImpact.NONE
    # Optional subgroup within the section (rendered as an indented header row).
    # None = the field appears directly under the section header, ungrouped.
    group: str | None = None
    # When set, the field is shown and navigable only when this predicate returns
    # True for the current working config. None = always visible.
    visible_when: typing.Callable[[config.TrainConfig], bool] | None = None


class IntField(FieldSpec):
    """An integer field; ``step`` is the +/- nudge increment."""

    step: int = 1


class OptionalIntField(FieldSpec):
    """An integer field whose ``None`` means "match another field" (e.g. the hand
    embed width tracking the card embed width). ``None`` displays as
    ``none_label``; typing a number sets an explicit value and typing ``none``
    (or clearing) resets the tracking default. LEFT/RIGHT nudges step from the
    resolved ``fallback_attr`` value when currently unset."""

    step: int = 1
    none_label: str
    fallback_attr: str


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


class LayersField(FieldSpec):
    """A per-layer width list (a network block's hidden widths).

    Edited by typing comma-separated widths (``256, 128, 64`` sets the sizes);
    LEFT / RIGHT nudges change the *number* of layers — RIGHT appends a layer
    (duplicating the last width), LEFT drops the trailing one — down to
    ``min_len`` (1 for the body blocks, 0 for the heads, which may be empty)."""

    min_len: int = 1


def spec_for(attr: str) -> FieldSpec:
    """The :class:`FieldSpec` for an editable attribute name."""
    return _BY_ATTR[attr]


def editable_attrs(cfg: config.TrainConfig | None = None) -> list[str]:
    """Editable attribute names in form display order (section order, then FIELD_SPECS
    order within each section).

    When ``cfg`` is supplied, fields whose ``visible_when`` predicate returns
    ``False`` are excluded so callers can navigate only the currently-visible
    subset. Omit ``cfg`` to get the full list (e.g. for ``_BY_ATTR`` construction).
    """
    return [
        spec.attr
        for section in SECTION_ORDER
        for spec in FIELD_SPECS
        if spec.section is section
        and (cfg is None or spec.visible_when is None or spec.visible_when(cfg))
    ]


def reset_hidden_fields(cfg: config.TrainConfig) -> config.TrainConfig:
    """Reset any field that is currently hidden (``visible_when`` returns False) to
    its factory default. Called after every config mutation so hidden state never
    lingers when the user re-enables a feature group."""
    defaults = _DEFAULTS
    updates: dict[str, object] = {}
    for spec in FIELD_SPECS:
        if spec.visible_when is not None and not spec.visible_when(cfg):
            default_val = getattr(defaults, spec.attr)
            if getattr(cfg, spec.attr) != default_val:
                updates[spec.attr] = default_val
    if not updates:
        return cfg
    return config.TrainConfig.model_validate({**cfg.model_dump(), **updates})


def read_field(cfg: config.TrainConfig, spec: FieldSpec) -> FieldValue:
    """The current value of ``spec``'s field. The single localized cast bridges
    the dynamic attribute-name access — every editable field is a ``FieldValue``,
    so re-typing the ``getattr`` result here keeps ``Any`` out of every caller."""
    return typing.cast("FieldValue", getattr(cfg, spec.attr))


def format_value(cfg: config.TrainConfig, spec: FieldSpec) -> str:
    """The display string for ``spec``'s current value."""
    value = read_field(cfg, spec)
    if isinstance(spec, OptionalIntField) and value is None:
        return spec.none_label
    if isinstance(spec, FloatField):
        if spec.scientific:
            return f"{value:.0e}"
        return f"{value:.6f}".rstrip("0").rstrip(".") or "0"
    if isinstance(spec, LayersField):
        widths = value if isinstance(value, tuple) else ()
        return ", ".join(str(width) for width in widths) or "none"
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
    success, or ``(cfg, error)`` if it cannot parse or the model rejects it.
    Failure is signalled by the error string — ``None`` is a *valid* parse for
    an :class:`OptionalIntField` (its tracking-default state)."""
    parsed, parse_error = _parse(spec, raw)
    if parse_error is not None:
        return cfg, parse_error
    return _validated_update(cfg, spec, parsed)


def nudge(
    cfg: config.TrainConfig, spec: FieldSpec, direction: int
) -> tuple[config.TrainConfig, str | None]:
    """Apply a LEFT (``direction == -1``) / RIGHT (``+1``) step to ``spec``.

    Numeric fields step by ``spec.step`` and are validated; a choice cycles; a
    layer list adds / removes a layer; a text / path field has no step and is
    returned unchanged.
    """
    if isinstance(spec, ChoiceField):
        return _cycle_choice(cfg, spec, direction), None
    if isinstance(spec, OptionalIntField):
        # Step from the resolved fallback when unset, so the first nudge lands
        # one step away from the value the None currently tracks.
        value = read_field(cfg, spec)
        base = value if isinstance(value, int) else _read_fallback_int(cfg, spec)
        return _validated_update(cfg, spec, base + direction * spec.step)
    if isinstance(spec, LayersField):
        widths = _read_layers(cfg, spec)
        if direction > 0:
            last_width = widths[-1] if widths else _NEW_LAYER_WIDTH
            new_widths = widths + (last_width,)
        elif len(widths) <= spec.min_len:
            return (
                cfg,
                f"{spec.label}: already at the minimum of {spec.min_len} layer(s)",
            )
        else:
            new_widths = widths[:-1]
        return _validated_update(cfg, spec, new_widths)
    if isinstance(spec, IntField):
        return _validated_update(
            cfg, spec, _read_int(cfg, spec) + direction * spec.step
        )
    if isinstance(spec, FloatField):
        stepped = _read_float(cfg, spec) + direction * spec.step
        return _validated_update(cfg, spec, round(stepped, _FLOAT_ROUND))
    return cfg, None


###### PRIVATE #######


def _parse(spec: FieldSpec, raw: str) -> tuple[FieldValue, str | None]:
    """Parse an edit-buffer string into the field's Python type, or an error.

    Success/failure is carried by the error slot (not a ``None`` value): an
    :class:`OptionalIntField` legitimately parses to ``None``."""
    text = raw.strip()
    if isinstance(spec, OptionalIntField):
        if not text or text.lower() == "none":
            return None, None
        try:
            return int(text), None
        except ValueError:
            return None, f"{spec.label}: expects a whole number or 'none'"
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
    if isinstance(spec, LayersField):
        tokens = [
            token for token in text.replace(",", " ").split() if token.lower() != "none"
        ]
        if not tokens:
            return (), None  # empty list (valid only where min_len is 0)
        try:
            return tuple(int(token) for token in tokens), None
        except ValueError:
            return None, f"{spec.label}: expects comma-separated whole numbers"
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


def _read_fallback_int(cfg: config.TrainConfig, spec: OptionalIntField) -> int:
    """The integer value an unset :class:`OptionalIntField` currently tracks."""
    fallback = typing.cast("int | None", getattr(cfg, spec.fallback_attr))
    return fallback if isinstance(fallback, int) else 0


def _read_layers(cfg: config.TrainConfig, spec: FieldSpec) -> tuple[int, ...]:
    value = read_field(cfg, spec)
    return value if isinstance(value, tuple) else ()


# ---------------------------------------------------------------------------
# The full editable surface — kept at the bottom so readers encounter the API
# and class hierarchy first. The derived module-level globals (_BY_ATTR, etc.)
# must immediately follow the table; functions resolve them at call time.

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
    IntField(
        attr="target_iterations",
        label="target iterations",
        section=ConfigSection.LOOP,
        unit="iters",
        step=1000,
        help="Pause at this iteration for a final evaluation + user acknowledgment. "
        "0 = no target. Must be ≤ max_iterations when both are > 0.",
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
        group="cadence",
        unit="iters",
        step=1,
        help="Run a paired eval block every N iterations; 0 disables evaluation "
        "entirely.",
    ),
    IntField(
        attr="eval_games",
        label="eval games",
        section=ConfigSection.EVAL,
        group="cadence",
        unit="games",
        step=8,
        help="Held-out games per eval, played as mirrored pairs to cancel the "
        "first-player edge (an odd value rounds down to the nearest pair).",
    ),
    FloatField(
        attr="eval_ewma_alpha",
        label="eval ewma α",
        section=ConfigSection.EVAL,
        group="cadence",
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Smoothing for the eval win-rate / margin trend (higher = more "
        "responsive). Re-smooths the restored history on resume.",
    ),
    IntField(
        attr="target_eval_games",
        label="target eval games",
        section=ConfigSection.EVAL,
        group="cadence",
        unit="games",
        step=100,
        help="Self-play games (model fixed, greedy) run at the target milestone. "
        "0 = auto: 10 × eval games.",
    ),
    FloatField(
        attr="opponent_reset_win_rate",
        label="opp. advance @",
        section=ConfigSection.EVAL,
        group="opponent",
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Freeze the current policy as the new opponent once smoothed win-rate "
        "clears this (0 disables). Lowering it below the current win-rate triggers "
        "an immediate advance + win-rate reset.",
    ),
    IntField(
        attr="opponent_max_iterations",
        label="opp. max iters",
        section=ConfigSection.EVAL,
        group="opponent",
        step=50,
        impact=ChangeImpact.REGIME,
        help="Force-advance the opponent after this many iterations even if the "
        "win-rate threshold is not yet met (0 disables). Only applies during "
        "self-play; the random-phase bootstrap uses its own graduation logic.",
    ),
    FloatField(
        attr="produce_ewma_alpha",
        label="produce ewma α",
        section=ConfigSection.EVAL,
        group="opponent",
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Smoothing for the PRODUCING band's score / margin readouts.",
    ),
    ChoiceField(
        attr="initial_vs_random",
        label="bootstrap vs random",
        section=ConfigSection.EVAL,
        group="bootstrap",
        choices=["True", "False"],
        help="Fresh runs only: start by collecting against the random agent "
        "(net at seat 0, eval paused) before switching to self-play. A resumed "
        "run keeps the phase stored in its checkpoint.",
    ),
    FloatField(
        attr="random_phase_win_rate",
        label="graduate @",
        section=ConfigSection.EVAL,
        group="bootstrap",
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Smoothed collection win-rate (vs random) at which the bootstrap "
        "phase freezes self·gen1 and switches to self-play. Lowering it below "
        "the current win-rate graduates immediately.",
    ),
    LayersField(
        attr="trunk_layers",
        label="trunk layers",
        section=ConfigSection.MODEL,
        group="encoders",
        unit="units",
        impact=ChangeImpact.FRESH,
        help="State-trunk hidden widths (input→output), e.g. 256,128. Type to set "
        "the sizes; ←/→ adds or removes a layer. Its last width is M, the trunk "
        "embedding fed to the value head and concatenated for scoring. Fresh run.",
    ),
    LayersField(
        attr="choice_layers",
        label="choice layers",
        section=ConfigSection.MODEL,
        group="encoders",
        unit="units",
        impact=ChangeImpact.FRESH,
        help="Per-choice encoder widths (input→output). Its last width is N, the "
        "choice embedding concatenated with M before the scorer heads. Independent "
        "of trunk layers. Fresh run.",
    ),
    ChoiceField(
        attr="head_layers_mode",
        label="scorer head mode",
        section=ConfigSection.MODEL,
        group="scorer heads",
        choices=["uniform", "per_family"],
        impact=ChangeImpact.FRESH,
        help="'uniform' — all decision families share one scorer head shape. "
        "'per_family' — configure each family's hidden widths independently. "
        "Fresh run.",
    ),
    LayersField(
        attr="head_layers",
        label="scorer head layers",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "uniform",
        help="Per-family scorer hidden widths between the M+N concat and the final "
        "logit. Empty (←  to 0 layers) = a direct (M+N)→1 readout. Fresh run.",
    ),
    LayersField(
        attr="head_layers_main_action",
        label="head: main action",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the main-action family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_draw_bird",
        label="head: draw bird",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the draw-bird family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_discard_bird",
        label="head: discard bird",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the discard-bird family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_gain_food",
        label="head: gain food",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the gain-food family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_spend_food",
        label="head: spend food",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the spend-food family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_lay_egg",
        label="head: lay egg",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the lay-egg family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_pay_egg",
        label="head: pay egg",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the pay-egg family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_skip_optional",
        label="head: skip optional",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the skip-optional family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_choose_bonus",
        label="head: choose bonus",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the choose-bonus family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_misc_rare",
        label="head: misc rare",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the misc-rare family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_play_bird",
        label="head: play bird",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the play-bird family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_reset_birdfeeder",
        label="head: reset birdfeeder",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the reset-birdfeeder family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_setup",
        label="head: setup",
        section=ConfigSection.MODEL,
        group="scorer heads",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.head_layers_mode == "per_family",
        help="Scorer head hidden widths for the setup family. Fresh run.",
    ),
    LayersField(
        attr="value_layers",
        label="value head layers",
        section=ConfigSection.MODEL,
        group="value head",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        help="Value-head hidden widths before the scalar output. Empty = a direct "
        "M→1 readout (the default). Fresh run.",
    ),
    ChoiceField(
        attr="activation",
        label="activation",
        section=ConfigSection.MODEL,
        group="regularization",
        choices=[name.value for name in architecture.ActivationName],
        impact=ChangeImpact.REGIME,
        help="Activation for every MLP block. Resumable (it doesn't change tensor "
        "shapes), but it reinterprets an in-progress run.",
    ),
    FloatField(
        attr="dropout",
        label="dropout",
        section=ConfigSection.MODEL,
        group="regularization",
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Dropout after each activation, active only in the learner's update "
        "(collection / eval run eval-mode). 0 disables it. Resumable.",
    ),
    ChoiceField(
        attr="layernorm",
        label="layernorm",
        section=ConfigSection.MODEL,
        group="regularization",
        choices=["True", "False"],
        impact=ChangeImpact.FRESH,
        help="Apply LayerNorm in the trunk / choice-encoder body blocks. Adds "
        "parameters, so toggling it requires a fresh run.",
    ),
    IntField(
        attr="card_embed_dim",
        label="card embed dim",
        section=ConfigSection.MODEL,
        group="card embeddings",
        unit="units",
        step=16,
        impact=ChangeImpact.FRESH,
        help="Width of the shared per-card vector (reused for every board / tray / "
        "hand / choice card slot) — the card encoder's output width. Changes the "
        "architecture — fresh run.",
    ),
    LayersField(
        attr="card_encoder_layers",
        label="card encoder layers",
        section=ConfigSection.MODEL,
        group="card embeddings",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        help="Card encoder MLP hidden widths (input→output). It maps each card's "
        "[static attributes ⊕ identity one-hot] to its card-embed-dim vector. Empty "
        "(←  to 0 layers) = a single linear projection; a stack makes it nonlinear. "
        "Fresh run.",
    ),
    ChoiceField(
        attr="encoder_final_activation",
        label="encoder final act",
        section=ConfigSection.MODEL,
        group="card embeddings",
        choices=["True", "False"],
        impact=ChangeImpact.REGIME,
        help="When on (the default for new runs), the card, hand, and choice encoders "
        "apply a final activation after their last layer, matching the trunk. Old "
        "checkpoints carry False and keep their original behaviour on load. Regime.",
    ),
    ChoiceField(
        attr="use_distinct_hand_model",
        label="distinct hand MLP",
        section=ConfigSection.MODEL,
        group="hand model",
        choices=["True", "False"],
        impact=ChangeImpact.FRESH,
        help="When on (the default), a dedicated MLP encodes the full hand (180-dim "
        "multi-hot ⊕ 10-dim summary) instead of mean-pooling through the shared "
        "card encoder. The 10-dim hand summary is redirected from the trunk's "
        "continuous input into this encoder. Fresh run.",
    ),
    LayersField(
        attr="hand_encoder_layers",
        label="hand encoder layers",
        section=ConfigSection.MODEL,
        group="hand model",
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.use_distinct_hand_model,
        help="Hand encoder MLP hidden widths (input→output). Active only when "
        "'distinct hand MLP' is on. Output width is the hand embed dim below. "
        "Empty = a single linear projection. Fresh run.",
    ),
    OptionalIntField(
        attr="hand_embed_dim",
        label="hand embed dim",
        section=ConfigSection.MODEL,
        group="hand model",
        unit="units",
        step=16,
        none_label="= card embed",
        fallback_attr="card_embed_dim",
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.use_distinct_hand_model,
        help="Output width N of the hand encoder — the multi-card *set* embedding "
        "(the hand, and every other card set embedded through it). Type 'none' to "
        "track card embed dim (M). Fresh run.",
    ),
    ChoiceField(
        attr="tray_set_embedding",
        label="tray set embedding",
        section=ConfigSection.MODEL,
        group="hand model",
        choices=["True", "False"],
        impact=ChangeImpact.FRESH,
        visible_when=lambda cfg: cfg.use_distinct_hand_model,
        help="Feed the trunk one hand-encoder embedding of the face-up tray *set* "
        "beside the three per-slot card lookups (3·M + N tray dims). Requires the "
        "distinct hand MLP; on by default alongside it. Fresh run.",
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
    ChoiceField(
        attr="use_setup_model",
        label="use setup model",
        section=ConfigSection.SETUP,
        choices=["True", "False"],
        impact=ChangeImpact.REGIME,
        help="Train the start-of-game keep with a separate value-regression net "
        "instead of the in-game policy. Off = unchanged behaviour. Has its own "
        "checkpoint, so toggling it never invalidates the main net's weights.",
    ),
    ChoiceField(
        attr="split_setup_bonus",
        label="split setup bonus",
        section=ConfigSection.SETUP,
        choices=["True", "False"],
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Only with the setup model on: the setup net picks cards/food while "
        "the bonus card is deferred to the in-game CHOOSE_BONUS head (asked over a "
        "round-1 opening). Feeds that head more data; shape-preserving, so it never "
        "invalidates either net's weights.",
    ),
    ChoiceField(
        attr="split_setup_food",
        label="split setup food",
        section=ConfigSection.SETUP,
        choices=["True", "False"],
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Only with the setup model on: the setup net picks cards only while "
        "food is deferred to sequential in-game GAIN_FOOD/SPEND_FOOD decisions "
        "(2/1/0 gains for 3/4/5 birds kept; 1/2 spends for 1/2 birds kept). "
        "Feeds those heads more data; food block zeros out but feature dim is "
        "unchanged, so it never invalidates either net's weights.",
    ),
    LayersField(
        attr="setup_hidden_layers",
        label="setup layers",
        section=ConfigSection.SETUP,
        group="network",
        unit="units",
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Setup-net MLP hidden widths (input→output). Type to set sizes; ←/→ "
        "adds/removes a layer. Changing it restarts ONLY the setup net (the main "
        "run resumes); the setup net then refits from its recorded samples.",
    ),
    ChoiceField(
        attr="setup_activation",
        label="setup activation",
        section=ConfigSection.SETUP,
        group="network",
        choices=[name.value for name in architecture.ActivationName],
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Activation for the setup net's MLP blocks (resumable for the main "
        "run; reinterprets the setup net).",
    ),
    FloatField(
        attr="setup_dropout",
        label="setup dropout",
        section=ConfigSection.SETUP,
        group="network",
        step=0.05,
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Dropout after each setup-net activation (training only). 0 disables.",
    ),
    FloatField(
        attr="setup_lr",
        label="setup lr",
        section=ConfigSection.SETUP,
        group="network",
        step=1e-4,
        scientific=True,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Adam step size for the setup net's MSE updates (its own optimizer).",
    ),
    FloatField(
        attr="setup_policy_temperature",
        label="setup temperature",
        section=ConfigSection.SETUP,
        group="network",
        step=0.05,
        visible_when=lambda cfg: cfg.use_setup_model and not cfg.setup_policy_greedy,
        help="Softmax temperature over the 504 candidates' predicted margins when "
        "sampling a setup during collection (eval takes the argmax). Ignored when "
        "setup greedy is True.",
    ),
    ChoiceField(
        attr="setup_policy_greedy",
        label="setup greedy",
        section=ConfigSection.SETUP,
        group="network",
        choices=["True", "False"],
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="When True, collection takes the hard argmax over predicted margins "
        "instead of softmax sampling — trains the in-game model on the best "
        "setup the setup net knows. Eval always uses argmax regardless.",
    ),
    IntField(
        attr="setup_record_start_iter",
        label="record start @",
        section=ConfigSection.SETUP,
        group="schedule",
        unit="iters",
        step=100,
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Iteration at which to start recording (setup, margin) samples — "
        "below it setups are random and unrecorded (skips early bad data).",
    ),
    IntField(
        attr="setup_train_iter",
        label="train @",
        section=ConfigSection.SETUP,
        group="schedule",
        unit="iters",
        step=100,
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Iteration at which the setup net is fit once offline on the recorded "
        "window and then drives selection + trains on-policy. Must exceed record "
        "start.",
    ),
    IntField(
        attr="setup_hand_combos",
        label="hand combos",
        section=ConfigSection.SETUP,
        group="sample generation",
        step=1,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Random generator: joint (P0,P1) keep-combos sampled per shared-deal "
        "batch.",
    ),
    IntField(
        attr="setup_food_sets",
        label="food sets",
        section=ConfigSection.SETUP,
        group="sample generation",
        step=1,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Random generator: food keeps sampled per kept hand (softmax-biased "
        "toward food that pays for more hand/tray birds).",
    ),
    IntField(
        attr="setup_tuples_per_batch",
        label="tuples / batch",
        section=ConfigSection.SETUP,
        group="sample generation",
        step=1,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Random generator: joint setups sampled per batch = games sharing one "
        "deal (should divide games/iter).",
    ),
    IntField(
        attr="setup_offline_epochs",
        label="offline epochs",
        section=ConfigSection.SETUP,
        group="sample generation",
        step=5,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Epochs over the recorded window in the one-time offline fit.",
    ),
    IntField(
        attr="setup_offline_batch_size",
        label="setup batch size",
        section=ConfigSection.SETUP,
        group="sample generation",
        unit="samples",
        step=64,
        visible_when=lambda cfg: cfg.use_setup_model,
        help="Minibatch size for the setup net's offline fit and on-policy steps.",
    ),
]

_BY_ATTR: dict[str, FieldSpec] = {spec.attr: spec for spec in FIELD_SPECS}
_DEFAULTS = config.TrainConfig()
_FLOAT_ROUND = 6  # decimal places a nudged float is rounded to (kills FP crud)
# Width seeded when a RIGHT-nudge adds the first layer to an empty list.
_NEW_LAYER_WIDTH = 128
