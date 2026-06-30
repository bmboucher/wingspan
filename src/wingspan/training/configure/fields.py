"""The editable ``RunConfig`` fields: per-kind display metadata and the pure
read / format / commit / nudge helpers the configurator builds on.

There is one :class:`FieldSpec` subclass per *kind* of value (mirroring the
``decisions.Choice`` hierarchy) so consumers branch via ``isinstance`` and read
typed, always-present attributes instead of ``None``-checking a flat spec. The
bounds a value must satisfy are deliberately NOT duplicated here — they live as
declarative ``Annotated[..., Field(...)]`` constraints on ``RunConfig`` itself,
and every edit is committed through :func:`commit` / :func:`nudge`, which route
the candidate value through ``RunConfig.model_validate`` so an out-of-range
value is rejected by the model. This module carries only presentation metadata,
the nudge step, and the parse layer.

``resume`` is intentionally absent: it is a launch-time decision (Start vs New
run), not a hyperparameter, so the controller sets it, not the form. The
architecture descriptors (``state_dim`` / ``choice_dim`` / ``family_order``) are
derived and never editable.

Fields are organised by a ``group_path: tuple[str, ...]`` of up to three levels
(top-level section, optional group, optional subgroup). Display order is the
order specs appear in ``FIELD_SPECS``; the first occurrence of each path prefix
fixes that header's position.
"""

from __future__ import annotations

import enum
import pathlib
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
    # Display hierarchy: (section,) or (section, group) or (section, group, subgroup).
    # Ordering is fixed by the first occurrence of each prefix in FIELD_SPECS.
    group_path: tuple[str, ...]
    help: str
    unit: str = ""
    impact: ChangeImpact = ChangeImpact.NONE
    # When set, the field is shown and navigable only when this predicate returns
    # True for the current working config. None = always visible.
    visible_when: typing.Callable[[config.RunConfig], bool] | None = None


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


class OptionalFloatField(FieldSpec):
    """A float field whose ``None`` means "inherit the global setting".

    ``None`` displays as ``none_label`` (default: "inherit"). Typing ``none``
    (or clearing) resets to inherited. LEFT/RIGHT nudges step from the resolved
    value of ``fallback_attr`` when currently unset."""

    step: float
    none_label: str = "inherit"
    fallback_attr: str


class ChoiceField(FieldSpec):
    """A field cycled through a fixed set of string options."""

    choices: list[str]


class OptionalChoiceField(FieldSpec):
    """A choice field whose ``None`` means "inherit the global setting".

    ``None`` displays as ``none_label`` (default: "inherit"). The nudge cycle
    is: None → choices[0] → choices[1] → … → None. Typing ``none`` (or clearing)
    resets to inherited; typing a valid choice name sets it explicitly."""

    choices: list[str]
    none_label: str = "inherit"


class OptionalActivationField(OptionalChoiceField):
    """An activation choice field where typing ``none`` selects
    ``ActivationName.NONE`` (drop the layer) rather than reverting to inherit.
    Clear/empty still resets to Python ``None`` (inherit the global setting)."""


class TextField(FieldSpec):
    """A free-text field (e.g. the run name)."""


class PathField(FieldSpec):
    """A filesystem-path field (e.g. the checkpoint directory)."""


class OptionalPathField(FieldSpec):
    """A nullable filesystem-path field. ``None`` displays as ``none_label``
    (e.g. "random agent"); typing a path string sets an explicit value and
    typing ``none`` (or clearing the buffer) resets to ``None``."""

    none_label: str = "none"


class BootstrapField(FieldSpec):
    """The bootstrap-opponent selector.

    Value is a string: ``"none"`` (no bootstrap phase), ``"random"`` (built-in
    random agent), or an absolute path to a ``.pt.gz`` checkpoint. Left/Right
    cycles through the fixed options plus archived checkpoints (managed by the
    controller, which has access to the live archive list); Enter opens free-text
    edit mode for a custom path."""


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


def editable_attrs(cfg: config.RunConfig | None = None) -> list[str]:
    """Editable attribute names in form display order (FIELD_SPECS order).

    When ``cfg`` is supplied, fields whose ``visible_when`` predicate returns
    ``False`` are excluded so callers can navigate only the currently-visible
    subset. Omit ``cfg`` to get the full list (e.g. for ``_BY_ATTR`` construction).
    """
    return [
        spec.attr
        for spec in FIELD_SPECS
        if cfg is None or spec.visible_when is None or spec.visible_when(cfg)
    ]


def reset_hidden_fields(cfg: config.RunConfig) -> config.RunConfig:
    """Reset any field that is currently hidden (``visible_when`` returns False) to
    its factory default. Called after every config mutation so hidden state never
    lingers when the user re-enables a feature group."""
    defaults = _DEFAULTS
    has_changes = False
    for spec in FIELD_SPECS:
        if spec.visible_when is not None and not spec.visible_when(cfg):
            if read_field(cfg, spec) != read_field(defaults, spec):
                has_changes = True
                break
    if not has_changes:
        return cfg
    # Apply all hidden-field resets by rebuilding through the nested path mapping.
    data = cfg.model_dump()
    for spec in FIELD_SPECS:
        if spec.visible_when is not None and not spec.visible_when(cfg):
            default_val = read_field(defaults, spec)
            if read_field(cfg, spec) != default_val:
                _inject_nested(data, spec.attr, default_val)
    return config.RunConfig.model_validate(data)


def read_field(cfg: config.RunConfig, spec: FieldSpec) -> FieldValue:
    """The current value of ``spec``'s field, navigating the nested section path.

    The single localized cast bridges the dynamic attribute-name access —
    every editable field is a ``FieldValue``, so re-typing the result here
    keeps ``Any`` out of every caller."""
    return typing.cast("FieldValue", _read_nested(cfg, spec.attr))


def format_value(cfg: config.RunConfig, spec: FieldSpec) -> str:
    """The display string for ``spec``'s current value."""
    value = read_field(cfg, spec)
    if isinstance(spec, OptionalIntField) and value is None:
        return spec.none_label
    if isinstance(spec, OptionalPathField) and value is None:
        return spec.none_label
    if isinstance(spec, OptionalChoiceField) and value is None:
        return spec.none_label
    if isinstance(spec, OptionalFloatField) and value is None:
        return spec.none_label
    if isinstance(spec, BootstrapField):
        # "none" and "random" pass through; for paths show just the parent
        # directory name (the archive label) so it fits in the value column.
        text = str(value)
        if text in ("none", "random"):
            return text
        parts = pathlib.PurePosixPath(text.replace("\\", "/")).parts
        # Show "…/<archive-label>/last.pt" — drop the checkpoint_dir prefix.
        return "/".join(parts[-2:]) if len(parts) >= 2 else text
    if isinstance(spec, (FloatField, OptionalFloatField)):
        if isinstance(spec, FloatField) and spec.scientific:
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
    working: config.RunConfig, saved: config.RunConfig | None, spec: FieldSpec
) -> bool:
    """Whether ``spec`` differs from the saved run's value (False if no saved
    run) — drives the changed-field marker."""
    if saved is None:
        return False
    return read_field(working, spec) != read_field(saved, spec)


def commit(
    cfg: config.RunConfig, spec: FieldSpec, raw: str
) -> tuple[config.RunConfig, str | None]:
    """Parse ``raw`` per ``spec``'s kind and return ``(new_cfg, None)`` on
    success, or ``(cfg, error)`` if it cannot parse or the model rejects it.
    Failure is signalled by the error string — ``None`` is a *valid* parse for
    an :class:`OptionalIntField` (its tracking-default state)."""
    parsed, parse_error = _parse(spec, raw)
    if parse_error is not None:
        return cfg, parse_error
    return _validated_update(cfg, spec, parsed)


def nudge(
    cfg: config.RunConfig, spec: FieldSpec, direction: int
) -> tuple[config.RunConfig, str | None]:
    """Apply a LEFT (``direction == -1``) / RIGHT (``+1``) step to ``spec``.

    Numeric fields step by ``spec.step`` and are validated; a choice cycles; a
    layer list adds / removes a layer; a text / path field has no step and is
    returned unchanged.
    """
    if isinstance(spec, OptionalChoiceField):
        return _cycle_optional_choice(cfg, spec, direction), None
    if isinstance(spec, ChoiceField):
        return _cycle_choice(cfg, spec, direction), None
    if isinstance(spec, OptionalIntField):
        # Step from the resolved fallback when unset, so the first nudge lands
        # one step away from the value the None currently tracks.
        value = read_field(cfg, spec)
        base = value if isinstance(value, int) else _read_fallback_int(cfg, spec)
        return _validated_update(cfg, spec, base + direction * spec.step)
    if isinstance(spec, OptionalFloatField):
        value = read_field(cfg, spec)
        base = (
            float(value)
            if isinstance(value, (int, float))
            else _read_fallback_float(cfg, spec)
        )
        stepped = base + direction * spec.step
        return _validated_update(cfg, spec, round(stepped, _FLOAT_ROUND))
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
    if isinstance(spec, OptionalPathField):
        if not text or text.lower() == "none":
            return None, None
        return text, None
    if isinstance(spec, OptionalActivationField):
        # Empty/clear → inherit (Python None); "none" → ActivationName.NONE.
        if not text:
            return None, None
        if text in spec.choices:
            return text, None
        return (
            None,
            f"{spec.label}: choose one of {', '.join(spec.choices)} or clear to inherit",
        )
    if isinstance(spec, OptionalChoiceField):
        if not text or text.lower() == "none":
            return None, None
        if text in spec.choices:
            return text, None
        return None, f"{spec.label}: choose one of {', '.join(spec.choices)} or 'none'"
    if isinstance(spec, OptionalFloatField):
        if not text or text.lower() == "none":
            return None, None
        try:
            return float(text), None
        except ValueError:
            return None, f"{spec.label}: expects a number or 'none'"
    if isinstance(spec, BootstrapField):
        if not text:
            return None, f"{spec.label}: cannot be empty (use 'none' or 'random')"
        return text, None
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
    cfg: config.RunConfig, spec: FieldSpec, value: FieldValue
) -> tuple[config.RunConfig, str | None]:
    """Apply ``value`` to ``spec``'s field through ``model_validate`` so the
    model's declarative bounds reject anything out of range. Injects into the
    correct nested section dict before validation."""
    data = cfg.model_dump()
    _inject_nested(data, spec.attr, value)
    try:
        return config.RunConfig.model_validate(data), None
    except pydantic.ValidationError as error:
        return cfg, _friendly_error(spec, error)


def _friendly_error(spec: FieldSpec, error: pydantic.ValidationError) -> str:
    """Turn a pydantic ValidationError into a one-line field-scoped message."""
    details = error.errors()
    message = details[0]["msg"] if details else "invalid value"
    return f"{spec.label}: {message}"


def _cycle_choice(
    cfg: config.RunConfig, spec: ChoiceField, direction: int
) -> config.RunConfig:
    current = format_value(cfg, spec)
    index = spec.choices.index(current) if current in spec.choices else 0
    chosen = spec.choices[(index + direction) % len(spec.choices)]
    updated, _ = _validated_update(cfg, spec, chosen)
    return updated


def _cycle_optional_choice(
    cfg: config.RunConfig, spec: OptionalChoiceField, direction: int
) -> config.RunConfig:
    # Cycle: None → choices[0] → … → choices[-1] → None. None is index 0.
    all_opts: list[str | None] = [None] + spec.choices
    current = read_field(cfg, spec)
    index = 0
    for idx, opt in enumerate(all_opts):
        if opt == current:
            index = idx
            break
    chosen = all_opts[(index + direction) % len(all_opts)]
    updated, _ = _validated_update(cfg, spec, chosen)
    return updated


def _read_int(cfg: config.RunConfig, spec: FieldSpec) -> int:
    value = read_field(cfg, spec)
    return value if isinstance(value, int) else 0


def _read_float(cfg: config.RunConfig, spec: FieldSpec) -> float:
    value = read_field(cfg, spec)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _read_fallback_int(cfg: config.RunConfig, spec: OptionalIntField) -> int:
    """The integer value an unset :class:`OptionalIntField` currently tracks."""
    fallback = typing.cast("FieldValue", _read_nested(cfg, spec.fallback_attr))
    return fallback if isinstance(fallback, int) else 0


def _read_fallback_float(cfg: config.RunConfig, spec: OptionalFloatField) -> float:
    """The float value an unset :class:`OptionalFloatField` currently tracks."""
    fallback = typing.cast("FieldValue", _read_nested(cfg, spec.fallback_attr))
    return float(fallback) if isinstance(fallback, (int, float)) else 0.0


def _read_layers(cfg: config.RunConfig, spec: FieldSpec) -> tuple[int, ...]:
    value = read_field(cfg, spec)
    return value if isinstance(value, tuple) else ()


# ---------------------------------------------------------------------------
# Nested-path routing for each editable attr.
#
# Maps each ``FieldSpec.attr`` to its dotted location inside ``RunConfig``.
# Two-element tuples are ``(section, field)``; three-element tuples are
# ``(section, subsection, field)`` (for ``training.setup.*`` fields).
# The ``_read_nested`` / ``_inject_nested`` helpers use this table so the
# rest of the module never has to know which section a field lives in.

_ATTR_PATH: dict[str, tuple[str, ...]] = {
    # run section
    "games_per_iter": ("run", "games_per_iter"),
    "max_iterations": ("run", "max_iterations"),
    "target_iterations": ("run", "target_iterations"),
    "target_eval_games": ("run", "target_eval_games"),
    "eval_every": ("run", "eval_every"),
    "eval_games": ("run", "eval_games"),
    "checkpoint_dir": ("run", "checkpoint_dir"),
    "run_name": ("run", "run_name"),
    "resume": ("run", "resume"),
    "history_len": ("run", "history_len"),
    # training section
    "lr": ("training", "lr"),
    "value_coef": ("training", "value_coef"),
    "entropy_coef": ("training", "entropy_coef"),
    "grad_clip": ("training", "grad_clip"),
    "score_norm": ("training", "score_norm"),
    "reward_mode": ("training", "reward_mode"),
    "reward_basis": ("training", "reward_basis"),
    "reward_discount": ("training", "reward_discount"),
    "end_game_bonus": ("training", "end_game_bonus"),
    "policy_loss": ("training", "policy_loss"),
    "ppo_clip_eps": ("training", "ppo_clip_eps"),
    "ppo_reuse_epochs": ("training", "ppo_reuse_epochs"),
    "gae_lambda": ("training", "gae_lambda"),
    "update_minibatch_steps": ("training", "update_minibatch_steps"),
    # training.setup section
    "setup_lr": ("training", "setup", "lr"),
    "setup_policy_temperature": ("training", "setup", "policy_temperature"),
    "setup_policy_greedy": ("training", "setup", "policy_greedy"),
    "setup_hand_combos": ("training", "setup", "hand_combos"),
    "setup_food_sets": ("training", "setup", "food_sets"),
    "setup_pg_coef": ("training", "setup", "pg_coef"),
    "setup_value_coef": ("training", "setup", "value_coef"),
    "setup_entropy_coef": ("training", "setup", "entropy_coef"),
    # opponent section
    "bootstrap_opponent": ("opponent", "bootstrap_opponent"),
    "random_phase_win_rate": ("opponent", "random_phase_win_rate"),
    "opponent_reset_win_rate": ("opponent", "opponent_reset_win_rate"),
    "opponent_max_iterations": ("opponent", "opponent_max_iterations"),
    "eval_ewma_alpha": ("opponent", "eval_ewma_alpha"),
    # misc section
    "seed": ("misc", "seed"),
    "device": ("misc", "device"),
    "produce_ewma_alpha": ("misc", "produce_ewma_alpha"),
    # architecture section (top-level toggles)
    "use_setup_model": ("architecture", "use_setup_model"),
    "split_setup_bonus": ("architecture", "split_setup_bonus"),
    "split_setup_food": ("architecture", "split_setup_food"),
    # architecture.main section
    "trunk_layers": ("architecture", "main", "trunk_layers"),
    "choice_layers": ("architecture", "main", "choice_layers"),
    "head_layers": ("architecture", "main", "head_layers"),
    "value_layers": ("architecture", "main", "value_layers"),
    "head_layers_mode": ("architecture", "main", "head_layers_mode"),
    "head_layers_main_action": ("architecture", "main", "head_layers_main_action"),
    "head_layers_draw_bird": ("architecture", "main", "head_layers_draw_bird"),
    "head_layers_discard_bird": ("architecture", "main", "head_layers_discard_bird"),
    "head_layers_gain_food": ("architecture", "main", "head_layers_gain_food"),
    "head_layers_spend_food": ("architecture", "main", "head_layers_spend_food"),
    "head_layers_lay_egg": ("architecture", "main", "head_layers_lay_egg"),
    "head_layers_pay_egg": ("architecture", "main", "head_layers_pay_egg"),
    "head_layers_skip_optional": ("architecture", "main", "head_layers_skip_optional"),
    "head_layers_choose_bonus": ("architecture", "main", "head_layers_choose_bonus"),
    "head_layers_misc_rare": ("architecture", "main", "head_layers_misc_rare"),
    "head_layers_play_bird": ("architecture", "main", "head_layers_play_bird"),
    "head_layers_reset_birdfeeder": (
        "architecture",
        "main",
        "head_layers_reset_birdfeeder",
    ),
    "head_layers_setup": ("architecture", "main", "head_layers_setup"),
    "between_activation": ("architecture", "main", "between_activation"),
    "final_activation": ("architecture", "main", "final_activation"),
    "dropout": ("architecture", "main", "dropout"),
    "layernorm": ("architecture", "main", "layernorm"),
    "card_embed_dim": ("architecture", "main", "card_embed_dim"),
    "card_encoder_layers": ("architecture", "main", "card_encoder_layers"),
    "use_distinct_hand_model": ("architecture", "main", "use_distinct_hand_model"),
    "hand_encoder_layers": ("architecture", "main", "hand_encoder_layers"),
    "hand_embed_dim": ("architecture", "main", "hand_embed_dim"),
    "hand_pooling": ("architecture", "main", "hand_pooling"),
    "use_board_attention": ("architecture", "main", "use_board_attention"),
    # architecture.main per-block overrides
    "card_between_activation": ("architecture", "main", "card_between_activation"),
    "card_final_activation": ("architecture", "main", "card_final_activation"),
    "card_dropout": ("architecture", "main", "card_dropout"),
    "card_layernorm": ("architecture", "main", "card_layernorm"),
    "trunk_between_activation": ("architecture", "main", "trunk_between_activation"),
    "trunk_final_activation": ("architecture", "main", "trunk_final_activation"),
    "trunk_dropout": ("architecture", "main", "trunk_dropout"),
    "trunk_layernorm": ("architecture", "main", "trunk_layernorm"),
    "choice_between_activation": ("architecture", "main", "choice_between_activation"),
    "choice_final_activation": ("architecture", "main", "choice_final_activation"),
    "choice_dropout": ("architecture", "main", "choice_dropout"),
    "choice_layernorm": ("architecture", "main", "choice_layernorm"),
    "value_between_activation": ("architecture", "main", "value_between_activation"),
    "value_final_activation": ("architecture", "main", "value_final_activation"),
    "head_between_activation": ("architecture", "main", "head_between_activation"),
    "head_final_activation": ("architecture", "main", "head_final_activation"),
    # architecture.setup section
    "setup_trunk_layers": ("architecture", "setup", "trunk_layers"),
    "setup_hidden_layers": ("architecture", "setup", "hidden_layers"),
    "setup_between_activation": ("architecture", "setup", "between_activation"),
    "setup_final_activation": ("architecture", "setup", "final_activation"),
    "setup_dropout": ("architecture", "setup", "dropout"),
    # dagger section
    "dagger_expert_checkpoint": ("dagger", "expert_checkpoint"),
    "clone_iters": ("dagger", "clone_iters"),
    # engine section
    "combine_gain_food": ("engine", "combine_gain_food"),
}


# Attrs whose UI value is the logical inverse of the stored config field.
# "includes bonus" = True in the UI means split_setup_bonus = False in config
# (i.e. the setup net handles bonus — not deferred to the in-game head).
_INVERTED_ATTRS: frozenset[str] = frozenset({"split_setup_bonus", "split_setup_food"})


def _read_nested(cfg: config.RunConfig, attr: str) -> object:
    """Read the value of ``attr`` from ``cfg`` by navigating the nested path in
    ``_ATTR_PATH``. Raises ``KeyError`` for unknown attrs (a programming error).
    Inverted attrs (see ``_INVERTED_ATTRS``) are negated before returning."""
    path = _ATTR_PATH[attr]
    obj: object = cfg
    for key in path:
        obj = getattr(obj, key)
    if attr in _INVERTED_ATTRS:
        return not obj
    return obj


def _inject_nested(data: dict[str, object], attr: str, value: object) -> None:
    """Write ``value`` into the nested ``data`` dict at the location described by
    ``_ATTR_PATH[attr]``. ``data`` is the result of ``RunConfig.model_dump()``
    and is mutated in place. Inverted attrs (see ``_INVERTED_ATTRS``) are
    negated before writing."""
    if attr in _INVERTED_ATTRS:
        bool_val = value if isinstance(value, bool) else str(value) == "True"
        value = not bool_val
    path = _ATTR_PATH[attr]
    node: dict[str, object] = data  # type: ignore[assignment]
    for key in path[:-1]:
        node = node[key]  # type: ignore[assignment]
    node[path[-1]] = value


# ---------------------------------------------------------------------------
# Shared visibility predicates (named so lambdas in FIELD_SPECS stay short)


def _use_setup(cfg: config.RunConfig) -> bool:
    return cfg.architecture.use_setup_model


def _bootstrap_is_checkpoint(cfg: config.RunConfig) -> bool:
    return cfg.bootstrap_opponent_checkpoint is not None


def _uniform_heads(cfg: config.RunConfig) -> bool:
    return cfg.architecture.main.head_layers_mode == "uniform"


def _per_family_heads(cfg: config.RunConfig) -> bool:
    return cfg.architecture.main.head_layers_mode == "per_family"


# ---------------------------------------------------------------------------
# The full editable surface — kept at the bottom so readers encounter the API
# and class hierarchy first. The derived module-level globals (_BY_ATTR, etc.)
# must immediately follow the table; functions resolve them at call time.
#
# Hierarchy: group_path=("SECTION",) for top-level fields;
#            group_path=("SECTION", "GROUP") for grouped fields;
#            group_path=("SECTION", "GROUP", "SUBGROUP") for three-level fields.

_ACT_CHOICES = [name.value for name in architecture.ActivationName]

FIELD_SPECS: list[FieldSpec] = [
    # -----------------------------------------------------------------------
    # RUN SETTINGS
    # -----------------------------------------------------------------------
    IntField(
        attr="games_per_iter",
        label="games / iter",
        group_path=("RUN SETTINGS",),
        unit="games",
        step=8,
        help="Self-play games collected and learned from per collect→update cycle. "
        "More games = a less noisy gradient but a slower iteration.",
    ),
    IntField(
        attr="max_iterations",
        label="max iterations",
        group_path=("RUN SETTINGS",),
        unit="iters",
        step=1,
        help="Additional iterations to run THIS session (0 = until you press Stop). "
        "On a resumed run this is counted from the resume point, not from zero.",
    ),
    IntField(
        attr="target_iterations",
        label="target iterations",
        group_path=("RUN SETTINGS",),
        unit="iters",
        step=1000,
        help="Pause at this iteration for a final evaluation + user acknowledgment. "
        "0 = no target. Must be ≤ max_iterations when both are > 0.",
    ),
    IntField(
        attr="seed",
        label="seed",
        group_path=("RUN SETTINGS",),
        step=1,
        impact=ChangeImpact.REGIME,
        help="RNG seed for Python / NumPy / torch and the per-game deal. Changing "
        "it on resume re-randomizes the game stream (reproducibility resets).",
    ),
    ChoiceField(
        attr="device",
        label="device",
        group_path=("RUN SETTINGS",),
        choices=["cpu", "cuda"],
        impact=ChangeImpact.REGIME,
        help="Compute device. Self-play collection is typically fastest on cpu "
        "(TRAINING.md §1.4); cuda mainly helps the update step.",
    ),
    PathField(
        attr="checkpoint_dir",
        label="checkpoint dir",
        group_path=("RUN SETTINGS",),
        help="Where last.pt / best.pt / the metrics log / the archive folder live. "
        "Editing this re-inspects the new directory for an existing run.",
    ),
    TextField(
        attr="run_name",
        label="run name",
        group_path=("RUN SETTINGS",),
        help="Names this run's log file and the default archive label.",
    ),
    IntField(
        attr="history_len",
        label="history length",
        group_path=("RUN SETTINGS",),
        unit="iters",
        step=128,
        help="In-memory iterations retained for the live convergence charts.",
    ),
    # -----------------------------------------------------------------------
    # COLLECTION
    # -----------------------------------------------------------------------
    FloatField(
        attr="produce_ewma_alpha",
        label="produce ewma α",
        group_path=("COLLECTION",),
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Smoothing for the PRODUCING band's score / margin readouts.",
    ),
    ChoiceField(
        attr="combine_gain_food",
        label="combine food gains",
        group_path=("COLLECTION",),
        choices=["True", "False"],
        impact=ChangeImpact.REGIME,
        help="When True, multi-die (feeder) and multi-token (supply) food gains are "
        "collapsed into a single FoodSubset decision presenting all legal subsets at "
        "once instead of one die/token at a time. Resumable — no shape change.",
    ),
    # COLLECTION ▸ BOOTSTRAP
    BootstrapField(
        attr="bootstrap_opponent",
        label="bootstrap opponent",
        group_path=("COLLECTION", "BOOTSTRAP"),
        impact=ChangeImpact.REGIME,
        help="Bootstrap phase opponent. 'none' skips the phase and starts directly "
        "in self-play. 'random' uses the built-in random agent. "
        "A checkpoint path loads that run's weights as the fixed opponent. "
        "←/→ cycles through none / random / archived runs; press enter to type a "
        "custom path. Only a checkpoint path requires device='cpu'.",
    ),
    FloatField(
        attr="random_phase_win_rate",
        label="graduate @",
        group_path=("COLLECTION", "BOOTSTRAP"),
        step=0.05,
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.initial_vs_random,
        help="Smoothed collection win-rate (vs random) at which the bootstrap "
        "phase freezes self·gen1 and switches to self-play. Lowering it below "
        "the current win-rate graduates immediately.",
    ),
    # COLLECTION ▸ BOOTSTRAP ▸ RANDOM SETUP
    IntField(
        attr="setup_hand_combos",
        label="hand combos",
        group_path=("COLLECTION", "BOOTSTRAP", "RANDOM SETUP"),
        step=1,
        visible_when=lambda cfg: (
            cfg.opponent.bootstrap_opponent == "random"
            and cfg.architecture.use_setup_model
        ),
        help="Random generator: joint (P0,P1) keep-combos sampled per shared-deal "
        "batch.",
    ),
    IntField(
        attr="setup_food_sets",
        label="food sets",
        group_path=("COLLECTION", "BOOTSTRAP", "RANDOM SETUP"),
        step=1,
        visible_when=lambda cfg: (
            cfg.opponent.bootstrap_opponent == "random"
            and cfg.architecture.use_setup_model
        ),
        help="Random generator: food keeps sampled per kept hand (softmax-biased "
        "toward food that pays for more hand/tray birds).",
    ),
    # -----------------------------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------------------------
    IntField(
        attr="eval_every",
        label="eval every",
        group_path=("EVALUATION",),
        unit="iters",
        step=1,
        help="Run a paired eval block every N iterations; 0 disables evaluation "
        "entirely.",
    ),
    IntField(
        attr="eval_games",
        label="eval games",
        group_path=("EVALUATION",),
        unit="games",
        step=8,
        help="Held-out games per eval, played as mirrored pairs to cancel the "
        "first-player edge (an odd value rounds down to the nearest pair).",
    ),
    FloatField(
        attr="eval_ewma_alpha",
        label="eval ewma α",
        group_path=("EVALUATION",),
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Smoothing for the eval win-rate / margin trend (higher = more "
        "responsive). Re-smooths the restored history on resume.",
    ),
    IntField(
        attr="target_eval_games",
        label="target eval games",
        group_path=("EVALUATION",),
        unit="games",
        step=100,
        help="Self-play games (model fixed, greedy) run at the target milestone. "
        "0 = auto: 10 × eval games.",
    ),
    # EVALUATION ▸ OPPONENT
    FloatField(
        attr="opponent_reset_win_rate",
        label="opp. advance @",
        group_path=("EVALUATION", "OPPONENT"),
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Freeze the current policy as the new opponent once smoothed win-rate "
        "clears this (0 disables). Lowering it below the current win-rate triggers "
        "an immediate advance + win-rate reset.",
    ),
    IntField(
        attr="opponent_max_iterations",
        label="opp. max iters",
        group_path=("EVALUATION", "OPPONENT"),
        step=50,
        impact=ChangeImpact.REGIME,
        help="Force-advance the opponent after this many iterations even if the "
        "win-rate threshold is not yet met (0 disables). Only applies during "
        "self-play; the random-phase bootstrap uses its own graduation logic.",
    ),
    # -----------------------------------------------------------------------
    # TRAINING
    # -----------------------------------------------------------------------
    ChoiceField(
        attr="policy_loss",
        label="policy loss",
        group_path=("TRAINING",),
        choices=[pl.value for pl in config.PolicyLoss],
        impact=ChangeImpact.REGIME,
        help="Policy-gradient objective. 'reinforce' = standard log-prob weighted "
        "advantage (default). 'ppo' = PPO clipped surrogate, enabling safe "
        "multi-epoch reuse of each collected batch via 'PPO reuse epochs'. "
        "Shape-preserving — reinterprets an in-progress run but resumes weights.",
    ),
    FloatField(
        attr="ppo_clip_eps",
        label="PPO clip ε",
        group_path=("TRAINING",),
        step=0.05,
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.training.policy_loss is config.PolicyLoss.PPO,
        help="PPO probability-ratio clip radius. The surrogate is clipped to "
        "[1−ε, 1+ε] × advantage, preventing large policy updates within a "
        "reuse epoch. Default 0.2. PPO mode only.",
    ),
    IntField(
        attr="ppo_reuse_epochs",
        label="PPO reuse epochs",
        group_path=("TRAINING",),
        step=1,
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.training.policy_loss is config.PolicyLoss.PPO,
        help="Number of full-batch gradient passes over each collected batch. "
        "More epochs extract more learning per collection round at the cost of "
        "policy lag (mitigated by the PPO clip). PPO mode only.",
    ),
    IntField(
        attr="update_minibatch_steps",
        label="update minibatch",
        group_path=("TRAINING",),
        step=512,
        impact=ChangeImpact.REGIME,
        help="Gradient-accumulation minibatch size (0 = whole batch, today's "
        "default).  > 0 splits the flattened batch into chunks of this many "
        "steps and accumulates gradients before each optimizer.step(), capping "
        "peak RAM at the minibatch size.  Recommended ≈ 4096 for "
        "large-games_per_iter runs.",
    ),
    FloatField(
        attr="lr",
        label="learning rate",
        group_path=("TRAINING",),
        step=1e-4,
        scientific=True,
        help="Adam step size for the REINFORCE + value-baseline update. Honored "
        "immediately on resume (the optimizer momentum is kept).",
    ),
    FloatField(
        attr="value_coef",
        label="value coef",
        group_path=("TRAINING",),
        step=0.05,
        help="Weight of the value-head (critic) loss term in the total loss.",
    ),
    FloatField(
        attr="entropy_coef",
        label="entropy coef",
        group_path=("TRAINING",),
        step=0.005,
        help="Entropy bonus that keeps the policy exploring; 0 disables it.",
    ),
    FloatField(
        attr="grad_clip",
        label="grad clip",
        group_path=("TRAINING",),
        step=0.5,
        help="Global gradient-norm clip — the TRAINING HEALTH band flags norms "
        "above this.",
    ),
    FloatField(
        attr="score_norm",
        label="score norm",
        group_path=("TRAINING",),
        step=5.0,
        impact=ChangeImpact.REGIME,
        help="Advantage / value-target scale. Changing it rescales the loss and "
        "points curves across a resume boundary, so the charts will step.",
    ),
    # TRAINING ▸ REWARD MODEL
    ChoiceField(
        attr="reward_basis",
        label="point metric",
        group_path=("TRAINING", "REWARD MODEL"),
        choices=[basis.value for basis in config.RewardBasis],
        impact=ChangeImpact.REGIME,
        help="Reward signal basis. 'margin' = own score minus opponent (signed); "
        "'own_score' = player's absolute final score (always positive). Shape-"
        "preserving — toggles the loss gradient direction without restarting.",
    ),
    ChoiceField(
        attr="reward_mode",
        label="reward mode",
        group_path=("TRAINING", "REWARD MODEL"),
        choices=[mode.value for mode in config.RewardMode],
        impact=ChangeImpact.REGIME,
        help="How each decision's return is computed. 'terminal_margin' broadcasts "
        "the end-of-game margin to every decision; 'decision_delta' credits each "
        "decision with its own margin change, discounted per unit of game-clock "
        "time; 'gae' uses the GAE critic-bootstrapped advantage. "
        "Shape-preserving — reinterprets an in-progress run but resumes weights.",
    ),
    FloatField(
        attr="reward_discount",
        label="reward discount γ",
        group_path=("TRAINING", "REWARD MODEL"),
        step=0.05,
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.training.reward_mode
        in (config.RewardMode.DECISION_DELTA, config.RewardMode.GAE),
        help="Discount γ for the decision-delta / GAE return, per game turn of "
        "clock time. γ=0 = immediate change only; γ=1 = final margin minus the "
        "current margin. Delta and GAE modes only.",
    ),
    FloatField(
        attr="gae_lambda",
        label="GAE lambda λ",
        group_path=("TRAINING", "REWARD MODEL"),
        step=0.05,
        impact=ChangeImpact.REGIME,
        visible_when=lambda cfg: cfg.training.reward_mode is config.RewardMode.GAE,
        help="GAE bias-variance trade-off λ (Schulman et al. 2016). λ=1 reduces "
        "to the full MC advantage (high variance, low bias); λ=0 reduces to "
        "one-step TD (low variance, high bias). Default 0.95. GAE mode only.",
    ),
    # TRAINING ▸ CLONING
    IntField(
        attr="clone_iters",
        label="clone iters",
        group_path=("TRAINING", "CLONING"),
        step=1,
        impact=ChangeImpact.REGIME,
        visible_when=_bootstrap_is_checkpoint,
        help="Number of initial iterations to train the student via cross-entropy "
        "to the bootstrap opponent's soft targets. After this many iterations "
        "imitation labeling stops and training continues as the normal actor-critic "
        "loop against the same checkpoint opponent. Visible only when a checkpoint "
        "bootstrap opponent is set.",
    ),
    # -----------------------------------------------------------------------
    # MODEL ARCHITECTURE
    # -----------------------------------------------------------------------
    # MODEL ARCHITECTURE ▸ CARD ENCODER
    IntField(
        attr="card_embed_dim",
        label="dimension",
        group_path=("MODEL ARCHITECTURE", "CARD ENCODER"),
        unit="units",
        step=16,
        impact=ChangeImpact.FRESH,
        help="Width of the shared per-card vector (reused for every board / tray / "
        "hand / choice card slot) — the card encoder's output width. Fresh run.",
    ),
    LayersField(
        attr="card_encoder_layers",
        label="layers",
        group_path=("MODEL ARCHITECTURE", "CARD ENCODER"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        help="Card encoder MLP hidden widths (input→output). Maps each card's "
        "[static attributes ⊕ identity one-hot] to its card-embed-dim vector. "
        "Empty (← to 0 layers) = a single linear projection. Fresh run.",
    ),
    OptionalActivationField(
        attr="card_between_activation",
        label="between activation",
        group_path=("MODEL ARCHITECTURE", "CARD ENCODER"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        help="Card encoder activation between layers (inherit = use global between). Resumable.",
    ),
    OptionalActivationField(
        attr="card_final_activation",
        label="final activation",
        group_path=("MODEL ARCHITECTURE", "CARD ENCODER"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        help="Card encoder activation after its last layer (inherit = use global final; none = no final). Resumable.",
    ),
    OptionalFloatField(
        attr="card_dropout",
        label="dropout",
        group_path=("MODEL ARCHITECTURE", "CARD ENCODER"),
        step=0.05,
        fallback_attr="dropout",
        impact=ChangeImpact.REGIME,
        help="Card encoder dropout (None = inherit global). Resumable.",
    ),
    OptionalChoiceField(
        attr="card_layernorm",
        label="layernorm",
        group_path=("MODEL ARCHITECTURE", "CARD ENCODER"),
        choices=["True", "False"],
        impact=ChangeImpact.FRESH,
        help="Card encoder LayerNorm (None = inherit global). Fresh run when changed.",
    ),
    # MODEL ARCHITECTURE ▸ HAND POOLING
    ChoiceField(
        attr="hand_pooling",
        label="hand pooling",
        group_path=("MODEL ARCHITECTURE", "HAND POOLING"),
        choices=["concat_max_sum", "max", "sum", "mean"],
        impact=ChangeImpact.FRESH,
        help="Permutation-invariant pooling mode for the hand set embedding. "
        "Replaces the retired dedicated hand encoder (use_distinct_hand_model=False). "
        "concat_max_sum (default): [max|sum|count] (2M+1 wide); max: [max|count] "
        "(M+1); sum: M; mean: M (reproduces old mean-pool). Forces a fresh run "
        "(architecture_key changes). Old distinct-encoder checkpoints are unaffected.",
    ),
    # MODEL ARCHITECTURE ▸ STATE TRUNK
    LayersField(
        attr="trunk_layers",
        label="layers",
        group_path=("MODEL ARCHITECTURE", "STATE TRUNK"),
        unit="units",
        impact=ChangeImpact.FRESH,
        help="State-trunk hidden widths (input→output), e.g. 256,128. Type to set "
        "the sizes; ←/→ adds or removes a layer. Its last width is M, the trunk "
        "embedding fed to the value head and concatenated for scoring. Fresh run.",
    ),
    OptionalActivationField(
        attr="trunk_between_activation",
        label="between activation",
        group_path=("MODEL ARCHITECTURE", "STATE TRUNK"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        help="State trunk activation between layers (inherit = use global between). Resumable.",
    ),
    OptionalActivationField(
        attr="trunk_final_activation",
        label="final activation",
        group_path=("MODEL ARCHITECTURE", "STATE TRUNK"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        help="State trunk activation after its last layer (inherit = use global between; none = no final). Resumable.",
    ),
    OptionalFloatField(
        attr="trunk_dropout",
        label="dropout",
        group_path=("MODEL ARCHITECTURE", "STATE TRUNK"),
        step=0.05,
        fallback_attr="dropout",
        impact=ChangeImpact.REGIME,
        help="State trunk dropout (None = inherit global). Resumable.",
    ),
    OptionalChoiceField(
        attr="trunk_layernorm",
        label="layernorm",
        group_path=("MODEL ARCHITECTURE", "STATE TRUNK"),
        choices=["True", "False"],
        impact=ChangeImpact.FRESH,
        help="State trunk LayerNorm (None = inherit global). Fresh run when changed.",
    ),
    ChoiceField(
        attr="use_board_attention",
        label="board attention",
        group_path=("MODEL ARCHITECTURE", "STATE TRUNK"),
        choices=["True", "False"],
        impact=ChangeImpact.FRESH,
        help="When True, each player's 15 board slots are attended over as tokens "
        "(card_embed ⊕ 9 mutable scalars) before the trunk, using two independent "
        "MultiheadAttention modules (own board + opponent board). Single-head for "
        "this first pass. Config-carried REGIME topology — but forces a fresh run "
        "(architecture_key changes) since weights differ with attention on vs off. "
        "Default False; old checkpoints load unchanged.",
    ),
    # MODEL ARCHITECTURE ▸ CHOICE ENCODER
    LayersField(
        attr="choice_layers",
        label="layers",
        group_path=("MODEL ARCHITECTURE", "CHOICE ENCODER"),
        unit="units",
        impact=ChangeImpact.FRESH,
        help="Per-choice encoder widths (input→output). Its last width is N, the "
        "choice embedding concatenated with M before the scorer heads. Fresh run.",
    ),
    OptionalActivationField(
        attr="choice_between_activation",
        label="between activation",
        group_path=("MODEL ARCHITECTURE", "CHOICE ENCODER"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        help="Choice encoder activation between layers (inherit = use global between). Resumable.",
    ),
    OptionalActivationField(
        attr="choice_final_activation",
        label="final activation",
        group_path=("MODEL ARCHITECTURE", "CHOICE ENCODER"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        help="Choice encoder activation after its last layer (inherit = use global final; none = no final). Resumable.",
    ),
    OptionalFloatField(
        attr="choice_dropout",
        label="dropout",
        group_path=("MODEL ARCHITECTURE", "CHOICE ENCODER"),
        step=0.05,
        fallback_attr="dropout",
        impact=ChangeImpact.REGIME,
        help="Choice encoder dropout (None = inherit global). Resumable.",
    ),
    OptionalChoiceField(
        attr="choice_layernorm",
        label="layernorm",
        group_path=("MODEL ARCHITECTURE", "CHOICE ENCODER"),
        choices=["True", "False"],
        impact=ChangeImpact.FRESH,
        help="Choice encoder LayerNorm (None = inherit global). Fresh run when changed.",
    ),
    # MODEL ARCHITECTURE ▸ CRITIC HEAD
    LayersField(
        attr="value_layers",
        label="layers",
        group_path=("MODEL ARCHITECTURE", "CRITIC HEAD"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        help="Value-head hidden widths before the scalar output. Empty = a direct "
        "M→1 readout (the default). Fresh run.",
    ),
    OptionalActivationField(
        attr="value_between_activation",
        label="between activation",
        group_path=("MODEL ARCHITECTURE", "CRITIC HEAD"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        help="Value head activation between hidden layers (inherit = use global between). Resumable.",
    ),
    OptionalActivationField(
        attr="value_final_activation",
        label="final activation",
        group_path=("MODEL ARCHITECTURE", "CRITIC HEAD"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        help="Value head activation after the final Linear(·,1) (inherit = use global final; none = bare scalar output). Resumable.",
    ),
    # MODEL ARCHITECTURE ▸ ACTOR HEADS
    ChoiceField(
        attr="head_layers_mode",
        label="scorer head mode",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        choices=["uniform", "per_family"],
        impact=ChangeImpact.FRESH,
        help="'uniform' — all decision families share one scorer head shape. "
        "'per_family' — configure each family's hidden widths independently. "
        "Fresh run.",
    ),
    LayersField(
        attr="head_layers",
        label="layers",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_uniform_heads,
        help="Per-family scorer hidden widths between the M+N concat and the final "
        "logit. Empty (←  to 0 layers) = a direct (M+N)→1 readout. Fresh run.",
    ),
    LayersField(
        attr="head_layers_main_action",
        label="head: main action",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the main-action family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_draw_bird",
        label="head: draw bird",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the draw-bird family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_discard_bird",
        label="head: discard bird",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the discard-bird family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_gain_food",
        label="head: gain food",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the gain-food family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_spend_food",
        label="head: spend food",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the spend-food family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_lay_egg",
        label="head: lay egg",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the lay-egg family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_pay_egg",
        label="head: pay egg",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the pay-egg family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_skip_optional",
        label="head: skip optional",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the skip-optional family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_choose_bonus",
        label="head: choose bonus",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the choose-bonus family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_misc_rare",
        label="head: misc rare",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the misc-rare family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_play_bird",
        label="head: play bird",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the play-bird family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_reset_birdfeeder",
        label="head: reset birdfeeder",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the reset-birdfeeder family. Fresh run.",
    ),
    LayersField(
        attr="head_layers_setup",
        label="head: setup",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        unit="units",
        min_len=0,
        impact=ChangeImpact.FRESH,
        visible_when=_per_family_heads,
        help="Scorer head hidden widths for the setup family. Fresh run.",
    ),
    OptionalActivationField(
        attr="head_between_activation",
        label="between activation",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        help="Scorer head activation between hidden layers (inherit = use global between). Resumable.",
    ),
    OptionalActivationField(
        attr="head_final_activation",
        label="final activation",
        group_path=("MODEL ARCHITECTURE", "ACTOR HEADS"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        help="Scorer head activation after the final Linear(·,1) (inherit = use global final; none = bare logit). Resumable.",
    ),
    # MODEL ARCHITECTURE ▸ GLOBAL DEFAULTS
    ChoiceField(
        attr="between_activation",
        label="between activation",
        group_path=("MODEL ARCHITECTURE", "GLOBAL DEFAULTS"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        help="Global between-layers activation for all MLP blocks. Per-block overrides "
        "take precedence when set. Resumable (doesn't change tensor shapes).",
    ),
    ChoiceField(
        attr="final_activation",
        label="final activation",
        group_path=("MODEL ARCHITECTURE", "GLOBAL DEFAULTS"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        help="Global final-layer activation for all MLP blocks except the trunk "
        "(which inherits between_activation). Per-block overrides take precedence. "
        "Default none = no final activation. Resumable.",
    ),
    FloatField(
        attr="dropout",
        label="dropout",
        group_path=("MODEL ARCHITECTURE", "GLOBAL DEFAULTS"),
        step=0.05,
        impact=ChangeImpact.REGIME,
        help="Global dropout after each activation (training only). Per-block "
        "overrides take precedence when set. 0 disables. Resumable.",
    ),
    ChoiceField(
        attr="layernorm",
        label="layernorm",
        group_path=("MODEL ARCHITECTURE", "GLOBAL DEFAULTS"),
        choices=["True", "False"],
        impact=ChangeImpact.FRESH,
        help="Global LayerNorm in the body blocks. Per-block overrides take "
        "precedence when set. Adds parameters — fresh run to toggle.",
    ),
    # MODEL ARCHITECTURE ▸ SETUP MODEL
    ChoiceField(
        attr="use_setup_model",
        label="enable",
        group_path=("MODEL ARCHITECTURE", "SETUP MODEL"),
        choices=["True", "False"],
        impact=ChangeImpact.REGIME,
        help="Train the start-of-game keep with a separate value-regression net "
        "instead of the in-game policy. Off = unchanged behaviour. Has its own "
        "checkpoint, so toggling it never invalidates the main net's weights.",
    ),
    ChoiceField(
        attr="split_setup_bonus",
        label="includes bonus",
        group_path=("MODEL ARCHITECTURE", "SETUP MODEL"),
        choices=["True", "False"],
        impact=ChangeImpact.REGIME,
        visible_when=_use_setup,
        help="True = setup net picks the bonus card. "
        "False = bonus pick is deferred to the in-game CHOOSE_BONUS head. "
        "Shape-preserving.",
    ),
    ChoiceField(
        attr="split_setup_food",
        label="includes food",
        group_path=("MODEL ARCHITECTURE", "SETUP MODEL"),
        choices=["True", "False"],
        impact=ChangeImpact.REGIME,
        visible_when=_use_setup,
        help="True = setup net picks food. "
        "False = food deferred to sequential GAIN_FOOD/SPEND_FOOD decisions. "
        "Shape-preserving.",
    ),
    LayersField(
        attr="setup_trunk_layers",
        label="trunk layers",
        group_path=("MODEL ARCHITECTURE", "SETUP MODEL"),
        unit="units",
        impact=ChangeImpact.REGIME,
        visible_when=_use_setup,
        help="Shared trunk widths before the value/policy split. Empty = no trunk.",
    ),
    LayersField(
        attr="setup_hidden_layers",
        label="head layers",
        group_path=("MODEL ARCHITECTURE", "SETUP MODEL"),
        unit="units",
        impact=ChangeImpact.REGIME,
        visible_when=_use_setup,
        help="Per-head MLP hidden widths (value + policy). Changing restarts only the setup net.",
    ),
    ChoiceField(
        attr="setup_between_activation",
        label="between activation",
        group_path=("MODEL ARCHITECTURE", "SETUP MODEL"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        visible_when=_use_setup,
        help="Activation between hidden layers in the setup net's MLP.",
    ),
    ChoiceField(
        attr="setup_final_activation",
        label="final activation",
        group_path=("MODEL ARCHITECTURE", "SETUP MODEL"),
        choices=_ACT_CHOICES,
        impact=ChangeImpact.REGIME,
        visible_when=_use_setup,
        help="Activation after the final Linear(·,1) in the setup net (none = bare scalar).",
    ),
    FloatField(
        attr="setup_dropout",
        label="dropout",
        group_path=("MODEL ARCHITECTURE", "SETUP MODEL"),
        step=0.05,
        impact=ChangeImpact.REGIME,
        visible_when=_use_setup,
        help="Dropout after each setup-net activation (training only). 0 disables.",
    ),
    FloatField(
        attr="setup_lr",
        label="lr",
        group_path=("MODEL ARCHITECTURE", "SETUP MODEL"),
        step=1e-4,
        scientific=True,
        visible_when=_use_setup,
        help="Adam step size for the setup net's actor-critic updates (its own optimizer).",
    ),
    FloatField(
        attr="setup_pg_coef",
        label="PG coef",
        group_path=("MODEL ARCHITECTURE", "SETUP MODEL"),
        step=0.1,
        impact=ChangeImpact.REGIME,
        visible_when=_use_setup,
        help="Policy-gradient loss weight for the setup actor-critic update.",
    ),
    FloatField(
        attr="setup_value_coef",
        label="value coef",
        group_path=("MODEL ARCHITECTURE", "SETUP MODEL"),
        step=0.1,
        impact=ChangeImpact.REGIME,
        visible_when=_use_setup,
        help="Value-head MSE loss weight for the setup actor-critic update.",
    ),
    FloatField(
        attr="setup_entropy_coef",
        label="entropy coef",
        group_path=("MODEL ARCHITECTURE", "SETUP MODEL"),
        step=0.005,
        scientific=True,
        impact=ChangeImpact.REGIME,
        visible_when=_use_setup,
        help="Entropy-bonus weight for the setup actor-critic update; 0 disables.",
    ),
]

_BY_ATTR: dict[str, FieldSpec] = {spec.attr: spec for spec in FIELD_SPECS}
_DEFAULTS = config.RunConfig()
_FLOAT_ROUND = 6  # decimal places a nudged float is rounded to (kills FP crud)
# Width seeded when a RIGHT-nudge adds the first layer to an empty list.
_NEW_LAYER_WIDTH = 128
