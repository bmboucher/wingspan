"""The configurator's live UI state and the small value-objects it carries.

:class:`ConfiguratorState` is the single mutable record the controller updates on
each keypress and the screen reads to repaint — the form's working config, the
saved run it is being compared against, the inspected directory summary, the
current mode, the in-flight edit buffer, a transient status message, and any
open confirmation prompt. Everything is a Pydantic model; the IO objects (the
rich ``Live`` and the :class:`keys.KeyReader`) stay in the controller, not here.
"""

from __future__ import annotations

import enum

import pydantic

from wingspan.training import config
from wingspan.training.configure import fields, runs


class Mode(enum.StrEnum):
    """What the configurator is doing — drives input dispatch and the footer."""

    NAVIGATE = "navigate"  # moving between / nudging fields
    EDIT = "edit"  # typing a new value into the focused field
    CONFIRM = "confirm"  # a modal prompt is open


class MessageKind(enum.StrEnum):
    """Severity of the one-line status message (drives its color)."""

    INFO = "info"
    SUCCESS = "success"
    WARN = "warn"
    ERROR = "error"


class Outcome(enum.StrEnum):
    """The result of dispatching one keypress."""

    CONTINUE = "continue"  # stay in the configurator
    QUIT = "quit"  # leave without launching
    LAUNCH = "launch"  # leave and start training with ``state.working``


class ConfirmAction(enum.StrEnum):
    """What a confirmation option does when chosen."""

    ARCHIVE_THEN_FRESH = "archive_then_fresh"
    OVERWRITE_THEN_FRESH = "overwrite_then_fresh"
    ARCHIVE_ONLY = "archive_only"
    RESET_TO_DEFAULTS = "reset_to_defaults"
    CANCEL = "cancel"


class Message(pydantic.BaseModel):
    """A transient status line shown in the footer."""

    kind: MessageKind
    text: str


class ConfirmOption(pydantic.BaseModel):
    """One choice in a confirmation prompt."""

    key: str  # the hotkey (single lowercase char)
    label: str
    action: ConfirmAction
    danger: bool = False  # rendered de-emphasized / in the alarm color


def _empty_lines() -> list[str]:
    return []


def _empty_options() -> list[ConfirmOption]:
    return []


class ConfirmPrompt(pydantic.BaseModel):
    """A modal confirmation: a title, explanatory lines, and keyed options."""

    title: str
    lines: list[str] = pydantic.Field(default_factory=_empty_lines)
    options: list[ConfirmOption] = pydantic.Field(default_factory=_empty_options)
    default_key: str = ""  # which option is highlighted

    def option_for(self, key: str) -> ConfirmOption | None:
        for option in self.options:
            if option.key == key:
                return option
        return None


class ConfiguratorState(pydantic.BaseModel):
    """Everything the screen needs to repaint a single configurator frame."""

    working: config.TrainConfig  # the config being edited / to be launched
    summary: runs.RunSummary  # the run currently on disk in checkpoint_dir
    saved: config.TrainConfig | None = None  # the saved run's config, if any
    cuda_available: bool = True
    selected_attr: str = ""  # which field is focused
    mode: Mode = Mode.NAVIGATE
    edit_buffer: str = ""  # the in-flight value while editing
    message: Message | None = None
    confirm: ConfirmPrompt | None = None
    # Whether the editor was seeded from the saved run rather than from defaults
    # (shown in the header so the user knows which settings they are tuning).
    seeded_from_saved: bool = False

    def selected_spec(self) -> fields.FieldSpec:
        """The :class:`fields.FieldSpec` of the focused field."""
        return fields.spec_for(self.selected_attr)

    def status(self) -> runs.RunStatus:
        """What Start will do against the inspected directory right now."""
        return runs.resolve_status(self.summary, self.working)

    def notify(self, kind: MessageKind, text: str) -> None:
        """Set the footer status message."""
        self.message = Message(kind=kind, text=text)
