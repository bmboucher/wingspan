"""The serializable instrumentation config and the per-run context.

``InstrumentationConfig`` is the Pydantic record a run carries (it is a field on
``training.config.TrainConfig`` and can also be loaded standalone for the
single-process driver). It has two layers, matching the user-facing YAML:

* ``handlers`` — named handler instances, each declared as a registered ``class``
  name plus that class's config kwargs.
* ``events`` — an assignment mapping each ``EventName`` to the handler names that
  fire on it. One handler may appear under several events and accumulate state
  across them.

Deserialization resolves each ``{class: Name, **kwargs}`` block to its concrete
``CallbackHandler`` subclass via the registry, and serialization writes it back
out the same way, so the config round-trips through ``model_dump`` /
``model_validate`` (checkpoints) and ``yaml.safe_load`` -> ``model_validate``
(the cloud run file). The only ``Any`` is the transient raw mapping at that
deserialize / serialize boundary — exactly the sanctioned yaml-to-model edge.

The ``events`` module is imported under the alias ``event_types`` so the field
literally named ``events`` does not shadow it during Pydantic's annotation
resolution.
"""

from __future__ import annotations

import pathlib
import typing

import pydantic

from wingspan.instrumentation import dispatcher
from wingspan.instrumentation import events as event_types
from wingspan.instrumentation import registry

# The discriminator key naming a handler's registered class in the config.
_CLASS_KEY = "class"


class RunContext(pydantic.BaseModel):
    """The per-run handle passed to every handler's ``open``.

    Carries the directory handlers write their output under, plus run identity
    for filenames and headers. ``worker_id`` is unset in the single-process
    driver and reserved for the future per-worker collection path, where each
    worker writes its own shard.
    """

    output_dir: pathlib.Path
    run_name: str
    seed: int
    matchup: tuple[str, str] | None = None
    worker_id: int | None = None


class InstrumentationConfig(pydantic.BaseModel):
    """Named handler instances plus an event-name -> handler-names assignment.

    The empty default attaches no handlers, so a run with no instrumentation
    pays nothing.
    """

    handlers: dict[str, pydantic.SerializeAsAny[event_types.CallbackHandler]] = (
        pydantic.Field(default_factory=dict[str, event_types.CallbackHandler])
    )
    events: dict[event_types.EventName, list[str]] = pydantic.Field(
        default_factory=dict[event_types.EventName, list[str]]
    )

    @pydantic.field_validator("handlers", mode="before")
    @classmethod
    def _resolve_handlers(cls, value: typing.Any) -> typing.Any:
        """Turn each ``{class: Name, **kwargs}`` block into the concrete handler
        instance the registry names. Already-resolved handler instances (the
        in-memory construction path) pass through untouched."""
        if not isinstance(value, dict):
            return value
        raw_handlers = typing.cast("dict[str, typing.Any]", value)
        return {name: _resolve_handler(name, raw) for name, raw in raw_handlers.items()}

    @pydantic.field_serializer("handlers")
    def _serialize_handlers(
        self, handlers: dict[str, event_types.CallbackHandler]
    ) -> dict[str, dict[str, typing.Any]]:
        """Write each handler back as ``{class: Name, **fields}`` so a reload
        reconstructs the identical concrete subclass."""
        out: dict[str, dict[str, typing.Any]] = {}
        for name, handler in handlers.items():
            registered = registry.name_for(type(handler))
            if registered is None:
                raise ValueError(
                    f"handler {type(handler).__name__} is not registered; "
                    "decorate it with @registry.register"
                )
            out[name] = {_CLASS_KEY: registered, **handler.model_dump()}
        return out

    @pydantic.model_validator(mode="after")
    def _check_assignment(self) -> InstrumentationConfig:
        """Every assigned name must resolve to a defined handler that actually
        implements the event (subclasses the event's handler base)."""
        for event, names in self.events.items():
            base = event_types.EVENT_BASE[event]
            for name in names:
                handler = self.handlers.get(name)
                if handler is None:
                    raise ValueError(
                        f"event {event.value!r} references unknown handler {name!r}"
                    )
                if not isinstance(handler, base):
                    raise ValueError(
                        f"handler {name!r} ({type(handler).__name__}) does not "
                        f"implement event {event.value!r}"
                    )
        return self

    def build(self) -> dispatcher.Instrumentation:
        """Construct the live router. The *same* handler instance is shared
        across every event it is assigned to, so a multi-event handler keeps one
        accumulator."""
        by_event: dict[event_types.EventName, list[event_types.CallbackHandler]] = {}
        for event, names in self.events.items():
            by_event[event] = [self.handlers[name] for name in names]
        return dispatcher.Instrumentation(by_event=by_event)


###### PRIVATE #######


def _resolve_handler(name: str, raw: typing.Any) -> event_types.CallbackHandler:
    """Resolve one handler spec to a concrete instance: an already-built handler
    passes through; a ``{class: Name, **kwargs}`` mapping is looked up in the
    registry and validated into its class."""
    if isinstance(raw, event_types.CallbackHandler):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(
            f"handler {name!r} must be a mapping of {_CLASS_KEY} + kwargs, "
            f"got {type(raw).__name__}"
        )
    spec = dict(typing.cast("dict[str, typing.Any]", raw))
    class_name = spec.pop(_CLASS_KEY, None)
    if not isinstance(class_name, str):
        raise ValueError(f"handler {name!r} is missing a {_CLASS_KEY!r} field")
    handler_class = registry.handler_class_for(class_name)
    if handler_class is None:
        raise ValueError(f"unknown handler class {class_name!r} for handler {name!r}")
    return handler_class.model_validate(spec)
