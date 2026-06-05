"""General-purpose event-callback instrumentation for Wingspan games.

A run can attach custom recorders to a game without touching engine code. The
engine fires a fixed set of named events (``events.EventName``); a recorder
subclasses the matching abstract handler base (``events.RoundStartHandler`` …)
and implements its keyword-only method. The run config
(``config.InstrumentationConfig``) defines named handler instances and assigns
them to event names; ``InstrumentationConfig.build`` produces the live
``dispatcher.Instrumentation`` an ``Engine`` holds.

- ``events``     — the ``EventName`` enum + the per-shape abstract handler bases
- ``registry``   — class-name <-> handler-class registry + ``@register``
- ``config``     — ``InstrumentationConfig`` + ``RunContext`` (the Pydantic records)
- ``dispatcher`` — ``Instrumentation``, the live router the engine fires into
- ``handlers``   — built-in recorders (imported here so they self-register)
"""

from wingspan.instrumentation import config, dispatcher, events, handlers, registry

__all__ = [
    "config",
    "dispatcher",
    "events",
    "handlers",
    "registry",
]
