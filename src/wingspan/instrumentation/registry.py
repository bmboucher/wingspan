"""The handler registry: a bijection between config class names and handler
classes.

The run config names handlers by a stable string (the ``class`` field); this
module maps that string to the concrete ``CallbackHandler`` subclass to
instantiate, and back again for serialization. New handlers self-register by
decorating their class with ``@registry.register("Name")`` — mirroring the
``@registry.handles`` pattern the engine's power dispatch uses
(``engine/powers/registry.py``). The ``handlers`` subpackage imports every
handler module so the table self-populates on import.
"""

from __future__ import annotations

import typing

from wingspan.instrumentation import events

# The forward and reverse maps are kept in lockstep by ``register`` so the
# config layer can resolve a name to a class (deserialize) and a class to its
# name (serialize) without scanning.
_REGISTRY: dict[str, type[events.CallbackHandler]] = {}
_NAMES: dict[type[events.CallbackHandler], str] = {}


def register[H: events.CallbackHandler](
    name: str,
) -> typing.Callable[[type[H]], type[H]]:
    """Register ``name`` as the config alias for the decorated handler class.

    Raises ``ValueError`` if ``name`` is already bound to a different class, so
    an accidental name clash fails loudly at import time rather than silently
    shadowing.
    """

    def decorator(handler_class: type[H]) -> type[H]:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not handler_class:
            raise ValueError(
                f"handler name {name!r} is already registered to "
                f"{existing.__name__}"
            )
        _REGISTRY[name] = handler_class
        _NAMES[handler_class] = name
        return handler_class

    return decorator


def handler_class_for(name: str) -> type[events.CallbackHandler] | None:
    """The handler class registered under ``name``, or ``None`` if unknown."""
    return _REGISTRY.get(name)


def name_for(handler_class: type[events.CallbackHandler]) -> str | None:
    """The registered name for ``handler_class``, or ``None`` if unregistered."""
    return _NAMES.get(handler_class)
