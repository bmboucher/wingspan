"""The bird-power handler registry.

Each effect handler lives in a sibling submodule and registers itself here
with the :func:`handles` decorator, keyed by its :class:`cards.EffectKind`.
The dispatcher reads :data:`_HANDLERS` at call time; the package ``__init__``
imports every handler submodule so the table is full before the first dispatch.
"""

from __future__ import annotations

import typing

from wingspan import cards, state

if typing.TYPE_CHECKING:
    from wingspan.engine import core


_EffectHandler = typing.Callable[
    [
        "core.Engine",
        "core.Agent",
        state.Player,
        state.PlayedBird,
        cards.Habitat,
        cards.Effect,
        str,
    ],
    None,
]

_HANDLERS: dict[cards.EffectKind, _EffectHandler] = {}


def handles(
    kind: cards.EffectKind,
) -> typing.Callable[[_EffectHandler], _EffectHandler]:
    """Register the decorated function as the handler for ``kind`` and return
    it unchanged. Decoration order is irrelevant; lookup is by ``EffectKind``."""

    def register(handler: _EffectHandler) -> _EffectHandler:
        _HANDLERS[kind] = handler
        return handler

    return register


def handler_for(kind: cards.EffectKind) -> _EffectHandler | None:
    """The registered handler for ``kind``, or ``None`` if nothing handles it
    (pink reactor and ``UNIMPLEMENTED`` kinds intentionally have no handler)."""
    return _HANDLERS.get(kind)
