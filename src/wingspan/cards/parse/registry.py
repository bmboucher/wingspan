"""Ordered registries for the bird power-text matchers.

A matcher takes the normalized power text and returns a ``schema.Effect`` or
``None``. Two ordered lists drive parsing: the general (when-played) patterns
and the pink (reactive) patterns. **Order is load-bearing** — a more-specific
pattern must run before an overlapping generic one. Within a submodule the
registration (decoration) order equals source order, and the package
``__init__`` imports ``matchers`` before ``pink_matchers``, so the general
patterns occupy the front of ``_PATTERN_MATCHERS`` and the two dual-membership
pink matchers its tail — reproducing the original hand-written tuples exactly.
"""

from __future__ import annotations

import typing

from wingspan.cards import schema

_Matcher = typing.Callable[[str], schema.Effect | None]

_PATTERN_MATCHERS: list[_Matcher] = []
_PINK_MATCHERS: list[_Matcher] = []


def pattern(matcher: _Matcher) -> _Matcher:
    """Register ``matcher`` in the general (non-reactive) pattern list."""
    _PATTERN_MATCHERS.append(matcher)
    return matcher


def pink_pattern(matcher: _Matcher) -> _Matcher:
    """Register ``matcher`` in the pink (reactive) pattern list."""
    _PINK_MATCHERS.append(matcher)
    return matcher


def matchers_for(reactive: bool) -> list[_Matcher]:
    """The ordered matchers to try: the pink list if ``reactive`` else the
    general list (see :func:`wingspan.cards.parse.power._extract_effects`)."""
    return _PINK_MATCHERS if reactive else _PATTERN_MATCHERS
