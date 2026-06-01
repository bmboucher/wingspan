"""Stable card -> dense-index maps for the RL encoder.

The encoder represents a card as a one-hot over all core-set cards; that needs
a stable dense index per card independent of any shuffle. These helpers derive
one from the full loaded catalog and cache it (keyed on card id so a trained
per-card embedding stays aligned across runs).
"""

from __future__ import annotations

import functools

from wingspan.cards import schema
from wingspan.cards.parse import loader


@functools.lru_cache(maxsize=1)
def _canonical_cards() -> tuple[tuple[schema.Bird, ...], tuple[schema.BonusCard, ...]]:
    birds, bonuses, _ = loader.load_all()
    return tuple(birds), tuple(bonuses)


@functools.lru_cache(maxsize=1)
def _bird_index_by_id() -> dict[int, int]:
    birds, _ = _canonical_cards()
    return {bird.id: i for i, bird in enumerate(birds)}


@functools.lru_cache(maxsize=1)
def _bonus_index_by_id() -> dict[int, int]:
    _, bonuses = _canonical_cards()
    return {bonus_card.id: i for i, bonus_card in enumerate(bonuses)}


def n_birds() -> int:
    """Number of distinct core-set birds — the length of the bird-identity
    one-hot (and the kept-set / hand multi-hot) stripe in the RL encoder."""
    return len(_canonical_cards()[0])


def n_bonus_cards() -> int:
    """Number of distinct core-set bonus cards — the length of the bonus-card
    identity one-hot stripe in the RL encoder."""
    return len(_canonical_cards()[1])


def bird_index(bird: schema.Bird) -> int:
    """Stable dense index of ``bird`` in the core-set catalog, used for the
    bird-identity one-hot. Keyed on the card id, so it is identical across
    games and a trained per-card embedding stays aligned."""
    return _bird_index_by_id()[bird.id]


def bonus_index(bonus_card: schema.BonusCard) -> int:
    """Stable dense index of ``bonus_card`` in the core-set catalog, used for
    the bonus-card identity one-hot."""
    return _bonus_index_by_id()[bonus_card.id]
