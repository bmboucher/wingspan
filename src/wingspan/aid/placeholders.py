"""Stand-ins for physically hidden cards: opponent hand contents and other
face-down draws the user cannot see.

A placeholder is an identity-distinct deep copy of a real catalog card --
not a synthetic sentinel -- so it survives incidental indexing or scoring
untouched (the encoder never reads opponent hand/bonus-card identities, only
counts, so a placeholder's *value* is never actually consulted). Because a
placeholder is a real, equal-valued card, :meth:`PlaceholderRegistry.is_placeholder`
can only ever answer by Python object identity: a genuine catalog card that
happens to equal the source card by value is NOT a placeholder.
"""

from __future__ import annotations

import collections.abc

from wingspan.cards import schema
from wingspan.cards.parse import catalog

# The two source cards every placeholder is a deep copy of -- resolved once
# from the catalog's stable ordering so a fresh interpreter always mints
# placeholders from the same source card.
_PLACEHOLDER_BIRD_SOURCE: schema.Bird = catalog.birds_ordered()[0]
_PLACEHOLDER_BONUS_SOURCE: schema.BonusCard = catalog.bonus_cards_ordered()[0]


class PlaceholderRegistry:
    """Mints and tracks placeholder cards for one aid session.

    Every minted instance is recorded so :meth:`is_placeholder` can answer
    by identity regardless of how far the card has traveled through the
    engine (hand, tray, discard, tucked)."""

    def __init__(self) -> None:
        self._instances: list[schema.Bird | schema.BonusCard] = []

    def mint_bird(self) -> schema.Bird:
        """A fresh identity-distinct placeholder bird."""
        placeholder = _PLACEHOLDER_BIRD_SOURCE.model_copy(deep=True)
        self._instances.append(placeholder)
        return placeholder

    def mint_bonus(self) -> schema.BonusCard:
        """A fresh identity-distinct placeholder bonus card."""
        placeholder = _PLACEHOLDER_BONUS_SOURCE.model_copy(deep=True)
        self._instances.append(placeholder)
        return placeholder

    def is_placeholder(self, card: schema.Bird | schema.BonusCard) -> bool:
        """Whether ``card`` is one of this registry's minted placeholders,
        by object identity -- not value equality (see the module
        docstring)."""
        return any(card is instance for instance in self._instances)

    def count_birds_in(self, hand: collections.abc.Sequence[schema.Bird]) -> int:
        """How many of ``hand`` are placeholder birds."""
        return sum(1 for bird in hand if self.is_placeholder(bird))

    def first_bird_in(
        self, hand: collections.abc.Sequence[schema.Bird]
    ) -> schema.Bird | None:
        """The first placeholder bird in ``hand``, or ``None`` if there is
        none."""
        for bird in hand:
            if self.is_placeholder(bird):
                return bird
        return None

    def swap_bird(self, hand: list[schema.Bird], real: schema.Bird) -> None:
        """Replace the first placeholder in ``hand`` with ``real``, in
        place. Raises ``ValueError`` if ``hand`` holds no placeholder."""
        for index, bird in enumerate(hand):
            if self.is_placeholder(bird):
                hand[index] = real
                return
        raise ValueError("hand has no placeholder bird to swap")

    def swap_bonus(
        self, cards_list: list[schema.BonusCard], real: schema.BonusCard
    ) -> None:
        """Replace the first placeholder in ``cards_list`` with ``real``, in
        place. Mirrors :meth:`swap_bird` for a player's bonus-card list.
        Raises ``ValueError`` if ``cards_list`` holds no placeholder."""
        for index, bonus_card in enumerate(cards_list):
            if self.is_placeholder(bonus_card):
                cards_list[index] = real
                return
        raise ValueError("cards_list has no placeholder bonus card to swap")
