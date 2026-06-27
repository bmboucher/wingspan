"""Setup candidates: the keep options the setup model scores, plus selection.

A :class:`SetupCandidate` is one decided setup for a single seat — which dealt
cards to keep, which food to keep, which bonus card to keep — decoupled from
:class:`wingspan.decisions.SetupChoice` so it can be built and reasoned about
without an engine (the random generator and the worker processes both need that).

:func:`enumerate_setup_candidates` produces the same 504-candidate set (and the
same order) that ``engine.core.Engine._build_setup_choices`` offers, so the
setup model scores exactly the options the engine would present.
:func:`select_by_margins` turns the model's per-candidate predicted margins into
a chosen index — a softmax sample during collection, the argmax at evaluation.
"""

from __future__ import annotations

import itertools
import random

import numpy as np
import pydantic

from wingspan import cards, decisions, sampling


class SetupCandidate(pydantic.BaseModel):
    """One seat's decided setup keep, independent of the engine.

    The player starts with one of each food; keeping a card costs one food, so
    ``kept_foods`` is the subset of foods retained after paying for
    ``kept_cards`` (size ``len(cards.ALL_FOODS) - len(kept_cards)``)."""

    model_config = pydantic.ConfigDict(frozen=True)

    kept_cards: tuple[cards.Bird, ...]
    kept_foods: tuple[cards.Food, ...]
    bonus_card: cards.BonusCard | None

    def to_setup_choice(self) -> decisions.SetupChoice:
        """The equivalent :class:`wingspan.decisions.SetupChoice` (label rendered
        lazily, like the engine's enumeration), for applying this keep to a
        player via the engine's ``_apply_setup_choice``."""
        return decisions.SetupChoice.model_construct(
            kept_cards=self.kept_cards,
            kept_foods=self.kept_foods,
            bonus_card=self.bonus_card,
        )

    @classmethod
    def from_setup_choice(cls, choice: decisions.SetupChoice) -> "SetupCandidate":
        """The inverse of :meth:`to_setup_choice`."""
        return cls(
            kept_cards=choice.kept_cards,
            kept_foods=choice.kept_foods,
            bonus_card=choice.bonus_card,
        )


def enumerate_setup_candidates(
    dealt_cards: list[cards.Bird],
    dealt_bonus: list[cards.BonusCard],
    *,
    include_bonus: bool = True,
    include_food: bool = True,
) -> list[SetupCandidate]:
    """Every legal setup keep for a deal, in the engine's ``(kept_mask,
    kept_food_combo, bonus)`` order — 504 for the standard 5-card / 2-bonus deal.

    Mirrors ``Engine._build_setup_choices`` exactly so the setup model's softmax
    runs over the same candidate set the engine offers an agent.

    ``include_bonus=False`` drops the bonus axis entirely: every candidate carries
    ``bonus_card=None`` (the ``split_setup_bonus`` regime, where the bonus is
    instead chosen by the in-game ``CHOOSE_BONUS`` head). That halves the count to
    the distinct ``(kept_mask, kept_food_combo)`` keeps — 252 for the standard
    deal — while preserving their order.

    ``include_food=False`` drops the food axis (the ``split_setup_food`` regime,
    where food is resolved by sequential in-game GAIN_FOOD/SPEND_FOOD decisions
    after the card-keep applies). Every candidate carries ``kept_foods=()`` and
    the food block of its feature vector is all-zero. With both axes dropped the
    count is 32 (one per card-keep mask for the standard 5-card deal)."""
    num_cards = len(dealt_cards)
    all_foods = list(cards.ALL_FOODS)
    bonuses: list[cards.BonusCard | None]
    if include_bonus:
        bonuses = list(dealt_bonus) if dealt_bonus else [None]
    else:
        bonuses = [None]
    out: list[SetupCandidate] = []
    for mask in range(1 << num_cards):
        kept = tuple(dealt_cards[i] for i in range(num_cards) if mask & (1 << i))

        # Food axis: either enumerate legal food-keep subsets or a single deferred
        # sentinel (kept_foods=()) that zeros the food block in the feature vector.
        if include_food:
            food_options: list[tuple[cards.Food, ...]] = list(
                itertools.combinations(all_foods, len(all_foods) - len(kept))
            )
        else:
            food_options = [()]

        for food_combo in food_options:
            for bonus_card in bonuses:
                out.append(
                    SetupCandidate(
                        kept_cards=kept,
                        kept_foods=food_combo,
                        bonus_card=bonus_card,
                    )
                )
    return out


def select_by_margins(
    margins: np.ndarray, temperature: float, rng: random.Random | None
) -> int:
    """Pick a candidate index from per-candidate predicted margins.

    ``rng is None`` takes the argmax (greedy strength play, for evaluation);
    otherwise a softmax over ``margins / temperature`` is sampled with the seeded
    ``rng`` (the on-policy exploration used during collection). Falls back to a
    uniform pick when the distribution is degenerate."""
    if rng is None:
        return int(np.argmax(margins))
    scaled = margins.astype(np.float64) / max(temperature, 1e-6)
    scaled -= scaled.max()
    weights = np.exp(scaled)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        return rng.randrange(len(margins))
    return sampling.weighted_index(rng, (weights / total).tolist())


###### PRIVATE #######
