"""The random-setup generation procedure (the pre-setup-model training phase).

Before the setup model is trained (and as the opponent's setup forever in the
vs-random bootstrap), setups are not chosen by any policy — they are drawn by a
seeded procedure designed to explore the joint setup space densely over a *fixed*
deal while biasing away from obviously-wasteful food keeps. For one deal a
:class:`RandomSetupGenerator` batch:

1. samples ``hand_combos`` joint ``(seat-0 keep, seat-1 keep)`` card subsets,
2. for each seat's kept hand, samples up to ``food_sets`` food keeps, biased by a
   softmax over how many of the kept-hand and tray birds that food could pay for
   (``1·hand + 0.5·tray``), so players rarely keep food that strands their birds,
3. pairs each seat's keep with one of its two dealt bonus cards,
4. cross-products the per-seat candidates and samples ``tuples_per_batch`` joint
   setups.

Each sampled joint tuple becomes one game (played from that fixed setup with an
independent in-game continuation), so a batch compares many keeps over one deal.
The procedure is a pure function of its ``rng``, so a seed reproduces the same
batch in any worker process.

When ``split_food=True`` (the ``split_setup_food`` regime), step 2 is skipped
entirely: candidates always carry ``kept_foods=()``, and food is resolved by the
engine via sequential in-game GAIN_FOOD / SPEND_FOOD decisions after card-keep.
"""

from __future__ import annotations

import itertools
import random

import numpy as np

from wingspan import cards, sampling, state
from wingspan.engine import helpers
from wingspan.setup_model import candidates, encode

# Per-seat dealt input: the five dealt cards and the two dealt bonus cards.
type SeatDeal = tuple[list[cards.Bird], list[cards.BonusCard]]
# A joint setup: the seat-0 and seat-1 decided keeps for one game.
type JointSetup = tuple[candidates.SetupCandidate, candidates.SetupCandidate]


class RandomSetupGenerator:
    """Seeded generator of joint random setups for a deal (see module docstring).

    When ``split_food=True`` (the ``split_setup_food`` regime) the food axis is
    omitted entirely: ``_seat_candidates`` returns one candidate per bonus option
    with ``kept_foods=()``, and ``setup_food_sets`` has no effect."""

    def __init__(
        self,
        hand_combos: int,
        food_sets: int,
        tuples_per_batch: int = 16,
        split_food: bool = False,
    ):
        self.hand_combos = hand_combos
        self.food_sets = food_sets
        self.tuples_per_batch = tuples_per_batch
        self.split_food = split_food

    def generate(
        self,
        rng: random.Random,
        dealt: tuple[SeatDeal, SeatDeal],
        context: encode.SetupContext,
    ) -> list[JointSetup]:
        """Sample up to ``tuples_per_batch`` joint setups over a fixed deal."""
        tray = [bird for bird in context.tray_birds if bird is not None]
        joint: list[JointSetup] = []
        for _ in range(self.hand_combos):
            cands_0 = self._seat_candidates(rng, dealt[0], tray)
            cands_1 = self._seat_candidates(rng, dealt[1], tray)
            for cand_0 in cands_0:
                for cand_1 in cands_1:
                    joint.append((cand_0, cand_1))
        if len(joint) <= self.tuples_per_batch:
            return joint
        return rng.sample(joint, self.tuples_per_batch)

    def generate_one(
        self,
        rng: random.Random,
        seat_deal: SeatDeal,
        context: encode.SetupContext,
    ) -> candidates.SetupCandidate:
        """One random keep for a single seat — the setup the random opponent uses
        once the AI seats have switched to the setup model. Food-aware unless
        ``split_food`` is set, in which case food is omitted (``kept_foods=()``)."""
        return rng.choice(
            self._seat_candidates(
                rng,
                seat_deal,
                [bird for bird in context.tray_birds if bird is not None],
            )
        )

    ###### PRIVATE #######

    def _seat_candidates(
        self, rng: random.Random, seat_deal: SeatDeal, tray: list[cards.Bird]
    ) -> list[candidates.SetupCandidate]:
        """The candidate keeps for one seat under a freshly-sampled card subset.

        Normal mode: food-biased food keeps × bonus cards.
        Split-food mode: a single deferred sentinel (``kept_foods=()``) × bonus cards.
        """
        dealt_cards, dealt_bonus = seat_deal
        kept_cards = self._random_keep(rng, dealt_cards)
        bonus_options: list[cards.BonusCard | None] = (
            list(dealt_bonus) if dealt_bonus else [None]
        )

        # In split-food mode skip food sampling entirely — food resolves in-game.
        if self.split_food:
            return [
                candidates.SetupCandidate(
                    kept_cards=kept_cards, kept_foods=(), bonus_card=bonus_card
                )
                for bonus_card in bonus_options
            ]

        kept_food_size = cards.N_FOODS - len(kept_cards)
        food_options = self._food_options(rng, kept_cards, kept_food_size, tray)
        return [
            candidates.SetupCandidate(
                kept_cards=kept_cards, kept_foods=foods, bonus_card=bonus_card
            )
            for foods in food_options
            for bonus_card in bonus_options
        ]

    @staticmethod
    def _random_keep(
        rng: random.Random, dealt_cards: list[cards.Bird]
    ) -> tuple[cards.Bird, ...]:
        """A uniformly-random subset of the dealt cards (one of the 2^n masks)."""
        num_cards = len(dealt_cards)
        mask = rng.randrange(1 << num_cards)
        return tuple(dealt_cards[i] for i in range(num_cards) if mask & (1 << i))

    def _food_options(
        self,
        rng: random.Random,
        kept_cards: tuple[cards.Bird, ...],
        kept_food_size: int,
        tray: list[cards.Bird],
    ) -> list[tuple[cards.Food, ...]]:
        """Sample up to ``food_sets`` food keeps of the required size, softmax-
        biased toward sets that pay for more of the kept-hand and tray birds."""
        combos = [
            tuple(combo)
            for combo in itertools.combinations(cards.ALL_FOODS, kept_food_size)
        ]
        if len(combos) <= self.food_sets:
            return combos
        scores = np.array(
            [self._food_score(combo, kept_cards, tray) for combo in combos],
            dtype=np.float64,
        )
        chosen = _softmax_sample_without_replacement(rng, scores, self.food_sets)
        return [combos[i] for i in chosen]

    @staticmethod
    def _food_score(
        food_combo: tuple[cards.Food, ...],
        kept_cards: tuple[cards.Bird, ...],
        tray: list[cards.Bird],
    ) -> float:
        """``1·(kept-hand birds payable) + 0.5·(tray birds payable)`` for a food
        set of one of each listed food — the keep-quality heuristic the bias
        prefers (so a keep rarely strands its own birds)."""
        pool = state.FoodPool.from_dict({food: 1 for food in food_combo})
        hand_payable = sum(
            1 for bird in kept_cards if helpers.any_payment_exists(pool, bird.food_cost)
        )
        tray_payable = sum(
            1 for bird in tray if helpers.any_payment_exists(pool, bird.food_cost)
        )
        return hand_payable + 0.5 * tray_payable


def _softmax_sample_without_replacement(
    rng: random.Random, scores: np.ndarray, count: int
) -> list[int]:
    """Sample ``count`` distinct indices in proportion to ``softmax(scores)``,
    drawing one at a time and removing each pick so the result has no repeats."""
    weights = np.exp(scores - scores.max())
    remaining = list(range(len(scores)))
    chosen: list[int] = []
    for _ in range(min(count, len(remaining))):
        sub = np.array([weights[i] for i in remaining], dtype=np.float64)
        total = float(sub.sum())
        if not np.isfinite(total) or total <= 0.0:
            pick_pos = rng.randrange(len(remaining))
        else:
            pick_pos = sampling.weighted_index(rng, (sub / total).tolist())
        chosen.append(remaining.pop(pick_pos))
    return chosen
