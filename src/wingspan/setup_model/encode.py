"""The setup model's per-candidate feature encoder.

The setup model scores one setup candidate (a *keep* of cards / food / bonus) in
the context of the deal, and the policy is a softmax over the scores of all 504
candidates for a dealt hand. Its input is deliberately far simpler than the
in-game encoder (:mod:`wingspan.encode`): plain multi-hot / one-hot / count
blocks straight into an MLP, with no shared card embedding.

:func:`encode_setup_candidate` builds one fixed-length vector from a
:class:`wingspan.setup_model.candidates.SetupCandidate` (the candidate-specific
part) and a :class:`SetupContext` (the shared per-deal context). The six blocks,
in order, are:

1. multi-hot over every core-set bird — the cards kept
2. multi-hot over the five foods — the food kept
3. one-hot over every bonus card — the bonus card kept (all-zero if none)
4. multi-hot over every core-set bird — the cards in the tray (context)
5. a six-vector of birdfeeder die-face counts (the five foods + the choice die)
6. four one-hots — the four rounds' end-of-round goals (context)

One player technically sees what the other kept at the start of the game; the
encoder ignores that (the context is each player's own view of the shared deal).
"""

from __future__ import annotations

import numpy as np
import pydantic

from wingspan import cards, encode, state
from wingspan.setup_model import candidates

# Round-goal one-hot width — the stable goal-category order the in-game encoder
# already pins, reused so the setup model's goal stripes line up with it.
SETUP_GOAL_DIM = encode.MAX_GOAL_CATEGORIES
# The four rounds' goals are encoded as four independent one-hots.
_NUM_SETUP_GOALS = len(state.ROUND_GOAL_PAYOUTS_2P)
# Birdfeeder block: one count per food plus the invertebrate/seed choice die.
_FEEDER_DIM = cards.N_FOODS + 1

# Block sizes and cumulative offsets — the contract the SetupNet's input width is
# derived from; nothing here may be reordered without invalidating saved weights.
_KEPT_CARDS_DIM = cards.n_birds()
_KEPT_FOODS_DIM = cards.N_FOODS
_BONUS_DIM = cards.n_bonus_cards()
_TRAY_DIM = cards.n_birds()
_GOALS_DIM = _NUM_SETUP_GOALS * SETUP_GOAL_DIM

_OFF_KEPT_CARDS = 0
_OFF_KEPT_FOODS = _OFF_KEPT_CARDS + _KEPT_CARDS_DIM
_OFF_BONUS = _OFF_KEPT_FOODS + _KEPT_FOODS_DIM
_OFF_TRAY = _OFF_BONUS + _BONUS_DIM
_OFF_FEEDER = _OFF_TRAY + _TRAY_DIM
_OFF_GOALS = _OFF_FEEDER + _FEEDER_DIM

SETUP_FEATURE_DIM = _OFF_GOALS + _GOALS_DIM


class SetupContext(pydantic.BaseModel):
    """The shared per-deal context every candidate of a seat is scored against.

    Decoupled from :class:`wingspan.state.GameState` (and free of any non-Pydantic
    identity) so a worker process can build it from its own catalog rather than
    pickling live game objects across the pipe."""

    model_config = pydantic.ConfigDict(frozen=True)

    tray_birds: tuple[cards.Bird, ...]
    # One count per food (``cards.ALL_FOODS`` order) then the choice-die count.
    birdfeeder_counts: tuple[int, ...]
    # The four rounds' goal categories (a tag in ``encode.GOAL_CATEGORIES``).
    round_goal_categories: tuple[str, ...]

    @classmethod
    def from_state(cls, game_state: state.GameState) -> "SetupContext":
        """Read the tray, birdfeeder, and round goals out of a fresh post-deal
        ``GameState`` into a frozen, picklable context."""
        feeder = [game_state.birdfeeder.counts[food] for food in cards.ALL_FOODS]
        feeder.append(game_state.birdfeeder.choice_dice)
        return cls(
            tray_birds=tuple(b for b in game_state.tray if b is not None),
            birdfeeder_counts=tuple(feeder),
            round_goal_categories=tuple(
                goal.category for goal in game_state.round_goals[:_NUM_SETUP_GOALS]
            ),
        )


def encode_setup_candidate(
    candidate: candidates.SetupCandidate, context: SetupContext
) -> np.ndarray:
    """Build the fixed-length feature vector for one setup candidate in context."""
    vec = np.zeros(SETUP_FEATURE_DIM, dtype=np.float32)

    # 1-3. Candidate-specific blocks: kept cards / kept foods / kept bonus.
    for bird in candidate.kept_cards:
        vec[_OFF_KEPT_CARDS + cards.bird_index(bird)] = 1.0
    for food in candidate.kept_foods:
        vec[_OFF_KEPT_FOODS + cards.food_index(food)] = 1.0
    if candidate.bonus_card is not None:
        vec[_OFF_BONUS + cards.bonus_index(candidate.bonus_card)] = 1.0

    # 4-5. Context blocks: tray identities and birdfeeder die-face counts.
    for bird in context.tray_birds:
        vec[_OFF_TRAY + cards.bird_index(bird)] = 1.0
    for offset, count in enumerate(context.birdfeeder_counts[:_FEEDER_DIM]):
        vec[_OFF_FEEDER + offset] = float(count)

    # 6. Context block: one one-hot per round goal in the shared category order.
    for round_idx, category in enumerate(context.round_goal_categories):
        if round_idx >= _NUM_SETUP_GOALS or category not in encode.GOAL_CATEGORIES:
            continue
        base = _OFF_GOALS + round_idx * SETUP_GOAL_DIM
        vec[base + encode.GOAL_CATEGORIES.index(category)] = 1.0

    return vec
