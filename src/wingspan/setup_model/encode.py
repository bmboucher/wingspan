# pyright: reportPrivateUsage=false
# (reads the in-game encoder's package-private normalization scales in
# ``encode.layout`` — deliberate intra-package coupling, identical to the
# stripes modules' convention, so the two encoders can never drift apart)
"""The setup model's per-candidate feature encoder.

The setup model scores one setup candidate (a *keep* of cards / food / bonus) in
the context of the deal, and the policy is a softmax over the scores of all 504
candidates for a dealt hand. Its input is a flat vector of multi-hot / one-hot /
count / index blocks; the card-identity blocks are embedded *inside*
:class:`wingspan.training.setup_net.SetupNet` through frozen copies of the main
net's shared embedders (the kept-cards multi-hot through the multi-card set
encoder, the tray index columns through the single-card table), so the setup MLP
evaluates candidates in the same representation the in-game model learns.

:func:`encode_setup_candidate` builds one fixed-length vector from a
:class:`wingspan.setup_model.candidates.SetupCandidate` (the candidate-specific
part) and a :class:`SetupContext` (the shared per-deal context). The eight
blocks, in order, are:

1. multi-hot over every core-set bird — the cards kept
2. multi-hot over the five foods — the food kept
3. one-hot over every bonus card — the bonus card kept (all-zero if none)
4. three positional integer card indices — the tray slots (context);
   ``bird_index + 1`` per occupied slot, 0 for an empty one
5. a six-vector of birdfeeder die-face counts (the five foods + the choice die)
6. four one-hots — the four rounds' end-of-round goals (context)
7. the kept bonus card priced against the keep — kept-card qualifiers, the
   stepped / linear VP they would pay, tray potential (all-zero if no bonus)
8. per round goal, how many kept cards would advance its category if played

One player technically sees what the other kept at the start of the game; the
encoder ignores that (the context is each player's own view of the shared deal).
"""

from __future__ import annotations

import numpy as np
import pydantic

from wingspan import cards, encode, state
from wingspan.encode import layout
from wingspan.setup_model import candidates

# Round-goal one-hot width — the stable goal-category order the in-game encoder
# already pins, reused so the setup model's goal stripes line up with it.
SETUP_GOAL_DIM = encode.MAX_GOAL_CATEGORIES
# The four rounds' goals are encoded as four independent one-hots.
_NUM_SETUP_GOALS = len(state.ROUND_GOAL_PAYOUTS_2P)
# Birdfeeder block: one count per food plus the invertebrate/seed choice die.
_FEEDER_DIM = cards.N_FOODS + 1

# Block sizes and cumulative offsets — the contract the SetupNet's slicing and
# input width are derived from; nothing here may be reordered without
# invalidating saved weights. The offsets are public because the network (in
# ``wingspan.training``) splits the raw vector on them.
_KEPT_CARDS_DIM = cards.n_birds()
_KEPT_FOODS_DIM = cards.N_FOODS
_BONUS_DIM = cards.n_bonus_cards()
_TRAY_DIM = state.TRAY_SIZE
_GOALS_DIM = _NUM_SETUP_GOALS * SETUP_GOAL_DIM
_KEPT_BONUS_VALUE_DIM = 4
_GOAL_AFFINITY_DIM = _NUM_SETUP_GOALS

# Within-block indices of the kept-bonus pricing block.
_KEPT_BONUS_QUAL = 0
_KEPT_BONUS_STEPPED = 1
_KEPT_BONUS_LINEAR = 2
_KEPT_BONUS_TRAY = 3

OFF_KEPT_CARDS = 0
OFF_KEPT_FOODS = OFF_KEPT_CARDS + _KEPT_CARDS_DIM
OFF_BONUS = OFF_KEPT_FOODS + _KEPT_FOODS_DIM
OFF_TRAY = OFF_BONUS + _BONUS_DIM
OFF_FEEDER = OFF_TRAY + _TRAY_DIM
OFF_GOALS = OFF_FEEDER + _FEEDER_DIM
OFF_KEPT_BONUS_VALUE = OFF_GOALS + _GOALS_DIM
OFF_GOAL_AFFINITY = OFF_KEPT_BONUS_VALUE + _KEPT_BONUS_VALUE_DIM

SETUP_FEATURE_DIM = OFF_GOAL_AFFINITY + _GOAL_AFFINITY_DIM


class SetupContext(pydantic.BaseModel):
    """The shared per-deal context every candidate of a seat is scored against.

    Decoupled from :class:`wingspan.state.GameState` (and free of any non-Pydantic
    identity) so a worker process can build it from its own catalog rather than
    pickling live game objects across the pipe."""

    model_config = pydantic.ConfigDict(frozen=True)

    # The face-up tray in slot order (``None`` = an empty slot), so the encoding
    # is positional like the in-game state's tray index block.
    tray_birds: tuple[cards.Bird | None, ...]
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
            tray_birds=tuple(game_state.tray),
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
        vec[OFF_KEPT_CARDS + cards.bird_index(bird)] = 1.0
    for food in candidate.kept_foods:
        vec[OFF_KEPT_FOODS + cards.food_index(food)] = 1.0
    if candidate.bonus_card is not None:
        vec[OFF_BONUS + cards.bonus_index(candidate.bonus_card)] = 1.0

    # 4-5. Context blocks: positional tray card indices (0 = empty slot, the card
    # table's zeroed padding row) and birdfeeder die-face counts.
    for slot, bird in enumerate(context.tray_birds[:_TRAY_DIM]):
        if bird is not None:
            vec[OFF_TRAY + slot] = cards.bird_index(bird) + 1
    for offset, count in enumerate(context.birdfeeder_counts[:_FEEDER_DIM]):
        vec[OFF_FEEDER + offset] = float(count)

    # 6. Context block: one one-hot per round goal in the shared category order.
    for round_idx, category in enumerate(context.round_goal_categories):
        if round_idx >= _NUM_SETUP_GOALS or category not in encode.GOAL_CATEGORIES:
            continue
        base = OFF_GOALS + round_idx * SETUP_GOAL_DIM
        vec[base + encode.GOAL_CATEGORIES.index(category)] = 1.0

    # 7. The kept bonus priced against the keep, so the net reads what the
    # bonus is worth to *this* candidate instead of inferring it from the
    # identity one-hot alone. All-zero when no bonus is kept.
    if candidate.bonus_card is not None:
        _fill_kept_bonus_value(
            vec, candidate.bonus_card, candidate.kept_cards, context.tray_birds
        )

    # 8. Candidate × context: per round goal, the summed static category
    # affinity of the kept cards (how many would advance the goal if played).
    # Egg-driven goals are rightly 0 — nothing has eggs at setup time.
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    for round_idx, category in enumerate(
        context.round_goal_categories[:_NUM_SETUP_GOALS]
    ):
        affinity = sum(
            scoring.goal_count_delta_for_bird(bird, category)
            for bird in candidate.kept_cards
        )
        vec[OFF_GOAL_AFFINITY + round_idx] = affinity / layout._GOAL_COUNT_SCALE

    return vec


###### PRIVATE #######


def _fill_kept_bonus_value(
    vec: np.ndarray,
    bonus_card: cards.BonusCard,
    kept_cards: tuple[cards.Bird, ...],
    tray_birds: tuple[cards.Bird | None, ...],
) -> None:
    """Write the kept-bonus pricing block: the kept cards passing the bonus
    card's test (for the hand-counting dynamic card every kept card counts),
    the stepped / linear VP the card would pay if they all reach the board,
    and the tray birds that could still qualify it."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    if scoring.bonus_count_delta_for_hand(bonus_card, 1) > 0:
        kept_qual = len(kept_cards)
    else:
        kept_qual = sum(
            1 for bird in kept_cards if bonus_card.name in bird.bonus_categories
        )
    base = OFF_KEPT_BONUS_VALUE
    vec[base + _KEPT_BONUS_QUAL] = kept_qual / layout._BONUS_COUNT_SCALE
    vec[base + _KEPT_BONUS_STEPPED] = (
        scoring.bonus_score_for_count(bonus_card, kept_qual) / layout._BONUS_VALUE_SCALE
    )
    vec[base + _KEPT_BONUS_LINEAR] = (
        scoring.bonus_linear_value_for_count(bonus_card, kept_qual)
        / layout._BONUS_VALUE_SCALE
    )
    tray_qual = sum(
        1
        for tray_bird in tray_birds
        if tray_bird is not None and bonus_card.name in tray_bird.bonus_categories
    )
    vec[base + _KEPT_BONUS_TRAY] = tray_qual / layout._BONUS_COUNT_SCALE
