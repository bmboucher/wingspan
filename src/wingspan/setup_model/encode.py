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

:func:`encode_setup_candidate` builds a variable-length vector whose stripes
depend on the active :class:`~wingspan.setup_model.architecture.SetupEncoding`
configuration. The always-present blocks are, in order:

1. multi-hot over every core-set bird — the cards kept
2. multi-hot over the five foods — the food kept (omitted when
   ``encoding.split_food`` is active)
3. **either** kept-bonus block (when ``split_bonus`` is off) **or** available-
   bonus block (when ``split_bonus`` is on):

   * **split_bonus=False**: one-hot over every bonus card — the bonus card kept
     (all-zero if none), followed by kept-bonus pricing (4-vector: qual count,
     stepped VP, linear VP, tray potential)
   * **split_bonus=True**: multi-hot over dealt bonus cards (which are on offer),
     followed by ``[min_bonus_card_affinity, max_bonus_card_affinity]`` — the
     min/max of each dealt bonus's qualifier count against the kept cards (both
     normalized ÷ 5)

4. three positional integer card indices — the tray slots (context)
5. a six-vector of birdfeeder die-face counts (the five foods + the choice die)
6. four one-hots — the four rounds' end-of-round goals (context)
7. per round goal, how many kept cards would advance its category if played
8. turn-1-playable multi-hot (only when ``include_turn1_playable``) — kept cards
   playable on turn 1 given concrete ``kept_foods``
9. playable-kept-cards multi-hot (only when ``include_playable_kept_cards``) —
   kept cards for which *some* keepable food set would allow turn-1 play

One player technically sees what the other kept at the start of the game; the
encoder ignores that (the context is each player's own view of the shared deal).

The legacy module-level ``OFF_*`` constants and ``SETUP_FEATURE_DIM`` remain for
the default-encoding (both splits off, 308 dims) case and for backward-compatible
deserialization of pre-0.2 artifacts.
"""

from __future__ import annotations

import numpy as np
import pydantic

from wingspan import cards, encode, state
from wingspan.encode import layout
from wingspan.setup_model import architecture as arch_module
from wingspan.setup_model import candidates

# Round-goal one-hot width — the stable goal-category order the in-game encoder
# already pins, reused so the setup model's goal stripes line up with it.
SETUP_GOAL_DIM = encode.MAX_GOAL_CATEGORIES
# The four rounds' goals are encoded as four independent one-hots.
_NUM_SETUP_GOALS = len(state.ROUND_GOAL_PAYOUTS_2P)
# Birdfeeder block: one count per food plus the invertebrate/seed choice die.
_FEEDER_DIM = cards.N_FOODS + 1

# Block sizes mirrored from architecture (the single source of truth).
_KEPT_CARDS_DIM = arch_module._KEPT_CARDS_DIM
_KEPT_FOODS_DIM = arch_module._KEPT_FOODS_DIM
_BONUS_DIM = arch_module._BONUS_DIM
_TRAY_DIM = state.TRAY_SIZE
_GOALS_DIM = _NUM_SETUP_GOALS * SETUP_GOAL_DIM
_KEPT_BONUS_VALUE_DIM = arch_module._KEPT_BONUS_VALUE_DIM
_GOAL_AFFINITY_DIM = arch_module._GOAL_AFFINITY_DIM

# Within-block indices of the kept-bonus pricing block.
_KEPT_BONUS_QUAL = 0
_KEPT_BONUS_STEPPED = 1
_KEPT_BONUS_LINEAR = 2
_KEPT_BONUS_TRAY = 3

# Legacy default-encoding offsets (both splits off, 308 dims) — kept for
# backward-compat deserialization and as the SetupNet's default slice points.
# New code should prefer SetupEncoding properties.
# Layout: kept_cards → kept_foods → kept_bonus → tray → feeder → goals
#         → kept_bonus_value → goal_affinity
OFF_KEPT_CARDS = 0
OFF_KEPT_FOODS = OFF_KEPT_CARDS + _KEPT_CARDS_DIM
OFF_BONUS = OFF_KEPT_FOODS + _KEPT_FOODS_DIM
OFF_TRAY = OFF_BONUS + _BONUS_DIM
OFF_FEEDER = OFF_TRAY + _TRAY_DIM
OFF_GOALS = OFF_FEEDER + _FEEDER_DIM
OFF_KEPT_BONUS_VALUE = OFF_GOALS + _GOALS_DIM
OFF_GOAL_AFFINITY = OFF_KEPT_BONUS_VALUE + _KEPT_BONUS_VALUE_DIM

SETUP_FEATURE_DIM = OFF_GOAL_AFFINITY + _GOAL_AFFINITY_DIM

# Normalization scale for the bonus-card affinity scalars.
_BONUS_AFFINITY_SCALE = layout._BONUS_COUNT_SCALE


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
    # The bonus cards dealt to this seat; needed for the split-bonus encoding
    # (``bonus_cards`` multi-hot + affinity stripes).  Empty tuple for callers
    # that predate the split-bonus path (backward compat).
    dealt_bonus_cards: tuple[cards.BonusCard, ...] = ()

    @classmethod
    def from_state(
        cls,
        game_state: state.GameState,
        dealt_bonus: list[cards.BonusCard] | None = None,
    ) -> "SetupContext":
        """Read the tray, birdfeeder, and round goals out of a fresh post-deal
        ``GameState`` into a frozen, picklable context.

        ``dealt_bonus`` should be the bonus cards available to this seat; pass
        it whenever the ``split_setup_bonus`` encoding is active so the
        ``bonus_cards`` multi-hot and affinity stripes are filled correctly."""
        feeder = [game_state.birdfeeder.counts[food] for food in cards.ALL_FOODS]
        feeder.append(game_state.birdfeeder.choice_dice)
        return cls(
            tray_birds=tuple(game_state.tray),
            birdfeeder_counts=tuple(feeder),
            round_goal_categories=tuple(
                goal.category for goal in game_state.round_goals[:_NUM_SETUP_GOALS]
            ),
            dealt_bonus_cards=tuple(dealt_bonus) if dealt_bonus else (),
        )


def encode_setup_candidate(
    candidate: candidates.SetupCandidate,
    context: SetupContext,
    encoding: arch_module.SetupEncoding | None = None,
) -> np.ndarray:
    """Build the feature vector for one setup candidate in context.

    ``encoding`` selects the active layout; defaults to ``SetupEncoding()``
    which reproduces the legacy 308-dim all-splits-off vector.
    """
    if encoding is None:
        encoding = arch_module.SetupEncoding()
    vec = np.zeros(encoding.total_dim, dtype=np.float32)

    # 1. Kept cards: always at offset 0, multi-hot over all core-set birds.
    for bird in candidate.kept_cards:
        vec[encoding.off_kept_cards + cards.bird_index(bird)] = 1.0

    # 2. Kept foods: present only when food is not deferred.
    if not encoding.split_food:
        off_foods = encoding.off_bonus_block - _KEPT_FOODS_DIM
        for food in candidate.kept_foods:
            vec[off_foods + cards.food_index(food)] = 1.0

    # 3. Bonus block: two shapes depending on whether the bonus is deferred.
    off_bonus = encoding.off_bonus_block
    if not encoding.split_bonus:
        # Kept-bonus one-hot only (26 dims); kept_bonus_value comes after goals.
        if candidate.bonus_card is not None:
            vec[off_bonus + cards.bonus_index(candidate.bonus_card)] = 1.0
    else:
        # Available-bonus multi-hot (26 dims) + min/max affinity (2 dims).
        for bonus_card in context.dealt_bonus_cards:
            vec[off_bonus + cards.bonus_index(bonus_card)] = 1.0
        _fill_bonus_card_affinity(
            vec,
            off_bonus + _BONUS_DIM,
            context.dealt_bonus_cards,
            candidate.kept_cards,
        )

    # 4. Tray: positional integer card indices (bird_index + 1; 0 = empty).
    for slot, bird in enumerate(context.tray_birds[:_TRAY_DIM]):
        if bird is not None:
            vec[encoding.off_tray + slot] = cards.bird_index(bird) + 1

    # 5. Birdfeeder: raw die-face counts — NOT normalized, unlike the state vector.
    for offset, count in enumerate(context.birdfeeder_counts[:_FEEDER_DIM]):
        vec[encoding.off_feeder + offset] = float(count)

    # 6. Round goals: one one-hot per round in the shared category order.
    for round_idx, category in enumerate(context.round_goal_categories):
        if round_idx >= _NUM_SETUP_GOALS or category not in encode.GOAL_CATEGORIES:
            continue
        base = encoding.off_goals + round_idx * SETUP_GOAL_DIM
        vec[base + encode.GOAL_CATEGORIES.index(category)] = 1.0

    # 7. Kept-bonus value pricing (only when not split): qual count, stepped VP,
    # linear VP, tray potential. Placed after goals so the tray offset is stable
    # whether or not split_bonus is active.
    if not encoding.split_bonus and candidate.bonus_card is not None:
        _fill_kept_bonus_value(
            vec,
            encoding.off_bonus_value,
            candidate.bonus_card,
            candidate.kept_cards,
            context.tray_birds,
        )

    # 8. Goal affinity: per-round summed static affinity of kept cards.
    # Egg-driven goals are rightly 0 — nothing has eggs at setup time.
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    for round_idx, category in enumerate(
        context.round_goal_categories[:_NUM_SETUP_GOALS]
    ):
        affinity = sum(
            scoring.goal_count_delta_for_bird(bird, category)
            for bird in candidate.kept_cards
        )
        vec[encoding.off_goal_affinity + round_idx] = (
            affinity / layout._GOAL_COUNT_SCALE
        )

    # 8. Turn-1 playability multi-hot (only when include_turn1_playable): which
    # kept cards could be played on turn 1 given the kept foods.
    if encoding.include_turn1_playable:
        from wingspan.engine import playability as _playability

        playable = _playability.setup_turn1_playable(
            candidate.kept_cards, candidate.kept_foods
        )
        for bird in playable:
            vec[encoding.off_turn1_playable + cards.bird_index(bird)] = 1.0

    # 9. Food-agnostic playability multi-hot (only when include_playable_kept_cards):
    # which kept cards could be played given *some* keepable food set.
    if encoding.include_playable_kept_cards:
        from wingspan.engine import playability as _playability

        playable_kept = _playability.setup_playable_kept_cards(candidate.kept_cards)
        for bird in playable_kept:
            vec[encoding.off_playable_kept_cards + cards.bird_index(bird)] = 1.0

    return vec


###### PRIVATE #######


def _kept_qual_for_bonus(
    bonus_card: cards.BonusCard, kept_cards: tuple[cards.Bird, ...]
) -> int:
    """Number of ``kept_cards`` that qualify ``bonus_card``."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    if scoring.bonus_count_delta_for_hand(bonus_card, 1) > 0:
        return len(kept_cards)
    return sum(1 for bird in kept_cards if bonus_card.name in bird.bonus_categories)


def _fill_kept_bonus_value(
    vec: np.ndarray,
    base: int,
    bonus_card: cards.BonusCard,
    kept_cards: tuple[cards.Bird, ...],
    tray_birds: tuple[cards.Bird | None, ...],
) -> None:
    """Write the kept-bonus pricing 4-vector at ``base``."""
    from wingspan.engine import scoring  # local: keeps encode engine-free at import

    kept_qual = _kept_qual_for_bonus(bonus_card, kept_cards)
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


def _fill_bonus_card_affinity(
    vec: np.ndarray,
    base: int,
    dealt_bonus_cards: tuple[cards.BonusCard, ...],
    kept_cards: tuple[cards.Bird, ...],
) -> None:
    """Write ``[min_bonus_card_affinity, max_bonus_card_affinity]`` at ``base``.

    For each dealt bonus card computes how many kept cards qualify it, then
    writes the min and max of those counts normalized ÷ 5.  When fewer than
    two bonus cards are dealt both values equal the single card's count."""
    counts = [
        _kept_qual_for_bonus(bonus_card, kept_cards) for bonus_card in dealt_bonus_cards
    ]
    if not counts:
        return
    vec[base] = min(counts) / _BONUS_AFFINITY_SCALE
    vec[base + 1] = max(counts) / _BONUS_AFFINITY_SCALE
