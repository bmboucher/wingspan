"""Tests for bonus-card qualification, scoring, and the setup help line.

These guard two bugs found in the bonus-card pipeline:

1. "Omnivore Specialist" is named differently in ``bonus.json`` than in the
   per-bird qualification column of ``master.json`` ("Omnivore Expert"), so
   no birds were tagged for it and it silently scored 0 VP. The alias in
   ``cards.parse`` resolves the rename.
2. Per-bird bonus cards ("N[point] per bird") produced no thresholds, so
   ``scoring.bonus_score`` returned 0 for every one of them. ``BonusCard``
   now carries ``per_bird_vp`` and scoring multiplies by the qualifying count.
"""

from __future__ import annotations

from wingspan import cards, state
from wingspan.agents import display
from wingspan.engine import scoring

# Core bonus cards whose qualification is dynamic (eggs laid, end-game hand
# size, board layout) and so name no birds by a fixed property. Everything
# else must tag at least one bird; this set is the allow-list for "0 tagged".
_DYNAMIC_BONUS_NAMES = {
    "Breeding Manager",
    "Ecologist",
    "Oologist",
    "Visionary Leader",
}


def _bonus(bonuses: list[cards.BonusCard], name: str) -> cards.BonusCard:
    return next(bonus for bonus in bonuses if bonus.name == name)


def _player_with_board(
    bonus: cards.BonusCard, board_birds: list[cards.Bird]
) -> state.Player:
    """A player holding ``bonus`` with ``board_birds`` placed in the forest."""
    player = state.Player(id=0, name="P0", bonus_cards=[bonus])
    player.board[cards.Habitat.FOREST] = [
        state.PlayedBird(bird=bird) for bird in board_birds
    ]
    return player


# --- Bug 1: the Omnivore Specialist column-name alias --------------------


def test_omnivore_specialist_tags_wild_cost_birds():
    """The renamed card tags exactly the birds with a wild symbol in their
    food cost — matching its printed "any bird that has a [wild] symbol"."""
    birds, _, _ = cards.load_all()
    tagged = {
        bird.name for bird in birds if "Omnivore Specialist" in bird.bonus_categories
    }
    wild_cost = {bird.name for bird in birds if bird.food_cost.wild > 0}
    assert tagged, "Omnivore Specialist should tag birds after the alias fix"
    assert tagged == wild_cost


def test_only_dynamic_bonus_cards_have_no_tagged_birds():
    """Regression guard: a future bonus-card rename that breaks the column
    lookup would surface here as a non-dynamic card with zero tagged birds."""
    birds, bonuses, _ = cards.load_all()
    untagged = {
        bonus.name
        for bonus in bonuses
        if not any(bonus.name in bird.bonus_categories for bird in birds)
    }
    assert untagged == _DYNAMIC_BONUS_NAMES


# --- Bug 2: per-bird payout parsing + scoring ----------------------------


def test_per_bird_payout_is_parsed():
    """Per-bird cards carry ``per_bird_vp`` and no thresholds; tiered cards
    carry thresholds and no ``per_bird_vp``."""
    _, bonuses, _ = cards.load_all()
    bird_counter = _bonus(bonuses, "Bird Counter")
    bird_feeder = _bonus(bonuses, "Bird Feeder")
    assert bird_counter.per_bird_vp == 2
    assert bird_counter.thresholds == ()
    assert bird_feeder.per_bird_vp is None
    assert bird_feeder.thresholds


def test_per_bird_bonus_scores_points_per_bird():
    """A per-bird card awards ``per_bird_vp`` for each qualifying bird."""
    birds, bonuses, _ = cards.load_all()
    bird_counter = _bonus(bonuses, "Bird Counter")
    flocking = [bird for bird in birds if "Bird Counter" in bird.bonus_categories][:3]
    player = _player_with_board(bird_counter, flocking)
    assert scoring.bonus_score(player, bird_counter) == 2 * len(flocking)


def test_omnivore_specialist_scores_end_to_end():
    """Both fixes together: a wild-cost board with Omnivore Specialist scores
    2 VP per qualifying bird instead of 0."""
    birds, bonuses, _ = cards.load_all()
    omnivore = _bonus(bonuses, "Omnivore Specialist")
    wild = [bird for bird in birds if bird.food_cost.wild > 0][:4]
    player = _player_with_board(omnivore, wild)
    assert scoring.bonus_score(player, omnivore) == 2 * len(wild)


def test_tiered_bonus_still_scores():
    """The threshold path is unchanged: a tiered card awards the highest
    matched band (Bird Feeder: 5-7 seed birds -> 3 VP)."""
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _bonus(bonuses, "Bird Feeder")
    seed = [bird for bird in birds if "Bird Feeder" in bird.bonus_categories][:6]
    player = _player_with_board(bird_feeder, seed)
    assert scoring.bonus_score(player, bird_feeder) == 3


def test_tiered_bonus_scores_higher_plateau():
    """The upper band triggers once the higher threshold is met (Bird Feeder:
    8+ seed birds -> 7 VP, not the 3 VP of the lower band)."""
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _bonus(bonuses, "Bird Feeder")
    seed = [bird for bird in birds if "Bird Feeder" in bird.bonus_categories][:8]
    assert len(seed) == 8
    player = _player_with_board(bird_feeder, seed)
    assert scoring.bonus_score(player, bird_feeder) == 7


def test_multiple_bonus_cards_sum():
    """A player may hold more than one bonus card; the running score sums each
    card's payout. Bird Feeder (tiered) + Bird Counter (per-bird) are scored on
    disjoint qualifying-bird sets placed in two habitats."""
    birds, bonuses, _ = cards.load_all()
    bird_feeder = _bonus(bonuses, "Bird Feeder")
    bird_counter = _bonus(bonuses, "Bird Counter")
    # Disjoint qualifying sets so each card's count (and thus VP) is exact.
    feeder_birds = [
        bird
        for bird in birds
        if "Bird Feeder" in bird.bonus_categories
        and "Bird Counter" not in bird.bonus_categories
    ][:6]
    counter_birds = [
        bird
        for bird in birds
        if "Bird Counter" in bird.bonus_categories
        and "Bird Feeder" not in bird.bonus_categories
    ][:3]
    assert len(feeder_birds) == 6
    assert len(counter_birds) == 3

    player = state.Player(id=0, name="P0", bonus_cards=[bird_feeder, bird_counter])
    player.board[cards.Habitat.FOREST] = [
        state.PlayedBird(bird=bird) for bird in feeder_birds
    ]
    player.board[cards.Habitat.GRASSLAND] = [
        state.PlayedBird(bird=bird) for bird in counter_birds
    ]

    assert scoring.bonus_score(player, bird_feeder) == 3  # 6 birds -> lower band
    assert scoring.bonus_score(player, bird_counter) == 2 * 3  # per-bird
    assert sum(scoring.bonus_score(player, bc) for bc in player.bonus_cards) == 9


# --- Setup help line: per-bird cards now assessable, with deck share ------


def test_setup_help_line_shows_counts_and_deck_share():
    """A type-counting card's pre-game line reports selected/total in hand,
    the display count, and the share of the whole core-set deck."""
    birds, bonuses, _ = cards.load_all()
    omnivore = _bonus(bonuses, "Omnivore Specialist")
    wild = [bird for bird in birds if bird.food_cost.wild > 0]
    expected_pct = round(100 * len(wild) / len(birds))

    rendered = display.format_bonus_with_setup_help(
        omnivore,
        hand_birds=wild[:1],
        tray_birds=wild[1:3],
        selected_hand_birds=wild[:1],
    )
    help_line = rendered.splitlines()[1]
    assert "1/1 in hand" in help_line
    assert "2 in the display" in help_line
    assert f"{expected_pct}% of all birds" in help_line


def test_setup_help_line_absent_for_dynamic_card():
    """Dynamic cards can't be assessed pre-game, so they get no second line."""
    birds, bonuses, _ = cards.load_all()
    oologist = _bonus(bonuses, "Oologist")
    rendered = display.format_bonus_with_setup_help(
        oologist, hand_birds=birds[:2], tray_birds=birds[2:4], selected_hand_birds=[]
    )
    assert "\n" not in rendered
    assert "% of all birds" not in rendered
