"""Compact turn-state summary lines for the game log.

Provides :func:`log_turn_summary`, called once at the top of every turn from
``engine.core._take_turn``, to print a 4-5 line snapshot of the game state so
a reader of the log does not need to reconstruct context from the event stream.
"""

from __future__ import annotations

import typing

from wingspan import cards, state
from wingspan.engine import scoring

if typing.TYPE_CHECKING:
    from wingspan.engine import core

# Habitat labels kept short for the compact board line.
_HABITAT_ABBR: dict[cards.Habitat, str] = {
    cards.Habitat.FOREST: "forest",
    cards.Habitat.GRASSLAND: "grass",
    cards.Habitat.WETLAND: "wetland",
}

# Food abbreviations matching display._FOOD_ABBR so log and board share labels.
_FOOD_ABBR: dict[cards.Food, str] = {
    cards.Food.INVERTEBRATE: "inv",
    cards.Food.SEED: "seed",
    cards.Food.FISH: "fish",
    cards.Food.FRUIT: "berry",
    cards.Food.RODENT: "rod",
}

# Width to left-pad the tray column before appending the feeder section.
_TRAY_COL_WIDTH = 55


def log_turn_summary(engine: "core.Engine") -> None:
    """Emit a compact snapshot of the game state at the start of a turn.

    Writes up to 5 lines into the game log: the active player's board, hand,
    shared tray+feeder, score breakdown for both players, and the active
    player's bonus card status. Omits the bonus line when the player has no
    bonus cards yet (early-game or pre-bonus-pick state)."""
    gs = engine.state
    player = gs.me()

    engine.log(_board_line(player))
    engine.log(_hand_line(player))
    engine.log(_tray_feeder_line(gs))
    engine.log(_score_line(gs))
    bonus = _bonus_line(player)
    if bonus:
        engine.log(bonus)


###### PRIVATE #######


def _compact_pb(pb: state.PlayedBird) -> str:
    """Very compact bird-in-play rendering: Name(e0/2) with optional c/t flags."""
    extras: list[str] = []
    if pb.cached_food.total():
        extras.append("c")
    if pb.tucked_cards:
        extras.append(f"t{pb.tucked_cards}")
    suffix = "," + ",".join(extras) if extras else ""
    return f"{pb.bird.name}(e{pb.eggs}/{pb.bird.egg_limit}{suffix})"


def _board_line(player: state.Player) -> str:
    """Compact single-line board: each habitat in brackets with its birds."""
    parts: list[str] = []
    for habitat, abbr in _HABITAT_ABBR.items():
        row = player.board[habitat]
        if row:
            birds_str = " ".join(_compact_pb(pb) for pb in row)
            parts.append(f"{abbr}=[{birds_str}]")
        else:
            parts.append(f"{abbr}=[]")
    return "Board: " + " ".join(parts)


def _hand_line(player: state.Player) -> str:
    """Bird names in hand separated by pipes, count in parens."""
    if not player.hand:
        return "Hand (0): (empty)"
    names = " | ".join(bird.name for bird in player.hand)
    return f"Hand ({len(player.hand)}): {names}"


def _tray_feeder_line(gs: state.GameState) -> str:
    """Tray cards pipe-separated, then feeder food counts, on one line."""
    tray_parts = [bird.name if bird is not None else "(empty)" for bird in gs.tray]
    tray_col = "Tray: " + " | ".join(tray_parts)

    feeder_parts = [
        f"{abbr}={gs.birdfeeder.counts[food]}" for food, abbr in _FOOD_ABBR.items()
    ]
    if gs.birdfeeder.choice_dice:
        feeder_parts.append(f"+{gs.birdfeeder.choice_dice} choice")
    feeder_col = "Feeder: " + "  ".join(feeder_parts)

    return f"{tray_col:<{_TRAY_COL_WIDTH}} {feeder_col}"


def _player_score_cell(player: state.Player) -> str:
    """Compact score breakdown for one player."""
    bird_pts = sum(pb.bird.points for row in player.board.values() for pb in row)
    bonus_pts = sum(scoring.bonus_score(player, bc) for bc in player.bonus_cards)
    total = (
        bird_pts
        + bonus_pts
        + player.total_eggs
        + player.total_tucked
        + player.total_cached
        + player.round_goal_points
    )
    return (
        f"{player.name}={total}"
        f" (birds={bird_pts} eggs={player.total_eggs}"
        f" tuck={player.total_tucked} cache={player.total_cached}"
        f" bonus={bonus_pts} goals={player.round_goal_points})"
    )


def _score_line(gs: state.GameState) -> str:
    """Score breakdown for both players separated by a pipe."""
    cells = [_player_score_cell(player) for player in gs.players]
    return "Score: " + " | ".join(cells)


def _bonus_card_status(player: state.Player, bc: cards.BonusCard) -> str:
    """Status string for one bonus card: current VP and progress to next tier."""
    count = scoring.bonus_qualifying_count(player, bc)
    current_vp = scoring.bonus_score(player, bc)

    if bc.per_bird_vp is not None:
        return f"{bc.name}={current_vp}VP [{count}×{bc.per_bird_vp}/bird]"

    # Tiered: find the first threshold above the current count.
    for thr, next_vp in bc.thresholds:
        if count < thr:
            needed = thr - count
            return f"{bc.name}={current_vp}VP [{count} birds; {needed} more for {next_vp}VP]"

    return f"{bc.name}={current_vp}VP [{count} birds; max tier]"


def _bonus_line(player: state.Player) -> str:
    """One-line bonus card summary for the active player; empty string if none."""
    if not player.bonus_cards:
        return ""
    statuses = " | ".join(_bonus_card_status(player, bc) for bc in player.bonus_cards)
    return f"Bonus: {statuses}"
