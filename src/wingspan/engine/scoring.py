"""End-of-round goal scoring and end-of-game final scoring."""

from __future__ import annotations

import typing

import pydantic

from wingspan import cards, state

if typing.TYPE_CHECKING:
    from wingspan.engine import core


class RoundGoalStanding(pydantic.BaseModel):
    """A player's live standing on the current round goal: their category
    ``count``, the ``place`` that count earns versus the opponent (ties share
    1st), and the ``vp`` that place would award if the round ended now."""

    count: int
    place: int
    vp: int


def score_round_goal(engine: "core.Engine", round_idx: int) -> None:
    """Award round-goal VP based on each player's category count. The payout
    scales by round (``state.ROUND_GOAL_PAYOUTS_2P``): in a 2P game the higher
    count takes 1st-place VP and the lower takes 2nd. Tied players split 1st and
    2nd, each taking the floor of the combined payout (the official tie rule).
    A player whose count is 0 does not place and scores nothing."""
    goal = engine.state.round_goals[round_idx]
    counts = [eval_goal(player, goal) for player in engine.state.players]
    engine.log(f"Round {round_idx + 1} goal '{goal.category}' counts: {counts}")
    first, second = state.ROUND_GOAL_PAYOUTS_2P[round_idx]
    count_0, count_1 = counts
    engine.state.players[0].round_goal_points += _placement_vp(
        count_0, count_1, first, second
    )
    engine.state.players[1].round_goal_points += _placement_vp(
        count_1, count_0, first, second
    )
    engine.instrumentation.round_goal_scored(
        engine=engine, round_num=round_idx, goal=goal, counts=counts
    )


def round_goal_standing(
    game_state: state.GameState, player: state.Player
) -> RoundGoalStanding:
    """``player``'s live standing on the *current* round goal. Thin wrapper over
    :func:`round_goal_standing_for_round` at ``game_state.round_idx``."""
    return round_goal_standing_for_round(game_state, player, game_state.round_idx)


def round_goal_standing_for_round(
    game_state: state.GameState, player: state.Player, round_idx: int
) -> RoundGoalStanding:
    """``player``'s live standing on the round ``round_idx`` goal, mirroring the
    2-player payout rule in :func:`score_round_goal` without mutating any state:
    the higher category count takes 1st, the lower takes 2nd, a tie splits the
    two places (floored), and a count of 0 does not place (0 VP). Assumes
    ``round_idx`` indexes an in-play goal (``0 <= round_idx <
    len(game_state.round_goals)``)."""
    goal = game_state.round_goals[round_idx]
    my_count = eval_goal(player, goal)
    opp_count = max(
        (eval_goal(other, goal) for other in game_state.players if other is not player),
        default=0,
    )
    first, second = state.ROUND_GOAL_PAYOUTS_2P[round_idx]
    place = 1 if my_count >= opp_count else 2
    return RoundGoalStanding(
        count=my_count,
        place=place,
        vp=_placement_vp(my_count, opp_count, first, second),
    )


def eval_goal(player: state.Player, goal: cards.EndRoundGoal) -> int:
    """Score ``player`` against ``goal.category``. Returns 0 for unknown
    categories so unsupported goals don't blow up the round."""
    counter = _CATEGORY_COUNTERS.get(goal.category)
    if counter is None:
        return 0
    return counter(player)


def goal_count_delta_for_bird(bird: cards.Bird, category: str) -> int:
    """Marginal change in category count from playing ``bird``.

    Returns 0 for egg / tuck categories (a freshly played bird starts with no
    eggs or tucked cards, so those counts are unaffected at play time)."""
    match category:
        case "birds_forest":
            return 1 if cards.Habitat.FOREST in bird.habitats else 0
        case "birds_grassland":
            return 1 if cards.Habitat.GRASSLAND in bird.habitats else 0
        case "birds_wetland":
            return 1 if cards.Habitat.WETLAND in bird.habitats else 0
        case "total_birds":
            return 1
        case "wingspan_under_30":
            # wingspan_cm == 0 means "no data" (mirrors _count_wingspan_under_30)
            return 1 if bird.wingspan_cm and bird.wingspan_cm < 30 else 0
        case "wingspan_over_65":
            return 1 if bird.wingspan_cm > 65 else 0
        case _:
            return 0


def goal_vp_delta_for_bird(
    player: state.Player,
    opp: state.Player,
    goal: cards.EndRoundGoal,
    bird: cards.Bird,
    payout: tuple[int, int],
) -> tuple[int, int]:
    """Return ``(count_delta, vp_delta)`` for playing ``bird`` against ``goal``.

    ``vp_delta`` is the change in 2P placement VP at current standings.
    Both values are 0 when the bird cannot affect this goal category."""
    count_delta = goal_count_delta_for_bird(bird, goal.category)
    before_count = eval_goal(player, goal)
    opp_count = eval_goal(opp, goal)
    old_vp = _placement_vp(before_count, opp_count, payout[0], payout[1])
    new_vp = _placement_vp(before_count + count_delta, opp_count, payout[0], payout[1])
    return count_delta, new_vp - old_vp


def final_scoring(engine: "core.Engine") -> None:
    """Compute each player's final score = birds + bonus + eggs + tucked
    + cached + round-goal points. Result is written to ``Player.final_score``
    and to the game log."""
    for player in engine.state.players:
        bird_pts = sum(pb.bird.points for row in player.board.values() for pb in row)
        bonus_pts = sum(bonus_score(player, bc) for bc in player.bonus_cards)
        eggs = player.total_eggs
        tucked = player.total_tucked
        cached = player.total_cached
        food_left = player.total_food()
        round_goal = player.round_goal_points
        total = bird_pts + bonus_pts + eggs + tucked + cached + round_goal
        engine.log(
            f"[{player.name}] FINAL: birds={bird_pts} bonus={bonus_pts} eggs={eggs}"
            f" tucked={tucked} cached={cached} round_goal={round_goal}"
            f" foodleft={food_left} -> {total}"
        )
        player.final_score = total
        engine.instrumentation.player_final_scored(
            engine=engine,
            player=player,
            total=total,
            bird_pts=bird_pts,
            bonus_pts=bonus_pts,
            eggs=eggs,
            tucked=tucked,
            cached=cached,
            round_goal=round_goal,
        )


def running_score(player: state.Player) -> int:
    """Victory points ``player`` would score if the game ended right now.

    Same formula as :func:`final_scoring` — birds + bonus + eggs + tucked
    + cached + round-goal points — but a pure read with no logging or state
    mutation, so the interactive display can show a live total each turn.
    Food still in the supply does not score and is excluded."""
    bird_pts = sum(pb.bird.points for row in player.board.values() for pb in row)
    bonus_pts = sum(bonus_score(player, bc) for bc in player.bonus_cards)
    return (
        bird_pts
        + bonus_pts
        + player.total_eggs
        + player.total_tucked
        + player.total_cached
        + player.round_goal_points
    )


def bonus_qualifying_count(player: state.Player, bc: cards.BonusCard) -> int:
    """Number of ``player``'s in-play birds that qualify for bonus card ``bc``
    (those whose ``bonus_categories`` include ``bc.name``)."""
    return sum(
        1
        for row in player.board.values()
        for pb in row
        if bc.name in pb.bird.bonus_categories
    )


def bonus_score_for_count(bc: cards.BonusCard, count: int) -> int:
    """Stepped VP bonus card ``bc`` pays for exactly ``count`` qualifying birds:
    a per-bird card pays ``per_bird_vp`` for each, a tiered card pays the
    highest threshold met. Pure in ``count`` so callers can price a
    hypothetical board (e.g. ``count + 1`` for a play candidate) without a
    :class:`state.Player`."""
    if bc.per_bird_vp is not None:
        return bc.per_bird_vp * count
    best = 0
    for thr, vp in bc.thresholds:
        if count >= thr and vp > best:
            best = vp
    return best


def bonus_linear_value_for_count(bc: cards.BonusCard, count: int) -> float:
    """Dense piecewise-linear payoff of bonus card ``bc`` at ``count``
    qualifying birds — the gradient-friendly form of
    :func:`bonus_score_for_count`. Interpolates linearly between ``(0, 0)`` and
    each ``(count, vp)`` threshold (ascending), holding flat at the final VP
    past the last threshold; per-bird cards are already linear and return
    ``per_bird_vp * count``. Pure in ``count``."""
    if bc.per_bird_vp is not None:
        return float(bc.per_bird_vp * count)
    if not bc.thresholds:
        return 0.0
    anchors: tuple[tuple[int, int], ...] = ((0, 0), *bc.thresholds)
    last_count, last_vp = anchors[-1]
    if count >= last_count:
        return float(last_vp)
    for i in range(1, len(anchors)):
        lo_count, lo_vp = anchors[i - 1]
        hi_count, hi_vp = anchors[i]
        if count < hi_count:
            span = hi_count - lo_count  # >0: anchors strictly ascending
            frac = (count - lo_count) / span
            return lo_vp + frac * (hi_vp - lo_vp)
    return float(last_vp)  # unreachable; satisfies strict pyright


def bonus_score(player: state.Player, bc: cards.BonusCard) -> int:
    """VP ``player`` scores from bonus card ``bc`` (stepped payout).

    Counts the qualifying birds in play, then applies ``bc``'s payout via
    :func:`bonus_score_for_count`."""
    return bonus_score_for_count(bc, bonus_qualifying_count(player, bc))


def bonus_linear_value(player: state.Player, bc: cards.BonusCard) -> float:
    """Dense piecewise-linear payoff estimate for bonus card ``bc``.

    Where :func:`bonus_score` jumps in steps at each threshold, this rewards
    incremental progress toward the next plateau — the qualifying count in play
    priced via :func:`bonus_linear_value_for_count`."""
    return bonus_linear_value_for_count(bc, bonus_qualifying_count(player, bc))


###### PRIVATE #######


def _placement_vp(my_count: int, opp_count: int, first: int, second: int) -> int:
    """VP one player earns on a 2P round goal given both category counts. A
    count of 0 never places (scores 0). Otherwise the higher count earns
    ``first`` and the lower earns ``second``; on a tie the two players occupy
    1st and 2nd place together, so each takes the floor of the combined payout
    (the official rulebook tie rule) — e.g. round 1 pays ``(4 + 1) // 2 == 2``
    to each tied player."""
    if my_count == 0:
        return 0
    if my_count > opp_count:
        return first
    if my_count < opp_count:
        return second
    return (first + second) // 2


def _count_birds_in(habitat: cards.Habitat) -> typing.Callable[[state.Player], int]:
    return lambda player: len(player.board[habitat])


def _count_eggs_in_habitat(
    habitat: cards.Habitat,
) -> typing.Callable[[state.Player], int]:
    return lambda player: sum(pb.eggs for pb in player.board[habitat])


def _count_eggs_on_nest(nest: cards.NestType) -> typing.Callable[[state.Player], int]:
    return lambda player: sum(
        pb.eggs for row in player.board.values() for pb in row if pb.bird.nest == nest
    )


def _count_birds_with_eggs_on_nest(
    nest: cards.NestType,
) -> typing.Callable[[state.Player], int]:
    return lambda player: sum(
        1
        for row in player.board.values()
        for pb in row
        if pb.bird.nest == nest and pb.eggs > 0
    )


def _count_wingspan_under_30(player: state.Player) -> int:
    return sum(
        1
        for row in player.board.values()
        for pb in row
        if pb.bird.wingspan_cm and pb.bird.wingspan_cm < 30
    )


def _count_wingspan_over_65(player: state.Player) -> int:
    return sum(
        1
        for row in player.board.values()
        for pb in row
        if pb.bird.wingspan_cm and pb.bird.wingspan_cm > 65
    )


def _count_tucked_birds(player: state.Player) -> int:
    return player.total_tucked


def _count_total_birds(player: state.Player) -> int:
    return sum(len(row) for row in player.board.values())


def _count_egg_sets_three_habitats(player: state.Player) -> int:
    """One 'set' is 1 egg in each of forest, grassland, and wetland, so the
    number of complete sets is the smallest per-habitat egg count."""
    return min(
        sum(pb.eggs for pb in player.board[habitat]) for habitat in cards.ALL_HABITATS
    )


_CATEGORY_COUNTERS: dict[str, typing.Callable[[state.Player], int]] = {
    "birds_forest": _count_birds_in(cards.Habitat.FOREST),
    "birds_grassland": _count_birds_in(cards.Habitat.GRASSLAND),
    "birds_wetland": _count_birds_in(cards.Habitat.WETLAND),
    "eggs_forest": _count_eggs_in_habitat(cards.Habitat.FOREST),
    "eggs_grassland": _count_eggs_in_habitat(cards.Habitat.GRASSLAND),
    "eggs_wetland": _count_eggs_in_habitat(cards.Habitat.WETLAND),
    "eggs_bowl": _count_eggs_on_nest(cards.NestType.BOWL),
    "eggs_cavity": _count_eggs_on_nest(cards.NestType.CAVITY),
    "eggs_ground": _count_eggs_on_nest(cards.NestType.GROUND),
    "eggs_platform": _count_eggs_on_nest(cards.NestType.PLATFORM),
    "bowl_birds_with_eggs": _count_birds_with_eggs_on_nest(cards.NestType.BOWL),
    "cavity_birds_with_eggs": _count_birds_with_eggs_on_nest(cards.NestType.CAVITY),
    "ground_birds_with_eggs": _count_birds_with_eggs_on_nest(cards.NestType.GROUND),
    "platform_birds_with_eggs": _count_birds_with_eggs_on_nest(cards.NestType.PLATFORM),
    "tucked_cards": _count_tucked_birds,
    "wingspan_under_30": _count_wingspan_under_30,
    "wingspan_over_65": _count_wingspan_over_65,
    "total_birds": _count_total_birds,
    "egg_sets_3habitats": _count_egg_sets_three_habitats,
}
