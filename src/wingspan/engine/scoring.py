"""End-of-round goal scoring and end-of-game final scoring."""

from __future__ import annotations

import typing

from wingspan import cards, state

if typing.TYPE_CHECKING:
    from wingspan.engine import core


def score_round_goal(engine: "core.Engine", round_idx: int) -> None:
    """Award round-goal VP based on each player's category score. In a 2P
    game, the winner takes 1st-place VP, the loser takes 2nd; ties split
    by both players taking 1st."""
    goal = engine.state.round_goals[round_idx]
    scores = [eval_goal(p, goal) for p in engine.state.players]
    engine.log(f"Round {round_idx + 1} goal '{goal.category}' scores: {scores}")
    first, second = goal.payouts_2p
    a, b = scores
    if a > b:
        engine.state.players[0].round_goal_points += first
        engine.state.players[1].round_goal_points += second
    elif b > a:
        engine.state.players[1].round_goal_points += first
        engine.state.players[0].round_goal_points += second
    else:
        engine.state.players[0].round_goal_points += first
        engine.state.players[1].round_goal_points += first


def eval_goal(p: state.Player, goal: cards.EndRoundGoal) -> int:
    """Score ``p`` against ``goal.category``. Returns 0 for unknown
    categories so unsupported goals don't blow up the round."""
    counter = _CATEGORY_COUNTERS.get(goal.category)
    if counter is None:
        return 0
    return counter(p)


def final_scoring(engine: "core.Engine") -> None:
    """Compute each player's final score = birds + bonus + eggs + tucked
    + cached + round-goal points. Result is written to ``Player.final_score``
    and to the game log."""
    for p in engine.state.players:
        bird_pts = sum(pb.bird.points for r in p.board.values() for pb in r)
        bonus_pts = sum(bonus_score(p, bc) for bc in p.bonus_cards)
        eggs = p.total_eggs
        tucked = p.total_tucked
        cached = p.total_cached
        food_left = p.total_food()
        round_goal = p.round_goal_points
        total = bird_pts + bonus_pts + eggs + tucked + cached + round_goal
        engine.log(
            f"[{p.name}] FINAL: birds={bird_pts} bonus={bonus_pts} eggs={eggs}"
            f" tucked={tucked} cached={cached} round_goal={round_goal}"
            f" foodleft={food_left} -> {total}"
        )
        p.final_score = total


def bonus_score(p: state.Player, bc: cards.BonusCard) -> int:
    """Highest VP threshold met by the count of qualifying birds in play
    that belong to ``bc``'s category."""
    count = sum(
        1 for r in p.board.values() for pb in r if bc.name in pb.bird.bonus_categories
    )
    best = 0
    for thr, vp in bc.thresholds:
        if count >= thr and vp > best:
            best = vp
    return best


###### PRIVATE #######


def _count_birds_in(habitat: cards.Habitat) -> typing.Callable[[state.Player], int]:
    return lambda p: len(p.board[habitat])


def _count_eggs_in_habitat(
    habitat: cards.Habitat,
) -> typing.Callable[[state.Player], int]:
    return lambda p: sum(pb.eggs for pb in p.board[habitat])


def _count_eggs_on_nest(nest: cards.NestType) -> typing.Callable[[state.Player], int]:
    return lambda p: sum(
        pb.eggs for r in p.board.values() for pb in r if pb.bird.nest == nest
    )


def _count_birds_with_eggs_on_nest(
    nest: cards.NestType,
) -> typing.Callable[[state.Player], int]:
    return lambda p: sum(
        1 for r in p.board.values() for pb in r if pb.bird.nest == nest and pb.eggs > 0
    )


def _count_wingspan_under_30(p: state.Player) -> int:
    return sum(
        1
        for r in p.board.values()
        for pb in r
        if pb.bird.wingspan_cm and pb.bird.wingspan_cm < 30
    )


def _count_wingspan_over_65(p: state.Player) -> int:
    return sum(
        1
        for r in p.board.values()
        for pb in r
        if pb.bird.wingspan_cm and pb.bird.wingspan_cm > 65
    )


def _count_tucked_birds(p: state.Player) -> int:
    return p.total_tucked


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
}
