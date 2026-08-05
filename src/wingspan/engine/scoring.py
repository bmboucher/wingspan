"""End-of-round goal scoring and end-of-game final scoring."""

from __future__ import annotations

import itertools
import typing

import pydantic

from wingspan import cards, state

if typing.TYPE_CHECKING:
    from wingspan.engine import core


def placement_payouts(
    counts: typing.Sequence[int], payouts: typing.Sequence[int]
) -> list[int]:
    """Award each seat's placement VP from its category ``counts`` against a
    round's payout ladder (``payouts[0]`` = 1st place, ``payouts[1]`` = 2nd,
    ...). The shared kernel behind round-goal scoring at any player count.

    Ranks ``counts`` descending; a count of 0 never places (scores 0 and
    occupies no place — the seats behind it move up to fill the gap). A
    k-way tie of equal, nonzero counts occupies places ``p..p+k-1`` together
    and each tied seat takes the floor of those places' combined payout,
    ``sum(effective_payouts[p:p+k]) // k`` — ``effective_payouts[i]`` is
    ``payouts[i]`` when in range, else 0 (a place past the end of the payout
    ladder scores nothing; only the top 3 places pay on the printed round-goal
    board). Reduces exactly to the legacy 2-player rule: the higher count
    takes 1st, the lower takes 2nd (or 0 if it is 0), and a tie splits the
    floor of 1st + 2nd."""
    vps = [0] * len(counts)
    ranked_seats = sorted(
        (seat for seat, count in enumerate(counts) if count > 0),
        key=lambda seat: counts[seat],
        reverse=True,
    )
    place = 0
    idx = 0
    while idx < len(ranked_seats):
        tied_seats = [ranked_seats[idx]]
        tied_count = counts[ranked_seats[idx]]
        idx += 1
        while idx < len(ranked_seats) and counts[ranked_seats[idx]] == tied_count:
            tied_seats.append(ranked_seats[idx])
            idx += 1
        n_tied = len(tied_seats)
        pool = sum(
            payouts[place_idx] if place_idx < len(payouts) else 0
            for place_idx in range(place, place + n_tied)
        )
        share = pool // n_tied
        for seat in tied_seats:
            vps[seat] = share
        place += n_tied
    return vps


def winners(players: typing.Sequence[state.Player]) -> list[int]:
    """The id(s) of the game's winner(s), per the official Wingspan tiebreak:
    the highest ``final_score``, and among those tied, the most unused food
    remaining in personal supply (``Player.total_food()`` — cached food sitting
    on birds, ``PlayedBird.cached_food``, does not count). Returns every id
    still tied after both tiebreaks — more than one entry is a genuine shared
    victory (see :func:`determine_winner`)."""
    top_score = max(player.final_score or 0 for player in players)
    score_leaders = [
        player for player in players if (player.final_score or 0) == top_score
    ]
    top_food = max(player.total_food() for player in score_leaders)
    return [player.id for player in score_leaders if player.total_food() == top_food]


def determine_winner(players: typing.Sequence[state.Player]) -> int:
    """The single winner's id, or ``-1`` when the game ends in a genuine
    shared victory (:func:`winners` still returns more than one id after the
    score and food tiebreaks — the official rulebook's final word: a tie
    truly shared)."""
    winner_ids = winners(players)
    return winner_ids[0] if len(winner_ids) == 1 else -1


class RoundGoalStanding(pydantic.BaseModel):
    """A player's standing on one round goal: their category ``count``, the
    ``other_counts`` of every other seat (clockwise from this player, see
    ``GameState.opponents_clockwise``), the ``place`` that count earns (1 +
    the number of other seats with a strictly greater count — ties share a
    place), and the ``vp`` that place awards. Live values for an unscored
    round; frozen at-scoring values for a scored one (see
    :func:`round_goal_standing_for_round`)."""

    count: int
    other_counts: list[int]
    place: int
    vp: int

    @property
    def opp_count(self) -> int:
        """The best (highest) other seat's count.

        A derived read-only view for the pre-N-player callers that only ever
        cared about a single opponent (``encode.state_encode``,
        ``agents.display``) — kept so they read correctly at any player
        count without themselves needing to know about ``other_counts``."""
        return max(self.other_counts, default=0)


def score_round_goal(engine: "core.Engine", round_idx: int) -> None:
    """Award round-goal VP based on each player's category count, via
    :func:`placement_payouts` against the round's payout ladder
    (``state.ROUND_GOAL_PAYOUTS[round_idx]``): the highest count takes 1st
    place VP, ties share the floor of their combined places' payouts, and a
    count of 0 never places (scores nothing). The outcome is frozen onto
    ``GameState.scored_goals`` — a scored round's standings never change
    again, however the boards evolve."""
    goal = engine.state.round_goals[round_idx]
    players = engine.state.players
    counts = [eval_goal(player, goal) for player in players]
    vps = placement_payouts(counts, state.ROUND_GOAL_PAYOUTS[round_idx])
    engine.log_global(
        f"Round {round_idx + 1} goal '{goal.category}' counts: {counts}" f" → VP: {vps}"
    )
    for player, vp in zip(players, vps):
        player.round_goal_points += vp
    engine.state.scored_goals.append(
        state.RoundGoalResult(counts=list(counts), vp_awarded=list(vps))
    )
    engine.events.record_round_goal(engine, round_idx, goal.category, counts, vps)
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
    """``player``'s standing on the round ``round_idx`` goal.

    For a round that has already been scored, the standing is read verbatim
    from the frozen ``GameState.scored_goals`` record — a scored goal's counts
    and VP never change again. For an unscored round it is the live standing,
    mirroring :func:`score_round_goal`'s placement rule without mutating any
    state. ``other_counts`` lists every other seat's count, clockwise from
    ``player`` (``GameState.opponents_clockwise``). Assumes ``round_idx``
    indexes an in-play goal (``0 <= round_idx < len(game_state.round_goals)``)."""
    others = game_state.opponents_clockwise(player.id)
    if round_idx < len(game_state.scored_goals):
        result = game_state.scored_goals[round_idx]
        my_count = result.counts[player.id]
        other_counts = [result.counts[other.id] for other in others]
        return RoundGoalStanding(
            count=my_count,
            other_counts=other_counts,
            place=1 + sum(1 for count in other_counts if count > my_count),
            vp=result.vp_awarded[player.id],
        )
    goal = game_state.round_goals[round_idx]
    my_count = eval_goal(player, goal)
    other_counts = [eval_goal(other, goal) for other in others]
    place = 1 + sum(1 for count in other_counts if count > my_count)
    vp = _my_placement_vp(my_count, other_counts, state.ROUND_GOAL_PAYOUTS[round_idx])
    return RoundGoalStanding(
        count=my_count,
        other_counts=other_counts,
        place=place,
        vp=vp,
    )


#### Round-goal evaluation and count deltas ####


def eval_goal(player: state.Player, goal: cards.EndRoundGoal) -> int:
    """Score ``player`` against ``goal.category``. Returns 0 for unknown
    categories so unsupported goals don't blow up the round."""
    counter = _CATEGORY_COUNTERS.get(goal.category)
    if counter is None:
        return 0
    return counter(player)


def goal_count_delta_for_bird(
    bird: cards.Bird,
    category: str,
    *,
    play_habitat: cards.Habitat | None = None,
) -> int:
    """Marginal change in category count from playing ``bird``.

    ``play_habitat`` is the row the play commits the bird to, when known: the
    per-habitat bird counts then move only if the landing row matches. When
    ``None`` (a candidate not yet committed to a row — hand / tray / setup
    keeps), a per-habitat count moves if *any* of the bird's card habitats
    matches: the optimistic could-count bound.

    Returns 0 for egg-total / tuck categories (a freshly played bird starts
    with no eggs or tucked cards, so those counts are unaffected at play
    time) — except ``birds_no_eggs``, which the eggless newcomer advances."""
    match category:
        case "birds_forest":
            return _habitat_goal_count_delta(bird, cards.Habitat.FOREST, play_habitat)
        case "birds_grassland":
            return _habitat_goal_count_delta(
                bird, cards.Habitat.GRASSLAND, play_habitat
            )
        case "birds_wetland":
            return _habitat_goal_count_delta(bird, cards.Habitat.WETLAND, play_habitat)
        case "total_birds":
            return 1
        case "birds_no_eggs":
            return 1  # a just-played bird has no eggs yet
        case "wingspan_under_30":
            # wingspan_cm == 0 means "no data" (mirrors _count_wingspan_under_30)
            return 1 if bird.wingspan_cm and bird.wingspan_cm < 30 else 0
        case "wingspan_over_65":
            return 1 if bird.wingspan_cm > 65 else 0
        case _:
            return 0


def goal_vp_delta_for_bird(
    player: state.Player,
    others: typing.Sequence[state.Player],
    goal: cards.EndRoundGoal,
    bird: cards.Bird,
    payouts: typing.Sequence[int],
    *,
    play_habitat: cards.Habitat | None = None,
) -> tuple[int, int]:
    """Return ``(count_delta, vp_delta)`` for playing ``bird`` against ``goal``.

    ``vp_delta`` is the change in placement VP versus every seat in
    ``others`` at current standings. Both values are 0 when the bird cannot
    affect this goal category. ``play_habitat`` is the committed landing row
    when known (see :func:`goal_count_delta_for_bird`)."""
    count_delta = goal_count_delta_for_bird(
        bird, goal.category, play_habitat=play_habitat
    )
    before_count = eval_goal(player, goal)
    other_counts = [eval_goal(other, goal) for other in others]
    old_vp = _my_placement_vp(before_count, other_counts, payouts)
    new_vp = _my_placement_vp(before_count + count_delta, other_counts, payouts)
    return count_delta, new_vp - old_vp


def goal_count_delta_for_bird_with_eggs(
    bird: cards.Bird,
    category: str,
    *,
    play_habitat: cards.Habitat | None = None,
    player: state.Player | None = None,
) -> int:
    """Per-goal-optimistic marginal count from playing ``bird`` and then
    egg-populating it to whatever level best advances ``category`` — the
    "played and egg-populated" counterpart to :func:`goal_count_delta_for_bird`,
    which only prices the play-instant delta (0 for every egg-driven
    category, since a freshly played bird starts with no eggs).

    The bird is assumed played: when ``player`` is known, at least one of its
    card habitats must have an open slot, or the play-hypothesis is
    impossible and every category returns 0 (the guard is skipped for a
    committed ``play_habitat`` — the offered row already exists). Its eggs
    are then assumed set to whatever level best advances ``category`` —
    typically ``bird.egg_limit`` on a reachable egg-total goal, wild star
    nests included (:func:`cards.nest_matches`). ``play_habitat`` is the
    committed landing row (play-bird rows); ``None`` is an uncommitted
    candidate (hand / tray / setup keeps), which uses the optimistic
    any-reachable-habitat bound. ``player=None`` means no board context at
    all (setup pricing over an empty board) — the playability guard is
    skipped and habitat reachability falls back to every card habitat."""
    if (
        player is not None
        and play_habitat is None
        and not any(player.can_play_in(habitat) for habitat in bird.habitats)
    ):
        return 0

    goal_nest = _EGGS_ON_NEST_CATEGORIES.get(category)
    if goal_nest is not None:
        return bird.egg_limit if cards.nest_matches(bird.nest, goal_nest) else 0

    goal_nest = _BIRDS_WITH_EGGS_CATEGORIES.get(category)
    if goal_nest is not None:
        matches = cards.nest_matches(bird.nest, goal_nest)
        return 1 if matches and bird.egg_limit > 0 else 0

    goal_habitat = _EGGS_IN_HABITAT_CATEGORIES.get(category)
    if goal_habitat is not None:
        if play_habitat is not None:
            reachable = play_habitat == goal_habitat
        else:
            reachable = goal_habitat in _candidate_habitats(bird, player)
        return bird.egg_limit if reachable else 0

    if category == "egg_sets_3habitats":
        return _egg_sets_delta_with_bird(bird, play_habitat, player)

    return goal_count_delta_for_bird(bird, category, play_habitat=play_habitat)


def goal_vp_delta_for_bird_with_eggs(
    player: state.Player,
    others: typing.Sequence[state.Player],
    goal: cards.EndRoundGoal,
    bird: cards.Bird,
    payouts: typing.Sequence[int],
    *,
    play_habitat: cards.Habitat | None = None,
) -> tuple[int, int]:
    """Return ``(count_delta, vp_delta)`` for playing ``bird`` and
    egg-populating it optimally against ``goal`` — the played-and-egg-populated
    counterpart to :func:`goal_vp_delta_for_bird`, mirrored structurally.
    ``vp_delta`` is the change in placement VP versus every seat in ``others``
    at current standings. ``count_delta`` comes from
    :func:`goal_count_delta_for_bird_with_eggs` (``player`` is always
    supplied, so its playability guard applies); ``play_habitat`` is the
    committed landing row when known (see
    :func:`goal_count_delta_for_bird_with_eggs`)."""
    count_delta = goal_count_delta_for_bird_with_eggs(
        bird, goal.category, play_habitat=play_habitat, player=player
    )
    before_count = eval_goal(player, goal)
    other_counts = [eval_goal(other, goal) for other in others]
    old_vp = _my_placement_vp(before_count, other_counts, payouts)
    new_vp = _my_placement_vp(before_count + count_delta, other_counts, payouts)
    return count_delta, new_vp - old_vp


def goal_affinity_for_kept(kept: typing.Sequence[cards.Bird], category: str) -> int:
    """Setup-time hand-level affinity: the optimistic per-goal bound for a
    setup keep, assuming every kept bird is eventually played and
    egg-populated optimally for ``category``.

    ``egg_sets_3habitats`` is not simply additive across birds — each bird
    can only land in one habitat, so :func:`_best_kept_egg_sets` searches the
    best habitat assignment across the whole keep. Every other category sums
    :func:`goal_count_delta_for_bird_with_eggs` (no committed habitat, no
    board context) over ``kept``."""
    if category == "egg_sets_3habitats":
        return _best_kept_egg_sets(kept)
    return sum(goal_count_delta_for_bird_with_eggs(bird, category) for bird in kept)


def goal_count_delta_for_egg(
    player: state.Player,
    habitat: cards.Habitat,
    played_bird: state.PlayedBird,
    category: str,
    delta_eggs: int,
) -> int:
    """Marginal change in ``category``'s count from adding (``delta_eggs > 0``)
    or removing (``< 0``) that many eggs on ``played_bird``, which sits in
    ``habitat`` on ``player``'s board.

    Covers the egg-driven categories: per-habitat and per-nest egg totals move
    by ``delta_eggs`` when the slot matches (star nests are wild —
    ``cards.nest_matches``); the ``*_birds_with_eggs`` counts move only on a
    has-eggs threshold crossing (``birds_no_eggs`` on the same crossing,
    inverted); ``egg_sets_3habitats`` recomputes the min-across-habitats set
    count before and after. 0 for every non-egg category (playing-a-bird
    deltas live in :func:`goal_count_delta_for_bird`)."""
    goal_habitat = _EGGS_IN_HABITAT_CATEGORIES.get(category)
    if goal_habitat is not None:
        return delta_eggs if habitat == goal_habitat else 0

    goal_nest = _EGGS_ON_NEST_CATEGORIES.get(category)
    if goal_nest is not None:
        if not cards.nest_matches(played_bird.bird.nest, goal_nest):
            return 0
        return delta_eggs

    goal_nest = _BIRDS_WITH_EGGS_CATEGORIES.get(category)
    if goal_nest is not None:
        if not cards.nest_matches(played_bird.bird.nest, goal_nest):
            return 0
        had_eggs = played_bird.eggs > 0
        has_eggs_after = played_bird.eggs + delta_eggs > 0
        return int(has_eggs_after) - int(had_eggs)

    if category == "egg_sets_3habitats":
        egg_sums = _per_habitat_egg_sums(player)
        before_sets = min(egg_sums.values())
        egg_sums[habitat] += delta_eggs
        return min(egg_sums.values()) - before_sets

    if category == "birds_no_eggs":
        had_eggs = played_bird.eggs > 0
        has_eggs_after = played_bird.eggs + delta_eggs > 0
        return int(had_eggs) - int(has_eggs_after)

    return 0


def goal_vp_delta_for_egg(
    player: state.Player,
    others: typing.Sequence[state.Player],
    goal: cards.EndRoundGoal,
    habitat: cards.Habitat,
    played_bird: state.PlayedBird,
    payouts: typing.Sequence[int],
    delta_eggs: int,
) -> tuple[int, int]:
    """Return ``(count_delta, vp_delta)`` for an egg event against ``goal`` —
    the egg analogue of :func:`goal_vp_delta_for_bird`. ``vp_delta`` is the
    change in placement VP versus every seat in ``others`` at current
    standings."""
    count_delta = goal_count_delta_for_egg(
        player, habitat, played_bird, goal.category, delta_eggs
    )
    before_count = eval_goal(player, goal)
    other_counts = [eval_goal(other, goal) for other in others]
    old_vp = _my_placement_vp(before_count, other_counts, payouts)
    new_vp = _my_placement_vp(before_count + count_delta, other_counts, payouts)
    return count_delta, new_vp - old_vp


def goal_count_delta_for_move(
    player: state.Player,
    from_habitat: cards.Habitat,
    to_habitat: cards.Habitat,
    played_bird: state.PlayedBird,
    category: str,
) -> int:
    """Marginal change in ``category``'s count from moving ``played_bird``
    (with the eggs sitting on it) from ``from_habitat`` to ``to_habitat`` on
    ``player``'s board.

    Per-habitat bird counts move by ±1, per-habitat egg totals by
    ``played_bird.eggs``, and ``egg_sets_3habitats`` recomputes the min with
    the egg block relocated. Nest-keyed, wingspan, total-bird, tucked, and
    no-egg categories are unaffected (the bird and its eggs are unchanged —
    only the habitat differs). 0 when the move is a stay."""
    if from_habitat == to_habitat:
        return 0

    goal_habitat = _BIRDS_IN_HABITAT_CATEGORIES.get(category)
    if goal_habitat is not None:
        return int(to_habitat == goal_habitat) - int(from_habitat == goal_habitat)

    goal_habitat = _EGGS_IN_HABITAT_CATEGORIES.get(category)
    if goal_habitat is not None:
        sign = int(to_habitat == goal_habitat) - int(from_habitat == goal_habitat)
        return sign * played_bird.eggs

    if category == "egg_sets_3habitats":
        egg_sums = _per_habitat_egg_sums(player)
        before_sets = min(egg_sums.values())
        egg_sums[from_habitat] -= played_bird.eggs
        egg_sums[to_habitat] += played_bird.eggs
        return min(egg_sums.values()) - before_sets

    return 0


def goal_vp_delta_for_move(
    player: state.Player,
    others: typing.Sequence[state.Player],
    goal: cards.EndRoundGoal,
    from_habitat: cards.Habitat,
    to_habitat: cards.Habitat,
    played_bird: state.PlayedBird,
    payouts: typing.Sequence[int],
) -> tuple[int, int]:
    """Return ``(count_delta, vp_delta)`` for a bird move against ``goal`` —
    the move analogue of :func:`goal_vp_delta_for_bird`."""
    count_delta = goal_count_delta_for_move(
        player, from_habitat, to_habitat, played_bird, goal.category
    )
    before_count = eval_goal(player, goal)
    other_counts = [eval_goal(other, goal) for other in others]
    old_vp = _my_placement_vp(before_count, other_counts, payouts)
    new_vp = _my_placement_vp(before_count + count_delta, other_counts, payouts)
    return count_delta, new_vp - old_vp


def goal_best_case_for_eggs(
    player: state.Player,
    others: typing.Sequence[state.Player],
    goal: cards.EndRoundGoal,
    payouts: typing.Sequence[int],
    n_eggs: int,
) -> tuple[int, int]:
    """Optimistic ``(count_delta, vp_delta)`` bound for a *commitment* to lay
    (``n_eggs > 0``) or remove (``n_eggs < 0``) eggs whose targets are picked
    in a follow-up decision.

    Laying is bounded by real capacity in qualifying slots (egg room in the
    goal's habitat / on matching-nest birds, empty matching birds for the
    ``*_birds_with_eggs`` counts, a greedy raise-the-min fill for the egg-set
    goal), assuming the player directs every egg toward this goal — or, for
    the anti-goal ``birds_no_eggs``, away from it (only forced overflow onto
    eggless birds counts against it). Removal is
    the exact least-damage case over the single eggs currently removable (the
    only removal commitments offered take one egg). The bound is what the
    commitment row can honestly advertise before the target picks resolve;
    the per-target rows price the realized delta exactly."""
    count_delta = _best_case_egg_count_delta(player, goal.category, n_eggs)
    before_count = eval_goal(player, goal)
    other_counts = [eval_goal(other, goal) for other in others]
    old_vp = _my_placement_vp(before_count, other_counts, payouts)
    new_vp = _my_placement_vp(before_count + count_delta, other_counts, payouts)
    return count_delta, new_vp - old_vp


#### Bonus card scoring ####


def bonus_vp_deltas_for_count_change(
    bc: cards.BonusCard, before: int, after: int
) -> tuple[float, float]:
    """``(stepped_delta, linear_delta)`` VP change of bonus card ``bc`` when
    its qualifying count moves from ``before`` to ``after`` — the uniform
    pricer behind every bonus-delta stripe fill."""
    stepped = float(
        bonus_score_for_count(bc, after) - bonus_score_for_count(bc, before)
    )
    linear = bonus_linear_value_for_count(bc, after) - bonus_linear_value_for_count(
        bc, before
    )
    return stepped, linear


def bonus_count_delta_for_egg(
    bc: cards.BonusCard, played_bird: state.PlayedBird, delta_eggs: int
) -> int:
    """Change in ``bc``'s qualifying count from adding/removing ``delta_eggs``
    eggs on ``played_bird``: ±1 when an egg-counting dynamic card's threshold
    is crossed (Breeding Manager at 4, Oologist at 1), else 0."""
    min_eggs = _EGG_COUNT_BONUS_MIN_EGGS.get(bc.name)
    if min_eggs is None:
        return 0
    qualified_before = played_bird.eggs >= min_eggs
    qualified_after = played_bird.eggs + delta_eggs >= min_eggs
    return int(qualified_after) - int(qualified_before)


def bonus_count_delta_for_hand(bc: cards.BonusCard, delta_cards: int) -> int:
    """Change in ``bc``'s qualifying count from the hand growing/shrinking by
    ``delta_cards`` — nonzero only for the hand-counting dynamic card
    (Visionary Leader)."""
    return delta_cards if bc.name == _VISIONARY_LEADER else 0


def bonus_count_delta_for_play_habitat(
    bc: cards.BonusCard, player: state.Player, habitat: cards.Habitat
) -> int:
    """Change in ``bc``'s qualifying count from playing one more bird into
    ``habitat`` — nonzero only for the habitat-spread dynamic card (Ecologist,
    whose count is the minimum row length): +1 exactly when ``habitat`` is the
    unique smallest row."""
    if bc.name != _ECOLOGIST:
        return 0
    row_lengths = {hab: len(player.board[hab]) for hab in cards.ALL_HABITATS}
    before_min = min(row_lengths.values())
    row_lengths[habitat] += 1
    return min(row_lengths.values()) - before_min


def bonus_count_delta_for_move(
    bc: cards.BonusCard,
    player: state.Player,
    from_habitat: cards.Habitat,
    to_habitat: cards.Habitat,
) -> int:
    """Change in ``bc``'s qualifying count from moving one bird between
    habitats — nonzero only for Ecologist (the min row length can shift when a
    bird leaves the smallest row or joins it)."""
    if bc.name != _ECOLOGIST or from_habitat == to_habitat:
        return 0
    row_lengths = {hab: len(player.board[hab]) for hab in cards.ALL_HABITATS}
    before_min = min(row_lengths.values())
    row_lengths[from_habitat] -= 1
    row_lengths[to_habitat] += 1
    return min(row_lengths.values()) - before_min


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
        engine.log_global(
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
    """Number of things currently qualifying for bonus card ``bc``.

    For the four *dynamic* cards — whose conditions read live game state
    rather than a printed per-bird trait — the count comes from the matching
    ``_DYNAMIC_BONUS_COUNTERS`` entry (eggs on birds, hand size, habitat
    spread). For every other card it is the static count: ``player``'s in-play
    birds whose ``bonus_categories`` include ``bc.name``."""
    dynamic_counter = _DYNAMIC_BONUS_COUNTERS.get(bc.name)
    if dynamic_counter is not None:
        return dynamic_counter(player)
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


def _habitat_goal_count_delta(
    bird: cards.Bird,
    goal_habitat: cards.Habitat,
    play_habitat: cards.Habitat | None,
) -> int:
    """Count delta for a ``birds_<habitat>`` goal from playing ``bird``: exact
    when the landing row is committed (``play_habitat``), the optimistic
    any-card-habitat bound when it is not (see
    :func:`goal_count_delta_for_bird`)."""
    if play_habitat is not None:
        return 1 if play_habitat == goal_habitat else 0
    return 1 if goal_habitat in bird.habitats else 0


def _candidate_habitats(
    bird: cards.Bird, player: state.Player | None
) -> list[cards.Habitat]:
    """Card habitats an uncommitted play of ``bird`` could reach: every
    printed habitat when ``player`` is unknown (setup pricing over an empty
    board), else only those with an open row slot (see
    :func:`goal_count_delta_for_bird_with_eggs`)."""
    if player is None:
        return list(bird.habitats)
    return [habitat for habitat in bird.habitats if player.can_play_in(habitat)]


def _my_placement_vp(
    my_count: int,
    other_counts: typing.Sequence[int],
    payouts: typing.Sequence[int],
) -> int:
    """The deciding player's placement VP given their own category count and
    every other seat's count — the shared pricer behind the goal-delta
    helpers above, each of which calls this once for the "before" count and
    once for the "after" count. A thin wrapper over :func:`placement_payouts`
    that reads off the first (the deciding player's own) slot."""
    return placement_payouts([my_count, *other_counts], payouts)[0]


def _count_birds_in(habitat: cards.Habitat) -> typing.Callable[[state.Player], int]:
    return lambda player: len(player.board[habitat])


def _count_eggs_in_habitat(
    habitat: cards.Habitat,
) -> typing.Callable[[state.Player], int]:
    return lambda player: sum(pb.eggs for pb in player.board[habitat])


def _count_eggs_on_nest(nest: cards.NestType) -> typing.Callable[[state.Player], int]:
    # ``cards.nest_matches``: star nests are wild and count toward every
    # concrete-nest goal (official rule).
    return lambda player: sum(
        pb.eggs
        for row in player.board.values()
        for pb in row
        if cards.nest_matches(pb.bird.nest, nest)
    )


def _count_birds_with_eggs_on_nest(
    nest: cards.NestType,
) -> typing.Callable[[state.Player], int]:
    return lambda player: sum(
        1
        for row in player.board.values()
        for pb in row
        if cards.nest_matches(pb.bird.nest, nest) and pb.eggs > 0
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


def _count_birds_no_eggs(player: state.Player) -> int:
    return sum(1 for row in player.board.values() for pb in row if pb.eggs == 0)


# Category-name keys the egg / move delta helpers branch on, parallel to the
# matching ``_CATEGORY_COUNTERS`` entries (kept adjacent so adding a goal
# category updates both together).
_BIRDS_IN_HABITAT_CATEGORIES: dict[str, cards.Habitat] = {
    "birds_forest": cards.Habitat.FOREST,
    "birds_grassland": cards.Habitat.GRASSLAND,
    "birds_wetland": cards.Habitat.WETLAND,
}
_EGGS_IN_HABITAT_CATEGORIES: dict[str, cards.Habitat] = {
    "eggs_forest": cards.Habitat.FOREST,
    "eggs_grassland": cards.Habitat.GRASSLAND,
    "eggs_wetland": cards.Habitat.WETLAND,
}
_EGGS_ON_NEST_CATEGORIES: dict[str, cards.NestType] = {
    "eggs_bowl": cards.NestType.BOWL,
    "eggs_cavity": cards.NestType.CAVITY,
    "eggs_ground": cards.NestType.GROUND,
    "eggs_platform": cards.NestType.PLATFORM,
}
_BIRDS_WITH_EGGS_CATEGORIES: dict[str, cards.NestType] = {
    "bowl_birds_with_eggs": cards.NestType.BOWL,
    "cavity_birds_with_eggs": cards.NestType.CAVITY,
    "ground_birds_with_eggs": cards.NestType.GROUND,
    "platform_birds_with_eggs": cards.NestType.PLATFORM,
}


def _per_habitat_egg_sums(player: state.Player) -> dict[cards.Habitat, int]:
    return {
        habitat: sum(pb.eggs for pb in player.board[habitat])
        for habitat in cards.ALL_HABITATS
    }


def _egg_sets_delta_with_bird(
    bird: cards.Bird,
    play_habitat: cards.Habitat | None,
    player: state.Player | None,
) -> int:
    """``egg_sets_3habitats`` branch of
    :func:`goal_count_delta_for_bird_with_eggs`: 0 with no board context (the
    setup path prices this hand-level via :func:`goal_affinity_for_kept`
    instead); otherwise the raised-min delta from adding ``bird.egg_limit`` to
    the committed habitat, or the best such delta over every reachable
    habitat when uncommitted (0 if none are reachable)."""
    if player is None:
        return 0
    egg_sums = _per_habitat_egg_sums(player)
    before_sets = min(egg_sums.values())
    if play_habitat is not None:
        egg_sums[play_habitat] += bird.egg_limit
        return min(egg_sums.values()) - before_sets
    deltas = [
        min({**egg_sums, habitat: egg_sums[habitat] + bird.egg_limit}.values())
        - before_sets
        for habitat in _candidate_habitats(bird, player)
    ]
    return max(deltas, default=0)


def _best_kept_egg_sets(kept: typing.Sequence[cards.Bird]) -> int:
    """Best achievable ``egg_sets_3habitats`` count from a setup keep: assign
    each kept bird to exactly one of its card habitats to maximize the min
    across the three per-habitat summed ``egg_limit`` totals. Brute-forced
    over every assignment (kept hands are capped at 5 birds, so at most
    :math:`3^5 = 243` combinations — see :func:`goal_affinity_for_kept`)."""
    if not kept:
        return 0
    best = 0
    for assignment in itertools.product(*(bird.habitats for bird in kept)):
        habitat_sums = dict.fromkeys(cards.ALL_HABITATS, 0)
        for bird, habitat in zip(kept, assignment):
            habitat_sums[habitat] += bird.egg_limit
        best = max(best, min(habitat_sums.values()))
    return best


def _best_case_egg_count_delta(player: state.Player, category: str, n_eggs: int) -> int:
    """The count bound behind :func:`goal_best_case_for_eggs`: capacity-capped
    optimistic laying for ``n_eggs > 0``, exact least-damage single-egg removal
    for ``n_eggs < 0`` (every removal commitment offered takes one egg)."""
    if n_eggs == 0:
        return 0

    # Removal: the best (least bad) single egg currently on the board, priced
    # exactly through the per-target helper.
    if n_eggs < 0:
        best: int | None = None
        for habitat in cards.ALL_HABITATS:
            for pb in player.board[habitat]:
                if pb.eggs <= 0:
                    continue
                delta = goal_count_delta_for_egg(player, habitat, pb, category, -1)
                best = delta if best is None else max(best, delta)
        return 0 if best is None else best

    # Laying toward a per-habitat or per-nest egg total: capped by real room.
    goal_habitat = _EGGS_IN_HABITAT_CATEGORIES.get(category)
    if goal_habitat is not None:
        capacity = sum(
            max(pb.bird.egg_limit - pb.eggs, 0) for pb in player.board[goal_habitat]
        )
        return min(n_eggs, capacity)
    goal_nest = _EGGS_ON_NEST_CATEGORIES.get(category)
    if goal_nest is not None:
        capacity = sum(
            max(pb.bird.egg_limit - pb.eggs, 0)
            for row in player.board.values()
            for pb in row
            if cards.nest_matches(pb.bird.nest, goal_nest)
        )
        return min(n_eggs, capacity)

    # Laying toward a birds-with-eggs count: one egg per empty matching bird.
    goal_nest = _BIRDS_WITH_EGGS_CATEGORIES.get(category)
    if goal_nest is not None:
        empty_matching = sum(
            1
            for row in player.board.values()
            for pb in row
            if cards.nest_matches(pb.bird.nest, goal_nest)
            and pb.eggs == 0
            and pb.bird.egg_limit > 0
        )
        return min(n_eggs, empty_matching)

    if category == "egg_sets_3habitats":
        return _best_case_egg_sets_raise(player, n_eggs)

    # Laying against the no-egg count (an anti-goal): the optimistic player
    # routes eggs onto birds that already have some; only the overflow past
    # that spare room is forced onto eggless birds, concentrated on the
    # roomiest ones so as few as possible lose their no-egg status.
    if category == "birds_no_eggs":
        return _best_case_no_egg_birds_lost(player, n_eggs)

    return 0


def _best_case_egg_sets_raise(player: state.Player, n_eggs: int) -> int:
    """How far ``n_eggs`` optimally-placed eggs can raise the egg-set count
    (the min across per-habitat egg totals), respecting per-bird egg room —
    greedy water-filling: each egg goes to a lowest habitat that still has
    capacity, stopping when the minimum is capacity-frozen."""
    egg_sums = _per_habitat_egg_sums(player)
    capacities = {
        habitat: sum(
            max(pb.bird.egg_limit - pb.eggs, 0) for pb in player.board[habitat]
        )
        for habitat in cards.ALL_HABITATS
    }
    before_sets = min(egg_sums.values())
    for _ in range(n_eggs):
        floor = min(egg_sums.values())
        open_floor_habitats = [
            habitat
            for habitat in cards.ALL_HABITATS
            if egg_sums[habitat] == floor and capacities[habitat] > 0
        ]
        if not open_floor_habitats:
            break  # the minimum row has no room left; the set count is frozen
        target = open_floor_habitats[0]
        egg_sums[target] += 1
        capacities[target] -= 1
    return min(egg_sums.values()) - before_sets


def _best_case_no_egg_birds_lost(player: state.Player, n_eggs: int) -> int:
    """The fewest currently-eggless birds that must take an egg when laying
    ``n_eggs`` optimally (returned as a non-positive count delta for the
    ``birds_no_eggs`` goal): spare room on already-egged birds absorbs eggs
    for free, and any overflow fills the roomiest eggless birds first."""
    spare_on_egged = sum(
        max(pb.bird.egg_limit - pb.eggs, 0)
        for row in player.board.values()
        for pb in row
        if pb.eggs > 0
    )
    overflow = n_eggs - spare_on_egged
    if overflow <= 0:
        return 0

    eggless_rooms = sorted(
        (
            pb.bird.egg_limit
            for row in player.board.values()
            for pb in row
            if pb.eggs == 0 and pb.bird.egg_limit > 0
        ),
        reverse=True,
    )
    birds_lost = 0
    for room in eggless_rooms:
        if overflow <= 0:
            break
        birds_lost += 1
        overflow -= room
    return -birds_lost


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
    "birds_no_eggs": _count_birds_no_eggs,
}


#### Dynamic bonus cards ####

# The four core bonus cards whose conditions read live game state instead of a
# printed per-bird trait. They tag no birds in ``bonus_categories``, so the
# static counter would always return 0; ``bonus_qualifying_count`` consults
# ``_DYNAMIC_BONUS_COUNTERS`` for them instead. Keys are printed card names
# (the same matching convention the static tags use).
_BREEDING_MANAGER = "Breeding Manager"
_OOLOGIST = "Oologist"
_VISIONARY_LEADER = "Visionary Leader"
_ECOLOGIST = "Ecologist"

# Minimum eggs sitting on a bird before it qualifies, per egg-counting card
# ("Birds that have at least 4 / at least 1 egg laid on them").
_EGG_COUNT_BONUS_MIN_EGGS: dict[str, int] = {
    _BREEDING_MANAGER: 4,
    _OOLOGIST: 1,
}


def _count_birds_with_min_eggs(min_eggs: int) -> typing.Callable[[state.Player], int]:
    return lambda player: sum(
        1 for row in player.board.values() for pb in row if pb.eggs >= min_eggs
    )


def _count_hand_cards(player: state.Player) -> int:
    return len(player.hand)


def _count_fewest_habitat_birds(player: state.Player) -> int:
    """Ecologist's count: the number of birds in the habitat with the fewest
    birds. The card's printed note pins the tie rule — "if all of your habitats
    have 3 birds in them, your habitat with the fewest birds has 3 birds in
    it" — so a tie never disqualifies and the count is simply the minimum row
    length."""
    return min(len(player.board[habitat]) for habitat in cards.ALL_HABITATS)


_DYNAMIC_BONUS_COUNTERS: dict[str, typing.Callable[[state.Player], int]] = {
    _BREEDING_MANAGER: _count_birds_with_min_eggs(
        _EGG_COUNT_BONUS_MIN_EGGS[_BREEDING_MANAGER]
    ),
    _OOLOGIST: _count_birds_with_min_eggs(_EGG_COUNT_BONUS_MIN_EGGS[_OOLOGIST]),
    _VISIONARY_LEADER: _count_hand_cards,
    _ECOLOGIST: _count_fewest_habitat_birds,
}
