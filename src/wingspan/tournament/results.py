"""Per-game results and the aggregated tournament report (the JSON root).

A finished game is a :class:`GameResult`; :func:`aggregate` rolls every game
into the :class:`TournamentReport` written to disk — per-pair win rates and
average point margins split by who went first, plus each competitor's final Elo
and overall record. Win rate counts a tie as half a win throughout (matching the
training evaluator's convention).
"""

from __future__ import annotations

import typing

import pydantic

from wingspan.tournament import config, elo, schedule


class GameResult(pydantic.BaseModel):
    """One finished tournament game, from competitor A's point of view.

    ``a_was_start_player`` records who actually went first (the deal's random
    first player mapped through A's seat), which is what the first/second splits
    bucket on — not the orientation label.
    """

    round_index: int
    pair_index: int
    orientation: schedule.Orientation
    player_a_id: str
    player_b_id: str
    a_score: int
    b_score: int
    a_was_start_player: bool

    @property
    def score_a(self) -> float:
        """A's game outcome as an Elo/​win-rate score: 1.0 win, 0.5 tie, 0.0 loss."""
        if self.a_score > self.b_score:
            return 1.0
        if self.a_score < self.b_score:
            return 0.0
        return 0.5


class SplitStats(pydantic.BaseModel):
    """Win rate and average point margin over one subset of a pair's games, from
    a fixed competitor's perspective (ties count as half a win)."""

    games: int = 0
    wins: float = 0.0
    win_rate: float = 0.0
    avg_margin: float = 0.0


class MatchupResult(pydantic.BaseModel):
    """One unordered pair's full result, broken down from each side as first
    player, as second player, and overall."""

    player_a_id: str
    player_b_id: str
    a_first: SplitStats
    a_second: SplitStats
    a_overall: SplitStats
    b_first: SplitStats
    b_second: SplitStats
    b_overall: SplitStats


class ParticipantResult(pydantic.BaseModel):
    """One competitor's tournament-wide standing."""

    id: str
    display_name: str
    final_elo: float
    wins: int
    losses: int
    ties: int
    win_rate: float
    avg_margin: float


class TournamentReport(pydantic.BaseModel):
    """The complete tournament result — the JSON written to ``out_path``."""

    config: config.TournamentConfig
    participants: list[ParticipantResult]
    matchups: list[MatchupResult]
    games: list[GameResult]


def aggregate(
    cfg: config.TournamentConfig, games: typing.Sequence[GameResult]
) -> TournamentReport:
    """Roll finished ``games`` into the full report: deterministic final Elo,
    per-pair first/second/overall splits, and per-competitor records. Competitors
    are sorted by final Elo (descending)."""
    table = elo.replay(cfg.participant_ids, cfg.elo_init, cfg.elo_k, games)
    display = {spec.id: spec.display_name for spec in cfg.participants}

    grouped = _group_by_pair(games)
    matchups = [_matchup(id_a, id_b, group) for (id_a, id_b), group in grouped.items()]

    sides = _sides_by_competitor(cfg.participant_ids, games)
    standings = [
        _participant_result(competitor_id, display[competitor_id], table, side)
        for competitor_id, side in sides.items()
    ]
    standings.sort(key=lambda result: result.final_elo, reverse=True)

    ordered_games = sorted(
        games,
        key=lambda game: (game.round_index, game.pair_index, int(game.orientation)),
    )
    return TournamentReport(
        config=cfg, participants=standings, matchups=matchups, games=ordered_games
    )


###### PRIVATE #######


class _Side(pydantic.BaseModel):
    """One competitor's (score, margin) entries across all its games, tagged with
    whether it went first — the raw material for its record."""

    scores: list[float] = pydantic.Field(default_factory=list[float])
    margins: list[float] = pydantic.Field(default_factory=list[float])


def _group_by_pair(
    games: typing.Sequence[GameResult],
) -> dict[tuple[str, str], list[GameResult]]:
    """Games grouped by their unordered pair, preserving first-seen pair order."""
    grouped: dict[tuple[str, str], list[GameResult]] = {}
    for game in games:
        grouped.setdefault((game.player_a_id, game.player_b_id), []).append(game)
    return grouped


def _matchup(id_a: str, id_b: str, group: typing.Sequence[GameResult]) -> MatchupResult:
    """Build one pair's six-way split (A/B × first/second/overall)."""
    a_first = [(g.score_a, _margin_a(g)) for g in group if g.a_was_start_player]
    a_second = [(g.score_a, _margin_a(g)) for g in group if not g.a_was_start_player]
    b_first = [
        (1.0 - g.score_a, -_margin_a(g)) for g in group if not g.a_was_start_player
    ]
    b_second = [(1.0 - g.score_a, -_margin_a(g)) for g in group if g.a_was_start_player]
    return MatchupResult(
        player_a_id=id_a,
        player_b_id=id_b,
        a_first=_split(a_first),
        a_second=_split(a_second),
        a_overall=_split(a_first + a_second),
        b_first=_split(b_first),
        b_second=_split(b_second),
        b_overall=_split(b_first + b_second),
    )


def _split(entries: typing.Sequence[tuple[float, float]]) -> SplitStats:
    """Win rate + average margin over (score, margin) entries (ties = 0.5 win)."""
    games = len(entries)
    if games == 0:
        return SplitStats()
    wins = sum(score for score, _ in entries)
    margin_total = sum(margin for _, margin in entries)
    return SplitStats(
        games=games, wins=wins, win_rate=wins / games, avg_margin=margin_total / games
    )


def _sides_by_competitor(
    ids: typing.Sequence[str], games: typing.Sequence[GameResult]
) -> dict[str, _Side]:
    """Each competitor's score/margin entries across every game it played."""
    sides: dict[str, _Side] = {competitor_id: _Side() for competitor_id in ids}
    for game in games:
        margin_a = _margin_a(game)
        sides[game.player_a_id].scores.append(game.score_a)
        sides[game.player_a_id].margins.append(margin_a)
        sides[game.player_b_id].scores.append(1.0 - game.score_a)
        sides[game.player_b_id].margins.append(-margin_a)
    return sides


def _participant_result(
    competitor_id: str, display_name: str, table: elo.EloTable, side: _Side
) -> ParticipantResult:
    """Roll one competitor's entries into wins/losses/ties + averages."""
    games = len(side.scores)
    wins = sum(1 for score in side.scores if score == 1.0)
    losses = sum(1 for score in side.scores if score == 0.0)
    ties = sum(1 for score in side.scores if score == 0.5)
    win_rate = sum(side.scores) / games if games else 0.0
    avg_margin = sum(side.margins) / games if games else 0.0
    return ParticipantResult(
        id=competitor_id,
        display_name=display_name,
        final_elo=table.ratings[competitor_id],
        wins=wins,
        losses=losses,
        ties=ties,
        win_rate=win_rate,
        avg_margin=avg_margin,
    )


def _margin_a(game: GameResult) -> float:
    """Competitor A's point margin in a game (its score minus B's)."""
    return float(game.a_score - game.b_score)
