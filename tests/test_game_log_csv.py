"""Tests for :mod:`wingspan.reporting.game_log_csv`.

Constructs minimal Pydantic objects directly — no engine required — to verify
that :func:`timeline_to_csv` and :func:`timeline_csv_data_uri` produce the
correct output for all routing branches.
"""

from __future__ import annotations

import base64
import csv
import io

from wingspan.reporting import game_log_csv, game_log_html


def _make_report(
    points: list[game_log_html.TimelinePoint],
) -> game_log_html.GameLogReport:
    return game_log_html.GameLogReport(
        player_names=["Alice", "Bob"],
        phases=[],
        timeline=points,
    )


def _parse_csv(text: str) -> list[list[str]]:
    return list(csv.reader(io.StringIO(text)))


class TestTimelineToCsv:
    def test_header_columns(self) -> None:
        report = _make_report([])
        rows = _parse_csv(game_log_csv.timeline_to_csv(report))
        assert rows[0] == [
            "timestamp",
            "phase_index",
            "player_id",
            "player_name",
            "score_p0",
            "score_p1",
            "p0_critic_value",
            "p1_critic_value",
            "p0_target_value",
            "p1_target_value",
        ]

    def test_empty_timeline_yields_header_only(self) -> None:
        report = _make_report([])
        rows = _parse_csv(game_log_csv.timeline_to_csv(report))
        assert len(rows) == 1

    def test_p0_decision_fills_p0_columns(self) -> None:
        point = game_log_html.TimelinePoint(
            timestamp=1.0,
            player_id=0,
            score_p0=10,
            score_p1=8,
            phase_index=3,
            value_return_p0=2.5,
            target_return_p0=3.0,
        )
        report = _make_report([point])
        rows = _parse_csv(game_log_csv.timeline_to_csv(report))
        assert len(rows) == 2
        row = rows[1]
        assert row[0] == "1.0"  # timestamp
        assert row[1] == "3"  # phase_index
        assert row[2] == "0"  # player_id
        assert row[3] == "Alice"  # player_name
        assert row[4] == "10"  # score_p0
        assert row[5] == "8"  # score_p1
        assert row[6] == "2.5"  # p0_critic_value filled
        assert row[7] == ""  # p1_critic_value blank
        assert row[8] == "3.0"  # p0_target_value filled
        assert row[9] == ""  # p1_target_value blank

    def test_p1_decision_fills_p1_columns(self) -> None:
        point = game_log_html.TimelinePoint(
            timestamp=2.33,
            player_id=1,
            score_p0=12,
            score_p1=15,
            phase_index=7,
            value_return_p0=-1.5,
            target_return_p0=-2.0,
        )
        report = _make_report([point])
        rows = _parse_csv(game_log_csv.timeline_to_csv(report))
        row = rows[1]
        assert row[2] == "1"  # player_id
        assert row[3] == "Bob"  # player_name
        assert row[6] == ""  # p0_critic_value blank
        assert row[7] == "-1.5"  # p1_critic_value filled
        assert row[8] == ""  # p0_target_value blank
        assert row[9] == "-2.0"  # p1_target_value filled

    def test_none_values_render_as_blank(self) -> None:
        """A decision with no trained net (value_return_p0=None) still emits scores."""
        point = game_log_html.TimelinePoint(
            timestamp=0.0,
            player_id=0,
            score_p0=5,
            score_p1=5,
            phase_index=0,
            value_return_p0=None,
            target_return_p0=None,
        )
        report = _make_report([point])
        rows = _parse_csv(game_log_csv.timeline_to_csv(report))
        row = rows[1]
        assert row[4] == "5"  # score_p0 still present
        assert row[5] == "5"  # score_p1 still present
        assert row[6] == ""  # p0_critic_value blank (None)
        assert row[7] == ""  # p1_critic_value blank (other seat)
        assert row[8] == ""  # p0_target_value blank (None)
        assert row[9] == ""  # p1_target_value blank (other seat)

    def test_row_order_matches_timeline_order(self) -> None:
        points = [
            game_log_html.TimelinePoint(
                timestamp=float(i),
                player_id=i % 2,
                score_p0=i,
                score_p1=i,
                phase_index=i,
            )
            for i in range(5)
        ]
        report = _make_report(points)
        rows = _parse_csv(game_log_csv.timeline_to_csv(report))
        assert len(rows) == 6  # header + 5 data rows
        for i, row in enumerate(rows[1:]):
            assert row[0] == str(float(i)), f"row {i} has wrong timestamp"


class TestTimelineCsvDataUri:
    def test_round_trip(self) -> None:
        point = game_log_html.TimelinePoint(
            timestamp=1.5,
            player_id=0,
            score_p0=7,
            score_p1=3,
            phase_index=2,
            value_return_p0=4.0,
            target_return_p0=4.5,
        )
        report = _make_report([point])
        uri = game_log_csv.timeline_csv_data_uri(report)
        prefix = "data:text/csv;charset=utf-8;base64,"
        assert uri.startswith(prefix)
        decoded = base64.b64decode(uri[len(prefix) :]).decode("utf-8")
        assert decoded == game_log_csv.timeline_to_csv(report)

    def test_uri_contains_csv_mime(self) -> None:
        report = _make_report([])
        uri = game_log_csv.timeline_csv_data_uri(report)
        assert "text/csv" in uri


class TestIntegration:
    """Confirm render_game_log_html embeds the download link."""

    def test_render_contains_csv_anchor(self) -> None:
        point = game_log_html.TimelinePoint(
            timestamp=1.0,
            player_id=0,
            score_p0=10,
            score_p1=8,
            phase_index=1,
        )
        report = _make_report([point])
        html = game_log_html.render_game_log_html(report)
        assert 'download="timeline.csv"' in html
        assert "data:text/csv" in html
