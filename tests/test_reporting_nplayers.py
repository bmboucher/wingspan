"""Tests for N-player support in the HTML/CSV game-log reporting (Stage 4).

Drives ``wingspan play --html`` end-to-end at 3 seats and inspects the
rendered document: the seat-2 view-toggle button, the embedded JSON payload's
per-seat data, and the generated seat-2 CSS. Also confirms the downloadable
CSV grows a ``score_p2`` / ``p2_critic_value`` / ``p2_target_value`` block at
3 seats while staying byte-identical to the legacy header at 2 seats.
"""

from __future__ import annotations

import base64
import json
import pathlib
import typing

from wingspan import cli
from wingspan.reporting import game_log_csv, game_log_html

_LEGACY_2P_CSV_HEADER = (
    "timestamp,phase_index,player_id,player_name,score_p0,score_p1,"
    "p0_critic_value,p1_critic_value,p0_target_value,p1_target_value"
)


def _render_html(tmp_path: pathlib.Path, num_seats: int, seed: int) -> str:
    """Run one ``wingspan play --html`` game at ``num_seats`` and return the
    rendered HTML document's full text."""
    argv = ["--p0", "random", "--p1", "random"]
    if num_seats >= 3:
        argv += ["--p2", "random"]
    if num_seats >= 4:
        argv += ["--p3", "random"]
    out_path = tmp_path / "game.html"
    argv += [
        "--seed",
        str(seed),
        "--quiet",
        "--html",
        str(out_path),
        "--instrument-out",
        str(tmp_path),
    ]
    exit_code = cli.main_play(argv)
    assert exit_code == 0
    return out_path.read_text(encoding="utf-8")


def _extract_json_payload(html: str) -> dict[str, typing.Any]:
    """The embedded ``#game-log-data`` JSON island, decoded."""
    marker = 'id="game-log-data"'
    start = html.index(marker)
    payload_start = html.index(">", start) + 1
    payload_end = html.index("</script>", payload_start)
    return json.loads(html[payload_start:payload_end])


def _extract_csv(html: str) -> str:
    """The base64-decoded CSV text embedded as the timeline download link."""
    prefix = "data:text/csv;charset=utf-8;base64,"
    start = html.index(prefix) + len(prefix)
    end = html.index('"', start)
    return base64.b64decode(html[start:end]).decode("utf-8")


###### HTML render smoke at N=3 ######


def test_3seat_html_view_toggle_has_seat2_button(tmp_path: pathlib.Path):
    """The seat-view toggle grows a 'Just P2' / data-view="p2" button, and the
    2-seat 'both' id is replaced by the generalized 'all' group id."""
    html = _render_html(tmp_path, num_seats=3, seed=11)
    assert 'data-view="p2"' in html
    assert 'data-view="all"' in html
    assert 'data-view="both"' not in html


def test_3seat_html_payload_carries_three_seat_series(tmp_path: pathlib.Path):
    """The embedded JSON payload has 3 player names and every timeline point's
    ``scores`` list has 3 entries."""
    html = _render_html(tmp_path, num_seats=3, seed=12)
    payload = _extract_json_payload(html)
    assert len(payload["player_names"]) == 3
    timeline = payload["timeline"]
    assert timeline, "a full game records at least one timeline point"
    assert all(len(point["scores"]) == 3 for point in timeline)
    assert {point["player_id"] for point in timeline} <= {0, 1, 2}


def test_3seat_html_generated_seat2_css_present(tmp_path: pathlib.Path):
    """The generated per-seat CSS defines seat-2 color rules across every
    panel that themes by seat (bars, decision boxes, notes, timeline lines)."""
    html = _render_html(tmp_path, num_seats=3, seed=13)
    assert ".bar.p2 {" in html
    assert ".di.p2 summary" in html
    assert ".note.p2 {" in html
    assert ".chart-line-p2 {" in html
    assert ".chart-line-value-p2 {" in html
    assert ".chart-line-target-p2 {" in html
    assert ".chart-line-adv-p2 {" in html
    # Seats 0/1 keep their literal pre-N-player colors untouched.
    assert ".bar.p0     { background: #93c5fd; }" in html
    assert ".bar.p1     { background: #fca5a5; }" in html


def test_4seat_html_has_seat3_button_and_no_seat4(tmp_path: pathlib.Path):
    """At 4 seats the toggle covers p0..p3 (not a stray p4)."""
    html = _render_html(tmp_path, num_seats=4, seed=14)
    for seat in range(4):
        assert f'data-view="p{seat}"' in html
    assert 'data-view="p4"' not in html


###### CSV: N=3 grows columns; N=2 stays byte-identical ######


def test_3seat_csv_has_seat2_columns(tmp_path: pathlib.Path):
    """The downloadable CSV embedded in a 3-seat game log has score_p2 /
    p2_critic_value / p2_target_value columns."""
    html = _render_html(tmp_path, num_seats=3, seed=15)
    csv_text = _extract_csv(html)
    header = csv_text.splitlines()[0]
    assert "score_p2" in header
    assert "p2_critic_value" in header
    assert "p2_target_value" in header


def test_2seat_csv_header_is_exactly_the_legacy_header(tmp_path: pathlib.Path):
    """At 2 seats the CSV header is EXACTLY today's legacy string — proof the
    dynamic column generation reproduces the pre-N-player shape byte-for-byte."""
    html = _render_html(tmp_path, num_seats=2, seed=16)
    csv_text = _extract_csv(html)
    header = csv_text.splitlines()[0]
    assert header == _LEGACY_2P_CSV_HEADER


def test_csv_header_helper_matches_legacy_string_directly():
    """Same assertion at the module level (no CLI / engine involved) —
    :func:`game_log_csv.timeline_to_csv` on an empty 2-seat report."""
    report = game_log_html.GameLogReport(
        player_names=["P0", "P1"], phases=[], timeline=[]
    )
    header = game_log_csv.timeline_to_csv(report).splitlines()[0]
    assert header == _LEGACY_2P_CSV_HEADER
