"""CSV export for the timeline data embedded in a ``wingspan play --html`` game log.

The timeline modal in the HTML viewer is backed by a list of
:class:`~wingspan.reporting.game_log_html.TimelinePoint` objects already
captured per decision.  This module projects that list into a CSV that can be
downloaded directly from the modal without leaving the page.

The CSV rows mirror the chart's two interleaved per-seat series: each decision
records the critic value and training target only for the seat that just moved
(the other seat's columns are blank), which matches exactly what the bottom SVG
panel plots.  All critic / target values are **P0-relative future-return
margins in VP** (the P1 net's prediction is already sign-flipped to P0-relative
before being stored in :attr:`TimelinePoint.value_return_p0`).

Public API:
  :func:`timeline_to_csv` — GameLogReport → CSV string
  :func:`timeline_csv_data_uri` — GameLogReport → ``data:text/csv;…;base64,…``
"""

from __future__ import annotations

import base64
import csv
import io
import typing

if typing.TYPE_CHECKING:
    from wingspan.reporting import game_log_html

_CSV_HEADER: list[str] = [
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


def timeline_to_csv(report: game_log_html.GameLogReport) -> str:
    """Render ``report.timeline`` as a CSV string (UTF-8, header + one row per decision).

    Critic and target columns are sparse: each row fills only the moving
    player's pair of cells; the other seat's cells are empty strings.  ``None``
    values (forced / setup decisions, or a seat without a trained net) also
    render as empty cells.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_CSV_HEADER)

    for point in report.timeline:
        player_name = (
            report.player_names[point.player_id]
            if point.player_id < len(report.player_names)
            else str(point.player_id)
        )

        # Route value/target into the moving seat's column; leave the other blank.
        if point.player_id == 0:
            p0_critic = "" if point.value_return_p0 is None else point.value_return_p0
            p1_critic = ""
            p0_target = "" if point.target_return_p0 is None else point.target_return_p0
            p1_target = ""
        else:
            p0_critic = ""
            p1_critic = "" if point.value_return_p0 is None else point.value_return_p0
            p0_target = ""
            p1_target = "" if point.target_return_p0 is None else point.target_return_p0

        writer.writerow(
            [
                point.timestamp,
                point.phase_index,
                point.player_id,
                player_name,
                point.score_p0,
                point.score_p1,
                p0_critic,
                p1_critic,
                p0_target,
                p1_target,
            ]
        )

    return buf.getvalue()


def timeline_csv_data_uri(report: game_log_html.GameLogReport) -> str:
    """Return a ``data:`` URI that browsers can use as a download link href.

    The CSV text is base64-encoded so embedded commas and newlines cannot break
    the ``href`` attribute value regardless of quoting style.
    """
    csv_bytes = timeline_to_csv(report).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode("ascii")
    return f"data:text/csv;charset=utf-8;base64,{b64}"
