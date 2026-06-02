"""FLOCK WATCH: a read-only roster of all cloud runs, from their S3 status snapshots.

``python -m wingspan.cloud.monitor --bucket B --prefix runs`` (console script
``wingspan-monitor``) lists every run under the bucket prefix and refreshes a few
times a minute. It reads only each run's tiny ``status.json`` — never a
checkpoint — so watching a whole fleet is cheap. A run is "in-flight" when its
heartbeat is fresh and its phase is not terminal.
"""

from __future__ import annotations

import argparse
import datetime
import time

from rich import box, console, live, table, text

from wingspan.cloud import runfile, s3sync, status

_DEFAULT_REFRESH_SECONDS = 10.0
# A run counts as in-flight only if its heartbeat is newer than this multiple of
# its own status cadence — tolerating a missed upload or two, not a dead run.
_FRESH_MULTIPLE = 3.0


def main(argv: list[str] | None = None) -> int:
    """Poll the bucket and repaint the roster until interrupted."""
    args = _parse_args(argv)
    s3 = runfile.S3Config(
        bucket=args.bucket,
        prefix=args.prefix,
        region=args.region,
        endpoint_url=args.endpoint_url,
    )
    term = console.Console()
    with live.Live(console=term, screen=False, auto_refresh=False) as display:
        try:
            while True:
                display.update(_render(_safe_list(s3)), refresh=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            pass
    return 0


###### PRIVATE #######


def _safe_list(s3: runfile.S3Config) -> list[status.RunStatus]:
    """List statuses, returning an empty roster on a transient S3 error."""
    try:
        return s3sync.iter_run_statuses(s3)
    except (
        Exception
    ):  # noqa: BLE001 — a transient list error should not crash the monitor
        return []


def _render(statuses: list[status.RunStatus]) -> table.Table:
    now = datetime.datetime.now(datetime.UTC)
    grid = table.Table(
        title="WINGSPAN // FLOCK WATCH", box=box.SIMPLE_HEAVY, expand=True
    )
    grid.add_column("", width=1)  # in-flight LED
    grid.add_column("run", style="bold")
    grid.add_column("phase")
    grid.add_column("iters", justify="right")
    grid.add_column("%", justify="right")
    grid.add_column("games", justify="right")
    grid.add_column("avg", justify="right")
    grid.add_column("win vs challenger", justify="right")
    grid.add_column("ETA", justify="right")
    if not statuses:
        grid.add_row("", "(no runs found)", "", "", "", "", "", "", "")
        return grid
    for snapshot in sorted(statuses, key=lambda item: item.run_name):
        grid.add_row(*_row(snapshot, now))
    return grid


def _row(snapshot: status.RunStatus, now: datetime.datetime) -> tuple[text.Text, ...]:
    target = snapshot.target_iterations or snapshot.max_iterations
    return (
        _led(snapshot, now),
        text.Text(snapshot.run_name),
        text.Text(snapshot.phase),
        text.Text(f"{snapshot.completed_iterations}/{target or '∞'}"),
        text.Text(f"{snapshot.pct_complete:.0f}%"),
        text.Text(f"{snapshot.total_games:,}"),
        text.Text(f"{snapshot.avg_score:.1f}"),
        text.Text(_win_cell(snapshot)),
        text.Text(_eta_cell(snapshot.eta_seconds)),
    )


def _led(snapshot: status.RunStatus, now: datetime.datetime) -> text.Text:
    """A colored glyph: error ✗, finished ✓, in-flight ●, stale dim ●."""
    if snapshot.error:
        return text.Text("✗", style="bold red")
    if snapshot.finished:
        return text.Text("✓", style="blue")
    if _is_fresh(snapshot, now):
        return text.Text("●", style="bold green")
    return text.Text("●", style="dim yellow")


def _is_fresh(snapshot: status.RunStatus, now: datetime.datetime) -> bool:
    try:
        updated = datetime.datetime.fromisoformat(snapshot.updated_at)
    except ValueError:
        return False
    age_seconds = (now - updated).total_seconds()
    return age_seconds < _FRESH_MULTIPLE * snapshot.status_interval_seconds


def _win_cell(snapshot: status.RunStatus) -> str:
    if snapshot.win_rate is None:
        return "—"
    ci = (
        f" ±{snapshot.win_rate_ci95 * 100:.0f}"
        if snapshot.win_rate_ci95 is not None
        else ""
    )
    return f"{snapshot.win_rate * 100:.0f}%{ci} vs {snapshot.opponent_label}"


def _eta_cell(eta_seconds: float | None) -> str:
    if eta_seconds is None:
        return "—"
    total = int(eta_seconds)
    hours, remainder = divmod(total, 3600)
    minutes, _ = divmod(remainder, 60)
    return f"{hours:d}h {minutes:02d}m" if hours else f"{minutes:d}m"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wingspan-monitor",
        description="Watch all Wingspan cloud runs from their S3 status snapshots.",
    )
    parser.add_argument("--bucket", required=True, help="the S3 bucket holding runs")
    parser.add_argument(
        "--prefix", default="runs", help="the key prefix runs live under"
    )
    parser.add_argument("--region", default=None, help="AWS region")
    parser.add_argument(
        "--endpoint-url", default=None, help="S3 endpoint override (e.g. MinIO)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=_DEFAULT_REFRESH_SECONDS,
        help="seconds between roster refreshes",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
