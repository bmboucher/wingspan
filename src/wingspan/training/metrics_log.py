"""Cached reader for a run's append-only ``metrics.jsonl`` history log.

The convergence charts in :mod:`wingspan.training.charts` need the *complete*
per-iteration history — beyond the in-memory ``RunState.history`` cap — so the
WIN RATE chart can span the whole run and the FINAL SCORE / MARGIN chart can show
a ``geometry.WINDOW_SCORE_MARGIN``-iteration window even though only
``config.history_len`` rows are kept in memory. This module reads every ``IterationMetrics`` row from disk and memoises
the parse, re-reading only when the file's size or modification time changes, so
the dashboard (which repaints several times a second) never re-parses thousands
of rows per frame.
"""

from __future__ import annotations

import pathlib

import pydantic

from wingspan.training import artifacts, metrics

# The parse cache, keyed by the resolved ``metrics.jsonl`` path. Module-level so
# the memoisation survives across the many dashboard refreshes that each build a
# fresh chart renderable.
_CACHE: dict[str, "_CachedLog"] = {}


class _CachedLog(pydantic.BaseModel):
    """A parsed ``metrics.jsonl`` snapshot tagged with the file fingerprint it was
    read from, so a later read can tell whether the file has grown since."""

    size: int
    mtime: float
    rows: list[metrics.IterationMetrics]


def read_iteration_history(checkpoint_dir: str) -> list[metrics.IterationMetrics]:
    """Every ``IterationMetrics`` row in ``checkpoint_dir/metrics.jsonl``.

    Returns an empty list when the log does not exist yet (a brand-new run, or a
    test with no on-disk run), so callers can fall back to the in-memory history.
    The parse is memoised per path and only redone when the file's size or
    modification time changes.
    """
    path = pathlib.Path(checkpoint_dir) / artifacts.METRICS_LOG
    try:
        stat = path.stat()
    except OSError:
        return []

    key = str(path.resolve())
    cached = _CACHE.get(key)
    if (
        cached is not None
        and cached.size == stat.st_size
        and cached.mtime == stat.st_mtime
    ):
        return cached.rows

    entry = _CachedLog(size=stat.st_size, mtime=stat.st_mtime, rows=_parse(path))
    _CACHE[key] = entry
    # Return the stored model's ``rows`` (pydantic re-wraps the list on
    # construction) so a subsequent cached read returns the *same* object.
    return entry.rows


def _parse(path: pathlib.Path) -> list[metrics.IterationMetrics]:
    """Parse one ``IterationMetrics`` per non-blank line, tolerating a truncated
    final line left behind by a crash mid-append."""
    rows: list[metrics.IterationMetrics] = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                rows.append(metrics.IterationMetrics.model_validate_json(line))
            except pydantic.ValidationError:
                continue
    return rows
