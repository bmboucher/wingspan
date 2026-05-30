"""Host telemetry sampling for the dashboard's SYSTEM band.

:class:`SystemMonitor` turns the live machine state into a
:class:`metrics.SystemStats` snapshot — system-wide CPU utilization, system RAM
in use, and this process's resident memory, all from :mod:`psutil`. Sampling is
*total*: any backend hiccup degrades a single field to a safe default rather
than raising into the monitor thread that calls it.
"""

from __future__ import annotations

import logging

import psutil

from wingspan.training import metrics

# Bytes per binary GB, matching the GiB figures Task Manager / `free` show.
_BYTES_PER_GB = 1024**3

_log = logging.getLogger(__name__)


class SystemMonitor:
    """Samples host CPU / RAM utilization.

    Construct once per run (it caches a handle to this process), then call
    :meth:`sample` on each monitor tick.
    """

    def __init__(self) -> None:
        self._process = psutil.Process()
        # psutil's percent meters report the delta since their previous call,
        # so prime both now; the first real sample then spans one monitor tick.
        psutil.cpu_percent(interval=None)
        self._process.cpu_percent(interval=None)

    def sample(self) -> metrics.SystemStats:
        """A best-effort snapshot of host load (never raises)."""
        try:
            memory = psutil.virtual_memory()
            return metrics.SystemStats(
                cpu_percent=psutil.cpu_percent(interval=None),
                ram_used_gb=memory.used / _BYTES_PER_GB,
                ram_total_gb=memory.total / _BYTES_PER_GB,
                proc_rss_gb=self._process_rss_gb(),
            )
        except Exception:  # noqa: BLE001 — telemetry must never break sampling
            _log.debug("system telemetry sample failed", exc_info=True)
            return metrics.SystemStats(
                cpu_percent=0.0, ram_used_gb=0.0, ram_total_gb=0.0, proc_rss_gb=0.0
            )

    ###### PRIVATE #######

    def _process_rss_gb(self) -> float:
        try:
            return self._process.memory_info().rss / _BYTES_PER_GB
        except psutil.Error:
            return 0.0
