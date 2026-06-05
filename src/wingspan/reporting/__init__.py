"""Model introspection and HTML report generation.

Re-exports the two public entry points so callers can either import from
this package (``reporting.generate_html_report``) or from the leaf modules
(``html.generate_html_report``, ``inspect_cli.main_inspect``).
"""

from __future__ import annotations

from wingspan.reporting.html import generate_html_report
from wingspan.reporting.inspect_cli import main_inspect

__all__ = ["generate_html_report", "main_inspect"]
