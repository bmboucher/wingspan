"""``python -m wingspan.training`` -> the live training dashboard."""

from __future__ import annotations

from wingspan.training import app

if __name__ == "__main__":
    raise SystemExit(app.main())
