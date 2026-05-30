"""The "FLIGHT PLAN" configurator screen — interactive pre-flight before FLYWAY
CONTROL.

``python -m wingspan.training --config`` opens this full-screen TUI to edit every
configurable :class:`~wingspan.training.config.TrainConfig` hyperparameter, manage
the runs stored in a checkpoint directory (archive an existing run to a side
folder, or overwrite it, before starting a fresh one), and then Start / Resume
into the live training dashboard. The package is split by concern:

- ``fields``     — the editable fields, their per-kind specs, and parse/commit
- ``runs``       — inspecting, archiving, and clearing runs in a checkpoint dir
- ``state``      — the configurator's live UI state + value-objects
- ``keys``       — cross-platform raw single-key input
- ``screen``     — the rich Layout + per-region renderers
- ``controller`` — the Live input loop + key dispatch (entry: ``run_configurator``)
"""

from wingspan.training.configure.controller import run_configurator

__all__ = ["run_configurator"]
