"""Shared pytest configuration for the whole suite.

* Makes ``src/`` importable without an install (every test module also inserts
  it for itself; this is the conftest-level copy so collection works no matter
  which module imports first).
* Caps torch's intra-op thread pool for test processes: per-decision inference
  batches are tiny, so a wide pool only adds sync overhead — and under
  ``pytest-xdist`` every worker process would otherwise spin up a full
  cores-wide pool. ``OMP_NUM_THREADS`` is set *before* any ``torch`` import so
  the cap applies from the first kernel without importing torch here (most
  engine/parser tests never need it).
* Restores the cap after every test: ``torch.set_num_threads`` is
  process-global, and both the mp_collect parity tests (which pin 1 for
  argmax-tie determinism) and ``TrainingLoop.__init__`` mutate it.
"""

from __future__ import annotations

import os
import sys
import typing

import pytest

# Mirrors ``wingspan.training.loop._CPU_INTRAOP_THREADS`` — the measured sweet
# spot for batch-of-one CPU inference (see the comment there).
_TEST_INTRAOP_THREADS = 2

# Test files dominated by multi-second tests (process-pool spawns, training
# iterations). Front-loaded so pytest-xdist workers start them immediately —
# in collection (alphabetical) order they land near the end of the run and
# anchor a long single-worker tail.
_HEAVY_TEST_FILES = frozenset(
    {
        "test_training_dashboard.py",
        "test_mp_collect.py",
        "test_model_and_self_play.py",
        "test_compat_v1_0.py",
        "test_setup_train_cpu.py",
        "test_setup_arch_key_restart.py",
        "test_setup_collect.py",
        "test_setup_feature_off.py",
    }
)

# Must happen before any test module imports torch: the env var is read once
# at torch import time.  The pythonpath plugin (pyproject.toml
# [tool.pytest.ini_options] pythonpath=["src"]) handles `wingspan` importability.
os.environ.setdefault("OMP_NUM_THREADS", str(_TEST_INTRAOP_THREADS))


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Schedule the heavy test files first (stable partition — order within
    each group is preserved) so parallel workers chew on the multi-second
    tests while the fast bulk fills the remaining slots."""
    items.sort(key=lambda item: 0 if item.path.name in _HEAVY_TEST_FILES else 1)


@pytest.fixture(autouse=True)
def restore_torch_thread_cap() -> typing.Iterator[None]:
    """Undo any per-test ``torch.set_num_threads`` mutation (process-global).

    Public (no leading underscore) so strict pyright sees an exported name —
    pytest discovers autouse fixtures via the decorator, never by reference."""
    yield
    if "torch" in sys.modules:
        # Imported lazily so torch-free test runs never pay the torch import.
        import torch

        torch.set_num_threads(_TEST_INTRAOP_THREADS)
