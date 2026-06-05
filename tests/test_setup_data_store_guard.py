"""The setup data store's stale-row guard.

A store that survived an encoding-layout change holds rows of mixed widths;
``iter_samples`` must yield only rows matching the *current*
``SETUP_FEATURE_DIM`` so the offline fit never stacks incompatible vectors.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan import setup_model  # noqa: E402

_OLD_FEATURE_DIM = 477  # the pre-shared-embedder layout's width


def _write_row(path: pathlib.Path, width: int, margin: float) -> None:
    row = {"features": [0.0] * width, "margin": margin, "iteration": 1}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def test_mixed_width_store_yields_only_current_width_rows(tmp_path: pathlib.Path):
    path = tmp_path / "setup_data.jsonl"
    _write_row(path, _OLD_FEATURE_DIM, margin=1.0)
    _write_row(path, setup_model.SETUP_FEATURE_DIM, margin=2.0)
    _write_row(path, _OLD_FEATURE_DIM, margin=3.0)
    _write_row(path, setup_model.SETUP_FEATURE_DIM, margin=4.0)

    store = setup_model.SetupDataStore(path)
    samples = list(store.iter_samples())
    assert [sample.margin for sample in samples] == [2.0, 4.0]
    for sample in samples:
        assert sample.features.shape == (setup_model.SETUP_FEATURE_DIM,)
        assert sample.features.dtype == np.float32


def test_current_width_rows_round_trip(tmp_path: pathlib.Path):
    path = tmp_path / "setup_data.jsonl"
    store = setup_model.SetupDataStore(path)
    store.append(
        [
            setup_model.SetupSample(
                features=np.zeros(setup_model.SETUP_FEATURE_DIM, dtype=np.float32),
                margin=5.0,
                iteration=7,
            )
        ]
    )
    samples = list(store.iter_samples())
    assert len(samples) == 1
    assert samples[0].margin == 5.0 and samples[0].iteration == 7
