"""The setup-model training sample and its on-disk store.

A :class:`SetupSample` is one recorded ``(setup features, realized margin)`` pair:
the feature vector a seat's chosen setup encoded to, paired with the score margin
that seat ended the game with. These are the regression targets the setup model
is fit on — its scalar output is trained (MSE) to predict ``margin``.

:class:`SetupDataStore` appends them to a JSONL file under the run's checkpoint
directory (``setup_data.jsonl``), mirroring the append-only ``games.jsonl``
history: it is written during the record window (1000–2000 iters) and read back
once for the one-time offline fit. The features serialize as a plain ``list[float]``
so the file stays human-inspectable and torch-free.
"""

from __future__ import annotations

import pathlib
import typing

import numpy as np
import pydantic


class SetupSample(pydantic.BaseModel):
    """One ``(setup features, realized margin)`` training sample.

    ``features`` is the :func:`wingspan.setup_model.encode.encode_setup_candidate`
    vector for the seat's chosen setup; ``margin`` is the seat's end-of-game
    ``own_total - opponent_total`` (the contextual-bandit reward), left
    unnormalized here so the learner can scale it by ``score_norm`` consistently
    with the in-game return.

    The two optional fields below are populated only in the actor-critic
    (``setup_use_actor_critic=True``) MODEL_DRIVEN phase — they carry the data
    needed to compute a REINFORCE gradient over all candidates at training time:

    * ``chosen_idx`` — which row in ``all_candidates`` was selected.
    * ``all_candidates`` — the ``(K, feature_dim)`` matrix of every candidate's
      encoded features (K = 504 or 252 with split-bonus). Compressed to float16
      before IPC; never persisted to the JSONL store.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    features: np.ndarray
    margin: float
    iteration: int
    chosen_idx: int | None = None
    all_candidates: np.ndarray | None = None


class _SetupRow(pydantic.BaseModel):
    """The JSONL serialization shape of a :class:`SetupSample` (features as a
    plain float list so the log stays torch-free and inspectable)."""

    features: list[float]
    margin: float
    iteration: int


class SetupDataStore:
    """Append-only JSONL store of :class:`SetupSample` rows for the offline fit."""

    def __init__(self, path: pathlib.Path):
        self._path = path

    def append(self, samples: typing.Sequence[SetupSample]) -> None:
        """Append a batch of samples as one buffered write (no-op if empty)."""
        if not samples:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        rows = "".join(
            _SetupRow(
                features=[float(value) for value in sample.features.tolist()],
                margin=sample.margin,
                iteration=sample.iteration,
            ).model_dump_json()
            + "\n"
            for sample in samples
        )
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(rows)

    def iter_samples(
        self, expected_dim: int | None = None
    ) -> typing.Iterator[SetupSample]:
        """Yield every recorded sample (features rehydrated to a float32 array).

        Rows whose feature length differs from ``expected_dim`` (defaults to
        the legacy ``encode.SETUP_FEATURE_DIM``) are skipped: they were
        recorded under an older encoding layout and cannot be stacked with (or
        scored by) the current network, so a store that survived a layout
        change degrades to its still-valid rows instead of crashing the offline
        fit."""
        # Imported lazily to keep the store importable without the encoder chain
        # (mirrors how the package splits torch-free pieces from the network).
        from wingspan.setup_model import encode

        dim = expected_dim if expected_dim is not None else encode.SETUP_FEATURE_DIM
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = _SetupRow.model_validate_json(line)
                if len(row.features) != dim:
                    continue
                yield SetupSample(
                    features=np.array(row.features, dtype=np.float32),
                    margin=row.margin,
                    iteration=row.iteration,
                )

    def count(self) -> int:
        """Number of recorded samples on disk (0 if the file is absent)."""
        if not self._path.exists():
            return 0
        with open(self._path, "r", encoding="utf-8") as handle:
            return sum(1 for line in handle if line.strip())

    def clear(self) -> None:
        """Truncate the store (called when a fresh run resets its history)."""
        if self._path.exists():
            self._path.write_text("", encoding="utf-8")
