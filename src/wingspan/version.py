"""The artifact-compatibility version and its load-time enforcement.

Every persisted training artifact (the ``model_config.json`` /
``setup_config.json`` sidecars and the ``*.pt`` checkpoint payloads) is stamped
with :data:`MODEL_VERSION`, a ``MAJOR.MINOR`` string that is bumped whenever the
encoding or network architecture changes shape. The compatibility contract:

* **Same MAJOR, artifact MINOR <= current MINOR** — the artifact must load and
  play games (inference / eval / tournament). Older-minor artifacts are kept
  loadable via version-specific shims (see :func:`adapt_encoding_for_version`),
  never via per-change config flags.
* **Different MAJOR, or artifact MINOR > current MINOR** — the loaders refuse
  with :class:`IncompatibleArtifactError`. A MAJOR bump is the deliberate
  escape hatch that deletes the accumulated shims and old test fixtures.

Training *resume* honors the same eras via pinning: a run carries its
``TrainConfig.encoding_version`` and keeps training at that era's frozen
geometry under newer same-MAJOR code (``loop_resume.adopt_checkpoint_era``,
``docs/VERSIONING.md``), stamping every artifact it writes with its own era.
The resume gate still refuses any genuine ``architecture_key`` mismatch and
starts fresh — and every *fresh* launch is re-keyed at the live
:data:`MODEL_VERSION`, so a new run never trains at a stale era.

This is distinct from the *package release* version
(``wingspan.__version__``): that tracks the codebase, this tracks the on-disk
artifact format. Kept torch-free and dependency-free (stdlib + pydantic only)
so every loader module can import it without cycles.
"""

from __future__ import annotations

import re

import pydantic

MODEL_VERSION = "1.4"
"""The current artifact-compatibility version (the only place it is defined).

1.4 is a **main-net** MINOR FRESH bump. It appends a 1-dim ``resets_feeder`` stripe
as the last *base* choice-feature stripe (immediately after ``becomes_unplayable``,
before the conditional setup stripes). The bit is set on a ``combine_gain_food``
``FoodSubsetChoice`` whose selection rerolls the birdfeeder — a partial take that
commits to a reset, or a full take that empties the feeder — so the model can tell a
smaller-but-rerolls gain apart from a plain smaller gain (the ``gain_food`` count
vector alone cannot). The main net's choice vector widens by 1, which
``architecture_key`` detects via ``choice_dim`` and refuses old checkpoints cleanly.

The **setup model is unchanged** (its choice encoding is independent of the main
choice width), so setup artifacts stay loadable and there is no setup-side shim.
v1.0–1.3 main-net artifacts are routed to compat shims: ``compat.v1_3`` strips the
``resets_feeder`` column (keeps ``becomes_unplayable``), and ``compat.v1_0`` inherits
it to compose both strips (v1.0 lacks both stripes). ``encoding_dims_for_era`` returns
a ``choice_dim`` one narrower for every era with minor ≤ 3.

1.3 is another **setup-artifact-only** MINOR FRESH bump. The separate setup model
is restructured into a two-tower actor-critic mirroring the in-game
``model.PolicyValueNet``: a shared **state trunk** encodes the action-independent
stripes into a ``state_enc`` that feeds both heads, and a separate **choice trunk**
encodes the action stripes into a ``choice_enc``. The value head reads ``state_enc``
only (still a true ``V(s)``); the policy head reads ``cat(state_enc, choice_enc)``
instead of the former fused per-candidate vector. ``SetupArchitecture`` gains
``trunk_layers`` / ``choice_layers`` / ``head_layers`` / ``value_layers`` (mirroring
``ModelArchitecture``'s field names), defaulting to ``(128,)`` state and ``(128,)``
choice trunks. The submodule set and the policy head's first ``Linear`` change, so
old ``setup.pt`` weights no longer fit.

As with 1.2 the **main net's encoding and topology are unchanged**, so there is no
``compat.v1_2`` shim and no ``encoding_dims_for_era`` entry — 1.3 main-net dims equal
1.2 equal live. Setup checkpoints are discarded, not migrated: a resumed run restarts
its setup model fresh via ``loop_setup.maybe_resume_setup``'s shape-mismatch path, and
``players.loaders.load_setup_net`` refuses an incompatible ``setup.pt`` with a clear
retrain message.

1.2 is a **setup-artifact-only** MINOR FRESH bump. The separate setup model's
value head becomes a state-only critic ``V(s)`` — reading only the
action-independent deal stripes (tray, birdfeeder, round goals, bonus-on-offer)
— instead of the former per-candidate ``Q(s, a)`` over the fused state ⊕ action
vector. Its first ``Linear`` is therefore narrower (≈304 vs ≈568 by default) and
old ``setup.pt`` weights no longer fit. The fix removes the action-dependent
baseline that made the setup advantage self-cancel, and reconciles the setup
target with the in-game return at ``t=0`` (``training.returns``; a
shape-preserving REGIME change).

The **main net's encoding and topology are unchanged**, so there is no
``compat.v1_1`` shim and no ``encoding_dims_for_era`` entry — 1.2 main-net dims
equal 1.1 equal live. Only the setup model is affected, and setup checkpoints are
discarded, not migrated (a Q-trained fused value head has no faithful ``V(s)``
reconstruction): a resumed run restarts its setup model fresh via
``loop_setup.maybe_resume_setup``'s shape-mismatch path, and
``players.loaders.load_setup_net`` refuses an incompatible ``setup.pt`` with a
clear retrain message. A run pinned to an earlier same-MAJOR era keeps training
its (unchanged) main net at that era while always building the live setup net.

1.1 is the first MINOR FRESH bump on top of the 1.0 clean-break baseline. It
introduces three changes:

1. *Architecture* — drops the trunk's special ``between_activation`` fallback:
   ``trunk_final_activation`` now inherits ``final_activation`` like every other
   block (see ``docs/VERSIONING.md``).

2. *Encoding* — adds the ``becomes_unplayable`` 180-dim multi-hot stripe to the
   base choice feature vector (immediately after ``becomes_playable``).

3. *Setup encoding* — the setup net's kept-card and optional playable-card sets
   are now embedded via the same pooling path as the main net's hand stripe
   (``hand_model.pool_card_set`` with the same ``hand_pooling`` mode), yielding
   ``pooled_hand_width = 2N+1 = 129`` for the default CONCAT_MAX_SUM mode instead
   of the old ``hand_embed_width = N = 64``.  The tray-set embedding that was
   hardcoded in the setup net is dropped: the tray now contributes only
   ``TRAY_SIZE × N = 3N = 192`` dims (slot card-table rows), matching the main
   net's state tray.  ``SetupEncoding.include_playable_kept_cards`` now defaults
   to ``True``, so the food-agnostic playable-kept-card set embedding is enabled
   in all new setup nets.  The resulting default ``setup_readout_input_dim``
   changes from 445 to 575 (= 125 passthrough + 2×129 sets + 3×64 tray).

v1.0 artifacts are routed to ``wingspan.compat.v1_0.PolicyValueNetV1_0``, which
restores the old trunk-final fallback and strips the ``becomes_unplayable`` stripe
from choice encodings. ``compat.encoding_dims_for_era`` returns a narrower
``choice_dim`` for v1.0.

1.0 was the MAJOR bump that dropped the accumulated pre-1.0 compat shims
(``wingspan.compat.v0_0`` … ``v0_8``), deleted the old fixture sets, and removed
the dead code paths those shims existed to support. No 0.x artifact loads under
1.x code. The per-version 0.1–0.8 changelog is recoverable from git history and
summarized in ``docs/VERSIONING.md``."""

PRE_VERSIONING_VERSION = "0.0"
"""The version assigned to artifacts that predate the ``version`` field.

Files lacking the field were by definition written before versioning existed,
so they read as the original era — this stays pinned at ``"0.0"`` forever while
:data:`MODEL_VERSION` advances."""

_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)$")


class Version(pydantic.BaseModel):
    """A parsed ``MAJOR.MINOR`` artifact version."""

    model_config = pydantic.ConfigDict(frozen=True)

    major: int
    minor: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}"


class IncompatibleArtifactError(Exception):
    """Raised when a persisted artifact's version cannot be loaded by this code."""


def parse_version(raw: str) -> Version:
    """Parse a ``MAJOR.MINOR`` string into a :class:`Version`.

    Raises ``ValueError`` for anything that is not exactly two dot-separated
    integers (``"1"``, ``"1.2.3"``, ``"abc"``)."""
    match = _VERSION_PATTERN.match(raw)
    if match is None:
        raise ValueError(
            f"Invalid artifact version {raw!r}: expected 'MAJOR.MINOR' "
            "(two dot-separated integers, e.g. '0.0')."
        )
    return Version(major=int(match.group(1)), minor=int(match.group(2)))


def check_artifact_compatible(artifact_version: str, *, what: str) -> None:
    """Refuse an artifact this code does not guarantee to load.

    ``what`` is a short label naming the artifact (e.g. ``"model_config.json at
    <dir>"``) folded into the error message. Passes silently when the artifact
    shares the current MAJOR and its MINOR is at most the current MINOR; raises
    :class:`IncompatibleArtifactError` otherwise."""
    artifact = parse_version(artifact_version)
    current = parse_version(MODEL_VERSION)
    if artifact.major != current.major:
        raise IncompatibleArtifactError(
            f"{what} has artifact version {artifact} but this code is version "
            f"{current}: different MAJOR versions are not loadable. Use a "
            f"codebase from the {artifact.major}.x line, or retrain."
        )
    if artifact.minor > current.minor:
        raise IncompatibleArtifactError(
            f"{what} has artifact version {artifact} but this code is version "
            f"{current}: the artifact is newer than this code understands. "
            "Update the codebase to load it."
        )


def adapt_encoding_for_version(artifact_version: str) -> None:
    """The seam where version-specific encoding shims are documented.

    The ``wingspan.compat`` package is currently empty — the pre-1.0 shims were
    dropped at the 1.0 MAJOR bump. The next MINOR FRESH change re-introduces one:
    a ``compat.v1_<N>`` module keyed on ``parse_version(artifact_version)``
    older-than-the-change, regenerating the prior shape for same-MAJOR artifacts,
    routed by the loaders (``model.PolicyValueNet.from_model_config`` →
    ``class_for_version``, ``players.loaders``).

    This function itself stays a validating no-op (this module is torch-free
    and must not import the shims); it remains so a future caller that only
    needs the validation keeps a stable seam.
    """
    parse_version(artifact_version)
