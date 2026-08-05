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

MODEL_VERSION = "1.6"
"""The current artifact-compatibility version (the only place it is defined).

1.6 is a **shape** MINOR FRESH bump that lands the "played and
egg-populated" goal pricing on the choice side. A new 8-dim
``goal_delta_ignoring_eggs`` choice stripe is appended at the tail of the base
choice-feature vector (immediately after ``resets_feeder``, the prior last
base stripe): per round goal, a ``(count_delta, vp_delta)`` pair pricing the
hypothesis that this row's bird is eventually played (a slot must be open in
one of its card habitats) and egg-populated to whatever level best advances
that goal (``scoring.goal_vp_delta_for_bird_with_eggs``, built on
``scoring.goal_count_delta_for_bird_with_eggs`` /
``scoring.goal_affinity_for_kept``). A bowl-nest bird counts toward both an
``eggs_bowl`` goal (its ``egg_limit``) and a ``bowl_birds_with_eggs`` goal
(1); star nests are wild (``cards.nest_matches``). The existing ``goal_delta``
stripe keeps its exact v1.5 play-instant semantics untouched — the new
stripe is filled by a separate featurizer, at the three call sites that
already fill ``goal_delta`` (``BirdChoice``, ``PlayBirdChoice``, the tray
``DrawSourceChoice`` row). Choice width grows by 8 (N=2 base: 509 → 517;
include_setup: 693 → 701; N=3 base: 512 → 520).

This era also carries a **value-level setup-encoding change**: the setup
``goal_affinity`` stripe gains egg-driven categories, so a setup keep is
priced with the same played-and-egg-populated optimism the choice stripe
uses in-game (see ``docs/VERSIONING.md`` for the setup-side seam mechanics
and ``docs/BONUSES.md`` for the affected goal categories). Both halves share one ``MODEL_VERSION`` bump because they are
two views of the same scoring upgrade landing together.

Era ≤1.5 artifacts predate the stripe. ``wingspan.compat.v1_5.PolicyValueNetV1_5``
strips it after live encoding (the geometry-narrowing analogue of the v1_3 /
v1_0 column strips): ``encoding_dims_for_era`` returns a ``choice_dim`` 8 less
for every pre-1.6 same-MAJOR era, and every existing shim
(``PolicyValueNetV1_4``, ``PolicyValueNetV1_3``, ``PolicyValueNetV1_0``) now
chains through ``PolicyValueNetV1_5`` — ``compat.v1_4.PolicyValueNetV1_4``
re-chains to subclass it directly, so its own ``goal_delta`` habitat-agnostic
refill runs *after* the parent's tail-strip, at unaffected offsets (all
< 509).

1.5 was a **behavior-only** MINOR FRESH bump — no tensor shape changes; the first
value-level (as opposed to width-level) encoding era since the v1.1
trunk-final-activation fix. The ``PlayBirdChoice`` featurizer's ``goal_delta``
stripe is now **conditioned on the row's landing habitat**: a ``birds_<habitat>``
round goal moves only on the row that actually plays the bird into that habitat.
Pre-1.5 rows priced the bird's *card* habitats (``scoring.goal_count_delta_for_bird``
with no ``play_habitat``), so a two-habitat bird advanced a habitat goal on both
of its rows — e.g. a Peregrine Falcon's grassland row claimed the "[bird] in
[wetland]" goal's count and VP delta. Candidate rows with no committed placement
(hand / tray / setup keeps) keep the optimistic any-card-habitat bound.

Era 1.4 artifacts route to ``wingspan.compat.v1_4.PolicyValueNetV1_4``, which
re-fills each play-bird row's ``goal_delta`` with the habitat-agnostic pricing
after live encoding (``choice_encode.refill_goal_delta_habitat_agnostic``). At
the time of the 1.5 bump, dims were unchanged for era 1.4 (the refill ran at
live width); since the 1.6 bump ``PolicyValueNetV1_4`` subclasses
``PolicyValueNetV1_5`` (the goal_delta_ignoring_eggs tail-strip), so era 1.4's
choice geometry is now 8 narrower than live too — the refill still runs at
unaffected offsets (all < 509), inside the ``super().encode_choices`` chain,
*after* the parent's tail-strip. ``architecture_key`` leads with the era, so a
1.4 run still resumes era-pinned as the shim class. ``compat.v1_3.PolicyValueNetV1_3``
inherits ``PolicyValueNetV1_4``, so every pre-1.4 era freezes the old pricing
too, on top of its own stripe strips.

1.4 is a **main-net encoding** MINOR FRESH bump that lands two independent
encoding changes together (both developed in parallel, folded into one era):

1. *State* — two 5-wide pass-through stripes are appended to the continuous state
   prefix (immediately after ``food_opp``): ``hand_food_unlock_me`` and
   ``tray_food_unlock_me`` — per food, the smallest count that would newly unlock a
   hand / tray bird (see ``encode.playability.min_food_to_unlock``). This is the
   first same-MAJOR change to alter the **state** width, which grows by 10.

2. *Choice* — a 1-dim ``resets_feeder`` stripe is appended as the last *base*
   choice-feature stripe (immediately after ``becomes_unplayable``, before the
   conditional setup stripes). The bit is set on a ``combine_gain_food``
   ``FoodSubsetChoice`` whose selection rerolls the birdfeeder — a partial take that
   commits to a reset, or a full take that empties the feeder — so the model can tell
   a smaller-but-rerolls gain apart from a plain smaller gain (the ``gain_food`` count
   vector alone cannot). The choice vector widens by 1.

``architecture_key`` detects both width changes (via ``state_dim`` / ``choice_dim``)
and refuses old checkpoints cleanly. The **setup model is unchanged**, so setup
artifacts stay loadable and there is no setup-side shim.

Pre-1.4 artifacts are routed to ``wingspan.compat.v1_3.PolicyValueNetV1_3``, which
strips **both** the two state stripes and the ``resets_feeder`` choice column from
live-encoded vectors and freezes the pre-1.4 state- and choice-embed offsets.
``compat.encoding_dims_for_era`` returns a ``state_dim`` (−10) and a ``choice_dim``
(−1) narrower for every pre-1.4 same-MAJOR era (the state narrowing is its first
state-dim branch). v1.0 artifacts route to ``compat.v1_0.PolicyValueNetV1_0``, which
inherits ``PolicyValueNetV1_3`` (so it strips the state stripes and ``resets_feeder``
too) and additionally strips the ``becomes_unplayable`` choice stripe and restores
the old trunk-final activation fallback.

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
