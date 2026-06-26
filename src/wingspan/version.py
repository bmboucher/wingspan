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

MODEL_VERSION = "0.9"
"""The current artifact-compatibility version (the only place it is defined).

0.9 compacts the state vector from 1155→1119 dims (default spec) by dropping
three redundant stripes and zeroing scored-round goal slots:

* ``misc_scalars`` 4→2 dims: dropped ``my_round_goal_pts`` and
  ``opp_round_goal_pts`` (the ``round_goals`` stripe captures standings fully).
* ``board_summary_me`` / ``board_summary_opp`` 18→6 dims each: kept only
  ``row_length`` + ``total_eggs`` per habitat (per-slot board state and card
  table make the rest redundant).
* ``hand_summary_me`` removed (10 dims): the distinct hand encoder now derives
  the 10-dim summary in-model from the hand multi-hot via
  ``set_summary_from_multihot``; the stripe is no longer in the state vector.
* ``round_goals``: values only — already-scored rounds are zeroed so past
  standings don't pollute future-decision features (width unchanged at 92 dims).

Total: state_dim 1155→1119 (−36). Choice vectors are unchanged. Pre-0.9
artifacts load and play through ``wingspan.compat.v0_8`` (``PolicyValueNetV08``).

0.8 changes the ``becomes_playable`` multi-hot stripe on **food-gain** choice
rows so that the egg-cost gate is dropped from the food-affordability check:
a hand bird is now flagged as "becomes playable" whenever gaining the offered
food makes its food cost payable AND it has any open slot, regardless of the
egg cost. The egg-gain path (``LAY_EGGS``, egg exchanges) is unchanged. This is
a code-carried FRESH change — no tensor widths change, but the value of
``becomes_playable`` bits on food-gain rows differs, so the change is
era-gated. Pre-0.8 artifacts load and play through ``wingspan.compat.v0_7``
(``PolicyValueNetV07``, which calls ``encode_choices`` with
``food_playable_ignores_eggs=False``). Pre-0.7 artifacts that also use the v0.6
card-feature shim (``PolicyValueNetV06``) now additionally carry the v0.7
eggs-included food encoding via a delegating ``encode_choices`` override.

0.7 adds an ``or_cost`` flag to the per-card attribute vector, growing
``CARD_FEATURE_DIM`` by 1 (224 → 225). The flag is 1.0 for birds that cost
exactly 1 food of any accepted type (OR cost) and 0.0 for birds that must pay
all listed food simultaneously (AND cost). State and choice vector widths are
unchanged; only the card encoder's first linear input grows. Pre-0.7 artifacts
load and play through the ``wingspan.compat.v0_6`` shim (``PolicyValueNetV06``
with frozen 224-wide card encoder and the pre-0.7 feature table).

0.6 adds hand-playability multi-hot stripes to both the state and choice
vectors, and adds a ``becomes_playable`` stripe to the choice spec. State grows
by 2 × 180 = 360 dims (795 → 1155); each choice grows by 180 dims. The default
``tray_set_embedding`` is flipped to ``False`` (REGIME — saved configs carry
their own value, so existing checkpoints are unaffected by the default change).
Pre-0.6 artifacts load and play through the ``wingspan.compat.v0_4`` shim
(``PolicyValueNetV04`` with frozen 795-dim state encoding and the pre-0.6 choice
encoding without the ``becomes_playable`` stripe).

0.5 unifies the per-run config files (``model_config.json``,
``setup_config.json``, ``process_<stamp>.json``) into a single
``run_config_<stamp>.json`` with a hierarchical Pydantic model. This is a
**config-container-only** bump — the encoding and network architecture are
identical to 0.4 (same ``state_dim`` / ``choice_dim`` / card features). No
``compat/v0_4.py`` encoding shim is needed: 0.4 artifacts already fall through
to live encoding paths unchanged. The only compat work is a config-format reader
dispatch in ``runmeta`` / ``setup_runmeta`` (≤0.4 run dirs still carry the
legacy trio; ≥0.5 dirs carry only ``run_config_<stamp>.json``).

0.4 refactored the round/cube encoding into a new leading ``turn_state`` stripe
and shrank ``misc_scalars`` from 26 dims to 4, growing the state vector by
5 dims (790 → 795): the 4-dim round one-hot and both 9-dim cube one-hots were
replaced by a 26-dim player-turn one-hot (which of the 26 personal turns is
being played, all-zeros during setup) plus a 1-bit is_first_player flag; the
opponent cube one-hot was dropped entirely (opponent cubes are determinable from
the player's own cubes plus the first-player flag). Pre-0.4 artifacts load and
play through the ``wingspan.compat.v0_3`` shim (``PolicyValueNetV03`` with
frozen 26-dim one-hot misc stripe).

0.3 replaced three raw scalars in ``_summary_misc_scalars`` with one-hot
vectors, growing the state vector by 19 dims (771 → 790): round_idx scalar
→ 4-dim one-hot (rounds 0–3), action_cubes_left scalar → 9-dim one-hot
(0–8 cubes) for each player.  Pre-0.3 artifacts load and play through the
``wingspan.compat.v0_2`` shim (``PolicyValueNetV02`` with frozen 7-scalar
misc stripe).

0.2 bundles two encoding changes: (a) setup input vector is now dynamic —
``kept_foods`` is omitted when ``split_setup_food=True``; ``kept_bonus`` +
``kept_bonus_value`` are replaced by ``bonus_cards`` (multi-hot of available
bonuses) + ``bonus_card_affinity`` (min/max qualifier counts, 2 dims) when
``split_setup_bonus=True``; the vector size varies (308 / 303 / 306 / 301
depending on config); pre-0.2 setup artifacts load as
``SetupEncoding(split_food=False, split_bonus=False)`` via Pydantic defaults —
no explicit shim needed; (b) card feature vector redesigned (CARD_FEATURE_DIM
229 → 224): bonus_categories pruned from 26 to 7 curated intrinsic-property
categories, a new caches_food flag, and a 13-dim power_exchange stripe; pre-0.2
main-net artifacts load and play through the ``wingspan.compat.v0_1`` shim.

0.1 reshaped the choice vector (landing-slot placement encoding, the single
``bird_id`` index column, the dedicated ``kept_multihot`` stripe); pre-0.1
artifacts load and play through the ``wingspan.compat.v0_0`` shim."""

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

    The first real shim landed with 0.1 and lives, as this docstring always
    promised, in the dedicated ``wingspan.compat`` package:
    ``compat.v0_0`` regenerates the pre-0.1 choice encoding for same-major
    artifacts (``compat.v0_0.encode_choices`` + ``PolicyValueNetV00``), routed
    by the loaders (``model.PolicyValueNet.from_model_config``,
    ``players.loaders.load_policy_net``) and by the era-aware
    expected-encoding keys in ``players.loaders``. Future MINOR encoding
    changes follow the same shape: a ``compat.v<X_Y>`` module keyed on
    ``parse_version(artifact_version)`` older-than-the-change.

    This function itself stays a validating no-op (this module is torch-free
    and must not import the shims); it remains so a future caller that only
    needs the validation keeps a stable seam.
    """
    parse_version(artifact_version)
