"""Pre-1.5 artifact compat shim: freeze the habitat-agnostic play-bird
``goal_delta`` pricing.

**What changed in v1.5 (encoding behavior — no shape change).** The
``PlayBirdChoice`` featurizer's ``goal_delta`` stripe became conditioned on the
row's committed landing habitat: a ``birds_<habitat>`` round goal now moves only
on the row that actually plays the bird into that habitat
(``scoring.goal_count_delta_for_bird``'s ``play_habitat`` parameter). Pre-1.5
rows priced the bird's *card* habitats, so a two-habitat bird advanced a habitat
goal — count and placement-VP delta alike — on **both** of its rows. Candidate
rows with no committed placement (hand / tray / setup keeps) kept the optimistic
any-card-habitat bound in both eras, so only play-bird rows differ.

**Shim strategy.** Every width, offset, and layout of era 1.4 equals live, so
this shim overrides only :meth:`encode_choices`: encode at live geometry, then
overwrite each ``PlayBirdChoice`` row's ``goal_delta`` stripe with the
habitat-agnostic pricing via
``choice_encode.refill_goal_delta_habitat_agnostic`` — a net trained against
the both-rows pricing keeps seeing it. This is the value-level analogue of the
column strips in ``compat.v1_3``: the same "regenerate the prior encoding after
live encoding" shape, applied to stripe *contents* instead of stripe columns.

**Routing.** ``PolicyValueNet.class_for_version`` routes era 1.4 here.
``compat.v1_3.PolicyValueNetV1_3`` **inherits** this net, so eras 1.0-1.3
freeze the old pricing too: their ``encode_choices`` ``super()``-chains through
this override (the refill happens at live column offsets, before their own
column strips shift anything).

**Fixture note.** A committed LFS checkpoint fixture is deferred (as for v1.0 /
v1.3): ``tests/test_compat_v1_4.py`` instead builds a v1.4-era net, saves it
with a v1.4 stamp, and round-trip-loads it through the production
``players.loaders.load_policy_net`` path.
"""

from __future__ import annotations

import typing

import numpy as np

from wingspan import decisions, state
from wingspan.encode import choice_encode
from wingspan.model import core


class PolicyValueNetV1_4(core.PolicyValueNet):
    """``PolicyValueNet`` with the pre-1.5 play-bird ``goal_delta`` pricing:
    every play-bird row prices a ``birds_<habitat>`` goal by the bird's card
    habitats instead of the row's landing habitat. Geometry is identical to
    live — only the stripe's values differ."""

    def encode_choices(
        self,
        decision: decisions.Decision[typing.Any],
        game_state: state.GameState,
    ) -> np.ndarray:
        """Encode at live v1.5 semantics, then overwrite each play-bird row's
        ``goal_delta`` stripe with the habitat-agnostic pre-1.5 pricing.

        The refill targets live column offsets, which is correct both here
        (era 1.4 geometry equals live) and under the ``v1_3`` / ``v1_0``
        subclasses (their column strips run after this override returns)."""
        full = super().encode_choices(decision, game_state)
        for row, choice in zip(full, decision.choices):
            if isinstance(choice, decisions.PlayBirdChoice):
                choice_encode.refill_goal_delta_habitat_agnostic(
                    row, decision.player_id, choice.bird, game_state
                )
        return full
