"""Tests for the v1.3 → v1.4 compat shim: the ``resets_feeder`` choice stripe.

v1.4 appended a 1-dim ``resets_feeder`` stripe as the last base choice stripe
(after ``becomes_unplayable``), marking a ``combine_gain_food`` ``FoodSubsetChoice``
whose selection rerolls the birdfeeder. v1.1–1.3 choice vectors lack it, so
``PolicyValueNetV1_3`` strips it after live encoding and shifts only ``kept_multihot``
(``bird_id`` / ``becomes_playable`` / ``becomes_unplayable`` all precede it).

Following the v1.0 precedent, the round-trip is exercised via freshly-built weights
rather than a saved LFS checkpoint.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# (tests access _choice_embed_offsets to pin the ChoiceEmbedOffsets seam)
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from wingspan import architecture, decisions, encode, engine, version
from wingspan.compat import v1_3 as compat_v1_3
from wingspan.model import core


def _small_arch() -> architecture.ModelArchitecture:
    return architecture.ModelArchitecture(
        trunk_layers=(8, 8),
        choice_layers=(8, 8),
        head_layers=(),
        value_layers=(),
        card_embed_dim=4,
    )


def _main_action_decision() -> decisions.MainActionDecision:
    return decisions.MainActionDecision(
        player_id=0,
        prompt="action",
        choices=[
            decisions.MainActionChoice(
                label="food", action=decisions.MainAction.GAIN_FOOD
            ),
            decisions.MainActionChoice(
                label="eggs", action=decisions.MainAction.LAY_EGGS
            ),
        ],
    )


class TestClassForVersionRouting:
    """``class_for_version`` routes v1.1–1.3 to the v1_3 shim, v1.4 to live."""

    def test_v1_1_through_v1_3_return_v1_3_shim(self) -> None:
        for minor in ("1.1", "1.2", "1.3"):
            assert (
                core.PolicyValueNet.class_for_version(minor)
                is compat_v1_3.PolicyValueNetV1_3
            )

    def test_current_version_returns_live_class(self) -> None:
        assert (
            core.PolicyValueNet.class_for_version(version.MODEL_VERSION)
            is core.PolicyValueNet
        )


class TestV1_3EncodingCompat:
    """The v1.3 shim strips resets_feeder from choice encodings and shifts only
    kept_multihot; it keeps becomes_unplayable (unlike the v1.0 shim)."""

    def _make_net(
        self, arch: architecture.ModelArchitecture
    ) -> compat_v1_3.PolicyValueNetV1_3:
        return compat_v1_3.PolicyValueNetV1_3(arch=arch)

    def test_encode_choices_narrower_than_live_by_resets_feeder(self) -> None:
        """The shim output is exactly CHOICE_RESETS_FEEDER_DIM columns narrower."""
        eng, *_ = engine.Engine.create(seed=100)
        shim_net = self._make_net(_small_arch())
        decision = _main_action_decision()
        live_cols = encode.encode_choices(decision, eng.state).shape[1]
        shim_cols = shim_net.encode_choices(decision, eng.state).shape[1]
        assert live_cols - shim_cols == encode.CHOICE_RESETS_FEEDER_DIM

    def test_encode_choices_matches_live_without_resets_feeder(self) -> None:
        """encode_choices on the shim matches live output with resets_feeder removed."""
        eng, *_ = engine.Engine.create(seed=100)
        shim_net = self._make_net(_small_arch())
        decision = _main_action_decision()

        live_full = encode.encode_choices(decision, eng.state)
        shim_out = shim_net.encode_choices(decision, eng.state)

        start = encode.CHOICE_RESETS_FEEDER_OFFSET
        end = start + encode.CHOICE_RESETS_FEEDER_DIM
        live_stripped = np.delete(live_full, slice(start, end), axis=1)

        assert shim_out.shape == live_stripped.shape
        assert np.array_equal(shim_out, live_stripped)

    def test_becomes_unplayable_kept_and_unshifted(self) -> None:
        """The v1.3 shim keeps becomes_unplayable at the live offset (it precedes
        the new stripe)."""
        arch = _small_arch()
        live_net = core.PolicyValueNet(arch=arch)
        shim_net = self._make_net(arch)
        live = live_net._choice_embed_offsets()
        shim = shim_net._choice_embed_offsets()
        assert shim.becomes_unplayable is not None
        assert shim.becomes_unplayable == live.becomes_unplayable
        assert shim.becomes_playable == live.becomes_playable
        assert shim.bird_id == live.bird_id

    def test_kept_multihot_offset_shifted_left(self) -> None:
        """With include_setup, kept_multihot shifts left by exactly the stripe width."""
        arch = _small_arch()
        spec = encode.EncodingSpec(include_setup=True)
        live_net = core.PolicyValueNet(spec=spec, arch=arch)
        shim_net = compat_v1_3.PolicyValueNetV1_3(spec=spec, arch=arch)

        live = live_net._choice_embed_offsets()
        shim = shim_net._choice_embed_offsets()
        assert live.kept_multihot is not None
        assert shim.kept_multihot is not None
        assert (
            live.kept_multihot - shim.kept_multihot == encode.CHOICE_RESETS_FEEDER_DIM
        )

    def test_forward_pass_runs(self) -> None:
        """A forward pass through the shim's tensors does not raise."""
        eng, *_ = engine.Engine.create(seed=101)
        shim_net = self._make_net(_small_arch())
        decision = _main_action_decision()

        state_vec = shim_net.encode_state(eng.state, decision)
        choice_feats = shim_net.encode_choices(decision, eng.state)
        family_idx = decisions.family_index_for(type(decision))

        state_t = torch.from_numpy(state_vec).unsqueeze(0)
        choices_t = torch.from_numpy(choice_feats).unsqueeze(0)
        mask_t = torch.ones(1, choices_t.shape[1])
        family_t = torch.tensor([family_idx], dtype=torch.long)

        logits, value = shim_net(state_t, choices_t, mask_t, family_t)
        assert logits.shape == (1, 2)
        assert value.shape == (1,)
