"""Tests for the v1.0 → v1.1 compat shim: trunk-final-activation and encoding.

v1.0 behavior:
  - ``trunk_final_activation=None`` fell back to ``between_activation`` (relu).
  - ``becomes_unplayable`` stripe was not yet in the choice feature vector.

v1.1 behavior:
  - ``trunk_final_activation=None`` falls back to ``final_activation`` (none).
  - ``becomes_unplayable`` 180-dim stripe is appended immediately after
    ``becomes_playable`` in the base choice vector.

The shim class ``PolicyValueNetV1_0`` restores both v1.0 behaviors so that
rehydrated v1.0 checkpoints compute identically to what they did at training time.
"""

from __future__ import annotations

# pyright: reportPrivateUsage=false
# (tests access _choice_embed_offsets to pin the ChoiceEmbedOffsets seam)
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from wingspan import architecture, decisions, encode, engine, version
from wingspan.compat import v1_0 as compat_v1_0
from wingspan.model import core


# A small arch that makes the trunk behavior observable. dropout=0 so the last
# module is the activation itself (not Dropout), making the assertion simple.
def _make_arch(
    final_activation: architecture.ActivationName,
) -> architecture.ModelArchitecture:
    return architecture.ModelArchitecture(
        trunk_layers=(64, 32),
        trunk_final_activation=None,  # must be None to exercise the fallback
        between_activation=architecture.ActivationName.RELU,
        final_activation=final_activation,
        trunk_dropout=0.0,
        dropout=0.0,
    )


def _last_trunk_module(net: core.PolicyValueNet) -> nn.Module:
    """The final module in the trunk Sequential."""
    modules = list(net.state_trunk)
    return modules[-1]


class TestClassForVersionRouting:
    """``class_for_version`` routes correctly for both eras."""

    def test_v1_0_returns_shim_class(self) -> None:
        cls = core.PolicyValueNet.class_for_version("1.0")
        assert cls is compat_v1_0.PolicyValueNetV1_0

    def test_v1_1_returns_live_class(self) -> None:
        cls = core.PolicyValueNet.class_for_version("1.1")
        assert cls is core.PolicyValueNet

    def test_current_version_returns_live_class(self) -> None:
        cls = core.PolicyValueNet.class_for_version(version.MODEL_VERSION)
        assert cls is core.PolicyValueNet


class TestTrunkFinalActivationBehavior:
    """The shim restores the v1.0 trunk-final-activation fallback.

    With ``trunk_final_activation=None``, ``between_activation=relu``,
    ``final_activation=none``:
    - v1.1 trunk ends with a ``Linear`` (no activation — inherited ``none``)
    - v1.0 shim trunk ends with a ``ReLU`` (inherited ``relu`` from between)
    """

    def test_v1_1_trunk_has_no_final_relu(self) -> None:
        arch = _make_arch(architecture.ActivationName.NONE)
        net = core.PolicyValueNet(arch=arch)
        assert isinstance(_last_trunk_module(net), nn.Linear)

    def test_v1_0_shim_trunk_has_final_relu(self) -> None:
        arch = _make_arch(architecture.ActivationName.NONE)
        net = compat_v1_0.PolicyValueNetV1_0(arch=arch)
        assert isinstance(_last_trunk_module(net), nn.ReLU)

    def test_explicit_trunk_final_activation_same_in_both(self) -> None:
        """When ``trunk_final_activation`` is explicitly set, shim and live agree."""
        arch = architecture.ModelArchitecture(
            trunk_layers=(64, 32),
            trunk_final_activation=architecture.ActivationName.RELU,
            between_activation=architecture.ActivationName.RELU,
            final_activation=architecture.ActivationName.NONE,
            trunk_dropout=0.0,
            dropout=0.0,
        )
        live_last = _last_trunk_module(core.PolicyValueNet(arch=arch))
        shim_last = _last_trunk_module(compat_v1_0.PolicyValueNetV1_0(arch=arch))
        assert type(live_last) is type(shim_last)
        assert isinstance(live_last, nn.ReLU)


# ---------------------------------------------------------------------------
# v1.0 encoding compat: becomes_unplayable stripe handling


class TestV1_0EncodingCompat:
    """The v1.0 shim strips becomes_unplayable from choice encodings and adjusts
    ChoiceEmbedOffsets so the model never sees the new stripe."""

    def _make_net(
        self, arch: architecture.ModelArchitecture
    ) -> compat_v1_0.PolicyValueNetV1_0:
        return compat_v1_0.PolicyValueNetV1_0(arch=arch)

    def _small_arch(self) -> architecture.ModelArchitecture:
        return architecture.ModelArchitecture(
            trunk_layers=(8, 8),
            choice_layers=(8, 8),
            head_layers=(),
            value_layers=(),
            card_embed_dim=4,
        )

    def test_v1_0_encode_choices_narrower_than_live_by_unplayable_stripe(self) -> None:
        """The v1.0 shim's encode_choices output is exactly CHOICE_BECOMES_UNPLAYABLE_DIM
        columns narrower than the live encoder output."""
        eng, *_ = engine.Engine.create(seed=100)
        arch = self._small_arch()
        shim_net = self._make_net(arch)

        decision = decisions.MainActionDecision(
            player_id=0,
            prompt="action",
            choices=[
                decisions.MainActionChoice(
                    label="food", action=decisions.MainAction.GAIN_FOOD
                ),
            ],
        )
        live_cols = encode.encode_choices(decision, eng.state).shape[1]
        shim_cols = shim_net.encode_choices(decision, eng.state).shape[1]
        assert live_cols - shim_cols == encode.CHOICE_BECOMES_UNPLAYABLE_DIM

    def test_v1_0_encode_choices_width_matches_live_without_unplayable_block(
        self,
    ) -> None:
        """encode_choices on the shim matches live output with unplayable cols removed."""
        eng, *_ = engine.Engine.create(seed=100)
        arch = self._small_arch()
        shim_net = self._make_net(arch)

        decision = decisions.MainActionDecision(
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
        live_full = encode.encode_choices(decision, eng.state)
        shim_out = shim_net.encode_choices(decision, eng.state)

        start = encode.CHOICE_BECOMES_UNPLAYABLE_OFFSET
        end = start + encode.CHOICE_BECOMES_UNPLAYABLE_DIM
        live_stripped = np.delete(live_full, slice(start, end), axis=1)

        assert shim_out.shape == live_stripped.shape
        assert np.array_equal(shim_out, live_stripped)

    def test_v1_0_choice_embed_offsets_has_becomes_unplayable_none(self) -> None:
        """The shim's _choice_embed_offsets returns becomes_unplayable=None."""
        arch = self._small_arch()
        shim_net = self._make_net(arch)
        offsets = shim_net._choice_embed_offsets()
        assert offsets.becomes_unplayable is None

    def test_v1_0_kept_multihot_offset_shifted_left(self) -> None:
        """With include_setup, kept_multihot offset is shifted left by the stripe width."""
        arch = architecture.ModelArchitecture(
            trunk_layers=(8, 8),
            choice_layers=(8, 8),
            head_layers=(),
            value_layers=(),
            card_embed_dim=4,
        )
        live_spec = encode.EncodingSpec(include_setup=True)
        live_net = core.PolicyValueNet(spec=live_spec, arch=arch)
        shim_net = compat_v1_0.PolicyValueNetV1_0(spec=live_spec, arch=arch)

        live_offsets = live_net._choice_embed_offsets()
        shim_offsets = shim_net._choice_embed_offsets()

        assert live_offsets.kept_multihot is not None
        assert shim_offsets.kept_multihot is not None
        assert (
            live_offsets.kept_multihot - shim_offsets.kept_multihot
            == encode.CHOICE_BECOMES_UNPLAYABLE_DIM
        )

    def test_v1_0_becomes_playable_offset_unchanged(self) -> None:
        """becomes_playable offset is the same in both live and shim."""
        arch = self._small_arch()
        live_net = core.PolicyValueNet(arch=arch)
        shim_net = self._make_net(arch)
        assert (
            live_net._choice_embed_offsets().becomes_playable
            == shim_net._choice_embed_offsets().becomes_playable
        )

    def test_v1_0_forward_pass_runs(self) -> None:
        """A forward pass through the shim's tensors does not raise."""
        eng, *_ = engine.Engine.create(seed=101)
        arch = self._small_arch()
        shim_net = self._make_net(arch)

        decision = decisions.MainActionDecision(
            player_id=0,
            prompt="action",
            choices=[
                decisions.MainActionChoice(
                    label="food", action=decisions.MainAction.GAIN_FOOD
                ),
            ],
        )
        # Encode via the shim's own encode_choices (returns narrowed v1.0 width).
        state_vec = shim_net.encode_state(eng.state, decision)
        choice_feats = shim_net.encode_choices(decision, eng.state)
        family_idx = decisions.family_index_for(type(decision))

        state_t = torch.from_numpy(state_vec).unsqueeze(0)
        choices_t = torch.from_numpy(choice_feats).unsqueeze(0)
        mask_t = torch.ones(1, choices_t.shape[1])
        family_t = torch.tensor([family_idx], dtype=torch.long)

        logits, value = shim_net(state_t, choices_t, mask_t, family_t)
        assert logits.shape == (1, 1)
        assert value.shape == (1,)
