"""Tests for the v1.0 → v1.1 trunk-final-activation compat shim.

v1.0 behavior: ``trunk_final_activation=None`` fell back to ``between_activation``
(typically relu).  v1.1 behavior: falls back to ``final_activation`` (typically
none).  The shim class ``PolicyValueNetV1_0`` restores the v1.0 rule so that
rehydrated v1.0 checkpoints compute identically to what they did at training time.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")
from torch import nn

from wingspan import architecture, version
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
