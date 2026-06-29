from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wingspan.cards import EffectKind, load_all


def test_discard_egg_gain_wild_parsed():
    birds, _, _ = load_all()
    by_name = {b.name: b for b in birds}
    amount_1 = ["American Crow", "Black-Crowned Night-Heron", "Fish Crow"]
    amount_2 = ["Chihuahuan Raven", "Common Raven"]
    for name in amount_1:
        b = by_name[name]
        effs = [e for e in b.power.effects if e.kind == EffectKind.DISCARD_EGG_FOR_WILD]
        assert effs, f"{name} missing"
        assert effs[0].amount == 1
    for name in amount_2:
        b = by_name[name]
        effs = [e for e in b.power.effects if e.kind == EffectKind.DISCARD_EGG_FOR_WILD]
        assert effs, f"{name} missing"
        assert effs[0].amount == 2
