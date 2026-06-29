from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wingspan.cards import EffectKind, Food, load_all


def test_gain_all_birdfeeder_parsed():
    birds, _, _ = load_all()
    by_name = {b.name: b for b in birds}
    cases = [("Bald Eagle", Food.FISH), ("Northern Flicker", Food.INVERTEBRATE)]
    for name, expected_food in cases:
        b = by_name[name]
        assert not any(
            e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects
        ), f"{name} UNIMPL"
        effs = [e for e in b.power.effects if e.kind == EffectKind.GAIN_ALL_FOOD_FEEDER]
        assert effs, f"{name} missing effect"
        assert effs[0].food == expected_food, f"{name} wrong food"
