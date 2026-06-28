from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wingspan.cards import load_all, EffectKind

TARGET_BIRDS = {
    "Barred Owl": 75, "Cooper's Hawk": 75, "Golden Eagle": 100,
    "Great Horned Owl": 100, "Greater Roadrunner": 50, "Northern Harrier": 75,
    "Peregrine Falcon": 100, "Red-Shouldered Hawk": 75, "Red-Tailed Hawk": 75,
    "Swainson's Hawk": 75,
}

def test_predator_tuck_parsed():
    birds, _, _ = load_all()
    by_name = {b.name: b for b in birds}
    for name, threshold in TARGET_BIRDS.items():
        assert name in by_name, f"{name} not found"
        b = by_name[name]
        assert not any(e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects), f"{name} still UNIMPLEMENTED"
        effs = [e for e in b.power.effects if e.kind == EffectKind.PREDATOR_HUNT]
        assert effs, f"{name} missing PREDATOR_HUNT"
        assert effs[0].max_wingspan_cm == threshold, f"{name} wrong threshold"
