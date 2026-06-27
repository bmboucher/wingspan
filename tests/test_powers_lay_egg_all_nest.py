from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wingspan.cards import load_all, EffectKind, NestType

BIRDS = [("Ash-Throated Flycatcher", NestType.CAVITY), ("Bobolink", NestType.GROUND),
         ("Inca Dove", NestType.PLATFORM), ("Say's Phoebe", NestType.BOWL)]

def test_lay_egg_all_nest_parsed():
    birds, _, _ = load_all()
    by_name = {b.name: b for b in birds}
    for name, expected_nest in BIRDS:
        b = by_name[name]
        assert not any(e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects), f"{name} UNIMPLEMENTED"
        effs = [e for e in b.power.effects if e.kind == EffectKind.LAY_EGG_ALL_NEST]
        assert effs, f"{name} missing LAY_EGG_ALL_NEST"
        assert effs[0].extra[0] == expected_nest, f"{name} wrong nest type"
