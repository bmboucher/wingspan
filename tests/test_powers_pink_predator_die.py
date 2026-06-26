from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wingspan.cards import load_all, EffectKind

def test_pink_predator_die_parsed():
    birds, _, _ = load_all()
    by_name = {b.name: b for b in birds}
    for name in ["Black Vulture", "Black-Billed Magpie", "Turkey Vulture"]:
        b = by_name[name]
        assert not any(e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects), f"{name} UNIMPL"
        assert any(e.kind == EffectKind.PINK_OPP_PREDATOR_DIE for e in b.power.effects), f"{name} missing"
