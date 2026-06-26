from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wingspan.cards import load_all, EffectKind

TARGET = ["Atlantic Puffin", "Whooping Crane", "Wood Stork", "Spotted Owl"]

def test_draw_bonus_keep_parsed():
    birds, _, _ = load_all()
    by_name = {b.name: b for b in birds}
    for name in TARGET:
        assert name in by_name, f"{name} not found"
        b = by_name[name]
        assert not any(e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects), f"{name} UNIMPLEMENTED"
        assert any(e.kind == EffectKind.DRAW_BONUS_KEEP for e in b.power.effects), f"{name} missing DRAW_BONUS_KEEP"
