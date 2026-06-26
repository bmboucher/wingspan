from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wingspan.cards import load_all, EffectKind

def test_draft_cards_parsed():
    birds, _, _ = load_all()
    by_name = {b.name: b for b in birds}
    b = by_name["American Oystercatcher"]
    assert not any(e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects)
    assert any(e.kind == EffectKind.DRAFT_CARDS for e in b.power.effects)
