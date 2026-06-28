from __future__ import annotations

from wingspan.cards import EffectKind, load_all


def test_green_heron_trade_food():
    birds, _, _ = load_all()
    by_name = {b.name: b for b in birds}
    b = by_name["Green Heron"]
    assert not any(e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects)
    assert any(e.kind == EffectKind.TRADE_WILD_FOOD for e in b.power.effects)
