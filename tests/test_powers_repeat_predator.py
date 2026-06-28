from __future__ import annotations

from wingspan.cards import EffectKind, load_all


def test_hooded_merganser_parsed():
    birds, _, _ = load_all()
    by_name = {b.name: b for b in birds}
    b = by_name["Hooded Merganser"]
    assert not any(e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects)
    assert any(e.kind == EffectKind.REPEAT_PREDATOR_POWER for e in b.power.effects)
