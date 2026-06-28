from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wingspan.cards import load_all, EffectKind, NestType


def test_all_players_lay_on_nest_parsed():
    birds, _, _ = load_all()
    by_name = {b.name: b for b in birds}
    cases = [
        ("Lazuli Bunting", NestType.BOWL),
        ("Pileated Woodpecker", NestType.CAVITY),
        ("Western Meadowlark", NestType.GROUND),
    ]
    for name, expected_nest in cases:
        b = by_name[name]
        assert not any(e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects), f"{name} UNIMPL"
        effs = [e for e in b.power.effects if e.kind == EffectKind.ALL_PLAYERS_LAY_EGG_ON_NEST]
        assert effs, f"{name} missing ALL_PLAYERS_LAY_EGG_ON_NEST effect"
        assert effs[0].nest == expected_nest, f"{name}: expected {expected_nest}, got {effs[0].nest}"
