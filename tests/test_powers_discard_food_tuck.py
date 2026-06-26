from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wingspan.cards import load_all, EffectKind, Food

def test_discard_food_tuck_parsed():
    birds, _, _ = load_all()
    by_name = {b.name: b for b in birds}
    seed_birds = ["Black-Bellied Whistling-Duck", "Canada Goose", "Sandhill Crane"]
    fish_birds = ["American White Pelican", "Double-Crested Cormorant"]
    for name in seed_birds:
        b = by_name[name]
        effs = [e for e in b.power.effects if e.kind == EffectKind.DISCARD_FOOD_TUCK_FROM_DECK]
        assert effs, f"{name} missing effect"
        assert effs[0].food == Food.SEED
        assert effs[0].amount == 2
    for name in fish_birds:
        b = by_name[name]
        effs = [e for e in b.power.effects if e.kind == EffectKind.DISCARD_FOOD_TUCK_FROM_DECK]
        assert effs, f"{name} missing effect"
        assert effs[0].food == Food.FISH
