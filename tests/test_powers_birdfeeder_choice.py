from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from wingspan.cards import load_all, EffectKind, Food

def test_birdfeeder_choice_parsed():
    birds, _, _ = load_all()
    by_name = {b.name: b for b in birds}
    inv_fruit = [("Indigo Bunting", Food.INVERTEBRATE, Food.FRUIT),
                 ("Western Tanager", Food.INVERTEBRATE, Food.FRUIT)]
    seed_fruit = [("Rose-Breasted Grosbeak", Food.SEED, Food.FRUIT)]
    for name, fa, fb in inv_fruit + seed_fruit:
        b = by_name[name]
        assert not any(e.kind == EffectKind.UNIMPLEMENTED for e in b.power.effects), f"{name} UNIMPL"
        effs = [e for e in b.power.effects if e.kind == EffectKind.GAIN_FOOD_BIRDFEEDER_CHOICE]
        assert effs, f"{name} missing"
        assert fa in effs[0].extra and fb in effs[0].extra, f"{name} wrong foods"
