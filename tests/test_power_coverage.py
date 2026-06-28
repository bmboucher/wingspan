"""Coverage guards that catch silently-broken card powers before a long run.

``cards.power_coverage`` only counts ``UNIMPLEMENTED`` effects, which is blind to
a *mis-parsed* power — e.g. a pink "when another player ..." reactor whose
consequent matched a generic when-played matcher. These tests pin both: every
core bird power is modelled, and every pink bird routes only to a real reactor
effect kind (so it actually fires between turns).
"""

from __future__ import annotations

from wingspan import cards  # noqa: E402

# The pink effect kinds the engine's reactor hooks actually dispatch. A pink bird
# carrying any other kind would never fire (it is not a when-played/when-activated
# bird, and no reactor would pick it up).
_REACTOR_KINDS = {
    cards.EffectKind.PINK_LAY_EGG_ON_NEST,
    cards.EffectKind.PINK_PREDATOR_FEEDER,
    cards.EffectKind.PINK_PLAY_BIRD_GAIN,
    cards.EffectKind.PINK_PLAY_BIRD_TUCK,
    cards.EffectKind.PINK_GAIN_FOOD_CACHE,
}


def test_no_core_bird_power_is_unimplemented() -> None:
    birds, _, _ = cards.load_all()
    implemented, total = cards.power_coverage(birds)
    unimplemented = [
        bird.name
        for bird in birds
        if any(
            effect.kind == cards.EffectKind.UNIMPLEMENTED
            for effect in bird.power.effects
        )
    ]
    assert implemented == total, f"unmodelled bird powers: {unimplemented}"


def test_every_pink_bird_routes_to_a_reactor_kind() -> None:
    """A pink bird whose parsed effects fall outside the reactor set is dead —
    dispatch_power skips pink, and no reactor would fire it."""
    birds, _, _ = cards.load_all()
    misrouted = {
        bird.name: [effect.kind.value for effect in bird.power.effects]
        for bird in birds
        if bird.color == cards.PowerColor.PINK
        and any(effect.kind not in _REACTOR_KINDS for effect in bird.power.effects)
    }
    assert misrouted == {}, f"pink birds that never fire: {misrouted}"
