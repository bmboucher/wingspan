"""The bird power-text parser entry point.

``parse_power`` normalizes a printed power string and runs it through the
ordered matcher registry, emitting a structured ``schema.Power`` of one or
more ``Effect`` records. Anything unrecognized becomes a single
``UNIMPLEMENTED`` effect so the simulator still runs.
"""

from __future__ import annotations

from wingspan.cards import schema
from wingspan.cards.parse import registry


def parse_power(color: schema.PowerColor, text: str) -> schema.Power:
    """Best-effort parser. Recognises a small set of common patterns and
    returns ``UNIMPLEMENTED`` for everything else. Idempotent and safe to
    call once per bird at load time."""
    text = (text or "").strip()
    if not text:
        return schema.Power(color=color, effects=(), raw_text="")
    normalized = _normalize(text)
    effects = _extract_effects(normalized)
    if not effects:
        effects.append(
            schema.Effect(kind=schema.EffectKind.UNIMPLEMENTED, raw_text=text)
        )
    return schema.Power(color=color, effects=tuple(effects), raw_text=text)


def _normalize(text: str) -> str:
    return text.replace("—", "-").replace("“", '"').replace("”", '"')


def _extract_effects(text: str) -> list[schema.Effect]:
    """Apply each recognized pattern in turn, accumulating matched effects.

    Order matters: more specific patterns must run before less specific
    overlapping patterns (e.g. the "or"-disjunction birdfeeder pattern runs
    before the generic ``Gain N [food] from the birdfeeder``).

    Reactive ("when another player ...") powers are matched only against the
    pink-reactor patterns: their consequent clause ("gain 1 [fish] from the
    supply", "tuck 1 [card] ...", "cache 1 [rodent] ...") would otherwise also
    match a generic when-played matcher and be mis-modelled as the bird's own
    effect, which never fires for a pink bird."""
    matchers = registry.matchers_for(_is_reactive(text))
    effects: list[schema.Effect] = []
    for matcher in matchers:
        eff = matcher(text)
        if eff is not None:
            effects.append(eff)
    return effects


def _is_reactive(text: str) -> bool:
    """Whether ``text`` is a pink between-turn power that triggers off another
    player's action (every such core power begins "When another player ...")."""
    return text.strip().lower().startswith("when another player")
