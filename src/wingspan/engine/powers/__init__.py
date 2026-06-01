"""Bird power dispatch.

``dispatch_power`` iterates a played bird's parsed ``Power`` effects and forwards
each to ``apply_effect`` — one lookup in a registry of per-``EffectKind`` handlers.
The package is split by handler family; each submodule registers its handlers via
``@registry.handles``, and this ``__init__`` imports them all so the registry is
fully populated on first use.

- ``registry``        — the ``_HANDLERS`` table + the ``handles`` decorator
- ``dispatch``        — ``dispatch_power`` / ``apply_effect`` / ``lay_one_egg_on_nest``
- ``grants``          — direct food / egg / card grants
- ``egg_trade``       — discard-egg-for-wild trade
- ``multi_actor``     — each-player / all-players prompts
- ``tray_trade``      — tray draw, wild-food trade, fewest-forest die
- ``drafting``        — extra play + card drafting
- ``nest_aggregate``  — nest-targeted eggs + aggregate food / tuck
- ``predator_repeat`` — predator hunt, bird movement, power repeat
"""

from wingspan.engine.powers import (
    drafting,
    egg_trade,
    grants,
    multi_actor,
    nest_aggregate,
    predator_repeat,
    tray_trade,
)
from wingspan.engine.powers.dispatch import (
    apply_effect,
    dispatch_power,
    lay_one_egg_on_nest,
)

# The handler submodules are imported for their ``@registry.handles`` side
# effects (they populate ``registry._HANDLERS`` on load). The reference below
# keeps the otherwise-unused imports live.
_ = (
    drafting,
    egg_trade,
    grants,
    multi_actor,
    nest_aggregate,
    predator_repeat,
    tray_trade,
)

__all__ = [
    "apply_effect",
    "dispatch_power",
    "lay_one_egg_on_nest",
]
