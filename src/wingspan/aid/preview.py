"""Combined upfront setup-keep recommendation for the aid advisor.

Under a split-setup regime the setup ``SetupDecision`` only offers a
card-keep pick, with the bonus and/or food picks arriving later as separate
in-game decisions (see ``wingspan.engine.setup_flow``). :func:`preview_setup`
simulates the model's own preferred keep through that *same*
deferred-resolution code path, on a throwaway clone of the live
``GameState``, so the advisor can show one combined recommendation line
before asking what was actually played at the table -- and so that line is
guaranteed consistent with whatever the later per-step advice ends up
showing, since both come from ``inner``, the same policy. Costs at most
~3 extra forward passes (one bonus pick plus up to two food picks) beyond
the one the advisor already ran for the real decision.
"""

from __future__ import annotations

import copy

from wingspan import decisions
from wingspan.agents import cli as agents_cli
from wingspan.aid import models
from wingspan.engine import core as engine_core
from wingspan.engine import setup_flow
from wingspan.players import decision_probe


def preview_setup(
    engine: engine_core.Engine,
    inner: engine_core.Agent,
    probe: decision_probe.DecisionProbe,
    decision: decisions.SetupDecision,
    preferred: decisions.SetupChoice,
) -> models.SetupPreview:
    """Replay ``preferred`` through the real deferred-resolution steps on a
    cloned state, returning the resulting kept cards / bonus card / food pool.

    Runs entirely on ``copy.deepcopy(engine.state)``: a throwaway ``Engine``
    wraps the clone so ``setup_flow``'s free functions -- and, under the
    deferred-food regime, ``inner``'s own forward passes -- never touch the
    real game state or prompt the real console. Drains ``probe`` once at the
    end so the preview's own ``inner`` calls don't leak into the caller's
    annotation; the caller must already have taken the real decision's
    value/annotation off ``probe`` before calling this function.
    """
    clone = copy.deepcopy(engine.state)
    clone_player = clone.players[decision.player_id]
    preview_engine = engine_core.Engine(
        clone,
        agents=[inner] * len(clone.players),
        combine_gain_food=engine.combine_gain_food,
    )

    # Deferred-food regimes never carry the food axis on the offered
    # choices; the dealt cards/bonus stay the real state's objects (safe --
    # ``apply_setup_choice``/``ledger`` match by Pydantic equality against the
    # clone's own deep-copied hand, not by identity).
    _, ask_food = agents_cli.setup_dialog_axes(decision)
    setup_flow.apply_setup_choice(
        preview_engine,
        clone_player,
        list(decision.dealt_cards),
        list(decision.dealt_bonus),
        preferred,
        defer_food=not ask_food,
    )
    kept_bonus = setup_flow.resolve_deferred_setup_bonus(
        preview_engine, clone_player, list(decision.dealt_bonus), preferred
    )
    setup_flow.resolve_deferred_setup_food(
        preview_engine,
        clone_player,
        inner,
        len(preferred.kept_cards),
        defer_food=not ask_food,
    )

    kept_foods = clone_player.food.model_copy(deep=True)
    probe.take()
    return models.SetupPreview(
        kept_cards=preferred.kept_cards,
        bonus_card=kept_bonus,
        kept_foods=kept_foods,
    )
