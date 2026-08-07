"""The relay agent -- the opponent's seat (seat 1).

Every engine decision on the opponent's behalf becomes a "what did they do?"
prompt driven from what the user can see across the physical table: a
pre-turn dialog (``hooks.AidHandler.turn_start``) already recorded which
bird(s) they played this turn, so the first two decision shapes auto-answer
from that report without prompting again. Anything routed entirely through
face-down cards (draft piles, unseen discards) auto-picks rather than asking
the user to guess an identity they cannot know. The opponent's own setup pick
is inferred from what is physically visible -- how many cards they kept and
which foods -- rather than asked outright.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.agents import cli as agents_cli
from wingspan.aid import console as console_module
from wingspan.aid import models, placeholders
from wingspan.cards import lookup
from wingspan.engine import core as engine_core


def relay_agent(
    con: console_module.Console,
    echo: console_module.LogEcho,
    registry: placeholders.PlaceholderRegistry,
    notes: models.TurnNotes,
) -> engine_core.Agent:
    """Build the seat-1 agent: auto-answer whatever the pre-turn hook already
    recorded, auto-pick when every option is a hidden card, special-case the
    opponent's setup pick, and otherwise ask a plain "what did they do?"
    menu."""

    def agent[C: decisions.Choice](
        engine: engine_core.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        echo.flush()

        noted = _auto_answer_from_notes(con, notes, decision)
        if noted is not None:
            return typing.cast(C, noted)

        if _all_choices_are_placeholders(registry, decision):
            con.say("opponent used a hidden card")
            return decision.choices[0]

        if decisions.is_setup_decision(decision):
            setup_decision = typing.cast(decisions.SetupDecision, decision)
            return typing.cast(C, _resolve_opponent_setup(con, setup_decision))

        player = engine.state.players[decision.player_id]
        return typing.cast(C, _resolve_generic(con, registry, decision, player))

    return agent


###### PRIVATE #######

#### Turn-note auto-answers ####


def _auto_answer_from_notes(
    con: console_module.Console,
    notes: models.TurnNotes,
    decision: decisions.Decision[typing.Any],
) -> decisions.Choice | None:
    """Auto-answer a ``MainActionDecision``/``PlayBirdDecision`` from an
    already-reported opponent play, without prompting. Returns ``None`` when
    there is no unconsumed note (or, for ``PlayBirdDecision``, no matching
    choice) so the caller falls through to the next stage."""
    has_unconsumed_note = notes.play_consumed_count < len(notes.plays)
    if isinstance(decision, decisions.MainActionDecision) and has_unconsumed_note:
        for choice in decision.choices:
            if choice.action == decisions.MainAction.PLAY_BIRD:
                con.say("opponent plays a bird (auto-selected from your report)")
                return choice
    if isinstance(decision, decisions.PlayBirdDecision) and has_unconsumed_note:
        note = notes.plays[notes.play_consumed_count]
        for choice in decision.choices:
            if choice.bird is note.bird and choice.habitat == note.habitat:
                notes.play_consumed_count += 1
                con.say(f"opponent plays {note.bird.name} in {note.habitat.value}")
                return choice
    return None


#### Hidden-card auto-pick ####


def _all_choices_are_placeholders(
    registry: placeholders.PlaceholderRegistry,
    decision: decisions.Decision[typing.Any],
) -> bool:
    """Whether every offered choice carries a placeholder bird or bonus card
    -- covers draft-pile picks, end-of-turn discards of unknown cards, and
    the opponent's deferred setup bonus pick (the ``split_setup_bonus``
    regime), where the user cannot know the identity either. ``False`` when
    any choice carries no card at all (mixed shapes never auto-pick)."""
    hidden_cards = [
        _carried_card(choice)
        for choice in decision.choices
        if _carried_card(choice) is not None
    ]
    if len(hidden_cards) != len(decision.choices):
        return False
    return all(
        card is not None and registry.is_placeholder(card) for card in hidden_cards
    )


def _carried_card(
    choice: decisions.Choice,
) -> cards.Bird | cards.BonusCard | None:
    """The hidden-capable card a choice carries: the bird of a bird-carrying
    choice, the bonus card of a ``BonusCardChoice``, else ``None``."""
    if isinstance(choice, (decisions.BirdChoice, decisions.PlayBirdChoice)):
        return choice.bird
    if isinstance(choice, decisions.BonusCardChoice):
        return choice.bonus_card
    return None


#### Opponent setup matching ####


def _resolve_opponent_setup(
    con: console_module.Console, decision: decisions.SetupDecision
) -> decisions.SetupChoice:
    """Infer the opponent's setup keep from what is physically visible: how
    many cards they kept, and (when the regime's ``SetupChoice`` carries a
    food axis) which foods. Returns the first offered choice matching both
    criteria -- any axis the regime omits (food, and always bonus, since the
    opponent's kept bonus is never visible) is left unconstrained, so ties
    resolve arbitrarily among functionally-identical placeholder options."""
    n_kept = _ask_int_in_range(
        con,
        f"How many bird cards did the opponent keep? (0-{state.STARTING_HAND_SIZE})",
        0,
        state.STARTING_HAND_SIZE,
    )
    kept_foods: tuple[cards.Food, ...] = ()
    if decision.choices[0].kept_foods:
        n_foods = len(cards.ALL_FOODS) - n_kept
        kept_foods = _ask_distinct_foods(con, n_foods)
    for choice in decision.choices:
        if len(choice.kept_cards) == n_kept and sorted(choice.kept_foods) == sorted(
            kept_foods
        ):
            return choice
    raise AssertionError("no offered SetupChoice matched the entered opponent keep")


def _ask_int_in_range(
    con: console_module.Console, prompt: str, low: int, high: int
) -> int:
    """Prompt for an integer in ``[low, high]``, re-asking until valid."""
    while True:
        raw = con.ask(prompt)
        if raw.isdigit() and low <= int(raw) <= high:
            return int(raw)
        con.say(f"Enter a number from {low} to {high}.")


def _ask_distinct_foods(
    con: console_module.Console, count: int
) -> tuple[cards.Food, ...]:
    """Ask which ``count`` distinct foods the opponent kept, comma-separated,
    re-asking until exactly ``count`` distinct resolvable foods are entered."""
    prompt = f"Which {count} food(s) did the opponent keep, comma-separated? "
    while True:
        answer = con.ask(prompt)
        tokens = [token.strip() for token in answer.split(",") if token.strip()]
        resolved = [lookup.find_food(token) for token in tokens]
        distinct = set(resolved)
        if len(tokens) == count and None not in distinct and len(distinct) == count:
            return tuple(food for food in resolved if food is not None)
        con.say(f"Enter exactly {count} distinct, resolvable food names.")


#### Generic fallback ####


def _resolve_generic(
    con: console_module.Console,
    registry: placeholders.PlaceholderRegistry,
    decision: decisions.Decision[typing.Any],
    player: state.Player,
) -> decisions.Choice:
    """The plain "what did the opponent do?" menu -- placeholder-carrying
    choices render as face-down rather than exposing hidden identities."""
    con.say(decision.prompt)
    for idx, choice in enumerate(decision.choices):
        carried = _carried_card(choice)
        if carried is not None and registry.is_placeholder(carried):
            con.say(f"  [{idx}] (face-down card)")
        else:
            con.say(agents_cli.format_choice_line(idx, choice, player))
    while True:
        raw = con.ask("opponent's move> ")
        if raw.isdigit() and int(raw) < len(decision.choices):
            return decision.choices[int(raw)]
        con.say("enter a valid choice index")
