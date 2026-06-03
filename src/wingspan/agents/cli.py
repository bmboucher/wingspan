"""Interactive human (stdin/stdout) agent and the mixed-hotseat helper.

The CLI agent walks a human through the offered choices index-by-index.
The combined ``SetupDecision`` is special-cased: enumerating all 504 legal
setup combinations on one screen is unusable, so the setup pick is broken
into three sub-dialogs (cards / foods / bonus) — each driven by an
arrow-key checkbox/radio widget from ``agents.interactive`` — and the
human's answer is reassembled back into one of the offered ``SetupChoice``
instances.
"""

from __future__ import annotations

import random
import typing

from wingspan import cards, decisions, state
from wingspan.agents import base, display, interactive
from wingspan.engine import core as engine_core


def cli_agent() -> engine_core.Agent:
    """Interactive human agent. Prints prompt and choices, reads index."""
    interactive.enable_ansi()

    def agent[C: decisions.Choice](
        engine: engine_core.Engine,
        decision: decisions.Decision[C],
    ) -> C:
        if isinstance(decision, decisions.SetupDecision):
            # SetupDecision is Decision[SetupChoice], but the type checker
            # can't propagate that narrowing back onto the bound C — so the
            # SetupChoice return is cast through.
            return typing.cast(
                C,
                _cli_resolve_setup_choice(
                    decision,
                    [b for b in engine.state.tray if b is not None],
                ),
            )
        # The main-action prompt — and the play-bird menu it can open — are the
        # natural moments to show the full board: the human needs the resource
        # picture both to pick an action type and to pick which bird to play.
        if isinstance(
            decision,
            (decisions.MainActionDecision, decisions.PlayBirdDecision),
        ):
            print()
            print(
                display.format_board(
                    engine.state, engine.state.players[decision.player_id]
                )
            )
        print()
        print(decision.prompt)
        player = engine.state.players[decision.player_id]
        for i, choice in enumerate(decision.choices):
            print(_format_choice_line(i, choice, player))
        while True:
            raw = input("choice> ").strip()
            try:
                idx = int(raw)
            except ValueError:
                print("  enter a number")
                continue
            if 0 <= idx < len(decision.choices):
                return typing.cast(C, decision.choices[idx])
            print("  out of range")

    return agent


def mixed_agents(
    rng: random.Random,
    human_index: int,
) -> tuple[engine_core.Agent, engine_core.Agent]:
    """Two-player roster with one human at ``human_index`` and a random opponent."""
    agent_a = cli_agent() if human_index == 0 else base.random_agent(rng)
    agent_b = cli_agent() if human_index == 1 else base.random_agent(rng)
    return (agent_a, agent_b)


###### PRIVATE #######


def _format_choice_line(
    idx: int, choice: decisions.Choice, player: state.Player
) -> str:
    """Render one offered choice line with type-aware extra context.

    Bird- and bonus-card-carrying choices are expanded to show food cost,
    power text, and scoring conditions respectively — the engine's stored
    ``label`` is too terse for a human at decision time. A bonus card also
    gets a second line with how useful it is to ``player`` right now. Other
    Choice subclasses fall through to ``label`` as-is.
    """
    if isinstance(choice, decisions.PlayBirdChoice):
        # The board (printed in full above the main-action prompt) already
        # shows every hand card's stats and power, so a play option only needs
        # the bird name, target habitat, and the specific payment.
        return (
            f"  [{idx}] play {choice.bird.name} in {choice.habitat.value} "
            f"for {display.format_food_pool(choice.payment)}"
        )
    if isinstance(choice, decisions.BirdChoice):
        return f"  [{idx}] {display.format_bird_full(choice.bird)}"
    if isinstance(choice, decisions.BonusCardChoice):
        head = f"  [{idx}] {display.format_bonus(choice.bonus_card)}"
        return (
            f"{head}\n      "
            f"{display.format_bonus_score_now(choice.bonus_card, player)}"
        )
    if isinstance(choice, decisions.PlayedBirdChoice):
        return f"  [{idx}] {display.format_played_bird(choice.played_bird)}"
    if isinstance(choice, decisions.DrawSourceChoice):
        if choice.bird is not None:
            return f"  [{idx}] {display.format_bird_full(choice.bird)}"
        return f"  [{idx}] Draw from the Deck"
    return f"  [{idx}] {choice.display_label()}"


def _cli_resolve_setup_choice(
    decision: decisions.SetupDecision,
    tray: list[cards.Bird],
) -> decisions.SetupChoice:
    """Two-step sub-dialog for the combined setup pick.

    Step 1 is one screen: keep any subset of the dealt birds and pick exactly
    one bonus card. Step 2 then keeps the matching number of foods. The
    assembled answer is located among ``decision.choices`` and returned.
    ``tray`` is the face-up bird display, used to gauge bonus-card value.
    """
    dealt_cards = decision.dealt_cards
    dealt_bonus = decision.dealt_bonus

    print()
    print(decision.prompt)

    kept_cards, bonus_card = _cli_pick_hand_and_bonus(dealt_cards, dealt_bonus, tray)
    kept_foods = _cli_pick_kept_foods(kept_cards)

    for choice in decision.choices:
        if (
            choice.kept_cards == kept_cards
            and choice.kept_foods == kept_foods
            and choice.bonus_card == bonus_card
        ):
            return choice
    raise AssertionError(
        "assembled setup answer did not match any offered SetupChoice: "
        f"keep={[bird.name for bird in kept_cards]} foods={kept_foods} bonus={bonus_card}"
    )


def _cli_pick_hand_and_bonus(
    dealt_cards: list[cards.Bird],
    dealt_bonus: list[cards.BonusCard],
    tray: list[cards.Bird],
) -> tuple[tuple[cards.Bird, ...], cards.BonusCard | None]:
    """Step 1: keep any subset of birds (each costs 1 food) plus one bonus card.

    Birds and bonus cards share one screen as a two-section form; the bird
    section is unconstrained (0+) while the bonus section is a radio requiring
    exactly one pick. Each bonus card shows how many qualifying birds sit in
    hand vs. the ``tray`` display. When no bonus cards were dealt the bonus
    section is omitted and ``None`` is returned for it.
    """
    name_width = max((len(bird.name) for bird in dealt_cards), default=0)
    bird_options = [
        display.format_bird_full(bird, indent="", name_width=name_width)
        for bird in dealt_cards
    ]
    sections = [
        interactive.Section(
            title="Keep which birds? (each kept bird costs 1 food)",
            options=bird_options,
        )
    ]
    live: typing.Callable[[list[set[int]]], list[list[str]]] | None = None
    if dealt_bonus:
        bonus_width = max(len(bc.name) for bc in dealt_bonus)

        def render_bonus(selected: list[cards.Bird]) -> list[str]:
            return [
                display.format_bonus_with_setup_help(
                    bc, dealt_cards, tray, selected, bonus_width
                )
                for bc in dealt_bonus
            ]

        sections.append(
            interactive.Section(
                title="Keep which bonus card? (choose exactly one)",
                options=render_bonus([]),
                mode=interactive.Mode.SINGLE,
            )
        )

        # The bonus help line counts selected matching birds, so re-render it
        # each frame against the live bird-section (index 0) selection.
        def _live(selections: list[set[int]]) -> list[list[str]]:
            selected = [dealt_cards[i] for i in selections[0]]
            return [bird_options, render_bonus(selected)]

        live = _live

    picks = interactive.select_form(
        sections, header="Step 1 — choose your opening hand:", live_options=live
    )
    kept_cards = tuple(dealt_cards[i] for i in picks[0])
    bonus_card = dealt_bonus[picks[1][0]] if dealt_bonus else None
    return kept_cards, bonus_card


def _cli_pick_kept_foods(
    kept_cards: tuple[cards.Bird, ...],
) -> tuple[cards.Food, ...]:
    """Step 2: check which ``keep_count`` distinct foods to keep.

    ``keep_count`` is ``len(ALL_FOODS) - len(kept_cards)``, since each kept card
    costs one food off the player's one-of-each starting stash. A live footer
    shows which of the ``kept_cards`` the current food selection could play
    immediately. The food order in the returned tuple follows
    :data:`cards.ALL_FOODS` to match the enumeration the engine offers.
    """
    keep_count = len(cards.ALL_FOODS) - len(kept_cards)
    section = interactive.Section(
        options=[food.value for food in cards.ALL_FOODS],
        required_count=keep_count,
    )

    def _can_play_footer(selections: list[set[int]]) -> list[str]:
        chosen_foods = [cards.ALL_FOODS[i] for i in selections[0]]
        return [display.format_can_play(kept_cards, chosen_foods)]

    picks = interactive.select_form(
        [section],
        header=(
            f"Step 2 — keep which {keep_count} food(s)? "
            f"(you start with one of each)"
        ),
        live_footer=_can_play_footer,
    )
    return tuple(cards.ALL_FOODS[i] for i in picks[0])
