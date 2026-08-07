"""Shared setup/identification dialogs for the aid session.

These are the "must-identify" dialogs -- unlike ``SessionOracle``'s reveal
prompts (which allow a blank answer to mint a placeholder for a face-down
card), every helper here loops until it resolves a real card: the advisor's
placeholder sweep, the relay's opponent-turn hand surgery, and
:func:`run_setup_entry` (the pre-game dialog) all need the *actual* card, not
a stand-in. Free-text answers resolve through :mod:`wingspan.cards.lookup`,
with a menu-based disambiguation step whenever a query matches more than one
candidate.
"""

from __future__ import annotations

import collections.abc
import typing

from wingspan import cards, state
from wingspan.agents import display
from wingspan.aid import console as console_module
from wingspan.aid import models
from wingspan.aid import oracle as oracle_module
from wingspan.cards import lookup

# How many round goals a setup entry always carries -- one per round.
_N_ROUNDS = len(state.ROUND_CUBES)


def identify_bird(con: console_module.Console, prompt: str) -> cards.Bird:
    """Ask for a bird name until it resolves to exactly one card -- no blank
    "face-down" escape, unlike ``SessionOracle.reveal_bird``."""
    return _resolve_named(con, con.ask(prompt), lookup.find_birds, "bird name")


def identify_bonus(con: console_module.Console, prompt: str) -> cards.BonusCard:
    """Bonus-card analogue of :func:`identify_bird`."""
    return _resolve_named(
        con, con.ask(prompt), lookup.find_bonus_cards, "bonus card name"
    )


def pick_habitat(con: console_module.Console, bird: cards.Bird) -> cards.Habitat:
    """Which of ``bird``'s legal habitats it was played in -- resolved
    without a prompt when only one habitat is legal."""
    if len(bird.habitats) == 1:
        return bird.habitats[0]
    chosen_idx = con.menu(
        f"Which habitat did {bird.name} play in?",
        [habitat.value for habitat in bird.habitats],
    )
    return bird.habitats[chosen_idx]


def run_setup_entry(con: console_module.Console) -> models.SetupEntry:
    """The pre-game dialog: everything the physical deal produced, entered in
    the order it happens at the table -- start seat, hand, bonus pair, round
    goals, tray, feeder roll. Each block loops on a "Correct?" confirm-echo
    before the next one starts."""
    start_player = _pick_start_player(con)
    hand = _collect_hand(con)
    bonus_pair = _collect_bonus_pair(con)
    goals = _collect_goals(con)
    tray = _collect_tray(con)
    feeder = _collect_feeder(con)
    return models.SetupEntry(
        hand=hand,
        bonus_pair=bonus_pair,
        goals=goals,
        tray=tray,
        feeder=feeder,
        start_player=start_player,
    )


###### PRIVATE #######

#### Name resolution ####


class _NamedCard(typing.Protocol):
    """Structural bound for :func:`_resolve_named`: both ``cards.Bird`` and
    ``cards.BonusCard`` expose a plain ``name`` field used for the
    disambiguation menu."""

    name: str


def _resolve_named[T: _NamedCard](
    con: console_module.Console,
    query: str,
    find: collections.abc.Callable[[str], list[T]],
    kind_label: str,
) -> T:
    """Resolve ``query`` to a ``T`` via ``find``, re-prompting -- never
    minting a placeholder -- until it matches; opens a disambiguation menu
    when multiple candidates tie."""
    matches = find(query)
    while not matches:
        query = con.ask(f'No match for "{query}" — enter the {kind_label} again: ')
        matches = find(query)
    if len(matches) == 1:
        return matches[0]
    chosen_idx = con.menu(
        f'Multiple matches for "{query}" — which one?',
        [item.name for item in matches],
    )
    return matches[chosen_idx]


def _resolve_goal(con: console_module.Console, query: str) -> cards.EndRoundGoal:
    """Goal analogue of :func:`_resolve_named`: goals have no printed name,
    so the disambiguation menu shows each candidate's description instead."""
    matches = lookup.find_goals(query)
    while not matches:
        query = con.ask(f'No match for "{query}" — enter the round goal again: ')
        matches = lookup.find_goals(query)
    if len(matches) == 1:
        return matches[0]
    chosen_idx = con.menu(
        f'Multiple matches for "{query}" — which one?',
        [goal.description for goal in matches],
    )
    return matches[chosen_idx]


#### Confirm-echo block runner ####


def _run_until_confirmed[T](
    con: console_module.Console,
    build: collections.abc.Callable[[], T],
    echo: collections.abc.Callable[[T], None],
) -> T:
    """Run ``build``, echo the result via ``echo``, and confirm with the
    user; redo (call ``build`` again) on "no"."""
    while True:
        value = build()
        echo(value)
        if con.confirm("Correct?"):
            return value


#### Setup-entry blocks ####


def _pick_start_player(con: console_module.Console) -> int:
    """Who takes the first turn -- a menu pick needs no confirm-echo, since
    the selection is already unambiguous."""
    return con.menu("Who goes first?", ["You", "Opponent"])


def _collect_hand(con: console_module.Console) -> tuple[cards.Bird, ...]:
    """Your ``STARTING_HAND_SIZE`` dealt birds, entered as one
    comma-separated line and resolved token by token."""

    def build() -> tuple[cards.Bird, ...]:
        while True:
            answer = con.ask(
                f"Enter your {state.STARTING_HAND_SIZE} dealt birds, "
                "comma-separated: "
            )
            token_lists = lookup.parse_bird_list(answer)
            if len(token_lists) != state.STARTING_HAND_SIZE:
                con.say(
                    f"Enter exactly {state.STARTING_HAND_SIZE} bird names, "
                    "comma-separated."
                )
                continue
            tokens = [token.strip() for token in answer.split(",")]
            return tuple(
                _resolve_named(con, token, lookup.find_birds, "bird name")
                for token in tokens
            )

    def echo(hand: tuple[cards.Bird, ...]) -> None:
        for bird in hand:
            con.say(display.format_bird(bird))

    return _run_until_confirmed(con, build, echo)


def _collect_bonus_pair(con: console_module.Console) -> tuple[cards.BonusCard, ...]:
    """Your ``STARTING_BONUS_CARDS_DEAL`` dealt bonus cards, one at a time."""

    def build() -> tuple[cards.BonusCard, ...]:
        return tuple(
            identify_bonus(
                con,
                f"Enter dealt bonus card {slot + 1} of "
                f"{state.STARTING_BONUS_CARDS_DEAL}: ",
            )
            for slot in range(state.STARTING_BONUS_CARDS_DEAL)
        )

    def echo(bonus_pair: tuple[cards.BonusCard, ...]) -> None:
        for bonus_card in bonus_pair:
            con.say(display.format_bonus(bonus_card))

    return _run_until_confirmed(con, build, echo)


def _collect_goals(con: console_module.Console) -> tuple[cards.EndRoundGoal, ...]:
    """The four end-of-round goal tiles, one per round."""

    def build() -> tuple[cards.EndRoundGoal, ...]:
        return tuple(
            _resolve_goal(con, con.ask(f"Enter round {round_num + 1}'s goal: "))
            for round_num in range(_N_ROUNDS)
        )

    def echo(goals: tuple[cards.EndRoundGoal, ...]) -> None:
        for round_num, goal in enumerate(goals):
            con.say(f"Round {round_num + 1}: {goal.description}")

    return _run_until_confirmed(con, build, echo)


def _collect_tray(con: console_module.Console) -> tuple[cards.Bird, ...]:
    """The initial face-up tray, left to right."""

    def build() -> tuple[cards.Bird, ...]:
        return tuple(
            identify_bird(
                con,
                f"Enter tray slot {slot + 1} of {state.TRAY_SIZE} " "(left to right): ",
            )
            for slot in range(state.TRAY_SIZE)
        )

    def echo(tray: tuple[cards.Bird, ...]) -> None:
        for slot, bird in enumerate(tray):
            con.say(f"Tray slot {slot + 1}: {display.format_bird(bird)}")

    return _run_until_confirmed(con, build, echo)


def _collect_feeder(con: console_module.Console) -> models.FeederEntry:
    """The initial birdfeeder roll, via the shared die-face grammar."""
    prompt = (
        f"Enter all {state.BIRDFEEDER_DICE} birdfeeder die faces, "
        'space-separated (food name/alias, or "choice"): '
    )

    def build() -> models.FeederEntry:
        while True:
            parsed = oracle_module.parse_die_faces(
                con.ask(prompt), state.BIRDFEEDER_DICE
            )
            if parsed is not None:
                counts, choice_dice = parsed
                return models.FeederEntry(
                    counts=list(counts.counts), choice_dice=choice_dice
                )
            con.say(
                f"Enter exactly {state.BIRDFEEDER_DICE} faces "
                '(food name/alias, or "choice"), separated by spaces.'
            )

    def echo(feeder: models.FeederEntry) -> None:
        con.say(
            f"Feeder: {display.format_food_pool(feeder.to_food_pool())}, "
            f"choice dice: {feeder.choice_dice}"
        )

    return _run_until_confirmed(con, build, echo)
