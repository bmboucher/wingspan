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
from wingspan.aid import widgets
from wingspan.cards import lookup

# How many round goals a setup entry always carries -- one per round.
_N_ROUNDS = len(state.ROUND_CUBES)


def identify_bird(
    con: console_module.Console,
    prompt: str,
    *,
    exclude: collections.abc.Collection[cards.Bird] = (),
) -> cards.Bird:
    """Ask for a bird name until it resolves to exactly one card -- no blank
    "face-down" escape, unlike ``SessionOracle.reveal_bird``.

    ``exclude`` only narrows the interactive typeahead's matches -- a dedup
    aid for e.g. successive hand slots so an already-picked bird cannot be
    picked twice. The text-mode fallback ignores it: a resolved free-text
    answer is unambiguous regardless of what else was already picked.
    """
    if con.supports_interactive():

        def find(query: str) -> list[cards.Bird]:
            return [bird for bird in lookup.find_birds(query) if bird not in exclude]

        picked = widgets.typeahead_pick(con, prompt, find, display.format_bird)
        assert picked is not None  # allow_blank defaults to False
        return picked
    return _resolve_named(con, con.ask(prompt), lookup.find_birds, "bird name")


def identify_bonus(
    con: console_module.Console,
    prompt: str,
    *,
    exclude: collections.abc.Collection[cards.BonusCard] = (),
) -> cards.BonusCard:
    """Bonus-card analogue of :func:`identify_bird`, including the
    interactive-only ``exclude`` dedup aid."""
    if con.supports_interactive():

        def find(query: str) -> list[cards.BonusCard]:
            return [
                bonus_card
                for bonus_card in lookup.find_bonus_cards(query)
                if bonus_card not in exclude
            ]

        picked = widgets.typeahead_pick(con, prompt, find, display.format_bonus)
        assert picked is not None  # allow_blank defaults to False
        return picked
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
    tray = _collect_tray(con, hand)
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
    """Your ``STARTING_HAND_SIZE`` dealt birds.

    On an interactive console, one typeahead pick per slot, each excluding
    the birds already picked this hand. The text-mode fallback enters them
    as one comma-separated line and resolves token by token."""

    def build() -> tuple[cards.Bird, ...]:
        if con.supports_interactive():
            picked: list[cards.Bird] = []
            for slot in range(state.STARTING_HAND_SIZE):
                bird = identify_bird(
                    con,
                    f"Dealt bird {slot + 1} of {state.STARTING_HAND_SIZE}:",
                    exclude=picked,
                )
                picked.append(bird)
            return tuple(picked)
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
    """Your ``STARTING_BONUS_CARDS_DEAL`` dealt bonus cards, one at a time.

    On an interactive console, each pick excludes the bonus cards already
    picked this deal."""

    def build() -> tuple[cards.BonusCard, ...]:
        if con.supports_interactive():
            picked: list[cards.BonusCard] = []
            for slot in range(state.STARTING_BONUS_CARDS_DEAL):
                bonus_card = identify_bonus(
                    con,
                    f"Enter dealt bonus card {slot + 1} of "
                    f"{state.STARTING_BONUS_CARDS_DEAL}: ",
                    exclude=picked,
                )
                picked.append(bonus_card)
            return tuple(picked)
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
    """The four end-of-round goal tiles, one per round.

    On an interactive console, one typeahead pick per round -- goals have no
    ``identify_*`` helper, so this drives ``widgets.typeahead_pick`` directly
    -- each excluding the goals already picked, since a physical deal never
    repeats a tile face. The text-mode fallback resolves free-text answers
    via :func:`_resolve_goal`."""

    def build() -> tuple[cards.EndRoundGoal, ...]:
        if con.supports_interactive():
            _, _, catalog_goals = cards.load_all()
            picked: list[cards.EndRoundGoal] = []

            def find(query: str) -> list[cards.EndRoundGoal]:
                return [goal for goal in lookup.find_goals(query) if goal not in picked]

            def render(goal: cards.EndRoundGoal) -> str:
                return goal.description

            for round_num in range(_N_ROUNDS):
                initial = [goal for goal in catalog_goals if goal not in picked]
                chosen = widgets.typeahead_pick(
                    con,
                    f"Enter round {round_num + 1}'s goal: ",
                    find,
                    render,
                    initial=initial,
                )
                assert chosen is not None  # allow_blank defaults to False
                picked.append(chosen)
            return tuple(picked)
        return tuple(
            _resolve_goal(con, con.ask(f"Enter round {round_num + 1}'s goal: "))
            for round_num in range(_N_ROUNDS)
        )

    def echo(goals: tuple[cards.EndRoundGoal, ...]) -> None:
        for round_num, goal in enumerate(goals):
            con.say(f"Round {round_num + 1}: {goal.description}")

    return _run_until_confirmed(con, build, echo)


def _collect_tray(
    con: console_module.Console, hand: tuple[cards.Bird, ...]
) -> tuple[cards.Bird, ...]:
    """The initial face-up tray, left to right.

    ``hand`` is the already-entered starting hand. On an interactive console
    each pick's exclusion set is ``hand`` plus the tray picks so far, since
    the tray can never repeat a card already seen in hand; the text-mode
    fallback ignores ``hand`` entirely (unchanged from before)."""

    def build() -> tuple[cards.Bird, ...]:
        if con.supports_interactive():
            picked: list[cards.Bird] = []
            for slot in range(state.TRAY_SIZE):
                bird = identify_bird(
                    con,
                    f"Enter tray slot {slot + 1} of {state.TRAY_SIZE} "
                    "(left to right): ",
                    exclude=list(hand) + picked,
                )
                picked.append(bird)
            return tuple(picked)
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
    """The initial birdfeeder roll.

    On an interactive console, one counts-widget entry across the five food
    fields plus the choice-face field. The text-mode fallback loops on the
    shared die-face grammar (:func:`wingspan.aid.oracle.parse_die_faces`),
    naming any unrecognized token in its retry message."""
    prompt = (
        f"Enter all {state.BIRDFEEDER_DICE} birdfeeder die faces, "
        'space-separated (food name/alias, or "choice"): '
    )
    labels = [food.value for food in cards.ALL_FOODS] + [
        oracle_module.CHOICE_FACE_TOKEN
    ]

    def build() -> models.FeederEntry:
        if con.supports_interactive():
            values = widgets.counts_entry(con, prompt, labels, state.BIRDFEEDER_DICE)
            return models.FeederEntry(
                counts=values[: cards.N_FOODS], choice_dice=values[cards.N_FOODS]
            )
        while True:
            answer = con.ask(prompt)
            parsed = oracle_module.parse_die_faces(answer, state.BIRDFEEDER_DICE)
            if parsed is not None:
                counts, choice_dice = parsed
                return models.FeederEntry(
                    counts=list(counts.counts), choice_dice=choice_dice
                )
            con.say(oracle_module.die_face_retry_message(answer, state.BIRDFEEDER_DICE))

    def echo(feeder: models.FeederEntry) -> None:
        con.say(
            f"Feeder: {display.format_food_pool(feeder.to_food_pool())}, "
            f"choice dice: {feeder.choice_dice}"
        )

    return _run_until_confirmed(con, build, echo)
