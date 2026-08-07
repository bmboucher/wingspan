"""The interactive oracle: the live authority for every hidden or random
game outcome during an aid session.

``SessionOracle`` sits between the engine's card-reveal/dice-roll seams
(``GameState.draw_bird``/``draw_bonus``, ``Birdfeeder.reroll``/
``roll_out_of_feeder`` -- see ``oracle_state.py``) and the human at the
physical table: every reveal either drains a pre-queued setup fact or
prompts interactively, resolving free-text answers through
``wingspan.cards.lookup``.
"""

from __future__ import annotations

import collections.abc
import typing

from wingspan import cards, state
from wingspan.aid import console as console_module
from wingspan.aid import models, placeholders
from wingspan.cards import lookup

# The literal token a die-face entry uses for the invertebrate/seed choice
# face, in place of a food name/alias.
_CHOICE_FACE_TOKEN = "choice"


class SessionOracle:
    """Resolves every hidden/random outcome for one aid session: pre-queued
    setup facts are drained silently; anything unqueued prompts the user
    interactively through ``console``, flushing the engine's log first so
    the prompt always follows the narration it depends on."""

    def __init__(
        self,
        console: console_module.Console,
        echo: console_module.LogEcho,
        registry: placeholders.PlaceholderRegistry,
    ) -> None:
        self.console = console
        self.echo = echo
        self.registry = registry
        self.bird_queue: collections.deque[cards.Bird | None] = collections.deque()
        self.bonus_queue: collections.deque[cards.BonusCard | None] = (
            collections.deque()
        )
        self.pending_feeder: models.FeederEntry | None = None
        # Backref attached by oracle_state.build_state once the state
        # exists; typed as the base GameState (not OracleGameState) to
        # avoid a circular import between this module and oracle_state.
        self.game_state: state.GameState | None = None

    # ------------------------------------------------------------------
    # Queueing (setup deals; consumed silently, in order)
    # ------------------------------------------------------------------

    def queue_bird_reveals(
        self, items: collections.abc.Iterable[cards.Bird | None]
    ) -> None:
        """Pre-seed upcoming :meth:`reveal_bird` calls. Each ``None`` mints
        a silent placeholder in its turn; a real ``Bird`` is returned as
        entered."""
        self.bird_queue.extend(items)

    def queue_bonus_reveals(
        self, items: collections.abc.Iterable[cards.BonusCard | None]
    ) -> None:
        """Pre-seed upcoming :meth:`reveal_bonus` calls, mirroring
        :meth:`queue_bird_reveals`."""
        self.bonus_queue.extend(items)

    def queue_feeder_roll(self, entry: models.FeederEntry) -> None:
        """Pre-seed the next :meth:`feeder_roll` call with an already-known
        entry (the setup deal's initial feeder roll)."""
        self.pending_feeder = entry

    # ------------------------------------------------------------------
    # Resolution (queued facts, else interactive prompts)
    # ------------------------------------------------------------------

    def reveal_bird(self) -> cards.Bird:
        """The next hidden/drawn bird: a queued reveal if one is pending,
        else an interactive prompt (Enter = face-down/unknown -> a
        placeholder)."""
        if self.bird_queue:
            queued = self.bird_queue.popleft()
            return self.registry.mint_bird() if queued is None else queued
        self.echo.flush()
        while True:
            answer = self.console.ask(
                "A bird card was just drawn or revealed — enter its name "
                "(Enter = face-down/unknown): "
            )
            if not answer:
                return self.registry.mint_bird()
            resolved = self._resolve_candidates(answer, lookup.find_birds(answer))
            if resolved is not None:
                return resolved

    def reveal_bonus(self) -> cards.BonusCard:
        """The next hidden/drawn bonus card, mirroring :meth:`reveal_bird`
        via :func:`wingspan.cards.lookup.find_bonus_cards`."""
        if self.bonus_queue:
            queued = self.bonus_queue.popleft()
            return self.registry.mint_bonus() if queued is None else queued
        self.echo.flush()
        while True:
            answer = self.console.ask(
                "A bonus card was just drawn or revealed — enter its name "
                "(Enter = face-down/unknown): "
            )
            if not answer:
                return self.registry.mint_bonus()
            resolved = self._resolve_candidates(answer, lookup.find_bonus_cards(answer))
            if resolved is not None:
                return resolved

    def feeder_roll(self) -> models.FeederEntry:
        """The birdfeeder's current ``BIRDFEEDER_DICE`` faces: the queued
        setup roll if pending, else an interactive prompt for all five
        physical dice."""
        if self.pending_feeder is not None:
            entry = self.pending_feeder
            self.pending_feeder = None
            return entry
        self.echo.flush()
        prompt = (
            f"The birdfeeder was rerolled — enter all {state.BIRDFEEDER_DICE} "
            'die faces, space-separated (food name/alias, or "choice"): '
        )
        while True:
            parsed = parse_die_faces(self.console.ask(prompt), state.BIRDFEEDER_DICE)
            if parsed is not None:
                counts, choice_dice = parsed
                return models.FeederEntry(
                    counts=list(counts.counts), choice_dice=choice_dice
                )
            self.console.say(
                f"Enter exactly {state.BIRDFEEDER_DICE} faces "
                '(food name/alias, or "choice"), separated by spaces.'
            )

    def dice_roll(self, n: int) -> tuple[state.FoodPool, int]:
        """``n`` dice rolled outside the feeder (a dice-predator power):
        always interactive -- public information regardless of seat, so
        never pre-queued."""
        self.echo.flush()
        prompt = (
            f"{n} dice were rolled outside the feeder — what did they show? "
            f'Enter {n} faces, space-separated (food name/alias, or "choice"): '
        )
        while True:
            parsed = parse_die_faces(self.console.ask(prompt), n)
            if parsed is not None:
                return parsed
            self.console.say(
                f'Enter exactly {n} faces (food name/alias, or "choice"), '
                "separated by spaces."
            )

    def _resolve_candidates[T: _NamedCard](
        self, query: str, matches: list[T]
    ) -> T | None:
        """Shared no-match/single-match/ambiguous handling for
        :meth:`reveal_bird`/:meth:`reveal_bonus`: ``None`` means "reprompt"
        (no match), a sole match returns directly, and multiple matches open
        a disambiguation menu."""
        if not matches:
            self.console.say(f'No match for "{query}" — try again.')
            return None
        if len(matches) == 1:
            return matches[0]
        chosen_idx = self.console.menu(
            f'Multiple matches for "{query}" — which one?',
            [match.name for match in matches],
        )
        return matches[chosen_idx]


def parse_die_faces(
    text: str, expected_count: int
) -> tuple[state.FoodPool, int] | None:
    """Parse a space-separated die-face entry into ``(single_face_counts,
    choice_face_count)``, or ``None`` if the token count is wrong or any
    token fails to resolve to a food name/alias or the literal
    :data:`_CHOICE_FACE_TOKEN`.

    Public (rather than a private helper of :meth:`SessionOracle.feeder_roll`
    / :meth:`SessionOracle.dice_roll`) so ``entry.run_setup_entry`` can reuse
    the same die-face grammar for the initial feeder-roll entry."""
    tokens = text.split()
    if len(tokens) != expected_count:
        return None
    counts = state.FoodPool()
    choice_count = 0
    for token in tokens:
        if token.casefold() == _CHOICE_FACE_TOKEN:
            choice_count += 1
            continue
        food = lookup.find_food(token)
        if food is None:
            return None
        counts[food] += 1
    return counts, choice_count


###### PRIVATE #######


class _NamedCard(typing.Protocol):
    """Structural bound for :meth:`SessionOracle._resolve_candidates`:
    ``cards.Bird`` and ``cards.BonusCard`` both expose a plain ``name``
    field used for the disambiguation menu."""

    name: str
