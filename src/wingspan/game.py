"""Game engine.

Drives the game forward by repeatedly asking the active player's agent to
resolve a Decision. ``Engine.play_one_game(agents)`` is the top-level entry
point; ``Engine.step(choice)`` is the lower-level interface for cases where
the caller wants to inspect state between decisions (e.g. RL self-play).
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from .actions import Choice, Decision, DecisionType, MainAction
from .cards import (
    ALL_FOODS, ALL_HABITATS, Bird, BonusCard, Effect, EffectKind, EndRoundGoal,
    Food, Habitat, NestType, PowerColor, power_coverage,
)
from .state import (
    GameState, PlayedBird, Player, ROW_SLOTS, ROUND_CUBES, STARTING_HAND_SIZE,
    STARTING_FOOD_TOTAL, STARTING_BONUS_CARDS_DEAL, TRAY_SIZE, new_game,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper utilities

def _row_activation_count(player: Player, habitat: Habitat) -> int:
    """How many birds are in the row (used to determine how many times the
    column power triggers, capped at row length)."""
    return len(player.board[habitat])


def _row_food_reward(habitat: Habitat, column: int) -> tuple[Food, int]:
    """Wingspan's row tracks have varying rewards. We approximate the printed
    tracks (food bonus per column when taking the Gain Food action)."""
    # Real reward tracks (core), simplified:
    #   Forest:    take 1, 1, 2, 2, 2 dice (and rolling-extras after)
    #   Grassland: lay 2, 2, 3, 3, 4 eggs
    #   Wetland:   draw 1, 1, 2, 2, 3 cards (+1 between turns)
    # For column powers the reward count is the number of birds in row (1..5).
    return Food.INVERTEBRATE, 1  # unused; we use the count-based helpers below


def _gain_food_count(num_birds: int) -> int:
    """Number of birdfeeder picks for the forest action with given filled cols."""
    return [1, 1, 2, 2, 2][min(num_birds, 4)] + max(0, num_birds - 5 + 0)


def _lay_eggs_count(num_birds: int) -> int:
    return [2, 2, 3, 3, 4][min(num_birds, 4)] + max(0, num_birds - 5 + 0)


def _draw_cards_count(num_birds: int) -> int:
    return [1, 1, 2, 2, 3][min(num_birds, 4)] + max(0, num_birds - 5 + 0)


def _cost_meets(pb_cost: dict[Food, int], wild: int, payment: dict[Food, int]) -> bool:
    """Check that ``payment`` is a legal way to pay ``pb_cost`` + ``wild`` wild."""
    remaining = dict(pb_cost)
    extra = 0
    for f, n in payment.items():
        need = remaining.get(f, 0)
        used = min(need, n)
        remaining[f] = need - used
        extra += n - used
    if any(v > 0 for v in remaining.values()):
        return False
    return extra >= wild and (extra + sum(remaining.values())) == wild  # exact wild


def _enumerate_payments(food: dict[Food, int], cost: dict[Food, int], wild: int) -> list[dict[Food, int]]:
    """Enumerate distinct (food-type-multiset) payments for a non-flocking bird.

    We pay specific costs first, then wild from any remaining food. To keep the
    branching factor manageable we group wild payments by the multiset of
    foods used. This is a small combinatorial enumeration suitable for the
    tiny supplies typical of Wingspan turns."""
    # First check we have enough specific foods.
    for f, n in cost.items():
        if food.get(f, 0) < n:
            return []
    remaining = {f: food[f] - cost.get(f, 0) for f in ALL_FOODS}
    if wild == 0:
        return [dict(cost)]

    results: list[dict[Food, int]] = []
    foods = ALL_FOODS

    def rec(idx: int, left: int, pay: dict[Food, int]) -> None:
        if left == 0:
            full = {f: cost.get(f, 0) + pay.get(f, 0) for f in foods}
            full = {f: v for f, v in full.items() if v > 0}
            if full not in results:
                results.append(full)
            return
        if idx == len(foods):
            return
        f = foods[idx]
        max_take = min(left, remaining[f])
        for k in range(0, max_take + 1):
            if k > 0:
                pay[f] = pay.get(f, 0) + k
            rec(idx + 1, left - k, pay)
            if k > 0:
                pay[f] -= k
                if pay[f] == 0:
                    del pay[f]

    rec(0, wild, {})
    return results


def _bird_matches_habitat(bird: Bird, habitat: Habitat) -> bool:
    return habitat in bird.habitats


# ---------------------------------------------------------------------------
# Engine

Agent = Callable[["Engine", Decision], Choice]


@dataclass
class TurnState:
    """Per-turn scratch space."""
    action: Optional[MainAction] = None
    play_habitat: Optional[Habitat] = None
    play_card: Optional[Bird] = None
    play_paid_eggs: int = 0
    extra_plays: int = 0
    skip_remaining_activations: bool = False


class Engine:
    def __init__(self, state: GameState):
        self.state = state
        self.turn_state = TurnState()

    # ------------------------------------------------------------------
    # Top-level driver
    # ------------------------------------------------------------------
    def play_one_game(self, agents: tuple[Agent, Agent]) -> None:
        self._log("=== Wingspan game start ===")
        self._setup_phase(agents)
        for r in range(4):
            self.state.round_idx = r
            for p in self.state.players:
                p.action_cubes_left = ROUND_CUBES[r]
                for row in p.board.values():
                    for pb in row:
                        pb.activations = 0
            self._log(f"--- Round {r+1} (each player gets {ROUND_CUBES[r]} actions) ---")
            self._log(f"Round goal: {self.state.round_goals[r].description} ({self.state.round_goals[r].category})")
            # Alternate turns; first player rotates each round
            first = r % 2
            while any(p.action_cubes_left > 0 for p in self.state.players):
                self.state.current_player = first
                if self.state.players[first].action_cubes_left > 0:
                    self._take_turn(agents[first])
                first = 1 - first
                if self.state.players[first].action_cubes_left > 0:
                    self._take_turn(agents[first])
                # If both passed mid-round we'd exit, but cubes ensure progress.
            self._score_round_goal(r)
        self._final_scoring()
        self.state.game_over = True
        self._log("=== Wingspan game end ===")

    def _take_turn(self, agent: Agent):
        self.state.current_player = self.state.current_player  # unchanged
        p = self.state.me()
        self.turn_state = TurnState()
        self._log(f"[{p.name}] starts turn ({p.action_cubes_left} cubes left)")
        choice = self._ask(agent, self._main_action_decision(p))
        action: MainAction = choice.payload
        self.turn_state.action = action
        p.action_cubes_left -= 1
        if action == MainAction.PLAY_BIRD:
            self._do_play_bird(agent)
        elif action == MainAction.GAIN_FOOD:
            self._do_gain_food(agent, Habitat.FOREST)
        elif action == MainAction.LAY_EGGS:
            self._do_lay_eggs(agent, Habitat.GRASSLAND)
        elif action == MainAction.DRAW_CARDS:
            self._do_draw_cards(agent, Habitat.WETLAND)
        # extra plays granted by activated powers
        while self.turn_state.extra_plays > 0:
            self.turn_state.extra_plays -= 1
            if not self._can_play_bird(p):
                break
            self._log(f"[{p.name}] takes an EXTRA play")
            self._do_play_bird(agent)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup_phase(self, agents):
        st = self.state
        for p in st.players:
            # 5 bird cards
            for _ in range(STARTING_HAND_SIZE):
                b = st.draw_bird()
                if b: p.hand.append(b)
            # 2 bonus, keep 1
            picks = []
            for _ in range(STARTING_BONUS_CARDS_DEAL):
                if st.bonus_deck:
                    picks.append(st.bonus_deck.pop())
            if picks:
                st.current_player = p.id
                ch = self._ask(agents[p.id], Decision(
                    type=DecisionType.SETUP_PICK_BONUS,
                    player_id=p.id,
                    prompt=f"[{p.name}] choose a bonus card to keep",
                    choices=[Choice(label=b.name, payload=b) for b in picks],
                ))
                kept: BonusCard = ch.payload
                p.bonus_cards.append(kept)
                for b in picks:
                    if b is not kept:
                        st.bonus_discard.append(b)
                self._log(f"[{p.name}] keeps bonus '{kept.name}'")
            # 5 food (one of each as starting)
            for f in ALL_FOODS:
                p.food[f] = 1  # 5 total food
            # Decide which cards to keep (each kept card = 1 food discarded)
            # Simpler: ask each card to keep-or-discard
            kept_cards: list[Bird] = []
            food_to_discard = 0
            for card in list(p.hand):
                st.current_player = p.id
                ch = self._ask(agents[p.id], Decision(
                    type=DecisionType.SETUP_KEEP_FOOD_OR_DISCARD_CARD,
                    player_id=p.id,
                    prompt=f"[{p.name}] keep {card.name}? (costs 1 food)",
                    choices=[
                        Choice(label=f"keep {card.name}", payload=("keep", card)),
                        Choice(label=f"discard {card.name}", payload=("discard", card)),
                    ],
                ))
                if ch.payload[0] == "keep":
                    kept_cards.append(card)
                    food_to_discard += 1
                else:
                    st.bird_discard.append(card)
            p.hand = kept_cards
            # Discard 1 food per kept card (player picks which)
            for _ in range(food_to_discard):
                available = [f for f, n in p.food.items() if n > 0]
                if not available:
                    break
                st.current_player = p.id
                ch = self._ask(agents[p.id], Decision(
                    type=DecisionType.BIRD_POWER_PICK_FOOD,
                    player_id=p.id,
                    prompt=f"[{p.name}] discard 1 food to pay for kept card",
                    choices=[Choice(label=f.value, payload=f) for f in available],
                    context={"reason": "setup_pay"},
                ))
                p.food[ch.payload] -= 1
            self._log(f"[{p.name}] starts with hand={[b.name for b in p.hand]} food={dict(p.food)}")

    # ------------------------------------------------------------------
    # Action: play a bird
    # ------------------------------------------------------------------
    def _can_play_bird(self, p: Player) -> bool:
        if not p.hand: return False
        if not any(p.can_play_in(h) for h in ALL_HABITATS): return False
        # at least one bird in hand must be affordable in some habitat
        for b in p.hand:
            for h in b.habitats:
                if not p.can_play_in(h): continue
                if not _enumerate_payments(p.food, b.food_cost, b.wild_food_cost):
                    continue
                egg_cost = self._egg_cost(p, h)
                if p.total_eggs < egg_cost:
                    continue
                return True
        return False

    def _egg_cost(self, p: Player, habitat: Habitat) -> int:
        # Standard egg cost ladder by column: 0,1,1,2,2 (Forest), 1,1,2,2,3 (Grass), 0,2,2,3,3 (Wet)
        col = len(p.board[habitat])
        ladders = {
            Habitat.FOREST:    [0, 1, 1, 2, 2],
            Habitat.GRASSLAND: [1, 1, 2, 2, 3],
            Habitat.WETLAND:   [0, 2, 2, 3, 3],
        }
        if col >= ROW_SLOTS: return 99
        return ladders[habitat][col]

    def _do_play_bird(self, agent: Agent):
        p = self.state.me()
        # pick card
        playable: list[Bird] = []
        for b in p.hand:
            if any(
                p.can_play_in(h)
                and _enumerate_payments(p.food, b.food_cost, b.wild_food_cost)
                and p.total_eggs >= self._egg_cost(p, h)
                for h in b.habitats
            ):
                playable.append(b)
        if not playable:
            self._log(f"[{p.name}] tried to play a bird but had no playable bird; action wasted")
            return
        ch = self._ask(agent, Decision(
            type=DecisionType.PLAY_BIRD_PICK_CARD,
            player_id=p.id,
            prompt=f"[{p.name}] pick a bird to play",
            choices=[Choice(label=b.name, payload=b) for b in playable],
        ))
        card: Bird = ch.payload
        habs = [h for h in card.habitats if p.can_play_in(h)]
        if len(habs) == 1:
            habitat = habs[0]
        else:
            ch = self._ask(agent, Decision(
                type=DecisionType.PLAY_BIRD_PICK_HABITAT,
                player_id=p.id,
                prompt=f"[{p.name}] pick habitat for {card.name}",
                choices=[Choice(label=h.value, payload=h) for h in habs],
            ))
            habitat = ch.payload
        # pay egg cost
        egg_cost = self._egg_cost(p, habitat)
        for _ in range(egg_cost):
            self._discard_an_egg(agent, p, reason=f"play {card.name}")
        # pay food cost
        payments = _enumerate_payments(p.food, card.food_cost, card.wild_food_cost)
        if not payments:
            self._log(f"[{p.name}] unable to pay for {card.name} (bug or shortage); wasting action")
            return
        if len(payments) == 1:
            payment = payments[0]
        else:
            ch = self._ask(agent, Decision(
                type=DecisionType.PLAY_BIRD_PICK_FOOD_PAYMENT,
                player_id=p.id,
                prompt=f"[{p.name}] pick food payment for {card.name}",
                choices=[Choice(label=", ".join(f"{n}{f.value}" for f, n in pay.items()), payload=pay)
                         for pay in payments],
            ))
            payment = ch.payload
        for f, n in payment.items():
            p.food[f] -= n
        p.hand.remove(card)
        pb = PlayedBird(bird=card)
        p.board[habitat].append(pb)
        self._log(f"[{p.name}] plays {card.name} into {habitat.value} (paid {payment}, {egg_cost} eggs)")
        # WHITE power triggers when played
        if card.color == PowerColor.WHITE:
            self._dispatch_power(agent, p, pb, habitat, "play")

    def _discard_an_egg(self, agent: Agent, p: Player, reason: str):
        choices = []
        for h, row in p.board.items():
            for i, pb in enumerate(row):
                if pb.eggs > 0:
                    choices.append(Choice(label=f"{pb.bird.name}@{h.value}[{i}]", payload=(h, i)))
        if not choices:
            return
        ch = self._ask(agent, Decision(
            type=DecisionType.PLAY_BIRD_PICK_EGG_TO_PAY,
            player_id=p.id,
            prompt=f"[{p.name}] discard an egg ({reason})",
            choices=choices,
        ))
        h, i = ch.payload
        p.board[h][i].eggs -= 1

    # ------------------------------------------------------------------
    # Action: gain food (Forest)
    # ------------------------------------------------------------------
    def _do_gain_food(self, agent: Agent, habitat: Habitat):
        p = self.state.me()
        n_birds = _row_activation_count(p, habitat)
        # column reward
        n_dice = _gain_food_count(n_birds)
        self._log(f"[{p.name}] gain food: row has {n_birds} birds, take {n_dice} dice")
        for _ in range(n_dice):
            avail = [(f, c) for f, c in self.state.birdfeeder.counts.items() if c > 0]
            if not avail:
                self.state.birdfeeder.reroll(self.state.rng)
                self._log(f"  birdfeeder empty; rerolled to {dict(self.state.birdfeeder.counts)}")
                avail = [(f, c) for f, c in self.state.birdfeeder.counts.items() if c > 0]
                if not avail: break
            ch = self._ask(agent, Decision(
                type=DecisionType.GAIN_FOOD_PICK_DIE,
                player_id=p.id,
                prompt=f"[{p.name}] take 1 die from birdfeeder",
                choices=[Choice(label=f"{f.value}({c})", payload=f) for f, c in avail],
            ))
            f = ch.payload
            self.state.birdfeeder.counts[f] -= 1
            p.food[f] += 1
            self._log(f"  +1 {f.value}")
        # Reroll if 1 or fewer faces showing
        types_left = sum(1 for c in self.state.birdfeeder.counts.values() if c > 0)
        if types_left <= 1 and self.state.birdfeeder.total() > 0:
            self.state.birdfeeder.reroll(self.state.rng)
        self._activate_row_powers(agent, p, habitat)

    # ------------------------------------------------------------------
    # Action: lay eggs (Grassland)
    # ------------------------------------------------------------------
    def _do_lay_eggs(self, agent: Agent, habitat: Habitat):
        p = self.state.me()
        n_birds = _row_activation_count(p, habitat)
        n_eggs = _lay_eggs_count(n_birds)
        self._log(f"[{p.name}] lay eggs: row has {n_birds} birds, lay {n_eggs} eggs")
        for _ in range(n_eggs):
            self._lay_one_egg(agent, p)
        self._activate_row_powers(agent, p, habitat)

    def _lay_one_egg(self, agent: Agent, p: Player):
        choices = []
        for h, row in p.board.items():
            for i, pb in enumerate(row):
                if pb.eggs < pb.bird.egg_limit:
                    choices.append(Choice(label=f"{pb.bird.name}@{h.value}[{i}]({pb.eggs}/{pb.bird.egg_limit})", payload=(h, i)))
        if not choices:
            return
        ch = self._ask(agent, Decision(
            type=DecisionType.LAY_EGG_PICK_BIRD,
            player_id=p.id,
            prompt=f"[{p.name}] lay 1 egg",
            choices=choices,
        ))
        h, i = ch.payload
        p.board[h][i].eggs += 1

    # ------------------------------------------------------------------
    # Action: draw cards (Wetland)
    # ------------------------------------------------------------------
    def _do_draw_cards(self, agent: Agent, habitat: Habitat):
        p = self.state.me()
        n_birds = _row_activation_count(p, habitat)
        n_cards = _draw_cards_count(n_birds)
        self._log(f"[{p.name}] draw cards: row has {n_birds} birds, draw {n_cards}")
        for _ in range(n_cards):
            self._draw_one_card(agent, p)
        self._activate_row_powers(agent, p, habitat)

    def _draw_one_card(self, agent: Agent, p: Player):
        # Choose tray slot or top of deck
        choices = []
        for i, b in enumerate(self.state.tray):
            choices.append(Choice(label=f"tray[{i}]={b.name}", payload=("tray", i)))
        if self.state.bird_deck or self.state.bird_discard:
            choices.append(Choice(label="deck", payload=("deck", None)))
        if not choices:
            return
        ch = self._ask(agent, Decision(
            type=DecisionType.DRAW_CARDS_PICK_SOURCE,
            player_id=p.id,
            prompt=f"[{p.name}] draw 1 card",
            choices=choices,
        ))
        src = ch.payload
        if src[0] == "tray":
            b = self.state.tray.pop(src[1])
            self.state.refill_tray()
            p.hand.append(b)
        else:
            b = self.state.draw_bird()
            if b: p.hand.append(b)

    # ------------------------------------------------------------------
    # Power activation
    # ------------------------------------------------------------------
    def _activate_row_powers(self, agent: Agent, p: Player, habitat: Habitat):
        # BROWN powers activate right to left in the activated row.
        for pb in reversed(p.board[habitat]):
            if pb.bird.color != PowerColor.BROWN:
                continue
            pb.activations += 1
            self._dispatch_power(agent, p, pb, habitat, "activate")

    def _dispatch_power(self, agent: Agent, p: Player, pb: PlayedBird, habitat: Habitat, trigger: str):
        bird = pb.bird
        for eff in bird.power.effects:
            self._apply_effect(agent, p, pb, habitat, eff, trigger)

    def _apply_effect(self, agent: Agent, p: Player, pb: PlayedBird, habitat: Habitat, eff: Effect, trigger: str):
        st = self.state
        bird = pb.bird
        if eff.kind == EffectKind.UNIMPLEMENTED:
            self._log(f"  (power on {bird.name} not modeled: {bird.raw_power_text!r}; skipped)")
            return
        if eff.kind == EffectKind.GAIN_FOOD_SUPPLY:
            if eff.food and st.food_supply.get(eff.food, 0) >= eff.amount:
                st.food_supply[eff.food] -= eff.amount
                p.food[eff.food] += eff.amount
                self._log(f"  {bird.name}: +{eff.amount} {eff.food.value} from supply")
        elif eff.kind == EffectKind.GAIN_FOOD_BIRDFEEDER:
            if eff.food and st.birdfeeder.counts.get(eff.food, 0) > 0:
                take = min(eff.amount, st.birdfeeder.counts[eff.food])
                st.birdfeeder.counts[eff.food] -= take
                p.food[eff.food] += take
                self._log(f"  {bird.name}: +{take} {eff.food.value} from birdfeeder")
        elif eff.kind == EffectKind.LAY_EGG_ON_THIS:
            cap = bird.egg_limit - pb.eggs
            n = min(eff.amount, cap)
            pb.eggs += n
            if n: self._log(f"  {bird.name}: +{n} egg on itself")
        elif eff.kind == EffectKind.LAY_EGG_ANY:
            for _ in range(eff.amount):
                self._lay_one_egg(agent, p)
        elif eff.kind == EffectKind.DRAW_CARDS:
            for _ in range(eff.amount):
                self._draw_one_card(agent, p)
        elif eff.kind == EffectKind.CACHE_FOOD:
            if eff.food and st.food_supply.get(eff.food, 0) >= eff.amount:
                st.food_supply[eff.food] -= eff.amount
                pb.cached_food += eff.amount
                self._log(f"  {bird.name}: cached {eff.amount} {eff.food.value}")
        elif eff.kind == EffectKind.TUCK_FROM_HAND:
            for _ in range(eff.amount):
                if not p.hand: break
                ch = self._ask(agent, Decision(
                    type=DecisionType.BIRD_POWER_TUCK_FROM_HAND,
                    player_id=p.id,
                    prompt=f"[{p.name}] tuck 1 card behind {bird.name} (or skip)",
                    choices=[Choice(label=c.name, payload=c) for c in p.hand]
                            + [Choice(label="skip", payload=None)],
                ))
                if ch.payload is None: break
                p.hand.remove(ch.payload)
                pb.tucked_cards += 1
                self._log(f"  {bird.name}: tucked {ch.payload.name}")
        elif eff.kind == EffectKind.PLAY_ADDITIONAL_BIRD:
            if not eff.habitat or eff.habitat == habitat:
                self.turn_state.extra_plays += 1
                self._log(f"  {bird.name}: granted +1 extra play")
        elif eff.kind == EffectKind.ALL_PLAYERS_GAIN_FOOD:
            if eff.food:
                for q in st.players:
                    if st.food_supply.get(eff.food, 0) >= eff.amount:
                        st.food_supply[eff.food] -= eff.amount
                        q.food[eff.food] += eff.amount
                self._log(f"  {bird.name}: all players +{eff.amount} {eff.food.value}")
        elif eff.kind == EffectKind.ALL_PLAYERS_DRAW:
            for q in st.players:
                for _ in range(eff.amount):
                    self._draw_one_card(agent, q)
        elif eff.kind == EffectKind.DRAW_BONUS:
            drawn = []
            for _ in range(eff.amount):
                if st.bonus_deck:
                    drawn.append(st.bonus_deck.pop())
            p.bonus_cards.extend(drawn)
            self._log(f"  {bird.name}: drew {len(drawn)} bonus card(s)")
        elif eff.kind == EffectKind.DRAW_BONUS_KEEP_ONE:
            keep = eff.extra[0] if eff.extra else 1
            drawn: list[BonusCard] = []
            for _ in range(eff.amount):
                if not st.bonus_deck:
                    if st.bonus_discard:
                        st.bonus_deck = st.bonus_discard
                        st.bonus_discard = []
                        st.rng.shuffle(st.bonus_deck)
                    else:
                        break
                drawn.append(st.bonus_deck.pop())
            if not drawn:
                self._log(f"  {bird.name}: bonus deck empty; no cards drawn")
                return
            keep_n = min(keep, len(drawn))
            for _ in range(keep_n):
                ch = self._ask(agent, Decision(
                    type=DecisionType.BIRD_POWER_PICK_BONUS_TO_KEEP,
                    player_id=p.id,
                    prompt=f"[{p.name}] keep 1 bonus card (from {bird.name})",
                    choices=[Choice(label=b.name, payload=b) for b in drawn],
                ))
                kept: BonusCard = ch.payload
                p.bonus_cards.append(kept)
                drawn.remove(kept)
            for b in drawn:
                st.bonus_discard.append(b)
            self._log(
                f"  {bird.name}: drew {keep_n + len(drawn)} bonus, kept {keep_n}, discarded {len(drawn)}"
            )

    # ------------------------------------------------------------------
    # Decision plumbing
    # ------------------------------------------------------------------
    def _main_action_decision(self, p: Player) -> Decision:
        choices = []
        if self._can_play_bird(p):
            choices.append(Choice(label="play bird", payload=MainAction.PLAY_BIRD, encoded=0))
        # the other three are always legal (will just be inefficient if board is empty)
        choices.append(Choice(label="gain food (forest)", payload=MainAction.GAIN_FOOD, encoded=1))
        choices.append(Choice(label="lay eggs (grassland)", payload=MainAction.LAY_EGGS, encoded=2))
        choices.append(Choice(label="draw cards (wetland)", payload=MainAction.DRAW_CARDS, encoded=3))
        return Decision(
            type=DecisionType.MAIN_ACTION,
            player_id=p.id,
            prompt=f"[{p.name}] choose a main action",
            choices=choices,
        )

    def _ask(self, agent: Agent, d: Decision) -> Choice:
        choice = agent(self, d)
        if not isinstance(choice, Choice):
            # tolerate raw index from CLI
            try:
                choice = d.choices[int(choice)]
            except Exception as ex:
                raise ValueError(f"agent {agent} returned non-Choice {choice!r}: {ex}")
        if choice not in d.choices:
            # fallback - find by label
            for c in d.choices:
                if c.label == choice.label and c.payload == choice.payload:
                    choice = c
                    break
            else:
                raise ValueError(f"agent returned illegal choice {choice.label!r} for {d.type}")
        return choice

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def _score_round_goal(self, r: int):
        goal = self.state.round_goals[r]
        scores = [self._eval_goal(p, goal) for p in self.state.players]
        self._log(f"Round {r+1} goal '{goal.category}' scores: {scores}")
        # 2P payouts: 1st gets 5, 2nd gets 2, ties share/sum (Wingspan: tied for 1st each get 5; nobody gets 2nd)
        first, second = goal.payouts_2p
        a, b = scores
        if a > b:
            self.state.players[0].round_goal_points += first
            self.state.players[1].round_goal_points += second
        elif b > a:
            self.state.players[1].round_goal_points += first
            self.state.players[0].round_goal_points += second
        else:
            self.state.players[0].round_goal_points += first
            self.state.players[1].round_goal_points += first

    def _eval_goal(self, p: Player, goal: EndRoundGoal) -> int:
        c = goal.category
        if c == "birds_forest":    return len(p.board[Habitat.FOREST])
        if c == "birds_grassland": return len(p.board[Habitat.GRASSLAND])
        if c == "birds_wetland":   return len(p.board[Habitat.WETLAND])
        if c == "eggs_forest":     return sum(pb.eggs for pb in p.board[Habitat.FOREST])
        if c == "eggs_grassland":  return sum(pb.eggs for pb in p.board[Habitat.GRASSLAND])
        if c == "eggs_wetland":    return sum(pb.eggs for pb in p.board[Habitat.WETLAND])
        if c == "eggs_bowl":       return sum(pb.eggs for r in p.board.values() for pb in r if pb.bird.nest == NestType.BOWL)
        if c == "eggs_cavity":     return sum(pb.eggs for r in p.board.values() for pb in r if pb.bird.nest == NestType.CAVITY)
        if c == "eggs_ground":     return sum(pb.eggs for r in p.board.values() for pb in r if pb.bird.nest == NestType.GROUND)
        if c == "eggs_platform":   return sum(pb.eggs for r in p.board.values() for pb in r if pb.bird.nest == NestType.PLATFORM)
        if c == "bowl_birds_with_eggs":     return sum(1 for r in p.board.values() for pb in r if pb.bird.nest == NestType.BOWL and pb.eggs > 0)
        if c == "cavity_birds_with_eggs":   return sum(1 for r in p.board.values() for pb in r if pb.bird.nest == NestType.CAVITY and pb.eggs > 0)
        if c == "ground_birds_with_eggs":   return sum(1 for r in p.board.values() for pb in r if pb.bird.nest == NestType.GROUND and pb.eggs > 0)
        if c == "platform_birds_with_eggs": return sum(1 for r in p.board.values() for pb in r if pb.bird.nest == NestType.PLATFORM and pb.eggs > 0)
        if c == "tucked_cards":             return p.total_tucked
        if c == "wingspan_under_30":        return sum(1 for r in p.board.values() for pb in r if pb.bird.wingspan_cm and pb.bird.wingspan_cm < 30)
        if c == "wingspan_over_65":         return sum(1 for r in p.board.values() for pb in r if pb.bird.wingspan_cm and pb.bird.wingspan_cm > 65)
        return 0

    def _final_scoring(self) -> None:
        for p in self.state.players:
            bird_pts = sum(pb.bird.points for r in p.board.values() for pb in r)
            bonus_pts = sum(self._bonus_score(p, bc) for bc in p.bonus_cards)
            eggs = p.total_eggs
            tucked = p.total_tucked
            cached = p.total_cached
            food_left = p.total_food()
            round_goal = p.round_goal_points
            total = bird_pts + bonus_pts + eggs + tucked + cached + round_goal
            self._log(
                f"[{p.name}] FINAL: birds={bird_pts} bonus={bonus_pts} eggs={eggs}"
                f" tucked={tucked} cached={cached} round_goal={round_goal} foodleft={food_left} -> {total}")
            # stash on player for the caller
            p_total = total
            setattr(p, "final_score", p_total)

    def _bonus_score(self, p: Player, bc: BonusCard) -> int:
        # Count qualifying birds: those that include this bonus card in their categories.
        count = sum(1 for r in p.board.values() for pb in r if bc.name in pb.bird.bonus_categories)
        # Find the highest threshold met
        best = 0
        for thr, vp in bc.thresholds:
            if count >= thr and vp > best:
                best = vp
        return best

    # ------------------------------------------------------------------
    # Log helper
    # ------------------------------------------------------------------
    def _log(self, msg: str) -> None:
        self.state.log.append(msg)


# ---------------------------------------------------------------------------
# Convenience

def make_engine(seed: int = 0) -> tuple[Engine, list[Bird], list[BonusCard], list[EndRoundGoal]]:
    from . import cards
    birds, bonuses, goals = cards.load_all()
    rng = random.Random(seed)
    state = new_game(rng, birds, bonuses, goals)
    return Engine(state), birds, bonuses, goals


def print_coverage_report():
    from . import cards
    birds, _, _ = cards.load_all()
    impl, total = power_coverage(birds)
    logger.info(f"Bird power coverage: {impl}/{total} ({impl*100//total}%) modelled")
