"""Implementations of Wingspan's four main actions and their direct helpers.

Each public function takes the live ``Engine`` as its first argument and
mutates the underlying ``GameState`` through it. The Engine class delegates
its public ``_do_play_bird`` / ``_do_gain_food`` / ``_do_lay_eggs`` /
``_do_draw_cards`` methods straight to the matching functions here.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine import helpers, powers, reactors

if typing.TYPE_CHECKING:
    from wingspan.engine import core


# ---------------------------------------------------------------------------
# Main action: play a bird


def can_play_bird(engine: "core.Engine", p: state.Player) -> bool:
    """True if ``p`` has at least one bird in hand that could legally be
    played into some habitat right now (affordable food, payable eggs)."""
    if not p.hand:
        return False
    if not any(p.can_play_in(h) for h in cards.ALL_HABITATS):
        return False
    for b in p.hand:
        for h in b.habitats:
            if not p.can_play_in(h):
                continue
            if not helpers.enumerate_payments(p.food, b.food_cost):
                continue
            if p.total_eggs < p.board.next_egg_cost(h):
                continue
            return True
    return False


def do_play_bird(engine: "core.Engine", agent: "core.Agent") -> None:
    """Run a Play Bird action for the current player."""
    p = engine.state.me()
    habitat_filter = engine.state.turn_extra_play_habitat
    playable = _playable_birds(p, habitat_filter)
    if not playable:
        _log_wasted_play(engine, p, habitat_filter)
        return

    card = _pick_card(engine, agent, p, playable)
    habitat = _pick_habitat(engine, agent, p, card, habitat_filter)

    # Pay egg + food costs in that order (matches printed action sequence).
    egg_cost = p.board.next_egg_cost(habitat)
    for _ in range(egg_cost):
        discard_an_egg(engine, agent, p, reason=f"play {card.name}")
    payment = _pick_food_payment(engine, agent, p, card)
    if payment is None:
        engine.log(
            f"[{p.name}] unable to pay for {card.name} (bug or shortage); wasting action"
        )
        return
    for f, n in payment.items():
        p.food[f] -= n

    p.hand.remove(card)
    pb = state.PlayedBird(bird=card)
    p.board[habitat].append(pb)
    engine.log(
        f"[{p.name}] plays {card.name} into {habitat.value} "
        f"(paid {payment.as_dict()}, {egg_cost} eggs)"
    )
    # WHITE power triggers when played.
    if card.color == cards.PowerColor.WHITE:
        powers.dispatch_power(engine, agent, p, pb, habitat, "play")


def discard_an_egg(engine: "core.Engine", agent: "core.Agent", p: state.Player, reason: str) -> None:
    """Force ``p`` to remove one egg from any of their birds (no-op if none).
    Used both as part of the play-bird cost and by any effect that demands
    an egg discard."""
    choices: list[decisions.BoardTargetChoice | decisions.SkipChoice] = []
    for h, row in p.board.items():
        for i, pb in enumerate(row):
            if pb.eggs > 0:
                choices.append(
                    decisions.BoardTargetChoice(
                        label=f"{pb.bird.name}@{h.value}[{i}]",
                        habitat=h,
                        slot=i,
                    )
                )
    if not choices:
        return
    ch = engine.ask(
        agent,
        decisions.PlayBirdPickEggToPayDecision(
            player_id=p.id,
            prompt=f"[{p.name}] discard an egg ({reason})",
            choices=choices,
        ),
    )
    assert isinstance(ch, decisions.BoardTargetChoice)
    p.board[ch.habitat][ch.slot].eggs -= 1


# ---------------------------------------------------------------------------
# Main action: gain food (Forest)


def do_gain_food(engine: "core.Engine", agent: "core.Agent") -> None:
    """Run a Gain Food action (always Forest): pull dice equal to the column
    reward then activate row powers right-to-left."""
    p = engine.state.me()
    n_birds = p.row_activation_count(cards.Habitat.FOREST)
    n_dice = p.board.gain_food_count()
    engine.log(f"[{p.name}] gain food: row has {n_birds} birds, take {n_dice} dice")
    for _ in range(n_dice):
        _take_one_die_active(engine, agent, p)
    # Reroll if 1 or fewer faces showing (printed rule).
    types_left = sum(1 for c in engine.state.birdfeeder.counts.values() if c > 0)
    if types_left <= 1 and engine.state.birdfeeder.total() > 0:
        engine.state.birdfeeder.reroll(engine.state.rng)
    activate_row_powers(engine, agent, p, cards.Habitat.FOREST)


def take_one_from_feeder(
    engine: "core.Engine",
    agent: "core.Agent",
    p: state.Player,
    pb: state.PlayedBird,
    avail: list[cards.Food],
    reason: str,
) -> None:
    """Pull one die from the birdfeeder into ``p``'s food. If only one food
    type is offered the choice is auto-resolved; otherwise the agent picks.
    ``avail`` must be non-empty and every entry must have a non-zero count
    in the birdfeeder."""
    st = engine.state
    if len(avail) == 1:
        f = avail[0]
    else:
        ch = engine.ask(
            agent,
            decisions.BirdPowerPickFoodDecision(
                player_id=p.id,
                prompt=f"[{p.name}] pick 1 from birdfeeder for {pb.bird.name}",
                choices=[
                    decisions.FoodChoice(label=f"{f.value}({st.birdfeeder.counts[f]})", food=f)
                    for f in avail
                ],
            ),
        )
        assert isinstance(ch, decisions.FoodChoice)
        f = ch.food
    st.birdfeeder.counts[f] -= 1
    p.food[f] += 1
    engine.log(f"  {pb.bird.name}: +1 {f.value} from birdfeeder")


# ---------------------------------------------------------------------------
# Main action: lay eggs (Grassland)


def do_lay_eggs(engine: "core.Engine", agent: "core.Agent") -> None:
    """Run a Lay Eggs action (always Grassland) then trigger pink reactors
    that fire on opponents' lay-egg actions."""
    p = engine.state.me()
    n_birds = p.row_activation_count(cards.Habitat.GRASSLAND)
    n_eggs = p.board.lay_eggs_count()
    engine.log(f"[{p.name}] lay eggs: row has {n_birds} birds, lay {n_eggs} eggs")
    for _ in range(n_eggs):
        lay_one_egg(engine, agent, p)
    activate_row_powers(engine, agent, p, cards.Habitat.GRASSLAND)
    reactors.trigger_pink_lay_eggs_reactors(engine, p)


def lay_one_egg(engine: "core.Engine", agent: "core.Agent", p: state.Player) -> None:
    """Prompt ``p`` to place one egg on any of their birds with room."""
    choices: list[decisions.BoardTargetChoice | decisions.SkipChoice] = [
        decisions.BoardTargetChoice(
            label=(f"{pb.bird.name}@{h.value}[{i}]" f"({pb.eggs}/{pb.bird.egg_limit})"),
            habitat=h,
            slot=i,
        )
        for h, row in p.board.items()
        for i, pb in enumerate(row)
        if pb.eggs < pb.bird.egg_limit
    ]
    if not choices:
        return
    ch = engine.ask(
        agent,
        decisions.LayEggPickBirdDecision(
            player_id=p.id,
            prompt=f"[{p.name}] lay 1 egg",
            choices=choices,
        ),
    )
    assert isinstance(ch, decisions.BoardTargetChoice)
    p.board[ch.habitat][ch.slot].eggs += 1


# ---------------------------------------------------------------------------
# Main action: draw cards (Wetland)


def do_draw_cards(engine: "core.Engine", agent: "core.Agent") -> None:
    """Run a Draw Cards action (always Wetland)."""
    p = engine.state.me()
    n_birds = p.row_activation_count(cards.Habitat.WETLAND)
    n_cards = p.board.draw_cards_count()
    engine.log(f"[{p.name}] draw cards: row has {n_birds} birds, draw {n_cards}")
    for _ in range(n_cards):
        draw_one_card(engine, agent, p)
    activate_row_powers(engine, agent, p, cards.Habitat.WETLAND)


def draw_one_card(engine: "core.Engine", agent: "core.Agent", p: state.Player) -> None:
    """Prompt ``p`` to draw a single card from any face-up tray slot or the
    top of the deck."""
    choices: list[decisions.DrawSourceChoice] = []
    for i, b in enumerate(engine.state.tray):
        choices.append(
            decisions.DrawSourceChoice(
                label=f"tray[{i}]={b.name}",
                source="tray",
                tray_index=i,
            )
        )
    if engine.state.bird_deck or engine.state.bird_discard:
        choices.append(decisions.DrawSourceChoice(label="deck", source="deck"))
    if not choices:
        return
    ch = engine.ask(
        agent,
        decisions.DrawCardsPickSourceDecision(
            player_id=p.id,
            prompt=f"[{p.name}] draw 1 card",
            choices=choices,
        ),
    )
    if ch.source == "tray" and ch.tray_index is not None:
        b = engine.state.tray.pop(ch.tray_index)
        engine.state.refill_tray()
        p.hand.append(b)
    else:
        b = engine.state.draw_bird()
        if b:
            p.hand.append(b)


# ---------------------------------------------------------------------------
# Row power activation


def activate_row_powers(
    engine: "core.Engine",
    agent: "core.Agent",
    p: state.Player,
    habitat: cards.Habitat,
) -> None:
    """Trigger BROWN powers right-to-left in the activated row."""
    for pb in reversed(p.board[habitat]):
        if pb.bird.color != cards.PowerColor.BROWN:
            continue
        pb.activations += 1
        powers.dispatch_power(engine, agent, p, pb, habitat, "activate")


###### PRIVATE #######

#### Play-bird sub-helpers ####


def _playable_birds(
    p: state.Player,
    habitat_filter: cards.Habitat | None,
) -> list[cards.Bird]:
    out: list[cards.Bird] = []
    for b in p.hand:
        if any(
            (habitat_filter is None or h == habitat_filter)
            and p.can_play_in(h)
            and helpers.enumerate_payments(p.food, b.food_cost)
            and p.total_eggs >= p.board.next_egg_cost(h)
            for h in b.habitats
        ):
            out.append(b)
    return out


def _log_wasted_play(
    engine: "core.Engine",
    p: state.Player,
    habitat_filter: cards.Habitat | None,
) -> None:
    if habitat_filter is not None:
        engine.log(
            f"[{p.name}] no playable bird in [{habitat_filter.value}]; "
            f"extra play wasted"
        )
    else:
        engine.log(
            f"[{p.name}] tried to play a bird but had no playable bird; "
            f"action wasted"
        )


def _pick_card(
    engine: "core.Engine",
    agent: "core.Agent",
    p: state.Player,
    playable: list[cards.Bird],
) -> cards.Bird:
    ch = engine.ask(
        agent,
        decisions.PlayBirdPickCardDecision(
            player_id=p.id,
            prompt=f"[{p.name}] pick a bird to play",
            choices=[decisions.BirdChoice(label=b.name, bird=b) for b in playable],
        ),
    )
    return ch.bird


def _pick_habitat(
    engine: "core.Engine",
    agent: "core.Agent",
    p: state.Player,
    card: cards.Bird,
    habitat_filter: cards.Habitat | None,
) -> cards.Habitat:
    habs = [
        h
        for h in card.habitats
        if p.can_play_in(h) and (habitat_filter is None or h == habitat_filter)
    ]
    if len(habs) == 1:
        return habs[0]
    ch = engine.ask(
        agent,
        decisions.PlayBirdPickHabitatDecision(
            player_id=p.id,
            prompt=f"[{p.name}] pick habitat for {card.name}",
            choices=[decisions.HabitatChoice(label=h.value, habitat=h) for h in habs],
        ),
    )
    return ch.habitat


def _pick_food_payment(
    engine: "core.Engine",
    agent: "core.Agent",
    p: state.Player,
    card: cards.Bird,
) -> state.FoodPool | None:
    payments = helpers.enumerate_payments(p.food, card.food_cost)
    if not payments:
        return None
    if len(payments) == 1:
        return payments[0]
    ch = engine.ask(
        agent,
        decisions.PlayBirdPickFoodPaymentDecision(
            player_id=p.id,
            prompt=f"[{p.name}] pick food payment for {card.name}",
            choices=[
                decisions.FoodPaymentChoice(
                    label=", ".join(f"{n}{f.value}" for f, n in pay.items() if n > 0),
                    payment=pay,
                )
                for pay in payments
            ],
        ),
    )
    return ch.payment


#### Gain-food sub-helpers ####


def _take_one_die_active(engine: "core.Engine", agent: "core.Agent", p: state.Player) -> None:
    """One iteration of the main Gain Food action loop: pull a die, rerolling
    once on an empty feeder, then stop if still empty."""
    avail = [(f, c) for f, c in engine.state.birdfeeder.counts.items() if c > 0]
    if not avail:
        engine.state.birdfeeder.reroll(engine.state.rng)
        engine.log(
            f"  birdfeeder empty; rerolled to {engine.state.birdfeeder.counts.as_dict()}"
        )
        avail = [(f, c) for f, c in engine.state.birdfeeder.counts.items() if c > 0]
        if not avail:
            return
    ch = engine.ask(
        agent,
        decisions.GainFoodPickDieDecision(
            player_id=p.id,
            prompt=f"[{p.name}] take 1 die from birdfeeder",
            choices=[decisions.FoodChoice(label=f"{f.value}({c})", food=f) for f, c in avail],
        ),
    )
    f = ch.food
    engine.state.birdfeeder.counts[f] -= 1
    p.food[f] += 1
    engine.log(f"  +1 {f.value}")
