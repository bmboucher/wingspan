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


def can_play_bird(engine: "core.Engine", player: state.Player) -> bool:
    """True if ``player`` has at least one bird in hand that could legally be
    played into some habitat right now (affordable food, payable eggs)."""
    return bool(playable_birds(player, habitat_filter=None))


def playable_birds(
    player: state.Player,
    habitat_filter: cards.Habitat | None,
) -> list[cards.Bird]:
    """Birds in ``player``'s hand that can legally be played right now (affordable
    food, payable egg cost, an open slot in a permitted habitat). When
    ``habitat_filter`` is set only that habitat is considered."""
    out: list[cards.Bird] = []
    for bird in player.hand:
        if any(
            (habitat_filter is None or habitat == habitat_filter)
            and player.can_play_in(habitat)
            and helpers.enumerate_payments(player.food, bird.food_cost)
            and player.total_eggs >= player.board.next_egg_cost(habitat)
            for habitat in bird.habitats
        ):
            out.append(bird)
    return out


def playable_bird_plays(
    player: state.Player,
    habitat_filter: cards.Habitat | None,
) -> list[tuple[cards.Bird, cards.Habitat, state.FoodPool]]:
    """Every fully-specified play ``player`` can make right now: one entry per
    legal ``(bird, habitat, food payment)`` combination.

    This is the expanded form of :func:`playable_birds` — instead of one entry
    per playable bird it enumerates each permitted habitat and each distinct
    food payment, so the main-action menu can offer the habitat / payment picks
    inline rather than as follow-up decisions. The egg cost is checked (a play
    whose egg cost can't be paid is excluded) but not enumerated; it is still
    resolved separately when the play is executed."""
    out: list[tuple[cards.Bird, cards.Habitat, state.FoodPool]] = []
    for bird in player.hand:
        payments = helpers.enumerate_payments(player.food, bird.food_cost)
        if not payments:
            continue
        for habitat in bird.habitats:
            if habitat_filter is not None and habitat != habitat_filter:
                continue
            if not player.can_play_in(
                habitat
            ) or player.total_eggs < player.board.next_egg_cost(habitat):
                continue
            for payment in payments:
                out.append((bird, habitat, payment))
    return out


def do_play_bird(
    engine: "core.Engine",
    agent: "core.Agent",
    card: cards.Bird | None = None,
    habitat: cards.Habitat | None = None,
    payment: state.FoodPool | None = None,
) -> None:
    """Run a Play Bird action for the current player.

    The main-action path resolves the whole play up front — ``MainActionDecision``
    offers one ``PlayBirdChoice`` per legal ``(bird, habitat, payment)`` — and
    passes all three in, so only the egg cost is asked for here. The extra-play
    path (``card is None``, which may carry a ``turn_extra_play_habitat``
    filter) leaves ``habitat`` / ``payment`` unset and prompts for the bird,
    habitat, and payment as follow-up decisions."""
    player = engine.state.me()
    habitat_filter = engine.state.turn_extra_play_habitat
    if card is None:
        playable = playable_birds(player, habitat_filter)
        if not playable:
            _log_wasted_play(engine, player, habitat_filter)
            return
        card = _pick_card(engine, agent, player, playable)
    if habitat is None:
        habitat = _pick_habitat(engine, agent, player, card, habitat_filter)

    # Pay egg + food costs in that order (matches printed action sequence). The
    # egg cost is always resolved here, even when the food payment was chosen
    # inline at the main-action stage.
    egg_cost = player.board.next_egg_cost(habitat)
    for _ in range(egg_cost):
        discard_an_egg(engine, agent, player, reason=f"play {card.name}")
    if payment is None:
        payment = _pick_food_payment(engine, agent, player, card)
    if payment is None:
        engine.log(
            f"[{player.name}] unable to pay for {card.name} (bug or shortage); wasting action"
        )
        return
    for food, amount in payment.items():
        player.food[food] -= amount

    player.hand.remove(card)
    pb = state.PlayedBird(bird=card)
    player.board[habitat].append(pb)
    engine.log(
        f"[{player.name}] plays {card.name} into {habitat.value} "
        f"(paid {payment.format()}, {egg_cost} eggs)"
    )
    # WHITE power triggers when played.
    if card.color == cards.PowerColor.WHITE:
        powers.dispatch_power(engine, agent, player, pb, habitat, "play")


def discard_an_egg(
    engine: "core.Engine", agent: "core.Agent", player: state.Player, reason: str
) -> None:
    """Force ``player`` to remove one egg from any of their birds (no-op if
    none). Used both as part of the play-bird cost and by any effect that
    demands an egg discard."""
    choices: list[decisions.BoardTargetChoice | decisions.SkipChoice] = []
    for habitat, row in player.board.items():
        for slot, pb in enumerate(row):
            if pb.eggs > 0:
                choices.append(
                    decisions.BoardTargetChoice(
                        label=f"{pb.bird.name}@{habitat.value}[{slot}]",
                        habitat=habitat,
                        slot=slot,
                    )
                )
    if not choices:
        return
    ch = engine.ask(
        agent,
        decisions.PlayBirdPickEggToPayDecision(
            player_id=player.id,
            prompt=f"[{player.name}] discard an egg ({reason})",
            choices=choices,
        ),
    )
    assert isinstance(ch, decisions.BoardTargetChoice)
    player.board[ch.habitat][ch.slot].eggs -= 1


# ---------------------------------------------------------------------------
# Main action: gain food (Forest)


def do_gain_food(engine: "core.Engine", agent: "core.Agent") -> None:
    """Run a Gain Food action (always Forest): pull dice equal to the column
    reward then activate row powers right-to-left."""
    player = engine.state.me()
    n_birds = player.row_activation_count(cards.Habitat.FOREST)
    n_dice = player.board.gain_food_count()
    engine.log(
        f"[{player.name}] gain food: row has {n_birds} birds, take {n_dice} dice"
    )
    for _ in range(n_dice):
        _take_one_die_active(engine, agent, player)
    _convert_gain_food(engine, agent, player)
    # Reroll if 1 or fewer faces showing (printed rule).
    types_left = sum(
        1 for count in engine.state.birdfeeder.counts.values() if count > 0
    )
    if types_left <= 1 and engine.state.birdfeeder.total() > 0:
        engine.state.birdfeeder.reroll(engine.state.rng)
    activate_row_powers(engine, agent, player, cards.Habitat.FOREST)


def take_one_from_feeder(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    pb: state.PlayedBird,
    avail: list[cards.Food],
    reason: str,
) -> None:
    """Pull one die from the birdfeeder into ``player``'s food. If only one food
    type is offered the choice is auto-resolved; otherwise the agent picks.
    ``avail`` must be non-empty and every entry must have a non-zero count
    in the birdfeeder."""
    st = engine.state
    if len(avail) == 1:
        chosen_food = avail[0]
    else:
        ch = engine.ask(
            agent,
            decisions.BirdPowerPickFoodDecision(
                player_id=player.id,
                prompt=f"[{player.name}] pick 1 from birdfeeder for {pb.bird.name}",
                choices=[
                    decisions.FoodChoice(
                        label=f"{food.value}({st.birdfeeder.counts[food]})", food=food
                    )
                    for food in avail
                ],
            ),
        )
        assert isinstance(ch, decisions.FoodChoice)
        chosen_food = ch.food
    st.birdfeeder.counts[chosen_food] -= 1
    player.food[chosen_food] += 1
    engine.log(f"  {pb.bird.name}: +1 {chosen_food.value} from birdfeeder")


# ---------------------------------------------------------------------------
# Main action: lay eggs (Grassland)


def do_lay_eggs(engine: "core.Engine", agent: "core.Agent") -> None:
    """Run a Lay Eggs action (always Grassland) then trigger pink reactors
    that fire on opponents' lay-egg actions."""
    player = engine.state.me()
    n_birds = player.row_activation_count(cards.Habitat.GRASSLAND)
    n_eggs = player.board.lay_eggs_count()
    engine.log(f"[{player.name}] lay eggs: row has {n_birds} birds, lay {n_eggs} eggs")
    for _ in range(n_eggs):
        lay_one_egg(engine, agent, player)
    _convert_lay_eggs(engine, agent, player)
    activate_row_powers(engine, agent, player, cards.Habitat.GRASSLAND)
    reactors.trigger_pink_lay_eggs_reactors(engine, player)


def lay_one_egg(
    engine: "core.Engine", agent: "core.Agent", player: state.Player
) -> None:
    """Prompt ``player`` to place one egg on any of their birds with room."""
    choices: list[decisions.BoardTargetChoice | decisions.SkipChoice] = [
        decisions.BoardTargetChoice(
            label=(
                f"{pb.bird.name}@{habitat.value}[{slot}]"
                f"({pb.eggs}/{pb.bird.egg_limit})"
            ),
            habitat=habitat,
            slot=slot,
        )
        for habitat, row in player.board.items()
        for slot, pb in enumerate(row)
        if pb.eggs < pb.bird.egg_limit
    ]
    if not choices:
        return
    ch = engine.ask(
        agent,
        decisions.LayEggPickBirdDecision(
            player_id=player.id,
            prompt=f"[{player.name}] lay 1 egg",
            choices=choices,
        ),
    )
    assert isinstance(ch, decisions.BoardTargetChoice)
    player.board[ch.habitat][ch.slot].eggs += 1


# ---------------------------------------------------------------------------
# Main action: draw cards (Wetland)


def do_draw_cards(engine: "core.Engine", agent: "core.Agent") -> None:
    """Run a Draw Cards action (always Wetland)."""
    player = engine.state.me()
    n_birds = player.row_activation_count(cards.Habitat.WETLAND)
    n_cards = player.board.draw_cards_count()
    engine.log(f"[{player.name}] draw cards: row has {n_birds} birds, draw {n_cards}")
    for _ in range(n_cards):
        draw_one_card(engine, agent, player)
    _convert_draw_cards(engine, agent, player)
    activate_row_powers(engine, agent, player, cards.Habitat.WETLAND)


def draw_one_card(
    engine: "core.Engine", agent: "core.Agent", player: state.Player
) -> None:
    """Prompt ``player`` to draw a single card from any face-up tray slot or the
    top of the deck.

    Taking a tray card leaves that slot empty for the rest of the turn — the
    tray is *not* refilled here. Refilling is deferred to the end of the turn
    (``Engine._take_turn``); only a bird power that explicitly says so refills
    mid-turn. So a second draw this turn sees a tray one card shorter, and the
    offered draw sources reflect only the cards still face-up."""
    choices: list[decisions.DrawSourceChoice] = []
    for tray_index, bird in enumerate(engine.state.tray):
        choices.append(
            decisions.DrawSourceChoice(
                label=f"tray[{tray_index}]={bird.name}",
                source="tray",
                tray_index=tray_index,
                bird=bird,
            )
        )
    if engine.state.bird_deck or engine.state.bird_discard:
        choices.append(decisions.DrawSourceChoice(label="deck", source="deck"))
    if not choices:
        return
    ch = engine.ask(
        agent,
        decisions.DrawCardsPickSourceDecision(
            player_id=player.id,
            prompt=f"[{player.name}] draw 1 card",
            choices=choices,
        ),
    )
    if ch.source == "tray" and ch.tray_index is not None:
        drawn = engine.state.tray.pop(ch.tray_index)
        player.hand.append(drawn)
    else:
        drawn = engine.state.draw_bird()
        if drawn:
            player.hand.append(drawn)


# ---------------------------------------------------------------------------
# Row power activation


def activate_row_powers(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    habitat: cards.Habitat,
) -> None:
    """Trigger BROWN powers right-to-left in the activated row."""
    for pb in reversed(player.board[habitat]):
        if pb.bird.color != cards.PowerColor.BROWN:
            continue
        pb.activations += 1
        powers.dispatch_power(engine, agent, player, pb, habitat, "activate")


###### PRIVATE #######

#### Play-bird sub-helpers ####


def _log_wasted_play(
    engine: "core.Engine",
    player: state.Player,
    habitat_filter: cards.Habitat | None,
) -> None:
    if habitat_filter is not None:
        engine.log(
            f"[{player.name}] no playable bird in [{habitat_filter.value}]; "
            f"extra play wasted"
        )
    else:
        engine.log(
            f"[{player.name}] tried to play a bird but had no playable bird; "
            f"action wasted"
        )


def _pick_card(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    playable: list[cards.Bird],
) -> cards.Bird:
    ch = engine.ask(
        agent,
        decisions.PlayBirdPickCardDecision(
            player_id=player.id,
            prompt=f"[{player.name}] pick a bird to play",
            choices=[
                decisions.BirdChoice(label=bird.name, bird=bird) for bird in playable
            ],
        ),
    )
    return ch.bird


def _pick_habitat(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    card: cards.Bird,
    habitat_filter: cards.Habitat | None,
) -> cards.Habitat:
    habs = [
        habitat
        for habitat in card.habitats
        if player.can_play_in(habitat)
        and (habitat_filter is None or habitat == habitat_filter)
    ]
    if len(habs) == 1:
        return habs[0]
    ch = engine.ask(
        agent,
        decisions.PlayBirdPickHabitatDecision(
            player_id=player.id,
            prompt=f"[{player.name}] pick habitat for {card.name}",
            choices=[
                decisions.HabitatChoice(label=habitat.value, habitat=habitat)
                for habitat in habs
            ],
        ),
    )
    return ch.habitat


def _pick_food_payment(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    card: cards.Bird,
) -> state.FoodPool | None:
    payments = helpers.enumerate_payments(player.food, card.food_cost)
    if not payments:
        return None
    if len(payments) == 1:
        return payments[0]
    ch = engine.ask(
        agent,
        decisions.PlayBirdPickFoodPaymentDecision(
            player_id=player.id,
            prompt=f"[{player.name}] pick food payment for {card.name}",
            choices=[
                decisions.FoodPaymentChoice(
                    label=", ".join(
                        f"{amount}{food.value}"
                        for food, amount in pay.items()
                        if amount > 0
                    ),
                    payment=pay,
                )
                for pay in payments
            ],
        ),
    )
    return ch.payment


#### Gain-food sub-helpers ####


def _take_one_die_active(
    engine: "core.Engine", agent: "core.Agent", player: state.Player
) -> None:
    """One iteration of the main Gain Food action loop: pull a die, rerolling
    once on an empty feeder, then stop if still empty."""
    avail = [
        (food, count)
        for food, count in engine.state.birdfeeder.counts.items()
        if count > 0
    ]
    if not avail:
        engine.state.birdfeeder.reroll(engine.state.rng)
        engine.log(
            f"  birdfeeder empty; rerolled to {engine.state.birdfeeder.counts.format()}"
        )
        avail = [
            (food, count)
            for food, count in engine.state.birdfeeder.counts.items()
            if count > 0
        ]
        if not avail:
            return
    ch = engine.ask(
        agent,
        decisions.GainFoodPickDieDecision(
            player_id=player.id,
            prompt=f"[{player.name}] take 1 die from birdfeeder",
            choices=[
                decisions.FoodChoice(label=f"{food.value}({count})", food=food)
                for food, count in avail
            ],
        ),
    )
    chosen_food = ch.food
    engine.state.birdfeeder.counts[chosen_food] -= 1
    player.food[chosen_food] += 1
    engine.log(f"  +1 {chosen_food.value}")


#### Habitat-action conversions ####
#
# The printed player mat puts a trade arrow on every other action space, so the
# cube lands on one only when the row holds an odd number of birds. There the
# action may make a single resource trade for one extra of its reward: Forest
# discards a card for a food, Grassland spends a food for an egg, Wetland
# discards an egg for a card. The trade is one-shot, not repeatable.


def _convert_gain_food(
    engine: "core.Engine", agent: "core.Agent", player: state.Player
) -> None:
    """Forest conversion: on a trade space, optionally discard one card from
    hand to take one extra food die (a single exchange)."""
    if not player.board.action_offers_convert(cards.Habitat.FOREST) or not player.hand:
        return
    choices: list[decisions.BirdChoice | decisions.SkipChoice] = [
        decisions.BirdChoice(label=f"discard {bird.name} -> +1 food", bird=bird)
        for bird in player.hand
    ]
    choices.append(decisions.SkipChoice(label="keep cards"))
    ch = engine.ask(
        agent,
        decisions.GainFoodConvertDecision(
            player_id=player.id,
            prompt=f"[{player.name}] discard a card to gain 1 extra food?",
            choices=choices,
        ),
    )
    if isinstance(ch, decisions.SkipChoice):
        return
    player.hand.remove(ch.bird)
    engine.state.bird_discard.append(ch.bird)
    engine.log(f"  convert: discard {ch.bird.name} for +1 food")
    _take_one_die_active(engine, agent, player)


def _convert_lay_eggs(
    engine: "core.Engine", agent: "core.Agent", player: state.Player
) -> None:
    """Grassland conversion: on a trade space, optionally spend one food to lay
    one extra egg (a single exchange)."""
    if not player.board.action_offers_convert(cards.Habitat.GRASSLAND):
        return
    if player.food.total() == 0 or not _has_open_egg_slot(player):
        return
    choices: list[decisions.FoodChoice | decisions.SkipChoice] = [
        decisions.FoodChoice(label=f"spend {food.value} -> +1 egg", food=food)
        for food in player.food.types_with_positive()
    ]
    choices.append(decisions.SkipChoice(label="keep food"))
    ch = engine.ask(
        agent,
        decisions.LayEggsConvertDecision(
            player_id=player.id,
            prompt=f"[{player.name}] spend 1 food to lay 1 extra egg?",
            choices=choices,
        ),
    )
    if isinstance(ch, decisions.SkipChoice):
        return
    player.food[ch.food] -= 1
    engine.log(f"  convert: spend {ch.food.value} for +1 egg")
    lay_one_egg(engine, agent, player)


def _convert_draw_cards(
    engine: "core.Engine", agent: "core.Agent", player: state.Player
) -> None:
    """Wetland conversion: on a trade space, optionally discard one egg to draw
    one extra card (a single exchange)."""
    if not player.board.action_offers_convert(cards.Habitat.WETLAND):
        return
    if player.total_eggs == 0 or not _cards_available_to_draw(engine):
        return
    ch = engine.ask(
        agent,
        decisions.DrawCardsConvertDecision(
            player_id=player.id,
            prompt=f"[{player.name}] discard 1 egg to draw 1 extra card?",
            choices=[
                decisions.PayCostChoice(label="discard 1 egg -> +1 card"),
                decisions.SkipChoice(label="keep eggs"),
            ],
        ),
    )
    if isinstance(ch, decisions.SkipChoice):
        return
    engine.log("  convert: discard 1 egg for +1 card")
    discard_an_egg(engine, agent, player, reason="convert to draw a card")
    draw_one_card(engine, agent, player)


def _has_open_egg_slot(player: state.Player) -> bool:
    """True if ``player`` has any bird in play below its egg capacity."""
    return any(
        pb.eggs < pb.bird.egg_limit for row in player.board.values() for pb in row
    )


def _cards_available_to_draw(engine: "core.Engine") -> bool:
    """True if a card can still be drawn from the tray or (re)stocked deck."""
    return bool(
        engine.state.tray or engine.state.bird_deck or engine.state.bird_discard
    )
