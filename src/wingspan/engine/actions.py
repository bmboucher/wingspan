"""Implementations of Wingspan's four main actions and their direct helpers.

Each public function takes the live ``Engine`` as its first argument and
mutates the underlying ``GameState`` through it. The Engine's turn loop calls
``do_play_bird_action`` / ``do_gain_food`` / ``do_lay_eggs`` / ``do_draw_cards``
directly as free functions — there are no ``_do_*`` wrapper methods on Engine.
``do_play_bird`` is the lower-level executor (given a fully-specified play) that
``do_play_bird_action`` and the extra-play loop both call.
"""

from __future__ import annotations

import typing

from wingspan import cards, decisions, state
from wingspan.engine import helpers, powers, reactors

if typing.TYPE_CHECKING:
    from wingspan.engine import core


# ---------------------------------------------------------------------------
# Main action: play a bird


def playable_bird_plays(
    player: state.Player,
    habitat_filter: cards.Habitat | None,
) -> list[tuple[cards.Bird, cards.Habitat, state.FoodPool]]:
    """Every fully-specified play ``player`` can make right now: one entry per
    legal ``(bird, habitat, food payment)`` combination.

    Each entry is one playable bird × each permitted habitat × each distinct
    food payment, so the play menu (``PlayBirdDecision``) — reached both from
    the main action's ``PLAY_BIRD`` branch and each power-granted extra play —
    can offer the habitat / payment picks inline as one ``PlayBirdChoice``.
    ``MainActionDecision`` also calls this to decide whether to offer
    ``PLAY_BIRD`` at all. ``habitat_filter`` restricts to a single habitat (House Wren's
    "play in this habitat" extra play). The egg cost is checked (a play whose
    egg cost can't be paid is excluded) but not enumerated; it is still resolved
    separately when the play is executed. An empty result means ``player`` has
    no legal play right now."""
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
    card: cards.Bird,
    habitat: cards.Habitat,
    payment: state.FoodPool,
) -> None:
    """Run a Play Bird action for the current player, given a fully-specified
    play.

    The caller resolves the bird, habitat, and food payment up front (via the
    ``PlayBirdDecision`` menu, for both the main action and extra plays), so
    only the egg cost is asked for here.

    Pay egg then food costs in that order (matching the printed action
    sequence); the egg cost is resolved via ``RemoveEggDecision``. Then place
    the bird and fire its WHITE 'when played' power."""
    player = engine.state.me()
    egg_cost = player.board.next_egg_cost(habitat)
    for _ in range(egg_cost):
        discard_an_egg(engine, agent, player, reason=f"play {card.name}")
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


def do_play_bird_action(engine: "core.Engine", agent: "core.Agent") -> None:
    """Run the 'play a bird' main action: offer the play menu (one
    ``PlayBirdChoice`` per legal ``(bird, habitat, payment)``) via
    ``PlayBirdDecision`` and run the chosen play.

    Only reached when at least one legal play exists — ``MainActionDecision``
    gates the ``PLAY_BIRD`` option on that — so the menu is normally non-empty;
    the empty case is a defensive no-op (the spent cube is wasted)."""
    player = engine.state.me()
    plays = playable_bird_plays(player, habitat_filter=None)
    if not plays:
        engine.log(f"[{player.name}] has no playable bird; action wasted")
        return
    choice = _ask_play_bird(engine, agent, player, plays, extra=False)
    do_play_bird(engine, agent, choice.bird, choice.habitat, choice.payment)


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
        decisions.RemoveEggDecision(
            player_id=player.id,
            prompt=f"[{player.name}] discard an egg ({reason})",
            choices=choices,
        ),
    )
    assert isinstance(ch, decisions.BoardTargetChoice)
    player.board[ch.habitat][ch.slot].eggs -= 1


def consume_extra_plays(
    engine: "core.Engine", player: state.Player, agent: "core.Agent"
) -> None:
    """Resolve any +extra-play credits ``player`` accrued during the turn.

    Each extra play is offered as a ``PlayBirdDecision`` — the same
    ``(bird, habitat, payment)`` ``PlayBirdChoice`` menu the main action's
    ``PLAY_BIRD`` branch uses, routed to the play-bird head — restricted to the
    granting power's habitat when one is set (House Wren). With no legal play the
    credit (and any remaining credits) is wasted. Called from
    ``Engine._take_turn`` after the main action resolves."""
    while engine.state.turn_extra_plays > 0:
        engine.state.turn_extra_plays -= 1
        habitat_filter = engine.state.turn_extra_play_habitat
        plays = playable_bird_plays(player, habitat_filter)
        if not plays:
            _log_wasted_extra_play(engine, player, habitat_filter)
            engine.state.turn_extra_play_habitat = None
            break
        if habitat_filter is not None:
            engine.log(
                f"[{player.name}] takes an EXTRA play in [{habitat_filter.value}]"
            )
        else:
            engine.log(f"[{player.name}] takes an EXTRA play")
        choice = _ask_play_bird(engine, agent, player, plays, extra=True)
        do_play_bird(engine, agent, choice.bird, choice.habitat, choice.payment)
        # Habitat lock applies to a single extra play only.
        engine.state.turn_extra_play_habitat = None


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
            decisions.GainFoodDecision(
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
        decisions.LayEggDecision(
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


def _ask_play_bird(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    plays: list[tuple[cards.Bird, cards.Habitat, state.FoodPool]],
    *,
    extra: bool,
) -> decisions.PlayBirdChoice:
    """Offer the play menu — one ``PlayBirdChoice`` per legal
    ``(bird, habitat, payment)`` — as a ``PlayBirdDecision`` and return the
    chosen play. Shared by the main action's ``PLAY_BIRD`` branch
    (``extra=False``) and each power-granted extra play (``extra=True``); only
    the prompt wording differs, so the play-bird head sees the same candidate
    shape in both contexts."""
    choices = [
        decisions.PlayBirdChoice(
            label=f"play {bird.name} in {habitat.value} for {payment.format()}",
            bird=bird,
            habitat=habitat,
            payment=payment,
        )
        for bird, habitat, payment in plays
    ]
    prompt = (
        f"[{player.name}] choose a bird to play (extra play)"
        if extra
        else f"[{player.name}] choose a bird to play"
    )
    return engine.ask(
        agent,
        decisions.PlayBirdDecision(
            player_id=player.id,
            prompt=prompt,
            choices=choices,
        ),
    )


def _log_wasted_extra_play(
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
            f"[{player.name}] tried to take an extra play but had no playable "
            f"bird; wasted"
        )


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
        decisions.GainFoodDecision(
            player_id=player.id,
            prompt=f"[{player.name}] take 1 die from birdfeeder",
            choices=[
                decisions.FoodChoice(label=f"{food.value}({count})", food=food)
                for food, count in avail
            ],
        ),
    )
    assert isinstance(ch, decisions.FoodChoice)
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
        decisions.GainExtraFoodDecision(
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
        decisions.LayExtraEggsDecision(
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
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=f"[{player.name}] discard 1 egg to draw 1 extra card?",
            choices=[
                decisions.PayCostChoice(
                    label="discard 1 egg -> +1 card",
                    paid_egg_count=1,
                    gained_card_count=1,
                ),
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
