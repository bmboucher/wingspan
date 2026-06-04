"""Implementations of Wingspan's four main actions and their direct helpers.

Each public function takes the live ``Engine`` as its first argument and
mutates the underlying ``GameState`` through it. The Engine's turn loop calls
``do_play_bird_action`` / ``do_gain_food`` / ``do_lay_eggs`` / ``do_draw_cards``
directly as free functions — there are no ``_do_*`` wrapper methods on Engine.
``do_play_bird`` is the lower-level executor (given a committed
``(bird, habitat)`` pick; it resolves the egg and food costs as follow-up
decisions) that ``do_play_bird_action`` and the extra-play loop both call.
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
) -> list[tuple[cards.Bird, cards.Habitat]]:
    """Every play ``player`` can complete right now: one entry per legal
    ``(bird, habitat)`` pair.

    A pair is legal when the habitat has an open slot, the egg cost is
    affordable, and at least one food payment covers the bird's printed cost
    (``helpers.any_payment_exists``). The costs themselves are not enumerated
    here — they resolve as follow-up decisions when the play executes
    (``RemoveEggDecision`` then ``PayBirdFoodDecision`` inside
    ``do_play_bird``). The play menu (``PlayBirdDecision``) — reached both from
    the main action's ``PLAY_BIRD`` branch and each power-granted extra play —
    offers one ``PlayBirdChoice`` per pair. ``habitat_filter`` restricts to a
    single habitat (House Wren's "play in this habitat" extra play). An empty
    result means ``player`` has no legal play right now."""
    out: list[tuple[cards.Bird, cards.Habitat]] = []
    for bird in player.hand:
        if not helpers.any_payment_exists(player.food, bird.food_cost):
            continue
        for habitat in bird.habitats:
            if habitat_filter is not None and habitat != habitat_filter:
                continue
            if not player.can_play_in(
                habitat
            ) or player.total_eggs < player.board.next_egg_cost(habitat):
                continue
            out.append((bird, habitat))
    return out


def any_playable_bird_play(player: state.Player) -> bool:
    """Whether ``player`` has *any* legal bird play right now.

    The truthiness twin of :func:`playable_bird_plays` with ``habitat_filter``
    of ``None``: ``MainActionDecision`` calls this once per turn purely to
    decide whether to offer ``PLAY_BIRD`` at all, and only needs the boolean.
    Returning on the first legal ``(bird, habitat)`` — and asking
    ``helpers.any_payment_exists`` instead of enumerating every payment —
    avoids building the full play menu (and a ``FoodPool`` per payment) just
    to learn that at least one play exists. The actual menu is enumerated
    later, only when ``PLAY_BIRD`` is chosen."""
    for bird in player.hand:
        if not helpers.any_payment_exists(player.food, bird.food_cost):
            continue
        for habitat in bird.habitats:
            if player.can_play_in(habitat) and player.total_eggs >= (
                player.board.next_egg_cost(habitat)
            ):
                return True
    return False


def do_play_bird(
    engine: "core.Engine",
    agent: "core.Agent",
    card: cards.Bird,
    habitat: cards.Habitat,
) -> None:
    """Run a Play Bird action for the current player, given a committed
    ``(bird, habitat)`` pick from the ``PlayBirdDecision`` menu (the main
    action's ``PLAY_BIRD`` branch or an extra play).

    The costs resolve here as follow-up decisions, eggs then food (matching
    the printed action sequence): the egg cost via ``RemoveEggDecision``
    (one ask per egg), then the food payment via ``PayBirdFoodDecision``
    (one ask, choosing among the legal payment multisets — forced and
    auto-resolved when only one is legal). Paying eggs never changes the
    player's food, so the payments enumerated here match what made the pair
    legal at menu time. Then place the bird and fire its WHITE 'when played'
    power."""
    player = engine.state.me()
    egg_cost = player.board.next_egg_cost(habitat)
    for _ in range(egg_cost):
        discard_an_egg(engine, agent, player, reason=f"play {card.name}")
    payment = _ask_bird_food_payment(engine, agent, player, card, habitat).payment
    for food, amount in payment.items():
        player.food[food] -= amount

    player.hand.remove(card)
    pb = state.PlayedBird(bird=card)
    player.board[habitat].append(pb)
    engine.instrumentation.bird_placed(
        engine=engine, player=player, bird=card, habitat=habitat, played_bird=pb
    )
    engine.log(
        f"[{player.name}] plays {card.name} into {habitat.value} "
        f"(paid {payment.format()}, {egg_cost} eggs)"
    )
    # WHITE power triggers when played.
    if card.color == cards.PowerColor.WHITE:
        powers.dispatch_power(engine, agent, player, pb, habitat, "play")
    # Pink reactors: an opponent's "when another player plays a bird in their
    # [habitat]" power fires when this play's habitat matches.
    reactors.trigger_pink_play_bird_reactors(engine, player, habitat)


def do_play_bird_action(engine: "core.Engine", agent: "core.Agent") -> None:
    """Run the 'play a bird' main action: offer the play menu (one
    ``PlayBirdChoice`` per legal ``(bird, habitat)``) via ``PlayBirdDecision``
    and run the chosen play (its costs resolve inside ``do_play_bird``).

    Only reached when at least one legal play exists — ``MainActionDecision``
    gates the ``PLAY_BIRD`` option on that — so the menu is normally non-empty;
    the empty case is a defensive no-op (the spent cube is wasted)."""
    player = engine.state.me()
    plays = playable_bird_plays(player, habitat_filter=None)
    if not plays:
        engine.log(f"[{player.name}] has no playable bird; action wasted")
        return
    choice = _ask_play_bird(engine, agent, player, plays, extra=False)
    do_play_bird(engine, agent, choice.bird, choice.habitat)


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
        engine.log_skipped_decision(player.id, "no choices")
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

    An extra play is optional ("you may play another bird"), so each credit
    with a legal play opens with an ``AcceptExchangeDecision`` — take the play
    or forfeit the credit — routed to the skip-optional head. On accept, the
    play is offered as a ``PlayBirdDecision`` — the same ``(bird, habitat)``
    ``PlayBirdChoice`` menu the main action's ``PLAY_BIRD`` branch uses, routed
    to the play-bird head — restricted to the granting power's habitat when one
    is set (House Wren). With no legal play the credit (and any remaining
    credits) is wasted without consulting the agent. Called from
    ``Engine._take_turn`` after the main action resolves."""
    while engine.state.turn_extra_plays > 0:
        engine.state.turn_extra_plays -= 1
        habitat_filter = engine.state.turn_extra_play_habitat
        plays = playable_bird_plays(player, habitat_filter)
        if not plays:
            _log_wasted_extra_play(engine, player, habitat_filter)
            engine.state.turn_extra_play_habitat = None
            break
        if not _accept_extra_play(engine, agent, player, habitat_filter):
            engine.log(f"[{player.name}] declines the extra play")
            engine.state.turn_extra_play_habitat = None
            continue
        if habitat_filter is not None:
            engine.log(
                f"[{player.name}] takes an EXTRA play in [{habitat_filter.value}]"
            )
        else:
            engine.log(f"[{player.name}] takes an EXTRA play")
        choice = _ask_play_bird(engine, agent, player, plays, extra=True)
        do_play_bird(engine, agent, choice.bird, choice.habitat)
        # Habitat lock applies to a single extra play only.
        engine.state.turn_extra_play_habitat = None


# ---------------------------------------------------------------------------
# Main action: gain food (Forest)


def do_gain_food(engine: "core.Engine", agent: "core.Agent") -> None:
    """Run a Gain Food action (always Forest): pull dice equal to the column
    reward then activate row powers right-to-left."""
    player = engine.state.me()
    food_before = list(player.food.counts)
    n_birds = player.row_activation_count(cards.Habitat.FOREST)
    n_dice = player.board.gain_food_count()
    engine.log(
        f"[{player.name}] gain food: row has {n_birds} birds, take {n_dice} dice"
    )
    for _ in range(n_dice):
        _take_one_die_active(engine, agent, player)
    _convert_gain_food(engine, agent, player)
    activate_row_powers(engine, agent, player, cards.Habitat.FOREST)
    # Pink reactors: an opponent's "when another player gains [food]" power
    # (Loggerhead Shrike) fires on the foods gained during this action.
    gained_foods = {
        food
        for i, food in enumerate(cards.ALL_FOODS)
        if player.food.counts[i] > food_before[i]
    }
    reactors.trigger_pink_gain_food_reactors(engine, player, gained_foods)
    engine.instrumentation.food_gained(
        engine=engine, player=player, gained=gained_foods
    )


def offer_birdfeeder_reset(
    engine: "core.Engine", agent: "core.Agent", player: state.Player
) -> None:
    """Apply the birdfeeder reset rules just before ``player`` takes food.

    This is the preparation step of the two feeder-gain entry points —
    :func:`take_one_from_feeder` and :func:`take_all_of_food` — which run it
    before reading the feeder, so the menu / count they build reflects any
    reroll. Power and reactor code must not call it directly: route every
    feeder gain through those entry points and the offer can never be skipped.
    (It stays public so the reset rules can be tested in isolation.)

    Two printed rules fire here, in order:

    * **Empty feeder — automatic (Rule 1).** A feeder with no dice is rerolled
      at once; it is never a player choice. :func:`gain_feeder_die` already
      refills the feeder the instant a take empties it, so in normal play the
      feeder is never empty here — this guard only matters for a feeder emptied
      out of band, and it keeps a gain from ever building a choice-less
      decision.
    * **Single face — optional (Rule 2).** When every die shows the same face —
      one single food, or all dice on the invertebrate/seed choice face
      (``Birdfeeder.reset_available()``) — the player may reroll the whole
      feeder before taking. ``engine.ask`` a ``ResetBirdfeederDecision``; on the
      affirmative choice, reroll."""
    feeder = engine.state.birdfeeder
    if feeder.is_empty():
        feeder.reroll(engine.state.rng)
        engine.log(f"  birdfeeder empty; rerolled to {feeder.counts.format()}")
    if not feeder.reset_available():
        return
    ch = engine.ask(
        agent,
        decisions.ResetBirdfeederDecision(
            player_id=player.id,
            prompt=f"[{player.name}] feeder shows one face; reroll it first?",
            choices=[
                decisions.ResetBirdfeederChoice(
                    label="reset the birdfeeder (reroll all dice)"
                ),
                decisions.SkipChoice(label="take from the feeder as-is"),
            ],
        ),
    )
    if isinstance(ch, decisions.ResetBirdfeederChoice):
        feeder.reroll(engine.state.rng)
        engine.log(f"  {player.name} resets the birdfeeder -> {feeder.counts.format()}")


def gain_feeder_die(
    engine: "core.Engine",
    player: state.Player,
    food: cards.Food,
    *,
    from_choice_die: bool = False,
) -> None:
    """Move one ``food`` die from the feeder into ``player``'s supply, then apply
    Rule 1: if that emptied the feeder, immediately reroll it.

    Every birdfeeder gain routes through here so the "reroll an empty feeder"
    rule lives in one place — the feeder is therefore never observed empty at any
    decision point. ``from_choice_die`` forwards to :meth:`Birdfeeder.take`, so a
    gain offered as the invertebrate/seed choice-die option spends a choice die
    rather than a single face. The caller must ensure ``food`` is gainable the
    requested way (see ``Birdfeeder.gain_options``)."""
    feeder = engine.state.birdfeeder
    feeder.take(food, from_choice_die=from_choice_die)
    player.food[food] += 1
    if feeder.is_empty():
        feeder.reroll(engine.state.rng)
        engine.log(f"  birdfeeder emptied; rerolled to {feeder.counts.format()}")


def take_one_from_feeder(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    *,
    prompt: str,
    allowed: list[cards.Food] | None = None,
) -> cards.Food | None:
    """Offer the pre-gain birdfeeder reset, then pull one die of ``player``'s
    choice into their food.

    The single sanctioned entry point for a "take 1 die you choose" feeder
    gain: the reset rules run first (:func:`offer_birdfeeder_reset`), the menu
    is built from the *post-reset* feeder (``allowed`` restricts it to specific
    foods, e.g. an either-of-two power; ``None`` offers everything showing),
    the pick routes through ``engine.ask`` (whose single-choice guard
    auto-resolves a one-option gain without consulting the agent), and the die
    moves via :func:`gain_feeder_die`. Returns the food gained, or ``None``
    when nothing in the post-reset feeder matches ``allowed`` — callers log
    their own gain / skip message. Callers must not pre-read the feeder to
    build the menu; everything feeder-dependent happens here, after the
    reset."""
    offer_birdfeeder_reset(engine, agent, player)
    feeder = engine.state.birdfeeder
    # ``gain_options`` expands the allowed foods into the distinct plain /
    # choice-die ways to take each (see Birdfeeder).
    options = feeder.gain_options(allowed)
    if not options:
        return None
    ch = engine.ask(
        agent,
        decisions.GainFoodDecision(
            player_id=player.id,
            prompt=prompt,
            choices=[
                decisions.FoodChoice(
                    label=feeder.gain_option_label(food, combo),
                    food=food,
                    from_choice_die=combo,
                )
                for food, combo in options
            ],
        ),
    )
    assert isinstance(ch, decisions.FoodChoice)
    gain_feeder_die(engine, player, ch.food, from_choice_die=ch.from_choice_die)
    return ch.food


def take_all_of_food(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    food: cards.Food,
    *,
    limit: int | None = None,
) -> int:
    """Offer the pre-gain birdfeeder reset, then take every die that could
    yield ``food`` (capped at ``limit``), crediting each to ``player``'s food.

    The no-choice sibling of :func:`take_one_from_feeder` for "gain all [food]
    in the birdfeeder" powers: after the reset rules run, the count is read
    from the post-reset feeder and the dice move one at a time through
    :func:`gain_feeder_die` (which falls back to choice dice for
    invertebrate / seed). Returns the number of dice taken — 0 when the
    post-reset feeder shows none — and callers log their own message."""
    offer_birdfeeder_reset(engine, agent, player)
    count = engine.state.birdfeeder.gainable_count(food)
    if limit is not None:
        count = min(limit, count)
    for _ in range(count):
        gain_feeder_die(engine, player, food)
    return count


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
    engine.instrumentation.eggs_laid(engine=engine, player=player, count=n_eggs)


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
        engine.log_skipped_decision(player.id, "no choices")
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
    engine.instrumentation.cards_drawn(engine=engine, player=player, count=n_cards)


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
        if bird is not None:
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
        engine.log_skipped_decision(player.id, "no choices")
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
        drawn = engine.state.tray[ch.tray_index]
        assert drawn is not None
        engine.state.tray[ch.tray_index] = None
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
    """Trigger BROWN powers right-to-left in the activated row.

    Every bird in the row gets a log header as it is visited — its power text
    for a brown power, a "no brown power" note otherwise — so the detailed game
    log accounts for the whole row even when a power fizzles or is declined."""
    for pb in reversed(player.board[habitat]):
        if pb.bird.color != cards.PowerColor.BROWN:
            engine.log(f"[{player.name}] @ {pb.bird.name} - no brown power")
            continue
        engine.log(f'[{player.name}] @ {pb.bird.name} - "{pb.bird.plain_power_text}"')
        pb.activations += 1
        powers.dispatch_power(engine, agent, player, pb, habitat, "activate")


###### PRIVATE #######

#### Play-bird sub-helpers ####


def _ask_play_bird(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    plays: list[tuple[cards.Bird, cards.Habitat]],
    *,
    extra: bool,
) -> decisions.PlayBirdChoice:
    """Offer the play menu — one ``PlayBirdChoice`` per legal
    ``(bird, habitat)`` pair — as a ``PlayBirdDecision`` and return the chosen
    play. Shared by the main action's ``PLAY_BIRD`` branch (``extra=False``)
    and each power-granted extra play (``extra=True``); only the prompt wording
    differs, so the play-bird head sees the same candidate shape in both
    contexts. The chosen play's costs resolve afterwards inside
    ``do_play_bird``."""
    choices = [
        decisions.PlayBirdChoice(
            label=f"play {bird.name} in {habitat.value}",
            bird=bird,
            habitat=habitat,
        )
        for bird, habitat in plays
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


def _ask_bird_food_payment(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    card: cards.Bird,
    habitat: cards.Habitat,
) -> decisions.FoodPaymentChoice:
    """Ask how to pay ``card``'s printed food cost — one ``FoodPaymentChoice``
    per legal payment multiset — as a ``PayBirdFoodDecision`` and return the
    chosen payment.

    Mandatory (no skip): the play was committed upstream, and
    ``playable_bird_plays`` only offered pairs with at least one legal payment,
    so the menu is never empty. With exactly one legal payment the decision is
    forced and ``Engine.ask`` auto-resolves it without consulting the agent."""
    choices = [
        decisions.FoodPaymentChoice(label=f"pay {payment.format()}", payment=payment)
        for payment in helpers.enumerate_payments(player.food, card.food_cost)
    ]
    return engine.ask(
        agent,
        decisions.PayBirdFoodDecision(
            player_id=player.id,
            prompt=f"[{player.name}] pay for {card.name} in {habitat.value}",
            choices=choices,
            bird=card,
            habitat=habitat,
        ),
    )


def _accept_extra_play(
    engine: "core.Engine",
    agent: "core.Agent",
    player: state.Player,
    habitat_filter: cards.Habitat | None,
) -> bool:
    """Ask whether ``player`` takes a power-granted extra play or forfeits the
    credit — an ``AcceptExchangeDecision`` whose accept option carries
    ``gained_play_count=1`` (the only term of this "trade": nothing is paid up
    front; the play's own costs resolve in the follow-ups). Routed to the
    skip-optional head like every other take-it-or-leave-it offer. Only asked
    when a legal play exists."""
    where = f" in {habitat_filter.value}" if habitat_filter is not None else ""
    ch = engine.ask(
        agent,
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=f"[{player.name}] play another bird{where}?",
            choices=[
                decisions.PayCostChoice(
                    label=f"play a bird{where}",
                    gained_play_count=1,
                ),
                decisions.SkipChoice(label="forfeit the extra play"),
            ],
        ),
    )
    return isinstance(ch, decisions.PayCostChoice)


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
    """One iteration of the main Gain Food action loop: one die of the
    player's choice via :func:`take_one_from_feeder` (which offers the
    optional single-face reset first).

    The feeder is kept non-empty by :func:`gain_feeder_die` (and the setup
    reroll), so there is always at least one gainable food to offer here."""
    gained = take_one_from_feeder(
        engine,
        agent,
        player,
        prompt=f"[{player.name}] take 1 die from birdfeeder",
    )
    assert gained is not None  # an unrestricted post-reset menu is never empty
    engine.log(f"  +1 {gained.value}")


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
    hand to take one extra food die (a single exchange).

    The exchange is split into three separate decisions so each judgment routes
    to its proper policy head: commit-to-cost (is the trade worth it?),
    discard-bird (which card can I most afford to lose?), and gain-food (which
    die do I want?).
    """
    if not player.board.action_offers_convert(cards.Habitat.FOREST) or not player.hand:
        return

    # Step 1 — commit to the exchange or skip.
    commit_ch = engine.ask(
        agent,
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=f"[{player.name}] discard a card to gain 1 extra food?",
            choices=[
                decisions.PayCostChoice(
                    label="discard 1 card -> +1 food",
                    paid_card_count=1,
                    gained_food_count=1,
                ),
                decisions.SkipChoice(label="keep cards"),
            ],
        ),
    )
    if isinstance(commit_ch, decisions.SkipChoice):
        return

    # Step 2 — pick which card to discard.
    discard_ch = engine.ask(
        agent,
        decisions.DiscardBirdForFoodDecision(
            player_id=player.id,
            prompt=f"[{player.name}] which card to discard?",
            choices=[
                decisions.BirdChoice(label=bird.name, bird=bird) for bird in player.hand
            ],
        ),
    )
    player.hand.remove(discard_ch.bird)
    engine.state.bird_discard.append(discard_ch.bird)
    engine.log(f"  convert: discard {discard_ch.bird.name} for +1 food")

    # Step 3 — pick which food die to take.
    _take_one_die_active(engine, agent, player)


def _convert_lay_eggs(
    engine: "core.Engine", agent: "core.Agent", player: state.Player
) -> None:
    """Grassland conversion: on a trade space, optionally spend one food to lay
    one extra egg (a single exchange).

    The exchange is split into three separate decisions so each judgment routes
    to its proper policy head: commit-to-cost (is the trade worth it?),
    spend-food (which food can I most afford to lose?), and lay-egg (which
    bird should get the extra egg?).
    """
    if not player.board.action_offers_convert(cards.Habitat.GRASSLAND):
        return
    if player.food.total() == 0 or not _has_open_egg_slot(player):
        return

    # Step 1 — commit to the exchange or skip.
    commit_ch = engine.ask(
        agent,
        decisions.AcceptExchangeDecision(
            player_id=player.id,
            prompt=f"[{player.name}] spend 1 food to lay 1 extra egg?",
            choices=[
                decisions.PayCostChoice(
                    label="spend 1 food -> +1 egg",
                    paid_food_count=1,
                    gained_egg_count=1,
                ),
                decisions.SkipChoice(label="keep food"),
            ],
        ),
    )
    if isinstance(commit_ch, decisions.SkipChoice):
        return

    # Step 2 — pick which food to spend.
    spend_ch = engine.ask(
        agent,
        decisions.SpendFoodForEggDecision(
            player_id=player.id,
            prompt=f"[{player.name}] which food to spend?",
            choices=[
                decisions.FoodChoice(label=f"spend {food.value}", food=food)
                for food in player.food.types_with_positive()
            ],
        ),
    )
    player.food[spend_ch.food] -= 1
    engine.log(f"  convert: spend {spend_ch.food.value} for +1 egg")

    # Step 3 — lay the egg.
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
        any(bird is not None for bird in engine.state.tray)
        or engine.state.bird_deck
        or engine.state.bird_discard
    )
