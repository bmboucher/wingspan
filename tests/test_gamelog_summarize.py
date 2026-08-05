"""Tests for the event summarizer — the shared header text both renderers use.

Two halves.  The first builds small hand-made event trees and asserts the exact
prose, because the header wording is the feature: it is what makes a turn
readable at a glance.  The second plays real games and asserts invariants that
must hold for every event any seeded game can produce.
"""

from __future__ import annotations

import random
import sys
import typing

# Add src/ to path as per test_smoke.py convention.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent / "src"))

from wingspan.gamelog import models, summarize  # noqa: E402


def _gain(player_id: int, food: str, amount: int = 1) -> models.GainFoodEffect:
    """A food gain from the birdfeeder, for the hand-built trees below."""
    return models.GainFoodEffect(
        player_id=player_id,
        food=food,
        amount=amount,
        source=models.FoodSource.FEEDER,
    )


def _decision(player_id: int, text: str) -> models.DecisionSubEvent:
    """A resolved decision carrying only the fields the summarizer reads."""
    return models.DecisionSubEvent(player_id=player_id, outcome_text=text)


def _all_events(
    tree: models.GameEventTree,
) -> list[models.GameEvent]:
    """Every event in the tree, flattened depth-first."""
    found: list[models.GameEvent] = []

    def walk(events: typing.Sequence[models.GameEvent]) -> None:
        for event in events:
            found.append(event)
            walk(event.children)

    for phase in tree.phases:
        walk(phase.events)
    return found


def _play_tree(seed: int = 42, num_players: int = 2) -> models.GameEventTree:
    """Run one random game and return its recorded event tree."""
    from wingspan import agents
    from wingspan import engine as engine_mod
    from wingspan.gamelog import recorder as gamelog_recorder

    rec = gamelog_recorder.EventRecorder(probes=(None,) * num_players)
    eng, *_ = engine_mod.Engine.create(seed=seed, num_players=num_players)
    rng = random.Random(seed)
    engine_mod.Engine.play_one_game(
        eng.state,
        tuple(agents.random_agent(rng) for _ in range(num_players)),
        event_recorder=rec,
    )
    return rec.root


#### Food rendering ####


def test_food_words_repeats_whole_words():
    """Two fish render as two whole words, so the HTML emoji pass sees both."""
    assert summarize.food_words({"fish": 2}) == "fish fish"


def test_food_words_switches_to_multiplier_when_long():
    """Past a handful of tokens a multiplier replaces the repetition."""
    words = summarize.food_words({"seed": 6})
    assert words == "6x seed"
    # The space keeps 'seed' on its own word boundary for the emoji pass.
    assert " seed" in words


def test_food_words_preserves_recording_order():
    """Food types appear in the order they were recorded, not sorted."""
    assert summarize.food_words({"rodent": 1, "fish": 1}) == "rodent fish"


def test_food_words_skips_empty_counts():
    """A zeroed entry contributes nothing."""
    assert summarize.food_words({"fish": 0, "seed": 1}) == "seed"


#### Folding ####


def test_summary_folds_descendant_effects():
    """An event's header rolls up effects recorded under its children too."""
    event = models.ActivateBaseEvent(
        player_id=0,
        habitat="forest",
        action="gain_food",
        sub_events=[_gain(0, "fish")],
        children=[
            models.ActivateBrownEvent(
                player_id=0,
                bird_name="Inner",
                is_brown=True,
                sub_events=[_gain(0, "fish")],
            )
        ],
    )
    assert summarize.summarize(event).food_gained == {"fish": 2}


def test_summary_excludes_another_seats_effects():
    """A pink reaction nested under the play that triggered it stays out of
    the acting player's header — its food went to the other seat."""
    event = models.PlayBirdEvent(
        player_id=0,
        sub_events=[
            models.PlayBirdEffect(
                player_id=0, card="Cooper's Hawk", habitat="forest", slot=0
            )
        ],
        children=[
            models.ReactionEvent(
                player_id=1,
                bird_name="Turkey Vulture",
                sub_events=[_gain(1, "rodent")],
            )
        ],
    )
    summary = summarize.summarize(event)
    assert summary.food_gained == {}, "another seat's gain leaked into the header"
    assert summarize.summary_text(event) == "Plays Cooper's Hawk in Forest"


def test_seatless_effects_belong_to_their_event():
    """A tray refill carries no seat, so it folds into whatever event opened it."""
    event = models.RefillTrayEvent(
        player_id=0,
        sub_events=[models.TrayRefillEffect(slot=1, card="Ruddy Duck")],
    )
    assert summarize.summary_text(event) == "Tray refill: Ruddy Duck"


def test_empty_summary_is_empty():
    """An event that recorded nothing has an empty summary."""
    event = models.ActivateBrownEvent(player_id=0, bird_name="X", is_brown=False)
    assert summarize.summarize(event).is_empty


def test_nonempty_summary_is_not_empty():
    """Any recorded effect makes the summary non-empty."""
    event = models.ActivateBrownEvent(
        player_id=0, bird_name="X", is_brown=True, sub_events=[_gain(0, "seed")]
    )
    assert not summarize.summarize(event).is_empty


#### Phrase composition ####


def test_effect_phrase_orders_clauses_causally():
    """A tuck-to-draw power reads in the order it happened, not alphabetically."""
    event = models.ActivateBrownEvent(
        player_id=0,
        bird_name="Yellow-Rumped Warbler",
        is_brown=True,
        sub_events=[
            models.TuckCardEffect(
                player_id=0,
                card="Black-Billed Magpie",
                bird="Yellow-Rumped Warbler",
                source=models.CardSource.HAND,
            ),
            models.DrawCardEffect(
                player_id=0, card="Wood Stork", source=models.CardSource.DECK
            ),
        ],
    )
    assert summarize.summary_text(event) == (
        "Yellow-Rumped Warbler (brown): Tucks Black-Billed Magpie, draws Wood Stork"
    )


def test_effect_phrase_pluralizes_eggs():
    """Egg counts read as '1 egg' / '2 eggs'."""

    def egg_event(count: int) -> models.ActivateBaseEvent:
        return models.ActivateBaseEvent(
            player_id=0,
            habitat="grassland",
            action="lay_eggs",
            sub_events=[
                models.LayEggEffect(
                    player_id=0, bird="X", habitat="grassland", slot=0, count=count
                )
            ],
        )

    assert summarize.summary_text(egg_event(1)) == "Lays 1 egg"
    assert summarize.summary_text(egg_event(2)) == "Lays 2 eggs"


#### Per-event headers ####


def test_main_action_header_names_the_habitat():
    """The main-action header names the row the action cube activates."""
    event = models.MainActionEvent(player_id=0, action="gain_food")
    assert summarize.summary_text(event) == "Main action: Forest (gain food)"


def test_main_action_header_without_a_stamped_action():
    """An unstamped main action still gets a truthful header."""
    assert summarize.summary_text(models.MainActionEvent(player_id=0)) == "Main action"


def test_powerless_bird_still_gets_a_header():
    """A bird with no brown power is named explicitly rather than omitted."""
    event = models.ActivateBrownEvent(
        player_id=0, bird_name="Turkey Vulture", is_brown=False
    )
    assert summarize.summary_text(event) == "Turkey Vulture — no brown power"


def test_power_header_falls_back_to_its_decision():
    """A power that changed nothing but was decided reports the decision.

    Without this it would read 'no effect', hiding that a choice was made."""
    event = models.WhitePowerEvent(
        player_id=0,
        bird_name="California Condor",
        sub_events=[_decision(0, "Declines")],
    )
    assert summarize.summary_text(event) == "California Condor (white): Declines"


def test_power_header_reports_no_effect_when_nothing_happened():
    """A power that neither decided nor changed anything says so."""
    event = models.WhitePowerEvent(player_id=0, bird_name="California Condor")
    assert summarize.summary_text(event) == "California Condor (white) — no effect"


def test_idle_habitat_action_names_itself():
    """An activation that produced nothing falls back to naming the action."""
    event = models.ActivateBaseEvent(
        player_id=0, habitat="grassland", action="lay_eggs"
    )
    assert summarize.summary_text(event) == "Lay eggs — no effect"


def test_extra_play_header_names_its_habitat_restriction():
    """A habitat-restricted extra play says which habitat."""
    assert (
        summarize.summary_text(models.ExtraPlayEvent(player_id=0, habitat="wetland"))
        == "Extra play (Wetland)"
    )
    assert summarize.summary_text(models.ExtraPlayEvent(player_id=0)) == "Extra play"


def test_scoring_headers_carry_their_seat_data():
    """Round-goal and final-scoring headers keep their per-seat numbers."""
    goal = models.RoundGoalEvent(
        round_idx=0, description="eggs_forest", counts=[3, 1], vps=[4, 1]
    )
    assert summarize.summary_text(goal) == (
        "Round 1 goal — eggs_forest [P0: 3/4VP, P1: 1/1VP]"
    )
    final = models.FinalScoringEvent(
        scores=[
            models.FinalScoreBreakdown(total=71),
            models.FinalScoreBreakdown(total=68),
        ]
    )
    assert summarize.summary_text(final) == "Final scoring [71, 68]"


#### Reveals ####


def test_deck_draw_is_a_reveal_and_names_the_card():
    """A deck draw is the only record of which card it was."""
    effect = models.DrawCardEffect(
        player_id=0, card="Wood Stork", source=models.CardSource.DECK
    )
    assert summarize.is_reveal(effect)
    assert summarize.reveal_text(effect) == "Draws Wood Stork from the deck"


def test_tray_draw_is_not_a_reveal():
    """The tray is face up — taking from it discloses nothing."""
    effect = models.DrawCardEffect(
        player_id=0, card="Mallard", source=models.CardSource.TRAY, tray_slot=1
    )
    assert not summarize.is_reveal(effect)


def test_deck_tuck_is_a_reveal():
    """A card tucked straight off the deck is seen nowhere else."""
    effect = models.TuckCardEffect(
        player_id=0,
        card="Bell's Vireo",
        bird="Cooper's Hawk",
        source=models.CardSource.DECK,
    )
    assert summarize.is_reveal(effect)
    assert "Bell's Vireo" in summarize.reveal_text(effect)


def test_hand_tuck_is_not_a_reveal():
    """Tucking from hand moves a card the owner already knew about."""
    effect = models.TuckCardEffect(
        player_id=0,
        card="Bell's Vireo",
        bird="Cooper's Hawk",
        source=models.CardSource.HAND,
    )
    assert not summarize.is_reveal(effect)


def test_dice_reveals_list_their_faces():
    """Both dice reveals name every rolled face."""
    reroll = models.FeederRerollEffect(player_id=0, faces=["seed", "fish"])
    assert summarize.is_reveal(reroll)
    assert summarize.reveal_text(reroll) == "Birdfeeder rerolled: seed fish"

    roll = models.DiceRollEffect(player_id=0, bird="Anhinga", faces=["rodent"])
    assert summarize.is_reveal(roll)
    assert summarize.reveal_text(roll) == "Anhinga rolls: rodent"


def test_food_gain_is_not_a_reveal():
    """Ordinary bookkeeping folds into a header instead of earning a row."""
    assert not summarize.is_reveal(_gain(0, "fish"))
    assert summarize.reveal_text(_gain(0, "fish")) == ""


#### Whole-game invariants ####


def test_every_event_gets_a_nonempty_header():
    """No event in a real game may render a blank or placeholder header.

    The failure this guards against is a new event subclass falling through to
    a class-name fallback — the drift that used to leak '[MainActionEvent]'
    into the log."""
    for seed in (7, 42):
        for event in _all_events(_play_tree(seed=seed)):
            header = summarize.summary_text(event)
            assert header.strip(), f"{type(event).__name__} rendered a blank header"
            assert (
                type(event).__name__ not in header
            ), f"{type(event).__name__} header leaked its class name: {header!r}"


def test_main_action_events_are_stamped_with_their_action():
    """Every main-action event a real game records knows which action won."""
    events = [
        event
        for event in _all_events(_play_tree(seed=42))
        if isinstance(event, models.MainActionEvent)
    ]
    assert events, "no main-action events recorded"
    unstamped = [event for event in events if event.action is None]
    assert not unstamped, f"{len(unstamped)} main-action events left unstamped"


def test_headers_hold_at_four_seats():
    """The summarizer is seat-count agnostic, like the rest of the log."""
    events = _all_events(_play_tree(seed=101, num_players=4))
    assert events
    for event in events:
        assert summarize.summary_text(event).strip()


def test_deck_draw_reveals_reach_the_tree():
    """A real game records the identity of cards drawn from the deck."""
    reveals = [
        sub
        for event in _all_events(_play_tree(seed=42))
        for sub in event.sub_events
        if isinstance(sub, models.Effect) and summarize.is_reveal(sub)
    ]
    assert reveals, "no reveal effects recorded across a full game"
    deck_draws = [
        sub
        for sub in reveals
        if isinstance(sub, models.DrawCardEffect)
        and sub.source is models.CardSource.DECK
    ]
    assert deck_draws, "no deck draws recorded — the reveal the log exists for"
