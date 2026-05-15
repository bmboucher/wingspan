"""Decision-point system.

Wingspan has a branching factor that's awkward to flatten. Instead the engine
exposes a sequence of *decision points*: at each, the agent is given a list of
legal choices and picks one. This makes both human CLI input and RL action
masking straightforward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .cards import Food, Habitat


class DecisionType(str, Enum):
    SETUP_KEEP_FOOD_OR_DISCARD_CARD = "setup_keep_food_or_discard_card"
    SETUP_PICK_BONUS = "setup_pick_bonus"
    MAIN_ACTION = "main_action"
    PLAY_BIRD_PICK_CARD = "play_bird_pick_card"
    PLAY_BIRD_PICK_HABITAT = "play_bird_pick_habitat"
    PLAY_BIRD_PICK_FOOD_PAYMENT = "play_bird_pick_food_payment"
    PLAY_BIRD_PICK_EGG_TO_PAY = "play_bird_pick_egg_to_pay"
    GAIN_FOOD_PICK_DIE = "gain_food_pick_die"
    LAY_EGG_PICK_BIRD = "lay_egg_pick_bird"
    DRAW_CARDS_PICK_SOURCE = "draw_cards_pick_source"   # deck or one of tray slots
    BIRD_POWER_PICK_FOOD = "bird_power_pick_food"
    BIRD_POWER_PICK_BIRD = "bird_power_pick_bird"
    BIRD_POWER_TUCK_FROM_HAND = "bird_power_tuck_from_hand"
    BIRD_POWER_PICK_STARTING_PLAYER = "bird_power_pick_starting_player"
    BIRD_POWER_PICK_HABITAT = "bird_power_pick_habitat"
    SKIP_OPTIONAL = "skip_optional"


class MainAction(str, Enum):
    PLAY_BIRD = "play_bird"
    GAIN_FOOD = "gain_food"
    LAY_EGGS = "lay_eggs"
    DRAW_CARDS = "draw_cards"


@dataclass
class Choice:
    """A single legal option presented at a decision point.

    The ``payload`` is opaque to the agent; it's what the engine consumes. The
    ``label`` is a short human description used by the CLI and the game log.
    """
    label: str
    payload: Any
    # An optional integer index used by the RL encoder to map a choice into a
    # globally consistent action slot. Not all decisions need this.
    encoded: Optional[int] = None


@dataclass
class Decision:
    type: DecisionType
    player_id: int
    prompt: str
    choices: list[Choice]
    # arbitrary engine-side context that needs to flow through to the resolver
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError(f"Decision {self.type} produced no legal choices")
