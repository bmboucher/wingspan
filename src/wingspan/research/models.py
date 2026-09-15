"""Data shapes for the research studies.

The study spec and its shard plan, the four sections of a setup keep-rate
*exposure* row, the seat-deal carrier the sampler hands the scorer, and the run
summary. The CSV row is a composition of flat sections (key / observation /
bird / deal) rather than one wide model, so each section maps to a domain
object built once at its natural point: the deal context once per seat, the
bird metadata once per card, the observation once per exposure.

Imports the canonical ``wingspan.setup_model`` shapes (``SetupCandidate``,
``SetupContext``) rather than redefining them; that package is torch-free.
"""

from __future__ import annotations

import math
import typing

import pydantic

from wingspan import cards, setup_model, state
from wingspan.research import constants

#### Study spec and work plan ####


class SetupKeepStudySpec(pydantic.BaseModel):
    """What to sample and how the model picks: the knobs of one keep-rate study.

    ``temperature=None`` takes the policy's argmax keep (its play-time
    behaviour); a temperature instead samples the softmax over
    ``logits / temperature`` (its collection-time behaviour) and reports
    ``keep_prob`` at that same temperature. Games use consecutive seeds from
    ``seed`` so a study can be extended by continuing the seed range.
    ``workers > 1`` fans shards across a process pool."""

    setups: typing.Annotated[int, pydantic.Field(ge=1)] = constants.DEFAULT_SETUPS
    seed: int = constants.DEFAULT_SEED
    num_players: typing.Annotated[
        int, pydantic.Field(ge=state.MIN_PLAYERS, le=state.MAX_PLAYERS)
    ] = constants.DEFAULT_NUM_PLAYERS
    temperature: typing.Annotated[float, pydantic.Field(gt=0.0)] | None = None
    workers: typing.Annotated[int, pydantic.Field(ge=1)] = constants.DEFAULT_WORKERS

    @property
    def games(self) -> int:
        """Games to deal: every seat of a game is one setup, so this is the
        ceiling of ``setups / num_players`` (the last game's surplus seats are
        dealt but not recorded)."""
        return math.ceil(self.setups / self.num_players)

    def game_seed(self, game_index: int) -> int:
        """The RNG seed of the ``game_index``-th game of the study."""
        return self.seed + game_index

    def setup_id(self, game_index: int, seat: int) -> int:
        """The study-wide index of one seat's setup, dense from 0."""
        return game_index * self.num_players + seat


class SetupKeepShard(pydantic.BaseModel):
    """A contiguous run of games — the unit of work scored in one batched
    forward pass (and handed to one pool worker at a time)."""

    first_game_index: typing.Annotated[int, pydantic.Field(ge=0)]
    num_games: typing.Annotated[int, pydantic.Field(ge=1)]

    @property
    def game_indices(self) -> range:
        """The study-wide indices of the shard's games."""
        return range(self.first_game_index, self.first_game_index + self.num_games)


class ShardTask(pydantic.BaseModel):
    """The picklable unit handed to a pool worker: the study spec plus one shard."""

    spec: SetupKeepStudySpec
    shard: SetupKeepShard


#### Record sections (CSV column groups, in header order) ####


class ExposureKey(pydantic.BaseModel):
    """Identifies one exposure: which dealt card of which seat of which game."""

    setup_id: int
    game_seed: int
    seat: int
    num_players: int
    # Position of the card in the dealt hand (0-based).
    card_slot: int


class KeepObservation(pydantic.BaseModel):
    """What the setup model did with the exposure's card."""

    # 0/1 flag: the card is in the chosen keep — the study's observation.
    kept: int
    # The policy's marginal probability of keeping the card: the summed
    # probability of every candidate keep that contains it.
    keep_prob: float
    # Cards in the chosen keep.
    n_kept: int
    # The critic's V(s) for the deal (its predicted end-of-game margin, in the
    # net's normalized units), identical across the seat's five exposures.
    deal_value: float
    # The bonus card the keep retained; blank when the run defers the bonus
    # pick to an in-game decision (``split_setup_bonus``).
    bonus_kept: str


class BirdMetadata(pydantic.BaseModel):
    """The printed attributes of the exposure's card."""

    bird_id: int
    bird_name: str
    points: int
    egg_limit: int
    wingspan_cm: int
    nest: cards.NestType
    power_color: cards.PowerColor
    habitat_forest: int
    habitat_grassland: int
    habitat_wetland: int
    n_habitats: int
    cost_invertebrate: int
    cost_seed: int
    cost_fish: int
    cost_fruit: int
    cost_rodent: int
    cost_wild: int
    cost_total: int
    # 1 when the printed cost is an OR cost (pay any one of the listed foods).
    cost_is_or: int
    flocking: int
    predator: int
    # The parsed power's effect kinds, ``constants.NAME_SEPARATOR``-joined.
    effect_kinds: str

    @classmethod
    def from_bird(cls, bird: cards.Bird) -> "BirdMetadata":
        """Project a catalog bird onto the flat metadata columns."""
        cost = bird.food_cost
        return cls(
            bird_id=bird.id,
            bird_name=bird.name,
            points=bird.points,
            egg_limit=bird.egg_limit,
            wingspan_cm=bird.wingspan_cm,
            nest=bird.nest,
            power_color=bird.color,
            habitat_forest=int(cards.Habitat.FOREST in bird.habitats),
            habitat_grassland=int(cards.Habitat.GRASSLAND in bird.habitats),
            habitat_wetland=int(cards.Habitat.WETLAND in bird.habitats),
            n_habitats=len(bird.habitats),
            cost_invertebrate=cost.specific_of(cards.Food.INVERTEBRATE),
            cost_seed=cost.specific_of(cards.Food.SEED),
            cost_fish=cost.specific_of(cards.Food.FISH),
            cost_fruit=cost.specific_of(cards.Food.FRUIT),
            cost_rodent=cost.specific_of(cards.Food.RODENT),
            cost_wild=cost.wild,
            cost_total=cost.total,
            cost_is_or=int(cost.is_or_cost),
            flocking=int(bird.flocking),
            predator=int(bird.predator),
            effect_kinds=constants.NAME_SEPARATOR.join(
                effect.kind.value for effect in bird.power.effects
            ),
        )


class DealContext(pydantic.BaseModel):
    """The seat's view of its deal: the hand, the shared table (tray,
    birdfeeder, round goals), and the seat's bonus-card offer."""

    # Every dealt bird's name, ``constants.NAME_SEPARATOR``-joined.
    hand: str
    tray_1: str
    tray_2: str
    tray_3: str
    feeder_invertebrate: int
    feeder_seed: int
    feeder_fish: int
    feeder_fruit: int
    feeder_rodent: int
    feeder_choice: int
    goal_1: str
    goal_2: str
    goal_3: str
    goal_4: str
    goal_1_category: str
    goal_2_category: str
    goal_3_category: str
    goal_4_category: str
    bonus_1: str
    bonus_2: str

    @classmethod
    def from_state(
        cls,
        game_state: state.GameState,
        dealt_cards: typing.Sequence[cards.Bird],
        dealt_bonus: typing.Sequence[cards.BonusCard],
    ) -> "DealContext":
        """Read the table out of a freshly dealt ``GameState`` for one seat."""
        tray = [bird.name if bird is not None else "" for bird in game_state.tray]
        feeder = game_state.birdfeeder
        goals = [goal.description for goal in game_state.round_goals]
        categories = [goal.category for goal in game_state.round_goals]
        bonus_names = [bonus.name for bonus in dealt_bonus]
        return cls(
            hand=constants.NAME_SEPARATOR.join(bird.name for bird in dealt_cards),
            tray_1=_text_at(tray, 0),
            tray_2=_text_at(tray, 1),
            tray_3=_text_at(tray, 2),
            feeder_invertebrate=feeder.counts[cards.Food.INVERTEBRATE],
            feeder_seed=feeder.counts[cards.Food.SEED],
            feeder_fish=feeder.counts[cards.Food.FISH],
            feeder_fruit=feeder.counts[cards.Food.FRUIT],
            feeder_rodent=feeder.counts[cards.Food.RODENT],
            feeder_choice=feeder.choice_dice,
            goal_1=_text_at(goals, 0),
            goal_2=_text_at(goals, 1),
            goal_3=_text_at(goals, 2),
            goal_4=_text_at(goals, 3),
            goal_1_category=_text_at(categories, 0),
            goal_2_category=_text_at(categories, 1),
            goal_3_category=_text_at(categories, 2),
            goal_4_category=_text_at(categories, 3),
            bonus_1=_text_at(bonus_names, 0),
            bonus_2=_text_at(bonus_names, 1),
        )


class SetupKeepRecord(pydantic.BaseModel):
    """One exposure row of the keep-rate table: the four sections in CSV order."""

    key: ExposureKey
    observation: KeepObservation
    bird: BirdMetadata
    deal: DealContext

    def to_csv_row(self) -> dict[str, object]:
        """The flat CSV row: every section's fields merged, enums as their
        values (the header is :func:`csv_columns`)."""
        row: dict[str, object] = {}
        for section in (self.key, self.observation, self.bird, self.deal):
            row.update(section.model_dump(mode="json"))
        return row


def csv_columns() -> list[str]:
    """The keep-rate CSV header: every :class:`SetupKeepRecord` section's
    fields, in section order."""
    columns: list[str] = []
    for field in SetupKeepRecord.model_fields.values():
        section = field.annotation
        assert isinstance(section, type) and issubclass(section, pydantic.BaseModel)
        columns.extend(section.model_fields)
    return columns


#### Sampler → scorer hand-off ####


class SeatDeal(pydantic.BaseModel):
    """One seat's dealt setup inputs plus everything the scorer needs to rank
    and record its keep — built by the sampler, consumed by the scorer."""

    setup_id: int
    game_seed: int
    seat: int
    # Seeds the softmax sample when the study runs at a temperature.
    sample_seed: int
    dealt_cards: tuple[cards.Bird, ...]
    dealt_bonus: tuple[cards.BonusCard, ...]
    context: setup_model.SetupContext
    deal: DealContext
    # Every keep the setup model is offered, in the engine's enumeration order.
    candidates: tuple[setup_model.SetupCandidate, ...]


#### Run summary ####


class SetupKeepSummary(pydantic.BaseModel):
    """What a finished study wrote."""

    out_path: str
    setups: int
    games: int
    rows: int
    workers: int
    elapsed_seconds: float


###### PRIVATE #######


def _text_at(items: typing.Sequence[str], index: int) -> str:
    """``items[index]``, or blank past the end (a short tray or goal list)."""
    return items[index] if index < len(items) else ""
