"""Public constants shared across the research package: CLI defaults, shard
sizing for the process pool, and CSV cell conventions."""

from wingspan import state

# The checkpoint directory a study resolves its model from when none is given
# — the active run's root, matching ``wingspan play``'s default.
DEFAULT_CHECKPOINT_DIR = "checkpoints"

# ``wingspan research setup-keep`` defaults. ``DEFAULT_NUM_PLAYERS`` only
# applies to programmatic specs; the CLI reads the run's trained seat count.
DEFAULT_SETUPS = 1000
DEFAULT_SEED = 0
DEFAULT_NUM_PLAYERS = 2
DEFAULT_WORKERS = 1
DEFAULT_SETUP_KEEP_OUT = "setup_keep.csv"

# Games dealt per unit of pool work. Every seat's candidate set of a shard is
# scored in one batched forward pass, so this bounds both the pipe payload per
# task and the transient feature matrix (32 games x 5 seats x 504 candidates x
# ~500 dims of float32 is ~160 MB in the un-split worst case).
GAMES_PER_SHARD = 32

# Joins multi-valued cells (the dealt hand, a bird's effect kinds) inside one
# CSV column — a character no card name or effect tag contains.
NAME_SEPARATOR = "|"

# ``wingspan research game-stats`` defaults and file names.
DEFAULT_GAMES = 10_000
GAMES_RAW_FILENAME = "games_raw.csv"
# Games per unit of pool work (each game is a full greedy self-play game).
GAME_STATS_GAMES_PER_SHARD = 8
# Minimum seconds between progress lines: an in-place rewrite on a terminal,
# an appended line when stderr is redirected to a log.
PROGRESS_TTY_INTERVAL_SECONDS = 1.0
PROGRESS_LOG_INTERVAL_SECONDS = 30.0

# Action cubes (= main actions = turns) each seat spends over a whole game:
# 8 + 7 + 6 + 5 across the four rounds (``state.ROUND_CUBES``).
ROUND_TURNS = tuple(state.ROUND_CUBES)
TURNS_PER_GAME = sum(ROUND_TURNS)

# Numbered-column prefixes of ``games_raw.csv``.
SCORE_PROJ_PREFIX = "score_proj_"
SCORE_RAW_PREFIX = "score_raw_"
ACTION_PREFIX = "action_"
