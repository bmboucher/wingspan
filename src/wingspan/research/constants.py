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

# ``wingspan research card-stats`` file names and shard sizing.
CARD_GAMES_RAW_FILENAME = "card_games_raw.csv"
BIRD_STATS_FILENAME = "bird_stats.csv"
BONUS_STATS_FILENAME = "bonus_stats.csv"
BIRD_STATS_MARKDOWN_FILENAME = "bird_stats.md"
BONUS_STATS_MARKDOWN_FILENAME = "bonus_stats.md"
BONUS_HIST_FILENAME = "bonus_hist.csv"
# The ``card_games_raw.csv`` column holding every bonus card held at game end.
FINAL_BONUS_COLUMN = "final_bonus"
# Splits one ``final_bonus`` entry into ``name:count:points`` (no bonus card
# name contains it); entries are joined by ``NAME_SEPARATOR``.
FINAL_BONUS_ENTRY_SEPARATOR = ":"
# Games per unit of pool work (each game is a full greedy self-play game).
CARD_STATS_GAMES_PER_SHARD = 8
# Number of rounds a game has (one per entry of ``state.ROUND_CUBES``).
NUM_ROUNDS = len(ROUND_TURNS)
# Two-sided 95% normal quantile for the Wilson interval of a win rate.
Z_95 = 1.96
# Numbered-column prefixes of ``card_games_raw.csv`` (suffix ``1..NUM_ROUNDS``).
BIRDS_ROUND_PREFIX = "birds_r"
ACQUIRED_BONUS_ROUND_PREFIX = "acquired_bonus_r"

# ``wingspan research forced-card``: the bird forced by default, the factor
# sweep run when none is given, file names, and the per-seat RNG seed format.
DEFAULT_FORCED_CARD_NAME = "Cedar Waxwing"
DEFAULT_FORCE_CARD_FACTORS = (0.0, 0.25, 0.5, 0.75, 1.0)
# Inclusive range of the bias factor ``x``: negative values avoid the forced
# card (``-1`` never picks it when an alternative exists), positive values push
# toward it (``1`` always does).
MIN_FORCE_CARD_FACTOR = -1.0
MAX_FORCE_CARD_FACTOR = 1.0
FORCED_CARD_RAW_FILENAME = "forced_card_raw.csv"
FORCED_CARD_SUMMARY_FILENAME = "forced_card_summary.csv"
FORCED_CARD_SUMMARY_MARKDOWN_FILENAME = "forced_card_summary.md"
# Per-factor sub-directory name; the factor is rendered with this many decimals
# (negatives keep their sign, e.g. ``factor_-0.500``).
FORCED_CARD_DIR_TEMPLATE = "factor_{factor:.3f}"
# Prefix of the per-factor directory name (the factor is parsed after it).
FORCED_CARD_DIR_PREFIX = "factor_"
# Paired-comparison outputs of ``forced-card-summary`` and its default baseline.
FORCED_CARD_PAIRED_FILENAME = "forced_card_paired.csv"
FORCED_CARD_PAIRED_MARKDOWN_FILENAME = "forced_card_paired.md"
DEFAULT_BASELINE_FACTOR = 0.0
# Two-sided 95% critical value of the standard normal, for paired-mean intervals.
NORMAL_95_Z = 1.96
# Each seat's sampling RNG is seeded by this string formatted with the game
# seed and the seat, so a game replays identically at every factor.
FORCED_CARD_RNG_SEED_TEMPLATE = "forced-card:{seed}:{seat}"

# ``wingspan research cards-html``: card size and output sub-directories.
CARD_WIDTH_PX = 240
CARD_HEIGHT_PX = 340
CARD_HTML_BIRD_DIR = "bird"
CARD_HTML_BONUS_DIR = "bonus"
