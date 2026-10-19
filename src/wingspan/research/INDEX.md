# research — Offline research studies over trained checkpoints

Fixed-model analysis jobs for the projects in `docs/RESEARCH.md`: sample game
situations, score them through a loaded network (no training), and write a flat,
pivotable CSV. Exposed as the `wingspan research <study>` CLI verb. Every study
is planned as seed-pure shards so the in-process path and the `--workers`
process pool write byte-identical output.

## Modules

**`__init__.py`**

**`constants.py`** — CLI defaults (`DEFAULT_CHECKPOINT_DIR`, `DEFAULT_SETUPS`,
`DEFAULT_SEED`, `DEFAULT_NUM_PLAYERS`, `DEFAULT_WORKERS`,
`DEFAULT_SETUP_KEEP_OUT`), `GAMES_PER_SHARD` (games per unit of pool work; every
seat's candidate set of a shard is scored in one batched forward pass), and
`NAME_SEPARATOR` (joins multi-valued CSV cells such as the dealt hand).

**`models.py`** — The package's Pydantic shapes:
- `SetupKeepStudySpec(setups, seed, num_players, temperature, workers)` — one
  keep-rate study's knobs. `temperature=None` takes the policy's argmax keep; a
  temperature samples the softmax over `logits / temperature` and reports
  `keep_prob` at that temperature. `games` (ceil of `setups / num_players`),
  `game_seed(game_index)` (consecutive seeds from `seed`, so a study extends by
  continuing the seed range), `setup_id(game_index, seat)`.
- `SetupKeepShard(first_game_index, num_games)` / `ShardTask(spec, shard)` —
  the unit of batched scoring and the picklable pool task.
- The four CSV **record sections**, in header order: `ExposureKey(setup_id,
  game_seed, seat, num_players, card_slot)`; `KeepObservation(kept, keep_prob,
  n_kept, deal_value, bonus_kept)`; `BirdMetadata` (points, egg limit,
  wingspan, nest, power color, per-habitat flags, per-food + wild cost columns,
  `cost_is_or`, flocking, predator, `effect_kinds`; `from_bird`); `DealContext`
  (the `hand`, `tray_1..3`, six `feeder_*` die counts, `goal_1..4` with their
  categories, `bonus_1` / `bonus_2`; `from_state`).
- `SetupKeepRecord(key, observation, bird, deal)` — one exposure row;
  `to_csv_row()` merges the sections (enums as values). `csv_columns()` is the
  matching header.
- `SeatDeal` — the sampler → scorer hand-off: one seat's dealt cards / bonus
  offer, its `setup_model.SetupContext`, its `DealContext`, its enumerated
  candidate keeps, and the seed of its softmax sample.
- `SetupKeepSummary(out_path, setups, games, rows, workers, elapsed_seconds)`.

**`setup_keep.py`** — The setup keep-rate experience study (RESEARCH.md "Setup
card stats", Q1). `run_setup_keep_study(spec, checkpoint_dir, device, out_path)
-> SetupKeepSummary` loads the run's setup net (`load_setup_net`, which raises
rather than degrading to random picks), plans shards (`plan_shards`), and
streams each shard's rows to the CSV in plan order — scored in-process for
`workers == 1`, otherwise by a spawn-context `ProcessPoolExecutor` whose workers
each load the net once (`torch.set_num_threads(1)`, mirroring `mp_collect`).
`study_shard(spec, shard, net, device)` is the pure unit of work: `deal_shard`
deals every game of the shard through `state.new_game` +
`Engine.deal_setup_inputs` (so each seat sees exactly what an agent would —
tray, feeder roll, goals, five birds, two bonus cards) and enumerates its
candidate keeps at the net's own split regime; all candidates of the shard are
encoded through `net.encode_candidate` and scored in one `policy_and_value`
pass; each seat's keep is the argmax (or a softmax sample at the study
temperature), `kept` flags each dealt card's membership in it, `keep_prob` is
the card's marginal over the candidate distribution, and `deal_value` is the
critic's V(s).

**`app.py`** — `main(argv)` for `wingspan research`. Sub-command `setup-keep`
(`--setups`, `--out`, `--checkpoint-dir`, `--seed`, `--temperature`,
`--workers`, `--device`, `--num-players`); the seat count defaults to the run
config's trained `num_players`. A run without a setup model exits 1 with a
message on stderr.

**`game_stats.py`** — The model-performance study (RESEARCH.md "Model
performance stats"). `run_game_stats_study(spec, checkpoint_dir, device,
out_path, *, progress_stream=None) -> GameStatsSummary` loads the run's `last.pt` policy net, its setup
net, and its regime (`load_game_nets` -> `GameNets`: nets, device, seat count,
`split_setup_bonus` / `split_setup_food` / `combine_gain_food`), plans shards
(`plan_shards`), and streams per-seat rows to `games_raw.csv` in shard order —
in-process for `workers == 1`, otherwise a spawn-context pool whose workers each
load the nets once (`torch.set_num_threads(1)`). `study_shard` / `play_game` are
the pure units of work: one greedy self-play game per index, seeded by
`spec.game_seed(index)` alone, the same `policy.greedy_agent(net, device,
setup_net=...)` on every seat, captured by `GameStatsCapture`.

**`progress.py`** — Live progress for shard-streamed studies.
`StudyProgress(total_games, stream, now=time.monotonic)` with `update(games_done)`
(throttled: `PROGRESS_TTY_INTERVAL_SECONDS` for an in-place `\r` line on a TTY,
`PROGRESS_LOG_INTERVAL_SECONDS` for appended lines otherwise) and
`finish(games_done)` (always prints). `format_progress(done, total, elapsed)` is
the pure status-line builder (count, percent, games/s, elapsed, ETA from the
overall average rate). `game-stats` passes `sys.stderr`; setup-keep does not use
it yet.

**`game_stats_agg.py`** — Aggregates and the HTML preview, from the raw CSV
(numpy only; re-runnable via `--aggregate-only`). `aggregate(raw_csv, out_dir, with_blog_snippets=False)`
writes `breakdown_table.csv` / `.md`, `turn_scores.csv`, `action_freq.csv`,
`game_pairs.csv`, and `preview.html` (five Plotly figures: table, histogram,
turn-score line with a p5-p95 percentile band — the CI columns stay in
`turn_scores.csv` — paired-scores scattergram, action bars; via the pinned CDN
URL from `reporting.training_summary`). Cohorts: `all` and `won` (sole winners; ties
excluded). `with_blog_snippets` (CLI `--blog-snippets`) also writes
`blog_snippets/` — the pinned CDN script tag plus one ready-to-paste embed block
per figure (`pair_traces` thins the scattergram to 10k markers
deterministically; legend counts stay full-data). Building blocks:
`read_raw_games`, `breakdown_rows`, `breakdown_markdown`, `turn_score_rows`,
`action_freq_rows`, `game_pair_rows`, `pair_traces`, `preview_html`,
`blog_snippets`.

**Game-stats models** (in `models.py`): `GameStatsSpec(games, seed, workers)`
with `game_seed(index)`; `GameStatsShard` / `GameStatsTask`;
`GameSeatRecord` (outcome, six-way breakdown, `score_proj` / `score_raw` of 27
entries, `actions` of 26; `to_csv_row()`) with `game_stats_csv_columns()` (the
stable header: scalar columns, `score_proj_0..26`, `score_raw_0..26`,
`action_1..26`); `GameStatsSummary`; the `Cohort` / `ScoreConvention` enums and
the aggregate row shapes `BreakdownRow`, `TurnScoreRow`, `ActionFreqRow`,
`GamePairRow`.

**`app.py` (game-stats)** — sub-command `game-stats` (`--games`, `--out` (a
directory, required), `--checkpoint-dir`, `--seed`, `--workers`, `--device`,
`--aggregate-only GAMES_RAW_CSV`); a play run always aggregates afterwards and
prints games/sec.
