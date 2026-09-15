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
