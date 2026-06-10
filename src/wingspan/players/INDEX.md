# players — Seat players from CLI specs

Resolves player-spec strings (e.g. `human`, `random`, `my-run`, `/path/to/checkpoint.pt`)
to `Agent` instances. Shared by `wingspan play` and the tournament runner so
both use the same loading and logging conventions.

## Modules

**`__init__.py`**

**`spec.py`** — The player-spec grammar: `PlayerSpec` Pydantic model and
`parse_spec(s: str) -> PlayerSpec`. Recognized forms:
- `"human"` → interactive CLI agent.
- `"random"` → uniform-random agent.
- `"<name>"` → named run in the default checkpoint directory.
- `"/path/to/run/"` → run directory with `model_config.json`.
- `"/path/to/checkpoint.pt"` → bare checkpoint file.
`PlayerSpec` carries the resolved `kind` enum and the raw string for error messages.

**`loaders.py`** — Both self-describing checkpoint load paths:
- `load_policy_net(path: Path, spec: EncodingSpec) -> PolicyValueNet` — loads a
  `.pt` checkpoint, checks its embedded `version` via `version.py`, rehydrates
  the embedded config at the payload's era
  (`config.train_config_from_artifact`), and constructs the era's net class at
  the era's dims (`PolicyValueNet.class_for_version`).
- `load_policy_net_from_run_dir(run_dir: Path) -> PolicyValueNet` — reads
  `model_config.json` from a run directory, locates the `BEST` checkpoint, and
  delegates to `load_policy_net`.
- `encoding_key` / `descriptor_encoding_key` / `expected_encoding_key` — the
  encoding-compatibility signatures both paths verify before seating a net;
  `expected_encoding_key` derives an era's true widths via
  `compat.encoding_dims_for_era`.

**`factory.py`** — `build_agent(spec, device, rng, greedy) -> (Agent, TrainConfig|None)`:
the top-level factory. Maps each `PlayerSpec.kind` to the appropriate agent
constructor. Policy agents write a per-decision annotation to the game log at every
genuine decision.

Log annotation format: `[P#] <DecisionType> | N choices | [greedy] | head:<family>`
followed by indented choice lines (4 spaces). The chose-line reads
`[P#] chose: <label> (xx.xxx%)`. All log lines use `decision.player_id` for
attribution — correct even when a pink power triggers the opponent's decision.
