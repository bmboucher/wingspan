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
  `.pt` checkpoint, checks its embedded `version` via `version.py`, applies any
  compat shim from `wingspan.compat`, and constructs the net from the embedded
  `model_config`.
- `load_policy_net_from_run_dir(run_dir: Path) -> PolicyValueNet` — reads
  `model_config.json` from a run directory, locates the `BEST` checkpoint, and
  delegates to `load_policy_net`.
- `encoding_compat_keys(descriptor) -> set[str]` — the `architecture_key`
  components that must match for resume (used by the training loop's graceful
  FRESH restart).

**`factory.py`** — `player_from_spec(spec: PlayerSpec, log: bool) -> Agent`:
the top-level factory. Maps each `PlayerSpec.kind` to the appropriate agent
constructor and wraps policy agents in a logging decorator when `log=True` (so
`wingspan play` can write a game log). Also `resolve_regime(run_dir) -> Regime`
— reads the latest `process_*.json` to determine whether a run is in the
bootstrap phase or the self-play phase.
