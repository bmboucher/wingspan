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
- `"/path/to/run/"` → run directory carrying its config descriptor
  (`run_config_<stamp>.json` for ≥0.5, legacy `model_config.json` otherwise).
- `"/path/to/checkpoint.pt"` → bare checkpoint file.
`PlayerSpec` carries the resolved `kind` enum and the raw string for error messages.

**`loaders.py`** — Both self-describing checkpoint load paths:
- `load_policy_net(path: Path, spec: EncodingSpec) -> PolicyValueNet` — loads a
  `.pt` checkpoint, checks its embedded `version` via `version.py`, rehydrates
  the embedded config at the payload's era
  (`config.run_config_from_artifact`, which reshapes a ≤0.4 flat config into the
  nested sections), and constructs the era's net class at the era's dims
  (`PolicyValueNet.class_for_version`).
- `load_policy_net_from_run_dir(run_dir: Path) -> PolicyValueNet` — reads the
  run's descriptor via `runmeta.read_model_config` (which dispatches on
  `run_config_<stamp>.json` vs legacy `model_config.json`), locates the `BEST`
  checkpoint, and delegates to `load_policy_net`.
- `encoding_key` / `descriptor_encoding_key` / `expected_encoding_key` — the
  encoding-compatibility signatures both paths verify before seating a net;
  `expected_encoding_key` derives an era's true widths via
  `compat.encoding_dims_for_era`.

**`decision_probe.py`** — `DecisionProbe`: a one-slot mailbox ferrying the critic
value *and* a `PolicyAnnotation` (probs, scores, chosen_idx) from the AI agent to
the instrumentation handler across the agent↔handler decoupling. `record(value)`
stores the critic output; `record_policy(annotation)` stores the distribution;
`take()` returns and clears both as `(float|None, PolicyAnnotation|None)`.
`PolicyAnnotation` carries two parallel encoding groups: main-net decisions populate
`state_vec` / `choice_feats` / `include_setup` / `card_embed_dim`; setup-net
decisions instead populate `setup_feats` (one raw candidate vector per choice, aligned
to `decision.choices`) and `setup_encoding` (the `SetupEncoding` describing their
layout). The other group's fields are left `None`.

**`factory.py`** — `build_agent(spec, device, rng, greedy) -> (Agent, TrainConfig|None)`:
the top-level factory. Maps each `PlayerSpec.kind` to the appropriate agent
constructor. AI policy agents record a `PolicyAnnotation` on the `DecisionProbe`
after every genuine decision (after `chosen_idx` is resolved), enabling the HTML
viewer's decision-box option bars.

Log annotation format (text log, unchanged): `[P#] <DecisionType> | N choices |
[greedy] | head:<family>` followed by indented choice lines (4 spaces). The
chose-line reads `[P#] chose: <label> (xx.xxx%)`. All log lines use
`decision.player_id` for attribution.
