# training.configure — Interactive "FLIGHT PLAN" configurator

Terminal UI for editing `TrainConfig`, browsing runs, and starting or archiving
training. Launched via `python -m wingspan.training --config`. Built on `rich`
for rendering and a cross-platform raw-key reader for input; no curses.

## Modules

**`__init__.py`**

**`fields.py`** — The field specification system:
- `FieldSpec` — abstract base with `attr`, `label`, `group_path: tuple[str, ...]`,
  `help`, and optional `visible_when`, `impact`, `unit`. The `group_path` encodes
  the display hierarchy at up to three levels, e.g.
  `("TRAINING", "REWARD MODEL")` or `("COLLECTION", "BOOTSTRAP", "RANDOM SETUP")`.
  Order of display is the order specs appear in `FIELD_SPECS`.
- Concrete subclasses: `IntField`, `OptionalIntField`, `FloatField`,
  `OptionalFloatField` (float with `None`=inherit, `fallback_attr` for nudge),
  `ChoiceField`, `OptionalChoiceField` (choice with `None`=inherit, cycles
  None → choices[0] → … → None), `TextField`, `PathField`, `LayersField`,
  `OptionalPathField`, `BootstrapField`.
- `FIELD_SPECS: list[FieldSpec]` — ordered list of all editable fields shown
  in the configurator. Locked-in fields (`use_distinct_hand_model`,
  `tray_set_embedding`, `dagger_expert_checkpoint`) are absent; per-block
  activation/dropout/layernorm
  overrides (14 fields) and `reward_basis` are present.
- Five top-level sections: `RUN SETTINGS`, `COLLECTION`, `EVALUATION`,
  `TRAINING`, `MODEL ARCHITECTURE`. CLONING visible when bootstrap_opponent is a
  checkpoint path; RANDOM SETUP when bootstrap=="random" and use_setup_model;
  PPO fields when policy_loss==PPO; GAE fields in delta/GAE modes.
- `read_field(cfg, spec) -> FieldValue`, `format_value(cfg, spec) -> str`,
  `commit(cfg, spec, raw) -> (TrainConfig, str | None)`,
  `nudge(cfg, spec, direction) -> (TrainConfig, str | None)`.

**`runs.py`** — Run management:
- `RunSummary(checkpoint_dir, exists, readable, train_config, …)` — compact
  snapshot of a run, its embedded config rehydrated at the artifact's era.
- `inspect_run(run_dir) -> RunSummary` — reads the run directory, keying off
  `last.pt` and rehydrating its embedded config via
  `config.run_config_from_artifact`.
- `resolve_status(summary, working) -> RunStatus` — what Start will do
  (EMPTY / RESUMABLE / INCOMPATIBLE / UNREADABLE), gated on
  `architecture_compatible` (the `architecture_key` comparison).
- `align_era(summary, working) -> TrainConfig` — re-pins the working config's
  `encoding_version` after any mutation: the saved run's era while the
  architecture still matches it (Start resumes), the live `MODEL_VERSION`
  otherwise (a fresh run never inherits a stale era). The editor-side mirror
  of `loop_resume.adopt_checkpoint_era`.
- `archive_run(run_dir, archive_dir)` — moves a run to an archive subdirectory.
- `clear_run(run_dir)` — deletes checkpoints but keeps logs.
- `list_archives(archive_dir) -> list[RunSummary]`.

**`user_defaults.py`** — The `[D] save defaults` persistence:
- `save_defaults(cfg) -> Path` / `load_defaults(current) -> LoadedDefaults` —
  write/read `./configurator_defaults.json` (cwd-anchored, checked into git).
- `EXCLUDED_FIELDS` — run-identity and derived fields that never persist
  (`encoding_version`, dims, `resume`, `checkpoint_dir`, `run_name`, `device`);
  on load these come from the caller's current config / factory defaults.
- Loading never raises: a missing file is an empty result, an unreadable or
  invalid one returns a `warning` and the caller falls back to factory
  defaults. The configurator seeds from this file when the target directory
  has no readable run, and `[R]` reset offers `user defaults` alongside
  `factory defaults`.

**`state.py`** — Configurator data model:
- `Mode` StrEnum: `CONFIG`, `RUNS`, `ARCH`, `CONFIRM`, `RUNNING`.
- `Outcome` StrEnum: `START`, `RESUME`, `QUIT`.
- `ConfirmPrompt(message, yes_outcome, no_outcome)` — modal prompt descriptor.
- `ConfiguratorState(config, mode, selected_field, runs, ...)` — the full
  immutable snapshot the screen renders from; mutations return a new instance.

**`keys.py`** — Cross-platform raw single-key reader:
- `KeyKind` StrEnum and `KeyEvent(kind, char)` Pydantic model — typed key event.
- `decode_char(ch) -> KeyEvent`, `decode_windows_special(code) -> KeyEvent`,
  `decode_unix_escape(tail) -> KeyEvent` — platform-specific decoders.
- `KeyReader` — context-manager that puts the terminal in raw mode; its
  `poll(timeout) -> KeyEvent | None` method is the non-blocking read entry point.
  Uses `msvcrt` on Windows, `termios`+`tty` on POSIX.

**`screen.py`** — `rich` layout and rendering:
- `build(view, frame) -> Layout` — builds the two-panel rich layout from the
  current `ConfiguratorState` snapshot; `frame` drives cursor blink.
- Private renderers: `_form_panel`, `_arch_panel`, `_detail`, `_header`.
  All accept the current `ConfiguratorState` and return `rich` renderables.

**`controller.py`** — Main loop and event dispatch:
- `run_configurator(config) -> (Outcome, TrainConfig)` — starts `rich.Live`,
  reads keys via `keys.py`, dispatches to `dispatch(state, key) -> ConfiguratorState`,
  and loops until an outcome is reached.
- `build_initial_state(config) -> ConfiguratorState` — console-free constructor
  for testing. Seeds from the saved run when one is readable, else from the
  user-defaults file, else factory defaults — always era-aligned via
  `runs.align_era`.
- `dispatch(state, key)` — pure state-transition function; no I/O.
- NAVIGATE keys: `[S]`tart, `[N]`ew run, `[A]`rchive, `[R]`eset (chooser:
  user defaults / factory defaults), `[D]` save current settings as defaults,
  `[Q]`uit. Every working-config mutation funnels through `_update_working`,
  which re-aligns the era and surfaces a footer notice when it moves; fresh
  launches are re-keyed at the live `MODEL_VERSION` in `_launch`.

**`arch_diagram.py`** — `ArchitectureDiagram`: a `rich` renderable that draws
the live architecture as a text-art block diagram, updated in real time as the
user nudges width fields in the configurator. `viewport(state) -> rich.text.Text`
is the public entry point.
