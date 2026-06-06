# v0.1 backwards-compatibility fixtures

A pinned snapshot of a real training run's artifacts, used by
`tests/test_compat_v0_1.py` to guarantee that artifact version **0.1** files
keep loading and playing games under the current code (see "Checkpoint
compatibility policy" in `CLAUDE.md`).

| Item | Value |
|------|-------|
| Snapshot date | 2026-06-06 |
| Source | a dedicated 4-iteration capture run (`games_per_iter=4`, setup net recorded from iteration 0 and fit at iteration 2, seed 20260606) through the production `TrainingLoop` |
| Captured with | the change that bumped `MODEL_VERSION` to 0.1 (the landing-slot / index-column choice-vector reshape) |
| Written by torch | 2.12.0+cpu |
| Files | `last.pt.gz` (main net checkpoint), `setup.pt.gz` (setup net checkpoint), `model_config.json`, `setup_config.json` |

The checkpoints are gzip-compressed (`gzip -9` over the raw `torch.save`
output) and stored in **Git LFS** (see `.gitattributes`), so the weights never
bloat the repo's own history. Tests decompress fully into memory before
`torch.load` (see `_load_gzipped_checkpoint` in `test_compat_v0_1.py`).

Unlike the v0.0 set, every file here **carries an explicit `version: "0.1"`**
(the field exists in this era), so these fixtures exercise the stamped-version
load path while `tests/data/compat/v0.0/` keeps covering the default-to-`"0.0"`
path.

When the next MINOR version bump lands, capture a **new** sibling directory
(`tests/data/compat/v<X.Y>/`) from a run at that version and keep this one —
all same-MAJOR fixture sets are retained until a MAJOR bump deletes them.
