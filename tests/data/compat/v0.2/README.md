# v0.2 backwards-compatibility fixtures

A pinned snapshot of a real training run's artifacts, used by
`tests/test_compat_v0_2.py` to guarantee that artifact version **0.2** files
keep loading and playing games under the current code (see "Checkpoint
compatibility policy" in `CLAUDE.md`).

| Item | Value |
|------|-------|
| Snapshot date | 2026-06-10 |
| Source | a dedicated capture run (`games_per_iter=4`, setup net enabled, seed 20260609) through the production `TrainingLoop` |
| Captured with | the change that bumped `MODEL_VERSION` to 0.2 (card feature vector redesign, CARD_FEATURE_DIM 229 → 224) |
| Written by torch | 2.12.0+cpu |
| Files | `last.pt.gz` (main net checkpoint), `setup.pt.gz` (setup net checkpoint), `model_config.json`, `setup_config.json` |

The checkpoints are gzip-compressed (`gzip -9` over the raw `torch.save`
output) and stored in **Git LFS** (see `.gitattributes`), so the weights never
bloat the repo's own history. Tests decompress fully into memory before
`torch.load` (see `_load_gzipped_checkpoint` in `test_compat_v0_2.py`).

Every file here **carries an explicit `version: "0.2"`**, exercising the
stamped-version load path. The v0.2 era is notable for:
- `state_dim = 771` (pre-v0.3 scalar misc encoding: round ÷ 3, cubes ÷ 8)
- `choice_dim = 215` (same as v0.3)
- `CARD_FEATURE_DIM = 224` (v0.2 live card features — no card-encoder shim needed)
- Main net reconstructs as `compat.v0_2.PolicyValueNetV02` (overrides
  `encode_state` to produce the frozen 771-dim vector)

When the next MINOR version bump lands, capture a **new** sibling directory
(`tests/data/compat/v<X.Y>/`) from a run at that version and keep this one —
all same-MAJOR fixture sets are retained until a MAJOR bump deletes them.
