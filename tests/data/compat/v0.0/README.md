# v0.0 backwards-compatibility fixtures

A pinned snapshot of a real training run's artifacts, used by
`tests/test_compat_v0_0.py` to guarantee that artifact version **0.0** files
keep loading and playing games under the current code (see "Checkpoint
compatibility policy" in `CLAUDE.md`).

| Item | Value |
|------|-------|
| Snapshot date | 2026-06-05 |
| Source | `checkpoints/` of the active `main` run |
| Source commit | `96bd70f0989d80431ec5735a4b77d985d6d9462a` |
| Written by torch | 2.12.0+cpu |
| Files | `last.pt.gz` (main net checkpoint), `setup.pt.gz` (setup net checkpoint), `model_config.json`, `setup_config.json` |

The checkpoints are gzip-compressed (`gzip -9` over the raw `torch.save`
output) and stored in **Git LFS** (see `.gitattributes`), so the weights never
bloat the repo's own history. Tests decompress fully into memory before
`torch.load` (see `_load_gzipped_checkpoint` in `test_compat_v0_0.py`). To
capture a future fixture set: `gzip -9 -k <run>/last.pt` and commit the `.gz`.

The JSON descriptors and the `.pt` payloads **deliberately lack a `version`
key** — they were written before the field existed, so they exercise the
default-to-`"0.0"` path every real pre-versioning artifact takes on load. Do
not rewrite them to add the key.

When a MINOR version bump lands, capture a **new** sibling directory
(`tests/data/compat/v<X.Y>/`) from a run at that version and keep this one —
all same-MAJOR fixture sets are retained until a MAJOR bump deletes them.
