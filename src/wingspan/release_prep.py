"""Maintainer tool: strip a trained run into the bundled release artifact.

Usage::

    python -m wingspan.release_prep --checkpoint-dir <run dir> \\
        [--source best|last] [--out src/wingspan/data/release]

Writes ``last.pt`` (the chosen checkpoint minus optimizer state and metrics),
``setup.pt`` (minus its optimizer), the newest ``run_config_*.json`` (copied
unchanged; the setup-net loader reads it as its descriptor) and a
``manifest.json``, then round-trips the result through the same
``players.loaders`` path end users hit, failing loudly on any error.
"""

from __future__ import annotations

import argparse
import datetime
import pathlib
import shutil
import typing

import torch

from wingspan import release
from wingspan.players import loaders
from wingspan.training import artifacts

DEFAULT_OUT = pathlib.Path("src/wingspan/data/release")
_SOURCES = {"best": artifacts.BEST_CKPT, "last": artifacts.LAST_CKPT}
# Payload keys dropped from the bundled copies (training-only state).
_MAIN_DROP_KEYS = ("optimizer", "metrics")
_SETUP_DROP_KEYS = ("setup_optimizer",)


def strip_payload(
    source: pathlib.Path, dest: pathlib.Path, drop_keys: typing.Iterable[str]
) -> dict[str, typing.Any]:
    """Load the checkpoint at ``source``, drop ``drop_keys``, save to ``dest``,
    and return the stripped payload."""
    payload = typing.cast(
        "dict[str, typing.Any]",
        torch.load(source, map_location="cpu", weights_only=False),
    )
    for key in drop_keys:
        payload.pop(key, None)
    torch.save(payload, dest)
    return payload


def newest_run_config(checkpoint_dir: pathlib.Path) -> pathlib.Path:
    """The newest ``run_config_<stamp>.json`` in ``checkpoint_dir`` (the stamp
    sorts chronologically). Raises ``FileNotFoundError`` when there is none."""
    matches = sorted(checkpoint_dir.glob(artifacts.RUN_CONFIG_GLOB))
    if not matches:
        raise FileNotFoundError(f"No run_config_*.json in {checkpoint_dir}")
    return matches[-1]


def build_manifest(
    payload: dict[str, typing.Any], checkpoint_dir: pathlib.Path, source_file: str
) -> release.ReleaseManifest:
    """The manifest describing a stripped main-checkpoint ``payload``."""
    progress = payload.get("progress")
    iteration: int | None = None
    if isinstance(progress, dict):
        raw_iteration = typing.cast("dict[str, typing.Any]", progress).get("iteration")
        if raw_iteration is not None:
            iteration = int(raw_iteration)
    git_sha = payload.get("git_sha")
    return release.ReleaseManifest(
        era=str(payload["version"]),
        run_name=checkpoint_dir.name,
        source_file=source_file,
        iteration=iteration,
        git_sha=str(git_sha) if git_sha else None,
        created=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    )


def verify(out_dir: pathlib.Path) -> None:
    """Round-trip the written artifacts through the production loaders; raises
    on any failure."""
    device = torch.device("cpu")
    loaders.load_policy_net(out_dir / artifacts.LAST_CKPT, device)
    if (out_dir / artifacts.SETUP_CKPT).exists():
        if loaders.load_setup_net(out_dir, device) is None:
            raise RuntimeError("Bundled setup.pt did not load (descriptor missing?)")


def prepare(
    checkpoint_dir: pathlib.Path, source: str, out_dir: pathlib.Path
) -> release.ReleaseManifest:
    """Build the release bundle in ``out_dir`` from ``checkpoint_dir`` and
    verify it. Returns the written manifest."""
    source_file = _SOURCES[source]
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = strip_payload(
        checkpoint_dir / source_file, out_dir / artifacts.LAST_CKPT, _MAIN_DROP_KEYS
    )
    setup_src = checkpoint_dir / artifacts.SETUP_CKPT
    if setup_src.exists():
        strip_payload(setup_src, out_dir / artifacts.SETUP_CKPT, _SETUP_DROP_KEYS)
    run_config = newest_run_config(checkpoint_dir)
    shutil.copyfile(run_config, out_dir / run_config.name)
    manifest = build_manifest(payload, checkpoint_dir, source_file)
    (out_dir / release.MANIFEST_FILE).write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    verify(out_dir)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m wingspan.release_prep",
        description="Strip a trained run into the bundled release artifact.",
    )
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--source", choices=sorted(_SOURCES), default="best")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    manifest = prepare(args.checkpoint_dir, args.source, args.out)
    print(f"Wrote release bundle to {args.out}: {manifest.model_dump_json()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
