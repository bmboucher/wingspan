"""Container entry point: load a YAML run-file and run it headless with S3 sync.

``python -m wingspan.cloud --config <path | s3://bucket/key>`` is the image's
command. The run-file is read from a mounted local path (local Docker) or an
``s3://`` key (Fargate, where the YAML is uploaded once and every relaunch reads
the same key), then handed to :class:`~wingspan.cloud.runner.HeadlessRunner`.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys

from wingspan.cloud import runfile, runner, s3sync


def main(argv: list[str] | None = None) -> int:
    """Parse args, load the run-file, and run it to completion."""
    args = _parse_args(argv)
    _configure_logging()
    run = _load_run(args)
    sync = s3sync.S3Sync(run.s3, run.run_name)
    return runner.HeadlessRunner(run, sync).run()


###### PRIVATE #######


def _load_run(args: argparse.Namespace) -> runfile.CloudRunFile:
    """Read the run-file (local path or ``s3://`` URI) and apply CLI overrides."""
    config_arg: str = args.config
    if s3sync.is_s3_uri(config_arg):
        text = s3sync.fetch_text(
            config_arg, region=args.region, endpoint_url=args.endpoint_url
        )
    else:
        text = pathlib.Path(config_arg).read_text(encoding="utf-8")
    return _apply_overrides(runfile.parse_run_file(text), args)


def _apply_overrides(
    run: runfile.CloudRunFile, args: argparse.Namespace
) -> runfile.CloudRunFile:
    """Override the run-file's S3 region / endpoint from flags (for local testing).

    Lets a single run-file target real AWS by default yet be pointed at a local
    MinIO during testing without editing the YAML.
    """
    s3_updates: dict[str, str] = {}
    if args.region:
        s3_updates["region"] = args.region
    if args.endpoint_url:
        s3_updates["endpoint_url"] = args.endpoint_url
    if not s3_updates:
        return run
    return run.model_copy(update={"s3": run.s3.model_copy(update=s3_updates)})


def _configure_logging() -> None:
    """Send all logging to stdout so the container log driver captures progress."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="wingspan cloud",
        description="Run one Wingspan training run headless, persisting to S3.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="run-file: a local path or an s3://bucket/key URI",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region override (also used to fetch an s3:// config)",
    )
    parser.add_argument(
        "--endpoint-url",
        default=None,
        help="S3 endpoint override (e.g. a MinIO URL for local testing)",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
