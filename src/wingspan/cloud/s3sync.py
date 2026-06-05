"""S3 persistence for a cloud training run — the sync sidecar around the loop.

The training loop keeps writing artifacts to its local ``checkpoint_dir``
exactly as it always has; :class:`S3Sync` decides *when* those bytes go to S3,
which is what keeps the run from spamming PUTs. It mirrors three kinds of
artifact at three cadences:

* the resumable **checkpoint set** (``last.pt`` + the small logs/descriptors) —
  uploaded together as a consistent unit every few iterations and at shutdown,
* the tiny **status snapshot** — uploaded frequently (it is ~1 KB), and
* the high-volume **per-game log** — offloaded as immutable, size-bounded chunks
  under a ``games/<session>/`` prefix, never re-uploaded whole.

On startup :meth:`download_run` pulls the prior state (everything *but* the game
chunks) back into the local dir so the loop's own resume logic continues the run.

Credentials come from the standard AWS chain (the Fargate task role in the
cloud); :class:`runfile.S3Config` carries only the bucket / prefix / region /
endpoint.
"""

from __future__ import annotations

import pathlib
import typing
import urllib.parse

import boto3

from wingspan.cloud import runfile, status
from wingspan.training import artifacts

if typing.TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

# The consistent resumable set uploaded together (any that don't exist for a
# given run are skipped). The big per-game log is handled separately (chunked);
# the target-milestone ``final_*`` files and dated ``process_*`` records are
# swept in by glob in :meth:`S3Sync.upload_checkpoint_set`.
_CHECKPOINT_SET: tuple[str, ...] = (
    artifacts.LAST_CKPT,
    artifacts.BEST_CKPT,
    artifacts.OPPONENT_CKPT,
    artifacts.SETUP_CKPT,
    artifacts.METRICS_LOG,
    artifacts.SETUP_DATA_LOG,
    artifacts.MODEL_CONFIG_JSON,
    artifacts.SETUP_CONFIG_JSON,
)


class S3Sync:
    """Run-scoped S3 access: everything lives under ``<prefix>/<run_name>/``."""

    def __init__(self, s3: runfile.S3Config, run_name: str):
        self._cfg = s3
        self._bucket = s3.bucket
        self._client: S3Client = _make_client(s3.region, s3.endpoint_url)
        prefix = s3.prefix.strip("/")
        self._run_prefix = f"{prefix}/{run_name}" if prefix else run_name

    # ------------------------------------------------------------------
    # Startup: pull the prior run state back to the local checkpoint dir
    # ------------------------------------------------------------------

    def download_run(self, local_dir: pathlib.Path) -> int:
        """Download every run object except the ``games/`` chunks into ``local_dir``.

        That covers ``last.pt`` / ``opponent.pt`` / ``setup.pt`` /
        ``setup_data.jsonl`` / ``metrics.jsonl`` / descriptors — everything the
        loop's resume path reads — while leaving the large append-only game
        history in S3 (it is analysis output, not needed to resume). Returns the
        number of objects pulled (0 means a fresh run with nothing to resume).
        """
        local_dir.mkdir(parents=True, exist_ok=True)
        games_prefix = f"{self._run_prefix}/{artifacts.GAMES_SUBDIR}/"
        pulled = 0
        for key in _list_keys(self._client, self._bucket, f"{self._run_prefix}/"):
            if key.endswith("/") or key.startswith(games_prefix):
                continue
            relative = key[len(self._run_prefix) + 1 :]
            destination = local_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(self._bucket, key, str(destination))
            pulled += 1
        return pulled

    # ------------------------------------------------------------------
    # Uploads
    # ------------------------------------------------------------------

    def upload_file(self, local_path: pathlib.Path, suffix: str) -> None:
        """Upload a local file to ``<run_prefix>/<suffix>``."""
        self._client.upload_file(str(local_path), self._bucket, self._key(suffix))

    def upload_bytes(self, data: bytes, suffix: str) -> None:
        """PUT in-memory bytes to ``<run_prefix>/<suffix>`` (for the status JSON)."""
        self._client.put_object(Bucket=self._bucket, Key=self._key(suffix), Body=data)

    def upload_checkpoint_set(self, local_dir: pathlib.Path) -> None:
        """Upload the consistent resumable set plus any milestone / session files.

        Skips members that don't exist for this run (e.g. ``opponent.pt`` before
        the first opponent advance, or the setup artifacts when the setup model
        is off), so it is safe to call unconditionally each cadence.
        """
        names = list(_CHECKPOINT_SET)
        names += sorted(path.name for path in local_dir.glob("final_*"))
        names += sorted(path.name for path in local_dir.glob(artifacts.PROCESS_GLOB))
        for name in names:
            path = local_dir / name
            if path.exists():
                self.upload_file(path, name)

    def offload_game_chunk(
        self,
        games_path: pathlib.Path,
        session_stamp: str,
        seq: int,
        from_offset: int,
    ) -> int:
        """Upload the complete game-log lines appended since ``from_offset``.

        Reads ``[from_offset, current_size)`` of the local games log, trims to the
        last newline so a chunk never splits a row, and PUTs it as an immutable
        ``games/<session_stamp>/chunk_<seq>.jsonl`` object. Returns the new offset
        (unchanged when there is no complete new line to send). Never re-uploads
        already-offloaded bytes and never overwrites an existing chunk.
        """
        if not games_path.exists():
            return from_offset
        size = games_path.stat().st_size
        if size <= from_offset:
            return from_offset
        with open(games_path, "rb") as handle:
            handle.seek(from_offset)
            data = handle.read(size - from_offset)
        last_newline = data.rfind(b"\n")
        if last_newline < 0:
            return from_offset  # no complete line yet
        data = data[: last_newline + 1]
        suffix = f"{artifacts.GAMES_SUBDIR}/{session_stamp}/chunk_{seq:05d}.jsonl"
        self.upload_bytes(data, suffix)
        return from_offset + len(data)

    ###### PRIVATE #######

    def _key(self, suffix: str) -> str:
        return f"{self._run_prefix}/{suffix}"


# ---------------------------------------------------------------------------
# Bucket-wide helpers (not tied to a single run) for the monitor + bootstrap
# ---------------------------------------------------------------------------


def iter_run_statuses(s3: runfile.S3Config) -> list[status.RunStatus]:
    """Read every run's ``status.json`` under the bucket prefix (for the monitor).

    Scans ``<prefix>/*/status.json`` and parses each into a :class:`RunStatus`,
    skipping any object that fails to read or validate (a half-written snapshot)
    so one bad run never blanks the roster.
    """
    client: S3Client = _make_client(s3.region, s3.endpoint_url)
    base = s3.prefix.strip("/")
    base_prefix = f"{base}/" if base else ""
    statuses: list[status.RunStatus] = []
    for key in _list_keys(client, s3.bucket, base_prefix):
        if not key.endswith(f"/{artifacts.STATUS_JSON}"):
            continue
        try:
            body = client.get_object(Bucket=s3.bucket, Key=key)["Body"].read()
            statuses.append(status.RunStatus.model_validate_json(body))
        except (
            Exception
        ):  # noqa: BLE001 — a half-written snapshot must not blank the roster
            continue
    return statuses


def fetch_text(uri: str, *, region: str | None, endpoint_url: str | None) -> str:
    """Read an ``s3://bucket/key`` object as UTF-8 text (for the bootstrap config).

    Uses an ephemeral client because the run-file's own ``S3Config`` is exactly
    what we are about to read; region / endpoint come from the entry point's
    flags (or the AWS chain) instead.
    """
    bucket, key = _split_s3_uri(uri)
    client: S3Client = _make_client(region, endpoint_url)
    return client.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")


def is_s3_uri(value: str) -> bool:
    """Whether ``value`` is an ``s3://`` URI (vs a local filesystem path)."""
    return value.startswith("s3://")


def _split_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise ValueError(f"not an s3:// URI: {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def _make_client(region: str | None, endpoint_url: str | None) -> S3Client:
    """Build an S3 client.

    The ``boto3-stubs`` ``client`` overload spans ~400 services (most returning
    ``Unknown`` unless their own stub package is installed), so ``boto3.client``
    reads as partially-unknown under strict pyright even though the ``"s3"``
    overload yields a concrete ``S3Client``. Suppress just that stub-shape report
    here, in the one place a client is built.
    """
    return boto3.client(  # pyright: ignore[reportUnknownMemberType]
        "s3", region_name=region, endpoint_url=endpoint_url
    )


def _list_keys(client: S3Client, bucket: str, prefix: str) -> list[str]:
    """Every object key under ``prefix`` (paginated)."""
    keys: list[str] = []
    token: str | None = None
    while True:
        if token is None:
            response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        else:
            response = client.list_objects_v2(
                Bucket=bucket, Prefix=prefix, ContinuationToken=token
            )
        for entry in response.get("Contents", []):
            key = entry.get("Key")
            if key:
                keys.append(key)
        # The stubs type ``NextContinuationToken`` as always-present, so reading
        # it only when ``IsTruncated`` is set keeps the loop both correct and free
        # of a dead ``token is None`` comparison.
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
    return keys
