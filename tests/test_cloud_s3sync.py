"""Tests for the S3 game-log chunk-offload boundary logic (no network).

``offload_game_chunk`` reads the bytes appended to the local games log since the
last offset and uploads them as an immutable chunk; these confirm it never
splits a row across chunks and never re-sends already-offloaded bytes. The boto3
client is stubbed (so ``S3Sync`` builds without resolving AWS credentials) and
the PUT is captured, so no AWS access is required.
"""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

pytest.importorskip("boto3")

import boto3

from wingspan.cloud import runfile, s3sync


def _make_sync(monkeypatch: pytest.MonkeyPatch) -> s3sync.S3Sync:
    """An ``S3Sync`` whose underlying boto3 client is an inert stub, so the
    constructor never reaches out to resolve real AWS credentials."""

    def _stub_client(*_args: object, **_kwargs: object) -> object:
        return object()

    monkeypatch.setattr(boto3, "client", _stub_client)
    return s3sync.S3Sync(
        runfile.S3Config(bucket="b", prefix="runs", region="us-east-1"), "run1"
    )


def test_offload_game_chunk_trims_to_last_newline(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync = _make_sync(monkeypatch)
    captured: list[tuple[bytes, str]] = []

    def _capture(data: bytes, suffix: str) -> None:
        captured.append((data, suffix))

    monkeypatch.setattr(sync, "upload_bytes", _capture)

    games = tmp_path / "games.jsonl"
    games.write_bytes(b"a\nb\nc")  # the trailing line is not yet complete
    offset = sync.offload_game_chunk(games, "sess", 0, 0)
    assert offset == 4  # only the two complete lines "a\nb\n"
    assert captured[0] == (b"a\nb\n", "games/sess/chunk_00000.jsonl")

    games.write_bytes(b"a\nb\nc\n")  # the dangling line is now complete
    offset2 = sync.offload_game_chunk(games, "sess", 1, offset)
    assert offset2 == 6
    assert captured[1] == (b"c\n", "games/sess/chunk_00001.jsonl")


def test_offload_game_chunk_noop_without_complete_line(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync = _make_sync(monkeypatch)
    calls: list[bytes] = []

    def _capture(data: bytes, suffix: str) -> None:
        calls.append(data)

    monkeypatch.setattr(sync, "upload_bytes", _capture)

    games = tmp_path / "games.jsonl"
    games.write_bytes(b"partial-no-newline")
    assert sync.offload_game_chunk(games, "sess", 0, 0) == 0
    assert calls == []
