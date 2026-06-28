"""``InstrumentationConfig`` rejects malformed configs with a clean
``ValidationError`` rather than a bare ``KeyError`` or a silent mis-wire.
"""

from __future__ import annotations

import pydantic
import pytest

from wingspan.instrumentation import config


def test_unknown_handler_class_rejected():
    with pytest.raises(pydantic.ValidationError):
        config.InstrumentationConfig.model_validate(
            {"handlers": {"x": {"class": "NoSuchHandler"}}, "events": {}}
        )


def test_missing_class_key_rejected():
    with pytest.raises(pydantic.ValidationError):
        config.InstrumentationConfig.model_validate(
            {"handlers": {"x": {"output_path": "x.jsonl"}}, "events": {}}
        )


def test_illegal_event_assignment_rejected():
    # CardVisitRecorder implements bird_placed/game_end but not round_start.
    with pytest.raises(pydantic.ValidationError):
        config.InstrumentationConfig.model_validate(
            {
                "handlers": {
                    "cv": {"class": "CardVisitRecorder", "output_path": "v.jsonl"}
                },
                "events": {"round_start": ["cv"]},
            }
        )


def test_event_references_unknown_handler_rejected():
    with pytest.raises(pydantic.ValidationError):
        config.InstrumentationConfig.model_validate(
            {"handlers": {}, "events": {"game_end": ["missing"]}}
        )
