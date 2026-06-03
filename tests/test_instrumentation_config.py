"""``InstrumentationConfig`` round-trips through every persistence path.

The registry discriminator must survive ``model_dump`` / ``model_validate``
(checkpoints), ``model_dump_json`` / ``model_validate_json`` and
``yaml.safe_load`` (the cloud run file), and nesting inside ``TrainConfig`` —
always reconstructing the identical concrete handler subclass.
"""

from __future__ import annotations

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wingspan.instrumentation import config, events
from wingspan.instrumentation.handlers import card_visits

_RAW = {
    "handlers": {"cv": {"class": "CardVisitRecorder", "output_path": "visits.jsonl"}},
    "events": {"bird_placed": ["cv"], "game_end": ["cv"]},
}


def _assert_recovered(cfg: config.InstrumentationConfig) -> None:
    handler = cfg.handlers["cv"]
    assert isinstance(handler, card_visits.CardVisitRecorder)
    assert handler.output_path == "visits.jsonl"
    assert cfg.events[events.EventName.BIRD_PLACED] == ["cv"]
    assert cfg.events[events.EventName.GAME_END] == ["cv"]


def test_model_dump_round_trip():
    cfg = config.InstrumentationConfig.model_validate(_RAW)
    _assert_recovered(config.InstrumentationConfig.model_validate(cfg.model_dump()))


def test_json_round_trip():
    cfg = config.InstrumentationConfig.model_validate(_RAW)
    _assert_recovered(
        config.InstrumentationConfig.model_validate_json(cfg.model_dump_json())
    )


def test_yaml_round_trip():
    text = """
handlers:
  cv:
    class: CardVisitRecorder
    output_path: visits.jsonl
events:
  bird_placed: [cv]
  game_end: [cv]
"""
    _assert_recovered(config.InstrumentationConfig.model_validate(yaml.safe_load(text)))


def test_default_is_empty():
    cfg = config.InstrumentationConfig()
    assert cfg.handlers == {}
    assert cfg.events == {}
    assert cfg.build().by_event == {}


def test_build_shares_instance_across_events():
    router = config.InstrumentationConfig.model_validate(_RAW).build()
    bird = router.by_event[events.EventName.BIRD_PLACED][0]
    end = router.by_event[events.EventName.GAME_END][0]
    assert bird is end


def test_nested_under_train_config():
    pytest.importorskip("torch")
    from wingspan.training import config as train_config

    cfg = train_config.TrainConfig(
        instrumentation=config.InstrumentationConfig.model_validate(_RAW)
    )
    restored = train_config.TrainConfig.model_validate(cfg.model_dump())
    _assert_recovered(restored.instrumentation)
    # Attaching instrumentation must not change the network-shape signature.
    assert restored.architecture_key == train_config.TrainConfig().architecture_key
