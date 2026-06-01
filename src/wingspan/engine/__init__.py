"""Game engine.

The engine is split across small modules organized by concern; the public
entry points (``Engine``, ``print_coverage_report``) are re-exported here.

- ``core``         — ``Engine`` class, turn loop, setup, decision plumbing
- ``actions``      — play_bird / gain_food / lay_eggs / draw_cards
- ``powers``       — bird-power dispatch (``apply_effect`` switch)
- ``reactors``     — pink between-turn reactor hooks
- ``scoring``      — round and final scoring
- ``helpers``      — pure helpers (food enumeration, egg ladders, etc.)
"""

from wingspan.engine.core import Agent, Engine, print_coverage_report

__all__ = [
    "Agent",
    "Engine",
    "print_coverage_report",
]
