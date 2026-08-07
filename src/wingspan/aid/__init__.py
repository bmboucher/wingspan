"""``wingspan aid`` -- live physical-game assistant.

Drives the real ``Engine`` for an in-progress physical 2-player Wingspan
game: every hidden or random outcome (card reveals, dice rolls, feeder
rolls, opponent hand contents) is supplied by the human at the table
through a ``SessionOracle`` instead of ``random.Random``. See ``INDEX.md``
for the module map and which pieces have landed so far.
"""
