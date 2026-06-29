"""Structured game-event tree: the single source of truth for game logs.

Every game produces one :class:`~wingspan.gamelog.models.GameEventTree` whose
phases mirror the instrumentation handler's phase snapshots.  Both the HTML
decision log and the plaintext detailed log are pure renderers over this tree.
"""
