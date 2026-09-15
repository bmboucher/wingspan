"""Offline research studies driven by trained checkpoints (``docs/RESEARCH.md``).

Each study is a fixed-model analysis job — no training, no gradient updates —
that samples game situations, scores them through a loaded network, and writes
a flat, pivotable CSV for downstream statistics. The first study is the setup
keep-rate experience table (``wingspan research setup-keep``); see
``setup_keep.py`` and ``INDEX.md``.
"""
