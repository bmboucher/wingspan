# Self-play collectors — which one runs, and why there are three

Collection plays self-play games and records every forked decision as training
data. There are three collector modules at three levels of parallelism. They are
**not** alternatives you pick by hand — `loop._collect` selects one per device.

| Module | Parallelism | Inference | Selected when |
|---|---|---|---|
| `collect.play_game` | one game, synchronous | batch-of-one per decision | the shared baseline (called by `mp_collect`); also the simplest thing to read |
| `batched_collect.collect_games` | many games, one OS thread each | **one shared forward pass** per decision round across all live games | `device == cuda` |
| `mp_collect.ProcessCollector` | many games across worker **processes** | batch-of-one inside each worker (each runs `collect.play_game`) | `device == cpu` (default) |

## The selection rule (`loop._collect`)

```
if device.type == "cpu":   ->  mp_collect.ProcessCollector   # one GIL per core
else (cuda):               ->  batched_collect.collect_games  # one shared GPU forward
```

## Why three

- **`collect`** is the single-game engine loop + recording. Start here to
  understand *what* a collected game is; the other two only change *how many run
  at once*.
- **`batched_collect`** shares the forward pass across concurrently-running game
  threads. On a GPU one shared forward beats one model copy per process. On CPU
  the per-decision engine + encoding work is GIL-bound, so threads gave only
  ~1.2–1.35x — which is why it is no longer the CPU path.
- **`mp_collect`** fans games across processes (one GIL per core), so CPU
  collection scales with physical cores. It is the default supported path
  (training is CPU-only — see the top-level `CLAUDE.md`).

All three return the same `collect.GameRecord` objects, so every downstream
aggregate (`loop`, `metrics`) is collector-agnostic. `mp_collect` returns them in
completion order rather than seed order; downstream aggregates are
order-independent.

## Shutdown: `close()` vs `terminate()`

`ProcessCollector` has two shutdown paths:

| Method | Wait for in-flight? | Use when |
|--------|---------------------|----------|
| `close()` | Yes (`shutdown(wait=True)`) | Normal end of run — all games finish, then files are removed |
| `terminate()` | No — workers are killed (`proc.kill()`) | Ctrl+C / SIGTERM — discard the current iteration, exit in <1s |

`terminate()` is called by `TrainingLoop.request_stop()`. In-flight games die
instantly; the `BrokenProcessPool` or `RuntimeError` propagating up through
`collect_games` is swallowed by the loop's stop-guard `try/except` so the run
reaches `Phase.STOPPED` cleanly. A subsequent `collect_games` call raises
`RuntimeError("collector terminated")` immediately so no new pool is spawned
after a terminate.
