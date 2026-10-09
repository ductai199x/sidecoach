# Writing tests for Sidecoach

How a test in this repo is written so that the whole suite stays fast, honest and deterministic.
These rules are Tai's (2026-10-01/02) and the measured lessons of making the suite meet them.
Every new test follows them; a review that finds one that does not sends it back. The phone's
version is [android/docs/testing.md](../android/docs/testing.md).

## The budget

- **The whole desktop suite** (`uv run pytest`, everything that runs by default) finishes in
  about 40 s and never more than 2 minutes, with the machine at most 70% busy, from a fresh
  checkout (the caches empty). Measure it with sampling of the whole machine (psutil every
  0.5 s), not only the suite's own processes: Defender and the desktop add their share. The
  last measured numbers are in `tests/conftest.py` (`WORKER_COST`, 2026-10-02: 8 workers,
  35 s cold, a peak of 60-65%).
- **Nothing reaches the budget by proving less.** Never skip, xfail, mark, move a case to an
  opt-in flag, shrink a sweep (sets, sizes, game length, tolerances) or loosen an assertion
  to save time. A suite that is fast because it tests less is a failure.
- **Opt-in runs are only for what needs hardware or real models, and the sweeps:** `-m slow`
  (the embedding net on the GPU in `tests/test_pieces.py`, the real speech recogniser in
  `tests/test_voice_hearing.py`) and the seeded sweeps past their default seeds (`--seeds N`,
  [below](#the-sweeps)). Tai, 2026-10-06: sweeps are not run by default; the default run plays
  a fixed handful of seeds of each, and the sweep is run deliberately, once, at the end of a
  piece of work. Nothing else joins them.
- **Every claim about the suite's speed carries a measured number:** the same command before and
  after, wall time, test count, the collected test ids diffed (`--collect-only -q`), whole-
  machine CPU peak and mean, and coverage before and after.

## Running tests

- **Targeted, not the whole suite.** While working, run the test files that import what you
  changed: `uv run pytest tests/test_reconcile.py tests/test_premoves.py`. The whole suite
  runs only when Tai asks for it. A guard (`.claude/hooks/guard_commands.py`) stops any pytest
  command that names no test file or node (`tests/` alone counts as the whole suite) unless the
  command carries `FULL_SUITE_ASKED=1`; the phone's `tools\test.ps1` without `-Class` is
  stopped the same way.
- **The phone's device tests on a fresh emulator.** `android\tools\test.ps1` pushes the files
  the device tests need (the engine's network, Silero, Parakeet, Kokoro) and then opens the push
  folder to the app (`chmod -R a+rwX`): on a new emulator that folder is the shell's (mode 2770,
  group `ext_data_rw`), the app's listener saw it empty, and the 0.1.88 gate skipped 54 device
  tests and failed three that needed a file without saying so (2026-10-09). A device that kept
  the files from an earlier install never showed it.
- **Workers.** `pyproject.toml` runs `-m 'not slow' -n auto --dist loadgroup -p no:cacheprovider`.
  `-n auto` is `int(cpus * 0.70 / 2)` workers (8 on a 24-thread machine), except that a run
  naming only test files, three files or fewer (`FEW_FILES`), gets no workers and runs in one
  process. Give a count (`-n 8`) to spread a run of one or two files, a sweep above all.
  Every test process runs at below-normal priority. The cases of one seed share a worker
  (`xdist_group`, `pytest_collection_modifyitems`).
- **Not alongside a live game.** The suite refuses to start (exit 3) while a panel serves on
  port 8765 or the desktop app has a session running (its `panel.lock` port answers
  `/api/health` with `running`): a dozen workers starved a live game's vision once. Set
  `SIDECOACH_TEST_ALONGSIDE=1` when that session is not a live game (a replay, your own
  panel).
- **What a test process is given** (`tests/conftest.py`): `SIDECOACH_VOICE_MODELS=0` (no Kokoro
  or Parakeet unless a test asks), `SIDECOACH_EMBED_DEVICE=cpu`, and `SIDECOACH_CONFIG`
  pointing at a scratch file, so a test that saves settings never writes the developer's
  `sidecoach.toml`.
- **Markers.** `-m slow` runs the slow tests (with the default `-m` replaced: `-m slow`
  runs only them, `-m ""` runs everything). `vision` marks the synthetic vision suite, which
  runs by default.
- **Durations.** `uv run pytest --store-durations` rewrites `tests/durations.json`, the order
  the workers are dealt the tests in (`tests/scheduling.py`).
- **Caches.** `SIDECOACH_TEST_CACHE=1` (default) reads and writes the embedding net's kept
  outputs, `0` neither, `rebuild` recomputes and overwrites, `verify` recomputes and compares
  (`tests/netcache.py`).
- Lint, format and types: `uv run ruff check .`, `uv run ruff format .`, `uv run pyright`.

## The sweeps

Every test that takes a `seed` argument plays its module's `SEEDS` by default, or the seeds
`--seeds` asks for: `--seeds N` plays seeds 1 to N, `--seeds A-B` the seeds A to B
(`pytest_generate_tests` in `tests/conftest.py`).

| File | What it sweeps | Default seeds |
|---|---|---|
| `tests/test_stress.py` | random games through the whole vision pipeline, every theme, set, size and decoration; two tests per seed (followed to its position, ends on screen) | 1-8 |
| `tests/test_board_geometry.py` | the board moved, resized, covered and scrolled; every-frame and gated per seed | 154, 287, 34, 175, 2, 3 |
| `tests/test_premove_sweep.py` | premove queues as chess.com and lichess run them | 1-4, 117 |
| `tests/test_flow_matrix.py` | game or puzzle x takebacks x looking back x premoves (16 cases per seed) | 1 |

The default seeds were cut on 2026-10-06 (flow matrix 48 games to 16, premove sweep 9 to 5;
ada9b20) when Tai decided sweeps are not part of the default run. Run a sweep in the
background, with a log, after the work is done:

```
uv run pytest tests/test_premove_sweep.py -q -n 8 --seeds 1000
uv run pytest tests/test_stress.py -q -n 8 --seeds 1000
uv run pytest tests/test_flow_matrix.py -q -n 8 --seeds 20
uv run pytest tests/test_board_geometry.py -q -n 8 --seeds 300
```

A sweep names one file, so `-n auto` would run it in one process (see Workers): give the
count. What each sweep asserts, and the current results, are in
[reconciler-model.md](reconciler-model.md#7-proving-it). A failing seed is reproduced alone
(`--seeds 742-742`) before any theory.

## Parity fixtures for the phone

The Android port (`android/`) is held to the desktop by fixtures the desktop
writes: the generators are `android/tools/make_*_fixtures.py` (and
`make_fixtures.py`), run from this repo, which has the renderer and python-chess:

```
uv run --no-sync python android/tools/make_reconcile_fixtures.py
```

Each records what the desktop code does -- the boards it renders, the readings, every
reconciler event frame by frame and every op of `tests/test_reconcile.py`,
`test_premoves.py`, the premove sweep's seeds and a few stress seeds, the coach's texts, the
engine's lines -- into the phone's test assets, and the phone's parity tests must answer the
same. Regenerate the fixtures of a module whenever you change it (the generator's imports
say which modules it records: `make_reconcile_fixtures.py` for `sidecoach.state.reconcile`,
`make_pipeline_fixtures.py` for `state/pipeline.py`, `make_coach_fixtures.py` for `coach/`,
...), and whenever a test it replays changes. Commit them with the change and its
Kotlin port; the phone's README lists the main ones.

## Tests prove logic, never timing

A test that can fail because the machine was slow is rewritten, never retried, never given a
longer timeout and never given a shorter wait (Tai, 2026-10-01).

- **Inject the clock.** `tests/fakes.py` `ManualClock` stands in for one module's own `time` or
  `threading` (`monkeypatch.setattr(panel_mod, "time", clock.time_module())`); its `sleep`
  moves the clock and its `Timer` fires when the test advances past it. Use the real delays on
  the fake clock: boundaries become exact (sent at 1.0 s of quiet, not at 0.99 s).
- **Wait on a signal the code already gives**, never poll with sleeps: wrap the object's own
  hook (`_settle`, `_results.put`, `_emit`) to notify a Condition (`AnalyserWatch`), wait on its
  Event (`room.settled`), join its thread. A deadline (`HANG_S`) is only a safety net that a
  working run never reaches, and a wait that times out FAILS (`assert go.wait(HANG_S)`), never
  carries on.
- **Stage races; don't hope for them.** Hold a step on an Event so the second action lands
  exactly in its window (a gated fake engine, a lock that reports a waiting thread, a stream
  that opens only when the test lets it).
- **Count work instead of timing it.** "At most 2 searches between the takeback and the verdict,
  only of the new line" replaces `waited < 2.2 * search`; "two box searches and ten glyph reads
  per frame, at any frame size" replaces `per_frame_ms < 5`.
- **A fake's timing must be possible in the real system.** A fake watchdog that fires before the
  search has started tests a moment the real one can never reach.
- **No real network, no fixed ports you didn't open, no unseeded randomness.** A real socket is
  used only when the socket is what is tested.
- **Prove a rewrite still catches the bug:** break the code on purpose (remove the lock, the
  check) and see the test fail; then restore it.

## Make the same test cheaper

In order of what it has bought so far:

1. **Schedule by measured duration.** `tests/scheduling.py` deals the longest tests first from
   `tests/durations.json` (`uv run pytest --store-durations` rewrites it). Group tests onto one
   worker (`xdist_group`) only when they really share an expensive state.
2. **Cap threads in the config, not by habit.** Workers = `int(cpus * 0.70 / 2)`; every native
   pool in a test process is one thread wide (OMP/BLAS env, `cv2.setNumThreads(1)`, onnxruntime
   intra-op 1, Stockfish `Threads=1`) — `tests/conftest.py`. The workers are the parallelism.
3. **Memoise pure test helpers.** `tests/synth.py` `Kept`: key on every input that decides the
   result (resolved objects, not names; the FEN when the side to move matters); hand each caller
   a copy of a read-only original.
4. **Cache deterministic third-party computation across runs** (`tests/netcache.py`, the
   embedding net's forward pass): key on a content hash of every input that decides the output
   (model bytes, runtime version, device, the exact input row, a format version); prove the
   computation is pure first (same bits in any batch and thread count); store safely for
   parallel workers (SQLite WAL, the file made whole aside and `os.link`ed in); switches
   `SIDECOACH_TEST_CACHE=0|rebuild|verify`.
5. **Share a warmed state only after proving it.** The overlay sweep reads its opening once per
   set and theme and runs each case on a `copy.deepcopy` of the warmed pipeline — after a test
   showed a copy gives exactly the events a fresh warm-up gives, and a guard test keeps checking
   it.
6. **Keep caches small.** At most ~256 MB per worker (Tai, 2026-10-02): store a frame as its
   difference from a shared background and rebuild it exactly (proven by SHA-256), evict by
   bytes. Never fit a cache by changing what the test sees.

Rules for every cache:

- **Never cache the code under test** — only its inputs (rendered frames) and pure external
  functions (an ONNX forward pass). The app's own work runs on every call.
- **Test invalidation, not only hits:** a changed input, model, runtime or device is a miss;
  kept equals fresh, bit for bit; a corrupt store still gives correct answers; concurrent
  writers don't collide. A cache's own tests run in its default mode whatever mode the run was
  started in.
- **The budget holds with the caches empty.** A warm cache is a bonus.
- **Measure both paths.** A cache whose miss costs more than computing afresh is removed.

## Duplicates

A test is removed only when another test runs the same code path with the same (or more)
inputs and asserts the same (or more) — and per-test coverage shows it adds nothing. Identical
line coverage is not enough: 41 such groups were checked and none was a duplicate. Each removal
is written up with both tests side by side; when in doubt, it stays.

## What only measuring showed

- Import app modules in `pytest_sessionstart` (trylast) or later: imported in
  `pytest_configure`, 300 lines read as never covered in the workers.
- A timing rewrite can lose paths the old timing reached by chance: diff branch coverage before
  and after, and give each lost arc a deterministic test of its own.
- A cache counter is shared state across tests in a worker: give a test that counts its own
  fresh cache (an autouse fixture).
- Fewer workers can cost almost nothing (14 workers 29 s, 8 workers 35 s, the machine's peak
  92% -> 65%): the wall is set by start-up, the longest test and warm-up, not total CPU.
- A flaky test that cannot be reproduced is reported as such, never as "fixed".

## Checklist for a new test

- [ ] Fails without the change it guards (checked on the parent code).
- [ ] No sleep, no wall-clock assertion, no poll loop; every wait is on a signal with a failing
      safety bound.
- [ ] Uses the shared fixtures and caches (`tests/synth.py`, `tests/coaching.py`,
      `tests/fakes.py`, `tests/netcache.py`); builds nothing expensive that another test already
      built.
- [ ] The smallest input that proves the point, unless size is the point (vision geometry and
      sizes are always full size).
- [ ] Runs in the default suite; `-m slow` only for real models or the GPU; a seeded sweep
      takes `seed` and keeps a few default `SEEDS`.
- [ ] Changed code the phone ports: its parity fixtures regenerated.
- [ ] Phone: the 1:1 test exists in android/ (or the reason it can't).
