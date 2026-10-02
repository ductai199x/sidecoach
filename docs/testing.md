# Writing tests for Sidecoach

How a test in this repo is written so that the whole suite stays fast, honest and deterministic.
These rules are Tai's (2026-10-01/02) and the measured lessons of making the suite meet them.
Every new test follows them; a review that finds one that does not sends it back. The phone's
version is `docs/testing.md` in sidecoach-android.

## The budget

- **The whole desktop suite** (`uv run pytest`, everything that runs by default) finishes in
  about 40 s and never more than 2 minutes, with the machine at most 70% busy, from a fresh
  checkout (the caches empty). Measure it with `scratchpad`-style sampling of the whole machine,
  not only the suite's own processes: Defender and the desktop add their share.
- **Nothing reaches the budget by proving less.** Never skip, xfail, mark, move a case to an
  opt-in flag, shrink a sweep (seeds, sets, sizes, game length, tolerances) or loosen an
  assertion to save time. A suite that is fast because it tests less is a failure.
- **Opt-in runs are only for what needs hardware or real models:** `-m slow` (voice models, the
  GPU, the real monitor) and the wider sweeps (`--seeds N`). They were opt-in before 2026-09-30
  and stay so; nothing else joins them.
- **Every claim about the suite's speed carries a measured number:** the same command before and
  after, wall time, test count, the collected test ids diffed (`--collect-only -q`), whole-
  machine CPU peak and mean, and coverage before and after.

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
- [ ] Runs in the default suite; `-m slow` only for real models, the GPU or the real monitor.
- [ ] Phone: the 1:1 test exists in sidecoach-android (or the reason it can't).
