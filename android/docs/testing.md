# Writing tests for the phone port

The rules are the desktop's (`docs/testing.md` in chess-coaching-companion): the budget, logic
never timing, caches with tested invalidation, duplicates only when proven, nothing made faster
by proving less. This file adds what is particular to the phone. How the suite is put together
(the two halves, lanes, `apart` processes) is in the README's "Testing".

## The budget

- **The whole phone suite** (`tools\test.ps1`: the device's half and the host's half, every case,
  no `-All` needed) aims at the desktop's ballpark — about 40-60 s — and never more than 2
  minutes, with the host at most 70% busy, measured with the APKs already built (build and
  install reported separately). Measured 2026-10-02 on emulator-5556 at 6 cores / 6 GB: 105-111 s
  for 315 device + 375 host tests, from 400 s before (not re-measured since). The floor on that emulator is ~70 s
  (≈420 guest CPU-seconds on 6 cores); getting lower needs less work per test or a second device.
- Never reach it by `@Ignore`, assumptions that skip, opt-in flags, fewer piece sets or
  scenarios, or shorter/longer waits. Every skipped test is printed by name at the end of a run;
  a test that is skipped in every whole run is a bug.
- Tests run only on the emulator (`emulator-5556`, `tools\test.ps1`'s default), never on the
  phone in use.
- While developing, run the classes you changed (`tools\test.ps1 -Class <class>`, `gradlew.bat
  :app:testPhoneUnitTest --tests <class>`); the whole suite only when asked (the desktop's agent
  guard wants `FULL_SUITE_ASKED=1` for it).

## Where a test goes

- **The host JVM (`app/src/test`) first.** A class that needs no OpenCV, onnxruntime, Stockfish,
  SQLite, Context or UI runs on the host: same fixtures (`TestAssets`), shared helpers in
  `src/sharedTest`, AOSP's `org.json` (`:libcore-json`). Android stubs throw, so nothing passes
  by accident.
- **The device (`app/src/androidTest`)** for everything else, in a lane (`Lanes`): the app lane
  for anything that touches the app's shared state (settings, singletons, files), the pools for
  the rest.
- **Its own process (`@InProcess(n)`)** for a class that churns through the Java heap. On
  Android the heap (192 MB on the emulator) and the collector are per process; the vision
  replays allocate gigabytes and, sharing one process, waited on the collector rather than the
  CPU. `largeHeap` in the test APK does nothing: the target app's process decides.

## Patterns that held

- **Measure allocation as well as CPU:** `Debug.getRuntimeStat("art.gc.bytes-allocated")` and
  `"art.gc.gc-count"` around each test; the timings table (`app/build/test-timings.tsv`) has
  process, lane, seconds, CPU, peak memory, heap allocated, collections and result.
- **One thread per net and per OpenCV pool in a test** (`TestNets.THREADS`,
  `Core.setNumThreads(1)`); the classes side by side are the parallelism. Measure before adding
  threads: a collector-bound class gets slower with more.
- **A clock the test moves** (`StepClock`, `ManualClock`, the caption box's `Clock`, the
  dictation's `SendClock`) and **signals from the code's own hooks** (`Analyser.Watch`, through
  `AnalyserWatch` in `src/sharedTest`) instead of sleeps; the shipped delays, never shortened
  ones.
- **A test of state the app keeps starts from a known state** (the default settings, a quiet fake
  voice) — earlier tests in the same process leave theirs behind.
- **Wait on the condition you mean, at its granularity:** a stopped process is every thread of
  `/proc/<pid>/task/*` stopped, not its main thread.
- **A wall-clock cap inside the code under test is timing too:** give it a seam whose default the
  app never changes (`Engine.extendSeconds`) and test by depth.

## Anti-patterns

- An unbounded wait (`latch.await()` with no bound): one hang and nothing after it reports.
  Every wait has a bound that fails, and is never the expected path.
- A shorter wait to make a timing test pass (the caption fade at 1 s instead of the shipped 5 s).
- More threads into a collector-bound process; lower priority for "short" lanes.
- Assuming the emulator scales with cores: past the floor only less work or more devices help.

## Checklist

- [ ] The 1:1 test of the desktop's exists (same cases, same assertions), or the reason it can't.
- [ ] Fails without the change it guards.
- [ ] On the host if it can be; on the device in the right lane / process if not.
- [ ] No sleep, no time assertion; every wait bounded and failing.
- [ ] Starts from a known state; leaves none behind that another test reads.
