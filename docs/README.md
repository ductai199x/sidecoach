# Documentation

One line per document: what it is for, and when to read it.

## Start here

- [../README.md](../README.md): what Sidecoach does, how to install it, run it from source
  and build it. Read first.
- [../DESIGN.md](../DESIGN.md): scope, policy, the pipeline stage by stage and every
  decision with its numbers; the source of truth. Read before changing architecture.
- [../CLAUDE.md](../CLAUDE.md): the commands and conventions an agent working in this repo
  follows (the test guard, disclosure, versions). Read before working in the repo.

## The product

- [app.md](app.md): the desktop shell (Tauri, the sidecar, packaging, the CUDA pack, Linux)
  and the Android plan, with the measurements behind them. Read before touching the shell,
  the installer or the platform work.
- [coach.md](coach.md): the coach as one conversation per game, its prompt, context and
  checks. Read before changing what the coach says or how it is asked.
- [game-review.md](game-review.md): Game Review, modelled on chess.com's: its decisions,
  flow and screens. Read before changing the review.
- [stats.md](stats.md): Stats, what a player's reviewed games say about them. Read before
  changing the Stats window or its numbers.
- [move-classification.md](move-classification.md): the move classes (brilliant to
  blunder, book, miss) as chess.com defines them, and how each is decided. Read before
  changing a class or its cut-offs.
- [reconciler-model.md](reconciler-model.md): what the reconciler assumes about the screen
  and the one algorithm that follows. Read before changing `state/`.
- [known-issues.md](known-issues.md): what is known to be wrong and shipped anyway, desktop
  and phone. Read before reporting or chasing a bug.

## Testing and evaluation

- [testing.md](testing.md): how a test is written here: the suite's budget, logic not
  timing, caches, duplicates. Read before writing or changing a test.
- [eval-results.md](eval-results.md): how the coach is evaluated (the judge, ICR, the
  benchmark, the audits), the numbers and the decisions taken on them. Read before an
  evaluation run or a claim about the coach's quality.

## Elsewhere

- [../desktop/README.md](../desktop/README.md): the desktop shell's development loop.
- [../tools/pieces/README.md](../tools/pieces/README.md): how Sidecoach's own piece set was
  made. Read before changing the set.
- `data/eval/README.md` (local, not committed): the layout of the evaluation data.
- [sidecoach-android README](../../sidecoach-android/README.md): the phone app, the port's
  rules, building, installing and testing it.
- [sidecoach-android docs/testing.md](../../sidecoach-android/docs/testing.md): what is
  particular to the phone's tests. Read with [testing.md](testing.md) before writing one.
