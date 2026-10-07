# Known issues

What we know is wrong today (0.1.84, 2026-10-07), desktop and phone alike (the phone is a 1:1
port), each with how it shows and where it was found. Fixed items leave this list. The
2026-10-07 audit found more rough edges than are listed here; this list holds the ones a player
can meet.

## The release gate (sweeps, 0.1.84)

| Sweep | Result | Failing seeds |
|---|---|---|
| `tests/test_premove_sweep.py --seeds 1000` | 997/1000 | 742 is wrong for one ply, then right; 70 and 956 repeat an earlier position by the same move and hold it as a ply shown |
| `tests/test_stress.py` criteria, 1000 seeds | 997/1000 | 543 (a knight under a result badge), 604 and 817 (the board is never picked up) |
| `tests/test_flow_matrix.py --seeds 20` | 319/320 | seed 20, puzzle with takebacks, look-backs and premoves |

All of them also failed before the 2026-10-05 reconciler rewrite. How to run them:
[testing.md](testing.md).

## Game state

- **A correction mid-game cuts the record.** A flip, a fixed misread or a recovery by the
  last-move mark adopts the position afresh (`_set_position`), so `GameEnded.pgn` and the
  review hold only the moves after it; before ply 10 the game leaves the library
  (`profile/library.py`). Found by the 2026-10-07 audit.
- **The player is taken to be the side at the bottom.** A site set to "White always on the
  bottom" while playing Black coaches the wrong side (`pipeline.py`, `Reconciler(user_side=
  bottom)`).
- **The move-list check grows with the game.** `_browsed_in_flight` copies the board for every
  ply on every frame that disagrees with the record: about 40 ms a frame at 120 plies, 110 ms
  at 200 (2026-10-07 audit, desktop CPU).

## Move labels, reviews and Stats

- **The peers' label counts are the old rule's** (`peers.json`, fitted 2026-09-30 from labels at
  0.04 / 0.10 / 0.28): the peers' mistakes, misses and blunders are set against the player's
  labels of today until it is refitted. See [stats.md](stats.md).

## Vision (found 2026-09-27; not re-measured one by one since)

- **Random games not followed to the end** in a few seeds of the stress sweep: mostly dark
  shaded piece sets on the dark board (riohacha, pirouetti) and 44 px cells where a set's
  piece does not land where its exemplar did. Left by decision (2026-09-19).
- **A queen read as the other side's in a cold read** (the wood board, the caliente set at
  96 px).
- **A piece whose outline is as heavy as its fill can read as the other side's after a large
  shrink** (a set learned at 40 px, read at 29).
- **A board nudged less than a third of a cell is followed only at the next periodic re-fit**,
  up to ten frames on; nothing is invented meanwhile.
- **After a large resize, a piece at rest can read as not landed** where its set draws its
  outline at a fixed width; only moves from or to that square wait.
- **A board smaller than 24 px a cell (a 192 px board) is not seen at all**: the detector's
  floor. The game waits until the board is larger again.

## Phone only

- **Chess.com's live games cannot be watched on the phone**: its app sets FLAG_SECURE on the
  game window, so the capture is black. Puzzles, reviews and bot games work.
