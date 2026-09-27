# Known issues

What we know is wrong and have chosen to ship with, desktop and phone alike (the phone is a
1:1 port), each with how it shows and where it was found. Fixed items leave this list.

## Vision

- **A piece that lands after the board was moved, on the last move's highlighted square, may
  not count as landed** when the landing reference was learned at one lock offset and the board
  is read at another, a pixel apart in a different direction (seen at 48 px cells after a
  diagonal move of 1.4 cells: 2...Nc6 was read as a knight but never settled, so the move was
  not committed). Found by the parity audit's move sweep, 2026-09-26.
- **Random games that are not followed to the end** in about 1.6% of the stress sweep
  (`tests/test_stress.py --seeds 1000`; seeds 327 372 391 436 454 492 497 543 604 610 728 817
  822 830 972 996): mostly dark shaded piece sets on the dark board (riohacha, pirouetti) and
  44 px cells where a set's piece does not land where its exemplar did. Left by decision
  (2026-09-19); see the stress test's docstring.
- **A board smaller than 24 px a cell (a 192 px board) is not seen at all**: the detector's
  floor. The game waits until the board is larger again.
