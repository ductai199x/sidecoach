# Known issues

What we know is wrong and have chosen to ship with, desktop and phone alike (the phone is a
1:1 port), each with how it shows and where it was found. Fixed items leave this list.

## Vision

- **Random games that are not followed to the end** in about 1.3% of the stress sweep
  (`tests/test_stress.py --seeds 1000`; seeds 327 436 454 497 543 604 610 688 728 817 822 830
  996): mostly dark shaded piece sets on the dark board (riohacha, pirouetti) and 44 px cells
  where a set's piece does not land where its exemplar did. Left by decision (2026-09-19); see
  the stress test's docstring. In 688 (the wooden set on the wood board, 96 px) a black queen on
  a light square reads as a knight: the start position shows each side's queen on one square
  colour only, and at the embedder's 40 px the black knight learned on a light square is
  nearer; a premove is then read as two pawn moves and the mate is never committed. Teaching
  the model recoloured kings and queens fixed it, but made a set never learned read as a
  position (2026-09-27).
- **A small board of a thin-outlined set may never start a game** (the geometry sweep's seed
  40: the thin set on the dark board at 32 px): the white pawn on a dark square covers 15.9 to
  16.1% of its cell, on the 16% floor for a piece, and the detector locks sharp-edged boards a
  pixel up and left (the comb settles on the first pixel of a two-pixel boundary), so the
  start position does not read as one and no game begins. Found 2026-09-27.
- **A piece whose outline is as heavy as its fill can read as the other side's after a large
  shrink** (the geometry sweep's seed 122: the caliente set learned at 40 px, read at 29): the
  white bishop's dark outline was taken for its fill, brightness voted black and the embedding
  nearly tied (0.961 against 0.957). The misread held back a three-ply story until six plies
  were missing, more than the missed-move search bridges, and the position was adopted anew.
  Found 2026-09-27.
- **After a fast animated drag, the lock can trail the board by up to a third of a cell for up
  to ten frames** (the geometry sweep's seed 97: the board slid 133 px up, the lock 27 px
  behind it), under what validation can see. For that second every piece reads as not landed
  and no move is read; nothing is invented (a move needs a cleanly read square to commit from,
  59763b9, after this lock once committed a phantom Kd3). Found 2026-09-27.
- **A lock up to 3 px off after a nudge near the frame's edge** in 4 of 1000 near-edge sweep
  cases, inside the re-fit's 2 px dead band. Reading holds at that offset. Found 2026-09-27.
- **A board smaller than 24 px a cell (a 192 px board) is not seen at all**: the detector's
  floor. The game waits until the board is larger again.
