# Known issues

What we know is wrong and have chosen to ship with, desktop and phone alike (the phone is a
1:1 port), each with how it shows and where it was found. Fixed items leave this list.

## Vision

- **Random games that are not followed to the end** in about 1.2% of the stress sweep
  (`tests/test_stress.py --seeds 1000`; seeds 436 454 497 543 604 610 688 695 699 728 817 996,
  2026-09-27 after the exact lock of 076cae1, which made 327, 822 and 830 pass and 695 and 699
  fail): mostly dark shaded piece sets on the dark board (riohacha, pirouetti) and 44 px cells
  where a set's piece does not land where its exemplar did. Left by decision (2026-09-19); see
  the stress test's docstring. In 688 (the wooden set on the wood board, 96 px) a black queen on
  a light square reads as a knight: the start position shows each side's queen on one square
  colour only, and at the embedder's 40 px the black knight learned on a light square is
  nearer; a premove is then read as two pawn moves and the mate is never committed. Teaching
  the model recoloured kings and queens fixed it, but made a set never learned read as a
  position (2026-09-27).
- **A move-quality badge beside a king can be read as a result badge, and the game ended
  mid-game** (the stress sweep's seed 699: the brown board, the classic set at 144 px). With
  Nxd8 and a badge drawn on d8, beside the black king, the wider search around a king read
  that badge as a win for black, and the white king on e1, with no badge on it, as a loss:
  the pair the result rule wants, so the game ended 0-1 after 10...Kxd8 and nothing was
  followed after. Chess.com draws quality badges in the game review, not in a live game, so
  this needs the synthetic badges; the detector's read of an empty corner as a loss is the
  part to fix. Found 2026-09-27.
- **A queen read as the other side's in a cold read** (the stress sweep's seed 695: the wood
  board, the caliente set at 96 px): the game was followed to its end, and the final position
  read cold from the cache took the white queen on c7 for a black one. Found 2026-09-27.
- **A piece whose outline is as heavy as its fill can read as the other side's after a large
  shrink** (the geometry sweep's seed 122: the caliente set learned at 40 px, read at 29): the
  white bishop's dark outline was taken for its fill, brightness voted black and the embedding
  nearly tied (0.961 against 0.957). The misread held back a three-ply story until six plies
  were missing, more than the missed-move search bridges, and the position was adopted anew.
  Found 2026-09-27.
- **A board nudged less than a third of a cell is followed only at the next periodic re-fit,
  up to ten frames on**: the checker fit cannot see such a shift, so the lock stays where the
  board was (the geometry sweep's seeds 83 and 97: 10 to 37 px off at 110 to 136 px cells).
  For that second every piece reads as not landed and no move is read; nothing is invented
  (a move needs a cleanly read square to commit from, 59763b9). A move the fit does see is
  re-fitted on the frame it fails, and again every frame while the board keeps moving.
  Found 2026-09-27.
- **After a large resize, a piece at rest can read as not landed** where its set draws its
  outline at a fixed width (the phone's landing fixture king-48-132: the synthetic classic set
  learned at 48 px and read at 132, the white king on e1 0.262 of a cell off). Learned small,
  the outline is thick for the cell; read large, thin; brought to one size the two crosses
  share no edge strong enough for the phase correlation's peak. The exact lock (076cae1)
  showed it: the lock a pixel off had left a strip of the next square in the reference, and
  that strip matched. Only moves from or to that square wait; sites' vector pieces scale their
  outlines. Found 2026-09-27.
- **A board smaller than 24 px a cell (a 192 px board) is not seen at all**: the detector's
  floor. The game waits until the board is larger again.
