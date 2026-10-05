# The reconciler's model (draft for review, 2026-10-05)

What the reconciler assumes about the screen and the game, and the one algorithm that
follows from it. Written before any code, to replace the cascade of special cases in
`state/reconcile.py` (single move, browsing, takeback, catch-up paths, queue undo, the
opponent's move alone, misread, stale retries, turn correction), each of which was added for
one failure and each of which can swallow a frame another needed.

## 1. What the reconciler may rely on

Only features every chess site must show, and that no theme or setting changes:

| Feature | Why it is robust |
|---|---|
| What stands on each square: empty, or a piece of a kind and a side | It is the game. Every site draws it. |
| Unknown | A square the vision cannot read (covered, dragged over, mid-animation, decorated) asserts nothing. |
| Whether a piece is settled (drawn centred on its square) | Geometry. A dragged or sliding piece is off-centre on every site. |
| The order of frames | Time only goes forward. |
| The rules of chess | Legal moves, and the sides alternate: a side never moves twice in a row. |
| Which side the user plays | The side at the bottom, read from the pieces and the coordinates. |

Never relied on, because a theme, a setting or another site changes them:

- every colour drawn over a square: the last-move highlight, the premove tint (red on chess.com, navy on lichess), the check glow, right-click marks;
- the colour of arrows, and the hover outline (chess.com only);
- the clock;
- the move list and any text.

They may still be read as hints that speed a decision up. A decision never depends on them:
the answer must be the same with every colour turned off. (Game-end badges are a separate
stage, not covered here.)

## 2. The hidden state

At any instant the screen is produced by a hidden state with three parts:

- **G, the game.** A start position and the moves played since. Its end is the real position
  P, with its side to move.
- **Q, the user's premove queue.** A sequence of moves by the user's side, queued while the
  opponent is to move. It may be empty.
- **V, the view.** The latest position, or an earlier ply of G when the user browses the move
  list.

## 3. How a state is drawn

The picture R(state) the site shows, square by square:

- **Browsing** (V is an earlier ply): that ply's position. The queue is empty: browsing
  cancels it on chess.com, and no site plays premoves while browsing.
- **Latest position, no queue:** P.
- **Latest position with a queue.** A site does one of two things:
  - **It draws the queue** (chess.com, `premove.move`/`v` in its board bundle). It starts from
    P and applies each queued step in queue order: lift the piece off its square, put it on
    the step's square, and overwrite whatever stood there, of either side. A step whose
    square no longer holds the user's piece (it was captured) is skipped. A promotion puts
    down the promoted piece. Castling puts the king on g/c and the rook on f/d. So the piece
    is drawn where its queue ends, its real square and every square it passed through are
    empty, and every piece on those squares is hidden.
  - **It does not draw it** (lichess: one premove, the piece stays where it is, only a tint).
    The picture is P.

  Both are allowed for every site. The model never needs to know which site it is.

A picture is consistent with a state when every square the vision read agrees with
R(state). Unknown squares agree with anything. A piece that is not settled agrees with
nothing: a frame with a piece in motion holds every decision it would touch.

## 4. What can happen between two frames

Any short sequence of these events, in order:

1. **The opponent moves** (opponent to move, latest view). G gains the move. Then, if Q is
   not empty, the first queued step is played at once if it is legal in the new position,
   and G gains it too. If it is not legal, the whole queue is cleared. (chess.com
   `consume()`; on lichess the same thing happens with a queue that was never drawn.)
2. **The user moves** (user to move, queue empty). G gains the move.
3. **The user edits the queue** (opponent to move). One step is appended, from where the
   piece is drawn. Steps are any move the piece could make on an empty board, blockers
   ignored, a pawn's diagonal always allowed (chess.com's generator `_`). Or the whole queue
   is cancelled.
4. **The user browses** to another ply, or back to the latest. On chess.com this cancels the
   queue.
5. **A takeback** removes the last one or two plies of G.
6. **A new game** starts: the start position, or a position adopted far from the game.

Each event has a cost: one for each move or queue edit, more for the rare ones (a takeback,
a new game, an adoption). A story is a sequence of events. Its cost says how much it asks
us to believe happened unseen.

## 5. The algorithm: keep every story that fits

The reconciler keeps a small set of hypotheses. Each is a state (G, Q, V) and the cost of
the story that led to it.

On every frame:

1. **Extend.** For each hypothesis, find the cheapest stories (up to a bound: four plies
   plus queue edits) whose resulting state draws a picture consistent with the frame. The
   search runs from the picture back, as now: the squares that disagree say which pieces
   moved, so only stories touching them are tried.
2. **Keep.** Keep the extensions whose cost is within a margin of the cheapest, up to a
   fixed number. A hypothesis no story extends is dropped. When none is left, the board is
   read whole and adopted (the position resync, as now).
3. **Settle.** A hypothesis must be consistent for the settle frames before any of its moves
   count. This is the same rule as now, now applied per hypothesis.
4. **Commit.** The coach is told the moves that every surviving hypothesis agrees on, in
   order: their common prefix. The side to move is told only when all hypotheses agree on it.
   When they disagree, it is "not known yet", and the coach does not hint for either side.

What disagreement looks like, and how it ends without a special rule:

- **The wrong side to move** (Tai's game, 2026-10-05). A queued step played on a picture
  nobody read leaves the same pieces one ply apart: two hypotheses, one per side. The next
  move by either side kills one, because the other side cannot move twice.
- **A premove versus a real move.** The user's piece moves while the turn is uncertain: one
  hypothesis reads a real move, the other a queue edit. The opponent's next move, or its
  absence, decides.
- **A queue's route.** The queue's steps are seen one at a time as the user drags the drawn
  piece on. A step queued between two frames leaves routes the picture cannot tell apart.
  Each route is a hypothesis. They differ only in the user's moves still to execute, and the
  steps as they execute (and the pieces they overwrite) prune them. Until then, only the
  common prefix is told.
- **Browsing versus a takeback**, a cancel versus an illegal step clearing the queue: both
  readings are kept until a later frame differs.

What this replaces:

- The ordered cascade. Order no longer decides anything: cost does. Two stories of equal
  cost both survive, and nothing is committed that one of them denies.
- Reading the red squares to undo a queue (`_unqueue`, `_routes`, `QUEUE_MEMORY_S`). The
  queue is part of the state and drawn forward, never undone from a colour.
- The turn correction and the clock's turn: the side to move is part of every hypothesis.

## 6. Guarantees

It may not follow every case: a picture can be truly ambiguous, or the vision can misread.
When it cannot follow, it must degrade, never break. These hold always, and the sweeps
assert them on every frame, not only at the end:

1. **Never a wrong side to move.** The coach is told a side to move only when every
   surviving hypothesis agrees. Otherwise it is "not known yet", and nothing is hinted,
   analysed for a side, or spoken for one.
2. **Never a move that was not played.** Only moves every hypothesis agrees on are told,
   and only after they have settled. A told move is never withdrawn, except by a takeback
   the board shows.
3. **Never an illegal game.** Every hypothesis is a legal game from its start. Two moves by
   one side in a row, or a move by the side not to move, are impossible by construction.
4. **Lost means re-read, not guess.** When no hypothesis fits for long enough, the board is
   read whole and adopted as a position with the side to move "not known yet", and the
   moves after it settle the turn. Being behind for a moment is acceptable; being wrong is
   not.

The order of failure modes, from acceptable to unacceptable: a move told late; the turn
"not known yet" for a move or two; a re-read that restarts the move numbering; a move told
wrong; the turn wrong. The last two must never happen. Failures show up in the logs and
dumps as the hypotheses that were alive, so they can be reproduced.

## 7. Proving it

- **The renderer** (`tests/synth.py`) draws queues with chess.com's own procedure (section
  3), and lichess's (not drawn). Every tint, highlight and check glow is drawn in random
  colours, and in none, on every seed. The result must be the same for all of them.
- **The premove sweep** queues the way chess.com allows: through pieces, onto the user's own
  pieces, pawn diagonals onto empty squares, castling. It cancels, browses, and lets the
  opponent capture a queued piece. It follows chess.com's code for each, and lichess's.
- **Every existing sweep and scenario** (stress, geometry, takebacks, slides, cold start,
  puzzles, the recordings) is the acceptance bar, unchanged. Nothing moves to opt-in.
- **The assertion** at the end of every sweep game: after the next move, the game told
  equals the game played, side to move included.
- **Speed:** time per frame is measured against the current reconciler on the recordings,
  and must not be worse at the median.

## 8. Settled with Tai (2026-10-05)

- It need not follow every case, but it must never break the app. A wrong turn order is
  completely unacceptable. So the coach stays silent while the turn is not known, and only
  the common prefix of all hypotheses is told (section 6).
- Robust features only: no colour, no clock, no site-specific drawing is ever required
  (section 1).
