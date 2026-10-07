# The reconciler's model

What the reconciler assumes about the screen and the game, and the algorithm that follows
from it. Built in `src/sidecoach/state/reconcile.py` (`Reconciler`) and
`src/sidecoach/state/stories.py` (`Story`, `Search`); the phone's `Reconcile.kt` is its 1:1
port, held to it by the parity fixtures ([testing.md](testing.md#parity-fixtures-for-the-phone)).
Where it sits in the pipeline: [DESIGN.md](../DESIGN.md) and [app.md](app.md).

The model was written on 2026-10-05, before the code, to replace a cascade of special cases
(single move, browsing, takeback, catch-up paths, queue undo, the opponent's move alone,
misread, stale retries, turn correction), each added for one failure and each able to
swallow a frame another needed. It was built weight-free on 2026-10-05/06 and is described
here as it stands (2026-10-07, after the audit's fixes to the turn told, the record kept
through a correction and the game ended on every replacement). Section 8 records what was
settled with Tai.

## 1. What the reconciler may rely on

Only features every chess site must show, and that no theme or setting changes:

| Feature | Why it is robust |
|---|---|
| What stands on each square: empty, or a piece of a kind and a side | It is the game. Every site draws it. |
| Unknown | A square the vision cannot read (covered, dragged over, mid-animation, decorated) asserts nothing. |
| Whether a piece is settled (drawn centred on its square) | Geometry. A dragged or sliding piece is off-centre on every site. |
| The order of frames | Time only goes forward. |
| The rules of chess | Legal moves, and the sides alternate: a side never moves twice in a row. |
| Which side the user plays | Taken to be the side at the bottom (`pipeline.py` builds every `Reconciler` with `user_side=bottom`). |

Not relied on for a move, because a theme, a setting or another site changes them:

- every colour drawn over a square: the last-move highlight, the premove tint (red on
  chess.com, navy on lichess), the check glow, right-click marks;
- the colour of arrows, and the hover outline (chess.com only);
- the clock;
- the move list and any text.

What the search decides -- which moves were played, the queue, the ply shown -- never
depends on them: the stories are checked against the pieces on the squares and nothing else.
Some of these are read all the same, each in one named place:

- **The hover outline** holds decisions back: while it shows, the frame is a drag in
  progress (`_blocked`), no move is witnessed and no position is re-read. It never makes
  a story.
- **The site's last-move mark** (the highlighted squares, `highlighted_squares`) is read:
  - by the recovery of section 4 (`_marked`, `_marked_resync`);
  - whenever a position is adopted from the screen (the re-read, the cold start, a puzzle
    read cold), to name the side to move and the en-passant square (`_adopt_placement`,
    `last_move_seen`);
  - to say which move joins a board read a moment earlier to the one adopted (`one_move`'s
    `lit`: with two squares lit, the move must be theirs);
  - as part of what the re-read holds still (the candidate's key is the placement and the
    lit squares), and of what makes a disagreeing cell "the same" frame to frame
    (`_advance_stale`).
- **The clock** settles a side to move left open by a re-read or a cold start
  (`pipeline._turn_from_clock` -> `Reconciler.turn_from_clock`): the running clock's colour
  is taken, when that side may legally be to move and nothing has been played since.

(Game-end badges are a separate stage, not covered here.)

## 2. The hidden state

At any instant the screen is produced by a hidden state with three parts:

- **G, the game.** A start position and the moves played since. Its end is the real position
  P, with its side to move.
- **Q, the user's premove queue.** A sequence of moves by the user's side, queued while the
  opponent is to move. It may be empty.
- **V, the view.** The latest position, or an earlier ply of G when the user browses the move
  list.

A story (`Story`) is one such state past the record: `head` (what was done to the record:
`("newgame",)`, `("back", n)`, `("flip",)`, `("fix", square, kind)`), `moves`, `queue`,
`view`, and its `cost`. It also carries `flight` (a piece caught on its way, section 4),
`hidden` (an opponent's move the queue hides) and `steps` (steps through the move list).

## 3. How a state is drawn

The picture the site shows, square by square (`Reconciler._shown`):

- **Browsing** (V is an earlier ply): that ply's position; no queue is drawn. Browsing
  cancels the queue on chess.com, and no site plays premoves while browsing. The reconciler
  keeps the queue under the ply all the same (a frame it reads as a ply browsed may be
  none), and the live board shown again without it is the queue cancelled.
- **Latest position, no queue:** P.
- **Latest position with a queue.** A site does one of two things:
  - **It draws the queue** (chess.com, `premove.move`/`v` in its board bundle;
    `stories.premove_picture`). It starts from P and applies each queued step in queue
    order: lift the piece off its square, put it on the step's square, and overwrite
    whatever stood there, of either side. A step whose square no longer holds the user's
    piece (it was captured) is skipped. A promotion puts down the promoted piece. Castling
    puts the king on g/c and the rook on f/d. So the piece is drawn where its queue ends,
    its real square and every square it passed through are empty, and every piece on those
    squares is hidden.
  - **It does not draw it** (lichess: one premove, the piece stays where it is, only a tint).
    The picture is P.

  Both are allowed for every site. The model never needs to know which site it is.

A picture is consistent with a frame when every square the vision read agrees with it
(`Evidence.wrong`, `cell_agrees`). Unknown squares agree with anything; a piece read without
a kind asserts only that the square is occupied. A piece that is not settled is left out of
the matching and witnesses no move: a frame with a piece in motion holds every decision it
would touch.

## 4. What can happen between two frames

Any short sequence of these events, in order:

1. **The opponent moves** (opponent to move, latest view). G gains the move. Then, if Q is
   not empty, the first queued step is played at once if it is legal in the new position,
   and G gains it too. If it is not legal, the whole queue is cleared. (chess.com
   `consume()`; on lichess the same thing happens with a queue that was never drawn.)
2. **The user moves** (user to move, queue empty). G gains the move.
3. **The user edits the queue** (opponent to move). One step is appended, from where the
   piece is drawn. Steps are any move the piece could make on an empty board, blockers
   ignored, a pawn's diagonal always allowed, castling while the rights allow it, a
   promotion to a queen (chess.com's generator `_`; `stories.lone_moves`). Or the whole
   queue is cancelled.
4. **The user browses** to another ply, or back to the latest. On chess.com this cancels the
   queue.
5. **A takeback** removes the last one or two plies of G.
6. **A new game** starts: the start position, or a position adopted far from the game.

### What a story costs

A story's cost is the pictures it needs that no frame showed (`stories.unseen`). Each event
between two frames makes a picture and only the last is seen, except that a move and the
instant reply to it, by either side, are one picture: a premove of the user's, a puzzle's or
a bot's answer. Two to a picture at most: a reply to a reply is a picture of its own.

Nothing is weighed (Tai, 2026-10-06: no priors, no heuristics). Of the stories that need the
fewest unseen pictures, the ones kept are those that assume the fewest **events**
(`_events`): each op done to the record (a takeback, a new game, a turn flipped, a piece's
kind corrected), each move, a premove queue as one however many steps (where it ends is
drawn, the way it went never is), and each step through the move list (a ply shown, another,
the live board again: `Story.steps`). As built:

- A frame's events may need at most one unseen picture when frames follow each other, and up
  to `max_plies` (4) after a gap: the board unseen on a tick, or a frame over `GAP_S` (0.5 s)
  after the last.
- A story told afresh from the record, throwing away the stories kept, costs a picture
  (`Story(cost=1.0)`). It is tried only when no story kept fits and some story kept had
  moved past the record.
- A story left alone (one story) has its cost reset to 0: its past is the past.

### Stories exempt from the event count

- **A piece in flight** (`_in_flight`). A story whose frame shows exactly one of its pieces
  off its square (read empty), and at most one other square wrong, is that story with the
  piece on its way when:
  - the piece is of the side to move in the story's picture: only a move slides a piece,
    and a premove is not drawn sliding until it fires;
  - it is read, by its kind, on a square the story has empty that a **legal move** of it
    passes over -- within half a square of the straight line from its square to the move's
    end, past its square (`_on_way`) -- or it is read nowhere (its square empty and no other
    square wrong: a negative mark of this frame, which no later frame can repeat);
  - read on its way in the frame before, the new square lies further along the line from
    its square through the old one;
  - it was not read settled on that same square in the frame before as well: centred on one
    square two frames running, it has stopped there.

  It assumes nothing yet and is kept beside the rest. On the next frame its story goes on
  only by a move that passes over (or lands on) where it was read (`_found`); or, for the
  user's own piece only, by no move at all (a drag let go). An opponent's piece is never
  "put back". A piece read nowhere sets no such bound.
- **A ply browsed with its move on its way** (`_browsed_in_flight`). A frame that is an
  earlier ply of a story's game, with that ply's move caught between its two squares, either
  way (played forward through the list, or slid back by a takeback or a step back), is that
  ply browsed. The move's square must read empty and the piece be read on its way, short of
  the far end: "read nowhere" never makes a ply browsed.
- **A move the queue hides** (`Search._unseen_replies`). When a story's drawn queue already
  matches the frame and the opponent is to move, each opponent's move whose squares the
  drawn queue covers whole (and the queued step it sets off) is kept as a story of its own
  beside the one without it (`hidden`): no picture could show it, and none says it did not
  happen. The moves after tell them apart.
- **A story whose position ends the game** (`result_on_board`: mate, stalemate, dead
  material, the third repetition, fifty moves) is kept whatever it assumes: the end shown (a
  badge, a dialog, the next game) decides (`pending_terminal`).

### Filters on the stories kept (`_kept`)

- **Visible before hidden** (Tai, 2026-10-06). When a story with no drawn queue (and no piece
  in flight) fits, the stories that need a drawn queue are dropped; a real queue fires with
  the opponent's next move. This also keeps browsing and takebacks ahead of a premove that
  draws the same pieces.
- A queue that draws nothing (played out, every piece where it stands) is no story.
- A side to move flipped is told by a move played, never by a queue alone; with the side to
  move open, a piece on its way also counts as that move.
- A piece's kind corrected (`fix`) only by a story that moves that piece.
- In the stale retry, a story may not contradict an exact reading among the cells ignored
  (`_spares`).
- A king landed on a square read as a decorated piece (the check glow, a badge) must end the
  story in check (`_decorations_hold`).
- Stories that end in one state (position, side to move, castling rights, queue, ply shown,
  repetition count: `Story.key`) are kept as one, the simplest by `_story_order` (cheapest;
  a takeback or new game before moves that retell it; fewer ops; shorter queue). At most
  `MAX_STORIES` (12) are kept, the first story of each side to move always among them.

### Browsing and takebacks

- A ply browsed is tried only for stories whose head is takebacks alone, and only up to the
  plies the screen has shown: the record's, and the story's own moves as far as each has
  been seen settled `settle_frames` times. A premove fired and never shown at rest is no ply
  to browse back to.
- A ply browsed keeps the story's queue under it, and what earlier frames showed of it.
- A move played from the ply shown is a takeback (or, from the start position of a game begun
  there, a new game); it is tried only when no ply browsed fits. The move the game itself
  played from that ply is the live board again, never a takeback.

### Recovery by the site's last-move mark

(Tai, 2026-10-06: "a good and reliable recovery mechanism".) On a site that draws premoves,
the user's moves and the same moves premoved, each fired by the opponent's next, draw every
frame alike: a record a ply out is never put back by the pieces alone. The squares the site
marks for the move just played tell them apart (a premove is not that move; chess.com's red
premove tint is not read as a mark).

- `_marked`: when the frame marks squares, stories fit, and no drag is under way, and no
  story that fits has its last move's two squares marked, the stories are told again with
  the side to move the other way (a `flip` root) and kept if one of them, with a move and no
  ply browsed, has it. A story with a queue, a ply browsed or a piece in flight, with no
  move, or whose last move is a castling, counts as agreeing with the mark (each carries its
  own marks, or says nothing).
- `_marked_resync`: when none told the other way has it either, the frame is counted. Counted
  frames must show one still board (the same pieces and the same marks), all 64 squares
  read, every piece of a known kind and settled, and no hover. After `settle_frames + 1` (3)
  such frames, while the game is not finished, the board is read whole and adopted with the
  side the mark names, the turn no longer open -- unless:
  - the mark does not settle the side to move;
  - the board is the record already;
  - it differs from the record on `resync_jump_cells` (12) squares or more: another position
    (a new puzzle or game), which the ordinary re-read begins afresh (Tai's puzzles,
    2026-10-06);
  - it is a position the game stood in, reached by a move whose two squares are marked:
    that is a ply shown or a takeback under way (the history guard).

A frame with nothing marked, or a site that marks nothing, changes nothing here.

## 5. The algorithm: keep every story that fits

The reconciler keeps a small set of stories, `_stories`, starting from the record's own
(`Story()`).

On every frame (`Reconciler.update`):

1. **Extend** (`_fitting`, `_found`, `Search.extend`). For each story kept, cheapest first,
   find the cheapest ways it goes on to draw the frame. Iterative deepening, shortest first:
   0 to `max_plies` (4) moves, the sides in turn, each move touching a square the picture
   gets wrong or one the queue hides (or, for one move of a story at most, only squares the
   frame cannot read: a king under the mate's badge) and witnessed (every square it changes read, settled, not under a drawing or the cursor;
   while the user drags, only the opponent moves). At each length the moves alone are tried,
   then with the queue edited (cancelled, grown by up to `max_queue_steps` (3) steps, or
   cancelled and begun again), and, with a queue standing, with the queue cancelled first.
   The search runs from the picture back: the squares that disagree say which pieces moved.
   It is bounded by `search_budget` (4000 positions a story) and `EDIT_BUDGET` (300 for one
   way the queue stood), and by a ceiling: a story that can only cost more than the cheapest
   found is not searched. A position's picture (`_wrong_at`) and the moves worth trying from
   it (`_steps`, ordered by what they leave wrong) are worked out once a frame, keyed by the
   position and the queue, and shared by every path and every pass of the deepening.
   While the side to move is open (`turn_unknown`), the record's story is also tried flipped.
   Plies browsed and takebacks are added (section 4), and the mark is checked (`_marked`).
2. **Keep** (`_kept`): the stories with the fewest unseen pictures, of those the fewest
   events, plus the exempt ones, filtered and one to each state (section 4).
3. **Settle** (`_decide`, `_seen_ops`). For every run of ops a story begins with, the frames
   that showed it settled are counted (`_seen`), for every story whether they agree yet or
   not. A move counts as seen on a frame only when the squares it changed are read settled
   in it and the moved piece is not read further along its line (`_sliding_on`). A frame
   showing a ply browsed confirms nothing of the story underneath, and what earlier frames
   showed of it stands.
4. **Commit** (`_decide`, `_play`):
   - While any story shows a ply browsed, nothing is played; the ply is shown (`viewing`).
   - When every story ends in one state, the record follows the simplest.
   - Otherwise it follows the ops every story begins with -- or, when the stories all have
     one side to move and draw one board and one of them has no queue (and every other can
     be told again from it), that one (`_shown_alike`): the board shown is its position.
   - An op is played once it has been seen settled `settle_frames` (2) times; one more in
     the cautious retries below. Every story is then told again from the new record
     (`rebased`, or `_retold` as a takeback of what it did not share).
   - When the stories disagree on the side to move, `TurnPending` says so (`turn_pending`,
     `_tell_turn`); the coach then says nothing unasked and holds questions until it is
     known again (`Coach`, `_turn_pending`). What is compared is what the coach holds, not
     what was last sent: an event that tells a position whole (GameStarted, Takeback,
     Resync, GameEnded) makes the coach take the turn as known, so it clears "told pending"
     too (`_told_whole`), and a frame that plays a takeback while the stories still part
     tells `TurnPending` again after it (2026-10-07).

When nothing fits the frame, in order:

1. A drag in progress: the stories wait for the drop as they are; the frame counts as
   unexplained.
2. **The stale retry.** Cells that have read the same wrong thing for more than
   `stale_after` (5) frames are ignored, one or two at a time, junk readings first (a badge,
   a ring), with junk cells as wildcards; the story found is played cautiously. Fewer than
   `resync_cells` (3) stale cells, all of the disagreement, are ignored outright.
3. **A misread piece** (`_misread_roots`, `fix`): a piece of a position read from the screen,
   never moved since, that left its square, tried as each other kind. Played, it keeps the
   game: the piece has stood on its square since the base, so the base is corrected and the
   moves since are played again from it (`_replayed`), their numbering kept; no `Resync`, the
   position told by `PositionChanged` and the move that showed the kind. Only when there is
   no move to keep, or one does not replay, is the position adopted again (a `Resync`).
4. **The side to move flipped** (`flip`): last of all, the record's turn read the other way.
   With the turn open (a position adopted) the first move settles it silently; otherwise
   it is a `Resync` ("side to move corrected"), the game told again from the position: the
   sides alternate, so the moves before it never replay from the base with the other side to
   move.
5. **The start position back on the board**: a new game (the old one ended first, with a
   pending terminal story if one proves its end).
6. **The position re-read** (`_maybe_position_resync`): a board that disagrees on at least
   `resync_cells` (3) squares, all 64 read, no hover, every piece settled and at most
   `resync_max_soft` (2) of no known kind (their best guess stands in), held still for
   `resync_hold_frames` (10) frames -- or, when `resync_jump_cells` (12) or more squares
   differ, for `resync_min_frames` (3) frames over `resync_hold_s` (0.4 s). Not within
   `QUEUE_RESYNC_S` (1 s) of a frame that fitted a story with a queue drawn. The board is
   made a position by `_adopt_placement` (castling rights from the home squares; the side
   to move from legality, then the mark, then the bottom side). A board shown briefly just
   before it, one legal move away, is taken as its start and the move played
   (`_opening_move`: a puzzle's first board and the site's opening move).
7. Otherwise the frame is unexplained (`_maybe_resync`): after `unexplained_note_frames`
   (20) frames the log gets each square's reading (`Unexplained`); after `resync_after`
   (150) with at least three persistent cells, an `Unexplained` with `told=True`. The game
   is kept as it stands; this path never adopts a position.

What disagreement looks like, and how it ends without a special rule:

- **The wrong side to move** (Tai's game, 2026-10-05). A queued step played on a picture
  nobody read leaves the same pieces one ply apart: two stories, one per side. The next
  move by either side kills one, because the other side cannot move twice. Where the pieces
  never part them (section 4), the mark does.
- **A premove versus a real move.** The user's piece moves while the turn is uncertain: one
  story reads a real move, the other a queue edit. Visible before hidden keeps the real
  move; the opponent's next move, or its absence, settles it.
- **A queue's route.** A step queued between two frames leaves routes the picture cannot
  tell apart; a queue is one event however it went, and the routes that end in one state
  are one story. The steps as they execute (and the pieces they overwrite) tell them apart.
- **Browsing versus a takeback**, a cancel versus an illegal step clearing the queue: both
  readings are kept until a later frame differs.

What this replaced (the cascade before 2026-10-06):

- The ordered cascade. Order no longer decides anything: cost, then events, do. Two stories
  of equal standing both survive, and nothing is committed that one of them denies.
- Reading the red squares to undo a queue (`_unqueue`, `_routes`, `QUEUE_MEMORY_S`, all
  gone). The queue is part of the state and drawn forward, never undone from a colour.
- The turn correction as a rule of its own: the side to move is part of every story. The
  flip remains as a story (`flip`), tried last when nothing else fits and by the mark's
  recovery; the clock still settles a turn left open (section 1).

## 6. Guarantees

It may not follow every case: a picture can be truly ambiguous, or the vision can misread.
When it cannot follow, it must degrade, never break. The sweeps assert these on every frame,
not only at the end (section 7):

1. **Never a wrong side to move told while the stories disagree.** While the stories kept
   disagree on the side to move, `TurnPending` is sent and the coach hints nothing unasked
   and holds questions. Only the stories' disagreement is told: a position adopted with
   both sides legal to move takes the mark's side or else the bottom side (`turn_unknown`
   set, no `TurnPending`), and a flipped story with no move is dropped (section 4), so the
   open turn is settled by the first move or the running clock, not announced.
2. **The position current once premoves resolve.** It may be misread briefly, but by the
   next move or two, and whenever premoves execute or are cancelled, the record is the
   game's (Tai, 2026-10-05). Which moves got there is third: told as right as the pictures
   allow, never at the cost of the first two.
3. **Never an illegal game.** Every story is a legal game from its start: the search plays
   only legal moves, the sides alternate, and `_commit` refuses an illegal move (an
   `Unexplained`, the stories cleared).
4. **Lost means re-read, not guess.** When no story fits and the board is still and whole,
   it is read and adopted as a position (section 5, step 6): the move numbering restarts
   from it. Being behind for a moment is acceptable; being wrong is not.
5. **The record is always a game.** The game kept (`pgn`, `GameEnded.pgn`) plays out from
   where it begins: a base other than the start position is named in its SetUp and FEN
   headers. A correction that keeps the game (a piece's kind, `fix`) keeps every move; one
   that cannot (a side to move flipped after moves, a board adopted by a re-read or by the
   mark) starts the record at the board corrected. The library takes the longer of the moves
   recorded before the first re-read and the game kept to the end (`library.watched_row`).
   Every path that replaces the reconciler ends its game first, with the mating story the
   stories leave when the dialog covered it (`pipeline._end_in_progress`: the next game, the
   board turned round, the board re-read).

The order of failure modes, from acceptable to unacceptable: a move told late; the turn
open for a move or two; a re-read that restarts the move numbering; a move told wrong; the
turn wrong. The last two must never happen. Failures show up in the logs and dumps as the
stories that were alive, so they can be reproduced.

## 7. Proving it

- **The renderer** (`tests/synth.py`) draws queues with chess.com's procedure
  (`premove_picture`, section 3) and lichess's (not drawn). The premove sweep draws the
  premove tint and the last-move highlight in each seed's own colours, or no premove tint
  at all.
- **The premove sweep** (`tests/test_premove_sweep.py`) queues the way chess.com allows:
  through pieces, onto the user's own pieces, pawn diagonals onto empty squares, castling.
  It cancels, browses, and lets the opponent capture a queued piece, following chess.com's
  code for each, and lichess's.
- **The flow matrix** (`tests/test_flow_matrix.py`): game or puzzle x takebacks x looking
  back x premoves, all 16 combinations, on the premove sweep's site.
- **Every other sweep and scenario** (stress, geometry, takebacks, slides, cold start,
  puzzles, the recordings) is the acceptance bar.
- **The assertions** of every sweep game: on every picture, a turn the stories disagree on
  is told pending; whenever the record says it has caught up with the screen, it holds the
  game's position and side to move; at the end, once the board is still and the opponent
  has moved after the user, the game exactly. Moves told that were not played are counted,
  not asserted.
- **How to run them:** [testing.md](testing.md#the-sweeps).
- **Speed** (2026-10-05, before the search caches): time per frame against the old
  reconciler, 20 stress games and 40 premove games: medians 0.44 and 0.32 ms (old 0.21 and
  0.18), 95th percentiles 1.2 and 1.6 ms (old 0.43 and 4.0), worst frames 92 and 318 ms (old
  4 and 996). The caches of 2026-10-06 (`_wrong_at`, `_steps`) took a frame of the slowest
  reconciler tests from 31 ms to 16 ms.

Current results (2026-10-06, after the mark's recovery, 20bdeb8 / 077fab4):

| Sweep | Seeds | Passing | The rest |
|---|---|---|---|
| premove sweep | 1000 | 997 | 742 wrong for one ply, then recovers; 70 and 956 repeat an earlier position by the same move, which the history guard holds to be a ply shown |
| stress (criteria) | 1000 | 997 | 543, 604, 817, pre-existing |
| flow matrix | 320 (20 seeds x 16) | 319 | seed 20, pre-existing |

## Known limits

- **The user's side is the bottom side.** Every reconciler is built with `user_side` =
  the side at the bottom; a board watched from the other side (a game spectated flipped)
  makes the opponent's moves the "user's", and their premove queue the one modelled.
- **The premove sweep's three seeds and the stress sweep's three** (table above). Seeds 70
  and 956: when the game goes back to a position it stood in by the same move the site
  marks, the mark's recovery takes it for a ply shown and does not re-read it.
- A bot answering faster than the capture, with every other picture dropped, makes a7-a6
  and a6-a5 draw what a7-a5 draws; a queue whose last step mates then stays a step behind
  when the dialog covers the board
  (`tests/test_premoves.py::test_a_long_queue_against_a_bot_faster_than_the_capture`,
  strict xfail).
- A premove that takes a piece straight back to the square it came from draws the move list
  stepped back, or the user's own move taken back: it is read as browsing (Tai: a takeback
  and browsing must always be read right), and put right when the opponent's next move
  plays the queue.
- A tint near the board's own colours can make the vision read a queued square as covered;
  the reconciler then sees nothing there.
- A position adopted with both sides legal to move shows the mark's side or the bottom side
  until a move or the running clock settles it; no `TurnPending` is sent for it (section 6).

## 8. Settled with Tai (2026-10-05, 2026-10-06)

- It need not follow every case, but it must never break the app. A wrong turn order is
  completely unacceptable. So the coach stays silent while the turn is not known, and only
  the common prefix of all stories is told (section 6).
- Robust features only: no colour, no clock, no site-specific drawing is ever required
  (section 1).
- The order of what matters: the side to move first, never wrong; then the position current
  once premoves resolve or are cancelled; then the moves themselves, as right as can be
  ("I don't think that's possible from the visuals alone").
- A takeback and browsing must always be read right; a premove may be read a bit wrong.
- 2026-10-06: no priors and no heuristics: a story is costed by the pictures nobody saw,
  then the events it assumes, and nothing is weighed. Visible before hidden: a story that
  needs no drawn queue goes before one that does.
- 2026-10-06: the site's last-move mark is "a good and reliable recovery mechanism", read
  only to recover (section 4).
