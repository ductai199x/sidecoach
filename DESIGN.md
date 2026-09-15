# Sidecoach design

Status: draft v0.1, 2026-09-14. Owner: Tai. This document is the source of truth for scope and architecture until code supersedes it. Decisions are dated; change them by editing this file.

## 1. What it is

Sidecoach is a background service that watches the screen, finds the chessboard, and coaches the player while they play, practise, or review on any chess site. It is vision-first and platform-agnostic: it reads pixels, never the site's DOM. It maintains a legal, reconciled game state, annotates every position with Stockfish and a motif tagger, and talks to the player through an LLM that only ever sees verified facts, at a disclosure level the player controls. A long-term player model drives post-game review, targeted puzzles, and lesson recommendations.

First target: Windows, chess.com, text coaching in bot games.

## 2. Goals and non-goals

Goals

- Improve the player's chess, measured by per-motif error rates, targeted-puzzle accuracy, and eval loss at decisive moments. Engagement is not a goal.
- Socratic by default. Before a move the coach asks and hints; it does not hand over moves unless asked.
- Never commit an illegal transition, never let the LLM invent a board fact, never leak beyond the current disclosure level.
- Work on any board theme, piece set, and board size on chess.com first, other sites later, without per-site integration.

Non-goals

- Playing moves for the user, automating the browser, or reading site internals.
- A cloud service. Everything runs locally except an optional hosted LLM fallback.
- Mobile in the first phases. Android and iOS come after the desktop pipeline is proven.

## 3. Policy

Decided 2026-09-14.

- Sidecoach is a generic tool. Where and how it is used is the user's own responsibility, the same as running any engine app beside a game.
- Live engine-informed coaching is always available. There is no structural gating by opponent type or game mode.
- Before a move the coach discloses only questions and hints. The best move, or the set of near-equal candidates, is withheld until the user names it, plays it, or explicitly asks for it.
- An EULA is the last step before any public release, not part of the personal-project phases.

## 4. Architecture

One process, a pipeline of stages connected by typed events. Every stage consumes events and emits events, so any stage can be replayed from a recording.

```mermaid
flowchart LR
  CAP[capture] --> VIS[vision]
  VIS --> STA[state]
  STA --> ANA[analysis]
  ANA --> COA[coach]
  COA --> UI[ui / terminal]
  UI --> COA
  COA --> PRO[profile]
  PRO --> COA
  ANA --> PRO
  REC[(recordings)] -.-> CAP
  CAP -.-> REC
  LLM[(local LLM)] <--> COA
  SF[(Stockfish)] <--> ANA
  DB[(SQLite)] <--> PRO
```

| Stage | Package | Consumes | Emits | Responsibility |
|---|---|---|---|---|
| capture | `sidecoach.capture` | timer | `Frame` | Screen frames at 10 fps, all monitors until a board is locked, then the board region only. Pixel-diff gating. Optional recording to disk. |
| vision | `sidecoach.vision` | `Frame` | `BoardLocated`, `BoardLost`, `SquareObservation`, `ClockObservation`, `ScreenState` | Board detection, per-square classification, orientation, clock and move-list OCR, screen-state inference. |
| state | `sidecoach.state` | vision events | `PositionChanged`, `MovePlayed`, `GameStarted`, `GameEnded`, `Takeback`, `Resync` | Reconciliation into a legal python-chess game. Move detection over legal moves. Cold-start and resync. |
| analysis | `sidecoach.analysis` | `PositionChanged`, `MovePlayed` | `Analysis`, `MoveAssessment` | Stockfish multi-PV, expected-score eval loss, decisive-moment flag, threats, motif tags, opening identification. Continuous analysis during the user's think time. |
| coach | `sidecoach.coach` | analysis events, `UserUtterance`, clock | `CoachMessage`, `ProfileEvidence` | Disclosure state machine, interjection policy, fact-bundle filtering, LLM calls, claim verification in both directions. |
| profile | `sidecoach.profile` | `ProfileEvidence`, `GameEnded` | `ProfileSnapshot`, `Recommendation` | SQLite storage, error taxonomy counts, session summaries, recommendations. |
| ui | `sidecoach.ui` | `CoachMessage` | `UserUtterance` | Terminal first, companion panel second, on-board overlay third. |
| voice | `sidecoach.voice` | audio, `CoachMessage` | `UserUtterance` | Later. Local STT, TTS, VAD. |
| replay | `sidecoach.replay` | recordings | `Frame` | Record and replay harness. Recordings are the test suite. |
| llm | `sidecoach.llm` | prompts | completions | Provider interface. Local server first, hosted Claude as a swappable fallback. |

Events are pydantic models in `sidecoach.events`. Every event carries a monotonic timestamp and the frame id it derives from, so latency can be measured end to end.

## 5. Stage designs

### 5.1 Capture

- Backend protocol: list monitors, grab a region as a numpy BGR array. First backend is `mss` (GDI BitBlt). Windows Graphics Capture through `windows-capture` is the planned upgrade: it captures a window even when overlapped and can exclude the cursor. `dxcam` (DXGI Desktop Duplication) is the alternative if CPU cost matters. None of them draw the cursor into frames unless asked.
- The cursor, user-drawn arrows and circles, and highlights are local outliers against 64 cells of evidence; detection scores the whole grid and tolerates them. The square classifier is trained with those overlays as augmentations.
- 10 fps. At 5 fps the frame interval alone eats the 200 ms latency target.
- Until a board is locked, scan every monitor at a reduced rate. After lock, capture only the board region plus a margin, and validate the lock each frame. Re-scan on validation failure.
- Pixel-diff gate: skip vision when the region is unchanged.
- A live session must not die with the screen. A grab that fails (a locked session, a sleeping monitor, a display change) is retried every half second with the capture rebuilt, up to thirty times in a row, before the loop gives up with a clear message; the frames are simply missed and the reconciler catches up. A recorder that fails (a full disk, a frame of the wrong shape) stops the recording and the watching goes on. A bad monitor index is refused with the count found; the default monitor is the primary, the one at the desktop origin, which `doctor` marks.
- Recording: full frames to a rolling window on disk during development, plus board crops labelled with the reconciled FEN when reconciliation is confident. Both configurable. A crops-only default is reserved for a later public release.

### 5.2 Vision

Board detection is classical computer vision. A chess.com board is an axis-aligned square split into an 8 by 8 grid of equal cells in exactly two theme colours.

1. Edge map, then 2D autocorrelation or projection profiles to find the dominant cell pitch.
2. Candidate outer squares at that pitch. Score each on checkerboard fit using the two dominant colours of its empty cells.
3. Prefer the largest candidate above the score threshold. Lobby and puzzle pages show several boards.
4. Lock the region. Re-detect only when the per-frame validation fails.

A small learned detector is a fallback for themes that defeat the classical path, not the primary method. Known hazards to test for: marble, glass, and 8-bit themes; last-move and check highlights; coordinate labels inside edge cells; partial occlusion by dialogs.

Square classification: a small CNN with 13 classes, six pieces in two colours plus empty, exported to ONNX and run on the 64 cell crops. Training data comes from synthetic renders using open-licensed Lichess piece sets with heavy augmentation, plus board crops harvested from the user's own sessions. Chess.com assets are proprietary and are only observed from the user's own screen.

Orientation is never assumed. Chess.com puts the user's pieces at the bottom by default, but a setting can pin white to the bottom or top, so the bottom side can be either colour (Tai, 2026-09-14). It is decided per game from the side whose pieces occupy the bottom two ranks at the start position, cross-checked against the coordinate labels, and a mid-game flip shows up as the entire piece pattern mirroring.

Window moves and resizes mid-game are handled by the tracker: per-frame validation fails within three frames, a narrow-scale search around the last cell size re-locks a moved window quickly, and the full multi-scale search runs only if that fails. A periodic wide re-fit catches sub-cell drift and small resizes, which the checker fit is blind to for up to a third of a cell; it is adopted only when it passes on its own and its boundary comb is clearly stronger than at the held geometry, since the fit score is quantised to a cell and a badge can make the wrong geometry score higher. A board that slid by a whole cell fails validation at once, because the locked colours carry the theme's polarity (which cells are light), and goes through re-detection. Second review (2026-09-14): the polarity also holds for a theme whose light squares are the darker ones; coarse candidates are the local maxima of the checker response per scale, so an empty board's plateau of perfect scores can no longer crowd out the game board at the same cell size; among boards of one size a better fit wins, then the more occupied one.

Review outcomes (2026-09-14, first line-by-line pass): the fit score reads the ring of cells just outside a candidate and rejects it when the checker pattern continues on any side, which is what tells a whole-cell slide or a window of a larger checkerboard from a board; a lock keeps the square colours and tolerance it was taken with, so a dialog over most of the board cannot rewrite its palette, while a fresh palette that validates re-locks the colours when the theme changed between games; a held lock reports the fit it measured, and the pipeline reads nothing from a lock the frame did not validate; the coarse response is computed on cell border rings and gated on the checker amplitude, so pieces do not drag it and background does not inflate it; the boundary comb sees only the strips around the cell boundaries, goes coarse to fine, and costs about 14 ms instead of 105 ms per re-fit at 1440p. The recogniser refuses a start frame in which any piece is not plausibly drawn on its own square (a cursor or outline on the king or queen would have been its exemplar for the whole game) and fuses the pawns and the pairs by majority vote; a tinted block is an overlay only when it is a filled rectangle, so a file of marks or a queued premove stays readable. Every cell is upscaled to a 96-pixel patch before analysis, so board size does not matter down to about 32 pixels a cell (a 256-pixel board): the six recordings replay identically at 720p (43-pixel cells), and at 32 pixels the live game of 16:49 has no wrong ply, with one 15-ply stretch bridged by a position resync instead of followed move by move. Below that a black piece's light outline outweighs its fill. The hover outline is thinnest on its left side, about a pixel narrower than the others, so the hover test needs the median side over the tolerance and the weakest over half of it.

What the game window does during a session is covered by an end-to-end scenario suite on the synthetic board (tests/test_scenarios.py, 2026-09-14): the window moved or resized mid-game, the board fully covered by another window while plies happen underneath, partly covered while a move lands on the visible side and then on the covered side, minimised for seconds and restored, a toast over a corner, the page scrolled so the board is partly off screen, and gaps of two and four plies. The rule that made the covered and scrolled cases pass: while the lock is lost, the cheap search near the last cell size runs every few frames, so a board that comes back re-locks within a few frames, and only the full multi-scale search backs off.

OCR: RapidOCR for the move list and player names, digit templates for clocks. OCR is a resync and cold-start aid, not the primary move source.

Screen state: derived from board presence, running clocks, and a few layout cues. No separate classifier in the first slice.

### 5.3 State reconciliation

The core idea: detect moves, not positions. Given a known legal position there are about thirty legal moves. Only the squares that changed need classifying, and the legal move most consistent with the diff wins. Full FEN reconstruction is only for cold start and resync.

- Settle: commit a move only after two consecutive frames agree.
- If no legal move explains the diff, hold. If the disagreement persists past a threshold, trigger resync: full-board classification, move-list OCR, and a search for a legal history that reaches the observed position.
- Edge cases handled explicitly: drag in progress empties the source square; premove highlights without moving; the promotion dialog; the red check highlight; hover highlights; user-drawn arrows and circles; capture animations; takebacks, which reverse the last move; rematch and new game, detected by the initial position; board flip; board resize.
- Cold start loses castling and en passant rights. The move list resolves them when readable; otherwise assume rights conservatively from king and rook placement.
- Takebacks are normal in bot games (Tai, 2026-09-14). One ply and two plies are both recognised: a takeback while the bot has already replied removes the user's move and the reply together. Because two retreats can produce the same squares as a two-ply takeback, a frame that a story from the viewed position explains is compared with the story from the latest position, and the one with fewer new plies wins: a takeback followed by the bot's instant reply is a takeback and two moves, never four moves of retreats.
- Game end: checkmate and stalemate come from the board state. Resignation, timeout, and agreed draws leave no trace on the squares. Until the screen-state and OCR work in M2 reads the game-over dialog and the result in the move list, the companion panel gets a button for "white resigns", "black resigns", and "draw" (Tai, 2026-09-14). A new start position always ends the previous game.
- Browsing history (Tai, 2026-09-14): during a game the user can step back through the move list with the arrow keys or by clicking a move, and the board shows an earlier position; if the opponent moves meanwhile, the board snaps to the latest position. A frame that matches an earlier position of the current game puts the reconciler in a viewing state and nothing is undone. A takeback is recorded only when a different move is played from the viewed position, which is the only board-visible difference between the two. Chess.com's PGN omits takebacks, so the final move list is the ground truth for the committed line only.
- Missed moves: when the window was minimised, a dialog covered the board, or a badge hid a piece, a bounded search finds the shortest path of up to four legal moves that explains the frame, requiring every touched square to be read exactly; transpositions to one position are accepted, paths to different positions are held as ambiguous.
- The recogniser is rebuilt from every new start position, so a theme or piece set changed between games is harmless and orientation is decided per game.
- Gaps no move search can bridge (the window minimised for a minute, a restart mid-game, a recording that starts mid-game after a start position was seen) are closed by a position resync: when every cell reads exactly and settled, the board disagrees with the known position on several cells, and it holds still for ten frames, the observed position is adopted. Castling rights survive only for a king and rook still on their home squares, en passant is dropped, and whose turn it is, when the position does not decide it, is settled by the first move seen afterwards. The move list before the gap is kept; numbering restarts after it. `sidecoach replay` re-aligns its scoring to the PGN at the adopted position.
- Lessons from the first real game (2026-09-14): a dragged piece is drawn over neighbouring squares and the origin square is shown selected, and a click-move animates the piece across squares that may still carry the previous move's highlight. Highlights are therefore not used to commit moves at all: a move commits only when every piece it touched is centred where its own exemplar sits, no square carries the hover outline chess.com draws under a dragged piece, and the state holds for the settle frames. That is agnostic to any highlight colour, selection tint, red right-click mark, or check tint (Tai, 2026-09-14); a highlight-based rule was tried first and broke on each of those in turn. Arrows are drawn shapes and add foreground that the occupancy thresholds and stale-cell handling absorb. Premoves happen in bot games: the piece is drawn on its destination over the piece it will capture until the opponent moves, then several plies land within a second, which the iterative-deepening catch-up explains shortest story first.
- Overlays (2026-09-14): chess.com's game-over dialog, the promotion picker, and tooltips are opaque and cover several cells. A translucent tint keeps light and dark squares apart under it, so two 4-neighbours whose backgrounds match while both sit away from their square colours are under an opaque overlay, whatever its colour; a cell next to a covered one whose silhouette holds a rectangle of the overlay's colour anchored on a side or a corner is covered at its edge. Both read as unknown, which is no evidence, and never as a hover. The tracker keeps its lock through it: cells whose background is far from both square colours are left out of the checker fit and their insides out of the boundary re-fit, as long as 24 cells remain readable. At the end of a game the final position stays uncovered for about a second before the dialog fades in. That second alone is a thin margin, since a badge spill next to the king first has to turn stale, so a cell a confirmed overlay has just covered keeps its last stable reading (exact, settled, and identical in two consecutive frames, which rules out the fading dialog and a piece in flight) for three seconds: nothing changes under a dialog, and the explanation found in the uncovered frames keeps settling under it instead of turning ambiguous (c8=Q# and Qd7-c8 cover the same visible cells once c7 and d7 are hidden). The memory applies only to confirmed overlays, never to a hint dot or a lone badge, and a piece that leaves a covered square for a visible one waits for the memory to expire. The dialog fades in over two or three frames, and a half-transparent dialog reads as junk rather than as a cover, junk the memory would then keep; so a block of six or more neighbouring cells whose colours all fit one translucent tint (the light-dark gap shrinks by the same factor in every channel) is covered as well. Highlights and marks tint a few cells and never reach the block size.
- Badges (2026-09-14): the checkmate and winner badges land on the kings in the same frame as the mating move, spill into the neighbouring cell, and stay. The spill reads as an occupied cell that no move explains, and it turns stale exactly when the move's own cells do, so stale cells are ignored one at a time, junk-looking readings first, then in pairs, with the rest of the disagreement still explained by legal moves. Junk-looking cells are always tried, however many cells of the blocked move have turned stale beside them; clean stale cells join only when the stale set is small. The hover outline test needs the weakest side of the cell's border to differ from its background by half the tolerance and the median side by all of it, since a badge or a dialog edge crosses one or two sides and leaves the others near zero.
- Premove queues (2026-09-14, fourth game): while the opponent is to move, the user may queue several moves of one piece by dragging it on from where it is drawn. The site draws the piece at the end of the queue and leaves its real square empty, tints every queued square red, executes the first queued move the moment the opponent moves without changing the picture, and clears the rest and snaps the piece back when the opponent's move makes a queued move illegal (or for reasons of its own: a queued promotion was dropped once). A single premove already looked like a move for the wrong side and waited for the opponent; a queue looked like a position no moves reach. So a frame with exactly one of the user's pieces missing from its square and drawn on a square it cannot reach in one move, during the opponent's turn, is read as a queue: the queue is the unique shortest way for that piece to get there (a pawn drawn on the last rank as a pawn means the promotion is not chosen yet, and nothing is known), and each visible opponent move commits together with the next queued step. Nothing about the queue is stored: it is re-read from the piece's true square every frame, so a queue the user extends, shortens, or cancels needs no special case.
- A story seen once survives a drag (2026-09-14, same game): with premoves, two plies land in one frame and the next drag starts a fraction of a second later, before the story has settled; by the time the board is quiet again it is several plies further on, and a piece may have returned to a square it left, which the missed-move search cannot express from the committed position. So a pending story is kept through frames a drag blocks (up to four seconds) and, when the next quiet frame no longer fits it, it is extended by the moves that explain the frame from the position it leads to; the whole chain then settles. The intermediate frame also settles transpositions the search alone would have to guess.
- The opponent's move commits through a drag (2026-09-14): a drag blocks the user's own moves, since a dragged piece is not a move, but it cannot move the opponent's pieces. So the opponent's move is read from the cells it touches alone, from the position the pending story leads to, as long as those cells are settled and not under the cursor; a pending story that ends with the opponent's move is confirmed the same way. Without this a bot that replies while the user is already dragging the next premove is seen only as a net displacement once the board is quiet, and a premove queue advances one step for every two or three the site executed.
- Order of explanations (2026-09-14): shortest story first. One move; the pending story confirmed or extended; a premove queue; the opponent's move on its own; and only then a path of missed moves, which would otherwise explain a queue display (a pawn drawn two squares on) with a longer story than what happened. In the stale-cell retry, the ignored cells are wildcards for the path search: a move may touch them instead of an unexplained cell and needs no exact reading there, so the mating move settles while the checkmate badge has made the king's square unreadable.
- Next to a dialog-sized overlay, a cell that does not read cleanly is the overlay's edge cutting through it (a strip of dialog over a pawn read as a queen, unsettled, with a hover outline that kept every commit blocked under the dialog). Nothing is dragged while a dialog is up, so such cells read as covered.
- Second review (2026-09-14): the lifecycle is now explicit. A finished game takes no more moves: browsing after the game is followed, moves played on the review board are analysis and ignored, the start position begins the next game, and a position far enough away to be adopted by a resync begins another game under a new id (a game seen from the middle after the last one ended). A resync that adopts a mated position ends the game; a start position after a resync ends the game it resynced. When a resync left the side to move open, the first story is searched for both sides and settles like any other, fixing the turn only when it commits, so a premove dragged during the bot's turn cannot flip it. Browsing is entered only from a quiet frame, since a piece dragged back over its own square shows an earlier position for a moment. The memory of covered cells forgets a cell the moment it is seen clearly with something else on it, and forgets everything once the board has gone unseen for several ticks (lost, covered, minimised: the pipeline reports each such tick), since a board that went unseen may have changed; a capture stall with the board in view is not a gap in evidence. A reading of something unnameable on a square the position has empty, once stale, is an arrow and counts as unknown, so a few arrows neither stall the game nor multiply the stale retries. A premove queue's route is read from the red squares when the site tints them (a3 then a4 is two steps, not the shortest a4), which keeps the ply count and the side to move right. A frame with fewer than half its cells readable is no evidence at all: a theme the recogniser was not built for reads as unknown almost everywhere without contradicting anything, and two tinted cells alone must not commit a move. Think time is measured from the frame timestamps between commits. In the pipeline, the bootstrap decision is taken on every verified tick, changed or not; a start position the recogniser cannot read as its own before the first move (the board flipped when the user was paired as black, or re-themed) rebuilds the recogniser and re-orients the same game rather than ending it; readings are keyed to the geometry they were taken with, so a re-lock re-reads the frame even when the change gate says nothing changed; and "reads as the start" counts contradictions (a piece read as another, an empty square where a piece belongs) rather than any stray cell, with most of the board readable.
- Piece colour comes from the nearest exemplar's brightness profile rather than a global threshold: some sets draw a white queen whose interior is half black detail, which a threshold reads as black, and that one persistent misread stalled a whole game. Legal-move hint dots and capture rings appear during drags; dots read as unknown by occupancy, and a piece whose type cannot be matched asserts only that the square is occupied, never a colour, because capture rings flip the measured colour and a wrong colour lets the move search invent captures.

### 5.4 Analysis

- Stockfish over UCI through python-chess, persistent process, MultiPV 4, WDL enabled.
- Live: about 300 ms per new position, then continuous analysis while the user thinks, so the coach's scaffolds are ready before the user asks. Post-game: 1 to 2 s per position.
- Everything is in expected score from the mover's view, where 0.5 is a dead draw and 1.0 is a win. Eval loss of a played move is the expected score of the best line minus what the move actually left, both measured at the same depth, which is why every position of a game is searched the same way and cached. Centipawns are never used for thresholds, because a 50 centipawn loss at +8 is nothing.
- Which curve turns a score into an expected score matters more than it looks (2026-09-15). Stockfish's own win-draw-loss numbers answer "how often does Stockfish beat Stockfish from here", and that is 1.000 from a knight up: measured against the game of 2026-09-14 20:06 they produced 7 blunders and not one inaccuracy or mistake in 88 plies, because inside a won position every move scores the same and only the moves that flipped the result registered at all. A club player's won position is not won, so the default model is Lichess's curve from centipawns, fitted to games between people: a knight up is 0.88 there, and the same game scores 7 inaccuracies, 2 mistakes and 5 blunders. The engine's numbers stay available as `engine.score_model = "engine"` and are carried on every line either way. A mate is 1.0 or 0.0 under both.
- The engine's own first choice costs nothing by definition; without that rule a search that returns a slightly different number for the same line at the same depth has the coach calling the best move on the board an inaccuracy.
- Severity thresholds in expected-score loss: inaccuracy 0.04, mistake 0.11, blunder 0.28. Fitted (2026-09-15) to chess.com's review tallies over six of Tai's games, 671 plies: they put our inaccuracy, mistake, miss and blunder counts at 55, 30, 25 and 10 against chess.com's 60, 38, 22 and 10. The remaining shortfall in mistakes is the other engine finding more than ours at depth 14, not a threshold.
- Move labels in chess.com's vocabulary (2026-09-15), because that is what the user reads after every game and a coach whose verdicts disagree with it loses trust. Two axes: the loss says how good a move was (best is the engine's first choice, excellent under 0.02, good under the inaccuracy line, then the three severities), and the moment says whether it mattered. A miss lets go of a chance the opponent's mistake handed over; a great move takes it with the best move, or is the only move; a brilliant move is a sound sacrifice. Neither great nor brilliant applies to an obvious move (a recapture, the only legal move) or in a decided position, where keeping a won game is just the move. Measured over the six games: great 23 against chess.com's 19, miss 25 against 22, brilliant 0 against 0, best 199 against 217. Two lessons from the first cut: forced recaptures passed the only-move test and tripled the greats, and a reply that captured and was taken straight back read as a sacrifice and made quiet developing moves brilliant.
- Accuracy (2026-09-15): the per-move number is Lichess's exponential of the drop in winning chances, but three times steeper, and a side's accuracy is the plain mean. Lichess's own steepness ran fifteen points above chess.com on every game; steepened, the twelve game-sides land within 4.4 points of chess.com with no bias, which says chess.com's accuracy is, in effect, how often you played a near-best move. Chess.com's formula is not published, so near is the most that can be claimed.
- Chess.com's review is the calibration reference, not ground truth: it runs its own engine at its own depth with its own thresholds. Its move list and overview panels, copied by hand out of the browser's developer tools and saved to a file, parse into a per-ply reference (`analysis/chesscom.py`): labels, times per move in tenths of a second, the per-side tallies, accuracy and game rating. `sidecoach import-reviews` files each game in such a page beside the recording whose truth PGN it replays against, and `sidecoach review --compare` prints our labels beside theirs and the tallies side by side. The times per move are ground truth for the think times the clock OCR will read later.
- Decisive moment: gap between the best and second-best line of at least 0.10, or an only-move, which is either the single legal move or a best move that beats the second by 0.30. Later, Maia predicting a losing move for the user's rating band is a third trigger.
- Opponent threats: analyse the position with a null move to find what the opponent would do if it were their turn.
- Motif tagger: pure functions over (board, move, principal variation) returning tags with evidence squares, using Lichess puzzle theme names. Groups, in implementation order: tactical basics, mating patterns and king safety, openings, positional themes, endgame themes. Tactical and mating tags are computable from attack maps and the PV. Positional and endgame tags are heuristics and will be tuned against the user's games.
- Openings: the Lichess chess-openings tables (3810 lines, CC0), fetched by `scripts/fetch_openings.py` into `data/openings` and compiled once into a cache that loads in about three milliseconds. Indexed by position rather than by move order, so transpositions resolve: the Queen's Gambit Declined is named whichever way the moves were played. The same index answers book membership and the first deviation, which is what the design reserved a Polyglot book for, so there is no `.bin` file. Absent tables simply mean no opening name.
- No single position may hold up a game (2026-09-15). Some positions are pathologically slow to reach a given depth: seventeen knight-and-pawn endgames in the game of 2026-09-14 16:13 took Stockfish minutes each to separate four lines at depth 14, and reviewing that game took over half an hour instead of two minutes. Every search therefore carries a time cap as well as a depth, five seconds by default, and each assessment records the depth actually reached, since a capped search makes the comparison between the position before a move and the position after it less exact.
- The engine never runs on the capture thread. `analysis.annotator.Analyser` owns a worker thread, takes moves as they are committed, and hands back assessments a moment later, which is when a coach would speak anyway; a failed analysis is counted, not raised, so a dead engine cannot end a session. The engine wrapper restarts a crashed process once per call for the same reason.

### 5.5 Coach

Disclosure state machine, one instance per position where the user is to move:

| Level | Coach may say | Reached by |
|---|---|---|
| 0 silent | nothing | new position |
| 1 question | an orienting question | decisive-moment trigger, stuck timer, or a hint request |
| 2 theme | the theme category, for example "a loose piece is involved" | next hint request or stuck timer |
| 3 region | a board region, for example "look at the kingside" or "something around f7" | next hint request or stuck timer. Autonomous ceiling. |
| 4 piece | the piece that carries the idea, never the destination | explicit hint request beyond region |
| 5 candidate | the move, or the near-equal set | user names it, plays it, or asks for the move |
| 6 line | the full principal variation | user asks for the line, or after the move |

Hiding is by construction, not by instruction. The fact bundle handed to the LLM is filtered by the current level: at level 2 the bundle contains the theme name and nothing about squares; at level 3 it contains the region; the move and the PV are absent from the prompt until level 5 and 6. After the move is played, feedback about the previous position is unrestricted.

Interjection policy. Unprompted triggers, decided 2026-09-14:

1. Answer explicit user input, always.
2. Post-mistake lesson after a move above the mistake threshold, framed as a counterfactual pattern.
3. Brief feedback after every user move, one or two sentences, length scaled by eval loss.
4. Pre-move orienting question at decisive moments.
5. Stuck hint after a configurable time on the current move, default 45 s.

Higher numbers yield to lower ones when they collide. At most one unprompted message per position. A quiet toggle silences everything except explicit asks.

Clock scaling: rapid and slower get all triggers with the stuck threshold scaled to the remaining time. Blitz gets post-move feedback only. Bullet is silent during the game and everything goes to the post-game review. Untimed bot games and analysis get all triggers.

Claim verification, both directions:

- The user's stated reasons are extracted into structured claims: attacks, defends, threatens, hanging, safe, best, plan. Board claims are checked with python-chess, evaluation claims against the engine, threat claims against the null-move analysis. Vague reasons are non-checkable and prompt a follow-up question.
- The coach's own draft is checked before it is shown. Any square, piece, or move it mentions is verified against the board, and any withheld content beyond the current level is a leak. A failed check regenerates once, then falls back to a template.

LLM: one OpenAI-compatible provider with two configured backends. `local` is LM Studio at `http://localhost:1234/v1`; its Qwen 3.8 builds reason before every answer and ignore every documented switch, so reasoning is suppressed with an empty think-block prefill as a trailing assistant message (measured: zero reasoning tokens, same answer quality). `dashscope` is Alibaba Cloud Model Studio in the US region with qwen3.8-max by default and qwen3.8-flash as the cheaper option; it honours `enable_thinking` in the request. Measured 2026-09-14 with thinking off: local 1 to 3 s, max 1.8 s, flash 1.1 s; with thinking on, 5 to 10 s typically, 45 s once on qwen3.8-max for an explanation, and an occasional runaway on the local model that exhausts the budget and returns nothing. Conversational turns run with thinking off; claim extraction and post-game review may turn it on. Hosted Claude remains a possible third backend. Persona: warm but direct, like a good club coach.

### 5.6 Profile

Error taxonomy has two axes.

- Cause, from engine evidence plus self-reports: missed opponent threat, miscalculated line, never considered the candidate, positional misjudgment, opening knowledge gap, endgame technique, time pressure. A mistake with under ten seconds on the clock defaults to time pressure.
- Motif, from the tagger.

Each cell is a Beta-distributed error rate with counts, so a single event never creates a weakness. Insights such as biggest cost category and knowledge gap versus habit are computed from the counts. Self-reports are elicited sparingly, at most once per few mistakes, and every profile entry is user-correctable. Session summaries give the coach continuity across sessions.

Initial prior: the user plays rapid and blitz in the 1000 to 1400 band.

### 5.7 Recommendations

- Lichess puzzle database, CC0, downloaded once and filtered offline by rating band and theme into SQLite. Served in-app for spaced repetition per motif. Outcome tracked.
- Chess.com lessons: a one-time rate-limited scrape of the paginated course list and course pages into a catalogue of title, URL, level, category, and description, then mapped to motifs and phases with a review pass. The coach links to lessons and tracks follow-through.

### 5.8 UI

Terminal first: the dialogue prints in a console and the user types replies. Then an always-on-top companion panel with the dialogue, a text input, the current disclosure level, and a mirrored board. Then a click-through transparent overlay over the detected board for region highlights and arrows.

### 5.9 Voice

Later. Local only: faster-whisper for STT with post-correction against the legal move list, Silero VAD for barge-in, Kokoro for TTS. SAN is converted to spoken form before synthesis.

### 5.10 Replay and recording

Every stage is replayable. Recordings hold frame sequences with timestamps and the events each stage emitted. The test suite is recordings plus expected outputs, so a vision or reconciliation regression fails a test instead of a live game.

Recordings are the regression suite (2026-09-14): `data/recordings/gNN-<opponent>-<note>/` holds one game each, with `truth.pgn` from chess.com beside it. Each is stored at half scale (1280x720 for a 1440p session) with `sidecoach shrink`, lossless FFV1, the whole screen kept: the program's input is the screen, a recording of less would test an easier problem, and the clocks and the move list are inputs for later stages. The half-scale copies double as the size-invariance test, and `sidecoach replay --scale` resizes further on the fly (Tai, 2026-09-14).

## 6. Storage

SQLite, single file, plus PGN files. Tables: games, positions, moves with eval loss and tags and cause, dialogue turns, claims and verdicts, profile cells (motif, cause, phase, alpha, beta), puzzles, lessons, recommendations with outcomes, sessions with summaries. Board crops and recordings live on disk under the data directory, referenced by path.

## 7. Latency budget

| Step | Target |
|---|---|
| Frame to settled move | 200 to 300 ms at 10 fps with two-frame settle |
| Engine, new position | about 300 ms live, continuous afterwards |
| Text turn, local LLM | under 1.5 s to first token; measured 0.43 s on 2026-09-14 |
| Voice turn end to end | under 2 s with streamed TTS, later |

## 8. Evaluation

Pipeline metrics first: per-square classification accuracy, move-detection precision and recall on recordings, false interjection rate, disclosure leak rate measured by the output checker, latency percentiles per stage. Skill metrics after: per-motif error rates over time, targeted-puzzle accuracy, eval loss at decisive moments, lesson follow-through.

## 9. Stack

| Concern | Choice |
|---|---|
| Language and tooling | Python 3.12, uv for everything, src layout, single package `sidecoach` |
| Capture | mss first; dxcam or windows-capture if needed |
| Vision | OpenCV, numpy; small CNN in PyTorch exported to ONNX Runtime; RapidOCR |
| Chess | python-chess, Stockfish release binary, Lichess chess-openings data, Polyglot book |
| LLM | OpenAI-compatible provider over httpx: LM Studio locally, DashScope qwen3.8-max or qwen3.8-flash hosted; keys in `.env` |
| Voice, later | faster-whisper, Silero VAD, Kokoro |
| Storage | SQLite, PGN files |
| CLI and terminal UI | typer, rich |
| Config and events | pydantic, TOML |
| Tests | pytest on recordings; ruff; pyright |

Machine: RTX 3090 Ti 24 GB, Ryzen 9 9900X, 62 GB RAM. LM Studio serves the 27B model from a remote device (h200-1), so the local GPU is free for vision training, STT, TTS, and Maia.

## 10. Roadmap

M1 See the board

- [x] `sidecoach doctor` checks Python, video codec, Stockfish, the LLM, and capture.
- [x] Record and replay harness: `sidecoach record` to lossless video with change gating, `sidecoach replay` into the pipeline, `sidecoach cells` to inspect readings.
- [x] Board detection on real chess.com frames (green and brown themes so far), tracker for moves, resizes, occlusion; orientation from the start position.
- [x] Piece recognition without training: silhouette exemplars bootstrapped from each game's start position. The learned cross-theme classifier remains a later upgrade, needed for cold starts mid-game.
- [x] Move detection over legal moves with settle, highlight gating, history browsing, takebacks, missed-move catch-up, and new-game handling.
- [x] From a 10 fps recording of a bot game, reconstruct the exact PGN: game one of 2026-09-14, 79 of 79 plies including promotion, checks, and checkmate.
- [x] `sidecoach watch` follows the game live in the terminal, printing every move with the board, and records the session so anything odd can be replayed with `sidecoach replay` and inspected with `sidecoach cells`.
- [x] A second game on the brown theme and a different piece set, from two recordings merged with `sidecoach concat` and a 91-second gap between them: 141 of 141 on-screen plies correct on the final line, 24 plies skipped across the gap and recovered by a position resync, a real two-ply takeback recognised as such, the order of Rd2 and Rf1+ (the same position within a second) settled by the intermediate frame, and the mating move committed under the game-over dialog from remembered cells.
- [x] A live session followed end to end: 109 of 109 plies of the 2026-09-14 16:49 game including two takebacks and the mating promotion under the checkmate badges, with the game-over dialog tolerated afterwards.
- [x] First meticulous review pass (2026-09-14): three reviewers plus a line-by-line read of the reconciler; 23 defects fixed with regression tests (bootstrap on browsing to the start, bootstrap starvation, whole-cell slides, a poisoned start frame, tinted blocks, fade junk, the scorer's game boundary, Windows paths in concat, the codec probe, the writer's return value, and more); a live watcher with keys (resign, draw, new game, quit), a decision log beside each recording, a health line, and one change gate for live and replay. 122 tests. Recordings kept at 720p as `gNN-<opponent>-<note>`, scoring identically to the full-resolution originals.
- [x] A fourth game with queued premoves (a5 to a8=Q while the bot's king walked) and the mating move under the checkmate badge: 135 of 135 plies of the 2026-09-14 20:06 game, with the queue read as a displaced piece and each step committed with the bot's reply.
- [x] Second meticulous review pass (2026-09-14/15): three reviewers over the vision stack, the recogniser, and the reconciler with the CLI, then fixes with regression tests for every confirmed finding: the reconciler's lifecycle (finished games, resyncs, new games), the side to move after a resync, takebacks with an instant reply, browsing from quiet frames only, memory that expires when the board goes unseen, arrows as unknown, queue routes from the red squares, no evidence from a mostly unreadable frame; the pipeline's bootstrap on every verified tick with a flip or theme change before the first move re-orienting the same game; the detector's polarity, local-maxima candidates and comb-strength re-fit acceptance; the recogniser's pair fusion by agreement, geometric tint rule, shift-invariant matching and settle floor; capture retries and a clean failure; the recording module's concat, truncated index and gate order. 270 tests. All six recordings score as before (79/79, 141 + 24 skipped, 54 + 2 taken back, 109/109, 135/135, 88/88) and g04 at 0.75 scale (32-px cells) rises from 94 correct with 15 skipped to 109/109.

M2 Analyze

- [x] Stockfish 19 over UCI behind a wrapper that keeps one process alive, asks for real win-draw-loss numbers, and restarts a crashed engine, since a coaching session must outlive a dead subprocess.
- [x] Expected-score loss at matching depth, severity, decisive moments and only-moves, opponent threats from a null move. The scoring curve is human-calibrated, not the engine's own win rate; see section 5.4 for what that measurement showed.
- [x] Motif tagger: 36 Lichess theme names over (board, move, principal variation), pure functions with evidence squares. Every material claim must gain something once defenders are counted, and a move is only tagged with a feature it brought about. Across the six recorded games, 671 plies, the tactical tags fire at rates a coach would recognise: fork 0.6%, pin 1.6%, skewer 0.1%, sacrifice 2.8%.
- [x] Opening identification from the Lichess tables, indexed by position so transpositions resolve, with book membership and the first deviation from the same index.
- [x] `sidecoach review` annotates a PGN, `sidecoach watch --analysis` assesses each move once it is played, and `sidecoach replay --assess` runs the whole chain from pixels to assessments on a recording. All six recorded games review without incident.

Still open before the coach can use the screen as a second source of truth: OCR of the clocks and the move list, and the screen-state detector. Both are vision work that section 5.2 describes and neither is needed for M3's dialogue, but the clock decides the interjection cadence and the move list would check the reconciler's ply parity, so they are worth placing before M4.

M3 Coach: disclosure state machine, interjection policy with clock scaling, fact-bundle filtering, local LLM provider, claim verification both ways, terminal dialogue. First end-to-end coached bot game.

M4 Review and profile: post-game review, SQLite, error taxonomy, session summaries, self-report elicitation.

M5 Companion panel. M6 Recommendations from Lichess puzzles and the chess.com catalogue. M7 Voice. M8 Overlay. M9 Maia and remaining motif groups tuned. Later: macOS and Linux capture, Android, iOS, EULA, public release.

## 11. Open items

- Stockfish 19 ships one universal Windows build that picks the CPU code path at runtime, so there is no avx2 versus vnni512 choice to make. Verify with a benchmark that it uses AVX-512 on the Ryzen 9 9900X.
- Stuck-threshold scaling formula for rapid. Start with the smaller of 45 s and ten percent of remaining time.
- Whether generic "hint" requests may advance past level 3 to level 4, or only a request that names the piece. Current answer: a hint request advances one rung, including to level 4; levels 5 and 6 need a request for the move or the line.
- Lichess puzzle rating window around the user's band. Start with 800 to 1600.
- Reasoning control on the local LLM, resolved 2026-09-14. The Qwen `/no_think` switch is ignored and LM Studio's native `reasoning` field returns 400 for these models, but an empty think-block prefill as a trailing assistant message on the OpenAI-compatible endpoint suppresses reasoning completely. If a future model or server version stops honouring assistant continuation, revisit.
