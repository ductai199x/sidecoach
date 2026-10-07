# Move classification

How every move gets its label -- brilliant, great, best, excellent, good, book, inaccuracy,
mistake, miss, blunder -- in the live coach, in Game Review and in Stats. One function decides
it: `label_move` in `src/sidecoach/analysis/score.py`, called through `verdict` by the live
annotator (`analysis/annotator.py`) and by a saved review made again
(`analysis/game_review.py` `relabel`), so the two cannot drift. The phone has the same code in
Kotlin (see [The phone](#the-phone)).

## Where the rules come from

Chess.com's labels, because that is the vocabulary the player reads after every game, and a
coach whose verdicts disagree with it loses trust ([DESIGN.md](../DESIGN.md) section 5.4, 2026-09-15).

Chess.com's classifier runs on its servers: its analysis page only draws the labels sent to it
(over `wss://analysis.chess.com`), so there is no client code to read. Lichess's
`Advice.scala` is open but names only inaccuracy, mistake and blunder (drops in winning
chances of 0.1, 0.2 and 0.3), with no brilliant, great, miss or book. So the source is
chess.com's published article, support.chess.com 8572705, "Classification V2" (Tai,
2026-10-06: "use their definition"):

| Label | Chess.com's definition |
|---|---|
| Best | 0.00 expected points lost |
| Excellent | 0.00 to 0.02 |
| Good | 0.02 to 0.05 |
| Inaccuracy | 0.05 to 0.10 |
| Mistake | 0.10 to 0.20 |
| Blunder | 0.20 to 1.00 |
| Brilliant | "always the best or nearly best move"; "a good piece sacrifice"; "you should not be in a bad position after"; "you should not be completely winning even if you hadn't found the move" |
| Great | "critical to the outcome of the game": a losing position turned equal, an equal one winning, or the only good move found |
| Miss | failing to capitalise on the opponent's mistake; missing the chance of a winning position |
| Book | a conventional opening move |

What we do not model: the article says chess.com is "more generous" with Great and with what
counts as a piece sacrifice for newer players. How generous is not published, so it is not
guessed at; our labels are the same at every rating.

Before 2026-10-06 the severity lines were fitted to six of Tai's reviewed games, at 0.04
(inaccuracy) and 0.28 (blunder). `config.RETIRED_DEFAULTS` moves a settings file that still
holds those old defaults to the new ones (0.05, 0.20) when it is read; a value the player chose
stays. The phone's `SettingsStore` does the same.

## Units: expected score, never centipawns

Every number here is expected score from the mover's side: 0.5 is a dead draw, 1.0 a win, on
Lichess's curve fitted to human games (`human_expected_score`, `HUMAN_K`; a knight up is about
0.88). A move's loss (`eval_loss`) is the best line's expected score minus what the move left,
both from the same search at the same depth; the engine's own first choice costs 0 by
definition. Centipawns never set a threshold: 50 centipawns at +8 is nothing, at level it can
be the game. The engine's own win-draw-loss model is `engine.score_model = "engine"`, off by
default (DESIGN.md section 5.4).

## The thresholds

`score.Thresholds`, filled from `CoachSettings` (`Thresholds.from_settings`):

| Field | Default | Meaning |
|---|---|---|
| `excellent` | 0.02 | loss below this is Excellent (`about_as_good`) |
| `inaccuracy` | 0.05 | loss from this is an inaccuracy (`graded_good` is below it) |
| `mistake` | 0.10 | loss from this is a mistake (`clearly_worse`) |
| `blunder` | 0.20 | loss from this is a blunder |
| `decided` | 0.90 | an expected score beyond this (or below 0.10) is a decided game |
| `decisive_gap` | 0.10 | best line this far ahead of the second: a decisive moment |
| `only_move_gap` | 0.30 | best line this far ahead of the second: an only move |

The bands are half-open: a loss of exactly 0.05 is an inaccuracy, exactly 0.10 a mistake,
exactly 0.20 a blunder (`severity` compares with `>=`), and Excellent is a loss strictly below
0.02.

## The rule, in the order the code applies it

`label_move(loss, thresholds, is_best, in_book, decisive, sacrifice, before_score,
opponent_blundered, second_score, chance, mate_let_go, mate_allowed, obvious)`. The flags are
read by `verdict` from the position before, the move and the engine's lines:

- `lost`: the mover's expected score before the move is at most 0.10 (`1 - decided`).
- `mate_allowed`: the line after the move has a forced mate for the opponent, and the mover
  did not already face one; cleared when the mover was already lost or played the best move
  (chess.com leaves those unmarked).
- `flagged`: the loss is 0.05 or more, or `mate_allowed`.

Then, first match wins:

1. **Book**: the position after the move is in the opening tables (`in_book`) and the loss is
   under 0.10. The tables hold dubious gambits too (1. e4 f5); chess.com's book calls 1...f5 a
   mistake, so a book move that throws away a tenth of a game is labelled by its loss.
2. **Miss, a mate let go** (`mate_let_go`, not the best move): the mover had a forced mate and
   the move played a mate in 1 not given; or left no forced mate; or made the mate
   `MATE_LONGER` (4) or more of the mover's moves longer while the opponent still has a piece
   besides king and pawns (against a bare king or king and pawns the technique mate is the
   lesson, so a longer route is fine). A miss whatever it cost, even while it still wins (Tai,
   2026-10-02 12:50). Any other longer mate is labelled by its loss.
3. **Miss, a winning chance let go** (`chance`, flagged, not the best move): the best line stood
   better or winning (`standing` 3 or more, from 0.62) and the move leaves the mover in a lower
   band (`winning_chance`). Whatever the size of the loss, from an inaccuracy up (Tai,
   2026-10-02); the engine's evaluation decides, not where the chance came from. A position
   with a forced mate is read by its mate instead (rule 2).
4. **Blunder, into a mate** (`mate_allowed`).
5. Not flagged (loss under 0.05):
   1. **Brilliant**: `sacrifice` (the motif below), and best or excellent (loss under 0.02),
      and the mover is not lost after it (expected score after the move above 0.10), and the
      second-best line (`second_score`, or the score before when there is no second line) is
      short of decided (below 0.90): without the move the mover would not be completely
      winning. Nothing else: no exclusion of recaptures or of the only legal move.
   2. **Great**: the best move, not `obvious`, not `lost`, and either `decisive` (the position's
      best line is 0.10 ahead of the second, or it is an only move) or the opponent's previous
      move was a blunder (its loss 0.20 or more). `obvious` is a recapture on the square the
      opponent just captured on, a capture nobody can take back (`exchange.can_retake`, by legal
      moves), or the only legal move. A great move in a won position stays great.
   3. **Best**: the engine's first move.
   4. **Excellent**: loss under 0.02; else **Good**.
6. Flagged: **Blunder** from 0.20, **Mistake** from 0.10, else **Inaccuracy**.

Two consequences worth knowing. A Miss is decided before the severity, so a move that lets a
winning chance go is a Miss even when its loss is a blunder's (Tai's rule of 2026-10-02). And
only Great carries the obvious-move and lost-position exclusions; Brilliant has only the
chess.com conditions.

## The sacrifice motif

`sacrifice` is true when the move carries the tagger's `sacrifice` tag
(`analysis/motifs.py` `_sacrifice`), read on the move and the engine's line after it. Material
is counted 1/3/3/5/9 (`_balance`). First a gate: unless the line mates for the mover, the line
must end with the mover's material no lower than before the move (a piece given for nothing is
a blunder, and the loss already says so). Then either form counts:

- **Taken in the line** (`_given_in_line`): after the opponent's reply the mover is at least
  two points down (`_SACRIFICE_THRESHOLD`), still behind after the mover's next move (a capture
  taken straight back is a trade), the piece the reply took is a minor piece or more (two pawns
  are not "a piece sacrifice"), and the reply's capture wins at least two points once the
  recaptures on its square are played out (`exchange.exchange`), however the line orders them.
  A knight given for two pawns counts (Nxg5 hxg5 Bxg5).
- **Offered, taken or not** (`_offered`): after the move some piece of the mover's (not a pawn,
  not the king, a minor or more) can be won by the opponent for at least a point more than the
  move itself took (`exchange.best_capture`, recaptures played out), and for more than the same
  piece could have been won before the move. Each piece is measured against itself, so a move
  that answers one threat while offering another piece counts. This is chess.com's "a piece
  left to be taken": the engine's line may decline it because taking loses (Tai's puzzle,
  2026-10-06: after Nxg5 the line declines the knight, hxg5 being a mistake).

A pawn gambit is never a sacrifice, and a capture taken back (Bxc3 bxc3) is a trade.

## Related calls

- `is_decisive`: an only move, or the best line 0.10 ahead of the second (`decisive_gap`).
- `is_only_move`: the only legal move, or the best line 0.30 ahead of the second.
- `move_accuracy`: Lichess's exponential of the loss, three times steeper (`ACCURACY_K` 0.25),
  so a side's plain mean lands near chess.com's accuracy (DESIGN.md section 5.4).
- `standing`: the five bands the coach speaks in (losing, worse, about equal, better,
  winning: 0.2, 0.38, 0.62, 0.8).

## Known gaps

- **Saved reviews keep old labels.** A saved review is relabelled only when
  `ui/review_room.py` `DERIVED` is raised (`rederive`). It is still 5 (the 2026-10-02 miss
  rule), so reviews saved before 0.1.84 keep the labels of the old thresholds and the old
  brilliant rule. `relabel` also reads the `sacrifice` tag kept with each saved move, so the
  2026-10-06 sacrifice motif reaches a saved review only when its moves are tagged again.
- **Fitted models read the new bands.** `profile/performance.py` `EDGES` takes its buckets
  from `DEFAULT_THRESHOLDS` (now 0.05 / 0.10 / 0.20), but `performance.json`,
  `categories.json` and `peers.json` were fitted on 2026-09-30 with the old lines (0.04 / 0.10 /
  0.28). See [stats.md](stats.md).

## The phone

`app/src/main/java/com/sidecoach/analysis/Score.kt` (`labelMove`, `Thresholds`) and
`Motifs.kt` (`givenInLine`, `offered`, `SACRIFICE_THRESHOLD`) in sidecoach-android are 1:1
ports. They are held to the desktop by recorded fixtures, made by running the desktop code:
`tools/make_score_fixtures.py` (read by `ScoreTest.kt`), `tools/make_motif_fixtures.py`
(`MotifsTest.kt`), `tools/make_annotator_fixtures.py` (`AnnotatorParityTest.kt`) and
`tools/make_review_fixtures.py` (`GameReviewParityTest.kt`), all in the Android repo. A change
to any rule here regenerates them and is ported in the same change.

See also: [game-review.md](game-review.md), [stats.md](stats.md), [coach.md](coach.md).
