# Game Review

A finished game reviewed whole, modelled on chess.com's Game Review -- its flow, depth and
feel, Diamond tier included -- in Sidecoach's own look (Tai, 2026-09-29: "I want the feel and
the UI UX to be modelled after chess.com ... but with our own visuals and design language").
How each move is labelled is in [move-classification.md](move-classification.md); the coach the
review speaks with is in [coach.md](coach.md); what reviewed games add up to is in
[stats.md](stats.md). The architecture is [DESIGN.md](../DESIGN.md)'s.

## Decisions

1. **A performance rating per game** that does not read the players' ratings (Tai, 2026-09-29;
   chess.com's Game Rating is criticised for echoing them). See [Calibration](#calibration-phase-0).
2. **Arrows off by default** (Tai, 2026-09-30: "if the gold best move arrows are on then there's
   no point having the retry feature"): the best move in gold and the threat and idea arrows are
   each a switch in the review's settings. With the gold arrow off, the review's words do not
   name the better move either; Best shows it.
3. **The model's words only when asked, on the key moves** (Tai, 2026-09-30: "the user themselves
   need to consciously press a button to generate AI review for any game ... it costs around 1
   cent per game and only on key moves"). Every move has the review's own words; a button asks
   the model for the key moves of that one game, never ahead of time and never for games the
   background review goes through. One call a key move, all at once (Tai: "why not just analyze
   the moves individually?"): told every move together, the model lent one move's facts to
   another and one unreadable answer lost the game. Measured on four of Tai's rapid games
   (2026-09-30, 8-16 key moves, luna at medium): 0.85-1.8¢ a game, about 0.1¢ a key move.
4. **A window of its own**, as the puzzles have (Tai, 2026-09-29); a full-screen page on the
   phone.
5. **No per-game chess.com link import** (its only instant source is an unofficial endpoint).
   Games come from what Sidecoach watched, chess.com and Lichess by username through their
   official public APIs, or a pasted or opened PGN (`profile/library.py`).
6. **No coach characters**: the review speaks as the coach does in play, with the persona and
   voice chosen in Settings.
7. **The phone reviews on demand** (Tai, 2026-09-30: no background review, as chess.com has
   none), and a review survives the app being killed (below).

## What we do better than chess.com

From what its users complain about most:

| Their complaint | Ours |
|---|---|
| Brilliant and Great feel random; the text is generic or wrong | Chess.com's published rules, the same at every rating; every AI paragraph is checked against the board |
| Game Rating echoes the players' ratings | Fitted on real players' moves; ratings not an input |
| Retry and Hint were removed (May 2026) | Retry kept, with the coach's hint ladder (two hints, then the move) |
| The best-move arrow gives the answer away | Off by default, and the words keep the move back while it is off |
| The coach and voice intrude | The voice has its own switch |

## The screens

The letters are the ones the code's comments use.

- **A. After a game**: the end-of-game card has a Game Review button straight into the review of
  the game just played (`desktop/src/lib/Coach.svelte`).
- **B. Summary**: the coach's line on the game, the eval graph (expected score, the key moves
  dotted in their class colour, click to jump), each side's accuracy, the counts of every kind
  of move (folded to brilliant, great, best, mistake, miss, blunder), each side's game rating,
  a grade for each phase, and the game's shape (`profile/game_summary.py` `SHAPES`: smooth,
  balanced, sudden, giveaway, wild, intense, sharp).
- **C. Step-through**: the move's squares tinted in its class colour and its badge on the
  destination; the arrows the settings turn on (gold the best move, red the reply that punishes
  a bad move, blue what a sound move threatens); the eval bar (beside the board on the desktop,
  across the top on the phone); the coach's card with the move's words. Actions: Retry, Best,
  Follow-up, Next. Next goes to the next key move -- a brilliant, great, inaccuracy, mistake,
  miss or blunder (`KEY_LABELS`), or the last book move -- playing the moves between, a third
  slower than at first (Tai, 2026-09-30). The voice reads the summary's line and each card.
- **D. Retry** (`ui/review_retry.py`, `lib/review/retry.svelte.ts`): the player plays another move
  in the position a move of the game was played from, graded by the same annotator and
  thresholds, the game's own measurements adopted; the coach's hints as in play; a wrong move
  shows red and goes back; giving up plays the answer. The summary shows the accuracy after
  retries.
- **E. The last card**: practice puzzles for the motifs the player's mistakes passed up
  (`ReviewRoom._practice`, two motifs, three puzzles each).
- **F. Analysis** (`ui/review_analysis.py`, `lib/review/AnalysisPane.svelte`): the player's own
  moves from any position, kept as a tree; the engine's best lines for the board's position; the
  explorer (the moves the player's own games went on with from it); the game's details and its
  PGN, with the review's verdicts and the side lines, to copy or save; charts of the score, the
  time each move took, and the themes.
- **G. Library**: every watched or imported game (`routes/games`), with filters, accuracy and
  Review; the insights across games are the Stats window ([stats.md](stats.md)).
- **H. Review settings**, kept for the next review: the best-move arrow, the threat and idea
  arrows, the classification on the board, whose key moves Next stops at, time spent per move,
  and how deep the engine looks.

Not built from the 2026-09-29 plan: Share, chess.com's Best Move Diff chart, the masters'
continuations in the explorer, the opening panel ("you have played this N times"), an autoplay
delay setting.

## How a review is made

- **Analysis** (`analysis/game_review.py`): every position of the game measured by several
  engines side by side (one thread each, up to `MAX_WORKERS`), whole -- with the threats and
  traps the live coach's facts are made of -- then one annotator walks the moves in order (a
  move's label needs the move before it). Strengths (`STRENGTHS`, chess.com's names): Fast depth
  14 with 3 lines, Standard 18 with 6 (default), Deep 22 with 6.
- **Kept and resumed**: each position is written to `review_progress` as it is measured, so a
  review stopped part way -- Pause, the app closed, a crash -- goes on from where it was (Tai,
  2026-09-30); the finished review is saved in the `reviews` table and reopens at once. The AI
  words are kept as each key move's answer arrives (`review_words` `on_answer`), so a stopped
  request pays again only for the moves it lacks.
- **Watched and imported twins**: a watched game and its import are paired, and the watched
  review is carried to the import without an engine when the two are the same game
  (`review_room.carry_twins`); a review with AI words is never lost to a copy (Tai, 2026-10-02).
- **Relabelling**: a saved review's labels, openings, summary and Stats facts are made again
  from its saved measurements when the rules change (`rederive`, keyed on `DERIVED`), without
  an engine; the AI words are never touched. See the known gap in
  [move-classification.md](move-classification.md).
- **Numbers**: labels and accuracy from `analysis/score.py`; the summary from
  `profile/game_summary.py`; the game rating from `profile/performance.py`.
- **The review's own words** (`desktop/src/lib/review/words.ts`): a headline per kind of move and
  a sentence from the facts the review computed (the better move, what the move changed, the
  motifs), free and instant.
- **The AI words** (`coach/review_words.py`, decision 3): one call a key move, in parallel. A
  model that reasons by its effort (OpenAI's) reasons at `REVIEW_EFFORT` medium whatever the live
  coach's effort (Tai, 2026-09-30: "Reasoning none is usually not very accurate"); a local model
  does not reason here. In: the move with the coach's fact bundle for the position it was
  played from and the review's measures. Out: why it mattered, for a bad move the better move,
  and a lesson. The model names the sides only as White and Black, and `to_student` turns them
  into "you" and "your opponent": asked to say "you" itself, it addressed whoever played the
  move in about one paragraph in four (luna, 2026-09-30). Only a false claim about the board
  refuses a paragraph (`claims.check_draft`); no rule over the words holds one back (Tai,
  2026-10-02). Each move, piece, square, file and term in a paragraph is a link to the board
  (`coach/review_links.py`, `lib/review/AiText.svelte`; Tai, 2026-09-30: hovering or tapping
  must interact with the board).
- **Code**: the room `ui/review_room.py` (one per open review, polled by the page, on the puzzle
  room's pattern); the page `desktop/src/routes/review/[key]/+page.svelte` with
  `lib/review/` (ReviewBoard, EvalBar, EvalGraph, AiText, AnalysisPane, arrows).

## The phone

A 1:1 port (sidecoach-android: `analysis/GameReview.kt`, `profile/GameSummaries.kt`,
`ui/ReviewViews.kt`, `coach/ReviewWords.kt`, `coach/ReviewLinks.kt`) behind the same page, held to
the desktop by `GameReviewParityTest`, `ReviewWordsParityTest` and `ReviewAnalysisParityTest`
from fixtures the desktop code records (`tools/make_review_fixtures.py`,
`tools/make_review_words_fixtures.py` in the Android repo). The phone reviews a game only when
the player asks; its Games list has no background reviewer (`GamesApi`). A review's positions
and AI answers are kept as they come, so a review the system killed goes on from where it was.

## Calibration (phase 0)

What the 2026-09-29/30 calibration found, and why the numbers are what they are.

**Accuracy.** Ours against chess.com's on 144 of the player's game sides (the official API's
`accuracies`): 5.8 points apart on average, no lean (-1.1), r 0.83. Kept as it is.

**Where the game rating is fitted from.** Nova bot games were the plan (decision 5) and proved
unusable: Nova's accuracy is flat at about 74 from 800 to 1600, where real players' rises. The
Nova authors' evaluation sample is real: Lichess rapid (March 2026, 100,000 positions a band from
800 to 2300 and up), 17,595 game sides whole from the first move, each with the player's rating.
`tools/review/human_losses.py` judges every move as the review does (depth 18, the loss in
expected score); `tools/review/fit_performance.py` fits and tests.

**What one game says.** Little. A band's mean accuracy is a point or two above the band below's
(65 at 800-1100, 68 at 1100-1700, 71-72 at 1700-2300, 78 above), and a player's wanders ten
points from game to game. Blunders say more (2.2 per 40 moves at 800, 1.1 at 2000, 0.6 above
2300). So the model reads every move's cost in its context, not the accuracy alone.

**The model** (`profile/performance.py`): each move in one of six buckets by its loss, in one of
six contexts (the opening or after; the mover lost, level or won); each bucket's chance a softmax
quadratic in the rating per context; a side's rating the mean over a grid weighted by its moves'
chances and a prior of 1500 ± 400 (Lichess scale); fewer than 8 moves, no rating. Shown on the
scale of the game's site (ChessGoals' Lichess-to-chess.com table, July 2026, floored at 100).

**Its error.** Fitted on 1,997 sides (66,624 moves), tested on 1,418 sides of players the fit
never saw: 317 from their rating on average, r 0.62 (297 under 2000); the line through accuracy
alone 384, r 0.34. It leans to the middle (+344 at 800-1100, -260 above 2300), as any estimate
from so little must. Reading the opponent's moves too took it only from 298 to 290 and misleads
against a bot, so each side is read alone. Against chess.com's Game Rating on six games: 314
apart on average, r 0.67, far apart mainly on the bots' sides. On Tai's 42 live chess.com games
the estimates run 300-450 above the players' ratings (more time than the sample's 10+0, and the
ChessGoals table extrapolated below chess.com 815). A player's level is better read across games
(Stats).

The buckets were fitted with the move bands of the time (0.04 / 0.10 / 0.28); `EDGES` now follows
`DEFAULT_THRESHOLDS` (0.05 / 0.10 / 0.20) while `performance.json` was not refitted. See
[move-classification.md](move-classification.md#known-gaps).
