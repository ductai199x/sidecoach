# Game Review

A review of a finished game modelled on chess.com's Game Review -- its flow, depth and feel,
Diamond tier included -- in Sidecoach's own look (Tai, 2026-09-29: "I want the feel and the UI
UX to be modelled after chess.com ... but with our own visuals and design language"; "for
diamond tier subs their feature is actually really insanely cool and I kind of want that too").
Branch `game-review` in both repositories.

The research behind it: chess.com's help centre, news posts and ~30 of their screenshots (web,
phone, the 2021 launch, the 2023 v2, the 2024 redesign, the settings), summarised in the
session's research report; the screenshots are not committed.

## Decisions (Tai, 2026-09-29)

1. **Performance rating per game: yes**, a fitted estimate that does not use the players'
   ratings (chess.com's is criticised for echoing them).
2. **Arrows off by default** (changed 2026-09-30: "if the gold best move arrows are on then
   there's no point having the retry feature"): the best move in gold, and threats (red) and
   ideas (blue), each a switch in the review's settings. With the gold arrow off, the coach's
   sentence does not name the better move either; Best shows it.
3. **The coach's words on every move**, written by the model and **prepared ahead of time** for
   all moves, not on demand.
4. **The review opens in a window of its own** (as puzzles do); a full-screen route on the phone.
5. **No per-game chess.com link import** (its only instant source is an unofficial endpoint).
   Games come from what Sidecoach watched, chess.com by username through the official public
   API, Lichess by username, or a pasted / opened PGN. Calibration data comes from Nova bot
   games and master games, not from chess.com.
6. **No coach characters:** the review speaks as the coach does in play, with the persona and
   voice chosen in Settings.

## What we take from chess.com, and what we do better

Taken: the summary screen, the step-through review with the coach's card on each move, key
moments and Next, Retry, Show Follow-Up, the eval bar and a clickable eval graph, self analysis
with engine lines, the coach's voice, a library of games and insights across them; Diamond's
explanations on every move, deep engine settings, the Score / Time / Best Move Diff / Themes
charts, game shapes, Share.

Better, from what their users complain about most:

| Their complaint | Ours |
|---|---|
| The game-over teaser shows labels the full review takes back | No teaser: what is shown is final; games watched live reuse the live depth-18 verdicts. |
| Brilliant / Great feel random; the text is generic or wrong | Brilliant stays rare and rating-independent; every explanation is built from the engine line and checked by the claims checker. |
| Game Rating echoes the players' ratings | Fitted on games at known strength; ratings not an input. |
| Retry and Hint were removed (May 2026) | Kept, with the three-level hint ladder (idea, where to look, the move). |
| Redesigns lost the whole-game overview | Graph and two-column move list always visible on the desktop. |
| The best-move arrow gives the answer away | Off by default, and the words keep the move back while it is off. |
| The coach, voice and pop-ups intrude | Voice and avatar each have a switch. |

## The screens, in chess.com's order

- **A. After a game:** the review card at the end gets a green **Game Review** button.
- **B. Summary:** the coach's one-line verdict in a bubble; the eval graph (expected score, a
  class-coloured dot on each key move, click to jump); players; accuracy each (white box for
  White, dark for Black); the counts table -- Brilliant !! #26C2A3, Great ! #749BBF, Best ★
  #81B64C, Excellent 👍 #81B64C, Good ✓ #95B776, Book 📖 #D5A47D, Inaccuracy ?! #F7C631,
  Mistake ? #FFA459, Miss ✕ #FF7769, Blunder ?? #FA412D (their colours, measured; ours come from
  our own badge palette) -- collapsed to six rows with a chevron; performance rating each; phase
  grades for opening / middlegame / endgame (a class icon, the phase accuracy on hover); the
  game's shape (Smooth, Balanced, Sharp, Sudden, Intense, Wild, Giveaway); **Start Review**.
- **C. Step-through:** the move's squares tinted in its class colour and its badge on the
  destination's corner; the arrows the settings turn on (gold the best move, red a threat, blue
  an idea); an eval
  bar by the board (vertical on the desktop, horizontal on top on the phone). The coach's card:
  "Rc2 is a blunder", an eval pill (+3.11, M2), one to three sentences with tactic words
  (fork, pin...) as chips that draw on the board, and the fitting button (Show Follow-Up, Show
  Miss, Show Checkmate...) that plays the line with a delay. Actions: Hint, Best, Retry, Next
  (to the next key moment); after a mistake Retry is the main button; Great and Brilliant get
  Share. Below: the two-column move list with class icons and colours on key moves (Me /
  Opponent / Both), the mini graph with the current move marked, the navigation buttons and the
  arrow keys. The voice reads each card; captions follow. At the last book move, the opening
  panel: the name, "you have played this N times, W% won", masters' continuations, Explore.
- **D. Retry:** the player moves on the board and is graded by the same rules ("Nxd5 is best --
  the only move that works!", "Nb5 is a blunder -- only one move worked there"), three graded
  hints, unlimited tries; at the end "Accuracy after retries 94.5 (+6.6)" with a You played /
  New move table.
- **E. The last card:** the coach's summary, the performance ratings and phase grades, and
  training from this game (practice puzzles for the motif missed, the habit it shows).
- **F. Self analysis:** free moves with variations inline; engine lines (count, depth); Explorer
  (masters, your own games); Details (game info, PGN copy / download); charts Score, Time (from
  PGN clocks or measured live), Best Move Diff, Themes.
- **G. Library and insights:** every watched or imported game, filters (source, result, time
  control, date), accuracy and rating chips, Review; insights across games -- accuracy trend,
  game shapes, phase strengths, tactics found v missed, openings -- joined to the profile's
  habits and causes.
- **H. Review settings:** arrows (best, coach's, both, none), whose key moves, classification
  on the board, autoplay delay, voice, time spent per move, engine strength.

## How

- **Analysis:** `GameAnnotator` over the whole game, both sides, several engines in parallel,
  presets Fast (depth 14, 2 lines), Standard (18, 3; default), Deep (22); progress shown; the
  result cached in a `reviews` table so a review reopens at once.
- **Numbers:** the existing labels and accuracy (fitted to chess.com, within 4.4 points); phase
  accuracy; a performance-rating model fitted on real players' games (see Phase 0 below); game
  shape and key moments from the eval curve.
- **Words:** a template per class on the facts already computed (threat, material, the
  refutation line, the motif) -- free, instant, the fallback; the model rewrites every move,
  prepared in the background after the analysis, at the configured effort, through the claims
  checker; spoken by the configured voice.
- **Code:** `analysis/game_review.py` (the whole-game pass), `profile/` (summary, phases, rating,
  shape), `coach/review_words.py` (the words); a review room on the puzzle room's pattern;
  endpoints `/api/games`, `/api/games/import`, `/api/review/{id}/...`; routes `review/[id]` and
  `games`; components EvalBar, EvalGraph, the board's badge overlay, the review move list, the
  coach card.
- **Phone:** the same page (stacked: eval bar on top, the swipeable move strip, the bottom action
  bar); the endpoints and logic ported 1:1 with parity fixtures; its engine is slower (depth
  14), a review takes a minute or two with progress, a game it watched is at once.

## Phases

| # | Phase | Finish line |
|---|---|---|
| 0 | Calibration | Accuracy compared with chess.com's on Tai's own games (official API); phase-grade bands; the performance model fitted and its error reported per band; key-moment rules. |
| 1 | Review engine and imports | Every game source; the parallel analysis with presets; the `reviews` cache and endpoints; tests. |
| 2 | Summary and step-through (desktop) | B, C and H with graph, eval bar, overlays, template words, voice and captions; run end to end in the real app. |
| 3 | Retry and the last card | D and E. |
| 4 | The model's words | Every move prepared ahead, checked; tactic chips drawing on the board; the opening panel. |
| 5 | Self analysis, library, insights, share | F and G; annotated PGN export. |
| 6 | Phone | The 1:1 port, parity fixtures green, installed on both. |

Cost: no paid development spend; model calls run only through the coach's own settings.

## Phase 0: what the calibration found (2026-09-29/30)

**Accuracy.** Ours against chess.com's own on 144 of the player's game sides (the official API's
`accuracies`): 5.8 points apart on average, no lean (-1.1), r 0.83. Kept as it is.

**Where the performance rating is fitted from.** Nova's games were the plan (decision 5) and
proved unusable: Nova's accuracy is flat at about 74 from 800 to 1600, where real players' rises.
Maia-3 was next, but real games beat any model of them, and the Nova authors' evaluation sample
is real: 600,000 positions of Lichess rapid (March 2026, 100,000 a band from 800 to 2300 and up),
17,595 game sides whole from the first move, each with the player's rating. `tools/review/
human_losses.py` judges every move of a side as the review does (depth 18, the loss in expected
score); `fit_performance.py` fits and tests.

**What one game says.** Little. A band's mean accuracy is only a point or two above the band
below's (65 at 800-1100, 68 at 1100-1700, 71-72 at 1700-2300, 78 above), and a player's wanders
ten points from game to game. Blunders say more (2.2 a 40 moves at 800, 1.1 at 2000, 0.6 above
2300). So the model reads every move's cost, in its context, not the accuracy alone.

**The model** (`profile/performance.py`): each move in one of six buckets by its loss (best,
under 0.02, under 0.04, inaccuracy, mistake, blunder), in one of six contexts (the opening or
after, the mover lost / level / won); the bucket's chance a softmax quadratic in the rating per
context; a side's rating the mean over a grid weighted by its moves' chances and a prior of
1500 +/- 400 (Lichess scale); fewer than 8 moves, no rating. Shown on the scale of the game's
site (ChessGoals' Lichess-to-chess.com table, July 2026, floored at 100) with the middle 80%.

**Its error.** Fitted on 1,997 sides (66,624 moves), tested on 1,418 sides of players the fit
never saw: 317 from their rating on average, r 0.62 (under 2000, the players Sidecoach serves:
297); the line through accuracy alone 384, r 0.34. It leans to the middle (+344 at 800-1100,
-260 above 2300), as any estimate from so little must. A fit on the first 350 sides alone was
7-10 points nearer on the same test, within the noise, and predicts the moves themselves worse
(held-out log-loss 1.3377 against 1.3366), so the larger fit ships. Tempering the evidence (the
moves of one game are not independent) and other priors, chosen on half the held-out players,
did no better on the other half. Reading the opponent's moves too (rated opponents are paired
close: gap SD 131) took it only from 298 to 290 on 262 whole games, and misleads in a game
against a bot, so each side is read alone.

**Against chess.com.** On the six games chess.com gave a Game Rating: ours 314 from theirs on
average, r 0.67; on the player's own six sides much nearer (1511 v 1850, 1948 v 1850, 1233 v
1300, 1780 v 1800, 1689 v 1400, 941 v 850), and far from theirs only on the bots' sides, whose
mistakes are not a person's. chess.com's own number ranges 850-1850 for the same ~970 player.
On the player's 42 live chess.com games our estimates run 300-450 above the players' ratings:
those players blunder like Lichess 1400-1700 players, and 15+10 gives more time than the
sample's 10+0; the ChessGoals table is extrapolated below chess.com 815. The per-game number is
shown with its range; a player's level is better read across games (Phase 5's insights).
