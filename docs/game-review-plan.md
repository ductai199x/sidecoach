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
2. **Best-move arrow:** shown by default, hidden while the player retries a position.
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
| The best-move arrow gives the answer away | Hidden while retrying. |
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
  destination's corner; the best move as a green arrow, the coach's own arrows orange; an eval
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
  accuracy; a performance-rating model fitted on Nova games (800-2300, Lichess scale, converted)
  and master games (2300+); game shape and key moments from the eval curve.
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
