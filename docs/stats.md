# Stats

What a player's reviewed games say about them, in a window of its own: chess.com's **Advanced
Stats** (Diamond; it rates openings, tactics, strategy and endgames per time control against
players of the same rating, and keeps a gallery of brilliant moves), in Sidecoach's design,
with what players say chess.com's lacks. The numbers come from the reviews of
[game-review.md](game-review.md); the move labels from
[move-classification.md](move-classification.md); the profile record they live beside is
[DESIGN.md](../DESIGN.md) section 5.6.

The research behind it (2026-09-30): chess.com's help and news articles and its shipped page
text, 116 screenshots of Advanced Stats and the older Insights, about 120 Reddit and forum
threads and 372 comments on chess.com's own posts, Lichess's open-source Insights and Tutor
(lila `modules/insight`, `modules/tutor`), and Aimchess. The notes and screenshots are not
committed.

## Decisions (Tai, 2026-09-30)

1. **A window of its own**, as the puzzles have one (02:40: "similar to puzzles" meant a separate
   window, not the puzzle screen's layout), laid out as chess.com's Advanced Stats: a menu of the
   sections, the filters, each section's figures; a figure opens its positions as small boards,
   each opening the game's review at that move, where Retry is.
2. **Every game**, with a filter for games against people or bots, the time control and the date
   range.
3. **Every game reviewed in the background** on the desktop, as a job the player sees and
   controls ("we don't want to kick off a long running intensive cpu job without the user's
   notice"). Only the player's Pause stops it (Tai, 2026-10-01: it once stepped aside on its own,
   which he never asked for). The phone has no background review (Game Review, decision 7).
4. **Category ratings** (openings, tactics, strategy, endgames), fitted as the game rating is, on
   real players' games at known ratings, each on its own kind of moves; the peers' column from
   the same games.
5. **Work on this** uses the page's own words for each area, no model (Tai, 2026-09-30: a model
   call here could cost a lot).

## The code

- `profile/stat_facts.py`: each reviewed move's facts and the game's, read off the review once and
  kept in `stat_games` and `stat_moves` (`profile/store.py`), so the window adds them up at once.
- `profile/stats.py`: every section's figures for the filter's games (`stats`, `figures`), the
  ratings, the peers and grades (`compare`), Work on this (`work`), and the positions behind any
  figure (`positions`).
- `ui/backlog.py`: the background review. First the facts of reviews saved before Stats (no
  engine), then each game no review covers, latest played first, at the standard strength, one
  at a time on two engines below normal priority; its progress on the panel and in the Games
  window; paused stays paused across restarts; a review stopped part way goes on from where it
  was.
- `ui/server.py`: the `/api/stats` endpoints (`_filter`; `_drills` adds the puzzles).
- `desktop/src/routes/stats/+page.svelte`, with `lib/stats/TrendChart.svelte` and
  `Sparkline.svelte`.
- `tools/review/fit_stats.py` and `tools/review/human_facts.py`: the peers and the category models.
- The phone: `profile/Stats.kt`, `profile/StatFacts.kt`, `ui/StatsApi.kt` in sidecoach-android,
  held to the desktop by `StatsParityTest` (fixtures from `tools/make_stats_fixtures.py`).

## The window

Sections, in the menu's order: Overview, Openings, Tactics, Strategy, Endgames, Time, Brilliant
Moves. Filters: the time control, the date range (30 days, 90 days, a year, all time), and
opponents (all, people, bots); the default is every time control. Each section shows its trend
and a table of figures: the player's value, the peers' and a grade; a row opens to its definition
and the positions behind it. Every figure carries how many it counts (`n`); a small sample is
greyed and gets no grade. The page says how many games are still waiting for the background
review, and why games with no known side for the player are left out (no username of theirs in
them).

- **Overview**: the results, the ratings (overall and per category, with their trends), Work on
  this, the results by how the games ended, and the game shapes.
- **Openings**: opening accuracy; mistakes, misses and blunders in the opening a game; book moves
  a game in the games where the player left book first; each opening family by side (games,
  score, accuracy).
- **Tactics**: per kind (chess.com's nine: checkmate, fork, pin, discovery, trapped piece,
  removing the defender, free piece, sacrifice, other), chances found of chances had, and chances
  allowed per 1,000 moves.
- **Strategy**: accuracy in non-tactical positions, all and by structure (open, semi-open,
  closed), attacking (standing better), defending (standing worse), and by theme (space,
  activity, structure).
- **Endgames**: the score from a better, equal or worse position at the endgame's start; accuracy
  overall and in pawn, rook, queen and other endgames.
- **Time**: time a move by phase; the mistake rate under time pressure and calm; games lost on
  time, and lost on time while standing better at the last move.
- **Brilliant Moves**: galleries of the player's brilliant, great, miss and blunder moves as
  boards, each opening its game's review.

## Definitions

Ours, as chess.com publishes none; each is shown on the page.

- **Tactic chance** (`stat_facts.py`): the mover's position had one move clearly best (the
  review's decisive call: the best line 0.10 ahead of the next), the line is a tactic (the
  tagger's motif, within the mover's first two moves), it leaves the mover at least level, and a
  mate counts only in three or fewer (no mate in 18 is "missed"). **Found** when the move played
  cost under 0.02 (`FOUND_LOSS`, Lichess's tolerance for "the same move").
- **Allowed**: the chance the move handed the opponent, by the same rule, in the position after.
- **Non-tactical**: a move whose own position held no tactic chance for the mover, and not a book
  move (`CATEGORIES["strategy"]`). The opponent's chances are not checked.
- **Structure**: open (a pawn or none on the d- and e-files), closed (a d- or e-pawn blocked head
  on by an enemy pawn), else semi-open.
- **Themes**: *space* when the mover has more of it, *activity* when a rook or queen of theirs
  stands on an open file or a minor piece on an outpost, *structure* when either side has a weak
  pawn (from `analysis/positional.py`).
- **Endgame**: from the first position with 6 pieces or fewer besides kings and pawns (Lichess's
  divider, `analysis/phase.py`); its kind from what is left; better, equal or worse by the
  mover's expected score there (0.6 and 0.4).
- **Opening**: named by position (transpositions come out right), grouped by family.
- **Time pressure** is three things in the code, not one: Stats uses the time left by time
  control (under 10 s in bullet, 20 s in blitz, 60 s in rapid: `stats.PRESSURE_MS`); the cause of
  a mistake uses under 10 s whatever the control (`causes.TIME_PRESSURE_MS`); and the profile's
  SQL examples for the coach group by under 30 s (`ask.py`).

## Peers, grades and category ratings

Fitted 2026-09-30 by `tools/review/fit_stats.py` on 1,800 judged Lichess rapid sides (six bands
of 300, from 800 to 2300 and up; `tools/review/human_facts.py`).

- **Peers** (`profile/peers.json`): each band added up by Stats' own functions (`stats.figures`),
  and each figure a weighted line through the bands, value against the band's mean rating. Left
  out: book moves before leaving book (needs the opponent's moves) and time a move (the sample's
  increment is unknown). The peers are players of the player's rating -- the mean of the ratings
  their games against people record, as chess.com's are -- and of their level of play only when no
  game records one. Against weaker players accuracy comes easier, so a player who plays above
  their rating reads ahead of their play's peers figure for figure, which says nothing (Tai's
  games against ~1000s: opening accuracy 80 against 73 for Lichess 2300s).
- **Grade** (`stats.grade`): how many rating points ahead of the peers the figure is, read back
  along the line, named as the kinds of move: brilliant from +400, great +250, best +125,
  excellent +40, good within 40, inaccuracy to -125, mistake to -250, blunder below. A figure of a
  few is pulled towards the peers' as if it had ten more at their value (`SHRINK`); a figure whose
  line fits with a weighted r² under 0.5 (`FOLLOWS`) gets the peers' value but no grade.
- **Category ratings** (`profile/categories.json`): the game rating's model fitted on each
  category's moves alone, shown from 20 of its moves (`CATEGORY_MOVES`). Held out (542 sides of
  players the fit never saw), mean absolute error in rating points:

  | Category | Moves a game | One game | Ten games pooled | Game model, same moves |
  |---|---|---|---|---|
  | Openings | 7.9 | 350 | 147 | 345 / 151 |
  | Tactics | 1.8 | (too few) | 279 | 384 / 399 |
  | Strategy | 32 | 336 | 138 | 337 / 163 |
  | Endgames | 6 | 411 | 344 | 426 / 586 |
  | All moves (the game model) | 35 | 328 | 176 | |

- **Pooled, not averaged**: a mean of one-game ratings leans to the middle (each game's prior pulls
  it): +310 for players of 900-1200, -233 at 1800-2100, -386 at 2100-2400 over ten games. Their
  moves pooled lean under 100 either way. So the rating's trend pools the last ten games' moves
  (`RATING_GAMES`); each game's own rating is a dot.

These models were fitted with the move bands of the time (inaccuracy 0.04, blunder 0.28). Since
2026-10-06 the labels use chess.com's 0.05 and 0.20, and the fitted files have not been refitted:
the peers' figures that count labels (mistakes, misses, blunders) and the rating buckets
(`performance.EDGES`) now read moves by bands they were not fitted on.

## Work on this

Every costly move counts in one area (`stats.area`): a tactic chance missed (by kind); else, for a
mistake, miss or blunder, a tactic handed over (by kind), the endgame's (by kind), the opening's,
or a quiet position's. Each area's cost is the expected score its moves lost, a game, with its
peers'. The three dearest (`WORK`) are named once there are 3 games (`WORK_GAMES`) and the area
has 3 moves (`WORK_MOVES`), each with its positions and three puzzles of its motif at the
player's puzzle level (`server._drills`). Areas are named as skills to work on, never as a list
of failures (Tai, 2026-10-02: "you missed" on every row read as a charge sheet). The coach's
"biggest cost" names the same first area (`biggest_cost`).

## Not built

- The score against the result expected from the ratings, per opening (`_openings` gives the plain
  score).
- A filter by source (chess.com, Lichess, watched, pasted).
- Intervals on hover, and time controls kept apart by default (the default filter takes all).
- Insights' "how games were won and lost" beyond the results by how they ended.
