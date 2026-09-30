# Stats: what a player's games say about them (plan)

chess.com's **Advanced Stats** (Diamond; beta 2026-04-01, launched 2026-06-15, replacing the older
**Insights** at the end of 2026) rates a player's openings, tactics, strategy and endgames per time
control against players of their rating, and keeps a gallery of their brilliant moves. Sidecoach's
**Stats** does the same from every game it has reviewed, in its own design, with what players say
chess.com's lacks. Researched 2026-09-30: chess.com's help and news articles and its shipped page
text, 116 screenshots of both products, about 120 Reddit and forum threads and 372 comments on
chess.com's own posts, Lichess's open-source Insights and Tutor (lila `modules/insight`,
`modules/tutor`), Aimchess and the literature; the notes and screenshots are in the session's
research folder, not committed.

## Decisions (Tai, 2026-09-30)

1. **A window of its own that works like the puzzle screen:** the board on the left shows the
   positions behind whatever is chosen (a missed fork, a lost endgame), to step through and retry;
   the stats on the right.
2. **Every game**, with filter chips for games against people and against bots (and the time
   control, the date range, the source).
3. **Imported games are reviewed on their own in the background**, at low priority, paused while
   a game is being watched; Stats counts what is reviewed and says how many are waiting.
4. **Category ratings** (openings, tactics, strategy, endgames), fitted as the game rating is: on
   real players' games at known ratings (the Nova authors' Lichess rapid sample, 3,409 sides
   judged at depth 18), each on its own kind of moves. The peers' column comes from the same games.
5. **Order:** Game Review's Retry (phase 3) first -- Stats reuses it -- then Stats, then the coach's
   words on every move (Game Review phase 4).

## What chess.com has

Filters: time control, date range (7 / 30 / 90 days, a year, all). A rating per category and time
control, recent games weighing more, shown after 5 games. Every category page: the rating's line
over time, then a table Stat | Grade | You | Peers, the grade a move-class icon for how the player
compares with peers of their rating; a row opens to its definition and its trend.

| Section | chess.com's definitions |
|---|---|
| Openings | Opening accuracy; average mistakes, misses and blunders in the opening a game; average book moves when the player leaves book first; performance by opening (games, win %, grade, accuracy, peers) as White / as Black |
| Tactics | Found / Allowed, nine kinds: checkmates, free pieces, forks, pins, discoveries, trapped pieces, removing the defender, brilliant sacrifices, other. Found: "percent of material-winning forks found when available"; allowed: per 1,000 moves |
| Strategy | All / Open / Semi-open / Closed: accuracy in non-tactical positions; attacking, defending, active-piece, pawn-structure, space-advantage accuracy |
| Endgames | Win % from a worse / equal / better position at the endgame's start (a draw half a win); accuracy in king-and-pawn, rook-and-pawn, queen and other endgames, and overall |
| Brilliant moves | Every "!!" as a board with the move drawn, filtered by the piece given up or an underpromotion; each opens its game |

From the older Insights, what players valued and Advanced Stats dropped: a grid of the latest
found and missed positions per tactic, each opening Game Review at that move with a retry;
comparisons with similar players and with the player's own past; how games were won and lost.

Not published: any formula -- the peer band, the tactic detection, "non-tactical", where the
endgame starts, the thresholds. No time-management stats in either product.

## What players say

Loved: missed tactics against peers with a retry ("I miss mate in 1 nearly half the time"),
results by opening, the brilliant gallery (delight, not teaching), progress over time.
Complaints: not actionable ("fun facts"); numbers that look wrong (a mate in 18 counted as a
missed mate, openings misnamed); no way from a number to its games; small samples and time
controls mixed; peers never defined; features moved behind the paywall. Wanted: training from the
player's own mistakes; why, at their level; each number linked to its positions; time
management (time trouble, losses on time from winning positions); trends before and after
training; every game counted (other sites, over the board).

## How ours is built

- **Every number opens its positions**: a grid of boards; one chosen goes on the board, to step
  through or retry (Game Review's Retry), and opens the game's review at that move.
- **Work on this**: the three weaknesses that cost the most expected score, each with its
  positions, puzzles of its motif (the Lichess set), and the coach's one line on why.
- **Definitions (ours, shown on the page):**
  - *Tactic chance*: a position where the mover had a clearly winning move (Lichess's puzzle rule:
    a large jump in winning chances, one move clearly best), found if played; its kind from the
    tagger on the best line; allowed when the move handed the opponent one. Only chances a player
    at the level could find count (no mate in 18).
  - *Endgame*: from 6 or fewer pieces other than kings and pawns (Lichess); its kind from the
    material then; better / equal / worse from the expected score then.
  - *Strategy*: open / semi-open / closed from the centre pawns; non-tactical = no tactic chance
    for either side; attacking / defending from who stands better; space, activity and structure
    from the positional layer.
  - *Openings*: named by position (transpositions come out right), book moves against the opening
    book, the player's score against the result expected from the ratings.
- **Beyond chess.com**: time management (time by phase, mistakes under time pressure, losses on
  time from winning positions, time left in losses); how games end; misses and blunders galleries
  beside the brilliants.
- **Honest numbers**: n beside every figure, small samples greyed, intervals on hover, time
  controls apart by default, every definition a click away.
- **Data**: each reviewed game's facts kept per move when it is reviewed, so the page is at once.

## Phases

| # | Phase | Finish line |
|---|---|---|
| S1 | Facts and the backlog | Per-move facts stored at review; the background review of imported games (low priority, paused while watching); the definitions above with tests on known positions. |
| S2 | Peers and ratings | The facts of the 3,409 judged real sides by band (peers); the category ratings fitted and their held-out error reported. |
| S3 | The window | The puzzle-screen layout; overview, openings, tactics, strategy, endgames, time, results, galleries; filters and chips; positions to the board with Retry; run end to end. |
| S4 | Work on this | Ranked weaknesses, puzzles by motif, the coach's words. |
| S5 | Phone | The 1:1 port with parity fixtures. |

Cost: no paid development spend.
