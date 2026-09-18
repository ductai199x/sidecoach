# Coach evaluation and ICR: method, results, decisions

State as of 2026-09-18. Everything below was measured with the tools in `tools/eval/`
on data under `data/eval/` (not committed; layout in `data/eval/README.md`); the raw runs
that produced numbers not needed again are in `data/eval-archive-2026-09-18.zip`.

## What ships

- **C = twelve distilled lessons** (`src/sidecoach/coach/context/corrections.md`, ~0.8k
  tokens), appended to the system prompt under "Lessons from earlier games". One line per
  situation, general to any game, each with a placeholder example.
- **Draft-time checks** on every candidate reply (`Speaker._problems`): the disclosure
  leak check, the board-claim checker, the directive's length (two sentences of slack)
  and closing question, the verdict word and the cost stated for the move just played.
- Models: `openai:gpt-5.6-luna@medium` for the coach and the writer; the summariser at
  effort none; the judge at effort none everywhere.

## The method (Tai's, 2026-09-18)

Master games only (the books carry a true reference). Per game one side; the game is played
from move 1 to the end through the **real** coach; at move k the harness carries the
coach's real earlier answers, as live. Every moment the policy fires (verdict, decisive
question, stuck hint, the player's questions) is one (input, output) pair; the judge scores
it; the worst become (input -> corrected output) exemplars. Rounds after the first sample
5 moments per game (1 early, 2 mid, 2 late; a fresh seed each round; never a moment already
in C) and replay each game only to its last sampled ply. Demonstrations are written by the
writer model (three candidates, each checked and judged), one per class per round.

100 games from the book collections, split 80 training / 20 held out (`icr-100/games.json`,
`split.json`). Held-out evaluation: 16 sampled moments per game = 320 moments; this is the
noise floor that separates differences of ±.03 (160 moments do not).

## The judge (binary, no reasoning)

`tools/eval/judge.py`. Per reply: deterministic checks (leak, board claims, form); the
reply split into atomic claims by the model at effort none; every claim the record can
decide is decided in code, final (counts vs the tally; costs; material; a line recited
from the engine lines; an effect phrase the harness states; worth-N-pawns; "almost as
strong as" the preferred move; a named theme; loose/undefended by counting defenders; the
opening name); the rest put to one yes/no question ("Does the record state this, in these
or other words?"), majority of three votes; tactical claims also to the board checker
(python-chess), which never overrides a code decision, tries an unowned piece for both
sides and the frames after the moves the reply names. Claims the decomposition should not
have produced are dropped, never voted: appraisals with nothing identifiable, under three
words, a piece with no square/move/material word, hedges naming no move, anything whose
move/square/piece the reply never names, anything drawn from a question sentence.
Required facts: a praise/criticism pair for verdict labels, the opening name in code, one
direct question otherwise. Relevance in code (a fact question answered only with
questions is 0). Pass = no leak, no false board claim, no unsupported claim, form ok,
relevance >= .5.

Cost of one 320-moment judge pass: ~$0.41 (the record in the **system** prompt is what
luna's cache serves again: 86% cached; in the user message it was 59% and $0.93).
Re-judging only the failed items is ~$0.02 (judgments are cached per game file).

## Results, held out (320 moments, 20 games)

All rows judged with the same code except where noted; pass / completeness / fabrication /
leak. Runs are from `icr-100-heldout-*.log`.

| C in the system prompt | tokens | pass | compl | fab | leak | notes |
|---|---|---|---|---|---|---|
| none (before the harness fixes) | 0 | .89 | .82 | .06 | .00 | first baseline |
| none, harness fixed (`noC-noC2`) | 0 | .93 | .86 | .06 | .00 | opening line + opponent-move directive gave the completeness |
| 12 demonstrations, full | 14.4k | .92 | .83 | .05 | .00 | |
| 12 demonstrations, lean (record cut to 3 plies) | 10.1k | .92 | .87 | .04 | .02 | |
| 12 demonstrations, compact record | 14.4k | .88 | .85 | .06 | .02 | rejected |
| 7 demonstrations, small | 5.4k | .86 | .86 | .08 | .05 | rejected |
| 12 demonstrations, task-only | 4.8k | .86 | .87 | .06 | .03 | rejected |
| C after round 5 (9 demos) | 11.3k | .89 | .86 | .09 | .01 | |
| lessons v1 (7 lines) | 0.5k | .87 | .90 | .07 | .05 | judge v5 |
| lessons v2 (8 lines) | 0.6k | .94 | .91 | .03 | .00 | shipped 2026-09-18 midday |
| lessons v3 (10 lines) | 0.7k | .93 | .86 | .05 | .00 | |
| **lessons v4 (12 lines)** | **0.8k** | **.994** | **.85** | **.00** | **.00** | **shipped; judge v12; 2 fails, both the coach's** |

Paired on the same 320 moments (`compare_rounds.py`), no C vs lessons v4: claim support
.97 -> 1.00, fabrication .06 -> .00, pass .93 -> .99; 18 moments better, 0 worse (sign
test p < .01). Completeness .86 -> .85, unchanged within noise.

Lessons v4 under earlier judge versions: .97 (11 fails, 7 of them judge errors), .975
(8 fails, 6 judge errors), .994 after each audit's rules were coded. The two remaining
fails: an evaluation question answered only with a question; a decisive-moment prompt
without its closing question (the draft check now enforces the latter).

Completeness stayed ~.85-.91 across every C: it is bounded by the harness and the
directives, not by C. The gain from .82 to .86 came from the harness (the opening line;
the opponent's-move directive), not from any C.

## Training rounds (icr-100, 80 games)

| round | moments | pass | compl | fab | demos added | cost |
|---|---|---|---|---|---|---|
| 1 (full play, catalogue) | 400 | .90 | .83 | .05 | 1 | ~$2.2 |
| 2 | 357 | .87 | .80 | .06 | 1 | ~$5 |
| 3 | 375 | .92 | .81 | .03 | 3 | ~$4.5 |
| 4 | 368 | .93 | .84 | .02 | 3 | ~$9 |
| 5 | 373 | .95 | .89 | .03 | 3 | ~$6 |
| 6 | 367 | .90 | .88 | .06 | 3 | ~$9 |

The 12 demonstrations (`icr-100/pool.json`): ask_opening, ask_threat x4, ask_game_so_far
x2, ask_last_move, ask_material x2, ask_plan, ask_eval. The lessons were distilled from
them by the writer model (`icr_play.py --distil`, `icr-100/lessons.md`).

Earlier: the 12-game `icr-play` run (r1 pass .69 fab .29 -> r2 .76/.22) and the pilot
(.65 -> .67) were on the old judge (a reasoning grader) and are not comparable.

## Summarisation and the record

Folding dialogue older than 20 plies into one block (`summarise_every=20`): -15% prompt
tokens, scores unchanged, wall time +3% (the summariser's own calls); a cost lever, not a
latency lever. The compact record (sound moves nobody spoke after as a move list) gave
nothing: nods dominate the dialogue. Neither is on by default in the app.

## What the five play-through reads fixed (all committed)

Facts in the third person per side; the Facts heading closed on the opponent's turn;
material names pieces and the expected recapture; nature looks down the line; motifs by
name; the decisive question once per 8 plies; plan questions unlock THEME without a
canned template; exposedKing and defensiveMove as themes; the verdict says where the
preferred line leads; the opening line in the record; the opponent's-move directive.

## Spend

Ledger (`data/eval/usage.jsonl`, every paid call): gpt-5.6-luna 798M prompt tokens
(522M cached), 19M completion, ~$89 at luna prices over the project; the board checker's
extractor was unmetered until 2026-09-18 evening, so the dashboard runs higher. Rules that
came out of it: estimate before spending; pilot; one process at a time; no mid-run
restarts; exemplars from already-judged moments, never a replay for its own sake;
re-judge only what failed.

## ChessQA

`tools/eval/chessqa.py`, `data/eval/chessqa/`: the paper's prompt .58 -> the object prompt
.86 (flash direct), .90 flash thinking, .92 max thinking; the diagram hurts, the eval
question needs words and the side, lists must say they are complete. Semantic is the
category ICR targets.

## Next, if wanted

Scale the exemplars to 30-50 from the already-judged training moments: round 1's 400
judged moments are kept with their harnesses (`icr-100/items-r1`, ~40 fails to draw
from; the later rounds' play-throughs were deleted, so their moments would need a replay
to the ply, ~$0.1 a game); one writer call per exemplar, re-distil once, one held-out pass
(~$0.4 judge-only on cached replies). Completeness is the harness's to raise, not C's.
