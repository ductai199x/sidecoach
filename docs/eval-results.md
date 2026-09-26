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

## The coach benchmark (2026-09-25)

Why it exists: the quality rounds before it drew 40 moments at a time from five 1300-level games. Those were the games being tuned on, with no held-out set, and nothing was paired between versions. So no change could be shown to generalise (Tai: "each round is different...so how tf are you able to know if what you change is stable").

**Method** (`tools/eval/bench.py`):
- **Games:** Nova against Nova at 800, 1200, 1600, 2000 and 2500, 12 games a level, the coached side alternating. The coach's rating is set to each game's level.
- **Split by game:** 6 games per level for dev (tuning) and 6 for test (locked), frozen in `data/eval/bench/manifest.json`.
- **Moments:** fixed and paired. 150 on dev and 300 on test, balanced by level and by family (after a move, hints, answers).
- **Cheap runs:** only the drawn moments call the model; the history before each is the templates'.
- **Statistics:** rates with 95% intervals by level and family; versions compared paired (McNemar, bootstrap).

**The judge, proven before any number is trusted:**
- Two independent labellers (Claude, blind to the judge) labelled 80 replies (60 AI, 20 templates) for false claims, required facts and every rubric answer. They checked board claims with python-chess.
- Every disagreement with the judge was adjudicated by hand (6 labels corrected; `data/eval/gold/bench-dev-base-adjudication.jsonl`).
- The judge was fixed against two thirds of the labels ("tune"). Its accuracy was then measured on the third it never saw ("confirm").

Fixes along the way:
- A claim grounded in the reply's own words is always judged; 142 had been dropped unjudged.
- The balance is read of the side the claim names.
- Required facts the record decides are decided in code: the cost, counts, a named move or square.
- Tone questions reworded.
- Two question criteria no longer asked: 80% agreement, reworded twice.

| Held out (confirm) | Agreement |
|---|---|
| required facts | 22/22 |
| gushing, scolding, hedging | 26/26 each |
| actionable | 14/14 |
| question can be answered from the board | 12/12 |
| why | 12/14 (lenient twice) |
| transfer | 12/14 (lenient twice) |
| false fabrication flags | 0 |
| **false claims caught** | **0 of 4** |

**Known blind spot:** subtly false explanations of engine lines. Examples:
- "the point of a4 is to take away the bishop's last safe square" (it had none already);
- "Rad8 keeps the b- and e-files open" (they are not open yet);
- "the check forces your rook to f7" (other moves exist).

The labellers found 5 in 60 AI replies, 4 of them answers to "show me the line". A stricter vote wording caught none and faulted 10 true claims, so it was not adopted: a no-reasoning yes/no vote cannot tell "forces" from "is followed by". The fix belongs at the source (the line directive) and in the labelling protocol.

**Dev baseline** (`bench-dev-base-g4`, code 83dc1c3, 150 moments):

| | AI | Template |
|---|---|---|
| pass | 0.99 [0.95, 1.00] | 0.95 [0.89, 0.97] |
| completeness | 0.95 [0.93, 0.97] | |
| excellence | 0.99 [0.98, 1.00] | 0.86 [0.82, 0.89] |
| preferred over the template | 95 of 110 | |

- **Levels:** no level differs outside the intervals.
- **What the numbers leave out:**
  - the line-explanation errors (about 1 AI reply in 12 per the gold set);
  - the pitch to the player's level, which no criterion asks.
- The rubric is near its ceiling for the AI, so it will not show further gains. The next measures are the fabrication rate by labelling, and the simulated students' outcomes.

**Spend:** $2.02 of the $2.50 Tai approved. Coach $0.24, judge $0.81, rubric $0.69, then regrinds, rubric fills and the vote trial.

## The locked test set and the false-claim audit (2026-09-26)

Tai's finish line (2026-09-26), with a hard cap of $3 for the rest of the project:
- E1: false chess claims in at most 1 AI reply in 50;
- E2: AI excellence at least the template's + 0.10 on the locked TEST set;
- E3: at least 90% of the player's questions answered;
- E4: his real-game bugs closed.

**What changed before the run.**
- The model sees everything (engine lines, legal moves, facts past the rung). The leak check only logs.
- Lines are given move by move, with what each move does (`effects.line_notes`).
- The draft check covers forcing and open-file claims.
- The real-game bugs are fixed: their threat on their turn, "your" on the opponent's turn, 35...Nxf5's fork, first hints of 7 to 10 sentences.

**The run.** `bench-test-final`: 300 moments, 298 answered by the model, for $0.41.
- The rubric ran on its own (`rubric.py --unjudged`) for $1.46.
- The parity judge was skipped: it would have broken the cap, and it cannot see false claims (0 of 4 above).

| | AI | Template |
|---|---|---|
| excellence (298 / 224) | 0.99 | 0.86 |
| paired (224): higher / lower / same | 94 / 2 / 128 | |

**E2 met (+0.13).**

**The audit ($0).** Four labellers (Claude, python-chess) read all 298 replies. Every flag was then checked by hand against the board and the harness: 11 of them first, the other 19 only after Tai asked. This section first said all had been, which was not yet true.
- Questions answered: 165 of 166 (**E3 met**).
- False claims: 27 of 298 replies (9.1%) confirmed, plus 1 borderline (misleading advice, R220).
  - Two flags were rejected: an echo of the sheet, and "Excellent" of a great move, which reads as praise.
  - 5 of the 27 came from the note wording "only a king move gets out of check" (00894e0), read as "Kh8 is the only king move". The note lists the king moves since 4365dfa.
  - 2 were started by our own sheet: a line outcome "wins two pawns" that meant the exchange's value, and a motif "attacked through another piece" of a direct attack.
- The families:
  - sides mixed up in lines ("advancing b4" of their pawn, "your queen checks" of theirs);
  - "only move" and "forces the King to h6" where other moves were legal;
  - a fact true before a move said after it (a target on d6 that recaptures; the material before a recapture);
  - relations to moves that cannot follow ("Ne4 allows exf6");
  - "loose" of defended pieces;
  - purposes the board denies.

**Fixes, and a dev check.**
- Lines now name the mover and list the king's moves.
- Fact questions (the material, the opening by name, the standing) get the template, which is exact.
- The draft check reads straight and curly quotes, adjectives, named checks' frames, "only king move", "allows", and state-before-piece. It catches 8 of the 30 flagged replies, and 18 new true flags across 9,832 stored replies.
- A dev run (`bench-dev-final`, $0.30) was audited the same way, and all 14 flags were checked by hand:
  - 10 of 129 (7.8%) are the model's own false claims;
  - 2 more came from our sheet: the material given before a recapture without the recapture, and a "passed pawn" motif of a pawn that is not passed;
  - 2 are borderline.
  - No measurable change from the source fixes.
- By class, over the 365 replies audited:

| class | replies with a false claim |
|---|---|
| ask_line | 35% |
| ask_move | 24% |
| ask_last_move | 19% |
| verdict (great moves) | 18% |
| lesson | 11% |
| hints | 5-14% |
| ask_threat, ask_game_so_far, ask_named_move, fact asks | 0% |

- The model's explanations of concrete sequences are where it is wrong.

**Decision (d8c687c).** A bare "which move?" or "show me the line" gets the exact template: the line said move by move with what each does (`effects.line_spoken`), and no model call. There the rubric scored the templates as high as the model (1.00), so by Tai's rule the AI speaks only where it adds to the offline text. Counted on the hand-checked test flags:
- the 5 note-caused errors are gone (4365dfa);
- 7 of the other 22 were answers to a bare move or line, now exact;
- that leaves 15 of 298, about 5%. This is an estimate, not a new run.

**E1 is not met.** The rest are in the AI's core (lessons, hints, why-answers, great-move verdicts). There its explanations are the excellence and the errors at once.

**Tai: "can you fix it? To make it as low as 1%?" (18:17).**

- **Step 1 ($0, 70931ab).** It fixed the four sheet bugs:
  - the outcome now names the pieces ("the exchange: a rook for a bishop");
  - an advanced pawn is no longer called passed;
  - the x-ray wording is fixed;
  - the recapture comes from the engine's line when the new position has not been read yet.

  Great and brilliant moves also get the board's sentence (Tai's rule of 2026-09-20; the model's delight carried a false claim in 18% of them).
  - Estimate: about 2% on test.
  - About 7% on dev, whose errors (lessons, hints, plans) no fix was designed from. Dev is the honest number.
- **Step 2 (1402650, 232fc11): a model lists the draft's claims, python-chess decides them** (`coach/draft_claims.py`). A claim is decided only on what its own words say.
  - Validation, over the 37 hand-confirmed false replies and 99 clean ones:
    - luna@none as the reader caught 11/37 (30%) and refused 0/99 clean;
    - the local Qwen 27B caught 5/37 and refused 0/99.
  - The looser verifier caught up to 20/37 but refused 26/99.
  - Luna adds a median 1.3 s per reply (p90 2.5 s) and about $0.0004.
  - It is off (`Speaker.claim_reader`): short of the half this was to reach, and 1% was not reached. On at 30%, the estimate is dev about 5%, test about 1.4%.

**Spend:** $2.30 of the $3 cap. Pilot $0.03, test coach $0.41, rubric $1.46, dev coach $0.30, claim-reader validation $0.10.

## Next, if wanted

Scale the exemplars to 30-50 from the already-judged training moments: round 1's 400
judged moments are kept with their harnesses (`icr-100/items-r1`, ~40 fails to draw
from; the later rounds' play-throughs were deleted, so their moments would need a replay
to the ply, ~$0.1 a game); one writer call per exemplar, re-distil once, one held-out pass
(~$0.4 judge-only on cached replies). Completeness is the harness's to raise, not C's.
