# The coach: the game as one conversation, evaluated, and trained through its context

Status: proposal for review (2026-09-16). Supersedes the sliding-window transcript and the
"Game so far" memory line in `coach/coach.py`. Once agreed, the settled parts move into
DESIGN.md.

## 1. Why

Today every coach message is produced from a snapshot: the fact sheet of one position, a
sliding window of the last 16 synthetic turns, and one line of memory. The model never sees
the game as moves. Asked "how did I do overall?", it has nothing to answer from and invents a
story (it told a player who won a queen that they had been a queen down). The rule of this
project -- facts are enforced by construction and by checks, never by prompt instructions
alone -- has no history to check against, because there is no history in the prompt.

The fix is the design that was asked for on 2026-09-15 and never built: the game *is* the
conversation. And the coach's replies are evaluated against ground truth and improved by
Iterative Context Refinement (Nguyen & Stamm, arXiv 2606.23917, section 5), not by hand-tuned
prompt wording.

## 2. The LLM interface: one conversation per game

### 2.1 Shape

- One chat conversation per game. `system` = the persona, the rules, the format contract and
  the ICR context C (section 7). Every move on the board -- the user's and the opponent's --
  is a `user` message. The coach is an `assistant` turn only when prompted.
- The coach is prompted (one API request, the whole conversation) when: the user has just
  moved (the verdict); a decisive moment is detected before the user moves; the user has been
  stuck; the opponent's move swung the game (a blunder, a big eval change); the user asks
  anything; the game ends; and, after the game, any question. Nothing else makes a call.
- The request is: `system` + every message so far + the last `user` message, which carries the
  directive for this moment (what kind of reply, length, the verbalized-sampling format). The
  chosen reply is stored as the `assistant` turn. Templated verdicts, if any survive (see
  2.5), are stored as `assistant` turns too, so the model knows what the coach already said.
- No sliding window. No memory line. No compaction unless measured to be necessary
  (section 2.6).

### 2.2 The move message

Built from the game state at the time the coach could be called, from the user's side,
always. Components, in order:

1. **Board state.** The PGN so far, then the ASCII board with coordinates, then whose turn.
   On a takeback the message opens `Takeback: 30. Bd4 was withdrawn; the game is back at the
   position after 29...Nd5.` and the PGN and board are the corrected ones. On a resync:
   `The board was re-read; the position is now as shown; moves may have gone unseen.` On a
   cover: `The board is covered by a dialog.` The model is told when the position jumps and
   why; it is never shown a silent switch.
2. **The move** just made, who made it, the think time, the clocks.
3. **The engine's verdict on the move and its effect**: the label (best / excellent / good /
   book / inaccuracy / mistake / blunder / miss / great / brilliant), the expected score
   before and after and the loss in points, in plain words, and the material balance from the
   user's side. For the opponent's move the same, so "they just blundered" is on the record.
4. **The engine's lines**: the best N (N = 3) in SAN with expected scores, then the worse
   alternatives including the move played, ranked, with the played move's rank stated
   ("f3 ranked 4th, 0.11 behind Nxe7"). For the position the user now faces, the lines and
   candidates are subject to disclosure (section 2.4); for a move already played, always given.
5. **Everything else that grounds a reply**: the motif tags with their squares (fork g4/e8, a
   pin on d1), the strategic features (development, files, outposts, passers, majorities,
   space, king safety, the endgame patterns), the opponent's threats for the side to move, the
   opening name while in book, the phase, decisive / only-move flags, and a running tally of
   the user's moves by label with the moves where ground was lost -- so "how did I do" has its
   answer in-context, computed by code.

A move message is written when the move is seen and filled in when the engine's assessment
lands (about a second). It is frozen the first time it is sent. Facts that arrive after
freezing go in a later engine-note message, never as an edit to history.

### 2.3 Other user messages

- New game: side, opponent (bot or human when known), time control.
- The user's own words, verbatim, when they ask something; with the deterministic verdicts on
  any claim they made ("their claims, checked against the board: ...").
- Game over: the result and how (checkmate, resignation, time, the dialog).

### 2.4 Disclosure, unchanged in substance

The disclosure machine keeps deciding what may be said about the position the user faces:
QUESTION, THEME, REGION, PIECE, CANDIDATE, LINE, unlocked by the stuck timer, decisive moments,
hints, naming a move, playing a move. The move message for the position the user faces carries
only what the current level allows; the prompted turn's directive states the level. Everything
about a move already played is at LINE. After the game, everything is at LINE. The leak check
on the reply stays as the second line of defence.

### 2.5 Prompting after every move

The design calls for a call after every user move. The measured cost decides how it is done:
the local 27B answers in 2-4 s with a warm prefix cache and the verdict for a routine move is
five words. If the measured latency in a long game stays under ~3 s, every user move prompts
the model with a strict-length directive. If it does not, routine moves (loss under the
inaccuracy line and the move was the engine's own) keep the templated verdict and only
non-routine moves prompt. Decided by measurement in step 2 of section 9, not by assumption.

### 2.6 Context and latency

- A 40-move game at ~120 tokens a move message plus replies is roughly 15k tokens; a 60-move
  game 25k. The context window is not a constraint at 250k. The loaded model's context length
  is read from LM Studio at startup and a warning is logged under 32k.
- Latency depends on prefix caching: LM Studio reuses the KV cache for an unchanged prefix, so
  only the new tail is processed. Measured in step 2 (two calls, one added message, compare
  prompt-processing time). Compaction (summarising the opening phase into one message once the
  game is past move 20) is designed but built only if that measurement says so.

### 2.7 Seeing it

The panel's Conversation tab shows the exact messages: every move message, every ask, every
coach turn, and for each call what the model was given and what it wrote.

## 3. The system message

1. Persona: a coach beside the player; warm, direct; adapts to the player's level (the level
   is stated in the new-game message: 1000-1400 today, up to 2400 later).
2. Hard rules: every claim traces to a fact in the messages; nothing above the disclosure
   level stated in the prompted turn; never name a move below CANDIDATE; plain words; the
   length bounds per kind of reply; a question at the end of a hint or a lesson.
3. The format contract: "each move arrives as a message with the engine's facts; you speak
   only when the last message asks; otherwise you are not called; the takeback/resync
   messages mean the position changed as they say."
4. The reply schema: the verbalized-sampling format (k candidates, each with a likelihood),
   which the speaker parses, checks and samples from as today.
5. **C**: the corrective demonstrations produced by ICR (section 7). Test-time training lives
   here, in the system message, and nowhere else.

## 4. What can be asked

The prompt space, with what each reply must contain (the "required facts", scored by the
judge) and where the ground truth comes from. All of it is in the facts we already compute.

### 4.1 Coach-initiated moments

| Moment | Required facts | Hard constraints |
|---|---|---|
| Verdict on the user's move | the label; that a better move existed if it did | <= 2 sentences; no lecture |
| Lesson after a mistake | the cost in plain words, the better move, its idea/theme, one check question | the idea must be a fact; ends with a question |
| Decisive-moment question | that the moment is critical; one orienting question | no square, piece or move; no evaluation |
| Stuck hint, levels 1-4 | exactly the fact the level allows | nothing above the level |
| The opponent's swing | that the balance changed and in whose favour | no candidate named |
| Game over | the result and how it ended | one sentence true to the record |

### 4.2 User questions

| Class | Examples | Required facts | Ground truth |
|---|---|---|---|
| The position now | what should I play; hint; the line; the plan; am I winning; what is he threatening; is my knight safe; can I take on e5; should I trade | the fact at the unlocked level, one reason | engine lines; threats; tags; positional features |
| A named move (a claim) | is Nf3 good; why not Bxf7; what if I castle | a verdict on that move (its rank and loss against the best), the reason | the engine's ranking of the named move |
| The last move | why was that a mistake; what was better; what did I miss | the cost, the better move, the idea | the assessment and tags |
| The opponent's move | why did he play that; what does it threaten; was it a blunder | the threat; the label | analysis of the reply |
| The game so far | how did I do; where did I lose it; what do I keep missing; what opening was that; my accuracy | the tally by label; the worst moves with move numbers; the material and eval arc; recurring themes; the opening | the move record |
| A master's move (study) | why did Fischer play this; what is the plan here | the master's idea (as our concept vocabulary names it), the engine's agreement or not | the annotated game (section 6.2) |
| Concepts | what is a fork; how do I convert a rook ending | a correct definition, tied to the game when possible | none yet -- held out of the metric until a curriculum exists |
| Control | repeat; slower; quiet; flip | an action, not a reply | not the model |

Each class gets paraphrases and spoken forms (STT is a source), and for claim questions the
named move is sampled from the engine ranking (best / near-best / bad) so the verdict is known.

## 5. Scoring

Per prompted item: `{the messages the model saw, the level, the class, the required facts,
the evidence, the reply}`. The judge returns:

- **Correctness** (Eq. 4 of the paper): of the facts the reply states, the share stated right --
  the move, square, piece, label, number, who is up what.
- **Completeness**: the share of the required facts the reply contains.
- **Fabrication** (hard fail): any claim with no supporting fact -- a pin that is not there, a
  queen never lost, a habit the record does not show.
- **Leak** (hard fail): anything above the level. Deterministic: `leak_check` and the board
  claim checker run first and their verdicts stand; the judge scores the rest.
- **Relevance**: the reply addresses the class asked (a "why not Bxf7" that ignores Bxf7 = 0).
- **Form**: length bounds; the closing question where required; no move names below
  CANDIDATE; no stump; the repetition rate across a game (the same phrasing twice running).
- **Pedagogy** (does the hint help without telling): human, by playing; not in the automatic
  metric.

Aggregated per class and per source corpus, since failure modes differ by class (expected:
fabrication in "game so far", omission in lessons, leaks in hints).

## 6. The evaluation set

Three corpora, one item schema, a held-out split (20% of each corpus plus one whole recorded
game) never used to build C.

### 6.1 The recordings

Every recorded game (eight today, more with every session): every ply is a verdict moment,
every mistake a lesson, every user-to-move position a possible stuck / decisive / ask moment;
the questions actually asked live are kept verbatim. Ground truth from the engine, the labels,
the tags and the record. Driven by `tools/eval/coach_games.py`, which already coaches a game
from its PGN with the real engine and tagger.

### 6.2 The master games

The annotated collections in `refs/` (Nunn, *Understanding Chess Move by Move*; Fischer, *My
60 Memorable Games*; the *Mammoth Book*) -- read as reference only, never copied. Their
converted text keeps every move as a heading with the move number and the destination square
intact (`### 3 tZ:lc3`, `### 4 ctJfd7`) followed by the master's explanation. The moves come
from pgnmentor's per-player PGNs (the OCR mangles piece letters); each heading is aligned to
its ply by move number + colour + destination square and verified against the game.
Nunn annotates every move of ~30 games; the other two are sparser: 4,602 plies carry a
readable note once the scan's table rows, captions, dangling move numbers and lost
variations are cut and its piece stand-ins restored where a tally shows them reliable.
The master's prose is the *judge's* reference for idea completeness; it enters no prompt, no
repository file, and no C.

A master game takes exactly the path a bot game takes (2026-09-16, Tai: the player's rating
never matters, here or anywhere): the same triggers, the same question rotation, the coach on
White's side. The one difference is that at an annotated ply the items *about that move*
(verdict, verdict_weaker, lesson, ask_last_move, ask_opponent_move) carry the note as
`reference`; 2,698 items do. For those the judge states the note's core point in one sentence
and answers two yes-or-no questions: `idea_match` (the reply makes it) and `idea_in_facts`
(the coach's evidence states it, with `idea_where`). In ICR a point missed *and* in the facts
is an omission the coach answers for (confirmed two-of-two, then a demonstration written from
the facts alone -- the writer never sees the note); a point missed and absent from the facts
is logged per round as a facts gap, the input to the facts layer, not to C.

### 6.3 The puzzles

Lichess puzzles with ground-truth themes and solutions (the set the tagger is validated on):
tactical variety, and the "what did I miss" class with a known answer.

### 6.4 Item schema

```
{source, game, ply, fen, level, class, question,
 messages: [...the conversation up to and including the prompted turn...],
 required_facts: [...],
 evidence: {lines, assessment, tags, features, record},
 reference: {master_annotation_id | puzzle_id | null}}
```

The messages are rendered by the same code the live coach uses (section 2), so the eval
measures the real prompt.

## 7. Iterative Context Refinement for the coach

The chess version of section 5 of the paper:

- **Evidence** = the facts the move messages carry, so every claim is checkable against
  something explicit, as each FKG triplet is.
- **Judge** = an LM given the evidence, the required facts, the level and the reply, returning
  the scores of section 5 as JSON. Default: DashScope `qwen3.8-max` (key set); fallback: the
  local model with thinking on. Calibrated once against ~30 items scored by hand.
- **Rounds**: start with C empty; run the coach over the training split; judge; take the
  worst fabrications and omissions per class; pair each with its errors as a corrective
  demonstration (the messages the model saw -> what it wrote -> what was wrong -> the
  corrected reply, written by the judge model and checked deterministically); append to C;
  prepend C to the system message; repeat until the average change in correctness and
  completeness over the training split is under epsilon = 0.01, or C reaches its cap (~50, with
  per-class quotas so a class the model already gets right cannot crowd out one it gets
  wrong).
- **Artifact**: C written to `src/sidecoach/coach/context/corrections.md`, versioned, loaded
  into the system message at startup. Re-run when the message format, the facts, or the model
  change.
- **Measurement**: before/after on the held-out split, per class; and the live question that
  started this ("how did I do overall this game?") on this morning's game.

## 8. Layout

```
src/sidecoach/coach/conversation.py   the per-game conversation and the move message builder
src/sidecoach/coach/context/corrections.md   C (shipped)
tools/eval/masters.py                 book headings -> aligned plies -> items
tools/eval/corpus.py                  items from the three corpora, the split
tools/eval/judge.py                   deterministic checks + the LM judge
tools/eval/icr.py                     the round loop, C, convergence
tools/eval/report.py                  per-class / per-corpus scores, before/after
data/eval/                            items, judgments, rounds (gitignored)
```

## 9. Order of work, with what "done" means

1. **The conversation** (`conversation.py`, the coach rewired, the transcript and memory line
   removed). Done when: a synthetic game with a takeback and a resync renders one message per
   ply with components 1-5, the disclosure of current-position lines is obeyed, asks are the
   user's words, replies are assistant turns; the Conversation tab shows it; all three live
   recordings replay with the real model and the reply to "how did I do overall this game?" is
   read against the PGN.
2. **Measure** context length and prefix caching on LM Studio; decide 2.5 and 2.6 from the
   numbers.
3. **The system message** rewrite with the format contract and the C slot (empty).
4. **The corpus**: `masters.py` (alignment verified game by game, a report of matched /
   unmatched headings), `corpus.py` over the three sources, the split.
5. **The taxonomy as a schema** (`class -> required facts`) and the question generator.
6. **The judge**, calibrated on the hand-scored items.
7. **A baseline report** on the held-out split before ICR.
8. **ICR** to convergence; ship C; the after report; the live check.
9. DESIGN.md updated with the settled design.

## 10. Decisions to take now

- Judge model: DashScope qwen3.8-max by default, local as fallback -- or Claude if a key is
  set.
- N lines in the move message: 3 best plus the worse ones down to the played move.
- The cap on C: ~50 demonstrations with per-class quotas.
- Prompt after every user move, or template routine verdicts: decided by the latency measured
  in step 2 (section 2.5).
- Concept questions: out of the automatic metric until a curriculum exists.
- Pedagogy: yours, by playing.

## 11. Status, 2026-09-16 afternoon

- Steps 1-3 done and committed (`19c2db2`, `2f59439`, `363140c`, `092fbd8`): the conversation,
  the measurements (262k context; 1.8 s per added message with the prefix cached), the system
  message with the C slot, every move of the player's a turn.
- Step 4 built (gitignored `tools/eval/`): `masters.py` (30 Nunn, 60 Fischer, 130 Mammoth games
  found in the text; located on pgnmentor by surname index + year; LCS alignment by move number,
  colour and destination under an OCR confusion map; 92% of Nunn's plies aligned), `facts.py` (the
  engine facts cached per ply), `corpus.py` (items through the real Coach with a capture),
  `split.py` (by game, 20% + this morning's game held out).
- Step 5 built: `taxonomy.py` -- 19 classes, required facts per class and level, question
  templates, the named-move sampler.
- Step 6 built: `judge.py` -- deterministic checks (leak, board, form) then the DashScope
  qwen3.8-max judge; every call on the ledger `data/eval/usage.jsonl` with a token budget and a
  dry-run estimate (measured ~770 tokens per judgment). `generate.py` answers items with the local
  model through the coach's own speaker. `report.py` aggregates per class and source.
- Step 8 written: `icr.py` -- the round loop over a fixed stratified training sample; the worst
  failures per class become demonstrations written by the judge model and checked like a coach
  draft; C appended, snapshotted per round; stops on epsilon, the cap, or the budget.
- Masters final (2026-09-16 night): 199 of 220 games, 14,811 plies, 4,602 with a readable
  note; the corpus 13,778 items over 213 games (8 processes, 84 s), 2,668 with a reference;
  split by game per book (masters 40 of 199 held out, recordings 3 of 14); every check in
  `validate.py` passes; a rebuilt game is byte-identical.
- Everything on DashScope qwen3.8-flash at the account's endpoint (`DASHSCOPE_BASE_URL`,
  commit f17b1f7): the coach, the claims check (only for a question naming a move; plain
  templates never carry a claim), the judge. Judge self-agreement on the idea questions over
  35 items: idea_match 94%, idea_in_facts 80%.

- ICR final (2026-09-17 morning): the paper's loop exactly (moment -> right reply, 5 a round,
  one per class), temperature 0.1, the object carrying computed move effects, the judge
  running the board-checker on disputed claims. Held-out (231 items, flash): C empty
  .52/.76/.79/.43 (pass/completeness/claim support/fabrication) -> 20 demos .58/.89/.86/.33
  -> + test-time verification .72/.89/.93/.19 (correctness .99). qwen3.8-max with the same C:
  .67/.89/.89/.30. C ships at 20 demonstrations (~22k tokens); `coach.verify = "dashscope"`
  turns verification on. Spend: ~700M flash + 26M max tokens lifetime, all on the ledger.
