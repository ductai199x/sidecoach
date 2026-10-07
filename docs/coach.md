# The coach

What the coach says, when, from which facts, and how a reply is checked before the player
hears it. The code is `src/sidecoach/coach` and the model behind it `src/sidecoach/llm`.
[DESIGN.md](../DESIGN.md) section 5.5 holds the design history; this document is the coach as built. How the
coach's words are evaluated and trained is in [eval-results.md](eval-results.md); how moves are
labelled is in [move-classification.md](move-classification.md).

## The modules

| Module | What it does |
|---|---|
| `coach.py` | `Coach` (events in, `CoachMessage`s out; the ladder; the triggers' directives and templates) and `Speaker` (the model calls on a worker thread, the draft checks, the fallback) |
| `policy.py` | `InterjectionPolicy`: which unprompted trigger fires, and when |
| `disclosure.py` | the per-position ladder (`Disclosure`), `leak_check`, `said_level` |
| `conversation.py` | the per-game harness the model reads |
| `prompt.py` | the system message: persona, contract, rules, C |
| `situation.py`, `story.py` | the first hint: the stakes, the key fact, the trap, told as one story of the best line |
| `idea.py`, `idea_words.py` | what the best line is for (goal, pattern, target), in words per rung |
| `plans.py` | the plan in the player's next four moves of the best line |
| `facts.py`, `effects.py`, `inventory.py`, `outcome.py` | the facts from the analysis: what a move does, what stands on the board, where a line leads |
| `alternatives.py` | the moves about as good as the best, and what goes wrong with the worse |
| `verdicts.py` | the word on a sound move, written from the board without a model |
| `named.py`, `engine_tools.py` | the engine on what the player names; the engine as tools for the model |
| `claims.py`, `draft_claims.py` | claim checks both ways: the player's stated reasons, and the coach's draft (`check_draft`) |
| `annotations.py` | arrows and highlights that go with a message |
| `summarise.py` | the earlier dialogue folded into a summary |
| `bag.py` | wordings drawn without replacement |
| `context/corrections.md` | C, the lessons from earlier games in the system message |
| `review_words.py`, `review_links.py` | Game Review's AI words ([game-review.md](game-review.md)) |

## When the coach speaks

`InterjectionPolicy` (`policy.py`) decides every unprompted message:

1. **Asked**: always answered, at any time and in any time control, quiet mode included.
2. **After the player's move**: every move gets a word. A mistake, a blunder or a miss gets a
   lesson (`Trigger.POST_MISTAKE`); every other move a short verdict (`Trigger.POST_MOVE`). The
   lesson waits up to `LESSON_WAIT_S` (1.5 s) for the analysis of the position the move left,
   so it can say what the move allowed.
3. **Stuck**: after `coach.stuck_seconds` (default 45) on a move, the first hint; after another
   full spell, the second; never the move unasked (`UNPROMPTED_PER_POSITION` = 2). With a clock
   the wait is the smaller of `stuck_seconds` and 8% of the time left, never under 20 s; under
   30 s on the clock no stuck hint at all. A question answered restarts the spell
   (`policy.asked`).

Nothing else while the player thinks: no question at a decisive moment, in a game or a puzzle
(Tai, 2026-10-01: nothing "until the user is stuck or actively request a hint or after they
have made a move"). `Trigger.DECISIVE_QUESTION` is still in `events.py` for the evaluation's
recorded requests but never fires live. No message is prompted by the opponent's move either.

Time controls (`time_control_of`, chess.com's bands): untimed and rapid get every trigger;
blitz gets the post-move word only (a mistake gets the verdict, not the lesson, and no stuck
hints); bullet gets nothing unasked. When the player moves faster than `FAST_PACE_S` (6 s a
move, over the last moves), the coach speaks from templates instead of the model until they
slow down; a question and the end-of-game review always go to the model. `coach.quiet` turns
off everything but answers. The board covered by a dialog, the opponent to move, or a side to
move not yet known also hold the stuck hint.

## The hint ladder

Each position the player faces has a `Disclosure` (`disclosure.py`) with five levels: SILENT,
IDEA, FOCUS, MOVE, LINE. Two hints, then the answer (Tai, 2026-09-23):

| Level | What it gives | Reached by |
|---|---|---|
| IDEA | a short lesson on the position (`situation.py`): the stakes, the key fact named on the board, the principle after the fact, the trap if there is one | the first stuck spell, or the first hint |
| FOCUS | the target where it stands, the piece to use, the question from there to the move | the second stuck spell (the ceiling unasked), or the second hint |
| MOVE | the move, played out and explained | the third hint, asking for the move, or naming it |
| LINE | the whole line | asking for the line, or after the move is played |

Each level is said once per combination, and never a lower one after a higher (Tai,
2026-09-29). `Coach._ladder` walks back through the player's positions while the move they
played was the engine's best and the position continues the same combination
(`_same_combination`: a forcing line the engine's line there continues, or the same tactical
theme at both); the highest level said there is carried (`Disclosure.carry`). A plan is not a
combination: each quiet position gets its own hints (Tai, 2026-10-02). A takeback keeps what
was said before the withdrawn move (`withdraw_move`). A game that is over withholds nothing.

How a question moves the ladder depends on the model:

- **Offline**, the words decide (`unlock_for`): a hint request climbs one rung, a request for
  the move or the line opens it, a move named opens MOVE.
- **With the AI on**, the model reads the question. The move or the line asked for outright --
  the Best move and Line buttons, or the same words typed or said -- is opened exactly
  (`asked_outright`; Tai, 2026-10-03: direct requests honoured exactly). Anything else is
  asked at the next rung (`_next_rung`), and once the reply is in, `said_level` reads which rung
  it actually carried (the line's first two moves, a candidate, the focus squares or piece, the
  idea's facts) and the ladder climbs to that (`Coach._climb`, `Disclosure.said`). A reply that
  only answers "am I winning?" moves nothing (Tai, 2026-10-02).

## The AI coach and the offline templates

`llm.provider` chooses. With a model, every question goes to it, the buttons' included (Tai,
2026-09-28: "I'm using AI coach so I better be able to ask whatever I want and get answers").
With `llm.provider = "none"` (`llm/offline.py`, `OfflineProvider`) no call is made and the
templates are the whole coach: free, offline, and true by construction.

Templates exist for every trigger: the rungs (`idea_template`, `focus_template`,
`move_template`, `line_template`), the lesson (`lesson_template`), the standing, the material,
the opening, the review's sentences, and the engine's word on a named move (`named.py`). They
are also the model's floor: a reply that fails its checks twice is replaced by the request's
template (below).

Some words never need a model. A best, excellent or book move gets its sentence from the board
(`verdicts.py`): a word of praise, then the mate, the tactic it made, the threat it sets or the
idea behind it, only when the board bears it out (`coach.local_verdicts`, default on; always
in blitz). A good move goes to the model, which names the better one; a great or brilliant move
goes to the model at full effort, told to show some delight, the board's sentence standing in
if the reply fails (Tai, 2026-09-26).

A question about the player over their games ("my weaknesses", "over the last games") is
answered by `profile/ask.py`: the model writes read-only SQL over the profile record
(DESIGN.md section 5.6).

## What the model sees

Since 2026-09-26 (Tai's decision) the model sees everything: the engine's lines at every level,
the legal moves, and the facts past the current rung, set apart under "Beyond this level: the
answer and how it works" (`conversation.BEYOND_TITLE`). The Task states the rung and what the
reply may give (`Coach._level_line`). The docstring at the top of `disclosure.py`, "hiding is by
construction", predates this.

Every draft is checked before it is shown (`Speaker`):

- **Refused** (`Speaker._problems`): what it gets wrong. A piece placed on a square it is not on,
  a move that is not playable, a claim the board contradicts (`claims.check_draft`, wired as
  `draft_check` in `cli.py`); "illegal" said of a legal move; the verdict word or cost for the
  move just played against the assessment; an earlier moment of the game it invents. A refused
  draft is sent back once with the problems named; a second failure is replaced by the
  template, so the player never hears a false claim and never hears nothing. `draft_claims.py`
  (a model lists the draft's claims, python-chess decides them) is built but not wired: the app
  passes no `claim_reader`, so it stays off (Tai, 2026-09-26).
- **Logged** (`Speaker._leaks`, from `disclosure.leak_check`): what the reply gives away of the
  answer the rung withholds -- the move to find, its square, the piece that makes it. A
  debugging signal, never a reason to refuse (Tai, 2026-09-26): an occasional hint that says
  too much is minor; a vague coach is not.
- **Logged** (`Speaker._notes`): form -- too long, a missing or unwanted closing question, a
  restatement, a figure nobody asked for, the engine named.

A second model can read every candidate against the facts before one is spoken
(`coach.verify`: empty is off, `same` uses the coach's model, a provider name another; the
`Speaker.verifier`). With voice streaming on and no verifier, the first candidate is spoken
sentence by sentence as it arrives, each sentence checked before it is spoken; once anything
has been spoken no template follows, and what was wrong with the rest is kept in the log. A
stream whose first sentence fails is read whole and checked as without a voice.

With `coach.verbalized_sampling` on (default off) the model writes two to four phrasings, each
with its likelihood (`vs_directive`), every one is checked, and a passing one is sampled by its
likelihood (Verbalized Sampling, arXiv 2510.01171).

## The conversation, one per game

`conversation.Conversation` is the game as the model reads it: one `user` message in sections
(`messages()`), rebuilt for every request, then the request's Task. The coach answers as the
assistant.

- **# Setting**: whom the coach coaches (the side, the pieces at the bottom), the time control,
  a puzzle or a position joined from the middle with its start, and what earlier games say
  about the player (`Coach._profile_lines`, computed by the profile stage).
- **# Game so far**: one line per move with the engine's verdict (label, cost, the move
  preferred, where its line leads), and under each move what was said after it -- `PLAYER
  said:`, `YOU (the coach) said:` -- and every event (`EVENT:` a takeback, a re-read of the board,
  a dialog over the board, the end of the game), so the position never changes silently. A
  move's line is frozen once the model has seen it; a fact that lands later goes in as a note.
- **# Position now**: the last position in full -- the board, the last move, the standing, the
  engine's best three lines and three weaker ones (eight plies each), the facts, the complete
  inventory of what stands on the board, the player's record by label, and what was already
  said here.
- **# The position they faced** with a lesson, and **# Earlier in this game** when earlier moves
  were played with the same idea (`same_idea`).

The player's own words are kept verbatim, with the board's verdicts on any claim in them
(`claims.py`, direction one: the model only extracts the claims; python-chess decides them).

Two levers keep a long game small, both off by default: `coach.summarise_every` folds the
dialogue older than the last 20 plies into one summary line (`summarise.py`; the summariser
sees only the dialogue, never the position, so it can add no fact), and `coach.compact_record`
writes sound moves nobody spoke about as a plain move list.

## Engine tools for questions

For a question of the player's, the model can ask the engine itself (`engine_tools.py`, Tai's
puzzle 2026-09-28): `analyse` (the position after any moves from the board now or before the
last move: who stands better, the best lines with what each move does, what hangs; how a
proposed first move rates against the best) and `square` (what stands on a square, what attacks
it, what can go there). Up to `TOOL_ROUNDS` (3) rounds, then it answers with what it has. The
boards the tools visited are kept, and the draft check reads the reply against them. Tools are
given only for questions, only with an engine and a model that takes them (`tools_ok`).

Before the model is called, the moves, pieces and squares the words name are examined by the
engine (`named.py`) and given with the request. Offline, that examination in plain words is the
answer.

## Tone

The rules are in the system message (`prompt.RULES`), the same for every persona:

- Gentle and welcoming: a mistake is named plainly, as something to learn from, never with
  blame; never count how often something went wrong, never "again" or "three times now", never
  sound disappointed -- say what to look for next time (Tai, 2026-10-02).
- Describe the position more than the player ("the knight on f3 was loose"); "you missed", "you
  should have" now and then, varied.
- Never a question about the player -- what they expected, saw or meant, why they played it,
  whether they noticed: after a mistake it reads as a challenge (Tai, 2026-10-02).
- A question, where the Task offers one, is one short question about this position, for
  something the reply has not said.
- No two replies opening the same way; never restate the previous turn.
- Plain spoken text, standard chess words, no hedging, no markdown; never mention the
  engine, the facts or the rules.

The templates vary their wording through `bag.Bags`: every wording of a template once before any
comes again, and a fresh bag never opens with the one just said (Tai, 2026-09-25). A rule of
thumb is held to a budget (`RULE_SPACING` 10 messages between any two, `RULE_COOLDOWN` 200
before the same one again).

The persona (`coach.persona`) changes only the manner: `warm_direct` (the one evaluated),
`calm_quiet`, `sharp_demanding`, `playful` (`prompt.PERSONAS`). The rating band in the settings
(`profile.rating_low`, `rating_high`) is stated in the system message as the level to pitch at,
and the templates drop terms of art below 1400 (`PLAIN_BELOW`) and textbook rules from 1800
(`STRONG_FROM`).

## The system message

`prompt.system_message`: four parts. The persona; the contract (how the game reaches the model,
what the sections and labels mean, that the course of the game lives only in the record); the
rules; and C, the lessons from earlier games read from `context/corrections.md` (one line per
situation, general, with a placeholder example; how it was made and measured is in
[eval-results.md](eval-results.md)). The reply format (one reply, or the verbalized-sampling
phrasings) is appended to the Task, not the system message.

## The provider

`llm/factory.py` `make_provider` builds the model from `[llm]` in the settings. Every provider
is OpenAI-compatible (`llm/openai_compat.py`), so any service can be swapped in:

| `llm.provider` | Endpoint | Default model | Reasoning switch |
|---|---|---|---|
| `openai` (default) | api.openai.com, the Responses API (its reasoning models take tools with an effort only there) | gpt-5.6-luna | effort |
| `local` | LM Studio at `http://localhost:1234/v1`, or `llm.base_url` | the loaded model | an empty think-block prefill |
| `dashscope` | Alibaba Cloud Model Studio, US region (`DASHSCOPE_BASE_URL` overrides) | qwen3.8-max | a request parameter |
| any other name | `llm.base_url` and `llm.model` | -- | none |
| `none` | no model (`OfflineProvider`) | -- | -- |

Keys come from the environment (`OPENAI_API_KEY`, `DASHSCOPE_API_KEY`, `CUSTOM_API_KEY`), never
from the settings file. `llm.reasoning_effort` defaults to `none` (Tai, 2026-10-04: latency
first) and `llm.routine_effort` (a word on a fine move) to `none`; the coach was evaluated at
medium. Game Review's words always reason at medium ([game-review.md](game-review.md)).
`llm.timeout_s` (60 s) bounds a call; past it the coach speaks from the board. `llm/usage.py`
counts tokens per endpoint and, with the prices set, the session's cost. `provider_from_spec`
gives the tools a short form (`openai:gpt-5.6-luna@medium`, `flash`). Paid calls in
`tools/eval/` go through `judge.Ledger`.

The phone's coach is a 1:1 port (`com.sidecoach.coach` in sidecoach-android).
