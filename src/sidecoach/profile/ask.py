"""Ask the profile: the coach writes its own SQL over the record (DESIGN section 5.6).

A question about the player over time -- "why do I keep losing in the middlegame?",
"which openings do I score worst in?" -- is answered from the record, not from
memory. The model is given the schema in words (SCHEMA_DOC: the tables, what each
column means and its units, the keys that join them) and a set of worked queries, and
the question. It answers with one ```sql block when it needs data; the rows come back
as the next message; after at most ROUNDS such exchanges, or as soon as it writes prose
instead, the prose is the answer. Nothing but plain text passes either way, so any
OpenAI-compatible model can do this, and every query is read-only by construction
(Store.query: SQLite's authorizer refuses anything else before it runs). The queries
and their rows are kept with the answer, so what it was read from can be shown.
"""

from __future__ import annotations

import math
import queue
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from sidecoach.llm.base import ChatMessage, Completion, LLMProvider
from sidecoach.profile.store import QUERY_ROWS, QueryRefused, Store

ROUNDS = 4
MAX_TOKENS = 900
CELL_WIDTH = 60

SCHEMA_DOC = """\
The record is a SQLite database of the player's games.
Scores are expected score, 0 to 1 (1 = certain win, 0.5 = level), from the side each column
names; losses are in expected score too (a mistake costs 0.10 or more, a blunder 0.20 or
more, an inaccuracy 0.05 or more). Times are milliseconds. Tag lists are JSON arrays of
strings; read them with json_each(). Boolean columns are 0/1.

It holds two records of the games:
- reviewed_games and reviewed_moves: every game of the player's that has been reviewed --
  the games Sidecoach watched and the games imported from chess.com, Lichess or a PGN --
  each once. These are the games the Stats window counts: read them for anything counted
  over the player's games (how many, results, openings, phases, tactics, the clock).
- games, moves, evidence, dialogue and sessions: only the games Sidecoach watched live, with
  what only a watched game has -- why each mistake happened (evidence), what the coach and the
  player said (dialogue), the sittings (sessions). Rows of moves with withdrawn = 1 were taken
  back: leave them out unless the question is about takebacks.

VIEW reviewed_games -- one row per reviewed game: the games Stats counts
  key TEXT            the game's key; a watched game's is 'watched:' || games.game_id
  source TEXT         'watched', 'chesscom', 'lichess' or 'pgn'
  played_at REAL      unix time it was played, or NULL
  time_class TEXT     'bullet', 'blitz', 'rapid', 'classical' or 'daily', or NULL
  user_side TEXT      'white' or 'black': the side the player had
  bot INTEGER         1 if the opponent was a computer
  result TEXT         '1-0', '0-1', '1/2-1/2', or '*' or NULL when no result was kept
  score REAL          1 if the player won, 0.5 for a draw, 0 if they lost; NULL without a result
  opening TEXT        the opening's name from the book, e.g. "C50 Italian Game", or NULL
  termination TEXT    how it ended: the site's words for an import ('Game drawn by repetition',
                      'Someone won on time', 'Time forfeit', 'Normal', ...), Sidecoach's for a
                      watched game (as games.reason); match it with LIKE

VIEW reviewed_moves -- one row per move of a reviewed game, both sides
  key TEXT            reviewed_games.key
  ply INTEGER         1 for the game's first move, 2 the reply, ...
  side TEXT           'white' or 'black': who moved; by_user INTEGER 1 if the player did
  fen TEXT            the position the move was played from
  uci TEXT, san TEXT  the move
  phase TEXT          'opening', 'middlegame' or 'endgame'
  loss REAL           what the move cost its mover in expected score
  label TEXT          the verdict, as in moves.label; NULL if none
  accuracy REAL       the move's accuracy, 0 to 100, or NULL
  score_before REAL   the MOVER's expected score before the move
  chance TEXT         the tactic the mover had there, by kind: 'checkmate', 'fork', 'pin',
                      'discovery', 'trapped piece', 'removing the defender', 'free piece',
                      'sacrifice' or 'other'; NULL when there was none
  found INTEGER       1 if the mover had a chance and took it (the move cost under 0.02)
  allowed TEXT        the tactic the move handed the opponent, by the same kinds, or NULL
  structure TEXT      'open', 'semi-open' or 'closed': the pawns on the d- and e-files
  themes TEXT         the strategy themes there, comma-separated: 'space', 'activity',
                      'structure'; '' for none
  endgame TEXT        in the endgame, 'pawn', 'rook', 'queen' or 'other'; else NULL
  spent_ms INTEGER    time spent on the move; clock_ms INTEGER the mover's clock; NULL if unknown

TABLE games -- one row per game Sidecoach watched
  game_id TEXT        primary key
  session_id TEXT     the session it was played in (sessions.session_id)
  started_at REAL     unix time; ended_at REAL when it ended (NULL while in progress)
  user_side TEXT      'white' or 'black': the side the player had
  start_fen TEXT      the position the game was followed from (the start, or joined mid-game)
  time_control TEXT   '10 min', '3 min', '30 s' from the clock's first reading, or NULL
  opponent TEXT       always NULL: the opponent's name is not read
  result TEXT         '1-0', '0-1', '1/2-1/2', or '*' (ended without a result); NULL in progress
  reason TEXT         how it ended. From the board or the site: 'checkmate', 'stalemate',
                      'insufficient material', 'repetition', 'fifty-move rule',
                      'game over on board', 'resignation', 'lost on time'.
                      Ended by the player in Sidecoach: 'user resigned or lost on time',
                      'opponent resigned or lost on time', 'draw agreed',
                      'abandoned by the user'. With result '*': 'new game started',
                      'board re-read', 'board reoriented', 'the end was not seen'.
                      NULL in progress
  opening TEXT        the opening's name from the book, e.g. "C50 Italian Game"
  accuracy REAL       the player's accuracy for the game, 0 to 100 (mean of their moves')
  final_score REAL    1 if the player won, 0.5 for a draw, 0 if they lost, from the result
  pgn TEXT, final_fen TEXT

TABLE moves -- one row per move played, both sides
  move_id INTEGER     primary key; the key evidence and dialogue join on
  game_id TEXT        games.game_id
  segment INTEGER     0 normally; +1 after each position resync (the ply count restarts)
  ply INTEGER         1 for the first move of the segment, 2 the reply, ...
  move_number INTEGER the move number as written: 1 for 1. e4 and 1... e5
  san TEXT, uci TEXT  the move
  by_user INTEGER     1 if the player made it, 0 the opponent
  fen_before TEXT, fen_after TEXT
  phase TEXT          'opening', 'middlegame' or 'endgame' for the position it was played in
  think_ms INTEGER    time spent on the move, NULL if unknown; ts REAL unix time it was seen
  clock_ms INTEGER    the mover's clock when the move was made, NULL if no clock was read
  capture INTEGER     1 if the move captured; gives_check INTEGER 1 if it gave check
  material INTEGER    material balance after the move from the player's side, in pawns
  label TEXT          the verdict: 'book','brilliant','great','best','excellent','good',
                      'inaccuracy','mistake','miss','blunder'; NULL until the engine judged it
  severity TEXT       'none','inaccuracy','mistake','blunder' (how much the move cost)
  expected_loss REAL  what the move cost its mover in expected score
  score_after REAL    expected score after the move from the player's side, both sides' moves
  best_uci TEXT, best_san TEXT  the engine's move in that position
  accuracy REAL       the move's accuracy, 0 to 100
  tags TEXT           JSON list: what the move was/did ('capture','check','fork','sacrifice',...)
  available_tags TEXT JSON list: tactics the mover had on the board before the move
  allowed_tags TEXT   JSON list: tactics the move gave the opponent
  depth INTEGER       search depth of the verdict; withdrawn INTEGER 1 if taken back

TABLE evidence -- one row per mistake or blunder of the PLAYER's, with why
  move_id INTEGER     primary key, moves.move_id
  game_id TEXT, segment INTEGER, ply INTEGER   as in moves
  phase TEXT, expected_loss REAL               as in moves
  cause TEXT          'missed opponent threat', 'miscalculated line',
                      'never considered the candidate', 'positional misjudgment',
                      'opening knowledge gap', 'endgame technique', 'time pressure'
  motif TEXT          the tactic involved ('fork','pin','hangingPiece','mate',...) or NULL
  time_pressure INTEGER  1 if the cause is 'time pressure' (under ten seconds on the
                      clock, or the player said so)
  self_report TEXT    what the player said about it, if asked; corrected INTEGER 1 if the
                      player changed the cause

TABLE dialogue -- what was said
  id INTEGER, game_id TEXT, move_id INTEGER (the move it followed, or NULL), ts REAL,
  role TEXT ('coach' or 'user'), kind TEXT (the trigger or 'text'/'voice'), ply INTEGER,
  text TEXT

TABLE sessions -- one row per sitting
  session_id TEXT, started_at REAL, ended_at REAL, games INTEGER, summary TEXT
"""

EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "How many games have I played, and how do I score?",
        "SELECT COUNT(*) AS games, ROUND(AVG(g.score), 2) AS score,"
        " ROUND(AVG(a.accuracy), 1) AS accuracy FROM reviewed_games g LEFT JOIN"
        " (SELECT key, AVG(COALESCE(accuracy, 0)) AS accuracy FROM reviewed_moves"
        " WHERE by_user = 1 GROUP BY key) a ON a.key = g.key",
    ),
    (
        "Where do I lose the most, by phase?",
        "SELECT phase, COUNT(*) AS moves, ROUND(SUM(loss), 2) AS lost,"
        " SUM(label IN ('mistake','miss','blunder')) AS mistakes"
        " FROM reviewed_moves WHERE by_user = 1 GROUP BY phase ORDER BY lost DESC",
    ),
    (
        "Why do I make mistakes in the middlegame?",
        "SELECT cause, motif, COUNT(*) AS n, ROUND(SUM(expected_loss), 2) AS lost"
        " FROM evidence WHERE phase = 'middlegame' GROUP BY cause, motif ORDER BY lost DESC",
    ),
    (
        "How does my position go over a game: do I come out of the opening well and then slip?",
        "SELECT key, phase, ROUND(AVG(score_before), 2) AS score,"
        " MIN(score_before) AS low, MAX(score_before) AS high"
        " FROM reviewed_moves WHERE by_user = 1"
        " GROUP BY key, phase ORDER BY key, MIN(ply)",
    ),
    (
        "Do I play worse when short of time?",
        "SELECT CASE WHEN clock_ms < 30000 THEN 'under 30 s' ELSE '30 s or more' END AS clock,"
        " COUNT(*) AS moves, ROUND(AVG(loss), 3) AS loss_per_move,"
        " SUM(label IN ('mistake','miss','blunder')) AS mistakes"
        " FROM reviewed_moves WHERE by_user = 1 AND clock_ms IS NOT NULL GROUP BY clock",
    ),
    (
        "Which tactic do I miss most often, as a share of the times it was on the board?",
        "SELECT chance AS kind, COUNT(*) AS chances, COUNT(*) - SUM(found) AS misses,"
        " ROUND(1.0 * (COUNT(*) - SUM(found)) / COUNT(*), 2) AS miss_rate"
        " FROM reviewed_moves WHERE by_user = 1 AND chance IS NOT NULL"
        " GROUP BY chance HAVING chances >= 3 ORDER BY miss_rate DESC",
    ),
    (
        "Which openings go badly for me?",
        "SELECT opening, user_side, COUNT(*) AS games, ROUND(AVG(score), 2) AS score"
        " FROM reviewed_games WHERE score IS NOT NULL AND opening IS NOT NULL"
        " GROUP BY opening, user_side ORDER BY games DESC, score",
    ),
    (
        "When I get a clear advantage, do I convert it?",
        "WITH ahead AS (SELECT key, MAX(score_before) AS peak FROM reviewed_moves"
        " WHERE by_user = 1 GROUP BY key)"
        " SELECT g.key, g.result, ROUND(a.peak, 2) AS peak, g.score"
        " FROM ahead a JOIN reviewed_games g ON g.key = a.key"
        " WHERE a.peak >= 0.7 AND g.score IS NOT NULL ORDER BY g.played_at",
    ),
    (
        "What were my worst moments recently, and what was better?",
        "SELECT g.started_at, m.move_number, m.san, m.label, ROUND(m.expected_loss, 2) AS lost,"
        " m.best_san, e.cause, e.motif"
        " FROM evidence e JOIN moves m ON m.move_id = e.move_id"
        " JOIN games g ON g.game_id = e.game_id"
        " ORDER BY g.started_at DESC, e.expected_loss DESC LIMIT 10",
    ),
)

SYSTEM = (
    "You are a chess coach answering a player's question about their own games from a "
    "record of them. You can query the record in SQL (SQLite). Work like this: when you "
    "need data, reply with exactly one ```sql code block and nothing else; the rows come "
    "back in the next message. Ask again if you need more, up to {rounds} queries. When you "
    "have what you need, reply in plain prose with no code block: the answer, in the "
    "second person, with the numbers that support it, three to six sentences, no headings "
    "and no lists. Say what the record shows, not what you assume; if the record is too "
    "small to say (a few games, a few mistakes), say that plainly and say what it does show. "
    "Never invent a number: every figure you give must come from the rows you were sent. "
    "Prefer aggregates to raw rows, and leave out withdrawn moves. When a grouping comes "
    "back with a group missing (no moves under 30 seconds, no endgame), the record has "
    "none: query the column's range or count once if you must, then answer; do not ask "
    "the same question again in other words. Say it warmly, as a coach on their side: "
    "a weakness is something to practise, with how often they got it right as well as "
    "wrong, never a tally of failures or a verdict on them.\n\n"
    "# The record\n{schema}\n"
    "# Worked queries\n{examples}"
)

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass
class Exchange:
    """One query the model wrote, and what came back."""

    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[object, ...]] = field(default_factory=list)
    error: str | None = None


@dataclass
class Answer:
    text: str
    exchanges: list[Exchange] = field(default_factory=list)
    calls: int = 0
    # The model never wrote prose: the text is the coach's own fallback.
    failed: bool = False


def system_message(rounds: int = ROUNDS) -> str:
    examples = "\n".join(f"Q: {q}\n```sql\n{sql}\n```" for q, sql in EXAMPLES)
    return SYSTEM.format(rounds=rounds, schema=SCHEMA_DOC, examples=examples)


def extract_sql(text: str) -> str | None:
    """The one SQL block of a reply, or None when the reply is prose."""
    found = _FENCE.findall(text)
    if not found:
        return None
    return found[0].strip()


def _cell(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)  # "SELECT 1e999" crashed the ask (the parity audit)
        return f"{value:.3f}".rstrip("0").rstrip(".") if value != int(value) else str(int(value))
    text = str(value)
    return text if len(text) <= CELL_WIDTH else text[: CELL_WIDTH - 1] + "…"


def render_rows(columns: Sequence[str], rows: Sequence[tuple[object, ...]]) -> str:
    """The rows as a plain table the model can read."""
    if not rows:
        return "(no rows)"
    lines = [" | ".join(columns)]
    lines += [" | ".join(_cell(v) for v in row) for row in rows]
    if len(rows) >= QUERY_ROWS:
        lines.append(f"(first {QUERY_ROWS} rows only; aggregate or add LIMIT)")
    return "\n".join(lines)


class _Run:
    """The model's calls of one record question, all within the provider's timeout counted
    from the question: each call waited for on its own thread, and a timer set to the timeout
    ends the wait. Every call had the whole timeout to itself, so a question that queried the
    record four times could wait five times it (the audit, 2026-10-07). Past it the call
    raises TimeoutError, as a call that timed out always has raised out of `answer`; the call
    left running ends on its own timeout, and nothing waits for it."""

    _DONE, _FAILED, _EXPIRED = "done", "failed", "expired"

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._handoff: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._timer = threading.Timer(seconds, self._handoff.put, args=((self._EXPIRED, None),))
        self._timer.daemon = True
        self._timer.start()

    def call(self, ask: Callable[[], Completion]) -> Completion:
        def run() -> None:
            try:
                self._handoff.put((self._DONE, ask()))
            except Exception as exc:  # noqa: BLE001 - raised on the caller's thread
                self._handoff.put((self._FAILED, exc))

        threading.Thread(target=run, name="sidecoach-record-call", daemon=True).start()
        kind, value = self._handoff.get()
        if kind == self._DONE:
            return value
        if kind == self._FAILED:
            raise value
        raise TimeoutError(f"the record question took longer than its timeout ({self.seconds:g} s)")

    def close(self) -> None:
        self._timer.cancel()


def answer(
    question: str,
    store: Store,
    provider: LLMProvider,
    *,
    rounds: int = ROUNDS,
    max_tokens: int = MAX_TOKENS,
) -> Answer:
    """Answer the question from the record, the model writing the queries; the whole run
    within the provider's timeout when it has one (`_Run`)."""
    budget = getattr(provider, "timeout_s", None)
    run = _Run(float(budget)) if isinstance(budget, int | float) else None
    try:
        return _answer(question, store, provider, run, rounds=rounds, max_tokens=max_tokens)
    finally:
        if run is not None:
            run.close()


def _answer(
    question: str,
    store: Store,
    provider: LLMProvider,
    run: _Run | None,
    *,
    rounds: int,
    max_tokens: int,
) -> Answer:
    system = system_message(rounds)
    messages: list[ChatMessage] = [{"role": "user", "content": f"Question: {question.strip()}"}]
    result = Answer(text="")
    for round_index in range(rounds + 1):

        def ask() -> Completion:
            return provider.chat(system, list(messages), max_tokens=max_tokens, temperature=0.2)

        completion = run.call(ask) if run is not None else ask()
        result.calls += 1
        reply = (completion.text or "").strip()
        sql = extract_sql(reply)
        if sql is None or round_index == rounds:
            prose = _FENCE.sub("", reply).strip()
            if prose:
                result.text = prose
                return result
            break
        exchange = Exchange(sql=sql)
        try:
            exchange.columns, exchange.rows = store.query(sql)
            body = render_rows(exchange.columns, exchange.rows)
        except QueryRefused as exc:
            exchange.error = str(exc)
            body = f"Query refused: {exc}. Write a different single SELECT."
        result.exchanges.append(exchange)
        messages.append({"role": "assistant", "content": reply})
        left = rounds - round_index - 1
        hint = (
            f" You may query {left} more time{'s' if left != 1 else ''}."
            if left > 0
            else " Answer now in prose from what you have."
        )
        messages.append({"role": "user", "content": f"Rows:\n{body}\n{hint}"})
    result.failed = True
    result.text = fallback(result)
    return result


def fallback(result: Answer) -> str:
    """When the model never wrote prose: the last rows it saw, in a sentence."""
    good = [e for e in result.exchanges if e.error is None]
    if not good:
        return "I could not read the record for that; ask me again in a moment."
    last = good[-1]
    if not last.rows:
        return "The record has nothing on that yet."
    first = ", ".join(f"{c} {_cell(v)}" for c, v in zip(last.columns, last.rows[0], strict=True))
    more = f" and {len(last.rows) - 1} more rows" if len(last.rows) > 1 else ""
    return f"From the record: {first}{more}."
