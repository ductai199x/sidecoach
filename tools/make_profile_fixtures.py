"""Record what the desktop's profile stage makes of real games, for the Kotlin port.

The profile turns a game's events into the record (every move with its facts), the evidence
(each mistake placed on the two axes, cause and motif), the review, the practice and the
snapshot of the player model -- and answers questions from the record with the coach's own
SQL. It is pure given its inputs, so it is compared exactly: the same events in, the same
events out, the same rows in every table.

The events are real. Four games are played on one board each, every move judged by the
desktop's annotator on the real Stockfish (one thread, so the recording is the same on every
run) with the motif tagger and the opening book, and fed to the profiler the way the panel
feeds it -- a clock before each move, the move, its verdict, the analysis of where it led:

- the Blackburne Shilling trap, the player White: a queen lost to a mate, a move made with
  eight seconds on the clock, a lesson from the coach on the mistake and the player's own
  words about it ("I didn't see the queen"), which become the self-report and the cause;
- Legal's mate, the player Black: the queen taken, a mate in two missed;
- a Ruy Lopez, the player White, with a blunder taken back and another move played, ended
  by resignation;
- a puzzle from the store, joined on the screen and solved move by move;
- a run of puzzles the site puts up one after another, each adopted by a resync over the
  last in the same game: two that are not in the store, played a few moves each, a second
  puzzle of the store solved between them, and a last one ended by mate -- none of it a
  game (Tai's Pixel, 2026-09-24: such a run was reviewed as "won by checkmate in 3 moves").

Then the session ends (its summary), a second session reads the first's summary back, and
the ask side is recorded against the database the games left: the system message, SQL found
in replies, the nine worked queries run and rendered, the statements refused and passed with
the words the model reads (the authorizer's "not authorized", SQLite's messages), a runaway
query interrupted and the row cap, values rendered as Python's str() shows them (infinities,
wide floats, blobs), and two conversations with a scripted model -- one answered, one that
never writes prose. And the review's first sentence for every way a game ends.

The puzzle draw is fixed (no shuffle) and the session id given, so nothing random is left.

Beside the games, `causes.wins_material` on its own -- every legal capture of the first
positions of the puzzle sample and of the desktop test's positions, where a guard pinned to
its own king takes nothing back (`exchange.can_retake`) -- and `causes.classify` on the
desktop test's pinned-guard blunder.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_profile_fixtures.py
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
import tempfile
from pathlib import Path
from typing import Any

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis import motifs  # noqa: E402
from sidecoach.analysis.annotator import GameAnnotator  # noqa: E402
from sidecoach.analysis.engine import Engine  # noqa: E402
from sidecoach.analysis.openings import OpeningBook  # noqa: E402
from sidecoach.config import find_stockfish, load_settings  # noqa: E402
from sidecoach.events import (  # noqa: E402
    ClockObservation,
    CoachMessage,
    DisclosureLevel,
    GameEnded,
    GameReview,
    GameStarted,
    MovePlayed,
    Resync,
    Side,
    Takeback,
    Trigger,
    UserUtterance,
)
from sidecoach.llm.base import Completion  # noqa: E402
from sidecoach.profile import ask, causes  # noqa: E402
from sidecoach.profile.profiler import Profiler  # noqa: E402
from sidecoach.profile.insights import motif_words  # noqa: E402
from sidecoach.profile.puzzles import PuzzleStore  # noqa: E402
from sidecoach.profile.review import lines_for  # noqa: E402
from sidecoach.profile.store import QueryRefused, Store  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "profile"
DEPTH = 12
MULTIPV = 4
PUZZLE_ROWS = 800


def dump(event: Any) -> dict:
    data = json.loads(event.model_dump_json(exclude={"ts", "source_frame_id"}))
    data["type"] = type(event).__name__
    return data


class Run:
    """The events fed, and what the profiler said back to each."""

    def __init__(self, profiler: Profiler) -> None:
        self.profiler = profiler
        self.steps: list[dict] = []
        self.ts = 1000.0

    def feed(self, event: Any, *, advance: float = 5.0) -> list[Any]:
        self.ts += advance
        out = self.profiler.on_event(event, self.ts)
        self.steps.append({"ts": self.ts, "event": dump(event), "out": [dump(e) for e in out]})
        return out

    def call(self, name: str, **kwargs: Any) -> None:
        """A method call on the profiler (tick, flush), recorded like an event."""
        self.ts += kwargs.pop("advance", 5.0)
        out = getattr(self.profiler, name)(self.ts)
        self.steps.append({"ts": self.ts, "call": name, "out": [dump(e) for e in out]})


def play(
    run: Run,
    annotator: GameAnnotator,
    game_id: str,
    user: Side,
    sans: list[str],
    *,
    clock_ms: int = 180_000,
    fast: dict[int, int] | None = None,
    lesson: tuple[int, str] | None = None,
    takeback: tuple[int, str] | None = None,
) -> chess.Board:
    """One game from the start: a clock, the move, its verdict and analysis, for each ply."""
    run.feed(GameStarted(game_id=game_id, user_side=user, start_fen=chess.STARTING_FEN))
    board = chess.Board()
    white, black = clock_ms, clock_ms
    ply = 0
    queue = list(sans)
    while queue:
        san = queue.pop(0)
        ply += 1
        mover = board.turn
        if fast and ply in fast:
            if mover == chess.WHITE:
                white = fast[ply]
            else:
                black = fast[ply]
        run.feed(ClockObservation(white_ms=white, black_ms=black, running=Side.WHITE if mover else Side.BLACK))
        before = board.fen()
        move = board.parse_san(san)
        assessment, analysis = annotator.assess(game_id, board, move, ply)
        board.push(move)
        by_user = (mover == chess.WHITE) == (user is Side.WHITE)
        run.feed(MovePlayed(game_id=game_id, ply=ply, san=san, uci=move.uci(), fen_before=before,
                            fen_after=board.fen(), by_user=by_user, think_ms=3000 + 100 * ply))
        run.feed(assessment)
        run.feed(analysis)
        if mover == chess.WHITE:
            white -= 4000
        else:
            black -= 4000
        if lesson and lesson[0] == ply:
            run.feed(CoachMessage(text="That knight grab let the queen in.", level=DisclosureLevel(3),
                                  trigger=Trigger.POST_MISTAKE, game_id=game_id, ply=ply - 1))
            run.feed(UserUtterance(text=lesson[1], source="text"), advance=10.0)
        if takeback and takeback[0] == ply:
            board.pop()
            ply -= 1
            run.feed(Takeback(game_id=game_id, plies=1, fen=board.fen()))
            queue.insert(0, takeback[1])
            takeback = None  # once: the move played instead stands
    return board


def puzzle_solved(run: Run, store: PuzzleStore) -> dict:
    """A puzzle of the store joined on the screen and solved move by move."""
    for motif in ("fork", "hangingPiece", "pin"):
        found = store.pick(motif, 1100, count=1, rng=_Fixed())
        if found:
            puzzle = found[0]
            break
    else:
        raise SystemExit("no puzzle in the sample to solve")
    board = chess.Board(puzzle.fen)
    board.push_uci(puzzle.moves[0])
    user = Side.WHITE if board.turn == chess.WHITE else Side.BLACK
    run.feed(GameStarted(game_id="puzzle-game", user_side=user, start_fen=board.fen(), kind="position"))
    for i, uci in enumerate(puzzle.moves[1:], start=1):
        before = board.fen()
        move = chess.Move.from_uci(uci)
        san = board.san(move)
        board.push(move)
        run.feed(MovePlayed(game_id="puzzle-game", ply=i, san=san, uci=uci, fen_before=before,
                            fen_after=board.fen(), by_user=i % 2 == 1))
    return {"id": puzzle.id}


def _moves_on(run: Run, game_id: str, board: chess.Board, user: Side, plies: int) -> None:
    """`plies` moves on a puzzle's position, the first legal one each time."""
    for i in range(1, plies + 1):
        move = next(iter(board.legal_moves))
        before = board.fen()
        san = board.san(move)
        by_user = (board.turn == chess.WHITE) == (user is Side.WHITE)
        board.push(move)
        run.feed(MovePlayed(game_id=game_id, ply=i, san=san, uci=move.uci(), fen_before=before,
                            fen_after=board.fen(), by_user=by_user))


def puzzle_run(run: Run, store: PuzzleStore, solved: str) -> dict:
    """Puzzles one after another in one game of the reconciler's, each next one adopted by
    a resync: every one a position of its own, a puzzle of the store scored in the middle."""
    found = [p for p in store.pick("fork", 1100, count=5, rng=_Fixed()) if p.id != solved]
    found += [p for p in store.pick("pin", 1100, count=5, rng=_Fixed()) if p.id != solved]
    assert found, "no second puzzle in the sample"
    second = found[0]
    user = Side.WHITE
    first = chess.Board("r1bq1k1r/1p2bp2/p2p3p/2p1n2Q/P3P3/1BN5/1PP2PPP/R1B2RK1 w - - 1 15")
    run.feed(GameStarted(game_id="puzzle-run", user_side=user, start_fen=first.fen(), kind="position"))
    run.feed(Resync(game_id="puzzle-run", fen=first.fen(), reason="the position announced again"))
    _moves_on(run, "puzzle-run", first, user, 4)
    other = chess.Board("r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R w KQkq - 0 1")
    run.feed(Resync(game_id="puzzle-run", fen=other.fen(), reason="position adopted after 24 cells"))
    _moves_on(run, "puzzle-run", other, user, 4)
    board = chess.Board(second.fen)
    board.push_uci(second.moves[0])
    shown = chess.Board(None)
    for sq, piece in board.piece_map().items():
        shown.set_piece_at(sq, piece)
    shown.turn = board.turn
    run.feed(Resync(game_id="puzzle-run", fen=shown.fen(), reason="position adopted after 30 cells"))
    for i, uci in enumerate(second.moves[1:], start=1):
        before = shown.fen()
        move = chess.Move.from_uci(uci)
        san = shown.san(move)
        shown.push(move)
        run.feed(MovePlayed(game_id="puzzle-run", ply=i, san=san, uci=uci, fen_before=before,
                            fen_after=shown.fen(), by_user=i % 2 == 1))
    last = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
    run.feed(Resync(game_id="puzzle-run", fen=last.fen(), reason="position adopted after 26 cells"))
    _moves_on(run, "puzzle-run", last, Side.WHITE, 3)
    run.feed(GameEnded(game_id="puzzle-run", result="1-0", reason="game over on board", pgn=None,
                       final_fen=last.fen()))
    return {"id": second.id}


class _Fixed(random.Random):
    """No shuffle: the draw among a recommendation's puzzles, fixed for the comparison."""

    def shuffle(self, x: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ARG002
        return None


class Scripted:
    """A model that says what it is told to, in order."""

    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def healthcheck(self) -> tuple[bool, str]:
        return True, "scripted"

    def chat(self, system: str, messages: list, **_: Any) -> Completion:
        self.seen.append([dict(m) for m in messages])
        return Completion(text=self.replies.pop(0) if self.replies else "```sql\nSELECT 1\n```")


# The desktop test's positions for `wins_material`: a guard pinned to its own king, the same
# guard free, the king that cannot take back onto a guarded square.
CAUSES_BUILT = [
    "4k3/8/5n2/8/1b2B3/2N5/8/4K3 b - - 0 1",
    "4k3/8/5n2/8/4B3/2N5/8/4K3 b - - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    "4k3/8/8/8/1b2n3/2N5/8/4K3 w - - 0 1",
]


def causes_cases(sample: Path) -> dict:
    """`wins_material` for every capture of some positions, and one `classify`."""
    boards = [chess.Board(fen) for fen in CAUSES_BUILT]
    with sample.open(encoding="utf-8", newline="") as handle:
        for row in list(csv.DictReader(handle))[:60]:
            board = chess.Board(row["FEN"])
            for uci in row["Moves"].split():
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    break
                board.push(move)
                boards.append(board.copy(stack=False))
    wins = []
    for board in boards:
        for move in board.legal_moves:
            if board.is_capture(move):
                wins.append({"fen": board.fen(), "reply": move.uci(),
                             "wins": causes.wins_material(board, move)})
    # Be4?? beside the f6 knight, the c3 knight that guards e4 pinned by the b4 bishop.
    board = chess.Board("4k3/8/5n2/8/1b6/2N2B2/8/4K3 w - - 0 1")
    cause, motif = causes.classify(board, chess.Move.from_uci("f3e4"), reply="f6e4")
    classified = [{"fen": board.fen(), "move": "f3e4", "reply": "f6e4", "cause": cause, "motif": motif}]
    return {"wins_material": wins, "classify": classified}


# The coach's SQL the store must refuse, and some it must not: the desktop test's nine, a write
# inside a WITH and the pragmas' table-valued functions (which the authorizer denies as "not
# authorized"), SQLite's own errors, and reads that pass. Each with the words the model reads.
REFUSALS = [
    "DELETE FROM games",
    "UPDATE games SET result = '1-0'",
    "INSERT INTO sessions (session_id, started_at) VALUES ('x', 0)",
    "DROP TABLE games",
    "PRAGMA table_info(games)",
    "ATTACH DATABASE 'x.db' AS x",
    "SELECT 1; DELETE FROM games",
    "",
    "SELECT * FROM nowhere",
    "   ;; ",
    "SELECT 1; SELECT 2",
    "PRAGMA table_info(moves)",
    "DELETE FROM moves",
    "SELECT nope FROM games",
    "SELEC 1",
    "SELECT FROM",
    "WITH x AS (SELECT 1) DELETE FROM games",
    "WITH x AS (SELECT 1) UPDATE games SET result = '1-0'",
    "WITH x AS (SELECT 1) INSERT INTO settings (key, value) VALUES ('a', 'b')",
    "WITH x AS (SELECT 1) REPLACE INTO settings (key, value) VALUES ('a', 'b')",
    "WITH x AS (SELECT 1) DELETE FROM nowhere",
    "SELECT * FROM pragma_table_info('games')",
    "SELECT name FROM pragma_database_list",
    "select * from PRAGMA_TABLE_INFO('moves')",
    "SELECT * FROM \"pragma_table_info\"('games')",
    "SELECT * FROM main.pragma_table_info('games')",
    # Not "SELECT load_extension('x')": refused on both, but Android's SQLite is built without
    # the function ("no such function: load_extension") where the desktop's denies it.
    "SELECT 'pragma_table_info' AS word",
    "select count(*) as n from games ;",
    "WITH g AS (SELECT * FROM games) SELECT COUNT(*) AS n FROM g",
    "SELECT json_each.value FROM games, json_each('[1,2]')",
    "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name",
]
RUNAWAY = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n) SELECT COUNT(*) FROM n"
CAPPED = "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n LIMIT 500) SELECT x FROM n"
RENDERS = [
    "SELECT 1e999 AS big, -1e999 AS small, 2.5 AS half, 1e20 AS huge, 0.1235 AS r, 3.0 AS three,"
    " -2.0 AS minus, x'0141ff27' AS blob, x'22' AS dq, x'' AS empty, 'it''s' AS s, NULL AS n, 7 AS i,"
    " '" + "y" * 70 + "' AS long",
    "SELECT 123456789012345678901234567890.0 AS wide, 1e300 AS vast, -0.0005 AS tiny, 0.0015 AS tie,"
    " -0.0001 AS signed, -0.0 AS nought",
]
# Texts as Python's csv module reads them (newline=""): line ends of every kind, quoted fields
# holding commas, quotes and line breaks, a quote inside an unquoted field, text after a closing
# quote, blank lines, no final line end, a quoted field the text ends in.
CSV_TEXTS = [
    "a,b\r\nc,d\r\n",
    "a,b\rc,d\r",
    'a,"b,c"\n"d""e",f\n',
    'x,"multi\nline\r\nfield",y\n',
    'p,q"r,s\n',
    '"ab"cd,e\n',
    "\n\na,b\n\n",
    "a,b",
    'a,"unterminated\nstill',
    "a,\r\n,\n",
    '"a"\r\n',
    "a,b\r\r\nc",
    '"",x,""\n',
]
# A puzzle file with all of that in it, and rows `import_csv` must keep or skip: a short row
# (no GameUrl, no OpeningTags), a rating with spaces and underscores, a rating that is no
# number, a rating below the band, no popularity, one move only, an illegal first move.
CSV_IMPORT = (
    "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags\r\n"
    "fork1,r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,g8f6 f3e5 c6e5,"
    '1000,80,90,500,"fork opening short",https://lichess.org/x#3,\r\n'
    "\r\n"
    "pin01,rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,d7d6 f1b5 c7c6,"
    " 1_200 ,80,,400,pin opening short\r"
    "mate1,6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1,g1h1  a1a8,900,70,95,900,"
    '"mateIn1 backRankMate","https://lichess.org/z#1","a ""b"", c"\n'
    "bad01,6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1,g1h1 a1a8,twelve,70,95,900,fork,,\n"
    "low01,6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1,g1h1 a1a8,599,70,95,900,fork,,\n"
    "one01,6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1,g1h1,900,70,95,900,fork,,\n"
    "ill01,6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1,h2h5 a8a1,900,70,95,900,fork,,"
)
JSON_LISTS = [
    [],
    ["fork", "pin"],
    ["del\x7f", "tab\t", "nl\n", "bell\x07", "quote\"", "back\\slash", "slash/"],
    ["café", "½", "\U0001f600", " ", "~ !"],
]
ENDINGS = [
    ("0-1", "user resigned or lost on time", Side.WHITE),
    ("1-0", "opponent resigned or lost on time", Side.WHITE),
    ("0-1", "lost on time", Side.WHITE),
    ("1-0", "lost on time", Side.BLACK),
    ("1-0", "game over on board", Side.WHITE),
    ("1/2-1/2", "fifty-move rule", Side.WHITE),
    ("1-0", "checkmate", Side.WHITE),
    ("1/2-1/2", "draw agreed", Side.WHITE),
    ("1/2-1/2", "stalemate", Side.BLACK),
    ("0-1", None, Side.BLACK),
    ("0-1", "", Side.BLACK),
    ("*", "user resigned or lost on time", Side.WHITE),
]


# tests/test_profile.py's profiler tests, run as they are (their own asserts hold) with every
# call on a profiler recorded: its events in and out, a tick, a flush, a correction, the
# session's end, and at the end every table of every store it wrote. The phone replays them.
SCENARIOS = [
    ("test_a_mistake_becomes_evidence_once_its_analysis_arrives", ()),
    ("test_the_opponents_mistakes_and_fine_moves_are_not_evidence", ()),
    ("test_the_clock_makes_it_time_pressure", ()),
    ("test_the_game_ends_with_a_review_and_a_snapshot", ()),
    ("test_a_takeback_withdraws_the_moves_and_their_evidence", ()),
    ("test_a_resync_starts_a_new_segment_and_keeps_the_earlier_moves", ()),
    ("test_the_session_summary_carries_to_the_next_session", ()),
    ("test_a_correction_moves_the_evidence", ()),
    ("test_the_player_may_be_either_colour", (Side.WHITE,)),
    ("test_the_player_may_be_either_colour", (Side.BLACK,)),
    ("test_the_review_waits_for_the_engines_last_verdicts", ()),
    ("test_a_review_left_waiting_is_built_when_the_grace_period_is_up_or_at_the_close", ()),
    ("test_the_next_game_flushes_the_last_games_review", ()),
    ("test_the_players_words_after_a_lesson_are_kept_beside_the_mistake", ()),
    ("test_a_position_is_coached_but_never_recorded", ()),
    ("test_a_game_joined_from_the_middle_is_still_a_game", ()),
    ("test_a_position_that_ends_early_leaves_nothing_behind", ()),
    ("test_puzzles_one_after_another_never_add_up_to_a_game", ()),
    ("test_a_joined_game_whose_turn_the_clock_settles_is_still_a_game", ()),
    ("test_a_game_whose_end_was_not_seen_is_still_reviewed_at_the_next_start", ()),
]


def scenarios() -> list[dict]:
    import tests.test_profile as tp

    real = tp.Profiler
    out = []
    for name, args in SCENARIOS:
        steps: list[dict] = []
        stores: dict[int, tuple[int, Store]] = {}
        made = [0]

        def recording(store: Store, session_id: str | None = None, ts: float = 0.0, puzzles: Any = None) -> Any:
            if id(store) not in stores:
                stores[id(store)] = (len(stores), store)
                steps.append({"new_store": stores[id(store)][0]})
            index = made[0]
            made[0] += 1
            steps.append({"new_profiler": index, "store": stores[id(store)][0], "session": session_id, "ts": ts})
            return _Recorded(real(store, session_id=session_id, ts=ts, puzzles=puzzles), index, steps)

        tp.Profiler = recording  # type: ignore[assignment]
        try:
            getattr(tp, name)(*args)
        finally:
            tp.Profiler = real  # type: ignore[assignment]
        out.append({
            "name": name + ("" if not args else f"[{args[0].value}]"),
            "steps": steps,
            "tables": [tables(store) for _, store in sorted(stores.values(), key=lambda v: v[0])],
        })
    return out


class _Recorded:
    """A profiler whose every call is written down."""

    def __init__(self, profiler: Profiler, index: int, steps: list[dict]) -> None:
        self._p = profiler
        self._i = index
        self._steps = steps

    def on_event(self, event: Any, ts: float) -> list[Any]:
        got = self._p.on_event(event, ts)
        self._steps.append({"profiler": self._i, "ts": ts, "event": dump(event), "out": [dump(e) for e in got]})
        return got

    def _call(self, name: str, ts: float) -> list[Any]:
        got = getattr(self._p, name)(ts)
        self._steps.append({"profiler": self._i, "ts": ts, "call": name, "out": [dump(e) for e in got]})
        return got

    def tick(self, ts: float) -> list[Any]:
        return self._call("tick", ts)

    def flush(self, ts: float) -> list[Any]:
        return self._call("flush", ts)

    def end_session(self, ts: float) -> str | None:
        summary = self._p.end_session(ts)
        self._steps.append({"profiler": self._i, "ts": ts, "call": "end_session", "summary": summary})
        return summary

    def correct(self, game_id: str, segment: int, ply: int, cause: str | None) -> bool:
        ok = self._p.correct(game_id, segment, ply, cause)
        self._steps.append({"profiler": self._i, "call": "correct", "args": [game_id, segment, ply, cause], "ok": ok})
        return ok

    def snapshot(self) -> Any:
        snap = self._p.snapshot()
        self._steps.append({"profiler": self._i, "call": "snapshot", "out": [dump(snap)]})
        return snap

    def __getattr__(self, name: str) -> Any:
        return getattr(self._p, name)


def causes_tables() -> dict:
    """tests/test_profile.py's cause cases (`classify` with every argument it takes, `tactical`),
    and the words: `from_words`, `motif_words`."""
    import tests.test_profile as tp

    tag = lambda t: {"name": t.name, "squares": list(t.squares), "move": t.move}  # noqa: E731
    fork, sac, capture = tp.FORK, tp.SAC, tp.CAPTURE
    start = chess.STARTING_FEN
    nxe5 = "r1bqkbnr/pppp1ppp/8/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    qh4 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"
    pinned = "4k3/8/5n2/8/1b6/2N2B2/8/4K3 w - - 0 1"
    bxf7 = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    cases = [
        (start, "e2e4", {"allowed_tags": [fork], "clock_ms": 8_000}),
        (start, "e2e4", {"allowed_tags": [fork]}),
        (start, "f2f3", {"reply_mate": 2}),
        (start, "e2e4", {"available_tags": [fork], "allowed_tags": [fork], "missed_chance": True}),
        (start, "e2e4", {"available_tags": [fork], "allowed_tags": [fork]}),
        (start, "e2e4", {"available_tags": [fork], "reply_mate": 1, "missed_chance": True}),
        (nxe5, "f3e5", {"allowed_tags": [fork]}),
        (nxe5, "f3e5", {"reply_mate": 2}),
        (qh4, "d8h4", {"reply": "f3h4"}),
        (pinned, "f3e4", {"reply": "f6e4"}),
        (bxf7, "c4f7", {"played_tags": [sac, capture]}),
        (bxf7, "c4f7", {"played_tags": [capture], "available_tags": [fork]}),
        (start, "a2a3", {"available_tags": [fork]}),
        (start, "a2a3", {"phase": "endgame"}),
        (start, "a2a3", {"phase": "opening"}),
        (start, "a2a3", {"phase": "middlegame", "left_book": True}),
        (start, "a2a3", {"phase": "middlegame"}),
        (start, "a2a3", {"allowed_tags": [capture], "clock_ms": 30_000}),
    ]
    classify = []
    for fen, uci, kwargs in cases:
        cause, motif = causes.classify(chess.Board(fen), chess.Move.from_uci(uci), **kwargs)
        given = {k: ([tag(t) for t in v] if k.endswith("_tags") else v) for k, v in kwargs.items()}
        classify.append({"fen": fen, "move": uci, "args": given, "cause": cause, "motif": motif})
    tactical = [
        {"tags": [tag(t) for t in tags], "motif": causes.tactical(tags)}
        for tags in ([capture], [capture, fork], [sac, fork], [])
    ]
    words = [
        "ok", "what should I play now?", "I completely missed it", "no time, I premoved",
        "I saw the knight coming but thought the queen was safe", "I was rushing",
        "I didn't see the queen coming to g5 at all",
    ]
    motifs_ = ["hangingPiece", "mateIn2", "attackingF2F7", "xRayAttack", "fork", "mate", "backRankMate"]
    return {
        "classify": classify,
        "tactical": tactical,
        "tactical_set": sorted(causes.TACTICAL),
        "from_words": [{"text": w, "cause": causes.from_words(w)} for w in words],
        "motif_words": [{"motif": m, "words": motif_words(m)} for m in motifs_],
    }


def csv_import() -> dict:
    """What `import_csv` keeps of CSV_IMPORT, row by row."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tricky.csv"
        path.write_bytes(CSV_IMPORT.encode("utf-8"))
        store = PuzzleStore(None)
        added = store.import_csv(path)
        rows = [list(r) for r in store._db.execute(
            "SELECT id, fen, moves, rating, popularity, plays, themes, position FROM puzzles ORDER BY id"
        ).fetchall()]
        store.close()
    return {"text": CSV_IMPORT, "added": added, "rows": rows}


def tables(store: Store) -> dict:
    out = {}
    for table in ("games", "moves", "evidence", "dialogue", "sessions", "recommendations",
                  "puzzle_attempts", "settings"):
        cursor = store._db.execute(f"SELECT * FROM {table} ORDER BY rowid")
        names = [d[0] for d in cursor.description]
        out[table] = {"columns": names, "rows": [list(r) for r in cursor.fetchall()]}
    return out


def main() -> None:
    random.Random.shuffle = _Fixed.shuffle  # type: ignore[method-assign]  # the profiler's draw
    path = find_stockfish(load_settings())
    assert path is not None, "no Stockfish for the desktop"
    book = OpeningBook.load(DESKTOP / "data" / "openings")
    assert book is not None
    OUT.mkdir(parents=True, exist_ok=True)

    # The puzzles: the first rows of the sample, written beside the fixture for the phone.
    sample = DESKTOP / "data" / "puzzles" / "sample.csv"
    with sample.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))[: PUZZLE_ROWS + 1]
    puzzles_csv = OUT / "puzzles.csv"
    with puzzles_csv.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)
    puzzles = PuzzleStore(None)
    imported = puzzles.import_csv(puzzles_csv)

    store = Store(None)
    profiler = Profiler(store, session_id="session-one", ts=1000.0, puzzles=puzzles)
    run = Run(profiler)
    first_snapshot = dump(profiler.snapshot())
    with Engine(path=path, threads=1, hash_mb=64, multipv=MULTIPV) as engine:
        def annotator() -> GameAnnotator:
            return GameAnnotator(engine=engine, depth=DEPTH, multipv=MULTIPV, tagger=motifs, book=book)

        board = play(
            run, annotator(), "shilling", Side.WHITE,
            "e4 e5 Nf3 Nc6 Bc4 Nd4 Nxe5 Qg5 Nxf7 Qxg2 Rf1 Qxe4+ Be2 Nf3#".split(),
            fast={11: 8000},
            lesson=(9, "I didn't see the queen coming to g5 at all"),
        )
        run.feed(GameEnded(game_id="shilling", result="0-1", reason="game over on board",
                           pgn=None, final_fen=board.fen()))
        board = play(
            run, annotator(), "legal", Side.BLACK,
            "e4 e5 Nf3 d6 Bc4 Bg4 Nc3 g6 Nxe5 Bxd1 Bxf7+ Ke7 Nd5#".split(),
        )
        run.feed(GameEnded(game_id="legal", result="1-0", reason="game over on board",
                           pgn=None, final_fen=board.fen()))
        board = play(
            run, annotator(), "ruy", Side.WHITE,
            ("e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Na5 Bc2 c5 "
             "d4 Qc7 Nbd2 cxd4 cxd4 Nc6 Nb3 a5 Be3 a4 Nbd2 Bd7 Qe2").split(),
            takeback=(33, "Rc1"),
        )
        run.feed(GameEnded(game_id="ruy", result="0-1", reason="user resigned or lost on time",
                           pgn=None, final_fen=board.fen()))
    puzzle = puzzle_solved(run, puzzles)
    run_of = puzzle_run(run, puzzles, puzzle["id"])
    run.call("tick")
    summary = profiler.end_session(run.ts + 5.0)
    second = Profiler(store, session_id="session-two", ts=run.ts + 10.0, puzzles=puzzles)
    second_snapshot = dump(second.snapshot())

    # The ask side, against the record the games left.
    queries = []
    for question, sql in ask.EXAMPLES:
        columns, rows_ = store.query(sql)
        queries.append({"question": question, "sql": sql, "columns": columns,
                        "rows": [list(r) for r in rows_], "rendered": ask.render_rows(columns, rows_)})
    refused = []
    for sql in REFUSALS:
        try:
            store.query(sql)
            refused.append({"sql": sql, "refused": False, "error": None})
        except QueryRefused as exc:
            refused.append({"sql": sql, "refused": True, "error": str(exc)})
    # A runaway query is interrupted, and the row cap holds (the desktop test's two).
    try:
        store.query(RUNAWAY, seconds=0.2)
        runaway = None
    except QueryRefused as exc:
        runaway = str(exc)
    _, capped = store.query(CAPPED, rows=10)
    bounded = {"runaway": RUNAWAY, "runaway_error": runaway, "capped": CAPPED, "capped_rows": len(capped)}
    # Values the rows show as Python's str() shows them: infinities, a float past 2**63,
    # blobs, a long text cut (`ask._cell`).
    renders = []
    for sql in RENDERS:
        columns, rows_ = store.query(sql)
        renders.append({"sql": sql, "rendered": ask.render_rows(columns, rows_)})
    cell_values = [{"value": v, "cell": ask._cell(float(v))} for v in ("inf", "-inf", "nan", "-0.0", "1e20")]
    # The review's first sentence, for every way a game ends (`review.HOW_IT_ENDED`).
    endings = []
    for result, reason, side in ENDINGS:
        review = GameReview(game_id="g", user_side=side, result=result, reason=reason)
        endings.append({"result": result, "reason": reason, "user_side": side.value,
                        "line": lines_for(review, 17)[0]})
    extracts = [
        {"text": t, "sql": ask.extract_sql(t)}
        for t in (
            "Here:\n```sql\nSELECT COUNT(*) FROM games\n```",
            "```\nSELECT 1\n```",
            "```SQL\n  SELECT 2  \n```\n```sql\nSELECT 3\n```",
            "You have played three games.",
        )
    ]
    conversations = []
    for replies in (
        ["Let me look.\n```sql\nSELECT COUNT(*) AS games, ROUND(AVG(final_score), 2) AS score FROM games"
         " WHERE result IS NOT NULL\n```",
         "```sql\nDELETE FROM games\n```",
         "You have finished three games and scored poorly in them."],
        [],
    ):
        provider = Scripted(replies)
        answer = ask.answer("How am I doing?", store, provider)  # type: ignore[arg-type]
        conversations.append({
            "replies": replies,
            "text": answer.text,
            "calls": answer.calls,
            "failed": answer.failed,
            "exchanges": [{"sql": e.sql, "columns": e.columns, "rows": [list(r) for r in e.rows],
                           "refused": e.error is not None} for e in answer.exchanges],
            "messages": provider.seen,
        })

    fixture = {
        "puzzles_imported": imported,
        "session": "session-one",
        "first_snapshot": first_snapshot,
        "steps": run.steps,
        "puzzle": puzzle,
        "puzzle_run": run_of,
        "summary": summary,
        "second_snapshot": second_snapshot,
        "tables": tables(store),
        "system": ask.system_message(),
        "queries": queries,
        "refused": refused,
        "bounded": bounded,
        "renders": renders,
        "cell_values": cell_values,
        "endings": endings,
        # A list as the record writes it (json.dumps): every character outside ' '..'~' escaped.
        "json_lists": [{"items": items, "json": json.dumps(items)} for items in JSON_LISTS],
        "csv_texts": [{"text": t, "records": list(csv.reader(io.StringIO(t, newline="")))} for t in CSV_TEXTS],
        "csv_import": csv_import(),
        "scenarios": scenarios(),
        "cause_tables": causes_tables(),
        "extracts": extracts,
        "conversations": conversations,
        "cells": [c.__dict__ for c in store.cells()],
        "causes": causes_cases(sample),
    }
    (OUT / "profile.json").write_text(json.dumps(fixture), encoding="utf-8")
    emitted = [e["type"] for s in run.steps for e in s["out"]]
    print(f"{len(run.steps)} steps; emitted {sorted(set(emitted))} "
          f"({emitted.count('ProfileEvidence')} evidence, {emitted.count('GameReview')} reviews); "
          f"{imported} puzzles; summary: {summary}")
    for line in second_snapshot["insights"]:
        print("  insight:", line)


if __name__ == "__main__":
    main()
