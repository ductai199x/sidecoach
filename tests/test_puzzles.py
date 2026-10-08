"""Puzzles for the weak spots (DESIGN 5.7, M6): the store from the Lichess CSV, the pick
by motif and level, the recommendation after a review, and the outcome read off the
screen as a joined position's moves."""

from __future__ import annotations

import random
from pathlib import Path

import chess

from sidecoach.events import (
    GameStarted,
    Label,
    MotifTag,
    MovePlayed,
    PuzzleOutcome,
    PuzzleRef,
    Recommendation,
    Resync,
    Side,
)
from sidecoach.profile import glicko
from sidecoach.profile.profiler import Profiler
from sidecoach.profile.puzzles import (
    PERF_KEY,
    Attempt,
    Puzzle,
    PuzzleStore,
    open_store,
    position_key,
    puzzle_level,
    puzzle_perf,
    record_attempt,
)
from sidecoach.profile.store import Store
from tests.coaching import drain, scripted_coach
from tests.fakes import puzzle_file
from tests.profiling import Feed

# Rows in the Lichess database's format: the opponent's move first, then the solution.
ROWS = [
    "000Pw,6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - - 2 37,e4d2 d4e2 g1f1 e2c3,1547,76,92,629,"
    "crushing endgame fork short,https://lichess.org/au2lCK5o#73,",
    "000rO,3R4/8/K7/pB2b3/1p6/1P2k3/3p4/8 w - - 4 58,a6a5 e5c7 a5b4 c7d8,1110,82,85,74,"
    "crushing endgame fork master short,https://lichess.org/tzeeBEc2#115,",
    "fork1,r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,g8f6 f3e5 c6e5,"
    "1000,80,90,500,fork opening short,https://lichess.org/x#3,",
    "pin01,rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,d7d6 f1b5 c7c6,"
    "1200,80,90,400,pin opening short,https://lichess.org/y#3,",
    # Black's king steps to the corner; White mates on the back rank.
    "mate1,6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1,g8h8 a1a8,900,70,95,900,"
    "mateIn1 backRankMate endgame oneMove,https://lichess.org/z#1,",
    "hi999,6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1,g8h8 a1a8,1900,70,95,900,fork,"
    "https://lichess.org/w#1,",
]


def _csv(tmp_path: Path) -> Path:
    return puzzle_file(tmp_path / "sample.csv", ROWS)


def _store(tmp_path: Path) -> PuzzleStore:
    puzzles = PuzzleStore(None)
    puzzles.import_csv(_csv(tmp_path))
    return puzzles


def test_the_store_imports_the_csv_and_finds_puzzles_by_position_and_theme(
    tmp_path: Path,
) -> None:
    store = PuzzleStore(None)
    assert store.import_csv(_csv(tmp_path)) == 6
    assert store.import_csv(_csv(tmp_path)) == 0, "the same puzzles again add nothing"
    fork = store.get("fork1")
    assert fork is not None and fork.solver_moves == ("f3e5",) and fork.url.endswith("/fork1")
    # The solver's position is after the opponent's first move: what the screen shows.
    board = chess.Board(fork.fen)
    board.push_uci("g8f6")
    assert fork.position() == position_key(board.fen())
    assert [p.id for p in store.at_position(fork.position())] == ["fork1"]
    # Picks: the motif's themes within the window of the level, the excluded left out,
    # the ones due first first.
    picked = store.pick("fork", 1100, count=3, rng=random.Random(1))
    assert {p.id for p in picked} == {"000rO", "fork1"}, "000Pw is 1547, hi999 is 1900"
    only = store.pick("fork", 1100, exclude={"fork1"}, rng=random.Random(1))
    assert only == [store.get("000rO")]
    first = store.pick("fork", 1100, first=["fork1"], count=1, rng=random.Random(1))
    assert first[0].id == "fork1"
    assert [p.id for p in store.pick("mate", 1000)] == ["mate1"], "mate means mateIn1/2/3"
    assert store.pick("skewer", 1100) == []


def test_open_store_fills_from_the_csv_beside_it_once(tmp_path: Path) -> None:
    assert open_store(tmp_path) is None, "no puzzles anywhere"
    (tmp_path / "puzzles").mkdir()
    _csv(tmp_path / "puzzles")
    store = open_store(tmp_path)
    assert store is not None and store.count() == 6
    store.close()
    again = open_store(tmp_path)
    assert again is not None and again.count() == 6
    assert again.path == tmp_path / "puzzles.sqlite"
    again.close()


def test_an_attempt_is_solved_by_the_line_and_failed_by_a_wrong_move() -> None:
    puzzle = Puzzle(
        id="000Pw",
        fen="6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - - 2 37",
        moves=("e4d2", "d4e2", "g1f1", "e2c3"),
        rating=1547,
        popularity=76,
        plays=629,
        themes=("fork",),
    )
    a = Attempt(puzzle, "fork")
    assert a.move("d4e2", by_user=True) is None, "one of two solver moves: not yet"
    assert a.move("g1f1", by_user=False) is None, "the site's reply"
    assert a.move("e2c3", by_user=True) == "solved"
    b = Attempt(puzzle, "fork")
    assert b.move("d4c2", by_user=True) == "failed"
    assert b.move("d4e2", by_user=True) is None, "decided once"
    # Any mate counts where the line mates.
    mate = Puzzle(
        "m", "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1", ("g8h8", "a1a8"), 900, 1, 1, ("mateIn1",)
    )
    assert Attempt(mate, "mate").move("a1a8", by_user=True) == "solved"


def _fork_missed(f: object, san: str) -> None:
    """A user move that was a mistake with a fork on the board."""
    f.play(  # type: ignore[attr-defined]
        san,
        loss=0.15,
        label=Label.MISTAKE,
        after_tags=[MotifTag(name="fork", squares=["e5", "c6", "g6"])],
    )


def test_a_review_brings_a_recommendation_for_the_weak_motif(tmp_path: Path) -> None:
    f = Feed()
    f.profiler = Profiler(f.store, session_id="s", ts=0.0, puzzles=_store(tmp_path))
    f.send(GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.STARTING_FEN))
    f.play("e4", label=Label.BOOK)
    f.play("e5", label=Label.BOOK)
    _fork_missed(f, "Qh5")  # the game's turning point, a fork on the board
    f.play("Nc6")
    f.play("Bc4", label=Label.BEST)
    f.play("Nf6")
    f.play("Qxf7#", label=Label.BEST)
    got = f.end("1-0", "game over on board")
    rec = next(e for e in got if isinstance(e, Recommendation))
    # Around a new player's puzzle rating, 1500: the fork puzzles within 150 of it.
    assert rec.motif == "fork" and rec.motif_words == "fork" and rec.level == 1500
    assert "turning point" in rec.reason, "one game: the turning point's motif, not a cell yet"
    assert [p.id for p in rec.puzzles] == ["000Pw"]
    assert all(p.url.startswith("https://lichess.org/training/") for p in rec.puzzles)
    _, rows = f.store.query("SELECT puzzle_ids FROM recommendations")
    assert str(rows[0][0]).split() == [p.id for p in rec.puzzles]


def test_the_weak_motif_is_the_one_missed_most_often_over_the_games(tmp_path: Path) -> None:
    """The profile keeps misses (a cause) and chances (no cause) in separate cells, and asking
    one cell for both matched none: every recommendation was the turning point's, while the
    insight line said "You miss a hanging piece about 50% ... (2 of 4)" (the parity audit,
    2026-09-29, both sides). The motif practised is the one that line rates worst."""
    from sidecoach.events import GameReview, TurningPoint
    from sidecoach.profile.store import CellRow

    profiler = Profiler(Store(None), session_id="s", ts=0.0, puzzles=_store(tmp_path))
    games = frozenset({"a", "b", "c"})
    rows = [
        CellRow("missedThreat", "pin", "middlegame", errors=3, chances=0, expected_loss=0.9),
        CellRow(None, "pin", "middlegame", 3, 9, 0.0, games),
        CellRow("missedThreat", "fork", "middlegame", errors=3, chances=0, expected_loss=0.5),
        CellRow(None, "fork", "middlegame", 3, 4, 0.0, games),
    ]
    profiler.store.cells = lambda: rows  # type: ignore[method-assign]
    review = GameReview(
        game_id="g",
        user_side=Side.WHITE,
        result="1-0",
        turning_points=[
            TurningPoint(ply=5, san="Qh5", label="mistake", expected_loss=0.3, motif="pin")
        ],
    )
    assert profiler._weak_motif(review) == ("fork", "fork: found 1 of 4 times over your games")
    # Missed in too few games: the game's turning point still leads.
    rows[1] = CellRow(None, "pin", "middlegame", 3, 9, 0.0, frozenset({"a"}))
    rows[3] = CellRow(None, "fork", "middlegame", 3, 4, 0.0, frozenset({"a", "b"}))
    assert profiler._weak_motif(review) == ("pin", "pin: the turning point of this game")


def _faced(puzzles: PuzzleStore, puzzle_id: str) -> chess.Board:
    """The puzzle's position after the site's first move, as a cold read gives it: placement
    and side, no rights or counters."""
    puzzle = puzzles.get(puzzle_id)
    assert puzzle is not None
    board = chess.Board(puzzle.fen)
    board.push_uci(puzzle.moves[0])
    shown = chess.Board(None)
    for sq, piece in board.piece_map().items():
        shown.set_piece_at(sq, piece)
    shown.turn = board.turn
    return shown


def _played(game_id: str, board: chess.Board, san: str) -> MovePlayed:
    move = board.parse_san(san)
    before = board.fen()
    after = board.copy()
    after.push(move)
    return MovePlayed(
        game_id=game_id,
        ply=1,
        san=san,
        uci=move.uci(),
        fen_before=before,
        fen_after=after.fen(),
        by_user=True,
    )


def test_a_recommended_puzzle_opened_on_the_screen_is_scored(tmp_path: Path) -> None:
    puzzles = _store(tmp_path)
    store = Store(None)
    profiler = Profiler(store, session_id="s", ts=0.0, puzzles=puzzles)
    profiler._recommended_for["fork1"] = "fork"
    shown = _faced(puzzles, "fork1")
    started = GameStarted(game_id="p", user_side=Side.WHITE, start_fen=shown.fen(), kind="position")
    assert profiler.on_event(started, 1.0) == []
    out = profiler.on_event(_played("p", shown, "Nxe5"), 2.0)
    outcome = next(e for e in out if isinstance(e, PuzzleOutcome))
    assert outcome.solved and outcome.puzzle_id == "fork1" and outcome.motif == "fork"
    # The rating as Lichess moves it: a new player's first puzzle, won against fork1's 1000 / 80.
    won = glicko.finish(glicko.Perf(), glicko.Glicko(1000, 80, 0.09), True, 2000)
    assert outcome.level == won.glicko.int_rating and profiler.puzzle_level() == outcome.level
    assert outcome.level > 1500
    assert store.attempts() == [("fork1", 2.0, True)]
    # A wrong move on the next one fails it and the rating goes down.
    shown = _faced(puzzles, "pin01")
    started = GameStarted(game_id="q", user_side=Side.WHITE, start_fen=shown.fen(), kind="position")
    profiler.on_event(started, 3.0)
    out = profiler.on_event(_played("q", shown, "Nc3"), 4.0)
    outcome = next(e for e in out if isinstance(e, PuzzleOutcome))
    lost = glicko.finish(glicko.read(won, 4000), glicko.Glicko(1200, 80, 0.09), False, 4000)
    assert not outcome.solved and outcome.level == lost.glicko.int_rating < won.glicko.int_rating
    assert outcome.motif is None, "not one the profile recommended"
    # A position that is no puzzle of the store is left alone.
    plain = GameStarted(
        game_id="r", user_side=Side.WHITE, start_fen=chess.STARTING_FEN, kind="position"
    )
    assert profiler.on_event(plain, 5.0) == []
    assert profiler.on_event(_played("r", chess.Board(), "e4"), 6.0) == []


def test_the_next_puzzle_adopted_by_a_resync_is_scored_too(tmp_path: Path) -> None:
    """The site puts the next puzzle up with no start the reconciler could see: it is
    adopted over the last, in the same game, and scored all the same (only the first of a
    run was)."""
    puzzles = _store(tmp_path)
    store = Store(None)
    profiler = Profiler(store, session_id="s", ts=0.0, puzzles=puzzles)
    shown = _faced(puzzles, "fork1")
    started = GameStarted(game_id="p", user_side=Side.WHITE, start_fen=shown.fen(), kind="position")
    profiler.on_event(started, 1.0)
    out = profiler.on_event(_played("p", shown, "Nxe5"), 2.0)
    assert next(e for e in out if isinstance(e, PuzzleOutcome)).puzzle_id == "fork1"
    shown = _faced(puzzles, "pin01")
    profiler.on_event(Resync(game_id="p", fen=shown.fen(), reason="position adopted"), 3.0)
    out = profiler.on_event(_played("p", shown, "Bb5+"), 4.0)
    outcome = next(e for e in out if isinstance(e, PuzzleOutcome))
    assert outcome.puzzle_id == "pin01" and outcome.solved
    assert [pid for pid, _, _ in store.attempts()] == ["pin01", "fork1"]


def test_the_coach_says_one_line_on_the_practice_and_a_word_on_a_fail() -> None:
    coach, provider = scripted_coach([])
    coach.on_event(
        GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.STARTING_FEN), 0.0
    )
    coach.on_event(
        Recommendation(
            game_id="g",
            motif="fork",
            motif_words="fork",
            reason="fork: found 2 of 5 times over your games",
            level=1100,
            puzzles=[PuzzleRef(id="fork1", url="https://lichess.org/training/fork1", rating=1000)],
        ),
        1.0,
    )
    # One line, naming the count and the motif, the level, and why.
    (text,) = [m.text for m in drain(coach)]
    assert "1 fork puzzle " in text and "1100" in text
    assert "fork: found 2 of 5 times over your games" in text
    solved = PuzzleOutcome(game_id="p", puzzle_id="fork1", motif="fork", solved=True, level=1125)
    coach.on_event(solved, 2.0)
    assert drain(coach) == [], "a solve: the site says so"
    failed = PuzzleOutcome(game_id="p", puzzle_id="fork1", motif="fork", solved=False, level=1100)
    coach.on_event(failed, 3.0)
    (text,) = [m.text for m in drain(coach)]
    assert "fork puzzle" in text, "a word on the fail, naming the motif"
    assert provider.prompts == [], "no model call"
    assert "Practice recommended" in coach.conversation.render()
    # On the panel's own board (the coach kept quiet) the bar says it; the coach does not.
    coach.quiet = True
    coach.on_event(failed, 4.0)
    assert drain(coach) == []


def test_a_row_too_short_for_a_rating_or_moves_is_skipped(tmp_path: Path) -> None:
    # csv.DictReader fills a short row with None: int(None) and None.split() crashed the import.
    path = puzzle_file(
        tmp_path / "short.csv", [ROWS[0], "short1,8/8/8/8/8/8/8/K6k w - - 0 1", "short2"]
    )
    assert PuzzleStore(None).import_csv(path) == 1


# The puzzles table as it was before the puzzles kept their rating deviation.
OLD_SCHEMA = """
CREATE TABLE puzzles (
    id TEXT PRIMARY KEY, fen TEXT NOT NULL, moves TEXT NOT NULL, rating INTEGER NOT NULL,
    popularity INTEGER NOT NULL DEFAULT 0, plays INTEGER NOT NULL DEFAULT 0,
    themes TEXT NOT NULL, position TEXT NOT NULL
);
"""


def test_a_puzzle_keeps_the_rating_deviation_lichess_gives_it(tmp_path: Path) -> None:
    # The player's rating is computed against the puzzle's own rating and deviation, as
    # Lichess's PuzzleFinisher does: the import keeps the CSV's RatingDeviation, 75 where a
    # row has none or not a number.
    store = _store(tmp_path)
    assert [store.get(i).deviation for i in ("000Pw", "000rO", "mate1")] == [76, 82, 70]  # type: ignore[union-attr]
    odd = ROWS[2].replace(",1000,80,", ",1000,,").replace("fork1", "none1", 1)
    bad = ROWS[3].replace(",1200,80,", ",1200,x,").replace("pin01", "bad01", 1)
    path = puzzle_file(tmp_path / "odd.csv", [odd, bad])
    assert store.import_csv(path) == 2
    assert store.get("none1").deviation == 75 and store.get("bad01").deviation == 75  # type: ignore[union-attr]
    assert Puzzle("x", chess.STARTING_FEN, ("e2e4", "e7e5"), 1000, 0, 0, ()).deviation == 75


def test_a_store_from_before_the_deviations_gets_them_from_its_csv_once(tmp_path: Path) -> None:
    import sqlite3

    data = tmp_path / "data"
    puzzle_file(data / "puzzles" / "sample.csv", ROWS)
    old = sqlite3.connect(data / "puzzles.sqlite")
    old.executescript(OLD_SCHEMA)
    old.execute(
        "INSERT INTO puzzles VALUES ('fork1', ?, 'g8f6 f3e5 c6e5', 1000, 80, 500, 'fork', 'k')",
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",),
    )
    old.execute(
        "INSERT INTO puzzles VALUES ('gone1', ?, 'g8f6 f3e5', 1000, 80, 500, 'fork', 'k')",
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",),
    )
    old.commit()
    old.close()
    store = open_store(data)
    assert store is not None and store.version() == 1
    assert store.count() == 2, "the puzzles stay; only their deviations are filled in"
    assert store.get("fork1").deviation == 80, "from the CSV"  # type: ignore[union-attr]
    assert store.get("gone1").deviation == 75, "none in the CSV: the fallback"  # type: ignore[union-attr]
    store.close()
    # Once: a CSV changed afterwards is not read again.
    puzzle_file(
        data / "puzzles" / "sample.csv", [r.replace(",1000,80,", ",1000,99,") for r in ROWS]
    )
    store = open_store(data)
    assert store is not None and store.get("fork1").deviation == 80  # type: ignore[union-attr]
    store.close()


def test_without_its_csv_an_old_store_keeps_the_fallback_until_the_csv_comes(
    tmp_path: Path,
) -> None:
    import sqlite3

    data = tmp_path / "data"
    data.mkdir()
    old = sqlite3.connect(data / "puzzles.sqlite")
    old.executescript(OLD_SCHEMA)
    old.execute(
        "INSERT INTO puzzles VALUES ('fork1', ?, 'g8f6 f3e5 c6e5', 1000, 80, 500, 'fork', 'k')",
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",),
    )
    old.commit()
    old.close()
    store = open_store(data)
    assert store is not None and store.version() == 0 and store.get("fork1").deviation == 75  # type: ignore[union-attr]
    store.close()
    puzzle_file(data / "puzzles" / "sample.csv", ROWS)
    store = open_store(data)
    assert store is not None and store.version() == 1 and store.get("fork1").deviation == 80  # type: ignore[union-attr]
    store.close()


def test_the_puzzle_rating_starts_fresh_and_moves_as_lichess_moves_it(tmp_path: Path) -> None:
    # Tai, 2026-10-08: "start fresh". Everyone starts as Lichess's new player, 1500 / 500 /
    # 0.09; the old level is left as it was beside the new rating, never read again.
    store = Store(None)
    store.set_setting("puzzle_level", "1125")
    assert puzzle_level(store) == 1500 and puzzle_perf(store, 0) == glicko.Perf()
    fork = _store(tmp_path).get("000Pw")
    assert fork is not None and (fork.rating, fork.deviation) == (1547, 76)
    day = 24 * 3600
    outcome = record_attempt(store, fork, "fork", True, ["Ne2+"], 10 * day)
    want = glicko.finish(glicko.Perf(), glicko.Glicko(1547, 76, 0.09), True, 10 * day * 1000)
    assert outcome.level == want.glicko.int_rating == puzzle_level(store)
    assert puzzle_perf(store, 10 * day * 1000) == want
    assert store.setting("puzzle_level") == "1125", "the old level untouched"
    assert store.setting(PERF_KEY) is not None
    # A month away: the deviation grows before the next puzzle, as Lichess reads it.
    later = puzzle_perf(store, 40 * day * 1000)
    assert later.glicko.rating == want.glicko.rating
    assert later.glicko.deviation > want.glicko.deviation
    lost = record_attempt(store, fork, "fork", False, ["Nc2"], 40 * day)
    expect = glicko.finish(later, glicko.Glicko(1547, 76, 0.09), False, 40 * day * 1000)
    assert lost.level == expect.glicko.int_rating
    # The puzzle's own rating is not changed (Lichess changes it; a copy of its ratings here).
    assert _store(tmp_path).get("000Pw").rating == 1547  # type: ignore[union-attr]
    # Something saved that is no rating starts over as a new player; nothing raises.
    store.set_setting(PERF_KEY, "not json")
    assert puzzle_level(store) == 1500


def test_a_rating_outside_the_stores_band_is_searched_at_its_nearest_end(tmp_path: Path) -> None:
    # Lichess's rating has a floor of 400 and no ceiling; the store keeps 600-2000. Past either
    # end the search is made around that end (WINDOW of it), and the rating is not changed.
    store = _store(tmp_path)
    low = ROWS[2].replace("fork1", "low01", 1).replace(",1000,80,", ",700,80,")
    store.import_csv(puzzle_file(tmp_path / "low.csv", [low]))
    assert [p.id for p in store.pick("fork", 2300)] == ["hi999"], "searched around 2000"
    assert [p.id for p in store.pick("fork", 450)] == ["low01"], "searched around 600"
    assert [p.id for p in store.pick("fork", 2000)] == ["hi999"]
    # The recommendation still names the rating the player has.
    profiler_store = Store(None)
    profiler_store.set_setting(
        PERF_KEY, '{"r": 2300.5, "d": 60.0, "v": 0.06, "nb": 300, "la": null}'
    )
    assert puzzle_level(profiler_store) == 2300
