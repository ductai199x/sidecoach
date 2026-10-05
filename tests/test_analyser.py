"""The engine service for a game followed live (`analysis/analyser.py`): every position searched
once, on a pool of workers; the look-ahead's searches promoted when the game reaches them, never
begun again; a search stopped only when nothing will read it; a move judged from the lines that
are there. Gated fake engines stage every race; nothing waits on a clock."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import chess
import pytest

from sidecoach.analysis.analyser import Analyser, open_analyser
from sidecoach.analysis.annotator import EARLY_DEPTH, GameAnnotator
from sidecoach.analysis.engine import Engine
from sidecoach.events import Analysis, Label, Line, MoveAssessment
from tests.fakes import AnalyserWatch, FakeEngine, GatedEngine
from tests.test_analysis import SCHOLAR, STOCKFISH, needs_engine

START = chess.Board().fen()


@pytest.fixture(scope="module", name="engine")
def _real_engine() -> Iterator[Engine]:
    """The real Stockfish, one thread as every search (`needs_engine`'s tests)."""
    assert STOCKFISH is not None
    with Engine(path=STOCKFISH, threads=1, hash_mb=64, multipv=4) as running:
        yield running


def _line(*moves: str, score: float = 0.5) -> Line:
    return Line(moves=list(moves), depth=18, expected_score=score)


def _after(*sans: str) -> str:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board.fen()


def _board(*sans: str) -> chess.Board:
    board = chess.Board()
    for san in sans:
        board.push_san(san)
    return board


def _analyser(
    engines: Sequence[FakeEngine], *, look_ahead: int = 2, threats: bool = False
) -> Analyser:
    annotator = GameAnnotator(engine=engines[0], depth=18, with_threats=threats)
    return Analyser(annotator, engines=engines, look_ahead=look_ahead)


def test_every_position_of_a_game_is_searched_once() -> None:
    """The engine audit (2026-10-04): 258 of 624 positions of a session were searched two to
    six times. A game played, its moves both looked at ahead and not, one taken back and played
    again: each position, with the moves that led to it, searched exactly once."""
    lines = {
        START: [_line("e2e4"), _line("d2d4")],
        _after("e4"): [_line("e7e5"), _line("c7c5")],
        _after("d4"): [_line("d7d5")],
        _after("e4", "e5"): [_line("g1f3"), _line("b1c3")],
    }
    engines = [FakeEngine(lines), FakeEngine(lines)]
    analyser = _analyser(engines)
    watch = AnalyserWatch(analyser)
    with analyser:
        board = chess.Board()
        analyser.submit_position("g", board, 0)
        for ply, san in enumerate(["e4", "e5", "Nf3"], start=1):
            move = board.parse_san(san)
            analyser.submit_move("g", board, move, ply)
            board.push(move)
            watch.idle()
        board.pop()
        analyser.withdraw("g", 2)
        analyser.submit_move("g", board, board.parse_san("Nf3"), 3)
        watch.idle()
        watch.until(lambda: not analyser._claimed, "the look-ahead done")
    searched = [
        (fen, tuple(h))
        for e in engines
        for fen, h, m in zip(e.calls, e.histories, e.multipvs, strict=True)
        if m > 1
    ]
    assert len(searched) == len(set(searched)), sorted(searched)
    assert analyser.searches == len(searched)


def test_a_move_among_the_lines_is_judged_without_a_search_of_its_own() -> None:
    """85% of a session's moves are among the lines of the position they were played from:
    judged from their own line there, at the same depth as the best move (the same label in
    319 of 320), and at once -- the search of the position after them, held here, is not
    waited for."""
    lines = {START: [_line("e2e4", score=0.60), _line("d2d4", score=0.55)]}
    engine = GatedEngine(lines, gated=[_after("d4")])
    analyser = _analyser([engine], look_ahead=0)
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        watch.idle()
        analyser.poll()
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("d2d4"), 1)
        assert engine.entered[_after("d4")].wait(5), "the position after d4 never searched"
        watch.until(lambda: analyser._next_verdict == 2, "d4 judged")
        verdict = next(e for e in analyser.poll() if isinstance(e, MoveAssessment))
        engine.release(_after("d4"))
        watch.idle()
    assert verdict.uci == "d2d4" and verdict.expected_loss == pytest.approx(0.05)
    assert verdict.label is not Label.BEST


def test_a_move_not_among_the_lines_waits_for_the_position_after() -> None:
    """A move none of the lines begins with is judged from the position after's search -- the
    game's next position, searched anyway -- so its verdict waits for that search."""
    lines = {
        START: [_line("e2e4", score=0.60)],
        _after("a3"): [_line("e7e5", score=0.55)],
    }
    engine = GatedEngine(lines, gated=[_after("a3")])
    analyser = _analyser([engine], look_ahead=0)
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        watch.idle()
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("a2a3"), 1)
        assert engine.entered[_after("a3")].wait(5)
        assert analyser._next_verdict == 1, "judged before the position after was searched"
        engine.release(_after("a3"))
        got = watch.settled()
    verdict = next(e for e in got if isinstance(e, MoveAssessment))
    assert verdict.expected_loss == pytest.approx(0.60 - (1 - 0.55))


def test_a_position_searched_ahead_is_promoted_when_the_game_reaches_it() -> None:
    """The game reaches a position its look-ahead is still searching: that search becomes the
    game's -- told as the game's position when it ends -- and is never begun again. It was
    stopped and searched again from depth 1 (the engine audit: 'look-ahead -> look-ahead ->
    early -> verdict -> full' on one position)."""
    lines = {START: [_line("e2e4"), _line("d2d4")]}
    gated = [_after("e4")]
    engines = [GatedEngine(lines, gated=gated), GatedEngine(lines, gated=gated)]
    analyser = _analyser(engines, look_ahead=1)
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        assert engines[1].entered[_after("e4")].wait(5), "e4 never looked at ahead"
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("e2e4"), 1)
        assert analyser.ahead_hits == 1
        engines[1].release(_after("e4"))
        got = watch.settled()
    searches = [f for e in engines for f, m in zip(e.calls, e.multipvs, strict=True) if m > 1]
    assert searches.count(_after("e4")) == 1, "searched again"
    assert not any(e.stopped for e in engines), "the promoted search was stopped"
    told = [e for e in got if isinstance(e, Analysis) and e.ply == 1]
    assert told and told[-1].fen == _after("e4") and not told[-1].provisional


def test_a_search_ahead_off_every_path_stops_and_one_still_on_a_path_goes_on() -> None:
    """A move played: the look-ahead's search of a position now on no path is stopped -- its
    answer would never be read -- and the one the game went to goes on, now the game's."""
    lines = {START: [_line("e2e4"), _line("d2d4")]}
    gated = [_after("e4"), _after("d4")]
    engines = [GatedEngine(lines, gated=gated) for _ in range(3)]
    analyser = _analyser(engines, look_ahead=1)
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        for fen in gated:
            watch.until(
                lambda fen=fen: any(e.entered[fen].is_set() for e in engines), f"{fen} ahead"
            )
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("e2e4"), 1)
        watch.until(lambda: any(e.stopped for e in engines), "the search after d4 stopped")
        for engine in engines:
            engine.release(_after("e4"))
        watch.idle()
    stopped = [fen for e in engines for fen in e.stopped]
    assert stopped == [_after("d4")], stopped
    assert analyser.stopped == 1


def test_the_game_has_a_worker_of_its_own() -> None:
    """The first worker takes only the game's own positions: a position the game reaches never
    waits behind the look-ahead. Asked while the look-ahead has work waiting and its worker is
    held, the game's worker turns it down, the other takes it; and a position of the game's is
    the game's worker's at once."""
    lines = {START: [_line("e2e4"), _line("d2d4")]}
    gated = [_after("e4"), _after("d4")]
    engines = [GatedEngine(lines, gated=gated) for _ in range(2)]
    analyser = _analyser(engines, look_ahead=1)
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        watch.until(lambda: engines[1].entered[_after("e4")].is_set(), "e4 looked at ahead")
        watch.idle()
        with analyser._cond:
            assert analyser._pick(0) is None, "the game's worker took the look-ahead"
            ahead = analyser._pick(1)
            assert ahead is not None and ahead.key == _board("d4").fen()
            board = chess.Board()
            board.push_san("Nf3")
            analyser._path[1] = analyser._node(board).key
            game = analyser._pick(0)
            assert game is not None and game.key == analyser._path[1], "the game's position waited"
            del analyser._path[1]
        engines[1].release(_after("e4"))
        engines[1].release(_after("d4"))
    assert not any(fen in engines[0].calls for fen in gated)


def test_a_position_is_told_at_each_depth_of_its_own_search_and_whole_last() -> None:
    """Tai, 2026-09-28: "first usable answer is depth 12 ... the computation continues until max
    depth of 18". Depth 12 is the position's own search's: one search, every depth from 12 told
    as it finishes, provisional, then whole -- a search of its own to depth 12 first cost 27%
    more than the full depth alone (the engine audit, 2026-10-04)."""
    engine = GatedEngine({START: [_line("e2e4")]})
    analyser = _analyser([engine], look_ahead=0)
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        got = [e for e in watch.settled() if isinstance(e, Analysis)]
    assert engine.calls == [START], "one search"
    assert [e.lines[0].depth for e in got] == list(range(EARLY_DEPTH, 19))
    assert [e.provisional for e in got] == [True] * (len(got) - 1) + [False]


def test_the_full_depth_is_whole_once_its_side_searches_are_in() -> None:
    """The coach's lessons wait for the whole analysis (threats, traps): the full depth is told
    provisional when its own search ends, and whole when its side searches are in -- those
    searching from the position's lines, never the position again."""
    engine = GatedEngine({START: [_line("e2e4")]})
    analyser = _analyser([engine], look_ahead=0, threats=True)
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        got = [e for e in watch.settled() if isinstance(e, Analysis) and e.lines[0].depth == 18]
    assert [e.provisional for e in got] == [True, False]
    assert engine.calls.count(START) == 1, "the position searched again for its side searches"
    assert analyser.side_searches == 1


def test_moves_are_judged_in_the_order_they_were_played() -> None:
    """A verdict reads the move before it ("the opponent just erred", "a recapture"): the
    second move's is held until the first's is out, though its own lines came first."""
    lines = {
        START: [_line("e2e4", score=0.55)],
        _after("e4"): [_line("e7e5", score=0.5)],
    }
    engines = [GatedEngine(lines, gated=[START]) for _ in range(2)]
    analyser = _analyser(engines, look_ahead=0)
    watch = AnalyserWatch(analyser)
    with analyser:
        board = chess.Board()
        analyser.submit_move("g", board, board.parse_san("e4"), 1)
        board.push_san("e4")
        analyser.submit_move("g", board, board.parse_san("e5"), 2)
        watch.until(lambda: analyser._nodes[analyser._path[1]].measured is not None, "e4 known")
        assert analyser._next_verdict == 1, "e5 judged before e4"
        for e in engines:
            e.release(START)
        got = watch.settled()
    assert [e.ply for e in got if isinstance(e, MoveAssessment)] == [1, 2]


def test_a_takeback_says_nothing_more_of_the_withdrawn_moves() -> None:
    """Tai, 2026-09-23: "takebacks make the program lag behind by a lot". The moves taken back
    get no verdict and no analysis, and nothing is owed for them; a search made only for them
    stops."""
    gated = [_after("e4", "e5", "Nf3")]
    engine = GatedEngine({}, gated=gated)
    analyser = _analyser([engine], look_ahead=0)
    watch = AnalyserWatch(analyser)
    # The moves in before the worker starts: it takes the newest first, and holds there.
    board = chess.Board()
    for ply, san in enumerate(["e4", "e5", "Nf3"], start=1):
        move = board.parse_san(san)
        analyser.submit_move("g", board, move, ply)
        board.push(move)
    with analyser:
        assert engine.entered[gated[0]].wait(5), "the position after Nf3 never searched"
        analyser.withdraw("g", 1)
        watch.until(lambda: engine.stopped == gated, "the withdrawn move's search stopped")
        got = watch.settled()
    assert [e for e in got if getattr(e, "ply", 0) >= 2] == [], got
    assert analyser.withdrawn == 2 and analyser.pending == 0


def test_a_position_seen_again_is_not_searched_again() -> None:
    """A puzzle tried again, a board re-read: a position searched before is answered from what
    was found, after the game's reset too."""
    engine = GatedEngine({START: [_line("e2e4")]})
    analyser = _analyser([engine], look_ahead=0)
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", chess.Board(), 0)
        watch.settled()
        analyser.reset()
        analyser.submit_position("h", chess.Board(), 0)
        again = watch.settled()
    assert engine.calls == [START]
    assert [e.game_id for e in again if isinstance(e, Analysis)] == ["h"]


def test_a_failed_search_does_not_end_the_session() -> None:
    class Broken(FakeEngine):
        def analyse(self, board: chess.Board, **kwargs: object) -> list[Line]:
            raise RuntimeError("engine gone")

    analyser = Analyser(GameAnnotator(engine=Broken({}), depth=6, with_threats=False))
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_move("g", chess.Board(), chess.Move.from_uci("e2e4"), 1)
        watch.until(lambda: analyser.failures >= 1, "the failure counted")
        assert analyser.last_error is not None and "engine gone" in analyser.last_error
        analyser.submit_position("h", _board("d4"), 0)
        watch.until(lambda: analyser.failures >= 2, "the next search tried, and failed alike")


def test_the_workers_engines_search_on_one_thread_each() -> None:
    """At a fixed depth more threads widen the search instead of speeding it: four took 7.7 times
    the processor time of one, and twice as long, for an answer no better (the engine audit,
    2026-10-04). Each worker's engine has one, and the pool its size."""
    made: list[dict[str, object]] = []

    def make(**kwargs: object) -> Engine:
        made.append(kwargs)
        return FakeEngine({})

    from sidecoach.analysis.score import Thresholds

    analyser = open_analyser(
        Path("stockfish"),
        depth=18,
        multipv=6,
        workers=3,
        hash_mb=64,
        pv_plies=12,
        thresholds=Thresholds(),
        make_engine=make,
    )
    analyser.close()
    assert [m["threads"] for m in made] == [1, 1, 1]
    assert len(analyser.engines) == 3


def test_the_look_ahead_goes_where_the_game_is_likely_to_go() -> None:
    """Tai, 2026-10-04: "an adaptive, dynamic system that is able to maximize both comp eff and
    the chance of catching the opponent's move". A forced line -- one move far better than the
    rest -- is followed deep and its alternatives left alone; a quiet position, its moves about
    equal, has each searched one ply on and nothing behind them (each too unlikely by then)."""
    forced = {
        START: [_line("e2e4", score=0.9), _line("d2d4", score=0.3)],
        _after("e4"): [_line("e7e5", score=0.9), _line("c7c5", score=0.2)],
        _after("e4", "e5"): [_line("g1f3", score=0.9), _line("b1c3", score=0.2)],
    }
    quiet_moves = ("e2e4", "d2d4", "c2c4", "g1f3")
    quiet = {
        START: [_line(m) for m in quiet_moves],
        _after("e4"): [_line("e7e5"), _line("c7c5")],
    }

    def looked_at(lines: dict[str, list[Line]]) -> set[str]:
        analyser = _analyser([FakeEngine(lines)], look_ahead=4)
        watch = AnalyserWatch(analyser)
        with analyser:
            analyser.submit_position("g", chess.Board(), 0)
            watch.idle()
            watch.until(
                lambda: (
                    not analyser._claimed
                    and all(analyser._nodes[k].measured for k in analyser._ahead)
                ),
                "the look-ahead done",
            )
            return {analyser._nodes[k].board.fen() for k in analyser._ahead}

    assert looked_at(forced) == {_after("e4"), _after("e4", "e5"), _after("e4", "e5", "Nf3")}
    quiet_seen = looked_at(quiet)
    assert quiet_seen == {
        _board(chess.Board().san(chess.Move.from_uci(m))).fen() for m in quiet_moves
    }


def test_how_a_side_plays_is_learned_from_this_game_and_forgotten_with_it() -> None:
    """Move accuracy depends on the game (Tai, 2026-10-04): nothing trained, nothing kept
    between games. A player who keeps playing the second line has it made likelier for them as
    the game goes -- the first moves decide it, the start counting for little -- and the next
    game starts over."""
    from sidecoach.analysis.analyser import T_START, _Move

    analyser = _analyser([FakeEngine({})])
    analyser.submit_position("g", chess.Board(), 0)
    lines = [_line("e2e4", score=0.60), _line("d2d4", score=0.52)]
    before = analyser._odds(True, lines)[1]
    for _ in range(4):
        analyser._learn(_Move(1, "a", "b", chess.Move.from_uci("d2d4"), True), lines)
    after = analyser._odds(True, lines)[1]
    assert analyser._t[True] > T_START and after > 2 * before, (before, after)
    assert analyser._t[False] == T_START, "the opponent's is their own"
    analyser.submit_position("h", chess.Board(), 0)  # the next game
    assert analyser._t[True] == T_START and analyser._odds(True, lines)[1] == pytest.approx(before)


def test_a_watched_games_review_searches_nothing_the_game_measured(tmp_path: Path) -> None:
    """A game followed live had every position measured whole -- its own search and its side
    searches, at the review's standard -- and its review measured them all again (the engine
    audit, 2026-10-04). Kept as the game ends, they are the review's: it searches none of them."""
    import io

    import chess.pgn

    from sidecoach.analysis.game_review import measured_from_json, review_game
    from sidecoach.analysis.score import Thresholds
    from sidecoach.cli import _Assessing
    from sidecoach.config import Settings
    from sidecoach.events import GameStarted, MovePlayed, Side
    from sidecoach.profile.store import Store

    assessing = _Assessing(Settings(data_dir=tmp_path), enabled=False)
    analyser = Analyser(GameAnnotator(engine=FakeEngine({}), depth=18, multipv=6))
    assessing.analyser = analyser
    watch = AnalyserWatch(analyser)
    analyser.start()
    assessing.submit(GameStarted(game_id="g", start_fen=START, user_side=Side.WHITE))
    board = chess.Board()
    for ply, san in enumerate(["e4", "e5", "Nf3"], start=1):
        before = board.fen()
        move = board.push_san(san)
        assessing.submit(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=before,
                fen_after=board.fen(),
                by_user=ply % 2 == 1,
            )
        )
    watch.idle()
    store = Store(tmp_path / "profile.sqlite")
    try:
        assert assessing.keep_for_review(store, "g") == 4, "the four positions, whole"
        known = {
            m.fen: m
            for m in map(measured_from_json, store.progress("watched:g", "standard", 18, 6))
        }
    finally:
        store.close()
        analyser.close()
    review_engine = FakeEngine({})
    game = chess.pgn.read_game(io.StringIO("1. e4 e5 2. Nf3 *"))
    assert game is not None
    reviewed = review_game(
        game, [review_engine], depth=18, multipv=6, thresholds=Thresholds(), known=known
    )
    assert len(reviewed.moves) == 3
    assert review_engine.calls == [], "the review searched what the game had measured"


def test_a_live_game_stays_in_the_book_past_its_first_move(tmp_path: Path) -> None:
    """The live path hands the analyser each move's board; the book needs the moves that led
    to it. An Italian Game submitted the way the panel submits it -- a takeback in the middle --
    is book to its last move and named as the whole line names it, not as its first move does."""
    import shutil

    from sidecoach.analysis.openings import OpeningBook
    from sidecoach.cli import _Assessing
    from sidecoach.config import Settings
    from sidecoach.events import GameStarted, MovePlayed, Side, Takeback

    folder = tmp_path / "openings"
    folder.mkdir()
    shutil.copy(Path(__file__).parent / "openings_sample.tsv", folder / "c.tsv")
    book = OpeningBook.load(folder)
    assert book is not None
    assessing = _Assessing(Settings(data_dir=tmp_path), enabled=False)
    assessing.analyser = Analyser(GameAnnotator(engine=FakeEngine({}), book=book), look_ahead=0)
    watch = AnalyserWatch(assessing.analyser)
    assessing.analyser.start()

    def play(board: chess.Board, san: str, ply: int) -> None:
        before = board.fen()
        move = board.push_san(san)
        assessing.submit(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=before,
                fen_after=board.fen(),
                by_user=ply % 2 == 1,
            )
        )

    assessing.submit(GameStarted(game_id="g", start_fen=chess.STARTING_FEN, user_side=Side.WHITE))
    board = chess.Board()
    for ply, san in enumerate(["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "d4"], start=1):
        play(board, san, ply)
    board.pop()  # a takeback of 3.d4: the next move is played from the position before it
    assessing.submit(Takeback(game_id="g", plies=1, fen=board.fen()))
    play(board, "d3", 7)
    labels: dict[int, object] = {}
    openings: dict[int, object] = {}
    for event in watch.settled():
        if isinstance(event, MoveAssessment):
            labels[event.ply] = event.label
        elif isinstance(event, Analysis):
            openings[event.ply] = event.opening
    assessing.analyser.close()
    assert [labels[p] for p in range(1, 8)] == [Label.BOOK] * 7, labels
    assert openings[1] == "B00 King's Pawn Game"
    assert openings[5] == "C50 Italian Game"
    assert openings[7] == "C50 Italian Game: Giuoco Pianissimo", openings


def test_what_the_engine_says_of_a_withdrawn_move_is_not_passed_on(tmp_path: Path) -> None:
    """A verdict on a move taken back came back at its ply and was read as the verdict on the
    move that replaced it; `_Assessing.drain` holds each result against the game as it stands."""
    from sidecoach.cli import _Assessing
    from sidecoach.config import Settings
    from sidecoach.events import GameStarted, MovePlayed, Severity, Side, Takeback

    assessing = _Assessing(Settings(data_dir=tmp_path), enabled=False)
    assessing.analyser = Analyser(
        GameAnnotator(engine=FakeEngine({}), depth=6, with_threats=False), look_ahead=0
    )

    def played(board: chess.Board, san: str, ply: int) -> MovePlayed:
        before = board.fen()
        move = board.push_san(san)
        return MovePlayed(
            game_id="g",
            ply=ply,
            san=san,
            uci=move.uci(),
            fen_before=before,
            fen_after=board.fen(),
            by_user=ply % 2 == 1,
        )

    assessing.submit(GameStarted(game_id="g", start_fen=chess.STARTING_FEN, user_side=Side.WHITE))
    board = chess.Board()
    for ply, san in enumerate(["e4", "e5", "Nf3", "Nc6"], start=1):
        assessing.submit(played(board, san, ply))
    # Nothing searched yet (the workers are not started): 3.Nf3 Nc6 taken back, 3.Bc4 played.
    board.pop()
    board.pop()
    assessing.submit(Takeback(game_id="g", plies=2, fen=board.fen()))
    assessing.submit(played(board, "Bc4", 3))
    # And a verdict on the withdrawn 3.Nf3, as if the engine had finished it just before.
    stale = MoveAssessment(
        game_id="g", ply=3, uci="g1f3", expected_loss=0.0, severity=Severity.NONE, best=[]
    )
    assessing.analyser._results.put(stale)
    watch = AnalyserWatch(assessing.analyser)
    assessing.analyser.start()
    watch.idle()
    got: list[object] = list(assessing.drain())
    assessing.analyser.close()
    verdicts = [e for e in got if isinstance(e, MoveAssessment)]
    assert [(e.ply, e.uci) for e in verdicts] == [(1, "e2e4"), (2, "e7e5"), (3, "f1c4")], verdicts
    assert all(e.ply <= 3 for e in got if isinstance(e, Analysis))


def test_a_position_re_read_mid_search_gets_no_word_on_the_old_reading(tmp_path: Path) -> None:
    """The side to move settled by the clock while the joined position is searched: the reset
    lets go of the search under way, and the glue passes on nothing of a position the game no
    longer stands on, nor of another game (the state review, 2026-10-01). The white reading's
    search is held until something stops it: the test acts mid-search by the engine's state."""
    from sidecoach.cli import _Assessing
    from sidecoach.config import Settings
    from sidecoach.events import GameStarted, Resync, Side

    white = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1"
    black = white.replace(" w ", " b ")
    assessing = _Assessing(Settings(data_dir=tmp_path), enabled=False)
    engine = GatedEngine({}, gated=[white])
    analyser = Analyser(GameAnnotator(engine=engine, depth=18, with_threats=False), look_ahead=0)
    assessing.analyser = analyser
    watch = AnalyserWatch(analyser)
    analyser.start()
    try:
        assessing.submit(
            GameStarted(game_id="g", user_side=Side.WHITE, start_fen=white, kind="position")
        )
        assert engine.entered[white].wait(5), "the search never began"
        assessing.submit(Resync(game_id="g", fen=black, reason="side to move settled by the clock"))
        watch.until(lambda: engine.stopped == [white], "the reset let go of the search")
        watch.idle()
        told = [e.fen for e in assessing.drain() if isinstance(e, Analysis)]
        assert told and set(told) == {black}, told
        word = {"ply": 0, "lines": [], "decisive": False, "only_move": False}
        analyser._results.put(Analysis(game_id="g", fen=white, **word))
        analyser._results.put(Analysis(game_id="h", fen=black, **word))
        assert assessing.drain() == []
    finally:
        assessing.close()


@needs_engine
def test_each_depth_of_a_real_search_is_told_as_it_finishes(
    engine: Engine, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real engine: the position told at depth 12, then at every depth its one search
    finishes, provisional, then whole -- the same position, deeper each time."""
    monkeypatch.setattr(engine, "close", lambda: None)  # the module's Stockfish stays up
    board = chess.Board(SCHOLAR)
    annotator = GameAnnotator(engine=engine, depth=16, multipv=3, with_threats=False)
    analyser = Analyser(annotator, look_ahead=0)
    watch = AnalyserWatch(analyser)
    with analyser:
        analyser.submit_position("g", board, 0)
        got = [e for e in watch.settled() if isinstance(e, Analysis)]
    depths = [e.lines[0].depth for e in got]
    assert [e.provisional for e in got] == [True] * (len(got) - 1) + [False]
    assert depths[0] == EARLY_DEPTH and depths[-1] == 16
    assert depths == sorted(depths) and len(set(depths)) == len(depths), depths
