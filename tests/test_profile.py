"""The profile stage: the store, the cause classifier, the review, the insights, and the
profiler fed the events of a game (DESIGN section 5.6). No engine: the verdicts are
fabricated the way the annotator would emit them, the assessment of a ply before the
analysis of the position it leads to."""

from __future__ import annotations

from typing import Any

import chess
import pytest

from sidecoach.events import (
    Analysis,
    ClockObservation,
    GameEnded,
    GameReview,
    GameStarted,
    Label,
    Line,
    MotifTag,
    MoveAssessment,
    MovePlayed,
    ProfileEvidence,
    ProfileSnapshot,
    Resync,
    Severity,
    Side,
    Takeback,
)
from sidecoach.profile import causes
from sidecoach.profile.insights import insights, motif_words
from sidecoach.profile.profiler import Profiler
from sidecoach.profile.review import UserMove, build_review
from sidecoach.profile.store import CellRow, Store

# ---- the store -------------------------------------------------------------------------


def test_store_keeps_games_moves_evidence_and_cells_in_memory() -> None:
    store = Store(None)
    store.start_session("s1", 0.0)
    store.start_game("g1", 1.0, "white", chess.STARTING_FEN, "s1")
    move_id = store.add_move(
        "g1",
        0,
        1,
        move_number=1,
        san="e4",
        uci="e2e4",
        by_user=True,
        fen_before=chess.STARTING_FEN,
        fen_after="x",
        phase="opening",
        think_ms=1200,
        clock_ms=598_000,
        capture=False,
        gives_check=False,
        material=0,
        ts=1.5,
    )
    store.assess_move(
        move_id,
        label="best",
        severity="none",
        expected_loss=0.0,
        best_uci="e2e4",
        best_san="e4",
        accuracy=99.0,
        tags=["quietMove"],
        available_tags=["fork"],
        depth=14,
    )
    store.analyse_move(move_id, score_after=0.55, allowed_tags=[])
    moves = store.moves("g1")
    assert len(moves) == 1 and moves[0].move_id == move_id
    assert moves[0].label == "best" and moves[0].tags == ["quietMove"]
    assert moves[0].think_ms == 1200 and moves[0].clock_ms == 598_000
    assert moves[0].score_after == 0.55 and moves[0].available_tags == ["fork"]
    second = store.add_move(
        "g1",
        0,
        7,
        move_number=4,
        san="Qd3",
        uci="d1d3",
        by_user=True,
        fen_before=chess.STARTING_FEN,
        fen_after="y",
        phase="middlegame",
        think_ms=None,
        clock_ms=None,
        capture=False,
        gives_check=False,
        material=-3,
        ts=2.0,
    )
    store.assess_move(
        second,
        label="mistake",
        severity="mistake",
        expected_loss=0.15,
        best_uci=None,
        best_san=None,
        accuracy=40.0,
        tags=[],
        available_tags=["fork"],
        depth=14,
    )
    store.add_evidence(
        second,
        game_id="g1",
        segment=0,
        ply=7,
        phase="middlegame",
        expected_loss=0.15,
        cause="missed opponent threat",
        motif="fork",
        time_pressure=False,
    )
    cells = {(c.cause, c.motif, c.phase): c for c in store.cells()}
    assert cells[("missed opponent threat", "fork", "middlegame")].errors == 1
    assert cells[(None, "fork", "middlegame")].chances == 1  # the tactic was on the board
    assert cells[(None, "fork", "opening")].chances == 1
    assert cells[(None, None, "middlegame")].chances == 1
    # The user's correction moves the error to the new cause; the cells follow.
    row = store.correct_evidence("g1", 0, 7, "miscalculated line")
    assert row is not None and row.corrected and row.cause == "miscalculated line"
    cells = {(c.cause, c.motif, c.phase): c for c in store.cells()}
    assert cells[("miscalculated line", "fork", "middlegame")].errors == 1
    assert ("missed opponent threat", "fork", "middlegame") not in cells
    store.self_report(second, "I thought the knight was pinned")
    assert store.evidence("g1")[0].self_report == "I thought the knight was pinned"
    store.end_game("g1", 9.0, "1-0", "checkmate", accuracy=88.0, final_score=1.0)
    assert store.games_count() == 1
    game = store.game("g1")
    assert game is not None and game.accuracy == 88.0 and game.final_score == 1.0
    store.set_game("g1", time_control="10 min")
    game = store.game("g1")
    assert game is not None and game.time_control == "10 min"
    store.end_session("s1", 10.0, "one game")
    assert store.last_session(before="s2") is not None
    assert store.last_session(before="s1") is None
    store.close()


def test_the_coachs_queries_are_read_only_and_bounded() -> None:
    from sidecoach.profile.store import QueryRefused

    store = Store(None)
    store.start_game("g1", 1.0, "white", chess.STARTING_FEN)
    names, rows = store.query("SELECT game_id, user_side FROM games")
    assert names == ["game_id", "user_side"] and rows == [("g1", "white")]
    names, rows = store.query("WITH g AS (SELECT * FROM games) SELECT COUNT(*) AS n FROM g")
    assert rows == [(1,)]
    for bad in (
        "DELETE FROM games",
        "UPDATE games SET result = '1-0'",
        "INSERT INTO sessions (session_id, started_at) VALUES ('x', 0)",
        "DROP TABLE games",
        "PRAGMA table_info(games)",
        "ATTACH DATABASE 'x.db' AS x",
        "SELECT 1; DELETE FROM games",
        "",
        "SELECT * FROM nowhere",
    ):
        with pytest.raises(QueryRefused):
            store.query(bad)
    assert store.game("g1") is not None  # nothing was written
    # A runaway query is interrupted, and the row cap holds.
    with pytest.raises(QueryRefused):
        store.query(
            "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n)"
            " SELECT COUNT(*) FROM n",
            seconds=0.2,
        )
    _, rows = store.query(
        "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n LIMIT 500) SELECT x FROM n",
        rows=10,
    )
    assert len(rows) == 10
    store.close()


def test_store_writes_a_file(tmp_path) -> None:
    store = Store(tmp_path / "deep" / "profile.sqlite")
    store.start_game("g", 0.0, "black", chess.STARTING_FEN)
    store.close()
    again = Store(tmp_path / "deep" / "profile.sqlite")
    assert again.game("g") is not None
    again.close()


# ---- causes ----------------------------------------------------------------------------

FORK = MotifTag(name="fork", squares=["e5", "c6", "g6"])
SAC = MotifTag(name="sacrifice", squares=["h7"])
CAPTURE = MotifTag(name="capture", squares=["e5"])


def test_time_pressure_comes_first() -> None:
    board = chess.Board()
    cause, motif = causes.classify(
        board, chess.Move.from_uci("e2e4"), allowed_tags=[FORK], clock_ms=8_000
    )
    assert cause == causes.TIME_PRESSURE and motif == "fork"


def test_a_tactic_allowed_is_a_missed_threat() -> None:
    board = chess.Board()
    cause, motif = causes.classify(board, chess.Move.from_uci("e2e4"), allowed_tags=[FORK])
    assert (cause, motif) == (causes.MISSED_THREAT, "fork")
    # A mate for the opponent, even without a tag.
    cause, motif = causes.classify(board, chess.Move.from_uci("f2f3"), reply_mate=2)
    assert (cause, motif) == (causes.MISSED_THREAT, "mate")


def test_a_piece_left_to_be_taken_is_a_missed_threat() -> None:
    # 1.e4 e5 2.Nf3 Qh4?? 3.Nxh4: the queen put where the knight takes it for nothing.
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    move = chess.Move.from_uci("d8h4")
    cause, motif = causes.classify(board, move, reply="f3h4")
    assert (cause, motif) == (causes.MISSED_THREAT, "hangingPiece")
    assert causes.wins_material(_after(board, move), chess.Move.from_uci("f3h4"))


def _after(board: chess.Board, move: chess.Move) -> chess.Board:
    after = board.copy(stack=False)
    after.push(move)
    return after


def test_an_attempted_tactic_or_a_capture_that_failed_is_a_miscalculation() -> None:
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    move = chess.Move.from_uci("c4f7")  # Bxf7+?, a sacrifice tagged as such
    cause, motif = causes.classify(board, move, played_tags=[SAC, CAPTURE])
    assert (cause, motif) == (causes.MISCALCULATED, "sacrifice")
    # A plain capture that lost ground, with a tactic that was there instead.
    cause, motif = causes.classify(board, move, played_tags=[CAPTURE], available_tags=[FORK])
    assert (cause, motif) == (causes.MISCALCULATED, "fork")


def test_a_tactic_left_on_the_board_was_never_considered() -> None:
    board = chess.Board()
    cause, motif = causes.classify(board, chess.Move.from_uci("a2a3"), available_tags=[FORK])
    assert (cause, motif) == (causes.NOT_CONSIDERED, "fork")


def test_the_phase_names_the_rest() -> None:
    board = chess.Board()
    quiet = chess.Move.from_uci("a2a3")
    assert causes.classify(board, quiet, phase="endgame") == (causes.ENDGAME, None)
    assert causes.classify(board, quiet, phase="opening") == (causes.OPENING_GAP, None)
    assert causes.classify(board, quiet, phase="middlegame", left_book=True) == (
        causes.OPENING_GAP,
        None,
    )
    assert causes.classify(board, quiet, phase="middlegame") == (causes.POSITIONAL, None)


def test_only_tactics_count_as_motifs() -> None:
    assert causes.tactical([CAPTURE]) is None
    assert causes.tactical([CAPTURE, FORK]) == "fork"
    assert "capture" not in causes.TACTICAL and "backRankMate" in causes.TACTICAL


# ---- the review ------------------------------------------------------------------------

START = chess.STARTING_FEN


def _user_move(ply: int, san: str, uci: str, fen_before: str, **kw: Any) -> UserMove:
    base: dict[str, Any] = dict(
        label="good",
        severity="none",
        expected_loss=0.01,
        best_uci=None,
        accuracy=90.0,
        phase="middlegame",
    )
    base.update(kw)
    return UserMove(ply=ply, san=san, uci=uci, fen_before=fen_before, **base)  # type: ignore[arg-type]


def test_the_review_tallies_and_finds_the_turning_points() -> None:
    after_e4 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
    moves = [
        _user_move(1, "e4", "e2e4", START, label="book", phase="opening", accuracy=100.0),
        _user_move(
            3,
            "Qh5",
            "d1h5",
            after_e4,
            label="mistake",
            severity="mistake",
            expected_loss=0.12,
            best_uci="g1f3",
            cause=causes.POSITIONAL,
            accuracy=60.0,
        ),
        _user_move(
            5,
            "Qxf7",
            "h5f7",
            after_e4,
            label="blunder",
            severity="blunder",
            expected_loss=0.4,
            best_uci="g1f3",
            cause=causes.MISSED_THREAT,
            motif="hangingPiece",
            accuracy=10.0,
        ),
        _user_move(7, "Nf3", "g1f3", after_e4, label="best", accuracy=100.0, phase="endgame"),
    ]
    review = build_review(
        game_id="g",
        user_side=Side.WHITE,
        result="0-1",
        reason="resignation",
        opening="King's Pawn",
        moves=moves,
        fullmoves=12,
    )
    assert review.moves == 4
    assert review.tallies == {"book": 1, "mistake": 1, "blunder": 1, "best": 1}
    assert review.accuracy == 67.5
    assert [tp.expected_loss for tp in review.turning_points] == [0.4, 0.12]
    assert review.turning_points[0].san == "2. Qxf7" and review.turning_points[0].best_san == "Nf3"
    assert [p.phase for p in review.phases] == ["opening", "middlegame", "endgame"]
    assert review.causes == {causes.MISSED_THREAT: 0.4, causes.POSITIONAL: 0.12}
    text = " ".join(review.lines)
    assert "You played white and lost by resignation in 12 moves." in text
    assert "The opening was the King's Pawn." in text
    assert "Accuracy 67.5." in text and "1 best, 1 book, 1 mistake, 1 blunder" in text
    assert "Ground lost at 2. Qxf7, a blunder costing 0.40; Nf3 was better" in text
    assert "(missed opponent threat: hanging piece)" in text
    assert "Most ground went to missed opponent threat (0.40)." in text


def test_a_clean_game_says_so() -> None:
    review = build_review(
        game_id="g",
        user_side=Side.BLACK,
        result="1/2-1/2",
        reason="draw agreed",
        opening=None,
        moves=[_user_move(2, "e5", "e7e5", START, label="best", accuracy=100.0)],
        fullmoves=30,
    )
    assert "You played black and drew by agreement in 30 moves." in review.lines[0]
    assert "No move cost you real ground." in review.lines


def test_motif_words() -> None:
    assert motif_words("hangingPiece") == "hanging piece"
    assert motif_words("mateIn2") == "mate in 2"
    assert motif_words("attackingF2F7") == "attacking f2/f7"
    assert motif_words("xRayAttack") == "x-ray attack"


# ---- insights --------------------------------------------------------------------------


def _cell(
    cause: str | None,
    motif: str | None,
    phase: str | None,
    errors: int = 0,
    chances: int = 0,
    loss: float = 0.0,
) -> CellRow:
    return CellRow(
        cause=cause, motif=motif, phase=phase, errors=errors, chances=chances, expected_loss=loss
    )


def test_insights_need_evidence_before_they_speak() -> None:
    assert insights([_cell(causes.POSITIONAL, None, "middlegame", errors=2, loss=0.3)], 1) == []


def test_insights_name_the_biggest_cost_and_whether_it_is_a_gap() -> None:
    rows = [
        _cell(causes.MISSED_THREAT, "fork", "middlegame", errors=3, loss=0.6),
        _cell(causes.MISSED_THREAT, "pin", "endgame", errors=1, loss=0.1),
        _cell(causes.POSITIONAL, None, "middlegame", errors=1, loss=0.1),
        _cell(None, "fork", "middlegame", chances=6),
        _cell(None, None, "middlegame", chances=30),
        _cell(None, None, "endgame", chances=12, loss=0.0),
    ]
    lines = insights(rows, games=3)
    assert lines[0] == (
        "Your biggest cost over 3 games is missed opponent threat: 4 mistakes, 0.70 expected score."
    )
    assert lines[1] == "That looks like a gap: it shows up as fork in the middlegame."
    assert any(x.startswith("You miss a fork about") and "(3 of 6)" in x for x in lines)
    assert any(x.startswith("The middlegame costs you most per move") for x in lines)


def test_a_cost_spread_across_positions_is_a_habit() -> None:
    rows = [
        _cell(causes.POSITIONAL, None, "opening", errors=1, loss=0.2),
        _cell(causes.POSITIONAL, None, "middlegame", errors=1, loss=0.2),
        _cell(causes.POSITIONAL, None, "endgame", errors=1, loss=0.2),
    ]
    lines = insights(rows, games=1)
    assert lines[1] == "That looks like a habit: it shows up across 3 kinds of position."


# ---- the profiler ----------------------------------------------------------------------


class Feed:
    """Events of a game the way the pipeline and the annotator emit them."""

    def __init__(self, user_side: Side = Side.WHITE) -> None:
        self.store = Store(None)
        self.profiler = Profiler(self.store, session_id="s", ts=0.0)
        self.board = chess.Board()
        self.ply = 0
        self.ts = 0.0
        self.user_side = user_side
        self.out: list = []
        self.send(GameStarted(game_id="g", user_side=user_side, start_fen=chess.STARTING_FEN))

    def send(self, event) -> list:
        self.ts += 1.0
        got = self.profiler.on_event(event, self.ts)
        self.out += got
        return got

    def play(
        self,
        san: str,
        *,
        loss: float = 0.0,
        label: Label = Label.GOOD,
        tags: list[MotifTag] | None = None,
        after_tags: list[MotifTag] | None = None,
        reply: str | None = None,
        reply_mate: int | None = None,
        analysis: bool = True,
    ) -> list:
        move = self.board.parse_san(san)
        fen_before = self.board.fen()
        by_user = (self.board.turn == chess.WHITE) == (self.user_side is Side.WHITE)
        self.board.push(move)
        self.ply += 1
        got = self.send(
            MovePlayed(
                game_id="g",
                ply=self.ply,
                san=san,
                uci=move.uci(),
                fen_before=fen_before,
                fen_after=self.board.fen(),
                by_user=by_user,
                think_ms=3000,
            )
        )
        severity = (
            Severity.BLUNDER
            if loss >= 0.28
            else Severity.MISTAKE
            if loss >= 0.10
            else Severity.INACCURACY
            if loss >= 0.04
            else Severity.NONE
        )
        got += self.send(
            MoveAssessment(
                game_id="g",
                ply=self.ply,
                uci=move.uci(),
                expected_loss=loss,
                severity=severity,
                best=["g1f3"] if loss else [move.uci()],
                tags=tags or [],
                depth=14,
                label=label,
                accuracy=max(0.0, 100.0 - loss * 200),
            )
        )
        if analysis:
            lines = (
                []
                if self.board.is_game_over()
                else [
                    Line(
                        moves=[reply] if reply else [],
                        depth=14,
                        expected_score=0.5,
                        mate=reply_mate,
                    )
                ]
            )
            got += self.send(
                Analysis(
                    game_id="g",
                    fen=self.board.fen(),
                    ply=self.ply,
                    lines=lines,
                    decisive=False,
                    only_move=False,
                    tags=after_tags or [],
                )
            )
        return got

    def end(self, result: str = "1-0", reason: str = "resignation") -> list:
        return self.send(
            GameEnded(
                game_id="g", result=result, reason=reason, pgn="1. e4", final_fen=self.board.fen()
            )
        )


def test_a_mistake_becomes_evidence_once_its_analysis_arrives() -> None:
    f = Feed()
    f.play("e4", label=Label.BOOK)
    f.play("e5", label=Label.BOOK)
    got = f.play("Qh5", loss=0.12, label=Label.MISTAKE, after_tags=[FORK])
    evidence = [e for e in got if isinstance(e, ProfileEvidence)]
    assert len(evidence) == 1
    assert evidence[0].cause == causes.MISSED_THREAT and evidence[0].motif == "fork"
    assert evidence[0].ply == 3 and evidence[0].phase == "opening"
    rows = f.store.evidence("g")
    assert len(rows) == 1 and rows[0].cause == causes.MISSED_THREAT
    cells = {(c.cause, c.motif, c.phase): c for c in f.store.cells()}
    assert cells[(causes.MISSED_THREAT, "fork", "opening")].errors == 1
    assert cells[(None, None, "opening")].chances == 2  # the player's two judged moves


def test_the_opponents_mistakes_and_fine_moves_are_not_evidence() -> None:
    f = Feed()
    f.play("e4")
    got = f.play("f6", loss=0.3, label=Label.BLUNDER, after_tags=[FORK])  # black's, not ours
    assert not any(isinstance(e, ProfileEvidence) for e in got)
    got = f.play("Nf3", loss=0.0, label=Label.BEST)
    assert not any(isinstance(e, ProfileEvidence) for e in got)
    assert f.store.evidence("g") == []


def test_the_clock_makes_it_time_pressure() -> None:
    f = Feed()
    f.play("e4")
    f.play("e5")
    f.send(ClockObservation(white_ms=6_000, black_ms=200_000, running=Side.WHITE))
    got = f.play("Qh5", loss=0.12, label=Label.MISTAKE, after_tags=[FORK])
    evidence = [e for e in got if isinstance(e, ProfileEvidence)]
    assert evidence[0].cause == causes.TIME_PRESSURE and evidence[0].time_pressure


def test_the_game_ends_with_a_review_and_a_snapshot() -> None:
    f = Feed()
    f.play("e4", label=Label.BOOK)
    f.play("e5", label=Label.BOOK)
    f.play("Qh5", loss=0.12, label=Label.MISTAKE, after_tags=[FORK])
    f.play("Nc6")
    f.play("Bc4", label=Label.BEST)
    f.play("Nf6")
    f.play("Qxf7#", label=Label.BEST)
    got = f.end("1-0", "game over on board")  # the reconciler's words for a mate
    review = next(e for e in got if isinstance(e, GameReview))
    snapshot = next(e for e in got if isinstance(e, ProfileSnapshot))
    assert review.result == "1-0" and review.moves == 4
    assert review.tallies == {"book": 1, "mistake": 1, "best": 2}
    assert len(review.turning_points) == 1 and review.turning_points[0].san == "2. Qh5"
    assert review.turning_points[0].cause == causes.MISSED_THREAT
    assert review.lines[0] == "You played white and won by checkmate in 4 moves."
    assert snapshot.games == 1 and snapshot.insights == []  # one mistake makes no weakness
    assert any(c.cause == causes.MISSED_THREAT for c in snapshot.cells)
    game = f.store.game("g")
    assert game is not None and game.result == "1-0" and game.accuracy == review.accuracy
    assert len(f.store.moves("g")) == 7
    # After the end, moves are not taken and a second end does nothing.
    assert f.end() == []


def test_a_takeback_withdraws_the_moves_and_their_evidence() -> None:
    f = Feed()
    f.play("e4")
    f.play("e5")
    f.play("Qh5", loss=0.12, label=Label.MISTAKE, after_tags=[FORK])
    assert len(f.store.evidence("g")) == 1
    f.board.pop()
    f.ply -= 1
    f.send(Takeback(game_id="g", plies=1, fen=f.board.fen()))
    assert f.store.evidence("g") == []
    assert [m.san for m in f.store.moves("g")] == ["e4", "e5"]
    assert [m.san for m in f.store.moves("g", withdrawn=True)] == ["e4", "e5", "Qh5"]
    cells = {(c.cause, c.motif, c.phase): c for c in f.store.cells()}
    cell = cells.get((causes.MISSED_THREAT, "fork", "opening"))
    assert cell is None or cell.errors == 0
    # Play on: the ply number is reused, the record stays consistent.
    f.play("Nf3", label=Label.BEST)
    assert [m.san for m in f.store.moves("g")] == ["e4", "e5", "Nf3"]


def test_a_resync_starts_a_new_segment_and_keeps_the_earlier_moves() -> None:
    f = Feed()
    f.play("e4")
    f.play("e5")
    f.send(Resync(game_id="g", fen=f.board.fen(), reason="test"))
    f.ply = 0
    f.play("Nf3", label=Label.BEST)
    rows = f.store.moves("g")
    assert [(m.segment, m.ply, m.san) for m in rows] == [(0, 1, "e4"), (0, 2, "e5"), (1, 1, "Nf3")]
    review = next(e for e in f.end() if isinstance(e, GameReview))
    assert review.moves == 2  # e4 and Nf3, both the player's


def test_the_session_summary_carries_to_the_next_session() -> None:
    f = Feed()
    f.play("e4")
    f.play("e5")
    f.play("Qh5", loss=0.12, label=Label.MISTAKE, after_tags=[FORK])
    f.end("0-1", "resignation")
    summary = f.profiler.end_session(99.0)
    assert summary is not None
    assert summary.startswith("1 game (1 lost), accuracy")
    assert "most ground went to missed opponent threat (0.12)" in summary
    assert "the worst moment was 2. Qh5, Nf3 was better" in summary
    later = Profiler(f.store, session_id="s2", ts=100.0)
    assert later.snapshot().last_session == summary


def test_a_correction_moves_the_evidence() -> None:
    f = Feed()
    f.play("e4")
    f.play("e5")
    f.play("Qh5", loss=0.12, label=Label.MISTAKE, after_tags=[FORK])
    assert f.profiler.correct("g", 0, 3, causes.MISCALCULATED)
    assert f.store.evidence("g")[0].cause == causes.MISCALCULATED
    review = next(e for e in f.end() if isinstance(e, GameReview))
    assert review.turning_points[0].cause == causes.MISCALCULATED
    assert not f.profiler.correct("g", 0, 99, causes.MISCALCULATED)


@pytest.mark.parametrize("side", [Side.WHITE, Side.BLACK])
def test_the_player_may_be_either_colour(side: Side) -> None:
    f = Feed(side)
    f.play("e4", loss=0.12, label=Label.MISTAKE, after_tags=[FORK])
    f.play("e5", loss=0.12, label=Label.MISTAKE, after_tags=[FORK])
    rows = f.store.evidence("g")
    assert [r.ply for r in rows] == [1 if side is Side.WHITE else 2]


def test_the_review_waits_for_the_engines_last_verdicts() -> None:
    f = Feed()
    f.play("e4")
    f.play("e5")
    # The player's last move reaches the profile before the engine has judged it.
    move = f.board.parse_san("Nf3")
    fen_before = f.board.fen()
    f.board.push(move)
    f.ply += 1
    f.send(
        MovePlayed(
            game_id="g",
            ply=f.ply,
            san="Nf3",
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=f.board.fen(),
            by_user=True,
            think_ms=100,
        )
    )
    got = f.end("1-0", "resignation")
    assert not any(isinstance(e, GameReview) for e in got)
    assert f.profiler.tick(f.ts + 10.0) == []  # still within the grace period
    # The verdict arrives: the review follows at once, with the move counted.
    got = f.send(
        MoveAssessment(
            game_id="g",
            ply=f.ply,
            uci=move.uci(),
            expected_loss=0.0,
            severity=Severity.NONE,
            best=[move.uci()],
            depth=14,
            label=Label.BEST,
            accuracy=100.0,
        )
    )
    review = next(e for e in got if isinstance(e, GameReview))
    assert review.moves == 2 and review.tallies == {"good": 1, "best": 1}
    assert f.profiler.tick(f.ts + 1000.0) == []  # reviewed once


def test_a_review_left_waiting_is_built_when_the_grace_period_is_up_or_at_the_close() -> None:
    from sidecoach.profile.profiler import REVIEW_GRACE_S

    f = Feed()
    f.play("e4")
    f.play("e5")
    f.send(
        MovePlayed(
            game_id="g",
            ply=3,
            san="Nf3",
            uci="g1f3",
            fen_before=f.board.fen(),
            fen_after=f.board.fen(),
            by_user=True,
            think_ms=100,
        )
    )
    assert not any(isinstance(e, GameReview) for e in f.end())
    got = f.profiler.tick(f.ts + REVIEW_GRACE_S + 1)
    review = next(e for e in got if isinstance(e, GameReview))
    assert review.moves == 1  # the judged move only
    # And a session that closes flushes a waiting review instead of dropping it.
    g = Feed()
    g.play("e4")
    g.send(
        MovePlayed(
            game_id="g",
            ply=2,
            san="e5",
            uci="e7e5",
            fen_before=g.board.fen(),
            fen_after=g.board.fen(),
            by_user=False,
            think_ms=100,
        )
    )
    g.send(
        MovePlayed(
            game_id="g",
            ply=3,
            san="Nf3",
            uci="g1f3",
            fen_before=g.board.fen(),
            fen_after=g.board.fen(),
            by_user=True,
            think_ms=100,
        )
    )
    assert not any(isinstance(e, GameReview) for e in g.end())
    assert g.profiler.end_session(g.ts + 1) is not None
    assert g.store.game("g") is not None and g.store.game("g").accuracy is not None  # type: ignore[union-attr]


def test_the_next_game_flushes_the_last_games_review() -> None:
    f = Feed()
    f.play("e4")
    f.send(
        MovePlayed(
            game_id="g",
            ply=2,
            san="e5",
            uci="e7e5",
            fen_before=f.board.fen(),
            fen_after=f.board.fen(),
            by_user=False,
            think_ms=100,
        )
    )
    f.send(
        MovePlayed(
            game_id="g",
            ply=3,
            san="Nf3",
            uci="g1f3",
            fen_before=f.board.fen(),
            fen_after=f.board.fen(),
            by_user=True,
            think_ms=100,
        )
    )
    assert not any(isinstance(e, GameReview) for e in f.end())
    got = f.send(GameStarted(game_id="g2", user_side=Side.WHITE, start_fen=chess.STARTING_FEN))
    assert any(isinstance(e, GameReview) and e.game_id == "g" for e in got)


# ---- the coach's own queries -------------------------------------------------------------


class ScriptedProvider:
    name = "scripted"

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[str, list]] = []

    def healthcheck(self) -> tuple[bool, str]:
        return True, "scripted"

    def chat(
        self,
        system: str,
        messages,
        *,
        max_tokens: int = 900,
        temperature: float = 0.7,
        thinking: bool = False,
    ):
        from sidecoach.llm.base import Completion

        self.prompts.append((system, list(messages)))
        return Completion(text=self.replies.pop(0) if self.replies else "")


def _played_store() -> Store:
    """A store with one judged game in it, the way the profiler writes one."""
    f = Feed()
    f.send(ClockObservation(white_ms=600_000, black_ms=600_000, running=Side.WHITE))
    f.play("e4", label=Label.BOOK)
    f.play("e5", label=Label.BOOK)
    f.play("Qh5", loss=0.12, label=Label.MISTAKE, after_tags=[FORK])
    f.play("Nc6")
    f.send(ClockObservation(white_ms=20_000, black_ms=500_000, running=Side.WHITE))
    f.play("Bc4", label=Label.BEST, tags=[], after_tags=[])
    f.play("Nf6", loss=0.3, label=Label.BLUNDER)
    f.play("Qxf7#", label=Label.BEST)
    f.end("1-0", "game over on board")
    f.profiler.end_session(f.ts + 1)
    return f.store


def test_every_worked_query_in_the_schema_doc_runs() -> None:
    from sidecoach.profile.ask import EXAMPLES, SCHEMA_DOC, system_message

    store = _played_store()
    for question, sql in EXAMPLES:
        columns, rows = store.query(sql)
        assert columns, question
    # The doc names every column the tables have, and no other.
    import re

    for table in ("games", "moves", "evidence", "dialogue", "sessions"):
        names, _ = store.query(f"SELECT * FROM {table} LIMIT 0")
        block = SCHEMA_DOC.split(f"TABLE {table}", 1)[1].split("\nTABLE ", 1)[0]
        for name in names:
            assert re.search(rf"\b{name}\b", block), f"{table}.{name} is not documented"
    assert "json_each" in system_message()


def test_the_record_carries_what_a_question_needs() -> None:
    store = _played_store()
    moves = store.moves("g")
    assert [m.phase for m in moves][:3] == ["opening", "opening", "opening"]
    assert moves[0].clock_ms == 600_000 and moves[4].clock_ms == 20_000
    assert moves[0].move_number == 1 and moves[1].move_number == 1 and moves[2].move_number == 2
    assert all(m.score_after is not None for m in moves)
    assert moves[-1].score_after == 1.0  # mate: the player won
    assert moves[-1].gives_check and moves[-1].capture
    assert moves[2].allowed_tags == ["fork"] and moves[3].available_tags == ["fork"]
    assert moves[2].best_san == "Nf3"
    game = store.game("g")
    assert game is not None and game.time_control == "10 min" and game.final_score == 1.0
    assert game.reason == "checkmate"


def test_the_model_queries_then_answers_in_prose() -> None:
    from sidecoach.profile.ask import answer

    store = _played_store()
    provider = ScriptedProvider(
        [
            "```sql\nSELECT phase, COUNT(*) AS n FROM evidence GROUP BY phase\n```",
            "Your one mistake so far was in the opening, so there is not much of a pattern yet.",
        ]
    )
    got = answer("why do I lose in the middlegame?", store, provider)  # type: ignore[arg-type]
    assert got.text.startswith("Your one mistake") and not got.failed
    assert got.calls == 2 and len(got.exchanges) == 1
    assert got.exchanges[0].columns == ["phase", "n"] and got.exchanges[0].rows == [("opening", 1)]
    system, messages = provider.prompts[1]
    assert "TABLE moves" in system and "Worked queries" in system
    assert messages[0]["content"] == "Question: why do I lose in the middlegame?"
    assert "Rows:\nphase | n\nopening | 1" in messages[2]["content"]


def test_a_refused_query_goes_back_to_the_model_and_the_rounds_are_bounded() -> None:
    from sidecoach.profile.ask import answer

    store = _played_store()
    provider = ScriptedProvider(
        [
            "```sql\nDELETE FROM games\n```",
            "```sql\nSELECT COUNT(*) AS games FROM games\n```",
            "You have played 1 game.",
        ]
    )
    got = answer("how many games?", store, provider)  # type: ignore[arg-type]
    assert got.text == "You have played 1 game." and got.calls == 3
    assert got.exchanges[0].error and "only SELECT" in got.exchanges[0].error
    assert "Query refused" in provider.prompts[1][1][2]["content"]
    assert store.games_count() == 1
    # A model that never writes prose: the rounds run out and the last rows stand in.
    provider = ScriptedProvider(["```sql\nSELECT COUNT(*) AS games FROM games\n```"] * 6)
    got = answer("how many games?", store, provider, rounds=2)  # type: ignore[arg-type]
    assert got.failed and got.calls == 3 and got.text == "From the record: games 1."
    assert "Answer now in prose" in provider.prompts[2][1][-1]["content"]


# ---- self-reports ----------------------------------------------------------------------


def test_the_players_words_after_a_lesson_are_kept_beside_the_mistake() -> None:
    from sidecoach.events import CoachMessage, DisclosureLevel, Trigger, UserUtterance

    f = Feed()
    f.play("e4")
    f.play("e5")
    f.play("Qh5", loss=0.12, label=Label.MISTAKE, after_tags=[FORK])
    assert f.store.evidence("g")[0].cause == causes.MISSED_THREAT
    # The lesson on 2. Qh5 is spoken from the position it was played in (ply 2).
    f.send(
        CoachMessage(
            text="Qh5 cost you; Nf3 was better. What did you see there?",
            level=DisclosureLevel.LINE,
            trigger=Trigger.POST_MISTAKE,
            game_id="g",
            ply=2,
        )
    )
    f.send(
        UserUtterance(text="I saw the knight coming but thought the queen was safe", source="voice")
    )
    row = f.store.evidence("g")[0]
    assert row.self_report == "I saw the knight coming but thought the queen was safe"
    assert row.cause == causes.MISCALCULATED and row.corrected
    # Only the first words after the lesson count, and only soon after it.
    f.send(UserUtterance(text="I was rushing", source="text"))
    assert f.store.evidence("g")[0].cause == causes.MISCALCULATED
    f.send(
        CoachMessage(
            text="lesson",
            level=DisclosureLevel.LINE,
            trigger=Trigger.POST_MISTAKE,
            game_id="g",
            ply=2,
        )
    )
    f.ts += 120.0
    f.send(UserUtterance(text="I was rushing", source="text"))
    assert f.store.evidence("g")[0].cause == causes.MISCALCULATED


def test_words_that_say_nothing_change_nothing() -> None:
    assert causes.from_words("ok") is None
    assert causes.from_words("what should I play now?") is None
    assert causes.from_words("I completely missed it") == causes.MISSED_THREAT
    assert causes.from_words("no time, I premoved") == causes.TIME_PRESSURE
