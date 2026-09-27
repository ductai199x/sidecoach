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


def test_a_chance_let_go_with_a_tactic_on_the_board_is_the_tactic_never_considered() -> None:
    """Even when the move also allowed one: the tactic-blind student was told it kept
    missing their threats and wrote down the threat habit (the simulated students)."""
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    cause, motif = causes.classify(
        board, move, available_tags=[FORK], allowed_tags=[FORK], missed_chance=True
    )
    assert (cause, motif) == (causes.NOT_CONSIDERED, "fork")
    # Not a miss: the tactic allowed comes first, as before.
    cause, _ = causes.classify(board, move, available_tags=[FORK], allowed_tags=[FORK])
    assert cause == causes.MISSED_THREAT
    # A mate allowed is the threat whatever else.
    cause, _ = causes.classify(board, move, available_tags=[FORK], reply_mate=1, missed_chance=True)
    assert cause == causes.MISSED_THREAT


def test_a_capture_or_check_that_comes_out_behind_is_a_line_miscalculated() -> None:
    """Even when the reply it allows wins material: the player began it. As a threat missed,
    the greedy student heard "what their moves threatened" after every game."""
    # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nd4 4.Nxe5?, allowing the fork ...Qg5.
    board = chess.Board("r1bqkbnr/pppp1ppp/8/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    cause, motif = causes.classify(board, chess.Move.from_uci("f3e5"), allowed_tags=[FORK])
    assert (cause, motif) == (causes.MISCALCULATED, "fork")
    # A mate allowed is still the threat.
    cause, _ = causes.classify(board, chess.Move.from_uci("f3e5"), reply_mate=2)
    assert cause == causes.MISSED_THREAT


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


def test_a_piece_guarded_only_by_a_pinned_piece_is_left_to_be_taken() -> None:
    # Be4?? beside the f6 knight: the c3 knight guards e4, but the b4 bishop pins it to its
    # king, so it cannot take back and ...Nxe4 wins the bishop for nothing.
    board = chess.Board("4k3/8/5n2/8/1b6/2N2B2/8/4K3 w - - 0 1")
    move = chess.Move.from_uci("f3e4")
    assert causes.wins_material(_after(board, move), chess.Move.from_uci("f6e4"))
    cause, motif = causes.classify(board, move, reply="f6e4")
    assert (cause, motif) == (causes.MISSED_THREAT, "hangingPiece")
    # Unpinned, the knight takes back: a knight for a bishop wins nothing.
    unpinned = chess.Board("4k3/8/5n2/8/4B3/2N5/8/4K3 b - - 0 1")
    assert not causes.wins_material(unpinned, chess.Move.from_uci("f6e4"))


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


def test_a_cause_behind_two_mistakes_is_named_second_with_the_habit_that_fixes_it() -> None:
    """Asked "What do I keep missing?", the reviews named the move that cost most and never
    the pattern (the simulated students, 2026-09-26). Second, after the result, so the
    offline review's first four lines carry it."""
    after_e4 = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"

    def missed(ply: int, loss: float, cause: str = causes.MISSED_THREAT) -> UserMove:
        return _user_move(
            ply, "Qh5", "d1h5", after_e4, label="mistake", severity="mistake",
            expected_loss=loss, cause=cause,
        )  # fmt: skip

    def review(moves: list[UserMove]):
        return build_review(
            game_id="g",
            user_side=Side.WHITE,
            result="0-1",
            reason=None,
            opening="King's Pawn",
            moves=moves,
            fullmoves=30,
        )

    lines = review([missed(3, 0.2), missed(9, 0.1), missed(15, 0.3, causes.POSITIONAL)]).lines
    assert lines[1] == (
        "What you kept missing: what their moves threatened -- 2 of your 3 mistakes. The habit"
        " that fixes it: before each move, look at what their last move attacks, and at the"
        " captures and checks your move would leave them."
    )
    # One mistake of each kind is no pattern.
    lines = review([missed(3, 0.2), missed(15, 0.3, causes.POSITIONAL)]).lines
    assert not any(line.startswith("What you kept missing") for line in lines)


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


def test_a_position_is_coached_but_never_recorded() -> None:
    """A puzzle or an analysis board begins as a GameStarted of kind "position" and is
    over within a few plies: the profile keeps no game, no evidence, no review and no
    session count for it (a solved puzzle stood in the record as a game won by
    checkmate in three moves at 100% accuracy, 2026-09-20). One that runs on is a game
    joined late; see the test below."""
    f = Feed()
    f.play("e4")
    puzzle = chess.Board("r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R b KQkq - 0 1")
    f.send(GameStarted(game_id="p", user_side=Side.BLACK, start_fen=puzzle.fen(), kind="position"))
    move = puzzle.parse_san("d4")
    fen_before = puzzle.fen()
    puzzle.push(move)
    f.send(
        MovePlayed(
            game_id="p",
            ply=1,
            san="d4",
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=puzzle.fen(),
            by_user=True,
            think_ms=2000,
        )
    )
    f.send(
        MoveAssessment(
            game_id="p",
            ply=1,
            uci=move.uci(),
            expected_loss=0.3,
            severity=Severity.BLUNDER,
            best=["e5e4"],
            label=Label.BLUNDER,
        )
    )
    ended = f.send(GameEnded(game_id="p", result="0-1", reason="game over on board", pgn="1... d4"))
    assert ended == [], "no review of a position"
    assert f.store.game("p") is None and f.store.moves("p") == [] and f.store.evidence("p") == []
    assert f.store.games_count() == 0
    assert f.profiler.end_session(f.ts + 1) is None, "nothing to sum up: no game was played"


def _feed_position(f: Feed, board: chess.Board, game_id: str, plies: int) -> None:
    """`plies` legal moves played on a joined position, each with its verdict."""
    for i in range(plies):
        move = next(iter(board.legal_moves))
        fen_before = board.fen()
        by_user = (board.turn == chess.WHITE) == (f.user_side is Side.WHITE)
        board.push(move)
        f.send(
            MovePlayed(
                game_id=game_id,
                ply=i + 1,
                san=move.uci(),
                uci=move.uci(),
                fen_before=fen_before,
                fen_after=board.fen(),
                by_user=by_user,
                think_ms=1500,
            )
        )
        f.send(
            MoveAssessment(
                game_id=game_id,
                ply=i + 1,
                uci=move.uci(),
                expected_loss=0.0,
                severity=Severity.NONE,
                best=[move.uci()],
                label=Label.GOOD,
                accuracy=100.0,
            )
        )


def test_a_game_joined_from_the_middle_is_still_a_game() -> None:
    """A position that runs long enough to be no puzzle is a game joined late: recorded
    from the join, reviewed at the end (Tai, 2026-09-22: a 133-ply bot game joined at
    move 3 ended in checkmate with no review, no summary, not a word -- 'a position')."""
    from sidecoach.profile.profiler import PROVISIONAL_PLIES

    f = Feed(user_side=Side.BLACK)
    joined = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3")
    f.send(GameStarted(game_id="j", user_side=Side.BLACK, start_fen=joined.fen(), kind="position"))
    assert f.store.game("j") is not None, "followed from the join, provisionally"
    _feed_position(f, joined, "j", PROVISIONAL_PLIES)
    ended = f.send(
        GameEnded(
            game_id="j", result="0-1", reason="game over on board", pgn="", final_fen=joined.fen()
        )
    )
    assert any(isinstance(e, GameReview) for e in ended), "a game joined late is reviewed"
    assert f.store.games_count() == 1
    assert len(f.store.moves("j")) == PROVISIONAL_PLIES


def test_a_position_that_ends_early_leaves_nothing_behind() -> None:
    """A puzzle solved in a few plies, or a position left for another, was never a
    game: its provisional rows go."""
    from sidecoach.profile.profiler import PROVISIONAL_PLIES

    f = Feed(user_side=Side.WHITE)
    puzzle = chess.Board("r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R w KQkq - 0 1")
    f.send(GameStarted(game_id="p", user_side=Side.WHITE, start_fen=puzzle.fen(), kind="position"))
    _feed_position(f, puzzle, "p", 3)
    assert f.store.game("p") is not None
    ended = f.send(GameEnded(game_id="p", result="1-0", reason="game over on board", pgn=""))
    assert ended == [] and f.store.game("p") is None and f.store.moves("p") == []
    # Left for the next puzzle before it ended: the same.
    other = chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1")
    f.send(GameStarted(game_id="q", user_side=Side.WHITE, start_fen=other.fen(), kind="position"))
    _feed_position(f, other, "q", PROVISIONAL_PLIES - 1)
    f.send(GameStarted(game_id="r", user_side=Side.WHITE, start_fen=other.fen(), kind="position"))
    assert f.store.game("q") is None and f.store.games_count() == 0
    # Closed on a position: the next session's start clears it.
    assert f.store.game("r") is not None
    Profiler(f.store, session_id="s2", ts=100.0)
    assert f.store.game("r") is None


def test_puzzles_one_after_another_never_add_up_to_a_game() -> None:
    """A run of puzzles on the site: the first read cold, each next one adopted by a resync
    over the last, with no start the reconciler could see. Each is a position of its own,
    and their plies never add up to a game (Tai's Pixel, 2026-09-24: after a few puzzles a
    mate in three was reviewed as "won by checkmate in 3 moves, accuracy 100 ... over 5
    games")."""
    from sidecoach.profile.profiler import PROVISIONAL_PLIES

    f = Feed(user_side=Side.WHITE)
    run = [
        chess.Board("r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R w KQkq - 0 1"),
        chess.Board("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1"),
        chess.Board("r1bq1k1r/1p2bp2/p2p3p/2p1n2Q/P3P3/1BN5/1PP2PPP/R1B2RK1 w - - 1 15"),
        chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3"),
        chess.Board("8/5pk1/6p1/8/8/6P1/5PK1/3R4 w - - 0 1"),
    ]
    each = 4
    assert len(run) * each >= PROVISIONAL_PLIES
    f.send(GameStarted(game_id="p", user_side=Side.WHITE, start_fen=run[0].fen(), kind="position"))
    for i, board in enumerate(run):
        if i:
            f.send(Resync(game_id="p", fen=board.fen(), reason="position adopted after 20 cells"))
        _feed_position(f, board, "p", each)
    assert len(f.store.moves("p")) == each, "the last puzzle's moves only"
    ended = f.send(
        GameEnded(
            game_id="p", result="1-0", reason="game over on board", pgn="", final_fen=run[-1].fen()
        )
    )
    assert not any(isinstance(e, GameReview) for e in ended), "no review of a puzzle"
    assert f.store.game("p") is None and f.store.games_count() == 0
    assert f.profiler.end_session(f.ts + 1) is None


def test_a_joined_game_whose_turn_the_clock_settles_is_still_a_game() -> None:
    """The resync that only settles the side to move, before any move, keeps the position:
    the game joined late still qualifies and is reviewed."""
    from sidecoach.profile.profiler import PROVISIONAL_PLIES

    f = Feed(user_side=Side.BLACK)
    joined = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR b KQkq - 2 3")
    f.send(GameStarted(game_id="j", user_side=Side.BLACK, start_fen=joined.fen(), kind="position"))
    joined.turn = chess.WHITE
    f.send(Resync(game_id="j", fen=joined.fen(), reason="side to move settled by the clock: white"))
    assert f.store.game("j") is not None
    _feed_position(f, joined, "j", PROVISIONAL_PLIES)
    ended = f.send(
        GameEnded(
            game_id="j", result="0-1", reason="game over on board", pgn="", final_fen=joined.fen()
        )
    )
    assert any(isinstance(e, GameReview) for e in ended)
    assert f.store.games_count() == 1


def test_a_game_whose_end_was_not_seen_is_still_reviewed_at_the_next_start() -> None:
    """The site's dialog hid the result and nobody said how it went: at the next game's
    start (or the session's end) the game is ended with the result unknown and its
    moves reviewed and kept, rather than forgotten. One stopped after a few moves is
    not a game of the record."""
    from sidecoach.profile.profiler import PROVISIONAL_PLIES

    f = Feed()
    for san in (
        "e4",
        "e5",
        "Nf3",
        "Nc6",
        "Bb5",
        "a6",
        "Ba4",
        "Nf6",
        "O-O",
        "Be7",
        "Re1",
        "b5",
        "Bb3",
        "d6",
        "c3",
        "O-O",
        "h3",
        "Nb8",
    ):
        f.play(san)
    assert f.ply >= PROVISIONAL_PLIES
    got = f.send(GameStarted(game_id="h", user_side=Side.WHITE, start_fen=chess.STARTING_FEN))
    review = next(e for e in got if isinstance(e, GameReview))
    assert review.result == "*"
    assert review.lines[0] == "You played white for 10 moves; I did not see how it ended."
    kept = f.store.game("g")
    assert kept is not None and kept.result == "*"
    assert "1 unfinished" in (f.profiler.end_session(f.ts + 1) or "")
    # The short one: gone.
    short = Feed()
    short.play("e4")
    short.play("e5")
    assert short.profiler.end_session(short.ts + 1) is None
    assert short.store.game("g") is None


def test_the_review_says_how_the_game_ended_in_plain_words() -> None:
    from sidecoach.events import GameReview
    from sidecoach.profile.review import lines_for

    def first(result: str, reason: str | None, side: Side = Side.WHITE) -> str:
        review = GameReview(game_id="g", user_side=side, result=result, reason=reason)
        return lines_for(review, 17)[0]

    # The End game button's reasons, and the board's own, read as English after the outcome.
    assert first("0-1", "user resigned or lost on time") == (
        "You played white and lost by resignation or on time in 17 moves."
    )
    assert first("1-0", "opponent resigned or lost on time") == (
        "You played white and won by resignation or on time in 17 moves."
    )
    assert first("0-1", "lost on time") == "You played white and lost on time in 17 moves."
    assert first("1-0", "game over on board") == "You played white and won in 17 moves."
    assert first("1/2-1/2", "fifty-move rule") == (
        "You played white and drew by the fifty-move rule in 17 moves."
    )
    assert first("1-0", "checkmate") == "You played white and won by checkmate in 17 moves."
    assert first("1/2-1/2", "draw agreed") == "You played white and drew by agreement in 17 moves."


def test_a_query_answering_infinity_is_shown_not_a_crash() -> None:
    from sidecoach.profile.ask import _cell

    assert _cell(float("inf")) == "inf"
    assert _cell(float("nan")) == "nan"
    assert _cell(2.5) == "2.5"
