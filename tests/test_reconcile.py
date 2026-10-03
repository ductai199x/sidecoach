import chess
import pytest

from sidecoach.events import (
    GameEnded,
    GameStarted,
    MovePlayed,
    PositionChanged,
    Resync,
    Side,
    Takeback,
    Unexplained,
)
from sidecoach.state.reconcile import Reconciler, cell_agrees
from sidecoach.vision.pieces import CellReading, ThemeModel
from sidecoach.vision.squares import row_col
from tests.games import recogniser, sans_of
from tests.synth import GEOM, synthetic_board

pytestmark = pytest.mark.vision  # the synthetic vision pipeline: run with -m vision


def _setup(bottom: Side = Side.WHITE) -> tuple[ThemeModel, Reconciler]:
    """The recogniser learned once for `bottom` (a plain helper, not a fixture: the phone's
    fixture maker calls these tests directly), and a fresh reconciler."""
    return recogniser(bottom), Reconciler(bottom=bottom, user_side=bottom, settle_frames=2)


def _feed(model, rec, board, bottom, ts, highlight=None, dragging=None):
    frame = synthetic_board(board=board, bottom=bottom, highlight=highlight, dragging=dragging)
    return rec.update(model.classify_frame(frame, GEOM), ts)


def _play(model, rec, board, sans, bottom=Side.WHITE, ts=0.0):
    """Push moves on `board` and feed each twice with its highlight so it commits."""
    for san in sans:
        move = board.push_san(san)
        for _ in range(2):
            ts += 0.1
            _feed(model, rec, board, bottom, ts, highlight=move)
    return ts


def test_moves_are_committed_after_settling() -> None:
    """Each move commits on its second settled frame, told as the move of the side whose
    turn it was, with its think time the time since the last commit on the frames' own
    timestamps (test_think_time_comes_from_the_frame_timestamps)."""
    model, rec = _setup()
    board = chess.Board()
    events = _feed(model, rec, board, Side.WHITE, 0.0)
    assert any(isinstance(e, GameStarted) for e in events)

    by_user = []
    think_ms = []
    ts = 0.0
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "Nxe4", "d4", "exd4"):
        move = board.parse_san(san)
        board.push(move)
        first = _feed(model, rec, board, Side.WHITE, ts + 0.1, highlight=move)
        assert not sans_of(first), "committed on one frame"
        ts += 0.2
        second = _feed(model, rec, board, Side.WHITE, ts, highlight=move)
        assert sans_of(second) == [san]
        by_user += [e.by_user for e in second if isinstance(e, MovePlayed)]
        think_ms += [e.think_ms for e in second if isinstance(e, MovePlayed)]
    assert rec.board.fen() == board.fen()
    assert rec.pgn() == "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. O-O Nxe4 5. d4 exd4"
    assert by_user == [True, False] * 5
    assert think_ms == [200] * 10


def test_first_move_is_not_committed_mid_drag() -> None:
    """The pawn is drawn over e4 under the cursor, off centre: still in the air. Then held
    exactly centred with the hovered square still outlined: not dropped yet either
    (test_hover_outline_alone_blocks_a_commit: the outline alone blocks the commit). Eight
    frames over its square, long enough for its cells to go stale, and still no move
    (test_long_hover_over_destination_is_not_a_move)."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    off_centre = ((0.3, 0.1), (0.2, -0.2), (-0.25, 0.15), (0.1, 0.3))
    centred = ((0.0, 0.0),) * 4
    for i, (dx, dy) in enumerate(off_centre + centred):
        drag = (chess.E2, chess.E4, dx, dy)
        assert not sans_of(_feed(model, rec, board, Side.WHITE, 0.1 * i, dragging=drag))
    move = board.push_san("e4")
    _feed(model, rec, board, Side.WHITE, 1.0, highlight=move)
    assert sans_of(_feed(model, rec, board, Side.WHITE, 1.1, highlight=move)) == ["e4"]


def test_browsing_history_is_not_a_takeback() -> None:
    """Stepping through the move list is browsing, never a takeback; and the opponent's move
    made while the user browses snaps the site back to the latest position plus that move,
    which commits as the game's next move (test_opponent_move_while_browsing_snaps_to_latest)."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5", "Nf3", "Nc6", "Bb5"])
    latest = board.fen()

    def position(ply: int) -> chess.Board:
        earlier = board.copy()
        while len(earlier.move_stack) > ply:
            earlier.pop()
        return earlier

    # Step back to ply 2, then ply 1, then forward to ply 3, then back to the latest, then
    # back to ply 1 again.
    ts = 1.0
    for ply, expect_view in ((2, 2), (1, 1), (3, 3), (5, None), (1, 1)):
        for _ in range(6):  # long enough for cells to go stale if they were allowed to
            ts += 0.1
            events = _feed(model, rec, position(ply), Side.WHITE, ts)
            assert not any(isinstance(e, Takeback | MovePlayed) for e in events)
        assert rec.viewing == expect_view, (ply, rec.viewing)
    assert rec.board.fen() == latest
    # The bot replies while the user looks at ply 1: the site jumps to the latest position
    # plus the new move.
    reply = board.push_san("a6")
    events = _feed(model, rec, board, Side.WHITE, ts + 0.1, highlight=reply)
    events += _feed(model, rec, board, Side.WHITE, ts + 0.2, highlight=reply)
    assert sans_of(events) == ["a6"]
    assert rec.viewing is None
    assert rec.board.fen() == board.fen()


# (the game, the plies taken back, the moves played instead)
TAKEBACKS = [
    # A bot-game takeback: the user's Nf3 and the reply vanish, then a different move.
    (("e4", "e5", "Nf3", "Nc6"), 2, ("d4",)),
    # A move legal only in the earlier position, the bishop back on f1: it was once checked
    # against the latest board and crashed (test_takeback_move_legal_only_in_the_earlier_position).
    (("e4", "e5", "Nf3", "Nc6", "Bb5"), 1, ("Bc4",)),
    # The bot answers at once: a takeback, not Ng1 Nb8 told as retreats
    # (test_takeback_with_a_quick_reply_is_not_told_as_retreats).
    (("e4", "e5", "Nf3", "Nc6"), 2, ("d4", "exd4")),
]


@pytest.mark.parametrize(
    ("line", "back", "instead"), TAKEBACKS, ids=["bot-game", "legal-only-before", "quick-reply"]
)
def test_a_different_move_from_an_earlier_position_is_a_takeback(
    line: tuple[str, ...], back: int, instead: tuple[str, ...]
) -> None:
    """The board steps back through the move list, which is browsing; a different move from
    there takes the plies after it back, once the move has settled, and is told as the
    game's next move (test_takeback_is_a_new_move_from_an_earlier_position)."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    ts = _play(model, rec, board, list(line))
    for _ in range(back):
        board.pop()
    for _ in range(2):
        ts += 0.1
        _feed(model, rec, board, Side.WHITE, ts)
    assert rec.viewing == len(line) - back
    for san in instead:
        board.push_san(san)
    events = []
    for i in range(4):
        ts += 0.1
        new = _feed(model, rec, board, Side.WHITE, ts, highlight=board.peek())
        assert i > 0 or not any(isinstance(e, Takeback) for e in new), "taken back on one frame"
        events += new
    assert [e.plies for e in events if isinstance(e, Takeback)] == [back]
    assert sans_of(events) == list(instead)
    assert rec.board.fen() == board.fen() and rec.board.move_stack == board.move_stack


def test_animation_across_a_still_highlighted_square_is_not_a_move() -> None:
    """The knight animates b5 to d6 across c7, which still carries the last move's highlight."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "c6", "Nc3", "d5", "Nb5"])
    reply = board.push_san("c5")  # c6-c5 highlights c6 and c5
    for ts in (1.1, 1.2):
        _feed(model, rec, board, Side.WHITE, ts, highlight=reply)
    assert rec.board.fen() == board.fen()

    # Mid-animation frames: the knight glides from b5 towards d6, passing over c6 and c7
    # off centre, with the previous move's squares still tinted.
    for ts, (over, dx, dy) in zip(
        (1.3, 1.4, 1.5),
        ((chess.C6, -0.3, 0.3), (chess.C6, 0.1, -0.1), (chess.C7, 0.35, 0.2)),
        strict=True,
    ):
        drag = (chess.B5, over, dx, dy)
        events = _feed(model, rec, board, Side.WHITE, ts, highlight=reply, dragging=drag)
        assert not sans_of(events), "animation frame taken as a move"
    # Landed: b5 and d6 highlighted, check on the black king.
    move = board.push_san("Nd6+")
    _feed(model, rec, board, Side.WHITE, 2.0, highlight=move)
    assert sans_of(_feed(model, rec, board, Side.WHITE, 2.1, highlight=move)) == ["Nd6+"]


def test_red_marks_do_not_block_moves() -> None:
    """Right-click marks tint squares red; they are annotations, not highlights."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4"])
    marks = (chess.D5, chess.F5, chess.H3)
    reply = board.push_san("e5")
    frame = synthetic_board(board=board, highlight=reply, red_marks=marks)
    readings = model.classify_frame(frame, GEOM)
    r, c = row_col(chess.D5, Side.WHITE)
    assert readings[r][c].highlighted is False
    r, c = row_col(chess.E5, Side.WHITE)
    assert readings[r][c].highlighted is True
    events = rec.update(readings, 1.0) + rec.update(readings, 1.1)
    assert sans_of(events) == ["e5"]


@pytest.mark.parametrize(
    ("before", "missed"),
    [((), ("e4", "c5")), (("e4", "e5"), ("Nf3", "Nc6", "Bb5", "a6"))],
    ids=["the-first-move-missed", "four-while-the-window-was-hidden"],
)
def test_plies_missed_between_frames_are_caught_up(
    before: tuple[str, ...], missed: tuple[str, ...]
) -> None:
    """Moves no frame showed, one or several, are told from the position after them
    (test_two_move_catch_up_when_one_move_was_missed,
    test_four_ply_catch_up_after_window_was_hidden)."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    ts = _play(model, rec, board, list(before))
    for san in missed:
        board.push_san(san)
    events = []
    for _ in range(3):
        ts += 0.1
        events += _feed(model, rec, board, Side.WHITE, ts, highlight=board.peek())
    assert sorted(sans_of(events)) == sorted(missed)
    assert rec.board.fen() == board.fen()


def test_ambiguous_paths_are_not_committed() -> None:
    model, rec = _setup()
    rec.max_plies = 2
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5"])
    board.push_san("Nf3")
    board.push_san("Nc6")
    readings = model.classify_frame(synthetic_board(board=board), GEOM)
    for sq in (chess.G1, chess.F3, chess.B8, chess.C6):
        r, c = row_col(sq, Side.WHITE)
        readings[r][c] = CellReading("unknown", None, None, 0.0, 0.05, 0.0, False)
    for ts in (1.0, 1.1, 1.2):
        assert not sans_of(rec.update(readings, ts))


def test_stale_cell_does_not_stall_the_game() -> None:
    model, rec = _setup()
    rec.stale_after = 2
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)

    def poison(readings):
        r, c = row_col(chess.A1, Side.WHITE)
        readings[r][c] = CellReading("piece", "Q", Side.WHITE, 0.9, 0.4, 240.0, False)
        return readings

    frame = synthetic_board(board=board)
    for i in range(3):
        rec.update(poison(model.classify_frame(frame, GEOM)), 0.1 * i)
    move = board.push_san("e4")
    frame = synthetic_board(board=board, highlight=move)
    events = rec.update(poison(model.classify_frame(frame, GEOM)), 1.0)
    events += rec.update(poison(model.classify_frame(frame, GEOM)), 1.1)
    assert sans_of(events) == ["e4"]


def test_promotion_and_en_passant() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "d5", "e5", "f5", "exf6"])
    assert rec.pgn() == "1. e4 d5 2. e5 f5 3. exf6"

    promo = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    rec2 = Reconciler(bottom=Side.WHITE, settle_frames=1)
    rec2.board = promo.copy()
    rec2._started = True
    promo.push_san("a8=N")
    assert sans_of(_feed(model, rec2, promo, Side.WHITE, 0.0)) == ["a8=N"]


def test_move_from_start_position_mid_game_is_a_new_game() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5", "Nf3"])
    first_id = rec.game_id
    fresh = chess.Board()
    for ts in (1.0, 1.1):
        _feed(model, rec, fresh, Side.WHITE, ts)
    assert rec.viewing == 0 and len(rec.board.move_stack) == 3
    move = fresh.push_san("d4")
    events = []
    for ts in (1.2, 1.3):
        events += _feed(model, rec, fresh, Side.WHITE, ts, highlight=move)
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert len(ended) == 1 and ended[0].pgn == "1. e4 e5 2. Nf3"
    assert any(isinstance(e, GameStarted) for e in events)
    assert not any(isinstance(e, Takeback) for e in events)
    assert sans_of(events) == ["d4"]
    assert rec.game_id != first_id and rec.pgn() == "1. d4"


def test_a_game_replaced_by_the_next_has_the_result_its_final_position_proves() -> None:
    """A game the next one replaced before its result was seen: a draw by repetition played
    on the board (the sites draw it when the position comes the third time) is a draw, as the
    library reads an import's ending, not a game left unfinished (Tai, 2026-10-02). A game whose
    position proves nothing stays "*" (test_move_from_start_position_mid_game_is_a_new_game,
    test_panel_engine's test_a_game_ended_by_the_next_one_is_not_offered_to_end_game)."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    shuffle = ["Nf3", "Nc6", "Ng1", "Nb8"]
    ts = _play(model, rec, board, ["e4", "e5", *shuffle, *shuffle])
    assert board.is_repetition(3) and not rec.finished
    first_id = rec.game_id
    fresh = chess.Board()
    for _ in range(2):
        ts += 0.1
        _feed(model, rec, fresh, Side.WHITE, ts)
    move = fresh.push_san("d4")
    events = []
    for _ in range(2):
        ts += 0.1
        events += _feed(model, rec, fresh, Side.WHITE, ts, highlight=move)
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert [(e.game_id, e.result, e.reason) for e in ended] == [(first_id, "1/2-1/2", "repetition")]
    assert rec.game_id != first_id and rec.pgn() == "1. d4"


def test_a_game_whose_end_was_never_seen_ends_at_the_session_by_its_final_position() -> None:
    """The session's end with a game under way (Tai, 2026-10-02): a position that proves
    nothing ends nothing (the record keeps the game unfinished, "*"); the third repetition
    played is a draw, by the rule a replaced game's result is set with, and once only."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    shuffle = ["Nf3", "Nc6", "Ng1", "Nb8"]
    ts = _play(model, rec, board, ["e4", "e5", *shuffle])
    assert rec.end_unseen() == [] and not rec.finished, "twice is no draw: nothing is set"
    _play(model, rec, board, shuffle, ts=ts)
    ended = rec.end_unseen()
    assert [(e.result, e.reason) for e in ended if isinstance(e, GameEnded)] == [
        ("1/2-1/2", "repetition")
    ]
    assert len(ended) == 1
    assert rec.finished and rec.result == "1/2-1/2"
    assert rec.end_unseen() == []


def test_the_final_position_proves_a_result_only_when_the_game_is_over_on_it() -> None:
    """`result_on_board`, the rule for a replaced game's result: a mate, a stalemate, dead
    material, a repetition or fifty moves played; nothing for a position a resignation, a
    flag or an agreement ended, nor for a repetition or fifty moves the next move could only
    reach."""
    from sidecoach.state.reconcile import result_on_board

    mated = chess.Board()
    for san in ("f3", "e5", "g4", "Qh4#"):
        mated.push_san(san)
    assert result_on_board(mated) == ("0-1", "checkmate")
    assert result_on_board(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == (
        "1/2-1/2",
        "stalemate",
    )
    assert result_on_board(chess.Board("8/8/4k3/8/8/2K5/8/8 w - - 0 60")) == (
        "1/2-1/2",
        "insufficient material",
    )
    rooks = "8/8/4k3/8/r7/2K5/8/R7 w - - {} 80"
    assert result_on_board(chess.Board(rooks.format(100))) == ("1/2-1/2", "fifty-move rule")
    assert result_on_board(chess.Board(rooks.format(99))) is None, "one half-move short"
    twice = chess.Board()
    for san in ("Nf3", "Nf6", "Ng1", "Ng8"):
        twice.push_san(san)
    assert result_on_board(twice) is None, "the start position twice: a third is only possible"
    assert result_on_board(chess.Board()) is None


def test_position_resync_after_a_long_gap() -> None:
    """Many plies happened while the board was hidden: adopt the position, then continue."""
    model, rec = _setup()
    rec.resync_hold_frames = 3
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5"])
    for san in ("Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3"):
        board.push_san(san)  # eleven plies, far beyond the move search
    events = []
    frame = synthetic_board(board=board)
    for ts in (1.0, 1.1, 1.2, 1.3):
        readings = model.classify_frame(frame, GEOM)
        # The last-moved bishop carries a badge: type hidden, best guess kept.
        r, c = row_col(chess.B3, Side.WHITE)
        old = readings[r][c]
        readings[r][c] = CellReading(
            "piece", None, old.colour, 0.5, old.fg_fraction, old.luminance, guess="B"
        )
        events += rec.update(readings, ts)
    resyncs = [e for e in events if isinstance(e, Resync)]
    assert len(resyncs) == 1
    assert resyncs[0].fen.split()[0] == board.board_fen()
    assert rec.board.board_fen() == board.board_fen()
    assert rec.board.has_kingside_castling_rights(chess.BLACK)  # king and rook at home
    assert not rec.board.has_castling_rights(chess.WHITE)  # white castled
    assert not sans_of(events)

    # The next move settles whose turn it was and is committed normally.
    move = board.push_san("d6")
    events = []
    for ts in (2.0, 2.1):
        events += _feed(model, rec, board, Side.WHITE, ts, highlight=move)
    assert sans_of(events) == ["d6"]
    assert not rec.turn_unknown
    assert rec.board.fen().split()[0] == board.board_fen()
    assert rec.pgn() == "1... d6"


def test_move_from_adopted_position_is_a_takeback_not_a_new_game() -> None:
    """After a resync, browsing back to the adopted position and playing on is a takeback."""
    model, rec = _setup()
    rec.resync_hold_frames = 2
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    for san in ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7"):
        board.push_san(san)
    events = []
    for ts in (1.0, 1.1, 1.2):
        events += _feed(model, rec, board, Side.WHITE, ts)
    assert any(isinstance(e, Resync) for e in events)
    adopted = board.copy()

    _play(model, rec, board, ["Re1", "b5"], ts=2.0)  # settles the turn and continues
    assert rec.pgn() == "1. Re1 b5"

    # Browse back to the adopted position, then play a different move from it.
    for ts in (3.0, 3.1):
        _feed(model, rec, adopted, Side.WHITE, ts)
    assert rec.viewing == 0
    move = adopted.push_san("d3")
    events = []
    for ts in (3.2, 3.3):
        events += _feed(model, rec, adopted, Side.WHITE, ts, highlight=move)
    assert any(isinstance(e, Takeback) and e.plies == 2 for e in events)
    assert not any(isinstance(e, GameEnded) for e in events)
    assert sans_of(events) == ["d3"]
    assert rec.board.fen().split()[0] == adopted.board_fen()


def test_illegal_commit_is_refused_not_fatal() -> None:
    rec = Reconciler(bottom=Side.WHITE)
    rec.start()
    events = rec._commit(chess.Move.from_uci("d4g4"))
    assert len(events) == 1 and isinstance(events[0], Unexplained), "a note, not a resync"
    assert rec.board.fen() == chess.Board().fen()


def test_colour_only_reading_means_occupied_not_a_colour() -> None:
    reading = CellReading("piece", None, Side.WHITE, 0.5, 0.5, 240.0, False)
    assert cell_agrees(reading, chess.Piece.from_symbol("q"))
    assert cell_agrees(reading, chess.Piece.from_symbol("Q"))
    assert not cell_agrees(reading, None)


def test_cells_unexplained_for_long_are_noted_and_the_game_is_kept() -> None:
    """Several cells the game cannot explain for a long while: a note (`Unexplained`), and
    the game goes on as it stands. It was a Resync, which the coach and the engine read as
    the numbering starting again while the reconciler kept its own (Tai's game, 2026-09-23:
    the takeback that came after was "10 plies" to one and "the last move" to the other).
    A piece lifted and held as long is a drag in progress: no move, no note
    (test_drag_in_progress_is_held)."""

    model, rec = _setup()
    rec.stale_after = 2
    rec.resync_after = 4
    rec.resync_hold_s = 60.0  # the note is what is tested, not when a resync would adopt
    rec.unexplained_note_frames = 2
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5"], ts=0.0)
    weird = chess.Board("8/8/8/3k4/8/8/8/3K4 w - - 0 1")
    events = []
    for i in range(6):
        events += _feed(model, rec, weird, Side.WHITE, 1.0 + 0.1 * i)
    notes = [e for e in events if isinstance(e, Unexplained)]
    # Early, for the log, what each square reads as against the game; later, the person told.
    assert [e.told for e in notes] == [False, True], notes
    assert notes[0].reason.startswith("unexplained for now: ")
    assert "a1 empty, game R" in notes[0].reason and "d5 k, game -" in notes[0].reason
    assert "a1 empty, game R" in notes[1].reason
    assert not any(isinstance(e, Resync) for e in events), "the position did not jump"
    assert rec.pgn() == "1. e4 e5", "the game is kept, its numbering with it"

    model2, rec2 = _setup()
    rec2.stale_after = 2
    rec2.resync_after = 4
    _feed(model2, rec2, chess.Board(), Side.WHITE, 0.0)
    lifted = chess.Board()
    lifted.remove_piece_at(chess.D2)
    events = []
    for i in range(12):
        events += _feed(model2, rec2, lifted, Side.WHITE, 0.1 * i)
    assert not any(isinstance(e, Resync | Unexplained | MovePlayed) for e in events)
    assert rec2.board.fen() == chess.STARTING_FEN


def test_badge_arriving_with_the_move_does_not_block_it() -> None:
    """The checkmate badge lands on the king the moment the mating move lands.

    Its cell then disagrees for exactly as long as the move's own cells do, so all of them
    turn stale together; ignoring the badge alone must still let the move through.
    """
    model, rec = _setup()
    rec.stale_after = 2
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6"])

    def badge(readings):
        r, c = row_col(chess.E8, Side.WHITE)  # the badge spills over the king's square
        readings[r][c] = CellReading("piece", None, Side.WHITE, 0.5, 0.4, 240.0, settled=False)
        return readings

    mate = board.push_san("Qxf7#")
    frame = synthetic_board(board=board, highlight=mate)
    events = []
    for i in range(8):
        events += rec.update(badge(model.classify_frame(frame, GEOM)), 2.0 + 0.1 * i)
    assert sans_of(events) == ["Qxf7#"]


def test_story_seen_before_a_drag_is_extended_after_it() -> None:
    """The king walk of the fourth game: two plies land, the user re-grabs the king at once.

    Ke7 and Kxa5 are seen settled twice, then the king is dragged around for a while, the
    bot answers Ke8 meanwhile, the king is dropped on b6 and grabbed again, and the board
    is only quiet again after Kd7 and Kb7. From the committed position that is six plies
    with the king back on b6, which no search finds. The story seen before the drag is
    kept, the bot's reply confirms it from its own cells while the drag goes on, and the
    quiet frames extend what is left.
    """
    fen = "8/3k4/1K6/pP2Bp1p/P4PpP/6P1/8/8 b - - 0 54"
    model, rec = _setup()
    _feed(model, rec, chess.Board(), Side.WHITE, 0.0)  # the game starts, then jump ahead
    rec.board = chess.Board(fen)
    rec._base_fen = fen
    board = chess.Board(fen)
    ts = 0.0
    played: list[str] = []

    def see(frames: int, dragging=None) -> None:
        nonlocal ts
        for _ in range(frames):
            ts += 0.1
            played.extend(sans_of(_feed(model, rec, board, Side.WHITE, ts, dragging=dragging)))

    see(2)
    board.push_san("Ke7")
    board.push_san("Kxa5")
    see(2)
    assert played == []
    see(3, dragging=(chess.A5, chess.B4, 0.1, -0.1))
    see(2, dragging=(chess.A5, chess.A5, 0.1, -0.1))
    assert played == [], "a drag is not a move"
    board.push_san("Ke8")
    see(6, dragging=(chess.A5, chess.A5, 0.1, -0.1))
    assert played == ["Ke7", "Kxa5", "Ke8"], "the bot's reply confirms the story mid-drag"
    board.push_san("Kb6")
    see(1)
    see(4, dragging=(chess.B6, chess.C7, 0.1, 0.1))
    board.push_san("Kd7")
    board.push_san("Kb7")
    see(3)
    assert played == ["Ke7", "Kxa5", "Ke8", "Kb6", "Kd7", "Kb7"]
    assert rec.board.fen() == board.fen()


def test_mate_settles_with_two_badge_cells_as_wildcards() -> None:
    """After 67.Qg8+ Kh6 68.Qaf8#: the checkmate badge spoils h6 and spills into g6.

    Both cells the king's move touched are junk, so the two-ply story can only be found
    by letting the ignored stale cells stand in for exact readings.
    """
    fen = "Q5Q1/8/2K3k1/4Bp1p/5PpP/6P1/8/8 b - - 0 67"
    model, rec = _setup()
    rec.stale_after = 2
    _feed(model, rec, chess.Board(), Side.WHITE, 0.0)
    rec.board = chess.Board(fen)
    rec._base_fen = fen
    board = chess.Board(fen)
    for i in range(2):
        _feed(model, rec, board, Side.WHITE, 0.1 + 0.1 * i)
    board.push_san("Kh6")
    board.push_san("Qaf8#")

    def badge(readings):
        r, c = row_col(chess.H6, Side.WHITE)
        readings[r][c] = CellReading("piece", "n", Side.BLACK, 0.4, 0.5, 60.0, settled=False)
        r, c = row_col(chess.G6, Side.WHITE)
        readings[r][c] = CellReading("piece", None, Side.BLACK, 0.5, 0.3, 90.0, settled=False)
        return readings

    frame = synthetic_board(board=board)
    events = []
    for i in range(8):
        events += rec.update(badge(model.classify_frame(frame, GEOM)), 1.0 + 0.1 * i)
    assert sans_of(events) == ["Kh6", "Qaf8#"]
    assert any(isinstance(e, GameEnded) for e in events)


def test_a_queen_nobody_can_read_is_not_moved_there_and_back() -> None:
    """Both queens read as pieces of no known kind for a whole game (a set whose queens the
    recogniser could not tell apart), so every cell a queen touches is a soft wildcard.
    After ...Qd7, h3, ...Qxg4, e4 (badged) the search once told d8-d7, h3, d7-d8, e4: a
    queen shuffle invented to fill Black's turns through the cells nobody could read; at
    another cadence a tour, d8-c7, h3, c7-h2, e4, onto a square that read empty but had
    gone stale. A piece that goes and comes straight back explains nothing, a second move
    told through wild cells alone is invented, and a stale exact reading may not be
    contradicted by a story that changes its square. What is told is what was played.
    (It ran at three, four and five frames a move. On 2026-09-29 each rule was switched off
    in turn: without the third every cadence failed alike, on the tour; without the first,
    the second or both, every cadence passed. One cadence is kept.)"""
    frames = 3
    model, rec = _setup(Side.BLACK)
    board = chess.Board()
    _feed(model, rec, board, Side.BLACK, 0.0)
    _play(model, rec, board, ["g4", "d5", "c4", "c5", "b4"], bottom=Side.BLACK)

    def blind_queens(readings):
        for r in range(8):
            for c in range(8):
                reading = readings[r][c]
                if reading.symbol in ("Q", "q"):
                    readings[r][c] = CellReading(
                        "piece", None, reading.colour, 0.5, 0.4, 100.0, settled=False
                    )
        return readings

    played = ("Qd7", "h3", "Qxg4", "e4", "Kd8", "e5")
    events = []
    ts = 1.0
    for san in played:
        move = board.push_san(san)
        badge = (move.to_square, "best") if san == "e4" else None
        for _ in range(frames):
            ts += 0.1
            frame = synthetic_board(board=board, bottom=Side.BLACK, highlight=move, badge=badge)
            events += rec.update(blind_queens(model.classify_frame(frame, GEOM)), ts)
    told = sans_of(events)
    assert all(san in played for san in told), told


# A Ruy Lopez of 60 plies, longer than the forty the history was once looked through.
LONG_GAME = [
    "e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5", "Bb3",
    "d6", "c3", "O-O", "h3", "Nb8", "d4", "Nbd7", "c4", "c6", "cxb5", "axb5", "Nc3", "Bb7",
    "Bg5", "b4", "Nb1", "h6", "Bh4", "c5", "dxe5", "Nxe4", "Bxe7", "Qxe7", "exd6", "Qf6",
    "Nbd2", "Nxd6", "Nc4", "Nxc4", "Bxc4", "Nb6", "Ne5", "Rae8", "Bxf7+", "Rxf7", "Nxf7",
    "Rxe1+", "Qxe1", "Kxf7", "Qe3", "Qg5", "Qxg5", "hxg5", "b3", "Ke6", "a3", "Kd6",
]  # fmt: skip


def test_browsing_anywhere_in_a_long_game_is_browsing() -> None:
    """A click anywhere in the move list shows that position, however far back: browsing,
    the game kept, and stepping forward through the list tells nothing. Only the last forty
    plies were looked through, and the start: the position after ten plies of sixty was one
    no move explains, adopted by a resync -- the game numbered again from it, or after the
    end a new game begun -- and stepping forward told the game's own moves a second time
    (the state review, 2026-10-01)."""
    for finished in (False, True):
        model, rec = _setup()
        _feed(model, rec, chess.Board(), Side.WHITE, 0.0)
        played = chess.Board()
        for san in LONG_GAME:
            played.push_san(san)
        assert len(played.move_stack) == 60
        rec.board = played.copy()
        first = rec.game_id
        if finished:
            rec.end("1/2-1/2", "draw agreed")
        shown = chess.Board()
        for move in played.move_stack[:10]:
            shown.push(move)
        events = []
        ts = 1.0
        for _ in range(12):
            ts += 0.1
            events += _feed(model, rec, shown, Side.WHITE, ts)
        for move in played.move_stack[10:16]:
            shown.push(move)
            for _ in range(3):
                ts += 0.1
                events += _feed(model, rec, shown, Side.WHITE, ts, highlight=move)
        assert not any(
            isinstance(e, GameStarted | GameEnded | Resync | Takeback | MovePlayed) for e in events
        ), [type(e).__name__ for e in events]
        assert rec.viewing == 16 and rec.game_id == first
        assert rec.board.move_stack == played.move_stack and rec.finished == finished
        for _ in range(2):
            ts += 0.1
            _feed(model, rec, played, Side.WHITE, ts, highlight=played.peek())
        assert rec.viewing is None


QUEUE_BOARD = "4k3/P7/8/3p4/4P3/8/P7/4K3 w - - 0 1"
# (position, origin, target, the piece drawn at the target, the queue's moves or None)
QUEUE_PATHS = [
    # A pawn takes the double step when that is shorter.
    (QUEUE_BOARD, chess.A2, chess.A5, "P", ["a2a4", "a4a5"]),
    # A pawn only captures on the way to its target, never through an empty square.
    (QUEUE_BOARD, chess.E4, chess.D5, "P", ["e4d5"]),
    (QUEUE_BOARD, chess.E4, chess.F5, "P", None),
    # Promotion to the piece drawn on the last rank; a pawn drawn there is not decided.
    (QUEUE_BOARD, chess.A7, chess.A8, "N", ["a7a8n"]),
    (QUEUE_BOARD, chess.A7, chess.A8, "P", None),
    (QUEUE_BOARD, chess.A7, chess.A8, "q", None),  # the wrong colour
    # A king two squares away has several shortest ways: nothing is assumed.
    (QUEUE_BOARD, chess.E1, chess.E3, "K", None),
    (QUEUE_BOARD, chess.E1, chess.E2, "K", ["e1e2"]),
    # A blocked file has no way at all.
    ("4k3/8/8/8/8/p7/P7/4K3 w - - 0 1", chess.A2, chess.A4, "P", None),
    # A queue never passes through the en-passant square: the pawn on e5 may go on up its
    # file, but not by way of d6 (test_queue_path_never_passes_through_the_en_passant_square).
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", chess.E5, chess.D7, "P", None),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2", chess.E5, chess.E7, "P", ["e5e6", "e6e7"]),
]


def test_queue_path_shortest_unique_ways() -> None:
    rec = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE)
    for fen, origin, target, symbol, expected in QUEUE_PATHS:
        if rec.board.fen() != fen:
            rec.board = chess.Board(fen)
        path = rec._queue_path(origin, target, symbol)
        got = None if path is None else [m.uci() for m in path]
        assert got == expected, (fen, chess.square_name(origin), chess.square_name(target))


def test_opponent_move_waits_while_its_squares_are_under_the_drag() -> None:
    """The bot answers e5 while the user drags a pawn across e5: nothing commits until the drop."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4"])
    board.push_san("e5")
    events = []
    for i in range(4):
        drag = (chess.D2, chess.E5, 0.05, 0.05)
        events += _feed(model, rec, board, Side.WHITE, 1.0 + 0.1 * i, dragging=drag)
    assert not sans_of(events)
    for i in range(2):
        events += _feed(model, rec, board, Side.WHITE, 2.0 + 0.1 * i)
    assert sans_of(events) == ["e5"]


MIDDLEGAME = "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"


# A puzzle for White, far from the game before it: Black, to move, plays the opening move
# (h6), then White checks on the back rank.
PUZZLE = "6k1/5ppp/8/8/8/8/5PPP/3R2K1 b - - 0 1"


def test_a_new_puzzles_opening_move_is_played_from_its_brief_first_board() -> None:
    """A new puzzle's first board stands for well under a second before the site plays the
    opening move: too briefly to be adopted, and the board after it was, so the move was lost
    (Tai's puzzles, 2026-10-03: missed "75-80% of the time"). The brief board is the start and
    the move is played; the solver's reply follows from there."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    ts = _play(model, rec, board, ["e4", "e5", "Nf3"])
    puzzle = chess.Board(PUZZLE)
    events = []
    for k in range(2):  # the new puzzle's first board, briefly
        events += _feed(model, rec, puzzle, Side.WHITE, ts + 1.0 + 0.1 * k)
    opening = puzzle.push_san("h6")
    for k in range(12):  # the site plays the opening move; the board after it holds
        events += _feed(model, rec, puzzle, Side.WHITE, ts + 1.3 + 0.15 * k, highlight=opening)
    resyncs = [e for e in events if isinstance(e, Resync)]
    assert len(resyncs) == 1 and "its first move seen" in resyncs[0].reason
    assert resyncs[0].fen.split()[0] == chess.Board(PUZZLE).board_fen(), (
        "the brief board is the start"
    )
    assert sans_of(events) == ["h6"], "the opening move is played"
    assert not rec.turn_unknown and rec.board.turn == chess.WHITE
    # The solver's move follows from there.
    check = puzzle.push_san("Rd8+")
    events = []
    for k in range(2):
        events += _feed(model, rec, puzzle, Side.WHITE, ts + 5.0 + 0.1 * k, highlight=check)
    assert sans_of(events) == ["Rd8+"]


def test_a_new_board_more_than_one_move_from_the_brief_one_is_adopted_alone() -> None:
    """Two boards shown in turn that no single move joins -- a page passing, two puzzles in a
    row -- are no story: the second is adopted as it is, nothing invented between them."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    ts = _play(model, rec, board, ["e4", "e5", "Nf3"])
    first = chess.Board(PUZZLE)
    events = []
    for k in range(2):
        events += _feed(model, rec, first, Side.WHITE, ts + 1.0 + 0.1 * k)
    second = chess.Board(MIDGAME_AFTER)
    for k in range(12):
        events += _feed(model, rec, second, Side.WHITE, ts + 1.3 + 0.15 * k)
    resyncs = [e for e in events if isinstance(e, Resync)]
    assert len(resyncs) == 1 and "its first move seen" not in resyncs[0].reason
    assert resyncs[0].fen.split()[0] == second.board_fen()
    assert not sans_of(events)


# Two moves away from PUZZLE's board (h6, Rd8+): no single move joins them.
MIDGAME_AFTER = "3R2k1/5pp1/7p/8/8/8/5PPP/6K1 b - - 1 2"

# Tai's puzzle on the Pixel (2026-10-03), as read: the knight on h4, on the highlighted
# square of the site's opening move, read as a bishop.
MISREAD_PUZZLE = "5rk1/ppp2ppp/2n3b1/2br4/6PB/1P2P2P/PB2BP2/R4RK1 b - - 0 1"


def test_a_piece_misread_in_a_position_read_cold_is_corrected_by_its_move() -> None:
    """A position read from the screen with one piece of the wrong kind: its first move looked
    illegal, and the game's next moves went unread until a resync seconds later (Tai's
    puzzle: Nxg6, hxg6 and Bf3 lost). The move that fits only with the piece corrected is
    played from the corrected position, and the game goes on from it."""
    model, rec = _setup(Side.BLACK)
    shown = chess.Board(MISREAD_PUZZLE)  # what the screen reads: a bishop on h4
    events = rec.start_from(shown.copy(), ts=0.0)
    assert any(isinstance(e, GameStarted) for e in events)
    ts = _play(model, rec, shown, ["Rd2"], bottom=Side.BLACK, ts=0.0)
    assert rec.board.piece_at(chess.H4) == chess.Piece(chess.BISHOP, chess.WHITE)
    truth = rec.board.copy()
    truth.set_piece_at(chess.H4, chess.Piece(chess.KNIGHT, chess.WHITE))
    events = []
    move = truth.push_san("Nxg6")  # the knight moves: its kind read right where it lands
    for k in range(4):
        events += _feed(model, rec, truth, Side.BLACK, ts + 0.5 + 0.1 * k, highlight=move)
    resyncs = [e for e in events if isinstance(e, Resync)]
    assert len(resyncs) == 1 and resyncs[0].reason == "the bishop read on h4 was a knight"
    assert sans_of(events) == ["Nxg6"]
    events = []
    for san in ["hxg6", "Bf3"]:
        move = truth.push_san(san)
        for _ in range(2):
            ts += 0.1
            events += _feed(model, rec, truth, Side.BLACK, ts + 1.0, highlight=move)
    assert sans_of(events) == ["hxg6", "Bf3"]


def test_a_piece_of_a_game_played_from_the_start_is_never_taken_for_misread() -> None:
    """Pieces of a game followed from the start position are known by their moves: a frame
    only another kind of piece would explain is no correction, and nothing is played."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    ts = _play(model, rec, board, ["e4", "e5"])
    fake = board.copy()
    fake.remove_piece_at(chess.F1)  # the bishop on f1 shown "moving" like a knight
    fake.set_piece_at(chess.E3, chess.Piece(chess.KNIGHT, chess.WHITE))
    events = []
    for k in range(6):
        events += _feed(model, rec, fake, Side.WHITE, ts + 0.5 + 0.1 * k)
    assert not sans_of(events) and not [e for e in events if isinstance(e, Resync)]


def test_start_position_after_a_resync_is_a_new_game() -> None:
    model, rec = _setup()
    rec.resync_hold_frames = 3
    _feed(model, rec, chess.Board(), Side.WHITE, 0.0)
    later = chess.Board(MIDDLEGAME)
    events = []
    for i in range(6):
        events += _feed(model, rec, later, Side.WHITE, 0.1 + 0.1 * i)
    assert any(isinstance(e, Resync) for e in events)
    first_id = rec.game_id
    events = []
    for i in range(2):
        events += _feed(model, rec, chess.Board(), Side.WHITE, 1.0 + 0.1 * i)
    kinds = [type(e).__name__ for e in events if isinstance(e, GameEnded | GameStarted)]
    assert kinds == ["GameEnded", "GameStarted"]
    assert rec.game_id != first_id and rec.base_fen == chess.STARTING_FEN
    fresh = chess.Board()
    move = fresh.push_san("d4")
    events = []
    for i in range(2):
        events += _feed(model, rec, fresh, Side.WHITE, 1.3 + 0.1 * i, highlight=move)
    assert sans_of(events) == ["d4"] and rec.pgn() == "1. d4"


def test_drag_frames_do_not_settle_the_side_to_move_after_a_resync() -> None:
    """A premove dragged during the bot's turn is a hypothesis, not the turn."""
    model, rec = _setup()
    rec.resync_hold_frames = 3
    _feed(model, rec, chess.Board(), Side.WHITE, 0.0)
    pos = chess.Board(MIDDLEGAME.replace(" w ", " b "))
    for i in range(5):
        _feed(model, rec, pos, Side.WHITE, 0.1 + 0.1 * i)
    assert rec.base_fen.split(" ")[0] == pos.board_fen() and rec.turn_unknown
    for i in range(3):
        drag = (chess.E4, chess.E5, 0.05, 0.05)
        _feed(model, rec, pos, Side.WHITE, 0.6 + 0.1 * i, dragging=drag)
    assert rec.turn_unknown
    move = pos.push_san("b5")
    events = []
    for i in range(2):
        events += _feed(model, rec, pos, Side.WHITE, 0.9 + 0.1 * i, highlight=move)
    assert sans_of(events) == ["b5"]
    assert not rec.turn_unknown and rec.board.turn == chess.WHITE


def test_memory_of_covered_cells_expires_when_the_board_went_unseen() -> None:
    model, rec = _setup()
    board = chess.Board()
    for i in range(3):
        _feed(model, rec, board, Side.WHITE, 0.1 * i)
    assert rec._last_seen  # the start position is remembered
    rec.board_unseen()
    _feed(model, rec, board, Side.WHITE, 5.0)
    assert rec._last_seen  # a capture stall or one doubtful tick keeps it
    for _ in range(5):
        rec.board_unseen()
    _feed(model, rec, board, Side.WHITE, 5.5)
    assert not rec._last_seen  # the board was lost or covered for half a second: forgotten


def test_contradicted_memory_is_forgotten_at_once() -> None:
    model, rec = _setup()
    board = chess.Board()
    for i in range(3):
        _feed(model, rec, board, Side.WHITE, 0.1 * i)
    d1 = row_col(chess.D1, Side.WHITE)
    assert chess.D1 in rec._last_seen
    moved = model.classify_frame(synthetic_board(board=board), GEOM)
    moved[d1[0]][d1[1]] = CellReading("empty", None, None, 1.0, 0.0, 0.0, False)
    rec.update(moved, 0.3)
    assert chess.D1 not in rec._last_seen
    covered = model.classify_frame(synthetic_board(board=board), GEOM)
    covered[d1[0]][d1[1]] = CellReading("unknown", None, None, 0.0, 0.0, 0.0, False, covered=True)
    assert chess.D1 not in rec._observe(covered)


def test_resync_onto_a_mated_position_ends_the_game() -> None:
    model, rec = _setup()
    rec.resync_hold_frames = 3
    _feed(model, rec, chess.Board(), Side.WHITE, 0.0)
    mated = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4")
    events = []
    for i in range(6):
        events += _feed(model, rec, mated, Side.WHITE, 0.1 + 0.1 * i)
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert len(ended) == 1 and ended[0].result == "1-0" and rec.finished


def test_nothing_is_played_on_a_finished_game() -> None:
    """A mate on the board ends the game once, with its result and record; after it the
    board is a review board, and only the start position begins the next game, with no
    second end for this one (test_checkmate_ends_the_game_once). A resignation likewise:
    the user's word ends the game once, with its record
    (test_user_can_end_the_game_by_resigning)."""
    model, rec = _setup()
    rec.resync_hold_s = 60.0  # a review board held long enough is another position; not here
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    events = []
    ts = 0.0
    for san in ("f3", "e5", "g4", "Qh4#"):
        move = board.push_san(san)
        for _ in range(2):
            ts += 0.1
            events += _feed(model, rec, board, Side.WHITE, ts, highlight=move)
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert len(ended) == 1 and ended[0].result == "0-1" and ended[0].pgn == "1. f3 e5 2. g4 Qh4#"
    assert rec.finished
    game_id = rec.game_id
    # Browsing back one ply and playing another move there is review, not the game.
    earlier = board.copy()
    earlier.pop()
    for _ in range(2):
        ts += 0.1
        _feed(model, rec, earlier, Side.WHITE, ts)
    assert rec.viewing == 3
    move = earlier.push_san("Nc6")
    events = []
    for _ in range(6):
        ts += 0.1
        events += _feed(model, rec, earlier, Side.WHITE, ts, highlight=move)
    assert not any(isinstance(e, MovePlayed | Takeback | GameEnded) for e in events)
    assert rec.finished and rec.game_id == game_id
    # The next game starts: no second end event for the finished game.
    events = []
    for _ in range(2):
        ts += 0.1
        events += _feed(model, rec, chess.Board(), Side.WHITE, ts)
    assert not any(isinstance(e, GameEnded) for e in events)
    assert any(isinstance(e, GameStarted) for e in events)
    assert not rec.finished
    # After a resignation the board keeps moving; none of it is this game.
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    ts = _play(model, rec, board, ["e4", "e5"])
    ended = rec.end("0-1", "user resigned")
    assert [(e.result, e.pgn) for e in ended if isinstance(e, GameEnded)] == [("0-1", "1. e4 e5")]
    assert len(ended) == 1 and rec.finished and rec.end("0-1", "again") == []
    move = board.push_san("Nf3")
    events = []
    for _ in range(3):
        ts += 0.1
        events += _feed(model, rec, board, Side.WHITE, ts, highlight=move)
    assert not any(isinstance(e, MovePlayed | GameEnded) for e in events)
    for _ in range(2):
        ts += 0.1
        events += _feed(model, rec, chess.Board(), Side.WHITE, ts)
    assert [type(e).__name__ for e in events if isinstance(e, GameEnded | GameStarted)] == [
        "GameStarted"
    ]
    assert not rec.finished


def test_position_after_a_finished_game_starts_another_game() -> None:
    """A position adopted after the game ended begins another, and that start is all: a
    Resync of the same position after it had the engine measure it twice and the coach note
    a board re-read that was never read before (the state review, 2026-10-01)."""
    model, rec = _setup()
    rec.resync_hold_frames = 3
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["f3", "e5", "g4", "Qh4#"])
    first_id = rec.game_id
    later = chess.Board(MIDDLEGAME)
    events = []
    for i in range(6):
        events += _feed(model, rec, later, Side.WHITE, 2.0 + 0.1 * i)
    kinds = [type(e).__name__ for e in events if isinstance(e, GameStarted | Resync)]
    assert kinds == ["GameStarted"]
    started = next(e for e in events if isinstance(e, GameStarted))
    assert started.kind == "position" and started.start_fen.split()[0] == later.board_fen()
    assert [e.ply for e in events if isinstance(e, PositionChanged)] == [0]
    assert rec.game_id != first_id and not rec.finished
    # The position adopted is the one on the board
    # (test_position_resync_after_a_finished_game_is_live_again).
    assert rec.board.board_fen() == later.board_fen()


def test_drag_over_the_previous_square_is_not_browsing() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    ts = _play(model, rec, board, ["e4", "e5", "Nf3"])
    for _ in range(3):
        ts += 0.1
        _feed(model, rec, board, Side.WHITE, ts, dragging=(chess.F3, chess.G1, 0.05, 0.05))
    assert rec.viewing is None


def test_arrows_over_empty_squares_do_not_stall_the_game() -> None:
    """Three arrow-like readings on empty squares, stale, and a two-ply catch-up still commits."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    ts = _play(model, rec, board, ["e4", "e5"])
    arrows = [row_col(sq, Side.WHITE) for sq in (chess.A3, chess.B3, chess.C3)]

    def frame(highlight=None):
        readings = model.classify_frame(synthetic_board(board=board, highlight=highlight), GEOM)
        for r, c in arrows:
            readings[r][c] = CellReading("piece", None, Side.WHITE, 0.5, 0.4, 240.0, settled=False)
        return readings

    for _ in range(8):
        ts += 0.1
        assert not sans_of(rec.update(frame(), ts))
    board.push_san("Nf3")
    move = board.push_san("Nc6")
    events = []
    for _ in range(4):
        ts += 0.1
        events += rec.update(frame(move), ts)
    assert sans_of(events) == ["Nf3", "Nc6"]


def test_the_clock_settles_the_side_to_move_before_a_takeback_can_mislead() -> None:
    """The live game of 2026-09-19: a position adopted with the side to move open, then a
    takeback of a move pair. Two pieces going back to where they stood read as two
    moves, in whichever order, and black-first fixed the turn on the wrong side -- the
    coach named the opponent's best move for three minutes of the player's own clock.
    The running clock settles the turn first; the same retreat then reads white-first
    and leaves white to move, as the board has it."""

    model, rec = _setup()
    # After 22... Be7 in that game: white to move, though either side could legally.
    pos = chess.Board("2kr4/pp1nb3/2npR3/2pp1r2/8/PB1P3P/1PP2P1P/R1B3K1 w - - 0 1")
    rec.start_from(pos.copy())
    assert rec.turn_unknown and rec.board.turn == chess.WHITE
    assert rec.turn_from_clock(chess.WHITE) == [], "the clock agrees with the default: quiet"
    assert not rec.turn_unknown
    # The takeback: 22. Bb3 Be7 undone, the bishops back on a4 and f8.
    undone = chess.Board("2kr1b2/pp1n4/2npR3/2pp1r2/B7/P2P3P/1PP2P1P/R1B3K1 w - - 0 1")
    events = []
    for i in range(3):
        events += _feed(model, rec, undone, Side.WHITE, 0.1 + 0.1 * i)
    assert sans_of(events) == ["Ba4", "Bf8"], "the retreat retold white-first"
    assert rec.board.turn == chess.WHITE, "white to move, as on the board"

    # The clock saying black re-announces the position with the turn.
    model, rec = _setup()
    rec.start_from(pos.copy())
    events = rec.turn_from_clock(chess.BLACK)
    assert [type(e).__name__ for e in events] == ["Resync", "PositionChanged"]
    assert isinstance(events[1], PositionChanged) and events[1].fen.split(" ")[1] == "b"
    assert rec.board.turn == chess.BLACK and not rec.turn_unknown
    assert rec.turn_from_clock(chess.WHITE) == [], "settled once; a later clock is not asked"


def test_no_move_is_told_through_wildcards_alone() -> None:
    """A board slid on after its lock was taken, the lock 0.28 of a cell off it, below the
    third of a cell validation sees: every piece reads unsettled, and the cells above two
    pieces read the tops of them, a rook on a2 and on d3. Both turn stale, and the retry
    that ignores stale cells took every unsettled piece as a wildcard: ignoring a2, the
    king's move to d3 explained d3, both its squares wildcards, and was committed -- a move
    nobody played (the geometry sweep, 2026-09-27, seed 97). A story must rest on a
    disagreement read cleanly."""
    fen = "rnbk1b1r/ppp1p2n/4p3/5p1p/8/P4P2/1PPKP1PP/RNBQ1B1R w - - 1 9"
    board = chess.Board(fen)
    rec = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE, settle_frames=2)
    rec.board = chess.Board(fen)
    rec._base_fen = fen
    rec._started = True
    shifted = {chess.A2: "R", chess.D3: "R"}
    readings = []
    for r in range(8):
        row = []
        for c in range(8):
            sq = chess.square(c, 7 - r)
            piece = board.piece_at(sq)
            symbol = piece.symbol() if piece else shifted.get(sq)
            if symbol is None:
                row.append(CellReading("empty", None, None, 1.0, 0.0, 0.0))
                continue
            colour = Side.WHITE if symbol.isupper() else Side.BLACK
            row.append(
                CellReading("piece", symbol, colour, 0.2, 0.4, 100.0, settled=False, offset=0.3)
            )
        readings.append(row)
    events = []
    for i in range(3 * rec.stale_after):
        events += rec.update(readings, 0.1 * i)
    assert sans_of(events) == [], sans_of(events)
    assert rec.board.fen() == fen
