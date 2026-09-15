import chess
import numpy as np

from sidecoach.events import GameEnded, GameStarted, MovePlayed, Region, Resync, Side, Takeback
from sidecoach.state.reconcile import Reconciler, cell_agrees
from sidecoach.vision.pieces import CellReading, ThemeModel
from sidecoach.vision.squares import BoardGeometry, row_col
from tests.synth import DARK, LIGHT, RED, YELLOW, synthetic_board, tint

GEOM = BoardGeometry(Region(300, 120, 512, 512), 64.0)


def _setup(bottom: Side = Side.WHITE) -> tuple[ThemeModel, Reconciler]:
    frame = synthetic_board(bottom=bottom)
    model = ThemeModel.from_start_position(frame, GEOM, bottom)
    assert model is not None
    return model, Reconciler(bottom=bottom, user_side=bottom, settle_frames=2)


def _feed(model, rec, board, bottom, ts, highlight=None, dragging=None):
    frame = synthetic_board(board=board, bottom=bottom, highlight=highlight, dragging=dragging)
    return rec.update(model.classify_frame(frame, GEOM), ts)


def _played(events) -> list[str]:
    return [e.san for e in events if isinstance(e, MovePlayed)]


def _play(model, rec, board, sans, bottom=Side.WHITE, ts=0.0):
    """Push moves on `board` and feed each twice with its highlight so it commits."""
    for san in sans:
        move = board.push_san(san)
        for _ in range(2):
            ts += 0.1
            _feed(model, rec, board, bottom, ts, highlight=move)
    return ts


def test_moves_are_committed_after_settling() -> None:
    model, rec = _setup()
    board = chess.Board()
    events = _feed(model, rec, board, Side.WHITE, 0.0)
    assert any(isinstance(e, GameStarted) for e in events)

    by_user = []
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "O-O", "Nxe4", "d4", "exd4"):
        move = board.parse_san(san)
        board.push(move)
        first = _feed(model, rec, board, Side.WHITE, 0.1, highlight=move)
        assert not _played(first), "committed on one frame"
        second = _feed(model, rec, board, Side.WHITE, 0.2, highlight=move)
        assert _played(second) == [san]
        by_user += [e.by_user for e in second if isinstance(e, MovePlayed)]
    assert rec.board.fen() == board.fen()
    assert rec.pgn() == "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. O-O Nxe4 5. d4 exd4"
    assert by_user == [True, False] * 5


def test_first_move_is_not_committed_mid_drag() -> None:
    """The pawn is drawn over e4 under the cursor, off centre: still in the air."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    for i, (dx, dy) in enumerate(((0.3, 0.1), (0.2, -0.2), (-0.25, 0.15), (0.1, 0.3))):
        drag = (chess.E2, chess.E4, dx, dy)
        assert not _played(_feed(model, rec, board, Side.WHITE, 0.1 * i, dragging=drag))
    move = board.push_san("e4")
    _feed(model, rec, board, Side.WHITE, 1.0, highlight=move)
    assert _played(_feed(model, rec, board, Side.WHITE, 1.1, highlight=move)) == ["e4"]


def test_hover_outline_alone_blocks_a_commit() -> None:
    """A piece exactly centred but the hovered square still outlined: not dropped yet."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    for i in range(4):
        drag = (chess.E2, chess.E4, 0.0, 0.0)
        assert not _played(_feed(model, rec, board, Side.WHITE, 0.1 * i, dragging=drag))
    move = board.push_san("e4")
    for ts in (1.0, 1.1):
        events = _feed(model, rec, board, Side.WHITE, ts, highlight=move)
    assert _played(events) == ["e4"]


def test_drag_in_progress_is_held() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    lifted = board.copy()
    lifted.remove_piece_at(chess.E2)
    for i in range(5):
        assert not _played(_feed(model, rec, lifted, Side.WHITE, 0.1 * i))
    assert rec.board.fen() == chess.Board().fen()


def test_long_hover_over_destination_is_not_a_move() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4"])

    for i in range(8):
        drag = (chess.E7, chess.E5, 0.15, -0.2)
        assert not _played(_feed(model, rec, board, Side.WHITE, 0.3 + 0.1 * i, dragging=drag))
    dropped = board.push_san("e5")
    _feed(model, rec, board, Side.WHITE, 1.2, highlight=dropped)
    assert _played(_feed(model, rec, board, Side.WHITE, 1.3, highlight=dropped)) == ["e5"]


def test_browsing_history_is_not_a_takeback() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5", "Nf3", "Nc6"])
    latest = board.fen()

    def position(ply: int) -> chess.Board:
        earlier = board.copy()
        while len(earlier.move_stack) > ply:
            earlier.pop()
        return earlier

    # Step back to ply 2, then ply 1, then forward to ply 3, then back to the latest.
    ts = 1.0
    for ply, expect_view in ((2, 2), (1, 1), (3, 3), (4, None)):
        for _ in range(6):  # long enough for cells to go stale if they were allowed to
            ts += 0.1
            events = _feed(model, rec, position(ply), Side.WHITE, ts)
            assert not any(isinstance(e, Takeback | MovePlayed) for e in events)
        assert rec.viewing == expect_view, (ply, rec.viewing)
    assert rec.board.fen() == latest


def test_opponent_move_while_browsing_snaps_to_latest() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5", "Nf3"])
    earlier = board.copy()
    earlier.pop()
    earlier.pop()
    for ts in (1.0, 1.1):
        _feed(model, rec, earlier, Side.WHITE, ts)
    assert rec.viewing == 1
    # The bot replies: the site jumps to the latest position plus the new move.
    reply = board.push_san("Nc6")
    events = _feed(model, rec, board, Side.WHITE, 2.0, highlight=reply)
    events += _feed(model, rec, board, Side.WHITE, 2.1, highlight=reply)
    assert _played(events) == ["Nc6"]
    assert rec.viewing is None
    assert rec.board.fen() == board.fen()


def test_takeback_is_a_new_move_from_an_earlier_position() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5", "Nf3", "Nc6"])

    # Bot-game takeback: the user's Nf3 and the reply vanish, then a different move.
    board.pop()
    board.pop()
    for ts in (1.0, 1.1):
        _feed(model, rec, board, Side.WHITE, ts)
    assert rec.viewing == 2
    move = board.push_san("d4")
    events = _feed(model, rec, board, Side.WHITE, 1.2, highlight=move)
    assert not any(isinstance(e, Takeback) for e in events)
    events = _feed(model, rec, board, Side.WHITE, 1.3, highlight=move)
    takebacks = [e for e in events if isinstance(e, Takeback)]
    assert len(takebacks) == 1 and takebacks[0].plies == 2
    assert _played(events) == ["d4"]
    assert rec.pgn() == "1. e4 e5 2. d4"
    assert rec.board.fen() == board.fen()


def test_animation_across_a_still_highlighted_square_is_not_a_move() -> None:
    """The knight animates b5 to d6 across c7, which still carries the last move's highlight."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "c6", "Nc3", "d5", "Nb5"])
    reply = board.push_san("c5")  # c6-c5 highlights c6 and c5
    _play(model, rec, board, [], ts=1.0)
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
        assert not _played(events), "animation frame taken as a move"
    # Landed: b5 and d6 highlighted, check on the black king.
    move = board.push_san("Nd6+")
    _feed(model, rec, board, Side.WHITE, 2.0, highlight=move)
    assert _played(_feed(model, rec, board, Side.WHITE, 2.1, highlight=move)) == ["Nd6+"]


def test_selected_piece_does_not_block_the_opponents_move() -> None:
    """The bot's move is on the board and the user has already clicked their next piece."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5", "Nf3"])
    reply = board.push_san("Nc6")
    frame = synthetic_board(board=board, highlight=reply)
    readings = model.classify_frame(frame, GEOM)
    r, c = row_col(chess.F1, Side.WHITE)  # the user selected the f1 bishop
    old = readings[r][c]
    readings[r][c] = CellReading(
        old.kind, old.symbol, old.colour, old.confidence, old.fg_fraction, old.luminance, True
    )
    events = rec.update(readings, 1.0) + rec.update(readings, 1.1)
    assert _played(events) == ["Nc6"]


def test_highlight_kinds() -> None:
    model = ThemeModel.from_start_position(synthetic_board(), GEOM, Side.WHITE)
    assert model is not None
    light = np.array(LIGHT, dtype=np.float32)
    assert model.highlight_kind(light, 0, 0) == "none"
    assert model.highlight_kind(np.array(tint(LIGHT, YELLOW), dtype=np.float32), 0, 0) == "move"
    assert model.highlight_kind(np.array(tint(LIGHT, RED), dtype=np.float32), 0, 0) == "red"
    assert model.highlight_kind(np.array([245, 245, 245], dtype=np.float32), 0, 0) == "foreign"
    assert model.highlight_kind(np.array([200, 200, 200], dtype=np.float32), 0, 0) == "foreign"
    dark = np.array(DARK, dtype=np.float32)
    assert model.highlight_kind(dark, 0, 1) == "none"
    assert model.highlight_kind(np.array(tint(DARK, YELLOW), dtype=np.float32), 0, 1) == "move"
    assert model.highlight_kind(np.array([200, 200, 200], dtype=np.float32), 0, 1) == "foreign"


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
    assert _played(events) == ["e5"]


def test_two_move_catch_up_when_one_move_was_missed() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    board.push_san("e4")
    reply = board.push_san("c5")
    events = []
    for ts in (0.1, 0.2, 0.3):
        events += _feed(model, rec, board, Side.WHITE, ts, highlight=reply)
    assert _played(events) == ["e4", "c5"]
    assert rec.board.fen() == board.fen()


def test_four_ply_catch_up_after_window_was_hidden() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5"])
    for san in ("Nf3", "Nc6", "Bb5", "a6"):
        board.push_san(san)
    last = board.peek()
    for ts in (1.0, 1.1, 1.2):
        _feed(model, rec, board, Side.WHITE, ts, highlight=last)
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
        assert not _played(rec.update(readings, ts))


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
    assert _played(events) == ["e4"]


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
    assert _played(_feed(model, rec2, promo, Side.WHITE, 0.0)) == ["a8=N"]


def test_checkmate_ends_the_game_once() -> None:
    from sidecoach.events import GameEnded

    model, rec = _setup()
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

    # The next game starts: no second end event for the finished game.
    events = []
    for ts in (2.0, 2.1):
        events += _feed(model, rec, chess.Board(), Side.WHITE, ts)
    assert not any(isinstance(e, GameEnded) for e in events)
    assert any(isinstance(e, GameStarted) for e in events)
    assert not rec.finished


def test_move_from_start_position_mid_game_is_a_new_game() -> None:
    from sidecoach.events import GameEnded

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
    assert _played(events) == ["d4"]
    assert rec.game_id != first_id and rec.pgn() == "1. d4"


def test_position_resync_after_a_long_gap() -> None:
    """Many plies happened while the board was hidden: adopt the position, then continue."""
    from sidecoach.events import Resync

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
    assert not _played(events)

    # The next move settles whose turn it was and is committed normally.
    move = board.push_san("d6")
    events = []
    for ts in (2.0, 2.1):
        events += _feed(model, rec, board, Side.WHITE, ts, highlight=move)
    assert _played(events) == ["d6"]
    assert not rec.turn_unknown
    assert rec.board.fen().split()[0] == board.board_fen()
    assert rec.pgn() == "1... d6"


def test_move_from_adopted_position_is_a_takeback_not_a_new_game() -> None:
    """After a resync, browsing back to the adopted position and playing on is a takeback."""
    from sidecoach.events import GameEnded, Resync

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
    assert _played(events) == ["d3"]
    assert rec.board.fen().split()[0] == adopted.board_fen()


def test_takeback_move_legal_only_in_the_earlier_position() -> None:
    """Regression: the branch move was once checked against the latest board and crashed."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5", "Nf3", "Nc6", "Bb5"])
    earlier = board.copy()
    earlier.pop()  # back to after Nc6: the bishop is on f1 again
    for ts in (1.0, 1.1):
        _feed(model, rec, earlier, Side.WHITE, ts)
    assert rec.viewing == 4
    move = earlier.push_san("Bc4")  # legal only with the bishop on f1
    events = []
    for ts in (1.2, 1.3):
        events += _feed(model, rec, earlier, Side.WHITE, ts, highlight=move)
    assert any(isinstance(e, Takeback) and e.plies == 1 for e in events)
    assert _played(events) == ["Bc4"]
    assert rec.board.fen() == earlier.fen()


def test_illegal_commit_is_refused_not_fatal() -> None:
    from sidecoach.events import Resync

    rec = Reconciler(bottom=Side.WHITE)
    rec.start()
    events = rec._commit(chess.Move.from_uci("d4g4"))
    assert len(events) == 1 and isinstance(events[0], Resync)
    assert rec.board.fen() == chess.Board().fen()


def test_colour_only_reading_means_occupied_not_a_colour() -> None:
    reading = CellReading("piece", None, Side.WHITE, 0.5, 0.5, 240.0, False)
    assert cell_agrees(reading, chess.Piece.from_symbol("q"))
    assert cell_agrees(reading, chess.Piece.from_symbol("Q"))
    assert not cell_agrees(reading, None)


def test_resync_needs_several_persistent_cells() -> None:
    model, rec = _setup()
    rec.stale_after = 2
    rec.resync_after = 4
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    weird = chess.Board("8/8/8/3k4/8/8/8/3K4 w - - 0 1")
    events = []
    for i in range(6):
        events += _feed(model, rec, weird, Side.WHITE, 0.1 * i)
    assert any(isinstance(e, Resync) for e in events)

    model2, rec2 = _setup()
    rec2.stale_after = 2
    rec2.resync_after = 4
    _feed(model2, rec2, chess.Board(), Side.WHITE, 0.0)
    lifted = chess.Board()
    lifted.remove_piece_at(chess.D2)
    events = []
    for i in range(12):
        events += _feed(model2, rec2, lifted, Side.WHITE, 0.1 * i)
    assert not any(isinstance(e, Resync) for e in events)


def test_black_at_bottom_user_black() -> None:
    model, rec = _setup(Side.BLACK)
    board = chess.Board()
    _feed(model, rec, board, Side.BLACK, 0.0)
    for san in ("e4", "c5", "Nf3"):
        move = board.push_san(san)
        _feed(model, rec, board, Side.BLACK, 0.1, highlight=move)
        assert _played(_feed(model, rec, board, Side.BLACK, 0.2, highlight=move)) == [san]
    assert rec.board.fen() == board.fen()


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
    assert _played(events) == ["Qxf7#"]


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
            played.extend(_played(_feed(model, rec, board, Side.WHITE, ts, dragging=dragging)))

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
    assert _played(events) == ["Kh6", "Qaf8#"]
    assert any(isinstance(e, GameEnded) for e in events)


def test_browsing_to_the_start_of_a_long_game_is_not_a_new_game() -> None:
    """The move list's first entry shows the start position; a game of 42 plies is still on."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    line = [
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
        "d4",
        "Nbd7",
        "Nbd2",
        "Bb7",
        "Bc2",
        "Re8",
        "Nf1",
        "Bf8",
        "Ng3",
        "g6",
        "a4",
        "c5",
        "d5",
        "c4",
        "Bg5",
        "h6",
        "Be3",
        "Nc5",
        "Qd2",
        "h5",
        "Bg5",
        "Be7",
        "Kh2",
        "Kg7",
    ]
    assert len(line) > rec.history_depth
    ts = _play(model, rec, board, line)
    assert len(rec.board.move_stack) == len(line)
    events = []
    for i in range(6):
        events += _feed(model, rec, chess.Board(), Side.WHITE, ts + 0.1 * (i + 1))
    assert not any(isinstance(e, GameStarted | GameEnded | Takeback) for e in events)
    assert rec.viewing == 0
    assert len(rec.board.move_stack) == len(line)


def test_position_resync_after_a_finished_game_is_live_again() -> None:
    model, rec = _setup()
    rec.resync_hold_frames = 3
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["f3", "e5", "g4", "Qh4#"])
    assert rec.finished
    later = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    events = []
    for i in range(6):
        events += _feed(model, rec, later, Side.WHITE, 2.0 + 0.1 * i)
    assert any(isinstance(e, Resync) for e in events)
    assert rec.board.board_fen() == later.board_fen()
    assert not rec.finished


def test_queue_path_shortest_unique_ways() -> None:
    rec = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE)
    rec.board = chess.Board("4k3/P7/8/3p4/4P3/8/P7/4K3 w - - 0 1")
    # A pawn takes the double step when that is shorter.
    path = rec._queue_path(chess.A2, chess.A5, "P")
    assert path is not None and [m.uci() for m in path] == ["a2a4", "a4a5"]
    # A pawn only captures on the way to its target, never through an empty square.
    assert rec._queue_path(chess.E4, chess.D5, "P") is not None
    assert rec._queue_path(chess.E4, chess.F5, "P") is None
    # Promotion to the piece drawn on the last rank; a pawn drawn there is not decided.
    path = rec._queue_path(chess.A7, chess.A8, "N")
    assert path is not None and [m.uci() for m in path] == ["a7a8n"]
    assert rec._queue_path(chess.A7, chess.A8, "P") is None
    assert rec._queue_path(chess.A7, chess.A8, "q") is None  # the wrong colour
    # A king two squares away has several shortest ways: nothing is assumed.
    assert rec._queue_path(chess.E1, chess.E3, "K") is None
    assert rec._queue_path(chess.E1, chess.E2, "K") is not None
    # A blocked file has no way at all.
    rec.board = chess.Board("4k3/8/8/8/8/p7/P7/4K3 w - - 0 1")
    assert rec._queue_path(chess.A2, chess.A4, "P") is None


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
    assert not _played(events)
    for i in range(2):
        events += _feed(model, rec, board, Side.WHITE, 2.0 + 0.1 * i)
    assert _played(events) == ["e5"]


def test_user_can_end_the_game_by_resigning() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    _play(model, rec, board, ["e4", "e5"])
    events = rec.end("0-1", "user resigned")
    assert len(events) == 1 and isinstance(events[0], GameEnded)
    assert events[0].result == "0-1" and events[0].pgn == "1. e4 e5"
    assert rec.finished and rec.end("0-1", "again") == []


def test_takeback_with_a_quick_reply_is_not_told_as_retreats() -> None:
    """Browse back two plies, play d4, the bot answers at once: a takeback, not Ng1 Nb8."""
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    ts = _play(model, rec, board, ["e4", "e5", "Nf3", "Nc6"])
    earlier = chess.Board()
    for san in ("e4", "e5"):
        earlier.push_san(san)
    for _ in range(2):
        ts += 0.1
        _feed(model, rec, earlier, Side.WHITE, ts)
    assert rec.viewing == 2
    earlier.push_san("d4")
    earlier.push_san("exd4")
    events = []
    for _ in range(4):
        ts += 0.1
        events += _feed(model, rec, earlier, Side.WHITE, ts, highlight=earlier.peek())
    takebacks = [e for e in events if isinstance(e, Takeback)]
    assert len(takebacks) == 1 and takebacks[0].plies == 2
    assert _played(events) == ["d4", "exd4"]
    assert rec.pgn() == "1. e4 e5 2. d4 exd4"


MIDDLEGAME = "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"


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
    assert _played(events) == ["d4"] and rec.pgn() == "1. d4"


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
    assert _played(events) == ["b5"]
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
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    ts = _play(model, rec, board, ["f3", "e5", "g4", "Qh4#"])
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
    # After a resignation the board keeps moving; none of it is this game.
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    ts = _play(model, rec, board, ["e4", "e5"])
    rec.end("0-1", "user resigned")
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


def test_position_after_a_finished_game_starts_another_game() -> None:
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
    assert kinds == ["GameStarted", "Resync"]
    assert rec.game_id != first_id and not rec.finished


def test_drag_over_the_previous_square_is_not_browsing() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    ts = _play(model, rec, board, ["e4", "e5", "Nf3"])
    for _ in range(3):
        ts += 0.1
        _feed(model, rec, board, Side.WHITE, ts, dragging=(chess.F3, chess.G1, 0.05, 0.05))
    assert rec.viewing is None


def test_queue_path_never_passes_through_the_en_passant_square() -> None:
    rec = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE)
    rec.board = chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2")
    assert rec._queue_path(chess.E5, chess.D7, "P") is None
    assert rec._queue_path(chess.E5, chess.E7, "P") is not None


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
        assert not _played(rec.update(frame(), ts))
    board.push_san("Nf3")
    move = board.push_san("Nc6")
    events = []
    for _ in range(4):
        ts += 0.1
        events += rec.update(frame(move), ts)
    assert _played(events) == ["Nf3", "Nc6"]


def test_think_time_comes_from_the_frame_timestamps() -> None:
    model, rec = _setup()
    board = chess.Board()
    _feed(model, rec, board, Side.WHITE, 0.0)
    events = []
    ts = 0.0
    for san in ("e4", "e5"):
        move = board.push_san(san)
        for _ in range(2):
            ts += 0.1
            events += _feed(model, rec, board, Side.WHITE, ts, highlight=move)
    assert [e.think_ms for e in events if isinstance(e, MovePlayed)] == [200, 200]
