"""The vision never locks onto the panel's own board: it carries a row-parity signature
(odd rows three units bluer) that a real board does not, and the tracker vetoes it."""

from __future__ import annotations

import numpy as np

from sidecoach.events import Region
from sidecoach.vision.ownboard import SHADE_STEP, is_own_board
from sidecoach.vision.tracker import BoardTracker

CELL = 60
LIGHT = (0xD2, 0xEE, 0xEE)  # BGR of #eeeed2
DARK = (0x56, 0x96, 0x76)  # BGR of #769656


def board_image(
    *, signed: bool, offset: tuple[int, int] = (40, 30), pieces: bool = True
) -> np.ndarray:
    """A screen with one board on it, the panel's (signed) or a plain one."""
    img = np.full((8 * CELL + 200, 8 * CELL + 200, 3), (45, 45, 45), dtype=np.uint8)
    ox, oy = offset
    rng = np.random.default_rng(1)
    for r in range(8):
        for c in range(8):
            colour = np.array(DARK if (r + c) % 2 == 0 else LIGHT, dtype=np.int16)
            if signed and r % 2 == 1:
                colour[0] += SHADE_STEP
            y0, x0 = oy + r * CELL, ox + c * CELL
            img[y0 : y0 + CELL, x0 : x0 + CELL] = colour.astype(np.uint8)
            # Pieces on two rows: dark blobs in the middle of the cell.
            if pieces and r in (1, 6):
                img[y0 + 12 : y0 + 48, x0 + 12 : x0 + 48] = (20, 20, 20)
    # A little sensor-like noise that must not defeat the signature.
    noise = rng.integers(-1, 2, size=img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_the_signature_is_recognised_and_a_plain_board_is_not() -> None:
    region = Region(40, 30, 8 * CELL, 8 * CELL)
    assert is_own_board(board_image(signed=True), region, CELL)
    assert is_own_board(board_image(signed=True, pieces=False), region, CELL)
    assert not is_own_board(board_image(signed=False), region, CELL)
    # A global colour shift (a warm screen at night) leaves the relative step intact.
    warm = np.clip(board_image(signed=True).astype(np.int16) + (-6, 0, 4), 0, 255).astype(np.uint8)
    assert is_own_board(warm, region, CELL)


def test_the_tracker_vetoes_the_panels_board_and_keeps_a_real_one() -> None:
    vetoing = BoardTracker(reject=is_own_board)
    assert vetoing.update(board_image(signed=True)) is None, "the panel's board was adopted"
    assert vetoing.rejected >= 1
    # After a miss the tracker backs off for a few frames before searching again, as it would
    # on a live screen; a real board is then adopted.
    real = board_image(signed=False)
    assert any(vetoing.update(real) is not None for _ in range(30)), "a real board was refused"
    plain = BoardTracker()
    assert plain.update(board_image(signed=True)) is not None  # without the veto it locks


def test_the_panels_board_near_the_old_size_does_not_hide_a_resized_game() -> None:
    """The game board resized out of the near search's range with the panel's board at the
    old size: the near search found only the panel's board, the veto dropped it after the
    search had stopped there, and the full search never ran (round 2 finding 1)."""
    import chess

    from tests.synth import PIECE_STYLES, busy_background, render_board

    def frame(game_cell: int) -> np.ndarray:
        out = busy_background(1600, 900, 3)
        render_board(out, 40, 40, game_cell, chess.Board(), style=PIECE_STYLES["cburnett"])
        x0, y0, cell = 900, 100, 72
        render_board(
            out, x0, y0, cell, chess.Board(None), light=(200, 220, 230), dark=(90, 140, 120)
        )
        for r in range(1, 8, 2):  # the signature: odd display rows a few units bluer
            rows = slice(y0 + r * cell, y0 + (r + 1) * cell)
            band = out[rows, x0 : x0 + 8 * cell, 0].astype(np.int16)
            out[rows, x0 : x0 + 8 * cell, 0] = np.clip(band + SHADE_STEP, 0, 255)
        return out

    tracker = BoardTracker(reject=is_own_board)
    for _ in range(3):
        tracker.update(frame(80))
    board = None
    for _ in range(60):
        board = tracker.update(frame(50))
    assert board is not None and tracker.verified, board
    assert abs(board.region.x - 40) <= 2 and abs(board.cell_px - 50) <= 1.5, board
