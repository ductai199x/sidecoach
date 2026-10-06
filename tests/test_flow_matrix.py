"""Every combination of what a player does on a board, against the reconciler: a seeded sweep.

Tai (2026-10-05): a game or a puzzle, with takebacks or not, looking back through the move
list or not, premoves or not -- every one of the 2 x 2 x 2 x 2 combinations must be followed.
Each case plays a random game on the site of `test_premove_sweep` (its `_Site`: chess.com's
or lichess's queue, drawn the site's way, tints of the seed's colours), with what the case
allows mixed in at random:

- a takeback: the site steps the game back one or two plies, to the user's move, and the
  user then plays on (a different move: a takeback shows only by the move played after it);
- looking back: the board shows an earlier position of the game for a while, then the live
  one again (on chess.com browsing clears the premove queue);
- premoves: queues made, cancelled and played, as in the premove sweep;
- a puzzle: a position read cold (`Reconciler.start_from`, its side to move open), the site's
  opening move, and the site's replies on the next picture or the same one.

What must hold is Tai's order, as in the premove sweep: on every picture, a side to move the
stories disagree on is told pending; whenever the record says it has caught up with the
screen, it holds the game's position and side to move; and at the end, once the board is
still and the opponent has moved after the user, the game exactly. A move told that was not
played is counted, not asserted. A handful of seeds run in the suite; `--seeds` sweeps:

    uv run pytest tests/test_flow_matrix.py -q -n auto --seeds 100
"""

from __future__ import annotations

import itertools
import random

import chess
import pytest

from sidecoach.events import Side
from tests.games import sans_of
from tests.synth import synthetic_board
from tests.test_premove_sweep import FRAMES, POSITIONS, STILL, _Site

SEEDS = (1, 2, 3)
CASES = [
    dict(zip(("puzzle", "takebacks", "lookbacks", "premoves"), flags, strict=True))
    for flags in itertools.product((False, True), repeat=4)
]


def _label(case: dict[str, bool]) -> str:
    return "-".join(name if on else f"no{name}" for name, on in case.items())


class _Flow(_Site):
    """The premove sweep's site, with takebacks, looking back and puzzles."""

    def take_back(self) -> None:
        """The site steps the game back to the user's last move or the one before, the queue
        cleared; the user's next move is a different one."""
        self.queue = []
        plies = 1 if self.board.turn != self.user else 2
        if len(self.board.move_stack) < plies:
            return
        for _ in range(plies):
            self.board.pop()
        self.played.append(f"<takeback {plies}>")
        self.last = self.board.peek() if self.board.move_stack else None
        self.show(self.rng.choice(FRAMES))

    def look_back(self) -> None:
        """The move list stepped back a few plies and shown a while, then the live board."""
        plies = len(self.board.move_stack)
        if plies == 0:
            return
        self.queue = []
        back = self.rng.randint(1, min(plies, 4))
        earlier = self.board.copy()
        for _ in range(back):
            earlier.pop()
        self.played.append(f"<look back {back}>")
        highlight = earlier.peek() if earlier.move_stack else None
        for _ in range(self.rng.choice((2, 3, 5))):
            frame = synthetic_board(
                board=earlier, bottom=self.game.bottom, highlight=highlight, **self.style
            )
            self.game.process(frame)
            self.check()
        self.show(self.rng.choice((2, 3)))


def _puzzle(site: _Flow, fen: str) -> None:
    """The board replaced by a puzzle's position read cold, the opponent to move; its opening
    move a moment later."""
    board = chess.Board(fen)
    board.turn = not site.user
    if not board.is_valid():
        board.turn = site.user
    game, rec = site.game, site.game.rec
    game.board = board
    placement = board.copy(stack=False)
    game.events += rec.start_from(placement, ts=game.ts)
    site.history = {site._key(board)}
    site.seen = {board.board_fen()}
    site.last = None
    site.show(3)
    if board.turn != site.user:
        site.opponent_moves()


@pytest.mark.vision
@pytest.mark.parametrize("case", CASES, ids=[_label(c) for c in CASES])
def test_every_flow_is_followed(seed: int, case: dict[str, bool]) -> None:
    rng = random.Random(f"{seed}-{_label(case)}")
    bottom = rng.choice((Side.WHITE, Side.BLACK))
    site = _Flow(rng, chess.STARTING_FEN, bottom)
    if case["puzzle"]:
        _puzzle(site, rng.choice(POSITIONS))
    elif site.board.turn != site.user:
        site.opponent_moves()
    for _ in range(12):
        if site.board.is_game_over():
            break
        if site.board.turn == site.user and not site.queue:
            roll = rng.random()
            if case["takebacks"] and roll < 0.2 and len(site.board.move_stack) >= 2:
                site.take_back()
                continue
            if case["lookbacks"] and roll > 0.8:
                site.look_back()
            site.user_moves()
            continue
        if case["premoves"] and rng.random() < 0.7:
            site.queue_up()
            if site.queue and rng.random() < 0.3:
                site.cancel()
        if case["lookbacks"] and rng.random() < 0.15:
            site.look_back()
        if case["takebacks"] and rng.random() < 0.15:
            site.take_back()
            continue
        site.opponent_moves()
    site.queue = []
    site.show(STILL)
    rec, board = site.game.rec, site.board
    where = (seed, case, bottom, site.style, site.played, sans_of(site.game.events), site.lies)
    for played in range(3):
        if board.is_game_over() or (played >= 2 and board.turn == site.user):
            break
        move = next((m for m in board.legal_moves if not site._repeats(m)), None)
        if move is None:
            return
        site._push(move)
        site.show(4)
    assert site._key(rec.board) == site._key(board), ("position", *where)
    assert not (rec.turn_unknown or rec.turn_pending), ("turn", *where)
