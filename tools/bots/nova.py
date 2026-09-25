"""Nova, the human-move model the bots play with (a development tool on this PC: never in the
app, never on the phone).

Nova (novachess.ai, 2026) predicts the move a human of a given rating and style would play: a
single forward pass, no search. Its weights live in `data/models/nova/` (gitignored), with
its LICENSE: personal, research and educational use on our own hardware is allowed, and the
games it produces are ours to use for anything; it may not go into a commercial product or a
service that competes with Nova Chess (see the LICENSE, sections 2 and 5). Its ratings are
Lichess rapid's: a chess.com 1000-1400 player is about Nova 1300-1700.

    policy = NovaPolicy()
    probs = policy.probabilities([board], [Style(rating=1500)])[0]   # {move: p}
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "data" / "models" / "nova" / "nova_v3b.onnx"

_PIECE = {
    "P": 0,
    "N": 1,
    "B": 2,
    "R": 3,
    "Q": 4,
    "K": 5,
    "p": 6,
    "n": 7,
    "b": 8,
    "r": 9,
    "q": 10,
    "k": 11,
}
_PROMOTION = {chess.KNIGHT: 4096, chess.BISHOP: 8192, chess.ROOK: 12288}
RATING_MIN, RATING_MAX = 800, 2700

# The sampling temperature at which Nova's mistakes match real players' at a rating, on the
# same positions (`calibrate.py`, 600 positions a band from the authors' March 2026 Lichess
# rapid sample, 2026-09-25): raw (1.0) matches through 1700 -- 1400-1700 almost exactly
# (mean loss .044 against .043, blunders 3.8% against 3.8%) -- and higher up the model plays
# worse than its rating unless sharpened (1700-2000 about 0.75, 2000-2300 about 0.7; above
# 2300 even 0.5 is weaker than the players, as the model card warns).
_TEMPERATURES = ((1700, 1.0), (1850, 0.75), (2150, 0.7), (2500, 0.5))


def temperature_for(rating: float) -> float:
    """The calibrated temperature for a rating, interpolated between the measured bands."""
    points = _TEMPERATURES
    if rating <= points[0][0]:
        return points[0][1]
    for (r0, t0), (r1, t1) in zip(points, points[1:], strict=False):
        if rating <= r1:
            return t0 + (t1 - t0) * (rating - r0) / (r1 - r0)
    return points[-1][1]


@dataclass(frozen=True)
class Style:
    """Who is playing: a rating on Nova's (Lichess rapid) scale, and the two style dials --
    `classical` (higher: classical main lines; lower: hypermodern) and `aggression`
    (captures, territory, pressure on the king)."""

    rating: float
    classical: float = 0.5
    aggression: float = 0.5

    def conditioning(self) -> np.ndarray:
        norm = (self.rating - RATING_MIN) / (RATING_MAX - RATING_MIN)
        return np.array(
            [min(1.0, max(0.0, norm)), self.classical, self.aggression], dtype=np.float32
        )


def planes(board: chess.Board) -> np.ndarray:
    """The board as Nova reads it: twelve piece planes (rank-major, a1 at [0, 0]), the side to
    move, four castling rights, and the en-passant file (only when a capture there is legal,
    as a FEN from python-chess writes it)."""
    p = np.zeros((18, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        p[_PIECE[piece.symbol()], chess.square_rank(square), chess.square_file(square)] = 1.0
    if board.turn == chess.WHITE:
        p[12].fill(1.0)
    if board.has_kingside_castling_rights(chess.WHITE):
        p[13].fill(1.0)
    if board.has_queenside_castling_rights(chess.WHITE):
        p[14].fill(1.0)
    if board.has_kingside_castling_rights(chess.BLACK):
        p[15].fill(1.0)
    if board.has_queenside_castling_rights(chess.BLACK):
        p[16].fill(1.0)
    if board.ep_square is not None and board.has_legal_en_passant():
        p[17, 0, chess.square_file(board.ep_square)] = 1.0
    return p


def move_index(move: chess.Move) -> int:
    return _PROMOTION.get(move.promotion or 0, 0) + move.from_square * 64 + move.to_square


class NovaPolicy:
    """The model on the CPU, about 27 ms a position. The released ONNX graph fixes its batch at
    one position (a reshape inside it is exported for batch 1), so positions go one at a time."""

    def __init__(self, model: Path = MODEL, threads: int = 1) -> None:
        import onnxruntime as ort

        if not model.exists():
            raise FileNotFoundError(f"Nova's weights are not at {model} (see reference-nova-bots)")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.log_severity_level = 3
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])

    def logits(self, boards: Sequence[chess.Board], styles: Sequence[Style]) -> np.ndarray:
        out = []
        for board, style in zip(boards, styles, strict=True):
            feed = {"positions": planes(board)[None], "conditioning": style.conditioning()[None]}
            out.append(np.asarray(self.session.run(None, feed)[0])[0])
        return np.stack(out)

    def probabilities(
        self, boards: Sequence[chess.Board], styles: Sequence[Style], temperature: float = 1.0
    ) -> list[dict[chess.Move, float]]:
        """For each board, its legal moves with the probability a player of that style plays
        them, sharpened (temperature below 1) or flattened (above 1)."""
        out: list[dict[chess.Move, float]] = []
        for board, logits in zip(boards, self.logits(boards, styles), strict=True):
            moves = list(board.legal_moves)
            if not moves:
                out.append({})
                continue
            z = np.array([logits[move_index(m)] for m in moves], dtype=np.float64) / temperature
            p = np.exp(z - z.max())
            p /= p.sum()
            out.append(dict(zip(moves, p.tolist(), strict=True)))
        return out
