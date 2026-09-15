"""From the analysis of a position to the facts a coach may draw on.

The disclosure machine filters these by level, so what goes in here decides what can ever
come out. Each field is written in the words the coach would use: the theme is a category
("a loose piece is involved"), the region is a part of the board with one anchor square,
the piece is named but never its destination, and the candidates and the line are moves in
the notation a person reads.
"""

from __future__ import annotations

from collections.abc import Sequence

import chess

from sidecoach.analysis.engine import san_line
from sidecoach.coach.disclosure import PositionFacts
from sidecoach.events import Analysis, MotifTag

# Tactical tags a coach would name as the theme, most specific first. Everything that a
# move merely is (capture, check, quietMove) says nothing about the idea and is left out.
THEMES: tuple[tuple[str, str], ...] = (
    ("backRankMate", "a weak back rank"),
    ("smotheredMate", "a smothered mate"),
    ("mateIn1", "a mating attack"),
    ("mateIn2", "a mating attack"),
    ("mateIn3", "a mating attack"),
    ("mateIn4", "a mating attack"),
    ("mateIn5", "a mating attack"),
    ("mate", "a mating attack"),
    ("fork", "a fork"),
    ("absolutePin", "a pin"),
    ("pin", "a pin"),
    ("skewer", "a skewer"),
    ("discoveredAttack", "a discovered attack"),
    ("doubleCheck", "a double check"),
    ("hangingPiece", "a loose piece"),
    ("trappedPiece", "a trapped piece"),
    ("capturingDefender", "a defender that can be removed"),
    ("deflection", "a defender that can be pulled away"),
    ("attraction", "a piece that can be lured onto a bad square"),
    ("sacrifice", "a sacrifice"),
    ("exposedKing", "king safety"),
    ("attackingF2F7", "the weak point next to the king"),
    ("promotion", "a passed pawn"),
    ("advancedPawn", "a passed pawn"),
    ("xRayAttack", "a piece attacked through another"),
    ("interference", "a line that can be cut"),
    ("clearance", "a square that can be cleared"),
)
_THEME_RANK = {name: index for index, (name, _) in enumerate(THEMES)}

PIECE_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
# Candidates within this much of the best line are "the near-equal set".
CANDIDATE_MARGIN = 0.02


def phase_of(board: chess.Board, in_book: bool = False) -> str:
    """Opening, middlegame or endgame, from the book and what is left on the board."""
    if in_book or board.fullmove_number <= 8:
        return "opening"
    material = sum(
        _VALUES[piece.piece_type]
        for piece in board.piece_map().values()
        if piece.piece_type not in (chess.PAWN, chess.KING)
    )
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    if material <= 13 or (queens == 0 and material <= 20):
        return "endgame"
    return "middlegame"


def material_text(board: chess.Board) -> str:
    """The material balance as the side to move would say it."""
    us = not_us = 0
    for piece in board.piece_map().values():
        value = _VALUES.get(piece.piece_type, 0)
        if piece.color == board.turn:
            us += value
        else:
            not_us += value
    diff = us - not_us
    if diff == 0:
        return "material is level"
    amount = abs(diff)
    if amount == 1:
        what = "a pawn"
    elif amount == 2:
        what = "two pawns"
    elif amount == 3:
        what = "a minor piece"
    elif amount in (4, 5):
        what = "a rook"
    elif amount < 9:
        what = "a rook and more"
    else:
        what = "a queen or more"
    return f"you are {'up' if diff > 0 else 'down'} {what}"


def theme_of(tags: Sequence[MotifTag]) -> MotifTag | None:
    """The tag a coach would name as the idea, most specific first."""
    ranked = [tag for tag in tags if tag.name in _THEME_RANK]
    if not ranked:
        return None
    return min(ranked, key=lambda tag: _THEME_RANK[tag.name])


def region_of(squares: Sequence[str]) -> str | None:
    """A part of the board with one anchor: "the kingside, around f7"."""
    if not squares:
        return None
    files = [chess.square_file(chess.parse_square(square)) for square in squares]
    ranks = [chess.square_rank(chess.parse_square(square)) for square in squares]
    kingside = sum(1 for file in files if file >= 4)
    central = sum(
        1 for file, rank in zip(files, ranks, strict=True) if 2 <= file <= 5 and 2 <= rank <= 5
    )
    if central >= len(files) and len(files) > 1:
        where = "the centre"
    elif kingside * 2 >= len(files) + 1:
        where = "the kingside"
    elif kingside * 2 <= len(files) - 1:
        where = "the queenside"
    else:
        where = "the centre"
    anchor = squares[0]
    return f"{where}, around {anchor}"


def describe_move(board: chess.Board, move: chess.Move) -> str:
    """A move in words, so a model never has to guess which piece moved where.

    "the knight from g1 to f3", "the bishop takes the knight on c6, giving check",
    "castles kingside", "the pawn from e7 to e8, promoting to a queen".
    """
    if board.is_castling(move):
        side = "kingside" if chess.square_file(move.to_square) > 4 else "queenside"
        return "castles " + side
    piece = board.piece_at(move.from_square)
    name = PIECE_NAMES[piece.piece_type] if piece else "piece"
    frm, to = chess.square_name(move.from_square), chess.square_name(move.to_square)
    if board.is_capture(move):
        taken = board.piece_at(move.to_square)
        if board.is_en_passant(move):
            what = "the pawn en passant"
        else:
            what = f"the {PIECE_NAMES[taken.piece_type]}" if taken else "a piece"
        text = f"the {name} takes {what} on {to}"
    else:
        text = f"the {name} from {frm} to {to}"
    if move.promotion:
        text += f", promoting to a {PIECE_NAMES[move.promotion]}"
    after = board.copy(stack=False)
    after.push(move)
    if after.is_checkmate():
        text += ", checkmate"
    elif after.is_check():
        text += ", giving check"
    return text


def _to_move(uci: str) -> chess.Move | None:
    try:
        return chess.Move.from_uci(uci)
    except ValueError:
        return None


def facts_from_analysis(
    board: chess.Board,
    analysis: Analysis,
    *,
    last_move: str | None = None,
    in_book: bool = False,
) -> PositionFacts:
    """The fact bundle for `board`, which is the position `analysis` measured.

    `board` must be the position the side to move faces; the coach speaks to that side.
    """
    lines = analysis.lines
    best_uci = lines[0].moves[0] if lines and lines[0].moves else None
    best = _to_move(best_uci) if best_uci else None
    if best is not None and best not in board.legal_moves:
        best = None

    theme = theme_of(analysis.tags)
    region = region_of(theme.squares) if theme else None
    focus = list(theme.squares) if theme else []
    piece: str | None = None
    if best is not None:
        moving = board.piece_at(best.from_square)
        if moving is not None:
            piece = f"your {PIECE_NAMES[moving.piece_type]}"

    candidates: list[str] = []
    if lines:
        top = lines[0].expected_score
        for line in lines:
            if not line.moves or top - line.expected_score > CANDIDATE_MARGIN:
                continue
            move = _to_move(line.moves[0])
            if move is not None and move in board.legal_moves:
                candidates.append(board.san(move))

    threats: list[str] = []
    if analysis.threats and not board.is_check():
        passed = board.copy(stack=False)
        passed.push(chess.Move.null())
        for uci in analysis.threats:
            move = _to_move(uci)
            if move is not None and move in passed.legal_moves:
                threats.append(passed.san(move))

    line = san_line(board, lines[0].moves, limit=8).split() if lines else []
    line = [token for token in line if not token.endswith(".") and not token.endswith("...")]

    return PositionFacts(
        fen=board.fen(),
        side_to_move="white" if board.turn == chess.WHITE else "black",
        phase=phase_of(board, in_book),
        material=material_text(board),
        last_move=last_move,
        decisive=analysis.decisive,
        only_move=analysis.only_move,
        theme=dict(THEMES)[theme.name] if theme else None,
        region=region,
        piece=piece,
        focus=focus,
        candidates=candidates,
        threats=threats,
        line=line,
    )
