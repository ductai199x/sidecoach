"""From the analysis of a position to the facts a coach may draw on.

The disclosure machine filters these by level, so what goes in here decides what can ever
come out. Each field is written in the words the coach would use: the theme is a category
("a loose piece is involved"), the region is a part of the board with one anchor square,
the piece is named but never its destination, and the candidates and the line are moves in
the notation a person reads.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import chess

from sidecoach.analysis import positional
from sidecoach.analysis.engine import san_line
from sidecoach.coach.disclosure import PositionFacts
from sidecoach.events import Analysis, MotifTag

# The best move's idea, as a square-free strategic theme the coach may teach at the theme
# level: it orients toward the plan without naming the move. Keyed by the idea's kind.
_PLAN_THEME: dict[str, str] = {
    "castle": "getting your king to safety",
    "rescue": "getting a piece out of danger",
    "prophylaxis": "stopping your opponent's plan before your own",
    "activateKing": "activating your King in the endgame",
    "develop": "finishing your development",
    "outpostMove": "planting a knight on an outpost",
    "centralise": "centralising a piece",
    "takeFile": "taking the open file",
    "rookBehindPasser": "getting your rook behind the passed pawn",
    "break": "breaking open their pawn structure",
    "centre": "fighting for the centre",
    "pawnStorm": "storming the pawns in front of their King",
    "pushPasser": "pushing your passed pawn",
    "activate": "activating your worst-placed piece",
    "space": "gaining space on the flank",
    "simplify": "trading down to convert your material",
}

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


def standing_words(score: float, *, third_person: bool = False) -> str:
    """An expected score from the player's side, in words: as the coach says it to the
    player ("you are better"), or as the harness writes it ("the player is better")."""
    who = "the player is" if third_person else "you are"
    if score >= 0.85:
        return f"{who} winning"
    if score >= 0.62:
        return f"{who} better"
    if score > 0.38:
        return "it is about equal"
    if score > 0.15:
        return f"{who} worse"
    return f"{who} losing"


def motif_words(name: str) -> str:
    """A motif tag's name in plain words: the theme table's phrase when it has one, else
    the camelCase name split into words ("discoveredAttack" -> "discovered attack")."""
    for tag, words in THEMES:
        if tag == name:
            return words
    return re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", name).lower()


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


_COUNTED = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN)
_PLURALS = {
    "queen": "queens",
    "rook": "rooks",
    "bishop": "bishops",
    "knight": "knights",
    "pawn": "pawns",
}
_NUMBERS = {1: "a", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}


def _pieces_words(counts: dict[int, int], *, article: bool = True) -> str:
    """The surplus pieces of one side, most valuable first: "a rook and two pawns"; without
    the article after a possessive ("your knight")."""
    parts = []
    for piece_type in _COUNTED:
        n = counts.get(piece_type, 0)
        if n <= 0:
            continue
        name = PIECE_NAMES[piece_type]
        if n == 1:
            parts.append(f"a {name}" if article else name)
        else:
            parts.append(f"{_NUMBERS.get(n, str(n))} {_PLURALS[name]}")
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + f" and {parts[-1]}"


def material_text(board: chess.Board, perspective: chess.Color | None = None) -> str:
    """The material balance as `perspective` (the user; by default the side to move) would
    say it, by the pieces: "you are up a bishop for a pawn", "you are down the exchange".
    A sum in pawns' worth ("down two pawns" for a bishop against a pawn) read as a pawn
    count. The first live games' coach told a player a queen up that they were a queen
    down: the balance had been read from the opponent's side whenever the opponent was
    to move, and the memory carried it into the post-game verdict."""
    who = board.turn if perspective is None else perspective
    surplus = {t: len(board.pieces(t, who)) - len(board.pieces(t, not who)) for t in _COUNTED}
    ours = {t: n for t, n in surplus.items() if n > 0}
    theirs = {t: -n for t, n in surplus.items() if n < 0}
    if not ours and not theirs:
        return "level"
    diff = sum(_VALUES[t] * n for t, n in surplus.items())
    if ours and theirs:
        more, less = (ours, theirs) if diff >= 0 else (theirs, ours)
        # A rook against a bishop or knight, and nothing else: the exchange.
        exchange = more == {chess.ROOK: 1} and sum(less.values()) == 1 and chess.PAWN not in less
        what = "the exchange" if exchange else f"{_pieces_words(more)} for {_pieces_words(less)}"
        if diff == 0:
            return (
                f"level in value: your {_pieces_words(ours, article=False)} against their "
                f"{_pieces_words(theirs, article=False)}"
            )
        return f"you are {'up' if diff > 0 else 'down'} {what}"
    return f"you are {'up' if ours else 'down'} {_pieces_words(ours or theirs)}"


# Tags that never headline: they hold on most quiet moves (a line that could be cut, a
# square that could be cleared, a bishop eyeing f7 in a book Ruy Lopez), and as the theme
# they sent the stuck hint asking "which line could be cut?" in positions about nothing.
_NOT_A_THEME = frozenset(
    {"interference", "clearance", "attackingF2F7", "quietMove", "defensiveMove"}
)


def theme_of(tags: Sequence[MotifTag]) -> MotifTag | None:
    """The tag a coach would name as the idea, most specific first."""
    ranked = [tag for tag in tags if tag.name in _THEME_RANK and tag.name not in _NOT_A_THEME]
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
    # The anchor lies in the region: "the queenside, around g4" was the first focus square
    # named for a region the other two squares had chosen.
    in_region = {
        "the kingside": lambda f: f >= 4,
        "the queenside": lambda f: f < 4,
        "the centre": lambda f: 2 <= f <= 5,
    }[where]
    anchor = next((sq for sq, f in zip(squares, files, strict=True) if in_region(f)), squares[0])
    return f"{where}, around {anchor}"


def describe_tactic(
    board: chess.Board, name: str, squares: Sequence[str], best_move: chess.Move | None = None
) -> str | None:
    """A theme's focus squares as a true sentence, or None when the board does not bear it.

    Some themes are static on the board in front of the player (a pin, a loose piece);
    others are made by the best move (a fork the move creates). So the description is tried
    on the current board first, then on the board after `best_move`, and only returned when
    every square it names holds the piece it claims.
    """
    frames = [board]
    if best_move is not None and best_move in board.legal_moves:
        after = board.copy(stack=False)
        after.push(best_move)
        frames.append(after)
    for frame in frames:
        text = _tactic_on(frame, name, squares)
        if text is not None:
            return text
    return None


def _tactic_on(board: chess.Board, name: str, squares: Sequence[str]) -> str | None:
    sqs = [chess.parse_square(s) for s in squares]
    pieces = [board.piece_at(sq) for sq in sqs]

    def named(i: int) -> str:
        piece = pieces[i]
        assert piece is not None
        return f"the {PIECE_NAMES[piece.piece_type]} on {squares[i]}"

    # A pin or skewer aligns three pieces: an enemy attacker, then two of the mover's own.
    # The tagger already checked the geometry; here we only confirm the colours still hold,
    # so a description is never built from pieces that have changed hands.
    aligned = (
        len(sqs) >= 3
        and all(pieces[:3])
        and pieces[0].color != pieces[1].color  # type: ignore[union-attr]
        and pieces[1].color == pieces[2].color  # type: ignore[union-attr]
    )
    if name in ("pin", "absolutePin") and aligned:
        return f"{named(1)} is pinned to {named(2)} by {named(0)}"
    elif name == "skewer" and aligned:
        return f"{named(0)} skewers {named(1)} and {named(2)} behind it"
    elif name == "fork" and len(sqs) >= 3 and pieces[0] and all(pieces[1:]):
        targets = " and ".join(named(i) for i in range(1, len(sqs)))
        return f"{named(0)} forks {targets}"
    elif name in ("hangingPiece", "trappedPiece") and pieces and pieces[0]:
        return f"{named(0)} is {'loose' if name == 'hangingPiece' else 'trapped'}"
    elif name in ("discoveredAttack", "capturingDefender") and len(sqs) >= 2 and all(pieces[:2]):
        return f"{named(0)} bears on {named(1)}"
    return None


def position_text(board: chess.Board, *, legal: bool = True) -> str:
    """The position as a model reads it best: the FEN, every piece by colour and type with
    its squares, and the legal moves for the side to move. On ChessQA a piece list beat the
    FEN alone and the ASCII diagram lost to both on every board-reading tier; the legal-move
    list is what the paper's own prompt carries."""
    names = {
        chess.KING: "K",
        chess.QUEEN: "Q",
        chess.ROOK: "R",
        chess.BISHOP: "B",
        chess.KNIGHT: "N",
        chess.PAWN: "P",
    }
    sides: list[str] = []
    for colour, word in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        parts = []
        for piece_type in (
            chess.KING,
            chess.QUEEN,
            chess.ROOK,
            chess.BISHOP,
            chess.KNIGHT,
            chess.PAWN,
        ):
            squares = sorted(chess.square_name(s) for s in board.pieces(piece_type, colour))
            if squares:
                parts.append(f"{names[piece_type]} {' '.join(squares)}")
        sides.append(f"{word}: " + ", ".join(parts))
    lines = [f"FEN: {board.fen()}", "Pieces -- " + " | ".join(sides)]
    if legal and not board.is_game_over():
        lines.append("Legal moves: " + " ".join(sorted(board.san(m) for m in board.legal_moves)))
    return "\n".join(lines)


def ascii_board(board: chess.Board) -> str:
    """The position as a labelled grid, which a language model reads far better than a FEN.

    White is drawn at the bottom in the usual way, uppercase for white and lowercase for
    black, a dot for an empty square, with rank numbers down the side and file letters
    below, so a square named in the text matches a square in the diagram. This shows only
    what the FEN already shows; it is a clearer rendering, not new information.
    """
    rows = []
    for rank in range(7, -1, -1):
        cells = []
        for file in range(8):
            piece = board.piece_at(chess.square(file, rank))
            cells.append(piece.symbol() if piece else ".")
        rows.append(f"{rank + 1} " + " ".join(cells))
    rows.append("  a b c d e f g h")
    return "\n".join(rows)


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


# A line whose play-out gains this much material (in pawns' worth) is a tactic even when
# its first move is quiet: a discovered attack, a knight move that forks next.
TACTIC_GAIN = 2


def nature_of(board: chess.Board, best: chess.Move, line: Sequence[str] = ()) -> str:
    """Whether the answer is forcing, a quiet move that wins material, or quiet, in the
    words the sheet uses. `line` is the engine's line from `best` (UCI), played out to
    see whether it wins material: "quiet: there is no tactic" was said of a won endgame
    whose best move, a quiet knight move, uncovered a rook on the queen."""
    if board.is_capture(best) or board.gives_check(best) or best.promotion:
        return "forcing -- the answer is a check, a capture or a promotion"
    if line and _line_gain(board, line) >= TACTIC_GAIN:
        return "a tactic -- the best move is quiet, but the line after it wins material"
    return "quiet -- no tactic to find; the move improves your position"


def _line_gain(board: chess.Board, line: Sequence[str]) -> int:
    """Material gained by the side to move over `line` (UCI), in pawns' worth; the line
    stops at the first illegal move."""
    us = board.turn
    after = board.copy(stack=False)
    for uci in line:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in after.legal_moves:
            break
        after.push(move)

    def balance(b: chess.Board) -> int:
        return sum(
            _VALUES.get(p.piece_type, 0) * (1 if p.color == us else -1)
            for p in b.piece_map().values()
        )

    return balance(after) - balance(board)


def facts_from_analysis(
    board: chess.Board,
    analysis: Analysis,
    *,
    last_move: str | None = None,
    in_book: bool = False,
    user: chess.Color | None = None,
) -> PositionFacts:
    """The fact bundle for `board`, which is the position `analysis` measured.

    `board` must be the position the side to move faces; the candidates, threats and
    plan are the side to move's. `user` is the side the coach speaks to -- the material
    is read from that side -- and by default the side to move.
    """
    lines = analysis.lines
    best_uci = lines[0].moves[0] if lines and lines[0].moves else None
    best = _to_move(best_uci) if best_uci else None
    if best is not None and best not in board.legal_moves:
        best = None
    phase = phase_of(board, in_book)
    us = board.turn

    # A tactic is the headline only when it is the point: a decisive or forced position, or
    # one where the best move plays into the very squares the tactic is on. A pin that happens
    # to sit on the board while the plan is elsewhere is incidental, and teaching it would
    # point the player away from the real idea, so its focus and description are withheld.
    theme = theme_of(analysis.tags)
    best_squares = {best.from_square, best.to_square} if best is not None else set()
    tactic_squares = {chess.parse_square(s) for s in theme.squares} if theme else set()
    tactic_relevant = theme is not None and (
        analysis.decisive or analysis.only_move or bool(best_squares & tactic_squares)
    )
    region = region_of(theme.squares) if tactic_relevant and theme else None
    focus = list(theme.squares) if tactic_relevant and theme else []
    tactic = (
        describe_tactic(board, theme.name, theme.squares, best)
        if tactic_relevant and theme
        else None
    )
    theme_text = dict(THEMES)[theme.name] if tactic_relevant and theme else None

    # The opponent's threat, as the coach would name it, and the move behind it: the engine's
    # best reply after a pass. The threat move lets the best move be read as prophylaxis.
    threats: list[str] = []
    threat_move: chess.Move | None = None
    if analysis.threats and not board.is_check():
        passed = board.copy(stack=False)
        passed.push(chess.Move.null())
        for uci in analysis.threats:
            move = _to_move(uci)
            if move is not None and move in passed.legal_moves:
                threats.append(passed.san(move))
                if threat_move is None:
                    threat_move = move

    # The idea half: the strategic point of the best move (a square-free theme), and the few
    # most instructive static features. The plan only competes for the theme slot when no
    # tactic leads, so the coach is never handed two headlines at once.
    plan: str | None = None
    if best is not None:
        worst = positional.worst_piece_square(board, us)
        idea = positional.best_move_idea(
            board, best, us, phase=phase, worst=worst, threat=threat_move
        )
        if idea is not None and not tactic_relevant:
            plan = _PLAN_THEME.get(idea.kind)
    ideas = [fact.text for fact in positional.top_features(board, us, phase=phase, limit=2)]

    piece: str | None = None
    nature: str | None = None
    if best is not None:
        moving = board.piece_at(best.from_square)
        if moving is not None:
            piece = f"your {PIECE_NAMES[moving.piece_type]}"
        nature = nature_of(board, best, lines[0].moves if lines else ())

    candidates: list[str] = []
    if lines:
        top = lines[0].expected_score
        for line in lines:
            if not line.moves or top - line.expected_score > CANDIDATE_MARGIN:
                continue
            move = _to_move(line.moves[0])
            if move is not None and move in board.legal_moves:
                candidates.append(board.san(move))

    line = san_line(board, lines[0].moves, limit=8).split() if lines else []
    line = [token for token in line if not token.endswith(".") and not token.endswith("...")]

    return PositionFacts(
        fen=board.fen(),
        side_to_move="white" if board.turn == chess.WHITE else "black",
        user_side=None if user is None else ("white" if user == chess.WHITE else "black"),
        phase=phase,
        material=material_text(board, user),
        last_move=last_move,
        decisive=analysis.decisive,
        only_move=analysis.only_move,
        nature=nature,
        theme=theme_text,
        region=region,
        piece=piece,
        focus=focus,
        tactic=tactic,
        plan=plan,
        ideas=ideas,
        candidates=candidates,
        threats=threats,
        line=line,
    )
