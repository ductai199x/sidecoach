"""From the analysis of a position to the facts a coach may draw on.

The disclosure machine filters these by level, so what goes in here decides what can ever
come out. Each field is written in the words the coach would use: the theme is a category
("a loose piece is involved"), the region is a part of the board with one anchor square,
the piece is named but never its destination, and the candidates and the line are moves in
the notation a person reads.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import chess

from sidecoach.analysis import exchange, positional
from sidecoach.analysis.engine import san_line
from sidecoach.analysis.phase import phase_of
from sidecoach.coach import idea as idea_of
from sidecoach.coach import idea_words, plans, situation, story
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
    "passerPawnEnding": "making a passed pawn",
}
# The best move's own idea that comes before the plan its line carries out: a king to get
# to safety, a piece to save, their threat to stop, a piece still at home. Anything else
# gives way to the plan of the player's next moves (`coach/plans.py`).
_BEFORE_THE_PLAN = frozenset({"castle", "rescue", "prophylaxis", "develop"})
# The best move's own ideas that belong beside a threat to meet: the rest (a rook to an open
# file, a knight to the centre) are what the move also does, and "stop it before it comes.
# The key is taking the open file" named the wrong reason (the master games, 2026-09-24).
_DEFENDING = frozenset({"castle", "rescue", "prophylaxis"})
# A threat to win less than this, in pawns, gives way to the move's own idea: "d4 goes after
# their pawn on e5" is the fight for the centre, not the initiative.
THREAT_MIN = 2
# In a pawn ending there is no king to storm and no space to take: a pawn move there is
# about the passed pawn it makes ("storming the pawns in front of their King" was said of
# h5 in a king-and-pawn ending, Lichess Yb1jJ).
_PAWN_ENDING_PLAN = {
    "break": "passerPawnEnding",
    "space": "passerPawnEnding",
    "pawnStorm": "passerPawnEnding",
    "centre": "passerPawnEnding",
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
    # The best move saves a piece that is attacked: the hint a stuck player needs most,
    # and the one the coach could not give while this was not a theme ("make your King
    # safer" with the rook en prise, and the rook was left there).
    ("defensiveMove", "a piece under attack"),
    # The tagger's exposed king is always the enemy's: said as "king safety" the coach
    # told a player with a won endgame to make their own king safer.
    ("exposedKing", "your opponent's exposed king"),
    ("attackingF2F7", "the weak point next to the king"),
    ("promotion", "a passed pawn"),
    # A pawn on the sixth or seventh rank, passed or not: "a passed pawn" was said of b6+
    # with their pawn on b7 in front (the test audit, 2026-09-26).
    ("advancedPawn", "a pawn deep in their half"),
    # "A piece attacked through another" was read as "attacks your knight through another
    # piece" of a direct attack; the x-ray is the attack running on past its target.
    ("xRayAttack", "an x-ray: the attack runs on through one piece to the one behind it"),
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
    # Won at 0.8, as the idea reader counts a won position (`idea.WON`): "better" of a
    # position four pawns up read wrong against ChessQA's bands (93% -> 96% at one line).
    if score >= 0.8:
        return f"{who} winning"
    if score >= 0.62:
        return f"{who} better"
    if score > 0.38:
        return "it is about equal"
    if score > 0.2:
        return f"{who} worse"
    return f"{who} losing"


def motif_words(name: str) -> str:
    """A motif tag's name in plain words: the theme table's phrase when it has one, else
    the camelCase name split into words ("discoveredAttack" -> "discovered attack")."""
    for tag, words in THEMES:
        if tag == name:
            return words
    return re.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", name).lower()


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
_NOT_A_THEME = frozenset({"interference", "clearance", "attackingF2F7", "quietMove"})


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

    def enemy_of(i: int, j: int) -> bool:
        return bool(pieces[i] and pieces[j]) and pieces[i].color != pieces[j].color  # type: ignore[union-attr]

    if name == "fork" and len(sqs) >= 3 and pieces[0] and all(pieces[1:]):
        # The forking piece must bear on every target, and every target must be the enemy's:
        # a fork read off a board that has moved on is otherwise built from the wrong pieces.
        forker_colour = pieces[0].color  # type: ignore[union-attr]
        takes = {
            t
            for t in sqs[1:]
            if pieces[sqs.index(t)] is not None
            and (
                pieces[sqs.index(t)].piece_type == chess.KING  # type: ignore[union-attr]
                or sqs[0] in exchange.capturers(board, forker_colour, t)
            )
        }
        if all(
            enemy_of(0, i) and sqs[i] in board.attacks(sqs[0]) and sqs[i] in takes
            for i in range(1, len(sqs))
        ):
            # The king first, then the most valuable, at most three: "the queen on f7 forks
            # the pawn on b3 and the king on h5 and the pawn on f6 and the pawn on h7" was
            # said of a queen check that picks up a pawn.
            def worth(i: int) -> int:
                kind = pieces[i].piece_type  # type: ignore[union-attr]
                return 100 if kind == chess.KING else _VALUES[kind]

            order = sorted(range(1, len(sqs)), key=lambda i: -worth(i))[:3]
            targets = [named(i) for i in order]
            listed = (
                targets[0] if len(targets) == 1 else ", ".join(targets[:-1]) + f" and {targets[-1]}"
            )
            return f"{named(0)} forks {listed}"
    elif name in ("hangingPiece", "trappedPiece") and pieces and pieces[0]:
        return f"{named(0)} is {'loose' if name == 'hangingPiece' else 'trapped'}"
    elif name == "discoveredAttack" and len(sqs) >= 3:
        # The tagger records the vacated square, the slider, then what it now attacks; the
        # slider's line is open only once the move has been played.
        if enemy_of(1, 2) and sqs[2] in board.attacks(sqs[1]):
            return f"{named(1)} now bears on {named(2)}"
    elif name == "capturingDefender" and len(sqs) >= 2 and enemy_of(0, 1):
        # The capture removes the guard of the square the mover uses later: worth a word
        # only while an enemy piece stands there to name ("removes a defender of g4", of
        # an empty square, said nothing the capture itself did not).
        return f"{named(0)} removes a defender of {named(1)}"
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


# What makes the other side's move after a pass a threat, as the sheet names its effects: it
# mates, takes, checks, or attacks something. A move that only defends, retreats or takes a
# file is what they would do with a free move, not a threat ("They threaten Rc3." of a rook
# guarding the third rank against a mate in three: dev audit, 2026-09-26).
_THREAT_EFFECTS = (
    "checkmate",
    "takes ",
    "gives check",
    "forks ",
    "attacks ",
    "uncovers an attack",
    "frees ",
    "pins ",
)


def _threatens(passed: chess.Board, move: chess.Move) -> bool:
    from sidecoach.coach.effects import describe_effects  # effects imports this module

    if move.promotion:
        return True
    effects = describe_effects(passed, move, limit=8, hazards=False)
    if any(e.startswith(_THREAT_EFFECTS) for e in effects):
        return True
    # A quiet move that threatens mate on an empty square attacks nothing yet.
    again = passed.copy(stack=False)
    again.push(move)
    if again.is_check() or again.is_game_over():
        return False
    again.push(chess.Move.null())
    for reply in list(again.legal_moves):
        again.push(reply)
        mate = again.is_checkmate()
        again.pop()
        if mate:
            return True
    return False


# A line whose play-out gains this much material (in pawns' worth) is a tactic even when
# its first move is quiet: a discovered attack, a knight move that forks next.
TACTIC_GAIN = 2
# A quiet first move of a tactic the idea reader finds, by its goal.
IN_CHECK_QUESTION = (
    "You are in check: can you take the checking piece, block it, or must your king move -- "
    "and which way out leaves you safest?"
)
IN_CHECK_NATURE = "a check to answer -- take the checking piece, block it, or move the king"

_QUIET_TACTIC = {
    "win": "a tactic -- the best move is quiet, but the line after it wins material",
    "combination": "a tactic -- the best move is quiet, but the line after it wins material",
    "mate": "a tactic -- the best move is quiet, but the line after it mates",
    "promote": "a tactic -- the best move is quiet, but the line after it queens a pawn",
    "attack": "a tactic -- the best move is quiet, but the attack after it wins",
}


def nature_of(board: chess.Board, best: chess.Move, line: Sequence[str] = ()) -> str:
    """Whether the answer is forcing, a quiet move that wins material, or quiet, in the
    words the sheet uses. `line` is the engine's line from `best` (UCI), played out to
    see whether it wins material: "quiet: there is no tactic" was said of a won endgame
    whose best move, a quiet knight move, uncovered a rook on the queen."""
    if board.is_capture(best) or board.gives_check(best) or best.promotion:
        return "forcing -- the answer is a check, a capture or a promotion"
    # Counted as the idea counts it, net of what they just took (`idea.exchange_owed`).
    owed = idea_of.worth(idea_of.exchange_owed(board))
    if line and _line_gain(board, line) - owed >= TACTIC_GAIN:
        return "a tactic -- the best move is quiet, but the line after it wins material"
    return "quiet -- no tactic to find; the move improves your position"


def _line_gain(board: chess.Board, line: Sequence[str]) -> int:
    """Material gained by the side to move over `line` (UCI), in pawns' worth, counted as
    the idea reader counts its line (`idea.line_gain`): after the opponent's replies, a
    line that stops mid-trade not counting what can still be taken back ("a3 Kd7 Nxe5" was
    a knight won, with dxe5 to come). The line stops at the first illegal move."""
    moves: list[chess.Move] = []
    boards = [board.copy(stack=False)]
    for uci in line:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            break
        if move not in boards[-1].legal_moves:
            break
        after = boards[-1].copy(stack=False)
        after.push(move)
        moves.append(move)
        boards.append(after)
    return idea_of.line_gain(boards, moves, len(moves))[0] if moves else 0


def facts_from_analysis(
    board: chess.Board,
    analysis: Analysis,
    *,
    last_move: str | None = None,
    in_book: bool = False,
    user: chess.Color | None = None,
    pick: Callable[[str, Sequence[str]], str] | None = None,
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

    # What the best line is for, read through the player's own moves (`coach/idea.py`): the
    # goal, the blow that delivers it on whichever move it comes, the target. The pattern is
    # the blow's, never a static feature elsewhere on the board: a static pin took the
    # headline from the fork on the player's second move, and a mate in one went out as
    # "getting a piece out of danger" (Tai's puzzles, 2026-09-23).
    idea = idea_of.read(board, analysis)
    tactical = idea.goal in ("mate", "win", "promote", "attack", "combination")
    blow = idea.blow if tactical else None
    theme_text = idea_words.pattern(idea) if tactical else None
    # The squares the pattern turns on, as they stand now: the blow's when it is the first
    # move, else what it goes after.
    if blow is not None and idea.blow_at == 1:
        focus = list(blow.squares)
    elif idea.target is not None and tactical:
        focus = [chess.square_name(idea.target)]
    else:
        focus = []
    region = region_of(focus) if focus else None
    tactic = (
        describe_tactic(idea.blow_board, blow.name, blow.squares, idea.blow_move)
        if blow is not None and idea.blow_board is not None
        else None
    )

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
                # Their best move if you passed stays the prophylactic question's ("if you let
                # them, their best is Kc2"); it is listed as a threat only when it is one.
                if _threatens(passed, move):
                    threats.append(passed.san(move))
                if threat_move is None:
                    threat_move = move

    # The idea half: the strategic point of the best move (a square-free theme), and the few
    # most instructive static features. The plan only competes for the theme slot when no
    # tactic leads, so the coach is never handed two headlines at once.
    plan: str | None = None
    plan_kind: str | None = None
    if best is not None and idea.goal in ("plan", "defend"):
        worst = positional.worst_piece_square(board, us)
        plan_idea = positional.best_move_idea(
            board, best, us, phase=phase, worst=worst, threat=threat_move
        )
        if plan_idea is not None:
            kind = plan_idea.kind
            pawn_ending = all(
                p.piece_type in (chess.KING, chess.PAWN) for p in board.piece_map().values()
            )
            if pawn_ending:
                kind = _PAWN_ENDING_PLAN.get(kind, kind)
            plan = _PLAN_THEME.get(kind)
            plan_kind = kind if plan else None
            if idea.goal == "defend" and plan_kind not in _DEFENDING:
                plan, plan_kind = None, None
    # The plan the best line carries out over the player's next moves: a king attack, rooks
    # doubled on a file, a knight's route to an outpost. What a grandmaster's note says of a
    # quiet position, where the best move alone reads as "a rook to an open file".
    # And the threat the best move makes, where the line wins nothing because they answer it:
    # a threat of mate comes before any plan, a threat to win material after the line's plan.
    line_plan: plans.LinePlan | None = None
    move_threat: plans.MoveThreat | None = None
    # A move out of check is not a plan or a threat: it is the way out (20.Ke2 was "you can
    # take the initiative").
    if (
        best is not None
        and idea.goal == "plan"
        and plan_kind not in _BEFORE_THE_PLAN
        and not board.is_check()
    ):
        pv = [m for m in (_to_move(u) for u in lines[0].moves) if m is not None]
        line_plan = plans.read(board, pv, lines[0].expected_score)
        move_threat = plans.threat(board, best, pv, lines[0].expected_score)
        if (
            move_threat is not None
            and move_threat.kind == "win"
            and move_threat.gain < THREAT_MIN
            and plan_kind is not None
        ):
            move_threat = None  # a pawn attacked is not the point when the move has its own
        if move_threat is not None and (move_threat.kind == "mate" or line_plan is None):
            line_plan = None
            plan, plan_kind = plans.THREAT_NAME[move_threat.kind], None
        elif line_plan is not None:
            move_threat = None
            plan, plan_kind = plans.name(line_plan), None
    ideas = [fact.text for fact in positional.top_features(board, us, phase=phase, limit=2)]

    piece: str | None = idea_words.piece(idea, board)
    nature: str | None = None
    if best is not None:
        nature = nature_of(board, best, lines[0].moves if lines else ())
        # The nature agrees with the idea: a tactic the idea reader finds -- a pawn won back,
        # a mate -- under TACTIC_GAIN was "quiet -- no tactic to find" beside "You have a
        # tactic that wins back a pawn" (the pre-spend audit, 2026-09-25).
        if tactical and nature.startswith("quiet"):
            nature = _QUIET_TACTIC.get(idea.goal, _QUIET_TACTIC["win"])
    mechanism = idea_words.mechanism(idea)
    if mechanism is None and plan_kind in idea_words.PLAN_WHY:
        mechanism = f"{plan}: {idea_words.PLAN_WHY[plan_kind]}"
    question = idea_words.question(idea, plan_kind, board)
    if board.is_check() and question in situation.ANY_POSITION:
        # In check the question is the check: "a quiet improvement -- which of your pieces has
        # no job right now?" was the hint in seven of ten AI sessions' check positions, and the
        # students listed it among what they did not understand (round 4, 2026-09-26).
        question = IN_CHECK_QUESTION
        if nature is not None and nature.startswith("quiet"):
            nature = IN_CHECK_NATURE
    target = idea_words.target(idea, board)
    nudge = idea_words.nudge(idea, board)
    why = idea_words.why(idea, board, describe_tactic)
    goal = idea_words.goal(idea)
    if move_threat is not None and best is not None:
        won = plans.THREAT_GOAL_WON if idea.winning else plans.THREAT_GOAL
        goal = won[move_threat.kind]
        mechanism = plans.threat_mechanism(move_threat)
        question = plans.THREAT_QUESTION[move_threat.kind]
        target = plans.threat_target(move_threat, board) or target
        nudge = plans.THREAT_NUDGE[move_threat.kind]
        why = plans.threat_why(move_threat, board, best)
    # A forcing capture with nothing else to point at: its focus is what it takes.
    if (
        target is None
        and best is not None
        and idea.goal == "plan"
        and move_threat is None
        and line_plan is None
        and board.is_capture(best)
    ):
        taken = board.piece_at(best.to_square)
        if taken is not None and taken.color != us:
            target = (
                f"their {chess.piece_name(taken.piece_type)} on {chess.square_name(best.to_square)}"
            )
    if line_plan is not None:
        mechanism = plans.mechanism(line_plan)
        question = plans.PLAN_QUESTION[line_plan.kind]
        target = plans.target(line_plan) or target
        piece = plans.pieces(line_plan, board) or piece
        nudge = plans.PLAN_NUDGE[line_plan.kind]
        why = plans.why(line_plan, board)

    # The first hint (`coach/situation.py`): the stakes, the key fact named on the board, the
    # principle behind it, the trap -- a short lesson on the position, never the move.
    stakes = situation.read_stakes(board, analysis)
    stakes_text = situation.stakes_words(board, stakes, analysis.opening) if stakes else None
    choices = list(stakes.choices) if stakes and stakes.kind in ("many", "book") else []
    pv = [m for m in (_to_move(u) for u in lines[0].moves) if m is not None] if lines else []
    key: str | None = None
    principle: str | None = None
    tactical_goal = idea.goal in (
        "mate",
        "win",
        "promote",
        "attack",
        "combination",
        "draw",
        "convert",
        "king",
        "breakthrough",
    )
    if idea.goal == "defend":
        key = situation.defend_key(board, idea, best)
    elif idea.goal in ("mate", "attack"):
        key = situation.mate_key(board, idea)
        principle = idea_words.lesson(idea_words.mechanism(idea), idea)
    elif tactical_goal:
        key = idea_words.target(idea, board)
        if key is not None and ":" not in key and " is " not in key and "," not in key:
            key = f"{key} is the target"  # "their rook on a8" alone said nothing
        elif key is not None and idea.goal == "promote" and ", " in key:
            key = key.replace(", ", " is ", 1)  # "your pawn on b6 is two squares from queening"
        if key is None and idea.goal == "win":
            key = situation.win_key(board, idea)
            if key is None and idea_of.ending_of(board):
                key = situation.king_key(board, idea)  # the pawn falls to a king's march
        elif key is None and idea.goal == "convert":
            key = situation.convert_key(board, idea)
        elif key is None and idea.goal == "king":
            key = situation.king_key(board, idea)
        # The principle is the pattern's: a sacrifice on the way, with no pattern, is not
        # what a winning capture teaches (17.hxg4 was "a sacrifice").
        principle = (
            idea_words.lesson(idea_words.mechanism(idea), idea)
            if idea.blow is not None
            or idea.goal
            in (
                "mate",
                "combination",
                "attack",
                "promote",
            )
            else None
        )
    elif move_threat is not None:
        key = situation.threat_key(board, move_threat)
        principle = (
            plans.TRADE_REASON[move_threat.why]
            if move_threat.kind == "trade"
            else plans.THREAT_REASON[move_threat.kind]
        )
    elif line_plan is not None:
        key = situation.plan_key(board, line_plan)
        principle = plans.PLAN_REASON[line_plan.kind]
    elif plan_kind is not None and best is not None:
        key = situation.positional_key(board, best, plan_kind, threat_move, phase)
        # The second hint's question follows from the fact the first named: without it, "which
        # pawn move lets it out?" asked of a position where nothing is shut in.
        if nudge is None and key is not None:
            nudge = situation.positional_nudge(board, plan_kind, key)
        elif plan_kind in ("centralise", "activate") and key is None and why is None:
            why = situation.reach_why(board, best)
        elif plan_kind == "centre" and key is None:
            said = situation.centre_move(board, best)
            if said is not None:
                nudge = nudge or said[0]
                why = why or said[1]
    # One story (`coach/story.py`): a piece that simply hangs is said as that, before any
    # pattern the line also shows.
    plain = story.plain_capture(board, idea) if tactical_goal else None
    if plain is not None and plain.whole:
        key, principle, question = plain.key, None, None
        nudge, why = plain.nudge, plain.why
    elif plain is not None:
        # The free piece first, then what the line wins after it.
        named = f"on {chess.square_name(plain.square)}"
        rest = f". After that, {key[0].lower() + key[1:]}" if key and named not in key else ""
        key = plain.key + rest
    # The key the move itself answers, and so its reason at the move rung; the fallbacks
    # below (a static feature, their idea) are context, not the move's reason.
    move_reason = key
    in_check = board.is_check()
    book = stakes is not None and stakes.kind == "book"
    if (
        key is None
        and best is not None
        and not tactical_goal
        and idea.goal != "defend"
        and not in_check
        and not book
    ):
        key = situation.passer_key(board, best) or situation.kings_key(board, best)
        if key is None:
            features = positional.top_features(board, us, phase=phase, limit=4)
            key = situation.relevant_feature(board, features, pv)
    if in_check and key is None:
        key = situation.check_key(board)
    if key is None and not tactical_goal and not book:
        # Nothing else named: what they would do if you let them -- the prophylactic question,
        # answered on the board.
        sharp = stakes is not None and stakes.kind in ("only", "few", "lost")
        key = situation.their_idea_key(board, threat_move, sharp=sharp)
    if in_check:
        nudge = "Which way out keeps you safest -- a king move, a block, or taking the checker?"
        if why is None and best is not None:
            why = situation.evasion_why(board, best)
    # A recapture: the best move takes back what the line wins, at once (8.bxc3). Beside it,
    # only the wrong piece to take back with is a trap -- every other capture is worse only for
    # not taking back -- and 23.exd5? Re8 in Polugaevsky-Nezhmetdinov is the lesson.
    recapture = (
        idea.goal == "win"
        and (idea.down or idea.back)
        and best is not None
        and board.is_capture(best)
        and best.to_square == idea.target
    )
    traps = [
        t
        for t in analysis.traps
        if not recapture or (best is not None and t.moves and t.moves[0][2:4] == best.uci()[2:4])
    ]
    trap = None
    if idea.goal != "mate":
        # The trap that matters most and can be said: on the target, into mate, the worst.
        for chosen in situation.ranked_traps(traps, idea.target):
            trap = situation.trap_words(
                board, chosen, idea.target, lines[0].expected_score if lines else None, best
            )
            if trap is not None:
                break
    if trap is not None and idea.target is not None and situation.NOT_YET in trap and plain is None:
        key = None  # the trap says it: the target can be won, but not at once
    parts: list[str] = []
    forced = board.is_check() and board.legal_moves.count() == 1
    calm = stakes is not None and stakes.kind in ("many", "open")
    if stakes_text and idea.goal == "defend" and calm:
        stakes_text = None  # "nothing is urgent" beside their threat said the opposite
    if stakes_text and (not tactical_goal or (stakes and stakes.kind in ("check", "book"))):
        parts.append(stakes_text)
    # Where only one or a few moves hold, the move's reason is what it does on the board --
    # takes, checks, attacks, stops their idea -- never "The move is Bb5+." alone.
    sharp = (
        stakes is not None
        and stakes_text is not None
        and (
            stakes.kind in ("only", "few")
            or (stakes.kind in ("won", "check") and "only" in stakes_text)
        )
    )
    if sharp and stakes is not None and best is not None and (why is None or "goes from" in why):
        edge = situation.edge_words(stakes.top)
        holds = (
            f"it is one of the few moves that keep {edge}"
            if stakes.kind == "few" and stakes.good > 1
            else f"it is the only move that keeps {edge}"
        )
        why, ask = story.reason(board, best, threat_move, holds)
        if why == f"{board.san(best)}: {holds}":
            why = story.line_reason(board, pv, holds) or why
        if nudge is None:
            nudge = ask
    # In check, the piece giving check comes first, unless the capture that wins takes it.
    checker = (
        situation.check_key(board, idea.target if tactical_goal else None)
        if in_check and not forced and not (plain is not None and plain.square in board.checkers())
        else None
    )
    pawn = (
        story.queening(board, idea)
        if tactical_goal and idea.goal != "promote" and not forced
        else None
    )
    if pawn is not None and key is not None and "queening" in key:
        pawn = None
    said = " ".join(part for part in (checker, key, trap) if part)
    prize = (
        story.prize(board, idea, said)
        if tactical_goal and not forced and (plain is None or not plain.whole)
        else None
    )
    # After a free piece taken first, the prize comes where the line wins it: last.
    prize_last = plain is not None
    # How a win with no named pattern comes about: what the first move does, and at the move
    # rung the line to it, never the result alone.
    told_how = (
        story.how(board, idea, threat_move, analysis.threatens[0] if analysis.threatens else None)
        if tactical_goal and not forced and plain is None
        else None
    )
    how = told_how.said if told_how is not None else None
    if told_how is not None and told_how.lesson:
        # The pattern the move carries, taught after it: "That's a trapped piece: ..."
        taught = idea_words.lesson(told_how.lesson, idea) or told_how.lesson
        how = f"{how}. {taught[0].upper()}{taught[1:]}"
        nudge = told_how.ask or nudge
    elif told_how is not None and told_how.ask and not in_check:
        # The second hint asks after what the first said the move is for: "which quiet move
        # sets up a mate?" after "the first move is quiet, but it threatens mate".
        nudge = told_how.ask
    if (
        tactical_goal
        and best is not None
        and idea.blow is None
        and (why is None or re.fullmatch(rf"{re.escape(board.san(best))}: you win [^;]*", why))
    ):
        why = story.move_why(board, idea, threat_move) or why
    # Why the obvious defence fails (`story.defence`): the recapture the line does not play,
    # and the other job of the piece that would make it -- the overloaded defender, said in
    # place of "a loose piece" when the loose piece was only half the story (Tai's chess.com
    # puzzles, 2026-09-24: "you can win their queen" read as a lie beside fxg3, and a rook on
    # f8 that had to stop f8=Q was "not defended at all").
    held = (
        story.defence(board, analysis.defences, lines[0].moves if lines else [])
        if tactical_goal and not forced and analysis.defences and idea.goal != "mate"
        else None
    )
    defended: str | None = None
    if held is not None:
        loose = idea.blow is not None and idea.blow.name == "hangingPiece"
        if key is None or (loose and (held.square in key or held.target in key)):
            # The loose piece was half the story: the defender is there, with another job.
            key, principle, question = held.said, idea_words.lesson(held.lesson, idea), held.ask
            if why is not None:
                why = re.sub(r"the \w+ on [a-h][1-8] is loose", held.clause, why)
        else:
            defended = held.said
            principle = principle or idea_words.lesson(held.lesson, idea)
        nudge = held.ask
        if why is not None:
            why = f"{why}; {held.line}"
    if forced:
        # One legal move: the rest of the reading is the move's, not a hint.
        tactical_goal, key, principle, trap = False, None, None, None
        move_threat = None
        nudge = None
    if checker is not None and not forced and checker != key:
        parts.append(checker)
    lost = stakes is not None and stakes.kind == "lost"
    # A line that wins by force -- not one that starts by taking a piece simply left hanging
    # or taking back (`story.plain_capture`) -- is a tactic, and the first hint says there is
    # one to find (Tai, 2026-09-25).
    is_tactic = (
        idea.goal in ("win", "combination")
        or (idea.goal == "promote" and (idea.blow is not None or bool(idea.sacrifices)))
    ) and plain is None
    if (
        tactical_goal
        or (move_threat is not None and not lost)
        or (idea.winning and not stakes_text)
    ):
        lead = goal
        if idea.goal == "plan" and idea.winning and move_threat is None:
            lead = "you are winning"
        elif move_threat is None:
            seed = idea_words.opener_seed(board)
            lead = idea_words.opener(idea, goal, tactic=is_tactic, seed=seed, pick=pick)
        if lead:
            parts.append(lead)
    # Facts about the same piece or square said as one sentence, the way a coach links them.
    shown_key = key
    if held is not None and not forced:
        held_as_key = key == held.said
        stops = held.stops

        def stops_it(text: str | None) -> bool:
            # The pawn the defender has to stop, named in `text`.
            return (
                text is not None
                and stops is not None
                and f"pawn on {stops} " in text
                and "queening" in text
            )

        also = f"{held.who} has to stop it, so it cannot also take back on {held.target}"
        if stops_it(pawn):
            # "Your pawn on e6 is two squares from queening, and their rook on f8 has to stop
            # it, so it can't also take back on b8."
            pawn = f"{pawn}, and {also}"
            shown_key = None if held_as_key else key
            defended = None
        elif stops_it(key) and not held_as_key:
            shown_key = f"{key}, and {also}"
            defended = None
        elif (
            defended is not None
            and key is not None
            and key.endswith(f"on {held.target} is the target")
        ):
            # "Their queen on g3 is the target, and their pawn on f2 can't take back there:
            # it also shields their king."
            shown_key = f"{key}, and {held.who} cannot take back there: it also {held.duty}"
            defended = None
    target_said = (
        re.fullmatch(r"[Tt]heir (\w+) on [a-h][1-8]", shown_key[: -len(" is the target")])
        if shown_key and shown_key.endswith(" is the target")
        else None
    )
    if how and shown_key and target_said is not None:
        their = "t" + shown_key[1 : -len(" is the target")]  # "their rook on a1"
        if how.startswith("The first move "):
            # "Their queen on d3 is the target, and the first move attacks it."
            rest = how[len("The first move ") :]
            rest = re.sub(rf"^(\w+) {re.escape(their)}\b", r"\1 it", rest)
            if "," in rest.split(". ", 1)[0]:
                how = f"The first move {rest}"  # a clause of its own: said apart
            else:
                shown_key = f"{shown_key}, and the first move {rest}"
                how = None
        if how is not None and their in how:
            # Named once: "... a check that also attacks the rook, which has no safe square".
            how = how.replace(their, f"the {target_said.group(1)}")
    if pawn:
        parts.append(pawn)
    if prize and not prize_last:
        parts.append(prize)
    if shown_key:
        parts.append(shown_key)
    if defended and not forced:
        parts.append(defended)
    if how:
        parts.append(how)
    if prize and prize_last:
        parts.append(prize)
    if principle and key:
        parts.append(principle)
    if trap:
        # The target named already (the piece, not only its square): the order is the lesson.
        prey = board.piece_at(idea.target) if idea.target is not None else None
        if prey is not None and idea.target is not None:
            named = f"{idea_words.PIECE_NAMES[prey.piece_type]} on {chess.square_name(idea.target)}"
            if named in " ".join(parts):
                trap = situation.in_order(trap)
        if not parts:
            trap = situation.first_said(trap)
        parts.append(trap)
    if not parts and not forced and best is not None:
        # Nothing else to say: what they would do with a free move, or the piece doing least --
        # never a question that fits any position ("which of your pieces has no job right
        # now?" at 32...Kh8 in Carlsen-Kramnik).
        fallback = situation.their_idea_key(board, threat_move, sharp=True)
        if fallback is None:
            passive = positional.worst_piece(board, us)
            fallback = passive[0].text if passive else None
        if fallback:
            parts.append(fallback)
    situation_text = " ".join(situation.sentence(part) for part in parts) or None
    # A tactic's first hint ends on the question that sets the player looking the right way
    # ("which check leaves their king no square at all?"); never one that fits any position.
    if (
        situation_text
        and tactical_goal
        and question
        and not forced
        and question not in situation.ANY_POSITION
    ):
        situation_text += " " + question
    if situation_text:
        situation_text = situation.spoken(situation.within_budget(situation_text))
    # A quiet move's own reason, said at the move rung: the key fact it answers.
    if why is None and move_reason and best is not None:
        why = f"{board.san(best)}: {move_reason[0].lower() + move_reason[1:]}"

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
        goal=goal,
        theme=theme_text,
        mechanism=mechanism,
        question=question,
        region=region,
        target=target,
        piece=piece,
        why=why,
        nudge=nudge,
        focus=focus,
        tactic=tactic,
        plan=plan,
        ideas=ideas,
        candidates=candidates,
        threats=threats,
        line=line,
        stakes=stakes_text,
        key=key,
        trap=trap,
        choices=choices,
        situation=situation_text,
    )
