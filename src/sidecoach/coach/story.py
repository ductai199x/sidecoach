"""The story of the best line, told in the order a player meets it.

Tai, 2026-09-24, after the puzzles were read by hand: the first hint's parts -- what the line
wins, which tactic to name, which piece is the target, which plan -- were each chosen on their
own, and nothing made them agree with the move to find. "You can win their queen. Their rook
on f5, on a line one of your pieces is blocking" when a pawn simply takes the queen; "their
bishop on f4 is the target" to a player in check from a queen that can be taken. The hints are
told here as one story, the most immediate beat first:

- in check, the piece giving check;
- a piece that simply hangs -- the capture on its own nets what the whole line does -- said as
  that, with no pattern beside it;
- the pawn about to queen, where the win rests on it;
- the prize, where it stands, when the pattern that wins it names other pieces;
- where only one or a few moves hold, what the move does: takes, checks, attacks, stops their
  idea -- never "The move is Bb5+." alone.

`tools/puzzles/story_checks.py` holds the hints to each beat over the Lichess puzzles and the
master games, read from the board and the engine's line, never from this code.
"""

from __future__ import annotations

from dataclasses import dataclass

import chess

from sidecoach.analysis import exchange, positional
from sidecoach.coach import idea as idea_of
from sidecoach.coach.idea import Idea

_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
_TIMES = {1: "once", 2: "twice", 3: "three times", 4: "four times"}
# A free first capture worth this much (a minor piece) is the story's first beat even when the
# line wins more after it.
FREE_MIN = 3
_NUMBER = {1: "one square", 2: "two squares", 3: "three squares", 4: "four squares"}


def _sq(square: chess.Square) -> str:
    return chess.square_name(square)


def _named(board: chess.Board, square: chess.Square) -> str:
    piece = board.piece_at(square)
    return f"{_NAMES[piece.piece_type]} on {_sq(square)}" if piece else _sq(square)


@dataclass(frozen=True)
class Plain:
    """A piece that simply hangs, as the three rungs say it."""

    square: chess.Square
    whole: bool  # the capture is the whole win; else the line wins more after it
    key: str  # the first hint: "Their queen on g4 can simply be taken: nothing defends it"
    nudge: str  # the second hint's question
    why: str  # the third: "fxg4 takes their queen: nothing defends it"


def plain_capture(board: chess.Board, idea: Idea) -> Plain | None:
    """The first move takes a piece that simply hangs: the exchange on its square nets, on its
    own, what the whole line wins (gNzJg: fxg4 and the queen is gone, no discovered attack
    about it) -- or nets the whole piece, a minor one or more, and the line wins more after it
    (DWvS7: Qxe8+ takes a free rook, and then the knight on e4 is loose)."""
    first = idea.first
    if idea.goal != "win" or first is None or first not in board.legal_moves:
        return None
    if not board.is_capture(first) or board.is_en_passant(first):
        return None
    net = exchange.exchange(board, first)
    square = first.to_square
    victim = board.piece_at(square)
    if victim is None:
        return None
    whole = net >= max(idea.gain, 1)
    free = net >= exchange.VALUE.get(victim.piece_type, 0) >= FREE_MIN
    if not (whole or free):
        return None
    us = board.turn
    guards = exchange.defenders(board, victim.color, square)
    hitters = exchange.capturers(board, us, square)
    cheapest = min(
        (exchange.VALUE.get(board.piece_type_at(s) or chess.PAWN, 0) for s in hitters),
        default=0,
    )
    worth = exchange.VALUE.get(victim.piece_type, 0)
    if not guards:
        reason = "nothing defends it"
    elif cheapest < worth:
        reason = "it is worth more than what can take it"
    elif len(hitters) > len(guards):
        reason = (
            f"it is attacked {_TIMES.get(len(hitters), str(len(hitters)))} and defended "
            f"{_TIMES.get(len(guards), str(len(guards)))}"
        )
    else:
        reason = "the trades on its square come out in your favour"
    name = _NAMES[victim.piece_type]
    checker = board.is_check() and square in board.checkers()
    # "Back" only for what their last move took with: never guessed from the material count
    # ("their rook on h7 can simply be taken back" of a rook that had only moved there).
    back = " back" if _took(board, square) else ""
    if checker:
        key = f"Their {name} on {_sq(square)} gives check, and it can be taken{back}: {reason}"
    else:
        key = f"Their {name} on {_sq(square)} can simply be taken{back}: {reason}"
    # They take something back later in the line: what is left is the win the goal names
    # (Nxg5 takes the queen, and after Bxe2 Nxe2 a bishop is what remains).
    if net > idea.gain + 1 and idea.won and " for " not in idea.won:
        key += f"; after what they take back, you are {idea.won} up"
    return Plain(
        square=square,
        whole=whole,
        key=key,
        nudge="Before you take, look at what can take back: is it safe?",
        why=f"{board.san(first)} takes their {name}: {reason}{_forcing_reply(board, idea)}",
    )


def took_last(board: chess.Board) -> int:
    """What their last move took, in pawns (0 when it took nothing or the board does not know
    its moves): what the player wins next, up to that, is won back."""
    if not board.move_stack:
        return 0
    before = board.copy()
    last = before.pop()
    if not before.is_capture(last):
        return 0
    if before.is_en_passant(last):
        return exchange.VALUE.get(chess.PAWN, 1)
    return exchange.VALUE.get(before.piece_type_at(last.to_square) or chess.PAWN, 0)


def _took(board: chess.Board, square: chess.Square) -> bool:
    """Whether their last move captured on `square`, when the board knows its moves."""
    if not board.move_stack:
        return False
    last = board.peek()
    if last.to_square != square:
        return False
    before = board.copy()
    before.pop()
    return before.is_capture(last)


def _forcing_reply(board: chess.Board, idea: Idea) -> str:
    """Their reply and the player's answer, when the reply is a check or a capture: the
    in-between move a plain capture must still meet."""
    line = list(idea.line[:3])
    if len(line) < 3:
        return ""
    b = board.copy(stack=False)
    said: list[str] = []
    forcing = False
    for j, move in enumerate(line):
        if move not in b.legal_moves:
            return ""
        if j == 1:
            forcing = b.is_capture(move) or b.gives_check(move)
        said.append(b.san(move))
        b.push(move)
    return f"; after {said[1]}, {said[2]}" if forcing else ""


def queening(board: chess.Board, idea: Idea) -> str | None:
    """The player's pawn the line pushes to the seventh or promotes, where the point rests on
    it: "your pawn on f6 is two squares from queening" (uSAIW: f7, and the rook must give
    itself up)."""
    if idea.goal not in ("win", "combination", "mate", "attack") or not idea.line:
        return None
    boards = [board.copy(stack=False)]
    for move in idea.line:
        if move not in boards[-1].legal_moves:
            break
        after = boards[-1].copy(stack=False)
        after.push(move)
        boards.append(after)
    line = list(idea.line[: len(boards) - 1])
    last = len(line) if idea.goal in ("mate", "attack") else idea.gain_ply + 1
    us = board.turn
    seventh = 6 if us == chess.WHITE else 1
    for j, move in enumerate(line[:last]):
        if j % 2 or boards[j].piece_type_at(move.from_square) != chess.PAWN:
            continue
        # Pushed now, or queening in the line: a pawn that only reaches the seventh later is
        # not the point, and naming it sent the player to push it (25.e6?? in Carlsen-Chao).
        if not (move.promotion or (j == 0 and chess.square_rank(move.to_square) == seventh)):
            continue
        start = idea_of.origin(board, boards, j, move.from_square)
        if start is None or board.piece_type_at(start) != chess.PAWN:
            return None
        rank = chess.square_rank(start)
        to_go = 7 - rank if us == chess.WHITE else rank
        return (
            f"Your pawn on {_sq(start)} is {_NUMBER.get(to_go, f'{to_go} squares')} from queening"
        )
    return None


def prize(board: chess.Board, idea: Idea, said: str) -> str | None:
    """The piece the line wins, where it stands, when what is `said` does not name it: "the
    prize is their queen on c7" beside a discovered attack whose line names their knight."""
    if idea.goal not in ("win", "combination") or idea.target is None:
        return None
    piece = board.piece_at(idea.target)
    if piece is None or piece.color == board.turn or piece.piece_type == chess.KING:
        return None
    if _named_any_won(board, idea, said):
        return None
    return f"The prize is their {_named(board, idea.target)}"


def _named_any_won(board: chess.Board, idea: Idea, said: str) -> bool:
    """Whether `said` names a piece of the kind the line wins that it takes on the way."""
    top = board.piece_type_at(idea.target) if idea.target is not None else None
    if top is None:
        return False
    boards = [board.copy(stack=False)]
    for move in idea.line:
        if move not in boards[-1].legal_moves:
            break
        after = boards[-1].copy(stack=False)
        after.push(move)
        boards.append(after)
    for j, move in enumerate(idea.line[: len(boards) - 1]):
        if j % 2:
            continue
        taken = boards[j].piece_at(move.to_square)
        if taken is None or taken.color == board.turn or taken.piece_type != top:
            continue
        start = idea_of.origin(board, boards, j, move.to_square)
        if start is not None and f"{_NAMES[top]} on {_sq(start)}" in said:
            return True
    return f"{_NAMES[top]} on {_sq(idea.target)}" in said if idea.target is not None else False


@dataclass(frozen=True)
class Effect:
    """What a move does, in words for the third hint and a question for the second."""

    said: str  # "gives check", "takes their pawn on h5", "stops Nd6+"
    ask: str  # "Which check is it?"


def effects(board: chess.Board, move: chess.Move, threat: chess.Move | None) -> list[Effect]:
    """What `move` does on the board, the most forcing first: a capture, a check, an attack
    on a piece of theirs it did not hit before, their idea stopped, a passed pawn pushed."""
    if move not in board.legal_moves:
        return []
    us = board.turn
    out: list[Effect] = []
    after = board.copy(stack=False)
    after.push(move)
    if board.is_capture(move):
        square = move.to_square
        if board.is_en_passant(move):
            square = chess.square(
                chess.square_file(move.to_square), chess.square_rank(move.from_square)
            )
        out.append(Effect(f"takes their {_named(board, square)}", "What can you take?"))
    if after.is_check():
        out.append(Effect("gives check", "Which check is it?"))
    before = board.attacks_mask(move.from_square)
    for square in chess.SquareSet(after.attacks_mask(move.to_square) & ~before):
        piece = after.piece_at(square)
        if piece is None or piece.color == us or piece.piece_type == chess.KING:
            continue
        loose = not exchange.defenders(after, piece.color, square)
        if piece.piece_type in (chess.QUEEN, chess.ROOK) or (
            loose and piece.piece_type != chess.PAWN
        ):
            out.append(Effect(f"attacks their {_named(after, square)}", "What can it attack?"))
            break
    if threat is not None:
        passed = board.copy(stack=False)
        passed.push(chess.Move.null())
        if threat in passed.legal_moves:
            their = passed.san(threat)
            worked = not passed.is_capture(threat) or exchange.exchange(passed, threat) > 0
            stopped = threat not in after.legal_moves or (
                passed.is_capture(threat)
                and after.is_capture(threat)
                and exchange.exchange(after, threat) <= 0
                and worked
            )
            if stopped and threat.to_square == move.to_square:
                # The move takes the very square theirs wanted: said by their piece, or the
                # square named would be the answer ("stops f3" when the move is f3).
                mover = f"their {_named(passed, threat.from_square)}"
                out.append(
                    Effect(f"takes the square {mover} wants", f"Where does {mover} want to go?")
                )
            elif stopped and threat.from_square != move.to_square:
                out.append(Effect(f"stops {their}", f"How do you stop {their}?"))
    moved = board.piece_at(move.from_square)
    if (
        moved is not None
        and moved.piece_type != chess.KING
        and exchange.can_be_won(board, move.from_square)
        and not exchange.can_be_won(after, move.to_square)
    ):
        out.append(
            Effect(
                f"takes your {_NAMES[moved.piece_type]} out of danger",
                "Which of your pieces is in danger?",
            )
        )
    for square in chess.SquareSet(board.occupied_co[us]):
        if square == move.from_square or not exchange.can_be_won(board, square):
            continue
        if after.piece_at(square) is not None and not exchange.can_be_won(after, square):
            out.append(
                Effect(f"covers your {_named(board, square)}", "Which of your pieces needs cover?")
            )
            break
    pawn_ending = all(p.piece_type in (chess.KING, chess.PAWN) for p in board.piece_map().values())
    if moved is not None and moved.piece_type == chess.KING and pawn_ending:
        ours, theirs = after.king(us), after.king(not us)
        if ours is not None and theirs is not None:
            files = abs(chess.square_file(ours) - chess.square_file(theirs))
            ranks = abs(chess.square_rank(ours) - chess.square_rank(theirs))
            if sorted((files, ranks)) == [0, 2]:
                out.append(Effect("takes the opposition", "Where does your king face theirs?"))
    piece = board.piece_at(move.from_square)
    if (
        piece is not None
        and piece.piece_type == chess.PAWN
        and not board.is_capture(move)
        and positional.is_passed(board, move.from_square)
    ):
        out.append(Effect("pushes your passed pawn", "How far can your passed pawn go?"))
    return out


def reason(
    board: chess.Board, move: chess.Move, threat: chess.Move | None, holds: str
) -> tuple[str, str]:
    """The third hint's reason and the second hint's question for a move that holds where
    little else does (`holds`: "it is the only move that keeps your edge"): "Bb5+ gives check
    and stops Nf6: it is the only move that keeps your edge"."""
    san = board.san(move)
    found = effects(board, move, threat)
    if found:
        words = " and ".join(e.said for e in found[:2])
        return f"{san} {words}: {holds}", found[0].ask
    return f"{san}: {holds}", "Which move keeps everything you have?"


def how(board: chess.Board, idea: Idea, threat: chess.Move | None) -> str | None:
    """How a win with no named pattern comes about, for the first hint: what the first move
    does, never the move. "Their pawn on h5 is the target" alone left the player to find the
    king walk that wins it (Lichess jlCZP, a zugzwang); "the first move attacks their rook on
    d8" says how."""
    if idea.goal not in ("win", "combination") or idea.blow is not None or idea.first is None:
        return None
    if idea.gain_ply == 0 and board.is_capture(idea.first):
        return None  # the capture is the how
    if idea.intermezzo:
        return "A check comes first, before anything is taken"
    if idea.sacrifices and idea.sacrifices[0][0] == 0:
        given = idea.sacrifices[0][1]
        return f"It starts by giving up {'the' if given == 'exchange' else 'a'} {given}"
    found = effects(board, idea.first, threat)
    # Taking a piece is the point when it is a piece, or the one giving check; a pawn taken on
    # the way is not, and what else the move does comes first.
    first = idea.first
    taken = board.piece_type_at(first.to_square) if board.is_capture(first) else None
    big = taken is not None and (
        exchange.VALUE.get(taken, 0) >= FREE_MIN or first.to_square in board.checkers()
    )
    if not big:
        found = [e for e in found if not e.said.startswith("takes their")] or found
    # "pushes your passed pawn" is the move itself, not how it wins.
    found = [e for e in found if not e.said.startswith("pushes")]
    if not found:
        return None
    return f"The first move {found[0].said}"


def move_why(board: chess.Board, idea: Idea, threat: chess.Move | None) -> str | None:
    """The third hint for a win with no named pattern: what the move does, the line's next two
    moves, what it wins -- "Kf4 takes the opposition, and after Kf6, Kg5; you win two pawns",
    never "Kf4: you win two pawns, a few moves on"."""
    first = idea.first
    if first is None or first not in board.legal_moves or not idea.won:
        return None
    san = board.san(first)
    found = effects(board, first, threat)
    did = " and ".join(e.said for e in found[:2])
    b = board.copy(stack=False)
    moves: list[str] = []
    for move in idea.line[:3]:
        if move not in b.legal_moves:
            break
        moves.append(b.san(move))
        b.push(move)
    then = f", and after {moves[1]}, {moves[2]}" if len(moves) == 3 else ""
    head = f"{san} {did}" if did else san
    if not did and not then:
        return None
    later = ", a few moves on" if idea.gain_ply > 2 else ""
    return f"{head}{then}; you win {idea.won}{later}"


def line_reason(board: chess.Board, line: list[chess.Move], holds: str) -> str | None:
    """The best line's next two moves, when nothing on the board says what the move does:
    "a3, and after Kf6, Kd4 -- it is the only move that keeps your edge"."""
    if len(line) < 3:
        return None
    b = board.copy(stack=False)
    said: list[str] = []
    for move in line[:3]:
        if move not in b.legal_moves:
            return None
        said.append(b.san(move))
        b.push(move)
    return f"{said[0]}, and after {said[1]}, {said[2]} -- {holds}"
