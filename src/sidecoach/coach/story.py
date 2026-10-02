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

import re
from dataclasses import dataclass

import chess

from sidecoach.analysis import exchange, positional
from sidecoach.coach import idea as idea_of
from sidecoach.coach import idea_words
from sidecoach.coach.idea import Idea
from sidecoach.events import Line

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


@dataclass(frozen=True)
class Defence:
    """Why the obvious defence fails: the recapture the best line does not play, and what
    the piece that would take back is also doing (`analysis/defences.py`)."""

    said: str  # "Their pawn on f2 cannot take back on g3: it also shields their king"
    clause: str  # at the move rung: "their pawn on f2 cannot take back on g3"
    square: str  # where that piece stands: "f2"
    target: str  # the square it would take back on: "g3"
    who: str  # "their pawn on f2"
    duty: str  # "shields their king"
    # The player's pawn it has to stop, where it stands now, when that is its other job: the
    # first hint says the two as one ("your pawn on e6 is two squares from queening, and
    # their rook on f8 has to stop it").
    stops: str | None
    lesson: str  # the overloaded defender's lesson
    ask: str  # the second hint's question
    line: str  # at the move rung: "if fxg3, Bc5+"


def _duty(
    before: chess.Board,
    recapture: chess.Move,
    refute: chess.Move | None,
    now: chess.Board | None = None,
) -> str | None:
    """What the piece that would take back is also doing, read off what the refutation does
    with the square it would leave: guards the square a pawn queens on, shields its king
    from a check through that square, guards the square, or defends the piece the refutation
    takes. None when the refutation uses none of that."""
    if refute is None or refute.to_square == recapture.to_square:
        # Taking the piece that took back is the square being guarded twice, not a second
        # job: "it also defends their queen on f5" of the queen that took on f5.
        return None
    left = recapture.from_square
    after = before.copy(stack=False)
    after.push(recapture)
    if refute not in after.legal_moves:
        return None
    mover = after.piece_at(refute.from_square)
    if mover is not None and mover.piece_type == chess.PAWN and refute.to_square == left:
        if refute.promotion:
            return f"guards {_sq(left)}, where your pawn queens"
        return f"guards {_sq(left)} against your pawn"
    struck = after.copy(stack=False)
    struck.push(refute)
    king = struck.king(struck.turn)
    if king is not None and struck.is_check():
        for checker in struck.checkers():
            if left in chess.SquareSet(chess.between(checker, king)):
                return "shields their king"
    if refute.to_square == left:
        return f"guards {_sq(left)}"
    taken = after.piece_at(refute.to_square)
    if (
        taken is not None
        and refute.to_square in before.attacks(left)
        and now is not None
        and now.piece_at(refute.to_square) == taken
    ):
        # Named where it stands now: a pawn that gets there later in the line is no fact yet.
        return f"defends their {_named(after, refute.to_square)}"
    return None


def defence(board: chess.Board, defences: list[Line], best: list[str]) -> Defence | None:
    """The analysis's declined recapture (from the best line) whose recapturing piece has
    another job, in words: the overloaded defender the line turns on. Of several, the one
    after the capture that takes the most: the rook the line wins on b8, not the pawn it takes
    on f7 on the way."""
    for d in sorted(defences, key=lambda d: -_captured_value(board, d.moves, best)):
        moves = [chess.Move.from_uci(u) for u in d.moves]
        cut = next((i for i, u in enumerate(d.moves) if i >= len(best) or u != best[i]), None)
        if cut is None or cut == 0 or cut % 2 == 0:
            continue  # the recapture is the opponent's move, after a capture of the player's
        before = board.copy(stack=False)
        try:
            for move in moves[:cut]:
                before.push(move)
        except (AssertionError, ValueError):
            continue
        recapture = moves[cut]
        if recapture not in before.legal_moves:
            continue
        piece = before.piece_at(recapture.from_square)
        if piece is None or piece.piece_type == chess.KING:
            continue  # a king has no second job to give up: "it also shields their king"
        # Named as the board shows them now: the piece that would take back on its square,
        # and something of theirs on the square it would take back on -- never a square the
        # line only reaches later, which is no fact yet and may be the first move's own.
        now_piece = board.piece_at(recapture.from_square)
        now_target = board.piece_at(recapture.to_square)
        if (
            now_piece != piece
            or now_target is None
            or now_target.color == board.turn
            or recapture.to_square != moves[cut - 1].to_square
        ):
            continue
        refute = moves[cut + 1] if cut + 1 < len(moves) else None
        duty = _duty(before, recapture, refute, board)
        if duty is None or refute is None:
            continue
        recapture_san = before.san(recapture)
        after = before.copy(stack=False)
        after.push(recapture)
        stops: str | None = None
        runner = after.piece_at(refute.from_square)
        if (
            runner is not None
            and runner.piece_type == chess.PAWN
            and refute.to_square == recapture.from_square
        ):
            boards = [board.copy(stack=False)]
            for move in moves[:cut]:
                nxt = boards[-1].copy(stack=False)
                nxt.push(move)
                boards.append(nxt)
            start = idea_of.origin(board, boards, cut, refute.from_square)
            if start is not None and board.piece_type_at(start) == chess.PAWN:
                stops = _sq(start)
        cannot = (
            f"{_NAMES[piece.piece_type]} on {_sq(recapture.from_square)} cannot take "
            f"back on {_sq(recapture.to_square)}"
        )
        return Defence(
            said=f"Their {cannot}: it also {duty}",
            clause=f"their {cannot}",
            square=_sq(recapture.from_square),
            target=_sq(recapture.to_square),
            who=f"their {_NAMES[piece.piece_type]} on {_sq(recapture.from_square)}",
            duty=duty,
            stops=stops,
            lesson=(
                f"{idea_words.PATTERN_NAMES['overloading']}: "
                f"{idea_words.PATTERN_WHY['overloading']}"
            ),
            ask="Which of their pieces has two jobs?",
            line=f"if {recapture_san}, {after.san(refute)}",
        )
    return None


def _captured_value(board: chess.Board, moves: list[str], best: list[str]) -> int:
    """What the player's capture before a defence's recapture takes, in pawns."""
    cut = next((i for i, u in enumerate(moves) if i >= len(best) or u != best[i]), None)
    if cut is None or cut == 0:
        return 0
    at = board.copy(stack=False)
    try:
        for uci in moves[: cut - 1]:
            at.push_uci(uci)
    except (AssertionError, ValueError):
        return 0
    taken = at.piece_at(chess.Move.from_uci(moves[cut - 1]).to_square)
    return idea_of.VALUES.get(taken.piece_type, 0) if taken is not None else 0


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


def trapped(board: chess.Board, square: chess.Square) -> bool:
    """The piece on `square`, its side to move, is a trapped piece: attacked for profit where
    it stands, and every move of it lands where it is won too, or takes something for less
    than itself. Tai's chess.com puzzle, 2026-09-24: Bg4 against a queen on h5 that had
    nowhere to go was hinted as "the first move attacks their queen", not as the trapped
    piece the puzzle teaches."""
    piece = board.piece_at(square)
    if piece is None or piece.piece_type in (chess.KING, chess.PAWN):
        return False
    if not exchange.can_be_won(board, square):
        return False
    for move in board.legal_moves:
        if move.from_square != square:
            continue
        if board.is_capture(move):
            if exchange.exchange(board, move) >= 0:
                return False  # it escapes by taking something worth it
            continue
        after = board.copy(stack=False)
        after.push(move)
        if not exchange.can_be_won(after, move.to_square):
            return False
    return True


@dataclass(frozen=True)
class Effect:
    """What a move does, in words for the third hint and a question for the second, and the
    pattern's lesson when the move carries one."""

    said: str  # "gives check", "takes their pawn on h5", "stops Nd6+"
    ask: str  # "Which check is it?"
    lesson: str | None = None


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
        if move.to_square not in exchange.capturers(after, us, square):
            # A line it cannot take along is no attack: a pinned piece, or a king onto a
            # guarded piece (the facts audit, 2026-10-01).
            continue
        loose = not exchange.defenders(after, piece.color, square)
        if trapped(after, square):
            out.append(
                Effect(
                    f"attacks their {_named(after, square)}, and it has no safe square to go to",
                    "Which of their pieces has no safe square left?",
                    f"{idea_words.PATTERN_NAMES['trappedPiece']}: "
                    f"{idea_words.PATTERN_WHY['trappedPiece']}",
                )
            )
            break
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
            # "Covers" only when the moved piece guards it; a block, a capture or a pin
            # that saves it makes it safe ("covers your pawn on f4" of Ne4, 2026-10-01).
            guards = move.to_square in exchange.defenders(after, us, square)
            if not guards and after.is_check():
                continue  # a check only puts the capture off by a move
            said = "covers your" if guards else "makes your"
            safe = "" if guards else " safe"
            out.append(
                Effect(f"{said} {_named(board, square)}{safe}", "Which of your pieces needs cover?")
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


def how(
    board: chess.Board, idea: Idea, threat: chess.Move | None, aim: Line | None = None
) -> Effect | None:
    """How a win with no named pattern comes about, for the first hint: what the first move
    does, never the move. "Their pawn on h5 is the target" alone left the player to find the
    king walk that wins it (Lichess jlCZP, a zugzwang); "the first move attacks their rook on
    d8" says how."""
    if idea.goal not in ("win", "combination") or idea.blow is not None or idea.first is None:
        return None
    if idea.gain_ply == 0 and board.is_capture(idea.first):
        return None  # the capture is the how
    if idea.intermezzo:
        return Effect("A check comes first, before anything is taken", "")
    if idea.sacrifices and idea.sacrifices[0][0] == 0:
        given = idea.sacrifices[0][1]
        return Effect(f"It starts by giving up {'the' if given == 'exchange' else 'a'} {given}", "")
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
    pushes = any(e.said.startswith("pushes") for e in found)
    found = [e for e in found if not e.said.startswith("pushes")]
    if not found:
        # A quiet move: what it threatens, else the bait or the passed pawn it sets going.
        return aim_how(board, idea, aim) or quiet_how(board, idea, pushes)
    first_effect = found[0]
    return Effect(f"The first move {first_effect.said}", first_effect.ask, first_effect.lesson)


_MATE_IN = {2: "two", 3: "three", 4: "four", 5: "five"}
_QUIET = "The first move is quiet, but it threatens"
_HIT = re.compile(r"attacks their (\w+ on [a-h][1-8])(, and it has no safe square to go to)?")


def aim_how(board: chess.Board, idea: Idea, aim: Line | None) -> Effect | None:
    """How a quiet first move wins: what it threatens (`Analysis.threatens`, the engine's best
    line for the player after the move and a free move for the opponent). "Their knight on c3
    is the target" alone left the player to find a quiet move with nothing to go on (the 28
    wins of the puzzle sample the story checks found bare, 2026-09-25); "the first move is
    quiet, but it threatens mate" is the lesson. Said by what the threat does -- a mate, a pawn
    that queens, a piece taken, a check that hits a piece too, material won -- naming only what
    stands on the board now, never the move or where it goes."""
    first = idea.first
    if aim is None or not aim.moves or first is None or first not in board.legal_moves:
        return None
    after = board.copy(stack=False)
    after.push(first)
    if after.is_check():
        return None
    passed = after.copy(stack=False)
    passed.push(chess.Move.null())
    try:
        threat = chess.Move.from_uci(aim.moves[0])
    except ValueError:
        return None
    if threat not in passed.legal_moves:
        return None
    if aim.mate is not None and aim.mate > 0:
        said = (
            f"{_QUIET} mate"
            if aim.mate == 1
            else f"{_QUIET} mate in {_MATE_IN[aim.mate]}"
            if aim.mate in _MATE_IN
            else f"{_QUIET} a mating attack"
        )
        return Effect(said, "Which quiet move sets up a mate?")
    # A pawn of yours that queens along the threat.
    b = passed.copy(stack=False)
    for j, uci in enumerate(aim.moves[:6]):
        move = chess.Move.from_uci(uci)
        if move not in b.legal_moves:
            break
        if j % 2 == 0 and move.promotion is not None:
            return Effect(
                f"{_QUIET} to queen a pawn", "Which quiet move makes your pawn unstoppable?"
            )
        b.push(move)
    found = effects(passed, threat, None)
    check = any(e.said == "gives check" for e in found)
    taken = passed.piece_at(threat.to_square) if passed.is_capture(threat) else None
    if (
        taken is not None
        and taken.piece_type not in (chess.PAWN, chess.KING)
        and exchange.exchange(passed, threat) > 0
    ):
        return Effect(
            f"{_QUIET} to take their {_named(passed, threat.to_square)}",
            "Which of their pieces would a quiet move put in danger?",
        )
    hit = next((m for e in found if (m := _HIT.fullmatch(e.said)) is not None), None)
    if hit is not None and check:
        tail = ", which has no safe square left" if hit.group(2) else ""
        return Effect(
            f"{_QUIET} a check that also attacks their {hit.group(1)}{tail}",
            "Which check could hit two things at once, once the way is ready?",
        )
    if hit is not None and hit.group(2):
        return Effect(
            f"{_QUIET} their {hit.group(1)}, which has no safe square left",
            "Which of their pieces is short of squares?",
        )
    net = _line_net(passed, aim.moves)
    if net >= 1:
        what = "a pawn" if net == 1 else "material"
        how_ = f"a check that wins {what}" if check else f"to win {what}"
        return Effect(f"{_QUIET} {how_}", "What would you play next if they did nothing?")
    # Nothing won yet within reach: the break or the check the move prepares.
    mover = passed.piece_at(threat.from_square)
    if (
        mover is not None
        and mover.piece_type == chess.PAWN
        and not passed.is_capture(threat)
        and any(
            (target := passed.piece_at(sq)) is not None
            and target.color != passed.turn
            and target.piece_type == chess.PAWN
            for sq in chess.SquareSet(chess.BB_PAWN_ATTACKS[passed.turn][threat.to_square])
        )
    ):
        return Effect(f"{_QUIET} a pawn break", "Which pawn break could a quiet move prepare?")
    if check:
        return Effect(f"{_QUIET} a check", "Which check could a quiet move prepare?")
    return None


def _line_net(board: chess.Board, moves: list[str], plies: int = 6) -> int:
    """What `moves` net the side to move in material, in pawns, after their last reply
    within `plies`."""
    us = board.turn
    b = board.copy(stack=False)

    def balance() -> int:
        return sum(
            idea_of.VALUES.get(piece.piece_type, 0) * (1 if piece.color == us else -1)
            for piece in b.piece_map().values()
        )

    start = balance()
    net = 0
    for j, uci in enumerate(moves[:plies]):
        move = chess.Move.from_uci(uci)
        if move not in b.legal_moves:
            break
        b.push(move)
        if j % 2 == 1:
            net = balance() - start
    return net


def quiet_how(board: chess.Board, idea: Idea, pushes: bool = False) -> Effect | None:
    """How a quiet first move wins when there is no threat to say (`aim_how`), read off the
    best line: the bait it sets -- the piece it moves is taken, and the next move takes the
    taker or what the taker had been guarding -- or the passed pawn it sets going, which costs
    them a piece to stop. Names only what stands on the board now, never the move."""
    first = idea.first
    line = list(idea.line)
    if first is None or not line or line[0] != first or first not in board.legal_moves:
        return None
    after = board.copy(stack=False)
    after.push(first)
    if len(line) >= 3 and line[1] in after.legal_moves:
        reply, follow = line[1], line[2]
        taker = after.piece_at(reply.from_square)
        # The bait is what the first move itself puts where it can be taken.
        if (
            after.is_capture(reply)
            and reply.to_square == first.to_square
            and taker is not None
            and taker.piece_type != chess.KING
        ):
            took = after.copy(stack=False)
            took.push(reply)
            if follow in took.legal_moves and took.is_capture(follow):
                name = _named(after, reply.from_square)
                if follow.to_square == reply.to_square:
                    return Effect(
                        f"The first move sets a bait: if their {name} takes, it is lost",
                        "What could you offer that they cannot afford to take?",
                    )
                if follow.to_square in after.attacks(reply.from_square):
                    return Effect(
                        f"The first move sets a bait: if their {name} takes, their "
                        f"{_named(after, follow.to_square)} loses its guard",
                        "Which of their pieces is doing the guarding?",
                    )
    # The passed pawn: pushed, or walked in behind by the king, and stopping it costs them.
    moved = board.piece_at(first.from_square)
    runner = pushes or (
        moved is not None
        and moved.piece_type == chess.KING
        and len(line) >= 3
        and _pushes_passer(after, line[1], line[2])
    )
    if runner and idea.won and " for " in idea.won:
        cost = idea.won.split(" for ")[0].removeprefix("back ")
        return Effect(
            f"The first move gets your passed pawn going, and stopping it costs them {cost}",
            "How far can your passed pawn go?",
        )
    return None


def _pushes_passer(after: chess.Board, reply: chess.Move, follow: chess.Move) -> bool:
    """Whether `follow`, after `reply`, pushes a passed pawn of the side that played first."""
    if reply not in after.legal_moves:
        return False
    b = after.copy(stack=False)
    b.push(reply)
    piece = b.piece_at(follow.from_square)
    return (
        follow in b.legal_moves
        and piece is not None
        and piece.piece_type == chess.PAWN
        and not b.is_capture(follow)
        and positional.is_passed(b, follow.from_square)
    )


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
