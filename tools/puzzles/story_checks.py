"""The story checks: the hints tell the story the best line tells, in the order a player meets it.

Tai, 2026-09-24, after the puzzles were read by hand: the first hint's parts -- what the line
wins, which tactic to name, which piece is the target, which plan -- were each chosen on their
own, and nothing made them agree with the move the player has to find. "You can win their
queen. Their rook on f5, on a line one of your pieces is blocking" when a pawn simply takes the
queen. Each check below is one way the parts can disagree, read from the board and the engine's
line alone, never from the code that writes the hint:

  won      what the line wins is named where it stands ("their knight on d4"), not only the
           pieces of the pattern that wins it
  plain    a first move that simply takes a piece -- the exchange on its square nets what the
           whole line does -- is said as that, with no other piece of theirs named
  free     a first move that takes a minor piece or more for nothing names it, even when the
           line wins more after it
  checker  in check, the piece giving check is named
  queening a pawn the first move pushes to the seventh, or the line promotes, is named
  reason   where only one or a few moves hold, the second hint asks a question and the third
           gives a reason, never "The move is Bb5+." alone
  how      a win with no named pattern says how it comes about: the first hint more than "their
           pawn on h5 is the target", the third more than "Kf4: you win two pawns"

Used by `coach_bench.py` on the Lichess puzzles and by `tools/eval/story_masters.py` on the
master games.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import chess

_NAMES = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}
_THEIR_ON = re.compile(r"\btheir (king|queen|rook|bishop|knight|pawn) on ([a-h][1-8])")
_SHARP = (
    "Only ",
    "You're winning, but only",
    "You're in check, and only",
    "You're in check but winning, and only",
)
_GENERIC_MOVE = re.compile(
    r"^The move is \S+\.$|goes from \d+ squares? to|centralising|fighting for the cent"
    # the stakes said again is not a reason: "Bf5: it is the only move that keeps your edge"
    r"|^\S+: it is (?:the only move|one of the few moves)"
)
_QUEENING = re.compile(r"queening|promot|new queen|to queen\b", re.I)
# A first hint that names only what falls, a third that says only what is won.
_BARE_FIRST = re.compile(
    r"^(?:You're[^.]*\. )?(?:Their \w+ on [a-h][1-8] gives check[^.]*\. )?"
    r"(?:You can win|[^.]*\btactic\b[^.]* wins) [^.]*\. "
    r"(?:The prize is [^.]*\. )?Their \w+ on [a-h][1-8] is the target\.$"
)
_BARE_MOVE = re.compile(r"^\S+: you win [^;.]*(, a few moves on)?\.$")


def _named(text: str, board: chess.Board, square: chess.Square) -> bool:
    piece = board.piece_at(square)
    if piece is None:
        return False
    return f"{_NAMES[piece.piece_type]} on {chess.square_name(square)}" in text


def _without_traps(text: str) -> str:
    return " ".join(
        s
        for s in re.split(r"(?<=[.?!])\s+", text)
        if not s.startswith(("Careful", "Take back with the right piece"))
        and ", but not at once: " not in s
    )


def _boards(board: chess.Board, pv: Sequence[chess.Move]) -> list[chess.Board]:
    out = [board.copy(stack=False)]
    for move in pv:
        if move not in out[-1].legal_moves:
            break
        nxt = out[-1].copy(stack=False)
        nxt.push(move)
        out.append(nxt)
    return out


def _start(
    board: chess.Board, boards: Sequence[chess.Board], ply: int, square: chess.Square
) -> chess.Square | None:
    """Where the piece on `square` at `ply` stands now: followed back through the moves that
    brought it there."""
    at = square
    for i in range(ply - 1, -1, -1):
        before, after = boards[i], boards[i + 1]
        piece = after.piece_at(at)
        if piece is not None and before.piece_at(at) != piece:
            came = [
                sq
                for sq in chess.SQUARES
                if before.piece_at(sq) == piece and after.piece_at(sq) is None
            ]
            if not came:
                return None
            at = came[0]
    return at if board.piece_at(at) is not None else None


def story_checks(
    board: chess.Board,
    pv: Sequence[chess.Move],
    idea: object,
    texts: dict[str, str],
) -> tuple[dict[str, bool], dict[str, str]]:
    """The story checks on the hints `texts` ("idea", "focus", "move") for `board`, whose best
    line is `pv` and whose reading by the idea reader is `idea`."""
    from sidecoach.analysis import exchange

    checks: dict[str, bool] = {}
    notes: dict[str, str] = {}
    first_text = texts.get("idea", "")
    us = board.turn
    goal = getattr(idea, "goal", None)
    gain = getattr(idea, "gain", 0) or 0
    gain_ply = getattr(idea, "gain_ply", 0) or 0
    target = getattr(idea, "target", None)
    boards = _boards(board, pv)
    pv = list(pv[: len(boards) - 1])
    forced = board.is_check() and board.legal_moves.count() == 1
    if forced:
        return checks, notes  # one legal move: the hint says so, and nothing else

    # won: what the line wins, named where it stands now -- any of the pieces of that kind
    # the line takes (two knights won: either one).
    if (
        goal in ("win", "combination")
        and target is not None
        and board.color_at(target) == (not us)
        and board.piece_type_at(target) != chess.KING
    ):
        kind = board.piece_type_at(target)
        taken = {target}
        for j, move in enumerate(pv):
            if (
                j % 2 == 0
                and boards[j].piece_type_at(move.to_square) == kind
                and boards[j].color_at(move.to_square) == (not us)
            ):
                taken.add(_start(board, boards, j, move.to_square))
        # A first move that takes a piece netting the whole win on its own is the story's
        # first beat: naming that piece is naming what the line turns on.
        if pv and board.is_capture(pv[0]) and exchange.exchange(board, pv[0]) >= max(gain, 1):
            taken.add(pv[0].to_square)
        checks["won"] = any(sq is not None and _named(first_text, board, sq) for sq in taken)
        if not checks["won"]:
            what = _NAMES[kind or chess.PAWN]
            notes["won"] = f"not named: {what} on {chess.square_name(target)}"

    # plain: a first move that takes a piece and nets, on its own, what the line wins.
    if pv and goal == "win" and board.is_capture(pv[0]) and not board.is_en_passant(pv[0]):
        net = exchange.exchange(board, pv[0])
        if net >= max(gain, 1):
            taken = pv[0].to_square
            said = _without_traps(first_text)
            others = [
                m
                for m in _THEIR_ON.finditer(said)
                if m.group(2) != chess.square_name(taken)
                and not (board.is_check() and chess.parse_square(m.group(2)) in board.checkers())
            ]
            checks["plain"] = _named(said, board, taken) and not others
            if not checks["plain"]:
                notes["plain"] = (
                    f"{board.san(pv[0])} takes the {_NAMES[board.piece_type_at(taken) or 1]} on "
                    f"{chess.square_name(taken)} (nets {net}); said: {said[:140]}"
                )

    # free: a first move that takes a whole minor piece or more names it.
    if pv and goal in ("win", "combination") and board.is_capture(pv[0]):
        piece = board.piece_at(pv[0].to_square)
        worth = exchange.VALUE.get(piece.piece_type, 0) if piece is not None else 0
        if worth >= 3 and exchange.exchange(board, pv[0]) >= worth:
            checks["free"] = _named(first_text, board, pv[0].to_square)
            if not checks["free"]:
                notes["free"] = f"{board.san(pv[0])} takes it for nothing; said: {first_text[:140]}"

    # checker: in check, the piece giving check.
    if board.is_check() and not forced:
        checks["checker"] = any(_named(first_text, board, sq) for sq in board.checkers())
        if not checks["checker"]:
            notes["checker"] = first_text[:140]

    # queening: a pawn of the player's pushed to the seventh or promoted where the point rests.
    if pv and goal in ("win", "combination", "mate", "promote", "attack"):
        last = len(pv) if goal in ("mate", "promote") else gain_ply + 1
        seventh = 6 if us == chess.WHITE else 1
        for j, move in enumerate(pv[:last]):
            if j % 2 or boards[j].piece_type_at(move.from_square) != chess.PAWN:
                continue
            if move.promotion or (j == 0 and chess.square_rank(move.to_square) == seventh):
                checks["queening"] = bool(_QUEENING.search(first_text))
                if not checks["queening"]:
                    notes["queening"] = (
                        f"{boards[j].san(move)} on move {j // 2 + 1}; said: {first_text[:120]}"
                    )
                break

    # how: a win with no named pattern says how it comes about.
    if (
        goal in ("win", "combination")
        and getattr(idea, "blow", None) is None
        and pv
        and not (gain_ply == 0 and board.is_capture(pv[0]))
    ):
        bare_first = _BARE_FIRST.match(first_text) is not None
        bare_move = _BARE_MOVE.match(texts.get("move", "")) is not None
        checks["how"] = not bare_first and not bare_move
        if not checks["how"]:
            notes["how"] = f"{first_text[:120]} | {texts.get('move', '')[:80]}"

    # reason: where only one or a few moves hold, a question and a reason.
    if first_text.startswith(_SHARP) and not forced:
        move_text = texts.get("move", "")
        focus = texts.get("focus", "")
        checks["reason"] = "?" in focus and not _GENERIC_MOVE.search(move_text)
        if not checks["reason"]:
            notes["reason"] = f"{focus[:80]} | {move_text[:80]}"
    return checks, notes


STORY_CHECKS = ("won", "plain", "free", "checker", "queening", "reason", "how")
