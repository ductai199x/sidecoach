"""The coach's own draft, claim by claim: a model lists what it asserts, the board checks it.

The draft check in `claims.check_draft` reads patterns ("your queen checks", "forces the King
to h6"), and patterns miss the wording a model finds for the same false statement. Hand-read,
the false claims of the locked test set and dev (2026-09-26, 37 of them) were nearly all
checkable facts put in other words: whose piece or move it is, what a move takes, attacks,
stops or allows at some moment of a line, the only move, a mate, a file or a pawn's state.

So a model does only what it is good at -- reading the sentence -- and lists each claim as a
record with the moment it is about (the moves leading there from the position given), and
python-chess decides every one. A claim this module cannot place on a board, or cannot
decide, is left alone: a true reply is never refused for being hard to read.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import chess

from sidecoach.analysis.positional import is_passed
from sidecoach.llm.base import LLMProvider

EXTRACT_SYSTEM = """\
You read a chess coach's reply and list every concrete claim it makes about the board, so \
each can be checked with a chess engine. You do not judge the claims: you only transcribe \
them.

You are given the position the reply is about (FEN), which side the coach speaks to \
("you"), the move just played if the reply is about one, the engine's best line from the \
position, and the reply.

Reply with a JSON array only, no prose, no code fences. Each element is an object with:
- "kind": one of the kinds below
- "after": the moves, in SAN, played from the given position to reach the moment the claim \
is about, as the reply itself puts it -- [] for the given position. A claim about the position \
after the move just played starts with that move. "After Bg4, Ne7+ wins the bishop" said of \
the line Rg3 Bg4 Ne7+ is about the moment after ["Rg3", "Bg4"], and the move is "Ne7+".
- "quote": the words the claim comes from, copied exactly from the reply
- the fields of its kind, below. Sides are "you" or "opponent".

Kinds:
- "piece": a piece stands on a square: "side", "piece" (king, queen, rook, bishop, knight, \
pawn), "square"
- "owns_move": a move is played by a side: "move", "side"
- "move_does": a move takes, attacks, defends, checks, pins or forks something: "move", \
"does" (takes | attacks | defends | checks | pins | forks), "target" (a square)
- "stops": a move stops or prevents another move: "move", "target_move"
- "allows": a move allows or lets another move happen right after it: "move", "target_move"
- "only_move": a move is the only move (or the only king move, or the only legal move): \
"move", "only" (legal | king)
- "forced": a check or move forces a piece to a square: "piece", "square"
- "mate": the move is checkmate: "move"
- "state": a piece is loose, undefended, hanging, defended, pinned, trapped or passed: \
"side", "piece", "square", "state" (loose | defended | pinned | trapped | passed)
- "file": a file is open, half-open or has no pawns: "file" (a-h), "state" (open | \
half-open | no pawns)
- "king_squares": the king has these free squares: "side", "squares" (list)

Only claims a board can settle. Skip advice, principles, praise, plans, evaluations, the \
course of the game and anything vague. Keep every move exactly as the reply writes it. \
Return [] when there is nothing to check.
"""

_FENCE = re.compile(r"```[A-Za-z]*\s*(.*?)```", re.DOTALL)
_PIECES = {
    "king": chess.KING,
    "queen": chess.QUEEN,
    "rook": chess.ROOK,
    "bishop": chess.BISHOP,
    "knight": chess.KNIGHT,
    "pawn": chess.PAWN,
}


@dataclass(frozen=True)
class DraftClaim:
    kind: str
    after: tuple[str, ...] = ()
    quote: str = ""
    fields: dict = field(default_factory=dict)


def extract(
    text: str,
    board: chess.Board,
    player: chess.Color,
    provider: LLMProvider,
    *,
    played: str | None = None,
    line: Sequence[str] = (),
    max_tokens: int = 1500,
) -> list[DraftClaim]:
    """The claims in `text`, as the model reads them; none when the call fails."""
    if not text.strip():
        return []
    side = "White" if player == chess.WHITE else "Black"
    parts = [
        f"FEN: {board.fen()}",
        f'The coach speaks to {side} ("you").',
        f"Move just played from this position: {played}" if played else "",
        f"Engine's best line from this position: {' '.join(line)}" if line else "",
        f'Reply:\n"""\n{text.strip()}\n"""',
    ]
    user = "\n".join(p for p in parts if p)
    try:
        completion = provider.chat(
            EXTRACT_SYSTEM,
            [{"role": "user", "content": user}],
            max_tokens=max_tokens,
            temperature=0.0,
        )
    except Exception:  # noqa: BLE001
        return []
    return parse(completion.text)


def parse(text: str) -> list[DraftClaim]:
    raw = text.strip()
    fenced = _FENCE.search(raw)
    if fenced:
        raw = fenced.group(1).strip()
    start, end = raw.find("["), raw.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        items = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return []
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("kind"), str):
            continue
        after = item.get("after") or []
        if not isinstance(after, list):
            after = []
        rest = {k: v for k, v in item.items() if k not in ("kind", "after", "quote")}
        out.append(
            DraftClaim(
                kind=item["kind"].strip().lower(),
                after=tuple(str(m) for m in after),
                quote=str(item.get("quote") or ""),
                fields=rest,
            )
        )
    return out


def _frame(board: chess.Board, after: Sequence[str]) -> chess.Board | None:
    """The position after `after` from `board`; None when a move of it cannot be played."""
    frame = board.copy(stack=False)
    for san in after:
        try:
            frame.push(frame.parse_san(str(san)))
        except ValueError:
            return None
    return frame


def _move(board: chess.Board, san: object) -> chess.Move | None:
    try:
        return board.parse_san(str(san))
    except ValueError:
        return None


def _side(value: object, player: chess.Color) -> chess.Color | None:
    word = str(value or "").strip().lower()
    if word in ("you", "your", "player"):
        return player
    if word in ("opponent", "they", "their", "them"):
        return not player
    return None


def _square(value: object) -> chess.Square | None:
    text = str(value or "").strip().lower()
    match = re.search(r"\b([a-h][1-8])\b", text)
    return chess.parse_square(match.group(1)) if match else None


def verify(
    claims: Sequence[DraftClaim],
    board: chess.Board,
    player: chess.Color,
    line: Sequence[str] = (),
    *,
    before: chess.Board | None = None,
) -> list[str]:
    """What the board contradicts, one reason per false claim. A claim that cannot be placed
    or decided says nothing, and a claim is decided only on what its own words say
    (`_grounded`): the extractor filled in squares, owners and "only" the reply never said,
    and 15 of 50 true replies were refused on the local model (2026-09-26). `before` is the
    position before the last move played: a claim about that move ("it takes your knight on
    d7") holds there. `line` is kept for the caller's record."""
    problems: list[str] = []
    for claim in claims:
        if not _grounded(claim):
            continue
        frame = _frame(board, claim.after)
        if frame is None:
            continue
        reason = _check(claim, frame, player)
        if reason is not None and before is not None:
            earlier = _frame(before, claim.after)
            if earlier is not None and _check(claim, earlier, player) is None:
                reason = None
        if reason:
            quote = f'"{claim.quote}"' if claim.quote else claim.kind
            problems.append(f"{quote}: {reason}")
    return list(dict.fromkeys(problems))


_OWNER = re.compile(r"\b(?:your|their|my|opponent'?s|white'?s|black'?s)\b", re.IGNORECASE)
_SQUARE = re.compile(r"\b[a-h][1-8]\b")
_ONLY = re.compile(
    r"\b(?:the|is the|was the) only (?:legal move|king move)\b|\bonly move out of check\b",
    re.IGNORECASE,
)
_ONLY_KING = re.compile(r"\bonly a king move\b", re.IGNORECASE)


def _grounded(claim: DraftClaim) -> bool:
    """Whether the claim's own words say what it is decided on: every square it names is in
    the quote, a state or a file's openness in so many words, "only" unqualified, an owner
    only when the quote names one. Anything else is the extractor's reading, not the reply's."""
    quote = straight(claim.quote)
    low = quote.lower()
    f = claim.fields
    squares = [f.get(k) for k in ("square", "target") if f.get(k)]
    if isinstance(f.get("squares"), list):
        squares += f["squares"]
    for value in squares:
        found = _SQUARE.search(str(value).lower())
        if found is None or found.group(0) not in low:
            return False
    if not _OWNER.search(quote):
        f.pop("side", None)
    kind = claim.kind
    if kind == "piece":
        return (
            re.search(r"\b(?:king|queen|rook|bishop|knight|pawn)s? on [a-h][1-8]\b", low)
            is not None
        )
    if kind == "state":
        state = str(f.get("state", "")).lower()
        words = {"loose": ("loose", "undefended", "unprotected", "hanging")}.get(state, (state,))
        return any(w in low for w in words)
    if kind == "only_move":
        if _ONLY_KING.search(quote):
            f["only"] = "king_forced"
            return True
        return _ONLY.search(quote) is not None and not re.search(
            r"only (?:legal |king )?move (?:that|which|to|for|keeping|preserving|holding)", low
        )
    if kind == "file":
        return "-file" in low and any(w in low for w in ("open", "no pawn"))
    if kind == "mate":
        return "mate" in low or "#" in quote
    if kind == "owns_move":
        return _OWNER.search(quote) is not None
    # "Before they could play c6": a "stops" claim has no plain contradiction to decide.
    return kind != "stops"


def straight(text: str) -> str:
    return text.replace("\u2019", "'").replace("\u2018", "'")


def _check(claim: DraftClaim, frame: chess.Board, player: chess.Color) -> str | None:  # noqa: C901, PLR0911, PLR0912
    f = claim.fields
    kind = claim.kind
    if kind == "piece":
        side, piece, square = (
            _side(f.get("side"), player),
            _PIECES.get(str(f.get("piece", "")).lower()),
            _square(f.get("square")),
        )
        if side is None or piece is None or square is None:
            return None
        there = frame.piece_at(square)
        if there is None or there.piece_type != piece or there.color != side:
            whose = "yours" if there is not None and there.color == player else "theirs"
            what = "empty" if there is None else f"{whose}: a {chess.piece_name(there.piece_type)}"
            return f"{chess.square_name(square)} is {what} then"
        return None
    if kind == "owns_move":
        side = _side(f.get("side"), player)
        if side is None:
            return None
        move = _move(frame, f.get("move"))
        if move is not None and frame.turn != side:
            return f"{f.get('move')} is not that side's move there"
        if move is None:
            flipped = frame.copy(stack=False)
            flipped.turn = not frame.turn
            if _move(flipped, f.get("move")) is not None and flipped.turn != side:
                return f"{f.get('move')} is the other side's move"
        return None
    if kind == "move_does":
        move = _move(frame, f.get("move"))
        target = _square(f.get("target"))
        does = str(f.get("does", "")).lower()
        if move is None or target is None:
            return None
        after = frame.copy(stack=False)
        after.push(move)
        if does == "takes":
            if move.to_square != target or not frame.is_capture(move):
                return f"{f.get('move')} does not take on {chess.square_name(target)}"
        elif does == "checks":
            if not frame.gives_check(move):
                return f"{f.get('move')} gives no check"
        elif does == "attacks":
            if target not in after.attacks(move.to_square):
                return (
                    f"after {f.get('move')} the piece does not attack {chess.square_name(target)}"
                )
        elif does == "defends" and target not in after.attacks(move.to_square):
            return f"after {f.get('move')} the piece does not cover {chess.square_name(target)}"
        return None
    if kind == "allows":
        move, reply = _move(frame, f.get("move")), f.get("target_move")
        if move is None or not reply:
            return None
        after = frame.copy(stack=False)
        after.push(move)
        if _move(after, reply) is None:
            return f"{reply} cannot be played after {f.get('move')}"
        return None
    if kind == "only_move":
        move = _move(frame, f.get("move"))
        if move is None:
            return None
        legal = list(frame.legal_moves)
        if f.get("only") == "king_forced":
            if legal and any(frame.piece_type_at(m.from_square) != chess.KING for m in legal):
                return "a move other than the king's gets out of check"
            return None
        if str(f.get("only", "")).lower() == "king":
            legal = [m for m in legal if frame.piece_type_at(m.from_square) == chess.KING]
        if len(legal) > 1:
            return f"{f.get('move')} is not the only move: {len(legal)} are legal"
        return None
    if kind == "forced":
        piece, square = _PIECES.get(str(f.get("piece", "")).lower()), _square(f.get("square"))
        if piece is None or square is None:
            return None
        legal = list(frame.legal_moves)
        if not legal:
            return None
        if not all(
            m.to_square == square and frame.piece_type_at(m.from_square) == piece for m in legal
        ):
            return f"nothing forces it to {chess.square_name(square)}: other moves are legal"
        return None
    if kind == "mate":
        move = _move(frame, f.get("move"))
        if move is None:
            return None
        after = frame.copy(stack=False)
        after.push(move)
        if not after.is_checkmate():
            return f"{f.get('move')} is not mate there"
        return None
    if kind == "state":
        side = _side(f.get("side"), player)
        square = _square(f.get("square"))
        state = str(f.get("state", "")).lower()
        if square is None:
            return None
        there = frame.piece_at(square)
        if there is None:
            return f"nothing stands on {chess.square_name(square)} then"
        if side is not None and there.color != side:
            whose = "yours" if there.color == player else "theirs"
            return f"the piece on {chess.square_name(square)} is {whose}"
        if state == "passed":
            if there.piece_type == chess.PAWN and not is_passed(frame, square):
                return f"the pawn on {chess.square_name(square)} is not passed"
            return None
        if state in ("loose", "undefended", "hanging", "pinned", "trapped"):
            # The draft check's own judgement, by legal moves (claims._tactic_problem).
            from sidecoach.coach.claims import _tactic_problem

            word = "loose" if state in ("undefended", "hanging") else state
            return _tactic_problem(frame, chess.square_name(square), word)
        return None
    if kind == "file":
        letter = str(f.get("file", "")).strip().lower()[:1]
        state = str(f.get("state", "")).lower().replace("-", " ")
        if letter not in "abcdefgh" or not letter:
            return None
        squares = [chess.square(ord(letter) - 97, r) for r in range(8)]
        pawns = [frame.piece_at(s) for s in squares if frame.piece_type_at(s) == chess.PAWN]
        if state in ("open", "no pawns") and pawns:
            return f"the {letter}-file has a pawn on it"
        return None
    if kind == "king_squares":
        side = _side(f.get("side"), player)
        squares = f.get("squares")
        if side is None or not isinstance(squares, list) or frame.turn != side:
            return None
        claimed = {s for s in (_square(x) for x in squares) if s is not None}
        free = {
            m.to_square
            for m in frame.legal_moves
            if frame.piece_type_at(m.from_square) == chess.KING
        }
        if claimed and claimed != free:
            said = ", ".join(sorted(chess.square_name(s) for s in free)) or "none"
            return f"the king's free squares are {said}"
        return None
    return None


def _stops(claim: DraftClaim, frame: chess.Board, line: Sequence[str]) -> str | None:
    """A move said to stop another is contradicted when the engine's line plays that other
    move straight after it ("Rae1 ... before they can use Re3" of Rae1 Re3: the test audit)."""
    move, reply = str(claim.fields.get("move") or "").rstrip("+#"), claim.fields.get("target_move")
    if not move or not reply:
        return None
    path = [m.rstrip("+#") for m in claim.after] + [move]
    if list(line[: len(path)]) != path or len(line) <= len(path):
        return None
    if line[len(path)] == str(reply).rstrip("+#"):
        return f"{reply} is still played straight after {claim.fields.get('move')}"
    return None
