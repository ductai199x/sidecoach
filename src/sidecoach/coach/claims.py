"""Claim verification, both directions. DESIGN.md section 5.5.

Direction one: what the user says in their own defence ("my knight is attacking e5 and the
pawn is hanging") is turned into structured claims by the LLM and every checkable one is
tested against the board before the coach agrees with anything. Only the extraction uses
the LLM. Resolving "my knight" to a square, counting attackers and defenders, trying the
null move and comparing with the engine's candidates all happen here, in python-chess,
where nothing can be made up. Attackers and defenders are counted by legal moves
(`analysis.exchange`): a piece pinned to its king takes nothing and takes nothing back, and
a king takes back only where it may.

Direction two: the coach's own draft is checked before it is shown. Every piece it places
on a square must be there, and every move it names must be playable, by the side to move
or as the opponent's threat after a pass.

Every reason is one sentence a coach could say, naming squares and pieces as they stand.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol

import chess

from sidecoach.analysis import exchange
from sidecoach.analysis.positional import is_passed
from sidecoach.coach.disclosure import PositionFacts
from sidecoach.coach.facts import CANDIDATE_MARGIN, PIECE_NAMES
from sidecoach.events import Line
from sidecoach.llm.base import ChatMessage, LLMProvider


class ClaimKind(StrEnum):
    ATTACKS = "attacks"  # piece X attacks square/piece Y
    DEFENDS = "defends"  # piece X defends square/piece Y
    THREATENS = "threatens"  # the opponent threatens Y (or "I threaten Y next move")
    HANGING = "hanging"  # piece on Y is undefended / can be taken for free
    SAFE = "safe"  # piece/square Y is safe / cannot be taken
    BEST = "best"  # move M is the best move / a good move
    PLAN = "plan"  # a strategic intention; not checkable, prompts a follow-up
    OTHER = "other"  # anything else; not checkable


Result = Literal["true", "false", "unclear"]


@dataclass(frozen=True)
class Claim:
    """One checkable thing the user said, in their own words.

    `subject` and `target` are kept as the user named them: a square ("f3") or a piece
    description ("my knight", "the pawn on e5"). `move` is SAN: the move judged for BEST,
    the threatened move for THREATENS when one was named, and for the board claims the move
    the user is weighing when the claim is about the position after it ("Nf3 defends d4").
    """

    kind: ClaimKind
    subject: str | None
    target: str | None
    move: str | None
    text: str


@dataclass(frozen=True)
class Verdict:
    claim: Claim
    result: Result
    reason: str


@dataclass(frozen=True)
class DraftProblem:
    text: str  # the offending phrase
    reason: str  # what the board says instead


class EngineLike(Protocol):
    """What `verify_claims` needs from an engine; `analysis.engine.Engine` satisfies it."""

    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]: ...


# --- Direction one: extraction --------------------------------------------------------------

EXTRACT_SYSTEM = """\
You turn what a chess student says about the position in front of them into a JSON array \
of claims, so a coach can check each one against the board before agreeing.

Reply with the JSON array only: no prose, no code fences, no keys other than these. Each \
element is an object:
- "kind": one of "attacks", "defends", "threatens", "hanging", "safe", "best", "plan", "other"
- "subject": the actor, in the student's words ("my knight", "the bishop", "f3", "I", \
"they"); null when the kind has no actor
- "target": what it acts on, in the student's words ("e5", "the pawn", "their queen", \
"my king"); null when there is none
- "move": a move in standard algebraic notation, or null
- "text": the clause the claim came from, copied verbatim

Kinds, with where the words go:
- attacks: a piece attacks a square or a piece ("my knight hits e5"): subject the piece, \
target the square or piece
- defends: a piece defends a square or a piece ("the bishop covers d4"): same as attacks
- threatens: the opponent threatens something ("they are threatening my rook") or the \
student does ("I threaten mate on f7"): subject "they" or "I", target the thing threatened, \
move the threatened move if one was named
- hanging: a piece is undefended or can be taken for free ("the pawn is hanging"): subject \
null, target the piece
- safe: a piece or square cannot be taken ("my knight is safe there"): subject null, target \
the piece or square
- best: a move is the best move or a good move ("Nf3 is best", "I want to play Bb5"): \
subject null, target null, move the move
- plan: a strategic intention that no single move settles ("push the h-pawn"): subject and \
target null
- other: anything else about the game: subject and target null

For attacks, defends, hanging and safe, "move" is the move the student is weighing when the \
claim is about the position after that move ("Nf3 defends d4"), else null.

Keep the student's own words; do not turn "my knight" into a square, and do not check the \
claims yourself. One sentence can hold several claims. The student is the side to move: "my" \
and "I" are the student, "your" and "their" the opponent. Return [] when nothing said is \
about the position. Decide quickly; the closest kind is good enough.
"""

_FENCE = re.compile(r"```[A-Za-z]*\s*(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_claims(
    text: str,
    board: chess.Board,
    provider: LLMProvider,
    *,
    thinking: bool = True,
    max_tokens: int = 4000,
) -> list[Claim]:
    """The claims in `text`, as the LLM reads them. Never raises; a failed call yields none.

    Measured 2026-09-15 on the local Qwen 3.8 27B: with thinking off the three claims in a
    typical sentence come back in about 5 s; with thinking on the model reasons for some
    3000 tokens first, about 50 s, and a smaller budget is eaten whole by the reasoning and
    returns no text at all. When that happens the call is retried once with thinking off,
    so a runaway costs time but never the claims.
    """
    if not text.strip():
        return []
    side = "white" if board.turn == chess.WHITE else "black"
    user = f"Side to move: {side} (the student).\nFEN: {board.fen()}\n\nStudent: {text.strip()}"
    messages: list[ChatMessage] = [{"role": "user", "content": user}]
    try:
        completion = provider.chat(
            EXTRACT_SYSTEM, messages, max_tokens=max_tokens, temperature=0.1, thinking=thinking
        )
        if thinking and not completion.text.strip():
            completion = provider.chat(
                EXTRACT_SYSTEM, messages, max_tokens=max_tokens, temperature=0.1, thinking=False
            )
    except Exception:  # noqa: BLE001
        return []
    return parse_claims(completion.text)


def parse_claims(text: str) -> list[Claim]:
    """Claims from a model reply: a JSON array, fenced or not, or a single object.

    Entries with an unknown kind are dropped. Anything unreadable yields nothing.
    """
    cleaned = _THINK.sub("", text).strip()
    fenced = _FENCE.search(cleaned)
    if fenced:
        cleaned = fenced.group(1).strip()
    data = _load_json(cleaned)
    if isinstance(data, dict):
        inner = data.get("claims")
        data = inner if isinstance(inner, list) else [data]
    if not isinstance(data, list):
        return []
    claims: list[Claim] = []
    for item in data:
        claim = _claim_from(item)
        if claim is not None:
            claims.append(claim)
    return claims


def _load_json(text: str) -> object:
    try:
        return json.loads(text)
    except ValueError:
        pass
    for opener, closer in (("[", "]"), ("{", "}")):
        start, end = text.find(opener), text.rfind(closer)
        if 0 <= start < end:
            try:
                return json.loads(text[start : end + 1])
            except ValueError:
                continue
    return None


def _claim_from(item: object) -> Claim | None:
    if not isinstance(item, dict):
        return None
    kind_raw = item.get("kind")
    if not isinstance(kind_raw, str):
        return None
    try:
        kind = ClaimKind(kind_raw.strip().lower())
    except ValueError:
        return None
    return Claim(
        kind=kind,
        subject=_text_or_none(item.get("subject")),
        target=_text_or_none(item.get("target")),
        move=_text_or_none(item.get("move")),
        text=str(item.get("text") or "").strip(),
    )


def _text_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or stripped.lower() in ("null", "none"):
        return None
    return stripped


# --- Direction one: verification ------------------------------------------------------------

_SQUARE_IN_WORDS = re.compile(r"(?<![a-z0-9])([a-h][1-8])(?![a-z0-9])")
_BARE_SQUARE = re.compile(r"[a-h][1-8]")
_WORDS = re.compile(r"[a-z']+")
_PIECE_WORDS: dict[str, chess.PieceType] = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "horse": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
    "king": chess.KING,
}
# The user speaks from their own side: "my" is the side to move, "your" and "their" the
# opponent. "white" and "black" are absolute.
_OURS = frozenset({"my", "mine", "i", "me", "our", "ours", "us"})
_THEIRS = frozenset(
    {"your", "yours", "their", "theirs", "they", "them", "opponent", "opponent's", "his", "her"}
)
_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}
_TRAILING_MARKS = re.compile(r"[+#!?]+$")


def verify_claims(
    claims: Sequence[Claim],
    board: chess.Board,
    *,
    facts: PositionFacts | None = None,
    engine: EngineLike | None = None,
    depth: int = 12,
) -> list[Verdict]:
    """A verdict for every claim, checked on `board`, whose side to move is the user.

    `facts` supplies the engine's candidates and threat line when they belong to this
    position; otherwise BEST falls back to `engine`, and to "unclear" without one.
    """
    if facts is not None and not _same_position(facts.fen, board):
        facts = None
    verdicts: list[Verdict] = []
    for claim in claims:
        result, reason = _verify(claim, board, facts, engine, depth)
        verdicts.append(Verdict(claim=claim, result=result, reason=reason))
    return verdicts


def _same_position(fen: str, board: chess.Board) -> bool:
    return fen.split(" ")[:3] == board.fen().split(" ")[:3]


@dataclass(frozen=True)
class _Position:
    """The board a claim is checked on, and who the user is on it."""

    board: chess.Board
    us: chess.Color
    played: chess.Move | None = None  # a hypothetical move already pushed on `board`
    played_san: str | None = None

    @property
    def them(self) -> chess.Color:
        return not self.us

    def side(self, color: chess.Color) -> str:
        return "your" if color == self.us else "their"

    def sides(self, color: chess.Color) -> str:
        return "yours" if color == self.us else "theirs"

    def name(self, square: chess.Square) -> str:
        """ "your knight on f3", or just "e5" for an empty square."""
        piece = self.board.piece_at(square)
        if piece is None:
            return chess.square_name(square)
        where = chess.square_name(square)
        return f"{self.side(piece.color)} {PIECE_NAMES[piece.piece_type]} on {where}"

    def short(self, square: chess.Square) -> str:
        """ "the f3 knight", or the square when empty."""
        piece = self.board.piece_at(square)
        if piece is None:
            return chess.square_name(square)
        return f"the {chess.square_name(square)} {PIECE_NAMES[piece.piece_type]}"

    def listed(self, squares: Iterable[chess.Square]) -> str:
        return _join([self.short(square) for square in sorted(squares)])

    def group(self, squares: Sequence[chess.Square]) -> str:
        """ "your knights on b1 and f3", for a description that fit several pieces."""
        pieces = [self.board.piece_at(square) for square in squares]
        first = pieces[0]
        if first is None or any(p is None or p != first for p in pieces):
            return "those pieces"
        where = _join([chess.square_name(square) for square in sorted(squares)])
        return f"{self.side(first.color)} {PIECE_NAMES[first.piece_type]}s on {where}"


def _join(items: Sequence[str]) -> str:
    if not items:
        return "nothing"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _times(count: int) -> str:
    return {0: "not at all", 1: "once", 2: "twice", 3: "three times"}.get(count, f"{count} times")


def _resolve(
    desc: str | None, board: chess.Board, us: chess.Color, prefer: chess.Color | None
) -> list[chess.Square]:
    """The squares a description could mean: "f3", "the pawn on e5", "my knight", "they".

    A square in the words settles it. Otherwise a colour word picks the side ("my" is `us`),
    `prefer` breaks the tie when none was said, and the piece word picks the pieces; a side
    with no piece word ("I", "they") means every piece of that side. Several squares mean
    the description was ambiguous and the caller tries each.
    """
    if not desc:
        return []
    lowered = desc.lower()
    square = _SQUARE_IN_WORDS.search(lowered)
    if square:
        return [chess.parse_square(square.group(1))]
    color: chess.Color | None = None
    piece_type: chess.PieceType | None = None
    for word in _WORDS.findall(lowered):
        if word in _OURS:
            color = us
        elif word in _THEIRS:
            color = not us
        elif word in ("white", "white's"):
            color = chess.WHITE
        elif word in ("black", "black's"):
            color = chess.BLACK
        else:
            piece_type = _PIECE_WORDS.get(word.rstrip("s"), piece_type)
    if color is None and piece_type is None:
        return []

    def of(side: chess.Color) -> list[chess.Square]:
        if piece_type is None:
            return list(chess.SquareSet(board.occupied_co[side]))
        return list(board.pieces(piece_type, side))

    if color is not None:
        return sorted(of(color))
    if prefer is None:
        return sorted(of(chess.WHITE) + of(chess.BLACK))
    return sorted(of(prefer) or of(not prefer))


def _said_ours(desc: str | None) -> bool:
    return bool(desc) and any(word in _OURS for word in _WORDS.findall(desc.lower()))


def _parse_move(board: chess.Board, text: str) -> tuple[chess.Move | None, str]:
    """A legal move from SAN or UCI, or None with why: "ambiguous", "illegal", "unreadable"."""
    cleaned = _TRAILING_MARKS.sub("", text.strip())
    try:
        return board.parse_san(cleaned), ""
    except chess.AmbiguousMoveError:
        return None, "ambiguous"
    except chess.IllegalMoveError:
        return None, "illegal"
    except chess.InvalidMoveError:
        pass
    try:
        move = chess.Move.from_uci(cleaned.lower())
    except chess.InvalidMoveError:
        return None, "unreadable"
    return (move, "") if move in board.legal_moves else (None, "illegal")


def _position(claim: Claim, board: chess.Board) -> _Position:
    """The board to check a board claim on: after the move the user is weighing, if any."""
    us = board.turn
    move: chess.Move | None = None
    if claim.move:
        move, _ = _parse_move(board, claim.move)
    if move is None and claim.subject and not _BARE_SQUARE.fullmatch(claim.subject.strip()):
        move, _ = _parse_move(board, claim.subject)
    if move is None:
        return _Position(board, us)
    after = board.copy(stack=False)
    san = board.san(move)
    after.push(move)
    return _Position(after, us, move, san)


def _verify(
    claim: Claim,
    board: chess.Board,
    facts: PositionFacts | None,
    engine: EngineLike | None,
    depth: int,
) -> tuple[Result, str]:
    match claim.kind:
        case ClaimKind.ATTACKS:
            return _check_attacks(_position(claim, board), claim)
        case ClaimKind.DEFENDS:
            return _check_defends(_position(claim, board), claim)
        case ClaimKind.HANGING:
            return _check_hanging(_position(claim, board), claim)
        case ClaimKind.SAFE:
            return _check_safe(_position(claim, board), claim)
        case ClaimKind.THREATENS:
            return _check_threatens(_Position(board, board.turn), claim, facts)
        case ClaimKind.BEST:
            return _check_best(board, claim, facts, engine, depth)
        case ClaimKind.PLAN:
            return (
                "unclear",
                "That is a plan rather than something I can check on the board; what do you "
                "expect it to achieve, and which move starts it?",
            )
        case ClaimKind.OTHER:
            return (
                "unclear",
                "I cannot check that against the board; what do you expect to happen, concretely?",
            )


def _subjects(pos: _Position, claim: Claim, prefer: chess.Color) -> list[chess.Square]:
    """Where the actor stands; the piece just moved when the subject was the move itself."""
    found = _resolve(claim.subject, pos.board, pos.us, prefer)
    if not found and pos.played is not None:
        return [pos.played.to_square]
    return found


def _unclear_subject(claim: Claim) -> tuple[Result, str]:
    if claim.subject:
        return "unclear", f"I cannot tell which piece you mean by '{claim.subject}'."
    return "unclear", "Which piece do you mean?"


def _unclear_target(claim: Claim) -> tuple[Result, str]:
    if claim.target:
        return "unclear", f"I cannot tell what you mean by '{claim.target}'."
    return "unclear", "Which square or piece do you mean?"


def _after_move(pos: _Position, reason: str) -> str:
    """Prefix a reason with the hypothetical move it was checked after, if there was one."""
    if pos.played_san is None:
        return reason
    return f"after {pos.played_san}, {reason}"


def _to_move(board: chess.Board, color: chess.Color) -> chess.Board:
    """`board` with `color` to move: the same position, handed over by a pass if need be."""
    if board.turn == color:
        return board
    view = board.copy(stack=False)
    view.push(chess.Move.null())
    return view


def _takes_back(board: chess.Board, capture: chess.Move) -> list[chess.Square]:
    """The pieces that can legally take back after `capture`, a move of the side to move."""
    after = board.copy(stack=False)
    after.push(capture)
    onto = after.generate_legal_moves(chess.BB_ALL, chess.BB_SQUARES[capture.to_square])
    return sorted({m.from_square for m in onto})


def _captures_of(
    board: chess.Board, square: chess.Square
) -> dict[chess.Square, list[chess.Square]]:
    """Each enemy piece that can legally take the piece on `square`, with the pieces that can
    legally take back once it has. Read after the capture itself, so the e8 king takes back
    after Bxf7+ but not after Qxf7 with the bishop still on c4."""
    piece = board.piece_at(square)
    if piece is None:
        return {}
    view = _to_move(board, not piece.color)
    found: dict[chess.Square, set[chess.Square]] = {}
    for capture in exchange.captures_on(board, not piece.color, square):
        found.setdefault(capture.from_square, set()).update(_takes_back(view, capture))
    return {attacker: sorted(backs) for attacker, backs in sorted(found.items())}


def _guards(board: chess.Board, square: chess.Square) -> list[chess.Square]:
    """The pieces that can legally take back for the piece on `square`: after any capture of
    it, or, when nothing can take it, were something to (`exchange.defenders`)."""
    piece = board.piece_at(square)
    if piece is None:
        return []
    captures = _captures_of(board, square)
    if not captures:
        return exchange.defenders(board, piece.color, square)
    return sorted({guard for backs in captures.values() for guard in backs})


def _idle(
    board: chess.Board, color: chess.Color, square: chess.Square, active: Iterable[chess.Square]
) -> list[chess.Square]:
    """The pieces of `color` that reach `square` but are not among `active`, the ones that
    can legally move there."""
    moving = set(active)
    return [reach for reach in board.attackers(color, square) if reach not in moving]


def _cannot(
    board: chess.Board, color: chess.Color, squares: Sequence[chess.Square], action: str
) -> str:
    """ "is pinned and cannot take it": why pieces of `color` that reach a square cannot move
    there, said as a pin when every one of them is pinned to its king."""
    if squares and all(board.is_pinned(color, square) for square in squares):
        return f"{'is' if len(squares) == 1 else 'are'} pinned and cannot {action}"
    return f"cannot {action}"


def _joins_in(
    pos: _Position, color: chess.Color, square: chess.Square, fronts: Iterable[chess.Square]
) -> str | None:
    """ "the d8 queen behind the d7 rook joins in": a piece of `color` lined up behind one of
    `fronts` that reaches `square` once the front one has taken there, an x-ray."""
    board = pos.board
    reach = board.attackers(color, square)
    for front in fronts:
        probe = board.copy(stack=False)
        probe.remove_piece_at(front)
        for backer in probe.attackers(color, square):
            if backer not in reach:
                return f"{pos.short(backer)} behind {pos.short(front)} joins in"
    return None


def _check_attacks(pos: _Position, claim: Claim) -> tuple[Result, str]:
    board = pos.board
    subjects = _subjects(pos, claim, pos.us)
    if not subjects:
        return _unclear_subject(claim)
    targets = _resolve(claim.target, board, pos.us, pos.them)
    if not targets:
        return _unclear_target(claim)
    occupied = [square for square in subjects if board.piece_at(square) is not None]
    if not occupied:
        return "false", f"There is nothing on {chess.square_name(subjects[0])}."
    for subject in occupied:
        for target in targets:
            if target not in board.attacks(subject):
                continue
            reason = f"{pos.name(subject)} attacks {pos.name(target)}"
            hit = board.piece_at(target)
            if (
                hit is not None
                and hit.color != board.color_at(subject)
                and hit.piece_type != chess.KING
            ):
                # Attacking is the geometry; taking is a legal move. A pinned piece still
                # attacks, but it cannot take, and then who defends does not arise.
                mover = not hit.color
                own = [
                    m
                    for m in exchange.captures_on(board, mover, target)
                    if m.from_square == subject
                ]
                if not own:
                    reason += f", but it {_cannot(board, mover, [subject], 'take it')}"
                else:
                    guards = _takes_back(_to_move(board, mover), own[0])
                    idle = _idle(board, hit.color, target, guards)
                    if guards:
                        reason += f", though it is defended by {pos.listed(guards)}"
                    elif idle:
                        cannot = _cannot(board, hit.color, idle, "take back")
                        reason += f", and {pos.listed(idle)} {cannot}"
                    else:
                        reason += ", and nothing defends it"
            return "true", _sentence(_after_move(pos, reason))
    wanted = _join([pos.name(target) for target in targets[:3]])
    if len(occupied) == 1:
        subject = occupied[0]
        hits = [
            square
            for square in board.attacks(subject)
            if board.color_at(square) not in (None, board.color_at(subject))
        ]
        what = f"it attacks {pos.listed(hits)}" if hits else "it attacks none of their pieces"
        reason = f"{pos.name(subject)} does not attack {wanted}; {what}"
    else:
        reason = f"none of {pos.group(occupied)} attacks {wanted}"
    ambiguous = len(occupied) > 1 or len(targets) > 1
    return ("unclear" if ambiguous else "false"), _sentence(_after_move(pos, reason))


def _check_defends(pos: _Position, claim: Claim) -> tuple[Result, str]:
    board = pos.board
    subjects = _subjects(pos, claim, pos.us)
    if not subjects:
        return _unclear_subject(claim)
    targets = _resolve(claim.target, board, pos.us, pos.us)
    if not targets:
        return _unclear_target(claim)
    occupied = [square for square in subjects if board.piece_at(square) is not None]
    if not occupied:
        return "false", f"There is nothing on {chess.square_name(subjects[0])}."
    for subject in occupied:
        for target in targets:
            if target not in board.attacks(subject):
                continue
            held = board.piece_at(target)
            if held is None:
                reason = f"{pos.name(subject)} covers {chess.square_name(target)}, which is empty"
                return "true", _sentence(_after_move(pos, reason))
            if held.color == board.color_at(subject):
                # Defending is taking back: a piece pinned to its king reaches the square and
                # still cannot.
                guards = _guards(board, target)
                if subject not in guards:
                    square = chess.square_name(target)
                    cannot = _cannot(board, held.color, [subject], f"take back on {square}")
                    return "false", _sentence(_after_move(pos, f"{pos.name(subject)} {cannot}"))
                others = [sq for sq in guards if sq != subject]
                reason = f"{pos.name(subject)} defends {pos.name(target)}"
                if others:
                    reason += f", together with {pos.listed(others)}"
                return "true", _sentence(_after_move(pos, reason))
            reason = f"{pos.name(target)} is not yours to defend; {pos.name(subject)} attacks it"
            return "false", _sentence(_after_move(pos, reason))
    wanted = _join([pos.name(target) for target in targets[:3]])
    if len(occupied) == 1:
        reason = f"{pos.name(occupied[0])} does not reach {wanted}"
    else:
        reason = f"none of {pos.group(occupied)} reaches {wanted}"
    ambiguous = len(occupied) > 1 or len(targets) > 1
    return ("unclear" if ambiguous else "false"), _sentence(_after_move(pos, reason))


def _check_hanging(pos: _Position, claim: Claim) -> tuple[Result, str]:
    board = pos.board
    targets = _resolve(claim.target or claim.subject, board, pos.us, None)
    if not targets:
        return _unclear_target(claim)
    fallback: str | None = None
    for target in targets:
        piece = board.piece_at(target)
        if piece is None:
            fallback = fallback or f"there is nothing on {chess.square_name(target)}"
            continue
        if piece.piece_type == chess.KING:
            fallback = fallback or f"{pos.name(target)} cannot hang; it can only be in check"
            continue
        captures = _captures_of(board, target)
        attackers = list(captures)
        if exchange.hanging(board, target):
            guards = _guards(board, target)
            idle = _idle(board, piece.color, target, guards)
            reason = f"{pos.name(target)} is attacked by {pos.listed(attackers)}"
            if guards:
                # Taken back after one capture, but not after another.
                free = next((a for a, backs in captures.items() if not backs), attackers[0])
                reason += f", and after {pos.short(free)} takes, nothing can take back"
            elif idle:
                cannot = _cannot(board, piece.color, idle, "take back")
                reason += f", and {pos.listed(idle)} {cannot}"
            else:
                reason += " and nothing defends it"
            return "true", _sentence(_after_move(pos, reason))
        if not attackers:
            idle = _idle(board, not piece.color, target, [])
            if idle:
                only, cannot = pos.listed(idle), _cannot(board, not piece.color, idle, "take it")
                reason = f"{pos.name(target)} is attacked only by {only}, which {cannot}"
            else:
                reason = f"{pos.name(target)} is not attacked at all"
        else:
            reason = (
                f"{pos.name(target)} is attacked by {pos.listed(attackers)} but defended by "
                f"{pos.listed(_guards(board, target))}, so it is not hanging outright"
            )
            where = chess.square_name(target)
            cheapest = min(attackers, key=lambda sq: _VALUE[board.piece_type_at(sq) or 0])
            if _VALUE[board.piece_type_at(cheapest) or 0] < _VALUE[piece.piece_type]:
                reason += (
                    f", though {pos.short(cheapest)} taking a {PIECE_NAMES[piece.piece_type]} "
                    "still wins material"
                )
            elif exchange.can_be_won(board, target):
                reason += f", though the exchange on {where} still wins it"
                joined = _joins_in(pos, not piece.color, target, attackers)
                if joined:
                    reason += f": {joined}"
            else:
                reason += f", and the exchange on {where} does not win it"
        # An attacked piece is the interesting one to explain when several fit the words.
        if fallback is None or attackers:
            fallback = reason
            if attackers:
                break
    assert fallback is not None
    if len(targets) > 1:
        fallback = f"none of {pos.group(targets)} is hanging: {fallback}"
    return ("unclear" if len(targets) > 1 else "false"), _sentence(_after_move(pos, fallback))


def _check_safe(pos: _Position, claim: Claim) -> tuple[Result, str]:
    board = pos.board
    targets = _resolve(claim.target or claim.subject, board, pos.us, pos.us)
    if not targets:
        return _unclear_target(claim)
    checked = [(target, *_safety(pos, target)) for target in targets]
    held = [reason for _, safe, reason in checked if safe]
    failed = [reason for _, safe, reason in checked if not safe]
    if held:
        # "Safe" is a universal claim, so a piece the words also fit that is not safe is
        # worth hearing about even when the claim holds for another.
        reason = held[0] + (f", though {failed[0]}" if failed else "")
        return "true", _sentence(_after_move(pos, reason))
    reason = failed[0]
    if len(targets) > 1:
        reason = f"none of {pos.group(targets)} is safe: {reason}"
    return ("unclear" if len(targets) > 1 else "false"), _sentence(_after_move(pos, reason))


def _safety(pos: _Position, target: chess.Square) -> tuple[bool, str]:
    """Whether the piece on `target` can be taken for profit, or a piece of yours on the
    empty square taken at all, with the counts. Only legal captures count, and the verdict is
    the exchange played out by legal moves (`exchange.can_be_won`), x-rays included; a
    king's safety is whether it is in check."""
    board = pos.board
    square = chess.square_name(target)
    piece = board.piece_at(target)
    if piece is None:
        # Who of theirs could take a piece of yours standing there.
        takers = exchange.defenders(board, pos.them, target)
        if takers:
            return False, f"{square} is attacked by {pos.listed(takers)}"
        idle = _idle(board, pos.them, target, [])
        if idle:
            cannot = _cannot(board, pos.them, idle, "take there")
            return True, f"{square} is attacked only by {pos.listed(idle)}, which {cannot}"
        return True, f"{square} is not attacked by any of their pieces"
    if piece.piece_type == chess.KING:
        checkers = board.attackers(not piece.color, target)
        if not checkers:
            return True, f"{pos.name(target)} is not in check"
        return False, f"{pos.name(target)} is in check from {pos.listed(checkers)}"
    attackers = list(_captures_of(board, target))
    if not attackers:
        idle = _idle(board, not piece.color, target, [])
        if idle:
            only, cannot = pos.listed(idle), _cannot(board, not piece.color, idle, "take it")
            return True, f"{pos.name(target)} is attacked only by {only}, which {cannot}"
        return True, f"{pos.name(target)} is not attacked"
    guards = _guards(board, target)
    counts = (
        f"{pos.name(target)} is attacked {_times(len(attackers))} ({pos.listed(attackers)}) "
        f"and defended {_times(len(guards))}"
    )
    if guards:
        counts += f" ({pos.listed(guards)})"
    if not exchange.can_be_won(board, target):
        return True, f"{counts}, so it holds"
    cheapest = min(attackers, key=lambda sq: _VALUE[board.piece_type_at(sq) or 0])
    if _VALUE[board.piece_type_at(cheapest) or 0] < _VALUE[piece.piece_type]:
        reason = (
            f"{counts}, but {pos.short(cheapest)} is worth less than a "
            f"{PIECE_NAMES[piece.piece_type]}, so it can still be taken profitably"
        )
    elif len(guards) < len(attackers):
        reason = f"{counts}, so it can be taken"
    else:
        # As many defenders, and still lost: a piece lined up behind an attacker joins in.
        reason = f"{counts}, but the exchange on {square} still wins it"
        joined = _joins_in(pos, not piece.color, target, attackers)
        if joined:
            reason += f": {joined}"
    idle = _idle(board, piece.color, target, guards)
    if idle:
        reason += f"; {pos.listed(idle)} {_cannot(board, piece.color, idle, 'take back')}"
    return False, reason


def _check_threatens(
    pos: _Position, claim: Claim, facts: PositionFacts | None
) -> tuple[Result, str]:
    board = pos.board
    threatener = pos.us if _said_ours(claim.subject) else pos.them
    victim = not threatener
    position = board
    if position.turn != threatener:
        if position.is_check():
            return "unclear", _sentence(
                f"{pos.side(position.turn)} king is in check, so the first question is how "
                "to meet that"
            )
        position = board.copy(stack=False)
        position.push(chess.Move.null())
    lead = "if you pass, " if threatener == pos.them else ""

    if claim.move:
        move, why = _parse_move(position, claim.move)
        if move is None:
            if why == "unreadable":
                return "unclear", f"I cannot read '{claim.move}' as a move."
            whose = "you" if threatener == pos.us else "they"
            return "false", _sentence(f"{claim.move} is not a move {whose} can play next")
        return "true", _sentence(lead + _describe_threat(pos, position, move))

    targets = _resolve(claim.target, board, pos.us, victim)
    if not targets:
        return _unclear_target(claim)

    engine_threats = _engine_threats(position, facts) if threatener == pos.them else []
    fallback: str | None = None
    for target in targets:
        piece = board.piece_at(target)
        if piece is None:
            fallback = fallback or (
                f"{chess.square_name(target)} is empty; what do you expect to happen there?"
            )
            continue
        if piece.color == threatener:
            fallback = fallback or f"{pos.name(target)} is {pos.sides(threatener)}, not a target"
            continue
        for move in engine_threats:
            if move.to_square == target or (
                piece.piece_type == chess.KING and position.gives_check(move)
            ):
                reason = f"the engine's main threat is {position.san(move)}: "
                reason += _describe_threat(pos, position, move)
                return "true", _sentence(lead + reason)
        if piece.piece_type == chess.KING:
            checks = [m for m in position.legal_moves if position.gives_check(m)]
            if checks:
                return "true", _sentence(lead + _describe_threat(pos, position, checks[0]))
            fallback = fallback or f"no {pos.side(threatener)} piece can give check next move"
            continue
        captures = [m for m in position.legal_moves if m.to_square == target]
        if captures:
            return "true", _sentence(lead + _describe_threat(pos, position, captures[0]))
        reason = f"nothing of {pos.sides(threatener)} can take {pos.name(target)} next move"
        if engine_threats:
            reason += f"; the engine's threat is {position.san(engine_threats[0])}"
        fallback = fallback or reason
    assert fallback is not None
    if len(targets) > 1:
        fallback = f"none of {pos.group(targets)} is threatened: {fallback}"
    return ("unclear" if len(targets) > 1 else "false"), _sentence(fallback)


def _engine_threats(position: chess.Board, facts: PositionFacts | None) -> list[chess.Move]:
    """The engine's threat line as moves in `position`, which is the board after a pass."""
    if facts is None or not facts.threats:
        return []
    moves: list[chess.Move] = []
    for san in facts.threats:
        move, _ = _parse_move(position, san)
        if move is not None:
            moves.append(move)
    return moves


def _describe_threat(pos: _Position, position: chess.Board, move: chess.Move) -> str:
    """ "their bishop on c4 takes your pawn on f7, and your king on e8 can take back".

    Who can take back is read by legal moves after the capture itself: after Qxf7# nothing
    can, and after Bxf7+ with the queen still on h5 the e8 king cannot either.
    """
    actor = pos.name(move.from_square)
    victim = position.piece_at(move.to_square)
    san = position.san(move)
    if victim is None:
        what = f"{actor} can play {san}"
        if position.gives_check(move):
            what += ", with check"
        return what
    what = f"{actor} takes {pos.name(move.to_square)} with {san}"
    recapturers = _takes_back(position, move)
    if recapturers:
        return f"{what}, and {pos.listed(recapturers)} can take back"
    if san.endswith("#"):
        return f"{what}, and it is checkmate"
    return f"{what} for free"


def _check_best(
    board: chess.Board,
    claim: Claim,
    facts: PositionFacts | None,
    engine: EngineLike | None,
    depth: int,
) -> tuple[Result, str]:
    san = claim.move or claim.subject
    if not san:
        return "unclear", "Which move do you mean?"
    move, why = _parse_move(board, san)
    if move is None:
        if why == "ambiguous":
            return "unclear", f"'{san}' could be more than one move here; which piece?"
        if why == "illegal":
            return "false", f"{san} is not a legal move in this position."
        return "unclear", f"I cannot read '{san}' as a move."
    said = board.san(move)

    candidates = _candidates(board, facts, engine, depth)
    if candidates is None:
        return "unclear", f"I have no engine view of this position yet, so I cannot rate {said}."
    if not candidates:
        return (
            "unclear",
            f"The engine has nothing to say about this position, so {said} is unrated.",
        )
    if move in candidates:
        shown = _join([board.san(m) for m in candidates])
        if len(candidates) == 1:
            return "true", f"{said} is the engine's best move here."
        return "true", f"{said} is among the engine's best moves here ({shown})."
    preferred = _join([board.san(m) for m in candidates])
    return "false", f"{said} is not among the engine's best moves here; it prefers {preferred}."


def _candidates(
    board: chess.Board,
    facts: PositionFacts | None,
    engine: EngineLike | None,
    depth: int,
) -> list[chess.Move] | None:
    """The near-equal best moves, from the facts when they have them, else the engine."""
    if facts is not None and facts.candidates:
        moves: list[chess.Move] = []
        for san in facts.candidates:
            move, _ = _parse_move(board, san)
            if move is not None:
                moves.append(move)
        if moves:
            return moves
    if engine is None:
        return None
    lines = engine.analyse(board, depth=depth)
    lines = [line for line in lines if line.moves]
    if not lines:
        return []
    top = max(line.expected_score for line in lines)
    moves = []
    for line in lines:
        if top - line.expected_score > CANDIDATE_MARGIN:
            continue
        move, _ = _parse_move(board, line.moves[0])
        if move is not None and move not in moves:
            moves.append(move)
    return moves


def _sentence(reason: str) -> str:
    """Capitalised and full-stopped, except that a square name stays as it is read."""
    reason = reason.strip()
    if not reason:
        return reason
    if not _BARE_SQUARE.match(reason):
        reason = reason[0].upper() + reason[1:]
    return reason if reason[-1] in ".?!" else reason + "."


# --- Direction two: the coach's draft --------------------------------------------------------

_COLOUR_WORDS = r"your|their|the opponent's|opponent's|white's|black's|white|black"
_PIECE_WORD = r"king|queen|rook|bishop|knight|pawn"
# Words a coach puts between the owner and the piece: "your isolated pawn on g2".
_PIECE_ADJECTIVES = (
    r"(?:(?:isolated|passed|backward|doubled|loose|hanging|pinned|trapped|advanced|extra|"
    r"lone|last|remaining|undefended|defended|attacked|active|passive|bad|good|"
    r"light-squared|dark-squared|central|strong|weak|exposed|stranded|"
    r"overloaded)\s+){0,2}"
)
# A pawn may carry its file: "your isolated f-pawn on f2" (the AI review, 2026-09-30).
_PIECE_ON = re.compile(
    rf"\b(?:(?P<colour>{_COLOUR_WORDS})\s+)?{_PIECE_ADJECTIVES}(?:[a-h]-)?(?P<piece>{_PIECE_WORD})s?"
    r"\s+(?:on|at)\s+"
    r"(?P<square>[a-h][1-8])\b",
    re.IGNORECASE,
)
_SQUARE_PIECE = re.compile(
    rf"\b(?:(?P<colour>{_COLOUR_WORDS})\s+)?(?P<square>[a-h][1-8])[- ](?P<piece>{_PIECE_WORD})\b",
    re.IGNORECASE,
)
# A token is a bare square ("f7") or a SAN move ending in its destination square ("Ng5",
# "Nxf7+", "exd5", "R1e1", "e8=Q"); the same idea as `disclosure.squares_in`. Group 1 is
# the destination square. Splitting on everything else gives token boundaries for free, so
# "area1" or "Be careful" never match.
_SAN_TOKEN = re.compile(r"[KQRBN]?[a-h]?[1-8]?x?([a-h][1-8])(?:=[QRBN])?[+#]?[!?]*")
_TOKEN_SPLIT = re.compile(r"[^A-Za-z0-9+#=!?]+")
_CASTLE = re.compile(r"(?<![A-Za-z0-9-])(O-O-O|O-O|0-0-0|0-0)[+#]?(?![A-Za-z0-9-])")


# A tactical adjective claimed of a piece on a square: "the bishop on b3 is pinned", "your
# rook on d1 is hanging". The coach must not assert these unless the board bears them out.
_TACTIC = re.compile(
    rf"(?:(?P<colour>{_COLOUR_WORDS})\s+)?(?P<piece>{_PIECE_WORD})s?\s+(?:on|at)\s+"
    r"(?P<square>[a-h][1-8])\s+"
    r"(?:(?:is|was|are|were)\s+(?:now\s+)?|,\s*(?:which\s+)?(?:is|was)\s+|now\s+)?"
    r"(?P<state>pinned|hanging|undefended|unprotected|loose|en\s*prise|trapped|safe|"
    r"forks?|forking|skewers?|skewering)",
    re.IGNORECASE,
)
# The same states said before the piece: "targeting the loose rook on e4" of a rook the
# pawn on f5 defends (the test audit, 2026-09-26).
_STATE_FIRST = re.compile(
    rf"\b(?:(?P<colour>{_COLOUR_WORDS})\s+)?(?P<state>loose|hanging|undefended|unprotected|"
    rf"pinned|trapped)\s+(?P<piece>{_PIECE_WORD})\s+(?:on|at)\s+(?P<square>[a-h][1-8])\b",
    re.IGNORECASE,
)
# The tagger's own piece values, so a fork and a skewer are judged the way it judges them.
_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


_SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)


def _pin_line(board: chess.Board, square: chess.Square) -> chess.Square | None:
    """The friendly piece a piece on `square` shields, or None if it shields nothing.

    A pin, absolute or relative, by the inventory's rule (`inventory.pin_target`), so a pin
    the coach was told about is not called false: an enemy slider that can legally take the
    piece on `square` attacks it along a line, and the next piece past it on that line is a
    friendly piece worth more that the slider would win once the front one moved -- the
    king, or a piece the capture wins (worth more than the slider, or undefended). A knight
    in front of a defended rook pins nothing.
    """
    piece = board.piece_at(square)
    if piece is None or piece.piece_type == chess.KING:
        return None
    for attacker in exchange.capturers(board, not piece.color, square):
        if board.piece_type_at(attacker) not in _SLIDERS:
            continue
        behind = _beyond(board, attacker, square)
        occupant = board.piece_at(behind) if behind is not None else None
        if (
            behind is not None
            and occupant is not None
            and occupant.color == piece.color
            and _VALUE[occupant.piece_type] > _VALUE[piece.piece_type]
            and _wins(board, attacker, behind, without=square)
        ):
            return behind
    return None


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def _beyond(board: chess.Board, frm: chess.Square, through: chess.Square) -> chess.Square | None:
    """The first occupied square past `through` on the line from `frm`, or None."""
    df = _sign(chess.square_file(through) - chess.square_file(frm))
    dr = _sign(chess.square_rank(through) - chess.square_rank(frm))
    if df == 0 and dr == 0:
        return None
    file_, rank_ = chess.square_file(through) + df, chess.square_rank(through) + dr
    while 0 <= file_ <= 7 and 0 <= rank_ <= 7:
        square = chess.square(file_, rank_)
        if board.piece_at(square) is not None:
            return square
        file_ += df
        rank_ += dr
    return None


def _wins(
    board: chess.Board,
    frm: chess.Square,
    target: chess.Square,
    *,
    without: chess.Square | None = None,
) -> bool:
    """Whether the piece on `frm` wins the piece on `target` by taking it, with the piece on
    `without` stepped off the line: the target is the king, or the capture is legal and nets
    material once the recaptures are played out (`exchange.exchange`)."""
    if board.piece_type_at(target) == chess.KING:
        return True
    color = board.color_at(frm)
    if color is None:
        return False
    probe = board.copy(stack=False)
    if without is not None:
        probe.remove_piece_at(without)
    probe = _to_move(probe, color)
    capture = next(
        iter(probe.generate_legal_moves(chess.BB_SQUARES[frm], chess.BB_SQUARES[target])), None
    )
    return capture is not None and exchange.exchange(probe, capture) > 0


def _fork_targets(board: chess.Board, square: chess.Square) -> list[chess.Square]:
    """Enemy pieces a fork on `square` would win, by the inventory's rule
    (`inventory.forks_in_place`): the king it checks, or a piece it can legally take that can
    be won -- worth more than the forker, or not defended well enough. A pinned knight takes
    nothing, so it forks nothing."""
    piece = board.piece_at(square)
    if piece is None:
        return []
    targets = []
    for target in board.attacks(square):
        other = board.piece_at(target)
        if other is None or other.color == piece.color:
            continue
        if other.piece_type == chess.KING or (
            square in exchange.capturers(board, piece.color, target)
            and exchange.can_be_won(board, target)
        ):
            targets.append(target)
    return targets


def _skewers(board: chess.Board, square: chess.Square) -> bool:
    """Whether the slider on `square` skewers, by the inventory's rule
    (`inventory.skewer_target`): it attacks an enemy piece that must step aside -- the king,
    or one it can legally take and win (worth more than the slider, or undefended) -- and
    past it on the same line stands a less valuable enemy piece the slider then wins."""
    piece = board.piece_at(square)
    if piece is None or piece.piece_type not in _SLIDERS:
        return False
    for front in board.attacks(square):
        target = board.piece_at(front)
        if target is None or target.color == piece.color or not _wins(board, square, front):
            continue
        behind = _beyond(board, square, front)
        occupant = board.piece_at(behind) if behind is not None else None
        if (
            behind is not None
            and occupant is not None
            and occupant.color != piece.color
            and _VALUE[occupant.piece_type] < _VALUE[target.piece_type]
            and _wins(board, square, behind, without=front)
        ):
            return True
    return False


def _tactic_problem(board: chess.Board, square_name: str, state: str) -> str | None:
    """Whether the board bears out a tactic claimed of a piece; None when it does or might.

    Only plainly false claims are faulted: a pin that is not there (to any piece, not only
    the king), a piece called loose that is defended, a piece called safe that can be won, a
    fork or skewer the piece does not make. Attack and defence are legal moves: a piece
    pinned to its king neither takes nor takes back, and a king takes back only where it may.
    """
    square = chess.parse_square(square_name.lower())
    piece = board.piece_at(square)
    if piece is None:
        return f"{square_name} is empty"
    name = PIECE_NAMES[piece.piece_type]
    state = state.lower().replace(" ", "").rstrip("s")
    if state == "pinned":
        if _pin_line(board, square) is None:
            return f"the {name} on {square_name} is not pinned"
    elif state in ("hanging", "enprise"):
        if not exchange.capturers(board, not piece.color, square):
            if board.attackers(not piece.color, square):
                return f"nothing can take the {name} on {square_name}"
            return f"nothing attacks the {name} on {square_name}"
        if not exchange.hanging(board, square):
            return f"the {name} on {square_name} is defended"
    elif state in ("undefended", "unprotected", "loose"):
        # A piece with no defender is undefended whether or not anything attacks it yet.
        if _guards(board, square):
            return f"the {name} on {square_name} is defended"
    elif state == "safe":
        if exchange.hanging(board, square):
            return f"the {name} on {square_name} is attacked and undefended"
        if exchange.can_be_won(board, square):
            return f"the {name} on {square_name} can be taken for profit"
    elif state == "trapped" and not board.attackers(not piece.color, square):
        return f"nothing is attacking the {name} on {square_name}"
    elif state in ("fork", "forking"):
        if len(_fork_targets(board, square)) < 2:
            if board.is_pinned(piece.color, square):
                return f"the {name} on {square_name} is pinned, so it forks nothing"
            return f"the {name} on {square_name} does not fork two pieces"
        if exchange.can_be_won(board, square):
            return f"the {name} on {square_name} can be taken for profit, so it forks nothing"
    elif state in ("skewer", "skewering") and not _skewers(board, square):
        return f"the {name} on {square_name} does not skewer anything"
    return None


# A piece placed at another moment than now: a plan ("aim to get a rook on c8"), a wish ("a
# knight on d5 would be a monster"), a future ("your king on g8 will be safe after
# castling", "castle, and the king on g1 covers g2"). Refused as false placements (the deep
# prompt review, 2026-10-01).
_ELSEWHERE = re.compile(
    r"\b(?:would|could|might|will|shall|if|aim|aims|aiming|(?<!at\s)once|after|belongs?|one\s+day|"
    r"eventually|later|someday|dream|ideal|imagine|suppose|wants?\s+to|get\s+(?:a|an|your)|"
    r"bring|reroute|heading)\b"
    r"|\b(?:a|an)\s+(?:king|queen|rook|bishop|knight|pawn)\s+(?:on|at)\s+[a-h][1-8]\b",
    re.IGNORECASE,
)


# Castling said: the king and rook it puts on their squares are there afterwards.
_CASTLING_SAID = re.compile(r"\bcastl\w*|O-O", re.IGNORECASE)


def _elsewhere(text: str, at: int, piece: str = "") -> bool:
    """Whether the sentence holding position `at` of `text` is about another moment (for a
    king or rook, castling's)."""
    start = max(text.rfind(stop, 0, at) for stop in (". ", "! ", "? ", "; ")) + 1
    ends = [i for i in (text.find(stop, at) for stop in (". ", "! ", "? ", "; ")) if i >= 0]
    sentence = text[start : min(ends) if ends else len(text)]
    if piece.lower() in ("king", "rook") and _CASTLING_SAID.search(sentence):
        return True
    return bool(_ELSEWHERE.search(sentence))


def check_draft(
    text: str,
    board: chess.Board,
    *,
    after: Sequence[chess.Board] = (),
    us: chess.Color | None = None,
) -> list[DraftProblem]:
    """Everything concrete in a coach draft that the board contradicts. Empty means clean.

    A "<piece> on <square>" phrase must match the board, colour word included ("your" is
    `us`, the player being coached, the side to move when not given: on the opponent's
    turn "your knight on f3" is the player's, and read as the side to move's it was
    flagged false -- the code review, 2026-09-26). A tactic claimed of such a piece (pinned,
    hanging, safe, a fork, a skewer) must hold on the board, or on one of the `after`
    positions, which are the boards along the best line: a fork the best move creates is
    real even though the current board does not show it yet. A move must be playable on one
    of the frames: legal now, legal for the opponent after a pass (a coach may name a
    threat), or legal further along the line the coach was given. Bare squares are never
    flagged; a coach may point at one.
    """
    # The model writes curly apostrophes ("your opponent’s bishop"); the patterns spell
    # them straight, and read that way a flipped owner went unchecked (the test audit,
    # 2026-09-26).
    text = straight_quotes(text)
    problems: list[DraftProblem] = []
    seen: set[str] = set()
    frames = [board, *after]
    us = board.turn if us is None else us

    def add(phrase: str, reason: str) -> None:
        if phrase not in seen:
            seen.add(phrase)
            problems.append(DraftProblem(text=phrase, reason=reason))

    for pattern in (_PIECE_ON, _SQUARE_PIECE):
        for match in pattern.finditer(text):
            # A placement is wrong only when it is wrong on every frame: a piece the best
            # move puts on a square is really there in the line the coach explains.
            reasons = [
                _placement_problem(
                    frame, match.group("colour"), match.group("piece"), match.group("square"), us
                )
                for frame in frames
            ]
            if all(reasons) and not _elsewhere(text, match.start(), match.group("piece")):
                add(match.group(0), reasons[0] or "")

    for match in [*_TACTIC.finditer(text), *_STATE_FIRST.finditer(text)]:
        square, state = match.group("square"), match.group("state")
        present = [
            _placement_problem(frame, None, match.group("piece"), square, us) for frame in frames
        ]
        if all(present):
            continue  # the piece is on that square on no frame; the placement loop has it
        reasons = [_tactic_problem(frame, square, state) for frame in frames]
        if any(reason is None for reason in reasons):
            continue
        clean = next((r for r in reasons if r), None)
        if clean:
            add(match.group(0), clean)

    for token in _TOKEN_SPLIT.split(text):
        match = _SAN_TOKEN.fullmatch(token)
        if not match:
            continue
        core = _TRAILING_MARKS.sub("", token)
        if core == match.group(1):
            continue  # a bare square
        reasons = [_move_problem(frame, core) for frame in frames]
        if all(reasons):
            add(token, reasons[0] or "")
    for match in _CASTLE.finditer(text):
        reasons = [_move_problem(frame, match.group(1).replace("0", "O")) for frame in frames]
        if all(reasons):
            add(match.group(0), reasons[0] or "")
    for phrase, reason in _strategic_problems(text, board, us):
        add(phrase, reason)
    for phrase, reason in _line_problems(text, frames, us):
        add(phrase, reason)
    for phrase, reason in _relation_problems(text, frames, us):
        add(phrase, reason)
    for phrase, reason in _mate_problems(text, frames):
        add(phrase, reason)
    return problems


# What a move is said to force: the king away, or a piece onto a square. "The check forced the
# King away" where a block was legal, "the check on d6 forces your rook onto f7" where other
# moves were: 2 of the 5 false claims the bench's labellers found (2026-09-25).
_FORCE_KING = re.compile(
    r"\b(?:forc(?:e|es|ed|ing)|driv(?:e|es|ing)|drove)\s+"
    r"(?P<colour>your|their|the|his|white's|black's)\s+king\s+"
    r"(?:away|back|out|off|to\s+move|to\s+step\s+aside|to\s+(?P<square>[a-h][1-8]))\b"
    r"|\b(?P<colour2>your|their)\s+king\s+(?:has\s+to|must|is\s+forced\s+to)\s+move\b",
    re.IGNORECASE,
)
_FORCE_ONTO = re.compile(
    r"\b(?:forc(?:e|es|ed|ing)|driv(?:e|es|ing)|drove)\s+"
    r"(?P<colour>your|their|the|his|white's|black's)\s+"
    r"(?P<piece>queen|rook|bishop|knight|pawn)\s+(?:back\s+)?(?:onto|to|into)\s+"
    r"(?P<square>[a-h][1-8])\b",
    re.IGNORECASE,
)
# Files said to be kept open, or to be open, more than one at a time: "Rad8 keeps the b-file
# and e-file open", "the b- and e-files are open". One file alone is `_OPEN_FILE`'s.
_FILES_OPEN = re.compile(
    r"\bkeep(?:s|ing)?\s+(?P<body>(?:the\s+)?[a-h]-(?:file)?(?:\s*(?:,|and)\s*(?:the\s+)?"
    r"[a-h]-(?:file)?)*s?)\s+open\b"
    r"|(?P<body2>\b[a-h]-(?:file)?(?:\s*(?:,|and)\s*(?:the\s+)?[a-h]-(?:file)?)+s?)\s+"
    r"(?:are|stay|remain)\s+open\b",
    re.IGNORECASE,
)
# A file said to have no pawns: "the g-file has no pawns" of a half-open file with a pawn on
# g6 (the pilot of 2026-09-26: the sheet said "takes the half-open g-file").
# "No pawns of yours on it" is a half-open file, and true of one: only a file said to have no
# pawns at all is checked.
_FILE_EMPTY = re.compile(
    r"\bthe\s+(?P<file>[a-h])-file\s+(?:has\s+no\s+pawns(?!\s+of\b)|is\s+(?:empty|"
    r"clear\s+of\s+pawns|free\s+of\s+pawns|pawnless))\b",
    re.IGNORECASE,
)
_SAN = r"(?:O-O(?:-O)?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h](?:x[a-h])?[1-8])(?:=[QRBN])?[+#]?"
# Only the move itself as the subject: "Ne4 allows exf6", "Ne4 saves the knight, but it
# allows exf6" -- read across clauses ("after Rb8, Rxa5 wins, while your move allowed
# fxe6") the subject was another move (the precision sweep, 2026-09-26).
_ALLOWS = re.compile(
    rf"\b(?P<first>{_SAN})(?![\w=])(?:[^.;!?,]{{0,50}}?,?\s+but\s+it|\s+it|)\s+"
    rf"(?:allows|allowed|lets|runs\s+into|ran\s+into|invites|invited)\s+"
    rf"(?P<reply>{_SAN})(?![\w=])"
)
# A piece named by its owner alone: "your queen checks" where the player had none.
# Only with a present action ("your queen checks"): a piece named in the game's story
# ("it took your bishop") is gone from the frames and true (the precision sweep).
_OWNED = re.compile(
    r"\b(?P<colour>your|their)\s+(?P<piece>queen|rook|bishop|knight)s?\s+"
    r"(?:checks?|gives\s+check|attacks|pins|forks|skewers|mates|takes|captures)\b",
    re.IGNORECASE,
)
# Only the unqualified claims: the only king move, the only legal move, the only move out
# of check. "The only move that keeps your edge" is the stakes, and true (the sweep).
_ONLY_MOVE = re.compile(
    rf"\b(?P<san>{_SAN})(?![\w=])\s+(?:is|was)\s+the\s+only\s+"
    r"(?:(?P<king>king\s+)move\b|legal\s+move\b|move\s+out\s+of\s+(?:the\s+)?check\b)"
)
_SAN_CHECK = re.compile(
    r"\b(?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h](?:x[a-h])?[1-8])(?:=[QRBN])?\+"
)


# A relation said of a piece in the present: "your knight on f6 attacks their queen on d1",
# "the queen on g5 forks your knight on e5 and the pawn on g2", "the bishop on c4 pins the
# pawn on f7". Never checked before: the deep prompt review (2026-10-01) let these through.
_PIECE_AT = (
    rf"(?:(?:the|a|an|{_COLOUR_WORDS})\s+)?{_PIECE_ADJECTIVES}(?:[a-h]-)?(?:{_PIECE_WORD})s?"
    r"\s+(?:on|at)\s+[a-h][1-8]\b"
)
_RELATION = re.compile(
    rf"\b(?:(?P<colour>{_COLOUR_WORDS})\s+)?{_PIECE_ADJECTIVES}(?:[a-h]-)?(?P<piece>{_PIECE_WORD})"
    r"\s+(?:on|at)\s+(?P<square>[a-h][1-8])\s+(?:(?:now|already|also|still|directly)\s+)?"
    r"(?:is\s+(?:now\s+|already\s+|also\s+|still\s+)?(?P<ing>attacking|hitting|defending|"
    r"protecting|guarding|pinning|forking)|(?P<verb>attacks|hits|defends|protects|guards|"
    r"covers|pins|forks))\s+(?P<objects>[^.;:!?]*)",
    re.IGNORECASE,
)
# The same said the other way round: "your knight on e5 is defended by the pawn on d4".
_RELATION_BY = re.compile(
    rf"\b(?:(?P<colour>{_COLOUR_WORDS})\s+)?{_PIECE_ADJECTIVES}(?:[a-h]-)?(?P<piece>{_PIECE_WORD})"
    r"\s+(?:on|at)\s+(?P<square>[a-h][1-8])\s+(?:is|was)\s+(?:now\s+|already\s+|also\s+|still\s+)?"
    r"(?P<verb>attacked|defended|protected|guarded|pinned|forked)\s+by\s+"
    rf"(?:(?:the|a|an|(?P<bycolour>{_COLOUR_WORDS}))\s+)?{_PIECE_ADJECTIVES}(?:[a-h]-)?"
    rf"(?P<bypiece>{_PIECE_WORD})\s+(?:on|at)\s+(?P<bysquare>[a-h][1-8])\b",
    re.IGNORECASE,
)
_OBJECT = re.compile(
    rf"(?:(?:the|a|an|(?P<colour>{_COLOUR_WORDS}))\s+)?{_PIECE_ADJECTIVES}(?:[a-h]-)?"
    rf"(?P<piece>{_PIECE_WORD})s?\s+(?:on|at)\s+(?P<square>[a-h][1-8])\b",
    re.IGNORECASE,
)
# Where the objects of a relation end: the next clause is about something else.
_CLAUSE_END = re.compile(
    r",(?!\s*(?:and\s+)?(?:the|a|an|your|their|white's|black's|white|black)\s)|\s(?:so|which|"
    r"while|but|because|since|if|when|after|before|until|then|that|who|where|whereas)\b",
    re.IGNORECASE,
)
# What makes a sentence about a moment that is not on the board, or about a line: a plan, a
# hope, a condition, a future ("once", "after", "would", "will"), an x-ray ("through").
_NOT_PRESENT = re.compile(
    r"\b(?:would|could|might|may|will|shall|should|can|cannot|can't|won't|wouldn't|if|once|"
    r"after|when|whenever|until|unless|aim|aims|aiming|want|wants|plan|plans|hope|try|tries|"
    r"trying|soon|later|eventually|next|then|imagine|suppose|once|through|x-ray|x-rays|behind|"
    r"line|lines|afterwards|ready|about to|going to|set to)\b",
    re.IGNORECASE,
)
# A move said to mate now: "Qxf7 is mate", "Qxf7 mates", "Qxf7 gives checkmate".
_MATE_SAID = re.compile(
    rf"(?<![\w=])(?P<san>{_SAN})(?![\w=])\s+(?:is|gives|delivers)\s+(?:check)?mate\b"
    rf"|(?<![\w=])(?P<san2>{_SAN})(?![\w=])\s+(?:checkmates|mates)\b",
)
# A mate in one said to be on the board: "there is mate in one", "you have a mate in one".
_MATE_IN_ONE = re.compile(
    r"\b(?:there(?:'s|\s+is)|you\s+have|they\s+have|(?:white|black)\s+has)\s+(?:a\s+)?"
    r"mate\s+in\s+(?:one|1)\b",
    re.IGNORECASE,
)
# What else puts a mate at another moment than now: what a move stops, allows or prepares.
_NOT_NOW = re.compile(
    r"\b(?:otherwise|unless|else|allow\w*|stop\w*|prevent\w*|avoid\w*|threat\w*|prepar\w*|"
    r"follow\w*|before|instead|missed|was|were|had)\b",
    re.IGNORECASE,
)


def _on(frame: chess.Board, colour: str | None, piece: str, square: str, us: chess.Color) -> bool:
    return _placement_problem(frame, colour, piece, square, us) is None


def _pins_it(board: chess.Board, attacker: chess.Square, front: chess.Square) -> bool:
    """Whether the slider on `attacker` pins the piece on `front`, by `_pin_line`'s rule: the
    piece behind, on the same line, is its side's and worth more, and would be won."""
    if board.piece_type_at(attacker) not in _SLIDERS or front not in board.attacks(attacker):
        return False
    piece = board.piece_at(front)
    behind = _beyond(board, attacker, front)
    occupant = board.piece_at(behind) if behind is not None else None
    return (
        piece is not None
        and behind is not None
        and occupant is not None
        and occupant.color == piece.color
        and piece.color != board.color_at(attacker)
        and _VALUE[occupant.piece_type] > _VALUE[piece.piece_type]
        and _wins(board, attacker, behind, without=front)
    )


def _relation_holds(
    board: chess.Board, kind: str, subject: chess.Square, objects: Sequence[chess.Square]
) -> bool:
    if kind == "pin":
        return all(_pins_it(board, subject, o) for o in objects)
    # Attack and defence as python-chess has them: the squares the piece bears on, a pinned
    # piece's included (a coach says a pinned knight still defends).
    return all(o in board.attacks(subject) for o in objects)


_KIND = {
    "attack": "attack",
    "hit": "attack",
    "defend": "defend",
    "protect": "defend",
    "guard": "defend",
    "cover": "defend",
    "pin": "pin",
    "fork": "attack",
}


def _verb_kind(verb: str) -> str:
    word = verb.lower()
    for stem, kind in _KIND.items():
        if word.startswith(stem):
            return kind
    return "attack"


_JOIN = re.compile(r"\s*(?:,\s*(?:and\s+)?|and\s+|as\s+well\s+as\s+)", re.IGNORECASE)


def _objects_first(rest: str) -> list[tuple[str | None, str, str]]:
    """The pieces named right after the verb, one after another ("your knight on e5 and the
    pawn on g2"): "attacks it and left the queen on c2 active" names no object of the verb."""
    found: list[tuple[str | None, str, str]] = []
    rest = rest.lstrip()
    pos = 0
    while True:
        match = _OBJECT.match(rest, pos)
        if match is None:
            break
        found.append((match.group("colour"), match.group("piece"), match.group("square")))
        join = _JOIN.match(rest, match.end())
        if join is None:
            break
        pos = join.end()
    return found


def _relation_problems(
    text: str, frames: Sequence[chess.Board], us: chess.Color
) -> list[tuple[str, str]]:
    """Relations said of pieces in the present that no frame bears out: an attack, a defence,
    a pin or a fork. Only a sentence about the present is read (no plan, condition, future,
    line or x-ray), only pieces named with their squares, and only where some frame has
    every named piece on its square: elsewhere the placement check speaks."""
    problems: list[tuple[str, str]] = []
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        relations: list[tuple[str, str, str | None, str, str, list[tuple[str | None, str, str]]]]
        relations = []
        for match in _RELATION.finditer(sentence):
            if _NOT_PRESENT.search(sentence[: match.start()]) or _NOT_PRESENT.search(
                sentence[match.start() : match.end("objects") - len(match.group("objects"))]
            ):
                continue
            verb = match.group("ing") or match.group("verb")
            rest = match.group("objects")
            cut = _CLAUSE_END.search(rest)
            rest = rest[: cut.start()] if cut else rest
            kind = _verb_kind(verb)
            if kind == "pin":
                # "pins the knight on f6 to the queen on d8": the pinned piece is the first.
                rest = re.split(r"\bto\b", rest, maxsplit=1)[0]
            if _NOT_PRESENT.search(rest):
                continue
            objects: list[tuple[str | None, str, str]] = _objects_first(rest)
            if not objects and kind == "defend":
                bare = re.match(r"\s*(?:the\s+)?([a-h][1-8])\b(?!\s*-)", rest)
                if bare:
                    objects = [(None, "", str(bare.group(1)))]
            if not objects:
                continue
            relations.append(
                (
                    match.group(0).strip(),
                    kind,
                    match.group("colour"),
                    match.group("piece"),
                    match.group("square"),
                    objects,
                )
            )
        for match in _RELATION_BY.finditer(sentence):
            if _NOT_PRESENT.search(sentence[: match.start()]):
                continue
            kind = _verb_kind(match.group("verb"))
            relations.append(
                (
                    match.group(0).strip(),
                    kind,
                    match.group("bycolour"),
                    match.group("bypiece"),
                    match.group("bysquare"),
                    [(match.group("colour"), match.group("piece"), match.group("square"))],
                )
            )
        for phrase, kind, colour, piece, square, objects in relations:
            subject = chess.parse_square(square.lower())
            placed = [
                frame
                for frame in frames
                if _on(frame, colour, piece, square, us)
                and all(not op or _on(frame, oc, op, sq, us) for oc, op, sq in objects)
            ]
            if not placed:
                continue
            targets = [chess.parse_square(sq.lower()) for _, _, sq in objects]
            if any(_relation_holds(frame, kind, subject, targets) for frame in placed):
                continue
            name = piece.lower()
            said = " and ".join(sq.lower() for _, _, sq in objects)
            verb = {"attack": "attack", "defend": "defend", "pin": "pin"}[kind]
            problems.append((phrase, f"the {name} on {square.lower()} does not {verb} {said}"))
    return problems


def _mate_problems(text: str, frames: Sequence[chess.Board]) -> list[tuple[str, str]]:
    """A move said to be mate that mates on no frame where it can be played, and "mate in
    one" where no frame has one for either side."""
    problems: list[tuple[str, str]] = []
    # Only a sentence about the present: a mate named in a line, a threat or a plan ("stops
    # Qxg2#", "after Kf7, Ra7 is mate", "then Qa8 would be mate") is about a later moment,
    # and checked so, 20 of 21 flags over the stored replies were true claims (2026-10-01).
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        if _NOT_PRESENT.search(sentence) or _NOT_NOW.search(sentence):
            continue
        for match in _MATE_SAID.finditer(sentence):
            core = _TRAILING_MARKS.sub("", match.group("san") or match.group("san2"))
            played = mates = False
            for frame in frames:
                for side_board in _both_sides(frame):
                    move, _ = _parse_move(side_board, core)
                    if move is None:
                        continue
                    played = True
                    after = side_board.copy(stack=False)
                    after.push(move)
                    mates = mates or after.is_checkmate()
            if played and not mates:
                problems.append((match.group(0), f"{core} is not mate"))
        said = _MATE_IN_ONE.search(sentence)
        if said and not any(_has_mate_in_one(b) for f in frames for b in _both_sides(f)):
            problems.append((said.group(0), "no mate in one is on the board"))
    return problems


def _both_sides(frame: chess.Board) -> list[chess.Board]:
    """The frame, and the frame with the other side to move when it is not in check (a
    threat is a move after a pass)."""
    if frame.is_check():
        return [frame]
    passed = frame.copy(stack=False)
    passed.push(chess.Move.null())
    return [frame, passed]


def _has_mate_in_one(board: chess.Board) -> bool:
    for move in board.legal_moves:
        board.push(move)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            return True
    return False


def straight_quotes(text: str) -> str:
    """`text` with curly apostrophes and quotes made straight."""
    return text.replace("\u2019", "'").replace("\u2018", "'")


def _legal_san(board: chess.Board, san: str) -> bool:
    try:
        board.parse_san(san)
    except ValueError:
        return False
    return True


def _side(word: str | None, us: chess.Color) -> chess.Color | None:
    """The side a colour word names: "your" is `us`; "the" and "his" either side."""
    word = (word or "").lower()
    if word == "your":
        return us
    if word == "their":
        return not us
    if word.startswith("white"):
        return chess.WHITE
    if word.startswith("black"):
        return chess.BLACK
    return None


def _checked_frames(
    text: str, frames: Sequence[chess.Board], side: chess.Color | None
) -> list[chess.Board]:
    """The frames where `side` (either, when None) is in check. When the sentence names a
    check ("Rh4+"), those after it, played where it is legal -- only those: "Qf2+ drives
    the King to h3" was borne out by some other check of the line (the test audit); else
    every frame in check."""
    named = _SAN_CHECK.findall(text)
    out = [] if named else [f for f in frames if f.is_check() and (side is None or f.turn == side)]
    for token in named:
        for frame in frames:
            try:
                move = frame.parse_san(token)
            except ValueError:
                continue
            after = frame.copy(stack=False)
            after.push(move)
            if after.is_check() and (side is None or after.turn == side):
                out.append(after)
    return out


def _line_problems(
    text: str, frames: Sequence[chess.Board], us: chess.Color
) -> list[tuple[str, str]]:
    """What a line is said to force, or to keep open, that none of the frames bears out. A
    claim no frame applies to -- no check on the board or the line for that side -- is left
    alone: it is about some other moment of the game."""
    problems: list[tuple[str, str]] = []
    for sentence_text in re.split(r"(?<=[.!?])\s+", text):
        for match in _FORCE_KING.finditer(sentence_text):
            side = _side(match.group("colour") or match.group("colour2"), us)
            checked = _checked_frames(sentence_text, frames, side)
            if not checked:
                continue
            to = match.group("square")
            if to is None:
                if any(
                    any(True for _ in f.legal_moves)
                    and all(f.piece_type_at(m.from_square) == chess.KING for m in f.legal_moves)
                    for f in checked
                ):
                    continue
                reason = (
                    "the check does not force the king to move: it can be blocked "
                    "or the checking piece taken"
                )
            else:
                # "Forces the King to h6" where Kf6 was legal too (the test audit).
                target = chess.parse_square(to.lower())
                if any(
                    any(True for _ in f.legal_moves)
                    and all(
                        f.piece_type_at(m.from_square) == chess.KING and m.to_square == target
                        for m in f.legal_moves
                    )
                    for f in checked
                ):
                    continue
                reason = f"nothing forces the king to {to.lower()}: other moves are legal there"
            problems.append((match.group(0), reason))
        for match in _FORCE_ONTO.finditer(sentence_text):
            side = _side(match.group("colour"), us)
            kind = _PIECE_WORDS.get(match.group("piece").lower())
            square = chess.parse_square(match.group("square").lower())
            candidates = [
                f
                for f in [*frames, *_checked_frames(sentence_text, frames, side)]
                if (side is None or f.turn == side)
                and any(
                    m.to_square == square and f.piece_type_at(m.from_square) == kind
                    for m in f.legal_moves
                )
            ]
            if not candidates:
                continue
            if any(
                all(
                    m.to_square == square and f.piece_type_at(m.from_square) == kind
                    for m in f.legal_moves
                )
                for f in candidates
            ):
                continue
            problems.append(
                (
                    match.group(0),
                    f"nothing forces the {match.group('piece').lower()} to "
                    f"{match.group('square').lower()}: other moves are legal there",
                )
            )
    for match in _OWNED.finditer(text):
        side = _side(match.group("colour"), us)
        kind = _PIECE_WORDS.get(match.group("piece").lower())
        if side is None or kind is None or kind == chess.KING:
            continue
        if any(f.pieces(kind, side) for f in frames):
            continue
        whose = "you have" if side == us else "they have"
        problems.append((match.group(0), f"{whose} no {match.group('piece').lower()}"))
    for match in _ONLY_MOVE.finditer(text):
        san, king_only = match.group("san"), bool(match.group("king"))
        where = [f for f in frames if _legal_san(f, san)]
        if not where:
            continue

        def alone(frame: chess.Board, king_only: bool = king_only) -> bool:
            moves = list(frame.legal_moves)
            if king_only:
                moves = [m for m in moves if frame.piece_type_at(m.from_square) == chess.KING]
            return len(moves) == 1

        if any(alone(f) for f in where):
            continue
        kind = "king move" if king_only else "move"
        problems.append((match.group(0), f"{san} is not the only {kind}: others are legal"))
    for match in _ALLOWS.finditer(text):
        first, reply = match.group("first"), match.group("reply")
        played_on = []
        for frame in frames:
            try:
                move = frame.parse_san(first)
            except ValueError:
                continue
            after = frame.copy(stack=False)
            after.push(move)
            played_on.append(after)
        if not played_on:
            continue  # a move of some other moment: the move check has it
        if any(_legal_san(after, reply) for after in played_on):
            continue
        problems.append(
            (match.group(0), f"{reply} cannot be played after {first}: it does not allow it")
        )
    for match in _FILE_EMPTY.finditer(text):
        letter = match.group("file").lower()
        index = ord(letter) - ord("a")
        if not any(
            all(f.piece_type_at(sq) != chess.PAWN for sq in _file_squares(index)) for f in frames
        ):
            problems.append((match.group(0), f"the {letter}-file has a pawn on it"))
    for match in _FILES_OPEN.finditer(text):
        body = match.group("body") or match.group("body2") or ""
        for letter in dict.fromkeys(re.findall(r"\b([a-h])-", body.lower())):
            index = ord(letter) - ord("a")
            if not any(
                all(f.piece_type_at(sq) != chess.PAWN for sq in _file_squares(index))
                for f in frames
            ):
                problems.append(
                    (match.group(0), f"the {letter}-file is not open; a pawn stands on it")
                )
    return problems


# A file called open or half-open, in either word order ("the open e-file", "the e-file is
# open"). A pawn called passed, by square or by file, in either order.
_OPEN_FILE = re.compile(
    r"\b(?:(?P<k1>open|half[-\s]?open)\s+(?P<f1>[a-h])[-\s]?file"
    r"|(?P<f2>[a-h])[-\s]?file\s+(?:is|are|remains?)\s+(?:now\s+)?(?P<k2>open|half[-\s]?open))\b",
    re.IGNORECASE,
)
_PASSED = re.compile(
    r"\bpassed\s+pawn\s+on\s+(?P<sq1>[a-h][1-8])"
    r"|\bpawn\s+on\s+(?P<sq2>[a-h][1-8])\s+(?:is|are)\s+(?:a\s+|now\s+)*passed"
    r"|\bpassed\s+(?P<file1>[a-h])[-\s]?pawn"
    r"|\b(?P<file2>[a-h])[-\s]?pawn\s+(?:is|are)\s+(?:a\s+|now\s+)*passed",
    re.IGNORECASE,
)


def _file_squares(file_index: int) -> list[chess.Square]:
    return [chess.square(file_index, rank) for rank in range(8)]


def _strategic_problems(
    text: str, board: chess.Board, us: chess.Color | None = None
) -> list[tuple[str, str]]:
    """Concrete strategic claims the board contradicts: a file said open that has pawns, a
    pawn said passed that an enemy pawn still holds up. Only clear, checkable contradictions
    are flagged; a fuzzy claim (space, king safety, activity) is left alone.
    """
    us = board.turn if us is None else us
    problems: list[tuple[str, str]] = []
    for match in _OPEN_FILE.finditer(text):
        kind = (match.group("k1") or match.group("k2")).lower().replace(" ", "").replace("-", "")
        letter = (match.group("f1") or match.group("f2")).lower()
        pawns = [board.piece_at(sq) for sq in _file_squares(ord(letter) - ord("a"))]
        ours = any(p and p.piece_type == chess.PAWN and p.color == us for p in pawns)
        theirs = any(p and p.piece_type == chess.PAWN and p.color != us for p in pawns)
        if kind == "open" and (ours or theirs):
            problems.append((match.group(0), f"the {letter}-file is not open; a pawn stands on it"))
        elif kind == "halfopen":
            # Whose file it is, as the sentence says: "for you" or "your" is the player's,
            # "their" or "the opponent's" theirs ("your opponent takes the half-open f-file":
            # round 2 refused two good replies read as the player's); unsaid, either side's.
            window = text[max(0, match.start() - 60) : match.end() + 20].lower()
            if re.search(r"\bopponent|\btheir\b|\bfor them\b", window):
                wrong, whose = theirs, "for them; they have a pawn there"
            elif re.search(r"\byour?\b|\bfor you\b", window):
                wrong, whose = ours, "for you; you have a pawn there"
            else:
                wrong, whose = ours and theirs, "for either side: both have a pawn there"
            if wrong:
                problems.append((match.group(0), f"the {letter}-file is not half-open {whose}"))
    for match in _PASSED.finditer(text):
        square_name = match.group("sq1") or match.group("sq2")
        letter = match.group("file1") or match.group("file2")
        if square_name:
            square = chess.parse_square(square_name.lower())
            piece = board.piece_at(square)
            if (
                piece is not None
                and piece.piece_type == chess.PAWN
                and not is_passed(board, square)
            ):
                problems.append(
                    (match.group(0), f"the pawn on {square_name.lower()} is not passed")
                )
        elif letter:
            ours = [
                sq
                for sq in _file_squares(ord(letter.lower()) - ord("a"))
                if board.piece_type_at(sq) == chess.PAWN and board.color_at(sq) == us
            ]
            if ours and not any(is_passed(board, sq) for sq in ours):
                problems.append(
                    (match.group(0), f"your {letter.lower()}-pawn is not a passed pawn")
                )
    return problems


def _placement_problem(
    board: chess.Board,
    colour_word: str | None,
    piece_word: str,
    square_name: str,
    us: chess.Color | None = None,
) -> str | None:
    """Whether a "<colour> <piece> on <square>" claim holds. `us` fixes whose "your" is.

    Across the current board and the board after the best move, whose turn it is flips, but
    "your" always means the player being coached, so `us` is passed rather than read from
    the frame.
    """
    if us is None:
        us = board.turn
    square = chess.parse_square(square_name.lower())
    piece = board.piece_at(square)
    wanted = _PIECE_WORDS[piece_word.lower()]
    if piece is None:
        return f"{square_name} is empty"
    if piece.piece_type != wanted:
        held = f"{'your' if piece.color == us else 'their'} {PIECE_NAMES[piece.piece_type]}"
        return f"{square_name} holds {held}, not a {piece_word.lower()}"
    if colour_word is None:
        return None
    word = colour_word.lower()
    if word == "your":
        expected: chess.Color = us
    elif word.startswith("white"):
        expected = chess.WHITE
    elif word.startswith("black"):
        expected = chess.BLACK
    else:
        expected = not us
    if piece.color == expected:
        return None
    if word.startswith(("white", "black")):
        actual = "white's" if piece.color == chess.WHITE else "black's"
        return f"the {piece_word.lower()} on {square_name} is {actual}, not {word}"
    if piece.color == us:
        return f"the {piece_word.lower()} on {square_name} is yours, not your opponent's"
    return f"the {piece_word.lower()} on {square_name} is your opponent's, not yours"


def _move_problem(board: chess.Board, san: str) -> str | None:
    move, why = _parse_move(board, san)
    if move is not None:
        return None
    if why == "ambiguous":
        return f"{san} could be more than one move here"
    if not board.is_check():
        passed = board.copy(stack=False)
        passed.push(chess.Move.null())
        threat, _ = _parse_move(passed, san)
        if threat is not None:
            return None
    return f"{san} cannot be played here by either side"
