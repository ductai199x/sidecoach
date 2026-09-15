"""Claim verification, both directions. DESIGN.md section 5.5.

Direction one: what the user says in their own defence ("my knight is attacking e5 and the
pawn is hanging") is turned into structured claims by the LLM and every checkable one is
tested against the board before the coach agrees with anything. Only the extraction uses
the LLM. Resolving "my knight" to a square, counting attackers and defenders, trying the
null move and comparing with the engine's candidates all happen here, in python-chess,
where nothing can be made up.

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
            if hit is not None and hit.color != board.color_at(subject):
                defenders = board.attackers(hit.color, target)
                if defenders:
                    reason += f", though it is defended by {pos.listed(defenders)}"
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
                others = [sq for sq in board.attackers(held.color, target) if sq != subject]
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
        attackers = board.attackers(not piece.color, target)
        defenders = board.attackers(piece.color, target)
        if attackers and not defenders:
            reason = (
                f"{pos.name(target)} is attacked by {pos.listed(attackers)} and nothing defends it"
            )
            return "true", _sentence(_after_move(pos, reason))
        if not attackers:
            reason = f"{pos.name(target)} is not attacked at all"
        else:
            reason = (
                f"{pos.name(target)} is attacked by {pos.listed(attackers)} but defended by "
                f"{pos.listed(defenders)}, so it is not hanging outright"
            )
            cheapest = min(attackers, key=lambda sq: _VALUE[board.piece_type_at(sq) or 0])
            if _VALUE[board.piece_type_at(cheapest) or 0] < _VALUE[piece.piece_type]:
                reason += (
                    f", though {pos.short(cheapest)} taking a {PIECE_NAMES[piece.piece_type]} "
                    "still wins material"
                )
            else:
                reason += "; I have not counted the whole exchange"
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
    """Whether the piece on `target` (or the empty square) can be taken, with the counts."""
    board = pos.board
    square = chess.square_name(target)
    piece = board.piece_at(target)
    if piece is None:
        attackers = board.attackers(pos.them, target)
        if not attackers:
            return True, f"{square} is not attacked by any of their pieces"
        return False, f"{square} is attacked by {pos.listed(attackers)}"
    attackers = board.attackers(not piece.color, target)
    defenders = board.attackers(piece.color, target)
    if piece.piece_type == chess.KING:
        if not attackers:
            return True, f"{pos.name(target)} is not in check"
        return False, f"{pos.name(target)} is in check from {pos.listed(attackers)}"
    if not attackers:
        return True, f"{pos.name(target)} is not attacked"
    counts = (
        f"{pos.name(target)} is attacked {_times(len(attackers))} ({pos.listed(attackers)}) "
        f"and defended {_times(len(defenders))}"
    )
    if defenders:
        counts += f" ({pos.listed(defenders)})"
    if len(defenders) < len(attackers):
        return False, f"{counts}, so it can be taken"
    cheapest = min(attackers, key=lambda sq: _VALUE[board.piece_type_at(sq) or 0])
    if _VALUE[board.piece_type_at(cheapest) or 0] < _VALUE[piece.piece_type]:
        return False, (
            f"{counts}, but {pos.short(cheapest)} is worth less than a "
            f"{PIECE_NAMES[piece.piece_type]}, so it can still be taken profitably"
        )
    return True, f"{counts}, so it holds"


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
    """ "their bishop on c4 takes your pawn on f7, and your king on e8 can take back"."""
    actor = pos.name(move.from_square)
    victim = position.piece_at(move.to_square)
    san = position.san(move)
    if victim is None:
        what = f"{actor} can play {san}"
        if position.gives_check(move):
            what += ", with check"
        return what
    what = f"{actor} takes {pos.name(move.to_square)} with {san}"
    recapturers = position.attackers(victim.color, move.to_square)
    if recapturers:
        return f"{what}, and {pos.listed(recapturers)} can take back"
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
_PIECE_ON = re.compile(
    rf"\b(?:(?P<colour>{_COLOUR_WORDS})\s+)?(?P<piece>{_PIECE_WORD})s?\s+(?:on|at)\s+"
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
# The tagger's own piece values, so a fork and a skewer are judged the way it judges them.
_VALUE: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 100,
}


def _pin_line(board: chess.Board, square: chess.Square) -> chess.Square | None:
    """The friendly piece a piece on `square` shields, or None if it shields nothing.

    A pin, absolute or relative: an enemy slider attacks `square` along a line, and the next
    piece past it on that line is a friendly piece worth more (the king counts as most). This
    is the tagger's own test, so a true pin to the queen is not called false.
    """
    piece = board.piece_at(square)
    if piece is None:
        return None
    for attacker in board.attackers(not piece.color, square):
        slider = board.piece_at(attacker)
        if slider is None or slider.piece_type not in (chess.BISHOP, chess.ROOK, chess.QUEEN):
            continue
        if chess.square_distance(attacker, square) == 0:
            continue
        df = _sign(chess.square_file(square) - chess.square_file(attacker))
        dr = _sign(chess.square_rank(square) - chess.square_rank(attacker))
        file_, rank_ = chess.square_file(square) + df, chess.square_rank(square) + dr
        while 0 <= file_ <= 7 and 0 <= rank_ <= 7:
            behind = chess.square(file_, rank_)
            occupant = board.piece_at(behind)
            if occupant is not None:
                if (
                    occupant.color == piece.color
                    and _VALUE[occupant.piece_type] > _VALUE[piece.piece_type]
                ):
                    return behind
                break
            file_ += df
            rank_ += dr
    return None


def _sign(n: int) -> int:
    return (n > 0) - (n < 0)


def _fork_targets(board: chess.Board, square: chess.Square) -> list[chess.Square]:
    """Enemy pieces a fork on `square` would win: worth as much as the forker, or the king."""
    piece = board.piece_at(square)
    if piece is None:
        return []
    own_value = _VALUE[piece.piece_type]
    targets = []
    for target in board.attacks(square):
        other = board.piece_at(target)
        if other is None or other.color == piece.color:
            continue
        if other.piece_type == chess.KING or _VALUE[other.piece_type] >= own_value:
            targets.append(target)
        elif not board.attackers(other.color, target):
            targets.append(target)  # an undefended lesser piece is still worth the fork
    return targets


def _tactic_problem(board: chess.Board, square_name: str, state: str) -> str | None:
    """Whether the board bears out a tactic claimed of a piece; None when it does or might.

    Only plainly false claims are faulted: a pin that is not there (to any piece, not only
    the king), a piece called loose that is defended, a piece called safe that hangs, a fork
    or skewer the piece does not make. The exchange is not evaluated, so "safe" is faulted
    only when the piece is attacked and not defended at all.
    """
    square = chess.parse_square(square_name.lower())
    piece = board.piece_at(square)
    if piece is None:
        return f"{square_name} is empty"
    name = PIECE_NAMES[piece.piece_type]
    state = state.lower().replace(" ", "").rstrip("s")
    attackers = board.attackers(not piece.color, square)
    defenders = board.attackers(piece.color, square)
    if state == "pinned":
        if _pin_line(board, square) is None:
            return f"the {name} on {square_name} is not pinned"
    elif state in ("hanging", "undefended", "unprotected", "loose", "enprise"):
        if not attackers:
            return f"nothing attacks the {name} on {square_name}"
        if defenders:
            return f"the {name} on {square_name} is defended"
    elif state == "safe":
        if attackers and not defenders:
            return f"the {name} on {square_name} is attacked and undefended"
    elif state == "trapped" and not attackers:
        return f"nothing is attacking the {name} on {square_name}"
    elif state in ("fork", "forking"):
        if len(_fork_targets(board, square)) < 2:
            return f"the {name} on {square_name} does not fork two pieces"
    elif state in ("skewer", "skewering"):
        # A skewer is a pin the other way round: the enemy piece behind is worth less.
        hits = [t for t in board.attacks(square) if _behind_is_enemy(board, square, t, piece.color)]
        if not hits:
            return f"the {name} on {square_name} does not skewer anything"
    return None


def _behind_is_enemy(
    board: chess.Board, frm: chess.Square, through: chess.Square, mover: chess.Color
) -> bool:
    """Whether, past an enemy piece on `through`, the same line holds another enemy piece."""
    occupant = board.piece_at(through)
    if occupant is None or occupant.color == mover:
        return False
    df = _sign(chess.square_file(through) - chess.square_file(frm))
    dr = _sign(chess.square_rank(through) - chess.square_rank(frm))
    if df == 0 and dr == 0:
        return False
    file_, rank_ = chess.square_file(through) + df, chess.square_rank(through) + dr
    while 0 <= file_ <= 7 and 0 <= rank_ <= 7:
        behind = board.piece_at(chess.square(file_, rank_))
        if behind is not None:
            return behind.color != mover
        file_ += df
        rank_ += dr
    return False


def check_draft(
    text: str, board: chess.Board, *, after: Sequence[chess.Board] = ()
) -> list[DraftProblem]:
    """Everything concrete in a coach draft that the board contradicts. Empty means clean.

    A "<piece> on <square>" phrase must match the board, colour word included ("your" is
    the side to move). A tactic claimed of such a piece (pinned, hanging, safe, a fork, a
    skewer) must hold on the board, or on one of the `after` positions, which are the boards
    after the best move: a fork the best move creates is real even though the current board
    does not show it yet. A move must be playable: legal now, or legal for the opponent
    after a pass, since a coach may name a threat. Bare squares are never flagged; a coach
    may point at one.
    """
    problems: list[DraftProblem] = []
    seen: set[str] = set()
    frames = [board, *after]
    us = board.turn

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
            if all(reasons):
                add(match.group(0), reasons[0] or "")

    for match in _TACTIC.finditer(text):
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
        reason = _move_problem(board, core)
        if reason:
            add(token, reason)
    for match in _CASTLE.finditer(text):
        reason = _move_problem(board, match.group(1).replace("0", "O"))
        if reason:
            add(match.group(0), reason)
    for phrase, reason in _strategic_problems(text, board):
        add(phrase, reason)
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


def _strategic_problems(text: str, board: chess.Board) -> list[tuple[str, str]]:
    """Concrete strategic claims the board contradicts: a file said open that has pawns, a
    pawn said passed that an enemy pawn still holds up. Only clear, checkable contradictions
    are flagged; a fuzzy claim (space, king safety, activity) is left alone.
    """
    us = board.turn
    problems: list[tuple[str, str]] = []
    for match in _OPEN_FILE.finditer(text):
        kind = (match.group("k1") or match.group("k2")).lower().replace(" ", "").replace("-", "")
        letter = (match.group("f1") or match.group("f2")).lower()
        pawns = [board.piece_at(sq) for sq in _file_squares(ord(letter) - ord("a"))]
        ours = any(p and p.piece_type == chess.PAWN and p.color == us for p in pawns)
        theirs = any(p and p.piece_type == chess.PAWN and p.color != us for p in pawns)
        if kind == "open" and (ours or theirs):
            problems.append((match.group(0), f"the {letter}-file is not open; a pawn stands on it"))
        elif kind == "halfopen" and ours:
            problems.append(
                (
                    match.group(0),
                    f"the {letter}-file is not half-open for you; you have a pawn there",
                )
            )
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
