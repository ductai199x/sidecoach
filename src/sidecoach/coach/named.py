"""The engine on what the player asks about, so the coach analyses it rather than guesses.

"What about Nf3?" once got "not among the engine's best moves; it prefers Bb5" -- true,
and all the object had. The player may name one move, a whole line, several alternatives,
a move of the opponent's ("what if he takes on f7?"), or an idea in words ("can I trade
queens?"). So the examination is agentic: the model decides what to examine (a list of
probes: sequences of moves from the position, labelled in the player's terms), the engine
examines each (the expected score after it from the player's side, how far behind the
engine's best, the engine's reply, what the first move does), and the results go into the
object before the coach answers. Every number is the engine's; every effect is computed on
the board. Below LINE the engine's own preferred move is not named: the line the player
asked about is theirs to hear about in full, the best line is not.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

import chess

from sidecoach.analysis import exchange
from sidecoach.analysis.engine import san_line
from sidecoach.coach.effects import effects_text
from sidecoach.coach.facts import PIECE_NAMES, standing_words
from sidecoach.coach.inventory import name_of, pin_target
from sidecoach.events import DisclosureLevel, Line
from sidecoach.llm.base import ChatMessage, LLMProvider

PLAN_SYSTEM = (
    "A chess player, in the middle of a game, asks their coach about a move, a line, or an "
    "idea. You turn their words into what the engine should examine: a list of probes, each a "
    "sequence of moves in standard notation from the position given, played in order, "
    "starting with the side to move. If they ask what the opponent might do, the probe "
    "starts with the word 'pass' (the player waits) and then the opponent's move; if they "
    "say their own move first and then the opponent's ('after I castle, he plays d5'), the "
    "probe is their move and then the opponent's, in that order. An idea in "
    "words ('trade queens', 'take on f7', 'castle') becomes the moves that do it. Several "
    "alternatives are several probes, at most four. Label each probe in the player's own "
    "terms ('your Nf3', 'the line Bb5 a6 Ba4', 'if they take on e5'). A question about a "
    "piece or a square rather than a move ('is my knight safe?', 'what does the bishop on "
    "d3 do?', 'what about e5?') is a probe with a square instead of moves. If their words "
    "name nothing to examine, return an empty list. Answer in JSON only: "
    '{"probes": [{"label": str, "moves": [str]} | {"label": str, "square": str}]}'
)

_JSON = re.compile(r"\{.*\}", re.DOTALL)
# A move as a player types it, with or without a move number; "pass" is the player waiting.
_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{1,3}\.(?:\.\.)?\s*)?((?:[a-h][1-8][a-h][1-8][qrbn]?|[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h]x?[a-h][1-8]|[a-h][1-8]|O-O-O|O-O|0-0-0|0-0)(?:=[QRBN])?[+#]?)(?![A-Za-z0-9])"
)
_UCI = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")


# A piece named with its square ("the white bishop on d3", "my knight at f6") or a bare
# square asked about ("what about e5", "square d5").
_PIECE_ON = re.compile(
    r"(?:(?P<colour>white|black|my|your|their|his|her)\s+)?(?P<piece>king|queen|rook|bishop|knight|pawn)s?\s+(?:on|at)\s+(?P<square>[a-h][1-8])\b",
    re.IGNORECASE,
)
_SQUARE_ASKED = re.compile(r"\b(?:square|on|about|at)\s+(?P<square>[a-h][1-8])\b", re.IGNORECASE)


class EngineLike(Protocol):
    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]: ...


@dataclass
class Probe:
    """One thing to examine: moves from the position, in the player's terms -- or a piece
    or square, when that is what they asked about."""

    label: str
    moves: list[str]  # SAN, or "pass" for a null move (the opponent's move first)
    parsed: list[chess.Move] = field(default_factory=list)
    illegal: str | None = None  # the first move that could not be played, if any
    square: chess.Square | None = None  # a piece or square probe instead of moves


def parse_probe(label: str, moves: list[str], board: chess.Board) -> Probe:
    """The probe's moves played on the board in order; an illegal one ends the probe and
    is remembered, so the coach can say so."""
    probe = Probe(label=label, moves=list(moves))
    replay = board.copy(stack=False)
    for token in moves:
        token = token.strip().replace("0-0-0", "O-O-O").replace("0-0", "O-O")
        if token.lower() == "pass":
            if replay.is_check():
                probe.illegal = "pass (in check)"
                break
            move = chess.Move.null()
        else:
            try:
                move = _parse(replay, token)
            except ValueError:
                probe.illegal = token
                break
        probe.parsed.append(move)
        replay.push(move)
    return probe


def probes_by_words(text: str, board: chess.Board) -> list[Probe]:
    """Without a planner: the moves named in the words, as one line, as far as legal; or
    the pieces and squares named, one probe each."""
    pieces = pieces_by_words(text, board)
    if pieces:
        return pieces
    moves: list[str] = []
    replay = board.copy(stack=False)
    for match in _TOKEN.finditer(text):
        token = match.group(1).replace("0-0-0", "O-O-O").replace("0-0", "O-O")
        try:
            move = _parse(replay, token)
        except ValueError:
            if not moves:
                continue
            break
        moves.append(replay.san(move))
        replay.push(move)
    return (
        [parse_probe(san_line(board, [m.uci() for m in _replay(board, moves)]), moves, board)]
        if moves
        else []
    )


def pieces_by_words(text: str, board: chess.Board) -> list[Probe]:
    """The pieces named with their square, or the squares asked about, as probes."""
    out: list[Probe] = []
    seen: set[chess.Square] = set()
    for m in _PIECE_ON.finditer(text):
        square = chess.parse_square(m.group("square").lower())
        if square in seen:
            continue
        seen.add(square)
        out.append(Probe(label=m.group(0).strip(), moves=[], square=square))
    if out:
        return out[:4]
    for m in _SQUARE_ASKED.finditer(text):
        square = chess.parse_square(m.group("square").lower())
        if square in seen:
            continue
        seen.add(square)
        out.append(Probe(label=chess.square_name(square), moves=[], square=square))
    return out[:4]


def _controller(board: chess.Board, at: chess.Square, square: chess.Square) -> str:
    """The piece on `at` as it controls `square`: marked pinned when it is pinned to its king
    off the line through `square` -- it still keeps the enemy king off, but takes nothing
    that lands there."""
    colour = board.color_at(at)
    if colour is not None and square not in board.pin(colour, at):
        return f"{name_of(board, at)} (pinned)"
    return name_of(board, at)


def examine_piece(probe: Probe, board: chess.Board, *, user_colour: chess.Color) -> str:
    """The board on one piece or square: what stands there, where it can go, what it
    attacks and defends, what attacks and defends it, whether it is pinned. Static facts
    of the position, no search. Attack and defence are legal captures and recaptures
    (`sidecoach.analysis.exchange`): a piece pinned to its king attacks and defends nothing
    off its line, and a king does not take back where the other side still guards."""
    assert probe.square is not None
    square = probe.square
    head = f"Engine on {probe.label}, which the player asked about:"
    piece = board.piece_at(square)
    if piece is None:
        holders = sorted(board.attackers(user_colour, square))
        theirs = sorted(board.attackers(not user_colour, square))
        parts = [f"{chess.square_name(square)} is empty"]
        if holders:
            parts.append(
                "the player controls it with the "
                + " and the ".join(_controller(board, s, square) for s in holders[:3])
            )
        if theirs:
            parts.append(
                "the opponent controls it with the "
                + " and the ".join(_controller(board, s, square) for s in theirs[:3])
            )
        if not holders and not theirs:
            parts.append("nobody controls it")
        return f"{head} " + "; ".join(parts) + "."
    whose = "the player's" if piece.color == user_colour else "the opponent's"
    name = f"{whose} {PIECE_NAMES[piece.piece_type]} on {chess.square_name(square)}"
    parts = [f"{name} stands there"]
    # Where it can go, on the board as it stands, with its own side to move. Never onto the
    # enemy king: with the turn handed to the side giving check, the generator offers it.
    view = board.copy(stack=False)
    view.turn = piece.color
    moves = [
        m
        for m in view.legal_moves
        if m.from_square == square and view.piece_type_at(m.to_square) != chess.KING
    ]
    if moves:
        parts.append(
            "it can go to " + ", ".join(sorted({chess.square_name(m.to_square) for m in moves}))
        )
    else:
        parts.append("it has no legal move")
    # It attacks what it could take among those moves, and the enemy king when it gives
    # check (a pinned piece still does); it defends the pieces it could legally take back on.
    attacks = [
        t
        for t in board.attacks(square)
        if (o := board.piece_at(t))
        and o.color != piece.color
        and (o.piece_type == chess.KING or any(m.to_square == t for m in moves))
    ]
    defends = [
        t
        for t in board.attacks(square)
        if (o := board.piece_at(t))
        and o.color == piece.color
        and o.piece_type != chess.KING
        and square in exchange.defenders(board, piece.color, t)
    ]
    if attacks:
        parts.append("it attacks the " + " and the ".join(name_of(board, t) for t in attacks[:4]))
    if defends:
        parts.append("it defends the " + " and the ".join(name_of(board, t) for t in defends[:4]))
    # A king is attacked by what gives it check, pinned or not; any other piece by what can
    # legally take it.
    attackers = (
        sorted(board.attackers(not piece.color, square))
        if piece.piece_type == chess.KING
        else exchange.capturers(board, not piece.color, square)
    )
    defenders = exchange.defenders(board, piece.color, square)
    if attackers:
        parts.append(
            "it is attacked by the " + " and the ".join(name_of(board, s) for s in attackers[:3])
        )
    else:
        parts.append("nothing attacks it")
    if piece.piece_type != chess.KING:
        if defenders:
            parts.append(
                "it is defended by the "
                + " and the ".join(name_of(board, s) for s in defenders[:3])
            )
        else:
            parts.append("nothing defends it")
        pin = pin_target(board, square)
        if pin is not None:
            parts.append(
                f"it is pinned to the {name_of(board, pin[1])} by the {name_of(board, pin[0])}"
            )
    return f"{head} " + "; ".join(parts) + "."


def _parse(board: chess.Board, token: str) -> chess.Move:
    """A move as the player wrote it: SAN, or UCI ("e2e4"); ValueError when illegal."""
    if _UCI.match(token.lower()):
        move = chess.Move.from_uci(token.lower())
        if move not in board.legal_moves:
            raise ValueError(token)
        return move
    return board.parse_san(token)


def examine_static(probe: Probe, board: chess.Board) -> str:
    """Without an engine: the position the probe's moves lead to, and what the first move
    does. No score; the FEN so the position can be read."""
    head = f"Engine on {probe.label}, which the player asked about:"
    if not probe.parsed:
        return f"{head} {probe.illegal} is not a legal move here."
    after = board.copy(stack=False)
    for move in probe.parsed:
        after.push(move)
    played = (
        san_line(board, [m.uci() for m in probe.parsed if m != chess.Move.null()])
        if all(m != chess.Move.null() for m in probe.parsed)
        else " ".join(probe.moves[: len(probe.parsed)])
    )
    parts = [head]
    if probe.illegal:
        parts.append(f"after {played}, {probe.illegal} is not a legal move;")
    parts.append(f"after {played} the position is {after.fen()}")
    if after.is_game_over():
        parts[-1] += f" (the game is over: {after.result()})"
    first = next((m for m in probe.parsed if m != chess.Move.null()), None)
    if first is not None and probe.parsed[0] != chess.Move.null():
        does = effects_text(board, first, hazards=True)
        if does:
            parts[-1] += f"; what {board.san(first)} does: {does}"
    return " ".join(parts) + "."


def _replay(board: chess.Board, sans: list[str]) -> list[chess.Move]:
    out: list[chess.Move] = []
    replay = board.copy(stack=False)
    for san in sans:
        move = replay.parse_san(san)
        out.append(move)
        replay.push(move)
    return out


def plan_probes(
    text: str, board: chess.Board, provider: LLMProvider, *, user_colour: chess.Color
) -> list[Probe]:
    """The model's reading of what the player wants examined, as probes on the board.
    Falls back to the moves named in the words when the model gives nothing usable."""
    side = "White" if board.turn == chess.WHITE else "Black"
    player = "White" if user_colour == chess.WHITE else "Black"
    content = (
        f"Position (FEN): {board.fen()}\n{side} to move. The player is {player}.\n"
        f"Legal moves: {' '.join(sorted(board.san(m) for m in board.legal_moves))}\n\n"
        f'The player says: "{text.strip()}"\n\nJSON only.'
    )
    messages: list[ChatMessage] = [{"role": "user", "content": content}]
    try:
        completion = provider.chat(PLAN_SYSTEM, messages, max_tokens=400, temperature=0.0)
        found = _JSON.search(completion.text or "")
        data = json.loads(found.group(0)) if found else {}
    except Exception:  # noqa: BLE001
        data = {}
    probes: list[Probe] = []
    for item in (data.get("probes") if isinstance(data, dict) else None) or []:
        if not isinstance(item, dict):
            continue
        square = str(item.get("square") or "").strip().lower()
        if square and re.fullmatch(r"[a-h][1-8]", square):
            probes.append(
                Probe(
                    label=str(item.get("label") or square),
                    moves=[],
                    square=chess.parse_square(square),
                )
            )
        else:
            moves = [str(m) for m in item.get("moves", []) if str(m).strip()]
            if not moves:
                continue
            probes.append(parse_probe(str(item.get("label") or " ".join(moves)), moves, board))
        if len(probes) >= 4:
            break
    return probes or probes_by_words(text, board)


def examine(
    probe: Probe,
    board: chess.Board,
    engine: EngineLike,
    best: Line | None,
    *,
    user_colour: chess.Color,
    level: DisclosureLevel,
    depth: int = 14,
) -> str:
    """The engine's word on one probe, one line of text for the object."""
    head = f"Engine on {probe.label}, which the player asked about:"
    if not probe.parsed:
        return f"{head} {probe.illegal} is not a legal move here."
    after = board.copy(stack=False)
    for move in probe.parsed:
        after.push(move)
    played = (
        san_line(board, [m.uci() for m in probe.parsed if m != chess.Move.null()])
        if all(m != chess.Move.null() for m in probe.parsed)
        else " ".join(probe.moves[: len(probe.parsed)])
    )
    parts = [head]
    if probe.illegal:
        parts.append(f"after {played}, {probe.illegal} is not a legal move;")
    if after.is_game_over():
        parts.append(f"the game is over after {played} ({after.result()}).")
        return " ".join(parts)
    lines = engine.analyse(after, depth=depth)
    if not lines:
        return f"{head} the engine has nothing to say about the position after {played}."
    score = lines[0].expected_score if after.turn == user_colour else 1.0 - lines[0].expected_score
    parts.append(f"after {played} the expected score is {score:.2f} from the player's side")
    # Their move first ("if they take on b4") is the player passing, which is no move of
    # theirs: measured against the best it came out "after pass axb4 you are losing, a
    # serious step down from the best (the best is Bxc3)" (the simulated students).
    passing = bool(probe.parsed) and probe.parsed[0] == chess.Move.null()
    if best is not None and not passing:
        gap = best.expected_score - score
        # The engine's own first move, searched again after it is played, comes back a hair
        # off its first score: "12. Nc3: ... a hair behind the best (the best is Nc3)" (the
        # simulated students, 2026-09-26). The best move is the best.
        if best.moves and len(probe.parsed) == 1 and probe.parsed[0].uci() == best.moves[0]:
            gap = 0.0
        if gap > 0.005:
            parts[-1] += f", {gap:.2f} behind the engine's best"
            if level >= DisclosureLevel.LINE and best.moves:
                parts[-1] += f" ({board.san(chess.Move.from_uci(best.moves[0]))})"
        elif gap < -0.005:
            parts[-1] += ", better than the engine's first line"
        else:
            parts[-1] += ", as good as the engine's best"
    parts[-1] += ";"
    reply_plies = None if level >= DisclosureLevel.LINE else 2
    reply = san_line(after, lines[0].moves, limit=reply_plies)
    if reply:
        whose = (
            "the opponent's reply would be"
            if after.turn != user_colour
            else "the engine then plays"
        )
        parts.append(f"{whose} {reply};")
    first = next((m for m in probe.parsed if m != chess.Move.null()), None)
    if first is not None and probe.parsed[0] != chess.Move.null():
        does = effects_text(board, first, hazards=True)
        if does:
            parts.append(f"what {board.san(first)} does: {does}.")
            return " ".join(parts)
    parts[-1] = parts[-1].rstrip(";") + "."
    return " ".join(parts)


_ENGINE_ON = re.compile(
    r"^Engine on (?P<label>.+?), which the player asked about: (?P<body>.*)$", re.DOTALL
)
_PROBE_LINE = re.compile(r"after (?P<moves>.+?) the expected score is")
_REPLY_LINE = re.compile(
    r"(?:the opponent's reply would be|the engine then plays) (?P<moves>[^;]+)"
)


def disclosed_frames(board: chess.Board, facts: list[str]) -> list[chess.Board]:
    """The positions along each examination: after every move of the probe, then after
    every move of the engine's reply. The coach may recite what it was shown, so a move
    or a piece that is real only further along that line is checked there, not faulted."""
    frames: list[chess.Board] = []
    for fact in facts:
        m = _ENGINE_ON.match(fact)
        if not m:
            continue
        body = m.group("body")
        replay = board.copy(stack=False)
        probe = _PROBE_LINE.search(body)
        tokens = probe.group("moves").split() if probe else []
        reply = _REPLY_LINE.search(body)
        if reply:
            tokens += reply.group("moves").rstrip(". ").split()
        for token in tokens:
            token = re.sub(r"^\d{1,3}\.(?:\.\.)?", "", token).strip()
            if not token:
                continue
            if token.lower() == "pass":
                move = chess.Move.null()
            else:
                try:
                    move = replay.parse_san(token)
                except ValueError:
                    break
            replay = replay.copy(stack=False)
            replay.push(move)
            frames.append(replay)
    return frames


def spoken(fact: str) -> str:
    """The examination as the coach says it out loud, the same facts in plainer words:
    "Bc4: after 3. Bc4 it is about equal, a hair behind the best; they would answer
    3... Bc5 4. d3; it attacks the pawn on f7." Numbers become words; nothing is added."""
    m = _ENGINE_ON.match(fact)
    if not m:
        return fact
    label, body = m.group("label"), m.group("body").strip()
    body = re.sub(
        r"^after (.+?) the expected score is (\d\.\d\d) from the player's side",
        lambda mm: (
            f"if they got to play {mm.group(1)[5:]} now, {_standing(float(mm.group(2)))}"
            if mm.group(1).startswith("pass ")
            else f"after {mm.group(1)} {_standing(float(mm.group(2)))}"
        ),
        body,
    )
    body = re.sub(
        r", (\d\.\d\d) behind the engine's best(?: \(([^)]+)\))?",
        lambda mm: (
            ", "
            + _gap_words(float(mm.group(1)))
            + (f" (the best is {mm.group(2)})" if mm.group(2) else "")
        ),
        body,
    )
    body = body.replace(", as good as the engine's best", ", as good as the best move").replace(
        ", better than the engine's first line", ", better than the engine's first choice"
    )
    body = body.replace("the opponent's reply would be", "they would answer").replace(
        "the engine then plays", "then the engine plays"
    )
    body = re.sub(r"what (\S+) does: ", r"\1 ", body)
    # A piece or a square examined (`examine_piece`) is said to the player, not about them:
    # "f3: the player's knight on f3 stands there; it can go to ..." went out as it was
    # written for the model (a simulated student's session, 2026-09-26).
    body = (
        body.replace("the player's", "your")
        .replace("the opponent's", "their")
        .replace("the player controls", "you control")
        .replace("the opponent controls", "they control")
    )
    piece = re.match(r"^(your|their) (\w+) on ([a-h][1-8]) stands there; it ", body)
    if piece:
        whose, name, square = piece.groups()
        return f"{whose.capitalize()} {name} on {square} " + body[piece.end() :]
    if re.match(r"^[a-h][1-8] is empty", body):
        return body  # names its square itself
    return f"{label}: {body}"


_standing = standing_words


def _gap_words(gap: float) -> str:
    if gap < 0.02:
        return "a hair behind the best"
    if gap < 0.05:
        return "a little behind the best"
    if gap < 0.15:
        return "clearly behind the best"
    return "a serious step down from the best"


def analyse_named(
    text: str,
    board: chess.Board,
    engine: EngineLike | None,
    *,
    user_colour: chess.Color,
    level: DisclosureLevel,
    planner: LLMProvider | None = None,
    depth: int = 14,
    static: bool = False,
    pieces_board: chess.Board | None = None,
) -> list[str]:
    """The engine's word on what the player asked about: one line per probe, or none when
    their words name nothing to examine. The planner (a model) decides the probes; without
    one, the moves named in the words are examined as one line, or the pieces and squares
    named each on their own. A piece or square needs no engine; a move without one is
    dropped, unless `static`, when it gets the position it leads to and what it does,
    without a score (a FEN is for a reader of the object, not for a coach to say)."""
    probes = (
        plan_probes(text, board, planner, user_colour=user_colour)
        if planner is not None
        else probes_by_words(text, board)
    )
    if (board.turn != user_colour and engine is not None) or (engine is None and not static):
        # Only the position the player faces is theirs to ask lines about; the board's
        # pieces are theirs to ask about at any time.
        probes = [p for p in probes if p.square is not None]
    if not probes:
        return []
    out: list[str] = []
    best: Line | None = None
    if engine is not None and any(p.square is None for p in probes):
        before = engine.analyse(board, depth=depth)
        best = before[0] if before else None
    for p in probes:
        if p.square is not None:
            # A piece or square is read on the board as it stands (`pieces_board`), even when
            # the question is about the position a move was played from: "b5 is empty" was
            # said with the player's bishop on b5 (the students' round 6).
            out.append(examine_piece(p, pieces_board or board, user_colour=user_colour))
        elif engine is None:
            out.append(examine_static(p, board))
        else:
            out.append(
                examine(p, board, engine, best, user_colour=user_colour, level=level, depth=depth)
            )
    return out
