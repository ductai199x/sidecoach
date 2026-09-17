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

from sidecoach.analysis.engine import san_line
from sidecoach.coach.effects import effects_text
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
    "terms ('your Nf3', 'the line Bb5 a6 Ba4', 'if they take on e5'). If their words name "
    "nothing to examine, return an empty list. Answer in JSON only: "
    '{"probes": [{"label": str, "moves": [str]}]}'
)

_JSON = re.compile(r"\{.*\}", re.DOTALL)
# A move as a player types it, with or without a move number; "pass" is the player waiting.
_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{1,3}\.(?:\.\.)?\s*)?((?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h]x?[a-h][1-8]|[a-h][1-8]|O-O-O|O-O|0-0-0|0-0)(?:=[QRBN])?[+#]?)(?![A-Za-z0-9])"
)


class EngineLike(Protocol):
    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]: ...


@dataclass
class Probe:
    """One thing to examine: moves from the position, in the player's terms."""

    label: str
    moves: list[str]  # SAN, or "pass" for a null move (the opponent's move first)
    parsed: list[chess.Move] = field(default_factory=list)
    illegal: str | None = None  # the first move that could not be played, if any


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
                move = replay.parse_san(token)
            except ValueError:
                probe.illegal = token
                break
        probe.parsed.append(move)
        replay.push(move)
    return probe


def probes_by_words(text: str, board: chess.Board) -> list[Probe]:
    """Without a planner: the moves named in the words, as one line, as far as legal."""
    moves: list[str] = []
    replay = board.copy(stack=False)
    for match in _TOKEN.finditer(text):
        token = match.group(1).replace("0-0-0", "O-O-O").replace("0-0", "O-O")
        try:
            move = replay.parse_san(token)
        except ValueError:
            if not moves:
                continue
            break
        moves.append(token)
        replay.push(move)
    return (
        [parse_probe(san_line(board, [m.uci() for m in _replay(board, moves)]), moves, board)]
        if moves
        else []
    )


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
    head = f"Engine on {probe.label}, which you asked about:"
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
    parts.append(f"after {played} the expected score is {score:.2f} from your side")
    if best is not None:
        gap = best.expected_score - score
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
            "the engine's reply would be" if after.turn != user_colour else "the engine then plays"
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


def analyse_named(
    text: str,
    board: chess.Board,
    engine: EngineLike,
    *,
    user_colour: chess.Color,
    level: DisclosureLevel,
    planner: LLMProvider | None = None,
    depth: int = 14,
) -> list[str]:
    """The engine's word on what the player asked about: one line per probe, or none when
    their words name nothing to examine. The planner (a model) decides the probes; without
    one, the moves named in the words are examined as one line."""
    if board.turn != user_colour:
        return []  # only the position the player faces is theirs to ask about
    probes = (
        plan_probes(text, board, planner, user_colour=user_colour)
        if planner is not None
        else probes_by_words(text, board)
    )
    if not probes:
        return []
    before = engine.analyse(board, depth=depth)
    best = before[0] if before else None
    return [
        examine(p, board, engine, best, user_colour=user_colour, level=level, depth=depth)
        for p in probes
    ]
