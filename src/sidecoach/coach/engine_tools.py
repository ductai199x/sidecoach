"""The engine at the coach model's call, while it answers a question of the player's.

The coach's request carries what was computed before the question came: the engine's lines
to eight plies, the facts, the lists of what stands on the board. Asked "ok and after
that?" past the end of the line, or "whose bishop? ... he will just move away to the right,
no?", the model had nothing more to go on; a separate model call guessed beforehand what to
examine and guessed wrong ("is my bishop safe?"), and the answers came out repeated or
invented (Tai's puzzle, 2026-09-28). With these tools the model asks the engine itself, from
the whole conversation, for what it needs: the position after any moves -- who stands
better and the engine's lines with what each move does -- and what attacks a square.

Everything returned is the engine's and the board's, in the harness's words ("the player",
"the opponent", expected scores from the player's side). The boards the tools went through
are kept (`frames`): the draft check then reads a reply about them against them.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

import chess

from sidecoach.analysis.engine import san_line
from sidecoach.coach.effects import line_notes
from sidecoach.coach.facts import PIECE_NAMES, standing_words
from sidecoach.coach.inventory import inventory_lines
from sidecoach.events import Line

# Rounds of calls a reply may make before it must answer: one to look, one to look further.
TOOL_ROUNDS = 3
# What the Task says of them.
TOOL_WORDS = (
    "You can ask the engine yourself, with the tools: the position after any moves you give "
    "it (who stands better, its best lines, what each move does) and what attacks a square. "
    "Ask it for what the messages above do not show -- what comes after the end of a line, "
    "a move they propose, whose piece a move takes, whether a square is safe -- before you "
    "answer, and say only what it returns."
)
# The engine lines an analysis gives, and the plies each.
LINES = 3
LINE_PLIES = 8

_START = {
    "type": "string",
    "enum": ["now", "before_last_move"],
    "description": "now (the default): the board as it stands; before_last_move: the "
    "position the last move was played from",
}
_MOVES = {
    "type": "array",
    "items": {"type": "string"},
    "description": "Moves in standard notation, in order and for both sides, played from "
    "the start position; empty for the start position itself.",
}
SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "analyse",
            "description": "Stockfish on the position reached by playing `moves` from the "
            "board now (or from before the last move): whose move it is, who stands better, "
            "the engine's best lines with what each move does, and what hangs, is pinned or "
            "can be taken there. When moves are given, also how the first one rates against "
            "the engine's best. For what the messages do not already show: what comes after "
            "the end of a line, a move the player proposes, what a move would allow.",
            "parameters": {
                "type": "object",
                "properties": {"moves": _MOVES, "start": _START},
                "required": ["moves"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "square",
            "description": "What stands on a square and which pieces of each side attack it, "
            "in the position reached by playing `moves` (as for analyse): whether a piece "
            "there is defended, or a square is safe to go to.",
            "parameters": {
                "type": "object",
                "properties": {
                    "square": {"type": "string", "description": "a square, such as g1"},
                    "moves": _MOVES,
                    "start": _START,
                },
                "required": ["square"],
            },
        },
    },
]
# A move as people write it, with its number or without: "4. Kg2", "4...Rg1+", "Kg2".
_NUMBER = re.compile(r"^\d+\.+")


class EngineLike(Protocol):
    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]: ...


class EngineTools:
    """The tools of one request: the board now, the position before the last move, the
    engine, and whose side "the player" is."""

    def __init__(
        self,
        now: chess.Board,
        before: chess.Board | None,
        engine: EngineLike,
        player: chess.Color,
        *,
        depth: int = 14,
    ) -> None:
        self.now = now.copy(stack=False)
        self.before = before.copy(stack=False) if before is not None else None
        self.engine = engine
        self.player = player
        self.depth = depth
        self.frames: list[chess.Board] = []
        self.log: list[str] = []  # what was asked, one entry a call, for the record

    specs = SPECS

    def run(self, name: str, arguments: dict[str, Any]) -> str:
        """The tool's answer, as text for the model; a bad call is told what was wrong."""
        moves = _tokens(arguments.get("moves") or [])
        start = str(arguments.get("start") or "now")
        self.log.append(
            f"{name}({' '.join(moves)}"
            + (f"; {start}" if start != "now" else "")
            + (f"; {arguments.get('square')})" if name == "square" else ")")
        )
        board = self.before if start == "before_last_move" else self.now
        if board is None:
            return "There is no last move to start from: the tools start from the board now."
        if name == "analyse":
            return self._analyse(board, moves, start)
        if name == "square":
            return self._square(board, moves, str(arguments.get("square") or ""), start)
        return f"No tool named {name!r}: the tools are analyse and square."

    # ---- the tools ----------------------------------------------------------------------

    def _analyse(self, start: chess.Board, moves: list[str], where: str) -> str:
        after, played, illegal = self._play(start, moves)
        head = self._heading(start, played, where)
        if illegal:
            head += f" {illegal} is not a legal move there, so the analysis stops before it."
        if after.is_game_over(claim_draw=True):
            return f"{head} The game is over there: {_result(after)}."
        try:
            lines = self.engine.analyse(after, depth=self.depth)[:LINES]
            rating = self._rating(start, played[:1], _origin(where)) if played else ""
        except Exception as exc:  # noqa: BLE001 - the engine gone or a position it refused
            return f"{head} The engine could not look at that position ({type(exc).__name__})."
        parts = [
            head + f" {_side(after.turn)} to move" + (", in check." if after.is_check() else ".")
        ]
        if rating:
            parts.append(rating)
        if lines:
            score = self._from_player(lines[0].expected_score, after.turn)
            parts.append(
                f"Standing: {standing_words(score, third_person=True)} ({score:.2f} from the "
                "player's side)."
            )
            parts.append("The engine's lines, best first:")
            for i, line in enumerate(lines, start=1):
                moves_text = san_line(after, line.moves[:LINE_PLIES])
                score_i = self._from_player(line.expected_score, after.turn)
                text = f"  {i}. {moves_text} ({score_i:.2f})"
                if i == 1:
                    notes = line_notes(after, line.moves[:LINE_PLIES], player=self.player)
                    if notes:
                        text += f" -- move by move: {notes}"
                    self._walk(after, line.moves[:LINE_PLIES])
                parts.append(text)
        parts.append("On the board there (complete lists):")
        parts.extend(f"  {line}" for line in inventory_lines(after, self.player))
        return "\n".join(parts)

    def _square(self, start: chess.Board, moves: list[str], name: str, where: str) -> str:
        try:
            square = chess.parse_square(name.strip().lower())
        except ValueError:
            return f"{name!r} is not a square: give one such as g1."
        after, played, illegal = self._play(start, moves)
        head = self._heading(start, played, where)
        if illegal:
            head += f" {illegal} is not a legal move there, so this is the position before it."
        piece = after.piece_at(square)
        what = (
            f"{_owner(piece.color, self.player)} {PIECE_NAMES[piece.piece_type]}"
            if piece is not None
            else "nothing"
        )
        parts = [f"{head} On {chess.square_name(square)}: {what}."]
        for colour in (self.player, not self.player):
            attackers = [
                PIECE_NAMES[after.piece_type_at(at)]  # type: ignore[index]
                + f" on {chess.square_name(at)}"
                + (" (pinned)" if square not in after.pin(colour, at) else "")
                for at in after.attackers(colour, square)
            ]
            role = "defended" if piece is not None and piece.color == colour else "attacked"
            parts.append(
                f"{role.capitalize()} by {_owner(colour, self.player)}: "
                + (", ".join(attackers) if attackers else "nothing")
                + "."
            )
        return " ".join(parts)

    # ---- the board ----------------------------------------------------------------------

    def _play(
        self, start: chess.Board, moves: list[str]
    ) -> tuple[chess.Board, list[chess.Move], str]:
        """The board after `moves`, the moves played, and the first one that could not be."""
        board = start.copy(stack=False)
        played: list[chess.Move] = []
        for token in moves:
            try:
                move = board.parse_san(token)
            except ValueError:
                try:
                    move = chess.Move.from_uci(token)
                except ValueError:
                    return board, played, token
                if move not in board.legal_moves:
                    return board, played, token
            played.append(move)
            board.push(move)
            self.frames.append(board.copy(stack=False))
        return board, played, ""

    def _walk(self, board: chess.Board, ucis: Sequence[str]) -> None:
        """The boards along a line the model is shown, for the draft check."""
        replay = board.copy(stack=False)
        for uci in ucis:
            try:
                move = chess.Move.from_uci(uci)
            except ValueError:
                return
            if move not in replay.legal_moves:
                return
            replay.push(move)
            self.frames.append(replay.copy(stack=False))

    def _rating(self, start: chess.Board, first: list[chess.Move], origin: str) -> str:
        """How the first move rates against the engine's best from `start`, said as the first
        move's: "Kg2 is the engine's best move there", printed after the line's end, was read
        as the best move there ("White's best is 8. Kg2", Tai's puzzle)."""
        best = self.engine.analyse(start, depth=self.depth)
        if not best:
            return ""
        after = start.copy(stack=False)
        after.push(first[0])
        san = san_line(start, [first[0].uci()])
        head = f"The first move given, {san}, is"
        if best[0].moves and best[0].moves[0] == first[0].uci():
            return f"{head} the engine's best move from {origin}."
        if after.is_game_over(claim_draw=True):
            return ""
        reply = self.engine.analyse(after, depth=self.depth)
        if not reply:
            return ""
        gap = best[0].expected_score - (1.0 - reply[0].expected_score)
        top = san_line(start, best[0].moves[:1]) if best[0].moves else "?"
        if gap <= 0.005:
            return f"{head} as good as the engine's best from {origin} ({top})."
        return f"{head} {gap:.2f} behind the engine's best from {origin}, {top}."

    def _heading(self, start: chess.Board, played: list[chess.Move], where: str) -> str:
        if not played:
            return f"From {_origin(where)}:"
        return f"After {san_line(start, [m.uci() for m in played])} (from {_origin(where)}):"

    def _from_player(self, expected: float, to_move: chess.Color) -> float:
        return expected if to_move == self.player else 1.0 - expected


def _tokens(moves: Sequence[Any]) -> list[str]:
    """The moves as tokens: one move each, their numbers taken off ("4. Kg2" -> "Kg2")."""
    out: list[str] = []
    for item in moves if not isinstance(moves, str) else [moves]:
        for token in str(item).replace(",", " ").split():
            token = _NUMBER.sub("", token.strip())
            if token and token not in ("...", ".."):
                out.append(token.replace("0-0-0", "O-O-O").replace("0-0", "O-O"))
    return out


def _origin(where: str) -> str:
    return "the position before the last move" if where == "before_last_move" else "the board now"


def _side(colour: chess.Color) -> str:
    return "White" if colour == chess.WHITE else "Black"


def _owner(colour: chess.Color, player: chess.Color) -> str:
    return "the player's" if colour == player else "the opponent's"


def _result(board: chess.Board) -> str:
    if board.is_checkmate():
        return f"checkmate, {_side(not board.turn)} wins"
    if board.is_stalemate():
        return "stalemate, a draw"
    return "a draw"
