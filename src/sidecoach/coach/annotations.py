"""Board marks that go with a coach message: arrows and highlights for what the words are about.

The marks come from the same disclosure-filtered fact bundle the message was written from,
never from the prose alone, so an arrow cannot give away more than the words do: at the
question level there is nothing to draw; at the region level a tint on the region; at the
piece level the focus squares and a ring on the piece; at the candidate level an arrow for
the move; at the line level the line. A message about a move already played -- where the
disclosure rules allow the best move to be named -- gets the played move coloured by how it
scored and a green arrow for the better one -- two arrows, no more.

A second, smaller source is the finished text: the squares and moves the coach actually
named. Those passed the leak check, so drawing them reveals nothing new, and it is the
literal link between a sentence and the board.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import chess

from sidecoach.coach.disclosure import PositionFacts, squares_in, visible_facts
from sidecoach.events import Annotation, DisclosureLevel, Trigger

# The played move's arrow takes the colour of how the move scored; a move that scored fine
# is "played" (neutral) rather than green, so green always means "the better move".
_SEVERITY_COLOR = {
    "none": "played",
    "inaccuracy": "warn",
    "mistake": "bad",
    "blunder": "bad",
}
# How far into a line to draw, and how the continuation fades.
_LINE_PLIES = 4
_LINE_OPACITY = (0.9, 0.55, 0.4, 0.3)

# A SAN token in prose (a piece move, a pawn move, a capture, a promotion, castling).
_SAN = re.compile(
    r"(?<![A-Za-z0-9])((?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?)(?![A-Za-z0-9])"
)

_FILES = "abcdefgh"


def _region_squares(region: str) -> list[str]:
    """The squares a region name covers, for a light tint: "the kingside, around f7"."""
    name = region.lower()
    if "kingside" in name:
        files = "efgh"
    elif "queenside" in name:
        files = "abcd"
    elif "centre" in name or "center" in name:
        return ["d4", "d5", "e4", "e5"]
    else:
        return []
    return [f + r for f in files for r in "12345678"]


def _parse(board: chess.Board, san: str) -> chess.Move | None:
    try:
        return board.parse_san(san)
    except (ValueError, AssertionError):
        return None


def _arrow(board: chess.Board, san: str, color: str, opacity: float = 1.0) -> Annotation | None:
    move = _parse(board, san)
    if move is None:
        return None
    return Annotation(
        kind="arrow",
        squares=[chess.square_name(move.from_square), chess.square_name(move.to_square)],
        color=color,
        opacity=opacity,
        label=san,
    )


def _line_arrows(board: chess.Board, sans: Iterable[str], color: str) -> list[Annotation]:
    """Arrows for the first plies of a line, fading, each on the board it is played from."""
    out: list[Annotation] = []
    work = board.copy(stack=False)
    for i, san in enumerate(list(sans)[:_LINE_PLIES]):
        move = _parse(work, san)
        if move is None:
            break
        out.append(
            Annotation(
                kind="arrow",
                squares=[chess.square_name(move.from_square), chess.square_name(move.to_square)],
                color=color,
                opacity=_LINE_OPACITY[min(i, len(_LINE_OPACITY) - 1)],
                label=san,
            )
        )
        work.push(move)
    return out


def annotations_for(
    facts: PositionFacts,
    level: DisclosureLevel,
    board: chess.Board,
    *,
    trigger: Trigger,
    played_san: str | None = None,
    severity: str | None = None,
) -> list[Annotation]:
    """The marks for a message written from `facts` at `level`, on `board` (the position the
    facts describe: the one a played move was played from, or the live one for a hint)."""
    visible = visible_facts(facts, level)
    marks: list[Annotation] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()

    def add(mark: Annotation | None) -> None:
        if mark is None:
            return
        key = (mark.kind, tuple(mark.squares))
        if key in seen:
            return
        seen.add(key)
        marks.append(mark)

    about_played = trigger in (Trigger.POST_MOVE, Trigger.POST_MISTAKE) and played_san
    if about_played:
        # Two arrows and no more: what was played, and what was better. The line behind
        # the better move is only drawn when the user asks for the line.
        assert played_san is not None
        add(_arrow(board, played_san, _SEVERITY_COLOR.get(severity or "none", "good")))
        better = (visible.get("candidates") or [None])[0]
        if better and better != played_san:
            add(_arrow(board, better, "good"))
        return marks

    # A hint about the live position: exactly what the level lets the words say.
    region = visible.get("region")
    if region:
        for square in _region_squares(region):
            add(Annotation(kind="square", squares=[square], color="region", opacity=0.35))
        for square in sorted(squares_in(region)):
            add(Annotation(kind="square", squares=[square], color="focus", opacity=0.6))
    for square in visible.get("focus") or []:
        add(Annotation(kind="square", squares=[square], color="focus"))
    piece = visible.get("piece")
    if piece:
        for square in sorted(squares_in(piece)):
            add(Annotation(kind="ring", squares=[square], color="focus"))
    for text in visible.get("ideas") or []:
        # Sorted: `squares_in` returns a set of strings, and CPython randomises string
        # hashing per process, so the marks came out in a different order on every run.
        for square in sorted(squares_in(text)):
            add(Annotation(kind="square", squares=[square], color="mention", opacity=0.6))
    for san in (visible.get("threats") or [])[:2]:
        # A threat is the opponent's move, read after a null move.
        after_pass = board.copy(stack=False)
        after_pass.push(chess.Move.null())
        add(_arrow(after_pass, san, "threat", 0.8))
    if "line" in visible and visible.get("line"):
        for mark in _line_arrows(board, visible["line"], "line"):
            add(mark)
    elif "candidates" in visible and visible.get("candidates"):
        add(_arrow(board, visible["candidates"][0], "good"))
    return marks


def mentions_in(text: str, board: chess.Board) -> list[Annotation]:
    """Marks for the squares and moves the finished message names, so the sentence and
    the board point at the same things. Moves are read on the message's board; a token that
    is not a legal move there is treated as a bare square."""
    marks: list[Annotation] = []
    seen: set[str] = set()
    landed: set[str] = set()  # squares an arrow already points at: no highlight on top
    for match in _SAN.finditer(text):
        san = match.group(1)
        move = _parse(board, san)
        if move is not None:
            key = "m:" + move.uci()
            if key not in seen:
                seen.add(key)
                to = chess.square_name(move.to_square)
                landed.add(to)
                marks.append(
                    Annotation(
                        kind="arrow",
                        squares=[chess.square_name(move.from_square), to],
                        color="mention",
                        opacity=0.7,
                        label=san,
                    )
                )
    for square in sorted(squares_in(text) - landed):
        marks.append(Annotation(kind="square", squares=[square], color="mention", opacity=0.5))
    return marks
