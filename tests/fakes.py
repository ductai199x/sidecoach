"""Stand-ins for what the app talks to outside itself -- the engine, the model, the
microphone, the Lichess puzzle file -- so a test needs no Stockfish, no model server, no
sound card and no download."""

from __future__ import annotations

import queue
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chess

from sidecoach.analysis.engine import Engine
from sidecoach.analysis.score import game_over
from sidecoach.events import Line
from sidecoach.llm.base import ChatMessage, Completion

# What a quiet position gets from the fake engine when the test gave it nothing to say.
QUIET = [Line(moves=["a2a3"], depth=1, expected_score=0.5)]


class FakeEngine(Engine):
    """An engine that answers from a script: the lines for a position's FEN, else `default`,
    and nothing for a finished game. It remembers every position it was asked about, with
    the depth and the number of lines asked for, and never starts a process."""

    def __init__(
        self, answers: dict[str, list[Line]] | None = None, *, default: list[Line] = QUIET
    ) -> None:
        super().__init__(path=Path("fake"))
        self.answers = answers or {}
        self.default = default
        self.calls: list[str] = []
        self.depths: list[int | None] = []
        self.multipvs: list[int] = []
        # The moves each position came with: what the engine is told of its history.
        self.histories: list[list[str]] = []

    def analyse(self, board: chess.Board, **kwargs: Any) -> list[Line]:
        self.calls.append(board.fen())
        self.histories.append([move.uci() for move in board.move_stack])
        self.depths.append(kwargs.get("depth"))
        self.multipvs.append(int(kwargs.get("multipv") or 1))
        if game_over(board):
            return []
        return self.answers.get(board.fen(), self.default)

    def open(self) -> None:  # never starts a process
        return

    def close(self) -> None:
        return


class ScriptedProvider:
    """A model that answers from a script and then with `then`, or with `fail` raises as a
    server that is down does. It remembers every call: `calls` everything it was asked,
    `prompts` the system message and the messages of each."""

    name = "scripted"

    def __init__(
        self, replies: Sequence[str] = (), *, then: str = "Keep looking.", fail: bool = False
    ) -> None:
        self.replies = list(replies)
        self.then = then
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    @property
    def prompts(self) -> list[tuple[str, list[ChatMessage]]]:
        """Read from `calls`: to forget what was asked, clear `calls`."""
        return [(call["system"], call["messages"]) for call in self.calls]

    def healthcheck(self) -> tuple[bool, str]:
        return True, "scripted"

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> Completion:
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "temperature": temperature,
                "thinking": thinking,
            }
        )
        if self.fail:
            raise RuntimeError("server down")
        return Completion(text=self.replies.pop(0) if self.replies else self.then)


class FedMic:
    """A microphone whose frames the test puts in; quiet when it puts none."""

    def __init__(self) -> None:
        self.frames: queue.Queue = queue.Queue()
        self.detail = "fed by the test"
        self.dropped = 0
        self.opened = False

    def healthcheck(self) -> tuple[bool, str]:
        return True, self.detail

    def open(self) -> bool:
        self.opened = True
        return True

    def close(self) -> None:
        self.opened = False


# The Lichess puzzle database's columns (database.lichess.org, lichess_db_puzzle.csv).
PUZZLE_HEADER = (
    "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags"
)


def puzzle_file(path: Path, rows: Sequence[str]) -> Path:
    """A puzzle file in the database's format at `path`: the header, then `rows`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([PUZZLE_HEADER, *rows, ""]), encoding="utf-8")
    return path
