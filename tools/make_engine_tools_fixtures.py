"""Record what the desktop's engine tools answer (`coach/engine_tools.py`, desktop c80b64b),
for the Kotlin port to be held to: every call on a few positions, with an engine that plays
the first legal moves (by UCI) so both sides see the same lines.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_engine_tools_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.coach.engine_tools import TOOL_ROUNDS, TOOL_WORDS, EngineTools  # noqa: E402
from sidecoach.events import Line  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"


class FirstLegal:
    """Plays the first legal moves by UCI, four plies, the mover a little ahead or behind by
    how many legal moves it has: the same on both sides."""

    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]:
        replay = board.copy(stack=False)
        moves: list[str] = []
        for _ in range(4):
            legal = sorted(replay.legal_moves, key=lambda m: m.uci())
            if not legal:
                break
            moves.append(legal[0].uci())
            replay.push(legal[0])
        if not moves:
            return []
        score = 0.3 + (board.legal_moves.count() % 7) / 10
        return [
            Line(moves=moves, depth=depth or 14, expected_score=score),
            Line(moves=moves[:2], depth=depth or 14, expected_score=score - 0.05),
        ]


# (the board now, the position before the last move, the player's side, the calls)
CASES = [
    (
        # Tai's puzzle after 3...Rxf1+: White to move, Kg2 the only move.
        "6k1/b1p2pp1/p7/1p6/1P6/P5pP/3R4/3N1r1K w - - 0 4",
        "6k1/b1p2pp1/p7/1p6/1P6/P5pP/3R4/3NrB1K b - - 2 3",
        chess.BLACK,
        [
            ("analyse", {"moves": []}),
            ("analyse", {"moves": ["4. Kg2", "4...Rg1+"]}),
            ("analyse", {"moves": ["Kg2, Rg1+ Kf3 g2"]}),
            ("analyse", {"moves": ["Kg2", "Qd8"]}),
            ("analyse", {"moves": ["Rxf1+"], "start": "before_last_move"}),
            ("analyse", {"moves": ["e1f1"], "start": "before_last_move"}),
            ("square", {"square": "g1", "moves": ["Kg2", "Rg1+"]}),
            ("square", {"square": "F1"}),
            ("square", {"square": "z9"}),
            ("square", {"square": "a7", "moves": ["Kg2"], "start": "now"}),
            ("fly", {}),
            ("analyse", {}),
        ],
    ),
    (
        # A mate on the move, and no last move to start from.
        "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
        None,
        chess.WHITE,
        [
            ("analyse", {"moves": ["Rd8#"]}),
            ("analyse", {"moves": [], "start": "before_last_move"}),
            ("square", {"square": "d8", "moves": ["Rd8#"]}),
        ],
    ),
    (
        # The Scholar's position: a pin, a hanging piece, a promotion-free middle game.
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
        chess.WHITE,
        [
            ("analyse", {"moves": ["Qxf7#"]}),
            ("analyse", {"moves": ["Qxe5+", "Qe7"]}),
            ("analyse", {"moves": ["Nf6"], "start": "before_last_move"}),
            ("square", {"square": "f7"}),
            ("square", {"square": "e5", "moves": ["d3"]}),
        ],
    ),
]


def main() -> None:
    cases = []
    for now, before, player, calls in CASES:
        tools = EngineTools(
            chess.Board(now), chess.Board(before) if before else None, FirstLegal(), player
        )
        out = []
        for name, arguments in calls:
            out.append({"name": name, "arguments": arguments, "answer": tools.run(name, arguments)})
        cases.append(
            {
                "now": now,
                "before": before,
                "player_white": player == chess.WHITE,
                "calls": out,
                "log": tools.log,
                "frames": [b.fen() for b in tools.frames],
            }
        )
    payload = {"tool_words": TOOL_WORDS, "tool_rounds": TOOL_ROUNDS, "cases": cases}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "engine_tools.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"{sum(len(c['calls']) for c in cases)} calls -> {OUT / 'engine_tools.json'}")


if __name__ == "__main__":
    main()
