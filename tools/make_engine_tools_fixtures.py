"""Record what the desktop's engine tools answer (`coach/engine_tools.py`, desktop c80b64b and
ce06c4e), for the Kotlin port to be held to: every call on a few positions, with an engine that
plays the first legal moves (by UCI) so both sides see the same lines -- and, on one, the lines
the sheet shows for the board now and before the last move, which a move on them is read from.

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

from sidecoach.coach.engine_tools import SPECS, TOOL_ROUNDS, TOOL_WORDS, EngineTools  # noqa: E402
from sidecoach.events import Line  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"


class FirstLegal:
    """Plays the first legal moves by UCI, four plies, the mover a little ahead or behind by
    how many legal moves it has: the same on both sides."""

    def analyse(
        self, board: chess.Board, *, depth: int | None = None, best_only: bool = False
    ) -> list[Line]:
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


# The lines the sheet shows for Tai's puzzle after 3...Rxf1+ and before it (`known_now`,
# `known_before`): scores of their own, so a move read off them is told from a search.
KNOWN_NOW = [
    Line(moves=["h1g2", "f1g1", "g2f3", "g3g2", "d1b2"], depth=20, expected_score=0.08),
    Line(moves=["h1g2", "f1f2", "g2g3"], depth=20, expected_score=0.05),
]
KNOWN_BEFORE = [
    Line(moves=["e1f1", "h1g2", "f1g1"], depth=20, expected_score=0.9),
    Line(moves=["g3g2", "h1g2", "e1d1"], depth=20, expected_score=0.3),
]

# (the board now, the position before the last move, the player's side, the calls, and the
# sheet's lines for each)
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
            # Arguments as a model may write them, read as Python reads them (the parity audit,
            # 2026-09-29): a null square, a false or numeric move list, a null move, a null start.
            ("square", {"square": None}),
            ("analyse", {"moves": False}),
            ("analyse", {"moves": ["Kg2", None]}),
            ("analyse", {"moves": 5}),
            ("analyse", {"moves": ["Kg2"], "start": None}),
        ],
        [],
        [],
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
        [],
        [],
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
        [],
        [],
    ),
    (
        # The puzzle again, with the sheet's lines: a move on one is read from it and rated by
        # its score; any other is searched (desktop ce06c4e).
        "6k1/b1p2pp1/p7/1p6/1P6/P5pP/3R4/3N1r1K w - - 0 4",
        "6k1/b1p2pp1/p7/1p6/1P6/P5pP/3R4/3NrB1K b - - 2 3",
        chess.BLACK,
        [
            ("analyse", {"moves": []}),
            ("analyse", {"moves": ["Kg2"]}),
            ("analyse", {"moves": ["Kg2", "Rf2+"]}),
            # The whole of a line: nothing after it to read, so searched.
            ("analyse", {"moves": ["Kg2", "Rf2+", "Kxg3"]}),
            ("analyse", {"moves": [], "start": "before_last_move"}),
            ("analyse", {"moves": ["g2+"], "start": "before_last_move"}),
            ("analyse", {"moves": ["Re2"], "start": "before_last_move"}),
            # A start the tools do not know is the board now, and no sheet's lines.
            ("analyse", {"moves": ["Kg2"], "start": "elsewhere"}),
            ("square", {"square": "g1"}),
            ("square", {"square": "g1", "moves": ["Kg2"]}),
            ("square", {"square": "f2", "start": "before_last_move"}),
        ],
        KNOWN_NOW,
        KNOWN_BEFORE,
    ),
]


def line_json(line: Line) -> dict:
    return {"moves": list(line.moves), "depth": line.depth, "expected_score": line.expected_score}


def main() -> None:
    cases = []
    for now, before, player, calls, known_now, known_before in CASES:
        tools = EngineTools(
            chess.Board(now),
            chess.Board(before) if before else None,
            FirstLegal(),
            player,
            known_now=known_now,
            known_before=known_before,
        )
        out = []
        for name, arguments in calls:
            out.append({"name": name, "arguments": arguments, "answer": tools.run(name, arguments)})
        cases.append(
            {
                "now": now,
                "before": before,
                "player_white": player == chess.WHITE,
                "known_now": [line_json(line) for line in known_now],
                "known_before": [line_json(line) for line in known_before],
                "calls": out,
                "log": tools.log,
                "frames": [b.fen() for b in tools.frames],
            }
        )
    payload = {"tool_words": TOOL_WORDS, "tool_rounds": TOOL_ROUNDS, "specs": SPECS, "cases": cases}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "engine_tools.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"{sum(len(c['calls']) for c in cases)} calls -> {OUT / 'engine_tools.json'}")


if __name__ == "__main__":
    main()
