"""Record what the desktop's Analysis gives (`ui/review_analysis.py`), for the phone's port
(ui/ReviewAnalysis.kt), from the seven real reviews in `reviews.json` and random games:

- `positions`: positions along the games and a few ends (mate, stalemate) -> `position`.
- `moves`: moves from those positions, legal and not -> `move` or its error.
- `lines`: the engine's lines for positions (the reviews' own measurements, answered by a fake
  engine) -> `Lines.lines`.
- `explore`: the explorer over the games' rows -> `explore` at positions along them and off them.
- `pgn`: each game with random side lines grafted on, annotated (with the review's views and AI
  words written for some key moves) and plain -> `pgn`.
- `names`: rows with names a file cannot carry -> `pgn_name` in UTC.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_analysis_fixtures.py
"""

from __future__ import annotations

import dataclasses
import json
import random
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

from sidecoach.analysis.game_review import STRENGTHS, measured_from_json  # noqa: E402
from sidecoach.config import Settings  # noqa: E402
from sidecoach.profile import library  # noqa: E402
from sidecoach.profile.store import LibraryRow  # noqa: E402
from sidecoach.ui import review_analysis  # noqa: E402
from tests.fakes import FakeEngine  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets"
OUT = ROOT / "ui" / "analysis.json"
REVIEWS = ROOT / "profile" / "reviews.json"

ENDS = [
    "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",  # mated
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",  # stalemate
    "8/8/8/8/8/8/8/K1k5 w - - 0 1",  # bare kings: drawn by material
]


def row_json(row: LibraryRow) -> dict[str, Any]:
    return dataclasses.asdict(row)


def random_game(rng: random.Random, n: int) -> str:
    game = chess.pgn.Game()
    side = rng.choice(["white", "black"])
    game.headers["White"] = "nosegazm" if side == "white" else f"opp{n}"
    game.headers["Black"] = "nosegazm" if side == "black" else f"opp{n}"
    game.headers["Date"] = f"2026.0{rng.randrange(1, 10)}.1{rng.randrange(0, 10)}"
    board = chess.Board()
    node: chess.pgn.GameNode = game
    for _ in range(rng.randrange(6, 40)):
        moves = list(board.legal_moves)
        if not moves:
            break
        # Lean on the common first moves so the games share positions.
        if board.ply() < 4:
            common = [m for m in moves if m.uci() in ("e2e4", "d2d4", "e7e5", "d7d5", "g1f3", "c7c5")]
            moves = common or moves
        m = rng.choice(moves)
        node = node.add_variation(m)
        board.push(m)
    game.headers["Result"] = board.result() if board.is_game_over() else rng.choice(["1-0", "0-1", "1/2-1/2", "*"])
    return str(game)


def side_lines(rng: random.Random, game: chess.pgn.Game, mainline: list[str]) -> list[dict[str, Any]]:
    """The analysis board's tree: the game's own moves, with random lines off them (some
    within others), and one move that is not legal (left out)."""

    def chain(ucis: list[str]) -> list[dict[str, Any]]:
        return [{"uci": ucis[0], "children": chain(ucis[1:])}] if ucis else []

    root = chain(mainline)
    b = game.board()
    path = []
    for uci in mainline:
        path.append(b.copy())
        b.push(chess.Move.from_uci(uci))
    # Lines off random moves of the game: from the position before the move.
    for _ in range(rng.randrange(1, 4)):
        if not mainline:
            break
        at = rng.randrange(len(mainline))
        start = path[at].copy()
        line = []
        for _ in range(rng.randrange(1, 4)):
            moves = list(start.legal_moves)
            if not moves:
                break
            m = rng.choice(moves)
            line.append(m.uci())
            start.push(m)
        if not line or line[0] == mainline[at]:
            continue
        # Graft it beside the game's move at `at`.
        parent_children = root if at == 0 else _node_at(root, at - 1)["children"]
        parent_children.append(chain(line)[0])
        # Sometimes a line within the line.
        if len(line) > 1 and rng.random() < 0.5:
            inner = path[at].copy()
            inner.push(chess.Move.from_uci(line[0]))
            alt = [m.uci() for m in inner.legal_moves if m.uci() != line[1]]
            if alt:
                parent_children[-1]["children"].append({"uci": rng.choice(alt), "children": []})
    if mainline:
        root.append({"uci": "a1a8", "children": []})  # not legal at the start: left out
    return root


def _node_at(root: list[dict[str, Any]], index: int) -> dict[str, Any]:
    node = root[0]
    for _ in range(index):
        node = node["children"][0]
    return node


def main() -> None:
    rng = random.Random(20260930)
    data = json.loads(REVIEWS.read_text(encoding="utf-8"))["games"]
    rows: list[LibraryRow] = []
    for g in data:
        rows.extend(library.from_pgn(g["pgn"], user_names=["nosegazm"]))
    for n in range(25):
        rows.extend(library.from_pgn(random_game(rng, n), user_names=["nosegazm"]))
    explorer = review_analysis.Explorer(rows)

    # Positions: along the games, and a few ends.
    fens: list[str] = []
    for row in rows[:12]:
        game = library.games_in(row.pgn)[0]
        board = game.board()
        for i, m in enumerate(game.mainline_moves()):
            if i % 3 == 0:
                fens.append(board.fen())
            board.push(m)
        fens.append(board.fen())
    fens.extend(ENDS)
    positions = [{"fen": f, "out": review_analysis.position(f)} for f in fens]
    moves = []
    for f in fens[:60]:
        board = chess.Board(f)
        legal = [m.uci() for m in board.legal_moves]
        for uci in rng.sample(legal, min(2, len(legal))) + ["e2e4", "a1a8", "zz", "e7e8q"]:
            try:
                out: Any = review_analysis.move(f, uci)
                moves.append({"fen": f, "uci": uci, "out": out})
            except ValueError as exc:
                moves.append({"fen": f, "uci": uci, "error": str(exc)})

    # The engine's lines: each position's own measurement, answered by a fake engine.
    review_analysis.find_stockfish = lambda _settings: Path("stockfish")  # type: ignore[assignment]
    lines = []
    for g in data:
        strength = STRENGTHS[g["strength"]]
        answers = {}
        for text in g["measured"][:20]:
            m = measured_from_json(json.dumps(text))
            answers[m.fen] = m.lines
        engine = FakeEngine(answers, default=[])
        made = review_analysis.Lines(Settings(), strength, make_engine=lambda _e=engine, **_: _e)
        for fen in list(answers)[:20]:
            # `found`: what `Lines` measured of the position (its `WholeSearcher`), drawn by
            # `render` -- the drawing is what the phone is held to.
            board = chess.Board(fen)
            found = made.search(board)
            out = review_analysis.Lines.render(board, found)
            lines.append({"fen": fen, "found": [l.model_dump(mode="json") for l in found], "out": out})

    explore = []
    for f in fens[:80] + [chess.STARTING_FEN, "8/8/8/4k3/8/8/8/4K3 w - - 0 1"]:
        explore.append({"fen": f, "out": explorer.explore(f)})

    pgns = []
    for i, g in enumerate(data):
        row = rows[i]
        game = library.games_in(row.pgn)[0]
        mainline = [m.uci() for m in game.mainline_moves()]
        texts = {
            str(p): {"why": f"Why {p}: the {{knight}} came in.", "better": f"Better {p}.", "lesson": "Look first."}
            for p in g["key_plies"][:3]
        }
        words = {"model": "m", "cost": 0.01, "texts": texts, "reveals": []}
        for tree in (side_lines(rng, game, mainline), []):
            for annotated in (True, False):
                pgns.append(
                    {
                        "row": row_json(row),
                        "moves": g["views"],
                        "tree": tree,
                        "annotated": annotated,
                        "words": words if i % 2 == 0 else None,
                        "out": review_analysis.pgn(row, g["views"], tree, annotated=annotated, words=words if i % 2 == 0 else None),
                    }
                )
    names = []
    for white, black in (("nosegazm", "opp"), ("A/B: C", "D*E?"), ("", None), ("Ünïcödé", "名前"), ("x" * 3, " sp ace ")):
        row = dataclasses.replace(rows[0], white=white, black=black)
        names.append({"row": row_json(row), "out": review_analysis.pgn_name(row, UTC)})
    comments = [
        {"label": label, "best": best, "played": played, "eval": ev, "out": review_analysis._comment(label, best, played, ev)}
        for label, best, played, ev in (
            ("blunder", "Nf3", "e4", {"cp": -312, "mate": None}),
            ("mistake", "e4", "e4", {"cp": 5, "mate": None}),
            ("best", None, "e4", {"cp": None, "mate": -3}),
            (None, None, "e4", None),
            ("book", "d4", "e4", {"cp": 0, "mate": None}),
            ("miss", "Qxf7#", "h3", {"cp": None, "mate": 2}),
        )
    ]
    fixture = {
        "rows": [row_json(r) for r in rows],
        "positions": positions,
        "moves": moves,
        "lines": lines,
        "explore": explore,
        "pgn": pgns,
        "names": names,
        "comments": comments,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture), encoding="utf-8")
    print(
        f"{OUT}: {len(positions)} positions, {len(moves)} moves, {len(lines)} lines, {len(explore)} explores, "
        f"{len(pgns)} PGNs, {len(names)} names ({OUT.stat().st_size // 1024} KB)"
    )


if __name__ == "__main__":
    main()
