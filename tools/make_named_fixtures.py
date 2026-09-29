"""Record what the desktop makes of what the player asks about, for the Kotlin to be held to.

`coach/named.py` turns "what about Nf3?" into something the engine examines, and turns the
engine's answer into a line of the coach's object and then into a sentence the coach says.
All of it is a pure function of the board, the words and the engine's lines -- so the engine
here is a recorded one: a deterministic stub whose lines are written into the fixture, and
which the Kotlin reads back, so `examine` is compared exactly without a search.

The positions are the opening, the puzzle sample, and puzzles of the tactical themes (pins,
skewers, forks, discovered attacks...) where attack and defence turn on legal moves: a piece
pinned to its king neither attacks nor defends off its line. On each, `examine_piece` is read
for every square of the board.

    cd D:/chess-coaching-companion
    PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python D:/sidecoach-android/tools/make_named_fixtures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis.score import human_expected_score  # noqa: E402
from sidecoach.coach import named as N  # noqa: E402
from sidecoach.events import DisclosureLevel, Line  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
PUZZLES = DESKTOP / "data" / "puzzles" / "sample.csv"
THEMES = DESKTOP / "data" / "puzzles" / "themes-7-120.csv"
# The tactical themes, and how many puzzles of each.
TACTICAL = (
    "pin",
    "skewer",
    "fork",
    "discoveredAttack",
    "xRayAttack",
    "hangingPiece",
    "trappedPiece",
    "capturingDefender",
    "doubleCheck",
    "deflection",
)
PER_THEME = 3
# The levels examined: the named move's reply is two plies below LINE, the whole line at it.
LEVELS = (DisclosureLevel.IDEA, DisclosureLevel.MOVE, DisclosureLevel.LINE)

ASKS = [
    "what about Nf3?",
    "is my knight on f6 safe?",
    "what does the bishop on c4 do?",
    "what about e5",
    "can I play Bb5 a6 Ba4?",
    "what if he takes on f7?",
    "pass Qh5",
    "I want to play e2e4",
    "what about Qz9?",
    "should I castle? O-O",
    "tell me about the pawn on d4 and the rook on a1",
    "nothing in particular",
    "1. e4 e5 2. Nf3",
    "square d5",
    "what about Nxf7 and then Kxf7",
]


class RecordedEngine:
    """A deterministic stand-in: the lines depend only on the position, and are written down."""

    def __init__(self) -> None:
        self.seen: dict[str, list[dict]] = {}

    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]:
        fen = board.fen()
        moves = list(board.legal_moves)
        if not moves:
            self.seen[fen] = []
            return []
        # A line of up to four plies, taken from the position itself, and a score derived
        # from it: no search, but the same answer every time for the same board.
        line: list[str] = []
        work = board.copy(stack=False)
        for _ in range(4):
            legal = list(work.legal_moves)
            if not legal:
                break
            pick = legal[(len(line) * 7 + len(legal)) % len(legal)]
            line.append(pick.uci())
            work.push(pick)
        cp = ((sum(ord(c) for c in fen) % 401) - 200)
        out = [
            Line(
                moves=line,
                depth=14,
                expected_score=human_expected_score(cp, None),
                score_cp=cp,
            )
        ]
        self.seen[fen] = [
            {
                "moves": list(one.moves),
                "depth": one.depth,
                "expected_score": one.expected_score,
                "score_cp": one.score_cp,
            }
            for one in out
        ]
        return out


def probe_json(probe: N.Probe) -> dict:
    return {
        "label": probe.label,
        "moves": list(probe.moves),
        "parsed": [m.uci() for m in probe.parsed],
        "illegal": probe.illegal,
        "square": None if probe.square is None else chess.square_name(probe.square),
    }


def case_for(board: chess.Board) -> dict:
    engine = RecordedEngine()
    us = board.turn
    # Probes as a planner would hand them over, including ones that go illegal partway: the
    # branch the words alone never reach. They do not depend on what was asked.
    legal = [board.san(m) for m in list(board.legal_moves)[:2]]
    planned = [
        N.parse_probe("a planned line", legal, board),
        N.parse_probe("an illegal follow-up", [*legal[:1], "Qz9"], board),
        N.parse_probe("straight to nonsense", ["Zz9"], board),
        N.parse_probe("the opponent first", ["pass", *legal[1:2]], board),
    ]
    # The board a move later: where the pieces stand now when the question is about the
    # position before (`pieces_board`).
    moved = board.copy(stack=False)
    if any(True for _ in board.legal_moves):
        moved.push(next(iter(board.legal_moves)))
    asks = []
    for text in ASKS:
        probes = N.probes_by_words(text, board)
        entry: dict = {
            "text": text,
            "probes": [probe_json(p) for p in probes],
            "pieces": [probe_json(p) for p in N.pieces_by_words(text, board)],
            # Only a move probe ever reaches examine_static; a square probe is read by
            # examine_piece, which is what analyse_named does.
            "static": [N.examine_static(p, board) for p in probes if p.square is None],
            "piece_reads": [
                N.examine_piece(p, board, user_colour=us) for p in probes if p.square is not None
            ],
            "levels": {},
        }
        before = engine.analyse(board)
        best = before[0] if before else None
        for level in LEVELS:
            examined = [
                N.examine(p, board, engine, best, user_colour=us, level=level, depth=14)
                if p.square is None
                else N.examine_piece(p, board, user_colour=us)
                for p in probes
            ]
            entry["levels"][str(int(level))] = {
                "examined": examined,
                "spoken": [N.spoken(f) for f in examined],
                "frames": [b.fen() for b in N.disclosed_frames(board, examined)],
                "named": N.analyse_named(
                    text, board, engine, user_colour=us, level=level, depth=14
                ),
                "named_static": N.analyse_named(
                    text, board, None, user_colour=us, level=level, static=True
                ),
                "named_none": N.analyse_named(
                    text, board, None, user_colour=us, level=level, static=False
                ),
                # The lines the sheet already shows (a move they start is read from them), and
                # the pieces read on the board as it stands now (TG-04).
                "named_known": N.analyse_named(
                    text, board, engine, user_colour=us, level=level, depth=14, known=before
                ),
                "named_pieces": N.analyse_named(
                    text, board, None, user_colour=us, level=level, pieces_board=moved
                ),
            }
        asks.append(entry)
    before = engine.analyse(board)
    best = before[0] if before else None
    planned_out = {
        "probes": [probe_json(p) for p in planned],
        "static": [N.examine_static(p, board) for p in planned],
        "levels": {
            str(int(level)): {
                "examined": [
                    N.examine(p, board, engine, best, user_colour=us, level=level, depth=14)
                    for p in planned
                ],
                "spoken": [
                    N.spoken(
                        N.examine(p, board, engine, best, user_colour=us, level=level, depth=14)
                    )
                    for p in planned
                ],
            }
            for level in LEVELS
        },
    }
    # Every square of the board, read as a piece or square probe: controllers pinned off the
    # line, what a piece attacks and defends by legal moves, what attacks and defends it.
    squares = [
        N.examine_piece(
            N.Probe(label=chess.square_name(sq), moves=[], square=sq), board, user_colour=us
        )
        for sq in chess.SQUARES
    ]
    return {
        "fen": board.fen(),
        "moved": moved.fen(),
        "asks": asks,
        "planned": planned_out,
        "squares": squares,
        "engine": engine.seen,
    }


def line_json(line: Line) -> dict:
    return {
        "moves": list(line.moves),
        "depth": line.depth,
        "expected_score": line.expected_score,
        "score_cp": line.score_cp,
    }


def desktop_tests() -> dict:
    """Every call `tests/test_named.py` and `tests/test_inventory.py` make to the named and
    inventory functions (and to check_draft), with what the desktop answered: their positions
    -- a pinned piece attacking and defending, a king in check, a pinned controller, hanging
    and undefended, a line recited from the examined one, the inventory's hanging pieces, pins,
    batteries and forks -- replayed on the phone (TG-04, TG-05)."""
    import importlib

    from desktop_tests import desktop_tests

    from sidecoach.coach import inventory as INV

    out: dict[str, list] = {"calls": [], "ran": []}
    current = {"test": ""}

    def board_arg(board: chess.Board) -> str:
        return board.fen()

    def probe_arg(probe: N.Probe) -> dict:
        return {
            "label": probe.label,
            "moves": list(probe.moves),
            "square": None if probe.square is None else chess.square_name(probe.square),
        }

    def recording_engine(engine, seen: dict):  # noqa: ANN001, ANN202
        if engine is None:
            return None

        class Recording:
            def analyse(self, b, *, depth=None):  # noqa: ANN001, ANN202
                lines = engine.analyse(b, depth=depth)
                seen[b.fen()] = [line_json(one) for one in lines]
                return lines

        return Recording()

    def wrap(real, fn: str, show):  # noqa: ANN001, ANN202
        def call(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            seen: dict = {}
            if fn in ("examine", "analyse_named"):
                args = list(args)
                args[2] = recording_engine(args[2], seen)
            result = real(*args, **kwargs)
            record = {"test": current["test"], "fn": fn, "args": show(*args, **kwargs), "out": None}
            record["engine"] = seen
            if fn in ("probes_by_words", "pieces_by_words"):
                record["out"] = [probe_json(p) for p in result]
            elif fn == "parse_probe":
                record["out"] = probe_json(result)
            elif fn == "disclosed_frames":
                record["out"] = [b.fen() for b in result]
            elif fn == "check_draft":
                record["out"] = [{"text": p.text, "reason": p.reason} for p in result]
            elif fn in ("pin_target", "skewer_target"):
                record["out"] = None if result is None else [chess.square_name(s) for s in result]
            elif fn == "batteries":
                record["out"] = [[chess.square_name(s) for s in group] for group in result]
            elif fn == "forks_in_place":
                record["out"] = [
                    [chess.square_name(s), [chess.square_name(t) for t in targets]]
                    for s, targets in result
                ]
            else:
                record["out"] = result
            out["calls"].append(record)
            return result

        return call

    shows = {
        "probes_by_words": lambda text, board: {"text": text, "fen": board_arg(board)},
        "pieces_by_words": lambda text, board: {"text": text, "fen": board_arg(board)},
        "parse_probe": lambda label, moves, board: {
            "label": label, "moves": list(moves), "fen": board_arg(board)
        },
        "examine": lambda probe, board, engine, best, *, user_colour, level, depth=14, known=(): {
            "probe": probe_arg(probe),
            "fen": board_arg(board),
            "best": None if best is None else line_json(best),
            "user_white": user_colour == chess.WHITE,
            "level": int(level),
            "depth": depth,
            "known": [line_json(one) for one in known],
        },
        "examine_piece": lambda probe, board, *, user_colour: {
            "probe": probe_arg(probe), "fen": board_arg(board), "user_white": user_colour == chess.WHITE
        },
        "spoken": lambda fact: {"fact": fact},
        "disclosed_frames": lambda board, facts: {"fen": board_arg(board), "facts": list(facts)},
        "analyse_named": lambda text, board, engine, *, user_colour, level, depth=14,
        static=False, pieces_board=None, known=(): {
            "text": text,
            "fen": board_arg(board),
            "engine": engine is not None,
            "user_white": user_colour == chess.WHITE,
            "level": int(level),
            "depth": depth,
            "static": static,
            "pieces_fen": None if pieces_board is None else pieces_board.fen(),
            "known": [line_json(one) for one in known],
        },
        "check_draft": lambda text, board, *, after=(), us=None: {
            "text": text, "fen": board_arg(board), "after": [b.fen() for b in after], "us": us
        },
        "pin_target": lambda board, sq: {"fen": board_arg(board), "square": chess.square_name(sq)},
        "skewer_target": lambda board, sq: {"fen": board_arg(board), "square": chess.square_name(sq)},
        "batteries": lambda board, colour: {"fen": board_arg(board), "white": colour == chess.WHITE},
        "forks_in_place": lambda board, colour: {"fen": board_arg(board), "white": colour == chess.WHITE},
    }
    for module_name in ("tests.test_named", "tests.test_inventory"):
        tests = importlib.import_module(module_name)
        saved = {}
        for fn, show in shows.items():
            if hasattr(tests, fn):
                saved[fn] = getattr(tests, fn)
                setattr(tests, fn, wrap(saved[fn], fn, show))
        try:
            for name, test in desktop_tests(tests):
                current["test"] = f"{module_name}.{name}"
                test()  # the desktop's own assertions hold as the calls are recorded
                out["ran"].append(current["test"])
        finally:
            for fn, real in saved.items():
                setattr(tests, fn, real)
    assert INV  # the inventory tests call it through their own imports
    return out


def main() -> None:
    cases: list[dict] = []
    boards = [chess.Board()]
    opening = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4"):
        opening.push_san(san)
        boards.append(opening.copy(stack=False))
    if PUZZLES.exists():
        with PUZZLES.open(newline="", encoding="utf-8") as handle:
            for row in list(csv.DictReader(handle))[:40]:
                board = chess.Board(row["FEN"])
                moves = row["Moves"].split()
                for index, uci in enumerate(moves):
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        break
                    if index % 2 == 1:
                        boards.append(board.copy(stack=False))
                    board.push(move)
    if THEMES.exists():
        taken: dict[str, int] = {}
        with THEMES.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                names = [t for t in row["Themes"].split() if t in TACTICAL]
                if not names or all(taken.get(t, 0) >= PER_THEME for t in names):
                    continue
                for t in names:
                    taken[t] = taken.get(t, 0) + 1
                board = chess.Board(row["FEN"])
                moves = row["Moves"].split()
                for index, uci in enumerate(moves[:4]):
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        break
                    board.push(move)
                    if index in (0, 2):
                        boards.append(board.copy(stack=False))
    for board in boards:
        cases.append(case_for(board))

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"cases": cases, "tests": desktop_tests()}
    (OUT / "named.json").write_text(json.dumps(payload), encoding="utf-8")
    size = (OUT / "named.json").stat().st_size
    print(f"{len(cases)} positions x {len(ASKS)} asks -> {OUT / 'named.json'} ({size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
