"""Record what the desktop's plan reader says, for the Kotlin port to be held to.

`sidecoach/coach/plans.py` (desktop 5e80cdf) reads the player's first four moves of the best
line as one plan -- an attack on the king, a pawn storm, heavy pieces doubled on a file, a
knight's route to an outpost, a passed pawn pushed, a pawn break prepared, space taken on the
far wing -- and the threat the best move makes when the line wins nothing: a mate, a piece it
goes after, a trade it offers. The facts then say these words at every rung, so a port that
differs anywhere says something different.

Every case is a position, a line (UCI) and the expected score it was read with; each is read
for the plan (`plans.read`) and for the threat of its first move (`plans.threat`), with every
word the coach says of either. The lines come from:

- the desktop's own tests (`tests/test_plans.py`), each case as written there;
- the master games' facts (`data/eval/facts/masters_v2/`): the engine's lines of every
  position of 199 annotated master games, where plans and threats are what the notes say;
- the puzzle bench's cache (`data/puzzles/analysis/`): the engine's lines of Lichess puzzles;
- the Lichess puzzle sample's solutions, read with no score.

Most lines carry no plan and no threat, so what is kept is chosen for coverage: every case
whose plan kind, or threat kind and reason, is still short of its quota, and one in STRIDE of
the rest. A second set, `moves`, asks `plans.threat` of every legal quiet move of a few hundred
master positions (no line, so no trade), where the mate and win threats and their refusals
live.

    cd D:/chess-coaching-companion
    PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python D:/sidecoach-android/tools/make_plans_fixtures.py
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

from sidecoach.coach import plans as PL  # noqa: E402
from sidecoach.events import Analysis  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
MASTERS = DESKTOP / "data" / "eval" / "facts" / "masters_v2"
PUZZLES = DESKTOP / "data" / "puzzles" / "sample.csv"
CACHE = DESKTOP / "data" / "puzzles" / "analysis" / "d18-pv6-13cdf1d231.jsonl"
# Rows of the cache read (it holds seventy thousand).
CACHE_ROWS = 6000
# Cases kept per plan kind, per threat kind and reason; one in STRIDE of those with neither.
QUOTA = 90
STRIDE = 40
# Master positions whose every legal quiet move is asked for its threat.
MOVE_POSITIONS = 160

# `tests/test_plans.py`: (name, fen, line, score) for the plan, and (name, fen, move, line) for
# the threat, as the tests read them.
TEST_READS = [
    ("doubled", "6k1/p4ppp/8/8/8/8/5PPP/R4RK1 w - - 0 1", "a1d1 a7a6 d1d2 a6a5 f1d1 a5a4", None),
    ("outpost", "4k3/8/3p4/8/4P3/8/8/1N2K3 w - - 0 1", "b1c3 e8f7 c3d5 f7e6", None),
    ("out and back", "4k3/8/3p4/3N4/4P3/8/8/4K3 w - - 0 1", "d5c3 e8f7 c3d5 f7e6", None),
    ("passer", "4k3/7p/8/8/P7/8/8/4K3 w - - 0 1", "a4a5 h7h6 a5a6 h6h5", None),
    ("storm", "6k1/p4ppp/8/8/8/8/6PP/2K5 w - - 0 1", "g2g4 a7a6 h2h4 a6a5 g4g5 a5a4", None),
    ("own king's pawns", "6k1/p4ppp/8/8/8/8/5PPP/6K1 w - - 0 1", "g2g4 a7a6 h2h4 a6a5", None),
    ("wing", "6k1/p4ppp/8/8/8/8/PP3PPP/6K1 w - - 0 1", "b2b4 g8f8 a2a4 f8e7", None),
    ("break", "4k3/8/8/4p3/8/8/3P4/4K1N1 w - - 0 1", "g1f3 e8e7 d2d4 e7e6", None),
    ("attack", "6k1/p4ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1", "d1d3 a7a6 e1e3 a6a5 e3h3 a5a4", None),
    ("attack when worse", "6k1/p4ppp/8/8/8/8/5PPP/3QR1K1 w - - 0 1", "d1d3 a7a6 e1e3 a6a5 e3h3 a5a4", 0.3),
    ("captures", "6k1/p4ppp/1p6/P7/8/8/1P3PPP/6K1 w - - 0 1", "a5b6 a7b6 b2b4 g8f8", None),
]
TEST_THREATS = [
    ("loose piece", "4k3/8/8/1b6/8/8/8/1N2K3 w - - 0 1", "b1c3", ""),
    ("gives it away", "4k3/8/8/1b6/1p6/8/8/1N2K3 w - - 0 1", "b1c3", ""),
    ("mate", "6k1/5ppp/8/8/8/8/1R3PPP/6K1 w - - 0 1", "b2e2", ""),
    ("trade", "7k/8/8/8/8/8/RK6/7r w - - 0 1", "a2a1", "a2a1 h1a1 b2a1"),
    ("no trade", "7k/8/8/8/8/8/RK6/7r w - - 0 1", "a2a1", "a2a1 h8g8 b2c2"),
]


def plan_json(plan: PL.LinePlan | None, board: chess.Board) -> dict | None:
    if plan is None:
        return None
    return {
        "kind": plan.kind,
        "square": None if plan.square is None else chess.square_name(plan.square),
        "where": plan.where,
        "pieces": [chess.square_name(s) for s in plan.pieces],
        "sans": list(plan.sans),
        "detail": plan.detail,
        "name": PL.name(plan),
        "mechanism": PL.mechanism(plan),
        "target": PL.target(plan),
        "pieces_text": PL.pieces(plan, board),
        "why": PL.why(plan, board),
        "question": PL.PLAN_QUESTION[plan.kind],
        "nudge": PL.PLAN_NUDGE[plan.kind],
    }


def threat_json(t: PL.MoveThreat | None, board: chess.Board, move: chess.Move) -> dict | None:
    if t is None:
        return None
    return {
        "kind": t.kind,
        "square": chess.square_name(t.square),
        "follow": t.follow.uci(),
        "gain": t.gain,
        "why": t.why,
        "mechanism": PL.threat_mechanism(t),
        "target": PL.threat_target(t, board),
        "why_text": PL.threat_why(t, board, move),
        "goal": PL.THREAT_GOAL[t.kind],
        "goal_won": PL.THREAT_GOAL_WON[t.kind],
        "name": PL.THREAT_NAME[t.kind],
        "question": PL.THREAT_QUESTION[t.kind],
        "nudge": PL.THREAT_NUDGE[t.kind],
    }


def moves_of(ucis: list[str]) -> list[chess.Move]:
    out = []
    for uci in ucis:
        try:
            out.append(chess.Move.from_uci(uci))
        except ValueError:
            continue
    return out


def case(source: str, board: chess.Board, pv: list[chess.Move], score: float | None) -> dict:
    """The plan of `pv` and the threat of its first move, as the facts read them."""
    plan = PL.read(board, pv, score)
    threat = None
    if pv and pv[0] in board.legal_moves:
        threat = PL.threat(board, pv[0], pv, score)
    return {
        "source": source,
        "fen": board.fen(),
        "pv": [m.uci() for m in pv],
        "score": score,
        "plan": plan_json(plan, board),
        "threat": None if threat is None else threat_json(threat, board, pv[0]),
    }


def keys_of(c: dict) -> list[str]:
    keys = []
    if c["plan"] is not None:
        keys.append(f"plan:{c['plan']['kind']}")
    if c["threat"] is not None:
        t = c["threat"]
        keys.append(f"threat:{t['kind']}" + (f":{t['why']}" if t["kind"] == "trade" else ""))
    return keys


def masters_analyses() -> list[tuple[str, Analysis]]:
    """(game, analysis) for every position the master games' facts measured."""
    out: list[tuple[str, Analysis]] = []
    for path in sorted(MASTERS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [data.get("start_analysis")] + [s.get("analysis") for s in data.get("script", [])]
        for row in rows:
            if row:
                out.append((path.stem, Analysis.model_validate(row)))
    return out


def cache_analyses() -> list[tuple[str, Analysis]]:
    out: list[tuple[str, Analysis]] = []
    if not CACHE.exists():
        return out
    with CACHE.open(encoding="utf-8") as fh:
        for line in fh:
            if len(out) >= CACHE_ROWS:
                break
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line the running bench is still writing
            out.append(("cache", Analysis.model_validate(row["analysis"])))
    return out


def main() -> None:
    tests: list[dict] = []
    for name, fen, line, score in TEST_READS:
        board = chess.Board(fen)
        tests.append(case(f"test_plans: {name}", board, moves_of(line.split()), score))
    for name, fen, move, line in TEST_THREATS:
        board = chess.Board(fen)
        m = chess.Move.from_uci(move)
        pv = moves_of(line.split())
        t = PL.threat(board, m, pv)
        tests.append({
            "source": f"test_plans: {name}",
            "fen": fen,
            "move": move,
            "pv": [x.uci() for x in pv],
            "score": None,
            "threat": threat_json(t, board, m),
        })

    candidates: list[dict] = []
    masters = masters_analyses()
    for source, analyses in (("masters", masters), ("cache", cache_analyses())):
        for game, analysis in analyses:
            board = chess.Board(analysis.fen)
            for k, line in enumerate(analysis.lines[:3]):
                pv = moves_of(list(line.moves))
                if not pv:
                    continue
                candidates.append(case(f"{source} {game} line {k}", board, pv, line.expected_score))
    if PUZZLES.exists():
        with PUZZLES.open(newline="", encoding="utf-8") as handle:
            for row in list(csv.DictReader(handle))[:600]:
                board = chess.Board(row["FEN"])
                moves = [chess.Move.from_uci(u) for u in row["Moves"].split()]
                for index, move in enumerate(moves):
                    if move not in board.legal_moves:
                        break
                    if index % 2 == 1:
                        candidates.append(case(f"lichess {row['PuzzleId']}", board, moves[index:], None))
                    board.push(move)

    counts: dict[str, int] = {}
    kept: list[dict] = []
    quiet = 0
    for index, c in enumerate(candidates):
        keys = keys_of(c)
        if keys:
            if any(counts.get(key, 0) < QUOTA for key in keys):
                kept.append(c)
                for key in keys:
                    counts[key] = counts.get(key, 0) + 1
        elif index % STRIDE == 0:
            kept.append(c)
            quiet += 1

    # Every legal quiet move of some master positions, asked for its threat with no line.
    moves: list[dict] = []
    step = max(1, len(masters) // MOVE_POSITIONS)
    threat_kinds: dict[str, int] = {}
    for game, analysis in masters[::step][:MOVE_POSITIONS]:
        board = chess.Board(analysis.fen)
        found = []
        for move in board.legal_moves:
            if board.is_capture(move):
                continue
            t = PL.threat(board, move)
            found.append({"move": move.uci(), "threat": threat_json(t, board, move)})
            if t is not None:
                threat_kinds[t.kind] = threat_kinds.get(t.kind, 0) + 1
        moves.append({"source": f"masters {game}", "fen": board.fen(), "moves": found})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "plans.json").write_text(
        json.dumps({"tests": tests, "cases": kept, "moves": moves}), encoding="utf-8"
    )
    size = (OUT / "plans.json").stat().st_size
    print(
        f"{len(tests)} test cases, {len(kept)} of {len(candidates)} lines "
        f"({quiet} with neither), {len(moves)} positions' moves "
        f"({sum(len(m['moves']) for m in moves)} threats asked) -> {OUT / 'plans.json'} "
        f"({size / 1e6:.1f} MB)"
    )
    print("kept:", dict(sorted(counts.items())))
    print("move threats:", dict(sorted(threat_kinds.items())))


if __name__ == "__main__":
    main()
