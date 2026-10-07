"""Record what the desktop's engine wrapper says to a UCI engine, and what it makes of the answers.

`sidecoach/analysis/engine.py` drives Stockfish through python-chess, and python-chess decides
what goes down the pipe: the options (sent only when they change), a new game before the first
search, and the position as the game's moves from where they began (`position startpos moves
...`), so the engine sees the repetitions in them -- the FEN alone when the history holds a
null move. The phone speaks UCI itself, so the port is held to the desktop's transcript.

The engine here is scripted, not Stockfish: this file run with `--fake` is a UCI engine that
announces Stockfish 19's options and answers each search from the position it was sent (its
legal moves, a mate in one when there is one, a score made from the command's text), with
the lines a real search also prints around them: an `info string`, a shallower bound first, a
line printed again one depth shallower after the rest, and the `info depth N currmove ...` that
Stockfish sends once a search has run three seconds -- python-chess merges every one of them
into its lines, the depth it reports with them. The
desktop's own `Engine` is run against it over a set of boards -- the start, a game's moves, a
repetition, castling sent as king-takes-rook, a FEN root with moves, an en passant square that
is no capture, a null move in the history, finished and claimable positions -- once with an
engine that gives win-draw-loss and once with one that does not (python-chess's model then).
Recorded: every line the desktop sent, the answer to each search keyed by the position and the
number of lines, and the lines `Engine.analyse` returned. `EngineParityTest` scripts the same
answers into a shell script, runs the Kotlin `Engine` against it and compares.

    cd D:/chess-coaching-companion
    uv run --no-sync python android/tools/make_engine_fixtures.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import chess

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
DESKTOP = Path(__file__).resolve().parents[2]

# What Stockfish 19 answers to `uci` (the desktop's tools/stockfish build), less its name.
OPTIONS = [
    "option name Debug Log File type string default <empty>",
    "option name NumaPolicy type string default auto",
    "option name Threads type spin default 1 min 1 max 1024",
    "option name Hash type spin default 16 min 1 max 33554432",
    "option name Clear Hash type button",
    "option name Ponder type check default false",
    "option name MultiPV type spin default 1 min 1 max 256",
    "option name Skill Level type spin default 20 min 0 max 20",
    "option name Move Overhead type spin default 10 min 0 max 5000",
    "option name nodestime type spin default 0 min 0 max 10000",
    "option name UCI_Chess960 type check default false",
    "option name UCI_LimitStrength type check default false",
    "option name UCI_Elo type spin default 1320 min 1320 max 3190",
    "option name UCI_ShowWDL type check default false",
    "option name SyzygyPath type string default <empty>",
    "option name SyzygyProbeDepth type spin default 1 min 1 max 100",
    "option name Syzygy50MoveRule type check default true",
    "option name SyzygyProbeLimit type spin default 7 min 0 max 7",
    "option name EvalFile type string default nn-1a298aa575a0.nnue",
]

# The engines: one that gives win-draw-loss when asked, one that has no such option.
PERSONAS = {
    "wdl": OPTIONS,
    "nowdl": [o for o in OPTIONS if "UCI_ShowWDL" not in o],
}

START = chess.STARTING_FEN
SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
# Each board: where it starts, the moves pushed on it ("0000" the null move), whether the moves
# stay on its stack, and the analyse call's number of lines and depth; and, when given, its
# `carry_best`.
SCENARIOS = [
    ("the start", START, [], True, 3, 10),
    ("a game's moves", START, ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4"], True, 3, 10),
    # The start again, a second time: not yet a draw, but only the moves say it repeats.
    ("a repetition in the moves", START, ["g1f3", "g8f6", "f3g1", "f6g8"], True, 3, 10),
    ("castling sent as king-takes-rook", START,
     ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5", "e1h1"], True, 1, 12),
    ("a FEN with a mate in one", SCHOLAR, [], True, 3, 10),
    ("a FEN root with an en passant square no pawn can take, and a move",
     "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", ["d2d4"], True, 3, 10),
    ("a double step with no history", START, ["e2e4"], False, 2, 10),
    ("a null move in the history", START, ["e2e4", "0000"], True, 1, 10),
    ("a claimable threefold repetition", START,
     ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"], True, 3, 10),
    ("checkmate", "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4", [], True, 3, 10),
    ("one legal move", "7k/8/8/8/8/8/6q1/7K w - - 0 1", [], True, 3, 10),
    ("black to move, deep in a game", "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 b - - 0 40",
     ["c6d4", "f3d4"], True, 3, 14),
    # The best line carried on (`carry_best`, the parity audit, 2026-09-29): three plies a
    # search here, short of MIN_BEST_PLIES, so it is searched on from its end, one line at
    # the extension's depth, EXTEND_TRIES times -- sent from the position the line reaches.
    ("a best line carried on", START, ["e2e4", "e7e5"], True, 3, 10, True),
]


def board_of(fen: str, moves: list[str], stack: bool) -> chess.Board:
    board = chess.Board(fen)
    for uci in moves:
        board.push(chess.Move.from_uci(uci))
    return board if stack else board.copy(stack=False)


# The scripted engine


def _position(command: str) -> chess.Board:
    words = command.split()
    if words[1] == "startpos":
        board, rest = chess.Board(), words[2:]
    else:
        board, rest = chess.Board(" ".join(words[2:8])), words[8:]
    if rest and rest[0] == "moves":
        for uci in rest[1:]:
            board.push_uci(uci)
    return board


def answer(command: str, board: chess.Board, multipv: int, show_wdl: bool, depth: int) -> list[str]:
    """The info lines and best move for a search of `board`, sent as `command`: its legal moves
    in UCI order, a mate in one first; each carried two plies on; a score from the command's
    own letters, so a different command is a different answer."""
    legal = sorted(board.legal_moves, key=lambda m: m.uci())
    mates = []
    for move in legal:
        board.push(move)
        if board.is_checkmate():
            mates.append(move)
        board.pop()
    ordered = mates + [m for m in legal if m not in mates]
    base = sum(map(ord, command)) % 301 - 150
    out = ["info string NNUE evaluation using nn-1a298aa575a0.nnue"]
    if ordered:
        # A shallower pass, its score a bound: merged, then overwritten.
        out.append(
            f"info depth {depth - 1} seldepth {depth} multipv 1 score cp {base + 7} lowerbound "
            f"nodes 500 nps 100000 time 5 pv {ordered[-1].uci()}"
        )
    for i, move in enumerate(ordered[:multipv]):
        pv = [move]
        after = board.copy(stack=False)
        after.push(move)
        for _ in range(2):
            replies = sorted(after.legal_moves, key=lambda m: m.uci())
            if not replies:
                break
            pv.append(replies[0])
            after.push(replies[0])
        if move in mates:
            score, wdl = "mate 1", " wdl 1000 0 0"
        else:
            cp = base - 20 * i
            score, wdl = f"cp {cp}", f" wdl {300 + cp} 400 {300 - cp}"
        out.append(
            f"info depth {depth} seldepth {depth + 2} multipv {i + 1} score {score}"
            f"{wdl if show_wdl else ''} nodes 1000 nps 100000 hashfull 0 tbhits 0 time 10 "
            f"pv {' '.join(m.uci() for m in pv)}"
        )
    shown = min(multipv, len(ordered))
    if shown >= 2 and ordered[shown - 1] not in mates:
        # The last line again, one depth shallower, one move long and scored lower: the last
        # word wins, whatever its depth.
        out.append(
            f"info depth {depth - 1} seldepth {depth} multipv {shown} score cp {base - 20 * shown - 5}"
            f" nodes 1500 nps 100000 time 15 pv {ordered[shown - 1].uci()}"
        )
    if ordered:
        # Three seconds in, the next depth's first root move: no line index, no moves.
        out.append(f"info depth {depth + 1} currmove {ordered[0].uci()} currmovenumber 1")
    out.append(f"bestmove {ordered[0].uci()}" if ordered else "bestmove (none)")
    return out


def fake(persona: str, folder: Path) -> None:
    """A UCI engine on stdin/stdout that logs every line it reads and each answer it gives."""
    log = (folder / f"{persona}.log").open("a", encoding="utf-8")
    replies = (folder / f"{persona}.replies").open("a", encoding="utf-8")
    out = sys.stdout.buffer
    board, command, multipv, show_wdl = chess.Board(), "", 1, False

    def say(*lines: str) -> None:
        for line in lines:
            out.write((line + "\n").encode())
        out.flush()

    for raw in sys.stdin:
        line = raw.rstrip("\r\n")
        log.write(line + "\n")
        log.flush()
        if line == "uci":
            say("id name Scripted", "id author sidecoach", *PERSONAS[persona], "uciok")
        elif line == "isready":
            say("readyok")
        elif line.startswith("setoption name MultiPV value "):
            multipv = int(line.split()[-1])
        elif line.startswith("setoption name UCI_ShowWDL value "):
            show_wdl = line.split()[-1] == "true"
        elif line.startswith("position"):
            command, board = line, _position(line)
        elif line.startswith("go"):
            words = line.split()
            depth = int(words[words.index("depth") + 1]) if "depth" in words else 12
            lines = answer(command, board, multipv, show_wdl, depth)
            replies.write(json.dumps({"key": f"{multipv}|{command}", "reply": lines}) + "\n")
            replies.flush()
            say(*lines)
        elif line == "quit":
            break


# The recording


def line_json(line) -> dict:  # noqa: ANN001
    return {
        "moves": line.moves,
        "depth": line.depth,
        "expected_score": line.expected_score,
        "score_cp": line.score_cp,
        "mate": line.mate,
        "wdl": list(line.wdl) if line.wdl is not None else None,
    }


def record(persona: str, folder: Path) -> dict:
    sys.path.insert(0, str(DESKTOP / "src"))
    import chess.engine

    from sidecoach.analysis.engine import Engine

    real = chess.engine.SimpleEngine.popen_uci

    def popen(_command: object, **kwargs: object) -> chess.engine.SimpleEngine:
        return real([sys.executable, str(Path(__file__).resolve()), "--fake", persona, str(folder)], **kwargs)

    chess.engine.SimpleEngine.popen_uci = popen  # type: ignore[method-assign]
    try:
        engine = Engine(path=Path("scripted"), threads=2, hash_mb=64, multipv=3, pv_plies=12)
        results = []
        for name, fen, moves, stack, multipv, depth, *carry in SCENARIOS:
            board = board_of(fen, moves, stack)
            carry_best = bool(carry and carry[0])
            lines = engine.analyse(board, depth=depth, multipv=multipv, carry_best=carry_best)
            if carry_best and (not lines or len(lines[0].moves) <= 3):
                raise SystemExit(f"{name}: the best line was not carried on: {lines}")
            results.append({
                "name": name,
                "fen": fen,
                "moves": moves,
                "stack": stack,
                "multipv": multipv,
                "depth": depth,
                "carry_best": carry_best,
                "lines": [line_json(line) for line in lines],
            })
        engine.close()
    finally:
        chess.engine.SimpleEngine.popen_uci = real  # type: ignore[method-assign]
    sent = (folder / f"{persona}.log").read_text(encoding="utf-8").splitlines()
    replies = {}
    for row in (folder / f"{persona}.replies").read_text(encoding="utf-8").splitlines():
        one = json.loads(row)
        replies[one["key"]] = one["reply"]
    return {"options": PERSONAS[persona], "sent": sent, "replies": replies, "scenarios": results}


def main() -> None:
    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(scratch)
        data = {persona: record(persona, folder) for persona in PERSONAS}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "engine.json").write_text(json.dumps(data, indent=1), encoding="utf-8")
    for persona, one in data.items():
        positions = [s for s in one["sent"] if s.startswith("position")]
        print(f"{persona}: {len(one['sent'])} lines sent, {len(positions)} searches")
        for line in positions:
            print(f"  {line}")
    print(f"-> {OUT / 'engine.json'}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--fake":
        fake(sys.argv[2], Path(sys.argv[3]))
    else:
        main()
