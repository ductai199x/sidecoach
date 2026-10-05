"""The workloads the engine's scheduling is measured on: a session as the engine glue hands it
to the analyser -- a dumped session's game, the player's chess.com games at their clocks (the
blitz again at three times its speed), a game browsed, and moves arriving at a fixed pace -- each
a list of steps on a clock. Read by measure.py (`workloads`), sweep2.py and workers_compare.py."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis.annotator import cache_key  # noqa: E402

CACHE = ROOT / "data" / "engine-cache"


@dataclass
class Step:
    t: float
    kind: str  # start, resync, move, takeback
    board: chess.Board | None = None  # start/resync: the position; move: the board before
    move: chess.Move | None = None
    ply: int = 0
    by_user: bool = False
    plies: int = 0
    game_id: str = ""
    user: bool | None = None  # the player's colour as known at this step


def session(dump: Path) -> list[Step]:
    """The dump's game, as the engine glue (`cli._Assessing`) hands it to the analyser."""
    steps: list[Step] = []
    board: chess.Board | None = None
    game_id, ply = "", 0
    user: bool | None = None
    for line in (dump / "debug.jsonl").open(encoding="utf-8"):
        d = json.loads(line)
        if d["chan"] != "game":
            continue
        t, text, data = d["t"], d["text"], d.get("data") or {}
        if "start_fen" in data:
            game_id, board, ply = data["game_id"], chess.Board(data["start_fen"]), 0
            user = chess.BLACK if text.endswith("you play black") else chess.WHITE
            steps.append(Step(t, "start", board.copy(), game_id=game_id, user=user))
        elif text.startswith("resync:") and "fen" in data:
            board, ply = chess.Board(data["fen"]), 0
            steps.append(Step(t, "resync", board.copy(), game_id=game_id, user=user))
        elif text.startswith("takeback") and board is not None:
            plies = int(re.match(r"takeback (\d+)", text)[1])
            ply = max(0, ply - plies)
            for _ in range(min(plies, len(board.move_stack))):
                board.pop()
            if board.fen() != data.get("fen", board.fen()):
                board = chess.Board(data["fen"])
            steps.append(
                Step(t, "takeback", board.copy(), plies=plies, ply=ply, game_id=game_id, user=user)
            )
        elif "uci" in data and board is not None:
            move = chess.Move.from_uci(data["uci"])
            if move not in board.legal_moves and ply == 0:
                board.turn = not board.turn  # a position adopted with the side to move pending
                board.ep_square = None
            if move not in board.legal_moves:
                continue
            ply = data["ply"]
            by_user = bool(data["by_user"])
            user = board.turn if by_user else not board.turn
            steps.append(
                Step(t, "move", board.copy(), move, ply, by_user, game_id=game_id, user=user)
            )
            board.push(move)
    return steps


def _after(step: Step) -> chess.Board:
    assert step.board is not None and step.move is not None
    after = step.board.copy()
    after.push(step.move)
    return after


def game_boards(steps: list[Step]) -> dict[str, chess.Board]:
    """Every position the sessions' games stand on."""
    out: dict[str, chess.Board] = {}
    for step in steps:
        for board in [step.board] if step.kind != "move" else [step.board, _after(step)]:
            if board is not None and not board.is_game_over():
                out[cache_key(board)] = board
    return out


def pgn_session(
    pgn: str, user: str, *, speed: float = 1.0, game_id: str = "g"
) -> list[Step] | None:
    """A game as it was played: each move at the time its clock says it was made (the time the
    mover spent, from the clocks and the increment), `speed` times faster; the player's side by
    their name. None for a game without clocks or without the player."""
    import io

    import chess.pgn

    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return None
    white = game.headers.get("White", "").lower() == user.lower()
    black = game.headers.get("Black", "").lower() == user.lower()
    if white == black:
        return None
    tc = game.headers.get("TimeControl", "")
    inc = float(tc.split("+")[1]) if "+" in tc else 0.0
    board = game.board()
    colour = chess.WHITE if white else chess.BLACK
    steps = [Step(0.0, "start", board.copy(), game_id=game_id, user=colour)]
    t = 0.0
    last = {chess.WHITE: None, chess.BLACK: None}
    for ply, node in enumerate(game.mainline(), start=1):
        clock = node.clock()
        if clock is None:
            return None
        mover = board.turn
        spent = 0.0 if last[mover] is None else max(0.0, last[mover] - clock + inc)
        last[mover] = clock
        t += max(0.05, spent) / speed
        steps.append(
            Step(
                t,
                "move",
                board.copy(),
                node.move,
                ply,
                mover == colour,
                game_id=game_id,
                user=colour,
            )
        )
        board.push(node.move)
    return steps


def browse_session(pgn: str, *, step_s: float = 0.3, game_id: str = "b") -> list[Step] | None:
    """An analysis board browsed: a game stepped through a move every `step_s`, back ten plies
    one at a time, forward again: every position reached, the engine never ahead of it."""
    import io

    import chess.pgn

    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return None
    moves = list(game.mainline_moves())[:40]
    board = game.board()
    steps = [Step(0.0, "start", board.copy(), game_id=game_id, user=chess.WHITE)]
    t = 0.0
    for ply, move in enumerate(moves[:30], start=1):
        t += step_s
        steps.append(
            Step(
                t,
                "move",
                board.copy(),
                move,
                ply,
                board.turn == chess.WHITE,
                game_id=game_id,
                user=chess.WHITE,
            )
        )
        board.push(move)
    for back in range(10):  # back, one ply at a time
        t += step_s
        board.pop()
        steps.append(
            Step(
                t,
                "takeback",
                board.copy(),
                plies=1,
                ply=29 - back,
                game_id=game_id,
                user=chess.WHITE,
            )
        )
    for ply, move in enumerate(moves[20:40], start=21):  # and forward again
        t += step_s
        steps.append(
            Step(
                t,
                "move",
                board.copy(),
                move,
                ply,
                board.turn == chess.WHITE,
                game_id=game_id,
                user=chess.WHITE,
            )
        )
        board.push(move)
    return steps


def chesscom_sessions(user: str, per_control: int = 1000) -> dict[str, list[Step]]:
    """The games `chesscom_games.py` fetched, at their clocks, at most `per_control` of each time
    control; the blitz ones at three times their speed too (bullet-like: there is no bullet in
    the archive)."""
    games = json.loads((CACHE / f"chesscom-{user}.json").read_text(encoding="utf-8"))
    out: dict[str, list[Step]] = {}
    taken: Counter[str] = Counter()
    for n, game in enumerate(games):
        tc = game.get("time_control", "?")
        if taken[tc] >= per_control:
            continue
        taken[tc] += 1
        steps = pgn_session(game["pgn"], user, game_id=f"{tc}-{n}")
        if steps:
            out[f"{tc} #{n}"] = steps
        if tc.startswith("180"):
            fast = pgn_session(game["pgn"], user, speed=3.0, game_id=f"bullet-{n}")
            if fast:
                out[f"blitz x3 #{n}"] = fast
    return out


def bursts(sessions: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Moves arriving at a fixed pace -- premoves, bullet, a game stepped through: the first 40
    plies of three long games at every 0.2, 0.5, 1 and 2 seconds, the user White."""
    games = [s for name, s in sessions.items() if name.startswith("900+10")][:3]
    out = {}
    for pace in (0.2, 0.5, 1.0, 2.0):
        for g, steps in enumerate(games):
            moves = [s for s in steps if s.kind == "move"][:40]
            start = steps[0]
            seq = [Step(0.0, "start", start.board, game_id=f"burst{g}", user=chess.WHITE)]
            for i, s in enumerate(moves, start=1):
                seq.append(
                    Step(
                        i * pace,
                        "move",
                        s.board,
                        s.move,
                        s.ply,
                        s.board.turn == chess.WHITE,
                        game_id=f"burst{g}",
                        user=chess.WHITE,
                    )
                )
            out[f"burst {pace}s #{g}"] = seq
    return out
