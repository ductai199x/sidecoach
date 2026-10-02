"""The Stockfish process's life: a closed engine stays closed under a search already under
way, and a wedged one does not hold its lock forever. Real Stockfish."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import chess
import psutil
import pytest

from sidecoach.analysis.engine import Engine, EngineUnavailable
from sidecoach.config import find_stockfish, load_settings

STOCKFISH = find_stockfish(load_settings())
if STOCKFISH is None:  # a worktree without tools/: the checkout's
    found = sorted(Path(r"D:\chess-coaching-companion\tools\stockfish").glob("stockfish*.exe"))
    STOCKFISH = found[0] if found else None

pytestmark = [
    pytest.mark.skipif(STOCKFISH is None, reason="no Stockfish binary"),
    pytest.mark.xdist_group("stockfish"),
]


def _children() -> list[psutil.Process]:
    return [p for p in psutil.Process().children(recursive=True) if "stockfish" in p.name().lower()]


def test_a_closed_engine_is_not_reopened_by_a_search_under_way() -> None:
    assert STOCKFISH is not None
    engine = Engine(STOCKFISH, threads=1, hash_mb=16, multipv=3)
    engine.open()
    before = {p.pid for p in _children()}
    raised: list[BaseException] = []

    def request() -> None:
        # A request of several searches: the coach's, then the next position's.
        try:
            engine.analyse(chess.Board(), depth=40, time_s=1.5, carry_best=True)
            board = chess.Board()
            board.push_san("e4")
            engine.analyse(board, depth=12)
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)

    worker = threading.Thread(target=request)
    worker.start()
    time.sleep(0.4)
    engine.close()  # the session's teardown
    worker.join(30)
    assert not worker.is_alive()
    time.sleep(0.3)
    assert engine._engine is None
    assert not ({p.pid for p in _children()} - before), "a Stockfish nobody closes"
    assert raised and isinstance(raised[0], EngineUnavailable)
    # Opened again by its owner, it works.
    engine.open()
    assert engine.analyse(chess.Board(), depth=4, multipv=1)
    engine.close()
