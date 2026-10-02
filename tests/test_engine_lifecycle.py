"""The Stockfish process's life: a closed engine stays closed under a search already under
way. Real Stockfish. (A wedged engine's watchdog is tested with its wait driven by the test:
tests/test_analysis.py.) No sleeps: the test waits for the search's first depth, and for the
processes' exit, never for a time."""

from __future__ import annotations

import threading
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
    searching = threading.Event()  # the first search has reported a depth: it is under way

    def request() -> None:
        # A request of several searches: the coach's, then the next position's.
        try:
            engine.analyse(
                chess.Board(),
                depth=40,
                time_s=30.0,
                carry_best=True,
                on_depth=lambda _lines: searching.set(),
            )
            board = chess.Board()
            board.push_san("e4")
            engine.analyse(board, depth=12)
        except BaseException as exc:  # noqa: BLE001
            raised.append(exc)

    worker = threading.Thread(target=request)
    worker.start()
    assert searching.wait(60), "the search never began"  # a safety net, not a delay
    engine.close()  # the session's teardown, under a search
    worker.join(60)
    assert not worker.is_alive()
    assert engine._engine is None
    # Every process the engine started is gone (waited on, not slept for).
    started = [p for p in _children() if p.pid not in before]
    _gone, alive = psutil.wait_procs(started, timeout=30)
    assert not alive, "a Stockfish nobody closes"
    assert raised and isinstance(raised[0], EngineUnavailable)
    # Opened again by its owner, it works.
    engine.open()
    assert engine.analyse(chess.Board(), depth=4, multipv=1)
    engine.close()
