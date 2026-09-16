"""The panel's structured debug log: a ring for the page, a JSONL file for later."""

from __future__ import annotations

import json
from pathlib import Path

from sidecoach.ui.log import DebugLog, elide


def test_records_are_structured_and_polled_incrementally() -> None:
    log = DebugLog()
    a = log.emit("game", "info", "1. e4  (you)", ply=1, uci="e2e4")
    b = log.emit("coach", "debug", "quiet: opponent to move")
    assert (a.seq, b.seq) == (1, 2)
    assert log.cursor == 2

    got = log.since(0)
    assert [r["text"] for r in got] == ["1. e4  (you)", "quiet: opponent to move"]
    assert got[0]["chan"] == "game" and got[0]["level"] == "info"
    assert got[0]["data"] == {"ply": 1, "uci": "e2e4"}
    assert got[1]["data"] == {}
    assert all(r["t"] >= 0 for r in got)
    # Polling past the cursor returns nothing new; from the middle, only the tail.
    assert log.since(2) == []
    assert [r["seq"] for r in log.since(1)] == [2]


def test_attach_writes_the_backlog_and_then_every_record(tmp_path: Path) -> None:
    log = DebugLog()
    log.emit("sys", "info", "before the file existed")
    path = tmp_path / "session" / "debug.jsonl"
    log.attach(path)  # the session folder is only known once the engine starts
    log.emit("sys", "warn", "after", code=7)
    log.close()

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert [ln["text"] for ln in lines] == ["before the file existed", "after"]
    assert lines[1]["level"] == "warn" and lines[1]["data"] == {"code": 7}


def test_ring_keeps_only_the_most_recent() -> None:
    log = DebugLog(maxlen=3)
    for i in range(5):
        log.emit("sys", "info", f"line {i}")
    assert [r["text"] for r in log.since(0)] == ["line 2", "line 3", "line 4"]
    assert log.cursor == 5


def test_elide_cuts_long_lines_and_collapses_whitespace() -> None:
    assert elide("short") == "short"
    long = "word " * 40
    cut = elide(long, 20)
    assert len(cut) <= 20 and cut.endswith("…")
    assert elide("a\n  b\t c") == "a b c"
