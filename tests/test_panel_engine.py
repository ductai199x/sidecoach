"""The panel's live engine, driven headlessly from a recording (no window, no screen).

It runs the same capture -> pipeline -> engine loop `watch` runs, so feeding it real frames
must lock the board, bootstrap the start position, follow the moves, and surface all of that
as state and as debug-terminal lines -- the thread-safe surface the web page polls.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from sidecoach.config import Settings, VoiceSettings
from sidecoach.replay.recording import Recording
from sidecoach.ui.panel import PanelApi, PanelEngine

REC = Path("data/recordings/g01-david-green")


def _frames(limit: int) -> Iterator[tuple[int, float, object]]:
    for i, (tick, frame) in enumerate(Recording(REC).frames()):
        if i >= limit:
            return
        yield (i, tick.ts, frame)


def test_engine_follows_a_recorded_game_and_reports_it(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, voice=VoiceSettings(enabled=False))
    engine = PanelEngine(
        settings,
        analysis=False,  # no Stockfish needed in a unit test
        coach=False,  # no LLM needed in a unit test
        record=False,
        frame_source=_frames(520),
    )
    engine.start()
    assert engine._thread is not None
    engine._thread.join(timeout=90)
    assert not engine._thread.is_alive(), "engine loop did not finish the recording in time"

    snap = engine.poll(0)
    state, lines = snap["state"], snap["lines"]
    levels = {ln["level"] for ln in lines}
    texts = [ln["text"] for ln in lines]

    assert any("board locked" in t for t in texts), "never reported locking the board"
    assert any("start position seen" in t for t in texts), "never bootstrapped the start position"
    assert "move" in levels, "no moves were surfaced to the terminal"
    assert state["fen"], "no position in the state"
    assert state["ply"] >= 8, f"expected several plies, got {state['ply']}"
    assert state["following"] or state["moves"], "engine never followed a game"
    assert state["running"] is False, "engine should have stopped after the recording ended"


def test_api_poll_is_incremental_and_serialisable(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, voice=VoiceSettings(enabled=False))
    engine = PanelEngine(settings, analysis=False, coach=False, frame_source=iter(()))
    api = PanelApi(engine)

    first = api.poll(0)
    assert set(first) == {"state", "lines", "cursor"}
    assert first["state"]["voice"] == "af_sarah"
    assert "af_sarah" in first["state"]["voices"]
    # The state must be plain JSON-friendly data (no chess/Side objects leaking through).
    import json

    json.dumps(first)

    # An emitted line shows up once, and polling past its cursor does not repeat it.
    engine._emit("info", "hello")
    got = api.poll(first["cursor"])
    assert [ln["text"] for ln in got["lines"]] == ["hello"]
    assert api.poll(got["cursor"])["lines"] == []


def test_flip_and_voice_controls_do_not_raise_without_a_game(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, voice=VoiceSettings(enabled=False))
    engine = PanelEngine(settings, analysis=False, coach=False, frame_source=iter(()))
    api = PanelApi(engine)
    assert api.set_voice("am_adam") is True
    assert engine.voice == "am_adam"
    assert api.set_muted(True) is True
    assert engine.muted is True
    # Queuing controls before the loop runs must not raise.
    assert api.flip() is True
    assert api.analyze() is True
    assert api.hint() is True
