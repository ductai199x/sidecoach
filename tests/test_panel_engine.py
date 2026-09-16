"""The panel's live engine, driven headlessly from a recording (no window, no screen).

It runs the same capture -> pipeline -> engine loop `watch` runs, so feeding it real frames
must lock the board, bootstrap the start position, follow the moves, and surface all of that
as state and as debug-terminal lines -- the thread-safe surface the web page polls.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

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
    texts = [ln["text"] for ln in lines]
    chans = {ln["chan"] for ln in lines}

    assert {"sys", "vision", "game"} <= chans, f"channels seen: {chans}"
    assert any("board locked" in t for t in texts), "never reported locking the board"
    assert any("start position seen" in t for t in texts), "never bootstrapped the start position"
    moves = [
        ln
        for ln in lines
        if ln["chan"] == "game" and "(you" in ln["text"] or "(opponent" in ln["text"]
    ]
    assert moves, "no moves were surfaced to the terminal"
    assert all("ply" in ln["data"] and "uci" in ln["data"] for ln in moves), (
        "move records carry no data"
    )
    assert state["health"].get("stage"), "no health strip in the state"
    assert state["session_dir"] and (Path(state["session_dir"]) / "debug.jsonl").exists(), (
        "no debug.jsonl written for the session"
    )
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

    # An emitted record shows up once, structured, and polling past its cursor does not
    # repeat it.
    engine._emit("sys", "info", "hello", detail=42)
    got = api.poll(first["cursor"])
    assert len(got["lines"]) == 1
    record = got["lines"][0]
    assert record["chan"] == "sys" and record["level"] == "info" and record["text"] == "hello"
    assert record["data"] == {"detail": 42} and record["t"] >= 0
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


def test_ending_a_game_on_the_users_word_scores_it_from_their_side() -> None:
    outcome = PanelEngine.outcome_result
    assert outcome("resigned", "white") == ("0-1", "user resigned or lost on time")
    assert outcome("resigned", "black") == ("1-0", "user resigned or lost on time")
    assert outcome("won", "white") == ("1-0", "opponent resigned or lost on time")
    assert outcome("won", "black") == ("0-1", "opponent resigned or lost on time")
    assert outcome("draw", "white") == ("1/2-1/2", "draw agreed")
    assert outcome("forget", "black")[0] == "*"


def test_end_game_refuses_unknown_outcomes(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, voice=VoiceSettings(enabled=False))
    engine = PanelEngine(settings, analysis=False, coach=False, frame_source=iter(()))
    with pytest.raises(ValueError):
        engine.end_game("rage-quit")
    engine.end_game("draw")
    assert engine._commands.get_nowait() == ("end", "draw")
