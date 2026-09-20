"""The panel's live engine, driven headlessly from rendered frames (no window, no screen).

It runs the same capture -> pipeline -> engine loop `watch` runs, so feeding it the frames
of a game must lock the board, bootstrap the start position, follow the moves, and surface
all of that as state and as debug-terminal lines -- the thread-safe surface the web page
polls.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sidecoach.config import Settings, VoiceSettings
from sidecoach.ui.panel import PanelApi, PanelEngine
from tests.synth import game_frames

# No voice models: the panel is silent and deaf (see PanelEngine._voice_settings).
_SILENT = VoiceSettings(provider="null", stt_provider="null")


def test_engine_follows_a_game_and_reports_it(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, voice=_SILENT)
    engine = PanelEngine(
        settings,
        analysis=False,  # no Stockfish needed in a unit test
        coach=False,  # no LLM needed in a unit test
        record=False,
        frame_source=game_frames(),
    )
    engine.start()
    assert engine._thread is not None
    engine._thread.join(timeout=90)
    assert not engine._thread.is_alive(), "engine loop did not finish the game in time"

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
    settings = Settings(data_dir=tmp_path, voice=_SILENT)
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
    settings = Settings(data_dir=tmp_path, voice=_SILENT)
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
    settings = Settings(data_dir=tmp_path, voice=_SILENT)
    engine = PanelEngine(settings, analysis=False, coach=False, frame_source=iter(()))
    with pytest.raises(ValueError):
        engine.end_game("rage-quit")
    engine.end_game("draw")
    assert engine._commands.get_nowait() == ("end", "draw")


def test_legal_move_dots_are_not_a_degraded_reading(tmp_path: Path) -> None:
    """A picked-up rook puts a dot on a dozen squares, each too small to be a piece and read
    as unknown; the first live game logged "reading degraded" on nearly every endgame move
    for it. Only covered cells and unknowns that are piece-sized or hovered count."""
    from types import SimpleNamespace

    from sidecoach.ui.panel import DEGRADED_FRAMES
    from sidecoach.vision.pieces import CellReading

    def cell(kind: str, fraction: float = 0.0, **kw: object) -> CellReading:
        return CellReading(kind, None, None, 0.0, fraction, 0.0, **kw)  # type: ignore[arg-type]

    settings = Settings(data_dir=tmp_path, voice=_SILENT)
    engine = PanelEngine(settings, analysis=False, coach=False, frame_source=iter(()))
    dots = [[cell("empty") for _ in range(8)] for _ in range(8)]
    for c in range(8):  # a rook's whole rank of dots, plus its own hovered square
        dots[3][c] = cell("unknown", 0.08)
    dots[3][0] = cell("unknown", 0.3, hover=True)
    degraded = False
    for _ in range(DEGRADED_FRAMES + 2):
        degraded = engine._check_degraded(SimpleNamespace(readings=dots), degraded)
    assert degraded is False
    # A dialog fading in covers cells; that is degraded.
    dialog = [[cell("empty") for _ in range(8)] for _ in range(8)]
    for r in range(3):
        for c in range(3):
            dialog[r][c] = cell("unknown", 0.5, covered=True)
    for _ in range(DEGRADED_FRAMES + 2):
        degraded = engine._check_degraded(SimpleNamespace(readings=dialog), degraded)
    assert degraded is True


def test_the_profile_records_the_game_and_the_review_reaches_the_page(tmp_path: Path) -> None:
    """Without an engine there are no verdicts, but the game itself -- its moves and its
    end -- reaches the profile store, the review card and the session summary; a replay
    (an injected frame source) keeps the profile in memory and writes no file."""
    settings = Settings(data_dir=tmp_path, voice=_SILENT)
    engine = PanelEngine(
        settings, analysis=False, coach=False, record=False, frame_source=game_frames()
    )
    engine.start()
    assert engine._thread is not None
    engine._thread.join(timeout=90)
    snap = engine.poll(0)
    state, lines = snap["state"], snap["lines"]
    assert state["profile"] is not None and state["profile"]["games"] in (0, 1)
    assert not (tmp_path / "profile.sqlite").exists(), "a replay must not write the profile"
    texts = [ln["text"] for ln in lines]
    if any("game over" in t for t in texts):
        assert state["review"] is not None and state["review"]["moves"] == 0  # no engine
        assert any(ln["chan"] == "coach" and "review" in ln.get("data", {}) for ln in lines)
        assert any(t.startswith("session:") for t in texts)


def test_a_correction_moves_through_the_api(tmp_path: Path) -> None:
    import chess

    from sidecoach.events import GameStarted, Side
    from sidecoach.profile.profiler import Profiler
    from sidecoach.profile.store import Store

    settings = Settings(data_dir=tmp_path, voice=_SILENT)
    engine = PanelEngine(settings, analysis=False, coach=False, frame_source=iter(()))
    api = PanelApi(engine)
    assert api.correct("g", 0, 3, "time pressure") is False  # no session, no profile
    store = Store(None)
    profiler = Profiler(store, session_id="s")
    profiler.on_event(
        GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.STARTING_FEN), 0.0
    )
    store.add_evidence(
        7,
        game_id="g",
        segment=0,
        ply=3,
        phase="opening",
        expected_loss=0.12,
        cause="positional misjudgment",
        motif=None,
        time_pressure=False,
    )
    engine._profiler = profiler
    engine._set(
        review={
            "game_id": "g",
            "turning_points": [{"segment": 0, "ply": 3, "cause": "positional misjudgment"}],
        }
    )
    assert api.correct("g", 0, 3, "time pressure") is True
    assert store.evidence("g")[0].cause == "time pressure"
    assert engine.poll(0)["state"]["review"]["turning_points"][0]["cause"] == "time pressure"
    with pytest.raises(ValueError):
        api.correct("g", 0, 3, "bad luck")
