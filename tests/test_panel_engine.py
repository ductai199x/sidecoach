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
_SILENT = VoiceSettings()


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


def test_the_engine_and_coach_outlive_a_session(tmp_path: Path) -> None:
    """Everything a session needs is built when the panel opens (`warm`), before any
    Start; Start and Stop only start and stop reading the screen (Tai, 2026-09-21), so the
    same engine and coach glue serve one session after another; `close` ends them."""
    settings = Settings(data_dir=tmp_path, voice=_SILENT)
    engine = PanelEngine(
        settings, analysis=False, coach=False, record=False, frame_source=game_frames()
    )
    assert engine._assessing is None and engine._coaching is None
    assert engine.warm_status()["done"] is False
    engine.warm(background=False)
    assessing, coaching = engine._assessing, engine._coaching
    assert assessing is not None and coaching is not None, "warm built nothing"
    assert engine._warm_done.is_set()
    # The splash screen's view of it: eight stages, none left pending or loading.
    warm = engine.warm_status()
    assert warm["done"] is True and len(warm["stages"]) == 8
    assert {s["state"] for s in warm["stages"]} <= {"done", "off"}, warm["stages"]
    assert warm["stages"][-1]["name"] == "ready" and warm["stages"][-1]["state"] == "done"
    assert all("seconds" in s for s in warm["stages"])
    texts = [ln["text"] for ln in engine.poll(0)["lines"]]
    assert sum(t.startswith("engine:") for t in texts) == 1
    assert sum(t.startswith("coach:") for t in texts) == 1

    engine.start()
    assert engine._thread is not None
    engine._thread.join(timeout=90)
    assert not engine._thread.is_alive()
    assert engine._assessing is assessing and engine._coaching is coaching, (
        "a session must use what the warm-up built, and Stop must keep it"
    )
    engine.start()  # the second session: the same stages, nothing rebuilt or re-announced
    assert engine._thread is not None
    engine._thread.join(timeout=30)
    assert engine._assessing is assessing and engine._coaching is coaching
    texts = [ln["text"] for ln in engine.poll(0)["lines"]]
    assert sum(t.startswith("engine:") for t in texts) == 1, texts
    assert sum(t.startswith("coach:") for t in texts) == 1, texts

    engine.close()
    assert engine._assessing is None and engine._coaching is None


def test_a_start_without_a_warm_up_builds_the_stages_once(tmp_path: Path) -> None:
    """The CLI's replay and the tests start without a warm-up: the session builds the
    stages itself, and keeps them for the next one."""
    settings = Settings(data_dir=tmp_path, voice=_SILENT)
    engine = PanelEngine(settings, analysis=False, coach=False, record=False, frame_source=iter(()))
    engine.start()
    assert engine._thread is not None
    engine._thread.join(timeout=30)
    assessing = engine._assessing
    assert assessing is not None
    engine.start()
    assert engine._thread is not None
    engine._thread.join(timeout=30)
    assert engine._assessing is assessing
    engine.close()


def test_the_panel_starts_with_the_settings_voice(tmp_path: Path) -> None:
    """The saved voice is the one the panel speaks with from the start (Tai, 2026-09-21:
    the settings said bm_george, the panel came up with af_sarah -- the field's old
    default always won). A caller may still name one, as a test does."""
    settings = Settings(data_dir=tmp_path, voice=VoiceSettings(voice="bm_george"))
    engine = PanelEngine(settings, analysis=False, coach=False, record=False, frame_source=iter(()))
    assert engine.voice == "bm_george"
    assert engine.poll(0)["state"]["voice"] == "bm_george"
    named = PanelEngine(
        settings, analysis=False, coach=False, record=False, frame_source=iter(()), voice="af_bella"
    )
    assert named.voice == "af_bella"


def test_recording_is_a_saved_opt_in_applied_on_the_spot(tmp_path: Path) -> None:
    """The app starts the sidecar with --no-record --no-persist; the setting
    capture.record (Settings > Screen, or Debug > Record sessions, the same switch) turns
    recording on for the running session or the next Start, and stays until turned off
    (Tai, 2026-09-21: off by default, but there must be a way to record a bug;
    2026-09-22: an option in Settings, working on the spot)."""
    from sidecoach.config import load_settings

    settings = Settings(data_dir=tmp_path, voice=_SILENT)
    settings._source = tmp_path / "sidecoach.toml"
    engine = PanelEngine(
        settings, analysis=False, coach=False, record=False, persist=False, frame_source=iter(())
    )
    state = engine.poll(0)["state"]
    assert state["record_wanted"] is False and state["recording_dir"] is None
    # The Debug menu writes the setting.
    reply = engine.set_record(True)
    assert reply == {"ok": True, "on": True, "running": False, "folder": None}
    assert engine.poll(0)["state"]["record_wanted"] is True
    assert load_settings(tmp_path / "sidecoach.toml").capture.record is True
    # The session folder follows the setting, not the launch flag.
    assert engine._session_dir(live=True).parent.name == "recordings"
    # Settings > Save applies at once.
    new = settings.model_copy(deep=True)
    new.capture.record = False
    assert "capture.record" in engine.apply_settings(new, ["capture.record"])
    assert engine.poll(0)["state"]["record_wanted"] is False
    assert engine._session_dir(live=True).parent.name == "sessions"
    assert any("from the next Start" in ln["text"] for ln in engine.poll(0)["lines"])


class _Recogniser:
    """Speech recognition that reads the words off the clip's length: one word per 0.4 s.
    A clip of exactly `held` samples waits for `gate` first (a slow partial)."""

    def __init__(self) -> None:
        import threading

        self.gate = threading.Event()
        self.gate.set()
        self.held = -1

    def transcribe(self, audio: object) -> str:
        n = len(audio)  # type: ignore[arg-type]
        if n == self.held:
            self.gate.wait(2)
        return " ".join(["hello"] * max(1, int(n / 16000 / 0.4)))

    def healthcheck(self) -> tuple[bool, str]:
        return True, "fake"


def test_dictation_streams_into_the_box_and_sends_after_a_second_of_quiet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What is said reaches the page as it is recognised, and is asked of the coach a second
    after the utterance ends -- unless more is said first, which appends (Tai, 2026-09-22:
    no 'heard:' line, the words in the text box, streamed, then sent)."""
    import time

    import numpy as np

    from sidecoach.ui import panel as panel_mod

    monkeypatch.setattr(panel_mod, "DICTATION_SEND_DELAY_S", 0.3)
    settings = Settings(data_dir=tmp_path, voice=_SILENT)
    engine = PanelEngine(settings, analysis=False, coach=False, record=False, frame_source=iter(()))
    engine._stt = _Recogniser()  # type: ignore[assignment]

    def dictation() -> dict:
        return engine.poll(0)["state"]["dictation"]

    assert dictation() == {"text": "", "live": False, "sends_at": None}
    # The key goes down: the box says it is listening.
    engine._dictation_hold()
    assert dictation()["live"] is True
    # Partials while the key is held: the words so far, live.
    engine._on_partial(np.zeros(16000, dtype=np.float32))
    deadline = time.monotonic() + 2
    while dictation()["text"] != "hello hello" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert dictation() == {"text": "hello hello", "live": True, "sends_at": None}
    # The key comes up: the final result replaces the partial and the send is armed.
    engine._on_utterance(np.zeros(16000 * 2, dtype=np.float32), 2.0)
    d = dictation()
    assert d["text"] == "hello hello hello hello hello" and d["live"] is False
    assert d["sends_at"] is not None
    assert engine._commands.empty()
    # Pressed again before it sends: it waits, and the next utterance appends.
    engine._dictation_hold()
    assert dictation()["sends_at"] is None
    engine._on_utterance(np.zeros(16000 // 2, dtype=np.float32), 0.5)
    assert dictation()["text"] == "hello hello hello hello hello hello"
    # ...and after the quiet, one ask with everything, and the box is clear.
    name, payload = engine._commands.get(timeout=2)
    assert (name, payload) == ("ask_spoken", "hello hello hello hello hello hello")
    assert dictation() == {"text": "", "live": False, "sends_at": None}
    # A partial still being recognised when its utterance ends is dropped: the final
    # result has those words, and they must not show twice.
    stt = engine._stt
    assert isinstance(stt, _Recogniser)
    stt.held = 16000
    stt.gate.clear()
    engine._on_partial(np.zeros(16000, dtype=np.float32))  # blocks in transcribe
    engine._on_utterance(np.zeros(16000 * 2, dtype=np.float32), 2.0)
    stt.gate.set()
    time.sleep(0.2)
    assert dictation()["text"] == "hello hello hello hello hello"
    engine.cancel_dictation()
    # No 'heard:' line at info level any more: the words are on the page, not in the terminal.
    assert not any(
        ln["text"].startswith("heard:") and ln["level"] == "info" for ln in engine.poll(0)["lines"]
    )


def test_typing_over_a_dictation_keeps_the_words_and_stops_the_send(tmp_path: Path) -> None:
    import numpy as np

    settings = Settings(data_dir=tmp_path, voice=_SILENT)
    engine = PanelEngine(settings, analysis=False, coach=False, record=False, frame_source=iter(()))
    engine._stt = _Recogniser()  # type: ignore[assignment]
    engine._on_utterance(np.zeros(16000, dtype=np.float32), 1.0)
    assert engine.poll(0)["state"]["dictation"]["sends_at"] is not None
    assert engine.cancel_dictation() == {"ok": True, "text": "hello hello"}
    assert engine.poll(0)["state"]["dictation"] == {"text": "", "live": False, "sends_at": None}
    import time

    time.sleep(1.2)
    assert engine._commands.empty()
