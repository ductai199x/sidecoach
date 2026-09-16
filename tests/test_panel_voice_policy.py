"""The voice never talks about a position the board has left behind.

A fast game, or a replay run faster than it was played, can move on while a line is still
being synthesised or spoken. The engine decides, per coach message, whether to speak it,
skip it, or cut in over what is playing -- and the replay is paced so that, at speed 1, the
coach and the voice have the time they had in the real game.
"""

from __future__ import annotations

import time
from pathlib import Path

from sidecoach.config import Settings, VoiceSettings
from sidecoach.ui.panel import STALE_PLIES, PanelEngine
from sidecoach.ui.server import make_engine

REC = Path("data/recordings/g01-david-green")
LONG = "Your knight on f5 is loose; the queen check on g6 wins it. What defends it?"
SHORT = "exf5: good."


def test_a_message_about_a_position_the_board_left_is_not_spoken() -> None:
    act = PanelEngine.voice_action
    assert act("post_mistake", LONG, ply=10, ply_now=10, busy=False) == "speak"
    assert act("post_mistake", LONG, ply=10, ply_now=10 + STALE_PLIES - 1, busy=False) == "speak"
    assert act("post_mistake", LONG, ply=10, ply_now=10 + STALE_PLIES, busy=False) == "stale"
    # The answer to the user's own question is never stale, however far the board has moved.
    assert act("user_ask", LONG, ply=None, ply_now=40, busy=True) == "supersede"
    assert act("user_ask", LONG, ply=10, ply_now=40, busy=False) == "speak"


def test_short_verdicts_yield_to_a_busy_voice_and_substance_cuts_in() -> None:
    act = PanelEngine.voice_action
    assert act("post_move", SHORT, ply=4, ply_now=4, busy=False) == "speak"
    assert act("post_move", SHORT, ply=4, ply_now=4, busy=True) == "skip"
    assert act("decisive_question", LONG, ply=4, ply_now=4, busy=True) == "supersede"
    assert act("stuck", LONG, ply=4, ply_now=5, busy=True) == "supersede"


def test_speak_records_the_action_and_stops_before_cutting_in(tmp_path: Path) -> None:
    class FakeTTS:
        name = "fake"

        def __init__(self) -> None:
            self.calls: list[str] = []
            self.busy = False

        def healthcheck(self) -> tuple[bool, str]:
            return True, "fake"

        def say(self, text: str) -> None:
            self.calls.append("say:" + text)

        def stop(self) -> None:
            self.calls.append("stop")

    engine = PanelEngine(
        Settings(data_dir=tmp_path, voice=VoiceSettings(enabled=False)),
        analysis=False,
        coach=False,
        frame_source=iter(()),
    )
    fake = FakeTTS()
    engine._tts = fake  # type: ignore[assignment]
    engine._state["ply"] = 12

    assert engine._speak("post_mistake", LONG, ply=12) == "speak"
    assert fake.calls == ["say:" + LONG]
    fake.calls.clear()
    fake.busy = True
    assert engine._speak("stuck", LONG, ply=12) == "supersede"
    assert fake.calls == ["stop", "say:" + LONG]
    fake.calls.clear()
    assert engine._speak("post_move", SHORT, ply=12) == "skip"
    assert engine._speak("post_move", SHORT, ply=9) == "stale"
    assert fake.calls == []
    engine.muted = True
    assert engine._speak("stuck", LONG, ply=12) == "muted"


def test_replay_is_paced_to_its_timestamps(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, voice=VoiceSettings(enabled=False))
    # Unpaced: the first frames arrive as fast as they can be read.
    fast = make_engine(settings, replay=REC, speed=0)
    assert fast.frame_source is not None
    t0 = time.monotonic()
    first = [next(iter(fast.frame_source)) for _ in range(1)]
    assert first and time.monotonic() - t0 < 2.0
    # Paced at 4x: the gap between two frames a second apart in the recording is ~0.25s of
    # wall time, not zero.
    paced = make_engine(settings, replay=REC, speed=4.0)
    assert paced.frame_source is not None
    src = iter(paced.frame_source)
    a = next(src)
    t0 = time.monotonic()
    b = next(src)
    while b[1] - a[1] < 1.0:  # walk until the recording itself has advanced a second
        b = next(src)
    elapsed = time.monotonic() - t0
    expected = (b[1] - a[1]) / 4.0
    assert elapsed >= expected * 0.6, f"replay ran unpaced: {elapsed:.2f}s for {expected:.2f}s"
