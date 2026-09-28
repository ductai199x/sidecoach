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
from sidecoach.events import Region
from sidecoach.replay.recording import Recorder
from sidecoach.ui.panel import PanelEngine
from sidecoach.ui.server import make_engine
from tests.synth import game_frames

# No voice models: the panel is silent and deaf (PanelEngine.voice_models, which
# SIDECOACH_VOICE_MODELS=0 turns off under pytest: tests/conftest.py).
_SILENT = VoiceSettings()

LONG = "Your knight on f5 is loose; the queen check on g6 wins it. What defends it?"
SHORT = "exf5: good."


def test_a_message_about_a_position_the_board_left_is_not_spoken() -> None:
    act = PanelEngine.voice_action
    # A verdict on the user's move (made from ply 10) is still the news after the move (11)
    # and the bot's instant reply (12) -- the normal case against a bot -- and stale only
    # once the user has moved again (13).
    assert act("post_mistake", LONG, ply=10, ply_now=10, busy=False) == "speak"
    assert act("post_mistake", LONG, ply=10, ply_now=12, busy=False) == "speak"
    assert act("post_move", SHORT, ply=10, ply_now=12, busy=False) == "speak"
    assert act("post_mistake", LONG, ply=10, ply_now=13, busy=False) == "stale"
    # A hint about the live position is moot as soon as the user moves.
    assert act("stuck", LONG, ply=10, ply_now=10, busy=False) == "speak"
    assert act("stuck", LONG, ply=10, ply_now=11, busy=False) == "stale"
    assert act("decisive_question", LONG, ply=10, ply_now=11, busy=False) == "stale"
    # The answer to the user's own question is never stale, however far the board has moved.
    assert act("user_ask", LONG, ply=None, ply_now=40, busy=True) == "supersede"
    assert act("user_ask", LONG, ply=10, ply_now=40, busy=False) == "speak"
    # Nor is the review: the game is over. After a mate it is written on the last position
    # with facts, two plies behind the board (the mating move and the one before have none).
    assert act("review", LONG, ply=12, ply_now=14, busy=False) == "speak"
    assert act("review", LONG, ply=12, ply_now=14, busy=True, playing=("post_move", 12)) == "speak"


def test_short_verdicts_yield_to_a_busy_voice_and_substance_cuts_in() -> None:
    act = PanelEngine.voice_action
    assert act("post_move", SHORT, ply=4, ply_now=4, busy=False) == "speak"
    assert act("post_move", SHORT, ply=4, ply_now=4, busy=True) == "skip"
    # Nothing known about what is playing: substance cuts in, as before.
    assert act("decisive_question", LONG, ply=4, ply_now=4, busy=True) == "supersede"
    assert act("stuck", LONG, ply=4, ply_now=4, busy=True) == "supersede"
    # What is playing is about a moment the board has left: cut in.
    assert (
        act("post_mistake", LONG, ply=6, ply_now=7, busy=True, playing=("post_move", 3))
        == "supersede"
    )
    # What is playing is still about the board as it stands: wait behind it -- the word
    # on the last move, the review, the practice, each heard whole at a game's end.
    assert act("review", LONG, ply=88, ply_now=88, busy=True, playing=("post_move", 87)) == "speak"
    assert act("review", LONG, ply=None, ply_now=88, busy=True, playing=("review", 88)) == "speak"
    assert act("stuck", LONG, ply=4, ply_now=4, busy=True, playing=("post_move", 3)) == "speak"
    # The user's own question never waits.
    assert act("user_ask", LONG, ply=4, ply_now=4, busy=True, playing=("review", 4)) == "supersede"


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
        Settings(data_dir=tmp_path, voice=_SILENT),
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
    assert engine._speak("stuck", LONG, ply=12) == "speak", "the lesson playing is current"
    assert fake.calls == ["say:" + LONG]
    fake.calls.clear()
    engine._state["ply"] = 16  # the board moved on past what is playing
    assert engine._speak("stuck", LONG, ply=16) == "supersede"
    assert fake.calls == ["stop", "say:" + LONG]
    fake.calls.clear()
    engine._state["ply"] = 12
    engine._voicing = ("post_move", 11)
    assert engine._speak("post_move", SHORT, ply=12) == "skip"
    assert engine._speak("post_move", SHORT, ply=8) == "stale"
    assert fake.calls == []
    engine.muted = True
    assert engine._speak("stuck", LONG, ply=12) == "muted"


def test_replay_is_paced_to_its_timestamps(tmp_path: Path) -> None:
    """A recording of a rendered game whose ticks are a second apart, replayed at 4x:
    the frames come a quarter of a second apart in wall time, not as fast as they read."""
    frames = [frame for _, _, frame in game_frames(("e4", "e5"), frames_per_state=1)]
    region = Region(0, 0, frames[0].shape[1], frames[0].shape[0])
    rec = tmp_path / "rec"
    recorder = Recorder(rec, monitor=0, region=region, fps=1, keyframe_interval_s=None)
    for i, frame in enumerate(frames):
        recorder.write(frame, ts=float(i), store=True)
    recorder.close()
    settings = Settings(data_dir=tmp_path, voice=_SILENT)

    fast = make_engine(settings, replay=rec, speed=0)
    assert fast.frame_source is not None
    t0 = time.monotonic()
    assert [ts for _, ts, _ in fast.frame_source] == [0.0, 1.0, 2.0]
    # Two seconds of recording: unpaced it reads in milliseconds, paced at 1x it would
    # take two seconds. The bound is loose because the suite runs on every core at once
    # and decoding three frames has taken most of half a second under that load.
    assert time.monotonic() - t0 < 1.0, "unpaced replay waited"

    paced = make_engine(settings, replay=rec, speed=4.0)
    assert paced.frame_source is not None
    src = iter(paced.frame_source)
    t0 = time.monotonic()  # the pace is kept from the first frame, whatever decoding cost
    next(src)
    next(src)
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.2, f"replay ran unpaced: {elapsed:.2f}s for a second at 4x"


def test_the_way_from_a_move_to_its_first_sound_is_timed(tmp_path: Path) -> None:
    """The status strip's move->speech latency: the move is noted when committed, the
    reply's first line remembered when queued, and the time taken the moment the voice
    says the line's audio has started; amortised over the session."""
    import time

    from sidecoach.events import Trigger
    from sidecoach.ui.panel import PanelEngine

    engine = PanelEngine(
        Settings(data_dir=tmp_path, voice=_SILENT),
        analysis=False,
        coach=False,
        frame_source=iter(()),
    )
    engine._note_move(1)
    assert engine._latency()["move_to_speech_s"] is None
    sink = engine._stream_sink(Trigger.POST_MOVE, 0)
    assert sink is not None
    time.sleep(0.05)
    assert sink("e4: best.")  # the first line of the word on ply 1, queued
    assert "e4: best." in engine._awaiting_audio
    time.sleep(0.05)
    engine._on_audio_starts("e4: best.")  # the voice: playing now
    lat = engine._latency()
    assert lat["move_to_speech_n"] == 1 and lat["move_to_speech_s"] is not None
    assert 0.08 <= lat["move_to_speech_s"] <= 2.0, lat
    assert lat["move_to_speech_last_s"] == lat["move_to_speech_s"]
    # A line nobody waited for, or a stuck hint, is not a move's latency.
    engine._on_audio_starts("something else")
    hint = engine._stream_sink(Trigger.STUCK, 1)
    assert hint is not None and hint("Look at the centre.")
    assert engine._latency()["move_to_speech_n"] == 1
    # The model's first token, amortised the same way.
    engine._note_first_token(1.0)
    engine._note_first_token(2.0)
    lat = engine._latency()
    assert (
        lat["first_token_s"] == 1.5
        and lat["first_token_last_s"] == 2.0
        and lat["first_token_n"] == 2
    )
