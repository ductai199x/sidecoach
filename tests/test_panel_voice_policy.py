"""The voice never talks about a position the board has left behind.

A fast game, or a replay run faster than it was played, can move on while a line is still
being synthesised or spoken. The engine decides, per coach message, whether to speak it,
skip it, or cut in over what is playing -- and the replay is paced so that, at speed 1, the
coach and the voice have the time they had in the real game.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from sidecoach.config import Settings
from sidecoach.events import Region
from sidecoach.replay.recording import Recorder
from sidecoach.ui import server
from sidecoach.ui.panel import PanelEngine
from sidecoach.ui.server import make_engine
from tests.fakes import ManualClock
from tests.synth import game_frames

LONG = "Your knight on f5 is loose; the queen check on g6 wins it. What defends it?"
SHORT = "exf5: good."


class FakeTTS:
    """A voice that writes down what it is asked to do."""

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


def test_the_voice_holds_its_word_in_a_fast_game_and_says_it_when_the_player_slows() -> None:
    """In a fast game a word on a move is shown, not said; an ask and the review are still
    said aloud, and a slower game has its words said again (`fast`)."""
    va = PanelEngine.voice_action
    assert va("post_move", "Qxf7#: mate.", 7, 7, False, None, True) == "text"
    assert va("post_mistake", "Bb7 was a miss.", 7, 7, False, None, True) == "text"
    assert va("user_ask", "Yes.", 7, 7, False, None, True) == "speak", "an ask is answered aloud"
    assert va("review", "You won.", 7, 7, False, None, True) == "speak"
    assert va("post_move", "Qxf7#: mate.", 7, 7, False, None, False) == "speak"


def test_speak_records_the_action_and_stops_before_cutting_in(
    make_panel: Callable[..., PanelEngine],
) -> None:
    engine = make_panel()
    engine.set_voice_pace("simple")  # the simple voice policy, the fallback
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


def test_a_word_held_in_a_fast_game_is_said_once_the_player_slows(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """A word the fast game kept on the page is said once the player has been thinking in
    that position for PACE_RESUME_S (`_say_held_word`), and said once; one about a position
    the board has left, or held when the game is over, is dropped (the parity audit,
    2026-09-29: pinned before only by the phone)."""
    from types import SimpleNamespace

    from sidecoach.ui.panel import PACE_RESUME_S
    from tests.coaching import SCHOLAR_MOVES, offline_coach, play, start_game

    coach = offline_coach()
    start_game(coach)
    play(coach, SCHOLAR_MOVES, think_ms=1000)  # a second a move; Nf6, ply 6, at 6 s
    assert coach.fast_now(7.0)
    engine = make_panel()
    engine.set_voice_pace("simple")  # the simple voice policy, the fallback
    fake = FakeTTS()
    engine._tts = fake  # type: ignore[assignment]
    coaching = SimpleNamespace(coach=coach)
    engine._coaching = coaching
    engine._loop_ts = 7.0
    engine._state["ply"] = 6
    assert engine._speak("post_move", "Nf6: good.", ply=5) == "text"
    assert fake.calls == []
    engine._say_held_word(coaching, 7.0)  # a second into the position: still fast
    assert fake.calls == []
    assert engine.poll()["state"]["caption_again"] == 0
    engine._say_held_word(coaching, 6.0 + PACE_RESUME_S + 1)  # the game slowed
    assert fake.calls == ["say:Nf6: good."]
    # The box, faded since the word was captioned, comes back while it is said (finding 20).
    assert engine.poll()["state"]["caption_again"] == 1
    engine._say_held_word(coaching, 20.0)
    assert len(fake.calls) == 1, "said once"
    # A word about a position the board has left is dropped, not said late.
    assert engine._speak("stuck", "Look at f7.", ply=6) == "text"
    engine._state["ply"] = 7
    engine._say_held_word(coaching, 30.0)
    engine._state["ply"] = 6
    engine._say_held_word(coaching, 30.0)
    assert len(fake.calls) == 1, "a hint the board left is dropped"
    # Nor after the game: the review speaks, not a late verdict.
    assert engine._speak("post_move", "Nf6: good.", ply=5) == "text"
    coach.finished = True
    engine._say_held_word(coaching, 30.0)
    coach.finished = False
    engine._say_held_word(coaching, 30.0)
    assert len(fake.calls) == 1, "a word held at the game's end is dropped"


def test_a_streamed_reply_in_a_fast_game_is_kept_and_held_whole_for_the_pace(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """A streamed reply whose first sentence met a fast game was declined: the model's reply was
    thrown away and the template shown in its place (Tai, 2026-10-02: kept, never replaced).
    It is taken sentence by sentence, captioned, not said, and held whole for the pace like a
    word on the page: said once the player slows."""
    from types import SimpleNamespace

    from sidecoach.events import Trigger
    from sidecoach.ui.panel import PACE_RESUME_S
    from tests.coaching import SCHOLAR_MOVES, offline_coach, play, start_game

    coach = offline_coach()
    start_game(coach)
    play(coach, SCHOLAR_MOVES, think_ms=1000)  # a second a move: a fast game
    engine = make_panel()
    engine.set_voice_pace("simple")  # the simple voice policy, the fallback
    fake = FakeTTS()
    engine._tts = fake  # type: ignore[assignment]
    coaching = SimpleNamespace(coach=coach)
    engine._coaching = coaching
    engine._loop_ts = 7.0
    engine._state["ply"] = 6
    sink = engine._stream_sink(Trigger.POST_MISTAKE, 5)
    assert sink is not None
    assert sink("Nf6 left f7 to the queen.") is True, "kept, not declined"
    assert sink("Qxf7 is mate now.") is True
    assert fake.calls == [], "a fast game: not said yet"
    held = engine._held_word
    assert held is not None and held[1] == "Nf6 left f7 to the queen. Qxf7 is mate now."
    engine._say_held_word(coaching, 6.0 + PACE_RESUME_S + 1)  # the game slowed
    assert fake.calls == ["say:Nf6 left f7 to the queen. Qxf7 is mate now."]
    # Cut off while held (the position left): the rest is captioned and nothing is said later.
    sink = engine._stream_sink(Trigger.POST_MISTAKE, 5)
    assert sink is not None and sink("Look at f7.")
    engine._position_moved_on("takeback")
    assert sink("Their king has no square.") is True
    assert engine._held_word is None


def test_replay_is_paced_to_its_timestamps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A recording of a rendered game whose ticks are a second apart, replayed at 4x:
    the frames come a quarter of a second apart in wall time, not as fast as they read.
    The replay keeps its pace on a clock the test owns (tests/fakes.py ManualClock): reading
    a frame takes no time on it, and every wait is written down."""
    frames = [frame for _, _, frame in game_frames(("e4", "e5"), frames_per_state=1)]
    region = Region(0, 0, frames[0].shape[1], frames[0].shape[0])
    rec = tmp_path / "rec"
    recorder = Recorder(rec, monitor=0, region=region, fps=1, keyframe_interval_s=None)
    for i, frame in enumerate(frames):
        recorder.write(frame, ts=float(i), store=True)
    recorder.close()
    settings = Settings(data_dir=tmp_path)
    clock = ManualClock()
    monkeypatch.setattr(server, "time", clock.time_module())

    fast = make_engine(settings, replay=rec, speed=0)
    assert fast.frame_source is not None
    t0 = clock.now
    assert [ts for _, ts, _ in fast.frame_source] == [0.0, 1.0, 2.0]
    # Two seconds of recording: unpaced it never waits; paced at 1x it would take two seconds.
    assert clock.sleeps == [], "unpaced replay waited"
    assert clock.now - t0 < 1.0, "unpaced replay waited"

    paced = make_engine(settings, replay=rec, speed=4.0)
    assert paced.frame_source is not None
    src = iter(paced.frame_source)
    t0 = clock.now  # the pace is kept from the first frame, whatever decoding cost
    next(src)
    assert clock.now == t0, "the first frame comes at once"
    next(src)
    elapsed = clock.now - t0
    assert elapsed >= 0.2, f"replay ran unpaced: {elapsed:.2f}s for a second at 4x"
    assert elapsed == 0.25, "a second of recording at 4x: a quarter of a second"
    next(src)
    assert clock.now - t0 == 0.5, "each frame at its own time, from the first"


def test_the_way_from_a_move_to_its_first_sound_is_timed(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status strip's move->speech latency: the move is noted when committed, the
    reply's first line remembered when queued, and the time taken the moment the voice
    says the line's audio has started; amortised over the session. The panel's clock is one
    the test moves: the time taken is exactly the time that passed."""
    from sidecoach.events import Trigger
    from sidecoach.ui import panel as panel_mod

    engine = make_panel()
    clock = ManualClock()
    monkeypatch.setattr(panel_mod, "time", clock.time_module())
    engine._note_move(1)
    assert engine._latency()["move_to_speech_s"] is None
    sink = engine._stream_sink(Trigger.POST_MOVE, 0)
    assert sink is not None
    clock.advance(0.25)  # the word is written ...
    assert sink("e4: best.")  # the first line of the word on ply 1, queued
    assert "e4: best." in engine._awaiting_audio
    clock.advance(0.5)  # ... and synthesised
    engine._on_audio_starts("e4: best.")  # the voice: playing now
    lat = engine._latency()
    assert lat["move_to_speech_n"] == 1 and lat["move_to_speech_s"] is not None
    assert 0.08 <= lat["move_to_speech_s"] <= 2.0, lat
    assert lat["move_to_speech_s"] == 0.75, "from the move, not from the line queued"
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
