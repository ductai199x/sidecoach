"""The captions' side of the panel: the words given to the voice, numbered, for the captions'
window (desktop/src/routes/captions), whether the voice is speaking, and their settings.

The phone's `Captions.show` is called wherever its voice is given words -- spoken, muted, held
back for pace, each sentence of a streamed reply -- and the desktop captions the same words
(Tai, 2026-09-29: "port that feature over to Desktop as well").
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from sidecoach.config import Settings
from sidecoach.events import Trigger
from sidecoach.ui import panel as panel_mod
from sidecoach.ui.panel import CAPTIONS_KEPT, PanelEngine
from tests.fakes import ManualClock

LONG = "Your knight on f5 is loose; the queen check on g6 wins it. What defends it?"


class FakeTTS:
    name = "fake"

    def __init__(self) -> None:
        self.said: list[str] = []
        self.busy = False

    def healthcheck(self) -> tuple[bool, str]:
        return True, "fake"

    def say(self, text: str) -> None:
        self.said.append(text)

    def stop(self) -> None:
        return


def _captioned(engine: PanelEngine) -> list[str]:
    return [text for _, text, _ in engine.poll()["state"]["caption_lines"]]


def test_the_words_given_to_the_voice_are_captioned_whole_said_or_not(
    make_panel: Callable[..., PanelEngine],
) -> None:
    engine = make_panel()
    fake = FakeTTS()
    engine._tts = fake  # type: ignore[assignment]
    # The poll says whether the voice is speaking: the box fades a few seconds after it stops.
    assert engine.poll()["state"]["speaking"] is False
    fake.busy = True
    assert engine.poll()["state"]["speaking"] is True
    fake.busy = False
    engine._state["ply"] = 12
    assert engine._speak("post_mistake", LONG, ply=12) == "speak"
    # Muted: not said, and captioned all the same.
    engine.muted = True
    assert engine._speak("stuck", "Look at f7.", ply=12) == "muted"
    engine.muted = False
    # Not said at all (a short word the busy voice skips, or one the board has left): captioned
    # all the same -- the captions show every line whatever the voice does (Tai, 2026-10-01).
    fake.busy = True
    engine._voicing = ("post_move", 11)
    assert engine._speak("post_move", "exf5: good.", ply=12) == "skip"
    assert engine._speak("post_move", "exf5: fine.", ply=8) == "stale"
    fake.busy = False
    # A reply written sentence by sentence: each one as it comes.
    sink = engine._stream_sink(Trigger.USER_ASK, 12)
    assert sink is not None
    sink("The knight is loose.")
    sink("What defends it?")
    assert _captioned(engine) == [
        LONG,
        "Look at f7.",
        "exf5: good.",
        "exf5: fine.",
        "The knight is loose.",
        "What defends it?",
    ]
    numbers = [n for n, _, _ in engine.poll()["state"]["caption_lines"]]
    assert numbers == [1, 2, 3, 4, 5, 6]
    # Each reply starts a line of its own in the box; a streamed reply's later sentences run
    # on from its first (Tai, 2026-09-29: the box had no new lines, hard to read).
    continues = [c for _, _, c in engine.poll()["state"]["caption_lines"]]
    assert continues == [False, False, False, False, False, True]


def test_only_the_last_captions_are_kept_for_the_window_to_catch_up_on(
    make_panel: Callable[..., PanelEngine],
) -> None:
    engine = make_panel()
    for i in range(CAPTIONS_KEPT + 5):
        engine.caption(f"line {i}")
    engine.caption("   ")  # nothing to show
    kept = engine.poll()["state"]["caption_lines"]
    assert len(kept) == CAPTIONS_KEPT
    assert kept[0] == [6, "line 5", False]
    assert kept[-1] == [CAPTIONS_KEPT + 5, f"line {CAPTIONS_KEPT + 4}", False]


def test_a_new_look_is_stamped_for_a_sample_line_and_keeping_them_up_is_not(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A change to the captions' look shows a sample line the moment it is saved (Tai,
    2026-09-23: "applies instantly on save"): the time of the change, which a window built by
    the same save -- captions turned on with it -- still sees as new. Keeping them on screen
    shows none, and neither does turning them on: the CC button brings up the box itself (Tai,
    2026-10-02). The panel's clock is one the test moves, so the stamp is known to the
    millisecond."""
    engine = make_panel()
    clock = ManualClock(start=1_759_400_000.0)  # a wall-clock time, in seconds
    monkeypatch.setattr(panel_mod, "time", clock.time_module())
    assert engine.poll()["state"]["captions"] == {
        "on": False,
        "keep": False,
        "text": "#FFFFFF",
        "background": "#1F1F1F",
        "opacity": 92,
        "size": 17,
        "preview": 0,
    }
    new = Settings.model_validate(engine.settings.model_dump())
    new.voice.captions_keep = True
    engine.apply_settings(new, ["voice.captions_keep"])
    captions = engine.poll()["state"]["captions"]
    assert captions["keep"] is True and captions["preview"] == 0
    on = Settings.model_validate(new.model_dump())
    on.voice.captions = True
    assert engine.apply_settings(on, ["voice.captions"]) == ["voice.captions"]
    captions = engine.poll()["state"]["captions"]
    assert captions["on"] is True and captions["preview"] == 0, "turned on: no sample line"
    clock.advance(2.5)  # the look changed later
    before = int(clock.time() * 1000)
    newer = Settings.model_validate(new.model_dump())
    newer.voice.captions = True
    newer.voice.captions_text_size = 24
    applied = engine.apply_settings(newer, ["voice.captions", "voice.captions_text_size"])
    captions = engine.poll()["state"]["captions"]
    assert {"voice.captions", "voice.captions_text_size"} <= set(applied)
    assert captions["on"] is True and captions["size"] == 24
    assert captions["preview"] >= before
    assert captions["preview"] == before, "stamped the moment it was saved"
