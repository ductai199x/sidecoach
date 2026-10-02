"""Push-to-talk held: the coach is silent and stops at once; its words are still shown (Tai,
2026-10-01: "if push to talk is on then the coach must be muted and stopped talking
immediately"; "captions show lines regardless of voice skips")."""

from __future__ import annotations

from collections.abc import Callable

from sidecoach.events import Trigger
from sidecoach.ui.panel import PanelEngine


class FakeTTS:
    name = "fake"

    def __init__(self) -> None:
        self.said: list[str] = []
        self.stops = 0
        self.busy = False

    def healthcheck(self) -> tuple[bool, str]:
        return True, "fake"

    def say(self, text: str, *, voice: str | None = None) -> None:
        self.said.append(text)
        self.busy = True

    def stop(self) -> None:
        self.stops += 1
        self.busy = False


def _panel(make_panel: Callable[..., PanelEngine]) -> tuple[PanelEngine, FakeTTS]:
    engine = make_panel()
    fake = FakeTTS()
    engine._tts = fake  # type: ignore[assignment]
    engine._listener.mode = "ptt"
    engine._state["ply"] = 12
    return engine, fake


def _captioned(engine: PanelEngine) -> list[str]:
    return [text for _, text, _ in engine.poll()["state"]["caption_lines"]]


def test_a_reply_that_comes_while_the_key_is_held_is_never_spoken(
    make_panel: Callable[..., PanelEngine],
) -> None:
    engine, fake = _panel(make_panel)
    engine.ptt(True)
    assert engine._speak("stuck", "Look at f7.", ply=12) == "muted"
    sink = engine._stream_sink(Trigger.USER_ASK, 12)
    assert sink is not None
    assert sink("The knight is loose.") is True  # taken: the reply stays whole
    engine.ptt(False)
    assert sink("What defends it?") is True  # the key is up, and this reply stays silent
    assert fake.said == [], "nothing was said while the player talked"
    assert _captioned(engine) == ["Look at f7.", "The knight is loose.", "What defends it?"]
    # The key up, the next line is spoken as ever.
    assert engine._speak("stuck", "Look again.", ply=12) == "speak"
    assert fake.said == ["Look again."]


def test_pressing_the_key_cuts_what_is_playing_at_once(
    make_panel: Callable[..., PanelEngine],
) -> None:
    engine, fake = _panel(make_panel)
    assert engine._speak("stuck", "A long hint about the knight on f7.", ply=12) == "speak"
    fake.busy = True
    engine.ptt(True)
    assert fake.stops >= 1, "the voice was stopped on the press"
