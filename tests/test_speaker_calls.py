"""What the Speaker does when a model call fails, is closed, or is replaced: every paid
call counted, none made twice for one answer or after the session is gone."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence
from typing import Any

import chess
import httpx
import pytest

from sidecoach.coach import coach as coach_module
from sidecoach.coach.coach import Speaker, _Request
from sidecoach.coach.disclosure import PositionFacts
from sidecoach.events import DisclosureLevel, Trigger
from sidecoach.llm.base import ChatMessage, Completion
from sidecoach.llm.openai_compat import OpenAICompatProvider
from tests.fakes import ScriptedProvider


def _request(trigger: Trigger = Trigger.USER_ASK) -> _Request:
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    return _Request(
        trigger=trigger,
        level=DisclosureLevel.LINE,
        system="You are a coach.",
        messages=[{"role": "user", "content": "What now?"}],
        facts=PositionFacts(
            fen=board.fen(),
            side_to_move="white",
            user_side="white",
            phase="opening",
            candidates=["Nf3"],
        ),
        board=board,
        fallback="TEMPLATE",
        ply=2,
        game_id="g",
        candidates=1,
    )


class Failing(ScriptedProvider):
    """Raises `errors` in turn, then answers from the script."""

    def __init__(self, errors: Sequence[Exception], timeout_s: float = 60.0) -> None:
        super().__init__(then="Develop the knight to f3.")
        self.errors = list(errors)
        self.timeout_s = timeout_s

    def chat(self, system: str, messages: Sequence[ChatMessage], **kw: Any) -> Completion:
        self.calls.append({"system": system, "messages": list(messages)})
        if self.errors:
            raise self.errors.pop(0)
        return Completion(text=self.then)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError('401: {"error": {"message": "Incorrect API key provided"}}'),
        httpx.ReadTimeout("timed out"),
        httpx.ConnectError("refused"),
    ],
)
def test_a_failed_call_is_not_a_draft_asked_again(error: Exception) -> None:
    model = Failing([error, error])
    spoken = Speaker(model).speak_now(_request())  # type: ignore[arg-type]
    # One call: the template answers, and no second paid call complains of an empty reply.
    assert len(model.calls) == 1
    assert spoken.message.text == "TEMPLATE"
    assert spoken.problems and "empty reply" not in spoken.problems


def test_a_rate_limit_is_waited_out_once_within_the_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(coach_module, "RATE_BACKOFF_S", 0.05)
    model = Failing([RuntimeError("429: Rate limit reached")])
    began = time.monotonic()
    spoken = Speaker(model).speak_now(_request())  # type: ignore[arg-type]
    assert time.monotonic() - began >= 0.03
    assert len(model.calls) == 2
    # The same request again, not a complaint about the failure.
    assert model.calls[1]["messages"] == model.calls[0]["messages"]
    assert spoken.message.text == "Develop the knight to f3." and not spoken.problems
    # No time left for the wait: the template, after one call.
    monkeypatch.setattr(coach_module, "RATE_BACKOFF_S", 5.0)
    model = Failing([RuntimeError("429: Rate limit reached")], timeout_s=1.0)
    spoken = Speaker(model).speak_now(_request())  # type: ignore[arg-type]
    assert len(model.calls) == 1 and spoken.message.text == "TEMPLATE"


def test_a_call_lost_after_it_was_sent_is_counted() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    provider = OpenAICompatProvider(
        "x", "http://h/v1", "m", transport=httpx.MockTransport(handler), timeout_s=1.0
    )
    with pytest.raises(httpx.ReadTimeout):
        provider.chat("sys", [{"role": "user", "content": "hi " * 100}])
    with pytest.raises(httpx.ReadTimeout):
        provider.chat_stream("sys", [{"role": "user", "content": "hi"}], on_delta=lambda _: True)
    assert provider.usage.calls == 2 and provider.usage.estimated == 2
    assert provider.usage.prompt > 0

    # A refused connection never reached the server: not billed, not counted.
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    provider = OpenAICompatProvider("x", "http://h/v1", "m", transport=httpx.MockTransport(refused))
    with pytest.raises(httpx.ConnectError):
        provider.chat("sys", [{"role": "user", "content": "hi"}])
    assert provider.usage.calls == 0


class Slow(ScriptedProvider):
    """A model that takes a while a call, and says when it has started one."""

    def __init__(self) -> None:
        super().__init__(then="Fine move.")
        self.started = threading.Event()

    def chat(self, system: str, messages: Sequence[ChatMessage], **kw: Any) -> Completion:
        self.started.set()
        time.sleep(0.3)
        return super().chat(system, messages)


def test_closing_drops_what_is_queued() -> None:
    model = Slow()
    speaker = Speaker(model)  # type: ignore[arg-type]
    speaker.start()
    for trigger in (Trigger.POST_MOVE, Trigger.USER_ASK, Trigger.USER_ASK, Trigger.USER_ASK):
        speaker.submit(_request(trigger))
    assert model.started.wait(5)
    speaker.close(timeout=5)
    time.sleep(0.5)
    # The call in flight finishes; nothing queued behind it is asked, or paid for.
    assert len(model.calls) == 1
