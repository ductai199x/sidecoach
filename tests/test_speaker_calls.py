"""What the Speaker does when a model call fails, is closed, or is replaced: every paid
call counted, none made twice for one answer or after the session is gone."""

from __future__ import annotations

import threading
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


def test_a_rate_limit_is_waited_out_once_within_the_timeout() -> None:
    # Time is the test's: a clock it moves and a wait it records, no real delay.
    now = [100.0]
    waits: list[float] = []

    def pause(seconds: float) -> bool:
        waits.append(seconds)
        now[0] += seconds
        return False  # not closed meanwhile

    model = Failing([RuntimeError("429: Rate limit reached")])
    speaker = Speaker(model)  # type: ignore[arg-type]
    speaker.clock, speaker.pause = lambda: now[0], pause
    spoken = speaker.speak_now(_request())
    assert waits == [coach_module.RATE_BACKOFF_S] and len(model.calls) == 2
    # The same request again, not a complaint about the failure.
    assert model.calls[1]["messages"] == model.calls[0]["messages"]
    assert spoken.message.text == "Develop the knight to f3." and not spoken.problems
    # No room left in the provider's timeout for the wait: the template, after one call.
    waits.clear()
    model = Failing(
        [RuntimeError("429: Rate limit reached")], timeout_s=coach_module.RATE_BACKOFF_S
    )
    speaker = Speaker(model)  # type: ignore[arg-type]
    speaker.clock, speaker.pause = lambda: now[0], pause
    spoken = speaker.speak_now(_request())
    assert waits == [] and len(model.calls) == 1 and spoken.message.text == "TEMPLATE"
    # Closed during the wait: the template, and no second call.
    model = Failing([RuntimeError("429: Rate limit reached")])
    speaker = Speaker(model)  # type: ignore[arg-type]
    speaker.clock, speaker.pause = lambda: now[0], lambda s: True
    spoken = speaker.speak_now(_request())
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


class Held(ScriptedProvider):
    """A model whose call waits for the test to let it finish."""

    def __init__(self) -> None:
        super().__init__(then="Fine move.")
        self.started = threading.Event()
        self.release = threading.Event()

    def chat(self, system: str, messages: Sequence[ChatMessage], **kw: Any) -> Completion:
        self.started.set()
        self.release.wait()
        return super().chat(system, messages)


def test_closing_drops_what_is_queued() -> None:
    model = Held()
    speaker = Speaker(model)  # type: ignore[arg-type]
    speaker.start()
    for trigger in (Trigger.POST_MOVE, Trigger.USER_ASK, Trigger.USER_ASK, Trigger.USER_ASK):
        speaker.submit(_request(trigger))
    model.started.wait()
    closer = threading.Thread(target=speaker.close, kwargs={"timeout": 60})
    closer.start()
    try:
        speaker._closed.wait()  # close has emptied the queue, waiting on the call in flight
    finally:
        model.release.set()
    closer.join()
    # The call in flight finished; nothing queued behind it was asked, or paid for.
    assert len(model.calls) == 1 and speaker.pending == 0


def _openai(name: str) -> OpenAICompatProvider:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "Fine."}}]})

    return OpenAICompatProvider(name, "http://h/v1", "m", transport=httpx.MockTransport(handler))


def test_replaced_providers_are_counted_and_closed() -> None:
    first, routine, verifier, second = (_openai(n) for n in ("a", "b", "c", "d"))
    speaker = Speaker(first, routine_provider=routine, verifier=verifier)
    first.chat("s", [{"role": "user", "content": "hi"}])
    verifier.chat("s", [{"role": "user", "content": "hi"}])
    speaker.replace_providers(second)
    second.chat("s", [{"role": "user", "content": "hi"}])
    # Every ledger, the replaced model's and the verifier's on its own endpoint, still counts.
    assert sum(u.calls for u in speaker.usages()) == 3
    speaker.close()
    for p in (first, routine, verifier, second):
        assert p._client.is_closed, p.name


def test_each_ledger_is_priced_as_its_own_model(tmp_path: Any) -> None:
    from sidecoach.config import Settings
    from sidecoach.llm.usage import KNOWN_PRICES, Usage, total

    luna = Usage()
    luna.model = "gpt-5.6-luna"
    luna.record({"prompt_tokens": 1_000_000, "completion_tokens": 0})
    # The coach on a local model of no known price; the verifier on luna is still priced.
    got = total([luna, luna], Settings(data_dir=tmp_path), "local-model")
    assert got["calls"] == 1
    assert got["cost"] == pytest.approx(KNOWN_PRICES["gpt-5.6-luna"].prompt)


def test_a_reopened_puzzle_keeps_the_closed_rooms_spend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from sidecoach.config import Settings
    from sidecoach.llm.usage import Usage
    from sidecoach.ui import puzzle_room

    made: list[Any] = []

    class Room:
        def __init__(self, *a: Any, **k: Any) -> None:
            self.closed = self.closing = False
            self.ledger = Usage()
            self.ledger.record({"prompt_tokens": 10, "completion_tokens": 1})
            made.append(self)

        def usages(self) -> list[Any]:
            return [self.ledger]

    class Puzzles:
        def get(self, puzzle_id: str) -> object:
            return object()

    monkeypatch.setattr(puzzle_room, "PuzzleRoom", Room)
    rooms = puzzle_room.PuzzleRooms(
        Settings(data_dir=tmp_path),
        store_path=None,
        puzzles=Puzzles(),  # type: ignore[arg-type]
    )
    rooms.open("p1")
    made[0].closed = True
    rooms.open("p1")
    assert len(made) == 2
    assert sum(u.calls for u in rooms.usages()) == 2


def test_opening_a_puzzle_does_not_hold_up_the_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    from sidecoach.config import Settings
    from sidecoach.ui import puzzle_room

    made: list[Any] = []
    making = threading.Event()
    release = threading.Event()

    class HeldRoom:
        # A room whose making asks a model that has not answered yet.
        def __init__(self, *a: Any, **k: Any) -> None:
            making.set()
            release.wait()
            self.closed = self.closing = False
            made.append(self)

        def usages(self) -> list[Any]:
            return []

    class Puzzles:
        def get(self, puzzle_id: str) -> object:
            return object()

    monkeypatch.setattr(puzzle_room, "PuzzleRoom", HeldRoom)
    rooms = puzzle_room.PuzzleRooms(
        Settings(data_dir=tmp_path),
        store_path=None,
        puzzles=Puzzles(),  # type: ignore[arg-type]
    )
    got: list[Any] = []
    first = threading.Thread(target=lambda: got.append(rooms.open("p1")))
    first.start()
    making.wait()  # the room is being made, the model not yet answered
    read = threading.Event()

    def session_tick() -> None:
        rooms.usages()  # what the session loop reads every tick
        read.set()

    threading.Thread(target=session_tick, daemon=True).start()
    # Read while the room is still being made: the lock is not held through it. (Held, this
    # is never set and the test fails at the bound, not by a measured duration.)
    ticked = read.wait(30)
    release.set()  # (a blocked tick must not leave the room held)
    assert ticked, "the session's tick is blocked behind the room being made"
    second = threading.Thread(target=lambda: got.append(rooms.open("p1")))
    second.start()  # whether it arrives before or after the room is made, it is the same room
    release.set()
    first.join()
    second.join()
    assert len(made) == 1 and got[0] is got[1] is made[0]
