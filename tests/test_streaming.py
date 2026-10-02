"""The coach's reply streamed to the voice as it is written: the first verbalized
candidate goes out sentence by sentence, each checked before it is said, the stream
closed once that candidate is complete; a mundane reply takes the routine model. (The
verdict on a move from one line before the position is measured whole is
test_analysis.py's test_each_position_is_measured_once.)"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import chess

from sidecoach.coach.coach import (
    IDEA_DIRECTIVE,
    Coach,
    Speaker,
    _FirstCandidate,
    _Request,
    directive_limits,
)
from sidecoach.coach.disclosure import DisclosureLevel, PositionFacts
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
from sidecoach.events import GameStarted, Side, Trigger
from sidecoach.llm.base import ChatMessage, Completion
from tests.coaching import SCHOLAR, mate_analysis


class StreamingProvider:
    """Delivers its reply through `on_delta` in small pieces, and notes whether the
    stream was stopped early."""

    name = "streaming"

    def __init__(self, reply: str, *, pieces: int = 7) -> None:
        self.reply = reply
        self.pieces = pieces
        self.stopped_early = False
        self.calls = 0
        self.streamed_calls = 0

    def healthcheck(self) -> tuple[bool, str]:
        return True, "streaming"

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> Completion:
        self.calls += 1
        return Completion(text=self.reply)

    def chat_stream(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
        on_delta: Callable[[str], bool],
    ) -> Completion:
        self.calls += 1
        self.streamed_calls += 1
        step = max(1, len(self.reply) // self.pieces)
        sent = []
        for i in range(0, len(self.reply), step):
            piece = self.reply[i : i + step]
            sent.append(piece)
            if not on_delta(piece):
                self.stopped_early = True
                break
        return Completion(text="".join(sent), stats={"streamed": True})


def _request(sink: Callable[[str], bool] | None, *, routine: bool = False) -> _Request:
    board = chess.Board(SCHOLAR)
    return _Request(
        trigger=Trigger.POST_MOVE,
        level=DisclosureLevel.LINE,
        system="coach",
        messages=[{"role": "user", "content": "# Task\nA word on the move, two sentences."}],
        facts=PositionFacts(fen=SCHOLAR, side_to_move="white", phase="opening", material="level"),
        board=board,
        fallback="Fine.",
        ply=6,
        game_id="g",
        stream_to=sink,
        routine=routine,
    )


def test_the_first_candidate_streams_sentence_by_sentence_and_closes_the_stream() -> None:
    reply = (
        "[p=0.4] Nf3 was best. It develops with a threat.\n"
        "[p=0.6] Good move, the knight belongs there. Keep going."
    )
    provider = StreamingProvider(reply)
    speaker = Speaker(provider)
    heard: list[str] = []
    spoken = speaker.speak_now(_request(lambda s: heard.append(s) or True))
    assert heard == ["Nf3 was best.", "It develops with a threat."], heard
    assert spoken.streamed and spoken.message.text == "Nf3 was best. It develops with a threat."
    assert provider.stopped_early, "the other candidates were never written"
    assert not spoken.problems and speaker.fallbacks == 0
    # A bare reply, without markers, is one candidate and streams too.
    heard.clear()
    speaker = Speaker(StreamingProvider("Just one plain reply. And a second sentence."))
    assert speaker.speak_now(_request(lambda s: heard.append(s) or True)).streamed
    assert heard == ["Just one plain reply.", "And a second sentence."]


def test_a_sentence_that_fails_the_check_stops_the_stream_with_what_was_said() -> None:
    def checker(
        text: str,
        board: chess.Board,
        after: Sequence[chess.Board] = (),
        us: chess.Color | None = None,
    ) -> list[str]:
        return ["the rook on f7 is not there"] if "rook on f7" in text else []

    reply = "[p=0.5] A fine move. Your rook on f7 is strong.\n[p=0.5] Clean reply here."
    provider = StreamingProvider(reply)
    speaker = Speaker(provider, draft_check=checker)
    heard: list[str] = []
    spoken = speaker.speak_now(_request(lambda s: heard.append(s) or True))
    assert heard == ["A fine move."]
    assert spoken.streamed and spoken.message.text == "A fine move."
    assert spoken.problems == ["the rook on f7 is not there"]
    assert speaker.fallbacks == 0, "no template follows a voice that has spoken"


def test_a_first_sentence_that_fails_falls_back_to_the_whole_reply_path() -> None:
    def checker(
        text: str,
        board: chess.Board,
        after: Sequence[chess.Board] = (),
        us: chess.Color | None = None,
    ) -> list[str]:
        return ["leak"] if "secret" in text else []

    reply = "[p=0.5] The secret is Qxf7.\n[p=0.5] Look at the f7 square."
    provider = StreamingProvider(reply)
    speaker = Speaker(provider, draft_check=checker)
    heard: list[str] = []
    spoken = speaker.speak_now(_request(lambda s: heard.append(s) or True))
    assert heard == []
    assert not spoken.streamed
    assert spoken.message.text == "Look at the f7 square."
    assert provider.streamed_calls == 1 and provider.calls == 2


def test_the_sink_can_stop_the_stream_when_the_board_moves_on() -> None:
    reply = "[p=1.0] One. Two. Three."
    provider = StreamingProvider(reply, pieces=12)
    speaker = Speaker(provider)
    heard: list[str] = []

    def sink(sentence: str) -> bool:
        heard.append(sentence)
        return len(heard) < 2  # the board moved on after the second sentence

    spoken = speaker.speak_now(_request(sink))
    assert heard == ["One.", "Two."], "the second was offered and declined; the third never"
    assert spoken.streamed and spoken.message.text == "One."


def test_without_a_sink_the_reply_is_sampled_whole_from_the_model_the_request_takes() -> None:
    main = StreamingProvider("[p=1.0] From the main model.")
    routine = StreamingProvider("[p=1.0] From the routine model.")
    speaker = Speaker(main, routine_provider=routine)
    spoken = speaker.speak_now(_request(None))
    assert not spoken.streamed and spoken.message.text == "From the main model."
    assert main.streamed_calls == 0
    assert speaker.speak_now(_request(None, routine=True)).message.text == "From the routine model."


def test_the_reader_handles_deltas_that_split_markers() -> None:
    heard: list[str] = []
    reader = _FirstCandidate(lambda s: heard.append(s) or True, lambda t: [])
    for delta in ["[p", "=0.", "3] Fir", "st one. Sec", "ond one.", " [p=0.7] Other."]:
        if not reader.feed(delta):
            break
    reader.finish()
    assert heard == ["First one.", "Second one."]
    assert reader.done


def test_a_stop_at_the_buffer_end_and_a_bare_move_number_are_not_sentences() -> None:
    heard: list[str] = []
    reader = _FirstCandidate(lambda s: heard.append(s) or True, lambda t: [])
    for delta in [
        "[p=0.3] 6.",
        " d3 was good, costing 0.",
        "03 of expected score.",
        " Bxc6+ was better.",
        " [p=0.7] Other.",
    ]:
        if not reader.feed(delta):
            break
    reader.finish()
    assert heard == ["6. d3 was good, costing 0.03 of expected score.", "Bxc6+ was better."], heard
    heard.clear()
    reader = _FirstCandidate(lambda s: heard.append(s) or True, lambda t: [])
    reader.feed("6. d3 was good, costing 0.03; 6. Bxc6+ was better. And so on.")
    reader.finish()
    assert heard == ["6. d3 was good, costing 0.03; 6. Bxc6+ was better.", "And so on."], heard


def test_the_stream_stops_at_the_tasks_length_and_its_slack() -> None:
    """The runtime audit: with the voice on, an eight-sentence homily went out whole. The
    length is the directive's own: the first hint's two or three sentences, and one more for
    the question it offers on top ("then, if you like, one question"), which it does not
    require (the prompt audit, 2026-09-28)."""
    assert directive_limits(IDEA_DIRECTIVE) == (4, None)
    heard: list[str] = []
    reader = _FirstCandidate(lambda t: heard.append(t) is None, lambda t: [], limit=1)
    reader.feed("One. Two. Three. Four. Five. ")
    reader.finish()
    assert sum(t.count(".") for t in heard) == 3, "the limit and its slack, no more"


def test_the_coach_hands_its_sink_to_the_request() -> None:
    """The coach asks its sink factory for each request it makes (`stream_sink`, with the
    trigger and the ply) and hands the sink it gets to the request, which the Speaker then
    streams to."""
    provider = StreamingProvider("[p=1.0] Their king has one square left.")
    coach = Coach(provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)))
    seen: list[tuple[object, int | None]] = []

    def sink(sentence: str) -> bool:
        return True

    def factory(trigger: Trigger, ply: int | None) -> Callable[[str], bool] | None:
        seen.append((trigger, ply))
        return sink

    coach.stream_sink = factory
    coach.on_event(GameStarted(game_id="g", user_side=Side.WHITE, start_fen=SCHOLAR), 0.0)
    coach.on_event(mate_analysis(0, decisive=False), 1.0)
    coach.tick(12.0)  # stuck: the idea, unprompted
    assert seen == [(Trigger.STUCK, 0)]
    queued = coach.speaker._requests.get_nowait()
    assert isinstance(queued, _Request) and queued.stream_to is sink
    coach.speaker._requests.put(queued)
    coach.speaker.run_pending()
    assert provider.streamed_calls == 1, "the reply was streamed to the sink"


class StreamingToolModel:
    """Calls the engine in its first round, writing a preamble in the same turn (the chat
    wire's content beside tool_calls), then answers; every round streamed in pieces of 7."""

    name = "stream-tools"
    tools_ok = True

    def __init__(self, preamble: str, answer: str) -> None:
        self.preamble = preamble
        self.answer = answer
        self.round = 0

    def healthcheck(self) -> tuple[bool, str]:
        return True, "x"

    def chat(self, system: str, messages: Sequence[ChatMessage], **kw: object) -> Completion:
        return self._go(None, kw)

    def chat_stream(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        on_delta: Callable[[str], bool],
        **kw: object,
    ) -> Completion:
        return self._go(on_delta, kw)

    def _go(self, on_delta: Callable[[str], bool] | None, kw: dict[str, object]) -> Completion:
        from sidecoach.llm.base import ToolCall

        self.round += 1
        calls: list[ToolCall] = []
        if self.round == 1 and kw.get("tools"):
            text = self.preamble
            calls = [ToolCall(id="c1", name="analyse", arguments={"moves": ["Qxf7#"]})]
        else:
            text = self.answer
        sent: list[str] = []
        if on_delta is not None:
            for i in range(0, len(text), 7):
                sent.append(text[i : i + 7])
                if not on_delta(text[i : i + 7]):
                    break
            text = "".join(sent)
        return Completion(text=text, tool_calls=calls)


def test_a_tool_rounds_preamble_is_not_said_and_the_answer_is_read_from_its_start() -> None:
    """The model wrote "Let me check that with the engine." in the round that called it, and
    the reader sliced the answer round with the preamble's spoken length: the voice said
    "ith the queen is mate" (the deep prompt review, 2026-10-01). The preamble is not said;
    the answer is said from its first word."""
    from sidecoach.coach.coach import Coach as _Coach
    from tests.coaching import _policy, drain, play_to_scholar
    from tests.test_engine_tools import FirstLegal

    answer = (
        "[p=0.6] Taking on f7 with the queen is mate: the king has no square left. "
        "[p=0.4] The queen takes on f7 and it is mate at once."
    )
    for preamble in (
        "Good question. Let me check that with the engine first. ",
        "Let me ask the engine about Qxf7. ",
    ):
        coach = _Coach(StreamingToolModel(preamble, answer), policy=_policy(), engine=FirstLegal())
        coach.sync_claims = True
        heard: list[str] = []

        def sink(sentence: str, heard: list[str] = heard) -> bool:
            heard.append(sentence)
            return True

        coach.stream_sink = lambda trigger, ply, sink=sink: sink
        play_to_scholar(coach)
        coach.on_event(mate_analysis(6, decisive=False), 7.0)
        drain(coach)
        heard.clear()
        coach.ask("what happens if I take on f7 with the queen?", 8.0)
        out = [m.text for m in drain(coach) if getattr(m, "text", None)]
        assert heard and heard[0].startswith("Taking on f7"), heard
        assert not any("engine" in s for s in heard), heard
        assert out and out[-1].startswith("Taking on f7"), out
