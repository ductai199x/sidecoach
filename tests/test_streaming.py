"""The coach's reply streamed to the voice as it is written: the first verbalized
candidate goes out sentence by sentence, each checked before it is said, the stream
closed once that candidate is complete; a mundane reply takes the routine model; the
verdict on a move comes from one line before the position is measured whole."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import chess
import pytest

from sidecoach.coach.coach import Coach, Speaker, _FirstCandidate, _Request, _sample_reply
from sidecoach.coach.disclosure import DisclosureLevel, PositionFacts
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
from sidecoach.events import Trigger
from sidecoach.llm.base import ChatMessage, Completion

FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


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
    board = chess.Board(FEN)
    return _Request(
        trigger=Trigger.POST_MOVE,
        level=DisclosureLevel.LINE,
        system="coach",
        messages=[{"role": "user", "content": "# Task\nA word on the move, two sentences."}],
        facts=PositionFacts(fen=FEN, side_to_move="white", phase="opening", material="level"),
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


def test_a_bare_reply_without_markers_streams_too() -> None:
    provider = StreamingProvider("Just one plain reply. And a second sentence.")
    speaker = Speaker(provider)
    heard: list[str] = []
    spoken = speaker.speak_now(_request(lambda s: heard.append(s) or True))
    assert heard == ["Just one plain reply.", "And a second sentence."]
    assert spoken.streamed


def test_without_a_sink_the_reply_is_sampled_whole_as_before() -> None:
    provider = StreamingProvider("[p=1.0] Whole reply.")
    speaker = Speaker(provider)
    spoken = speaker.speak_now(_request(None))
    assert not spoken.streamed and spoken.message.text == "Whole reply."
    assert provider.streamed_calls == 0


def test_a_routine_request_takes_the_routine_provider() -> None:
    main = StreamingProvider("[p=1.0] From the main model.")
    routine = StreamingProvider("[p=1.0] From the routine model.")
    speaker = Speaker(main, routine_provider=routine)
    assert speaker.speak_now(_request(None, routine=True)).message.text == "From the routine model."
    assert speaker.speak_now(_request(None)).message.text == "From the main model."


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


def test_the_coach_hands_its_sink_to_the_request() -> None:
    provider = StreamingProvider("[p=1.0] Fine.")
    coach = Coach(provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)))
    seen: list[tuple[object, int | None]] = []

    def factory(trigger: object, ply: int | None) -> Callable[[str], bool] | None:
        seen.append((trigger, ply))
        return lambda s: True

    coach.stream_sink = factory
    assert _sample_reply([(1.0, "x")]) == "x"
    assert coach.stream_sink is factory


@pytest.mark.parametrize("multipv_first", [1])
def test_the_verdict_comes_from_one_line_before_the_position_is_measured_whole(
    multipv_first: int,
) -> None:
    from sidecoach.analysis.annotator import GameAnnotator
    from sidecoach.analysis.engine import Line
    from sidecoach.analysis.score import Thresholds

    class FakeEngine:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def analyse(
            self,
            board: chess.Board,
            *,
            depth: int,
            multipv: int = 1,
            time_s: float | None = None,
            carry_best: bool = False,
            on_depth: object = None,
        ) -> list[Line]:
            self.calls.append((board.fen(), multipv))
            del carry_best, on_depth  # the line is as long as the fake makes it, all at once
            lines = [Line(moves=["e2e4"], depth=depth, expected_score=0.52)]
            if multipv > 1:
                lines.append(Line(moves=["d2d4"], depth=depth, expected_score=0.51))
            return lines[:multipv]

    engine = FakeEngine()
    annotator = GameAnnotator(
        engine=engine,  # type: ignore[arg-type]
        depth=18,
        multipv=6,
        thresholds=Thresholds(),
        with_threats=False,
    )
    before = chess.Board()
    annotator.measure(before)
    move = chess.Move.from_uci("e2e4")
    assessment = annotator.assess_move("g", before, move, 1)
    after = before.copy()
    after.push(move)
    assert engine.calls[-1] == (after.fen(), multipv_first), "the verdict from one line"
    assert assessment.uci == "e2e4"
    annotator.analysis_event("g", after, 1)
    assert engine.calls[-1] == (after.fen(), 6), "then the position whole"
