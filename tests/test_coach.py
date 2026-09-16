"""The coach's dialogue: what it says, when, from which facts, and what it may never say."""

from __future__ import annotations

from collections.abc import Sequence

import chess

from sidecoach.coach.coach import Coach, fact_sheet, unlock_for
from sidecoach.coach.disclosure import PositionFacts, Unlock
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
from sidecoach.events import (
    Analysis,
    DisclosureLevel,
    GameStarted,
    Label,
    Line,
    MotifTag,
    MoveAssessment,
    MovePlayed,
    Severity,
    Side,
    Trigger,
)
from sidecoach.llm.base import ChatMessage, Completion

# 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??: white to move, Qxf7 mates.
SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


class ScriptedProvider:
    """Answers from a script, and remembers every prompt it was given."""

    name = "scripted"

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[str, list[ChatMessage]]] = []

    def healthcheck(self) -> tuple[bool, str]:
        return True, "scripted"

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> Completion:
        self.prompts.append((system, list(messages)))
        text = self.replies.pop(0) if self.replies else "Keep looking."
        return Completion(text=text)


def _coach(replies: Sequence[str]) -> tuple[Coach, ScriptedProvider]:
    provider = ScriptedProvider(replies)
    coach = Coach(provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)))
    return coach, provider


def _mate_analysis(ply: int, decisive: bool = True) -> Analysis:
    return Analysis(
        game_id="g",
        fen=SCHOLAR,
        ply=ply,
        lines=[
            Line(moves=["h5f7"], depth=12, expected_score=1.0, mate=1),
            Line(moves=["c4f7"], depth=12, expected_score=0.55),
        ],
        decisive=decisive,
        only_move=True,
        tags=[MotifTag(name="mateIn1", squares=["f7", "e8"], move="h5f7")],
    )


def _start(coach: Coach, user_side: Side = Side.WHITE) -> None:
    coach.on_event(
        GameStarted(game_id="g", user_side=user_side, start_fen=chess.Board().fen()), 0.0
    )


def _play_to_scholar(coach: Coach) -> None:
    """White, the user, to move with Qxf7# on the board, after black's Nf6."""
    _start(coach)
    board = chess.Board()
    for ply, san in enumerate(("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"), start=1):
        move = board.parse_san(san)
        before = board.fen()
        board.push(move)
        coach.on_event(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=before,
                fen_after=board.fen(),
                by_user=(ply % 2 == 1),
            ),
            float(ply),
        )
    assert coach.user_to_move and coach.board.fen() == SCHOLAR


def _messages(coach: Coach):
    coach.speaker.run_pending()
    return coach.poll()


def test_a_decisive_moment_earns_one_question_that_names_nothing() -> None:
    coach, provider = _coach(["What is your opponent's most dangerous idea right now?"])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    out = _messages(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.DECISIVE_QUESTION
    assert out[0].level is DisclosureLevel.QUESTION
    # The model was handed the position and nothing about the answer.
    _, messages = provider.prompts[0]
    sheet = messages[-1]["content"]
    assert "White to move; you are white" in sheet and "decisive: True" in sheet
    assert "a b c d e f g h" in sheet, "the position is drawn as a board"
    for secret in ("Qxf7", "mating", "queen", "h5f7", "theme", "candidates"):
        assert secret not in sheet, secret
    # And a second question for the same position is not asked.
    coach.on_event(_mate_analysis(6), 8.0)
    assert _messages(coach) == []


def test_a_leaked_draft_is_regenerated_then_replaced() -> None:
    coach, provider = _coach(["Play Qxf7, it is mate!", "Your queen is ready to strike on f7."])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    out = _messages(coach)
    assert len(out) == 1
    assert "Qxf7" not in out[0].text and "f7" not in out[0].text and "queen" not in out[0].text
    spoken = coach.spoken[-1]
    assert spoken.attempts == 2 and spoken.problems, "both drafts leaked; the template spoke"
    assert coach.speaker.fallbacks == 1
    # The second attempt was told what was wrong.
    _, retry = provider.prompts[1]
    assert "not allowed" in retry[-1]["content"] and "Qxf7" in retry[-1]["content"]


def test_a_hint_unlocks_the_theme_and_asking_for_the_move_unlocks_it() -> None:
    coach, provider = _coach(
        [
            "What is your opponent's king missing right now?",
            "Think about a mating attack: where is their king weakest?",
            "Qxf7# ends it.",
        ]
    )
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    assert _messages(coach) == [], "nothing unprompted in a quiet position"
    # The first hint in a fresh position is the orienting question; the next one, the theme.
    coach.ask("give me a hint", 8.0)
    out = _messages(coach)
    assert out[0].level is DisclosureLevel.QUESTION
    assert "theme" not in provider.prompts[0][1][-1]["content"]
    coach.ask("another hint", 8.5)
    out = _messages(coach)
    assert out[0].level is DisclosureLevel.THEME
    sheet = provider.prompts[1][1][-1]["content"]
    assert "theme: a mating attack" in sheet and "Qxf7" not in sheet and "queen" not in sheet
    coach.ask("ok just tell me the move", 9.0)
    out = _messages(coach)
    assert out[0].level is DisclosureLevel.CANDIDATE and "Qxf7" in out[0].text
    sheet = provider.prompts[2][1][-1]["content"]
    assert "candidates: Qxf7#" in sheet


def test_naming_the_move_yourself_earns_the_candidate_level() -> None:
    coach, _ = _coach(["What do you think?", "Yes, Qxf7 is mate."])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    _messages(coach)
    coach.ask("Is Qxf7 mate?", 8.0)
    out = _messages(coach)
    assert out[0].level is DisclosureLevel.CANDIDATE
    board = chess.Board(SCHOLAR)
    assert unlock_for("Is Qxf7 mate?", board, ["Qxf7#"]) is Unlock.USER_NAMED_MOVE
    assert unlock_for("What about Bxf7?", board, ["Qxf7#"]) is None, "not the candidate"
    assert unlock_for("show me the line", board, []) is Unlock.ASK_LINE
    assert unlock_for("HINT please", board, []) is Unlock.HINT
    assert unlock_for("what should I play here", board, []) is Unlock.ASK_MOVE
    assert unlock_for("how are you", board, []) is None


def test_feedback_after_the_users_move_may_say_everything_about_that_position() -> None:
    coach, provider = _coach(["...", "Bxf7+ was fine, but Qxf7 was mate on the spot."])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    _messages(coach)
    # The user plays Bxf7+ instead of the mate.
    board = chess.Board(SCHOLAR)
    move = board.parse_san("Bxf7+")
    board.push(move)
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=7,
            san="Bxf7+",
            uci=move.uci(),
            fen_before=SCHOLAR,
            fen_after=board.fen(),
            by_user=True,
        ),
        10.0,
    )
    coach.on_event(
        MoveAssessment(
            game_id="g",
            ply=7,
            uci=move.uci(),
            expected_loss=0.45,
            severity=Severity.BLUNDER,
            best=["h5f7"],
            label=Label.MISS,
        ),
        11.0,
    )
    out = _messages(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.POST_MISTAKE
    assert out[0].level is DisclosureLevel.LINE and "Qxf7" in out[0].text
    sheet = provider.prompts[-1][1][-1]["content"]
    assert "candidates: Qxf7#" in sheet and "Bxf7+" in sheet and "miss" in sheet


def test_stuck_gives_a_hint_once_and_the_opponents_turn_is_silent() -> None:
    coach, _ = _coach(["Take stock of what changed."])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    coach.tick(12.0)
    assert _messages(coach) == [], "not stuck yet: ten seconds from the move at 6.0"
    coach.tick(17.0)
    out = _messages(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.STUCK
    coach.tick(40.0)
    assert _messages(coach) == [], "one unprompted message per position"
    # With the opponent to move, nothing unprompted.
    quiet, _ = _coach(["..."])
    _start(quiet, Side.BLACK)
    quiet.on_event(_mate_analysis(0), 1.0)
    quiet.tick(100.0)
    assert _messages(quiet) == []


def test_a_good_move_earns_two_words_without_the_model() -> None:
    coach, provider = _coach(["never used"])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    board = chess.Board(SCHOLAR)
    move = board.parse_san("Qxf7#")
    board.push(move)
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=7,
            san="Qxf7#",
            uci=move.uci(),
            fen_before=SCHOLAR,
            fen_after=board.fen(),
            by_user=True,
        ),
        10.0,
    )
    coach.on_event(
        MoveAssessment(
            game_id="g",
            ply=7,
            uci=move.uci(),
            expected_loss=0.0,
            severity=Severity.NONE,
            best=["h5f7"],
            label=Label.BEST,
        ),
        11.0,
    )
    out = _messages(coach)
    from sidecoach.coach.coach import VERDICTS

    assert len(out) == 1 and out[0].text in [p.format(san="Qxf7#") for p in VERDICTS["best"]]
    assert provider.prompts == [], "no model call for a good move"
    # The wording is by the label and never the same twice running: "good, good, good"
    # for a whole game wore thin.
    seen = {coach._verdict("Nf3", "good") for _ in range(40)}
    assert len(seen) >= 3
    previous = None
    for _ in range(40):
        current = coach._verdict("Nf3", "good")
        assert current != previous
        previous = current
    assert coach._verdict("e4", "book") in [p.format(san="e4") for p in VERDICTS["book"]]


def test_the_draft_checker_can_veto_a_false_claim() -> None:
    def checker(text: str, board: chess.Board, after: Sequence[chess.Board] = ()) -> list[str]:
        return ["the rook on f7 is not there"] if "rook on f7" in text else []

    provider = ScriptedProvider(["Your rook on f7 is strong.", "Your king is safe for now."])
    coach = Coach(
        provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)), draft_check=checker
    )
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    out = _messages(coach)
    assert out[0].text == "Your king is safe for now."
    assert coach.spoken[-1].attempts == 2 and coach.speaker.fallbacks == 0


def test_verbalized_replies_parse_into_weighted_candidates() -> None:
    from sidecoach.coach.coach import _parse_candidates, _sample_reply

    raw = "[p=0.5] First reply here.\n[p=0.3] Second reply,\nwrapping on.\n[p=0.2] Third."
    cands = _parse_candidates(raw)
    assert [round(p, 2) for p, _ in cands] == [0.5, 0.3, 0.2]
    assert cands[1][1] == "Second reply,\nwrapping on.", "a wrapped reply is kept whole"
    # A reply with no markers is treated as a single candidate, so a plain model still works.
    assert _parse_candidates("Just one plain reply.") == [(1.0, "Just one plain reply.")]
    # A think block is stripped before parsing.
    assert _parse_candidates("<think>hidden</think>\n[p=1] Clean.") == [(1.0, "Clean.")]
    # Sampling from a single passed candidate is deterministic.
    assert _sample_reply([(0.9, "only one")]) == "only one"


def test_verbalized_sampling_keeps_the_checked_candidate() -> None:
    # The model offers a leaky reply and a clean one in one distribution; VS keeps the clean
    # one on the first attempt, no retry, no template.
    def checker(text: str, board: chess.Board, after: Sequence[chess.Board] = ()) -> list[str]:
        return ["the rook on f7 is not there"] if "rook on f7" in text else []

    block = "[p=0.6] Your rook on f7 is strong.\n[p=0.4] Your king is safe for now."
    provider = ScriptedProvider([block])
    coach = Coach(
        provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)), draft_check=checker
    )
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    out = _messages(coach)
    assert out[0].text == "Your king is safe for now."
    assert coach.spoken[-1].attempts == 1 and coach.speaker.fallbacks == 0


def test_the_fact_sheet_hides_by_level() -> None:
    facts = PositionFacts(
        fen=SCHOLAR,
        side_to_move="white",
        phase="opening",
        material="material is level",
        theme="a mating attack",
        region="the kingside, around f7",
        piece="your queen",
        candidates=["Qxf7#"],
        line=["Qxf7#"],
        decisive=True,
    )
    assert "theme" not in fact_sheet(facts, DisclosureLevel.QUESTION)
    assert "region: the kingside, around f7" in fact_sheet(facts, DisclosureLevel.REGION)
    assert "candidates" not in fact_sheet(facts, DisclosureLevel.REGION)
    assert "line: Qxf7#" in fact_sheet(facts, DisclosureLevel.LINE)


def test_the_game_memory_carries_the_arc_into_the_prompt() -> None:
    from sidecoach.coach.coach import GameMemory

    memory = GameMemory()
    memory.note_move(6, "opening", "material is level", "C50 Italian Game")
    memory.note_move(24, "middlegame", "you are down a pawn", None)
    block = memory.block()
    assert "C50 Italian Game" in block and "move 12, middlegame" in block
    assert "down a pawn" in block and "level" not in block
    # A recurring missed idea is surfaced only once it has happened more than once.
    memory.note_mistake(20, "a fork")
    assert "missed more than once" not in memory.block()
    memory.note_mistake(28, "a fork")
    assert "a fork (2x)" in memory.block()
    # A strategic lapse is recorded too, from the plan when there is no tactic to miss.
    memory.note_mistake(30, None, "finishing your development")
    memory.note_mistake(34, None, "finishing your development")
    assert "finishing your development (2x)" in memory.block()


def test_the_memory_reaches_the_model_and_grows_with_the_game() -> None:
    coach, provider = _coach(["Careful here.", "That let it slip."])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    _messages(coach)
    assert "Game so far:" in provider.prompts[0][1][-1]["content"]
    # A mistake in a themed position is remembered.
    board = chess.Board(SCHOLAR)
    move = board.parse_san("Bxf7+")
    board.push(move)
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=7,
            san="Bxf7+",
            uci=move.uci(),
            fen_before=SCHOLAR,
            fen_after=board.fen(),
            by_user=True,
        ),
        10.0,
    )
    coach.on_event(
        MoveAssessment(
            game_id="g",
            ply=7,
            uci=move.uci(),
            expected_loss=0.45,
            severity=Severity.BLUNDER,
            best=["h5f7"],
            label=Label.MISS,
        ),
        11.0,
    )
    _messages(coach)
    assert coach.memory.missed.get("a mating attack") == 1


def test_asking_before_anything_is_analysed_gets_an_honest_answer() -> None:
    coach, _ = _coach([])
    _start(coach)
    coach.ask("what should I play", 1.0)
    out = _messages(coach)
    assert len(out) == 1 and "not seen enough" in out[0].text


def test_clock_scales_the_policy_and_a_new_game_resets_it() -> None:
    """A ClockObservation feeds the interjection cadence: it bands the time control and
    sets the user's remaining time; a new game clears it back to untimed until a clock is
    seen again; and never seeing a clock leaves the policy on its untimed defaults."""
    from sidecoach.coach.policy import TimeControl
    from sidecoach.events import ClockObservation

    coach, _ = _coach([])
    coach.on_event(
        GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.STARTING_FEN), 0.0
    )
    # Before any clock, the policy keeps its untimed default (every trigger fires).
    assert coach.policy.time_control is TimeControl.UNTIMED
    # A fresh 10+0 game: the peak clock bands it as rapid.
    coach.on_event(ClockObservation(white_ms=600_000, black_ms=600_000, running=Side.WHITE), 1.0)
    assert coach.policy.time_control is TimeControl.RAPID
    # The user is White, so their remaining is White's; a low clock shrinks the stuck hint
    # (down to the floor, which the test coach's ten-second setting already sits under).
    coach.policy.settings = PolicySettings(stuck_seconds=45)
    coach.on_event(ClockObservation(white_ms=50_000, black_ms=300_000, running=Side.WHITE), 2.0)
    assert coach.policy._remaining_ms == 50_000
    assert coach.policy.stuck_threshold() < coach.policy.settings.stuck_seconds
    # A new game clears the clock back to untimed until one is seen again.
    coach.on_event(
        GameStarted(game_id="g2", user_side=Side.BLACK, start_fen=chess.STARTING_FEN), 3.0
    )
    assert coach.policy.time_control is TimeControl.UNTIMED
    assert coach.policy._remaining_ms is None
    # Now the user plays Black: their remaining is the black clock, and ~3 min is blitz.
    coach.on_event(ClockObservation(white_ms=185_000, black_ms=180_000, running=Side.BLACK), 4.0)
    assert coach.policy._remaining_ms == 180_000
    assert coach.policy.time_control is TimeControl.BLITZ


def test_a_canned_ask_is_answered_at_once_and_the_model_builds_on_it() -> None:
    """Hint, Best move and Line: the template for the level goes out immediately, with its
    marks, and the model's fuller reply follows; the buttons felt slow when the answer
    waited on two model calls."""
    coach, provider = _coach(["Because it keeps the knight out of play, and eyes e5."])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    coach.ask_move(8.0)
    quick = coach.poll()  # before the model has answered anything
    assert [m.text for m in quick] == ["The move is Qxf7#."]
    assert quick[0].trigger is Trigger.USER_ASK and quick[0].level is DisclosureLevel.CANDIDATE
    assert any(mark.kind == "arrow" for mark in quick[0].annotations)
    out = _messages(coach)
    assert [m.text for m in out] == ["Because it keeps the knight out of play, and eyes e5."]
    prompt = provider.prompts[0][1][-1]["content"]
    assert "have just been shown that it is Qxf7#" in prompt
    assert "TWO different ways" in prompt, "an answer the player waits for asks for fewer"


def test_claims_are_checked_off_the_callers_thread_and_only_when_there_are_any() -> None:
    import threading
    import time

    from sidecoach.coach.coach import may_carry_claims

    assert not may_carry_claims("hint")
    assert not may_carry_claims("what should I play")
    assert not may_carry_claims("show me the line")
    assert may_carry_claims("is Nf3 good here")
    assert may_carry_claims("my knight is hanging")
    assert may_carry_claims("can I castle")

    calls: list[str] = []
    started = threading.Event()

    def slow_claims(text: str, board: chess.Board, facts: object) -> list[str]:
        calls.append(threading.current_thread().name)
        started.set()
        time.sleep(0.2)
        return ["'Nf3 is good': not the strongest, Qxf7# mates"]

    provider = ScriptedProvider(["Not quite. What is your most active piece already eyeing?"])
    coach = Coach(
        provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)), claims=slow_claims
    )
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    begun = time.perf_counter()
    coach.ask("is Nf3 good here", 8.0)
    assert time.perf_counter() - begun < 0.1, "the caller never waits on the check"
    assert started.wait(2.0)
    time.sleep(0.4)
    assert calls and calls[0] != threading.main_thread().name
    out = _messages(coach)
    assert [m.text for m in out] == ["Not quite. What is your most active piece already eyeing?"]
    assert "Their claims, checked against the board" in provider.prompts[0][1][-1]["content"]
    # A bare hint never runs the check.
    coach.hint(9.0)
    _messages(coach)
    assert len(calls) == 1


def test_an_ask_overtakes_unprompted_requests_still_waiting() -> None:
    from sidecoach.coach.coach import Speaker, _Request

    provider = ScriptedProvider([])
    speaker = Speaker(provider)
    coach, _ = _coach([])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    position = coach.positions[6]
    assert position.facts is not None

    def request(trigger: Trigger) -> _Request:
        return _Request(
            trigger=trigger,
            level=DisclosureLevel.QUESTION,
            system="",
            messages=[{"role": "user", "content": "x"}],
            facts=position.facts,  # type: ignore[arg-type]
            board=position.board,
            fallback="fallback",
            ply=6,
            game_id="g",
        )

    speaker.submit(request(Trigger.STUCK))
    speaker.submit(request(Trigger.POST_MOVE))
    speaker.submit(request(Trigger.USER_ASK))
    assert speaker.dropped == 2 and speaker.pending == 1
    speaker.run_pending()
    assert [s.message.trigger for s in speaker.poll()] == [Trigger.USER_ASK]
