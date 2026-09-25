"""The coach's dialogue: what it says, when, from which facts, and what it may never say."""

from __future__ import annotations

from collections.abc import Sequence

import chess

from sidecoach.coach.coach import Coach, unlock_for
from sidecoach.coach.disclosure import PositionFacts, Unlock, visible_facts
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
from sidecoach.events import (
    Analysis,
    DisclosureLevel,
    GameEnded,
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


def _mate_analysis(ply: int, decisive: bool = True, game_id: str = "g") -> Analysis:
    return Analysis(
        game_id=game_id,
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


def _given(provider: ScriptedProvider, index: int = -1) -> list[ChatMessage]:
    """The messages the model was given on the `index`-th call: the conversation, then
    the directive."""
    return provider.prompts[index][1]


def _harness(messages: list[ChatMessage]) -> str:
    """The one user message: the harness with the task at its end."""
    assert len(messages) == 1 and messages[0]["role"] == "user"
    return messages[0]["content"]


def _current_move_message(messages: list[ChatMessage]) -> str:
    """The 'Position now' section: the position the player faces, at the level in force."""
    text = _harness(messages)
    assert "# Position now" in text, "no position in the harness"
    return text.split("# Position now", 1)[1].split("# Task", 1)[0]


def _directive(messages: list[ChatMessage]) -> str:
    return _harness(messages).rsplit("# Task" + chr(10), 1)[1]


def _record(messages: list[ChatMessage]) -> str:
    """The 'Game so far' section."""
    text = _harness(messages)
    return (
        text.split("# Game so far", 1)[1].split("# Position now", 1)[0]
        if "# Game so far" in text
        else ""
    )


def test_a_decisive_moment_earns_the_idea_and_names_no_move() -> None:
    coach, provider = _coach(["You can mate right now. Which check leaves their king stuck?"])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    out = _messages(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.DECISIVE_QUESTION
    assert out[0].level is DisclosureLevel.IDEA
    # The model was handed the game as one message: the setting, the record one line per
    # move, the position the player faces in full with the engine's lines withheld.
    messages = _given(provider, 0)
    harness = _harness(messages)
    assert harness.startswith("# Setting\nYou coach White.")
    record = _record(messages)
    assert [line.split(" (")[0] for line in record.strip().splitlines()] == [
        "1. e4",
        "1... e5",
        "2. Bc4",
        "2... Nc6",
        "3. Qh5",
    ]
    current = _current_move_message(messages)
    assert "White to move." in current and "decisive: yes" in current
    assert "3... Nf6 by the opponent" in current
    assert "Pieces -- White: K e1, Q h5" in current, "the position is given piece by piece"
    assert "Legal moves" not in current, "the moves are withheld with the lines at this level"
    assert "## Engine lines" + chr(10) + "withheld" in current
    # The idea is given -- the goal and the pattern -- and nothing that plays it.
    assert "goal: the player can mate in one move" in current
    assert "pattern: a mating attack" in current
    for secret in ("Qxf7", "queen", "h5f7", "candidates", "Checks"):
        assert secret not in current, secret
    assert "the idea: the goal, the pattern and why it works" in _directive(messages)
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


def test_two_hints_give_the_idea_then_the_focus_and_asking_for_the_move_gives_it() -> None:
    coach, provider = _coach(
        [
            "You can mate in one: their king has no square. Which check is it?",
            "Their king on e8 is boxed in; your queen does it. Which check?",
            "Qxf7# ends it.",
        ]
    )
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    assert _messages(coach) == [], "nothing unprompted in a quiet position"
    # The first hint in a fresh position is the idea: the goal and the pattern.
    coach.ask("give me a hint", 8.0)
    out = _messages(coach)
    assert out[0].level is DisclosureLevel.IDEA
    current = _current_move_message(_given(provider, 0))
    assert "goal: the player can mate in one move" in current and "Qxf7" not in current
    facts = current.split("## Facts (the player to move)")[1].split("## Said here")[0]
    assert "queen" not in facts and "target:" not in facts, "not the piece or the target yet"
    # The player's words stand under the position, verbatim.
    assert '## Said here\n    PLAYER said: "give me a hint"' in current
    # The second, the focus: the target and the piece, never the move.
    coach.ask("another hint", 8.5)
    out = _messages(coach)
    assert out[0].level is DisclosureLevel.FOCUS
    current = _current_move_message(_given(provider, 1))
    facts = current.split("## Facts (the player to move)")[1].split("## Said here")[0]
    assert "target: the opponent's king on e8" in facts
    assert "piece: the player's queen on h5" in facts
    assert "Qxf7" not in current, "not even among the checks on offer"
    coach.ask("ok just tell me the move", 9.0)
    out = _messages(coach)
    assert out[0].level is DisclosureLevel.MOVE and "Qxf7" in out[0].text
    current = _current_move_message(_given(provider, 2))
    assert "Engine lines" in current and "Qxf7#" in current
    # Every earlier coach turn stands under the position as an annotation: two hints, each
    # answered at once with the template and then by the model -- and this ask's own quick
    # answer is already there when the model is asked to build on it.
    said = current.split("## Said here\n", 1)[1]
    lines = said.strip().splitlines()
    assert sum(line.startswith("    YOU (the coach) said: ") for line in lines) == 5
    assert lines[-2] == '    PLAYER said: "ok just tell me the move"'
    assert lines[-1].startswith('    YOU (the coach) said: "Qxf7# is mate')
    assert "have just been shown that it is Qxf7" in _directive(_given(provider, 2))


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
    # The move's own message carries the verdict and what the engine preferred; the
    # position before it, now history, shows its lines in full.
    messages = _given(provider)
    played = _current_move_message(messages)
    assert "4. Bxf7+ by the player: miss; it cost 0.45 of expected score" in played
    assert "the engine preferred 4. Qxf7#" in played
    # The record is at full disclosure: the earlier verdicts stand in full.
    assert "3... Nf6 (the opponent)" in _record(messages)


def test_stuck_gives_the_idea_then_the_focus_and_the_opponents_turn_is_silent() -> None:
    coach, _ = _coach(["You can mate here.", "Look at their king."])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    coach.tick(12.0)
    assert _messages(coach) == [], "not stuck yet: ten seconds from the move at 6.0"
    coach.tick(17.0)
    out = _messages(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.STUCK
    assert out[0].level is DisclosureLevel.IDEA
    coach.tick(22.0)
    assert _messages(coach) == [], "the second waits a full spell from the first"
    coach.tick(28.0)
    out = _messages(coach)
    assert len(out) == 1 and out[0].level is DisclosureLevel.FOCUS
    coach.tick(80.0)
    assert _messages(coach) == [], "two unprompted messages per position; the move only asked"
    # With the opponent to move, nothing unprompted.
    quiet, _ = _coach(["..."])
    _start(quiet, Side.BLACK)
    quiet.on_event(_mate_analysis(0), 1.0)
    quiet.tick(100.0)
    assert _messages(quiet) == []


def _mate_in_one_played(coach: Coach) -> None:
    """The user plays Qxf7#, the engine's own move, after the scholar's-mate position."""
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


def test_a_good_move_gets_one_sentence_from_the_model_with_the_template_to_fall_back_on() -> None:
    """With local verdicts off (coach/verdicts.py writes them by default)."""
    # Every move of the player's is a turn for the coach: a fine move gets one short
    # sentence from the model, asked for as two candidates so the call stays short.
    coach, provider = _coach(["[p=0.6] Mate on the board, clean finish.\n[p=0.4] That ends it."])
    coach.local_verdicts = False  # the model's word, not the board's
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    _mate_in_one_played(coach)
    out = _messages(coach)
    assert len(out) == 1 and out[0].text in ("Mate on the board, clean finish.", "That ends it.")
    assert out[0].trigger is Trigger.POST_MOVE
    directive = _directive(_given(provider, 0))
    assert "The player just played Qxf7#" in directive
    assert "best -- the engine's own choice" in directive
    assert "fifteen words at most" in directive and "give TWO different ways" in directive
    # When the reply fails, the two-word template stands in.
    from sidecoach.coach.coach import VERDICTS

    coach, provider = _coach(["", ""])
    coach.local_verdicts = False  # the model's word, not the board's
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    _mate_in_one_played(coach)
    out = _messages(coach)
    assert len(out) == 1 and out[0].text in [p.format(san="Qxf7#") for p in VERDICTS["best"]]
    assert len(provider.prompts) == 2, "one call, one retry, then the template"


def test_in_blitz_a_good_move_earns_two_words_without_the_model() -> None:
    from sidecoach.coach.coach import VERDICTS
    from sidecoach.coach.policy import TimeControl

    coach, provider = _coach(["never used"])
    coach.local_verdicts = False
    _play_to_scholar(coach)
    coach.policy.time_control = TimeControl.BLITZ
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    _mate_in_one_played(coach)
    out = _messages(coach)
    assert len(out) == 1 and out[0].text in VERDICTS["best"]
    assert provider.prompts == [], "no model call for a good move in blitz"
    # The wording is by the label and never the same twice running: "good, good, good"
    # for a whole game wore thin. Never the move: the player just played it.
    seen = {coach._verdict("good") for _ in range(40)}
    assert len(seen) >= 3
    previous = None
    for _ in range(40):
        current = coach._verdict("good")
        assert current != previous
        previous = current
    assert coach._verdict("book") in VERDICTS["book"]


def test_in_blitz_a_mistake_gets_a_word_that_does_not_call_it_fine() -> None:
    # Blitz keeps only the word after the move (the policy), so a blunder comes to the
    # directive as POST_MOVE: no lesson and no question, but not "fine" either.
    coach, _ = _coach([])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 6.5)
    position = coach.positions[6]
    assert position.facts is not None
    board = chess.Board(SCHOLAR)
    move = board.parse_san("a3")
    board.push(move)
    played = MovePlayed(
        game_id="g",
        ply=7,
        san="a3",
        uci=move.uci(),
        fen_before=SCHOLAR,
        fen_after=board.fen(),
        by_user=True,
    )
    blunder = MoveAssessment(
        game_id="g",
        ply=7,
        uci=move.uci(),
        expected_loss=0.5,
        severity=Severity.BLUNDER,
        best=["h5f7"],
        label=Label.BLUNDER,
    )
    instruction, fallback = coach._instruction(
        Trigger.POST_MOVE, position, DisclosureLevel.LINE, None, (), blunder, played
    )
    from sidecoach.coach.coach import COST_YOU, FINE_TIGHTER

    assert fallback in [c.format(best="Qxf7#") + "." for c in COST_YOU]
    assert "which was a blunder" in instruction and "fine" not in instruction
    assert "no lesson, no question" in instruction
    # A move a little behind the best is still fine.
    good = blunder.model_copy(
        update={"expected_loss": 0.03, "severity": Severity.NONE, "label": Label.GOOD}
    )
    _, fallback = coach._instruction(
        Trigger.POST_MOVE, position, DisclosureLevel.LINE, None, (), good, played
    )
    # The better move and what it does, and the pattern to keep.
    assert any(fallback.startswith(f.format(best="Qxf7#")) for f in FINE_TIGHTER)
    from sidecoach.coach.coach import KEEP_LEADS

    assert "it is mate" in fallback
    assert any(f"{lead.capitalize()}:" in fallback for lead in KEEP_LEADS)
    # Said once, the lesson is not said again a move later: the verdict and the better move
    # stand alone (Nova bot games, 2026-09-25: 27% of lessons came again within three moves).
    coach.conversation.coach_said(fallback)
    _, again = coach._instruction(
        Trigger.POST_MOVE, position, DisclosureLevel.LINE, None, (), good, played
    )
    assert any(again.startswith(f.format(best="Qxf7#")) for f in FINE_TIGHTER)
    assert "it is mate" not in again
    assert not any(f"{lead.capitalize()}:" in again for lead in KEEP_LEADS), again
    assert "Do not open with the move they played" in instruction


def test_a_lesson_is_said_of_the_board_the_player_sees_and_not_twice() -> None:
    from sidecoach.coach.coach import _stale, fresh

    # 4.d3 had just let the bishop out: "still shut in behind the pawn on d2" was the
    # position before the move (Nova bot games, 2026-09-25).
    assert _stale("your bishop on c1 is still shut in behind the pawn on d2", "d2d3")
    assert not _stale("your bishop on c1 is still shut in behind the pawn on d2", "b1c3")
    # A line is about the moves to come, and stands.
    assert not _stale("after Qg2, Rdg1, and the queen on g2 is trapped", "g2g3")
    said = (
        "That works, though d4 was a touch better -- your bishop on c1 is still shut in behind "
        "the pawn on d2. The lesson to keep: the centre is where pieces have the most room."
    )
    text = (
        "Playable -- d4 was a little more precise -- your bishop on c1 is still shut in behind "
        "the pawn on d2. Worth remembering: the centre is where pieces have the most room."
    )
    assert fresh(text, said) == "Playable -- d4 was a little more precise."
    assert fresh(text, "Nice.") == text


def test_the_draft_checker_can_veto_a_false_claim() -> None:
    def checker(text: str, board: chess.Board, after: Sequence[chess.Board] = ()) -> list[str]:
        return ["the rook on f7 is not there"] if "rook on f7" in text else []

    provider = ScriptedProvider(
        ["Your rook on f7 is strong.", "Your king is safe for now, is it not?"]
    )
    coach = Coach(
        provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)), draft_check=checker
    )
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    out = _messages(coach)
    assert out[0].text == "Your king is safe for now, is it not?"
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

    block = "[p=0.6] Your rook on f7 is strong.\n[p=0.4] Your king is safe for now, is it not?"
    provider = ScriptedProvider([block])
    coach = Coach(
        provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)), draft_check=checker
    )
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    out = _messages(coach)
    assert out[0].text == "Your king is safe for now, is it not?"
    assert coach.spoken[-1].attempts == 1 and coach.speaker.fallbacks == 0


def test_the_fact_sheet_hides_by_level() -> None:
    facts = PositionFacts(
        fen=SCHOLAR,
        side_to_move="white",
        phase="opening",
        material="level",
        theme="a mating attack",
        region="the kingside, around f7",
        piece="your queen",
        candidates=["Qxf7#"],
        line=["Qxf7#"],
        decisive=True,
    )
    assert visible_facts(facts, DisclosureLevel.IDEA)["theme"] == "a mating attack"
    assert "region" not in visible_facts(facts, DisclosureLevel.IDEA)
    assert "piece" not in visible_facts(facts, DisclosureLevel.IDEA)
    assert visible_facts(facts, DisclosureLevel.FOCUS)["region"] == "the kingside, around f7"
    assert "candidates" not in visible_facts(facts, DisclosureLevel.FOCUS)
    assert visible_facts(facts, DisclosureLevel.LINE)["line"] == ["Qxf7#"]


def test_the_conversation_is_the_game_one_message_per_move_with_the_tally() -> None:
    """The model sees the game as it went: every move a message with the engine's word on
    it and the running tally of the player's moves; nothing is summarised away."""
    coach, provider = _coach(["Two clean moves so far. Keep developing."])
    _play_to_scholar(coach)
    for ply, (uci, label) in {1: ("e2e4", Label.BOOK), 3: ("f1c4", Label.EXCELLENT)}.items():
        coach.on_event(
            MoveAssessment(
                game_id="g",
                ply=ply,
                uci=uci,
                expected_loss=0.0,
                severity=Severity.NONE,
                best=[uci],
                label=label,
            ),
            7.0,
        )
    coach.on_event(_mate_analysis(6, decisive=False), 8.0)
    coach.ask("how is my opening?", 9.0)
    _messages(coach)
    messages = _given(provider)
    record = _record(messages).strip().splitlines()
    assert record[0] == "1. e4 (the player): book."
    assert record[2] == "2. Bc4 (the player): excellent."
    assert len(record) == 5, "one line per move played before the position now"
    current = _current_move_message(messages)
    assert "3... Nf6 by the opponent" in current
    assert "The player's moves so far (2 judged): 1 excellent, 1 book." in current
    assert current.rstrip().endswith('PLAYER said: "how is my opening?"')


def test_a_takeback_is_a_message_that_says_so_and_history_keeps_the_withdrawn_move() -> None:
    from sidecoach.events import Takeback

    coach, provider = _coach(["Fine."])
    _play_to_scholar(coach)
    fen_before = chess.Board(SCHOLAR).fen()
    board = chess.Board(SCHOLAR)
    move = board.parse_san("Bxf7+")
    board.push(move)
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=7,
            san="Bxf7+",
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=board.fen(),
            by_user=True,
        ),
        10.0,
    )
    coach.on_event(Takeback(game_id="g", plies=1, fen=fen_before), 11.0)
    board = chess.Board(SCHOLAR)
    move = board.parse_san("Qxf7#")
    board.push(move)
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=7,
            san="Qxf7#",
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=board.fen(),
            by_user=True,
        ),
        12.0,
    )
    coach.on_event(_mate_analysis(6, decisive=False), 12.5)  # facts for the ask below
    coach.ask("what happened there?", 13.0)
    _messages(coach)
    harness = _harness(_given(provider))
    withdrawn = harness.index("4. Bxf7+ (the player)")
    takeback = harness.index("    EVENT: Takeback: 4. Bxf7+ withdrawn")
    replacement = harness.index("4. Qxf7# by the player")
    assert withdrawn < takeback < replacement
    assert "back at the position after 3... Nf6" in harness
    # The withdrawn move stays in the record as something that happened, marked so; the
    # position now is the replacement's.
    assert harness.count("4. Bxf7+") == 2, "the withdrawn move: its line and the takeback"


def test_a_reread_of_the_board_is_a_message_and_the_numbering_restarts() -> None:
    from sidecoach.events import Resync

    coach, provider = _coach(["Understood."])
    _play_to_scholar(coach)
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/2N5/PPPP1PPP/R1B1K1NR b KQkq - 5 4"
    coach.on_event(
        Resync(game_id="g", fen=fen, reason="position adopted after 3 unexplained cells"), 20.0
    )
    board = chess.Board(fen)
    move = board.parse_san("d6")
    board.push(move)
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=1,
            san="d6",
            uci=move.uci(),
            fen_before=fen,
            fen_after=board.fen(),
            by_user=False,
        ),
        21.0,
    )
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=1,
            lines=[Line(moves=["c3d5"], depth=12, expected_score=0.6)],
            decisive=False,
            only_move=False,
        ),
        22.0,
    )
    coach.ask("what now?", 23.0)
    _messages(coach)
    harness = _harness(_given(provider))
    assert "    EVENT: The board was re-read" in harness and "Moves may have gone unseen" in harness
    # The numbering follows the position the board was read at, not a ply count.
    assert "4... d6 by the opponent" in _current_move_message(_given(provider))


def test_asking_before_the_engine_has_looked_is_answered_when_it_has() -> None:
    """A question asked before the engine has looked at the position on the board is told
    to wait a moment and answered when the analysis lands, never from an older position
    (Tai, 2026-09-24: a hint pressed as a puzzle appeared got "I have not seen enough of
    this game yet to say. Give me a move or two.")."""
    coach, _ = _coach([])
    _start(coach)
    coach.ask("hint", 1.0)
    out = _messages(coach)
    assert [m.text for m in out] == ["One moment -- still looking at this position."]
    coach.ask("hint", 1.5)  # asked again while waiting: one "moment" is enough
    assert _messages(coach) == []
    start = chess.Board()
    coach.on_event(
        Analysis(
            game_id="g",
            fen=start.fen(),
            ply=0,
            lines=[
                Line(moves=["e2e4", "e7e5", "g1f3"], depth=18, expected_score=0.55),
                Line(moves=["d2d4", "d7d5"], depth=18, expected_score=0.54),
            ],
            decisive=False,
            only_move=False,
        ),
        3.0,
    )
    answered = _messages(coach)
    assert answered and all(m.trigger is Trigger.USER_ASK for m in answered)
    assert "One moment" not in answered[0].text


# The puzzle of Tai's desktop, 2026-09-24: Black to move, 1...Qd1+ 2.Rxd1 Rxd1+ 3.Qe1 Rxe1#.
QD1 = "2kr2nr/2p5/2p1p1pp/Q3Pp2/3R4/P3P3/1Pq2PPP/R1B3K1 b - - 1 1"
QD1_LINE = ["c2d1", "d4d1", "d8d1", "a5e1", "d1e1"]


def _puzzle(coach: Coach, fen: str = QD1) -> chess.Board:
    coach.on_event(
        GameStarted(game_id="g", user_side=Side.BLACK, start_fen=fen, kind="position"), 0.0
    )
    return chess.Board(fen)


def _analysed(
    coach: Coach, board: chess.Board, ply: int, line: list[str], ts: float, mate: int | None
) -> None:
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=ply,
            lines=[
                Line(moves=line, depth=18, expected_score=1.0, mate=mate),
                Line(moves=["h6h5"], depth=18, expected_score=0.1),
            ],
            decisive=True,
            only_move=True,
        ),
        ts,
    )


def _played(coach: Coach, board: chess.Board, ply: int, uci: str, by_user: bool, ts: float) -> None:
    move = chess.Move.from_uci(uci)
    before = board.fen()
    san = board.san(move)
    board.push(move)
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=ply,
            san=san,
            uci=uci,
            fen_before=before,
            fen_after=board.fen(),
            by_user=by_user,
        ),
        ts,
    )


def _hint_then(coach: Coach, board: chess.Board) -> None:
    """The puzzle appears, the player presses Hint before the engine has looked, and the
    hint comes when it has: no puzzle question was asked, as on Tai's desktop."""
    coach.ask("hint", 0.5)
    _analysed(coach, board, 0, QD1_LINE, 1.0, 3)
    got = _messages(coach)
    assert got and all(m.trigger is Trigger.USER_ASK for m in got)


def test_no_new_question_while_the_player_plays_out_the_line_they_found() -> None:
    """After 1...Qd1+ 2.Rxd1 Rxd1+ 3.Qe1 the coach asked "You can mate in one move ...
    Which check leaves their king no square at all?" -- a fresh question in the middle of
    the combination the player was carrying out (Tai's desktop, 2026-09-24)."""
    coach, _ = _coach([])
    board = _puzzle(coach)
    _hint_then(coach, board)
    for ply, uci in enumerate(QD1_LINE[:4], start=1):
        _played(coach, board, ply, uci, ply % 2 == 1, 2.0 + ply)
        if ply % 2 == 0:
            _analysed(coach, board, ply, QD1_LINE[ply:], 2.5 + ply, (5 - ply) // 2)
    asked = [m.text for m in _messages(coach) if m.trigger is Trigger.DECISIVE_QUESTION]
    assert asked == [], asked


def test_an_opponent_who_leaves_the_line_brings_the_question_back() -> None:
    coach, _ = _coach([])
    board = _puzzle(coach)
    _hint_then(coach, board)
    _played(coach, board, 1, "c2d1", True, 3.0)
    _played(coach, board, 2, "a5e1", False, 4.0)  # blocks instead of taking: off the line
    _analysed(coach, board, 2, ["d1e1"], 4.5, 1)
    asked = [m for m in _messages(coach) if m.trigger is Trigger.DECISIVE_QUESTION]
    assert len(asked) == 1


def test_the_next_puzzle_gets_its_question_after_a_long_one() -> None:
    """The decisive stretch pointed at in one puzzle said nothing about the next: move
    numbers start again, and a new puzzle's first position fell inside the last one's
    stretch, so its question was never asked (Tai's desktop, 2026-09-24)."""
    from sidecoach.events import Resync

    coach, _ = _coach([])
    board = _puzzle(coach, "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1")
    for ply, uci in enumerate(["g8f8", "a1a2", "f8e7", "a2a3"], start=1):
        _played(coach, board, ply, uci, ply % 2 == 1, float(ply))
    _analysed(coach, board, 4, ["e7d6", "a3a6"], 5.0, None)
    assert [m.trigger for m in _messages(coach)] == [Trigger.DECISIVE_QUESTION]
    coach.on_event(
        Resync(game_id="g", fen=QD1, reason="position adopted after 30 unexplained cells"), 20.0
    )
    _analysed(coach, chess.Board(QD1), 0, QD1_LINE, 21.0, 3)
    assert [m.trigger for m in _messages(coach)] == [Trigger.DECISIVE_QUESTION]


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
    assert [m.text for m in quick] == ["Qxf7# is mate."]
    assert quick[0].trigger is Trigger.USER_ASK and quick[0].level is DisclosureLevel.MOVE
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


def test_material_is_always_from_the_users_side_and_the_review_draws_on_the_tally() -> None:
    """The first post-game question ("how did I do overall?") was answered "you were down
    a queen for much of the middlegame" to a player who had won a queen: the material was
    read from whichever side was to move, and the memory carried that into the verdict.
    Now the facts, the sheet and the memory all speak from the user's side, a finished
    game withholds nothing, and an overall verdict is drawn from the engine's tally."""
    from sidecoach.events import GameEnded

    coach, provider = _coach(["You kept it clean: nothing lost, and the finish was sharp."])
    _start(coach)
    board = chess.Board()
    # 1. e4 e5 2. Qh5 Nc6 3. Qxe5+ (White, the user, wins a pawn; then Black to move.)
    for ply, san in enumerate(("e4", "e5", "Qh5", "Nc6", "Qxe5+"), start=1):
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
    # The engine looks at the position with Black to move.
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=5,
            lines=[Line(moves=["f8e7"], depth=12, expected_score=0.4)],
            decisive=False,
            only_move=False,
        ),
        6.0,
    )
    facts = coach.positions[5].facts
    assert facts is not None
    assert facts.material == "you are up a pawn", "from the user's side, not the mover's"
    assert facts.user_side == "white" and facts.side_to_move == "black"
    assert "Material: the player is up a pawn." in coach.conversation.render_ply(
        coach.conversation.records[-1], DisclosureLevel.LINE
    )
    # The user's moves get their labels tallied.
    for ply, (uci, label) in {
        1: ("e2e4", Label.BOOK),
        3: ("d1h5", Label.INACCURACY),
        5: ("h5e5", Label.BEST),
    }.items():
        coach.on_event(
            MoveAssessment(
                game_id="g",
                ply=ply,
                uci=uci,
                expected_loss=0.03 if label is Label.INACCURACY else 0.0,
                severity=Severity.INACCURACY if label is Label.INACCURACY else Severity.NONE,
                best=[uci],
                label=label,
            ),
            7.0,
        )
    _messages(coach)
    # The game ends; an overall question is answered from the tally, at full disclosure.
    coach.on_event(
        GameEnded(game_id="g", result="1-0", reason="resigned", pgn="", final_fen=board.fen()), 8.0
    )
    coach.ask("how did I do overall this game?", 9.0)
    out = _messages(coach)
    assert out and out[-1].level is DisclosureLevel.LINE
    messages = _given(provider)
    directive = _directive(messages)
    assert "They asked how they did overall" in directive
    assert "do not infer the course of the game from the position" in directive
    harness = _harness(messages)
    assert "The player's moves so far (3 judged): 1 best, 1 book, 1 inaccuracy." in harness
    assert "Ground lost at: 2. Qh5 (inaccuracy)" in harness
    assert "    EVENT: Game over: 1-0" in harness
    # Other ways of asking are reviews too, and when the model's review fails the checks
    # the template is the record, not "I have no line for this yet".
    from sidecoach.coach.coach import _ASK_REVIEW

    for words in ("what do I keep missing?", "where did I lose it?", "how am I playing so far?"):
        assert _ASK_REVIEW.search(words), words
    assert not _ASK_REVIEW.search("what's the plan so far?")
    provider.replies = ["", ""]
    coach.ask("what do I keep missing?", 10.0)
    out = _messages(coach)
    assert out[-1].text == (
        "1 of your 3 moves were best or excellent. You lost the most ground at 2. Qh5 (inaccuracy)."
    )


def test_a_quick_answer_is_a_turn_before_the_model_and_a_failed_reply_adds_nothing() -> None:
    # The model's two candidates both name the mating move at the IDEA level: rejected
    # twice. With the quick answer already out, nothing is said in its place -- the same
    # line a second time was what the first rendered example showed.
    coach, provider = _coach(["[p=0.5] Play Qxf7#.\n[p=0.5] Qxf7 mates.", "[p=1.0] Qxf7#!"])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    coach.ask("hint", 8.0)
    out = _messages(coach)
    # The quick answer is the idea template: the goal and the pattern.
    assert len(out) == 1 and "mate in one" in out[0].text.split(".")[0]
    quick = out[0].text
    # The request carried the quick answer as the coach's own turn, then the directive.
    given = _given(provider, 0)
    said = _current_move_message(given).split("## Said here", 1)[1].strip(chr(10)).splitlines()
    assert said[-2] == '    PLAYER said: "hint"'
    assert said[-1] == f'    YOU (the coach) said: "{quick}"'
    assert "build on it, do not repeat it" in _directive(given)
    assert "PLAYER said" not in _directive(given)
    # The rejected reply is in the log, with its calls, and nowhere else.
    rejected = [s for s in coach.spoken if not s.message.text]
    assert len(rejected) == 1 and rejected[0].attempts == 2 and rejected[0].problems
    turns = [e for e in coach.conversation.entries if e.role == "assistant"]
    assert [e.text for e in turns] == [quick]


def test_a_fact_question_falls_back_to_the_fact_not_the_hint() -> None:
    """ "How many pieces am I up?" once fell back, when the model's reply failed, to the
    hint for the position ("Something forcing is on..."): the fallback for a question
    about a fact that is never withheld is that fact."""
    coach, provider = _coach(["", ""])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    coach.ask("how many pieces am I up?", 8.0)
    out = _messages(coach)
    assert out and out[-1].text.startswith("Material:")
    provider.replies = ["", ""]
    coach.ask("am I winning?", 9.0)
    out = _messages(coach)
    assert out and out[-1].text.startswith("Right now you are winning")
    provider.replies = ["", ""]
    coach.ask("what opening is this?", 10.0)
    out = _messages(coach)
    assert out and (
        out[-1].text == "We left the opening book early."
        or out[-1].text.startswith("The opening was the")
    )


def test_a_retry_does_not_grow_the_prompt() -> None:
    """When every candidate fails the checks, the second try carries the complaint inside
    the directive: appending the failed attempt as two more messages doubled the prompt
    on a long game, past the model's context."""
    coach, provider = _coach(["[p=0.5] Play Qxf7#.\n[p=0.5] Qxf7 mates.", "[p=1.0] Look around."])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    coach.ask("hint", 8.0)
    _messages(coach)
    first, second = _given(provider, 0), _given(provider, 1)
    assert len(second) == len(first)
    assert "Those replies are not allowed" in second[-1]["content"]
    assert "Your first attempt was" in second[-1]["content"]


def _bishop_check_instead_of_mate(coach: Coach) -> None:
    """The user plays Bxf7+ where Qxf7 was mate, and the engine has judged it."""
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    _messages(coach)
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


def test_a_question_about_the_last_move_gets_its_verdict_in_full() -> None:
    """ "Why was that bad?" is about a position that is past: the directive states the
    verdict, the cost, the move the engine preferred and the idea, and says nothing is
    withheld. The generic ask directive ("the better move is withheld") had the model
    answer with a hint about the position now instead."""
    coach, provider = _coach(["...", "..."])
    _bishop_check_instead_of_mate(coach)
    provider.replies = ["", ""]
    coach.ask("why was that bad?", 12.0)
    out = _messages(coach)
    directive = _directive(_given(provider))
    assert "They ask about the move they already played, Bxf7+" in directive
    assert "cost 45% of the game" in directive and "the engine preferred Qxf7#" in directive
    assert "nothing about it is withheld" in directive
    # Spoken from the position the move was played from, with everything shown: naming
    # the better move is not a leak, as it is not in the lesson after a mistake.
    assert out and out[-1].level is DisclosureLevel.LINE
    assert out[-1].fen == SCHOLAR
    # The fallback is the verdict itself, not a hint.
    assert out[-1].text.startswith("It cost you 45% of the game. Qxf7# was better")


def test_a_question_about_a_fine_last_move_says_it_was_fine() -> None:
    coach, provider = _coach(["...", ""])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    _messages(coach)
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
    _messages(coach)
    coach.ask("was that a mistake?", 12.0)
    _messages(coach)
    directive = _directive(_given(provider))
    assert "It was not bad: the verdict was best" in directive
    assert "do not invent a fault" in directive


def test_a_move_of_the_game_so_far_is_never_a_leak() -> None:
    """Asked why the opponent played their last move, the coach names it. Below CANDIDATE
    the leak check flags every move token, and it flagged the history: a move already
    played may be discussed in full."""
    coach, _ = _coach(["[p=1.0] Nf6 develops with a threat; what does it leave behind?"])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    _messages(coach)
    coach.ask("why did he play that?", 8.0)
    out = _messages(coach)
    assert out and out[-1].text.startswith("Nf6 develops")
    assert out[-1].level is DisclosureLevel.QUESTION


def test_a_named_move_is_analysed_by_the_engine_and_put_before_the_coach() -> None:
    """ "What about Bc4?" once got only "not among the engine's best moves". With an engine
    the object carries the engine's word on that move: the score after it, how far behind
    the best, the reply, what it does -- and the directive says it is theirs to hear."""
    from sidecoach.events import Line

    class FakeEngine:
        def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]:
            # Before the move: the best is Bb5 at 0.55; after 3. Bc4: 0.48 for Black to
            # move (0.52 for White), reply ...Bc5.
            if board.turn == chess.WHITE:
                return [Line(moves=["f1b5"], depth=14, expected_score=0.55)]
            return [Line(moves=["f8c5", "d2d3"], depth=14, expected_score=0.48)]

    # The coach's own model plans what to examine (first call), then answers (second).
    coach, provider = _coach(
        [
            '{"probes": [{"label": "your Bc4", "moves": ["Bc4"]}]}',
            "[p=1.0] Bc4 is fine: a little behind the best. What does it aim at?",
        ]
    )
    coach.engine = FakeEngine()
    _start(coach)
    board = chess.Board()
    for ply, san in enumerate(("e4", "e5", "Nf3", "Nc6"), start=1):
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
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=4,
            lines=[Line(moves=["f1b5"], depth=12, expected_score=0.55)],
            decisive=False,
            only_move=False,
            tags=[],
        ),
        5.0,
    )
    _messages(coach)
    coach.ask("what about Bc4?", 6.0)
    out = _messages(coach)
    # The examination goes out first, in plain words, as the coach's own turn; the model's
    # reply builds on it.
    assert [m.text for m in out] == [
        "your Bc4: after 3. Bc4 it is about equal, a little behind the best; they would answer "
        "3... Bc5 4. d3; Bc4 attacks the pawn on f7.",
        "Bc4 is fine: a little behind the best. What does it aim at?",
    ]
    said = (
        _current_move_message(_given(provider))
        .split("## Said here", 1)[1]
        .strip(chr(10))
        .splitlines()
    )
    assert said[-1].startswith('    YOU (the coach) said: "your Bc4:')
    directive = _directive(_given(provider))
    assert "have just been shown the engine's word on it" in directive
    assert (
        "Engine on your Bc4, which the player asked about: after 3. Bc4 the expected score "
        "is 0.52" in directive
    )
    assert "0.03 behind the engine's best;" in directive
    assert "the opponent's reply would be 3... Bc5 4. d3" in directive
    assert "what Bc4 does: attacks the pawn on f7" in directive
    # Below LINE the engine's own preferred move is not named.
    assert "Bb5" not in directive
    assert "Name no move the messages do not give" in directive


def test_the_review_is_spoken_from_the_profiles_facts_and_the_next_game_knows_the_player() -> None:
    """The game ends and the profile's review arrives: it joins the conversation as a
    message of its own and the coach sums it up through the model, the review's sentences
    as the template. The profile's snapshot puts what earlier games say into the next
    game's setting."""
    from sidecoach.events import GameEnded, GameReview, ProfileSnapshot, TurningPoint

    summary = (
        "You won by checkmate. Your one lapse was 3. Qh5, where Nf3 kept the edge; "
        "the rest was clean. Work on spotting what your opponent threatens."
    )
    # The same reply to every call: the earlier triggers (the decisive question, the
    # verdict on the mate) take their share of them; the review is the last.
    coach, provider = _coach([summary] * 8)
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 6.5)  # facts for the position the review is checked on
    _mate_in_one_played(coach)
    coach.on_event(GameEnded(game_id="g", result="1-0", reason="checkmate"), 12.0)
    review = GameReview(
        game_id="g",
        user_side=Side.WHITE,
        result="1-0",
        reason="checkmate",
        moves=4,
        accuracy=88.0,
        tallies={"best": 3, "mistake": 1},
        turning_points=[
            TurningPoint(
                ply=5,
                san="3. Qh5",
                label="mistake",
                expected_loss=0.11,
                best_san="Nf3",
                cause="missed opponent threat",
                motif="fork",
            )
        ],
        lines=[
            "You played white and won by checkmate in 4 moves.",
            "Accuracy 88.0. Your 4 moves: 3 best, 1 mistake.",
            "Ground lost at 3. Qh5, a mistake costing 0.11; Nf3 was better"
            " (missed opponent threat: fork).",
        ],
    )
    coach.on_event(review, 12.5)
    said = _messages(coach)
    assert said and said[-1].trigger.value == "review"
    assert "Qh5" in said[-1].text and "Nf3" in said[-1].text
    given = _given(provider)
    assert "The review of the game:" in _harness(given)
    assert "Sum it up from the review" in _directive(given)
    # The snapshot's insights and last session frame the next game.
    coach.on_event(
        ProfileSnapshot(
            games=3,
            insights=[
                "Your biggest cost over 3 games is missed opponent threat: 4 mistakes,"
                " 0.70 expected score."
            ],
            last_session="2 games (1 won, 1 lost), accuracy 80.",
        ),
        13.0,
    )
    coach.on_event(
        GameStarted(game_id="g2", user_side=Side.BLACK, start_fen=chess.Board().fen()), 14.0
    )
    setting = coach.conversation.render()
    assert "What earlier games say about this player" in setting
    assert "missed opponent threat" in setting and "Last session: 2 games" in setting


def test_a_question_about_the_player_over_their_games_goes_to_the_record() -> None:
    """The record's answerer is asked off the coach's thread; the answer is spoken as the
    coach's turn, the player's words and a note of the lookup join the conversation, and
    the queries are kept for the log. Questions about this game stay with the game."""
    import time

    from sidecoach.profile.ask import Answer, Exchange

    coach, provider = _coach(["Keep looking."])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 6.5)
    asked: list[str] = []

    def lookup(question: str) -> Answer:
        asked.append(question)
        return Answer(
            text="Over 9 games you left the opening ahead 7 times and were behind by move 25.",
            exchanges=[Exchange(sql="SELECT 1", columns=["n"], rows=[(1,)])],
            calls=2,
        )

    coach.profile_ask = lookup
    coach.ask("why am I always losing in the middlegame?", 7.0)
    deadline = time.monotonic() + 5
    while not coach.profile_answers and time.monotonic() < deadline:
        time.sleep(0.02)
    said = _messages(coach)
    assert asked == ["why am I always losing in the middlegame?"]
    assert (
        said and said[-1].text.startswith("Over 9 games") and said[-1].trigger.value == "user_ask"
    )
    assert coach.profile_answers[0][0] == "answer"
    record = coach.conversation.render()
    assert 'PLAYER said: "why am I always losing in the middlegame?"' in record
    assert "looked the question up in the record (1 query)" in record
    # A question about this position never goes to the record.
    coach.ask("what should I play here?", 8.0)
    assert asked == ["why am I always losing in the middlegame?"]


def test_every_third_mistake_ends_by_asking_what_the_player_saw() -> None:
    from sidecoach.coach.coach import ELICIT_ANSWER_S, ELICIT_EVERY

    coach, _ = _coach([])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 6.5)
    position = coach.positions[6]
    assert position.facts is not None
    board = chess.Board(SCHOLAR)
    played = MovePlayed(
        game_id="g",
        ply=7,
        san="a3",
        uci="a2a3",
        fen_before=SCHOLAR,
        fen_after=board.fen(),
        by_user=True,
    )
    assessment = MoveAssessment(
        game_id="g",
        ply=7,
        uci="a2a3",
        expected_loss=0.5,
        severity=Severity.BLUNDER,
        best=["h5f7"],
        label=Label.BLUNDER,
    )
    instruction, fallback = coach._instruction(
        Trigger.POST_MISTAKE, position, DisclosureLevel.LINE, None, (), assessment, played
    )
    assert "expected" not in instruction and "expecting" not in fallback
    instruction, fallback = coach._instruction(
        Trigger.POST_MISTAKE,
        position,
        DisclosureLevel.LINE,
        None,
        (),
        assessment,
        played,
        elicit=True,
    )
    # What they expected, in the coach's own words: the directive never spells out the
    # "missed it or misjudged it" question the model would then repeat verbatim.
    assert "what they expected their move to do" in instruction
    assert "misjudge" not in instruction and "altogether" not in instruction
    assert fallback.endswith("What were you expecting that move to do?")

    # Every third mistake -- while the player answers. The first answered within a
    # minute keeps the cadence; one that goes unanswered is the last of the game.
    due = []
    for n in range(1, 2 * ELICIT_EVERY + 1):
        coach._mistakes = n
        due.append(coach._elicit_due(10.0 * n))
        if due[-1]:
            coach.ask("I thought the queen would have to move", 10.0 * n + ELICIT_ANSWER_S / 2)
    assert due == [n % ELICIT_EVERY == 0 for n in range(1, 2 * ELICIT_EVERY + 1)]
    coach._mistakes = 3 * ELICIT_EVERY
    assert coach._elicit_due(1000.0), "answered: asked again"
    coach._mistakes = 4 * ELICIT_EVERY
    assert not coach._elicit_due(2000.0), "unanswered: not again this game"
    coach._mistakes = 5 * ELICIT_EVERY
    coach.ask("hint", 2000.0 + ELICIT_ANSWER_S + 1)
    assert not coach._elicit_due(3000.0), "a minute later is not an answer"
    _start(coach)
    coach._mistakes = ELICIT_EVERY
    assert coach._elicit_due(4001.0), "a new game starts afresh"


def test_the_coach_is_told_a_position_is_not_a_game() -> None:
    """A puzzle or an analysis board: the coach hears it is a position with no history,
    the player the side at the bottom and, on a puzzle, to move."""
    coach, _ = _coach([])
    puzzle = "r3k2r/pp3ppp/2n1b3/2bpp3/8/1PN1PN2/PBP2PPP/R3KB1R b KQkq - 0 1"
    coach.on_event(
        GameStarted(game_id="p", user_side=Side.BLACK, start_fen=puzzle, kind="position"), 0.0
    )
    record = coach.conversation.render()
    assert "You coach Black." in record
    assert "This is a position, not a game from the start" in record
    assert "joined in progress" not in record


def test_a_puzzle_solved_by_mate_gets_no_word_and_no_summary() -> None:
    """The mating move on a position -- a puzzle -- gets nothing from the coach, and the
    end no summary: the site says the puzzle is solved and moves on (Tai). Other moves
    on a puzzle get their word as in a game."""
    coach, provider = _coach([])
    puzzle = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
    for n in range(2):
        coach.on_event(
            GameStarted(game_id=f"p{n}", user_side=Side.WHITE, start_fen=puzzle, kind="position"),
            10.0 * n,
        )
        coach.on_event(_mate_analysis(0, game_id=f"p{n}"), 10.0 * n + 0.5)
        _messages(coach)  # the decisive-moment nudge on the position itself
        provider.prompts.clear()
        board = chess.Board(puzzle)
        move = board.push_san("Qxf7#")
        coach.on_event(
            MovePlayed(
                game_id=f"p{n}",
                ply=1,
                san="Qxf7#",
                uci=move.uci(),
                fen_before=puzzle,
                fen_after=board.fen(),
                by_user=True,
            ),
            10.0 * n + 1.0,
        )
        coach.on_event(
            MoveAssessment(
                game_id=f"p{n}",
                ply=1,
                uci=move.uci(),
                expected_loss=0.0,
                severity=Severity.NONE,
                best=[move.uci()],
                label=Label.BEST,
            ),
            10.0 * n + 1.5,
        )
        assert _messages(coach) == [], "the mating move is not commented on"
        coach.on_event(
            GameEnded(game_id=f"p{n}", result="1-0", reason="game over on board", pgn="1. Qxf7#"),
            10.0 * n + 2.0,
        )
        assert _messages(coach) == [], "no summary of a puzzle"
    assert provider.prompts == [], "no model call at all"
    # A move on a puzzle that does not end it gets its word, as in a game.
    coach.on_event(
        GameStarted(game_id="q", user_side=Side.WHITE, start_fen=puzzle, kind="position"), 50.0
    )
    coach.on_event(_mate_analysis(0, game_id="q"), 50.5)
    _messages(coach)
    board = chess.Board(puzzle)
    move = board.push_san("Nf3")
    coach.on_event(
        MovePlayed(
            game_id="q",
            ply=1,
            san="Nf3",
            uci=move.uci(),
            fen_before=puzzle,
            fen_after=board.fen(),
            by_user=True,
        ),
        51.0,
    )
    coach.on_event(
        MoveAssessment(
            game_id="q",
            ply=1,
            uci=move.uci(),
            expected_loss=0.5,
            severity=Severity.BLUNDER,
            best=["h5f7"],
            label=Label.BLUNDER,
        ),
        51.5,
    )
    assert [m.trigger for m in _messages(coach)] == [Trigger.POST_MISTAKE]


def test_a_fast_game_gets_the_templates_and_no_model_call() -> None:
    """The user's think times set the pace: under FAST_PACE_S a move, the coach's word
    on a move -- even a mistake's lesson -- is the template written from the board,
    at once, with no model call that could not keep up (Tai, 2026-09-22: a queen
    ending at two seconds a move, the coach silent from ply 80 to the mate). What the
    player asks still goes to the model."""
    from sidecoach.coach.coach import FAST_PACE_S

    coach, provider = _coach(["never used for a move", "[p=1.0] Asked and answered."])
    coach.local_verdicts = False
    _start(coach)
    board = chess.Board()
    plies = ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6")
    for ply, san in enumerate(plies, start=1):
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
                think_ms=int(FAST_PACE_S * 1000 / 3),  # three times faster than the line
            ),
            float(ply),
        )
    assert coach.fast_pace, "two quick moves of the user's and the game counts as fast"
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    _mate_in_one_played(coach)
    out = _messages(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.POST_MOVE and out[0].text
    assert provider.prompts == [], "no model call in a fast game"
    # The ask is the exception: the player is waiting for it.
    coach.ask("what was that about", 12.0)
    out = _messages(coach)
    assert out and out[-1].trigger is Trigger.USER_ASK
    assert len(provider.prompts) >= 1, "the ask goes to the model"
    # Slow down and the model comes back.
    for ply in (8, 9):
        coach.on_event(
            MovePlayed(
                game_id="g",
                ply=ply,
                san="a3",
                uci="a2a3",
                fen_before=board.fen(),
                fen_after=board.fen(),
                by_user=True,
                think_ms=int(FAST_PACE_S * 3000),
            ),
            20.0 + ply,
        )
    assert not coach.fast_pace


def test_the_voice_holds_its_word_in_a_fast_game_and_says_it_when_the_player_slows() -> None:
    from sidecoach.ui.panel import PanelEngine

    va = PanelEngine.voice_action
    assert va("post_move", "Qxf7#: mate.", 7, 7, False, None, True) == "text"
    assert va("post_mistake", "Bb7 was a miss.", 7, 7, False, None, True) == "text"
    assert va("user_ask", "Yes.", 7, 7, False, None, True) == "speak", "an ask is answered aloud"
    assert va("review", "You won.", 7, 7, False, None, True) == "speak"
    assert va("post_move", "Qxf7#: mate.", 7, 7, False, None, False) == "speak"


def test_a_player_who_slows_down_in_a_fast_game_gets_the_model_back() -> None:
    """The pace is the last few think times; the player thinking longer than that in
    the position at hand has slowed down, and a hint or lesson may take the model's
    time (and the voice) again."""
    from sidecoach.coach.coach import FAST_PACE_S

    coach, _provider = _coach([])
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
                think_ms=1000,
            ),
            float(ply),
        )
    assert coach.fast_pace and coach.user_to_move
    assert coach.fast_now(6.0 + 1.0), "a second into the position: still fast"
    assert not coach.fast_now(6.0 + FAST_PACE_S + 1), "thinking longer than the pace: slowed"
