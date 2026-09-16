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


def _given(provider: ScriptedProvider, index: int = -1) -> list[ChatMessage]:
    """The messages the model was given on the `index`-th call: the conversation, then
    the directive."""
    return provider.prompts[index][1]


def _current_move_message(messages: list[ChatMessage]) -> str:
    """The last move message: the position the player faces, at the level in force."""
    moves = [m["content"] for m in messages if m["role"] == "user" and "Move: " in m["content"]]
    assert moves, "no move message in the conversation"
    return moves[-1]


def _directive(messages: list[ChatMessage]) -> str:
    return messages[-1]["content"]


def test_a_decisive_moment_earns_one_question_that_names_nothing() -> None:
    coach, provider = _coach(["What is your opponent's most dangerous idea right now?"])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    out = _messages(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.DECISIVE_QUESTION
    assert out[0].level is DisclosureLevel.QUESTION
    # The model was handed the game as messages: one per move, the last one the position
    # the player faces, drawn as a board, with the engine's lines withheld at this level.
    messages = _given(provider, 0)
    assert messages[0]["content"].startswith("New game. You coach the player, who is White")
    moves = [m for m in messages if m["role"] == "user" and "Move: " in m["content"]]
    assert [m["content"].split("Move: ")[1].split(" by")[0] for m in moves] == [
        "1. e4",
        "1... e5",
        "2. Bc4",
        "2... Nc6",
        "3. Qh5",
        "3... Nf6",
    ]
    current = _current_move_message(messages)
    assert "White to move." in current and "decisive: True" in current
    assert "a b c d e f g h" in current, "the position is drawn as a board"
    assert "Engine lines for the position now: withheld" in current
    for secret in ("Qxf7", "mating", "queen", "h5f7", "candidates"):
        assert secret not in current, secret
    assert "an orienting question only" in _directive(messages)
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
    current = _current_move_message(_given(provider, 0))
    assert "theme:" not in current and "Qxf7" not in current
    # The player's words are a message of the conversation, verbatim.
    asks = [m["content"] for m in _given(provider, 0) if m["content"].startswith("The player says")]
    assert asks == ['The player says: "give me a hint"']
    coach.ask("another hint", 8.5)
    out = _messages(coach)
    assert out[0].level is DisclosureLevel.THEME
    current = _current_move_message(_given(provider, 1))
    assert "theme: a mating attack" in current and "Qxf7" not in current
    assert "queen" not in current.split("Facts for the position you face")[1]
    coach.ask("ok just tell me the move", 9.0)
    out = _messages(coach)
    assert out[0].level is DisclosureLevel.CANDIDATE and "Qxf7" in out[0].text
    current = _current_move_message(_given(provider, 2))
    assert "Engine lines for the position now" in current and "Qxf7#" in current
    # Every earlier coach turn is in the conversation as an assistant message: two hints,
    # each answered at once with the template and then by the model -- and this ask's own
    # quick answer is already a turn when the model is asked to build on it.
    given = _given(provider, 2)
    roles = [m["role"] for m in given]
    assert roles.count("assistant") == 5
    assert roles[-3:] == ["user", "assistant", "user"]
    assert given[-3]["content"] == 'The player says: "ok just tell me the move"'
    assert given[-2]["content"].startswith("The move is Qxf7")
    assert "have just been shown that it is Qxf7" in given[-1]["content"]


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
    assert "Move: 4. Bxf7+ by you" in played
    assert "Engine verdict on Bxf7+: miss; it cost 0.45 of expected score" in played
    assert "the engine preferred 4. Qxf7#" in played
    before = [m["content"] for m in messages if "Move: 3... Nf6" in m["content"]][0]
    assert "1. 4. Qxf7#" in before, "history is at full disclosure"


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
    # Every move of the player's is a turn for the coach: a fine move gets one short
    # sentence from the model, asked for as two candidates so the call stays short.
    coach, provider = _coach(["[p=0.6] Mate on the board, clean finish.\n[p=0.4] That ends it."])
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
    _play_to_scholar(coach)
    coach.policy.time_control = TimeControl.BLITZ
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    _mate_in_one_played(coach)
    out = _messages(coach)
    assert len(out) == 1 and out[0].text in [p.format(san="Qxf7#") for p in VERDICTS["best"]]
    assert provider.prompts == [], "no model call for a good move in blitz"
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
    e4 = [m["content"] for m in messages if "Move: 1. e4 by you" in m["content"]][0]
    assert "Engine verdict on e4: book; it cost nothing" in e4
    assert "Your moves so far (1 judged): 1 book." in e4
    bc4 = [m["content"] for m in messages if "Move: 2. Bc4 by you" in m["content"]][0]
    assert "Your moves so far (2 judged): 1 excellent, 1 book." in bc4
    # The moves so far are written on the current message only; a history message carries
    # its own move and no copy of the growing list.
    assert "Moves so far:" not in bc4
    current = [
        m["content"] for m in messages if m["role"] == "user" and "Moves so far:" in m["content"]
    ]
    assert len(current) == 1
    assert current[0].startswith("Moves so far: 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6")
    assert messages[-2]["content"] == 'The player says: "how is my opening?"'


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
    texts = [m["content"] for m in _given(provider)]
    withdrawn = [t for t in texts if "Move: 4. Bxf7+ by you" in t]
    takeback = [t for t in texts if t.startswith("Takeback: 4. Bxf7+ withdrawn")]
    replacement = [t for t in texts if "Move: 4. Qxf7# by you" in t]
    assert withdrawn and takeback and replacement
    assert texts.index(withdrawn[0]) < texts.index(takeback[0]) < texts.index(replacement[0])
    assert "back at the position after 3... Nf6" in takeback[0]
    # The move list in the replacement's message no longer holds the withdrawn move.
    assert "Moves so far: 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#" in replacement[0]
    assert "Bxf7+" not in replacement[0].split("Moves so far:")[1].split("\n")[0]


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
    texts = [m["content"] for m in _given(provider)]
    reread = [t for t in texts if t.startswith("The board was re-read")]
    assert reread and "Moves may have gone unseen" in reread[0]
    # The numbering follows the position the board was read at, not a ply count.
    after = [t for t in texts if "Move: 4... d6 by your opponent" in t]
    assert after and "Moves so far: 4... d6" in after[0]


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
    sheet = fact_sheet(facts, DisclosureLevel.QUESTION)
    assert "Black (your opponent) to move; you are white" in sheet
    assert "Material: you are up a pawn." in coach.conversation.render_ply(
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
    last_own = [m["content"] for m in messages if "Move: 3. Qxe5+ by you" in m["content"]][0]
    assert "Your moves so far (3 judged): 1 best, 1 book, 1 inaccuracy." in last_own
    assert "Ground lost at: 2. Qh5 (inaccuracy)" in last_own
    assert any(m["content"].startswith("Game over: 1-0") for m in messages)
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
        "1 of your 3 moves were best or excellent. You lost ground at 2. Qh5 (inaccuracy)."
    )


def test_a_quick_answer_is_a_turn_before_the_model_and_a_failed_reply_adds_nothing() -> None:
    # The model's two candidates both name the mating move at the QUESTION level: rejected
    # twice. With the quick answer already out, nothing is said in its place -- the same
    # line a second time was what the first rendered example showed.
    coach, provider = _coach(["[p=0.5] Play Qxf7#.\n[p=0.5] Qxf7 mates.", "[p=1.0] Qxf7#!"])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    coach.ask("hint", 8.0)
    out = _messages(coach)
    quick = "Something forcing is on: is there a check or a capture worth calculating first?"
    assert [m.text for m in out] == [quick]
    # The request carried the quick answer as the coach's own turn, then the directive.
    given = _given(provider, 0)
    assert given[-3]["content"] == 'The player says: "hint"'
    assert given[-2]["role"] == "assistant" and given[-2]["content"].startswith("Something forcing")
    assert given[-1]["role"] == "user" and "build on it, do not repeat it" in given[-1]["content"]
    assert "The player says" not in given[-1]["content"]
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
