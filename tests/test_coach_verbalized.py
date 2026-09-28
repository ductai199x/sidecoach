def test_a_reply_cut_off_mid_marker_leaks_no_marker_fragment() -> None:
    """The token budget can end the model's output in the middle of the next candidate's
    marker; that fragment must never reach the user (it did: "...move?\n\n[p=0.2")."""
    from sidecoach.coach.coach import _parse_candidates

    raw = (
        "[p=0.5] Placing it on c5 pins the piece. What would that gain?\n\n"
        "[p=0.3] Consider the long-term implications. How might a pawn move help?\n\n[p=0.2"
    )
    out = _parse_candidates(raw)
    assert [t for _, t in out] == [
        "Placing it on c5 pins the piece. What would that gain?",
        "Consider the long-term implications. How might a pawn move help?",
    ]
    # A lone reply with no proper markers, cut the same way, is cleaned too.
    assert _parse_candidates("Think about the e-file. What does it need?\n\n[p=")[0][1] == (
        "Think about the e-file. What does it need?"
    )


def test_a_reply_cut_off_mid_sentence_is_trimmed_to_its_last_full_sentence() -> None:
    """The first live game spoke "Given" and "The key was using": stumps of replies the
    budget cut short that happened to be the only candidates passing every check."""
    from sidecoach.coach.coach import _parse_candidates, complete_sentences

    assert complete_sentences("Your idea is spot-on. Given") == "Your idea is spot-on."
    assert complete_sentences("Given") == ""
    assert complete_sentences("What now? Reflect") == "What now?"
    assert complete_sentences('He said "go." Then') == 'He said "go."'
    assert complete_sentences("Look at e5 (the pawn!)") == "Look at e5 (the pawn!)"
    raw = "[p=0.5] Fine move. What is the plan?\n[p=0.3] Given"
    assert [t for _, t in _parse_candidates(raw)] == ["Fine move. What is the plan?"]
    assert _parse_candidates("[p=0.5] The key was using") == []
    assert _parse_candidates("The key was using") == []


def test_a_stump_never_reaches_the_user_even_when_it_is_the_only_clean_candidate() -> None:
    """A stump passes the leak check trivially; it must not win by default."""
    from collections.abc import Sequence

    import chess

    from sidecoach.coach.coach import Coach
    from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
    from tests.test_coach import ScriptedProvider, _mate_analysis, _messages, _play_to_scholar

    def checker(
        text: str,
        board: chess.Board,
        after: Sequence[chess.Board] = (),
        us: chess.Color | None = None,
    ) -> list[str]:
        return ["the rook on f7 is not there"] if "rook on f7" in text else []

    first = "[p=0.6] Your rook on f7 is strong. What next?\n[p=0.4] Given"
    second = "[p=0.7] Their king on e8 is short of squares. What is their threat?"
    provider = ScriptedProvider([first, second])
    coach = Coach(
        provider, policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)), draft_check=checker
    )
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    out = _messages(coach)
    assert out[0].text == "Their king on e8 is short of squares. What is their threat?"
    spoken = coach.spoken[-1]
    # The false claim is sent back and the second draft spoken: no template.
    assert spoken.attempts == 2 and coach.speaker.fallbacks == 0
    # Both calls are kept for the log: what the model wrote and how the candidates fared.
    assert [call.raw for call in spoken.calls] == [first, second]
    assert [(call.candidates, call.passed) for call in spoken.calls] == [(1, 0), (1, 1)]


def test_a_bare_hint_is_asked_for_as_a_hint_not_as_a_claim() -> None:
    """ "hint" carries no claim; prompted to be honest about "their claim", the model answered
    its own previous message as though the player had said it ("Your suggestion to focus on
    opening lines is spot-on")."""
    from sidecoach.coach.coach import _ASK_HINT
    from tests.test_coach import _coach, _mate_analysis, _messages, _play_to_scholar

    for said in ("hint", "Hint", "give me a hint", "a clue please", "help", "any tips?"):
        assert _ASK_HINT.match(said), said
    for said in ("what should I play", "is Nf3 a good idea", "help me with the knight"):
        assert not _ASK_HINT.match(said), said
    coach, provider = _coach(["Where is their king weakest?"])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    coach.hint(8.0)
    _messages(coach)
    prompt = provider.prompts[0][1][-1]["content"]
    assert "They asked for a hint" in prompt
    assert "their own claim" not in prompt and "anything said earlier" in prompt


def test_a_verifier_keeps_the_clean_candidate_and_repairs_when_none_is() -> None:
    """With a verifier, a candidate the second model finds unsupported is not spoken; when
    every candidate has an unsupported statement, the likeliest is repaired once."""
    import chess

    from sidecoach.coach.coach import Speaker, _Request
    from sidecoach.coach.disclosure import PositionFacts
    from sidecoach.events import DisclosureLevel, Trigger
    from sidecoach.llm.base import Completion
    from tests.test_coach import ScriptedProvider

    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")

    def _request() -> _Request:
        return _Request(
            trigger=Trigger.USER_ASK,
            level=DisclosureLevel.LINE,
            system="You are a coach.",
            messages=[
                {"role": "user", "content": "Moves so far: 1. e4 e5 -- White to move."},
                {"role": "user", "content": "Answer them."},
            ],
            facts=PositionFacts(
                fen=board.fen(),
                side_to_move="white",
                user_side="white",
                phase="opening",
                candidates=["Nf3"],
            ),
            board=board,
            fallback="Think.",
            ply=2,
            game_id="g",
            candidates=2,
        )

    class Verifier:
        name = model = "scripted-verifier"

        def __init__(self, answers: list[str]) -> None:
            self.answers = answers
            self.seen: list[str] = []

        def healthcheck(self):
            return True, "ok"

        def chat(self, system, messages, **kw):
            self.seen.append(messages[-1]["content"])
            return Completion(text=self.answers.pop(0))

    coach_model = ScriptedProvider(
        ["[p=0.6] Nf3 develops and eyes e5.\n[p=0.4] Nf3 develops a piece."]
    )
    # The first candidate invents "eyes e5"; the second is clean.
    verifier = Verifier(['{"unsupported": ["eyes e5"]}', '{"unsupported": []}'])
    speaker = Speaker(coach_model, verifier=verifier)  # type: ignore[arg-type]
    spoken = speaker.speak_now(_request())
    assert spoken.message.text == "Nf3 develops a piece."
    assert spoken.verified == [["eyes e5"], []] and not spoken.repaired

    # Both candidates unsupported: the likelier one is repaired.
    coach_model = ScriptedProvider(
        ["[p=0.7] Nf3 develops and eyes e5.\n[p=0.3] Nf3 pins the knight."]
    )
    verifier = Verifier(
        [
            '{"unsupported": ["eyes e5"]}',
            '{"unsupported": ["pins the knight"]}',
            '{"reply": "Nf3 develops a piece."}',
        ]
    )
    speaker = Speaker(coach_model, verifier=verifier)  # type: ignore[arg-type]
    spoken = speaker.speak_now(_request())
    assert spoken.message.text == "Nf3 develops a piece." and spoken.repaired
