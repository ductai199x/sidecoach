from collections.abc import Sequence

import chess
import pytest

from sidecoach.coach.coach import Coach, _parse_candidates, _sample_reply, complete_sentences
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
from tests.coaching import drain, mate_analysis, play_to_scholar
from tests.fakes import ScriptedProvider

CUT_MID_MARKER = (
    "[p=0.5] Placing it on c5 pins the piece. What would that gain?\n\n"
    "[p=0.3] Consider the long-term implications. How might a pawn move help?\n\n[p=0.2"
)


@pytest.mark.parametrize(
    ("raw", "texts", "weights"),
    [
        pytest.param(
            "[p=0.5] First reply here.\n[p=0.3] Second reply,\nwrapping on.\n[p=0.2] Third.",
            ["First reply here.", "Second reply,\nwrapping on.", "Third."],
            [0.5, 0.3, 0.2],
            id="weighted-and-a-wrapped-reply-kept-whole",
        ),
        # A reply with no markers is one candidate, so a plain model still works.
        pytest.param("Just one plain reply.", ["Just one plain reply."], [1.0], id="plain"),
        pytest.param("<think>hidden</think>\n[p=1] Clean.", ["Clean."], [1.0], id="think-block"),
        # The token budget can end the output inside the next candidate's marker: the fragment
        # never reaches the user (it did: "...move?\n\n[p=0.2").
        pytest.param(
            CUT_MID_MARKER,
            [
                "Placing it on c5 pins the piece. What would that gain?",
                "Consider the long-term implications. How might a pawn move help?",
            ],
            None,
            id="cut-mid-marker",
        ),
        pytest.param(
            "Think about the e-file. What does it need?\n\n[p=",
            ["Think about the e-file. What does it need?"],
            None,
            id="a-lone-reply-cut-mid-marker",
        ),
        # The first live game spoke "Given" and "The key was using": stumps of replies the
        # budget cut short that happened to be the only candidates passing every check.
        pytest.param(
            "[p=0.5] Fine move. What is the plan?\n[p=0.3] Given",
            ["Fine move. What is the plan?"],
            None,
            id="a-stump-dropped",
        ),
        pytest.param("The key was using", [], None, id="only-a-stump"),
    ],
)
def test_a_reply_parses_into_weighted_candidates_and_never_a_fragment(
    raw: str, texts: list[str], weights: list[float] | None
) -> None:
    candidates = _parse_candidates(raw)
    assert [text for _, text in candidates] == texts
    if weights is not None:
        assert [round(p, 2) for p, _ in candidates] == weights
    if len(candidates) == 1:
        assert _sample_reply(candidates) == texts[0], "one candidate: no draw"


@pytest.mark.parametrize(
    ("text", "kept"),
    [
        ("Your idea is spot-on. Given", "Your idea is spot-on."),
        ("Given", ""),
        ("What now? Reflect", "What now?"),
        ('He said "go." Then', 'He said "go."'),
        ("Look at e5 (the pawn!)", "Look at e5 (the pawn!)"),
    ],
)
def test_a_reply_cut_off_mid_sentence_keeps_its_last_full_sentence(text: str, kept: str) -> None:
    assert complete_sentences(text) == kept


def test_the_clean_candidate_is_kept_at_once_and_a_stump_never_reaches_the_user() -> None:
    """The model offers a false reply and a clean one in one distribution: the clean one is
    kept on the first attempt, no retry, no template. A stump passes the checks trivially;
    it must not win by default."""

    def checker(
        text: str,
        board: chess.Board,
        after: Sequence[chess.Board] = (),
        us: chess.Color | None = None,
    ) -> list[str]:
        return ["the rook on f7 is not there"] if "rook on f7" in text else []

    def coach_on(replies: list[str]) -> Coach:
        coach = Coach(
            ScriptedProvider(replies),
            policy=InterjectionPolicy(PolicySettings(stuck_seconds=10)),
            draft_check=checker,
        )
        play_to_scholar(coach)
        coach.on_event(mate_analysis(6), 7.0)
        return coach

    coach = coach_on(
        [
            "[p=0.6] Your rook on f7 is strong.\n"
            "[p=0.4] Their king on e8 is short of squares, is it not?"
        ]
    )
    assert drain(coach)[0].text == "Their king on e8 is short of squares, is it not?"
    assert coach.spoken[-1].attempts == 1 and coach.speaker.fallbacks == 0
    first = "[p=0.6] Your rook on f7 is strong. What next?\n[p=0.4] Given"
    second = "[p=0.7] Their king on e8 is short of squares. What is their threat?"
    coach = coach_on([first, second])
    assert drain(coach)[0].text == "Their king on e8 is short of squares. What is their threat?"
    spoken = coach.spoken[-1]
    # The false claim is sent back and the second draft spoken: no template.
    assert spoken.attempts == 2 and coach.speaker.fallbacks == 0
    # Both calls are kept for the log: what the model wrote and how the candidates fared.
    assert [call.raw for call in spoken.calls] == [first, second]
    assert [(call.candidates, call.passed) for call in spoken.calls] == [(1, 0), (1, 1)]


def test_a_verifier_keeps_the_clean_candidate_and_repairs_when_none_is() -> None:
    """With a verifier, a candidate the second model finds unsupported is not spoken; when
    every candidate has an unsupported statement, the likeliest is repaired once."""
    from sidecoach.coach.coach import Speaker, _Request
    from sidecoach.coach.disclosure import PositionFacts
    from sidecoach.events import DisclosureLevel, Trigger

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

    coach_model = ScriptedProvider(
        ["[p=0.6] Nf3 develops and eyes e5.\n[p=0.4] Nf3 develops a piece."]
    )
    # The first candidate invents "eyes e5"; the second is clean.
    verifier = ScriptedProvider(['{"unsupported": ["eyes e5"]}', '{"unsupported": []}'])
    speaker = Speaker(coach_model, verifier=verifier)  # type: ignore[arg-type]
    spoken = speaker.speak_now(_request())
    assert spoken.message.text == "Nf3 develops a piece."
    assert spoken.verified == [["eyes e5"], []] and not spoken.repaired

    # Both candidates unsupported: the likelier one is repaired.
    coach_model = ScriptedProvider(
        ["[p=0.7] Nf3 develops and eyes e5.\n[p=0.3] Nf3 pins the knight."]
    )
    verifier = ScriptedProvider(
        [
            '{"unsupported": ["eyes e5"]}',
            '{"unsupported": ["pins the knight"]}',
            '{"reply": "Nf3 develops a piece."}',
        ]
    )
    speaker = Speaker(coach_model, verifier=verifier)  # type: ignore[arg-type]
    spoken = speaker.speak_now(_request())
    assert spoken.message.text == "Nf3 develops a piece." and spoken.repaired
