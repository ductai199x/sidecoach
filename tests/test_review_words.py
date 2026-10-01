"""The model's review of a game's key moves (coach/review_words.py): which moves it covers, the
facts in colours, the student's words made from colours, and every check a part must pass -- the
board's, the side a move is given to, the side named but by its colour, the answer kept back --
on a short game built by hand and a model that is not one."""

from __future__ import annotations

import json
from typing import Any

import chess
import pytest

from sidecoach.coach import review_words
from sidecoach.events import (
    Analysis,
    Label,
    Line,
    MoveAssessment,
    ReviewedGame,
    ReviewedMove,
    Severity,
    Side,
)
from sidecoach.llm.base import Completion
from sidecoach.llm.usage import Prices


def _game() -> ReviewedGame:
    """1. e4 e5 2. Nf3 Nc6 3. Bc4 Nd4: White's Nf3 a mistake (Bc4 was best), Black's Nd4 a
    blunder (Nf6 best), Black's Nc6 great."""
    moves = [
        ("e4", None, []),
        ("e5", None, []),
        ("Nf3", Label.MISTAKE, ["f1c4"]),
        ("Nc6", Label.GREAT, ["b8c6"]),
        ("Bc4", None, []),
        ("Nd4", Label.BLUNDER, ["g8f6"]),
    ]
    board = chess.Board()
    start = Analysis(
        game_id="g",
        fen=board.fen(),
        ply=0,
        decisive=False,
        only_move=False,
        lines=[Line(moves=["e2e4"], depth=18, expected_score=0.5)],
    )
    out = []
    for ply, (san, label, best) in enumerate(moves, start=1):
        move = board.parse_san(san)
        side = Side.WHITE if board.turn == chess.WHITE else Side.BLACK
        before = board.fen()
        board.push(move)
        reply = next(iter(board.legal_moves)).uci()
        after = Analysis(
            game_id="g",
            fen=board.fen(),
            ply=ply,
            decisive=False,
            only_move=False,
            lines=[Line(moves=[reply], depth=18, expected_score=0.5)],
        )
        out.append(
            ReviewedMove(
                ply=ply,
                san=san,
                side=side,
                fen_before=before,
                after=after,
                assessment=MoveAssessment(
                    game_id="g",
                    ply=ply,
                    uci=move.uci(),
                    severity=Severity.NONE,
                    expected_loss=0.2 if label in (Label.MISTAKE, Label.BLUNDER) else 0.0,
                    best=best,
                    label=label,
                    accuracy=50.0,
                ),
            )
        )
    return ReviewedGame(
        headers={"Result": "1-0"}, start=start, moves=out, strength="standard", depth=18
    )


class _Model:
    """A model that answers each key move with the parts given for its ply (a ply given an
    exception fails), and counts what it was asked."""

    name = model = "fake"

    def __init__(self, answers: dict[int, dict[str, str] | Exception]) -> None:
        self.answers = answers
        self.asked: list[int] = []

    def healthcheck(self) -> tuple[bool, str]:
        return True, "fake"

    def chat(self, system: str, messages: Any, **_: Any) -> Completion:
        ply = json.loads(messages[0]["content"])["key move"]["ply"]
        self.asked.append(ply)
        answer = self.answers[ply]
        if isinstance(answer, Exception):
            raise answer
        text = "\n".join(f"### {part}\n{words}" for part, words in answer.items())
        return Completion(text=text, stats={"prompt_tokens": 1000, "completion_tokens": 300})


def test_the_key_moves_are_the_reviews_own_dearest_first() -> None:
    game = _game()
    assert review_words.key_plies(game) == [3, 4, 6]
    review_words.MOST, most = 2, review_words.MOST
    try:
        # Two at most: the blunder and the great move before the mistake, in the game's order.
        assert review_words.key_plies(game) == [4, 6]
    finally:
        review_words.MOST = most


def test_the_facts_are_told_in_colours() -> None:
    """The coach's facts speak to the side to move; the model is told them in colours, the
    student's moves marked, and a bad move comes with its better move outright."""
    fact = {"piece": "your bishop on c4", "key": "If you let them, their best is Nxe5"}
    assert review_words.in_colours(fact, "White") == {
        "piece": "White's bishop on c4",
        "key": "If White let Black, Black's best is Nxe5",
    }
    game = _game()
    told = json.loads(review_words.prompt(game, 3, "black", None))
    assert told["game"]["the student plays"] == "Black"
    move = told["key move"]
    assert move["played by"] == "White"
    assert move["what it does"] == "White's knight from g1 to f3"
    assert move["White's better move"] == "Bc4", "never to be mixed up with the reply"
    position = move["the position before it"]
    facts = json.dumps(position).lower()
    assert " your " not in facts and " their " not in facts
    # Whose each fact is, named: the threats are the other side's (luna gave White's to Black).
    assert not {"threats", "candidates", "line"} & set(position)
    assert review_words.in_colours("Bf5 and Bd6 among them", "Black") == "Bf5 and Bd6 among them"
    assert json.loads(review_words.prompt(game, 4, "black", None))["key move"]["played by"] == (
        "Black (the student)"
    )


# The student's colour, a sentence of the model's in colours, and what the student reads.
STUDENT_READS: list[tuple[str | None, str, str]] = [
    (
        "black",
        "White's bishop on c4 eyes f7, so Black must defend.",
        "Your opponent's bishop on c4 eyes f7, so you must defend.",
    ),
    (
        "black",
        "Black has a strong knight, and Black also threatens Nd4.",
        "You have a strong knight, and you also threaten Nd4.",
    ),
    (
        "black",
        "Black is a pawn up; White was careless.",
        "You are a pawn up; your opponent was careless.",
    ),
    # Not a subject: the word after it is no verb of its.
    ("black", "This gives Black chances.", "This gives you chances."),
    (
        "black",
        "The Black king is safe; the white king is not, and White pieces hang.",
        # Said again in a sentence, "your opponent's" is "their".
        "Your king is safe; your opponent's king is not, and their pieces hang.",
    ),
    (
        "white",
        "If Black takes, White recaptures and White doesn't lose time.",
        "If your opponent takes, you recapture and you don't lose time.",
    ),
    (
        "white",
        "White's winning chances fall because Black's knight reaches e4.",
        "Your winning chances fall because your opponent's knight reaches e4.",
    ),
    # A side the model named "they": the one its sentence named on its own, not the owner.
    ("black", "Black can reorganize their pieces.", "You can reorganize your pieces."),
    (
        "black",
        "White must meet Black's threat before pursuing their own plans.",
        "Your opponent must meet your threat before pursuing their own plans.",
    ),
    (
        "black",
        "Black's rooks and their files are open.",
        "Your rooks and their files are open.",
    ),
    # "Them" of pieces stays theirs (luna, 2026-09-30).
    (
        "black",
        "White forks the queen and the pawn. Black can save only one of them.",
        "Your opponent forks the queen and the pawn. You can save only one of them.",
    ),
    (
        "black",
        "Black's king and rook are not as active as they could be.",
        "Your king and rook are not as active as they could be.",
    ),
    # Luna's British plural for a side, singular for "your opponent".
    (
        "black",
        "White are down a queen, and White need to act.",
        "Your opponent is down a queen, and your opponent needs to act.",
    ),
    # A verb joined to the student's agrees too; one of another subject does not.
    (
        "black",
        "Black blocks the file and leaves White's queen safe.",
        "You block the file and leave your opponent's queen safe.",
    ),
    (
        "black",
        "Black neutralizes the threat and stabilizes the position.",
        "You neutralize the threat and stabilize the position.",
    ),
    (
        "black",
        "Black not only escapes the check but also solidifies the defense.",
        "You not only escape the check but also solidify the defense.",
    ),
    (
        "black",
        "This allows Black to regroup and opens lines.",
        "This allows you to regroup and opens lines.",
    ),
    (
        "black",
        "Black first improves the rook, waits for the reply, and then uses it.",
        "You first improve the rook, wait for the reply, and then use it.",
    ),
    (
        "black",
        "White makes a threat Black cannot meet, then the knight reaches g6 and attacks.",
        "Your opponent makes a threat you cannot meet, then the knight reaches g6 and attacks.",
    ),
    # Governed by a preposition, a side is no subject; a verb known as one agrees anywhere.
    ("black", "The plan for Black is to push.", "The plan for you is to push."),
    (
        "black",
        "The position is forcing, meaning Black has no quiet moves.",
        "The position is forcing, meaning you have no quiet moves.",
    ),
    ("black", "White forces Black to defend.", "Your opponent forces you to defend."),
    (
        "black",
        "A clear win for Black, who is up a queen.",
        "A clear win for you, who are up a queen.",
    ),
    (None, "Black has a strong knight.", "Black has a strong knight."),
]


def test_the_student_reads_you_and_your_opponent() -> None:
    wrong = [
        (user, said, got, read)
        for user, said, read in STUDENT_READS
        if (got := review_words.to_student(said, user)) != read
    ]
    assert not wrong, "\n".join(f"{u} {s!r}: read {g!r}, wanted {r!r}" for u, s, g, r in wrong)


def test_a_side_named_but_by_its_colour_is_dropped() -> None:
    """In colours, "your" could be either side: the part goes, and with "why" the whole move."""
    game = _game()
    kept, dropped, _ = review_words.checked(
        {"why": "Your knight on c6 guards e5.", "lesson": "Guard your centre."}, game, 4, "black"
    )
    assert (kept, dropped) == ({}, 'why: names a side "Your"')


def test_a_piece_given_to_the_wrong_side_is_dropped() -> None:
    game = _game()  # the pawn on f2 is White's; a pawn named by its file is checked too
    kept, dropped, _ = review_words.checked(
        {"why": "Black's isolated f-pawn on f2 is weak."}, game, 4, "black"
    )
    assert kept == {}
    assert dropped.startswith("why: Black's isolated f-pawn on f2: the pawn on f2 is white's")


def test_a_move_given_to_the_wrong_side_is_dropped() -> None:
    game = _game()  # Nf3 (ply 3) and its better move Bc4 are White's
    assert review_words.wrong_side("Black played Nf3 too early.", game, 3)
    assert review_words.wrong_side("Black could have played Bc4.", game, 3)
    assert not review_words.wrong_side("White played Nf3 too early.", game, 3)
    assert not review_words.wrong_side("White's Bc4 develops first.", game, 3)
    # Named without a side, or as a threat of the other's, a move is no one's in particular.
    assert not review_words.wrong_side("Black must stop Bc4.", game, 3)


def test_the_better_move_waits_for_the_answer() -> None:
    """On the student's own bad move, a "why" giving the better move is kept for when the answer
    is out; on the opponent's, the better move is not the student's to find."""
    game = _game()
    parts = {"why": "Bc4 was the move.", "better": "Bc4 develops first and eyes f7."}
    kept, dropped, reveals = review_words.checked(parts, game, 3, "white")
    assert (kept, dropped, reveals) == (parts, "", True)
    assert review_words.checked(parts, game, 3, "black")[2] is False
    assert review_words.names_the_answer("Bc4 was the move.", game, 3), "side unknown: kept back"
    assert not review_words.names_the_answer("Developing the bishop first was better.", game, 3)
    assert not review_words.names_the_answer("Nc6 defends e5.", game, 4), "a great move is named"


def test_a_lesson_is_general_advice() -> None:
    """The lesson alone speaks to "you"; one naming a side or a square goes, the rest stays. A
    good move has no "better"."""
    game = _game()
    why = "Black's knight on c6 guards Black's pawn on e5."
    general = "Before you chase a piece, see what your opponent threatens."
    kept, dropped, _ = review_words.checked(
        {"why": why, "better": "x", "lesson": general}, game, 4, "black"
    )
    assert kept == {"why": "Your knight on c6 guards your pawn on e5.", "lesson": general}
    assert dropped == ""
    kept, dropped, _ = review_words.checked(
        {"why": why, "lesson": "Black should guard e5 first."}, game, 4, "black"
    )
    assert (list(kept), dropped) == (["why"], 'lesson: names "Black"')


def test_the_answer_is_read_by_its_heads() -> None:
    assert review_words.parse("### why\nA\nand B\n### Better:\nC\n```\n### lesson\nD\n```") == {
        "why": "A and B",
        "better": "C",
        "lesson": "D",
    }
    assert review_words.parse("no heads here") == {}
    assert review_words.parse("### better\nC") == {}, "a move without its why has nothing"


def test_the_review_is_a_call_a_move_checked_and_costed() -> None:
    game = _game()
    lesson = "Develop before you commit your knights."
    model = _Model(
        {
            3: {
                "why": "White played Nf3 before the bishop came out, and Black's pawn on e5 holds.",
                "better": "Bc4 first.",
                "lesson": lesson,
            },
            4: {"why": "Black's knight on c6 guards Black's pawn on e5."},
            # Wrong: the knight on c6 is Black's.
            6: {"why": "White's knight on c6 is loose."},
        }
    )
    words = review_words.write(
        game, model, user="black", opening=None, prices=Prices(0.20, 0.02, 1.20)
    )
    assert words.texts == {
        3: {
            "why": "Your opponent played Nf3 before the bishop came out, and your pawn on e5 "
            "holds.",
            "better": "Bc4 first.",
            "lesson": lesson,
        },
        4: {"why": "Your knight on c6 guards your pawn on e5."},
    }
    assert set(words.dropped) == {6} and words.reveals == set()
    assert sorted(model.asked) == [3, 4, 6], "one call a key move"
    assert words.cost is not None and abs(words.cost - 3 * (1000 * 0.20 + 300 * 1.20) / 1e6) < 1e-9


def test_answers_already_paid_for_are_not_asked_again() -> None:
    """The app was closed in the middle of the words: the answers paid for were kept, the words
    go on with the moves left, each new answer handed back as it arrives (to keep), and the
    cost and the estimate are of the new calls only."""
    game = _game()
    model = _Model(
        {
            3: {"why": "White played Nf3 early."},
            4: {"why": "Black's knight on c6 guards Black's pawn on e5."},
            6: {"why": "Black's knight left e5 alone."},
        }
    )
    got: dict[int, str] = {}
    prices = Prices(0.20, 0.02, 1.20)
    words = review_words.write(
        game,
        model,
        user="black",
        opening=None,
        prices=prices,
        answered={3: "### why\nWhite played Nf3 early."},
        on_answer=lambda ply, text, _stats: got.__setitem__(ply, text),
    )
    assert sorted(model.asked) == [4, 6] and set(got) == {4, 6}
    assert set(words.texts) == {3, 4, 6}
    assert words.cost is not None and abs(words.cost - 2 * (1000 * 0.20 + 300 * 1.20) / 1e6) < 1e-9
    whole = review_words.estimate(game, "black", None, prices)
    rest = review_words.estimate(game, "black", None, prices, frozenset({3}))
    assert whole is not None and rest is not None and 0 < rest < whole
    # Every move answered before: nothing asked.
    none = _Model({})
    done = review_words.write(
        game,
        none,
        user="black",
        opening=None,
        prices=prices,
        answered={p: "### why\nWhite played Nf3 early." for p in (3, 4, 6)},
    )
    assert none.asked == [] and set(done.texts) == {3, 4, 6}


def test_a_failed_call_drops_its_move_alone() -> None:
    game = _game()
    answers: dict[int, dict[str, str] | Exception] = {
        3: {"why": "White played Nf3 early."},
        4: RuntimeError("429: slow down"),
        6: {"why": "Black's knight left e5 alone."},
    }
    words = review_words.write(game, _Model(answers), user="black", opening=None, prices=Prices())
    assert set(words.texts) == {3, 6}
    assert words.dropped == {4: "the call failed: 429: slow down"}
    with pytest.raises(RuntimeError, match="down"):
        review_words.write(
            game,
            _Model(dict.fromkeys((3, 4, 6), RuntimeError("down"))),
            user="black",
            opening=None,
            prices=Prices(),
        )
