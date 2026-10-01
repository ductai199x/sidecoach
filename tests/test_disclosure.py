import pytest

from sidecoach.coach.disclosure import (
    Disclosure,
    PositionFacts,
    Unlock,
    leak_check,
    visible_facts,
)
from sidecoach.events import DisclosureLevel as L

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def facts() -> PositionFacts:
    return PositionFacts(
        fen=START,
        side_to_move="white",
        phase="opening",
        theme="loose piece",
        region="around f7",
        piece="your knight",
        candidates=["Ng5"],
        threats=["Qxd4"],
        line=["Ng5", "d5", "Nxf7"],
    )


@pytest.mark.parametrize(
    ("start", "unlocks", "levels"),
    [
        # Unprompted: the idea, then the focus, and never the move.
        pytest.param(
            L.SILENT, [Unlock.STUCK, Unlock.STUCK, Unlock.STUCK], [L.IDEA, L.FOCUS, None],
            id="unprompted-stops-at-the-focus",
        ),
        # Two hints, then the answer: the third hint is the move, explained.
        pytest.param(
            L.SILENT, [Unlock.HINT] * 4, [L.IDEA, L.FOCUS, L.MOVE, None],
            id="two-hints-then-the-move",
        ),
        pytest.param(L.SILENT, [Unlock.ASK_MOVE] * 2, [L.MOVE, None], id="asking-for-the-move"),
        pytest.param(L.IDEA, [Unlock.USER_NAMED_MOVE], [L.MOVE], id="naming-the-move"),
        pytest.param(L.SILENT, [Unlock.MOVE_PLAYED], [L.LINE], id="playing-gives-the-line"),
    ],
)  # fmt: skip
def test_the_rungs_climb_by_what_unlocks_them(
    start: L, unlocks: list[Unlock], levels: list[L | None]
) -> None:
    d = Disclosure(level=start)
    assert [d.apply(unlock) for unlock in unlocks] == levels
    assert d.level == [level for level in levels if level is not None][-1]


def test_visible_facts_hide_moves_below_the_move_level_and_show_all_at_the_line() -> None:
    idea = visible_facts(facts(), L.IDEA)
    assert idea["theme"] == "loose piece" and "fen" in idea
    for hidden in ("region", "piece", "candidates", "line"):
        assert hidden not in idea
    # Their threat is the first hint's key fact as often as not: words from the idea level.
    assert idea["threats"] == ["Qxd4"]
    focus = visible_facts(facts(), L.FOCUS)
    assert focus["region"] == "around f7" and focus["piece"] == "your knight"
    for hidden in ("candidates", "line"):
        assert hidden not in focus
    line = visible_facts(facts(), L.LINE)
    assert line["line"] == ["Ng5", "d5", "Nxf7"] and line["candidates"] == ["Ng5"]


@pytest.mark.parametrize(
    ("text", "level", "leaks"),
    [
        # Tai, 2026-09-25: "leaking is not a criminal offense ... clamp down egregious offense
        # but be smart about it". Below the move, the moves to find and the squares they go to
        # are withheld; a plan, a weakness, another move or square may be named.
        ("Have you considered Ng5, hitting f7?", L.IDEA, ["move Ng5", "square f7"]),
        ("A knight on g5 would be strong.", L.IDEA, ["square g5"]),
        ("Maybe d3 or Nc3 here, then a plan on the queenside?", L.IDEA, []),
        # The player's later move in the line is theirs to find too (the parity audit,
        # 2026-09-29: the phone held this case, the desktop did not).
        ("Try Ng5, then Nxf7.", L.IDEA, ["move Ng5", "move Nxf7"]),
        # The opponent's move in the line is not the player's answer.
        ("If they push d5, the centre opens.", L.IDEA, []),
        # At the move level the move may be said, the rest of the line not.
        ("Ng5 is the idea.", L.MOVE, []),
        ("After Ng5 d5 Nxf7 you win.", L.MOVE, ["square d5", "move Nxf7"]),
        # The focus level names the region, and the square its words anchor on.
        ("Look at what is happening around f7.", L.FOCUS, []),
        # The piece to move is the focus level's.
        ("What is your knight doing?", L.IDEA, ["piece knight"]),
        ("What is your knight doing?", L.FOCUS, []),
    ],
)
def test_leak_check_withholds_the_answer_and_nothing_else(
    text: str, level: L, leaks: list[str]
) -> None:
    found = leak_check(text, facts(), level)
    if leaks:
        assert set(leaks) <= set(found), found
    else:
        assert found == []


def test_leak_check_allows_squares_and_pieces_the_user_named() -> None:
    # The user asked about f7 and the knight; confirming them back is not a leak.
    said = "Is my bishop on c4 attacking f7? And what is my knight doing?"
    assert leak_check("Yes, your bishop hits f7.", facts(), L.IDEA, said=said) == []
    assert leak_check("Your knight is well placed.", facts(), L.IDEA, said=said) == []
    # An answer square the user did not name is still withheld.
    assert "square g5" in leak_check("Look at g5.", facts(), L.IDEA, said=said)


def test_a_move_of_the_games_record_is_no_leak() -> None:
    """A move said with its number is the record: the review names "17. Ng5" where ground was
    lost (the eval audit's gate, 2026-09-25). Only a move the game played: "5. cxd5 wins it
    cleanly" of the move to find was let through by any number at all (the second audit)."""
    record = ["17. Ng5"]
    said = "You lost the most ground at 17. Ng5 (mistake)."
    assert leak_check(said, facts(), L.IDEA, record=record) == []
    assert "move Ng5" in leak_check("5. Ng5 wins it cleanly.", facts(), L.IDEA, record=record)
    assert "move Ng5" in leak_check(said, facts(), L.IDEA)


# Below the focus level the piece to move is withheld, and only it: every other word for its
# kind may be said. (the answer, the piece to move, the first hint (the sheet's situation),
# replies that are no leak, replies that name the piece to move.)
PIECE_WORDS = [
    # A pawn on another square than the one to move is a fact, not the hint (the eval audit's
    # gate, 2026-09-25); round 2 refused four good replies for "the pawn on c3", "your pawn on
    # h2". The one that takes, or goes, names it: "take it with a pawn", "the c-pawn should
    # take it" (the second audit).
    pytest.param(
        "cxd5", "the player's pawn on c4", "Their pawn on d5 can simply be taken.",
        [
            "Their pawn on d5 can be taken; they threaten dxc3, taking the pawn on c3.",
            "Ng4 is coming, attacking your pawn on h2.",
        ],
        [
            "Take it with a pawn, not a piece.", "The c-pawn should take it.", "Use your pawn.",
            "Your pawn on c4 is the one.",
        ],
        id="the-pawn-that-takes-not-every-pawn",
    ),
    # The first hint may name the target's square even where the move goes; the piece that
    # goes there is then the answer ("take it with the queen": the second audit).
    pytest.param(
        "Qxf7", "the player's queen on h5", "Their pawn on f7 is the target.",
        ["Their pawn on f7 is weak."], ["Their pawn on f7 is weak: take it with the queen."],
        id="the-answers-square-then-its-piece",
    ),
    # Asked for the threat, the coach says the move and what it does, as the sheet does:
    # "g5, attacking your queen on c3" was refused while the queen was the piece to move (the
    # smoke run, 2026-09-25). Naming the queen as the one to use is still the hint.
    pytest.param(
        "Qd2", "the player's queen on c3", None,
        [
            "They threaten b4, attacking your queen on c3.",
            "b4 is coming: it attacks your queen.",
            "Their b4 threatens your queen.",
            "They threaten Nd5, winning your queen on c3.",
        ],
        ["Move your queen.", "Take your queen to safety.", "Attack with your queen."],
        id="what-their-threat-goes-after",
    ),
    # The first hint says "the first move takes your bishop out of danger": a paraphrase of it
    # was refused and the bare template spoke (the benchmark's smoke run, 2026-09-25). What
    # the sheet says at this level may be said in other words; the piece is still withheld
    # where the sheet never names it.
    pytest.param(
        "Bd3", "the player's bishop on b5",
        "There's a tactic here that wins a pawn. The first move takes your bishop out of danger.",
        ["Your first move takes your bishop out of danger, and it wins a pawn."], [],
        id="the-first-hints-own-piece-in-other-words",
    ),
    pytest.param(
        "Bd3", "the player's bishop on b5", "There's a tactic here that wins a pawn.",
        [], ["Your first move takes your bishop out of danger, and it wins a pawn."],
        id="a-piece-the-sheet-never-names",
    ),
]  # fmt: skip


@pytest.mark.parametrize(("answer", "piece", "situation", "fine", "leaking"), PIECE_WORDS)
def test_only_a_word_that_points_at_the_piece_to_move_is_a_leak(
    answer: str, piece: str, situation: str | None, fine: list[str], leaking: list[str]
) -> None:
    f = facts()
    f.candidates, f.piece, f.situation = [answer], piece, situation
    word = "piece " + piece.split()[2]
    for text in fine:
        assert leak_check(text, f, L.IDEA) == [], text
    for text in leaking:
        assert word in leak_check(text, f, L.IDEA), text
