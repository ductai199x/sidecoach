"""The coach's ties to the game, decided in code (the systematic round, 2026-09-25): a lesson
links only to the earlier moments the coach lists -- the player's moves played with the very
same idea -- and names no other; the material is said whole; "why was that bad?" says what
the move allowed; and a figure where none was asked for is a fault the directive states."""

from __future__ import annotations

import chess

from sidecoach.coach.coach import NO_EARLIER, TIE_TO_THE_GAME, unlisted_links
from sidecoach.coach.conversation import Conversation
from sidecoach.coach.disclosure import PositionFacts
from sidecoach.events import (
    Analysis,
    DisclosureLevel,
    GameStarted,
    Label,
    Line,
    MoveAssessment,
    MovePlayed,
    Severity,
    Side,
    Trigger,
)
from tests.test_coach import (
    SCHOLAR,
    _coach,
    _directive,
    _given,
    _harness,
    _mate_analysis,
    _messages,
    _start,
)


def _move(coach, board: chess.Board, ply: int, san: str, *, by_user: bool, ts: float) -> str:
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
            by_user=by_user,
        ),
        ts,
    )
    return move.uci()


def _scholar_with_an_earlier_moment(coach) -> None:
    """The scholar's game, with 2. Bc4 judged best from a position read as the same idea (a
    mating attack) and the coach's word after it."""
    _start(coach)
    board = chess.Board()
    for ply, san in enumerate(("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"), start=1):
        uci = _move(coach, board, ply, san, by_user=ply % 2 == 1, ts=float(ply))
        if san == "Bc4":
            coach.on_event(
                MoveAssessment(
                    game_id="g",
                    ply=ply,
                    uci=uci,
                    expected_loss=0.0,
                    severity=Severity.NONE,
                    best=[uci],
                    label=Label.BEST,
                ),
                float(ply),
            )
            coach.conversation.coach_said("Nice -- Bc4 eyes f7.")
    faced = coach.conversation.records[1]  # after 1... e5: the position 2. Bc4 was played from
    faced.facts = PositionFacts(
        fen=faced.move.fen_after, side_to_move="white", phase="opening", theme="a mating attack"
    )
    assert coach.board.fen() == SCHOLAR


def _miss_the_mate(coach, ts: float = 10.0) -> None:
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
        ts,
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
        ts + 1,
    )


def test_a_lesson_ties_only_to_the_moments_the_coach_lists() -> None:
    coach, provider = _coach(
        [
            "Their king on e8 is nearly out of squares.",
            # Names an earlier move the list does not give, with its number.
            "Qxf7# was mate on the spot; as after 3. Qh5, look for the check first. Which "
            "check ends it?",
            "Qxf7# was mate on the spot: as with Bc4, the target was f7. Which check ends it?",
        ]
    )
    _scholar_with_an_earlier_moment(coach)
    coach.on_event(_mate_analysis(6), 7.0)
    _messages(coach)
    _miss_the_mate(coach)
    out = _messages(coach)
    assert len(out) == 1 and out[0].trigger is Trigger.POST_MISTAKE
    content = _harness(_given(provider, 1))
    section = content.split("# Earlier in this game, the same idea: a mating attack\n")[1]
    # The coach's words on a line of their own, as everywhere in the record: the judge marks
    # them as the coach's, evidence only of what was said.
    assert section.startswith(
        "- At Bc4 (the player's move): best -- the player found it.\n"
        '    YOU (the coach) said: "Nice -- Bc4 eyes f7."'
    )
    directive = _directive(_given(provider, 1))
    assert "Under 'Earlier in this game'" in directive and TIE_TO_THE_GAME not in directive
    assert "(as with Bg4)" not in directive, "no example move for the model to copy"
    # The first reply named 3. Qh5, an earlier move not listed: told so, and redone.
    retry = str(_given(provider, 2)[-1]["content"])
    assert "names Qh5, an earlier moment not listed" in retry
    assert "a move with its number" in retry
    assert out[0].text.startswith("Qxf7# was mate on the spot: as with Bc4")
    assert coach.spoken[-1].problems == [], coach.spoken[-1].problems


def test_with_no_earlier_moment_of_the_same_idea_the_lesson_speaks_of_none() -> None:
    coach, provider = _coach(
        ["Their king on e8 is nearly out of squares.", "Qxf7# was mate. Which check ends it?"]
    )
    _start(coach)
    board = chess.Board()
    for ply, san in enumerate(("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"), start=1):
        _move(coach, board, ply, san, by_user=ply % 2 == 1, ts=float(ply))
    coach.on_event(_mate_analysis(6), 7.0)
    _messages(coach)
    _miss_the_mate(coach)
    _messages(coach)
    content = _harness(_given(provider, 1))
    assert "# Earlier in this game" not in content
    assert NO_EARLIER in _directive(_given(provider, 1))


def _conversation_game() -> Conversation:
    """1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4: every position the player faced read as the same
    plan, the first move book, 2. Nf3 a mistake with the coach's word after it."""
    conversation = Conversation()
    conversation.new_game(
        GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.Board().fen())
    )
    board = chess.Board()
    judged = {
        1: (Label.BOOK, Severity.NONE, 0.0, None),
        3: (Label.MISTAKE, Severity.MISTAKE, 0.12, "b1c3"),
        5: (Label.BEST, Severity.NONE, 0.0, None),
        7: (Label.GOOD, Severity.NONE, 0.02, "b5c6"),
    }
    for ply, san in enumerate(("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4"), start=1):
        move = board.parse_san(san)
        before = board.fen()
        board.push(move)
        record = conversation.move(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=before,
                fen_after=board.fen(),
                by_user=ply % 2 == 1,
            )
        )
        record.facts = PositionFacts(
            fen=board.fen(), side_to_move="w", phase="opening", plan="finishing your development"
        )
        if ply in judged:
            label, severity, loss, best = judged[ply]
            conversation.assess(
                MoveAssessment(
                    game_id="g",
                    ply=ply,
                    uci=move.uci(),
                    expected_loss=loss,
                    severity=severity,
                    best=[best or move.uci()],
                    label=label,
                )
            )
        if san == "Nf3":
            conversation.coach_said("That one cost you: Nc3 was better.")
    return conversation


def test_the_earlier_moments_are_the_latest_of_the_same_idea_and_never_book() -> None:
    conversation = _conversation_game()
    lines = conversation.same_idea("finishing your development", before_ply=9)
    assert lines == [
        "- At Bb5 (the player's move): best -- the player found it.",
        "- At Ba4 (the player's move): good; Bxc6 was better.",
    ]
    lines = conversation.same_idea("finishing your development", before_ply=7, limit=3)
    assert lines == [
        "- At Nf3 (the player's move): a mistake; Nc3 was better.\n"
        '    YOU (the coach) said: "That one cost you: Nc3 was better."',
        "- At Bb5 (the player's move): best -- the player found it.",
    ], "1. e4 was book, from no faced position; 4. Ba4 is the move taught"
    assert conversation.same_idea("a fork", before_ply=9) == []
    # The moves a reply may not name: pieces, captures and castling, never a pawn's step.
    assert conversation.moves_before(6) == frozenset({"Nf3", "Nc6", "Bb5", "Nc3"})


def test_an_unlisted_earlier_move_and_a_move_number_are_faults() -> None:
    unlisted = frozenset({"Bc4", "Qh5"})
    assert unlisted_links("Keep your bishop on c4 in mind.", unlisted) == []
    assert unlisted_links("As with Bc4, the target was f7.", unlisted) == [
        "names Bc4, an earlier moment not listed: tie only to the moments listed"
    ]
    assert unlisted_links("The same lesson as after 9...a6.", frozenset()) == [
        "a move with its number: name moves bare"
    ]
    assert unlisted_links("It cost you 8% of the game. Rc1 was better.", unlisted) == []


def _capture_game(coach) -> None:
    """1. e4 d5 2. Nc3 dxe4: the player, White, to move, a pawn down, with Nxe4 to come."""
    _start(coach)
    board = chess.Board()
    for ply, san in enumerate(("e4", "d5", "Nc3", "dxe4"), start=1):
        _move(coach, board, ply, san, by_user=ply % 2 == 1, ts=float(ply))
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=4,
            lines=[Line(moves=["c3e4", "g8f6"], depth=12, expected_score=0.5)],
            decisive=False,
            only_move=False,
        ),
        5.0,
    )


def test_the_material_is_said_now_and_after_the_recapture_when_it_gives_nothing_away() -> None:
    coach, _ = _coach([])
    _capture_game(coach)
    position = coach.positions[4]
    directive, template = coach._instruction(
        Trigger.USER_ASK, position, DisclosureLevel.IDEA, "am I up material?", (), None, None
    )
    # Before the move rung the recapture is the player's move to find: the material now,
    # only ("once you take back" pointed at the answer where the choice of piece was the
    # whole question: the pre-spend audit, 2026-09-25).
    assert template == "Material: you are down a pawn."
    assert "in these terms: you are down a pawn." in directive
    _, template = coach._instruction(
        Trigger.USER_ASK, position, DisclosureLevel.MOVE, "am I up material?", (), None, None
    )
    assert (
        template == "Material: you are down a pawn -- and after your recapture Nxe4, it is level."
    )


def test_why_was_that_bad_says_what_the_move_allowed() -> None:
    # 1. Qxd5?? leaves the rook on a1 to Rxa1+; Rxa8+ was the move.
    start = "r3k3/5ppp/8/3p4/8/8/5PPP/R2QK3 w - - 0 1"
    coach, provider = _coach([])
    coach.on_event(GameStarted(game_id="g", user_side=Side.WHITE, start_fen=start), 0.0)
    coach.on_event(
        Analysis(
            game_id="g",
            fen=start,
            ply=0,
            lines=[Line(moves=["a1a8", "e8e7"], depth=12, expected_score=0.99)],
            decisive=False,
            only_move=False,
        ),
        0.5,
    )
    board = chess.Board(start)
    uci = _move(coach, board, 1, "Qxd5", by_user=True, ts=1.0)
    coach.on_event(
        MoveAssessment(
            game_id="g",
            ply=1,
            uci=uci,
            expected_loss=0.45,
            severity=Severity.BLUNDER,
            best=["a1a8"],
            label=Label.BLUNDER,
        ),
        1.5,
    )
    after = board.copy()
    line = []
    for san in ("Rxa1+", "Ke2", "Ra2+", "Kf3"):
        move = after.parse_san(san)
        line.append(move.uci())
        after.push(move)
    coach.on_event(
        Analysis(
            game_id="g",
            fen=board.fen(),
            ply=1,
            lines=[Line(moves=line, depth=12, expected_score=0.95)],
            decisive=False,
            only_move=False,
        ),
        2.0,
    )
    _messages(coach)
    coach.ask("why was that bad?", 3.0)
    _messages(coach)
    directive = _directive(_given(provider))
    assert "What Qxd5 allowed, from the engine's line: after Qxd5, Rxa1+ wins a rook." in (
        directive
    )
    assert "say why, concretely: what Rxa8+ does, and what Qxd5 missed or allowed" in directive
    assert "Name moves bare" in directive


def test_a_figure_in_a_hint_nobody_asked_a_figure_of_is_a_fault() -> None:
    coach, provider = _coach(
        [
            "You are ahead by about 1.2 pawns: their king on e8 has one square. Which check?",
            "Their king on e8 has one square left. Which check takes it away?",
        ]
    )
    _start(coach)
    board = chess.Board()
    for ply, san in enumerate(("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"), start=1):
        _move(coach, board, ply, san, by_user=ply % 2 == 1, ts=float(ply))
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    coach.ask("give me a hint", 8.0)
    out = _messages(coach)
    assert "Quote no figure" in _directive(_given(provider, 0))
    retry = str(_given(provider, 1)[-1]["content"])
    assert "a figure in a remark nobody asked for" in retry
    assert out[-1].text.startswith("Their king on e8 has one square left")
