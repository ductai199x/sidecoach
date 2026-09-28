"""The conversation renderer: what a ply message says about the move and its verdict."""

from __future__ import annotations

import chess

from sidecoach.coach.conversation import Conversation
from sidecoach.events import GameStarted, Label, MoveAssessment, MovePlayed, Severity, Side


def test_the_preferred_line_is_dropped_only_when_its_first_move_was_played() -> None:
    """After 10...O-O (a mistake) the engine's line was 10...Nxe4 11. Qxd4 Nf6 12. O-O:
    it ends on the move played, and the verdict lost the line entirely. The line is
    dropped only when its first move is the one played."""
    convo = Conversation()
    convo.new_game(GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.Board().fen()))
    fen_before = "r1bqk2r/2p1bppp/p2p1n2/1p1P4/3pP3/1B6/PPP1NPPP/R1BQK2R b KQkq - 1 10"
    board = chess.Board(fen_before)
    move = board.parse_san("O-O")
    board.push(move)
    record = convo.move(
        MovePlayed(
            game_id="g",
            ply=20,
            san="O-O",
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=board.fen(),
            by_user=False,
        )
    )
    convo.assess(
        MoveAssessment(
            game_id="g",
            ply=20,
            uci=move.uci(),
            expected_loss=0.13,
            severity=Severity.MISTAKE,
            best=["f6e4", "d1d4", "e4f6", "e1g1"],
            label=Label.MISTAKE,
        )
    )
    text = convo.render_ply(record)
    assert "the engine preferred 10... Nxe4 11. Qxd4 Nf6 12. O-O" in text
    # The line whose first move is the one played is not repeated back.
    convo.assess(
        MoveAssessment(
            game_id="g",
            ply=20,
            uci=move.uci(),
            expected_loss=0.01,
            severity=Severity.NONE,
            best=[move.uci(), "a2a4"],
            label=Label.GOOD,
        )
    )
    assert "the engine preferred" not in convo.render_ply(record)


def test_the_facts_are_reworded_in_the_third_person_for_the_side_to_move() -> None:
    """A fact is written for the side to move as "you"; the harness names the side. One
    pass: a pattern table turned "your opponent's" into "your" and then into "their", and
    the opponent's isolated pawn became the player's."""
    from sidecoach.coach.person import third_person

    assert third_person("you still have 2 pieces at home (b8, c8), not yet in the game") == (
        "the player still has 2 pieces at home (b8, c8), not yet in the game"
    )
    assert third_person("your king is still in the centre on e8", side="opponent") == (
        "the opponent's king is still in the centre on e8"
    )
    assert third_person("Your opponent's rook holds the file; it attacks you.") == (
        "The opponent's rook holds the file; it attacks the player."
    )
    assert third_person("your opponent's isolated d-pawn on d5", side="opponent") == (
        "the player's isolated d-pawn on d5"
    )
    assert third_person("storming the pawns in front of your opponent's king", side="opponent") == (
        "storming the pawns in front of the player's king"
    )
    assert third_person("you win a pawn; they are worse; you need to castle") == (
        "the player wins a pawn; the opponent is worse; the player needs to castle"
    )
    assert third_person("yours is the better bishop") == "the player's is the better bishop"


def test_the_verdict_says_what_the_move_does_and_what_the_better_move_would_do() -> None:
    """The coach's commonest invention was the reason a move was good or bad. The verdict
    now carries the mechanism, computed on the board: what the move takes, attacks,
    defends, leaves undefended; the same for the move the engine preferred. (What a move
    does on the board is test_effects_legal.py's.)"""
    # In the message: the played move's effects and the better move's.
    convo = Conversation()
    convo.new_game(GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.Board().fen()))
    fen_before = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
    board = chess.Board(fen_before)
    move = board.parse_san("Nc3")
    board.push(move)
    record = convo.move(
        MovePlayed(
            game_id="g",
            ply=5,
            san="Nc3",
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=board.fen(),
            by_user=True,
        )
    )
    convo.assess(
        MoveAssessment(
            game_id="g",
            ply=5,
            uci=move.uci(),
            expected_loss=0.03,
            severity=Severity.INACCURACY,
            best=["f1b5"],
            label=Label.INACCURACY,
        )
    )
    text = convo.render_ply(record)
    assert "What Bb5 would do: attacks the knight on c6." in text
    assert "What Nc3 does:" not in text  # a quiet developing move: nothing the board can name


def test_the_position_sections_on_the_opponents_turn() -> None:
    """After the player's capture, with the opponent to move: the facts heading closes its
    bracket and says whose plan they are; the weaker lines say whose; the material notes
    the recapture the engine expects, so "up a knight" is not read as a knight won."""
    from sidecoach.coach.facts import facts_from_analysis
    from sidecoach.events import Analysis, DisclosureLevel, Line

    convo = Conversation()
    convo.new_game(GameStarted(game_id="g", user_side=Side.BLACK, start_fen=chess.Board().fen()))
    # Black (the player) takes a knight on e6 that a pawn on d5 recaptures.
    fen_before = "4k3/2n5/4N3/3P4/8/8/8/4K3 b - - 0 40"
    board = chess.Board(fen_before)
    move = board.parse_san("Nxe6")
    board.push(move)
    record = convo.move(
        MovePlayed(
            game_id="g",
            ply=79,
            san="Nxe6",
            uci=move.uci(),
            fen_before=fen_before,
            fen_after=board.fen(),
            by_user=True,
        )
    )
    convo.assess(
        MoveAssessment(
            game_id="g",
            ply=79,
            uci=move.uci(),
            expected_loss=0.0,
            severity=Severity.NONE,
            best=[move.uci()],
            label=Label.BEST,
        )
    )
    analysis = Analysis(
        game_id="g",
        fen=board.fen(),
        ply=79,
        lines=[
            Line(moves=["d5e6", "e8e7"], depth=20, expected_score=0.5),
            Line(moves=["e1e2", "e6d4"], depth=20, expected_score=0.1),
            Line(moves=["e1d1", "e6d4"], depth=20, expected_score=0.1),
            Line(moves=["e1f1", "e6d4"], depth=20, expected_score=0.1),
        ],
        decisive=True,
        only_move=True,
    )
    convo.analysed(analysis, facts_from_analysis(board, analysis, user=chess.BLACK))
    text = convo.render_ply(record, DisclosureLevel.QUESTION)
    assert (
        "Material: the player is up a knight for a pawn before the expected recapture dxe6, "
        "then the player is down a pawn."
    ) in text
    assert "## Facts (the opponent to move; the opponent's plan and ideas)\n" in text
    assert "decisive: yes\nonly move: yes\n" in text
    assert "  weaker for the opponent:\n" in text
    assert "## Standing\nIt is about equal" in text


def test_the_earlier_dialogue_folds_into_a_summary_and_the_last_plies_stay_verbatim() -> None:
    """Dialogue compaction: what was said earlier than the last `keep` plies is folded into
    one summary block at the top of the record; the summariser sees only the dialogue, each
    line with the move it followed; the last plies keep their words."""
    from sidecoach.coach.summarise import summarise
    from sidecoach.llm.base import Completion

    class Summariser:
        name = "s"
        prompts: list[str] = []

        def healthcheck(self) -> tuple[bool, str]:
            return True, "s"

        def chat(self, system: str, messages, *, max_tokens=1500, temperature=0.7, thinking=False):
            self.prompts.append(str(messages[-1]["content"]))
            return Completion(text="The coach praised 1. e4 and told the player to develop.")

    convo = Conversation()
    convo.new_game(GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.Board().fen()))
    board = chess.Board()
    for ply, san in enumerate(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"], start=1):
        move = board.parse_san(san)
        fen_before = board.fen()
        board.push(move)
        convo.move(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=fen_before,
                fen_after=board.fen(),
                by_user=ply % 2 == 1,
            )
        )
        if ply == 1:
            convo.coach_said("A fine first move.")
        if ply == 2:
            convo.user_said("what now?")
            convo.coach_said("Develop a piece.")
        if ply == 5:
            convo.coach_said("The Ruy Lopez.")
    summariser = Summariser()
    assert summarise(convo, summariser, keep=2)
    prompt = summariser.prompts[-1]
    assert "after 1. e4, COACH: A fine first move." in prompt
    assert 'after 1... e5, PLAYER: "what now?"' in prompt
    assert "The Ruy Lopez" not in prompt, "within the last two plies: kept verbatim"
    text = convo.render()
    assert "Earlier dialogue, summarised: The coach praised 1. e4" in text
    assert "A fine first move." not in text and "Develop a piece." not in text
    assert 'YOU (the coach) said: "The Ruy Lopez."' in text
    # Nothing older to fold the second time: no call is made.
    assert not summarise(convo, summariser, keep=2)
    assert len(summariser.prompts) == 1


def test_the_compact_record_drops_stale_nods_and_runs_sound_moves_together() -> None:
    """The routine nod after a sound move was most of a mid-game harness. With the
    compact record, nods older than the last few plies leave the record (the verdict is
    on the move's line), exchanges stay, and sound moves nobody spoke after run together
    as a move list; a move that lost ground keeps its line."""
    from sidecoach.coach.conversation import NOD_PLIES

    convo = Conversation(compact_record=True)
    convo.new_game(GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.Board().fen()))
    board = chess.Board()
    sans = [
        "e4",
        "e5",
        "Nf3",
        "Nc6",
        "Bb5",
        "a6",
        "Ba4",
        "Nf6",
        "O-O",
        "Be7",
        "Re1",
        "b5",
        "Bb3",
        "d6",
    ]
    for ply, san in enumerate(sans, start=1):
        move = board.parse_san(san)
        fen_before = board.fen()
        board.push(move)
        convo.move(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=fen_before,
                fen_after=board.fen(),
                by_user=ply % 2 == 1,
            )
        )
        severity = Severity.INACCURACY if ply == 5 else Severity.NONE
        convo.assess(
            MoveAssessment(
                game_id="g",
                ply=ply,
                uci=move.uci(),
                expected_loss=0.05 if ply == 5 else 0.0,
                severity=severity,
                best=[move.uci()] if ply != 5 else ["f1c4"],
                label=Label.INACCURACY if ply == 5 else Label.BOOK,
            )
        )
        if ply % 2 == 1:
            convo.coach_said(f"Nod after {san}.", nod=True)
        if ply == 6:
            convo.user_said("what now?")
            convo.coach_said("An answer.")
    text = convo.render()
    record = text.split("# Game so far\n", 1)[1].split("# Position now", 1)[0]
    assert "Nod after e4." not in record and "Nod after Nf3." not in record, "stale nods are gone"
    assert f"Nod after {sans[-2]}." in record, "the last nods stay"
    assert 'PLAYER said: "what now?"' in record and 'YOU (the coach) said: "An answer."' in record
    assert "3. Bb5 (the player): inaccuracy, cost 0.05" in record, "a slip keeps its line"
    assert "Moves: 1. e4 e5 2. Nf3 Nc6" in record
    # Stale nods are not summarised either: only the exchange is offered to the summariser.
    pending = convo.foldable(2)
    assert [e.text for _, e in pending] == ['"what now?"', "An answer."]
    assert NOD_PLIES == 6


def test_a_capture_the_engine_takes_back_is_said_as_such_before_the_position_is_read() -> None:
    """Qxe5 takes their queen and Rxe5 takes back: with the new position not yet analysed,
    the engine's own line for the move names the recapture. Without it "up a queen for a
    pawn" was said, and the model called the queen trade a queen won (the dev audit,
    2026-09-26)."""
    from sidecoach.coach.conversation import PlyRecord

    before = chess.Board("1r5r/pbp3p1/5nkp/4Qq2/8/2N2P2/PP2R1PP/R1B3K1 b - - 6 19")
    move = before.parse_san("Qxe5")
    after = before.copy()
    after.push(move)
    record = PlyRecord(
        move=MovePlayed(
            game_id="g",
            ply=38,
            san="Qxe5",
            uci=move.uci(),
            fen_before=before.fen(),
            fen_after=after.fen(),
            by_user=True,
        ),
        assessment=MoveAssessment(
            game_id="g",
            ply=38,
            uci=move.uci(),
            expected_loss=0.0,
            severity=Severity.NONE,
            best=[move.uci(), "e2e5", "h8e8"],
            label=Label.GREAT,
        ),
    )
    found = Conversation._expected_recapture(record, before, after)
    assert found is not None and found[0] == "Rxe5"
