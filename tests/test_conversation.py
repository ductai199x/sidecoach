"""The conversation renderer: what a ply message says about the move and its verdict."""

from __future__ import annotations

import chess
import pytest

from sidecoach.coach.conversation import Conversation, PlyRecord
from sidecoach.events import (
    Analysis,
    GameStarted,
    Label,
    Line,
    MoveAssessment,
    MovePlayed,
    Resync,
    Severity,
    Side,
    Takeback,
)
from tests.coaching import drain, harness_of, scripted_coach, sent, stuck
from tests.fakes import ScriptedProvider


def _conversation(side: Side = Side.WHITE, *, compact_record: bool = False) -> Conversation:
    convo = Conversation(compact_record=compact_record)
    convo.new_game(GameStarted(game_id="g", user_side=side, start_fen=chess.Board().fen()))
    return convo


def _move(
    convo: Conversation, board: chess.Board, ply: int, san: str, *, by_user: bool
) -> PlyRecord:
    """`san` played on `board`, which it leaves after the move, and its record."""
    move = board.parse_san(san)
    before = board.fen()
    board.push(move)
    return convo.move(
        MovePlayed(
            game_id="g",
            ply=ply,
            san=san,
            uci=move.uci(),
            fen_before=before,
            fen_after=board.fen(),
            by_user=by_user,
        )
    )


def _judge(
    convo: Conversation,
    record: PlyRecord,
    label: Label,
    severity: Severity,
    loss: float,
    best: list[str],
) -> None:
    convo.assess(
        MoveAssessment(
            game_id="g",
            ply=record.move.ply,
            uci=record.move.uci,
            expected_loss=loss,
            severity=severity,
            best=best,
            label=label,
        )
    )


def test_the_preferred_line_is_dropped_only_when_its_first_move_was_played() -> None:
    """After 10...O-O (a mistake) the engine's line was 10...Nxe4 11. Qxd4 Nf6 12. O-O:
    it ends on the move played, and the verdict lost the line entirely. The line is
    dropped only when its first move is the one played."""
    convo = _conversation()
    board = chess.Board("r1bqk2r/2p1bppp/p2p1n2/1p1P4/3pP3/1B6/PPP1NPPP/R1BQK2R b KQkq - 1 10")
    record = _move(convo, board, 20, "O-O", by_user=False)
    line = ["f6e4", "d1d4", "e4f6", "e1g1"]
    _judge(convo, record, Label.MISTAKE, Severity.MISTAKE, 0.13, line)
    assert "the engine preferred 10... Nxe4 11. Qxd4 Nf6 12. O-O" in convo.render_ply(record)
    # The line whose first move is the one played is not repeated back.
    _judge(convo, record, Label.GOOD, Severity.NONE, 0.01, [record.move.uci, "a2a4"])
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
    # "you" as an object takes no verb: "insteads", "twices" (the deep prompt review).
    assert third_person("it gives you a pawn instead; it hits you twice") == (
        "it gives the player a pawn instead; it hits the player twice"
    )
    assert third_person("castling gives you better chances") == (
        "castling gives the player better chances"
    )
    assert third_person("this leaves you instead of them with the pawn") == (
        "this leaves the player instead of the opponent with the pawn"
    )


def test_the_verdict_says_what_the_move_does_and_what_the_better_move_would_do() -> None:
    """The coach's commonest invention was the reason a move was good or bad. The verdict
    now carries the mechanism, computed on the board: what the move takes, attacks,
    defends, leaves undefended; the same for the move the engine preferred. (Bb5 attacking the
    knight on c6 is test_effects_legal.py's.)"""
    from sidecoach.coach.effects import describe_effects

    # A queen stepping onto an open file with nothing to attack: the file is named.
    board = chess.Board("4k3/8/8/8/8/8/8/Q3K3 w - - 0 1")
    assert "takes the open d-file" in describe_effects(board, board.parse_san("Qd1"))
    # In the message: the played move's effects and the better move's.
    convo = _conversation()
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    record = _move(convo, board, 5, "Nc3", by_user=True)
    _judge(convo, record, Label.INACCURACY, Severity.INACCURACY, 0.03, ["f1b5"])
    text = convo.render_ply(record)
    assert "What Bb5 would do: attacks the knight on c6." in text
    assert "What Nc3 does:" not in text  # a quiet developing move: nothing the board can name


def test_the_setting_says_left_and_right_as_the_player_sees_them() -> None:
    """Playing Black, "he will just move away to the right" (Kh1-g2) was answered "their King
    cannot move to the right" (Tai's puzzle, 2026-09-28)."""
    for side, right in ((Side.WHITE, "h"), (Side.BLACK, "a")):
        setting = _conversation(side).messages()[0]["content"]
        assert f"'to the right' means toward the {right}-file" in setting, side


def test_left_and_right_in_their_words_are_said_again_by_the_question() -> None:
    """Said once in the Setting, far above, left and right were still read from White's side
    ("from h1 there is no square to the right", both models): a question that says them
    carries them."""
    from sidecoach.coach.coach import _SIDEWAYS

    for said in ("he will just move away to the right, no?", "on my left side", "rightwards"):
        assert _SIDEWAYS.search(said), said
    for said in ("is that right?", "right, so what now?", "what's left to do?"):
        assert not _SIDEWAYS.search(said), said
    # The moves in question given their way: Kh1-g2 is to Black's right, Rf1-g1 to its left.
    from sidecoach.coach.conversation import ways

    board = chess.Board("6k1/b1p2pp1/p7/1p6/1P6/P5pP/3R4/3N1r1K w - - 0 4")
    assert ways(board, ["Kg2", "Rg1+"], Side.BLACK) == (
        "Kg2 (h1 to g2) goes to their right; Rg1+ (f1 to g1) goes to their left"
    )
    assert ways(chess.Board(), ["e4"], Side.WHITE) == "e4 (e2 to e4) goes along the file"
    assert ways(board, ["Rxf1+"], Side.BLACK) == ""  # not a move on this board


def test_the_position_sections_on_the_opponents_turn() -> None:
    """After the player's capture, with the opponent to move: the facts heading closes its
    bracket and says whose plan they are; the weaker lines say whose; the material notes
    the recapture the engine expects, so "up a knight" is not read as a knight won."""
    from sidecoach.coach.facts import facts_from_analysis
    from sidecoach.events import Analysis, DisclosureLevel, Line

    convo = _conversation(Side.BLACK)
    # Black (the player) takes a knight on e6 that a pawn on d5 recaptures.
    board = chess.Board("4k3/2n5/4N3/3P4/8/8/8/4K3 b - - 0 40")
    record = _move(convo, board, 79, "Nxe6", by_user=True)
    _judge(convo, record, Label.BEST, Severity.NONE, 0.0, [record.move.uci])
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
    text = convo.render_ply(record, DisclosureLevel.IDEA)
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

    convo = _conversation()
    board = chess.Board()
    for ply, san in enumerate(["e4", "e5", "Nf3", "Nc6", "Bb5", "a6"], start=1):
        _move(convo, board, ply, san, by_user=ply % 2 == 1)
        if ply == 1:
            convo.coach_said("A fine first move.")
        if ply == 2:
            convo.user_said("what now?")
            convo.coach_said("Develop a piece.")
        if ply == 5:
            convo.coach_said("The Ruy Lopez.")
    summariser = ScriptedProvider(then="The coach praised 1. e4 and told the player to develop.")
    assert summarise(convo, summariser, keep=2)
    prompt = str(sent(summariser)[-1]["content"])
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
    # The summary is the game's own: the next game starts without it, and one still being
    # written when that game began is dropped -- it opened the next game's record (the parity
    # audit, 2026-09-29).
    pending = [entry for _, entry in convo.foldable(1)]
    assert pending
    convo.new_game(GameStarted(game_id="g2", user_side=Side.WHITE, start_fen=chess.Board().fen()))
    assert "Earlier dialogue" not in convo.render()
    assert not convo.fold(pending, "The last game's dialogue.") and convo.summary == ""


def test_the_compact_record_drops_stale_nods_and_runs_sound_moves_together() -> None:
    """The routine nod after a sound move was most of a mid-game harness. With the
    compact record, nods older than the last few plies leave the record (the verdict is
    on the move's line), exchanges stay, and sound moves nobody spoke after run together
    as a move list; a move that lost ground keeps its line."""
    from sidecoach.coach.conversation import NOD_PLIES

    convo = _conversation(compact_record=True)
    board = chess.Board()
    sans = ["e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "b5"]
    sans += ["Bb3", "d6"]
    for ply, san in enumerate(sans, start=1):
        record = _move(convo, board, ply, san, by_user=ply % 2 == 1)
        if ply == 5:
            _judge(convo, record, Label.INACCURACY, Severity.INACCURACY, 0.05, ["f1c4"])
        else:
            _judge(convo, record, Label.BOOK, Severity.NONE, 0.0, [record.move.uci])
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
    # The first nods (after plies 1 and 3) are older than the last NOD_PLIES plies.
    assert len(sans) - NOD_PLIES > 3, "the game is long enough for the first nods to be stale"


def test_a_capture_the_engine_takes_back_is_said_as_such_before_the_position_is_read() -> None:
    """Qxe5 takes their queen and Rxe5 takes back: with the new position not yet analysed,
    the engine's own line for the move names the recapture. Without it "up a queen for a
    pawn" was said, and the model called the queen trade a queen won (the dev audit,
    2026-09-26)."""
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


# Tai's puzzle (2026-10-01), Black to move: the stuck hint was asked of the model with no
# position now, and it called Black's own Qxf2+ "your opponent's threat".
JOINED = "4r1k1/p6p/6p1/2Q5/8/N1P2q1P/PP1K1P2/R6R b - - 0 1"
# The board read anew, the player to move: White's and Black's.
REREAD = {
    Side.WHITE: "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
    Side.BLACK: "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/2N5/PPPP1PPP/R1B1K1NR b KQkq - 5 4",
}
STARTS = {
    "game": (chess.STARTING_FEN, Side.WHITE, "game"),
    "joined": (JOINED, Side.BLACK, "game"),
    "position": (JOINED, Side.BLACK, "position"),
    "puzzle": (JOINED, Side.BLACK, "puzzle"),
}


def _analysis(board: chess.Board, ply: int, *, provisional: bool = False) -> Analysis:
    """The engine's look at `board`: two of its legal moves, the first the best; `provisional`
    the early depth, the full one to follow."""
    first, second = sorted(board.legal_moves, key=lambda m: (not board.is_capture(m), m.uci()))[:2]
    depth = 12 if provisional else 18
    return Analysis(
        game_id="g",
        fen=board.fen(),
        ply=ply,
        lines=[
            Line(moves=[first.uci()], depth=depth, expected_score=0.7),
            Line(moves=[second.uci()], depth=depth, expected_score=0.5),
        ],
        decisive=False,
        only_move=False,
        provisional=provisional,
    )


@pytest.mark.parametrize("trigger", ["stuck", "asked"])
@pytest.mark.parametrize("state", ["start", "moved", "taken back", "re-read"])
@pytest.mark.parametrize("start", list(STARTS))
def test_the_position_now_is_the_board_the_player_faces(start: str, state: str, trigger: str):
    """Every way a game begins, every way its position changes, every way the coach is
    asked to speak: the model is given the position the player faces -- its board, the side
    to move, the engine's lines, the facts -- and no other as the position now. At a puzzle's
    start, a joined game's, a re-read board's or after a takeback to the start, it was given
    none, or the withdrawn move's (Tai, 2026-10-01). Asked before the engine has looked, the
    question is answered from its early depth, and the position now shows that depth's lines
    and facts: only the full depth reached the record, and the model was told "Engine lines:
    not in yet." with no facts (the coach review, 2026-10-01). Only the position now has its
    legal moves: the setting, the takeback and the re-read show a board as where the record
    starts or went back to, and after a re-read the setting's board was a second "position as
    it stands"."""
    fen, side, kind = STARTS[start]
    coach, provider = scripted_coach([])
    coach.on_event(GameStarted(game_id="g", user_side=side, start_fen=fen, kind=kind), 0.0)
    board = chess.Board(fen)
    # Asked, the position now's analysis lands after the question, at its early depth.
    later: list[tuple[chess.Board, int]] = []

    def look(at: chess.Board, ply: int, ts: float, *, now: bool = False) -> None:
        if now and trigger == "asked":
            later.append((at.copy(), ply))
        else:
            coach.on_event(_analysis(at, ply), ts)

    look(board, 0, 0.5, now=state == "start")
    gone: str | None = None
    if state in ("moved", "taken back"):
        for ply in (1, 2):
            move = sorted(board.legal_moves, key=lambda m: m.uci())[0]
            san, before = board.san(move), board.fen()
            board.push(move)
            coach.on_event(
                MovePlayed(
                    game_id="g",
                    ply=ply,
                    san=san,
                    uci=move.uci(),
                    fen_before=before,
                    fen_after=board.fen(),
                    by_user=ply == 1,
                ),
                float(ply),
            )
            look(board, ply, ply + 0.5, now=state == "moved" and ply == 2)
        if state == "taken back":
            gone = board.fen()
            board = chess.Board(fen)
            coach.conversation.coach_said("Look at the knight.")  # of the withdrawn position
            coach.on_event(Takeback(game_id="g", plies=2, fen=fen), 3.0)
            coach.conversation.clock(60_000, 50_000)
            assert all(r.white_ms is None for r in coach.conversation.records), "withdrawn moves"
    elif state == "re-read":
        gone = fen
        board = chess.Board(REREAD[side])
        coach.conversation.coach_said("Look at the knight.")  # of the board before the re-read
        coach.on_event(Resync(game_id="g", fen=board.fen(), reason="re-read"), 3.0)
        look(board, 0, 3.5, now=True)
    assert coach.user_to_move and coach.board.fen() == board.fen()
    assert not coach._said_since_move(), "nothing said yet at the position now"
    if trigger == "stuck":
        stuck(coach)
    else:
        coach.ask("what should I be thinking about here?", 10.0)
        for at, ply in later:
            drain(coach)
            assert not provider.calls, "the question waits for the engine's look"
            coach.on_event(_analysis(at, ply, provisional=True), 10.5)
        drain(coach)
    assert provider.calls, "the model was asked"
    harness = harness_of(sent(provider))
    assert harness.count("# Position now\n") == 1
    now = harness.split("# Position now\n", 1)[1].split("\n# Task\n", 1)[0]
    assert f"FEN: {board.fen()}" in now
    if gone is not None:
        assert f"FEN: {gone}" not in now, "a position no longer the game's"
        assert "Look at the knight." not in now, "said of a position no longer the game's"
    assert ("White" if board.turn else "Black") + " to move" in now
    assert harness.count("Legal moves: ") == 1 and "Legal moves: " in now, "one board to play on"
    best = board.san(chess.Move.from_uci(_analysis(board, 0).lines[0].moves[0]))
    lines = now.split("## Engine lines", 1)[1].split("\n## ", 1)[0]
    assert f" {best}  0.70" in lines.split("\n")[1], "the engine's best line for this position"
    assert "\n## Facts (" in now, "the facts sheet for this position"
    assert coach.harness_problems == [], "the prompt check passes every prompt built right"


def test_a_late_engine_note_is_the_records_and_has_its_lines() -> None:
    """The analysis of the position the player's move left can land after the opponent's
    reply (the engine measures the newest position first): its note went under "Said here" of
    the position the player now faces, the past position's standing beside the one now's, and
    without its engine lines, filtered by a title they no longer have (the coach review,
    2026-10-01). It is the record's, under the move it is about, lines and all."""
    from tests.coaching import SCHOLAR, SCHOLAR_MOVES, play, start_game

    coach, provider = scripted_coach([])
    start_game(coach)
    play(coach, SCHOLAR_MOVES)
    coach.on_event(_analysis(chess.Board(SCHOLAR), 6), 7.0)
    coach.ask("what is my plan?", 8.0)  # the model has seen the record: its lines are frozen
    drain(coach)
    after_qh5 = chess.Board()
    for san in SCHOLAR_MOVES[:5]:
        after_qh5.push_san(san)
    coach.on_event(_analysis(after_qh5, 5), 9.0)  # 3. Qh5's position, read late
    coach.ask("and what should I watch out for?", 10.0)
    drain(coach)
    harness = harness_of(sent(provider))
    record, now = harness.split("# Position now\n", 1)
    assert "Engine note on the position after 3. Qh5" in record
    note = record.split("Engine note on the position after 3. Qh5", 1)[1]
    assert "Engine lines (the opponent's move" in note and "Standing:" in note
    assert "Engine note" not in now


def test_a_final_word_that_changes_no_line_the_model_read_adds_no_note() -> None:
    """The analyser tells a position whole (provisional), then final once its side searches are
    in, the lines the same: the model read them whole, and a note repeating them on every move
    the game has left only grew the prompt (the audit of 2026-10-05)."""
    from tests.coaching import SCHOLAR, SCHOLAR_MOVES, play, start_game

    coach, provider = scripted_coach([])
    start_game(coach)
    play(coach, SCHOLAR_MOVES)
    coach.on_event(_analysis(chess.Board(SCHOLAR), 6), 6.5)
    after_qh5 = chess.Board()
    for san in SCHOLAR_MOVES[:5]:
        after_qh5.push_san(san)
    whole = _analysis(after_qh5, 5)
    coach.on_event(whole.model_copy(update={"provisional": True}), 7.0)
    coach.ask("what is my plan?", 8.0)  # the model has seen the record: its lines are frozen
    drain(coach)
    coach.on_event(whole, 9.0)  # final: the same lines
    coach.ask("and what should I watch out for?", 10.0)
    drain(coach)
    record = harness_of(sent(provider)).split("# Position now\n", 1)[0]
    assert "Engine note on the position after 3. Qh5" not in record


def test_the_prompt_check_names_what_is_wrong() -> None:
    """The check run before every request: the puzzle start's prompt (no position now), a
    withdrawn position, and the wrong colour are each named; a prompt built right passes."""
    from sidecoach.coach.conversation import harness_problem

    board = chess.Board(JOINED)
    good = (
        "# Setting\nYou coach Black. They see the board from Black's side.\n\n"
        f"# Position now\n## Board\nFEN: {board.fen()}\nPieces -- ..."
    )
    assert harness_problem(good, board, Side.BLACK) is None
    setting_only = good.split("\n\n# Position now")[0]
    assert harness_problem(setting_only, board, Side.BLACK) == (
        "no position now (# Position now with its board)"
    )
    withdrawn = good.replace(board.fen(), chess.STARTING_FEN)
    assert "the position now is rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w" in str(
        harness_problem(withdrawn, board, Side.BLACK)
    )
    wrong_side = harness_problem(good, board, Side.WHITE)
    assert wrong_side == "the setting does not say the player is White"
    # The engine's lines and the facts, when the coach has them for the position now: the early
    # depth was answered from and never shown (the coach review, 2026-10-01).
    unread = good + "\n\n## Engine lines\nnot in yet.\n\n## On the board (complete lists)\n..."
    assert harness_problem(unread, board, Side.BLACK) is None, "nothing known: nothing missing"
    assert harness_problem(unread, board, Side.BLACK, lines=True) == (
        "the position now has no engine lines, though the coach has them"
    )
    read = good + (
        "\n\n## Engine lines (the player's move)\n  1. Qxf2+  0.70\n\n"
        "## On the board (complete lists)\n...\n\n## Said here\n    EVENT: ## Facts (a quote)"
    )
    assert harness_problem(read, board, Side.BLACK, lines=True) is None
    assert harness_problem(read, board, Side.BLACK, lines=True, facts=True) == (
        "the position now has no facts, though the coach has them"
    )
    with_facts = read.replace("\n\n## On the board", "\n\n## Facts (the player to move)\nphase: x")
    assert harness_problem(with_facts, board, Side.BLACK, lines=True, facts=True) is None


def test_a_past_positions_free_squares_are_said_of_that_position() -> None:
    """The sheet of the position they faced said the king's free squares "(as it stands,
    before any move)": read as the board now, where they are no longer free (the deep prompt
    review, 2026-10-01)."""
    convo = _conversation()
    board = chess.Board()
    target = "their king on e8, with one free square: d8"
    now = convo._fact_lines({"target": target}, board, "player")
    past = convo._fact_lines({"target": target}, board, "player", past=True)
    assert now[0].endswith("(as it stands, before any move)"), now
    assert past[0].endswith("(in that position, before the move the player made there)"), past


def test_what_was_said_before_any_move_is_said_at_the_start_position() -> None:
    """At a puzzle, what was said before the first move hung under "Game so far" with no
    position, read as said after a move never made (the deep prompt review, 2026-10-01)."""
    convo = _conversation()
    convo.user_said("hint")
    convo.coach_said("Look at the f-file.")
    _move(convo, chess.Board(), 1, "e4", by_user=True)
    _move(
        convo,
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"),
        2,
        "e5",
        by_user=False,
    )
    record = convo.render().split("# Game so far\n", 1)[1]
    head = 'At the start position, before any move:\n    PLAYER said: "hint"'
    assert record.startswith(head), record
