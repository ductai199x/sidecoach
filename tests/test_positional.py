"""Tests for the positional layer: static strategic features and the best-move idea.

Every fact these produce is deterministic geometry, so each test is a position where the
answer is unambiguous. The point is that the coach's strategic facts are as grounded and
un-fabricatable as the tactical ones.
"""

from __future__ import annotations

import chess
import pytest

from sidecoach.analysis import positional as P


def kinds(facts: list[P.PositionalFact]) -> list[str]:
    return [fact.kind for fact in facts]


def test_a_pawn_with_no_enemy_pawn_ahead_is_passed() -> None:
    # White a-pawn on a5; no black pawn on the a or b files ahead of it.
    board = chess.Board("4k3/8/8/P7/8/8/8/4K3 w - - 0 1")
    assert P.is_passed(board, chess.A5)
    assert kinds(P.passed_pawns(board, chess.WHITE)) == ["passedPawn"]
    # A black pawn on b6 stops it: it can capture on a-file's promotion path.
    board = chess.Board("4k3/8/1p6/P7/8/8/8/4K3 w - - 0 1")
    assert not P.is_passed(board, chess.A5)
    assert P.passed_pawns(board, chess.WHITE) == []


def test_a_file_with_no_pawns_is_open_and_with_only_enemy_pawns_half_open() -> None:
    board = chess.Board("3rk3/8/8/8/8/8/8/3RK3 w - - 0 1")
    assert kinds(P.open_files(board, chess.WHITE)) == ["openFile"]
    board = chess.Board("3rk3/3p4/8/8/8/8/8/3RK3 w - - 0 1")
    assert kinds(P.open_files(board, chess.WHITE)) == ["halfOpenFile"]
    # A friendly pawn on the file means it is neither open nor half-open for you.
    board = chess.Board("3rk3/8/8/8/8/8/3P4/3RK3 w - - 0 1")
    assert P.open_files(board, chess.WHITE) == []


def test_a_pawn_guarded_square_no_enemy_pawn_can_reach_is_an_outpost() -> None:
    # White knight on d5, guarded by the e4 pawn; black has no c- or e-pawn to challenge it.
    board = chess.Board("4k3/8/8/3N4/4P3/8/8/4K3 w - - 0 1")
    assert P._is_outpost(board, chess.D5, chess.WHITE)
    assert kinds(P.outposts(board, chess.WHITE)) == ["outpost"]
    # A black c-pawn that can advance to c6 and hit d5 spoils it.
    board = chess.Board("4k3/2p5/8/3N4/4P3/8/8/4K3 w - - 0 1")
    assert not P._is_outpost(board, chess.D5, chess.WHITE)


def test_development_counts_minor_pieces_at_home() -> None:
    board = chess.Board()
    board.push_san("e4")
    facts = P.development(board, chess.BLACK)
    assert kinds(facts) == ["undeveloped"]
    assert set(facts[0].squares) == {"b8", "c8", "f8", "g8"}


def test_an_uncastled_king_with_rights_is_named() -> None:
    board = chess.Board("rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    assert kinds(P.uncastled_king(board, chess.WHITE)) == ["uncastled"]
    board.push_san("O-O")
    # After castling, the black king still in the centre is the one named for black.
    assert kinds(P.uncastled_king(board, chess.WHITE)) == []


def test_up_a_piece_is_a_convert_plan_and_down_a_piece_a_complicate_plan() -> None:
    up = chess.Board("4k3/8/8/8/8/8/8/N3K3 w - - 0 1")
    assert kinds(P.material_plan(up, chess.WHITE)) == ["convert"]
    assert kinds(P.material_plan(up, chess.BLACK)) == ["complicate"]


# The strategic point of a best move (`best_move_idea`), white's move in each. (case, FEN,
# the move, the phase (None: the default), their threat (UCI, or None), the idea's kind (None:
# no idea), and whether the idea is that kind (True) or anything but it (False)).
ITALIAN = "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
# The bishop on b5 is attacked by the a6 pawn.
RUY_A6 = "rnbqkbnr/1ppp1ppp/p7/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3"
# Black threatens ...Nxe4: the e4 pawn hangs to the f6 knight.
NXE4 = "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4"
# Black has castled kingside (g8).
CASTLED_BLACK = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP1QPPP/R1B1K2R w KQ - 0 7"
# The d5 pawn attacks the knight on c4.
KNIGHT_C4 = "4k3/p7/8/3p4/2N5/8/8/4K3 w - - 0 1"
IDEAS: list[tuple[str, str, str, str | None, str | None, str | None, bool]] = [
    ("castle", ITALIAN, "O-O", "opening", None, "castle", True),
    ("develop", ITALIAN, "Nc3", "opening", None, "develop", True),
    # A knight to a pawn-guarded square in enemy ground is an outpost move.
    ("outpost", "4k3/8/8/8/4P3/2N5/8/4K3 w - - 0 1", "Nd5", None, None, "outpostMove", True),
    # A rook onto a file with no friendly pawn takes the file.
    ("take the file", "3rk3/8/8/8/8/8/4P3/R3K3 w - - 0 1", "Rd1", None, None, "takeFile", True),
    ("push the passer", "4k3/8/8/P7/8/8/8/4K3 w - - 0 1", "a6", None, None, "pushPasser", True),
    # A king step toward the centre in the endgame.
    (
        "activate the king",
        "8/8/8/8/8/8/4K3/6k1 w - - 0 1",
        "Kd3",
        "endgame",
        None,
        "activateKing",
        True,
    ),
    # A pawn push that hits an enemy pawn is a break.
    ("break", "4k3/8/3p4/8/2P5/8/8/4K3 w - - 0 1", "c5", None, None, "break", True),
    # A rook dropping behind a passed pawn is the classic rook-behind-passer idea.
    (
        "rook behind the passer",
        "4k3/8/8/P7/8/8/8/1R2K3 w - - 0 1",
        "Ra1",
        None,
        None,
        "rookBehindPasser",
        True,
    ),
    # g4 storms the wing black castled on; a4 is only queenside space.
    ("storm", CASTLED_BLACK, "g4", None, None, "pawnStorm", True),
    ("space", CASTLED_BLACK, "a4", None, None, "space", True),
    # --- rescue. Moving the bishop away from ...axb5 is a rescue, not development.
    ("rescue", RUY_A6, "Ba4", "opening", "a6b5", "rescue", True),
    # A piece that could be taken is in danger only when taking it is their threat (the
    # engine's best move for them, were it theirs): "a piece in danger comes first" was said
    # of a rook check that led to mate, and of moves whose threat was elsewhere (the lesson
    # audit, 2026-09-27).
    ("rescue, no threat", RUY_A6, "Ba4", "opening", None, "rescue", False),
    ("rescue, their threat elsewhere", RUY_A6, "Ba4", "opening", "d8h4", "rescue", False),
    # A pawn that steps on from an attack is a push, not a rescue: e4, hit by d5, to e5.
    ("a pawn steps on", "4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1", "e5", None, "d5e4", "rescue", False),
    # A knight beside the enemy king, guarded by its pawn: the king cannot take it (the king
    # used to count as a capturer worth nothing).
    (
        "guarded from the king",
        "8/8/5k2/4N3/3P4/8/8/4K3 w - - 0 1",
        "Nd3",
        None,
        None,
        "rescue",
        False,
    ),
    # A knight attacked only by the c6 knight, which the b5 bishop pins to its king.
    (
        "attacked by a pinned piece",
        "4k3/8/2n5/1B6/3N4/8/8/4K3 w - - 0 1",
        "Nf3",
        None,
        None,
        "rescue",
        False,
    ),
    # Running to b6, where the a7 pawn takes it, is no rescue; e3 is.
    ("rescue to an unsafe square", KNIGHT_C4, "Nb6", None, "d5c4", "rescue", False),
    ("rescue to a safe square", KNIGHT_C4, "Ne3", None, "d5c4", "rescue", True),
    # --- prophylaxis. Nc3 defends e4, and that outranks the fact that it also develops.
    ("prophylaxis", NXE4, "Nc3", "opening", "f6e4", "prophylaxis", True),
    # A move that ignores the threat is not prophylaxis, and names no idea.
    ("ignores the threat", NXE4, "a3", "opening", "f6e4", None, True),
    # With no threat supplied, Nc3 is ordinary development.
    ("no threat given", NXE4, "Nc3", "opening", None, "develop", True),
    # ...bxc3 threatens the knight. Bd2 guards it, but the pawn still wins a knight for a
    # pawn, so the threat is not stopped.
    (
        "a guard that loses the piece",
        "4k3/8/8/8/1p6/2N5/8/2B1K3 w - - 0 1",
        "Bd2",
        None,
        "b4c3",
        "prophylaxis",
        False,
    ),
    # A guard pinned to its own king takes nothing back: the g2 pawn guards the h3 bishop
    # ...Nxh3 threatens, but the d5 bishop pins it, so a quiet a4 stops nothing.
    (
        "a pinned guard",
        "4k3/8/8/3b4/5n2/P6B/6P1/7K w - - 0 1",
        "a4",
        None,
        "f4h3",
        "prophylaxis",
        False,
    ),
]


@pytest.mark.parametrize(
    ("case", "fen", "san", "phase", "threat", "kind", "is_it"), IDEAS, ids=[r[0] for r in IDEAS]
)
def test_the_best_move_s_idea(
    case: str,
    fen: str,
    san: str,
    phase: str | None,
    threat: str | None,
    kind: str | None,
    is_it: bool,
) -> None:
    board = chess.Board(fen)
    kw: dict[str, object] = {} if phase is None else {"phase": phase}
    if threat is not None:
        kw["threat"] = chess.Move.from_uci(threat)
    idea = P.best_move_idea(board, board.parse_san(san), chess.WHITE, **kw)  # type: ignore[arg-type]
    got = idea.kind if idea is not None else None
    assert (got == kind) is is_it, f"{case}: {got}"


def test_a_piece_is_in_danger_only_from_a_legal_capture() -> None:
    # The king cannot take a knight its pawn guards.
    guarded = chess.Board("8/8/5k2/4N3/3P4/8/8/4K3 w - - 0 1")
    assert not P._piece_is_attacked(guarded, chess.E5, chess.WHITE)
    # A knight pinned to its king takes nothing.
    pinned = chess.Board("4k3/8/2n5/1B6/3N4/8/8/4K3 w - - 0 1")
    assert not P._piece_is_attacked(pinned, chess.D4, chess.WHITE)
    # The d5 pawn does take the knight on c4.
    assert P._piece_is_attacked(chess.Board(KNIGHT_C4), chess.C4, chess.WHITE)


def test_top_features_are_ranked_deduped_and_capped() -> None:
    # A position rich in features: up material, a passed a-pawn, and open files.
    board = chess.Board("3r2k1/6pp/8/P7/8/8/6PP/3R2K1 w - - 0 1")
    top = P.top_features(board, chess.WHITE, phase="middlegame", limit=2)
    assert len(top) <= 2
    # Passed pawn outranks the open file, which outranks nothing here; no kind repeats.
    assert "passedPawn" in kinds(top)
    assert len(set(kinds(top))) == len(kinds(top))
    # A wing pawn majority surfaces as one of the position's ideas the coach can teach.
    majority = chess.Board("4k3/pp6/8/8/8/8/PPP5/4K3 w - - 0 1")
    assert "pawnMajority" in kinds(P.top_features(majority, chess.WHITE, phase="endgame"))


def test_pawn_weaknesses_name_an_isolated_pawn_the_enemy_s_first() -> None:
    # White c4 pawn, no white b- or d-pawn: isolated. (A lone d-pawn is the IQP, tested apart.)
    board = chess.Board("4k3/8/8/8/2P5/8/8/4K3 w - - 0 1")
    assert kinds(P.pawn_weaknesses(board, chess.WHITE)) == ["weakPawn"]
    assert P.is_isolated(board, chess.C4)
    # An enemy isolated pawn is a target, listed first. (A d-pawn would be the IQP instead.)
    board = chess.Board("4k3/8/8/2p5/8/8/8/4K3 w - - 0 1")
    facts = P.pawn_weaknesses(board, chess.WHITE)
    assert facts[0].kind == "targetPawn" and "target" in facts[0].text


def test_the_iqp_is_named_for_both_sides_and_its_plan_by_phase() -> None:
    # White has an isolated d4 pawn (no c- or e-pawn): an IQP.
    board = chess.Board("r1bq1rk1/pp3ppp/2n1pn2/8/3P4/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 10")
    assert P.is_isolated(board, chess.D4)
    assert kinds(P.isolated_d_pawn(board, chess.WHITE)) == ["iqp"]
    assert kinds(P.isolated_d_pawn(board, chess.BLACK)) == ["iqpTarget"]
    # The generic weak-pawn detector leaves the d-pawn to the IQP feature.
    assert "weakPawn" not in kinds(P.pawn_weaknesses(board, chess.WHITE))
    # Middlegame, owner: an asset -- the outposts it supports (c5, e5), keep pieces on to attack.
    owner = P.isolated_d_pawn(board, chess.WHITE, phase="middlegame")[0]
    assert "outpost" in owner.text and "keep pieces on" in owner.text
    assert set(owner.squares) == {"d4", "c5", "e5"}
    # Endgame, owner: the energy is spent, so it flips to a target to hold.
    endgame = P.isolated_d_pawn(board, chess.WHITE, phase="endgame")[0]
    assert "target" in endgame.text and "trade rooks" in endgame.text
    assert owner.text != endgame.text
    # The side facing it gets the blockade-and-trade plan, naming the blockade square (d5).
    defender = P.isolated_d_pawn(board, chess.BLACK)[0]
    assert "blockade it on d5" in defender.text and "trade the minor pieces" in defender.text
    assert defender.squares == ("d4", "d5")


def test_hanging_pawns_are_the_central_pair_and_the_plan_names_the_flanking_files() -> None:
    # White c4 and d4 abreast, no b- or e-pawn beside them: hanging pawns; the flanking b- and
    # e-files are the owner's to use.
    board = chess.Board("r1bq1rk1/pp3ppp/2n1pn2/8/2PP4/5N2/P4PPP/R1BQ1RK1 w - - 0 11")
    owners = P.hanging_pawns(board, chess.WHITE)
    assert kinds(owners) == ["hangingPawns"]
    assert "b- and e-files" in owners[0].text and "keep them mobile" in owners[0].text
    defenders = P.hanging_pawns(board, chess.BLACK)
    assert kinds(defenders) == ["hangingPawnsTarget"]
    assert "blockade" in defenders[0].text and "advance" in defenders[0].text


def test_a_passive_king_in_the_endgame_is_named() -> None:
    board = chess.Board("6k1/8/8/8/8/8/8/6K1 w - - 0 1")
    assert kinds(P.king_activity(board, chess.WHITE, phase="endgame")) == ["passiveKing"]


def test_central_infiltration_is_not_called_centralising() -> None:
    # A knight dropping to c2 (rank 2) is infiltration, not a central strong square (c3-f6).
    board = chess.Board("4k3/8/8/8/8/1n6/8/4K3 b - - 0 1")
    assert not P._is_central_strong(board, chess.C2, chess.BLACK)
    # But a knight to d4 is a central strong square.
    assert P._is_central_strong(board, chess.D4, chess.BLACK)


# A wing pawn majority: the side to play on, advancing it into a passed pawn. (case, FEN,
# whose majority, the wing named (None: no majority), its pawns.)
MAJORITIES: list[tuple[str, str, chess.Color, str | None, set[str] | None]] = [
    # White has a healthy 3-2 queenside majority (a,b,c vs a,b) and nothing on the kingside.
    (
        "queenside 3-2",
        "4k3/pp6/8/8/8/8/PPP5/4K3 w - - 0 1",
        chess.WHITE,
        "queenside",
        {"a2", "b2", "c2"},
    ),
    # Black has no majority anywhere.
    ("black has none", "4k3/pp6/8/8/8/8/PPP5/4K3 w - - 0 1", chess.BLACK, None, None),
    ("kingside 2-1", "4k3/7p/8/8/8/8/6PP/4K3 w - - 0 1", chess.WHITE, "kingside", {"g2", "h2"}),
    # Equal pawn counts on the queenside: no majority.
    ("equal", "4k3/pp6/8/8/8/8/PP6/4K3 w - - 0 1", chess.WHITE, None, None),
    # A 3-2 count that is a majority only because of doubled c-pawns cannot make a passer --
    # more pawns, but not more advancing files than the enemy can blockade -- so it is not
    # named (the book's caution about crippled majorities).
    ("crippled by doubled pawns", "4k3/1pp5/8/8/8/2P5/1PP5/4K3 w - - 0 1", chess.WHITE, None, None),
]


@pytest.mark.parametrize(
    ("case", "fen", "side", "wing", "pawns"), MAJORITIES, ids=[r[0] for r in MAJORITIES]
)
def test_a_wing_pawn_majority_is_named_as_the_side_to_play_on(
    case: str, fen: str, side: chess.Color, wing: str | None, pawns: set[str] | None
) -> None:
    facts = P.pawn_majority(chess.Board(fen), side)
    if wing is None:
        assert facts == [], case
        return
    assert kinds(facts) == ["pawnMajority"], case
    assert wing in facts[0].text and set(facts[0].squares) == pawns, case


def test_space_advantage_and_cramp_are_named_past_a_clear_margin() -> None:
    # King's Indian: White's c4/d4/e4 centre is a clear space advantage; Black is cramped.
    kid = chess.Board("rnbq1rk1/ppp1ppbp/3p1np1/8/2PPP3/2N2N2/PP2BPPP/R1BQK2R b KQ - 0 6")
    white = P.space_plan(kid, chess.WHITE)
    assert kinds(white) == ["spaceAdvantage"] and "avoid trades" in white[0].text
    assert set(white[0].squares) == {"c4", "d4", "e4"}
    black = P.space_plan(kid, chess.BLACK)
    assert kinds(black) == ["spaceCramp"] and "trade a pair" in black[0].text
    # A symmetric position is neither cramped nor spacious.
    assert P.space_plan(chess.Board(), chess.WHITE) == []
    symmetric = chess.Board("rnbqkbnr/pp2pppp/8/2pp4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3")
    assert P.space_plan(symmetric, chess.WHITE) == []


def test_the_bishop_pair_reads_the_openness_of_the_position() -> None:
    # White has both bishops, black has one: the pair. Open position (no central pawns): the
    # pair is a live asset -- keep lines open.
    open_pos = chess.Board("2b1k3/8/8/8/8/8/8/2B1KB2 w - - 0 1")
    pair = P.bishop_pair(open_pos, chess.WHITE)
    assert kinds(pair) == ["bishopPair"] and "position is open" in pair[0].text
    # Closed centre (locked d/e pawns): the plan is to open lines to activate the pair.
    closed = chess.Board("2b1k3/3pp3/8/8/8/8/3PP3/2B1KB2 w - - 0 1")
    assert "closed" in P.bishop_pair(closed, chess.WHITE)[0].text


def test_the_bad_bishop_names_its_three_cures() -> None:
    # A light-squared bishop on f1 walled in by four of its own pawns on light squares.
    board = chess.Board("4k3/8/8/8/2P1P3/1P6/6P1/4KB2 w - - 0 1")
    bad = P.bad_bishop(board, chess.WHITE)
    assert kinds(bad) == ["badBishop"]
    text = bad[0].text
    assert "trade it" in text and "off its colour" in text and "outside the pawn chain" in text


def test_pawn_pointing_names_the_wing_only_in_a_closed_centre() -> None:
    # French Advance: White's d4/e5 chain points to the kingside (push the f-pawn).
    french = chess.Board("r1bqkbnr/pp3ppp/2n1p3/2ppP3/3P4/2N5/PPP2PPP/R1BQKBNR w KQkq - 0 5")
    fact = P.pawn_pointing(french, chess.WHITE)[0]
    assert fact.kind == "pawnPointing" and "kingside" in fact.text and "f-pawn" in fact.text
    # King's Indian: Black's e5/d6 chain points to the kingside -- the ...f5 plan.
    kid = chess.Board("r1bq1rk1/ppp2pbp/2np1np1/4p3/2PPP3/2N2N2/PP2BPPP/R1BQ1RK1 b - - 0 7")
    assert "kingside" in P.pawn_pointing(kid, chess.BLACK)[0].text
    # Czech Benoni: White's d5/e4 chain points to the queenside (push the c-pawn).
    benoni = chess.Board("rnbqkb1r/pp3ppp/3p1n2/2pPp3/2P1P3/2N5/PP3PPP/R1BQKBNR w KQkq - 0 6")
    assert "queenside" in P.pawn_pointing(benoni, chess.WHITE)[0].text
    # An open centre with the d-pawns still home (1.e4 e5, Scholar's) says nothing.
    italian = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3")
    assert P.pawn_pointing(italian, chess.WHITE) == []
    assert P.pawn_pointing(chess.Board(), chess.WHITE) == []


def test_a_backward_pawn_is_a_weakness_only_on_a_half_open_file() -> None:
    # White d3 is backward (its advance is held by the black c5 pawn). Black has no d-pawn, so
    # the file is half-open and the pawn is a real target.
    board = chess.Board("4k3/8/8/2p5/2P1P3/3P4/8/4K3 w - - 0 1")
    assert P.is_backward(board, chess.D3)
    assert any(
        f.kind == "targetPawn" and f.squares == ("d3",) and "backward" in f.text
        for f in P.pawn_weaknesses(board, chess.BLACK)
    )
    # A black d-pawn on the file shields it: still backward, but no longer attackable down the
    # file, so it is not named a weakness.
    shielded = chess.Board("4k3/3p4/8/2p5/2P1P3/3P4/8/4K3 w - - 0 1")
    assert P.is_backward(shielded, chess.D3)
    assert not any(f.squares == ("d3",) for f in P.pawn_weaknesses(shielded, chess.BLACK))


def test_doubled_pawns_name_the_half_open_file_they_hand_the_owner() -> None:
    # White has doubled c-pawns (c2, c3) and no b-pawn: the b-file is half-open for the rooks.
    board = chess.Board("4k3/8/8/8/8/2P5/P1P3PP/4K3 w - - 0 1")
    doubled = [f for f in P.pawn_weaknesses(board, chess.WHITE) if "doubled" in f.text]
    assert doubled and "half-open b-file" in doubled[0].text


def test_square_of_the_pawn_names_the_decisive_race() -> None:
    # White a5-pawn, both kings on the kingside: Black's king is outside the a-pawn's square.
    board = chess.Board("7k/8/8/P7/8/8/8/7K w - - 0 1")
    facts = P.square_of_the_pawn(board, chess.WHITE, phase="endgame")
    assert kinds(facts) == ["runningPawn"] and "a-pawn" in facts[0].text
    # From Black's side, the same pawn outruns Black's king.
    assert kinds(P.square_of_the_pawn(board, chess.BLACK, phase="endgame")) == ["outrunPawn"]
    # A king inside the square catches the pawn: nothing decisive to say.
    close = chess.Board("8/8/1k6/P7/8/8/8/7K w - - 0 1")
    assert P.square_of_the_pawn(close, chess.WHITE, phase="endgame") == []
    # Only fires in the endgame, where a lone king must chase a pawn.
    assert P.square_of_the_pawn(board, chess.WHITE, phase="middlegame") == []


def test_the_opposition_is_named_in_king_and_pawn_endgames() -> None:
    # Kings facing on the e-file two squares apart, White to move: Black holds the opposition.
    board = chess.Board("8/8/4k3/8/4K3/8/4P3/8 w - - 0 1")
    assert kinds(P.opposition(board, chess.BLACK)) == ["opposition"]
    assert "must give way" in P.opposition(board, chess.BLACK)[0].text
    assert kinds(P.opposition(board, chess.WHITE)) == ["oppositionAgainst"]
    # With Black to move, White takes the opposition.
    board.turn = chess.BLACK
    assert kinds(P.opposition(board, chess.WHITE)) == ["opposition"]
    # Kings not on a line with an odd gap between them: no opposition.
    off = chess.Board("8/8/5k2/8/4K3/8/4P3/8 w - - 0 1")
    assert P.opposition(off, chess.WHITE) == []
    # Not a pure king-and-pawn endgame (a rook is on): the opposition is not the battle.
    with_piece = chess.Board("8/8/4k3/8/4K3/8/4P3/4R3 w - - 0 1")
    assert P.opposition(with_piece, chess.WHITE) == []


def test_key_squares_in_king_and_pawn_versus_king() -> None:
    # White c4-pawn: its key squares are b6/c6/d6. The king on c6 holds one -> the pawn queens.
    on_key = chess.Board("4k3/8/2K5/8/2P5/8/8/8 w - - 0 1")
    assert kinds(P.key_squares(on_key, chess.WHITE)) == ["keySquare"]
    assert "key square on c6" in P.key_squares(on_key, chess.WHITE)[0].text
    # King not yet on a key square: the goal names the squares to march to.
    off_key = chess.Board("4k3/8/8/8/2P5/8/2K5/8 w - - 0 1")
    goal = P.key_squares(off_key, chess.WHITE)[0]
    assert goal.kind == "keySquareGoal" and "b6, c6, d6" in goal.text
    # A rook pawn has no winning key squares (the ending is usually a draw).
    rook = chess.Board("4k3/8/2K5/8/P7/8/8/8 w - - 0 1")
    assert P.key_squares(rook, chess.WHITE) == []


def test_lucena_position_is_recognised_as_a_won_rook_ending() -> None:
    # King on the queening square d8, pawn d7, enemy king cut off on the f-file: a Lucena win.
    board = chess.Board("3K4/3P4/5k2/8/8/8/8/R6r w - - 0 1")
    assert kinds(P.lucena(board, chess.WHITE)) == ["lucena"]
    assert "build a bridge" in P.lucena(board, chess.WHITE)[0].text
    # A rook pawn is not won by the bridge, so it is not flagged as a Lucena.
    rook = chess.Board("K7/P7/2k5/8/8/8/8/R6r w - - 0 1")
    assert P.lucena(rook, chess.WHITE) == []
    # A queen on the board: not a pure rook-and-pawn ending.
    with_queen = chess.Board("3K4/3P4/5k2/8/8/8/7q/R6r w - - 0 1")
    assert P.lucena(with_queen, chess.WHITE) == []


def test_philidor_position_is_recognised_as_the_drawing_defence() -> None:
    # Black defends: king in front of White's e5-pawn, rook on the 6th rank (Black's third).
    board = chess.Board("8/4k3/r7/4P3/4K3/8/8/7R b - - 0 1")
    assert kinds(P.philidor(board, chess.BLACK)) == ["philidor"]
    assert "third rank" in P.philidor(board, chess.BLACK)[0].text
    # With the rook off the third rank the drawing defence is not set up.
    broken = chess.Board("8/4k3/8/4P3/4K3/8/8/r6R b - - 0 1")
    assert P.philidor(broken, chess.BLACK) == []
