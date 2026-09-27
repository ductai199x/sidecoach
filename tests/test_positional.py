"""Tests for the positional layer: static strategic features and the best-move idea.

Every fact these produce is deterministic geometry, so each test is a position where the
answer is unambiguous. The point is that the coach's strategic facts are as grounded and
un-fabricatable as the tactical ones.
"""

from __future__ import annotations

import chess

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


def _idea_kind(board: chess.Board, san: str, us: chess.Color, **kw) -> str | None:
    idea = P.best_move_idea(board, board.parse_san(san), us, **kw)
    return idea.kind if idea is not None else None


def test_best_move_idea_reads_the_strategic_point_of_the_move() -> None:
    italian = "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
    board = chess.Board(italian)
    assert _idea_kind(board, "O-O", chess.WHITE, phase="opening") == "castle"
    assert _idea_kind(board, "Nc3", chess.WHITE, phase="opening") == "develop"
    # A knight to a pawn-guarded square in enemy ground is an outpost move.
    board = chess.Board("4k3/8/8/8/4P3/2N5/8/4K3 w - - 0 1")
    assert _idea_kind(board, "Nd5", chess.WHITE) == "outpostMove"
    # A rook onto a file with no friendly pawn takes the file.
    board = chess.Board("3rk3/8/8/8/8/8/4P3/R3K3 w - - 0 1")
    assert _idea_kind(board, "Rd1", chess.WHITE) == "takeFile"
    # Pushing a passed pawn.
    board = chess.Board("4k3/8/8/P7/8/8/8/4K3 w - - 0 1")
    assert _idea_kind(board, "a6", chess.WHITE) == "pushPasser"


def test_best_move_idea_rescues_a_piece_in_danger() -> None:
    # The bishop on b5 is attacked by the a6 pawn; moving it away is a rescue, not development.
    board = chess.Board("rnbqkbnr/1ppp1ppp/p7/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3")
    axb5 = chess.Move.from_uci("a6b5")
    idea = P.best_move_idea(
        board, board.parse_san("Ba4"), chess.WHITE, phase="opening", threat=axb5
    )
    assert idea is not None and idea.kind == "rescue"


def test_a_rescue_is_of_the_piece_their_threat_goes_after() -> None:
    """A piece that could be taken is in danger only when taking it is their threat (the
    engine's best move for them, were it theirs): "a piece in danger comes first" was said of
    a rook check that led to mate, and of moves whose threat was elsewhere (the lesson audit,
    2026-09-27). A pawn that steps on from an attack is a push, not a rescue."""
    board = chess.Board("rnbqkbnr/1ppp1ppp/p7/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 3")
    ba4 = board.parse_san("Ba4")
    assert P.best_move_idea(board, ba4, chess.WHITE, phase="opening") is None or (
        _idea_kind(board, "Ba4", chess.WHITE, phase="opening") != "rescue"
    )
    elsewhere = chess.Move.from_uci("d8h4")  # their threat is not the bishop
    assert _idea_kind(board, "Ba4", chess.WHITE, phase="opening", threat=elsewhere) != "rescue"
    # The e4 pawn attacked by the d5 pawn steps on to e5: no rescue.
    pawn = chess.Board("4k3/8/8/3p4/4P3/8/8/4K3 w - - 0 1")
    dxe4 = chess.Move.from_uci("d5e4")
    assert _idea_kind(pawn, "e5", chess.WHITE, threat=dxe4) != "rescue"


def test_a_rescue_needs_a_piece_that_can_be_won_and_a_safe_square() -> None:
    # A knight beside the enemy king, guarded by its pawn: the king cannot take it, so it is
    # not in danger (the king used to count as a capturer worth nothing).
    guarded = chess.Board("8/8/5k2/4N3/3P4/8/8/4K3 w - - 0 1")
    assert not P._piece_is_attacked(guarded, chess.E5, chess.WHITE)
    assert _idea_kind(guarded, "Nd3", chess.WHITE) != "rescue"
    # A knight attacked only by the c6 knight, which the b5 bishop pins to its king.
    pinned = chess.Board("4k3/8/2n5/1B6/3N4/8/8/4K3 w - - 0 1")
    assert not P._piece_is_attacked(pinned, chess.D4, chess.WHITE)
    assert _idea_kind(pinned, "Nf3", chess.WHITE) != "rescue"
    # The d5 pawn attacks the knight, so it is in danger; running to b6, where the a7 pawn
    # takes it, is no rescue, and e3 is.
    board = chess.Board("4k3/p7/8/3p4/2N5/8/8/4K3 w - - 0 1")
    assert P._piece_is_attacked(board, chess.C4, chess.WHITE)
    dxc4 = chess.Move.from_uci("d5c4")
    assert _idea_kind(board, "Nb6", chess.WHITE, threat=dxc4) != "rescue"
    assert _idea_kind(board, "Ne3", chess.WHITE, threat=dxc4) == "rescue"


def test_top_features_are_ranked_deduped_and_capped() -> None:
    # A position rich in features: up material, a passed a-pawn, and open files.
    board = chess.Board("3r2k1/6pp/8/P7/8/8/6PP/3R2K1 w - - 0 1")
    top = P.top_features(board, chess.WHITE, phase="middlegame", limit=2)
    assert len(top) <= 2
    # Passed pawn outranks the open file, which outranks nothing here; no kind repeats.
    assert "passedPawn" in kinds(top)
    assert len(set(kinds(top))) == len(kinds(top))


def test_pawn_weaknesses_name_isolated_doubled_and_backward_pawns() -> None:
    # White c4 pawn, no white b- or d-pawn: isolated. (A lone d-pawn is the IQP, tested apart.)
    board = chess.Board("4k3/8/8/8/2P5/8/8/4K3 w - - 0 1")
    assert kinds(P.pawn_weaknesses(board, chess.WHITE)) == ["weakPawn"]
    assert P.is_isolated(board, chess.C4)
    # An enemy isolated pawn is a target, listed first. (A d-pawn would be the IQP instead.)
    board = chess.Board("4k3/8/8/2p5/8/8/8/4K3 w - - 0 1")
    facts = P.pawn_weaknesses(board, chess.WHITE)
    assert facts[0].kind == "targetPawn" and "target" in facts[0].text
    # Backward: white d3 behind c4 and e4, its advance to d4 held by the black c5 pawn.
    board = chess.Board("4k3/8/8/2p5/2P1P3/3P4/8/4K3 w - - 0 1")
    assert P.is_backward(board, chess.D3)


def test_isolated_queen_pawn_is_named_for_both_sides() -> None:
    # White has an isolated d4 pawn (no c- or e-pawn): an IQP.
    board = chess.Board("r1bq1rk1/pp3ppp/2n1pn2/8/3P4/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 10")
    assert P.is_isolated(board, chess.D4)
    assert kinds(P.isolated_d_pawn(board, chess.WHITE)) == ["iqp"]
    assert kinds(P.isolated_d_pawn(board, chess.BLACK)) == ["iqpTarget"]
    # The generic weak-pawn detector leaves the d-pawn to the IQP feature.
    assert "weakPawn" not in kinds(P.pawn_weaknesses(board, chess.WHITE))


def test_hanging_pawns_are_the_central_pair_on_half_open_files() -> None:
    # White c4 and d4 abreast, no b- or e-pawn beside them: hanging pawns.
    board = chess.Board("r1bq1rk1/pp3ppp/2n1pn2/8/2PP4/5N2/P4PPP/R1BQ1RK1 w - - 0 11")
    assert kinds(P.hanging_pawns(board, chess.WHITE)) == ["hangingPawns"]
    assert kinds(P.hanging_pawns(board, chess.BLACK)) == ["hangingPawnsTarget"]


def test_bishop_pair_and_bad_bishop() -> None:
    # White has both bishops, black has one: the pair.
    board = chess.Board("2b1k3/8/8/8/8/8/8/2B1KB2 w - - 0 1")
    assert kinds(P.bishop_pair(board, chess.WHITE)) == ["bishopPair"]
    # A light-squared bishop on f1 walled in by four of its own pawns on light squares.
    board = chess.Board("4k3/8/8/8/2P1P3/1P6/6P1/4KB2 w - - 0 1")
    assert kinds(P.bad_bishop(board, chess.WHITE)) == ["badBishop"]


def test_endgame_king_activity_and_pawn_break_and_rook_behind_passer() -> None:
    # Endgame: a passive King on the rim is named, and a King step toward the centre is the idea.
    board = chess.Board("6k1/8/8/8/8/8/8/6K1 w - - 0 1")
    assert kinds(P.king_activity(board, chess.WHITE, phase="endgame")) == ["passiveKing"]
    board = chess.Board("8/8/8/8/8/8/4K3/6k1 w - - 0 1")
    assert _idea_kind(board, "Kd3", chess.WHITE, phase="endgame") == ("activateKing")
    # A pawn push that hits an enemy pawn is a break.
    board = chess.Board("4k3/8/3p4/8/2P5/8/8/4K3 w - - 0 1")
    assert _idea_kind(board, "c5", chess.WHITE) == "break"
    # A rook dropping behind a passed pawn is the classic rook-behind-passer idea.
    board = chess.Board("4k3/8/8/P7/8/8/8/1R2K3 w - - 0 1")
    assert _idea_kind(board, "Ra1", chess.WHITE) == "rookBehindPasser"


def test_pawn_storm_is_a_pawn_push_at_the_castled_king() -> None:
    # Black has castled kingside (g8); g4 storms that wing, a4 is only queenside space.
    board = chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP1QPPP/R1B1K2R w KQ - 0 7")
    assert _idea_kind(board, "g4", chess.WHITE) == "pawnStorm"
    assert _idea_kind(board, "a4", chess.WHITE) == "space"


def test_prophylaxis_is_a_quiet_move_that_stops_the_opponent_s_threat() -> None:
    # Black threatens ...Nxe4: the e4 pawn hangs to the f6 knight. Nc3 defends it -- prophylaxis,
    # and it outranks the fact that Nc3 also develops.
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4")
    threat = chess.Move.from_uci("f6e4")
    idea = P.best_move_idea(
        board, board.parse_san("Nc3"), chess.WHITE, phase="opening", threat=threat
    )
    assert idea is not None and idea.kind == "prophylaxis"
    # A move that ignores the threat is not prophylaxis.
    assert (
        P.best_move_idea(board, board.parse_san("a3"), chess.WHITE, phase="opening", threat=threat)
        is None
    )
    # With no threat supplied, Nc3 is read as ordinary development, not prophylaxis.
    assert _idea_kind(board, "Nc3", chess.WHITE, phase="opening") == ("develop")


def test_a_guard_that_still_loses_the_piece_is_not_prophylaxis() -> None:
    # ...bxc3 threatens the knight. Bd2 guards it, but the pawn still wins a knight for a
    # pawn, so the threat is not stopped.
    board = chess.Board("4k3/8/8/8/1p6/2N5/8/2B1K3 w - - 0 1")
    threat = chess.Move.from_uci("b4c3")
    assert _idea_kind(board, "Bd2", chess.WHITE, threat=threat) != "prophylaxis"
    # A guard pinned to its own king takes nothing back: the g2 pawn guards the h3 bishop
    # ...Nxh3 threatens, but the d5 bishop pins it, so a quiet a4 stops nothing.
    board = chess.Board("4k3/8/8/3b4/5n2/P6B/6P1/7K w - - 0 1")
    threat = chess.Move.from_uci("f4h3")
    assert _idea_kind(board, "a4", chess.WHITE, threat=threat) != "prophylaxis"


def test_central_infiltration_is_not_called_centralising() -> None:
    # A knight dropping to c2 (rank 2) is infiltration, not a central strong square (c3-f6).
    board = chess.Board("4k3/8/8/8/8/1n6/8/4K3 b - - 0 1")
    assert not P._is_central_strong(board, chess.C2, chess.BLACK)
    # But a knight to d4 is a central strong square.
    assert P._is_central_strong(board, chess.D4, chess.BLACK)


def test_a_wing_pawn_majority_is_named_as_the_side_to_play_on() -> None:
    # White has a healthy 3-2 queenside majority (a,b,c vs a,b) and nothing on the kingside;
    # Black has no majority anywhere. The plan is to advance it into a passed pawn.
    board = chess.Board("4k3/pp6/8/8/8/8/PPP5/4K3 w - - 0 1")
    facts = P.pawn_majority(board, chess.WHITE)
    assert kinds(facts) == ["pawnMajority"]
    assert "queenside" in facts[0].text
    assert set(facts[0].squares) == {"a2", "b2", "c2"}
    assert P.pawn_majority(board, chess.BLACK) == []
    # It surfaces as one of the position's ideas the coach can teach.
    assert "pawnMajority" in kinds(P.top_features(board, chess.WHITE, phase="endgame"))


def test_a_kingside_two_to_one_majority_is_also_named() -> None:
    board = chess.Board("4k3/7p/8/8/8/8/6PP/4K3 w - - 0 1")
    facts = P.pawn_majority(board, chess.WHITE)
    assert kinds(facts) == ["pawnMajority"] and "kingside" in facts[0].text
    assert set(facts[0].squares) == {"g2", "h2"}


def test_an_equal_wing_and_a_crippled_doubled_majority_are_not_named() -> None:
    # Equal pawn counts on the queenside: no majority.
    equal = chess.Board("4k3/pp6/8/8/8/8/PP6/4K3 w - - 0 1")
    assert P.pawn_majority(equal, chess.WHITE) == []
    # A 3-2 count that is a majority only because of doubled c-pawns cannot make a passer --
    # more pawns, but not more advancing files than the enemy can blockade -- so it is not
    # named (the book's caution about crippled majorities).
    crippled = chess.Board("4k3/1pp5/8/8/8/2P5/1PP5/4K3 w - - 0 1")
    assert P.pawn_majority(crippled, chess.WHITE) == []


def test_the_iqp_plan_is_named_by_side_and_phase() -> None:
    # White has an isolated d4 pawn: a classic IQP.
    board = chess.Board("r1bq1rk1/pp3ppp/2n1pn2/8/3P4/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 10")
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


def test_the_hanging_pawns_plan_names_the_flanking_files_for_both_sides() -> None:
    # White has c4 and d4 hanging pawns; the flanking b- and e-files are the owner's to use.
    board = chess.Board("r1bq1rk1/pp3ppp/2n1pn2/8/2PP4/5N2/P4PPP/R1BQ1RK1 w - - 0 11")
    owner = P.hanging_pawns(board, chess.WHITE)[0]
    assert owner.kind == "hangingPawns"
    assert "b- and e-files" in owner.text and "keep them mobile" in owner.text
    defender = P.hanging_pawns(board, chess.BLACK)[0]
    assert defender.kind == "hangingPawnsTarget"
    assert "blockade" in defender.text and "advance" in defender.text


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
    # Open position (no central pawns): the pair is a live asset -- keep lines open.
    open_pos = chess.Board("2b1k3/8/8/8/8/8/8/2B1KB2 w - - 0 1")
    assert "position is open" in P.bishop_pair(open_pos, chess.WHITE)[0].text
    # Closed centre (locked d/e pawns): the plan is to open lines to activate the pair.
    closed = chess.Board("2b1k3/3pp3/8/8/8/8/3PP3/2B1KB2 w - - 0 1")
    assert "closed" in P.bishop_pair(closed, chess.WHITE)[0].text


def test_the_bad_bishop_names_its_three_cures() -> None:
    board = chess.Board("4k3/8/8/8/2P1P3/1P6/6P1/4KB2 w - - 0 1")
    text = P.bad_bishop(board, chess.WHITE)[0].text
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
    assert any("backward d-pawn" in f.text for f in P.pawn_weaknesses(board, chess.BLACK))
    # A black d-pawn on the file shields it: still backward, but no longer attackable down the
    # file, so it is not named a weakness.
    shielded = chess.Board("4k3/3p4/8/2p5/2P1P3/3P4/8/4K3 w - - 0 1")
    assert P.is_backward(shielded, chess.D3)
    assert not any("backward d-pawn" in f.text for f in P.pawn_weaknesses(shielded, chess.BLACK))


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
