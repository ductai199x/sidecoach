"""The draft's claims decided by the board (`coach/draft_claims.py`), on the false claims the
locked test set's audit found (2026-09-26) and their true twins. No model: the claims are
written as the extractor would give them, each with the words it came from."""

from __future__ import annotations

import chess

from sidecoach.coach.draft_claims import DraftClaim, parse, verify


def _claim(kind: str, after: tuple[str, ...] = (), quote: str = "", **fields: object) -> DraftClaim:
    return DraftClaim(kind=kind, after=after, quote=quote, fields=dict(fields))


def test_a_capture_said_of_a_piece_that_has_moved_is_false() -> None:
    """Rg3 Bg4 Ne7+: "Ne7+ wins back the bishop on f5" -- the bishop is on g4 by then."""
    board = chess.Board("5rk1/pp3pp1/2p4p/3N1b1q/2P2P2/1P2R3/PB4PP/4R1K1 w - - 0 22")
    wrong = _claim(
        "piece",
        ("Rg3", "Bg4"),
        "Ne7+ wins back the bishop on f5",
        side=None,
        piece="bishop",
        square="f5",
    )
    assert verify([wrong], board, chess.WHITE)
    # "Wins" is no capture in so many words: decided by where the bishop stands, not by Ne7+.
    wins = _claim(
        "move_does",
        ("Rg3", "Bg4"),
        "Ne7+ wins back the bishop on f5",
        move="Ne7+",
        does="takes",
        target="f5",
    )
    assert verify([wins], board, chess.WHITE) == []
    pin = _claim("move_does", (), "Rg3 attacks g7", move="Rg3", does="attacks", target="g7")
    assert verify([pin], board, chess.WHITE) == []


def test_whose_move_and_whose_piece() -> None:
    """In Qxa7 c3 Rf2 b4, b4 is the opponent's; the bishop on d5 is the player's."""
    board = chess.Board("1r3rk1/p4pp1/3p1q1p/1p1BnP2/2p1P3/1P2Q2P/P5P1/3R1RK1 w - - 0 22")
    theirs = _claim("owns_move", ("Qxa7", "c3", "Rf2"), "your b4", move="b4", side="you")
    assert verify([theirs], board, chess.WHITE)
    ok = _claim("owns_move", ("Qxa7", "c3", "Rf2"), "their b4", move="b4", side="opponent")
    assert verify([ok], board, chess.WHITE) == []
    placed = _claim("piece", (), "their bishop on d5", side="opponent", piece="bishop", square="d5")
    assert verify([placed], board, chess.WHITE)


def test_only_moves_forced_squares_and_mates() -> None:
    board = chess.Board("6k1/5pp1/p2p4/1p3Pb1/2p1r1q1/P1P1p3/1P5K/4R3 b - - 1 48")
    forced = _claim(
        "forced", ("Qh4+", "Kg2", "Qf2+"), "drives the King to h3", piece="king", square="h3"
    )
    assert verify([forced], board, chess.BLACK), "Kh1 is legal too"
    only = _claim("only_move", ("Qh4+",), "Kg2 is the only king move", move="Kg2", only="king")
    assert verify([only], board, chess.BLACK), "Kg1 is legal too"
    forced_king = _claim(
        "only_move", ("Qh4+",), "only a king move gets out of check", move="Kg2", only="king"
    )
    assert verify([forced_king], board, chess.BLACK) == [], "true: every way out is a king's"
    kept = _claim(
        "only_move", (), "Qh4+ is the only move that keeps the win", move="Qh4+", only="legal"
    )
    assert verify([kept], board, chess.BLACK) == [], "the stakes, not the legal moves"
    mate = _claim("mate", ("Qh4+", "Kg2", "Qf2+", "Kh3"), "Rh4# is mate", move="Rh4#")
    assert verify([mate], board, chess.BLACK) == []
    not_mate = _claim("mate", ("Qh4+", "Kg2"), "Qf2+ is mate", move="Qf2+")
    assert verify([not_mate], board, chess.BLACK)


def test_allows_against_legality_and_stops_left_alone() -> None:
    """Ne4 cannot allow exf6. "Before they could play c6" read two ways in the audit, so a
    "stops" claim is not decided."""
    knight = chess.Board("r1bqk2r/pppp1pp1/2n2n1p/4P3/2BP4/5N2/P1P2PPP/R1BQK2R b KQkq - 0 8")
    allows = _claim("allows", (), "Ne4 allows exf6", move="Ne4", target_move="exf6")
    assert verify([allows], knight, chess.BLACK)
    board = chess.Board("4rrk1/p1p2p2/1p5p/1Q1p2q1/P2P3p/2P2P1P/6P1/R4RK1 w - - 1 22")
    stops = _claim("stops", (), "before they can use Re3", move="Rae1", target_move="Re3")
    assert verify([stops], board, chess.WHITE, ["Rae1", "Re3", "Qc6"]) == []


def test_states_and_files_in_so_many_words() -> None:
    board = chess.Board("2qr2k1/5pp1/p2Qb2p/1p2Pp2/1P1NrP2/P1P3RP/6P1/5R1K w - - 1 38")
    loose = _claim(
        "state",
        (),
        "the loose rook on e4",
        side="opponent",
        piece="rook",
        square="e4",
        state="loose",
    )
    assert verify([loose], board, chess.WHITE), "the pawn on f5 defends it"
    said = _claim(
        "state", (), "the rook on e4 is attacked twice", piece="rook", square="e4", state="loose"
    )
    assert verify([said], board, chess.WHITE) == [], "the quote never says loose"
    g6 = chess.Board("rn2k3/pp2bp2/2p3p1/4Q2p/P1P5/1P3BK1/7R/RN4q1 w q - 7 33")
    g_file = _claim("file", (), "the open g-file", file="g", state="open")
    assert verify([g_file], g6, chess.WHITE)
    d_file = _claim("file", (), "the open d-file", file="d", state="open")
    assert verify([d_file], g6, chess.WHITE) == []


def test_a_claim_that_cannot_be_placed_or_is_not_in_its_words_says_nothing() -> None:
    board = chess.Board()
    lost = _claim(
        "move_does", ("Nf3", "Qxz9"), "e4 hits d5", move="e4", does="attacks", target="d5"
    )
    assert verify([lost], board, chess.WHITE) == []
    assert verify([_claim("mystery")], board, chess.WHITE) == []
    invented = _claim("piece", (), "the skewer wins", side="you", piece="queen", square="a2")
    assert verify([invented], board, chess.WHITE) == [], "a2 is nowhere in the quote"


def test_a_claim_about_the_last_move_holds_on_the_position_before_it() -> None:
    """ "Nxd7 takes your knight on d7", said after Nxd7: true of the position it was played
    from."""
    before = chess.Board("r1bq1rk1/pppn1ppp/5n2/3pN3/8/8/PPP2PPP/RNBQ1RK1 w - - 0 9")
    after = before.copy()
    after.push_san("Nxd7")
    claim = _claim("piece", (), "your knight on d7", side="you", piece="knight", square="d7")
    assert verify([claim], after, chess.BLACK)
    assert verify([claim], after, chess.BLACK, before=before) == []


def test_the_extractors_reply_is_read_fenced_or_not_and_junk_is_none() -> None:
    text = '```json\n[{"kind": "mate", "after": ["Qh4+"], "quote": "Qg7#", "move": "Qg7#"}]\n```'
    (claim,) = parse(text)
    assert claim.kind == "mate" and claim.after == ("Qh4+",) and claim.fields["move"] == "Qg7#"
    assert parse("no claims here") == [] and parse("[1, 2]") == []
