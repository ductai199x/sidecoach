"""The word on a sound move written from the board (coach/verdicts.py): the verdict for
the label, then what the move does when the board says so, never the same opener twice
running, and no model call for it in the coach."""

from __future__ import annotations

import random

import chess

from sidecoach.coach.verdicts import OPENERS, verdict, what_it_does
from sidecoach.events import MotifTag

SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def _uci(fen: str, uci: str) -> tuple[chess.Board, chess.Move]:
    return chess.Board(fen), chess.Move.from_uci(uci)


def test_the_clause_says_what_the_board_shows() -> None:
    board, mate = _uci(SCHOLAR, "h5f7")
    assert what_it_does(board, mate, []) == "checkmate"
    board, push = _uci("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "d7d6")
    assert what_it_does(board, push, []) is None, "a quiet push says nothing"
    board, nxe5 = _uci("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f3e5")
    assert what_it_does(board, nxe5, []) == "taking the pawn on e5"
    board, castle = _uci(
        "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "e1g1"
    )
    assert what_it_does(board, castle, []) == "castling kingside"
    board, check = _uci("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "d1h5")
    assert what_it_does(board, check, []) is None, "Qh5 gives no check here"
    board, qh5 = _uci("rnbqkbnr/pppp2pp/8/4pp2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "d1h5")
    assert what_it_does(board, qh5, []) == "with check"
    board, promote = _uci("8/1P4k1/8/8/8/8/8/4K3 w - - 0 1", "b7b8q")
    assert what_it_does(board, promote, []) == "promoting to a queen"
    # A tag's tactic, described only when the squares bear it out after the move.
    board, fork = _uci("6k1/5ppp/8/r7/3r4/8/5PPP/2N3K1 w - - 0 1", "c1b3")
    tag = MotifTag(name="fork", squares=["b3", "a5", "d4"])
    assert what_it_does(board, fork, [tag]) == (
        "the knight on b3 forks the rook on a5 and the rook on d4"
    )
    stale = MotifTag(name="fork", squares=["b3", "a2", "d4"])  # nothing on a2: not said
    assert what_it_does(board, fork, [stale]) is None


def test_the_verdict_joins_the_opener_and_the_clause_and_varies() -> None:
    board, mate = _uci(SCHOLAR, "h5f7")
    text = verdict(board, mate, "best", rng=random.Random(3))
    assert text.startswith("Qxf7#") and text.endswith(" -- checkmate.")
    assert text.split(" -- checkmate.")[0].rstrip(".") in [
        o.format(san="Qxf7#") for o in OPENERS["best"]
    ]
    board, nxe5 = _uci("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "f3e5")
    assert verdict(board, nxe5, "good", rng=random.Random(1)).endswith(", taking the pawn on e5.")
    board, fork = _uci("6k1/5ppp/8/r7/3r4/8/5PPP/2N3K1 w - - 0 1", "c1b3")
    tag = MotifTag(name="fork", squares=["b3", "a5", "d4"])
    said = verdict(board, fork, "best", [tag], rng=random.Random(1))
    assert said.endswith(" -- the knight on b3 forks the rook on a5 and the rook on d4.")
    # Quiet: the opener alone. Never the same opener twice running; a book move says book.
    board, nf3 = _uci(chess.STARTING_FEN, "g1f3")
    seen = {verdict(board, nf3, "good", rng=random.Random(i)) for i in range(40)}
    assert len(seen) >= 3 and all(s.startswith("Nf3") and s.endswith(".") for s in seen)
    previous = None
    for i in range(40):
        current = verdict(board, nf3, "good", last=previous, rng=random.Random(i))
        assert current != previous
        previous = current
    assert verdict(board, chess.Move.from_uci("e2e4"), "book", rng=random.Random(0)) in [
        o.format(san="e4") + "." for o in OPENERS["book"]
    ]
    for text in seen:
        assert "?" not in text and not any(ch.isdigit() for ch in text.replace("Nf3", ""))


def test_the_coach_writes_a_sound_moves_word_itself_and_calls_no_model() -> None:
    from tests.test_coach import (
        _coach,
        _mate_analysis,
        _mate_in_one_played,
        _messages,
        _play_to_scholar,
    )

    coach, provider = _coach(["never used"])
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    _mate_in_one_played(coach)
    out = _messages(coach)
    assert len(out) == 1 and out[0].text.startswith("Qxf7#") and "checkmate" in out[0].text
    assert provider.prompts == [], "no model call for the engine's own move"
    # Off, the model writes it as before.
    coach, provider = _coach(["[p=1.0] Qxf7# was best, mate on the board."])
    coach.local_verdicts = False
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    _mate_in_one_played(coach)
    out = _messages(coach)
    assert len(out) == 1 and out[0].text == "Qxf7# was best, mate on the board."
    assert len(provider.prompts) == 1
