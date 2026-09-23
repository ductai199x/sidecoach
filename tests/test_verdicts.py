"""The word on a sound move written from the board (coach/verdicts.py): praise for the
label, then what the move teaches when the board shows it -- never the move itself nor what
it plainly did -- never the same praise twice running, and no model call for it in the
coach."""

from __future__ import annotations

import random

import chess

from sidecoach.coach.verdicts import MATE, OPENERS, PRINCIPLES, lesson, verdict
from sidecoach.events import MotifTag

SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
# What the player sees on the board for themselves: never said back to them (Tai,
# 2026-09-23).
PLAIN = ("taking the", "takes the", "with check", "castling", "castles", "promoting", "promotes")


def _uci(fen: str, uci: str) -> tuple[chess.Board, chess.Move]:
    return chess.Board(fen), chess.Move.from_uci(uci)


def test_the_lesson_is_what_the_board_teaches_not_what_the_move_did() -> None:
    board, mate = _uci(SCHOLAR, "h5f7")
    assert lesson(board, mate) == ("mate", "checkmate")
    # A capture, a check, a promotion: said by the board, not by the coach.
    board, check = _uci("rnbqkbnr/pppp2pp/8/4pp2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "d1h5")
    assert lesson(board, check, phase="opening") is None
    board, push = _uci("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "d7d6")
    assert lesson(board, push, phase="opening") is None, "no idea the board can name"
    # The idea behind the move, as the principle it follows.
    board, castle = _uci(
        "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "e1g1"
    )
    assert lesson(board, castle, phase="opening") == ("castle", PRINCIPLES["castle"][0])
    assert lesson(board, castle, phase="opening", turn=1) == ("castle", PRINCIPLES["castle"][1])
    board, rook = _uci("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1d1")
    assert lesson(board, rook, phase="endgame") == ("takeFile", "rooks belong on open files")
    board, e4 = _uci(chess.STARTING_FEN, "e2e4")
    assert lesson(board, e4, phase="opening") == ("centre", PRINCIPLES["centre"][0])
    # A tag's tactic, described only when the squares bear it out after the move.
    board, fork = _uci("6k1/5ppp/8/r7/3r4/8/5PPP/2N3K1 w - - 0 1", "c1b3")
    tag = MotifTag(name="fork", squares=["b3", "a5", "d4"])
    assert lesson(board, fork, [tag]) == (
        "fork",
        "the knight on b3 forks the rook on a5 and the rook on d4",
    )
    stale = MotifTag(name="fork", squares=["b3", "a2", "d4"])  # nothing on a2: not said
    assert lesson(board, fork, [stale]) != lesson(board, fork, [tag])
    # A lesson taught in one of the last few words waits its turn.
    assert lesson(board, fork, [tag], taught=["fork"]) != lesson(board, fork, [tag])
    board, e4 = _uci(chess.STARTING_FEN, "e2e4")
    assert lesson(board, e4, phase="opening", taught=["centre"]) is None


def test_the_verdict_is_praise_then_the_lesson_and_varies() -> None:
    board, mate = _uci(SCHOLAR, "h5f7")
    text, kind = verdict(board, mate, "best", rng=random.Random(3))
    assert text in MATE and kind == "mate"
    board, rook = _uci("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1d1")
    text, kind = verdict(board, rook, "best", phase="endgame", rng=random.Random(1))
    praise, clause = text.split(" -- ")
    assert praise in OPENERS["best"] and clause.rstrip(".") in PRINCIPLES["takeFile"]
    assert kind == "takeFile"
    board, fork = _uci("6k1/5ppp/8/r7/3r4/8/5PPP/2N3K1 w - - 0 1", "c1b3")
    tag = MotifTag(name="fork", squares=["b3", "a5", "d4"])
    text, kind = verdict(board, fork, "best", [tag], rng=random.Random(1))
    assert text.endswith(" -- the knight on b3 forks the rook on a5 and the rook on d4.")
    # Nothing to teach: the praise alone. Never the same praise twice running.
    board, d6 = _uci("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "d7d6")
    seen = {verdict(board, d6, "good", phase="opening", rng=random.Random(i))[0] for i in range(40)}
    assert seen == {o + "." for o in OPENERS["good"]}
    previous = None
    for i in range(40):
        current, _ = verdict(
            board, d6, "good", phase="opening", last=previous, rng=random.Random(i)
        )
        assert current != previous
        previous = current
    text, _ = verdict(board, d6, "book", phase="opening", rng=random.Random(0))
    assert text in [o + "." for o in OPENERS["book"]]


def test_no_word_on_a_sound_move_opens_with_it_or_says_what_it_did() -> None:
    """Over random games, every move and label: the word never starts with the move, never
    names it, and never says the capture, check, castling or promotion back."""
    rng = random.Random(11)
    said = 0
    for _game in range(40):
        board = chess.Board()
        taught: list[str] = []
        for _ in range(rng.randrange(10, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            san = board.san(move)
            label = rng.choice(list(OPENERS))
            phase = "opening" if board.fullmove_number <= 10 else "middlegame"
            text, kind = verdict(board, move, label, phase=phase, taught=taught, rng=rng)
            assert not text.startswith(san.rstrip("+#")), text
            if len(san) > 2:  # a pawn's move is a square, which a lesson may name
                assert san.rstrip("+#") not in text, (san, text)
            assert not any(word in text.lower() for word in PLAIN), (san, text)
            assert "?" not in text and not any(ch.isdigit() for ch in text), text
            if kind is not None:
                taught = (taught + [kind])[-3:]
            said += 1
            board.push(move)
    assert said > 1000


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
    assert len(out) == 1 and out[0].text in MATE
    assert provider.prompts == [], "no model call for the engine's own move"
    # Off, the model writes it as before.
    coach, provider = _coach(["[p=1.0] Mate on the board -- well played."])
    coach.local_verdicts = False
    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    _mate_in_one_played(coach)
    out = _messages(coach)
    assert len(out) == 1 and out[0].text == "Mate on the board -- well played."
    assert len(provider.prompts) == 1


def _played(coach, provider, san: str, label, loss: float = 0.01):  # noqa: ANN001
    """A sound move of the user's from the scholar's-mate position, labelled `label`."""
    from sidecoach.events import MoveAssessment, MovePlayed, Severity
    from tests.test_coach import SCHOLAR as SCHOLAR_FEN
    from tests.test_coach import _mate_analysis, _messages, _play_to_scholar

    _play_to_scholar(coach)
    coach.on_event(_mate_analysis(6, decisive=False), 7.0)
    board = chess.Board(SCHOLAR_FEN)
    move = board.push_san(san)
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=7,
            san=san,
            uci=move.uci(),
            fen_before=SCHOLAR_FEN,
            fen_after=board.fen(),
            by_user=True,
        ),
        8.0,
    )
    coach.on_event(
        MoveAssessment(
            game_id="g",
            ply=7,
            uci=move.uci(),
            expected_loss=loss,
            severity=Severity.NONE,
            best=["h5f7"],
            label=label,
        ),
        8.5,
    )
    return [m.text for m in _messages(coach)]


def test_a_good_move_goes_to_the_model_which_names_the_better_one() -> None:
    """A little behind the best (Tai: something was better): the model says what, at
    its full effort. An excellent move is as strong as the best and needs no model."""
    from sidecoach.coach.coach import _exciting, _local
    from sidecoach.events import Label
    from tests.test_coach import _coach

    assert all(_local(Label(x)) for x in ("book", "best", "excellent"))
    assert not any(_local(Label(x)) for x in ("good", "great", "brilliant", "inaccuracy"))
    assert all(_exciting(Label(x)) for x in ("great", "brilliant"))
    coach, provider = _coach(["[p=1.0] Fine, but Qxf7# was there: the mate."])
    texts = _played(coach, provider, "Nc3", Label.GOOD, loss=0.03)
    assert texts == ["Fine, but Qxf7# was there: the mate."]
    assert len(provider.prompts) == 1, "the model, for the better move"
    directive = provider.prompts[0][1][-1]["content"]
    assert "fine but not the strongest" in directive and "Qxf7#" in directive
    assert "Do not open with the move they played" in directive
    coach, provider = _coach(["never used"])
    texts = _played(coach, provider, "Nc3", Label.EXCELLENT)
    assert texts and not texts[0].startswith("Nc3") and provider.prompts == [], "the board's word"
    assert texts[0].split(" -- ")[0].rstrip(".") in OPENERS["excellent"]


def test_a_great_or_brilliant_move_gets_the_models_delight() -> None:
    """Rare and worth a real sentence: the model at its full effort, told to show some
    delight; the board's own line stands in if the reply fails."""
    from sidecoach.events import Label
    from tests.test_coach import _coach

    coach, provider = _coach(["[p=1.0] Yes -- the one move, and it ends the game."])
    texts = _played(coach, provider, "Qxf7#", Label.GREAT, loss=0.0)
    assert texts == ["Yes -- the one move, and it ends the game."]
    assert len(provider.prompts) == 1
    directive = provider.prompts[0][1][-1]["content"]
    assert "Show some delight" in directive and "great" in directive
    assert "Do not open with the move they played" in directive
    assert coach.speaker.provider is provider, "the main model, not the routine copy"
    # The reply failing, the board's own line stands in: the mate.
    coach, provider = _coach(["", ""])
    texts = _played(coach, provider, "Qxf7#", Label.BRILLIANT, loss=0.0)
    assert len(texts) == 1 and texts[0] in MATE
