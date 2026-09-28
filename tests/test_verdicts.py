"""The word on a sound move written from the board (coach/verdicts.py): praise for the
label, then what the move teaches when the board shows it -- never the move itself nor what
it plainly did -- never the same praise twice running, and no model call for it in the
coach."""

from __future__ import annotations

import random

import chess

from sidecoach.coach.verdicts import MATE, MATE_THREAT, OPENERS, PRINCIPLES, lesson, verdict
from sidecoach.events import MotifTag

SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
# What the player sees on the board for themselves: never said back to them (Tai,
# 2026-09-23).
PLAIN = ("taking the", "takes the", "with check", "castling", "castles", "promoting", "promotes")


# A rook to the open d-file, the king with a square to go to: the file is the point.
OPEN_FILE = "6k1/5pp1/7p/8/8/8/5PPP/R5K1 w - - 0 1"


def _uci(fen: str, uci: str) -> tuple[chess.Board, chess.Move]:
    return chess.Board(fen), chess.Move.from_uci(uci)


def test_the_lesson_is_what_the_board_teaches_not_what_the_move_did() -> None:
    board, mate = _uci(SCHOLAR, "h5f7")
    assert lesson(board, mate) == ("mate", ["checkmate"])
    # A capture, a check, a promotion: said by the board, not by the coach.
    board, check = _uci("rnbqkbnr/pppp2pp/8/4pp2/4P3/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", "d1h5")
    assert lesson(board, check, phase="opening") is None
    board, push = _uci("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2", "d7d6")
    assert lesson(board, push, phase="opening") is None, "no idea the board can name"
    # The idea behind the move, as the principle it follows.
    board, castle = _uci(
        "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4", "e1g1"
    )
    assert lesson(board, castle, phase="opening") == ("castle", list(PRINCIPLES["castle"]))
    board, rook = _uci(OPEN_FILE, "a1d1")
    assert lesson(board, rook, phase="endgame") == ("takeFile", list(PRINCIPLES["takeFile"]))
    # A pawn to the centre is a lesson only with what it opens: e4 lets the bishop out; a
    # bare "the fight is for the centre" after every pawn move taught nothing (Tai,
    # 2026-09-24).
    board, e4 = _uci(chess.STARTING_FEN, "e2e4")
    assert lesson(board, e4, phase="opening") == ("centre", ["that lets your bishop out"])
    board, c4 = _uci("rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R w KQkq - 2 2", "c2c4")
    said = lesson(board, c4, phase="opening")
    assert said is None or said[0] != "centre"
    # A tag's tactic, described only when the squares bear it out after the move.
    board, fork = _uci("6k1/5ppp/8/r7/3r4/8/5PPP/2N3K1 w - - 0 1", "c1b3")
    tag = MotifTag(name="fork", squares=["b3", "a5", "d4"])
    assert lesson(board, fork, [tag]) == (
        "fork",
        ["the knight on b3 forks the rook on a5 and the rook on d4"],
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
    board, rook = _uci(OPEN_FILE, "a1d1")
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
    for _game in range(10):
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
    assert said > 250


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
    # The label said, which the judge asks the sense of: "fine" alone was said of inaccuracies.
    assert "which was good: playable, not the strongest" in directive and "Qxf7#" in directive
    assert "Do not open with the move they played" in directive
    coach, provider = _coach(["never used"])
    texts = _played(coach, provider, "Nc3", Label.EXCELLENT)
    assert texts and not texts[0].startswith("Nc3") and provider.prompts == [], "the board's word"
    assert texts[0].split(" -- ")[0].rstrip(".") in OPENERS["excellent"]


def test_a_great_or_brilliant_move_gets_the_models_delight() -> None:
    """Rare and worth a real sentence: the model at its full effort, told to show some
    delight; the board's own line stands in if the reply fails (Tai, 2026-09-26: great and
    brilliant moves go through the API)."""
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


def test_a_player_below_1400_hears_no_term_of_art(monkeypatch) -> None:
    """The simulated 800-rated student listed "prophylaxis: stop their plan before pushing
    your own" among what it did not understand; every principle keeps a plain wording."""
    from sidecoach.coach import verdicts

    for kind, words in PRINCIPLES.items():
        assert any(not verdicts.JARGON.search(w) for w in words), kind
    clauses = list(PRINCIPLES["prophylaxis"])
    monkeypatch.setattr(verdicts, "lesson", lambda *a, **k: ("prophylaxis", clauses))
    move = chess.Move.from_uci("e2e4")
    said = {
        verdict(chess.Board(), move, "good", rng=random.Random(i), plain=True)[0] for i in range(20)
    }
    assert not any("prophylaxis" in s for s in said)
    rich = {verdict(chess.Board(), move, "good", rng=random.Random(i))[0] for i in range(40)}
    assert any("prophylaxis" in s for s in rich)


def test_a_player_of_1800_and_up_hears_no_textbook_principle(monkeypatch) -> None:
    """The strong simulated students most often said they already knew "rooks belong on
    open files" (75 times): from 1800 the praise stands alone."""
    from sidecoach.coach import verdicts

    clauses = list(PRINCIPLES["takeFile"])
    monkeypatch.setattr(verdicts, "lesson", lambda *a, **k: ("takeFile", clauses))
    move = chess.Move.from_uci("e2e4")
    text, kind = verdict(chess.Board(), move, "good", rng=random.Random(1), strong=True)
    assert kind is None and "rook" not in text
    text, kind = verdict(chess.Board(), move, "good", rng=random.Random(1))
    assert kind == "takeFile"


def test_a_principle_only_when_it_is_the_moves_reason() -> None:
    """Tai, 2026-09-27: "rooks on open files" of a rook that checked, set up mate, pinned or
    skewered; "passed pawns must be pushed" of the push that mated. "If it says something it
    should be relevant". A move with a concrete point gets that point or the praise alone."""
    # Rd1 on the back rank with no luft threatens Rd8 mate: that is the lesson, not the file.
    board, rook = _uci("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", "a1d1")
    assert lesson(board, rook, phase="endgame") == ("mateThreat", list(MATE_THREAT))
    assert lesson(board, rook, phase="endgame", taught=["mateThreat"]) is None
    # The same file, with a check: the check is its point.
    board, check = _uci("3k4/5pp1/7p/8/8/8/5PPP/R5K1 w - - 0 1", "a1d1")
    assert board.gives_check(check)
    assert lesson(board, check, phase="endgame") is None
    # A line that mates (the annotator's mate tag), a sacrifice, a deflection: no principle.
    board, rook = _uci(OPEN_FILE, "a1d1")
    for name in ("mate", "mateIn3", "sacrifice", "deflection", "clearance", "pin"):
        tag = MotifTag(name=name, squares=["d1"])
        assert lesson(board, rook, [tag], phase="endgame") is None, name
    # A rook that lines up on a knight in front of a rook (a pin the line does not cash in,
    # so untagged): the pin is its point.
    board, pin = _uci("3r2k1/5pp1/3n3p/8/8/8/5PPP/R5K1 w - - 0 1", "a1d1")
    assert lesson(board, pin, phase="endgame") is None
    # A move that makes a threat to win a piece: the threat is its point.
    board, hit = _uci("6k1/5pp1/7p/3n4/8/8/5PPP/R5K1 w - - 0 1", "a1d1")
    assert lesson(board, hit, phase="endgame") is None
    # A capture that wins material is about the material: bxc3 taking back the knight was
    # "that lets your bishop out".
    board, take = _uci("rnbqkb1r/ppp1pp1p/6p1/8/3PP3/2n5/PP3PPP/R1BQKBNR w KQkq - 0 6", "b2c3")
    assert lesson(board, take, phase="opening") is None


def test_a_threat_met_is_said_as_one_and_a_pawn_ending_has_no_pieces_to_free() -> None:
    # Nc3 guards e4 against Nxe4: their threat met, not "prophylaxis".
    board, nc3 = _uci("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "b1c3")
    said = lesson(board, nc3, phase="opening", threats=["Nxe4"])
    assert said is not None and said[0] in ("defend", "develop")
    # A pawn break in a pawn ending: no pieces for the lines to open.
    board, push = _uci("8/5k2/8/4p3/3P4/8/5K2/8 w - - 0 1", "d4d5")
    said = lesson(board, push, phase="endgame")
    assert said is None or said[0] not in ("break", "centre", "space", "pawnStorm")
