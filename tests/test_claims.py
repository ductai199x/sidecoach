import json
import os
import time

import chess
import pytest

from sidecoach.coach.claims import (
    Claim,
    ClaimKind,
    EngineLike,
    Verdict,
    check_draft,
    extract_claims,
    verify_claims,
)
from sidecoach.coach.disclosure import PositionFacts
from sidecoach.events import Line
from tests.fakes import FakeEngine, ScriptedProvider

# 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6, white to move. The f3 knight attacks e5, which the c6
# knight defends; the e4 pawn is attacked by the f6 knight and defended by nothing.
ITALIAN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
# The same with the opponent (Black) thinking: the player is still White.
ITALIAN_BLACK = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
# 4.Ng5 on top of that, black to move: white threatens Nxf7 and Bxf7+.
FRIED_LIVER = "r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 5 4"
# A white knight on e5 attacked by the d6 pawn and defended by the d4 pawn, white to move.
KNIGHT_E5 = "rnbqkb1r/ppp2ppp/3p1n2/4N3/3PP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5"
# Scholar's mate threatened, black to move: the h5 queen and the c4 bishop both hit f7, so
# after either one takes, the other still covers f7 and the e8 king cannot take back.
SCHOLAR_THREAT = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 4 4"
# The white knight on c3 is pinned to its king by the b4 bishop and reaches the black knight
# on e4, which it cannot take. White to move, then black.
PINNED_ATTACKER_W = "4k3/8/8/8/1b2n3/2N5/8/4K3 w - - 0 1"
PINNED_ATTACKER_B = "4k3/8/8/8/1b2n3/2N5/8/4K3 b - - 0 1"
# The same pinned knight is the only white piece reaching the e4 pawn, which the f6 knight
# attacks: it cannot take back. White to move, then black.
PINNED_DEFENDER_W = "4k3/8/5n2/8/1b2P3/2N5/8/4K3 w - - 0 1"
PINNED_DEFENDER_B = "4k3/8/5n2/8/1b2P3/2N5/8/4K3 b - - 0 1"
# The d4 rook is attacked once (d7 rook) and defended once (d1 queen), but the d8 queen
# stands behind the d7 rook: Rxd4 Qxd4 Qxd4 wins it.
X_RAY = "3q2k1/3r4/8/8/3R4/8/8/3Q2K1 w - - 0 1"
# After 1.f3 e5 2.g4 Qh4#: white is mated, and in check either way.
MATED = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
# After 9...h5, white to move: the bishop on b3 is not pinned, the knight on f3 not attacked.
AFTER_H5 = "r1bq1rk1/2p2pp1/p1np1n2/1p2p2p/4P3/1BNP1N2/PPP2PPP/R1BQ1RK1 w - - 0 10"
# A white knight on f3 and the two kings: nothing defends it, nothing attacks it.
LONE_KNIGHT = "4k3/8/8/8/8/5N2/8/4K3 w - - 0 1"


def claim(
    kind: str,
    subject: str | None = None,
    target: str | None = None,
    move: str | None = None,
    text: str = "",
) -> Claim:
    return Claim(kind=ClaimKind(kind), subject=subject, target=target, move=move, text=text)


def verdict(
    board: chess.Board,
    one: Claim,
    *,
    facts: PositionFacts | None = None,
    engine: EngineLike | None = None,
) -> Verdict:
    verdicts = verify_claims([one], board, facts=facts, engine=engine)
    assert len(verdicts) == 1
    return verdicts[0]


CANNED = [
    {
        "kind": "attacks",
        "subject": "my knight",
        "target": "e5",
        "move": None,
        "text": "my knight is attacking e5",
    },
    {
        "kind": "hanging",
        "subject": None,
        "target": "the pawn",
        "move": None,
        "text": "the pawn is hanging",
    },
    {"kind": "best", "subject": None, "target": None, "move": "Nxe5", "text": "Nxe5 is best"},
]
ATTACKS, HANGING, BEST, PLAN = ClaimKind.ATTACKS, ClaimKind.HANGING, ClaimKind.BEST, ClaimKind.PLAN


# --- extraction --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "kinds"),
    [
        pytest.param(json.dumps(CANNED), [ATTACKS, HANGING, BEST], id="an-array"),
        pytest.param(
            "Here you go:\n```json\n" + json.dumps(CANNED[:1]) + "\n```\n", [ATTACKS], id="fenced"
        ),
        pytest.param(json.dumps(CANNED[1]), [HANGING], id="a-single-object"),
        pytest.param(
            json.dumps([{"kind": "vibes", "text": "x"}, {"kind": "PLAN", "text": "push h"}]),
            [PLAN],
            id="an-unknown-kind-dropped",
        ),
        pytest.param(
            "<think>\nhmm, the knight...\n</think>\n" + json.dumps(CANNED[:1]),
            [ATTACKS],
            id="a-leaked-think-block",
        ),
        pytest.param("I think the knight is great!", [], id="prose"),
        pytest.param("", [], id="empty"),
        pytest.param('{"kind": ', [], id="cut-short"),
        pytest.param("[1, 2, 3]", [], id="not-claims"),
    ],
)
def test_extract_reads_the_reply_or_nothing(reply: str, kinds: list[ClaimKind]) -> None:
    claims = extract_claims("...", chess.Board(ITALIAN), ScriptedProvider([reply]), thinking=False)
    assert [c.kind for c in claims] == kinds


def test_extract_keeps_the_fields_and_never_raises() -> None:
    board = chess.Board(ITALIAN)
    provider = ScriptedProvider([json.dumps(CANNED)])
    claims = extract_claims("my knight is attacking e5 ...", board, provider, thinking=False)
    assert claims[0].subject == "my knight" and claims[0].target == "e5"
    assert claims[0].text == "my knight is attacking e5"
    assert claims[1].subject is None and claims[1].target == "the pawn"
    assert claims[2].move == "Nxe5"
    # Missing keys are None.
    reply = json.dumps([{"kind": "PLAN", "text": "push h"}])
    (plan,) = extract_claims("...", board, ScriptedProvider([reply]), thinking=False)
    assert plan.subject is None and plan.move is None
    # A failed call, or nothing to read, is no claims.
    assert extract_claims("...", board, ScriptedProvider(fail=True), thinking=False) == []
    blank = ScriptedProvider([json.dumps(CANNED)])
    assert extract_claims("   ", board, blank, thinking=False) == [] and blank.calls == []


def test_extract_asks_at_low_temperature_with_the_position() -> None:
    provider = ScriptedProvider(["[]"])
    # Reasoning off, as every caller asks (cli._claims_check, the judge).
    extract_claims("the pawn is hanging", chess.Board(ITALIAN), provider, thinking=False)
    (call,) = provider.calls  # one call: no retry
    assert call["thinking"] is False
    assert call["temperature"] <= 0.2
    assert "JSON" in call["system"] and '"hanging"' in call["system"]
    message = call["messages"][0]
    assert message["role"] == "user"
    assert ITALIAN in message["content"] and "the pawn is hanging" in message["content"]


# --- the player's claims, decided on the board -------------------------------------------

# (position, the claim, the verdict, what its reason says, what it must not say). A reason
# is given to the model beside the player's words: it is held to what it says, not to its
# every word.
VERIFY = [
    # attacks and defends
    pytest.param(
        ITALIAN, claim("attacks", "the knight on f3", "e5"), "true",
        ["knight on f3 attacks their pawn on e5", "defended by the c6 knight"], [],
        id="attacks-and-its-defender",
    ),
    pytest.param(
        ITALIAN, claim("attacks", "the knight on f3", "c5"), "false",
        ["knight on f3 does not attack c5", "e5 pawn"], [], id="attacks-what-it-does-attack",
    ),
    # "My knight" with two knights: the one the claim holds for; neither, both named.
    pytest.param(
        ITALIAN, claim("attacks", "my knight", "e5"), "true", ["knight on f3"], ["b1"],
        id="my-knight-the-one-it-holds-for",
    ),
    pytest.param(
        ITALIAN, claim("attacks", "my knight", "h7"), "unclear", ["knights on b1 and f3"], [],
        id="my-knight-neither",
    ),
    # "Their" is the opponent, "your" the player, in the reason as in the words.
    pytest.param(
        ITALIAN, claim("attacks", "their knight", "e4"), "true",
        ["Their knight on f6 attacks your pawn on e4", "nothing defends it"], [],
        id="their-is-the-opponent",
    ),
    pytest.param(ITALIAN, claim("attacks", "it", "e5"), "unclear", ["'it'"], [], id="it"),
    pytest.param(ITALIAN, claim("attacks", None, "e5"), "unclear", [], [], id="no-subject"),
    pytest.param(
        ITALIAN, claim("defends", "my knight", "d2"), "true",
        ["knight on b1 defends your pawn on d2"], [], id="defends",
    ),
    pytest.param(
        ITALIAN, claim("defends", "the knight on f3", "d4"), "true",
        ["covers d4, which is empty"], [], id="defends-an-empty-square",
    ),
    pytest.param(
        ITALIAN, claim("defends", "the knight on f3", "e4"), "false",
        ["knight on f3 does not reach your pawn on e4"], [], id="does-not-defend",
    ),
    # "I think Nf3 is good because it defends d4": checked after the move, on a copy.
    pytest.param(
        chess.STARTING_FEN, claim("defends", "my knight", "d4", move="Nf3"), "true",
        ["After Nf3, your knight on f3 covers d4"], [], id="after-a-move",
    ),
    pytest.param(
        chess.STARTING_FEN, claim("defends", "Nf3", "d4"), "true", ["After Nf3"], [],
        id="after-a-move-named-as-the-subject",
    ),
    # hanging and safe
    pytest.param(
        ITALIAN, claim("hanging", None, "the pawn on e5"), "false",
        ["attacked by the f3 knight but defended by the c6 knight", "not hanging outright"], [],
        id="a-defended-pawn-is-not-hanging",
    ),
    pytest.param(
        ITALIAN, claim("hanging", None, "my pawn on e4"), "true",
        ["pawn on e4 is attacked by the f6 knight and nothing defends it"], [],
        id="an-undefended-piece-is-hanging",
    ),
    pytest.param(
        ITALIAN, claim("hanging", None, "the bishop on c4"), "false",
        ["bishop on c4 is not attacked"], [], id="an-unattacked-piece-is-not-hanging",
    ),
    pytest.param(
        ITALIAN, claim("hanging", "the pawn", None), "true", ["pawn on e4"], [],
        id="hanging-finds-the-hanging-one",
    ),
    pytest.param(
        ITALIAN, claim("safe", None, "the e5 pawn"), "true",
        ["attacked once (the f3 knight) and defended once (the c6 knight)"], [],
        id="safe-states-the-counts",
    ),
    pytest.param(
        ITALIAN, claim("safe", None, "my pawn on e4"), "false",
        ["attacked once (the f6 knight) and defended not at all"], [], id="not-safe",
    ),
    pytest.param(
        ITALIAN, claim("safe", None, "my bishop on c4"), "true",
        ["bishop on c4 is not attacked"], [], id="safe-unattacked",
    ),
    pytest.param(
        ITALIAN, claim("safe", None, "d5"), "false", ["d5 is attacked by the f6 knight"], [],
        id="a-square-is-not-safe",
    ),
    pytest.param(
        KNIGHT_E5, claim("safe", None, "my knight on e5"), "false",
        ["attacked once (the d6 pawn) and defended once (the d4 pawn)", "worth less than a knight"],
        [], id="not-safe-when-a-cheaper-piece-attacks",
    ),
    pytest.param(
        KNIGHT_E5, claim("safe", None, "my knight"), "true",
        ["knight on b1 is not attacked", "knight on e5 is attacked"], [],
        id="safe-for-one-knight-names-the-other",
    ),
    # threats: theirs through the null move, mine now
    pytest.param(
        FRIED_LIVER, claim("threatens", "they", "my pawn on f7"), "true",
        # The c4 bishop still covers f7 after Nxf7, so the e8 king cannot take back.
        ["If you pass, their knight on g5 takes your pawn on f7 with Nxf7", "Nxf7 for free"], [],
        id="a-real-threat-via-the-null-move",
    ),
    pytest.param(
        FRIED_LIVER, claim("threatens", "they", "my king"), "true", ["Bxf7+"], [],
        id="a-threat-to-the-king",
    ),
    pytest.param(
        FRIED_LIVER, claim("threatens", "they", "my knight on c6"), "false",
        ["Nothing of theirs can take your knight on c6"], [], id="no-threat",
    ),
    pytest.param(
        FRIED_LIVER, claim("threatens", "they", None, move="Nxf7"), "true", ["Nxf7"], [],
        id="a-named-threat",
    ),
    pytest.param(
        FRIED_LIVER, claim("threatens", "they", None, move="Nxe5"), "false",
        ["not a move they can play next"], [], id="a-named-threat-they-cannot-play",
    ),
    pytest.param(
        ITALIAN, claim("threatens", "I", "the e5 pawn"), "true",
        ["Your knight on f3 takes their pawn on e5 with Nxe5"], [], id="my-own-threat",
    ),
    pytest.param(
        ITALIAN, claim("threatens", "I", None, move="Bxf7+"), "true", ["Bxf7+"], [],
        id="my-own-named-threat",
    ),
    pytest.param(
        MATED, claim("threatens", "they", "my king"), "unclear", ["in check"], [],
        id="a-threat-while-in-check",
    ),
    # best, with no engine view
    pytest.param(
        ITALIAN, claim("best", move="Nc3"), "unclear", ["no engine view"], [], id="best-unseen"
    ),
    pytest.param(
        ITALIAN, claim("best", move="Qh5"), "false", ["Qh5 is not a legal move"], [],
        id="best-illegal",
    ),
    pytest.param(
        ITALIAN, claim("best", move="the knight thing"), "unclear", ["cannot read"], [],
        id="best-unreadable",
    ),
    # plan and other invite something concrete
    pytest.param(
        ITALIAN, claim("plan", text="the plan is to push the h-pawn"), "unclear",
        ["which move starts it"], [], id="a-plan",
    ),
    pytest.param(
        ITALIAN, claim("other", text="I feel good here"), "unclear", ["concretely"], [],
        id="other",
    ),
    # The 2026-09-23 audit: attack and defence by legal moves, a threat by who can take back.
    pytest.param(
        SCHOLAR_THREAT, claim("threatens", "they", "my f7 pawn"), "true",
        ["their queen on h5 takes your pawn on f7 with Qxf7#", "checkmate"], [],
        id="a-threat-that-mates",
    ),
    pytest.param(
        SCHOLAR_THREAT, claim("threatens", "they", None, move="Bxf7+"), "true",
        ["their bishop on c4 takes your pawn on f7 with Bxf7+", "for free"], [],
        id="a-threat-no-one-can-take-back",
    ),
    pytest.param(
        PINNED_ATTACKER_W, claim("hanging", None, "their knight on e4"), "false",
        ["attacked only by the c3 knight", "pinned and cannot take it"], [],
        id="a-pinned-attacker-takes-nothing",
    ),
    pytest.param(
        SCHOLAR_THREAT, claim("hanging", None, "my f7 pawn"), "true",
        ["attacked by the c4 bishop and the h5 queen", "the e8 king cannot take back"], [],
        id="the-king-cannot-take-back",
    ),
    pytest.param(
        PINNED_DEFENDER_B, claim("hanging", None, "their pawn on e4"), "true",
        ["attacked by the f6 knight", "the c3 knight is pinned and cannot take back"], [],
        id="a-pinned-defender-defends-nothing",
    ),
    # With the bishop the only piece on f7, the king does take back after Bxf7+.
    pytest.param(
        ITALIAN, claim("hanging", None, "their f7 pawn"), "false",
        ["defended by the e8 king", "the exchange on f7 does not win it"], [],
        id="the-king-takes-back",
    ),
    pytest.param(
        PINNED_DEFENDER_W, claim("safe", None, "my e4 pawn"), "false",
        ["defended not at all", "the c3 knight is pinned and cannot take back"], [],
        id="safe-plays-out-a-pinned-defender",
    ),
    pytest.param(
        PINNED_ATTACKER_B, claim("safe", None, "my e4 knight"), "true",
        ["attacked only by the c3 knight", "pinned and cannot take it"], [],
        id="safe-from-a-pinned-attacker",
    ),
    pytest.param(
        X_RAY, claim("safe", None, "my d4 rook"), "false",
        ["the exchange on d4 still wins it", "the d8 queen behind the d7 rook joins in"], [],
        id="safe-plays-out-the-x-ray",
    ),
    pytest.param(
        PINNED_DEFENDER_W, claim("defends", "my knight", "e4"), "false",
        ["knight on c3 is pinned and cannot take back on e4"], [], id="a-pinned-piece-defends-not",
    ),
    pytest.param(
        PINNED_ATTACKER_W, claim("attacks", "my knight", "e4"), "true",
        ["knight on c3 attacks their knight on e4", "pinned and cannot take it"], [],
        id="a-pinned-piece-attacks-but-cannot-take",
    ),
]  # fmt: skip


@pytest.mark.parametrize(("fen", "one", "result", "says", "never"), VERIFY)
def test_a_claim_is_decided_on_the_board(
    fen: str, one: Claim, result: str, says: list[str], never: list[str]
) -> None:
    board = chess.Board(fen)
    v = verdict(board, one)
    assert v.result == result, v.reason
    for words in says:
        assert words in v.reason, v.reason
    for words in never:
        assert words not in v.reason, v.reason
    assert board.fen() == fen, "the board is untouched"


def test_threat_uses_the_engine_line_when_facts_have_it() -> None:
    board = chess.Board(ITALIAN)
    facts = PositionFacts(fen=board.fen(), side_to_move="white", phase="opening", threats=["Nxe4"])
    v = verdict(board, claim("threatens", "they", "e4"), facts=facts)
    assert v.result == "true" and "the engine's main threat is Nxe4" in v.reason
    v = verdict(board, claim("threatens", "they", "my bishop on c4"), facts=facts)
    assert v.result == "false" and "the engine's threat is Nxe4" in v.reason


def test_best_against_candidates() -> None:
    board = chess.Board(ITALIAN)
    facts = PositionFacts(
        fen=board.fen(), side_to_move="white", phase="opening", candidates=["Nc3", "d3"]
    )
    v = verdict(board, claim("best", None, None, move="Nc3"), facts=facts)
    assert v.result == "true" and "Nc3 is among the engine's best moves" in v.reason
    v = verdict(board, claim("best", None, None, move="Ng5"), facts=facts)
    assert v.result == "false" and "it prefers Nc3 and d3" in v.reason


def test_best_against_a_fake_engine() -> None:
    board = chess.Board(ITALIAN)
    engine = FakeEngine(
        default=[
            Line(moves=["d2d4"], depth=12, expected_score=0.55),
            Line(moves=["b1c3"], depth=12, expected_score=0.54),
            Line(moves=["a2a3"], depth=12, expected_score=0.45),
        ]
    )
    v = verdict(board, claim("best", None, None, move="Nc3"), engine=engine)
    assert v.result == "true" and "(d4 and Nc3)" in v.reason
    v = verdict(board, claim("best", None, None, move="a3"), engine=engine)
    assert v.result == "false" and "it prefers d4 and Nc3" in v.reason
    assert engine.calls == [board.fen(), board.fen()]


def test_best_prefers_facts_but_ignores_facts_for_another_position() -> None:
    board = chess.Board(ITALIAN)
    engine = FakeEngine(default=[Line(moves=["d2d4"], depth=12, expected_score=0.55)])
    stale = PositionFacts(
        fen=chess.STARTING_FEN, side_to_move="white", phase="opening", candidates=["e4"]
    )
    v = verdict(board, claim("best", None, None, move="d4"), facts=stale, engine=engine)
    assert v.result == "true" and "d4 is the engine's best move" in v.reason
    assert engine.calls == [board.fen()]


def test_verify_keeps_order_and_claims() -> None:
    board = chess.Board(ITALIAN)
    claims = [claim("plan"), claim("attacks", "the knight on f3", "e5"), claim("other")]
    verdicts = verify_claims(claims, board)
    assert [v.claim for v in verdicts] == claims
    assert [v.result for v in verdicts] == ["unclear", "true", "unclear"]


# --- the coach's draft -------------------------------------------------------------------

W, B = chess.WHITE, chess.BLACK
# A row's faults: each string is found in one problem ("<words>: <reason>"), and there are no
# others -- unless the row ends with `...`, where more may follow. () is a clean draft.
DRAFT = [
    pytest.param(
        ITALIAN, "Your knight on f3 is attacking e5. What defends the e5 pawn, and is Nxe5 real?",
        None, (), (), id="clean",
    ),
    pytest.param(
        ITALIAN, "Your knight on e5 is doing a lot.", None, (),
        ("Your knight on e5: e5 holds their pawn, not a knight",), id="wrong-piece-on-a-square",
    ),
    pytest.param(
        ITALIAN, "The rook on d4 is loose.", None, (), ("d4 is empty",), id="an-empty-square"
    ),
    pytest.param(
        ITALIAN, "Their knight on f3 is well placed.", None, (),
        ("the knight on f3 is yours, not your opponent's",), id="wrong-side",
    ),
    pytest.param(
        ITALIAN, "Black's knight on f3 is well placed.", None, (),
        ("the knight on f3 is white's, not black's",), id="wrong-colour",
    ),
    pytest.param(
        ITALIAN, "White's knight on f3 and black's knight on c6 face off.", None, (), (),
        id="colours-right",
    ),
    pytest.param(
        ITALIAN, "The opponent's knight on c6 defends e5.", None, (), (), id="opponents-right"
    ),
    # On the opponent's turn "your" is still the player being coached: read as the side to
    # move, "your knight on f3" was flagged false and a true draft thrown away.
    pytest.param(ITALIAN_BLACK, "Your knight on f3 eyes e5.", W, (), (), id="your-on-their-turn"),
    pytest.param(
        ITALIAN_BLACK, "Your knight on c6 defends e5.", W, (),
        ("the knight on c6 is your opponent's, not yours", ...), id="your-is-not-theirs",
    ),
    pytest.param(
        ITALIAN_BLACK, "Their knight on c6 defends e5.", W, (), (), id="their-on-their-turn"
    ),
    pytest.param(
        ITALIAN, "The e5 pawn and the f6-knight are the targets.", None, (), (),
        id="square-piece-form",
    ),
    pytest.param(
        ITALIAN, "Your e5 pawn is weak.", None, (), ("your opponent's",),
        id="square-piece-form-wrong-side",
    ),
    pytest.param(
        ITALIAN, "Consider Qh5 here, or maybe Nxe5!?", None, (),
        ("Qh5: Qh5 cannot be played here by either side",), id="an-illegal-move",
    ),
    pytest.param(
        ITALIAN, "O-O-O is not on yet, but O-O is.", None, (), ("O-O-O:",),
        id="an-illegal-castling",
    ),
    pytest.param(
        ITALIAN, "They are threatening Nxe4 next move.", None, (), (), id="their-threat"
    ),
    pytest.param(
        "r1bqkb1r/ppp2ppp/2np1n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",
        "Nd2 is possible.", None, (), ("more than one move",), id="an-ambiguous-move",
    ),
    pytest.param(
        "r1bqkb1r/ppp2ppp/2np1n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",
        "Nbd2 is possible.", None, (), (), id="the-move-disambiguated",
    ),
    pytest.param(
        ITALIAN, "Take your time and look at what changed.", None, (), (), id="prose"
    ),
    pytest.param(
        ITALIAN, "Be careful in area1; the area12 is fine, as is h1n1.", None, (), (),
        id="words-like-squares",
    ),
    pytest.param(
        ITALIAN, "Point at f7 or a1, bare squares are fine.", None, (), (), id="bare-squares"
    ),
    # Pins, hanging and safe pieces, forks and skewers: said only where they hold.
    # After 9...h5 the bishop on b3 is not pinned; the coach must not say it is.
    pytest.param(
        AFTER_H5, "Your bishop on b3 is now pinned by that line.", None, (),
        ("the bishop on b3 is not pinned",), id="a-fabricated-pin",
    ),
    # The rook on d2 shields its king from the rook on d8.
    pytest.param(
        "3rk3/8/8/8/8/8/3R4/3K4 w - - 0 1", "Your rook on d2 is pinned.", None, (), (),
        id="a-real-pin",
    ),
    # The black knight on d3 is pinned to the queen on d7 by the rook on d1 (a relative pin,
    # not to the king); the old king-only test rejected these.
    pytest.param(
        "1r2r1k1/3q1pp1/2p5/p3P2Q/2P5/1P1n3P/6P1/3R1RK1 w - - 1 29",
        "The knight on d3 is pinned to the queen.", None, (), (), id="a-relative-pin",
    ),
    # The queen's line to the rook on d1 is blocked by the knight.
    pytest.param(
        "1r2r1k1/3q1pp1/2p5/p3P2Q/2P5/1P1n3P/6P1/3R1RK1 w - - 1 29",
        "Your rook on d1 is pinned.", None, (), ("the rook on d1 is not pinned",),
        id="a-blocked-line-pins-nothing",
    ),
    pytest.param(
        AFTER_H5, "Your knight on f3 is hanging.", None, (),
        ("nothing attacks the knight on f3",), id="a-defended-piece-called-loose",
    ),
    # The rook on f2 hits the undefended knight on f4.
    pytest.param(
        "4k3/8/8/8/5n2/8/5R2/4K3 w - - 0 1", "Their knight on f4 is hanging.", None, (), (),
        id="a-hanging-piece",
    ),
    pytest.param(
        "4k3/8/8/4p3/5n2/8/5R2/4K3 w - - 0 1", "Their knight on f4 is hanging.", None, (),
        ("the knight on f4 is defended",), id="an-attacked-piece-with-a-defender",
    ),
    pytest.param(
        "4k3/8/8/8/5n2/8/5R2/4K3 b - - 0 1", "Your knight on f4 is safe.", None, (),
        ("the knight on f4 is attacked and undefended",), id="a-hanging-piece-called-safe",
    ),
    pytest.param(ITALIAN, "Your knight on f3 is safe.", None, (), (), id="a-quiet-piece-safe"),
    # "Undefended" means no defender; only "hanging" needs an attacker as well (2026-09-17).
    pytest.param(
        LONE_KNIGHT, "Your knight on f3 is undefended.", None, (), (), id="undefended-alone"
    ),
    pytest.param(
        LONE_KNIGHT, "Your knight on f3 is hanging.", None, (),
        ("nothing attacks the knight on f3",), id="hanging-with-no-attacker",
    ),
    pytest.param(
        "4k3/8/8/8/8/5N2/4P3/4K3 w - - 0 1", "Your knight on f3 is undefended.", None, (),
        ("the knight on f3 is defended",), id="undefended-but-defended",
    ),
    # The 2026-09-23 audit: attack and defence count only legal captures and recaptures.
    pytest.param(
        PINNED_ATTACKER_W, "Their knight on e4 is hanging.", None, (),
        ("nothing can take the knight on e4",), id="hanging-to-a-pinned-attacker",
    ),
    pytest.param(
        PINNED_DEFENDER_B, "Their pawn on e4 is hanging.", None, (), (),
        id="hanging-behind-a-pinned-defender",
    ),
    pytest.param(SCHOLAR_THREAT, "Your pawn on f7 is hanging.", None, (), (), id="hanging-f7"),
    pytest.param(
        KNIGHT_E5, "Your knight on e5 is safe.", None, (),
        ("the knight on e5 can be taken for profit",), id="safe-but-lost-to-the-exchange",
    ),
    # A pin, a fork, a skewer must win something. The queen on a5 hits the c3 knight with
    # the rook on d2 behind, but the king takes back on d2: the knight is not pinned.
    pytest.param(
        "4k3/8/8/q7/8/2N5/3R4/4K3 w - - 0 1", "Your knight on c3 is pinned.", None, (),
        ("the knight on c3 is not pinned",), id="a-pin-that-wins-nothing",
    ),
    pytest.param(
        PINNED_DEFENDER_W, "Your knight on c3 is pinned.", None, (), (), id="a-pin-to-the-king"
    ),
    pytest.param(
        "4q1k1/8/5N2/8/8/8/8/7K b - - 0 1", "The knight on f6 forks the queen and king.", None,
        (), (), id="a-real-fork",
    ),
    pytest.param(
        "4k3/8/8/8/8/5N2/8/4K3 w - - 0 1", "The knight on f3 forks two pieces.", None, (),
        ("the knight on f3 does not fork two pieces",), id="a-fork-of-nothing",
    ),
    # The fork is not on the board in front of the player: Nd5-f6 makes it. With the
    # after-best-move frame, naming it is not faulted; without it, it is.
    pytest.param(
        "4q1k1/8/8/3N4/8/8/8/7K w - - 0 1", "The knight on f6 forks the queen and king.", None,
        ("Nf6",), (), id="a-fork-the-best-move-makes",
    ),
    pytest.param(
        "4q1k1/8/8/3N4/8/8/8/7K w - - 0 1", "The knight on f6 forks the queen and king.", None,
        (), ("knight on f6:", ...), id="a-fork-not-yet-made",
    ),
    # The d5 knight hits the queen and the rook, but it is pinned to its king by the a8 bishop.
    pytest.param(
        "b3k3/2q1r3/8/3N4/2P5/8/8/7K b - - 0 1", "Their knight on d5 forks the queen and rook.",
        None, (), ("the knight on d5 is pinned, so it forks nothing",), id="a-pinned-forker",
    ),
    # Unpinned, but the e6 pawn takes it for free.
    pytest.param(
        "4k3/2q1r3/4p3/3N4/8/8/8/7K b - - 0 1", "Their knight on d5 forks the queen and rook.",
        None, (), ("the knight on d5 can be taken for profit, so it forks nothing",),
        id="a-forker-lost-for-nothing",
    ),
    # The white rook on b2 skewers the black queen on b7 to the rook on b8 behind it.
    pytest.param(
        "1r6/1q6/8/8/8/8/1R6/K6k w - - 0 1", "White's rook on b2 skewers the queen and rook.",
        None, (), (), id="a-real-skewer",
    ),
    pytest.param(
        "8/1q6/8/8/8/8/1R6/K6k w - - 0 1", "The rook on b2 skewers two pieces.", None, (),
        ("the rook on b2 does not skewer anything",), id="nothing-behind",
    ),
    # The b7 rook in front is worth no more than the b2 rook and the c6 king defends it: it
    # need not move, and the knight behind is not won.
    pytest.param(
        "1n6/1r6/2k5/8/8/8/1R6/K7 w - - 0 1", "Your rook on b2 skewers the rook and knight.",
        None, (), ("the rook on b2 does not skewer anything",), id="a-skewer-that-wins-nothing",
    ),
    # With the front rook undefended it must move, and the knight behind falls.
    pytest.param(
        "1n6/1r6/8/8/8/8/1R6/K1k5 w - - 0 1", "Your rook on b2 skewers the rook and knight.",
        None, (), (), id="a-skewer-that-wins",
    ),
    # Files and passed pawns. The d-file has no pawns; the e-file has the pawn on e2.
    pytest.param(
        "3rk3/8/8/8/8/8/4P3/3RK3 w - - 0 1", "Your rook holds the open d-file.", None, (), (),
        id="an-open-file",
    ),
    pytest.param(
        "3rk3/8/8/8/8/8/4P3/3RK3 w - - 0 1", "The d-file is open.", None, (), (),
        id="an-open-file-said-plainly",
    ),
    pytest.param(
        "3rk3/8/8/8/8/8/4P3/3RK3 w - - 0 1", "The e-file is open for you.", None, (),
        ("not open",), id="a-file-that-is-not-open",
    ),
    # Half-open: only an enemy pawn on the file is fine; a friendly pawn is not.
    pytest.param(
        "3rk3/3p4/8/8/8/8/8/3RK3 w - - 0 1", "You have the half-open d-file.", None, (), (),
        id="a-half-open-file",
    ),
    pytest.param(
        "3rk3/8/8/8/8/8/3P4/3RK3 w - - 0 1", "The d-file is half-open for you.", None, (),
        ("not half-open",), id="a-file-with-our-pawn",
    ),
    # Round 2: "your opponent's best is Qf7, taking the half-open f-file" is theirs.
    pytest.param(
        "4k3/8/8/8/8/8/5P2/4K3 w - - 0 1", "Your opponent's rook takes the half-open f-file.",
        None, (), (), id="a-half-open-file-of-theirs",
    ),
    pytest.param(
        "4k3/8/8/8/8/8/5P2/4K3 w - - 0 1", "The f-file is half-open for you.", None, (), (...,),
        id="a-half-open-file-not-ours",
    ),
    pytest.param(
        "4k3/8/8/8/8/8/5P2/4K3 w - - 0 1", "The half-open f-file is theirs to use.", None, (),
        (), id="a-half-open-file-theirs-to-use",
    ),
    # Black to move: the a5 pawn is passed; the c6 pawn is not, the white c4 pawn is ahead.
    pytest.param(
        "4r1k1/2q2np1/2p5/p6Q/2P2R2/6PP/1r1R2K1/8 b - - 0 1", "Your a-pawn is a passed pawn.",
        None, (), (), id="a-passed-pawn",
    ),
    pytest.param(
        "4r1k1/2q2np1/2p5/p6Q/2P2R2/6PP/1r1R2K1/8 b - - 0 1", "You have a passed pawn on a5.",
        None, (), (), id="a-passed-pawn-adjective-first",
    ),
    pytest.param(
        "4r1k1/2q2np1/2p5/p6Q/2P2R2/6PP/1r1R2K1/8 b - - 0 1", "Your c-pawn is a passed pawn.",
        None, (), (...,), id="not-passed-noun-first",
    ),
    pytest.param(
        "4r1k1/2q2np1/2p5/p6Q/2P2R2/6PP/1r1R2K1/8 b - - 0 1", "You have a passed pawn on c6.",
        None, (), (...,), id="not-passed-adjective-first",
    ),
    # The bench's gold false claims, from explanations of lines (2026-09-25): a rook "forced
    # onto f7" where other moves were legal, files "kept open" with pawns on them. A claim no
    # frame applies to is left alone.
    pytest.param(
        "8/p1q1k2K/2p1p3/3p4/3P1R1P/6r1/1r6/8 w - - 0 49",
        "Black then uses the check on d6 to force your rook onto f7.", W, ("Rf1", "Qd6"),
        ("force your rook onto f7:",), id="forced-where-other-moves-were-legal",
    ),
    pytest.param(
        "r6r/pQ3pkp/4p1p1/P2BPb2/3b4/8/1P3PPP/R4RK1 b - - 0 21",
        "Rad8 keeps the b-file and e-file open.", None, (),
        ("the b-file is not open; a pawn stands on it",), id="kept-open-with-a-pawn-on-it",
    ),
    pytest.param(
        "8/8/8/8/8/5k2/8/R3K3 w - - 0 1", "Ra3+ forces their king to move.", W, (), (),
        id="a-check-that-forces",
    ),
    pytest.param(
        "8/8/8/8/5n2/5k2/8/R3K3 w - - 0 1", "Ra3+ forces their king to move.", W, (),
        ("can be blocked", ...), id="a-check-that-can-be-blocked",
    ),
    # No check anywhere: about another moment, left alone.
    pytest.param(
        ITALIAN, "The check forced their king away.", W, (), (), id="a-check-of-another-moment"
    ),
    # "The g-file has no pawns" of a half-open file (the pilot, 2026-09-26); "no pawns of
    # yours on it" is a half-open file, which the template says, and it is true.
    pytest.param(
        "rn2k3/pp2bp2/2p3p1/4Q2p/P1P5/1P3BK1/7R/RN4q1 w q - 7 33",
        "The g-file has no pawns, so use it.", W, (), ("the g-file has a pawn on it",),
        id="no-pawns-on-a-half-open-file",
    ),
    pytest.param(
        "rn2k3/pp2bp2/2p3p1/4Q2p/P1P5/1P3BK1/7R/RN4q1 w q - 7 33",
        "The d-file has no pawns, so use it.", W, (), (), id="no-pawns-true",
    ),
    pytest.param(
        "rn2k3/pp2bp2/2p3p1/4Q2p/P1P5/1P3BK1/7R/RN4q1 w q - 7 33",
        "Rg2 was better -- the g-file has no pawns of yours on it.", W, (), (),
        id="no-pawns-of-yours",
    ),
    # The locked test set's audit (2026-09-26): false claims the checker now catches, each
    # beside the true sentence it must not fault. "Qf2+ drives the King to h3": Kh1 was
    # legal too; judged on Qf2+'s own frame.
    pytest.param(
        "6k1/5pp1/p2p4/1p3Pb1/2p1r1q1/P1P1p3/1P5K/4R3 b - - 1 48", "Qf2+ drives the King to h3.",
        B, ("Qh4+", "Kg2", "Qf2+", "Kh3", "Rh4#"), ("nothing forces the king to h3", ...),
        id="drives-the-king-where-it-had-a-choice",
    ),
    # "Kh7 is the only king move" where four were legal (Qd4+ given: the king has five);
    # "the only move that keeps your edge" is the stakes, and is not checked.
    pytest.param(
        "8/6k1/8/8/3Q4/8/5PPP/6K1 b - - 0 1", "After the check, Kh7 is the only king move.", W,
        (), ("Kh7 is not the only king move: others are legal", ...), id="not-the-only-king-move",
    ),
    pytest.param(
        "8/6k1/8/8/3Q4/8/5PPP/6K1 b - - 0 1", "Kh7 was the only move that keeps it level.", W,
        (), (), id="the-only-move-is-the-stakes",
    ),
    # "Ne4 ... but it allows exf6" where exf6 cannot follow Ne4; across clauses, not judged.
    pytest.param(
        "r1bqkb1r/pppp1ppp/2n2n2/4P3/2B5/8/PPPP1PPP/RNBQK1NR b KQkq - 0 4",
        "Ne4 attacks f2, but it allows exf6.", B, (), ("exf6 cannot be played after Ne4", ...),
        id="allows-what-cannot-follow",
    ),
    pytest.param(
        "r1bqkb1r/pppp1ppp/2n2n2/4P3/2B5/8/PPPP1PPP/RNBQK1NR b KQkq - 0 4",
        "After Nd5, Bxd5 wins a pawn, while your move allowed exf6.", B, (), (),
        id="allows-across-clauses",
    ),
    # "Your queen checks" with no queen; "it took your bishop" is the story, not checked.
    pytest.param(
        "k7/8/8/8/8/8/8/K5R1 b - - 0 1", "Your queen checks the king.", B, (),
        ("you have no queen", ...), id="no-queen",
    ),
    pytest.param(
        "k7/8/8/8/8/8/8/K5R1 b - - 0 1", "That exchange took your bishop.", B, (), (),
        id="the-story",
    ),
    # Asked for the line, the coach recites it: a move only playable further along the line,
    # and a piece on the square a later move puts it, are real on those frames. The checker
    # once flagged "Rxb7 cannot be played here" on the line's own fourth move.
    pytest.param(
        chess.STARTING_FEN,
        "The line goes e4 e5 Nf3 Nc6 Bb5: your bishop on b5 pins nothing yet, but Nc6 is met.",
        None, ("e4", "e5", "Nf3", "Nc6", "Bb5"), (), id="the-line-it-was-given",
    ),
    # Without the frames a later move is not playable now, and the bishop is not on b5.
    pytest.param(
        chess.STARTING_FEN,
        "The line goes e4 e5 Nf3 Nc6 Bb5: your bishop on b5 pins nothing yet, but Nc6 is met.",
        None, (), ("Bb5:", "your bishop on b5:", ...), id="the-line-without-its-frames",
    ),
]  # fmt: skip


@pytest.mark.parametrize(("fen", "text", "us", "line", "faults"), DRAFT)
def test_a_draft_says_only_what_the_board_bears_out(
    fen: str, text: str, us: chess.Color | None, line: tuple[str, ...], faults: tuple
) -> None:
    board = chess.Board(fen)
    after = []
    frame = board.copy()
    for san in line:
        frame.push_san(san)
        after.append(frame.copy())
    said = [f"{p.text}: {p.reason}" for p in check_draft(text, board, after=after, us=us)]
    wanted = [f for f in faults if f is not ...]
    for fault in wanted:
        assert any(fault in s for s in said), (fault, said)
    if ... in faults:
        assert len(said) >= max(1, len(wanted)), said
    else:
        assert len(said) == len(wanted), said


# --- the real thing ----------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("SIDECOACH_LLM_TESTS") != "1",
    reason="set SIDECOACH_LLM_TESTS=1 to talk to the local LLM server",
)
def test_extract_with_the_local_provider() -> None:
    from sidecoach.config import LLMSettings
    from sidecoach.llm.factory import make_provider

    provider = make_provider(LLMSettings(), timeout_s=180)
    sentence = "My knight is attacking e5 and I think the pawn is hanging, so Nxe5 looks best."
    started = time.perf_counter()
    claims = extract_claims(sentence, chess.Board(ITALIAN), provider)
    elapsed = time.perf_counter() - started
    print(f"\n{provider.model}: {len(claims)} claims in {elapsed:.1f}s")
    for one in claims:
        print(f"  {one}")
    for v in verify_claims(claims, chess.Board(ITALIAN)):
        print(f"  [{v.result}] {v.reason}")
    kinds = {one.kind for one in claims}
    assert ClaimKind.ATTACKS in kinds
