import json
import os
import time
from collections.abc import Sequence
from typing import Any

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
from sidecoach.llm.base import ChatMessage, Completion

# 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6, white to move. The f3 knight attacks e5, which the c6
# knight defends; the e4 pawn is attacked by the f6 knight and defended by nothing.
ITALIAN = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
# 4.Ng5 on top of that, black to move: white threatens Nxf7 and Bxf7+.
FRIED_LIVER = "r1bqkb1r/pppp1ppp/2n2n2/4p1N1/2B1P3/8/PPPP1PPP/RNBQK2R b KQkq - 5 4"
# A white knight on e5 attacked by the d6 pawn and defended by the d4 pawn, white to move.
KNIGHT_E5 = "rnbqkb1r/ppp2ppp/3p1n2/4N3/3PP3/8/PPP2PPP/RNBQKB1R w KQkq - 0 5"


class FakeProvider:
    name = "fake"

    def __init__(self, reply: str, *, fail: bool = False) -> None:
        self.reply = reply
        self.fail = fail
        self.calls: list[dict[str, Any]] = []

    def healthcheck(self) -> tuple[bool, str]:
        return True, "fake"

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> Completion:
        self.calls.append(
            {
                "system": system,
                "messages": list(messages),
                "temperature": temperature,
                "thinking": thinking,
            }
        )
        if self.fail:
            raise RuntimeError("server down")
        return Completion(text=self.reply)


class FakeEngine:
    def __init__(self, lines: list[Line]) -> None:
        self.lines = lines
        self.boards: list[str] = []

    def analyse(self, board: chess.Board, *, depth: int | None = None) -> list[Line]:
        self.boards.append(board.fen())
        return self.lines


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


# --- extraction --------------------------------------------------------------------------


def test_extract_from_json_array() -> None:
    provider = FakeProvider(json.dumps(CANNED))
    claims = extract_claims("my knight is attacking e5 ...", chess.Board(ITALIAN), provider)
    assert [c.kind for c in claims] == [ClaimKind.ATTACKS, ClaimKind.HANGING, ClaimKind.BEST]
    assert claims[0].subject == "my knight" and claims[0].target == "e5"
    assert claims[1].subject is None and claims[1].target == "the pawn"
    assert claims[2].move == "Nxe5"
    assert claims[0].text == "my knight is attacking e5"


def test_extract_asks_for_thinking_at_low_temperature_with_the_position() -> None:
    provider = FakeProvider("[]")
    extract_claims("the pawn is hanging", chess.Board(ITALIAN), provider)
    call = provider.calls[0]
    assert call["thinking"] is True
    assert call["temperature"] <= 0.2
    assert "JSON" in call["system"] and '"hanging"' in call["system"]
    message = call["messages"][0]
    assert message["role"] == "user"
    assert ITALIAN in message["content"] and "the pawn is hanging" in message["content"]


def test_extract_from_code_fences() -> None:
    provider = FakeProvider("Here you go:\n```json\n" + json.dumps(CANNED[:1]) + "\n```\n")
    claims = extract_claims("...", chess.Board(ITALIAN), provider)
    assert len(claims) == 1 and claims[0].kind == ClaimKind.ATTACKS


def test_extract_from_a_single_object() -> None:
    provider = FakeProvider(json.dumps(CANNED[1]))
    claims = extract_claims("...", chess.Board(ITALIAN), provider)
    assert len(claims) == 1 and claims[0].kind == ClaimKind.HANGING


def test_extract_drops_unknown_kinds_and_tolerates_missing_keys() -> None:
    reply = json.dumps([{"kind": "vibes", "text": "x"}, {"kind": "PLAN", "text": "push h"}])
    claims = extract_claims("...", chess.Board(ITALIAN), FakeProvider(reply))
    assert [c.kind for c in claims] == [ClaimKind.PLAN]
    assert claims[0].subject is None and claims[0].move is None


def test_extract_returns_nothing_on_garbage_or_failure() -> None:
    board = chess.Board(ITALIAN)
    assert extract_claims("...", board, FakeProvider("I think the knight is great!")) == []
    assert extract_claims("...", board, FakeProvider("")) == []
    assert extract_claims("...", board, FakeProvider('{"kind": ')) == []
    assert extract_claims("...", board, FakeProvider("[1, 2, 3]")) == []
    assert extract_claims("...", board, FakeProvider("[]", fail=True)) == []
    assert extract_claims("   ", board, FakeProvider(json.dumps(CANNED))) == []


class RunawayProvider(FakeProvider):
    """Reasoning eats the whole budget with thinking on, as the local model sometimes does."""

    def chat(
        self,
        system: str,
        messages: Sequence[ChatMessage],
        *,
        max_tokens: int = 1500,
        temperature: float = 0.7,
        thinking: bool = False,
    ) -> Completion:
        completion = super().chat(
            system, messages, max_tokens=max_tokens, temperature=temperature, thinking=thinking
        )
        return Completion(text="") if thinking else completion


def test_extract_retries_without_thinking_when_reasoning_eats_the_budget() -> None:
    provider = RunawayProvider(json.dumps(CANNED))
    claims = extract_claims("...", chess.Board(ITALIAN), provider)
    assert len(claims) == 3
    assert [call["thinking"] for call in provider.calls] == [True, False]
    provider = RunawayProvider(json.dumps(CANNED))
    assert extract_claims("...", chess.Board(ITALIAN), provider, thinking=False)
    assert [call["thinking"] for call in provider.calls] == [False]


def test_extract_strips_a_leaked_think_block() -> None:
    reply = "<think>\nhmm, the knight...\n</think>\n" + json.dumps(CANNED[:1])
    claims = extract_claims("...", chess.Board(ITALIAN), FakeProvider(reply))
    assert len(claims) == 1


# --- attacks and defends ------------------------------------------------------------------


def test_attacks_true_names_the_piece_and_its_defender() -> None:
    v = verdict(chess.Board(ITALIAN), claim("attacks", "the knight on f3", "e5"))
    assert v.result == "true"
    assert "knight on f3 attacks their pawn on e5" in v.reason
    assert "defended by the c6 knight" in v.reason


def test_attacks_false_says_what_the_piece_does_attack() -> None:
    v = verdict(chess.Board(ITALIAN), claim("attacks", "the knight on f3", "c5"))
    assert v.result == "false"
    assert "knight on f3 does not attack c5" in v.reason
    assert "e5 pawn" in v.reason


def test_my_knight_with_two_knights_reports_the_one_it_holds_for() -> None:
    board = chess.Board(ITALIAN)  # white knights on b1 and f3
    v = verdict(board, claim("attacks", "my knight", "e5"))
    assert v.result == "true"
    assert "knight on f3" in v.reason and "b1" not in v.reason

    v = verdict(board, claim("attacks", "my knight", "h7"))
    assert v.result == "unclear"
    assert "knights on b1 and f3" in v.reason


def test_their_and_your_mean_the_opponent_for_the_user() -> None:
    board = chess.Board(ITALIAN)
    v = verdict(board, claim("attacks", "their knight", "e4"))
    assert v.result == "true"
    assert v.reason == "Their knight on f6 attacks your pawn on e4, and nothing defends it."


def test_unresolvable_piece_is_unclear() -> None:
    v = verdict(chess.Board(ITALIAN), claim("attacks", "it", "e5"))
    assert v.result == "unclear" and "'it'" in v.reason
    v = verdict(chess.Board(ITALIAN), claim("attacks", None, "e5"))
    assert v.result == "unclear"


def test_defends_true_and_false() -> None:
    board = chess.Board(ITALIAN)
    v = verdict(board, claim("defends", "my knight", "d2"))
    assert v.result == "true"
    assert "knight on b1 defends your pawn on d2" in v.reason

    v = verdict(board, claim("defends", "the knight on f3", "d4"))
    assert v.result == "true"
    assert "covers d4, which is empty" in v.reason

    v = verdict(board, claim("defends", "the knight on f3", "e4"))
    assert v.result == "false"
    assert "knight on f3 does not reach your pawn on e4" in v.reason


def test_a_claim_about_the_position_after_a_move_is_checked_after_it() -> None:
    board = chess.Board()  # "I think Nf3 is good because it defends d4"
    v = verdict(board, claim("defends", "my knight", "d4", move="Nf3"))
    assert v.result == "true"
    assert v.reason.startswith("After Nf3, your knight on f3 covers d4")
    v = verdict(board, claim("defends", "Nf3", "d4"))
    assert v.result == "true" and "After Nf3" in v.reason
    assert board.turn == chess.WHITE and board.piece_at(chess.F3) is None  # untouched


# --- hanging and safe ---------------------------------------------------------------------


def test_a_defended_pawn_is_not_hanging() -> None:
    v = verdict(chess.Board(ITALIAN), claim("hanging", None, "the pawn on e5"))
    assert v.result == "false"
    assert "attacked by the f3 knight but defended by the c6 knight" in v.reason
    assert "not hanging outright" in v.reason


def test_an_undefended_piece_is_hanging() -> None:
    v = verdict(chess.Board(ITALIAN), claim("hanging", None, "my pawn on e4"))
    assert v.result == "true"
    assert "pawn on e4 is attacked by the f6 knight and nothing defends it" in v.reason


def test_an_unattacked_piece_is_not_hanging() -> None:
    v = verdict(chess.Board(ITALIAN), claim("hanging", None, "the bishop on c4"))
    assert v.result == "false"
    assert "bishop on c4 is not attacked" in v.reason


def test_hanging_with_only_a_piece_word_finds_the_hanging_one() -> None:
    v = verdict(chess.Board(ITALIAN), claim("hanging", "the pawn", None))
    assert v.result == "true" and "pawn on e4" in v.reason


def test_safe_states_the_counts() -> None:
    board = chess.Board(ITALIAN)
    v = verdict(board, claim("safe", None, "the e5 pawn"))
    assert v.result == "true"
    assert "attacked once (the f3 knight) and defended once (the c6 knight)" in v.reason

    v = verdict(board, claim("safe", None, "my pawn on e4"))
    assert v.result == "false"
    assert "attacked once (the f6 knight) and defended not at all" in v.reason

    v = verdict(board, claim("safe", None, "my bishop on c4"))
    assert v.result == "true" and "bishop on c4 is not attacked" in v.reason

    v = verdict(board, claim("safe", None, "d5"))
    assert v.result == "false" and v.reason == "d5 is attacked by the f6 knight."


def test_safe_is_not_safe_when_a_cheaper_piece_attacks() -> None:
    v = verdict(chess.Board(KNIGHT_E5), claim("safe", None, "my knight on e5"))
    assert v.result == "false"
    assert "attacked once (the d6 pawn) and defended once (the d4 pawn)" in v.reason
    assert "worth less than a knight" in v.reason


def test_safe_holding_for_one_piece_still_mentions_the_other() -> None:
    v = verdict(chess.Board(KNIGHT_E5), claim("safe", None, "my knight"))
    assert v.result == "true"
    assert "knight on b1 is not attacked" in v.reason
    assert "knight on e5 is attacked" in v.reason


# --- threats -----------------------------------------------------------------------------


def test_a_real_threat_via_the_null_move() -> None:
    board = chess.Board(FRIED_LIVER)
    v = verdict(board, claim("threatens", "they", "my pawn on f7"))
    assert v.result == "true"
    assert "If you pass, their knight on g5 takes your pawn on f7 with Nxf7" in v.reason
    assert "the e8 king can take back" in v.reason

    v = verdict(board, claim("threatens", "they", "my king"))
    assert v.result == "true" and "Bxf7+" in v.reason

    v = verdict(board, claim("threatens", "they", "my knight on c6"))
    assert v.result == "false"
    assert "Nothing of theirs can take your knight on c6" in v.reason


def test_a_named_threat_is_checked_for_legality_after_the_pass() -> None:
    board = chess.Board(FRIED_LIVER)
    v = verdict(board, claim("threatens", "they", None, move="Nxf7"))
    assert v.result == "true" and "Nxf7" in v.reason
    v = verdict(board, claim("threatens", "they", None, move="Nxe5"))
    assert v.result == "false" and "not a move they can play next" in v.reason


def test_my_own_threat_is_checked_now() -> None:
    board = chess.Board(ITALIAN)
    v = verdict(board, claim("threatens", "I", "the e5 pawn"))
    assert v.result == "true"
    assert v.reason.startswith("Your knight on f3 takes their pawn on e5 with Nxe5")
    v = verdict(board, claim("threatens", "I", None, move="Bxf7+"))
    assert v.result == "true" and "Bxf7+" in v.reason


def test_threat_uses_the_engine_line_when_facts_have_it() -> None:
    board = chess.Board(ITALIAN)
    facts = PositionFacts(fen=board.fen(), side_to_move="white", phase="opening", threats=["Nxe4"])
    v = verdict(board, claim("threatens", "they", "e4"), facts=facts)
    assert v.result == "true" and "the engine's main threat is Nxe4" in v.reason
    v = verdict(board, claim("threatens", "they", "my bishop on c4"), facts=facts)
    assert v.result == "false" and "the engine's threat is Nxe4" in v.reason


def test_threat_while_in_check_is_unclear() -> None:
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    board.push_san("Qh4")  # white is mated, and in check either way
    v = verdict(board, claim("threatens", "they", "my king"))
    assert v.result == "unclear" and "in check" in v.reason


# --- best --------------------------------------------------------------------------------


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
        [
            Line(moves=["d2d4"], depth=12, expected_score=0.55),
            Line(moves=["b1c3"], depth=12, expected_score=0.54),
            Line(moves=["a2a3"], depth=12, expected_score=0.45),
        ]
    )
    v = verdict(board, claim("best", None, None, move="Nc3"), engine=engine)
    assert v.result == "true" and "(d4 and Nc3)" in v.reason
    v = verdict(board, claim("best", None, None, move="a3"), engine=engine)
    assert v.result == "false" and "it prefers d4 and Nc3" in v.reason
    assert engine.boards == [board.fen(), board.fen()]


def test_best_prefers_facts_but_ignores_facts_for_another_position() -> None:
    board = chess.Board(ITALIAN)
    engine = FakeEngine([Line(moves=["d2d4"], depth=12, expected_score=0.55)])
    stale = PositionFacts(
        fen=chess.STARTING_FEN, side_to_move="white", phase="opening", candidates=["e4"]
    )
    v = verdict(board, claim("best", None, None, move="d4"), facts=stale, engine=engine)
    assert v.result == "true" and "d4 is the engine's best move" in v.reason
    assert engine.boards == [board.fen()]


def test_best_without_any_engine_view_is_unclear() -> None:
    v = verdict(chess.Board(ITALIAN), claim("best", None, None, move="Nc3"))
    assert v.result == "unclear" and "no engine view" in v.reason


def test_best_with_an_illegal_or_unreadable_move() -> None:
    board = chess.Board(ITALIAN)
    v = verdict(board, claim("best", None, None, move="Qh5"))
    assert v.result == "false" and "Qh5 is not a legal move" in v.reason
    v = verdict(board, claim("best", None, None, move="the knight thing"))
    assert v.result == "unclear" and "cannot read" in v.reason


# --- plan and other ----------------------------------------------------------------------


def test_plan_and_other_invite_something_concrete() -> None:
    board = chess.Board(ITALIAN)
    v = verdict(board, claim("plan", None, None, text="the plan is to push the h-pawn"))
    assert v.result == "unclear" and "which move starts it" in v.reason
    v = verdict(board, claim("other", None, None, text="I feel good here"))
    assert v.result == "unclear" and "concretely" in v.reason


def test_verify_keeps_order_and_claims() -> None:
    board = chess.Board(ITALIAN)
    claims = [claim("plan"), claim("attacks", "the knight on f3", "e5"), claim("other")]
    verdicts = verify_claims(claims, board)
    assert [v.claim for v in verdicts] == claims
    assert [v.result for v in verdicts] == ["unclear", "true", "unclear"]


# --- the coach's draft -------------------------------------------------------------------


def test_draft_clean_sentence() -> None:
    board = chess.Board(ITALIAN)
    text = "Your knight on f3 is attacking e5. What defends the e5 pawn, and is Nxe5 real?"
    assert check_draft(text, board) == []


def test_draft_wrong_piece_on_a_square() -> None:
    problems = check_draft("Your knight on e5 is doing a lot.", chess.Board(ITALIAN))
    assert [p.text for p in problems] == ["Your knight on e5"]
    assert problems[0].reason == "e5 holds their pawn, not a knight"
    problems = check_draft("The rook on d4 is loose.", chess.Board(ITALIAN))
    assert len(problems) == 1 and problems[0].reason == "d4 is empty"


def test_draft_wrong_colour() -> None:
    board = chess.Board(ITALIAN)
    problems = check_draft("Their knight on f3 is well placed.", board)
    assert len(problems) == 1
    assert problems[0].reason == "the knight on f3 is yours, not your opponent's"
    problems = check_draft("Black's knight on f3 is well placed.", board)
    assert len(problems) == 1
    assert problems[0].reason == "the knight on f3 is white's, not black's"
    assert check_draft("White's knight on f3 and black's knight on c6 face off.", board) == []
    assert check_draft("The opponent's knight on c6 defends e5.", board) == []


def test_draft_square_piece_form() -> None:
    board = chess.Board(ITALIAN)
    assert check_draft("The e5 pawn and the f6-knight are the targets.", board) == []
    problems = check_draft("Your e5 pawn is weak.", board)
    assert len(problems) == 1 and "your opponent's" in problems[0].reason


def test_draft_illegal_move() -> None:
    board = chess.Board(ITALIAN)
    problems = check_draft("Consider Qh5 here, or maybe Nxe5!?", board)
    assert [p.text for p in problems] == ["Qh5"]
    assert problems[0].reason == "Qh5 cannot be played here by either side"
    problems = check_draft("O-O-O is not on yet, but O-O is.", board)
    assert [p.text for p in problems] == ["O-O-O"]


def test_draft_accepts_a_threat_the_opponent_could_play() -> None:
    assert check_draft("They are threatening Nxe4 next move.", chess.Board(ITALIAN)) == []


def test_draft_flags_a_fabricated_pin() -> None:
    # After 9...h5 the bishop on b3 is not pinned; the coach must not say it is.
    board = chess.Board("r1bq1rk1/2p2pp1/p1np1n2/1p2p2p/4P3/1BNP1N2/PPP2PPP/R1BQ1RK1 w - - 0 10")
    problems = check_draft("Your bishop on b3 is now pinned by that line.", board)
    assert len(problems) == 1 and problems[0].reason == "the bishop on b3 is not pinned"
    # A real pin is not flagged: the rook on d2 shields its king from the rook on d8.
    pin = chess.Board("3rk3/8/8/8/8/8/3R4/3K4 w - - 0 1")
    assert check_draft("Your rook on d2 is pinned.", pin) == []


def test_draft_flags_a_defended_piece_called_loose() -> None:
    board = chess.Board("r1bq1rk1/2p2pp1/p1np1n2/1p2p2p/4P3/1BNP1N2/PPP2PPP/R1BQ1RK1 w - - 0 10")
    problems = check_draft("Your knight on f3 is hanging.", board)
    assert len(problems) == 1 and problems[0].reason == "nothing attacks the knight on f3"
    # A truly hanging piece is not flagged: the rook on f2 hits the undefended knight on f4.
    hang = chess.Board("4k3/8/8/8/5n2/8/5R2/4K3 w - - 0 1")
    assert check_draft("Their knight on f4 is hanging.", hang) == []
    # But an attacked piece with a defender is not hanging.
    held = chess.Board("4k3/8/8/4p3/5n2/8/5R2/4K3 w - - 0 1")
    problems = check_draft("Their knight on f4 is hanging.", held)
    assert len(problems) == 1 and problems[0].reason == "the knight on f4 is defended"


def test_draft_flags_a_hanging_piece_called_safe() -> None:
    # Black to move; the black knight on f4 is attacked by the white rook on f2 and no one
    # defends it, so "safe" is false.
    board = chess.Board("4k3/8/8/8/5n2/8/5R2/4K3 b - - 0 1")
    problems = check_draft("Your knight on f4 is safe.", board)
    assert len(problems) == 1
    assert problems[0].reason == "the knight on f4 is attacked and undefended"
    # A quiet piece called safe passes: white's own knight on f3 in the Italian.
    assert check_draft("Your knight on f3 is safe.", chess.Board(ITALIAN)) == []


def test_draft_accepts_a_relative_pin() -> None:
    # Black knight d3 is pinned to the black queen d7 by the white rook d1 (a relative pin,
    # not to the king). The old king-only test rejected these; the general one must not.
    board = chess.Board("1r2r1k1/3q1pp1/2p5/p3P2Q/2P5/1P1n3P/6P1/3R1RK1 w - - 1 29")
    assert check_draft("The knight on d3 is pinned to the queen.", board) == []
    # The white rook on d1 is not pinned: the queen's line to it is blocked by the knight.
    problems = check_draft("Your rook on d1 is pinned.", board)
    assert len(problems) == 1 and problems[0].reason == "the rook on d1 is not pinned"


def test_draft_verifies_a_fork() -> None:
    # A real fork: the knight on f6 hits both the queen on e8 and the king on g8.
    real = chess.Board("4q1k1/8/5N2/8/8/8/8/7K b - - 0 1")
    assert check_draft("The knight on f6 forks the queen and king.", real) == []
    # A knight that forks nothing is flagged.
    lone = chess.Board("4k3/8/8/8/8/5N2/8/4K3 w - - 0 1")
    problems = check_draft("The knight on f3 forks two pieces.", lone)
    assert len(problems) == 1
    assert problems[0].reason == "the knight on f3 does not fork two pieces"


def test_draft_accepts_a_fork_the_best_move_creates() -> None:
    # The fork is not on the board in front of the player: Nd5-f6 makes it. With the
    # after-best-move frame supplied, naming it is not faulted; without it, it is.
    before = chess.Board("4q1k1/8/8/3N4/8/8/8/7K w - - 0 1")
    after = before.copy(stack=False)
    after.push_san("Nf6")
    assert check_draft("The knight on f6 forks the queen and king.", before, after=[after]) == []
    problems = check_draft("The knight on f6 forks the queen and king.", before)
    assert problems and problems[0].text == "knight on f6"


def test_draft_verifies_a_skewer() -> None:
    # White rook b2 skewers the black queen b7 to the black rook b8 behind it.
    real = chess.Board("1r6/1q6/8/8/8/8/1R6/K6k w - - 0 1")
    assert check_draft("White's rook on b2 skewers the queen and rook.", real) == []
    # With nothing behind the queen there is no skewer.
    none_behind = chess.Board("8/1q6/8/8/8/8/1R6/K6k w - - 0 1")
    problems = check_draft("The rook on b2 skewers two pieces.", none_behind)
    assert len(problems) == 1
    assert problems[0].reason == "the rook on b2 does not skewer anything"


def test_draft_ambiguous_move() -> None:
    board = chess.Board(ITALIAN)
    board.push_san("d3")
    board.push_san("d6")  # now both white knights can reach d2
    problems = check_draft("Nd2 is possible.", board)
    assert len(problems) == 1 and "more than one move" in problems[0].reason
    assert check_draft("Nbd2 is possible.", board) == []


def test_draft_prose_with_no_board_content() -> None:
    board = chess.Board(ITALIAN)
    assert check_draft("Take your time and look at what changed.", board) == []
    assert check_draft("Be careful in area1; the area12 is fine, as is h1n1.", board) == []
    assert check_draft("Point at f7 or a1, bare squares are fine.", board) == []


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
