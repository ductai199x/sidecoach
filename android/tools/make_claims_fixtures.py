"""Record what the desktop makes of a claim and of a coach draft, for the Kotlin to match.

`coach/claims.py` runs in both directions and only the extraction uses a model:
`parse_claims` reads a reply, `verify_claims` checks each claim against the board, and
`check_draft` checks the coach's own sentences before they are shown. All three are pure
functions of text, board, facts and engine lines -- so the engine here is a recorded stub
whose lines go into the fixture, and the model replies are written out verbatim.

Attack and defence are counted by legal moves (`analysis/exchange.py`): a piece pinned to its
king takes nothing and takes nothing back, a king takes back only where it may, a piece lined
up behind an attacker joins in. So beside the fixed claims and drafts, every position is asked
about each of its pieces -- hanging, safe, what attacks and defends it, whether it is pinned,
loose, trapped, forks or skewers -- and the positions include puzzles of the tactical themes.

    cd D:/chess-coaching-companion
    PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python android/tools/make_claims_fixtures.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis.score import human_expected_score  # noqa: E402
from sidecoach.coach import claims as C  # noqa: E402
from sidecoach.coach.disclosure import PositionFacts  # noqa: E402
from sidecoach.events import Line  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
PUZZLES = DESKTOP / "data" / "puzzles" / "sample.csv"
THEMES = DESKTOP / "data" / "puzzles" / "themes-7-120.csv"
# The tactical themes, and how many puzzles of each.
TACTICAL = (
    "pin",
    "skewer",
    "fork",
    "discoveredAttack",
    "xRayAttack",
    "hangingPiece",
    "trappedPiece",
    "capturingDefender",
    "doubleCheck",
    "deflection",
    "attraction",
    "defensiveMove",
)
PER_THEME = 3
# The pieces of a position asked about, most first (by square), and the states claimed.
PIECES_ASKED = 10
STATES = ("pinned", "hanging", "undefended", "safe", "trapped", "forks", "skewers", "en prise")


# --- the model replies `parse_claims` has to survive -----------------------------------------

REPLIES = [
    '[{"kind": "attacks", "subject": "my knight", "target": "e5", "move": null, '
    '"text": "my knight hits e5"}]',
    '```json\n[{"kind": "hanging", "subject": null, "target": "the pawn on d4", '
    '"move": null, "text": "the d4 pawn is hanging"}]\n```',
    '```\n[{"kind":"best","subject":null,"target":null,"move":"Nf3","text":"Nf3 is best"}]\n```',
    '<think>The student says two things, so two claims.</think>\n'
    '[{"kind":"safe","subject":null,"target":"my knight","move":null,"text":"safe"},'
    '{"kind":"threatens","subject":"they","target":"my rook","move":null,"text":"threat"}]',
    '{"kind": "plan", "subject": null, "target": null, "move": null, "text": "push the h-pawn"}',
    '{"claims": [{"kind": "other", "subject": null, "target": null, "move": null, '
    '"text": "I am winning"}]}',
    'Here are the claims:\n[{"kind": "defends", "subject": "the bishop", "target": "d4", '
    '"move": null, "text": "the bishop covers d4"}]\nThat is all.',
    '[{"kind": "nonsense", "subject": null, "target": null, "move": null, "text": "x"},'
    '{"kind": "ATTACKS ", "subject": "  my rook  ", "target": "none", "move": "  ", '
    '"text": "  spaced  "}]',
    '[{"kind": "attacks", "subject": 7, "target": ["e5"], "move": false, "text": 0}]',
    '[{"kind": 5, "text": "no kind"}, "a string", 42, null, '
    '{"kind": "safe", "target": "null", "move": "None"}]',
    "[]",
    "",
    "no json at all, just prose about the position",
    "[{broken json",
    '{"claims": "not a list", "kind": "best", "move": "e4", "text": "e4"}',
    '[{"kind":"threatens","subject":"I","target":null,"move":"Qxf7#","text":"mate on f7"}]',
    "null",
    '   \n\n [{"kind":"hanging","subject":"the knight","target":null,"move":null,"text":"hangs"}] ',
    # Read as Python's json.loads reads them, strict (the parity audit cc-05): org.json alone
    # took unquoted keys, single quotes, comments and trailing text; a field's value is
    # str()'d as Python writes it, and a falsy one is "".
    '[{kind: "hanging", target: "e4", text: "unquoted keys"}]',
    "[{'kind': 'hanging', 'target': 'e4', 'text': 'single quotes'}]",
    '[{"kind": "hanging", /* a comment */ "target": "e4", "text": "a comment"}]',
    '[{"kind": "hanging", "target": "e4", "text": "hash"}] # trailing',
    '{"kind": "safe", "target": "e4", "text": "one"} {"kind": "safe"}',
    '[{"kind": "hanging", "target": "e4", "text": null}, {"kind": "safe", "text": true}, '
    '{"kind": "safe", "text": false}, {"kind": "safe", "text": ["e4", 1, null, 2.5]}, '
    '{"kind": "safe", "text": {"a": null}}, {"kind": "safe", "text": 1.0}, '
    '{"kind": "safe", "text": 0}, {"kind": "safe", "text": ""}, {"kind": "safe", "text": []}]',
    '[{"kind": "best", "move": NaN, "text": NaN}, {"kind": "best", "text": 1e20}, '
    '{"kind": "best", "text": -0.0}, {"kind": "best", "text": 1e-05}]',
    '[{"kind": "hanging", "target": "e4", "text": "caf\\u00e9 \\"quoted\\" \\\\ tab\\t"}]',
    '[{"kind": "hanging", "target": "e4", "text": "a raw\ttab"}]',
    '[{"kind": "hanging", "target": "e4", "text": "a", "text": "last key wins"}]',
    '[{"kind": "hanging", "target": "e4", "text": "trailing comma",}]',
    "[01]",
    '[{"kind": "safe", "text": "x"}, 1.]',
]

# Claim texts as the model is shown them once checked: `f"{v.claim.text!r}: ..."` in
# `cli._claims_check`, Python's repr (the parity audit cm-01).
LINE_TEXTS = [
    "my knight is safe",
    "it's hanging",
    'the "free" pawn',
    "both ' and \"",
    "a back\\slash",
    "tab\there",
    "new\nline and\rreturn",
    "bell\x07 and del\x7f",
    "café, naïve",
    "no break",
    "zero​width",
    "line sep",
    "emoji \U0001F600",
    "private ",
    "c1 control \x85",
    "",
]


# --- the claims every position is asked about ------------------------------------------------
#
# A template is (kind, subject, target, move); "@1" and "@2" stand for the position's first
# and second legal move in SAN, so the hypothetical-move branch fires on every board.

TEMPLATES = [
    ("attacks", "my knight", "e5", None),
    ("attacks", "my knight", "their queen", None),
    ("attacks", "the bishop", "my king", None),
    ("attacks", "f3", "e5", None),
    ("attacks", "d4", "d5", None),
    ("attacks", "my rook", "their king", None),
    ("attacks", "I", "their queen", None),
    ("attacks", "they", "my king", None),
    ("attacks", "my knight", None, None),
    ("attacks", None, "e5", None),
    ("attacks", "the unicorn", "e5", None),
    ("attacks", "my queen", "e4", "@1"),
    ("attacks", "@1", "e5", None),
    ("attacks", "white's knight", "black's pawn", None),
    ("defends", "my knight", "e5", None),
    ("defends", "my bishop", "my pawn", None),
    ("defends", "the rook", "their queen", None),
    ("defends", "e1", "e2", None),
    ("defends", "my king", None, None),
    ("defends", None, None, None),
    ("defends", "my knight", "d4", "@1"),
    ("hanging", None, "the pawn", None),
    ("hanging", None, "my queen", None),
    ("hanging", None, "their knight", None),
    ("hanging", None, "e4", None),
    ("hanging", None, "my king", None),
    ("hanging", "my rook", None, None),
    ("hanging", None, None, None),
    ("hanging", None, "their queen", "@1"),
    ("safe", None, "my knight", None),
    ("safe", None, "my king", None),
    ("safe", None, "e5", None),
    ("safe", None, "my pawn", None),
    ("safe", "d4", None, None),
    ("safe", None, "their rook", None),
    ("safe", None, None, None),
    ("safe", None, "my bishop", "@2"),
    ("threatens", "they", "my queen", None),
    ("threatens", "they", "my king", None),
    ("threatens", "they", "e4", None),
    ("threatens", "they", "their pawn", None),
    ("threatens", "I", "their queen", None),
    ("threatens", "I", "their king", None),
    ("threatens", "they", None, "@1"),
    ("threatens", "I", None, "@1"),
    ("threatens", "I", None, "Qz9"),
    ("threatens", "they", None, "Nf3"),
    ("threatens", "they", None, None),
    ("threatens", "they", "my rook", None),
    ("best", None, None, "@1"),
    ("best", None, None, "@2"),
    ("best", None, None, "Nf3"),
    ("best", None, None, "Qz9"),
    ("best", None, None, "e9"),
    ("best", None, None, None),
    ("best", "@1", None, None),
    ("best", None, None, "O-O"),
    ("plan", None, None, None),
    ("other", "my knight", "e5", "@1"),
]


# --- the coach drafts every position is checked against --------------------------------------

DRAFTS = [
    "Your knight on f3 eyes e5.",
    "The bishop on c4 is aiming at f7.",
    "White's queen on d1 has not moved.",
    "Their rook on a8 is still at home.",
    "The f3-knight covers d4 and h4.",
    "Your c4 bishop and the e4 pawn hold the centre.",
    "The knight on f3 is pinned.",
    "Your rook on a1 is hanging.",
    "The pawn on e4 is loose.",
    "The bishop on c4 is safe.",
    "The knight on f3 forks their queen and rook.",
    "The rook on a1 skewers the king.",
    "The bishop on c4 is trapped.",
    "The queen on d1 is en prise.",
    "The knight on f3 is undefended.",
    "The pawn on d4 is now unprotected.",
    "Consider Nf3, then Bb5 and finally O-O.",
    "Nxf7+ wins the exchange, and after Kxf7 you have Qh5+.",
    "Play e8=Q and the game is over.",
    "What about Qh5 followed by Qxf7#?",
    "Look at f7 and e4 before you decide.",
    "0-0-0 is also possible here.",
    "Bb5! is the move, though a4?! is not.",
    "The open e-file belongs to you.",
    "The d-file is open and your rook wants it.",
    "Take the half-open c-file.",
    "The a-file is now half-open.",
    "Your passed pawn on d5 decides the game.",
    "The pawn on e4 is passed.",
    "Your passed a-pawn is the trump.",
    "The h-pawn is a passed pawn.",
    "Be careful in that area1 of the board; nothing here is a move.",
    "Your knight on f3 is pinned, the rook on a1 is hanging, and Nf3 is impossible.",
    "Nothing concrete at all, just a feeling about the position.",
    # What a line is said to force or keep open (desktop 00894e0): checked on the frames with
    # a check for that side, or after a check the sentence names; left alone where none apply.
    "The check forced their king away.",
    "Your king has to move.",
    "Qh5+ forces the king to step aside.",
    "Bb5+ forces their knight onto d7.",
    "The check forces your queen back to e2.",
    "Rad8 keeps the b-file and e-file open.",
    "The b- and e-files are open.",
    # A file said to have no pawns (desktop cc06e83).
    "The g-file has no pawns, so use it.",
    "There are no pawns on the d-file.",
    # "No pawns of yours on it" is a half-open file, left alone (desktop ba861d6).
    "The b-file has no pawns of yours on it.",
    # The test audit's families (desktop 4365dfa): curly quotes, an adjective before the piece,
    # the state said first, a king driven to a square, the only move, what a move allows, and
    # a piece named by its owner alone.
    "Your opponent’s bishop on c4 is loose.",
    "Your isolated pawn on d4 needs a defender.",
    "Aim at the loose rook on a1.",
    "Qh5+ drives the king to e7.",
    "Kf8 is the only king move.",
    "Nxe5 is the only legal move.",
    "Nf3 allows e5.",
    "Bc4 develops, but it allows Nxe4.",
    "Your queen checks the king.",
    "Their rook attacks the pawn.",
]


class RecordedEngine:
    """A deterministic stand-in: the lines depend only on the position, and are written down."""

    def __init__(self) -> None:
        self.seen: dict[str, list[dict]] = {}

    def analyse(
        self, board: chess.Board, *, depth: int | None = None, best_only: bool = False
    ) -> list[Line]:
        fen = board.fen()
        legal = list(board.legal_moves)
        if not legal:
            self.seen[fen] = []
            return []
        # Four candidate lines whose scores straddle the grading bands (score.about_as_good,
        # score.graded_good), so the near-equal set is sometimes one move and sometimes several,
        # and a move is sometimes good, a little behind, and sometimes not. No search, but the
        # same answer every time for the same board.
        base = (sum(ord(c) for c in fen) % 401) - 200
        behind = (0, 1, 30, 80)
        out: list[Line] = []
        for index in range(min(4, len(legal))):
            pick = legal[(index * 5 + len(legal)) % len(legal)]
            work = board.copy(stack=False)
            work.push(pick)
            reply = list(work.legal_moves)
            moves = [pick.uci()]
            if reply:
                moves.append(reply[(index + len(reply)) % len(reply)].uci())
            cp = base - behind[index]
            out.append(
                Line(
                    moves=moves,
                    depth=14,
                    expected_score=human_expected_score(cp, None),
                    score_cp=cp,
                )
            )
        self.seen[fen] = [
            {
                "moves": list(one.moves),
                "depth": one.depth,
                "expected_score": one.expected_score,
                "score_cp": one.score_cp,
            }
            for one in out
        ]
        return out


def facts_for(board: chess.Board) -> PositionFacts:
    """Candidates, a good move a little behind them and a threat line that belong to this
    position, plus one unreadable each."""
    legal = list(board.legal_moves)
    candidates = [board.san(move) for move in legal[:2]] + ["Qz9"]
    good = [board.san(move) for move in legal[2:3]] + ["Qz8"]
    passed = board.copy(stack=False)
    threats: list[str] = []
    if not board.is_check():
        passed.push(chess.Move.null())
        threats = [passed.san(move) for move in list(passed.legal_moves)[:2]] + ["Zz9"]
    return PositionFacts(
        fen=board.fen(),
        side_to_move="white" if board.turn == chess.WHITE else "black",
        phase="middlegame",
        candidates=candidates,
        good=good,
        threats=threats,
    )


def facts_json(facts: PositionFacts) -> dict:
    return {
        "fen": facts.fen,
        "side_to_move": facts.side_to_move,
        "phase": facts.phase,
        "candidates": list(facts.candidates),
        "good": list(facts.good),
        "threats": list(facts.threats),
    }


def claims_for(board: chess.Board) -> list[C.Claim]:
    legal = list(board.legal_moves)
    sans = [board.san(move) for move in legal[:2]]

    def fill(value: str | None) -> str | None:
        if value == "@1":
            return sans[0] if sans else None
        if value == "@2":
            return sans[1] if len(sans) > 1 else None
        return value

    out = []
    for kind, subject, target, move in TEMPLATES:
        out.append(
            C.Claim(
                kind=C.ClaimKind(kind),
                subject=fill(subject),
                target=fill(target),
                move=fill(move),
                text=f"{kind}: {subject} / {target} / {move}",
            )
        )
    # A move graded Good, a little behind the best (`facts_for`'s `good`), called good and
    # called the best: true, and false (desktop T6, 2026-10-02).
    for san in [board.san(move) for move in legal[2:3]] + ["Qz8"]:
        for said in (f"{san} is a good move", f"{san} is the best move"):
            out.append(C.Claim(kind=C.ClaimKind("best"), subject=None, target=None, move=san,
                               text=said))
    # Each piece of the position by its square: hanging, safe, and what reaches it -- by
    # geometry, so a pinned attacker or a king that may not take back is asked about too.
    for square in asked_squares(board):
        name = chess.square_name(square)
        for kind in ("hanging", "safe"):
            out.append(C.Claim(kind=C.ClaimKind(kind), subject=None, target=name, move=None,
                               text=f"{kind}: {name}"))
        for colour in (chess.WHITE, chess.BLACK):
            for reach in sorted(board.attackers(colour, square))[:2]:
                by = chess.square_name(reach)
                for kind in ("attacks", "defends"):
                    out.append(C.Claim(kind=C.ClaimKind(kind), subject=by, target=name, move=None,
                                       text=f"{kind}: {by} / {name}"))
    # Empty squares, for "safe" there: who of theirs could take a piece put there.
    empties = [sq for sq in chess.SQUARES if board.piece_at(sq) is None]
    for square in empties[:: max(1, len(empties) // 6)][:6]:
        name = chess.square_name(square)
        out.append(C.Claim(kind=C.ClaimKind("safe"), subject=None, target=name, move=None,
                           text=f"safe: {name}"))
    return out


def asked_squares(board: chess.Board) -> list[chess.Square]:
    """The pieces asked about: every piece but the kings, up to PIECES_ASKED, the attacked
    ones first."""
    pieces = [
        sq for sq in chess.SQUARES
        if (p := board.piece_at(sq)) is not None and p.piece_type != chess.KING
    ]
    pieces.sort(key=lambda sq: (not board.is_attacked_by(not board.color_at(sq), sq), sq))
    return pieces[:PIECES_ASKED]


def drafts_for(board: chess.Board) -> list[str]:
    """The fixed drafts, and a tactic claimed of each piece asked about."""
    out = list(DRAFTS)
    for square in asked_squares(board):
        piece = board.piece_at(square)
        assert piece is not None
        word = chess.piece_name(piece.piece_type)
        where = chess.square_name(square)
        for state in STATES:
            out.append(f"The {word} on {where} is {state}." if not state.endswith("s")
                       or state == "pinned" else f"The {word} on {where} {state}.")
    return out


def claim_json(claim: C.Claim) -> dict:
    return {
        "kind": str(claim.kind),
        "subject": claim.subject,
        "target": claim.target,
        "move": claim.move,
        "text": claim.text,
    }


def verdicts_json(verdicts: list[C.Verdict]) -> list[dict]:
    return [{"result": v.result, "reason": v.reason} for v in verdicts]


def after_frames(board: chess.Board) -> list[chess.Board]:
    """The boards along a short line, as the coach's draft check is given them."""
    frames: list[chess.Board] = []
    work = board.copy(stack=False)
    for _ in range(2):
        legal = list(work.legal_moves)
        if not legal:
            break
        work.push(legal[len(legal) // 2])
        frames.append(work.copy(stack=False))
    return frames


def case_for(board: chess.Board) -> dict:
    engine = RecordedEngine()
    facts = facts_for(board)
    stale = PositionFacts(
        fen=chess.Board().fen(),
        side_to_move="white",
        phase="opening",
        candidates=["e4", "d4"],
        threats=["e5"],
    )
    claims = claims_for(board)
    frames = after_frames(board)
    return {
        "fen": board.fen(),
        "facts": facts_json(facts),
        "stale_facts": facts_json(stale),
        "after": [frame.fen() for frame in frames],
        "claims": [claim_json(claim) for claim in claims],
        # Bare, with the facts, with the engine, and with facts that belong elsewhere.
        "bare": verdicts_json(C.verify_claims(claims, board)),
        "with_facts": verdicts_json(C.verify_claims(claims, board, facts=facts)),
        "with_engine": verdicts_json(C.verify_claims(claims, board, engine=engine, depth=14)),
        "with_stale": verdicts_json(
            C.verify_claims(claims, board, facts=stale, engine=engine, depth=14)
        ),
        "drafts": [
            {
                "text": text,
                "now": [
                    {"text": p.text, "reason": p.reason} for p in C.check_draft(text, board)
                ],
                "line": [
                    {"text": p.text, "reason": p.reason}
                    for p in C.check_draft(text, board, after=frames)
                ],
            }
            for text in drafts_for(board)
        ],
        "engine": engine.seen,
    }


def line_json(line: Line) -> dict:
    return {
        "moves": list(line.moves),
        "depth": line.depth,
        "expected_score": line.expected_score,
        "score_cp": line.score_cp,
    }


def desktop_tests() -> dict:
    """Every call `tests/test_claims.py` makes to verify_claims, check_draft and
    extract_claims, with what the desktop answered: the tests' own positions, claims, drafts
    and model replies, replayed on the phone (TG-02, TG-03). Each test runs as pytest would;
    the one that needs the local model is left out, as `-m 'not slow'` and its skip leave it.
    """
    import importlib

    from desktop_tests import desktop_tests

    tests = importlib.import_module("tests.test_claims")
    real_verify, real_draft, real_extract = (
        tests.verify_claims,
        tests.check_draft,
        tests.extract_claims,
    )
    out: dict[str, list] = {"verify": [], "drafts": [], "extract": []}
    current = {"test": ""}

    def verify(claims, board, *, facts=None, engine=None, depth=12):  # noqa: ANN001, ANN202
        seen: dict[str, list[dict]] = {}
        recording = None
        if engine is not None:

            class Recording:
                def analyse(self, b, *, depth=None, best_only=False):  # noqa: ANN001, ANN202
                    lines = engine.analyse(b, depth=depth)
                    seen[b.fen()] = [line_json(one) for one in lines]
                    return lines

            recording = Recording()
        verdicts = real_verify(claims, board, facts=facts, engine=recording, depth=depth)
        out["verify"].append(
            {
                "test": current["test"],
                "fen": board.fen(),
                "claims": [claim_json(c) for c in claims],
                "facts": None if facts is None else facts_json(facts),
                "engine": None if engine is None else seen,
                "depth": depth,
                "verdicts": verdicts_json(verdicts),
            }
        )
        return verdicts

    def draft(text, board, *, after=(), us=None):  # noqa: ANN001, ANN202
        problems = real_draft(text, board, after=after, us=us)
        out["drafts"].append(
            {
                "test": current["test"],
                "fen": board.fen(),
                "text": text,
                "after": [frame.fen() for frame in after],
                "us": us,
                "problems": [{"text": p.text, "reason": p.reason} for p in problems],
            }
        )
        return problems

    def extract(text, board, provider, *, thinking=True, max_tokens=4000):  # noqa: ANN001, ANN202
        calls: list[dict] = []

        class Recording:
            name = "recording"

            def chat(self, system, messages, *, max_tokens=1500, temperature=0.7, thinking=False):  # noqa: ANN001, ANN202
                call = {
                    "system": system,
                    "messages": [dict(m) for m in messages],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "thinking": thinking,
                }
                calls.append(call)
                try:
                    completion = provider.chat(
                        system,
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        thinking=thinking,
                    )
                except Exception as exc:  # noqa: BLE001
                    call["raises"] = repr(exc)
                    raise
                call["reply"] = completion.text
                return completion

        claims = real_extract(text, board, Recording(), thinking=thinking, max_tokens=max_tokens)
        out["extract"].append(
            {
                "test": current["test"],
                "text": text,
                "fen": board.fen(),
                "thinking": thinking,
                "max_tokens": max_tokens,
                "calls": calls,
                "claims": [claim_json(c) for c in claims],
            }
        )
        return claims

    tests.verify_claims, tests.check_draft, tests.extract_claims = verify, draft, extract
    ran = []
    try:
        for name, fn in desktop_tests(tests, frozenset({"test_extract_with_the_local_provider"})):
            current["test"] = name
            fn()  # the desktop's own assertions hold as the calls are recorded
            ran.append(name)
    finally:
        tests.verify_claims, tests.check_draft, tests.extract_claims = (
            real_verify,
            real_draft,
            real_extract,
        )
    out["ran"] = ran
    return out


def claim_lines() -> list[dict]:
    """Each LINE_TEXTS claim checked on the opening board, and the line the model is shown."""
    board = chess.Board()
    out = []
    for text in LINE_TEXTS:
        one = C.Claim(kind=C.ClaimKind("hanging"), subject=None, target="e4", move=None, text=text)
        verdict = C.verify_claims([one], board)[0]
        out.append(
            {
                "claim": claim_json(one),
                # cli._claims_check's own format.
                "line": f"{verdict.claim.text!r}: {verdict.result}, {verdict.reason}",
            }
        )
    return out


def main() -> None:
    boards = [chess.Board()]
    opening = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4", "O-O"):
        opening.push_san(san)
        boards.append(opening.copy(stack=False))
    # A few settled positions: an endgame, a check, a stalemate and a mate.
    for fen in (
        "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
        "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
        "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4k3/8/8/8/8/8/4P3/R3K2R w KQ - 0 1",
    ):
        boards.append(chess.Board(fen))
    if PUZZLES.exists():
        with PUZZLES.open(newline="", encoding="utf-8") as handle:
            for row in list(csv.DictReader(handle))[:24]:
                board = chess.Board(row["FEN"])
                for index, uci in enumerate(row["Moves"].split()):
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        break
                    if index % 2 == 1:
                        boards.append(board.copy(stack=False))
                    board.push(move)

    if THEMES.exists():
        taken: dict[str, int] = {}
        with THEMES.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                names = [t for t in row["Themes"].split() if t in TACTICAL]
                if not names or all(taken.get(t, 0) >= PER_THEME for t in names):
                    continue
                for t in names:
                    taken[t] = taken.get(t, 0) + 1
                board = chess.Board(row["FEN"])
                moves = row["Moves"].split()
                for index, uci in enumerate(moves[:4]):
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        break
                    board.push(move)
                    if index in (0, 1, 2):
                        boards.append(board.copy(stack=False))

    cases = [case_for(board) for board in boards]
    replies = [{"text": text, "claims": [claim_json(c) for c in C.parse_claims(text)]}
               for text in REPLIES]

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "system": C.EXTRACT_SYSTEM,
        "replies": replies,
        "cases": cases,
        "lines": claim_lines(),
        "tests": desktop_tests(),
    }
    (OUT / "claims.json").write_text(json.dumps(payload), encoding="utf-8")
    size = (OUT / "claims.json").stat().st_size
    print(
        f"{len(cases)} positions x {len(TEMPLATES)}+ claims + {len(DRAFTS)}+ drafts, "
        f"{len(replies)} replies -> {OUT / 'claims.json'} ({size / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    main()
