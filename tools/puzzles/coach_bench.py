"""The offline coach against Lichess puzzles: does what it says at each rung agree with the
puzzle, and is every word of it true on the board?

For each puzzle the position is the one after the opponent's setup move; the coach reads it
as it would live (`facts_from_analysis` over the app's analysis, `analysis_cache`) and says
its idea, focus, move and lesson templates. The Lichess solution and themes are the answer
key, read only here, never by the coach (a real game has no tags). Each puzzle is checked:

  move     the coach's move is the solution's first move (a mate puzzle: any move that
           mates as fast; otherwise the solution line scored as well as the engine's)
  goal     a mate puzzle reads as a mate, no slower than the puzzle's; a crushing or
           advantage puzzle as a win, a promotion, an attack on the king or a mate
  amount   what the coach says a win wins is no less than the solution wins, less a pawn,
           where the solution is the engine's own line: counted one reply past its end (a
           solution stops on the capture, before the knight it left hanging is taken
           back); where the defence differs the engine's is the better one, and the
           coach reads that
  leak     the idea and focus rungs pass the leak check and never name the solution (a
           pawn move is written as its square, and squares may be named at the focus rung:
           there it is only a move when the coach's own leak check says so)
  sacrifice (reference) a puzzle Lichess tags a sacrifice gets one said at the idea rung,
           and one said is material given up in the line
  engine   (reference) a crushing or advantage puzzle the engine at the app's depth does
           not see as better (below 0.6): the coach says what its engine sees, so a goal
           miss there is the engine's, counted apart
  grounded every "your/their <piece> on <square>" is true on the board, every "the <piece>
           on <square>" in the move rung true on the board or along the line
  claims   every tactic the move rung describes holds on the board it happens on,
           checked here from the rules, not by the tagger that found it: a fork's piece
           attacks every target and no bystander can take it for profit (a pinned one
           cannot); a pin's three pieces share an open line and the piece behind is the
           king, worth more than the pinner, or undefended (Tai's rule); a
           skewer's front piece is the more valuable; a loose piece has no defender; a
           trapped piece has no square it is not attacked on; a line "now bears" is open
  text     every rung says something (no "still reading", no "None", no stock question)
  pattern  (reference) the pattern the coach names is among the puzzle's themes

A puzzle is correct when move, goal, amount, leak, grounded, claims and text all hold. The
engine half is cached (`analysis_cache`); a rerun after a change to the coach takes seconds.

  uv run python tools/puzzles/coach_bench.py --source sample --n 400
  uv run --with zstandard python tools/puzzles/coach_bench.py --source full --n 100000
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import multiprocessing as mp
import os
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess  # noqa: E402

PUZZLES = ROOT / "data/puzzles"
RESULTS = PUZZLES / "bench"
# The pattern names the coach can give that Lichess also tags.
PATTERNS = {
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "doubleCheck",
    "hangingPiece",
    "trappedPiece",
    "deflection",
    "attraction",
    "capturingDefender",
    "smotheredMate",
    "backRankMate",
    "anastasiaMate",
    "arabianMate",
    "bodenMate",
    "hookMate",
    "dovetailMate",
    "doubleBishopMate",
    "killBoxMate",
    "vukovicMate",
    "swallowstailMate",
    "epauletteMate",
    "triangleMate",
    "blindSwineMate",
    "cornerMate",
    "operaMate",
    "pillsburysMate",
    "morphysMate",
    "balestraMate",
}
_ALIASES = {"absolutePin": "pin"}
# Lines no rung may say: the stock questions the rewrite removed (Tai, 2026-09-23).
_STOCK = re.compile(
    # "None" leaked from code, case and all: "none of your rooks" is English.
    r"whole board|most dangerous idea|doing the least|still reading|(?-i:\bNone\b)|  ",
    re.I,
)
_CONCRETE = re.compile(
    r"\b[a-h][1-8]\b|\b[a-h]-file\b|[KQRBN]?[a-h]?x?[a-h][1-8]|opening theory|legal move"
)
_PIECE_ON = re.compile(r"\b(their|your) (king|queen|rook|bishop|knight|pawn) on ([a-h][1-8])")
_THE_PIECE_ON = re.compile(r"\bthe (king|queen|rook|bishop|knight|pawn) on ([a-h][1-8])")
_TYPES = {
    "king": chess.KING,
    "queen": chess.QUEEN,
    "rook": chess.ROOK,
    "bishop": chess.BISHOP,
    "knight": chess.KNIGHT,
    "pawn": chess.PAWN,
}


# ---- the puzzles ------------------------------------------------------------------------


def _rows_sample() -> list[dict]:
    with (PUZZLES / "sample.csv").open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _rows_full(n: int, seed: int) -> list[dict]:
    """`n` puzzles drawn evenly from the whole Lichess database (a reservoir sample),
    kept in a file of their own so the next run skips the five million rows."""
    kept = PUZZLES / f"full-{seed}-{n}.csv"
    if kept.exists():
        with kept.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    import zstandard  # pyright: ignore[reportMissingImports]  (uv run --with zstandard)

    rng = random.Random(seed)
    chosen: list[dict] = []
    with (PUZZLES / "lichess_db_puzzle.csv.zst").open("rb") as raw:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
        for i, row in enumerate(csv.DictReader(text)):
            if len(chosen) < n:
                chosen.append(row)
            else:
                j = rng.randrange(i + 1)
                if j < n:
                    chosen[j] = row
    fields = list(chosen[0].keys())
    with kept.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(chosen)
    return chosen


def _rows_themes(per_theme: int, seed: int) -> list[dict]:
    """Up to `per_theme` puzzles for every theme in the whole Lichess database (a reservoir
    per theme, one pass), so the themes a balanced sample never draws -- equality,
    zugzwang, intermezzo, quiet moves -- are measured too. Kept in a file of its own."""
    kept = PUZZLES / f"themes-{seed}-{per_theme}.csv"
    if kept.exists():
        with kept.open(encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    import zstandard  # pyright: ignore[reportMissingImports]  (uv run --with zstandard)

    rng = random.Random(seed)
    pools: dict[str, list[dict]] = defaultdict(list)
    seen: Counter[str] = Counter()
    fields: list[str] = []
    with (PUZZLES / "lichess_db_puzzle.csv.zst").open("rb") as raw:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(raw), encoding="utf-8")
        reader = csv.DictReader(text)
        fields = list(reader.fieldnames or [])
        for row in reader:
            for theme in row["Themes"].split():
                seen[theme] += 1
                pool = pools[theme]
                if len(pool) < per_theme:
                    pool.append(row)
                else:
                    j = rng.randrange(seen[theme])
                    if j < per_theme:
                        pool[j] = row
    chosen = list({r["PuzzleId"]: r for pool in pools.values() for r in pool}.values())
    with kept.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(chosen)
    (PUZZLES / f"themes-{seed}-{per_theme}.counts.json").write_text(
        json.dumps(dict(seen.most_common()), indent=1), encoding="utf-8"
    )
    return chosen


# What saying a theme looks like in the coach's words: its goal, its pattern, or a phrase at
# the idea rung. A theme with no entry is one the coach has no words for yet.
def conveys(theme: str, record: dict) -> bool | None:
    idea = record["texts"]["idea"].lower()
    goal, pattern = record["goal"], record["pattern"]
    if theme.startswith("mateIn") or theme == "mate":
        return goal == "mate"
    if theme.endswith("Mate"):
        return pattern == theme
    if theme in (
        "fork",
        "pin",
        "skewer",
        "discoveredAttack",
        "doubleCheck",
        "trappedPiece",
        "hangingPiece",
        "capturingDefender",
        "deflection",
        "attraction",
    ):
        named = {
            "discoveredAttack": "discovered",
            "doubleCheck": "double check",
            "trappedPiece": "trapped",
            "hangingPiece": "loose",
            "capturingDefender": "defender",
            "deflection": "pulled away",
            "attraction": "lured",
        }.get(theme, theme)
        return pattern == theme or named in idea
    if theme == "discoveredCheck":
        return "discovered" in idea or pattern == "doubleCheck"
    if theme == "sacrifice":
        return record["sacrifice"]["said"]
    if theme in ("promotion", "advancedPawn", "underPromotion"):
        return goal == "promote" or "queen" in idea and "pawn" in idea
    if theme in ("exposedKing", "kingsideAttack", "queensideAttack", "attackingF2F7"):
        return goal in ("mate", "attack") or "king" in idea
    if theme == "clearance":
        return "clear" in idea
    if theme in ("crushing", "advantage"):
        return goal in ("mate", "win", "promote", "attack", "combination")
    if theme == "pin" and "pinned" in idea:
        return True
    if theme == "intermezzo":
        return "in-between" in idea
    if theme == "quietMove":
        return "quiet move" in idea
    if theme == "enPassant":
        return "en passant" in record["texts"]["move"].lower()
    if theme == "equality":
        return goal == "draw" or bool(re.search(r"level|back|hold|save", idea))
    if theme in ("pawnEndgame", "zugzwang"):
        return goal in ("convert", "king", "breakthrough", "promote")
    if theme in (
        "interference",
        "intermezzo",
        "xRayAttack",
        "quietMove",
        "defensiveMove",
        "enPassant",
        "castling",
    ):
        return False  # no words for it yet
    return None  # length, phase, source: not an idea


def position(row: dict) -> tuple[chess.Board, list[str]]:
    """The puzzle's position (after the setup move) and its solution in UCI."""
    board = chess.Board(row["FEN"])
    moves = row["Moves"].split()
    board.push_uci(moves[0])
    return board, moves[1:]


# ---- one puzzle -------------------------------------------------------------------------


def _mate_in(themes: set[str], solution: list[str]) -> int | None:
    for n in range(1, 6):
        if f"mateIn{n}" in themes:
            return n
    if "mate" in themes:
        return (len(solution) + 1) // 2
    return None


def _grounded(text: str, board: chess.Board, us: chess.Color) -> list[str]:
    """What `text` says stands somewhere that does not."""
    wrong = []
    for whose, name, square in _PIECE_ON.findall(text):
        piece = board.piece_at(chess.parse_square(square))
        colour = us if whose == "your" else not us
        if piece is None or piece.piece_type != _TYPES[name] or piece.color != colour:
            wrong.append(f"{whose} {name} on {square}")
    return wrong


def _grounded_on_line(text: str, boards: list[chess.Board]) -> list[str]:
    wrong = []
    for name, square in _THE_PIECE_ON.findall(text):
        sq = chess.parse_square(square)
        if not any(
            (p := b.piece_at(sq)) is not None and p.piece_type == _TYPES[name] for b in boards
        ):
            wrong.append(f"the {name} on {square}")
    return wrong


_P = r"(king|queen|rook|bishop|knight|pawn) on ([a-h][1-8])"
_FORK = re.compile(rf"the {_P} forks ((?:the {_P}(?:, | and )?)+)")
_PIN = re.compile(rf"the {_P} is pinned to the {_P} by the {_P}")
_SKEWER = re.compile(rf"the {_P} skewers the {_P} and the {_P} behind it")
_LOOSE = re.compile(rf"the {_P} is (loose|trapped)")
_BEARS = re.compile(rf"the {_P} now bears on the {_P}")
_VALUE = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 100}


def _between(a: chess.Square, b: chess.Square) -> list[chess.Square]:
    return list(chess.SquareSet(chess.between(a, b)))


def _slides(board: chess.Board, frm: chess.Square, to: chess.Square) -> bool:
    """The piece on `frm` attacks `to` along a line, nothing between (x-rays not)."""
    return to in board.attacks(frm)


def _claim_holds(kind: str, parts: tuple, boards: list[chess.Board]) -> bool:
    """Whether a described tactic is true on any of `boards` (the blow's before and after)."""

    def at(board: chess.Board, name: str, square: str) -> chess.Piece | None:
        piece = board.piece_at(chess.parse_square(square))
        return piece if piece is not None and piece.piece_type == _TYPES[name] else None

    for board in boards:
        if kind == "fork":
            (fname, fsq), targets = parts
            forker = at(board, fname, fsq)
            if forker is None or len(targets) < 2:
                continue
            ok = True
            for name, square in targets:
                target = at(board, name, square)
                if target is None or target.color == forker.color:
                    ok = False
                    break
                if chess.parse_square(square) not in board.attacks(chess.parse_square(fsq)):
                    ok = False
                    break
            # Tai's rule: a fork whose piece a bystander can take for profit forks nothing.
            # A capture is a legal move of the side to move: never a pinned piece, never a
            # capture that leaves the king in check (a discovered check must be answered).
            f = chess.parse_square(fsq)
            hit = {chess.parse_square(sq) for _, sq in targets}
            takes = (
                [m.from_square for m in board.legal_moves if m.to_square == f]
                if board.turn != forker.color
                else []
            )
            bystanders = [a for a in takes if a not in hit]
            cheap = any(
                _VALUE[chess.piece_name(board.piece_type_at(a))] < _VALUE[fname]
                for a in bystanders
                if board.piece_type_at(a)
            )
            loose = bool(bystanders) and not board.attackers(forker.color, f)
            if ok and not cheap and not loose:
                return True
        elif kind == "pin":
            (bn, bs), (cn, cs), (an, as_) = parts
            front, behind, pinner = at(board, bn, bs), at(board, cn, cs), at(board, an, as_)
            if not (front and behind and pinner):
                continue
            if front.color != behind.color or pinner.color == front.color:
                continue
            a, b, c = (chess.parse_square(x) for x in (as_, bs, cs))
            line = chess.SquareSet(chess.ray(a, c))
            if not line or b not in line or b not in chess.SquareSet(chess.between(a, c)):
                continue
            if any(board.piece_at(x) for x in _between(a, c) if x != b):
                continue
            if not _slides(board, a, b):
                continue
            # Tai's rule: the king behind, or something the pin can win -- with the front
            # piece gone, the pinner takes it and the exchange on its square, played out
            # with legal moves, nets something.
            if cn == "king" or _pin_wins(board, a, b, c, pinner.color):
                return True
        elif kind == "skewer":
            (an, as_), (bn, bs), (cn, cs) = parts
            slider, front, back = at(board, an, as_), at(board, bn, bs), at(board, cn, cs)
            if not (slider and front and back) or front.color != back.color:
                continue
            if slider.color == front.color:
                continue
            a, b, c = (chess.parse_square(x) for x in (as_, bs, cs))
            if b not in chess.SquareSet(chess.between(a, c)):
                continue
            if any(board.piece_at(x) for x in _between(a, c) if x != b):
                continue
            if _slides(board, a, b) and _VALUE[bn] > _VALUE[cn]:
                return True
        elif kind in ("loose", "trapped"):
            (name, square), state = parts
            piece = at(board, name, square)
            if piece is None:
                continue
            sq = chess.parse_square(square)
            if state == "loose":
                # Loose: taken, nothing can take back -- by legal moves on both sides.
                if _nothing_takes_back(board, sq, piece.color):
                    return True
            elif _trapped(board, sq, piece.color):
                return True
        elif kind == "bears":
            (an, as_), (bn, bs) = parts
            slider, target = at(board, an, as_), at(board, bn, bs)
            opposed = slider is not None and target is not None and slider.color != target.color
            if opposed and chess.parse_square(bs) in board.attacks(chess.parse_square(as_)):
                return True
    return False


def _as_turn(board: chess.Board, color: chess.Color) -> chess.Board:
    view = board.copy(stack=False)
    if view.turn != color:
        view.push(chess.Move.null())
    return view


def _pin_wins(
    board: chess.Board, a: chess.Square, b: chess.Square, c: chess.Square, color: chess.Color
) -> bool:
    """With the front piece off the board, the pinner on `a` takes on `c` legally and the
    exchange there nets it something."""
    from sidecoach.analysis import exchange

    probe = board.copy(stack=False)
    probe.remove_piece_at(b)
    view = _as_turn(probe, color)
    move = chess.Move(a, c)
    return move in view.legal_moves and exchange.exchange(view, move) > 0


def _nothing_takes_back(board: chess.Board, sq: chess.Square, color: chess.Color) -> bool:
    """The piece of `color` on `sq` can be taken legally and nothing takes back legally."""
    view = _as_turn(board, not color)
    for move in [m for m in view.legal_moves if m.to_square == sq]:
        after = view.copy(stack=False)
        after.push(move)
        if not any(m.to_square == sq for m in after.legal_moves):
            return True
    return False


def _trapped(board: chess.Board, sq: chess.Square, color: chess.Color) -> bool:
    """The piece of `color` on `sq` is lost whatever its side plays: after every legal move
    of that side it can still be won (a move of the piece itself included)."""
    from sidecoach.analysis import exchange

    view = _as_turn(board, color)
    if not exchange.can_be_won(view, sq):
        return False
    for move in view.legal_moves:
        after = view.copy(stack=False)
        after.push(move)
        where = move.to_square if move.from_square == sq else sq
        piece = after.piece_at(where)
        if piece is None or piece.color != color or not exchange.can_be_won(after, where):
            return False
    return True


def claims_wrong(text: str, boards: list[chess.Board]) -> list[str]:
    """The tactics `text` describes that do not hold on `boards`."""
    wrong = []
    for m in _FORK.finditer(text):
        targets = re.findall(_P, m.group(3))
        if not _claim_holds("fork", ((m.group(1), m.group(2)), targets), boards):
            wrong.append(m.group(0))
    for m in _PIN.finditer(text):
        g = m.groups()
        if not _claim_holds("pin", ((g[0], g[1]), (g[2], g[3]), (g[4], g[5])), boards):
            wrong.append(m.group(0))
    for m in _SKEWER.finditer(text):
        g = m.groups()
        if not _claim_holds("skewer", ((g[0], g[1]), (g[2], g[3]), (g[4], g[5])), boards):
            wrong.append(m.group(0))
    for m in _LOOSE.finditer(text):
        g = m.groups()
        if not _claim_holds(
            "loose" if g[2] == "loose" else "trapped", ((g[0], g[1]), g[2]), boards
        ):
            wrong.append(m.group(0))
    for m in _BEARS.finditer(text):
        g = m.groups()
        if not _claim_holds("bears", ((g[0], g[1]), (g[2], g[3])), boards):
            wrong.append(m.group(0))
    return wrong


def solution_gain(board: chess.Board, solution: list[str], pv: list[str]) -> int | None:
    """What the solution wins, in pawns, counted as the coach counts its own line
    (`idea.line_gain`) along the engine's line to one reply past the solution's end; None
    when the engine's line is not the solution."""
    from sidecoach.coach import idea as idea_of

    if pv[: len(solution)] != solution:
        return None
    moves = [chess.Move.from_uci(u) for u in pv[: len(solution) + 1]]
    boards = [board.copy(stack=False)]
    for move in moves:
        after = boards[-1].copy(stack=False)
        after.push(move)
        boards.append(after)
    return idea_of.line_gain(boards, moves, len(moves))[0]


def score(item: tuple[dict, str]) -> dict:
    """Everything the coach says about one puzzle, and the checks on it."""
    from sidecoach.events import Analysis

    row, analysis_json = item
    board, solution = position(row)
    return score_board(
        board,
        solution,
        set(row["Themes"].split()),
        Analysis.model_validate_json(analysis_json),
        row["PuzzleId"],
        int(row["Rating"]),
    )


def score_board(
    board: chess.Board,
    solution: list[str],
    themes: set[str],
    analysis: object,
    pid: str,
    rating: int,
) -> dict:
    """The checks on what the coach says about `board`, whose answer is `solution` (UCI)
    and whose themes are `themes` -- a Lichess puzzle, or a ChessQA tactic."""
    from sidecoach.coach import idea as idea_of
    from sidecoach.coach.coach import (
        COST_YOU,
        focus_template,
        idea_template,
        lesson_template,
        move_template,
    )
    from sidecoach.coach.disclosure import leak_check
    from sidecoach.coach.facts import facts_from_analysis
    from sidecoach.events import Analysis, DisclosureLevel

    assert isinstance(analysis, Analysis)
    us = board.turn
    facts = facts_from_analysis(board, analysis, user=us)
    idea = idea_of.read(board, analysis)
    texts = {
        "idea": idea_template(facts),
        "focus": focus_template(facts),
        "move": move_template(facts),
    }
    best = facts.candidates[0] if facts.candidates else None
    texts["lesson"] = lesson_template(facts, best, COST_YOU[0].format(best=best)) if best else ""
    first = chess.Move.from_uci(solution[0])
    first_san = board.san(first)
    checks: dict[str, bool] = {}
    notes: dict[str, str] = {}

    # The move.
    mate_n = _mate_in(themes, solution)
    top = analysis.lines[0] if analysis.lines else None
    same = best is not None and best == first_san
    if mate_n is not None:
        fast = top is not None and top.mate is not None and 0 < top.mate <= mate_n
        checks["move"] = same or fast
    else:
        # A second winning move the puzzle did not want is still the move, when the
        # engine scores the solution's as well.
        equal = any(
            line.moves
            and line.moves[0] == solution[0]
            and top is not None
            and top.expected_score - line.expected_score <= 0.02
            for line in analysis.lines
        )
        checks["move"] = same or equal
    if not checks["move"]:
        notes["move"] = f"coach {best}, puzzle {first_san}"

    # The goal.
    engine_sees = top is not None and top.expected_score >= 0.6
    # Where the engine at the app's depth does not see what the puzzle is: a mate it does
    # not find, a crushing or advantage position it does not call better, an equality it
    # calls better. The coach says what its engine sees; such a puzzle is counted apart.
    engine_differs = (
        (mate_n is not None and (top is None or top.mate is None or top.mate <= 0))
        or (bool(themes & {"crushing", "advantage"}) and not engine_sees)
        or ("equality" in themes and top is not None and top.expected_score >= 0.62)
    )
    if mate_n is not None:
        # A mate the engine finds slower than the puzzle's is still the mate.
        checks["goal"] = idea.goal == "mate"
    elif "equality" in themes:
        # Saving the game: never said as a win.
        idea_text = texts["idea"].lower()
        claims_win = "winning attack" in idea_text or (
            "you can win" in idea_text and not re.search(r"level|back", idea_text)
        )
        checks["goal"] = not claims_win and idea.goal not in ("mate", "attack")
    elif themes & {"crushing", "advantage"} and engine_sees and top.expected_score < 0.8:
        # The engine calls it better, not won: a plan or a threat to meet is an honest
        # reading; a claim of a win the engine does not see would not be.
        checks["goal"] = idea.goal != "mate" and not (
            idea.goal == "attack" and "winning attack" in texts["idea"]
        )
    elif themes & {"crushing", "advantage"} and engine_sees:
        # Won by the engine's score: said as won -- a tactic, a technique, or "you are
        # winning" before the plan.
        if idea.goal in ("plan", "defend"):
            checks["goal"] = "winning" in texts["idea"]
        else:
            checks["goal"] = idea.goal in (
                "win",
                "promote",
                "mate",
                "attack",
                "combination",
                "convert",
                "king",
                "breakthrough",
            )
    if "goal" in checks and not checks["goal"]:
        notes["goal"] = f"coach {idea.goal} (mate {idea.mate_in}, gain {idea.gain})"

    # How much a win wins.
    theirs = solution_gain(board, solution, list(top.moves) if top else [])
    if idea.goal == "win" and theirs is not None:
        checks["amount"] = idea.gain >= theirs - 1
        if not checks["amount"]:
            notes["amount"] = f"coach {idea.won} ({idea.gain}), solution wins {theirs}"

    # No leak below the move rung.
    leaks = leak_check(texts["idea"], facts, DisclosureLevel.IDEA) + leak_check(
        texts["focus"], facts, DisclosureLevel.FOCUS
    )
    bare = first_san.rstrip("+#")
    pawn_move = re.fullmatch(r"[a-h][1-8](=[QRBN])?", bare) is not None
    rungs = ("idea",) if pawn_move else ("idea", "focus")
    # The moves the first hint names when several are as good (or in opening theory) are
    # said in the open; the answer is never named otherwise.
    named = [
        t
        for t in rungs
        if re.search(rf"(?<![\w]){re.escape(bare)}", texts[t]) and first_san not in facts.choices
    ]
    checks["leak"] = not leaks and not named
    if not checks["leak"]:
        notes["leak"] = "; ".join(leaks + [f"{t} names {bare}" for t in named])

    # Every piece named stands where it is said to.
    replay = [board.copy(stack=False)]
    for move in idea.line:
        after = replay[-1].copy(stack=False)
        after.push(move)
        replay.append(after)
    wrong = _grounded(texts["idea"] + " " + texts["focus"], board, us)
    wrong += _grounded_on_line(texts["move"], replay)
    checks["grounded"] = not wrong
    if wrong:
        notes["grounded"] = ", ".join(wrong)

    # Every tactic described holds where it happens.
    frames = [board]
    if idea.blow_board is not None:
        frames = [idea.blow_board]
        if idea.blow_move is not None:
            after_blow = idea.blow_board.copy(stack=False)
            after_blow.push(idea.blow_move)
            frames.append(after_blow)
    false = claims_wrong(texts["move"] + " " + texts["lesson"], frames)
    checks["claims"] = not false
    if false:
        notes["claims"] = "; ".join(false)

    # The first hint names the board: a square, a file, a move, opening theory -- never a
    # principle alone (Tai, 2026-09-24: "fight for the centre" teaches nothing mid-game).
    checks["concrete"] = bool(_CONCRETE.search(texts["idea"]))
    if not checks["concrete"]:
        notes["concrete"] = texts["idea"][:160]

    # Every rung says something of this position.
    stock = [t for t in ("idea", "focus", "move") if not texts[t] or _STOCK.search(texts[t])]
    checks["text"] = not stock
    if stock:
        notes["text"] = ", ".join(stock)

    # The sacrifices, for reference: Lichess's tag against what the idea rung says.
    says_sac = bool(re.search(r"sacrific|giv(e|en|ing) up", texts["idea"], re.I))
    sac_tagged = "sacrifice" in themes

    # The pattern, for reference.
    pattern = idea.blow.name if idea.blow is not None else None
    pattern = _ALIASES.get(pattern or "", pattern)
    tagged = themes & PATTERNS
    agree = None
    if pattern in PATTERNS and tagged:
        agree = pattern in tagged
    return {
        "id": pid,
        "rating": rating,
        "themes": sorted(themes),
        "fen": board.fen(),
        "solution": first_san,
        "best": best,
        "goal": idea.goal,
        "mate_in": idea.mate_in,
        "won": idea.won,
        "gain": idea.gain,
        "solution_gain": theirs,
        "pattern": pattern,
        "pattern_agrees": agree,
        "sacrifice": {"tagged": sac_tagged, "said": says_sac, "count": len(idea.sacrifices)},
        "engine_sees": engine_sees,
        "engine_differs": engine_differs,
        "engine_score": round(top.expected_score, 3) if top else None,
        "texts": texts,
        "checks": checks,
        "notes": notes,
        "correct": engine_differs or all(checks.values()),
    }


# ---- the run ----------------------------------------------------------------------------


def _commit() -> str:
    head = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    ).stdout.strip()
    return head + ("-dirty" if dirty else "")


def summarise(records: list[dict], show: int) -> dict:
    n = len(records)
    counts = Counter()
    scored = Counter()
    for r in records:
        for name, ok in r["checks"].items():
            scored[name] += 1
            counts[name] += ok
    correct = sum(r["correct"] for r in records)
    print(f"\n{n} puzzles: {correct / n:.2%} correct")
    for name in ("move", "goal", "amount", "leak", "grounded", "claims", "text"):
        if scored[name]:
            share = counts[name] / scored[name]
            wrong = scored[name] - counts[name]
            print(f"  {name:9} {share:.2%}  ({wrong} wrong of {scored[name]})")
    agreed = [r["pattern_agrees"] for r in records if r["pattern_agrees"] is not None]
    if agreed:
        share = sum(agreed) / len(agreed)
        print(f"  pattern   {share:.2%} among the theme's own ({len(agreed)} named)")
    by_theme: dict[str, list[bool]] = defaultdict(list)
    for r in records:
        for theme in r["themes"]:
            by_theme[theme].append(r["correct"])
    print(
        "  by theme (correct, n): "
        + ", ".join(
            f"{t} {sum(v) / len(v):.0%}/{len(v)}"
            for t, v in sorted(by_theme.items(), key=lambda kv: -len(kv[1]))[:24]
        )
    )
    tagged = [r for r in records if r["sacrifice"]["tagged"]]
    if tagged:
        said = sum(r["sacrifice"]["said"] for r in tagged)
        extra = sum(1 for r in records if r["sacrifice"]["said"] and not r["sacrifice"]["tagged"])
        print(
            f"  sacrifice {said / len(tagged):.2%} of the {len(tagged)} tagged are said; "
            f"said in {extra} puzzles Lichess does not tag"
        )
    blind = [
        r for r in records if not r["engine_sees"] and set(r["themes"]) & {"crushing", "advantage"}
    ]
    if blind:
        print(f"  engine    {len(blind)} crushing/advantage puzzles the engine scores below 0.6")
    differs = [r for r in records if r.get("engine_differs")]
    if differs:
        wrong = sum(1 for r in differs if not all(r["checks"].values()))
        print(
            f"  engine    {len(differs)} puzzles where the engine sees otherwise (not counted; "
            f"{wrong} of them fail a check)"
        )
    better = [
        r
        for r in records
        if r["engine_sees"]
        and (r["engine_score"] or 0) < 0.8
        and set(r["themes"]) & {"crushing", "advantage"}
        and r["goal"] in ("plan", "defend")
    ]
    if better:
        print(
            f"  better    {len(better)} crushing/advantage puzzles the engine calls better, "
            "not won, read as a plan or a threat to meet"
        )
    per: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        for theme in r["themes"]:
            per[theme].append(r)
    table = []
    for theme, rs in per.items():
        said = [conveys(theme, r) for r in rs]
        if all(x is None for x in said):
            continue
        rate = sum(bool(x) for x in said) / len(rs)
        table.append((rate, theme, len(rs), sum(r["correct"] for r in rs) / len(rs)))
    if table:
        print("  themes (said, correct, n), least said first:")
        for rate, theme, n, ok in sorted(table)[:60]:
            print(f"    {theme:22} said {rate:6.1%}  correct {ok:6.1%}  n={n}")
    goals = Counter(r["goal"] for r in records)
    print(f"  goals: {dict(goals)}")
    if show:
        for name in ("move", "goal", "amount", "leak", "grounded", "claims", "text"):
            bad = [r for r in records if not r["checks"].get(name, True)][:show]
            for r in bad:
                print(f"\n[{name}] {r['id']} ({r['rating']}) {' '.join(r['themes'])}")
                print(f"  {r['fen']}  solution {r['solution']}")
                print(f"  note: {r['notes'].get(name)}")
                for rung in ("idea", "focus", "move"):
                    print(f"  {rung}: {r['texts'][rung]}")
        confused = Counter(
            (r["pattern"], " ".join(sorted(set(r["themes"]) & PATTERNS)))
            for r in records
            if r["pattern_agrees"] is False
        )
        if confused:
            print("\npattern disagreements (coach, puzzle's):")
            for (ours, theirs), k in confused.most_common(15):
                print(f"  {k:4}  {ours}  vs  {theirs}")
    return {
        "n": n,
        "correct": correct / n,
        **{name: counts[name] / scored[name] for name in scored},
        "pattern": (sum(agreed) / len(agreed)) if agreed else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=["sample", "full", "themes"], default="sample")
    parser.add_argument("--per-theme", type=int, default=120)
    parser.add_argument("--n", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--show", type=int, default=3, help="failures to print per check")
    parser.add_argument("--theme", default=None, help="only puzzles with this theme")
    args = parser.parse_args()

    started = time.monotonic()
    if args.source == "sample":
        rows = _rows_sample()
        random.Random(args.seed).shuffle(rows)
    elif args.source == "themes":
        rows = _rows_themes(args.per_theme, args.seed)
        args.n = max(args.n, len(rows))
    else:
        rows = _rows_full(args.n, args.seed)
    if args.theme:
        rows = [r for r in rows if args.theme in r["Themes"].split()]
    rows = rows[: args.n]
    fens = [position(r)[0].fen() for r in rows]
    took = time.monotonic() - started
    print(f"{len(rows)} puzzles ({args.source}, seed {args.seed}) in {took:.1f}s", flush=True)

    from analysis_cache import analyse_all

    analyses = analyse_all(fens, workers=args.workers)
    engine_done = time.monotonic()
    items = [(r, analyses[f].model_dump_json()) for r, f in zip(rows, fens, strict=True)]
    if len(items) >= 2000:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            records = pool.map(score, items, chunksize=64)
    else:
        records = [score(item) for item in items]
    took = time.monotonic() - engine_done
    each = 1000 * took / max(len(records), 1)
    print(f"coach: {len(records)} puzzles in {took:.1f}s ({each:.1f} ms each)")

    summary = summarise(records, args.show)
    commit = _commit()
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{commit}-{args.source}-{len(records)}"
    with (RESULTS / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    summary.update(commit=commit, source=args.source, seed=args.seed, file=f"{name}.jsonl")
    (RESULTS / f"{name}.summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\ncommit {commit}; records in data/puzzles/bench/{name}.jsonl")


if __name__ == "__main__":
    main()
