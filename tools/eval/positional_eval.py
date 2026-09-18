"""Positional eval harness: grade the coach's strategic plans on real master positions.

The strategic analog of the tactic puzzle harness. The reference books (gitignored refs/) name
which famous games illustrate each concept and what plan a master teaches there; the books'
OCR'd moves are unusable, so the moves come from a clean external source -- pgnmentor.com's
per-player PGN collections, fetched by the readable game header. Every position is a real
master game position, machine-verified legal. For each we assert the coach's positional layer
teaches the expected concept as a leading idea (top few). Two tiers:

- MARQUEE cases are the model games the books annotate, where the master's actual plan is
  documented: the coach's plan is checked to match it (Kamsky-Karpov's IQP, Bertok-Fischer's
  hanging pawns, a Kasparov King's Indian's kingside pawn-storm).
- COVERAGE cases are real master positions (found by scanning famous players' games) where a
  structure genuinely occurs; they pin the detector and its ranking against regressions on real
  boards, not crafted FENs.

Run:  uv run python tools/eval/positional_eval.py
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import chess
import chess.pgn

from sidecoach.analysis import positional as P
from sidecoach.coach.facts import phase_of

CACHE = Path(__file__).with_name(".cache")
UA = "sidecoach-eval-harness (research; contact taiducnguyen.drexel@gmail.com)"
LEADING = 3  # the concept must be among the top-N ideas: a leading plan, not a buried one


def fetch_player(player: str) -> list[chess.pgn.Game]:
    """Every game in pgnmentor.com's collection for `player`, downloaded once and cached."""
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{player}.pgn"
    if not cached.exists():
        req = urllib.request.Request(
            f"https://www.pgnmentor.com/players/{player}.zip", headers={"User-Agent": UA}
        )
        with urllib.request.urlopen(req, timeout=90) as response:  # noqa: S310 - fixed host
            blob = response.read()
        name = next(n for n in zipfile.ZipFile(io.BytesIO(blob)).namelist() if n.endswith(".pgn"))
        text = zipfile.ZipFile(io.BytesIO(blob)).read(name).decode("utf-8", "replace")
        # The download is CRLF; normalise before writing, else Windows makes it \r\r\n and
        # python-chess then fails to parse the moves.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        cached.write_text(text, encoding="utf-8", newline="\n")
    games: list[chess.pgn.Game] = []
    stream = io.StringIO(cached.read_text(encoding="utf-8"))
    while (game := chess.pgn.read_game(stream)) is not None:
        games.append(game)
    return games


def find_game(
    player: str, *, white: str = "", black: str = "", year: str = "", event: str = ""
) -> chess.pgn.Game:
    """Locate one game in a player's collection by header fragments (case-insensitive)."""
    for game in fetch_player(player):
        h = game.headers
        if (
            white.lower() in h.get("White", "").lower()
            and black.lower() in h.get("Black", "").lower()
            and h.get("Date", "").startswith(year)
            and event.lower() in h.get("Event", "").lower()
        ):
            return game
    raise LookupError(f"no game: {player} {white}-{black} {year} {event}")


def board_after(game: chess.pgn.Game, ply: int) -> chess.Board:
    board = game.board()
    for i, move in enumerate(game.mainline_moves()):
        if i >= ply:
            break
        board.push(move)
    return board


def top_kinds(board: chess.Board, us: chess.Color) -> list[P.PositionalFact]:
    return P.top_features(board, us, phase=phase_of(board), limit=LEADING)


@dataclass
class Case:
    kind: str  # the concept the coach must teach as a leading idea here
    player: str  # whose pgnmentor collection holds the game
    locate: dict[str, str]  # header fragments to find the game
    ply: int
    us: chess.Color
    note: str = ""


W, B = chess.WHITE, chess.BLACK

# Marquee: the books' model games, master's plan documented.
MARQUEE: list[Case] = [
    Case(
        "iqpTarget",
        "Karpov",
        {"white": "Kamsky", "black": "Karpov", "year": "1996"},
        18,
        B,
        "Kamsky-Karpov, Elista 1996 (Nunn Game 25): Karpov blockades the IQP",
    ),
    Case(
        "iqp",
        "Karpov",
        {"white": "Kamsky", "black": "Karpov", "year": "1996"},
        18,
        W,
        "the same board from White's (the IQP owner's) side",
    ),
    Case(
        "hangingPawns",
        "Fischer",
        {"white": "Bertok", "black": "Fischer", "year": "1962"},
        26,
        B,
        "Bertok-Fischer, Stockholm 1962 (Silman's hanging-pawns model)",
    ),
    Case(
        "pawnPointing",
        "Kasparov",
        {"white": "Garcia Gonzales", "black": "Kasparov", "year": "1979"},
        12,
        B,
        "a Kasparov King's Indian: the closed centre points Black to the kingside (...f5)",
    ),
]

# Coverage: real master positions found by scanning famous players' games.
COVERAGE: list[Case] = [
    Case("iqp", "Karpov", {"white": "Karpov", "black": "Ponomariov", "year": "1961"}, 41, B, ""),
    Case(
        "iqpTarget", "Karpov", {"white": "Karpov", "black": "Ponomariov", "year": "1961"}, 41, W, ""
    ),
    Case(
        "hangingPawns",
        "Karpov",
        {"white": "Karpov", "black": "Kalashnikov", "year": "1961"},
        35,
        B,
        "",
    ),
    Case(
        "hangingPawnsTarget",
        "Fischer",
        {"white": "Fischer", "black": "Matthai", "year": "1956"},
        41,
        B,
        "",
    ),
    Case(
        "pawnMajority",
        "Kasparov",
        {"white": "Kasparov", "black": "Galle", "year": "1976"},
        23,
        W,
        "",
    ),
    Case(
        "pawnMajority",
        "Karpov",
        {"white": "Karpov", "black": "Ponomariov", "year": "1961"},
        41,
        W,
        "",
    ),
    Case(
        "pawnMajority",
        "Fischer",
        {"white": "Thomason", "black": "Fischer", "year": "1955"},
        31,
        W,
        "",
    ),
    Case(
        "pawnPointing", "Karpov", {"white": "Karpov", "black": "Budakov", "year": "1961"}, 21, B, ""
    ),
    Case(
        "spaceAdvantage",
        "Karpov",
        {"white": "Karpov", "black": "Ponomariov", "year": "1961"},
        33,
        B,
        "",
    ),
    Case(
        "spaceAdvantage",
        "Fischer",
        {"white": "Thomason", "black": "Fischer", "year": "1955"},
        17,
        W,
        "",
    ),
    Case(
        "spaceCramp",
        "Fischer",
        {"white": "Thomason", "black": "Fischer", "year": "1955"},
        17,
        B,
        "",
    ),
    Case(
        "bishopPair",
        "Karpov",
        {"white": "Karpov", "black": "Ponomariov", "year": "1961"},
        31,
        W,
        "",
    ),
    Case(
        "bishopPair",
        "Fischer",
        {"white": "Thomason", "black": "Fischer", "year": "1955"},
        19,
        W,
        "",
    ),
    Case(
        "badBishop", "Karpov", {"white": "Karpov", "black": "Ponomariov", "year": "1961"}, 25, B, ""
    ),
    Case(
        "badBishop", "Fischer", {"white": "Thomason", "black": "Fischer", "year": "1955"}, 17, W, ""
    ),
    Case("passedPawn", "Fischer", {"white": "Fischer", "black": "Ames", "year": "1955"}, 19, W, ""),
    Case(
        "passedPawn", "Karpov", {"white": "Karpov", "black": "Budakov", "year": "1961"}, 45, W, ""
    ),
    Case("outpost", "Fischer", {"white": "Fischer", "black": "Casado", "year": "1956"}, 33, W, ""),
    Case(
        "outpost", "Karpov", {"white": "Karpov", "black": "Ponomariov", "year": "1961"}, 39, W, ""
    ),
    Case(
        "targetPawn", "Fischer", {"white": "Fischer", "black": "Warner", "year": "1955"}, 41, W, ""
    ),
    Case(
        "targetPawn",
        "Karpov",
        {"white": "Karpov", "black": "Ponomariov", "year": "1961"},
        33,
        W,
        "",
    ),
]


def run(cases: list[Case], title: str) -> tuple[int, int]:
    print(f"\n=== {title} ===")
    passed = graded = 0
    for case in cases:
        try:
            board = board_after(find_game(case.player, **case.locate), case.ply)
        except LookupError as exc:
            print(f"[skip] {case.kind}: {exc}")
            continue
        facts = top_kinds(board, case.us)
        hit = next((f for f in facts if f.kind == case.kind), None)
        graded += 1
        passed += hit is not None
        side = "W" if case.us == W else "b"
        who = " - ".join(case.locate.get(k, "?") for k in ("white", "black"))
        mark = "PASS" if hit else "FAIL"
        print(
            f"[{mark}] {case.kind:20s} {who} ({side})" + (f"  -- {case.note}" if case.note else "")
        )
        if hit:
            print(f"         -> {hit.text[:96]}")
        else:
            print(f"         -> leading ideas were: {[f.kind for f in facts]}")
    return passed, graded


def main() -> None:
    mp, mg = run(MARQUEE, "MARQUEE (book model games, master's plan documented)")
    cp, cg = run(COVERAGE, "COVERAGE (real master positions, detector + ranking regression)")
    print(f"\nMARQUEE {mp}/{mg}   COVERAGE {cp}/{cg}   TOTAL {mp + cp}/{mg + cg}")


if __name__ == "__main__":
    main()
