"""Games reviewed whole by the desktop (`analysis/game_review.py`), for the phone's Game Review.

For each game: every position measured by real engines side by side at a review strength
(`measure_all`), then the review walked from those measurements (`review_game` with all of them
known, so the walk searches nothing) -- with the motif tagger and the opening book. Recorded:
the PGN, the measurements (`measured_json`), the review (`ReviewedGame`), each move's clock and
time spent (`clocks`), the summary (`summarize`), the moves as the page draws them
(`move_views`), the tactics the mistakes passed up (`missed_motifs`, for each side and both),
and the key moves the AI coach writes on (`review_words.key_plies`).

The phone adopts the same measurements and walks the same moves (no engine), and is held to
the review, the summary and the views; the Stats and the AI coach's words read the reviews.

The games: three of Tai's chess.com games (a copy of his data: clocks with an increment), a
mate in four with clocks, a game from a set-up position with a promotion, a Lichess blitz game
with a repetition, and a game Sidecoach watched (no clocks, "You" for the player).

    cd D:/chess-coaching-companion
    uv run --no-sync python android/tools/make_review_fixtures.py
"""

from __future__ import annotations

import io
import json
import sqlite3
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402
import chess.pgn  # noqa: E402

from sidecoach.analysis import game_review, motifs  # noqa: E402
from sidecoach.analysis.engine import Engine  # noqa: E402
from sidecoach.analysis.openings import OpeningBook  # noqa: E402
from sidecoach.analysis.score import Thresholds  # noqa: E402
from sidecoach.coach import review_words  # noqa: E402
from sidecoach.config import find_stockfish, load_settings  # noqa: E402
from sidecoach.events import Side  # noqa: E402
from sidecoach.profile.game_summary import missed_motifs, summarize  # noqa: E402
from sidecoach.ui.review_room import move_views  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "profile" / "reviews.json"
COPY = Path(
    "C:/Users/Tai/AppData/Local/Temp/claude/d--chess-coaching-companion/"
    "5a849ff1-467c-42d3-861a-c579f032804a/scratchpad/appcopy/profile.sqlite"
)
ENGINES = 6

BY_HAND = {
    "a mate in four": (
        "fast",
        "black",
        '[Event "Live Chess"]\n[Site "Chess.com"]\n[White "Someone"]\n[Black "nosegazm"]\n'
        '[Result "1-0"]\n[WhiteElo "900"]\n[BlackElo "1000"]\n[TimeControl "180"]\n\n'
        "1. e4 {[%clk 0:03:00]} e5 {[%clk 0:02:59.5]} 2. Qh5 {[%clk 0:02:58]} Nc6 {[%clk 0:02:57]} "
        "3. Bc4 {[%clk 0:02:55.1]} Nf6 {[%clk 0:02:50]} 4. Qxf7# {[%clk 0:02:54]} 1-0\n",
    ),
    "a set-up position": (
        "fast",
        None,
        '[FEN "8/5P2/8/8/2k5/8/5K2/3r4 w - - 0 50"]\n[SetUp "1"]\n[Result "*"]\n\n'
        "50. f8=Q Rd2+ 51. Ke3 Rd8 52. Qf4+ Kb3 53. Qb8+ Ka2 54. Qxd8 *\n",
    ),
    "a lichess blitz game": (
        "standard",
        "white",
        '[Event "Rated Blitz game"]\n[Site "https://lichess.org/AbCdEfGh"]\n[White "nosegazm"]\n'
        '[Black "other"]\n[Result "1/2-1/2"]\n[WhiteElo "1500"]\n[BlackElo "1480"]\n'
        '[TimeControl "180+2"]\n\n'
        "1. d4 { [%clk 0:03:00] } 1... d5 { [%clk 0:03:00] } 2. c4 { [%clk 0:03:01] } "
        "2... e6 { [%clk 0:02:59] } 3. Nc3 { [%clk 0:02:58] } 3... Nf6 { [%clk 0:02:57] } "
        "4. Nf3 { [%clk 0:02:55] } 4... Ng8 { [%clk 0:02:50] } 5. Ng1 { [%clk 0:02:54] } "
        "5... Nf6 { [%clk 0:02:49] } 6. Nf3 { [%clk 0:02:52] } 6... Ng8 { [%clk 0:02:47] } "
        "7. Ng1 { [%clk 0:02:50] } 7... Nf6 { [%clk 0:02:45] } 1/2-1/2\n",
    ),
    "a game Sidecoach watched": (
        "fast",
        "black",
        '[White "Opponent"]\n[Black "You"]\n[Result "0-1"]\n[Date "2026.09.28"]\n'
        '[TimeControl "10 min"]\n\n'
        "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nd4 4. Nxe5 Qg5 5. Nxf7 Qxg2 6. Rf1 Qxe4+ 7. Be2 Nf3# 0-1\n",
    ),
}


def review(game: chess.pgn.Game, strength_name: str, engines: list[Engine], book: OpeningBook | None) -> dict:
    strength = game_review.STRENGTHS[strength_name]
    thresholds = Thresholds()
    boards = game_review.positions(game)
    measured = game_review.measure_all(
        boards, engines, depth=strength.depth, multipv=strength.multipv, thresholds=thresholds
    )
    known = {m.fen: m for m in measured}
    reviewed = game_review.review_game(
        game,
        engines,
        depth=strength.depth,
        multipv=strength.multipv,
        thresholds=thresholds,
        tagger=motifs,  # type: ignore[arg-type]
        book=book,
        strength=strength.name,
        game_id="review",
        known=known,
    )
    return {
        "measured": [json.loads(game_review.measured_json(m)) for m in measured],
        "reviewed": reviewed.model_dump(mode="json"),
        "clocks": [list(c) for c in game_review.clocks(game)],
        "summary": summarize(reviewed).model_dump(mode="json"),
        "views": move_views(reviewed),
        "missed": {
            "white": missed_motifs(reviewed, {Side.WHITE}),
            "black": missed_motifs(reviewed, {Side.BLACK}),
            "both": missed_motifs(reviewed, {Side.WHITE, Side.BLACK}),
        },
        "key_plies": review_words.key_plies(reviewed),
    }


def main() -> None:
    texts: list[tuple[str, str, str | None, str]] = []
    if COPY.exists():
        db = sqlite3.connect(COPY)
        for key, pgn, side in db.execute(
            "SELECT key, pgn, user_side FROM library WHERE source = 'chesscom' ORDER BY key LIMIT 3"
        ):
            texts.append((f"chess.com {key}", "fast", side, pgn))
        db.close()
    else:
        print("no copy of Tai's data: his chess.com games are left out", file=sys.stderr)
    for name, (strength, side, pgn) in BY_HAND.items():
        texts.append((name, strength, side, pgn))
    path = find_stockfish(load_settings())
    if path is None:
        raise SystemExit("no Stockfish")
    book = OpeningBook.load(DESKTOP / "data" / "openings")
    if book is None:
        raise SystemExit("no opening tables under data/openings")
    engines = [Engine(path=path, threads=1, hash_mb=64, multipv=6) for _ in range(ENGINES)]
    games = []
    try:
        for e in engines:
            e.open()
        for name, strength, side, pgn in texts:
            game = chess.pgn.read_game(io.StringIO(pgn))
            assert game is not None and not game.errors, name
            for e in engines:  # every game from a fresh hash, as a room's engines start
                e.close()
                e.open()
            done = review(game, strength, engines, book)
            games.append({"name": name, "strength": strength, "user_side": side, "pgn": pgn, **done})
            print(f"{name}: {len(done['reviewed']['moves'])} moves at {strength}", flush=True)
    finally:
        for e in engines:
            e.close()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"games": games}), encoding="utf-8")
    print(f"{OUT}: {len(games)} games, {OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
