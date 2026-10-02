"""Record what the desktop's Stats make of reviewed games, for the Kotlin port (profile/StatFacts.kt,
profile/Stats.kt, the Store's Stats section, ui/StatsApi.kt); and ship the models Stats reads
beside its code -- `categories.json` and `peers.json`, fitted by the desktop's
`tools/review/fit_stats.py` -- as the phone's resources (app/src/main/resources/com/sidecoach/
profile/), so a refit reaches the phone together with the fixture that holds it to the desktop.

- `ops`: a store built step by step -- library rows; the seven real reviews of `reviews.json`
  kept (`keep_stats`) under their own rows and again under others (other keys, sources, time
  classes, dates, ratings, sides, and headers that make a bot game, a termination, a result); a
  game's facts kept again from another review; random facts saved whole (`save_stats`: every
  kind, label, phase, structure, theme, endgame, clock, result, termination, opening); watched
  games an import stands for; reviews with and without facts; the player's puzzle level.
- `rows`: what the store then holds -- `stat_games()`, `stat_moves()` of every key,
  `reviews_without_stats()`.
- `stats`: `/api/stats`'s body (the route's `_filter` and `_drills`) under many filters -- time
  class, the last so many days, opponents -- with "now" fixed (`now`). The puzzles each area to
  work on gets are a random draw, so each area's whole pool (`pools`: id -> rating) is recorded.
- `positions`: `/api/stats/positions`'s body for every selector and its kinds, under some filters.
- `units`: `kind_of`; `chance_in` (every position of the reviews, and their best lines given other
  motifs, mates and scores); `structure_of`, `themes_of` and `endgame_of` (every position of the
  reviews, random games and random placements); `standing`; `bot_game`; `family`;
  `termination_kind`; `grade`; `performance.lichess`; `fmean`.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_stats_fixtures.py
"""

from __future__ import annotations

import dataclasses
import json
import random
import shutil
import sys
import tempfile
import types
from pathlib import Path
from statistics import fmean
from typing import Any

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.events import Analysis, MotifTag, ReviewedGame  # noqa: E402
from sidecoach.profile import library, performance, stat_facts, stats  # noqa: E402
from sidecoach.profile.puzzles import open_store, puzzle_level  # noqa: E402
from sidecoach.profile.store import LibraryRow, Store  # noqa: E402
from sidecoach.ui import server  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "app" / "src" / "androidTest" / "assets" / "profile"
REVIEWS = ASSETS / "reviews.json"
PUZZLES = ASSETS / "puzzles.csv"
OUT = ASSETS / "stats.json"
MODELS = ROOT / "app" / "src" / "main" / "resources" / "com" / "sidecoach" / "profile"
NOW = 1_790_000_000.0  # "now" for every filter by days; the games' dates count back from it
DAY = 86400.0

KINDS = list(stat_facts.KINDS)
LABELS = ["best"] * 6 + ["excellent"] * 4 + ["good"] * 3 + ["inaccuracy"] * 2 + ["mistake"] * 2
LABELS += ["miss", "blunder", "blunder", "great", "brilliant", None]
LOSS = {
    "best": (0.0, 0.0),
    "great": (0.0, 0.0),
    "brilliant": (0.0, 0.0),
    "book": (0.0, 0.0),
    "excellent": (0.0, 0.02),
    "good": (0.02, 0.05),
    "inaccuracy": (0.05, 0.1),
    "mistake": (0.1, 0.2),
    "miss": (0.1, 0.3),
    "blunder": (0.2, 0.6),
    None: (0.0, 0.3),
}
OPENINGS = [
    "C50 Italian Game: Giuoco Piano",
    "C50 Italian Game",
    "B01 Scandinavian Defense: Mieses-Kotroc Variation",
    "Sicilian Defense: Najdorf Variation",
    "D00 Queen's Pawn Game: Mason Attack",
    "E60 King's Indian Defense",
    "A00 ",
    "",
    None,
]
TERMINATIONS = [
    "nosegazm won by checkmate",
    "opp won on time",
    "nosegazm won on time",
    "Time forfeit",
    "Normal",
    None,
    "Game abandoned",
    "Game drawn by agreement",
    "Game drawn by repetition",
    "Game drawn by stalemate",
    "Game drawn by insufficient material",
    "Game drawn by 50-move rule",
    "nosegazm won by resignation",
    "opp won by resignation",
    "Rules infraction",
]
RESULTS = ["1-0", "0-1", "1/2-1/2", "1-0", "0-1", "*", None]
# Headers a kept review is given in place of its own: a bot or coach game, a title, a
# termination, a result to read or to fall back from (None takes the header out).
HEADER_CHANGES: list[dict[str, str | None]] = [
    {},
    {},
    {"Event": "Play vs Bot"},
    {"Event": "Casual game vs Coach"},
    {"Event": "Rated vs Computer game"},
    {"WhiteTitle": "BOT"},
    {"BlackTitle": "bot"},
    {"White": "stockfish-BOT"},
    {"Black": "Coach-Danny"},
    {"White": "coach-danny"},
    {"Termination": "nosegazm won on time"},
    {"Termination": "Time forfeit"},
    {"Termination": "Game drawn by agreement"},
    {"Termination": "Game abandoned"},
    {"Result": ""},
    {"Result": "*"},
    {"Result": None},
    {"Termination": None, "Result": "1/2-1/2"},
]


def row_json(row: LibraryRow) -> dict[str, Any]:
    return dataclasses.asdict(row)


# ---- the store -----------------------------------------------------------------------------------


class Builder:
    """The store and the ops that built it, for the phone to replay in order."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.ops: list[dict[str, Any]] = []
        self.keys: set[str] = set()
        self.dates: set[float] = set()
        # The strength each game's facts were kept at: its review's, for the game shapes.
        self.strengths: dict[str, str] = {}

    def when(self, rng: random.Random, low: float, high: float) -> float:
        """A date `low` to `high` days before now, none the same as another's (the store orders
        the games by it, and a tie is SQLite's to break)."""
        while True:
            t = round(NOW - rng.uniform(low, high) * DAY, 3)
            if t not in self.dates:
                self.dates.add(t)
                return t

    def library(self, row: LibraryRow) -> None:
        self.store.add_library_game(row)
        self.ops.append({"op": "library", "row": row_json(row)})

    def keep(self, strength: str, index: int, reviewed: ReviewedGame, headers: dict[str, str] | None, row: LibraryRow) -> None:
        game = reviewed if headers is None else reviewed.model_copy(update={"headers": headers})
        stat_facts.keep_stats(self.store, strength, game, row)
        self.strengths[row.key] = strength
        self.keys.add(row.key)
        self.ops.append({"op": "keep", "strength": strength, "review": index, "headers": headers, "row": row_json(row)})

    def save(self, strength: str, game: dict[str, Any], moves: list[dict[str, Any]]) -> None:
        self.store.save_stats(strength, game, moves)
        self.strengths[game["key"]] = strength
        self.keys.add(game["key"])
        self.ops.append({"op": "save", "strength": strength, "game": game, "moves": moves})

    def twin(self, watched: str, imported: str) -> None:
        self.store.add_twin(watched, imported)
        self.ops.append({"op": "twin", "watched": watched, "imported": imported})

    def review(self, key: str, strength: str, reviewed_at: float, summary: str | None = None) -> None:
        self.store.save_review(key, strength, 3, reviewed_at, "{}", summary)
        self.ops.append({"op": "review", "key": key, "strength": strength, "version": 3, "reviewed_at": reviewed_at, "report": "{}", "summary": summary})

    def setting(self, key: str, value: str) -> None:
        self.store.set_setting(key, value)
        self.ops.append({"op": "setting", "key": key, "value": value})


def own_row(index: int, g: dict[str, Any], b: Builder, rng: random.Random) -> LibraryRow:
    """A real review's own row: its source and key by its name, its date counted back from now."""
    game = library.games_in(g["pgn"])[0]
    name = g["name"]
    if name.startswith("chess.com "):
        source, key = "chesscom", name.split(" ", 1)[1]
    elif "lichess" in name:
        source, key = "lichess", f"lichess:review{index}"
    elif "watched" in name:
        source, key = "watched", f"watched:review{index}"
    else:
        source, key = "pgn", f"pgn:review{index}"
    row = library.row_for(game, source=source, key=key, now=NOW - DAY)
    return dataclasses.replace(row, user_side=g["user_side"], played_at=b.when(rng, 0.2, 20))


def variant_row(index: int, j: int, own: LibraryRow, b: Builder, rng: random.Random) -> LibraryRow:
    """The same review under another row: another key, source, time class, date, side, ratings."""
    source = rng.choice(["chesscom", "chesscom", "lichess", "pgn", "watched"])
    elo = [None, 0, *range(400, 2600, 150)]
    return dataclasses.replace(
        own,
        key=f"{source}:v{index}-{j}",
        source=source,
        time_class=rng.choice(["blitz", "rapid", "rapid", "daily", None]),
        played_at=b.when(rng, 0.5, 500),
        user_side=rng.choice(["white", "black", "white", "black", None]),
        white_elo=rng.choice(elo),
        black_elo=rng.choice(elo),
        result=rng.choice(RESULTS),
    )


def changed(headers: dict[str, str], change: dict[str, str | None]) -> dict[str, str]:
    out = dict(headers)
    for k, v in change.items():
        if v is None:
            out.pop(k, None)
        else:
            out[k] = v
    return out


def random_line(rng: random.Random, plies: int, board: chess.Board | None = None) -> list[tuple[int, str, str, str, str]]:
    """(ply, mover, FEN before, UCI, SAN) of a random game."""
    board = board or chess.Board()
    out = []
    for ply in range(1, plies + 1):
        moves = list(board.legal_moves)
        if not moves:
            break
        move = rng.choice(moves)
        out.append((ply, "white" if board.turn else "black", board.fen(), move.uci(), board.san(move)))
        board.push(move)
    return out


def synthetic(rng: random.Random, key: str, source: str, time_class: str | None, played_at: float | None, side: str | None, bot: bool, plies: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """A game's facts by chance: every kind, label, phase, structure, theme, endgame and clock."""
    line = random_line(rng, plies)
    book_until = rng.randrange(0, 14)
    endgame_from = rng.choice([None, None, max(1, int(len(line) * 0.6)), max(1, int(len(line) * 0.3))])
    base = {"bullet": 60_000, "blitz": 180_000, "rapid": 600_000}.get(time_class or "")
    clocks = {"white": base, "black": base}
    moves = []
    for ply, mover, fen, uci, san in line:
        phase = "endgame" if endgame_from is not None and ply >= endgame_from else "opening" if ply <= 16 else "middlegame"
        label = "book" if ply <= book_until and phase == "opening" else rng.choice(LABELS)
        low, high = LOSS[label]
        chance = rng.choice(KINDS) if rng.random() < 0.15 else None
        spent = None if rng.random() < 0.1 else rng.randrange(0, 40_000)
        clock = None
        if clocks[mover] is not None and rng.random() < 0.9:
            clocks[mover] = max(0, clocks[mover] - (spent or rng.randrange(0, 20_000)))
            clock = clocks[mover]
        elif time_class == "daily" and rng.random() < 0.5:
            clock = rng.randrange(0, 86_400_000)
        moves.append(
            {
                "ply": ply,
                "side": mover,
                "fen": fen,
                "uci": uci,
                "san": san,
                "phase": phase,
                "loss": round(rng.uniform(low, high), 4),
                "label": label,
                "accuracy": None if rng.random() < 0.05 else rng.uniform(0, 100),
                "score": rng.choice([0.4, 0.6, round(rng.random(), 4), round(rng.random(), 4), round(rng.random(), 4)]),
                "chance": chance,
                "found": chance is not None and rng.random() < 0.5,
                "allowed": rng.choice(KINDS) if rng.random() < 0.08 else None,
                "structure": rng.choice(stat_facts.STRUCTURES),
                "themes": ",".join(t for t in stat_facts.THEMES if rng.random() < 0.3),
                "endgame": rng.choice([*stat_facts.ENDGAMES, None]) if phase == "endgame" else None,
                "spent_ms": spent,
                "clock_ms": clock,
            }
        )
    game = {
        "key": key,
        "source": source,
        "time_class": time_class,
        "played_at": played_at,
        "user_side": side,
        "bot": bot,
        "result": rng.choice(RESULTS),
        "opening": rng.choice(OPENINGS),
        "termination": rng.choice(TERMINATIONS),
    }
    return game, moves


def build(b: Builder, reviews: list[dict[str, Any]], rng: random.Random) -> None:
    reviewed = [ReviewedGame.model_validate(g["reviewed"]) for g in reviews]
    own = []
    for i, g in enumerate(reviews):
        row = own_row(i, g, b, rng)
        own.append(row)
        b.library(row)
        b.keep(g["strength"], i, reviewed[i], None, row)
    # Each review again under other rows and headers.
    for i, g in enumerate(reviews):
        for j in range(5):
            row = variant_row(i, j, own[i], b, rng)
            if rng.random() < 0.8:
                b.library(row)
            change = rng.choice(HEADER_CHANGES)
            headers = changed(reviewed[i].headers, change) if change else None
            b.keep(g["strength"], i, reviewed[i], headers, row)
    # A game's facts from another review take the place of the first's.
    b.keep("standard", 0, reviewed[0], None, own[0])
    # Random facts: bullet all on Lichess (the page's scale is Lichess's for those), the rest mixed.
    plan = (
        [("bullet", "lichess")] * 10
        + [("blitz", s) for s in ("chesscom", "lichess", "pgn", "watched", "chesscom")] * 3
        + [("rapid", s) for s in ("chesscom", "chesscom", "lichess", "pgn", "watched")] * 4
        + [(None, "chesscom"), (None, "pgn"), (None, "watched"), (None, "lichess"), (None, "chesscom")]
    )
    for n, (time_class, source) in enumerate(plan):
        key = f"{source}:s{n}"
        side = rng.choice(["white", "black"]) if n % 17 else rng.choice([None, "?"])
        played_at = None if n == 33 else b.when(rng, 0.01, 500)
        game, moves = synthetic(rng, key, source, time_class, played_at, side, rng.random() < 0.3, rng.choice([5, 12, 30, 50, 70, 90, 120]))
        if rng.random() < 0.7:
            elo = [None, 0, *range(500, 2700, 100)]
            b.library(
                LibraryRow(
                    key=key,
                    source=source,
                    added_at=NOW - DAY,
                    pgn='[White "?"]\n\n*\n',
                    played_at=played_at,
                    white="nosegazm" if side == "white" else "opp",
                    black="nosegazm" if side == "black" else "opp",
                    white_elo=rng.choice(elo),
                    black_elo=rng.choice(elo),
                    result=game["result"],
                    time_class=time_class,
                    user_side=side,
                )
            )
        b.save(rng.choice(["fast", "standard"]), game, moves)
    # One daily game, short and against a bot: too few moves for a rating, no peers.
    game, moves = synthetic(rng, "chesscom:daily", "chesscom", "daily", b.when(rng, 1, 20), "white", True, 9)
    b.save("fast", game, moves)
    # Watched games their imports stand for: out of Stats, and of the reviews still to keep.
    watched = sorted(k for k in b.keys if k.startswith("watched:s"))
    b.twin("watched:review6", "chesscom:v6-0")
    b.twin(watched[0], "chesscom:s10")
    b.twin("watched:nostats", "pgn:s12")
    # Reviews: one kept already, one whose facts are not (its latest only), one a twin stands for.
    b.review("chesscom:s10", "fast", NOW - 50)
    b.review("pgn:norev1", "fast", NOW - 300)
    b.review("pgn:norev1", "standard", NOW - 200)
    b.review("pgn:norev2", "fast", NOW - 100)
    b.review("watched:nostats", "fast", NOW - 100)
    b.review(own[0].key, "fast", NOW - 10)
    # The game shapes (Stats' `shapes`): the real reviews' summaries at the strength their facts
    # were kept at, and the synthetic games' with a shape drawn, none or a summary not read.
    for i, g in enumerate(reviews):
        b.review(own[i].key, b.strengths[own[i].key], NOW - 5 - i, json.dumps(g["summary"]))
    sides = {"white": {"side": "white"}, "black": {"side": "black"}}
    for n, key in enumerate(sorted(k for k in b.keys if ":s" in k)):
        shape = rng.choice([*stats.SHAPES, None])
        summary = "not json" if n == 3 else json.dumps({**sides, "shape": shape})
        b.review(key, b.strengths[key], NOW - 400 - n, summary)
    b.setting("puzzle_level", "1300")
    # The library's classes are as the rule left them: the routes' `refresh` then leaves the rows
    # (their random classes) as they are, on both sides, whichever test reads the store first.
    b.setting(library.CLASSES_KEY, str(library.CLASSES))


# ---- the routes -----------------------------------------------------------------------------------


def _filter(time_class: str, days: int, opponents: str) -> stats.Filter:
    """The route's `_filter` (nested in `create_app`)."""
    return stats.Filter(time_class=time_class or None, days=days or None, opponents=opponents)


def args(query: dict[str, str]) -> tuple[str, int, str]:
    return query.get("time_class", ""), int(query.get("days", "0")), query.get("opponents", "all")


def stats_case(path: Path, data_dir: Path, query: dict[str, str]) -> dict[str, Any]:
    """`/api/stats` as `stats_now` answers it, and each area's whole pool of puzzles."""
    store = Store(path)
    try:
        numbers = stats.stats(store, _filter(*args(query)))
        server._drills(numbers["work"], data_dir, store)
        body = {"ok": True, "stats": numbers, "backlog": None}
        puzzles = open_store(data_dir)
        assert puzzles is not None
        level = puzzle_level(store)
        pools = [
            {p.id: p.rating for p in puzzles.pick(w["motif"], level, count=10**6)} if w["motif"] else {}
            for w in numbers["work"]
        ]
        puzzles.close()
    finally:
        store.close()
    return {"query": query, "body": json.loads(json.dumps(body)), "pools": pools}


def positions_case(path: Path, query: dict[str, str]) -> dict[str, Any]:
    store = Store(path)
    try:
        found = stats.positions(store, _filter(*args(query)), query.get("what", ""), query.get("arg", ""))
    finally:
        store.close()
    return {"query": query, "body": {"ok": True, "positions": found}}


def stats_queries() -> list[dict[str, str]]:
    out: list[dict[str, str]] = [{}]
    for opponents in ("all", "people", "bots"):
        for time_class in ("", "bullet", "blitz", "rapid", "daily", "classical"):
            out.append({"time_class": time_class, "days": "0", "opponents": opponents})
    for days in ("1", "7", "30", "90", "365"):
        out.append({"time_class": "", "days": days, "opponents": "all"})
    out += [
        {"time_class": "rapid", "days": "30", "opponents": "people"},
        {"time_class": "blitz", "days": "365", "opponents": "bots"},
        {"time_class": "bullet", "days": "90", "opponents": "all"},
        {"days": "7"},
        {"opponents": "people"},
    ]
    return out


def selectors() -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = [("opening.mistakes", ""), ("time.pressure", ""), ("nothing", "")]
    for what in ("tactics.found", "tactics.missed", "tactics.allowed"):
        out += [(what, k) for k in KINDS]
    out += [("strategy.mistakes", s) for s in ("", "all", *stat_facts.STRUCTURES)]
    out += [("endgame.mistakes", k) for k in ("", "overall", *stat_facts.ENDGAMES)]
    out += [("gallery", k) for k in ("brilliant", "great", "miss", "blunder", "best", "book")]
    costs = [f"missed.{k}" for k in KINDS] + [f"allowed.{k}" for k in KINDS]
    out += [("area", a) for a in (*costs, "opening", *(f"endgame.{k}" for k in stat_facts.ENDGAMES), "quiet")]
    return out


def positions_queries() -> list[dict[str, str]]:
    out = [{"what": what, "arg": arg} for what, arg in selectors()]
    some = [("gallery", "blunder"), ("area", "quiet"), ("time.pressure", ""), ("tactics.missed", "fork"), ("strategy.mistakes", "all"), ("endgame.mistakes", "")]
    for query in ({"time_class": "rapid", "days": "0", "opponents": "people"}, {"time_class": "bullet"}, {"days": "30", "opponents": "bots"}, {"time_class": "classical"}):
        out += [{"what": what, "arg": arg, **query} for what, arg in some]
    return out


# ---- the parts -------------------------------------------------------------------------------------


def tactic_names() -> list[str]:
    return sorted(stat_facts.TACTICS)


def kind_cases(rng: random.Random) -> list[list[Any]]:
    names = [*tactic_names(), "capture", "check", "quietMove", "promotion", "castling"]
    cases = []
    for _ in range(400):
        tags = rng.sample(names, rng.randrange(0, 5))
        cases.append([tags, stat_facts.kind_of([MotifTag(name=n, squares=["e4"]) for n in tags])])
    return cases


def trimmed(a: Analysis) -> Analysis:
    """An analysis with what `chance_in` reads: its best line (the rest only make the file long)."""
    return a.model_copy(update={"lines": a.lines[:1], "traps": [], "defences": [], "threatens": [], "threats": []})


def chance_cases(rng: random.Random, reviewed: list[ReviewedGame]) -> list[list[Any]]:
    analyses = [a for game in reviewed for a in (game.start, *(m.after for m in game.moves))]
    names = tactic_names()
    cases = []
    for a in analyses:
        t = trimmed(a)
        cases.append([t.model_dump(mode="json"), stat_facts.chance_in(t)])
    for _ in range(300):
        a = trimmed(rng.choice(analyses))
        if not a.lines or not a.lines[0].moves:
            continue
        best = a.lines[0]
        line = best.model_copy(
            update={
                "mate": rng.choice([None, None, 1, 2, 3, 4, 7, -2, 0]),
                "expected_score": rng.choice([0.2, 0.49, 0.5, 0.7, 0.95, best.expected_score]),
            }
        )
        moves = [*best.moves[:4], "a1a2", None]
        tags = [MotifTag(name=rng.choice(names), squares=["e4"], move=rng.choice(moves)) for _ in range(rng.randrange(0, 4))]
        if rng.random() < 0.3:
            tags.append(MotifTag(name=rng.choice(["capture", "check", "quietMove"]), squares=["d4"], move=rng.choice(moves)))
        v = a.model_copy(update={"decisive": rng.random() < 0.85, "lines": [line], "tags": tags})
        cases.append([v.model_dump(mode="json"), stat_facts.chance_in(v)])
    return cases


def random_placement(rng: random.Random) -> chess.Board | None:
    """A legal position of few pieces: the endgames random play seldom reaches."""
    board = chess.Board(None)
    squares = list(chess.SQUARES)
    rng.shuffle(squares)
    board.set_piece_at(squares.pop(), chess.Piece(chess.KING, chess.WHITE))
    board.set_piece_at(squares.pop(), chess.Piece(chess.KING, chess.BLACK))
    for _ in range(rng.randrange(0, 9)):
        kind = rng.choice([chess.PAWN, chess.PAWN, chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.ROOK, chess.QUEEN])
        square = squares.pop()
        if kind == chess.PAWN and chess.square_rank(square) in (0, 7):
            continue
        board.set_piece_at(square, chess.Piece(kind, rng.random() < 0.5))
    board.turn = rng.random() < 0.5
    return board if board.is_valid() else None


BY_HAND = [
    "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",  # the French advance: closed
    "r1bq1rk1/pp2bppp/2n1pn2/8/2BP4/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 9",  # an isolated queen's pawn
    "r2q1rk1/pb2bppp/1pn1pn2/8/2PP4/P1NB1N2/1P3PPP/R1BQ1RK1 b - - 0 11",  # hanging pawns
    "8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 40",  # a pawn endgame
    "8/5pk1/6p1/8/8/6P1/5PK1/3R1r2 w - - 0 40",  # a rook endgame
    "8/5pk1/6p1/2q5/8/4Q1P1/5PK1/8 w - - 0 40",  # a queen endgame
    "8/5pk1/4b1p1/8/8/3N2P1/5PK1/8 w - - 0 40",  # minor pieces
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "2r3k1/5ppp/8/3N4/8/8/5PPP/2R3K1 w - - 0 30",  # a knight on an outpost, a rook on the open file
]


def board_cases(rng: random.Random, reviewed: list[ReviewedGame]) -> list[list[Any]]:
    fens = [m.fen_before for game in reviewed for m in game.moves] + list(BY_HAND)
    for _ in range(40):
        line = random_line(rng, rng.randrange(1, 120))
        fens += [f for _, _, f, _, _ in rng.sample(line, min(len(line), 4))]
    while len(fens) < 600:
        board = random_placement(rng)
        if board is not None:
            fens.append(board.fen())
    cases = []
    for fen in fens:
        board = chess.Board(fen)
        cases.append([fen, stat_facts.structure_of(board), stat_facts.themes_of(board), stat_facts.endgame_of(board)])
    return cases


BOT_HEADERS: list[dict[str, str]] = [
    {},
    {"Event": "Live Chess", "White": "nosegazm", "Black": "Opp"},
    {"Event": "Play vs Bot"},
    {"Event": "Casual vs coach game"},
    {"Event": "VS COMPUTER"},
    {"Event": "vsbot"},
    {"WhiteTitle": "BOT"},
    {"BlackTitle": "Bot"},
    {"BlackTitle": "GM"},
    {"White": "maia1-bot"},
    {"Black": "Stockfish-BOT"},
    {"Black": "botez"},
    {"White": "Coach-Danny"},
    {"Black": "Coach-Mae"},
    {"White": "coach-danny"},
    {"White": "Coach"},
    {"White": "straße-bot"},
]


def unit_cases(rng: random.Random, reviewed: list[ReviewedGame]) -> dict[str, Any]:
    lines = stats._peer_lines()
    grades = []
    paths = sorted(lines)
    for _ in range(300):
        path = rng.choice(paths)
        line = dict(lines[path])
        if rng.random() < 0.1:
            line["b"] = 0.0
        n = rng.randrange(0, 60)
        level = rng.uniform(500, 2900)
        if rng.random() < 0.7:  # near the peers' figure, so every grade comes up
            peer = line["a"] + line["b"] * min(max(level, line["lo"]), line["hi"])
            value = peer + line["b"] * rng.uniform(-700, 700) * (n + stats.SHRINK) / max(n, 1)
        else:
            value = rng.uniform(-50, 150) if rng.random() < 0.8 else rng.uniform(0, 1)
        peer, kind = stats.grade(value, n, line, level)
        grades.append([value, n, {k: line[k] for k in ("a", "b", "r2", "lo", "hi")}, level, peer, kind])
    means = []
    for values in ([1e-16, 1.0, 1e16], [0.1] * 10, [1e100, 1.0, -1e100, 1e-100], [64.7, 97.4, 55.4], [2.0**53, 1.0, 1.0], [-0.0], [5, 7, 8]):
        means.append([[float(v) for v in values], fmean(values)])
    for _ in range(200):
        values = [rng.uniform(0, 100) for _ in range(rng.randrange(1, 60))]
        if rng.random() < 0.3:
            values = [rng.choice([1e-3, 1.0, 1e8, -1e8, 0.1, 0.3]) * rng.random() for _ in values]
        means.append([values, fmean(values)])
    return {
        "kind_of": kind_cases(rng),
        "chance_in": chance_cases(rng, reviewed),
        "boards": board_cases(rng, reviewed),
        "standing": [[s, stat_facts.standing(s)] for s in (0.0, 0.2, 0.4, 0.4000001, 0.5, 0.5999999, 0.6, 0.9, 1.0)],
        "bot_game": [[h, stat_facts.bot_game(h)] for h in BOT_HEADERS],
        "family": [[o, stats.family(o)] for o in (*OPENINGS, "E99 King's Indian Defense: Orthodox Variation, Classical System", "F00 Not an ECO: thing", "C5 short", "  Italian Game : x", "D00\tQueen's Pawn Game: Mason Attack", "A00Polish")],
        "termination_kind": [[t, stats.termination_kind(t)] for t in (*TERMINATIONS, "", "TIMEOUT", "won by Resignation", "Checkmated", "Stalemate", "Insufficient Material")],
        "grade": grades,
        "lichess": [[r, performance.lichess(r)] for r in (100, 500, 815, 900, 1000, 1200, 1500, 1800, 2000, 2430, 2500, 3000)],
        "fmean": means,
    }


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    for name in ("categories.json", "peers.json"):
        shutil.copyfile(DESKTOP / "src" / "sidecoach" / "profile" / name, MODELS / name)
    # "now", fixed for the filters by days.
    stats.time = types.SimpleNamespace(time=lambda: NOW)  # type: ignore[assignment]
    rng = random.Random(20260930)
    reviews = json.loads(REVIEWS.read_text(encoding="utf-8"))["games"]
    reviewed = [ReviewedGame.model_validate(g["reviewed"]) for g in reviews]
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        (data_dir / "puzzles").mkdir()
        shutil.copyfile(PUZZLES, data_dir / "puzzles" / "puzzles.csv")
        path = data_dir / "profile.sqlite"
        store = Store(path)
        b = Builder(store)
        build(b, reviews, rng)
        keys = sorted(b.keys | {"watched:review6", "watched:nostats"})
        rows = {
            "stat_games": store.stat_games(),
            "stat_moves": store.stat_moves(keys),
            "keys": keys,
            "reviews_without_stats": [dataclasses.asdict(r) for r in store.reviews_without_stats()],
        }
        store.close()
        stats_cases = [stats_case(path, data_dir, q) for q in stats_queries()]
        positions_cases = [positions_case(path, q) for q in positions_queries()]
    fixture = {
        "now": NOW,
        "ops": b.ops,
        "rows": rows,
        "stats": stats_cases,
        "positions": positions_cases,
        "units": unit_cases(rng, reviewed),
    }
    OUT.write_text(json.dumps(fixture), encoding="utf-8")
    games = len(rows["stat_games"])
    moves = len(rows["stat_moves"])
    found = sum(len(c["body"]["positions"]) for c in positions_cases)
    work = sum(len(c["body"]["stats"]["work"]) for c in stats_cases)
    print(
        f"{OUT}: {len(b.ops)} ops, {games} games and {moves} moves kept, {len(stats_cases)} filters "
        f"({work} areas to work on), {len(positions_cases)} position queries ({found} positions); "
        f"{OUT.stat().st_size // 1024} KB"
    )


if __name__ == "__main__":
    main()
