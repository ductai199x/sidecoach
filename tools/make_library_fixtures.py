"""Record what the desktop's games library makes of its inputs, for the Kotlin port
(profile/Library.kt).

- `pgn`: pasted PGN texts (the awkward headers by hand, three of Tai's chess.com games from a
  copy of his data, random games) under several usernames -> the rows `from_pgn` gives.
- `time_classes`: TimeControl texts and Sidecoach's own clock words -> `time_class_of`.
- `blocks`: pairs of move lists -> difflib's matching blocks and `_agreeing`'s share.
- `chesscom` / `lichess`: sites' answers (by hand, as their APIs give them) -> the URLs asked,
  in order, and the rows, or the error said.
- `stores`: stores of watched records (random games, misread, cut short, joined late, resynced,
  moves withdrawn) and imports (the same games, in and out of the window, the wrong side or
  result, a date alone) -> `from_watched`'s rows and `reconcile`'s pairs.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_library_fixtures.py
"""

from __future__ import annotations

import calendar
import dataclasses
import json
import logging
import random
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import chess
import chess.pgn
import httpx

from sidecoach.profile import library
from sidecoach.profile.store import Store

OUT = Path("D:/sidecoach-android/app/src/androidTest/assets/profile/library.json")
COPY = Path(
    "C:/Users/Tai/AppData/Local/Temp/claude/d--chess-coaching-companion/"
    "5a849ff1-467c-42d3-861a-c579f032804a/scratchpad/appcopy/profile.sqlite"
)
logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)

GAME = """[Event "Live Chess"]
[Site "Chess.com"]
[Date "2026.09.29"]
[White "Anakin_737"]
[Black "nosegazm"]
[Result "0-1"]
[WhiteElo "880"]
[BlackElo "1018"]
[TimeControl "600"]
[UTCDate "2026.09.29"]
[UTCTime "23:30:00"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 0-1
"""
LICHESS = GAME.replace('[Site "Chess.com"]', '[Site "https://lichess.org/AbCdEfGh"]')

PGNS = {
    "a chess.com game": GAME,
    "two games and some text": GAME + "\n" + GAME.replace("Anakin_737", "Someone").replace("3. Bc4 Nf6", "3. Bb5 a6") + "\nnot a game at all",
    "a date alone": GAME.replace('[UTCDate "2026.09.29"]\n[UTCTime "23:30:00"]\n', ""),
    "no date": GAME.replace('[Date "2026.09.29"]', '[Date "????.??.??"]').replace('[UTCDate "2026.09.29"]\n[UTCTime "23:30:00"]\n', ""),
    "a day that is not": GAME.replace('[UTCDate "2026.09.29"]', '[UTCDate "2026.02.30"]'),
    "single digits": GAME.replace('[UTCDate "2026.09.29"]', '[UTCDate "2026.9.3"]').replace('"23:30:00"', '"7:05:09"'),
    "a time with more": GAME.replace('"23:30:00"', '"23:30:00.250"'),
    "a time with less": GAME.replace('"23:30:00"', '"23:30"'),
    "a date alone, single digits": GAME.replace('[UTCDate "2026.09.29"]\n[UTCTime "23:30:00"]\n', "").replace('"2026.09.29"', '"2026.1.5"'),
    "odd ratings": GAME.replace('"880"', '"?"').replace('"1018"', '" 1018 "'),
    "no ratings": GAME.replace('[WhiteElo "880"]\n', "").replace('[BlackElo "1018"]', '[BlackElo "abc"]'),
    "no names": GAME.replace('"Anakin_737"', '""').replace('"nosegazm"', '"?"'),
    "a set-up position": '[White "nosegazm"]\n[FEN "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"]\n[SetUp "1"]\n\n1. e4 Kd7 2. e5 Ke6 1/2-1/2\n',
    "controls": "".join(
        GAME.replace('"600"', f'"{c}"').replace("3. Bc4 Nf6", f"3. Bc4 {m}") + "\n"
        for c, m in (("60", "Nf6"), ("180+2", "Bc5"), ("1/86400", "Be7"), ("-", "d6"), ("5400+30", "h6"))
    ),
    "lichess": LICHESS,
    "no moves": '[White "nosegazm"]\n\n*\n',
    "an illegal move": GAME.replace("3. Bc4 Nf6", "3. Ke3 Nf6"),
}
NAMES = [["NoseGazm"], [], [" nosegazm ", ""], ["anakin_737", "nosegazm"], ["someone"]]
CONTROLS = [
    "60", "120+1", "180", "300+5", "600", "900+10", "5400+30", "1/86400", "?", "", "-",
    "30 s", "3 min", "15 min", "2.5 min", "10 min", "60 min", "1.5 s", "0", "179", "180+0",
    "599", "3599", "3600", "1/", "600+", "+5", "10  min", "10 min ", "5 hours",
]  # fmt: skip


def row_json(row: library.LibraryRow) -> dict:
    out = dataclasses.asdict(row)
    out.pop("added_at")
    return out


def pgn_row(row: library.LibraryRow) -> dict:
    """A pasted game's row, and where its time is a date alone, that day's noon in UTC (the
    device's own zone applied by the test, as `mktime` applies this machine's)."""
    out = row_json(row)
    game = library.games_in(row.pgn)[0]
    h = game.headers
    if not (h.get("UTCDate") and h.get("UTCTime")) and row.played_at is not None:
        stamp = time.strptime(f"{h.get('Date', '')} 12:00:00", "%Y.%m.%d %H:%M:%S")
        out["played_at"] = None
        out["utc_noon"] = float(calendar.timegm(stamp))
    return out


def random_pgn(rng: random.Random) -> str:
    game = chess.pgn.Game()
    game.headers["White"] = rng.choice(["nosegazm", "NoseGazm", "Bob", "?"])
    game.headers["Black"] = rng.choice(["nosegazm", "Alice", ""])
    game.headers["Date"] = f"20{rng.randrange(20, 27)}.{rng.randrange(1, 13):02d}.{rng.randrange(1, 29):02d}"
    if rng.random() < 0.6:
        game.headers["UTCDate"] = game.headers["Date"]
        game.headers["UTCTime"] = f"{rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d}"
    game.headers["WhiteElo"] = str(rng.randrange(400, 2800))
    game.headers["TimeControl"] = rng.choice(["600", "180+2", "60", "1/86400", "900+10"])
    board = chess.Board()
    node: chess.pgn.GameNode = game
    for _ in range(rng.randrange(1, 60)):
        moves = list(board.legal_moves)
        if not moves:
            break
        move = rng.choice(moves)
        node = node.add_variation(move)
        board.push(move)
    game.headers["Result"] = board.result() if board.is_game_over() else rng.choice(["*", "1-0", "0-1", "1/2-1/2"])
    return str(game) + "\n"


def blocks_cases(rng: random.Random) -> list[dict]:
    cases = []
    for _ in range(300):
        alphabet = rng.choice(["ab", "abc", "abcdef", "abcdefghijklmnop"])
        a = [rng.choice(alphabet) for _ in range(rng.randrange(0, 30))]
        b = [rng.choice(alphabet) for _ in range(rng.randrange(0, 30))]
        if rng.random() < 0.4 and a:  # b as a with a few changes: the common case
            b = list(a)
            for _ in range(rng.randrange(0, 4)):
                op = rng.randrange(3)
                i = rng.randrange(len(b) + 1)
                if op == 0 and b:
                    del b[min(i, len(b) - 1)]
                elif op == 1:
                    b.insert(i, rng.choice(alphabet))
                elif b:
                    b[min(i, len(b) - 1)] = rng.choice(alphabet)
        matcher = __import__("difflib").SequenceMatcher(a=a, b=b, autojunk=False)
        cases.append(
            {
                "a": a,
                "b": b,
                "blocks": [list(m) for m in matcher.get_matching_blocks()],
                "agreeing": library._agreeing(a, b),
            }
        )
    return cases


class Site:
    """A site's answers by URL, as `httpx.MockTransport` gives them; every URL asked kept."""

    def __init__(self, answers: dict[str, tuple[int, str]]) -> None:
        self.answers = answers
        self.asked: list[str] = []

    def client(self) -> httpx.Client:
        def answer(request: httpx.Request) -> httpx.Response:
            url = str(request.url).split("?", 1)[0]
            self.asked.append(url)
            status, text = self.answers.get(url, (404, "{}"))
            return httpx.Response(status, text=text)

        return httpx.Client(transport=httpx.MockTransport(answer))


ARCHIVES = "https://api.chess.com/pub/player/nosegazm/games/archives"
MONTH = "https://api.chess.com/pub/player/nosegazm/games/2026/{:02d}"


def chesscom_cases(chesscom_pgns: list[str]) -> list[dict]:
    entries = [
        {"url": "https://www.chess.com/game/live/184578971282", "pgn": GAME, "rules": "chess", "time_class": "rapid"},
        {"url": "https://www.chess.com/game/live/1", "pgn": GAME, "rules": "chess960", "time_class": "rapid"},
        {"url": "https://www.chess.com/game/daily/77/", "pgn": GAME.replace("3. Bc4", "3. Bb5"), "rules": "chess", "time_class": "daily"},
        {"pgn": GAME.replace("3. Bc4", "3. d3"), "rules": "chess"},  # no url: keyed by its moves
        {"url": "https://www.chess.com/game/live/2", "pgn": "", "rules": "chess"},
        {"url": "https://www.chess.com/game/live/3", "rules": "chess"},
        {"url": "https://www.chess.com/game/live/4", "pgn": GAME, "rules": "chess", "time_class": None},
        {"url": "https://www.chess.com/game/live/5", "pgn": "garbage", "rules": "chess"},
    ]
    for i, pgn in enumerate(chesscom_pgns):
        entries.append({"url": f"https://www.chess.com/game/live/9{i}", "pgn": pgn, "rules": "chess", "time_class": "blitz"})
    months = [MONTH.format(m) for m in (6, 7, 8, 9)]
    answers = {ARCHIVES: (200, json.dumps({"archives": months}))}
    for i, m in enumerate(months):
        answers[m] = (200, json.dumps({"games": entries[i::2]}))
    cases = []
    for name, user, got, how_many in (
        ("the last two months", "NoseGazm", answers, 2),
        ("every month", "nosegazm", answers, 0),
        ("more months than there are", "nosegazm", answers, 12),
        ("no player", "nosegazmm", {}, 3),
        ("a month the site fails", "nosegazm", {**answers, months[-1]: (500, "oops")}, 3),
        ("the archive list fails", "nosegazm", {ARCHIVES: (503, "busy")}, 3),
        ("no archives", "nosegazm", {ARCHIVES: (200, "{}")}, 3),
        ("a month with no games key", "nosegazm", {**answers, months[-1]: (200, "{}")}, 1),
    ):
        site = Site(got)
        case: dict = {"name": name, "user": user, "months": how_many, "answers": got}
        try:
            rows = library.from_chesscom(user, months=how_many, client=site.client())
            case["rows"] = [row_json(r) for r in rows]
        except library.NoPlayer as exc:
            case["no_player"] = str(exc)
        except httpx.HTTPStatusError:
            case["http_error"] = True
        case["asked"] = site.asked
        cases.append(case)
    return cases


LICHESS_URL = "https://lichess.org/api/games/user/{}"


def lichess_cases(rng: random.Random) -> list[dict]:
    atomic = LICHESS.replace('[Site "https://lichess.org/AbCdEfGh"]', '[Site "https://lichess.org/Zz"]\n[Variant "Atomic"]')
    c960 = LICHESS.replace('[Site "https://lichess.org/AbCdEfGh"]', '[Site "https://lichess.org/C9"]\n[Variant "Chess960"]')
    standard = LICHESS.replace('[Site "https://lichess.org/AbCdEfGh"]', '[Site "https://lichess.org/St/"]\n[Variant "Standard"]')
    elsewhere = LICHESS.replace('[Site "https://lichess.org/AbCdEfGh"]', '[Site "somewhere"]')
    text = "\n".join([LICHESS, atomic, c960, standard, elsewhere, *(random_pgn(rng) for _ in range(5))])
    cases = []
    for name, user, got in (
        ("games", "NoseGazm", {LICHESS_URL.format("NoseGazm"): (200, text)}),
        ("no player", "nosegazmm", {}),
        ("the site fails", "nosegazm", {LICHESS_URL.format("nosegazm"): (429, "slow down")}),
        ("nothing", "nosegazm", {LICHESS_URL.format("nosegazm"): (200, "")}),
    ):
        site = Site(got)
        case: dict = {"name": name, "user": user, "answers": got}
        try:
            case["rows"] = [row_json(r) for r in library.from_lichess(user, client=site.client())]
        except library.NoPlayer as exc:
            case["no_player"] = str(exc)
        except httpx.HTTPStatusError:
            case["http_error"] = True
        case["asked"] = site.asked
        cases.append(case)
    return cases


def random_line(rng: random.Random, board: chess.Board, plies: int) -> list[chess.Move]:
    out = []
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        move = rng.choice(moves)
        out.append(move)
        board.push(move)
    return out


def store_case(rng: random.Random, folder: Path, index: int) -> dict:
    store = Store(folder / f"s{index}.sqlite")
    base = 1_780_000_000 + rng.random() * 10_000_000
    for g in range(rng.randrange(1, 6)):
        gid = f"g{index}-{g}"
        board = chess.Board()
        if rng.random() < 0.2:  # joined mid-game, where Sidecoach began
            random_line(rng, board, rng.randrange(2, 20))
        start_fen = board.fen()
        start_board = board.copy()
        line = random_line(rng, board, rng.randrange(4, 70))
        b = start_board.copy()
        sans = []
        for move in line:
            sans.append(b.san(move))
            b.push(move)
        recorded = list(sans)
        mood = rng.random()
        if mood < 0.15 and len(recorded) > 6:
            recorded[rng.randrange(len(recorded))] = rng.choice(["d4", "Nf3", "Qxf7#", "O-O", "exd5"])
        elif mood < 0.3:
            recorded = recorded[: max(1, len(recorded) - rng.randrange(1, 6))]
        elif mood < 0.4:
            recorded = recorded[rng.randrange(1, 8):] or recorded
        elif mood < 0.5:
            recorded = [s.rstrip("+#") for s in recorded]
        side = rng.choice(["white", "black"])
        started = base + rng.uniform(0, 30_000)
        store.start_game(gid, started, side, start_fen if rng.random() < 0.9 else chess.STARTING_FEN)
        store.set_game(
            gid,
            time_control=rng.choice([None, "10 min", "3 min", "600+5", "30 s"]),
            opponent=rng.choice([None, "", "Bob", "Magnus"]),
        )
        resync = rng.randrange(len(recorded)) if rng.random() < 0.15 else None
        for ply, san in enumerate(recorded, start=1):
            segment = 1 if resync is not None and ply > resync else 0
            store.add_move(
                gid, segment, ply, move_number=(ply + 1) // 2, san=san, uci="0000", by_user=ply % 2 == 0,
                fen_before="-", fen_after="-", phase=None, think_ms=None, clock_ms=None,
                capture=False, gives_check=False, material=None, ts=started + ply,
            )  # fmt: skip
        if rng.random() < 0.15 and len(recorded) > 3:
            store.withdraw_moves(gid, 0, rng.randrange(1, len(recorded)))
        result = rng.choice(["1-0", "0-1", "1/2-1/2", "*", "1-0"])
        if rng.random() < 0.9:
            ended = started + rng.uniform(0, 2000) if rng.random() < 0.9 else 0.0
            store.end_game(gid, ended, result, "resignation")
        if rng.random() < 0.75:  # the same game imported
            pgn = chess.pgn.Game()
            if start_fen != chess.STARTING_FEN and rng.random() < 0.5:
                pgn.setup(start_board)  # from where Sidecoach began
                moves = line
            else:  # the whole game, from its start
                moves = [*start_board.move_stack, *line]
            node: chess.pgn.GameNode = pgn
            for move in moves:
                node = node.add_variation(move)
            names = {"white": "nosegazm", "black": "Opp"} if side == "white" else {"white": "Opp", "black": "nosegazm"}
            if rng.random() < 0.1:
                names = {"white": names["black"], "black": names["white"]}  # the wrong side
            if rng.random() < 0.1:
                names = {"white": "Opp", "black": "Other"}  # whose side unknown
            pgn.headers["White"], pgn.headers["Black"] = names["white"], names["black"]
            pgn.headers["Result"] = result if rng.random() < 0.85 else rng.choice(["1-0", "0-1", "*", "1/2-1/2"])
            offset = rng.choice([-4 * 3600, -2 * 3600, -600, -30, 0, 60, 500, 700, 3600]) + rng.uniform(-50, 50)
            when = time.gmtime(started + offset)
            pgn.headers["Date"] = time.strftime("%Y.%m.%d", when)
            if rng.random() < 0.8:
                pgn.headers["UTCDate"] = pgn.headers["Date"]
                pgn.headers["UTCTime"] = time.strftime("%H:%M:%S", when)
            for row in library.from_pgn(str(pgn), user_names=["nosegazm"]):
                store.add_library_game(row)
    for _ in range(rng.randrange(0, 3)):  # games of the library that are no watched game
        for row in library.from_pgn(random_pgn(rng), user_names=["nosegazm"]):
            store.add_library_game(row)
    games = store.games(200)
    case = {
        "games": [dataclasses.asdict(g) for g in games],
        "moves": {g.game_id: [dataclasses.asdict(m) for m in store.moves(g.game_id)] for g in games},
        "library": [dataclasses.asdict(r) for r in store.library()],
        "watched": [row_json(r) for r in library.from_watched(store)],
        "twins": library.reconcile(store),
    }
    store.close()
    return case


def by_hand_store(folder: Path) -> dict:
    """tests/test_library.py's store: a misread twin, one joined late, another game, one too late."""
    store = Store(folder / "by-hand.sqlite")
    long = (
        "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. d3 Bc5 5. c3 d6 6. O-O O-O 7. Re1 a6 8. Bb3 Ba7"
        " 9. h3 h6 10. Nbd2 Re8 0-1"
    )
    sans = [t for t in long.split() if not t[0].isdigit()]
    (row,) = library.from_pgn(GAME.replace("1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 0-1", long), user_names=["nosegazm"])
    store.add_library_game(row)
    start = row.played_at or 0.0

    def watch(gid: str, moves: list[str], at: float) -> None:
        store.start_game(gid, at, "black", chess.STARTING_FEN)
        for ply, san in enumerate(moves, start=1):
            store.add_move(
                gid, 0, ply, move_number=(ply + 1) // 2, san=san, uci="0000", by_user=ply % 2 == 0,
                fen_before="-", fen_after="-", phase=None, think_ms=None, clock_ms=None,
                capture=False, gives_check=False, material=None, ts=at + ply,
            )  # fmt: skip
        store.end_game(gid, at + 900, "0-1", "resignation")

    watch("g1", [*sans[:6], "d4", *sans[7:18]], start + 60)
    watch("g2", sans[6:], start + 300)
    own_way = ["c3", "O-O", "d4", "exd4", "cxd4", "Bb4+", "Bd2", "Bxd2+", "Nbxd2", "d5", "exd5", "Nxd5", "Qb3"]
    watch("g3", [*sans[:6], *own_way], start + 120)
    watch("g4", sans, start + 5 * 3600)
    games = store.games(200)
    case = {
        "games": [dataclasses.asdict(g) for g in games],
        "moves": {g.game_id: [dataclasses.asdict(m) for m in store.moves(g.game_id)] for g in games},
        "library": [dataclasses.asdict(r) for r in store.library()],
        "watched": [row_json(r) for r in library.from_watched(store)],
        "twins": library.reconcile(store),
    }
    store.close()
    return case


def main() -> None:
    rng = random.Random(20260930)
    chesscom_pgns: list[str] = []
    if COPY.exists():
        db = sqlite3.connect(COPY)
        chesscom_pgns = [p for (p,) in db.execute("SELECT pgn FROM library WHERE source = 'chesscom' ORDER BY key LIMIT 3")]
        db.close()
    else:
        print("no copy of Tai's data: its chess.com games are left out", file=sys.stderr)
    texts = dict(PGNS)
    for i, pgn in enumerate(chesscom_pgns):
        texts[f"chess.com {i}"] = pgn
    for i in range(30):
        texts[f"random {i}"] = random_pgn(rng)
    pgn_cases = [
        {"name": name, "text": text, "names": names, "rows": [pgn_row(r) for r in library.from_pgn(text, user_names=names)]}
        for name, text in texts.items()
        for names in NAMES
    ]
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        stores = [by_hand_store(folder), *(store_case(rng, folder, i) for i in range(60))]
    fixture = {
        "pgn": pgn_cases,
        "time_classes": [[c, library.time_class_of(c)] for c in CONTROLS],
        "blocks": blocks_cases(rng),
        "chesscom": chesscom_cases(chesscom_pgns),
        "lichess": lichess_cases(rng),
        "stores": stores,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    twins = sum(len(s["twins"]) for s in stores)
    watched = sum(len(s["watched"]) for s in stores)
    games = sum(len(s["games"]) for s in stores)
    print(
        f"{OUT}: {len(pgn_cases)} pasted texts ({sum(len(c['rows']) for c in pgn_cases)} rows), "
        f"{len(fixture['blocks'])} block cases, {len(stores)} stores ({games} records, {watched} "
        f"games watched, {twins} twins)"
    )


if __name__ == "__main__":
    main()
