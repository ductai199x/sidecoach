"""The games Game Review can open, brought in from where they were played
(docs/game-review-plan.md): a pasted or opened PGN, the player's chess.com games through
chess.com's official public data (api.chess.com/pub), their Lichess games through Lichess's
public API. Each becomes a `LibraryRow`: who played, when, at what ratings and time control,
which side was the player's (their username for the site, from the settings), and the PGN.

Only public data is read, one request at a time as chess.com asks, under a User-Agent that
names the app; nothing is sent to the player's accounts.
"""

from __future__ import annotations

import bisect
import calendar
import difflib
import hashlib
import io
import math
import re
import time
from collections.abc import Iterable

import chess.pgn
import httpx

from sidecoach.profile.store import GameRow, LibraryRow, MoveRow, Store, Trace

USER_AGENT = "Sidecoach (a personal chess coach)"
CHESSCOM_ARCHIVES = "https://api.chess.com/pub/player/{user}/games/archives"
LICHESS_GAMES = "https://lichess.org/api/games/user/{user}"
TIMEOUT_S = 20.0


class NoPlayer(Exception):
    """The site knows no player by that name: a username mistyped in Settings."""


def games_in(text: str) -> list[chess.pgn.Game]:
    """Every game in a PGN text that has moves; a text that is no PGN gives none."""
    out: list[chess.pgn.Game] = []
    handle = io.StringIO(text)
    while True:
        game = chess.pgn.read_game(handle)
        if game is None:
            return out
        if game.errors or not any(True for _ in game.mainline_moves()):
            continue
        out.append(game)


def row_for(
    game: chess.pgn.Game,
    *,
    source: str,
    key: str | None = None,
    user_names: Iterable[str] = (),
    time_class: str | None = None,
    now: float | None = None,
) -> LibraryRow:
    """A game as the library keeps it. Without a `key` of its own (a pasted PGN), the key is
    a hash of its moves and players, so the same game pasted twice is one game."""
    headers = game.headers
    pgn = str(game)
    names = {n.strip().lower() for n in user_names if n.strip()}
    white, black = headers.get("White"), headers.get("Black")
    user_side = (
        "white"
        if white and white.lower() in names
        else "black"
        if black and black.lower() in names
        else None
    )
    if key is None:
        movetext = " ".join(m.uci() for m in game.mainline_moves())
        digest = hashlib.sha1(f"{white}|{black}|{headers.get('Date')}|{movetext}".encode())
        key = f"{source}:{digest.hexdigest()[:16]}"
    return LibraryRow(
        key=key,
        source=source,
        added_at=now if now is not None else time.time(),
        pgn=pgn,
        played_at=_played_at(headers),
        white=white,
        black=black,
        white_elo=_int(headers.get("WhiteElo")),
        black_elo=_int(headers.get("BlackElo")),
        result=headers.get("Result"),
        time_class=time_class or time_class_of(headers.get("TimeControl", "")),
        user_side=user_side,
    )


def from_pgn(text: str, *, user_names: Iterable[str] = ()) -> list[LibraryRow]:
    """The games of a pasted or opened PGN."""
    names = list(user_names)
    return [row_for(g, source="pgn", user_names=names) for g in games_in(text)]


def from_chesscom(
    user: str, *, months: int = 3, client: httpx.Client | None = None
) -> list[LibraryRow]:
    """The player's standard games of the last `months` months on chess.com (their public
    archive; the newest month can lag the site by up to a day, their documentation says)."""
    own = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
    try:
        answer = own.get(CHESSCOM_ARCHIVES.format(user=user.lower()))
        if answer.status_code == 404:
            raise NoPlayer(f'chess.com has no player named "{user}".')
        answer.raise_for_status()
        archives = answer.json().get("archives", [])
        rows: list[LibraryRow] = []
        for month in archives[-months:] if months > 0 else archives:
            got = own.get(month)
            got.raise_for_status()
            for entry in got.json().get("games", []):
                if entry.get("rules") != "chess" or not entry.get("pgn"):
                    continue
                for game in games_in(entry["pgn"]):
                    game_id = str(entry.get("url", "")).rstrip("/").rsplit("/", 1)[-1]
                    rows.append(
                        row_for(
                            game,
                            source="chesscom",
                            key=f"chesscom:{game_id}" if game_id else None,
                            user_names=[user],
                            time_class=entry.get("time_class"),
                        )
                    )
        return rows
    finally:
        if client is None:
            own.close()


def from_lichess(
    user: str, *, max_games: int = 60, client: httpx.Client | None = None
) -> list[LibraryRow]:
    """The player's latest standard games on Lichess, with their clocks."""
    own = client or httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_S)
    try:
        response = own.get(
            LICHESS_GAMES.format(user=user),
            params={
                "max": max_games,
                "clocks": "true",
                "opening": "true",
                "perfType": "ultraBullet,bullet,blitz,rapid,classical,correspondence",
            },
            headers={"Accept": "application/x-chess-pgn"},
        )
        if response.status_code == 404:
            raise NoPlayer(f'Lichess has no player named "{user}".')
        response.raise_for_status()
        rows = []
        for game in games_in(response.text):
            if game.headers.get("Variant", "Standard") != "Standard":
                continue
            site = game.headers.get("Site", "")
            game_id = site.rstrip("/").rsplit("/", 1)[-1] if "lichess.org/" in site else ""
            rows.append(
                row_for(
                    game,
                    source="lichess",
                    key=f"lichess:{game_id}" if game_id else None,
                    user_names=[user],
                )
            )
        return rows
    finally:
        if client is None:
            own.close()


def find(store: Store, key: str) -> LibraryRow | None:
    """A game of the library by its key: imported, or one Sidecoach watched."""
    if key.startswith("watched:"):
        game = store.game(key[len("watched:") :])
        if game is None or game.result is None:
            return None
        return watched_row(game, store.moves(game.game_id))
    return store.library_game(key)


# A watched record is a game when it finished with a result and its moves play out from where
# it began for this many plies at least: a puzzle's mating line recorded from the starting
# position, or a game cut off when the board was turned round, is not one (Tai's app, 09-28).
WATCHED_PLIES = 10


def from_watched(store: Store, limit: int | None = None) -> list[LibraryRow]:
    """The games Sidecoach watched to their end, as the library shows them: their moves as
    recorded (withdrawn ones left out), the player named "You", the opponent as recorded. A
    record with no result, or whose moves do not play out as a game, is left out. Every one
    unless `limit`: the Games window listed the latest 200 while the background review and Stats
    counted all (the review of 2026-10-01)."""
    rows = []
    for game in store.games(limit):
        row = watched_row(game, store.moves(game.game_id))
        if row is not None:
            rows.append(row)
    return rows


def watched_row(game: GameRow, moves: list[MoveRow]) -> LibraryRow | None:
    """One watched record as the library shows it, or None when it is no game (`from_watched`)."""
    if not moves or game.result not in ("1-0", "0-1", "1/2-1/2"):
        return None
    pgn = chess.pgn.Game()
    board = chess.Board(game.start_fen)
    if game.start_fen != chess.STARTING_FEN:
        pgn.setup(board)
    node: chess.pgn.GameNode = pgn
    played = 0
    for row in sorted(moves, key=lambda m: (m.segment, m.ply)):
        try:
            move = board.parse_san(row.san)
        except ValueError:
            break  # a resync's new line: the review stops where the record does
        node = node.add_variation(move)
        board.push(move)
        played += 1
    if played < WATCHED_PLIES:
        return None
    you, them = "You", game.opponent or "Opponent"
    pgn.headers["White"] = you if game.user_side == "white" else them
    pgn.headers["Black"] = you if game.user_side == "black" else them
    pgn.headers["Result"] = game.result or "*"
    pgn.headers["Date"] = time.strftime("%Y.%m.%d", time.gmtime(game.started_at))
    if game.time_control:
        pgn.headers["TimeControl"] = game.time_control
    return LibraryRow(
        key=f"watched:{game.game_id}",
        source="watched",
        added_at=game.ended_at or game.started_at,
        pgn=str(pgn),
        played_at=game.started_at,
        white=pgn.headers["White"],
        black=pgn.headers["Black"],
        result=game.result,
        time_class=time_class_of(game.time_control or ""),
        user_side=game.user_side,
    )


# A watched game and an import are one game when the import started at most TWIN_BEFORE_S
# before Sidecoach began following it (it can join late, never early) and no later than
# TWIN_AFTER_S after, and nearly every move Sidecoach recorded -- TWIN_SHARE of them, TWIN_PLIES
# at least -- turns up in the import in order: Sidecoach can misread a move or miss a few, so
# the record need not agree move for move (Tai, 2026-09-30: "sometimes our app is a bit
# inaccurate").
TWIN_BEFORE_S = 3 * 3600.0
TWIN_AFTER_S = 600.0
TWIN_SHARE = 0.9
TWIN_PLIES = 12
# An import that says when in the day it started (chess.com, Lichess); else a date alone, the
# same day or the next: within DATE_WINDOW_S of the watched game's start. The timed window lies
# inside that one, so `reconcile` looks only at the imports played that near.
_TIMED = re.compile(r'^\[UTCTime "', re.MULTILINE)
DATE_WINDOW_S = 36 * 3600.0


def reconcile(store: Store) -> dict[str, str]:
    """Each game Sidecoach watched that was imported too, from chess.com, Lichess or a PGN:
    the import is the game's record -- every move, the clocks, the names, the ratings -- and
    stands for the watched one in the library, the background review and Stats. Pairs found
    are kept in the store; returns them all, watched key -> imported key. Every watched game,
    not the latest 200 (the review of 2026-10-01), each against the imports played near it."""
    twins = store.twins()
    imported = store.library()
    dated = sorted((r.played_at, i) for i, r in enumerate(imported) if r.played_at is not None)
    stamps = [t for t, _ in dated]
    undated = [i for i, r in enumerate(imported) if r.played_at is None]
    moves_of: dict[str, list[str] | None] = {}  # an import's moves, read once
    for game in store.games(None):
        key = f"watched:{game.game_id}"
        if key in twins:
            continue
        lo = bisect.bisect_left(stamps, game.started_at - DATE_WINDOW_S)
        hi = bisect.bisect_right(stamps, game.started_at + DATE_WINDOW_S)
        # In the library's order, as every import was looked at before: the first best wins.
        near = [imported[i] for i in sorted([i for _, i in dated[lo:hi]] + undated)]
        near = [row for row in near if _may_be(game, row)]
        if not near:
            continue
        seen = [m.san.rstrip("+#") for m in sorted(store.moves(game.game_id), key=_order)]
        best: tuple[float, str] | None = None
        for row in near:
            if row.key not in moves_of:
                moves_of[row.key] = _sans(row.pgn)
            share = _agreeing(seen, moves_of[row.key] or [])
            if share is not None and (best is None or share > best[0]):
                best = (share, row.key)
        if best is not None:
            store.add_twin(key, best[1])
            twins[key] = best[1]
    return twins


def _order(move: MoveRow) -> tuple[int, int]:
    return move.segment, move.ply


def left_out(store: Store, rows: list[LibraryRow]) -> set[str]:
    """The games among `rows` the player removed (Store.remove_game), which bringing games in
    again leaves out: by key; by being a game the player pasted and removed (the review of
    2026-10-01: the same game brought in from the site came back under the site's key); or --
    a watched game removed before its import came in -- by being that game, as `reconcile`
    knows an import."""
    return left_out_of(rows, store.removed_keys(), store.removed_traces())


def pasted_key(row: LibraryRow) -> str | None:
    """The key the game has when pasted as a PGN (`row_for` with no key of its own: a hash of
    its players, its date and its moves); None for a PGN with no game in it."""
    if row.source == "pgn":
        return row.key
    games = games_in(row.pgn)
    return row_for(games[0], source="pgn").key if games else None


def left_out_of(rows: list[LibraryRow], keys: set[str], traces: list[Trace]) -> set[str]:
    out: set[str] = set()
    pasted = {k for k in keys if k.startswith("pgn:")}
    for row in rows:
        if row.key in keys or (pasted and pasted_key(row) in pasted):
            out.add(row.key)
            continue
        played: list[str] | None = None
        for trace in traces:
            if not _may_be(trace, row):
                continue
            if played is None:
                played = _sans(row.pgn) or []
            if _agreeing(trace.sans, played) is not None:
                out.add(row.key)
                break
    return out


def _may_be(game: GameRow | Trace, row: LibraryRow) -> bool:
    """Whether an import can be the watched game at all: the same side, the same result when
    both are known, started in the window (a date alone, the same day or the next)."""
    if row.user_side is not None and row.user_side != game.user_side:
        return False
    if row.result and game.result and row.result != "*" and row.result != game.result:
        return False
    if row.played_at is None:
        return True
    if not _TIMED.search(row.pgn):
        return abs(row.played_at - game.started_at) <= DATE_WINDOW_S
    return game.started_at - TWIN_BEFORE_S <= row.played_at <= game.started_at + TWIN_AFTER_S


def _sans(pgn: str) -> list[str] | None:
    games = games_in(pgn)
    if not games:
        return None
    board = games[0].board()
    out = []
    for move in games[0].mainline_moves():
        out.append(board.san(move).rstrip("+#"))
        board.push(move)
    return out


def _agreeing(seen: list[str], played: list[str]) -> float | None:
    """The share of the moves Sidecoach recorded that the import has in the same order, when
    it is enough to call them one game; None otherwise."""
    if not seen or not played:
        return None
    matcher = difflib.SequenceMatcher(a=seen, b=played, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    share = matched / len(seen)
    enough = matched >= min(TWIN_PLIES, len(played))
    return share if share >= TWIN_SHARE and enough else None


_CONTROL = re.compile(r"^(\d+)(?:\+(\d+))?$")
# As Sidecoach words the clock it read at a game's start (profile/profiler.py): "10 min", "30 s";
# a reading this close below a whole minute is that minute (profiler.START_SLACK_MS).
_WATCHED_CONTROL = re.compile(r"^(\d+(?:\.\d+)?) (min|s)$")
_START_SLACK_S = 5.0


def time_class_of(control: str) -> str | None:
    """The time class of a PGN `TimeControl` ("600+5"), or of the clock Sidecoach read ("10
    min"), by its expected length (the base plus forty increments): bullet under 3 minutes,
    blitz under 10, rapid under 60, classical beyond; "1/86400" and the like are
    correspondence ("daily")."""
    if control.startswith("1/"):
        return "daily"
    watched = _WATCHED_CONTROL.match(control)
    match = _CONTROL.match(control)
    if watched and watched.group(2) == "min":
        # A few seconds short of a whole minute is that minute, as profile/profiler.py words it
        # now: records kept before say the first reading as read ("9.98333 min", a ten-minute
        # game counted as blitz; the review of 2026-10-01).
        seconds = float(watched.group(1)) * 60
        whole = math.ceil(seconds / 60) * 60.0
        if whole - seconds <= _START_SLACK_S:
            seconds = whole
    elif watched:
        seconds = float(watched.group(1))
    elif match:
        seconds = int(match.group(1)) + 40 * int(match.group(2) or 0)
    else:
        return None
    if seconds < 180:
        return "bullet"
    if seconds < 600:
        return "blitz"
    if seconds < 3600:
        return "rapid"
    return "classical"


def _played_at(headers: chess.pgn.Headers) -> float | None:
    """When the game was played, from its UTC date and time (chess.com, Lichess), else its
    date alone -- a day on the calendar, taken at noon here so it shows as that day; None when
    the PGN does not say."""
    form = "%Y.%m.%d %H:%M:%S"
    try:
        if headers.get("UTCDate") and headers.get("UTCTime"):
            stamp = f"{headers['UTCDate']} {headers['UTCTime'][:8]}"
            return float(calendar.timegm(time.strptime(stamp, form)))
        return time.mktime(time.strptime(f"{headers.get('Date', '')} 12:00:00", form))
    except ValueError:
        return None


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value not in (None, "", "?") else None
    except ValueError:
        return None
