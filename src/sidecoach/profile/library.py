"""The games Game Review can open, brought in from where they were played
(docs/game-review-plan.md): a pasted or opened PGN, the player's chess.com games through
chess.com's official public data (api.chess.com/pub), their Lichess games through Lichess's
public API. Each becomes a `LibraryRow`: who played, when, at what ratings and time control,
which side was the player's (their username for the site, from the settings), and the PGN.

Only public data is read, one request at a time as chess.com asks, under a User-Agent that
names the app; nothing is sent to the player's accounts.
"""

from __future__ import annotations

import calendar
import hashlib
import io
import re
import time
from collections.abc import Iterable

import chess.pgn
import httpx

from sidecoach.profile.store import LibraryRow, Store

USER_AGENT = "Sidecoach (a personal chess coach)"
CHESSCOM_ARCHIVES = "https://api.chess.com/pub/player/{user}/games/archives"
LICHESS_GAMES = "https://lichess.org/api/games/user/{user}"
TIMEOUT_S = 20.0


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
        archives = own.get(CHESSCOM_ARCHIVES.format(user=user.lower())).json().get("archives", [])
        rows: list[LibraryRow] = []
        for month in archives[-months:] if months > 0 else archives:
            for entry in own.get(month).json().get("games", []):
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
        return next((r for r in from_watched(store) if r.key == key), None)
    return store.library_game(key)


def from_watched(store: Store, limit: int = 200) -> list[LibraryRow]:
    """The games Sidecoach watched to their end, as the library shows them: their moves as
    recorded (withdrawn ones left out), the player named "You", the opponent as recorded."""
    rows = []
    for game in store.games(limit):
        moves = store.moves(game.game_id)
        if not moves:
            continue
        pgn = chess.pgn.Game()
        board = chess.Board(game.start_fen)
        if game.start_fen != chess.STARTING_FEN:
            pgn.setup(board)
        node: chess.pgn.GameNode = pgn
        for row in sorted(moves, key=lambda m: (m.segment, m.ply)):
            try:
                move = board.parse_san(row.san)
            except ValueError:
                break  # a resync's new line: the review stops where the record does
            node = node.add_variation(move)
            board.push(move)
        you, them = "You", game.opponent or "Opponent"
        pgn.headers["White"] = you if game.user_side == "white" else them
        pgn.headers["Black"] = you if game.user_side == "black" else them
        pgn.headers["Result"] = game.result or "*"
        pgn.headers["Date"] = time.strftime("%Y.%m.%d", time.gmtime(game.started_at))
        if game.time_control:
            pgn.headers["TimeControl"] = game.time_control
        rows.append(
            LibraryRow(
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
        )
    return rows


_CONTROL = re.compile(r"^(\d+)(?:\+(\d+))?$")


def time_class_of(control: str) -> str | None:
    """The time class of a PGN `TimeControl`, by its expected length (the base plus forty
    increments): bullet under 3 minutes, blitz under 10, rapid under 60, classical beyond;
    "1/86400" and the like are correspondence ("daily")."""
    if control.startswith("1/"):
        return "daily"
    match = _CONTROL.match(control)
    if not match:
        return None
    seconds = int(match.group(1)) + 40 * int(match.group(2) or 0)
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
