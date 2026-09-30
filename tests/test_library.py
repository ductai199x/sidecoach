"""The games Game Review can open (profile/library.py, profile/store.py): brought in from a PGN,
chess.com's public archive or Lichess's API, which side was the player's, kept and reviewed."""

from __future__ import annotations

import json

import chess
import httpx

from sidecoach.profile.library import (
    NoPlayer,
    from_chesscom,
    from_lichess,
    from_pgn,
    from_watched,
    reconcile,
    time_class_of,
)
from sidecoach.profile.store import Store

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
OTHER = GAME.replace("Anakin_737", "Someone").replace("3. Bc4 Nf6", "3. Bb5 a6")


def test_a_pasted_pgn_brings_in_every_game_and_knows_whose_side() -> None:
    rows = from_pgn(GAME + "\n" + OTHER + "\nnot a game at all", user_names=["NoseGazm"])
    assert len(rows) == 2
    first = rows[0]
    assert first.source == "pgn" and first.key.startswith("pgn:")
    assert (first.white, first.black, first.white_elo, first.black_elo) == (
        "Anakin_737",
        "nosegazm",
        880,
        1018,
    )
    assert first.user_side == "black"  # the name matched, whatever its case
    assert first.time_class == "rapid" and first.result == "0-1"
    assert first.played_at == 1790724600.0  # 2026-09-29 23:30 UTC
    # The same game pasted again is the same game.
    assert from_pgn(GAME, user_names=["nosegazm"])[0].key == first.key
    assert rows[1].key != first.key


def test_time_classes_by_the_expected_length_of_a_game() -> None:
    assert [
        time_class_of(c)
        for c in ("60", "120+1", "180", "300+5", "600", "900+10", "5400+30", "1/86400", "?")
    ] == [
        "bullet",
        "bullet",  # 2+1: 160 s expected, bullet on chess.com and Lichess alike
        "blitz",
        "blitz",
        "rapid",
        "rapid",
        "classical",
        "daily",
        None,
    ]
    # As Sidecoach words the clock it read at a watched game's start.
    assert [time_class_of(c) for c in ("30 s", "3 min", "15 min", "2.5 min")] == [
        "bullet",
        "blitz",
        "rapid",
        "bullet",
    ]


def test_a_username_the_site_does_not_know_is_said_as_such() -> None:
    """A mistyped username: the site's 404, not "no games found"."""
    import pytest

    missing = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(404, json={})))
    with pytest.raises(NoPlayer, match='chess.com has no player named "nosegazmm"'):
        from_chesscom("nosegazmm", client=missing)
    with pytest.raises(NoPlayer, match="Lichess has no player"):
        from_lichess("nosegazmm", client=missing)


def test_chesscom_games_come_from_the_public_archive_one_month_at_a_time() -> None:
    asked: list[str] = []

    def answer(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        assert "Sidecoach" in request.headers["user-agent"]
        if request.url.path.endswith("/archives"):
            return httpx.Response(
                200,
                json={
                    "archives": [
                        "https://api.chess.com/pub/player/nosegazm/games/2026/07",
                        "https://api.chess.com/pub/player/nosegazm/games/2026/08",
                        "https://api.chess.com/pub/player/nosegazm/games/2026/09",
                    ]
                },
            )
        games = [
            {
                "url": "https://www.chess.com/game/live/184578971282",
                "pgn": GAME,
                "rules": "chess",
                "time_class": "rapid",
            },
            {
                "url": "https://www.chess.com/game/live/1",
                "pgn": GAME,
                "rules": "chess960",
                "time_class": "rapid",
            },  # not standard chess: left out
        ]
        return httpx.Response(200, json={"games": games})

    client = httpx.Client(
        transport=httpx.MockTransport(answer), headers={"User-Agent": "Sidecoach test"}
    )
    rows = from_chesscom("NoseGazm", months=2, client=client)
    # The archive list (asked in lower case), then the last two months only.
    assert asked[0].endswith("/player/nosegazm/games/archives")
    assert [a.rsplit("/games/", 1)[1] for a in asked[1:]] == ["2026/08", "2026/09"]
    assert [r.key for r in rows] == ["chesscom:184578971282", "chesscom:184578971282"]
    assert rows[0].user_side == "black" and rows[0].source == "chesscom"


def test_lichess_games_come_as_pgn_with_their_clocks() -> None:
    lichess = GAME.replace('[Site "Chess.com"]', '[Site "https://lichess.org/AbCdEfGh"]')
    atomic = lichess.replace(
        '[Site "https://lichess.org/AbCdEfGh"]',
        '[Site "https://lichess.org/Zz"]\n[Variant "Atomic"]',
    )

    def answer(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "application/x-chess-pgn"
        assert request.url.params["clocks"] == "true"
        return httpx.Response(200, text=lichess + "\n" + atomic)

    rows = from_lichess("nosegazm", client=httpx.Client(transport=httpx.MockTransport(answer)))
    assert [r.key for r in rows] == ["lichess:AbCdEfGh"]  # the atomic game left out


LONG = (
    "1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. d3 Bc5 5. c3 d6 6. O-O O-O 7. Re1 a6 8. Bb3 Ba7"
    " 9. h3 h6 10. Nbd2 Re8 0-1"
)
SANS = [t for t in LONG.split() if not t[0].isdigit()]  # 20 plies


def _watch(store: Store, game_id: str, sans: list[str], started_at: float, **kw: str) -> str:
    """A game as Sidecoach records one it watched: its moves as read, in order."""
    store.start_game(
        game_id, started_at, kw.get("side", "black"), kw.get("fen", chess.STARTING_FEN)
    )
    for ply, san in enumerate(sans, start=1):
        store.add_move(
            game_id, 0, ply, move_number=(ply + 1) // 2, san=san, uci="0000", by_user=ply % 2 == 0,
            fen_before="-", fen_after="-", phase=None, think_ms=None, clock_ms=None,
            capture=False, gives_check=False, material=None, ts=started_at + ply,
        )  # fmt: skip
    store.end_game(game_id, started_at + 900, kw.get("result", "0-1"), "resignation")
    return f"watched:{game_id}"


def _stats(store: Store, key: str) -> None:
    store.save_stats(
        "standard",
        {"key": key, "source": key.split(":")[0], "time_class": "rapid", "played_at": 0.0,
         "user_side": "black", "bot": False, "result": "0-1", "opening": None,
         "termination": None},
        [],
    )  # fmt: skip


def test_a_watched_game_imported_too_is_one_game(tmp_path) -> None:  # noqa: ANN001
    """The import stands for the watched game when nearly every move Sidecoach read is in it,
    in order -- a misread move and the last moves missed are no matter -- and the game started
    in the window; the library, the background review and Stats have it once."""
    store = Store(tmp_path / "profile.sqlite")
    (row,) = from_pgn(
        GAME.replace("1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 0-1", LONG), user_names=["nosegazm"]
    )
    store.add_library_game(row)
    start = row.played_at or 0.0
    misread = [*SANS[:6], "d4", *SANS[7:18]]  # 4. d3 read as d4, the last two moves missed
    twin = _watch(store, "g1", misread, start + 60)
    # Joined mid-game: the moves from its 7th ply on.
    late = _watch(store, "g2", SANS[6:], start + 300)
    # Another game, the same opening for six plies, then its own way.
    own_way = [
        "c3",
        "O-O",
        "d4",
        "exd4",
        "cxd4",
        "Bb4+",
        "Bd2",
        "Bxd2+",
        "Nbxd2",
        "d5",
        "exd5",
        "Nxd5",
        "Qb3",
    ]
    other = _watch(store, "g3", [*SANS[:6], *own_way], start + 120)
    # The same moves, but hours after the import had started: not that game.
    too_late = _watch(store, "g4", SANS, start + 5 * 3600)
    twins = reconcile(store)
    assert twins == {twin: row.key, late: row.key}
    assert other not in twins and too_late not in twins
    assert store.twins() == twins, "kept"

    # Stats has the game once, as the import.
    for key in (row.key, twin, other):
        _stats(store, key)
    assert {g["key"] for g in store.stat_games()} == {row.key, other}
    # The import taken out of the library: the watched game is its own again.
    store.delete_library_game(row.key)
    assert twin not in store.twins()
    store.close()


def test_a_watched_record_that_is_no_game_stays_out(tmp_path) -> None:  # noqa: ANN001
    """What Sidecoach recorded is a game when it finished and its moves play out from where it
    began: a puzzle's mating line recorded from the start position, a game never finished,
    six plies -- the records in Tai's app on 09-28 -- are not listed (nor reviewed, nor
    counted)."""
    store = Store(tmp_path / "profile.sqlite")
    good = _watch(store, "g1", SANS, 1000.0)
    _watch(store, "g2", ["Rxf2", "Rg3+", "Kf7", "Qg6+", "Kf8", "Qg8#"], 2000.0, result="1-0")
    _watch(store, "g3", SANS, 3000.0, result="*")
    _watch(store, "g4", SANS[:6], 4000.0)
    assert [r.key for r in from_watched(store)] == [good]
    store.close()


def test_the_library_keeps_games_and_their_reviews(tmp_path) -> None:  # noqa: ANN001
    store = Store(tmp_path / "profile.sqlite")
    older, newer = from_pgn(
        GAME + "\n" + OTHER.replace("23:30:00", "23:50:00"), user_names=["nosegazm"]
    )
    store.add_library_game(older)
    store.add_library_game(newer)
    assert [r.key for r in store.library()] == [newer.key, older.key]  # the latest played first
    assert store.library_game(older.key) == older

    store.save_review(
        older.key, "standard", 1, 100.0, json.dumps({"moves": []}), json.dumps({"accuracy": 80})
    )
    store.set_review_words(older.key, "standard", json.dumps({"1": "Book."}))
    review = store.review(older.key, "standard")
    assert (
        review is not None
        and review.version == 1
        and json.loads(review.words or "{}") == {"1": "Book."}
    )
    # Reviewed again: the words of the older review go with it.
    store.save_review(older.key, "standard", 2, 200.0, "{}", None)
    again = store.review(older.key, "standard")
    assert again is not None and again.version == 2 and again.words is None
    assert store.review(older.key, "deep") is None

    store.delete_library_game(older.key)
    assert store.library_game(older.key) is None and store.review(older.key, "standard") is None
    store.close()
