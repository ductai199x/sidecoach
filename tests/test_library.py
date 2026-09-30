"""The games Game Review can open (profile/library.py, profile/store.py): brought in from a PGN,
chess.com's public archive or Lichess's API, which side was the player's, kept and reviewed."""

from __future__ import annotations

import json

import httpx

from sidecoach.profile.library import from_chesscom, from_lichess, from_pgn, time_class_of
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
