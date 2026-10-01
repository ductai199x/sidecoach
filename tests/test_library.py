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
    # The game removed: the import and the watched records it stood for, all of it (a watched
    # record coming back as the import goes would be the same game again); Undo puts all back.
    assert set(store.remove_game(row.key)) == {row.key, twin, late}
    assert store.library_game(row.key) is None and store.twins() == {}
    assert not {twin, late} & {r.key for r in from_watched(store)}
    assert {g["key"] for g in store.stat_games()} == {other}
    assert set(store.restore_game(row.key)) == {row.key, twin, late}
    assert store.twins() == twins and store.library_game(row.key) == row
    assert {g["key"] for g in store.stat_games()} == {row.key, other}
    store.close()


def test_a_removed_game_stays_out_and_undo_puts_it_back(tmp_path, monkeypatch) -> None:  # noqa: ANN001
    """Tai, 2026-10-01: any game can be removed -- its rows go, every one, and bringing games in
    again leaves it out; a watched game removed before its import came in leaves that import
    out too (it is that game). Undo puts it back as it was, for a while; after that only its key
    is kept."""
    from sidecoach.profile import store as store_module
    from sidecoach.profile.library import left_out

    store = Store(tmp_path / "profile.sqlite")
    (row,) = from_pgn(
        GAME.replace("1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 0-1", LONG), user_names=["nosegazm"]
    )
    store.add_library_game(row)
    store.save_review(row.key, "standard", 1, 1.0, "{}", None)
    _stats(store, row.key)

    assert store.remove_game(row.key) == [row.key]
    assert store.remove_game(row.key) == [], "a second click keeps what the first took"
    assert store.library_game(row.key) is None and store.review(row.key, "standard") is None
    assert store.stat_games() == [] and store.is_removed(row.key)
    assert left_out(store, [row]) == {row.key}, "bringing it in again leaves it out"
    assert store.restore_game(row.key) == [row.key]
    assert store.library_game(row.key) == row and store.review(row.key, "standard") is not None
    assert [g["key"] for g in store.stat_games()] == [row.key] and left_out(store, [row]) == set()

    # Watched, removed, then imported: the import is that game, so it stays out.
    store.remove_game(row.key)
    store.unremove([row.key])  # pasted again by hand: no longer left out by its key
    watched = _watch(store, "g1", SANS, (row.played_at or 0.0) + 60)
    assert store.remove_game(watched) == [watched]
    assert store.moves("g1") == [] and from_watched(store) == []
    assert left_out(store, [row]) == {row.key}
    assert store.restore_game(watched) == [watched] and len(store.moves("g1")) == len(SANS)

    # Removed too long ago: nothing to put back, the key kept for good.
    monkeypatch.setattr(store_module, "UNDO_KEEP_S", -1.0)
    store.remove_game(watched)
    assert store.restore_game(watched) == [] and store.is_removed(watched)
    store.close()


def test_every_game_is_listed_however_many() -> None:
    """The review of 2026-10-01: the Games window listed at most 500 imports and 200 watched
    games, while the background review and Stats counted every one: a game past the caps could
    not be seen, opened or removed, nor a watched one paired with its import."""
    import dataclasses

    from sidecoach.profile import library
    from sidecoach.profile.library import find

    # Pairing looks only at the imports played within the date window: the timed one inside it.
    assert max(library.TWIN_BEFORE_S, library.TWIN_AFTER_S) <= library.DATE_WINDOW_S
    store = Store(None)
    (row,) = from_pgn(
        GAME.replace("1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 0-1", LONG), user_names=["nosegazm"]
    )
    start = row.played_at or 0.0
    # The first game played: imported and watched, older than every other.
    store.add_library_game(row)
    oldest = _watch(store, "g0", SANS, start + 60)
    # Then 500 imports and 200 watched games, days later and far apart: none pair.
    later = start + 10 * 86400
    for i in range(500):
        store.add_library_game(
            dataclasses.replace(row, key=f"chesscom:{i}", played_at=later + i * 86400)
        )
    for i in range(200):
        _watch(store, f"w{i}", SANS, later + 600 * 86400 + i * 86400)
    assert len(store.library()) == 501 and row.key in {r.key for r in store.library()}
    watched = from_watched(store)
    assert len(watched) == 201 and oldest in {r.key for r in watched}
    assert find(store, oldest) is not None, "opened"
    assert reconcile(store) == {oldest: row.key}, "paired with its import"
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

    assert store.save_review(
        older.key, "standard", 1, 100.0, json.dumps({"moves": []}), json.dumps({"accuracy": 80})
    )
    store.set_review_words(older.key, "standard", json.dumps({"1": "Book."}))
    review = store.review(older.key, "standard")
    assert (
        review is not None
        and review.version == 1
        and json.loads(review.words or "{}") == {"1": "Book."}
    )
    # Another reviewer of the same format finishing second: the review saved first stays, with
    # its words (the review of 2026-10-01: the background review wiped the words paid for).
    assert not store.save_review(older.key, "standard", 1, 150.0, "{}", None)
    assert store.review(older.key, "standard") == review
    # Reviewed again in a newer format: the words of the older review go with it.
    assert store.save_review(older.key, "standard", 2, 200.0, "{}", None)
    again = store.review(older.key, "standard")
    assert again is not None and again.version == 2 and again.words is None
    assert store.review(older.key, "deep") is None

    store.remove_game(older.key)
    assert store.library_game(older.key) is None and store.review(older.key, "standard") is None
    assert not store.save_review(older.key, "deep", 2, 300.0, "{}", None), "removed: not saved"
    assert store.review(older.key, "deep") is None
    store.close()


def test_the_end_of_game_card_offers_game_review_only_for_a_game_it_opens() -> None:
    from types import SimpleNamespace

    from sidecoach.ui.panel import PanelEngine

    store = Store(None)
    profiler = SimpleNamespace(store=store)
    game = _watch(store, "g-long", SANS, 1000.0)
    _watch(store, "g-short", SANS[:6], 2000.0)  # a puzzle's line, not a game
    assert PanelEngine._reviewable(profiler, "g-long") == game
    assert PanelEngine._reviewable(profiler, "g-short") is None
    assert PanelEngine._reviewable(profiler, "no-such-game") is None
    store.close()
