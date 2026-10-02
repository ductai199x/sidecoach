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
from sidecoach.profile.store import LibraryRow, Store

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


def test_games_kept_by_the_old_lines_are_classed_again() -> None:
    """A Lichess game kept as blitz by chess.com's lines (8+0) is rapid once the library is
    refreshed; a chess.com game keeps its archive's class; once done, not again."""
    from sidecoach.profile import library

    store = Store(None)
    lichess = '[Site "https://lichess.org/abc"]\n[TimeControl "480"]\n\n1. e4 *'
    store.add_library_game(LibraryRow("lichess:abc", "lichess", 0.0, lichess, time_class="blitz"))
    pasted = '[Site "https://lichess.org/xyz"]\n[TimeControl "-"]\n\n1. e4 *'
    store.add_library_game(LibraryRow("pgn:xyz", "pgn", 0.0, pasted))
    chesscom = '[TimeControl "480"]\n\n1. e4 *'
    store.add_library_game(LibraryRow("chesscom:1", "chesscom", 0.0, chesscom, time_class="blitz"))
    library.refresh(store, [])
    classes = {r.key: r.time_class for r in store.library()}
    assert classes == {"lichess:abc": "rapid", "pgn:xyz": "daily", "chesscom:1": "blitz"}
    store.set_time_class("lichess:abc", "blitz")
    library.refresh(store, [])
    assert store.library_game("lichess:abc").time_class == "blitz", "once"  # type: ignore[union-attr]


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
    # Lichess's own lines (its estimated time, the base plus forty increments): 8+0 is rapid,
    # 3+2 blitz, 20+0 rapid, 25+0 classical; "-" is correspondence. chess.com's 8+0 is blitz.
    assert [
        time_class_of(c, lichess=True) for c in ("15", "60", "480", "180+2", "1200", "1500", "-")
    ] == ["bullet", "bullet", "rapid", "blitz", "rapid", "classical", "daily"]
    assert (time_class_of("480"), time_class_of("-")) == ("blitz", None)
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
    # A site's games are told apart by their ids, whatever they play (a pasted one is not).
    row = dataclasses.replace(row, source="chesscom", key="chesscom:first")
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


def test_a_pasted_game_removed_stays_out_when_the_site_brings_it_in() -> None:
    """The review of 2026-10-01: a game pasted as a PGN, then removed, came back when the
    player's games were brought in from chess.com -- the same game under the site's key. It
    stays out; pasted again by hand, it comes back (that is the player asking for it)."""
    from sidecoach.profile.library import games_in, left_out, pasted_key, row_for

    store = Store(None)
    (pasted,) = from_pgn(GAME, user_names=["nosegazm"])
    fetched = row_for(
        games_in(GAME)[0],
        source="chesscom",
        key="chesscom:174300959956",
        user_names=["nosegazm"],
        time_class="rapid",
    )
    assert pasted_key(fetched) == pasted.key and pasted_key(pasted) == pasted.key
    store.add_library_game(pasted)
    store.remove_game(pasted.key)
    assert left_out(store, [fetched]) == {fetched.key}, "the site's copy stays out"
    (other,) = from_pgn(OTHER, user_names=["nosegazm"])
    assert left_out(store, [other]) == set(), "another game is not"
    store.unremove([pasted.key])  # pasted again by hand
    assert left_out(store, [fetched]) == set()
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
    assert from_watched(store)[0].time_class is None, "no clock read at the start: no class"
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


def _site(text: str, key: str, source: str = "chesscom") -> LibraryRow:
    """A game as the site's import keeps it: its own key and class."""
    from sidecoach.profile.library import games_in, row_for

    return row_for(
        games_in(text)[0], source=source, key=key, user_names=["nosegazm"], time_class="rapid"
    )


def _shown(store: Store) -> list[str]:
    """What the list, the backlog and Stats count: every game once."""
    return sorted(r.key for r in store.library())


def test_a_game_both_pasted_and_imported_is_one_game_whichever_came_first() -> None:
    """Tai, 2026-10-02: "that's obvious". The same game pasted as a PGN and brought in from the
    site is one game, the site's record (its headers); the pasted copy stands aside in the
    library, Stats and the backlog, as a watched game does for its import."""
    for first_pasted in (True, False):
        store = Store(None)
        (pasted,) = from_pgn(GAME, user_names=["nosegazm"])
        site = _site(GAME, "chesscom:1")
        order = [pasted, site] if first_pasted else [site, pasted]
        for row in order:
            store.add_library_game(row)
            reconcile(store)
        assert _shown(store) == [site.key]
        assert store.twins() == {pasted.key: site.key}
        assert [r.key for r in store.unreviewed()] == [site.key]
        # Stats: the pasted copy was counted before the import came in; once, then.
        _stats(store, pasted.key)
        assert {g["key"] for g in store.stat_games()} == {pasted.key}
        _stats(store, site.key)
        assert {g["key"] for g in store.stat_games()} == {site.key}
        # Nothing more to do on another look, and removing the game takes its copy too.
        assert reconcile(store) == {pasted.key: site.key}
        assert set(store.remove_game(site.key)) == {site.key, pasted.key}
        assert _shown(store) == [] and store.stat_games() == []
        assert set(store.restore_game(site.key)) == {site.key, pasted.key}
        assert _shown(store) == [site.key]
        store.close()


def test_three_copies_of_a_game_are_one_and_a_better_copy_takes_over() -> None:
    """Pasted, pasted again with another date, then brought in: the site's record stands for
    all three -- and the watched game that stood aside for the first now stands aside for it."""
    store = Store(None)
    (first,) = from_pgn(
        GAME.replace("1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 0-1", LONG), user_names=["nosegazm"]
    )
    again = from_pgn(
        GAME.replace("1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 0-1", LONG).replace(
            '[Date "2026.09.29"]', '[Date "2026.09.30"]'
        ),
        user_names=["nosegazm"],
    )[0]
    assert again.key != first.key
    store.add_library_game(first)
    watched = _watch(store, "g1", SANS, (first.played_at or 0.0) + 60)
    assert reconcile(store) == {watched: first.key}
    store.add_library_game(again)
    reconcile(store)
    assert _shown(store) == [first.key], "the first brought in stands for the pair"
    site = _site(GAME.replace("1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 0-1", LONG), "chesscom:7")
    store.add_library_game(site)
    assert reconcile(store) == {first.key: site.key, again.key: site.key, watched: site.key}
    assert _shown(store) == [site.key]
    assert [r.key for r in from_watched(store) if r.key not in store.twins()] == []
    store.close()


def test_games_that_only_look_alike_are_not_merged() -> None:
    """The same opening, the same players, even the same moves but a late one: two games. Two
    games of one site, and a pasted game of other players, are two as well."""
    store = Store(None)
    site = _site(GAME, "chesscom:1")
    (other_way,) = from_pgn(OTHER, user_names=["nosegazm"])  # the opening to move 3, then its own
    late_b = GAME.replace("3. Bc4 Nf6 0-1", "3. Bc4 Nf6 4. d3 Bc5 0-1")
    late_c = GAME.replace("3. Bc4 Nf6 0-1", "3. Bc4 Nf6 4. d3 Be7 0-1")
    (pasted_b,) = from_pgn(late_b, user_names=["nosegazm"])
    site_c = _site(late_c, "chesscom:3")  # differs from the pasted one by the last move only
    strangers = from_pgn(late_b.replace("Anakin_737", "Magnus"), user_names=["nosegazm"])[0]
    site_twin = _site(GAME, "chesscom:2")  # a second game of the site with the same moves
    for row in (site, other_way, pasted_b, site_c, strangers, site_twin):
        store.add_library_game(row)
    reconcile(store)
    assert _shown(store) == sorted(
        [site.key, other_way.key, pasted_b.key, site_c.key, strangers.key, site_twin.key]
    )
    assert store.twins() == {}
    store.close()


def test_the_same_game_played_again_another_day_is_another_game() -> None:
    """The audit of 2026-10-02: the same moves, the same players, the same result -- a short game
    against the same bot played again a week later -- were taken for one game pasted twice, and
    the second left the list, the background review and Stats. A copy was played when the game
    was: to the second when both say the time of day, else the same day or the next."""
    for dated in (True, False):
        store = Store(None)
        later = GAME.replace("2026.09.29", "2026.10.06")
        if not dated:  # a date alone, no time of day
            later = later.replace('[UTCTime "23:30:00"]\n', "").replace(
                '[UTCDate "2026.10.06"]\n', ""
            )
        site = _site(GAME, "chesscom:1")
        (again,) = from_pgn(later, user_names=["nosegazm"])
        for row in (site, again):
            store.add_library_game(row)
        reconcile(store)
        assert _shown(store) == sorted([site.key, again.key]), "two games"
        assert store.twins() == {}
        # The same game's copy, a date alone on the day after its UTC date (another time zone):
        # one game.
        (copy,) = from_pgn(
            GAME.replace('[UTCTime "23:30:00"]\n', "")
            .replace('[UTCDate "2026.09.29"]\n', "")
            .replace("2026.09.29", "2026.09.30"),
            user_names=["nosegazm"],
        )
        store.add_library_game(copy)
        reconcile(store)
        assert store.twins() == {copy.key: site.key}
        store.close()
