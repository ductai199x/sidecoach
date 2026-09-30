"""Game Review over the panel's API (ui/server.py, ui/review_room.py): a game brought into the
library, listed, reviewed in a room of its own with its progress, then opened again from the
review saved for it."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sidecoach.ui.panel import PanelEngine
from tests.fakes import FakeEngine

GAME = """[Event "Live Chess"]
[Site "Chess.com"]
[Date "2026.09.29"]
[White "Anakin_737"]
[Black "nosegazm"]
[Result "0-1"]
[WhiteElo "880"]
[BlackElo "1018"]
[TimeControl "600"]

1. e4 {[%clk 0:09:58]} e5 {[%clk 0:09:55]} 2. Nf3 {[%clk 0:09:50]} Nc6 {[%clk 0:09:40]}
3. Bc4 {[%clk 0:09:45]} Nf6 {[%clk 0:09:30]} 4. d3 Bc5 5. Nc3 d6 6. h3 Na5 0-1
"""


def _until_ready(client: Any, key: str, seconds: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + seconds
    polled: dict[str, Any] = {}
    while time.monotonic() < deadline:
        polled = client.get(f"/api/review/{key}/poll").json()
        if polled.get("state", {}).get("status") in ("ready", "failed"):
            return polled
        time.sleep(0.05)
    state = polled.get("state", {})
    lines = [line.get("text") for line in polled.get("lines", [])]
    raise AssertionError(
        f"the review never finished: {state.get('status')} {state.get('progress')} {lines[-3:]}"
    )


def test_a_game_is_brought_in_reviewed_and_reopened_from_its_saved_review(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from sidecoach.ui import review_room
    from sidecoach.ui.server import build_app

    made: list[FakeEngine] = []

    def engine_for(**_: Any) -> FakeEngine:
        made.append(FakeEngine())
        return made[-1]

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "Engine", engine_for)
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 2)
    engine = make_panel()
    engine.settings.profile.chesscom_user = "NoseGazm"
    client = TestClient(build_app(engine))
    assert client.get("/games").status_code == 200, "the library's page"

    # Brought in from a pasted PGN, and listed, not yet reviewed.
    added = client.post("/api/games/import", json={"source": "pgn", "text": GAME}).json()
    assert added["ok"] and (added["added"], added["already"]) == (1, 0)
    key = added["keys"][0]
    # The same game again: nothing new, and said so.
    again = client.post("/api/games/import", json={"source": "pgn", "text": GAME}).json()
    assert (again["added"], again["already"]) == (0, 1)
    listed = client.get("/api/games").json()
    assert listed["users"]["chesscom"] == "NoseGazm"
    row = next(g for g in listed["games"] if g["key"] == key)
    assert (row["white"], row["black"], row["user_side"], row["reviewed"]) == (
        "Anakin_737",
        "nosegazm",
        "black",
        False,
    )
    assert client.get(f"/review/{key}").status_code == 200, "the review's page"

    # Reviewed in a room of its own: two engines side by side, then the summary and the moves.
    opened = client.post(f"/api/review/{key}/open", json={"strength": "fast"}).json()
    assert opened["ok"] and opened["state"]["strength"] == "fast"
    state = _until_ready(client, key)["state"]
    assert state["status"] == "ready", state["error"]
    assert len(made) == 2
    summary, moves = state["summary"], state["moves"]
    assert (summary["white"]["name"], summary["black"]["rating"]) == ("Anakin_737", 1018)
    assert len(moves) == 12 and len(summary["graph"]) == 13
    first = moves[0]
    assert (first["san"], first["side"], first["from"], first["to"]) == ("e4", "white", "e2", "e4")
    assert first["clock_ms"] == 598_000 and first["spent_ms"] == 2_000
    assert set(first["eval"]) == {"cp", "mate", "result"}
    # The lines its card plays on the board: the best from before it, what follows from after.
    steps = first["best_line"] + first["follow_line"]
    assert steps and all(set(s) == {"san", "uci", "fen"} for s in steps)
    # Six moves a side say too little to name the rating they were played at.
    assert summary["white"]["performance"] is None
    assert state["progress"]["done"] == state["progress"]["total"] == 13
    # Its facts kept for Stats as it is saved.
    from sidecoach.profile.store import Store

    kept = Store(engine.settings.data_dir / "profile.sqlite")
    assert [g["key"] for g in kept.stat_games()] == [key] and len(kept.stat_moves([key])) == 12
    kept.close()
    # Added up in Stats (the player was Black): six moves of theirs.
    assert client.get("/stats").status_code == 200, "the Stats window's page"
    numbers = client.get("/api/stats").json()["stats"]
    assert (numbers["overview"]["games"], numbers["overview"]["moves"]) == (1, 6)
    assert (
        client.get("/api/stats", params={"opponents": "bots"}).json()["stats"]["overview"]["games"]
        == 0
    )
    listed = client.get("/api/stats/positions", params={"what": "gallery", "arg": "best"}).json()
    assert listed["ok"] and all(p["side"] == "black" for p in listed["positions"])

    # Listed with its accuracies now; opened again, the saved review is read, not redone.
    row = next(g for g in client.get("/api/games").json()["games"] if g["key"] == key)
    assert row["reviewed"] and set(row["accuracy"]) == {"white", "black"}
    assert client.post(f"/api/review/{key}/close").json()["ok"]
    time.sleep(0.5)
    made.clear()
    again = (
        _until_ready(client, key)
        if client.post(f"/api/review/{key}/open", json={"strength": "fast"}).json()["ok"]
        else {}
    )
    assert again["state"]["moves"] == moves and not made, "no engine for a saved review"

    # A game that is not in the library opens nothing.
    assert client.post("/api/review/pgn:none/open", json={}).json()["ok"] is False
    # Taken out of the library, with its review.
    assert client.delete(f"/api/games/{key}").json()["ok"]
    assert all(g["key"] != key for g in client.get("/api/games").json()["games"])


class RecordingVoice:
    """The panel's voice, keeping what it was told."""

    name = "recording"
    busy = False

    def __init__(self) -> None:
        self.heard: list[str] = []

    def healthcheck(self) -> tuple[bool, str]:
        return True, "recording"

    def say(self, text: str) -> None:
        self.heard.append(f"say {text}")

    def stop(self) -> None:
        self.heard.append("stop")


def test_a_card_is_said_in_the_panels_voice_and_captioned(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from sidecoach.ui import review_room
    from sidecoach.ui.server import build_app

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "Engine", lambda **_: FakeEngine())
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    engine = make_panel()
    voice = RecordingVoice()
    engine._tts = voice  # type: ignore[assignment]
    client = TestClient(build_app(engine))
    key = client.post("/api/games/import", json={"source": "pgn", "text": GAME}).json()["keys"][0]
    assert client.post(f"/api/review/{key}/open", json={"strength": "fast"}).json()["voice"]

    # A card cuts off the one before it, is said, and is captioned as the panel's words are.
    assert client.post(f"/api/review/{key}/say", json={"text": "Na5 is a mistake."}).json()["ok"]
    assert voice.heard == ["stop", "say Na5 is a mistake."]
    captioned = [text for _, text, _ in engine.poll()["state"]["caption_lines"]]
    assert "Na5 is a mistake." in captioned
    # Hushed, and closing the review silences it too.
    assert client.post(f"/api/review/{key}/hush").json()["ok"]
    client.post(f"/api/review/{key}/close")
    assert voice.heard[-2:] == ["stop", "stop"]


def test_a_move_is_retried_and_graded_as_the_review_graded_it(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from sidecoach.ui import review_retry, review_room
    from sidecoach.ui.server import build_app

    for module in (review_room, review_retry):
        monkeypatch.setattr(module, "find_stockfish", lambda _settings: Path("stockfish"))
    made: list[FakeEngine] = []

    def engine_for(**_: Any) -> FakeEngine:
        made.append(FakeEngine())
        return made[-1]

    for module in (review_room, review_retry):
        monkeypatch.setattr(module, "Engine", engine_for)
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    client = TestClient(build_app(make_panel()))
    key = client.post("/api/games/import", json={"source": "pgn", "text": GAME}).json()["keys"][0]
    client.post(f"/api/review/{key}/open", json={"strength": "fast"})
    moves = _until_ready(client, key)["state"]["moves"]
    reviewing = len(made)

    # The position Black's 3...Nf6 was played from, with every legal move.
    position = client.get(f"/api/review/{key}/retry", params={"ply": 6}).json()
    assert position["ok"] and position["side"] == "black"
    assert position["fen"] == moves[5]["fen_before"] and "f6" in position["legal"]["g8"]
    # The game's own move again: the review's verdict, searched for nothing new.
    again = client.post(f"/api/review/{key}/try", json={"ply": 6, "uci": "g8f6"}).json()
    played = moves[5]
    assert (again["san"], again["label"], again["loss"]) == ("Nf6", played["label"], played["loss"])
    assert made[reviewing:] and not made[-1].calls, "the game's positions are the review's"
    # Another move: graded, the position it leads to searched.
    other = client.post(f"/api/review/{key}/try", json={"ply": 6, "uci": "d7d6"}).json()
    assert other["ok"] and other["san"] == "d6" and other["label"] and made[-1].calls
    assert not client.post(f"/api/review/{key}/try", json={"ply": 6, "uci": "e1e2"}).json()["ok"]
    # The coach's hints, the move only at the third.
    hint_path = f"/api/review/{key}/hint"
    hints = [client.post(hint_path, json={"ply": 6, "level": n}).json() for n in (1, 2, 3)]
    assert all(h["ok"] and h["text"] for h in hints)
    assert [h["best"] is not None for h in hints] == [False, False, True]


def review_room_game(engine: PanelEngine, key: str) -> Any:
    """The review saved for a game, as the room reads it."""
    from sidecoach.events import ReviewedGame
    from sidecoach.profile.store import Store

    store = Store(engine.settings.data_dir / "profile.sqlite")
    try:
        saved = store.latest_review(key)
        assert saved is not None
        return ReviewedGame.model_validate_json(saved.report)
    finally:
        store.close()


def test_the_ai_coachs_words_are_written_only_when_asked_and_kept(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No AI coach set up: the page is told why. With one, asked: the words are written in the
    background, from the player's side and without reasoning for a local model, counted in the
    panel's strip, and opened again with the saved review."""
    from fastapi.testclient import TestClient

    from sidecoach.coach.review_words import ReviewWords
    from sidecoach.llm import factory
    from sidecoach.llm.usage import Usage
    from sidecoach.ui import review_room
    from sidecoach.ui.server import build_app

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "Engine", lambda **_: FakeEngine())
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)

    class Model:
        model = "local-model"
        usage = Usage()

    asked: list[dict[str, Any]] = []

    def write(game: Any, model: Any, **told: Any) -> ReviewWords:
        asked.append(told)
        return ReviewWords(
            texts={4: {"why": "Your knight on c6 guards your pawn on e5."}},
            reveals={4},
            model=model.model,
            cost=0.0,
        )

    monkeypatch.setattr(factory, "make_provider", lambda llm, timeout_s=None: Model())
    monkeypatch.setattr(review_room.review_words, "write", write)
    engine = make_panel()
    engine.settings.profile.chesscom_user = "NoseGazm"
    engine.settings.llm.provider = "none"
    client = TestClient(build_app(engine))
    key = client.post("/api/games/import", json={"source": "pgn", "text": GAME}).json()["keys"][0]
    client.post(f"/api/review/{key}/open", json={"strength": "fast"})
    ai = _until_ready(client, key)["state"]["ai"]
    assert ai == {
        "status": "none",
        "error": None,
        "words": None,
        "unavailable": review_room.NO_AI,
        "moves": len(review_room.review_words.key_plies(review_room_game(engine, key))),
        "answered": 0,
        "estimate": None,
    }
    refused = client.post(f"/api/review/{key}/ai").json()
    assert refused == {"ok": False, "error": review_room.NO_AI} and not asked

    engine.settings.llm.provider = "local"
    assert client.post(f"/api/review/{key}/ai").json() == {"ok": True, "error": None}
    deadline = time.monotonic() + 10
    while ai["status"] != "ready" and time.monotonic() < deadline:
        ai = client.get(f"/api/review/{key}/poll").json()["state"]["ai"]
        time.sleep(0.05)
    why = "Your knight on c6 guards your pawn on e5."
    words = {
        "model": "local-model",
        "cost": 0.0,
        "texts": {"4": {"why": why}},
        "reveals": [4],
        # Shown with its links to the board (coach/review_links.py).
        "links": {
            "4": {
                "why": [
                    {"text": "Your knight on c6", "kind": "piece", "squares": ["c6"]},
                    " guards ",
                    {"text": "your pawn on e5", "kind": "piece", "squares": ["e5"]},
                    ".",
                ]
            }
        },
    }
    # A model on this computer: no prices, so no cost to say.
    assert (ai["status"], ai["words"], ai["unavailable"], ai["estimate"]) == (
        "ready",
        words,
        None,
        None,
    )
    assert [(a["user"], a["thinking"]) for a in asked] == [("black", False)]
    assert any(Model.usage in source() for source in engine.extra_usage)

    client.post(f"/api/review/{key}/close")
    time.sleep(0.5)
    client.post(f"/api/review/{key}/open", json={"strength": "fast"})
    assert _until_ready(client, key)["state"]["ai"]["words"] == words
    assert len(asked) == 1, "kept, not written again"


def test_a_review_the_app_was_closed_in_goes_on_from_where_it_was(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tai, 2026-09-30: a review must not break if the app is closed at any moment. The review
    stopped part way kept every position measured; opened again it measures only the rest, and
    what it kept goes once it is saved. Words the app was closed in the middle of keep the
    answers paid for: the page says how far they got, and only the rest are asked."""
    import threading

    from fastapi.testclient import TestClient

    from sidecoach.analysis.game_review import STRENGTHS
    from sidecoach.coach.review_words import ReviewWords
    from sidecoach.llm import factory
    from sidecoach.llm.usage import Usage
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room
    from sidecoach.ui.server import build_app

    made: list[FakeEngine] = []

    def engine_for(**_: Any) -> FakeEngine:
        made.append(FakeEngine())
        return made[-1]

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "Engine", engine_for)
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    engine = make_panel()
    engine.settings.profile.chesscom_user = "NoseGazm"
    client = TestClient(build_app(engine))
    key = client.post("/api/games/import", json={"source": "pgn", "text": GAME}).json()["keys"][0]
    fast = STRENGTHS["fast"]

    # Reviewed until five positions were measured, then the app closed.
    store = Store(engine.settings.data_dir / "profile.sqlite")
    row = library.find(store, key)
    assert row is not None
    halt = threading.Event()

    def progress(done: int, _total: int) -> None:
        if done >= 5:
            halt.set()

    stopped = review_room.review_row(
        engine.settings, row, store, fast, workers=1, on_progress=progress, stop=halt
    )
    assert stopped is None and store.review(key, "fast") is None
    kept = {review_room.measured_from_json(t).fen for t in store.progress(key, "fast", 14, 3)}
    assert 5 <= len(kept) < 13
    store.close()

    # Opened again: only the rest measured, the review saved, what was kept gone.
    made.clear()
    client.post(f"/api/review/{key}/open", json={"strength": "fast"})
    state = _until_ready(client, key)["state"]
    assert state["status"] == "ready" and len(state["moves"]) == 12
    searched = {fen for e in made for fen in e.calls}
    assert searched and not searched & kept, "nothing measured twice"
    store = Store(engine.settings.data_dir / "profile.sqlite")
    assert store.progress(key, "fast", 14, 3) == [] and store.review(key, "fast") is not None

    # The words: two answers paid for when the app closed (the fake engine finds no mistakes,
    # so the key moves are given).
    plies = [2, 4, 6]
    monkeypatch.setattr(review_room.review_words, "key_plies", lambda _game: plies)
    first, second = plies[:2]
    partial = {
        "partial": True,
        "model": "gpt-x",
        "cost": 0.004,
        "answers": {str(first): "### why\nWhite's pawn.", str(second): "### why\nBlack's pawn."},
    }
    store.set_review_words(key, "fast", json.dumps(partial))
    store.close()
    client.post(f"/api/review/{key}/close")
    time.sleep(0.5)
    client.post(f"/api/review/{key}/open", json={"strength": "fast"})
    ai = _until_ready(client, key)["state"]["ai"]
    assert (ai["status"], ai["answered"], ai["words"]) == ("none", 2, None)

    class Model:
        model = "gpt-x"
        usage = Usage()

    told: list[dict[str, Any]] = []

    def write(game: Any, model: Any, **kw: Any) -> ReviewWords:
        told.append(kw)
        rest = [p for p in plies if p not in kw["answered"]]
        for p in rest:
            kw["on_answer"](p, "### why\nA move.", {"prompt_tokens": 10, "completion_tokens": 10})
        return ReviewWords(texts={p: {"why": "A move."} for p in plies}, model="gpt-x", cost=0.002)

    monkeypatch.setattr(factory, "make_provider", lambda llm, timeout_s=None: Model())
    monkeypatch.setattr(review_room.review_words, "write", write)
    engine.settings.llm.provider = "local"
    assert client.post(f"/api/review/{key}/ai").json()["ok"]
    deadline = time.monotonic() + 10
    while ai["status"] != "ready" and time.monotonic() < deadline:
        ai = client.get(f"/api/review/{key}/poll").json()["state"]["ai"]
        time.sleep(0.05)
    assert ai["status"] == "ready" and ai["answered"] == 0
    assert set(told[0]["answered"]) == {first, second}, "the answers paid for, not asked again"
    assert abs(ai["words"]["cost"] - 0.006) < 1e-9, "what was paid before counted in"


def test_bringing_in_from_a_site_needs_the_username_first(
    make_panel: Callable[..., PanelEngine],
) -> None:
    from fastapi.testclient import TestClient

    from sidecoach.ui.server import build_app

    client = TestClient(build_app(make_panel()))
    said = client.post("/api/games/import", json={"source": "chesscom"}).json()
    assert not said["ok"] and "chess.com username" in said["error"]
    said = client.post("/api/games/import", json={"source": "lichess"}).json()
    assert not said["ok"] and "Lichess username" in said["error"]
