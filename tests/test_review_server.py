"""Game Review over the panel's API (ui/server.py, ui/review_room.py): a game brought into the
library, listed, reviewed in a room of its own with its progress, then opened again from the
review saved for it."""

from __future__ import annotations

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
    assert added["ok"] and added["added"] == 1
    key = added["keys"][0]
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
    assert state["progress"]["done"] == state["progress"]["total"] == 13

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
