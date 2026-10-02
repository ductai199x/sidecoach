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
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
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
    # While it is analysed, the game's positions for the board to play out as the progress goes.
    playing = review_room.playback(GAME)
    assert len(playing) == 13 and playing[0]["uci"] is None and playing[1]["uci"] == "e2e4"
    if opened["state"]["status"] != "ready":
        assert opened["state"]["playback"] == playing
    state = _until_ready(client, key)["state"]
    assert state["status"] == "ready", state["error"]
    assert state["playback"] is None, "the review shown, the playback gone"
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
    # An open that names no strength reviews at the server's own: Standard on a computer.
    plain = client.post(f"/api/review/{key}/open", json={}).json()
    assert plain["ok"] and plain["state"]["strength"] == "standard"
    _until_ready(client, key)

    # A game that is not in the library opens nothing.
    assert client.post("/api/review/pgn:none/open", json={}).json()["ok"] is False
    # Taken out of the library, with its review.
    assert client.post("/api/games/remove", json={"keys": [key]}).json()["ok"]
    assert all(g["key"] != key for g in client.get("/api/games").json()["games"])


def test_the_analysis_board_plays_any_move_reads_the_engine_and_the_players_games(
    make_panel: Callable[..., PanelEngine],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Analysis (the plan's F): a move from any position checked and named, the engine's best
    lines for the board from White's side, what the player's games went on with from a
    position and how they went, and the game as a PGN with the player's side line -- annotated
    from the review, or as it was -- copied or kept in Downloads."""
    import chess
    from fastapi.testclient import TestClient

    from sidecoach.events import Line
    from sidecoach.ui import review_analysis, review_room
    from sidecoach.ui.server import build_app

    after_e4 = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
    answers = {
        after_e4: [
            Line(moves=["c7c5", "g1f3", "d7d6"], depth=14, expected_score=0.47, score_cp=30),
            Line(moves=["e7e5", "g1f3"], depth=14, expected_score=0.46, score_cp=35),
        ]
    }
    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "Engine", lambda **_: FakeEngine())
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    monkeypatch.setattr(review_analysis, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_analysis, "Engine", lambda **_: FakeEngine(answers))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    engine = make_panel()
    engine.settings.profile.chesscom_user = "nosegazm"
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    key = client.post("/api/games/import", json={"source": "pgn", "text": GAME}).json()["keys"][0]
    client.post(f"/api/review/{key}/open", json={"strength": "fast"})
    state = _until_ready(client, key)["state"]
    base = f"/api/review/{key}"

    # Any move from any position: checked, named, and the position it leaves.
    start = client.post(f"{base}/position", json={"fen": chess.STARTING_FEN}).json()
    assert start["ok"] and start["side"] == "white" and start["legal"]["e2"] == ["e3", "e4"]
    played = client.post(f"{base}/move", json={"fen": chess.STARTING_FEN, "uci": "e2e4"}).json()
    assert (played["san"], played["fen"], played["side"]) == ("e4", after_e4, "black")
    assert "c7" in played["legal"] and played["over"] is None
    wrong = client.post(f"{base}/move", json={"fen": chess.STARTING_FEN, "uci": "e2e5"}).json()
    assert not wrong["ok"] and "not legal" in wrong["error"]

    # The engine's best lines, from White's side, played out.
    lines = client.post(f"{base}/lines", json={"fen": after_e4}).json()
    assert lines["ok"] and lines["depth"] == 14 and len(lines["lines"]) == 2
    first = lines["lines"][0]
    assert first["eval"] == {"cp": -30, "mate": None}, "Black's +30 is White's -30"
    assert [s["san"] for s in first["steps"]] == ["c5", "Nf3", "d6"]

    # The player's games from a position: the moves they went on with, and how those went.
    explored = client.post(f"{base}/explore", json={"fen": chess.STARTING_FEN, "moves": []}).json()
    assert explored["ok"] and explored["games"] == 1
    assert explored["book"] is False and explored["opening"] is None, "no opening tables here"
    (e4,) = explored["moves"]
    assert (e4["san"], e4["games"], e4["won"], e4["lost"]) == ("e4", 1, 1, 0), "won as Black"
    assert e4["last"]["key"] == key
    nothing = client.post(f"{base}/explore", json={"fen": after_e4.replace(" b ", " w ")}).json()
    assert nothing["games"] == 0 and nothing["moves"] == []

    # The game as a PGN: the player's side line off the first move, the review's words on the
    # game's own moves with their clocks kept; or the moves alone.
    mainline = [m["uci"] for m in state["moves"]]

    def chain(ucis: list[str]) -> list[dict[str, Any]]:
        return [{"uci": ucis[0], "children": chain(ucis[1:])}] if ucis else []

    tree = [{"uci": "e2e4", "children": [*chain(mainline[1:]), *chain(["c7c5", "g1f3"])]}]
    kept = client.post(f"{base}/pgn", json={"tree": tree, "save": True}).json()
    assert kept["ok"] and "( 1... c5 2. Nf3 )" in kept["pgn"]
    assert "[%eval " in kept["pgn"] and "[%clk 0:09:58]" in kept["pgn"]
    assert '[Annotator "Sidecoach"]' in kept["pgn"]
    saved = Path(kept["saved"])
    assert (
        saved.parent == tmp_path / "Downloads" and saved.read_text(encoding="utf-8") == kept["pgn"]
    )
    assert saved.name == "Anakin_737 vs nosegazm 2026-09-29.pgn"
    again = client.post(f"{base}/pgn", json={"tree": tree, "save": True}).json()
    assert Path(again["saved"]).name == "Anakin_737 vs nosegazm 2026-09-29 (2).pgn"
    plain = client.post(f"{base}/pgn", json={"tree": [], "annotated": False}).json()
    assert "[%eval " not in plain["pgn"] and "Annotator" not in plain["pgn"]
    assert "saved" not in plain


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
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    key = client.post("/api/games/import", json={"source": "pgn", "text": GAME}).json()["keys"][0]
    assert client.post(f"/api/review/{key}/open", json={"strength": "fast"}).json()["voice"]

    # The live coach speaking: a card does not cut it off (voice finding 12, 2026-10-01) ...
    engine._voicing = ("post_move", 10)
    assert client.post(f"/api/review/{key}/say", json={"text": "Na5 is a mistake."}).json()["ok"]
    assert voice.heard == ["say Na5 is a mistake."]
    captioned = [text for _, text, _ in engine.poll()["state"]["caption_lines"]]
    assert "Na5 is a mistake." in captioned
    # ... and a card cuts off the card before it.
    assert client.post(f"/api/review/{key}/say", json={"text": "Bb5 was better."}).json()["ok"]
    assert voice.heard[-2:] == ["stop", "say Bb5 was better."]
    # Muted: the card is captioned, not said.
    engine.set_muted(True)
    voice.heard.clear()
    assert client.post(f"/api/review/{key}/say", json={"text": "Nf3 is best."}).json()["ok"]
    assert [h for h in voice.heard if h.startswith("say")] == []
    assert "Nf3 is best." in [text for _, text, _ in engine.poll()["state"]["caption_lines"]]
    engine.set_muted(False)
    # Hushed, and closing the review silences it too.
    assert client.post(f"/api/review/{key}/say", json={"text": "Qd2 holds."}).json()["ok"]
    assert client.post(f"/api/review/{key}/hush").json()["ok"]
    client.post(f"/api/review/{key}/close")
    assert voice.heard[-2:] == ["stop", "stop"]
    # Closing a review while the coach has the voice leaves the coach speaking.
    engine._voicing = ("post_move", 12)
    voice.heard.clear()
    client.post(f"/api/review/{key}/close")
    assert voice.heard == []


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
    client = TestClient(base_url="http://127.0.0.1", app=build_app(make_panel()))
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
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
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

    # Saved in Settings, the room sees it at once: the settings it holds follow every save
    # (they kept the launch-time copy until a restart, the parity audit 2026-10-01).
    put = client.put("/api/settings", json={"settings": {"llm": {"provider": "local"}}})
    assert put.json()["ok"]
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
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
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


# GAME two moves longer: one move misread still leaves the twelve that make a pair.
LONGER = GAME.replace("6. h3 Na5 0-1", "6. h3 Na5 7. Bb3 Nxb3 8. axb3 c6 0-1")
SANS = [
    "e4",
    "e5",
    "Nf3",
    "Nc6",
    "Bc4",
    "Nf6",
    "d3",
    "Bc5",
    "Nc3",
    "d6",
    "h3",
    "Na5",
    "Bb3",
    "Nxb3",
    "axb3",
    "c6",
]


def _watch(store: Any, game_id: str, sans: list[str], started_at: float) -> str:
    """A game as Sidecoach records one it watched (the player Black): its moves as read."""
    import chess

    store.start_game(game_id, started_at, "black", chess.STARTING_FEN)
    for ply, san in enumerate(sans, start=1):
        store.add_move(
            game_id, 0, ply, move_number=(ply + 1) // 2, san=san, uci="0000", by_user=ply % 2 == 0,
            fen_before="-", fen_after="-", phase=None, think_ms=None, clock_ms=None,
            capture=False, gives_check=False, material=None, ts=started_at + ply,
        )  # fmt: skip
    store.end_game(game_id, started_at + 900, "0-1", "resignation")
    return f"watched:{game_id}"


def _watched_and_imported(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch, sans: list[str]
) -> tuple[Any, PanelEngine, Any, str, str, list[FakeEngine]]:
    """LONGER brought in, and Sidecoach's record of it (`sans` as read) reviewed at Standard
    before the two were paired. Returns the client, the panel, the store, the import's and the
    watched game's keys, and the fake engines made."""
    from fastapi.testclient import TestClient

    from sidecoach.analysis.game_review import STRENGTHS
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
    engine = make_panel()
    engine.settings.profile.chesscom_user = "NoseGazm"
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    key = client.post("/api/games/import", json={"source": "pgn", "text": LONGER}).json()["keys"][0]
    store = Store(engine.settings.data_dir / "profile.sqlite")
    row = library.find(store, key)
    assert row is not None and row.played_at is not None
    watched = _watch(store, "g1", sans, row.played_at + 3600)
    (wrow,) = [r for r in library.from_watched(store) if r.key == watched]
    assert review_room.review_row(engine.settings, wrow, store, STRENGTHS["standard"], workers=1)
    return client, engine, store, key, watched, made


def test_a_watched_games_review_goes_with_the_import_that_stands_for_it(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tai, 2026-09-30: a watched game reviewed, then brought in from chess.com, lost its
    accuracy -- the import stood for it unreviewed. Sidecoach read every move, so the import
    takes the watched review whole the moment the two are paired, no engine: its own headers
    and clocks, its summary, its words and its Stats facts."""
    client, _engine, store, key, watched, made = _watched_and_imported(
        make_panel, monkeypatch, SANS
    )
    old = store.review(watched, "standard")
    store.set_review_words(watched, "standard", json.dumps({"texts": {"4": {"why": "A move."}}}))
    engines = len(made)

    listed = {g["key"]: g for g in client.get("/api/games").json()["games"]}
    assert watched not in listed, "the import stands for the watched game"
    game = listed[key]
    assert game["reviewed"] and game["accuracy"]["black"] is not None
    assert len(made) == engines, "carried, not searched"

    carried = store.review(key, "standard")
    assert old is not None and carried is not None
    report, was = json.loads(carried.report), json.loads(old.report)
    assert report["headers"]["BlackElo"] == "1018", "the import's headers"
    assert [m["clock_ms"] for m in report["moves"][:2]] == [598_000, 595_000], "and clocks"
    assert [m["clock_ms"] for m in was["moves"][:2]] == [None, None]
    assert [m["assessment"] for m in report["moves"]] == [
        {**m["assessment"], "game_id": key} for m in was["moves"]
    ], "the watched review's measurements"
    assert json.loads(carried.words or "{}") == {"texts": {"4": {"why": "A move."}}}
    assert {g["key"] for g in store.stat_games()} == {key}, "Stats has it once, as the import"
    store.close()


def test_an_import_whose_record_differs_is_reviewed_from_the_watched_positions(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sidecoach misread a move (4. d3 as d4): still one game, but not the same positions from
    there on, so the import is reviewed as itself -- and the positions the watched review
    measured, up to the misread move, are not searched again. Until it is, the game keeps the
    accuracy its watched review gave it in the list, and its place in Stats (the review of
    2026-10-01: both went the moment the import came in, for good on the phone)."""
    from sidecoach.analysis.game_review import STRENGTHS, positions
    from sidecoach.profile import library
    from sidecoach.ui import review_room

    misread = [*SANS[:6], "d4", *SANS[7:]]
    client, engine, store, key, watched, made = _watched_and_imported(
        make_panel, monkeypatch, misread
    )
    listed = {g["key"]: g for g in client.get("/api/games").json()["games"]}
    assert watched not in listed and not listed[key]["reviewed"], "nothing to carry whole"
    old = store.review(watched, "standard")
    assert old is not None
    summary = json.loads(old.summary or "{}")
    assert listed[key]["accuracy"] == {s: summary[s]["accuracy"] for s in ("white", "black")}
    assert {g["key"] for g in store.stat_games()} == {watched}, "counted, as the watched game"

    was = json.loads(old.report)
    measured = {was["start"]["fen"], *(m["after"]["fen"] for m in was["moves"])}
    row = library.find(store, key)
    assert row is not None
    made.clear()
    assert review_room.review_row(engine.settings, row, store, STRENGTHS["standard"], workers=1)
    searched = {fen for e in made for fen in e.calls}
    own = [b.fen() for b in positions(library.games_in(row.pgn)[0])]
    assert all(f in measured for f in own[:7]), "the positions before the misread move agree"
    assert not searched & measured, "nothing the watched review measured is searched again"
    assert set(own[7:]) <= searched, "the import's own positions from the misread move on are"
    assert {g["key"] for g in store.stat_games()} == {key}, "Stats has it once, as the import"
    assert [r.key for r in store.reviews_without_stats()] == []
    store.close()


def test_a_removed_game_stays_out_until_undone_or_pasted_again(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tai, 2026-10-01: any game can be removed from the player's record. Gone from the list,
    left out when the site's games are brought in again, back with Undo; a game pasted by hand
    comes back (that is the player asking for it); Undo too late says so."""
    from fastapi.testclient import TestClient

    from sidecoach.profile import library
    from sidecoach.profile import store as store_module
    from sidecoach.ui.server import build_app

    engine = make_panel()
    engine.settings.profile.chesscom_user = "NoseGazm"
    monkeypatch.setattr(
        library, "from_chesscom", lambda user, months=3: library.from_pgn(GAME, user_names=[user])
    )
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    keys = lambda: {g["key"] for g in client.get("/api/games").json()["games"]}  # noqa: E731
    first = client.post("/api/games/import", json={"source": "chesscom"}).json()
    (key,) = first["keys"]
    assert first["added"] == 1 and key in keys()

    # Two games selected and removed in one go, the second brought in by hand.
    other = client.post("/api/games/import", json={"source": "pgn", "text": LONGER}).json()
    (second,) = other["keys"]
    both = {"keys": [key, second]}
    assert client.post("/api/games/remove", json=both).json() == {
        "ok": True,
        "removed": [key, second],
    }
    assert not {key, second} & keys()
    again = client.post("/api/games/import", json={"source": "chesscom"}).json()
    assert (again["added"], again["already"], again["left_out"]) == (0, 0, 1)
    assert key not in keys(), "brought in again, it stays out"

    restored = client.post("/api/games/restore", json=both).json()
    assert restored == {"ok": True, "restored": [key, second]} and {key, second} <= keys()

    client.post("/api/games/remove", json={"keys": [key]})
    pasted = client.post("/api/games/import", json={"source": "pgn", "text": GAME}).json()
    assert (pasted["added"], pasted["left_out"]) == (1, 0) and key in keys(), "pasted, it is back"

    client.post("/api/games/remove", json={"keys": [key]})
    monkeypatch.setattr(store_module, "UNDO_KEEP_S", -1.0)
    late = client.post("/api/games/restore", json={"keys": [key]}).json()
    assert not late["ok"] and "too long ago" in late["error"]


def test_a_review_under_way_saves_nothing_for_a_game_removed_meanwhile(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The background reviewer (or a review window) part way through a game the player removes:
    it saves no review, no Stats, nothing kept part way -- the game stays gone."""
    from fastapi.testclient import TestClient

    from sidecoach.analysis.game_review import STRENGTHS
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room
    from sidecoach.ui.server import build_app

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "Engine", lambda **_: FakeEngine())
    engine = make_panel()
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    key = client.post("/api/games/import", json={"source": "pgn", "text": GAME}).json()["keys"][0]
    store = Store(engine.settings.data_dir / "profile.sqlite")
    row = library.find(store, key)
    assert row is not None

    def progress(done: int, _total: int) -> None:
        if done == 3:
            store.remove_game(key)

    done = review_room.review_row(
        engine.settings, row, store, STRENGTHS["fast"], workers=1, on_progress=progress
    )
    assert done is None
    assert store.review(key, "fast") is None and store.stat_games() == []
    assert store.progress(key, "fast", 14, 3) == [] and store.is_removed(key)
    store.close()


def test_the_review_saved_first_stays_with_the_words_paid_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review of 2026-10-01: the background review and a review window on one game at once.
    The window saved first and the player paid for the AI coach's words; the background then
    saved its own review over it, and the words with it. The review saved first stays, words and
    all, and the one that finished second is handed it."""
    from sidecoach.analysis.game_review import STRENGTHS
    from sidecoach.config import Settings
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    settings = Settings(data_dir=tmp_path)
    store = Store(tmp_path / "profile.sqlite")
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    standard = STRENGTHS["standard"]

    def engines(**_: Any) -> FakeEngine:
        return FakeEngine()

    assert review_room.review_row(settings, row, store, standard, workers=1, make_engine=engines)
    words = {"model": "luna", "cost": 0.03, "texts": {"4": {"why": "Paid for."}}, "reveals": []}
    store.set_review_words(row.key, "standard", json.dumps(words))
    saved = store.review(row.key, "standard")
    # The background review, begun before the window saved, finishes now.
    second = review_room.review_row(settings, row, store, standard, workers=1, make_engine=engines)
    assert store.review(row.key, "standard") == saved, "the review saved first, and its words"
    assert saved is not None and json.loads(saved.words or "{}") == words
    assert second is not None and second[0].model_dump_json() == saved.report, "handed the saved"
    store.close()


def test_the_ai_words_are_kept_when_the_review_closes_while_they_are_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review of 2026-10-01: the window closed (the page sends /close as it unloads) while the
    AI coach's words were written closed the room's store, and every answer that came after --
    each one paid for -- was thrown away. The room stays open until the words are saved; opened
    again meanwhile, it is the same room, still writing (nothing asked twice), and it stays."""
    import threading

    from sidecoach.coach.review_words import ReviewWords
    from sidecoach.config import Settings
    from sidecoach.llm import factory
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    go = threading.Event()
    asked: list[int] = []

    class Model:
        model = "gpt-x"

    def write(game: Any, model: Any, **kw: Any) -> ReviewWords:
        asked.append(1)
        go.wait(10)  # each answer arrives, and is paid for, after the window closed
        paid = {"prompt_tokens": 1, "completion_tokens": 1}
        for ply in (2, 4):
            kw["on_answer"](ply, f"### why\nMove {ply}.", paid)
        return ReviewWords(texts={2: {"why": "a"}, 4: {"why": "b"}}, model="gpt-x", cost=0.02)

    monkeypatch.setattr(factory, "make_provider", lambda llm, timeout_s=None: Model())
    monkeypatch.setattr(review_room.review_words, "write", write)
    settings = Settings(data_dir=tmp_path)
    settings.llm.provider = "local"
    path = tmp_path / "profile.sqlite"
    store = Store(path)
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    rooms = review_room.ReviewRooms(settings, store_path=path, make_engine=lambda **_: FakeEngine())

    def until(check: Callable[[], bool]) -> bool:
        deadline = time.monotonic() + 10
        while not check() and time.monotonic() < deadline:
            time.sleep(0.02)
        return check()

    room = rooms.open(row.key, "fast")
    assert room is not None and until(lambda: room.status == "ready")
    assert room.write_words() is None
    room.close()
    time.sleep(review_room.TICK_S * 3)
    assert rooms.get(row.key) is room and not room.closed, "open while its words are written"
    assert rooms.open(row.key, "fast") is room, "opened again: the same room, still writing"
    assert room.state()["ai"]["status"] == "writing"
    go.set()
    assert until(lambda: room.words_status == "ready")
    saved = store.review(row.key, "fast")
    assert saved is not None and json.loads(saved.words or "{}")["texts"] == {
        "2": {"why": "a"},
        "4": {"why": "b"},
    }, "saved whole"
    assert asked == [1], "asked once"
    time.sleep(review_room.TICK_S * 3)
    assert rooms.get(row.key) is room, "opened again, it stays open"

    room.close()
    assert until(lambda: room.closed), "closed with nothing being written: closed at once"

    # Closed while writing, not opened again: open until the words are saved, then closed.
    go.clear()
    store.set_review_words(row.key, "fast", "")
    again = rooms.open(row.key, "fast")
    assert again is not None and again is not room and until(lambda: again.status == "ready")
    assert again.write_words() is None
    again.close()
    time.sleep(review_room.TICK_S * 3)
    go.set()
    assert until(lambda: again.closed), "closed once its words are saved"
    saved = store.review(row.key, "fast")
    assert saved is not None and "partial" not in json.loads(saved.words or "{}")
    store.close()


def test_a_deep_review_stopped_part_way_keeps_its_positions_through_standard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The review of 2026-10-01: a Deep review stopped part way lost every position it had
    measured the moment the game's Standard review was saved (the background review) or opened
    -- what a review keeps part way went for every strength. Only its own strength's goes."""
    import threading

    from sidecoach.analysis.game_review import STRENGTHS
    from sidecoach.config import Settings
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    settings = Settings(data_dir=tmp_path)
    path = tmp_path / "profile.sqlite"
    store = Store(path)
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    deep, standard = STRENGTHS["deep"], STRENGTHS["standard"]
    halt = threading.Event()

    def progress(done: int, _total: int) -> None:
        if done >= 4:
            halt.set()

    def engines(**_: Any) -> FakeEngine:
        return FakeEngine()

    assert (
        review_room.review_row(
            settings,
            row,
            store,
            deep,
            workers=1,
            make_engine=engines,
            on_progress=progress,
            stop=halt,
        )
        is None
    )

    def deep_kept() -> list[str]:
        return sorted(store.progress(row.key, "deep", deep.depth, deep.multipv))

    kept = deep_kept()
    assert len(kept) >= 4
    assert review_room.review_row(settings, row, store, standard, workers=1, make_engine=engines)
    assert deep_kept() == kept, "Standard saved"
    room = review_room.ReviewRoom(
        settings, row, store_path=path, strength="standard", make_engine=engines
    )
    deadline = time.monotonic() + 10
    while room.status != "ready" and time.monotonic() < deadline:
        time.sleep(0.02)
    room.close()
    room.join()
    assert room.status == "ready"
    assert deep_kept() == kept, "Standard opened"
    assert store.progress(row.key, "standard", standard.depth, standard.multipv) == []
    store.close()


def test_bringing_in_from_a_site_needs_the_username_first(
    make_panel: Callable[..., PanelEngine],
) -> None:
    from fastapi.testclient import TestClient

    from sidecoach.ui.server import build_app

    client = TestClient(base_url="http://127.0.0.1", app=build_app(make_panel()))
    said = client.post("/api/games/import", json={"source": "chesscom"}).json()
    assert not said["ok"] and "chess.com username" in said["error"]
    said = client.post("/api/games/import", json={"source": "lichess"}).json()
    assert not said["ok"] and "Lichess username" in said["error"]
