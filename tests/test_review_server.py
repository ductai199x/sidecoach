"""Game Review over the panel's API (ui/server.py, ui/review_room.py): a game brought into the
library, listed, reviewed in a room of its own with its progress, then opened again from the
review saved for it."""

from __future__ import annotations

import json
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


# How long an event may take before the test calls it lost: a bound on a hang, never what a
# test asserts.
HANG_S = 60.0


def _room(client: Any, key: str) -> Any:
    room = client.app.state.reviews.get(key)
    assert room is not None, f"no room open for {key}"
    return room


def _until_ready(client: Any, key: str) -> dict[str, Any]:
    """The room's state once its review is ready or failed: waited on, not polled."""
    assert _room(client, key).settled.wait(HANG_S), "the review never settled"
    return client.get(f"/api/review/{key}/poll").json()


def _until_written(client: Any, key: str) -> dict[str, Any]:
    """The room's AI words once they are no longer being written: waited on, not polled."""
    assert _room(client, key).words_done.wait(HANG_S), "the words were never done"
    return client.get(f"/api/review/{key}/poll").json()["state"]["ai"]


def _close(client: Any, key: str) -> None:
    """The room closed, and its thread ended."""
    room = _room(client, key)
    assert client.post(f"/api/review/{key}/close").json()["ok"]
    room.join(HANG_S)
    assert room.closed


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
    _close(client, key)
    made.clear()
    again = (
        _until_ready(client, key)
        if client.post(f"/api/review/{key}/open", json={"strength": "fast"}).json()["ok"]
        else {}
    )
    assert again["state"]["moves"] == moves and not made, "no engine for a saved review"
    # An open that names no strength opens at the strongest review saved: it opened at Standard
    # always, and a game reviewed Deep was reviewed again, its Stats overwritten (the review of
    # 2026-10-01).
    plain = client.post(f"/api/review/{key}/open", json={}).json()
    assert plain["ok"] and plain["state"]["strength"] == "fast" and not made
    _until_ready(client, key)
    client.post(f"/api/review/{key}/open", json={"strength": "standard"})
    assert _until_ready(client, key)["state"]["strength"] == "standard"
    # A Fast look after it lowers nothing: Stats and Your games keep the strongest.
    store = Store(engine.settings.data_dir / "profile.sqlite")
    store.save_stats("fast", {**store.stat_games()[0], "strength": "fast"}, [])
    assert [(g["strength"], len(store.stat_moves([key]))) for g in store.stat_games()] == [
        ("standard", 12)
    ]
    assert store.best_review(key).strength == "standard"  # type: ignore[union-attr]
    store.close()
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

    def say(self, text: str, *, voice: str | None = None, owner: str = "coach") -> None:
        self.heard.append(f"say {text}" if owner == "coach" else f"say {text} ({owner})")

    def stop(self, owner: str | None = None) -> None:
        self.heard.append("stop" if owner is None else f"stop ({owner})")


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
    # The room's own thread (the review on the fake engine, its saves to the store) done
    # before the cards: nothing of it runs beside the steps below.
    room = _room(client, key)
    assert room.settled.wait(HANG_S), "the review never settled"

    # The live coach speaking: a card does not cut it off (voice finding 12, 2026-10-01): the
    # stop before it is the review's own lines' (round 3 audit, E5) ...
    engine._voicing = ("post_move", 10)
    assert client.post(f"/api/review/{key}/say", json={"text": "Na5 is a mistake."}).json()["ok"]
    assert voice.heard == ["stop (review)", "say Na5 is a mistake. (review)"]
    captioned = [text for _, text, _ in engine.poll()["state"]["caption_lines"]]
    assert "Na5 is a mistake." in captioned
    # ... and a card cuts off the card before it.
    assert client.post(f"/api/review/{key}/say", json={"text": "Bb5 was better."}).json()["ok"]
    assert voice.heard[-2:] == ["stop (review)", "say Bb5 was better. (review)"]
    # Muted: the card is captioned, not said.
    engine.set_muted(True)
    voice.heard.clear()
    assert client.post(f"/api/review/{key}/say", json={"text": "Nf3 is best."}).json()["ok"]
    assert [h for h in voice.heard if h.startswith("say")] == []
    assert "Nf3 is best." in [text for _, text, _ in engine.poll()["state"]["caption_lines"]]
    engine.set_muted(False)
    # The push-to-talk key held: the player is talking, the card is captioned, not said (Tai,
    # 2026-10-01: "if push to talk is on then the coach must be muted"; round 3 audit).
    engine._set(ptt=True)
    voice.heard.clear()
    assert client.post(f"/api/review/{key}/say", json={"text": "Rc1 next."}).json()["ok"]
    assert [h for h in voice.heard if h.startswith("say")] == []
    assert "Rc1 next." in [text for _, text, _ in engine.poll()["state"]["caption_lines"]]
    engine._set(ptt=False)
    # Hushed, and closing the review silences it too.
    assert client.post(f"/api/review/{key}/say", json={"text": "Qd2 holds."}).json()["ok"]
    assert client.post(f"/api/review/{key}/hush").json()["ok"]
    client.post(f"/api/review/{key}/close")
    assert voice.heard[-2:] == ["stop (review)", "stop (review)"]
    room.join(HANG_S)
    assert room.closed, "the room's thread ended, its store closed"
    # Closing a review while the coach has the voice stops the review's lines only: the
    # coach's are the voice's to keep (test_voice_lifecycle's owner tests).
    engine._voicing = ("post_move", 12)
    voice.heard.clear()
    client.post(f"/api/review/{key}/close")
    assert voice.heard == ["stop (review)"]
    assert "stop" not in voice.heard, "nothing of the coach's stopped"


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
        saved = store.best_review(key)
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
    ai = _until_written(client, key)
    why = "Your knight on c6 guards your pawn on e5."
    words = {
        "model": "local-model",
        "cost": 0.0,
        "texts": {"4": {"why": why}},
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

    _close(client, key)
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
    _close(client, key)
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
    ai = _until_written(client, key)
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


def test_the_words_paid_on_a_watched_game_show_in_its_imports_review(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """For Tai #9 (decided 2026-10-02): the AI coach's words paid for on a watched game's review
    whose record differs from its import (4. d3 read as d4) were kept but shown nowhere once the
    import stood for the game. They show in the import's review on the moves the two share -- the
    same position, the same move -- and are not asked (paid) again; the words on moves the import
    does not have stay with the watched review, which keeps every word it had. Nothing is lost
    and nothing is shown, saved or paid twice."""
    from sidecoach.events import Line
    from sidecoach.llm import factory
    from sidecoach.llm.base import Completion
    from sidecoach.llm.usage import Usage
    from sidecoach.ui import review_room

    searched = FakeEngine.analyse

    def legal(self: FakeEngine, board: Any, **kwargs: Any) -> list[Any]:
        """Lines of a legal move, as the words' facts and checks play them out."""
        searched(self, board, **kwargs)
        first = next(iter(board.legal_moves), None)
        return [Line(moves=[first.uci()], depth=14, expected_score=0.5)] if first else []

    monkeypatch.setattr(FakeEngine, "analyse", legal)
    misread = [*SANS[:6], "d4", *SANS[7:]]
    client, _engine, store, key, watched, _made = _watched_and_imported(
        make_panel, monkeypatch, misread
    )
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    # Plies 3 (Nf3) and 5 (Bc4) are the key moves; 7 is the misread move, 9 comes after it.
    monkeypatch.setattr(review_room.review_words, "key_plies", lambda _game: [3, 5])
    paid = {
        "model": "luna",
        "cost": 0.02,
        "texts": {
            "3": {"why": "Paid for on Nf3."},
            "7": {"why": "Paid for on d4."},
            "9": {"why": "Paid for on Nc3."},
        },
    }
    store.set_review_words(watched, "standard", json.dumps(paid))
    assert client.get("/api/games").json()["games"], "listed: the import stands for it now"
    assert store.twins() == {watched: key}

    asked: list[int] = []

    class Model:
        model = "local-model"
        usage = Usage()

        def chat(self, _system: str, messages: Any, **_: Any) -> Completion:
            ply = json.loads(messages[0]["content"])["key move"]["ply"]
            asked.append(ply)
            return Completion(text=f"### why\nOwn words on ply {ply}.", stats={})

    monkeypatch.setattr(factory, "make_provider", lambda llm, timeout_s=None: Model())
    assert client.put("/api/settings", json={"settings": {"llm": {"provider": "local"}}}).json()[
        "ok"
    ]
    client.post(f"/api/review/{key}/open", json={"strength": "standard"})
    ai = _until_ready(client, key)["state"]["ai"]
    assert ai["words"]["texts"] == {"3": {"why": "Paid for on Nf3."}}, "where the moves match"
    assert ai["status"] == "none", "the key move with no words is there to ask for"

    assert client.post(f"/api/review/{key}/ai").json() == {"ok": True, "error": None}
    ai = _until_written(client, key)
    assert asked == [5], "the move with words paid for is not asked again"
    assert ai["status"] == "ready", ai["error"]
    assert ai["words"]["texts"] == {
        "3": {"why": "Paid for on Nf3."},
        "5": {"why": "Own words on ply 5."},
    }
    own = store.review(key, "standard")
    assert own is not None and json.loads(own.words or "{}")["texts"] == {
        "5": {"why": "Own words on ply 5."}
    }, "the import saves its own words only: none saved twice"
    kept = store.review(watched, "standard")
    assert kept is not None and json.loads(kept.words or "{}") == paid, "the watched keeps all"

    _close(client, key)
    client.post(f"/api/review/{key}/open", json={"strength": "standard"})
    again = _until_ready(client, key)["state"]["ai"]
    assert (again["status"], again["words"]["texts"]) == ("ready", ai["words"]["texts"])
    assert asked == [5], "opened again: nothing asked"
    _close(client, key)
    store.close()


def test_the_end_of_game_card_says_the_reviews_accuracy_once_it_is_done(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """For Tai #11 (decided 2026-10-02): the card said 78.0 and Game Review 78.3 -- the live
    verdicts and the review's own analysis. The card says the live accuracy while the game's
    review is not done, then Game Review's, as the server's poll carries it: read again when a
    review is saved (by any store: the review window's, the background review's), and not asked
    of the store between changes."""
    from types import SimpleNamespace

    from sidecoach.events import GameReview, Side
    from sidecoach.profile.store import Store

    engine = make_panel()
    path = engine.settings.data_dir / "profile.sqlite"
    store = Store(path)
    watched = _watch(store, "g1", SANS, 1_790_000_000.0)

    class Ends:
        """The profiler as the game ends: its review, live."""

        def __init__(self) -> None:
            self.store = store

        def on_event(self, _event: Any, _now: float) -> list[Any]:
            return [GameReview(game_id="g1", user_side=Side.BLACK, result="0-1", accuracy=78.0)]

    engine._profile(Ends(), [SimpleNamespace()])
    card = next(line for line in engine.poll()["lines"] if (line.get("data") or {}).get("review"))
    assert card["data"]["review_key"] == watched and card["data"]["review"]["accuracy"] == 78.0
    assert engine.poll()["state"]["review_accuracy"] == {}, "not reviewed: the live accuracy"

    read: list[str] = []
    best = Store.best_review
    monkeypatch.setattr(Store, "best_review", lambda self, key: read.append(key) or best(self, key))
    engine.poll()
    assert read == [], "nothing changed: the store is not asked"

    other = Store(path)  # the review window's own store
    summary = {"white": {"accuracy": 90.1}, "black": {"accuracy": 78.3}}
    assert other.save_review(watched, "standard", 1, 1.0, "{}", json.dumps(summary))
    other.close()
    assert engine.poll()["state"]["review_accuracy"] == {watched: 78.3}, "Game Review's"
    assert read == [watched]
    engine.poll()
    assert read == [watched], "read once for the change"
    store.close()


def test_words_go_by_position_and_move_whatever_the_move_numbers(tmp_path: Path) -> None:
    """A watched game joined late counts its moves from another number: its FENs' clocks differ
    from the import's, its positions do not. Each word goes to the import's move played from the
    same position with the same move, the n-th such move to the n-th."""
    import chess

    from sidecoach.profile.store import LibraryRow, Store
    from sidecoach.ui import review_room
    from tests.test_review_words import _game

    store = Store(tmp_path / "profile.sqlite")
    imported = _game()
    moves = []
    for move in imported.moves:
        fields = move.fen_before.split()
        fields[5] = str(int(fields[5]) + 20)  # joined at move 21
        moves.append(move.model_copy(update={"fen_before": " ".join(fields)}))
    watched = imported.model_copy(update={"moves": moves})
    store.start_game("w", 1.0, "black", chess.STARTING_FEN)
    store.add_library_game(LibraryRow("pgn:i", "pgn", 0.0, "*"))
    assert store.save_review("watched:w", "fast", 1, 1.0, watched.model_dump_json(), None)
    store.set_review_words(
        "watched:w", "fast", json.dumps({"model": "m", "texts": {"4": {"why": "W"}}})
    )
    store.add_twin("watched:w", "pgn:i")
    assert review_room.carried_words(store, "pgn:i", imported, "standard") == (
        {4: {"why": "W"}},
        "m",
    )
    assert review_room.carried_words(store, "watched:w", watched, "fast") == ({}, ""), "one way"
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


def test_a_review_window_on_a_game_removed_meanwhile_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A game removed while its review window analysed it left the page at 100% for good (the
    review of 2026-10-01): the room ends failed, saying why."""
    from sidecoach.config import Settings
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    path = tmp_path / "profile.sqlite"
    store = Store(path)
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    progress = review_room.ReviewRoom._progress

    def removing(self: Any, done: int, total: int) -> None:
        if done == 3:
            store.remove_game(row.key)
        progress(self, done, total)

    monkeypatch.setattr(review_room.ReviewRoom, "_progress", removing)
    rooms = review_room.ReviewRooms(
        Settings(data_dir=tmp_path), store_path=path, make_engine=lambda **_: FakeEngine()
    )
    room = rooms.open(row.key, "fast")
    assert room is not None and room.settled.wait(HANG_S)
    state = room.state()
    assert state["status"] == "failed" and "no longer" in state["error"], state
    room.close()
    store.close()


def test_no_ai_words_are_paid_for_on_a_game_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AI words on a game removed meanwhile were paid for, then saved on nothing (the review of
    2026-10-01). Removed while they are written, no more calls are made and the room says why;
    removed before, none are asked."""
    from sidecoach.coach.review_words import ReviewWords
    from sidecoach.config import Settings
    from sidecoach.llm import factory
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    path = tmp_path / "profile.sqlite"
    store = Store(path)
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    stopped: list[bool] = []

    class Model:
        model = "gpt-x"

    def write(game: Any, model: Any, **kw: Any) -> ReviewWords:
        stopped.append(kw["stop"]())
        store.remove_game(row.key)  # removed while the first answer is written
        stopped.append(kw["stop"]())
        return ReviewWords(texts={2: {"why": "a"}}, model="gpt-x", cost=0.01)

    monkeypatch.setattr(factory, "make_provider", lambda llm, timeout_s=None: Model())
    monkeypatch.setattr(review_room.review_words, "write", write)
    settings = Settings(data_dir=tmp_path)
    settings.llm.provider = "local"
    rooms = review_room.ReviewRooms(settings, store_path=path, make_engine=lambda **_: FakeEngine())
    room = rooms.open(row.key, "fast")
    assert room is not None and room.settled.wait(HANG_S) and room.status == "ready"
    assert room.write_words() is None
    assert room.words_done.wait(HANG_S)
    assert stopped == [False, True], "no more calls once it is gone"
    assert (room.words_status, room.words_error) == ("failed", review_room.GONE)
    assert room.write_words() == review_room.GONE, "none asked on a game gone"
    room.close()
    store.close()


def test_a_review_closed_while_it_runs_stops_and_is_no_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The player closes Game Review while its engine is still at the game: the review stops
    there, saves nothing, and is not taken for a failure (nor for a game that left the
    record) -- the room just ends. Its engine's first search is held until the room is
    closed, so the close lands mid-review every time."""
    import threading

    from sidecoach.config import Settings
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room

    searching, gate = threading.Event(), threading.Event()

    class HeldEngine(FakeEngine):
        def analyse(self, board: Any, **kwargs: Any) -> Any:
            if not searching.is_set():
                searching.set()
                assert gate.wait(HANG_S), "the held search was never let go"
            return super().analyse(board, **kwargs)

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    path = tmp_path / "profile.sqlite"
    store = Store(path)
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    settings = Settings(data_dir=tmp_path)
    rooms = review_room.ReviewRooms(settings, store_path=path, make_engine=lambda **_: HeldEngine())
    room = rooms.open(row.key, "fast")
    assert room is not None and searching.wait(HANG_S), "the review never reached its engine"
    room.close()
    gate.set()
    assert room.settled.wait(HANG_S), "the review never settled"
    room.join(HANG_S)
    assert room.closed
    assert room.status != "failed" and room.error is None, (room.status, room.error)
    texts = [line["text"] for line in room.poll()["lines"]]
    assert not any("left the record" in t or "review failed" in t for t in texts), texts
    assert store.review(row.key, "fast") is None, "a stopped review is not saved"
    store.close()


def test_words_that_fail_with_the_store_failing_too_still_end_and_the_room_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The audit of 2026-10-02: the words' failure asked the store whether the game went,
    unguarded; a store failing there too ended the thread with the words "writing" for good,
    and the room, which stays open while words are written, never closed. The failure is said
    and the room closes as any other."""
    import sqlite3

    from sidecoach.config import Settings
    from sidecoach.llm import factory
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    path = tmp_path / "profile.sqlite"
    store = Store(path)
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)

    class Model:
        model = "gpt-x"

    def write(game: Any, model: Any, **kw: Any) -> Any:
        raise RuntimeError("the model is down")

    monkeypatch.setattr(factory, "make_provider", lambda llm, timeout_s=None: Model())
    monkeypatch.setattr(review_room.review_words, "write", write)
    settings = Settings(data_dir=tmp_path)
    settings.llm.provider = "local"
    rooms = review_room.ReviewRooms(settings, store_path=path, make_engine=lambda **_: FakeEngine())
    room = rooms.open(row.key, "fast")
    assert room is not None and room.settled.wait(HANG_S) and room.status == "ready"
    asked: list[str] = []

    def in_record(key: str) -> bool:
        asked.append(key)
        if len(asked) > 1:  # the words' own check passed; then the store fails too
            raise sqlite3.ProgrammingError("Cannot operate on a closed database.")
        return True

    monkeypatch.setattr(room.store, "in_record", in_record)
    assert room.write_words() is None
    assert room.words_done.wait(HANG_S)
    assert room.words_status == "failed" and "the model is down" in (room.words_error or "")
    room.close()
    # Asked at the room's own times, not the real clock: closed, it ends even just touched.
    assert room.closing(room.touched), "nothing being written: it closes"
    room.join(HANG_S)
    assert room.closed
    store.close()


def test_a_saved_review_that_cannot_be_summed_up_again_is_said(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The audit of 2026-10-02: a saved review rederive could not read was skipped in silence."""
    from sidecoach.profile.store import LibraryRow, Store
    from sidecoach.ui import review_room

    store = Store(tmp_path / "profile.sqlite")
    store.add_library_game(LibraryRow("pgn:1", "pgn", 0.0, "*"))  # a review is of a listed game
    store.save_review("pgn:1", "fast", review_room.REVIEW_VERSION, 1.0, "not a report", None)
    store.set_setting(review_room.DERIVED_KEY, "1")
    assert review_room.rederive(store) == 0
    assert "the saved review of pgn:1 (fast) could not be summed up again" in caplog.text
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
    words = {"model": "luna", "cost": 0.03, "texts": {"4": {"why": "Paid for."}}}
    store.set_review_words(row.key, "standard", json.dumps(words))
    saved = store.review(row.key, "standard")
    # The background review, begun before the window saved, finishes now.
    second = review_room.review_row(settings, row, store, standard, workers=1, make_engine=engines)
    assert store.review(row.key, "standard") == saved, "the review saved first, and its words"
    assert saved is not None and json.loads(saved.words or "{}") == words
    assert second is not None and second[0].model_dump_json() == saved.report, "handed the saved"
    store.close()


def test_a_pasted_game_reviewed_then_imported_keeps_its_review_and_paid_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tai, 2026-10-02: a game pasted and imported is one game. The pasted copy's review and the
    AI coach's words the player paid for go to the site's record (the one that stands for the
    game), whichever strength they are; a review the site's record has stays, and gains the
    words it lacks; the game is in Stats once."""
    from sidecoach.analysis.game_review import STRENGTHS
    from sidecoach.config import Settings
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    settings = Settings(data_dir=tmp_path)
    store = Store(tmp_path / "profile.sqlite")
    (pasted,) = library.from_pgn(GAME, user_names=["nosegazm"])
    store.add_library_game(pasted)

    def review(row: Any, strength: str) -> None:
        assert review_room.review_row(
            settings,
            row,
            store,
            STRENGTHS[strength],
            workers=1,
            make_engine=lambda **_: FakeEngine(),
        )

    review(pasted, "deep")
    words = json.dumps({"model": "luna", "cost": 0.03, "texts": {"4": {"why": "Paid for."}}})
    store.set_review_words(pasted.key, "deep", words)
    site = library.row_for(
        library.games_in(GAME)[0], source="chesscom", key="chesscom:9", user_names=["nosegazm"]
    )
    store.add_library_game(site)
    review(site, "standard")  # the site's record was reviewed at a lower strength meanwhile
    twins = library.reconcile(store)
    assert twins == {pasted.key: site.key}
    review_room.carry_twins(store, twins)

    assert [r.key for r in store.library()] == [site.key], "listed once"
    deep = store.review(site.key, "deep")
    assert deep is not None and deep.words == words, "the stronger review, with the paid words"
    assert json.loads(deep.report)["headers"]["Site"] == "Chess.com"
    assert store.review(site.key, "standard") is not None, "its own review stays"
    assert store.review(pasted.key, "deep") is not None, "and the pasted copy's is kept as it was"
    assert {g["key"] for g in store.stat_games()} == {site.key}, "Stats counts it once"
    assert store.stat_strengths() == {site.key: "deep", pasted.key: "deep"}

    # Words paid for on the pasted copy's Standard review reach the site's, which lacks them.
    review(pasted, "standard")
    store.set_review_words(pasted.key, "standard", words)
    review_room.carry_twins(store, twins)
    kept = store.review(site.key, "standard")
    assert kept is not None and kept.words == words
    again = store.review(site.key, "standard")
    review_room.carry_twins(store, twins)  # nothing more to do the next time
    assert store.review(site.key, "standard") == again
    store.close()


def test_saved_reviews_are_summed_up_again_when_the_rule_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The phases became Lichess's (efdd8a3) with the reviews saved as they were: 98 of 106
    sides' phase grades and 745 of 3,522 moves' phases in Stats were the old rule's (the review
    of 2026-10-01). A saved review's summary and its facts for Stats are made again from its
    report, once, no engine."""
    from sidecoach.analysis.game_review import STRENGTHS
    from sidecoach.config import Settings
    from sidecoach.events import GameSummary, ReviewedGame
    from sidecoach.profile import library
    from sidecoach.profile.game_summary import summarize
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    store = Store(tmp_path / "profile.sqlite")
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    fast = STRENGTHS["fast"]
    made = review_room.review_row(
        Settings(data_dir=tmp_path),
        row,
        store,
        fast,
        workers=1,
        make_engine=lambda **_: FakeEngine(),
    )
    assert made is not None
    saved = store.review(row.key, "fast")
    assert saved is not None
    fresh = summarize(ReviewedGame.model_validate_json(saved.report))
    phases = [m["phase"] for m in store.stat_moves([row.key])]
    # As an older rule left them, the rule's version of before.
    stale = fresh.model_copy(update={"shape": "old"})
    store.set_review_summary(row.key, "fast", stale.model_dump_json())
    store._db.execute("UPDATE stat_moves SET phase = 'endgame'")
    store.set_setting(review_room.DERIVED_KEY, "1")
    assert review_room.rederive(store) == 1
    again = store.review(row.key, "fast")
    assert again is not None and GameSummary.model_validate_json(again.summary or "{}") == fresh
    assert [m["phase"] for m in store.stat_moves([row.key])] == phases
    assert review_room.rederive(store) == 0, "once"
    store.close()


class _VariedEngine(FakeEngine):
    """An engine whose word on a position is a function of it: its first legal move (by UCI)
    as the best, and a score from the FEN's CRC -- a game of best moves, slips, mistakes and
    blunders on both sides, the same every run."""

    def analyse(self, board: Any, **kwargs: Any) -> list[Any]:
        import zlib

        from sidecoach.analysis.score import game_over
        from sidecoach.events import Line

        self.calls.append(board.fen())
        if game_over(board):
            return []
        best = sorted(m.uci() for m in board.legal_moves)[0]
        score = 0.2 + (zlib.crc32(board.fen().encode()) % 60) / 100
        return [Line(moves=[best], depth=6, expected_score=score)]


def _varied_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    """GAME reviewed at the fast strength with `_VariedEngine`: the store, and its library row."""
    from sidecoach.analysis.game_review import STRENGTHS
    from sidecoach.config import Settings
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    store = Store(tmp_path / "profile.sqlite")
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    made = review_room.review_row(
        Settings(data_dir=tmp_path),
        row,
        store,
        STRENGTHS["fast"],
        workers=1,
        make_engine=lambda **_: _VariedEngine(),
    )
    assert made is not None
    return store, row


def test_a_saved_review_relabelled_from_what_it_kept_is_labelled_as_the_review_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`relabel` reads a move's label from the saved measurements by the rule the annotator
    labels with (`score.verdict`): over a game of every kind of move, the labels it gives a
    review made today are the ones the review gave."""
    from sidecoach.analysis.game_review import relabel
    from sidecoach.analysis.score import Thresholds
    from sidecoach.events import ReviewedGame

    store, row = _varied_review(tmp_path, monkeypatch)
    saved = store.review(row.key, "fast")
    assert saved is not None
    reviewed = ReviewedGame.model_validate_json(saved.report)
    labels = [m.assessment.label for m in reviewed.moves]
    assert len(set(labels)) >= 4, labels  # a game of more than one kind of move
    assert [m.assessment.label for m in relabel(reviewed, Thresholds()).moves] == labels
    store.close()


def test_saved_reviews_are_relabelled_by_the_new_rules_and_keep_their_words(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tai, 2026-10-02: reviews saved before the label rules changed (a miss is failing to
    punish the opponent's error; chess.com's great, brilliant and book) and the openings'
    naming are made again from their saved data at startup, once -- and the AI coach's words,
    paid for, are kept to the byte. A review saved with the old rule's labels (every move a
    miss here) and an old opening name gets today's -- one relabelled by an earlier mate rule
    (stamped 4, Tai 12:16) too (Tai, 2026-10-02 12:50)."""
    import shutil

    from sidecoach.analysis.openings import OpeningBook
    from sidecoach.events import Label, ReviewedGame
    from sidecoach.ui import review_room

    store, row = _varied_review(tmp_path, monkeypatch)
    saved = store.review(row.key, "fast")
    assert saved is not None
    fresh = ReviewedGame.model_validate_json(saved.report)
    old = fresh.model_copy(
        update={
            "moves": [
                m.model_copy(
                    update={
                        "assessment": m.assessment.model_copy(update={"label": Label.MISS}),
                        "after": m.after.model_copy(update={"opening": "C20 King's Pawn Game"}),
                    }
                )
                for m in fresh.moves
            ]
        }
    )
    store._db.execute(
        "UPDATE reviews SET report = ? WHERE key = ?", (old.model_dump_json(), row.key)
    )
    words = json.dumps({"model": "luna", "cost": 0.03, "texts": {"4": {"why": "Paid for."}}})
    store.set_review_words(row.key, "fast", words)
    store.set_setting(review_room.DERIVED_KEY, "4")
    tables = tmp_path / "openings"
    tables.mkdir()
    shutil.copy(Path(__file__).parent / "openings_sample.tsv", tables / "c.tsv")
    book = OpeningBook.load(tables)
    assert review_room.rederive(store, book=book) == 1
    again = store.review(row.key, "fast")
    assert again is not None
    relabelled = ReviewedGame.model_validate_json(again.report)
    labels = [m.assessment.label for m in relabelled.moves]
    assert labels == [m.assessment.label for m in fresh.moves]
    assert [m.after.opening for m in relabelled.moves][4:] == ["C50 Italian Game"] * 8
    assert again.words == words, "the paid words are never touched"
    assert review_room.rederive(store, book=book) == 0, "once"
    store.close()


def test_ai_words_cut_short_by_failed_calls_offer_try_again_for_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some calls failed and the words were saved as done: the failed moves had none for good,
    and nothing offered to finish them (the review of 2026-10-01). The room says it could not
    finish, keeps the answers paid for, and Try again asks the rest alone."""
    from sidecoach.coach.review_words import ReviewWords
    from sidecoach.config import Settings
    from sidecoach.llm import factory
    from sidecoach.profile import library
    from sidecoach.profile.store import Store
    from sidecoach.ui import review_room

    monkeypatch.setattr(review_room, "find_stockfish", lambda _settings: Path("stockfish"))
    monkeypatch.setattr(review_room, "workers_for", lambda cores=None: 1)
    asked: list[set[int]] = []

    class Model:
        model = "gpt-x"

    def write(game: Any, model: Any, **kw: Any) -> ReviewWords:
        answered = set(kw["answered"])
        asked.append(answered)
        paid = {"prompt_tokens": 1, "completion_tokens": 1}
        if not answered:
            kw["on_answer"](2, "### why\nMove 2.", paid)
            return ReviewWords(
                texts={2: {"why": "a"}}, dropped={4: "the call failed: 429"}, failed={4}
            )
        kw["on_answer"](4, "### why\nMove 4.", paid)
        return ReviewWords(texts={2: {"why": "a"}, 4: {"why": "b"}}, model="gpt-x", cost=0.01)

    monkeypatch.setattr(factory, "make_provider", lambda llm, timeout_s=None: Model())
    monkeypatch.setattr(review_room.review_words, "write", write)
    monkeypatch.setattr(review_room.review_words, "key_plies", lambda game: [2, 4])
    settings = Settings(data_dir=tmp_path)
    settings.llm.provider = "local"
    path = tmp_path / "profile.sqlite"
    store = Store(path)
    row = library.from_pgn(GAME, user_names=["nosegazm"])[0]
    store.add_library_game(row)
    rooms = review_room.ReviewRooms(settings, store_path=path, make_engine=lambda **_: FakeEngine())
    room = rooms.open(row.key, "fast")
    assert room is not None and room.settled.wait(HANG_S) and room.status == "ready"
    assert room.write_words() is None
    assert room.words_done.wait(HANG_S)
    ai = room.state()["ai"]
    assert ai["status"] == "failed" and "1 of 2" in ai["error"] and ai["answered"] == 1
    saved = store.review(row.key, "fast")
    assert saved is not None and json.loads(saved.words or "{}").get("partial"), "kept, not done"
    assert room.write_words() is None  # Try again
    assert room.words_done.wait(HANG_S) and room.words_status == "ready"
    assert asked == [set(), {2}], "the rest alone"
    room.close()
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
        assert go.wait(HANG_S)  # each answer arrives, and is paid for, after the window closed
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
    room = rooms.open(row.key, "fast")
    assert room is not None and room.settled.wait(HANG_S) and room.status == "ready"
    assert room.write_words() is None
    room.close()
    # The room's own rule, asked at any moment: closed while writing, it does not end -- not
    # even once it has been left idle past IDLE_CLOSE_S.
    for now in (room.touched, room.touched + review_room.IDLE_CLOSE_S + 1):
        assert not room.closing(now), "open while its words are written"
    assert rooms.get(row.key) is room and not room.closed
    assert rooms.open(row.key, "fast") is room, "opened again: the same room, still writing"
    assert room.state()["ai"]["status"] == "writing"
    go.set()
    assert room.words_done.wait(HANG_S) and room.words_status == "ready"
    saved = store.review(row.key, "fast")
    assert saved is not None and json.loads(saved.words or "{}")["texts"] == {
        "2": {"why": "a"},
        "4": {"why": "b"},
    }, "saved whole"
    assert asked == [1], "asked once"
    assert not room.closing(room.touched), "opened again, it stays open"
    assert not room.closing(room.touched + review_room.IDLE_CLOSE_S), "... while it is used"
    assert room.closing(room.touched + review_room.IDLE_CLOSE_S + 1), "... and not when idle"
    assert rooms.get(row.key) is room

    room.close()
    room.join(HANG_S)
    assert room.closed, "closed with nothing being written: closed at once"

    # Closed while writing, not opened again: open until the words are saved, then closed.
    go.clear()
    store.set_review_words(row.key, "fast", "")
    again = rooms.open(row.key, "fast")
    assert again is not None and again is not room
    assert again.settled.wait(HANG_S) and again.status == "ready"
    assert again.write_words() is None
    again.close()
    assert not again.closing(again.touched) and not again.closed, "writing: stays"
    go.set()
    again.join(HANG_S)
    assert again.closed, "closed once its words are saved"
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
    assert room.settled.wait(HANG_S)
    room.close()
    room.join(HANG_S)
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


def test_a_miss_in_the_pgn_is_marked_by_what_it_cost() -> None:
    """A miss has no PGN mark of its own; "?" went on every one -- one that lost nothing (a
    mate let go, still winning) as on a blunder-sized one (the engine review, 2026-10-01). It
    takes its loss's mark: none, ?!, ? or ??. The other labels keep theirs."""
    from types import SimpleNamespace

    from sidecoach.ui.review_analysis import pgn

    row = SimpleNamespace(pgn="1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 *")
    rows = [
        ("miss", "none"),
        ("miss", "inaccuracy"),
        ("miss", "mistake"),
        ("miss", "blunder"),
        ("mistake", "mistake"),
        ("great", "none"),
    ]
    sans = ["e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"]
    moves = [
        {"ply": ply, "san": san, "label": label, "severity": severity, "best_san": None}
        for ply, (san, (label, severity)) in enumerate(zip(sans, rows, strict=True), start=1)
    ]
    text = pgn(row, moves, [])  # type: ignore[arg-type]
    marked = " ".join(text.split("\n\n", 1)[1].split())
    # PGN's numbers for them: $6 ?!, $2 ?, $4 ??, $1 !.
    for san, mark in zip(sans, [" {", " $6 {", " $2 {", " $4 {", " $2 {", " $1 {"], strict=True):
        assert f"{san}{mark}" in marked, (san, mark, marked)
