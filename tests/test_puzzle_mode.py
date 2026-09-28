"""A puzzle in a tab of its own (ui/puzzle.py, ui/puzzle_room.py): the position from the
solver's side, the line's replies, a wrong move shown and taken back, any mate where the
line mates, the solution on give-up; and the room around it, over the server's API --
its own coach and log, the outcome to the profile once, the screen's engine untouched."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import chess

from sidecoach.events import GameStarted, MovePlayed, Side
from sidecoach.profile.puzzles import Puzzle
from sidecoach.ui.panel import PanelEngine
from sidecoach.ui.puzzle import PuzzleSession
from tests.fakes import puzzle_file

FORK = Puzzle(
    id="000Pw",
    fen="6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - - 2 37",
    moves=("e4d2", "d4e2", "g1f1", "e2c3"),
    rating=1547,
    popularity=76,
    plays=629,
    themes=("fork", "endgame"),
)
# Black's king steps to the corner; White mates on the back rank.
MATE = Puzzle(
    id="mate1",
    fen="6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1",
    moves=("g8h8", "a1a8"),
    rating=900,
    popularity=70,
    plays=95,
    themes=("mateIn1", "backRankMate"),
)


def test_the_session_shows_the_solvers_position_and_walks_the_line() -> None:
    s = PuzzleSession(FORK, "fork")
    assert s.side is Side.BLACK and s.state()["bottom"] == "black"
    assert s.start_fen.split(" ")[1] == "b", "after the opponent's e4d2, Black to move"
    started = s.started()
    assert isinstance(started, GameStarted) and started.kind == "puzzle"
    assert started.game_id == "puzzle-000Pw" and started.user_side is Side.BLACK
    legal = s.state()["legal"]
    assert legal["d4"] == ["b3", "b5", "c2", "c6", "e2", "f3", "f5"]
    assert "e5" in legal and "g8" in legal and "d2" not in legal, "the solver's pieces only"
    assert s.state()["your_move"] and s.state()["remaining"] == 2
    before = s.state()
    assert [p["uci"] for p in before["positions"]] == [None], "the start alone"
    events = s.move("d4e2")
    assert [e.san for e in events] == ["Ne2+", "Kf1"], "the move, then the line's reply"
    assert all(isinstance(e, MovePlayed) for e in events)
    assert events[0].by_user and not events[1].by_user
    assert events[0].game_id == "puzzle-000Pw" and events[0].ply == 1 and events[1].ply == 2
    after = s.state()
    assert after["your_move"] and after["remaining"] == 1
    assert after["version"] > before["version"], "a newer answer tells itself apart"
    # One position a ply, for the page to step through: its move, the check, who played it.
    plies = after["positions"][1:]
    assert [(p["uci"], p["check"], p["by_user"]) for p in plies] == [
        ("d4e2", "g1", True),
        ("g1f1", None, False),
    ]
    assert plies[-1]["fen"] == s.board.fen()
    events = s.move("e2c3")
    assert [e.san for e in events] == ["Nxc3"] and s.status == "solved" and s.done
    assert s.state()["message"] == "Solved." and not s.failed
    assert s.move("g8f8") == [], "over"


def test_a_wrong_move_is_shown_and_taken_back_and_counts_as_a_miss() -> None:
    s = PuzzleSession(FORK, "fork")
    assert s.move("d4c2") == []
    st = s.state()
    assert st["wrong"] == "d4c2" and not st["your_move"] and st["misses"] == 1
    assert "Not that one" in st["message"]
    assert s.move("d4e2") == [], "nothing moves while the wrong move is shown"
    assert st["legal"] == {}
    s.retract()
    assert s.state()["wrong"] is None and s.state()["your_move"]
    assert s.state()["version"] > st["version"], "the take-back is a newer state"
    assert s.board.fen() == s.start_fen, "the position as it was"
    assert s.move("d4e2") and s.move("e2c3") and s.status == "solved" and s.failed
    assert s.state()["message"] == "Solved, after a miss."
    # Not legal, not their piece, not their turn: ignored, no miss.
    t = PuzzleSession(FORK, "fork")
    assert t.move("d4d8") == [] and t.move("d2e4") == [] and t.move("zz") == []
    assert t.misses == 0 and t.wrong is None


def test_a_mate_puzzle_and_giving_up_on_it() -> None:
    s = PuzzleSession(MATE, "mate")
    assert s.side is Side.WHITE and s.board.turn == chess.WHITE
    events = s.move("a1a8")
    assert [e.san for e in events] == ["Ra8#"] and s.status == "solved" and not s.failed
    # Any mate counts where the line's move mates; a non-mating move does not.
    t = PuzzleSession(MATE, "mate")
    assert t._mates(chess.Move.from_uci("a1a8"))
    assert not t._mates(chess.Move.from_uci("a1a7"))
    assert t.move("a1a7") == [] and t.wrong == "a1a7"
    t.retract()
    events = t.give_up()
    assert [e.uci for e in events] == ["a1a8"] and t.status == "given_up" and t.failed
    assert t.state()["message"] == "The solution, played out."
    assert t.give_up() == [], "once"


def _csv(tmp_path: Path) -> None:
    rows = [
        f"{FORK.id},{FORK.fen},{' '.join(FORK.moves)},{FORK.rating},76,92,629,fork endgame,u,",
        f"{MATE.id},{MATE.fen},{' '.join(MATE.moves)},{MATE.rating},70,95,900,mateIn1,u,",
    ]
    puzzle_file(tmp_path / "puzzles" / "sample.csv", rows)


def _logged(client, puzzle_id: str, prefix: str) -> None:  # noqa: ANN001
    """Wait for the room's log line that starts with `prefix`: the outcome is recorded by the
    room's loop, a tick after the answer that ended the puzzle."""
    import time

    lines: list[str] = []
    for _ in range(100):
        lines = [ln["text"] for ln in client.get(f"/api/puzzle/{puzzle_id}/poll").json()["lines"]]
        if any(ln.startswith(prefix) for ln in lines):
            return
        time.sleep(0.05)
    raise AssertionError(f"no line {prefix!r} in {lines}")


def test_a_room_runs_a_puzzle_over_the_api_and_records_the_outcome_once(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    from fastapi.testclient import TestClient

    from sidecoach.profile.store import Store
    from sidecoach.ui.server import build_app

    _csv(tmp_path)
    engine = make_panel()
    engine.puzzle_motifs[FORK.id] = "fork"
    engine.practice_ids = [FORK.id, MATE.id]
    client = TestClient(build_app(engine))
    assert client.get(f"/puzzle/{FORK.id}").status_code == 200, "the page"
    assert client.post("/api/puzzle/nope/open").json()["ok"] is False

    opened = client.post(f"/api/puzzle/{FORK.id}/open").json()
    assert opened["ok"] and opened["next"] == [MATE.id]
    state = opened["state"]
    assert state["id"] == FORK.id and state["motif"] == "fork" and state["bottom"] == "black"
    assert state["your_move"] and not state["closed"]
    assert state["legal"]["d4"] == ["b3", "b5", "c2", "c6", "e2", "f3", "f5"]
    # Each move is answered with the state it leaves, at once. A wrong move shows, then is
    # taken back.
    wrong = client.post(f"/api/puzzle/{FORK.id}/move", json={"uci": "d4c2"}).json()
    assert wrong["ok"] and wrong["state"]["wrong"] == "d4c2"
    back = client.post(f"/api/puzzle/{FORK.id}/retract").json()["state"]
    assert back["wrong"] is None and back["your_move"]
    assert back["version"] > wrong["state"]["version"]
    # The line: the reply comes with the right move.
    right = client.post(f"/api/puzzle/{FORK.id}/move", json={"uci": "d4e2"}).json()["state"]
    assert right["moves"] == ["Ne2+", "Kf1"] and right["your_move"]
    done = client.post(f"/api/puzzle/{FORK.id}/move", json={"uci": "e2c3"}).json()["state"]
    assert done["status"] == "solved" and done["message"] == "Solved, after a miss."
    _logged(client, FORK.id, f"puzzle {FORK.id} failed (fork)")
    store = Store(tmp_path / "profile.sqlite")
    assert [a[0] for a in store.attempts()] == [FORK.id] and not store.attempts()[0][2]
    assert store.setting("puzzle_level") == "1075"
    # The user's question reaches the room's coach (off here: nothing answers, no error).
    assert client.post(f"/api/puzzle/{FORK.id}/ask", json={"text": "hint"}).json()["ok"]
    assert client.post(f"/api/puzzle/{FORK.id}/voice", json={"on": False}).json()["ok"]
    # Closing ends the room; the outcome is not recorded twice.
    assert client.post(f"/api/puzzle/{FORK.id}/close").json()["ok"]
    import time

    for _ in range(50):
        if client.get(f"/api/puzzle/{FORK.id}/poll").json().get("closed"):
            break
        time.sleep(0.05)
    assert client.get(f"/api/puzzle/{FORK.id}/poll").json()["closed"]
    assert len(store.attempts()) == 1
    store.close()
    # Giving up on the next one: a fail, the solution played out, recorded once.
    assert client.post(f"/api/puzzle/{MATE.id}/open").json()["ok"]
    gave = client.post(f"/api/puzzle/{MATE.id}/giveup").json()["state"]
    assert gave["status"] == "given_up" and gave["moves"] == ["Ra8#"]
    _logged(client, MATE.id, f"puzzle {MATE.id} failed")
    store = Store(tmp_path / "profile.sqlite")
    assert [a[0] for a in store.attempts()] == [MATE.id, FORK.id]
    assert store.setting("puzzle_level") == "1050"
    store.close()
    client.post(f"/api/puzzle/{MATE.id}/close")


def test_puzzles_fetched_after_the_server_started_open_in_a_room(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    # Setup fetches the puzzles while the app runs: the session finds them at its next
    # Start and recommends one, and the room must find it too, without a restart.
    from fastapi.testclient import TestClient

    from sidecoach.ui.server import build_app

    engine = make_panel()
    client = TestClient(build_app(engine))
    assert client.post(f"/api/puzzle/{FORK.id}/open").json()["ok"] is False
    _csv(tmp_path)
    opened = client.post(f"/api/puzzle/{FORK.id}/open").json()
    assert opened["ok"] and opened["state"]["id"] == FORK.id
    client.post(f"/api/puzzle/{FORK.id}/close")
