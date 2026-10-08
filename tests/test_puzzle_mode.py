"""A puzzle in a tab of its own (ui/puzzle.py, ui/puzzle_room.py): the position from the
solver's side, the line's replies, a wrong move shown and taken back, any mate where the
line mates, the solution on give-up; and the room around it, over the server's API --
its own coach and log, the outcome to the profile once, the screen's engine untouched."""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chess
import pytest

from sidecoach.events import GameStarted, MovePlayed, Side
from sidecoach.profile import glicko
from sidecoach.profile.puzzles import Puzzle, puzzle_level
from sidecoach.ui.panel import PanelEngine
from sidecoach.ui.puzzle import PuzzleSession
from sidecoach.ui.puzzle_room import PuzzleRoom
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
    assert st["message"] == "Not quite. Take another look."
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
    # The result line's numbers come once the room has recorded the outcome.
    assert s.state()["result"] is None
    s.recorded_as(True, 1100, 1125)
    assert s.state()["result"] == {"solved": True, "level": 1125, "change": 25}
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


class _RoomSignals:
    """What the rooms' loop threads did, as signals the test waits on rather than polling
    the API against a clock: each outcome line a room logs (the outcome is recorded by the
    room's loop, a tick after the answer that ended the puzzle), and each room's end."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.outcomes: queue.Queue[tuple[str, str]] = queue.Queue()
        self.lines: queue.Queue[tuple[str, str]] = queue.Queue()  # every line, any channel
        self.finished: dict[str, threading.Event] = {}
        self._lock = threading.Lock()
        real_emit = PuzzleRoom._emit
        real_finish = PuzzleRoom._finish
        signals = self

        def emit(room: PuzzleRoom, chan: str, level: str, text: str, **data: Any) -> None:
            real_emit(room, chan, level, text, **data)
            signals.lines.put((room._id, text))
            if chan == "game" and text.startswith("puzzle "):
                signals.outcomes.put((room._id, text))

        def finish(room: PuzzleRoom) -> None:
            real_finish(room)
            signals.ended(room._id).set()

        monkeypatch.setattr(PuzzleRoom, "_emit", emit)
        monkeypatch.setattr(PuzzleRoom, "_finish", finish)

    def ended(self, puzzle_id: str) -> threading.Event:
        with self._lock:
            return self.finished.setdefault(puzzle_id, threading.Event())

    def logged(self, client, puzzle_id: str, prefix: str) -> None:  # noqa: ANN001
        """The room's loop logs the outcome line starting with `prefix`, and the page's poll
        shows it. The 60 s is only a safety net: the line comes at the loop's next tick."""
        room_id, text = self.outcomes.get(timeout=HANG_S)
        assert room_id == puzzle_id and text.startswith(prefix), (room_id, text, prefix)
        lines = [ln["text"] for ln in client.get(f"/api/puzzle/{puzzle_id}/poll").json()["lines"]]
        assert any(ln.startswith(prefix) for ln in lines), lines

    def said(self, client, puzzle_id: str, text: str) -> None:  # noqa: ANN001
        """The room says `text` (any channel; the lines before it are passed over), and the
        page's poll shows it. The 60 s is only a safety net."""
        while True:
            room_id, line = self.lines.get(timeout=HANG_S)
            if room_id == puzzle_id and line == text:
                break
        lines = [ln["text"] for ln in client.get(f"/api/puzzle/{puzzle_id}/poll").json()["lines"]]
        assert text in lines, lines


HANG_S = 60.0  # a safety net for a thread's signal, never the expected path


def test_a_room_runs_a_puzzle_over_the_api_and_records_the_outcome_once(
    make_panel: Callable[..., PanelEngine], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi.testclient import TestClient

    from sidecoach.profile.store import Store
    from sidecoach.ui.server import build_app

    signals = _RoomSignals(monkeypatch)
    _csv(tmp_path)
    engine = make_panel()
    engine.puzzle_motifs[FORK.id] = "fork"
    engine.practice_ids = [FORK.id, MATE.id]
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    assert client.get(f"/puzzle/{FORK.id}").status_code == 200, "the page"
    assert client.post("/api/puzzle/nope/open").json()["ok"] is False

    opened = client.post(f"/api/puzzle/{FORK.id}/open").json()
    assert opened["ok"] and opened["next"] == [MATE.id]
    state = opened["state"]
    assert state["id"] == FORK.id and state["motif"] == "fork" and state["bottom"] == "black"
    assert state["your_move"] and not state["closed"]
    assert state["result"] is None, "no result line while the puzzle is played"
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
    signals.logged(client, FORK.id, f"puzzle {FORK.id} failed (fork)")
    store = Store(tmp_path / "profile.sqlite")
    assert [a[0] for a in store.attempts()] == [FORK.id] and not store.attempts()[0][2]
    # The puzzle rating as Lichess moves it: a new player (1500) loses to the puzzle's 1547 / 76,
    # opened as a fork puzzle (the page says "Puzzle · fork"): Lichess's themed training keeps
    # 0.7 of a loss on a theme that hints at the solution.
    lost = glicko.finish(glicko.Perf(), glicko.Glicko(1547, 76, 0.09), False, 0, theme="fork")
    full = glicko.finish(glicko.Perf(), glicko.Glicko(1547, 76, 0.09), False, 0)
    assert full.glicko.int_rating < lost.glicko.int_rating
    first = lost.glicko.int_rating
    assert puzzle_level(store) == first < 1500
    assert store.setting("puzzle_level") is None, "the old level is neither read nor written"
    # The page's result line reads the outcome as the profile took it: a miss is not a solve.
    result = client.get(f"/api/puzzle/{FORK.id}/poll").json()["state"]["result"]
    assert result == {"solved": False, "level": first, "change": first - 1500}
    # The user's question reaches the room. Its coach is off here: the room says so, as the
    # panel does, or the page's typing dots waited for an answer for good.
    assert client.post(f"/api/puzzle/{FORK.id}/ask", json={"text": "hint"}).json()["ok"]
    signals.said(client, FORK.id, "no coach to ask: hint")
    texts = [ln["text"] for ln in client.get(f"/api/puzzle/{FORK.id}/poll").json()["lines"]]
    assert "you: hint" not in texts, "no turn of a conversation nobody answers"
    assert client.post(f"/api/puzzle/{FORK.id}/voice", json={"on": False}).json()["ok"]
    # Closing ends the room; the outcome is not recorded twice, counted once the room's loop
    # has drained the close and finished (not merely once the close was asked for).
    assert client.post(f"/api/puzzle/{FORK.id}/close").json()["ok"]
    assert signals.ended(FORK.id).wait(HANG_S), "the room never finished"
    assert client.get(f"/api/puzzle/{FORK.id}/poll").json()["closed"]
    assert len(store.attempts()) == 1
    assert signals.outcomes.empty(), "an outcome logged twice"
    store.close()
    # Giving up on the next one: a fail, the solution played out, recorded once.
    assert client.post(f"/api/puzzle/{MATE.id}/open").json()["ok"]
    gave = client.post(f"/api/puzzle/{MATE.id}/giveup").json()["state"]
    assert gave["status"] == "given_up" and gave["moves"] == ["Ra8#"]
    signals.logged(client, MATE.id, f"puzzle {MATE.id} failed")
    store = Store(tmp_path / "profile.sqlite")
    assert [a[0] for a in store.attempts()] == [MATE.id, FORK.id]
    second = puzzle_level(store)
    assert second < first, "a give-up is a loss"
    store.close()
    result = client.get(f"/api/puzzle/{MATE.id}/poll").json()["state"]["result"]
    assert result == {"solved": False, "level": second, "change": second - first}
    client.post(f"/api/puzzle/{MATE.id}/close")
    assert signals.ended(MATE.id).wait(HANG_S), "the room never finished"


def test_a_solved_puzzle_shows_the_rating_it_moved_to(
    make_panel: Callable[..., PanelEngine], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The result line after a solve: the puzzle rating after it and the change Lichess shows
    # (the difference of the two ratings as shown, truncated).
    from fastapi.testclient import TestClient

    from sidecoach.ui.server import build_app

    signals = _RoomSignals(monkeypatch)
    _csv(tmp_path)
    won = glicko.finish(glicko.Perf(), glicko.Glicko(900, 70, 0.09), True, 0).glicko.int_rating
    assert 1500 < won < 1600, "a solve against a much weaker puzzle: a small gain"
    engine = make_panel()
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    assert client.post(f"/api/puzzle/{MATE.id}/open").json()["state"]["result"] is None
    done = client.post(f"/api/puzzle/{MATE.id}/move", json={"uci": "a1a8"}).json()["state"]
    assert done["status"] == "solved"
    # This room's line alone: a room another test closed may still log its own outcome.
    signals.said(client, MATE.id, f"puzzle {MATE.id} solved; puzzle level {won}")
    result = client.get(f"/api/puzzle/{MATE.id}/poll").json()["state"]["result"]
    assert result == {"solved": True, "level": won, "change": won - 1500}
    client.post(f"/api/puzzle/{MATE.id}/close")
    assert signals.ended(MATE.id).wait(HANG_S), "the room never finished"


def test_a_puzzle_stats_offers_for_an_area_opens_with_its_motif(
    make_panel: Callable[..., PanelEngine], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stats' "Puzzles on it": a puzzle picked for an area to work on opens with that area's
    # motif, so its attempt counts for the motif (on 0.1.85 it opened with none: the rooms knew
    # only the motifs of the live session's recommendations).
    from fastapi.testclient import TestClient

    from sidecoach.profile import stats as stat_sums
    from sidecoach.ui.server import build_app

    _csv(tmp_path)  # FORK's 1547 is within 150 of a new player's puzzle rating, 1500
    real = stat_sums.stats

    def with_fork_to_work_on(store: Any, f: Any) -> dict[str, Any]:
        numbers = real(store, f)
        numbers["work"] = [{"area": "tactic.fork", "motif": "fork"}]
        return numbers

    monkeypatch.setattr(stat_sums, "stats", with_fork_to_work_on)
    engine = make_panel()
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    work = client.get("/api/stats").json()["stats"]["work"]
    assert [p["id"] for p in work[0]["puzzles"]] == [FORK.id]
    opened = client.post(f"/api/puzzle/{FORK.id}/open").json()
    assert opened["ok"] and opened["state"]["motif"] == "fork"
    client.post(f"/api/puzzle/{FORK.id}/close")


def test_puzzles_fetched_after_the_server_started_open_in_a_room(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    # Setup fetches the puzzles while the app runs: the session finds them at its next
    # Start and recommends one, and the room must find it too, without a restart.
    from fastapi.testclient import TestClient

    from sidecoach.ui.server import build_app

    engine = make_panel()
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    assert client.post(f"/api/puzzle/{FORK.id}/open").json()["ok"] is False
    _csv(tmp_path)
    opened = client.post(f"/api/puzzle/{FORK.id}/open").json()
    assert opened["ok"] and opened["state"]["id"] == FORK.id
    client.post(f"/api/puzzle/{FORK.id}/close")


def test_a_room_closes_with_its_page_never_for_being_left_alone(
    make_panel: Callable[..., PanelEngine], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tai, 2026-10-02: a puzzle left open over lunch was gone after 20 idle minutes. A room
    closes when its page says it goes; the one safety net is a page that said it was in view
    and then went quiet (its process died). The room's own clock is the test's; its loop's
    checks are signals the test waits on."""
    from fastapi.testclient import TestClient

    from sidecoach.ui import puzzle_room
    from sidecoach.ui.server import build_app

    signals = _RoomSignals(monkeypatch)
    now = [1000.0]
    checks: queue.Queue[tuple[float, bool]] = queue.Queue()
    real_vanished = PuzzleRoom.vanished

    def vanished(room: PuzzleRoom) -> bool:
        room.clock = lambda: now[0]
        gone = real_vanished(room)
        checks.put((now[0], gone))
        return gone

    def checked_at(t: float) -> bool:
        """The loop's verdict at the clock's `t`, the checks before it passed over."""
        while True:
            at, gone = checks.get(timeout=HANG_S)
            if at == t:
                return gone

    monkeypatch.setattr(PuzzleRoom, "vanished", vanished)
    _csv(tmp_path)
    client = TestClient(base_url="http://127.0.0.1", app=build_app(make_panel()))
    assert client.post(f"/api/puzzle/{FORK.id}/open").json()["ok"]
    # Out of view, as the page says when it is hidden: hours of quiet close nothing.
    assert client.get(f"/api/puzzle/{FORK.id}/poll?hidden=1").json()["ok"]
    now[0] += 3 * 3600
    assert checked_at(now[0]) is False and checked_at(now[0]) is False
    # Back in view: polled, then the same hours with the page in view and not a poll.
    poll = client.get(f"/api/puzzle/{FORK.id}/poll").json()
    assert poll["ok"] and not poll["state"]["closed"]
    now[0] += puzzle_room.VANISHED_S
    assert checked_at(now[0]) is False  # not past it
    now[0] += 1.0
    assert checked_at(now[0]) is True
    assert signals.ended(FORK.id).wait(HANG_S), "the vanished page's room never finished"
    said = [line for room_id, line in list(signals.lines.queue) if room_id == FORK.id]
    assert "the page is gone without closing; closing" in said
    assert client.get(f"/api/puzzle/{FORK.id}/poll").json() == {"ok": False, "closed": True}
    # A page that says it goes closes its room at once (its close, or its beacon).
    assert client.post(f"/api/puzzle/{FORK.id}/open").json()["ok"]
    assert client.post(f"/api/puzzle/{FORK.id}/close").json()["ok"]
    assert client.get(f"/api/puzzle/{FORK.id}/poll").json()["closed"]


def test_hint_best_move_or_line_while_playing_makes_the_attempt_unrated() -> None:
    # Lichess: a hint shown, the round is not rated. The page's chips send these words; a
    # question of the player's own words does not count, nor a chip once the puzzle is over.
    for chip in ("hint", "what should I play", "show me the line", "  Hint "):
        s = PuzzleSession(FORK, "fork")
        s.asked(chip)
        assert s.assisted, chip
    s = PuzzleSession(FORK, "fork")
    s.asked("why does the knight go there?")
    assert not s.assisted
    s.move("d4e2")
    s.move("e2c3")
    assert s.done
    s.asked("hint")
    assert not s.assisted, "after the end: a review, not help"


class _Coached:
    """A room's coach that is on and takes questions (a real one needs Stockfish)."""

    on = True
    coach = None
    detail = "on"
    sink = None

    def __init__(self) -> None:
        self.asked: list[str] = []

    def ask(self, text: str, ts: float) -> None:
        self.asked.append(text)

    def events(self, events: Any, ts: float) -> None:
        pass

    def tick(self, ts: float) -> None:
        pass

    def drain(self) -> None:
        pass

    def close(self) -> None:
        pass


def test_a_puzzle_solved_after_a_hint_leaves_the_rating_as_it_was(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sidecoach.config import Settings
    from sidecoach.profile.store import Store

    signals = _RoomSignals(monkeypatch)
    settings = Settings(data_dir=tmp_path)
    room = PuzzleRoom(
        settings, FORK, "fork", store_path=tmp_path / "profile.sqlite", analysis=False, coach=False
    )
    coached = _Coached()
    room._coaching = coached  # type: ignore[assignment]
    try:
        room.command("ask", "hint")
        room.move("d4e2")
        done = room.move("e2c3")
        assert done["status"] == "solved"
        # This room's outcome line: a room another test closed may still log its own.
        while not signals.outcomes.get(timeout=HANG_S)[1].startswith(f"puzzle {FORK.id} solved"):
            pass
        assert coached.asked == ["hint"]
        store = Store(tmp_path / "profile.sqlite")
        assert puzzle_level(store) == 1500, "unrated: the rating stays"
        assert [(pid, ok) for pid, _, ok in store.attempts()] == [(FORK.id, True)], "kept"
        store.close()
        assert room.state()["result"] == {"solved": True, "level": 1500, "change": 0}
    finally:
        room.command("close")
        assert signals.ended(FORK.id).wait(HANG_S)


def test_a_puzzle_closed_before_any_wrong_move_records_nothing(
    make_panel: Callable[..., PanelEngine], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Lichess records a puzzle's result at its first wrong move or at its end; one left before
    # either is not played. One closed after a wrong move is the fail it already was.
    from fastapi.testclient import TestClient

    from sidecoach.profile.store import Store
    from sidecoach.ui.server import build_app

    signals = _RoomSignals(monkeypatch)
    _csv(tmp_path)
    engine = make_panel()
    client = TestClient(base_url="http://127.0.0.1", app=build_app(engine))
    assert client.post(f"/api/puzzle/{FORK.id}/open").json()["ok"]
    right = client.post(f"/api/puzzle/{FORK.id}/move", json={"uci": "d4e2"}).json()["state"]
    assert right["moves"] == ["Ne2+", "Kf1"], "one right move, then left"
    assert client.post(f"/api/puzzle/{FORK.id}/close").json()["ok"]
    assert signals.ended(FORK.id).wait(HANG_S), "the room never finished"
    store = Store(tmp_path / "profile.sqlite")
    assert store.attempts() == [] and puzzle_level(store) == 1500
    store.close()
    # A wrong move, then closed: the fail is recorded.
    assert client.post(f"/api/puzzle/{MATE.id}/open").json()["ok"]
    wrong = client.post(f"/api/puzzle/{MATE.id}/move", json={"uci": "a1a7"}).json()["state"]
    assert wrong["wrong"] == "a1a7"
    assert client.post(f"/api/puzzle/{MATE.id}/close").json()["ok"]
    assert signals.ended(MATE.id).wait(HANG_S), "the room never finished"
    store = Store(tmp_path / "profile.sqlite")
    assert [(pid, ok) for pid, _, ok in store.attempts()] == [(MATE.id, False)]
    assert puzzle_level(store) < 1500
    store.close()
