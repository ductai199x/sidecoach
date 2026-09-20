"""A puzzle solved on the panel's own board (ui/puzzle.py): the position from the solver's
side, the line's replies, a wrong move shown and taken back, any mate where the line
mates, the solution on give-up; and the engine's commands around it -- the coach quiet,
the screen's game not followed, the outcome to the profile once."""

from __future__ import annotations

from pathlib import Path

import chess

from sidecoach.events import GameStarted, MovePlayed, PuzzleOutcome, Side
from sidecoach.profile.puzzles import Puzzle
from sidecoach.ui.puzzle import PuzzleSession

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
    assert s.targets("d4") == ["b3", "b5", "c2", "c6", "e2", "f3", "f5"]
    assert s.targets("e5") and s.targets("g8") and s.targets("d2") == [], "not theirs"
    assert s.state()["your_move"] and s.state()["remaining"] == 2
    events = s.move("d4e2")
    assert [e.san for e in events] == ["Ne2+", "Kf1"], "the move, then the line's reply"
    assert all(isinstance(e, MovePlayed) for e in events)
    assert events[0].by_user and not events[1].by_user
    assert events[0].game_id == "puzzle-000Pw" and events[0].ply == 1 and events[1].ply == 2
    assert s.state()["your_move"] and s.state()["remaining"] == 1
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
    assert s.targets("d4") == []
    s.retract()
    assert s.state()["wrong"] is None and s.state()["your_move"]
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


class _Coach:
    quiet = False


class _Coaching:
    coach = _Coach()


class _Profiler:
    def __init__(self) -> None:
        self.recorded: list[tuple[str, str | None, bool, list[str]]] = []
        self.puzzles = _Store()

    def recommended_motif(self, puzzle_id: str) -> str | None:
        return "fork" if puzzle_id == FORK.id else None

    def record_puzzle(self, puzzle_id, motif, solved, moves, ts):  # noqa: ANN001
        self.recorded.append((puzzle_id, motif, solved, moves))
        return [
            PuzzleOutcome(
                game_id=f"puzzle-{puzzle_id}",
                puzzle_id=puzzle_id,
                motif=motif,
                solved=solved,
                level=1100,
            )
        ]


class _Store:
    def get(self, puzzle_id: str) -> Puzzle | None:
        return {FORK.id: FORK, MATE.id: MATE}.get(puzzle_id)


class _Pipe:
    reconciler = None
    model = None
    bottom = None


def test_the_engine_runs_a_puzzle_on_its_board_and_records_the_outcome_once(
    tmp_path: Path,
) -> None:
    from sidecoach.config import Settings, VoiceSettings
    from sidecoach.ui.panel import PanelEngine

    engine = PanelEngine(
        Settings(data_dir=tmp_path, voice=VoiceSettings(provider="null", stt_provider="null")),
        analysis=False,
        coach=False,
        frame_source=iter(()),
    )
    profiler = _Profiler()
    engine._profiler = profiler
    coaching = _Coaching()
    pipe = _Pipe()
    events = engine._puzzle_command("start", FORK.id, pipe, coaching, 1.0)
    assert [type(e).__name__ for e in events] == ["GameStarted"]
    assert coaching.coach.quiet, "the coach says nothing unasked"
    state = engine.poll(0)["state"]["puzzle"]
    assert state["id"] == FORK.id and state["motif"] == "fork" and state["your_move"]
    assert engine.puzzle_targets("d4") == ["b3", "b5", "c2", "c6", "e2", "f3", "f5"]
    assert engine._puzzle_command("move", "d4c2", pipe, coaching, 2.0) == []
    assert engine.poll(0)["state"]["puzzle"]["wrong"] == "d4c2"
    engine._puzzle_command("retract", None, pipe, coaching, 2.5)
    events = engine._puzzle_command("move", "d4e2", pipe, coaching, 3.0)
    assert [e.san for e in events if isinstance(e, MovePlayed)] == ["Ne2+", "Kf1"]
    assert profiler.recorded == []
    events = engine._puzzle_command("move", "e2c3", pipe, coaching, 4.0)
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["MovePlayed", "PuzzleOutcome"]
    assert profiler.recorded == [(FORK.id, "fork", False, ["Ne2+", "Kf1", "Nxc3"])], (
        "solved after a miss is a fail for the record"
    )
    assert engine.poll(0)["state"]["puzzle"]["status"] == "solved"
    # Closing records nothing twice and gives the coach its voice back.
    assert engine._puzzle_command("close", None, pipe, coaching, 5.0) == []
    assert not coaching.coach.quiet and engine.poll(0)["state"]["puzzle"] is None
    assert len(profiler.recorded) == 1
    # Giving up on the next one is a fail, recorded once as well.
    engine._puzzle_command("start", MATE.id, pipe, coaching, 6.0)
    events = engine._puzzle_command("giveup", None, pipe, coaching, 7.0)
    assert any(isinstance(e, PuzzleOutcome) and not e.solved for e in events)
    assert profiler.recorded[-1][0] == MATE.id and profiler.recorded[-1][1] is None
    assert engine._puzzle_command("close", None, pipe, coaching, 8.0) == []
    assert len(profiler.recorded) == 2
    # Next takes the recommended puzzles in turn; none recommended, a word in the log.
    engine._practice_ids = [FORK.id]
    events = engine._puzzle_command("next", None, pipe, coaching, 9.0)
    assert [type(e).__name__ for e in events] == ["GameStarted"]
    assert engine._puzzle_command("next", None, pipe, coaching, 10.0) == [], "none left"
