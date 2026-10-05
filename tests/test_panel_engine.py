"""The panel's live engine, driven headlessly from rendered frames (no window, no screen).

It runs the same capture -> pipeline -> engine loop `watch` runs, so feeding it the frames
of a game must lock the board, bootstrap the start position, follow the moves, and surface
all of that as state and as debug-terminal lines -- the thread-safe surface the web page
polls.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from sidecoach.config import Settings, VoiceSettings
from sidecoach.ui.panel import PanelApi, PanelEngine
from tests.fakes import FakeEngine, FedMic
from tests.synth import game_frames

# Scholar's mate: a game short enough to render cheaply, with an end the board shows.
MATE = ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6", "Qxf7#")


@pytest.mark.embedder  # a session started builds the stages: the net loads
def test_engine_follows_a_game_and_reports_it(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    """A game from its start to its mate, as the page sees it: the board locked, the moves
    and the end in the terminal and the state. Without an engine there are no verdicts,
    but the game itself reaches the profile, the review card and the session summary; a
    replay (an injected frame source) keeps the profile in memory and writes no file."""
    engine = make_panel(record=False, frame_source=game_frames(MATE))
    engine.start()
    assert engine._thread is not None
    engine._thread.join(timeout=90)
    assert not engine._thread.is_alive(), "engine loop did not finish the game in time"

    snap = engine.poll(0)
    state, lines = snap["state"], snap["lines"]
    texts = [ln["text"] for ln in lines]
    chans = {ln["chan"] for ln in lines}

    assert {"sys", "vision", "game"} <= chans, f"channels seen: {chans}"
    assert any("board locked" in t for t in texts), "never reported locking the board"
    assert any("start position seen" in t for t in texts), "never bootstrapped the start position"
    moves = [
        ln
        for ln in lines
        if ln["chan"] == "game" and ("(you" in ln["text"] or "(opponent" in ln["text"])
    ]
    assert len(moves) == len(MATE), "not every move was surfaced to the terminal"
    assert all("ply" in ln["data"] and "uci" in ln["data"] for ln in moves), (
        "move records carry no data"
    )
    assert state["health"].get("stage"), "no health strip in the state"
    assert state["session_dir"] and (Path(state["session_dir"]) / "debug.jsonl").exists(), (
        "no debug.jsonl written for the session"
    )
    assert state["fen"], "no position in the state"
    assert state["ply"] == len(MATE), f"expected every ply, got {state['ply']}"
    assert state["following"] or state["moves"], "engine never followed a game"
    assert state["running"] is False, "engine should have stopped after the recording ended"
    # The end, the profile and the review card.
    assert any(t.startswith("game over 1-0") for t in texts), "the mate was not seen"
    assert state["profile"] is not None and state["profile"]["games"] == 1
    assert not (tmp_path / "profile.sqlite").exists(), "a replay must not write the profile"
    review = state["review"]
    assert review is not None and review["result"] == "1-0" and review["reason"] == "checkmate"
    assert review["moves"] == 0, "no engine: no move judged"
    assert any(ln["chan"] == "coach" and "review" in ln.get("data", {}) for ln in lines)
    assert any(t.startswith("session:") for t in texts)


@pytest.mark.embedder  # a session started builds the stages: the net loads
def test_a_session_ended_on_a_repetition_played_records_the_draw(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """The session ends (here the recording runs out) with a game whose end was never seen:
    its final position is the third repetition, which the sites draw. It is a draw by
    repetition, as a game the next one replaced is (Tai, 2026-10-02), not a game left
    unfinished ("*", "the end was not seen")."""
    opening = ("d4", "d5", "e3", "e6", "Bd3", "Bd6", "Nf3", "Nf6")
    shuffle = ("Nc3", "Nc6", "Nb1", "Nb8")
    engine = make_panel(record=False, frame_source=game_frames((*opening, *shuffle, *shuffle)))
    engine.start()
    assert engine._thread is not None
    engine._thread.join(timeout=90)
    assert not engine._thread.is_alive(), "engine loop did not finish the game in time"
    snap = engine.poll(0)
    review = snap["state"]["review"]
    assert review is not None
    assert (review["result"], review["reason"]) == ("1/2-1/2", "repetition")
    assert any(ln["text"] == "game over 1/2-1/2 (repetition)" for ln in snap["lines"])


def test_ending_a_game_on_the_users_word_scores_it_from_their_side() -> None:
    outcome = PanelEngine.outcome_result
    assert outcome("resigned", "white") == ("0-1", "user resigned or lost on time")
    assert outcome("resigned", "black") == ("1-0", "user resigned or lost on time")
    assert outcome("won", "white") == ("1-0", "opponent resigned or lost on time")
    assert outcome("won", "black") == ("0-1", "opponent resigned or lost on time")
    assert outcome("draw", "white") == ("1/2-1/2", "draw agreed")
    assert outcome("forget", "black")[0] == "*"


def test_a_board_lost_track_of_is_told_once_until_it_is_read_again(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """A board the reader fails on is never left in silence (Tai, 2026-10-03): the long
    `Unexplained` note is said in the coach's pane, once, and again only after the board has
    been read since (a move); the early note and a refused commit stay in the log."""
    import chess

    from sidecoach.events import MovePlayed, Unexplained
    from sidecoach.ui.panel import LOST_TRACK

    engine = make_panel()

    class Glue:
        def submit(self, event: object) -> None:
            pass

    def lost() -> int:
        return sum(1 for r in engine._log.since(0) if r.get("text", "").startswith("board · lost"))

    for told in (False, True, True):
        engine._dispatch_game(
            Unexplained(game_id="g", fen=chess.STARTING_FEN, reason="unexplained: x", told=told),
            Glue(),
            chess,
        )
    assert lost() == 1
    assert any(r.get("data", {}).get("full") == LOST_TRACK for r in engine._log.since(0))
    board = chess.Board()
    before = board.fen()
    move = board.push_san("e4")
    played = MovePlayed(
        game_id="g",
        ply=1,
        san="e4",
        uci=move.uci(),
        fen_before=before,
        fen_after=board.fen(),
        by_user=True,
    )
    engine._dispatch_game(played, Glue(), chess)
    engine._dispatch_game(
        Unexplained(game_id="g", fen=board.fen(), reason="unexplained: x", told=True),
        Glue(),
        chess,
    )
    assert lost() == 2


def test_end_game_refuses_unknown_outcomes(make_panel: Callable[..., PanelEngine]) -> None:
    engine = make_panel()
    with pytest.raises(ValueError):
        engine.end_game("rage-quit")
    engine.end_game("draw")
    assert engine._commands.get_nowait() == ("end", "draw")


def test_with_no_session_flip_re_read_and_end_wait_for_the_next_start(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """With no session running, Flip, Re-read and End game are queued for the loop the next
    Start runs, in the order pressed; an ask between them is answered at once and not queued
    (the parity audit, 2026-09-29: the phone pinned the queue, the desktop only the end)."""
    engine = make_panel(record=False)
    api = PanelApi(engine)
    assert api.flip() and api.analyze() and api.end("resigned") and api.ask("hint")
    queued = []
    while not engine._commands.empty():
        queued.append(engine._commands.get_nowait())
    assert queued == [("flip", None), ("analyze", None), ("end", "resigned")]


def test_legal_move_dots_are_not_a_degraded_reading(make_panel: Callable[..., PanelEngine]) -> None:
    """A picked-up rook puts a dot on a dozen squares, each too small to be a piece and read
    as unknown; the first live game logged "reading degraded" on nearly every endgame move
    for it. Only covered cells and unknowns that are piece-sized or hovered count."""
    from types import SimpleNamespace

    from sidecoach.ui.panel import DEGRADED_FRAMES
    from sidecoach.vision.pieces import CellReading

    def cell(kind: str, fraction: float = 0.0, **kw: object) -> CellReading:
        return CellReading(kind, None, None, 0.0, fraction, 0.0, **kw)  # type: ignore[arg-type]

    engine = make_panel()
    dots = [[cell("empty") for _ in range(8)] for _ in range(8)]
    for c in range(8):  # a rook's whole rank of dots, plus its own hovered square
        dots[3][c] = cell("unknown", 0.08)
    dots[3][0] = cell("unknown", 0.3, hover=True)
    degraded = False
    for _ in range(DEGRADED_FRAMES + 2):
        degraded = engine._check_degraded(SimpleNamespace(readings=dots), degraded)
    assert degraded is False
    # A dialog fading in covers cells; that is degraded.
    dialog = [[cell("empty") for _ in range(8)] for _ in range(8)]
    for r in range(3):
        for c in range(3):
            dialog[r][c] = cell("unknown", 0.5, covered=True)
    for _ in range(DEGRADED_FRAMES + 2):
        degraded = engine._check_degraded(SimpleNamespace(readings=dialog), degraded)
    assert degraded is True


def test_a_correction_moves_through_the_api(make_panel: Callable[..., PanelEngine]) -> None:
    import chess

    from sidecoach.events import GameStarted, Side
    from sidecoach.profile.profiler import Profiler
    from sidecoach.profile.store import Store

    engine = make_panel()
    api = PanelApi(engine)
    assert api.correct("g", 0, 3, "time pressure") is False  # no session, no profile
    store = Store(None)
    profiler = Profiler(store, session_id="s")
    profiler.on_event(
        GameStarted(game_id="g", user_side=Side.WHITE, start_fen=chess.STARTING_FEN), 0.0
    )
    store.add_evidence(
        7,
        game_id="g",
        segment=0,
        ply=3,
        phase="opening",
        expected_loss=0.12,
        cause="positional misjudgment",
        motif=None,
        time_pressure=False,
    )
    engine._profiler = profiler
    engine._set(
        review={
            "game_id": "g",
            "turning_points": [{"segment": 0, "ply": 3, "cause": "positional misjudgment"}],
        }
    )
    assert api.correct("g", 0, 3, "time pressure") is True
    assert store.evidence("g")[0].cause == "time pressure"
    assert engine.poll(0)["state"]["review"]["turning_points"][0]["cause"] == "time pressure"
    with pytest.raises(ValueError):
        api.correct("g", 0, 3, "bad luck")


@pytest.mark.embedder  # a session started builds the stages: the net loads
def test_the_engine_coach_and_listening_outlive_a_session(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """Everything a session needs is built when the panel opens (`warm`), before any
    Start; Start and Stop only start and stop reading the screen (Tai, 2026-09-21), so the
    same engine and coach glue serve one session after another; `close` ends them. No
    game is needed: a session that reads nothing still builds or reuses the stages.
    Listening, too, is left as it was: a session used to turn the listener off on its way
    out, and after a Stop push-to-talk did nothing while the page still showed it on
    (2026-09-23)."""
    engine = make_panel(record=False)
    assert engine._assessing is None and engine._coaching is None
    assert engine.warm_status()["done"] is False
    engine.warm(background=False)
    # The voice models load on after the splash (`_voice_thread`): waited for, not hoped done.
    assert engine._voice_thread is not None
    engine._voice_thread.join(timeout=30)
    assert not engine._voice_thread.is_alive()
    mic = FedMic()
    engine._mic = mic  # type: ignore[assignment]
    engine._listener.mic = mic  # type: ignore[assignment]
    engine._listener.set_mode("ptt")  # what set_listen does, without hooking the keyboard
    engine._set(listen="ptt")
    assessing, coaching = engine._assessing, engine._coaching
    assert assessing is not None and coaching is not None, "warm built nothing"
    assert engine._warm_done.is_set()
    # The splash screen's view of it: eight stages, none left pending or loading.
    warm = engine.warm_status()
    assert warm["done"] is True and len(warm["stages"]) == 8
    assert {s["state"] for s in warm["stages"]} <= {"done", "off"}, warm["stages"]
    assert warm["stages"][-1]["name"] == "ready" and warm["stages"][-1]["state"] == "done"
    assert all("seconds" in s for s in warm["stages"])
    texts = [ln["text"] for ln in engine.poll(0)["lines"]]
    assert sum(t.startswith("engine:") for t in texts) == 1
    assert sum(t.startswith("coach:") for t in texts) == 1

    engine.start()
    assert engine._thread is not None
    engine._thread.join(timeout=30)
    assert not engine._thread.is_alive()
    assert engine._assessing is assessing and engine._coaching is coaching, (
        "a session must use what the warm-up built, and Stop must keep it"
    )
    engine.start()  # the second session: the same stages, nothing rebuilt or re-announced
    assert engine._thread is not None
    engine._thread.join(timeout=30)
    assert engine._assessing is assessing and engine._coaching is coaching
    texts = [ln["text"] for ln in engine.poll(0)["lines"]]
    assert sum(t.startswith("engine:") for t in texts) == 1, texts
    assert sum(t.startswith("coach:") for t in texts) == 1, texts
    assert engine._listener.mode == "ptt" and mic.opened, "a session closed the ear"
    assert engine.poll(0)["state"]["listen"] == "ptt"

    engine.close()
    assert engine._assessing is None and engine._coaching is None
    assert engine._listener.mode == "off" and not mic.opened


def test_after_a_stop_the_coach_says_nothing_unasked_until_the_board_is_read_again(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """The coach outlives a Stop, and kept the position it last read: the first ticks after
    the next Start gave stuck hints about the last session's board (the game-state review,
    2026-10-01). After a Stop it is quiet until the board is read again."""
    from sidecoach.events import Trigger
    from tests.coaching import drain, mate_analysis, offline_coach, play_to_scholar, stuck

    engine = make_panel(record=False)
    engine.warm(background=False)
    assert engine._coaching is not None
    coach = offline_coach()
    play_to_scholar(coach)
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    engine._coaching.coach = coach  # a coach that has read a position
    engine.start()  # a session that reads nothing, then stops
    assert engine._thread is not None
    engine._thread.join(timeout=30)
    assert not engine._thread.is_alive()
    for ts in (100.0, 1_000.0, 100_000.0):
        coach.tick(ts)
    assert drain(coach) == [], "the last session's position"
    play_to_scholar(coach)  # the board read again
    coach.on_event(mate_analysis(6, decisive=False), 7.0)
    assert [m.trigger for m in stuck(coach)] == [Trigger.STUCK]


def test_the_splash_lets_go_before_the_voice_models_are_loaded(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The warm-up was every stage one after another, about eight seconds; the voice and the
    speech recognition, three of them, are needed by no board read (Tai, 2026-10-03: startup
    to "4 or 5" seconds). The splash lets go once the pieces, the vision and the engine are
    ready; the voice models load on after it, and their stages say so when they are done."""
    import threading

    from sidecoach.vision import pieces

    engine = make_panel(record=False)
    release = threading.Event()
    real = engine._warm_voice

    def held_voice() -> None:
        assert release.wait(30), "never released"
        real()

    monkeypatch.setattr(pieces, "embedder", lambda: None)
    monkeypatch.setattr(engine, "_warm_voice", held_voice)
    monkeypatch.setattr(engine, "_warm_engines", lambda: None)
    try:
        engine.warm(background=False)
        status = engine.warm_status()
        assert status["done"] is True, "the splash waited for the voice"
        states = {s["name"]: s["state"] for s in status["stages"]}
        assert states["ready"] == "done" and states["voice"] == "pending", states
        release.set()
        assert engine._voice_thread is not None
        engine._voice_thread.join(timeout=30)
        states = {s["name"]: s["state"] for s in engine.warm_status()["stages"]}
        assert states["voice"] in ("done", "off") and states["mic"] in ("done", "off"), states
    finally:
        release.set()
        engine.close()


def test_the_session_is_ended_before_the_wait_for_the_coachs_replies(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """At a quit the shell's close gives the loop 10 s, and the loop waited up to 30 s for the
    coach's replies before it wrote the session's end: with a reply on its way, the session
    was never ended. The end is written while the coach is still being waited for."""
    import threading

    from sidecoach.profile.profiler import Profiler

    engine = make_panel(record=False)
    engine.warm(background=False)
    ended = threading.Event()
    waiting = threading.Event()
    release = threading.Event()
    ended_while_waiting: list[bool] = []
    real_end = Profiler.end_session

    def end_session(self: Profiler, ts: float) -> str | None:
        ended.set()
        return real_end(self, ts)

    def coach_wait(*_: object, **__: object) -> None:
        ended_while_waiting.append(ended.is_set())
        waiting.set()
        release.wait(30)  # a reply on its way, until the test lets it land

    monkeypatch.setattr(Profiler, "end_session", end_session)
    prepare = engine._prepare_stages

    def stages() -> tuple[object, object]:
        assessing, coaching = prepare()
        coaching.wait = coach_wait
        return assessing, coaching

    monkeypatch.setattr(engine, "_prepare_stages", stages)
    engine.start()  # a session that reads nothing, then ends
    try:
        assert waiting.wait(30), "the loop waits for the coach"
        assert ended_while_waiting == [True], "the session's end comes first"
    finally:
        release.set()
        assert engine._thread is not None
        engine._thread.join(timeout=30)
    assert not engine._thread.is_alive()


def test_a_coach_message_is_handled_once_across_sessions(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """The coach outlives a Stop, so what of its messages the panel has handled lives as long
    as the coach: a count started again at each Start sent every earlier session's messages
    through again -- logged, recorded into the game's dialogue, spoken when their ply matched
    (2026-10-01). A new coach's are all new; so are a prompt check's failures."""
    from types import SimpleNamespace

    from sidecoach.coach.coach import Coach, Spoken
    from sidecoach.events import CoachMessage, DisclosureLevel, Trigger
    from sidecoach.llm.offline import OfflineProvider

    engine = make_panel(record=False)
    said: list[str] = []
    engine._speak = lambda trigger, text, ply, moved_on=False, unvoiced=False: (
        said.append(text) or "text"
    )  # type: ignore[method-assign]
    coach = Coach(OfflineProvider())
    coaching = SimpleNamespace(coach=coach)
    message = CoachMessage(
        text="Look at the knight.", level=DisclosureLevel.IDEA, trigger=Trigger.STUCK, ply=0
    )
    once = Spoken(message=message, level=DisclosureLevel.IDEA, problems=[], attempts=1)
    coach.spoken.append(once)
    coach.harness_problems.append("stuck at ply 0: no position now")
    for _session in range(2):  # the end of one session, then the next
        engine._coach_meta(coaching)
    assert said == ["Look at the knight."], "spoken once"
    lines = [ln["text"] for ln in engine.poll(0)["lines"]]
    assert sum("prompt check failed" in t for t in lines) == 1, lines

    fresh = Coach(OfflineProvider())  # a new coach (settings changed): its list from the start
    fresh.spoken.append(once)
    engine._coach_meta(SimpleNamespace(coach=fresh))
    assert said == ["Look at the knight."] * 2


@pytest.mark.embedder  # a session started builds the stages: the net loads
def test_a_start_without_a_warm_up_builds_the_stages_once(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """The CLI's replay and the tests start without a warm-up: the session builds the
    stages itself, and keeps them for the next one."""
    engine = make_panel(record=False)
    engine.start()
    assert engine._thread is not None
    engine._thread.join(timeout=30)
    assessing = engine._assessing
    assert assessing is not None
    engine.start()
    assert engine._thread is not None
    engine._thread.join(timeout=30)
    assert engine._assessing is assessing
    engine.close()


def test_an_engine_installed_after_the_warm_up_is_built_at_the_next_start(
    make_panel: Callable[..., PanelEngine], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A first run warms up before Setup has fetched Stockfish; the stage built then has
    no engine. Once Setup has put one in place, the next Start builds the engine -- and
    the coach, which was built knowing there was none -- rather than waiting for a
    restart."""
    import sidecoach.cli as cli_mod
    import sidecoach.config as config_mod

    built: list[bool] = []

    class _Stage:
        def __init__(self, _settings: Settings, enabled: bool, **_: object) -> None:
            found = config_mod.find_stockfish(_settings) is not None
            self.on = enabled and found
            # A real analyser carries its annotator, and a built one here has its book.
            annotator = type("Annotator", (), {"book": object()})()
            self.analyser = type("Analyser", (), {"annotator": annotator})() if self.on else None
            self.detail = "Stockfish, depth 18" if self.on else "no Stockfish found"
            self.computed = None
            self.closed = False
            built.append(self.on)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(cli_mod, "_Assessing", _Stage)
    monkeypatch.setattr(config_mod, "find_stockfish", lambda _s: None)
    engine = make_panel(analysis=True, record=False)
    first, coaching = engine._prepare_stages()
    assert not first.on
    assert engine._prepare_stages()[0] is first, "nothing new to find: kept"

    monkeypatch.setattr(config_mod, "find_stockfish", lambda _s: tmp_path / "stockfish")
    second, recoached = engine._prepare_stages()
    assert second is not first and second.on, "the engine Setup installed is built"
    assert first.closed
    assert recoached is not coaching, "the coach is built again with the engine"
    assert engine._prepare_stages()[0] is second, "and kept from then on"
    assert built == [False, True]


def test_opening_tables_fetched_after_the_warm_up_reach_the_kept_engine(
    make_panel: Callable[..., PanelEngine], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The engine is built at the warm-up; Setup may fetch the opening tables after it. The
    next Start gives the kept annotator the book, without building anything again."""
    import shutil

    import sidecoach.cli as cli_mod
    from sidecoach.analysis.annotator import GameAnnotator

    class _Stage:
        def __init__(self, _settings: Settings, enabled: bool, **_: object) -> None:
            self.on = enabled
            annotator = GameAnnotator(engine=FakeEngine({}))
            self.analyser = type("A", (), {"annotator": annotator})()
            self.computed = annotator.computed
            self.detail = "Stockfish, depth 18"

        def close(self) -> None:
            return

    monkeypatch.setattr(cli_mod, "_Assessing", _Stage)
    engine = make_panel(analysis=True, record=False)
    first, _ = engine._prepare_stages()
    assert first.analyser.annotator.book is None, "no tables yet"

    folder = tmp_path / "openings"
    folder.mkdir()
    shutil.copy(Path(__file__).parent / "openings_sample.tsv", folder / "c.tsv")
    again, _ = engine._prepare_stages()
    assert again is first, "the engine is kept, not built again"
    assert again.analyser.annotator.book is not None, "the book the tables make"


def test_the_panel_starts_with_the_settings_voice(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    """The saved voice is the one the panel speaks with from the start (Tai, 2026-09-21:
    the settings said bm_george, the panel came up with af_sarah -- the field's old
    default always won). A caller may still name one, as a test does."""
    settings = Settings(data_dir=tmp_path, voice=VoiceSettings(voice="bm_george"))
    engine = make_panel(settings, record=False)
    assert engine.voice == "bm_george"
    assert engine.poll(0)["state"]["voice"] == "bm_george"
    named = make_panel(settings, record=False, voice="af_bella")
    assert named.voice == "af_bella"


def test_recording_is_a_saved_opt_in_applied_on_the_spot(
    make_panel: Callable[..., PanelEngine], tmp_path: Path
) -> None:
    """The app starts the sidecar with --no-record --no-persist; the setting
    capture.record (Settings > Screen, or Debug > Record sessions, the same switch) turns
    recording on for the running session or the next Start, and stays until turned off
    (Tai, 2026-09-21: off by default, but there must be a way to record a bug;
    2026-09-22: an option in Settings, working on the spot)."""
    from sidecoach.config import load_settings

    settings = Settings(data_dir=tmp_path)
    settings._source = tmp_path / "sidecoach.toml"
    engine = make_panel(settings, record=False, persist=False)
    state = engine.poll(0)["state"]
    assert state["record_wanted"] is False and state["recording_dir"] is None
    # The Debug menu writes the setting.
    reply = engine.set_record(True)
    assert reply == {"ok": True, "on": True, "running": False, "folder": None}
    assert engine.poll(0)["state"]["record_wanted"] is True
    assert load_settings(tmp_path / "sidecoach.toml").capture.record is True
    # The session folder follows the setting, not the launch flag.
    assert engine._session_dir(live=True).parent.name == "recordings"
    # Settings > Save applies at once.
    new = settings.model_copy(deep=True)
    new.capture.record = False
    assert "capture.record" in engine.apply_settings(new, ["capture.record"])
    assert engine.poll(0)["state"]["record_wanted"] is False
    assert engine._session_dir(live=True).parent.name == "sessions"
    assert any("from the next Start" in ln["text"] for ln in engine.poll(0)["lines"])


class _Recogniser:
    """Speech recognition that reads the words off the clip's length: one word per 0.4 s.
    A clip of exactly `held` samples waits for `gate` first (a slow partial)."""

    def __init__(self) -> None:
        import threading

        self.gate = threading.Event()
        self.gate.set()
        self.held = -1

    def transcribe(self, audio: object) -> str:
        n = len(audio)  # type: ignore[arg-type]
        if n == self.held:
            self.gate.wait(2)
        return " ".join(["hello"] * max(1, int(n / 16000 / 0.4)))

    def healthcheck(self) -> tuple[bool, str]:
        return True, "fake"


def _partial_done(engine: PanelEngine) -> None:
    """Wait for the partial recognition running now to end: it holds `_partial_busy`, taken
    before its thread starts and let go once its words are in the box (or dropped)."""
    from tests.fakes import HANG_S

    assert engine._partial_busy.acquire(timeout=HANG_S), "the partial never ended"
    engine._partial_busy.release()


def test_dictation_streams_into_the_box_and_sends_after_a_second_of_quiet(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """What is said reaches the page as it is recognised, and is asked of the coach a second
    after the utterance ends -- unless more is said first, which appends, or the player types
    over it, which keeps the words in the box and stops the send (Tai, 2026-09-22: no
    'heard:' line, the words in the text box, streamed, then sent). The second of quiet is
    the panel's timer on a clock the test moves (tests/fakes.py ManualClock); a partial runs on
    a thread of its own, waited on through the lock it holds while it works."""
    import numpy as np

    from sidecoach.ui import panel as panel_mod
    from tests.fakes import ManualClock

    engine = make_panel(record=False)
    clock = ManualClock()
    monkeypatch.setattr(panel_mod, "time", clock.time_module())
    monkeypatch.setattr(panel_mod, "threading", clock.threading_module())
    quiet = panel_mod.DICTATION_SEND_DELAY_S
    engine._stt = _Recogniser()  # type: ignore[assignment]
    engine._state["running"] = True  # a session: the ask is queued for its loop

    def dictation() -> dict:
        return engine.poll(0)["state"]["dictation"]

    assert dictation() == {"text": "", "live": False, "sends_at": None}
    # The key goes down: the box says it is listening.
    engine._dictation_hold()
    assert dictation()["live"] is True
    # Partials while the key is held: the words so far, live.
    engine._on_partial(np.zeros(16000, dtype=np.float32))
    _partial_done(engine)
    assert dictation() == {"text": "hello hello", "live": True, "sends_at": None}
    # The key comes up: the final result replaces the partial and the send is armed.
    engine._on_utterance(np.zeros(16000 * 2, dtype=np.float32), 2.0)
    d = dictation()
    assert d["text"] == "hello hello hello hello hello" and d["live"] is False
    assert d["sends_at"] is not None
    assert d["sends_at"] == int((clock.now + quiet) * 1000), "a second after the utterance"
    assert engine._commands.empty()
    # Pressed again before it sends: it waits, and the next utterance appends.
    clock.advance(quiet / 2)
    engine._dictation_hold()
    assert dictation()["sends_at"] is None
    assert clock.pending() == [], "the send that was armed is called off"
    engine._on_utterance(np.zeros(16000 // 2, dtype=np.float32), 0.5)
    assert dictation()["text"] == "hello hello hello hello hello hello"
    # ...and after the quiet, one ask with everything, and the box is clear.
    ended = clock.now
    clock.advance_to(ended + quiet - 0.01)
    assert engine._commands.empty(), "not before a second of quiet"
    clock.advance_to(ended + quiet)
    name, payload = engine._commands.get_nowait()
    assert (name, payload) == ("ask_spoken", "hello hello hello hello hello hello")
    assert dictation() == {"text": "", "live": False, "sends_at": None}
    # A partial still being recognised when its utterance ends is dropped: the final
    # result has those words, and they must not show twice.
    stt = engine._stt
    assert isinstance(stt, _Recogniser)
    stt.held = 16000
    stt.gate.clear()
    engine._on_partial(np.zeros(16000, dtype=np.float32))  # blocks in transcribe
    engine._on_utterance(np.zeros(16000 * 2, dtype=np.float32), 2.0)
    stt.gate.set()
    _partial_done(engine)
    assert dictation()["text"] == "hello hello hello hello hello"
    # Typed over before it sends: the words go to the box to edit, and nothing is asked.
    assert dictation()["sends_at"] is not None
    assert engine.cancel_dictation() == {"ok": True, "text": "hello hello hello hello hello"}
    assert dictation() == {"text": "", "live": False, "sends_at": None}
    assert clock.pending() == [], "the send is called off"
    clock.advance(10 * quiet)  # past the send delay
    assert engine._commands.empty()
    # No 'heard:' line at info level any more: the words are on the page, not in the terminal.
    assert not any(
        ln["text"].startswith("heard:") and ln["level"] == "info" for ln in engine.poll(0)["lines"]
    )


def test_a_click_after_the_words_does_not_hold_their_send(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Words said, then a click in voice mode before they send: the click starts an
    utterance (the send waits) and is dropped -- the words go on to send, and the box stops
    saying it is listening. They used to wait for good (2026-09-23). The second of quiet is
    the panel's timer on a clock the test moves."""
    import numpy as np

    from sidecoach.ui import panel as panel_mod
    from tests.fakes import ManualClock

    engine = make_panel(record=False)
    clock = ManualClock()
    monkeypatch.setattr(panel_mod, "time", clock.time_module())
    monkeypatch.setattr(panel_mod, "threading", clock.threading_module())
    quiet = panel_mod.DICTATION_SEND_DELAY_S
    engine._stt = _Recogniser()  # type: ignore[assignment]
    engine._state["running"] = True
    engine._on_utterance(np.zeros(16000, dtype=np.float32), 1.0)
    engine._on_speech_start()  # the click begins
    assert engine.poll(0)["state"]["dictation"] == {
        "text": "hello hello",
        "live": True,
        "sends_at": None,
    }
    assert clock.pending() == [], "the send waits while the click lasts"
    engine._on_dropped()  # ...and is dropped
    d = engine.poll(0)["state"]["dictation"]
    assert d["live"] is False and d["sends_at"] is not None
    dropped = clock.now
    clock.advance_to(dropped + quiet - 0.01)
    assert engine._commands.empty(), "not before a second of quiet"
    clock.advance_to(dropped + quiet)
    assert engine._commands.get_nowait() == ("ask_spoken", "hello hello")
    # With nothing said before it, the box simply stops listening.
    engine._on_speech_start()
    engine._on_dropped()
    assert engine.poll(0)["state"]["dictation"] == {"text": "", "live": False, "sends_at": None}


def test_an_ask_before_start_is_answered_at_once_not_kept_for_a_later_session(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """A question typed or spoken with no session running used to sit in the command
    queue and be asked of the coach at the next Start -- minutes later, about another
    game. It gets a plain answer now, and nothing is queued; the captions show it too, where
    a question asked from their buttons looks for its answer."""
    engine = make_panel(record=False)
    PanelApi(engine).ask("what should I play")
    assert engine._commands.empty()
    polled = engine.poll(0)
    lines = polled["lines"]
    assert any(ln["text"] == "you: what should I play" for ln in lines)
    reply = next(ln for ln in lines if ln["chan"] == "coach" and "Press Start first" in ln["text"])
    assert reply["data"]["full"].startswith("Press Start first")
    assert polled["state"]["caption_lines"][-1][1] == reply["data"]["full"]


def test_with_the_ai_coach_on_an_ask_with_no_session_goes_to_the_model(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """Tai, 2026-10-02: with the AI coach on and no session running, a question was answered
    with the canned "Press Start first". It goes to the model now, told that no game is being
    watched; nothing is queued for a later Start. The offline coach keeps its plain line (the
    test above)."""
    import threading

    from sidecoach.coach.coach import NO_BOARD_UNWATCHED, Coach
    from tests.coaching import directive_of, sent
    from tests.fakes import ScriptedProvider

    provider = ScriptedProvider(["The Italian: quick development, a safe king."])
    coach = Coach(provider)
    coach.spawn = lambda work, _name: work()  # the question's worker, run here

    class Coaching:  # the panel's coach stage, its replies drained as `_Coaching.drain`
        def __init__(self) -> None:
            self.coach: Coach | None = coach
            self.on = True

        def ask(self, text: str, ts: float) -> None:
            coach.ask(text, ts)

        def drain(self) -> None:
            coach.speaker.run_pending()
            coach.poll()

    engine = make_panel(record=False)
    engine._coaching = Coaching()
    PanelApi(engine).ask("what opening should I learn?")
    assert engine._commands.empty()
    for thread in threading.enumerate():
        if thread.name == "sidecoach-ask-unwatched":
            thread.join(timeout=60)  # its end is the signal; the 60 s only a safety net
    assert NO_BOARD_UNWATCHED in directive_of(sent(provider)), "told no game is watched"
    lines = engine.poll(0)["lines"]
    assert any(ln["text"] == "you: what opening should I learn?" for ln in lines)
    coach_lines = [ln for ln in lines if ln["chan"] == "coach" and ln["level"] == "info"]
    assert [ln["data"].get("full") for ln in coach_lines if ln["data"].get("full")] == [
        "The Italian: quick development, a safe king."
    ]
    assert not any("Press Start first" in ln["text"] for ln in lines)


def test_with_the_ai_coach_on_an_ask_before_any_coach_is_built_goes_to_the_model(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tai, 2026-10-02 (UI #4): in the first seconds after launch -- the warm-up has built no
    coach yet -- or with no Stockfish (no coach is built without one), a question with the AI
    coach on got "Press Start first". A coach is built for it on demand, and the question goes
    to the model, told no game is watched; a second question finds the same coach."""
    import threading

    from sidecoach.coach.coach import NO_BOARD_UNWATCHED
    from tests.coaching import directive_of
    from tests.fakes import ScriptedProvider

    reply = "The Italian: quick development, a safe king."
    provider = ScriptedProvider(then=reply)
    monkeypatch.setattr("sidecoach.llm.factory.make_provider", lambda *_a, **_k: provider)
    engine = make_panel(record=False, coach=True)
    assert engine._coaching is None, "nothing built yet: the first seconds after launch"

    def ask(text: str) -> None:
        PanelApi(engine).ask(text)
        for thread in threading.enumerate():
            if thread.name == "sidecoach-ask-unwatched":
                thread.join(timeout=60)  # its end is the signal; the 60 s only a safety net

    try:
        ask("what opening should I learn?")
        assert engine._commands.empty()
        asked = [directive_of(messages) for _system, messages in provider.prompts]
        assert any(NO_BOARD_UNWATCHED in task for task in asked), "told no game is watched"
        lines = engine.poll(0)["lines"]
        assert not any("Press Start first" in ln["text"] for ln in lines)
        fulls = [ln["data"].get("full") for ln in lines if ln["chan"] == "coach"]
        assert reply in fulls
        built = engine._on_demand
        ask("and against 1. d4?")
        assert engine._on_demand is built, "the coach built on demand is kept"
    finally:
        engine.close()


def test_the_move_list_badges_follow_takebacks_and_resyncs(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """A takeback's withdrawn moves take their badges with them -- a replacement played at
    the same ply showed the withdrawn move's -- and a resync, which numbers the plies again,
    clears them all; both reach the engine glue (Tai's game, 2026-09-23)."""
    import chess

    from sidecoach.events import (
        GameStarted,
        Label,
        MoveAssessment,
        MovePlayed,
        Resync,
        Severity,
        Side,
        Takeback,
    )

    engine = make_panel()

    class Glue:
        def __init__(self) -> None:
            self.got: list[object] = []

        def submit(self, event: object) -> None:
            self.got.append(event)

    glue = Glue()
    engine._dispatch_game(
        GameStarted(game_id="g", start_fen=chess.STARTING_FEN, user_side=Side.WHITE), glue, chess
    )
    board = chess.Board()
    for ply, san in enumerate(["e4", "e5", "Nf3", "Nc6"], start=1):
        before = board.fen()
        move = board.push_san(san)
        engine._dispatch_game(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=before,
                fen_after=board.fen(),
                by_user=ply % 2 == 1,
            ),
            glue,
            chess,
        )
        engine._dispatch_engine(
            [
                MoveAssessment(
                    game_id="g",
                    ply=ply,
                    uci=move.uci(),
                    expected_loss=0.0,
                    severity=Severity.NONE,
                    best=[],
                    label=Label.BEST,
                )
            ],
            None,
            chess,
        )
    assert set(engine._state["labels"]) == {"1", "2", "3", "4"}
    board.pop()
    board.pop()
    engine._dispatch_game(Takeback(game_id="g", plies=2, fen=board.fen()), glue, chess)
    assert set(engine._state["labels"]) == {"1", "2"}, "the withdrawn moves' badges went"
    assert any(isinstance(e, Takeback) for e in glue.got), "the engine glue was told"
    engine._dispatch_game(
        Resync(game_id="g", fen=board.fen(), reason="position adopted"), glue, chess
    )
    assert engine._state["labels"] == {}, "a resync numbers the plies again"


def test_a_game_ended_by_the_next_one_is_not_offered_to_end_game(
    make_panel: Callable[..., PanelEngine],
) -> None:
    """A game whose end was not read, ended because the start position came back: the page
    said "End game… can set it", and End game ends the game the reconciler holds -- the new
    one. The player did; the new game's moves were refused from then on and the record kept
    a one-move win (the state review, 2026-10-01). Only a game still followed is offered."""
    import chess

    from sidecoach.events import GameEnded, Side
    from sidecoach.state.reconcile import Reconciler
    from tests.games import recogniser
    from tests.synth import GEOM, synthetic_board

    model = recogniser(Side.WHITE)
    rec = Reconciler(bottom=Side.WHITE, user_side=Side.WHITE, settle_frames=2)
    ts = 0.0

    def feed(board: chess.Board, highlight: chess.Move | None = None) -> list[object]:
        nonlocal ts
        ts += 0.1
        frame = synthetic_board(board=board, bottom=Side.WHITE, highlight=highlight)
        return list(rec.update(model.classify_frame(frame, GEOM), ts))

    board = chess.Board()
    events = feed(board)
    # A game: GAME_MIN_PLIES half-moves (fewer, replaced unread, is no game at all).
    for san in ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"):
        move = board.push_san(san)
        events += feed(board, move) + feed(board, move)
    first = rec.game_id
    # The start position back, and a first move played from it: the next game.
    board = chess.Board()
    events += feed(board) + feed(board)
    move = board.push_san("d4")
    events += feed(board, move) + feed(board, move)
    assert rec.game_id != first
    ended = [e for e in events if isinstance(e, GameEnded)]
    assert [(e.game_id, e.result) for e in ended] == [(first, "*")]

    engine = make_panel()

    class Glue:
        def submit(self, event: object) -> None:
            pass

    for event in events:
        engine._dispatch_game(event, Glue(), chess, followed=rec.game_id)
    over = [ln["text"] for ln in engine.poll(0)["lines"] if ln["text"].startswith("game over")]
    assert over == ["game over (new game started); the result was not read"], over
    # The next game, one move in, replaced unread: no game, and nothing said of its end
    # (Tai, 2026-10-05: the puzzles page's start board, ended "a game" of one move).
    replaced = GameEnded(
        game_id=rec.game_id, result="*", reason="board reoriented", pgn="1. d4", final_fen=None
    )
    engine._dispatch_game(replaced, Glue(), chess, followed=None)
    assert not [
        ln
        for ln in engine.poll(0)["lines"]
        if "board reoriented" in ln["text"] and ln["text"].startswith("game over")
    ]
    # A game ended with its result unknown that is still the one followed can be set.
    engine._game_plies = 40  # a game followed to its middle
    kept = GameEnded(game_id="k", result="*", reason="abandoned", pgn=None, final_fen=None)
    engine._dispatch_game(kept, Glue(), chess, followed="k")
    assert engine.poll(0)["lines"][-1]["text"].endswith("End game… can set it")


def test_the_splash_never_waits_for_the_local_model(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup finding 18: the warm-up's one request to a local model server waited for the
    model's first load (up to a minute), and the splash with it. It runs on its own now: the
    warm-up is done while the model's request is still held, unanswered."""
    import threading

    import sidecoach.vision.pieces as pieces

    engine = make_panel(record=False)
    released = threading.Event()
    asked = threading.Event()
    answered = threading.Event()

    def slow_model() -> None:
        asked.set()
        released.wait(10)
        answered.set()

    monkeypatch.setattr(pieces, "embedder", lambda: None)
    monkeypatch.setattr(engine, "_warm_voice", lambda: None)
    monkeypatch.setattr(engine, "_warm_pipeline", lambda: None)
    monkeypatch.setattr(engine, "_warm_engines", lambda: None)
    monkeypatch.setattr(engine, "_warm_model", slow_model)
    try:
        engine.warm(background=False)
        assert not answered.is_set(), "the warm-up waited for the model's answer"
        assert engine.warm_status()["done"] is True
        assert asked.wait(5), "the model is still warmed, in the background"
    finally:
        released.set()
        engine.close()
