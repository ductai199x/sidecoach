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


def test_ending_a_game_on_the_users_word_scores_it_from_their_side() -> None:
    outcome = PanelEngine.outcome_result
    assert outcome("resigned", "white") == ("0-1", "user resigned or lost on time")
    assert outcome("resigned", "black") == ("1-0", "user resigned or lost on time")
    assert outcome("won", "white") == ("1-0", "opponent resigned or lost on time")
    assert outcome("won", "black") == ("0-1", "opponent resigned or lost on time")
    assert outcome("draw", "white") == ("1/2-1/2", "draw agreed")
    assert outcome("forget", "black")[0] == "*"


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


def test_dictation_streams_into_the_box_and_sends_after_a_second_of_quiet(
    make_panel: Callable[..., PanelEngine], monkeypatch: pytest.MonkeyPatch
) -> None:
    """What is said reaches the page as it is recognised, and is asked of the coach a second
    after the utterance ends -- unless more is said first, which appends, or the player types
    over it, which keeps the words in the box and stops the send (Tai, 2026-09-22: no
    'heard:' line, the words in the text box, streamed, then sent)."""
    import time

    import numpy as np

    from sidecoach.ui import panel as panel_mod

    monkeypatch.setattr(panel_mod, "DICTATION_SEND_DELAY_S", 0.3)
    engine = make_panel(record=False)
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
    deadline = time.monotonic() + 2
    while dictation()["text"] != "hello hello" and time.monotonic() < deadline:
        time.sleep(0.02)
    assert dictation() == {"text": "hello hello", "live": True, "sends_at": None}
    # The key comes up: the final result replaces the partial and the send is armed.
    engine._on_utterance(np.zeros(16000 * 2, dtype=np.float32), 2.0)
    d = dictation()
    assert d["text"] == "hello hello hello hello hello" and d["live"] is False
    assert d["sends_at"] is not None
    assert engine._commands.empty()
    # Pressed again before it sends: it waits, and the next utterance appends.
    engine._dictation_hold()
    assert dictation()["sends_at"] is None
    engine._on_utterance(np.zeros(16000 // 2, dtype=np.float32), 0.5)
    assert dictation()["text"] == "hello hello hello hello hello hello"
    # ...and after the quiet, one ask with everything, and the box is clear.
    name, payload = engine._commands.get(timeout=2)
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
    time.sleep(0.2)
    assert dictation()["text"] == "hello hello hello hello hello"
    # Typed over before it sends: the words go to the box to edit, and nothing is asked.
    assert dictation()["sends_at"] is not None
    assert engine.cancel_dictation() == {"ok": True, "text": "hello hello hello hello hello"}
    assert dictation() == {"text": "", "live": False, "sends_at": None}
    time.sleep(0.4)  # past the send delay
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
    saying it is listening. They used to wait for good (2026-09-23)."""
    import numpy as np

    from sidecoach.ui import panel as panel_mod

    monkeypatch.setattr(panel_mod, "DICTATION_SEND_DELAY_S", 0.3)
    engine = make_panel(record=False)
    engine._stt = _Recogniser()  # type: ignore[assignment]
    engine._state["running"] = True
    engine._on_utterance(np.zeros(16000, dtype=np.float32), 1.0)
    engine._on_speech_start()  # the click begins
    assert engine.poll(0)["state"]["dictation"] == {
        "text": "hello hello",
        "live": True,
        "sends_at": None,
    }
    engine._on_dropped()  # ...and is dropped
    d = engine.poll(0)["state"]["dictation"]
    assert d["live"] is False and d["sends_at"] is not None
    assert engine._commands.get(timeout=2) == ("ask_spoken", "hello hello")
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
