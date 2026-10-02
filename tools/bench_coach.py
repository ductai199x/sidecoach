"""How long the coach takes to answer a move, after the vision: from the move's commit to the
coach's first words on the page and to the first sound of its voice, stage by stage.

A game is played to the real panel (`PanelEngine`, everything a live session runs: the engine,
the coach, the model, the voice) from rendered frames at the pace of a game, the player taking
`--think` seconds a move, so the coach speaks rather than holding its words for a fast game. For
every move of the player's it records, on the panel's clock:

- committed:  the panel saw the move (`PanelEngine._note_move`) -- the end of the vision;
- verdict:    the engine's word on the move reached the coach (`MoveAssessment`);
- text:       the coach's first words reached the page (`PanelEngine.caption`);
- sound:      the voice began to play them (`KokoroTTS.on_play`).

The settings are the user's (`sidecoach.toml`), with the model replaced by the local one
(free: no paid call is made) and the data folder by a scratch one; the voice's model files are
read from where the app keeps them. The voice plays into a stream that consumes samples at the
device's pace and makes no sound.

    uv run --no-sync python tools/bench_coach.py
"""

from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))  # tests.synth only; see bench_vision.py

from sidecoach.config import load_settings  # noqa: E402
from sidecoach.events import Analysis, MoveAssessment  # noqa: E402
from sidecoach.ui.panel import PanelEngine  # noqa: E402
from tests.synth import synthetic_board  # noqa: E402

LOCAL_MODEL = "huihui-qwen3.8-27b-abliterated@q5_k_xl"  # Tai's pick for local checks
# The Blackburne Shilling Gambit, the player White: book moves, then 4.Nxe5? and 5.Nxf7??,
# so the coach has both its quick words for a sound move and a model's lesson for a mistake.
GAME = ("e4", "e5", "Nf3", "Nc6", "Bc4", "Nd4", "Nxe5", "Qg5", "Nxf7", "Qxg2", "Rf1", "Qxe4+")
FPS = 10.0
# The coach's words on the player's move (`Trigger.POST_MOVE`, `POST_MISTAKE`).
WORDS_ON_A_MOVE = frozenset({"post_move", "post_mistake"})
# Where the installed app keeps its settings, its voice and its engine (the shell's folder).
INSTALLED_SETTINGS = Path.home() / "AppData" / "Roaming" / "Sidecoach" / "sidecoach.toml"
INSTALLED_DATA = Path.home() / "AppData" / "Local" / "Sidecoach" / "data"


@dataclass
class Move:
    ply: int
    san: str
    committed: float | None = None
    verdict: float | None = None
    early: float | None = None  # the position after the move, analysed at the early depth
    whole: float | None = None  # ... and at the full depth
    request: float | None = None  # the model was asked (a model's word only)
    prompt_chars: int = 0
    text: float | None = None
    sound: float | None = None
    words: str = ""
    kind: str = ""  # "board" (written from the board, no model call) or "model"


@dataclass
class Timeline:
    moves: dict[int, Move] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


class SilentStream:
    """`sounddevice.OutputStream` that asks for samples at the device's pace and plays none:
    the voice's first callback comes as it would on a sound card, without a sound."""

    def __init__(self, samplerate: int, channels: int, dtype: str, device: Any, callback: Any):
        self.samplerate, self.callback = samplerate, callback
        self.closed = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def run() -> None:
            block = 480
            period = block / self.samplerate
            while not self.closed:
                out = np.zeros((block, 1), dtype=np.float32)
                self.callback(out, block, None, None)
                time.sleep(period)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True


def frames(think: float, reply: float) -> Iterator[tuple[int, float, np.ndarray]]:
    """The game as a capture hands it over, at its own pace: the start, then every move held
    for the time its player took over the next one."""
    board = chess.Board()
    frame_id = 0
    begun = time.monotonic()

    def hold(seconds: float, move: chess.Move | None) -> Iterator[tuple[int, float, np.ndarray]]:
        nonlocal frame_id
        image = synthetic_board(board=board, highlight=move)
        for _ in range(max(1, int(seconds * FPS))):
            ts = frame_id / FPS
            delay = begun + ts - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            yield frame_id, ts, image
            frame_id += 1

    yield from hold(3.0, None)
    for ply, san in enumerate(GAME):
        move = board.push_san(san)
        # The player's moves are White's: after one, the opponent replies quickly; after the
        # opponent's, the player thinks.
        yield from hold(reply if ply % 2 == 0 else think, move)
    yield from hold(think, None)


def instrument(engine: PanelEngine, timeline: Timeline) -> None:
    """Stamp each stage on the panel's clock (`time.monotonic`)."""
    note_move = engine._note_move

    def noted(ply: int) -> None:
        with timeline.lock:
            if ply % 2 == 1 and ply in timeline.moves:  # the player's (White's) moves
                timeline.moves[ply].committed = time.monotonic()
        note_move(ply)

    engine._note_move = noted  # type: ignore[method-assign]

    def word_on(trigger: Any, ply: int | None) -> Move | None:
        """The move a word is about: a word after a move carries the ply of the position the
        move was played from (`PanelEngine._await_audio`); other words (hints, answers) none."""
        name = getattr(trigger, "value", str(trigger))
        if name not in WORDS_ON_A_MOVE or ply is None:
            return None
        return timeline.moves.get(ply + 1)

    def shown(move: Move, text: str, kind: str) -> None:
        with timeline.lock:
            if move.text is None:
                move.text, move.words, move.kind = time.monotonic(), text, kind

    # The board's own words (a template, a verdict written from the board) come whole.
    speak = engine._speak

    def spoken(trigger: str, text: str, ply: int | None, moved_on: bool = False) -> str:
        move = word_on(trigger, ply)
        if move is not None:
            shown(move, text, "board")
        return speak(trigger, text, ply, moved_on)

    engine._speak = spoken  # type: ignore[method-assign]

    # The model's words come sentence by sentence: the first one is the first text.
    stream_sink = engine._stream_sink

    def streamed(trigger: Any, ply: int | None, moved_on: Any = None) -> Any:
        sink = stream_sink(trigger, ply, moved_on)
        move = word_on(trigger, ply)
        if sink is None or move is None:
            return sink

        def first(sentence: str) -> bool:
            shown(move, sentence, "model")
            return sink(sentence)

        return first

    engine._stream_sink = streamed  # type: ignore[method-assign]

    # The line the voice is given for a move, then its first sound.
    lines: dict[str, Move] = {}
    await_audio = engine._await_audio

    def awaited(trigger: str, ply: int | None, line: str) -> None:
        move = word_on(trigger, ply)
        if move is not None:
            with timeline.lock:
                lines.setdefault(line, move)
        await_audio(trigger, ply, line)

    engine._await_audio = awaited  # type: ignore[method-assign]

    def hook_voice() -> None:
        while getattr(engine._tts, "on_play", None) is None:
            time.sleep(0.05)
        on_play = engine._tts.on_play  # type: ignore[attr-defined]

        def played(line: str) -> None:
            with timeline.lock:
                move = lines.pop(line, None)
                if move is not None and move.sound is None:
                    move.sound = time.monotonic()
            on_play(line)

        engine._tts.on_play = played  # type: ignore[attr-defined]

    threading.Thread(target=hook_voice, daemon=True).start()


def watch_requests(timeline: Timeline) -> None:
    """When the coach sends a model request, and how long its prompt is: given to the last
    committed move with a verdict and no request yet (the coach asks about the move just
    judged; the bench's game gives it time to answer before the next)."""
    from sidecoach.llm.openai_compat import OpenAICompatProvider

    def stamp(system: str, messages: Any) -> None:
        with timeline.lock:
            judged = [
                m for m in timeline.moves.values() if m.verdict is not None and m.request is None
            ]
            if judged:
                move = max(judged, key=lambda m: m.ply)
                move.request = time.monotonic()
                move.prompt_chars = len(system) + sum(len(m["content"]) for m in messages)

    # A reply written sentence by sentence (the voice on) comes by `chat_stream`, a reply
    # read whole (the voice off) by `chat`.
    for name in ("chat", "chat_stream"):
        original = getattr(OpenAICompatProvider, name)

        def asked(
            self: OpenAICompatProvider,
            system: str,
            messages: Any,
            *args: Any,
            _original: Any = original,
            **kw: Any,
        ) -> Any:
            stamp(system, messages)
            return _original(self, system, messages, *args, **kw)

        setattr(OpenAICompatProvider, name, asked)


def watch_verdicts(engine: PanelEngine, timeline: Timeline) -> None:
    """The engine's verdicts as the coach receives them, once its stages are built."""

    def hook() -> None:
        while engine._coaching is None:
            time.sleep(0.05)
        coach = engine._coaching.coach
        on_event = coach.on_event

        def seen(event: Any, ts: float) -> Any:
            now = time.monotonic()
            with timeline.lock:
                move = timeline.moves.get(getattr(event, "ply", -1))
                if move is not None:
                    if isinstance(event, MoveAssessment) and move.verdict is None:
                        move.verdict = now
                    elif isinstance(event, Analysis):
                        if event.provisional and move.early is None:
                            move.early = now
                        elif not event.provisional and move.whole is None:
                            move.whole = now
            return on_event(event, ts)

        coach.on_event = seen  # type: ignore[method-assign]

    threading.Thread(target=hook, daemon=True).start()


def ms(values: list[float]) -> str:
    if not values:
        return "--"
    return f"median {statistics.median(values) * 1e3:6.0f} ms   max {max(values) * 1e3:6.0f} ms"


def main() -> None:
    parser = argparse.ArgumentParser(description="The coach's latency after the vision.")
    parser.add_argument("--think", type=float, default=10.0, help="seconds the player takes")
    parser.add_argument("--reply", type=float, default=2.0, help="seconds the opponent takes")
    parser.add_argument(
        "--voice",
        choices=("settings", "on", "off"),
        default="settings",
        help="the voice as the settings have it, or on (unmuted) or off (muted)",
    )
    args = parser.parse_args()

    import sounddevice

    # Before anything opens a stream: the voice plays into the silent one.
    sounddevice.OutputStream = SilentStream  # type: ignore[misc,assignment]

    settings = load_settings(INSTALLED_SETTINGS)
    scratch = Path(tempfile.mkdtemp(prefix="bench-coach-"))
    settings = settings.model_copy(
        update={"data_dir": scratch, "tools_dir": INSTALLED_DATA / "tools"}
    )
    settings.voice.model_dir = str(INSTALLED_DATA / "models" / "kokoro")
    if args.voice != "settings":
        settings.voice.muted = args.voice == "off"
    settings.llm.provider = "local"
    settings.llm.model = LOCAL_MODEL
    settings.llm.base_url = None

    # The player's moves, by ply (1 for 1.e4): White's, the even indices of the game.
    timeline = Timeline({i + 1: Move(i + 1, san) for i, san in enumerate(GAME) if i % 2 == 0})
    engine = PanelEngine(settings, frame_source=frames(args.think, args.reply), persist=False)
    instrument(engine, timeline)
    watch_verdicts(engine, timeline)
    watch_requests(timeline)
    engine.start()
    while engine._thread is not None and engine._thread.is_alive():
        time.sleep(0.5)
    engine.stop()
    # The panel's own log of the voice and the coach's model calls, beside the scratch data.
    log = scratch / "panel.log"
    with log.open("w", encoding="utf-8") as out:
        for line in engine.poll(0)["lines"]:  # LogRecord.as_dict()
            if line["chan"] in ("voice", "coach", "llm", "sys"):
                out.write(f"{line['t']:8.2f} {line['chan']:<6} {line['level']:<5} {line['text']}\n")
    print(f"log     {log}")

    print(
        f"model   local {LOCAL_MODEL} (reasoning off), voice {settings.voice.voice}"
        f"{' (muted)' if settings.voice.muted else ''}"
    )
    print(f"pace    the player {args.think:g} s a move, the opponent {args.reply:g} s")
    print(f"{'move':<8}{'by':<7}{'verdict':>10}{'text':>10}{'sound':>10}  words")
    stages: dict[str, dict[str, list[float]]] = {}
    for move in sorted(timeline.moves.values(), key=lambda m: m.ply):
        if move.committed is None:
            print(f"{move.san:<8}never committed")
            continue

        def after(stamp: float | None, base: float) -> str:
            return f"{(stamp - base) * 1e3:8.0f}ms" if stamp is not None else "      --  "

        # A word the model was asked for is the model's, shown whole (the voice off) or
        # sentence by sentence (on).
        kind = "model" if move.request is not None else move.kind or "none"
        print(
            f"{move.san:<8}{kind:<7}{after(move.verdict, move.committed)}"
            f"{after(move.text, move.committed)}{after(move.sound, move.committed)}"
            f"  {move.words[:56]}"
        )
        if move.request is not None:
            print(f"{'':15}prompt {move.prompt_chars:,} characters")
        by_kind = stages.setdefault(kind, {})
        for name, start, end in (
            ("committed->verdict", move.committed, move.verdict),
            ("committed->early", move.committed, move.early),
            ("committed->whole", move.committed, move.whole),
            ("verdict->request", move.verdict, move.request),
            ("whole->request", move.whole, move.request),
            ("request->text", move.request, move.text),
            ("committed->text", move.committed, move.text),
            ("committed->sound", move.committed, move.sound),
            ("text->sound", move.text, move.sound),
        ):
            if start is not None and end is not None:
                by_kind.setdefault(name, []).append(end - start)
    for kind, named in stages.items():
        print(f"-- words by the {kind}")
        for name, values in named.items():
            print(f"   {name:<20} {ms(values)}")


if __name__ == "__main__":
    main()
