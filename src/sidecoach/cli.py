"""Command-line entry point."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import platform
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TextIO, cast

import typer
from rich.console import Console
from rich.table import Table

from sidecoach import __version__
from sidecoach.capture.base import primary_monitor
from sidecoach.config import Settings, find_stockfish, load_settings
from sidecoach.events import Region, Side

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import chess

    from sidecoach.analysis.annotator import Analyser, Book, Tagger
    from sidecoach.analysis.chesscom import Overview
    from sidecoach.capture.base import CaptureBackend
    from sidecoach.coach.coach import Coach, DraftCheck
    from sidecoach.events import Event, MoveAssessment
    from sidecoach.replay.recording import Recording
    from sidecoach.vision.theme_cache import ThemeCache

app = typer.Typer(
    help="Sidecoach: vision-first chess coaching companion.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="Path to sidecoach.toml")]


def _check_vision_libs() -> tuple[str, str]:
    try:
        import cv2
        import numpy
    except Exception as exc:  # noqa: BLE001
        return "error", str(exc)
    return "ok", f"opencv {cv2.__version__}, numpy {numpy.__version__}"


def _check_video_codec() -> tuple[str, str]:
    import tempfile

    import cv2

    from sidecoach.replay.recording import FOURCC

    with tempfile.TemporaryDirectory() as probe_dir:
        writer = cv2.VideoWriter(
            str(Path(probe_dir) / "codec_probe.mkv"),
            cv2.VideoWriter.fourcc(*FOURCC),
            10.0,
            (64, 64),
        )
        ok = writer.isOpened()
        writer.release()
    return (
        ("ok", f"{FOURCC} lossless writer available") if ok else ("error", f"{FOURCC} unavailable")
    )


def _check_engine(settings: Settings) -> tuple[str, str]:
    path = find_stockfish(settings)
    if path is None:
        return "missing", "run: uv run python scripts/fetch_stockfish.py"
    try:
        import chess

        from sidecoach.analysis.engine import Engine

        started = time.perf_counter()
        with Engine(
            path=path,
            threads=settings.engine.threads,
            hash_mb=settings.engine.hash_mb,
            multipv=settings.engine.multipv,
        ) as engine:
            name = engine.name
            lines = engine.analyse(chess.Board(), depth=settings.engine.depth)
        took = (time.perf_counter() - started) * 1000.0
        if not lines:
            return "error", f"{name} returned no lines"
        wdl = "win-draw-loss" if lines[0].wdl else "centipawns only"
        return (
            "ok",
            f"{name}, depth {settings.engine.depth} in {took:.0f} ms, {wdl}",
        )
    except Exception as exc:  # noqa: BLE001
        return "error", f"{path}: {exc}"


def _check_openings(settings: Settings) -> tuple[str, str]:
    from sidecoach.analysis.openings import OpeningBook

    started = time.perf_counter()
    book = OpeningBook.load(settings.data_dir / "openings")
    if book is None:
        return "missing", "run: uv run python scripts/fetch_openings.py"
    took = (time.perf_counter() - started) * 1000.0
    return "ok", f"{len(book)} named lines, loaded in {took:.0f} ms"


def _check_llm(settings: Settings) -> tuple[str, str]:
    from sidecoach.llm.factory import make_provider

    try:
        provider = make_provider(settings.llm, timeout_s=15)
    except Exception as exc:  # noqa: BLE001
        return "error", str(exc)
    ok, detail = provider.healthcheck()
    return ("ok" if ok else "error"), f"{provider.name}: {detail}"


def _check_capture(backend: str = "auto") -> tuple[str, str]:
    try:
        from sidecoach.capture.wgc_backend import make_capture

        cap = make_capture(backend)
        try:
            monitors = cap.monitors()
            primary = primary_monitor(monitors)
            frame = cap.grab(monitors[primary])
        finally:
            cap.close()
    except Exception as exc:  # noqa: BLE001
        return "error", str(exc)
    sizes = ", ".join(
        f"{i}: {m.w}x{m.h}" + (" (primary)" if i == primary else "") for i, m in enumerate(monitors)
    )
    name = getattr(cap, "name", "capture")
    return (
        "ok",
        f"{name}; {len(monitors)} monitor(s) [{sizes}]; grab ok {frame.shape[1]}x{frame.shape[0]}",
    )


@app.command()
def doctor(config: ConfigOpt = None) -> None:
    """Check Python, vision libraries, video codec, Stockfish, the LLM, and screen capture."""
    settings = load_settings(config)
    checks: list[tuple[str, str, str]] = [
        ("Python", "ok", f"{platform.python_version()} ({sys.executable})"),
        ("Vision libs", *_check_vision_libs()),
        ("Video codec", *_check_video_codec()),
        ("Stockfish", *_check_engine(settings)),
        ("Openings", *_check_openings(settings)),
        ("LLM", *_check_llm(settings)),
        ("Capture", *_check_capture()),
    ]
    table = Table(title=f"sidecoach {__version__} doctor")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    for name, status, detail in checks:
        colour = {"ok": "green", "missing": "yellow"}.get(status, "red")
        table.add_row(name, f"[{colour}]{status}[/{colour}]", detail)
    console.print(table)
    if any(status == "error" for _, status, _ in checks):
        raise typer.Exit(code=1)


@app.command()
def snap(
    monitor: Annotated[int | None, typer.Option(help="Monitor index; default all")] = None,
    out: Annotated[Path | None, typer.Option(help="Output folder")] = None,
    config: ConfigOpt = None,
) -> None:
    """Save one PNG screenshot per monitor into data/screenshots."""
    import cv2

    from sidecoach.capture.wgc_backend import make_capture

    settings = load_settings(config)
    folder = out or settings.data_dir / "screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cap = make_capture(settings.capture.backend)
    try:
        monitors = cap.monitors()
        targets = [monitor] if monitor is not None else range(len(monitors))
        for i in targets:
            frame = cap.grab(monitors[i])
            path = folder / f"{stamp}_m{i}.png"
            cv2.imwrite(str(path), frame)
            console.print(f"monitor {i} {monitors[i].w}x{monitors[i].h} -> {path}")
    finally:
        cap.close()


def _monitor_region(cap: CaptureBackend, monitor: int | None) -> tuple[int, Region]:
    """The chosen monitor's index and region; the primary when none was chosen."""
    monitors = cap.monitors()
    if monitor is None:
        monitor = primary_monitor(monitors)
    if not 0 <= monitor < len(monitors):
        console.print(
            f"[red]no monitor {monitor}[/red]; {len(monitors)} found, see `sidecoach doctor`"
        )
        raise typer.Exit(1)
    return monitor, monitors[monitor]


@app.command()
def record(
    monitor: Annotated[
        int | None, typer.Option(help="Monitor index, see doctor; default the primary")
    ] = None,
    seconds: Annotated[float | None, typer.Option(help="Stop after this long")] = None,
    fps: Annotated[int, typer.Option(help="Capture rate")] = 10,
    keyframe: Annotated[float, typer.Option(help="Seconds between forced keyframes")] = 1.0,
    out: Annotated[Path | None, typer.Option(help="Recording folder")] = None,
    config: ConfigOpt = None,
) -> None:
    """Record a monitor to a lossless recording until Ctrl+C or --seconds."""
    from sidecoach.capture.loop import CaptureFailed, capture_loop
    from sidecoach.capture.wgc_backend import make_capture
    from sidecoach.replay.recording import Recorder, board_gate
    from sidecoach.vision.tracker import BoardTracker

    settings = load_settings(config)
    folder = out or settings.data_dir / "recordings" / time.strftime("%Y%m%d-%H%M%S")
    cap = make_capture(settings.capture.backend)
    tracker = BoardTracker()

    try:
        monitor, region = _monitor_region(cap, monitor)
        board = tracker.update(cap.grab(region))
        change_region = board_gate(board, region.w, region.h)
        if board:
            console.print(
                f"board found at {board.region} cell {board.cell_px:.1f}px; "
                f"storing frames when {change_region} changes, plus a keyframe every {keyframe}s"
            )
        else:
            console.print("no board found on the first frame; storing every changed frame")
        recorder = Recorder(
            folder,
            monitor=monitor,
            region=region,
            fps=fps,
            change_region=change_region,
            keyframe_interval_s=keyframe,
        )
        console.print(f"recording monitor {monitor} {region.w}x{region.h} at {fps} fps -> {folder}")
        console.print("press Ctrl+C to stop")
        started = time.monotonic()
        try:
            for frame_id, ts, frame in capture_loop(cap, region, fps, max_seconds=seconds):
                board = tracker.update(frame)
                gate = board_gate(board, region.w, region.h)
                if gate != recorder.change_region:
                    recorder.change_region = gate
                    where = f"board now at {board.region}" if board else "board lost"
                    print(f"\n{where}; gating on {gate}", flush=True)
                tick = recorder.write(frame, ts)
                if frame_id % fps == 0:
                    elapsed = time.monotonic() - started
                    print(
                        f"\r{elapsed:6.1f}s ticks={tick.frame_id + 1} stored={recorder.stored} "
                        f"size={recorder.size_bytes() / 1e6:.0f}MB",
                        end="",
                        flush=True,
                    )
        except KeyboardInterrupt:
            pass
        except CaptureFailed as exc:
            print()
            console.print(f"[red]capture failed:[/red] {exc}")
        finally:
            recorder.close()
            print()
        console.print(
            f"done: {recorder.ticks} ticks, {recorder.stored} stored frames, "
            f"{recorder.size_bytes() / 1e6:.1f} MB in {folder}"
        )
    finally:
        cap.close()


@app.command()
def replay(
    recording: Path,
    detect: Annotated[bool, typer.Option(help="Track the board through every tick")] = False,
    moves: Annotated[bool, typer.Option(help="Reconstruct the moves played")] = False,
    pgn: Annotated[Path | None, typer.Option(help="Ground-truth PGN to score against")] = None,
    game: Annotated[int, typer.Option(help="Which game in the recording the PGN scores")] = 1,
    scale: Annotated[
        float, typer.Option(help="Resize every frame by this factor before the pipeline")
    ] = 1.0,
    assess: Annotated[bool, typer.Option(help="Also run the engine over the moves found")] = False,
    coach: Annotated[bool, typer.Option(help="Also let the coach speak, as it would live")] = False,
    depth: Annotated[int | None, typer.Option(help="Engine depth for --assess")] = None,
    profile: Annotated[
        Path | None,
        typer.Option(
            help="With --assess: write the game into this profile database (a record built"
            " from recordings), instead of the memory-only profile a replay keeps"
        ),
    ] = None,
    themes: Annotated[
        bool,
        typer.Option(
            help="Read a board caught mid-game cold from the theme cache (data/themes), as "
            "the panel does; off, only a start position begins a game"
        ),
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Summarise a recording, track the board through it, or reconstruct the game.

    `--scale` replays the same frames smaller or larger, which is how size invariance is
    tested without storing another copy of the recording. `--assess` runs the engine
    alongside, which is how the live watcher's wiring is exercised without a game on screen.
    `--themes` joins a game from the middle through the theme cache, as the panel does;
    off by default so a reconstruction depends on the recording alone.
    """
    from sidecoach.replay.recording import Recording

    settings = load_settings(config)
    rec = Recording(recording)
    console.print(
        f"{recording}: monitor {rec.meta.monitor} {rec.meta.width}x{rec.meta.height} "
        f"at {rec.meta.fps} fps, {len(rec.ticks)} ticks, {rec.stored_count} stored frames, "
        f"started {rec.meta.started_at}"
    )
    if moves or pgn is not None:
        truth = pgn if pgn is not None else recording / "truth.pgn"
        assessing = _Assessing(
            settings, assess or coach, depth=depth, keep_every_position=profile is not None
        )
        coaching = _Coaching(settings, coach and assessing.on)
        cache = None
        if themes:
            from sidecoach.vision.theme_cache import ThemeCache

            cache = ThemeCache(settings.data_dir / "themes")
            console.print(f"theme cache: {cache.directory}")
        try:
            _replay_moves(
                rec,
                truth if truth.exists() else None,
                game,
                scale,
                assessing,
                coaching,
                profile,
                cache=cache,
            )
        finally:
            assessing.close()
            coaching.close()
        return
    if not detect:
        return
    from sidecoach.vision.tracker import BoardTracker

    tracker = BoardTracker()
    last_summary = None
    timings: list[float] = []
    for tick, frame in rec.frames():
        started = time.perf_counter()
        board = tracker.update(frame)
        timings.append(time.perf_counter() - started)
        summary = f"board at {board.region} cell {board.cell_px:.1f}" if board else "no board"
        if summary != last_summary:
            fit = f" fit {board.fit:.2f}" if board else ""
            console.print(f"tick {tick.frame_id:5d} {timings[-1] * 1000:5.0f} ms {summary}{fit}")
            last_summary = summary
    if timings:
        timings.sort()
        console.print(
            f"{len(timings)} ticks, {tracker.detections} full detections, per-tick "
            f"median {timings[len(timings) // 2] * 1000:.1f} ms, "
            f"max {timings[-1] * 1000:.0f} ms"
        )


def _read_truth(truth_pgn: Path) -> list[str]:
    import chess.pgn

    with truth_pgn.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    if game is None:
        return []
    board = game.board()
    sans: list[str] = []
    for move in game.mainline_moves():
        sans.append(board.san(move))
        board.push(move)
    return sans


def _replay_moves(
    rec: Recording,
    truth_pgn: Path | None = None,
    scored_game: int = 1,
    scale: float = 1.0,
    assessing: _Assessing | None = None,
    coaching: _Coaching | None = None,
    profile_path: Path | None = None,
    cache: ThemeCache | None = None,
) -> None:
    """Run the game pipeline over a recording and, if a truth PGN exists, score one game.

    With an `assessing` the engine follows along as well, which is the same wiring the live
    watcher uses and the only way to exercise it without a game on the screen.
    """
    import chess
    import cv2

    from sidecoach.events import (
        BoardUnread,
        GameEnded,
        GameStarted,
        MovePlayed,
        Resync,
        Takeback,
        Unexplained,
    )
    from sidecoach.state.pipeline import GamePipeline
    from sidecoach.vision.clock import ClockReader

    # The profile follows an assessed replay: the review and the evidence are printed.
    # In memory unless `profile_path` says otherwise, since nothing should be learned
    # from a game that was already played -- except on purpose, to build a record from
    # the recordings.
    profiling = (
        _Profiling(load_settings(None), live=False, path=profile_path)
        if assessing is not None and assessing.on
        else None
    )
    if profiling is not None and coaching is not None:
        profiling.attach(coaching)
        coaching.events([profiling.snapshot()], time.monotonic())
    truth = _read_truth(truth_pgn) if truth_pgn is not None else []
    truth_placements: list[str] = []
    if truth:
        replay_board = chess.Board()
        truth_placements.append(replay_board.board_fen())
        for san in truth:
            replay_board.push_san(san)
            truth_placements.append(replay_board.board_fen())
    pipe = GamePipeline(clock=ClockReader(), cache=cache)
    games: list[tuple[str, str]] = []  # (pgn, final fen) of finished games
    bootstraps = 0
    truth_index = 0  # truth plies consumed for the scored game
    # Final-line scoring: committed moves with takebacks applied, in segments split at
    # position resyncs, each aligned to the truth at its start.
    line: list[str] = []
    segments: list[tuple[int, list[str]]] = []
    segment_start = 0
    started = time.perf_counter()
    for tick, frame in rec.frames():
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        events = pipe.process(
            frame, tick.ts, changed=tick.stored is not None, frame_id=tick.frame_id
        )
        if pipe.bootstraps != bootstraps:
            bootstraps = pipe.bootstraps
            bottom = pipe.bottom.value if pipe.bottom else "?"
            console.print(
                f"tick {tick.frame_id:5d} recogniser built from a start position, "
                f"{bottom} at the bottom"
            )
        if assessing is not None:
            assessed = assessing.drain(indent="            ")
            profile_events = (
                profiling.events([*events, *assessed], indent="            ")
                if profiling is not None
                else []
            )
            if coaching is not None:
                coaching.events(events, tick.ts)
                coaching.events(assessed, tick.ts)
                coaching.events(profile_events, tick.ts)
                coaching.tick(tick.ts)
                coaching.drain(indent="            ")
        for event in events:
            # Decided per event: a game-over, a game-start, and the next game's first move
            # can share one tick.
            scoring = bool(truth) and len(games) == scored_game - 1
            if isinstance(event, MovePlayed):
                if assessing is not None:
                    assessing.submit(event)
                who = "you" if event.by_user else "opp"
                verdict = ""
                if scoring:
                    expected = truth[truth_index] if truth_index < len(truth) else None
                    if expected == event.san:
                        verdict = "[green]ok[/green]"
                    else:
                        verdict = f"[yellow]differs, truth {expected}[/yellow]"
                    truth_index += 1
                    line.append(event.san)
                console.print(
                    f"tick {tick.frame_id:5d} ply {event.ply:3d} {event.san:8s} {who} {verdict}"
                )
            elif isinstance(event, Takeback):
                console.print(
                    f"tick {tick.frame_id:5d} [yellow]Takeback {event.plies} ply[/yellow]"
                )
                if scoring:
                    truth_index = max(0, truth_index - event.plies)
                    del line[max(0, len(line) - event.plies) :]
            elif isinstance(event, Unexplained):
                console.print(
                    f"tick {tick.frame_id:5d} [yellow]Unexplained: {event.reason}[/yellow]"
                )
            elif isinstance(event, BoardUnread):
                console.print(
                    f"tick {tick.frame_id:5d} [yellow]Board unread: {event.reason}[/yellow]"
                )
            elif isinstance(event, Resync):
                console.print(f"tick {tick.frame_id:5d} [yellow]Resync: {event.reason}[/yellow]")
                if scoring and event.reason.startswith("position adopted"):
                    placement = event.fen.split()[0]
                    segments.append((segment_start, line))
                    segment_end = segment_start + len(line)
                    line = []
                    hits = [
                        k
                        for k in range(segment_end, len(truth_placements))
                        if truth_placements[k] == placement
                    ]
                    if hits:
                        console.print(
                            f"tick {tick.frame_id:5d} re-aligned to truth after ply {hits[0]} "
                            f"({hits[0] - segment_end} plies skipped)"
                        )
                        truth_index = hits[0]
                        segment_start = hits[0]
                    else:
                        segment_start = segment_end
                        console.print(
                            f"tick {tick.frame_id:5d} [red]adopted position is not in the "
                            "truth game[/red]"
                        )
            elif isinstance(event, GameEnded):
                console.print(f"tick {tick.frame_id:5d} GameEnded {event.result} ({event.reason})")
                if event.pgn:
                    games.append((event.pgn, event.final_fen or ""))
            elif isinstance(event, GameStarted):
                console.print(f"tick {tick.frame_id:5d} GameStarted")
    elapsed = time.perf_counter() - started
    if assessing is not None:
        # The engine and the model fall behind a replay that runs faster than the game did;
        # a replay waits for them, since nothing else is waiting on it.
        assessed = assessing.wait(timeout=1800.0, indent="            ")
        profile_events = (
            profiling.events(assessed, indent="            ", flush=True)
            if profiling is not None
            else []
        )
        if coaching is not None:
            coaching.events(assessed, time.monotonic())
            coaching.events(profile_events, time.monotonic())
            coaching.wait(timeout=1800.0, indent="            ")
            if coaching.on:
                console.print(f"            [dim]{coaching.summary()}[/dim]")
        if profiling is not None:
            profiling.close(indent="            ")
    # A game still in progress when the recording ends is reported as it stands; a finished
    # one was already reported by its game-over event.
    if (
        pipe.reconciler is not None
        and pipe.reconciler.board.move_stack
        and not pipe.reconciler.finished
    ):
        games.append((pipe.reconciler.pgn(), pipe.reconciler.board.fen()))
    if not games:
        console.print("[yellow]no start position seen; nothing reconstructed[/yellow]")
        return
    for i, (game_pgn, final_fen) in enumerate(games, start=1):
        console.print(f"\ngame {i} PGN: {game_pgn}")
        console.print(f"game {i} final FEN: {final_fen}")
    if truth and scored_game <= len(games):
        segments.append((segment_start, line))
        correct = 0
        wrong = 0
        skipped = 0
        previous_end = 0
        for start, sans in segments:
            skipped += max(0, start - previous_end)
            for i, san in enumerate(sans):
                if start + i < len(truth) and truth[start + i] == san:
                    correct += 1
                else:
                    wrong += 1
            previous_end = start + len(sans)
        console.print(
            f"\ntruth: {len(truth)} plies; game {scored_game} final line: {correct} plies "
            f"correct, {wrong} wrong, {skipped} skipped across gaps; takebacks excluded"
        )
    elif truth:
        console.print(f"\n[yellow]game {scored_game} not found in the recording[/yellow]")
    console.print(f"{len(rec.ticks)} ticks in {elapsed:.1f} s")


def _part_spec(part: str) -> tuple[Path, int | None, int | None]:
    """`dir`, `dir:first`, `dir:first:last`, or `dir::last`; a drive letter keeps its colon."""
    head, *tail = part.rsplit(":", 2)
    if len(tail) == 2 and not tail[0].isdigit() and tail[0] != "":
        head, tail = f"{head}:{tail[0]}", tail[1:]
    if len(tail) == 1 and not (tail[0].isdigit() or tail[0] == ""):
        head, tail = f"{head}:{tail[0]}", []
    first = int(tail[0]) if len(tail) > 0 and tail[0] else None
    last = int(tail[1]) if len(tail) > 1 and tail[1] else None
    return Path(head), first, last


@app.command()
def concat(
    out: Path,
    parts: Annotated[
        list[str], typer.Argument(help="Recording dirs, each optionally as dir:first:last")
    ],
) -> None:
    """Join tick ranges of several recordings into one new recording."""
    from sidecoach.replay.recording import concat_recordings

    spec: list[tuple[Path, int | None, int | None]] = []
    for part in parts:
        spec.append(_part_spec(part))
    rec = concat_recordings(spec, out)
    console.print(
        f"{out}: {len(rec.ticks)} ticks, {rec.stored_count} stored frames, "
        f"{rec.meta.width}x{rec.meta.height}"
    )


@app.command()
def shrink(
    out: Path,
    recording: Path,
    short: Annotated[int, typer.Option(help="Target length of the frame's short side")] = 720,
    first: Annotated[int | None, typer.Option(help="First tick to keep")] = None,
    last: Annotated[int | None, typer.Option(help="Last tick to keep")] = None,
) -> None:
    """Re-encode a recording at about 720p, losslessly, keeping the whole screen.

    The frame is scaled so that its short side is `short` pixels (never enlarged). A 1440p
    session becomes 1280x720, a quarter of the size, and still holds everything the program
    sees: the board, the clocks, the move list. It replays as if the game had been played on
    a smaller screen, which is also how size invariance gets tested.
    """
    from sidecoach.replay.recording import Recording, shrink_recording

    src = Recording(recording)
    scale = min(1.0, short / min(src.meta.width, src.meta.height))
    rec = shrink_recording(recording, out, scale, first, last)
    console.print(
        f"{out}: {len(rec.ticks)} ticks, {rec.stored_count} stored frames, "
        f"{rec.meta.width}x{rec.meta.height} at scale {rec.meta.scale:g}"
    )


@app.command()
def cells(
    recording: Path,
    squares: Annotated[str, typer.Option(help="Comma-separated squares, e.g. f5,h6")],
    start: Annotated[int, typer.Option(help="First tick")] = 0,
    end: Annotated[int, typer.Option(help="Last tick")] = 10**9,
    out: Annotated[Path | None, typer.Option(help="Folder for crops")] = None,
) -> None:
    """Print cell readings for chosen squares over a tick range and save their crops.

    Flags after the symbol: * highlighted, ~ hover outline, ! not settled (off centre),
    # covered by an overlay; ?X is the best guess when no symbol is asserted.
    """
    import chess
    import cv2

    from sidecoach.replay.recording import Recording
    from sidecoach.state.pipeline import GamePipeline
    from sidecoach.vision.squares import BoardGeometry, row_col

    rec = Recording(recording)
    wanted = [chess.parse_square(name.strip()) for name in squares.split(",") if name.strip()]
    folder = out or (recording / "cells")
    folder.mkdir(parents=True, exist_ok=True)
    pipe = GamePipeline()
    for tick, frame in rec.frames():
        if tick.frame_id > end:
            break
        pipe.process(frame, tick.ts, changed=tick.stored is not None)
        if tick.frame_id < start or tick.stored is None:
            continue
        if pipe.model is None or pipe.board is None or pipe.bottom is None:
            continue
        geometry = BoardGeometry(pipe.board.region, pipe.board.cell_px)
        readings = pipe.readings
        if readings is None:
            continue
        parts = []
        for sq in wanted:
            r, c = row_col(sq, pipe.bottom)
            crop = geometry.crop(frame, r, c, size=96)
            reading = readings[r][c]
            flags = ("*" if reading.highlighted else "") + ("~" if reading.hover else "")
            flags += "" if reading.settled else "!"
            flags += "#" if reading.covered else ""
            guess = f"?{reading.guess}" if reading.guess and not reading.symbol else ""
            tint = f" tint={reading.tint}" if reading.tint else ""
            parts.append(
                f"{chess.square_name(sq)}={reading.kind}:{reading.symbol or '-'}{guess}{flags} "
                f"fg={reading.fg_fraction:.2f} off={reading.offset:.2f} "
                f"conf={reading.confidence:.2f}{tint}"
            )
            cv2.imwrite(
                str(folder / f"t{tick.frame_id:05d}_{chess.square_name(sq)}.png"),
                cv2.resize(crop, (192, 192), interpolation=cv2.INTER_NEAREST),
            )
        console.print(f"tick {tick.frame_id:5d}  " + "  ".join(parts))
    if pipe.model is None:
        console.print("[yellow]no start position seen[/yellow]")


@app.command()
def detect(
    image: Path,
    out: Annotated[Path | None, typer.Option(help="Write an annotated PNG here")] = None,
) -> None:
    """Run board detection on a screenshot and print the candidates."""
    import cv2

    from sidecoach.vision.board_detect import detect_boards, draw

    frame = cv2.imread(str(image))
    if frame is None:
        console.print(f"[red]could not read {image}[/red]")
        raise typer.Exit(code=1)
    started = time.perf_counter()
    boards = detect_boards(frame)
    elapsed = time.perf_counter() - started
    console.print(
        f"{image.name}: {frame.shape[1]}x{frame.shape[0]}, "
        f"{len(boards)} board(s) in {elapsed * 1000:.0f} ms"
    )
    for i, cand in enumerate(boards):
        r = cand.region
        console.print(
            f"  {i}: x={r.x} y={r.y} side={r.w} cell={cand.cell_px:.2f} "
            f"fit={cand.fit:.3f} score={cand.score:.3f} contrast={cand.contrast:.1f}"
        )
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), draw(frame, boards))
        console.print(f"annotated -> {out}")


def _load_tagger() -> Tagger:
    """The motif tagger: what a move did, in the names a coach uses."""
    from sidecoach.analysis import motifs

    return cast("Tagger", motifs)


def _load_book(settings: Settings) -> Book | None:
    """The opening book, when the tables have been fetched."""
    try:
        from sidecoach.analysis.openings import OpeningBook
    except ImportError:
        return None
    return OpeningBook.load(settings.data_dir / "openings")


def _make_analyser(settings: Settings, *, depth: int | None = None) -> Analyser | None:
    """An analyser with its engine running, or None when there is no engine to run."""
    path = find_stockfish(settings)
    if path is None:
        return None
    from sidecoach.analysis.annotator import open_analyser
    from sidecoach.analysis.score import Thresholds

    return open_analyser(
        path,
        score_model=settings.engine.score_model,
        max_seconds=settings.engine.max_seconds,
        depth=depth or settings.engine.depth,
        multipv=settings.engine.multipv,
        threads=settings.engine.threads,
        hash_mb=settings.engine.hash_mb,
        pv_plies=settings.engine.pv_plies,
        thresholds=Thresholds.from_settings(settings.coach),
        tagger=_load_tagger(),
        book=_load_book(settings),
    )


class _Assessing:
    """The engine alongside a game as it is followed: submit moves, print what comes back.

    Shared by the live watcher and by replay, so both wire the engine up the same way.
    Nothing here blocks: assessments arrive on the worker thread and are printed on the
    tick that finds them, a second or two after the move, which is when a coach would
    speak anyway. Only moves already played are described, never the position on the
    board, which is what the disclosure rules require.
    """

    def __init__(
        self,
        settings: Settings,
        enabled: bool,
        *,
        depth: int | None = None,
        keep_every_position: bool = False,
    ) -> None:
        self.analyser = _make_analyser(settings, depth=depth) if enabled else None
        if self.analyser is not None and keep_every_position:
            # A record built from a recording must judge every move: the replay outruns
            # the engine, and the analyser's queue would otherwise drop the oldest.
            self.analyser.max_pending = 1_000_000
        # A few words on the engine's state, for the panel's log: on (with its depth), or why not.
        if not enabled:
            self.detail = "off"
        elif self.analyser is None:
            self.detail = "no Stockfish found (see `sidecoach doctor`)"
        else:
            self.detail = f"Stockfish, depth {depth or settings.engine.depth}"
        if enabled and self.analyser is None:
            console.print("[yellow]no Stockfish: moves will not be assessed[/yellow]")
        # Where each move was played from, so a better move can be named in the notation a
        # person reads; dropped as soon as that assessment is printed.
        self._from: dict[int, str] = {}
        # The game's board with the moves that led to it, for the opening book: the
        # opening is a fact about the moves, and the annotator asks with them.
        self._game: Any = None
        # An optional sink for the assessment lines (the panel's debug terminal); when set,
        # lines go to it instead of the console.
        self.sink: Callable[[str], None] | None = None
        # The game as it stands now: its id, its last ply, and each ply's move and the
        # position after it -- what a takeback withdraws, and what `drain` holds the engine's
        # words against.
        self._game_id: str | None = None
        self._ply = 0
        self._line: dict[int, str] = {}
        self._fens: dict[int, str] = {}

    @property
    def on(self) -> bool:
        return self.analyser is not None

    def submit(self, event: object) -> None:
        """A move to assess; or a position to measure, when a game begins from the middle
        (a puzzle, an analysis board, a game joined late) or jumps there (a resync): the
        coach's hint and best move need the position's analysis before any move is
        played, and without it the coach had "not seen enough of this game yet"."""
        import chess

        from sidecoach.events import GameStarted, MovePlayed, Resync, Takeback

        if self.analyser is None:
            return
        if isinstance(event, MovePlayed):
            before = self._history_to(event.fen_before)
            move = chess.Move.from_uci(event.uci)
            self.analyser.submit_move(event.game_id, before, move, event.ply, by_user=event.by_user)
            self._from[event.ply] = event.fen_before
            before.push(move)
            self._game = before
            self._game_id = event.game_id
            self._ply = event.ply
            self._line[event.ply] = event.uci
            self._fens[event.ply] = event.fen_after
        elif isinstance(event, Takeback):
            # Moves withdrawn: the engine lets go of them at once, and what it still says
            # about them is not passed on (`drain`).
            ply = max(0, self._ply - event.plies)
            self.analyser.withdraw(event.game_id, ply)
            for gone in [p for p in self._line if p > ply]:
                self._line.pop(gone, None)
                self._fens.pop(gone, None)
                self._from.pop(gone, None)
            self._ply = ply
            self._game = self._history_to(event.fen)
        elif isinstance(event, GameStarted):
            # The engine's queue is the last game's: dropped, so this game is measured
            # from its first move (the last game was reviewed from what had come).
            self.analyser.reset()
            self._from.clear()
            self._forget(event.game_id)
            self._game = chess.Board(event.start_fen)
            if event.start_fen != chess.STARTING_FEN:
                self.analyser.submit_position(event.game_id, chess.Board(event.start_fen), 0)
        elif isinstance(event, Resync):
            self.analyser.reset()
            self._from.clear()
            self._forget(event.game_id)
            # A position adopted, not played to: it has no history to look up.
            self._game = chess.Board(event.fen)
            self.analyser.submit_position(event.game_id, chess.Board(event.fen), 0)

    def _forget(self, game_id: str) -> None:
        """A game begins, or its numbering starts again (a resync): no plies stand yet."""
        self._game_id = game_id
        self._ply = 0
        self._line.clear()
        self._fens.clear()

    def _stale(self, event: object) -> bool:
        """The engine's word on a move or a position the game has since taken back: a ply
        past the game's last, or one whose move or position is no longer the game's. Passed
        on, a withdrawn move's verdict was taken for the replacement played at its ply."""
        from sidecoach.events import Analysis, MoveAssessment

        if not isinstance(event, (MoveAssessment, Analysis)) or event.game_id != self._game_id:
            return False
        if event.ply > self._ply:
            return True
        if isinstance(event, MoveAssessment):
            played = self._line.get(event.ply)
            return played is not None and played != event.uci
        fen = self._fens.get(event.ply)
        return fen is not None and fen.split(" ")[:4] != event.fen.split(" ")[:4]

    def _history_to(self, fen: str) -> Any:
        """The game's board standing on `fen`, with the moves that led there: the next
        move, or a takeback's position further back along them. A position the game never
        stood on starts again from itself, with no history -- as a resync does."""
        import chess

        target = chess.Board(fen)
        walked = self._game.copy() if self._game is not None else None
        while walked is not None:
            if walked.epd() == target.epd():
                return walked
            if not walked.move_stack:
                break
            walked.pop()
        return target

    def reset(self) -> None:
        if self.analyser is not None:
            self.analyser.reset()
            self._from.clear()

    def drain(self, log: TextIO | None = None, indent: str = "  ") -> list[Event]:
        """Print what the engine has finished, log it, and hand it back for the coach."""
        import chess

        from sidecoach.events import MoveAssessment

        if self.analyser is None:
            return []
        drained: list[Event] = []
        for assessed in self.analyser.poll():
            if self._stale(assessed):
                continue
            drained.append(assessed)
            if isinstance(assessed, MoveAssessment):
                fen = self._from.pop(assessed.ply, None)
                if fen is not None:
                    text = _assessment_text(assessed, chess.Board(fen))
                    if self.sink is not None:
                        self.sink(f"ply {assessed.ply} {text}")
                    else:
                        console.print(f"{indent}[dim]ply {assessed.ply}[/dim] {text}")
            if log is not None:
                log.write(assessed.model_dump_json() + "\n")
        return drained

    @property
    def idle(self) -> bool:
        """Nothing queued or being measured: every verdict that will come has come."""
        return self.analyser is None or not self.analyser.pending

    def wait(
        self, timeout: float = 30.0, log: TextIO | None = None, indent: str = "  "
    ) -> list[Event]:
        """Let the engine catch up at the end of a game, so nothing is lost on the way out."""
        if self.analyser is None:
            return []
        drained: list[Event] = []
        deadline = time.monotonic() + timeout
        while self.analyser.pending and time.monotonic() < deadline:
            drained += self.drain(log, indent)
            time.sleep(0.05)
        return drained + self.drain(log, indent)

    def close(self) -> None:
        if self.analyser is not None:
            self.analyser.close()
            self.analyser = None


class _Coaching:
    """The coach alongside a game: events in, messages out, the user's words in between.

    Shared by the live watcher and by replay. The model runs on the coach's own worker
    thread; messages are printed on the tick that finds them.
    """

    def __init__(self, settings: Settings, enabled: bool) -> None:
        self.coach: Coach | None = None
        self.log: TextIO | None = None
        # An optional sink for coach messages: when set (the panel), each message goes to it
        # instead of the terminal console. Left None, messages print as before.
        self.sink: Callable[[object], None] | None = None
        # A few words on the coach's state, for the panel's log: on (with its model), or why not.
        self.detail = "off"
        if not enabled:
            return
        from sidecoach.coach.coach import LESSON_WAIT_S, Coach
        from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
        from sidecoach.llm.factory import make_provider

        # A model that cannot be made (no key yet, a preset unknown) or reached (the local
        # server down) leaves the coach on, offline: the board and the templates carry it
        # until the settings say otherwise (Tai, 2026-09-21: the app works without a model).
        from sidecoach.llm.offline import OfflineProvider

        why = ""
        provider: Any = None
        detail = ""
        try:
            provider = make_provider(settings.llm)
        except Exception as exc:  # noqa: BLE001
            why = f"no model: {exc}"
        else:
            ok, detail = provider.healthcheck()
            if not ok:
                why = f"model not reachable: {detail}"
        if why:
            provider = OfflineProvider()
            _, detail = provider.healthcheck()
            detail = f"{detail}; {why}"
            console.print(f"[yellow]coach offline: {why}[/yellow]")
        policy = InterjectionPolicy(
            PolicySettings(stuck_seconds=settings.coach.stuck_seconds, quiet=settings.coach.quiet)
        )
        # No model (llm.provider none): nothing that would call one -- no claim extraction
        # from the user's words, no verifier; the templates and the board carry the coach.
        offline = bool(getattr(provider, "offline", False))
        self.coach = Coach(
            provider,
            policy=policy,
            draft_check=_draft_check(),
            # The claims in their words are parsed at the routine effort: thinking=False is
            # ignored where the effort is the provider's, and at medium every ask with a claim
            # waited on a second full model call first (the runtime audit, 2026-09-25).
            claims=None if offline else _claims_check(_routine_provider(settings, provider)),
            verifier=None if offline else _verifier(settings, provider),  # type: ignore[arg-type]
            engine=_named_engine(settings),  # type: ignore[arg-type]
            summarise_every=settings.coach.summarise_every,
            compact_record=settings.coach.compact_record,
            routine_provider=_routine_provider(settings, provider),  # type: ignore[arg-type]
            verbalized_sampling=settings.coach.verbalized_sampling,
            local_verdicts=settings.coach.local_verdicts,
            lesson_wait_s=LESSON_WAIT_S,
            persona=settings.coach.persona,
            rating=(settings.profile.rating_low, settings.profile.rating_high),
        )
        self.coach.start()
        self.detail = f"{provider.name} ({detail})"
        console.print(f"[green]coach on[/green]: {provider.name}")

    @property
    def on(self) -> bool:
        return self.coach is not None

    def reprovision(self, settings: Settings) -> None:
        """The model changed in the settings: the providers are rebuilt and handed to the
        running coach, with the same fallback to offline as at start."""
        if self.coach is None:
            return
        from sidecoach.llm.factory import make_provider
        from sidecoach.llm.offline import OfflineProvider

        why = ""
        provider: Any = None
        detail = ""
        try:
            provider = make_provider(settings.llm)
        except Exception as exc:  # noqa: BLE001
            why = f"no model: {exc}"
        else:
            ok, detail = provider.healthcheck()
            if not ok:
                why = f"model not reachable: {detail}"
        if why:
            provider = OfflineProvider()
            _, detail = provider.healthcheck()
            detail = f"{detail}; {why}"
        offline = bool(getattr(provider, "offline", False))
        self.coach.set_providers(
            provider,
            routine_provider=_routine_provider(settings, provider),  # type: ignore[arg-type]
            verifier=None if offline else _verifier(settings, provider),  # type: ignore[arg-type]
            claims=None if offline else _claims_check(_routine_provider(settings, provider)),
        )
        self.detail = f"{provider.name} ({detail})"

    def events(self, events: Sequence[Event], ts: float) -> None:
        if self.coach is None:
            return
        for event in events:
            try:
                self.coach.on_event(event, ts)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]coach error:[/red] {exc!r}")

    def tick(self, ts: float) -> None:
        if self.coach is not None:
            self.coach.tick(ts)

    def ask(self, text: str, ts: float) -> None:
        if self.coach is not None:
            self.coach.ask(text, ts)

    def drain(self, indent: str = "") -> None:
        if self.coach is None:
            return
        for message in self.coach.poll():
            if self.sink is not None:
                self.sink(message)
            else:
                console.print(f"{indent}[bold cyan]coach:[/bold cyan] {message.text}")
            if self.log is not None:
                self.log.write(message.model_dump_json() + "\n")

    def wait(self, timeout: float = 30.0, indent: str = "") -> None:
        if self.coach is None:
            return
        deadline = time.monotonic() + timeout
        while self.coach.speaker.pending and time.monotonic() < deadline:
            self.drain(indent)
            time.sleep(0.05)
        self.drain(indent)

    def summary(self) -> str:
        if self.coach is None:
            return ""
        spoken = self.coach.spoken
        caught = sum(1 for item in spoken if item.problems)
        leaks = sum(1 for item in spoken if item.leaks)
        return (
            f"{len(spoken)} coach messages, {caught} drafts caught by the checks, "
            f"{self.coach.speaker.fallbacks} templates used, {leaks} leaks noted"
        )

    def close(self) -> None:
        if self.coach is not None:
            self.coach.close()
            self.coach = None


class _Profiling:
    """The profile stage alongside a game: the events in, the review and the evidence out
    (to the coach, and to the console). A live session teaches the profile on disk; a
    replay a copy in memory that is forgotten with it."""

    def __init__(self, settings: Settings, live: bool, path: Path | None = None) -> None:
        from sidecoach.profile.profiler import Profiler
        from sidecoach.profile.store import Store

        if path is None and live:
            path = settings.data_dir / "profile.sqlite"
        self.store = Store(path)
        puzzles = None
        if live:
            from sidecoach.profile.puzzles import open_store

            with contextlib.suppress(Exception):
                puzzles = open_store(settings.data_dir)
        self.profiler = Profiler(self.store, ts=time.time(), puzzles=puzzles)

    def events(
        self, events: Sequence[Event], indent: str = "", *, flush: bool = False
    ) -> list[Event]:
        from sidecoach.events import GameReview

        out: list[Event] = []
        try:
            for event in events:
                out += self.profiler.on_event(event, time.time())
            out += self.profiler.flush(time.time()) if flush else self.profiler.tick(time.time())
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]profile error:[/red] {exc!r}")
        for event in out:
            if isinstance(event, GameReview):
                console.print(
                    f"{indent}[bold magenta]review:[/bold magenta] " + " ".join(event.lines)
                )
        return out

    def snapshot(self) -> Event:
        return self.profiler.snapshot()

    def attach(self, coaching: _Coaching) -> None:
        """Let the coach answer questions about the player from the record, its model
        writing the queries (profile/ask.py)."""
        coach = coaching.coach
        if coach is None:
            return
        from sidecoach.profile import ask as profile_ask

        store = self.store
        # The provider as it is when asked: a model changed on Save answers at once.
        coach.profile_ask = lambda q: profile_ask.answer(q, store, coach.speaker.provider)

    def close(self, indent: str = "") -> None:
        try:
            summary = self.profiler.end_session(time.time())
            if summary:
                console.print(f"{indent}[dim]session: {summary}[/dim]")
        finally:
            self.store.close()


def _draft_check() -> DraftCheck | None:
    """The board check over the coach's draft, when the claims module is built."""
    try:
        claims = importlib.import_module("sidecoach.coach.claims")
    except ImportError:
        return None

    def check(
        text: str,
        board: chess.Board,
        after: Sequence[chess.Board] = (),
        us: chess.Color | None = None,
    ) -> list[str]:
        problems = claims.check_draft(text, board, after=after, us=us)
        return [f"{item.text}: {item.reason}" for item in problems]

    return check


def _named_engine(settings: Settings) -> object:
    """A second Stockfish for the coach, for the moves the player names: the annotator's
    engine is on its worker thread with the game's own positions. Bounded so a question
    never waits on a slow position; None when Stockfish is not installed."""
    path = find_stockfish(settings)
    if path is None:
        return None
    from sidecoach.analysis.engine import Engine

    engine = Engine(
        path=path,
        threads=max(1, settings.engine.threads // 2),
        hash_mb=min(128, settings.engine.hash_mb),
        multipv=3,
        pv_plies=settings.engine.pv_plies,
        score_model=settings.engine.score_model,
        max_seconds=3.0,
    )
    engine.open()
    return engine


def _routine_provider(settings: Settings, provider: object) -> object:
    """The coach's model at the effort for mundane replies (see LLMSettings.routine_effort);
    the provider itself when that is unset or the provider has no effort to set."""
    effort = settings.llm.routine_effort
    with_effort = getattr(provider, "with_effort", None)
    if not effort or with_effort is None:
        return provider
    return with_effort(effort)


def _verifier(settings: Settings, provider: object) -> object:
    """The verifier the settings ask for: none, the coach's own model, or another provider."""
    kind = settings.coach.verify.strip().lower()
    if not kind:
        return None
    if kind == "same":
        return provider
    from sidecoach.config import LLMSettings
    from sidecoach.llm.factory import make_provider

    # One that cannot be built -- no key saved, a name mistyped -- leaves the coach without a
    # second opinion, never without a coach: raised here, it stopped the coach from starting
    # (the parity audit, 2026-09-29; the phone's CoachRunner.verifier).
    try:
        return make_provider(
            LLMSettings(provider=kind, model=settings.coach.verify_model), timeout_s=60
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[yellow]no verifier: {exc}[/yellow]")
        return None


def _claims_check(provider: object) -> Callable[[str, chess.Board, object], list[str]] | None:
    """The user's stated reasons, checked against the board, when the module is built."""
    try:
        claims = importlib.import_module("sidecoach.coach.claims")
    except ImportError:
        return None

    def check(text: str, board: chess.Board, facts: object) -> list[str]:
        # Reasoning off: the local model spends the whole budget thinking otherwise, and
        # the claims come out the same in a fifth of the time.
        found = claims.extract_claims(text, board, provider, thinking=False)
        verdicts = claims.verify_claims(found, board, facts=facts)
        # What the board cannot decide tells the model nothing: "whose bishop? mine?" came to
        # it as "unclear, I cannot check that against the board; what do you expect to happen,
        # concretely?" (Tai's puzzle, 2026-09-28).
        return [
            f"{v.claim.text!r}: {v.result}, {v.reason}" for v in verdicts if v.result != "unclear"
        ]

    return check


class _TypedLine:
    """A question typed at the terminal while the watcher runs, one key per tick."""

    def __init__(self) -> None:
        self.buffer: str | None = None

    @property
    def active(self) -> bool:
        return self.buffer is not None

    def open(self) -> None:
        self.buffer = ""
        console.print("[cyan]you:[/cyan] ", end="")

    def feed(self, key: str) -> str | None:
        """Add a key; the finished line when Enter is pressed."""
        assert self.buffer is not None
        if key in ("\r", "\n"):
            print()
            line, self.buffer = self.buffer, None
            return line.strip()
        if key in ("\x08", "\x7f"):
            if self.buffer:
                self.buffer = self.buffer[:-1]
                print("\b \b", end="", flush=True)
            return None
        if key == "\x1b":
            print()
            self.buffer = None
            return None
        self.buffer += key
        print(key, end="", flush=True)
        return None


_SEVERITY_COLOUR = {
    "blunder": "red",
    "mistake": "yellow",
    "inaccuracy": "cyan",
    "none": "green",
}
_LABEL_ORDER = (
    "brilliant",
    "great",
    "best",
    "excellent",
    "good",
    "book",
    "inaccuracy",
    "mistake",
    "miss",
    "blunder",
)
_LABEL_COLOUR = {
    "brilliant": "bright_cyan",
    "great": "blue",
    "best": "green",
    "excellent": "green",
    "good": "green",
    "book": "magenta",
    "inaccuracy": "cyan",
    "mistake": "yellow",
    "miss": "yellow",
    "blunder": "red",
}
# Labels worth a mark in the move table; the rest are the ordinary run of a game.
_NOTABLE = {"brilliant", "great", "inaccuracy", "mistake", "miss", "blunder"}
_QUIET_LABELS = {"best", "excellent", "good"}


def _coloured(label: str) -> str:
    colour = _LABEL_COLOUR.get(label, "white")
    return f"[{colour}]{label}[/{colour}]"


def _print_overview(
    overview: Overview, labels: dict[bool, dict[str, int]], accuracies: dict[bool, list[float]]
) -> None:
    """Our tallies beside chess.com's, side by side."""
    import chess

    from sidecoach.analysis.score import game_accuracy

    table = Table(box=None, pad_edge=False, title="tallies: ours / chess.com")
    table.add_column("")
    table.add_column("white", justify="right")
    table.add_column("black", justify="right")
    for key in _LABEL_ORDER:
        ours_w = labels[chess.WHITE].get(key, 0)
        ours_b = labels[chess.BLACK].get(key, 0)
        theirs_w = overview.counts.get("white", {}).get(key)
        theirs_b = overview.counts.get("black", {}).get(key)
        if not any((ours_w, ours_b, theirs_w, theirs_b)):
            continue
        table.add_row(key, f"{ours_w} / {theirs_w}", f"{ours_b} / {theirs_b}")
    ours = {side: game_accuracy(accuracies[side]) or 0.0 for side in (chess.WHITE, chess.BLACK)}
    theirs = overview.accuracy
    table.add_row(
        "accuracy",
        f"{ours[chess.WHITE]:.1f} / {theirs.get('white', '?')}",
        f"{ours[chess.BLACK]:.1f} / {theirs.get('black', '?')}",
    )
    if overview.rating:
        table.add_row(
            "game rating",
            f"- / {overview.rating.get('white', '?')}",
            f"- / {overview.rating.get('black', '?')}",
        )
    console.print(table)


def _print_agreement(agreements: list[tuple[str, str, str]]) -> None:
    """How our labels line up with chess.com's on the moves either side flagged."""
    if not agreements:
        console.print("no labelled moves to compare")
        return
    same = sum(1 for _, ours, theirs in agreements if ours == theirs)
    console.print(
        f"\n[bold]agreement with chess.com[/bold]: {same} of {len(agreements)} "
        f"labelled moves match exactly"
    )
    table = Table(box=None, pad_edge=False)
    for column in ("move", "ours", "chess.com"):
        table.add_column(column)
    for move, ours, theirs in agreements:
        if ours == theirs:
            continue
        table.add_row(move, _coloured(ours), _coloured(theirs))
    if table.row_count:
        console.print(table)


def _assessment_text(assessment: MoveAssessment, before: chess.Board) -> str:
    """One line about a played move: what it cost, what was better, what it did.

    The best move is named only for a move already played, which is what the disclosure
    rules allow; the live position's own best line is never printed here.
    """
    from sidecoach.analysis.engine import san_line

    colour = _SEVERITY_COLOUR.get(assessment.severity.value, "white")
    parts = [f"loss {assessment.expected_loss:.2f}"]
    if assessment.severity.value != "none":
        parts.append(f"[{colour}]{assessment.severity.value}[/{colour}]")
        best = san_line(before, assessment.best, limit=4)
        if best:
            parts.append(f"better: {best}")
    tags = ", ".join(dict.fromkeys(tag.name for tag in assessment.tags))
    if tags:
        parts.append(f"[dim]{tags}[/dim]")
    return "  ".join(parts)


@app.command()
def review(
    pgn: Annotated[Path, typer.Argument(help="PGN file to annotate")],
    depth: Annotated[int | None, typer.Option(help="Search depth per position")] = None,
    game: Annotated[int, typer.Option(help="Which game in the file")] = 1,
    side: Annotated[str, typer.Option(help="Whose moves to list: white, black, both")] = "both",
    threats: Annotated[bool, typer.Option(help="Also ask what the opponent threatened")] = False,
    quiet: Annotated[bool, typer.Option(help="Only the summary, no move table")] = False,
    out: Annotated[Path | None, typer.Option(help="Write the assessments as JSON")] = None,
    compare: Annotated[
        Path | None,
        typer.Option(help="A saved chess.com review move list (.html or .json) to compare with"),
    ] = None,
    config: ConfigOpt = None,
) -> None:
    """Annotate a game from a PGN: what each move cost, and where the game turned.

    Every position is searched to the same depth, so the loss of a move is the difference
    between two comparable numbers. Losses are in expected score, where 1.0 is a win: a
    0.20 loss is a blunder whatever the material on the board.
    """
    import chess
    import chess.pgn

    from sidecoach.analysis.annotator import GameAnnotator
    from sidecoach.analysis.engine import Engine, san_line
    from sidecoach.analysis.score import Thresholds

    settings = load_settings(config)
    path = find_stockfish(settings)
    if path is None:
        console.print("[red]no Stockfish[/red]; run: uv run python scripts/fetch_stockfish.py")
        raise typer.Exit(1)
    if not pgn.exists():
        console.print(f"[red]no such file[/red] {pgn}")
        raise typer.Exit(1)
    wanted = side.lower()
    if wanted not in ("white", "black", "both"):
        console.print("[red]--side must be white, black, or both[/red]")
        raise typer.Exit(1)

    with pgn.open(encoding="utf-8") as handle:
        parsed = None
        for _ in range(game):
            parsed = chess.pgn.read_game(handle)
            if parsed is None:
                break
    if parsed is None:
        console.print(f"[red]no game {game}[/red] in {pgn}")
        raise typer.Exit(1)

    moves = list(parsed.mainline_moves())
    headers = parsed.headers
    console.print(
        f"[bold]{headers.get('White', '?')} vs {headers.get('Black', '?')}[/bold] "
        f"{headers.get('Result', '*')}, {len(moves)} plies"
    )
    search_depth = depth or settings.engine.depth
    engine = Engine(
        path=path,
        threads=settings.engine.threads,
        hash_mb=settings.engine.hash_mb,
        multipv=settings.engine.multipv,
        pv_plies=settings.engine.pv_plies,
        score_model=settings.engine.score_model,
        max_seconds=settings.engine.max_seconds,
    )
    annotator = GameAnnotator(
        engine=engine,
        depth=search_depth,
        multipv=settings.engine.multipv,
        thresholds=Thresholds.from_settings(settings.coach),
        tagger=_load_tagger(),
        book=_load_book(settings),
        with_threats=threats,
    )
    reference: dict[int, object] = {}
    overview: Overview | None = None
    if compare is not None:
        from sidecoach.analysis import chesscom

        try:
            loaded = chesscom.load(compare)
            chesscom.verify(loaded.moves, moves)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            console.print(f"[red]cannot use {compare}[/red]: {exc}")
            raise typer.Exit(1) from exc
        reference = {item.ply: item for item in loaded.moves}
        overview = loaded.overview
        console.print(
            f"comparing with {compare}: {len(loaded.moves)} plies listed"
            + (", with the overview" if overview else "")
        )

    table = Table(box=None, pad_edge=False)
    columns = ["move", "", "loss", "label", "better", "motifs"]
    if reference:
        columns.insert(4, "chess.com")
    for column in columns:
        table.add_column(column, overflow="fold")

    board = parsed.board()
    counts = {"inaccuracy": 0, "mistake": 0, "blunder": 0}
    total_loss = {chess.WHITE: 0.0, chess.BLACK: 0.0}
    plies = {chess.WHITE: 0, chess.BLACK: 0}
    accuracies: dict[bool, list[float]] = {chess.WHITE: [], chess.BLACK: []}
    labels: dict[bool, dict[str, int]] = {chess.WHITE: {}, chess.BLACK: {}}
    agreements: list[tuple[str, str, str]] = []  # move, ours, theirs
    rows: list[dict[str, object]] = []
    opening: str | None = None
    started = time.perf_counter()
    try:
        with engine:
            for ply, move in enumerate(moves, start=1):
                mover = board.turn
                san = board.san(move)
                assessment, analysis = annotator.assess(
                    headers.get("Site", "review"), board, move, ply
                )
                opening = analysis.opening or opening
                total_loss[mover] += assessment.expected_loss
                plies[mover] += 1
                if assessment.accuracy is not None:
                    accuracies[mover].append(assessment.accuracy)
                ours = assessment.label.value if assessment.label else "?"
                labels[mover][ours] = labels[mover].get(ours, 0) + 1
                if assessment.severity.value in counts:
                    counts[assessment.severity.value] += 1
                show = wanted == "both" or (wanted == "white") == (mover == chess.WHITE)
                number = board.fullmove_number
                label = f"{number}." if mover == chess.WHITE else f"{number}..."
                theirs = getattr(reference.get(ply), "label", None)
                if reference and (theirs is not None or ours in _NOTABLE):
                    agreements.append((f"{label} {san}", ours, theirs or "unlabelled"))
                better = (
                    san_line(board, assessment.best, limit=4)
                    if assessment.severity.value != "none"
                    else ""
                )
                tags = ", ".join(dict.fromkeys(tag.name for tag in assessment.tags))
                rows.append(
                    {
                        "ply": ply,
                        "san": san,
                        "uci": assessment.uci,
                        "loss": round(assessment.expected_loss, 4),
                        "severity": assessment.severity.value,
                        "best": assessment.best,
                        "tags": [tag.model_dump() for tag in assessment.tags],
                        "threats": analysis.threats,
                        "decisive": analysis.decisive,
                        "label": ours,
                        "accuracy": assessment.accuracy,
                        "depth": assessment.depth,
                        "chesscom": theirs,
                        "before_score": round(annotator.measure(board).score, 4),
                        "only_move": annotator.measure(board).only_move,
                    }
                )
                if show and not quiet:
                    colour = _LABEL_COLOUR.get(ours, "white")
                    marker = "*" if ours in _NOTABLE else ""
                    cells = [
                        label,
                        f"{san}{marker}",
                        f"{assessment.expected_loss:.2f}",
                        f"[{colour}]{ours}[/{colour}]" if ours not in _QUIET_LABELS else "",
                        better,
                        tags,
                    ]
                    if reference:
                        theirs_colour = _LABEL_COLOUR.get(theirs or "", "white")
                        cells.insert(
                            4, f"[{theirs_colour}]{theirs}[/{theirs_colour}]" if theirs else ""
                        )
                    table.add_row(*cells)
                board.push(move)
    except KeyboardInterrupt:
        console.print("[yellow]stopped[/yellow]")

    if not quiet:
        console.print(table)
    if opening:
        console.print(f"opening: {opening}")
    from sidecoach.analysis.score import game_accuracy

    for colour, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        if plies[colour]:
            accuracy = game_accuracy(accuracies[colour])
            tally = ", ".join(
                f"{labels[colour].get(key, 0)} {key}"
                for key in _LABEL_ORDER
                if labels[colour].get(key, 0)
            )
            console.print(
                f"{name}: {plies[colour]} moves, accuracy "
                f"{accuracy if accuracy is not None else 0.0:.1f}, "
                f"average loss {total_loss[colour] / plies[colour]:.3f}; {tally}"
            )
    if reference:
        _print_agreement(agreements)
    if overview is not None:
        _print_overview(overview, labels, accuracies)
    console.print(
        f"[cyan]{counts['inaccuracy']} inaccuracies[/cyan], "
        f"[yellow]{counts['mistake']} mistakes[/yellow], "
        f"[red]{counts['blunder']} blunders[/red] "
        f"at depth {search_depth} in {time.perf_counter() - started:.1f}s"
    )
    if out is not None:
        import json

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps({"headers": dict(headers), "opening": opening, "moves": rows}, indent=1),
            encoding="utf-8",
        )
        console.print(f"assessments -> {out}")


@app.command()
def puzzles(
    action: Annotated[
        str,
        typer.Argument(help="fetch: download a sample of the Lichess database; import: load a CSV"),
    ] = "fetch",
    path: Annotated[Path | None, typer.Argument(help="For import: a Lichess puzzle CSV")] = None,
    per_theme: Annotated[int, typer.Option(help="For fetch: puzzles kept per theme")] = 200,
    config: ConfigOpt = None,
) -> None:
    """The puzzles the coach recommends after a game (DESIGN 5.7), under data/puzzles.sqlite.

    `fetch` streams the Lichess puzzle database (CC0, ~300 MB compressed) and keeps a
    sample balanced across the tactical themes (tools/puzzles/fetch.py); `import` loads
    any CSV in its format, the whole database included. The panel opens the store on its
    own and fills it from data/puzzles/*.csv the first time.
    """
    from sidecoach.profile.puzzles import PuzzleStore

    settings = load_settings(config)
    store = PuzzleStore(settings.data_dir / "puzzles.sqlite")
    if action == "fetch":
        import runpy

        if importlib.util.find_spec("zstandard") is None:
            raise typer.BadParameter(
                "the fetch needs zstandard: uv run --with zstandard sidecoach puzzles fetch"
            )
        sys.argv = ["fetch.py", str(per_theme)]
        fetcher = Path(__file__).resolve().parents[2] / "tools" / "puzzles" / "fetch.py"
        runpy.run_path(str(fetcher), run_name="__main__")  # writes data/puzzles/sample.csv
        path = Path("data/puzzles/sample.csv")
    elif action != "import" or path is None:
        raise typer.BadParameter("puzzles fetch | puzzles import <csv>")
    added = store.import_csv(path)
    console.print(f"{added} puzzles added from {path}; {store.count()} in {store.path}")
    store.close()


@app.command()
def profile(
    action: Annotated[str, typer.Argument(help="reset: empty the player model entirely")],
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do it without asking")] = False,
    config: ConfigOpt = None,
) -> None:
    """The player model under data/profile.sqlite: `reset` empties it -- every game, every
    verdict, the taxonomy's counts, the sessions, the puzzle attempts and level. The
    panel's DATA menu does the same."""
    from sidecoach.maintenance import reset_profile

    if action != "reset":
        raise typer.BadParameter("profile reset")
    settings = load_settings(config)
    path = settings.data_dir / "profile.sqlite"
    if not path.exists():
        console.print(f"no profile at {path}")
        return
    if not yes and not typer.confirm(f"Empty the whole player model at {path}?"):
        raise typer.Abort()
    reset_profile(settings)
    console.print(f"profile reset: {path} is empty")


@app.command()
def cache(
    action: Annotated[str, typer.Argument(help="clear: remove the caches")],
    config: ConfigOpt = None,
) -> None:
    """The caches under the data directory: the theme cache (a recogniser per board
    palette, learned again from the next start position) and the puzzle store (built
    again from data/puzzles/*.csv at the next start). `clear` removes them; recordings,
    sessions and the profile are not caches and stay."""
    from sidecoach.maintenance import clear_caches

    if action != "clear":
        raise typer.BadParameter("cache clear")
    done = clear_caches(load_settings(config))
    console.print("; ".join(done) if done else "nothing cached")


@app.command("import-reviews")
def import_reviews(
    page: Annotated[Path, typer.Argument(help="A saved chess.com page, any number of games")],
    recordings: Annotated[Path | None, typer.Option(help="Folder of recordings")] = None,
    config: ConfigOpt = None,
) -> None:
    """File each review in a saved chess.com page beside the recording it belongs to.

    The page is whatever was copied out of the browser's developer tools: move lists and
    overview panels, several games one after another, in any order as long as each
    overview comes before its move list. Each game is matched to a recording by replaying
    its moves against that recording's truth PGN, and written there as chesscom.json,
    which `review --compare` reads.
    """
    from sidecoach.analysis import chesscom

    settings = load_settings(config)
    folder = recordings or settings.data_dir / "recordings"
    if not page.exists():
        console.print(f"[red]no such file[/red] {page}")
        raise typer.Exit(1)
    games = chesscom.split_games(page.read_text(encoding="utf-8", errors="replace"))
    console.print(f"{len(games)} game(s) in {page}")
    truths: list[tuple[Path, list[chess.Move]]] = []
    for pgn in sorted(folder.glob("*/truth.pgn")):
        truths.append((pgn.parent, list(_read_moves(pgn))))
    for index, game in enumerate(games, start=1):
        homes = [home for home, truth in truths if chesscom.match_game(game, truth)]
        summary = f"{len(game.moves)} plies, {sum(1 for m in game.moves if m.label)} labelled" + (
            ", with overview" if game.overview else ", no overview"
        )
        if len(homes) != 1:
            why = "matches no recording" if not homes else f"matches {len(homes)} recordings"
            console.print(f"  game {index}: {summary}; [yellow]{why}[/yellow], skipped")
            continue
        target = homes[0] / "chesscom.json"
        chesscom.save(game, target)
        console.print(f"  game {index}: {summary} -> {target}")


def _read_moves(pgn: Path) -> list[chess.Move]:
    import chess.pgn

    with pgn.open(encoding="utf-8") as handle:
        game = chess.pgn.read_game(handle)
    return list(game.mainline_moves()) if game is not None else []


@app.command()
def watch(
    monitor: Annotated[
        int | None, typer.Option(help="Monitor index, see doctor; default the primary")
    ] = None,
    fps: Annotated[int, typer.Option(help="Capture rate")] = 10,
    record: Annotated[bool, typer.Option(help="Also record the session to disk")] = True,
    board: Annotated[bool, typer.Option(help="Print the board after every move")] = True,
    analysis: Annotated[bool, typer.Option(help="Assess every move with the engine")] = True,
    coach: Annotated[bool, typer.Option(help="Coach through the local model")] = True,
    seconds: Annotated[float | None, typer.Option(help="Stop after this long")] = None,
    out: Annotated[Path | None, typer.Option(help="Recording folder")] = None,
    config: ConfigOpt = None,
) -> None:
    """Watch the screen live: lock onto the board, follow the game, print every move.

    Start it, then begin a new game so the recogniser can bootstrap from the start position.
    With --record (the default) the session is recorded so anything odd can be replayed
    with `sidecoach replay`, and every event is logged beside it. Keys while watching:
    r = you resign, d = draw agreed, n = new game (forget the current one), q = quit.
    With the coach on: h = a hint, m = ask for the move, l = ask for the line, and
    : (colon) to type a question, Enter to send it.

    With --analysis the engine assesses each move once it is played, which is what the
    disclosure rules allow; what to play in the position on the board is never printed.
    The coach sees only what the disclosure level lets it see, and every message is
    checked before it is shown.
    """
    import chess

    from sidecoach.capture.loop import CaptureFailed, capture_loop
    from sidecoach.capture.wgc_backend import make_capture
    from sidecoach.events import (
        BoardUnread,
        GameEnded,
        GameStarted,
        MovePlayed,
        Resync,
        Takeback,
        Unexplained,
    )
    from sidecoach.replay.recording import ChangeGate, Recorder, board_gate
    from sidecoach.state.pipeline import GamePipeline
    from sidecoach.ui.keys import KeyReader
    from sidecoach.vision.clock import ClockReader

    settings = load_settings(config)
    cap = make_capture(settings.capture.backend)
    monitor, region = _monitor_region(cap, monitor)
    pipe = GamePipeline(clock=ClockReader())
    recorder: Recorder | None = None
    folder: Path | None = None
    gate = ChangeGate(keyframe_interval_s=1.0)
    log = None
    if record:
        folder = out or settings.data_dir / "recordings" / time.strftime("%Y%m%d-%H%M%S")
        recorder = Recorder(
            folder, monitor=monitor, region=region, fps=fps, keyframe_interval_s=1.0
        )
        log = (folder / "events.jsonl").open("w", encoding="utf-8")
    console.print(
        f"watching monitor {monitor} {region.w}x{region.h} at {fps} fps"
        + (f", recording to {folder}" if folder else ", not recording")
    )
    console.print("start a new game on the board; keys: r resign, d draw, n new game, q quit\n")

    assessing = _Assessing(settings, analysis)
    coaching = _Coaching(settings, coach and assessing.on)
    coaching.log = log
    profiling = _Profiling(settings, live=True)
    profiling.attach(coaching)
    coaching.events([profiling.snapshot()], time.monotonic())
    typed = _TypedLine()

    locked = False
    bootstraps = 0
    last_move_ts: float | None = None
    last_status = time.monotonic()
    ticks_since_status = 0
    pipeline_ms: list[float] = []
    games: list[str] = []

    def user_result(resigned: bool) -> str:
        if not resigned or pipe.reconciler is None:
            return "1/2-1/2"
        return "0-1" if pipe.reconciler.user_side == Side.WHITE else "1-0"

    try:
        with KeyReader() as keys:
            if not keys.enabled:
                console.print("[yellow]no terminal on stdin: keys are off[/yellow]")
            for frame_id, ts, frame in capture_loop(cap, region, fps, max_seconds=seconds):
                gate.region = board_gate(pipe.board, region.w, region.h)
                if recorder is not None:
                    recorder.change_region = gate.region
                    try:
                        changed = recorder.write(frame, ts).stored is not None
                    except (ValueError, RuntimeError) as exc:
                        # A display change or a full disk must not end the session; the
                        # recording stops, the watching goes on.
                        console.print(f"[red]recording stopped:[/red] {exc}")
                        recorder.close()
                        recorder = None
                        changed = gate.changed(frame, ts)
                else:
                    changed = gate.changed(frame, ts)
                begun = time.perf_counter()
                try:
                    events = pipe.process(frame, ts, changed=changed, frame_id=frame_id)
                except Exception as exc:  # noqa: BLE001
                    # Never let a pipeline bug end a live session; the recording keeps the
                    # evidence and the next frames get a fresh chance.
                    console.print(f"[red]pipeline error at frame {frame_id}:[/red] {exc!r}")
                    events = []
                pipeline_ms.append((time.perf_counter() - begun) * 1000.0)
                ticks_since_status += 1

                key = keys.read()
                if typed.active and key is not None:
                    line = typed.feed(key)
                    if line:
                        coaching.ask(line, ts)
                    key = None
                if key == "q":
                    console.print("quit")
                    break
                if key == ":" and coaching.on:
                    typed.open()
                elif key == "h":
                    coaching.ask("hint", ts)
                elif key == "m":
                    coaching.ask("what should I play", ts)
                elif key == "l":
                    coaching.ask("show me the line", ts)
                if key in ("r", "d"):
                    reason = "user resigned" if key == "r" else "draw agreed"
                    events += pipe.end_game(user_result(key == "r"), reason, ts, frame_id)
                elif key == "n":
                    events += pipe.end_game("*", "abandoned by the user", ts, frame_id)
                    pipe.reset()
                    console.print("[yellow]forgot the game; waiting for a start position[/yellow]")

                if log is not None:
                    for event in events:
                        log.write(event.model_dump_json() + "\n")

                if (pipe.board is not None) != locked:
                    locked = pipe.board is not None
                    if locked and pipe.board is not None:
                        b = pipe.board
                        console.print(
                            f"[green]board locked[/green] at {b.region} cell {b.cell_px:.1f}px"
                        )
                    else:
                        console.print("[yellow]board lost[/yellow]")
                if pipe.bootstraps != bootstraps:
                    bootstraps = pipe.bootstraps
                    bottom = pipe.bottom.value if pipe.bottom else "?"
                    console.print(f"[green]start position seen[/green], {bottom} at the bottom")

                for event in events:
                    if isinstance(event, GameStarted):
                        console.print(
                            f"[bold]new game[/bold] {event.game_id}, "
                            f"you play {event.user_side.value}"
                        )
                        assessing.submit(event)
                        last_move_ts = ts
                    elif isinstance(event, MovePlayed):
                        after = chess.Board(event.fen_after)
                        before = chess.Board(event.fen_before)
                        assessing.submit(event)
                        number = before.fullmove_number
                        dots = "." if before.turn == chess.WHITE else "..."
                        who = "you" if event.by_user else "opponent"
                        elapsed = f" {ts - last_move_ts:.1f}s" if last_move_ts is not None else ""
                        last_move_ts = ts
                        console.print(f"[bold]{number}{dots} {event.san}[/bold]  {who}{elapsed}")
                        if board:
                            console.print(str(after))
                            console.print()
                    elif isinstance(event, Takeback):
                        console.print(f"[yellow]takeback[/yellow] {event.plies} ply")
                        assessing.submit(event)
                    elif isinstance(event, Resync):
                        console.print(f"[yellow]resync[/yellow]: {event.reason}")
                        assessing.submit(event)
                    elif isinstance(event, Unexplained):
                        console.print(f"[yellow]board[/yellow]: {event.reason}; the game is kept")
                    elif isinstance(event, BoardUnread):
                        console.print(f"[yellow]board unread[/yellow]: {event.reason}")
                    elif isinstance(event, GameEnded):
                        console.print(f"[bold]game over[/bold] {event.result} ({event.reason})")
                        if event.pgn:
                            games.append(event.pgn)
                            console.print(f"PGN: {event.pgn}")

                assessed = assessing.drain(log)
                coaching.events(events, ts)
                coaching.events(assessed, ts)
                coaching.events(profiling.events([*events, *assessed]), ts)
                coaching.tick(ts)
                coaching.drain()

                now = time.monotonic()
                if now - last_status >= 5.0:
                    rate = ticks_since_status / (now - last_status)
                    if pipe.board is None:
                        state = "no board"
                    elif pipe.tracker.verified:
                        state = f"locked fit {pipe.board.fit:.2f}"
                    else:
                        state = "lock in doubt"
                    stage = "following" if pipe.model is not None else "waiting for a start"
                    plies = len(pipe.reconciler.board.move_stack) if pipe.reconciler else 0
                    lag = (time.monotonic() - ts) * 1000.0
                    timings = sorted(pipeline_ms) or [0.0]
                    p50 = timings[len(timings) // 2]
                    p95 = timings[min(len(timings) - 1, int(len(timings) * 0.95))]
                    stored = (
                        f", {recorder.stored} frames, {recorder.size_bytes() / 1e6:.0f} MB"
                        if recorder
                        else ""
                    )
                    console.print(
                        f"[dim]{rate:.1f} fps, {state}, {stage}, ply {plies}, "
                        f"pipeline {p50:.0f}/{p95:.0f} ms, lag {lag:.0f} ms{stored}[/dim]"
                    )
                    last_status = now
                    ticks_since_status = 0
                    pipeline_ms = []
    except KeyboardInterrupt:
        pass
    except CaptureFailed as exc:
        # The screen is gone for good (a display removed, a session that stays locked);
        # the game so far is still reported below.
        console.print(f"[red]capture failed:[/red] {exc}")
    finally:
        late = assessing.wait(log=log)
        coaching.events(late, time.monotonic())
        coaching.events(profiling.events(late, flush=True), time.monotonic())
        assessing.close()
        coaching.wait()
        if coaching.on:
            console.print(f"[dim]{coaching.summary()}[/dim]")
        coaching.close()
        profiling.close()
        if recorder is not None:
            recorder.close()
        if log is not None:
            log.close()
        cap.close()
    if (
        pipe.reconciler is not None
        and pipe.reconciler.board.move_stack
        and not pipe.reconciler.finished
    ):
        games.append(pipe.reconciler.pgn())
    for i, pgn in enumerate(games, start=1):
        console.print(f"\ngame {i} PGN: {pgn}")
    if folder is not None:
        console.print(
            f"\nrecording saved to {folder}; replay with: sidecoach replay {folder} --moves"
        )


@app.command()
def panel(
    monitor: Annotated[
        int | None, typer.Option(help="Monitor index, see doctor; default the primary")
    ] = None,
    fps: Annotated[
        int | None, typer.Option(help="Capture rate; the settings' capture.fps by default")
    ] = None,
    port: Annotated[
        int, typer.Option(help="Port for the local panel server; 0 picks a free one")
    ] = 8765,
    replay: Annotated[
        Path | None,
        typer.Option(help="Feed the panel from a recording instead of the live screen (dev)"),
    ] = None,
    browser: Annotated[bool, typer.Option(help="Also open the panel in the browser")] = False,
    record: Annotated[
        bool, typer.Option(help="Record a live session (frames, events and the debug log)")
    ] = True,
    speed: Annotated[
        float, typer.Option(help="Replay pace: 1.0 = as played, 2.0 = twice as fast, 0 = unpaced")
    ] = 1.0,
    loop: Annotated[
        bool, typer.Option("--loop", help="Play the recording again when it ends (a soak)")
    ] = False,
    tts_device: Annotated[
        str | None,
        typer.Option(help="Where the voice runs: cuda (default, falls back to cpu) or cpu"),
    ] = None,
    stt_device: Annotated[
        str | None,
        typer.Option(
            help="Where speech recognition runs: cuda (default, falls back to cpu) or cpu"
        ),
    ] = None,
    listen: Annotated[
        str | None, typer.Option(help="How the coach listens: off, ptt (hold to talk), or vad")
    ] = None,
    persist: Annotated[
        bool,
        typer.Option(
            help="Keep a session folder (the debug log, the recording); --no-persist is the "
            "app's mode: nothing of a session reaches the disk, frames stay in memory"
        ),
    ] = True,
    parent_pid: Annotated[
        int | None,
        typer.Option(help="Exit when this process is gone (the desktop shell that spawned us)"),
    ] = None,
    token: Annotated[
        str, typer.Option(help="A bearer every /api request must carry (the shell's)")
    ] = "",
    announce: Annotated[
        bool,
        typer.Option(help='Print {"port": N} as one JSON line on stdout for a shell to read'),
    ] = False,
    config: ConfigOpt = None,
) -> None:
    """Open the panel: the live coach in the browser -- a board, coach messages, a voice, an
    orientation flip and an analyse-this-position button, and a debug terminal at the bottom.

    Needs the panel extras: uv sync --group panel (and --group voice to hear the coach). It
    serves a local FastAPI backend (--browser opens it in your browser) -- the same live engine as
    `watch`. Start it, then begin a game on the board, or use --replay <recording> to drive
    the panel from a saved game while developing, no live screen needed. A live session is
    recorded under data/recordings/ with its events and debug log, like `watch`.
    """
    try:
        from sidecoach.ui.server import run_server
    except ImportError as exc:
        console.print(f"[red]panel extras missing:[/red] {exc}; run `uv sync --group panel`")
        raise typer.Exit(1) from exc
    settings = load_settings(config)
    # The app's downloads and the CUDA pack (setup/): the speech-recognition cache under the data
    # directory, and the pack ahead of the base environment before onnxruntime loads.
    from sidecoach.setup import apply_hf_home
    from sidecoach.setup.gpu import activate

    apply_hf_home(settings)
    activate(settings)
    if fps is not None:
        settings.capture.fps = fps
    if not announce:
        where = f"http://127.0.0.1:{port}/" if port else "a free port"
        console.print(f"panel serving at {where}  (Ctrl-C to stop)")
    run_server(
        settings,
        monitor=monitor,
        fps=settings.capture.fps,
        port=port,
        replay=replay,
        record=record,
        persist=persist,
        speed=speed,
        loop=loop,
        open_browser=browser,
        tts_device=tts_device,
        stt_device=stt_device,
        listen=listen,
        parent_pid=parent_pid,
        token=token,
        announce=announce,
    )


if __name__ == "__main__":
    app()
