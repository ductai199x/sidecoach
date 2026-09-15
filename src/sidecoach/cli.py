"""Command-line entry point."""

from __future__ import annotations

import importlib
import platform
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, TextIO, cast

import typer
from rich.console import Console
from rich.table import Table

from sidecoach import __version__
from sidecoach.capture.base import primary_monitor
from sidecoach.config import Settings, find_stockfish, load_settings
from sidecoach.events import Region, Side

if TYPE_CHECKING:
    import chess

    from sidecoach.analysis.annotator import Analyser, Book, Tagger
    from sidecoach.capture.base import CaptureBackend
    from sidecoach.events import MoveAssessment
    from sidecoach.replay.recording import Recording

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


def _check_llm(settings: Settings) -> tuple[str, str]:
    from sidecoach.llm.factory import make_provider

    try:
        provider = make_provider(settings.llm, timeout_s=15)
    except Exception as exc:  # noqa: BLE001
        return "error", str(exc)
    ok, detail = provider.healthcheck()
    return ("ok" if ok else "error"), f"{provider.name}: {detail}"


def _check_capture() -> tuple[str, str]:
    try:
        from sidecoach.capture.mss_backend import MssCapture

        cap = MssCapture()
        try:
            monitors = cap.monitors()
            frame = cap.grab(monitors[0])
        finally:
            cap.close()
    except Exception as exc:  # noqa: BLE001
        return "error", str(exc)
    primary = primary_monitor(monitors)
    sizes = ", ".join(
        f"{i}: {m.w}x{m.h}" + (" (primary)" if i == primary else "") for i, m in enumerate(monitors)
    )
    return "ok", f"{len(monitors)} monitor(s) [{sizes}]; grab ok {frame.shape[1]}x{frame.shape[0]}"


@app.command()
def doctor(config: ConfigOpt = None) -> None:
    """Check Python, vision libraries, video codec, Stockfish, the LLM, and screen capture."""
    settings = load_settings(config)
    checks: list[tuple[str, str, str]] = [
        ("Python", "ok", f"{platform.python_version()} ({sys.executable})"),
        ("Vision libs", *_check_vision_libs()),
        ("Video codec", *_check_video_codec()),
        ("Stockfish", *_check_engine(settings)),
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

    from sidecoach.capture.mss_backend import MssCapture

    settings = load_settings(config)
    folder = out or settings.data_dir / "screenshots"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    cap = MssCapture()
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
    from sidecoach.capture.mss_backend import MssCapture
    from sidecoach.replay.recording import Recorder, board_gate
    from sidecoach.vision.tracker import BoardTracker

    settings = load_settings(config)
    folder = out or settings.data_dir / "recordings" / time.strftime("%Y%m%d-%H%M%S")
    cap = MssCapture()
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
    config: ConfigOpt = None,
) -> None:
    """Summarise a recording, track the board through it, or reconstruct the game.

    `--scale` replays the same frames smaller or larger, which is how size invariance is
    tested without storing another copy of the recording. `--assess` runs the engine
    alongside, which is how the live watcher's wiring is exercised without a game on screen.
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
        assessing = _Assessing(settings, assess)
        try:
            _replay_moves(rec, truth if truth.exists() else None, game, scale, assessing)
        finally:
            assessing.close()
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
) -> None:
    """Run the game pipeline over a recording and, if a truth PGN exists, score one game.

    With an `assessing` the engine follows along as well, which is the same wiring the live
    watcher uses and the only way to exercise it without a game on the screen.
    """
    import chess
    import cv2

    from sidecoach.events import GameEnded, GameStarted, MovePlayed, Resync, Takeback
    from sidecoach.state.pipeline import GamePipeline

    truth = _read_truth(truth_pgn) if truth_pgn is not None else []
    truth_placements: list[str] = []
    if truth:
        replay_board = chess.Board()
        truth_placements.append(replay_board.board_fen())
        for san in truth:
            replay_board.push_san(san)
            truth_placements.append(replay_board.board_fen())
    pipe = GamePipeline()
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
            assessing.drain(indent="            ")
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
        assessing.wait(indent="            ")
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


def _load_tagger() -> Tagger | None:
    """The motif tagger, when it is built; the analysis runs without it until then."""
    try:
        module = importlib.import_module("sidecoach.analysis.motifs")
    except ImportError:
        return None
    return cast("Tagger", module)


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

    def __init__(self, settings: Settings, enabled: bool, *, depth: int | None = None) -> None:
        self.analyser = _make_analyser(settings, depth=depth) if enabled else None
        if enabled and self.analyser is None:
            console.print("[yellow]no Stockfish: moves will not be assessed[/yellow]")
        # Where each move was played from, so a better move can be named in the notation a
        # person reads; dropped as soon as that assessment is printed.
        self._from: dict[int, str] = {}

    @property
    def on(self) -> bool:
        return self.analyser is not None

    def submit(self, event: object) -> None:
        import chess

        from sidecoach.events import MovePlayed

        if self.analyser is None or not isinstance(event, MovePlayed):
            return
        before = chess.Board(event.fen_before)
        self.analyser.submit_move(event.game_id, before, chess.Move.from_uci(event.uci), event.ply)
        self._from[event.ply] = event.fen_before

    def reset(self) -> None:
        if self.analyser is not None:
            self.analyser.annotator.reset()
            self._from.clear()

    def drain(self, log: TextIO | None = None, indent: str = "  ") -> None:
        import chess

        from sidecoach.events import MoveAssessment

        if self.analyser is None:
            return
        for assessed in self.analyser.poll():
            if isinstance(assessed, MoveAssessment):
                fen = self._from.pop(assessed.ply, None)
                if fen is not None:
                    text = _assessment_text(assessed, chess.Board(fen))
                    console.print(f"{indent}[dim]ply {assessed.ply}[/dim] {text}")
            if log is not None:
                log.write(assessed.model_dump_json() + "\n")

    def wait(self, timeout: float = 30.0, log: TextIO | None = None, indent: str = "  ") -> None:
        """Let the engine catch up at the end of a game, so nothing is lost on the way out."""
        if self.analyser is None:
            return
        deadline = time.monotonic() + timeout
        while self.analyser.pending and time.monotonic() < deadline:
            self.drain(log, indent)
            time.sleep(0.05)
        self.drain(log, indent)

    def close(self) -> None:
        if self.analyser is not None:
            self.analyser.close()
            self.analyser = None


_SEVERITY_COLOUR = {
    "blunder": "red",
    "mistake": "yellow",
    "inaccuracy": "cyan",
    "none": "green",
}


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
    table = Table(box=None, pad_edge=False)
    for column in ("move", "", "loss", "severity", "better", "motifs"):
        table.add_column(column, overflow="fold")

    board = parsed.board()
    counts = {"inaccuracy": 0, "mistake": 0, "blunder": 0}
    total_loss = {chess.WHITE: 0.0, chess.BLACK: 0.0}
    plies = {chess.WHITE: 0, chess.BLACK: 0}
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
                if assessment.severity.value in counts:
                    counts[assessment.severity.value] += 1
                show = wanted == "both" or (wanted == "white") == (mover == chess.WHITE)
                number = board.fullmove_number
                label = f"{number}." if mover == chess.WHITE else f"{number}..."
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
                    }
                )
                if show and not quiet:
                    colour = _SEVERITY_COLOUR.get(assessment.severity.value, "white")
                    marker = "" if assessment.severity.value == "none" else "*"
                    table.add_row(
                        label,
                        f"{san}{marker}",
                        f"{assessment.expected_loss:.2f}",
                        f"[{colour}]{assessment.severity.value}[/{colour}]"
                        if assessment.severity.value != "none"
                        else "",
                        better,
                        tags,
                    )
                board.push(move)
    except KeyboardInterrupt:
        console.print("[yellow]stopped[/yellow]")

    if not quiet:
        console.print(table)
    if opening:
        console.print(f"opening: {opening}")
    for colour, name in ((chess.WHITE, "white"), (chess.BLACK, "black")):
        if plies[colour]:
            console.print(
                f"{name}: {plies[colour]} moves, "
                f"average loss {total_loss[colour] / plies[colour]:.3f}"
            )
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
def watch(
    monitor: Annotated[
        int | None, typer.Option(help="Monitor index, see doctor; default the primary")
    ] = None,
    fps: Annotated[int, typer.Option(help="Capture rate")] = 10,
    record: Annotated[bool, typer.Option(help="Also record the session to disk")] = True,
    board: Annotated[bool, typer.Option(help="Print the board after every move")] = True,
    analysis: Annotated[bool, typer.Option(help="Assess every move with the engine")] = True,
    seconds: Annotated[float | None, typer.Option(help="Stop after this long")] = None,
    out: Annotated[Path | None, typer.Option(help="Recording folder")] = None,
    config: ConfigOpt = None,
) -> None:
    """Watch the screen live: lock onto the board, follow the game, print every move.

    Start it, then begin a new game so the recogniser can bootstrap from the start position.
    With --record (the default) the session is recorded so anything odd can be replayed
    with `sidecoach replay`, and every event is logged beside it. Keys while watching:
    r = you resign, d = draw agreed, n = new game (forget the current one), q = quit.

    With --analysis the engine assesses each move once it is played, which is what the
    disclosure rules allow; what to play in the position on the board is never printed.
    """
    import chess

    from sidecoach.capture.loop import CaptureFailed, capture_loop
    from sidecoach.capture.mss_backend import MssCapture
    from sidecoach.events import (
        GameEnded,
        GameStarted,
        MovePlayed,
        Resync,
        Takeback,
    )
    from sidecoach.replay.recording import ChangeGate, Recorder, board_gate
    from sidecoach.state.pipeline import GamePipeline
    from sidecoach.ui.keys import KeyReader

    settings = load_settings(config)
    cap = MssCapture()
    monitor, region = _monitor_region(cap, monitor)
    pipe = GamePipeline()
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
                if key == "q":
                    console.print("quit")
                    break
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
                    elif isinstance(event, Resync):
                        console.print(f"[yellow]resync[/yellow]: {event.reason}")
                        assessing.reset()
                    elif isinstance(event, GameEnded):
                        console.print(f"[bold]game over[/bold] {event.result} ({event.reason})")
                        if event.pgn:
                            games.append(event.pgn)
                            console.print(f"PGN: {event.pgn}")

                assessing.drain(log)

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
        assessing.wait(log=log)
        assessing.close()
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


if __name__ == "__main__":
    app()
