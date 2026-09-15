"""Command-line entry point."""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from sidecoach import __version__
from sidecoach.config import Settings, find_stockfish, load_settings

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
    import cv2

    from sidecoach.replay.recording import FOURCC

    writer = cv2.VideoWriter(
        str(Path(typer.get_app_dir("sidecoach")) / "codec_probe.mkv"),
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
        import chess.engine

        with chess.engine.SimpleEngine.popen_uci(str(path)) as engine:
            name = engine.id.get("name", "unknown")
    except Exception as exc:  # noqa: BLE001
        return "error", f"{path}: {exc}"
    return "ok", f"{name} at {path}"


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
    sizes = ", ".join(f"{i}: {m.w}x{m.h}" for i, m in enumerate(monitors))
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


@app.command()
def record(
    monitor: Annotated[int, typer.Option(help="Monitor index, see doctor")] = 0,
    seconds: Annotated[float | None, typer.Option(help="Stop after this long")] = None,
    fps: Annotated[int, typer.Option(help="Capture rate")] = 10,
    keyframe: Annotated[float, typer.Option(help="Seconds between forced keyframes")] = 1.0,
    out: Annotated[Path | None, typer.Option(help="Recording folder")] = None,
    config: ConfigOpt = None,
) -> None:
    """Record a monitor to a lossless recording until Ctrl+C or --seconds."""
    from sidecoach.capture.loop import capture_loop
    from sidecoach.capture.mss_backend import MssCapture
    from sidecoach.events import Region
    from sidecoach.replay.recording import Recorder, clamp_region
    from sidecoach.vision.board_detect import BoardCandidate
    from sidecoach.vision.tracker import BoardTracker

    settings = load_settings(config)
    folder = out or settings.data_dir / "recordings" / time.strftime("%Y%m%d-%H%M%S")
    cap = MssCapture()
    tracker = BoardTracker()

    def gate_for(board: BoardCandidate | None, width: int, height: int) -> Region | None:
        if board is None:
            return None
        pad = int(round(board.cell_px / 2))
        r = board.region
        return clamp_region(
            Region(r.x - pad, r.y - pad, r.w + 2 * pad, r.h + 2 * pad), width, height
        )

    try:
        region = cap.monitors()[monitor]
        board = tracker.update(cap.grab(region))
        change_region = gate_for(board, region.w, region.h)
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
                gate = gate_for(board, region.w, region.h)
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
    config: ConfigOpt = None,
) -> None:
    """Summarise a recording, track the board through it, or reconstruct the game."""
    from sidecoach.replay.recording import Recording

    rec = Recording(recording)
    console.print(
        f"{recording}: monitor {rec.meta.monitor} {rec.meta.width}x{rec.meta.height} "
        f"at {rec.meta.fps} fps, {len(rec.ticks)} ticks, {rec.stored_count} stored frames, "
        f"started {rec.meta.started_at}"
    )
    if moves or pgn is not None:
        truth = pgn if pgn is not None else recording / "truth.pgn"
        _replay_moves(rec, truth if truth.exists() else None, game)
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


def _replay_moves(rec: object, truth_pgn: Path | None = None, scored_game: int = 1) -> None:
    """Run the game pipeline over a recording and, if a truth PGN exists, score one game."""
    import chess

    from sidecoach.events import GameEnded, GameStarted, MovePlayed, Resync, Takeback
    from sidecoach.replay.recording import Recording
    from sidecoach.state.pipeline import GamePipeline

    assert isinstance(rec, Recording)
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
        events = pipe.process(frame, tick.ts, changed=tick.stored is not None)
        if pipe.bootstraps != bootstraps:
            bootstraps = pipe.bootstraps
            bottom = pipe.bottom.value if pipe.bottom else "?"
            console.print(
                f"tick {tick.frame_id:5d} recogniser built from a start position, "
                f"{bottom} at the bottom"
            )
        scoring = bool(truth) and len(games) == scored_game - 1  # the scored game
        for event in events:
            if isinstance(event, MovePlayed):
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
        pieces = part.split(":")
        path = Path(pieces[0])
        first = int(pieces[1]) if len(pieces) > 1 and pieces[1] else None
        last = int(pieces[2]) if len(pieces) > 2 and pieces[2] else None
        spec.append((path, first, last))
    rec = concat_recordings(spec, out)
    console.print(
        f"{out}: {len(rec.ticks)} ticks, {rec.stored_count} stored frames, "
        f"{rec.meta.width}x{rec.meta.height}"
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

    Flags after the symbol: * highlighted, ~ hover outline, ! not settled (off centre).
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
        parts = []
        for sq in wanted:
            r, c = row_col(sq, pipe.bottom)
            crop = geometry.crop(frame, r, c, size=96)
            reading = pipe.model.classify_patch(crop, r, c)
            flags = ("*" if reading.highlighted else "") + ("~" if reading.hover else "")
            flags += "" if reading.settled else "!"
            parts.append(
                f"{chess.square_name(sq)}={reading.kind}:{reading.symbol or '-'}{flags} "
                f"fg={reading.fg_fraction:.2f} off={reading.offset:.2f} "
                f"conf={reading.confidence:.2f}"
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


@app.command()
def watch(
    monitor: Annotated[int, typer.Option(help="Monitor index, see doctor")] = 0,
    fps: Annotated[int, typer.Option(help="Capture rate")] = 10,
    record: Annotated[bool, typer.Option(help="Also record the session to disk")] = True,
    board: Annotated[bool, typer.Option(help="Print the board after every move")] = True,
    seconds: Annotated[float | None, typer.Option(help="Stop after this long")] = None,
    out: Annotated[Path | None, typer.Option(help="Recording folder")] = None,
    config: ConfigOpt = None,
) -> None:
    """Watch the screen live: lock onto the board, follow the game, print every move.

    Start it, then begin a new game so the recogniser can bootstrap from the start position.
    Every session is recorded so anything odd can be replayed with `sidecoach replay`.
    """
    import chess

    from sidecoach.capture.loop import capture_loop
    from sidecoach.capture.mss_backend import MssCapture
    from sidecoach.events import (
        GameEnded,
        GameStarted,
        MovePlayed,
        Region,
        Resync,
        Takeback,
    )
    from sidecoach.replay.recording import Recorder, clamp_region, frames_differ
    from sidecoach.state.pipeline import GamePipeline

    settings = load_settings(config)
    cap = MssCapture()
    region = cap.monitors()[monitor]
    pipe = GamePipeline()
    recorder: Recorder | None = None
    folder: Path | None = None
    if record:
        folder = out or settings.data_dir / "recordings" / time.strftime("%Y%m%d-%H%M%S")
        recorder = Recorder(
            folder, monitor=monitor, region=region, fps=fps, keyframe_interval_s=1.0
        )
    console.print(
        f"watching monitor {monitor} {region.w}x{region.h} at {fps} fps"
        + (f", recording to {folder}" if folder else "")
    )
    console.print("start a new game on the board; press Ctrl+C to stop\n")

    def gate_for(width: int, height: int) -> Region | None:
        if pipe.board is None:
            return None
        pad = int(round(pipe.board.cell_px / 2))
        r = pipe.board.region
        return clamp_region(
            Region(r.x - pad, r.y - pad, r.w + 2 * pad, r.h + 2 * pad), width, height
        )

    last_frame = None
    locked = False
    bootstraps = 0
    last_move_ts: float | None = None
    last_status = time.monotonic()
    ticks_since_status = 0
    games: list[str] = []
    try:
        for _frame_id, ts, frame in capture_loop(cap, region, fps, max_seconds=seconds):
            gate = gate_for(region.w, region.h)
            if recorder is not None:
                recorder.change_region = gate
                changed = recorder.write(frame, ts).stored is not None
            else:
                changed = last_frame is None or frames_differ(
                    last_frame, frame, region=gate, min_fraction=0.0005
                )
                last_frame = frame
            try:
                events = pipe.process(frame, ts, changed=changed)
            except Exception as exc:  # noqa: BLE001
                # Never let a pipeline bug end a live session; the recording keeps the
                # evidence and the next frames get a fresh chance.
                console.print(f"[red]pipeline error:[/red] {exc!r}")
                events = []
            ticks_since_status += 1

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
                        f"[bold]new game[/bold] {event.game_id}, you play {event.user_side.value}"
                    )
                    last_move_ts = ts
                elif isinstance(event, MovePlayed):
                    after = chess.Board(event.fen_after)
                    number = (event.ply + 1) // 2
                    dots = "." if event.ply % 2 == 1 else "..."
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
                elif isinstance(event, GameEnded):
                    console.print(f"[bold]game over[/bold] {event.result} ({event.reason})")
                    if event.pgn:
                        games.append(event.pgn)
                        console.print(f"PGN: {event.pgn}")

            now = time.monotonic()
            if now - last_status >= 5.0:
                rate = ticks_since_status / (now - last_status)
                state = "locked" if pipe.board is not None else "no board"
                stage = "following" if pipe.model is not None else "waiting for a start position"
                plies = len(pipe.reconciler.board.move_stack) if pipe.reconciler else 0
                stored = f", {recorder.stored} frames stored" if recorder else ""
                console.print(f"[dim]{rate:.1f} fps, {state}, {stage}, ply {plies}{stored}[/dim]")
                last_status = now
                ticks_since_status = 0
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            recorder.close()
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
