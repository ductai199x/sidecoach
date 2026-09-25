"""Stockfish's analysis of many positions, at the app's settings, computed once in parallel
and kept on disk: the slow half of every coach benchmark.

A benchmark reads a position the way the live coach does -- `GameAnnotator.analysis_event`
at the settings' depth and lines, the best line carried on to its point, the threats, the
motif tags -- and that is a second or more of Stockfish per position. So it is done once,
by one single-threaded Stockfish per worker process on every core but two, and written to
`data/puzzles/analysis/<key>.jsonl` as it comes in; the key names the settings, so a change
to them starts a new file and a benchmark rerun after a change to the coach reads its
positions back in milliseconds.

  from analysis_cache import analyse_all
  analyses = analyse_all(fens, workers=22)   # {fen: Analysis}
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.events import Analysis  # noqa: E402

CACHE_DIR = ROOT / "data/puzzles/analysis"
# One Stockfish thread per worker, a small hash each: the positions are independent.
WORKER_HASH_MB = 64
# What an analysis holds beyond the engine's lines, by version: 2 added the recaptures the
# best line declines (`analysis/defences.py`) and the solution played out of order among the
# traps (`traps.tempting`). A cache of the version before is upgraded in place of a new
# search: the lines are kept, only what the new version adds is searched.
ANALYSIS_VERSION = 3  # 3: the solution out of order only on a square of its own


def settings_key(version: int = ANALYSIS_VERSION) -> str:
    """The name of the cache file for the app's current engine settings, and for what an
    analysis holds (`ANALYSIS_VERSION`; the first had no number in the name)."""
    from sidecoach.analysis import engine as engine_module
    from sidecoach.config import find_stockfish, load_settings

    s = load_settings().engine
    path = find_stockfish(load_settings())
    parts = {
        "depth": s.depth,
        "multipv": s.multipv,
        "pv_plies": s.pv_plies,
        "score_model": s.score_model,
        "max_seconds": s.max_seconds,
        "min_best": engine_module.MIN_BEST_PLIES,
        "extend": [engine_module.EXTEND_DEPTH, engine_module.EXTEND_SECONDS],
        "stockfish": path.name if path else None,
        "threads": 1,
    }
    if version > 1:
        parts["analysis"] = version
    digest = hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:10]
    return f"d{s.depth}-pv{s.multipv}-{digest}"


def load(key: str | None = None) -> dict[str, Analysis]:
    """Every analysis cached for `key` (the current settings by default), by FEN."""
    path = CACHE_DIR / f"{key or settings_key()}.jsonl"
    out: dict[str, Analysis] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out[row["fen"]] = Analysis.model_validate(row["analysis"])
    return out


_annotator = None


def _init() -> None:
    global _annotator
    from sidecoach import cli
    from sidecoach.analysis.annotator import GameAnnotator
    from sidecoach.analysis.engine import Engine
    from sidecoach.analysis.score import Thresholds
    from sidecoach.config import find_stockfish, load_settings

    settings = load_settings()
    path = find_stockfish(settings)
    assert path is not None, "no Stockfish"
    engine = Engine(
        path=path,
        threads=1,
        hash_mb=WORKER_HASH_MB,
        multipv=settings.engine.multipv,
        pv_plies=settings.engine.pv_plies,
        score_model=settings.engine.score_model,
        max_seconds=settings.engine.max_seconds,
    )
    engine.open()
    _annotator = GameAnnotator(
        engine=engine,
        depth=settings.engine.depth,
        multipv=settings.engine.multipv,
        thresholds=Thresholds.from_settings(settings.coach),
        tagger=cli._load_tagger(),
        book=None,  # a puzzle has no game behind it
    )


def _analyse(job: tuple[str, str | None]) -> tuple[str, str]:
    """A position analysed: from scratch, or, given the previous version's analysis, only
    what this version adds -- its lines kept, the traps and the declined recaptures searched."""
    import chess

    from sidecoach.analysis.defences import defence_lines
    from sidecoach.analysis.traps import trap_lines

    fen, older = job
    assert _annotator is not None
    _annotator.reset()  # no position is another's history
    board = chess.Board(fen)
    if older is None:
        event = _annotator.analysis_event("bench", board, 0)
    else:
        event = Analysis.model_validate_json(older)
        engine = _annotator.engine
        event = event.model_copy(
            update={
                "traps": trap_lines(engine, board, event.lines),
                "defences": defence_lines(engine, board, event.lines),
            }
        )
    return fen, event.model_dump_json()


def analyse_all(
    fens: Iterable[str], *, workers: int | None = None, progress: bool = True
) -> dict[str, Analysis]:
    """The analysis of every FEN, from the cache where it is there, the rest computed in
    parallel and appended to the cache as each one lands (a stopped run loses nothing)."""
    key = settings_key()
    cached = load(key)
    wanted = list(dict.fromkeys(fens))
    missing = [fen for fen in wanted if fen not in cached]
    older = load(settings_key(ANALYSIS_VERSION - 1)) if missing and ANALYSIS_VERSION > 1 else {}
    jobs = [(fen, older[fen].model_dump_json() if fen in older else None) for fen in missing]
    if missing:
        upgraded = sum(1 for _, text in jobs if text is not None)
        if upgraded:
            print(
                f"cache: {upgraded} of {len(missing)} upgraded from version {ANALYSIS_VERSION - 1}"
            )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        n = workers or max(1, (os.cpu_count() or 4) - 2)
        started = time.monotonic()
        bar = None
        if progress:
            from tqdm import tqdm

            bar = tqdm(total=len(missing), desc=f"engine ({n} workers)", unit="pos", mininterval=2)
        with (
            (CACHE_DIR / f"{key}.jsonl").open("a", encoding="utf-8") as out,
            mp.get_context("spawn").Pool(n, initializer=_init) as pool,
        ):
            for fen, text in pool.imap_unordered(_analyse, jobs, chunksize=2):
                out.write(json.dumps({"fen": fen, "analysis": json.loads(text)}) + "\n")
                out.flush()
                cached[fen] = Analysis.model_validate_json(text)
                if bar is not None:
                    bar.update(1)
        if bar is not None:
            bar.close()
        rate = len(missing) / max(time.monotonic() - started, 1e-9)
        print(f"engine: {len(missing)} positions at {rate:.1f}/s on {n} workers", flush=True)
    return {fen: cached[fen] for fen in wanted}
