"""The eval games' engine facts at the app's own engine settings, many games at once.

One single-threaded Stockfish per worker at the app's settings (depth, lines, the best line
carried on, threats, traps, declined recaptures, what a quiet best move threatens) -- the
facts the live coach would have had. Stockfish runs on the CPU only, so the speed comes from
games in parallel: a worker per core does more than one engine with many threads. Uses
tools/eval/facts.py's own `sources` and `compute`.

Output: data/eval/facts/<source>_v2/ for masters and recordings (the older depth-14 facts
stay as they were), data/eval/facts/bots/ for the Nova bot games (tools/bots).

  uv run python tools/eval/facts_parallel.py masters --workers 20
  uv run python tools/eval/facts_parallel.py bots --workers 20 --limit 20
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_ann = None
_out: Path | None = None


def _init(out: str) -> None:
    global _ann, _out
    from sidecoach.analysis.annotator import GameAnnotator
    from sidecoach.analysis.engine import Engine
    from sidecoach.analysis.score import Thresholds
    from sidecoach.cli import _load_book, _load_tagger
    from sidecoach.config import load_settings, require_stockfish

    settings = load_settings()
    engine = Engine(
        path=require_stockfish(settings),
        threads=1,
        hash_mb=128,
        multipv=settings.engine.multipv,
        pv_plies=settings.engine.pv_plies,
        score_model=settings.engine.score_model,
        max_seconds=settings.engine.max_seconds,
    )
    engine.open()
    _ann = GameAnnotator(
        engine=engine,
        depth=settings.engine.depth,
        multipv=settings.engine.multipv,
        thresholds=Thresholds.from_settings(settings.coach),
        tagger=_load_tagger(),
        book=_load_book(settings),
        with_threats=True,
    )
    _out = Path(out)


def _one(item: tuple[str, dict]) -> str:
    import facts as F

    name, game = item
    assert _out is not None and _ann is not None
    path = _out / f"{name.replace('/', '__')}.json"
    if path.exists():
        return f"{name}: cached"
    F.compute(name, game, _ann, path)
    return name


def main() -> None:
    import facts as F

    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["masters", "recordings", "bots", "bench", "students"])
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=0, help="students: the batch (students.py)")
    args = ap.parse_args()
    folder = args.source if args.source in ("bots", "bench", "students") else f"{args.source}_v2"
    if args.source == "students" and args.batch:
        folder = f"students_b{args.batch}"
    out = ROOT / "data" / "eval" / "facts" / folder
    out.mkdir(parents=True, exist_ok=True)
    todo = F.sources(args.source, args.batch)
    if args.limit:
        todo = todo[: args.limit]
    # Longest games first so the pool ends together.
    todo.sort(key=lambda it: -len(it[1]["script"]))
    t0 = time.time()
    context = mp.get_context("spawn")
    with context.Pool(args.workers, initializer=_init, initargs=(str(out),)) as pool:
        for i, name in enumerate(pool.imap_unordered(_one, todo), 1):
            print(f"[{i}/{len(todo)}] {name} ({time.time() - t0:.0f}s)", flush=True)
    print(f"done: {len(todo)} games in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
