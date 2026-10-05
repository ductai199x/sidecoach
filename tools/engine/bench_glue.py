"""Before and after, end to end: the app's engine glue (`cli._Assessing`, its analyser built from
the default settings, as the app builds it) fed three of the player's games move by move, each
move from idle -- every answer of the move before in, nothing owed. Per move: how long until
its verdict, until the first word on the position it made, until that position is told whole;
and the Stockfish processes' CPU. The same script runs against any checkout's code (`--src`:
`main` as installed, or this branch), so the two are measured alike.

    uv run --no-sync python tools/engine/bench_glue.py --label branch
    uv run --no-sync python tools/engine/bench_glue.py --label main --src <a checkout of main>/src
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "engine-cache"
GAMES = 3
PLIES = 40
PATIENCE_S = 300.0  # a bound on a hang, never the expected path


def games() -> list[list[str]]:
    """The first `GAMES` 10-minute games of the player's archive, `PLIES` plies each (UCI)."""
    import chess.pgn

    archive = json.loads((CACHE / "chesscom-nosegazm.json").read_text(encoding="utf-8"))
    out = []
    for game in archive:
        if game.get("time_control") != "600":
            continue
        parsed = chess.pgn.read_game(io.StringIO(game["pgn"]))
        if parsed is None:
            continue
        moves = [m.uci() for m in parsed.mainline_moves()][:PLIES]
        if len(moves) == PLIES:
            out.append(moves)
        if len(out) == GAMES:
            return out
    raise SystemExit("not enough games in the archive")


def engine_cpu() -> float:
    """The CPU seconds of every Stockfish this process has running."""
    import psutil

    total = 0.0
    for child in psutil.Process().children(recursive=True):
        try:
            if "stockfish" in child.name().lower():
                t = child.cpu_times()
                total += t.user + t.system
        except psutil.Error:
            continue
    return total


def settle(assessing: Any, ply: int, since: float, row: dict | None) -> None:
    """Until nothing is owed: the moments the move at `ply`'s answers came, into `row`."""
    from sidecoach.events import Analysis, MoveAssessment

    while time.perf_counter() - since < PATIENCE_S:
        now = time.perf_counter() - since
        # Read before the drain: a word is put before nothing is owed, so once this is 0 the
        # drain below holds the last of them.
        owed = assessing.analyser.pending
        for event in assessing.drain():
            if row is None or getattr(event, "ply", None) != ply:
                continue
            if isinstance(event, MoveAssessment):
                row.setdefault("verdict", now)
                row.setdefault("verdict_depth", event.depth)
            elif isinstance(event, Analysis):
                row.setdefault("first", now)
                if not event.provisional:
                    row["whole"] = now
                    # The depth the position's lines reached: the shallowest of them.
                    row["whole_depth"] = min((line.depth for line in event.lines), default=0)
        if owed == 0:
            return
        time.sleep(0.002)
    raise SystemExit(f"ply {ply}: not settled in {PATIENCE_S} s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--src", type=Path, default=ROOT / "src")
    args = parser.parse_args()
    sys.path.insert(0, str(args.src.resolve()))

    import chess
    from tqdm import tqdm

    import sidecoach
    from sidecoach.cli import _Assessing
    from sidecoach.config import Settings, find_stockfish
    from sidecoach.events import GameStarted, MovePlayed, Side

    print(f"code from {Path(sidecoach.__file__).parent}")
    stockfish = find_stockfish(Settings())
    if stockfish is None:
        raise SystemExit("no Stockfish under tools/stockfish")
    rows = []
    played = games()
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings.model_validate(
            {"data_dir": tmp, "engine": {"stockfish_path": str(stockfish)}}
        )
        print(f"engine settings: {settings.engine.model_dump()}")
        bar = tqdm(total=GAMES * PLIES, desc=args.label, file=sys.stdout, mininterval=2)
        for g, moves in enumerate(played):
            assessing = _Assessing(settings, True)
            analyser = assessing.analyser
            assert analyser is not None
            game_id = f"bench{g}"
            board = chess.Board()

            assessing.submit(
                GameStarted(game_id=game_id, start_fen=chess.STARTING_FEN, user_side=Side.WHITE)
            )
            settle(assessing, 0, time.perf_counter(), None)
            for ply, uci in enumerate(moves, start=1):
                move = chess.Move.from_uci(uci)
                before = board.fen()
                san = board.san(move)
                board.push(move)
                cpu0 = engine_cpu()
                row: dict = {"game": g, "ply": ply}
                t0 = time.perf_counter()
                assessing.submit(
                    MovePlayed(
                        game_id=game_id,
                        ply=ply,
                        san=san,
                        uci=uci,
                        fen_before=before,
                        fen_after=board.fen(),
                        by_user=ply % 2 == 1,
                    )
                )
                settle(assessing, ply, t0, row)
                row["settled"] = time.perf_counter() - t0
                row["cpu"] = engine_cpu() - cpu0
                rows.append(row)
                bar.update(1)
            assessing.close()
        bar.close()

    out = CACHE / f"bench_glue-{args.label}.json"
    out.write_text(json.dumps(rows), encoding="utf-8")

    def q(v: list[float], p: float) -> float:
        v = sorted(v)
        return v[min(len(v) - 1, int(p * len(v)))]

    print(f"\n{args.label}: {len(rows)} moves")
    for key in ("verdict", "first", "whole", "settled", "cpu"):
        v = [r[key] for r in rows if key in r]
        missing = len(rows) - len(v)
        print(
            f"  {key:8} p50 {q(v, 0.5):6.2f}  p90 {q(v, 0.9):6.2f}  max {max(v):6.2f}  "
            f"sum {sum(v):7.1f}" + (f"  ({missing} never came)" if missing else "")
        )
    for key in ("verdict_depth", "whole_depth"):
        v = [r[key] for r in rows if key in r]
        below = sum(1 for d in v if d < settings.engine.depth)
        print(f"  {key:13} min {min(v)}  p10 {q(v, 0.1)}  below the depth asked: {below}/{len(v)}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
