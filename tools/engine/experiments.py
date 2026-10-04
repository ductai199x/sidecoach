"""Scheduling variants of the live analyser, simulated on every cached session at once
(tools/engine/simulate.py): each variant's processor time, search time stopped, and the waits
after a move pooled over the sessions -- the numbers a change to `Analyser._pick` is held to.

    uv run python tools/engine/experiments.py <dump> [--chesscom]"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import simulate as sim  # noqa: E402

from sidecoach.analysis import analyser as an  # noqa: E402

ORIGINAL_PICK = an.Analyser._pick


def likeliest_first(self, index):  # noqa: ANN001
    """The look-ahead ordered by how likely the game is to go there -- the sum of its ranks
    first, then its distance -- instead of ply by ply."""
    saved = self._ahead
    self._ahead = dict(sorted(saved.items(), key=lambda kv: (sum(kv[1][1]), kv[1][0])))
    try:
        busy = {(j.kind, j.key) for j in self._claimed.values()}
        for ply in sorted(self._path, reverse=True):
            node = self._nodes[self._path[ply]]
            if node.measured is None and not node.failed and (an.MAIN, node.key) not in busy:
                return an._Job(an.MAIN, node.key, self._era)
        if self.annotator.with_threats:
            for ply in sorted(self._path, reverse=True):
                node = self._nodes[self._path[ply]]
                if (
                    node.measured is not None
                    and node.whole is None
                    and not node.failed
                    and (an.SIDE, node.key) not in busy
                ):
                    return an._Job(an.SIDE, node.key, self._era)
        if index == 0 and len(self.engines) > 1:
            return None
        for key in self._ahead:
            node = self._nodes.get(key)
            if (
                node is not None
                and node.measured is None
                and not node.failed
                and (an.MAIN, key) not in busy
            ):
                return an._Job(an.MAIN, key, self._era)
        return None
    finally:
        self._ahead = saved


def side_after_first_ply(self, index):  # noqa: ANN001
    """The game's side searches after the look-ahead's first ply (the player's candidates),
    not before it."""
    busy = {(j.kind, j.key) for j in self._claimed.values()}
    for ply in sorted(self._path, reverse=True):
        node = self._nodes[self._path[ply]]
        if node.measured is None and not node.failed and (an.MAIN, node.key) not in busy:
            return an._Job(an.MAIN, node.key, self._era)
    reserved = index == 0 and len(self.engines) > 1
    ahead = sorted(self._ahead.items(), key=lambda kv: kv[1])
    if not reserved:
        for key, (distance, _) in ahead:
            node = self._nodes.get(key)
            if (
                distance == 1
                and node is not None
                and node.measured is None
                and not node.failed
                and (an.MAIN, key) not in busy
            ):
                return an._Job(an.MAIN, key, self._era)
    if self.annotator.with_threats:
        for ply in sorted(self._path, reverse=True):
            node = self._nodes[self._path[ply]]
            if (
                node.measured is not None
                and node.whole is None
                and not node.failed
                and (an.SIDE, node.key) not in busy
            ):
                return an._Job(an.SIDE, node.key, self._era)
    if reserved:
        return None
    for key, _ in ahead:
        node = self._nodes.get(key)
        if (
            node is not None
            and node.measured is None
            and not node.failed
            and (an.MAIN, key) not in busy
        ):
            return an._Job(an.MAIN, key, self._era)
    return None


def no_reserved_worker(self, index):  # noqa: ANN001
    """Every worker takes the look-ahead: none kept for the game's own positions."""
    engines = self.engines
    self.engines = engines[:1]  # `_pick` reserves the first only when there are several
    try:
        return ORIGINAL_PICK(self, index)
    finally:
        self.engines = engines


VARIANTS = {
    "today": ORIGINAL_PICK,
    "likeliest first": likeliest_first,
    "side after first ply": side_after_first_ply,
    "no reserved worker": no_reserved_worker,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dump", type=Path, nargs="?")
    parser.add_argument("--chesscom", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--look", type=int, default=2)
    args = parser.parse_args()
    sessions = {}
    if args.dump:
        sessions["puzzles"] = sim.session(args.dump)
    sessions.update(
        sim.chesscom_sessions("nosegazm") if args.chesscom else sim.library_sessions("nosegazm", 4)
    )
    cache = sim.load_cache(Path("all"))
    scale = sim.json.loads((sim.CACHE / "scale.json").read_text(encoding="utf-8"))["scale"]
    print(f"{len(sessions)} sessions, {args.workers} workers, look-ahead {args.look}")
    print(
        f"{'variant':22} {'core-s':>7} {'stopped s':>9} {'twice':>5} "
        f"{'you verdict p90':>15} {'opp verdict p90':>15} "
        f"{'you full med/p90':>16} {'opp full p90':>12}"
    )
    for name, pick in VARIANTS.items():
        an.Analyser._pick = pick
        pooled = defaultdict(list)
        core = stopped = twice = 0.0
        for steps in sessions.values():
            r = sim.simulate(
                steps, cache, workers=args.workers, look=args.look, scale=scale, raw=True
            )
            core += r["stockfish_core_s"]
            stopped += r["stopped_s"]
            twice += r["searched_twice"]
            for k, v in r["raw_waits"].items():
                pooled[k] += v

        def p(key: str, q: float, pooled=pooled) -> float:  # noqa: B008
            v = sorted(pooled[key])
            return (
                v[int(len(v) * q)]
                if q < 0.5
                else (statistics.median(v) if q == 0.5 else v[int(len(v) * q)])
            )

        print(
            f"{name:22} {core:7.0f} {stopped:9.0f} {twice:5.0f} "
            f"{p('classification (you)', 0.9):15.2f} {p('classification (opponent)', 0.9):15.2f} "
            f"{p('full analysis (you)', 0.5):7.2f}/{p('full analysis (you)', 0.9):<8.2f} "
            f"{p('full analysis (opponent)', 0.9):12.2f}"
        )
    an.Analyser._pick = ORIGINAL_PICK


if __name__ == "__main__":
    main()
