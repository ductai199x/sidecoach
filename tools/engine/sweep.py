"""The engine's scheduling, as an optimisation: every policy of a grid played through the same
workloads on a virtual clock, from clean measurements of every search (measure.py), and the
front of latency against processor cost.

The work (measured, one thread, depth 18, tools/engine/atoms*.py): a position's search with two
lines costs 0.34 of one with six; its threats 0.17; more threads never paid. What the product
reads (the audit of 2026-10-04): on every position, the best line and the second (the verdict,
the labels, the profile, the decisive-moment and only-move calls); six lines and the threats
only where the coach speaks -- the position the user faces (hints, and the lesson on the move
they make from it) -- and the threats of the position after a mistake.

So a position the user faces (U) is wanted searched with six lines and its threats; a position
the opponent is to move in (O), with two. A move is judged once the position before has lines
that include it, or once the position after has been searched; verdicts go out in move order.

The knobs:
- workers: searches at once (one thread each);
- reserve: workers kept for the game's own work, never taken by the look-ahead;
- quick: a U position searched with two lines first (its verdict sooner, 16% more for it);
- passed: a U position the game left before its search began gets two lines only ("2") or six
  and its threats all the same ("6");
- look, reach: the look-ahead -- positions at most `look` plies on whose chance of being reached
  is at least `reach`, the chances from each side's lines as this game's moves have shown
  (the analyser's own model: exp(-loss / T), T learned per side per game).

Measured per answer, from the moment the game reaches the position or plays the move: the
verdict; for U positions, the first usable answer (six lines at depth 12) and the whole one (six
lines at 18 and the threats); and the processor time -- all of it, and what was thrown away
(searches stopped, searches of positions never reached).

    uv run --no-sync python tools/engine/sweep.py --missing      # measure what the look-ahead needs
    uv run --no-sync python tools/engine/sweep.py                # the sweep
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis.annotator import cache_key  # noqa: E402
from tools.engine.measure import DB, measure, workloads  # noqa: E402

CACHE = ROOT / "data" / "engine-cache"
OUT = CACHE / "sweep.json"
# Six searches at once, as measure.py makes them, run 1.13 times as long as one alone (its
# --check, 60 positions: p10 1.07, p90 1.21): every time read from the store is divided by it.
SIX_AT_ONCE = 1.13
# One search beside k-1 others, over one alone (atoms.py B, medians): what the pool's own
# searches cost each other.
BESIDE = {1: 1.0, 2: 1.05, 3: 1.02, 4: 1.04, 5: 1.07, 6: 1.10}
FIRST_DEPTH = 12
T_START, T_START_WEIGHT, OUTSIDE_START = 0.05, 2.0, 0.15
T_GRID = [0.005 * 1.35**i for i in range(20)]


# ---- the measurements ------------------------------------------------------------------------


@dataclass
class Atom:
    """One position's measured searches."""

    two: float
    six: float
    six_first: float  # when depth 12 of the six-line search came
    threats: float
    two_lines: list[dict[str, Any]]
    six_lines: list[dict[str, Any]]


def load() -> dict[str, Atom]:
    db = sqlite3.connect(DB)
    out = {}
    for key, found in db.execute("SELECT key, found FROM clean"):
        f = json.loads(found)
        first = next((t for d, t in f["6"]["depths"] if d >= FIRST_DEPTH), f["6"]["wall"])
        out[key] = Atom(
            f["2"]["wall"] / SIX_AT_ONCE,
            f["6"]["wall"] / SIX_AT_ONCE,
            first / SIX_AT_ONCE,
            f["threats"]["wall"] / SIX_AT_ONCE,
            f["2"]["lines"],
            f["6"]["lines"],
        )
    db.close()
    return out


# ---- the workloads ---------------------------------------------------------------------------


def bursts(sessions: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Moves arriving at a fixed pace -- premoves, bullet, a game stepped through: the first 40
    plies of three long games at every 0.2, 0.5, 1 and 2 seconds, the user White."""
    from tools.engine.simulate import Step

    games = [s for name, s in sessions.items() if name.startswith("900+10")][:3]
    out = {}
    for pace in (0.2, 0.5, 1.0, 2.0):
        for g, steps in enumerate(games):
            moves = [s for s in steps if s.kind == "move"][:40]
            start = steps[0]
            seq = [Step(0.0, "start", start.board, game_id=f"burst{g}", user=chess.WHITE)]
            for i, s in enumerate(moves, start=1):
                seq.append(
                    Step(
                        i * pace,
                        "move",
                        s.board,
                        s.move,
                        s.ply,
                        s.board.turn == chess.WHITE,
                        game_id=f"burst{g}",
                        user=chess.WHITE,
                    )
                )
            out[f"burst {pace}s #{g}"] = seq
    return out


# ---- the policy and the clock ----------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    workers: int
    reserve: int
    quick: bool
    passed: str  # "2" or "6"
    look: int
    reach: float
    order: str = "newest"  # "newest": the newest position whole first; "verdicts": verdicts first

    def name(self) -> str:
        return (
            f"w{self.workers} r{self.reserve} {'quick' if self.quick else 'plain'} "
            f"passed{self.passed} look{self.look} reach{self.reach} {self.order}"
        )


@dataclass
class Job:
    key: str
    kind: str  # "2", "6", "threats"
    start: float
    end: float
    worker: int
    ahead: bool  # started for the look-ahead


@dataclass
class Pos:
    """A position's state in a run."""

    key: str
    board: chess.Board
    done: dict[str, float] = field(default_factory=dict)  # kind -> when it was done


class Missing(Exception):
    """A position the store has no measurement of."""

    def __init__(self, boards: dict[str, chess.Board]) -> None:
        super().__init__(f"{len(boards)} positions unmeasured")
        self.boards = boards


class Run:
    """One policy through one session."""

    def __init__(self, policy: Policy, atoms: dict[str, Atom], steps: list[Any]) -> None:
        self.p, self.atoms, self.steps = policy, atoms, steps
        self.pos: dict[str, Pos] = {}
        self.path: dict[int, str] = {}
        self.moves: dict[int, tuple[str, str, chess.Move, bool, float]] = {}  # ply -> ...
        self.current = 0
        self.user: bool | None = None
        self.next_verdict: int | None = None
        self.jobs: dict[int, Job] = {}  # by worker
        self.now = 0.0
        self.ahead: dict[str, float] = {}  # key -> chance
        self.ahead_ever: set[str] = set()
        self.reached_at: dict[str, float] = {}  # U positions: when reached
        self.played: dict[bool, list[tuple[list[float], int | None]]] = {True: [], False: []}
        self.t = {True: T_START, False: T_START}
        self.missing: dict[str, chess.Board] = {}
        # What it measures.
        self.verdicts: list[tuple[bool, float]] = []  # (by the user, latency)
        self.first: list[float] = []
        self.whole: list[float] = []
        self.cpu = 0.0
        self.wasted = 0.0
        self.ahead_cpu: dict[str, float] = defaultdict(float)
        self._open_u: dict[str, float] = {}  # U position reached, its answers not yet told
        self._open_first: dict[str, float] = {}
        self.left_first = 0  # U positions left before their first answer came
        self.left_whole = 0

    # -- positions --

    def node(self, board: chess.Board) -> Pos:
        key = cache_key(board)
        if key not in self.pos:
            p = self.pos[key] = Pos(key, board.copy())
            if board.is_game_over():
                # Nothing to search: the engine answers a finished game with no lines at once.
                p.done = {"2": -1.0, "6": -1.0, "threats": -1.0}
                self.atoms.setdefault(key, Atom(0.0, 0.0, 0.0, 0.0, [], []))
        return self.pos[key]

    def atom(self, key: str) -> Atom:
        atom = self.atoms.get(key)
        if atom is None:  # a position of the game's own: every one was measured beforehand
            raise Missing({key: self.pos[key].board.copy()})
        return atom

    def is_u(self, board: chess.Board) -> bool:
        return self.user is None or board.turn == self.user

    def wanted(self, key: str) -> list[str]:
        """The searches a position on the game's path is wanted with, in order."""
        board = self.pos[key].board
        if not self.is_u(board):
            return ["2"]
        on_path_now = self.path.get(self.current) == key
        if not on_path_now and self.p.passed == "2" and "6" not in self.pos[key].done:
            # Left before its search began (or ran): what its verdicts need.
            if not any(j.key == key and j.kind == "6" for j in self.jobs.values()):
                return ["2"]
        kinds = (["2"] if self.p.quick else []) + ["6", "threats"]
        return kinds

    def lines(self, key: str) -> list[dict[str, Any]] | None:
        done = self.pos[key].done
        if "6" in done:
            return self.atom(key).six_lines
        if "2" in done:
            return self.atom(key).two_lines
        return None

    # -- the game --

    def step(self, s: Any) -> None:
        was = self.path.get(self.current)
        self._step(s)
        if was is not None and was != self.path.get(self.current):
            self.leave(was)
        # A search ahead the game has reached is the game's now: owed to it, and no longer the
        # look-ahead's share of the workers.
        on_path = set(self.path.values())
        for job in self.jobs.values():
            if job.key in on_path:
                job.ahead = False

    def leave(self, key: str) -> None:
        """The game left a position the user faced: an answer not come by then never came
        while it was wanted -- counted, never left out."""
        if self._open_first.pop(key, None) is not None:
            self.left_first += 1
        if self._open_u.pop(key, None) is not None:
            self.left_whole += 1

    def _step(self, s: Any) -> None:
        if s.kind in ("start", "resync"):
            self.path.clear()
            self.moves.clear()
            self.next_verdict = None
            self.user = s.user
            if s.kind == "start":
                self.played = {True: [], False: []}
                self.t = {True: T_START, False: T_START}
            self.current = 0
            p = self.node(s.board)
            self.path[0] = p.key
            self.reach(p.key)
        elif s.kind == "move":
            self.user = s.user
            before = self.node(s.board)
            after_board = s.board.copy()
            after_board.push(s.move)
            after = self.node(after_board)
            if self.path.get(s.ply - 1) != before.key:
                self.path[s.ply - 1] = before.key
            self.path[s.ply] = after.key
            for p in [p for p in self.path if p > s.ply]:
                del self.path[p]
            self.moves[s.ply] = (before.key, after.key, s.move, s.by_user, self.now)
            for p in [p for p in self.moves if p > s.ply]:
                del self.moves[p]
            if self.next_verdict is None or s.ply < self.next_verdict:
                self.next_verdict = s.ply
            self.current = s.ply
            self.reach(after.key)
        elif s.kind == "takeback":
            for p in [p for p in self.path if p > s.ply]:
                del self.path[p]
            for p in [p for p in self.moves if p > s.ply]:
                del self.moves[p]
            if self.next_verdict is not None and self.next_verdict > s.ply + 1:
                self.next_verdict = s.ply + 1
            if not self.moves:
                self.next_verdict = None
            self.current = s.ply
            if s.ply in self.path:
                self.reach(self.path[s.ply])
            elif s.board is not None:
                p = self.node(s.board)
                self.path[s.ply] = p.key
                self.reach(p.key)
        self.judge()
        self.reframe()

    def reach(self, key: str) -> None:
        p = self.pos[key]
        if self.is_u(p.board):
            self._open_first[key] = self.now
            self._open_u[key] = self.now
            self.tell(key)

    def tell(self, key: str) -> None:
        p = self.pos[key]
        if key in self._open_first:
            if "6" in p.done:
                self.first.append(max(0.0, min(p.done["6"], self.now) - self._open_first.pop(key)))
            else:
                job = next((j for j in self.jobs.values() if j.key == key and j.kind == "6"), None)
                if job is not None:
                    at = job.start + self.atom(key).six_first * BESIDE[self.p.workers]
                    if at <= self.now:
                        self.first.append(max(0.0, at - self._open_first.pop(key)))
        if key in self._open_u and "6" in p.done and "threats" in p.done:
            self.whole.append(max(0.0, self.now - self._open_u.pop(key)))

    def judge(self) -> None:
        while self.next_verdict is not None and self.next_verdict in self.moves:
            before, after, move, by_user, at = self.moves[self.next_verdict]
            lines = self.lines(before)
            if lines is None:
                return
            uci = move.uci()
            own = next((i for i, ln in enumerate(lines) if ln["moves"][:1] == [uci]), None)
            if own is None and self.lines(after) is None:
                return
            self.verdicts.append((by_user, self.now - at))
            self.learn(by_user, lines, own)
            self.next_verdict += 1

    # -- the look-ahead (the analyser's model) --

    def learn(self, by_user: bool, lines: list[dict[str, Any]], own: int | None) -> None:
        best = lines[0]["expected_score"]
        losses = [max(0.0, best - ln["expected_score"]) for ln in lines]
        self.played[by_user].append((losses, own))
        chosen = [(ls, i) for ls, i in self.played[by_user] if i is not None]

        def likelihood(t: float) -> float:
            total = 0.0
            for ls, i in chosen:
                logs = [-x / t for x in ls]
                top = max(logs)
                total += logs[i] - top - math.log(sum(math.exp(x - top) for x in logs))
            return total - T_START_WEIGHT * math.log(t / T_START) ** 2

        self.t[by_user] = max(T_GRID, key=likelihood)

    def odds(self, player: bool, lines: list[dict[str, Any]]) -> list[float]:
        best = lines[0]["expected_score"]
        w = [math.exp(-max(0.0, best - ln["expected_score"]) / self.t[player]) for ln in lines]
        total = sum(w)
        played = self.played[player]
        outside = (OUTSIDE_START * T_START_WEIGHT + sum(1 for _, i in played if i is None)) / (
            T_START_WEIGHT + len(played)
        )
        return [(1 - outside) * x / total for x in w]

    def reframe(self) -> None:
        ahead: dict[str, float] = {}
        root = self.path.get(self.current)
        frontier = [(-1.0, 0, root)] if root else []
        on_path = set(self.path.values())
        while frontier:
            neg, dist, key = heapq.heappop(frontier)
            if dist >= self.p.look:
                continue
            lines = self.lines(key)
            if not lines:  # not searched yet, or a finished game
                continue
            board = self.pos[key].board
            mover = board.turn == self.user if self.user is not None else dist % 2 == 0
            for ln, odd in zip(lines, self.odds(mover, lines), strict=True):
                chance = -neg * odd
                if chance < self.p.reach or not ln["moves"]:
                    continue
                move = chess.Move.from_uci(ln["moves"][0])
                if move not in board.legal_moves:
                    continue
                child = board.copy()
                child.push(move)
                if child.is_game_over():
                    continue
                c = self.node(child)
                if c.key in ahead or c.key in on_path:
                    continue
                ahead[c.key] = chance
                heapq.heappush(frontier, (-chance, dist + 1, c.key))
        self.ahead = ahead
        self.ahead_ever.update(ahead)
        # Searches no path wants any more: stopped, their time thrown away.
        for w, job in list(self.jobs.items()):
            if job.key not in on_path and job.key not in ahead:
                spent = self.now - job.start
                self.cpu += spent
                self.wasted += spent
                del self.jobs[w]

    # -- the workers --

    def todo(self) -> list[tuple[str, str, bool]]:
        """What the workers should do next, best first: the game's own work (the newest
        position, then each move's verdict oldest first, then the rest of the path's), then
        the look-ahead, likeliest first."""
        busy = {(j.key, j.kind) for j in self.jobs.values()}
        out: list[tuple[str, str, bool]] = []

        def add(key: str, ahead: bool, lines_only: bool = False) -> None:
            p = self.pos[key]
            if ahead and key not in self.atoms:
                self.missing[key] = p.board.copy()  # measured in the next round
                return
            for kind in self.wanted(key) if not ahead else self.ahead_kinds(key):
                if lines_only and kind == "threats":
                    break
                if kind in p.done or (key, kind) in busy:
                    if lines_only:
                        break  # its lines are found or being found: what a verdict needs
                    continue
                if kind == "threats" and "6" not in p.done:
                    break  # the threats are searched from the position's own lines
                out.append((key, kind, ahead))
                if lines_only:
                    break

        def verdicts() -> None:
            """What the moves not yet judged need, oldest first: the position before's lines,
            and the position after's only when the move is not among them."""
            if self.next_verdict is None:
                return
            for ply in sorted(p for p in self.moves if p >= self.next_verdict):
                before, after, move, *_ = self.moves[ply]
                lines = self.lines(before)
                if lines is None:
                    add(before, False, lines_only=True)
                elif not any(ln["moves"][:1] == [move.uci()] for ln in lines):
                    add(after, False, lines_only=True)

        newest = self.path.get(self.current)
        if self.p.order == "verdicts":
            verdicts()
        if newest:
            add(newest, False)
        if self.p.order == "newest":
            verdicts()
        for ply in sorted(self.path, reverse=True):
            add(self.path[ply], False)
        for key, _ in sorted(self.ahead.items(), key=lambda kv: -kv[1]):
            add(key, True)
        seen = set()
        unique = []
        for item in out:
            if item[:2] not in seen:
                seen.add(item[:2])
                unique.append(item)
        return unique

    def ahead_kinds(self, key: str) -> list[str]:
        board = self.pos[key].board
        if not self.is_u(board):
            return ["2"]
        return (["2"] if self.p.quick else []) + ["6", "threats"]

    def assign(self) -> None:
        free = [w for w in range(self.p.workers) if w not in self.jobs]
        if not free:
            return
        for key, kind, ahead in self.todo():
            if not free:
                break
            ahead_running = sum(1 for j in self.jobs.values() if j.ahead)
            if ahead and ahead_running >= self.p.workers - self.p.reserve:
                continue
            atom = self.atom(key)
            dur = {"2": atom.two, "6": atom.six, "threats": atom.threats}[kind]
            w = free.pop(0)
            self.jobs[w] = Job(key, kind, self.now, self.now + dur * BESIDE[self.p.workers], w, ahead)

    def finish(self, job: Job) -> None:
        del self.jobs[job.worker]
        spent = job.end - job.start
        self.cpu += spent
        if job.ahead:
            self.ahead_cpu[job.key] += spent
        self.pos[job.key].done[job.kind] = self.now
        self.judge()
        for key in list(self._open_first) + list(self._open_u):
            if key in self.pos:
                self.tell(key)
        self.reframe()

    def run(self) -> dict[str, Any]:
        steps = sorted(self.steps, key=lambda s: s.t)
        t0 = steps[0].t
        i = 0
        while True:
            self.assign()
            # First-answer moments of six-line searches under way.
            for key in list(self._open_first):
                self.tell(key)
            next_end = min((j.end for j in self.jobs.values()), default=math.inf)
            next_step = steps[i].t - t0 if i < len(steps) else math.inf
            first_due = min(
                (
                    j.start + self.atom(j.key).six_first * BESIDE[self.p.workers]
                    for j in self.jobs.values()
                    if j.kind == "6" and j.key in self._open_first
                ),
                default=math.inf,
            )
            nxt = min(next_end, next_step, first_due)
            if nxt == math.inf:
                break
            self.now = nxt
            if nxt == first_due and first_due < min(next_end, next_step):
                for key in list(self._open_first):
                    self.tell(key)
                continue
            if next_end <= next_step:
                for job in [j for j in self.jobs.values() if j.end <= self.now]:
                    self.finish(job)
            else:
                self.step(steps[i])
                i += 1
            if i >= len(steps) and not self.owed():
                break
        wall = max(self.now, steps[-1].t - t0, 1e-9)
        reached = set(self.path.values()) | {k for m in self.moves.values() for k in m[:2]}
        never = sum(c for k, c in self.ahead_cpu.items() if k not in reached)
        return {
            "verdict_user": [x for u, x in self.verdicts if u],
            "verdict_opp": [x for u, x in self.verdicts if not u],
            "first": self.first,
            "whole": self.whole,
            "left_first": self.left_first,
            "left_whole": self.left_whole,
            "cpu": self.cpu,
            "wasted": self.wasted + never,
            "wall": wall,
            "moves": sum(1 for s in steps if s.kind == "move"),
        }

    def owed(self) -> bool:
        """Whether the game is still owed work: a search wanted for it not begun, or one under
        way (searches ahead it reached included)."""
        return any(not ahead for _, _, ahead in self.todo()) or any(
            not j.ahead for j in self.jobs.values()
        )


# ---- the sweep -------------------------------------------------------------------------------


def grid() -> list[Policy]:
    return [
        Policy(w, r, q, passed, look, reach, order)
        for w, r, q, passed, look, reach, order in itertools.product(
            [1, 2, 3, 4, 6],
            [0, 1],
            [False, True],
            ["2", "6"],
            [0, 1, 2, 3, 4],
            [0.05, 0.15, 0.3],
            ["newest", "verdicts"],
        )
        if r < w and not (look == 0 and reach != 0.15)
    ]


def _one(args: tuple[Policy, dict[str, list[Any]]]) -> tuple[Policy, dict[str, Any], dict]:
    policy, sessions = args
    atoms = _ATOMS["atoms"]
    out = {}
    missing: dict[str, chess.Board] = {}
    for name, steps in sessions.items():
        run = Run(policy, atoms, steps)
        try:
            out[name] = run.run()
        except Missing as m:
            missing.update(m.boards)
        missing.update(run.missing)
    return policy, out, missing


_ATOMS: dict[str, Any] = {}


def _init() -> None:
    _ATOMS["atoms"] = load()


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("--missing", action="store_true", help="measure what runs lack, in rounds")
    parser.add_argument("--procs", type=int, default=20)
    args = parser.parse_args()
    sessions = workloads()
    sessions.update(bursts(sessions))
    policies = grid()
    print(f"{len(policies)} policies x {len(sessions)} sessions")
    for round_ in range(8 if args.missing else 1):
        results: list[tuple[Policy, dict[str, Any]]] = []
        lacking: dict[str, chess.Board] = {}
        with ProcessPoolExecutor(args.procs, initializer=_init) as pool:
            futures = [pool.submit(_one, (p, sessions)) for p in policies]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="policies", file=sys.stdout, mininterval=2):
                policy, out, missing = fut.result()
                results.append((policy, out))
                lacking.update(missing)
        if not lacking:
            break
        print(f"round {round_ + 1}: {len(lacking)} positions the look-ahead needs, unmeasured")
        if not args.missing:
            break
        measure(lacking, f"the look-ahead's positions, round {round_ + 1}")
    OUT.write_text(
        json.dumps([{"policy": p.__dict__, "name": p.name(), "sessions": out} for p, out in results]),
        encoding="utf-8",
    )
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
