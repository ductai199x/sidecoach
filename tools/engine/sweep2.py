"""The engine's scheduling as an optimisation, on the one-line design: every policy of a grid
played through the same workloads on a virtual clock, from clean measurements of every search,
and the front of latency against processor cost.

What the measurements decided (tools/engine/atoms*.py, explode.py, single*.py, decided.py):
a search of several lines to depth 18 explodes on decided and forcing positions (up to 47 s;
84 of 8,886 positions over 5 s); one line to depth 18 does not (p50 0.19 s, p99 1.1 s, two over
5 s, all decided) -- and for those, a decided score at the deepest depth done by 1 s stands in
for depth 18's (Tai's call, 2026-10-05: its severity flips are as small as two depth-18 searches'
own disagreement). Six lines to depth 12 never take more than 1.9 s.

So a position's searches are:
- "1": one line to depth 18 -- every position: its best move and its score, all a verdict reads;
- "m12": six lines to depth 12 -- a position the user faces (U): the hint at once;
- "threats" -- a U position: the null-move search the lessons read;
- each of a U position's `cands` best moves' own position, searched "1": the candidates the coach
  reads at depth 18 (19 from the position), and exactly the positions the game may go to next.

A move is judged from the position before's "1" when it is that line's move (loss 0), else
against the position after's "1" (or its decided stand-in); verdicts go out in move order.

The knobs: workers, reserve (kept from the look-ahead), cands (0, 3, 6), look and reach (the
look-ahead: positions at most `look` plies on whose chance of being reached is at least
`reach`, the chances from each side's lines as this game has shown), order (the newest position
first, or the verdicts first).

    uv run --no-sync python tools/engine/sweep2.py --missing     # measure what runs lack
    uv run --no-sync python tools/engine/sweep2.py               # the sweep
    uv run --no-sync python tools/engine/sweep2.py --check       # the clock against the hand
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis.annotator import cache_key  # noqa: E402

CACHE = ROOT / "data" / "engine-cache"
OUT = CACHE / "sweep2.json"
SIX_AT_ONCE = 1.13  # measure.py --check: six at once over one alone
BESIDE = {1: 1.0, 2: 1.05, 3: 1.02, 4: 1.04, 5: 1.07, 6: 1.10}  # atoms.py B, medians
RULE_S, DECIDED = 1.0, (0.15, 0.85)
T_START, T_START_WEIGHT, OUTSIDE_START = 0.05, 2.0, 0.15
T_GRID = [0.005 * 1.35**i for i in range(20)]


# ---- the measurements ------------------------------------------------------------------------


@dataclass
class Atom:
    one: float  # one line to depth 18
    one_line: dict[str, Any] | None  # its best line (moves, expected_score)
    rule: float | None  # when the decided stand-in is there (RULE_S), if it would fire
    m12: float  # six lines to depth 12
    six: list[dict[str, Any]]  # the candidates, best first (the six-line search's moves)
    threats: float


def load() -> dict[str, Atom]:
    from tools.engine.decided import _taken

    single = {
        r["key"]: r for r in json.loads((CACHE / "single.json").read_text(encoding="utf-8"))
    }
    decided = {
        r["fen"]: r for r in json.loads((CACHE / "decided.json").read_text(encoding="utf-8"))
    }
    db = sqlite3.connect(CACHE / "clean.sqlite")
    out = {}
    for key, found in db.execute("SELECT key, found FROM clean"):
        f = json.loads(found)
        # The one-line search: single.py's for the first 8,886 positions, measure.py's own since.
        if "1" in f:
            s = {"one": f["1"]["wall"] / SIX_AT_ONCE, "one_lines": f["1"]["lines"]}
        else:
            s = single.get(key)
        if s is None:
            continue
        m12 = next((t for d, t in f["6"]["depths"] if d >= 12), f["6"]["wall"]) / SIX_AT_ONCE
        fen = " ".join(key.split(" ")[:6])
        rule = None
        marks = f["1"].get("marks") if "1" in f else (decided[fen]["marks"] if fen in decided else None)
        if marks:
            # Measured six at once: the times over one search alone, as everything else here.
            alone = [(d, t / SIX_AT_ONCE, s) for d, t, s in marks] if "1" in f else marks
            if _taken(alone, RULE_S) is not None:
                rule = RULE_S
        out[key] = Atom(
            s["one"],
            s["one_lines"][0] if s["one_lines"] else None,
            rule,
            m12,
            f["6"]["lines"],
            f["threats"]["wall"] / SIX_AT_ONCE,
        )
    db.close()
    return out


# ---- the policy, the run ---------------------------------------------------------------------


@dataclass(frozen=True)
class Policy:
    workers: int
    reserve: int
    cands: int
    look: int
    reach: float
    order: str = "newest"
    coach: bool = False  # the AI coach's requests (lessons, questions on either turn)

    def name(self) -> str:
        return (
            f"w{self.workers} r{self.reserve} c{self.cands} look{self.look} "
            f"reach{self.reach} {self.order}"
        )


@dataclass
class Job:
    key: str
    kind: str  # "1", "m12", "threats"
    start: float
    end: float
    worker: int
    ahead: bool


@dataclass
class Pos:
    key: str
    board: chess.Board
    done: dict[str, float] = field(default_factory=dict)
    value_at: float | None = None  # when its score is known (its "1", or the decided stand-in)


class Missing(Exception):
    def __init__(self, boards: dict[str, chess.Board]) -> None:
        super().__init__(f"{len(boards)} unmeasured")
        self.boards = boards


class Run:
    def __init__(self, policy: Policy, atoms: dict[str, Atom], steps: list[Any]) -> None:
        self.p, self.atoms, self.steps = policy, atoms, steps
        self.pos: dict[str, Pos] = {}
        self.path: dict[int, str] = {}
        self.moves: dict[int, tuple[str, str, chess.Move, bool, float]] = {}
        self.current = 0
        self.user: bool | None = None
        self.next_verdict: int | None = None
        self.jobs: dict[int, Job] = {}
        self.now = 0.0
        self.ahead: dict[str, float] = {}
        self.played: dict[bool, list[int | None]] = {True: [], False: []}
        self.missing: dict[str, chess.Board] = {}
        self.verdicts: list[tuple[bool, float]] = []
        self.hint: list[float] = []  # U reached -> its six lines at depth 12
        self.deep: list[float] = []  # U reached -> its "1", threats and candidates all in
        self.left_hint = self.left_deep = 0
        self._open_hint: dict[str, float] = {}
        self._open_deep: dict[str, float] = {}
        self.cpu = self.wasted = 0.0
        self.ahead_cpu: dict[str, float] = {}
        self.skipped: set[str] = set()  # candidate positions never measured, left out
        # The coach's own requests: a lesson's prompt after each user move, a question on the
        # opponent's turn, a move asked about; each (name, the searches it needs, when asked).
        self.requests: list[tuple[str, list[tuple[str, list[str]]], float]] = []
        self.asked: dict[str, list[float]] = {}
        self.superseded = 0

    # -- positions --

    def node(self, board: chess.Board) -> Pos:
        key = cache_key(board)
        p = self.pos.get(key)
        if p is None:
            p = self.pos[key] = Pos(key, board.copy())
            if board.is_game_over():
                p.done = {"1": -1.0, "m12": -1.0, "threats": -1.0}
                p.value_at = -1.0
        return p

    def atom(self, key: str) -> Atom:
        a = self.atoms.get(key)
        if a is None:
            if self.pos[key].board.is_game_over():
                return Atom(0.0, None, None, 0.0, [], 0.0)
            raise Missing({key: self.pos[key].board.copy()})
        return a

    def is_u(self, board: chess.Board) -> bool:
        return self.user is None or board.turn == self.user

    def children(self, key: str) -> list[str]:
        """The keys of a U position's `cands` best moves' positions -- those measured: one the
        measuring rounds never reached is left out and counted (`skipped`), never searched."""
        p = self.pos[key]
        if p.board.is_game_over() or key not in self.atoms:
            return []
        out = []
        for line in self.atoms[key].six[: self.p.cands]:
            if not line["moves"]:
                continue
            move = chess.Move.from_uci(line["moves"][0])
            if move not in p.board.legal_moves:
                continue
            child = p.board.copy()
            child.push(move)
            c = self.node(child)
            if c.key not in self.atoms and not child.is_game_over():
                self.skipped.add(c.key)
                continue
            out.append(c.key)
        return out

    def best_move(self, key: str) -> str | None:
        line = self.atom(key).one_line
        return line["moves"][0] if line and line["moves"] else None

    # -- the game --

    def step(self, s: Any) -> None:
        was = self.path.get(self.current)
        # What the coach asked about a moment the game has moved past is not said any more: its
        # requests are let go (counted), never searched for nothing.
        self.superseded += len(self.requests)
        self.requests = []
        self._step(s)
        now = self.path.get(self.current)
        if was is not None and was != now:
            if self._open_hint.pop(was, None) is not None:
                self.left_hint += 1
            if self._open_deep.pop(was, None) is not None:
                self.left_deep += 1
        on_path = set(self.path.values())
        for job in self.jobs.values():
            if job.key in on_path:
                job.ahead = False
        self.judge()
        self.reframe()

    def _step(self, s: Any) -> None:
        if s.kind in ("start", "resync"):
            self.path.clear()
            self.moves.clear()
            self.next_verdict = None
            self.user = s.user
            if s.kind == "start":
                self.played = {True: [], False: []}
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
            if s.by_user:
                # The lesson on the user's move: the position they moved from whole, and the
                # position after it -- its best line and its threats (the model's prompt).
                self.ask(
                    "lesson",
                    [(before.key, ["1", "m12", "threats"]), (after.key, ["1", "threats"])],
                )
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

    def reach(self, key: str) -> None:
        board = self.pos[key].board
        if self.is_u(board):
            self._open_hint[key] = self.now
            self._open_deep[key] = self.now
            self.tell(key)
            # A move the user (or the model) asks about: the third candidate, its own search.
            a = self.atoms.get(key)
            if a and len(a.six) > 2 and a.six[2]["moves"]:
                move = chess.Move.from_uci(a.six[2]["moves"][0])
                if move in board.legal_moves:
                    child = board.copy()
                    child.push(move)
                    c = self.node(child)
                    if c.key in self.atoms or child.is_game_over():
                        self.ask("question", [(c.key, ["1"])])
                    else:
                        self.skipped.add(c.key)
        elif not board.is_game_over():
            # A question on the opponent's turn, the moment it begins: the model's prompt
            # renders the position whole.
            self.ask("their_turn", [(key, ["1", "m12", "threats"])])

    def ask(self, name: str, needs: list[tuple[str, list[str]]]) -> None:
        """A prompt or an answer that needs these searches: timed from now until all are in."""
        if not self.p.coach:
            return
        self.requests.append((name, needs, self.now))
        self.answer()

    def answer(self) -> None:
        left = []
        for name, needs, at in self.requests:
            if all(k in self.pos[key].done for key, kinds in needs for k in kinds):
                self.asked.setdefault(name, []).append(self.now - at)
            else:
                left.append((name, needs, at))
        self.requests = left

    def deep_done(self, key: str) -> bool:
        p = self.pos[key]
        if not ("1" in p.done and "threats" in p.done):
            return False
        return all("1" in self.pos[c].done for c in self.children(key))

    def tell(self, key: str) -> None:
        p = self.pos[key]
        if key in self._open_hint and "m12" in p.done:
            self.hint.append(max(0.0, min(p.done["m12"], self.now) - self._open_hint.pop(key)))
        if key in self._open_deep and self.deep_done(key):
            self.deep.append(max(0.0, self.now - self._open_deep.pop(key)))

    def judge(self) -> None:
        while self.next_verdict is not None and self.next_verdict in self.moves:
            before, after, move, by_user, at = self.moves[self.next_verdict]
            b = self.pos[before]
            if b.value_at is None:
                return
            best = self.best_move(before) if not b.board.is_game_over() else None
            own = best == move.uci()
            if not own and self.pos[after].value_at is None:
                return
            self.verdicts.append((by_user, self.now - at))
            self.played[by_user].append(0 if own else None)
            self.next_verdict += 1

    # -- the look-ahead --

    def outside(self, player: bool) -> float:
        played = self.played[player]
        return (OUTSIDE_START * T_START_WEIGHT + sum(1 for i in played if i is None)) / (
            T_START_WEIGHT + len(played)
        )

    def odds(self, player: bool, lines: list[dict[str, Any]]) -> list[float]:
        """The best line gets what this game's moves gave the best (the share outside it the
        rest); among several lines, in proportion to exp(-loss / T_START)."""
        if not lines:
            return []
        best = lines[0]["expected_score"]
        w = [math.exp(-max(0.0, best - ln["expected_score"]) / T_START) for ln in lines]
        total = sum(w)
        stay = 1.0 - self.outside(player)
        return [stay * x / total for x in w]

    def lines_of(self, key: str) -> list[dict[str, Any]]:
        """What a position's searches have shown of its moves: a U position's candidates once
        its six lines are in, else its one line."""
        p = self.pos[key]
        a = self.atoms.get(key)
        if a is None:
            return []
        if self.is_u(p.board) and "m12" in p.done:
            return a.six
        if "1" in p.done and a.one_line:
            return [a.one_line]
        return []

    def reframe(self) -> None:
        ahead: dict[str, float] = {}
        root = self.path.get(self.current)
        frontier = [(-1.0, 0, root)] if root else []
        on_path = set(self.path.values())
        while frontier:
            neg, dist, key = heapq.heappop(frontier)
            if dist >= self.p.look:
                continue
            lines = self.lines_of(key)
            if not lines:
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
        wanted = on_path | set(ahead)
        for key in list(on_path) + list(ahead):
            if self.is_u(self.pos[key].board):
                wanted.update(self.children(key))
        for w, job in list(self.jobs.items()):
            if job.key not in wanted:
                spent = self.now - job.start
                self.cpu += spent
                self.wasted += spent
                del self.jobs[w]

    # -- the workers --

    def wants(self, key: str, verdict_only: bool = False) -> list[str]:
        p = self.pos[key]
        if verdict_only or not self.is_u(p.board):
            return ["1"]
        return ["1", "m12", "threats"]

    def todo(self) -> list[tuple[str, str, bool]]:
        busy = {(j.key, j.kind) for j in self.jobs.values()}
        out: list[tuple[str, str, bool]] = []
        seen: set[tuple[str, str]] = set()

        def add(key: str, ahead: bool, verdict_only: bool = False) -> None:
            if key not in self.atoms:
                if ahead:
                    self.missing[key] = self.pos[key].board.copy()
                    return
            p = self.pos[key]
            for kind in self.wants(key, verdict_only):
                if kind in p.done or (key, kind) in busy or (key, kind) in seen:
                    continue
                seen.add((key, kind))
                out.append((key, kind, ahead))
            if not verdict_only and self.is_u(p.board) and "m12" in p.done:
                for c in self.children(key):
                    add(c, ahead, verdict_only=True)

        def verdicts() -> None:
            if self.next_verdict is None:
                return
            for ply in sorted(p for p in self.moves if p >= self.next_verdict):
                before, after, move, *_ = self.moves[ply]
                if self.pos[before].value_at is None:
                    add(before, False, verdict_only=True)
                elif self.best_move(before) != move.uci() and self.pos[after].value_at is None:
                    add(after, False, verdict_only=True)

        def requests() -> None:
            """The coach's requests, oldest first: each search they need."""
            for _, needs, _ in self.requests:
                for key, kinds in needs:
                    if key not in self.atoms:
                        continue
                    p = self.pos[key]
                    for kind in kinds:
                        if kind in p.done or (key, kind) in busy or (key, kind) in seen:
                            continue
                        seen.add((key, kind))
                        out.append((key, kind, False))

        newest = self.path.get(self.current)
        # The verdicts first (the word on each move, in order), then the coach's requests, then
        # the rest of the newest position.
        verdicts()
        requests()
        if newest:
            add(newest, False)
        for ply in sorted(self.path, reverse=True):
            add(self.path[ply], False, verdict_only=self.path[ply] != newest)
        for key, _ in sorted(self.ahead.items(), key=lambda kv: -kv[1]):
            add(key, True)
        return out

    def assign(self) -> None:
        free = [w for w in range(self.p.workers) if w not in self.jobs]
        if not free:
            return
        for key, kind, ahead in self.todo():
            if not free:
                break
            if ahead and sum(1 for j in self.jobs.values() if j.ahead) >= self.p.workers - self.p.reserve:
                continue
            a = self.atom(key)
            dur = {"1": a.one, "m12": a.m12, "threats": a.threats}[kind]
            w = free.pop(0)
            self.jobs[w] = Job(key, kind, self.now, self.now + dur * BESIDE[self.p.workers], w, ahead)

    def finish(self, job: Job) -> None:
        del self.jobs[job.worker]
        spent = job.end - job.start
        self.cpu += spent
        if job.ahead:
            self.ahead_cpu[job.key] = self.ahead_cpu.get(job.key, 0.0) + spent
        p = self.pos[job.key]
        p.done[job.kind] = self.now
        if job.kind == "1" and p.value_at is None:
            p.value_at = self.now
        self.judge()
        self.answer()
        for key in set(self._open_hint) | set(self._open_deep):
            self.tell(key)
        self.reframe()

    def rule_due(self) -> float:
        """The next moment a decided stand-in comes, for a "1" search under way."""
        due = math.inf
        for j in self.jobs.values():
            if j.kind == "1" and self.pos[j.key].value_at is None:
                r = self.atom(j.key).rule
                if r is not None:
                    due = min(due, j.start + r * BESIDE[self.p.workers])
        return due

    def apply_rule(self) -> None:
        for j in self.jobs.values():
            p = self.pos[j.key]
            if j.kind == "1" and p.value_at is None:
                r = self.atom(j.key).rule
                if r is not None and j.start + r * BESIDE[self.p.workers] <= self.now + 1e-12:
                    p.value_at = self.now

    def owed(self) -> bool:
        return any(not a for _, _, a in self.todo()) or any(not j.ahead for j in self.jobs.values())

    def run(self) -> dict[str, Any]:
        steps = sorted(self.steps, key=lambda s: s.t)
        t0 = steps[0].t
        i = 0
        while True:
            self.assign()
            next_end = min((j.end for j in self.jobs.values()), default=math.inf)
            next_step = steps[i].t - t0 if i < len(steps) else math.inf
            due = self.rule_due()
            nxt = min(next_end, next_step, due)
            if nxt == math.inf:
                break
            self.now = nxt
            if due <= min(next_end, next_step) and due < math.inf:
                self.apply_rule()
                self.judge()
                continue
            if next_end <= next_step:
                for job in [j for j in self.jobs.values() if j.end <= self.now]:
                    self.finish(job)
            else:
                self.step(steps[i])
                i += 1
            if i >= len(steps) and not self.owed():
                break
        reached = set(self.path.values()) | {k for m in self.moves.values() for k in m[:2]}
        never = sum(c for k, c in self.ahead_cpu.items() if k not in reached)
        return {
            "verdict_user": [x for u, x in self.verdicts if u],
            "verdict_opp": [x for u, x in self.verdicts if not u],
            "hint": self.hint,
            "deep": self.deep,
            "left_hint": self.left_hint,
            "left_deep": self.left_deep,
            "cpu": self.cpu,
            "wasted": self.wasted + never,
            "wall": max(self.now, steps[-1].t - t0, 1e-9),
            "moves": sum(1 for s in steps if s.kind == "move"),
            "skipped": len(self.skipped),
            "asked": self.asked,
            "unanswered": len(self.requests),
            "superseded": self.superseded,
        }


# ---- the sweep -------------------------------------------------------------------------------


def grid(small: bool = False) -> list[Policy]:
    """Every policy; `small`: the points that can be on the front (one worker never keeps up
    with the game's own work, six only add heat; reach 0.05 searched what was never reached)."""
    workers = [2, 3, 4] if small else [1, 2, 3, 4, 6]
    cands = [0, 6] if small else [0, 3, 6]
    reaches = [0.15, 0.3] if small else [0.05, 0.15, 0.3]
    return [
        Policy(w, r, c, look, reach, order)
        for w, r, c, look, reach, order in itertools.product(
            workers, [0, 1], cands, [0, 1, 2, 3], reaches, ["newest", "verdicts"],
        )
        if r < w and not (look == 0 and reach != 0.15)
    ]


_ATOMS: dict[str, Any] = {}


def _init() -> None:
    _ATOMS["atoms"] = load()


def _one(args: tuple[Policy, dict[str, list[Any]]]) -> tuple[Policy, dict[str, Any], dict]:
    policy, sessions = args
    out, missing = {}, {}
    for name, steps in sessions.items():
        run = Run(policy, _ATOMS["atoms"], steps)
        try:
            out[name] = run.run()
        except Missing as m:
            missing.update(m.boards)
        missing.update(run.missing)
    return policy, out, missing


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from tqdm import tqdm

    from tools.engine.measure import measure, workloads
    from tools.engine.sweep import bursts

    parser = argparse.ArgumentParser()
    parser.add_argument("--missing", action="store_true")
    parser.add_argument("--procs", type=int, default=18)
    parser.add_argument("--small", action="store_true")
    args = parser.parse_args()
    sessions = workloads()
    sessions.update(bursts(sessions))
    policies = grid(args.small)
    print(f"{len(policies)} policies x {len(sessions)} sessions")
    for round_ in range(8 if args.missing else 1):
        results, lacking = [], {}
        with ProcessPoolExecutor(args.procs, initializer=_init) as pool:
            futures = [pool.submit(_one, (p, sessions)) for p in policies]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="policies", file=sys.stdout, mininterval=2):
                policy, out, missing = fut.result()
                results.append((policy, out))
                lacking.update(missing)
        if not lacking:
            break
        print(f"round {round_ + 1}: {len(lacking)} positions unmeasured")
        if not args.missing:
            break
        measure(lacking, f"round {round_ + 1}")
    OUT.write_text(
        json.dumps([{"policy": p.__dict__, "name": p.name(), "sessions": o} for p, o in results]),
        encoding="utf-8",
    )
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
