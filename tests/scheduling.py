"""The order the suite's tests go to the workers: the longest first.

xdist's `loadgroup` deals its work units (a test, or the tests of one `xdist_group`) in
collection order, or by how MANY tests a unit holds. Neither knows how long a test takes, so
one long unit dealt late set the wall time: on 2026-10-02 the six geometry seeds, dealt
together as one unit, ran 47 s on one worker while the other eleven had finished at 34 s.
Dealt longest first (the LPT rule), the long tests start at once and the short ones fill in
around them, and the workers end within a short test of each other.

How long a test takes comes from `tests/durations.json`: the tests of a tenth of a second or
more, measured, written by `uv run pytest --store-durations` (tests/conftest.py). A test not
in it is estimated at its function's other cases, else at FAST. A stale table only changes
the order the tests are dealt in, never which tests run or what they check.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import OrderedDict, defaultdict
from pathlib import Path

import pytest
from xdist.scheduler import LoadGroupScheduling

DURATIONS = Path(__file__).with_name("durations.json")
# The table keeps the tests at least this long; the rest are dealt last, as FAST.
STORED = 0.1
FAST = 0.02
# Tests a worker holds beyond the one it runs. xdist keeps up to two more, dealt as soon as
# a worker is down to two: a long test dealt to a busy worker then waits there while another
# worker idles. One is enough to keep a worker from waiting on the controller.
QUEUED = 1


def plain_id(nodeid: str) -> str:
    """The test's id without xdist's `@group` suffix (a group can change; the test not)."""
    if nodeid.rfind("@") > nodeid.rfind("]"):
        return nodeid[: nodeid.rfind("@")]
    return nodeid


def _function(nodeid: str) -> str:
    """The test function's id without the parameters of the case."""
    return re.sub(r"\[.*\]$", "", nodeid)


def load_durations(path: Path = DURATIONS) -> dict[str, float]:
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): float(v) for k, v in table.items() if isinstance(v, (int, float))}


def store_durations(measured: dict[str, float], path: Path = DURATIONS) -> None:
    """This run's seconds in place of the table's: the tests of STORED seconds or more,
    rounded to a tenth, one per line in id order (a diff shows which tests got slower)."""
    table = {plain_id(k): round(v, 1) for k, v in measured.items() if round(v, 1) >= STORED}
    lines = ",\n".join(f"  {json.dumps(k)}: {v}" for k, v in sorted(table.items()))
    path.write_text("{\n" + lines + "\n}\n", encoding="utf-8", newline="\n")


class Estimates:
    """Seconds for any test id: the table's, else its function's other cases', else FAST."""

    def __init__(self, table: dict[str, float]) -> None:
        self.table = table
        cases: dict[str, list[float]] = defaultdict(list)
        for nodeid, seconds in table.items():
            cases[_function(nodeid)].append(seconds)
        self.function = {f: statistics.median(v) for f, v in cases.items()}

    def __call__(self, nodeid: str) -> float:
        plain = plain_id(nodeid)
        if plain in self.table:
            return self.table[plain]
        return self.function.get(_function(plain), FAST)


class LongestFirst(LoadGroupScheduling):
    """`loadgroup` (an `xdist_group`'s tests stay on one worker, in order), with the work
    units dealt longest first and each worker holding QUEUED tests beyond its own."""

    def __init__(self, config: pytest.Config, log=None) -> None:  # noqa: ANN001
        super().__init__(config, log)
        self.estimate = Estimates(load_durations())

    def schedule(self) -> None:
        assert self.collection_is_completed
        if self.collection is not None:
            for node in self.nodes:
                self._reschedule(node)
            return
        if not self._check_nodes_have_same_collection():
            self.log("**Different tests collected, aborting run**")
            return
        self.collection = list(next(iter(self.registered_collections.values())))
        if not self.collection:
            return
        units: OrderedDict[str, dict[str, bool]] = OrderedDict()
        for nodeid in self.collection:
            units.setdefault(self._split_scope(nodeid), {})[nodeid] = False
        cost = {scope: sum(self.estimate(n) for n in tests) for scope, tests in units.items()}
        # Longest first; ties in collection order (sorted is stable).
        for scope in sorted(units, key=lambda s: -cost[s]):
            self.workqueue[scope] = units[scope]
        for _ in range(max(0, len(self.nodes) - len(self.workqueue))):
            unused, _assigned = self.assigned_work.popitem()
            unused.shutdown()
        for node in self.nodes:
            self._assign_work_unit(node)
        for node in self.nodes:
            self._reschedule(node)
        if not self.workqueue:
            for node in self.nodes:
                node.shutdown()

    def _reschedule(self, node) -> None:  # noqa: ANN001
        if node.shutting_down:
            return
        if not self.workqueue:
            node.shutdown()
            return
        if self._pending_of(self.assigned_work[node]) > QUEUED:
            return
        self._assign_work_unit(node)
