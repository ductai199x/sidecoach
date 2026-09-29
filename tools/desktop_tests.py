"""The desktop's test functions as the fixture generators run them: each plain test, and each
case of a parametrized one, called with its arguments.

The desktop's tests became tables on 2026-09-28 (9538a14, 5352217), and the generators, which
skipped any test that takes parameters, replayed 7 of the claim tests' 64: a test the
generators cannot call (a fixture argument, stacked tables) is named on stderr, never dropped
silently.
"""

from __future__ import annotations

import functools
import inspect
import sys
from collections.abc import Callable, Iterator
from types import ModuleType


def desktop_tests(
    module: ModuleType, skip: frozenset[str] = frozenset()
) -> Iterator[tuple[str, Callable[[], None]]]:
    """(name, call) for every test of `module` in name order; a table's cases as
    "name[id]"."""
    for name, fn in inspect.getmembers(module, inspect.isfunction):
        if not name.startswith("test_") or fn.__module__ != module.__name__ or name in skip:
            continue
        params = inspect.signature(fn).parameters
        if not params:
            yield name, fn
            continue
        tables = [m for m in getattr(fn, "pytestmark", []) if m.name == "parametrize"]
        argnames = _argnames(tables[0].args[0]) if len(tables) == 1 else []
        if set(argnames) != set(params):
            print(f"desktop_tests: {module.__name__}.{name} not run", file=sys.stderr)
            continue
        ids = tables[0].kwargs.get("ids")
        for i, case in enumerate(tables[0].args[1]):
            values = getattr(case, "values", None)  # pytest.param(...)
            if values is None:
                values = (case,) if len(argnames) == 1 else tuple(case)
            label = ids[i] if isinstance(ids, (list, tuple)) else str(i)
            yield f"{name}[{label}]", functools.partial(fn, **dict(zip(argnames, values)))


def _argnames(names: str | tuple[str, ...] | list[str]) -> list[str]:
    if isinstance(names, str):
        return [n.strip() for n in names.split(",") if n.strip()]
    return list(names)
