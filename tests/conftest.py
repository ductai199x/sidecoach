"""Suite-wide options: `--seeds N` widens the random-game stress sweep (tests/test_stress.py)."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--seeds",
        action="store",
        default="0",
        help="how many seeded random games the stress tests play (default: a fixed handful)",
    )
