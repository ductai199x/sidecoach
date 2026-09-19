"""Suite-wide options: `--seeds N` (or `--seeds A-B`) widens the random-game stress sweep
(tests/test_stress.py)."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--seeds",
        action="store",
        default="",
        help="how many seeded random games the stress tests play, or a range A-B of seeds "
        "(default: a fixed handful)",
    )
