"""The page's own API client (desktop/src/lib/api.ts), run under node: its tests are in
desktop/tests/*.test.mjs, with the desktop shell's commands and the network faked. The page has
no test runner of its own; node runs its TypeScript as it is."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

DESKTOP = Path(__file__).resolve().parents[1] / "desktop"


def test_the_page_follows_a_restarted_sidecar() -> None:
    """The panel's Restart starts a new sidecar on another port; every other window kept
    calling the old one, dead until reopened (the round-3 UI audit, 2026-10-02)."""
    node = shutil.which("node")
    if node is None or not (DESKTOP / "node_modules" / "@tauri-apps" / "api").exists():
        pytest.skip("the page's packages are not installed (npm ci in desktop/)")
    run = subprocess.run(
        [node, "--test", "tests/api_follow.test.mjs"],
        cwd=DESKTOP,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]
