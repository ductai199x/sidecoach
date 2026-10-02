"""The page's own API client (desktop/src/lib/api.ts), run under node: its tests are in
desktop/tests/*.test.mjs, with the desktop shell's commands and the network faked. The page has
no test runner of its own; node runs its TypeScript as it is."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

DESKTOP = Path(__file__).resolve().parents[1] / "desktop"


def _node_test(name: str) -> None:
    node = shutil.which("node")
    if node is None or not (DESKTOP / "node_modules" / "@tauri-apps" / "api").exists():
        pytest.skip("the page's packages are not installed (npm ci in desktop/)")
    run = subprocess.run(
        [node, "--test", f"tests/{name}"],
        cwd=DESKTOP,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]


def test_the_page_follows_a_restarted_sidecar() -> None:
    """The panel's Restart starts a new sidecar on another port; every other window kept
    calling the old one, dead until reopened (the round-3 UI audit, 2026-10-02)."""
    _node_test("api_follow.test.mjs")


def test_stopped_only_when_the_process_ended() -> None:
    """The banner "Sidecoach stopped" rested on 8 failed polls in a row, so a live sidecar answering a
    500 for two seconds read as stopped, and its Restart killed a live session. Now only the
    shell's word that the process ended says stopped; an error answer shows the error (Tai,
    2026-10-02)."""
    _node_test("api_stopped.test.mjs")
