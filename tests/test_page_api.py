"""The page's own API client (desktop/src/lib/api.ts), run under node: its tests are in
desktop/tests/*.test.mjs, with the desktop shell's commands and the network faked. The page has
no test runner of its own; node runs its TypeScript as it is."""

from __future__ import annotations

import re
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


def test_the_captions_box_goes_through_a_long_line_at_a_reading_pace() -> None:
    """A line longer than the captions' box was shown by its end only (voice finding 18); now the
    box goes through it row by row at a reading pace, then the next line (Tai, 2026-10-02).
    desktop/tests/captions_pace.test.mjs, on src/lib/captions.ts alone (no packages needed)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    run = subprocess.run(
        [node, "--test", "tests/captions_pace.test.mjs"],
        cwd=DESKTOP,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]


def test_a_review_move_says_why_it_got_its_label() -> None:
    """Why this label (the audit of 2026-10-07, features #1): Game Review's move card says in one
    line what earned the move its label, from the numbers the review kept.
    desktop/tests/why.test.mjs, on src/lib/review/why.ts alone (no packages needed)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    run = subprocess.run(
        [node, "--test", "tests/why.test.mjs"],
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
    """The banner "Sidecoach stopped" rested on 8 failed polls in a row, so a live sidecar
    answering a 500 for two seconds read as stopped, and its Restart killed a live session.
    Now only the shell's word that the process ended says stopped; an error answer shows the
    error (Tai, 2026-10-02)."""
    _node_test("api_stopped.test.mjs")


def test_an_error_answer_or_none_in_time_is_not_success() -> None:
    """get/post/put/del never checked `r.ok`: a 4xx JSON error read as success, a plain-text 500
    threw a SyntaxError, and with no timeout a hung sidecar froze the panel's poll on stale
    state with no offline banner (the 2026-10-07 audit, #7)."""
    _node_test("api_errors.test.mjs")


def test_an_analysis_error_goes_once_the_analysis_answers_again() -> None:
    """Game Review's Analysis kept an error on screen after the analysis answered again (the
    2026-10-07 audit): what it says now belongs to the position asked
    (desktop/tests/analysis_error.test.mjs; TypeScript from the page's packages)."""
    _node_test("analysis_error.test.mjs")


def test_a_line_is_stepped_never_run_by_itself() -> None:
    """Give up played a puzzle's solution itself, a move every 400 ms (the 2026-10-07 audit,
    #8). The review's line stepper is now one module the puzzle shares: its first move, then
    only what the player asks (desktop/tests/lineplayer.test.mjs, no packages needed)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    run = subprocess.run(
        [node, "--test", "tests/lineplayer.test.mjs"],
        cwd=DESKTOP,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    assert run.returncode == 0, run.stdout[-3000:] + run.stderr[-2000:]


def _function(source: str, name: str) -> str:
    """The body of `function name(...)` (async or not) in a page's script: up to the next
    function at the same indentation."""
    start = re.search(rf"\n(\s*)(?:async )?function {name}\(", source)
    assert start is not None, name
    indent = start.group(1)
    rest = source[start.end() :]
    end = re.search(rf"\n{indent}(?:async )?function \w+\(|\n{indent}// ----", rest)
    return rest[: end.start()] if end else rest


@pytest.mark.parametrize(
    "page", ["lib/Composer.svelte", "routes/puzzle/[id]/+page.svelte"], ids=["panel", "puzzle"]
)
def test_a_typed_question_stays_in_the_box_until_it_is_sent(page: str) -> None:
    """The composer cleared the box before the request, so a question that never reached the
    coach was lost (the 2026-10-07 audit, #7): the box clears only after the request resolved,
    and a failure says "Not sent" with Retry."""
    source = (DESKTOP / "src" / page).read_text(encoding="utf-8")
    send = _function(source, "send")
    cleared = send.index('text = ""')
    assert re.search(r"await (store\.)?ask\(t\)", send[:cleared]), "cleared before it was sent"
    assert "Not sent" in source and "Retry" in source


@pytest.mark.parametrize(
    "page", ["lib/panel.svelte.ts", "routes/puzzle/[id]/+page.svelte"], ids=["panel", "puzzle"]
)
def test_a_poll_follows_a_sidecar_started_again(page: str) -> None:
    """The polls kept the highest cursor seen, so after the sidecar was stopped and started the
    new one's lines (a question's reply among them) were never read and the typing dots stayed
    (found running the panel for the 2026-10-07 audit, #7): `followCursor` (api.ts) reads a
    sidecar whose cursor went back from its start."""
    source = (DESKTOP / "src" / page).read_text(encoding="utf-8")
    assert "followCursor(" in source
    assert not re.search(r"Math\.max\([\w.]*cursor", source)


def test_give_up_shows_the_solution_stepped() -> None:
    """The puzzle page's Give up ran the solution at 400 ms a move with no catch and no busy
    guard (the 2026-10-07 audit, #8): now the room's given-up state starts the shared line
    player (its first move, then stepped), the auto-advance stops there, and the button is
    guarded and its failure said."""
    source = (DESKTOP / "src" / "routes" / "puzzle" / "[id]" / "+page.svelte").read_text(
        encoding="utf-8"
    )
    assert "new LinePlayer(" in source and "<LineControls" in source
    advance = _function(source, "advance")
    assert re.search(r"if \(!room \|\| solution \|\|", advance), "the solution still runs itself"
    assert 's.status === "given_up" && !solution' in _function(source, "render")
    give_up = _function(source, "giveUp")
    assert "givingUp" in give_up and "catch" in give_up
