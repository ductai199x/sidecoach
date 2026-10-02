"""Guards on the review page's source (desktop/src/routes/review/[key]/+page.svelte) for rules
the page alone keeps; the page has no test runner of its own. Each names the finding it guards
(the review of 2026-10-01)."""

from __future__ import annotations

import re
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1] / "desktop" / "src"
PAGE = DESKTOP / "routes" / "review" / "[key]" / "+page.svelte"


def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_the_follow_up_moves_the_eval_bar_to_the_position_after_the_move() -> None:
    """The follow-up starts from the position after the move; the bar stayed at the one before
    it, as for the best line and a retry."""
    page = _page()
    bar = re.search(r"const barScore = \$derived\((.*)\);", page)
    assert bar is not None
    assert "line || retry" not in bar.group(1), "every line read as from before the move"
    assert re.search(r'const fromBefore = \$derived\(line\?\.kind === "best" \|\| !!retry\)', page)


def test_retry_is_the_players_own_moves_only() -> None:
    """Retry was offered on the opponent's mistakes, and the opponent had an accuracy "after
    retries"."""
    page = _page()
    button = re.search(r'<button class="rv-tool" disabled=\{([^}]*)\} title="Find a better', page)
    assert button is not None and "!mine(move)" in button.group(1)
    assert "if (!move || !mine(move)) return;" in page
    retries = page[page.index("const afterRetries") :][:600]
    assert "side !== room.game.user_side) continue" in retries


def test_move_numbers_come_from_the_position_not_the_ply() -> None:
    """A game set up from a position (Black to move at move 23) was numbered from "1." by its
    plies, its pairs off by one."""
    for path in (
        PAGE,
        DESKTOP / "lib" / "review" / "AnalysisPane.svelte",
        DESKTOP / "routes" / "stats" / "+page.svelte",
    ):
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"Math\.ceil\(\w+\.ply / 2\)", source), path.name


def test_stats_filter_every_time_class_your_games_does() -> None:
    """Stats had no Classical filter, though Your games has one."""
    stats = (DESKTOP / "routes" / "stats" / "+page.svelte").read_text(encoding="utf-8")
    games = (DESKTOP / "routes" / "games" / "+page.svelte").read_text(encoding="utf-8")
    speeds = re.search(r"const SPEEDS = \[([^\]]*)\]", games)
    assert speeds is not None
    for speed in re.findall(r'"(\w+)"', speeds.group(1)):
        assert f'["{speed}", ' in stats, speed


def test_a_sparkline_never_blows_a_small_change_up() -> None:
    """Each sparkline scaled to its own range: 81.0 to 81.6 filled the height as a big climb."""
    spark = (DESKTOP / "lib" / "stats" / "Sparkline.svelte").read_text(encoding="utf-8")
    assert "Math.max(most - least, span)" in spark
    assert "(hi - lo)" not in spark
