"""The puzzle sample Setup fetches (setup.items, "puzzles"): the Lichess file streamed and
decompressed, a few puzzles kept per theme, the progress in the item's own units."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from tests.fakes import PUZZLE_HEADER

FEN = "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1"


def _row(i: int, themes: str) -> str:
    return f"p{i:05d},{FEN},g8h8 a1a8,{900 + i % 700},70,90,100,{themes},https://lichess.org/x#1,"


def test_the_sample_keeps_a_few_per_theme_and_reports_in_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import zstandard

    import sidecoach.setup as setup
    from sidecoach.config import Settings

    lines = [PUZZLE_HEADER]
    # Three forks, two pins (one row is both): the pins never fill, so the scan runs on to its
    # limit through rows nothing wants.
    lines += [_row(0, "fork short"), _row(1, "pin long"), _row(2, "fork pin"), _row(3, "fork")]
    lines += [_row(i, "endgame") for i in range(4, 40_000)]
    blob = zstandard.ZstdCompressor().compress(("\n".join(lines) + "\n").encode("utf-8"))

    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *exc: object) -> None:
            self.close()

    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **k: Response(blob))
    monkeypatch.setattr(setup, "PUZZLE_THEMES", ["fork", "pin"])
    monkeypatch.setattr(setup, "PUZZLES_PER_THEME", 3)
    monkeypatch.setattr(setup, "PUZZLES_MAX_SCAN", 40_000)

    item = next(i for i in setup.items(Settings(data_dir=tmp_path)) if i.id == "puzzles")
    assert item.optional and not item.present()
    seen: list[tuple[int, int | None]] = []
    item.fetch(lambda done, total: seen.append((done, total)))

    assert item.present()
    written = (tmp_path / "puzzles" / "sample.csv").read_text(encoding="utf-8").splitlines()
    # The forks, then the pins; a puzzle of both is written once, where it first came.
    assert written[0] == PUZZLE_HEADER
    assert [line.split(",")[0] for line in written[1:]] == ["p00000", "p00002", "p00003", "p00001"]
    # The bar moves in the item's bytes, never past its size, and ends on it.
    assert seen == [(item.size // 2, item.size), (item.size, item.size)]
