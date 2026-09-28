"""Opening identification and book deviation.

Everything but the last test runs off `openings_sample.tsv`, a handful of real Lichess
lines copied into a temporary folder, so the suite never needs the download. The last
test uses the fetched tables when they are there and is skipped when they are not.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import chess
import pytest

from sidecoach.analysis import openings as openings_module
from sidecoach.analysis.openings import CACHE_FILE, Opening, OpeningBook

SAMPLE = Path(__file__).parent / "openings_sample.tsv"
REAL_TABLES = Path(__file__).resolve().parents[1] / "data" / "openings"


def moves(line: str) -> list[chess.Move]:
    """SAN line to the moves a game would hand the book."""
    board = chess.Board()
    played: list[chess.Move] = []
    for token in line.split():
        if token[0].isdigit():
            continue
        played.append(board.push_san(token))
    return played


def sample_folder(tmp_path: Path) -> Path:
    folder = tmp_path / "openings"
    folder.mkdir()
    shutil.copy(SAMPLE, folder / "c.tsv")
    return folder


@pytest.fixture
def book(tmp_path: Path) -> OpeningBook:
    loaded = OpeningBook.load(sample_folder(tmp_path))
    assert loaded is not None
    return loaded


def test_missing_data_is_not_an_error(tmp_path: Path) -> None:
    assert OpeningBook.load(tmp_path / "never-fetched") is None
    empty = tmp_path / "empty"
    empty.mkdir()
    assert OpeningBook.load(empty) is None


def test_fixture_loads(book: OpeningBook) -> None:
    assert len(book) == 15


def test_identify_takes_the_longest_match_and_grows_more_specific(book: OpeningBook) -> None:
    assert book.identify(moves("1. e4")) == Opening("B00", "King's Pawn Game", 1)
    assert book.identify(moves("1. e4 e5")) == Opening("C20", "King's Pawn Game", 2)
    assert book.identify(moves("1. e4 e5 2. Nf3")) == Opening("C40", "King's Knight Opening", 3)
    assert book.identify(moves("1. e4 e5 2. Nf3 Nc6 3. Bc4")) == Opening("C50", "Italian Game", 5)
    assert book.identify(moves("1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. d3")) == Opening(
        "C50", "Italian Game: Giuoco Pianissimo", 7
    )


def test_unnamed_position_keeps_the_last_name(book: OpeningBook) -> None:
    """3...Bc5 is in book but unnamed, so the game is still the Italian Game."""
    line = moves("1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5")
    assert book.in_book(line)
    assert book.identify(line) == Opening("C50", "Italian Game", 5)


def test_empty_move_list(book: OpeningBook) -> None:
    assert book.identify([]) is None
    assert book.in_book([]) is True
    assert book.deviation([]) is None


def test_deviation_mid_game(book: OpeningBook) -> None:
    line = moves("1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. Nc3")
    assert book.in_book(line) is False
    assert book.deviation(line) == 6
    assert book.identify(line) == Opening("C50", "Italian Game", 5)


def test_deviation_at_the_very_first_move(book: OpeningBook) -> None:
    line = moves("1. g4")
    assert book.deviation(line) == 0
    assert book.in_book(line) is False
    assert book.identify(line) is None
    assert book.identify(moves("1. g4 d5")) is None


def test_game_that_never_leaves_the_book(book: OpeningBook) -> None:
    najdorf = moves("1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6")
    assert book.deviation(najdorf) is None
    assert book.in_book(najdorf) is True
    assert book.identify(najdorf) == Opening("B90", "Sicilian Defense: Najdorf Variation", 10)


def test_transposition_matches_by_position(book: OpeningBook) -> None:
    """Both move orders reach the same position, so both get the same name."""
    direct = book.identify(moves("1. d4 d5 2. c4 e6 3. Nc3 Nf6"))
    indian_order = book.identify(moves("1. d4 Nf6 2. c4 e6 3. Nc3 d5"))
    assert direct == Opening("D35", "Queen's Gambit Declined: Normal Defense", 6)
    assert indian_order == direct


def test_cache_is_written_and_reused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    folder = sample_folder(tmp_path)
    first = OpeningBook.load(folder)
    assert first is not None
    assert (folder / CACHE_FILE).is_file()

    def explode(_tables: object) -> None:
        raise AssertionError("the cache should have answered")

    monkeypatch.setattr(openings_module, "_compile", explode)
    second = OpeningBook.load(folder)
    assert second is not None
    italian = moves("1. e4 e5 2. Nf3 Nc6 3. Bc4")
    assert second.identify(italian) == first.identify(italian)


def test_changed_tables_recompile(tmp_path: Path) -> None:
    folder = sample_folder(tmp_path)
    assert OpeningBook.load(folder) is not None
    (folder / "c.tsv").write_text("eco\tname\tpgn\nA00\tAmar Opening\t1. Nh3\n", encoding="utf-8")

    rebuilt = OpeningBook.load(folder)
    assert rebuilt is not None
    assert rebuilt.identify(moves("1. Nh3")) == Opening("A00", "Amar Opening", 1)
    assert rebuilt.identify(moves("1. e4")) is None


def test_corrupt_cache_is_ignored(tmp_path: Path) -> None:
    folder = sample_folder(tmp_path)
    assert OpeningBook.load(folder) is not None
    (folder / CACHE_FILE).write_text("{not json", encoding="utf-8")

    recovered = OpeningBook.load(folder)
    assert recovered is not None
    assert recovered.identify(moves("1. e4 e5 2. Nf3 Nc6 3. Bc4")) == Opening(
        "C50", "Italian Game", 5
    )


def test_unparsable_line_is_skipped(tmp_path: Path) -> None:
    folder = tmp_path / "openings"
    folder.mkdir()
    (folder / "x.tsv").write_text(
        "eco\tname\tpgn\nA00\tNonsense\t1. Qz9\nA00\tAmar Opening\t1. Nh3\n", encoding="utf-8"
    )
    loaded = OpeningBook.load(folder)
    assert loaded is not None
    assert loaded.identify(moves("1. Nh3")) == Opening("A00", "Amar Opening", 1)


@pytest.mark.skipif(
    not (REAL_TABLES / "a.tsv").is_file(),
    reason="run scripts/fetch_openings.py for the real tables",
)
def test_real_tables() -> None:
    book = OpeningBook.load(REAL_TABLES)
    assert book is not None
    assert len(book) > 3500

    started = time.perf_counter()
    warm = OpeningBook.load(REAL_TABLES)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert warm is not None
    assert elapsed_ms < 300, f"warm load took {elapsed_ms:.0f} ms"

    named = {
        "1. e4 e5 2. Nf3 Nc6 3. Bc4": ("C50", "Italian Game"),
        "1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. d3": ("C50", "Italian Game: Giuoco Pianissimo"),
        "1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6": (
            "B90",
            "Sicilian Defense: Najdorf Variation",
        ),
        "1. e4 e6 2. d4 d5": ("C00", "French Defense"),
        "1. e4 e6 2. d4 d5 3. Nc3 Bb4": ("C15", "French Defense: Winawer Variation"),
        "1. d4 d5 2. c4 e6": ("D30", "Queen's Gambit Declined"),
        "1. e4 e5 2. Nf3 Nc6 3. Bb5": ("C60", "Ruy Lopez"),
        "1. e4 e5 2. Nf3 Nc6 3. Bb5 Nf6": ("C65", "Ruy Lopez: Berlin Defense"),
    }
    for line, (eco, name) in named.items():
        found = book.identify(moves(line))
        assert found is not None, line
        assert (found.eco, found.name) == (eco, name), line

    # Same position by two move orders, one of them through the Indian move order.
    assert book.identify(moves("1. d4 Nf6 2. c4 e6 3. Nc3 d5")) == book.identify(
        moves("1. d4 d5 2. c4 e6 3. Nc3 Nf6")
    )

    # A real game that walks out of theory: 4. h4 is nobody's line.
    off_theory = moves("1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. h4")
    assert book.deviation(off_theory) == 6
    assert book.in_book(moves("1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5")) is True
