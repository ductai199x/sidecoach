"""Wiping what the app keeps: the profile, and the caches.

Two things a person wants to be able to throw away (Tai, 2026-09-20): the player model
-- everything the profile learned, every game, every attempt, the puzzle level -- and
the caches that only speed the next run up: the theme cache (a recogniser per board
palette, rebuilt from the next start position) and the puzzle store (rebuilt from the
CSV beside it). Both from the CLI (`sidecoach profile reset`, `sidecoach cache clear`)
and from the panel. Recordings, sessions and the eval data are not caches and are not
touched.
"""

from __future__ import annotations

import contextlib
import sqlite3
from pathlib import Path
from typing import Any

from sidecoach.config import Settings


def reset_profile(settings: Settings) -> Path:
    """Empty the profile database in place (every table), so a store open on it -- the
    running panel's -- stays valid and simply finds nothing. Returns the path."""
    path = settings.data_dir / "profile.sqlite"
    if not path.exists():
        return path
    db = sqlite3.connect(str(path))
    try:
        tables = [
            row[0]
            for row in db.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        for table in tables:
            db.execute(f'DELETE FROM "{table}"')  # noqa: S608 - names from the schema
        db.commit()
        with contextlib.suppress(sqlite3.OperationalError):  # another connection busy
            db.execute("VACUUM")
    finally:
        db.close()
    return path


def clear_caches(settings: Settings, puzzles: Any | None = None) -> list[str]:
    """Remove the caches under the data directory; returns what was done, in words.

    The theme files are unlinked. The puzzle store, when a running process holds it
    open (`puzzles`), is emptied and filled again from the CSVs beside it rather than
    unlinked under that process; from the CLI the file goes and the next start rebuilds
    it."""
    done: list[str] = []
    for path in sorted(settings.data_dir.glob("themes/theme-*.pkl")):
        with contextlib.suppress(OSError):
            path.unlink()
            done.append(f"removed {path.name}")
    if puzzles is not None:
        puzzles.clear()
        added = 0
        for csv_path in sorted((settings.data_dir / "puzzles").glob("*.csv")):
            added += puzzles.import_csv(csv_path)
        done.append(f"puzzle store rebuilt: {added} puzzles")
    else:
        for name in ("puzzles.sqlite", "puzzles.sqlite-journal"):
            path = settings.data_dir / name
            if path.exists():
                with contextlib.suppress(OSError):
                    path.unlink()
                    done.append(f"removed {name}")
    return done
