"""Sample the Lichess puzzle database, balanced across tactical themes.

The full database is ~300 MB of zstd-compressed CSV. We stream and decompress it on the
fly and keep up to K puzzles per theme of interest, stopping as soon as every bucket is
full, so only the leading part of the file is ever downloaded. The sample is ground truth
for validating our motif tagger, our engine's solutions and describe_tactic; it is written
under data/puzzles/ and never committed.

Run:  uv run --with zstandard python tools/puzzles/fetch.py
Columns: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
from pathlib import Path

import zstandard

URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
OUT = Path("data/puzzles/sample.csv")

# The motif themes our tagger emits and Lichess also tags, so a match is meaningful. Move
# types (capture, check) and difficulty tags (crushing, advantage, short) are not here.
THEMES = [
    "fork",
    "pin",
    "skewer",
    "discoveredAttack",
    "hangingPiece",
    "trappedPiece",
    "deflection",
    "attraction",
    "interference",
    "clearance",
    "exposedKing",
    "sacrifice",
    "advancedPawn",
    "promotion",
    "xRayAttack",
    "capturingDefender",
    "doubleCheck",
    "backRankMate",
    "smotheredMate",
    "anastasiaMate",
    "arabianMate",
    "bodenMate",
    "hookMate",
    "dovetailMate",
    "mateIn1",
    "mateIn2",
    "mateIn3",
]
PER_THEME = 200
MAX_SCAN = 600_000  # bound the download: stop scanning after this many rows regardless


def main() -> int:
    per_theme = PER_THEME
    max_scan = MAX_SCAN
    if len(sys.argv) > 1:
        per_theme = int(sys.argv[1])
    kept: dict[str, list[list[str]]] = {theme: [] for theme in THEMES}
    seen: set[str] = set()
    OUT.parent.mkdir(parents=True, exist_ok=True)

    dctx = zstandard.ZstdDecompressor()
    scanned = 0
    with urllib.request.urlopen(URL) as response:  # noqa: S310  (trusted host)
        reader = dctx.stream_reader(response)
        text = io.TextIOWrapper(reader, encoding="utf-8", newline="")
        rows = csv.reader(text)
        header = next(rows)
        for row in rows:
            scanned += 1
            if scanned % 50_000 == 0:
                full = sum(1 for t in THEMES if len(kept[t]) >= per_theme)
                print(f"  scanned {scanned:,}  buckets full {full}/{len(THEMES)}", flush=True)
            themes = set(row[7].split())
            wanted = [t for t in THEMES if t in themes and len(kept[t]) < per_theme]
            if not wanted or row[0] in seen:
                if all(len(kept[t]) >= per_theme for t in THEMES) or scanned >= max_scan:
                    break
                continue
            seen.add(row[0])
            for theme in wanted:
                kept[theme].append(row)
            if all(len(kept[t]) >= per_theme for t in THEMES) or scanned >= max_scan:
                break

    unique = {row[0]: row for rows_ in kept.values() for row in rows_}
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(unique.values())
    print(f"scanned {scanned:,} rows; wrote {len(unique):,} unique puzzles to {OUT}")
    for theme in THEMES:
        print(f"  {theme:18s} {len(kept[theme]):4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
