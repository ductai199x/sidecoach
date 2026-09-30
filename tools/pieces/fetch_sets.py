"""The freely licensed piece sets Sidecoach ships besides its own, fetched from Lichess's
repository (lichess-org/lila, public/piece/<set>/) into desktop/static/pieces/<set>/.

Only sets whose licence allows any use, commercial included, as Lichess's COPYING.md lists
them (checked 2026-09-30): permissive (MIT, Apache 2.0, CC0, CC BY 4.0) and copyleft (GPL,
AGPL, CC BY-SA), the copyleft ones shipped unmodified, their licences covering the piece
files only. None with a non-commercial clause, none whose terms are unclear. Not "mono": its
two sides are drawn alike.

    uv run --no-sync python tools/pieces/fetch_sets.py
"""

from pathlib import Path

import httpx

OUT = Path(__file__).parents[2] / "desktop" / "static" / "pieces"
RAW = "https://raw.githubusercontent.com/lichess-org/lila/master/public/piece/{set}/{code}.svg"
CODES = [c + p for c in "wb" for p in "KQRBNP"]

# name: (author, licence), as lila's COPYING.md gives them.
SETS = {
    "cburnett": ("Colin M.L. Burnett", "GPLv2+"),
    "merida": ("Armando Hernandez Marroquin", "GPLv2+"),
    "chessnut": ("Alexis Luengas", "Apache 2.0"),
    "rhosgfx": ("RhosGFX", "CC0 1.0"),
    "celtic": ("Maurizio Monge", "MIT"),
    "fantasy": ("Maurizio Monge", "MIT"),
    "spatial": ("Maurizio Monge", "MIT"),
    "firi": ("James Faure", "CC BY 4.0"),
    "kiwen-suwi": ("neverRare", "CC BY 4.0"),
    "papercut": ("Nikolay Anzarov", "CC BY 4.0"),
    "totoy": ("Kosal Sen", "CC BY 4.0"),
    "shapes": ("flugsio", "CC BY-SA 4.0"),
    "mpchess": ("Maxime Chupin", "GPLv3+"),
    "pirouetti": ("pirouetti", "AGPLv3+"),
    "pixel": ("therealqtpi", "AGPLv3+"),
    "letter": ("usolando", "AGPLv3+"),
}

if __name__ == "__main__":
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for name in SETS:
            folder = OUT / name
            folder.mkdir(parents=True, exist_ok=True)
            for code in CODES:
                r = client.get(RAW.format(set=name, code=code))
                r.raise_for_status()
                if b"<svg" not in r.content[:2000]:
                    raise SystemExit(f"{name}/{code}: not an SVG")
                (folder / f"{code}.svg").write_bytes(r.content)
            size = sum(f.stat().st_size for f in folder.glob("*.svg"))
            print(f"{name}: 12 pieces, {size // 1024} KB")
