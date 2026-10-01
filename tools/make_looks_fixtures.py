"""Record what the desktop's `looks.py` gives, for the phone's port (ui/Looks.kt): the catalogue
the settings page offers, the look for every set and theme (and names it does not know), which
piece a file's name says, an image's type from its bytes, and a player's own pieces saved
(taken, skipped and why, missing), read back, and cleared.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_looks_fixtures.py
"""

from __future__ import annotations

import base64
import json
import sys
import tempfile
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP / "src"))

from sidecoach import looks  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "ui" / "looks.json"

NAMES = [
    "wK.png", "bq.svg", "w_n.png", "b-p.webp", "w r.jpg", "white-knight.png", "knight_black.svg",
    "Chess_nlt45.svg", "Chess_qdt45.svg", "chess_kdt.png", "WhiteKing.PNG", "black_queen.svg",
    "white.png", "king.png", "white_king_queen.png", "wx.png", "wKK.png", "folder/wB.svg",
    "C:\\pieces\\bR.png", ".png", "wk", "pawn-white.svg", "BLACK BISHOP.png", "bn.tar.gz", "",
]  # fmt: skip
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"\x00" * 8
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 45 45"><circle r="5"/></svg>'
SAMPLES = {
    "png": PNG,
    "jpeg": JPEG,
    "webp": WEBP,
    "svg": SVG,
    "xml svg": b'<?xml version="1.0"?>\n' + SVG,
    "comment svg": b"<!-- made by hand -->\n" + SVG,
    "spaced svg": b"\n\t  " + SVG,
    "xml not svg": b'<?xml version="1.0"?><html></html>',
    "text": b"hello there",
    "empty": b"",
    "riff not webp": b"RIFF\x10\x00\x00\x00WAVEfmt ",
}


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        empty_catalogue = looks.catalogue(data_dir)
        looks_ = [
            {"pieces": p, "theme": t, "look": looks.look(p, t, data_dir)}
            for p in [*looks.PIECE_SETS, "custom", "nonesuch"]
            for t in [*looks.BOARD_THEMES, "nonesuch"]
        ]
        saves = []
        rounds = [
            [{"name": "wK.png", "data": b64(PNG)}, {"name": "Chess_qdt45.svg", "data": b64(SVG)}],
            [
                {"name": "notes.txt", "data": b64(b"hi")},
                {"name": "bP.png", "data": "not base64!!"},
                {"name": "bN.png", "data": b64(b"plain text")},
                {"name": "white-knight.webp", "data": b64(WEBP)},
                {"name": "wK.jpg", "data": b64(JPEG)},  # in place of the PNG
            ],
            [],
        ]
        for files in rounds:
            result = looks.save_custom(data_dir, files)
            saves.append(
                {
                    "files": files,
                    "result": result,
                    "images": looks.custom_images(data_dir),
                    "look": looks.look("custom", "green", data_dir),
                    "catalogue": looks.catalogue(data_dir),
                }
            )
        looks.clear_custom(data_dir)
        cleared = looks.catalogue(data_dir)
    fixture = {
        "catalogue": empty_catalogue,
        "looks": looks_,
        "codes": [[n, looks.piece_code(n)] for n in NAMES],
        "sniff": [[name, b64(data), looks.sniff(data)] for name, data in SAMPLES.items()],
        "saves": saves,
        "cleared": cleared,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"{OUT}: {len(looks_)} looks, {len(NAMES)} names, {len(SAMPLES)} images, {len(saves)} saves")


if __name__ == "__main__":
    main()
