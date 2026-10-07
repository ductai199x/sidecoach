"""Record what the desktop's `looks.py` gives, for the phone's port (ui/Looks.kt): the catalogue
the settings page offers, the look for every set and theme (and names it does not know), which
piece a file's name says, an image's type from its bytes, and a player's own sets step by
step: the one set from before taken in as "My pieces", sets made (names made unique), images
added (taken, skipped and why, missing), renamed, deleted, put back, and gone past their Undo.

    cd D:/chess-coaching-companion
    uv run --no-sync python android/tools/make_looks_fixtures.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[2]
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
        steps: list[dict] = []

        def step(op: str, result: object, **args: object) -> None:
            steps.append({"op": op, **args, "result": result})

        def sets() -> None:
            step("sets", looks.own_sets(data_dir))

        def catalogue() -> None:
            step("catalogue", looks.catalogue(data_dir))

        def look(pieces: str, theme: str = "green") -> None:
            step("look", looks.look(pieces, theme, data_dir), pieces=pieces, theme=theme)

        # The one own set there was before: its pieces beside a file that is none.
        old = {"wK.png": PNG, "bQ.svg": SVG, "notes.txt": b"not a piece"}
        old_dir = data_dir / "pieces" / "custom"
        old_dir.mkdir(parents=True)
        for name, data in old.items():
            (old_dir / name).write_bytes(data)
        steps.append({"op": "legacy", "files": [{"name": n, "data": b64(d)} for n, d in old.items()]})
        sets()
        look("custom")
        look("own:custom")
        catalogue()

        def create(name: str, files: list[dict[str, str]]) -> None:
            step("create", looks.create_own(data_dir, name, files), name=name, files=files)

        def add(set_id: str, files: list[dict[str, str]]) -> None:
            step("add", looks.add_images(data_dir, set_id, files), id=set_id, files=files)

        one = [{"name": "wK.png", "data": b64(PNG)}, {"name": "Chess_qdt45.svg", "data": b64(SVG)}]
        create("Neo", one)
        create("neo", [{"name": "bP.png", "data": b64(PNG)}])  # the same name: made unique
        create("", [{"name": "notes.txt", "data": b64(b"hi")}])  # nothing taken, nothing made
        create("", [{"name": "wN.webp", "data": b64(WEBP)}])  # no name: "My set"
        create("  Néo   ünïcode, a set with a name far longer than forty letters  ", one)
        add(
            "neo",
            [
                {"name": "notes.txt", "data": b64(b"hi")},
                {"name": "bP.png", "data": "not base64!!"},
                {"name": "bN.png", "data": b64(b"plain text")},
                {"name": "white-knight.webp", "data": b64(WEBP)},
                {"name": "wK.jpg", "data": b64(JPEG)},  # in place of the PNG
            ],
        )
        add("nope", one)
        sets()
        for set_id, name in [("neo", "  Neo   wood "), ("neo", "   "), ("nope", "x"), ("../x", "x")]:
            step("rename", looks.rename_own(data_dir, set_id, name), id=set_id, name=name)
        look("own:neo", "brown")
        look("own:../x")
        catalogue()
        for op, set_id in [("delete", "neo"), ("delete", "../x"), ("delete", "nope")]:
            step(op, looks.delete_own(data_dir, set_id), id=set_id)
        sets()
        look("own:neo")
        step("restore", looks.restore_own(data_dir, "neo"), id="neo")
        step("restore", looks.restore_own(data_dir, "neo"), id="neo")  # back already
        look("own:neo")
        # Past its Undo it is gone for good.
        step("delete", looks.delete_own(data_dir, "neo-2"), id="neo-2")
        aside = looks.own_root(data_dir) / looks.REMOVED / "neo-2"
        old_time = aside.stat().st_mtime - looks.UNDO_S - 5
        os.utime(aside, (old_time, old_time))
        steps.append({"op": "expire", "id": "neo-2"})
        sets()
        step("restore", looks.restore_own(data_dir, "neo-2"), id="neo-2")
        catalogue()
    fixture = {
        "catalogue": empty_catalogue,
        "looks": looks_,
        "codes": [[n, looks.piece_code(n)] for n in NAMES],
        "sniff": [[name, b64(data), looks.sniff(data)] for name, data in SAMPLES.items()],
        "steps": steps,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"{OUT}: {len(looks_)} looks, {len(NAMES)} names, {len(SAMPLES)} images, {len(steps)} steps")


if __name__ == "__main__":
    main()
