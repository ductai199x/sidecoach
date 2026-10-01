"""How the boards look: the piece sets Sidecoach ships, the board colour themes, and a
player's own piece images.

The shipped sets are Sidecoach's own (the default, tools/pieces/README.md) and every Lichess
set whose licence allows any use (tools/pieces/fetch_sets.py): static files beside the page,
under /pieces/ and /pieces/<set>/. The themes are plain colours. A player's own sets are
images they choose, each set named and kept apart in their data folder (data/pieces/own/<id>/,
its name in set.json) and handed to the page inline, so they need no route of their own (the
API's token guards every /api route, and a stylesheet's image cannot send it). A set deleted
waits UNDO_S in pieces/own/.removed/ for its Undo. The one own set there was before
(data/pieces/custom) becomes the set "My pieces" (own:custom). Nothing names or fetches any
other product's set.

The board's squares carry the panel's signature for the vision (vision/ownboard.py): its odd
rows three units bluer than its even rows, square colour and highlight alike. Every theme
here keeps it: each colour's odd shade is the colour plus SHADE_STEP in blue, so no colour's
blue may pass 255 - SHADE_STEP.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SHADE_STEP = 3  # vision/ownboard.py's SHADE_STEP
CODES = tuple(c + p for c in "wb" for p in "KQRBNP")
DEFAULT_SET = "sidecoach"
CUSTOM = "custom"  # the one own set before there were several; now the own set of that id
OWN = "own:"  # an own set as the board setting names it: "own:<id>"
DEFAULT_THEME = "green"


@dataclass(frozen=True)
class PieceSet:
    label: str
    author: str
    licence: str
    folder: str  # under /pieces/: "" for Sidecoach's own
    version: str  # the ?v= tag: a new one when the set's files change

    def url(self, code: str) -> str:
        return f"/pieces/{self.folder}{code}.svg?v={self.version}"


PIECE_SETS: dict[str, PieceSet] = {
    "sidecoach": PieceSet("Sidecoach", "Sidecoach", "Sidecoach's own", "", "own2"),
    "cburnett": PieceSet("Classic", "Colin M.L. Burnett", "GPLv2+", "cburnett/", "1"),
    "merida": PieceSet("Merida", "Armando Hernandez Marroquin", "GPLv2+", "merida/", "1"),
    "chessnut": PieceSet("Chessnut", "Alexis Luengas", "Apache 2.0", "chessnut/", "1"),
    "celtic": PieceSet("Celtic", "Maurizio Monge", "MIT", "celtic/", "1"),
    "spatial": PieceSet("Spatial", "Maurizio Monge", "MIT", "spatial/", "1"),
    "fantasy": PieceSet("Fantasy", "Maurizio Monge", "MIT", "fantasy/", "1"),
    "papercut": PieceSet("Papercut", "Nikolay Anzarov", "CC BY 4.0", "papercut/", "1"),
    "mpchess": PieceSet("MPChess", "Maxime Chupin", "GPLv3+", "mpchess/", "1"),
    "pirouetti": PieceSet("Pirouetti", "pirouetti", "AGPLv3+", "pirouetti/", "1"),
    "rhosgfx": PieceSet("RhosGFX", "RhosGFX", "CC0 (public domain)", "rhosgfx/", "1"),
    "totoy": PieceSet("Totoy", "Kosal Sen", "CC BY 4.0", "totoy/", "1"),
    "kiwen-suwi": PieceSet("Kiwen-suwi", "neverRare", "CC BY 4.0", "kiwen-suwi/", "1"),
    "firi": PieceSet("Firi", "James Faure", "CC BY 4.0", "firi/", "1"),
    "pixel": PieceSet("Pixel", "therealqtpi", "AGPLv3+", "pixel/", "1"),
    "shapes": PieceSet("Shapes", "flugsio", "CC BY-SA 4.0", "shapes/", "1"),
    "letter": PieceSet("Letters", "usolando", "AGPLv3+", "letter/", "1"),
}


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _rgb(h: str) -> tuple[int, int, int]:
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


def _odd(h: str) -> str:
    r, g, b = _rgb(h)
    return _hex((r, g, b + SHADE_STEP))


def _highlight(h: str) -> str:
    """A square lit for the last move: the square half-way to a warm yellow, as a translucent
    yellow over it would look; the blue kept clear of the signature's headroom."""
    r, g, b = _rgb(h)
    y = (255, 255, 51)
    mixed = tuple(round(c + (t - c) * 0.5) for c, t in zip((r, g, b), y, strict=True))
    return _hex((mixed[0], mixed[1], min(mixed[2], 255 - SHADE_STEP)))


@dataclass(frozen=True)
class BoardTheme:
    label: str
    light: str
    dark: str
    light_hl: str = ""
    dark_hl: str = ""

    def colours(self) -> dict[str, str]:
        light_hl = self.light_hl or _highlight(self.light)
        dark_hl = self.dark_hl or _highlight(self.dark)
        return {
            "light": self.light,
            "light_odd": _odd(self.light),
            "dark": self.dark,
            "dark_odd": _odd(self.dark),
            "light_hl": light_hl,
            "light_hl_odd": _odd(light_hl),
            "dark_hl": dark_hl,
            "dark_hl_odd": _odd(dark_hl),
        }


# Plain colours: nothing here is anyone's artwork. Green is the panel's first board, its
# highlights as they were.
BOARD_THEMES: dict[str, BoardTheme] = {
    "green": BoardTheme("Green", "#eeeed2", "#769656", "#f6f669", "#baca2b"),
    "brown": BoardTheme("Brown", "#f0d9b5", "#b58863"),
    "blue": BoardTheme("Blue", "#dee3e6", "#8ca2ad"),
    "ocean": BoardTheme("Ocean", "#eae9d2", "#4b7399"),
    "grey": BoardTheme("Grey", "#e3e3e3", "#a1a1a1"),
    "purple": BoardTheme("Purple", "#eeeaf2", "#8877b7"),
    "olive": BoardTheme("Olive", "#e8e7c8", "#8a9153"),
    "sand": BoardTheme("Sand", "#f2e4c3", "#c9a26b"),
    "walnut": BoardTheme("Walnut", "#e3c49a", "#8b5a3c"),
    "cherry": BoardTheme("Cherry", "#f1d9c9", "#b56060"),
    "coral": BoardTheme("Coral", "#f4e3d7", "#e07a5f"),
    "ice": BoardTheme("Ice", "#eef3f7", "#a3bfd6"),
    "tournament": BoardTheme("Tournament", "#eeeee6", "#3d6b50"),
    "teal": BoardTheme("Teal", "#e0ece8", "#4f8a83"),
    "rose": BoardTheme("Rose", "#f3e3e6", "#c07a8a"),
    "night": BoardTheme("Night", "#9aa3ad", "#4a5460"),
}


# ---- a player's own pieces ----

MAX_BYTES = 2_000_000  # an image
_NAMES = {"k": "K", "q": "Q", "r": "R", "b": "B", "n": "N", "p": "P"}
_WORDS = {"king": "K", "queen": "Q", "rook": "R", "bishop": "B", "knight": "N", "pawn": "P"}
_EXT = {"image/png": ".png", "image/svg+xml": ".svg", "image/webp": ".webp", "image/jpeg": ".jpg"}


def piece_code(filename: str) -> str | None:
    """Which piece a file is, from its name: "wK.png", "bq.svg", "w_n.png", "white-knight.png",
    "knight_black.svg", "Chess_nlt45.svg" (the Wikimedia naming: l is white, d is black)."""
    stem = Path(filename).stem.lower()
    m = re.fullmatch(r"([wb])[-_ ]?([kqrbnp])", stem)
    if m:
        return m.group(1) + _NAMES[m.group(2)]
    m = re.search(r"chess_([kqrbnp])([ld])t\d*", stem)
    if m:
        return ("w" if m.group(2) == "l" else "b") + _NAMES[m.group(1)]
    colour = "w" if "white" in stem else "b" if "black" in stem else None
    kinds = [v for w, v in _WORDS.items() if w in stem]
    if colour and len(kinds) == 1:
        return colour + kinds[0]
    return None


def sniff(data: bytes) -> str | None:
    """The image type from its first bytes: PNG, JPEG, WebP or SVG; else None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    head = data[:2048].lstrip().lower()
    if head.startswith((b"<svg", b"<?xml", b"<!--")) and b"<svg" in data[:4096].lower():
        return "image/svg+xml"
    return None


LEGACY_NAME = "My pieces"
DEFAULT_NAME = "My set"
REMOVED = ".removed"
UNDO_S = 600  # a deleted set can come back this long
MAX_NAME = 40
_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,39}")


def own_root(data_dir: Path) -> Path:
    return Path(data_dir) / "pieces" / "own"


def own_id(pieces: str) -> str | None:
    """The own set a board setting names ("own:<id>", or "custom", the one own set before there
    were several), or None for a shipped set."""
    if pieces == CUSTOM:
        return CUSTOM
    if pieces.startswith(OWN) and _ID.fullmatch(pieces[len(OWN) :]):
        return pieces[len(OWN) :]
    return None


def _meta(folder: Path) -> dict[str, Any]:
    try:
        meta = json.loads((folder / "set.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _write_meta(folder: Path, name: str, order: int) -> None:
    (folder / "set.json").write_text(json.dumps({"name": name, "order": order}), encoding="utf-8")


def _clean_name(name: object) -> str:
    return " ".join(str(name or "").split())[:MAX_NAME].strip()


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:32].strip("-")
    return slug or "set"


def _images(folder: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not folder.is_dir():
        return out
    for path in sorted(folder.iterdir()):
        code = path.stem
        if code not in CODES or not path.is_file():
            continue
        data = path.read_bytes()
        kind = sniff(data)
        if kind:
            out[code] = f"data:{kind};base64,{base64.b64encode(data).decode('ascii')}"
    return out


def _migrate(data_dir: Path) -> None:
    """The one own set there was before (pieces/custom) as the set "My pieces": copied, the copy
    read back, then the old files removed; nothing is lost if a copy is not the same."""
    old = Path(data_dir) / "pieces" / CUSTOM
    if not old.is_dir():
        return
    files = [f for f in old.iterdir() if f.is_file() and f.stem in CODES]
    new = own_root(data_dir) / CUSTOM
    if not files or new.exists():
        return
    new.mkdir(parents=True)
    for f in files:
        shutil.copy2(f, new / f.name)
    if all((new / f.name).read_bytes() == f.read_bytes() for f in files):
        _write_meta(new, LEGACY_NAME, 0)
        for f in files:
            f.unlink()
        if not any(old.iterdir()):
            old.rmdir()  # nothing else was in it
    else:
        shutil.rmtree(new)


def _purge(data_dir: Path) -> None:
    """Sets deleted longer ago than their Undo, gone for good."""
    bin_ = own_root(data_dir) / REMOVED
    if not bin_.is_dir():
        return
    now = time.time()
    for d in bin_.iterdir():
        if d.is_dir() and now - d.stat().st_mtime > UNDO_S:
            shutil.rmtree(d, ignore_errors=True)


def own_sets(data_dir: Path) -> list[dict[str, Any]]:
    """The player's own sets, in the order they were added: {"id", "name", "order"}."""
    _migrate(data_dir)
    _purge(data_dir)
    root = own_root(data_dir)
    if not root.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for d in root.iterdir():
        if d.is_dir() and _ID.fullmatch(d.name):
            meta = _meta(d)
            name = _clean_name(meta.get("name")) or d.name
            out.append({"id": d.name, "name": name, "order": int(meta.get("order") or 0)})
    out.sort(key=lambda o: (o["order"], o["name"].lower(), o["id"]))
    return out


def own_images(data_dir: Path, set_id: str) -> dict[str, str]:
    """An own set's pieces as data URLs, by code: the ones it has."""
    _migrate(data_dir)
    if not _ID.fullmatch(set_id):
        return {}
    return _images(own_root(data_dir) / set_id)


def _save(folder: Path, files: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]]]:
    """Each image a piece names kept in the set's folder (in place of that piece's last one);
    what was not taken, and why."""
    taken: list[str] = []
    skipped: list[dict[str, str]] = []
    for f in files:
        name = str(f.get("name", ""))
        code = piece_code(name)
        if code is None:
            skipped.append({"name": name, "why": "the name does not say which piece it is"})
            continue
        try:
            data = base64.b64decode(str(f.get("data", "")), validate=True)
        except ValueError:
            skipped.append({"name": name, "why": "it could not be read"})
            continue
        if len(data) > MAX_BYTES:
            skipped.append({"name": name, "why": "it is larger than 2 MB"})
            continue
        kind = sniff(data)
        if kind is None:
            skipped.append({"name": name, "why": "it is not a PNG, JPEG, WebP or SVG image"})
            continue
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob(f"{code}.*"):
            old.unlink()
        (folder / f"{code}{_EXT[kind]}").write_bytes(data)
        taken.append(code)
    return taken, skipped


def _result(
    data_dir: Path, set_id: str, taken: list[str], skipped: list[dict[str, str]]
) -> dict[str, Any]:
    have = own_images(data_dir, set_id)
    return {
        "ok": bool(taken),
        "id": OWN + set_id,
        "name": _clean_name(_meta(own_root(data_dir) / set_id).get("name")) or set_id,
        "taken": taken,
        "skipped": skipped,
        "missing": [c for c in CODES if c not in have],
    }


def _missing(set_id: str) -> dict[str, Any]:
    return {"ok": False, "id": OWN + set_id, "error": "There is no such set."}


def create_own(data_dir: Path, name: str, files: list[dict[str, str]]) -> dict[str, Any]:
    """A new own set from the images chosen, named (made unique: "My set 2"); nothing is made
    when no image was taken."""
    sets = own_sets(data_dir)
    names = {o["name"].lower() for o in sets}
    base = _clean_name(name) or DEFAULT_NAME
    shown, n = base, 2
    while shown.lower() in names:
        shown, n = f"{base[: MAX_NAME - 4]} {n}", n + 1
    root = own_root(data_dir)
    set_id, k = _slug(shown), 2
    while (root / set_id).exists() or (root / REMOVED / set_id).exists():
        set_id, k = f"{_slug(shown)}-{k}", k + 1
    folder = root / set_id
    taken, skipped = _save(folder, files)
    if not taken:
        shutil.rmtree(folder, ignore_errors=True)
        return {"ok": False, "taken": [], "skipped": skipped, "missing": list(CODES)}
    _write_meta(folder, shown, max((o["order"] for o in sets), default=0) + 1)
    return _result(data_dir, set_id, taken, skipped)


def add_images(data_dir: Path, set_id: str, files: list[dict[str, str]]) -> dict[str, Any]:
    """Images added to an own set, each in place of its piece's last one."""
    folder = own_root(data_dir) / set_id
    if not _ID.fullmatch(set_id) or not folder.is_dir():
        return _missing(set_id)
    taken, skipped = _save(folder, files)
    return _result(data_dir, set_id, taken, skipped)


def rename_own(data_dir: Path, set_id: str, name: str) -> dict[str, Any]:
    folder = own_root(data_dir) / set_id
    if not _ID.fullmatch(set_id) or not folder.is_dir():
        return _missing(set_id)
    shown = _clean_name(name)
    if not shown:
        return {"ok": False, "id": OWN + set_id, "error": "A set needs a name."}
    _write_meta(folder, shown, int(_meta(folder).get("order") or 0))
    return {"ok": True, "id": OWN + set_id, "name": shown}


def delete_own(data_dir: Path, set_id: str) -> dict[str, Any]:
    """An own set put aside for its Undo (UNDO_S), then gone."""
    root = own_root(data_dir)
    folder = root / set_id
    if not _ID.fullmatch(set_id) or not folder.is_dir():
        return _missing(set_id)
    aside = root / REMOVED / set_id
    aside.parent.mkdir(parents=True, exist_ok=True)
    if aside.exists():
        shutil.rmtree(aside)
    folder.rename(aside)
    os.utime(aside)
    return {"ok": True, "id": OWN + set_id}


def restore_own(data_dir: Path, set_id: str) -> dict[str, Any]:
    """A set deleted within its Undo, back as it was."""
    root = own_root(data_dir)
    aside = root / REMOVED / set_id
    if not _ID.fullmatch(set_id) or not aside.is_dir() or (root / set_id).exists():
        return {
            "ok": False,
            "id": OWN + set_id,
            "error": "That set was deleted too long ago to put back.",
        }
    aside.rename(root / set_id)
    return {"ok": True, "id": OWN + set_id}


# ---- what the pages read ----


def look(pieces: str, theme: str, data_dir: Path) -> dict[str, Any]:
    """The boards' look now: an image for every piece (a player's own, where they gave one,
    else Sidecoach's), and the theme's colours."""
    base = PIECE_SETS[DEFAULT_SET]
    images = {c: base.url(c) for c in CODES}
    own = own_id(pieces)
    if own is not None:
        images.update(own_images(data_dir, own))
    elif pieces in PIECE_SETS:
        images = {c: PIECE_SETS[pieces].url(c) for c in CODES}
    board = BOARD_THEMES.get(theme, BOARD_THEMES[DEFAULT_THEME])
    colours = board.colours()
    return {
        "pieces": pieces,
        "theme": theme,
        "images": images,
        "board": colours,
        "stamp": stamp(pieces, theme, images, colours),
    }


def stamp(pieces: str, theme: str, images: dict[str, str], colours: dict[str, str]) -> str:
    """A short hash of the look: it changes with the set, the colours or any image of an own
    set. Every window reads it every few seconds and reads the look again when it changes, so a
    window follows the settings whatever changed them -- not only a Settings page that tells
    the others (2026-10-01: the Neo set loaded by another way left every window as it was)."""
    text = "\n".join(
        [
            pieces,
            theme,
            *(f"{c}={images.get(c, '')}" for c in CODES),
            *(f"{k}={colours[k]}" for k in sorted(colours)),
        ]
    )
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def catalogue(data_dir: Path) -> dict[str, Any]:
    """What the settings page offers: every set with its preview images, author and licence,
    every theme with its colours, and the player's own sets, each with its pieces and what it
    lacks (Sidecoach's stand in)."""
    own = []
    for o in own_sets(data_dir):
        images = own_images(data_dir, o["id"])
        own.append(
            {
                "name": OWN + o["id"],
                "label": o["name"],
                "images": images,
                "missing": [c for c in CODES if c not in images],
            }
        )
    return {
        "sets": [
            {
                "name": n,
                "label": s.label,
                "author": s.author,
                "licence": s.licence,
                "images": {c: s.url(c) for c in CODES},
            }
            for n, s in PIECE_SETS.items()
        ],
        "themes": [{"name": n, "label": t.label, **t.colours()} for n, t in BOARD_THEMES.items()],
        "own": own,
    }
