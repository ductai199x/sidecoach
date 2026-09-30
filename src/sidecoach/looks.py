"""How the boards look: the piece sets Sidecoach ships, the board colour themes, and a
player's own piece images.

The shipped sets are Sidecoach's own (the default, tools/pieces/README.md) and every Lichess
set whose licence allows any use (tools/pieces/fetch_sets.py): static files beside the page,
under /pieces/ and /pieces/<set>/. The themes are plain colours. A player's own pieces are
images they choose, kept only in their data folder (data/pieces/custom) and handed to the
page inline, so they need no route of their own (the API's token guards every /api route,
and a stylesheet's image cannot send it). Nothing names or fetches any other product's set.

The board's squares carry the panel's signature for the vision (vision/ownboard.py): its odd
rows three units bluer than its even rows, square colour and highlight alike. Every theme
here keeps it: each colour's odd shade is the colour plus SHADE_STEP in blue, so no colour's
blue may pass 255 - SHADE_STEP.
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SHADE_STEP = 3  # vision/ownboard.py's SHADE_STEP
CODES = tuple(c + p for c in "wb" for p in "KQRBNP")
DEFAULT_SET = "sidecoach"
CUSTOM = "custom"
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
    "sidecoach": PieceSet("Sidecoach", "Sidecoach", "Sidecoach's own", "", "own1"),
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


def custom_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "pieces" / CUSTOM


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


def custom_images(data_dir: Path) -> dict[str, str]:
    """The player's own pieces as data URLs, by code: the ones there are."""
    folder = custom_dir(data_dir)
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


def save_custom(data_dir: Path, files: list[dict[str, str]]) -> dict[str, Any]:
    """Images the player chose, each {"name", "data" (base64)}: each one a piece names is kept
    (in place of that piece's last one); what was not taken is said, and why."""
    folder = custom_dir(data_dir)
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
    have = set(custom_images(data_dir))
    return {
        "ok": bool(taken),
        "taken": taken,
        "skipped": skipped,
        "missing": [c for c in CODES if c not in have],
    }


def clear_custom(data_dir: Path) -> None:
    """The player's own pieces gone (the files one by one; the folder stays)."""
    folder = custom_dir(data_dir)
    if folder.is_dir():
        for path in folder.iterdir():
            if path.is_file():
                path.unlink()


# ---- what the pages read ----


def look(pieces: str, theme: str, data_dir: Path) -> dict[str, Any]:
    """The boards' look now: an image for every piece (a player's own, where they gave one,
    else Sidecoach's), and the theme's colours."""
    base = PIECE_SETS[DEFAULT_SET]
    images = {c: base.url(c) for c in CODES}
    if pieces == CUSTOM:
        images.update(custom_images(data_dir))
    elif pieces in PIECE_SETS:
        images = {c: PIECE_SETS[pieces].url(c) for c in CODES}
    board = BOARD_THEMES.get(theme, BOARD_THEMES[DEFAULT_THEME])
    return {"pieces": pieces, "theme": theme, "images": images, "board": board.colours()}


def catalogue(data_dir: Path) -> dict[str, Any]:
    """What the settings page offers: every set with its preview images, author and licence,
    every theme with its colours, and the player's own pieces as they are."""
    own = custom_images(data_dir)
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
        "custom": {"images": own, "missing": [c for c in CODES if c not in own]},
    }
