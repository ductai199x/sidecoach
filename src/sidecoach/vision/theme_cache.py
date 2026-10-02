"""Remember a learned board-and-piece style so recognition is paid once, not every game.

The recogniser (`ThemeModel`) is normally rebuilt from the start position each game: it learns
the twelve pieces and the two square colours from the known move-0 layout. That
makes analysing a puzzle or a mid-game position from scratch impossible -- there is no start
position to learn from.

This cache persists a learned model keyed by a fingerprint of the board's two square colours,
which can be sampled from *any* board (a running game, a puzzle) without a model, and by the
piece set: one board's colours can carry several sets (a set changed, another site with the
same colours, a test board), each learned once and kept beside the others; a set learned
again replaces itself. So a theme seen once is recognised forever: the models are loaded
from the cache and the one that reads the board is used to read the position cold. (Knowing
which side sits at the bottom -- the orientation -- is a separate problem the start position
also used to answer; the cache supplies the piece style, not the orientation.)

Serialisation is pickle: the cache is local, self-generated, and never shared, so the format
is ours to read and write. A cache written by an older version that no longer loads is simply
ignored and the theme relearned.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import pickle
from pathlib import Path

import numpy as np

from sidecoach.vision.pieces import ThemeModel
from sidecoach.vision.squares import CELL, BoardGeometry, is_light

log = logging.getLogger(__name__)

# 2: exemplars are tone masks; 3: tone is nearer the core than the square; 4: the set's
# colours are learned, for the drawing rule; 5, 6: the fill is the depth-weighted median.
# 10: the start cells' embeddings and landing crops travel with the theme; 11: the
# embeddings are of the crop at one resolution (pieces.canonical); 12: the exemplars keep
# only what the embedding reads (their tone and brightness); 13: a landing crop per square
# colour.
CACHE_VERSION = 13
# The version before is read too, its one landing crop a kind standing for both square colours
# until the set is learned again: a bump left every learned set unread until a start position
# showed, and a puzzle never shows one (Tai's phone, 0.1.19; the phone reads every version
# from 4, `ThemeCache.kt`; the parity audit, 2026-09-29). Older pickles hold models of an
# older shape, and are learned again.
PREVIOUS_VERSION = 12
_QUANTUM = 8  # square colours within this many levels share a fingerprint (tolerates lighting)
MATCH_TOLERANCE = 20.0  # palettes closer than this (in BGR) are treated as the same theme


def _distance(
    a_light: np.ndarray, a_dark: np.ndarray, b_light: np.ndarray, b_dark: np.ndarray
) -> float:
    a = np.concatenate([a_light, a_dark]).astype(float)
    b = np.concatenate([b_light, b_dark]).astype(float)
    return float(np.linalg.norm(a - b))


def same_theme(
    a_light: np.ndarray, a_dark: np.ndarray, b_light: np.ndarray, b_dark: np.ndarray
) -> bool:
    """Whether two palettes are one theme, by the cache's own tolerance."""
    return _distance(a_light, a_dark, b_light, b_dark) <= MATCH_TOLERANCE


def palette_key(light_bg: np.ndarray, dark_bg: np.ndarray) -> str:
    """A stable fingerprint for a board theme from its two square colours.

    Quantised so near-identical palettes (a shade of lighting apart) map together, and ordered
    light-then-dark so the same theme always hashes the same way.
    """
    quantised = np.concatenate([light_bg, dark_bg]).astype(float)
    quantised = np.round(quantised / _QUANTUM).astype(int)
    return hashlib.sha1(quantised.tobytes()).hexdigest()[:16]


def sample_palette(
    frame: np.ndarray, geometry: BoardGeometry
) -> tuple[np.ndarray, np.ndarray] | None:
    """The board's light and dark square colours, sampled from cell corners on any board.

    The centre of a cell may hold a piece; its corners are almost always the bare square, so a
    median over every cell's corners reads the palette whether the board is empty or mid-game.
    Returns None if the two colours cannot be told apart (a covered board, a bad lock).
    """
    lights: list[np.ndarray] = []
    darks: list[np.ndarray] = []
    inset = max(1, CELL // 6)
    for row in range(8):
        for col in range(8):
            cell = geometry.crop(frame, row, col)
            if cell.size == 0 or cell.shape[0] < 2 * inset or cell.shape[1] < 2 * inset:
                continue
            corners = np.stack(
                [
                    cell[:inset, :inset].reshape(-1, cell.shape[2]),
                    cell[:inset, -inset:].reshape(-1, cell.shape[2]),
                    cell[-inset:, :inset].reshape(-1, cell.shape[2]),
                    cell[-inset:, -inset:].reshape(-1, cell.shape[2]),
                ]
            ).reshape(-1, cell.shape[2])
            (lights if is_light(row, col) else darks).append(np.median(corners, axis=0))
    if not lights or not darks:
        return None
    light_bg = np.median(np.array(lights), axis=0)
    dark_bg = np.median(np.array(darks), axis=0)
    if float(np.linalg.norm(light_bg - dark_bg)) < _QUANTUM:
        return None
    return light_bg, dark_bg


def _set_key(model: ThemeModel) -> str:
    """A name for a learned set among those on the same board colours, from its start
    cells' embeddings."""
    assert model.exemplar_vectors is not None
    return hashlib.sha1(np.ascontiguousarray(model.exemplar_vectors).tobytes()).hexdigest()[:8]


class ThemeCache:
    """A directory of learned themes, one pickled `ThemeModel` per board-and-piece set."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        # Each file's palette, read once per file version (path, size, mtime): a board
        # waiting to be read cold asks twice a second, and a theme file is a few hundred
        # KB. Only the matching file is unpickled, and afresh each time, so every caller
        # gets a model of its own.
        self._palettes: dict[tuple[Path, int, int], tuple[np.ndarray, np.ndarray] | None] = {}

    def _path(self, key: str) -> Path:
        return self.directory / f"theme-{key}.pkl"

    def save(self, model: ThemeModel, key: str | None = None) -> str | None:
        """Store a learned model. `key` names the learned set it replaces -- the same pieces
        on the same board learned again (`GamePipeline._learned_set`); without one it is
        kept beside any other set on these board colours. Returns the key, or None when the
        model has no palette to key on."""
        if model.light_bg is None or model.dark_bg is None:
            return None
        if key is None:
            key = f"{palette_key(model.light_bg, model.dark_bg)}-{_set_key(model)}"
        payload = {
            "version": CACHE_VERSION,
            "model": model,
            "light_bg": model.light_bg,
            "dark_bg": model.dark_bg,
        }
        # Written beside and moved over the old file: written in place, a save cut short
        # (the app killed, a full disk) left half a file and lost the set it replaced
        # (round 2, finding 9). A save that fails keeps the old file and returns None.
        path = self._path(key)
        partial = path.with_name(path.name + ".part")
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            with partial.open("wb") as handle:
                pickle.dump(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(partial, path)
        except OSError as exc:
            log.warning("theme %s not saved: %s", key, exc)
            with contextlib.suppress(OSError):
                partial.unlink()
            return None
        return key

    @staticmethod
    def _read(path: Path) -> dict | None:
        """A cache payload, or None if it cannot be read: an older format, a truncated
        write, or damaged bytes, which raise whatever the unpickler meets (ValueError,
        UnicodeDecodeError, ...: round 2, finding 9)."""
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)  # noqa: S301 - our own local, self-written cache
        except Exception:  # noqa: BLE001 - any failure to read is a file not there
            return None
        if (
            isinstance(payload, dict)
            and payload.get("version") in (CACHE_VERSION, PREVIOUS_VERSION)
            and isinstance(payload.get("model"), ThemeModel)
            and payload.get("light_bg") is not None
            and payload.get("dark_bg") is not None
        ):
            if payload["version"] == PREVIOUS_VERSION:
                model = payload["model"]
                model.landing_crops = {
                    (symbol, light): crop
                    for symbol, crop in model.landing_crops.items()
                    for light in (True, False)
                }
            return payload
        return None

    def _palette(self, path: Path) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        key = (path, stat.st_size, stat.st_mtime_ns)
        if key not in self._palettes:
            payload = self._read(path)
            self._palettes[key] = (
                None if payload is None else (payload["light_bg"], payload["dark_bg"])
            )
        return self._palettes[key]

    def candidates(self, light_bg: np.ndarray, dark_bg: np.ndarray) -> list[tuple[str, ThemeModel]]:
        """Every learned set whose board colours are within tolerance of these, nearest
        first, with its key.

        A nearest-palette match rather than an exact key so a shade of rendering noise between
        sessions never causes a miss; a user has only a handful of themes, so the scan is cheap.
        Each model is read afresh, so every caller gets one of its own.
        """
        near: list[tuple[float, Path]] = []
        for path in sorted(self.directory.glob("theme-*.pkl")):
            palette = self._palette(path)
            if palette is None:
                continue
            distance = _distance(light_bg, dark_bg, *palette)
            if distance <= MATCH_TOLERANCE:
                near.append((distance, path))
        found: list[tuple[str, ThemeModel]] = []
        for _, path in sorted(near, key=lambda item: item[0]):
            payload = self._read(path)
            if payload is not None:
                found.append((path.stem.removeprefix("theme-"), payload["model"]))
        return found

    def get(self, light_bg: np.ndarray, dark_bg: np.ndarray) -> ThemeModel | None:
        """The learned model whose palette is nearest this one within tolerance, or None."""
        found = self.candidates(light_bg, dark_bg)
        return found[0][1] if found else None
