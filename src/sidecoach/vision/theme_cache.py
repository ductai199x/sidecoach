"""Remember a learned board-and-piece style so recognition is paid once, not every game.

The recogniser (`ThemeModel`) is normally rebuilt from the start position each game: it learns
the twelve piece silhouettes and the two square colours from the known move-0 layout. That
makes analysing a puzzle or a mid-game position from scratch impossible -- there is no start
position to learn from.

This cache persists a learned model keyed by a fingerprint of the board's two square colours,
which can be sampled from *any* board (a running game, a puzzle) without a model. So a theme
seen once is recognised forever: the model is loaded from the cache and the position is read
cold. (Knowing which side sits at the bottom -- the orientation -- is a separate problem the
start position also used to answer; the cache supplies the piece style, not the orientation.)

Serialisation is pickle: the cache is local, self-generated, and never shared, so the format
is ours to read and write. A cache written by an older version that no longer loads is simply
ignored and the theme relearned.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np

from sidecoach.vision.pieces import ThemeModel
from sidecoach.vision.squares import CELL, BoardGeometry, is_light

# 2: exemplars are tone masks; 3: tone is nearer the core than the square; 4: the set's
# colours are learned, for the drawing rule; 5, 6: the fill is the depth-weighted median.
CACHE_VERSION = 9
_QUANTUM = 8  # square colours within this many levels share a fingerprint (tolerates lighting)
MATCH_TOLERANCE = 20.0  # palettes closer than this (in BGR) are treated as the same theme


def _distance(
    a_light: np.ndarray, a_dark: np.ndarray, b_light: np.ndarray, b_dark: np.ndarray
) -> float:
    a = np.concatenate([a_light, a_dark]).astype(float)
    b = np.concatenate([b_light, b_dark]).astype(float)
    return float(np.linalg.norm(a - b))


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


class ThemeCache:
    """A directory of learned themes, one pickled `ThemeModel` per palette fingerprint."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def _path(self, key: str) -> Path:
        return self.directory / f"theme-{key}.pkl"

    def save(self, model: ThemeModel) -> str | None:
        """Store a learned model under its own palette fingerprint. Returns the key, or None
        when the model has no palette to key on."""
        if model.light_bg is None or model.dark_bg is None:
            return None
        key = palette_key(model.light_bg, model.dark_bg)
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "model": model,
            "light_bg": model.light_bg,
            "dark_bg": model.dark_bg,
        }
        with self._path(key).open("wb") as handle:
            pickle.dump(payload, handle)
        return key

    @staticmethod
    def _read(path: Path) -> dict | None:
        """A cache payload, or None if it cannot be read (an older format, a truncated write)."""
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)  # noqa: S301 - our own local, self-written cache
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError, ImportError):
            return None
        if (
            isinstance(payload, dict)
            and payload.get("version") == CACHE_VERSION
            and isinstance(payload.get("model"), ThemeModel)
            and payload.get("light_bg") is not None
            and payload.get("dark_bg") is not None
        ):
            return payload
        return None

    def get(self, light_bg: np.ndarray, dark_bg: np.ndarray) -> ThemeModel | None:
        """The learned model whose palette is nearest this one within tolerance, or None.

        A nearest-palette match rather than an exact key so a shade of rendering noise between
        sessions never causes a miss; a user has only a handful of themes, so the scan is cheap.
        """
        best: ThemeModel | None = None
        best_distance = MATCH_TOLERANCE
        for path in sorted(self.directory.glob("theme-*.pkl")):
            payload = self._read(path)
            if payload is None:
                continue
            distance = _distance(light_bg, dark_bg, payload["light_bg"], payload["dark_bg"])
            if distance <= best_distance:
                best_distance = distance
                best = payload["model"]
        return best
