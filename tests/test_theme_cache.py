"""The theme cache: learn a board-and-piece style once, recognise it forever.

A model learned from a start position must round-trip through the cache and be found again by
a palette sampled cold from any board of the same theme -- the mechanism that lets a puzzle or
a mid-game position be read without bootstrapping from move 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sidecoach.events import Region, Side
from sidecoach.vision.pieces import ThemeModel
from sidecoach.vision.squares import BoardGeometry
from sidecoach.vision.theme_cache import ThemeCache, palette_key, sample_palette
from tests.synth import synthetic_board

GEOM = BoardGeometry(Region(300, 120, 512, 512), 64.0)


def test_palette_key_is_stable_and_discriminating() -> None:
    light = np.array([200.0, 210.0, 190.0])
    dark = np.array([90.0, 100.0, 80.0])
    assert palette_key(light, dark) == palette_key(light, dark)
    # A different board colour is a different theme.
    green = np.array([120.0, 180.0, 120.0])
    assert palette_key(light, dark) != palette_key(green, dark)


def test_sample_palette_keys_to_the_learned_model() -> None:
    frame = synthetic_board()
    model = ThemeModel.from_start_position(frame, GEOM, Side.WHITE)
    assert model is not None and model.light_bg is not None and model.dark_bg is not None
    sampled = sample_palette(frame, GEOM)
    assert sampled is not None
    # The palette sampled cold from the board keys to the same theme the model learned.
    assert palette_key(*sampled) == palette_key(model.light_bg, model.dark_bg)


def test_theme_cache_round_trips_a_learned_model(tmp_path: Path) -> None:
    frame = synthetic_board()
    model = ThemeModel.from_start_position(frame, GEOM, Side.WHITE)
    assert model is not None and model.light_bg is not None
    cache = ThemeCache(tmp_path)
    assert cache.save(model) is not None
    # Found again by a palette sampled from a board of the same theme -- no model needed to look up.
    sampled = sample_palette(frame, GEOM)
    assert sampled is not None
    loaded = cache.get(*sampled)
    assert loaded is not None
    assert loaded.symbols == model.symbols
    assert np.allclose(loaded.light_bg, model.light_bg)
    # A shade of rendering noise still matches (nearest palette within tolerance).
    assert cache.get(model.light_bg + 4, model.dark_bg - 3) is not None
    # An unseen palette is a miss.
    assert cache.get(np.array([10.0, 10.0, 10.0]), np.array([250.0, 250.0, 250.0])) is None
