"""The theme cache: learn a board-and-piece style once, recognise it forever.

A model learned from a start position must round-trip through the cache and be found again by
a palette sampled cold from any board of the same theme -- the mechanism that lets a puzzle or
a mid-game position be read without bootstrapping from move 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from sidecoach.events import Side
from sidecoach.state.pipeline import GamePipeline
from sidecoach.vision.pieces import ThemeModel
from sidecoach.vision.theme_cache import ThemeCache, palette_key, sample_palette
from tests.synth import GEOM, synthetic_board


def test_palette_key_is_stable_and_discriminating() -> None:
    light = np.array([200.0, 210.0, 190.0])
    dark = np.array([90.0, 100.0, 80.0])
    assert palette_key(light, dark) == palette_key(light, dark)
    # A different board colour is a different theme.
    green = np.array([120.0, 180.0, 120.0])
    assert palette_key(light, dark) != palette_key(green, dark)


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
    assert loaded is not None and loaded.light_bg is not None
    assert [e.symbol for e in loaded.exemplars] == [e.symbol for e in model.exemplars]
    assert np.allclose(loaded.light_bg, model.light_bg)
    assert model.light_bg is not None and model.dark_bg is not None
    # A shade of rendering noise still matches (nearest palette within tolerance).
    assert cache.get(model.light_bg + 4, model.dark_bg - 3) is not None
    # An unseen palette is a miss.
    assert cache.get(np.array([10.0, 10.0, 10.0]), np.array([250.0, 250.0, 250.0])) is None


def test_a_theme_learned_before_the_last_version_is_still_read(tmp_path: Path) -> None:
    """Version 12 kept one landing crop a kind; 13 keeps one per square colour. A 12 file is
    read with its crop standing for both, where every set was ignored until a start position
    showed, and a puzzle never shows one (the phone reads its old files too; the parity
    audit, 2026-09-29). Anything older is learned again."""
    import pickle

    from sidecoach.vision.theme_cache import CACHE_VERSION, PREVIOUS_VERSION

    model = ThemeModel.from_start_position(synthetic_board(), GEOM, Side.WHITE)
    assert model is not None and model.light_bg is not None and model.dark_bg is not None
    cache = ThemeCache(tmp_path)
    path = tmp_path / "theme-old.pkl"
    kinds = {symbol: crop for (symbol, _light), crop in model.landing_crops.items()}
    model.landing_crops = kinds  # type: ignore[assignment] - the version 12 shape
    for version, read in ((PREVIOUS_VERSION, True), (PREVIOUS_VERSION - 1, False)):
        payload = {
            "version": version,
            "model": model,
            "light_bg": model.light_bg,
            "dark_bg": model.dark_bg,
        }
        path.write_bytes(pickle.dumps(payload))
        loaded = cache._read(path)
        assert (loaded is not None) is read, version
    assert CACHE_VERSION == PREVIOUS_VERSION + 1
    path.write_bytes(pickle.dumps({**payload, "version": PREVIOUS_VERSION}))
    loaded = cache._read(path)
    assert loaded is not None
    crops = loaded["model"].landing_crops
    assert set(crops) == {(symbol, light) for symbol in kinds for light in (True, False)}


def test_a_cached_theme_carries_no_frame(tmp_path: Path) -> None:
    """The theme file holds the exemplars, the palette and the digits, not the last
    frame's 64 cells: pickled with them it was 2.8 MB instead of 60 KB, and a theme read
    back would have started its next game believing it had read a frame."""
    cache = ThemeCache(tmp_path)
    warm = GamePipeline(cache=cache)
    for i in range(3):
        warm.process(synthetic_board(theme="green", coordinates=True), 0.1 * i)
    assert warm.model is not None and warm.model._cells, "the live model caches its cells"
    files = list(tmp_path.glob("theme-*.pkl"))
    # The embedding matcher's 64 exemplar vectors (float16) are 164 KB of the file.
    assert len(files) == 1 and files[0].stat().st_size < 400_000, files[0].stat().st_size
    light, dark = warm.model.light_bg, warm.model.dark_bg
    assert light is not None and dark is not None
    again = cache.get(light, dark)
    assert again is not None and again._cells == {}
