"""How the boards look (looks.py): the shipped piece sets are all there, every colour theme
keeps the panel's signature for the vision, a player's own pieces are read from their names
and kept, and the pages get the look over the API."""

from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sidecoach import looks
from sidecoach.config import BoardSettings
from sidecoach.events import Region
from sidecoach.ui.panel import PanelEngine
from sidecoach.ui.server import build_app
from sidecoach.vision.ownboard import SHADE_STEP, is_own_board

STATIC = Path(__file__).parents[1] / "desktop" / "static" / "pieces"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 40
SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"/>'


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_every_shipped_set_has_its_twelve_pieces_and_a_licence_line() -> None:
    licence = (STATIC / "LICENSE.txt").read_text(encoding="utf-8")
    for name, s in looks.PIECE_SETS.items():
        for code in looks.CODES:
            assert (STATIC / f"{s.folder}{code}.svg").is_file(), f"{name}: {code} missing"
        if name != looks.DEFAULT_SET:
            assert name in licence, f"{name} has no line in pieces/LICENSE.txt"


def test_no_set_with_a_non_commercial_or_unclear_licence_ships() -> None:
    for s in looks.PIECE_SETS.values():
        assert "NC" not in s.licence and "non-commercial" not in s.licence.lower()
    assert not {"staunty", "tatiana", "maestro", "alpha", "horsey", "companion"} & set(
        looks.PIECE_SETS
    )


def _panel_board(colours: dict[str, str], cell: int = 40) -> np.ndarray:
    """The panel's board as the stylesheet paints it with a theme: odd rows in the odd shades."""

    def bgr(h: str) -> tuple[int, int, int]:
        return int(h[5:7], 16), int(h[3:5], 16), int(h[1:3], 16)

    img = np.full((8 * cell + 40, 8 * cell + 40, 3), 40, dtype=np.uint8)
    for r in range(8):
        for c in range(8):
            light = (r + c) % 2 == 0
            key = ("light" if light else "dark") + ("_odd" if r % 2 else "")
            img[20 + r * cell : 20 + (r + 1) * cell, 20 + c * cell : 20 + (c + 1) * cell] = bgr(
                colours[key]
            )
    return img


@pytest.mark.parametrize("name", list(looks.BOARD_THEMES))
def test_every_theme_keeps_the_signature_the_vision_knows_the_panel_by(name: str) -> None:
    colours = looks.BOARD_THEMES[name].colours()
    assert looks.SHADE_STEP == SHADE_STEP
    for base in ("light", "dark", "light_hl", "dark_hl"):
        b, odd = colours[base], colours[base + "_odd"]
        assert b[1:5] == odd[1:5], f"{name} {base}: only the blue may differ"
        assert int(odd[5:7], 16) - int(b[5:7], 16) == SHADE_STEP, f"{name} {base}"
    assert is_own_board(_panel_board(colours), Region(20, 20, 320, 320), 40)


def test_the_green_theme_is_the_board_the_panel_had() -> None:
    green = looks.BOARD_THEMES["green"].colours()
    assert green == {
        "light": "#eeeed2", "light_odd": "#eeeed5", "dark": "#769656", "dark_odd": "#769659",
        "light_hl": "#f6f669", "light_hl_odd": "#f6f66c",
        "dark_hl": "#baca2b", "dark_hl_odd": "#baca2e",
    }  # fmt: skip


@pytest.mark.parametrize(
    ("name", "code"),
    [
        ("wK.svg", "wK"), ("bq.png", "bQ"), ("wn.png", "wN"), ("b_p.webp", "bP"),
        ("white-knight.png", "wN"), ("knight_black.svg", "bN"), ("Chess_klt45.svg", "wK"),
        ("Chess_pdt45.svg", "bP"), ("king.png", None), ("white.png", None), ("logo.png", None),
    ],
)  # fmt: skip
def test_a_file_says_which_piece_it_is_by_its_name(name: str, code: str | None) -> None:
    assert looks.piece_code(name) == code


def test_own_pieces_are_kept_what_is_not_taken_is_said_and_the_rest_fall_back(
    tmp_path: Path,
) -> None:
    files = [{"name": f"{c[0]}{c[1].lower()}.png", "data": b64(PNG)} for c in looks.CODES[:11]]
    files += [
        {"name": "notes.png", "data": b64(PNG)},
        {"name": "bP.png", "data": b64(b"not an image")},
        {"name": "wQ.svg", "data": b64(b"x" * (looks.MAX_BYTES + 1))},
    ]
    res = looks.save_custom(tmp_path, files)
    assert res["ok"] and sorted(res["taken"]) == sorted(looks.CODES[:11])
    assert res["missing"] == ["bP"]
    assert {s["name"] for s in res["skipped"]} == {"notes.png", "bP.png", "wQ.svg"}
    look = looks.look(looks.CUSTOM, "brown", tmp_path)
    assert look["images"]["wK"].startswith("data:image/png;base64,")
    assert look["images"]["bP"] == looks.PIECE_SETS["sidecoach"].url("bP")  # ours in its place
    # One image replaces its piece's last one, of whatever type.
    looks.save_custom(tmp_path, [{"name": "wK.svg", "data": b64(SVG)}])
    assert looks.custom_images(tmp_path)["wK"].startswith("data:image/svg+xml;base64,")
    assert len(list(looks.custom_dir(tmp_path).glob("wK.*"))) == 1
    looks.clear_custom(tmp_path)
    assert looks.custom_images(tmp_path) == {}


def test_the_settings_take_only_known_sets_and_themes() -> None:
    assert BoardSettings(pieces="custom", theme="night").pieces == "custom"
    with pytest.raises(ValidationError):
        BoardSettings(pieces="neo")
    with pytest.raises(ValidationError):
        BoardSettings(theme="plaid")


def test_the_pages_read_the_look_and_a_player_loads_their_own(
    make_panel: Callable[..., PanelEngine],
) -> None:
    engine = make_panel()
    client = TestClient(build_app(engine))
    first = client.get("/api/look").json()
    assert first["pieces"] == "sidecoach" and first["images"]["wK"] == "/pieces/wK.svg?v=own2"
    assert first["board"]["dark"] == "#769656"
    assert client.get("/pieces/cburnett/wK.svg").status_code == 200
    sent = client.post(
        "/api/pieces/custom",
        json={"files": [{"name": c + ".png", "data": b64(PNG)} for c in looks.CODES]},
    ).json()
    assert sent["ok"] and sent["missing"] == []
    catalogue = client.get("/api/settings").json()["looks"]
    assert len(catalogue["custom"]["images"]) == 12 and catalogue["sets"][0]["name"] == "sidecoach"
    saved = client.put(
        "/api/settings", json={"settings": {"board": {"pieces": "custom", "theme": "blue"}}}
    ).json()
    assert saved["ok"], saved
    now = client.get("/api/look").json()
    assert (
        now["images"]["bQ"].startswith("data:image/png;base64,")
        and now["board"]["light"] == "#dee3e6"
    )
    assert client.delete("/api/pieces/custom").json()["ok"]
    assert client.get("/api/look").json()["images"]["bQ"] == "/pieces/bQ.svg?v=own2"
