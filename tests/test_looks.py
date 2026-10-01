"""How the boards look (looks.py): the shipped piece sets are all there, every colour theme
keeps the panel's signature for the vision, a player's own pieces are read from their names
and kept, and the pages get the look over the API."""

from __future__ import annotations

import base64
import os
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


def test_an_own_set_keeps_what_it_takes_says_what_it_does_not_and_the_rest_fall_back(
    tmp_path: Path,
) -> None:
    files = [{"name": f"{c[0]}{c[1].lower()}.png", "data": b64(PNG)} for c in looks.CODES[:11]]
    files += [
        {"name": "notes.png", "data": b64(PNG)},
        {"name": "bP.png", "data": b64(b"not an image")},
        {"name": "wQ.svg", "data": b64(b"x" * (looks.MAX_BYTES + 1))},
    ]
    res = looks.create_own(tmp_path, "Neo", files)
    assert res["ok"] and res["id"] == "own:neo" and res["name"] == "Neo"
    assert sorted(res["taken"]) == sorted(looks.CODES[:11]) and res["missing"] == ["bP"]
    assert {s["name"] for s in res["skipped"]} == {"notes.png", "bP.png", "wQ.svg"}
    look = looks.look("own:neo", "brown", tmp_path)
    assert look["images"]["wK"].startswith("data:image/png;base64,")
    assert look["images"]["bP"] == looks.PIECE_SETS["sidecoach"].url("bP")  # ours in its place
    # More images for it: each in place of its piece's last one, of whatever type.
    more = looks.add_images(tmp_path, "neo", [{"name": "wK.svg", "data": b64(SVG)}])
    assert more["ok"] and more["name"] == "Neo"
    assert looks.own_images(tmp_path, "neo")["wK"].startswith("data:image/svg+xml;base64,")
    assert len(list((looks.own_root(tmp_path) / "neo").glob("wK.*"))) == 1
    # Nothing taken, nothing made.
    none = looks.create_own(tmp_path, "Junk", [{"name": "notes.png", "data": b64(PNG)}])
    assert not none["ok"] and [o["name"] for o in looks.own_sets(tmp_path)] == ["Neo"]


def test_own_sets_are_several_named_renamed_deleted_and_put_back(tmp_path: Path) -> None:
    one = [{"name": "wK.png", "data": b64(PNG)}]
    a = looks.create_own(tmp_path, "Neo", one)
    b = looks.create_own(tmp_path, "neo", one)  # the same name again: made unique
    c = looks.create_own(tmp_path, "", one)  # no name: "My set"
    assert (a["id"], b["id"], c["id"]) == ("own:neo", "own:neo-2", "own:my-set")
    assert [o["name"] for o in looks.own_sets(tmp_path)] == ["Neo", "neo 2", "My set"]
    shelf = looks.catalogue(tmp_path)["own"]
    assert [o["label"] for o in shelf] == ["Neo", "neo 2", "My set"]
    assert shelf[0]["name"] == "own:neo" and len(shelf[0]["missing"]) == 11
    assert looks.rename_own(tmp_path, "neo", "  Neo   wood ")["name"] == "Neo wood"
    assert not looks.rename_own(tmp_path, "neo", "   ")["ok"]
    assert looks.own_sets(tmp_path)[0]["name"] == "Neo wood"
    # Deleted: out of the list, the board falls back to Sidecoach's, and the Undo brings it back.
    assert looks.delete_own(tmp_path, "neo")["ok"]
    assert "Neo wood" not in [o["name"] for o in looks.own_sets(tmp_path)]
    assert looks.look("own:neo", "green", tmp_path)["images"]["wK"] == looks.PIECE_SETS[
        "sidecoach"
    ].url("wK")
    assert looks.restore_own(tmp_path, "neo")["ok"]
    assert looks.own_sets(tmp_path)[0]["name"] == "Neo wood"
    assert looks.look("own:neo", "green", tmp_path)["images"]["wK"].startswith("data:image/png")
    # Past its Undo it is gone for good.
    assert looks.delete_own(tmp_path, "neo")["ok"]
    aside = looks.own_root(tmp_path) / looks.REMOVED / "neo"
    old = aside.stat().st_mtime - looks.UNDO_S - 5
    os.utime(aside, (old, old))
    looks.own_sets(tmp_path)
    assert not aside.exists() and not looks.restore_own(tmp_path, "neo")["ok"]
    # What is not a set is refused, never a path outside the data folder.
    assert (
        not looks.delete_own(tmp_path, "../x")["ok"]
        and not looks.rename_own(tmp_path, "nope", "x")["ok"]
    )


def test_the_one_own_set_from_before_becomes_my_pieces(tmp_path: Path) -> None:
    old = tmp_path / "pieces" / "custom"
    old.mkdir(parents=True)
    for c in looks.CODES:
        (old / f"{c}.png").write_bytes(PNG)
    (old / "notes.txt").write_text("not a piece")
    shelf = looks.own_sets(tmp_path)
    assert shelf == [{"id": "custom", "name": "My pieces", "order": 0}]
    assert len(looks.own_images(tmp_path, "custom")) == 12
    assert not list(old.glob("*.png")) and (old / "notes.txt").exists()  # only what moved goes
    # The board setting from before ("custom") still names it.
    assert looks.look("custom", "green", tmp_path)["images"]["bQ"].startswith("data:image/png")
    assert looks.look("own:custom", "green", tmp_path)["images"]["bQ"].startswith("data:image/png")
    # With nothing else in it, the old folder goes too.
    bare = tmp_path / "bare"
    (bare / "pieces" / "custom").mkdir(parents=True)
    (bare / "pieces" / "custom" / "wK.png").write_bytes(PNG)
    assert [o["name"] for o in looks.own_sets(bare)] == ["My pieces"]
    assert not (bare / "pieces" / "custom").exists()


def test_the_settings_take_only_known_sets_and_themes() -> None:
    assert BoardSettings(pieces="custom", theme="night").pieces == "custom"
    assert BoardSettings(pieces="own:neo-2").pieces == "own:neo-2"
    with pytest.raises(ValidationError):
        BoardSettings(pieces="own:../x")
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
        "/api/pieces/own",
        json={
            "name": "Neo",
            "files": [{"name": c + ".png", "data": b64(PNG)} for c in looks.CODES],
        },
    ).json()
    assert sent["ok"] and sent["id"] == "own:neo" and sent["missing"] == []
    catalogue = client.get("/api/settings").json()["looks"]
    assert [o["label"] for o in catalogue["own"]] == ["Neo"] and catalogue["sets"][0][
        "name"
    ] == "sidecoach"
    # A save of the board alone keeps the rest of the settings (the page sends what changed).
    saved = client.put(
        "/api/settings", json={"settings": {"board": {"pieces": "own:neo", "theme": "blue"}}}
    ).json()
    assert saved["ok"], saved
    now = client.get("/api/look").json()
    assert (
        now["images"]["bQ"].startswith("data:image/png;base64,")
        and now["board"]["light"] == "#dee3e6"
    )
    renamed = client.put("/api/pieces/own/neo", json={"name": "Neo wood"}).json()
    assert renamed["ok"] and renamed["name"] == "Neo wood"
    added = client.post(
        "/api/pieces/own/neo/images", json={"files": [{"name": "wK.svg", "data": b64(SVG)}]}
    )
    assert added.json()["taken"] == ["wK"]
    assert client.delete("/api/pieces/own/neo").json()["ok"]
    assert client.get("/api/look").json()["images"]["bQ"] == "/pieces/bQ.svg?v=own2"
    assert client.post("/api/pieces/own/neo/restore").json()["ok"]
    assert client.get("/api/look").json()["images"]["bQ"].startswith("data:image/png")
    assert client.delete("/api/pieces/own/neo").json()["ok"]
    assert client.get("/api/look").json()["images"]["bQ"] == "/pieces/bQ.svg?v=own2"
