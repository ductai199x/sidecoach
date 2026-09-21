"""The first-run setup's choice of Stockfish build, per platform (Phase 2: the Linux names
changed in Stockfish 19, and the finder must see what the tarball holds)."""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

import pytest

from sidecoach.config import EngineSettings, Settings, find_stockfish
from sidecoach.setup import stockfish_assets, unpack_stockfish


def test_the_stockfish_asset_follows_the_platform() -> None:
    assert stockfish_assets("Windows", "AMD64")[0] == "stockfish-windows-x86-64-universal.zip"
    assert stockfish_assets("Linux", "x86_64")[0] == "stockfish-linux-x86-64-universal.tar.gz"
    assert stockfish_assets("Linux", "aarch64")[0] == "stockfish-linux-arm64-universal.tar.gz"
    assert stockfish_assets("Darwin", "arm64")[0] == "stockfish-macos-universal.tar.gz"
    # Older releases named the Linux builds by ISA; they stay as fallbacks.
    assert "stockfish-ubuntu-x86-64-avx2.tar" in stockfish_assets("Linux", "x86_64")


@pytest.mark.skipif(os.name == "nt", reason="executable bits are a POSIX thing")
def test_a_linux_tarball_is_unpacked_and_found(tmp_path: Path) -> None:
    """What the app downloads on Linux: stockfish/stockfish-linux-x86-64-universal (and the
    notes beside it), no extension. It comes out executable and the finder picks it."""
    tools = tmp_path / "tools"
    tools.mkdir()
    inner = tmp_path / "stockfish"
    inner.mkdir()
    (inner / "stockfish-linux-x86-64-universal").write_bytes(b"#!/bin/sh\necho uci\n")
    (inner / "AUTHORS").write_text("them", encoding="utf-8")
    archive = tools / "stockfish-linux-x86-64-universal.tar.gz"
    with tarfile.open(archive, "w:gz") as t:
        t.add(inner, arcname="stockfish")
    found = unpack_stockfish(archive, tools)
    assert [f.name for f in found] == ["stockfish-linux-x86-64-universal"]
    assert os.access(found[0], os.X_OK)
    settings = Settings(data_dir=tmp_path, tools_dir=tools, engine=EngineSettings())
    assert find_stockfish(settings) == found[0]
