"""The first-run setup's downloads and its run, against a fake local server (no network):
a body cut short, a whole .part, a double click, Cancel mid-item, too little disk, the
pinned Stockfish, the speech models fetched without loading them, the HF cache."""

from __future__ import annotations

import socket
import sys
import threading
import types
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

import sidecoach.setup as setup_mod
from sidecoach.config import Settings
from sidecoach.setup import download

FULL = bytes(range(256)) * 4  # 1024 bytes

Handler = Callable[[socket.socket, str], None]


@pytest.fixture
def serve() -> Iterator[Callable[[Handler], str]]:
    """A one-thread HTTP server answering with a handler; its base URL."""
    servers: list[socket.socket] = []

    def start(handler: Handler) -> str:
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(8)
        servers.append(srv)

        def run() -> None:
            while True:
                try:
                    conn, _ = srv.accept()
                except OSError:
                    return
                with conn:
                    data = b""
                    while b"\r\n\r\n" not in data:
                        chunk = conn.recv(4096)
                        if not chunk:
                            break
                        data += chunk
                    handler(conn, data.decode("latin-1"))

        threading.Thread(target=run, daemon=True).start()
        return f"http://127.0.0.1:{srv.getsockname()[1]}"

    yield start
    for srv in servers:
        srv.close()


def _range_start(request: str) -> int | None:
    for line in request.split("\r\n"):
        if line.lower().startswith("range: bytes="):
            return int(line.split("=", 1)[1].rstrip("-"))
    return None


def ranged(body: bytes) -> Handler:
    """A server that resumes as GitHub's CDN does: 206 for a Range, 416 past the end."""

    def handle(conn: socket.socket, request: str) -> None:
        start = _range_start(request)
        if start is None:
            head = f"HTTP/1.1 200 OK\r\nContent-Length: {len(body)}\r\nConnection: close\r\n\r\n"
            conn.sendall(head.encode() + body)
        elif start >= len(body):
            conn.sendall(
                b"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */"
                + str(len(body)).encode()
                + b"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
        else:
            rest = body[start:]
            head = (
                f"HTTP/1.1 206 Partial Content\r\nContent-Length: {len(rest)}\r\n"
                f"Content-Range: bytes {start}-{len(body) - 1}/{len(body)}\r\n"
                "Connection: close\r\n\r\n"
            )
            conn.sendall(head.encode() + rest)

    return handle


def test_a_body_cut_short_is_not_taken_as_the_file(
    serve: Callable[[Handler], str], tmp_path: Path
) -> None:
    def truncated(conn: socket.socket, request: str) -> None:
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 1024\r\nConnection: close\r\n\r\n")
        conn.sendall(FULL[:500])
        conn.shutdown(socket.SHUT_WR)

    url = serve(truncated)
    target = tmp_path / "kokoro-v1.0.onnx"
    with pytest.raises(setup_mod.IncompleteDownload):
        download(f"{url}/x", target)
    assert not target.exists()
    # What came is kept, to resume from.
    assert (tmp_path / "kokoro-v1.0.onnx.part").stat().st_size == 500


def test_a_cut_download_resumes_to_the_whole_file(
    serve: Callable[[Handler], str], tmp_path: Path
) -> None:
    url = serve(ranged(FULL))
    target = tmp_path / "voices-v1.0.bin"
    target.with_suffix(".bin.part").write_bytes(FULL[:300])
    download(f"{url}/y", target, expect_size=len(FULL))
    assert target.read_bytes() == FULL


def test_a_whole_part_is_finished_not_refused_forever(
    serve: Callable[[Handler], str], tmp_path: Path
) -> None:
    url = serve(ranged(FULL))
    target = tmp_path / "voices-v1.0.bin"
    part = target.with_suffix(".bin.part")
    part.write_bytes(FULL)  # a kill between the last write and the rename
    download(f"{url}/y", target)  # the server answers 416 to the Range
    assert target.read_bytes() == FULL
    assert not part.exists()
    # With the size known, no request at all.
    other = tmp_path / "other.bin"
    other.with_suffix(".bin.part").write_bytes(FULL)
    download("http://127.0.0.1:9/never", other, expect_size=len(FULL))
    assert other.read_bytes() == FULL


def test_a_part_longer_than_the_file_starts_over(
    serve: Callable[[Handler], str], tmp_path: Path
) -> None:
    url = serve(ranged(FULL))
    target = tmp_path / "f.bin"
    target.with_suffix(".bin.part").write_bytes(FULL + b"junk")
    download(f"{url}/z", target)  # 416 with a different length: start over
    assert target.read_bytes() == FULL


def test_kokoro_is_present_only_at_its_full_size(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, tools_dir=tmp_path / "tools")
    kokoro = next(i for i in setup_mod.items(settings) if i.id == "kokoro")
    folder = settings.voice_model_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for name in setup_mod.KOKORO_FILES:
        (folder / name).write_bytes(b"x" * 10)  # cut short
    assert not kokoro.present()


HANG_S = 60.0  # a safety net for a thread's signal, never the expected path


def _run_over(s: setup_mod.Setup) -> None:
    """Wait for the setup's run thread to end (joined, not polled against a clock)."""
    assert s._thread is not None
    s._thread.join(HANG_S)
    assert not s._thread.is_alive(), "the setup run never ended"
    assert not s.run_state()["running"]


def test_a_double_click_runs_one_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A second Start while the first still looks at the items (detection and the disk checks
    take a while in the real items()), or while it fetches them, is refused: one run, one
    fetch. Each step is held by an Event, so the second click lands exactly there."""
    fetched: list[str] = []
    looked: list[int] = []
    looking, looked_enough = threading.Event(), threading.Event()
    fetching, fetched_enough = threading.Event(), threading.Event()

    def items(settings: Settings) -> list[Any]:
        looked.append(1)
        looking.set()
        assert looked_enough.wait(HANG_S)

        def fetch(progress: Any) -> None:
            fetched.append(threading.current_thread().name)
            fetching.set()
            assert fetched_enough.wait(HANG_S)

        return [setup_mod.Item("kokoro", "voice", 10, lambda: False, fetch)]

    monkeypatch.setattr(setup_mod, "items", items)
    s = setup_mod.Setup(Settings(data_dir=tmp_path))
    results: list[bool] = []
    a = threading.Thread(target=lambda: results.append(s.start()))
    a.start()
    assert looking.wait(HANG_S), "the first click never looked at the items"
    results.append(s.start())  # the second click, while the first one looks
    assert results == [False] and looked == [1], "refused without looking at the items again"
    looked_enough.set()
    a.join(HANG_S)
    assert not a.is_alive()
    assert sorted(results) == [False, True]
    assert fetching.wait(HANG_S), "the run never fetched"
    assert s.start() is False, "a click while the item is fetched is refused too"
    fetched_enough.set()
    _run_over(s)
    assert len(fetched) == 1 and fetched == ["setup"], fetched
    # And a later run is allowed once this one is over.
    assert s.start() is True
    _run_over(s)
    assert len(fetched) == 2


def test_cancel_stops_the_item_under_way(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cancel pressed while the first item is mid-way: the item stops at its next chunk (its
    progress call raises), and the second item never begins. The fetch holds at chunk 10
    until the Cancel is in, so the chunk it stops at is exact."""
    chunks: list[int] = []
    begun: list[str] = []
    midway, cancelled = threading.Event(), threading.Event()

    def items(settings: Settings) -> list[Any]:
        def fetcher(name: str) -> Callable[[Any], None]:
            def fetch(progress: Any) -> None:
                begun.append(name)
                for n in range(200):
                    progress(n, 200)
                    chunks.append(n)
                    if n == 10:
                        midway.set()
                        assert cancelled.wait(HANG_S)

            return fetch

        return [
            setup_mod.Item("a", "A", 200, lambda: False, fetcher("a")),
            setup_mod.Item("b", "B", 200, lambda: False, fetcher("b")),
        ]

    monkeypatch.setattr(setup_mod, "items", items)
    s = setup_mod.Setup(Settings(data_dir=tmp_path))
    assert s.start()
    assert midway.wait(HANG_S), "the first item never got under way"
    assert s.run_state()["item"] == "a" and s.run_state()["running"]
    s.cancel()
    cancelled.set()
    _run_over(s)
    state = s.run_state()
    assert not state["running"]
    assert len(chunks) < 100  # the first item stopped mid-way, the second never began
    assert chunks == list(range(11)), "stopped at the first chunk after the Cancel"
    assert begun == ["a"]
    assert state["finished"] == [] and state["errors"] == {}


def test_too_little_disk_is_an_error_not_a_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fetched: list[str] = []

    def items(settings: Settings) -> list[Any]:
        return [
            setup_mod.Item("big", "Big", 10**12, lambda: False, lambda p: fetched.append("big"))
        ]

    monkeypatch.setattr(setup_mod, "items", items)
    monkeypatch.setattr(setup_mod, "free_space", lambda path: 10**9)
    s = setup_mod.Setup(Settings(data_dir=tmp_path))
    assert s.start()
    _run_over(s)
    assert fetched == []
    assert "free space" in s.run_state()["errors"]["big"]


def test_stockfish_is_the_pinned_release_unpacked_aside(
    serve: Callable[[Handler], str], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = tmp_path / "sf.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("stockfish/stockfish-windows-x86-64-universal.exe", b"MZ" + b"0" * 100)
        z.writestr("stockfish/AUTHORS", "them")
    body = archive.read_bytes()
    asked: list[str] = []

    def handle(conn: socket.socket, request: str) -> None:
        path = request.split(" ")[1]
        asked.append(path)
        if path.endswith("universal.zip"):
            ranged(body)(conn, request)
        else:
            conn.sendall(
                b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )

    url = serve(handle)
    monkeypatch.setattr(setup_mod, "STOCKFISH_RELEASES", url)
    monkeypatch.setattr(
        setup_mod,
        "stockfish_assets",
        lambda *a: ("stockfish-windows-x86-64-gone.zip", "stockfish-windows-x86-64-universal.zip"),
    )
    tools = tmp_path / "tools"
    found = setup_mod.install_stockfish(tools, tmp_path / "staging")
    assert asked[0] == f"/{setup_mod.STOCKFISH_TAG}/stockfish-windows-x86-64-gone.zip"
    assert asked[-1] == f"/{setup_mod.STOCKFISH_TAG}/stockfish-windows-x86-64-universal.zip"
    assert [f.relative_to(tools).as_posix() for f in found] == [
        "stockfish/stockfish-windows-x86-64-universal.exe"
    ]
    assert found[0].read_bytes().startswith(b"MZ")
    # Nothing left in the staging folder but the folder.
    assert [p.name for p in (tmp_path / "staging").iterdir()] == []
    assert setup_mod.STOCKFISH_TAG == "sf_19"


def test_speech_models_are_fetched_without_loading_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolved: list[str] = []

    class Resolver:
        def __init__(self, name: str) -> None:
            self.name = name

        def resolve_model(self) -> dict[str, Path]:
            resolved.append(self.name)
            return {}

    loader = types.ModuleType("onnx_asr.loader")
    loader.create_asr_resolver = Resolver  # type: ignore[attr-defined]
    loader.create_vad_resolver = Resolver  # type: ignore[attr-defined]
    package = types.ModuleType("onnx_asr")

    def no_load(*a: Any, **k: Any) -> None:
        raise AssertionError("the model was loaded to download it")

    package.load_model = no_load  # type: ignore[attr-defined]
    package.load_vad = no_load  # type: ignore[attr-defined]
    package.loader = loader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "onnx_asr", package)
    monkeypatch.setitem(sys.modules, "onnx_asr.loader", loader)
    monkeypatch.delenv("SIDECOACH_DATA_DIR", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    stt = next(i for i in setup_mod.items(Settings(data_dir=tmp_path)) if i.id == "stt")
    stt.fetch(lambda d, t: None)
    assert resolved == ["nemo-parakeet-tdt-0.6b-v3", "silero"]


def test_the_app_uses_its_own_hf_cache_whatever_the_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SIDECOACH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "theirs"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "theirs" / "hub"))
    settings = Settings(data_dir=tmp_path)
    setup_mod.apply_hf_home(settings)
    import os

    assert os.environ["HF_HOME"] == str(setup_mod.hf_home(settings))
    assert "HF_HUB_CACHE" not in os.environ
