"""The release scripts and the doctor (setup findings 14, 23, 25): an install that would do
nothing is refused, the sidecar's check writes no bytecode and runs on data of its own,
and the doctor reports the voice, speech recognition and the GPU, and survives a check
that raises."""

from __future__ import annotations

import importlib.util
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_installing_a_version_already_installed_is_refused() -> None:
    """The installer of an installed version runs the installed app and quits
    (nsis/hooks.nsh): --install reported success while the old build stayed."""
    release = _script("release_desktop")
    problem = release._same_version_problem("0.1.53", "0.1.53")
    assert problem and "bump_version" in problem
    assert release._same_version_problem("0.1.52", "0.1.53") is None
    assert release._same_version_problem("", "0.1.53") is None


def test_an_install_that_changed_nothing_is_seen(tmp_path: Path) -> None:
    release = _script("release_desktop")
    exe = tmp_path / "sidecoach.exe"
    assert release._stamp(exe) is None
    exe.write_bytes(b"old")
    before = release._stamp(exe)
    assert release._stamp(exe) == before
    exe.write_bytes(b"newer")
    assert release._stamp(exe) != before


def test_the_sidecar_check_writes_no_bytecode_and_uses_its_own_data(tmp_path: Path) -> None:
    build = _script("build_sidecar")
    env = build.check_env(tmp_path / "data")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["SIDECOACH_DATA_DIR"] == str(tmp_path / "data")
    assert "VIRTUAL_ENV" not in env
    # A check this machine fails is not a broken build; a crash is.
    assert not build.doctor_failed(0, "ok")
    assert not build.doctor_failed(1, "LLM error")
    assert build.doctor_failed(1, "Traceback (most recent call last):\n  boom")
    assert build.doctor_failed(2, "")


def test_the_doctor_reports_voice_speech_and_gpu_and_survives_a_raising_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from sidecoach import cli

    def boom() -> tuple[str, str]:
        raise RuntimeError("codec probe exploded")

    monkeypatch.setattr(cli, "_check_video_codec", boom)
    monkeypatch.setattr(cli, "_check_vision_libs", lambda: ("ok", "fine"))
    monkeypatch.setattr(cli, "_check_engine", lambda s: ("missing", "none"))
    monkeypatch.setattr(cli, "_check_openings", lambda s: ("missing", "none"))
    monkeypatch.setattr(cli, "_check_llm", lambda s: ("ok", "fake"))
    monkeypatch.setattr(cli, "_check_capture", lambda: ("ok", "fake"))
    monkeypatch.setattr(cli, "_check_gpu", lambda s: ("ok", "CPU"))
    monkeypatch.setenv("SIDECOACH_DATA_DIR", str(tmp_path))
    result = CliRunner().invoke(cli.app, ["doctor"], env={"COLUMNS": "200"})
    assert "codec probe exploded" in result.output
    for row in ("Voice", "Speech recognition", "GPU"):
        assert row in result.output
    assert result.exit_code == 1  # an error row, reported; not a crash
    assert result.exception is None or isinstance(result.exception, SystemExit)


def _elf(size: int) -> bytes:
    """A 64-bit little-endian ELF file of `size` bytes whose section headers end it."""
    import struct

    head = bytearray(size)
    head[:6] = b"\x7fELF\x02\x01"
    shentsize, shnum = 64, 2
    struct.pack_into("<Q", head, 0x28, size - shentsize * shnum)
    struct.pack_into("<HH", head, 0x3A, shentsize, shnum)
    return bytes(head)


def test_the_appimage_gets_the_static_runtime_and_keeps_its_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tauri's AppImage runtime loads libfuse.so.2, which stock Ubuntu 24.04 lacks: the AppImage
    did not start (2026-10-13). The release swaps the runtime for the static one and keeps the
    squashfs image after it byte for byte; a second pass changes nothing."""
    release = _script("release_desktop")
    old, new = _elf(512), _elf(768)
    image = b"hsqs" + bytes(range(256)) * 4
    appimage = tmp_path / "Sidecoach_0.1.89_amd64.AppImage"
    appimage.write_bytes(old + image)
    monkeypatch.setattr(release, "_runtime", lambda: new)
    release._static_runtime(appimage)
    assert appimage.read_bytes() == new + image
    release._static_runtime(appimage)
    assert appimage.read_bytes() == new + image
    assert release._elf_end(new) == len(new)
    with pytest.raises(ValueError):
        release._elf_end(b"MZ" + bytes(100))
    appimage.write_bytes(old + b"not an image")
    with pytest.raises(SystemExit, match="no squashfs"):
        release._static_runtime(appimage)


def test_the_appimage_is_packed_again_without_the_systems_graphics_libraries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ubuntu 22.04's libwayland and libdrm in the AppImage left Fedora 44's window empty
    (WebKit had no EGL display under the newer Mesa, 2026-10-14): they are removed from the
    AppDir and the image packed again from it; the rest of the AppDir stays, and an image
    without them is left as it is. The packer is a stand-in that records its call."""
    release = _script("release_desktop")
    bundle = tmp_path / "appimage"
    libs = bundle / "Sidecoach.AppDir" / "usr" / "lib"
    libs.mkdir(parents=True)
    names = ("libwayland-client.so.0", "libwayland-egl.so.1", "libdrm-b0291a67.so.2.4.0")
    for name in (*names, "libwebkit2gtk-4.1.so.0", "libgtk-3.so.0"):
        (libs / name).write_bytes(b"x")
    image = bundle / "Sidecoach_0.1.89_amd64.AppImage"
    image.write_bytes(b"tauri's")
    packer = tmp_path / "packer"
    packer.write_bytes(b"")
    calls: list[tuple[list[str], dict[str, str]]] = []

    def run(name: str, command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
        calls.append((command, env or {}))
        Path((env or {})["OUTPUT"]).write_bytes(b"packed")

    monkeypatch.setattr(release, "APPIMAGE_PACKER", packer)
    monkeypatch.setattr(release, "_run", run)
    release._without_system_libraries(image)
    assert sorted(p.name for p in libs.iterdir()) == ["libgtk-3.so.0", "libwebkit2gtk-4.1.so.0"]
    assert image.read_bytes() == b"packed"
    ((command, env),) = calls
    assert command == [str(packer), "--appdir", str(libs.parent.parent)]
    assert env["OUTPUT"] == str(image) and env["APPIMAGE_EXTRACT_AND_RUN"] == "1"
    release._without_system_libraries(image)  # nothing left to remove: not packed again
    assert len(calls) == 1 and image.read_bytes() == b"packed"


def test_the_update_manifest_offers_each_installer_only_its_own_file() -> None:
    """latest.json (scripts/update_manifest.py): the Windows installer under both names the
    updater asks for, the AppImage and the .deb each under its own installer's name only (no
    plain linux-x86_64, which would hand an AppImage a .deb), the URLs at the release's own
    downloads; a file of another version is not listed."""
    manifest = _script("update_manifest").manifest
    files = {
        "Sidecoach_0.1.90_x64-setup.exe": "sig-exe",
        "Sidecoach_0.1.90_amd64.AppImage": "sig-appimage",
        "Sidecoach_0.1.90_amd64.deb": "sig-deb",
        "Sidecoach_0.1.89_amd64.deb": "sig-old",
    }
    when = datetime(2026, 10, 16, 4, 30, tzinfo=UTC)
    base = "https://github.com/ductai199x/sidecoach/releases/download/v0.1.90"
    data = manifest("0.1.90", files, "What's new", base, when)
    assert data["version"] == "0.1.90" and data["notes"] == "What's new"
    assert data["pub_date"] == "2026-10-16T04:30:00Z"
    platforms = data["platforms"]
    assert sorted(platforms) == [
        "linux-x86_64-appimage",
        "linux-x86_64-deb",
        "windows-x86_64",
        "windows-x86_64-nsis",
    ]
    assert (
        platforms["windows-x86_64"]
        == platforms["windows-x86_64-nsis"]
        == {
            "signature": "sig-exe",
            "url": base + "/Sidecoach_0.1.90_x64-setup.exe",
        }
    )
    assert platforms["linux-x86_64-appimage"]["url"] == base + "/Sidecoach_0.1.90_amd64.AppImage"
    assert platforms["linux-x86_64-deb"]["signature"] == "sig-deb"
