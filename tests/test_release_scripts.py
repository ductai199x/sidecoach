"""The release scripts and the doctor (setup findings 14, 23, 25): an install that would do
nothing is refused, the sidecar's check writes no bytecode and runs on data of its own,
and the doctor reports the voice, speech recognition and the GPU, and survives a check
that raises."""

from __future__ import annotations

import importlib.util
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
