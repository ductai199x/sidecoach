"""The CUDA pack: the manifests shipped with the app name PyPI wheels for each platform,
and `install` fetches, verifies and unpacks them into the data folder (Tai, 2026-09-21:
a new user with nothing gets the GPU on their own; nothing hosted, nothing lying about)."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from sidecoach.config import Settings
from sidecoach.setup import gpu

PACKS = Path(gpu.__file__).parent / "packs"


@pytest.mark.parametrize(
    "tag,mark", [("windows-x86_64", "win_amd64"), ("linux-x86_64", "manylinux")]
)
def test_the_shipped_manifest_names_pypi_wheels(tag: str, mark: str) -> None:
    m = json.loads((PACKS / f"cuda-{tag}.json").read_text(encoding="utf-8"))
    assert m["platform"] == tag and m["version"] == gpu.PACK_VERSION
    names = {w["name"] for w in m["wheels"]}
    assert {"onnxruntime-gpu", "nvidia-cudnn-cu13", "nvidia-cublas", "nvidia-cuda-runtime"} <= names
    for w in m["wheels"]:
        assert w["url"].startswith("https://files.pythonhosted.org/") and w["filename"].endswith(
            ".whl"
        )
        assert mark in w["filename"] and len(w["sha256"]) == 64 and w["size"] > 0
    assert m["size"] == sum(w["size"] for w in m["wheels"])
    assert gpu.pack_size(m) == m["size"]


def _one_wheel_pack(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, str],
    version: str,
    sha256: str | None = None,
) -> Path:
    """A manifest of one wheel holding `files`, served from a file URL, put in place of the
    shipped one; the wheel's path. `sha256` stands in for the wheel's real digest."""
    wheel = tmp_path / "foo-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    entry = {
        "name": "foo",
        "version": "1.0",
        "filename": wheel.name,
        "url": wheel.as_uri(),
        "size": wheel.stat().st_size,
        "sha256": sha256 or hashlib.sha256(wheel.read_bytes()).hexdigest(),
    }
    manifest = {"version": version, "platform": gpu.platform_tag(), "wheels": [entry]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(gpu, "PACK_MANIFEST_OVERRIDE", str(tmp_path / "manifest.json"))
    return wheel


def test_install_fetches_verifies_and_unpacks_the_wheels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest of one wheel served from a file URL: the wheel's packages and its
    `.data/purelib` land in site-packages, its scripts do not, the wheel is not kept."""
    files = {
        "foo/__init__.py": "VALUE = 1\n",
        "foo-1.0.dist-info/METADATA": "Name: foo\nVersion: 1.0\n",
        "foo-1.0.data/purelib/bar.py": "BAR = 2\n",
        "foo-1.0.data/scripts/foo-cli": "#!/bin/sh\n",
    }
    wheel = _one_wheel_pack(tmp_path, monkeypatch, files, "7")
    settings = Settings(data_dir=tmp_path / "data")
    seen: list[tuple[int, int | None]] = []
    pack = gpu.install(settings, lambda done, total: seen.append((done, total)))
    site = pack / "site-packages"
    assert pack == gpu.packs_dir(settings) / "7" and gpu.installed(settings) == pack
    assert (site / "foo" / "__init__.py").read_text() == "VALUE = 1\n"
    assert (site / "bar.py").exists() and (site / "foo-1.0.dist-info" / "METADATA").exists()
    assert not list(site.rglob("foo-cli")), "scripts are not ours"
    assert not list(pack.rglob("*.whl")) and not (pack.parent / "7.staging").exists()
    assert json.loads((pack / "manifest.json").read_text())["version"] == "7"
    assert seen and seen[-1][0] == wheel.stat().st_size and seen[-1][1] == wheel.stat().st_size
    # A second install is a no-op; removing the pack is removing its folder.
    assert gpu.install(settings) == pack
    assert gpu.remove(settings) and gpu.installed(settings) is None


def test_a_wrong_digest_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _one_wheel_pack(tmp_path, monkeypatch, {"foo/__init__.py": ""}, "1", sha256="0" * 64)
    settings = Settings(data_dir=tmp_path / "data")
    with pytest.raises(RuntimeError, match="checksum"):
        gpu.install(settings)
    assert gpu.installed(settings) is None


def test_an_interrupted_pack_resumes_where_it_stopped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Setup finding 20: a failure part-way deleted the staging folder, so every retry of
    the ~1.5 GB pack started from nothing. Wheels already unpacked are not fetched again."""
    first = _one_wheel_pack(tmp_path, monkeypatch, {"foo/__init__.py": "A = 1\n"}, "3")
    second = tmp_path / "bar-1.0-py3-none-any.whl"
    with zipfile.ZipFile(second, "w") as z:
        z.writestr("bar/__init__.py", "B = 2\n")
    m = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    m["wheels"].append(
        {
            "name": "bar",
            "version": "1.0",
            "filename": second.name,
            "url": second.as_uri(),
            "size": second.stat().st_size,
            "sha256": "0" * 64,  # wrong: the run fails on this wheel
        }
    )
    (tmp_path / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    settings = Settings(data_dir=tmp_path / "data")
    with pytest.raises(RuntimeError, match="checksum"):
        gpu.install(settings)
    assert gpu.installed(settings) is None
    # The second wheel fixed; the first must not be fetched again.
    m["wheels"][1]["sha256"] = hashlib.sha256(second.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    fetched: list[str] = []
    import sidecoach.setup as setup_mod

    real = setup_mod.download

    def counting(url: str, *a: Any, **k: Any) -> Path:
        fetched.append(url)
        return real(url, *a, **k)

    monkeypatch.setattr(setup_mod, "download", counting)
    pack = gpu.install(settings)
    assert fetched == [second.as_uri()]
    assert (pack / "site-packages" / "foo" / "__init__.py").read_text() == "A = 1\n"
    assert (pack / "site-packages" / "bar" / "__init__.py").read_text() == "B = 2\n"
    assert first.exists() and not (pack.parent / "3.staging").exists()


def test_the_setup_status_asks_nvml_once_in_a_while(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setup finding 21: the status (polled twice a second) asked NVML twice a call."""
    calls: list[int] = []
    monkeypatch.setattr(gpu, "detect", lambda: calls.append(1) or {"eligible": False})
    monkeypatch.setattr(gpu, "_detected", {"at": -1e9, "value": None})
    settings = Settings()
    for _ in range(5):
        gpu.status(settings)
        gpu.pack_item(settings)
    assert len(calls) == 1


def test_new_wheels_under_the_shipped_version_are_refused() -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "build_pack.py"
    spec = importlib.util.spec_from_file_location("build_pack", path)
    assert spec and spec.loader
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    old = {"version": "1", "wheels": [{"name": "a", "sha256": "x"}]}
    assert build.needs_bump(old, {"version": "1", "wheels": [{"name": "a", "sha256": "y"}]})
    assert not build.needs_bump(old, {"version": "2", "wheels": [{"name": "a", "sha256": "y"}]})
    assert not build.needs_bump(old, dict(old))
    assert not build.needs_bump({}, dict(old))
