"""The CUDA pack: GPU acceleration fetched on its own when the machine can use it
(docs/app-plan.md, section 7).

The app ships with the CPU build of onnxruntime. When an eligible NVIDIA GPU is found
(`detect`), the pack -- onnxruntime-gpu and the nvidia CUDA/cuDNN wheels -- is one more
line in the first-run setup, or a toast later. The pack is nothing we host: its manifest,
shipped in the package (`packs/cuda-<platform>.json`, written by scripts/build_pack.py
from PyPI's index), names the exact wheel files with their PyPI URLs and SHA-256; `install`
downloads them from PyPI, verifies each, and unpacks them into a `site-packages` folder
under the data directory (Tai, 2026-09-21: a new user with nothing must get the GPU on
their own). `activate` puts an installed pack ahead of the base environment on `sys.path`
before onnxruntime is first imported and preloads its DLLs; a pack that fails to
initialise leaves the CPU in charge and says so. The base environment is never modified:
removing the pack is deleting its folder.

`gpu.mode` in the settings: auto (the default), cpu (never), cuda (always; a failure is
an error). The state of the pack and of the GPU is what `status` reports to the page.
"""

from __future__ import annotations

import contextlib
import json
import os
import platform
import shutil
import sys
import urllib.request
import zipfile
from importlib import resources
from pathlib import Path
from typing import Any

from sidecoach.config import Settings

# The CUDA 13 wheels want a driver from the R580 branch or later (NVIDIA's CUDA 13.0
# release notes); the pack's manifest carries the exact minimum it was built against.
MIN_DRIVER_MAJOR = 580
MIN_VRAM_BYTES = 4 * 1024**3
PACK_VERSION = "1"
# A manifest elsewhere (a path or a URL) for trying one out: the shipped one otherwise.
PACK_MANIFEST_OVERRIDE = os.environ.get("SIDECOACH_PACK_MANIFEST", "")

_activated: dict[str, Any] = {"pack": None, "cuda": False, "detail": ""}


def platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        machine = "x86_64"
    return f"{system}-{machine}"


def detect() -> dict[str, Any]:
    """The NVIDIA GPU, if any: name, driver, VRAM, and whether it is eligible for the pack."""
    out: dict[str, Any] = {"found": False, "eligible": False, "reason": ""}
    try:
        import pynvml  # type: ignore[import-not-found]

        pynvml.nvmlInit()
        try:
            count = pynvml.nvmlDeviceGetCount()
            if count == 0:
                out["reason"] = "no NVIDIA GPU"
                return out
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            driver = pynvml.nvmlSystemGetDriverVersion()
            memory = int(pynvml.nvmlDeviceGetMemoryInfo(handle).total)
        finally:
            with contextlib.suppress(Exception):
                pynvml.nvmlShutdown()
        name = name.decode() if isinstance(name, bytes) else str(name)
        driver = driver.decode() if isinstance(driver, bytes) else str(driver)
        out.update({"found": True, "name": name, "driver": driver, "vram": int(memory)})
        major = int(driver.split(".")[0]) if driver.split(".")[0].isdigit() else 0
        if major < MIN_DRIVER_MAJOR:
            out["reason"] = (
                f"driver {driver} is older than the {MIN_DRIVER_MAJOR} branch CUDA 13 needs"
            )
        elif memory < MIN_VRAM_BYTES:
            out["reason"] = f"{memory / 1024**3:.1f} GB of VRAM is under the 4 GB the models want"
        else:
            out["eligible"] = True
    except Exception as exc:  # noqa: BLE001 - no NVML means no NVIDIA driver
        out["reason"] = f"no NVIDIA driver ({type(exc).__name__})"
    return out


def packs_dir(settings: Settings) -> Path:
    return settings.data_dir / "packs" / "cuda"


def installed(settings: Settings) -> Path | None:
    """The newest installed pack's folder (its `site-packages` inside), or None."""
    root = packs_dir(settings)
    if not root.exists():
        return None
    # A pack is complete when its manifest is beside its site-packages; a staging folder
    # of an install that did not finish is neither.
    versions = sorted(
        (
            p
            for p in root.iterdir()
            if not p.name.endswith(".staging")
            and (p / "site-packages").is_dir()
            and (p / "manifest.json").is_file()
        ),
        key=lambda p: p.name,
    )
    return versions[-1] if versions else None


def manifest_source() -> str:
    """Where this platform's manifest comes from: the package, unless overridden."""
    return PACK_MANIFEST_OVERRIDE or f"sidecoach.setup.packs/cuda-{platform_tag()}.json"


def manifest() -> dict[str, Any]:
    """The pack for this platform: the wheels to fetch, from PyPI, with their digests."""
    if PACK_MANIFEST_OVERRIDE:
        if PACK_MANIFEST_OVERRIDE.startswith(("http://", "https://", "file:")):
            request = urllib.request.Request(
                PACK_MANIFEST_OVERRIDE, headers={"User-Agent": "sidecoach-setup"}
            )
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return json.load(response)
        return json.loads(Path(PACK_MANIFEST_OVERRIDE).read_text(encoding="utf-8"))
    name = f"cuda-{platform_tag()}.json"
    ref = resources.files("sidecoach.setup.packs") / name
    if not ref.is_file():
        raise RuntimeError(f"no CUDA pack for {platform_tag()} (no {name} shipped)")
    return json.loads(ref.read_text(encoding="utf-8"))


def pack_size(m: dict[str, Any]) -> int:
    return int(m.get("size") or sum(int(w.get("size") or 0) for w in m.get("wheels", [])))


def unpack_wheel(wheel: Path, site: Path) -> None:
    """A wheel into a site-packages folder: its top-level files as they are, and the
    `<name>.data/purelib|platlib` trees merged in; scripts and headers are not ours."""
    with zipfile.ZipFile(wheel) as z:
        for info in z.infolist():
            parts = info.filename.split("/")
            if len(parts) > 1 and parts[0].endswith(".data"):
                if len(parts) > 2 and parts[1] in ("purelib", "platlib"):
                    rel = "/".join(parts[2:])
                else:
                    continue
            else:
                rel = info.filename
            if not rel or rel.endswith("/"):
                continue
            out = site / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, out.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def install(settings: Settings, progress: Any = None) -> Path:
    """Fetch this platform's wheels from PyPI, verify them, and unpack them into a pack
    under the data directory. Progress counts bytes over the whole pack."""
    from sidecoach.setup import download

    m = manifest()
    version = str(m.get("version", PACK_VERSION))
    target = packs_dir(settings) / version
    if (target / "site-packages").exists():
        return target
    staging = packs_dir(settings) / f"{version}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    wheels_dir = staging / "wheels"
    site = staging / "site-packages"
    site.mkdir(parents=True)
    total = pack_size(m)
    done_before = 0
    try:
        for wheel in m["wheels"]:
            base = done_before

            def report(done: int, _total: int | None, base: int = base) -> None:
                if progress:
                    progress(base + done, total)

            path = download(
                wheel["url"], wheels_dir / wheel["filename"], report, wheel.get("sha256")
            )
            size = int(wheel.get("size") or path.stat().st_size)
            unpack_wheel(path, site)
            path.unlink()
            done_before += size
        shutil.rmtree(wheels_dir, ignore_errors=True)
        (staging / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        staging.replace(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)  # nothing half-done is left behind
        raise
    return target


def remove(settings: Settings) -> bool:
    root = packs_dir(settings)
    if not root.exists():
        return False
    shutil.rmtree(root)
    return True


def activate(settings: Settings) -> dict[str, Any]:
    """Before onnxruntime is imported: put the pack first on sys.path and preload its
    DLLs, as `gpu.mode` allows. Returns what happened, also kept for `status`."""
    mode = settings.gpu.mode
    _activated.update({"pack": None, "cuda": False, "detail": ""})
    if mode == "cpu":
        _activated["detail"] = "CPU by setting"
        return dict(_activated)
    if "onnxruntime" in sys.modules:
        _activated["detail"] = "onnxruntime was imported before the pack could be placed"
        return dict(_activated)
    pack = installed(settings)
    if pack is None:
        _activated["detail"] = "no pack installed"
        return dict(_activated)
    site_packages = pack / "site-packages"
    sys.path.insert(0, str(site_packages))
    try:
        import onnxruntime as ort  # noqa: PLC0415

        with contextlib.suppress(Exception):
            ort.preload_dlls()
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            for bin_dir in sorted(site_packages.glob("nvidia/*/bin")):
                with contextlib.suppress(Exception):
                    ort.preload_dlls(directory=str(bin_dir))
                if hasattr(os, "add_dll_directory"):
                    with contextlib.suppress(Exception):
                        os.add_dll_directory(str(bin_dir))
        cuda = "CUDAExecutionProvider" in ort.get_available_providers()
        _activated.update({"pack": str(pack), "cuda": cuda})
        _activated["detail"] = f"pack {pack.name}: CUDA {'ready' if cuda else 'not available; CPU'}"
        if not cuda and mode == "cuda":
            _activated["detail"] += " (gpu.mode is cuda: this is an error)"
    except Exception as exc:  # noqa: BLE001 - the CPU build stays in charge
        with contextlib.suppress(ValueError):
            sys.path.remove(str(site_packages))
        _activated["detail"] = f"pack failed: {type(exc).__name__}: {exc}; CPU"
    return dict(_activated)


def status(settings: Settings) -> dict[str, Any]:
    pack = installed(settings)
    return {
        "mode": settings.gpu.mode,
        "gpu": detect(),
        "installed": str(pack) if pack else None,
        "active": dict(_activated),
        "manifest": manifest_source(),
    }


def pack_item(settings: Settings) -> Any:
    """The pack as a setup item: offered (ticked) when the GPU is eligible and the mode
    allows, absent otherwise."""
    from sidecoach.setup import Item

    if settings.gpu.mode == "cpu":
        return None
    gpu = detect()
    if not gpu.get("eligible"):
        return None
    try:
        size = pack_size(manifest())
    except Exception:  # noqa: BLE001 - no manifest for this platform: no item
        return None
    return Item(
        "cuda",
        f"GPU acceleration for the voice ({gpu.get('name', 'NVIDIA')})",
        size,
        lambda: installed(settings) is not None,
        lambda progress: install(settings, progress) and None,
        optional=True,
        detail="onnxruntime-gpu and the CUDA 13 / cuDNN 9 libraries; used from the next start",
    )
