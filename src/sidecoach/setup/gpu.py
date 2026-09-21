"""The CUDA pack: GPU acceleration fetched on its own when the machine can use it
(docs/app-plan.md, section 7).

The app ships with the CPU build of onnxruntime. When an eligible NVIDIA GPU is found
(`detect`), the pack -- onnxruntime-gpu and the nvidia CUDA/cuDNN wheels, zipped by
scripts/build_pack.py into a `site-packages` folder with a manifest -- is one more line
in the first-run setup, or a toast later. `activate` puts an installed pack ahead of the
base environment on `sys.path` before onnxruntime is first imported and preloads its
DLLs; a pack that fails to initialise leaves the CPU in charge and says so. The base
environment is never modified: removing the pack is deleting its folder.

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
from pathlib import Path
from typing import Any

from sidecoach.config import Settings

# The CUDA 13 wheels want a driver from the R580 branch or later (NVIDIA's CUDA 13.0
# release notes); the pack's manifest carries the exact minimum it was built against.
MIN_DRIVER_MAJOR = 580
MIN_VRAM_BYTES = 4 * 1024**3
PACK_VERSION = "1"
# Where the packs are published: <base>/cuda-<platform>.json describes the zip beside it.
PACK_BASE = os.environ.get(
    "SIDECOACH_PACK_URL",
    "https://github.com/ductai199x/sidecoach/releases/download/packs-v" + PACK_VERSION,
)

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
    versions = sorted(
        (p for p in root.iterdir() if (p / "site-packages").exists()), key=lambda p: p.name
    )
    return versions[-1] if versions else None


def manifest_url() -> str:
    return f"{PACK_BASE}/cuda-{platform_tag()}.json"


def fetch_manifest() -> dict[str, Any]:
    request = urllib.request.Request(manifest_url(), headers={"User-Agent": "sidecoach-setup"})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.load(response)


def install(settings: Settings, progress: Any = None) -> Path:
    """Fetch the pack for this platform and unpack it under the data directory."""
    from sidecoach.setup import download

    manifest = fetch_manifest()
    version = str(manifest.get("version", PACK_VERSION))
    target = packs_dir(settings) / version
    if (target / "site-packages").exists():
        return target
    url = manifest["url"]
    if not url.startswith(("http://", "https://", "file:")):
        url = f"{PACK_BASE}/{url}"
    archive = download(
        url, packs_dir(settings) / f"cuda-{version}.zip", progress, manifest.get("sha256")
    )
    staging = packs_dir(settings) / f"{version}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    with zipfile.ZipFile(archive) as z:
        z.extractall(staging)
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    staging.replace(target)
    archive.unlink()
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
        "manifest_url": manifest_url(),
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
    return Item(
        "cuda",
        f"GPU acceleration for the voice ({gpu.get('name', 'NVIDIA')})",
        1_400_000_000,
        lambda: installed(settings) is not None,
        lambda progress: install(settings, progress) and None,
        optional=True,
        detail="onnxruntime-gpu and the CUDA 13 / cuDNN 9 libraries; used from the next start",
    )
