"""Write the CUDA pack's manifests, shipped with the app (sidecoach/setup/packs/).

The pack is onnxruntime-gpu and the nvidia CUDA 13 / cuDNN 9 wheels at the versions of
this environment (what the tests and the games ran on). Nothing is built or hosted: for
each platform the manifest names the exact wheel files on PyPI -- URL, size, SHA-256,
from PyPI's own index -- and the app's first-run setup downloads them from there,
verifies them and unpacks them into `<data>/packs/cuda/<version>/site-packages`
(setup/gpu.py). Run it after `uv lock` moves any of these packages.

    uv run --no-sync python scripts/build_pack.py            # both platforms' manifests
    uv run --no-sync python scripts/build_pack.py --check    # only say what would change
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "sidecoach" / "setup" / "packs"
PACK_VERSION = "1"
MIN_DRIVER = "580.00"
PACKAGES = (
    "onnxruntime-gpu",
    "nvidia-cudnn-cu13",
    "nvidia-cublas",
    "nvidia-cuda-runtime",
    "nvidia-cuda-nvrtc",
    "nvidia-cufft",
    "nvidia-curand",
    "nvidia-nvjitlink",
)
# The wheel each platform takes: the platform tag it must carry, and the Python tags allowed.
PLATFORMS = {
    "windows-x86_64": ("win_amd64",),
    "linux-x86_64": ("manylinux", "x86_64"),
}
PY_TAGS = ("py3", "cp312", "abi3")


def pins() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            print(f"warning: {name} is not installed here; skipped", file=sys.stderr)
    return out


def pypi_files(name: str, version: str) -> list[dict]:
    request = urllib.request.Request(
        f"https://pypi.org/pypi/{name}/{version}/json", headers={"User-Agent": "sidecoach"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["urls"]


def pick(files: list[dict], tag: str) -> dict | None:
    """The one wheel for `tag`: the platform tags in its name, and a Python tag we run."""
    must = PLATFORMS[tag]
    wanted = [
        f
        for f in files
        if f["packagetype"] == "bdist_wheel"
        and all(m in f["filename"] for m in must)
        and any(f"-{py}-" in f["filename"] for py in PY_TAGS)
    ]
    if not wanted:
        return None
    # Newest manylinux baseline last; the first is the most compatible choice.
    wanted.sort(key=lambda f: f["filename"])
    return wanted[0]


def manifest(tag: str, versions: dict[str, str]) -> dict:
    wheels = []
    for name, version in versions.items():
        f = pick(pypi_files(name, version), tag)
        if f is None:
            raise SystemExit(f"{name}=={version}: no wheel for {tag} on PyPI")
        wheels.append(
            {
                "name": name,
                "version": version,
                "filename": f["filename"],
                "url": f["url"],
                "size": int(f["size"]),
                "sha256": f["digests"]["sha256"],
            }
        )
    return {
        "version": PACK_VERSION,
        "platform": tag,
        "min_driver": MIN_DRIVER,
        "size": sum(w["size"] for w in wheels),
        "wheels": wheels,
        "written": time.strftime("%Y-%m-%d"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="compare with the shipped manifests")
    args = ap.parse_args()
    versions = pins()
    OUT.mkdir(parents=True, exist_ok=True)
    changed = 0
    for tag in PLATFORMS:
        m = manifest(tag, versions)
        path = OUT / f"cuda-{tag}.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        same = {k: v for k, v in old.items() if k != "written"} == {
            k: v for k, v in m.items() if k != "written"
        }
        note = "" if same else " (changed)"
        print(f"{tag}: {len(m['wheels'])} wheels, {m['size'] / 1e6:.0f} MB{note}")
        if same:
            continue
        changed += 1
        if not args.check:
            path.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
            print(f"  -> {path}")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
