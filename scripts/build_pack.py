"""Build the CUDA pack the app fetches for an eligible NVIDIA GPU (setup/gpu.py).

docs/app-plan.md, section 7: onnxruntime-gpu and the nvidia CUDA 13 / cuDNN 9 wheels at
the versions of this environment (what Tai runs), installed into a scratch environment
with `uv pip --no-deps` (their dependencies are the base environment's), zipped as
`site-packages/...` with a manifest beside it. The sidecar puts the pack's folder ahead
of its own site-packages when activating it, so the pack's onnxruntime shadows the CPU
one; nothing in the base environment changes.

    uv run --no-sync python scripts/build_pack.py --out C:/somewhere/packs

Publish the zip and the JSON as release assets of the `packs-v<version>` tag; the
sidecar reads `<base>/cuda-<platform>.json` (setup/gpu.py, SIDECOACH_PACK_URL to test
against a local folder: `file:///C:/somewhere/packs`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
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


def platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if machine in ("amd64", "x86_64"):
        machine = "x86_64"
    return f"{system}-{machine}"


def pins() -> list[str]:
    """The packages at the versions installed here (the ones the tests and the games ran on)."""
    out = []
    for name in PACKAGES:
        try:
            out.append(f"{name}=={metadata.version(name)}")
        except metadata.PackageNotFoundError:
            print(f"warning: {name} is not installed here; skipped", file=sys.stderr)
    return out


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=ROOT / "desktop" / "resources" / "packs")
    ap.add_argument(
        "--staging", type=Path, default=None, help="scratch folder (a temp dir by default)"
    )
    args = ap.parse_args()
    t0 = time.monotonic()
    staging = args.staging or Path(tempfile.mkdtemp(prefix="sidecoach-pack-"))
    env = staging / "env"
    if env.exists():
        shutil.rmtree(env)
    run("uv", "venv", "--python", "3.12", str(env))
    python = env / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    run("uv", "pip", "install", "--python", str(python), "--no-deps", *pins())
    site = env / (
        "Lib/site-packages" if platform.system() == "Windows" else "lib/python3.12/site-packages"
    )
    for p in site.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)

    tag = platform_tag()
    args.out.mkdir(parents=True, exist_ok=True)
    zip_name = f"cuda-{tag}-v{PACK_VERSION}.zip"
    zip_path = args.out / zip_name
    if zip_path.exists():
        zip_path.unlink()
    files = [p for p in site.rglob("*") if p.is_file()]
    print(f"zipping {len(files)} files from {site}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for p in files:
            z.write(p, "site-packages/" + p.relative_to(site).as_posix())
    digest = hashlib.sha256()
    with zip_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    manifest = {
        "version": PACK_VERSION,
        "platform": tag,
        "url": zip_name,
        "size": zip_path.stat().st_size,
        "sha256": digest.hexdigest(),
        "min_driver": MIN_DRIVER,
        "packages": pins(),
        "built": time.strftime("%Y-%m-%d"),
    }
    (args.out / f"cuda-{tag}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    shutil.rmtree(env, ignore_errors=True)
    print(f"pack: {zip_path} ({manifest['size'] / 1e6:.0f} MB) in {time.monotonic() - t0:.0f} s")
    print(f"manifest: {args.out / f'cuda-{tag}.json'}")


if __name__ == "__main__":
    main()
