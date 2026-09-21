"""Build the embedded Python the desktop shell runs the sidecar with.

docs/app-plan.md, section 6, step 5: a python-build-standalone CPython (the one uv
manages, copied) with a real `site-packages` installed by `uv pip`, carrying the package
and its panel and voice extras on the CPU build of onnxruntime -- no CUDA wheels, no
torch: the CUDA pack (scripts/build_pack.py) adds the GPU at first run when the machine
can use it. Output: `desktop/resources/python/` (gitignored), which tauri.conf.json
bundles as a resource; the shell runs `resources/python/python.exe -m sidecoach panel`.

    uv run --no-sync python scripts/build_sidecar.py            # build
    uv run --no-sync python scripts/build_sidecar.py --check    # run the result's doctor

The requirement set is the lock's (`uv export`), with onnxruntime-gpu swapped for
onnxruntime and the nvidia wheels dropped, so the sidecar runs exactly the versions the
tests ran, on the CPU. Pruned: bytecode, tests shipped inside packages, the interpreter's
own test suite and pip. Size printed at the end (target: under 300 MB).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "desktop" / "resources" / "python"
# Left out of the CPU base: the GPU runtime and its CUDA wheels (the pack), and the
# export-only tooling. nvidia-ml-py stays: the pack's GPU detection needs it.
DROP_EXACT = {"onnxruntime-gpu", "torch", "torchvision", "ultralytics", "onnx"}
DROP_PREFIX = ("nvidia-cu", "nvidia-cudnn", "nvidia-nv", "nvidia-cufft", "nvidia-curand")
PRUNE_DIRS = ("__pycache__", "tests", "test", "testing")
PRUNE_TOP = ("Lib/test", "Lib/idlelib", "Lib/tkinter", "Lib/turtledemo", "Lib/ensurepip")


def run(*args: str, **kw: object) -> str:
    print("+", " ".join(args), flush=True)
    done = subprocess.run(args, text=True, capture_output=True, **kw)  # type: ignore[arg-type]
    if done.returncode != 0:
        sys.stderr.write(done.stderr[-4000:])
        raise SystemExit(f"{args[0]} failed ({done.returncode})")
    return done.stdout


def managed_python() -> Path:
    """uv's managed CPython 3.12 (a python-build-standalone install_only build), asked
    from outside the project so the answer is the distribution, not this venv."""
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    out = run(
        "uv",
        "python",
        "find",
        "--managed-python",
        "--no-project",
        "3.12",
        cwd=tempfile.gettempdir(),
        env=env,
    ).strip()
    exe = Path(out)
    root = exe.parent
    if (root.parent / "pyvenv.cfg").exists():  # a venv after all: the distribution behind it
        for line in (root.parent / "pyvenv.cfg").read_text(encoding="utf-8").splitlines():
            if line.startswith("home ="):
                root = Path(line.split("=", 1)[1].strip())
    if not (root / "python.exe").exists() or not (root / "Lib").exists():
        raise SystemExit(f"cannot find a CPython distribution from {out}")
    return root


def requirements() -> list[str]:
    """The lock's requirement set for the panel and the voice, on the CPU."""
    text = run(
        "uv",
        "export",
        "--group",
        "panel",
        "--group",
        "voice",
        "--no-hashes",
        "--no-emit-project",
        "--no-dev",
        cwd=ROOT,
    )
    lines: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        name = re.split(r"[<>=!~; ]", line, maxsplit=1)[0].lower()
        if name in DROP_EXACT or name.startswith(DROP_PREFIX):
            continue
        if line.startswith("onnxruntime==") and "sys_platform == 'never'" in line:
            # The override that kept the CPU build out of the dev environment: here it is
            # the one we want.
            line = line.split(";")[0].strip()
        lines.append(line)
    if not any(ln.startswith("onnxruntime==") for ln in lines):
        gpu = [ln for ln in text.splitlines() if ln.startswith("onnxruntime-gpu==")]
        version = gpu[0].split("==")[1].split()[0] if gpu else "1.30.0"
        lines.append(f"onnxruntime=={version}")
    # For the GPU pack's detection, tiny.
    if not any(ln.startswith("nvidia-ml-py") for ln in lines):
        lines.append("nvidia-ml-py")
    return lines


def prune(root: Path) -> None:
    for rel in PRUNE_TOP:
        shutil.rmtree(root / rel, ignore_errors=True)
    site = root / "Lib" / "site-packages"
    for name in ("pip", "setuptools", "wheel"):
        for p in site.glob(f"{name}*"):
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    for p in list(root.rglob("*")):
        if p.is_dir() and p.name in PRUNE_DIRS and site in p.parents:
            # Package test suites only; a package named `test` at the top of site-packages
            # is nobody's dependency here.
            shutil.rmtree(p, ignore_errors=True)
    for p in root.rglob("*.pyc"):
        p.unlink(missing_ok=True)


def size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


def build(out: Path) -> None:
    t0 = time.monotonic()
    dist = managed_python()
    print(f"distribution: {dist}")
    if out.exists():
        shutil.rmtree(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dist, out, ignore=shutil.ignore_patterns("__pycache__"))
    # uv marks its own distributions as not to be modified; this copy is ours to fill.
    for marker in out.rglob("EXTERNALLY-MANAGED"):
        marker.unlink()
    python = out / "python.exe"
    req = out.parent / "requirements-sidecar.txt"
    req.write_text("\n".join(requirements()) + "\n", encoding="utf-8")
    # From outside the project: its [tool.uv] override (onnxruntime never) must not apply.
    neutral = tempfile.gettempdir()
    run("uv", "pip", "install", "--python", str(python), "-r", str(req), cwd=neutral)
    run("uv", "pip", "install", "--python", str(python), "--no-deps", str(ROOT), cwd=neutral)
    prune(out)
    # A marker the shell and doctor can show.
    (out / "SIDECOACH_BUILD").write_text(
        time.strftime("%Y-%m-%d %H:%M:%S") + "\n", encoding="utf-8"
    )
    mb = size(out) / 1e6
    print(f"built {out} in {time.monotonic() - t0:.0f} s: {mb:.0f} MB")


def check(out: Path) -> None:
    python = out / "python.exe"
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)
    print(
        subprocess.run(
            [
                str(python),
                "-c",
                "import sidecoach, onnxruntime, fastapi, kokoro_onnx, onnx_asr; "
                "print('imports ok;', onnxruntime.__version__, "
                "onnxruntime.get_available_providers())",
            ],
            text=True,
            capture_output=True,
            env=env,
            check=True,
        ).stdout
    )
    print(
        subprocess.run(
            [str(python), "-m", "sidecoach", "doctor"],
            text=True,
            capture_output=True,
            env=env,
            cwd=ROOT,
        ).stdout[-1500:]
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--check", action="store_true", help="only run the built interpreter's doctor")
    args = ap.parse_args()
    if args.check:
        check(args.out)
        return
    build(args.out)
    check(args.out)


if __name__ == "__main__":
    sys.exit(main())
