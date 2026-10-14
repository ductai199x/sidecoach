"""Build the desktop app and, with --install, install it on this machine.

    uv run --no-sync python scripts/release_desktop.py                  # sidecar + installer
    uv run --no-sync python scripts/release_desktop.py --install        # and install it (Windows)
    uv run --no-sync python scripts/release_desktop.py --skip-sidecar   # the shell only

The steps, each logged under data/build/:
1. `scripts/build_sidecar.py`: the embedded Python is a copy of the package, so it is rebuilt
   before every installer build, or the installer ships the old coach (docs/app.md).
2. `tauri build` in desktop/: the NSIS installer on Windows; the AppImage and the .deb on
   Linux (`APPIMAGELAUNCHER_DISABLE=1` set; the linuxdeploy plugin pinned in
   ~/.cache/tauri/ is a one-time setup of the machine, docs/app.md). The updater's
   artifacts need its signing key (TAURI_SIGNING_PRIVATE_KEY); without the key the build
   leaves them out (src-tauri/tauri.no-updater.json).
3. With --install (Windows): the installer run silently, then the installed version checked
   against tauri.conf.json. It refuses while the app is running -- never over a live game.

It refuses to build when the platforms' versions differ (scripts/bump_version.py --check):
bump them together with scripts/bump_version.py <version>.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESKTOP = ROOT / "desktop"
LOGS = ROOT / "data" / "build"
WINDOWS = sys.platform == "win32"


def _version() -> str:
    conf = json.loads((DESKTOP / "src-tauri" / "tauri.conf.json").read_text(encoding="utf-8"))
    return str(conf["version"])


def _run(name: str, command: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    """`command`, its output in data/build/<name>-<version>.log; stops the release on failure."""
    log = LOGS / f"{name}-{_version()}.log"
    print(f"{name}: {' '.join(command)}  (log: {log})", flush=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as out:
        code = subprocess.call(command, cwd=cwd, stdout=out, stderr=subprocess.STDOUT, env=env)
    if code != 0:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
        sys.exit(f"{name} failed (exit {code}):\n" + "\n".join(tail))
    print(f"{name}: done in {time.monotonic() - started:.0f} s", flush=True)


def _running() -> bool:
    """Whether the installed app is running (Windows)."""
    listed = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq sidecoach.exe", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return "sidecoach.exe" in listed.lower()


def _installed_version(exe: Path) -> str:
    """The product version of the installed exe, or "" when there is none."""
    if not exe.exists():
        return ""
    return subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"(Get-Item '{exe}').VersionInfo.ProductVersion",
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def _same_version_problem(installed: str, version: str) -> str | None:
    """Why installing would silently do nothing, or None. The installer of a version that is
    already installed runs the installed app and quits instead (nsis/hooks.nsh): a rebuild
    without a bump was reported as installed while the old build stayed (and the old app
    was started)."""
    if installed != version:
        return None
    return (
        f"Sidecoach {version} is already installed: its installer would only run the "
        "installed app, not install this build. Bump the version (scripts/bump_version.py)."
    )


def _stamp(path: Path) -> tuple[int, int] | None:
    """What tells one copy of a file from the next: its size and modification time."""
    if not path.exists():
        return None
    st = path.stat()
    return (st.st_size, st.st_mtime_ns)


def _install(version: str) -> None:
    setup = (
        DESKTOP / "src-tauri" / "target" / "release" / "bundle" / "nsis"
    ) / f"Sidecoach_{version}_x64-setup.exe"
    if not setup.exists():
        sys.exit(f"no installer at {setup}")
    if _running():
        sys.exit("Sidecoach is running: not installing over a live session. Close it first.")
    exe = Path(os.environ["LOCALAPPDATA"]) / "Sidecoach" / "sidecoach.exe"
    problem = _same_version_problem(_installed_version(exe), version)
    if problem:
        sys.exit(problem)
    before = _stamp(exe)
    print(f"install: {setup.name} /S", flush=True)
    code = subprocess.call([str(setup), "/S"])
    if code != 0:
        sys.exit(f"the installer failed (exit {code})")
    installed = _installed_version(exe)
    if installed != version:
        sys.exit(f"installed {installed or 'nothing'}, expected {version}")
    if _stamp(exe) == before:
        sys.exit(f"the installer left {exe} as it was: nothing was installed")
    print(f"install: Sidecoach {installed} at {exe}", flush=True)


# The AppImage runtime Tauri's bundler puts in front of the image loads libfuse.so.2, which
# Ubuntu stopped installing in 22.04: on a stock 24.04 the AppImage did not start ("AppImages
# require FUSE to run", 2026-10-13). AppImage's type2 runtime is linked statically and needs no
# FUSE 2. An AppImage is its runtime with the squashfs image after it: the image is kept, the
# runtime swapped for this one, pinned and checked.
RUNTIME_URL = "https://github.com/AppImage/type2-runtime/releases/download/20251108/runtime-x86_64"
RUNTIME_SHA256 = "2fca8b443c92510f1483a883f60061ad09b46b978b2631c807cd873a47ec260d"
# Libraries the AppImage takes from the system it runs on, never the build machine's: they go
# with the system's Mesa, which the AppImage never carries, and every desktop has them. With
# Ubuntu 22.04's in it, WebKit could not make its EGL display under Fedora 44's newer Mesa and
# the window stayed empty (2026-10-14). linuxdeploy, as Tauri runs it, takes no exclusions:
# they are removed from the AppDir and the image packed again (`_without_system_libraries`).
APPIMAGE_SYSTEM_LIBRARIES = ("libwayland-*", "libdrm*")
# The AppImage packer Tauri downloads for its own build (one-time setup, docs/app.md).
APPIMAGE_PACKER = Path.home() / ".cache" / "tauri" / "linuxdeploy-plugin-appimage.AppImage"


def _elf_end(head: bytes) -> int:
    """Where a 64-bit ELF file's own bytes end (its section headers come last): the offset
    of the squashfs image an AppImage carries after its runtime."""
    import struct

    if head[:4] != b"\x7fELF" or head[4] != 2 or head[5] != 1:
        raise ValueError("not a little-endian 64-bit ELF file")
    shoff = struct.unpack_from("<Q", head, 0x28)[0]
    shentsize, shnum = struct.unpack_from("<HH", head, 0x3A)
    return shoff + shentsize * shnum


def _runtime() -> bytes:
    import hashlib
    import urllib.request

    cache = Path.home() / ".cache" / "sidecoach-build" / f"runtime-x86_64-{RUNTIME_SHA256[:12]}"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(RUNTIME_URL, timeout=60) as response:
            cache.write_bytes(response.read())
    data = cache.read_bytes()
    if hashlib.sha256(data).hexdigest() != RUNTIME_SHA256:
        cache.unlink()
        sys.exit(f"the AppImage runtime from {RUNTIME_URL} is not the pinned one")
    return data


def _without_system_libraries(image: Path) -> None:
    """`image` packed again from its AppDir without APPIMAGE_SYSTEM_LIBRARIES."""
    appdirs = list(image.parent.glob("*.AppDir"))
    if len(appdirs) != 1:
        sys.exit(f"{image.name}: expected one AppDir beside it, found {len(appdirs)}")
    libs = appdirs[0] / "usr" / "lib"
    removed = sorted({path for pattern in APPIMAGE_SYSTEM_LIBRARIES for path in libs.glob(pattern)})
    if not removed:
        return
    if not APPIMAGE_PACKER.exists():
        sys.exit(f"{APPIMAGE_PACKER} is missing: Tauri downloads it with its first AppImage")
    for path in removed:
        path.unlink()
    env = dict(os.environ, ARCH="x86_64", OUTPUT=str(image), LDAI_OUTPUT=str(image))
    env["APPIMAGE_EXTRACT_AND_RUN"] = "1"  # the packer is an AppImage too; no FUSE here
    image.unlink()
    _run("repack-appimage", [str(APPIMAGE_PACKER), "--appdir", str(appdirs[0])], image.parent, env)
    if not image.exists():
        sys.exit(f"{image.name}: the AppImage was not packed again")
    print(f"{image.name}: without {', '.join(p.name for p in removed)}", flush=True)


def _static_runtime(image: Path) -> None:
    """`image` with its runtime swapped for the static one (see RUNTIME_URL)."""
    data = image.read_bytes()
    start = _elf_end(data[:4096])
    if data[start : start + 4] != b"hsqs":
        sys.exit(f"{image.name}: no squashfs image after its runtime at {start}")
    runtime = _runtime()
    if data[:start] == runtime:
        return
    swapped = image.with_suffix(".swap")
    swapped.write_bytes(runtime + data[start:])
    swapped.chmod(0o755)
    swapped.replace(image)
    print(f"{image.name}: static runtime (no FUSE 2 needed)", flush=True)
    # The updater's signature was made of the image as Tauri left it: made again of this one,
    # or an update to it would fail its check (the Linux audit, 2026-10-13).
    signature = image.with_name(image.name + ".sig")
    if signature.exists() or os.environ.get("TAURI_SIGNING_PRIVATE_KEY"):
        npm = shutil.which("npm")
        if npm is None:
            sys.exit("npm is not on PATH: the AppImage's updater signature cannot be made again")
        signature.unlink(missing_ok=True)
        _run("sign-appimage", [npm, "run", "tauri", "--", "signer", "sign", str(image)], DESKTOP)
        if not signature.exists():
            sys.exit(f"{image.name}: the updater signature was not made again")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--install", action="store_true", help="install it here (Windows)")
    parser.add_argument("--skip-sidecar", action="store_true", help="keep the embedded Python")
    args = parser.parse_args()
    if args.install and not WINDOWS:
        sys.exit("--install is for Windows; on Linux run the AppImage or install the .deb")
    if args.install and _running():
        sys.exit("Sidecoach is running: not installing over a live session. Close it first.")
    # One version for every platform (scripts/bump_version.py): a build refuses when they
    # differ. The check needs the phone's folder, android/.
    if (ROOT / "android").exists():
        check = [sys.executable, str(ROOT / "scripts" / "bump_version.py"), "--check"]
        if subprocess.call(check, stdout=subprocess.DEVNULL) != 0:
            sys.exit("the platforms' versions differ: bump_version.py --check says where")
    else:
        print("no android/ here: the versions are not checked", flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    version = _version()
    print(f"Sidecoach {version}", flush=True)
    if args.install:
        # Before the build: the installer of an installed version installs nothing.
        exe = Path(os.environ["LOCALAPPDATA"]) / "Sidecoach" / "sidecoach.exe"
        problem = _same_version_problem(_installed_version(exe), version)
        if problem:
            sys.exit(problem)

    if not args.skip_sidecar:
        _run("sidecar", [sys.executable, str(ROOT / "scripts" / "build_sidecar.py")], ROOT)

    npm = shutil.which("npm")
    if npm is None:
        sys.exit("npm is not on PATH")
    if not WINDOWS:
        # Into the .deb as they are on disk: a rule file is read, never run (a copy from
        # Windows came out executable).
        for path in (DESKTOP / "src-tauri" / "linux").glob("*.rules"):
            path.chmod(0o644)
        for path in (DESKTOP / "src-tauri" / "linux").glob("*.sh"):
            path.chmod(0o755)
    command = [npm, "run", "tauri", "--", "build"]
    env = dict(os.environ)
    if not env.get("TAURI_SIGNING_PRIVATE_KEY"):
        command += ["--config", str(DESKTOP / "src-tauri" / "tauri.no-updater.json")]
    if not WINDOWS:
        env["APPIMAGELAUNCHER_DISABLE"] = "1"
    _run("tauri", command, DESKTOP, env)

    bundle = DESKTOP / "src-tauri" / "target" / "release" / "bundle"
    if not WINDOWS:
        for image in list(bundle.glob(f"appimage/*{version}*.AppImage")):
            _without_system_libraries(image)
            _static_runtime(image)
    built = [p for p in bundle.rglob(f"*{version}*") if p.is_file()]
    for path in built:
        print(f"built: {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.0f} MB)")
    if args.install:
        _install(version)


if __name__ == "__main__":
    main()
