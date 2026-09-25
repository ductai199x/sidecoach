"""Build the desktop app and, with --install, install it on this machine.

    uv run --no-sync python scripts/release_desktop.py                  # sidecar + installer
    uv run --no-sync python scripts/release_desktop.py --install        # and install it (Windows)
    uv run --no-sync python scripts/release_desktop.py --skip-sidecar   # the shell only

The steps, each logged under data/build/:
1. `scripts/build_sidecar.py`: the embedded Python is a copy of the package, so it is rebuilt
   before every installer build, or the installer ships the old coach (docs/app-plan.md).
2. `tauri build` in desktop/: the NSIS installer on Windows; the AppImage and the .deb on
   Linux (`APPIMAGELAUNCHER_DISABLE=1` set; the linuxdeploy plugin pinned in
   ~/.cache/tauri/ is a one-time setup of the machine, docs/app-plan.md). The updater's
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


def _install(version: str) -> None:
    setup = (
        DESKTOP / "src-tauri" / "target" / "release" / "bundle" / "nsis"
    ) / f"Sidecoach_{version}_x64-setup.exe"
    if not setup.exists():
        sys.exit(f"no installer at {setup}")
    if _running():
        sys.exit("Sidecoach is running: not installing over a live session. Close it first.")
    print(f"install: {setup.name} /S", flush=True)
    code = subprocess.call([str(setup), "/S"])
    if code != 0:
        sys.exit(f"the installer failed (exit {code})")
    exe = Path(os.environ["LOCALAPPDATA"]) / "Sidecoach" / "sidecoach.exe"
    installed = subprocess.run(
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
    if installed != version:
        sys.exit(f"installed {installed or 'nothing'}, expected {version}")
    print(f"install: Sidecoach {installed} at {exe}", flush=True)


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
    # differ. The check needs the phone's repo beside this one.
    if (ROOT.parent / "sidecoach-android").exists():
        check = [sys.executable, str(ROOT / "scripts" / "bump_version.py"), "--check"]
        if subprocess.call(check, stdout=subprocess.DEVNULL) != 0:
            sys.exit("the platforms' versions differ: bump_version.py --check says where")
    else:
        print("no ../sidecoach-android here: the versions are not checked", flush=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    version = _version()
    print(f"Sidecoach {version}", flush=True)

    if not args.skip_sidecar:
        _run("sidecar", [sys.executable, str(ROOT / "scripts" / "build_sidecar.py")], ROOT)

    npm = shutil.which("npm")
    if npm is None:
        sys.exit("npm is not on PATH")
    command = [npm, "run", "tauri", "--", "build"]
    env = dict(os.environ)
    if not env.get("TAURI_SIGNING_PRIVATE_KEY"):
        command += ["--config", str(DESKTOP / "src-tauri" / "tauri.no-updater.json")]
    if not WINDOWS:
        env["APPIMAGELAUNCHER_DISABLE"] = "1"
    _run("tauri", command, DESKTOP, env)

    bundle = DESKTOP / "src-tauri" / "target" / "release" / "bundle"
    built = [p for p in bundle.rglob(f"*{version}*") if p.is_file()]
    for path in built:
        print(f"built: {path.relative_to(ROOT)} ({path.stat().st_size / 1e6:.0f} MB)")
    if args.install:
        _install(version)


if __name__ == "__main__":
    main()
