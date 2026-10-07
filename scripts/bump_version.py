"""One version for every platform: the desktop app (Windows, Linux) and the phone.

    uv run --no-sync python scripts/bump_version.py 0.1.14    # set it everywhere
    uv run --no-sync python scripts/bump_version.py --check   # exit 1 when they differ

Every platform takes the new version, including one whose code did not change (Tai,
2026-09-25: "the version number should be synced for all platforms ... even though sometimes
we update only 1"), so a version names the same coach on the desktop and the phone. The
phone is the android/ folder (or --android); its versionCode is the version as a
number (0.1.14 -> 114), always rising. Nothing is committed.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (file, pattern with the version as group 2 between groups 1 and 3), in the desktop repo.
DESKTOP = [
    ("pyproject.toml", r'(?m)^(version = ")([^"]+)(")'),
    ("src/sidecoach/__init__.py", r'(?m)^(__version__ = ")([^"]+)(")'),
    ("uv.lock", r'(name = "sidecoach"\r?\nversion = ")([^"]+)(")'),
    ("desktop/package.json", r'(?m)^(  "version": ")([^"]+)(")'),
    ("desktop/src-tauri/tauri.conf.json", r'(?m)^(  "version": ")([^"]+)(")'),
    ("desktop/src-tauri/Cargo.toml", r'(?m)^(version = ")([^"]+)(")'),
    ("desktop/src-tauri/Cargo.lock", r'(name = "sidecoach-desktop"\r?\nversion = ")([^"]+)(")'),
]
GRADLE = "app/build.gradle.kts"
VERSION_NAME = r'(versionName = ")([^"]+)(")'
VERSION_CODE = r"(versionCode = )(\d+)()"


def code_of(version: str) -> int:
    """The versionCode for `version`: 0.1.14 -> 114, 1.2.3 -> 10203."""
    major, minor, patch = (int(part) for part in version.split("."))
    if minor > 99 or patch > 99:
        raise ValueError(f"{version}: minor and patch must stay under 100")
    return major * 10000 + minor * 100 + patch


def _read(path: Path, pattern: str) -> str:
    match = re.search(pattern, path.read_text(encoding="utf-8"))
    if match is None:
        sys.exit(f"no version in {path}")
    return match.group(2)


def _write(path: Path, pattern: str, value: str) -> None:
    raw = path.read_bytes().decode("utf-8")
    new, count = re.subn(pattern, lambda m: m.group(1) + value + m.group(3), raw, count=1)
    if count != 1:
        sys.exit(f"no version in {path}")
    path.write_bytes(new.encode("utf-8"))


def versions(android: Path) -> dict[str, str]:
    """Every file's version, by file."""
    found = {rel: _read(ROOT / rel, pattern) for rel, pattern in DESKTOP}
    found[f"android/{GRADLE}"] = _read(android / GRADLE, VERSION_NAME)
    code = int(_read(android / GRADLE, VERSION_CODE))
    name = found[f"android/{GRADLE}"]
    if code != code_of(name):
        found[f"android/{GRADLE} (versionCode {code})"] = f"code {code} != {code_of(name)}"
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("version", nargs="?", help="the new version, e.g. 0.1.14")
    parser.add_argument("--check", action="store_true", help="exit 1 when the platforms differ")
    parser.add_argument("--android", type=Path, default=ROOT / "android")
    args = parser.parse_args()
    if not (args.android / GRADLE).exists():
        sys.exit(f"no Android app at {args.android} (--android)")

    if args.check or args.version is None:
        found = versions(args.android)
        distinct = set(found.values())
        for rel, value in found.items():
            print(f"{value:>10}  {rel}")
        if len(distinct) != 1:
            sys.exit("the platforms' versions differ: scripts/bump_version.py <version>")
        print(f"in sync at {distinct.pop()}")
        return

    version = args.version
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        sys.exit(f"{version}: not MAJOR.MINOR.PATCH")
    code = code_of(version)
    for rel, pattern in DESKTOP:
        _write(ROOT / rel, pattern, version)
    _write(args.android / GRADLE, VERSION_NAME, version)
    _write(args.android / GRADLE, VERSION_CODE, str(code))
    print(f"{version} (Android versionCode {code}) in {len(DESKTOP)} desktop files and {GRADLE}")


if __name__ == "__main__":
    main()
