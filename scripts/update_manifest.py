"""The updater's `latest.json` for a release: each desktop file signed, then listed per platform.

    uv run --no-sync python scripts/update_manifest.py 0.1.90 DIR --notes NOTES.txt

DIR holds the release's desktop files as they will be uploaded (Sidecoach_<version>_x64-setup.exe,
_amd64.AppImage, _amd64.deb). Each is signed with the updater's key (TAURI_SIGNING_PRIVATE_KEY_PATH,
else ~/.tauri/sidecoach.key; its password in TAURI_SIGNING_PRIVATE_KEY_PASSWORD, empty by default),
the signature written beside it as <file>.sig, and DIR/latest.json made of them. Upload latest.json
with the files: the installed apps read it from the latest release (tauri.conf.json,
plugins.updater) and offer the update after a yes (docs/app.md, "The updater").

Signing the files as uploaded, not inside the build, is what makes the signatures fit: the
AppImage is repacked after Tauri builds it, and the Linux files are built in WSL, away from the key.
NOTES is the text the update's question shows (a few plain lines; the page shows 600 characters).
--base-url replaces the release's download address (a test server).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DESKTOP = ROOT / "desktop"
REPO = "ductai199x/sidecoach"

# The updater asks for "{os}-{arch}-{installer}" first, then "{os}-{arch}" (tauri-plugin-updater's
# get_urls). Windows has the one installer, so it gets both names; Linux only the per-installer
# names, so an AppImage is never offered a .deb nor a .deb an AppImage.
PLATFORMS = {
    "_x64-setup.exe": ["windows-x86_64-nsis", "windows-x86_64"],
    "_amd64.AppImage": ["linux-x86_64-appimage"],
    "_amd64.deb": ["linux-x86_64-deb"],
}


def _key() -> Path:
    named = os.environ.get("TAURI_SIGNING_PRIVATE_KEY_PATH")
    key = Path(named) if named else Path.home() / ".tauri" / "sidecoach.key"
    if not key.is_file():
        sys.exit(f"no updater key at {key}: set TAURI_SIGNING_PRIVATE_KEY_PATH")
    return key


def _sign(path: Path, key: Path) -> str:
    """The updater signature of `path` (minisign, by the Tauri CLI), also left as <path>.sig."""
    # node on the CLI's own script, not npm: npm's shell on Windows drops an empty argument, and
    # an empty password would take the file's name as the password.
    node = shutil.which("node")
    cli = DESKTOP / "node_modules" / "@tauri-apps" / "cli" / "tauri.js"
    if not node or not cli.is_file():
        sys.exit("node and desktop/node_modules (npm ci) are needed: the Tauri CLI signs the files")
    signature = path.with_name(path.name + ".sig")
    signature.unlink(missing_ok=True)
    password = os.environ.get("TAURI_SIGNING_PRIVATE_KEY_PASSWORD", "")
    env = {k: v for k, v in os.environ.items() if not k.startswith("TAURI_SIGNING_PRIVATE_KEY")}
    done = subprocess.run(
        [node, str(cli), "signer", "sign", "-f", str(key), "-p", password, str(path)],
        cwd=DESKTOP,
        env=env,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0 or not signature.is_file():
        sys.exit(f"{path.name}: not signed\n{done.stdout}{done.stderr}")
    return signature.read_text(encoding="utf-8").strip()


def manifest(
    version: str, files: dict[str, str], notes: str, base_url: str, when: datetime
) -> dict:
    """latest.json: `files` maps each file's name to its signature."""
    platforms: dict[str, dict[str, str]] = {}
    for name, signature in sorted(files.items()):
        for suffix, targets in PLATFORMS.items():
            if name == f"Sidecoach_{version}{suffix}":
                for target in targets:
                    platforms[target] = {
                        "signature": signature,
                        "url": f"{base_url.rstrip('/')}/{name}",
                    }
    return {
        "version": version,
        "notes": notes,
        "pub_date": when.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platforms": dict(sorted(platforms.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version")
    parser.add_argument("dir", type=Path)
    parser.add_argument("--notes", type=Path, required=True)
    parser.add_argument("--base-url")
    args = parser.parse_args()

    names = [f"Sidecoach_{args.version}{suffix}" for suffix in PLATFORMS]
    present = [name for name in names if (args.dir / name).is_file()]
    if not present:
        sys.exit(f"none of {', '.join(names)} in {args.dir}")
    for name in sorted(set(names) - set(present)):
        print(f"not in {args.dir}, not offered: {name}")
    key = _key()
    files = {}
    for name in present:
        files[name] = _sign(args.dir / name, key)
        print(f"signed {name}")
    base_url = args.base_url or f"https://github.com/{REPO}/releases/download/v{args.version}"
    notes = args.notes.read_text(encoding="utf-8").strip()
    out = args.dir / "latest.json"
    data = manifest(args.version, files, notes, base_url, datetime.now(UTC))
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{out}: {', '.join(data['platforms'])}")


if __name__ == "__main__":
    main()
