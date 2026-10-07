"""Copy the desktop panel's built page into the app's assets.

The Android app serves the desktop's own frontend (`src/sidecoach/ui/web`, the SvelteKit
build) from a local HTTP server, so what runs on the phone is the same page that runs on a
computer -- including its `stacked` layout, which its stylesheet describes as being for "a
phone, a tablet held upright, a narrow window".

Run this whenever the frontend is rebuilt (`npm run build` in `desktop/`, which writes into
`src/sidecoach/ui/web`).

    cd D:/chess-coaching-companion
    uv run --no-sync python android/tools/sync_web.py
"""

from __future__ import annotations

import filecmp
import shutil
from pathlib import Path

WEB = (Path(__file__).resolve().parents[2] / "src/sidecoach/ui/web")
OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "main" / "assets" / "web"


def main() -> None:
    if not WEB.is_dir():
        raise SystemExit(f"no built page at {WEB}")
    before = {p.relative_to(OUT) for p in OUT.rglob("*") if p.is_file()} if OUT.is_dir() else set()
    OUT.mkdir(parents=True, exist_ok=True)
    after: set[Path] = set()
    copied = 0
    for source in sorted(WEB.rglob("*")):
        if not source.is_file():
            continue
        relative = source.relative_to(WEB)
        after.add(relative)
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and filecmp.cmp(source, target, shallow=False):
            continue
        shutil.copy2(source, target)
        copied += 1
    # The build renames its chunks, so last build's files would pile up.
    for gone in sorted(before - after):
        (OUT / gone).unlink()
        print(f"removed {gone}")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{len(after)} files ({total / 1e3:.0f} kB), {copied} copied -> {OUT}")


if __name__ == "__main__":
    main()
