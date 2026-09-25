"""The code a result was made with, so a result from other code is never read as this code's.

`code_version()` is the short commit hash of HEAD, and, when the coach or the evaluation
tools differ from it (edited or new files under src/sidecoach or tools/eval), a digest of
that difference: "2219e11" or "2219e11-d4f09a1". Items and judgments carry it; a runner
puts it in its run names and refuses to reuse what another version made unless told to
(`--reuse-stale`). The coach changed under the same run name was the silent failure: the
cached replies and verdicts of the old coach, reported as the new one's (the eval audit,
2026-09-25).
"""

from __future__ import annotations

import functools
import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# What the numbers depend on: the coach, and the tools that play, judge and score it.
WATCHED = ("src/sidecoach", "tools/eval")


class StaleData(RuntimeError):
    """A cached result made by other code than the code running now."""


def _git(*args: str) -> bytes:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, timeout=60
    ).stdout


@functools.cache
def code_version() -> str:
    """The short commit hash, plus a digest of any uncommitted change to the watched code."""
    try:
        head = _git("rev-parse", "--short", "HEAD").decode().strip()
        diff = _git("diff", "HEAD", "--", *WATCHED)
        untracked = (
            _git("ls-files", "--others", "--exclude-standard", "--", *WATCHED).decode().split()
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if not diff and not untracked:
        return head
    digest = hashlib.sha1(diff)
    for name in sorted(untracked):
        digest.update(name.encode())
        path = ROOT / name
        if path.is_file():
            digest.update(path.read_bytes())
    return f"{head}-d{digest.hexdigest()[:7]}"


def check(found: str | None, what: str, *, reuse_stale: bool, current: str | None = None) -> None:
    """Raise StaleData when `what` was made by other code than `current` (the code running
    now, by default), unless `reuse_stale`."""
    want = current or code_version()
    if found == want or reuse_stale:
        return
    raise StaleData(
        f"{what} was made by code {found or 'of unknown version'}; the code now is {want}. "
        "Its replies or verdicts would be reported as this code's. Use a fresh run, or pass "
        "--reuse-stale to read it anyway."
    )
