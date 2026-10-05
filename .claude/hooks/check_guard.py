"""The command guard checked on commands whose answer is known: what it must refuse, what it
must let through. Run after any change to guard_commands.py:

    uv run --no-sync python .claude/hooks/check_guard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from guard_commands import check  # noqa: E402

REFUSED = [
    # Git Bash path conversion
    ("Bash", "adb -s X push net.nnue /sdcard/Android/data/com.sidecoach.apart1/files/"),
    ("Bash", '"$A" -s $S shell ls /sdcard/Android/data'),
    ("Bash", "adb pull /data/local/tmp/x.json out.json"),
    ("Bash", "taskkill /PID 1234 /F"),
    ("Bash", "cmd /c dir"),
    ("Bash", "robocopy a b /E"),
    # the rules from before
    ("Bash", "sed -i 's/a/b/' file.txt"),
    ("Bash", "git reset --hard HEAD"),
    ("Bash", "rm -rf data/engine-cache"),
    ("Bash", "cat <<EOF > x\nhi\nEOF"),
    ("Bash", "uv run pytest"),
]
ALLOWED = [
    ("Bash", "MSYS_NO_PATHCONV=1 adb push net.nnue /sdcard/Android/data/x/files/"),
    ("Bash", 'export MSYS_NO_PATHCONV=1; "$A" -s $S shell "ls /sdcard/x"'),
    ("Bash", "ls /d/chess-coaching-companion/data/engine-cache"),
    ("Bash", "tail -3 /d/chess-coaching-companion/data/engine-cache/phone-atoms.log"),
    ("Bash", "cat data/engine-cache/x.log | tail -2"),
    ("Bash", "grep -n x /tmp/b.log"),
    ("Bash", "tasklist | head"),
    ("Bash", "uv run pytest tests/test_analyser.py -q"),
    ("PowerShell", "adb push net.nnue /sdcard/Android/data/x/files/"),
    ("PowerShell", "taskkill /PID 1234"),
]


def main() -> None:
    wrong = 0
    for tool, command in REFUSED:
        if check(command, tool) is None:
            wrong += 1
            print(f"LET THROUGH, should refuse: [{tool}] {command!r}")
    for tool, command in ALLOWED:
        reason = check(command, tool)
        if reason is not None:
            wrong += 1
            print(f"REFUSED, should let through: [{tool}] {command!r}: {reason[:80]}")
    print(f"{len(REFUSED) + len(ALLOWED) - wrong} of {len(REFUSED) + len(ALLOWED)} as they should be")
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
