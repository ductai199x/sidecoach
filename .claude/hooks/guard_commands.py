"""PreToolUse guard for Bash and PowerShell: refuse commands that have cost Tai data or time.

Blocked (exit 2, the reason goes back to Claude):
  - heredocs and here-strings (bash <<EOF, <<<, PowerShell @'...'@): they mangled backslashes
    in code (\\n, \\b) again and again -- write the file with the Write tool, then run it
  - pkill, pgrep, killall, taskkill /IM, Stop-Process -Name: stop processes by PID only
  - git commands that throw away work: checkout -- / checkout . / checkout -f, restore
    (without --staged), reset --hard, clean -f, stash drop/clear, push --force, branch -D,
    --no-verify
  - recursive deletes (rm -r, rmdir /s, rd /s, del /s, Remove-Item -Recurse, find -delete,
    find -exec rm) outside Claude's scratch/temp folders
  - robocopy /MOV /MOVE /MIR /PURGE: a looped robocopy /MOVE with a mangled path lost 16 of
    Tai's 17 game recordings (2026-09-21)
  - moving anything under a data/recordings/dumps/Sidecoach path from the shell (mv, move,
    Move-Item): moves of data go through a Write'd Python script, copy-verify-then-remove
  - deleting or moving files from an inline `python -c` (rmtree, remove, unlink, rename)
  - format, diskpart, Format-Volume, cipher /w
  - in Git Bash, an argument Git Bash would rewrite as a path before a Windows program sees
    it, without MSYS_NO_PATHCONV=1 on the command: an Android path (/sdcard/..., /data/...)
    became "C:/Program Files/Git/sdcard/..." and the push went nowhere (2026-10-04), and a
    Windows switch (/PID, /c, /s) becomes a Git path the same way. Tai: "you already
    catastrophically did it once" (the 2026-09-21 robocopy /MOVE with a mangled path)
  - the WHOLE test suite, desktop or phone (pytest with no test file named; the phone's
    tools/test.ps1 without -Class; connected*AndroidTest or am instrument without a class):
    Tai, 2026-10-01, the third time, "DO NOT RUN THE WHOLE FUCKING TEST SUITE". Not a block
    but a reminder to say yes to: only when Tai asked for the full suite does the command
    carry FULL_SUITE_ASKED=1, and then it runs.

Reads the hook JSON on stdin: {"tool_name": ..., "tool_input": {"command": ...}}.
"""

from __future__ import annotations

import json
import re
import shlex
import sys

SCRATCH = re.compile(r"(?i)(?:[\\/]Temp[\\/]claude[\\/]|^/tmp/|[\\/]scratchpad[\\/])")
DATA_PATH = re.compile(r"(?i)(?:\bdata\b|recordings|dumps|sidecoach[\\/]data|[\\/]Sidecoach\b)")

# Splits a command line into the commands it runs: ; && || | newlines, $( ( and backticks.
SEPARATORS = re.compile(r"\|\||&&|[;|\n`]|\$\(|(?<![\w$])\(")
PREFIXES = {"sudo", "nice", "timeout", "xargs", "command", "exec", "env", "time", "nohup"}


def segments(command: str) -> list[list[str]]:
    """Each command of the line as its words, the leading VAR=x, sudo, nice -n 19, timeout
    N and the like dropped, so the first word is the program."""
    out: list[list[str]] = []
    for part in SEPARATORS.split(command):
        try:
            words = shlex.split(part, posix=True)
        except ValueError:
            words = part.split()
        while words:
            first = words[0]
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=.*", first):
                words = words[1:]
            elif first in PREFIXES:
                words = words[1:]
                while words and (
                    words[0].startswith("-") or re.fullmatch(r"[\d.]+[smh]?", words[0])
                ):
                    words = words[1:]
            else:
                break
        if words:
            out.append(words)
    return out


def program(words: list[str]) -> str:
    name = words[0].replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name[:-4] if name.endswith(".exe") else name


def paths_of(words: list[str]) -> list[str]:
    return [w for w in words[1:] if not w.startswith("-") and not re.fullmatch(r"/[A-Za-z]+", w)]


def all_scratch(paths: list[str]) -> bool:
    return bool(paths) and all(SCRATCH.search(p) for p in paths)


FULL_SUITE = (
    "this runs the WHOLE {what} test suite. Tai's rule (2026-10-01, said three times): never "
    "the full suite unless he asked for it. Run only the test files (phone: the classes) that "
    "import what you changed. If Tai did ask for the full suite, say so and rerun with "
    "FULL_SUITE_ASKED=1 in the command."
)


def _targeted(args: list[str]) -> bool:
    """Whether pytest's arguments name a test file or node (tests/x.py, x.py::name): the
    tests folder alone is the whole suite."""
    return any(not a.startswith("-") and (".py" in a or "::" in a) for a in args)


def _base(word: str) -> str:
    name = word.replace("\\", "/").rsplit("/", 1)[-1].lower()
    return name[:-4] if name.endswith(".exe") else name


def full_suite(words: list[str]) -> str | None:
    """'desktop' or 'phone' when the command RUNS every test of that suite, else None. Only
    a word in the program's place counts -- pytest itself, `uv run pytest`, `python -m pytest`,
    the phone's test script run by PowerShell -- so reading or grepping those files is no run
    (a `sed -n` on test.ps1 was refused, 2026-10-01)."""
    low = [w.lower() for w in words]
    prog = _base(words[0])
    # Where the program's own arguments begin: after `uv run [options]`, after `python -m`.
    at = 0
    if prog == "uv" and len(low) > 1 and low[1] == "run":
        at = 2
        while at < len(low) and low[at].startswith("-"):
            at += 1
    if (
        at < len(low)
        and _base(words[at]).startswith("python")
        and low[at + 1 : at + 3]
        == [
            "-m",
            "pytest",
        ]
    ):
        return None if _targeted(words[at + 3 :]) else "desktop"
    if at < len(low) and _base(words[at]) in ("pytest", "py.test"):
        return None if _targeted(words[at + 1 :]) else "desktop"
    shell = prog in ("powershell", "pwsh", "&", ".") or prog.endswith("test.ps1")
    for i, w in enumerate(low):
        name = _base(w)
        if shell and name.endswith("test.ps1"):  # tools\test.ps1 reaches here as toolstest.ps1
            return None if "-class" in low[i + 1 :] else "phone"
        if prog.startswith("gradlew") and re.fullmatch(r":?(app:)?connected\w*androidtest", name):
            named = any("instrumentationrunnerarguments.class" in a for a in low)
            return None if named else "phone"
        if prog == "adb" and w == "instrument" and i > 0 and low[i - 1] == "am":
            return None if "class" in low[i + 1 :] else "phone"
    return None


# An argument that starts with one of these is a path on the phone, never on this machine.
DEVICE_PATH = re.compile(
    r"(?:^|(?<=[\s\"'=]))/(?:sdcard|storage|data|system|vendor|apex|mnt|odm|product|cache)(?=/|\s|$|[\"'])"
)
# Windows programs whose switches start with a slash: Git Bash turns a switch into a path.
SLASH_SWITCHED = {
    "cmd",
    "taskkill",
    "tasklist",
    "robocopy",
    "xcopy",
    "schtasks",
    "reg",
    "sc",
    "net",
    "wmic",
    "icacls",
    "findstr",
    "where",
    "shutdown",
    "attrib",
    "msiexec",
    "certutil",
}
MANGLED = (
    "Git Bash would rewrite {what} as a Windows path before the program sees it (MSYS path "
    "conversion: /sdcard/x becomes C:/Program Files/Git/sdcard/x, /PID becomes "
    "C:/Program Files/Git/PID). Put MSYS_NO_PATHCONV=1 on the command (export "
    "MSYS_NO_PATHCONV=1; ... or MSYS_NO_PATHCONV=1 cmd ...), and give this machine's paths "
    "as D:/... (they are no longer converted then), or use the PowerShell tool."
)


def mangling(command: str) -> str | None:
    """The reason a Git Bash command would hand a mangled path to a Windows program, or None."""
    if "MSYS_NO_PATHCONV=1" in command or "MSYS2_ARG_CONV_EXCL" in command:
        return None
    found = DEVICE_PATH.search(command)
    if found:
        return MANGLED.format(what=f"the Android path {found.group(0).strip()!r}")
    for words in segments(command):
        if program(words) in SLASH_SWITCHED:
            switch = next((w for w in words[1:] if re.fullmatch(r"/[A-Za-z?][\w:]*", w)), None)
            if switch:
                return MANGLED.format(what=f"the switch {switch!r} of {program(words)}")
    return None


def check(command: str, tool: str = "Bash") -> str | None:
    """The reason to refuse `command`, or None."""
    if tool == "Bash":
        reason = mangling(command)
        if reason:
            return reason
    if "FULL_SUITE_ASKED=1" not in command:
        for words in segments(command):
            what = full_suite(words)
            if what:
                return FULL_SUITE.format(what=what)
    # Heredocs and here-strings, anywhere on the line (not a shift inside $(( ))).
    if re.search(r"(?<!<)<<(?!<)-?\s*['\"]?[A-Za-z_]", command) or "<<<" in command:
        return (
            "heredocs are banned (Tai, 2026-09-25): they mangle backslashes in code. Write the "
            "file with the Write tool, then run it; for a commit message, `git commit -F <file>`."
        )
    if re.search(r"@['\"][ \t]*\r?\n", command):
        return (
            "PowerShell here-strings are banned like heredocs: write the file with the Write "
            "tool, then run it; for a commit message, `git commit -F <file>`."
        )
    for words in segments(command):
        name = program(words)
        low = [w.lower() for w in words]
        rest = " ".join(low[1:])
        if name == "sed" and any(w.startswith("-i") or w == "--in-place" for w in low[1:]):
            return (
                "sed -i is banned: it rewrites files without a look (and leaves .bak files on "
                "Windows). Use the Edit tool."
            )
        if name in ("pkill", "pgrep", "killall"):
            return f"{name} is banned: find the PID (ps, tasklist, Get-Process) and stop that PID."
        if name == "taskkill" and "/im" in low:
            return "taskkill /IM is banned: stop the process by /PID."
        if name in ("stop-process", "spps") and any(w.startswith("-n") for w in low[1:]):
            return "Stop-Process -Name is banned: stop the process by -Id."
        if name == "git" and len(low) > 1:
            sub = low[1]
            if sub == "checkout" and (
                "--" in low[2:] or "." in low[2:] or "-f" in low[2:] or "--force" in low[2:]
            ):
                return "git checkout that discards changes is banned: ask Tai, or stash first."
            if sub == "restore" and "--staged" not in low and "-s" not in low[2:]:
                return "git restore of the working tree is banned: it discards changes."
            if sub == "reset" and "--hard" in low:
                return "git reset --hard is banned: it discards work."
            if sub == "clean" and any(w.startswith("-") and "f" in w for w in low[2:]):
                return "git clean -f is banned: it deletes untracked files (data/ is untracked)."
            if sub == "stash" and len(low) > 2 and low[2] in ("drop", "clear"):
                return "git stash drop/clear is banned: it throws away stashed work."
            if sub == "push" and any(w in ("-f", "--force", "--force-with-lease") for w in low):
                return "git push --force is banned."
            if sub == "branch" and "-d" in low[2:] and "-D" in words[2:]:
                return "git branch -D is banned: use -d, which refuses unmerged work."
            if "--no-verify" in low:
                return "--no-verify is banned: never skip hooks unless Tai asks."
        if name in ("rm", "remove-item", "ri", "del", "erase", "rmdir", "rd"):
            recursive = any(
                re.fullmatch(r"-[a-z]*r[a-z]*", w) or w in ("--recursive", "-recurse", "/s")
                for w in low[1:]
            )
            if recursive and not all_scratch(paths_of(words)):
                return (
                    "recursive deletes are banned outside the scratch folder: delete through a "
                    "Write'd Python script that lists what it removes, one item first."
                )
        if name == "find" and (
            "-delete" in low or ("-exec" in low and re.search(r"-exec\s+rm\b", rest))
        ):
            return "find -delete / -exec rm is banned: list the files, then delete by script."
        if name == "robocopy" and any(w in ("/mov", "/move", "/mir", "/purge") for w in low):
            return (
                "robocopy /MOV /MOVE /MIR /PURGE is banned: it lost 16 of Tai's 17 recordings. "
                "Copy, verify, then remove, by a Write'd Python script."
            )
        if name in ("mv", "move", "move-item", "mi") and any(
            DATA_PATH.search(p) for p in paths_of(words)
        ):
            return (
                "moving data from the shell is banned: use a Write'd Python script that copies, "
                "verifies, then removes, one item first."
            )
        if name in ("format", "diskpart", "format-volume", "clear-disk") or (
            name == "cipher" and "/w" in low
        ):
            return f"{name} is banned."
        python = name.startswith("python") or (name == "uv" and "python" in low)
        if (
            python
            and "-c" in low
            and re.search(
                r"rmtree|os\.remove|os\.unlink|\.unlink\(|os\.rmdir|\.rmdir\(|os\.rename|"
                r"shutil\.move|\.replace\(\s*['\"]?[A-Za-z]:|send2trash",
                command,
            )
        ):
            return (
                "deleting or moving files from inline python -c is banned: Write a script "
                "that copies, verifies, then removes, one item first."
            )
    return None


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (ValueError, OSError):
        return
    command = (payload.get("tool_input") or {}).get("command")
    if not isinstance(command, str):
        return
    reason = check(command, str(payload.get("tool_name") or "Bash"))
    if reason:
        print(f"Blocked by guard_commands.py: {reason}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
