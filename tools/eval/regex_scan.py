"""Every regex of the eval tools and the coach package -- module-level patterns and inline
literals alike -- run against a real 12k-char harness, a long reply and a long record
line, each in its own process with a timeout. A pattern that backtracks catastrophically
shows up as a timeout. Run it before a paid run: one such pattern froze the judge for an
afternoon (`re` holds the GIL, so one stuck thread stops every worker).

  uv run --no-sync python tools/eval/regex_scan.py [--harness path/to/items.jsonl]
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import glob
import json
import multiprocessing as mp
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FILES = sorted(
    glob.glob(str(ROOT / "tools" / "eval" / "*.py"))
    + glob.glob(str(ROOT / "src" / "sidecoach" / "coach" / "**" / "*.py"), recursive=True)
)
MODULES = (
    "judge",
    "corpus",
    "taxonomy",
    "icr_play",
    "pool",
    "providers",
    "sidecoach.coach.coach",
    "sidecoach.coach.claims",
    "sidecoach.coach.disclosure",
    "sidecoach.coach.conversation",
    "sidecoach.coach.person",
    "sidecoach.coach.facts",
)
TIMEOUT_S = 3.0


def module_patterns() -> list[tuple[str, str, int]]:
    import importlib

    out = []
    for name in MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {name}: {exc}")
            continue
        for attr, val in vars(mod).items():
            if isinstance(val, re.Pattern):
                out.append((f"{name}.{attr}", val.pattern, val.flags))
    return out


def inline_patterns() -> list[tuple[str, str, int]]:
    """Every `re.<fn>(r"...")` literal in the files, with its flags when they are a plain
    `re.X | re.Y` expression."""
    out = []
    for path in FILES:
        tree = ast.parse(Path(path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "re"
                and node.args
            ):
                continue
            first = node.args[0]
            if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                continue
            flags = 0
            for arg in [*node.args[1:], *[k.value for k in node.keywords if k.arg == "flags"]]:
                with contextlib.suppress(Exception):
                    flags |= eval(compile(ast.Expression(arg), "<flags>", "eval"), {"re": re})  # noqa: S307
            out.append((f"{Path(path).name}:{node.lineno}", first.value, flags))
    return out


def _run(pattern: str, flags: int, texts: list[str], q) -> None:
    try:
        rx = re.compile(pattern, flags)
    except re.error as exc:
        q.put(("error", str(exc)))
        return
    t0 = time.time()
    for text in texts:
        rx.findall(text)
        rx.search(text)
    q.put(("ok", time.time() - t0))


def sample_texts(harness_path: str | None) -> list[str]:
    harness = ""
    path = harness_path
    if not path:
        found = sorted(glob.glob(str(ROOT / "data" / "eval" / "icr-100" / "items-r1" / "*.jsonl")))
        path = found[0] if found else None
    if path:
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        harness = str(rows[-1]["messages"][-1]["content"])
    record_line = (
        "12. Nf3 (the player): best, cost 0.00 -- the engine preferred Nf3 and it was played; "
        + "developing a piece toward the centre and keeping the tension in the middle of the"
        " board for now " * 4
    )
    reply = (
        "Your bishop on e2 attacks the knight on c6 and pins it to the rook on a8; after Bxf3 "
        "the line continues Nxd4 Qxd4 and you are up a rook. What would you play now? " * 6
    )
    return [t for t in (harness, harness * 3, record_line, reply) if t]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", help="a jsonl of items whose last harness is the sample text")
    args = ap.parse_args()
    texts = sample_texts(args.harness)
    patterns = module_patterns() + inline_patterns()
    print(
        f"{len(patterns)} patterns against {len(texts)} texts"
        f" ({max(len(t) for t in texts):,} chars max)"
    )
    slow: list[str] = []
    for name, pattern, flags in patterns:
        q: mp.Queue = mp.Queue()
        proc = mp.Process(target=_run, args=(pattern, flags, texts, q))
        proc.start()
        proc.join(TIMEOUT_S)
        if proc.is_alive():
            proc.terminate()
            slow.append(name)
            print(f"TIMEOUT {name}: {pattern[:100]}")
            continue
        kind, value = q.get()
        if kind == "error":
            print(f"compile error {name}: {value}")
        elif value > 0.3:
            print(f"slow {value:.2f}s {name}: {pattern[:100]}")
    print("timeouts:", slow or "none")
    sys.exit(1 if slow else 0)


if __name__ == "__main__":
    main()
