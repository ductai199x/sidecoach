"""Every judged moment of a parity round, the AI's reply beside its template, for reading.

  uv run --no-sync python tools/eval/pairs.py --run parity-pilot-<version>-r1 [--kind stuck]

Reads the run's judgments (data/eval/judgments/<run> and <run>-floor) and prints, per
judged moment: the class and level, whether each arm passed and why not, the AI's reply
and the template. No calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

JUDGMENTS = Path(__file__).resolve().parents[2] / "data" / "eval" / "judgments"


def _load(run: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted((JUDGMENTS / run).glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                out[record["id"]] = record
    return out


def _why(record: dict | None) -> str:
    if record is None:
        return "not judged"
    scores = record.get("scores") or {}
    faults: list[str] = []
    for kind, found in (record.get("deterministic") or {}).items():
        if isinstance(found, list) and found:
            faults.append(f"{kind}: {'; '.join(str(f) for f in found)}")
        elif isinstance(found, dict) and found.get("problems"):
            faults.append(f"{kind}: {'; '.join(str(f) for f in found['problems'])}")
    for claim in (record.get("verdict") or {}).get("claims") or []:
        if not claim.get("supported"):
            faults.append(f"unsupported: {claim.get('text', claim.get('n'))} ({claim.get('why')})")
    completeness = scores.get("completeness")
    head = "pass" if scores.get("pass") else "FAIL"
    tail = f" completeness {completeness:.2f}" if isinstance(completeness, float) else ""
    return f"{head}{tail}" + (" -- " + " | ".join(faults)[:400] if faults else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the AI arm's run name")
    ap.add_argument("--kind", default="", help="only moments whose id contains this")
    args = ap.parse_args()
    ai, floor = _load(args.run), _load(f"{args.run}-floor")
    n = 0
    for key in sorted(ai):
        if args.kind and args.kind not in key:
            continue
        a, t = ai[key], floor.get(key)
        n += 1
        print(f"\n=== {key}")
        print(f"  AI [{_why(a)}]: {a.get('reply', '')}")
        print(f"  TEMPLATE [{_why(t)}]: {(t or {}).get('reply', '') or '(none)'}")
    print(f"\n{n} moments")


if __name__ == "__main__":
    main()
