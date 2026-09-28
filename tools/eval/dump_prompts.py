"""The requests the AI coach would send, written out and never sent: parity pass 2.

Every game given is played through the real coach (`playthrough.play`), and each request
the Speaker receives is recorded -- the trigger, the level, whether it is a routine word,
what the reply must name, the template that stands in when the model fails, and the whole
message the model would read -- then answered by the template, as with no model, so the
game goes on exactly as the offline coach plays it. Nothing is sent anywhere and nothing
costs money: the point is to read each request as the model would, and to size them.

One markdown file per game and side under --out, with the system prompt once in
`system.md`, and a summary: requests by trigger, prompt sizes, and the share a prompt cache
would serve (the prefix each request shares with the one before it in the same game).

  uv run --no-sync python tools/eval/dump_prompts.py --bots --limit 4 --out data/eval/prompts/p2
  uv run --no-sync python tools/eval/dump_prompts.py --side white \
      --game mammoth__080-kasparov-karpov
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from playthrough import FACTS, play  # noqa: E402

# Roughly four characters a token for English and chess notation: enough to size a run.
CHARS_PER_TOKEN = 4


@dataclass
class Recorded:
    ply: int
    trigger: str
    level: str
    routine: bool
    template_only: bool
    must_name: tuple[str, ...]
    fallback: str
    system: str
    user: str


def _common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def record_game(game: str, side: str) -> tuple[list[Recorded], list[str]]:
    """The requests of one game, coached from `side`, and its transcript."""
    recorded: list[Recorded] = []

    def hook(coach) -> None:  # noqa: ANN001
        speak = coach.speaker.speak_now

        def recording(request):  # noqa: ANN001, ANN202
            recorded.append(
                Recorded(
                    ply=request.ply,
                    trigger=request.trigger.value,
                    level=request.level.name,
                    routine=request.routine,
                    template_only=request.template_only,
                    must_name=tuple(request.must_name),
                    fallback=request.fallback,
                    system=request.system,
                    user=str(request.messages[-1]["content"]),
                )
            )
            # The template stands in, as with no model: nothing is sent.
            request.template_only = True
            return speak(request)

        coach.speaker.speak_now = recording

    transcript, _, _ = play(game, side, "none", on_coach=hook)
    return recorded, transcript


def write_game(out: Path, game: str, side: str, requests: list[Recorded]) -> None:
    lines = [f"# {game}, coaching {side}: {len(requests)} requests\n"]
    for i, r in enumerate(requests, start=1):
        flags = ", ".join(
            x
            for x in (
                "routine" if r.routine else "",
                "template only" if r.template_only else "",
                f"must name one of {', '.join(r.must_name)}" if r.must_name else "",
            )
            if x
        )
        head = f"\n## {i}. ply {r.ply}: {r.trigger}, level {r.level}"
        lines.append(head + (f" ({flags})" if flags else ""))
        lines.append(f"\n**Template (the fallback):** {r.fallback or '(none)'}\n")
        lines.append(f"~{len(r.user) // CHARS_PER_TOKEN} tokens in the message\n")
        lines.append("```text\n" + r.user + "\n```")
    (out / f"{game}-{side}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", action="append", help="facts stems (masters unless --bots)")
    ap.add_argument("--bots", action="store_true", help="the Nova bot games")
    ap.add_argument("--side", choices=["white", "black"], help="one side only")
    ap.add_argument("--limit", type=int, default=0, help="the first N games only")
    ap.add_argument("--out", type=Path, default=Path("data/eval/prompts/latest"))
    args = ap.parse_args()

    folder = FACTS.parent / "bots" if args.bots else FACTS
    games = args.game or sorted(p.stem for p in folder.glob("*.json"))
    if args.limit:
        games = games[: args.limit]
    sides = [args.side] if args.side else ["white", "black"]
    args.out.mkdir(parents=True, exist_ok=True)

    by_trigger: Counter[str] = Counter()
    sizes: list[int] = []
    cached = 0
    total = 0
    system = ""
    for game in games:
        for side in sides:
            requests, _ = record_game(game, side)
            write_game(args.out, game, side, requests)
            previous = ""
            for r in requests:
                system = r.system
                by_trigger[r.trigger + (" (routine)" if r.routine else "")] += 1
                text = r.system + "\n" + r.user
                sizes.append(len(text) // CHARS_PER_TOKEN)
                cached += _common_prefix(previous, text) // CHARS_PER_TOKEN
                total += len(text) // CHARS_PER_TOKEN
                previous = text
    (args.out / "system.md").write_text(system + "\n", encoding="utf-8")

    n = len(sizes)
    print(f"{len(games)} games x {len(sides)} sides: {n} requests -> {args.out}{os.sep}")
    for trigger, count in by_trigger.most_common():
        print(f"  {count:5d}  {trigger}")
    if n:
        sizes.sort()
        print(
            f"prompt tokens per request (~): median {sizes[n // 2]}, p90 {sizes[n * 9 // 10]}, "
            f"max {sizes[-1]}; system prompt ~{len(system) // CHARS_PER_TOKEN}"
        )
        print(f"cacheable share (prefix shared with the request before): {cached / total:.0%}")


if __name__ == "__main__":
    main()
