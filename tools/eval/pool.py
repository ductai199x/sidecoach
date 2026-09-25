"""The demonstrations an ICR run has collected (its C before distillation), how a judged
moment is ranked for correction, and the words the writer must not use."""

from __future__ import annotations

import json
import re
from pathlib import Path

from sidecoach.coach import prompt

# The sheet's own words: a demonstration or a lesson that uses them teaches the coach to
# speak of "the verdict" or "expected score" to the player.
JARGON = re.compile(
    r"\b(?:expected[- ]score|expected result|evidence|verdict|label(?:led)?|centipawns?|"
    r"disclosure|fact sheet|the sheet|required facts?)\b",
    re.IGNORECASE,
)

C_HEADER = (
    "Demonstrations from earlier games. Each is a moment as it reached you -- the game so "
    "far, the position, the task -- and the reply that was right for it. They show the "
    "manner: what such a reply states, how directly, from which facts of its own message. "
    "They are other games: never reuse their moves, names or facts. Every fact in your "
    "reply comes from the game in front of you.\n"
)


def failure_score(j: dict) -> float:
    """How severely the reply fabricated or omitted, the paper's ranking: an unsupported
    claim, a leak or a false board claim first; then what was left out; then what was
    stated wrong. A master's point the coach's own facts stated and the reply missed is
    an omission; one the facts never stated is not the coach's fault."""
    s = j["scores"]
    score = 0.0
    if s["fabrication"] or s["leak"] or s["board"]:
        score += 3.0 + 0.5 * len(s.get("unsupported") or [])
    if j.get("fallback_used"):
        score += 2.0
    if s["completeness"] is not None:
        score += 2.0 * (1.0 - s["completeness"])
    if s["correctness"] is not None and s["correctness"] < 1.0:
        score += 1.0
    if s["relevance"] is not None and s["relevance"] < 0.5:
        score += 1.0
    if s.get("idea_match") is False and s.get("idea_in_facts"):
        score += 1.0
    if not s["form_ok"]:
        score += 0.5
    return score


def facts_gap(j: dict) -> bool:
    """The master's point was missed and the facts never stated it: a gap in what the
    coach is given, for the facts layer to close, not for C."""
    s = j["scores"]
    return s.get("idea_match") is False and s.get("idea_in_facts") is False


def render_demo(n: int, d: dict) -> str:
    """One demonstration in C: the input as the coach saw it (its lean form when one was
    rendered), then the right reply."""
    return "\n".join(
        [f"### {n}", d.get("lean_input") or d["input"], "", f'Reply: "{d["reply"]}"', ""]
    )


class Pool:
    """The demonstrations of a run, kept in a JSON file."""

    def __init__(self, demos: list[dict], path: Path) -> None:
        self.demos = demos
        self.path = path

    @classmethod
    def load(cls, path: Path) -> Pool:
        demos = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        return cls(demos, path)

    def save(self) -> None:
        self.path.write_text(json.dumps(self.demos, indent=1, ensure_ascii=False), encoding="utf-8")

    def text(self) -> str:
        return "".join(render_demo(k, d) + "\n" for k, d in enumerate(self.demos, start=1))

    def tokens(self) -> int:
        return int(len(self.text()) / 3.6)

    def system_text(
        self, *, persona: str | None = None, rating: tuple[int, int] | None = None
    ) -> str:
        """The system message with all of C: the coach's prompt (the app's persona and
        rating) without the shipped lessons, then the demonstrations."""
        base = prompt.system_message(with_corrections=False, persona=persona, rating=rating)
        c = self.text()
        return base + ("\n\n" + C_HEADER + c if c else "")
