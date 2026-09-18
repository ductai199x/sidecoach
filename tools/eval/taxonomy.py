"""What can be asked of the coach, and what a good answer must contain.

Every item of the evaluation set belongs to one class here. A class says who starts it (the
coach, on a trigger, or the player, with a question), what the answer must contain -- the
required facts, each one something the messages state, so the judge can score it -- and the
form the answer must keep. The required facts depend on the disclosure level in force for
the position the player faces: below CANDIDATE the answer to "what should I play" is an
orienting question, not the move, and the judge scores it as such.

The question templates are what a player types or says; spoken forms come through STT, so
they are plain words, no notation. A class with a named move draws the move from the engine's
lines (best, near-best, weak) so its verdict is known.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Any

import chess

from sidecoach.coach.disclosure import PositionFacts
from sidecoach.events import DisclosureLevel


@dataclass
class Fact:
    """One thing a good answer states. `key` names it, `value` is the truth in the words
    the messages use, `how` says what counts as stating it."""

    key: str
    value: str
    how: str = "named"  # named | sense | words

    def as_dict(self) -> dict[str, str]:
        return {"key": self.key, "value": self.value, "how": self.how}


@dataclass
class Moment:
    """What the item's builder knows at the moment the coach speaks."""

    facts: PositionFacts  # the position the coach speaks about
    level: DisclosureLevel  # in force for the position the player faces
    board: chess.Board
    user_to_move: bool
    # The last move played and its assessment, when the moment follows a move.
    last_san: str | None = None
    last_by_user: bool | None = None
    last_label: str | None = None
    last_loss: float | None = None
    last_best_san: str | None = None
    # The engine's lines for the position now, as (san, expected score from the user's side).
    lines: list[tuple[str, float]] = field(default_factory=list)
    # Where each of those lines leads, by first move, as the harness says it ("the player
    # wins a pawn; the e-file opens"); and where the line the engine preferred to the last
    # move led.
    leads: dict[str, str] = field(default_factory=dict)
    last_best_leads: str | None = None
    # The player's moves so far with their labels, as the record shows them.
    record: list[str] = field(default_factory=list)
    # The player's record so far: label counts and the moves where ground was lost.
    tally: dict[str, int] = field(default_factory=dict)
    lost: list[str] = field(default_factory=list)
    opening: str | None = None
    material: str | None = None
    # Who stands better, as the harness states it: "It is about equal (Stockfish: White
    # ahead by about 0.7 pawns)."
    standing: str | None = None
    result: str | None = None
    # A move named in the question (class ask_named_move), with its rank and loss.
    named_san: str | None = None
    named_rank: int | None = None
    named_loss: float | None = None
    # The move before the last one (the opponent's, when the last was the player's), and
    # the position the last move was played from.
    previous_san: str | None = None
    fen_before_last: str | None = None
    # How the material went over the game: "level until 11. Bxe8, then up the exchange".
    material_arc: str | None = None
    # The facts of the position the player's last move was played from, when the item
    # speaks after that move from the position it led to (a question about the move):
    # the idea behind the better move lives there, not in the opponent's position.
    facts_before: PositionFacts | None = None


@dataclass
class Class:
    id: str
    kind: str  # "coach" | "ask"
    about: str
    questions: list[str] = field(default_factory=list)
    # The trigger that starts a coach-initiated class (Trigger value), else None.
    trigger: str | None = None
    # Form constraints the judge checks.
    max_sentences: int | None = None
    ends_with_question: bool | None = None
    no_names_below: DisclosureLevel | None = DisclosureLevel.CANDIDATE


def _loss_words(loss: float) -> str:
    """The cost of a move as the reply should convey it, with the honest number: a 0.06
    loss was required as "about a tenth", and the writer wrote "6%, roughly a tenth"."""
    if loss < 0.005:
        return "nothing"
    pct = f"about {round(loss * 100)}% of the game"
    if loss < 0.05:
        return f"a little ({pct})"
    if loss < 0.15:
        return f"a good deal ({pct})"
    if loss < 0.35:
        return f"a lot ({pct})"
    return f"the game, or close to it ({pct})"


def _plural(label: str, n: int) -> str:
    """'1 miss', '2 misses', '3 mistakes'."""
    if n == 1:
        return label
    return label + ("es" if label == "miss" else "s")


def _eval_words(score: float) -> str:
    if score >= 0.85:
        return "winning"
    if score >= 0.62:
        return "better"
    if score > 0.38:
        return "about equal"
    if score > 0.15:
        return "worse"
    return "losing"


def _level_fact(m: Moment) -> list[Fact]:
    """The one fact the level allows about the position the player faces."""
    f = m.facts
    if m.level <= DisclosureLevel.QUESTION:
        return [Fact("nature", f.nature or "", "sense")] if f.nature else []
    if m.level == DisclosureLevel.THEME:
        return [Fact("theme", f.theme or f.plan or "", "sense")]
    if m.level == DisclosureLevel.REGION:
        return [
            Fact("theme", f.theme or f.plan or "", "sense"),
            Fact("region", f.region or "", "named"),
        ]
    if m.level == DisclosureLevel.PIECE:
        return [Fact("piece", f.piece or "", "named")]
    if m.level == DisclosureLevel.CANDIDATE:
        return [Fact("candidate", f.candidates[0], "named")] if f.candidates else []
    return [Fact("line", " ".join(f.line), "named")] if f.line else []


def required(cls: Class, m: Moment) -> list[Fact]:
    """What the answer must contain, given the moment and the level in force."""
    f = m.facts
    out: list[Fact] = []
    if cls.id in ("verdict", "verdict_weaker"):
        # The move just played need not be named in a one-line nod; its sense must be right.
        if cls.id == "verdict_weaker":
            out.append(Fact("move", m.last_san or "", "named"))
        out.append(Fact("label", m.last_label or "", "sense"))
        if cls.id == "verdict_weaker" and m.last_best_san:
            out.append(Fact("better_move", m.last_best_san, "named"))
    elif cls.id in ("lesson", "ask_last_move"):
        out.append(Fact("cost", _loss_words(m.last_loss or 0.0), "words"))
        if m.last_best_san:
            out.append(Fact("better_move", m.last_best_san, "named"))
        # The idea is the one of the position the move was played from: for a lesson that
        # is the item's own facts; for a question asked after the move, facts_before.
        fb = m.facts_before if cls.id == "ask_last_move" and m.facts_before is not None else f
        idea = fb.theme or fb.plan or (fb.ideas[0] if fb.ideas else "")
        if idea:
            out.append(Fact("idea", idea, "sense"))
        # At full disclosure the tactic itself -- what forks or pins what -- is the lesson.
        if fb.tactic and m.level >= DisclosureLevel.PIECE:
            out.append(Fact("tactic", fb.tactic, "sense"))
    elif cls.id in ("decisive_question", "stuck_hint", "ask_hint"):
        out.extend(_level_fact(m) if cls.id != "decisive_question" else [])
    elif cls.id == "ask_move" or cls.id == "ask_line":
        out.extend(_level_fact(m))
    elif cls.id == "ask_plan":
        if m.level >= DisclosureLevel.THEME and (f.plan or f.theme):
            out.append(Fact("plan", f.plan or f.theme or "", "sense"))
        else:
            out.extend(_level_fact(m))
    elif cls.id == "ask_eval":
        if m.lines:
            out.append(Fact("eval", _eval_words(m.lines[0][1]), "sense"))
        if m.material:
            out.append(Fact("material", m.material, "sense"))
    elif cls.id == "ask_threat":
        if m.level >= DisclosureLevel.CANDIDATE and f.threats:
            out.extend(Fact(f"threat_{i}", t, "named") for i, t in enumerate(f.threats[:3], 1))
        else:
            out.extend(_level_fact(m))
    elif cls.id == "ask_named_move":
        if m.named_san:
            if m.named_rank == 1:
                out.append(Fact("named_verdict", f"{m.named_san} is the best move", "sense"))
            elif m.named_rank is not None and (m.named_loss or 0) < 0.05:
                out.append(
                    Fact("named_verdict", f"{m.named_san} is fine, close to the best", "sense")
                )
            else:
                out.append(
                    Fact(
                        "named_verdict",
                        f"{m.named_san} is not the strongest; it loses"
                        f" {_loss_words(m.named_loss or 0.3)}",
                        "sense",
                    )
                )
            if m.level >= DisclosureLevel.CANDIDATE and f.candidates and m.named_rank != 1:
                out.append(Fact("better_move", f.candidates[0], "named"))
    elif cls.id == "ask_opponent_move":
        out.append(Fact("move", m.last_san or "", "sense"))
        out.append(Fact("label", m.last_label or "", "sense"))
        if m.level >= DisclosureLevel.CANDIDATE and f.threats:
            out.extend(Fact(f"threat_{i}", t, "named") for i, t in enumerate(f.threats[:3], 1))
    elif cls.id == "ask_game_so_far":
        # One yes-or-no each: how many moves were strong, each kind of slip and its count,
        # each move where ground was lost, the opening.
        if m.tally:
            total = sum(m.tally.values())
            strong = sum(m.tally.get(k, 0) for k in ("brilliant", "great", "best", "excellent"))
            out.append(
                Fact(
                    "strong_moves",
                    f"{strong} of the player's {total} judged moves were best, excellent,"
                    " great or brilliant",
                    "words",
                )
            )
            for k in ("inaccuracy", "mistake", "miss", "blunder"):
                if m.tally.get(k):
                    out.append(
                        Fact(
                            f"count_{k}",
                            f"{m.tally[k]} {_plural(k, m.tally[k])}",
                            "words",
                        )
                    )
        # The one thing to work on: the most serious slip (the worst label; the latest of
        # equals). A four-sentence review cannot name every lost move, and the coach's own
        # directive asks for the one thing.
        if m.lost:
            rank = {"blunder": 4, "miss": 3, "mistake": 2, "inaccuracy": 1}
            worst = max(
                enumerate(m.lost),
                key=lambda e: (
                    max((v for k, v in rank.items() if f"({k}" in e[1]), default=0),
                    e[0],
                ),
            )[1]
            out.append(Fact("lost_worst", worst, "named"))
        if m.opening:
            out.append(Fact("opening", m.opening, "named"))
    elif cls.id == "ask_opening":
        if m.opening:
            out.append(Fact("opening", m.opening, "named"))
    elif cls.id == "ask_material":
        if m.material:
            out.append(Fact("material", m.material, "sense"))
    elif cls.id == "game_over":
        if m.result:
            out.append(Fact("result", m.result, "sense"))
    return [x for x in out if x.value]


CLASSES: dict[str, Class] = {
    c.id: c
    for c in [
        Class(
            "verdict",
            "coach",
            "a word on the player's fine move",
            trigger="post_move",
            max_sentences=1,
            ends_with_question=False,
        ),
        Class(
            "verdict_weaker",
            "coach",
            "the player's move was fine but not the strongest",
            trigger="post_move",
            max_sentences=2,
            ends_with_question=False,
        ),
        Class(
            "lesson",
            "coach",
            "the player's move cost something: the pattern to remember",
            trigger="post_mistake",
            max_sentences=4,
            ends_with_question=True,
        ),
        Class(
            "decisive_question",
            "coach",
            "a critical moment before the player moves: one orienting question",
            trigger="decisive_question",
            max_sentences=3,
            ends_with_question=True,
        ),
        Class(
            "stuck_hint",
            "coach",
            "the player has been thinking a long time: the hint the level allows",
            trigger="stuck",
            max_sentences=3,
            ends_with_question=True,
        ),
        Class("game_over", "coach", "the game ended", trigger="game_over", max_sentences=3),
        Class(
            "ask_hint",
            "ask",
            "a hint at the level in force",
            [
                "hint",
                "give me a hint",
                "any hints?",
                "I'm stuck",
                "where should I look?",
                "a nudge please",
                "help",
            ],
            max_sentences=3,
            ends_with_question=True,
        ),
        Class(
            "ask_move",
            "ask",
            "the move to play",
            [
                "what should I play?",
                "what's the best move here?",
                "just tell me the move",
                "what would you play?",
                "which move?",
            ],
            max_sentences=3,
        ),
        Class(
            "ask_line",
            "ask",
            "the line",
            [
                "show me the line",
                "what's the continuation?",
                "and then what?",
                "what happens after the best move?",
                "show me the variation",
            ],
            max_sentences=3,
        ),
        Class(
            "ask_plan",
            "ask",
            "the plan in this position",
            [
                "what's the plan here?",
                "what should I be aiming for?",
                "what's the idea in this position?",
                "how do I make progress?",
                "what's my plan?",
            ],
            max_sentences=4,
        ),
        Class(
            "ask_eval",
            "ask",
            "who stands better",
            [
                "am I winning?",
                "who's better here?",
                "how am I doing?",
                "is this position good for me?",
                "am I in trouble?",
            ],
            max_sentences=3,
        ),
        Class(
            "ask_threat",
            "ask",
            "what the opponent threatens",
            [
                "what is he threatening?",
                "what's the threat?",
                "what does my opponent want?",
                "is anything hanging?",
                "what are they going for?",
            ],
            max_sentences=3,
        ),
        Class(
            "ask_named_move",
            "ask",
            "a move the player names",
            [
                "is {san} good?",
                "what about {san}?",
                "why not {san}?",
                "can I play {san}?",
                "I'm thinking of {san}",
                "what if I play {san}?",
            ],
            max_sentences=4,
        ),
        Class(
            "ask_last_move",
            "ask",
            "the player's last move, a mistake",
            [
                "why was that a mistake?",
                "what was better?",
                "what did I miss?",
                "why was that bad?",
                "what should I have played?",
            ],
            max_sentences=4,
        ),
        Class(
            "ask_opponent_move",
            "ask",
            "the opponent's last move",
            [
                "why did he play that?",
                "what does that move do?",
                "was that a blunder?",
                "what's the point of that move?",
                "did they just mess up?",
            ],
            max_sentences=4,
        ),
        Class(
            "ask_game_so_far",
            "ask",
            "the game so far",
            [
                "how did I do overall this game?",
                "how am I playing so far?",
                "where did I lose it?",
                "what do I keep missing?",
                "summarise my game",
                "how did I do?",
                "review my game",
            ],
            max_sentences=4,
        ),
        Class(
            "ask_opening",
            "ask",
            "the opening",
            ["what opening is this?", "what opening did we play?", "is this still theory?"],
            max_sentences=3,
        ),
        Class(
            "ask_material",
            "ask",
            "the material balance",
            [
                "what's the material?",
                "am I up material?",
                "who's up material?",
                "how many pieces am I up?",
            ],
            max_sentences=2,
        ),
        Class(
            "ask_free",
            "ask",
            "a question the player really asked that fits no template; judged on relevance"
            " and grounding",
            [],
            max_sentences=4,
        ),
    ]
}


# The classes whose subject is the move just played. A master's note is about that move,
# so it is the reference for exactly these items: the same items a bot game yields, with
# the note attached where the book has one.
ABOUT_LAST_MOVE = frozenset(
    {"verdict", "verdict_weaker", "lesson", "ask_last_move", "ask_opponent_move"}
)

# The template questions that carry no claim about the board (a question is not a claim),
# so the claims check need not ask a model about them; the ones naming a move do.
PLAIN_QUESTIONS = frozenset(
    q for cls in CLASSES.values() for q in cls.questions if "{san}" not in q
)


def question_for(cls: Class, rng: random.Random, san: str | None = None) -> str:
    text = rng.choice(cls.questions)
    return text.format(san=san) if "{san}" in text else text


def pick_named_move(
    lines: list[tuple[str, float]], rng: random.Random
) -> tuple[str, int, float] | None:
    """A move to ask about: the best, a near-best or a weak one from the engine's lines,
    with its rank and how far behind the best it is."""
    if not lines:
        return None
    best = lines[0][1]
    choices = (
        [0] + ([1, 2] if len(lines) > 2 else []) + ([len(lines) - 1] if len(lines) > 3 else [])
    )
    k = rng.choice(choices)
    san, score = lines[k]
    return san, k + 1, max(0.0, best - score)


_SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*(?=\s|$)")


def sentence_count(text: str) -> int:
    return len(_SENTENCE_END.findall(text)) or (1 if text.strip() else 0)


def classify_trigger(
    trigger: str, label: str | None, loss: float | None, best_is_played: bool
) -> str | None:
    """The coach-initiated class of a request by its trigger and the move it follows."""
    if trigger == "post_mistake":
        return "lesson"
    if trigger == "post_move":
        if loss is not None and loss >= 0.02 and not best_is_played:
            return "verdict_weaker"
        return "verdict"
    if trigger == "decisive_question":
        return "decisive_question"
    if trigger == "stuck":
        return "stuck_hint"
    if trigger == "game_over":
        return "game_over"
    return None


def as_json(m: Moment) -> dict[str, Any]:
    return {
        "level": m.level.name,
        "user_to_move": m.user_to_move,
        "last_san": m.last_san,
        "last_by_user": m.last_by_user,
        "last_label": m.last_label,
        "last_loss": m.last_loss,
        "last_best_san": m.last_best_san,
        "lines": m.lines,
        "leads": m.leads,
        "last_best_leads": m.last_best_leads,
        "record": m.record,
        "tally": m.tally,
        "lost": m.lost,
        "opening": m.opening,
        "material": m.material,
        "standing": m.standing,
        "result": m.result,
        "named_san": m.named_san,
        "named_rank": m.named_rank,
        "named_loss": m.named_loss,
        "previous_san": m.previous_san,
        "fen_before_last": m.fen_before_last,
        "material_arc": m.material_arc,
        "facts": m.facts.model_dump(),
        "facts_before": m.facts_before.model_dump() if m.facts_before is not None else None,
    }
