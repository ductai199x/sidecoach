"""The coach: what to say, when, from which facts, checked before it is shown.

Three things keep it honest. The model sees everything -- the engine lines, the legal
moves, the facts past the current level -- so its hints can be precise and true, and the
disclosure level says what it may say (Tai, 2026-09-26). The level only ever moves through
the deterministic unlocks in `disclosure.py`, driven by what the user did or asked, never
by the model's own judgement. And every draft is checked before it is shown for claims
about the board that the board contradicts, a verdict against the record, and the form
the task asks for; a draft that fails is regenerated once with the problem named, and then
replaced by a template, so the user never sees a false claim and never sees nothing. What
a reply gives away of the answer is logged (`Speaker._leaks`), never a reason to refuse it.

The model is slow, a second or three, so it runs on a worker thread; the capture loop
hands in events and polls for messages, exactly as it does with the engine.
"""

from __future__ import annotations

import contextlib
import json
import queue
import random
import re
import statistics
import threading
from collections import Counter, deque
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import chess

from sidecoach.analysis.computed import position_key
from sidecoach.analysis.exchange import VALUE, defenders
from sidecoach.coach import draft_claims, idea_words, situation
from sidecoach.coach.annotations import annotations_for, mentions_in
from sidecoach.coach.bag import Bags
from sidecoach.coach.conversation import Conversation, PlyRecord, hands, harness_problem, ways
from sidecoach.coach.disclosure import (
    Disclosure,
    PositionFacts,
    Unlock,
    leak_check,
    squares_in,
    visible_facts,
)
from sidecoach.coach.effects import effects_text, line_notes, line_spoken
from sidecoach.coach.engine_tools import TOOL_ROUNDS, TOOL_WORDS, EngineTools
from sidecoach.coach.facts import PIECE_NAMES, describe_move, facts_from_analysis, standing_words
from sidecoach.coach.named import EngineLike as NamedEngine
from sidecoach.coach.named import analyse_named, disclosed_frames
from sidecoach.coach.named import spoken as spoken_examination
from sidecoach.coach.policy import InterjectionPolicy, TimeControl, time_control_of
from sidecoach.coach.prompt import system_message
from sidecoach.coach.summarise import summarise
from sidecoach.coach.verdicts import OPENERS, PRINCIPLES
from sidecoach.coach.verdicts import verdict as local_verdict
from sidecoach.events import (
    Analysis,
    Annotation,
    ClockObservation,
    CoachMessage,
    DisclosureLevel,
    Event,
    GameEnded,
    GameReview,
    GameStarted,
    Label,
    Line,
    MoveAssessment,
    MovePlayed,
    ProfileEvidence,
    ProfileSnapshot,
    PuzzleOutcome,
    Recommendation,
    Resync,
    ScreenState,
    ScreenStateChanged,
    Severity,
    Side,
    Takeback,
    Trigger,
)
from sidecoach.llm.base import ChatMessage, Completion, LLMProvider, ToolRound, refused_for_good
from sidecoach.profile.causes import HABIT
from sidecoach.profile.insights import motif_words

MAX_TOKENS = 220
# The top of the rating band below which the templates say no term of art (verdicts.JARGON).
PLAIN_BELOW = 1400
# The bottom of the rating band from which the templates add no textbook rule to a lesson.
STRONG_FROM = 1800
# After every this-many mistakes the lesson ends by asking what the player saw: a
# self-report the profile keeps beside the mistake (DESIGN 5.6, "sparingly"). One that
# goes unanswered is the last of the game: a player with the mic off heard "did you miss
# your opponent's idea altogether, or see it and misjudge the line?" seven times in one
# game (2026-09-19), the model echoing the directive's words each time.
ELICIT_EVERY = 3
# The player's words within this long of the question count as its answer (the
# profiler's SELF_REPORT_S, which keeps those words beside the mistake).
ELICIT_ANSWER_S = 60.0
# The word on a move by its label (Tai, 2026-09-20). LOCAL: written from the board, no
# model -- theory, the engine's own move, and an excellent one (as strong as the best).
# EXCITING: great and brilliant, the model at its full effort with some delight, since
# they are rare and worth a real sentence (Tai, 2026-09-26: they go through the API). A good
# move was a little behind the best, and the model names the better one.
LOCAL = frozenset({"book", "best", "excellent"})
EXCITING = frozenset({"great", "brilliant"})


def _local(label: Label | None) -> bool:
    return label is not None and label.value in LOCAL


def _exciting(label: Label | None) -> bool:
    return label is not None and label.value in EXCITING


def _great_meaning(facts: PositionFacts) -> str:
    """Why a move is great (score.label_move), from the sheet of the position it was played
    in: the only move there, or the best reply to the opponent's error. Offered both, the model
    called a punished error "the one move that keeps the advantage" (dev audit, 2026-09-26)."""
    if facts.only_move:
        return "the only good move: every other move lost a lot"
    return "the chance taken: their last move was an error, and this is the best reply to it"


# How many of the last lessons a sound move's word may not teach again (verdicts.lesson):
# "develop first" after every developing move of an opening is a lecture, not a lesson.
TAUGHT_MEMORY = 3
# The templates said without a model are the coach on a phone with no model, and the floor
# the model must rise above (Tai, 2026-09-23: "the template coach must be good already").
# Each rung says something true about this position that helps to find the answer: the
# idea (the goal, the pattern and why it works), the focus (the target and the piece), the
# move (played out, correct by construction). Nothing here could be said of any position:
# the stock questions it replaces -- "what is your opponent's most dangerous idea?", "look
# at the whole board" -- were said on a mate in one.
# The word on a move a little behind the best, and the start of the lesson on one that cost
# something; "{best}" is the better move. Without the move played in them, the same better
# move twice running gave the same sentence twice (the offline sweep, 2026-09-23).
FINE_TIGHTER = [
    "Fine, but {best} was a little tighter",
    "That works, though {best} was a touch better",
    "Playable -- {best} was a little more precise",
]
COST_YOU = [
    "That one cost you: {best} was better",
    "That cost you something: {best} was the move",
    "Not that one -- {best} was better",
]
# The pattern to keep, said after a mistake that missed it: the rule of thumb, not the
# position. Keyed by the pattern's name as the fact sheet gives it.
PATTERN_LESSONS: dict[str, str] = {
    "a fork": "when two of their pieces can be hit from one square, look for the move that "
    "gets there",
    "a skewer": "a check or an attack on a line with something bigger behind it wins what is "
    "behind",
    "a pin": "a pinned piece cannot run, so hit it again",
    "a discovered attack": "when one of your pieces blocks another's line, moving it with a "
    "threat makes two threats at once",
    "a double check": "against a double check the king must move, whatever else is attacked",
    "a trapped piece": "a piece with no safe square is lost once it is attacked",
    "removing the defender": "take the guard first, and what it guarded falls",
    "a deflection": "a piece guarding two things guards neither once one is attacked",
    "a decoy": "a sacrifice can drag a piece onto the square where it gets hit",
    "a loose piece": "before every move, look at what they have left undefended",
    "a mating attack": "when their king is short of squares, look at every check, the "
    "sacrifices too",
}
_MATES = ("mate", "Boden's")
# Before the facts of the position have landed: an honest word, not a stock question.
STILL_READING = "Give me a moment: I am still reading this position."


_PAWN_MOVE = re.compile(r"[a-h](x[a-h])?[1-8]")


def _sentence(text: str) -> str:
    """`text` as a sentence: a capital first, a stop last."""
    text = text.strip()
    if not text:
        return ""
    if not _PAWN_MOVE.match(text):  # "f7+ forces mate", never "F7+"
        text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def idea_template(facts: PositionFacts) -> str:
    """The first hint, without a model: the situation as a short lesson on the position --
    what is at stake, the key fact named on the board, the principle behind it, the trap
    (`coach/situation.py`). Never the move."""
    if facts.situation:
        return facts.situation
    parts: list[str] = []
    if facts.goal:
        parts.append(_sentence(facts.goal))
    if facts.question:
        parts.append(facts.question)
    return " ".join(parts) if parts else STILL_READING


def focus_template(facts: PositionFacts) -> str:
    """The second hint, without a model: the piece that does it, and the question that leads
    from there to the move. The target was the first hint's."""
    parts: list[str] = []
    if (
        facts.target
        and facts.target.startswith("their king on")
        and facts.goal
        and ("mate" in facts.goal or "attack" in facts.goal)
    ):
        # A mate's or an attack's squares, held back from the first hint: one may be where
        # the check lands.
        parts.append(_sentence(f"look at {facts.target}"))
    if facts.piece:
        parts.append(_sentence(f"use {facts.piece}"))
    if facts.nudge:
        parts.append(facts.nudge)
    elif facts.question and parts and facts.question not in situation.ANY_POSITION:
        # Questions that fit any position narrow nothing: never the second hint's.
        parts.append(facts.question)
    return " ".join(parts) if parts else idea_template(facts)


def move_template(facts: PositionFacts) -> str:
    """The move rung, without a model: the move, played out and explained."""
    if facts.why:
        return _sentence(facts.why)
    if not facts.candidates:
        return "I have no move for this yet."
    if facts.plan:
        return _sentence(f"the move is {facts.candidates[0]}: {facts.plan}")
    return f"The move is {facts.candidates[0]}."


def their_move_now(board: chess.Board, facts: PositionFacts) -> str:
    """What the opponent threatens when it is their move: their best move now and what it
    does on the board, read from the position as it stands (no pass: they are to move)."""
    if not facts.candidates:
        return "It's their move; I have no read on it yet."
    san = facts.candidates[0]
    try:
        move = board.parse_san(san)
    except ValueError:
        return "It's their move; I have no read on it yet."
    does = effects_text(board, move, hazards=False)
    if does:
        return f"It's their move, and their best is {san} ({does})."
    return f"It's their move, and their best is a quiet one, {san}: it takes nothing at once."


def in_check_threat(board: chess.Board) -> str:
    """What the opponent threatens when the player is in check: the check, and what else the
    checking piece can win, read from the board ("their queen on e6 gives check, and it also
    attacks your pawn on c6")."""
    us = board.turn

    def kind(square: chess.Square) -> chess.PieceType:
        return board.piece_type_at(square) or chess.PAWN

    def name(square: chess.Square) -> str:
        return f"{PIECE_NAMES[kind(square)]} on {chess.square_name(square)}"

    checkers = sorted(board.checkers())
    if not checkers:
        return "Nothing is threatened right now: no move of theirs wins anything."
    text = f"Their {' and '.join(name(s) for s in checkers)} "
    text += "gives check" if len(checkers) == 1 else "give check"
    if len(checkers) == 1:
        # What it can win next, besides the king: undefended, or worth more than it.
        worth = VALUE[kind(checkers[0])]
        hit = sorted(
            (
                s
                for s in board.attacks(checkers[0]) & board.occupied_co[us]
                if kind(s) != chess.KING and (not defenders(board, us, s) or VALUE[kind(s)] > worth)
            ),
            key=lambda s: (-VALUE[kind(s)], s),
        )
        if hit:
            text += ", and it also attacks " + " and ".join(f"your {name(s)}" for s in hit[:2])
    return text + "."


def line_template(facts: PositionFacts, board: chess.Board | None = None) -> str:
    """The line, and why its first move works. With the `board` the facts describe, each move
    with what it does, computed (`effects.line_spoken`): the answer to "show me the line",
    exact where the model's explanations went wrong a third of the time."""
    if not facts.line:
        return "I have no line for this yet."
    spoken = line_spoken(board, list(facts.line)) if board is not None else ""
    text = f"The line runs {spoken}." if spoken else f"The line is {' '.join(facts.line)}."
    return f"{text} {_sentence(facts.why)}" if facts.why else text


def hint_template(
    facts: PositionFacts, level: DisclosureLevel, board: chess.Board | None = None
) -> str:
    """The template for a hint at `level`: two hints, then the answer."""
    if level >= DisclosureLevel.LINE:
        return line_template(facts, board)
    if level >= DisclosureLevel.MOVE:
        return move_template(facts)
    if level >= DisclosureLevel.FOCUS:
        return focus_template(facts)
    return idea_template(facts)


# The verbs a move's "why" goes on with after its move: said of the move, they take "it".
_WHY_VERBS = (
    "is ",
    "forces ",
    "stops ",
    "starts ",
    "holds ",
    "takes ",
    "goes ",
    "meets ",
    "puts ",
    "threatens ",
    "doubles ",
    "brings ",
    "rolls ",
    "pushes ",
    "gains ",
)


# A "why" that names its move a sacrifice first: "a queen sacrifice, and after Qxf7, ...".
_SACRIFICE = re.compile(r"(an? [a-z]+ sacrifice), (.+)", re.S)


def _recapture(board: chess.Board | None, best: str, played: str | None) -> str | None:
    """When the player made the right capture with the other piece: "take back with the
    rook on e1, not the one on a1" (13.Raxd1 for Rexd1 was told "you win their queen",
    which the player had just done, Tai's game, 2026-09-24)."""
    if board is None or played is None:
        return None
    try:
        good = board.parse_san(best)
        done = chess.Move.from_uci(played)
    except ValueError:
        return None
    if (
        good.to_square != done.to_square
        or good.from_square == done.from_square
        or not board.is_capture(good)
        or not board.is_capture(done)
    ):
        return None
    piece = board.piece_at(good.from_square)
    other = board.piece_at(done.from_square)
    if piece is None or other is None:
        return None
    mine = chess.piece_name(piece.piece_type)
    theirs = (
        "the one"
        if other.piece_type == piece.piece_type
        else f"the {chess.piece_name(other.piece_type)}"
    )
    return (
        f"take with the {mine} on {chess.square_name(good.from_square)}, not {theirs} on "
        f"{chess.square_name(done.from_square)}"
    )


RECAPTURE_RULE = "when two pieces can take, ask which you want to keep where it is"


def rule_of(
    facts: PositionFacts,
    best: str,
    *,
    board: chess.Board | None = None,
    played: str | None = None,
) -> str | None:
    """The rule of thumb `lesson_template` would add after the move, when there is one that
    is the mistake's own: the recapture's, or the pattern of the tactic missed
    (`PATTERN_LESSONS`, a mating attack's for a mate). A plan's mechanism is not: "pieces
    still at home take no part in the game" was said of a move that let Qxg7 in, where the
    lesson was the threat (Tai's phone, 2026-09-27). The coach holds a rule back while it is
    cooling down (`Coach._rule_ok`)."""
    if _recapture(board, best, played) is not None:
        return RECAPTURE_RULE
    theme = facts.theme or ""
    keep = PATTERN_LESSONS.get(theme)
    if keep is None and any(word in theme for word in _MATES):
        keep = PATTERN_LESSONS["a mating attack"]
    return keep


def lesson_template(
    facts: PositionFacts,
    best: str,
    opener: str,
    *,
    board: chess.Board | None = None,
    played: str | None = None,
    allowed: str | None = None,
    with_rule: bool = True,
    strong: bool = False,
) -> str:
    """After a move that cost something: the better move, what it does, what the move made
    allowed, and the rule of thumb. `opener` is the rotated "That one cost you: {best} was
    better"; `board` and `played` (UCI) are the position the player faced and the move made,
    `allowed` what it let the opponent do (`idea_words.allowed`), when known. The rule is said
    as a sentence of its own, never announced: "Keep this one:", "Worth remembering:" and
    three more rotated before it, and nagged (Tai, 2026-09-27). `with_rule` False leaves it
    out (it was said lately: `Coach._rule_ok`), and so does `strong` (a player of 1800 and
    up): the concrete point is the lesson."""
    text = opener
    why = facts.why or ""
    rule = rule_of(facts, best, board=board, played=played) if with_rule and not strong else None
    other_piece = _recapture(board, best, played)
    if other_piece is not None:
        text += f" -- {other_piece}"
        if allowed:
            text = _sentence(text) + " " + allowed
        if rule is None:
            return _sentence(text)
        return _sentence(text) + " " + _sentence(rule)
    if why.startswith(best):
        rest = why[len(best) :].lstrip(" ,:;")
        if rest.startswith("and ") and not rest.startswith("and after"):
            rest = why  # "Rfc8 and Qc6 double your rooks": the plan, said whole
        else:
            rest = rest[4:] if rest.startswith("and ") else rest
            sacrifice = _SACRIFICE.match(rest)
            if sacrifice is not None:
                # "axb3, a knight sacrifice, stops bxc2": "-- it is a knight sacrifice, and
                # it stops bxc2" (the master games, Sokolov-Dreev, 2026-09-24).
                named, tail = sacrifice.group(1), sacrifice.group(2)
                if tail.startswith(_WHY_VERBS):
                    tail = "it " + tail
                rest = f"it is {named}, " + (tail if tail.startswith("and ") else f"and {tail}")
            elif rest.startswith(_WHY_VERBS):
                rest = "it " + rest  # "Nf2# was better -- it is mate", "-- it stops Nxd3"
        if rest and _stale(rest, played):
            rest = ""
        if rest:
            text += f" -- {rest}"
    text = _sentence(text)
    if allowed:
        text += " " + _sentence(allowed)
    if rule is not None and not _said(rule, text):
        text += " " + _sentence(rule)
    return text


def _with_line(text: str, opener: str, best: str, facts: PositionFacts | None) -> str:
    """An answer to "why was that bad?" with the engine's line when it would name nothing
    but the better move: "g6 was better." alone answered nothing (13 of 95 answers in the
    offline sweep's bot games, 2026-09-27). The line is the concrete reason, and the
    position the move was played from withholds nothing."""
    if facts is None or text.rstrip(".") != opener or not facts.line or facts.line[0] != best:
        return text
    return _sentence(f"{opener} -- the line runs {' '.join(facts.line[:5])}")


def _without(text: str, sentence: str) -> str:
    """`text` with `sentence` taken out, the spaces closed up."""
    return " ".join(text.replace(sentence, " ").split())


def _stale(reason: str, played: str | None) -> bool:
    """Whether a lesson's reason, read off the position the player faced, names a piece on a
    square their move then left or landed on -- the position before it, not the one now: "d4
    was a little more precise -- your bishop on c1 is still shut in behind the pawn on d2"
    after 4.d3, which had just let it out (Nova bot games, 2026-09-25). A line ("after Qg2,
    Rdg1, the queen on g2 is trapped") is about the moves to come, and stands."""
    if played is None or reason.startswith("after "):
        return False
    try:
        move = chess.Move.from_uci(played)
    except ValueError:
        return False
    squares = (chess.square_name(move.from_square), chess.square_name(move.to_square))
    return any(re.search(rf"\bon {sq}\b", reason) for sq in squares)


# A rule of thumb -- "a rule to take with you: ...", "rooks belong on open files" -- is not
# said again for this many of the coach's messages, across games and puzzles alike, and no
# two rules come within RULE_SPACING messages of each other (Tai, 2026-09-27, after a run of
# puzzles: "it keeps on nagging me ... there has to be a cooldown of some number of messages
# before it can appear again"). A puzzle is a new game, so a memory per game started over at
# every one. Thirty messages and four apart still nagged -- a rule in one message of ten,
# on fine moves and mistakes alike (Tai's phone, 2026-09-27: "the coach is a teacher, not a
# nagging annoying bot"): a rule once a session, and one in ten messages at most.
RULE_COOLDOWN = 200
RULE_SPACING = 10
# The model's word while a rule is cooling down: the concrete point, no rule of thumb. Every
# held wording carries RULE_HELD_MARK, which the rubric reads (tools/eval/rubric.py:
# "transfer" is not asked of a reply told to leave the rule out).
RULE_HELD_MARK = "No rule of thumb this time"
RULE_HELD = f"{RULE_HELD_MARK}: only what this position shows."


def fresh(text: str, recent: str, opener: str | None = None) -> str:
    """`text` without what the coach said lately (`recent`, lower case): the reason after
    " -- " in the first sentence. The verdict and the better move always stay: that is news.
    `opener` is what the first sentence opens with, the verdict and the better move; the
    reason is what follows it. (A rule of thumb said lately is `Coach._rule_ok`'s to hold.)"""
    said = recent.lower()
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    if not sentences:
        return text
    first, rest = sentences[0], sentences[1:]
    head, reason = first, ""
    if opener is not None:
        # An opener may carry its own " -- " ("Not that one -- Ng3 was better"): with no
        # reason after it, the better move was taken for one and dropped when the same move
        # had just been said (Nova bot games, 2026-09-25).
        if first.lower().startswith(opener.lower() + " -- "):
            head, reason = first[: len(opener)], first[len(opener) + 4 :]
    elif " -- " in first:
        # The last clause ("Playable -- d4 was a little more precise -- your bishop on c1 is
        # still shut in ...").
        head, reason = first.rsplit(" -- ", 1)
    if reason and reason.rstrip(".").lower() in said:
        first = head.rstrip(".") + "."
    return " ".join([first, *rest])


def _said(keep: str, text: str) -> bool:
    """Whether `text` already says `keep`: four of its words in a row."""
    words = re.findall(r"[a-z']+", keep.lower())
    said = " ".join(re.findall(r"[a-z']+", text.lower()))
    return any(" ".join(words[j : j + 4]) in said for j in range(len(words) - 3))


# The board check over a draft: the text, the board, the boards along the best line, and
# the player's colour (whose "your"); the problems, one line each.
DraftCheck = Callable[[str, chess.Board, Sequence[chess.Board], chess.Color | None], list[str]]

# Verbalized sampling: how many candidate replies to ask for, and the room and warmth to
# give the distribution. Warmer than a single reply on purpose, since a check filters them.
VS_CANDIDATES = 4
# An answer to the player's own question is wanted fast, and it is checked like any other;
# two candidates halve the wait and still leave a spare for the checks to reject.
VS_ASK_CANDIDATES = 2
# A word on a fine move is one short sentence; two candidates keep the call short.
VS_ROUTINE_CANDIDATES = 2
VS_MAX_TOKENS = 2048
# Cold. The candidates are already a distribution the model verbalizes (their p= marks);
# hot sampling on top of that is where the invented sentence came from.
VS_TEMPERATURE = 0.1
_NUMBER_WORDS = {2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE"}
# No figures unless the player asks: the record says what a move cost in expected score and
# the model quoted it in every note ("costing 0.03"; Tai, 2026-09-19), and a hint said
# "ahead by about 1.2 pawns". The code notes a figure where a directive carries this
# (`Speaker._notes`), so every directive that forbids one says so in these words.
NO_FIGURES = (
    "No figures -- no percentage, no score, no move number: say how a move rates, and what it "
    "cost, in words; the numbers are for when the player asks. Name moves bare."
)
# The player has just played the move and sees it on the board: naming it, or saying what
# it plainly did ("Nxd5 -- the knight taking the pawn on d5"), tells them nothing (Tai,
# 2026-09-23: "we want the coach to be instructive, always find a moment to teach
# something, not just a dumb bot").
NOT_THE_MOVE = (
    "They see their own move on the board, so naming it or saying what it plainly did -- the "
    "capture, the check, the castling, the promotion -- tells them nothing: do not open with "
    "it. Say what it means instead: the idea behind it and the pattern to keep and, when there "
    "was a better move, that move and why."
)
# While a rule cools down: the same, without the pattern to keep. Told to hold the rule back
# at the end of a directive that still asked for "the pattern to keep", the local model gave
# one anyway in 2 of 17 lessons (the free check, 2026-09-27).
NOT_THE_MOVE_HELD = NOT_THE_MOVE.replace(
    "the idea behind it and the pattern to keep", "the idea behind it"
)


def _hold_rule(directive: str) -> str:
    """`directive` with its ask for a rule to keep taken out, and RULE_HELD said."""
    for asked, held in (
        (NOT_THE_MOVE, NOT_THE_MOVE_HELD),
        (" -- as a pattern to remember.", "."),
        ("as a pattern to remember, in", "in"),
        (" as a lesson to keep", ""),
    ):
        directive = directive.replace(asked, held)
    return f"{directive} {RULE_HELD}"


def _lesson_data(board: chess.Board | None, best: str, played: str | None, rule: str | None) -> str:
    """What a lesson's template says that the sheet does not, as facts for the model: which
    piece should have taken, and the rule of thumb that fits the mistake, when one may be
    said. The template's finished lesson, handed over as the least a reply says, came back
    word for word in two lessons of three (the prompt audit, 2026-09-28)."""
    out = ""
    other = _recapture(board, best, played)
    if other is not None:
        out += f" The right capture: {other}."
    if rule is not None:
        out += f" A rule of thumb that fits, if you give one: {rule}."
    return out


# What the first hint may say, as the Task's level line states it: the key fact is named on
# the board (Tai, 2026-09-24), and only the answer is kept back (Tai, 2026-09-26: a vague
# coach is worse than one that says a little too much).
IDEA_LEVEL_WORDS = (
    "the idea: what is at stake and the key fact, with every piece, square and move of "
    "theirs the sheet names, the target's square included. Keep back only the answer: the "
    "player's move, the square it goes to (unless the target stands there), and which of "
    "their pieces makes it"
)
SEE_ALL_WORDS = (
    "You see the answer so that the hint can be precise and true; a hint that gives a little "
    "too much is a lesser fault than a vague one."
)


# What the model is asked to do on each rung before the player moves: the facts the rung's
# template is built from, said the model's own way, sharper for this position (Tai,
# 2026-09-23: "AI is supposed to elevate good to great"). What the rung may give away is the
# level line's to say (`Coach._level_line`), once. The parts it names are not to be said
# back: "The key fact is that ...; the principle here is ..." was in 41% of the local
# model's hints (the prompt audit's baseline, 2026-09-28).
IDEA_DIRECTIVE = (
    "Give the first hint as a short lesson on this position, from what the sheet says a first "
    "hint rests on: what is at stake (how many moves hold and what the next best costs; or the "
    "goal, or the nature, where it gives no stakes), then the key fact on the board, named as "
    "the sheet names it -- the target, their threat, the plan or the pattern -- and what it "
    "threatens or allows, with the principle behind it, so they know what to look for; and the "
    "trap when the sheet has one. Those are what the lesson covers, not words to say: give the "
    "fact itself, never its name. The sheet's words are facts to teach from, not a script: "
    "every sentence yours, and true only of this board. Two or three sentences, then, if you "
    "like, one question."
)
# The game so far, used: an earlier moment of the same idea is the coach's best material, and
# the excellence rubric found it used in half the lessons that had one (2026-09-25). The coach
# finds those moments itself (`Conversation.same_idea`) and lists them: told to find its own
# in the record, the model misremembered one ("as after 9...a6, when Bc4 saved your bishop":
# neither happened) and copied the directive's own example move as a link (quality rounds 2
# and 3, 2026-09-25). TIE_TO_THE_GAME marks the directives a link belongs in; `_speak` puts
# the rule for the moments listed, or for none, in its place.
TIE_TO_THE_GAME = "Tie this lesson to the game only through the earlier moments listed."
TIE_LISTED = (
    "Under 'Earlier in this game' are the player's earlier moves played with the very same "
    "idea as this one. Tie this lesson to the one that fits best, in a clause -- naming its "
    "move bare, as the list does -- and say of it only what the list says. Speak of no other "
    "earlier moment."
)
NO_EARLIER = "Speak of no earlier moment of the game: none had the same idea."
# A line asked for once it was given: what they want is what comes after it.
LINE_GIVEN = (
    "You have given them this line already, in your turns above: they want what comes after "
    "its end -- the next moves and the idea behind them."
)


FOCUS_DIRECTIVE = (
    "Name the piece to use and what it goes after -- the target -- as the fact sheet names "
    "them, with the principle behind it in a clause, and ask the one question that leads from "
    "there to the move. Do not name the move. One or two sentences."
)
MOVE_DIRECTIVE = (
    "The next hint is the answer: give the move and explain it correctly from the fact "
    "sheet's why -- what it does and what it leads to -- and the pattern to keep. One or two "
    "sentences, no question."
)


def vs_directive(candidates: int = VS_CANDIDATES) -> str:
    count = _NUMBER_WORDS.get(candidates, str(candidates))
    return (
        f"\n\nInstead of one reply, give {count} different ways you could say this, varied in "
        "wording and in the angle each takes, sampled from your full range rather than the "
        "single most obvious phrasing. Put each on its own line as: [p=0.30] then the reply, "
        "the replies in a random order, not the likeliest first. "
        "Each reply obeys every rule above and keeps to the length asked, on that one line. "
        "The p values are how likely you would be to choose each; they need not sum to one."
    )


_VS_MARKER = re.compile(r"\[\s*p?\s*=?\s*([0-9]*\.?[0-9]+)\s*\]\s*", re.IGNORECASE)
# A marker fragment left in a candidate's text: a whole one, or one cut off mid-way.
_STRAY_MARKER = re.compile(r"\s*\[\s*p\s*=\s*[0-9.]*\]?\s*")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# Where a reply's last complete sentence ends: terminal punctuation, then any closing
# quote or bracket. What follows the last of these is a sentence the model never finished.
# A move number's stop ("after 6. d3", "12. Bxc6+") ends no sentence: a bare number of
# up to three digits, at the start or after a space, before the stop.
_SENTENCE_END = re.compile(
    r"(?<!(?<![\w.])\d)(?<!(?<![\w.])\d\d)(?<!(?<![\w.])\d\d\d)[.!?…][\"'”’)\]]*(?=\s|$)"
)


_PIECE_WORD = {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}

# A reply may run this many sentences past the number its directive gives before it is
# faulted: the length is reasonable, not strict.
LENGTH_SLACK = 2
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}
_SAYS_BEST = re.compile(
    r"\b(?:the |your )?(?:best|strongest|top|engine's) (?:move|choice)\b|\bwas best\b|\bis best\b",
    re.IGNORECASE,
)
_SAYS_SLIP = re.compile(r"\b(?:blunder|mistake|inaccuracy|a miss)\b", re.IGNORECASE)
# A figure after "cost": a share of the game (0.10, 10%), never a count of moves or pawns
# ("lost the pawn on e4 2 moves ago" gave a cost of 2).
_SAYS_COST = re.compile(
    r"\b(?:cost|costing|costs|lost|losing)\b[^.;]{0,30}?(?<![A-Za-z0-9])(\d+(?:\.\d+)?)"
    r"(?![.\d]*[A-Za-z])(?!\s*(?:moves?|plies|pawns?|pieces?|tempi|tempo|points?)\b)\s*(%|percent)?",
    re.IGNORECASE,
)


# A reply's clauses: a sentence, or the halves of one joined by " -- " or ", but / though".
_CLAUSE = re.compile(r"(?<=[.;:!?])\s+|\s+--\s+|,\s+(?:but|though|while)\s+")
# A clause about the move just played without naming it: "That was ...", "It cost ...".
_ABOUT_IT = re.compile(r"^\W*(?:that|it|this|your (?:last )?move)\b", re.IGNORECASE)


def _verdict_problems(text: str, request: _Request) -> list[str]:
    """Faults in what a reply says of the move just assessed: a "best" that was not, a
    slip word for a sound move, a cost that is not the recorded one -- clause by clause,
    where the clause is about that move (it names it, or opens with "that" or "it"). Every
    lesson names the better move too, and skipping any reply that did let "That was a
    blunder: a3 was better" through for a good move (the runtime audit, 2026-09-25)."""
    a = request.assessment
    played = request.played_san
    if a is None or not played:
        return []
    core = played.rstrip("+#")
    names_it = re.compile(rf"(?<![A-Za-z]){re.escape(core)}[+#]?(?![A-Za-z0-9])")
    label = a.label.value if a.label else a.severity.value
    # A move that cost nothing is as good as the best: "the best move on the board" for a
    # great move the engine chose itself is true.
    best_ok = label in ("best", "excellent") or a.expected_loss < 0.005
    problems: list[str] = []
    for clause in _CLAUSE.split(text):
        if not (names_it.search(clause) or _ABOUT_IT.match(clause)):
            continue
        others = [m for m in _SAN.findall(clause) if m.rstrip("+#") != core]
        if (
            not others
            and not best_ok
            and _SAYS_BEST.search(clause)
            and not re.search(
                r"\bnot (?:the )?(?:best|strongest)\b|\bwas better\b|\bwas stronger\b",
                clause,
                re.I,
            )
        ):
            problems.append(f"called {played} the best move; its verdict is {label}")
        if (
            a.severity is Severity.NONE
            and _SAYS_SLIP.search(clause)
            and not re.search(
                r"\bnot (?:a )?(?:blunder|mistake)\b|\bno (?:blunder|mistake)\b", clause, re.I
            )
        ):
            problems.append(f"called {played} a slip; its verdict is {label}")
        for found in _SAYS_COST.finditer(clause):
            value = float(found.group(1))
            if found.group(2) or value > 1.0:
                value /= 100.0
            if abs(value - a.expected_loss) > max(0.02, 0.35 * a.expected_loss):
                problems.append(
                    f"gave the cost of {played} as {found.group(1)}; it was {a.expected_loss:.2f}"
                )
    return list(dict.fromkeys(problems))


_LIMIT_WORDS = re.compile(
    r"\b(?:(?:at most|no more than) (one|two|three|four|five|six) sentences?"
    r"|(one|two|three|four|five|six)(?: or (one|two|three|four|five|six))? (?:short )?sentences?"
    r"|(one|two|three|four|five|six) sentences? at most"
    r"|(?:fifteen|twenty) words at most)\b",
    re.IGNORECASE,
)


def directive_limits(directive: str) -> tuple[int | None, bool | None]:
    """The length the directive states, in sentences, and whether it asks for a closing
    question (True), forbids one (False) or says nothing (None)."""
    limit: int | None = None
    for m in _LIMIT_WORDS.finditer(directive):
        if m.group(0).lower().endswith("words at most"):
            n = 1
        else:
            n = max(_WORD_NUMBERS[g.lower()] for g in m.groups() if g)
        limit = n if limit is None else max(limit, n)
    low = directive.lower()
    question: bool | None = None
    if re.search(r"\bask one (?:orienting )?question\b", low):
        question = True
    if re.search(r"\bthen, if you like, one question\b", low):
        # A question offered on top of the sentences: one more is allowed, none required.
        limit = None if limit is None else limit + 1
    elif re.search(r"\bthen one question\b", low):
        # A question asked for on top of the sentences: one more is allowed.
        question = True
        limit = None if limit is None else limit + 1
    elif re.search(r"\bending with (?:one |a )?question\b", low):
        question = True
    if re.search(r"\bno question\b|\bdo not answer with a question\b", low):
        question = False
    return limit, question


def trim_to_limit(text: str, limit: int, wants_question: bool | None) -> str:
    """The first `limit` complete sentences of `text`; the closing question kept in place
    of the last one when the directive asks for a question and the text ends with one."""
    ends = [m.end() for m in _SENTENCE_END.finditer(text)]
    if len(ends) <= limit:
        return text
    sentences = [text[a:b].strip() for a, b in zip([0, *ends[:-1]], ends, strict=True)]
    if wants_question and sentences[-1].endswith("?") and limit >= 2:
        kept = [*sentences[: limit - 1], sentences[-1]]
    else:
        kept = sentences[:limit]
    return " ".join(kept)


def _sentence_total(text: str) -> int:
    return len(_SENTENCE_END.findall(text))


def _vs_wrap(messages: list[ChatMessage], candidates: int = VS_CANDIDATES) -> list[ChatMessage]:
    """Append the verbalized-sampling directive to the final user turn; none for one reply."""
    if not messages or candidates <= 1:
        return messages
    wrapped = list(messages)
    last = dict(wrapped[-1])
    last["content"] = str(last.get("content", "")) + vs_directive(candidates)
    wrapped[-1] = last  # type: ignore[assignment]
    return wrapped


def complete_sentences(text: str) -> str:
    """The text up to the end of its last complete sentence; empty if it has none.

    A reply the token budget cut off ends mid-sentence -- "...is spot-on. Given" -- and in
    the first live game a one-word stump like "Given" was the candidate that passed every
    check and got spoken. A stump is not a reply, so it is trimmed away or, alone, dropped.
    """
    ends = list(_SENTENCE_END.finditer(text))
    return text[: ends[-1].end()].strip() if ends else ""


# Markdown the model writes around words ("**Rc1**", "`e4`", "__now__") and at the start of a
# line ("# ", "- ", "* "): the reply is spoken and shown as it is, and "?**" at its end failed
# the closing-question check (the parity pilot, 2026-09-25).
_EMPHASIS = re.compile(r"\*\*|__|`")
_LINE_MARK = re.compile(r"(?m)^\s*(?:#{1,6}\s+|[-*]\s+(?=\S))")


def plain(text: str) -> str:
    """`text` without markdown: the emphasis marks and the line marks go, the words stay."""
    return _LINE_MARK.sub("", _EMPHASIS.sub("", text))


def _parse_candidates(raw: str) -> list[tuple[float, str]]:
    """The verbalized replies as (probability, text). A reply with no markers is one candidate.

    The text of each candidate runs from its marker to the next, so a reply that wraps onto
    another line is kept whole. A model that ignores the format still yields one usable reply.
    Every candidate is cut back to its last complete sentence.
    """
    cleaned = plain(_THINK_BLOCK.sub("", raw)).strip()
    markers = list(_VS_MARKER.finditer(cleaned))
    if not markers:
        cleaned = complete_sentences(_STRAY_MARKER.sub("", cleaned).strip())
        return [(1.0, cleaned)] if cleaned else []
    out: list[tuple[float, str]] = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(cleaned)
        # A reply cut off by the token budget can end in a half-written next marker
        # ("...move? [p=0.2" with no closing bracket); no fragment may reach the user.
        text = complete_sentences(_STRAY_MARKER.sub("", cleaned[start:end]).strip())
        if not text:
            continue
        try:
            prob = float(marker.group(1))
        except ValueError:
            prob = 0.0
        out.append((prob, text))
    return out


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _json_object(text: str) -> object:
    """The first JSON object in a model's answer, or None."""
    m = _JSON_OBJECT.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _sample_reply(passed: list[tuple[float, str]]) -> str:
    """Sample one reply, weighted by its stated likelihood; ties and zeros still get a chance."""
    weights = [max(prob, 0.01) for prob, _ in passed]
    return random.choices([text for _, text in passed], weights=weights, k=1)[0]


class _FirstCandidate:
    """Reads the first verbalized candidate off the model's stream and hands it to a sink
    sentence by sentence. Each new complete sentence is checked (`Speaker._problems`: what
    it gets wrong) before it goes; a fault stops the stream with what was said so far kept. The
    stream is stopped once the candidate is complete -- the second marker seen -- so the
    other candidates are never written. A reply with no marker at all is one candidate."""

    def __init__(
        self,
        sink: Callable[[str], bool],
        check: Callable[[str], list[str]],
        *,
        limit: int | None = None,
    ) -> None:
        self.sink = sink
        self.check = check
        # The task's length: the stream is closed there (an eight-sentence homily went out
        # whole with the voice on: the runtime audit).
        self.limit = limit
        self.buffer = ""
        self.spoken = ""  # what the sink has been given
        self.problems: list[str] = []
        self.done = False
        self.declined = False  # the sink refused the first sentence: the voice is not wanted

    def _candidate(self) -> tuple[str, bool]:
        """The first candidate's text so far, and whether it is complete."""
        cleaned = plain(_THINK_BLOCK.sub("", self.buffer))
        markers = list(_VS_MARKER.finditer(cleaned))
        if not markers:
            # No marker yet: the model may be writing one, or writing a bare reply.
            if cleaned.lstrip().startswith("[") or len(cleaned) < 12:
                return "", False
            return cleaned, False
        start = markers[0].end()
        if len(markers) > 1:
            return cleaned[start : markers[1].start()], True
        return cleaned[start:], False

    def _offer(self, text: str, final: bool = False) -> bool:
        """Hand the sink the complete sentences of `text` not yet given. False to stop.

        A sentence has ended only once the space after its full stop has arrived: at
        the buffer's end "costing 0." is the start of "costing 0.02", and a cost read
        as 0 failed the check and stopped the stream. A chunk without letters ("6.", a
        move number the model put first) waits for the sentence it introduces."""
        cleaned = _STRAY_MARKER.sub("", text).strip()
        ends = [m for m in _SENTENCE_END.finditer(cleaned) if final or m.end() < len(cleaned)]
        if self.limit is not None:
            # Cut at the task's length before anything is said, however large the chunk.
            ends = ends[: self.limit + LENGTH_SLACK]
        complete = cleaned[: ends[-1].end()].strip() if ends else ""
        if len(complete) <= len(self.spoken):
            return True
        fresh = complete[len(self.spoken) :].strip()
        if sum(ch.isalpha() for ch in fresh) < 2:
            return True  # a number and a stop: the sentence it begins is still coming
        faults = self.check(fresh) if fresh else []
        if faults:
            self.problems = faults
            return False
        if not self.sink(fresh):
            self.declined = True
            return False
        self.spoken = complete
        # As long as the task allows: the rest is not said.
        return self.limit is None or _sentence_total(self.spoken) < self.limit + LENGTH_SLACK

    def feed(self, delta: str) -> bool:
        if self.done:
            return False
        self.buffer += delta
        text, complete = self._candidate()
        # A complete candidate (its successor's marker seen) ends on its last stop, with
        # nothing after: that stop counts.
        if not self._offer(text, final=complete):
            self.done = True
            return False
        if complete:
            self.done = True
            return False
        return True

    def finish(self) -> None:
        """The stream ended: whatever complete sentences remain go out."""
        if self.done:
            return
        self.done = True
        text, _ = self._candidate()
        if text:
            self._offer(text, final=True)


@dataclass
class ModelCall:
    """One call to the model: the turn it was given (the last user message: memory, fact
    sheet, instruction), what it wrote, what the server said about the call (finish
    reason, tokens, seconds), and how the candidates fared. The debug log's answer to
    "why did the coach say that?"."""

    raw: str
    stats: dict[str, Any]
    candidates: int
    passed: int
    prompt: str = ""


@dataclass
class Spoken:
    """One coach message, with what it was checked against, for the log."""

    message: CoachMessage
    level: DisclosureLevel
    problems: list[str]
    attempts: int
    calls: list[ModelCall] = field(default_factory=list)
    # Already a turn of the conversation (a quick answer recorded before the model was
    # asked to build on it), so polling it must not record it again.
    recorded: bool = False
    # The reply went out through the request's sink as it was written: the voice has
    # had it, sentence by sentence, and nothing is to be spoken again.
    streamed: bool = False
    # What the verifier found, when one runs: per candidate the statements the messages do
    # not support, and whether the reply spoken was a repair of the best candidate.
    verified: list[list[str]] = field(default_factory=list)
    repaired: bool = False
    # What the model's reply gave away of the answer the level withholds (`Speaker._leaks`),
    # and what it did against the form asked of it (`Speaker._notes`): logged for debugging,
    # never a reason the reply was refused.
    leaks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # The moment of the game it was asked for (`Coach._epoch`): one that lands after a new
    # game, a takeback or a re-read is no turn of the moment now.
    epoch: int = 0


@dataclass
class _Position:
    ply: int
    board: chess.Board
    analysis: Analysis | None = None
    facts: PositionFacts | None = None
    disclosure: Disclosure = field(default_factory=Disclosure)
    last_move: str | None = None
    # The first hint's wordings, peeked from the coach's bags as the facts were written and
    # taken from them when the hint is said (`coach/bag.py`).
    leads: dict[str, str] = field(default_factory=dict)


def board_after(board: chess.Board, event: MovePlayed) -> chess.Board:
    """The board after `event`, carrying the moves that led to it: the idea reader reads an
    exchange in progress from them (`idea.exchange_owed`), and a board rebuilt from the FEN
    alone read 20...Nxh5 Qxh5 as "you can win a knight" (Tai's puzzles, 2026-09-26). The
    move goes onto `board` when that is where it was played from, else onto the position it
    was played from; a board whose FEN is not the event's own (the analysis is matched on it)
    is the event's FEN, without moves."""
    for start in (board, chess.Board(event.fen_before)):
        if start.fen() != event.fen_before:
            continue
        try:
            move = chess.Move.from_uci(event.uci)
        except ValueError:
            break
        if move not in start.legal_moves:
            break
        after = start.copy()
        after.push(move)
        if after.fen() == event.fen_after:
            return after
        break
    return chess.Board(event.fen_after)


def board_back(board: chess.Board, fen: str) -> chess.Board:
    """The board taken back to `fen`, keeping the moves before it when they reach it."""
    back = board.copy()
    while back.move_stack and back.fen() != fen:
        back.pop()
    return back if back.fen() == fen else chess.Board(fen)


# The game's pace: the user's last few think times. Under FAST_PACE_S a move, the game
# is fast -- a model reply (two to five seconds) would land after the next move, and a
# spoken sentence (three to five) would be talked over the next two. So a fast game
# gets the coach's templates, written from the board at once, and gets them as text;
# the voice resumes when the player slows down (see the panel's PACE_RESUME_S) and for
# what they ask (Tai, 2026-09-22: "the coach has to be able to keep up with the game").
# The pace is the game's own: carried over, the quick last moves of one puzzle -- a line
# found and played out -- made the next one fast, and a find after two minutes' thought
# was only shown (Tai's puzzles, 2026-09-27: "sometimes it's just completely muted").
FAST_PACE_S = 6.0
PACE_MOVES = 4
# How many plies back the coach keeps a position and its facts.
POSITIONS_KEPT = 40
# How long the lesson on a mistake waits for the analysis of the position the move left:
# the verdict comes first, from the best line alone, and the analysis about a second after
# it -- without it the lesson had no "what your move allowed" (the parity audit, 2026-09-25).
LESSON_WAIT_S = 1.5


@dataclass
class _Request:
    trigger: Trigger
    level: DisclosureLevel
    system: str
    messages: list[ChatMessage]
    facts: PositionFacts
    board: chess.Board
    fallback: str
    ply: int
    game_id: str
    # What the user said this turn, if anything. A square, piece or move the user named is
    # not a leak for the coach to repeat back: confirming it reveals nothing they did not say.
    user_text: str | None = None
    # What the directive itself disclosed this turn -- the engine's word on the move or
    # line the player asked about -- which the coach may therefore repeat.
    disclosed: str = ""
    # Board marks built from the same filtered facts the model sees (see coach.annotations).
    annotations: list[Annotation] = field(default_factory=list)
    # How many replies to ask the model for; fewer for an answer the player is waiting on.
    candidates: int = VS_CANDIDATES
    # The moves of the record (SAN): those played so far, and those the verdicts named as
    # better. A reply about the game as it went may name either; the board checker must
    # not fault "Rg6" from move 30 as unplayable now, nor the leak check count it.
    played: frozenset[str] = frozenset()
    # The move just assessed and its assessment, when the turn is about it: what the reply
    # says of its verdict and cost is checked against them.
    played_san: str | None = None
    assessment: MoveAssessment | None = None
    # Where to send the reply sentence by sentence as the model writes it (the panel's
    # voice), or None for a reply that goes out whole once checked. The sink returns
    # False to stop the stream: the board has moved on, or the voice is not wanted.
    stream_to: Callable[[str], bool] | None = None
    # A mundane reply -- a word on a fine move -- which the Speaker's routine model
    # writes (the same model at less reasoning, by default none: the first sound comes
    # in under a second instead of three).
    routine: bool = False
    # A fast game: the fallback -- the template written from the board -- is the reply,
    # and the model is not called (it could not keep up).
    template_only: bool = False
    # What a reply must name, one of: the better move the lesson's template names, or a
    # square the hint's template names. The template is the floor; a true reply that says
    # less than it was a loss the checks let through (parity pass, 2026-09-25).
    must_name: tuple[str, ...] = ()
    # More boards a statement may be true on: after the move the player just made, and along
    # the engine's line from there -- the board the model is shown for a lesson after a move.
    frames: list[chess.Board] = field(default_factory=list)
    # When the reply is about a position already moved from and the player is to move again:
    # the answer there (its best move and line), which the reply must not give away -- a late
    # verdict's lesson said "and now Qxf7# is mate" (the runtime audit, 2026-09-25).
    now_answer: tuple[str, ...] = ()
    # The game's moves as the record numbers them ("17. Rh1"): said so, they are history.
    record: frozenset[str] = frozenset()
    # For a directive a link belongs in (TIE_TO_THE_GAME): the earlier moves of the game the
    # reply may not name -- all but those the list of earlier moments and the sheets give.
    # None when the directive has no place for a link.
    unlisted: frozenset[str] | None = None
    # The engine at the model's call while it answers (a question of the player's), when the
    # coach has an engine and the model takes tools.
    tools: EngineTools | None = None
    # The moment of the game the request was built in (`Coach._epoch`).
    epoch: int = 0


def _player(facts: PositionFacts) -> chess.Color | None:
    """The side the coach speaks to, from the facts: whose "your" a draft means."""
    if facts.user_side is None:
        return None
    return Side(facts.user_side).color


def _after_best(request: _Request) -> list[chess.Board]:
    """The boards along the engine's line, so what the line does reads as real.

    A fork or pin the recommended move creates is not on the board in front of the player;
    the draft checker is handed the frames after each move of the line so it does not fault
    the coach for naming a piece on the square a move will put it on, or -- when asked for
    the line and reciting it -- for a move that is only playable further along it.
    """
    line = request.facts.line or request.facts.candidates[:1]
    frames: list[chess.Board] = []
    board = request.board.copy(stack=False)
    for san in line:
        try:
            move = board.parse_san(san)
        except ValueError:
            break
        board = board.copy(stack=False)
        board.push(move)
        frames.append(board)
    # And along what the directive disclosed: the line the player asked about, examined; and
    # the boards the model's own calls to the engine went through.
    if request.disclosed:
        frames += disclosed_frames(request.board, request.disclosed.splitlines())
    return frames + request.frames + (request.tools.frames if request.tools is not None else [])


def _rated(label: str) -> str:
    """A label as said of a move: "a mistake", "an inaccuracy", "a miss"; the good ones as
    they are ("best", "excellent"), and theory as "a book move"."""
    if label in ("inaccuracy", "mistake", "miss", "blunder"):
        return ("an " if label[0] in "aeiou" else "a ") + label
    return "a book move" if label == "book" else label


def _names(text: str, item: str) -> bool:
    """Whether `text` names `item`: a square, or a move in SAN (with or without its mark)."""
    if re.fullmatch(r"[a-h][1-8]", item):
        return item in squares_in(text)
    core = item.rstrip("+#")
    if core.startswith("O-O"):
        # Castling: "O-O" is not named by "O-O-O" (the recheck of round 3's replies).
        pattern = rf"(?<![-A-Za-z0-9]){re.escape(core)}(?!-O)[+#]?(?![A-Za-z0-9])"
        return re.search(pattern, text) is not None
    return re.search(rf"(?<![A-Za-z]){re.escape(core)}[+#]?(?![A-Za-z0-9])", text) is not None


_ENGINE_NAMED = re.compile(r"\b(?:stockfish|the engine|engine's|an engine)\b", re.IGNORECASE)


_ILLEGAL_CLAIM = re.compile(
    r"\b(O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]? (?:is|was) "
    r"(?:not (?:a )?legal|illegal)"
)


def false_illegal(text: str, board: chess.Board) -> list[str]:
    """A move the reply calls illegal that is legal for the side to move on the board it is
    about. The AI said "O-O is not legal here because your opponent's bishop is still on
    e2", "Qxf7+ is illegal here" and "Kf7 is illegal because f7 is controlled by the King
    on e8" of legal moves (its own sessions, checked against the boards, 2026-09-26). A
    claim inside a line ("after 22...Bxg5, 23.Bxg5 is illegal") is about a later position
    and is left alone, as is a move of the side not to move."""
    out: list[str] = []
    for sentence in re.split(r"(?<=[.!?;])\s+", text):
        for found in _ILLEGAL_CLAIM.finditer(sentence):
            if "after" in sentence[: found.start()].lower():
                continue
            try:
                board.parse_san(found.group(1))
            except ValueError:
                continue
            out.append(f"{found.group(1)}: called illegal, but it is a legal move here")
    return out


def figure_said(text: str) -> bool:
    """Whether `text` quotes a figure: any digit outside a move, a square or an ordinal ("the
    7th rank") -- "ahead by about 1.2 pawns" went through a check for percentages alone."""
    rest = _SAN.sub(" ", text)
    rest = re.sub(r"\b\d+(?:st|nd|rd|th)\b|\b[A-E]\d\d\b", " ", rest)
    return re.search(r"\d", rest) is not None


_PLAIN_PIECE = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop"}
_PLAIN_PIECE |= {chess.ROOK: "rook", chess.QUEEN: "queen"}


def restated(text: str, request: _Request) -> list[str]:
    """What a word after the player's move says that they see on the board (Tai,
    2026-09-23): opening with the move just played, what it plainly did -- the capture, the
    check, the castling, the promotion -- unless the better move does the same. The templates
    never do; the model could, and nothing checked it (the eval audit, 2026-09-25). A figure
    in it is NO_FIGURES' note (`Speaker._notes`)."""
    if request.trigger not in (Trigger.POST_MOVE, Trigger.POST_MISTAKE) or not request.played_san:
        return []
    played = request.played_san.rstrip("+#")
    faults: list[str] = []
    if re.match(rf"\W*{re.escape(played)}(?![A-Za-z0-9])", text):
        faults.append(f"opens with {played}, the move just played")
    try:
        move = request.board.parse_san(request.played_san)
    except ValueError:
        return faults
    better: chess.Move | None = None
    if request.assessment is not None and request.assessment.best:
        with contextlib.suppress(ValueError):
            better = chess.Move.from_uci(request.assessment.best[0])
    board = request.board
    low = text.lower()
    victim = board.piece_at(move.to_square)
    if victim is not None and victim.piece_type in _PLAIN_PIECE:
        square = chess.square_name(move.to_square)
        name = _PLAIN_PIECE[victim.piece_type]
        verb = r"\b(?:takes|taking|took|take|captures?|capturing|grabs?|grabbing|wins?|winning|won)"
        took = rf"{verb}\s+(?:the|their|a|that)\s+(?:{name}\s+on\s+{square}|{square}\s+{name})\b"
        took += rf"|\b(?:takes|taking|took|take)\s+on\s+{square}\b"
        if re.search(took, low) and not (better is not None and better.to_square == move.to_square):
            faults.append(f"says what {played} plainly did: the capture on {square}")
    # What the better move does as well is the lesson, not a restatement.
    legal = better is not None and better in board.legal_moves
    better_castles = legal and board.is_castling(better)  # type: ignore[arg-type]
    better_checks = legal and board.gives_check(better)  # type: ignore[arg-type]
    better_promotes = better is not None and better.promotion is not None
    if board.is_castling(move) and "castl" in low and not better_castles:
        faults.append(f"says what {played} plainly did: castling")
    # The check said of the move just played: in a clause about it -- one that names it or
    # opens with "that", "it", "your move", "you gave" -- and names no other move. "After
    # Bg4 your knight jumps in with check" is the better line's check, and was faulted as
    # the played move's (the benchmark's smoke run, 2026-09-25).
    names_played = re.compile(rf"(?<![A-Za-z]){re.escape(played)}[+#]?(?![A-Za-z0-9])")
    says_check = any(
        re.search(r"\b(?:with|gives?|gave|giving) (?:a )?check\b", clause, re.I)
        and (
            names_played.search(clause)
            or _ABOUT_IT.match(clause)
            or re.match(r"\W*you (?:just )?(?:gave|give|played|put)\b", clause, re.I)
        )
        and not [m for m in _SAN.findall(clause) if m.rstrip("+#") != played]
        for clause in _CLAUSE.split(text)
    )
    if board.gives_check(move) and says_check and not better_checks:
        faults.append(f"says what {played} plainly did: the check")
    if move.promotion and "promot" in low and not better_promotes:
        faults.append(f"says what {played} plainly did: the promotion")
    return faults


# The game's history in a request, up to the next section: the moves that stand only there
# are the earlier moments (`unlisted_links`).
_GAME_SO_FAR = re.compile(r"# Game so far\n.*?(?=\n# )", re.DOTALL)
# A move with its number in front: "after 9...a6", "12. Qc2".
_MOVE_NUMBER = re.compile(r"\b\d+\s*\.(?:\s*\.\.)?\s*(?:O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8])")


def unlisted_links(text: str, unlisted: frozenset[str]) -> list[str]:
    """What a lesson says of an earlier moment the coach did not list: a move of the game
    before the position taught that neither the list of earlier moments nor the sheets give,
    and any move with its number -- "the same safety lesson as after 9...a6, when Bc4 saved
    your bishop" named a moment that never happened (quality round 3, 2026-09-25)."""
    faults = [
        f"names {move}, an earlier moment not listed: tie only to the moments listed"
        for move in sorted(unlisted)
        if _names(text, move)
    ]
    if _MOVE_NUMBER.search(text):
        faults.append("a move with its number: name moves bare")
    return faults


def omitted(text: str, must_name: Sequence[str]) -> str | None:
    """The fault when `text` names none of `must_name`; None when it names one, or there is
    nothing it must name."""
    if not must_name or any(_names(text, item) for item in must_name):
        return None
    if len(must_name) == 1:
        return f"left out {must_name[0]}: the reply must name it"
    return f"left out the key fact: the reply must name one of {', '.join(must_name)}"


VERIFY_SYSTEM = (
    "You check a chess coach's reply against the messages the coach was given: the game so "
    "far, the engine's verdicts (the label, the cost, the move the engine preferred, what "
    "the moves do), the lines, the facts (the nature of the position, the theme, the plan, "
    "the threats, the ideas), the tally, and the directive. A statement is supported when a "
    "message states it, or restates it in other words, or it follows directly from the move "
    "itself (a capture takes the piece on that square, a check makes the king answer, a move "
    "lands where it lands), or it is a plain inference the messages make obvious, or "
    "ordinary chess usage of a stated fact (a rook on an open file 'active', a king on e1 "
    "'in the centre'). Approximate or hedged wording of a supported fact is supported. "
    "Unsupported is only what is specific and absent: what a piece attacks, defends, pins "
    "or threatens when no message says so; a plan, weakness, structure or evaluation no "
    "message gives; a reason for a move no message gives; a move, win or loss that did not "
    "happen; a habit the record does not show. A question is not a statement. Quote each "
    "unsupported statement in the reply's own words; when in doubt, it is supported. "
    'Answer in JSON only: {"unsupported": [str]}'
)

REPAIR_SYSTEM = (
    "You edit a chess coach's reply. Remove or reword only the statements listed as "
    "unsupported, so that what remains is supported by the messages the coach was given. "
    "Change nothing else: every other sentence stays exactly as it is -- in particular keep "
    "every fact about the move that the messages give (its label, what it cost, the move "
    "the engine preferred, the idea) -- the form stays (a closing question stays a closing "
    'question), the voice stays; add nothing. Answer in JSON only: {"reply": str}'
)


class Speaker:
    """Runs the model on a worker thread and checks what it says.

    A draft with a false claim about the board or against the record is sent back once with
    the problem named; a second failure is replaced by the template, and the failure
    counted. What a reply gives away of the answer, and what it does against the form asked
    of it, are logged, not refused.
    With a `verifier`, every candidate that passes those checks is also read against the
    messages by a second model, the clean ones are sampled from, and when none is clean the
    likeliest is repaired once: test-time verification, on top of the demonstrations in C.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        draft_check: DraftCheck | None = None,
        verifier: LLMProvider | None = None,
        routine_provider: LLMProvider | None = None,
        claim_reader: LLMProvider | None = None,
    ) -> None:
        self.provider = provider
        # A model that lists the draft's claims for the board to decide
        # (`coach/draft_claims.py`): off without one.
        self.claim_reader = claim_reader
        self._claims_seen: dict[tuple[str, str], list[str]] = {}
        # The model for a mundane reply (a word on a fine move), where nothing is at
        # stake and the first sound matters: the same model at less reasoning (Tai:
        # low at most; none is what makes it fast, low reasons as long as medium).
        # Everything else -- a mistake, a decisive moment, a nudge, a question of the
        # player's -- keeps `provider` at its effort.
        self.routine_provider = routine_provider or provider
        self.draft_check = draft_check
        self.verifier = verifier
        self.failures = 0
        self.fallbacks = 0
        self.dropped = 0  # unprompted requests an ask overtook before they started
        self.last_error: str | None = None
        self._requests: queue.Queue[_Request | Spoken | None] = queue.Queue()
        self._results: queue.SimpleQueue[Spoken] = queue.SimpleQueue()
        self._thread: threading.Thread | None = None
        self._busy = threading.Event()

    @property
    def pending(self) -> int:
        return self._requests.qsize() + (1 if self._busy.is_set() else 0)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="sidecoach-coach", daemon=True)
            self._thread.start()

    def close(self, timeout: float = 10.0) -> None:
        thread, self._thread = self._thread, None
        if thread is not None:
            self._requests.put(None)
            thread.join(timeout)

    def submit(self, request: _Request) -> None:
        """Queue a request. An ask goes ahead of any unprompted request still waiting: the
        player is waiting for this one, and a post-move remark or a nudge that has not
        started yet would only be stale by the time the answer came out."""
        if request.trigger is Trigger.USER_ASK:
            kept: list[_Request | Spoken | None] = []
            while True:
                try:
                    item = self._requests.get_nowait()
                except queue.Empty:
                    break
                if isinstance(item, _Request) and item.trigger is not Trigger.USER_ASK:
                    self.dropped += 1
                    continue
                kept.append(item)
            for item in kept:
                self._requests.put(item)
        self._requests.put(request)

    def submit_ready(self, spoken: Spoken, *, now: bool = False) -> None:
        """A message that is already final; it goes out through the same queue, in order --
        or, with `now`, straight to the results, ahead of whatever the model is writing:
        the quick answer to a button press must not wait behind a post-move remark."""
        if now:
            self._results.put(spoken)
        else:
            self._requests.put(spoken)

    def run_pending(self) -> None:
        """Answer every queued request on the calling thread: for tests and for replay."""
        while True:
            try:
                request = self._requests.get_nowait()
            except queue.Empty:
                return
            if isinstance(request, Spoken):
                self._results.put(request)
            elif request is not None:
                self._results.put(self.speak_now(request))

    def poll(self) -> list[Spoken]:
        out: list[Spoken] = []
        while True:
            try:
                out.append(self._results.get_nowait())
            except queue.Empty:
                return out

    def speak_now(self, request: _Request) -> Spoken:
        """Verbalized sampling: ask for several replies, keep a checked one, vary the voice.

        A single reply collapses onto the one most typical phrasing, which is why the coach
        repeated itself. Asking the model to verbalize a small distribution of replies (arxiv
        2510.01171) recovers its range; the board and form checks then filter that set, and a
        reply is sampled by its own stated likelihood from those that pass. Because several are
        offered at once, a clean one is usually among them, so the template is rarely needed.
        """
        messages = _vs_wrap(list(request.messages), request.candidates)
        problems: list[str] = []
        text = ""
        attempts = 0
        calls: list[ModelCall] = []
        verified: list[list[str]] = []
        repaired = False
        streamed = False
        provider = self.routine_provider if request.routine else self.provider
        if getattr(provider, "offline", False) or request.template_only:
            # No model (llm.provider none), or no time for one (a fast game): the
            # request's fallback is the reply -- the verdict from the board, the lesson
            # or hint template, the review's own sentences -- or silence when there is
            # none. Free and offline.
            self.fallbacks += 1
            text = request.fallback
            return Spoken(
                message=CoachMessage(
                    text=text,
                    level=request.level,
                    trigger=request.trigger,
                    game_id=request.game_id,
                    ply=request.ply,
                    fen=request.board.fen(),
                    annotations=[*request.annotations, *mentions_in(text, request.board)],
                ),
                level=request.level,
                problems=[],
                attempts=0,
                epoch=request.epoch,
            )
        while attempts < 2:
            attempts += 1
            stats: dict[str, Any] = {}
            reader: _FirstCandidate | None = None
            stream = getattr(provider, "chat_stream", None)
            if (
                request.stream_to is not None
                and attempts == 1
                and self.verifier is None
                and stream is not None
            ):
                # The first candidate goes to the voice sentence by sentence as it is
                # written, each sentence checked before it is said; the stream
                # is closed once that candidate is complete, the others unwritten.
                limit, _ = directive_limits(
                    str(request.messages[-1]["content"]).rsplit("# Task", 1)[-1]
                )
                reader = _FirstCandidate(
                    request.stream_to, lambda t: self._problems(t, request), limit=limit
                )
            try:
                completion = self._complete(
                    provider, request, messages, reader.feed if reader is not None else None
                )
                raw = completion.text
                stats = dict(completion.stats)
            except Exception as exc:  # noqa: BLE001
                self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                raw = ""
            if reader is not None:
                reader.finish()
                if reader.spoken:
                    # Said already, in part at least: the record keeps what was said and
                    # what, if anything, was wrong with the rest; no template follows a
                    # voice that has spoken.
                    text = reader.spoken
                    problems = reader.problems
                    streamed = True
                    given = "\n\n---\n\n".join(str(m["content"]) for m in messages[-2:])
                    calls.append(ModelCall(raw, stats, 1, 0 if problems else 1, given))
                    break
                # Nothing said: the first candidate failed its first sentence, or the
                # voice declined. The reply is read whole from here, as without a sink;
                # what stopped the stream goes in the call's stats for the log.
                stats["stream_problems"] = reader.problems or (
                    ["the voice declined"] if reader.declined else ["the stream ended early"]
                )
            scored = [
                (prob, candidate, self._problems(candidate, request))
                for prob, candidate in _parse_candidates(raw)
            ]
            passed = [(prob, candidate) for prob, candidate, faults in scored if not faults]
            given = "\n\n---\n\n".join(str(m["content"]) for m in messages[-2:])
            calls.append(ModelCall(raw, stats, len(scored), len(passed), given))
            if passed and self.verifier is not None:
                passed, verified, repaired = self._verify(passed, request)
            if passed:
                text, problems = _sample_reply(passed), []
                break
            problems = scored[0][2] if scored else ["empty reply"]
            if reader is not None and reader.declined:
                # The voice declined the reply (the board had moved on, or the voice was
                # busy): a second call about that moment would be paid for nothing. What
                # the stream gave is not clean, so the template stands.
                break
            # The second try carries the complaint inside the directive: appending the failed
            # attempt and the complaint as two more messages doubled the prompt on a long
            # game (131k tokens became 276k, past the model's context).
            rejected = "; ".join(problems)
            shown = raw.strip()[:600] or "(no reply)"
            retry = dict(messages[-1])
            retry["content"] = (
                str(retry.get("content", ""))
                + f"\n\nYour first attempt was:\n{shown}\n\nThose replies are not allowed: "
                + f"{rejected}. Give "
                + (
                    "a fresh reply that avoids those"
                    if request.candidates <= 1
                    else f"{_NUMBER_WORDS.get(request.candidates, request.candidates)} "
                    "fresh, varied replies that avoid those"
                )
                + ", keeping to the messages above."
            )
            messages = [*messages[:-1], retry]  # type: ignore[list-item]
        if problems and not streamed:
            self.fallbacks += 1
            text = request.fallback
            leaks: list[str] = []
            notes: list[str] = []
        else:
            notes = self._notes(text, request)
            text = self._trimmed(text, request)
            leaks = self._leaks(text, request)
        message = CoachMessage(
            text=text,
            level=request.level,
            trigger=request.trigger,
            game_id=request.game_id,
            ply=request.ply,
            fen=request.board.fen(),
            annotations=[*request.annotations, *mentions_in(text, request.board)],
        )
        return Spoken(
            message=message,
            level=request.level,
            problems=problems,
            attempts=attempts,
            calls=calls,
            verified=verified,
            repaired=repaired,
            streamed=streamed,
            leaks=leaks,
            notes=notes,
            epoch=request.epoch,
        )

    @staticmethod
    def _complete(
        provider: LLMProvider,
        request: _Request,
        messages: list[ChatMessage],
        on_delta: Callable[[str], bool] | None,
    ) -> Completion:
        """The model's reply, streamed to `on_delta` when given. With engine tools on the
        request and a model that takes them, the engine calls it makes are answered and it is
        asked again, TOOL_ROUNDS times at most, the last time to answer with what it has; the
        calls go in the stats, for the log. A server that refuses the tools is asked again
        without them, and the refusal goes in the stats: on chat completions Luna with a
        reasoning effort took no tools, and the coach said nothing at all -- and then, asked
        without them, the log still listed the calls as if they had been answered."""
        tools = request.tools if getattr(provider, "tools_ok", False) else None
        turns: list[ToolRound] = []
        refused = ""
        round_ = 0
        while True:
            extra: dict[str, Any] = {}
            if tools is not None:
                extra = {"tools": tools.specs, "turns": turns}
                if round_ == TOOL_ROUNDS:
                    extra["tool_choice"] = "none"
            try:
                if on_delta is not None:
                    completion = provider.chat_stream(  # type: ignore[attr-defined]
                        request.system,
                        messages,
                        max_tokens=VS_MAX_TOKENS,
                        temperature=VS_TEMPERATURE,
                        on_delta=on_delta,
                        **extra,
                    )
                else:
                    completion = provider.chat(
                        request.system,
                        messages,
                        max_tokens=VS_MAX_TOKENS,
                        temperature=VS_TEMPERATURE,
                        **extra,
                    )
            except Exception as exc:
                if tools is None or not refused_for_good(exc):
                    raise
                tools = None
                refused = str(exc)[:200]
                continue
            if tools is None or not completion.tool_calls or round_ == TOOL_ROUNDS:
                break
            results = [tools.run(call.name, call.arguments) for call in completion.tool_calls]
            turns = [*turns, ToolRound(completion, results)]
            round_ += 1
        if request.tools is not None and request.tools.log:
            completion.stats["tools"] = list(request.tools.log)
        if refused:
            completion.stats["tools_refused"] = refused
        return completion

    @staticmethod
    def _trimmed(text: str, request: _Request) -> str:
        """`text` cut to its directive's length and the slack past it: the sentences beyond
        are not said (the voice reads every one while the player is playing)."""
        directive = (
            str(request.messages[-1]["content"]).rsplit("# Task", 1)[-1] if request.messages else ""
        )
        limit, wants_question = directive_limits(directive)
        if limit is None or _sentence_total(text) <= limit + LENGTH_SLACK:
            return text
        return trim_to_limit(text, limit + LENGTH_SLACK, wants_question)

    def _verify(
        self, passed: list[tuple[float, str]], request: _Request
    ) -> tuple[list[tuple[float, str]], list[list[str]], bool]:
        """The candidates read against the messages by the verifier: those with nothing
        unsupported stay; when none does, the likeliest is repaired once and kept if the
        repair passes the deterministic checks. Returns the candidates to sample from, what
        the verifier found per candidate, and whether a repair was used."""
        assert self.verifier is not None
        # The latest messages: the position the player faces, the one they moved from, the
        # dialogue since, the directive -- the facts a reply may draw on are in these.
        context = "\n\n".join(str(m["content"]) for m in request.messages[-5:])

        def check(candidate: str) -> list[str]:
            content = (
                "The messages the coach was given (the latest):\n\n"
                + context
                + '\n\nThe coach\'s reply:\n"""\n'
                + candidate
                + '\n"""\n\nJSON only.'
            )
            try:
                completion = self.verifier.chat(  # type: ignore[union-attr]
                    VERIFY_SYSTEM,
                    [{"role": "user", "content": content}],
                    max_tokens=400,
                    temperature=0.0,
                )
            except Exception:  # noqa: BLE001
                return []  # the verifier failed: the candidate is not held against it
            found = _json_object(completion.text)
            items = found.get("unsupported") if isinstance(found, dict) else None
            return [str(x) for x in items if str(x).strip()] if isinstance(items, list) else []

        with ThreadPoolExecutor(max_workers=max(1, len(passed))) as pool:
            verified = list(pool.map(check, [c for _, c in passed]))
        clean = [pc for pc, faults in zip(passed, verified, strict=True) if not faults]
        if clean:
            return clean, verified, False
        prob, best = max(zip(passed, verified, strict=True), key=lambda pv: pv[0][0])[0]
        faults = verified[[c for _, c in passed].index(best)]
        # Two tries at the repair: the second is also told what the deterministic checks
        # found wrong with the first.
        note = ""
        for _ in range(2):
            content = (
                "The messages the coach was given (the latest):\n\n"
                + context
                + '\n\nThe reply:\n"""\n'
                + best
                + '\n"""\n\nUnsupported statements: '
                + "; ".join(faults)
                + note
                + "\n\nJSON only."
            )
            try:
                completion = self.verifier.chat(
                    REPAIR_SYSTEM,
                    [{"role": "user", "content": content}],
                    max_tokens=500,
                    temperature=0.0,
                )
            except Exception:  # noqa: BLE001
                return [], verified, False
            found = _json_object(completion.text)
            repaired = str(found.get("reply") or "").strip() if isinstance(found, dict) else ""
            if not repaired:
                return [], verified, False
            faults_now = self._problems(repaired, request)
            if not faults_now:
                return [(prob, repaired)], verified, True
            note = "\n\nYour previous edit was rejected for: " + "; ".join(faults_now) + "."
        return [], verified, False

    def _problems(self, text: str, request: _Request) -> list[str]:
        """What stops a draft: what it gets wrong -- a false claim about the board, a verdict
        or cost against the record, an earlier moment of the game it invents. Its form is
        not among them: that is noted (`_notes`), never a reason to refuse a reply (Tai,
        2026-09-28: "we don't need to throw anything out ... we just need good prompt
        engineering"), and neither is a leak (`_leaks`; Tai, 2026-09-26: "it should be used
        as a debugging tool and not something used to reject response")."""
        found: list[str] = []
        # The verdict word and the cost the reply gives for the move just played must be
        # the assessment's own: "best" for an excellent move, "cost 0.10" for a 0.05 loss,
        # were the judge's commonest remaining faults, and the coach holds the facts.
        found.extend(_verdict_problems(text, request))
        if request.unlisted is not None:
            # An earlier moment named that the coach did not list is one it made up ("as
            # after 9...a6, when Bc4 saved your bishop"); a move number is only form.
            found.extend(
                f for f in unlisted_links(text, request.unlisted) if not f.startswith("a move with")
            )
        if self.draft_check is not None:
            for problem in self.draft_check(
                text, request.board, _after_best(request), _player(request.facts)
            ):
                san = problem.split(":", 1)[0].strip()
                if san in request.played:
                    continue  # a move of the game's history, named as such
                found.append(problem)
        found.extend(false_illegal(text, request.board))
        found.extend(self._claim_problems(text, request))
        return list(dict.fromkeys(found))

    def _notes(self, text: str, request: _Request) -> list[str]:
        """What a reply does against the form its directive asks for, or the owner's rules
        of form: logged with the reply, like a leak, and never a reason to refuse it --
        the prompt asks for the form, and a slip of it is minor (Tai, 2026-09-28). A reply
        past its length is still cut there (`_trimmed`)."""
        found: list[str] = []
        directive = (
            str(request.messages[-1]["content"]).rsplit("# Task", 1)[-1] if request.messages else ""
        )
        limit, wants_question = directive_limits(directive)
        if limit is not None and _sentence_total(text) > limit + LENGTH_SLACK:
            found.append(f"too long: {_sentence_total(text)} sentences, {limit} asked for")
        if wants_question is True and not text.rstrip().endswith("?"):
            found.append("no closing question")
        if wants_question is False and text.rstrip().endswith("?"):
            found.append("a question where the task allows none")
        left_out = omitted(text, request.must_name)
        if left_out:
            found.append(left_out)
        found.extend(restated(text, request))
        if request.unlisted is not None and _MOVE_NUMBER.search(text):
            found.append("a move with its number: name moves bare")
        # A figure where the directive keeps the figures for when the player asks -- "46...Rh5
        # was excellent, costing only 0.01" (quality round 3, 2026-09-25).
        if NO_FIGURES in directive and figure_said(text):
            found.append("a figure in a remark nobody asked for: say it in words")
        # RULES 7: the coach speaks as the one who saw the game -- "Stockfish gives White
        # about 0.8 pawns" came in seven of the pilot's answers.
        asked_engine = re.search(r"stockfish|engine", request.user_text or "", re.I)
        if _ENGINE_NAMED.search(text) and not asked_engine:
            found.append("named the engine: say it as the coach who saw the game")
        return list(dict.fromkeys(found))

    def _claim_problems(self, text: str, request: _Request) -> list[str]:
        """The draft's claims, listed by `claim_reader` and decided on the board: those it
        contradicts. None without a reader, or when the call fails -- a draft is never held
        up by the check itself. The same text is read once."""
        if self.claim_reader is None or not text.strip():
            return []
        key = (request.board.fen(), text)
        if key not in self._claims_seen:
            player = _player(request.facts)
            player = request.board.turn if player is None else player
            line = list(request.facts.line)
            claims = draft_claims.extract(
                text,
                request.board,
                player,
                self.claim_reader,
                played=request.played_san,
                line=line,
            )
            self._claims_seen[key] = [
                f"false: {p}" for p in draft_claims.verify(claims, request.board, player, line)
            ]
        return self._claims_seen[key]

    def _leaks(self, text: str, request: _Request) -> list[str]:
        """What `text` gives away of the answer the level withholds, for the log: the
        move to find, its square, the piece that makes it. A debugging signal, never a
        reason to refuse the reply (see `_problems`)."""
        # What the user said this turn, what the directive itself disclosed (the engine on
        # what they asked about) and the moves of the game so far are not leaks: a move
        # already played may be discussed in full, whatever the level in force.
        # The piece a played move moved may be named too: "your opponent's rook took on
        # c6" about a move of the record is no hint, but the word "rook" was a leak
        # whenever the withheld piece happened to be a rook as well.
        # What the user said and what the directive disclosed are theirs already. The moves of
        # the game are no exemption: only the answer is withheld (disclosure.leak_check), and
        # exempting the record's moves, squares and pieces had let the answer through.
        said = " ".join(filter(None, [request.user_text, request.disclosed]))
        # What the sheet says the threats do ("Ne6 (uncovers an attack on the pawn on d2)")
        # is shown to the model: a piece named there is the threat, not the hint.
        threat_words = (
            " ".join(Conversation._threat_words(request.board, t) for t in request.facts.threats)
            if visible_facts(request.facts, request.level).get("threats")
            else ""
        )
        found = [
            f"withheld {item}"
            for item in leak_check(
                text,
                request.facts,
                request.level,
                said=said or None,
                record=request.record,
                shown=threat_words,
            )
        ]
        found += [
            f"gives away {move} in the position they face now"
            for move in request.now_answer
            if _names(text, move)
        ]
        return list(dict.fromkeys(found))

    def _run(self) -> None:
        while True:
            request = self._requests.get()
            if request is None:
                return
            self._busy.set()
            try:
                if isinstance(request, Spoken):
                    self._results.put(request)
                else:
                    self._results.put(self.speak_now(request))
            except Exception as exc:  # noqa: BLE001
                self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
            finally:
                self._busy.clear()


# What the user's words mean for disclosure. Deterministic on purpose: the model never
# decides how much may be said.
# A hint, or a question about the plan ("what should I be aiming for?", "what's the idea
# here?"): both earn the next level, where the theme and the plan may be named. Answered at
# the question level, a question about the plan got a question back.
_HINT = re.compile(
    r"\b(hint|clue|nudge|help me|help|stuck|where should i look|"
    r"what (?:should|do|can|could) i (?:be )?(?:aim|go|look|play|work|strive)(?:ing)? for|"
    r"what(?:'s| is) (?:the|my) (?:plan|idea|goal|aim)|what to (?:aim|go) for|"
    r"what am i (?:aiming|going|playing) for)\b",
    re.I,
)
# An overall verdict on the game: "how did I do", "how was my game", "overall", "summary".
_ASK_REVIEW = re.compile(
    r"\b(?:how (?:did|have) i (?:do|play|done|played|playing|doing)|how am i playing|"
    r"how was (?:my|the) game|"
    r"overall|summar(?:y|ise|ize)|review (?:my|the) game|rate my (?:game|play)|"
    r"where did i (?:lose|go wrong)|what (?:do|did) i keep (?:missing|doing)|"
    r"what (?:did|have) i (?:do|done) wrong|my (?:mistakes|worst moves?)|"
    r"what should i work on)\b",
    re.IGNORECASE,
)
# A question about the player over their games, not about this position: the record
# answers it, the coach writing its own queries (profile/ask.py). "This game" and "the
# last move" stay with the game's own paths above.
_ASK_PROFILE = re.compile(
    r"\b(?:my (?:games|history|record|profile|stats|statistics|weakness(?:es)?|"
    r"strengths?|habits?|pattern|trend|progress|results?|last (?:few |\d+ )?games)|"
    r"(?:over|in) (?:the )?(?:last|past|previous) (?:few |\d+ )?"
    r"(?:games|days|weeks|months|sessions)|"
    r"(?:all|previous|earlier|other|past|recent) (?:games|sessions)|last session|"
    r"(?:across|between|over) (?:games|sessions)|how many games|"
    r"(?:do|am|have) i (?:always|usually|often|tend to|keep|generally|typically|normally)|"
    r"why (?:do|am) i (?:always|usually|often|keep|so|bad|weak|losing|blundering)|"
    r"(?:am|are) i (?:getting )?(?:better|worse|improving)|"
    r"(?:what|which) (?:openings?|tactics?|motifs?|phase) (?:do|am|have) i|"
    r"(?:my|the) (?:biggest|worst|most common|main|recurring) "
    r"(?:weakness|problem|mistake|cost|leak)s?|"
    r"what should i (?:study|practise|practice|train|focus on|"
    r"work on (?:in general|next|overall)))\b",
    re.IGNORECASE,
)
# Questions about facts that are never withheld: the material, the opening, who stands better.
_ASK_MATERIAL = re.compile(
    r"\b(?:material|(?:how many|what) (?:pieces|pawns)|(?:am|are) i up|who'?s up|up material|"
    r"down material|piece count)\b",
    re.IGNORECASE,
)
_ASK_OPENING = re.compile(r"\b(?:opening|theory|book move)\b", re.IGNORECASE)
# A question about the move the player already played: its verdict is in the record and is
# theirs in full -- that position is past, and the disclosure level binds only the one now.
_ASK_LAST_MOVE = re.compile(
    r"\b(?:why was (?:that|it|my (?:last )?move) (?:bad|wrong|a mistake|a blunder|an inaccuracy)|"
    r"what was better|what should i have (?:played|done)|what did i miss|what went wrong|"
    r"was (?:that|it|my (?:last )?move) (?:a )?(?:mistake|blunder|bad|wrong)|"
    r"what'?s wrong with (?:that|it|my (?:last )?move)|why (?:is|was) (?:that|it) (?:so )?bad|"
    r"what did i do wrong|did i (?:just )?(?:blunder|mess up))\b",
    re.IGNORECASE,
)
# A question about the opponent's last move: what it does, why it was played, whether it
# was good. Its verdict and effects are in the record ("Last move") and are the player's
# to hear in full; answered as a generic ask, the coach gave a question back and named
# neither the move nor its verdict in most of them.
_ASK_OPPONENT_MOVE = re.compile(
    r"\b(?:why did (?:he|she|they) (?:play|do) that|what does that (?:move )?do|"
    r"what(?:'s| is) the (?:point|idea) of that(?: move)?|was that a (?:blunder|mistake|good move)|"
    r"did (?:he|she|they) (?:just )?(?:mess up|blunder)|"
    r"what (?:is|was) (?:he|she|they) (?:going for|up to|doing)|"
    r"why (?:that|this) move)\b",
    re.IGNORECASE,
)
_ASK_EVAL = re.compile(
    r"\b(?:am i (?:winning|losing|better|worse|ok|okay|in trouble)|who'?s (?:better|winning)|"
    r"how am i doing|is this (?:position )?good for me|"
    r"what(?:'?s| is) the (?:eval|evaluation|score)|how bad is it|am i lost)\b",
    re.IGNORECASE,
)
# What a move cost, asked in so many words: only then is the figure said.
_ASK_COST = re.compile(
    r"\b(?:how much (?:did|does) (?:it|that|this|my move)\s*(?:\w+\s*)?cost|what did (?:it|that) "
    r"cost|how (?:bad|costly) was (?:it|that|my move)|how much (?:did|have) i lose)\b",
    re.IGNORECASE,
)
# A bare request for a hint, with nothing claimed: "hint", "give me a hint", "a clue".
_ASK_HINT = re.compile(
    r"^\W*(?:(?:give me |i need |can i have |i want |any |another |one more )?(?:a |an |some )?"
    r"(?:hint|clue|nudge|help|tip)s?(?: please)?|i'?m stuck|i am stuck|where (?:should|do) i look"
    r"|i don'?t know what to (?:do|play))\W*$",
    re.IGNORECASE,
)
# The player's own move, named as theirs: asked after the opponent has replied too.
_MY_MOVE = re.compile(
    r"\b(?:what should i have (?:played|done)|what did i (?:miss|do wrong)|did i (?:just )?"
    r"(?:blunder|mess up)|my (?:last )?move|was i wrong|should i have)\b",
    re.IGNORECASE,
)
# A question about the opponent's threat.
_ASK_THREAT = re.compile(
    r"\b(?:what(?:'s|s| is| are) (?:their|the|his|her|my opponent's|the opponent's) threats?"
    r"|what (?:do|does) (?:they|he|she|my opponent|the opponent) threaten"
    r"|what (?:is|are|'s|'re) (?:they|he|she|my opponent|the opponent) (?:threatening|up to)"
    r"|is (?:there|anything) (?:a )?threat(?:ened)?|am i (?:in danger|threatened)"
    r"|what (?:do|does) (?:they|he|she|my opponent|the opponent) want"
    r"|is (?:anything|something) (?:hanging|loose|en prise)"
    r"|what (?:are|is) (?:they|he|she|my opponent|the opponent) going for)\b",
    re.IGNORECASE,
)
# A question about what to aim for: the first hint answers it, and its template is the floor.
_ASK_AIM = re.compile(
    r"\bwhat (?:should|do|can|could) i (?:be )?(?:aim|go|look|play|work|strive)(?:ing)? for\b"
    r"|\bwhat(?:'s| is) (?:the|my) (?:plan|idea|goal|aim)\b|\bwhat am i (?:aiming|going|playing) "
    r"for\b|\bwhat to (?:aim|go) for\b|\bhow (?:do|can|should) i (?:make progress|improve|"
    r"continue|proceed)\b",
    re.IGNORECASE,
)
# Words that can carry a claim about the board worth checking: a move, a square, a piece.
# A bare "hint" or "what should I play" carries none, and skipping the check is what makes
# the buttons answer at once.
_CLAIM_WORDS = re.compile(
    r"\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?|[a-h][1-8]|"
    r"pawn|knight|bishop|rook|queen|king|castl\w*|check|mate|fork|pin|skewer|hang\w*|"
    r"threat\w*|attack\w*|capture\w*|take\w*|trade\w*|sac\w*|win\w*|los\w*|"
    r"safe|blunder|mistake|good|bad|better|best|worse|worst)\b",
    re.IGNORECASE,
)


_QUESTION_START = re.compile(
    r"^(?:what|what's|whats|where|why|how|which|who|whose|is|isn't|are|can|could|should|"
    r"would|will|does|do|did|was|were|am)\b",
    re.I,
)


def question_part(text: str) -> str:
    """The sentences of the player's words that ask something -- a "?", or a question's
    first word, as dictation leaves it -- or "" when none does."""
    sentences = re.findall(r"[^.!?]+[.!?]*", text.replace("...", "…"))
    asked = [
        s.strip() for s in sentences if s.strip().endswith("?") or _QUESTION_START.match(s.strip())
    ]
    return " ".join(asked).replace("…", "...")


# The opening asked for by name ("what opening is this?"), not judged ("how is my opening?").
_OPENING_FACT = re.compile(
    r"\b(?:what|which)(?:'s| is| was)? (?:the |this |my )?opening\b|"
    r"\bis (?:this|it|that) (?:still )?(?:theory|book)\b|\bname of (?:the |this )?opening\b",
    re.IGNORECASE,
)


def fact_question(text: str) -> bool:
    """Whether the words ask only for a fact the record holds whole -- the material, the
    opening, who stands better -- which the template says exactly. The model added nothing
    there (the locked test set, 2026-09-26: excellence 1.00 for the templates on all three)
    and once gave the material after a recapture as the material now; answered by the
    template, no call is made."""
    if not (_ASK_MATERIAL.search(text) or _OPENING_FACT.search(text) or _ASK_EVAL.search(text)):
        return False
    others = (
        _ASK_HINT.match(text),
        _ASK_AIM.search(text),
        _ASK_THREAT.search(text),
        _ASK_MOVE.search(text),
        _ASK_LINE.search(text),
        _ASK_LAST_MOVE.search(text),
        _ASK_OPPONENT_MOVE.search(text),
        _ASK_REVIEW.search(text),
        _ASK_COST.search(text),
        re.search(r"\b(?:why|how come|what should|what do i|explain|plan)\b", text, re.I),
        _SAN.search(text),
    )
    # Only the words beyond the question itself can carry a claim: "winning" in "am I
    # winning?" is the question.
    rest = text
    for pattern in (_ASK_MATERIAL, _OPENING_FACT, _ASK_EVAL):
        rest = pattern.sub(" ", rest)
    return not any(others) and not may_carry_claims(rest)


def may_carry_claims(text: str) -> bool:
    """Whether the user's words could state something about the board to check. "What is he
    threatening?" states nothing: it is answered from the facts' threats, not examined as a
    named line (the examination read it back, question and move numbers, as the answer)."""
    if _ASK_HINT.match(text) or _ASK_REVIEW.search(text) or _ASK_THREAT.search(text):
        return False
    return _CLAIM_WORDS.search(text) is not None


# The move asked for, in words that ask for it: "which move" and "the answer" said of a move
# already played -- "which move did they just play?", "what was the answer to their last
# threat?" -- opened the move level, and the answer was given (the coach review, 2026-10-01).
_ASK_MOVE = re.compile(
    r"\b((?<!\bwas the )(?<!\bwas that the )(?<!\bwas it the )best move|"
    r"what (should|do|would) i (play|do)|what would you play|"
    r"which move(?! (?:did|was|were|had|has|have|do|does) (?:they|he|she|i|it|that|the opponent|"
    r"my opponent)\b)|"
    r"just tell me|what'?s the move|tell me the move|(?<!\bwas )(?<!\bwere )the answer)\b",
    re.I,
)
# How the position goes on is the line: "how would this position continues?" and "ok and after
# that?" after a puzzle's last move unlocked nothing (Tai's puzzle, 2026-09-28). "After that"
# only as a question: in a statement ("and after that I lost my queen") it asks for nothing.
# A line asked for as one -- "the line", "some lines", "the main variation" -- not the word: "are
# there any open lines for my rooks?" opened the line (the coach review, 2026-10-01).
_ASK_LINE = re.compile(
    r"\b(?:the|a|some|main|best|engine'?s?)\s+(?:(?:main|best|whole|full|engine'?s?|critical|"
    r"forcing|winning)\s+)?(?:lines?|variations?|continuations?)\b"
    r"|\b(?:what happens (?:after|next)|"
    r"and then what|what follows|what(?:'s| is| comes) next|then what)\b"
    r"|\bafter that\s*\?"
    r"|\bhow (?:would|will|does|do|could|can) (?:it|this|that|the game|(?:this|the) position|"
    r"things) (?:continue|continues|go on|goes on|play out|proceed)\b",
    re.I,
)
_SAN = re.compile(r"\b(O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b")
# Left or right as a direction ("to the right", "on the left side"), not "is that right?".
_SIDEWAYS = re.compile(
    r"\b(?:to|on|toward|towards) (?:the|my|his|her|their) (?:left|right)\b"
    r"|\b(?:left|right)(?:wards?|-hand| side| wing)\b",
    re.I,
)


def unlock_for(text: str, board: chess.Board, candidates: Sequence[str]) -> Unlock | None:
    """The unlock the user's words earn, if any."""
    if _ASK_LINE.search(text):
        return Unlock.ASK_LINE
    if _ASK_MOVE.search(text):
        return Unlock.ASK_MOVE
    if _HINT.search(text):
        return Unlock.HINT
    for token in _SAN.findall(text):
        try:
            move = board.parse_san(token)
        except ValueError:
            continue
        if board.san(move) in candidates:
            return Unlock.USER_NAMED_MOVE
    return None


def _in_words(board: chess.Board, san: str) -> str:
    """A SAN move described in words, or the SAN itself if the board cannot play it."""
    try:
        return describe_move(board, board.parse_san(san))
    except ValueError:
        return san


class Coach:
    """The coach in a game: takes events and the user's words, hands back messages."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        policy: InterjectionPolicy | None = None,
        draft_check: DraftCheck | None = None,
        claims: Callable[[str, chess.Board, PositionFacts | None], list[str]] | None = None,
        verifier: LLMProvider | None = None,
        claim_reader: LLMProvider | None = None,
        engine: NamedEngine | None = None,
        summarise_every: int = 0,
        summariser: LLMProvider | None = None,
        compact_record: bool = False,
        routine_provider: LLMProvider | None = None,
        verbalized_sampling: bool = True,
        local_verdicts: bool = True,
        persona: str | None = None,
        rating: tuple[int, int] | None = None,
        lesson_wait_s: float = 0.0,
    ) -> None:
        self.policy = policy or InterjectionPolicy()
        # The lesson on a mistake, held until the position the move left is analysed (or
        # LESSON_WAIT_S passes, or the game moves on): the app waits, the tests do not.
        self.lesson_wait_s = lesson_wait_s
        self._held_lesson: tuple[float, _Position, MoveAssessment, MovePlayed, bool] | None = None
        # One reply instead of a verbalized distribution (see config.CoachSettings).
        self.verbalized_sampling = verbalized_sampling
        # The word on a sound move written from the board (coach/verdicts.py) instead of
        # asked of the model: free, immediate, true by construction (Tai, 2026-09-20).
        self.local_verdicts = local_verdicts
        self.speaker = Speaker(
            provider,
            draft_check=draft_check,
            verifier=verifier,
            routine_provider=routine_provider,
            claim_reader=claim_reader,
        )
        # The panel's voice, when it wants replies as they are written: a factory giving,
        # for a trigger and a ply, the sink the reply's sentences go to, or None.
        self.stream_sink: Callable[[Trigger, int | None], Callable[[str], bool] | None] | None = (
            None
        )
        self.claims = claims
        # Dialogue compaction: every `summarise_every` plies the dialogue older than the
        # last `summarise_every` plies is folded into a summary by `summariser` (the
        # coach's own model when None). 0 is off.
        self.summarise_every = summarise_every
        self.summariser = summariser or provider
        # The engine that analyses a move or line the player names, on demand; without
        # one, a named move is only checked against the candidates the facts hold.
        self.engine = engine
        self.reword(persona, rating)
        # Run the claims check on the caller's thread (replay and evaluation) instead of a
        # worker: live it is threaded so a button press never stalls the pipeline.
        self.sync_claims = False
        # The same for the dialogue summary: live it folds in the background, a few seconds
        # after the move; in evaluation it folds before the next moment, so runs repeat.
        self.sync_summaries = False
        self.game_id: str | None = None
        self.user_side: Side = Side.WHITE
        self.finished = False  # the game is over: everything may be said
        self.board = chess.Board()
        self.positions: dict[int, _Position] = {}
        self.moves: dict[int, MovePlayed] = {}
        # The game as the model sees it: one message per move, the coach's turns between.
        self.conversation = Conversation(compact_record=compact_record)
        self.spoken: list[Spoken] = []
        self._current: _Position | None = None
        self._clock_peak_ms: int | None = None
        # The conversation is read from the caller's thread and from a claims check.
        self._lock = threading.RLock()
        # Every wording drawn like a marble from a bag, each once before any comes again.
        self._bags = Bags()
        # The kinds of lesson the last sound moves' words taught (verdicts.lesson).
        self._taught: deque[str] = deque(maxlen=TAUGHT_MEMORY)
        # The coach's messages so far this session, and the rules of thumb said, each with
        # the count it was said at: kept across games and puzzles (RULE_COOLDOWN).
        self._messages = 0
        self._rules: dict[str, int] = {}
        self._last_rule: int | None = None
        # The last template said from each rotating list, keyed by the list.
        # What the game is (GameStarted.kind): a game from the start, or a position -- a
        # puzzle, an analysis board -- where the mating move gets no word and the end no
        # summary (Tai, 2026-09-20: a solved puzzle is left, not summed up); every other
        # move gets its word or its lesson as in a game.
        self.kind = "game"
        # The user's last few think times, for the game's pace (see FAST_PACE_S).
        self._think_s: deque[float] = deque(maxlen=PACE_MOVES)
        # The player model as the profile stage last reported it, and the review of the
        # game just ended, for the coach's summary.
        self.profile: ProfileSnapshot | None = None
        self._review_lines: list[str] = []
        # The causes of the session's mistakes, from the player model: the pattern is said
        # when one comes back (`_on_evidence`).
        self._causes_seen: Counter[str] = Counter()
        # The causes whose pattern has been said: once a session.
        self._pattern_told: set[str] = set()
        # The record's question-answerer (profile/ask.py), set by whoever owns the store;
        # the answers it gave, with their queries, for the panel's log.
        self.profile_ask: Callable[[str], Any] | None = None
        self.profile_answers: list[tuple[str, Any]] = []
        # The prompts that failed their check before going to the model (`harness_problem`),
        # for the panel's log: each one is a bug in how the prompt is built.
        self.harness_problems: list[str] = []
        self._mistakes = 0  # the player's mistakes this game, for ELICIT_EVERY
        # When the coach last asked what the player saw, and whether they answered.
        self._elicit_asked: float | None = None
        # A question asked before the engine had looked at the position on the board: it is
        # answered the moment the analysis lands (see `ask`).
        self._pending_ask: tuple[str, float] | None = None
        self._elicit_answered = False
        # The moment of the game: a new game, a takeback and a re-read each start another. A
        # reply asked for in an earlier one and landing in this one is not a turn of this one:
        # recorded where it landed, the lesson on a withdrawn move stood under "Said here" of
        # the position taken back to, and a reply about the last puzzle under the next one's,
        # where they made the player's next words a follow-up or the answer to their question
        # (the coach review, 2026-10-01).
        self._epoch = 0

    # Lifecycle

    def start(self) -> None:
        self.speaker.start()

    def close(self) -> None:
        self.speaker.close()

    @property
    def user_colour(self) -> chess.Color:
        return self.user_side.color

    @property
    def user_to_move(self) -> bool:
        return self.board.turn == self.user_side.color

    # Events in

    def on_event(self, event: Event, ts: float) -> None:
        if isinstance(event, GameStarted):
            self._new_game(event, ts)
        elif isinstance(event, MovePlayed):
            self._on_move(event, ts)
        elif isinstance(event, Analysis):
            self._on_analysis(event, ts)
        elif isinstance(event, MoveAssessment):
            self._on_assessment(event, ts)
        elif isinstance(event, ClockObservation):
            self._on_clock(event, ts)
        elif isinstance(event, Takeback):
            self._held_lesson = None  # the move it was on is gone
            self._on_takeback(event, ts)
        elif isinstance(event, Resync):
            self._held_lesson = None
            self._on_resync(event, ts)
        elif isinstance(event, GameEnded):
            self._release_lesson(ts)
            self._current = None
            self.finished = True
            self.conversation.ended(event)
        elif isinstance(event, Recommendation):
            self._on_recommendation(event, ts)
        elif isinstance(event, PuzzleOutcome):
            self._on_puzzle_outcome(event, ts)
        elif isinstance(event, GameReview):
            self._on_review(event, ts)
        elif isinstance(event, ProfileEvidence):
            self._on_evidence(event, ts)
        elif isinstance(event, ProfileSnapshot):
            self.profile = event
        elif isinstance(event, ScreenStateChanged):
            covered = event.state is ScreenState.COVERED
            if covered and not self.policy.covered:
                self.conversation.covered()
            self.policy.covered = covered

    @property
    def quiet(self) -> bool:
        """Nothing unasked -- no word on a move, no nudge, no hint on its own -- the
        policy's quiet mode, turned on while a puzzle is worked on the panel's board
        (Tai: "no auto hint or anything from the coach"). A question of the player's is
        answered as ever."""
        return self.policy.settings.quiet

    @quiet.setter
    def quiet(self, value: bool) -> None:
        self.policy.settings.quiet = value

    def tick(self, ts: float) -> None:
        """Time passing: a held lesson once its wait is up; a hint if the user has been stuck
        long enough."""
        if self._held_lesson is not None and ts - self._held_lesson[0] >= self.lesson_wait_s:
            self._release_lesson(ts)
        trigger = self.policy.tick(ts)
        if trigger is None or self._current is None or self._current.facts is None:
            return
        if self._current.board.is_game_over():
            return  # nothing to be stuck on: the game is over on the board
        # A hint unasked only ever climbs: at the top of its ladder it is not said again.
        self._current.disclosure.carry(self._ladder(self._current))
        level = self._current.disclosure.apply(Unlock.STUCK)
        if level is None:
            self.policy.exhausted(self._current.ply)
            return
        self.policy.said(self._current.ply, ts)
        self._speak(trigger, self._current, level, ts)

    def ask(self, text: str, ts: float) -> None:
        """The user said something. Always answered, at the level their words earn.

        Returns at once. A canned ask -- a hint, the move, the line -- gets the template
        for its level immediately, with its marks, so the answer is on the board before the
        model has started; the model's fuller reply follows. Words that may carry a claim
        about the board are checked first, on a worker thread: the check is a model call of
        its own, and run on the caller's thread it stalled the whole pipeline for seconds
        (the first live games: every button press froze the vision until it returned).
        """
        if self.profile_ask is not None and _ASK_PROFILE.search(text):
            self._ask_profile(text, ts)
            return
        if self._elicit_asked is not None and ts - self._elicit_asked <= ELICIT_ANSWER_S:
            self._elicit_answered = True
        position = self._current
        if (
            position is not None
            and position.facts is None
            and self.user_to_move
            and not self.finished
        ):
            # The engine is still looking at the position on the board: said so, and
            # answered the moment its analysis lands -- never from an older position's
            # (Tai, 2026-09-24: a hint pressed as a puzzle appeared got "I have not seen
            # enough of this game yet to say. Give me a move or two.").
            if self._pending_ask is None:
                self._queue_plain(
                    Trigger.USER_ASK, "One moment -- still looking at this position.", ts
                )
            self._pending_ask = (text, ts)
            return
        if position is None or position.facts is None:
            position = self._latest_with_facts()
        if position is None or position.facts is None:
            self._queue_plain(
                Trigger.USER_ASK,
                "I have not seen a position to talk about yet.",
                ts,
            )
            return
        unlock = unlock_for(text, position.board, position.facts.candidates)
        # A hint asked for climbs from what the combination has had already.
        position.disclosure.carry(self._ladder(position))
        if unlock is not None:
            position.disclosure.apply(unlock)
        level = max(position.disclosure.level, DisclosureLevel.IDEA)
        if self.finished or position.board.is_game_over():
            # Nothing is withheld from a game that is over.
            level = DisclosureLevel.LINE
        with self._lock:
            self.conversation.user_said(text)
        if (own := self._asked_own_move(text)) is not None:
            # About the move they already played: spoken from the position it was played
            # from, with everything shown, as the lesson after a mistake is. That position
            # is past; the level in force binds only the one they face now.
            before = self.positions.get(own.move.ply - 1)
            if before is not None and before.facts is not None:
                position, level = before, DisclosureLevel.LINE
        assert position.facts is not None
        # With a model, every question is the model's to answer, the buttons' included:
        # its answer starts within a second or two (Tai, 2026-09-28: "I'm using AI coach so
        # I better be able to ask whatever I want and get answers for them ... it's already
        # almost instant"). Asked "what should I play against this opening? give me a line",
        # the template for the opening's name was all he got. The templates are the offline
        # coach's, and the model's fallback when a reply gets something wrong.
        offline = self._offline
        # Offline, the answer at once: the template for a bare "hint", the move or the line.
        # The move or the line asked for, or asked about where it is already open: "what
        # happens after the best move?" at the line level is the line, though it unlocks
        # nothing (the harness check of the test audit, 2026-09-26).
        exact = unlock in (Unlock.ASK_MOVE, Unlock.ASK_LINE) or (
            (level >= DisclosureLevel.LINE and _ASK_LINE.search(text) is not None)
            or (level >= DisclosureLevel.MOVE and _ASK_MOVE.search(text) is not None)
        )
        # A claim to check needs something named: "what happens after the best move?" names
        # no move or square, whatever its words.
        named = may_carry_claims(text) and bool(
            _SAN.search(text) or re.search(r"\b[a-h][1-8]\b", text)
        )
        canned = offline and (
            (exact and not named)
            or (
                (
                    bool(_ASK_HINT.match(text))
                    # A question about the plan: the first hint is the lesson it asks for
                    # (the stakes, the key fact, the principle), as for a bare hint.
                    or bool(_ASK_AIM.search(text))
                )
                and not may_carry_claims(text)
            )
        )
        if canned:
            _, answer = self._instruction(Trigger.USER_ASK, position, level, text, (), None, None)
            self._take_leads(position, answer)
            self._queue_plain(
                Trigger.USER_ASK,
                answer,
                ts,
                ply=position.ply,
                level=level,
                annotations=annotations_for(
                    position.facts, level, position.board, trigger=Trigger.USER_ASK
                ),
                fen=position.board.fen(),
                now=True,
                record_now=True,
            )
            return
        if offline and fact_question(text):
            # The material, the opening, the standing: said by the template, exactly.
            _, answer = self._instruction(Trigger.USER_ASK, position, level, text, (), None, None)
            self._queue_plain(
                Trigger.USER_ASK,
                answer,
                ts,
                ply=position.ply,
                level=level,
                fen=position.board.fen(),
                now=True,
                record_now=True,
            )
            return

        asked = question_part(text)
        shown_examination = False

        def examined() -> list[str]:
            """The engine on what they asked about: the model's to answer from, given with
            the request. Offline its plain-words form is the answer, said as the coach's
            turn (correct by construction): only what a question names -- named in a
            statement ("I see the fork now: Nf6 attacks the queen on e8 and the rook on d7")
            or before the question ("my rook on e1 stays active. Why is 12...h5 their best
            reply?"), the pieces and squares came back as the whole answer and the question
            went unanswered (the simulated students, 2026-09-26)."""
            nonlocal shown_examination
            found = self._named_analysis(asked or text, position, level)
            if not offline:
                return found  # the model answers, from these
            # A move named in their words that is no legal move is the model's
            # to know, not the player's to hear: "after pass, Kd1 is not a legal move" was
            # read out as the answer to "How does Qc7+ prevent Re8?" (the students).
            said = [f for f in found if "is not a legal move" not in f]
            # A move they propose in a statement ("I'll play Qd1, blocking the check") is
            # examined aloud as a question about it would be: that is the answer. Without it
            # the templates said the hint that had asked for it again, word for word, three
            # times (the students' round 5); pieces and squares named in passing stay unsaid.
            if not asked:
                said = [f for f in said if "the expected score is" in f]
            if said:
                shown_examination = True
                self._queue_plain(
                    Trigger.USER_ASK,
                    " ".join(spoken_examination(f) for f in said),
                    ts,
                    ply=position.ply,
                    level=level,
                    fen=position.board.fen(),
                    now=True,
                    record_now=True,
                )
            return found

        # With the engine at the model's call its claims are the model's to check, with the whole
        # conversation in view: checked beforehand against the board as it stands, "Rxf1+ check
        # the white king" -- the move just played -- came back "false, Rxf1+ is not a move you
        # can play next" (Tai's puzzle, 2026-09-28).
        tools = self.engine is not None and not offline
        if self.claims is None or tools or not may_carry_claims(text):
            verdicts = examined() if may_carry_claims(text) else []
            if not shown_examination:  # offline, the engine's word on it was the answer
                self._speak(
                    Trigger.USER_ASK, position, level, ts, user_text=text, verdicts=verdicts
                )
            return

        def checked() -> None:
            try:
                verdicts = self.claims(text, position.board, position.facts)  # type: ignore[misc]
            except Exception:  # noqa: BLE001
                verdicts = []
            found = examined()
            if shown_examination:
                return  # offline, the engine's word on it was the answer
            self._speak(
                Trigger.USER_ASK,
                position,
                level,
                ts,
                user_text=text,
                verdicts=[*verdicts, *found],
            )

        if self.sync_claims:
            checked()  # replay and evaluation: the request must exist when ask() returns
        else:
            threading.Thread(target=checked, name="sidecoach-claims", daemon=True).start()

    def _ask_profile(self, text: str, ts: float) -> None:
        """A question about the player over their games: answered from the record by the
        profile's own model loop, off this thread (several model calls), and spoken as
        the coach's turn when it lands. The queries it ran are kept for the log."""
        assert self.profile_ask is not None
        with self._lock:
            self.conversation.user_said(text)
        ask = self.profile_ask

        def run() -> None:
            try:
                answer = ask(text)
            except Exception as exc:  # noqa: BLE001 - the record must not take the coach down
                self._queue_plain(
                    Trigger.USER_ASK,
                    "I could not read the record just now; ask me again in a moment.",
                    ts,
                    record_now=True,
                )
                self.profile_answers.append(("error", repr(exc)))
                return
            with self._lock:
                self.conversation.note(
                    f"The coach looked the question up in the record ({len(answer.exchanges)} "
                    f"quer{'y' if len(answer.exchanges) == 1 else 'ies'})."
                )
            self.profile_answers.append(("answer", answer))
            self._queue_plain(Trigger.USER_ASK, answer.text, ts, record_now=True)

        threading.Thread(target=run, name="profile-ask", daemon=True).start()

    def hint(self, ts: float) -> None:
        self.ask("hint", ts)

    def poll(self) -> list[CoachMessage]:
        out: list[CoachMessage] = []
        for spoken in self.speaker.poll():
            self.spoken.append(spoken)
            if not spoken.message.text:
                # A failed model reply with nothing to fall back on: kept for the log (its
                # calls say why), said to no one.
                continue
            if not spoken.recorded and spoken.epoch == self._epoch:
                with self._lock:
                    self.conversation.coach_said(
                        spoken.message.text, nod=spoken.message.trigger is Trigger.POST_MOVE
                    )
            self._messages += 1
            out.append(spoken.message)
        return out

    # Internals

    def reword(self, persona: str | None, rating: tuple[int, int] | None) -> None:
        """The system message and the templates' register for the player: set at the start
        and again when the persona or the rating is saved (the register was kept until the
        coach was rebuilt: the parity audit, 2026-09-26)."""
        self.system = system_message(persona=persona, rating=rating)
        # A player below 1400 hears no term of art in the templates (verdicts.JARGON).
        self.plain_words = rating is not None and rating[1] < PLAIN_BELOW
        # A player of 1800 and up hears no textbook rule: the strong simulated students listed
        # about 4.5 a session they already knew ("rooks belong on open files", 75 times).
        self.strong_player = rating is not None and rating[0] >= STRONG_FROM

    def _new_game(self, event: GameStarted, ts: float) -> None:
        self._epoch += 1
        # A lesson held for its analysis is the last game's: kept, the next tick said it in
        # this one, over this one's first position (the coach review, 2026-10-01).
        self._held_lesson = None
        self.game_id = event.game_id
        self.user_side = event.user_side
        self.kind = event.kind
        self.board = chess.Board(event.start_fen)
        self.positions = {0: _Position(ply=0, board=self.board.copy())}
        self.moves = {}
        self.conversation.new_game(event, profile=self._profile_lines())
        self._think_s.clear()
        self._mistakes = 0
        # The review is the game's own: a new game answered "how did I do overall?" with
        # the last game's sentences when the model failed (the parity audit, 2026-09-26).
        self._review_lines = []
        self._elicit_asked, self._elicit_answered = None, False
        self._pending_ask = None
        self.finished = False
        self._current = self.positions[0]
        self.policy.restart()
        self.policy.new_position(0, ts, user_to_move=self.user_to_move)
        self._clock_peak_ms = None
        self.policy.set_remaining(None)
        self.policy.time_control = TimeControl.UNTIMED

    def _profile_lines(self) -> list[str]:
        """What the player model says, for the setting of a new game: the insights, and
        the last session in a sentence."""
        if self.profile is None:
            return []
        lines = list(self.profile.insights)
        if self.profile.last_session:
            lines.append(f"Last session: {self.profile.last_session}")
        return lines

    def _on_review(self, event: GameReview, ts: float) -> None:
        """The game is over and reviewed: the review joins the conversation and the coach
        sums the game up from it -- through the model when a position with facts is there
        to check the reply against, else the review's own sentences."""
        if event.game_id != self.game_id:
            return
        lines = list(event.lines)
        if self.strong_player:
            # The pattern and its count, not the textbook habit: a 2000 player was told "bring
            # your King to the middle and push your passed pawns" (the AI's sessions).
            lines = [re.sub(r" The habit that fixes it: .*$", "", line) for line in lines]
        with self._lock:
            self.conversation.review(lines)
        position = self._latest_with_facts()
        if position is None or not lines:
            self._queue_plain(Trigger.REVIEW, " ".join(lines[:4]), ts, record_now=True)
            return
        self._review_lines = lines
        self._speak(Trigger.REVIEW, position, DisclosureLevel.LINE, ts)

    def _on_evidence(self, event: ProfileEvidence, ts: float) -> None:
        """A mistake of the player's placed by the player model. The third time one cause
        is behind the session's mistakes, the pattern is said once, with the habit that
        answers it, right after the lesson on the move; the game's review sums it up after
        that. Each lesson had taught its own moment and a principle of its own, and the
        threat-blind student's notebook held five unrelated rules, none of them the habit
        (the simulated students, 2026-09-26); said the second time and then in every game,
        "this keeps happening -- 5 times now" came on top of a lesson (Tai's phone,
        2026-09-27). A player of 1800 and up hears the pattern without the textbook habit.
        Said from the model's cause, no model call."""
        if event.game_id != self.game_id or event.cause not in HABIT:
            return
        self._causes_seen[event.cause] += 1
        if self._causes_seen[event.cause] < 3 or event.cause in self._pattern_told:
            return
        self._pattern_told.add(event.cause)
        words, habit = HABIT[event.cause]
        text = f"That is three times now: {words}."
        if not self.strong_player:
            text += f" The habit to build: {habit}."
        if not self._rule_ok(None):
            return  # a rule of thumb just said: the review carries the pattern
        self._rule_said(f"habit:{event.cause}")
        with self._lock:
            self.conversation.note(f"The pattern, told to the player: {text}")
        self._queue_plain(Trigger.POST_MISTAKE, text, ts, record_now=True)

    def _on_recommendation(self, event: Recommendation, ts: float) -> None:
        """The practice the profile picked after the review: a line naming the motif and
        the puzzles, on the card; no model call, the card carries the links."""
        if event.game_id != self.game_id or not event.puzzles:
            return
        n = len(event.puzzles)
        text = (
            f"For practice: {n} {event.motif_words} puzzle{'s' if n != 1 else ''} around "
            f"{event.level} {'are' if n != 1 else 'is'} on the review card -- {event.reason}."
        )
        with self._lock:
            self.conversation.note(f"Practice recommended: {text}")
        self._queue_plain(Trigger.REVIEW, text, ts, record_now=True)

    def _on_puzzle_outcome(self, event: PuzzleOutcome, ts: float) -> None:
        """A puzzle from the store worked through on the screen: a word on a fail (the
        site says nothing useful), nothing on a solve (the site does)."""
        with self._lock:
            self.conversation.note(
                f"Puzzle {event.puzzle_id} {'solved' if event.solved else 'failed'}; "
                f"puzzle level now {event.level}."
            )
        if event.solved or self.quiet:
            return  # on the panel's own board the bar says it; nothing unasked
        motif = f" That one was a {motif_words(event.motif)} puzzle." if event.motif else ""
        self._queue_plain(
            Trigger.POST_MISTAKE,
            f"Not that one -- have another look before the site shows you.{motif}",
            ts,
            record_now=True,
        )

    def _on_clock(self, event: ClockObservation, ts: float) -> None:
        """The screen's clock, when one is drawn. It scales the interjection cadence: the
        stuck-hint threshold shrinks as the user's clock runs down, and the time control
        decides which unprompted triggers fire at all (rapid gets everything, blitz only
        brief feedback, bullet nothing). A missing, hidden, or unreadable clock never
        reaches here, so the policy simply keeps its untimed defaults.
        """
        user_ms = event.white_ms if self.user_side == Side.WHITE else event.black_ms
        if user_ms is not None:
            self.policy.set_remaining(user_ms)
        self.conversation.clock(event.white_ms, event.black_ms)
        seen = [ms for ms in (event.white_ms, event.black_ms) if ms is not None]
        if not seen:
            return
        peak = max(seen)
        self._clock_peak_ms = (
            peak if self._clock_peak_ms is None else max(self._clock_peak_ms, peak)
        )
        # The clock starts at its highest, so the peak seen approximates the initial time
        # and thus the control's band; increment slack does not cross the coarse bands.
        self.policy.time_control = time_control_of(self._clock_peak_ms)

    @property
    def pace_s(self) -> float | None:
        """The user's recent think time in this game, in seconds: the median of its last
        few moves, or None before there are two to go by."""
        if len(self._think_s) < 2:
            return None
        return statistics.median(self._think_s)

    @property
    def fast_pace(self) -> bool:
        """Whether the game is being played faster than the coach can keep up with a
        model and a voice: templates and text until the player slows down."""
        pace = self.pace_s
        return pace is not None and pace < FAST_PACE_S

    def fast_now(self, ts: float) -> bool:
        """The pace is fast and the player has not slowed down in this position: once
        they have been thinking longer than the fast pace, a hint or a lesson may take
        the model's time and the voice again."""
        if not self.fast_pace:
            return False
        since = self.policy.position_since
        return not (self.policy.user_to_move and since is not None and ts - since >= FAST_PACE_S)

    def _release_lesson(self, ts: float) -> None:
        """The held lesson on a mistake, spoken now."""
        held, self._held_lesson = self._held_lesson, None
        if held is None:
            return
        _, position, assessment, played, elicit = held
        self._speak(
            Trigger.POST_MISTAKE,
            position,
            DisclosureLevel.LINE,
            ts,
            assessment=assessment,
            played=played,
            elicit=elicit,
        )

    def _on_move(self, event: MovePlayed, ts: float) -> None:
        # The game moved on: a lesson still waiting on its analysis goes now, before the
        # new move is in the record.
        self._release_lesson(ts)
        if event.by_user and event.think_ms is not None:
            self._think_s.append(event.think_ms / 1000.0)
        if event.by_user:
            self._pending_ask = None  # they played on: the question is answered by the move
        self.board = board_after(self.board, event)
        self.moves[event.ply] = event
        self.conversation.move(event)
        position = _Position(ply=event.ply, board=self.board.copy(), last_move=event.san)
        self.positions[event.ply] = position
        # Kept for a while: a verdict lands a few plies late when the engine is behind
        # (2026-09-22: at six, the verdicts of plies 82-126 found no position and the
        # coach said nothing, not even in the log).
        for old in [ply for ply in self.positions if ply < event.ply - POSITIONS_KEPT]:
            del self.positions[old]
        self._current = position
        self.policy.new_position(event.ply, ts, user_to_move=self.user_to_move)
        if self.summarise_every and event.ply % self.summarise_every == 0:
            if self.sync_summaries:
                self._summarise()
            else:
                threading.Thread(
                    target=self._summarise, name="sidecoach-summary", daemon=True
                ).start()

    def _summarise(self) -> None:
        """Fold the earlier dialogue into the summary. The model call runs outside the
        lock (seconds); the fold itself takes it."""
        try:
            summarise(
                self.conversation, self.summariser, keep=self.summarise_every, lock=self._lock
            )
        except Exception as exc:  # noqa: BLE001
            self.speaker.last_error = f"summary: {type(exc).__name__}: {exc}"

    def _on_takeback(self, event: Takeback, ts: float) -> None:
        """Moves withdrawn: the conversation says so, and the game goes on from the position
        the board is back at; the replacement moves arrive as ordinary moves."""
        self._epoch += 1
        self.conversation.takeback(event)
        self.board = board_back(self.board, event.fen)
        active = self.conversation.active_records()
        ply = active[-1].move.ply if active else 0
        for gone in [p for p in self.positions if p > ply]:
            del self.positions[gone]
        # The withdrawn moves go too: a late word on one was taken for the move that replaced it.
        for gone in [p for p in self.moves if p > ply]:
            del self.moves[gone]
        position = self.positions.get(ply)
        if position is None:
            position = _Position(ply=ply, board=self.board.copy())
            self.positions[ply] = position
        self._current = position
        self.policy.new_position(ply, ts, user_to_move=self.user_to_move)

    def _on_resync(self, event: Resync, ts: float) -> None:
        """The board was re-read: the position jumped; the move numbering restarts."""
        self._epoch += 1
        self.board = chess.Board(event.fen)
        self.conversation.resync(event, self.board)
        self.positions = {0: _Position(ply=0, board=self.board.copy())}
        self.moves = {}
        self._current = self.positions[0]
        self._pending_ask = None
        self.policy.restart()
        self.policy.new_position(0, ts, user_to_move=self.user_to_move)

    def _on_analysis(self, event: Analysis, ts: float) -> None:
        if event.game_id != self.game_id:
            return  # the engine's late word on the game before this one
        position = self.positions.get(event.ply)
        if position is None or position.board.fen() != event.fen:
            return
        full = position.analysis is not None and not position.analysis.provisional
        if event.provisional and full:
            return  # the full depth is in already
        position.analysis = event
        leads: dict[str, str] = {}

        def peek(key: str, options: Sequence[str]) -> str:
            leads[key] = self._bags.peek(key, options)
            return leads[key]

        position.facts = facts_from_analysis(
            position.board, event, last_move=position.last_move, user=self.user_colour, pick=peek
        )
        position.leads = leads
        # The record shows what the coach answers from, the early depth until the full one
        # replaces it: kept back until the full depth, a question answered from the early
        # one was asked of the model with "Engine lines: not in yet." and no facts (the coach
        # review, 2026-10-01).
        self.conversation.analysed(event, position.facts)
        if event.provisional:
            # Measured early, the full depth to follow: what the player asks is answered
            # from it now; the coach's own words -- a lesson -- wait for the full depth.
            if position is self._current and self._pending_ask is not None:
                text, _asked = self._pending_ask
                self._pending_ask = None
                self.ask(text, ts)
            return
        if self._held_lesson is not None and self._held_lesson[3].ply == event.ply:
            self._release_lesson(ts)  # what the move allowed is known now
        if position is self._current and self._pending_ask is not None:
            text, _asked = self._pending_ask
            self._pending_ask = None
            self.ask(text, ts)
        # Nothing is said of its own at a decisive moment: while the player thinks, the
        # first word unasked is the stuck hint (Tai, 2026-10-01: nothing "until the user is
        # stuck or actively request a hint or after they have made a move").

    def _ladder(self, position: _Position) -> DisclosureLevel:
        """The highest hint the combination this position continues has had: back through
        the player's positions while the move they played there was the engine's best, the
        levels said before they moved (Tai, 2026-09-29: "Each level can only be said once
        and if the level higher was said then the lower one can't be said"). A move off the
        best line starts a new one; a puzzle's wrong moves are taken back."""
        level = DisclosureLevel.SILENT
        ply = position.ply
        while True:
            own = self.moves.get(ply - 1)
            before = self.positions.get(ply - 2)
            if own is None or not own.by_user or before is None or not self._was_best(before, own):
                return level
            level = max(level, before.disclosure.hinted)
            ply -= 2

    @staticmethod
    def _was_best(position: _Position, move: MovePlayed) -> bool:
        """Whether `move` was the engine's best from `position`."""
        analysis = position.analysis
        if analysis is not None and analysis.lines and analysis.lines[0].moves:
            return analysis.lines[0].moves[0] == move.uci
        facts = position.facts
        return facts is not None and bool(facts.candidates) and facts.candidates[0] == move.san

    def _on_assessment(self, event: MoveAssessment, ts: float) -> None:
        if event.game_id != self.game_id:
            return  # the engine's late word on the game before this one
        move = self.moves.get(event.ply)
        if move is not None and move.uci != event.uci:
            return  # the verdict on a move taken back since, not on the one played there now
        self.conversation.assess(event)
        if move is None or not move.by_user:
            return
        before = self.positions.get(event.ply - 1)
        if before is None or before.facts is None:
            # Nothing to speak from: the position is gone, or its analysis never came.
            # Said in the log, never silently.
            why = "forgotten" if before is None else "not analysed"
            self.policy.last_reason = (
                f"no word on ply {event.ply}: the position before it was {why}"
            )
            return
        trigger = self.policy.on_user_move_assessed(event)
        if trigger is None:
            return
        elicit = False
        if trigger is Trigger.POST_MISTAKE:
            self._mistakes += 1
            elicit = self._elicit_due(ts)
        before.disclosure.apply(Unlock.MOVE_PLAYED)
        after = self.positions.get(event.ply)
        if self.kind != "game" and after is not None and after.board.is_checkmate():
            return  # the puzzle is solved with a mate: the site says so, and moves on
        # Every move of the player's gets a word from the coach. A lecture is only for a
        # move that cost something. Theory, the engine's own move and an excellent one get
        # a sentence written from the board (coach/verdicts.py: a word of praise, and what
        # the move teaches when the board shows it), no model call; in blitz too. A great
        # or brilliant move gets the model's delight at its full effort; a good move the
        # model's word on the better one; the board's sentence stands in if the reply fails.
        if trigger is Trigger.POST_MOVE and event.severity is Severity.NONE:
            assert before.facts is not None
            best = before.facts.candidates[0] if before.facts.candidates else None
            routine = best is None or best == move.san or _local(event.label)
            blitz = self.policy.time_control is TimeControl.BLITZ
            # Great and brilliant moves get the model's delight (Tai, 2026-09-26 18:57: "They
            # need to go through the api"), though its delight carried a false claim in 18% of
            # them on the locked test set: the draft checks stand guard over it.
            local = routine and not _exciting(event.label) and (self.local_verdicts or blitz)
            if not local:
                self._speak(
                    trigger,
                    before,
                    DisclosureLevel.LINE,
                    ts,
                    assessment=event,
                    played=move,
                    candidates=VS_ROUTINE_CANDIDATES if routine else VS_CANDIDATES,
                    routine=routine and not _exciting(event.label),
                )
                return
            label = event.label.value if event.label else None
            if self.local_verdicts:
                text, kind = local_verdict(
                    before.board,
                    chess.Move.from_uci(move.uci),
                    label,
                    event.tags,
                    phase=before.facts.phase,
                    threats=before.facts.threats,
                    taught=self._verdict_taught(),
                    bags=self._bags,
                    plain=self.plain_words,
                    strong=self.strong_player,
                )
                self._taught_by_verdict(kind)
            else:
                text = self._verdict(label)
            marks = annotations_for(
                before.facts,
                DisclosureLevel.LINE,
                before.board,
                trigger=trigger,
                played_san=move.san,
                severity=event.severity.value,
            )
            self._queue_plain(
                trigger,
                text,
                ts,
                ply=before.ply,
                level=DisclosureLevel.LINE,
                annotations=marks,
                fen=before.board.fen(),
            )
            return
        if (
            trigger is Trigger.POST_MISTAKE
            and self.lesson_wait_s > 0
            and (after is None or after.analysis is None)
        ):
            self._held_lesson = (ts, before, event, move, elicit)
            return
        self._speak(
            trigger,
            before,
            DisclosureLevel.LINE,
            ts,
            assessment=event,
            played=move,
            elicit=elicit,
        )

    def _elicit_due(self, ts: float) -> bool:
        """Whether this mistake's lesson ends by asking what the player saw: every
        ELICIT_EVERY-th mistake, and never again in a game once one such question went
        unanswered -- the player is not talking back, and the question is noise."""
        if self._mistakes % ELICIT_EVERY:
            return False
        if self._elicit_asked is not None and not self._elicit_answered:
            return False
        self._elicit_asked, self._elicit_answered = ts, False
        return True

    def _verdict(self, label: str | None) -> str:
        """The word of praise on a fine move, by its label, from the bag the local verdicts
        draw from too: a game of "Good. Good. Good." wears thin."""
        openers = OPENERS.get(label or "", OPENERS["good"])
        return self._bags.draw(f"praise:{label or 'good'}", openers) + "."

    def set_providers(
        self,
        provider: LLMProvider,
        *,
        routine_provider: LLMProvider | None = None,
        verifier: LLMProvider | None = None,
        summariser: LLMProvider | None = None,
        claims: Callable[[str, chess.Board, PositionFacts | None], list[str]] | None = None,
    ) -> None:
        """A new model mid-game (the settings saved): the next call goes to it; a call in
        flight finishes on the old one. The conversation is untouched."""
        with self._lock:
            self.speaker.provider = provider
            self.speaker.routine_provider = routine_provider or provider
            self.speaker.verifier = verifier
            self.summariser = summariser or provider
            self.claims = claims

    def _take_leads(self, position: _Position, template: str) -> None:
        """The first hint's wordings taken from their bags once the hint is said: this turn's
        template is the position's first hint, or holds it."""
        facts = position.facts
        if (
            facts is not None
            and facts.situation
            and (
                facts.situation in template
                or (facts.principle and _without(facts.situation, facts.principle) in template)
            )
        ):
            for key, wording in position.leads.items():
                self._bags.take(key, wording)
            position.leads = {}

    def _cool_hint(self, facts: PositionFacts, text: str) -> tuple[str, str]:
        """A hint's template with its principle (`PositionFacts.principle`) left out while it
        cools down, and the word to the model that goes with it; counted as said when it
        stays in."""
        principle = facts.principle
        if not principle or principle not in text:
            return text, ""
        if self._rule_ok(principle):
            self._rule_said(principle)
            return text, ""
        note = f' {RULE_HELD_MARK}: leave out "{principle}" -- the player heard it lately.'
        return _without(text, principle), note

    def _fresh_lesson(
        self,
        opener: str,
        lesson: Callable[[bool], str],
        rule: str | None,
        facts: PositionFacts | None = None,
        *,
        asked: bool = False,
    ) -> tuple[str, bool]:
        """`lesson`, opening with `opener`, written with its rule when one may be said, less
        what was said lately (`fresh`), and whether the rule was held back: said within
        RULE_COOLDOWN messages, or another within RULE_SPACING. The rule is the template's
        (`rule`, `rule_of`) or, for the model -- asked for "a pattern to remember" on every
        lesson, it stated one every time -- the pattern it teaches (`facts`' theme or plan).
        A held rule is left out, and the model is told (RULE_HELD). The rule is counted as
        said only when the lesson still says it -- or, with a model, when the model is asked
        for it. `asked`: the player asked about the move, and hears the reason whole, what the
        lesson after it said or not: trimmed of it, "why was that bad?" was answered "f3 was
        better." and nothing more, 61 times in 95 (the offline sweep's bot games, 2026-09-27)."""
        key = rule or (facts.theme if facts is not None else None) or None
        held = key is None or not self._rule_ok(key)
        text = lesson(not held)
        if not asked:
            text = fresh(text, self._recently_said(), opener)
        if not held and rule is not None and _said(rule, text):
            self._rule_said(rule)
        elif not held and key is not None and not self._offline:
            self._rule_said(key)
        return text, held

    @property
    def _offline(self) -> bool:
        """No model: every word is the templates' (llm.provider = none)."""
        return bool(getattr(self.speaker.provider, "offline", False))

    def _rule_ok(self, rule: str | None) -> bool:
        """Whether a rule of thumb may be said now: none said in the last RULE_SPACING
        messages, and this one (by its words, or a sound move's lesson kind) not in the last
        RULE_COOLDOWN. `rule` None asks only the first."""
        if self._last_rule is not None and self._messages - self._last_rule < RULE_SPACING:
            return False
        said = self._rules.get(rule) if rule is not None else None
        return said is None or self._messages - said >= RULE_COOLDOWN

    def _rule_said(self, rule: str) -> None:
        self._rules[rule] = self._last_rule = self._messages

    def _budgeted(self, instruction: str, key: str) -> str:
        """A word on a sound move held to the rule budget: its pattern to keep asked for only
        while a rule may be said (`_rule_ok`), and then counted as said, under `key` -- with a
        model, being asked for it is the saying (as in `_fresh_lesson`)."""
        if not self._rule_ok(None):
            return _hold_rule(instruction)
        if not self._offline:
            self._rule_said(key)
        return instruction

    def _verdict_taught(self) -> tuple[str, ...]:
        """What a sound move's word may not teach now: the kinds of the last few lessons
        (TAUGHT_MEMORY), and every principle that may not be said (`_rule_ok`)."""
        return (*self._taught, *(kind for kind in PRINCIPLES if not self._rule_ok(kind)))

    def _taught_by_verdict(self, kind: str | None) -> None:
        """A sound move's word taught `kind`: remembered for the next words, and counted as
        a rule when it is a principle."""
        if kind is None:
            return
        self._taught.append(kind)
        if kind in PRINCIPLES:
            self._rule_said(kind)

    def _recently_said(self) -> str:
        """What the coach has said in this game, as one text: a reason is not given twice in
        a game. Remembered over six plies, "your pawn on h2 can strike at their pawn on g5"
        was said at move 7 and again at 11 (Tai's phone, 2026-09-27); over none, 27% of the
        lessons came again within three moves (Nova bot games, 2026-09-25)."""
        return " ".join(e.text for e in self.conversation.entries if e.kind == "coach")

    def _opening_so_far(self) -> str | None:
        """The opening the game was in: the last name the book gave."""
        with self._lock:
            return self.conversation.opening()

    def _sheet_lines(self, board: chess.Board | None) -> list[Line]:
        """The engine's lines the record shows for `board`, when it is one of the game's
        positions still kept; none otherwise."""
        if board is None:
            return []
        key = position_key(board)
        with self._lock:
            for position in self.positions.values():
                if position.analysis is not None and position_key(position.board) == key:
                    return list(position.analysis.lines)
        return []

    def _named_analysis(self, text: str, position: _Position, level: DisclosureLevel) -> list[str]:
        """The engine's word on the moves, pieces and squares the player's words name; one
        line per probe, none when nothing was named. A piece or square named needs no
        engine; a line does. What else the words mean is the model's to ask the engine
        about (`engine_tools`)."""
        try:
            return analyse_named(
                text,
                position.board,
                self.engine,
                user_colour=self.user_colour,
                level=level,
                # Pieces and squares as the board stands now, whatever position the question
                # is about.
                pieces_board=self.board,
                # The lines the sheet already shows: a move they start is read from them.
                known=position.analysis.lines if position.analysis is not None else (),
            )
        except Exception as exc:  # noqa: BLE001
            self.speaker.last_error = f"named analysis: {type(exc).__name__}: {exc}"
            return []

    def _said_since_move(self) -> list[str]:
        """What the coach has said since the position now was reached: the last move, or a
        takeback or a re-read of the board (what was said of a withdrawn move is not)."""
        with self._lock:
            entries = list(self.conversation.entries)
        out: list[str] = []
        for entry in reversed(entries):
            if entry.kind in ("ply", "takeback", "resync"):
                break
            if entry.kind == "coach":
                out.append(entry.text)
        return out

    def _following_up(self, said: str) -> bool:
        """Whether the player's words follow up what the coach has said since the last move,
        not an answer to its question: the verdict and the better move were told already,
        and a follow-up gets only what it asks."""
        return not self._answering_our_question(said) and bool(self._said_since_move())

    def _answering_our_question(self, said: str) -> bool:
        """Whether the player's words answer the question the coach just asked: its last
        turn ended with one, nothing has happened since, and their words ask nothing."""
        if question_part(said):
            return False
        with self._lock:
            entries = list(self.conversation.entries)
        # Their words themselves, when ask() has recorded them already, are not the thread.
        if entries and entries[-1].kind == "ask":
            entries = entries[:-1]
        for entry in reversed(entries):
            if entry.kind == "coach":
                return entry.text.rstrip().endswith("?")
            if entry.kind != "note":
                return False
        return False

    def _last_own_move(self, *, mine: bool = False) -> PlyRecord | None:
        """The player's own move, when it is the last move played and has its verdict. Once
        the opponent has replied, "that move" is theirs -- unless the words say it is the
        player's (`mine`): "what should I have played?", "did I blunder?"."""
        with self._lock:
            records = self.conversation.active_records()
            if mine:
                own = next((r for r in reversed(records) if r.move.by_user), None)
                return own if own is not None and own.assessment is not None else None
            if records and records[-1].move.by_user and records[-1].assessment is not None:
                return records[-1]
        return None

    def _asked_own_move(self, text: str) -> PlyRecord | None:
        """The player's move a question is about, when it asks about their own last move:
        against a bot the reply comes at once, and "what should I have played?" then found
        the opponent's move last and got a hint for the next one (the runtime audit).

        Offline, words said while the opponent is to move -- the player's own move the last
        one played -- are about that move too, unless they ask about the opponent's threat or
        move, the material, the opening, the standing or the game: a simulated student's "I
        didn't even notice they were aiming for my rook" after 9.Nf3? was answered for the side
        to move, "You can win a knight. Their knight on f3 can simply be taken" -- the student's
        own knight (2026-09-26). With a model that guess is the model's to make, from the
        record: made here, "how would this position continue?" after a puzzle's last move was
        answered four times running as a question about that move, from the position before
        it ("Best, and it cost nothing: Rxf1+ ...", Tai's puzzle, 2026-09-28)."""
        if _ASK_LAST_MOVE.search(text) or _ASK_COST.search(text):
            return self._last_own_move(mine=bool(_MY_MOVE.search(text)))
        if self.finished or self.user_to_move or not self._offline:
            return None
        elsewhere = (
            _ASK_THREAT,
            _ASK_OPPONENT_MOVE,
            _ASK_LINE,
            _ASK_MATERIAL,
            _ASK_OPENING,
            _ASK_EVAL,
            _ASK_REVIEW,
        )
        if any(pattern.search(text) for pattern in elsewhere):
            return None
        return self._last_own_move()

    def _last_opponent_move(self) -> PlyRecord | None:
        """The opponent's move, when it is the last move played and has its verdict."""
        with self._lock:
            records = self.conversation.active_records()
            if records and not records[-1].move.by_user and records[-1].assessment is not None:
                return records[-1]
        return None

    def _standing_template(self, position: _Position, material: str | None = None) -> str:
        """Who stands better, from the engine's score for the position, in plain words, and the
        material (`material`, said whole by `_material_words`; else the facts')."""
        analysis = position.analysis
        if analysis is None or not analysis.lines:
            return "I have no evaluation for this yet."
        score = analysis.lines[0].expected_score
        if position.board.turn != self.user_colour:
            score = 1.0 - score
        # The Standing line's own words: its bands and this template's differed (0.8 and
        # 0.85 for winning), and a reply that said what the line said was faulted.
        words = standing_words(score)
        if not material and position.facts is not None:
            material = position.facts.material
        return f"Right now {words}" + (f"; material: {material}" if material else "") + "."

    def _material_words(self, level: DisclosureLevel, facts: PositionFacts) -> str | None:
        """The material as the game stands, to the player: now, and after the recapture the
        engine expects when the last move was a capture -- named, unless it is the player's
        own move before the move rung. The template said only the first half and the model
        only the second ("you are up a bishop and a pawn for a knight", the recapture still
        to come: quality round 3, 2026-09-25)."""
        with self._lock:
            found = self.conversation.material_now()
        if found is None:
            return facts.material
        now, recapture, then = found
        if recapture is None or then is None:
            return now
        then = f"it is {then}" if then.startswith("level") else then
        if self.user_to_move and level < DisclosureLevel.MOVE:
            # Both halves, the recapture unnamed: which piece takes back is the question, and
            # "once you take back" does not answer it. Said as the material now alone, the
            # answer to "am I up material?" was the vaguer and the more misleading (Tai,
            # 2026-09-25: "a vague coach is worse than no coach at all").
            return f"{now} -- and once you take back, {then}"
        whose = "your" if self.user_to_move else "their"
        return f"{now} -- and after {whose} recapture {recapture}, {then}"

    def _review_template(self) -> str:
        """The record in a sentence, for when the model's review fails the checks: the
        tally and where the most ground was lost, from the same records the messages
        carry; once the game is over, the review's own sentences (a lost game's "how did
        I do" once named three late inaccuracies and not the blunder, offline)."""
        if self._review_lines:
            return " ".join(self._review_lines[:4])
        with self._lock:
            judged = [
                r
                for r in self.conversation.judged_records()
                if r.move.by_user and r.assessment is not None
            ]
            if not judged:
                return "I have not seen enough of this game yet to say."
            counts: dict[str, int] = {}
            lost: list[tuple[float, int, str]] = []
            for r in judged:
                assert r.assessment is not None
                label = (
                    r.assessment.label.value if r.assessment.label else r.assessment.severity.value
                )
                counts[label] = counts.get(label, 0) + 1
                if r.assessment.severity.value != "none":
                    numbered = f"{self.conversation._numbered(r.move)} ({label})"
                    lost.append((r.assessment.expected_loss, r.move.ply, numbered))
        good = sum(counts.get(k, 0) for k in ("brilliant", "great", "best", "excellent"))
        text = f"{good} of your {len(judged)} moves were best or excellent."
        # What they keep missing, from the session's causes, before the game has a review.
        kept = [(n, cause) for cause, n in self._causes_seen.items() if n >= 2 and cause in HABIT]
        if kept:
            n, cause = max(kept)
            words, habit = HABIT[cause]
            text += f" What you keep missing: {words}, {n} times."
            if not self.strong_player:
                text += f" The habit that fixes it: {habit}."
        if lost:
            # The three that cost the most, told in the order they were played.
            worst = sorted(sorted(lost, key=lambda t: -t[0])[:3], key=lambda t: t[1])
            text += " You lost the most ground at " + ", ".join(t[2] for t in worst) + "."
        else:
            text += " Nothing cost you real ground."
        return text

    def _latest_with_facts(self) -> _Position | None:
        for ply in sorted(self.positions, reverse=True):
            if self.positions[ply].facts is not None:
                return self.positions[ply]
        return None

    def _speak(
        self,
        trigger: Trigger,
        position: _Position,
        level: DisclosureLevel,
        ts: float,
        *,
        user_text: str | None = None,
        verdicts: Sequence[str] = (),
        assessment: MoveAssessment | None = None,
        played: MovePlayed | None = None,
        candidates: int | None = None,
        routine: bool = False,
        elicit: bool = False,
    ) -> None:
        assert position.facts is not None
        instruction, fallback = self._instruction(
            trigger, position, level, user_text, verdicts, assessment, played, elicit=elicit
        )
        self._take_leads(position, fallback)
        must_name = self._must_name(
            trigger, position, level, fallback, assessment, played, user_text
        )
        if trigger is Trigger.USER_ASK and fallback and fallback in self._said_since_move():
            # Nothing the coach said since the last move is said again word for word: a
            # student's answer to the hint got the same hint back, three times (round 5).
            fallback = ""
        # A reply about a position already moved from -- the lesson on a move, an answer
        # about it -- is rendered at the level of the position on the board now, and must not
        # give away the answer there (a verdict can land after the opponent has replied).
        now = self._current
        past = now is not None and now is not position and now.ply > position.ply
        render_level = now.disclosure.level if past and now is not None else level
        now_answer: tuple[str, ...] = ()
        if (
            past
            and now is not None
            and now.facts is not None
            and now.board.turn == self.user_colour
            and now.disclosure.level < DisclosureLevel.MOVE
        ):
            spoken_of = {played.san if played else "", *self._played_moves()}
            if assessment is not None and assessment.best:
                with contextlib.suppress(ValueError):
                    spoken_of.add(position.board.san(chess.Move.from_uci(assessment.best[0])))
            answer = [*now.facts.candidates[:1], *now.facts.line]
            now_answer = tuple(m for m in dict.fromkeys(answer) if m not in spoken_of)
        # The player faces a position again only once it is their turn: on the opponent's,
        # what the move allowed is the lesson, and "nothing about the position" forbade it.
        facing = past and now is not None and now.board.turn == self.user_colour
        line = self._level_line(level, position, render_level if facing else None, past=past)
        # A question is answered as asked: its words head the Task. Found only under "Said
        # here", far above, "whose bishop? ... he will just move away to the right, no?" got the
        # line again (Tai's puzzle, 2026-09-28).
        asked = f'Their words, which you answer: "{user_text}"\n\n' if user_text else ""
        if user_text and _SIDEWAYS.search(user_text):
            # Said once in the Setting, far above, left and right were still read from White's
            # side: "from h1 there is no square to the right" (Tai's puzzle, both models).
            # Told so, the model still wrote "the only legal reply is Kg2, toward the a-file" as
            # no move to the right: the moves in question are given their way.
            side = "White" if self.user_side == Side.WHITE else "Black"
            asked += f"They say left and right from {side}'s side: {hands(self.user_side)}."
            seen = [ways(position.board, [san], self.user_side) for san in _SAN.findall(user_text)]
            if position.facts is not None:
                seen.append(ways(position.board, position.facts.line[:2], self.user_side))
            seen = [s for s in dict.fromkeys(seen) if s]
            if seen:
                asked += f" Seen so: {'; '.join(seen)}."
            asked += "\n\n"
        directive = f"{line}{asked}{instruction}"
        faced = ""
        frames: list[chess.Board] = []
        moved = played if trigger in (Trigger.POST_MOVE, Trigger.POST_MISTAKE) else None
        faced_at = position  # the position `moved` was played from
        if moved is None and trigger is Trigger.USER_ASK:
            own = self._asked_own_move(user_text or "")
            last = self._last_own_move()
            if own is not None and own.move.ply == position.ply + 1:
                moved = own.move
            elif (
                own is None
                and last is not None
                and last.move.ply in (position.ply, position.ply + 1)
            ):
                # The player's move is the last one played (its new position read, or not
                # yet): its sheet goes with any question, and the model reads which position
                # the words are about -- "why does the bishop check not work?" is about the
                # move, "how would this position continue?" is not (Tai's puzzle, 2026-09-28).
                prior = (
                    position
                    if last.move.ply == position.ply + 1
                    else self.positions.get(last.move.ply - 1)
                )
                if prior is not None and prior.facts is not None:
                    moved, faced_at = last.move, prior
        if moved is not None:
            # The lesson after a move is taught from the position the player faced, which the
            # record keeps as one line: its sheet goes with the request (parity pass,
            # 2026-09-25).
            after = self._frames_after(faced_at, moved)[:2]
            if (
                len(after) == 1
                and assessment is not None
                and len(assessment.best) > 1
                and assessment.best[0] == moved.uci
            ):
                # The new position not yet read: the engine's own line names the reply.
                with contextlib.suppress(ValueError):
                    reply = chess.Move.from_uci(assessment.best[1])
                    if reply in after[0].legal_moves:
                        replied = after[0].copy(stack=False)
                        replied.push(reply)
                        after.append(replied)
            assert faced_at.facts is not None
            with self._lock:
                sheet = self.conversation.faced_text(
                    faced_at.facts,
                    faced_at.board,
                    after[-1] if after else None,
                    after[0] if after else None,
                )
            if sheet:
                faced = f"\n\n# The position they faced, before {moved.san}\n{sheet}"
            frames = self._frames_after(faced_at, moved)
        frames += self._analysis_frames(position)
        # The earlier moments of the same idea, for a directive a link belongs in: found in
        # the record by the coach and listed with how each went and what was said, and every
        # other earlier move of the game held back from the reply (`unlisted_links`).
        earlier = ""
        before: frozenset[str] | None = None
        if TIE_TO_THE_GAME in directive:
            key = position.facts.theme or position.facts.plan
            taught = moved.ply if moved is not None else position.ply + 1
            with self._lock:
                links = self.conversation.same_idea(key, taught) if key else []
                before = self.conversation.moves_before(position.ply)
            if links:
                earlier = f"\n\n# Earlier in this game, the same idea: {key}\n" + "\n".join(links)
            directive = directive.replace(TIE_TO_THE_GAME, TIE_LISTED if links else NO_EARLIER)
        # And the board before the last move played: an answer about it says what it took
        # ("15...Qxe7 took your knight on e7"), which the board after it no longer shows.
        with self._lock:
            records = self.conversation.active_records()
        last_before: chess.Board | None = None
        if records:
            with contextlib.suppress(ValueError):
                last_before = chess.Board(records[-1].move.fen_before)
                frames.append(last_before)
        # A question of the player's: the model may ask the engine what the messages do not
        # show, from the board now or from before the last move (`engine_tools`).
        tools = (
            EngineTools(
                self.board,
                last_before,
                self.engine,
                self.user_colour,
                # The lines the record shows for each: a move on one is read from it.
                known_now=self._sheet_lines(self.board),
                known_before=self._sheet_lines(last_before),
            )
            if trigger is Trigger.USER_ASK and self.engine is not None and not self._offline
            else None
        )
        if tools is not None:
            if LINE_GIVEN in directive and position.facts.line:
                # What comes after the line's end, asked of the engine here: left to the model,
                # Luna at no effort called the tool one time in two and gave the line again.
                answer = tools.run("analyse", {"moves": list(position.facts.line)})
                directive += f"\n\nThe engine, asked about the line's end:\n{answer}"
            directive += f"\n\n{TOOL_WORDS}"
        with self._lock:
            self.conversation.level = render_level
            harness = self.conversation.render()
            messages: list[ChatMessage] = [
                {"role": "user", "content": f"{harness}{faced}{earlier}\n\n# Task\n{directive}"},
            ]
            # What the coach holds for the position on the board, which the position now must
            # show (the early depth was answered from but never shown: the coach review).
            held = self._current
            if held is not None and held.board.fen() != self.board.fen():
                held = None
            problem = harness_problem(
                harness,
                self.board,
                self.user_side,
                lines=held is not None and held.analysis is not None and bool(held.analysis.lines),
                facts=held is not None and held.facts is not None,
            )
        if problem:
            self.harness_problems.append(f"{trigger.value} at ply {position.ply}: {problem}")
        unlisted: frozenset[str] | None = None
        if before is not None:
            shown = _GAME_SO_FAR.sub("", str(messages[0]["content"]))
            for constant in (NO_FIGURES, NOT_THE_MOVE, NOT_THE_MOVE_HELD, TIE_LISTED, NO_EARLIER):
                shown = shown.replace(constant, " ")
            facts = position.facts
            sanctioned = {m.rstrip("+#") for m in _SAN.findall(shown)}
            sanctioned |= {m.rstrip("+#") for m in [*facts.candidates, *facts.line, *facts.threats]}
            unlisted = frozenset(m for m in before if m.rstrip("+#") not in sanctioned)
        request = _Request(
            trigger=trigger,
            level=level,
            system=self.system,
            messages=messages,
            facts=position.facts,
            board=position.board,
            fallback=fallback,
            ply=position.ply,
            game_id=self.game_id or "",
            user_text=user_text,
            disclosed="\n".join(v for v in verdicts if v.startswith("Engine on ")),
            annotations=annotations_for(
                position.facts,
                level,
                position.board,
                trigger=trigger,
                played_san=played.san if played else None,
                severity=assessment.severity.value if assessment else None,
            ),
            candidates=(
                candidates or (VS_ASK_CANDIDATES if trigger is Trigger.USER_ASK else VS_CANDIDATES)
            )
            if self.verbalized_sampling
            else 1,
            played=self._record_moves(),
            played_san=played.san if played else None,
            assessment=assessment,
            stream_to=self.stream_sink(trigger, position.ply) if self.stream_sink else None,
            routine=routine,
            # A fast game gets the template, not a model reply; what the player asks, and
            # the review of a game that is over, always get the model.
            template_only=(self.fast_now(ts) and trigger not in (Trigger.USER_ASK, Trigger.REVIEW)),
            must_name=must_name,
            frames=frames,
            now_answer=now_answer,
            record=self._numbered_moves(),
            unlisted=unlisted,
            tools=tools,
            epoch=self._epoch,
        )
        self.speaker.submit(request)

    def _must_name(
        self,
        trigger: Trigger,
        position: _Position,
        level: DisclosureLevel,
        template: str,
        assessment: MoveAssessment | None,
        played: MovePlayed | None,
        user_text: str | None,
    ) -> tuple[str, ...]:
        """What a model reply must name, as its template does: the better move, after a move
        that was not the best; for a hint, by its rung -- one of the key fact's squares, the
        target's or the piece's, the move; their threat, asked for it. Never a move the sheet
        lists as one of several good ones: the directive forbids the player's moves, and
        requiring them rejected every obedient reply (the runtime audit, 2026-09-25).
        Nothing for a word of praise, another question, or a template with none."""
        if not template:
            return ()
        said = user_text or ""
        facts = position.facts
        own = self._asked_own_move(said) if trigger is Trigger.USER_ASK else None
        if own is not None:
            # "Why was that bad?": the lesson on their last move, as after it. Not on a
            # follow-up, which the directive tells not to repeat the better move: requiring
            # it refused every obedient reply, and the template said the verdict again (the
            # prompt audit, 2026-09-28).
            if own.move.ply != position.ply + 1 or self._following_up(said):
                return ()
            trigger, assessment, played = Trigger.POST_MISTAKE, own.assessment, own.move
        if trigger in (Trigger.POST_MOVE, Trigger.POST_MISTAKE):
            if assessment is None or played is None or not assessment.best:
                return ()
            try:
                best = position.board.san(chess.Move.from_uci(assessment.best[0]))
            except ValueError:
                return ()
            return (best,) if best != played.san and _names(template, best) else ()
        if facts is None:
            return ()
        if trigger is Trigger.USER_ASK and _ASK_THREAT.search(said):
            # On the opponent's turn their threat is their best move now (`their_move_now`).
            theirs = (
                facts.candidates[:1] if position.board.turn != self.user_colour else facts.threats
            )
            return tuple(t for t in theirs if _names(template, t))
        hint = trigger in (Trigger.DECISIVE_QUESTION, Trigger.STUCK) or (
            trigger is Trigger.USER_ASK and bool(_ASK_HINT.match(said) or _ASK_AIM.search(said))
        )
        if not hint:
            return ()
        if trigger is Trigger.STUCK:
            level = min(level, DisclosureLevel.FOCUS)  # the rung its template is on
        if level >= DisclosureLevel.MOVE:
            best = facts.candidates[0] if facts.candidates else ""
            return (best,) if best and _names(template, best) else ()
        choices = squares_in(" ".join(facts.choices))
        if level >= DisclosureLevel.FOCUS:
            source = " ".join(x for x in (facts.target, facts.piece) if x)
        else:
            source = facts.key or ""
        named = squares_in(template) - choices
        squares = (squares_in(source) & named) or named
        return tuple(sorted(squares))

    @staticmethod
    def _analysis_frames(position: _Position) -> list[chess.Board]:
        """Boards along every line the facts were read from -- the engine's lines, the traps,
        the recaptures the best line declines, what a quiet best move threatens, the
        opponent's threats after a pass -- so a sentence the template itself says ("after
        Rxa5, Kxf6 ...") is not faulted as unplayable when the model says it too (the eval
        audit, 2026-09-25). The weaker lines only to four plies."""
        analysis = position.analysis
        if analysis is None:
            return []
        root = position.board
        frames: list[chess.Board] = []

        def walk(start: chess.Board, ucis: Sequence[str], limit: int) -> None:
            board = start
            for uci in ucis[:limit]:
                try:
                    move = chess.Move.from_uci(uci)
                except ValueError:
                    return
                if move not in board.legal_moves:
                    return
                board = board.copy(stack=False)
                board.push(move)
                frames.append(board)

        for i, line in enumerate(analysis.lines):
            walk(root, line.moves, 8 if i == 0 else 4)
        for line in [*analysis.traps, *analysis.defences]:
            walk(root, line.moves, 8)
        if not root.is_check():
            passed = root.copy(stack=False)
            passed.push(chess.Move.null())
            for san in analysis.threats:
                with contextlib.suppress(ValueError):
                    after = passed.copy(stack=False)
                    after.push(after.parse_san(san))
                    frames.append(after)
            if analysis.threatens and analysis.lines and analysis.lines[0].moves:
                with contextlib.suppress(ValueError):
                    best = chess.Move.from_uci(analysis.lines[0].moves[0])
                    if best in root.legal_moves:
                        quiet = root.copy(stack=False)
                        quiet.push(best)
                        if not quiet.is_check():
                            quiet.push(chess.Move.null())
                            for line in analysis.threatens:
                                walk(quiet, line.moves, 8)
                                walk(root, line.moves, 8)
        return frames

    def _frames_after(self, position: _Position, played: MovePlayed) -> list[chess.Board]:
        """The board after the move just played, and along the engine's line from there: the
        board the model is shown, so a true sentence about it is not faulted against the
        position before the move."""
        try:
            move = chess.Move.from_uci(played.uci)
        except ValueError:
            return []
        if move not in position.board.legal_moves:
            return []
        board = position.board.copy(stack=False)
        board.push(move)
        frames = [board]
        left = self.positions.get(played.ply)
        if left is not None and left.analysis is not None and left.analysis.lines:
            for uci in left.analysis.lines[0].moves[:8]:
                try:
                    step = chess.Move.from_uci(uci)
                except ValueError:
                    break
                if step not in board.legal_moves:
                    break
                board = board.copy(stack=False)
                board.push(step)
                frames.append(board)
        return frames

    def _numbered_moves(self) -> frozenset[str]:
        """The moves played in the game so far as the record numbers them: "17. Rh1"."""
        with self._lock:
            return frozenset(
                self.conversation._numbered(r.move) for r in self.conversation.judged_records()
            )

    def _played_moves(self) -> frozenset[str]:
        """The moves played in the game so far, as SAN."""
        with self._lock:
            return frozenset(r.move.san for r in self.conversation.judged_records())

    def _record_moves(self) -> frozenset[str]:
        """The moves the messages name about the game so far: every move played, and the
        move each verdict said the engine preferred."""
        out: set[str] = set()
        with self._lock:
            for r in self.conversation.judged_records():
                out.add(r.move.san)
                if r.assessment is not None and r.assessment.best:
                    with contextlib.suppress(ValueError):
                        board = chess.Board(r.move.fen_before)
                        out.add(board.san(chess.Move.from_uci(r.assessment.best[0])))
        return frozenset(out)

    @staticmethod
    def _level_line(
        level: DisclosureLevel,
        position: _Position,
        now_level: DisclosureLevel | None = None,
        *,
        past: bool = False,
    ) -> str:
        """What may be said about the position the player faces, stated on the directive, the
        one place it is stated; with `now_level`, the reply is about a position already moved
        from, and the one on the board has its own level. Below the move, why the model sees
        the answer all the same: said as "never" in five places, it gave the vague hint Tai
        ranks below one that says a little too much (the prompt audit, 2026-09-28)."""
        if position.board.is_game_over():
            return ""
        words = {
            DisclosureLevel.SILENT: "nothing about the position",
            DisclosureLevel.IDEA: IDEA_LEVEL_WORDS,
            DisclosureLevel.FOCUS: "the idea, the target and the piece; not the move",
            DisclosureLevel.MOVE: "the move and why it works; not the whole line",
            DisclosureLevel.LINE: "everything, the line included",
        }
        if now_level is not None:
            return (
                "The position this is about has been moved from: everything about it may be "
                "said. Disclosure for the position the player faces now: "
                f"{words[now_level]}.\n\n"
            )
        if past:
            return (
                "The position this is about has been moved from: everything about it may be "
                "said, and what the opponent can do now.\n\n"
            )
        why = f" {SEE_ALL_WORDS}" if DisclosureLevel.IDEA <= level < DisclosureLevel.MOVE else ""
        return f"Disclosure for the position the player faces: {words[level]}.{why}\n\n"

    def _queue_plain(
        self,
        trigger: Trigger,
        text: str,
        ts: float,
        *,
        ply: int | None = None,
        level: DisclosureLevel = DisclosureLevel.SILENT,
        annotations: list[Annotation] | None = None,
        fen: str | None = None,
        now: bool = False,
        record_now: bool = False,
    ) -> None:
        """A message that needs no model; it queues behind whatever the model is writing,
        unless `now`, when it goes out at once. With `record_now` it is a turn of the
        conversation from this moment, so a request built next already shows it."""
        message = CoachMessage(
            text=text,
            level=level,
            trigger=trigger,
            game_id=self.game_id,
            ply=ply,
            fen=fen,
            annotations=list(annotations or []),
        )
        if record_now:
            with self._lock:
                self.conversation.coach_said(text)
        self.speaker.submit_ready(
            Spoken(
                message=message,
                level=level,
                problems=[],
                attempts=0,
                recorded=record_now,
                epoch=self._epoch,
            ),
            now=now,
        )

    def _instruction(
        self,
        trigger: Trigger,
        position: _Position,
        level: DisclosureLevel,
        user_text: str | None,
        verdicts: Sequence[str],
        assessment: MoveAssessment | None,
        played: MovePlayed | None,
        *,
        elicit: bool = False,
    ) -> tuple[str, str]:
        """The turn's instruction to the model, and the template used if it fails. `elicit`
        ends a mistake's lesson by asking what the player saw (see `_elicit_due`)."""
        facts = position.facts
        assert facts is not None

        if trigger is Trigger.REVIEW:
            return (
                "The game is over. Sum it up from the review in the last message and the "
                "verdicts on the moves above, and from nothing else: the result, what went "
                "well, and the one thing to work on. When the review says what they kept "
                "missing, that is the thing: the pattern in plain words and the habit that "
                "fixes it, as the review gives it, with the moment it cost most. Otherwise the "
                "moment the game turned, the better move there, and why it happened, in plain "
                "words. Three or four sentences, no question. Do not infer the course of the "
                "game from the position on the board.",
                " ".join(self._review_lines[:4]),
            )
        if trigger is Trigger.DECISIVE_QUESTION:
            # Never fired live since 2026-10-01 (nothing unasked before the stuck limit);
            # the eval's recorded requests of it still replay (tools/eval/judge.py).
            template, note = self._cool_hint(facts, idea_template(facts))
            return (f"{IDEA_DIRECTIVE} {NO_FIGURES}{note}", template)
        if trigger is Trigger.STUCK:
            # The hints unasked stop at the focus: the move is only for when they ask. The
            # template on the level a question had opened named the move while the model was
            # told not to, and every reply that obeyed went for the template (Tai's game,
            # 2026-09-28: "d5: your bishop on c8 is still shut in", twice).
            rung = min(level, DisclosureLevel.FOCUS)
            directive = FOCUS_DIRECTIVE if rung >= DisclosureLevel.FOCUS else IDEA_DIRECTIVE
            template, note = self._cool_hint(facts, hint_template(facts, rung))
            return (
                "The player has been thinking for a while and is stuck. "
                + directive
                + " Take a different angle from anything already said about this position. "
                + NO_FIGURES
                + note,
                template,
            )
        if trigger in (Trigger.POST_MOVE, Trigger.POST_MISTAKE) and assessment and played:
            # The better move is the engine's own from the position the player faced
            # (assessment.best, UCI); the fact sheet's first candidate stands in when the
            # assessment has none. Without a model this sentence is the whole lesson, so
            # it must never name the move played as the better one.
            best = facts.candidates[0] if facts.candidates else "another move"
            if assessment.best:
                with contextlib.suppress(ValueError, IndexError):
                    best = position.board.san(chess.Move.from_uci(assessment.best[0]))
            if best == played.san and trigger is Trigger.POST_MISTAKE:
                best = "another move"
            loss = assessment.expected_loss
            label = assessment.label.value if assessment.label else assessment.severity.value
            better_words = _in_words(position.board, best) if facts.candidates else best
            routine = not facts.candidates or best == played.san or _local(assessment.label)
            if trigger is Trigger.POST_MISTAKE:
                # The better move played out, and the pattern to keep: never "the idea was
                # a mating attack" and nothing more.
                # What the move let them do, from the position it left, when that has
                # been analysed: the grounded half of "what their move allowed".
                left = self.positions.get(played.ply)
                allowed = (
                    idea_words.allowed(left.board, left.analysis, played.san)
                    if left is not None and left.analysis is not None
                    else None
                )
                opener = self._bags.draw("cost", COST_YOU).format(best=best)
                rule = rule_of(facts, best, board=position.board, played=played.uci)
                fallback, held = self._fresh_lesson(
                    opener,
                    lambda with_rule: lesson_template(
                        facts,
                        best,
                        opener,
                        board=position.board,
                        played=played.uci,
                        allowed=allowed,
                        with_rule=with_rule,
                        strong=self.strong_player,
                    ),
                    rule,
                    facts,
                )
                given = rule if not held and not self.strong_player else None
                data = _lesson_data(position.board, best, played.uci, given)
                if allowed:
                    data += f" What {played.san} allowed, from the engine's line: {allowed}."
                # A question to close on is offered, not asked for: required, it was the one
                # a lesson had room for once the pattern was said, and often a quiz on what
                # the reply had just told them.
                instruction = (
                    f"The player just played {played.san}, which was {_rated(label)}: it cost "
                    f"{loss:.0%} of the game. The better move was {best} ({better_words}). "
                    "Teach the difference so they spot it next time: why the better move works "
                    "and what their move missed or allowed, from the sheet of the position they "
                    "faced -- its goal, its pattern, the tactic when it names one -- as a "
                    "pattern to remember. Three sentences at most, then, if you like, one "
                    "question that sends them to find on the board something your reply has "
                    f"not said. {TIE_TO_THE_GAME} {NOT_THE_MOVE} {NO_FIGURES}{data}"
                )
                if elicit:
                    # Sparingly: what they saw at that moment, for the profile's record
                    # (causes.from_words reads "I thought it would..." as a line
                    # misjudged, "I didn't see..." as a threat missed). Asked as what
                    # they expected, in the coach's words: told to ask whether they
                    # "missed the idea altogether or misjudged the line", the model
                    # said exactly that, every time.
                    fallback += " What were you expecting that move to do?"
                    instruction = (
                        f"The player just played {played.san}, which was {_rated(label)}: it "
                        f"cost {loss:.0%} of the game. The better move was {best} "
                        f"({better_words}). Teach the difference so they spot it next time: why "
                        "the better move works, from the sheet of the position they faced -- its "
                        "goal, its pattern, the tactic when it names one -- as a pattern to "
                        "remember, in three sentences at most, then -- instead of a check "
                        "question -- ask, in one short question of your own, what they "
                        f"expected their move to do. {NOT_THE_MOVE} {NO_FIGURES}"
                        + _lesson_data(position.board, best, played.uci, given)
                    )
                if held:
                    instruction = _hold_rule(instruction)
            elif _exciting(assessment.label):
                # Rare and worth it: the board's own sentence stands in, the model says
                # what makes it the move, with some delight.
                fallback, kind = local_verdict(
                    position.board,
                    chess.Move.from_uci(played.uci),
                    label,
                    assessment.tags,
                    phase=facts.phase,
                    threats=facts.threats,
                    taught=self._verdict_taught(),
                    bags=self._bags,
                    plain=self.plain_words,
                    strong=self.strong_player,
                )
                # Said by the model or, when its reply fails, by this sentence: either way
                # the idea is taught ("the king is a fighting piece" after a great Kd6, then
                # again after Kd5: the offline sweep, 2026-09-27).
                self._taught_by_verdict(kind)
                meaning = {
                    "brilliant": "a sound sacrifice",
                    "great": _great_meaning(facts),
                }[label]
                # Only the reason that holds: offered "the sacrifice that works" beside a
                # great move, the model called 11...Nxf3+ gxf3, a knight for a knight, "the
                # knight sacrifice" (the dev audit, 2026-09-26).
                reason = (
                    "the sacrifice that works" if label == "brilliant" else meaning.split(":", 1)[0]
                )
                instruction = (
                    f"The player just played {played.san}: {label} -- {meaning}. Show some "
                    "delight, in your own voice, in one or two sentences: name what makes it "
                    f"the move -- {reason}, and the idea it serves -- from the messages above, "
                    "nothing invented. No question, no advice, and not the same words as your "
                    "last such sentence. "
                    f"{NOT_THE_MOVE} {NO_FIGURES}"
                )
                # The pattern to keep is a rule of thumb like any other, on the one budget:
                # two great moves running got "Keep spotting this pattern ..." and "keep
                # recognizing this pattern ..." back to back (Tai's puzzle, 2026-09-28).
                instruction = self._budgeted(instruction, f"verdict:{kind}")
            elif routine:
                fallback = self._verdict(label)
                meaning = {
                    "book": "opening theory",
                    "brilliant": "a sound sacrifice",
                    "great": _great_meaning(facts),
                    "best": "the engine's own choice",
                    "excellent": "as strong as the best move",
                    "good": "a sound move, a little behind the best",
                }.get(label, "a sound move")
                instruction = (
                    f"The player just played {played.san}: {label} -- {meaning}. One short "
                    "sentence, fifteen words at most, in your own voice: a word of praise, and "
                    "the idea behind the move as a lesson to keep when a message gives one -- "
                    "the plan, the tactic -- and only when it is what the move is for. No "
                    "question, no lecture, no advice, no guess at what it allows or risks, and "
                    "not the same words as your last such sentence. "
                    f"{NOT_THE_MOVE} {NO_FIGURES}"
                )
                instruction = self._budgeted(instruction, f"verdict:{label}")
            elif assessment.severity in (Severity.MISTAKE, Severity.BLUNDER) or label == "miss":
                # A mistake in blitz: the policy keeps it to the word after the move (no
                # lesson), but the word says what the move was -- never "fine".
                fallback = self._bags.draw("cost", COST_YOU).format(best=best) + "."
                instruction = (
                    f"The player just played {played.san}, which was {_rated(label)}: {best} "
                    f"({better_words}) was better. It is a fast game: one or two sentences, no "
                    f"lesson, no question. {NOT_THE_MOVE} {NO_FIGURES}"
                )
            else:
                # The better move and what it does, when the sheet has it, is the lesson.
                opener = self._bags.draw("fine", FINE_TIGHTER).format(best=best)
                fallback, held = self._fresh_lesson(
                    opener,
                    lambda with_rule: lesson_template(
                        facts,
                        best,
                        opener,
                        board=position.board,
                        played=played.uci,
                        with_rule=with_rule,
                        strong=self.strong_player,
                    ),
                    None,  # a fine move: the better move and what it does, no rule of thumb
                )
                # How it rated goes in passing: asked for first, it opened half these replies
                # with "Fine", "Playable" or "That works, though" (the prompt audit, 2026-09-28).
                instruction = (
                    f"The player just played {played.san}, which was {_rated(label)}: sound, a "
                    f"little behind {best} ({better_words}). One or two sentences: the better "
                    "move, what it does, and the idea behind it as a lesson to keep, with how "
                    "theirs rated in passing. No lecture, no question. "
                    f"{TIE_TO_THE_GAME} {NOT_THE_MOVE} {NO_FIGURES}"
                    + _lesson_data(position.board, best, played.uci, None)
                )
                if held:
                    instruction = _hold_rule(instruction)
            return instruction, fallback
        # An explicit ask. The player's words are already the last message of the
        # conversation; the directive says only how to answer them.
        said = user_text or "hint"
        checks = ""
        claims_checked = [v for v in verdicts if not v.startswith("Engine on ")]
        engine_words = [v for v in verdicts if v.startswith("Engine on ")]
        if claims_checked:
            checks = (
                "Their claims, checked against the board:\n- "
                + "\n- ".join(claims_checked)
                + "\n\n"
            )
        if engine_words:
            # The engine on what they asked about: theirs to hear about in full.
            checks += "\n".join(engine_words) + "\n\n"
        # The rung's template: two hints, then the answer; the line when asked for it. The
        # hints speak to the side to move: on the opponent's turn a failed reply falls back to
        # what they will play, never their hint in the player's voice.
        fallback = hint_template(facts, level, position.board)
        if position.board.turn != self.user_colour and level < DisclosureLevel.LINE:
            fallback = their_move_now(position.board, facts)
        # The position answered about is not the one on the board (its analysis has not
        # landed): say so, or the template speaks of a past position as the present --
        # "the line is Bb3 ..." right after Bb3 was played (the offline pass, 2026-09-21).
        current = self._current
        if current is not None and position.ply != current.ply:
            own = self._last_own_move()
            if own is not None and own.move.ply == current.ply and position.ply == own.move.ply - 1:
                san = own.move.san
                rest = facts.line[1:] if facts.line and facts.line[0] == san else facts.line
                if level >= DisclosureLevel.LINE and rest:
                    fallback = f"It's their move. After your move, the line runs {' '.join(rest)}."
                elif level >= DisclosureLevel.MOVE and facts.candidates:
                    choice = facts.candidates[0]
                    fallback = "It's their move now. " + (
                        "Yours was the engine's choice too."
                        if choice == san
                        else f"The engine's choice there was {choice}."
                    )
                else:
                    fallback = "It's their move. See what they do about yours, then ask again."
            else:
                fallback = "Give me a moment; I am still reading this position."
        # A question about a fact that is never withheld -- the material, the opening, who
        # stands better -- falls back to that fact, not to the hint for the position.
        material = (
            self._material_words(level, facts)
            if _ASK_MATERIAL.search(said) or _ASK_EVAL.search(said)
            else None
        )
        if material and _ASK_MATERIAL.search(said):
            fallback = f"Material: {material}."
        elif _OPENING_FACT.search(said):
            # The opening asked for by name: "what should I play against this opening? give me
            # a line" was answered with its name (Tai's game, 2026-09-28).
            opening = self._opening_so_far()
            fallback = (
                f"The opening was the {opening}." if opening else "We left the opening book early."
            )
        elif _ASK_EVAL.search(said):
            fallback = self._standing_template(position, material)
        # Their words answer the question the coach just asked: the thread is closed, not
        # carried on with another question. Every answer had earned a new question and none
        # was resolved -- the students' "what Nc4 was expected to do was asked but not
        # answered" (2026-09-26).
        answering = self._answering_our_question(said)
        close = (
            " They are answering the question you just asked, your last turn above: say first, "
            "plainly and from the facts, whether they have it right; if they do not, give the "
            "answer yourself in a sentence. Ask nothing more -- this closes that question."
        )
        # After the coach has already spoken since the move, a follow-up gets only what it
        # asks: the verdict and the better move came back in every reply ("e6 was a miss
        # costing 41% of the game; Nxc3 was better" twice running), and "I understand: I
        # should play Nxc3" got the lesson again and "Which White piece is still at home on
        # g1?" (the AI's sessions, 2026-09-26).
        followup = self._following_up(said)
        asks = bool(question_part(said))
        follow = (
            " You have already told them the verdict on it and the better move, in your turns "
            "above: do not repeat either."
            + (
                " Answer only what they ask now -- concretely, from the facts, in one to three "
                "sentences -- and ask nothing back."
                if asks
                else " They ask nothing: reply in one short sentence -- say whether what they "
                "say is right, or put it right -- and ask nothing."
            )
        )
        if _ASK_REVIEW.search(said):
            # An overall verdict comes from the tally in "Game so far", never from an
            # impression of the last position: the first review told a player a queen up
            # that they had been a queen down.
            directive = (
                "They asked how they did overall. Answer from the engine's verdicts on their "
                "moves in the messages above -- the labels, the tally in their last move's "
                "message, where they lost ground, how the material went -- and from nothing "
                "else: do not infer the course of the game from the position on the board. "
                "Two or three sentences: what went well -- how many of their moves were strong, "
                "and their slips by kind, in the tally's numbers -- and the one thing to work "
                "on: when the review, or the pattern already told them, says what they keep "
                "missing, that pattern in plain words and the habit that fixes it, as given; "
                "otherwise the move the "
                "tally lists first where ground was lost most, the better move there, and why, "
                "in plain words. If no move has a verdict yet, say you have not seen enough to "
                "say."
            )
            # A review has a record to fall back on, not a line.
            fallback = self._review_template()
        elif _ASK_HINT.match(said) or _ASK_AIM.search(said):
            # A bare "hint" carries no claim to be honest about; treated as one, the model
            # answered its own last message as if the player had said it. Two hints, then
            # the answer: the rung decides what the hint may give. A question about what to
            # aim for is answered by the same hint: the generic answer ("be honest about their
            # own claim") gave the model no lesson to give (parity pass 2, 2026-09-25).
            aim = "" if _ASK_HINT.match(said) else "They asked what to aim for: answer with "
            aim += "the hint at this rung. " if aim else ""
            if level >= DisclosureLevel.MOVE:
                directive = aim + MOVE_DIRECTIVE
            elif level >= DisclosureLevel.FOCUS:
                directive = (aim or "They asked for another hint. ") + FOCUS_DIRECTIVE
            else:
                directive = (aim or "They asked for a hint. ") + IDEA_DIRECTIVE
            directive += f" {NO_FIGURES} Do not comment on anything said earlier."
        elif level >= DisclosureLevel.LINE and _ASK_LINE.search(said) and facts.line:
            # The line move by move, computed: the only reasons a move of it may be given
            # (4 of the 5 false claims the bench's labellers found were in these answers).
            notes = line_notes(position.board, list(facts.line), player=self.user_colour)
            by_move = (
                f" The line move by move, as the board gives it: {notes}. Explain it only from "
                "these notes: a move 'forces' something only where a note says it is the only "
                "legal move or that only a king move gets out of check."
                if notes
                else ""
            )
            directive = (
                f"They asked to see the line: give it -- {' '.join(facts.line)} -- plainly, then "
                "one short sentence on the idea behind it." + by_move + " Do not answer with a "
                "question."
            )
            given = [m.rstrip("+#") for m in facts.line[:3]]
            if any(all(m in s for m in given) for s in self._said_since_move()):
                # Given once already, the line asked for again is what comes after it: "ok and
                # after that?" got the same line back (Tai's puzzle, 2026-09-28). The engine's
                # word on its end is put before the model (`_speak`).
                directive += f" {LINE_GIVEN}"
        elif level >= DisclosureLevel.MOVE and _ASK_MOVE.search(said) and facts.candidates:
            directive = (
                f"They asked what to play: tell them plainly it is {facts.candidates[0]}, then why "
                "it works, from the fact sheet's why and pattern, correctly. Do not answer with a "
                "question this time."
            )
        elif _ASK_OPPONENT_MOVE.search(said) and (their := self._last_opponent_move()) is not None:
            assert their.assessment is not None
            verdict = their.assessment
            san = their.move.san
            label = verdict.label.value if verdict.label else verdict.severity.value
            fallback = f"Their {san} was {_rated(label)}."
            directive = (
                f"They ask about the opponent's last move, {san}. It is in the record under "
                f"'Last move' with its verdict ({label}) and what it does; all of that is theirs "
                "to hear. Name the move, say how it rated and what it does, in plain words, "
                "and what they threaten now when the facts list a threat (at most two), two or "
                "three sentences; then, if you like, one question about what it asks of them. "
                f"Never answer about the player's own move instead. {NO_FIGURES}"
            )
        elif (record := self._asked_own_move(said)) is not None:
            # Asked about their own last move. Its verdict was given in its move message and
            # the facts of the position it was played from in the message before: all of it
            # may be said outright. Without this the model took "the better move is withheld"
            # for the position now to cover the past one and answered with a hint instead.
            assert record.assessment is not None
            verdict = record.assessment
            san = record.move.san
            label = verdict.label.value if verdict.label else verdict.severity.value
            loss = verdict.expected_loss
            best = ""
            if verdict.best:
                try:
                    best = chess.Board(record.move.fen_before).san(
                        chess.Move.from_uci(verdict.best[0])
                    )
                except ValueError:
                    best = ""
            best = "" if best == san else best
            # The request's facts are those of the position the move was played from when
            # the coach has them (ask() chose that position); the idea is theirs. The theme
            # or the plan, never a static feature: "the idea was the isolated d-pawn ..."
            # became the reason a bishop retreat was an inaccuracy.
            idea = facts.theme or facts.plan or ""
            if verdict.severity is Severity.NONE and best and loss >= 0.005:
                # Sound but not the best: said as the word after it said it ("Fine, but a3
                # was a little tighter"), never "it cost nothing" (the runtime audit).
                opener = self._bags.draw("fine", FINE_TIGHTER).format(best=best)
                fallback, held = self._fresh_lesson(
                    opener,
                    lambda with_rule: lesson_template(
                        facts,
                        best,
                        opener,
                        played=record.move.uci,
                        with_rule=with_rule,
                        strong=self.strong_player,
                    ),
                    None,  # a fine move: the better move and what it does, no rule of thumb
                    asked=True,
                )
                fallback = _with_line(fallback, opener, best, facts)
                # What they ask is the answer; the verdict is its ground, said once: told "say
                # so plainly" on every question about the move, the verdict came back four
                # times running (Tai's puzzle, 2026-09-28).
                directive = (
                    f"They ask about the move they already played, {san}. It was {label}: sound, "
                    f"a little behind {best}. Answer what they ask"
                    + (
                        "; your turns above already gave the verdict and the better move, so "
                        "repeat them only where that is what they ask."
                        if followup
                        else "; where they ask how the move was, say so plainly -- it was fine -- "
                        "then the better move and the idea behind it."
                    )
                    + f" Two sentences at most, no question. {NO_FIGURES}"
                )
            elif verdict.severity is Severity.NONE:
                fallback = self._verdict(label)
                directive = (
                    f"They ask about the move they already played, {san}: its verdict was "
                    f"{label}, and it cost nothing. Answer what they ask, from the record and the "
                    "sheet of the position it was played from, and invent no fault"
                    + (
                        "; your turns above already gave the verdict, so repeat it only where "
                        "that is what they ask."
                        if followup
                        else "; where they ask how the move was, say so plainly."
                    )
                    + f" One or two sentences, no question. {NO_FIGURES}"
                )
            else:
                # What it cost only when they ask: "It cost you 22% of the game." opened every
                # answer, and Tai had said the figure is for when he asks (2026-09-20).
                asked_cost = bool(_ASK_COST.search(said))
                fallback = f"It cost you {loss:.0%} of the game." if asked_cost else ""
                # What the move let them do, from the position it left, as the lesson after
                # the move has it: without it "why was that bad?" named the better move and a
                # rule, and no piece, square or move of the reason (quality round 3).
                left = self.positions.get(record.move.ply)
                allowed = (
                    idea_words.allowed(left.board, left.analysis, san)
                    if left is not None and left.analysis is not None
                    else None
                )
                faced = position.ply == record.move.ply - 1
                board = position.board if faced else None
                held = False
                data = f" What {san} allowed, from the engine's line: {allowed}." if allowed else ""
                if best:
                    # The better move played out and the pattern to keep, as the lesson says
                    # it -- not "the idea was a pin" and nothing more -- and said whole, what
                    # the lesson after the move said or not: they asked.
                    opener = f"{best} was better"
                    rule = rule_of(facts, best, board=board, played=record.move.uci)
                    lesson, held = self._fresh_lesson(
                        opener,
                        lambda with_rule: lesson_template(
                            facts,
                            best,
                            opener,
                            board=board,
                            played=record.move.uci,
                            allowed=allowed,
                            with_rule=with_rule,
                            strong=self.strong_player,
                        ),
                        rule,
                        facts,
                        asked=True,
                    )
                    lesson = _with_line(lesson, opener, best, facts)
                    fallback = (fallback + " " if fallback else "") + lesson
                    given = rule if not held and not self.strong_player else None
                    data += _lesson_data(board, best, record.move.uci, given)
                elif allowed:
                    fallback = (fallback + " " if fallback else "") + _sentence(
                        allowed[0].upper() + allowed[1:]
                    )
                if not fallback:
                    fallback = f"That was {_rated(label)}."
                reason = f"what {best} does, and " if best else ""
                cost = (
                    f" -- it cost {loss:.0%} of the game: they asked, so say it"
                    if asked_cost
                    else ""
                )
                # A question of the coach's own after the answer is offered, not asked for:
                # they asked theirs, and required, one came after every answer. On a follow-up
                # the verdict is not said again: "say that plainly" beside "do not repeat
                # either" asked for both (the prompt audit, 2026-09-28).
                tell = "Say why" if followup and not answering else "Say that plainly, and say why"
                directive = (
                    f"They ask about the move they already played, {san}, from a position that "
                    f"is past, so all of it may be said: it was {_rated(label)}{cost}"
                    + (f", {best} was better" if best else "")
                    + (f", and the idea was {idea}" if idea else "")
                    + f". {tell}, concretely: {reason}what {san} missed or allowed, with the "
                    "pieces, squares or moves from the sheet of the position they faced (its "
                    "pattern, and its tactic when it names one)"
                    + (f" -- {RULE_HELD_MARK.lower()}: they heard one lately." if held else "")
                    + ("" if held else " -- and the pattern to keep.")
                    + data
                    + (
                        close + " Three sentences at most."
                        if answering
                        else follow
                        if followup
                        else " Three sentences at most, then, if you like, one question that "
                        "sends them to find on the board something your reply has not said."
                    )
                    + (
                        " Name moves bare, never with a move number in front."
                        if asked_cost
                        else f" {NO_FIGURES}"
                    )
                    + f" {TIE_TO_THE_GAME}"
                )
            if followup and not asks:
                fallback = ""  # a nod needs no template: nothing, rather than the verdict again
        elif _ASK_MATERIAL.search(said) or _OPENING_FACT.search(said) or _ASK_EVAL.search(said):
            # A plain question about the game as it stands: answered plainly, no question
            # back (the generic answer's "point them ... with a question" asked one).
            directive = (
                "They asked a plain question about the game as it stands. Answer it from the "
                "message -- the material, the opening, or who stands better, as it gives them "
                "-- in one or two sentences, and no question."
            )
            if _ASK_EVAL.search(said):
                directive += (
                    " Who stands better, in the words of the Standing line, and the material "
                    "as it stands."
                )
            if _OPENING_FACT.search(said):
                directive += " The opening by the name the record gives it."
            if material:
                directive += (
                    f" The material is this, and your answer gives all of it, in these terms: "
                    f"{material}. Never count the pieces yourself."
                )
        elif _ASK_THREAT.search(said) and position.board.turn != self.user_colour:
            # The opponent is to move: their threat is their best move now. The sheet's
            # threats are the side NOT to move's -- the player's own -- and answered from
            # them, "what are they threatening?" got the player's threat (the code review,
            # 2026-09-26).
            fallback = their_move_now(position.board, facts)
            directive = (
                "They ask what the opponent threatens, and it is the opponent's move: say what "
                "the opponent's best move now is and what it does, from the engine lines (the "
                "opponent's move) -- not 'the player's threats', which are the player's own. "
                "One or two sentences, no question. " + NO_FIGURES
            )
        elif _ASK_THREAT.search(said) and position.board.is_check():
            # In check there is no "if you passed": the sheet lists no threat, and "nothing
            # is threatened" was said with their queen checking and hitting a pawn (dev audit,
            # 2026-09-26). The check is the threat, and what else the checker attacks.
            fallback = in_check_threat(position.board)
            directive = (
                "They ask what the opponent threatens, and they are in check: the check is "
                f"the threat. Say it as the board gives it: {fallback} One or two sentences, "
                "no question. " + NO_FIGURES
            )
        elif _ASK_THREAT.search(said):
            threats = facts.threats[:2]
            if threats:
                words = [Conversation._threat_words(position.board, t) for t in threats]
                fallback = "They threaten " + " and ".join(words) + "."
            else:
                fallback = "Nothing is threatened right now: no move of theirs wins anything."
            directive = (
                "They ask what the opponent threatens. Say it as the facts give it -- each threat "
                "they list, at most two: the move and what it does, from 'the opponent's "
                "threats' -- or, when the facts list "
                "none, say plainly that nothing is threatened. One or two sentences, no question. "
                + NO_FIGURES
            )
        else:
            # What is kept back is the level line's to say: "never reveal that unless the sheet
            # lists it" stood here after the sheet had come to list everything (the prompt
            # audit, 2026-09-28). A claim is theirs only when they make one: told to be honest
            # about "their own claim", Luna answered a plain question with "You are right that
            # g1 is controlled by your King" (Tai's puzzle, 2026-09-29).
            directive = (
                "Answer them directly. What they state or take for granted, take as they meant "
                "it: say what in it is right as plainly as what is wrong -- and credit them with "
                "nothing they did not say: a question states nothing to agree with. When they "
                "propose a move, or name a square or piece, say plainly whether it holds up and "
                "what it does or allows, from the engine's word on it where given above -- "
                "theirs to hear in full: the score after it, how far behind the best, the reply, "
                "what the move does; when the engine does not rate their move, say it is not the "
                "strongest and, in general terms, why. Of the position they face, keep back only "
                "what the level line does, and point them toward the right idea"
                + (
                    " in words."
                    if answering or (followup and not asks)
                    else " -- with a question, if that serves."
                )
                + " What you already said in this position is under 'Said here': do not say "
                "it again in other words; give them something it did not."
                + (
                    close
                    if answering
                    else " They ask nothing: reply in one short sentence -- say whether what "
                    "they say is right, or put it right -- and ask nothing."
                    if followup and not asks
                    else ""
                )
            )
            if followup and not asks and not engine_words:
                fallback = ""  # a nod needs no template: nothing, rather than a hint again
        fallback, note = self._cool_hint(facts, fallback)
        return f"{checks}{directive}{note}", fallback
