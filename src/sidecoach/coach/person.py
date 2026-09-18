"""Who "you" is.

The harness is written for the coach's eyes, so in it "you" is the coach, the person being
coached is "the player" and the other side is "the opponent". The facts and ideas are
first written for the side to move as "you"/"your" (that is how they read when the coach
says them aloud), so the harness rewrites them into the third person before it shows them.
The coach's own speech is not touched: there "you" is the player, as it should be.
"""

from __future__ import annotations

import re

# Whole words; the longer forms first so "your opponent's" is not eaten by "your".
_FIXED: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\byour opponent's\b", re.IGNORECASE), "the opponent's"),
    (re.compile(r"\byour opponent\b", re.IGNORECASE), "the opponent"),
    (re.compile(r"\byours\b", re.IGNORECASE), "the player's"),
    (re.compile(r"\byour\b", re.IGNORECASE), "the player's"),
    (re.compile(r"\btheirs\b", re.IGNORECASE), "the opponent's"),
    (re.compile(r"\btheir\b", re.IGNORECASE), "the opponent's"),
    (re.compile(r"\bthem\b", re.IGNORECASE), "the opponent"),
    (re.compile(r"\byou are\b", re.IGNORECASE), "the player is"),
    (re.compile(r"\byou were\b", re.IGNORECASE), "the player was"),
    (re.compile(r"\byou have\b", re.IGNORECASE), "the player has"),
    (re.compile(r"\byou still have\b", re.IGNORECASE), "the player still has"),
    (re.compile(r"\bthey are\b", re.IGNORECASE), "the opponent is"),
    (re.compile(r"\bthey were\b", re.IGNORECASE), "the opponent was"),
    (re.compile(r"\bthey have\b", re.IGNORECASE), "the opponent has"),
    (re.compile(r"\bthey still have\b", re.IGNORECASE), "the opponent still has"),
    (
        re.compile(
            r"\byou (can|cannot|could|would|should|must|may|will|did|do not|don't)\b", re.IGNORECASE
        ),
        r"the player \1",
    ),
    (
        re.compile(
            r"\bthey (can|cannot|could|would|should|must|may|will|did|do not|don't)\b",
            re.IGNORECASE,
        ),
        r"the opponent \1",
    ),
)
# "you <verb>" / "they <verb>": the subject changes number, so the verb takes its -s.
_SUBJECT_VERB = re.compile(
    r"\b(you|they)\b (?P<adverb>(?:still|now|just|already|also|only) )?(?P<verb>[a-z]+)\b",
    re.IGNORECASE,
)
_NOT_VERBS = {
    "the",
    "a",
    "an",
    "it",
    "them",
    "him",
    "her",
    "and",
    "or",
    "but",
    "to",
    "of",
    "on",
    "in",
    "at",
    "by",
    "for",
    "with",
    "from",
    "up",
    "down",
    "back",
    "both",
    "each",
    "all",
}


def _conjugate(verb: str) -> str:
    if verb in ("have",):
        return "has"
    if verb in ("are", "were", "is", "was") or verb.endswith("ed"):
        return verb  # already agreeing, or past tense
    if verb.endswith(("s", "sh", "ch", "x", "z")):
        return verb + "es"
    if verb.endswith("y") and len(verb) > 1 and verb[-2] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def _subject_verb(match: re.Match[str]) -> str:
    subject = "the player" if match.group(1).lower() == "you" else "the opponent"
    adverb = match.group("adverb") or ""
    verb = match.group("verb")
    if verb.lower() in _NOT_VERBS:
        return f"{subject} {adverb}{verb}"
    return f"{subject} {adverb}{_conjugate(verb)}"


def third_person(text: str) -> str:
    """A fact written for the side to move as "you", reworded for the harness: "your king
    is still in the centre" -> "the player's king is still in the centre"; "they are not in
    the game yet" -> "the opponent is not in the game yet"; "you win a pawn" -> "the player
    wins a pawn"."""
    capital = text[:1].isupper()
    for pattern, new in _FIXED:
        text = pattern.sub(new, text)
    text = _SUBJECT_VERB.sub(_subject_verb, text)
    # A text that began with a capital ("Your king ...") keeps one ("The player's king ...").
    return text[:1].upper() + text[1:] if capital else text
