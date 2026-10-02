"""Who "you" is.

The harness is written for the coach's eyes, so in it "you" is the coach, the person being
coached is "the player" and the other side is "the opponent". The facts and ideas are
first written for the side to move as "you"/"your" (that is how they read when the coach
says them aloud), so the harness rewrites them into the third person before it shows them:
for the player's turn "you" is the player; for the opponent's turn "you" is the opponent
and "your opponent" is the player. The coach's own speech is not touched: there "you" is
the player, as it should be.

The rewrite is one pass over the text. A table applied pattern by pattern turned "your
opponent's" into "your" and then, on the next pattern, into "their" -- the opponent's
pawn became the player's.
"""

from __future__ import annotations

import re

# One alternation, longest forms first: "your opponent's" before "your", "you are" before a
# bare "you". A subject ("you"/"they") takes its verb along so the verb can agree.
_PRONOUNS = re.compile(
    r"\b(?:"
    r"(?P<other_poss>your opponent's|theirs|their)"
    r"|(?P<other>your opponent|them)"
    r"|(?P<own_poss>yours|your)"
    r"|(?P<subject>you|they)"
    r"(?: (?P<adverb>still|now|just|already|also|only|never|no longer))?"
    r"(?: (?P<verb>[a-z]+))?"
    r")\b",
    re.IGNORECASE,
)
_MODALS = frozenset(
    {"can", "cannot", "could", "would", "should", "must", "may", "might", "will", "did", "had"}
)
# Words that follow a subject without being its verb: "you, the player" or "they and".
_NOT_VERBS = frozenset(
    {
        "the", "a", "an", "it", "them", "him", "her", "and", "or", "but", "to", "of", "on",
        "in", "at", "by", "for", "with", "from", "up", "down", "back", "both", "each", "all",
    }
)  # fmt: skip
# Words after "you" that are never its verb: "it gives you a pawn instead", "costs you
# twice" became "the player insteads", "the player twices" (the deep prompt review, 2026-10-01).
_NOT_VERBS |= frozenset(
    {
        "instead", "twice", "better", "worse", "more", "less", "too", "again", "first", "here",
        "there", "nothing", "something", "enough", "time", "tempo", "material", "trouble",
        "much", "little", "plenty", "room", "options", "chances", "any", "some", "no", "one",
        "two", "three", "an", "extra", "free", "away", "out", "over", "into", "onto", "off",
    }
)  # fmt: skip
# The word before "you" that makes it an object: a preposition, or a verb taking one --
# "it attacks you twice", "this leaves you a pawn down".
_OBJECT_AFTER = frozenset(
    {
        "to", "for", "with", "against", "than", "from", "at", "on", "by", "of", "into",
        "onto", "toward", "towards", "behind", "past", "around", "near", "beside", "like",
        "give", "gives", "leave", "leaves", "let", "lets", "cost", "costs", "get", "gets",
        "make", "makes", "offer", "offers", "hand", "hands", "show", "shows", "tell", "tells",
        "ask", "asks", "attack", "attacks", "hit", "hits", "check", "checks", "pin", "pins",
        "fork", "forks", "threaten", "threatens", "help", "helps", "allow", "allows",
        "force", "forces", "beat", "beats", "hurt", "hurts", "trap", "traps", "keep", "keeps",
        "put", "puts", "send", "sends", "stop", "stops", "deny", "denies", "answer", "answers",
    }
)  # fmt: skip
# Verbs ending in -ed that are present tense, not past.
_PRESENT_ED = frozenset({"need", "succeed", "proceed", "exceed", "feed", "bleed", "speed", "shed"})


def _conjugate(verb: str) -> str:
    """The verb of "you <verb>" as it reads after "the player": have -> has, win -> wins."""
    low = verb.lower()
    if low in _MODALS or low in ("is", "was", "has"):
        return verb
    if low == "are":
        return "is"
    if low == "were":
        return "was"
    if low == "have":
        return "has"
    if low == "do":
        return "does"
    if low == "don't":
        return "doesn't"
    if low.endswith("ed") and low not in _PRESENT_ED:
        return verb  # past tense: "you played" -> "the player played"
    if low.endswith(("s", "sh", "ch", "x", "z", "o")):
        return verb + "es"
    if low.endswith("y") and len(low) > 1 and low[-2] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def third_person(text: str, *, side: str = "player") -> str:
    """A fact written for the side to move as "you", reworded for the harness. `side` is
    who "you" is: "player" (the default) or "opponent".

    "your king is still in the centre" -> "the player's king is still in the centre";
    "you win a pawn" -> "the player wins a pawn"; with side="opponent", "your opponent's
    isolated pawn" -> "the player's isolated pawn"."""
    own = "the player" if side == "player" else "the opponent"
    other = "the opponent" if side == "player" else "the player"

    def rewrite(match: re.Match[str]) -> str:
        if match.group("other_poss"):
            return f"{other}'s"
        if match.group("other"):
            return other
        if match.group("own_poss"):
            return f"{own}'s"
        subject = own if match.group("subject").lower() == "you" else other
        adverb = match.group("adverb")
        verb = match.group("verb")
        out = subject
        if adverb:
            out += f" {adverb}"
        before = re.search(r"([A-Za-z']+)[^A-Za-z']*$", match.string[: match.start()])
        is_object = before is not None and before.group(1).lower() in _OBJECT_AFTER
        if verb:
            kept = verb.lower() in _NOT_VERBS or is_object
            out += " " + (verb if kept else _conjugate(verb))
        return out

    out = _PRONOUNS.sub(rewrite, text)
    # A text that began with a capital ("Your king ...") keeps one ("The player's king ...").
    return out[:1].upper() + out[1:] if text[:1].isupper() else out
