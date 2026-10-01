"""The model's review of a game's key moves (docs/game-review-plan.md, decision 3).

Asked for only when the player presses Game Review's button, about a cent a game, never ahead of
time and never for the games the background review goes through. One call a key move, all at
once: told every move together, the model lent one move's facts to another and one answer it
could not read lost the whole game (luna, 2026-09-30). In: the move with the coach's own fact
bundle for the position it was played from (what its answers are grounded in) and the review's
measures -- who played it, what it cost, the reply that punishes it, the better move and its
line. Out: a paragraph on why it mattered and how the position works, not what happened, which
the page already shows; for a bad move, a second on the better move; and a lesson in a sentence
or two.

The model names the sides only by colour, and the student's words are made from its paragraphs
(`to_student`: White and Black become "you" and "your opponent"). Asked to say "you" itself, it
spoke to whoever played the move -- about one shown paragraph in four, the student's own move
given to their opponent among them (luna at medium effort, 2026-09-30) -- and colours it does not
turn round. Each paragraph is checked in colours first: against the board (`claims.check_draft`:
whose piece stands where), for the side it gives a move to, and for a side named any other way;
the lesson, general advice, names no side, square or move. A "why" that names the better move of
the student's own mistake is shown only once the answer is (Retry asks them to find it).
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import chess

from sidecoach.analysis.score import position_score
from sidecoach.coach.claims import check_draft, straight_quotes
from sidecoach.coach.disclosure import DisclosureLevel, visible_facts
from sidecoach.coach.facts import facts_from_analysis
from sidecoach.events import ReviewedGame
from sidecoach.llm.base import Completion, LLMProvider
from sidecoach.llm.usage import Prices, Usage

# The key moves, dearest first when there are more than MOST: the review's key kinds.
PRIORITY = ("blunder", "miss", "brilliant", "great", "mistake", "inaccuracy")
BAD = {"inaccuracy", "mistake", "miss", "blunder"}
MOST = 16
# A move's answer's own tokens (a provider that thinks adds its headroom). Room for full
# paragraphs while the words are made right (Tai, 2026-09-30: "let's first get it right before
# get it concise"); to come down to about a cent a game at gpt-5.6-luna's prices after.
ANSWER_TOKENS = 1500
# A move's answer as it comes, reasoning included: gpt-5.6-luna at medium effort wrote 764
# tokens a move, 482 of them reasoning, over 55 key moves of Tai's games (2026-09-30).
TYPICAL_ANSWER = 770
# The coach's facts a review needs; the rest serve a hint in play (the question to ask, the nudge),
# repeat others (the choices, the situation) or the board itself (the FEN).
DROP = frozenset(
    {"fen", "user_side", "question", "nudge", "choices", "situation", "focus", "last_move"}
)
LINE_PLIES = 4

SYSTEM = """You are a chess coach going over a finished game with your student, one key move at a
time.

For the key move below, explain WHY it mattered and HOW the position works, as a strong coach
would at the board: the plans of both sides, the pieces and squares that count, what the move
changed, and what each side can do next because of it. The student can already see the move,
its label and the evaluation, so never just say what happened: explain it. Take the room it
needs.

Ground every concrete statement in the facts given. Name only the pieces and squares the facts
name, and only moves that appear in them. Never invent a variation. Speak as the coach who sees
the board: never mention the facts, "the line given" or "the stated" anything.

Name the two sides only as White and Black: "White's bishop on c1", "Black can take on e5",
"White's winning chances fall". Never write "you", "your", "the opponent", "he", "they" or "its"
for a side: the page turns White and Black into "you" and "your opponent" for the student, which
it can do only when every side is named by its colour.

Write these parts:
"why": a full paragraph. For an inaccuracy, mistake, miss or blunder: what went wrong and how the
other side can punish it (the best reply given may be named). When the move is the student's own,
or the student's side is unknown, never give the better move, in notation or by naming its piece
and its square: the student will try to find it first. For a brilliant or great move: what makes
it work, and why it is hard to see.
"better", for an inaccuracy, mistake, miss or blunder only: a paragraph that names the better
move given outright and walks through its line: the idea behind it and how the line works
(never the best reply, which is the other side's). The page shows "better" only once the
student has found the move or asked for it.
"lesson": a sentence or two of advice to take into the next games. Here alone speak to the
student as "you", of "your opponent" in general; no colours, squares or moves.

No headings inside a part, no lists, no move numbers, and do not open with the move or its label.

Answer in this form and nothing else:
### why
(the paragraph)
### better
(the paragraph)
### lesson
(the advice)"""


@dataclass
class ReviewWords:
    """The student's words by ply ("why", "lesson", and for a bad move "better"); the plies with
    a part dropped, and why; the plies whose "why" names the better move (shown only once the
    answer is); the model's answers by ply (in colours, for reading by hand); what the calls
    cost, and each call's measures."""

    texts: dict[int, dict[str, str]]
    dropped: dict[int, str] = field(default_factory=dict)
    reveals: set[int] = field(default_factory=set)
    answers: dict[int, str] = field(default_factory=dict)
    model: str = ""
    cost: float | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)


def key_plies(game: ReviewedGame) -> list[int]:
    """The plies of the game's key moves, in order: MOST of them at most, the dearest kinds
    first and then the costliest."""
    keyed = [
        (PRIORITY.index(m.assessment.label.value), -m.assessment.expected_loss, m.ply)
        for m in game.moves
        if m.assessment.label is not None and m.assessment.label.value in PRIORITY
    ]
    return sorted(ply for *_, ply in sorted(keyed)[:MOST])


# The coach's facts speak to the side to move ("your bishop", "if you let them"); a review
# speaks of both sides' moves, so the facts go to the model in colours. "Among them" is of
# moves ("several moves keep the balance, Bf5 and Bd6 among them").
_PRONOUNS = re.compile(
    r"(?<!among )(?<!of )(?<!between )\b(yours|your|you|their|theirs|them|they)\b", re.IGNORECASE
)


def in_colours(value: Any, mover: str) -> Any:
    """`value` (a fact, or a list or dict of them) with the side to move's pronouns turned into
    its colour and the other side's into theirs: "your bishop" -> "White's bishop"."""
    other = "Black" if mover == "White" else "White"
    names = {"your": f"{mover}'s", "yours": f"{mover}'s", "you": mover}
    names |= {"their": f"{other}'s", "theirs": f"{other}'s", "them": other, "they": other}
    if isinstance(value, str):
        return _PRONOUNS.sub(lambda m: names[m.group(1).lower()], value)
    if isinstance(value, list):
        return [in_colours(v, mover) for v in value]
    if isinstance(value, dict):
        return {k: in_colours(v, mover) for k, v in value.items()}
    return value


def _moved(board: chess.Board, uci: str) -> str:
    """What a move does, in colours: "Black's knight from a5 to c4, taking White's pawn"."""
    move = chess.Move.from_uci(uci)
    mover = "White" if board.turn == chess.WHITE else "Black"
    other = "Black" if mover == "White" else "White"
    if board.is_castling(move):
        return f"{mover} castles {'kingside' if board.is_kingside_castling(move) else 'queenside'}"
    piece = board.piece_type_at(move.from_square)
    said = (
        f"{mover}'s {chess.piece_name(piece or chess.PAWN)} from "
        f"{chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}"
    )
    if board.is_capture(move):
        taken = chess.PAWN if board.is_en_passant(move) else board.piece_type_at(move.to_square)
        said += f", taking {other}'s {chess.piece_name(taken or chess.PAWN)}"
    if move.promotion:
        said += f", promoting to a {chess.piece_name(move.promotion)}"
    return said


def move_facts(game: ReviewedGame, ply: int, user: str | None) -> dict[str, Any]:
    """What the model is told of one key move, in colours: who played it, what it did, its kind,
    the winning chances before and after, the reply that punishes it, the better move and its
    line, its motifs, and the coach's facts for the position it was played from."""
    i = ply - 1
    move = game.moves[i]
    before = game.moves[i - 1].after if i else game.start
    board = chess.Board(move.fen_before)
    previous = game.moves[i - 1].assessment.uci if i else None
    facts = facts_from_analysis(board, before, last_move=previous)
    mover = move.side.value.capitalize()
    other = "Black" if mover == "White" else "White"
    # The facts whose owner goes without saying to the side to move, named: the threats are
    # the other side's (luna gave White's Qa4+ to Black), the candidates and the line the mover's.
    owned = {
        "threats": f"{other}'s threats",
        "candidates": f"{mover}'s candidates",
        "line": f"{mover}'s best line",
    }
    position = in_colours(
        {
            owned.get(key, key): value[:LINE_PLIES] if key == "line" else value
            for key, value in visible_facts(facts, DisclosureLevel.LINE).items()
            if key not in DROP and value not in (None, [], "", False)
        },
        mover,
    )
    after = board.copy(stack=False)
    after.push_uci(move.assessment.uci)
    chances = position_score(board, before.lines)
    left = 1.0 - position_score(after, move.after.lines)  # the mover's, after
    reply = move.after.lines[0].moves[0] if move.after.lines and move.after.lines[0].moves else None
    out: dict[str, Any] = {
        "ply": ply,
        "move": move.san,
        "played by": f"{mover} (the student)" if move.side.value == user else mover,
        "what it does": _moved(board, move.assessment.uci),
        "kind": move.assessment.label.value if move.assessment.label else None,
        f"{mover}'s winning chances": f"{round(100 * chances)}% before, {round(100 * left)}% after",
        "motifs": sorted({t.name for t in move.assessment.tags}),
        "the position before it": position,
    }
    if out["kind"] in BAD:
        if reply:
            out[f"{other}'s best reply"] = after.san(chess.Move.from_uci(reply))
        # The better move outright, and its line: read out of the facts, it was mistaken for
        # the reply (luna offered Black's answer as White's better move).
        if move.assessment.best:
            line = board.copy(stack=False)
            better = []
            for uci in move.assessment.best[:LINE_PLIES]:
                better.append(line.san(chess.Move.from_uci(uci)))
                line.push_uci(uci)
            out[f"{mover}'s better move"] = better[0]
            out["its line"] = better
    return out


def prompt(game: ReviewedGame, ply: int, user: str | None, opening: str | None) -> str:
    """A call's message: the game in a line, then the key move's facts."""
    head = {
        "the student plays": user.capitalize() if user else "unknown",
        "opening": opening,
        "result": game.headers.get("Result") if game.headers else None,
    }
    return json.dumps(
        {"game": head, "key move": move_facts(game, ply, user)}, ensure_ascii=False, default=str
    )


# "### why": each part's head; the part runs to the next. Plain text, not JSON: a quote left
# unescaped in one paragraph lost a whole answer (luna, 2026-09-30).
PARTS = ("why", "better", "lesson")
_HEAD = re.compile(
    rf"^[ \t]*#{{1,6}}[ \t]*\"?({'|'.join(PARTS)})\b[^\n]*$", re.IGNORECASE | re.MULTILINE
)


def parse(text: str) -> dict[str, str]:
    """The parts an answer holds, each one paragraph; none without its "why"."""
    heads = list(_HEAD.finditer(text))
    ends = [head.start() for head in heads[1:]] + [len(text)]
    out: dict[str, str] = {}
    for head, end in zip(heads, ends, strict=False):
        paragraph = " ".join(text[head.end() : end].replace("```", " ").split())
        if paragraph:
            out[head.group(1).lower()] = paragraph
    return out if "why" in out else {}


# --- The student's words -------------------------------------------------------------------

_POSSESSIVE = re.compile(r"\b(?P<side>[Ww]hite|[Bb]lack)'s\b")
# A side's colour before its pieces: "the Black king", "White pawns"; after "the", before what
# else is a side's: "the White position". Not "White forces": that is a verb.
_PIECES = r"(?:king|queen|rook|bishop|knight|pawn|piece)s?"
_ADJECTIVE = re.compile(
    rf"\b(?:[Tt]he\s+(?P<a>[Ww]hite|[Bb]lack)\s+(?=(?:{_PIECES}|army|forces|camp|side|position|"
    rf"structure|majority|pair|player)\b)|(?P<b>White|Black)\s+(?={_PIECES}\b))"
)
# A side on its own, and the verb it governs as a subject ("Black also has"), which agrees
# with "you" once it is the student's.
_ADVERBS = (
    r"also|still|then|now|only|not only|not|already|just|instead|never|often|always|even|soon|"
    r"again|thus|therefore|perhaps|sometimes|rather|no longer|first|next|later|[a-z]+ly"
)
_BARE = re.compile(
    rf"\b(?P<side>White|Black)\b(?P<adverbs>(?:\s+(?:{_ADVERBS})\b)*)"
    r"(?:\s+(?P<verb>[a-z]+(?:'t)?)\b)?"
)
# The words after which a side starts a clause, and so governs the verb that follows it; and
# the prepositions after which it is governed.
_CLAUSE = re.compile(
    r"(?:^|[.!?;:,(\n\u2014\u2013]|\b(?:and|but|so|because|while|when|whenever|if|after|before|"
    r"until|since|that|as|although|though|whereas|unless|once|where|then|now|here|instead|"
    r"meanwhile|otherwise|yet|or|nor|which|meaning))\s*$",
    re.IGNORECASE,
)
_GOVERNED = re.compile(
    r"\b(?:for|to|against|of|with|from|by|on|at|into|onto|over|like|than|between|among|"
    r"toward|towards|upon|about|without)\s*$",
    re.IGNORECASE,
)
_AGREES = {
    "is": "are",
    "was": "were",
    "has": "have",
    "does": "do",
    "goes": "go",
    "isn't": "aren't",
    "wasn't": "weren't",
    "hasn't": "haven't",
    "doesn't": "don't",
}
# The same for the opponent the other way: luna writes a side's verbs plural at times ("White
# are down a queen"), as British English may; "your opponent" takes them singular.
_SINGULAR = {
    "are": "is",
    "were": "was",
    "have": "has",
    "do": "does",
    "aren't": "isn't",
    "weren't": "wasn't",
    "haven't": "hasn't",
    "don't": "doesn't",
}
# Verbs a review says of a side, as "White" governs them: known for verbs wherever they stand,
# where a word in -s after a side could be a noun ("gives Black chances").
_VERB_WORDS = (
    r"accepts|achieves|activates|addresses|advances|allows|answers|attacks|avoids|becomes|"
    r"begins|blocks|breaks|brings|builds|calculates|captures|castles|centralises|centralizes|"
    r"challenges|chases|chooses|clears|closes|collects|comes|commits|completes|concedes|"
    r"connects|consolidates|continues|controls|converts|covers|creates|defends|delays|"
    r"determines|develops|dictates|dominates|doubles|drives|eliminates|ends|ensures|enters|"
    r"escapes|establishes|exchanges|executes|exploits|exposes|eyes|faces|fails|finds|finishes|"
    r"fixes|follows|forces|frees|gains|gets|gives|goes|grabs|guards|handles|hangs|has|helps|"
    r"hits|holds|ignores|improves|increases|initiates|intends|invites|joins|keeps|lacks|lands|"
    r"launches|leads|leaves|lets|lifts|limits|locks|loses|maintains|makes|manages|marches|"
    r"meets|misses|mobilises|mobilizes|moves|needs|neglects|neutralises|neutralizes|obtains|"
    r"occupies|offers|opens|overlooks|pays|permits|picks|pins|places|plays|prepares|preserves|"
    r"presses|prevents|proceeds|produces|promotes|protects|provides|pulls|punishes|pursues|"
    r"pushes|puts|reaches|reacts|realises|realizes|recaptures|recognises|recognizes|recovers|"
    r"reduces|regains|regroups|reinforces|relieves|relocates|remains|removes|replies|"
    r"repositions|reroutes|resigns|resolves|responds|restricts|retains|retreats|returns|"
    r"reveals|risks|runs|sacrifices|saves|secures|seeks|sees|seizes|sends|sets|settles|shifts|"
    r"simplifies|slows|solidifies|solves|spends|stabilises|stabilizes|stands|starts|stays|"
    r"steps|stops|strengthens|strikes|supports|survives|swaps|switches|takes|targets|"
    r"threatens|tightens|trades|transfers|transforms|traps|tries|turns|undermines|unpins|uses|"
    r"vacates|waits|wants|wastes|weakens|wins|withdraws"
)
_VERBS = re.compile(_VERB_WORDS)
# A plural the model's "their" may be of, between the side and it: "Black's rooks and their
# files".
_PLURAL = re.compile(
    r"\b(?:pieces|pawns|rooks|knights|bishops|queens|squares|files|ranks|diagonals)\b"
    r"|\b(?:king|queen|rook|bishop|knight|pawn)\s+and\s+(?:\w+\s+)?(?:king|queen|rook|bishop|"
    r"knight|pawn)\b"
)
_SIDE_OR_THEIRS = re.compile(
    r"\b(?P<side>White|Black)(?P<owns>'s)?\b|(?<!of )(?<!among )(?<!between )"
    r"\b(?P<pronoun>their|they|them)\b|(?P<end>[.!?])",
    re.IGNORECASE,
)
# A further verb of the same subject: "you block the file and leave the queen", "you improve
# the rook, wait for the reply, and then use it".
_JOINED = re.compile(
    rf"(?:,\s*|\b(?:and|or|but)\s+)(?:(?:and|or|but)\s+)?(?:(?:{_ADVERBS})\s+)*"
    r"(?P<verb>[a-z]+)\b(?:\s+(?P<next>[a-z]+))?"
)
# The words a verb's object opens with: "and stabilizes your position".
_OBJECT = re.compile(
    r"the|a|an|your|their|its|his|her|this|that|these|those|every|any|no|more|both|each|all"
)
# What ends a subject's verbs: another subject ("then the knight reaches"), a colon or dash.
_NEW_SUBJECT = re.compile(
    r"\b(?:your opponent|it|he|she|which|who|that|this|these|those|they)\b|[;:(\u2014\u2013]"
    rf"|\b(?:the|a|an)\s+[a-z]+\s+(?:{_VERB_WORDS})\b"
)


def _base(verb: str) -> str:
    """A verb as "you" says it: "has" -> "have", "threatens" -> "threaten", "pushes" -> "push"."""
    if verb in _AGREES:
        return _AGREES[verb]
    if len(verb) < 4 or not verb.endswith("s") or verb.endswith(("ss", "us", "is")):
        return verb
    if verb.endswith("ies") and len(verb) > 4:
        return verb[:-3] + "y"
    if verb.endswith(("ches", "shes", "sses", "xes", "zzes")):
        return verb[:-2]
    return verb[:-1]


def _third(verb: str) -> str:
    """A verb as "your opponent" says it, from its plain form: "need" -> "needs"."""
    if verb.endswith(("s", "sh", "ch", "x", "z")):
        return verb + "es"
    if verb.endswith("y") and verb[-2:-1] not in "aeiou":
        return verb[:-1] + "ies"
    return verb + "s"


def _opens_sentence(text: str, start: int) -> bool:
    before = text[:start].rstrip()
    return not before or before[-1] in ".!?\n"


def _cased(words: str, text: str, start: int) -> str:
    return words[0].upper() + words[1:] if _opens_sentence(text, start) else words


def _owned(text: str, ours: str) -> str:
    """The model's "their", "they", "them" of the student's side, in its colour. A pronoun is
    of the side its sentence named on its own last (a subject, most often), else of the side it
    named as an owner last, else of the side named last before it -- unless a plural came
    between ("Black's rooks and their files")."""
    alone: tuple[str, int] | None = None  # this sentence's, and where each was named
    owner: tuple[str, int] | None = None
    last: tuple[str, int] | None = None
    out, at = [], 0
    for found in _SIDE_OR_THEIRS.finditer(text):
        if found.group("end"):
            alone = owner = None
            continue
        if found.group("side"):
            named = (found.group("side").capitalize(), found.end())
            last = named
            if found.group("owns"):
                owner = named
            else:
                alone = named
            continue
        side = alone or owner or last
        if side is None or side[0] != ours or _PLURAL.search(text, side[1], found.start()):
            continue
        word = found.group("pronoun").lower()
        out.append(text[at : found.start()] + (f"{ours}'s" if word == "their" else ours))
        at = found.end()
    return "".join(out) + text[at:]


def _joined(text: str) -> str:
    """The verbs joined to the student's own ("you block the file and leaves") agreeing too, in
    the same sentence and before any other subject."""
    out, at = [], 0
    for sentence in re.finditer(r"[^.!?]+[.!?]?", text):
        body, begin = sentence.group(0), sentence.start()
        for subject in re.finditer(r"\byou\b", body, re.IGNORECASE):
            if not _CLAUSE.search(body[: subject.start()]):
                continue  # "allows you to consolidate and opens": the subject is another's
            cut = _NEW_SUBJECT.search(body, subject.end())
            stop = cut.start() if cut else len(body)
            for joined in _JOINED.finditer(body, subject.end(), stop):
                verb = joined.group("verb")
                i = begin + joined.start("verb")
                named = _VERBS.fullmatch(verb) or (
                    verb.endswith("s") and _OBJECT.fullmatch(joined.group("next") or "")
                )
                if named and i >= at:
                    out.append(text[at:i] + _base(verb))
                    at = i + len(verb)
    return "".join(out) + text[at:]


def to_student(paragraph: str, user: str | None) -> str:
    """A paragraph written in colours as the student reads it: their side "you" (its verbs
    agreeing), the other "your opponent"; unchanged when their side is not known."""
    if user is None:
        return paragraph
    ours = user.capitalize()

    def possessive(found: re.Match[str]) -> str:
        mine = found.group("side").capitalize() == ours
        return _cased("your" if mine else "your opponent's", found.string, found.start())

    def adjective(found: re.Match[str]) -> str:
        side = (found.group("a") or found.group("b")).capitalize()
        words = "your " if side == ours else "your opponent's "
        return _cased(words, found.string, found.start())

    def bare(found: re.Match[str]) -> str:
        text, start = found.string, found.start()
        tail = found.group("adverbs")
        verb = found.group("verb")
        before = text[:start]
        if found.group("side") != ours:
            words = "your opponent"
            if verb is not None and not _GOVERNED.search(before):
                if verb in _SINGULAR:
                    verb = _SINGULAR[verb]
                elif _CLAUSE.search(before) and not _VERBS.fullmatch(verb):
                    verb = _third(verb) if _VERBS.fullmatch(_third(verb)) else verb
        else:
            words = "you"
            known = verb in _AGREES or bool(_VERBS.fullmatch(verb or ""))
            if verb is not None and not _GOVERNED.search(before):
                verb = _base(verb) if known or _CLAUSE.search(before) else verb
        return _cased(words, text, start) + tail + (f" {verb}" if verb is not None else "")

    text = _owned(straight_quotes(paragraph), ours)
    text = _POSSESSIVE.sub(possessive, text)
    text = _ADJECTIVE.sub(adjective, text)
    text = _BARE.sub(bare, text)
    text = re.sub(r"\byou, who (is|was|has)\b", lambda m: f"you, who {_AGREES[m.group(1)]}", text)
    return _varied(_joined(text))


def _varied(text: str) -> str:
    """ "your opponent's" said again in a sentence as "their": "your opponent needs to play where
    their pawns point" (in the student's words "their" can be no one else's)."""
    out = []
    for sentence in re.split(r"(?<=[.!?])(\s+)", text):
        first = re.search(r"\byour opponent\b", sentence, re.IGNORECASE)
        if first is None:
            out.append(sentence)
            continue
        head, tail = sentence[: first.end()], sentence[first.end() :]
        out.append(head + re.sub(r"\byour opponent's\b", "their", tail))
    return "".join(out)


# --- The checks, in colours ----------------------------------------------------------------

# A side named but by its colour: in colours, "your" and "the opponent" could be either.
_PERSON = re.compile(r"\b(you|your|yours|yourself|opponent|opponent's|opponents)\b", re.IGNORECASE)
# What general advice leaves out: a side, a square (a move ends in one).
_PARTICULAR = re.compile(r"\b(White|Black|[a-h][1-8])\b")
# A move given to a side: "White's Nf3", "Black played Nc6", "White could have played Bc4".
_GIVEN = (
    r"(?:'s|\s+(?:played|plays|chose|chooses|went\s+for|goes\s+for|"
    r"(?:can|could|should|must|would|will)(?:\s+have)?\s+play(?:ed)?|"
    r"(?:answer|answers|answered|reply|replies|replied|respond|responds|responded)\s+with))"
)


def _frames(game: ReviewedGame, ply: int) -> list[chess.Board]:
    """The positions a paragraph may speak of: after the move, along the reply's line, and along
    the best line from before it."""
    move = game.moves[ply - 1]
    before = chess.Board(move.fen_before)
    frames = []
    after = before.copy(stack=False)
    after.push_uci(move.assessment.uci)
    frames.append(after.copy(stack=False))
    if move.after.lines:
        for uci in move.after.lines[0].moves[:LINE_PLIES]:
            after.push_uci(uci)
            frames.append(after.copy(stack=False))
    along = before.copy(stack=False)
    for uci in move.assessment.best[:LINE_PLIES]:
        along.push_uci(uci)
        frames.append(along.copy(stack=False))
    return frames


def _players(game: ReviewedGame, ply: int) -> dict[str, str]:
    """Who plays each move a paragraph on move `ply` may name, by colour: the move and the
    reply's line after it, the better move's line; a move both sides play there is left out."""
    move = game.moves[ply - 1]
    start = chess.Board(move.fen_before)
    reply = move.after.lines[0].moves[:LINE_PLIES] if move.after.lines else []
    seen: dict[str, set[str]] = {}
    for line in ([move.assessment.uci, *reply], move.assessment.best[:LINE_PLIES]):
        board = start.copy(stack=False)
        for uci in line:
            played = chess.Move.from_uci(uci)
            side = "White" if board.turn == chess.WHITE else "Black"
            seen.setdefault(board.san(played).rstrip("+#"), set()).add(side)
            board.push(played)
    return {san: sides.pop() for san, sides in seen.items() if len(sides) == 1}


def wrong_side(paragraph: str, game: ReviewedGame, ply: int) -> str:
    """A move the paragraph gives to the side that does not play it ("Black played Nf3" of
    White's Nf3), in words; empty when none."""
    text = straight_quotes(paragraph)
    for san, side in _players(game, ply).items():
        found = re.search(rf"\b(White|Black){_GIVEN}\s+{re.escape(san)}(?![\w])", text)
        if found and found.group(1) != side:
            return f"{found.group(0)}: {san} is {side}'s"
    return ""


def names_the_answer(text: str, game: ReviewedGame, ply: int, user: str | None = None) -> bool:
    """Whether a paragraph on the student's bad move gives the better move in notation: it is
    Retry's answer. The opponent's moves are not the student's to find (with the student's side
    unknown, every bad move is kept back)."""
    move = game.moves[ply - 1]
    label = move.assessment.label.value if move.assessment.label else None
    theirs = user is not None and move.side.value != user
    if label not in BAD or not move.assessment.best or theirs:
        return False
    board = chess.Board(move.fen_before)
    best = board.san(chess.Move.from_uci(move.assessment.best[0])).rstrip("+#")
    return re.search(rf"(?<![\w]){re.escape(best)}(?![\w])", text) is not None


def problems(part: str, paragraph: str, game: ReviewedGame, ply: int) -> str:
    """What stops a part on move `ply` from being shown, in words; empty when nothing. "why"
    and "better", written in colours: a side named but by its colour, a piece the board puts
    elsewhere or with the other side, a move given to the side that does not play it. The
    lesson, general advice: a side, a square or a move named."""
    if part == "lesson":
        named = _PARTICULAR.search(paragraph)
        return f'names "{named.group(1)}"' if named else ""
    person = _PERSON.search(paragraph)
    if person:
        return f'names a side "{person.group(1)}"'
    board = chess.Board(game.moves[ply - 1].fen_before)
    found = check_draft(paragraph, board, after=_frames(game, ply))
    if found:
        return "; ".join(f"{p.text}: {p.reason}" for p in found)
    return wrong_side(paragraph, game, ply)


def checked(
    parts: dict[str, str], game: ReviewedGame, ply: int, user: str | None
) -> tuple[dict[str, str], str, bool]:
    """A move's parts the checks pass, in the student's words, each on its own; why a part was
    dropped (empty when none); whether its "why" names the better move."""
    label = game.moves[ply - 1].assessment.label
    # "better" is a bad move's alone: a good move was the better move.
    wanted = PARTS if label is not None and label.value in BAD else ("why", "lesson")
    kept: dict[str, str] = {}
    reasons = []
    for part in wanted:
        if part not in parts:
            continue
        problem = problems(part, parts[part], game, ply)
        if problem:
            reasons.append(f"{part}: {problem}")
            if part == "why":
                # Wrong about the board or a side: the model misread the move, so nothing it
                # said of it stands.
                return {}, "; ".join(reasons), False
            continue
        kept[part] = to_student(parts[part], user)
    reveals = "why" in kept and names_the_answer(parts["why"], game, ply, user)
    return kept, "; ".join(reasons), reveals


def write(
    game: ReviewedGame,
    provider: LLMProvider,
    *,
    user: str | None,
    opening: str | None,
    prices: Prices,
    thinking: bool = False,
    answered: dict[int, str] | None = None,
    on_answer: Callable[[int, str, dict[str, Any]], None] | None = None,
) -> ReviewWords:
    """The model's review of the game's key moves, checked: a call a move, all at once. Without
    thinking unless asked: a small model reasons worse about chess, not better (Tai,
    2026-09-30). A call that fails drops its move; when every one fails (and none was answered
    before), the first failure is raised. `answered`: the answers already paid for (words the
    app was closed in the middle of), not asked again; `on_answer(ply, text, stats)` is handed
    each new answer the moment it arrives, to keep it. The cost is of the new calls."""
    answered = dict(answered or {})
    plies = key_plies(game)
    words = ReviewWords(texts={}, model=getattr(provider, "model", ""))
    if not plies:
        return words
    todo = [p for p in plies if p not in answered]
    at_effort = getattr(provider, "with_effort", None)
    if not thinking and at_effort is not None:
        # A model that reasons by its effort (OpenAI's) reasons unless told not to.
        provider = at_effort("none")

    def ask(ply: int) -> Completion | Exception:
        try:
            return provider.chat(
                SYSTEM,
                [{"role": "user", "content": prompt(game, ply, user, opening)}],
                max_tokens=ANSWER_TOKENS,
                temperature=0.4,
                thinking=thinking,
            )
        except Exception as failed:  # noqa: BLE001 -- one move's call; the rest stand
            return failed

    replies: dict[int, Completion | Exception] = {}
    if todo:
        with ThreadPoolExecutor(max_workers=len(todo)) as pool:
            asked = {pool.submit(ask, ply): ply for ply in todo}
            for done in as_completed(asked):
                ply, reply = asked[done], done.result()
                replies[ply] = reply
                if on_answer is not None and not isinstance(reply, Exception):
                    with contextlib.suppress(Exception):  # not kept only costs a resume
                        on_answer(ply, reply.text, reply.stats)
    failures = [r for r in replies.values() if isinstance(r, Exception)]
    if todo and len(failures) == len(todo) and not answered:
        raise failures[0]
    spent = Usage()
    for ply in plies:
        reply = replies.get(ply)
        if isinstance(reply, Exception):
            words.dropped[ply] = f"the call failed: {reply}"
            continue
        if reply is not None:
            spent.record(reply.stats)
            words.calls.append(reply.stats)
            text = reply.text
        else:
            text = answered[ply]
        words.answers[ply] = text
        kept, dropped, reveals = checked(parse(text), game, ply, user)
        if kept:
            words.texts[ply] = kept
        if dropped:
            words.dropped[ply] = dropped
        if reveals:
            words.reveals.add(ply)
    words.cost = spent.cost(prices)
    return words


def estimate(
    game: ReviewedGame,
    user: str | None,
    opening: str | None,
    prices: Prices,
    answered: set[int] | frozenset[int] = frozenset(),
) -> float | None:
    """What the calls come to at `prices` (dollars), as the button says it: the prompts as sent,
    and a typical answer a move (TYPICAL_ANSWER); the moves `answered` already not asked."""
    from sidecoach.llm.usage import CHARS_PER_TOKEN

    if not prices.known:
        return None
    assert prices.prompt is not None and prices.completion is not None
    plies = [p for p in key_plies(game) if p not in answered]
    sent = sum(len(SYSTEM) + len(prompt(game, ply, user, opening)) for ply in plies)
    answers = len(plies) * TYPICAL_ANSWER
    return sent / CHARS_PER_TOKEN / 1e6 * prices.prompt + answers / 1e6 * prices.completion
