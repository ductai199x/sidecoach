"""The coach: what to say, when, from which facts, checked before it is shown.

Three things keep it honest. The disclosure level decides which facts reach the model at
all, so it cannot repeat what it never saw. The level only ever moves through the
deterministic unlocks in `disclosure.py`, driven by what the user did or asked, never by
the model's own judgement. And every draft is checked twice before it is shown: for leaks
past the current level, and for claims about the board that the board contradicts. A
draft that fails is regenerated once with the problem named, and then replaced by a
template, so the user never sees a leak and never sees nothing.

The model is slow, a second or three, so it runs on a worker thread; the capture loop
hands in events and polls for messages, exactly as it does with the engine.
"""

from __future__ import annotations

import contextlib
import json
import queue
import random
import re
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import chess

from sidecoach.coach.annotations import annotations_for, mentions_in
from sidecoach.coach.conversation import Conversation, PlyRecord
from sidecoach.coach.disclosure import (
    Disclosure,
    PositionFacts,
    Unlock,
    leak_check,
)
from sidecoach.coach.facts import describe_move, facts_from_analysis
from sidecoach.coach.named import EngineLike as NamedEngine
from sidecoach.coach.named import analyse_named, disclosed_frames
from sidecoach.coach.named import spoken as spoken_examination
from sidecoach.coach.policy import InterjectionPolicy, TimeControl, time_control_of
from sidecoach.coach.prompt import system_message
from sidecoach.coach.summarise import summarise
from sidecoach.events import (
    Analysis,
    Annotation,
    ClockObservation,
    CoachMessage,
    DisclosureLevel,
    Event,
    GameEnded,
    GameStarted,
    MoveAssessment,
    MovePlayed,
    Resync,
    ScreenState,
    ScreenStateChanged,
    Severity,
    Side,
    Takeback,
    Trigger,
)
from sidecoach.llm.base import ChatMessage, LLMProvider

MAX_TOKENS = 220
# A move that lost less than this gets a two-word confirmation, without the model.
TRIVIAL_LOSS = 0.02
# The confirmations, by the move's label, in the vocabulary chess.com uses; "{san}" is the
# move. A book move is theory, a best move is the engine's own, the rest are fine moves.
VERDICTS: dict[str, list[str]] = {
    "book": ["{san}: book.", "{san}, still theory.", "{san}: known territory."],
    "brilliant": ["{san}: brilliant.", "{san} -- that is a find."],
    "great": ["{san}: great move.", "{san} -- the one move that works."],
    "best": [
        "{san}: best.",
        "{san} -- that's the move.",
        "{san}, spot on.",
        "{san}: exactly.",
        "{san}: the engine's choice too.",
    ],
    "excellent": ["{san}: excellent.", "{san}: strong.", "{san} -- very good.", "{san}: nice."],
    "good": ["{san}: good.", "{san}: fine.", "{san}, solid.", "{san} works.", "{san}: sound."],
}
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


def vs_directive(candidates: int = VS_CANDIDATES) -> str:
    count = _NUMBER_WORDS.get(candidates, str(candidates))
    return (
        f"\n\nInstead of one reply, give {count} different ways you could say this, varied in "
        "wording and in the angle each takes, sampled from your full range rather than the "
        "single most obvious phrasing. Put each on its own line as: [p=0.30] then the reply. "
        "Each reply obeys every rule above and keeps to the length asked, on that one line. "
        "The p values are how likely you would be to choose each; they need not sum to one."
    )


VS_DIRECTIVE = vs_directive()
_VS_MARKER = re.compile(r"\[\s*p?\s*=?\s*([0-9]*\.?[0-9]+)\s*\]\s*", re.IGNORECASE)
# A marker fragment left in a candidate's text: a whole one, or one cut off mid-way.
_STRAY_MARKER = re.compile(r"\s*\[\s*p\s*=\s*[0-9.]*\]?\s*")
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
# Where a reply's last complete sentence ends: terminal punctuation, then any closing
# quote or bracket. What follows the last of these is a sentence the model never finished.
_SENTENCE_END = re.compile(r"[.!?…][\"'”’)\]]*(?=\s|$)")


def _vs_wrap(messages: list[ChatMessage], candidates: int = VS_CANDIDATES) -> list[ChatMessage]:
    """Append the verbalized-sampling directive to the final user turn."""
    if not messages:
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


def _parse_candidates(raw: str) -> list[tuple[float, str]]:
    """The verbalized replies as (probability, text). A reply with no markers is one candidate.

    The text of each candidate runs from its marker to the next, so a reply that wraps onto
    another line is kept whole. A model that ignores the format still yields one usable reply.
    Every candidate is cut back to its last complete sentence.
    """
    cleaned = _THINK_BLOCK.sub("", raw).strip()
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
    # What the verifier found, when one runs: per candidate the statements the messages do
    # not support, and whether the reply spoken was a repair of the best candidate.
    verified: list[list[str]] = field(default_factory=list)
    repaired: bool = False


@dataclass
class _Position:
    ply: int
    board: chess.Board
    analysis: Analysis | None = None
    facts: PositionFacts | None = None
    disclosure: Disclosure = field(default_factory=Disclosure)
    last_move: str | None = None


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
    # And along what the directive disclosed: the line the player asked about, examined.
    if request.disclosed:
        frames += disclosed_frames(request.board, request.disclosed.splitlines())
    return frames


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

    A draft with a leak or a false claim about the board is sent back once with the
    problem named; a second failure is replaced by the template, and the failure counted.
    With a `verifier`, every candidate that passes those checks is also read against the
    messages by a second model, the clean ones are sampled from, and when none is clean the
    likeliest is repaired once: test-time verification, on top of the demonstrations in C.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        draft_check: Callable[[str, chess.Board, Sequence[chess.Board]], list[str]] | None = None,
        verifier: LLMProvider | None = None,
    ) -> None:
        self.provider = provider
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
        2510.01171) recovers its range; the leak and board checks then filter that set, and a
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
        while attempts < 2:
            attempts += 1
            stats: dict[str, Any] = {}
            try:
                completion = self.provider.chat(
                    request.system, messages, max_tokens=VS_MAX_TOKENS, temperature=VS_TEMPERATURE
                )
                raw = completion.text
                stats = dict(completion.stats)
            except Exception as exc:  # noqa: BLE001
                self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                raw = ""
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
            # The second try carries the complaint inside the directive: appending the failed
            # attempt and the complaint as two more messages doubled the prompt on a long
            # game (131k tokens became 276k, past the model's context).
            rejected = "; ".join(problems)
            shown = raw.strip()[:600] or "(no reply)"
            retry = dict(messages[-1])
            retry["content"] = (
                str(retry.get("content", ""))
                + f"\n\nYour first attempt was:\n{shown}\n\nThose replies are not allowed: "
                f"{rejected}. Give {_NUMBER_WORDS.get(request.candidates, request.candidates)} "
                "fresh, varied replies that avoid those, keeping to the messages above."
            )
            messages = [*messages[:-1], retry]  # type: ignore[list-item]
        if problems:
            self.fallbacks += 1
            text = request.fallback
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
        )

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
        # What the user said this turn, what the directive itself disclosed (the engine on
        # what they asked about) and the moves of the game so far are not leaks: a move
        # already played may be discussed in full, whatever the level in force.
        said = " ".join(
            filter(None, [request.user_text, request.disclosed, *sorted(request.played)])
        )
        found = [
            f"withheld {item}"
            for item in leak_check(text, request.facts, request.level, said=said or None)
        ]
        if self.draft_check is not None:
            for problem in self.draft_check(text, request.board, _after_best(request)):
                san = problem.split(":", 1)[0].strip()
                if san in request.played:
                    continue  # a move of the game's history, named as such
                found.append(problem)
        return found

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
    r"\b(?:how (?:did|have|am) i (?:do|play|done|played|playing|doing)|how was (?:my|the) game|"
    r"overall|summar(?:y|ise|ize)|review (?:my|the) game|rate my (?:game|play)|"
    r"where did i (?:lose|go wrong)|what (?:do|did) i keep (?:missing|doing)|"
    r"what (?:did|have) i (?:do|done) wrong|my (?:mistakes|worst moves?)|"
    r"what should i work on)\b",
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
_ASK_EVAL = re.compile(
    r"\b(?:am i (?:winning|losing|better|worse|ok|okay|in trouble)|who'?s (?:better|winning)|"
    r"how am i doing|is this (?:position )?good for me|what'?s the (?:eval|evaluation|score)|"
    r"how bad is it|am i lost)\b",
    re.IGNORECASE,
)
# A bare request for a hint, with nothing claimed: "hint", "give me a hint", "a clue".
_ASK_HINT = re.compile(
    r"^\W*(?:(?:give me |i need |can i have |i want |any |another |one more )?(?:a |an |some )?"
    r"(?:hint|clue|nudge|help|tip)s?(?: please)?)\W*$",
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


def may_carry_claims(text: str) -> bool:
    """Whether the user's words could state something about the board to check."""
    if _ASK_HINT.match(text) or _ASK_REVIEW.search(text):
        return False
    return _CLAIM_WORDS.search(text) is not None


_ASK_MOVE = re.compile(
    r"\b(best move|what (should|do|would) i (play|do)|what would you play|which move|"
    r"just tell me|what'?s the move|tell me the move|the answer)\b",
    re.I,
)
_ASK_LINE = re.compile(
    r"\b(the line|variation|continuation|what happens (after|next)|show me the line|"
    r"and then what|what follows)\b",
    re.I,
)
_SAN = re.compile(r"\b(O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\b")


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
        draft_check: Callable[[str, chess.Board, Sequence[chess.Board]], list[str]] | None = None,
        claims: Callable[[str, chess.Board, PositionFacts | None], list[str]] | None = None,
        verifier: LLMProvider | None = None,
        engine: NamedEngine | None = None,
        summarise_every: int = 0,
        summariser: LLMProvider | None = None,
    ) -> None:
        self.policy = policy or InterjectionPolicy()
        self.speaker = Speaker(provider, draft_check=draft_check, verifier=verifier)
        self.claims = claims
        # Dialogue compaction: every `summarise_every` plies the dialogue older than the
        # last `summarise_every` plies is folded into a summary by `summariser` (the
        # coach's own model when None). 0 is off.
        self.summarise_every = summarise_every
        self.summariser = summariser or provider
        # The engine that analyses a move or line the player names, on demand; without
        # one, a named move is only checked against the candidates the facts hold.
        self.engine = engine
        self.system = system_message()
        # Run the claims check on the caller's thread (replay and evaluation) instead of a
        # worker: live it is threaded so a button press never stalls the pipeline.
        self.sync_claims = False
        self.game_id: str | None = None
        self.user_side: Side = Side.WHITE
        self.finished = False  # the game is over: everything may be said
        self.board = chess.Board()
        self.positions: dict[int, _Position] = {}
        self.moves: dict[int, MovePlayed] = {}
        # The game as the model sees it: one message per move, the coach's turns between.
        self.conversation = Conversation()
        self.spoken: list[Spoken] = []
        self._current: _Position | None = None
        self._clock_peak_ms: int | None = None
        # The conversation is read from the caller's thread and from a claims check.
        self._lock = threading.RLock()
        self._last_verdict: str | None = None

    # Lifecycle

    def start(self) -> None:
        self.speaker.start()

    def close(self) -> None:
        self.speaker.close()

    @property
    def user_colour(self) -> chess.Color:
        return chess.WHITE if self.user_side == Side.WHITE else chess.BLACK

    @property
    def user_to_move(self) -> bool:
        return self.board.turn == (chess.WHITE if self.user_side == Side.WHITE else chess.BLACK)

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
            self._on_takeback(event, ts)
        elif isinstance(event, Resync):
            self._on_resync(event, ts)
        elif isinstance(event, GameEnded):
            self._current = None
            self.finished = True
            self.conversation.ended(event)
        elif isinstance(event, ScreenStateChanged):
            covered = event.state is ScreenState.COVERED
            if covered and not self.policy.covered:
                self.conversation.covered()
            self.policy.covered = covered

    def tick(self, ts: float) -> None:
        """Time passing: a hint if the user has been stuck long enough."""
        trigger = self.policy.tick(ts)
        if trigger is None or self._current is None or self._current.facts is None:
            return
        level = self._current.disclosure.apply(Unlock.STUCK) or self._current.disclosure.level
        self.policy.said(self._current.ply)
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
        position = self._current
        if position is None or position.facts is None:
            position = self._latest_with_facts()
        if position is None or position.facts is None:
            self._queue_plain(
                Trigger.USER_ASK,
                "I have not seen enough of this game yet to say. Give me a move or two.",
                ts,
            )
            return
        unlock = unlock_for(text, position.board, position.facts.candidates)
        if unlock is not None:
            position.disclosure.apply(unlock)
        level = max(position.disclosure.level, DisclosureLevel.QUESTION)
        if self.finished or position.board.is_game_over():
            # Nothing is withheld from a game that is over.
            level = DisclosureLevel.LINE
        with self._lock:
            self.conversation.user_said(text)
        if _ASK_LAST_MOVE.search(text) and (own := self._last_own_move()) is not None:
            # About the move they already played: spoken from the position it was played
            # from, with everything shown, as the lesson after a mistake is. That position
            # is past; the level in force binds only the one they face now.
            before = self.positions.get(own.move.ply - 1)
            if before is not None and before.facts is not None:
                position, level = before, DisclosureLevel.LINE
        assert position.facts is not None
        # The quick answer, shown before the model has started: the template for a bare
        # "hint", the move or the line. A question about the plan earns the hint's level
        # but not its template -- "What is your opponent's most dangerous idea?" as the
        # first of two answers to "what should I be aiming for?" was noise.
        canned = (
            unlock in (Unlock.ASK_MOVE, Unlock.ASK_LINE) or bool(_ASK_HINT.match(text))
        ) and not may_carry_claims(text)
        if canned:
            _, quick = self._instruction(Trigger.USER_ASK, position, level, text, (), None, None)
            self._queue_plain(
                Trigger.USER_ASK,
                quick,
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

        def examined() -> list[str]:
            """The engine on what they asked about; its plain-words form goes out at once
            as the coach's turn (correct by construction), and the model builds on it."""
            found = self._named_analysis(text, position, level)
            if found:
                self._queue_plain(
                    Trigger.USER_ASK,
                    " ".join(spoken_examination(f) for f in found),
                    ts,
                    ply=position.ply,
                    level=level,
                    fen=position.board.fen(),
                    now=True,
                    record_now=True,
                )
            return found

        if self.claims is None or not may_carry_claims(text):
            verdicts: list[str] = []
            if may_carry_claims(text):
                verdicts = examined()
            self._speak(
                Trigger.USER_ASK,
                position,
                level,
                ts,
                user_text=text,
                verdicts=verdicts,
                quick=canned or bool(verdicts),
            )
            return

        def checked() -> None:
            try:
                verdicts = self.claims(text, position.board, position.facts)  # type: ignore[misc]
            except Exception:  # noqa: BLE001
                verdicts = []
            # The engine on what they asked about, examined now and spoken first.
            found = examined()
            verdicts = [*verdicts, *found]
            self._speak(
                Trigger.USER_ASK,
                position,
                level,
                ts,
                user_text=text,
                verdicts=verdicts,
                quick=bool(found),
            )

        if self.sync_claims:
            checked()  # replay and evaluation: the request must exist when ask() returns
        else:
            threading.Thread(target=checked, name="sidecoach-claims", daemon=True).start()

    def hint(self, ts: float) -> None:
        self.ask("hint", ts)

    def ask_move(self, ts: float) -> None:
        self.ask("what should I play", ts)

    def ask_line(self, ts: float) -> None:
        self.ask("show me the line", ts)

    def poll(self) -> list[CoachMessage]:
        out: list[CoachMessage] = []
        for spoken in self.speaker.poll():
            self.spoken.append(spoken)
            if not spoken.message.text:
                # A failed model reply with nothing to fall back on: kept for the log (its
                # calls say why), said to no one.
                continue
            if not spoken.recorded:
                with self._lock:
                    self.conversation.coach_said(spoken.message.text)
            out.append(spoken.message)
        return out

    # Internals

    def _new_game(self, event: GameStarted, ts: float) -> None:
        self.game_id = event.game_id
        self.user_side = event.user_side
        self.board = chess.Board(event.start_fen)
        self.positions = {0: _Position(ply=0, board=self.board.copy())}
        self.moves = {}
        self.conversation.new_game(event)
        self.finished = False
        self._current = self.positions[0]
        self.policy.new_position(0, ts, user_to_move=self.user_to_move)
        self._clock_peak_ms = None
        self.policy.set_remaining(None)
        self.policy.time_control = TimeControl.UNTIMED

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

    def _on_move(self, event: MovePlayed, ts: float) -> None:
        self.board = chess.Board(event.fen_after)
        self.moves[event.ply] = event
        self.conversation.move(event)
        position = _Position(ply=event.ply, board=self.board.copy(), last_move=event.san)
        self.positions[event.ply] = position
        for old in [ply for ply in self.positions if ply < event.ply - 6]:
            del self.positions[old]
        self._current = position
        self.policy.new_position(event.ply, ts, user_to_move=self.user_to_move)
        if self.summarise_every and event.ply % self.summarise_every == 0:
            with self._lock:
                summarise(self.conversation, self.summariser, keep=self.summarise_every)

    def _on_takeback(self, event: Takeback, ts: float) -> None:
        """Moves withdrawn: the conversation says so, and the game goes on from the position
        the board is back at; the replacement moves arrive as ordinary moves."""
        self.conversation.takeback(event)
        self.board = chess.Board(event.fen)
        active = self.conversation.active_records()
        ply = active[-1].move.ply if active else 0
        for gone in [p for p in self.positions if p > ply]:
            del self.positions[gone]
        position = self.positions.get(ply)
        if position is None:
            position = _Position(ply=ply, board=self.board.copy())
            self.positions[ply] = position
        self._current = position
        self.policy.new_position(ply, ts, user_to_move=self.user_to_move)

    def _on_resync(self, event: Resync, ts: float) -> None:
        """The board was re-read: the position jumped; the move numbering restarts."""
        self.board = chess.Board(event.fen)
        self.conversation.resync(event, self.board)
        self.positions = {0: _Position(ply=0, board=self.board.copy())}
        self.moves = {}
        self._current = self.positions[0]
        self.policy.new_position(0, ts, user_to_move=self.user_to_move)

    def _on_analysis(self, event: Analysis, ts: float) -> None:
        position = self.positions.get(event.ply)
        if position is None or position.board.fen() != event.fen:
            return
        position.analysis = event
        position.facts = facts_from_analysis(
            position.board, event, last_move=position.last_move, user=self.user_colour
        )
        self.conversation.analysed(event, position.facts)
        if position is self._current and self.user_to_move:
            trigger = self.policy.on_position_analysed(event.ply, decisive=event.decisive)
            if trigger is not None:
                level = position.disclosure.apply(Unlock.DECISIVE) or position.disclosure.level
                self.policy.said(position.ply)
                self._speak(trigger, position, level, ts)

    def _on_assessment(self, event: MoveAssessment, ts: float) -> None:
        self.conversation.assess(event)
        move = self.moves.get(event.ply)
        if move is None or not move.by_user:
            return
        before = self.positions.get(event.ply - 1)
        if before is None or before.facts is None:
            return
        trigger = self.policy.on_user_move_assessed(event)
        if trigger is None:
            return
        before.disclosure.apply(Unlock.MOVE_PLAYED)
        # Every move of the player's gets a word from the coach. A lecture is only for a
        # move that cost something; a fine move gets one sentence, and that sentence comes
        # from the model too, so it can speak to the move and the game rather than repeat a
        # stock phrase -- with a warm prefix the call is a few seconds, and the two-word
        # template stands in if the reply fails. Blitz keeps the template: speed wins there.
        if trigger is Trigger.POST_MOVE and event.severity is Severity.NONE:
            assert before.facts is not None
            best = before.facts.candidates[0] if before.facts.candidates else None
            routine = event.expected_loss < TRIVIAL_LOSS or best is None or best == move.san
            if not routine or self.policy.time_control is not TimeControl.BLITZ:
                self._speak(
                    trigger,
                    before,
                    DisclosureLevel.LINE,
                    ts,
                    assessment=event,
                    played=move,
                    candidates=VS_ROUTINE_CANDIDATES if routine else VS_CANDIDATES,
                )
                return
            text = self._verdict(move.san, event.label.value if event.label else None)
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
        self._speak(trigger, before, DisclosureLevel.LINE, ts, assessment=event, played=move)

    def _verdict(self, san: str, label: str | None) -> str:
        """The two-word confirmation of a fine move, worded by its label and never the same
        as the last one: a game of "Nf3: good. Bb5: good. Nc3: good." wears thin."""
        phrases = VERDICTS.get(label or "", VERDICTS["good"])
        choices = [p for p in phrases if p != self._last_verdict] or phrases
        chosen = random.choice(choices)
        self._last_verdict = chosen
        return chosen.format(san=san)

    def _opening_so_far(self) -> str | None:
        """The opening the game was in: the last name the book gave."""
        with self._lock:
            for record in reversed(self.conversation.records):
                if record.analysis is not None and record.analysis.opening:
                    return record.analysis.opening
        return None

    def _named_analysis(self, text: str, position: _Position, level: DisclosureLevel) -> list[str]:
        """The engine's word on what the player asked about: the model decides what to
        examine, the engine examines it; one line per probe, none when nothing was named.
        A piece or square named needs no engine; a line does."""
        try:
            return analyse_named(
                text,
                position.board,
                self.engine,
                user_colour=self.user_colour,
                level=level,
                # Without an engine only pieces and squares can be examined, and the
                # words name those well enough: no planner call.
                planner=self.speaker.provider if self.engine is not None else None,
            )
        except Exception as exc:  # noqa: BLE001
            self.speaker.last_error = f"named analysis: {type(exc).__name__}: {exc}"
            return []

    def _last_own_move(self) -> PlyRecord | None:
        """The player's own move, when it is the last move played and has its verdict.
        Once the opponent has replied, "that move" is theirs: the generic ask handles it."""
        with self._lock:
            records = self.conversation.active_records()
            if records and records[-1].move.by_user and records[-1].assessment is not None:
                return records[-1]
        return None

    def _standing_template(self, position: _Position) -> str:
        """Who stands better, from the engine's score for the position, in plain words."""
        analysis = position.analysis
        if analysis is None or not analysis.lines:
            return "I have no evaluation for this yet."
        score = analysis.lines[0].expected_score
        if position.board.turn != self.user_colour:
            score = 1.0 - score
        if score >= 0.85:
            words = "you are winning"
        elif score >= 0.62:
            words = "you are better"
        elif score > 0.38:
            words = "it is about equal"
        elif score > 0.15:
            words = "you are worse"
        else:
            words = "you are losing"
        material = ""
        if position.facts is not None and position.facts.material:
            material = f"; material: {position.facts.material}"
        return f"Right now {words}{material}."

    def _review_template(self) -> str:
        """The record in a sentence, for when the model's review fails the checks: the
        tally and where ground was lost, from the same records the messages carry."""
        with self._lock:
            judged = [
                r
                for r in self.conversation.judged_records()
                if r.move.by_user and r.assessment is not None
            ]
            if not judged:
                return "I have not seen enough of this game yet to say."
            counts: dict[str, int] = {}
            lost: list[str] = []
            for r in judged:
                assert r.assessment is not None
                label = (
                    r.assessment.label.value if r.assessment.label else r.assessment.severity.value
                )
                counts[label] = counts.get(label, 0) + 1
                if r.assessment.severity.value != "none":
                    lost.append(f"{self.conversation._numbered(r.move)} ({label})")
        good = sum(counts.get(k, 0) for k in ("brilliant", "great", "best", "excellent"))
        text = f"{good} of your {len(judged)} moves were best or excellent."
        if lost:
            text += " You lost ground at " + ", ".join(lost[-3:]) + "."
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
        quick: bool = False,
        candidates: int | None = None,
    ) -> None:
        assert position.facts is not None
        instruction, fallback = self._instruction(
            trigger, position, level, user_text, verdicts, assessment, played, quick=quick
        )
        if quick:
            # The template is already out as the quick answer; a model reply that fails the
            # checks adds nothing, rather than the same line a second time.
            fallback = ""
        directive = f"{self._level_line(level, position)}{instruction}"
        with self._lock:
            self.conversation.level = level
            harness = self.conversation.render()
            messages: list[ChatMessage] = [
                {"role": "user", "content": f"{harness}\n\n# Task\n{directive}"},
            ]
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
            candidates=candidates
            or (VS_ASK_CANDIDATES if trigger is Trigger.USER_ASK else VS_CANDIDATES),
            played=self._record_moves(),
        )
        self.speaker.submit(request)

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
    def _level_line(level: DisclosureLevel, position: _Position) -> str:
        """What may be said about the position the player faces, stated on the directive."""
        if position.board.is_game_over():
            return ""
        words = {
            DisclosureLevel.SILENT: "nothing about the position",
            DisclosureLevel.QUESTION: "an orienting question only; no theme, square, piece or move",
            DisclosureLevel.THEME: "the theme; no square, piece or move",
            DisclosureLevel.REGION: "the theme and the region; no piece or move",
            DisclosureLevel.PIECE: "the theme, the region and the piece; no move",
            DisclosureLevel.CANDIDATE: "the candidate moves; not the line",
            DisclosureLevel.LINE: "everything, the line included",
        }[level]
        return f"Disclosure for the position the player faces: {words}.\n\n"

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
            Spoken(message=message, level=level, problems=[], attempts=0, recorded=record_now),
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
        quick: bool = False,
    ) -> tuple[str, str]:
        """The turn's instruction to the model, and the template used if it fails.

        `quick` means the template has already been shown as an immediate answer, so the
        model is asked to build on it rather than repeat it.
        """
        facts = position.facts
        assert facts is not None
        theme = facts.theme or facts.plan or "the position"
        region = facts.region or "the whole board"
        # The theme-level template: the idea when there is one; when the sheet names none,
        # what the nature says ("The idea here is the position" was said, once).
        if facts.theme or facts.plan:
            theme_hint = f"The idea here is {theme}. What would make it count?"
        elif facts.nature and facts.nature.startswith("forcing"):
            theme_hint = "The answer is forcing: which check or capture is worth calculating first?"
        else:
            theme_hint = (
                "Nothing forcing is on. Which of your pieces is doing the least, and where "
                "would it do more?"
            )
        if trigger is Trigger.DECISIVE_QUESTION:
            return (
                "The player is to move and this is a critical moment. Ask one orienting "
                "question that makes them look for the right thing without pointing at "
                "any square, piece or move. Do not evaluate the position out loud.",
                "This is a moment to slow down. What is your opponent's most dangerous idea "
                "here, and which of your pieces is doing the least?",
            )
        if trigger is Trigger.STUCK:
            if level >= DisclosureLevel.REGION:
                fallback = f"Look at {region}. Something is going on there."
            elif level >= DisclosureLevel.THEME:
                fallback = theme_hint
            else:
                fallback = "Take stock: what changed with the last move, and what does it allow?"
            return (
                "The player has been thinking for a while and is stuck. Give one hint, at the "
                "level of what the fact sheet allows, in one or two sentences, ending with a "
                "question. Do not name a move. Take a different angle from every hint you "
                "have already given in this game; never repeat one.",
                fallback,
            )
        if trigger in (Trigger.POST_MOVE, Trigger.POST_MISTAKE) and assessment and played:
            best = facts.candidates[0] if facts.candidates else "another move"
            loss = assessment.expected_loss
            label = assessment.label.value if assessment.label else assessment.severity.value
            words = _in_words(position.board, played.san)
            better_words = _in_words(position.board, best) if facts.candidates else best
            routine = loss < TRIVIAL_LOSS or not facts.candidates or best == played.san
            if trigger is Trigger.POST_MISTAKE:
                fallback = f"{played.san} cost you. {best} was better; the idea was {theme}."
                instruction = (
                    f"The player just played {played.san} ({words}), which was a {label}: it "
                    f"cost {loss:.0%} of the game. The better move was {best} ({better_words}). "
                    f"Explain the difference as a pattern to remember, in three sentences at "
                    f"most, then ask one question that checks they see it."
                )
            elif routine:
                fallback = self._verdict(played.san, label)
                meaning = {
                    "book": "opening theory",
                    "brilliant": "a sound sacrifice",
                    "great": "the one move that keeps the advantage",
                    "best": "the engine's own choice",
                    "excellent": "as strong as the best move",
                    "good": "a sound move, a little behind the best",
                }.get(label, "a sound move")
                instruction = (
                    f"The player just played {played.san} ({words}): {label} -- {meaning}. "
                    "One short sentence, fifteen words at most, in your own voice: a nod, or "
                    "what the move does when a message says so. No question, no lecture, no "
                    "advice, no guess at what it allows or risks, and not the same words as "
                    "your last such sentence."
                )
            else:
                fallback = f"{played.san} was fine; {best} was a little tighter."
                instruction = (
                    f"The player just played {played.san} ({words}), which was fine but not "
                    f"the strongest: {best} ({better_words}) was better, by a little. One or "
                    f"two sentences. No lecture."
                )
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
            # The engine on what they asked about: theirs to hear about in full, and
            # already said out loud as the coach's last turn when `quick`.
            checks += "\n".join(engine_words) + "\n\n"
        if level >= DisclosureLevel.LINE:
            fallback = (
                f"The line is {' '.join(facts.line)}."
                if facts.line
                else "I have no line for this yet."
            )
        elif level >= DisclosureLevel.CANDIDATE:
            fallback = (
                f"The move is {facts.candidates[0]}."
                if facts.candidates
                else "I have no move for this yet."
            )
        elif level >= DisclosureLevel.PIECE and facts.piece:
            fallback = f"It is about {facts.piece}. Where does it want to go?"
        elif level >= DisclosureLevel.REGION:
            fallback = f"Look at {region}."
        elif level >= DisclosureLevel.THEME:
            fallback = theme_hint
        elif facts.nature and facts.nature.startswith("forcing"):
            # The one fact the QUESTION level allows: whether the answer is forcing. The
            # generic "opponent's idea" question pointed the wrong way in a position where
            # the player had a capture to find.
            fallback = (
                "Something forcing is on: is there a check or a capture worth calculating first?"
            )
        else:
            fallback = "What is your opponent's most dangerous idea here?"
        # A question about a fact that is never withheld -- the material, the opening, who
        # stands better -- falls back to that fact, not to the hint for the position.
        if _ASK_MATERIAL.search(said) and facts.material:
            fallback = f"Material: {facts.material}."
        elif _ASK_OPENING.search(said):
            opening = self._opening_so_far()
            fallback = (
                f"The opening was the {opening}." if opening else "We left the opening book early."
            )
        elif _ASK_EVAL.search(said):
            fallback = self._standing_template(position)
        shown = (
            " Your last turn above is the quick answer they have already been shown; build on "
            "it, do not repeat it."
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
                "Two or three sentences: what went well, the one thing to work on, in plain "
                "words. If no move has a verdict yet, say you have not seen enough to say."
            )
            # A review has a record to fall back on, not a line.
            fallback = self._review_template()
        elif _ASK_HINT.match(said):
            # A bare "hint" carries no claim to be honest about; treated as one, the model
            # answered its own last message as if the player had said it.
            directive = (
                "They asked for a hint. Give one, at the level of what the fact sheet allows, "
                "in one or two sentences, ending with a question. Do not name a move, and do "
                "not comment on anything said earlier." + (shown if quick else "")
            )
        elif level >= DisclosureLevel.LINE and _ASK_LINE.search(said) and facts.line:
            directive = (
                (
                    f"They asked to see the line and have just been shown it: "
                    f"{' '.join(facts.line)}. Give one or two short sentences on the idea "
                    "behind it. Do not answer with a question."
                )
                if quick
                else (
                    f"They asked to see the line: give it -- {' '.join(facts.line)} -- plainly, "
                    "then one short sentence on the idea behind it. Do not answer with a question."
                )
            )
        elif level >= DisclosureLevel.CANDIDATE and _ASK_MOVE.search(said) and facts.candidates:
            directive = (
                (
                    f"They asked what to play and have just been shown that it is "
                    f"{facts.candidates[0]}. Give one or two short sentences on why. Do not "
                    "answer with a question this time."
                )
                if quick
                else (
                    f"They asked what to play: tell them plainly it is {facts.candidates[0]}, "
                    "then one short sentence on why. Do not answer with a question this time."
                )
            )
        elif engine_words and quick:
            # The examination is out already; the coach adds the lesson in it, not the
            # numbers again.
            directive = (
                "They asked about a move or line and have just been shown the engine's word on "
                "it -- your last turn above. Build on it in one or two sentences: what it means "
                "for their idea, in plain chess words, saying only what that analysis and the "
                "messages state; do not repeat its numbers. Then one question that points them "
                "at the idea. Name no move the messages do not give."
            )
            fallback = ""
        elif _ASK_LAST_MOVE.search(said) and (record := self._last_own_move()) is not None:
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
            if verdict.best and verdict.severity is not Severity.NONE:
                try:
                    best = chess.Board(record.move.fen_before).san(
                        chess.Move.from_uci(verdict.best[0])
                    )
                except ValueError:
                    best = ""
            # The request's facts are those of the position the move was played from when
            # the coach has them (ask() chose that position); the idea is theirs. The theme
            # or the plan, never a static feature: "the idea was the isolated d-pawn ..."
            # became the reason a bishop retreat was an inaccuracy.
            idea = facts.theme or facts.plan or ""
            if verdict.severity is Severity.NONE:
                fallback = self._verdict(san, label)
                directive = (
                    f"They ask about the move they already played, {san}. It was not bad: the "
                    f"verdict was {label}, and it cost nothing. Tell them so plainly, in one or "
                    "two sentences, and do not invent a fault. No question."
                )
            else:
                fallback = f"{san} cost you {loss:.0%} of the game."
                if best:
                    fallback += f" {best} was better" + (f"; the idea was {idea}." if idea else ".")
                directive = (
                    f"They ask about the move they already played, {san}. Its verdict is in its "
                    f"move message and is theirs in full: it was a {label} that cost {loss:.0%} "
                    "of the game"
                    + (f", the engine preferred {best}" if best else "")
                    + (f", and the idea was {idea}" if idea else "")
                    + ". Say that plainly -- the position it was played from is past, so "
                    "nothing about it is withheld; the disclosure level binds only the position "
                    "they face now. Three sentences at most, then one question that checks they "
                    "see the pattern."
                )
        else:
            directive = (
                "Answer them directly and be honest about their own claim. If they proposed a "
                "move, or named a square or piece, you may say it back and tell them plainly "
                "whether it holds up -- when the engine does not rate their move, say it is not "
                "the strongest and, in general terms, why. When the engine's word on the move "
                "or line they named is given above, that analysis is theirs to hear in full: "
                "the score after it, how far behind the best, the reply, what the move does. "
                "Only the engine's better move or line is withheld: never reveal that unless "
                "the sheet lists it, but do point them toward the right idea with a question. "
                "What you already said in this position is under 'Said here': do not say it "
                "again in other words; give them something it did not."
            )
        return f"{checks}{directive}", fallback
