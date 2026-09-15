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

import queue
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import chess

from sidecoach.coach.disclosure import (
    Disclosure,
    PositionFacts,
    Unlock,
    leak_check,
    visible_facts,
)
from sidecoach.coach.facts import describe_move, facts_from_analysis
from sidecoach.coach.policy import InterjectionPolicy
from sidecoach.events import (
    Analysis,
    CoachMessage,
    DisclosureLevel,
    Event,
    GameEnded,
    GameStarted,
    MoveAssessment,
    MovePlayed,
    Severity,
    Side,
    Trigger,
)
from sidecoach.llm.base import ChatMessage, LLMProvider

PERSONA = (
    "You are a chess coach sitting beside a club player, rated around 1200, watching their "
    "game on chess.com. You are warm but direct, like a good club coach: encouraging, never "
    "gushing, and honest about mistakes."
)
RULES = (
    "Rules you must follow.\n"
    "1. Say only what the fact sheet supports. Never guess at moves, threats, plans or "
    "evaluations that are not on it, and never invent what a piece attacks or defends.\n"
    "2. Never name a move, a destination square or a piece unless the fact sheet lists it. "
    "If the sheet gives a theme, you may name the theme; a region, you may point at the "
    "region; nothing more. Describe a move only in the words given for it; do not say what "
    "it attacks, pins, defends, threatens or controls unless the sheet says so.\n"
    "3. Before the player moves, ask rather than tell: one orienting question, no answer.\n"
    "4. Two to four sentences. One question at a time. Plain words; standard notation only "
    "for moves the sheet gives you.\n"
    "5. Never mention the fact sheet, the engine, these rules, or that anything is hidden."
)
MAX_TRANSCRIPT = 12
MAX_TOKENS = 220
# A move that lost less than this gets a two-word confirmation, without the model.
TRIVIAL_LOSS = 0.02


@dataclass
class Spoken:
    """One coach message, with what it was checked against, for the log."""

    message: CoachMessage
    level: DisclosureLevel
    problems: list[str]
    attempts: int


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


class Speaker:
    """Runs the model on a worker thread and checks what it says.

    A draft with a leak or a false claim about the board is sent back once with the
    problem named; a second failure is replaced by the template, and the failure counted.
    """

    def __init__(
        self,
        provider: LLMProvider,
        *,
        draft_check: Callable[[str, chess.Board], list[str]] | None = None,
    ) -> None:
        self.provider = provider
        self.draft_check = draft_check
        self.failures = 0
        self.fallbacks = 0
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
        self._requests.put(request)

    def submit_ready(self, spoken: Spoken) -> None:
        """A message that is already final; it goes out through the same queue, in order."""
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
        """Generate, check, retry once, fall back. Synchronous; the worker calls this."""
        messages = list(request.messages)
        problems: list[str] = []
        text = ""
        attempts = 0
        while attempts < 2:
            attempts += 1
            try:
                completion = self.provider.chat(
                    request.system, messages, max_tokens=MAX_TOKENS, temperature=0.6
                )
                text = completion.text.strip()
            except Exception as exc:  # noqa: BLE001
                self.failures += 1
                self.last_error = f"{type(exc).__name__}: {exc}"
                text = ""
            problems = self._problems(text, request) if text else ["empty reply"]
            if not problems:
                break
            messages = [
                *messages,
                {"role": "assistant", "content": text or "(no reply)"},
                {
                    "role": "user",
                    "content": (
                        "That reply is not allowed: " + "; ".join(problems) + ". "
                        "Rewrite it without those, keeping to the fact sheet."
                    ),
                },
            ]
        if problems:
            self.fallbacks += 1
            text = request.fallback
        message = CoachMessage(
            text=text,
            level=request.level,
            trigger=request.trigger,
            game_id=request.game_id,
            ply=request.ply,
        )
        return Spoken(message=message, level=request.level, problems=problems, attempts=attempts)

    def _problems(self, text: str, request: _Request) -> list[str]:
        found = [f"withheld {item}" for item in leak_check(text, request.facts, request.level)]
        if self.draft_check is not None:
            found += self.draft_check(text, request.board)
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
_HINT = re.compile(r"\b(hint|clue|nudge|help me|help|stuck|where should i look)\b", re.I)
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


def fact_sheet(facts: PositionFacts, level: DisclosureLevel) -> str:
    visible = visible_facts(facts, level)
    lines = ["Fact sheet:"]
    for key, value in visible.items():
        if value in (None, "", [], False):
            continue
        if isinstance(value, list):
            value = (
                ", ".join(str(item) for item in value)
                if value and not isinstance(value[0], dict)
                else value
            )
        lines.append(f"- {key.replace('_', ' ')}: {value}")
    return "\n".join(lines)


class Coach:
    """The coach in a game: takes events and the user's words, hands back messages."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        policy: InterjectionPolicy | None = None,
        draft_check: Callable[[str, chess.Board], list[str]] | None = None,
        claims: Callable[[str, chess.Board, PositionFacts | None], list[str]] | None = None,
    ) -> None:
        self.policy = policy or InterjectionPolicy()
        self.speaker = Speaker(provider, draft_check=draft_check)
        self.claims = claims
        self.game_id: str | None = None
        self.user_side: Side = Side.WHITE
        self.board = chess.Board()
        self.positions: dict[int, _Position] = {}
        self.moves: dict[int, MovePlayed] = {}
        self.transcript: list[ChatMessage] = []
        self.spoken: list[Spoken] = []
        self._current: _Position | None = None

    # Lifecycle

    def start(self) -> None:
        self.speaker.start()

    def close(self) -> None:
        self.speaker.close()

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
        elif isinstance(event, GameEnded):
            self._current = None

    def tick(self, ts: float) -> None:
        """Time passing: a hint if the user has been stuck long enough."""
        trigger = self.policy.tick(ts)
        if trigger is None or self._current is None or self._current.facts is None:
            return
        level = self._current.disclosure.apply(Unlock.STUCK) or self._current.disclosure.level
        self.policy.said(self._current.ply)
        self._speak(trigger, self._current, level, ts)

    def ask(self, text: str, ts: float) -> None:
        """The user said something. Always answered, at the level their words earn."""
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
        verdicts: list[str] = []
        if self.claims is not None:
            try:
                verdicts = self.claims(text, position.board, position.facts)
            except Exception:  # noqa: BLE001
                verdicts = []
        self._speak(
            Trigger.USER_ASK,
            position,
            position.disclosure.level,
            ts,
            user_text=text,
            verdicts=verdicts,
        )

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
            self.transcript.append({"role": "assistant", "content": spoken.message.text})
            del self.transcript[:-MAX_TRANSCRIPT]
            out.append(spoken.message)
        return out

    # Internals

    def _new_game(self, event: GameStarted, ts: float) -> None:
        self.game_id = event.game_id
        self.user_side = event.user_side
        self.board = chess.Board(event.start_fen)
        self.positions = {0: _Position(ply=0, board=self.board.copy())}
        self.moves = {}
        self.transcript = []
        self._current = self.positions[0]
        self.policy.new_position(0, ts, user_to_move=self.user_to_move)

    def _on_move(self, event: MovePlayed, ts: float) -> None:
        self.board = chess.Board(event.fen_after)
        self.moves[event.ply] = event
        position = _Position(ply=event.ply, board=self.board.copy(), last_move=event.san)
        self.positions[event.ply] = position
        for old in [ply for ply in self.positions if ply < event.ply - 6]:
            del self.positions[old]
        self._current = position
        self.policy.new_position(event.ply, ts, user_to_move=self.user_to_move)

    def _on_analysis(self, event: Analysis, ts: float) -> None:
        position = self.positions.get(event.ply)
        if position is None or position.board.fen() != event.fen:
            return
        position.analysis = event
        position.facts = facts_from_analysis(position.board, event, last_move=position.last_move)
        if position is self._current and self.user_to_move:
            trigger = self.policy.on_position_analysed(event.ply, decisive=event.decisive)
            if trigger is not None:
                level = position.disclosure.apply(Unlock.DECISIVE) or position.disclosure.level
                self.policy.said(position.ply)
                self._speak(trigger, position, level, ts)

    def _on_assessment(self, event: MoveAssessment, ts: float) -> None:
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
        # A lecture is only for a move that actually cost something. A move below the
        # inaccuracy line gets a word: a nod when it lost almost nothing, a light touch
        # otherwise. Neither needs the model, and neither interrupts with a question.
        if trigger is Trigger.POST_MOVE and event.severity is Severity.NONE:
            assert before.facts is not None
            best = before.facts.candidates[0] if before.facts.candidates else None
            if event.expected_loss < TRIVIAL_LOSS or best is None or best == move.san:
                text = f"{move.san}: good."
            else:
                text = f"{move.san} is fine; {best} was a touch sharper."
            self._queue_plain(trigger, text, ts, ply=before.ply, level=DisclosureLevel.LINE)
            return
        self._speak(trigger, before, DisclosureLevel.LINE, ts, assessment=event, played=move)

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
    ) -> None:
        assert position.facts is not None
        instruction, fallback = self._instruction(
            trigger, position, level, user_text, verdicts, assessment, played
        )
        prompt = f"{fact_sheet(position.facts, level)}\n\n{instruction}"
        self.transcript.append({"role": "user", "content": prompt})
        del self.transcript[:-MAX_TRANSCRIPT]
        request = _Request(
            trigger=trigger,
            level=level,
            system=f"{PERSONA}\n\n{RULES}",
            messages=list(self.transcript),
            facts=position.facts,
            board=position.board,
            fallback=fallback,
            ply=position.ply,
            game_id=self.game_id or "",
        )
        self.speaker.submit(request)

    def _queue_plain(
        self,
        trigger: Trigger,
        text: str,
        ts: float,
        *,
        ply: int | None = None,
        level: DisclosureLevel = DisclosureLevel.SILENT,
    ) -> None:
        """A message that needs no model; it queues behind whatever the model is writing."""
        message = CoachMessage(
            text=text, level=level, trigger=trigger, game_id=self.game_id, ply=ply
        )
        self.speaker.submit_ready(Spoken(message=message, level=level, problems=[], attempts=0))

    def _instruction(
        self,
        trigger: Trigger,
        position: _Position,
        level: DisclosureLevel,
        user_text: str | None,
        verdicts: Sequence[str],
        assessment: MoveAssessment | None,
        played: MovePlayed | None,
    ) -> tuple[str, str]:
        """The turn's instruction to the model, and the template used if it fails."""
        facts = position.facts
        assert facts is not None
        theme = facts.theme or "the position"
        region = facts.region or "the whole board"
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
                fallback = f"Think about {theme}."
            else:
                fallback = "Take stock: what changed with the last move, and what does it allow?"
            return (
                "The player has been thinking for a while and is stuck. Give one hint, at the "
                "level of what the fact sheet allows, in one or two sentences, ending with a "
                "question. Do not name a move.",
                fallback,
            )
        if trigger in (Trigger.POST_MOVE, Trigger.POST_MISTAKE) and assessment and played:
            best = facts.candidates[0] if facts.candidates else "another move"
            loss = assessment.expected_loss
            label = assessment.label.value if assessment.label else assessment.severity.value
            words = _in_words(position.board, played.san)
            better_words = _in_words(position.board, best) if facts.candidates else best
            if trigger is Trigger.POST_MISTAKE:
                fallback = f"{played.san} cost you. {best} was better; the idea was {theme}."
                instruction = (
                    f"The player just played {played.san} ({words}), which was a {label}: it "
                    f"cost {loss:.0%} of the game. The better move was {best} ({better_words}). "
                    f"Explain the difference as a pattern to remember, in three sentences at "
                    f"most, then ask one question that checks they see it."
                )
            else:
                fallback = f"{played.san} was fine; {best} was a little tighter."
                instruction = (
                    f"The player just played {played.san} ({words}), which was fine but not "
                    f"the strongest: {best} ({better_words}) was better, by a little. One or "
                    f"two sentences. No lecture."
                )
            return instruction, fallback
        # An explicit ask.
        said = user_text or "hint"
        checks = ""
        if verdicts:
            checks = "\n\nTheir claims, checked against the board:\n- " + "\n- ".join(verdicts)
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
            fallback = f"Think about {theme}."
        else:
            fallback = "What is your opponent's most dangerous idea here?"
        return (
            f'The player says: "{said}".{checks}\n\nAnswer them directly at the level the fact '
            f"sheet allows. If they claimed something the board contradicts, correct it gently "
            f"and specifically. If they asked for what the sheet does not give you, say you "
            f"will not give that away yet and offer the next smaller nudge instead.",
            fallback,
        )
