"""The coach's voice follows the player's pace ("Follow my pace", `voice.pace = "follow"`).

What the player hears, tuned by `tools/voice_sim.py` over puzzles and games played slowly
and fast, the move found and missed, found at once and after a long think (Tai,
2026-10-03: "simulate what you would hear ... and tune such proposal such that the user
experience is good"):

- slow play: every word, whole;
- a fast line (moves under QUICK_S apart, or the game's fast pace): nothing during it; once
  the line is done (no move for LINE_END_S, or the next puzzle is up), one sentence on the
  move that took the thought -- the model's word on it, if it is LINE_WORD_FRESH_S young;
- a mistake: its lesson, even after one more move of the player's (a premove), its first
  sentence only once they have moved on, never after a retry of that move (the lesson names
  the answer), never older than LESSON_FRESH_S;
- the player's question: the answer, whole, whatever they play meanwhile; the game's review
  too;
- a hint: only while its position is on the board;
- a short word (a template) at a fast pace: said when the move took SHORT_OK_S or more;
- no sentence cut by a move, but a hint whose position is gone and a lesson whose move is
  taken back;
- nothing about a move older than its freshness: no word is said minutes later.

The plan keeps its own queue and hands the voice one line at a time (`next_line`), with the
next ready before the one playing ends (the voice makes it meanwhile), so what waits is
judged again when the player moves, before it is said. Pure logic on an injected clock: the
panel and the phone's PanelVoice call it, and `tests/test_voice_pace.py` holds it to the
simulator's scenarios.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

QUICK_S = 3.0  # a move this soon after its position appeared is part of a fast line
LINE_END_S = 2.5  # a fast line is over once the player has not moved for this long
# The word on the player's latest move is said this soon after it, or not at all: the model's
# reply comes 4-5 s after the move (measured on the panel, 2026-10-03), and while the player
# has not moved since, it is still the news.
PRAISE_FRESH_S = 9.0
LINE_WORD_FRESH_S = 9.0  # the line's word is said at its end if its move is this young
LESSON_FRESH_S = 15.0  # a lesson is never begun later than this after its move
LESSON_MOVES = 1  # a lesson is still said after this many more moves of the player's
REVIEW_FRESH_S = 30.0  # the game's summing up, said this soon after the game's end
SHORT_OK_S = 4.0  # a template word at a fast pace, on a move that took this long
FAST_PACE_S = 6.0  # the coach's fast pace: the median of the last think times under this
PACE_MOVES = 4

# Who is said first when several wait: the player's question, the game's review, a lesson,
# a word on a move, a hint.
PRIORITY = {"answer": 0, "review": 1, "lesson": 2, "praise": 3, "short": 3, "hint": 4}

SAY, HOLD, TEXT = "say", "hold", "text"


def kind_of(trigger: str, streamed: bool) -> str:
    """The plan's kind of a coach message: by its trigger, and for a word on a move whether
    the model wrote it (streamed, a lesson to hold for a line's end) or a template did."""
    if trigger == "user_ask":
        return "answer"
    if trigger == "review":
        return "review"
    if trigger == "post_mistake":
        return "lesson"
    if trigger in ("stuck", "decisive_question"):
        return "hint"
    return "praise" if streamed else "short"


@dataclass(frozen=True)
class Line:
    """One sentence of a coach message, as the plan hands it to the voice."""

    key: int  # the message's
    kind: str
    index: int  # its place in the message: 0 is the first sentence
    text: str
    prepared: bool = False  # the line's word, made while it was held: it plays at once


@dataclass
class _Message:
    key: int
    kind: str
    trigger: str
    about: tuple[int, int] | None  # (epoch, ply) of the player's move (a hint: its position)
    asked_at: float  # when its move was played (a hint, a question: when it was asked)
    model: bool
    # The player's moves counted when its move was played (a hint, a question: when asked):
    # the moves since are counted, never compared by time -- two moves can share a tick of
    # the clock (Windows' monotonic clock moves in steps of about 15 ms).
    seq: int


class PacePlan:
    """The voice's plan for one session: what is said, held, or left to the text.

    Thread-safe: the coach's thread offers a model's sentences as it writes them, the
    session's loop the rest; every method takes the plan's lock."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._epoch = 0  # a new game or puzzle, a re-read: another board altogether
        self._ply = 0  # the board's last ply in this epoch
        self._count = 0  # the player's moves so far, in any game
        # (epoch, ply) -> when the player's move there was played, and its number (`_count`)
        self._move_at: dict[tuple[int, int], tuple[float, int]] = {}
        self._withdrawn: set[tuple[int, int]] = set()
        self._position_since = clock()
        self._thinks: list[float] = []
        self._quick = False
        self._last_think = 0.0
        self._last_move_at = -1e9
        self._messages: dict[int, _Message] = {}
        self._queue: list[Line] = []
        self._held: list[Line] = []  # the fast line's word, until the line is done
        self._released: set[int] = set()  # a line's word, said at the line's end
        self._speaking: set[int] = set()  # words on a move begun in slow play
        self._dead: set[int] = set()
        self._fed: list[Line] = []  # handed to the voice, in order: playing, then waiting

    # ---- the board ----------------------------------------------------------------------

    def player_moved(self, ply: int, think_s: float | None) -> bool:
        """The player played the move at `ply` after `think_s` of thought (None: not read).
        Returns whether the voice must stop now: a hint whose position is gone."""
        with self._lock:
            now = self._clock()
            think = think_s if think_s is not None else now - self._position_since
            recent = [*self._thinks, think][-PACE_MOVES:]
            fast_pace = len(recent) >= 2 and statistics.median(recent) < FAST_PACE_S
            # A move after real thought is slow whatever the pace was: the pace lingers four
            # moves after a burst of recaptures, and a find after 22 s went to text.
            self._quick = think < QUICK_S or (fast_pace and think < FAST_PACE_S)
            self._last_think = think
            self._thinks.append(think)
            self._last_move_at = now
            self._count += 1
            self._move_at[(self._epoch, ply)] = (now, self._count)
            self._ply = ply
            self._position_since = now
            return self._after_change(now, stop_kinds=("hint",))

    def opponent_moved(self, ply: int) -> None:
        with self._lock:
            self._ply = ply
            self._position_since = self._clock()

    def taken_back(self, to_ply: int) -> bool:
        """A takeback to `to_ply`: the player's moves after it are withdrawn, and a lesson on
        one of them is never said again (it names the answer the retry is looking for).
        Returns whether the voice must stop now."""
        with self._lock:
            now = self._clock()
            for epoch, ply in list(self._move_at):
                if epoch == self._epoch and ply > to_ply:
                    self._withdrawn.add((epoch, ply))
            self._ply = to_ply
            self._position_since = now
            return self._after_change(now, stop_kinds=("hint",), stop_withdrawn=True)

    def new_position(self) -> bool:
        """A new game or puzzle, or the board re-read: another board altogether. The fast
        line's word is said now, if still young. Returns whether the voice must stop now."""
        with self._lock:
            now = self._clock()
            self._epoch += 1
            self._ply = 0
            self._thinks = []
            self._position_since = now
            self._release(now)
            return self._after_change(now, stop_kinds=("hint",))

    def _after_change(
        self, now: float, stop_kinds: tuple[str, ...], stop_withdrawn: bool = False
    ) -> bool:
        """The board changed: what waits is judged again; whether what plays must stop."""
        self._queue = [line for line in self._queue if self._keep(line, now)]
        playing = self._fed[0] if self._fed else None
        if playing is None:
            return False
        msg = self._messages.get(playing.key)
        if msg is None:
            return False
        if msg.kind in stop_kinds:
            return msg.kind != "hint" or not self._hint_fresh(msg)
        return bool(stop_withdrawn and msg.about in self._withdrawn)

    # ---- the coach's messages -----------------------------------------------------------

    def offer(
        self,
        key: int,
        trigger: str,
        ply: int | None,
        index: int,
        text: str,
        *,
        streamed: bool,
    ) -> str:
        """A sentence of a coach message is ready: SAY (queued for the voice), HOLD (the
        fast line's word, said at its end) or TEXT (shown, not said). `ply` is the message's
        position: for a word on a move, the position the move was played from."""
        with self._lock:
            now = self._clock()
            msg = self._messages.get(key)
            if msg is None:
                msg = self._message(key, trigger, ply, streamed, now)
            if key in self._dead:
                return TEXT
            if msg.kind in ("praise", "short"):
                return self._offer_word(msg, index, text, now)
            if not self._sentence_fresh(msg, index, now):
                if index == 0:
                    self._dead.add(key)
                return TEXT
            self._enqueue(Line(key, msg.kind, index, text))
            return SAY

    def _message(
        self, key: int, trigger: str, ply: int | None, streamed: bool, now: float
    ) -> _Message:
        kind = kind_of(trigger, streamed)
        about: tuple[int, int] | None = None
        asked_at, seq = now, self._count
        if kind in ("praise", "short", "lesson") and ply is not None:
            # A word on a move carries the position it was played from; the move is the next
            # ply, played when the plan heard of it.
            about = (self._epoch, ply + 1)
            asked_at, seq = self._move_at.get(about, (now, self._count))
        elif kind == "hint" and ply is not None:
            about = (self._epoch, ply)
        msg = _Message(key, kind, trigger, about, asked_at, streamed, seq)
        self._messages[key] = msg
        if len(self._messages) > 400:  # a long session: forget the oldest
            for old in sorted(self._messages)[:200]:
                self._messages.pop(old, None)
        return msg

    def _offer_word(self, msg: _Message, index: int, text: str, now: float) -> str:
        key = msg.key
        if key in self._released or any(line.key == key for line in self._held):
            # The line's word is its first sentence: the rest is text.
            return TEXT
        if key in self._speaking:
            if self._sentence_fresh(msg, index, now):
                self._enqueue(Line(key, msg.kind, index, text))
                return SAY
            return TEXT
        latest = (
            self._moves_since(msg) == 0 and msg.about is not None and (msg.about[0] == self._epoch)
        )
        short_ok = msg.kind == "short" and self._last_think >= SHORT_OK_S
        if latest and (not self._quick or short_ok) and self._fresh(msg, now):
            # Slow play, the word on the move just made: said, after what plays.
            self._speaking.add(key)
            self._enqueue(Line(key, msg.kind, index, text))
            return SAY
        if msg.model and index == 0 and not self._held:
            # A fast line, or the player already moved on: the model's word on a find is
            # held for the end of the line (the line's first such).
            self._held.append(Line(key, msg.kind, 0, text, prepared=True))
            return HOLD
        self._dead.add(key)
        return TEXT

    def _enqueue(self, line: Line) -> None:
        self._queue.append(line)
        self._queue.sort(key=lambda line: (PRIORITY[line.kind], line.key, line.index))

    # ---- freshness ------------------------------------------------------------------------

    def _moves_since(self, msg: _Message) -> int:
        """The player's moves after the one the message is about (any board)."""
        return self._count - msg.seq

    def _hint_fresh(self, msg: _Message) -> bool:
        return msg.about == (self._epoch, self._ply)

    def _fresh(self, msg: _Message, now: float) -> bool:
        if msg.key in self._dead:
            return False
        age = now - msg.asked_at
        if msg.key in self._released:
            return age <= LINE_WORD_FRESH_S + 6.0  # begun at the line's end: it is said
        if msg.kind == "answer":
            return True
        if msg.kind == "review":
            return age <= REVIEW_FRESH_S
        if msg.kind == "hint":
            return self._hint_fresh(msg)
        if msg.kind == "lesson":
            if msg.about in self._withdrawn:
                return False
            return self._moves_since(msg) <= LESSON_MOVES and age <= LESSON_FRESH_S
        # A word on a move: the move is still the player's latest, and it is just made.
        if msg.about is None or msg.about[0] != self._epoch:
            return False
        if msg.about in self._withdrawn:
            return False
        if msg.key in self._speaking:
            return self._moves_since(msg) == 0
        return self._moves_since(msg) == 0 and age <= PRAISE_FRESH_S

    def _sentence_fresh(self, msg: _Message, index: int, now: float) -> bool:
        """A message's sentence is said while the message is fresh; once the player has moved
        on, only its first sentence -- the key fact -- is said, the rest is text."""
        if not self._fresh(msg, now):
            return False
        moved_on = self._moves_since(msg) > 0 or (
            msg.about is not None and msg.about[0] != self._epoch
        )
        return not (index > 0 and msg.kind not in ("answer", "review") and moved_on)

    def _keep(self, line: Line, now: float) -> bool:
        msg = self._messages.get(line.key)
        return msg is not None and self._sentence_fresh(msg, line.index, now)

    # ---- the voice ------------------------------------------------------------------------

    def tick(self) -> None:
        """The fast line's word is released once the line is done (no move for LINE_END_S)."""
        with self._lock:
            now = self._clock()
            if self._held and now - self._last_move_at >= LINE_END_S:
                self._release(now)

    def _release(self, now: float) -> None:
        if not self._held:
            return
        line = self._held[0]
        self._held = []
        msg = self._messages.get(line.key)
        if msg is not None and now - msg.asked_at <= LINE_WORD_FRESH_S:
            self._released.add(line.key)
            self._enqueue(line)
        else:
            self._dead.add(line.key)

    def held_text(self) -> str | None:
        """The fast line's word being held, for the voice to make ahead (`prepare`)."""
        with self._lock:
            return self._held[0].text if self._held else None

    def next_line(self) -> Line | None:
        """The next sentence for the voice, still fresh now; None when nothing waits."""
        with self._lock:
            now = self._clock()
            while self._queue:
                line = self._queue.pop(0)
                if self._keep(line, now):
                    self._fed.append(line)
                    return line
            return None

    def voice_state(self, busy: bool, waiting: int | None) -> None:
        """What the voice holds now: `busy` (a line plays or waits) and how many lines wait
        behind the one playing (None: the voice cannot say). The plan's record of the lines
        handed over keeps only those still with the voice."""
        with self._lock:
            if not busy:
                self._fed = []
            elif waiting is not None:
                keep = waiting + 1
                self._fed = self._fed[-keep:] if keep < len(self._fed) else self._fed

    def waiting_stale(self) -> bool:
        """Whether a line handed to the voice and waiting behind the one playing is no longer
        the news (the voice should drop what waits)."""
        with self._lock:
            now = self._clock()
            if len(self._fed) < 2:
                return False
            stale = [line for line in self._fed[1:] if not self._keep(line, now)]
            if stale:
                self._fed = self._fed[:1]
                return True
            return False

    def stopped(self) -> None:
        """The voice was stopped: nothing handed over plays any more."""
        with self._lock:
            self._fed = []

    def clear(self) -> None:
        """Muted, or the session over: nothing waits, nothing is held."""
        with self._lock:
            self._queue = []
            self._held = []
            self._fed = []
