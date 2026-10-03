"""What the player hears: the coach's voice policy simulated over scripted sessions.

Each scenario is a session as the player lives it -- puzzles and games, played slowly or
fast, the move found or missed, found at once or after a long think, a hint, a question, a
retry -- with the coach's messages timed as the phone times them (a template at once, a
model reply's first sentence 1-2.3 s after the move and the next ones as it writes them,
the voice's first sound ~0.6x a sentence's length after it is queued, speech at ~15
characters a second). A policy decides, as the panel does, what the voice does with each
message: speak, queue, cut what is playing, hold, or leave it to the text.

For each scenario and policy it prints the timeline of what is heard -- each sentence
whole, cut (with the words heard), or silent and why -- and scores it:

  lesson     a lesson on a mistake or miss heard whole (the point of the coach)
  praise     the word on a found move heard whole
  hint       a hint heard while its position is still on the board
  answer     the answer to the player's question heard whole
  stale      a sentence begun when the board is 2+ of the player's moves past what it is
             about, or 10+ s after that move (talking about the past) -- bad
  cut        sentences cut mid-way -- bad
  overtalk   seconds of speech while the player is moving fast (the next move came < 3 s
             after the previous) -- bad, the voice talks over play

    uv run --no-sync python tools/voice_sim.py [--policy current proposal] [--scenario P2]
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass, field

# ---- timings (phone, measured 2026-10-02 from the Pixel dump) ------------------------------

CHARS_PER_S = 15.0  # Kokoro bm_george at speed 1.0
SYNTH_RATIO = 0.6  # a piece is made in about 0.6x its speech; the first sound waits for it
SYNTH_MIN_S, SYNTH_MAX_S = 0.4, 1.8
TEMPLATE_S = 0.15  # a template, written from the board, is ready at once
MODEL_FIRST_S = 1.6  # a model reply's first sentence after it is asked (0.9-2.3 s seen)
MODEL_NEXT_S = 0.35  # each next sentence after the one before
LESSON_WAIT_S = 0.3  # a mistake's lesson waits for the analysis (in puzzles it is quick)
STUCK_S = 45.0
FAST_PACE_S = 6.0  # the coach's fast pace: the median of the last few think times
PACE_RESUME_S = 6.0
PACE_MOVES = 4


def speech_s(text: str) -> float:
    return max(0.5, len(text) / CHARS_PER_S)


def synth_s(text: str) -> float:
    return min(SYNTH_MAX_S, max(SYNTH_MIN_S, SYNTH_RATIO * speech_s(text)))


# ---- what the coach says (lengths from the dump's real lines) ------------------------------

LESSON = [
    "This was a miss: Rg8 was the quiet mating attack, taking the g-file and the king's squares.",
    "By moving the rook to the d-file, you let their king breathe.",
    "Next time, look for the quiet move that takes squares from the king.",
]
PRAISE_MODEL = [
    "Excellent: that was the only good move, keeping the mating attack alive.",
    "It pulls the defender away so the rook can finish.",
]
PRAISE_SHORT = ["Exactly."]  # a template: best, excellent, book
LESSON_SHORT = ["This was a big moment: Rg8 was much stronger."]  # the fast pace's template
HINT1 = [
    "Only one move keeps the winning advantage; the next best gives most of it back.",
    "Their king on g8 has only two squares, and you have two checks.",
    "Which check leaves it no square at all?",
]
HINT2 = [
    "Use your rook against the king: the idea is a back-rank mate.",
    "Which square does it go to?",
]
ANSWER = ["Play Qa8+.", "It deflects the rook from the back rank, and mate follows."]


@dataclass
class Msg:
    """A coach message: what it is about, and when each sentence is ready."""

    id: int
    kind: str  # lesson | praise | short | hint | answer
    about_move: int  # the player's move count it is about (hint: the position after that many)
    asked_at: float
    sentences: list[str]
    model: bool
    ready: list[float] = field(default_factory=list)
    moment: int = 0  # the player's move count when asked (the coach's `moved_on`)


@dataclass
class Act:
    """One thing the player does at `t`: a move (found or not), a takeback, a question, a
    new puzzle or game."""

    t: float
    what: str  # move | takeback | ask | new
    good: bool = True
    exciting: bool = False  # a great/only move the model praises (else a template)


# ---- scenarios -----------------------------------------------------------------------------


def acts(spec: list[tuple]) -> list[Act]:
    """(gap_s, what, good, exciting) steps into timed actions; the opponent's reply is
    instant in a puzzle and part of the gap in a game."""
    t = 0.0
    out = []
    for gap, what, *rest in spec:
        t += gap
        good = rest[0] if rest else True
        exciting = rest[1] if len(rest) > 1 else False
        out.append(Act(t, what, good, exciting))
    return out


SCENARIOS: dict[str, tuple[str, list[Act]]] = {
    "P1": (
        "puzzle, slow: thinks 40 s, finds it, then two forced moves 8 s apart",
        acts(
            [
                (40, "move", True, True),
                (8, "move", True, False),
                (8, "move", True, True),
                (3, "new"),
            ]
        ),
    ),
    "P2": (
        "puzzle, fast: saw the line, plays it out at 1.5 s a move",
        acts(
            [
                (30, "move", True, True),
                (1.5, "move", True, True),
                (1.5, "move", True, False),
                (1.5, "move", True, True),
                (2.5, "new"),
            ]
        ),
    ),
    "P3": (
        "puzzle: a long think, wrong move, retries and finds it, then plays the line fast",
        acts(
            [
                (40, "move", False),
                (5, "takeback"),
                (6, "move", True, True),
                (1.6, "move", True, False),
                (1.6, "move", True, True),
                (2.5, "new"),
            ]
        ),
    ),
    "P4": (
        "puzzle: stuck, hears the hint at 45 s, moves 2 s later, line found fast",
        acts(
            [
                (47, "move", True, True),
                (1.8, "move", True, False),
                (1.8, "move", True, True),
                (2.5, "new"),
            ]
        ),
    ),
    "P5": (
        "puzzle rush: wrong move after 5 s, next puzzle at once, again",
        acts(
            [
                (5, "move", False),
                (2, "new"),
                (6, "move", False),
                (2, "new"),
                (4, "move", True),
                (2, "new"),
            ]
        ),
    ),
    "P6": (
        "puzzle: stuck, two hints (45 s, 90 s), finds it slowly, then one more move",
        acts([(100, "move", True, True), (10, "move", True, False), (3, "new")]),
    ),
    "P7": (
        "puzzle: asks 'what should I play' at 30 s, plays it 5 s later, line fast",
        acts(
            [
                (30, "ask"),
                (5, "move", True, False),
                (1.6, "move", True, True),
                (2.5, "new"),
            ]
        ),
    ),
    "P8": (
        "puzzle: wrong move played fast after a quick look, retry, slow find",
        acts(
            [
                (8, "move", False),
                (2, "takeback"),
                (25, "move", True, True),
                (6, "move", True),
                (3, "new"),
            ]
        ),
    ),
    "G1": (
        "rapid game vs a person: 15-25 s a move, one mistake mid-game",
        acts(
            [
                (18, "move"),
                (22, "move"),
                (15, "move", True, True),
                (20, "move", False),
                (25, "move"),
                (16, "move"),
                (19, "move", True, True),
            ]
        ),
    ),
    "G2": (
        "blitz: 3 s a move, a mistake among them",
        acts(
            [
                (3, "move"),
                (2.5, "move"),
                (3, "move", True, True),
                (3.2, "move", False),
                (2.8, "move"),
                (3, "move"),
                (2.6, "move"),
            ]
        ),
    ),
    "G3": (
        "rapid, then a quick exchange of recaptures (1.2 s each), then slow again",
        acts(
            [
                (20, "move"),
                (18, "move", True, True),
                (1.2, "move"),
                (1.2, "move"),
                (1.2, "move", True, True),
                (22, "move"),
                (20, "move", False),
                (24, "move"),
            ]
        ),
    ),
    "G4": (
        "rapid: a blunder, then the next move premoved (0.8 s), then slow",
        acts(
            [
                (25, "move", False),
                (0.8, "move"),
                (20, "move"),
                (18, "move"),
            ]
        ),
    ),
    "G5": (
        "moderate pace, 4-6 s a move: two finds and a mistake",
        acts(
            [
                (5, "move"),
                (4.5, "move", True, True),
                (5.5, "move"),
                (4, "move", False),
                (6, "move"),
                (5, "move", True, True),
                (4.5, "move"),
            ]
        ),
    ),
    "G6": (
        "slow play, then a mating attack played out fast (1.5 s a move) to the end",
        acts(
            [
                (20, "move"),
                (25, "move", True, True),
                (1.5, "move", True, True),
                (1.5, "move", True, False),
                (1.5, "move", True, True),
                (4, "new"),
            ]
        ),
    ),
    "P9": (
        "puzzle: finds the first move fast (4 s), then thinks 20 s on the second",
        acts(
            [
                (4, "move", True, True),
                (20, "move", True, True),
                (1.5, "move"),
                (2.5, "new"),
            ]
        ),
    ),
}


# ---- the coach: which message a player action brings, and when it is ready ----------------


def messages_for(acts_: list[Act], fast_templates: bool) -> list[Msg]:
    """The coach's messages, as the phone's coach makes them: a word on every move (a
    template for a routine one, the model for an exciting one, a lesson for a mistake), a
    stuck hint at 45 s (a second at 90 s), the answer to a question. A fast pace (median of
    the last think times under 6 s) gets templates only (`fast_templates` mirrors the coach)."""
    out: list[Msg] = []
    moves = 0
    since = 0.0  # when the current position appeared
    thinks: list[float] = []
    hints_given = 0
    nid = 0

    def add(kind: str, about: int, at: float, sentences: list[str], model: bool) -> None:
        nonlocal nid
        nid += 1
        first = at + (MODEL_FIRST_S if model else TEMPLATE_S)
        if kind == "lesson":
            first += LESSON_WAIT_S
        ready = [first + i * MODEL_NEXT_S for i in range(len(sentences))]
        out.append(Msg(nid, kind, about, at, sentences, model, ready, moment=moves))

    for act in acts_:
        # Stuck hints while the player thinks in this position.
        while hints_given < 2 and since + STUCK_S * (hints_given + 1) < act.t:
            at = since + STUCK_S * (hints_given + 1)
            add("hint", moves, at, HINT1 if hints_given == 0 else HINT2, True)
            hints_given += 1
        if act.what == "ask":
            add("answer", moves, act.t, ANSWER, True)
            continue
        if act.what == "new":
            moves += 100  # another position altogether: everything before is gone
            since, hints_given, thinks = act.t, 0, []
            continue
        if act.what == "takeback":
            moves += 1  # the position the lesson was about is withdrawn
            since, hints_given = act.t, 0
            continue
        thinks.append(act.t - since)
        fast = (
            fast_templates
            and len(thinks) >= 2
            and statistics.median(thinks[-PACE_MOVES:]) < FAST_PACE_S
        )
        about = moves + 1
        moves += 1
        since, hints_given = act.t, 0
        if not act.good:
            add("lesson", about, act.t, LESSON_SHORT if fast else LESSON, not fast)
        elif act.exciting and not fast:
            add("praise", about, act.t, PRAISE_MODEL, True)
        else:
            add("short", about, act.t, PRAISE_SHORT, False)
    return out


# ---- the voice ------------------------------------------------------------------------------


@dataclass
class Heard:
    msg: Msg
    sentence: str
    start: float | None  # when its sound began (None: never)
    end: float | None
    words_heard: int
    why: str  # whole | cut: ... | silent: ...


class Voice:
    """A queue of sentences, each made then played; `stop` silences what plays and drops
    what waits."""

    # Whether the next sentence is made while the one before plays (True), or only once it
    # has ended (the phone's KokoroTts today: one line at a time, made then played).
    PRESYNTH = False

    def __init__(self) -> None:
        self.queue: list[tuple[Msg, str]] = []
        self.playing: tuple[Msg, str, float, float] | None = None  # msg, sentence, start, end
        self.free_at = 0.0  # when the synth is free for the next piece (pipelined)
        self.log: list[Heard] = []
        self.ended_at = -1e9  # when the last sentence ended

    def busy(self, now: float) -> bool:
        return self.playing is not None or bool(self.queue)

    def say(self, msg: Msg, sentence: str) -> None:
        self.queue.append((msg, sentence))

    def stop(self, now: float, why: str) -> None:
        if self.playing is not None:
            msg, s, start, end = self.playing
            if now < start:
                self.log.append(Heard(msg, s, None, None, 0, f"silent: {why} before a sound"))
            else:
                frac = (now - start) / (end - start)
                words = int(len(s.split()) * frac)
                self.log.append(Heard(msg, s, start, now, words, f"cut: {why}"))
            self.playing = None
        for msg, s in self.queue:
            self.log.append(Heard(msg, s, None, None, 0, f"silent: {why}, dropped from the queue"))
        self.queue.clear()

    def tick(self, now: float) -> None:
        if self.playing is not None and now >= self.playing[3]:
            msg, s, start, end = self.playing
            self.log.append(Heard(msg, s, start, end, len(s.split()), "whole"))
            self.playing = None
            self.ended_at = end
        if self.playing is None and self.queue:
            msg, s = self.queue.pop(0)
            if isinstance(s, Held) or (self.PRESYNTH and self.queue_was_ready(now)):
                # Made while it was held (the line's word), or while the one before played:
                # it follows at once.
                start = max(now, self.ended_at + 0.15)
            else:
                start = now + synth_s(s)
            self.playing = (msg, s, start, start + speech_s(s))

    def queue_was_ready(self, now: float) -> bool:
        """Whether the next sentence was made already: it was queued while the one before
        played (the voice is just now free), not to an idle voice."""
        return now - self.ended_at <= 0.1


# ---- policies -------------------------------------------------------------------------------


class Policy:
    """Decides what the voice does: `on_move` when the player moves (or takes back, or a new
    position comes), `on_sentence` when a sentence of a message is ready, `on_tick` each
    step (a held word said later)."""

    name = "base"
    fast_templates = True

    def __init__(self, voice: Voice) -> None:
        self.v = voice
        self.moves = 0
        self.position_since = 0.0
        self.thinks: list[float] = []
        self.silent: dict[int, str] = {}  # a message decided silent, and why
        self.started: set[int] = set()

    def note(self, msg: Msg, sentence: str, why: str) -> None:
        self.v.log.append(Heard(msg, sentence, None, None, 0, f"silent: {why}"))

    def fast(self, now: float) -> bool:
        pace = statistics.median(self.thinks[-PACE_MOVES:]) if len(self.thinks) >= 2 else None
        if pace is None or pace >= FAST_PACE_S:
            return False
        return now - self.position_since < FAST_PACE_S

    def on_act(self, act: Act, now: float) -> None:
        if act.what == "new":
            self.moves += 100
            self.thinks = []
        elif act.what == "takeback":
            self.moves += 1
        elif act.what == "move":
            self.thinks.append(now - self.position_since)
            self.moves += 1
        if act.what != "ask":
            self.position_since = now

    def on_sentence(self, msg: Msg, i: int, now: float) -> None: ...
    def on_tick(self, now: float) -> None: ...


class Current(Policy):
    """The phone and the desktop as of 0.1.63 (panel.voice_action, _stream_sink, _speak,
    _say_held_word, _position_moved_on)."""

    name = "current"
    STALE_AFTER = {"short": 2, "praise": 2, "lesson": 2, "hint": 0}

    def __init__(self, voice: Voice) -> None:
        super().__init__(voice)
        self.held: tuple[Msg, list[str], float] | None = None

    def on_act(self, act: Act, now: float) -> None:
        super().on_act(act, now)
        if act.what in ("move", "takeback", "new"):
            self.v.stop(now, "the player moved" if act.what == "move" else f"a {act.what}")
            self.cut_streams = True
            for mid in list(self.started):
                self.silent.setdefault(mid, "cut off mid-reply")
        if act.what in ("takeback", "new"):
            self.held = None

    def plies_behind(self, msg: Msg) -> int:
        # The coach counts plies: the player's move and the opponent's reply.
        return (
            2 * (self.moves - msg.about_move) + 1
            if msg.kind != "hint"
            else 2 * (self.moves - msg.about_move)
        )

    def on_sentence(self, msg: Msg, i: int, now: float) -> None:
        s = msg.sentences[i]
        if msg.id in self.silent:
            self.note(msg, s, self.silent[msg.id])
            return
        if i > 0:
            if self.held is not None and self.held[0] is msg:
                self.held = (msg, self.held[1] + [s], self.held[2])
                self.note(msg, s, "held for the pace (text)")
                return
            self.v.say(msg, s)
            return
        if msg.kind != "answer" and self.moves != msg.moment:
            # The coach's moved_on: the player moved again since it was asked.
            self.silent[msg.id] = "late: asked for a moment that is over"
            self.note(msg, s, self.silent[msg.id])
            return
        if msg.kind == "answer":
            if self.v.busy(now):
                self.v.stop(now, "superseded by the answer")
            self.started.add(msg.id)
            self.v.say(msg, s)
            return
        if self.fast(now):
            self.held = (msg, [s], now)
            self.silent[msg.id] = "held for the pace (text)"
            self.note(msg, s, "held for the pace (text)")
            return
        allowed = self.STALE_AFTER[msg.kind]
        if self.plies_behind(msg) > allowed:
            self.silent[msg.id] = "stale"
            self.note(msg, s, "stale")
            return
        if msg.kind == "short" and len(s) < 40 and self.v.busy(now):
            self.silent[msg.id] = "skipped: the voice was busy"
            self.note(msg, s, self.silent[msg.id])
            return
        self.started.add(msg.id)
        self.v.say(msg, s)

    def on_tick(self, now: float) -> None:
        if self.held is None or self.v.busy(now):
            return
        msg, sentences, _ = self.held
        if self.plies_behind(msg) > self.STALE_AFTER[msg.kind]:
            self.held = None
            return
        if now - self.position_since >= PACE_RESUME_S:
            self.held = None
            for s in sentences:
                self.v.say(msg, Said(s))


class Said(str):
    """A sentence said late, after the pace slowed (for the timeline)."""


class Held(str):
    """A fast line's word, held to the line's end and made while it waited (for the
    timeline, and the voice: it plays at once)."""


class Proposal(Policy):
    """One spoken queue, by priority and freshness (2026-10-02 draft, tuned below).

    - The player's move never cuts a sentence mid-way: the sentence playing finishes; what
      waits behind it is re-judged (`fresh`).
    - Each message is fresh while it is still the news: an answer always; a lesson until the
      player's next mistake, a new position, or LESSON_FRESH_S after its move; praise only
      for the player's latest move and PRAISE_FRESH_S after it; a hint while its position is
      on the board.
    - Newer of the same kind replaces older (only the latest praise / lesson waits).
    - Priority when choosing what to say next: answer > lesson > praise > hint.
    - A reply that begins while the voice is free is said; while busy, it waits its turn if
      it is still fresh when its turn comes, else it is shown only.
    - In a fast stretch the model is not asked (templates, as now) but the short template
      IS said when the voice is free (one word, said before the next move lands).
    """

    name = "proposal"
    LESSON_FRESH_S = 12.0
    LESSON_MOVES = 1  # a lesson is still said after this many more of the player's moves
    PRAISE_FRESH_S = 2.5
    PRIORITY = {"answer": 0, "lesson": 1, "praise": 2, "short": 2, "hint": 3}
    FINISH_SENTENCE = True

    def __init__(self, voice: Voice) -> None:
        super().__init__(voice)
        self.pending: list[tuple[Msg, str, float]] = []  # sentences waiting for the voice
        self.dead: set[int] = set()
        self.last_move_at = 0.0

    def fresh(self, msg: Msg, now: float) -> bool:
        if msg.id in self.dead:
            return False
        age = now - msg.asked_at
        behind = self.moves - msg.about_move
        if msg.kind == "answer":
            return True
        if msg.kind == "hint":
            return self.moves == msg.about_move
        if msg.kind == "lesson":
            return behind <= self.LESSON_MOVES and age <= self.LESSON_FRESH_S and behind < 50
        return behind == 0 and age <= self.PRAISE_FRESH_S

    def on_act(self, act: Act, now: float) -> None:
        super().on_act(act, now)
        if act.what in ("move", "takeback", "new"):
            self.last_move_at = now
            # What waits is re-judged; what plays finishes its sentence (or stops, if the
            # position it speaks of is gone: a hint once the player moved, anything at a
            # new position).
            playing = self.v.playing
            if playing is not None:
                msg = playing[0]
                gone = act.what == "new" or (msg.kind == "hint")
                if gone or not self.FINISH_SENTENCE:
                    self.v.stop(now, f"the {act.what}")
            # The queue drops what is no longer fresh.
            keep = []
            for msg, s in self.v.queue:
                if self.fresh(msg, now):
                    keep.append((msg, s))
                else:
                    self.dead.add(msg.id)
                    self.note(msg, s, "no longer the news")
            self.v.queue = keep

    def on_sentence(self, msg: Msg, i: int, now: float) -> None:
        s = msg.sentences[i]
        if not self.fresh(msg, now):
            self.dead.add(msg.id)
            self.note(msg, s, "no longer the news")
            return
        if i == 0:
            # Newer of the same family replaces what waits of the older.
            fam = "praise" if msg.kind in ("praise", "short") else msg.kind
            keep = []
            for m, q in self.v.queue:
                mf = "praise" if m.kind in ("praise", "short") else m.kind
                if mf == fam and m.id != msg.id:
                    self.dead.add(m.id)
                    self.note(m, q, "replaced by a newer one")
                else:
                    keep.append((m, q))
            self.v.queue = keep
            # An answer is said next, ahead of everything waiting.
        self.v.queue.append((msg, s))
        self.v.queue.sort(key=lambda p: (self.PRIORITY[p[0].kind], p[0].id))

    def on_tick(self, now: float) -> None:
        # Before the next sentence starts: still the news?
        if self.v.playing is None and self.v.queue:
            keep = []
            for m, q in self.v.queue:
                if self.fresh(m, now):
                    keep.append((m, q))
                else:
                    self.dead.add(m.id)
                    self.note(m, q, "no longer the news")
            self.v.queue = keep


class Burst(Policy):
    """The voice follows the player's pace, move by move (2026-10-03, after the first runs).

    What the player should hear:
      - slow play: every word, whole;
      - a fast line (moves under QUICK_S apart): nothing during it, then -- once the line
        is done (no move for BURST_END_S, or the next puzzle is up) -- one whole word on the
        move that took the thought, if the model praised it and it is SUMMARY_FRESH_S young;
      - a mistake: its lesson, even after one more move of the player's (a premove), never
        after a retry of that move (the lesson names the answer), never older than
        LESSON_FRESH_S;
      - the player's question: the answer, whole, whatever they play meanwhile;
      - a hint: only while its position is on the board;
      - no sentence cut by a move, except a hint whose position is gone, and a lesson whose
        move is taken back;
      - nothing about a move older than its freshness: words never said minutes later.
    """

    name = "burst"
    QUICK_S = 3.0
    SHORT_OK_S = 4.0
    last_think = 0.0
    BURST_END_S = 2.5
    PRAISE_FRESH_S = 3.0
    SUMMARY_FRESH_S = 9.0
    LESSON_FRESH_S = 15.0
    LESSON_MOVES = 1
    PRIORITY = {"answer": 0, "lesson": 1, "praise": 2, "short": 2, "hint": 3}

    def __init__(self, voice: Voice) -> None:
        super().__init__(voice)
        self.dead: set[int] = set()
        self.withdrawn: set[int] = set()  # the player's moves taken back
        self.last_move_at = -1e9
        self.quick = False  # the latest move came within QUICK_S of its position
        self.summary: list[tuple[Msg, str]] = []  # the burst's best word, held to its end
        self.summarised: set[int] = set()  # words released at a line's end
        self.speaking: set[int] = set()  # praise said in slow play (its later sentences follow)
        self.positions = 0  # bumps on a new puzzle or game (another position altogether)

    # -- freshness ------------------------------------------------------------------------

    def fresh(self, msg: Msg, now: float) -> bool:
        if msg.id in self.dead:
            return False
        if msg.id in self.summarised:
            return now - msg.asked_at <= self.SUMMARY_FRESH_S + 6.0  # its sentences, once begun
        if msg.id in self.speaking and msg.kind in ("praise", "short"):
            # Begun in slow play: its next sentence follows unless the player has moved on.
            return self.moves == msg.about_move
        age = now - msg.asked_at
        behind = self.moves - msg.about_move
        if msg.kind == "answer":
            return True
        if msg.kind == "hint":
            return self.moves == msg.about_move
        if msg.kind == "lesson":
            if msg.about_move in self.withdrawn:
                return False
            # Across a new puzzle the moves jump by 100: the lesson on the last puzzle's
            # wrong move is still worth its first seconds (puzzle rush).
            behind = behind % 100
            return behind <= self.LESSON_MOVES and age <= self.LESSON_FRESH_S
        return behind == 0 and age <= self.PRAISE_FRESH_S

    def moved_on_from(self, msg: Msg) -> bool:
        """The player has moved since the move this is about (or the next puzzle is up)."""
        return self.moves != msg.about_move

    def fresh_sentence(self, msg: Msg, s: str, now: float) -> bool:
        """A message's sentence is said while the message is fresh; once the player has moved
        on, only its first sentence -- the key fact -- is said, the rest is text."""
        if not self.fresh(msg, now):
            return False
        later = s != msg.sentences[0]
        return not (later and msg.kind != "answer" and self.moved_on_from(msg))

    def drop_stale_queue(self, now: float, why: str) -> None:
        keep = []
        for m, s in self.v.queue:
            if self.fresh_sentence(m, s, now):
                keep.append((m, s))
            else:
                self.note(m, s, why)
        self.v.queue = keep

    # -- the player's actions -------------------------------------------------------------

    def on_act(self, act: Act, now: float) -> None:
        if act.what == "ask":
            return
        # A quick move, or the game's fast pace (the coach's own: the median of the last
        # think times under FAST_PACE_S): praise is text, only lessons are said.
        think = now - self.position_since
        recent = (self.thinks + [think])[-PACE_MOVES:]
        fast_pace = len(recent) >= 2 and statistics.median(recent) < FAST_PACE_S
        # A move after real thought is slow whatever the pace was: the pace lingers four
        # moves after a burst of recaptures, and a find after 22 s went to text.
        quick = act.what == "move" and (think < self.QUICK_S or (fast_pace and think < FAST_PACE_S))
        if act.what == "takeback":
            self.withdrawn.add(self.moves)
        super().on_act(act, now)
        if act.what == "move":
            self.last_move_at = now
            self.quick = quick
            self.last_think = think
        playing = self.v.playing
        if playing is not None:
            msg = playing[0]
            gone = (msg.kind == "hint") or (
                msg.kind == "lesson" and msg.about_move in self.withdrawn
            )
            if gone:
                self.v.stop(
                    now, "its position is gone" if msg.kind == "hint" else "the move was taken back"
                )
        if act.what == "new":
            self.positions += 1
            self.release_summary(now)
        self.drop_stale_queue(now, "no longer the news")

    # -- the coach's sentences ------------------------------------------------------------

    def on_sentence(self, msg: Msg, i: int, now: float) -> None:
        s = msg.sentences[i]
        if msg.kind in ("praise", "short"):
            self.on_praise(msg, i, s, now)
            return
        if not self.fresh_sentence(msg, s, now):
            if i == 0:
                self.dead.add(msg.id)
            self.note(msg, s, "no longer the news")
            return
        self.v.queue.append((msg, s))
        self.v.queue.sort(key=lambda p: (self.PRIORITY[p[0].kind], p[0].id))

    def on_praise(self, msg: Msg, i: int, s: str, now: float) -> None:
        if msg.id in self.summarised or any(m.id == msg.id for m, _ in self.summary):
            # The line's word is its first sentence: the rest is text.
            self.note(msg, s, "the line's word: its first sentence only")
            return
        if msg.id in self.speaking:
            self.v.queue.append((msg, s))  # a later sentence of a word being said
            return
        latest = self.moves == msg.about_move
        # A short word (a template, half a second) on a move that took SHORT_OK_S or more is
        # said even at a fast pace: it is over long before the next move lands.
        short_ok = msg.kind == "short" and self.last_think >= self.SHORT_OK_S
        if latest and (not self.quick or short_ok) and self.fresh(msg, now):
            # Slow play, the word on the move just made: said now, after what plays.
            self.speaking.add(msg.id)
            self.v.queue.append((msg, s))
            self.v.queue.sort(key=lambda p: (self.PRIORITY[p[0].kind], p[0].id))
            return
        # A fast line, or the player already moved on: the model's word on a find is held for
        # the end of the line (the line's first such); a template on a quick move is text.
        if msg.model and not self.summary:
            self.summary.append((msg, s))
            return
        self.dead.add(msg.id)
        self.note(msg, s, "a fast line: text only")

    def release_summary(self, now: float) -> None:
        if not self.summary:
            return
        msg = self.summary[0][0]
        if now - msg.asked_at <= self.SUMMARY_FRESH_S:
            self.summarised.add(msg.id)
            for m, s in self.summary:
                self.v.queue.append((m, Held(s)))
        else:
            for m, s in self.summary:
                self.note(m, s, "the line's word went stale")
        self.summary = []

    def on_tick(self, now: float) -> None:
        if self.summary and now - self.last_move_at >= self.BURST_END_S:
            self.release_summary(now)
        if self.v.playing is None and self.v.queue:
            self.drop_stale_queue(now, "no longer the news")


POLICIES = {p.name: p for p in (Current, Proposal, Burst)}


# ---- the run ------------------------------------------------------------------------------


def run(policy_cls: type[Policy], acts_: list[Act]) -> tuple[list[Heard], list[Msg]]:
    voice = Voice()
    pol = policy_cls(voice)
    msgs = messages_for(acts_, policy_cls.fast_templates)
    events: list[tuple[float, int, object]] = []
    for a in acts_:
        events.append((a.t, 0, a))
    for m in msgs:
        for i, t in enumerate(m.ready):
            events.append((t, 1, (m, i)))
    events.sort(key=lambda e: (e[0], e[1]))
    end = max(e[0] for e in events) + 15
    step = 0.05
    now = 0.0
    k = 0
    while now < end:
        while k < len(events) and events[k][0] <= now:
            _, typ, obj = events[k]
            if typ == 0:
                pol.on_act(obj, now)  # type: ignore[arg-type]
            else:
                m, i = obj  # type: ignore[misc]
                pol.on_sentence(m, i, now)
            k += 1
        pol.on_tick(now)
        voice.tick(now)
        now += step
    voice.stop(end, "end")
    return voice.log, msgs


def score(log: list[Heard], msgs: list[Msg], acts_: list[Act]) -> dict[str, str]:
    by_msg: dict[int, list[Heard]] = {}
    for h in log:
        by_msg.setdefault(h.msg.id, []).append(h)
    moves_at = [a.t for a in acts_ if a.what in ("move", "takeback", "new")]

    def whole(m: Msg) -> bool:
        hs = by_msg.get(m.id, [])
        return len(hs) >= len(m.sentences) and all(h.why == "whole" for h in hs[: len(m.sentences)])

    def any_heard(m: Msg) -> bool:
        return any(h.start is not None for h in by_msg.get(m.id, []))

    out = {}
    for kind, label in (
        ("lesson", "lesson"),
        ("praise", "praise"),
        ("short", "praise"),
        ("hint", "hint"),
        ("answer", "answer"),
    ):
        ms = [m for m in msgs if m.kind == kind]
        if not ms:
            continue
        w = sum(whole(m) for m in ms)
        p = sum(any_heard(m) and not whole(m) for m in ms)
        cur = out.get(label)
        if cur:
            a, b, c = (int(x) for x in cur.replace("+", "/").split("/"))
            w, p, n = w + a, p + b, len(ms) + c
        else:
            n = len(ms)
        out[label] = f"{w}+{p}/{n}"
    cut = sum(1 for h in log if h.why.startswith("cut"))
    stale = spoiler = 0
    overtalk = 0.0
    for h in log:
        if h.start is None:
            continue
        m = h.msg
        age = h.start - m.asked_at
        # The player's actions between the message's moment and this sentence's start.
        between = [a for a in acts_ if m.asked_at < a.t <= h.start]
        moves_between = sum(a.what == "move" for a in between)
        if m.kind == "lesson":
            # A retry: the move it is about taken back, the takeback the next thing done.
            first = next((a for a in acts_ if a.t > m.asked_at and a.what != "ask"), None)
            if first is not None and first.what == "takeback" and first.t <= h.start:
                spoiler += 1
            if age > 15 or moves_between > 1:
                stale += 1
        elif m.kind in ("praise", "short"):
            if age > 9:
                stale += 1
        elif m.kind == "hint" and any(a.what in ("move", "new", "takeback") for a in between):
            stale += 1
        # Speech while the player plays fast: between two moves less than 3 s apart.
        for a, b in zip(moves_at, moves_at[1:], strict=False):
            if b - a < 3:
                lo, hi = max(a, h.start), min(b, h.end or h.start)
                if hi > lo:
                    overtalk += hi - lo
    out["cut"] = str(cut)
    out["stale"] = str(stale)
    out["spoiler"] = str(spoiler)
    out["overtalk"] = f"{overtalk:.1f}s"
    return out


def show(log: list[Heard]) -> None:
    for h in sorted(log, key=lambda h: (h.start if h.start is not None else 1e9, h.msg.id)):
        when = f"{h.start:6.1f}-{h.end:5.1f}" if h.start is not None else "      --    "
        late = (
            " (said after the game slowed)"
            if isinstance(h.sentence, Said)
            else " (the line's word, at its end)"
            if isinstance(h.sentence, Held)
            else ""
        )
        text = (
            h.sentence
            if h.why == "whole"
            else (" ".join(h.sentence.split()[: h.words_heard]) + " ..." if h.words_heard else "")
        )
        print(
            f"   {when}  [{h.msg.kind:<6} m{h.msg.about_move % 100}] {h.why:<44} {text[:70]}{late}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policy", nargs="+", default=list(POLICIES))
    ap.add_argument("--scenario", nargs="+", default=list(SCENARIOS))
    ap.add_argument("--quiet", action="store_true", help="the scores only, no timelines")
    ap.add_argument(
        "--presynth", action="store_true", help="make the next sentence while one plays"
    )
    args = ap.parse_args()
    Voice.PRESYNTH = args.presynth
    table: dict[str, dict[str, dict[str, str]]] = {}
    for key in args.scenario:
        title, acts_ = SCENARIOS[key]
        moves = ", ".join(f"{a.t:.1f}s {a.what}{'' if a.good else ' (wrong)'}" for a in acts_)
        if not args.quiet:
            print(f"\n=== {key}: {title}\n    {moves}")
        for name in args.policy:
            log, msgs = run(POLICIES[name], acts_)
            table.setdefault(key, {})[name] = score(log, msgs, acts_)
            if not args.quiet:
                print(f"  -- {name}")
                show(log)
    cols = ["lesson", "praise", "hint", "answer", "cut", "stale", "spoiler", "overtalk"]
    print("\nscores (whole+partial/total; cut, stale, overtalk: lower is better)")
    print(f"{'':4} {'policy':<10}" + "".join(f"{c:>10}" for c in cols))
    for key, per in table.items():
        for name, sc in per.items():
            print(f"{key:4} {name:<10}" + "".join(f"{sc.get(c, '-'):>10}" for c in cols))


if __name__ == "__main__":
    main()
