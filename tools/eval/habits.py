"""The simulated students' planted habits: when each one bites, which moves it leads to, how
often it takes the student's move, and the note that would cure it.

A habit is one of the profile's mistake causes (sidecoach/profile/causes.py) made into a bias
on Nova's moves (tools/bots/nova.py). Where the habit bites, the student plays one of the
moves it leads to often enough that its miss rate there is the habit's target at its rating.
The moves stay a human's -- Nova's, at the student's rating -- and the mistakes are the same
kind every time, so the coach has a pattern to find and teach.

What bites and what a miss is are counted in material on the engine's lines, each read
where it settles (`settled`: past the window, at the first point where no capture or check is
pending) and net of what the opponent's last capture took (`coach/idea.exchange_owed`, as the
coach counts it), in live positions only (the best move's expected score between LIVE and
1 - LIVE: in a decided game nothing is missed):

  threat   if the student passed, the opponent's best line would win at least the exchange
           (TACTIC_GAIN) within THREAT_PLIES, or mate within MATE_MOVES. Counted from the
           position as it stands, so a piece that has just taken and can retreat is no threat,
           and a capture that attacks something new is one.
  tactic   the student's best line wins at least the exchange within TACTIC_PLIES, net of what
           the opponent's last capture took (taking back is not a tactic, nor is an
           in-between check before taking back), or mates within MATE_MOVES.
  a miss   a move that loses MISS or more against the best (the coach's "mistake"); at a
           threat, one whose own line also loses at least the exchange within THREAT_PLIES
           (the threat not met -- a poor answer to it is another mistake).

Measured on real players (tools/bots/habit_rates.py) and read by hand before use. A measure of
"missed" must pass: misses fall with the rating; they are commoner where a threat or a tactic
is on the board than elsewhere; examples at both ends of the ratings read right.

The first detectors (2026-09-26, the same night) were each caught by one of those checks: a
static count took recaptures for tactics and exchanges for threats (2300s "passed up" a free
piece 18% of the time); a threat read from what passing costs counted the student's own strong
move as the opponent's threat (3 of 8 read by hand); a tactic read from a 7-ply line counted
slow gains and in-between recaptures, and flipped between engine runs. Tai: "If I wasn't here
you would have just accepted shitty numbers."

Where the habit bites, it takes the student's move half the time (TAKEOVER), whatever Nova
would have played: a blind spot planted on top of the student's own play, seen two or three
times a game, which the coach can find and its notes can cure. The first rule set the student's
miss rate at twice real players' (HUMAN_MISS, kept as the reference) -- but Nova in its own
games already misses about that often, and the habit chose 2 moves in 20 games (2026-09-26).

Each habit also carries the note a student would ideally have written for it (the exam's
ceiling) and an off-topic note (true, useful, about something else: the exam's placebo).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

import chess
import chess.engine

from sidecoach.analysis.score import human_expected_score
from sidecoach.coach import idea as idea_of

GOOD = 0.04  # a good move: less than an inaccuracy (score.Thresholds)
MISS = 0.10  # a miss: a mistake or worse
LIVE = 0.10
TACTIC_GAIN = 2  # at least the exchange
# How far a line is read before it is followed to where it settles (captures and checks are
# always followed through): a tactic with a quiet move in it -- a deflection, then the fork,
# then the capture -- takes eight plies (Tai, 2026-09-26: "6 plies is the minimum ... some
# difficult tactics are like 8 plies").
THREAT_PLIES = 6  # their move and the reply, three times
TACTIC_PLIES = 8  # the player's move and the reply, four times
MATE_MOVES = 4  # a mate within the tactic window
SETTLE_MAX = 14  # the furthest a line is followed to where it settles
LOST = -100  # a line that is mated
# Nova's candidates: its likeliest moves, up to this share of the probability or this many.
CANDIDATE_MASS = 0.95
CANDIDATE_LIMIT = 8
# The engine's lines a tactic or a threat is looked for in: those within NEAR of the best. MISS
# (a mistake apart) let in slow plans a mistake worse than the best that picked up material
# four moves on (the depth-18 hand review, 2026-09-26); GOOD still takes two moves as good.
NEAR = GOOD
TOP_LINES = 3
# The coach's own depth (sidecoach.toml [engine] depth): the habit and the coach read the
# same board. Depth 12 with the hash of earlier positions flipped a quarter of the labels
# between two runs; depth 14, repeatable once the hash was cleared per position, still
# disagreed with depth 18 on 38% of the tactics either marked (2026-09-26).
DEPTH: int | None = None  # None: the settings' depth

# Real players' miss rates by rating band (midpoints), habit_rates.py at depth 18
# (2026-09-26, 2,500 positions a band): threats as measured, 14.8% at 800-1100 falling to
# 3.0% at 2300+; tactics 27, 30, 27, 11, 15, 13% with about 130 tactic positions a band
# (+-8%), made to fall by pooling each rise with the band below it.
_BANDS = (950, 1250, 1550, 1850, 2150, 2450)
HUMAN_MISS = {
    "threat_blind": tuple(zip(_BANDS, (0.148, 0.113, 0.097, 0.084, 0.047, 0.030), strict=True)),
    "tactic_blind": tuple(zip(_BANDS, (0.284, 0.284, 0.273, 0.130, 0.130, 0.130), strict=True)),
}
# How often the habit takes the move where it bites.
TAKEOVER = 0.5


def interpolate(points: Sequence[tuple[float, float]], x: float) -> float:
    if x <= points[0][0]:
        return points[0][1]
    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return points[-1][1]


def top_moves(
    probs: dict[chess.Move, float], mass: float = CANDIDATE_MASS, limit: int = CANDIDATE_LIMIT
) -> list[chess.Move]:
    """Nova's likeliest moves, likeliest first, until they hold `mass` of the probability or
    number `limit`."""
    out: list[chess.Move] = []
    total = 0.0
    for move, p in sorted(probs.items(), key=lambda kv: -kv[1]):
        if len(out) >= limit or total >= mass:
            break
        out.append(move)
        total += p
    return out


def material(board: chess.Board, colour: chess.Color) -> int:
    return sum(
        idea_of.VALUES.get(p.piece_type, 0) * (1 if p.color == colour else -1)
        for p in board.piece_map().values()
    )


def settled(board: chess.Board, line: Sequence[chess.Move], plies: int) -> int:
    """What `line` nets for the side to move on `board`, in pawns, read where it settles: after
    at least `plies` plies, at the first point where the line's next move takes nothing and
    gives no check (or where it ends, at SETTLE_MAX). So an in-between check before taking
    back, a recapture that comes later, or an exchange the window cut short are counted
    through: the best moment of a line (`idea.line_gain`) read "...Bxf4 Re1+ Kf8 Qxf4" as a
    bishop won, and "...Qxd2 Nc7+ Kd7" as a queen with Bxd2 and Kxc7 to come (2026-09-26)."""
    us = board.turn
    b = board.copy(stack=False)
    start = material(b, us)
    for k, move in enumerate(line[:SETTLE_MAX]):
        if move not in b.legal_moves:
            break
        if k >= plies and not b.is_capture(move) and not b.gives_check(move):
            break
        b.push(move)
    return material(b, us) - start


def swing(board: chess.Board, line: Sequence[chess.Move], mate: int | None) -> int:
    """What the side to move comes out with, in pawns, when it plays `line` (its move, then
    the opponent's best), read where it settles past the opponent's THREAT_PLIES after it.
    LOST when the line is mated."""
    if mate is not None and mate < 0:
        return LOST
    if not line or line[0] not in board.legal_moves:
        return 0
    return settled(board, line, 1 + THREAT_PLIES)


class Evaluator:
    """Stockfish on one thread: a position's best line, chosen moves' worth and lines, and the
    opponent's line if the side to move passed."""

    def __init__(self, depth: int | None = DEPTH, hash_mb: int = 64) -> None:
        from sidecoach.config import load_settings, require_stockfish

        settings = load_settings()
        self.depth = depth or settings.engine.depth
        self.engine = chess.engine.SimpleEngine.popen_uci(str(require_stockfish(settings)))
        self.engine.configure({"Threads": 1, "Hash": hash_mb})

    def close(self) -> None:
        self.engine.quit()

    def fresh(self) -> None:
        """A clean hash: what a position reads does not depend on what was searched before."""
        self.engine.configure({"Clear Hash": None})

    def top(
        self, board: chess.Board, n: int = TOP_LINES
    ) -> list[tuple[float, list[chess.Move], int | None]]:
        """The engine's `n` best lines: (expected score for the side to move, the line, the
        mate in it), best first."""
        infos = self.engine.analyse(board, chess.engine.Limit(depth=self.depth), multipv=n)
        out = []
        for info in infos:
            pv = list(info.get("pv") or [])
            if pv:
                score, mate = self._score(info, board.turn)
                out.append((score, pv, mate))
        return out

    @staticmethod
    def _score(info: chess.engine.InfoDict, turn: chess.Color) -> tuple[float, int | None]:
        found = info.get("score")
        if found is None:
            return 0.5, None
        score = found.pov(turn)
        return human_expected_score(score.score(), score.mate()), score.mate()

    def best_line(self, board: chess.Board) -> tuple[float, list[chess.Move], int | None]:
        """The side to move's expected score with its best move, the line, and the mate in
        it (moves; negative when the side to move is mated)."""
        info = self.engine.analyse(board, chess.engine.Limit(depth=self.depth))
        score, mate = self._score(info, board.turn)
        return score, list(info.get("pv") or []), mate

    def lines(
        self, board: chess.Board, moves: Sequence[chess.Move]
    ) -> dict[chess.Move, tuple[float, list[chess.Move], int | None]]:
        """For each of `moves`: the side to move's expected score after it, its line (the
        move first) and the mate in it."""
        infos = self.engine.analyse(
            board,
            chess.engine.Limit(depth=self.depth),
            multipv=len(moves),
            root_moves=list(moves),
        )
        out: dict[chess.Move, tuple[float, list[chess.Move], int | None]] = {}
        for info in infos:
            pv = list(info.get("pv") or [])
            if pv:
                score, mate = self._score(info, board.turn)
                out[pv[0]] = (score, pv, mate)
        # A move the search did not report (it can drop one at a shallow depth) is searched on
        # its own.
        for move in moves:
            if move not in out:
                after = board.copy(stack=False)
                after.push(move)
                if after.is_game_over():
                    out[move] = (1.0 if after.is_checkmate() else 0.5, [move], None)
                else:
                    score, pv, mate = self.best_line(after)
                    out[move] = (1.0 - score, [move, *pv], -mate if mate is not None else None)
        return out

    def passed(self, board: chess.Board) -> tuple[int, bool, list[chess.Move]] | None:
        """If the side to move passed: the most the opponent's lines within NEAR of their best
        would win, read where they settle past THREAT_PLIES, whether one mates within
        MATE_MOVES, and the line that wins the most. None in check (no pass)."""
        if board.is_check():
            return None
        passed = board.copy(stack=False)
        passed.push(chess.Move.null())
        # From its FEN: a board whose history holds a null move is sent to the engine without
        # it, with a warning on every call.
        passed = chess.Board(passed.fen())
        lines = self.top(passed)
        if not lines:
            return 0, False, []
        near = [(pv, mate) for score, pv, mate in lines if score >= lines[0][0] - NEAR]
        mates = any(mate is not None and 0 < mate <= MATE_MOVES for _, mate in near)
        win, line = max(
            ((settled(passed, pv, THREAT_PLIES), pv) for pv, _ in near), key=lambda x: x[0]
        )
        return win, mates, line


@dataclass
class Situation:
    """A position as the habits read it: Nova's candidates with their worth and what each
    comes out with, the best line, and the opponent's line if the student passed."""

    best: float
    probs: dict[chess.Move, float]
    scores: dict[chess.Move, float]
    candidates: list[chess.Move] = field(default_factory=list)
    swings: dict[chess.Move, int] = field(default_factory=dict)
    threat_win: int | None = None  # None: in check
    threat_mates: bool = False
    tactic_win: int = 0  # net of what the opponent's last capture took
    tactic_mates: bool = False
    owed: int = 0  # what the opponent's last capture took (`idea.exchange_owed`)
    # The lines the threat and the tactic were read from (UCI), for reading by hand.
    threat_line: list[str] = field(default_factory=list)
    tactic_line: list[str] = field(default_factory=list)

    def loss(self, move: chess.Move) -> float:
        return max(0.0, self.best - self.scores[move])

    @property
    def live(self) -> bool:
        return LIVE <= self.best <= 1 - LIVE

    @property
    def threat(self) -> bool:
        return (
            self.live
            and self.threat_win is not None
            and (self.threat_win >= TACTIC_GAIN or self.threat_mates)
        )

    @property
    def tactic(self) -> bool:
        return self.live and (self.tactic_win >= TACTIC_GAIN or self.tactic_mates)

    def good(self) -> list[chess.Move]:
        return [m for m in self.candidates if self.loss(m) < GOOD]


def situation(
    board: chess.Board,
    probs: dict[chess.Move, float],
    evaluator: Evaluator,
    need: Callable[[Situation], bool] | None = None,
    **kw,
) -> Situation:
    """`board` as the habits read it. The board should carry its moves: the last ones are
    what tells an exchange in progress (`idea.exchange_owed`).

    Read in stages, the costly search last: the engine's top lines (a decided position stops
    there, with no candidates: nothing bites in it); the pass and the tactic from those lines;
    and only where `need` says so of that reading (every live position when it is None),
    Nova's candidates, each searched to its line. At the coach's depth a position read whole
    took about five seconds, most of it the candidates (2026-09-26)."""
    evaluator.fresh()
    top = evaluator.top(board)
    first = Situation(best=top[0][0] if top else 0.5, probs=probs, scores={})
    if not first.live:
        return first
    passed = evaluator.passed(board)
    win, mates, tactic_line = tactic_of(board, top, first.best)
    first = replace(
        first,
        threat_win=None if passed is None else passed[0],
        threat_mates=bool(passed and passed[1]),
        tactic_win=win,
        tactic_mates=mates,
        owed=idea_of.worth(idea_of.exchange_owed(board)),
        threat_line=[m.uci() for m in passed[2]] if passed else [],
        tactic_line=[m.uci() for m in tactic_line],
    )
    if need is not None and not need(first):
        return first
    candidates = top_moves(probs, **kw)
    lines = evaluator.lines(board, candidates)
    best = max(first.best, *(score for score, _, _ in lines.values()))
    win, mates, tactic_line = tactic_of(board, [*top, *lines.values()], best)
    return Situation(
        best=best,
        probs=probs,
        scores={m: score for m, (score, _, _) in lines.items()},
        candidates=candidates,
        swings={m: swing(board, line, mt) for m, (_, line, mt) in lines.items()},
        threat_win=None if passed is None else passed[0],
        threat_mates=bool(passed and passed[1]),
        tactic_win=win,
        tactic_mates=mates,
        owed=idea_of.worth(idea_of.exchange_owed(board)),
        threat_line=[m.uci() for m in passed[2]] if passed else [],
        tactic_line=[m.uci() for m in tactic_line],
    )


def tactic_of(
    board: chess.Board,
    lines: Sequence[tuple[float, list[chess.Move], int | None]],
    best: float,
) -> tuple[int, bool, list[chess.Move]]:
    """The student's tactic, from every line within NEAR of the best (the engine's and its
    candidates'): the most one nets, read where it settles past TACTIC_PLIES and net of what
    the opponent's last capture took, whether one mates within MATE_MOVES, and the line that
    nets the most. Read from the single best line, the label flipped whenever two moves were
    as good (2026-09-26)."""
    owed = idea_of.worth(idea_of.exchange_owed(board))
    near = [(pv, mate) for score, pv, mate in lines if pv and score >= best - NEAR]
    mates = any(mate is not None and 0 < mate <= MATE_MOVES for _, mate in near)
    if not near:
        return -owed, mates, []
    win, line = max(((settled(board, pv, TACTIC_PLIES), pv) for pv, _ in near), key=lambda x: x[0])
    return win - owed, mates, line


@dataclass(frozen=True)
class Habit:
    key: str
    cause: str  # the profile's name for it (profile/causes.py)
    trait: str  # what it makes the student do, for the reports; never told to the student
    gold_note: str
    off_note: str

    def on(self, s: Situation) -> bool:
        return s.threat if self.key == "threat_blind" else s.tactic

    def misses(self, s: Situation) -> list[chess.Move]:
        """The candidates that miss it: a mistake whose own line, where it settles, loses at
        least the exchange at a threat (the threat not met), or does not win the exchange at a
        tactic (net of what they just took). A poor move that still meets the threat or cashes
        the tactic is another mistake: 25...fxg3, the wrong recapture that still won the knight
        on a5, was read as the tactic missed (the hand review, 2026-09-26)."""
        out = [m for m in s.candidates if s.loss(m) >= MISS]
        if self.key == "threat_blind":
            return [m for m in out if s.swings.get(m, 0) <= -TACTIC_GAIN]
        return [m for m in out if s.swings.get(m, 0) - s.owed < TACTIC_GAIN]

    def bites(self, s: Situation) -> bool:
        """The habit's kind of position, with a miss and a move that is not one among the
        candidates (where every move misses, there is nothing to get right)."""
        if not self.on(s):
            return False
        misses = self.misses(s)
        return bool(misses) and any(s.loss(m) < MISS for m in s.candidates)

    def human(self, rating: float) -> float:
        """How often real players of `rating` miss it (the reference, habit_rates.py)."""
        return interpolate(HUMAN_MISS[self.key], rating)

    def student(self, natural: float) -> float:
        """How often the student misses where the habit bites, Nova missing `natural` of the
        time on its own."""
        return TAKEOVER + (1 - TAKEOVER) * natural

    def natural(self, s: Situation) -> float:
        """How often Nova itself plays one of the habit's misses here, its candidates'
        probability renormalised."""
        mass = sum(s.probs.get(m, 0.0) for m in s.candidates)
        if mass <= 0:
            return 0.0
        return sum(s.probs.get(m, 0.0) for m in self.misses(s)) / mass


# A true, useful note about something neither habit is: the exam's placebo -- does any note at
# all change what the student plays?
OFF_NOTE = (
    "In the opening, get my knights and bishops out before moving a piece twice, and castle early."
)

HABITS: dict[str, Habit] = {
    h.key: h
    for h in (
        Habit(
            key="threat_blind",
            cause="missed opponent threat",
            trait="misses what the opponent threatens and loses material to it",
            gold_note=(
                "Before every move, look at what their last move attacks and at every capture "
                "or check they could play next; if a piece of mine is attacked and not safely "
                "defended, deal with that first."
            ),
            off_note=OFF_NOTE,
        ),
        Habit(
            key="tactic_blind",
            cause="never considered the candidate",
            trait="misses its own tactics: the capture, check or fork that wins material",
            gold_note=(
                "Before every move, look at all my checks, captures and attacks first; if one "
                "of them wins material or leads to mate, play it."
            ),
            off_note=OFF_NOTE,
        ),
    )
}


def sample(probs: dict[chess.Move, float], moves: Sequence[chess.Move], rng: random.Random):
    """One of `moves`, drawn by Nova's probabilities."""
    weights = [probs.get(m, 0.0) for m in moves]
    if sum(weights) <= 0:
        return rng.choice(list(moves))
    return rng.choices(list(moves), weights=weights, k=1)[0]


def student_move(
    habit: Habit,
    rating: float,
    board: chess.Board,
    probs: dict[chess.Move, float],
    evaluator: Evaluator,
    rng: random.Random,
) -> tuple[chess.Move, bool, bool]:
    """The student's move. Where the habit bites, TAKEOVER of the time one of its misses, by
    Nova's probabilities; elsewhere, and when the draw is lost, Nova's own move. The flags: the
    habit bit, and it chose the move."""
    s = situation(board, probs, evaluator, need=habit.on)
    if habit.bites(s):
        if rng.random() < TAKEOVER:
            return sample(probs, habit.misses(s), rng), True, True
        return sample(probs, list(probs), rng), True, False
    return sample(probs, list(probs), rng), False, False
