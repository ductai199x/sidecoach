"""Simulated students: does the coaching change how a player plays?

The rubric judges one message at a time. This tests the outcome coaching is for: a student
with a planted habit plays a game while the coach coaches it, writes down what it will
remember, and then sits an exam on new positions where that habit bites, with only its
notebook to go on. One student, two coaches (the AI and the templates), the same game: the
difference in the exam is the difference the coaching made.

The student has two parts, kept apart so its moves stay a human's:
- **Its moves are Nova's** at its rating (tools/bots/nova.py), with the habit as a bias
  (habits.py): where the habit bites, the student often plays the move it leads to.
- **Its voice is a language model's:** a persona at its rating that talks back to the
  coach (answers its questions, asks its own), asks for the review after the game and
  writes the notebook. It never calculates. In the exam it is shown its instinct (the
  habit's move) and Nova's other candidates, and decides whether a note applies.

The exam's arms, on the same positions:
  none      no notebook: the instinct every time (the habit's floor)
  off       a true, useful note about something else (the placebo: does any note help?)
  gold      the note that cures the habit, written by hand (the ceiling)
  <run>     the notebook the student wrote after a session with a coach (a run's name)
Error positions test the habit; control positions (a good instinct, a bad move on offer)
test that a note does not make the student throw away a good move.

  uv run --no-sync python tools/eval/students.py games --workers 20
  uv run --no-sync python tools/eval/facts_parallel.py students --workers 20
  uv run --no-sync python tools/eval/students.py exam-draw --workers 20
  uv run --no-sync python tools/eval/students.py session --run tmpl --coach none --student local
  uv run --no-sync python tools/eval/students.py exam --student local --arms none,off,gold,tmpl
  uv run --no-sync python tools/eval/students.py report --student local --runs tmpl,ai

Nova is a development tool on this PC only (its licence: reference-nova-bots).
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "tools" / "bots"))

import chess  # noqa: E402
import chess.pgn  # noqa: E402
from habits import (  # noqa: E402
    GOOD,
    HABITS,
    MISS,
    Evaluator,
    Situation,
    situation,
    student_move,
)

LEVELS = (800, 1200, 1600, 2000, 2500)
SEED = 2609
# Student games played for each habit and level, and the ones kept for its session: those
# where the habit shows most. Two natural games gave the habit 0-5 moves a cell -- a game
# turns lopsided early, and in a decided position nothing bites (2026-09-26); a session is
# a test of whether the coach finds and teaches a habit the student shows.
GAMES_PER_CELL = 6
GAMES_KEPT = 2
EXAM_GAMES_PER_LEVEL = 16
ERRORS_PER_CELL = 16  # exam positions where the habit bites, for each habit and level
CONTROLS_PER_LEVEL = 8  # exam positions where nothing bites and the instinct is good
EXAM_CANDIDATES = 5
PER_GAME = 2  # exam positions of a kind from one game, at most
# Of the live positions with neither a threat nor a tactic, the share whose candidates are
# searched (the controls are drawn from them; the search is the costly part).
CONTROL_SAMPLE = 0.25
MIN_PLY = 10
MAX_PLIES = 200
MAX_STUDENT_TURNS = 8  # the student's replies to the coach in one game
END_QUESTION = "How did I do this game? What do I keep missing?"

GAMES = ROOT / "data" / "students" / "games"
EXAM_GAMES = ROOT / "data" / "students" / "exam"
FACTS = ROOT / "data" / "eval" / "facts" / "students"
OUT = ROOT / "data" / "eval" / "students"
EXAM = OUT / "exam.json"
POSITIONS = OUT / "positions.jsonl"  # every exam-game position's raw numbers
ENGINE_CACHE = OUT / "engine_cache.json"


# ---- games -------------------------------------------------------------------------------

_nova = None
_evaluator: Evaluator | None = None


def _init_worker() -> None:
    global _nova, _evaluator
    from nova import NovaPolicy  # pyright: ignore[reportMissingImports]

    _nova = NovaPolicy()
    _evaluator = Evaluator()


def _style(level: int):
    from nova import Style  # pyright: ignore[reportMissingImports]

    return Style(rating=level)


def play_student(job: tuple[str, int, int, int]) -> str:
    """One game: a student with `habit` at `level` against Nova at the same level, from the
    side the index gives. The plies where the habit chose the move go in a header."""
    from nova import temperature_for  # pyright: ignore[reportMissingImports]

    habit_key, level, index, seed = job
    assert _nova is not None and _evaluator is not None
    habit = HABITS[habit_key]
    rng = random.Random(seed)
    side = chess.WHITE if index % 2 == 0 else chess.BLACK
    style, t = _style(level), temperature_for(level)
    board = chess.Board()
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game
    habit_plies: list[int] = []
    chances: list[int] = []
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        probs = _nova.probabilities([board], [style], temperature=t)[0]
        if board.turn == side:
            move, bit, by_habit = student_move(habit, level, board, probs, _evaluator, rng)
            if bit:
                chances.append(board.ply() + 1)
            if by_habit:
                habit_plies.append(board.ply() + 1)
        else:
            move = rng.choices(list(probs), weights=list(probs.values()), k=1)[0]
        node = node.add_variation(move)
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    student = f"Student {level} ({habit_key})"
    opponent = f"Nova {level}"
    game.headers.update(
        {
            "Event": "Sidecoach simulated student",
            "Site": "local",
            "Date": time.strftime("%Y.%m.%d"),
            "White": student if side == chess.WHITE else opponent,
            "Black": opponent if side == chess.WHITE else student,
            "WhiteElo": str(level),
            "BlackElo": str(level),
            "Result": outcome.result() if outcome else "1/2-1/2",
            "Termination": outcome.termination.name.lower() if outcome else "max plies",
            "CoachedSide": "white" if side == chess.WHITE else "black",
            "Habit": habit_key,
            "Level": str(level),
            # Where the habit bit (it had moves to lead to), and where it chose the move.
            "HabitChances": ",".join(map(str, chances)),
            "HabitPlies": ",".join(map(str, habit_plies)),
            "Annotator": f"Nova (novachess.ai) policy + habit {habit_key}, seed {seed}",
        }
    )
    return str(game)


def play_plain(job: tuple[int, int]) -> str:
    """One game of Nova against Nova at `level`, for the exam's positions."""
    from nova import temperature_for  # pyright: ignore[reportMissingImports]

    level, seed = job
    assert _nova is not None
    rng = random.Random(seed)
    style, t = _style(level), temperature_for(level)
    board = chess.Board()
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        probs = _nova.probabilities([board], [style], temperature=t)[0]  # type: ignore[union-attr]
        move = rng.choices(list(probs), weights=list(probs.values()), k=1)[0]
        node = node.add_variation(move)
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    game.headers.update(
        {
            "Event": "Sidecoach exam game",
            "White": f"Nova {level}",
            "Black": f"Nova {level}",
            "Result": outcome.result() if outcome else "1/2-1/2",
            "Level": str(level),
            "Annotator": f"Nova (novachess.ai) policy, seed {seed}",
        }
    )
    return str(game)


def cmd_games(args: argparse.Namespace) -> None:
    GAMES.mkdir(parents=True, exist_ok=True)
    EXAM_GAMES.mkdir(parents=True, exist_ok=True)
    student_jobs: dict[Path, list[tuple[str, int, int, int]]] = {}
    for habit in HABITS:
        for level in LEVELS:
            path = GAMES / f"{habit}-L{level}.pgn"
            if path.exists() and not (args.force or args.redo == "students"):
                continue
            student_jobs[path] = [
                (habit, level, i, SEED * 1_000_003 + hash_of(habit) * 10_007 + level * 101 + i)
                for i in range(args.games_per_cell)
            ]
    plain_jobs: dict[Path, list[tuple[int, int]]] = {}
    for level in LEVELS:
        path = EXAM_GAMES / f"L{level}.pgn"
        if path.exists() and not (args.force or args.redo == "exam"):
            continue
        plain_jobs[path] = [
            (level, SEED * 7_000_003 + level * 1_009 + i) for i in range(EXAM_GAMES_PER_LEVEL)
        ]
    t0 = time.time()
    context = mp.get_context("spawn")
    with context.Pool(args.workers, initializer=_init_worker) as pool:
        student_async = {p: pool.map_async(play_student, jobs) for p, jobs in student_jobs.items()}
        plain_async = {p: pool.map_async(play_plain, jobs) for p, jobs in plain_jobs.items()}
        for path, result in [*student_async.items(), *plain_async.items()]:
            pgns = result.get()
            if path in student_async:
                every = GAMES / "all" / path.name
                every.parent.mkdir(parents=True, exist_ok=True)
                every.write_text("\n\n".join(pgns) + "\n\n", encoding="utf-8")
                pgns = most_habit(pgns, args.keep)
            path.write_text("\n\n".join(pgns) + "\n\n", encoding="utf-8")
            print(f"{path.relative_to(ROOT)}: {len(pgns)} games ({time.time() - t0:.0f} s)")
    for path in sorted(GAMES.glob("*.pgn")):
        for game in read_games(path):
            h = game.headers
            plies = len(list(game.mainline_moves()))
            took = len([x for x in h.get("HabitPlies", "").split(",") if x])
            bit = len([x for x in h.get("HabitChances", "").split(",") if x])
            print(
                f"  {path.stem} ({h['CoachedSide']}): {h['Result']} {plies} plies, the habit bit"
                f" {bit} times and chose {took} moves"
            )


def most_habit(pgns: list[str], keep: int) -> list[str]:
    """The `keep` games in which the habit chose the most moves (then bit the most), in the
    order they were played."""

    def count(pgn: str, header: str) -> int:
        found = re.search(rf'\[{header} "([^"]*)"\]', pgn)
        return len([x for x in found.group(1).split(",") if x]) if found else 0

    ranked = sorted(
        range(len(pgns)),
        key=lambda i: (-count(pgns[i], "HabitPlies"), -count(pgns[i], "HabitChances"), i),
    )
    return [pgns[i] for i in sorted(ranked[:keep])]


def hash_of(text: str) -> int:
    """A small stable number for a name (Python's own hash is salted per process)."""
    return sum((i + 1) * ord(c) for i, c in enumerate(text)) % 997


def read_games(path: Path) -> list[chess.pgn.Game]:
    out = []
    with path.open(encoding="utf-8") as fh:
        while (game := chess.pgn.read_game(fh)) is not None:
            out.append(game)
    return out


# ---- the exam's positions --------------------------------------------------------------------


def _exam_positions(job: tuple[str, int, str]) -> list[dict]:
    """Every position of an exam game from MIN_PLY on, in the numbers the habits read
    (habits.situation): Nova's candidates at the level, each one's worth and what it comes out
    with, the best, what passing would let the opponent win, what the student's near-best
    lines net. Whether a habit bites is decided from these later (`position_situation`), so
    its thresholds can change without Stockfish."""
    from nova import temperature_for  # pyright: ignore[reportMissingImports]

    game_id, level, pgn = job
    assert _nova is not None and _evaluator is not None
    game = chess.pgn.read_game(io.StringIO(pgn))
    assert game is not None
    board = game.board()
    style, t = _style(level), temperature_for(level)
    out = []
    last_san = None
    sample = random.Random(f"{SEED}:{game_id}")

    def need(s: Situation) -> bool:
        return s.threat or s.tactic or sample.random() < CONTROL_SAMPLE

    for move in game.mainline_moves():
        if board.ply() >= MIN_PLY and not board.is_game_over():
            probs = _nova.probabilities([board], [style], temperature=t)[0]
            s = situation(board, probs, _evaluator, need=need)
            if s.live:
                out.append(record(game_id, level, board, last_san, probs, s))
        last_san = board.san(move)
        board.push(move)
    return out


def record(
    game_id: str,
    level: int,
    board: chess.Board,
    last_san: str | None,
    probs: dict[chess.Move, float],
    s: Situation,
) -> dict:
    """An exam game's position as stored: the reading and each candidate's worth and line."""
    return {
        "game": game_id,
        "level": level,
        "ply": board.ply(),
        "fen": board.fen(),
        "last_san": last_san,
        "best": round(s.best, 4),
        "threat_win": s.threat_win,
        "threat_mates": s.threat_mates,
        "tactic_win": s.tactic_win,
        "tactic_mates": s.tactic_mates,
        "owed": s.owed,
        "threat_line": s.threat_line,
        "tactic_line": s.tactic_line,
        "candidates": [
            {
                "uci": m.uci(),
                "san": board.san(m),
                "p": round(probs[m], 4),
                "score": round(s.scores[m], 4),
                "loss": round(s.loss(m), 4),
                "swing": s.swings.get(m, 0),
            }
            for m in s.candidates
        ],
    }


def position_situation(p: dict) -> Situation:
    """A stored position as the habits read it."""
    moves = [chess.Move.from_uci(c["uci"]) for c in p["candidates"]]
    return Situation(
        best=p["best"],
        probs={m: c["p"] for m, c in zip(moves, p["candidates"], strict=True)},
        scores={m: c["score"] for m, c in zip(moves, p["candidates"], strict=True)},
        candidates=moves,
        swings={m: c["swing"] for m, c in zip(moves, p["candidates"], strict=True)},
        threat_win=p["threat_win"],
        threat_mates=p["threat_mates"],
        tactic_win=p["tactic_win"],
        tactic_mates=p["tactic_mates"],
        owed=p["owed"],
        threat_line=p["threat_line"],
        tactic_line=p["tactic_line"],
    )


def habit_reading(p: dict) -> dict[str, list[str]]:
    """For each habit, the moves it leads to in this position (empty: it does not bite)."""
    s = position_situation(p)
    return {k: [m.uci() for m in h.misses(s)] if h.bites(s) else [] for k, h in HABITS.items()}


def select_items(positions: list[dict], rng: random.Random) -> list[dict]:
    """The exam: for each level, ERRORS_PER_CELL positions where each habit bites and
    CONTROLS_PER_LEVEL where the instinct is right. The student is shown its instinct and
    Nova's likeliest other moves, EXAM_CANDIDATES in all: at an error position the instinct is
    the habit's likeliest move and a good move is among the others; at a control the instinct
    is Nova's likeliest move, a good one, and a bad move is among the others (a note applied
    where it does not fit could lead the student astray). At most PER_GAME of a kind from a
    game; no position twice."""
    items: list[dict] = []
    for level in LEVELS:
        here = [p for p in positions if p["level"] == level]
        rng.shuffle(here)
        used: set[tuple[str, int]] = set()
        for kind in [*HABITS, "control"]:
            per_game: dict[str, int] = defaultdict(int)
            want = CONTROLS_PER_LEVEL if kind == "control" else ERRORS_PER_CELL
            chosen = []
            for p in here:
                if len(chosen) >= want:
                    break
                if per_game[p["game"]] >= PER_GAME or (p["game"], p["ply"]) in used:
                    continue
                cands = p["candidates"]
                if not cands:
                    continue  # read without its candidates: no exam item
                reading = habit_reading(p)
                if kind == "control":
                    if not position_situation(p).live:
                        continue
                    instinct = cands[0]
                    if instinct["loss"] >= GOOD:
                        continue
                else:
                    wrong = [c for c in cands if c["uci"] in reading[kind]]
                    if not wrong:
                        continue
                    instinct = max(wrong, key=lambda c: c["p"])
                others = [c for c in cands if c is not instinct][: EXAM_CANDIDATES - 1]
                if kind == "control":
                    if not any(c["loss"] >= MISS for c in others):
                        continue
                elif not any(c["loss"] < GOOD for c in others):
                    continue
                per_game[p["game"]] += 1
                used.add((p["game"], p["ply"]))
                chosen.append(
                    {
                        "id": f"L{level}-{kind}-{p['game']}-p{p['ply']:03d}",
                        "kind": kind,
                        "instinct": instinct["uci"],
                        **p,
                        "candidates": [instinct, *others],
                    }
                )
            if len(chosen) < want:
                print(f"  L{level} {kind}: only {len(chosen)} of {want} positions found")
            items.extend(chosen)
    return items


def exam_positions(workers: int) -> list[dict]:
    """The exam games' positions, read once (POSITIONS) and kept; read again when they were
    stored without what the habits now read (the line's gain, the last move)."""
    if POSITIONS.exists():
        kept = [json.loads(x) for x in POSITIONS.read_text("utf-8").splitlines() if x]
        if kept and all("owed" in p for p in kept):
            return kept
    jobs = []
    for level in LEVELS:
        for i, game in enumerate(read_games(EXAM_GAMES / f"L{level}.pgn")):
            jobs.append((f"L{level}x{i:02d}", level, str(game)))
    t0 = time.time()
    positions: list[dict] = []
    context = mp.get_context("spawn")
    with context.Pool(workers, initializer=_init_worker) as pool:
        for i, found in enumerate(pool.imap_unordered(_exam_positions, jobs), 1):
            positions.extend(found)
            print(f"  [{i}/{len(jobs)}] {len(positions)} positions ({time.time() - t0:.0f} s)")
    OUT.mkdir(parents=True, exist_ok=True)
    POSITIONS.write_text("".join(json.dumps(p) + "\n" for p in positions), encoding="utf-8")
    return positions


def bite_rates(positions: list[dict]) -> None:
    """How often each habit bites, by level: its share of the live positions (and per 40
    of them), how often Nova misses there anyway, and the student's target."""
    for level in LEVELS:
        here = [p for p in positions if p["level"] == level and position_situation(p).live]
        parts = []
        for key, habit in HABITS.items():
            bitten = [position_situation(p) for p in here if habit_reading(p)[key]]
            natural = [habit.natural(s) for s in bitten]
            share = len(bitten) / max(1, len(here))
            nova = sum(natural) / max(1, len(natural))
            parts.append(
                f"{key} {share:.0%} ({40 * share:.1f} per 40), real players miss"
                f" {habit.human(level):.0%}, Nova {nova:.0%}, the student {habit.student(nova):.0%}"
            )
        print(f"  L{level}: {len(here)} live positions; " + "; ".join(parts))


def cmd_exam_draw(args: argparse.Namespace) -> None:
    positions = exam_positions(args.workers)
    bite_rates(positions)
    if args.rates:
        return
    if EXAM.exists() and not args.force:
        raise SystemExit(f"{EXAM} is frozen; --force draws a new exam")
    items = select_items(positions, random.Random(SEED))
    OUT.mkdir(parents=True, exist_ok=True)
    EXAM.write_text(
        json.dumps({"seed": SEED, "drawn": time.strftime("%Y-%m-%d %H:%M"), "items": items}),
        encoding="utf-8",
    )
    counts = defaultdict(int)
    for it in items:
        counts[(it["level"], it["kind"])] += 1
    print(f"{len(items)} exam positions -> {EXAM.relative_to(ROOT)}: {dict(counts)}")


# ---- the student's voice ---------------------------------------------------------------------

VOICE = {
    800: (
        "You know how the pieces move and the basic rules of thumb (don't leave pieces "
        "hanging, get your pieces out, castle), but not chess jargon such as 'outpost', "
        "'prophylaxis' or 'zwischenzug'."
    ),
    1200: "You know the basic tactics (forks, pins, skewers) and the opening principles.",
    1600: (
        "You see simple tactics and know common strategy: weak squares, open files, pawn structure."
    ),
    2000: (
        "You are a strong club player: you know the strategic vocabulary and calculate "
        "several moves deep."
    ),
    2500: "You are a master: you know all of chess's vocabulary; basic advice bores you.",
}
TEMPERAMENTS = (
    "curious: you like to ask why",
    "terse: you say little and rarely ask anything",
    "a little defensive about your mistakes",
    "eager to improve",
)
_PIECE_NAMES = (
    (chess.KING, "King"),
    (chess.QUEEN, "Queen"),
    (chess.ROOK, "Rook"),
    (chess.BISHOP, "Bishop"),
    (chess.KNIGHT, "Knight"),
    (chess.PAWN, "Pawn"),
)


def board_text(board: chess.Board) -> str:
    """The board as a list of each side's pieces and their squares."""
    lines = []
    for colour, name in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        parts = []
        for piece, word in _PIECE_NAMES:
            squares = [chess.square_name(s) for s in board.pieces(piece, colour)]
            if squares:
                parts.append(f"{word}{'s' if len(squares) > 1 else ''} {', '.join(squares)}")
        lines.append(f"{name}: " + "; ".join(parts))
    return "\n".join(lines)


def persona(level: int, temperament: str | None) -> str:
    voice = VOICE[min(VOICE, key=lambda k: abs(k - level))]
    return (
        f"You are a chess player rated about {level} on Lichess. {voice}"
        + (f" You are {temperament}." if temperament else "")
        + " You do not know what your weaknesses are. You never mention being an AI or a"
        " simulation."
    )


def chat(provider, ledger, system: str, user: str, label: str, **kw) -> str:
    """One call of the student's model, retried on a refused request; on the ledger when the
    model is paid (judge.BudgetExceeded stops the run)."""
    last: Exception | None = None
    for attempt in range(4):
        if ledger is not None:
            ledger.check()
        t0 = time.time()
        try:
            completion = provider.chat(
                system, [{"role": "user", "content": user}], thinking=False, **kw
            )
        except Exception as exc:  # noqa: BLE001
            if type(exc).__name__ == "BudgetExceeded":
                raise
            last = exc
            time.sleep((15 if "429" in str(exc) else 5) * (attempt + 1))
            continue
        if ledger is not None:
            from providers import spec_of

            ledger.record(label, completion.stats, time.time() - t0, model=spec_of(provider))
        return completion.text.strip()
    raise RuntimeError(f"the student's model failed: {last}")


def _numbered(board: chess.Board, plies: int = 10) -> str:
    """The last moves of the game, numbered."""
    replay = chess.Board()
    sans = []
    for move in board.move_stack:
        sans.append((replay.fullmove_number, replay.turn, replay.san(move)))
        replay.push(move)
    out = []
    for number, turn, san in sans[-plies:]:
        out.append(f"{number}. {san}" if turn == chess.WHITE else f"{number}... {san}")
    return " ".join(out) or "(none yet)"


class Student:
    """The student in a session: hears the coach, sometimes answers or asks, and at the end
    writes its notebook and says what it did not understand."""

    def __init__(self, level: int, colour: chess.Color, temperament: str, provider, ledger):
        self.level, self.colour = level, colour
        self.system = persona(level, temperament)
        self.temperament = temperament
        self.provider, self.ledger = provider, ledger
        self.transcript: list[dict] = []
        self.turns = 0
        self.game_turns_from = 0  # `turns` when the current game began: the cap is a game's

    def heard(self, ply: int, texts: list[str], trigger: str | None = None) -> None:
        for text in texts:
            self.transcript.append({"ply": ply, "who": "coach", "text": text, "trigger": trigger})

    def said(self, ply: int, text: str) -> None:
        self.transcript.append({"ply": ply, "who": "you", "text": text})

    @staticmethod
    def line(t: dict) -> str:
        """A transcript entry as the student reads it: the coach's words, its own, or a marker
        (a new game)."""
        if t["who"] == "coach":
            return f"Coach: {t['text']}"
        if t["who"] == "you":
            return f"You: {t['text']}"
        return f"({t['text']})"

    def _tail(self, n: int = 12) -> str:
        return "\n".join(self.line(t) for t in self.transcript[-n:])

    def react(self, board: chess.Board, said_now: list[str]) -> str | None:
        """What the student types back after the coach's words, or None."""
        if self.turns - self.game_turns_from >= MAX_STUDENT_TURNS:
            return None
        side = "White" if self.colour == chess.WHITE else "Black"
        user = (
            f"You are playing {side}. The board now:\n{board_text(board)}\n"
            f"The last moves: {_numbered(board)}\n\n"
            f"The conversation so far (most recent last):\n{self._tail()}\n\n"
            f'The coach just said: "{" ".join(said_now)}"\n\n'
            "Reply as you would type in the app's chat box, in at most two short sentences. If "
            "the coach asked you something, answer it from what you can see on the board (you "
            "may be wrong). If something is unclear or you want to understand more, ask one "
            "short question. If you have nothing to say, reply with a single dash: -"
        )
        text = chat(self.provider, self.ledger, self.system, user, "student", temperature=0.8)
        text = text.strip().strip('"').strip()
        if not text or text.strip("-— .") == "":
            return None
        self.turns += 1
        return text

    def notebook(self, result: str) -> dict:
        """The notes the student keeps, and what it says about the coaching."""
        whole = "\n".join(self.line(t) for t in self.transcript)
        user = (
            f"Your games are over ({result}). Everything the coach said to you in them, and "
            f"what you said, in order:\n{whole}\n\n"
            "Write down what you will remember, as a real player jots a few notes after a "
            "lesson: at most five short lines, in your own words, about what to do or look "
            "for in your next games. Then list the coach's words or ideas you did not "
            "understand, and the things it told you that you already knew well.\n"
            'Answer with JSON only: {"notes": ["..."], "unclear": ["..."], '
            '"already_knew": ["..."], "helpful": <1 to 5>}'
        )
        for _ in range(3):
            text = chat(
                self.provider, self.ledger, self.system, user, "student-notes", temperature=0.4
            )
            found = re.search(r"\{.*\}", text, re.DOTALL)
            if not found:
                continue
            try:
                data = json.loads(found.group(0))
            except json.JSONDecodeError:
                continue
            notes = [str(n).strip() for n in data.get("notes") or [] if str(n).strip()][:5]
            return {
                "notes": notes,
                "unclear": [str(x) for x in data.get("unclear") or []],
                "already_knew": [str(x) for x in data.get("already_knew") or []],
                "helpful": data.get("helpful"),
            }
        return {"notes": [], "unclear": [], "already_knew": [], "helpful": None, "failed": True}


# ---- a session -----------------------------------------------------------------------------


def _result_words(result: str, colour: chess.Color) -> str:
    if result == "1/2-1/2":
        return "a draw"
    won = (result == "1-0") == (colour == chess.WHITE)
    return "you won" if won else "you lost"


def run_session(facts_paths: list[Path], coach_spec: str, student_spec: str, ledger) -> dict:
    """A student's games played through the coach one after the other, event by event as
    build_game sends them (corpus.py), with the student reading what the coach says and
    talking back, and writing its notebook after the last. Two games a session: one game gave
    the coach one or two of the habit's mistakes to find (2026-09-26)."""
    from bench import settings_for
    from coach_build import build_coach
    from corpus import CachedEngine
    from providers import Metered, coach_provider, provider_for

    games = [json.loads(p.read_text(encoding="utf-8")) for p in facts_paths]
    cell = cell_of(games[0]["name"])
    headers = games[0]["headers"]
    level = int(headers["Level"])
    rng = random.Random(f"students:{SEED}:{cell}")
    random.seed(f"sidecoach-eval:{cell}")
    temperament = rng.choice(TEMPERAMENTS)

    provider = coach_provider(coach_spec, ledger, "coach") if coach_spec != "none" else None
    inner = provider_for(student_spec, timeout_s=300)
    student_llm = inner if student_spec == "local" else Metered(inner, ledger, "student")
    student = Student(level, chess.WHITE, temperament, student_llm, None)
    engine_cache = (
        json.loads(ENGINE_CACHE.read_text(encoding="utf-8")) if ENGINE_CACHE.exists() else {}
    )
    engine = CachedEngine(engine_cache)
    coach = build_coach(
        provider,
        ledger=ledger,
        settings=settings_for(level),
        engine=engine,
        seed=f"students:{cell}",
    )
    played = [play_coached(facts, coach, student) for facts in games]
    results = "; ".join(
        f"game {i}: {_result_words(p['result'], p['colour'])}" for i, p in enumerate(played, 1)
    )
    notebook = student.notebook(results)
    engine.close()
    return {
        "game": cell,
        "games": [p["game"] for p in played],
        "habit": headers["Habit"],
        "level": level,
        "results": [p["result"] for p in played],
        "coach": coach_spec,
        "student": student_spec,
        "temperament": temperament,
        "habit_plies": [p["habit_plies"] for p in played],
        "student_moves": sum(p["student_moves"] for p in played),
        "mistakes": sum(p["mistakes"] for p in played),
        "coach_messages": sum(1 for t in student.transcript if t["who"] == "coach"),
        "student_turns": student.turns,
        "fallbacks": sum(1 for s in coach.spoken if s.problems),
        "review": [p["review"] for p in played],
        "transcript": student.transcript,
        **notebook,
        "engine_new": engine.new,
    }


def cell_of(game: str) -> str:
    """A student game's cell, its habit and level: "threat_blind-L800__001" is
    "threat_blind-L800"."""
    return game.split("__", 1)[0]


def play_coached(facts: dict, coach, student: Student) -> dict:
    """One game of the student's through the coach, with the student talking back; what the
    notebook and the report need from it."""
    from sidecoach.events import Analysis, GameEnded, GameStarted, MoveAssessment, MovePlayed, Side

    game = facts["name"]
    headers = facts["headers"]
    colour = chess.WHITE if facts["user_side"] == "white" else chess.BLACK
    student.colour = colour
    student.game_turns_from = student.turns
    student.transcript.append(
        {
            "ply": 0,
            "who": "note",
            "text": f"A new game: you play {'White' if colour == chess.WHITE else 'Black'}.",
        }
    )
    ts = 0.0

    def drain() -> list:
        coach.speaker.run_pending()
        return coach.poll()

    def talk(board: chess.Board, ply: int, messages, *, invite: bool) -> None:
        """The student hears the coach; where the words call for it (a lesson on its own
        mistake, a question to it, an answer to its question), it may reply, and the coach
        answers -- one reply more at most on the same thread."""
        nonlocal ts
        texts = [m.text for m in messages if m.text]
        if not texts:
            return
        student.heard(ply, texts, messages[-1].trigger.value)
        depth = 0
        while invite or any(t.rstrip().endswith("?") for t in texts):
            reply = student.react(board, texts)
            if reply is None:
                return
            student.said(ply, reply)
            ts += 4
            coach.ask(reply, ts)
            answer = drain()
            texts = [m.text for m in answer if m.text]
            if not texts:
                return
            student.heard(ply, texts, "answer")
            depth += 1
            if depth >= 2:
                return
            invite = True

    user_side = Side.WHITE if colour == chess.WHITE else Side.BLACK
    coach.on_event(GameStarted(game_id=game, user_side=user_side, start_fen=facts["start_fen"]), ts)
    coach.on_event(Analysis.model_validate({**facts["start_analysis"], "game_id": game}), ts)
    board = chess.Board(facts["start_fen"])
    talk(board, 0, drain(), invite=False)
    student_moves = mistakes = 0
    result = headers.get("Result", "*")
    for step in facts["script"]:
        if step["kind"] == "ended":
            ts += 2
            coach.on_event(
                GameEnded(
                    game_id=game,
                    result=step["result"],
                    reason=step.get("reason"),
                    pgn="",
                    final_fen=coach.board.fen(),
                ),
                ts,
            )
            student.heard(board.ply(), [m.text for m in drain() if m.text], "game_end")
            break
        if step["kind"] != "move":
            continue
        ts += 5
        move = MovePlayed(
            game_id=game,
            ply=step["ply"],
            san=step["san"],
            uci=step["uci"],
            fen_before=step["fen_before"],
            fen_after=step["fen_after"],
            by_user=step["by_user"],
            think_ms=None,
        )
        coach.on_event(move, ts)
        coach.on_event(MoveAssessment.model_validate({**step["assessment"], "game_id": game}), ts)
        coach.on_event(Analysis.model_validate({**step["analysis"], "game_id": game}), ts)
        board.push(chess.Move.from_uci(step["uci"]))
        messages = drain()
        costly = step["by_user"] and step["assessment"].get("severity") in ("mistake", "blunder")
        student_moves += step["by_user"]
        mistakes += costly
        talk(board, step["ply"], messages, invite=costly)
    ts += 3
    student.said(board.ply(), END_QUESTION)
    coach.ask(END_QUESTION, ts)
    review = [m.text for m in drain() if m.text]
    student.heard(board.ply(), review, "review")
    return {
        "game": game,
        "colour": colour,
        "result": result,
        "habit_plies": [int(x) for x in headers.get("HabitPlies", "").split(",") if x],
        "student_moves": student_moves,
        "mistakes": mistakes,
        "review": review,
    }


def cmd_session(args: argparse.Namespace) -> None:
    import stamp
    from judge import BudgetExceeded, Ledger

    paid = args.coach != "none" or args.student != "local"
    ledger = (
        Ledger(
            f"students-{args.run}",
            "students",
            args.coach if args.coach != "none" else args.student,
            budget_usd=args.budget_usd,
            resume=True,
            version=stamp.code_version(),
        )
        if paid
        else None
    )
    if paid and args.budget_usd is None:
        raise SystemExit("a paid run needs --budget-usd")
    out = OUT / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    cells: dict[str, list[Path]] = defaultdict(list)
    for p in sorted(FACTS.glob("*.json")):
        cells[cell_of(p.stem)].append(p)
    todo = [
        (cell, paths)
        for cell, paths in sorted(cells.items())
        if not (out / f"{cell}.json").exists()
        and (not args.only or any(k in cell for k in args.only))
    ]
    print(f"{len(todo)} sessions to run -> {out.relative_to(ROOT)}", flush=True)
    t0 = time.time()
    new_engine: dict = {}

    def one(job: tuple[str, list[Path]]) -> str:
        cell, paths = job
        record = run_session(paths, args.coach, args.student, ledger)
        new_engine.update(record.pop("engine_new"))
        (out / f"{cell}.json").write_text(json.dumps(record, ensure_ascii=False, indent=1), "utf-8")
        spent = f", ${ledger.cost():.3f}" if ledger is not None else ""
        return (
            f"{record['game']}: {record['mistakes']} mistakes, {record['coach_messages']} coach"
            f" messages, {record['student_turns']} replies, {len(record['notes'])} notes"
            f" ({time.time() - t0:.0f} s{spent})"
        )

    try:
        with ThreadPoolExecutor(args.workers) as pool:
            for line in pool.map(one, todo):
                print("  " + line, flush=True)
    except BudgetExceeded as exc:
        print(f"stopped: {exc}")
    finally:
        if new_engine:
            cache = (
                json.loads(ENGINE_CACHE.read_text(encoding="utf-8"))
                if ENGINE_CACHE.exists()
                else {}
            )
            cache.update(new_engine)
            ENGINE_CACHE.write_text(json.dumps(cache), encoding="utf-8")
    if ledger is not None:
        print(ledger.summary())


# ---- the exam --------------------------------------------------------------------------------


def _norm(san: str) -> str:
    return san.strip().strip(".,;:'\"").rstrip("+#!?").replace("0-0", "O-O")


def decide(item: dict, notes: list[str], provider, ledger) -> tuple[str, bool, str]:
    """The student's move in an exam position with this notebook: (uci, a valid answer, what
    it said). An empty notebook is the instinct, no call."""
    instinct = next(c for c in item["candidates"] if c["uci"] == item["instinct"])
    if not notes:
        return instinct["uci"], True, ""
    rng = random.Random(item["id"])
    others = [c for c in item["candidates"] if c["uci"] != item["instinct"]]
    rng.shuffle(others)
    board = chess.Board(item["fen"])
    side = "White" if board.turn == chess.WHITE else "Black"
    last = f"Their last move: {item['last_san']}\n" if item.get("last_san") else ""
    notebook = "\n".join(f"- {n}" for n in notes)
    user = (
        f"You are playing {side}, and it is your move. The board:\n{board_text(board)}\n{last}\n"
        f"Your notebook, from your lessons with your coach:\n{notebook}\n\n"
        f"Your first instinct is to play {instinct['san']}. The other moves you are also "
        f"considering: {', '.join(c['san'] for c in others)}.\n\n"
        "Check your notebook. If something in it applies to this position and changes your "
        "mind, play the move it leads you to; if nothing in it applies, play your instinct.\n"
        "Answer on one line exactly:\nMOVE: <one of the moves above> | NOTE: <the note you "
        "used, or none>"
    )
    text = chat(provider, ledger, persona(item["level"], None), user, "exam", temperature=0.0)
    found = re.search(r"MOVE:\s*([^|\n]+)", text)
    said = _norm(found.group(1).split()[0]) if found and found.group(1).split() else ""
    for c in item["candidates"]:
        if _norm(c["san"]) == said:
            return c["uci"], True, text
    return instinct["uci"], False, text


def _notebooks(run: str) -> dict[tuple[str, int], list[tuple[str, list[str]]]]:
    """A run's notebooks by (habit, level): (the game, its notes) for each session."""
    out: dict[tuple[str, int], list[tuple[str, list[str]]]] = defaultdict(list)
    for path in sorted((OUT / "runs" / run).glob("*.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        out[(s["habit"], s["level"])].append((s["game"], s["notes"]))
    return out


def cmd_exam(args: argparse.Namespace) -> None:
    import stamp
    from judge import BudgetExceeded, Ledger
    from providers import Metered, provider_for

    items = json.loads(EXAM.read_text(encoding="utf-8"))["items"]
    inner = provider_for(args.student, timeout_s=300)
    ledger = None
    provider = inner
    if args.student != "local":
        if args.budget_usd is None:
            raise SystemExit("a paid student needs --budget-usd")
        ledger = Ledger(
            f"students-exam-{args.student}",
            "students",
            args.student,
            budget_usd=args.budget_usd,
            resume=True,
            version=stamp.code_version(),
        )
        provider = Metered(inner, ledger, "exam")
    folder = OUT / "exam" / tag(args.student)
    folder.mkdir(parents=True, exist_ok=True)
    for arm in args.arms.split(","):
        path = folder / f"{arm}.jsonl"
        done = set()
        if path.exists():
            done = {
                (r["item"], r["notebook"])
                for r in (json.loads(x) for x in path.read_text("utf-8").splitlines() if x)
            }
        jobs = []
        for habit in HABITS:
            books: list[tuple[str, list[str]]] = []
            if arm == "none":
                books = [("none", [])]
            elif arm == "off":
                books = [("off", [HABITS[habit].off_note])]
            elif arm == "gold":
                books = [("gold", [HABITS[habit].gold_note])]
            for level in LEVELS:
                if arm not in ("none", "off", "gold"):
                    books = _notebooks(arm).get((habit, level), [])
                cell = [
                    it for it in items if it["level"] == level and it["kind"] in (habit, "control")
                ]
                for game, notes in books:
                    for it in cell:
                        if (it["id"], f"{habit}:{game}") not in done:
                            jobs.append((habit, level, game, notes, it))
        print(f"{arm}: {len(jobs)} decisions to make", flush=True)

        def one(job):
            habit, level, game, notes, it = job
            uci, valid, said = decide(it, notes, provider, ledger)
            chosen = next(c for c in it["candidates"] if c["uci"] == uci)
            return {
                "item": it["id"],
                "notebook": f"{habit}:{game}",
                "habit": habit,
                "level": level,
                "kind": it["kind"],
                "choice": uci,
                "loss": chosen["loss"],
                "good": chosen["loss"] < GOOD,
                "switched": uci != it["instinct"],
                "valid": valid,
                "said": said,
            }

        t0 = time.time()
        try:
            with (
                ThreadPoolExecutor(args.workers) as pool,
                path.open("a", encoding="utf-8") as fh,
            ):
                for i, rec in enumerate(pool.map(one, jobs), 1):
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if i % 50 == 0:
                        print(f"  {i}/{len(jobs)} ({time.time() - t0:.0f} s)", flush=True)
        except BudgetExceeded as exc:
            print(f"stopped: {exc}")
            break
    if ledger is not None:
        print(ledger.summary())


def tag(spec: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", spec)


# ---- the report ------------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> None:
    from bench import mcnemar, wilson

    folder = OUT / "exam" / tag(args.student)
    arms = ["none", "off", "gold", *[r for r in args.runs.split(",") if r]]
    rows: dict[str, list[dict]] = {}
    for arm in arms:
        path = folder / f"{arm}.jsonl"
        if path.exists():
            rows[arm] = [json.loads(x) for x in path.read_text("utf-8").splitlines() if x]

    def rate(recs: list[dict]) -> str:
        k, n = sum(r["good"] for r in recs), len(recs)
        if not n:
            return "-"
        lo, hi = wilson(k, n)
        return f"{k / n:.2f} [{lo:.2f}, {hi:.2f}] n={n}"

    print(f"Exam, student {args.student}: share of positions where the move played is good")
    print(f"  (loss under {GOOD}); error positions test the habit, controls a good instinct")
    for scope in ("all", *map(str, LEVELS)):
        print(f"\n{'all levels' if scope == 'all' else 'L' + scope}")
        for arm, recs in rows.items():
            here = [r for r in recs if scope == "all" or str(r["level"]) == scope]
            errors = [r for r in here if r["kind"] != "control"]
            controls = [r for r in here if r["kind"] == "control"]
            invalid = sum(not r["valid"] for r in here)
            print(
                f"  {arm:>14}  errors fixed {rate(errors):<28} controls kept {rate(controls):<28}"
                + (f" invalid {invalid}" if invalid else "")
            )
    runs = [r for r in args.runs.split(",") if r in rows]
    if len(runs) == 2:
        a, b = runs
        by_a = {(r["item"], r["notebook"].split(":")[1]): r for r in rows[a]}
        pairs = [
            (by_a[(r["item"], r["notebook"].split(":")[1])], r)
            for r in rows[b]
            if (r["item"], r["notebook"].split(":")[1]) in by_a
        ]
        for kind in ("errors", "controls"):
            ps = [(x, y) for x, y in pairs if (x["kind"] == "control") == (kind == "controls")]
            only_a = sum(x["good"] and not y["good"] for x, y in ps)
            only_b = sum(y["good"] and not x["good"] for x, y in ps)
            print(
                f"\npaired {a} v {b}, {kind}: {len(ps)} pairs, only {a} good {only_a}, only {b}"
                f" good {only_b}, McNemar p={mcnemar(only_a, only_b):.3f}"
            )
    for run in runs:
        sessions = [
            json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((OUT / "runs" / run).glob("*.json"))
        ]
        print(f"\nsessions {run}: {len(sessions)}")
        for s in sessions:
            chose = sum(len(h) for h in s["habit_plies"])
            print(
                f"  {s['game']:<20} {' '.join(s['results']):<16} habit {chose:>2}"
                f" mistakes {s['mistakes']:>2} said {s['coach_messages']:>3} replies"
                f" {s['student_turns']} unclear {len(s['unclear'])} knew"
                f" {len(s['already_knew'])} helpful {s.get('helpful')}"
            )
            for note in s["notes"]:
                print(f"      - {note}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("games")
    g.add_argument("--workers", type=int, default=20)
    g.add_argument("--games-per-cell", type=int, default=GAMES_PER_CELL)
    g.add_argument("--keep", type=int, default=GAMES_KEPT, help="the games a session is given")
    g.add_argument("--force", action="store_true", help="play every game again")
    g.add_argument("--redo", choices=["students", "exam"], help="play only these again")
    d = sub.add_parser("exam-draw")
    d.add_argument("--workers", type=int, default=20)
    d.add_argument("--force", action="store_true")
    d.add_argument("--rates", action="store_true", help="only print how often each habit bites")
    s = sub.add_parser("session")
    s.add_argument("--run", required=True)
    s.add_argument("--coach", default="none", help="none, or a model spec (providers.py)")
    s.add_argument("--student", default="local", help="the student's model spec")
    s.add_argument("--budget-usd", type=float)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--only", nargs="*", help="only the games whose names hold one of these")
    e = sub.add_parser("exam")
    e.add_argument("--student", default="local")
    e.add_argument("--arms", default="none,off,gold")
    e.add_argument("--budget-usd", type=float)
    e.add_argument("--workers", type=int, default=4)
    r = sub.add_parser("report")
    r.add_argument("--student", default="local")
    r.add_argument("--runs", default="")
    args = ap.parse_args()
    {
        "games": cmd_games,
        "exam-draw": cmd_exam_draw,
        "session": cmd_session,
        "exam": cmd_exam,
        "report": cmd_report,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
