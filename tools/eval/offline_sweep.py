"""Every game with facts, both sides, through the coach with no model, and every sentence
it says checked against the board: the offline templates' regression sweep.

What is flagged, per sentence: a piece named on a square that does not hold it (a capture's
victim read from the board before the move); a fork, discovery, removal or skewer aimed at
the mover's own piece; a pin whose colours are wrong; a lesson naming the played move as
the better one; the same sentence twice in a row; a placeholder showing through. Nothing
here needs a model, Stockfish or a GPU: the facts files carry the analysis. About two
minutes for the 199 master games.

  uv run python tools/eval/offline_sweep.py            # masters, both sides
  uv run python tools/eval/offline_sweep.py --recordings --show 20
  uv run python tools/eval/offline_sweep.py --game fischer__002-fischer-larsen --transcripts out/
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import Counter
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent))

from playthrough import FACTS, load_facts, play  # noqa: E402

NAMES = {
    "pawn": chess.PAWN,
    "knight": chess.KNIGHT,
    "bishop": chess.BISHOP,
    "rook": chess.ROOK,
    "queen": chess.QUEEN,
    "king": chess.KING,
}
PIECE = re.compile(r"the (pawn|knight|bishop|rook|queen|king) on ([a-h][1-8])")
TAKING = re.compile(r"taking the (pawn|knight|bishop|rook|queen|king) on ([a-h][1-8])")
PIN = re.compile(
    r"the \w+ on ([a-h][1-8]) is pinned to the \w+ on ([a-h][1-8]) by the \w+ on ([a-h][1-8])"
)
MOVE_LINE = re.compile(r"^(\d+)(\.|\.\.\.) (\S+)\s+\((the player|the opponent), (\w+)\)")
COACH_LINE = re.compile(
    r"^\s+\[(after the move|answer|the player is stuck|game over)\] COACH: (.*)$"
)
ATTACKS = (" forks ", " now bears on ", " removes a defender of ", " skewers ")
PLACEHOLDERS = ("the idea was the position", "None", "nan", "{", "}", "undefined", "  ")
# What the player sees on the board for themselves, never said back after their move (Tai,
# 2026-09-23).
PLAIN = re.compile(
    r"(taking|takes) the (pawn|knight|bishop|rook|queen|king) on|with check|castling "
    r"(king|queen)side|castles|promoting to"
)


def _pieces(board: chess.Board, before: chess.Board | None, text: str) -> list[str]:
    """Every 'the X on sq' names what stands there; a capture's victim stood there before."""
    out = []
    taken = set(TAKING.findall(text))
    for name, sq in PIECE.findall(text):
        frame = before if (name, sq) in taken and before is not None else board
        piece = frame.piece_at(chess.parse_square(sq))
        if piece is None or piece.piece_type != NAMES[name]:
            out.append(f"'{name} on {sq}' but the square holds {piece}")
    return out


def _colours(board: chess.Board, text: str) -> list[str]:
    """Targets are the enemy's; the pinned piece and its shield share a colour, the pinner
    is the other's."""
    out = []

    def colour(sq: chess.Square) -> bool | None:
        piece = board.piece_at(sq)
        return piece.color if piece else None

    named = [chess.parse_square(s) for _, s in PIECE.findall(text)]
    if any(word in text for word in ATTACKS) and len(named) >= 2:
        attacker = colour(named[0])
        for sq in named[1:]:
            if colour(sq) is not None and colour(sq) == attacker:
                out.append(f"target {chess.square_name(sq)} is the attacker's own colour")
    m = PIN.search(text)
    if m:
        a, b, c = (chess.parse_square(s) for s in m.groups())
        if colour(a) != colour(b) or colour(c) == colour(a):
            out.append("pin colours wrong")
    return out


def check_transcript(lines: list[str], facts: dict) -> list[str]:
    """The problems in one transcript, each with the ply it was said at. The boards come
    from the facts (a recorded game's moves are not one straight line: takebacks)."""
    moves = [s for s in facts["script"] if s["kind"] == "move"]
    problems: list[str] = []
    board = chess.Board(facts["start_fen"])
    before: chess.Board | None = None
    ply = 0
    last_coach: str | None = None
    last_played: str | None = None
    for line in lines:
        m = MOVE_LINE.match(line)
        if m:
            board = chess.Board(moves[ply]["fen_before"]) if ply < len(moves) else board
            before = board.copy()
            board.push_san(m.group(3))
            ply += 1
            last_played = m.group(3) if m.group(4) == "the player" else None
            continue
        c = COACH_LINE.match(line)
        if not c:
            continue
        said = c.group(2)
        found: list[str] = []
        found += [f"placeholder {bad!r}" for bad in PLACEHOLDERS if bad in said]
        found += _pieces(board, before if c.group(1) == "after the move" else None, said)
        found += _colours(board, said)
        after_move = c.group(1) == "after the move"
        lesson = re.match(
            r"^(?:That one cost you: |That cost you something: |Not that one -- )(\S+)", said
        )
        if lesson and lesson.group(1) == last_played:
            found.append("the played move named as the better one")
        fine = re.match(r"^(?:Fine, but |That works, though |Playable -- )(\S+)", said)
        if fine and fine.group(1) == last_played:
            found.append("fine/tighter names the same move")
        if after_move and last_played and said.startswith(last_played):
            found.append("opens with the move just played")
        plain = PLAIN.search(said.lower()) if after_move else None
        if plain:
            found.append(f"says the move's {plain.group(0)!r} back")
        if said == last_coach:
            found.append("repeated verbatim")
        problems += [f"ply {ply}: {what}: {said!r}" for what in found]
        last_coach = said
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", action="append", help="only these facts stems")
    ap.add_argument("--recordings", action="store_true", help="the recorded games instead")
    ap.add_argument("--side", choices=["white", "black"], help="one side only")
    ap.add_argument("--show", type=int, default=40, help="problems to print")
    ap.add_argument("--transcripts", type=Path, help="keep every transcript here")
    args = ap.parse_args()

    folder = FACTS.parent / "recordings" if args.recordings else FACTS
    games = args.game or sorted(p.stem for p in folder.glob("*.json"))
    sides = [args.side] if args.side else ["white", "black"]
    if args.transcripts:
        args.transcripts.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    problems: list[str] = []
    kinds: Counter[str] = Counter()
    sentences = mentions = 0
    for game in games:
        facts = load_facts(game)
        for side in sides:
            if args.recordings and facts.get("user_side") not in (None, side):
                continue
            transcript, _, _ = play(game, side, "none")
            if args.transcripts:
                (args.transcripts / f"{game}-{side}.md").write_text(
                    "\n".join(transcript) + "\n", encoding="utf-8"
                )
            for line in transcript:
                c = COACH_LINE.match(line)
                if c:
                    sentences += 1
                    mentions += len(PIECE.findall(c.group(2)))
            for problem in check_transcript(transcript, facts):
                problems.append(f"{game} ({side}) {problem}")
                kinds[problem.split(": ", 2)[1]] += 1
    print(
        f"{len(games)} games, {sentences} sentences, {mentions} pieces named, "
        f"{len(problems)} problems, {time.time() - t0:.0f} s"
    )
    for kind, n in kinds.most_common():
        print(f"  {n:5d}  {kind}")
    for problem in problems[: args.show]:
        print(problem)
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
