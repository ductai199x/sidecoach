"""Bot games: two Nova players, each with a rating, a style and a temperature, playing each
other from the start, saved as PGN (`data/bots/games/`) for the coach to be run on -- the
amateur games the benches of puzzles and master games do not have.

Each move is drawn from Nova's distribution for that player (`nova.NovaPolicy`), sharpened or
flattened by the side's temperature -- by default the one at which Nova's mistakes match real
players' at its rating (`nova.temperature_for`, measured by `calibrate.py`). A game ends by
the rules (mate, stalemate, repetition, the fifty-move rule, insufficient material) or at
MAX_PLIES, as a draw.

    uv run --no-sync python tools/bots/play.py --games 20 --white 1500 --black 1500 --workers 4
    uv run --no-sync python tools/bots/play.py --games 10 --white 1300:0.5:0.9 --black 1600:0.2:0.3

A side is RATING[:CLASSICAL[:AGGRESSION]], on Nova's (Lichess rapid) scale.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import chess  # noqa: E402
import chess.pgn  # noqa: E402
from nova import NovaPolicy, Style, temperature_for  # noqa: E402

OUT = ROOT / "data" / "bots" / "games"
MAX_PLIES = 300

_policy: NovaPolicy | None = None


@dataclass(frozen=True)
class Player:
    style: Style
    temperature: float

    @property
    def name(self) -> str:
        s = self.style
        return f"Nova {s.rating:.0f} c{s.classical:.1f} a{s.aggression:.1f} t{self.temperature:g}"


def parse_side(text: str, temperature: float | None) -> Player:
    """A side from RATING[:CLASSICAL[:AGGRESSION]]; its temperature the calibrated one for
    its rating (`nova.temperature_for`) unless one is given."""
    parts = [float(x) for x in text.split(":")]
    rating = parts[0]
    classical = parts[1] if len(parts) > 1 else 0.5
    aggression = parts[2] if len(parts) > 2 else 0.5
    t = temperature if temperature is not None else temperature_for(rating)
    return Player(Style(rating, classical, aggression), t)


def _init() -> None:
    global _policy
    _policy = NovaPolicy()


def play(job: tuple[int, Player, Player]) -> str:
    """One game, as PGN text."""
    seed, white, black = job
    rng = random.Random(seed)
    assert _policy is not None
    board = chess.Board()
    game = chess.pgn.Game()
    node: chess.pgn.GameNode = game
    while not board.is_game_over(claim_draw=True) and board.ply() < MAX_PLIES:
        player = white if board.turn == chess.WHITE else black
        probs = _policy.probabilities([board], [player.style], temperature=player.temperature)[0]
        move = rng.choices(list(probs), weights=list(probs.values()), k=1)[0]
        node = node.add_variation(move)
        board.push(move)
    outcome = board.outcome(claim_draw=True)
    game.headers.update(
        {
            "Event": "Sidecoach bot game",
            "Site": "local",
            "Date": time.strftime("%Y.%m.%d"),
            "White": white.name,
            "Black": black.name,
            "WhiteElo": f"{white.style.rating:.0f}",
            "BlackElo": f"{black.style.rating:.0f}",
            "Result": outcome.result() if outcome else "1/2-1/2",
            "Termination": outcome.termination.name.lower() if outcome else "max plies",
            "Annotator": f"Nova (novachess.ai) policy, seed {seed}",
        }
    )
    return str(game)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--white", default="1500", help="RATING[:CLASSICAL[:AGGRESSION]]")
    parser.add_argument("--black", default="1500", help="RATING[:CLASSICAL[:AGGRESSION]]")
    parser.add_argument(
        "--temperature", type=float, default=None, help="default: calibrated for each rating"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    white = parse_side(args.white, args.temperature)
    black = parse_side(args.black, args.temperature)
    jobs = [(args.seed * 1_000_003 + i, white, black) for i in range(args.games)]
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.white}-v-{args.black}.pgn".replace(
        ":", "_"
    )
    started = time.time()
    results: dict[str, int] = {}
    with mp.Pool(args.workers, initializer=_init) as pool, path.open("w", encoding="utf-8") as fh:
        for i, pgn in enumerate(pool.imap_unordered(play, jobs), 1):
            fh.write(pgn + "\n\n")
            result = pgn.split('[Result "', 1)[1].split('"', 1)[0]
            results[result] = results.get(result, 0) + 1
            print(f"  {i}/{args.games} {result}  {time.time() - started:.0f} s", flush=True)
    print(f"{white.name} v {black.name}: {results}; {path}")


if __name__ == "__main__":
    main()
