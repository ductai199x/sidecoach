"""Record what the desktop's annotator makes of real games, with its tagger and its book.

The annotator is the layer between the engine and the coach: the loss of a move, its label
(a sacrifice makes one brilliant, the book makes one theory), what the move did and what the
position holds (the motif tags), and the opening the game is in. The phone's annotator was
ported with neither collaborator; this holds it, now with both, to the Python.

The engine is taken out of the comparison: every call the desktop's annotator makes is
answered by the real Stockfish and recorded -- position, number of lines, whether the best
line is carried on to MIN_BEST_PLIES (`carry_best`, asked for the whole measurement only),
and the lines it gave, per game and in the order asked, since a warm engine can answer the
same position differently the second time -- and the Kotlin test answers the same calls from
the recording, in the same order. What is compared is everything the annotator does with
them, and a call the phone makes with another `carry_best` finds no answer.

The games: the thirty plies the emulator plays (the Italian with the h7 sacrifice), Legal's
mate (a queen given up for mate), a Queen's Gambit reached by transposition, and knights
that shuffle into a threefold repetition -- the draw only a board with its history can see.
Each is played on one board, move after move, so the annotator sees the moves that led to
every position, as the live path now gives them.

    cd D:/chess-coaching-companion
    uv run --no-sync python android/tools/make_annotator_fixtures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis import motifs  # noqa: E402
from sidecoach.analysis.annotator import GameAnnotator  # noqa: E402
from sidecoach.analysis.engine import Engine  # noqa: E402
from sidecoach.analysis.openings import OpeningBook  # noqa: E402
from sidecoach.config import find_stockfish, load_settings  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
DEPTH = 12
MULTIPV = 4

GAMES = {
    "italian": (
        "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d4 exd4 cxd4 Bb4+ Nc3 Nxe4 O-O Bxc3 d5 Bf6 Re1 Ne7 "
        "Rxe4 d6 Bg5 Bxg5 Nxg5 O-O Nxh7 Kxh7 Qh5+ Kg8"
    ),
    "legal": "e4 e5 Nf3 d6 Bc4 Bg4 Nc3 g6 Nxe5 Bxd1 Bxf7+ Ke7 Nd5#",
    "transposed": "d4 Nf6 c4 e6 Nc3 d5 Bg5 Be7 e3 O-O Nf3 h6",
    "repetition": "Nf3 Nf6 Ng1 Ng8 Nf3 Nf6 Ng1 Ng8 e4",
}


def line_json(line) -> dict:
    return {
        "moves": list(line.moves),
        "depth": line.depth,
        "expected_score": line.expected_score,
        "score_cp": line.score_cp,
        "mate": line.mate,
        "wdl": list(line.wdl) if line.wdl is not None else None,
    }


def tag_json(tag) -> dict:
    return {"name": tag.name, "squares": list(tag.squares), "move": tag.move}


class Recording(Engine):
    """The real engine, every answer kept under the position and the number of lines."""

    def __init__(self, real: Engine) -> None:
        super().__init__(path=Path("recording"))
        self.real = real
        self.answers: dict[str, list[list[dict]]] = {}

    def analyse(self, board: chess.Board, **kwargs: object) -> list:
        lines = self.real.analyse(board, **kwargs)  # type: ignore[arg-type]
        carry = str(bool(kwargs.get("carry_best", False))).lower()
        key = f"{board.fen()}|{kwargs.get('multipv')}|{carry}"
        self.answers.setdefault(key, []).append([line_json(x) for x in lines])
        return lines

    def open(self) -> None:
        return

    def close(self) -> None:
        return


def main() -> None:
    path = find_stockfish(load_settings())
    assert path is not None, "no Stockfish for the desktop"
    book = OpeningBook.load(DESKTOP / "data" / "openings")
    assert book is not None, "no opening tables"
    games = []
    # One thread: two make the search, and so the fixture, differ from run to run.
    with Engine(path=path, threads=1, hash_mb=64, multipv=MULTIPV) as real:
        for name, sans in GAMES.items():
            recording = Recording(real)
            annotator = GameAnnotator(
                engine=recording, depth=DEPTH, multipv=MULTIPV, tagger=motifs, book=book
            )
            board = chess.Board()
            plies = []
            for ply, san in enumerate(sans.split(), start=1):
                move = board.parse_san(san)
                assessment, analysis = annotator.assess(name, board, move, ply)
                board.push(move)
                plies.append({
                    "uci": move.uci(),
                    "label": assessment.label.value if assessment.label else None,
                    "severity": assessment.severity.value,
                    "expected_loss": assessment.expected_loss,
                    "best": list(assessment.best),
                    "move_tags": [tag_json(t) for t in assessment.tags],
                    "depth": assessment.depth,
                    "accuracy": assessment.accuracy,
                    "position_tags": [tag_json(t) for t in analysis.tags],
                    "opening": analysis.opening,
                    "decisive": analysis.decisive,
                    "only_move": analysis.only_move,
                    "threats": list(analysis.threats),
                    # The captures that look free and are not (desktop fc30fa7), and whether the
                    # game is still in the book there.
                    "traps": [line_json(t) for t in analysis.traps],
                    # The recaptures the best line declines (desktop a1f7f81).
                    "defences": [line_json(t) for t in analysis.defences],
                    # What a quiet best move threatens (desktop 29ae423).
                    "threatens": [line_json(t) for t in analysis.threatens],
                    "in_book": analysis.in_book,
                })
            games.append({"name": name, "plies": plies, "answers": recording.answers})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "annotator.json").write_text(
        json.dumps({"depth": DEPTH, "multipv": MULTIPV, "games": games}),
        encoding="utf-8",
    )
    labels = sorted({p["label"] for g in games for p in g["plies"] if p["label"]})
    tags = sorted({t["name"] for g in games for p in g["plies"] for t in p["move_tags"]})
    calls = sum(len(a) for g in games for a in g["answers"].values())
    traps = sum(len(p["traps"]) for g in games for p in g["plies"])
    defences = sum(len(p["defences"]) for g in games for p in g["plies"])
    threatens = sum(len(p["threatens"]) for g in games for p in g["plies"])
    booked = sum(1 for g in games for p in g["plies"] if p["in_book"])
    print(f"{sum(len(g['plies']) for g in games)} plies, {calls} engine answers; "
          f"labels {labels}; move tags {tags}; {traps} traps, {defences} defences, "
          f"{threatens} threats of quiet moves, "
          f"{booked} plies in book")
    for g in games:
        print(g["name"], [p["opening"] for p in g["plies"]][-1], [p["label"] for p in g["plies"]])


if __name__ == "__main__":
    main()
