"""Analysis in a reviewed game (docs/game-review-plan.md, screen F): the player's own moves from
any position of the game, the engine's best lines for the position on the board, the moves the
player has played from it in their other games (the explorer), and the game as a PGN -- the
review's verdicts on it, the player's side lines with it -- to copy or keep.

The page holds the tree of moves (the game's, and the player's side lines off it); a move on the
board is checked and named here, and each position the board shows is asked of the engine and
the explorer. The engine is the room's own, made when the analysis first needs it.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterable
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any

import chess
import chess.pgn

from sidecoach.analysis.engine import Engine
from sidecoach.analysis.game_review import Strength
from sidecoach.analysis.score import position_score
from sidecoach.config import Settings, find_stockfish
from sidecoach.profile import library
from sidecoach.profile.store import LibraryRow

# The engine's best lines for a position, and how far each is shown.
LINES = 3
LINE_PLIES = 10
# A review's labels as PGN's move marks, and as the word its comment begins with. A miss has
# no mark of its own: it takes its loss's (`severity`) -- "?" was put on every miss, one that
# lost nothing (a mate let go, still winning) and a blunder-sized one alike.
NAGS = {
    "brilliant": chess.pgn.NAG_BRILLIANT_MOVE,
    "great": chess.pgn.NAG_GOOD_MOVE,
    "inaccuracy": chess.pgn.NAG_DUBIOUS_MOVE,
    "mistake": chess.pgn.NAG_MISTAKE,
    "blunder": chess.pgn.NAG_BLUNDER,
}


def nag(label: str | None, severity: str | None) -> int | None:
    """The PGN mark of a reviewed move: its label's, a miss its loss's, else none."""
    return NAGS.get(severity or "") if label == "miss" else NAGS.get(label or "")


WORDS = {
    "brilliant": "Brilliant",
    "great": "Great move",
    "best": "Best",
    "excellent": "Excellent",
    "good": "Good",
    "book": "Book",
    "inaccuracy": "Inaccuracy",
    "mistake": "Mistake",
    "miss": "Miss",
    "blunder": "Blunder",
}


def legal_of(board: chess.Board) -> dict[str, list[str]]:
    """Every legal move as the board takes them: a square -> the squares its piece may go to."""
    legal: dict[str, list[str]] = {}
    for move in board.legal_moves:
        legal.setdefault(chess.square_name(move.from_square), []).append(
            chess.square_name(move.to_square)
        )
    return {sq: sorted(set(to)) for sq, to in legal.items()}


def position(fen: str) -> dict[str, Any]:
    """A position as the analysis board takes it: whose move, its legal moves, and the result
    when the game is over there."""
    board = chess.Board(fen)
    outcome = board.outcome(claim_draw=False)
    return {
        "fen": board.fen(),
        "side": "white" if board.turn == chess.WHITE else "black",
        "legal": legal_of(board),
        "over": outcome.result() if outcome is not None else None,
    }


def move(fen: str, uci: str) -> dict[str, Any]:
    """A move played on the analysis board: named, and the position it leaves. ValueError when
    it is not legal there."""
    board = chess.Board(fen)
    try:
        played = chess.Move.from_uci(uci)
    except ValueError:
        raise ValueError(f"{uci} is not a move") from None
    if played not in board.legal_moves:
        raise ValueError(f"{uci} is not legal here")
    san = board.san(played)
    board.push(played)
    return {"uci": played.uci(), "san": san, **position(board.fen())}


def steps(board: chess.Board, ucis: Iterable[str]) -> list[dict[str, str]]:
    """A line of moves from `board` as far as it stays legal, each in SAN and UCI with the
    position it leaves."""
    b = board.copy(stack=False)
    out = []
    for uci in ucis:
        try:
            m = chess.Move.from_uci(uci)
        except ValueError:
            break
        if m not in b.legal_moves:
            break
        san = b.san(m)
        b.push(m)
        out.append({"san": san, "uci": uci, "fen": b.fen()})
    return out


def white_eval(board: chess.Board, line: Any) -> dict[str, int | None]:
    """A line's evaluation from White's side: a mate in moves (+ White mates) or centipawns."""
    sign = 1 if board.turn == chess.WHITE else -1
    if line.mate is not None:
        return {"cp": None, "mate": sign * line.mate}
    return {"cp": sign * (line.score_cp or 0), "mate": None}


class Lines:
    """The engine's best lines for the positions the analysis board shows: one engine, made on
    the first position asked of it and closed with the room; one search at a time."""

    def __init__(
        self,
        settings: Settings,
        strength: Strength,
        make_engine: Callable[..., Engine] | None = None,
    ) -> None:
        self.settings = settings
        self.strength = strength
        self._make_engine = make_engine
        self._engine: Engine | None = None
        self._lock = threading.Lock()

    def lines(self, fen: str) -> dict[str, Any]:
        """The best LINES lines from `fen` at the review's depth, each from White's side and
        played out LINE_PLIES moves; none in a finished position."""
        board = chess.Board(fen)
        with self._lock:
            found = self._ready().analyse(board, depth=self.strength.depth, multipv=LINES)
        lines = [
            {"eval": white_eval(board, line), "steps": steps(board, line.moves[:LINE_PLIES])}
            for line in found
        ]
        white = position_score(board, found) if found else None
        return {
            "fen": board.fen(),
            "depth": found[0].depth if found else 0,
            "white_score": (
                round(white if board.turn == chess.WHITE else 1.0 - white, 4)
                if white is not None
                else None
            ),
            "lines": [line for line in lines if line["steps"]],
        }

    def close(self) -> None:
        with self._lock:
            if self._engine is not None:
                self._engine.close()
            self._engine = None

    def _ready(self) -> Engine:
        if self._engine is not None:
            return self._engine
        path = find_stockfish(self.settings)
        if path is None:
            raise RuntimeError("no Stockfish to analyse with")
        engine = self.settings.engine
        self._engine = (self._make_engine or Engine)(
            path=path,
            threads=1,
            hash_mb=min(engine.hash_mb, 128),
            multipv=LINES,
            pv_plies=engine.pv_plies,
            score_model=engine.score_model,
        )
        self._engine.open()
        return self._engine


# ---- the explorer: the player's own games ----


def _key(board: chess.Board) -> str:
    """A position as the explorer knows it, whatever the moves' count: its placement, side to
    move, castling and a capturable en passant square."""
    return board.epd()


class Explorer:
    """The moves played from each position of the player's games -- every game of the library,
    imported or watched -- how often, and how those games went for the player."""

    def __init__(self, rows: Iterable[LibraryRow]) -> None:
        # position -> uci -> {san, games, won, drawn, lost, last}
        self._moves: dict[str, dict[str, dict[str, Any]]] = {}
        for row in rows:
            games = library.games_in(row.pgn)
            if not games:
                continue
            outcome = _outcome(row.result, row.user_side)
            board = games[0].board()
            seen: set[tuple[str, str]] = set()  # a position played twice in a game counts once
            for m in games[0].mainline_moves():
                at, uci = _key(board), m.uci()
                if (at, uci) not in seen:
                    seen.add((at, uci))
                    fresh = {"san": board.san(m), "games": 0, "won": 0, "drawn": 0, "lost": 0}
                    stats = self._moves.setdefault(at, {}).setdefault(uci, {**fresh, "last": None})
                    stats["games"] += 1
                    if outcome:
                        stats[outcome] += 1
                    when = row.played_at or row.added_at
                    if stats["last"] is None or when > stats["last"]["played_at"]:
                        stats["last"] = {"key": row.key, "played_at": when}
                board.push(m)

    def explore(self, fen: str) -> dict[str, Any]:
        """The moves the player's games went on with from `fen`, the most played first."""
        found = self._moves.get(_key(chess.Board(fen)), {})
        moves = sorted(
            ({"uci": uci, **stats} for uci, stats in found.items()),
            key=lambda m: (-m["games"], m["san"]),
        )
        return {"games": sum(m["games"] for m in moves), "moves": moves}


def _outcome(result: str | None, side: str | None) -> str | None:
    """How a game went for the player: won, drawn or lost; None when it is not known."""
    if side not in ("white", "black") or result not in ("1-0", "0-1", "1/2-1/2"):
        return None
    if result == "1/2-1/2":
        return "drawn"
    return "won" if (result == "1-0") == (side == "white") else "lost"


# ---- the game as a PGN ----


def _comment(label: str | None, best_san: str | None, played: str, eval_: Any) -> str:
    """A reviewed move's comment: its kind, the better move when it was not the best, and the
    evaluation after it ([%eval], as Lichess and chess.com write it)."""
    words = []
    if label in WORDS:
        words.append(f"{WORDS[label]}.")
    if best_san and best_san != played and label in ("inaccuracy", "mistake", "miss", "blunder"):
        words.append(f"{best_san} was best.")
    mark = ""
    if eval_ is not None:
        if eval_.get("mate") is not None:
            mark = f"[%eval #{eval_['mate']}]"
        elif eval_.get("cp") is not None:
            mark = f"[%eval {eval_['cp'] / 100:.2f}]"
    return " ".join(w for w in [*words, mark] if w)


def pgn(
    row: LibraryRow,
    moves: list[dict[str, Any]] | None,
    tree: list[dict[str, Any]],
    *,
    annotated: bool = True,
    words: dict[str, Any] | None = None,
) -> str:
    """The game as a PGN: its headers and moves as they came, the player's side lines from the
    analysis board, and -- `annotated`, with a review -- each move's kind as PGN's mark (!!, !,
    ?!, ?, ??) and in words with the evaluation after it, and the AI coach's words on the key
    moves when they were written. `tree`: the analysis board's moves from the start, each
    {"uci", "children"}, the game's own continuation first."""
    games = library.games_in(row.pgn)
    if not games:
        raise ValueError("the game's PGN has no moves")
    game = games[0]
    views = {m["ply"]: m for m in moves or []}
    texts = (words or {}).get("texts", {})

    def annotate(node: chess.pgn.ChildNode, ply: int) -> None:
        view = views.get(ply)
        if not annotated or view is None:
            return
        label = view.get("label")
        mark = nag(label, view.get("severity"))
        if mark is not None:
            node.nags.add(mark)
        said = texts.get(str(ply)) or {}
        parts = [_comment(label, view.get("best_san"), view["san"], view.get("eval"))]
        parts += [said[p] for p in ("why", "better", "lesson") if said.get(p)]
        mine = " ".join(p for p in parts if p)
        node.comment = f"{mine} {node.comment}".strip() if node.comment else mine

    def graft(node: chess.pgn.GameNode, children: list[dict[str, Any]]) -> None:
        """The analysis board's moves from `node`: the game's own continuation stays first, the
        player's lines added as side lines, each line carried on from its node."""
        for child in children:
            try:
                m = chess.Move.from_uci(child["uci"])
            except ValueError:
                continue
            if m not in node.board().legal_moves:
                continue
            on = node.variation(m) if node.has_variation(m) else node.add_variation(m)
            graft(on, child.get("children", []))

    # The game's own moves first, annotated, then the player's lines grafted on.
    node: chess.pgn.GameNode = game
    ply = 0
    while node.variations:
        node = node.variations[0]
        ply += 1
        annotate(node, ply)  # type: ignore[arg-type]
    graft(game, tree)
    if annotated and moves:
        game.headers["Annotator"] = "Sidecoach"
    return str(game)


def pgn_name(row: LibraryRow, zone: tzinfo | None = None) -> str:
    """A file name for the game's PGN: its players and date (in `zone`, else here), as a file can
    be named."""
    date = datetime.fromtimestamp(row.played_at or row.added_at, zone).strftime("%Y-%m-%d")
    name = f"{row.white or 'White'} vs {row.black or 'Black'} {date}"
    return re.sub(r"[^\w .-]+", "_", name).strip() + ".pgn"


def save(text: str, name: str, folder: Path | None = None) -> Path:
    """The PGN kept in the player's Downloads folder (a number added when the name is taken)."""
    folder = folder or Path.home() / "Downloads"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    n = 2
    while path.exists():
        path = folder / f"{Path(name).stem} ({n}).pgn"
        n += 1
    path.write_text(text, encoding="utf-8")
    return path
