"""What a reviewed game says for Stats (docs/stats-plan.md): each move's facts and the game's,
read off the review once and kept, so the Stats window adds them up at once.

The definitions are ours, as chess.com publishes none, and each is shown on the page:

- A **tactic chance**: the mover's position had one move clearly best (the review's decisive
  call: the best line beats the next by a tenth of a game or more), its line a tactic (the
  tagger's motif on it, dealt within the mover's first two moves), and leaves the mover at least
  level; a mate counts only when it is in three or fewer, so no mate in 18 is "missed". Found when
  the move played cost under 0.02 of a game (Lichess's tolerance for "the same move").
- **Allowed**: the chance the move handed the opponent, by the same rule, in the position after.
- **Kinds**, chess.com's nine: checkmate, fork, pin, discovery, trapped piece, removing the
  defender, free piece, sacrifice, other.
- **Structure** of the position the move was played from: open (a pawn or none on the d- and
  e-files), closed (a d- or e-pawn blocked head on by an enemy pawn), else semi-open.
- **Themes** of strategy there, from the positional layer (`analysis.positional`): *space*
  when the mover has more of it, *activity* when a rook or queen of theirs stands on an open
  file or a minor piece on an outpost, *structure* when either side has a weak pawn (isolated,
  backward, doubled, hanging, the isolated queen's pawn).
- **Endgames** where the review's phases put them (`analysis.phase`), by what is left besides
  kings and pawns: pawn, rook, queen, other; entered better, equal or worse by the mover's
  expected score then (0.6 and 0.4).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import chess

from sidecoach.analysis import positional
from sidecoach.analysis.phase import game_phases
from sidecoach.analysis.score import position_score
from sidecoach.events import Analysis, MotifTag, ReviewedGame
from sidecoach.profile.causes import TACTICAL
from sidecoach.profile.store import LibraryRow, Store

KINDS = (
    "checkmate",
    "fork",
    "pin",
    "discovery",
    "trapped piece",
    "removing the defender",
    "free piece",
    "sacrifice",
    "other",
)
_KIND_OF = {
    "fork": "fork",
    "pin": "pin",
    "absolutePin": "pin",
    "discoveredAttack": "discovery",
    "doubleCheck": "discovery",
    "trappedPiece": "trapped piece",
    "capturingDefender": "removing the defender",
    "hangingPiece": "free piece",
    "sacrifice": "sacrifice",
}
FOUND_LOSS = 0.02
MATE_FINDABLE = 3
BLOW_PLIES = 3  # the line's first plies: the mover's first two moves are its first and third
# The tactics: the tagger's (`causes.TACTICAL`), and a double check, a discovery at its sharpest.
TACTICS = TACTICAL | {"doubleCheck"}
BETTER, WORSE = 0.6, 0.4
ENDGAMES = ("pawn", "rook", "queen", "other")
THEMES = ("space", "activity", "structure")
STRUCTURES = ("open", "semi-open", "closed")


@dataclass(frozen=True)
class MoveFact:
    ply: int
    side: str
    fen: str  # the position the move was played from
    uci: str
    san: str
    phase: str
    loss: float
    label: str | None
    accuracy: float | None
    score: float  # the mover's expected score before the move
    chance: str | None  # the tactic the mover had, by kind
    found: bool
    allowed: str | None  # the tactic the move handed the opponent, by kind
    structure: str
    themes: str  # the strategy themes there, comma-separated
    endgame: str | None
    spent_ms: int | None
    clock_ms: int | None


@dataclass(frozen=True)
class GameFact:
    key: str
    source: str
    time_class: str | None
    played_at: float | None
    user_side: str | None
    bot: bool
    result: str | None
    opening: str | None
    termination: str | None


def kind_of(tags: Sequence[MotifTag]) -> str | None:
    """A tactic's kind from the motifs of its line: a mate first, then the named patterns, a free
    piece only when nothing more precise is there."""
    names = {t.name for t in tags if t.name in TACTICS}
    if not names:
        return None
    if any(n == "mate" or n.startswith("mateIn") or n.endswith("Mate") for n in names):
        return "checkmate"
    for name in ("fork", "pin", "absolutePin", "discoveredAttack", "doubleCheck", "trappedPiece"):
        if name in names:
            return _KIND_OF[name]
    for name in ("capturingDefender", "hangingPiece", "sacrifice"):
        if name in names:
            return _KIND_OF[name]
    return "other"


def chance_in(analysis: Analysis) -> str | None:
    """The tactic the side to move had in the analysed position, by kind, or None."""
    if not analysis.decisive or not analysis.lines or not analysis.lines[0].moves:
        return None
    best = analysis.lines[0]
    board = chess.Board(analysis.fen)
    if position_score(board, analysis.lines) < 0.5:
        return None
    mover = best.moves[:BLOW_PLIES:2]
    tags = [t for t in analysis.tags if t.move is None or t.move in mover]
    kind = kind_of(tags)
    if kind == "checkmate" and (best.mate is None or not 0 < best.mate <= MATE_FINDABLE):
        return None
    return kind


def structure_of(board: chess.Board) -> str:
    """Open, semi-open or closed, from the centre pawns."""
    pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
    centre = [s for s in pawns if chess.square_file(s) in (3, 4)]
    if len(centre) <= 1:
        return "open"
    for square in board.pieces(chess.PAWN, chess.WHITE):
        if chess.square_file(square) in (3, 4) and chess.square_rank(square) < 7:
            ahead = board.piece_at(square + 8)
            if ahead is not None and ahead.piece_type == chess.PAWN and ahead.color == chess.BLACK:
                return "closed"
    return "semi-open"


def themes_of(board: chess.Board) -> list[str]:
    """The strategy themes of the position for the side to move (`THEMES`)."""
    us = board.turn
    out = []
    if any(f.kind == "spaceAdvantage" for f in positional.space_plan(board, us)):
        out.append("space")
    if positional.open_files(board, us) or positional.outposts(board, us):
        out.append("activity")
    if (
        positional.pawn_weaknesses(board, us)
        or positional.hanging_pawns(board, us)
        or positional.hanging_pawns(board, not us)
        or positional.isolated_d_pawn(board, us)
    ):
        out.append("structure")
    return out


def endgame_of(board: chess.Board) -> str:
    """An endgame's kind by what is left besides kings and pawns."""
    left = {p.piece_type for p in board.piece_map().values()} - {chess.KING, chess.PAWN}
    if not left:
        return "pawn"
    if left == {chess.ROOK}:
        return "rook"
    if left == {chess.QUEEN}:
        return "queen"
    return "other"


def standing(score: float) -> str:
    return "better" if score >= BETTER else "worse" if score <= WORSE else "equal"


def bot_game(headers: dict[str, str]) -> bool:
    """Whether a side was a computer: chess.com's bot and coach games, Lichess's BOT title, a
    name marked as a bot."""
    event = headers.get("Event", "").lower()
    if any(word in event for word in ("vs bot", "vs coach", "vs computer")):
        return True
    for colour in ("White", "Black"):
        name = headers.get(colour, "")
        if headers.get(f"{colour}Title", "").upper() == "BOT" or name.upper().endswith("-BOT"):
            return True
        if name.startswith("Coach-"):
            return True
    return False


def termination_kind(text: str | None) -> str:
    """How a game ended, from chess.com's and Lichess's Termination header."""
    t = (text or "").lower()
    for word, kind in (
        ("checkmate", "checkmate"),
        ("resign", "resignation"),
        ("on time", "time"),
        ("time forfeit", "time"),
        ("timeout", "time"),
        ("abandon", "abandoned"),
        ("agreement", "agreement"),
        ("repetition", "repetition"),
        ("stalemate", "stalemate"),
        ("insufficient", "insufficient material"),
        ("50", "fifty moves"),
    ):
        if word in t:
            return kind
    return "other"


def ended_by(text: str | None, result: str | None, game: ReviewedGame) -> str | None:
    """How a game ended, in words `termination_kind` reads: its Termination header when that
    says, else what the final position proves (a mate, a stalemate, dead material, a repetition,
    fifty moves), else, for Lichess's "Normal" (a game neither lost on time nor abandoned), a
    resignation or a draw agreed. Lichess's "Normal" and a watched game's missing header made
    most of "How your games ended" "other" (the review of 2026-10-01)."""
    if termination_kind(text) != "other":
        return text
    board = chess.Board(game.start.fen)
    try:
        for move in game.moves:
            board.push_uci(move.assessment.uci)
    except ValueError:
        return text
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if result == "1/2-1/2":
        if board.is_insufficient_material():
            return "insufficient material"
        if board.is_repetition(3):  # the sites draw it as the position comes the third time
            return "repetition"
        if board.halfmove_clock >= 100:
            return "50 moves"
    if (text or "").strip().lower() == "normal":
        if result in ("1-0", "0-1"):
            return "resignation"
        if result == "1/2-1/2":
            return "agreement"
    return text


def facts_of(game: ReviewedGame, row: LibraryRow) -> tuple[GameFact, list[MoveFact]]:
    """A reviewed game's facts: the game's, and every move's."""
    moves: list[MoveFact] = []
    before = game.start
    # The game divided as Lichess divides it (`phase.game_phases`).
    phases = game_phases(chess.Board(move.fen_before) for move in game.moves)
    for move, phase in zip(game.moves, phases, strict=True):
        board = chess.Board(move.fen_before)
        score = position_score(board, before.lines)
        chance = chance_in(before)
        label = move.assessment.label
        moves.append(
            MoveFact(
                ply=move.ply,
                side=move.side.value,
                fen=move.fen_before,
                uci=move.assessment.uci,
                san=move.san,
                phase=phase,
                loss=round(move.assessment.expected_loss, 4),
                label=label.value if label is not None else None,
                accuracy=move.assessment.accuracy,
                score=round(score, 4),
                chance=chance,
                found=chance is not None and move.assessment.expected_loss < FOUND_LOSS,
                allowed=chance_in(move.after),
                structure=structure_of(board),
                themes=",".join(themes_of(board)),
                endgame=endgame_of(board) if phase == "endgame" else None,
                spent_ms=move.spent_ms,
                clock_ms=move.clock_ms,
            )
        )
        before = move.after
    opening = next((m.after.opening for m in reversed(game.moves) if m.after.opening), None)
    headers = game.headers
    result = headers.get("Result") or row.result
    fact = GameFact(
        key=row.key,
        source=row.source,
        time_class=row.time_class,
        played_at=row.played_at,
        user_side=row.user_side,
        bot=bot_game(headers),
        result=result,
        opening=opening or game.start.opening,
        termination=ended_by(headers.get("Termination") or _termination(row), result, game),
    )
    return fact, moves


def _termination(row: LibraryRow) -> str | None:
    """The Termination header of the library's copy of a game: a watched game's says how
    Sidecoach saw it end ("lost on time", "checkmate")."""
    found = re.search(r'^\[Termination "([^"]*)"\]', row.pgn, re.M)
    return found.group(1) if found else None


def keep_stats(store: Store, strength: str, reviewed: ReviewedGame, row: LibraryRow) -> None:
    """A reviewed game's facts kept for Stats, in place of any from an older review."""
    game, moves = facts_of(reviewed, row)
    store.save_stats(strength, asdict(game), [asdict(m) for m in moves])
