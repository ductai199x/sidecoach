"""The profile stage: the events of a game in, the record, the evidence and the review
out (DESIGN sections 5.6 and 5.7).

It keeps its own account of each game from the same events the coach reads -- the
moves, the engine's verdicts, the analyses of the positions they lead to, the clocks --
and writes everything to the store as it happens. A judged move of the player's counts
as a chance in its phase and for every tactic that was on the board; a mistake or a
blunder is placed on the two axes (causes.py) once its assessment and the analysis of
the position it left have both arrived, and becomes evidence. When the game ends the
review is built from the record (review.py) and the snapshot of the whole profile is
refreshed, both as events for the coach and the panel -- once the engine has caught up
with the last moves (it trails a fast finish by a few positions), or after a grace
period, or when the session closes, whichever comes first.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import chess

from sidecoach.analysis.phase import phase_of
from sidecoach.events import (
    Analysis,
    ClockObservation,
    CoachMessage,
    Event,
    GameEnded,
    GameReview,
    GameStarted,
    Label,
    MoveAssessment,
    MovePlayed,
    ProfileEvidence,
    ProfileSnapshot,
    Resync,
    Side,
    Takeback,
    UserUtterance,
)
from sidecoach.profile import causes
from sidecoach.profile.insights import as_cells, insights
from sidecoach.profile.review import ERRORS, UserMove, build_review
from sidecoach.profile.store import Store

# A finished game waits this long for the engine's last verdicts before its review is
# built from what is there.
REVIEW_GRACE_S = 90.0


@dataclass
class _Ply:
    move: MovePlayed
    assessment: MoveAssessment | None = None
    clock_ms: int | None = None  # the mover's clock when the move was made
    phase: str = "middlegame"
    cause: str | None = None
    motif: str | None = None
    placed: bool = False  # the evidence written, the cells bumped


@dataclass
class _Game:
    game_id: str
    user_side: Side
    start_fen: str
    started_at: float
    segment: int = 0
    plies: dict[tuple[int, int], _Ply] = field(default_factory=dict)
    # The analysis of the position after (segment, ply); ply 0 is the segment's base.
    analyses: dict[tuple[int, int], Analysis] = field(default_factory=dict)
    opening: str | None = None
    last_fen: str = chess.STARTING_FEN
    ended: bool = False
    # The game is over; its review waits for the last verdicts (see `_maybe_review`).
    end: GameEnded | None = None
    ended_at: float = 0.0
    reviewed: bool = False


def _reason_words(event: GameEnded) -> str | None:
    """The reason as the review says it: the reconciler's "game over on board" is a mate,
    a stalemate or dead material, which the final position tells."""
    if event.reason != "game over on board" or not event.final_fen:
        return event.reason
    board = chess.Board(event.final_fen)
    if board.is_checkmate():
        return "checkmate"
    if board.is_stalemate():
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    return event.reason


class Profiler:
    def __init__(self, store: Store, *, session_id: str | None = None, ts: float = 0.0) -> None:
        self.store = store
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.store.start_session(self.session_id, ts)
        self._game: _Game | None = None
        self._clock: tuple[int | None, int | None] = (None, None)
        self._reviews: list[GameReview] = []

    # ---- events ------------------------------------------------------------------------

    def on_event(self, event: Event, ts: float) -> list[Event]:
        """Consume one event; the events the profile emits in return, if any."""
        if isinstance(event, GameStarted):
            return [*self.flush(ts), *self._start(event, ts)]
        if isinstance(event, ClockObservation):
            self._clock = (event.white_ms, event.black_ms)
            return []
        game = self._game
        if game is None:
            return []
        if game.ended and isinstance(event, MovePlayed | Takeback | Resync | GameEnded):
            return []
        if isinstance(event, MovePlayed):
            return self._move(game, event, ts)
        if isinstance(event, Analysis):
            return [*self._analysis(game, event), *self._maybe_review(game, ts)]
        if isinstance(event, MoveAssessment):
            return [*self._assessment(game, event), *self._maybe_review(game, ts)]
        if isinstance(event, Takeback):
            return self._takeback(game, event)
        if isinstance(event, Resync):
            game.segment += 1
            game.last_fen = event.fen
            return []
        if isinstance(event, CoachMessage):
            self.store.add_dialogue(
                game.game_id, ts, "coach", event.trigger.value, event.text, event.ply
            )
            return []
        if isinstance(event, UserUtterance):
            self.store.add_dialogue(game.game_id, ts, "user", event.source, event.text, None)
            return []
        if isinstance(event, GameEnded):
            return self._end(game, event, ts)
        return []

    def _start(self, event: GameStarted, ts: float) -> list[Event]:
        self._game = _Game(
            game_id=event.game_id,
            user_side=event.user_side,
            start_fen=event.start_fen,
            started_at=ts,
            last_fen=event.start_fen,
        )
        self.store.start_game(
            event.game_id, ts, event.user_side.value, event.start_fen, self.session_id
        )
        return []

    def _move(self, game: _Game, event: MovePlayed, ts: float) -> list[Event]:
        key = (game.segment, event.ply)
        ply = _Ply(move=event)
        mover = chess.Board(event.fen_before).turn
        ply.clock_ms = self._clock[0] if mover == chess.WHITE else self._clock[1]
        game.plies[key] = ply
        game.last_fen = event.fen_after
        self.store.add_move(
            game.game_id,
            game.segment,
            event.ply,
            event.san,
            event.uci,
            by_user=event.by_user,
            fen_before=event.fen_before,
            fen_after=event.fen_after,
            think_ms=event.think_ms,
            ts=ts,
        )
        return []

    def _analysis(self, game: _Game, event: Analysis) -> list[Event]:
        game.analyses[(game.segment, event.ply)] = event
        if event.opening:
            game.opening = event.opening
        return self._place(game, (game.segment, event.ply))

    def _assessment(self, game: _Game, event: MoveAssessment) -> list[Event]:
        key = (game.segment, event.ply)
        ply = game.plies.get(key)
        if ply is None:
            return []
        ply.assessment = event
        before = chess.Board(ply.move.fen_before)
        ply.phase = phase_of(before, in_book=event.label is Label.BOOK)
        self.store.assess_move(
            game.game_id,
            game.segment,
            event.ply,
            label=event.label.value if event.label else None,
            severity=event.severity.value,
            expected_loss=event.expected_loss,
            best_uci=event.best[0] if event.best else None,
            accuracy=event.accuracy,
            tags=[t.name for t in event.tags],
            depth=event.depth,
        )
        if not ply.move.by_user:
            return []
        # A judged move of the player's is a chance: in its phase, and for every tactic that
        # was on the board for them.
        self.store.bump_cell(None, None, ply.phase, chances=1)
        faced = game.analyses.get((game.segment, event.ply - 1))
        if faced is not None:
            for name in {t.name for t in faced.tags if t.name in causes.TACTICAL}:
                self.store.bump_cell(None, name, ply.phase, chances=1)
        return self._place(game, key)

    def _place(self, game: _Game, key: tuple[int, int]) -> list[Event]:
        """Place a mistake on the two axes once its assessment and the analysis of the
        position it led to are both in."""
        ply = game.plies.get(key)
        if ply is None or ply.placed or ply.assessment is None or not ply.move.by_user:
            return []
        if ply.assessment.severity.value not in ERRORS:
            return []
        after = game.analyses.get(key)
        if after is None:
            return []
        return [self._evidence(game, key, ply, after)]

    def _evidence(
        self, game: _Game, key: tuple[int, int], ply: _Ply, after: Analysis
    ) -> ProfileEvidence:
        assert ply.assessment is not None
        segment, number = key
        before = chess.Board(ply.move.fen_before)
        faced = game.analyses.get((segment, number - 1))
        previous = game.plies.get((segment, number - 1))
        left_book = (
            previous is not None
            and previous.assessment is not None
            and previous.assessment.label is Label.BOOK
            and ply.assessment.label is not Label.BOOK
        )
        reply = after.lines[0].moves[0] if after.lines and after.lines[0].moves else None
        cause, motif = causes.classify(
            before,
            chess.Move.from_uci(ply.move.uci),
            played_tags=ply.assessment.tags,
            available_tags=faced.tags if faced is not None else (),
            allowed_tags=after.tags,
            reply=reply,
            reply_mate=after.lines[0].mate if after.lines else None,
            phase=ply.phase,
            left_book=left_book,
            clock_ms=ply.clock_ms,
        )
        ply.cause, ply.motif, ply.placed = cause, motif, True
        loss = ply.assessment.expected_loss
        self.store.add_evidence(
            game.game_id,
            segment,
            number,
            phase=ply.phase,
            expected_loss=loss,
            cause=cause,
            motif=motif,
            time_pressure=cause == causes.TIME_PRESSURE,
        )
        self.store.bump_cell(cause, motif, ply.phase, errors=1, loss=loss)
        return ProfileEvidence(
            game_id=game.game_id,
            ply=number,
            phase=ply.phase,
            expected_loss=loss,
            motif=motif,
            cause=cause,
            time_pressure=cause == causes.TIME_PRESSURE,
        )

    def _takeback(self, game: _Game, event: Takeback) -> list[Event]:
        plies = sorted(k for k in game.plies if k[0] == game.segment)
        if not plies:
            return []
        keep = max(0, len(plies) - event.plies)
        from_ply = plies[keep][1] if keep < len(plies) else plies[-1][1] + 1
        for key in plies[keep:]:
            gone = game.plies.pop(key)
            if gone.placed and gone.assessment is not None:
                self.store.bump_cell(
                    gone.cause,
                    gone.motif,
                    gone.phase,
                    errors=-1,
                    loss=-gone.assessment.expected_loss,
                )
            game.analyses.pop(key, None)
        self.store.withdraw_moves(game.game_id, game.segment, from_ply)
        game.last_fen = event.fen
        return []

    def _end(self, game: _Game, event: GameEnded, ts: float) -> list[Event]:
        game.ended = True
        game.end = event
        game.ended_at = ts
        self.store.end_game(
            game.game_id,
            ts,
            event.result,
            event.reason,
            pgn=event.pgn,
            final_fen=event.final_fen,
            opening=game.opening,
        )
        return self._maybe_review(game, ts)

    def _maybe_review(self, game: _Game, ts: float, *, force: bool = False) -> list[Event]:
        """The review of a finished game, once every move of the player's has its verdict
        or the grace period is up; `force` builds it from what is there."""
        if game.end is None or game.reviewed:
            return []
        waiting = any(p.assessment is None for p in game.plies.values() if p.move.by_user)
        if waiting and not force and ts - game.ended_at < REVIEW_GRACE_S:
            return []
        game.reviewed = True
        review = self.review(game, game.end.result, _reason_words(game.end))
        self.store.end_game(
            game.game_id,
            game.ended_at,
            game.end.result,
            game.end.reason,
            pgn=game.end.pgn,
            final_fen=game.end.final_fen,
            accuracy=review.accuracy,
            opening=game.opening,
        )
        self._reviews.append(review)
        return [review, self.snapshot()]

    def tick(self, ts: float) -> list[Event]:
        """Time passing: a finished game whose verdicts never came is reviewed as it is."""
        game = self._game
        if game is None:
            return []
        return self._maybe_review(game, ts)

    def flush(self, ts: float) -> list[Event]:
        """Review a finished game now, from what is there: the session is closing, or the
        next game is starting."""
        game = self._game
        if game is None:
            return []
        return self._maybe_review(game, ts, force=True)

    # ---- the review and the snapshot ---------------------------------------------------

    def user_moves(self, game: _Game) -> list[UserMove]:
        out: list[UserMove] = []
        for (segment, number), ply in sorted(game.plies.items()):
            if not ply.move.by_user or ply.assessment is None:
                continue
            a = ply.assessment
            out.append(
                UserMove(
                    ply=number,
                    segment=segment,
                    san=ply.move.san,
                    uci=ply.move.uci,
                    fen_before=ply.move.fen_before,
                    label=a.label.value if a.label else None,
                    severity=a.severity.value,
                    expected_loss=a.expected_loss,
                    best_uci=a.best[0] if a.best else None,
                    accuracy=a.accuracy,
                    phase=ply.phase,
                    cause=ply.cause,
                    motif=ply.motif,
                )
            )
        return out

    def review(self, game: _Game, result: str, reason: str | None) -> GameReview:
        return build_review(
            game_id=game.game_id,
            user_side=game.user_side,
            result=result,
            reason=reason,
            opening=game.opening,
            moves=self.user_moves(game),
            fullmoves=chess.Board(game.last_fen).fullmove_number,
        )

    def snapshot(self) -> ProfileSnapshot:
        rows = self.store.cells()
        games = self.store.games_count()
        last = self.store.last_session(before=self.session_id)
        return ProfileSnapshot(
            games=games,
            cells=as_cells(rows),
            insights=insights(rows, games),
            last_session=last.summary if last is not None else None,
        )

    def correct(self, game_id: str, segment: int, ply: int, cause: str | None) -> bool:
        """The user's word on why a mistake happened. Returns whether there was one."""
        row = self.store.correct_evidence(game_id, segment, ply, cause)
        if row is None:
            return False
        game = self._game
        if game is not None and game.game_id == game_id:
            item = game.plies.get((segment, ply))
            if item is not None:
                item.cause = cause
        return True

    def self_report(self, text: str) -> None:
        """What the player said about their last mistake, kept beside it."""
        game = self._game
        if game is None:
            return
        placed = [k for k, p in game.plies.items() if p.placed]
        if not placed:
            return
        segment, ply = max(placed)
        self.store.self_report(game.game_id, segment, ply, text)

    # ---- the session -------------------------------------------------------------------

    def end_session(self, ts: float) -> str | None:
        """Close the session with a summary of its games, for the next session's coach. A
        game still waiting for its review is reviewed as it stands first."""
        self.flush(ts)
        reviews = self._reviews
        summary: str | None = None
        if reviews:
            outcomes = Counter()
            for r in reviews:
                won = (r.result == "1-0") == (r.user_side is Side.WHITE)
                outcomes["drew" if r.result == "1/2-1/2" else "won" if won else "lost"] += 1
            parts = [f"{n} {k}" for k, n in outcomes.items()]
            accs = [r.accuracy for r in reviews if r.accuracy is not None]
            text = f"{len(reviews)} game{'s' if len(reviews) != 1 else ''} ({', '.join(parts)})"
            if accs:
                text += f", accuracy {sum(accs) / len(accs):.0f}"
            costs: dict[str, float] = defaultdict(float)
            for r in reviews:
                for cause, loss in r.causes.items():
                    costs[cause] += loss
            if costs:
                cause, loss = max(costs.items(), key=lambda kv: kv[1])
                text += f"; most ground went to {cause} ({loss:.2f})"
            worst = max(
                (tp for r in reviews for tp in r.turning_points),
                key=lambda tp: tp.expected_loss,
                default=None,
            )
            if worst is not None:
                better = f", {worst.best_san} was better" if worst.best_san else ""
                text += f"; the worst moment was {worst.san}{better}"
            summary = text + "."
        self.store.end_session(self.session_id, ts, summary)
        return summary
