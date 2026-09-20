"""The profile stage: the events of a game in, the record, the evidence and the review
out (DESIGN sections 5.6 and 5.7).

It keeps its own account of each game from the same events the coach reads -- the
moves, the engine's verdicts, the analyses of the positions they lead to, the clocks --
and writes everything to the store as it happens, each move with the facts a question
might later need (the phase, the player's clock, the material, the expected score after
it from the player's side, the tactics on the board before and after). A mistake or a
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
from sidecoach.analysis.score import terminal_expected_score
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
    Recommendation,
    Resync,
    Side,
    Takeback,
    UserUtterance,
)
from sidecoach.profile import causes
from sidecoach.profile.insights import MIN_CHANCES, as_cells, insights, motif_words
from sidecoach.profile.puzzles import (
    Attempt,
    PuzzleStore,
    position_key,
    puzzle_level,
    record_attempt,
)
from sidecoach.profile.review import ERRORS, UserMove, build_review
from sidecoach.profile.store import Store

# A finished game waits this long for the engine's last verdicts before its review is
# built from what is there.
REVIEW_GRACE_S = 90.0
# The player's first words within this long after the coach's lesson on a mistake are
# about that mistake: kept beside it as their self-report.
SELF_REPORT_S = 60.0
# A puzzle solved this recently is not recommended again; one failed this long ago comes
# back first (spaced repetition of the plainest kind).
RETRY_S = 14 * 24 * 3600.0
RETRY_FAIL_S = 3600.0


_VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material_balance(board: chess.Board, side: chess.Color) -> int:
    """Material from `side`'s view, in pawns: positive is up."""
    total = 0
    for piece in board.piece_map().values():
        value = _VALUES.get(piece.piece_type, 0)
        total += value if piece.color == side else -value
    return total


def time_control_words(white_ms: int | None, black_ms: int | None) -> str | None:
    """The time control from the clock's first reading: "10 min", "3 min", "30 s"."""
    start = max((ms for ms in (white_ms, black_ms) if ms is not None), default=None)
    if start is None or start <= 0:
        return None
    if start >= 60_000:
        minutes = start / 60_000
        return f"{minutes:g} min"
    return f"{start // 1000} s"


@dataclass
class _Ply:
    move: MovePlayed
    move_id: int = 0
    assessment: MoveAssessment | None = None
    clock_ms: int | None = None  # the mover's clock when the move was made
    phase: str = "middlegame"
    cause: str | None = None
    motif: str | None = None
    placed: bool = False  # the evidence written


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
    time_control: str | None = None
    ended: bool = False
    # The mistake the coach last gave a lesson on, and when: the player's next words are
    # its self-report.
    lesson: tuple[tuple[int, int], float] | None = None
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
    def __init__(
        self,
        store: Store,
        *,
        session_id: str | None = None,
        ts: float = 0.0,
        puzzles: PuzzleStore | None = None,
    ) -> None:
        self.store = store
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self.store.start_session(self.session_id, ts)
        self._game: _Game | None = None
        self._clock: tuple[int | None, int | None] = (None, None)
        self._reviews: list[GameReview] = []
        # The puzzles to practise from (profile/puzzles.py), when there are any; the
        # puzzle under way on the screen, if a joined position is one; and the motif each
        # recommended puzzle was for, so its outcome lands in the right cell.
        self.puzzles = puzzles
        self._attempt: Attempt | None = None
        self._attempt_game: str | None = None
        self._recommended_for: dict[str, str] = {}

    # ---- events ------------------------------------------------------------------------

    def on_event(self, event: Event, ts: float) -> list[Event]:
        """Consume one event; the events the profile emits in return, if any."""
        if isinstance(event, GameStarted):
            return [*self.flush(ts), *self._start(event, ts)]
        if isinstance(event, ClockObservation):
            self._clock = (event.white_ms, event.black_ms)
            game = self._game
            if game is not None and game.time_control is None and not game.ended:
                game.time_control = time_control_words(event.white_ms, event.black_ms)
                if game.time_control is not None:
                    self.store.set_game(game.game_id, time_control=game.time_control)
            return []
        game = self._game
        if game is None:
            if isinstance(event, MovePlayed) and event.game_id == self._attempt_game:
                return self._puzzle_move(event, ts)
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
            at = game.plies.get((game.segment, event.ply)) if event.ply is not None else None
            self.store.add_dialogue(
                game.game_id,
                ts,
                "coach",
                event.trigger.value,
                event.text,
                event.ply,
                at.move_id if at is not None else None,
            )
            if event.trigger.value == "post_mistake" and event.ply is not None:
                # The lesson is about the move played from `ply`: the next ply's record.
                game.lesson = ((game.segment, event.ply + 1), ts)
            return []
        if isinstance(event, UserUtterance):
            self.store.add_dialogue(game.game_id, ts, "user", event.source, event.text, None)
            self._self_report(game, event.text, ts)
            return []
        if isinstance(event, GameEnded):
            return self._end(game, event, ts)
        return []

    def _start(self, event: GameStarted, ts: float) -> list[Event]:
        if event.kind != "game":
            # A puzzle, an analysis board, a game joined from the middle: coached, not
            # recorded -- a solved puzzle was "a game won by checkmate in 3 moves,
            # accuracy 100" in the record (2026-09-20). A position that is a puzzle of
            # the store is followed for its outcome, though.
            self._game = None
            self._attempt = None
            self._attempt_game = None
            # A puzzle on the panel's own board (kind "puzzle") is scored by the panel
            # through `record_puzzle`; one on the screen is matched here.
            if self.puzzles is not None and event.kind == "position":
                self._attempt = self._puzzle_at(event.start_fen)
                self._attempt_game = event.game_id if self._attempt is not None else None
            return []
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
        before = chess.Board(event.fen_before)
        after = chess.Board(event.fen_after)
        move = chess.Move.from_uci(event.uci)
        mover = before.turn
        ply.clock_ms = self._clock[0] if mover == chess.WHITE else self._clock[1]
        ply.phase = phase_of(before)
        game.plies[key] = ply
        game.last_fen = event.fen_after
        user = chess.WHITE if game.user_side is Side.WHITE else chess.BLACK
        ply.move_id = self.store.add_move(
            game.game_id,
            game.segment,
            event.ply,
            move_number=before.fullmove_number,
            san=event.san,
            uci=event.uci,
            by_user=event.by_user,
            fen_before=event.fen_before,
            fen_after=event.fen_after,
            phase=ply.phase,
            think_ms=event.think_ms,
            clock_ms=ply.clock_ms,
            capture=before.is_capture(move) if move in before.legal_moves else False,
            gives_check=after.is_check(),
            material=material_balance(after, user),
            ts=ts,
        )
        return []

    def _score_for_user(self, game: _Game, event: Analysis) -> float | None:
        """The expected score of the analysed position from the player's side: the
        engine's is from the side to move's."""
        board = chess.Board(event.fen)
        if event.lines:
            score = event.lines[0].expected_score
        else:
            terminal = terminal_expected_score(board)
            if terminal is None:
                return None
            score = terminal
        user = chess.WHITE if game.user_side is Side.WHITE else chess.BLACK
        return score if board.turn == user else 1.0 - score

    def _analysis(self, game: _Game, event: Analysis) -> list[Event]:
        key = (game.segment, event.ply)
        game.analyses[key] = event
        if event.opening:
            game.opening = event.opening
        ply = game.plies.get(key)
        if ply is not None and ply.move_id:
            self.store.analyse_move(
                ply.move_id,
                score_after=self._score_for_user(game, event),
                allowed_tags=[t.name for t in event.tags if t.name in causes.TACTICAL],
            )
        return self._place(game, key)

    def _assessment(self, game: _Game, event: MoveAssessment) -> list[Event]:
        key = (game.segment, event.ply)
        ply = game.plies.get(key)
        if ply is None:
            return []
        ply.assessment = event
        before = chess.Board(ply.move.fen_before)
        ply.phase = phase_of(before, in_book=event.label is Label.BOOK)
        faced = game.analyses.get((game.segment, event.ply - 1))
        available = (
            sorted({t.name for t in faced.tags if t.name in causes.TACTICAL})
            if faced is not None
            else []
        )
        best_uci = event.best[0] if event.best else None
        best_san: str | None = None
        if best_uci is not None:
            best_move = chess.Move.from_uci(best_uci)
            if best_move in before.legal_moves:
                best_san = before.san(best_move)
        self.store.assess_move(
            ply.move_id,
            label=event.label.value if event.label else None,
            severity=event.severity.value,
            expected_loss=event.expected_loss,
            best_uci=best_uci,
            best_san=best_san,
            accuracy=event.accuracy,
            tags=[t.name for t in event.tags],
            available_tags=available,
            depth=event.depth,
            phase=ply.phase,
        )
        if not ply.move.by_user:
            return []
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
            ply.move_id,
            game_id=game.game_id,
            segment=segment,
            ply=number,
            phase=ply.phase,
            expected_loss=loss,
            cause=cause,
            motif=motif,
            time_pressure=cause == causes.TIME_PRESSURE,
        )
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
            game.plies.pop(key)
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
        final = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(game.end.result)
        if final is not None and game.user_side is Side.BLACK:
            final = 1.0 - final
        self.store.end_game(
            game.game_id,
            game.ended_at,
            game.end.result,
            _reason_words(game.end),
            pgn=game.end.pgn,
            final_fen=game.end.final_fen,
            accuracy=review.accuracy,
            final_score=final,
            opening=game.opening,
        )
        self._reviews.append(review)
        return [review, *self._recommend(game, review, ts), self.snapshot()]

    # ---- puzzles (DESIGN section 5.7) ------------------------------------------------------

    def puzzle_level(self) -> int:
        return puzzle_level(self.store)

    def _recommend(self, game: _Game, review: GameReview, ts: float) -> list[Event]:
        """A few puzzles for the motif that costs the player most: the profile's worst
        motif cell once it has evidence enough, else this game's worst mistake's motif."""
        if self.puzzles is None or self.puzzles.count() == 0:
            return []
        motif, reason = self._weak_motif(review)
        if motif is None:
            return []
        level = self.puzzle_level()
        recent = self.store.attempts(limit=200)
        solved_lately = {pid for pid, when, solved in recent if solved and ts - when < RETRY_S}
        failed = [pid for pid, when, solved in recent if not solved and ts - when >= RETRY_FAIL_S]
        chosen = self.puzzles.pick(motif, level, exclude=solved_lately, first=failed)
        if not chosen:
            return []
        for puzzle in chosen:
            self._recommended_for[puzzle.id] = motif
        ids = [p.id for p in chosen]
        self.store.add_recommendation(game.game_id, ts, motif, reason, level, ids)
        return [
            Recommendation(
                game_id=game.game_id,
                motif=motif,
                motif_words=motif_words(motif),
                reason=reason,
                level=level,
                puzzles=[p.ref() for p in chosen],
            )
        ]

    def _weak_motif(self, review: GameReview) -> tuple[str | None, str]:
        rows = [
            r
            for r in self.store.cells()
            if r.cause is None and r.motif is not None and r.chances >= MIN_CHANCES and r.errors
        ]
        if rows:
            worst = max(rows, key=lambda r: ((r.errors + 1) / (r.chances + 2), r.expected_loss))
            return worst.motif, (
                f"{motif_words(worst.motif)}: {worst.errors} of {worst.chances} chances missed"
                " over your games"
            )
        for point in review.turning_points:
            if point.motif:
                return point.motif, f"{motif_words(point.motif)}: the turning point of this game"
        return None, ""

    def recommended_motif(self, puzzle_id: str) -> str | None:
        """The motif a puzzle was recommended for this session, if it was."""
        return self._recommended_for.get(puzzle_id)

    def record_puzzle(
        self, puzzle_id: str, motif: str | None, solved: bool, moves: list[str], ts: float
    ) -> list[Event]:
        """An outcome judged elsewhere (a puzzle solved on the puzzle page's board): kept
        beside the motif, the level moved; the event for the coach and the panel."""
        return [record_attempt(self.store, puzzle_id, motif, solved, moves, ts)]

    def _puzzle_at(self, fen: str) -> Attempt | None:
        """The puzzle whose solver's position this is, a recommended one first."""
        assert self.puzzles is not None
        try:
            key = position_key(fen)
        except ValueError:
            return None
        found = self.puzzles.at_position(key)
        if not found:
            return None
        found.sort(key=lambda p: p.id not in self._recommended_for)
        puzzle = found[0]
        return Attempt(puzzle, self._recommended_for.get(puzzle.id))

    def _puzzle_move(self, event: MovePlayed, ts: float) -> list[Event]:
        attempt = self._attempt
        if attempt is None:
            return []
        outcome = attempt.move(event.uci, by_user=event.by_user)
        if outcome is None:
            return []
        self._attempt = None
        return self.record_puzzle(
            attempt.puzzle.id, attempt.motif, outcome == "solved", attempt.played, ts
        )

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

    def _self_report(self, game: _Game, text: str, ts: float) -> None:
        """The player's first words within SELF_REPORT_S of the coach's lesson on a mistake
        are kept beside that mistake; when they say plainly what happened, the cause
        follows their word (causes.from_words)."""
        if game.lesson is None:
            return
        key, when = game.lesson
        game.lesson = None
        if ts - when > SELF_REPORT_S:
            return
        ply = game.plies.get(key)
        if ply is None or not ply.placed or ply.move_id == 0:
            return
        self.store.self_report(ply.move_id, text)
        said = causes.from_words(text)
        if said is not None and said != ply.cause:
            self.store.correct_evidence(game.game_id, key[0], key[1], said)
            ply.cause = said

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
