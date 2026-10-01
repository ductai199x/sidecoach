"""The coach under test: a coach on a scripted model or on none, the scholar's game it is
taken through (1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??, where 4. Qxf7# mates), and readers of the
harness the model is given."""

from __future__ import annotations

from collections.abc import Sequence

import chess

from sidecoach.coach.coach import Coach
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings
from sidecoach.events import (
    Analysis,
    CoachMessage,
    GameStarted,
    Label,
    Line,
    MotifTag,
    MoveAssessment,
    MovePlayed,
    Severity,
    Side,
)
from sidecoach.llm.base import ChatMessage
from sidecoach.llm.offline import OfflineProvider
from tests.fakes import ScriptedProvider

# 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??: white to move, Qxf7 mates.
SCHOLAR_MOVES = ("e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6")
SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def _policy() -> InterjectionPolicy:
    return InterjectionPolicy(PolicySettings(stuck_seconds=10))


def scripted_coach(replies: Sequence[str]) -> tuple[Coach, ScriptedProvider]:
    """A coach whose model answers `replies` in turn."""
    provider = ScriptedProvider(replies)
    return Coach(provider, policy=_policy()), provider


def offline_coach() -> Coach:
    """The coach with no model (llm.provider = none): its templates are the answer."""
    return Coach(OfflineProvider(), policy=_policy())


def start_game(coach: Coach, user_side: Side = Side.WHITE) -> None:
    coach.on_event(
        GameStarted(game_id="g", user_side=user_side, start_fen=chess.Board().fen()), 0.0
    )


def play(
    coach: Coach, sans: Sequence[str], *, user_side: Side = Side.WHITE, think_ms: int | None = None
) -> chess.Board:
    """`sans` played from the start position, ply n at n seconds, and the board they reach."""
    board = chess.Board()
    for ply, san in enumerate(sans, start=1):
        move = board.parse_san(san)
        before = board.fen()
        board.push(move)
        coach.on_event(
            MovePlayed(
                game_id="g",
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=before,
                fen_after=board.fen(),
                by_user=(ply % 2 == 1) == (user_side is Side.WHITE),
                think_ms=think_ms,
            ),
            float(ply),
        )
    return board


def play_to_scholar(coach: Coach) -> None:
    """White, the user, to move with Qxf7# on the board, after black's Nf6."""
    start_game(coach)
    play(coach, SCHOLAR_MOVES)
    assert coach.user_to_move and coach.board.fen() == SCHOLAR


def mate_analysis(ply: int, decisive: bool = True, game_id: str = "g") -> Analysis:
    """The engine's look at SCHOLAR: Qxf7 mates."""
    return Analysis(
        game_id=game_id,
        fen=SCHOLAR,
        ply=ply,
        lines=[
            Line(moves=["h5f7"], depth=12, expected_score=1.0, mate=1),
            Line(moves=["c4f7"], depth=12, expected_score=0.55),
        ],
        decisive=decisive,
        only_move=True,
        tags=[MotifTag(name="mateIn1", squares=["f7", "e8"], move="h5f7")],
    )


def _judged_at_scholar(
    coach: Coach, san: str, loss: float, severity: Severity, label: Label, ts: float
) -> None:
    """The user's move from SCHOLAR (ply 7), and a second later the engine's verdict on it."""
    board = chess.Board(SCHOLAR)
    move = board.parse_san(san)
    board.push(move)
    coach.on_event(
        MovePlayed(
            game_id="g",
            ply=7,
            san=san,
            uci=move.uci(),
            fen_before=SCHOLAR,
            fen_after=board.fen(),
            by_user=True,
        ),
        ts,
    )
    coach.on_event(
        MoveAssessment(
            game_id="g",
            ply=7,
            uci=move.uci(),
            expected_loss=loss,
            severity=severity,
            best=["h5f7"],
            label=label,
        ),
        ts + 1,
    )


def mate_played(coach: Coach) -> None:
    """The user plays Qxf7#, the engine's own move, from SCHOLAR."""
    _judged_at_scholar(coach, "Qxf7#", 0.0, Severity.NONE, Label.BEST, 10.0)


def mate_missed(coach: Coach, ts: float = 10.0) -> None:
    """The user plays Bxf7+ from SCHOLAR, where Qxf7 was mate: a miss."""
    _judged_at_scholar(coach, "Bxf7+", 0.45, Severity.BLUNDER, Label.MISS, ts)


def drain(coach: Coach) -> list[CoachMessage]:
    """The coach's pending requests run, and the messages it then has for the player."""
    coach.speaker.run_pending()
    return coach.poll()


def stuck(coach: Coach) -> list[CoachMessage]:
    """The player stuck on the position the stuck limit long: the coach's first word unasked
    (the only one while they think -- Tai, 2026-10-01), and what it then has for them."""
    since = coach.policy.stuck_from
    assert since is not None
    coach.tick(since + coach.policy.stuck_threshold())
    return drain(coach)


def sent(provider: ScriptedProvider, index: int = -1) -> list[ChatMessage]:
    """The messages the model was sent on the `index`-th call."""
    return provider.prompts[index][1]


def harness_of(messages: list[ChatMessage]) -> str:
    """The one user message: the harness with the task at its end."""
    assert len(messages) == 1 and messages[0]["role"] == "user"
    return messages[0]["content"]


def directive_of(messages: list[ChatMessage]) -> str:
    """The harness's task: what the model is asked to do."""
    return harness_of(messages).rsplit("# Task" + chr(10), 1)[1]
