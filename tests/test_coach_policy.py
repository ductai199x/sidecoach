"""The fact bundle a coach draws on, and when the coach may speak unprompted."""

from __future__ import annotations

import chess

from sidecoach.coach.facts import (
    facts_from_analysis,
    material_text,
    phase_of,
    region_of,
    theme_of,
)
from sidecoach.coach.policy import (
    InterjectionPolicy,
    PolicySettings,
    TimeControl,
    time_control_of,
)
from sidecoach.events import (
    Analysis,
    Label,
    Line,
    MotifTag,
    MoveAssessment,
    Severity,
    Trigger,
)

# 1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6??, white to move: Qxf7 is mate.
SCHOLAR = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"


def _analysis(fen: str, **kw: object) -> Analysis:
    base: dict[str, object] = dict(
        game_id="g",
        fen=fen,
        ply=6,
        lines=[Line(moves=["h5f7"], depth=12, expected_score=1.0, mate=1)],
        decisive=True,
        only_move=False,
    )
    base.update(kw)
    return Analysis(**base)  # type: ignore[arg-type]


# The facts


def test_facts_name_the_theme_region_piece_and_candidates() -> None:
    board = chess.Board(SCHOLAR)
    analysis = _analysis(
        SCHOLAR,
        tags=[MotifTag(name="mateIn1", squares=["f7", "e8"], move="h5f7")],
        threats=["f6h5"],
    )
    facts = facts_from_analysis(board, analysis, last_move="Nf6")
    assert facts.side_to_move == "white" and facts.phase == "opening"
    assert facts.theme == "a mating attack"
    assert facts.region == "the kingside, around e8", "around the king being mated"
    assert facts.piece == "your queen on h5", "the piece where it stands, never its destination"
    assert facts.candidates == ["Qxf7#"]
    assert facts.threats == ["Nxh5"], "what the opponent would play if it were their turn"
    assert facts.line == ["Qxf7#"]
    assert facts.decisive and facts.last_move == "Nf6"


def test_facts_survive_a_line_the_board_cannot_play() -> None:
    board = chess.Board(SCHOLAR)
    analysis = _analysis(SCHOLAR, lines=[Line(moves=["a7a5", "b2b3"], depth=1, expected_score=0.5)])
    facts = facts_from_analysis(board, analysis)
    assert facts.candidates == [] and facts.piece is None and facts.line == []
    assert facts.theme is None and facts.region is None


def test_near_equal_candidates_are_all_named() -> None:
    board = chess.Board()
    analysis = _analysis(
        board.fen(),
        lines=[
            Line(moves=["e2e4"], depth=12, expected_score=0.53),
            Line(moves=["d2d4"], depth=12, expected_score=0.52),
            Line(moves=["g1f3"], depth=12, expected_score=0.50),
        ],
        decisive=False,
    )
    assert facts_from_analysis(board, analysis).candidates == ["e4", "d4"]


def test_an_incidental_tactic_is_demoted_and_the_strategic_plan_leads() -> None:
    # A pin sits on the e-file but the best move (Bf1-e2, developing) has nothing to do with
    # it and the position is not decisive, so the tactic is withheld and the plan takes over.
    fen = "r1bqk2r/pppp1ppp/2n2n2/2b1p3/4P3/2NP1N2/PPP2PPP/R1BQKB1R w KQkq - 0 9"
    board = chess.Board(fen)
    analysis = _analysis(
        fen,
        lines=[Line(moves=["f1e2"], depth=12, expected_score=0.40)],
        decisive=False,
        only_move=False,
        tags=[MotifTag(name="pin", squares=["e5", "e4", "e1"], move=None)],
    )
    facts = facts_from_analysis(board, analysis)
    assert facts.theme is None and facts.focus == [] and facts.tactic is None
    assert facts.plan == "finishing your development"
    assert facts.ideas, "the position's static features are given as ideas"


def test_a_decisive_tactic_still_leads_and_no_plan_competes() -> None:
    board = chess.Board(SCHOLAR)
    analysis = _analysis(
        SCHOLAR, tags=[MotifTag(name="mateIn1", squares=["f7", "e8"], move="h5f7")]
    )
    facts = facts_from_analysis(board, analysis, last_move="Nf6")
    assert facts.theme == "a mating attack"  # decisive: the tactic is the point
    assert facts.plan is None, "no second headline when a tactic leads"


def test_theme_ranks_the_most_specific_tag_first() -> None:
    tags = [
        MotifTag(name="capture", squares=["e5"]),
        MotifTag(name="hangingPiece", squares=["e5"]),
        MotifTag(name="fork", squares=["c7", "e8", "a8"]),
    ]
    chosen = theme_of(tags)
    assert chosen is not None and chosen.name == "fork"
    assert theme_of([MotifTag(name="quietMove", squares=["e4"])]) is None
    # Bookkeeping tags never headline: they hold on most quiet moves.
    assert theme_of([MotifTag(name="interference", squares=["e4"])]) is None
    assert theme_of([MotifTag(name="attackingF2F7", squares=["f7"])]) is None


def test_regions_and_anchors() -> None:
    assert region_of(["f7", "g8", "h7"]) == "the kingside, around f7"
    assert region_of(["b2", "a1"]) == "the queenside, around b2"
    assert region_of(["d4", "e5"]) == "the centre, around d4"
    assert region_of([]) is None
    # The anchor lies in the region the squares chose.
    assert region_of(["g4", "d7", "c8"]) == "the queenside, around d7"


def test_phase_and_material_in_the_users_words() -> None:
    assert phase_of(chess.Board()) == "opening"
    endgame = chess.Board("8/5k2/8/8/8/8/P4K2/8 w - - 0 50")
    assert phase_of(endgame) == "endgame"
    middlegame = chess.Board(
        "r1bq1rk1/pp2ppbp/2np1np1/2p5/2P1P3/2N2NP1/PP1PBP1P/R1BQ1RK1 w - - 0 12"
    )
    assert phase_of(middlegame) == "middlegame"
    assert material_text(chess.Board()) == "level"
    assert material_text(
        chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    ) == ("you are down a queen")
    assert material_text(
        chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1")
    ) == ("you are up a queen")
    assert material_text(chess.Board("4k3/8/8/8/8/8/8/4KN2 w - - 0 1")) == "you are up a knight"
    # The pieces, not a sum: a bishop against a pawn is not "down two pawns".
    assert material_text(chess.Board("4k3/4p3/8/8/8/8/8/2B1K3 b - - 0 1")) == (
        "you are down a bishop for a pawn"
    )
    assert material_text(chess.Board("4k3/8/8/8/8/8/8/2B1K2R b - - 0 1")) == (
        "you are down a rook and a bishop"
    )
    assert material_text(chess.Board("2b1k3/8/8/8/8/8/8/4K2R w - - 0 1")) == (
        "you are up the exchange"
    )
    assert material_text(chess.Board("2b1k3/8/8/8/8/8/8/1N2K3 w - - 0 1")) == (
        "level in value: your knight against their bishop"
    )


# The policy


def _assessed(severity: Severity, label: Label | None = None) -> MoveAssessment:
    return MoveAssessment(
        game_id="g",
        ply=5,
        uci="e2e4",
        expected_loss=0.2 if severity is Severity.BLUNDER else 0.0,
        severity=severity,
        best=["d2d4"],
        label=label,
    )


def test_time_controls_follow_the_sites_bands() -> None:
    assert time_control_of(None) is TimeControl.UNTIMED
    assert time_control_of(15 * 60_000) is TimeControl.RAPID
    assert time_control_of(10 * 60_000) is TimeControl.RAPID
    assert time_control_of(5 * 60_000) is TimeControl.BLITZ
    assert time_control_of(3 * 60_000, 2_000) is TimeControl.BLITZ
    assert time_control_of(60_000, 1_000) is TimeControl.BULLET
    assert time_control_of(2 * 60_000, 12_000) is TimeControl.RAPID, "increment counts forty moves"


def test_two_unprompted_messages_per_position_the_idea_then_the_focus() -> None:
    policy = InterjectionPolicy(PolicySettings(stuck_seconds=10))
    policy.new_position(6, 100.0, user_to_move=True)
    assert policy.on_position_analysed(6, decisive=True) is Trigger.DECISIVE_QUESTION
    policy.said(6, 101.0)
    assert policy.on_position_analysed(6, decisive=True) is None, "already spoke here"
    assert policy.tick(105.0) is None, "a full stuck spell from the idea"
    assert policy.tick(112.0) is Trigger.STUCK, "then the focus"
    policy.said(6, 112.0)
    assert policy.tick(200.0) is None, "and no third: the move only when asked"
    policy.new_position(8, 200.0, user_to_move=True)
    assert policy.tick(205.0) is None, "not stuck yet"
    assert policy.tick(211.0) is Trigger.STUCK
    assert policy.on_position_analysed(9, decisive=True) is None, "not the position on the board"


def test_nothing_unprompted_while_the_opponent_thinks_or_when_quiet() -> None:
    policy = InterjectionPolicy(PolicySettings(stuck_seconds=1))
    policy.new_position(7, 0.0, user_to_move=False)
    assert policy.on_position_analysed(7, decisive=True) is None
    assert policy.tick(50.0) is None
    quiet = InterjectionPolicy(PolicySettings(stuck_seconds=1, quiet=True))
    quiet.new_position(6, 0.0, user_to_move=True)
    assert quiet.on_position_analysed(6, decisive=True) is None
    assert quiet.tick(50.0) is None
    assert quiet.on_user_move_assessed(_assessed(Severity.BLUNDER)) is None


def test_a_mistake_earns_a_lesson_and_a_good_move_a_word() -> None:
    policy = InterjectionPolicy()
    assert policy.on_user_move_assessed(_assessed(Severity.NONE)) is Trigger.POST_MOVE
    assert policy.on_user_move_assessed(_assessed(Severity.INACCURACY)) is Trigger.POST_MOVE
    assert policy.on_user_move_assessed(_assessed(Severity.MISTAKE)) is Trigger.POST_MISTAKE
    assert policy.on_user_move_assessed(_assessed(Severity.BLUNDER)) is Trigger.POST_MISTAKE
    assert policy.on_user_move_assessed(_assessed(Severity.INACCURACY, Label.MISS)) is (
        Trigger.POST_MISTAKE
    ), "a missed chance is worth a lesson whatever it cost"


def test_blitz_keeps_only_the_post_move_word_and_bullet_nothing() -> None:
    blitz = InterjectionPolicy(PolicySettings(stuck_seconds=1), time_control=TimeControl.BLITZ)
    blitz.new_position(6, 0.0, user_to_move=True)
    assert blitz.on_position_analysed(6, decisive=True) is None
    assert blitz.tick(60.0) is None
    assert blitz.on_user_move_assessed(_assessed(Severity.BLUNDER)) is Trigger.POST_MOVE
    bullet = InterjectionPolicy(time_control=TimeControl.BULLET)
    bullet.new_position(6, 0.0, user_to_move=True)
    assert bullet.on_user_move_assessed(_assessed(Severity.BLUNDER)) is None
    assert bullet.on_position_analysed(6, decisive=True) is None


def test_the_stuck_threshold_shrinks_with_the_clock() -> None:
    policy = InterjectionPolicy(PolicySettings(stuck_seconds=45, stuck_floor_seconds=20))
    assert policy.stuck_threshold() == 45
    policy.set_remaining(15 * 60_000)
    assert policy.stuck_threshold() == 45, "plenty of time: the default holds"
    policy.set_remaining(5 * 60_000)
    assert policy.stuck_threshold() == 24, "eight per cent of five minutes"
    policy.set_remaining(100_000)
    assert policy.stuck_threshold() == 20, "never below the floor"


def test_no_stuck_hints_in_a_time_scramble() -> None:
    # With 100 s left a hint still comes, at the floor; with 25 s left the user has no time
    # to hear a question, let alone act on it.
    policy = InterjectionPolicy(PolicySettings(stuck_seconds=45, stuck_floor_seconds=20))
    policy.new_position(6, 0.0, user_to_move=True)
    policy.set_remaining(100_000)
    assert policy.tick(21.0) is Trigger.STUCK
    policy = InterjectionPolicy(PolicySettings(stuck_seconds=45, stuck_floor_seconds=20))
    policy.new_position(6, 0.0, user_to_move=True)
    policy.set_remaining(25_000)
    assert policy.tick(60.0) is None
    assert policy.last_reason == "time scramble: no stuck hints"


def test_the_decisive_question_is_asked_once_per_decisive_stretch() -> None:
    """A won endgame is decisive at every move (the lines diverge at each turn), and the
    question was asked after every one of them: "before making a quiet move, what forcing
    possibilities ..." six times in a row. Once asked, it waits for a quiet position in
    between, or eight plies."""
    policy = InterjectionPolicy()
    policy.new_position(60, 0.0, user_to_move=True)
    assert policy.on_position_analysed(60, decisive=True) is Trigger.DECISIVE_QUESTION
    policy.said(60)
    policy.new_position(62, 10.0, user_to_move=True)
    assert policy.on_position_analysed(62, decisive=True) is None
    assert policy.last_reason is not None and "already pointed at" in policy.last_reason
    policy.new_position(64, 20.0, user_to_move=True)
    assert policy.on_position_analysed(64, decisive=False) is None, "a quiet position between"
    policy.new_position(66, 30.0, user_to_move=True)
    assert policy.on_position_analysed(66, decisive=True) is Trigger.DECISIVE_QUESTION
    policy.said(66)
    policy.new_position(74, 40.0, user_to_move=True)
    assert policy.on_position_analysed(74, decisive=True) is Trigger.DECISIVE_QUESTION, (
        "eight plies on, even with no quiet position between"
    )
