"""What a reviewed game says for Stats (profile/stat_facts.py): the definitions on known
positions -- tactic chances found, missed and allowed, their kinds, the pawn structure, the
endgame's kind, bot games."""

from __future__ import annotations

import chess

from sidecoach.events import (
    Analysis,
    Label,
    Line,
    MotifTag,
    MoveAssessment,
    ReviewedGame,
    ReviewedMove,
    Severity,
    Side,
)
from sidecoach.profile.stat_facts import (
    bot_game,
    chance_in,
    endgame_of,
    facts_of,
    kind_of,
    structure_of,
)
from sidecoach.profile.store import LibraryRow


def _tags(*names: str, move: str | None = None) -> list[MotifTag]:
    return [MotifTag(name=n, squares=["e5"], move=move) for n in names]


def test_a_tactic_is_named_by_its_most_telling_motif() -> None:
    assert kind_of(_tags("fork", "mateIn2")) == "checkmate"
    assert kind_of(_tags("hangingPiece", "fork")) == "fork", "a free piece only when alone"
    assert kind_of(_tags("hangingPiece")) == "free piece"
    assert kind_of(_tags("absolutePin")) == "pin"
    assert kind_of(_tags("doubleCheck")) == "discovery"
    assert kind_of(_tags("capturingDefender")) == "removing the defender"
    assert kind_of(_tags("skewer")) == "other"
    assert kind_of(_tags("capture", "check", "quietMove")) is None, "not a tactic"


def _analysis(
    fen: str = chess.STARTING_FEN,
    *,
    score: float = 0.8,
    decisive: bool = True,
    mate: int | None = None,
    tags: list[MotifTag] | None = None,
    best: tuple[str, ...] = ("e2e4", "e7e5", "g1f3", "b8c6"),
) -> Analysis:
    return Analysis(
        game_id="g",
        fen=fen,
        ply=0,
        lines=[Line(moves=list(best), depth=18, expected_score=score, mate=mate)],
        decisive=decisive,
        only_move=decisive,
        tags=tags if tags is not None else _tags("fork", move="g1f3"),
    )


def test_a_chance_is_a_clearly_best_tactic_a_player_could_find() -> None:
    assert chance_in(_analysis()) == "fork"
    assert chance_in(_analysis(decisive=False)) is None, "several moves as good: no chance"
    assert chance_in(_analysis(score=0.3)) is None, "a tactic that still loses"
    # Dealt by the mover within their first two moves of the line, not later.
    assert chance_in(_analysis(tags=_tags("fork", move="e2e4"))) == "fork"
    assert chance_in(_analysis(tags=_tags("fork", move="e7e5"))) is None, "the opponent's move"
    late = _analysis(tags=_tags("fork", move="f1c4"), best=("e2e4", "e7e5", "g1f3", "b8c6", "f1c4"))
    assert chance_in(late) is None
    # A mate counts when a player could see it: in three, not in eighteen.
    assert chance_in(_analysis(mate=2, tags=_tags("mateIn2"))) == "checkmate"
    assert chance_in(_analysis(mate=18, tags=_tags("mate"))) is None


def test_the_centre_pawns_make_a_position_open_semi_open_or_closed() -> None:
    assert structure_of(chess.Board()) == "semi-open"
    french = chess.Board("rnbqkbnr/pp3ppp/4p3/2ppP3/3P4/8/PPP2PPP/RNBQKBNR w KQkq - 0 4")
    assert structure_of(french) == "closed", "d4 blocked by d5"
    exchanged = chess.Board("r1bqkb1r/ppp2ppp/2n2n2/8/8/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 6")
    assert structure_of(exchanged) == "open", "no pawn on the d- or e-file"


def test_an_endgame_is_named_by_what_is_left_besides_kings_and_pawns() -> None:
    assert endgame_of(chess.Board("8/5k2/4p3/8/3P4/8/5K2/8 w - - 0 1")) == "pawn"
    assert endgame_of(chess.Board("3r4/5k2/4p3/8/3P4/8/5K2/3R4 w - - 0 1")) == "rook"
    assert endgame_of(chess.Board("3q4/5k2/4p3/8/3P4/8/5K2/3Q4 w - - 0 1")) == "queen"
    assert endgame_of(chess.Board("3r4/5k2/4p3/8/3P4/8/5K2/3B4 w - - 0 1")) == "other"


def test_a_game_against_a_computer_is_told_by_its_headers() -> None:
    assert bot_game({"Event": "Play vs Bot", "White": "nosegazm", "Black": "David-BOT"})
    assert bot_game({"Event": "Play vs Coach", "White": "Coach-David", "Black": "nosegazm"})
    assert bot_game({"Event": "Rated game", "White": "me", "BlackTitle": "BOT", "Black": "x"})
    assert not bot_game({"Event": "Live Chess", "White": "Anakin_737", "Black": "nosegazm"})


def test_a_game_gives_each_move_its_facts() -> None:
    board = chess.Board()
    start = _analysis(tags=_tags("fork", move="e2e4"), best=("e2e4", "e7e5"))
    moves = []
    for ply, (uci, loss, after) in enumerate(
        [
            ("e2e4", 0.0, _analysis(score=0.3, decisive=False, tags=[])),
            ("e7e5", 0.3, _analysis(score=0.8, tags=_tags("pin", move="g1f3"))),
        ],
        start=1,
    ):
        move = chess.Move.from_uci(uci)
        side = Side.WHITE if board.turn == chess.WHITE else Side.BLACK
        before = board.fen()
        san = board.san(move)
        board.push(move)
        moves.append(
            ReviewedMove(
                ply=ply,
                san=san,
                side=side,
                fen_before=before,
                assessment=MoveAssessment(
                    game_id="g",
                    ply=ply,
                    uci=uci,
                    expected_loss=loss,
                    severity=Severity.NONE,
                    best=[],
                    label=Label.BEST if loss == 0 else Label.BLUNDER,
                    accuracy=100.0 if loss == 0 else 20.0,
                ),
                after=after.model_copy(update={"fen": board.fen(), "ply": ply}),
                spent_ms=4000,
            )
        )
    game = ReviewedGame(
        headers={"Event": "Live Chess", "White": "a", "Black": "b", "Result": "1-0"},
        start=start,
        moves=moves,
        strength="standard",
        depth=18,
    )
    row = LibraryRow(
        key="pgn:x",
        source="pgn",
        added_at=0.0,
        played_at=1.0,
        white="a",
        black="b",
        white_elo=None,
        black_elo=None,
        result="1-0",
        time_class="rapid",
        user_side="white",
        pgn="",
    )
    game_fact, facts = facts_of(game, row)
    assert (game_fact.bot, game_fact.time_class, game_fact.user_side) == (False, "rapid", "white")
    first, second = facts
    # White had the fork and played it; Black's reply handed White a pin.
    assert (first.chance, first.found, first.allowed) == ("fork", True, None)
    assert (second.chance, second.found, second.allowed) == (None, False, "pin")
    assert first.phase == "opening" and first.structure == "semi-open" and first.spent_ms == 4000
