# ruff: noqa: E501
"""Reading a saved chess.com review, and the labels and accuracy we put beside it.

The fixture is markup copied from the review page and keeps its long lines on purpose.
"""

from __future__ import annotations

from pathlib import Path

import chess
import pytest

from sidecoach.analysis.chesscom import (
    Reference,
    ReferenceMove,
    load,
    match_game,
    parse_move_list,
    parse_overview,
    save,
    split_games,
    verify,
)
from sidecoach.analysis.score import Thresholds, game_accuracy, label_move, move_accuracy
from sidecoach.events import Label

# The markup as copied from the review page's developer tools, cut down to what matters:
# figurine spans inside the move text, the icon's SVG id, the CSS colour that names the
# label, and the per-ply timestamps in tenths of a second.
MOVE_LIST = """
<wc-move-list>
<div class="main-line-row move-list-row light-row" data-whole-move-number="1">1.
  <div data-node="0-0" class="node white-move main-line-ply">
    <span class="node-highlight-content offset-for-annotation-icon selected">e4 </span></div>
  <div data-node="0-1" class="node black-move main-line-ply">
    <span class="node-highlight-content offset-for-annotation-icon">e5 </span></div>
  <div data-move-list-el="timestamp" data-ply="1" data-time="10" class="time-white">1.0s</div>
  <div data-move-list-el="timestamp" data-ply="2" data-time="1" class="time-black">0.1s</div>
</div>
<div class="main-line-row move-list-row dark-row" data-whole-move-number="2">2.
  <div data-node="0-2" class="node white-move main-line-ply">
    <span class="node-highlight-content offset-for-annotation-icon"><span class="icon-font-chess knight-white " data-figurine="N"></span> f3 </span></div>
  <div data-node="0-3" class="node black-move main-line-ply"><span class="node-annotation-icon">
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 19">
      <g id="book"><path class="icon-shadow" d="M9,.5a9,9,0,1,0,9,9A9,9,0,0,0,9,.5Z"></path></g>
    </svg></span> <span class="node-highlight-content" style="color: var(--color-classification-book)"><span class="icon-font-chess knight-black " data-figurine="N"></span> c6 </span></div>
  <div data-move-list-el="timestamp" data-ply="3" data-time="20" class="time-white">2.0s</div>
  <div data-move-list-el="timestamp" data-ply="4" data-time="8" class="time-black">0.8s</div>
</div>
<div class="main-line-row move-list-row light-row" data-whole-move-number="3">3.
  <div data-node="0-4" class="node white-move main-line-ply"><span class="node-annotation-icon">
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 19">
      <defs><style>.cls-2{fill:#FF7769;}</style></defs><g id="incorrect"><path class="cls-2" d="M9,0"></path></g>
    </svg></span> <span class="node-highlight-content" style="color: var(--color-classification-miss)"><span class="icon-font-chess bishop-white " data-figurine="B"></span> c4 </span></div>
  <div data-node="0-5" class="node black-move main-line-ply"><span class="node-annotation-icon">
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 18 19">
      <g id="blunder"><path d="M9,.5"></path></g>
    </svg></span> <span class="node-highlight-content" style="color: var(--color-classification-blunder)"><span class="icon-font-chess knight-black " data-figurine="N"></span> f6 </span></div>
  <div data-move-list-el="timestamp" data-ply="5" data-time="649" class="time-white">1m 4s</div>
  <div data-move-list-el="timestamp" data-ply="6" data-time="9" class="time-black">0.9s</div>
</div>
<div class="main-line-row move-list-row dark-row" data-whole-move-number="4">4.
  <div data-node="0-6" class="node white-move main-line-ply"><span class="node-annotation-icon">
    <svg><g id="best"><path d="M9"></path></g></svg></span> <span class="node-highlight-content" style="color: var(--color-classification-best)"><span class="icon-font-chess knight-white " data-figurine="N"></span> xe5 </span></div>
  <div data-node="0-7" class="node black-move main-line-ply">
    <span class="node-highlight-content offset-for-annotation-icon">d6 </span></div>
</div>
</wc-move-list>
"""


def test_the_move_list_parses_to_plies_labels_and_times() -> None:
    moves = parse_move_list(MOVE_LIST)
    assert [m.san for m in moves] == ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Nxe5", "d6"]
    assert [m.label for m in moves] == [None, None, None, "book", "miss", "blunder", "best", None]
    assert [m.time_ds for m in moves] == [10, 1, 20, 8, 649, 9, None, None]
    assert moves[4].ply == 5


def test_the_moves_are_checked_against_the_game() -> None:
    moves = parse_move_list(MOVE_LIST)
    board = chess.Board()
    truth = [board.push_san(san) for san in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Nxe5"]]
    with pytest.raises(ValueError, match="beyond the 7 plies"):
        verify(moves, truth)
    verify(moves[:7], truth)
    truth[4] = chess.Move.from_uci("b1c3")
    with pytest.raises(ValueError, match="review has Bc4, the game has Nc3"):
        verify(moves[:7], truth)
    with pytest.raises(ValueError, match="does not play"):
        verify([ReferenceMove(1, "Ke2", None)])


def test_a_list_that_is_not_a_game_is_refused() -> None:
    doubled = MOVE_LIST.replace(
        'class="node black-move main-line-ply"', 'class="node white-move main-line-ply"', 1
    )
    with pytest.raises(ValueError, match="not a game"):
        parse_move_list(doubled)


# The overview panel, cut down the same way: tallies come as pairs of spans, white then
# black, and the accuracy and game rating rows share one component class.
OVERVIEW = """
<div class="game-overview-component">
<div class="game-overview-row"><span class="game-overview-row-title"> Accuracy</span>
  <div class="game-overview-row-item"><div class="review-rating-component review-rating-white"><!----> <span>82.3</span></div></div>
  <div class="game-overview-row-item"><div class="review-rating-component review-rating-black"><!----> <span>74.1</span></div></div></div>
<div class="tally-row-row"><div class="tally-row-rowTitle"><span> Skills</span></div> <span class="tally-row-value analysis-skill">+1</span> <span class="tally-row-glyph"><svg><g id="skills"></g></svg></span> <span class="tally-row-value analysis-skill"></span></div>
<div class="tally-row-row"><div class="tally-row-rowTitle"><span> Great</span></div> <span class="tally-row-value analysis-greatFind">2</span> <span class="tally-row-glyph"><svg><g id="great_find"></g></svg></span> <span class="tally-row-value analysis-greatFind">0</span></div>
<div class="tally-row-row"><div class="tally-row-rowTitle"><span> Book</span></div> <span class="tally-row-value analysis-book">6</span> <span class="tally-row-glyph"></span> <span class="tally-row-value analysis-book">6</span></div>
<div class="tally-row-row"><div class="tally-row-rowTitle"><span> Best</span></div> <span class="tally-row-value analysis-bestMove">22</span> <span class="tally-row-glyph"></span> <span class="tally-row-value analysis-bestMove">16</span></div>
<div class="tally-row-row"><div class="tally-row-rowTitle"><span> Mistake</span></div> <span class="tally-row-value analysis-mistake">0</span> <span class="tally-row-glyph"></span> <span class="tally-row-value analysis-mistake">6</span></div>
<div class="game-overview-row"><span class="game-overview-row-title"> Game Rating</span>
  <div class="game-overview-row-item"><div class="review-rating-component review-rating-white"><!----> <span>1800</span></div></div>
  <div class="game-overview-row-item"><div class="review-rating-component review-rating-black"><!----> <span>1600</span></div></div></div>
</div>
"""


def test_the_overview_gives_the_tallies_accuracy_and_rating() -> None:
    overview = parse_overview(OVERVIEW)
    assert overview is not None
    assert overview.counts["white"] == {"great": 2, "book": 6, "best": 22, "mistake": 0}
    assert overview.counts["black"] == {"great": 0, "book": 6, "best": 16, "mistake": 6}
    assert overview.accuracy == {"white": 82.3, "black": 74.1}
    assert overview.rating == {"white": 1800, "black": 1600}
    assert parse_overview(MOVE_LIST) is None, "a move list alone has no overview"


def test_a_page_with_several_reviews_splits_into_games() -> None:
    second = MOVE_LIST.replace(">e4 <", ">d4 <").replace(">e5 <", ">d5 <")
    page = OVERVIEW + MOVE_LIST + "\n<hr>\n" + second + OVERVIEW.replace("82.3", "50.0")
    games = split_games(page)
    assert len(games) == 2
    assert games[0].overview is not None and games[0].overview.accuracy["white"] == 82.3
    assert games[0].moves[0].san == "e4" and games[1].moves[0].san == "d4"
    assert games[1].overview is None, "an overview belongs to the list after it, not before"
    board = chess.Board()
    truth = [board.push_san(san) for san in ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6", "Nxe5", "d6"]]
    assert match_game(games[0], truth) and not match_game(games[1], truth)
    assert match_game(games[0], truth + [board.push_san("Nf3")]), "a prefix still matches"
    assert not match_game(games[0], truth[:3]), "but the list may not be longer than the game"


def test_saved_and_loaded_references_round_trip(tmp_path: Path) -> None:
    moves = parse_move_list(MOVE_LIST)
    html = tmp_path / "chesscom.html"
    html.write_text(OVERVIEW + MOVE_LIST, encoding="utf-8")
    reference = load(html)
    assert reference.moves == moves and reference.overview is not None
    as_json = tmp_path / "chesscom.json"
    save(reference, as_json)
    assert load(as_json) == reference
    html.write_text(MOVE_LIST, encoding="utf-8")
    assert load(html) == Reference(moves=moves, overview=None)


def _label(
    loss: float,
    *,
    is_best: bool = False,
    in_book: bool = False,
    only_move: bool = False,
    sacrifice: bool = False,
    before_score: float = 0.5,
    opponent_erred: bool = False,
    obvious: bool = False,
) -> Label:
    return label_move(
        loss,
        Thresholds(),
        is_best=is_best,
        in_book=in_book,
        only_move=only_move,
        sacrifice=sacrifice,
        before_score=before_score,
        opponent_erred=opponent_erred,
        obvious=obvious,
    )


def test_labels_follow_the_review_vocabulary() -> None:
    assert _label(0.0, in_book=True) is Label.BOOK
    assert _label(0.0, is_best=True) is Label.BEST
    assert _label(0.01) is Label.EXCELLENT
    assert _label(0.03) is Label.GOOD
    assert _label(0.06) is Label.INACCURACY
    assert _label(0.12) is Label.MISTAKE
    assert _label(0.30) is Label.BLUNDER
    # The opponent handed over a chance: taking it with the best move is great, letting it
    # go is a miss however large the loss, and a decent move that is neither is just good.
    assert _label(0.0, opponent_erred=True, is_best=True) is Label.GREAT
    assert _label(0.0, opponent_erred=True, only_move=True) is Label.GREAT
    assert _label(0.03, opponent_erred=True) is Label.GOOD
    assert _label(0.06, opponent_erred=True) is Label.MISS
    assert _label(0.90, opponent_erred=True) is Label.MISS
    # A sound sacrifice is brilliant, unless the position was already decided either way.
    assert _label(0.0, sacrifice=True) is Label.BRILLIANT
    assert _label(0.0, sacrifice=True, before_score=0.97, is_best=True) is Label.BEST
    assert _label(0.0, sacrifice=True, before_score=0.05, is_best=True) is Label.BEST
    # The only move is great whether or not it was the engine's first choice, unless it is
    # obvious (a recapture, the only legal move) or the game was already decided.
    assert _label(0.0, only_move=True) is Label.GREAT
    assert _label(0.0, only_move=True, obvious=True) is Label.BEST
    assert _label(0.01, only_move=True, obvious=True) is Label.EXCELLENT
    assert _label(0.0, only_move=True, before_score=0.95) is Label.BEST
    assert _label(0.0, opponent_erred=True, is_best=True, obvious=True) is Label.BEST
    # Nothing better was found: best, whether or not it was the engine's first choice.
    assert _label(0.0) is Label.BEST
    # An unsound sacrifice is judged by its loss like any move.
    assert _label(0.3, sacrifice=True) is Label.BLUNDER


def test_accuracy_is_a_hundred_for_no_loss_and_falls_with_it() -> None:
    assert move_accuracy(0.0) == 100.0
    assert 78 < move_accuracy(0.05) < 82
    assert 20 < move_accuracy(0.30) < 30
    assert move_accuracy(1.0) == 0.0
    assert game_accuracy([]) is None
    assert game_accuracy([100.0, 100.0]) == 100.0
    assert game_accuracy([100.0, 50.0]) == 75.0
    assert game_accuracy([100.0, 0.0]) == 50.0, "one lost game does not halve the number"
