from sidecoach.coach.disclosure import (
    Disclosure,
    PositionFacts,
    Unlock,
    leak_check,
    visible_facts,
)
from sidecoach.events import DisclosureLevel as L

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def facts() -> PositionFacts:
    return PositionFacts(
        fen=START,
        side_to_move="white",
        phase="opening",
        theme="loose piece",
        region="around f7",
        piece="your knight",
        candidates=["Ng5"],
        threats=["Qxd4"],
        line=["Ng5", "d5", "Nxf7"],
    )


def test_autonomous_escalation_stops_at_region() -> None:
    d = Disclosure()
    assert d.apply(Unlock.DECISIVE) == L.QUESTION
    assert d.apply(Unlock.STUCK) == L.THEME
    assert d.apply(Unlock.STUCK) == L.REGION
    assert d.apply(Unlock.STUCK) is None
    assert d.level == L.REGION


def test_hint_request_may_reach_piece_but_not_candidate() -> None:
    d = Disclosure(level=L.REGION)
    assert d.apply(Unlock.HINT) == L.PIECE
    assert d.apply(Unlock.HINT) is None
    assert d.level == L.PIECE


def test_asking_for_the_move_unlocks_candidates() -> None:
    d = Disclosure()
    assert d.apply(Unlock.ASK_MOVE) == L.CANDIDATE
    assert d.apply(Unlock.ASK_MOVE) is None


def test_naming_the_move_unlocks_candidates() -> None:
    d = Disclosure(level=L.THEME)
    assert d.apply(Unlock.USER_NAMED_MOVE) == L.CANDIDATE


def test_playing_the_move_unlocks_the_line() -> None:
    d = Disclosure()
    assert d.apply(Unlock.MOVE_PLAYED) == L.LINE
    assert d.history == [(Unlock.MOVE_PLAYED, L.LINE)]


def test_visible_facts_hide_moves_below_candidate() -> None:
    bundle = visible_facts(facts(), L.REGION)
    assert bundle["theme"] == "loose piece"
    assert bundle["region"] == "around f7"
    for hidden in ("piece", "candidates", "threats", "line"):
        assert hidden not in bundle
    assert "fen" in bundle


def test_visible_facts_show_everything_at_line() -> None:
    bundle = visible_facts(facts(), L.LINE)
    assert bundle["line"] == ["Ng5", "d5", "Nxf7"]
    assert bundle["candidates"] == ["Ng5"]


def test_leak_check_flags_withheld_square_and_move() -> None:
    leaks = leak_check("Have you considered Ng5, hitting f7?", facts(), L.THEME)
    assert "move Ng5" in leaks
    assert "square f7" in leaks


def test_leak_check_flags_any_move_below_candidate_even_if_not_withheld() -> None:
    leaks = leak_check("Maybe d3 or Nc3 here?", facts(), L.QUESTION)
    assert "square d3" in leaks
    assert "move Nc3" in leaks


def test_leak_check_allows_candidates_but_not_line_at_candidate_level() -> None:
    assert leak_check("Ng5 is the idea.", facts(), L.CANDIDATE) == []
    leaks = leak_check("After Ng5 d5 Nxf7 you win.", facts(), L.CANDIDATE)
    assert "square d5" in leaks
    assert "move Nxf7" in leaks


def test_leak_check_allows_last_move_square() -> None:
    f = facts()
    f.last_move = "e4"
    assert leak_check("After e4, what changed?", f, L.QUESTION) == []


def test_leak_check_allows_region_anchor_square() -> None:
    assert leak_check("Look at what is happening around f7.", facts(), L.REGION) == []


def test_leak_check_flags_piece_below_piece_level() -> None:
    assert "piece knight" in leak_check("What is your knight doing?", facts(), L.QUESTION)
    assert leak_check("What is your knight doing?", facts(), L.PIECE) == []


def test_leak_check_clean_question() -> None:
    assert leak_check("What is your opponent threatening right now?", facts(), L.QUESTION) == []
