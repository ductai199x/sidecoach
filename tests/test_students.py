"""The simulated students (tools/eval/habits.py, tools/eval/students.py): when a habit bites
and which moves it leads to, how often the student misses, how the exam is drawn, and how the
student's answer is read."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "eval"))

import habits  # noqa: E402  # pyright: ignore[reportMissingImports]
import students  # noqa: E402  # pyright: ignore[reportMissingImports]


def _situation(scores: dict[str, float], best: float, **kw):
    moves = [chess.Move.from_uci(u) for u in scores]
    return habits.Situation(
        best=best,
        probs={m: 1 / len(moves) for m in moves},
        scores={chess.Move.from_uci(u): s for u, s in scores.items()},
        candidates=moves,
        **kw,
    )


def _reads(key: str, s) -> list[str]:
    h = habits.HABITS[key]
    return sorted(m.uci() for m in h.misses(s)) if h.bites(s) else []


def _line(board: chess.Board, sans: str) -> list[chess.Move]:
    b = board.copy()
    out = []
    for san in sans.split():
        out.append(b.parse_san(san))
        b.push(out[-1])
    return out


def test_candidates_are_the_likeliest_moves_until_the_mass_or_the_limit():
    probs = {chess.Move.from_uci(u): p for u, p in [("e2e4", 0.5), ("d2d4", 0.3), ("g1f3", 0.2)]}
    assert [m.uci() for m in habits.top_moves(probs, mass=0.75)] == ["e2e4", "d2d4"]
    assert [m.uci() for m in habits.top_moves(probs, mass=1.0, limit=1)] == ["e2e4"]


def test_a_line_is_read_where_it_settles_not_at_its_best_moment():
    # If White passed here, ...Bxf4 takes the bishop, but Re1+ Kf8 Qxf4 takes back: a trade,
    # which the line's best moment read as a bishop won (the hand review, 2026-09-26).
    passed = chess.Board("rn2k2r/ppp2ppp/3b3n/2qP4/2B2B2/2PP1QPb/PP3P1P/RN3RK1 b kq - 1 2")
    assert habits.settled(passed, _line(passed, "Bxf4 Re1+ Kf8 Qxf4"), 4) == 0
    # A line cut in the middle of an exchange is followed through it.
    board = chess.Board()
    line = _line(board, "e4 d5 exd5 Qxd5")
    assert habits.settled(board, line, 1) == 0  # past exd5, Qxd5 takes back


def test_a_tactic_is_read_from_every_near_best_line_net_of_what_was_just_taken():
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6"]:
        board.push_uci(uci)  # 4.Bxc6: Black is owed the knight
    retake = (0.5, _line(board, "dxc6"), None)
    assert habits.tactic_of(board, [retake], 0.5)[:2] == (0, False)  # taking back is no tactic
    # Two moves as good: a quiet one, and one whose line wins material. The tactic is there
    # whichever the engine happened to list first.
    start = chess.Board("4k3/8/8/3q4/8/8/3R4/4K3 w - - 0 1")
    quiet = (0.9, _line(start, "Ke2"), None)
    win = (0.9, _line(start, "Rxd5"), None)
    assert habits.tactic_of(start, [quiet, win], 0.9)[0] == 9
    assert habits.tactic_of(start, [win, quiet], 0.9)[0] == 9
    # A line far below the best is not the position's tactic.
    assert habits.tactic_of(start, [quiet, (0.5, win[1], None)], 0.9)[0] == 0


def test_a_threat_is_material_the_opponent_would_win_and_a_miss_loses_it():
    scores = {"a2a3": 0.40, "b1c3": 0.58, "h2h3": 0.45}
    swings = {chess.Move.from_uci("a2a3"): -3, chess.Move.from_uci("h2h3"): 0}
    s = _situation(scores, best=0.60, threat_win=3, swings=swings)
    # h3 is a mistake too, but keeps the material: not the threat missed.
    assert _reads("threat_blind", s) == ["a2a3"]
    # Nothing to win by passing; a decided game; in check (no pass).
    assert _reads("threat_blind", _situation(scores, best=0.6, threat_win=1, swings=swings)) == []
    assert _reads("threat_blind", _situation(scores, best=0.95, threat_win=3, swings=swings)) == []
    assert _reads("threat_blind", _situation(scores, best=0.6, swings=swings)) == []
    mate = _situation(scores, best=0.6, threat_win=0, threat_mates=True, swings=swings)
    assert _reads("threat_blind", mate) == ["a2a3"]


def test_a_greedy_miss_is_a_capture_or_check_that_comes_out_behind():
    # 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nd4: Nxe5 (a capture) and Bxf7+ (a check and a capture) are on
    # offer beside the quiet Nxd4... here scored by hand.
    scores = {"f3e5": 0.35, "c4f7": 0.30, "b1c3": 0.58, "h2h3": 0.44}
    move = chess.Move.from_uci
    board = chess.Board("r1bqkbnr/pppp1ppp/8/4p3/2BnP3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    forcing = habits.forcing_moves(board, [move(u) for u in scores])
    assert forcing == {move("f3e5"), move("c4f7")}
    swings = {move("f3e5"): 1, move("c4f7"): -2, move("h2h3"): -3}
    s = _situation(scores, best=0.60, swings=swings, forcing=forcing)
    # Nxe5 is a mistake that keeps the material (not greed that loses); h3 loses material but
    # is no capture or check.
    assert _reads("greedy", s) == ["c4f7"]
    # Without a forcing move among the candidates the habit is not on; nor in a decided game.
    assert _reads("greedy", _situation(scores, best=0.60, swings=swings)) == []
    assert _reads("greedy", _situation(scores, best=0.95, swings=swings, forcing=forcing)) == []


def test_a_tactic_position_s_misses_are_the_mistakes():
    scores = {"d1h5": 0.85, "a2a3": 0.60, "h2h3": 0.80}
    assert _reads("tactic_blind", _situation(scores, best=0.86, tactic_win=3)) == ["a2a3"]
    assert _reads("tactic_blind", _situation(scores, best=0.86, tactic_mates=True))
    assert not _reads("tactic_blind", _situation(scores, best=0.86, tactic_win=1))
    # Every candidate a mistake: nothing to get right among them.
    hopeless = _situation({"a2a3": 0.5, "h2h3": 0.5}, best=0.86, tactic_win=3)
    assert not _reads("tactic_blind", hopeless)


def test_real_players_miss_less_as_they_get_stronger():
    h = habits.HABITS["threat_blind"]
    assert abs(h.human(800) - 0.148) < 1e-9  # real 800-1100: 14.8%
    assert abs(h.human(2500) - 0.030) < 1e-9  # real 2300+: 3.0%
    ratings = (800, 1200, 1600, 2000, 2500)
    for key in habits.HABITS:
        rates = [habits.HABITS[key].human(r) for r in ratings]
        assert rates == sorted(rates, reverse=True)


def test_the_student_misses_half_the_time_more_than_nova():
    h = habits.HABITS["threat_blind"]
    assert h.student(0.0) == habits.TAKEOVER
    assert abs(h.student(0.2) - (habits.TAKEOVER + (1 - habits.TAKEOVER) * 0.2)) < 1e-9


class _Fake:
    """An engine for `student_move`: e4 good, a3 a mistake that loses a knight; passing lets
    the opponent win `threat` pawns."""

    def __init__(self, threat: int):
        self.threat = threat

    def fresh(self):
        pass

    def lines(self, board, moves):
        out = {}
        for m in moves:
            if m.uci() == "e2e4":
                out[m] = (0.6, [m], None)
            else:
                out[m] = (0.3, [m, chess.Move.from_uci("g8f6")], None)
        return out

    def top(self, board, n=3):
        return [(0.6, [chess.Move.from_uci("e2e4")], None)]

    def passed(self, board):
        return self.threat, False, []


def test_where_the_habit_bites_the_student_misses_at_its_target_rate(monkeypatch):
    # a3's line loses a knight.
    monkeypatch.setattr(habits, "swing", lambda b, line, mate: 0 if len(line) == 1 else -3)
    probs = {chess.Move.from_uci(u): p for u, p in [("e2e4", 0.9), ("a2a3", 0.1)]}
    h = habits.HABITS["threat_blind"]
    rng = random.Random(1)
    n = 4000
    missed = sum(
        habits.student_move(h, 800, chess.Board(), probs, _Fake(3), rng)[0].uci() == "a2a3"
        for _ in range(n)
    )
    # Nova plays a3 10% of the time; the habit takes the move half the time: 55%.
    assert abs(missed / n - h.student(0.1)) < 0.02
    _, bit, by_habit = habits.student_move(h, 800, chess.Board(), probs, _Fake(0), rng)
    assert not bit and not by_habit  # nothing threatened: Nova's own move


_UCI = ["e2e4", "d2d4", "g1f3", "b1c3", "a2a3", "h2h3", "a2a4", "h2h4"]


def _position(game: str, ply: int, losses: list[float], swings=None, **kw: object) -> dict:
    """A stored exam position: the candidates in Nova's order (likeliest first)."""
    n = len(losses)
    swings = swings or [0] * n
    return {
        "game": game,
        "level": 800,
        "ply": ply,
        "fen": chess.STARTING_FEN,
        "last_san": None,
        "best": 0.6,
        "threat_win": 0,
        "threat_mates": False,
        "tactic_win": 0,
        "tactic_mates": False,
        "owed": 0,
        "threat_line": [],
        "tactic_line": [],
        "candidates": [
            {
                "uci": u,
                "san": u,
                "p": round((n - i) / (n * (n + 1) / 2), 4),
                "score": 0.6 - loss,
                "loss": loss,
                "swing": sw,
            }
            for i, (u, loss, sw) in enumerate(zip(_UCI, losses, swings, strict=False))
        ],
    } | kw


def test_the_exam_takes_the_habits_likeliest_move_as_the_instinct_and_caps_each_game(
    monkeypatch,
):
    monkeypatch.setattr(students, "LEVELS", (800,))
    monkeypatch.setattr(students, "ERRORS_PER_CELL", 3)
    monkeypatch.setattr(students, "CONTROLS_PER_LEVEL", 1)
    # A threat: the likeliest move loses the piece, the third answers it.
    threats = [
        _position(f"x{i // 3}", i, [0.3, 0.2, 0.0], [-3, -3, 0], threat_win=3) for i in range(9)
    ]
    # Nothing at stake, the likeliest move good and a mistake among the rest: a control.
    control = _position("c0", 40, [0.0, 0.12, 0.05])
    items = students.select_items([*threats, control], random.Random(0))
    threat = [it for it in items if it["kind"] == "threat_blind"]
    assert len(threat) == 3 and all(it["instinct"] == "e2e4" for it in threat)
    assert all(it["candidates"][0]["uci"] == it["instinct"] for it in threat)
    assert max(sum(it["game"] == g for it in threat) for g in ("x0", "x1", "x2")) <= 2
    assert [it["instinct"] for it in items if it["kind"] == "control"] == ["e2e4"]
    assert not [it for it in items if it["kind"] == "tactic_blind"]
    assert len({(it["game"], it["ply"]) for it in items}) == len(items)  # no position twice


def test_a_tactic_position_enters_the_exam_with_its_miss_as_the_instinct(monkeypatch):
    monkeypatch.setattr(students, "LEVELS", (800,))
    shot = _position("t", 20, [0.2, 0.0, 0.3], tactic_win=3)
    items = students.select_items([shot], random.Random(0))
    # e4 and Nf3 both miss it; e4 is the likelier.
    assert [(it["kind"], it["instinct"]) for it in items] == [("tactic_blind", "e2e4")]


def test_a_control_needs_a_live_position_a_good_instinct_and_a_miss_on_offer(monkeypatch):
    monkeypatch.setattr(students, "LEVELS", (800,))
    monkeypatch.setattr(students, "CONTROLS_PER_LEVEL", 5)
    bad_instinct = _position("a", 20, [0.2, 0.0])
    no_trap = _position("b", 20, [0.0, 0.02])
    decided = _position("d", 20, [0.0, 0.3], best=0.97)
    items = students.select_items([bad_instinct, no_trap, decided], random.Random(0))
    assert not [it for it in items if it["kind"] == "control"]


def test_an_error_position_shows_a_good_move_among_the_first_few_others(monkeypatch):
    monkeypatch.setattr(students, "LEVELS", (800,))
    monkeypatch.setattr(students, "EXAM_CANDIDATES", 3)
    # The only good move is Nova's sixth choice: past what the student is shown.
    buried = _position("a", 20, [0.3, 0.2, 0.2, 0.2, 0.2, 0.0], [-3] * 5 + [0], threat_win=3)
    items = students.select_items([buried], random.Random(0))
    assert not [it for it in items if it["kind"] == "threat_blind"]


def test_the_board_is_listed_piece_by_piece():
    text = students.board_text(chess.Board())
    assert text.startswith("White: King e1; Queen d1; Rooks a1, h1;")
    assert "Black: King e8" in text


def _item(**kw):
    """An exam position from the first draw: Black's rook on c2 hangs to the bishop on d3;
    the instinct, ...c5, leaves it there."""
    return {
        "id": "L800-threat_blind-L800x11-p113",
        "level": 800,
        "fen": "8/3K4/1kp5/8/8/3BR3/2r5/8 b - - 3 57",
        "last_san": "Bd3",
        "instinct": "c6c5",
        "threat_win": 5,
        "threat_mates": False,
        "threat_line": ["d3c2", "b6c5"],
        "tactic_win": 0,
        "tactic_mates": False,
        "tactic_line": ["c2c3"],
        "owed": 0,
        "candidates": [
            {"uci": "c6c5", "san": "c5", "loss": 0.428, "swing": -5},
            {"uci": "c2c3", "san": "Rc3", "loss": 0.0, "swing": 0},
            {"uci": "c2d2", "san": "Rd2", "loss": 0.006, "swing": 0},
        ],
        **kw,
    }


class _Never:
    def chat(self, *a, **kw):
        raise AssertionError("no call")


def test_an_empty_notebook_or_one_that_asks_for_no_check_keeps_the_instinct_without_a_call():
    assert students.decide(_item(), [], {}, _Never(), None) == ("c6c5", True, "")
    # A note for the opening only, in an endgame.
    book = ["In the opening, castle early."]
    checks = {"opening": ["development"], "middlegame": [], "endgame": []}
    assert students.decide(_item(), book, checks, _Never(), None) == ("c6c5", True, "")


def test_the_checks_the_notebook_calls_for_are_answered_as_a_player_who_looks_would_see():
    item = _item()
    assert students.answer_check("threats", item) == (
        "If you do nothing about it, they play Bxc2 (their bishop on d3 takes your rook on c2),"
        " and when the exchanges are over they are 5 points of material up."
    )
    assert students.answer_check("safety", item) == (
        "No: after c5 you lose 5 points of material when the exchanges are over."
    )
    assert students.answer_check("tactics", item).startswith("No:")
    # In check there is no passing move to read a threat from.
    assert students.answer_check("threats", _item(threat_win=None)) == (
        "You are in check from Bd3."
    )


def test_a_tactic_is_named_among_the_moves_the_student_weighs():
    """The engine's own first move is not always a candidate: the student then named a move
    it could not play (3 of 156 in the first gate)."""
    item = _item(tactic_win=3, tactic_line=["b6b5"])
    item["candidates"][2]["swing"] = 3
    assert students.answer_check("tactics", item).startswith("Yes: Rd2 (your rook from c2 to")
    item["candidates"][2]["swing"] = 0
    assert students.answer_check("tactics", item).startswith("No: none of the moves you are")


def test_the_students_move_is_read_from_its_answer_after_its_checks():
    def said(text):
        class One:
            def chat(self, system, messages, **kw):
                prompt = messages[0]["content"]
                assert "The move that came to mind first: c5 (your pawn from c6 to c5)" in prompt
                assert "they play Bxc2 (their bishop on d3 takes your rook on c2)" in prompt
                return SimpleNamespace(text=text, stats={})

        checks = {"opening": ["threats"], "middlegame": ["threats"], "endgame": ["threats"]}
        return students.decide(_item(), ["look at what they attack"], checks, One(), None)

    assert said("MOVE: Rc3")[:2] == ("c2c3", True)
    assert said("MOVE: Rd2 (your rook from c2 to d2)")[:2] == ("c2d2", True)
    # A move not among the candidates: the instinct, marked as not a valid answer.
    assert said("MOVE: Qh5")[:2] == ("c6c5", False)
    assert said("I would play Rc3")[:2] == ("c6c5", False)


def test_the_notebook_is_read_note_by_note_and_only_the_notes_there_are():
    def read(text, notes):
        class One:
            def chat(self, system, messages, **kw):
                return SimpleNamespace(text=text, stats={})

        return students.notebook_checks(notes, One(), None)[0]

    # The reader split a one-note notebook at its semicolon and answered for a note 2.
    got = read("1: threats | any\n2: threats, tactics, safety | any", ["look at what it attacks"])
    assert got == {"opening": ["threats"], "middlegame": ["threats"], "endgame": ["threats"]}
    got = read("1: development | opening\n2: none | any", ["castle early", "stay calm"])
    assert got == {"opening": ["development"], "middlegame": [], "endgame": []}


def test_a_note_counts_only_when_it_comes_from_the_coachs_words():
    coach = [
        "That cost you something: Bxg1 was the move -- it takes their rook: nothing defends "
        "it; after fxg6, hxg6. What were you expecting that move to do?"
    ]
    assert students.from_coach("it takes their rook: nothing defends it", coach)
    assert students.from_coach("Bxg1 takes their rook, nothing defends it", coach)
    assert students.from_coach("What were you expecting that move to do", coach)
    # The student's own rule, and a quote too short to be one.
    assert not students.from_coach("scan forcing captures first", coach)
    assert not students.from_coach("rook", coach)


def test_a_session_keeps_the_games_where_the_habit_shows_most():
    def pgn(chose: str, bit: str, tag: str) -> str:
        return f'[Event "{tag}"]\n[HabitChances "{bit}"]\n[HabitPlies "{chose}"]\n\n1. e4 *'

    games = [pgn("12", "12,14", "a"), pgn("8,20,31", "8,20,31", "b"), pgn("", "5", "c")]
    kept = students.most_habit(games, 2)
    assert [g.split('"')[1] for g in kept] == ["a", "b"]  # in the order they were played
