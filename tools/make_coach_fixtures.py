"""Record what the desktop's coach primitives say, for the Kotlin port to be held to.

Everything under `sidecoach/coach/` that is a pure function of the board and the analysis:
who "you" is, the disclosure filter and the leak check, the fact bundle, the idea of the best
line (`coach/idea.py`) and its words (`coach/idea_words.py`), the offline coach's templates,
what a move does, what stands on the board, where a line leads, the word on a sound move,
when the coach may speak, and the marks that go on the board. The coach reads these out; a
port that differs anywhere says something different.

The positions come four ways, and every Analysis fed to the desktop is written into the
fixture as it was fed, so the Kotlin reads exactly the same one:

- the Lichess puzzle sample and random games, with the analysis synthesised from the
  puzzle's own solution line (candidates, threats, a line and tags, the shapes the engine
  hands over in a real game);
- per theme of `data/puzzles/themes-7-120.csv` (mates by pattern, forks, pins, equality,
  pawn endgames, zugzwang, quiet moves, sacrifices...), the engine's real analysis from the
  puzzle bench's cache (`data/puzzles/analysis/`), so every goal of the idea reader is met;
- the desktop's own idea tests (`tests/test_idea.py`), their lines written out, and two
  hand-made draws (a stalemate, a repetition);
- positions from the same cache whose idea reaches the rarer goals and fields (RARE: a trade
  into a won ending, the king leading, an attack, a combination, a threat to meet, a
  perpetual, a pinned guard, an in-between move...);
- positions of the master games (`data/eval/facts/masters_v2/`, their engine analysis as the
  eval harness recorded it), chosen for the plan the facts carry (`coach/plans.py`, desktop
  5e80cdf): each kind of line plan -- an attack, a storm, doubled heavy pieces, a knight's
  outpost, a passed pawn, a break, space on the wing -- and each threat of the best move -- a
  mate, a piece to win, a trade and what it is for -- up to MASTERS_PER_KIND, and a stride of
  the rest; and the threats the opponent prepares (`idea._threat`, desktop 6637098: a quiet
  reply to a pass that would threaten mate or a piece next, met by the best move), up to
  MASTERS_PREPARED of each; and what the rungs read since desktop 268990e: a win that comes a
  move later (`idea.gain_ply`), a mate too long to read out whole (past `MATE_READ`), a forcing
  capture's target, and a pawn threat that gives way to the move's own idea, up to
  MASTERS_RUNGS of each; and, since desktop fc30fa7, the first hint's trap (a capture that
  looks free and is not), up to MASTERS_TRAPS.

The first hint (desktop fc30fa7) reads the Analysis's traps -- the captures that look free and
are not, searched by the annotator -- and whether the game is in the book. The cached analyses
predate them, so the traps are searched here with the desktop's Stockfish (`trap_lines`, one
thread, so the same every run) for every Lichess and cache position, and for master positions
while MASTERS_TRAPS is not met; opening games played out move by move carry the book's
`in_book` and name. Every case also says what the move that left it allowed
(`idea_words.allowed`, desktop 25a3333).

Since desktop 10fdd21/a7d4ada the hints tell one story (`coach/story.py`): every case carries
the desktop's story checks on its three hints (`tools/puzzles/story_checks.py`, desktop tooling:
won, plain, free, checker, queening, reason, how), and where the Analysis has traps, their
ranking and each one's words with the best move (`situation.ranked_traps`, `trap_words`). The
master positions are also chosen for each beat of the story, up to MASTERS_STORY of each; the
six puzzles of `tests/test_story.py` are recorded whole, with their hints and whether every
story check holds; and a few traps are made by hand for the ranking, the wrong piece to take
back with, and the balance lost.

Since desktop a1f7f81 the traps searched beside an analysis come with the recaptures its best
line declines (`analysis/defences.py`), and the two puzzles of `tests/test_story.py` on why the
obvious defence fails are recorded as `_hints_with` builds them; since desktop 29ae423, with
what a quiet best move threatens (`threats.aim_lines`), and the three quiet-move puzzles too.

Every case's lessons are also said as the move-by-move lesson says them, with the position and
the move played (`lesson_template(..., board=, played=)`): every legal capture onto the better
move's square by another piece (the right capture with the other piece), and the first legal
move.

    cd D:/chess-coaching-companion
    PYTHONDONTWRITEBYTECODE=1 uv run --no-sync python D:/sidecoach-android/tools/make_coach_fixtures.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

DESKTOP = Path("D:/chess-coaching-companion")
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

import chess  # noqa: E402

from sidecoach.analysis.defences import defence_lines  # noqa: E402
from sidecoach.analysis.engine import Engine  # noqa: E402
from sidecoach.analysis.motifs import tags_for_position  # noqa: E402
from sidecoach.analysis.openings import OpeningBook  # noqa: E402
from sidecoach.analysis.threats import aim_lines  # noqa: E402
from sidecoach.analysis.traps import tempting, trap_lines  # noqa: E402
from sidecoach.analysis.phase import phase_of  # noqa: E402
from sidecoach.analysis.score import human_expected_score  # noqa: E402
from sidecoach.coach import annotations as A  # noqa: E402
from sidecoach.coach import disclosure as D  # noqa: E402
from sidecoach.coach import effects as E  # noqa: E402
from sidecoach.coach import facts as F  # noqa: E402
from sidecoach.coach import idea as IDEA  # noqa: E402
from sidecoach.coach import idea_words as W  # noqa: E402
from sidecoach.coach import inventory as I  # noqa: E402
from sidecoach.coach import outcome as O  # noqa: E402
from sidecoach.coach import plans as PL  # noqa: E402
from sidecoach.coach import situation as SIT  # noqa: E402
from sidecoach.coach import policy as P  # noqa: E402
from sidecoach.coach import verdicts as V  # noqa: E402
from sidecoach.coach.bag import Bags  # noqa: E402
from sidecoach.coach.coach import (  # noqa: E402
    COST_YOU,
    FINE_TIGHTER,
    focus_template,
    hint_template,
    idea_template,
    lesson_template,
    line_template,
    move_template,
)
from sidecoach.coach.person import third_person  # noqa: E402
from sidecoach.config import find_stockfish, load_settings  # noqa: E402

sys.path.insert(0, str(DESKTOP / "tools" / "puzzles"))
from story_checks import story_checks  # noqa: E402
from sidecoach.events import Analysis, DisclosureLevel, Line, MotifTag, Trigger  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
PUZZLES = DESKTOP / "data" / "puzzles" / "sample.csv"
THEMES = DESKTOP / "data" / "puzzles" / "themes-7-120.csv"
CACHE = DESKTOP / "data" / "puzzles" / "analysis" / "d18-pv6-13cdf1d231.jsonl"
# Puzzles per theme with the engine's real analysis; a puzzle counts for every theme it has.
PER_THEME = 5
MASTERS = DESKTOP / "data" / "eval" / "facts" / "masters_v2"
# Master positions kept per plan the facts carry (a line plan's kind, a threat's kind and what
# a trade is for); one in MASTERS_STRIDE of the rest, whose plan is the best move's own or none.
MASTERS_PER_KIND = 24
MASTERS_STRIDE = 150
# Master positions kept per prepared threat to meet (mate, a piece).
MASTERS_PREPARED = 40
# Master positions kept per rung reading of desktop 268990e (a later win, a long mate, a
# forcing capture's target, a pawn threat that yields).
MASTERS_RUNGS = 30
# Master positions kept whose first hint names a trap.
MASTERS_TRAPS = 40
# Master positions kept per beat of the story (desktop 10fdd21): a piece that simply hangs, a
# free piece first, the checker, the pawn to queen, the prize, how a win comes, what a move
# does where only it holds, a win back.
MASTERS_STORY = 25

# `tests/test_story.py` (desktop a7d4ada): Lichess puzzles read by hand, the lines as the app's
# settings saw them: (name, fen, [(line, score)], threats).
STORY_TESTS = [
    (
        "gNzJg: a queen that simply hangs",
        "5r1k/p5pp/1p6/3p1r2/3P2q1/2P1QP1n/PP5P/R4R1K w - - 0 29",
        [(["f3g4", "h3f2", "h1g1", "f5f3", "e3f2", "f3f2", "f1f2", "f8f2", "g1f2"], 0.949)],
        (),
    ),
    (
        "DWvS7: a free piece first",
        "4r1k1/6p1/3qp2p/1Q1pr3/p1p1nP2/Pp6/1P4PP/4RRK1 w - - 0 29",
        [(["b5e8", "g8h7", "f4e5", "d6c5", "g1h1", "e4f2", "f1f2", "c5f2", "e1g1"], 0.762)],
        (),
    ),
    (
        "EcCX4: in check, the checker first",
        "4r3/5ppp/5n2/2Q2k2/P1qP1B2/7P/1P2p1P1/4R1K1 b - - 4 30",
        [
            (["c4c5", "d4c5", "f5f4", "g1f2", "f6e4", "f2e2", "e4c5", "e2d2", "e8e1"], 0.911),
            (["c4d5", "f4d2", "d5c5", "d4c5", "e8b8", "g2g4", "f5e4", "e1e2", "e4d3"], 0.278),
        ],
        (),
    ),
    (
        "uSAIW: the pawn the win rests on",
        "5k2/p7/5PRp/1p6/7p/8/r2r2P1/5RK1 w - - 0 36",
        [
            (["f6f7", "d2g2", "g6g2", "a2g2", "g1g2", "b5b4", "g2h3", "b4b3", "h3h4"], 0.88),
            (["g6g7", "a7a5", "g1h2", "h4h3", "h2h3", "d2f2", "f1c1", "a2c2", "c1a1"], 0.454),
        ],
        (),
    ),
    (
        "jlCZP: a win with no pattern says how",
        "8/8/p3k3/1p3p1p/1P6/P3K1PP/8/8 w - - 1 41",
        [
            (["e3f4", "e6f6", "h3h4", "f6e6", "f4g5", "e6d5", "g5h5", "a6a5", "b4a5"], 0.895),
            (["e3d4", "e6d6", "h3h4", "d6e6", "d4e3", "e6e7", "e3f4", "e7f6", "f4f3"], 0.5),
        ],
        ("e6e5",),
    ),
    (
        "TzhLp: where only one move holds",
        "8/4k3/3p1pp1/7p/4PKPP/5P2/8/8 w - - 1 44",
        [
            (["g4h5", "g6h5", "f4f5", "e7f7", "f3f4", "f7e7", "f5g6", "e7e6", "f4f5"], 0.862),
            (["f4e3", "e7f7", "e3f4", "f7e6", "g4g5", "e6f7", "g5f6", "f7f6", "f4e3"], 0.512),
            (["g4g5", "e7f7", "g5f6", "f7f6", "f4e3", "g6g5", "h4g5", "f6g5", "f3f4"], 0.512),
        ],
        (),
    ),
]

# `tests/test_story.py` (desktop a1f7f81, 29ae423): the Analysis as `_hints_with` builds it --
# decisive, one move, the declined recaptures, the traps and what a quiet move threatens given:
# (name, fen, [(line, score)], [(defence, score, mate)], [(trap, score, mate)],
# [(threat, score, mate)]).
DEFENCE_STORY_TESTS = [
    (
        "Qxg3: why the obvious recapture fails is the lesson",
        "2kr1b1r/ppq2p2/2p1pp2/8/2B5/3P2Q1/PPPB1PP1/R4RK1 b - - 1 1",
        [
            (["c7g3", "f1e1", "g3h4", "g1f1", "h4g4", "f1g1"], 0.975),
            (["f8d6", "f2f4", "f7f5", "f1f3", "c7b6", "g3f2"], 0.267),
        ],
        [(["c7g3", "f2g3", "f8c5", "d3d4", "c5d4", "f1f2", "d4f2"], 0.931, None)],
        [],
        [],
    ),
    (
        "Qxb8: a defender with two jobs is not a loose piece",
        "1r3rk1/pp3ppp/2pQP3/1P1pP1q1/3Nn3/6P1/P5KP/2R2R2 w - - 1 2",
        [
            (["e6f7", "g8h8", "d6b8", "g5d2", "g2h1", "d2b4"], 0.975),
            (["d6c7", "g5d2", "g2h1", "f7e6", "d4e6"], 0.261),
        ],
        [
            (["e6f7", "f8f7", "d6b8", "g5d8", "b8d8", "f7f8", "d8f8"], 1.0, 3),
            (["e6f7", "g8h8", "d6b8", "f8b8", "f7f8q", "b8f8", "f1f8"], 1.0, 2),
        ],
        [(["d6b8", "g5d2", "g2h1", "f8b8", "d4f3"], 0.09, None)],
        [],
    ),
    (
        "06VoP: a quiet move is told by what it threatens",
        "2rr2k1/5pp1/p1qp3p/4pP1N/1p6/1P4PK/PP5P/R2Q4 w - - 5 32",
        [
            (["d1g4", "g8f8", "g4g7", "f8e8", "g7g8", "e8e7", "f5f6", "e7e6"], 0.924),
            (["d1f1"], 0.241),
        ],
        [],
        [],
        [(["g4g7"], 1.0, 1)],
    ),
    (
        "05cb6: a quiet move that threatens a piece names it where it stands",
        "r4rk1/p3pp1p/1p4p1/3P4/2P4b/1Q3R2/PP2N3/1q1K2R1 w - - 3 35",
        [
            (["d1d2", "b1e4", "f3f4", "e4e5", "f4h4", "e7e6", "b3c3", "e5c3"], 0.797),
            (["e2c1", "a8c8", "g1g4", "h4f6", "f3f6", "e7f6", "b3c2", "b1c2"], 0.093),
        ],
        [],
        [],
        [(["g1b1", "e7e6", "b1h1", "a7a5", "h1h4", "a5a4"], 0.959, None)],
    ),
    (
        "00pWF: a quiet pawn move threatens a break",
        "8/8/p3k3/1p1pP3/5P2/1P3K2/8/8 b - - 1 53",
        [
            (
                [
                    "a6a5", "f3g4", "e6f7", "f4f5", "a5a4", "b3a4", "b5a4", "f5f6",
                    "f7g6", "f6f7", "g6f7", "g4f5", "f7e7", "e5e6", "a4a3", "f5e5",
                ],
                0.874,
            ),
            (["b5b4"], 0.456),
            (
                [
                    "e6f5", "b3b4", "a6a5", "b4a5", "b5b4", "f3e3", "b4b3", "e3d3",
                    "b3b2", "d3c2", "f5e4",
                ],
                0.089,
            ),
        ],
        [],
        [],
        [(["a5a4", "b3a4", "b5a4", "f3e3", "a4a3", "f4f5"], 0.872, None)],
    ),
]

# Traps made by hand (desktop 10fdd21's ranking and words): (name, fen, best, target, top,
# [(line, score, mate)]). Tai's game g03, 13.Raxd1 for Rexd1: the wrong rook, and the balance
# lost; Capablanca-Marshall 16.Re2: the rook on a8 is the target, and a mate beside it.
TRAP_TESTS = [
    (
        "the wrong piece to take back with",
        "r3k2r/p1p1bppp/2p5/4p3/2b1P3/2N1B2P/PPP2PP1/R2qR1K1 w kq - 0 13",
        "e1d1",
        None,
        0.54,
        [(["a1d1", "e7b4"], 0.3, None)],
    ),
    (
        "on the target, into mate, the worst",
        "r1b2rk1/2p2ppp/p2b4/1p6/3P3q/1BP2Q1P/PP3nP1/RNB1R1K1 w - - 0 16",
        "e1e2",
        "a8",
        0.68,
        [
            (["f3f2", "h4f2", "g1f2", "c8e6"], 0.35, None),
            (["f3f2", "d6h2", "g1f1", "h2g1"], 0.0, -4),
            (["f3a8", "f2h3", "g2h3", "h4e1", "g1g2", "e1e2", "g2g1", "e2h2"], 0.0, -8),
        ],
    ),
]
# Opening games played out move by move, each position in the book (desktop fc30fa7's
# "this is opening theory -- the X; the main moves here are ...").
OPENING_GAMES = [
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6",
    "d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5",
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6",
]
# What a move allowed is said as "after {played}, ...": the played move's words only.
PLAYED = "e4"

# Sentences written for the side to move as "you", the way the facts and ideas read.
PERSON_TEXTS = [
    "your king is still in the centre",
    "you win a pawn",
    "your opponent's isolated pawn is a target",
    "they have the opposition, so you must give way",
    "you are up material, so trade pieces where you can",
    "you still have 2 pieces at home (b1, c1), not yet in the game",
    "Your rook holds the open e-file",
    "you can still castle",
    "your opponent takes on f7",
    "they were better, but you are now winning",
    "you need to watch the back rank",
    "theirs is the bishop pair",
    "you just played a quiet move and they answered",
    "you no longer have the initiative",
    "you and the opponent both have a passed pawn",
    "your queen, the strongest piece, is loose",
    "you try to push the passer",
    "they study the position",
    "you fix the pawn and they cannot free it",
    "you don't have time for that",
    # tests/test_conversation.py's: "you" as the object, and "yours" (the parity audit, 2026-09-29).
    "Your opponent's rook holds the file; it attacks you.",
    "yours is the better bishop",
]

LEAK_TEXTS = [
    "What is your opponent trying to do?",
    "Look at f7.",
    "Have you checked every capture? Nxf7 is one.",
    "The knight is the piece to think about.",
    "Your rook on a1 is undefended.",
    "You are up a rook and a pawn.",
    "The Queen's Gambit is a fine opening.",
    "Consider Bb5+ before anything else.",
    "There is a weakness around g7.",
    "e4 and d4 are the squares that matter.",
    "Qh4+ wins on the spot.",
    "Think about the kingside.",
    "Your king on g1 is safe.",
    "That takes the pawn on d5 and opens the file.",
    "Nothing to say.",
    # The piece-word rule of 2026-09-23: their pieces, what a line wins, and the king are
    # never the player's own piece.
    "You can win their queen here.",
    "Their rook is loose, and your bishop is the one to use.",
    "Their king runs out of squares.",
    "You could trap a knight, or win two pawns.",
    "Your queen and your rook work together.",
    "The enemy bishop is pinned.",
]

# The desktop's idea tests (`tests/test_idea.py`): (fen, SAN line, score, mate, threats,
# decisive), each a puzzle the coach misread before the reader was rebuilt.
IDEA_TESTS = [
    ("3r4/3r2k1/p7/3p2p1/3RP1Pp/1P3K2/P2R4/8 b - - 0 38", "dxe4+ Ke3 Rxd4", 0.95, None, [], True),
    (
        "1bK5/pP3p2/B7/1P4p1/6k1/8/7p/8 w - - 0 44",
        "Kxb8 h1=Q Kxa7 Kh3 b8=Q Qa1 Qc8+ g4 b6",
        0.95,
        None,
        [],
        True,
    ),
    (
        "2r2r2/2b1qpk1/p1n1p2p/1pPp2pn/1P1P4/P2Q3P/2BB1PP1/R3R1K1 w - - 0 24",
        "Qh7+ Kf6 Qxh6#",
        0.95,
        2,
        [],
        True,
    ),
    (
        "r3qr1k/1p2b1p1/4p1Qp/2ppP3/p4P2/2P1P1P1/PP4P1/1K1R3R w - - 1 21",
        "Rxh6+ gxh6 Qxh6+ Kg8 Rh1 Bh4 Rxh4 Qd7 Rg4+ Kf7 Rg7+ Ke8",
        0.95,
        None,
        [],
        True,
    ),
    (
        "rbb2rk1/1p3ppp/p1n2n2/3p4/3Pp3/P1N1B2P/1qP1BPPN/R2QK2R w KQ - 0 14",
        "Na4 Qxa1 Qxa1",
        0.95,
        None,
        [],
        True,
    ),
    (
        "rn1Q2rk/1bp1n2p/4p3/pp6/2N1P3/2q5/P1P2P1P/1K1R2R1 w - - 2 27",
        "Rxg8+ Nxg8 Rg1 Qg7 Rxg7 Kxg7 Qxc7+ Nd7 Qxd7+",
        0.95,
        None,
        [],
        True,
    ),
    ("r5k1/1p4bp/p1n1p1p1/3q4/3P3r/1Q2P2B/PP3P1P/3RK2R w K - 0 22", "Bxe6+ Qxe6 Qxe6+", 0.95, None, [], True),
    (
        "8/6kp/2Qq1pp1/p7/3r4/1P5P/P4PP1/2R3K1 b - - 0 32",
        "Rd1+ Rxd1 Qxc6 Re1 Qc2 Re7+ Kh6 g4 Qb1+ Kg2 Qxa2",
        0.95,
        None,
        [],
        True,
    ),
    ("5rk1/5p2/P7/8/8/8/5R2/5RK1 w - - 0 1", "Rxf7 Rxf7 Rxf7 Kxf7 a7", 0.95, None, [], True),
    ("r3nr1k/4N1pp/p2p1p2/4nR2/1p2P3/2q4Q/P3B1PP/4R2K w - - 4 24", "Qxh7+ Kxh7 Rh5#", 0.95, 2, [], True),
    ("6R1/8/4r3/5P1p/7P/k5PK/8/1q6 w - - 0 54", "Ra8+ Kb4 Rb8+ Kc5 Rxb1", 0.95, None, [], True),
    (
        "r2r1bk1/1b5p/p4Pp1/1p2N1P1/4P2Q/1PPpq2P/6K1/R4R2 w - - 1 31",
        "f7+ Kh8 Nxg6+ Kg7 Qh6#",
        0.95,
        3,
        [],
        True,
    ),
    ("r3k3/p1pb4/2p2p2/2q1p1p1/2B3p1/2P4r/P1P2PQP/3RR1K1 w q - 4 21", "Qe4", 0.55, None, ["c5c4"], False),
    ("1k2r3/1p3ppp/3p4/2n5/1rPN4/1P3P2/P5PP/2KRR3 b - - 2 23", "Rxe1 Rxe1 Nd3+ Kd2 Nxe1", 0.95, None, [], True),
    # `tests/test_allowed.py` (desktop 25a3333): a blunder that drops a rook, a move that lets
    # them trade into a won ending, a quiet line, a win far down the line, and g06's a3 (the
    # last two in `tests/test_rungs.py` since the consolidation of 2026-09-29).
    (
        "1n1qk2r/2pp1pp1/r7/3Qp2p/8/2B2P2/1PP3PP/R3K1NR b KQk - 0 13",
        "Rxa1+ Kf2 Qf6 Bxe5 Qf5 Qe4 Qxe4 fxe4",
        0.92,
        None,
        [],
        False,
    ),
    ("r1b3k1/ppp2ppp/3b4/8/3Qq3/P4N2/1BP2PPP/R5K1 b - - 1 18", "Qxd4 Bxd4 Bf5 c3 b6 Be5 Rd8 Rd1", 0.82, None, [], False),
    ("4k3/8/8/8/8/8/4P3/4K3 b - - 0 1", "Kd7 Kd2 Kd6 Kd3", 0.5, None, [], False),
    (
        "rnbqkb1r/pppp4/4p1p1/5n2/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 7",
        "h4 d5 g4 Nxh4 Bg5 Be7 Bxh4 Bxh4 g5 Nc6 Rxh4 Rxh4 Nxh4 Qxg5",
        0.9,
        None,
        [],
        False,
    ),
    ("rnbqkb1r/pppp4/4p1p1/5n2/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 7", "a3 d5", 0.5, None, [], False),
    # `tests/test_threats_to_meet.py`: Short-Timman before 11...Nd4, where Rd1 would come and
    # then Rxd8 -- met by Nd4, and not by Kh8 (desktop 6637098).
    (
        "r1bq1rk1/1pp1ppbp/1nn3p1/p3P3/P7/1B3N1P/1PP1QPP1/RNB2RK1 b - - 0 11",
        "Nd4 Nxd4 Qxd4 Re1",
        0.39,
        None,
        ["f1d1"],
        False,
    ),
    (
        "r1bq1rk1/1pp1ppbp/1nn3p1/p3P3/P7/1B3N1P/1PP1QPP1/RNB2RK1 b - - 0 11",
        "Kh8 Rd1",
        0.39,
        None,
        ["f1d1"],
        False,
    ),
    # `tests/test_rungs.py` (desktop 268990e): Tai's games g02-g04 -- a pawn that falls a move
    # later, the right capture with the other rook, a lesson said of the better move (in
    # `tests/test_allowed.py` since 2026-09-29), a mate in twelve, a forcing capture, and a
    # pawn threat that gives way to the pawn ending's idea.
    (
        "r1bq1rk1/2pnbppp/p2p4/1p1P4/3pP3/1B3P2/PPP1N1PP/R1BQK2R w KQ - 1 12",
        "O-O Nc5 Nxd4 Bd7 c3 Nxb3 axb3 a5",
        0.67,
        None,
        [],
        False,
    ),
    ("r3k2r/p1p1bppp/2p5/4p3/2b1P3/2N1B2P/PPP2PP1/R2qR1K1 w kq - 0 13", "Rexd1 Bb4 Na4 Be2", 0.54, None, [], False),
    ("r2qk2r/p1p1bppp/2p5/4p3/2b1P3/2N4P/PPP2PP1/R1BQR1K1 w kq - 0 12", "Qg4 g6 b3 Be6", 0.6, None, [], False),
    (
        "8/k6n/2K3NP/8/1P4P1/2P5/8/8 w - - 1 53",
        "b5 Ka8 Ne5 Nf8 b6 Ne6 b7+ Ka7 b8=Q+ Kxb8 h7 Nd8+ Kb6 Nf7 Nxf7 Kc8",
        1.0,
        12,
        [],
        False,
    ),
    (
        "r1bqk2r/2p1bppp/p1np1n2/1p2p3/3PP3/1BN2N2/PPP2PPP/R1BQK2R w KQkq - 0 8",
        "dxe5 dxe5 Qxd8+ Bxd8",
        0.53,
        None,
        [],
        False,
    ),
    ("4k3/8/8/4p3/8/8/3P4/4K3 w - - 0 1", "d4 Kd7", 0.5, None, [], False),
    # Two draws no cached line reaches: a stalemate on the board, and a position that comes
    # round again (no perpetual: two checks are not three).
    ("7k/4Q3/6K1/8/8/8/8/8 w - - 0 1", "Qf7", 0.5, None, [], False),
    ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Nf3 Nf6 Ng1 Ng8", 0.5, None, [], False),
]

# Positions whose engine analysis (in the bench's cache) reaches the rarer goals and fields of
# the idea reader, found by reading the whole cache once (2026-09-24): a trade into a won
# ending, the king leading (with the opposition), an attack on the king, a combination, a
# threat to meet, a draw by perpetual, a pawn won back to level, a pinned guard, an
# in-between move, a second tactic, two sacrifices, a won position with nothing to take. No
# cached line reaches "breakthrough": the pawn that runs to the seventh makes the line a
# promotion first (`idea._read`), on the desktop as on the phone.
# In check, the best move the one king step after which no check follows: the move rung says
# so ("Ke7 steps out of check to the one square where no check follows"). Found in random games.
NO_CHECK_FOLLOWS = [
    ("r1b2kn1/p5pr/np1pN3/2p2p1p/Pb2qP2/1R1P2K1/2P3PP/1NQ1BB1R b - - 1 27", "f8e7"),
    ("4kbr1/1rnq1pp1/pnp5/2P1NP2/pP2P2p/6Pb/1Q1BBRK1/7R w - - 0 36", "g2g1"),
    ("5r1k/Qbp2p2/3pP2p/p1nP4/PpP1PpPP/1PB5/3NK3/Rq3BR1 b - - 0 36", "h8h7"),
]

RARE = [
    ("convert", "5r2/3P1k2/1b6/1P1R4/7P/3K4/8/8 w - - 5 48"),
    ("convert", "6k1/6p1/8/p7/P7/2P4r/2R1K3/8 b - - 1 39"),
    ("convert", "8/8/p6p/1p2kp2/1P2b2P/P3K1P1/2B5/8 w - - 4 53"),
    ("convert", "7r/p5k1/6p1/1p1p2p1/3P1pP1/P7/1P1K1P2/7R w - - 1 32"),
    ("convert", "8/5p2/4p3/1k1pP1pp/r2P4/R1PK4/6PP/8 w - - 0 30"),
    ("convert", "8/8/p2r3p/P2k1p2/5P2/4K3/2R3P1/8 w - - 1 42"),
    ("convert", "1rr2k2/5p2/7p/2bP1Pq1/3BR3/pP5R/P4Q2/7K b - - 2 41"),
    ("convert", "8/7k/p5pp/1p6/3b4/P1B4P/1P4PK/8 b - - 1 40"),
    ("king", "8/1R6/6pk/1P3p1p/5P1P/P4K2/1r3p2/8 w - - 0 41"),
    ("king", "8/7p/3k4/p6P/1p2K3/8/PP6/8 w - - 0 43"),
    ("king", "8/7k/5pp1/4p1p1/6P1/4PK1P/5P2/8 w - - 0 47"),
    ("king", "4k3/8/5PKp/2p3pP/p7/8/PP6/8 b - - 4 40"),
    ("king", "8/8/7p/p1k3p1/5pP1/1P1K1P1P/8/8 b - - 7 49"),
    ("king", "7K/8/4kp2/p1p5/P1P2pP1/5P2/8/8 w - - 1 44"),
    ("king", "8/8/3k4/1p2p1p1/1P2K2p/5P1P/6P1/8 b - - 3 41"),
    ("king", "8/8/5p2/1ppk3p/5p1P/P1P2P2/1K4P1/8 b - - 2 70"),
    ("attack", "3rkb1r/p3pppp/5q2/3b4/8/3Q1P2/PPP3PP/2KR1B1R w k - 0 14"),
    ("attack", "r1bqkr2/1pp3pN/p1np4/4pp2/2B1n3/P1NPK2P/1PP3P1/R1BQ3R b q - 1 11"),
    ("attack", "r1b1kb1r/1pqpnppp/p1n1p3/8/2BNP3/P1N5/1PP2PPP/R1BQK2R w KQkq - 1 8"),
    ("attack", "r2r4/pp3k2/2p1p2Q/4qp1R/2P5/5BPP/1P3P2/6K1 w - - 2 27"),
    ("attack", "1r2r1k1/Q1p3pp/6q1/2nb4/8/5P2/PPP3PP/RNB2RK1 b - - 0 19"),
    ("attack", "2r5/pp3kpQ/2p2r1p/8/3P2bq/2P5/PPBB2P1/5RK1 w - - 2 27"),
    ("attack", "3r4/p4R2/1pb2Pp1/n1p1Qqkp/8/P7/1P4PP/6RK w - - 3 33"),
    ("attack", "2b5/8/3p1Pk1/2p1p1P1/2q1P2K/2P5/6Q1/8 w - - 0 43"),
    ("combination", "r4rk1/1p4pn/p1b1q1Np/2p5/2P5/4P3/PPQB1PPP/3R1RK1 b - - 0 19"),
    ("defend", "r3kb1r/1p1b1p2/p2ppp1p/2q2P1B/2nNP3/2N3R1/P1PQ2PP/1R4K1 w kq - 1 20"),
    ("defend", "r4rk1/3b2pp/8/p3QB2/1pqP4/8/PP4PP/4RRK1 b - - 0 22"),
    ("defend", "8/pQ3pkp/4r3/2BNb1p1/8/1R4q1/P5P1/5K2 b - - 2 34"),
    ("defend", "8/1k4p1/1p2pp1p/1K1P4/4P3/8/5PPP/8 b - - 0 31"),
    ("defend", "8/8/2n2k2/P3pP2/4K2P/8/8/8 w - - 2 49"),
    ("defend", "8/pb6/1p6/PP3k2/2P1p3/4K2p/5P2/7B w - - 2 42"),
    ("defend", "8/8/3R2p1/6K1/1p4P1/k2b4/8/8 b - - 1 46"),
    ("defend", "2r5/p2r1kp1/1p1P2p1/2P2p2/1P1R1P2/6N1/P4KP1/8 w - - 1 33"),
    ("draw", "rqb1k2r/3p2p1/p3pp2/1p3P1p/4P1P1/P1N5/BP2Q1P1/2R2R1K b kq - 0 20"),
    ("draw", "3r1rk1/ppp2pp1/1b6/7p/4P1R1/2Pq3P/P5P1/R1Q1B2K w - - 0 24"),
    ("draw", "7R/bP4k1/3q2p1/1Q1p1p2/4p3/7P/6P1/r4N1K w - - 3 40"),
    ("draw", "r5k1/p4p1p/6p1/3P4/4q3/4R3/PP2K3/R2Q4 b - - 1 27"),
    ("draw", "8/5p1k/5Ppb/2p3P1/qp6/8/KB5Q/8 w - - 5 59"),
    ("draw", "rk5r/1p2b1p1/4Bn1p/p7/7q/1P3P2/P1Q4P/2R4K w - - 0 33"),
    ("draw", "r1b4r/ppp3k1/4P2p/7p/3PQ3/b7/Pq3PPP/n2KR3 w - - 0 23"),
    ("draw", "8/5k1p/p3p2r/1p1pPp2/5P2/P1P1P2R/6P1/6K1 b - - 0 42"),
    ("promote", "1bK5/pP3p2/B7/1P4p1/6k1/8/7p/8 w - - 0 44"),
    ("promote", "3r3Q/Pk2b3/6p1/2p2p2/1p6/3q4/1PnP1PPP/R1B3K1 w - - 0 25"),
    ("promote", "1Rb5/p4p1k/3q1Pp1/2p5/3p1p1P/5P2/PQ4P1/6K1 b - - 1 48"),
    ("promote", "6k1/8/2P3rp/1p1pp3/1P1N4/2P2Pn1/1P4P1/1K3n2 w - - 0 36"),
    ("level_win", "1r5r/pp1qppk1/6pp/2p1P3/2B3n1/3P4/P1P1NPQ1/1R4K1 w - - 1 22"),
    ("level_win", "1Q2n1kr/B4ppp/q7/bb6/4p3/2P1P3/P3NPPP/RN2K2R b KQ - 0 21"),
    ("level_win", "r2qr1k1/p1p1bpBp/2pp1n2/8/4P1bP/6Q1/PPP2PP1/RN2K2R w KQ - 2 13"),
    ("level_win", "r3kbnr/ppp1qppp/2n5/1B1pP3/5B2/4PQ2/PPP2PPP/RN2K2R b KQkq - 2 7"),
    ("pinned_guard", "2kr3r/ppp3pp/6q1/3Pnp2/1PPQp3/7P/P3NPP1/R4RK1 b - - 2 17"),
    ("pinned_guard", "r4rk1/1p1n1pp1/p3p2p/3PPn2/4pP1q/8/PP2B1PP/R2Q1RBK b - - 0 18"),
    ("pinned_guard", "2k3rr/ppp2p1p/2b5/4Nn2/3pPQ2/1P4Pq/P1PP1P1P/RN3RK1 b - - 0 18"),
    ("pinned_guard", "4Q3/6bk/6pp/pp6/8/P2qBP1P/1P6/K1R5 b - - 0 37"),
    ("intermezzo", "r4rk1/pp3ppp/4bq2/3p4/1P1Nn3/P5P1/4PPBP/R2Q1RK1 b - - 0 16"),
    ("intermezzo", "3r1b1R/1pq3p1/p1k3P1/3p4/1P1Q4/4PP2/P1P5/2K5 w - - 0 27"),
    ("intermezzo", "r1b2r2/pp1n2k1/2p3pp/4Np2/2BP4/8/PP4PP/2KRR3 w - - 1 19"),
    ("intermezzo", "8/pp5p/8/3N4/2P5/1P1bR2P/P2r1k1K/8 b - - 2 43"),
    ("also", "6k1/5pb1/p6p/1p4P1/8/5BQ1/PP1q1PP1/6K1 b - - 0 36"),
    ("also", "5rk1/1p5p/p2Ppq2/4n1p1/1P1QP1N1/P6P/5Pp1/3R2K1 b - - 1 33"),
    ("also", "r1r3k1/3n1ppp/p2bpnb1/8/P1pN3P/2N1P3/3PBPq1/B1RQK2R w K - 0 17"),
    ("also", "r3k2r/ppp1bpp1/3pp2p/8/2PPN2q/7P/PP2QPP1/R3K2R w KQkq - 1 14"),
    ("opposition", "8/5p1p/1p6/2k1pP2/4K2P/5PP1/8/8 b - - 2 41"),
    ("opposition", "8/5p2/8/3k4/5KPP/Pb6/8/8 w - - 1 56"),
    ("sacrifices", "2r4r/pp2q1k1/2pp1bp1/5p2/2P2N2/1P4PP/P1Q3K1/3R1R2 w - - 1 27"),
    ("sacrifices", "3rk3/1p1n1p2/4p2p/p1Bb2p1/6PP/1P1BRP2/r7/4R1K1 w - - 2 30"),
    ("sacrifices", "3r1rk1/pbp1qppp/1p2pb2/8/2PB2Q1/5P1N/PP2KP1P/3R2R1 w - - 7 20"),
    ("sacrifices", "r3brk1/1pqnb2p/p3ppp1/2ppN2Q/5P2/1P1PP1R1/PBPN2PP/R5K1 w - - 0 16"),
    ("winning_plan", "8/2k4p/p3P3/1p3p2/1P1Kp3/P7/6PP/8 w - - 0 37"),
    ("winning_plan", "8/p7/6pp/2kp1p2/5P2/1P1K3P/P5P1/8 w - - 0 36"),
    ("winning_plan", "8/1p6/p3p3/4k2p/P1Pr2pP/1P2R1P1/4K3/8 b - - 3 43"),
    ("winning_plan", "8/1pp5/p2k1p2/1P6/P1KPP3/8/8/8 w - - 0 42"),
    # Desktop 3895242: their threat goes to the square the best move takes first, said by their
    # piece ("they are preparing e6" named the answer, e6).
    ("vgdbX", "1rbqkb1r/3npppp/2B5/p2QP3/8/5N2/PPP2nPP/R1B1K2R w KQk - 0 12"),
]

# `tests/test_situation.py` (desktop fc30fa7): 29.exf4 in Johner-Nimzowitsch, material given
# for their attack in a lost game (the line as UCI, the score as the test reads it).
UCI_TESTS = [
    (
        "6rk/3b1prq/1p1p1n2/p1pP1n1p/P1P1pp2/2P1P1PP/R6K/2BQ1NRB w - - 0 29",
        "e3f4 f6g4 h3g4 h5h4 h1e4 h4g3 h2g2 f5h4 g2h3 d7g4",
        0.055,
    ),
]


def facts_json(facts: D.PositionFacts) -> dict:
    return facts.model_dump()


def annotation_json(mark) -> dict:
    return {
        "kind": mark.kind,
        "squares": list(mark.squares),
        "color": mark.color,
        "opacity": mark.opacity,
        "label": mark.label,
    }


def analysis_json(analysis: Analysis) -> dict:
    """The Analysis exactly as the desktop was fed it, for the Kotlin to rebuild."""
    return {
        "lines": [
            {
                "moves": list(line.moves),
                "depth": line.depth,
                "expected_score": line.expected_score,
                "score_cp": line.score_cp,
                "mate": line.mate,
            }
            for line in analysis.lines
        ],
        "decisive": analysis.decisive,
        "only_move": analysis.only_move,
        "threats": list(analysis.threats),
        "traps": [
            {
                "moves": list(line.moves),
                "depth": line.depth,
                "expected_score": line.expected_score,
                "score_cp": line.score_cp,
                "mate": line.mate,
            }
            for line in analysis.traps
        ],
        # The recaptures the best line declines (desktop a1f7f81).
        "defences": [
            {
                "moves": list(line.moves),
                "depth": line.depth,
                "expected_score": line.expected_score,
                "score_cp": line.score_cp,
                "mate": line.mate,
            }
            for line in analysis.defences
        ],
        # What a quiet best move threatens (desktop 29ae423).
        "threatens": [
            {
                "moves": list(line.moves),
                "depth": line.depth,
                "expected_score": line.expected_score,
                "score_cp": line.score_cp,
                "mate": line.mate,
            }
            for line in analysis.threatens
        ],
        "tags": [
            {"name": tag.name, "squares": list(tag.squares), "move": tag.move}
            for tag in analysis.tags
        ],
        "opening": analysis.opening,
        "in_book": analysis.in_book,
    }


def square_or_none(square: chess.Square | None) -> str | None:
    return None if square is None else chess.square_name(square)


def tag_json(tag: MotifTag | None) -> dict | None:
    if tag is None:
        return None
    return {"name": tag.name, "squares": list(tag.squares), "move": tag.move}


def idea_json(idea: IDEA.Idea) -> dict:
    """Every field of the idea, squares by name and moves in UCI."""
    return {
        "goal": idea.goal,
        "first": None if idea.first is None else idea.first.uci(),
        "line": [m.uci() for m in idea.line],
        "mate_in": idea.mate_in,
        "gain": idea.gain,
        "won": idea.won,
        "blow": tag_json(idea.blow),
        "blow_move": None if idea.blow_move is None else idea.blow_move.uci(),
        "blow_board": None if idea.blow_board is None else idea.blow_board.fen(),
        "blow_at": idea.blow_at,
        "gain_ply": idea.gain_ply,
        "preparation": list(idea.preparation),
        "starts_with": idea.starts_with,
        "target": square_or_none(idea.target),
        "threat": None if idea.threat is None else idea.threat.uci(),
        "threatened": idea.threatened,
        "prepared": idea.prepared,
        "follow": None if idea.follow is None else idea.follow.uci(),
        "flight": [chess.square_name(s) for s in idea.flight],
        "checks": idea.checks,
        "level": idea.level,
        "down": idea.down,
        "draw": idea.draw,
        "ending": idea.ending,
        "opposition": idea.opposition,
        "runner": square_or_none(idea.runner),
        "pinned_guard": square_or_none(idea.pinned_guard),
        "intermezzo": idea.intermezzo,
        "also": idea.also,
        "sacrifices": [[ply, kind] for ply, kind in idea.sacrifices],
        "winning": idea.winning,
        # The best line's expected score for the player, and whether what the line takes is
        # won back (the parity audit ck-09: neither was recorded).
        "score": idea.score,
        "back": idea.back,
    }


def words_json(idea: IDEA.Idea, board: chess.Board) -> dict:
    """The idea in the coach's words, straight from `idea_words` (the facts add the plan)."""
    out = {
        "pattern": W.pattern(idea),
        "goal": W.goal(idea),
        "mechanism": W.mechanism(idea),
        "question": W.question(idea, None),
        "nudge": W.nudge(idea),
        "target": W.target(idea, board),
        "piece": W.piece(idea, board),
        "why": W.why(idea, board, F.describe_tactic),
    }
    # With the board, as the facts ask them: a line that does not clear its own path is asked
    # about differently (`clears_path`; the parity audit ck-09). Written only where it differs;
    # absent, it is the same as without.
    question, nudge = W.question(idea, None, board), W.nudge(idea, board)
    if question != out["question"]:
        out["question_board"] = question
    if nudge != out["nudge"]:
        out["nudge_board"] = nudge
    return out


def played_moves(board: chess.Board, best: str) -> list[str]:
    """The moves a lesson is said after, as UCI: every legal capture onto the better move's
    square by another piece -- the right capture with the other piece -- and the first legal
    move."""
    out: list[str] = []
    try:
        good = board.parse_san(best)
    except ValueError:
        good = None
    if good is not None:
        for move in board.legal_moves:
            if move.to_square == good.to_square and move.from_square != good.from_square:
                if board.is_capture(move) and move.uci() not in out:
                    out.append(move.uci())
    first = next(iter(board.legal_moves), None)
    if first is not None and first.uci() not in out:
        out.append(first.uci())
    return out


def templates_json(
    facts: D.PositionFacts, board: chess.Board, allowed: str | None = None
) -> dict:
    """The offline coach's words at each rung, and the lesson after a costly move."""
    levels = list(DisclosureLevel.__members__.values())
    levels = sorted(set(levels), key=int)
    best = facts.candidates[0] if facts.candidates else (facts.line[0] if facts.line else "e4")
    opener = COST_YOU[0].format(best=best)
    return {
        "idea": idea_template(facts),
        "focus": focus_template(facts),
        "move": move_template(facts),
        "line": line_template(facts),
        # With the board, each move of the line with what it does (desktop d8c687c).
        "line_board": line_template(facts, board),
        "hints": {str(int(level)): hint_template(facts, level) for level in levels},
        "hints_board": {str(int(level)): hint_template(facts, level, board) for level in levels},
        "best": best,
        # The lesson after a costly move, and as the answer to "why was that bad?" says it when
        # the model gives nothing (`Coach`'s fallback: "{best} was better", desktop 5e80cdf).
        "lessons": [
            {"opener": opener, "text": lesson_template(facts, best, opener)}
            for opener in (
                COST_YOU[0].format(best=best),
                FINE_TIGHTER[1].format(best=best),
                f"{best} was better",
            )
        ],
        # The lesson after the move played, with the position it was played from (desktop
        # 268990e): the right capture with the other piece is taught as that.
        "lessons_played": [
            {
                "opener": opener,
                "played": played,
                "allowed": said,
                "text": lesson_template(
                    facts, best, opener, board=board, played=played, allowed=said
                ),
            }
            for played in played_moves(board, best)
            for said in ((None, allowed) if allowed else (None,))
        ],
        # Each rung's words checked at every level: the scrubbing of the visible facts and the
        # piece-word rule see every sentence the templates can say.
        "leaks": [
            {
                "rung": int(rung),
                "level": int(level),
                "leaks": D.leak_check(hint_template(facts, rung), facts, level),
            }
            for rung in levels
            for level in levels
        ],
    }


def analysis_for(board: chess.Board, line_ucis: list[str], decisive: bool, only: bool) -> Analysis:
    """An analysis of the shape the engine hands over, built from a known line."""
    moves = []
    work = board.copy(stack=False)
    for uci in line_ucis:
        move = chess.Move.from_uci(uci)
        if move not in work.legal_moves:
            break
        moves.append(uci)
        work.push(move)
    lines = [
        Line(moves=moves, depth=18, expected_score=human_expected_score(120, None), score_cp=120)
    ]
    # A second, slightly worse line, so the near-equal set has something to filter.
    others = [m for m in board.legal_moves if not moves or m.uci() != moves[0]]
    if others:
        lines.append(
            Line(
                moves=[others[0].uci()],
                depth=18,
                expected_score=human_expected_score(115, None),
                score_cp=115,
            )
        )
    threats: list[str] = []
    if not board.is_check():
        passed = board.copy(stack=False)
        passed.push(chess.Move.null())
        first = next(iter(passed.legal_moves), None)
        if first is not None:
            threats.append(first.uci())
    pv = [chess.Move.from_uci(u) for u in moves]
    tags = tags_for_position(board, pv=pv)
    return Analysis(
        game_id="g",
        fen=board.fen(),
        ply=0,
        lines=lines,
        decisive=decisive,
        only_move=only,
        threats=threats,
        tags=tags,
    )


def test_analysis(fen: str, line: str, score: float, mate: int | None, threats, decisive) -> Analysis:
    """`tests/test_idea.py`'s `analysis`: the SAN line written out as UCI."""
    board = chess.Board(fen)
    moves = []
    for san in line.split():
        move = board.parse_san(san)
        moves.append(move.uci())
        board.push(move)
    return Analysis(
        game_id="g",
        fen=fen,
        ply=0,
        lines=[Line(moves=moves, depth=20, expected_score=score, mate=mate)],
        decisive=decisive,
        only_move=False,
        threats=list(threats),
    )


def case_for(
    board: chess.Board,
    analysis: Analysis,
    *,
    deep: bool = True,
    source: str = "synth",
    marks_deep: bool | None = None,
) -> dict:
    us = board.turn
    phase = phase_of(board)
    facts = F.facts_from_analysis(board, analysis, last_move="e4", user=us)
    idea = IDEA.read(board, analysis)

    moves = list(board.legal_moves)
    sample = moves[:8]
    best = moves[0] if moves else None

    levels = sorted(set(DisclosureLevel.__members__.values()), key=int)
    leaks = []
    for text in (LEAK_TEXTS if deep else []):
        for level in levels:
            leaks.append(
                {
                    "text": text,
                    "level": int(level),
                    "said": None,
                    "leaks": D.leak_check(text, facts, level),
                }
            )
        leaks.append(
            {
                "text": text,
                "level": int(DisclosureLevel.IDEA),
                "said": "what about Nxf7 and the knight?",
                "leaks": D.leak_check(
                    text, facts, DisclosureLevel.IDEA, said="what about Nxf7 and the knight?"
                ),
            }
        )
    # The facts' own sentences at each level: a goal or a mechanism that names what a line
    # wins is scrubbed, the player's own piece is not.
    for key in ("goal", "mechanism", "question", "target", "piece", "why", "nudge"):
        text = getattr(facts, key)
        if not text:
            continue
        for level in levels:
            leaks.append(
                {"text": text, "level": int(level), "said": None, "leaks": D.leak_check(text, facts, level)}
            )

    marks = []
    for level in (levels if (deep if marks_deep is None else marks_deep) else []):
        for trigger in (Trigger.STUCK, Trigger.POST_MOVE):
            played = board.san(best) if best is not None and trigger is Trigger.POST_MOVE else None
            marks.append(
                {
                    "level": int(level),
                    "trigger": trigger.value,
                    "played": played,
                    "severity": "mistake" if played else None,
                    "marks": [
                        annotation_json(m)
                        for m in A.annotations_for(
                            facts,
                            level,
                            board,
                            trigger=trigger,
                            played_san=played,
                            severity="mistake" if played else None,
                        )
                    ],
                }
            )

    verdicts = []

    class First:
        """An rng whose every draw is the first choice, as the test's pick is."""

        def choice(self, seq):  # noqa: ANN001, ANN202
            return list(seq)[0]

    class Last:
        """An rng whose every draw is the last choice."""

        def choice(self, seq):  # noqa: ANN001, ANN202
            return list(seq)[-1]

    class Unshuffled:
        """A bag's rng that leaves the wordings in their order: the Kotlin's Bags { }."""

        def shuffle(self, seq):  # noqa: ANN001, ANN202
            return None

    for move in (sample[:4] if deep else []):
        found = V.lesson(board, move, analysis.tags, phase=facts.phase, threats=facts.threats)
        for label in ("best", "great", "excellent", "good", "book", "brilliant", None):
            openers = V.OPENERS.get(label or "", V.OPENERS["good"])
            last = openers[0] + "." if label == "best" else None
            choices = [o for o in openers if last is None or not last.startswith(o)] or openers
            taught = [found[0]] if found is not None and label == "good" else []

            def variants(sentence: str, kind: str | None) -> dict:
                """A player of 1800 and up, one below 1400, and the live coach's bags with their
                shuffle fixed (the parity audit TG-12), written only where they say something:
                `strong` where it differs from the sentence; the plain words drawn last, beside
                the same draw without them, where the two differ (the first wordings are all
                plain); the bags for the praise of a best and a good move."""
                out: dict = {}

                def said(**kw) -> list:  # noqa: ANN003
                    return list(
                        V.verdict(
                            board, move, label, analysis.tags, phase=facts.phase,
                            threats=facts.threats, taught=taught, last=last, **kw,
                        )
                    )

                strong = said(rng=First(), strong=True)
                if strong != [sentence, kind]:
                    out["strong"] = strong
                drawn, plain = said(rng=Last()), said(rng=Last(), plain=True)
                if plain != drawn:
                    out["last_drawn"], out["plain"] = drawn, plain
                if label in ("best", "good"):
                    out["bagged"] = said(bags=Bags(rng=Unshuffled()))
                return out

            sentence, kind = V.verdict(
                board,
                move,
                label,
                analysis.tags,
                phase=facts.phase,
                threats=facts.threats,
                taught=taught,
                last=last,
                rng=First(),  # type: ignore[arg-type]
            )
            verdicts.append(
                {
                    "move": move.uci(),
                    "label": label,
                    "last": last,
                    "taught": taught,
                    "choices": choices,
                    "lesson": None if found is None else [found[0], found[1]],
                    "sentence": sentence,
                    "kind": kind,
                    **variants(sentence, kind),
                }
            )

    return {
        "source": source,
        "fen": board.fen(),
        # The moves that led here, when the board knows them: the story's "taken back" and "win
        # back" read their last move (desktop 10fdd21), so the Kotlin replays them.
        "history": (
            {"root": board.root().fen(), "moves": [m.uci() for m in board.move_stack]}
            if board.move_stack
            else None
        ),
        "phase": phase,
        "analysis": analysis_json(analysis),
        "facts": facts_json(facts),
        "idea": idea_json(idea),
        "words": words_json(idea, board),
        "templates": templates_json(facts, board, W.allowed(board, analysis, PLAYED)),
        # What the move that left this position allowed, read by the same idea reader.
        "allowed": W.allowed(board, analysis, PLAYED),
        # The desktop's story checks on the three hints (desktop tooling; recorded, not ported).
        "story": story_of(board, analysis, idea, facts),
        # The traps ranked as the first hint wants them, each said with the best move.
        "trap_reads": trap_reads(board, analysis, idea),
        "material_us": F.material_text(board, us),
        "material_them": F.material_text(board, not us),
        "position_text": F.position_text(board),
        "ascii": F.ascii_board(board),
        "inventory_lines": I.inventory_lines(board, us),
        "line_outcome": O.line_outcome(board, list(analysis.lines[0].moves), us)
        if analysis.lines
        else None,
        # The best line move by move, as the lines block and the line directive give it.
        "line_notes": E.line_notes(board, list(analysis.lines[0].moves))
        if analysis.lines
        else None,
        # The same, each move said as the player's or the opponent's (desktop 4365dfa).
        "line_notes_player": E.line_notes(board, list(analysis.lines[0].moves), player=us)
        if analysis.lines
        else None,
        "effects": [
            {
                "move": m.uci(),
                "described": F.describe_move(board, m),
                "hazards": E.describe_effects(board, m, hazards=True),
                "no_hazards": E.describe_effects(board, m, hazards=False),
                "text": E.effects_text(board, m),
            }
            for m in sample
        ],
        "leaks": leaks,
        "marks": marks,
        "verdicts": verdicts,
        "mentions": [
            {
                "text": text,
                "marks": [annotation_json(m) for m in A.mentions_in(text, board)],
            }
            for text in (LEAK_TEXTS[:6] if deep else [])
        ],
    }


def policy_cases() -> list[dict]:
    """Scripted runs of the interjection policy, so the whole state machine is compared: two
    unprompted words per position, each after its own stuck spell, counted from `stuck_from`
    (since desktop 2026-10-01 nothing comes before the first: no question at a decisive
    moment); and (the parity audit, 2026-09-29) a threshold the clock's share sets between
    the floor and the setting. CoachPrimitivesTest runs the same script, step for step."""
    from sidecoach.events import Label, MoveAssessment, Severity

    def assessment(severity: Severity, label: Label | None) -> MoveAssessment:
        return MoveAssessment(
            game_id="g", ply=1, uci="e2e4", expected_loss=0.2, severity=severity,
            best=[], label=label,
        )

    runs = []
    for control in P.TimeControl:
        for quiet in (False, True):
            policy = P.InterjectionPolicy(
                settings=P.PolicySettings(quiet=quiet), time_control=control
            )
            steps: list[dict] = []

            def note(what: str, trigger) -> None:
                steps.append(
                    {
                        "step": what,
                        "trigger": None if trigger is None else trigger.value,
                        "reason": policy.last_reason,
                        "threshold": policy.stuck_threshold(),
                        "scramble": policy.in_scramble,
                        "stuck_from": policy.stuck_from,
                    }
                )

            policy.new_position(2, 0.0, user_to_move=True)
            note("new position", None)
            note("tick 10", policy.tick(10.0))
            note("tick 60", policy.tick(60.0))
            policy.said(2, 60.0)
            note("tick soon after said", policy.tick(80.0))
            note("tick after said", policy.tick(120.0))
            policy.said(2, 120.0)
            note("tick after two", policy.tick(300.0))
            policy.new_position(2, 300.0, user_to_move=True)
            note("same ply again", policy.tick(400.0))
            policy.new_position(4, 100.0, user_to_move=True)
            policy.said(4)
            note("tick after said without a time", policy.tick(150.0))
            policy.said(3, 150.0)
            note("said of another ply", policy.tick(151.0))
            policy.new_position(12, 200.0, user_to_move=True)
            policy.set_remaining(20_000)
            note("tick in scramble", policy.tick(400.0))
            policy.set_remaining(600_000)
            note("tick with clock", policy.tick(500.0))
            policy.new_position(14, 500.0, user_to_move=False)
            note("opponent to move", policy.tick(600.0))
            policy.new_position(16, 600.0, user_to_move=True)
            policy.covered = True
            note("covered", policy.tick(700.0))
            policy.covered = False
            note("blunder", policy.on_user_move_assessed(assessment(Severity.BLUNDER, Label.BLUNDER)))
            note("fine move", policy.on_user_move_assessed(assessment(Severity.NONE, Label.BEST)))
            note("miss", policy.on_user_move_assessed(assessment(Severity.NONE, Label.MISS)))
            # The clock's share between the floor and the setting: five minutes gives 24 s;
            # 100 s, above the scramble, still hints at the floor.
            policy.new_position(22, 900.0, user_to_move=True)
            policy.set_remaining(300_000)
            note("five minutes, before the share", policy.tick(920.0))
            note("five minutes, past the share", policy.tick(925.0))
            policy.set_remaining(100_000)
            policy.new_position(24, 1000.0, user_to_move=True)
            note("100 s, at the floor", policy.tick(1021.0))
            runs.append({"control": control.value, "quiet": quiet, "steps": steps})
    return runs


def cached_analyses(fens: set[str]) -> dict[str, Analysis]:
    """The puzzle bench's engine analyses of `fens`, read from its cache (never written)."""
    out: dict[str, Analysis] = {}
    if not CACHE.exists():
        return out
    with CACHE.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line the running bench is still writing
            if row.get("fen") in fens:
                out[row["fen"]] = Analysis.model_validate(row["analysis"])
    return out


def theme_positions() -> list[tuple[str, str, chess.Board]]:
    """(puzzle id, theme, board) for up to PER_THEME puzzles of every theme: the puzzle's
    position after the opponent's first move, and the one after the player's first reply."""
    if not THEMES.exists():
        return []
    with THEMES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    wanted: list[tuple[str, str, chess.Board]] = []
    for row in rows:
        board = chess.Board(row["FEN"])
        moves = row["Moves"].split()
        for index, uci in enumerate(moves[:3]):
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                break
            board.push(move)
            if index in (0, 2):
                wanted.append((row["PuzzleId"], row["Themes"], board.copy(stack=False)))
    return wanted


def masters_positions() -> list[tuple[str, int, Analysis]]:
    """(game, index, analysis) of every position the master games' facts measured: the start
    and the position after each move, as the eval harness recorded the engine's analysis."""
    out: list[tuple[str, int, Analysis]] = []
    for path in sorted(MASTERS.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = [data.get("start_analysis")] + [s.get("analysis") for s in data.get("script", [])]
        for index, row in enumerate(rows):
            if row:
                out.append((path.stem, index, Analysis.model_validate(row)))
    return out


def story_of(
    board: chess.Board, analysis: Analysis, idea: IDEA.Idea, facts: D.PositionFacts
) -> dict[str, bool]:
    """The desktop's story checks on the three hints of `facts`."""
    if not analysis.lines:
        return {}
    texts = {"idea": idea_template(facts), "focus": focus_template(facts), "move": move_template(facts)}
    pv = [m for m in (F._to_move(u) for u in analysis.lines[0].moves) if m is not None]
    checks, _ = story_checks(board, pv, idea, texts)
    return dict(checks)


def trap_reads(board: chess.Board, analysis: Analysis, idea: IDEA.Idea) -> dict | None:
    """The Analysis's traps ranked (their first moves), and each one's words with the best
    move, as the facts would say them."""
    if not analysis.traps:
        return None
    best = F._to_move(analysis.lines[0].moves[0]) if analysis.lines and analysis.lines[0].moves else None
    top = analysis.lines[0].expected_score if analysis.lines else None
    ranked = SIT.ranked_traps(analysis.traps, idea.target)
    return {
        "ranked": [list(t.moves) for t in ranked],
        "words": [SIT.trap_words(board, t, idea.target, top, best) for t in ranked],
    }


def story_category(facts: D.PositionFacts, idea: IDEA.Idea, full: dict[str, int]) -> str | None:
    """The first beat of the story (desktop 10fdd21) these facts tell that is still short of
    MASTERS_STORY."""
    situation = facts.situation or ""
    why = facts.why or ""
    beats = [
        ("story:plain", "can simply be taken" in situation and ". After that," not in situation),
        ("story:free", ". After that," in situation),
        ("story:checker", "gives check" in situation),
        ("story:queening", "from queening" in situation and idea.goal != "promote"),
        ("story:prize", "The prize is" in situation),
        (
            "story:how",
            any(
                word in situation
                for word in ("The first move ", "It starts by giving up", "A check comes first")
            ),
        ),
        ("story:reason", "keeps" in why and (" -- it is" in why or ": it is" in why)),
        ("story:win back", "win back" in (facts.goal or "")),
    ]
    for name, told in beats:
        if told and full.get(name, 0) < MASTERS_STORY:
            return name
    return None


def rung_category(board: chess.Board, analysis: Analysis, facts: D.PositionFacts) -> str | None:
    """What the rungs of desktop 268990e read here, if anything: a win that comes a move
    later, a mate too long to read out whole, a forcing capture's target, or a pawn threat of
    the best move that gives way to its own idea."""
    idea = IDEA.read(board, analysis)
    if idea.goal == "mate" and idea.mate_in and idea.mate_in > W.MATE_READ:
        return "rung:long mate"
    if idea.goal == "win" and idea.gain_ply > 0:
        return "rung:win later"
    top = analysis.lines[0]
    pv = [m for m in (F._to_move(u) for u in top.moves) if m is not None]
    best = pv[0] if pv and pv[0] in board.legal_moves else None
    if best is None or idea.goal != "plan":
        return None
    if board.is_capture(best):
        taken = board.piece_at(best.to_square)
        if taken is not None and facts.target == (
            f"their {chess.piece_name(taken.piece_type)} on {chess.square_name(best.to_square)}"
        ):
            return "rung:capture target"
    threat = PL.threat(board, best, pv, top.expected_score)
    if (
        threat is not None
        and threat.kind == "win"
        and threat.gain < F.THREAT_MIN
        and facts.plan is not None
        and facts.plan not in PL.THREAT_NAME.values()
    ):
        return "rung:pawn threat yields"
    return None


def plan_category(facts: D.PositionFacts) -> str:
    """Which plan the facts carry, read off their goal and mechanism: a threat the opponent
    prepares, a line plan's kind, a threat's kind (and what a trade is for), or the best
    move's own idea ("move") or none."""
    mechanism = facts.mechanism or ""
    goal = facts.goal or ""
    if "your opponent is preparing mate" in goal:
        return "prepared:mate"
    if "your opponent is preparing to win material" in goal:
        return "prepared:win"
    for kind, reason in PL.PLAN_REASON.items():
        if facts.plan and mechanism.endswith(": " + reason):
            return f"plan:{kind}"
    for kind, reason in PL.THREAT_REASON.items():
        if mechanism == f"{PL.THREAT_NAME[kind]}: {reason}":
            return f"threat:{kind}"
    for why, reason in PL.TRADE_REASON.items():
        if mechanism == f"{PL.THREAT_NAME['trade']}: {reason}":
            return f"threat:trade:{why}"
    return "move" if facts.plan else "none"


def with_traps(engine: Engine | None, board: chess.Board, analysis: Analysis) -> Analysis:
    """The analysis with the traps, the declined recaptures and what a quiet best move
    threatens, as the annotator would search them for it, in its order (`trap_lines`,
    `defence_lines` from desktop a1f7f81, `aim_lines` from 29ae423)."""
    if engine is None:
        return analysis
    return analysis.model_copy(
        update={
            "traps": trap_lines(engine, board, analysis.lines),
            "defences": defence_lines(engine, board, analysis.lines),
            "threatens": aim_lines(engine, board, analysis.lines),
        }
    )


def opening_cases(book: OpeningBook | None) -> list[tuple[chess.Board, Analysis]]:
    """Every position of OPENING_GAMES, with the book's `in_book` and the opening's name as
    the annotator gives them, the line the game's own next moves."""
    if book is None:
        return []
    out: list[tuple[chess.Board, Analysis]] = []
    for game in OPENING_GAMES:
        board = chess.Board()
        sans = game.split()
        for index, san in enumerate(sans):
            rest = []
            work = board.copy()
            for later in sans[index:]:
                move = work.parse_san(later)
                rest.append(move.uci())
                work.push(move)
            analysis = analysis_for(board.copy(stack=False), rest, False, False)
            found = book.identify(list(board.move_stack)) if board.move_stack else None
            name = getattr(found, "name", None) if found is not None else None
            eco = getattr(found, "eco", None) if found is not None else None
            opening = (f"{eco} {name}" if eco else str(name)) if name is not None else None
            in_book = bool(board.move_stack) and book.in_book(list(board.move_stack))
            analysis = analysis.model_copy(update={"opening": opening, "in_book": in_book})
            out.append((board.copy(stack=False), analysis))
            board.push_san(san)
    return out


def desktop_tests() -> dict:
    """Every call the desktop's knowledge tests make to the facts, effects, disclosure and
    annotation functions, with the desktop's answers (the parity audit ck-05, ck-06, ck-07,
    TG-08): `tests/test_effects_legal.py`, `tests/test_facts.py` and
    `tests/test_coach_annotations.py`, each run as pytest would. (`tests/test_coach_policy.py`
    holds the policy's tests alone since the consolidations of 2026-09-28/29, its facts tests
    now `tests/test_facts.py`'s, and made no call here: it is not run.) The functions are
    wrapped where the tests find them -- their own imports, and the modules a test imports from
    inside a test."""
    import importlib

    from desktop_tests import desktop_tests as each_test

    from sidecoach.analysis import phase as PH

    out: dict[str, list] = {"calls": [], "ran": []}
    current = {"test": ""}

    def colour(c) -> bool | None:  # noqa: ANN001
        return None if c is None else c == chess.WHITE

    shows = {
        "facts_from_analysis": lambda board, analysis, *, last_move=None, user=None, pick=None: {
            "fen": board.fen(),
            "analysis": {**analysis_json(analysis), "fen": analysis.fen},
            "last_move": last_move,
            "user": colour(user),
        },
        "visible_facts": lambda facts, level: {"facts": facts_json(facts), "level": int(level)},
        "describe_move": lambda board, move: {"fen": board.fen(), "move": move.uci()},
        "describe_tactic": lambda board, name, squares, best_move=None: {
            "fen": board.fen(), "name": name, "squares": list(squares),
            "best": None if best_move is None else best_move.uci(),
        },
        "describe_effects": lambda board, move, *, limit=3, hazards=True: {
            "fen": board.fen(), "move": move.uci(), "limit": limit, "hazards": hazards
        },
        "line_notes": lambda board, moves, *, limit=8, player=None: {
            "fen": board.fen(), "moves": list(moves), "limit": limit, "player": colour(player)
        },
        "line_spoken": lambda board, moves, *, limit=6: {"fen": board.fen(), "moves": list(moves), "limit": limit},
        "region_of": lambda squares: {"squares": list(squares)},
        "phase_of": lambda board: {"fen": board.fen()},
        "material_text": lambda board, perspective=None: {"fen": board.fen(), "perspective": colour(perspective)},
        "annotations_for": lambda facts, level, board, *, trigger, played_san=None, severity=None: {
            "facts": facts_json(facts), "level": int(level), "fen": board.fen(),
            "trigger": trigger.value, "played_san": played_san, "severity": severity,
        },
        "mentions_in": lambda text, board: {"text": text, "fen": board.fen()},
    }

    def shown(fn: str, result):  # noqa: ANN001, ANN202
        if fn == "facts_from_analysis":
            return facts_json(result)
        if fn == "visible_facts":
            return {k: (list(v) if isinstance(v, (list, tuple)) else v) for k, v in result.items()}
        if fn in ("annotations_for", "mentions_in"):
            return [annotation_json(mark) for mark in result]
        return result

    depth = {"n": 0}

    def wrap(real, fn: str):  # noqa: ANN001, ANN202
        def call(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            # Only the test's own calls: the ones these functions make of each other are
            # replayed inside them.
            depth["n"] += 1
            try:
                result = real(*args, **kwargs)
            finally:
                depth["n"] -= 1
            if depth["n"] == 0:
                out["calls"].append(
                    {"test": current["test"], "fn": fn, "args": shows[fn](*args, **kwargs), "out": shown(fn, result)}
                )
            return result

        call.__wrapped_real__ = real  # type: ignore[attr-defined]
        return call

    # Where a test may find each function: its own module, and the source modules it imports
    # from inside a test body.
    homes = {
        "facts_from_analysis": [F],
        "visible_facts": [D],
        "describe_move": [F],
        "describe_tactic": [F],
        "describe_effects": [E],
        "line_notes": [E],
        "line_spoken": [E],
        "region_of": [F],
        "phase_of": [F, PH],
        "material_text": [F],
        "annotations_for": [A],
        "mentions_in": [A],
    }
    for module_name in (
        "tests.test_effects_legal",
        "tests.test_facts",
        "tests.test_coach_annotations",
    ):
        tests = importlib.import_module(module_name)
        saved: list[tuple[object, str, object]] = []
        for fn, modules in homes.items():
            real = getattr(modules[0], fn)
            for module in [tests, *modules]:
                if hasattr(module, fn) and getattr(module, fn) is real:
                    saved.append((module, fn, real))
                    setattr(module, fn, wrap(real, fn))
        try:
            # Each case of a table as well: skipped, the effects and descriptions tables went
            # unreplayed once the desktop's tests became tables (2026-09-28).
            for name, test in each_test(tests):
                current["test"] = f"{module_name}.{name}"
                test()  # the desktop's own assertions hold as the calls are recorded
                out["ran"].append(current["test"])
        finally:
            for module, fn, real in saved:
                setattr(module, fn, real)
    # The played move's colour by every severity, and none: an inaccuracy is "warn" (TG-08),
    # an empty severity is "none"'s (the parity audit cc-07).
    ann = importlib.import_module("tests.test_coach_annotations")
    current["test"] = "severities"
    for severity in ("none", "inaccuracy", "mistake", "blunder", "", None, "unheard"):
        wrap(A.annotations_for, "annotations_for")(
            ann.FACTS, DisclosureLevel.LINE, ann.BOARD, trigger=Trigger.POST_MISTAKE,
            played_san="Bxc6", severity=severity,
        )
    out["ran"].append(current["test"])
    return out


def main() -> None:
    rng = random.Random(20260922)
    path = find_stockfish(load_settings())
    assert path is not None, "no Stockfish for the desktop"
    engine = Engine(path=path, threads=1, hash_mb=64, multipv=6)
    engine.open()
    book = OpeningBook.load(DESKTOP / "data" / "openings")
    cases: list[dict] = []
    if PUZZLES.exists():
        with PUZZLES.open(newline="", encoding="utf-8") as handle:
            for row in list(csv.DictReader(handle))[:120]:
                board = chess.Board(row["FEN"])
                moves = row["Moves"].split()
                for index, uci in enumerate(moves):
                    move = chess.Move.from_uci(uci)
                    if move not in board.legal_moves:
                        break
                    if index % 2 == 1:
                        # The whole sweep -- every leak text at every level, every mark, every
                        # verdict -- on the first forty positions; the rest carry the facts,
                        # the inventory and the effects, which is where the variety is.
                        decisive = rng.random() < 0.5
                        only = rng.random() < 0.3
                        analysis = analysis_for(board, moves[index:], decisive, only)
                        cases.append(case_for(board, analysis, deep=len(cases) < 40))
                    board.push(move)
    for _ in range(25):
        board = chess.Board()
        for ply in range(rng.randint(2, 40)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
            if ply % 7 == 0:
                rest = [m.uci() for m in list(board.legal_moves)[:4]]
                decisive = rng.random() < 0.5
                only = rng.random() < 0.3
                analysis = analysis_for(board, rest, decisive, only)
                cases.append(case_for(board, analysis, deep=len(cases) % 37 == 0))
    synthesised = len(cases)

    # Opening theory, from the book.
    booked = 0
    for board, analysis in opening_cases(book):
        cases.append(case_for(board, analysis, deep=booked % 6 == 0, source="opening"))
        booked += 1

    # `tests/test_story.py`: the six puzzles read by hand, and the two of why the obvious
    # defence fails (desktop a1f7f81), their hints and the story checks.
    told_stories = [
        (
            name,
            fen,
            Analysis(
                game_id="",
                fen=fen,
                ply=0,
                lines=[Line(moves=moves, depth=18, expected_score=score) for moves, score in story_lines],
                decisive=False,
                only_move=False,
                threats=list(threats),
            ),
        )
        for name, fen, story_lines, threats in STORY_TESTS
    ] + [
        (
            name,
            fen,
            Analysis(
                game_id="",
                fen=fen,
                ply=0,
                lines=[Line(moves=moves, depth=18, expected_score=score) for moves, score in story_lines],
                decisive=True,
                only_move=True,
                defences=[
                    Line(moves=moves, depth=14, expected_score=score, mate=mate)
                    for moves, score, mate in made_defences
                ],
                traps=[
                    Line(moves=moves, depth=14, expected_score=score, mate=mate)
                    for moves, score, mate in made_traps
                ],
                threatens=[
                    Line(moves=moves, depth=14, expected_score=score, mate=mate)
                    for moves, score, mate in made_threats
                ],
            ),
        )
        for name, fen, story_lines, made_defences, made_traps, made_threats in DEFENCE_STORY_TESTS
    ]
    stories = []
    for name, fen, analysis in told_stories:
        board = chess.Board(fen)
        facts = F.facts_from_analysis(board, analysis, user=board.turn)
        texts = {
            "idea": idea_template(facts),
            "focus": focus_template(facts),
            "move": move_template(facts),
        }
        pv = [chess.Move.from_uci(u) for u in analysis.lines[0].moves]
        checks, _ = story_checks(board, pv, IDEA.read(board, analysis), texts)
        assert all(checks.values()), (name, checks)
        stories.append(
            {
                "name": name,
                "fen": fen,
                "analysis": analysis_json(analysis),
                "texts": texts,
                "checks": dict(checks),
                "told": all(checks.values()),
            }
        )
        cases.append(case_for(board, analysis, deep=True, source=f"test_story: {name}"))

    # The move rung's words for the one king step no check follows (the parity audit ck-09).
    for fen, best in NO_CHECK_FOLLOWS:
        board = chess.Board(fen)
        cases.append(case_for(board, analysis_for(board, [best], False, False), deep=True, source="no check follows"))

    # Traps made by hand: the ranking and the words.
    trap_tests = []
    for name, fen, best_uci, target_name, top, made in TRAP_TESTS:
        board = chess.Board(fen)
        traps = [Line(moves=moves, depth=14, expected_score=score, mate=mate) for moves, score, mate in made]
        target = chess.parse_square(target_name) if target_name else None
        best = chess.Move.from_uci(best_uci)
        ranked = SIT.ranked_traps(traps, target)
        trap_tests.append(
            {
                "name": name,
                "fen": fen,
                "best": best_uci,
                "target": target_name,
                "top": top,
                "traps": [
                    {"moves": moves, "depth": 14, "expected_score": score, "score_cp": None, "mate": mate}
                    for moves, score, mate in made
                ],
                "ranked": [list(t.moves) for t in ranked],
                "words": [SIT.trap_words(board, t, target, top, best) for t in ranked],
                "words_without_best": [SIT.trap_words(board, t, target, top) for t in ranked],
            }
        )

    # `tests/test_situation.py`'s lines written as UCI.
    for fen, line, score in UCI_TESTS:
        board = chess.Board(fen)
        analysis = Analysis(
            game_id="",
            fen=fen,
            ply=0,
            lines=[Line(moves=line.split(), depth=18, expected_score=score)],
            decisive=False,
            only_move=False,
        )
        cases.append(case_for(board, analysis, deep=True, source="test_situation"))

    # The desktop's idea tests, their lines written out.
    for fen, line, score, mate, threats, decisive in IDEA_TESTS:
        board = chess.Board(fen)
        analysis = test_analysis(fen, line, score, mate, threats, decisive)
        cases.append(case_for(board, analysis, deep=True, source="test_idea"))

    # Every theme's puzzles, with the engine's own analysis.
    positions = theme_positions()
    analyses = cached_analyses({board.fen() for _, _, board in positions} | {fen for _, fen in RARE})
    per_theme: dict[str, int] = {}
    seen: set[str] = set()
    themed = 0
    for puzzle, themes, board in positions:
        fen = board.fen()
        if fen in seen or fen not in analyses:
            continue
        names = themes.split()
        if all(per_theme.get(name, 0) >= PER_THEME for name in names):
            continue
        for name in names:
            per_theme[name] = per_theme.get(name, 0) + 1
        seen.add(fen)
        # The marks at every level on one in four: the target and the piece at the focus rung.
        cases.append(
            case_for(
                board,
                with_traps(engine, board, analyses[fen]),
                deep=themed % 8 == 0,
                marks_deep=themed % 4 == 0,
                source=f"lichess {puzzle}: {themes}",
            )
        )
        themed += 1

    # The rarer goals, from the whole cache.
    rare = 0
    for label, fen in RARE:
        if fen in seen or fen not in analyses:
            continue
        seen.add(fen)
        cases.append(
            case_for(
                chess.Board(fen),
                with_traps(engine, chess.Board(fen), analyses[fen]),
                deep=rare % 6 == 0,
                marks_deep=rare % 2 == 0,
                source=f"cache: {label}",
            )
        )
        rare += 1

    # The master games: the plan of the best line and the threat of the best move, where the
    # facts carry them. A position is read whole only when its line could carry one (the cheap
    # read of lines[0] finds a plan or a threat) or it falls on the stride.
    masters = masters_positions()
    random.Random(20260924).shuffle(masters)
    by_plan: dict[str, int] = {}
    mastered = 0
    for index, (game, ply, analysis) in enumerate(masters):
        if not analysis.lines or not analysis.lines[0].moves:
            continue
        board = chess.Board(analysis.fen)
        if board.fen() in seen:
            continue
        top = analysis.lines[0]
        pv = [m for m in (F._to_move(u) for u in top.moves) if m is not None]
        best = pv[0] if pv else None
        could = best is not None and best in board.legal_moves and (
            PL.read(board, pv, top.expected_score) is not None
            or PL.threat(board, best, pv, top.expected_score) is not None
            or IDEA._threat(board, analysis.threats, best)[2] is not None
            or board.is_capture(best)
            or (top.mate is not None and top.mate > W.MATE_READ)
            or F._line_gain(board, top.moves) >= 1
        )
        tempted = bool(tempting(board, analysis.lines))
        if not could and not tempted and index % MASTERS_STRIDE:
            continue
        if tempted and by_plan.get("trap", 0) < MASTERS_TRAPS:
            analysis = with_traps(engine, board, analysis)
        facts = F.facts_from_analysis(board, analysis, last_move="e4", user=board.turn)
        kind = story_category(facts, IDEA.read(board, analysis), by_plan) or (
            "trap"
            if facts.trap
            else rung_category(board, analysis, facts) or plan_category(facts)
        )
        if kind.startswith("story:"):
            quota = MASTERS_STORY
        elif kind == "trap":
            quota = MASTERS_TRAPS
        elif kind.startswith("rung:"):
            quota = MASTERS_RUNGS
        elif kind.startswith("prepared:"):
            quota = MASTERS_PREPARED
        elif kind.startswith(("plan:", "threat:")):
            quota = MASTERS_PER_KIND
        else:
            quota = MASTERS_PER_KIND // 2
        if by_plan.get(kind, 0) >= quota:
            continue
        by_plan[kind] = by_plan.get(kind, 0) + 1
        seen.add(board.fen())
        cases.append(
            case_for(
                board,
                analysis,
                deep=mastered % 12 == 0,
                marks_deep=mastered % 3 == 0,
                source=f"masters {game} #{ply}: {kind}",
            )
        )
        mastered += 1
    engine.close()

    persons = [
        {"text": text, "side": side, "out": third_person(text, side=side)}
        for text in PERSON_TEXTS
        for side in ("player", "opponent")
    ]
    unlocks = []
    for start in sorted(set(DisclosureLevel.__members__.values()), key=int):
        for unlock in D.Unlock:
            state = D.Disclosure(level=start)
            result = state.apply(unlock)
            unlocks.append(
                {
                    "start": int(start),
                    "unlock": unlock.value,
                    "result": None if result is None else int(result),
                    "level": int(state.level),
                }
            )
    controls = [
        {"initial": initial, "increment": increment, "control": P.time_control_of(initial, increment).value}
        for initial, increment in [
            (None, 0), (600_000, 0), (300_000, 0), (180_000, 0), (120_000, 1_000),
            (60_000, 0), (180_000, 2_000), (900_000, 10_000), (0, 0),
        ]
    ]
    standings = [
        {"score": score, "you": F.standing_words(score), "third": F.standing_words(score, third_person=True)}
        for score in (0.0, 0.1, 0.15, 0.2, 0.2000001, 0.3, 0.38, 0.3800001, 0.5, 0.62, 0.7, 0.79,
                      0.7999999, 0.8, 0.85, 0.9, 1.0)
    ]

    goals: dict[str, int] = {}
    for case in cases:
        goals[case["idea"]["goal"]] = goals.get(case["idea"]["goal"], 0) + 1

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coach.json").write_text(
        json.dumps(
            {
                "cases": cases,
                "persons": persons,
                "unlocks": unlocks,
                "controls": controls,
                "standings": standings,
                "policy": policy_cases(),
                "stories": stories,
                "trap_tests": trap_tests,
            }
        ),
        encoding="utf-8",
    )
    size = (OUT / "coach.json").stat().st_size
    # The desktop's knowledge tests on their own: coach.json is parsed whole on a phone's heap.
    (OUT / "knowledge.json").write_text(json.dumps({"tests": desktop_tests()}), encoding="utf-8")
    print(
        f"{len(cases)} positions ({synthesised} synthesised, {len(IDEA_TESTS)} idea tests, "
        f"{themed} Lichess by theme over {len(per_theme)} themes, {rare} rarer goals, "
        f"{mastered} master positions), "
        f"{len(persons)} rewrites "
        f"-> {OUT / 'coach.json'} ({size / 1e6:.1f} MB)"
    )
    print("goals:", dict(sorted(goals.items(), key=lambda kv: -kv[1])))
    print("master positions by plan:", dict(sorted(by_plan.items())))
    rungs: dict[str, int] = {}
    for case in cases:
        idea = case["idea"]
        if idea["goal"] == "win" and idea["gain_ply"] > 0:
            rungs["win later"] = rungs.get("win later", 0) + 1
        if idea["goal"] == "mate" and (idea["mate_in"] or 0) > W.MATE_READ:
            rungs["long mate"] = rungs.get("long mate", 0) + 1
        for lesson in case["templates"]["lessons_played"]:
            if " -- take with the " in lesson["text"]:
                rungs["recapture lesson"] = rungs.get("recapture lesson", 0) + 1
        rungs["lessons played"] = rungs.get("lessons played", 0) + len(
            case["templates"]["lessons_played"]
        )
    print("rungs:", dict(sorted(rungs.items())))
    first: dict[str, int] = {}
    for case in cases:
        f = case["facts"]
        for key in ("stakes", "key", "trap", "situation"):
            if f.get(key):
                first[key] = first.get(key, 0) + 1
        if f.get("choices"):
            first["choices"] = first.get("choices", 0) + 1
        if case["analysis"]["traps"]:
            first["analysis traps"] = first.get("analysis traps", 0) + 1
        if case["analysis"]["in_book"]:
            first["in book"] = first.get("in book", 0) + 1
        if case["allowed"]:
            first["allowed"] = first.get("allowed", 0) + 1
    print("first hint:", dict(sorted(first.items())), f"({booked} opening positions)")
    failed: dict[str, int] = {}
    for case in cases:
        for name, ok in case["story"].items():
            if not ok:
                failed[name] = failed.get(name, 0) + 1
    beats: dict[str, int] = {}
    for case in cases:
        text = case["facts"].get("situation") or ""
        for beat, said in (
            ("plain", "can simply be taken"),
            ("free", ". After that,"),
            ("checker", "gives check"),
            ("queening", "from queening"),
            ("prize", "The prize is"),
            ("how", "The first move "),
            ("take it with the right piece", "Take it with the right piece"),
            ("clearly worse", "you are left clearly worse"),
        ):
            if said in text:
                beats[beat] = beats.get(beat, 0) + 1
        if "win back" in (case["facts"].get("goal") or ""):
            beats["win back"] = beats.get("win back", 0) + 1
        if " -- it is the only move" in (case["facts"].get("why") or "") or ": it is the only move" in (
            case["facts"].get("why") or ""
        ):
            beats["reason"] = beats.get("reason", 0) + 1
    print("story beats:", dict(sorted(beats.items())))
    print(f"story checks failing (desktop tooling, recorded): {dict(sorted(failed.items()))}")
    print(f"{len(stories)} story tests, all told; {len(trap_tests)} trap tests")


if __name__ == "__main__":
    main()
