"""Record what the desktop's coach primitives say, for the Kotlin port to be held to.

Everything under `sidecoach/coach/` that is a pure function of the board and the analysis:
who "you" is, the disclosure filter and the leak check, the fact bundle, what a move does,
what stands on the board, where a line leads, the word on a sound move, when the coach may
speak, and the marks that go on the board. The coach reads these out; a port that differs
anywhere says something different.

The positions are the Lichess puzzle sample and random games, with the analysis synthesised
from the puzzle's own solution line so the fact bundle has candidates, threats, a line and
tags to work from -- the same shapes the engine hands it in a real game.

    cd D:/chess-coaching-companion
    uv run --no-sync python D:/sidecoach-android/tools/make_coach_fixtures.py
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

from sidecoach.analysis.motifs import tags_for_position  # noqa: E402
from sidecoach.analysis.phase import phase_of  # noqa: E402
from sidecoach.analysis.score import human_expected_score  # noqa: E402
from sidecoach.coach import annotations as A  # noqa: E402
from sidecoach.coach import disclosure as D  # noqa: E402
from sidecoach.coach import effects as E  # noqa: E402
from sidecoach.coach import facts as F  # noqa: E402
from sidecoach.coach import inventory as I  # noqa: E402
from sidecoach.coach import outcome as O  # noqa: E402
from sidecoach.coach import policy as P  # noqa: E402
from sidecoach.coach import verdicts as V  # noqa: E402
from sidecoach.coach.person import third_person  # noqa: E402
from sidecoach.events import Analysis, DisclosureLevel, Line, Trigger  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"
PUZZLES = DESKTOP / "data" / "puzzles" / "sample.csv"

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


def case_for(board: chess.Board, line_ucis: list[str], rng: random.Random, deep: bool = True) -> dict:
    us = board.turn
    phase = phase_of(board, False)
    decisive = rng.random() < 0.5
    only = rng.random() < 0.3
    analysis = analysis_for(board, line_ucis, decisive, only)
    facts = F.facts_from_analysis(board, analysis, last_move="e4", in_book=False, user=us)

    moves = list(board.legal_moves)
    sample = moves[:8]
    best = moves[0] if moves else None

    levels = list(DisclosureLevel)
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
                "level": int(DisclosureLevel.THEME),
                "said": "what about Nxf7 and the knight?",
                "leaks": D.leak_check(
                    text, facts, DisclosureLevel.THEME, said="what about Nxf7 and the knight?"
                ),
            }
        )

    marks = []
    for level in (levels if deep else []):
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
    for move in (sample[:4] if deep else []):
        for label in ("best", "great", "excellent", "good", "book", "brilliant", None):
            openers = V.OPENERS.get(label or "", V.OPENERS["good"])
            san = board.san(move)
            last = "{san}: best".format(san=san) if label == "best" else None
            choices = [o for o in openers if last is None or not last.startswith(o.format(san=san))]
            choices = choices or openers
            clause = V.what_it_does(board, move, analysis.tags)
            if clause is None:
                sentence = choices[0].format(san=san) + "."
            elif clause == "checkmate":
                sentence = f"{choices[0].format(san=san)} -- checkmate."
            else:
                joiner = " -- " if clause.startswith("the ") else ", "
                sentence = f"{choices[0].format(san=san)}{joiner}{clause}."
            verdicts.append(
                {
                    "move": move.uci(),
                    "label": label,
                    "last": last,
                    "choices": choices,
                    "clause": clause,
                    "sentence": sentence,
                }
            )

    return {
        "fen": board.fen(),
        "phase": phase,
        "line": line_ucis,
        "decisive": decisive,
        "only_move": only,
        "facts": facts_json(facts),
        "material_us": F.material_text(board, us),
        "material_them": F.material_text(board, not us),
        "position_text": F.position_text(board),
        "ascii": F.ascii_board(board),
        "inventory_lines": I.inventory_lines(board, us),
        "inventory_text": I.inventory_text(board, us),
        "line_outcome": O.line_outcome(board, [m for m in analysis.lines[0].moves], us),
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
    """Scripted runs of the interjection policy, so the whole state machine is compared."""
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
                    }
                )

            policy.new_position(2, 0.0, user_to_move=True)
            note("new position", None)
            note("tick 10", policy.tick(10.0))
            note("tick 60", policy.tick(60.0))
            note("analysed quiet", policy.on_position_analysed(2, decisive=False))
            note("analysed decisive", policy.on_position_analysed(2, decisive=True))
            policy.said(2)
            note("tick after said", policy.tick(120.0))
            policy.new_position(4, 100.0, user_to_move=True)
            note("analysed decisive again", policy.on_position_analysed(4, decisive=True))
            policy.new_position(12, 200.0, user_to_move=True)
            note("analysed decisive later", policy.on_position_analysed(12, decisive=True))
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
            runs.append({"control": control.value, "quiet": quiet, "steps": steps})
    return runs


def main() -> None:
    rng = random.Random(20260922)
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
                        cases.append(case_for(board, moves[index:], rng, deep=len(cases) < 40))
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
                cases.append(case_for(board, rest, rng, deep=len(cases) % 37 == 0))

    persons = [
        {"text": text, "side": side, "out": third_person(text, side=side)}
        for text in PERSON_TEXTS
        for side in ("player", "opponent")
    ]
    unlocks = []
    for start in list(DisclosureLevel):
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
    priorities = [
        {
            "in": [t.value for t in order],
            "out": [t.value for t in P.in_priority(list(order))],
        }
        for order in [
            [Trigger.STUCK, Trigger.USER_ASK],
            [Trigger.REVIEW, Trigger.POST_MOVE, Trigger.POST_MISTAKE],
            [Trigger.POST_MOVE, Trigger.POST_MOVE, Trigger.DECISIVE_QUESTION],
            list(Trigger),
        ]
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "coach.json").write_text(
        json.dumps(
            {
                "cases": cases,
                "persons": persons,
                "unlocks": unlocks,
                "controls": controls,
                "priorities": priorities,
                "policy": policy_cases(),
            }
        ),
        encoding="utf-8",
    )
    size = (OUT / "coach.json").stat().st_size
    print(f"{len(cases)} positions, {len(persons)} rewrites -> {OUT / 'coach.json'} ({size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
