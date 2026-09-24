"""The story checks and the annotators' answer key over the 199 annotated master games.

1. The story checks (`tools/puzzles/story_checks.py`) at every position of the games: the same
   rules the Lichess puzzles are held to, on positions a grandmaster's game actually reached.
2. The annotators' refutations as an answer key: every alternative a note marks "?" or "??",
   or brings in with "not"/"Not", at a position of the game itself -- the move the note
   annotates or the reply to it, legal there, and not the move played -- and
   that the engine confirms (TRAP_DROP or more below the best line, or into mate). At each, does
   the coach's first hint warn of it: the move named (as a trap, or "not yet"), or the stakes
   said sharp ("only one move holds", "only two moves keep ...")?

The traps are searched here (the stored analyses predate them), one Stockfish per worker.

  uv run --no-sync python tools/eval/story_masters.py --workers 12 --out story_masters.md
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools" / "puzzles"))

import chess  # noqa: E402
from story_checks import STORY_CHECKS  # noqa: E402

MASTERS = ROOT / "data" / "eval" / "masters"
FACTS = ROOT / "data" / "eval" / "facts" / "masters_v2"
# A move in a note: "14 hxg4?", "15...Bxg4?", "not 16 Qxf2", with the mark after it.
_NOTE_MOVE = re.compile(
    r"(?P<not>\b[Nn]ot\s+)?(?P<n>\d{1,3})\s*(?P<dots>\.\.\.|…|\.)?\s*"
    r"(?P<san>O-O-O|O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=?[QRBN])?)[+#]?(?P<mark>\?\?|\?!|!\?|\?|!!|!)?"
)
_SHARP = (
    "Only ",
    "You're winning, but only",
    "You're in check, and only",
    "You're in check but winning, and only",
)

_engine = None


def games() -> list[tuple[Path, Path]]:
    out = []
    for f in sorted(FACTS.glob("*.json")):
        book, game = f.stem.split("__", 1)
        m = MASTERS / book / f"{game}.json"
        if m.exists():
            out.append((f, m))
    return out


def _init() -> None:
    global _engine
    from sidecoach.analysis.engine import Engine
    from sidecoach.config import find_stockfish, load_settings

    settings = load_settings()
    path = find_stockfish(settings)
    assert path is not None, "no Stockfish"
    _engine = Engine(
        path=path,
        threads=1,
        hash_mb=64,
        multipv=settings.engine.multipv,
        pv_plies=settings.engine.pv_plies,
        score_model=settings.engine.score_model,
        max_seconds=settings.engine.max_seconds,
    )
    _engine.open()


def refuted(master: dict, positions: dict[tuple[int, bool], dict]) -> list[dict]:
    """The alternatives the notes refute at positions of the game itself."""
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for ply in master["plies"]:
        note = ply.get("annotation") or ""
        here = int(ply["ply"])
        for m in _NOTE_MOVE.finditer(note):
            mark = m.group("mark") or ""
            if not (mark in ("?", "??") or m.group("not")):
                continue
            number = int(m.group("n"))
            black = m.group("dots") in ("...", "…")
            # The position the note is about: the move it annotates, or the reply to it --
            # never one deep in a side line that shares a move number with the game.
            if 2 * (number - 1) + (2 if black else 1) not in (here, here + 1):
                continue
            step = positions.get((number, black))
            if step is None:
                continue
            board = chess.Board(step["fen_before"])
            try:
                move = board.parse_san(m.group("san"))
            except ValueError:
                continue
            if move.uci() == step["uci"]:
                continue
            key = (step["fen_before"], move.uci())
            if key in seen:
                continue
            seen.add(key)
            start = max(0, m.start() - 60)
            out.append(
                {
                    "step": step,
                    "move": move.uci(),
                    "san": board.san(move),
                    "note": note[start : m.end() + 90].replace("\n", " "),
                }
            )
    return out


def _texts(board: chess.Board, analysis: object) -> tuple[object, object, dict[str, str]]:
    from sidecoach.coach import idea as idea_of
    from sidecoach.coach.coach import focus_template, idea_template, move_template
    from sidecoach.coach.facts import facts_from_analysis

    facts = facts_from_analysis(board, analysis, user=board.turn)  # type: ignore[arg-type]
    idea = idea_of.read(board, analysis)  # type: ignore[arg-type]
    texts = {
        "idea": idea_template(facts),
        "focus": focus_template(facts),
        "move": move_template(facts),
    }
    return facts, idea, texts


def run_game(pair: tuple[str, str]) -> dict:
    from story_checks import story_checks

    from sidecoach.analysis.traps import TRAP_DEPTH, _worse, trap_lines
    from sidecoach.events import Analysis, Line

    facts_path, master_path = Path(pair[0]), Path(pair[1])
    facts = json.loads(facts_path.read_text(encoding="utf-8"))
    master = json.loads(master_path.read_text(encoding="utf-8"))
    before = Analysis.model_validate(facts["start_analysis"])
    positions: dict[tuple[int, bool], dict] = {}
    analyses: dict[str, Analysis] = {}
    story_rows: list[dict] = []
    for step in [s for s in facts["script"] if s["kind"] == "move"]:
        board = chess.Board(step["fen_before"])
        if before.lines and before.fen.split()[:4] == board.fen().split()[:4]:
            positions[(board.fullmove_number, board.turn == chess.BLACK)] = step
            analyses[step["fen_before"]] = before
            _, idea, texts = _texts(board, before)
            pv = [chess.Move.from_uci(u) for u in before.lines[0].moves]
            checks, notes = story_checks(board, pv, idea, texts)
            story_rows.append(
                {
                    "game": facts_path.stem,
                    "move": f"{board.fullmove_number}{'...' if board.turn == chess.BLACK else '.'}"
                    f"{step['san']}",
                    "checks": checks,
                    "notes": notes,
                    "idea": texts["idea"],
                }
            )
        before = Analysis.model_validate(step["analysis"])

    key_rows: list[dict] = []
    assert _engine is not None
    for alt in refuted(master, positions):
        step = alt["step"]
        board = chess.Board(step["fen_before"])
        analysis = analyses[step["fen_before"]]
        top = analysis.lines[0]
        scored = {line.moves[0]: line for line in analysis.lines if line.moves}
        line = scored.get(alt["move"])
        if line is None:
            after = board.copy(stack=False)
            after.push(chess.Move.from_uci(alt["move"]))
            if after.is_game_over():
                continue
            replies = _engine.analyse(chess.Board(after.fen()), depth=TRAP_DEPTH, multipv=1)
            if not replies:
                continue
            reply = replies[0]
            line = Line(
                moves=[alt["move"], *reply.moves],
                depth=reply.depth,
                expected_score=1.0 - reply.expected_score,
                mate=-reply.mate if reply.mate is not None else None,
            )
        confirmed = _worse(line, top)
        with_traps = analysis.model_copy(
            update={"traps": trap_lines(_engine, board, analysis.lines)}
        )
        _, _, texts = _texts(board, with_traps)
        said = texts["idea"]
        capture = board.is_capture(chess.Move.from_uci(alt["move"]))
        key_rows.append(
            {
                "game": facts_path.stem,
                "move": f"{board.fullmove_number}{'...' if board.turn == chess.BLACK else '.'}",
                "played": step["san"],
                "alt": alt["san"],
                "capture": capture,
                "confirmed": confirmed,
                "drop": round(top.expected_score - line.expected_score, 3),
                "named": re.search(rf"(?<![A-Za-z0-9]){re.escape(alt['san'])}(?![A-Za-z0-9])", said)
                is not None,
                "sharp": said.startswith(_SHARP),
                "note": alt["note"],
                "idea": said,
            }
        )
    return {"story": story_rows, "key": key_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", default=None)
    parser.add_argument("--games", type=int, default=0, help="only the first n games")
    args = parser.parse_args()
    pairs = [(str(f), str(m)) for f, m in games()]
    if args.games:
        pairs = pairs[: args.games]
    story: list[dict] = []
    key: list[dict] = []
    with mp.Pool(args.workers, initializer=_init) as pool:
        for done, result in enumerate(pool.imap_unordered(run_game, pairs), 1):
            story += result["story"]
            key += result["key"]
            print(f"\r{done}/{len(pairs)} games", end="", flush=True)
    print()

    scored: Counter[str] = Counter()
    passed: Counter[str] = Counter()
    failures: dict[str, list[dict]] = defaultdict(list)
    for row in story:
        for name, ok in row["checks"].items():
            scored[name] += 1
            passed[name] += ok
            if not ok:
                failures[name].append(row)
    lines = [f"# Story checks and the annotators' answer key ({len(pairs)} games)", ""]
    lines.append(f"{len(story)} positions read.")
    lines.append("")
    lines.append("| check | positions | failures |")
    lines.append("|---|---|---|")
    for name in STORY_CHECKS:
        if scored[name]:
            lines.append(f"| {name} | {scored[name]} | {scored[name] - passed[name]} |")
    confirmed = [r for r in key if r["confirmed"]]
    captures = [r for r in confirmed if r["capture"]]
    lines += [
        "",
        f"Annotators' refuted alternatives at game positions: {len(key)}; the engine confirms "
        f"{len(confirmed)}.",
        "",
        f"- refuted captures named in the first hint: {sum(r['named'] for r in captures)} of "
        f"{len(captures)}",
        f"- all confirmed refutations warned of (the move named, or the stakes said sharp): "
        f"{sum(r['named'] or r['sharp'] for r in confirmed)} of {len(confirmed)}",
        "",
    ]
    for name, rows in failures.items():
        lines.append(f"## {name} failures ({len(rows)})")
        for row in rows[:12]:
            lines.append(f"- {row['game']} {row['move']}: {row['notes'].get(name, '')[:200]}")
        lines.append("")
    lines.append("## Confirmed refutations not warned of")
    for row in [r for r in confirmed if not (r["named"] or r["sharp"])][:60]:
        lines.append(
            f"- {row['game']} {row['move']} (played {row['played']}; refuted {row['alt']}, "
            f'drop {row["drop"]}): note "{row["note"]}" | coach: {row["idea"][:220]}'
        )
    lines.append("")
    lines.append("## Confirmed refutations warned of")
    for row in [r for r in confirmed if r["named"] or r["sharp"]][:40]:
        lines.append(
            f"- {row['game']} {row['move']} (refuted {row['alt']}): coach: {row['idea'][:220]}"
        )
    text = "\n".join(lines) + "\n"
    print(text[:2500])
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        (Path(args.out).with_suffix(".json")).write_text(
            json.dumps({"story": story, "key": key}, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
