"""Engine facts for every ply of the corpus games, computed once and cached.

The evaluation set renders each game through the live coach's own Conversation, which
needs what the live pipeline gives it: the Analysis of each position and the MoveAssessment
of each move. Those come from Stockfish through the same GameAnnotator the panel uses, and
they are the slow part (about a second a ply), so they are computed here once per game and
stored as JSON events under data/eval/facts/<source>/<game>.json. Everything downstream
(the corpus, the judge, ICR) replays them and never runs the engine.

Sources:
  masters      data/eval/masters/<book>/<game>.json   (from masters.py)
  recordings   data/recordings/<session>/events.jsonl (the live sessions: moves and takebacks
               as they happened) or truth.pgn + chesscom.json (the older, hand-truthed ones)
  bots         data/bots/games/*.pgn (Nova bot games, tools/bots/play.py: the amateur games
               the other sources lack), every game of every file

Run:  uv run python tools/eval/facts.py masters [--depth 14] [--seconds 2.0] [--limit N]
      uv run python tools/eval/facts.py recordings
      uv run python tools/eval/facts.py bots [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import chess
import chess.pgn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sidecoach.analysis.annotator import GameAnnotator  # noqa: E402
from sidecoach.analysis.engine import Engine  # noqa: E402
from sidecoach.analysis.score import Thresholds  # noqa: E402
from sidecoach.cli import _load_book, _load_tagger  # noqa: E402
from sidecoach.config import load_settings, require_stockfish  # noqa: E402

OUT = ROOT / "data" / "eval" / "facts"
USER_HANDLE = "nosegazm"  # Tai on chess.com


def game_from_events(path: Path) -> dict | None:
    """A live session's game as played: the user's side, the moves in order, and takebacks
    as they happened (a takeback withdraws plies; the withdrawn moves stay in the script
    so the conversation shows them as it did live)."""
    user_side = None
    start_fen = chess.STARTING_FEN
    script: list[dict] = []
    asks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "user_side" in e and "start_fen" in e:
                if user_side is not None:
                    break  # a second game in the session: keep the first
                user_side, start_fen = e["user_side"], e["start_fen"]
            elif "san" in e and "fen_before" in e:
                script.append(
                    {
                        "kind": "move",
                        **{
                            k: e[k]
                            for k in (
                                "ply",
                                "san",
                                "uci",
                                "fen_before",
                                "fen_after",
                                "by_user",
                                "think_ms",
                            )
                        },
                    }
                )
            elif "plies" in e and "fen" in e and "game_id" in e:
                script.append(
                    {"kind": "takeback", "plies": e["plies"], "fen": e["fen"], "ts": e.get("ts")}
                )
            elif "result" in e and "final_fen" in e:
                script.append({"kind": "ended", "result": e["result"], "reason": e.get("reason")})
    if user_side is None or not script:
        return None
    asks = live_asks(path.with_name("debug.jsonl"))
    return {"user_side": user_side, "start_fen": start_fen, "script": script, "asks": asks}


def live_asks(debug: Path) -> list[dict]:
    """The questions the player actually asked in the session, each with the ply of the
    last move seen before it (the debug log's game lines carry the ply) and whether the game
    was already over."""
    asks: list[dict] = []
    if not debug.exists():
        return asks
    ply = 0
    over = False
    with debug.open(encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = e.get("text") or ""
            if not isinstance(text, str):
                continue
            if e.get("chan") == "game":
                data = e.get("data") or {}
                if isinstance(data.get("ply"), int):
                    ply = data["ply"]
                if text.startswith("game over"):
                    over = True
                if text.startswith("new game"):
                    ply, over = 0, False
            if text.startswith("you: "):
                asks.append({"t": e.get("t"), "ply": ply, "after_end": over, "text": text[5:]})
    return asks


def game_from_pgn(path: Path, user_side: str) -> dict:
    game = chess.pgn.read_game(path.open(encoding="utf-8"))
    assert game is not None
    return game_from_game(game, user_side)


def game_from_game(game: chess.pgn.Game, user_side: str) -> dict:
    board = game.board()
    script = []
    for ply, move in enumerate(game.mainline_moves(), start=1):
        by_user = (board.turn == chess.WHITE) == (user_side == "white")
        before = board.fen()
        san = board.san(move)
        board.push(move)
        script.append(
            {
                "kind": "move",
                "ply": ply,
                "san": san,
                "uci": move.uci(),
                "fen_before": before,
                "fen_after": board.fen(),
                "by_user": by_user,
                "think_ms": None,
            }
        )
    result = game.headers.get("Result", "*")
    if result != "*":
        script.append(
            {
                "kind": "ended",
                "result": result,
                "reason": "checkmate" if board.is_checkmate() else None,
            }
        )
    return {
        "user_side": user_side,
        "start_fen": game.board().fen(),
        "script": script,
        "asks": [],
        "headers": dict(game.headers),
    }


def sources(kind: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    if kind == "masters":
        for path in sorted((ROOT / "data" / "eval" / "masters").glob("*/*.json")):
            item = json.loads(path.read_text(encoding="utf-8"))
            board = chess.Board()
            script = []
            for p in item["plies"]:
                move = chess.Move.from_uci(p["uci"])
                by_user = (
                    board.turn == chess.WHITE
                )  # the eval asks about White's moves; corpus.py may flip
                board.push(move)
                script.append(
                    {
                        "kind": "move",
                        "ply": p["ply"],
                        "san": p["san"],
                        "uci": p["uci"],
                        "fen_before": p["fen_before"],
                        "fen_after": board.fen(),
                        "by_user": by_user,
                        "think_ms": None,
                    }
                )
            if item.get("result", "*") != "*":
                script.append(
                    {
                        "kind": "ended",
                        "result": item["result"],
                        "reason": "checkmate" if board.is_checkmate() else None,
                    }
                )
            out.append(
                (
                    f"{item['book']}/{path.stem}",
                    {
                        "user_side": "white",
                        "start_fen": chess.STARTING_FEN,
                        "script": script,
                        "asks": [],
                        "headers": item.get("headers", {}),
                    },
                )
            )
    elif kind == "recordings":
        for d in sorted((ROOT / "data" / "recordings").iterdir()):
            if (d / "events.jsonl").exists():
                game = game_from_events(d / "events.jsonl")
                if game:
                    out.append((d.name, game))
            elif (d / "truth.pgn").exists():
                # The user's side from the PGN: Tai's chess.com handle is White or Black.
                text = (d / "truth.pgn").read_text(encoding="utf-8")
                white = re.search(r'\[White "([^"]*)"\]', text)
                black = re.search(r'\[Black "([^"]*)"\]', text)
                if black and USER_HANDLE.lower() in black.group(1).lower():
                    side = "black"
                elif white and USER_HANDLE.lower() in white.group(1).lower():
                    side = "white"
                else:
                    raise RuntimeError(
                        f"{d.name}: cannot tell the user's side from {white} / {black}"
                    )
                out.append((d.name, game_from_pgn(d / "truth.pgn", side)))
    elif kind == "bots":
        # Both sides are coached (the sweep plays each); White is the facts' nominal user.
        for path in sorted((ROOT / "data" / "bots" / "games").glob("*.pgn")):
            with path.open(encoding="utf-8") as fh:
                i = 0
                while (game := chess.pgn.read_game(fh)) is not None:
                    out.append((f"{path.stem}__{i:03d}", game_from_game(game, "white")))
                    i += 1
    return out


def compute(name: str, game: dict, ann: GameAnnotator, out_path: Path) -> None:
    ann.reset()
    t0 = time.time()
    gid = name.replace("/", "-")
    start = chess.Board(game["start_fen"])
    facts: dict = {
        "name": name,
        "user_side": game["user_side"],
        "start_fen": game["start_fen"],
        "headers": game.get("headers", {}),
        "asks": game.get("asks", []),
        "start_analysis": json.loads(ann.analysis_event(gid, start, 0).model_dump_json()),
        "script": [],
    }
    # The board carries the moves that got here: the opening book and the "book" label
    # need the history, and a position built from its FEN alone has none.
    board = chess.Board(game["start_fen"])
    for step in game["script"]:
        if step["kind"] == "takeback":
            facts["script"].append(step)
            while board.move_stack and board.fen() != step["fen"]:
                board.pop()
            if board.fen() != step["fen"]:
                board = chess.Board(step["fen"])
            continue
        if step["kind"] != "move":
            facts["script"].append(step)
            continue
        if board.fen() != step["fen_before"]:
            board = chess.Board(step["fen_before"])
        before = board.copy()
        move = chess.Move.from_uci(step["uci"])
        assessment, analysis = ann.assess(gid, before, move, step["ply"])
        board.push(move)
        facts["script"].append(
            {
                **step,
                "assessment": json.loads(assessment.model_dump_json()),
                "analysis": json.loads(analysis.model_dump_json()),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(facts, ensure_ascii=False), encoding="utf-8")
    n = sum(1 for s in game["script"] if s["kind"] == "move")
    print(f"{name}: {n} plies in {time.time() - t0:.0f}s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["masters", "recordings", "bots"])
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    settings = load_settings()
    engine = Engine(
        path=require_stockfish(settings),
        threads=args.threads,
        hash_mb=1024,
        multipv=6,
        max_seconds=args.seconds,
    )
    ann = GameAnnotator(
        engine=engine,
        depth=args.depth,
        multipv=6,
        thresholds=Thresholds.from_settings(settings.coach),
        tagger=_load_tagger(),
        book=_load_book(settings),
        with_threats=True,
    )
    todo = sources(args.source)
    if args.limit:
        todo = todo[: args.limit]
    done = 0
    try:
        for name, game in todo:
            out_path = OUT / args.source / f"{name.replace('/', '__')}.json"
            if out_path.exists() and not args.force:
                continue
            compute(name, game, ann, out_path)
            done += 1
    finally:
        engine.close()
    print(f"computed {done} of {len(todo)} games; cache at {OUT / args.source}")


if __name__ == "__main__":
    main()
