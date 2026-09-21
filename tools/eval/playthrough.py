"""Play one master game through the real coach, from move 1 to the end, coached from one
side, with the model answering at every moment the policy fires: the verdict after each of
the player's moves, a question at decisive moments, a hint when the player is stuck, and
answers to a few questions the player asks. Every answer is logged into the harness, so the
next moment's input carries it.

The output has two parts: the transcript (every move, numbered as the harness numbers it,
with what the coach said after it), then the harness as it stood at the chosen moves.

  uv run python tools/eval/playthrough.py --game mammoth__080-kasparov-karpov --side white
      --coach openai:gpt-5.6-luna@medium --dump 12,30,50 > data/eval/playthrough-x.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from providers import coach_provider  # noqa: E402

from sidecoach.cli import _draft_check  # noqa: E402
from sidecoach.coach import prompt  # noqa: E402
from sidecoach.coach.coach import Coach  # noqa: E402
from sidecoach.coach.policy import InterjectionPolicy, PolicySettings  # noqa: E402
from sidecoach.events import (  # noqa: E402
    Analysis,
    GameEnded,
    GameStarted,
    MoveAssessment,
    MovePlayed,
    Side,
)

FACTS = ROOT / "data" / "eval" / "facts" / "masters"
STUCK_SECONDS = 30


def numbered(step: dict) -> str:
    board = chess.Board(step["fen_before"])
    n = board.fullmove_number
    return f"{n}. {step['san']}" if board.turn == chess.WHITE else f"{n}... {step['san']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--game", required=True, help="facts file stem, e.g. mammoth__080-kasparov-karpov"
    )
    ap.add_argument("--side", choices=["white", "black"], required=True)
    ap.add_argument("--coach", default="openai:gpt-5.6-luna@medium")
    ap.add_argument(
        "--dump", default="", help="plies at which to keep the harness, comma-separated"
    )
    ap.add_argument(
        "--stuck-every", type=int, default=5, help="the player is stuck on every Nth of their turns"
    )
    ap.add_argument("--no-corrections", action="store_true", help="an empty C")
    ap.add_argument(
        "--summarise-every",
        type=int,
        default=0,
        help="fold the earlier dialogue into a summary every N plies (0 = off)",
    )
    ap.add_argument(
        "--summariser", default="openai:gpt-5.6-luna@none", help="the model that folds the dialogue"
    )
    args = ap.parse_args()

    # A master game by default; a recorded game's facts (data/eval/facts/recordings) when
    # the stem is found there instead.
    path = FACTS / f"{args.game}.json"
    if not path.exists():
        path = FACTS.parent / "recordings" / f"{args.game}.json"
    facts = json.loads(path.read_text(encoding="utf-8"))
    moves = [s for s in facts["script"] if s["kind"] == "move"]
    side = Side.WHITE if args.side == "white" else Side.BLACK
    user_colour = chess.WHITE if side == Side.WHITE else chess.BLACK
    dump_at = {int(x) for x in args.dump.split(",") if x.strip()}

    provider = coach_provider(args.coach)
    coach = Coach(
        provider,
        policy=InterjectionPolicy(PolicySettings(stuck_seconds=STUCK_SECONDS)),
        draft_check=_draft_check(),
        summarise_every=args.summarise_every,
        summariser=coach_provider(args.summariser) if args.summarise_every else None,
    )
    coach.sync_summaries = True
    coach.system = prompt.system_message(with_corrections=not args.no_corrections)
    gid = args.game
    ts = 0.0
    t_start = time.time()
    transcript: list[str] = []
    dumps: list[tuple[str, str]] = []

    def drain() -> list[str]:
        coach.speaker.run_pending()
        return [m.text for m in coach.poll()]

    def said(label: str) -> None:
        for text in drain():
            transcript.append(f"        [{label}] COACH: {text}")

    coach.on_event(GameStarted(game_id=gid, user_side=side, start_fen=facts["start_fen"]), ts)
    coach.on_event(Analysis.model_validate({**facts["start_analysis"], "game_id": gid}), ts)
    drain()
    own_turns = 0
    for step in moves:
        by_user = chess.Board(step["fen_before"]).turn == user_colour
        ts += 6
        coach.on_event(
            MovePlayed(
                game_id=gid,
                ply=step["ply"],
                san=step["san"],
                uci=step["uci"],
                fen_before=step["fen_before"],
                fen_after=step["fen_after"],
                by_user=by_user,
                think_ms=None,
            ),
            ts,
        )
        coach.on_event(Analysis.model_validate({**step["analysis"], "game_id": gid}), ts)
        coach.on_event(MoveAssessment.model_validate({**step["assessment"], "game_id": gid}), ts)
        label = step["assessment"].get("label") or step["assessment"]["severity"]
        transcript.append(
            f"{numbered(step):14s} ({'the player' if by_user else 'the opponent'}, {label})"
        )
        said("after the move")
        if not by_user and step is not moves[-1]:
            own_turns += 1
            if own_turns % args.stuck_every == 0:
                ts += STUCK_SECONDS + 15
                coach.tick(ts)
                said("the player is stuck")
            if own_turns % (args.stuck_every * 2) == 3:
                ts += 4
                transcript.append('        PLAYER: "what should I be aiming for here?"')
                coach.ask("what should I be aiming for here?", ts)
                said("answer")
        if (
            by_user
            and step["assessment"].get("severity") not in (None, "none")
            and own_turns % 3 == 0
        ):
            ts += 4
            transcript.append('        PLAYER: "why was that bad?"')
            coach.ask("why was that bad?", ts)
            said("answer")
        if step["ply"] in dump_at:
            dumps.append(
                (f"after {numbered(step)} (ply {step['ply']})", coach.conversation.render())
            )
    if facts["script"] and facts["script"][-1]["kind"] == "ended":
        end = facts["script"][-1]
        ts += 2
        coach.on_event(
            GameEnded(
                game_id=gid,
                result=end["result"],
                reason=end.get("reason"),
                pgn="",
                final_fen=coach.board.fen(),
            ),
            ts,
        )
        transcript.append(f"Game over: {end['result']}")
        said("game over")
        ts += 3
        transcript.append('        PLAYER: "how did I do overall?"')
        coach.ask("how did I do overall?", ts)
        said("answer")

    print(f"# Play-through: {args.game}, coaching {args.side}, coach {args.coach}\n")
    print("# Transcript\n\n```")
    print("\n".join(transcript))
    print("```\n")
    for title, text in dumps:
        print(f"# Harness {title}\n\n```\n{text}\n```\n")
    print(f"# Final harness\n\n```\n{coach.conversation.render()}\n```\n")
    print(
        f"{time.time() - t_start:.0f}s; coach calls {len(transcript)}; failures"
        f" {coach.speaker.failures} (last: {coach.speaker.last_error}); fallbacks"
        f" {coach.speaker.fallbacks}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
