"""Score the coach's replies against the evidence: deterministic checks first, then an LM
judge for what needs reading.

Per reply:
  leak          deterministic: the coach's own leak_check at the item's level
  board         deterministic: the board-claim checker (a piece said to attack what it
                does not, a pin that is not there)
  form          deterministic: sentence count, the closing question where required, empty
  completeness  judge: the share of the required facts the reply mentions (Eq. 4, tau)
  correctness   judge: of the required facts it mentions, the share stated right (Eq. 4, Q)
  claim_support judge: the reply is first split into atomic claims (one checkable statement
                each), then each claim is verified on its own with a yes or no against the
                evidence; this is the supported share. Any unsupported claim is a
                fabrication, a hard fail.
  relevance     judge: 0..1, does the reply answer what was asked (under 0.5 is a fail)
  idea_match    judge, items with a master's note: the judge states the note's point in
                one sentence, then answers yes or no: does the reply make that point
  idea_in_facts judge, same items: is that point stated, in any words, in the evidence
                the coach had -- if not, the miss is a gap in the facts, not the coach's

Every judge call is written to the ledger (data/eval/usage/) with its token counts and
the run stops at the token budget.

Used by icr_play.py; every check is one call at reasoning effort none, on the ledger.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chess
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from taxonomy import sentence_count  # noqa: E402

from sidecoach.cli import _draft_check  # noqa: E402
from sidecoach.coach.coach import _after_best, _Request  # noqa: E402
from sidecoach.coach.disclosure import PositionFacts, leak_check  # noqa: E402
from sidecoach.coach.effects import effects_text  # noqa: E402
from sidecoach.coach.facts import ascii_board  # noqa: E402
from sidecoach.coach.inventory import inventory_lines  # noqa: E402
from sidecoach.coach.named import disclosed_frames  # noqa: E402
from sidecoach.events import DisclosureLevel, Trigger  # noqa: E402

JUDGMENTS = ROOT / "data" / "eval" / "judgments"
USAGE = ROOT / "data" / "eval" / "usage.jsonl"
# Parallel runs each append to their own file, so two processes never interleave a line;
# lifetime_usage reads them all.
USAGE_MINE = ROOT / "data" / "eval" / "usage" / f"usage-{os.getpid()}.jsonl"

DECOMPOSE_SYSTEM = (
    "You split a chess coach's reply into atomic claims, each one checkable with a single "
    "yes or no against the record of the game. A claim is one of these kinds and nothing "
    "else: (1) the cost of a move, or that one move was better than another ('it cost "
    "about a tenth of the game', 'c3 was better than Qf3'); (2) what a move does or would do "
    "('Bxe4 takes the knight', 'Rd1 attacks the queen on d7', 'Nf3 pins the bishop'); (3) "
    "the state of a piece or square ('the rook on e7 is loose', 'your king is still in the "
    "centre', 'the knight on e5 is pinned'); (4) material ('you are up a pawn', 'a decisive "
    "material advantage'); (5) who stands better -- always a claim, however it is worded "
    "('the position is about equal', 'White is winning', 'you have a decisive advantage', "
    "'a small edge', 'you are losing'); (6) a threat ('your "
    "opponent threatens Nxf6+'); (7) the plan or idea stated for the position ('the plan is "
    "to push the f-pawn', 'the idea is a discovered attack'); (8) an engine move or line "
    "('c3 was better', 'after Rhd1 Qxc3 the file opens'); (9) the course of the game ('you "
    "made 9 best moves', 'move 14 was your only inaccuracy', 'the opening was the Najdorf', "
    "'the game ended in mate'). One fact per claim, in the reply's own words, its move "
    "names and squares kept -- and never a square, piece or move the reply does not name. "
    "Moves given as what would follow ('Qg7, followed by ...Rdg8 and Rb7+', 'then', "
    "'after') are a line, kind 8, worded as a line ('after Qg7 the line continues ...Rdg8 "
    "Rb7+'), never as moves that were played. "
    "Everything else is NOT a claim and is left out: questions, "
    "advice, instructions, the verdict word on the move just played ('best', 'excellent', "
    "'fine', 'a mistake', 'precisely played' -- that is checked separately), general chess "
    "knowledge not about this game ('the Najdorf is often called the Sozin'), and every "
    "interpretation or appraisal ('spent a tempo', 'more active', 'solid', 'keeps the "
    "initiative', 'a real threat', 'natural', 'the wrong piece', 'exposes the king'), and "
    "every statement of intent, aim, purpose or character ('their aim is to exploit your "
    "position', 'the point was to keep it secure', 'c5 defines the strategic character', "
    "'g3 was the defensive move', 'it gives them a way to create pressure'). When in doubt "
    "whether an arbiter could check a statement against the record, leave it out. A "
    "warning that names a specific tactic ('be alert to the fork on e3') is a "
    "claim of kind 3. A statement is a claim only when what it is about is identified: a "
    "move by name, a piece with its square or its owner and square, a side, a count, the "
    "opening; leave out a statement about 'the rook', 'a piece', 'the attack' when the "
    "reply does not say which. If the reply makes no claim, return an empty list. Answer "
    'in JSON only: {"claims": [str]}'
)

BINARY_SYSTEM = (
    "You answer one question about a chess coach's reply and the record of the game. When "
    "the question asks yes or no, answer with that one word only, no explanation. When it "
    "asks for the reply's atomic claims, answer in JSON only."
)


class Ledger:
    """Every paid call on the record, with a budget that stops the run."""

    def __init__(self, run: str, purpose: str, model: str, budget_tokens: int) -> None:
        self.run, self.purpose, self.model = run, purpose, model
        self.budget = budget_tokens
        self.calls = 0
        self.prompt = 0
        self.cached = 0  # of the prompt tokens, the ones served from the prefix cache
        self.completion = 0
        self.price_in = float(os.environ.get("DASHSCOPE_PRICE_IN_PER_M", "0") or 0)
        self.price_out = float(os.environ.get("DASHSCOPE_PRICE_OUT_PER_M", "0") or 0)
        self.price_cached = float(os.environ.get("PRICE_CACHED_PER_M", "0") or 0)
        self._lock = threading.Lock()
        USAGE_MINE.parent.mkdir(parents=True, exist_ok=True)

    @property
    def total(self) -> int:
        return self.prompt + self.completion

    def cost(self) -> float | None:
        if not (self.price_in or self.price_out):
            return None
        fresh = self.prompt - self.cached
        return (
            fresh / 1e6 * self.price_in
            + self.cached / 1e6 * (self.price_cached or self.price_in)
            + self.completion / 1e6 * self.price_out
        )

    def record(self, item_id: str, stats: dict[str, Any], elapsed: float) -> None:
        p = int(stats.get("prompt_tokens") or 0)
        c = int(stats.get("completion_tokens") or 0)
        cached = int(stats.get("cached_tokens") or 0)
        with self._lock:
            self.calls += 1
            self.prompt += p
            self.cached += cached
            self.completion += c
        with self._lock, USAGE_MINE.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "run": self.run,
                        "purpose": self.purpose,
                        "model": self.model,
                        "item": item_id,
                        "prompt_tokens": p,
                        "cached_tokens": cached,
                        "completion_tokens": c,
                        "elapsed_s": round(elapsed, 2),
                    }
                )
                + "\n"
            )

    def over_budget(self) -> bool:
        return self.total >= self.budget

    def summary(self) -> str:
        cost = self.cost()
        money = (
            f", about ${cost:.2f}"
            if cost is not None
            else " (set DASHSCOPE_PRICE_IN_PER_M / _OUT_PER_M for a dollar figure)"
        )
        return (
            f"{self.calls} calls, {self.prompt:,} prompt ({self.cached:,} cached) + "
            f"{self.completion:,} completion = {self.total:,} tokens of a {self.budget:,} "
            f"budget{money}"
        )


def lifetime_usage() -> dict[str, int]:
    """All DashScope tokens ever recorded in the ledger, by model."""
    out: dict[str, int] = {}
    paths = ([USAGE] if USAGE.exists() else []) + sorted(USAGE_MINE.parent.glob("usage-*.jsonl"))
    for path in paths:
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[rec["model"]] = (
                out.get(rec["model"], 0) + rec["prompt_tokens"] + rec["completion_tokens"]
            )
    return out


# ---- the request an item holds ------------------------------------------------------


def request_for(item: dict, system: str) -> _Request:
    return _Request(
        trigger=Trigger(item["trigger"]),
        level=DisclosureLevel[item["level"]],
        system=system,
        messages=item["messages"],
        facts=PositionFacts.model_validate(item["moment"]["facts"]),
        board=chess.Board(item["fen"]),
        fallback=item["fallback"],
        ply=item["ply"],
        game_id=item["game"],
        user_text=item.get("question"),
        # What the directive disclosed (the engine on what the player asked about).
        disclosed=_disclosed(item),
        candidates=item.get("candidates", 4),
        played=frozenset(_played_sans(item)),
    )


_PREFERRED = re.compile(r"(?:the )?engine preferred (?:\d+\.(?:\.\.)? )?([A-Za-z0-9=+#-]+)")


def _disclosed(item: dict) -> str:
    """What the directive disclosed: the engine on what the player asked about."""
    directive = str(item["messages"][-1]["content"]) if item.get("messages") else ""
    return chr(10).join(x for x in directive.splitlines() if x.startswith("Engine on "))


_RECORD_LINE = re.compile(r"^\d+\.(?:\.\.)? (\S+) \((?:the player|the opponent)\)")
_LAST_MOVE_LINE = re.compile(r"^\d+\.(?:\.\.)? (\S+) by (?:the player|the opponent)")


def _played_sans(item: dict) -> list[str]:
    """The moves of the record, read off the harness: every move played (a line of "Game
    so far", or the "Last move" section) and the move each verdict said the engine
    preferred (the coach's own _record_moves, from the same text)."""
    out = []
    for m in item["messages"]:
        if m["role"] != "user":
            continue
        for line in m["content"].splitlines():
            found = _RECORD_LINE.match(line) or _LAST_MOVE_LINE.match(line)
            if found:
                out.append(found.group(1))
            for preferred in _PREFERRED.finditer(line):
                out.append(preferred.group(1))
    return out


# ---- deterministic ------------------------------------------------------------------


def _level_words(level: DisclosureLevel) -> str:
    return {
        DisclosureLevel.SILENT: "nothing about the position",
        DisclosureLevel.IDEA: (
            "the idea: the goal, the pattern and why it works; no square, no move, and not "
            "which of the player's own pieces does it"
        ),
        DisclosureLevel.FOCUS: "the idea, the target and the piece; not the move",
        DisclosureLevel.MOVE: "the move and why it works; not the whole line",
        DisclosureLevel.LINE: "everything",
    }[level]


_PIECE_WORDS = {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}


def deterministic(item: dict, reply: str, draft_check) -> dict[str, Any]:
    facts = PositionFacts.model_validate(item["moment"]["facts"])
    level = DisclosureLevel[item["level"]]
    board = chess.Board(item["fen"])
    # The checks exactly as the coach runs them: what the player said and the moves of
    # the game so far are not leaks; the board check has the frames along the engine's
    # line, and a move of the game's history named as such is not a claim about the board.
    request = request_for(item, item["system"])
    # As the coach's own check: the piece a played move moved, the material line and
    # the opening's name are not hints at the piece to move.
    moved = {_PIECE_WORDS.get(san[0], "pawn") for san in request.played if san and san[0] != "O"}
    extra = [item["moment"].get("material") or "", item["moment"].get("opening") or ""]
    said = " ".join(
        filter(
            None,
            [
                item.get("question"),
                request.disclosed,
                *sorted(request.played),
                *sorted(moved),
                *extra,
            ],
        )
    )
    leaks = [f"withheld {x}" for x in leak_check(reply, facts, level, said=said or None)]
    board_problems: list[str] = []
    if draft_check and reply:
        for problem in draft_check(reply, board, _after_best(request)):
            if problem.split(":", 1)[0].strip() in request.played:
                continue
            board_problems.append(problem)
    board_problems += owner_problems(reply, item)
    form: dict[str, Any] = {"sentences": sentence_count(reply)}
    fm = item.get("form") or {}
    problems: list[str] = []
    if not reply.strip():
        problems.append("empty")
    if fm.get("max_sentences") and form["sentences"] > fm["max_sentences"] + 2:
        problems.append(f"{form['sentences']} sentences, {fm['max_sentences']} allowed")
    if fm.get("ends_with_question") is True and not reply.rstrip().endswith("?"):
        problems.append("no closing question")
    if fm.get("ends_with_question") is False and reply.rstrip().endswith("?"):
        problems.append("ends with a question where none was wanted")
    form["problems"] = problems
    return {"leak": leaks, "board": board_problems, "form": form}


_OWNED = re.compile(
    r"\b(?P<owner>White|Black|your opponent|the opponent|opponent|their|his|her|your|my|the player)"
    r"(?:['\u2019]s)?\s+(?P<piece>king|queen|rook|bishop|knight|pawn)s?\s+(?:on|at)\s+(?P<square>[a-h][1-8])\b",
    re.IGNORECASE,
)
_PIECE_TYPES = {
    "king": chess.KING,
    "queen": chess.QUEEN,
    "rook": chess.ROOK,
    "bishop": chess.BISHOP,
    "knight": chess.KNIGHT,
    "pawn": chess.PAWN,
}


def owner_problems(reply: str, item: dict) -> list[str]:
    """A piece named with an owner and a square ("Black's rook on e7", "your knight on
    f3") checked on the board: the piece there must be that piece and belong to that side,
    on the position now or on the one the last move was played from. "Black's rook on e7"
    said of White's rook passed the model judge."""
    f = item["moment"]["facts"]
    side = (f.get("user_side") or f.get("side_to_move") or "").lower()
    if side not in ("white", "black"):
        return []
    player = chess.WHITE if side == "white" else chess.BLACK
    fens = [item.get("fen_now") or item["fen"]]
    before = item["moment"].get("fen_before_last")
    if before and before not in fens:
        fens.append(before)
    boards = [chess.Board(x) for x in fens]
    problems: list[str] = []
    for match in _OWNED.finditer(reply):
        owner = match.group("owner").lower()
        if owner in ("white",):
            colour = chess.WHITE
        elif owner in ("black",):
            colour = chess.BLACK
        elif owner in ("your", "my", "the player"):
            colour = player
        else:
            colour = not player
        want = _PIECE_TYPES[match.group("piece").lower()]
        square = chess.parse_square(match.group("square").lower())
        found = [b.piece_at(square) for b in boards]
        if any(p is not None and p.piece_type == want and p.color == colour for p in found):
            continue
        if any(p is not None and p.piece_type == want and p.color != colour for p in found):
            whose = "White's" if colour == chess.BLACK else "Black's"
            problems.append(
                f"{match.group(0)}: the {match.group('piece').lower()} on"
                f" {match.group('square')} is {whose}"
            )
        else:
            problems.append(f"{match.group(0)}: no such piece on {match.group('square')}")
    return problems


_NEGATED = re.compile(
    r"\b(not|no|never|neither|nor|without|isn't|doesn't|don't|can't|cannot|won't|wasn't|didn't)\b",
    re.IGNORECASE,
)
_TACTICAL = re.compile(
    r"\b(attack|fork|pin|skewer|hang|loose|threat|defend|protect|en prise|discover"
    r"|undefended|trap|capture|win)",
    re.IGNORECASE,
)


# ---- the judge ----------------------------------------------------------------------


def evidence_block(item: dict, *, with_reference: bool = True) -> str:
    m = item["moment"]
    f = m["facts"]
    lines = [
        f"Class: {item['class']} -- level in force for the position the player faces:"
        f" {item['level']} ({_level_words(DisclosureLevel[item['level']])})."
    ]
    side = (f.get("user_side") or f.get("side_to_move") or "").lower()
    if side in ("white", "black"):
        other = "Black" if side == "white" else "White"
        lines.append(
            f"The player is {side.capitalize()}; the opponent is {other}. 'You'/'your' in the"
            f" reply means {side.capitalize()}'s side."
        )
    if item.get("question"):
        lines.append(f'The player asked: "{item["question"]}"')
    else:
        lines.append(f"The coach spoke on its own ({item['trigger']}).")
    if m.get("previous_san"):
        lines.append(f"The move before the last one: {m['previous_san']}.")
    if m.get("last_san"):
        who = "the player" if m.get("last_by_user") else "the opponent"
        v = f"Last move: {m['last_san']} by {who}; verdict {m.get('last_label')}"
        if m.get("last_loss") is not None:
            v += f", cost {m['last_loss']:.2f} of expected score"
        if m.get("last_best_san") and m["last_best_san"] != m["last_san"]:
            v += f"; the engine preferred {m['last_best_san']}"
            if m.get("last_best_leads"):
                v += f", which leads to: {m['last_best_leads']}"
        lines.append(v + ".")
        # The harness's own line on the move, verbatim: the motifs the tagger named, the
        # material before and after the expected recapture. "Kf2, a discovered attack"
        # was faulted with "Motifs in the move: a discovered attack" in the message.
        harness_line = last_move_line(item)
        if harness_line:
            lines.append("The coach's message on that move said: " + harness_line)
        # What the move did and what the better move would have done, as the coach's own
        # messages state them (computed on the board).
        if m.get("fen_before_last"):
            b0 = chess.Board(m["fen_before_last"])
            try:
                did = effects_text(
                    b0,
                    b0.parse_san(m["last_san"]),
                    hazards=(m.get("last_label") or "")
                    in ("inaccuracy", "mistake", "miss", "blunder"),
                )
                if did:
                    lines.append(f"What {m['last_san']} does: {did}.")
                if m.get("last_best_san") and m["last_best_san"] != m["last_san"]:
                    would = effects_text(b0, b0.parse_san(m["last_best_san"]), hazards=False)
                    if would:
                        lines.append(f"What {m['last_best_san']} would do: {would}.")
            except ValueError:
                pass
    # For a verdict on the player's move the coach's facts (theme, tactic, candidates,
    # line) are for the position the move was played FROM; the board now is after it.
    facts_before = (
        bool(m.get("last_by_user"))
        and item["fen"] != item.get("fen_now")
        and item["fen"] == m.get("fen_before_last")
    )
    if facts_before and m.get("fen_before_last"):
        b0 = chess.Board(m["fen_before_last"])
        lines.append(
            "The board the last move was played from (rank 8 at the top; capitals White)"
            " -- the facts below are for THIS position:"
        )
        lines.append(ascii_board(b0))
    if item.get("fen_now"):
        b = chess.Board(item["fen_now"])
        lines.append("The board now, after the last move (rank 8 at the top; capitals White):")
        lines.append(ascii_board(b))
        lines.append(("White" if b.turn else "Black") + " to move.")
    if m.get("material"):
        lines.append(f"Material (from the player's side): {m['material']}.")
    if m.get("standing"):
        lines.append(f"Standing, as the coach's messages state it: {m['standing']}")
    if m.get("lines"):
        b_now = chess.Board(item["fen_now"]) if item.get("fen_now") else None
        rendered = []
        for s, sc in m["lines"]:
            does = ""
            if b_now is not None:
                try:
                    does = effects_text(b_now, b_now.parse_san(s), hazards=False)
                except ValueError:
                    does = ""
            leads = (m.get("leads") or {}).get(s)
            rendered.append(
                f"{s} {sc:.2f}" + (f" ({does})" if does else "") + (f" -> {leads}" if leads else "")
            )
        lines.append(
            "Engine lines for the position now, (move, expected score from the player's side,"
            " what the move does, -> where the line leads): " + "; ".join(rendered) + "."
        )
    if item.get("fen_now") and f.get("user_side") or f.get("side_to_move"):
        # What stands on the board, complete, as the harness lists it: hanging, attacked,
        # pinned, double attacks, checks and captures on. True by construction.
        try:
            b_inv = chess.Board(item["fen_now"] or item["fen"])
            side = (f.get("user_side") or f.get("side_to_move") or "white").lower()
            colour = chess.WHITE if side == "white" else chess.BLACK
            lines.append(
                "On the board now (complete lists): "
                + " | ".join(inventory_lines(b_inv, colour))
                + "."
            )
        except ValueError:
            pass
    record = m.get("record") or record_from_harness(item) or []
    if record:
        lines.append("The moves of the game with their verdicts: " + "; ".join(record) + ".")
    slips = slip_lines(item)
    if slips:
        lines.append(
            "The moves that lost ground, as the coach's message states them: " + " | ".join(slips)
        )
    said = said_here(item)
    if said:
        lines.append(
            "Already said at this position (the coach's own earlier words here, and the player's): "
            + " | ".join(said)
        )
    facts = []
    if f.get("decisive"):
        facts.append(
            "decisive: yes -- a critical moment, the best move stands well clear of the rest"
        )
    if f.get("only_move"):
        facts.append("only move: yes -- one move holds, the rest lose ground")
    for key in (
        "phase",
        "nature",
        "theme",
        "plan",
        "region",
        "piece",
        "tactic",
        "threats",
        "ideas",
        "candidates",
        "line",
        "focus",
    ):
        val = f.get(key)
        if val in (None, "", [], False):
            continue
        if key == "threats" and isinstance(val, list):
            val = [threat_words(chess.Board(item["fen"]), str(v)) for v in val]
        if isinstance(val, list):
            val = ", ".join(str(v) for v in val)
        facts.append(f"{key}: {val}")
    if facts:
        owner = (
            "the position the last move was played from (the player's, before the move)"
            if facts_before
            else "the position now (the side to move's)"
        )
        lines.append(f"Facts for {owner}: " + "; ".join(facts) + ".")
    if m.get("facts_before") and not facts_before:
        # A question about the player's last move, asked from the position it led to: the
        # facts of the position it was played from are the ones the move is judged by.
        fb = m["facts_before"]
        before = []
        for key in (
            "nature",
            "theme",
            "plan",
            "region",
            "piece",
            "tactic",
            "threats",
            "ideas",
            "candidates",
            "line",
            "focus",
        ):
            val = fb.get(key)
            if val in (None, "", [], False):
                continue
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            before.append(f"{key}: {val}")
        if before:
            lines.append(
                "Facts for the position the last move was played from (the player's, before"
                " the move): " + "; ".join(before) + "."
            )
    if m.get("opening"):
        lines.append(f"Opening: {m['opening']}.")
    if m.get("tally"):
        lines.append(
            "The player's moves so far by label: "
            + ", ".join(f"{v} {k}" for k, v in m["tally"].items())
            + "."
        )
    if m.get("lost"):
        lines.append("Where the player lost ground: " + "; ".join(m["lost"]) + ".")
    if m.get("material_arc") and item["class"] in ("ask_game_so_far", "ask_free"):
        lines.append("How the material went: " + m["material_arc"] + ".")
    if m.get("result"):
        lines.append(f"Game over: {m['result']}.")
    if m.get("named_san"):
        lines.append(
            f"The move the player named: {m['named_san']}, ranked {m['named_rank']} by the"
            f" engine, {m['named_loss']:.2f} behind the best."
        )
    # The engine's word on a move the player named, as the directive carried it, and the
    # coach's own turn that spoke it in plain words (the reply builds on that turn, so
    # its words are the coach's to repeat).
    directive = str(item["messages"][-1]["content"]) if item.get("messages") else ""
    for line in directive.splitlines():
        if line.startswith("Engine on "):
            lines.append(line)
    if item.get("quick_followup"):
        # The coach's own last turn is the last thing said under "Said here" in the harness.
        said = (
            [
                x
                for x in directive.split("## Said here", 1)[-1].splitlines()
                if x.strip().startswith("YOU (the coach) said: ")
            ]
            if "## Said here" in directive
            else []
        )
        if said:
            lines.append(
                "The coach's own last turn, spoken from the facts above: "
                + said[-1].strip().removeprefix("YOU (the coach) said: ").strip(chr(34))
            )
    lines.append("Required facts: " + json.dumps(item["required"], ensure_ascii=False))
    if with_reference and item.get("reference"):
        lines.append(
            "The master's note on the move just played (the reference for idea): "
            + item["reference"][:2500]
        )
    return "\n".join(lines)


def threat_words(board: chess.Board, san: str) -> str:
    """A threat with what it does, as the coach's message renders it."""
    if board.is_check():
        return san
    passed = board.copy(stack=False)
    passed.push(chess.Move.null())
    try:
        text = effects_text(passed, passed.parse_san(san), hazards=False)
    except ValueError:
        return san
    return f"{san} ({text})" if text else san


# The coach speaks to the player ("you", "your"); the checker reads a student speaking of
# themself ("I", "my"). Whole words; the opponent's words are left as they are.
_TO_STUDENT = (
    (re.compile(r"\byour opponent's\b", re.I), "their"),
    (re.compile(r"\byour opponent\b", re.I), "they"),
    (re.compile(r"\bYours\b"), "Mine"),
    (re.compile(r"\byours\b"), "mine"),
    (re.compile(r"\bYour\b"), "My"),
    (re.compile(r"\byour\b"), "my"),
    (re.compile(r"\bYou are\b"), "I am"),
    (re.compile(r"\byou are\b"), "I am"),
    (re.compile(r"\bYou\b"), "I"),
    (re.compile(r"\byou\b"), "I"),
)


def _player_to_move(fen: str, user_side: str) -> chess.Board | None:
    board = chess.Board(fen)
    want = chess.WHITE if user_side == "white" else chess.BLACK
    if board.turn == want:
        return board
    if board.is_check():
        return None
    board.push(chess.Move.null())
    return board


_SAN_IN_TEXT = re.compile(
    r"\b(O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?(?![a-zA-Z])"
)


def _after_null(board: chess.Board) -> chess.Board | None:
    """The same position with the other side to move, or None when that side is in check
    (a null move is not legal there)."""
    if board.is_check():
        return None
    b = board.copy(stack=False)
    b.push(chess.Move.null())
    return b


def checker_rules(text: str, item: dict, provider, reply: str = "") -> tuple[str, str, str]:
    """The board-checker's ruling on one claim the judge marked unsupported: ("true" |
    "false" | "unclear", the checker's reason, the claim kind). True on the position now
    or on the one the last move was played from counts as true; false only when false on
    both; the kinds the checker cannot read are unclear."""
    from sidecoach.coach import claims as claims_mod

    m = item["moment"]
    facts = PositionFacts.model_validate(m["facts"])
    fens = [item.get("fen_now") or item["fen"]]
    if m.get("fen_before_last") and m["fen_before_last"] not in fens:
        fens.append(m["fen_before_last"])
    # And the positions along the line the player asked about, as the directive disclosed
    # it: a move the coach recites from the engine's reply is legal there, not here.
    directive = str(item["messages"][-1]["content"]) if item.get("messages") else ""
    engine_on = [line for line in directive.splitlines() if line.startswith("Engine on ")]
    for frame in disclosed_frames(chess.Board(item["fen"]), engine_on):
        if frame.fen() not in fens:
            fens.append(frame.fen())
    # And the position after each move the claim names that either side could play now:
    # "Nxe5 reveals the bishop's attack on b5" is about the board after Nxe5, and was
    # ruled false on the board before it, where the knight still blocked the diagonal.
    # A claim that names no move of its own is read after the moves the reply names too:
    # 'your bishop pins the knight on c6 to the rook on a8' was said of the board after
    # the Bxf3 the reply recommends.
    named_moves = list(_SAN_IN_TEXT.findall(text))  # a bare square may be a pawn move
    if not any(not re.fullmatch(r"[a-h][1-8]", x) for x in named_moves) and reply:
        named_moves += [
            x for x in _SAN_IN_TEXT.findall(reply) if not re.fullmatch(r"[a-h][1-8]", x)
        ]
    for san in named_moves:
        for start in list(fens[:2]):
            b = chess.Board(start)
            for side_board in (b, _after_null(b)):
                if side_board is None:
                    continue
                try:
                    mv = side_board.parse_san(san)
                except ValueError:
                    continue
                nxt = side_board.copy(stack=False)
                nxt.push(mv)
                if nxt.fen() not in fens:
                    fens.append(nxt.fen())
    student = text
    for pattern, new in _TO_STUDENT:
        student = pattern.sub(new, student)
    first = _player_to_move(fens[0], m["facts"]["user_side"])
    if first is None:
        return "unclear", "", "other"
    found = claims_mod.extract_claims(student, first, provider, thinking=False, max_tokens=4000)
    # A piece named with no owner ('the King', 'the bishop') may be either side's: the
    # claim is tried with the opponent's as well, and holds when either reading does.
    # 'a square attacked by the King' was Black's king on e8 attacking f7, ruled false
    # against the player's own king.
    found = _with_owner_variants(found)
    # A move of the game's record, named as such (a review: "17.Nxd8 was accurate"), is
    # not a claim about the board now: it was legal where it was played.
    played = set(request_for(item, item["system"]).played)
    says_best = (
        re.search(
            r"\b(?:best|strongest|top|only) (?:move|choice|reply)\b|\b(?:is|was) best\b", text, re.I
        )
        is not None
    )
    checkable = [
        c
        for c in found
        if c.kind.value in ("attacks", "defends", "threatens", "hanging", "safe", "best")
        and not (c.move and c.move.rstrip("+#") in played and c.kind.value == "best")
        and not (c.kind.value == "best" and not says_best)
    ]
    if not checkable:
        return "unclear", "", (found[0].kind.value if found else "other")
    results = []
    for fen in fens:
        board = _player_to_move(fen, m["facts"]["user_side"])
        if board is None:
            continue
        verdicts = claims_mod.verify_claims(
            checkable, board, facts=facts if fen == item["fen"] else None
        )
        results.append(_merge_variants(verdicts))
    # Every checkable claim in the sentence must hold on one position for the sentence to
    # count as true; one claim false on every position makes it false.
    for verdicts in results:
        if all(v.result == "true" for v in verdicts):
            return "true", "; ".join(v.reason for v in verdicts), checkable[0].kind.value
    if results and all(any(v.result == "false" for v in verdicts) for verdicts in results):
        bad = next(v for v in results[0] if v.result == "false")
        if (
            re.search(r"\b(undefended|unprotected|unguarded|no defender|loose)\b", text, re.I)
            and "not attacked" in bad.reason
        ):
            # "Undefended" / "loose" says nothing about attackers; the checker's "hanging"
            # does. Whether the piece has a defender is read off the board instead.
            return _undefended_ruling(bad.claim, fens, m["facts"]["user_side"], bad.reason)
        return "false", bad.reason, bad.claim.kind.value
    return "unclear", results[0][0].reason if results else "", checkable[0].kind.value


_OWNER_WORD = re.compile(
    r"\b(my|mine|i|me|our|your|their|they|them|opponent|opponent's|his|her|white|black|white's|black's)\b|[a-h][1-8]",
    re.IGNORECASE,
)


def _with_owner_variants(found):
    """Each claim, and for a claim whose subject names a piece with no owner and no
    square, a twin with the opponent's ('the King' -> 'their King'), marked as such."""
    from dataclasses import replace

    out = []
    for c in found:
        out.append(c)
        if (
            c.subject
            and not _OWNER_WORD.search(c.subject)
            and c.kind.value in ("attacks", "defends")
        ):
            out.append(replace(c, subject="their " + c.subject, text=c.text + " [their]"))
    return out


def _merge_variants(verdicts):
    """A claim and its owner twin fold into one verdict: true when either is true, false
    only when both are, else unclear."""
    out = []
    by_text = {}
    for v in verdicts:
        base = v.claim.text.removesuffix(" [their]")
        by_text.setdefault(base, []).append(v)
    for vs in by_text.values():
        if any(v.result == "true" for v in vs):
            out.append(next(v for v in vs if v.result == "true"))
        elif all(v.result == "false" for v in vs):
            out.append(vs[0])
        else:
            first = vs[0]
            out.append(type(first)(claim=first.claim, result="unclear", reason=first.reason))
    return out


def _undefended_ruling(claim, fens: list[str], user_side: str, reason: str) -> tuple[str, str, str]:
    """A piece called loose or undefended: true when, on one of the positions, the piece
    the claim names has no defender; false when it has one on every position; unclear
    when the words fit no single piece."""
    from sidecoach.coach import claims as claims_mod

    seen = False
    for fen in fens:
        board = _player_to_move(fen, user_side)
        if board is None:
            continue
        squares = claims_mod._resolve(claim.target or claim.subject, board, board.turn, None)
        squares = [sq for sq in squares if board.piece_at(sq) is not None]
        if len(squares) != 1:
            continue
        seen = True
        piece = board.piece_at(squares[0])
        assert piece is not None
        defenders = board.attackers(piece.color, squares[0])
        where = chess.square_name(squares[0])
        if not defenders:
            return (
                "true",
                f"the {chess.piece_name(piece.piece_type)} on {where} has no defender",
                claim.kind.value,
            )
    if not seen:
        return "unclear", reason, claim.kind.value
    return "false", "the piece the claim names is defended", claim.kind.value


_IDENTIFIED = re.compile(
    # a move or a square
    r"\b(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)(?![A-Za-z])"
    r"|\b(?:king|queen|rook|bishop|knight|pawn|pawns|pieces?|material|exchange)\b"
    r"|\d"  # a count, a cost, a move number
    r"|\b(?:white|black|equal|level|even|balanced|winning|losing|better|worse|ahead|behind|advantage|edge|decisive|drawn|draw|mate|checkmate|stalemate|resign|threat|threatens|threatening|fork|pin|skewer|check|opening|endgame|middlegame|book|best|excellent|great|brilliant|good|inaccuracy|mistake|miss|blunder|file|rank|diagonal|castle|castled|castling|isolated|passed|doubled|backward|outpost|tempo|initiative)\b",
    re.IGNORECASE,
)


_PIECE_IN_TEXT = re.compile(r"\b(king|queen|rook|bishop|knight|pawn)s?\b", re.IGNORECASE)
_HEDGED = re.compile(r"\b(could|might|may|potentially|possibly|perhaps)\b", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _is_claim_of(claim: str, reply: str) -> bool:
    """Whether the decomposition's claim is the reply's own. Every move, square and piece
    word in it must appear in the reply (at effort none the decomposition wrote 'your
    h-pawn on h2 is under attack' and 'Re8+ creates a two-layer attack' from the record,
    for replies that named neither); a claim whose words come from a question sentence
    is not a claim ('...while keeping your King safe?'); a hedge (could, might) that
    names no move is not a fact."""
    low = reply.lower()
    for san in _SAN_IN_TEXT.findall(claim):
        core = san.rstrip("+#")
        if core.lower() not in low:
            return False
    for word in _PIECE_IN_TEXT.findall(claim):
        if word.lower() not in low:
            return False
    if _HEDGED.search(claim) and not any(
        not re.fullmatch(r"[a-h][1-8]", x) for x in _SAN_IN_TEXT.findall(claim)
    ):
        return False
    words = {w for w in re.findall(r"[a-z][a-z'-]{3,}", claim.lower()) if w not in _CLAIM_STOP}
    if words:
        best, best_hit = None, 0
        for sentence in _SENTENCE_SPLIT.split(reply.strip()):
            hit = len(words & set(re.findall(r"[a-z][a-z'-]{3,}", sentence.lower())))
            if hit > best_hit:
                best, best_hit = sentence, hit
        if best is not None and best.rstrip().endswith("?") and best_hit >= max(2, len(words) // 2):
            return False
    return True


_CLAIM_STOP = frozenset(
    [
        "your",
        "their",
        "this",
        "that",
        "with",
        "from",
        "into",
        "over",
        "after",
        "before",
        "which",
        "what",
        "when",
        "then",
        "than",
        "there",
        "here",
        "were",
        "was",
        "is",
        "are",
        "the",
        "and",
        "but",
        "for",
        "not",
    ]
)


_MATERIAL_WORD = re.compile(
    r"\b(material|up|down|ahead|behind|level|even|equal|exchange|worth|wins?|winning|loses?|losing|takes?|captures?|threat\w*|attack\w*|pinn?\w*|fork\w*|check\w*|hanging|loose|undefended|defend\w*|trapped|safe)\b",
    re.IGNORECASE,
)


def _identified(claim: str) -> bool:
    """Whether a claim is about something the record could identify: a move, a square, a
    piece, a count, a side, the standing, a named theme. "You played a strong game overall"
    is an appraisal the decomposition should have left out; it is dropped, not voted on.
    So is a claim under three words ("The endgame"), and a piece named with no square,
    no move and no material or tactical word ("bringing your King toward a secure
    endgame square")."""
    if len(claim.split()) < 3:
        return False
    if _IDENTIFIED.search(claim) is None:
        return False
    has_piece = _PIECE_IN_TEXT.search(claim) is not None
    has_place = (
        any(not re.fullmatch(r"[a-h][1-8]", x) for x in _SAN_IN_TEXT.findall(claim))
        or re.search(r"\b[a-h][1-8]\b", claim) is not None
    )
    return not (
        has_piece
        and not has_place
        and not _MATERIAL_WORD.search(claim)
        and not re.search(r"\d", claim)
    )


def decompose_messages(reply: str, record: str | None = None) -> list[dict[str, str]]:
    """The instruction to split the reply into claims. With `record`, the record and the
    reply are in the system prompt (as for the checks that follow, so one cache entry
    serves them all) and the instruction alone is the user message; the record is context
    only: the claims are the reply's."""
    if record is not None:
        content = DECOMPOSE_SYSTEM
    else:
        content = (
            'The coach\'s reply:\n"""\n' + reply + '\n"""\n\nList its atomic claims. JSON only.'
        )
    return [{"role": "user", "content": content}]


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def parse_judgment(text: str) -> dict[str, Any] | None:
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def score(
    item: dict, det: dict[str, Any], verdict: dict[str, Any] | None, claims: list[str] | None = None
) -> dict[str, Any]:
    """Eq. 4 over the required facts (completeness, correctness) and, over the reply's own
    atomic claims, the share the evidence supports; any unsupported claim is a fabrication."""
    required = item["required"]
    completeness = correctness = None
    unsupported: list[str] = []
    unruled: list[str] = []
    failures: list[dict] = []  # every failure with its kind and the judge's lesson
    claims_total = claims_supported = 0
    relevance = None
    idea = None
    if verdict:
        marks = {r.get("key"): r for r in verdict.get("required", []) if isinstance(r, dict)}
        if required:
            mentioned = [marks.get(r["key"], {}).get("mentioned") for r in required]
            completeness = sum(1 for x in mentioned if x) / len(required)
            stated = [
                marks.get(r["key"], {})
                for r in required
                if marks.get(r["key"], {}).get("mentioned")
            ]
            correctness = (
                (sum(1 for s in stated if s.get("correct")) / len(stated)) if stated else None
            )
        else:
            completeness = 1.0
        # Every atomic claim gets one verdict; a claim the judge did not rule on counts as
        # supported (the benefit of the doubt) and is noted.
        claims_total = len(claims or [])
        ruled: dict[int, dict] = {}
        for c in verdict.get("claims", []):
            if isinstance(c, dict) and isinstance(c.get("n"), int):
                ruled[c["n"]] = c
        for n, text in enumerate(claims or [], start=1):
            c = ruled.get(n)
            if c is None:
                claims_supported += 1
                unruled.append(text)
            elif c.get("supported"):
                claims_supported += 1
            else:
                unsupported.append(f"{text} -- {c.get('why', '')}".strip(" -"))
                failures.append(
                    {
                        "what": "claim",
                        "text": text,
                        "why": str(c.get("why") or ""),
                        "kind": c.get("kind")
                        if c.get("kind") in ("false", "unstated")
                        else "unstated",
                        "lesson": str(c.get("lesson") or "").strip() or None,
                    }
                )
        for r in required:
            mark = marks.get(r["key"], {})
            if not mark.get("mentioned"):
                failures.append(
                    {
                        "what": "omission",
                        "text": r["key"],
                        "why": "",
                        "kind": "omitted",
                        "lesson": str(mark.get("lesson") or "").strip()
                        or f"left out the {r['key'].replace('_', ' ')}",
                    }
                )
            elif mark.get("correct") is False:
                failures.append(
                    {
                        "what": "wrong",
                        "text": r["key"],
                        "why": "",
                        "kind": "wrong",
                        "lesson": str(mark.get("lesson") or "").strip()
                        or f"got the {r['key'].replace('_', ' ')} wrong",
                    }
                )
        try:
            raw_relevance = verdict.get("relevance")
            relevance = float(raw_relevance) if raw_relevance is not None else None
        except (TypeError, ValueError):
            relevance = None
        idea_raw = verdict.get("idea")
        if isinstance(idea_raw, dict) and item.get("reference"):
            idea = {
                "point": str(idea_raw.get("point") or "").strip(),
                "match": bool(idea_raw.get("in_reply")),
                "in_facts": bool(idea_raw.get("in_facts")),
                "where": idea_raw.get("where") if isinstance(idea_raw.get("where"), str) else None,
            }
    hard = (
        bool(det["leak"])
        or bool(det["board"])
        or bool(unsupported)
        or "empty" in det["form"]["problems"]
    )
    # An answer to a different question is no answer.
    if relevance is not None and relevance < 0.5:
        hard = True
    return {
        "completeness": completeness,
        "correctness": correctness,
        "claims": claims_total,
        "claims_supported": claims_supported,
        "claim_support": (claims_supported / claims_total) if claims_total else None,
        "fabrication": bool(unsupported),
        "unsupported": unsupported,
        "unruled": unruled,
        "failures": failures,
        "leak": bool(det["leak"]),
        "board": bool(det["board"]),
        "form_ok": not det["form"]["problems"],
        "relevance": relevance,
        # A master's point, when the item has one: whether the reply made it, and whether
        # the coach's own evidence stated it (the miss is teachable only then).
        "idea_point": idea["point"] if idea else None,
        "idea_match": idea["match"] if idea else None,
        "idea_in_facts": idea["in_facts"] if idea else None,
        "idea_where": idea["where"] if idea else None,
        "pass": not hard and not det["form"]["problems"],
    }


class JudgeFailed(RuntimeError):
    """The judge model gave no usable answer (the server refused, or the JSON never
    parsed): not a judgment. The reply is left unjudged for a later pass, never scored."""


def _chat_with_backoff(provider, system: str, messages, ledger: Ledger, item_id: str, **kw):
    """One judge call, retried on a refused request (a 429 or a timeout) with a growing
    pause: a rate limit must cost time, never a verdict."""
    last: Exception | None = None
    for attempt in range(5):
        t0 = time.time()
        try:
            completion = provider.chat(system, messages, **kw)
            ledger.record(item_id, completion.stats, time.time() - t0)
            return completion
        except Exception as exc:  # noqa: BLE001
            last = exc
            pause = (15 if "429" in str(exc) else 5) * (attempt + 1)
            print(
                f"  [{item_id}] call refused ({str(exc)[:80]}); retry {attempt + 1} in {pause}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(pause)
    raise JudgeFailed(f"{item_id}: {type(last).__name__}: {last}")


class _Metered:
    """A provider whose every call goes through the ledger and the backoff: the board
    checker's extractor calls `chat` itself."""

    def __init__(self, inner, ledger: Ledger, item_id: str) -> None:
        self.inner, self.ledger, self.item_id = inner, ledger, item_id
        self.name, self.model = getattr(inner, "name", ""), getattr(inner, "model", "")

    def chat(self, system, messages, **kw):
        return _chat_with_backoff(self.inner, system, messages, self.ledger, self.item_id, **kw)


VOTES = 3


def _yes_no(binary, ledger: Ledger, item_id: str, context: str, question: str) -> bool | None:
    """One binary check, asked three times: the record and the reply in the system prompt
    (the part this model's cache serves again; a shared prefix of the user message was not
    -- measured), the question alone in the user message. A single no-reasoning answer
    flips on paraphrase; the majority does not. yes -> True, no -> False, no majority ->
    None (not judged)."""
    votes = []
    for _ in range(VOTES):
        completion = _chat_with_backoff(
            binary,
            BINARY_SYSTEM + "\n\n" + context,
            [{"role": "user", "content": "Question: " + question + "\nAnswer yes or no."}],
            ledger,
            item_id,
            max_tokens=5,
            temperature=0.0,
            thinking=False,
        )
        word = (completion.text or "").strip().lower().strip(".!\"'")
        votes.append(True if word.startswith("yes") else (False if word.startswith("no") else None))
    yes, no = votes.count(True), votes.count(False)
    if yes > VOTES // 2:
        return True
    if no > VOTES // 2:
        return False
    return None


# A verdict's tier: praise for the strong labels, criticism for the slips; "good" is
# either (mild praise or "fine, but").
_TIERS = {
    "best": "praise",
    "excellent": "praise",
    "great": "praise",
    "brilliant": "praise",
    "book": "praise",
    "good": "both",
    "inaccuracy": "criticism",
    "mistake": "criticism",
    "miss": "criticism",
    "blunder": "criticism",
}


# Questions whose answer the record always holds, and those it holds from a level up: a
# reply to one of these made only of questions answered nothing. A hint request or a plan
# question below the theme level is rightly answered with a question.
_FACT_ASKS = {
    "ask_opponent_move": DisclosureLevel.IDEA,
    "ask_last_move": DisclosureLevel.IDEA,
    "ask_eval": DisclosureLevel.IDEA,
    "ask_material": DisclosureLevel.IDEA,
    "ask_opening": DisclosureLevel.IDEA,
    "ask_game_so_far": DisclosureLevel.IDEA,
    "ask_named_move": DisclosureLevel.IDEA,
    "ask_plan": DisclosureLevel.IDEA,
    "ask_threat": DisclosureLevel.MOVE,
    "ask_move": DisclosureLevel.MOVE,
    "ask_line": DisclosureLevel.LINE,
}


def _relevance(item: dict, reply: str) -> float | None:
    """Decided in code, never asked: 0 when a fact was asked for, the level allowed it, and
    the reply is nothing but questions; 1 otherwise for such asks; None (not judged) for
    everything else."""
    if not item.get("question"):
        return None
    floor = _FACT_ASKS.get(item["class"])
    if floor is None or DisclosureLevel[item["level"]] < floor:
        return None
    clauses = [
        x.strip() for x in re.split(r"(?<=[.!?;:])\s+|\s+[-\u2014]\s+", reply.strip()) if x.strip()
    ]
    if clauses and all(x.endswith("?") for x in clauses):
        return 0.0
    return 1.0


_RECORD_LINE = re.compile(r"^(\d+\.(?:\.\.)? \S+) \((the player|the opponent)\): ([a-z]+)")
_LAST_MOVE = re.compile(r"^(\d+\.(?:\.\.)? \S+) by (the player|the opponent): ([a-z]+)")


_SLIP_LINE = re.compile(
    r"^\d+\.(?:\.\.)? \S+ \((?:the player|the opponent)\): [a-z]+, cost [0-9.]+.*$"
)


def slip_lines(item: dict) -> list[str]:
    """The harness's record lines for the moves that lost ground, verbatim: the cost, the
    move the engine preferred and what it would have done."""
    text = str(item["messages"][-1]["content"]) if item.get("messages") else ""
    return [line.strip() for line in text.splitlines() if _SLIP_LINE.match(line)]


def said_here(item: dict) -> list[str]:
    """What was said at this position before the reply, as the harness lists it."""
    text = str(item["messages"][-1]["content"]) if item.get("messages") else ""
    if "## Said here\n" not in text:
        return []
    block = text.split("## Said here\n", 1)[1].split("\n# Task", 1)[0]
    return [line.strip() for line in block.splitlines() if line.strip()]


def last_move_line(item: dict) -> str:
    """The '## Last move' line of the harness the coach saw, verbatim."""
    text = str(item["messages"][-1]["content"]) if item.get("messages") else ""
    if "## Last move\n" not in text:
        return ""
    return text.split("## Last move\n", 1)[1].split("\n", 1)[0].strip()


def record_from_harness(item: dict) -> list[str]:
    """Every move of the game with its label, read off the harness the coach saw (the
    'Game so far' lines and the 'Last move' section): the ground truth for a review."""
    text = str(item["messages"][-1]["content"]) if item.get("messages") else ""
    out = []
    for line in text.splitlines():
        found = _RECORD_LINE.match(line) or _LAST_MOVE.match(line)
        if found:
            out.append(f"{found.group(1)} ({found.group(2)}, {found.group(3)})")
    return out


_OPENING_STOP = frozenset(
    [
        "defense",
        "defence",
        "variation",
        "attack",
        "opening",
        "game",
        "system",
        "line",
        "main",
        "gambit",
        "accepted",
        "declined",
        "the",
        "of",
        "and",
        "with",
    ]
)


def opening_named(reply: str, name: str) -> bool:
    """Whether the reply names the opening: every distinctive word of the name's first
    part (before the first comma, ECO code dropped) appears in the reply. "Sicilian
    Defense: Najdorf Variation, Main Line" is named by "the Sicilian Najdorf"."""
    head = re.sub(r"^[A-E]\d\d\s*", "", name).split(",")[0]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z'’-]+", head) if w.lower() not in _OPENING_STOP]
    text = reply.lower().replace("’", "'").replace("–", "-").replace("—", "-")
    return bool(words) and all(w.lower() in text for w in words)


def _required_words(fact: dict) -> str:
    """A required fact as the two questions name it."""
    key = fact["key"].replace("_", " ")
    words: dict[str, str] = {
        "label": "a verdict on the move (how good or bad it was)",
        "move": "the move itself",
        "cost": "how much the move cost",
        "better move": "the move that was better",
        "idea": "the idea behind the position or the move",
        "tactic": "the tactic on the board",
        "eval": "who stands better",
        "material": "the material balance",
        "opening": "the opening",
        "plan": "the plan",
        "nature": "whether the position calls for a forcing move or a quiet one",
        "candidate": "the candidate move",
        "line": "the line",
        "named verdict": "the verdict on the move the player named",
        "strong moves": "how many of the player's moves were strong",
        "lost worst": "the move where the player lost the most ground",
    }
    return words.get(key) or key


_COUNT = re.compile(
    r"\b(\d+)\s+(best|excellent|great|brilliant|good|book|inaccurac(?:y|ies)|mistakes?|miss(?:es)?|blunders?)\b",
    re.IGNORECASE,
)
_LABEL_OF = {
    "inaccuracies": "inaccuracy",
    "mistakes": "mistake",
    "misses": "miss",
    "blunders": "blunder",
}
_COUNT_WERE = re.compile(
    r"\b(\d+) (?:of your |of the )?()moves (?:were|was|are|rated|judged|scored)?\s*"
    r"(?:judged |rated |as |at least )?((?:best|excellent|great|brilliant|good|book"
    r"|inaccurac(?:y|ies)|mistakes?|miss(?:es)?|blunders?|,|\s|or|and)+)",
    re.IGNORECASE,
)
_COUNT_OF = re.compile(
    r"\b(\d+) of (?:your |the |the player's )?(\d+)(?: judged)? moves"
    r"(?:were|was|are|rated|judged|scored)?\s*(?:judged |rated |as |at least )?"
    r"((?:best|excellent|great|brilliant|good|book|inaccurac(?:y|ies)|mistakes?|miss(?:es)?"
    r"|blunders?|,|\s|or|and)+)",
    re.IGNORECASE,
)
_COST = re.compile(
    r"\b(?:cost|costing|costs)\b[^.;]{0,40}?(?:about |around |roughly |nearly |over |some )?"
    r"(?<![A-Za-z0-9])(\d+(?:\.\d+)?)(?![.\d]*[A-Za-z])"
    r"(?!\s*(?:moves?|plies|pawns?|pieces?|tempi|tempo|points?)\b)"
    r"\s*(%|percent|of expected score|of the game|in expected score)?",
    re.IGNORECASE,
)
_COST_WORDS = {
    "a fifth": (0.15, 0.30),
    "a quarter": (0.20, 0.32),
    "a third": (0.27, 0.40),
    "half": (0.40, 0.60),
    "a tenth": (0.06, 0.15),
    "a twentieth": (0.03, 0.08),
}


_MAT_LEVEL = re.compile(
    r"\b(level|even|equal|balanced|neither side is (?:ahead|up)"
    r"|no (?:material )?(?:advantage|edge|imbalance)|not (?:up|ahead|down|behind)"
    r"|nobody is (?:ahead|up))\b",
    re.I,
)
_MAT_UP = re.compile(
    r"\b(?:are|'re|is|am|be|being|remain|stay|were|was)\s+(?:up|ahead)\b"
    r"|\b(?:up|ahead) (?:a |an |two |three |the |by |in |on )?"
    r"(?:pawn|piece|rook|bishop|knight|queen|exchange|material)",
    re.I,
)
_MAT_DOWN = re.compile(r"\b(down|behind)\b", re.I)
_MAT_WORDS = re.compile(
    r"\b(material|materially|pawns?|pieces?|exchange|knights?|bishops?|rooks?|queens?)\b", re.I
)


_NOT_MATERIAL = re.compile(
    r"\b(stockfish|engine|evaluat|edge|score|by about|by roughly|by around|pawns? ahead"
    r"|ahead by|better by|gives? up|gave up|giving up|set up|pick up|end(?:s|ed)? up|was|were"
    r"|after|would|will|could|then|moved|went|became|before|earlier|since|from|used to"
    r"|leaves?|leaving|left|wins?|winning|gains?|gets?)\b",
    re.IGNORECASE,
)


def material_check(text: str, record: str | None) -> bool | None:
    """A present-tense claim about the material balance against the record's material
    line ("level", "level in value: your bishop against their knight", "you are up a
    pawn"). "Neither side is ahead materially" was voted false against "level in value";
    the words are few, so this is decided in code. None when the claim is not about the
    balance now: an evaluation in pawns, a line's outcome, the material at an earlier
    move are left to the vote."""
    if not record or not _MAT_WORDS.search(text) or _NOT_MATERIAL.search(text):
        return None
    if not (_MAT_LEVEL.search(text) or _MAT_UP.search(text) or _MAT_DOWN.search(text)):
        return None
    if "exchange" not in text.lower() and not re.search(
        r"\b(material|materially|up|down|ahead|behind|level|even|equal)\b", text, re.I
    ):
        return None
    says_level = bool(_MAT_LEVEL.search(text))
    says_up = bool(_MAT_UP.search(text)) and not says_level
    says_down = bool(_MAT_DOWN.search(text)) and not says_level
    rec = record.lower()
    rec_level = rec.startswith("level")
    rec_up = rec.startswith("you are up")
    rec_down = rec.startswith("you are down")
    # "Not up material" holds when level or down; "not down" when level or up.
    if re.search(r"\bnot (?:up|ahead)\b", text, re.I) and not re.search(
        r"\bnot (?:down|behind)\b", text, re.I
    ):
        return not rec_up
    if re.search(r"\bnot (?:down|behind)\b", text, re.I):
        return not rec_down
    if says_level and not (says_up or says_down):
        return rec_level
    if says_up and not says_down:
        return rec_up
    if says_down and not says_up:
        return rec_down
    return None


_RECORD_COST = re.compile(
    r"^\d+\.(?:\.\.)? (\S+) \((?:the player|the opponent)\): [a-z]+, cost ([0-9.]+)"
)


def record_costs(item: dict) -> dict[str, float]:
    """The cost of every earlier move that lost ground, by SAN, read off the harness's
    record lines ("13. Qe2 (the player): inaccuracy, cost 0.07; ...")."""
    text = str(item["messages"][-1]["content"]) if item.get("messages") else ""
    out: dict[str, float] = {}
    for line in text.splitlines():
        found = _RECORD_COST.match(line)
        if found:
            out[found.group(1).rstrip("+#")] = float(found.group(2))
    return out


_SAN_TOKEN = re.compile(r"[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?")


def engine_lines(harness: str) -> list[list[str]]:
    """The moves of each row of the harness's '## Engine lines' block, one list per row:
    '  1. 1... c6 2. d4 d5 3. e5 Bf5  0.56' -> ['c6', 'd4', 'd5', 'e5', 'Bf5']. Plain
    string work, no regex over free text."""
    if "## Engine lines" not in harness:
        return []
    block = harness.split("## Engine lines", 1)[1].split("\n## ", 1)[0]
    rows: list[list[str]] = []
    for line in block.splitlines():
        line = line.strip()
        head, dot, rest = line.partition(". ")
        if not dot or not head.isdigit():
            continue
        moves = rest.split("  ", 1)[0]
        rows.append([t.rstrip("+#") for t in _SAN_TOKEN.findall(moves)])
    return rows


_EFFECT_PHRASE = re.compile(
    r"\b((?:takes|attacks|pins|forks|skewers|defends|threatens|gives check|gets|checks|wins"
    r"|blocks|opens|closes|controls|frees|develops|centralises|castles)\b[^;.()\n]{3,80})",
    re.IGNORECASE,
)


def effect_phrases(item: dict) -> list[str]:
    """Every effect phrase the harness states of a move ('takes the pawn on c5', 'attacks
    the pawn on b6', 'gets the pawn out of attack'), from the last-move line, the engine
    lines and the record's verdict lines."""
    text = str(item["messages"][-1]["content"]) if item.get("messages") else ""
    out = []
    for phrase in _EFFECT_PHRASE.findall(text):
        phrase = phrase.strip().lower().rstrip(",")
        if len(phrase.split()) >= 2:
            out.append(phrase)
    return out


def effects_check(text: str, item: dict) -> bool | None:
    """A claim that repeats, word for word, an effect the harness states of a move is
    stated by the record: 'dxc5 attacks the pawn on b6' when the harness says 'What dxc5
    does: takes the pawn on c5; attacks the pawn on b6'. Verb forms are matched loosely
    ('removed the pawn from attack' for 'gets the pawn out of attack'). None otherwise."""
    low = text.lower()
    for phrase in effect_phrases(item):
        if phrase in low:
            return True
        # 'gets <piece> out of attack' in other words: the piece and an escape verb.
        m = re.match(r"gets (?:the |your |their )?(\w+)(?: on [a-h][1-8])? out of attack", phrase)
        if (
            m
            and re.search(rf"\b{m.group(1)}\b", low)
            and re.search(
                r"\b(out of|from|away from|escap\w*|remov\w*|sav\w*|unattack\w*) (?:the )?"
                r"(?:attack|danger)|\bout of (?:attack|danger)\b|\bfrom (?:attack|danger)\b"
                r"|\bto safety\b|\bsafe from\b",
                low,
            )
        ):
            return True
    return None


_WORTH = re.compile(r"\bworth (\d+) pawns?\b", re.IGNORECASE)
_PIECE_VALUE = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "exchange": 2}
_GAINS_PIECE = re.compile(
    r"\b(?:up|down|wins?|winning|loses?|losing|gives up|giving up|gains?) (?:a |an |the )?"
    r"(pawn|knight|bishop|rook|queen|exchange)\b",
    re.IGNORECASE,
)


def worth_check(text: str, item: dict) -> bool | None:
    """'material worth 5 pawns' against the record: true when the record says worth that
    many pawns, or names a piece of that value won, lost or held ('up a rook'). None when
    the claim gives no worth."""
    said = _WORTH.search(text)
    if not said:
        return None
    n = int(said.group(1))
    harness = str(item["messages"][-1]["content"]) if item.get("messages") else ""
    if any(int(x) == n for x in _WORTH.findall(harness)):
        return True
    return any(_PIECE_VALUE.get(p.lower()) == n for p in _GAINS_PIECE.findall(harness)) or None


_ALMOST = re.compile(
    r"\b(?:almost|nearly|about|close to|practically|virtually) as (?:strong|good|accurate) as\b"
    r"|\bclose second\b|\balmost matched\b",
    re.IGNORECASE,
)


def almost_check(text: str, item: dict) -> bool | None:
    """'h4 was almost as strong as Qxb7', naming the move just played and the one the
    engine preferred: true when the played move's verdict is excellent (or best) and it
    cost under 0.03. None when the claim is not of that shape."""
    m = item["moment"]
    last = (m.get("last_san") or "").rstrip("+#")
    pref = (m.get("last_best_san") or "").rstrip("+#")
    if not last or not pref or last == pref or not _ALMOST.search(text):
        return None
    named = {x.rstrip("+#") for x in _SAN_IN_TEXT.findall(text)}
    if last not in named or pref not in named:
        return None
    loss = m.get("last_loss")
    return (m.get("last_label") or "") in ("best", "excellent") and loss is not None and loss < 0.03


_THEME_WORD = re.compile(
    r"\b(fork|pin|skewer|discovered attack|discovered check|double check|back rank|back-rank"
    r"|smothered|overload\w*|deflection|decoy|zwischenzug|intermezzo|passed pawn|outpost"
    r"|mating net|mate|checkmate|trapped piece|hanging piece|x-ray|windmill|clearance"
    r"|interference|zugzwang|stalemate|perpetual|opposition|breakthrough|sacrifice"
    r"|exchange sacrifice|attraction|removal of the defender|underpromotion|promotion)\b",
    re.IGNORECASE,
)


def theme_check(text: str, item: dict) -> bool | None:
    """A short claim that names a theme the facts name ('the position is a fork' with
    theme 'a fork') is stated. None otherwise."""
    if len(text.split()) > 10:
        return None
    f = item["moment"]["facts"]
    facts_text = " ".join(
        str(f.get(k) or "") for k in ("theme", "tactic", "nature", "plan", "ideas")
    ).lower()
    said = {w.lower() for w in _THEME_WORD.findall(text)}
    if not said:
        return None
    return any(w in facts_text for w in said) or None


def line_check(text: str, item: dict) -> bool | None:
    """A claim that names two or more moves which appear, in that order, in one row of
    the harness's engine lines is stated by the record. None when the claim names fewer
    than two moves or no row has them in that order."""
    named = [
        x.rstrip("+#") for x in _SAN_IN_TEXT.findall(text) if not re.fullmatch(r"[a-h][1-8]", x)
    ]
    if len(named) < 2:
        return None
    harness = str(item["messages"][-1]["content"]) if item.get("messages") else ""
    for seq in engine_lines(harness):
        pos = 0
        for mv in named:
            try:
                pos = seq.index(mv, pos) + 1
            except ValueError:
                break
        else:
            return True
    return None


_KING_ATTACKED = re.compile(
    r"\b(?:checks?|attacks?|attacking|checking|hits?) (?:your|the player's|my) king\b", re.I
)
_NOTHING_LOST = re.compile(
    r"\b(?:cost|costs|costing|conced(?:es|ed)|los(?:es|t)|gave up|gives up|giving up)"
    r" (?:you |the player )?nothing\b|\bdid not (?:give|lose|concede) anything\b"
    r"|\bcost nothing\b|\bat no cost\b|\bno cost\b",
    re.I,
)
_STRONGEST = re.compile(
    r"\b(?:matches|matched|is|was|equals|equalled) (?:the )?(?:engine's )?"
    r"(?:strongest|best|top) (?:move|choice)\b|\bas strong as the best move\b",
    re.I,
)


def _about_opponents_move(text: str, item: dict) -> bool:
    """Whether the claim names the opponent's last move ("30...Nd5 uncovers ...")."""
    m = item["moment"]
    if m.get("last_by_user") or not m.get("last_san"):
        return False
    san = re.escape(m["last_san"].rstrip("+#"))
    return re.search(rf"(?<![A-Za-z]){san}[+#]?(?![A-Za-z0-9])", text) is not None


def code_check(text: str, item: dict) -> bool | None:
    """Claims the record decides without a model: a count of moves by label against the
    tally; the cost of the last move against its recorded loss. None when the claim is
    not one of these."""
    m = item["moment"]
    tally = m.get("tally") or {}
    if re.search(r"\b(?:opponent|their|they|his|her)\b", text, re.I):
        tally = {}  # the tally is the player's; a count of the opponent's moves is the vote's
    of = (_COUNT_OF.search(text) or _COUNT_WERE.search(text)) if tally else None
    if of:
        n = int(of.group(1))
        total = int(of.group(2)) if of.group(2) else None
        labels = [
            _LABEL_OF.get(x.lower(), x.lower())
            for x in re.findall(
                r"\b(best|excellent|great|brilliant|good|book|inaccurac(?:y|ies)|mistakes?|miss(?:es)?|blunders?)\b",
                of.group(3),
                re.I,
            )
        ]
        if not labels:
            return None
        judged = sum(tally.values())
        return (total is None or total == judged) and n == sum(
            tally.get(x, 0) for x in dict.fromkeys(labels)
        )
    counts = _COUNT.findall(text) if tally else []
    if counts:
        for n, label in counts:
            label = _LABEL_OF.get(label.lower(), label.lower())
            if tally.get(label, 0) != int(n):
                return False
        return True
    recited = line_check(text, item)
    if recited is not None:
        return recited
    effect = effects_check(text, item)
    if effect is not None:
        return effect
    for check in (worth_check, almost_check, theme_check):
        decided = check(text, item)
        if decided is not None:
            return decided
    material = material_check(text, m.get("material"))
    if material is False:
        # The balance after the expected recapture, as the harness states it, holds too.
        after = re.search(
            r"before the expected recapture \S+, then (.+?)(?:\.|$)", last_move_line(item)
        )
        if after:
            later = after.group(1).strip()
            material = material_check(
                text,
                later
                if later.startswith(("level", "you are"))
                else "you are " + later.removeprefix("the player is "),
            )
    if material is not None:
        return material
    # The last move's verdict, in the coach's words: "cost nothing" is its recorded loss;
    # "the strongest move" is a best or an excellent (as strong as the best) verdict.
    last = (m.get("last_san") or "").rstrip("+#")
    names_last = (
        bool(last)
        and re.search(rf"(?<![A-Za-z]){re.escape(last)}[+#]?(?![A-Za-z0-9])", text) is not None
    )
    names_other = any(
        x.rstrip("+#") != last
        for x in _SAN_IN_TEXT.findall(text)
        if not re.fullmatch(r"[a-h][1-8]", x)
    )
    if names_last and not names_other:
        loss = m.get("last_loss")
        if _NOTHING_LOST.search(text) and loss is not None:
            return loss < 0.005
        if _STRONGEST.search(text) and not _NEGATED.search(text):
            return (m.get("last_label") or "") in ("best", "excellent")
    # "Checks your King" / "attacks your King" after the opponent's move: true when the
    # player's king is in check now (the checker attributed the queen to the wrong side).
    if _KING_ATTACKED.search(text) and not _NEGATED.search(text):
        try:
            board = chess.Board(item.get("fen_now") or item["fen"])
            f = m["facts"]
            side = (f.get("user_side") or f.get("side_to_move") or "").lower()
            if (
                side in ("white", "black")
                and board.turn == (chess.WHITE if side == "white" else chess.BLACK)
                and board.is_check()
            ):
                return True
        except ValueError:
            pass
    loss = m.get("last_loss")
    named = [x for x in _SAN_IN_TEXT.findall(text) if not re.fullmatch(r"[a-h][1-8]", x)]
    if loss is not None and re.search(r"\b(cost|costing|costs)\b", text, re.I):
        last = (m.get("last_san") or "").rstrip("+#")
        if named and any(x.rstrip("+#") != last for x in named):
            # The cost of an earlier move: in the record's own lines when it is there.
            costs = record_costs(item)
            for x in named:
                if x.rstrip("+#") in costs:
                    found = _COST.search(text)
                    if not found:
                        return None
                    value = float(found.group(1))
                    unit = (found.group(2) or "").lower()
                    if unit in ("%", "percent") or value > 1.0:
                        value = value / 100.0
                    return abs(value - costs[x.rstrip("+#")]) <= max(
                        0.02, 0.35 * costs[x.rstrip("+#")]
                    )
            return None
        found = _COST.search(text)
        if found:
            value = float(found.group(1))
            unit = (found.group(2) or "").lower()
            if unit in ("%", "percent") or value > 1.0:
                value = value / 100.0
            return abs(value - loss) <= max(0.02, 0.35 * loss)
        for words, (lo, hi) in _COST_WORDS.items():
            if words in text.lower():
                return lo <= loss <= hi
    return None


def judge_one(
    item: dict, rec: dict, provider, ledger: Ledger, draft_check, *, thinking: bool
) -> dict:
    """Deterministic checks; the reply split into atomic claims; then one binary question
    per check, answered without reasoning: does the record state each claim; does the
    reply give each required fact, and the right value; does it answer the question. A
    tactical claim is put to the board checker as well, which rules either way. Raises
    JudgeFailed when the decomposition never parses, so a failed call is never a judgment."""
    det = deterministic(item, rec["reply"], draft_check)
    verdict: dict[str, Any] | None = None
    claims: list[str] = []
    reply = rec["reply"]
    # The judge never reasons: every call -- the split into claims, each vote, the board
    # checker's extractor, the master's point -- is the no-reasoning provider.
    provider = binary_judge(provider)
    if reply.strip():
        record = (
            "The record:\n"
            + evidence_block(item, with_reference=False)
            + '\n\nThe coach\'s reply:\n"""\n'
            + reply
            + '\n"""'
        )
        parsed = None
        raw = ""
        for _attempt in range(2):
            completion = _chat_with_backoff(
                provider,
                BINARY_SYSTEM + "\n\n" + record,
                decompose_messages(reply, record),
                ledger,
                rec["id"],
                max_tokens=4000,
                temperature=0.0,
                thinking=False,
            )
            raw = completion.text
            parsed = parse_judgment(raw)
            if parsed is not None and isinstance(parsed.get("claims"), list):
                break
            parsed = None
        if parsed is None:
            raise JudgeFailed(f"{rec['id']}: the decomposition never parsed: {raw[:120]!r}")
        claims = [
            str(c)
            for c in parsed.get("claims", [])
            if str(c).strip() and _identified(str(c)) and _is_claim_of(str(c), reply)
        ]
        binary = provider
        # Each claim: does the record state it (or does it follow from the move itself)?
        marked: list[dict] = []
        for n, text in enumerate(claims, start=1):
            decided = code_check(text, item)
            if decided is not None:
                answer = decided
                why = "by the record's tally or loss: " + ("right" if decided else "wrong")
            else:
                answer = _yes_no(
                    binary,
                    ledger,
                    rec["id"],
                    record,
                    f'Does the record state this, in these or other words? Statement: "{text}"',
                )
                why = "yes" if answer else ("no" if answer is False else "not judged")
            c: dict[str, Any] = {
                "n": n,
                "supported": answer,
                "why": why,
                "kind": None if answer else ("false" if decided is False else "unstated"),
                "lesson": None,
            }
            opponent_move = _about_opponents_move(text, item)
            if (
                decided is None
                and _TACTICAL.search(text)
                and not _NEGATED.search(text)
                and not opponent_move
            ):
                # The board rules on what a piece attacks, defends, threatens or hangs. A
                # negated claim ("c5 is not an attack on e4") is left to the vote: the
                # checker verifies the positive and would call the true claim false. So is
                # a claim about the opponent's move: the checker reads "the queen" as the
                # player's. A claim the code decided is final.
                ruling, reason, _kind = checker_rules(
                    text, item, _Metered(binary, ledger, rec["id"]), reply
                )
                c["checker"] = ruling
                if ruling == "true":
                    c["supported"], c["kind"], c["why"] = (
                        True,
                        None,
                        "true on the board: " + reason[:200],
                    )
                elif ruling == "false":
                    c["supported"], c["kind"], c["why"] = (
                        False,
                        "false",
                        "false on the board: " + reason[:200],
                    )
            if c["supported"] is None:
                c["supported"] = True  # not judged: the benefit of the doubt, noted
                c["unjudged"] = True
            marked.append(c)
        # Each required fact, one direct question: does the reply say this? A verdict is
        # two existence checks -- praise, criticism -- mapped to the label's tier in code.
        required_marks: list[dict] = []
        for fact in item["required"]:
            if fact["key"] in ("label", "named_verdict") and str(fact["value"]).lower() in _TIERS:
                move = (
                    next((r["value"] for r in item["required"] if r["key"] == "move"), None)
                    or item["moment"].get("last_san")
                    or "the move"
                )
                praised = _yes_no(
                    binary,
                    ledger,
                    rec["id"],
                    record,
                    f"Does the reply praise the move {move} (call it good, best, strong,"
                    " precise, right, or the like)?",
                )
                faulted = _yes_no(
                    binary,
                    ledger,
                    rec["id"],
                    record,
                    f"Does the reply criticise the move {move} (call it a mistake, an"
                    " inaccuracy, a blunder, a miss, weaker than another move, or say it cost"
                    " something)?",
                )
                want = _TIERS[str(fact["value"]).lower()]
                said = (
                    "praise"
                    if praised and not faulted
                    else (
                        "criticism"
                        if faulted and not praised
                        else ("both" if praised and faulted else None)
                    )
                )
                mentioned = said is not None
                correct = (
                    None if not mentioned else (said == "both" or want == "both" or said == want)
                )
                required_marks.append(
                    {"key": fact["key"], "mentioned": mentioned, "correct": correct, "lesson": None}
                )
                continue
            if fact["key"] == "opening":
                stated = opening_named(reply, str(fact["value"]))
                required_marks.append(
                    {
                        "key": fact["key"],
                        "mentioned": stated,
                        "correct": True if stated else None,
                        "lesson": None,
                    }
                )
                continue
            words = _required_words(fact)
            stated = _yes_no(
                binary,
                ledger,
                rec["id"],
                record,
                f'Does the reply say this about {words}, in its own words: "{fact["value"]}"?',
            )
            required_marks.append(
                {
                    "key": fact["key"],
                    "mentioned": bool(stated),
                    "correct": True if stated else None,
                    "lesson": None,
                }
            )
        relevance = _relevance(item, reply)
        idea = None
        if item.get("reference"):
            point = _chat_with_backoff(
                provider,
                "You read a chess master's note on a move and state its one point in a single"
                " short sentence: the judgment on the move, or the one reason given, or the"
                " better move it names. No variations. Plain text.",
                [{"role": "user", "content": "The note:\n" + item["reference"][:2500]}],
                ledger,
                rec["id"],
                max_tokens=120,
                temperature=0.0,
                thinking=False,
            ).text.strip()
            if point:
                in_reply = _yes_no(
                    binary,
                    ledger,
                    rec["id"],
                    record,
                    f'Does the reply make this point, in any words: "{point}"?',
                )
                in_facts = _yes_no(
                    binary,
                    ledger,
                    rec["id"],
                    record,
                    f'Does the record state this point, in any words: "{point}"?',
                )
                idea = {
                    "point": point,
                    "in_reply": bool(in_reply),
                    "in_facts": bool(in_facts),
                    "where": None,
                }
        verdict = {
            "claims": marked,
            "required": required_marks,
            "relevance": relevance,
            "idea": idea,
        }
    return {
        "id": rec["id"],
        "class": item["class"],
        "source": item["source"],
        "level": item["level"],
        "reply": reply,
        "raw_candidates": rec.get("raw", ""),
        "problems": rec.get("problems", []),
        "fallback_used": rec.get("fallback_used", False),
        "deterministic": det,
        "claims": claims,
        "verdict": verdict,
        "judge_raw": "",
        "scores": score(item, det, verdict, claims),
    }


_BINARY: dict[int, Any] = {}


def binary_judge(provider):
    """The provider for the yes-or-no checks: the judge's model with reasoning off."""
    key = id(provider)
    if key not in _BINARY:
        from providers import without_reasoning

        _BINARY[key] = without_reasoning(provider)
    return _BINARY[key]


def judge_all(
    items: dict[str, dict],
    replies: list[dict],
    run: str,
    provider,
    ledger: Ledger,
    *,
    thinking: bool,
    quiet: bool = False,
    workers: int = 8,
) -> list[dict]:
    """Judge every reply not yet judged under `run`, `workers` calls in flight; returns all
    judgments for `replies`. The budget is checked before each call is submitted."""
    out_dir = JUDGMENTS / run
    out_dir.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    for path in out_dir.glob("*.jsonl"):
        for line in path.open(encoding="utf-8"):
            rec = json.loads(line)
            done[rec["id"]] = rec
    draft_check = _draft_check()
    n = 0
    lock = threading.Lock()
    todo = [r for r in replies if r["id"] not in done]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for rec in todo:
            if ledger.over_budget():
                print("budget reached; not submitting more", flush=True)
                break
            futures[
                pool.submit(
                    judge_one,
                    items[rec["id"]],
                    rec,
                    provider,
                    ledger,
                    draft_check,
                    thinking=thinking,
                )
            ] = rec
        failed = 0
        bar = tqdm(
            total=len(futures),
            desc=f"judge {run}",
            unit="item",
            file=sys.stderr,
            mininterval=5,
            disable=False,
        )
        for fut in as_completed(futures):
            rec = futures[fut]
            bar.update(1)
            try:
                judged = fut.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                if failed <= 5:
                    print(
                        f"  judge failed on {rec['id']}: {type(exc).__name__}: {str(exc)[:160]}",
                        flush=True,
                    )
                continue
            item = items[rec["id"]]
            with lock:
                out_path = out_dir / f"{item['source']}__{item['game'].replace('/', '__')}.jsonl"
                with out_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(judged, ensure_ascii=False) + "\n")
                done[rec["id"]] = judged
                n += 1
        bar.close()
    if failed:
        print(
            f"  {failed} replies left unjudged (the judge gave no usable answer); a later"
            " pass judges them",
            flush=True,
        )
    if not quiet:
        print(f"judged {n} new; {ledger.summary()}", flush=True)
    return [done[r["id"]] for r in replies if r["id"] in done]
