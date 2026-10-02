"""The player's games added up (docs/stats-plan.md): the Stats window's numbers, from the facts
each review keeps (profile/stat_facts.py), for the player's own moves in the games the filter
takes. Every figure carries how many it counts (`n`), so the page can grey a small sample; the
positions behind a figure come from `positions` with the same selector.

Beside the player's figures: a rating a category (the game rating's model fitted on that
category's moves alone, `categories.json`), and every figure the peers' -- the same figure for
real players of the player's rating (of their play, when no game records it), from the same
functions run over the judged Lichess sides (`peers.json`; both fitted by
`tools/review/fit_stats.py`) -- with a grade for how far ahead or behind the player's figure
is, in rating points.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from statistics import fmean
from typing import Any

from sidecoach.profile import performance
from sidecoach.profile.game_summary import SHAPES
from sidecoach.profile.stat_facts import (
    ENDGAMES,
    KINDS,
    STRUCTURES,
    THEMES,
    standing,
    termination_kind,
)
from sidecoach.profile.store import Store

BAD = ("mistake", "miss", "blunder")
# A move under this much clock is played under time pressure, by the time control.
PRESSURE_MS = {"bullet": 10_000, "blitz": 20_000, "rapid": 60_000}
GALLERIES = ("brilliant", "great", "miss", "blunder")

# The moves each category is rated on, and its trend and figures drawn from.
CATEGORIES: dict[str, Callable[[dict[str, Any]], bool]] = {
    "openings": lambda m: m["phase"] == "opening",
    "tactics": lambda m: m["chance"] is not None,
    "strategy": lambda m: m["chance"] is None and m["label"] != "book",
    "endgames": lambda m: m["phase"] == "endgame",
}
CATEGORIES_PATH = Path(__file__).with_name("categories.json")
PEERS_PATH = Path(__file__).with_name("peers.json")
# A grade: how far ahead of the peers the player's figure is, in rating points (the peers' line
# read backwards), as the kinds of move say it; below the last, a blunder.
GRADES = (
    (400, "brilliant"),
    (250, "great"),
    (125, "best"),
    (40, "excellent"),
    (-40, "good"),
    (-125, "inaccuracy"),
    (-250, "mistake"),
)
# A figure of a few is pulled towards the peers' as if it had so many more at their value; a
# figure the peers' line hardly follows (its fit's weighted r² below this) gets no grade.
SHRINK = 10
FOLLOWS = 0.5
# A category's rating from this many of its moves: a tactic chance comes about twice a game, and
# ten games' chances still leave the rating some 280 points out (tools/review/fit_stats.py).
CATEGORY_MOVES = 20
# The games the rating's trend pools at each game (the game and those before it).
RATING_GAMES = 10


@dataclass(frozen=True)
class Filter:
    time_class: str | None = None  # bullet, blitz, rapid, daily; None for all
    days: int | None = None  # the last so many days; None for all time
    opponents: str = "all"  # all, people, bots


def _games(store: Store, f: Filter) -> list[dict[str, Any]]:
    """The games the filter takes, whose player's side is known, oldest first."""
    since = time.time() - f.days * 86400 if f.days else None
    out = []
    for g in store.stat_games():
        if g["user_side"] not in ("white", "black"):
            continue
        if f.time_class and g["time_class"] != f.time_class:
            continue
        if since is not None and (g["played_at"] or 0) < since:
            continue
        if (f.opponents == "people" and g["bot"]) or (f.opponents == "bots" and not g["bot"]):
            continue
        out.append(g)
    return out


def _score(result: str | None, side: str) -> float | None:
    """The player's points from the game: 1, a half, 0."""
    if result == "1/2-1/2":
        return 0.5
    if result in ("1-0", "0-1"):
        return 1.0 if (result == "1-0") == (side == "white") else 0.0
    return None


def _figure(values: Iterable[float]) -> dict[str, Any]:
    """A mean, with how many it is of."""
    items = list(values)
    return {"value": round(fmean(items), 1) if items else None, "n": len(items)}


def _rate(hits: int, total: int) -> dict[str, Any]:
    return {"value": round(100 * hits / total, 1) if total else None, "hits": hits, "n": total}


def family(opening: str | None) -> str | None:
    """An opening's family: the name before its variation, without the ECO code."""
    if not opening:
        return None
    name = re.sub(r"^[A-E]\d\d\s+", "", opening)
    return name.split(":")[0].strip()


class _Data:
    """Games (their `key`, `user_side`, `result`...) and their moves, both sides'; the player's
    moves apart. From the store (`of`), or from any rows shaped as its (the peers' sample)."""

    def __init__(self, games: list[dict[str, Any]], moves: Iterable[dict[str, Any]]) -> None:
        self.games = games
        by_key = {g["key"]: g for g in games}
        self.moves_by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.all_by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for m in moves:
            self.all_by_game[m["key"]].append(m)
            if m["side"] == by_key[m["key"]]["user_side"]:
                self.moves_by_game[m["key"]].append(m)
        self.moves = [m for g in games for m in self.moves_by_game[g["key"]]]

    @classmethod
    def of(cls, store: Store, f: Filter) -> _Data:
        """The filter's games in the store, each with the player's rating in it when the game
        records one (the library's copy of its headers)."""
        games = _games(store, f)
        for g in games:
            row = store.library_game(g["key"])
            elo = (
                None
                if row is None
                else row.white_elo
                if g["user_side"] == "white"
                else row.black_elo
            )
            g["user_elo"] = elo
        return cls(games, store.stat_moves([g["key"] for g in games]))


def stats(store: Store, f: Filter) -> dict[str, Any]:
    """Every section's numbers for the filter's games, the ratings, and the peers beside each
    figure."""
    d = _Data.of(store, f)
    out = {"filter": f.__dict__, **figures(d), "trends": _trends(d), "shapes": _shapes(store, d)}
    level = _level(d)
    out["ratings"] = _ratings(d, level)
    peers = _peers_level(d, level)
    out["peers"] = None
    if peers is not None:
        compare(out, peers[0])
        out["peers"] = {"rating": performance.for_site(peers[0], _site(d)), "from": peers[1]}
    out["work"] = work(out["costs"])
    # The reviewed games whose player's side is not known (no username of theirs in them): the
    # page says why they are left out, rather than showing nothing (the review of 2026-10-01).
    out["unsided"] = sum(1 for g in store.stat_games() if g["user_side"] not in ("white", "black"))
    return out


def figures(d: _Data) -> dict[str, Any]:
    """The sections' figures for `d`: the player's, or the peers' (the same code for both)."""
    return {
        "overview": _overview(d),
        "openings": _openings(d),
        "tactics": _tactics(d),
        "strategy": _strategy(d),
        "endgames": _endgames(d),
        "time": _time(d),
        "results": _results(d),
        "galleries": {k: sum(1 for m in d.moves if m["label"] == k) for k in GALLERIES},
        "costs": _costs(d),
    }


def walk(section: Any, path: str = "") -> Iterator[tuple[str, dict[str, Any]]]:
    """Every figure of a section (a dict with its `value` and `n`), by its dotted path."""
    if not isinstance(section, dict):
        return
    if "value" in section and "n" in section:
        yield path, section
        return
    for key, inner in section.items():
        yield from walk(inner, f"{path}.{key}" if path else key)


# ---- what the player's game costs them, and what to work on ----------------------------------

# An area's puzzles: its motif as the puzzle store names it (Lichess's themes); none where no
# puzzle drills it (a quiet position, a tactic of no named kind).
MOTIFS = {
    "missed.checkmate": "mate",
    "missed.fork": "fork",
    "missed.pin": "pin",
    "missed.discovery": "discoveredAttack",
    "missed.trapped piece": "trappedPiece",
    "missed.removing the defender": "capturingDefender",
    "missed.free piece": "hangingPiece",
    "missed.sacrifice": "sacrifice",
    "allowed.checkmate": "mate",
    "allowed.fork": "fork",
    "allowed.pin": "pin",
    "allowed.discovery": "discoveredAttack",
    "allowed.trapped piece": "trappedPiece",
    "allowed.removing the defender": "capturingDefender",
    "allowed.free piece": "hangingPiece",
    "opening": "opening",
    "endgame.pawn": "pawnEndgame",
    "endgame.rook": "rookEndgame",
    "endgame.queen": "queenEndgame",
    "endgame.other": "endgame",
}
WORK = 3  # the areas Work on this names
WORK_GAMES = 3  # the games before it names any
WORK_MOVES = 3  # the moves an area needs before it is named


def area(m: dict[str, Any]) -> str | None:
    """The one area a move's cost counts in, when it cost anything worth a word: a tactic it
    missed (by kind), else, for a mistake, miss or blunder, a tactic it handed over (by kind), the
    endgame's (by kind), the opening's, a quiet position's."""
    if m["chance"] and not m["found"]:
        return f"missed.{m['chance']}"
    if m["label"] not in BAD:
        return None
    if m["allowed"]:
        return f"allowed.{m['allowed']}"
    if m["phase"] == "endgame":
        return f"endgame.{m['endgame'] or 'other'}"
    return "opening" if m["phase"] == "opening" else "quiet"


def _costs(d: _Data) -> dict[str, Any]:
    """Each area's cost: the expected score its moves lost, a game (`n`, the games), and how
    many moves it counts (`moves`). The cost ranks the areas and meets the peers'; it is no
    count of points: a position lost can swing back, and the losses add up past the games
    lost ("11.1 points" for one area when 9 were lost in all, the review of 2026-10-01)."""
    lost: dict[str, float] = defaultdict(float)
    count: dict[str, int] = defaultdict(int)
    for m in d.moves:
        where = area(m)
        if where is not None:
            lost[where] += m["loss"]
            count[where] += 1
    games = len(d.games)
    names = [
        *(f"missed.{k}" for k in KINDS),
        *(f"allowed.{k}" for k in KINDS),
        "opening",
        *(f"endgame.{k}" for k in ENDGAMES),
        "quiet",
    ]
    out: dict[str, Any] = {}
    for name in names:
        figure = {
            "value": round(lost[name] / games, 4) if games else None,
            "n": games,
            "moves": count[name],
        }
        head, _, kind = name.partition(".")
        if kind:
            out.setdefault(head, {})[kind] = figure
        else:
            out[head] = figure
    return out


def work(costs: dict[str, Any]) -> list[dict[str, Any]]:
    """The areas that cost the player most (WORK of them), dearest first: each with its cost a
    game and its peers' (when known), the moves it counts, and its puzzles' motif; its
    positions are `positions`' "area" selector's. Nothing over fewer than WORK_GAMES games, and
    no area of fewer than WORK_MOVES moves: one game's blunder is no area to work on."""
    ranked = sorted(
        (
            (path, figure)
            for path, figure in walk(costs)
            if figure["value"] and figure["n"] >= WORK_GAMES and figure["moves"] >= WORK_MOVES
        ),
        key=lambda pf: -pf[1]["value"],
    )
    out = []
    for path, figure in ranked[:WORK]:
        out.append(
            {
                "area": path,
                "cost": figure["value"],
                "games": figure["n"],
                "moves": figure["moves"],
                "peers": figure.get("peers"),
                "grade": figure.get("grade"),
                "motif": MOTIFS.get(path),
            }
        )
    return out


# ---- the ratings and the peers ---------------------------------------------------------------


def _rated(moves: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Moves as the rating model reads them: the plies before each (by the position it was
    played from: a game set up from a position starts at its own move), the loss, the mover's
    expected score."""
    return [
        {"ply": performance.plies_before(m["fen"]), "loss": m["loss"], "score": m["score"]}
        for m in moves
    ]


def _site(d: _Data) -> str | None:
    """The scale the page speaks: Lichess's when every game is a Lichess game."""
    return "lichess" if d.games and all(g["source"] == "lichess" for g in d.games) else None


def _level(d: _Data) -> float | None:
    """The player's rating over all their moves here (Lichess rapid), their level of play. None
    for too few moves."""
    if len(d.moves) < performance.MIN_MOVES:
        return None
    return float(performance.estimate(_rated(d.moves)).rating)


def _peers_level(d: _Data, level: float | None) -> tuple[float, str] | None:
    """Who the player is compared with, on Lichess's scale, and why: players of their rating (the
    mean of the ratings their games against people record, as chess.com's peers are), else of
    their level of play (`level`). Against weaker players accuracy comes easier, so one who plays
    above their rating is ahead of their rating's peers and, figure for figure, often of their
    play's too: their own rating is the comparison that holds."""
    rated = [
        g["user_elo"] if g["source"] == "lichess" else performance.lichess(g["user_elo"])
        for g in d.games
        if g.get("user_elo") and not g["bot"]
    ]
    if rated:
        return fmean(rated), "rating"
    return None if level is None else (level, "play")


@cache
def _category_models() -> dict[str, Any]:
    return json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))


@cache
def _peer_lines() -> dict[str, dict[str, float]]:
    return json.loads(PEERS_PATH.read_text(encoding="utf-8"))["figures"]


def _ratings(d: _Data, level: float | None) -> dict[str, Any]:
    """The player's rating over all their moves, and each category's over its own moves, on the
    page's scale; None where the moves are too few."""
    site = _site(d)
    overall = None if level is None else performance.for_site(level, site)
    out: dict[str, Any] = {
        "overall": None if overall is None else {"rating": overall, "moves": len(d.moves)}
    }
    for name, select in CATEGORIES.items():
        moves = [m for m in d.moves if select(m)]
        if len(moves) < CATEGORY_MOVES:
            out[name] = None
            continue
        p = performance.estimate(_rated(moves), model=_category_models()[name])
        out[name] = {
            "rating": performance.for_site(p.rating, site),
            "low": performance.for_site(p.low, site),
            "high": performance.for_site(p.high, site),
            "moves": p.moves,
        }
    return out


def grade(value: float, n: int, line: dict[str, float], level: float) -> tuple[float, str | None]:
    """The peers' figure at `level` (Lichess rapid) by their fitted line, and the player's grade:
    how many rating points ahead of it the figure is (read back along the line), a figure of a
    few pulled towards the peers' first. No grade where the line hardly follows the rating."""
    rating = min(max(level, line["lo"]), line["hi"])  # the sample's range, not beyond
    peer = line["a"] + line["b"] * rating
    if line["r2"] < FOLLOWS or line["b"] == 0:
        return peer, None
    shrunk = (n * value + SHRINK * peer) / (n + SHRINK)
    ahead = (shrunk - peer) / line["b"]
    return peer, next((kind for edge, kind in GRADES if ahead >= edge), "blunder")


def compare(out: dict[str, Any], level: float) -> None:
    """Every figure with a peers' line gets the peers' value and the player's grade, in place."""
    lines = _peer_lines()
    for path, figure in walk(out):
        line = lines.get(path)
        if line is None or figure["value"] is None:
            continue
        peer, kind = grade(figure["value"], figure["n"], line, level)
        figure["peers"] = round(peer, 2 if path.startswith("tactics.allowed") else 1)
        figure["grade"] = kind


def _game_rating(game: dict[str, Any], moves: list[dict[str, Any]]) -> int | None:
    """The rating the player played the game at (profile/performance.py), on the scale of the
    game's site."""
    if len(moves) < performance.MIN_MOVES:
        return None
    played = _rated(moves)
    site = "lichess" if game["source"] == "lichess" else None
    return performance.for_site(performance.estimate(played).rating, site)


def _trends(d: _Data) -> dict[str, list[dict[str, Any]]]:
    """Game by game, oldest first: the player's rating as of each game (`v`, their moves over it
    and the games before it, RATING_GAMES in all, pooled; `game`, the game's own), and each
    category's figure in the games that have one (the opening's accuracy, the tactics found, the
    strategic accuracy, the endgame's accuracy).

    Pooled, not a mean of each game's estimate: one game's estimate is pulled to the middle by the
    prior it needs, so their mean is too (on the held-out sides of tools/review/fit_stats.py, 300
    to 400 points for players over 2100 or under 1200); the moves of ten games pooled are not."""

    def mean(moves: list[dict[str, Any]]) -> float | None:
        return round(fmean(m["accuracy"] or 0.0 for m in moves), 1) if moves else None

    def found(moves: list[dict[str, Any]]) -> float | None:
        chances = [m for m in moves if m["chance"]]
        return round(100 * sum(m["found"] for m in chances) / len(chances), 1) if chances else None

    pickers: dict[str, Callable[[dict[str, Any], list[dict[str, Any]]], float | None]] = {
        "openings": lambda _g, ms: mean([m for m in ms if m["phase"] == "opening"]),
        "tactics": lambda _g, ms: found(ms),
        "strategy": lambda _g, ms: mean([m for m in ms if CATEGORIES["strategy"](m)]),
        "endgames": lambda _g, ms: mean([m for m in ms if m["phase"] == "endgame"]),
    }
    out: dict[str, list[dict[str, Any]]] = {"rating": [], **{name: [] for name in pickers}}
    site = _site(d)
    recent: list[list[dict[str, Any]]] = []
    for g in d.games:
        moves = d.moves_by_game[g["key"]]
        own = _game_rating(g, moves)
        if own is not None:
            recent = [*recent[-(RATING_GAMES - 1) :], moves]
            pooled = performance.estimate(_rated(m for ms in recent for m in ms)).rating
            point = {"t": g["played_at"], "key": g["key"], "v": performance.for_site(pooled, site)}
            out["rating"].append({**point, "game": own})
        for name, pick in pickers.items():
            value = pick(g, moves)
            if value is not None:
                out[name].append({"t": g["played_at"], "key": g["key"], "v": value})
    return out


def _overview(d: _Data) -> dict[str, Any]:
    scores = [_score(g["result"], g["user_side"]) for g in d.games]
    accuracies = [
        fmean(m["accuracy"] or 0.0 for m in d.moves_by_game[g["key"]])
        for g in d.games
        if d.moves_by_game[g["key"]]
    ]
    return {
        "games": len(d.games),
        "wins": sum(1 for s in scores if s == 1.0),
        "draws": sum(1 for s in scores if s == 0.5),
        "losses": sum(1 for s in scores if s == 0.0),
        "accuracy": _figure(accuracies),
        "moves": len(d.moves),
    }


def _left_book_first(moves: list[dict[str, Any]], side: str) -> bool:
    """Whether the player played the game's first move out of book."""
    first = next((m for m in moves if m["label"] != "book"), None)
    return first is not None and first["side"] == side


def _openings(d: _Data) -> dict[str, Any]:
    opening_moves = [m for m in d.moves if m["phase"] == "opening"]
    mistakes = [
        sum(1 for m in d.moves_by_game[g["key"]] if m["phase"] == "opening" and m["label"] in BAD)
        for g in d.games
    ]
    book = [
        sum(1 for m in d.moves_by_game[g["key"]] if m["label"] == "book")
        for g in d.games
        if _left_book_first(d.all_by_game[g["key"]], g["user_side"])
    ]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for g in d.games:
        name = family(g["opening"])
        if name:
            groups[(name, g["user_side"])].append(g)
    by_opening = []
    for (name, side), games in groups.items():
        scores = [s for g in games if (s := _score(g["result"], side)) is not None]
        accuracies = [
            fmean(m["accuracy"] or 0.0 for m in d.moves_by_game[g["key"]])
            for g in games
            if d.moves_by_game[g["key"]]
        ]
        by_opening.append(
            {
                "opening": name,
                "side": side,
                "games": len(games),
                "score": round(100 * fmean(scores), 1) if scores else None,
                "accuracy": round(fmean(accuracies), 1) if accuracies else None,
            }
        )
    by_opening.sort(key=lambda o: (-o["games"], o["opening"]))
    return {
        "accuracy": _figure(m["accuracy"] or 0.0 for m in opening_moves),
        "mistakes": _figure(mistakes),
        "book_moves": _figure(book),
        "by_opening": by_opening,
    }


def _tactics(d: _Data) -> dict[str, Any]:
    found = {
        kind: _rate(
            sum(1 for m in d.moves if m["chance"] == kind and m["found"]),
            sum(1 for m in d.moves if m["chance"] == kind),
        )
        for kind in KINDS
    }
    total = len(d.moves)
    allowed = {}
    for kind in KINDS:
        count = sum(1 for m in d.moves if m["allowed"] == kind)
        allowed[kind] = {
            "value": round(1000 * count / total, 2) if total else None,
            "hits": count,
            "n": total,
        }
    chances = [m for m in d.moves if m["chance"]]
    return {
        "found": found,
        "allowed": allowed,
        "overall": _rate(sum(1 for m in chances if m["found"]), len(chances)),
    }


def _strategy(d: _Data) -> dict[str, Any]:
    quiet = [m for m in d.moves if CATEGORIES["strategy"](m)]
    out = {}
    for structure in ("all", *STRUCTURES):
        moves = quiet if structure == "all" else [m for m in quiet if m["structure"] == structure]
        row = {
            "overall": _figure(m["accuracy"] or 0.0 for m in moves),
            "attacking": _figure(
                m["accuracy"] or 0.0 for m in moves if standing(m["score"]) == "better"
            ),
            "defending": _figure(
                m["accuracy"] or 0.0 for m in moves if standing(m["score"]) == "worse"
            ),
        }
        for theme in THEMES:
            row[theme] = _figure(
                m["accuracy"] or 0.0 for m in moves if theme in (m["themes"] or "").split(",")
            )
        out[structure] = row
    return out


def _endgames(d: _Data) -> dict[str, Any]:
    entered: dict[str, list[float]] = {"better": [], "equal": [], "worse": []}
    for g in d.games:
        first = next((m for m in d.all_by_game[g["key"]] if m["phase"] == "endgame"), None)
        score = _score(g["result"], g["user_side"])
        if first is None or score is None:
            continue
        mine = first["score"] if first["side"] == g["user_side"] else 1.0 - first["score"]
        entered[standing(mine)].append(score)
    endgame = [m for m in d.moves if m["phase"] == "endgame"]
    return {
        "win": {
            k: {"value": round(100 * fmean(v), 1) if v else None, "n": len(v)}
            for k, v in entered.items()
        },
        "accuracy": {
            "overall": _figure(m["accuracy"] or 0.0 for m in endgame),
            **{
                kind: _figure(m["accuracy"] or 0.0 for m in endgame if m["endgame"] == kind)
                for kind in ENDGAMES
            },
        },
    }


def _time(d: _Data) -> dict[str, Any]:
    timed = [m for m in d.moves if m["spent_ms"] is not None]
    by_phase = {
        phase: _figure(m["spent_ms"] / 1000 for m in timed if m["phase"] == phase)
        for phase in ("opening", "middlegame", "endgame")
    }
    pressed, calm = [], []
    for g in d.games:
        limit = PRESSURE_MS.get(g["time_class"] or "")
        if limit is None:
            continue
        for m in d.moves_by_game[g["key"]]:
            if m["clock_ms"] is not None:
                (pressed if m["clock_ms"] < limit else calm).append(m)
    lost_on_time = 0
    lost_on_time_winning = 0
    for g in d.games:
        if termination_kind(g["termination"]) != "time" or _score(g["result"], g["user_side"]):
            continue
        lost_on_time += 1
        mine = d.moves_by_game[g["key"]]
        if mine and standing(mine[-1]["score"]) == "better":
            lost_on_time_winning += 1
    return {
        "by_phase": by_phase,
        "pressure": _rate(sum(1 for m in pressed if m["label"] in BAD), len(pressed)),
        "calm": _rate(sum(1 for m in calm if m["label"] in BAD), len(calm)),
        "lost_on_time": lost_on_time,
        "lost_on_time_winning": lost_on_time_winning,
    }


def _shapes(store: Store, d: _Data) -> list[dict[str, Any]]:
    """How the player's games went by their shape (game_summary.shape_of, as each review's
    summary has it: smooth, balanced, sudden...): how many, and won, drawn and lost -- the
    plan's G, which kinds of game suit the player. Only the shapes their games had, in
    SHAPES' order."""
    counts: dict[str, dict[str, int]] = {}
    for g in d.games:
        review = store.review(g["key"], g["strength"])
        try:
            shape = json.loads(review.summary).get("shape") if review and review.summary else None
        except ValueError:
            shape = None
        if not shape:
            continue
        c = counts.setdefault(shape, {"games": 0, "won": 0, "drawn": 0, "lost": 0})
        c["games"] += 1
        score = _score(g["result"], g["user_side"])
        if score is not None:
            c["won" if score == 1.0 else "drawn" if score == 0.5 else "lost"] += 1
    return [{"shape": s, **counts[s]} for s in SHAPES if s in counts]


def _results(d: _Data) -> dict[str, Any]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"won": 0, "drawn": 0, "lost": 0})
    for g in d.games:
        score = _score(g["result"], g["user_side"])
        if score is None:
            continue
        outcome = "won" if score == 1.0 else "drawn" if score == 0.5 else "lost"
        out[termination_kind(g["termination"])][outcome] += 1
    return dict(out)


# ---- the positions behind a figure --------------------------------------------------------------

_SELECT: dict[str, Callable[[dict[str, Any], str], bool]] = {
    "opening.mistakes": lambda m, _: m["phase"] == "opening" and m["label"] in BAD,
    "tactics.found": lambda m, kind: m["chance"] == kind and m["found"],
    "tactics.missed": lambda m, kind: m["chance"] == kind and not m["found"],
    "tactics.allowed": lambda m, kind: m["allowed"] == kind,
    "strategy.mistakes": lambda m, structure: (
        m["chance"] is None
        and m["label"] in BAD
        and (structure in ("", "all") or m["structure"] == structure)
    ),
    "endgame.mistakes": lambda m, kind: (
        m["phase"] == "endgame"
        and m["label"] in BAD
        and (kind in ("", "overall") or m["endgame"] == kind)
    ),
    "time.pressure": lambda m, _: m["label"] in BAD,  # of the moves under pressure
    "gallery": lambda m, label: m["label"] == label,
    "area": lambda m, name: area(m) == name,
}
POSITIONS_MAX = 60


def positions(store: Store, f: Filter, what: str, arg: str = "") -> list[dict[str, Any]]:
    """The player's moves behind a figure, the latest first: `what` a selector of `_SELECT`,
    `arg` its kind (a tactic's, an endgame's, a structure, a gallery's label)."""
    select = _SELECT.get(what)
    if select is None:
        return []
    d = _Data.of(store, f)
    out = []
    for g in reversed(d.games):
        limit = PRESSURE_MS.get(g["time_class"] or "")
        for m in d.moves_by_game[g["key"]]:
            if what == "time.pressure" and (
                limit is None or m["clock_ms"] is None or m["clock_ms"] >= limit
            ):
                continue
            if select(m, arg):
                out.append(
                    {
                        "key": g["key"],
                        "ply": m["ply"],
                        "fen": m["fen"],
                        "uci": m["uci"],
                        "san": m["san"],
                        "label": m["label"],
                        "side": m["side"],
                        "played_at": g["played_at"],
                        "opening": g["opening"],
                    }
                )
        if len(out) >= POSITIONS_MAX:
            break
    return out[:POSITIONS_MAX]
