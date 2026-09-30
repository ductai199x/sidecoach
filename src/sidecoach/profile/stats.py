"""The player's games added up (docs/stats-plan.md): the Stats window's numbers, from the facts
each review keeps (profile/stat_facts.py), for the player's own moves in the games the filter
takes. Every figure carries how many it counts (`n`), so the page can grey a small sample; the
positions behind a figure come from `positions` with the same selector.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from sidecoach.profile import performance
from sidecoach.profile.stat_facts import ENDGAMES, KINDS, STRUCTURES, THEMES, standing
from sidecoach.profile.store import Store

BAD = ("mistake", "miss", "blunder")
# A move under this much clock is played under time pressure, by the time control.
PRESSURE_MS = {"bullet": 10_000, "blitz": 20_000, "rapid": 60_000}
GALLERIES = ("brilliant", "great", "miss", "blunder")


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


def termination_kind(text: str | None) -> str:
    """How a game ended, from chess.com's and Lichess's Termination header."""
    t = (text or "").lower()
    for word, kind in (
        ("checkmate", "checkmate"),
        ("resign", "resignation"),
        ("on time", "time"),
        ("time forfeit", "time"),
        ("timeout", "time"),
        ("abandon", "abandoned"),
        ("agreement", "agreement"),
        ("repetition", "repetition"),
        ("stalemate", "stalemate"),
        ("insufficient", "insufficient material"),
        ("50", "fifty moves"),
    ):
        if word in t:
            return kind
    return "other"


class _Data:
    """The filter's games and the player's moves in them."""

    def __init__(self, store: Store, f: Filter) -> None:
        self.filter = f
        self.games = _games(store, f)
        by_key = {g["key"]: g for g in self.games}
        self.moves_by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.all_by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for m in store.stat_moves(list(by_key)):
            self.all_by_game[m["key"]].append(m)
            if m["side"] == by_key[m["key"]]["user_side"]:
                self.moves_by_game[m["key"]].append(m)
        self.moves = [m for g in self.games for m in self.moves_by_game[g["key"]]]


def stats(store: Store, f: Filter) -> dict[str, Any]:
    """Every section's numbers for the filter's games."""
    d = _Data(store, f)
    return {
        "filter": f.__dict__,
        "overview": _overview(d),
        "openings": _openings(d),
        "tactics": _tactics(d),
        "strategy": _strategy(d),
        "endgames": _endgames(d),
        "time": _time(d),
        "results": _results(d),
        "galleries": {k: sum(1 for m in d.moves if m["label"] == k) for k in GALLERIES},
        "trends": _trends(d),
    }


def _game_rating(game: dict[str, Any], moves: list[dict[str, Any]]) -> int | None:
    """The rating the player played the game at (profile/performance.py), on the scale of the
    game's site."""
    if len(moves) < performance.MIN_MOVES:
        return None
    played = [{"ply": m["ply"] - 1, "loss": m["loss"], "score": m["score"]} for m in moves]
    site = "lichess" if game["source"] == "lichess" else None
    return performance.for_site(performance.estimate(played).rating, site)


def _trends(d: _Data) -> dict[str, list[dict[str, Any]]]:
    """Game by game, oldest first: the player's game rating, and each category's figure in the
    games that have one (the opening's accuracy, the tactics found, the strategic accuracy, the
    endgame's accuracy)."""

    def mean(moves: list[dict[str, Any]]) -> float | None:
        return round(fmean(m["accuracy"] or 0.0 for m in moves), 1) if moves else None

    def found(moves: list[dict[str, Any]]) -> float | None:
        chances = [m for m in moves if m["chance"]]
        return round(100 * sum(m["found"] for m in chances) / len(chances), 1) if chances else None

    pickers: dict[str, Callable[[dict[str, Any], list[dict[str, Any]]], float | None]] = {
        "rating": _game_rating,
        "openings": lambda _g, ms: mean([m for m in ms if m["phase"] == "opening"]),
        "tactics": lambda _g, ms: found(ms),
        "strategy": lambda _g, ms: mean(
            [m for m in ms if m["chance"] is None and m["label"] != "book"]
        ),
        "endgames": lambda _g, ms: mean([m for m in ms if m["phase"] == "endgame"]),
    }
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in pickers}
    for g in d.games:
        moves = d.moves_by_game[g["key"]]
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
    quiet = [m for m in d.moves if m["chance"] is None and m["label"] != "book"]
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
}
POSITIONS_MAX = 60


def positions(store: Store, f: Filter, what: str, arg: str = "") -> list[dict[str, Any]]:
    """The player's moves behind a figure, the latest first: `what` a selector of `_SELECT`,
    `arg` its kind (a tactic's, an endgame's, a structure, a gallery's label)."""
    select = _SELECT.get(what)
    if select is None:
        return []
    d = _Data(store, f)
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
