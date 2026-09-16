"""Chess.com's game review as a reference to calibrate against.

Chess.com's review is not ground truth: it runs its own engine at its own depth with its own
thresholds. It is what the user sees, though, and a coach whose verdicts disagree with the
review the user reads afterwards loses trust. So the review is a calibration reference: the
thresholds are tuned until the two vocabularies agree on most moves, and the moves they
still disagree on are looked at one by one.

The review cannot be exported, but the move list's HTML can be copied out of the browser's
developer tools by hand. Each move is a `div.node` whose `span.node-highlight-content` has
the classification as a CSS colour name (`--color-classification-miss`) when the move is
annotated, and each ply has a `data-time` in tenths of a second. Only notable moves carry a
label in the list: the last book move, the best reply to a blunder, and every inaccuracy,
mistake, miss, blunder, great and brilliant move. Ordinary moves carry none, which is why
the summary counts on the review page say more "best" and "excellent" moves than the list.

The review's overview panel is structured the same way and carries what the list does not:
the full tally per side (`span.tally-row-value.analysis-<key>`, white first, then black),
the accuracy, and the game rating. A saved file may hold both panels; each parser finds
its own.

Reading the HTML is done offline from a saved file, never from a live page.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path

import chess

# Chess.com's vocabulary, from the review page. Ordinary moves have no icon in the list.
LABELS = (
    "brilliant",
    "great",
    "best",
    "excellent",
    "good",
    "book",
    "inaccuracy",
    "mistake",
    "miss",
    "blunder",
)
# The icon's SVG id is not always the label's name.
_ICON_LABELS = {"incorrect": "miss"}
_COLOUR = re.compile(r"--color-classification-([a-z]+)")


# The overview's tally keys, in its order, with the names the move list uses.
_TALLY_KEYS = {
    "brilliant": "brilliant",
    "greatFind": "great",
    "book": "book",
    "bestMove": "best",
    "excellent": "excellent",
    "good": "good",
    "inaccuracy": "inaccuracy",
    "mistake": "mistake",
    "miss": "miss",
    "blunder": "blunder",
}


@dataclass(frozen=True)
class Overview:
    """The review page's summary: per-side counts, accuracy, and game rating."""

    counts: dict[str, dict[str, int]]  # side -> label -> count
    accuracy: dict[str, float]  # side -> 0..100
    rating: dict[str, int]  # side -> chess.com's estimated performance


class _OverviewParser(HTMLParser):
    """The tallies are pairs of spans, white then black; the ratings come in rows."""

    def __init__(self) -> None:
        super().__init__()
        self.tallies: dict[str, list[int]] = {}
        self.ratings: dict[str, list[float]] = {"white": [], "black": []}
        self._tally_key: str | None = None
        self._rating_side: str | None = None
        self._rating_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {key: value or "" for key, value in attrs}
        classes = a.get("class", "").split()
        if tag == "span" and "tally-row-value" in classes:
            keys = [c[len("analysis-") :] for c in classes if c.startswith("analysis-")]
            self._tally_key = keys[0] if keys else None
        elif tag == "div" and "review-rating-component" in classes:
            self._rating_side = "white" if "review-rating-white" in classes else "black"
            self._rating_depth = 1
        elif self._rating_side is not None and tag == "div":
            self._rating_depth += 1

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if self._tally_key is not None and text:
            with contextlib.suppress(ValueError):
                self.tallies.setdefault(self._tally_key, []).append(int(text.lstrip("+")))
        if self._rating_side is not None and text:
            with contextlib.suppress(ValueError):
                self.ratings[self._rating_side].append(float(text))

    def handle_endtag(self, tag: str) -> None:
        if tag == "span" and self._tally_key is not None:
            self._tally_key = None
        elif tag == "div" and self._rating_side is not None:
            self._rating_depth -= 1
            if self._rating_depth == 0:
                self._rating_side = None


def parse_overview(html: str) -> Overview | None:
    """The summary panel, when the saved page has one."""
    parser = _OverviewParser()
    parser.feed(html)
    parser.close()
    counts: dict[str, dict[str, int]] = {"white": {}, "black": {}}
    for key, name in _TALLY_KEYS.items():
        values = parser.tallies.get(key)
        if values and len(values) >= 2:
            counts["white"][name] = values[0]
            counts["black"][name] = values[1]
    if not counts["white"]:
        return None
    accuracy: dict[str, float] = {}
    rating: dict[str, int] = {}
    for side in ("white", "black"):
        numbers = parser.ratings[side]
        if numbers:
            accuracy[side] = numbers[0]
        if len(numbers) >= 2:
            rating[side] = int(numbers[1])
    return Overview(counts=counts, accuracy=accuracy, rating=rating)


@dataclass(frozen=True)
class ReferenceMove:
    ply: int
    san: str
    label: str | None
    # Time spent on the move, in tenths of a second, when the list shows it.
    time_ds: int | None = None


class _MoveListParser(HTMLParser):
    """A stateful walk over the move list: one node per ply, in document order."""

    def __init__(self) -> None:
        super().__init__()
        self.nodes: list[tuple[str, str | None, str | None]] = []  # colour, label, san
        self.times: dict[int, int] = {}
        self._colour: str | None = None
        self._div_depth = 0
        self._label: str | None = None
        self._icon: str | None = None
        self._content_depth: int | None = None  # span nesting inside the move text
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {key: value or "" for key, value in attrs}
        classes = a.get("class", "").split()
        if tag == "div" and a.get("data-move-list-el") == "timestamp":
            with contextlib.suppress(KeyError, ValueError):
                self.times[int(a["data-ply"])] = int(a["data-time"])
        if tag == "div" and "main-line-ply" in classes:
            self._colour = "white" if "white-move" in classes else "black"
            self._div_depth = 1
            self._label = None
            self._icon = None
            self._text = []
            return
        if self._colour is None:
            return
        if tag == "div":
            self._div_depth += 1
        elif tag == "g" and self._icon is None and a.get("id"):
            self._icon = a["id"]
        elif tag == "span":
            if "node-highlight-content" in classes:
                found = _COLOUR.search(a.get("style", ""))
                if found:
                    self._label = found.group(1)
                self._content_depth = 1
            elif self._content_depth is not None:
                self._content_depth += 1
                if "icon-font-chess" in classes:
                    self._text.append(a.get("data-figurine", ""))

    def handle_data(self, data: str) -> None:
        if self._colour is not None and self._content_depth is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._colour is None:
            return
        if tag == "span" and self._content_depth is not None:
            self._content_depth -= 1
            if self._content_depth == 0:
                self._content_depth = None
        elif tag == "div":
            self._div_depth -= 1
            if self._div_depth == 0:
                label = self._label or (_ICON_LABELS.get(self._icon or "", self._icon))
                if label is not None and label not in LABELS:
                    label = None
                san = "".join(self._text).replace(" ", "").replace("\n", "")
                self.nodes.append((self._colour, label, san or None))
                self._colour = None


def parse_move_list(html: str) -> list[ReferenceMove]:
    """The moves of a saved chess.com move list, in order, with their labels and times."""
    parser = _MoveListParser()
    parser.feed(html)
    parser.close()
    moves: list[ReferenceMove] = []
    for index, (colour, label, san) in enumerate(parser.nodes, start=1):
        if san is None:
            continue
        expected = "white" if index % 2 else "black"
        if colour != expected:
            raise ValueError(f"ply {index} is a {colour} move in the list; the list is not a game")
        moves.append(
            ReferenceMove(ply=index, san=san, label=label, time_ds=parser.times.get(index))
        )
    return moves


def verify(moves: Sequence[ReferenceMove], truth: Sequence[chess.Move] | None = None) -> None:
    """Every SAN must play from the start position, and match the truth game when given."""
    board = chess.Board()
    for reference in moves:
        if truth is not None and reference.ply > len(truth):
            raise ValueError(f"ply {reference.ply}: beyond the {len(truth)} plies of the game")
        try:
            move = board.parse_san(reference.san)
        except ValueError as exc:
            raise ValueError(f"ply {reference.ply}: {reference.san!r} does not play") from exc
        if truth is not None and truth[reference.ply - 1] != move:
            raise ValueError(
                f"ply {reference.ply}: review has {reference.san}, the game has "
                f"{board.san(truth[reference.ply - 1])}"
            )
        board.push(move)


@dataclass(frozen=True)
class Reference:
    moves: list[ReferenceMove]
    overview: Overview | None = None


def save(reference: Reference, path: Path) -> None:
    data = {
        "source": "chess.com review",
        "moves": [asdict(m) for m in reference.moves],
        "overview": asdict(reference.overview) if reference.overview else None,
    }
    path.write_text(json.dumps(data, indent=1), encoding="utf-8")


_MOVE_LIST_TAG = "<wc-move-list"
_OVERVIEW_TAG = 'class="game-overview-component"'


def split_games(html: str) -> list[Reference]:
    """Every game in a saved page: each move list with the overview pasted before it.

    A page holding several reviews is the easiest thing to save by hand, so this takes the
    whole file and finds the games in it. Panels are paired in document order: an overview
    belongs to the first move list that follows it.
    """
    marks: list[tuple[int, str]] = []
    start = 0
    while (found := html.find(_MOVE_LIST_TAG, start)) != -1:
        marks.append((found, "moves"))
        start = found + len(_MOVE_LIST_TAG)
    start = 0
    while (found := html.find(_OVERVIEW_TAG, start)) != -1:
        marks.append((found, "overview"))
        start = found + len(_OVERVIEW_TAG)
    marks.sort()
    games: list[Reference] = []
    pending: Overview | None = None
    for index, (offset, kind) in enumerate(marks):
        end = marks[index + 1][0] if index + 1 < len(marks) else len(html)
        chunk = html[offset:end]
        if kind == "overview":
            # The tag sits inside the opening div; step back to include it.
            pending = parse_overview("<div " + chunk)
            continue
        moves = parse_move_list(chunk)
        if moves:
            games.append(Reference(moves=moves, overview=pending))
        pending = None
    return games


def match_game(reference: Reference, truth: Sequence[chess.Move]) -> bool:
    """Whether the listed moves are the truth game, or its opening part."""
    if not reference.moves or len(reference.moves) > len(truth):
        return False
    try:
        verify(reference.moves, truth)
    except ValueError:
        return False
    return True


def load(path: Path) -> Reference:
    """A reference from a saved page (.html: the move list and overview panels copied out
    of the developer tools, in any order) or from its parsed form (.json)."""
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        overview = data.get("overview")
        return Reference(
            moves=[ReferenceMove(**row) for row in data["moves"]],
            overview=Overview(**overview) if overview else None,
        )
    html = path.read_text(encoding="utf-8", errors="replace")
    return Reference(moves=parse_move_list(html), overview=parse_overview(html))
