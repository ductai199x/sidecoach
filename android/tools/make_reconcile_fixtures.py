"""Record what the desktop reconciler makes of a game, frame by frame -- and of every test the
desktop holds it to.

The reconciler is the hardest thing in the pipeline to get right and the easiest to get
subtly wrong: it decides when a move has settled, when a drag is only a drag, when a badge
may be ignored, when the board is showing history rather than the game. So the port is not
judged by whether it follows a game -- it is judged against the Python's own output, frame
for frame.

This renders games with the desktop's own renderer (`tests/synth.py`, the real chess.com
themes and piece sets), reads every frame with the desktop's own recogniser, writes down
the readings it produced, and runs the desktop's `Reconciler` over them, writing down every
event it emitted on every frame. `ReconcileTest` on the phone feeds the same readings to the
ported reconciler and must produce the same events in the same order.

The second half (`reconcile_ops.json`) runs the desktop's own tests of the reconciler --
`tests/test_reconcile.py`, `test_premoves.py`, the premove sweep's seeds
(`test_premove_sweep.py`), a few seeds of the stress sweep (`test_stress.py`, `RECORDED_SEEDS`),
and the pipeline tests whose games go through one (`RECORDED`) --
as they stand, with every reconciler they build swapped for one that writes
down what is done to it: built with which tunables, each frame of readings it is given and the
events it answers, every call from outside (`start_from`, `turn_from_clock`, `end`,
`board_unseen`, `commit_pending_terminal`, `reorient`, and the tests' own `_commit` and
`_observe`), every field a test sets on it (`rec.board = ...`, `rec._base_fen`,
`rec.stale_after = 2`), and after each of those what it then says of itself (the position, the
record, whether it is browsing, whose turn is open, whether a story is pending, ...). The
tests' own assertions run as well, so a fixture is only written from tests that pass.
`ReconcileOpsTest` on the phone does the same things to the ported reconciler and must answer
the same, op for op (the parity audit's st-T01, st-T02). The cases the desktop's test audit
folded into other tests are still played, as they stood and under the names they had
(`KEPT` here, `FOLDED` in `reconcile_folded_cases.py`), so the fixture loses none.

Run from the desktop repo, whose environment has python-chess and the test helpers:

    cd D:/chess-coaching-companion
    uv run python android/tools/make_reconcile_fixtures.py
"""

from __future__ import annotations

import functools
import importlib
import inspect
import itertools
import json
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import chess

DESKTOP = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(DESKTOP / "src"))

from sidecoach.events import (  # noqa: E402
    GameEnded,
    Region,
    GameStarted,
    MovePlayed,
    PositionChanged,
    Resync,
    Unexplained,
    Side,
    Takeback,
    TurnPending,
)
from sidecoach.state.reconcile import Reconciler  # noqa: E402
from sidecoach.vision.pieces import ThemeModel, orientation_from_start  # noqa: E402
from sidecoach.vision.squares import BoardGeometry  # noqa: E402
from tests import synth  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "app" / "src" / "androidTest" / "assets" / "chess"

# The phone's half-resolution capture: the size `make_fixtures.py` renders at.
WIDTH, HEIGHT = 672, 1496
CELL = 83
X0, Y0 = 4, 300
FPS = 10.0


def reading_json(reading) -> dict:
    """The fields the reconciler reads. Nothing else crosses over."""
    return {
        "kind": reading.kind,
        "symbol": reading.symbol,
        "colour": reading.colour.value if reading.colour is not None else None,
        "settled": reading.settled,
        "hover": reading.hover,
        "covered": reading.covered,
        "tint": reading.tint,
        "drawn": reading.drawn,
        "highlighted": reading.highlighted,
        "guess": reading.guess,
    }


def event_json(event) -> str:
    """One event as a line to compare. The game id is random, so it is left out."""
    if isinstance(event, GameStarted):
        return f"GameStarted {event.kind} {event.user_side.value} {event.start_fen}"
    if isinstance(event, PositionChanged):
        return f"PositionChanged {event.ply} {event.fen}"
    if isinstance(event, MovePlayed):
        return (
            f"MovePlayed {event.ply} {event.san} {event.uci} "
            f"{'user' if event.by_user else 'them'} {event.think_ms}"
        )
    if isinstance(event, Takeback):
        return f"Takeback {event.plies} {event.fen}"
    if isinstance(event, Resync):
        return f"Resync {event.reason} | {event.fen}"
    if isinstance(event, Unexplained):
        return f"Unexplained {event.reason} | {event.fen}" + (" | told" if event.told else "")
    if isinstance(event, GameEnded):
        return f"GameEnded {event.result} {event.reason} | {event.pgn} | {event.final_fen}"
    if isinstance(event, TurnPending):
        return f"TurnPending {event.pending}"
    return type(event).__name__


def board_kwargs(**extra) -> dict:
    return dict(
        width=WIDTH, height=HEIGHT, x0=X0, y0=Y0, cell=CELL, coordinates=True, **extra
    )


def plain_game(sans: tuple[str, ...], theme: str, style: str, per_state: int = 3) -> list[dict]:
    """The start position, then every move, each held for a few frames."""
    board = chess.Board()
    states: list[dict] = [board_kwargs(board=board.copy(), theme=theme, style=style)]
    for san in sans:
        move = board.push_san(san)
        states.append(
            board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move)
        )
    return [dict(state, _repeat=per_state) for state in states]


def dragged_game(sans: tuple[str, ...], theme: str, style: str) -> list[dict]:
    """Every move played by hand: the piece lifted, carried, then landed."""
    board = chess.Board()
    states: list[dict] = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for san in sans:
        move = board.parse_san(san)
        for progress in (0.25, 0.7):
            states.append(
                dict(
                    board_kwargs(
                        board=board.copy(),
                        theme=theme,
                        style=style,
                        dragging=(move.from_square, move.to_square, progress, progress),
                    ),
                    _repeat=1,
                )
            )
        board.push(move)
        states.append(
            dict(
                board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move),
                _repeat=3,
            )
        )
    return states


def animated_game(sans: tuple[str, ...], theme: str, style: str) -> list[dict]:
    """Every move animated across the board, as the opponent's moves arrive."""
    board = chess.Board()
    states: list[dict] = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for san in sans:
        move = board.parse_san(san)
        states.append(
            dict(
                board_kwargs(
                    board=board.copy(),
                    theme=theme,
                    style=style,
                    animating=(move.from_square, move.to_square, 0.5),
                ),
                _repeat=1,
            )
        )
        board.push(move)
        states.append(
            dict(
                board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move),
                _repeat=3,
            )
        )
    return states


def decorated_game(sans: tuple[str, ...], theme: str, style: str) -> list[dict]:
    """A game with the user's arrows and right-click marks over it."""
    board = chess.Board()
    states: list[dict] = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for i, san in enumerate(sans):
        move = board.push_san(san)
        extra: dict = {"highlight": move}
        if i % 3 == 1:
            extra["arrows"] = (chess.Move(chess.D2, chess.D4),)
        if i % 3 == 2:
            extra["red_marks"] = (chess.E5, chess.F6)
        states.append(
            dict(board_kwargs(board=board.copy(), theme=theme, style=style, **extra), _repeat=3)
        )
    return states


def mating_game(theme: str, style: str) -> list[dict]:
    """The scholar's mate, then the site's badges and the dialog over the board."""
    sans = ("e4", "e5", "Qh5", "Nc6", "Bc4", "Nf6", "Qxf7")
    board = chess.Board()
    states: list[dict] = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for san in sans:
        move = board.push_san(san)
        states.append(
            dict(board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move), _repeat=3)
        )
    for alpha in (0.3, 0.6, 1.0, 1.0, 1.0):
        states.append(
            dict(
                board_kwargs(
                    board=board.copy(), theme=theme, style=style, result="1-0", dialog=alpha
                ),
                _repeat=2,
            )
        )
    return states


def browsing_game(sans: tuple[str, ...], theme: str, style: str) -> list[dict]:
    """The game, then the user steps back through the move list and plays on."""
    board = chess.Board()
    states: list[dict] = [dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=3)]
    for san in sans:
        move = board.push_san(san)
        states.append(
            dict(board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move), _repeat=3)
        )
    # Back two plies, held, then a different move from there: a takeback.
    board.pop()
    board.pop()
    states.append(dict(board_kwargs(board=board.copy(), theme=theme, style=style), _repeat=4))
    move = board.push_san("d4")
    states.append(
        dict(board_kwargs(board=board.copy(), theme=theme, style=style, highlight=move), _repeat=4)
    )
    return states


SCENARIOS = {
    "italian-green-classic": lambda: plain_game(
        ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5", "c3", "Nf6", "d4", "exd4"), "green", "classic"
    ),
    "sicilian-brown-wooden": lambda: plain_game(
        ("e4", "c5", "Nf3", "d6", "d4", "cxd4", "Nxd4", "Nf6", "Nc3", "a6"), "brown", "wooden"
    ),
    "castling-green-classic": lambda: plain_game(
        ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7", "Re1", "O-O"),
        "green",
        "classic",
    ),
    "dragged-green-classic": lambda: dragged_game(
        ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"), "green", "classic"
    ),
    "animated-brown-classic": lambda: animated_game(
        ("d4", "d5", "c4", "e6", "Nc3", "Nf6"), "brown", "classic"
    ),
    "decorated-green-classic": lambda: decorated_game(
        ("e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6"), "green", "classic"
    ),
    "mate-green-classic": lambda: mating_game("green", "classic"),
    "browsing-green-classic": lambda: browsing_game(
        ("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5"), "green", "classic"
    ),
}


def run(name: str, states: list[dict]) -> dict:
    geometry = BoardGeometry(region=Region(X0, Y0, CELL * 8, CELL * 8), cell_px=float(CELL))

    frames: list[dict] = []
    model: ThemeModel | None = None
    reconciler: Reconciler | None = None
    bottom = Side.WHITE
    frame_id = 0
    for state in states:
        repeat = state.pop("_repeat", 1)
        image = synth.synthetic_board(**state)
        for _ in range(repeat):
            ts = frame_id / FPS
            if model is None:
                bottom = orientation_from_start(image, geometry)
                model = ThemeModel.from_start_position(image, geometry, bottom)
                if model is None:
                    raise SystemExit(f"{name}: the theme would not bootstrap")
                reconciler = Reconciler(bottom=bottom, user_side=bottom)
            readings = model.classify_frame(image, geometry)
            events = reconciler.update(readings, ts)
            frames.append(
                {
                    "ts": ts,
                    "readings": [reading_json(readings[r][c]) for r in range(8) for c in range(8)],
                    "events": [event_json(event) for event in events],
                }
            )
            frame_id += 1
    return {
        "name": name,
        "bottom": bottom.value,
        "frames": frames,
        "pgn": reconciler.pgn() if reconciler is not None else "",
        "final_fen": reconciler.board.fen() if reconciler is not None else "",
    }


# ---- The desktop's own tests, recorded -------------------------------------------------

# The desktop tests whose games go through a reconciler, by module: None for every test in
# the module (a test that builds no reconciler records nothing and is left out).
RECORDED: dict[str, list[str] | None] = {
    "tests.test_reconcile": None,
    "tests.test_premoves": None,
    # The premove sweep's own seeds (its SEEDS): chess.com's queues simulated.
    "tests.test_premove_sweep": None,
    "tests.test_pipeline": None,
    "tests.test_scenarios": None,
    "tests.test_screen_state": None,
    "tests.test_fast_puzzles": None,
    "tests.test_occlusion": [
        "test_mating_move_commits_while_the_game_over_dialog_covers_the_board",
        "test_mate_in_the_middle_commits_from_two_uncovered_frames",
        "test_move_out_from_under_a_cover_waits_for_the_memory_to_expire",
        "test_fading_dialog_reads_as_covered_and_keeps_the_memory_clean",
        "test_a_dialog_that_stays_over_a_live_board_covers_the_game_but_ends_nothing",
        "test_the_dialog_confirms_a_mate_read_once",
    ],
    "tests.test_result_badges": [
        "test_a_resignation_ends_the_game_with_the_sites_result",
        "test_a_draw_badge_is_not_read_until_one_is_on_record",
        "test_a_flag_is_a_loss_on_time",
        "test_one_frame_of_badges_is_not_enough",
        "test_a_mate_on_the_board_ends_the_game_once_whatever_the_badges_say",
        "test_the_badges_commit_a_premoved_mate_still_settling",
    ],
    "tests.test_cold_start": [
        "test_cold_start_reads_a_rendered_mid_game_board",
        "test_asked_to_reread_a_board_with_a_piece_left_selected",
        "test_cold_start_without_coordinates_takes_the_side_it_is_told",
        "test_cold_start_reads_the_real_frames_the_same",
        "test_a_game_joined_from_the_middle_is_adopted_without_being_asked",
        "test_the_side_to_move_of_an_adopted_position_comes_from_the_running_clock",
        "test_a_board_mid_move_is_never_adopted_and_a_start_position_bootstraps",
        "test_flip_re_reads_a_joined_position_the_other_way_up",
        "test_a_resync_onto_a_board_the_other_way_up_reads_the_labels",
        "test_two_piece_sets_on_one_board_are_both_kept_and_each_read",
        "test_a_selected_piece_is_not_explained",
        "test_a_board_without_coordinates_is_explained_as_its_orientation",
        "test_labels_cached_wrong_are_read_past_by_the_8_and_learned_again",
        "test_a_flip_teaches_labels_that_read_no_side_where_the_8_cannot_be_read",
        "test_a_flip_by_mistake_teaches_no_labels",
        "test_the_next_puzzle_from_the_other_side_is_turned_round_at_once",
        "test_a_new_position_on_a_slow_stream_is_taken_after_a_moment_not_ten_frames",
        "test_a_puzzle_read_cold_keeps_the_opening_move_played_before_it_was_adopted",
    ],
    "tests.test_slides": ["test_a_move_seen_sliding_across_squares_is_one_move"],
    # A few seeds of the stress sweep (`RECORDED_SEEDS`): the reconciler's rules no other
    # recorded test runs (the parity audit, 2026-09-29).
    "tests.test_stress": [
        "test_a_random_game_is_followed_to_its_position",
        "test_a_random_game_ends_on_screen",
    ],
}

# The seeds a seeded test is recorded with, in place of its module's SEEDS, where they are not
# those. Chosen by the lines of `reconcile.py` they run that no other recorded test does
# (measured with the line monitor, 2026-09-29): the followed game of stress 73 has a move refused
# through a square an arrow is drawn over (`_complete`'s `self._drawn`), stress 1's the
# opponent's part of a story not yet clean during a drag; the ending of stress 1 has the mate
# land on a king under its own decoration (`_king_lands`, `_checked_kings`), stress 8's a piece
# of the user's on a red square that is really there; premove 47 and 90 have a piece of the
# user's on a red square that is a queue's ghost, never a move played (90 is the seed that
# found it).
RECORDED_SEEDS: dict[tuple[str, str], tuple[int, ...]] = {
    ("tests.test_stress", "test_a_random_game_is_followed_to_its_position"): (1, 73),
    ("tests.test_stress", "test_a_random_game_ends_on_screen"): (1, 8),
    ("tests.test_premove_sweep", "test_premove_queues_are_followed"): (*range(1, 9), 47, 90),
}


# Desktop tests since folded into others there (the test audit of 2026-09-28: another test
# pins the same behaviour), whose games are still recorded here under the names they had --
# their frames as they showed them, and their assertions -- so the fixture keeps every case.
def _a_draw_badge_is_not_read_until_one_is_on_record() -> None:
    from sidecoach.state.pipeline import BADGE_FRAMES
    from tests.games import Game

    g = Game(pipeline=True)
    g.play("e4", "e5", "Nf3", "Nc6")
    new = g.show(BADGE_FRAMES + 2, highlight=chess.Move.from_uci("b8c6"), result="1/2-1/2")
    assert not any(isinstance(e, GameEnded) for e in new)


def _a_flag_is_a_loss_on_time() -> None:
    from sidecoach.state.pipeline import BADGE_FRAMES
    from tests.games import Game

    g = Game(pipeline=True)
    g.play("e4", "e5")
    g.pipe._last_clock = (0, 240000, None)  # white's clock at zero
    new = g.show(BADGE_FRAMES, highlight=chess.Move.from_uci("e7e5"), result="0-1")
    ended = [e for e in new if isinstance(e, GameEnded)]
    assert [(e.result, e.reason) for e in ended] == [("0-1", "lost on time")]


def _a_transition_emits_once_not_every_frame() -> None:
    from sidecoach.events import ScreenState, ScreenStateChanged
    from sidecoach.state.pipeline import GamePipeline

    pipe = GamePipeline()
    board = chess.Board()
    ts = 0.0
    events = []
    for _ in range(6):
        events += pipe.process(synth.synthetic_board(board=board), ts)
        ts += 0.1
    states = [e.state for e in events if isinstance(e, ScreenStateChanged)]
    assert states.count(ScreenState.LIVE_GAME) == 1


KEPT: dict[str, dict[str, Callable[[], None]]] = {
    "tests.test_result_badges": {
        "test_a_draw_badge_is_not_read_until_one_is_on_record": (
            _a_draw_badge_is_not_read_until_one_is_on_record
        ),
        "test_a_flag_is_a_loss_on_time": _a_flag_is_a_loss_on_time,
    },
    "tests.test_screen_state": {
        "test_a_transition_emits_once_not_every_frame": _a_transition_emits_once_not_every_frame,
    },
}

# The reconciler's tunables: dataclass fields a test may build it with or set.
TUNABLES = (
    "settle_frames",
    "max_plies",
    "search_budget",
    "stale_after",
    "resync_cells",
    "resync_after",
    "resync_hold_frames",
    "resync_min_frames",
    "resync_hold_s",
    "resync_jump_cells",
    "max_stale_subset_cells",
    "max_queue_steps",
    "resync_max_soft",
    "memory_frames",
    "memory_gap_ticks",
    "min_readable_cells",
)
# What a test (or the pipeline) sets on a reconciler from outside; anything else stops the run.
SETTABLE = frozenset(TUNABLES) | {
    "board", "_base_fen", "_started", "_finished", "bottom", "user_side", "unexplained_note_frames",
    "_mark_denied",
}

KINDS = {"empty": "e", "piece": "p", "unknown": "u"}
TINTS = {None: "-", "yellow": "y", "red": "r", "other": "o"}


def cell_code(reading) -> str:
    """One reading in six characters: kind, symbol, colour, the flags (settled 1, hover 2,
    covered 4, drawn 8, highlighted 16, from 'A'), the tint and the guess. The fields the
    reconciler reads, as `reading_json` has them; nothing else crosses over."""
    flags = (
        (1 if reading.settled else 0)
        | (2 if reading.hover else 0)
        | (4 if reading.covered else 0)
        | (8 if reading.drawn else 0)
        | (16 if reading.highlighted else 0)
    )
    colour = "-" if reading.colour is None else reading.colour.value[0]
    if reading.tint not in TINTS:
        raise SystemExit(f"a tint the fixture cannot write: {reading.tint!r}")
    for one in (reading.symbol, reading.guess):
        if one is not None and len(one) != 1:
            raise SystemExit(f"a symbol the fixture cannot write: {one!r}")
    return (
        KINDS[reading.kind]
        + (reading.symbol or "-")
        + colour
        + chr(ord("A") + flags)
        + TINTS[reading.tint]
        + (reading.guess or "-")
    )


class _Log:
    """What was done to every reconciler of the test running now, and what it answered."""

    def __init__(self) -> None:
        self.table: list[str] = []
        self.index: dict[str, int] = {}
        self.runs: list[dict] = []
        self.ops: list[dict] | None = None
        self.name = ""
        self.ids: dict[int, int] = {}
        # Every reconciler recorded, held for the test: an `id` is an address, and one freed
        # (a pipeline's reset drops its reconciler) is handed to the next, which was then
        # recorded as the old one built again.
        self.held: list[object] = []
        self.games: dict[str, int] = {}
        self.states: dict[int, dict] = {}

    def begin(self, name: str) -> None:
        self.name = name
        self.ops = []
        self.ids = {}
        self.held = []
        self.games = {}
        self.states = {}

    def end(self) -> int:
        ops = self.ops or []
        if ops:
            self.runs.append({"name": self.name, "ops": ops})
        self.ops = None
        return len(ops)

    def frame(self, readings) -> int:
        code = "".join(cell_code(readings[r][c]) for r in range(8) for c in range(8))
        if code not in self.index:
            self.index[code] = len(self.table)
            self.table.append(code)
        return self.index[code]

    def state(self, rec) -> dict:
        """What the reconciler says of itself: the fields the desktop's tests look at."""
        if rec.game_id not in self.games:
            self.games[rec.game_id] = len(self.games)
        return {
            "fen": rec.board.fen(),
            "base": rec.base_fen,
            "pgn": rec.pgn(),
            "viewing": rec.viewing,
            "turn_unknown": rec.turn_unknown,
            "started": rec.started,
            "finished": rec.finished,
            "result": rec.result,
            "kind": rec.kind,
            "pending": rec.pending_story,
            "blocked": rec.blocked,
            "unexplained": rec.unexplained_frames,
            "agrees": rec.agrees,
            "turn_pending": rec.turn_pending,
            "game": self.games[rec.game_id],
            "last_seen": sorted(rec._last_seen),
        }

    def record(self, rec, op: str, **fields) -> None:
        if self.ops is None:
            return  # built outside a recorded test
        key = id(rec)
        if key not in self.ids:
            self.ids[key] = len(self.ids)
            self.held.append(rec)
        entry = {"r": self.ids[key], "op": op, **fields}
        # The state as it changed since the last op on this reconciler.
        now = self.state(rec)
        before = self.states.get(key, {})
        entry["state"] = {k: v for k, v in now.items() if before.get(k, object()) != v}
        self.states[key] = now
        self.ops.append(entry)


LOG = _Log()


def _board_json(board: chess.Board) -> dict:
    return {"fen": board.root().fen(), "moves": [m.uci() for m in board.move_stack]}


def _value_json(name: str, value):
    if name == "board":
        return _board_json(value)
    if isinstance(value, Side):
        return value.value
    return value


class _Recording(Reconciler):
    """The desktop's reconciler, writing down what is done to it from outside."""

    def __init__(self, *args, **kwargs) -> None:
        object.__setattr__(self, "_rec_depth", 1)
        unknown = set(kwargs) - {"bottom", "user_side", *TUNABLES}
        if unknown or len(args) > 2:
            raise SystemExit(f"{LOG.name}: a reconciler built with {sorted(unknown)} {args}")
        super().__init__(*args, **kwargs)
        object.__setattr__(self, "_rec_depth", 0)
        made = {name: getattr(self, name) for name in TUNABLES}
        made["bottom"] = self.bottom.value
        made["user_side"] = self.user_side.value
        LOG.record(self, "new", args=made)

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        if getattr(self, "_rec_depth", 1) == 0:
            if name not in SETTABLE:
                raise SystemExit(f"{LOG.name}: a test set reconciler.{name}, which no fixture says")
            LOG.record(self, "set", name=name, value=_value_json(name, value))


def _events(out) -> dict:
    return {"events": [event_json(e) for e in out]}


def _nothing(out) -> dict:
    return {}


# The calls written down: what each takes, as the fixture has it, and what it answers.
CALLS = {
    "update": (lambda a: {"readings": LOG.frame(a["readings"]), "ts": a["ts"]}, _events),
    "board_unseen": (lambda a: {}, _nothing),
    "start": (lambda a: {}, _events),
    "start_from": (
        lambda a: {
            "fen": a["placement"].fen(),
            "ts": a["ts"],
            "highlighted": sorted(a["highlighted"]),
            "before": None
            if a["before"] is None
            else {"fen": a["before"][0].fen(), "highlighted": sorted(a["before"][1])},
        },
        _events,
    ),
    "turn_from_clock": (lambda a: {"colour": a["colour"]}, _events),
    "end": (lambda a: {"result": a["result"], "reason": a["reason"]}, _events),
    "commit_pending_terminal": (lambda a: {"result": a["result"]}, _events),
    "reorient": (lambda a: {"bottom": a["bottom"].value}, _events),
    "_commit": (lambda a: {"uci": a["move"].uci()}, _events),
    "_observe": (
        lambda a: {"readings": LOG.frame(a["readings"])},
        lambda out: {"seen": sorted(out)},
    ),
    # On the frame the test observed last (`test_the_mark_adopts_only_a_board_held_still`).
    "_marked_resync": (lambda a: {}, _events),
}


def _wrap(name: str) -> None:
    base = getattr(Reconciler, name)
    signature = inspect.signature(base)
    takes, gives = CALLS[name]

    @functools.wraps(base)
    def method(self, *args, **kwargs):  # noqa: ANN001, ANN202
        if self._rec_depth:
            return base(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        fields = takes(bound.arguments)
        object.__setattr__(self, "_rec_depth", 1)
        try:
            out = base(self, *args, **kwargs)
        finally:
            object.__setattr__(self, "_rec_depth", 0)
        LOG.record(self, name.lstrip("_"), **fields, **gives(out))
        return out

    setattr(_Recording, name, method)


for _name in CALLS:
    _wrap(_name)


def _cases(func, module=None, seeds=None) -> list[tuple[str, dict]]:  # noqa: ANN001
    """The parameter sets pytest would run `func` with: every `parametrize` crossed, and a
    test taking `seed` once for each of its module's `SEEDS`, as the desktop's conftest
    parametrises it without `--seeds` -- or for each of `seeds`, when given."""
    grids: list[list[tuple[str, dict]]] = []
    if "seed" in inspect.signature(func).parameters:
        chosen = module.SEEDS if seeds is None else seeds
        grids.append([(str(seed), {"seed": seed}) for seed in chosen])
    for mark in reversed(getattr(func, "pytestmark", [])):
        if mark.name != "parametrize":
            continue
        names, values = mark.args[0], mark.args[1]
        names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
        rows = []
        for row, value in enumerate(values):
            value = getattr(value, "values", value)  # a pytest.param
            one = dict(zip(names, value if len(names) > 1 else (value,), strict=True))
            short = [
                str(getattr(v, "value", v))
                for v in one.values()
                if isinstance(v, Side | bool | int | float) or (isinstance(v, str) and len(v) <= 10)
            ]
            label = "-".join(short) if len(short) == len(one) else f"#{row}"
            rows.append((label, one))
        grids.append(rows)
    cases = []
    for combo in itertools.product(*grids):
        label = "-".join(part for part, _ in combo)
        params: dict = {}
        for _, one in combo:
            params.update(one)
        cases.append((label, params))
    return cases or [("", {})]


def _record_case(name: str, func: Callable[..., None], params: dict) -> None:
    """One test case run with its arguments and pytest's `tmp_path` and `monkeypatch`, what
    its reconcilers were told written down under `name`."""
    import pytest

    with tempfile.TemporaryDirectory() as tmp:
        patch = pytest.MonkeyPatch()
        kwargs = dict(params)
        wants = inspect.signature(func).parameters
        if "tmp_path" in wants:
            kwargs["tmp_path"] = Path(tmp)
        if "monkeypatch" in wants:
            kwargs["monkeypatch"] = patch
        LOG.begin(name)
        try:
            func(**kwargs)
        finally:
            patch.undo()
            ops = LOG.end()
    print(f"  {name}: {ops} ops")


def record_tests() -> dict:
    """Every recorded desktop test run with the recording reconciler in place, and after
    each module's own tests the cases of it the state audit folded (`reconcile_folded_cases`:
    played as they stood, under the names they had)."""
    from reconcile_folded_cases import FOLDED

    from sidecoach.state import pipeline as pipeline_module
    from tests import games

    # The pipeline's reconcilers, and the ones the tests' shared helpers build
    # (tests/games.py: `Game`, `reconciler_at`).
    pipeline_module.Reconciler = _Recording
    games.Reconciler = _Recording
    for module_name, wanted in RECORDED.items():
        module = importlib.import_module(module_name)
        if hasattr(module, "Reconciler"):
            module.Reconciler = _Recording
        if hasattr(module, "_opened") and hasattr(module, "_overlay_opening"):
            # The overlay cases share one opening by copying its pipeline, and a copied
            # reconciler was never built where a fixture could see it: each case reads
            # its own opening here, as the copy is proven to (test_pipeline).
            module._opened = lambda style, theme, m=module: m._overlay_opening(style, theme)
        short = module_name.split(".")[-1]
        kept = KEPT.get(module_name, {})
        names = wanted or sorted({n for n in dir(module) if n.startswith("test_")} | set(kept))
        for test_name in names:
            func = kept.get(test_name) or getattr(module, test_name)
            if any(mark.name == "xfail" for mark in getattr(func, "pytestmark", [])):
                # A known limit, failing by design: its assertions say what it cannot do.
                print(f"  {short}::{test_name}: xfail, not recorded")
                continue
            seeds = RECORDED_SEEDS.get((module_name, test_name))
            for label, params in _cases(func, module, seeds):
                _record_case(f"{short}::{test_name}" + (f"[{label}]" if label else ""), func, params)
        for test_name, func, params in FOLDED.get(module_name, []):
            _record_case(f"{short}::{test_name}", func, params)
    return {"readings": LOG.table, "runs": LOG.runs}


def main() -> None:
    runs = []
    for name, build in SCENARIOS.items():
        run_data = run(name, build())
        moves = sum(
            1
            for frame in run_data["frames"]
            for line in frame["events"]
            if line.startswith("MovePlayed")
        )
        print(f"{name}: {len(run_data['frames'])} frames, {moves} moves — {run_data['pgn']}")
        runs.append(run_data)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "reconcile.json").write_text(json.dumps({"runs": runs}), encoding="utf-8")
    print(f"-> {OUT / 'reconcile.json'}")

    print("the desktop's tests, recorded:")
    ops = record_tests()
    (OUT / "reconcile_ops.json").write_text(json.dumps(ops), encoding="utf-8")
    count = sum(len(r["ops"]) for r in ops["runs"])
    print(
        f"-> {OUT / 'reconcile_ops.json'}: {len(ops['runs'])} tests, {count} ops, "
        f"{len(ops['readings'])} distinct frames"
    )


if __name__ == "__main__":
    main()
