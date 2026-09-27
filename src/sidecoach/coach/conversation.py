"""The game as the harness the model reads: one user message, in sections.

`# Setting` says whom the coach coaches. `# Game so far` is the record: one line per move
with the engine's verdict on it -- the label, the cost, the move the engine preferred and
where its line leads -- and under each move, indented, what was said after it (the
player's words, the coach's own replies) and any event (a takeback, a re-read of the board,
a dialog, the end of the game), the way a game record carries its annotations. Once the
model has seen a move's line it is frozen and never edited; facts that arrive later go in
a note of their own. `# Position now` is the last ply in full at the disclosure level in
force for the position the player faces: the board, the last move, the standing, the
engine's lines, the facts in the third person per side (below MOVE on the player's turn,
those past the rung set apart, for the model to teach from), the disclosure line -- the
model sees everything and the reply is checked for the answer (`disclosure.leak_check`) --
the complete inventory of what stands on the board,
the player's record so far, and what was already said here. The coach appends its task and
answers as the assistant.

Two levers keep a long game's harness small. The dialogue older than the last N plies can
be folded by a summariser into one "Earlier dialogue, summarised" line (`foldable`,
`fold`; the coach runs it in the background). With `compact_record`, the coach's routine
nods after sound moves are written out only while recent, and sound moves with nothing
said after them run together as a move list.

The `Coach` owns one of these per game and asks it for the messages of every request.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import chess

from sidecoach.analysis.engine import san_line
from sidecoach.coach.disclosure import PositionFacts, visible_facts
from sidecoach.coach.effects import effects_text, line_notes
from sidecoach.coach.facts import material_text, motif_words, position_text, standing_words
from sidecoach.coach.inventory import inventory_lines
from sidecoach.coach.outcome import line_outcome
from sidecoach.coach.person import third_person
from sidecoach.events import (
    Analysis,
    DisclosureLevel,
    GameEnded,
    GameStarted,
    Line,
    MoveAssessment,
    MovePlayed,
    Resync,
    Severity,
    Side,
    Takeback,
)
from sidecoach.llm.base import ChatMessage

# What the first hint may say, as the disclosure line and the directive state it: the key
# fact is named on the board (Tai, 2026-09-24), so "no square" -- the rule before the first
# hint was a lesson -- told the model to say less than the template.
IDEA_LEVEL_WORDS = (
    "the idea: what is at stake and the key fact on the board -- the pieces, squares and "
    "their moves the facts name, the target's square too, even where the move goes -- with "
    "the principle behind it; never the player's move itself, or which of their own pieces "
    "makes it"
)
# Below the move level the model sees everything -- the engine lines, the legal moves, the
# facts past the rung -- and says only what the rung allows; the reply is checked for the
# answer before it is spoken (disclosure.leak_check).
BEYOND_TITLE = (
    "Past this level (for you to teach from; do not say the move, its square or the piece "
    "that makes it)"
)
SEE_ALL_WORDS = (
    "You see everything, the answer included, so the hint can be precise and true: say what "
    "this level allows, built from what you know."
)

# The engine lines the harness shows: the best three, then three weaker ones, each to
# this many plies.
BEST_LINES = 3
WEAKER_LINES = 3
LINE_PLIES = 8
# With the compact record, a routine nod is written out for this many plies, so the coach
# still sees its own last words and varies them; older ones go.
NOD_PLIES = 6

# A record line whose verdict is a bare label: "12. Bxf6 (the player): best." Such a move
# joins the compact record's move list; one with a cost, a preferred move or a note stands.
_BARE_VERDICT = re.compile(
    r"^(\d+\.(?:\.\.)? \S+) \((?:the player|the opponent)\): "
    r"(?:best|excellent|great|brilliant|good|book)\.$"
)

# The motifs worth naming on a move. The tagger also emits bookkeeping tags (quietMove,
# defensiveMove, capture, check) and geometric ones that hold on most quiet moves
# (interference, clearance); named in the harness they had the coach narrating "a6 cuts the
# h5/f3/d1 line" on book moves.
MOVE_MOTIFS = frozenset(
    {
        "fork",
        "pin",
        "absolutePin",
        "skewer",
        "discoveredAttack",
        "discoveredCheck",
        "doubleCheck",
        "sacrifice",
        "deflection",
        "attraction",
        "capturingDefender",
        "trappedPiece",
        "hangingPiece",
        "xRayAttack",
        "backRankMate",
        "smotheredMate",
        "mate",
        "mateIn1",
        "mateIn2",
        "mateIn3",
        "promotion",
        "underPromotion",
        "zugzwang",
        "exposedKing",
        "attackingF2F7",
        "advancedPawn",
    }
)
LABEL_ORDER = (
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


@dataclass
class PlyRecord:
    """One move as it was seen, with the engine's word on it as it arrives."""

    move: MovePlayed
    assessment: MoveAssessment | None = None
    # The analysis of the position the move led to (the side to move's lines), and the
    # facts built from it for the user's side.
    analysis: Analysis | None = None
    facts: PositionFacts | None = None
    white_ms: int | None = None
    black_ms: int | None = None
    withdrawn: bool = False  # taken back: out of the move list, still in history
    superseded: bool = False  # before a re-read of the board: the move list restarted


_PLACED_PIECE = re.compile(r"\b(king|queen|rook|bishop|knight|pawn) on ([a-h][1-8])\b")
_PIECE_TYPES = {
    "king": chess.KING,
    "queen": chess.QUEEN,
    "rook": chess.ROOK,
    "bishop": chess.BISHOP,
    "knight": chess.KNIGHT,
    "pawn": chess.PAWN,
}


def _gone(text: str, board: chess.Board) -> bool:
    """Whether `text` names a piece on a square that no longer holds that piece on `board`."""
    for word, square in _PLACED_PIECE.findall(text.lower()):
        if board.piece_type_at(chess.parse_square(square)) != _PIECE_TYPES[word]:
            return True
    return False


def _also_stopped(facts: PositionFacts, played: chess.Board | None) -> str | None:
    """The threat the better move's reason says it stops, when the move played stopped it
    too: on the board right after it, the threat can no longer be played."""
    if played is None or not facts.threats or not facts.why:
        return None
    threat = str(facts.threats[0])
    if f"stops {threat}" not in facts.why:
        return None
    try:
        played.parse_san(threat)
    except ValueError:
        return threat
    return None


@dataclass
class Entry:
    """One message of the conversation. A ply entry renders itself from its record until it
    is frozen; every other kind is plain text."""

    role: str
    kind: str  # "game", "ply", "takeback", "resync", "covered", "ended", "ask", "coach", "note"
    text: str = ""
    record: PlyRecord | None = None
    frozen: str | None = None
    folded: bool = False  # dialogue folded into the summary: no longer written out
    # A routine word after a sound move (the POST_MOVE nod): the verdict is on the move's
    # line, so the nod is written out only while recent, and never summarised.
    nod: bool = False


@dataclass
class Conversation:
    user_side: Side = Side.WHITE
    level_text: str = "around 1200"
    entries: list[Entry] = field(default_factory=list)
    records: list[PlyRecord] = field(default_factory=list)
    start_fen: str = chess.STARTING_FEN
    # The disclosure level to render the position the user faces with, set by the coach
    # before each request from the current position's disclosure state.
    level: DisclosureLevel = DisclosureLevel.IDEA
    # The earlier dialogue, summarised (see `summarise`); empty until the first fold.
    summary: str = ""
    # The record compacted: the coach's routine nods after sound moves are written out
    # only for the last NOD_PLIES plies (the verdict is on the move's line), and sound
    # moves with nothing said after them run together as a move list.
    compact_record: bool = False
    _drained: int = 0

    # ---- the game's events, in the order they happen -----------------------------------

    def new_game(
        self,
        event: GameStarted,
        *,
        opponent: str = "a bot",
        control: str = "",
        profile: Sequence[str] = (),
    ) -> None:
        """The setting of a new game. `profile` is what the player model says about this
        player from earlier games -- the biggest cost, the last session -- in sentences the
        profile stage computed, so the coach has continuity without inventing any."""
        self.user_side = event.user_side
        self.start_fen = event.start_fen
        self.entries = []
        self.records = []
        self._drained = 0
        side = "White" if event.user_side == Side.WHITE else "Black"
        text = f"You coach {side}."
        if control:
            text += f" Time control: {control}."
        if event.kind == "puzzle":
            text += (
                f"\nThis is a puzzle the player is solving on the panel's board, {side} to "
                "move and to find the best line. Say nothing unasked; when they ask for a "
                "hint, give one at the level allowed -- the move itself only at the third hint "
                "or when asked for it:\n" + position_text(chess.Board(event.start_fen))
            )
        elif event.kind != "game":
            text += (
                "\nThis is a position, not a game from the start: a puzzle, an analysis "
                "board, or a game joined from the middle. There is no earlier history to "
                "speak of. The player is the side at the bottom and, on a puzzle, to move; "
                "coach the position as it stands:\n" + position_text(chess.Board(event.start_fen))
            )
        elif event.start_fen != chess.STARTING_FEN:
            text += "\nThe game is joined in progress; the position is:\n" + position_text(
                chess.Board(event.start_fen)
            )
        if profile:
            text += (
                "\nWhat earlier games say about this player (computed, not opinion): "
                + " ".join(profile)
            )
        self.entries.append(Entry("user", "game", text))

    def move(self, event: MovePlayed) -> PlyRecord:
        record = PlyRecord(move=event)
        self.records.append(record)
        self.entries.append(Entry("user", "ply", record=record))
        return record

    def assess(self, event: MoveAssessment) -> PlyRecord | None:
        record = self._record_at(event.ply)
        if record is None or record.move.uci != event.uci:
            return None
        record.assessment = event
        self._note_if_frozen(record, "assessment")
        return record

    def analysed(self, event: Analysis, facts: PositionFacts) -> PlyRecord | None:
        record = self._record_at(event.ply)
        if record is None or record.move.fen_after.split(" ")[:3] != event.fen.split(" ")[:3]:
            return None
        record.analysis = event
        record.facts = facts
        self._note_if_frozen(record, "analysis")
        return record

    def clock(self, white_ms: int | None, black_ms: int | None) -> None:
        if self.records:
            last = self.records[-1]
            if last.white_ms is None and last.black_ms is None:
                last.white_ms, last.black_ms = white_ms, black_ms

    def takeback(self, event: Takeback) -> None:
        """The last `plies` moves are withdrawn; they stay in history, marked so."""
        active = self.active_records()
        withdrawn = active[-event.plies :] if event.plies else []
        for record in withdrawn:
            record.withdrawn = True
        board = chess.Board(event.fen)
        names = ", ".join(self._numbered(r.move) for r in withdrawn) or "the last move"
        text = (
            f"Takeback: {names} withdrawn. The game is back at the position after "
            f"{self._last_active_name()}, {self._turn_text(board)}:\n{position_text(board)}\n"
            "Play continues from here; the next message is the replacement move."
        )
        self.entries.append(Entry("user", "takeback", text))

    def resync(self, event: Resync, board: chess.Board) -> None:
        """The board was re-read: the position jumped and some moves may have gone unseen."""
        for record in self.records:
            record.superseded = True  # the move list restarts here
        text = (
            f"The board was re-read ({event.reason}). Moves may have gone unseen. The position "
            f"is now, {self._turn_text(board)}:\n{position_text(board)}\n"
            "The move numbering restarts from this position."
        )
        self.entries.append(Entry("user", "resync", text))
        self.start_fen = board.fen()

    def covered(self) -> None:
        self.entries.append(
            Entry(
                "user",
                "covered",
                "The board is covered by a dialog. The game may be over or interrupted; wait.",
            )
        )

    def ended(self, event: GameEnded) -> None:
        how = f" ({event.reason})" if event.reason else ""
        self.entries.append(Entry("user", "ended", f"Game over: {event.result}{how}."))

    def note(self, text: str) -> None:
        """Something that happened, in a sentence, for the record."""
        self.entries.append(Entry("user", "note", text))

    def review(self, lines: Sequence[str]) -> None:
        """The review of the game once it is over: the tally, the accuracy, where it turned
        and why, from the engine's verdicts (profile/review.py)."""
        self.entries.append(Entry("user", "review", "The review of the game: " + " ".join(lines)))

    def user_said(self, text: str, verdicts: list[str] | None = None) -> None:
        body = f'"{text}"'
        if verdicts:
            body += " -- their claims, checked against the board: " + "; ".join(verdicts)
        self.entries.append(Entry("user", "ask", body))

    def coach_said(self, text: str, *, nod: bool = False) -> None:
        self.entries.append(Entry("assistant", "coach", text, nod=nod))

    # ---- rendering ---------------------------------------------------------------------

    def messages(self) -> list[ChatMessage]:
        """The harness output: one user message, in sections. The coach appends its task
        to it and answers as the assistant."""
        return [{"role": "user", "content": self.render()}]

    def render(self) -> str:
        """The sections: the setting; the game so far -- one line per move with its verdict,
        and under each move, indented, what was said after it (the player's words, your
        reply) and any event, the way a game record carries its annotations; the position
        now in full (the last ply, at the disclosure level in force), with what was said
        after it under it. Nothing is written twice: the move list is the record."""
        header = next((e.text for e in self.entries if e.kind == "game"), "")
        ply_entries = [e for e in self.entries if e.kind == "ply"]
        last = ply_entries[-1] if ply_entries else None
        game: list[str] = []
        now: list[str] = []
        sink = game
        if self.summary:
            game.append(f"Earlier dialogue, summarised: {self.summary}")
        stale_nods = self._stale_nods() if self.compact_record else set()
        for entry in self.entries:
            if entry.kind == "game" or entry.folded or id(entry) in stale_nods:
                continue
            if entry.kind == "ply":
                assert entry.record is not None
                if entry is last:
                    sink = now
                    sink.append(self.render_ply(entry.record, self.level))
                    continue
                if entry.frozen is None:
                    entry.frozen = self.render_ply(entry.record, full=False)
                sink.append(entry.frozen)
            elif entry.kind == "ask":
                sink.append(f"    PLAYER said: {entry.text}")
            elif entry.kind == "coach":
                sink.append(f'    YOU (the coach) said: "{entry.text}"')
            else:  # takeback, resync, covered, ended, note
                sink.append(f"    EVENT: {entry.text}")
        sections = [f"# Setting\n{header}"]
        if game:
            if self.compact_record:
                game = self._compact(game)
            sections.append("# Game so far\n" + "\n".join(game))
        if now:
            said = [x for x in now[1:]]
            sections.append(
                "# Position now\n"
                + now[0]
                + ("\n\n## Said here\n" + "\n".join(said) if said else "")
            )
        return "\n\n".join(x for x in sections if x)

    def _stale_nods(self) -> set[int]:
        """The nods older than the last NOD_PLIES plies, by id: left out of the record."""
        ply_entries = [e for e in self.entries if e.kind == "ply"]
        if len(ply_entries) <= NOD_PLIES:
            return set()
        cutoff = ply_entries[-NOD_PLIES]
        stale: set[int] = set()
        for entry in self.entries:
            if entry is cutoff:
                break
            if entry.kind == "coach" and entry.nod:
                stale.add(id(entry))
        return stale

    def _compact(self, lines: list[str]) -> list[str]:
        """The record's lines with the quiet plies run together: a ply line followed by no
        annotation, whose verdict is a bare label, joins a move list; every other line
        stands. "12. Bxf6 (the player): best." x 40 was most of the record."""
        out: list[str] = []
        run: list[str] = []

        def flush() -> None:
            if run:
                out.append("Moves: " + " ".join(run))
                run.clear()

        n = len(lines)
        for i, line in enumerate(lines):
            is_ply = line[:1].isdigit()
            annotated = i + 1 < n and lines[i + 1].startswith("    ")
            if is_ply and not annotated and _BARE_VERDICT.match(line):
                numbered = _BARE_VERDICT.match(line).group(1)  # type: ignore[union-attr]
                san = numbered.split(" ", 1)[1]
                if "..." in numbered and run and not run[-1].endswith("."):
                    run.append(san)  # Black's reply right after White's move: bare
                else:
                    run.append(numbered)
                continue
            flush()
            out.append(line)
        flush()
        if any(x.startswith("Moves: ") for x in out):
            out.insert(
                0,
                "(A move in a 'Moves:' list was sound -- best, excellent, great, good or book "
                "-- and nothing was said after it.)",
            )
        return out

    # ---- dialogue compaction -------------------------------------------------------------

    def foldable(self, keep_plies: int) -> list[tuple[str, Entry]]:
        """The dialogue entries older than the last `keep_plies` plies that are not yet in
        the summary, each with the move it followed."""
        ply_entries = [e for e in self.entries if e.kind == "ply"]
        if len(ply_entries) <= keep_plies:
            return []
        cutoff = ply_entries[-keep_plies]
        out: list[tuple[str, Entry]] = []
        after = "the start"
        for entry in self.entries:
            if entry is cutoff:
                break
            if entry.kind == "ply" and entry.record is not None:
                after = self._numbered(entry.record.move)
            elif entry.kind in ("ask", "coach") and not entry.folded:
                if self.compact_record and entry.nod:
                    entry.folded = True  # a nod is not summarised: the verdict is on its line
                    continue
                out.append((after, entry))
        return out

    def fold(self, entries: list[Entry], summary: str) -> None:
        """Replace `entries` in the record by `summary` (the whole earlier dialogue, as
        the summariser rewrote it)."""
        for entry in entries:
            entry.folded = True
        self.summary = summary.strip()

    def drain_new(self) -> list[Entry]:
        """Entries added since the last call, for the panel's conversation view."""
        new = self.entries[self._drained :]
        self._drained = len(self.entries)
        return new

    def render_ply(
        self,
        record: PlyRecord,
        level: DisclosureLevel | None = None,
        *,
        full: bool = True,
    ) -> str:
        """One ply of the harness. In `full`, the position now: the board, the last move
        with its verdict, the standing, the lines, the facts, what stands on the board,
        the player's record, in titled sections at `level`. Otherwise the record's one
        line for it: the move and its verdict."""
        level = self.level if level is None else level
        move = record.move
        board_after = chess.Board(move.fen_after)
        board_before = chess.Board(move.fen_before)
        user_colour = chess.WHITE if self.user_side == Side.WHITE else chess.BLACK
        who = "the player" if move.by_user else "the opponent"
        if not full:
            return f"{self._numbered(move)} ({who}): {self._verdict_line(record, board_before)}"
        # The legal moves name every move, the mating one with its mark: shown at every
        # level, the answer among them, since the model sees everything and what it says is
        # checked (Tai, 2026-09-26: "The model need to see everything including engine lines
        # at hint").
        lines: list[str] = [
            position_text(board_after, legal=True),
            self._turn_text(board_after).capitalize() + ".",
        ]
        clock = ""
        if record.white_ms is not None or record.black_ms is not None:
            clock = (
                f" Clock: white {self._clock(record.white_ms)}, "
                f"black {self._clock(record.black_ms)}."
            )
        took = f" ({move.think_ms / 1000:.1f} s)" if move.think_ms is not None and clock else ""
        lines.append(
            f"{self._numbered(move)} by {who}{took}: "
            + third_person(self._verdict_text(record, board_before, user_colour))
            + clock
        )
        sections = [("Board", "\n".join(lines[:2])), ("Last move", lines[2])]
        sections += self._position_sections(record, board_after, user_colour, level)
        return "\n\n".join(f"## {title}\n{body}" for title, body in sections)

    def _position_sections(
        self,
        record: PlyRecord,
        board: chess.Board,
        user_colour: chess.Color,
        level: DisclosureLevel,
    ) -> list[tuple[str, str]]:
        """The position the side to move faces, as (title, body) sections: the standing,
        the engine lines, the facts, the disclosure level, what stands on the board, the
        player's record."""
        user_to_move = board.turn == user_colour
        out: list[tuple[str, str]] = []
        analysis = record.analysis
        if analysis is not None and analysis.lines:
            out.append(
                ("Standing", self._standing_text(analysis.lines[0], board.turn, user_colour))
            )
        head, body = self._lines_block(record, board, user_colour, level)
        out.append((f"Engine lines {head}".rstrip(), "\n".join(body)))
        facts = self._facts_lines(record, board, user_colour, level)
        # Over the board, the game is over: a first hint for the mated position ("their queen
        # on b7 gives check") is noise beside the review.
        if facts and not board.is_game_over():
            out.append((f"Facts ({self._facts_owner(user_to_move)})", "\n".join(facts)))
        hiding = user_to_move and level < DisclosureLevel.MOVE and not board.is_game_over()
        if hiding:
            # The facts past the rung -- the piece, the target, the move and why it works --
            # are the model's to teach from, set apart so the rung's own stay plain (a hint
            # built on the heuristic sheet alone could only reword it).
            beyond = self._beyond_lines(record, board, level)
            if beyond:
                out.append((BEYOND_TITLE, "\n".join(beyond)))
            out.append(
                (
                    "Disclosure",
                    f"Level {level.name.lower()}: {self._level_words(level)}. {SEE_ALL_WORDS}",
                )
            )
        # The inventory lists the checks and captures on offer, the answer among them.
        out.append(
            ("On the board (complete lists)", "\n".join(inventory_lines(board, user_colour)))
        )
        tally = self._tally_text(record)
        if tally:
            out.append(("The player's record", tally))
        return out

    def _verdict_line(self, record: PlyRecord, board_before: chess.Board) -> str:
        """The record's line for an older ply: label, cost, the engine's preferred move and
        what it would have done."""
        a = record.assessment
        if a is None:
            return "verdict not in."
        label = a.label.value if a.label else a.severity.value
        if a.expected_loss < 0.005:
            return f"{label}."
        text = f"{label}, cost {a.expected_loss:.2f}"
        if a.best and a.best[0] != record.move.uci:
            try:
                better = chess.Move.from_uci(a.best[0])
                text += f"; engine preferred {board_before.san(better)}"
                would = effects_text(board_before, better, hazards=False)
                if would:
                    text += f" ({would})"
            except ValueError:
                pass
        return text + "."

    # ---- pieces of a ply message -------------------------------------------------------

    def _verdict_text(
        self, record: PlyRecord, board_before: chess.Board, user_colour: chess.Color
    ) -> str:
        move = record.move
        assessment = record.assessment
        board_after = chess.Board(move.fen_after)
        material, san, then = self._material_words(record, board_before, board_after, user_colour)
        if san is not None:
            # A capture the engine expects to be answered: "up a knight" after Nxe6 was
            # read as a knight won, and the coach said so, with dxe6 to come.
            material += f" before the expected recapture {san}, then {then}"
        if assessment is None:
            return f"verdict not in yet. Material: {material}."
        label = assessment.label.value if assessment.label else assessment.severity.value
        loss = assessment.expected_loss
        parts = [label]
        if loss >= 0.005:
            parts.append(f"it cost {loss:.2f} of expected score")
        else:
            parts.append("it cost nothing")
        if assessment.best and loss >= 0.005 and assessment.best[0] != move.uci:
            # The engine's line, when its first move is not the one played, and where it
            # leads: the reason the move was better, computed, so a "why was that bad?"
            # has an answer when the better move's own effects are nothing to say.
            # (Comparing the line's last move to the one played dropped the line whenever
            # a four-ply line happened to end on the same move: castling, most often.)
            better = san_line(board_before, assessment.best, limit=4)
            if better:
                leads = line_outcome(board_before, assessment.best[:LINE_PLIES], user_colour)
                notes = line_notes(board_before, assessment.best, limit=4, player=user_colour)
                parts.append(
                    f"the engine preferred {better}"
                    + (f" -> {leads}" if leads else "")
                    + (f" (move by move: {notes})" if notes else "")
                )
        text = "; ".join(parts) + f". Material: {material}."
        # What the move did on the board, and what the better move would have done: the
        # mechanism, computed, so the coach has a true reason to give instead of one it
        # makes up.
        try:
            played = board_before.parse_san(move.san)
            # A move the engine judged sound has its hazards resolved by the tactics.
            did = effects_text(
                board_before, played, hazards=assessment.severity is not Severity.NONE
            )
        except ValueError:
            did = ""
        if did:
            text += f" What {move.san} does: {did}."
        if assessment.best and loss >= 0.005 and assessment.best[0] != move.uci:
            try:
                better_move = chess.Move.from_uci(assessment.best[0])
                would = effects_text(board_before, better_move, hazards=False)
                better_san = board_before.san(better_move)
            except ValueError:
                would, better_san = "", ""
            if would:
                text += f" What {better_san} would do: {would}."
        tags = [t for t in assessment.tags if t.name in MOVE_MOTIFS]
        if tags:
            # By name only: the squares a tag carries are the tagger's evidence, and named
            # in the harness they were narrated as "the f7/f6/f4 piece sequence". The words
            # are the mover's ("your opponent's exposed king"), so they are set in the
            # third person for the side that moved.
            mover = "player" if move.by_user else "opponent"
            named = ", ".join(
                dict.fromkeys(third_person(motif_words(t.name), side=mover) for t in tags[:4])
            )
            text += f" Motifs in the move: {named}."
        return text

    def _material_words(
        self,
        record: PlyRecord,
        board_before: chess.Board,
        board_after: chess.Board,
        user_colour: chess.Color,
    ) -> tuple[str, str | None, str | None]:
        """The material after the move, to the player; and when the engine expects the
        capture taken back, the recapture (SAN) and the material after it."""
        material = material_text(board_after, user_colour)
        recapture = self._expected_recapture(record, board_before, board_after)
        if recapture is None:
            return material, None, None
        san, then = recapture
        return material, san, material_text(then, user_colour)

    def material_now(self) -> tuple[str, str | None, str | None] | None:
        """The material as the game stands, as the record's last move gives it: now, and
        the expected recapture with the material after it (`_material_words`). None before
        the first move."""
        active = self.active_records()
        if not active:
            return None
        record = active[-1]
        user_colour = chess.WHITE if self.user_side == Side.WHITE else chess.BLACK
        return self._material_words(
            record,
            chess.Board(record.move.fen_before),
            chess.Board(record.move.fen_after),
            user_colour,
        )

    @staticmethod
    def _expected_recapture(
        record: PlyRecord, board_before: chess.Board, board_after: chess.Board
    ) -> tuple[str, chess.Board] | None:
        """When the move was a capture and the engine's reply takes back on the same
        square: that reply in SAN, and the board after it."""
        analysis = record.analysis
        best = record.assessment.best if record.assessment is not None else []
        if analysis is not None and analysis.lines and analysis.lines[0].moves:
            expected = analysis.lines[0].moves[0]
        elif len(best) > 1 and best[0] == record.move.uci:
            # The position's own analysis not in yet: the engine's line for the move played
            # names the reply. Without it "up a queen for a pawn" was said of a queen about to
            # be taken back, and the model called the queen trade a queen won (the dev audit,
            # 2026-09-26).
            expected = best[1]
        else:
            return None
        try:
            played = chess.Move.from_uci(record.move.uci)
            reply = chess.Move.from_uci(expected)
        except ValueError:
            return None
        if not board_before.is_capture(played) or reply not in board_after.legal_moves:
            return None
        if reply.to_square != played.to_square or not board_after.is_capture(reply):
            return None
        then = board_after.copy(stack=False)
        then.push(reply)
        return board_after.san(reply), then

    def _lines_block(
        self,
        record: PlyRecord,
        board: chess.Board,
        user_colour: chess.Color,
        level: DisclosureLevel,
    ) -> tuple[str, list[str]]:
        """The engine lines section: its title's tail (whose move, whose scores) and its
        lines, in full at every level -- below the move level on the player's turn the title
        says the first move is the answer they are finding -- and the standing is a section
        of its own."""
        analysis = record.analysis
        if analysis is None or not analysis.lines:
            return "", ["not in yet."]
        to_move = board.turn
        user_to_move = to_move == user_colour
        whose = "the player's" if user_to_move else "the opponent's"
        answer = (
            "; the best first move is the answer they are finding: teach from it, never name it"
            if user_to_move and level < DisclosureLevel.MOVE
            else ""
        )
        top = analysis.lines[0].expected_score
        limit = LINE_PLIES
        best = analysis.lines[:BEST_LINES]
        weaker = analysis.lines[BEST_LINES : BEST_LINES + WEAKER_LINES]
        out: list[str] = []
        for i, line in enumerate(best, start=1):
            out.append(
                f"  {i}. {san_line(board, line.moves, limit=limit)}  "
                f"{self._score(line.expected_score, to_move, user_colour)}"
                + self._does(board, line.moves)
                + self._leads(board, line.moves[:limit], user_colour)
            )
            if i == 1:
                # The best line move by move, computed: what a line may be explained by.
                notes = line_notes(board, line.moves, limit=limit, player=user_colour)
                if notes:
                    out.append(f"     move by move: {notes}")
        if weaker:
            # The scores are from the player's side, so on the opponent's turn a weaker
            # line for them shows a higher number: the title says whose.
            out.append(f"  weaker for {whose.removesuffix("'s")}:")
            for i, line in enumerate(weaker, start=len(best) + 1):
                drop = top - line.expected_score
                gap = f"{drop:.2f} behind the best" if drop >= 0.005 else "level with the best"
                out.append(
                    f"  {i}. {san_line(board, line.moves, limit=limit)}  "
                    f"{self._score(line.expected_score, to_move, user_colour)} ({gap})"
                )
        return f"({whose} move; scores from the player's side{answer})", out

    def _standing_text(self, top: Line, to_move: chess.Color, user_colour: chess.Color) -> str:
        """Who stands better, in words, with the engine's number named by side: "The player
        is better (Stockfish: White ahead by about 1.2 pawns)." A bare 0-1 number was read
        as a pawn score, or not at all; the words are what a coach says, and the side named
        leaves no sign to guess."""
        score = top.expected_score if to_move == user_colour else 1.0 - top.expected_score
        words = standing_words(score, third_person=True)
        text = f"{words[:1].upper()}{words[1:]} (by the numbers: "
        if top.mate is not None:
            n = abs(top.mate)
            winner = to_move if top.mate > 0 else not to_move
            text += f"mate in {n} for {'White' if winner == chess.WHITE else 'Black'}"
        elif top.score_cp is not None:
            cp_white = top.score_cp if to_move == chess.WHITE else -top.score_cp
            if abs(cp_white) < 15:
                text += "level, 0.0 pawns"
            else:
                ahead = "White" if cp_white > 0 else "Black"
                text += f"{ahead} ahead by about {abs(cp_white) / 100:.1f} pawns"
        else:
            text += "no number"
        return text + ")."

    @staticmethod
    def _threat_words(board: chess.Board, san: str) -> str:
        """A threat with what it does: "Nd3+ (forks the queen on d2 and the rook on f1)"."""
        if board.is_check():
            return san
        passed = board.copy(stack=False)
        passed.push(chess.Move.null())
        try:
            text = effects_text(passed, passed.parse_san(san), hazards=False)
        except ValueError:
            return san
        return f"{san} ({text})" if text else san

    @staticmethod
    def _leads(board: chess.Board, moves: list[str], user_colour: chess.Color) -> str:
        """Where the line leads, for the lines block: "-> you win a pawn; the e-file opens"."""
        text = line_outcome(board, moves, user_colour)
        return f"  -> {text}" if text else ""

    @staticmethod
    def _does(board: chess.Board, moves: list[str]) -> str:
        """What a line's first move does on the board, for the lines block."""
        if not moves:
            return ""
        try:
            text = effects_text(board, chess.Move.from_uci(moves[0]), hazards=False)
        except ValueError:
            return ""
        return f"  ({text})" if text else ""

    # The facts in the order a coach reads them: what kind of moment, the goal and the
    # pattern, where to look, then the move.
    _FACT_KEYS = (
        "phase",
        "decisive",
        "only_move",
        "nature",
        "situation",
        "goal",
        "theme",
        "mechanism",
        "plan",
        "question",
        "target",
        "piece",
        "nudge",
        "region",
        "why",
        "tactic",
        "threats",
        "ideas",
        "focus",
    )
    # What a fact is called on the sheet, where its key would not say it.
    _FACT_NAMES = {
        # The template's first hint whole: its stakes, key fact, principle and trap, several
        # of which exist nowhere else on the sheet. Without it the model was asked for "the
        # stakes" and "the trap" and never given them (parity pass, 2026-09-25).
        "situation": (
            "the first hint in the board's own words, all of it true (to the player, as it "
            "stands): the least a hint says"
        ),
        "theme": "pattern",
        "mechanism": "how it works",
        "question": "a question that sets them looking (to the player, as it stands)",
        "nudge": "a question from the target to the move (to the player, as it stands)",
        "why": "the move and what it does",
        # Whose: the other side's moves, were the side to move to pass.
        # Named per side in `_fact_lines`: whose threats, and at what moment.
        "threats": "threats",
    }
    # The facts addressed to the player, kept in the second person.
    _TO_THE_PLAYER = ("question", "nudge", "situation")

    @staticmethod
    def _facts_owner(user_to_move: bool) -> str:
        return (
            "the player to move"
            if user_to_move
            else "the opponent to move; the opponent's plan and ideas"
        )

    def _facts_lines(
        self,
        record: PlyRecord,
        board: chess.Board,
        user_colour: chess.Color,
        level: DisclosureLevel,
    ) -> list[str]:
        """The facts of the position, one per line, in the third person for the side to
        move: the opponent's facts are the opponent's plan and ideas, and say so."""
        facts = record.facts
        if facts is None:
            return []
        user_to_move = board.turn == user_colour
        level = level if user_to_move else DisclosureLevel.LINE
        visible = visible_facts(facts, level)
        if not user_to_move:
            # The hints are for the side to move: on the opponent's turn the first hint and the
            # two questions are the opponent's, and labelled "to the player" the model asked
            # the player the opponent's question (parity pass 2, 2026-09-25).
            for key in ("situation", "question", "nudge"):
                visible.pop(key, None)
        side = "player" if user_to_move else "opponent"
        out: list[str] = []
        if record.analysis is not None and record.analysis.opening:
            previous = self._record_before(record)
            before = previous.analysis.opening if previous and previous.analysis else None
            if before != record.analysis.opening:
                out.append(f"opening: {record.analysis.opening}")
        out.extend(self._fact_lines(visible, board, side))
        return out

    def _beyond_lines(
        self, record: PlyRecord, board: chess.Board, level: DisclosureLevel
    ) -> list[str]:
        """The facts past `level` on the player's turn, one per line: the piece, the target,
        the move and why it works -- the model's to teach from, not to say."""
        facts = record.facts
        if facts is None:
            return []
        shown = visible_facts(facts, level)
        every = visible_facts(facts, DisclosureLevel.LINE)
        beyond = {key: value for key, value in every.items() if key not in shown}
        return self._fact_lines(beyond, board, "player")

    def faced_text(
        self,
        facts: PositionFacts,
        board: chess.Board,
        after: chess.Board | None = None,
        played: chess.Board | None = None,
    ) -> str:
        """The facts of the position the player faced before their move, all of them (it is
        past): the lesson after the move is taught from them, and the record keeps only one
        line for that position. With `after`, the board once the move (and the reply the
        engine expects) is played: an idea about a piece that no longer stands there is left
        out -- "the opponent's backward d-pawn on d6 is a lasting target" beside Qxc5 dxc5
        was said as "preserving your lasting target on d6" (the test audit, 2026-09-26).
        With `played`, the board right after the move: a better move that "stops" their
        threat, where the move played stopped it too, says so -- "Qc4+ ... stops gxh1=Q+"
        beside 19.Qxg2, which took the pawn, was taught as "the priority was to stop the
        promotion" (the dev audit, 2026-09-26)."""
        visible = visible_facts(facts, DisclosureLevel.LINE)
        # The hint's questions were for finding the move; the lesson is on the move made.
        for key in ("question", "nudge"):
            visible.pop(key, None)
        if after is not None and visible.get("ideas"):
            visible["ideas"] = [i for i in visible["ideas"] if not _gone(str(i), after)]
        lines = self._fact_lines(visible, board, "player", past=True)
        stopped = _also_stopped(facts, played)
        if stopped:
            lines.append(f"the move played also stops {stopped}")
        return "\n".join([position_text(board, legal=False), *lines])

    def _fact_lines(
        self, visible: dict, board: chess.Board, side: str, *, past: bool = False
    ) -> list[str]:  # noqa: C901
        """The sheet's lines for `visible`, in reading order, in the third person for `side`;
        `past` for a position the player has moved from."""
        names = self._FACT_NAMES
        if side == "opponent":
            # Their best move now, not the player's: "axb4 meets the player's idea" was taken
            # into the player's better line (the dev audit, 2026-09-26).
            names = {**names, "why": "the opponent's best move now and what it does"}
        if past:
            names = {
                **names,
                "situation": "the first hint for that position (to the player, as it stood)",
            }
        out: list[str] = []
        for key in self._FACT_KEYS:
            value = visible.get(key)
            if value in (None, "", [], False):
                continue
            name = names.get(key, key.replace("_", " "))
            if value is True:
                out.append(f"{name}: yes")
            elif key == "ideas":
                out.append("ideas:")
                out.extend(f"  - {third_person(str(v), side=side)}" for v in value)
            elif key == "threats":
                # A threat is the other side's move if the side to move passed; what it does
                # is computed on that position. Named with its owner and moment: "what the
                # other side threatens" was read as the opponent's where it was the player's
                # ("your opponent's queen ... e5" of the player's own e5), and a threat as
                # something a move stopped ("Rae1 before they can use Re3": the test audit).
                owner, mover = (
                    ("the opponent's", "the player")
                    if side == "player"
                    else ("the player's", "the opponent")
                )
                out.append(
                    f"{owner} threats, if {mover} passed: "
                    + ", ".join(f"{owner} {self._threat_words(board, str(v))}" for v in value)
                )
            elif isinstance(value, list):
                out.append(f"{name}: " + ", ".join(str(v) for v in value))
            elif key in self._TO_THE_PLAYER:
                out.append(f"{name}: {value}")
            elif key == "target" and "free square" in str(value):
                # The king's free squares as it stands: taken as what a move left them ("Qb4
                # leaves only e4 and e5" where Qb4 took e4 away: the test audit, 2026-09-26).
                said = third_person(str(value), side=side)
                out.append(f"{name}: {said} (as it stands, before any move)")
            else:
                out.append(f"{name}: {third_person(str(value), side=side)}")
        return out

    def _index(self, record: PlyRecord) -> int:
        """Where `record` itself stands in the records: by identity, as two records can be
        equal field for field (the parity audit, 2026-09-26)."""
        return next(i for i, r in enumerate(self.records) if r is record)

    def _tally_text(self, through: PlyRecord) -> str:
        """The player's moves judged so far, as of `through`: the tally reads as the game
        went, and the last move's message holds the whole of it."""
        counts: dict[str, int] = {}
        lost: list[tuple[float, str]] = []
        n = 0
        for record in self.judged_records():
            if self._index(record) > self._index(through):
                break
            if not record.move.by_user or record.assessment is None:
                continue
            n += 1
            label = (
                record.assessment.label.value
                if record.assessment.label
                else record.assessment.severity.value
            )
            counts[label] = counts.get(label, 0) + 1
            if record.assessment.severity.value != "none":
                cost = record.assessment.expected_loss
                lost.append((cost, f"{self._numbered(record.move)} ({label})"))
        if not n:
            return ""
        tally = ", ".join(f"{counts[k]} {k}" for k in LABEL_ORDER if k in counts)
        text = f"The player's moves so far ({n} judged): {tally}."
        if lost:
            # The costliest three, then the latest: "how did I do?" is answered from where
            # the game turned, and the last six alone named 39.Rb7 and 52.Kh3 over the
            # blunders at moves 19 and 24 (round 2).
            costliest = [m for _, m in sorted(lost, key=lambda x: -x[0])[:3]]
            latest = [m for _, m in lost[-3:] if m not in costliest]
            text += " Ground lost most at: " + "; ".join(costliest) + "."
            if latest:
                text += " Lately: " + "; ".join(latest) + "."
        # The opening the game was in, as the book last named it: it was shown only at
        # the ply where the name changed, and asked later the coach named one from memory.
        opening = self._opening_through(through)
        if opening:
            text += f" Opening: {opening}."
        return text

    def _opening_through(self, through: PlyRecord) -> str | None:
        """The last opening name the book gave, up to `through`."""
        for record in reversed(self.records[: self._index(through) + 1]):
            if record.withdrawn or record.analysis is None:
                continue
            if record.analysis.opening:
                return record.analysis.opening
        return None

    # ---- helpers -----------------------------------------------------------------------

    def active_records(self) -> list[PlyRecord]:
        """The moves of the game as it stands: not taken back, since the last re-read."""
        return [r for r in self.records if not r.withdrawn and not r.superseded]

    def judged_records(self) -> list[PlyRecord]:
        """Every move that was played and judged, across re-reads, for the tally."""
        return [r for r in self.records if not r.withdrawn]

    def same_idea(self, key: str, before_ply: int, limit: int = 2) -> list[str]:
        """The player's moves before `before_ply` played in a position with the very same
        idea -- the pattern or the plan its facts name, `key` -- the latest `limit` of them,
        one line each: the move, how it went, and what the coach said after it. Book moves
        are theory, not a pattern found. The only earlier moments a lesson may tie itself to:
        told to find its own in the record, the model misremembered them (quality rounds 2
        and 3, 2026-09-25)."""
        active = self.active_records()
        lines: list[str] = []
        for i, record in enumerate(active):
            move = record.move
            if i == 0 or move.ply >= before_ply or not move.by_user:
                continue
            faced = active[i - 1]
            if faced.move.ply != move.ply - 1 or faced.facts is None:
                continue
            if (faced.facts.theme or faced.facts.plan) != key:
                continue
            a = record.assessment
            if a is None or (a.label is not None and a.label.value == "book"):
                continue
            label = a.label.value if a.label else a.severity.value
            better = ""
            if a.best and a.best[0] != move.uci:
                try:
                    better = chess.Board(move.fen_before).san(chess.Move.from_uci(a.best[0]))
                except ValueError:
                    better = ""
            if not better:
                how = f"{label} -- the player found it"
            elif a.severity is Severity.NONE:
                how = f"{label}; {better} was better"
            else:
                rated = ("an " if label[0] in "aeiou" else "a ") + label
                how = f"{rated}; {better} was better"
            line = f"- At {move.san} (the player's move): {how}."
            said = self._said_after(record)
            if said:
                line += f'\n    YOU (the coach) said: "{said}"'
            lines.append(line)
        return lines[-limit:]

    def _said_after(self, record: PlyRecord) -> str:
        """The coach's first words after `record`'s move, before the next move: "" if none."""
        after = False
        for entry in self.entries:
            if entry.kind == "ply":
                if after:
                    return ""
                after = entry.record is record
            elif after and entry.kind == "ask":
                return ""
            elif after and entry.kind == "coach" and entry.text:
                return entry.text
        return ""

    def moves_before(self, ply: int) -> frozenset[str]:
        """The moves of the game before `ply` (SAN) -- those played and those the verdicts
        preferred -- that name a piece, a capture or castling: a pawn's step reads the same
        as a square, and a reply names squares for what stands on them."""
        out: set[str] = set()
        for record in self.active_records():
            if record.move.ply >= ply:
                continue
            out.add(record.move.san)
            a = record.assessment
            if a is not None and a.best:
                with contextlib.suppress(ValueError):
                    out.add(chess.Board(record.move.fen_before).san(chess.Move.from_uci(a.best[0])))
        return frozenset(m for m in out if not re.fullmatch(r"[a-h][1-8](?:=[QRBN])?[+#]?", m))

    def _record_before(self, record: PlyRecord) -> PlyRecord | None:
        active = self.active_records()
        i = next((k for k, r in enumerate(active) if r is record), None)
        return active[i - 1] if i else None

    def _record_at(self, ply: int) -> PlyRecord | None:
        for record in reversed(self.records):
            if not record.withdrawn and not record.superseded and record.move.ply == ply:
                return record
        return None

    def _note_if_frozen(self, record: PlyRecord, what: str) -> None:
        """A fact that lands after its ply message went out goes in a note of its own."""
        for entry in self.entries:
            if entry.record is record and entry.frozen is not None:
                board_before = chess.Board(record.move.fen_before)
                user_colour = chess.WHITE if self.user_side == Side.WHITE else chess.BLACK
                name = self._numbered(record.move)
                if what == "assessment":
                    verdict = self._verdict_text(record, board_before, user_colour)
                    text = f"Engine note on {name}: {verdict}"
                else:
                    after = chess.Board(record.move.fen_after)
                    shown = "\n".join(
                        f"{title}: {body}"
                        for title, body in self._position_sections(
                            record, after, user_colour, DisclosureLevel.LINE
                        )
                        if title in ("Standing", "Engine lines")
                    )
                    text = f"Engine note on the position after {name}:\n{shown}"
                self.entries.append(Entry("user", "note", text))
                return

    @staticmethod
    def _numbered(move: MovePlayed) -> str:
        """The move with its number, from the position it was played in (not the ply
        parity: after a re-read the numbering restarts wherever the board stood)."""
        before = chess.Board(move.fen_before)
        number = before.fullmove_number
        return f"{number}. {move.san}" if before.turn == chess.WHITE else f"{number}... {move.san}"

    def _last_active_name(self) -> str:
        active = self.active_records()
        return self._numbered(active[-1].move) if active else "the start"

    @staticmethod
    def _turn_text(board: chess.Board) -> str:
        if board.is_game_over():
            return f"the game is over ({board.result()})"
        return ("White" if board.turn == chess.WHITE else "Black") + " to move"

    @staticmethod
    def _clock(ms: int | None) -> str:
        if ms is None:
            return "?"
        seconds = ms // 1000
        return f"{seconds // 60}:{seconds % 60:02d}"

    @staticmethod
    def _score(expected: float, to_move: chess.Color, user_colour: chess.Color) -> str:
        """An expected score from the mover's view, shown from the user's side."""
        value = expected if to_move == user_colour else 1.0 - expected
        return f"{value:.2f}"

    @staticmethod
    def _level_words(level: DisclosureLevel) -> str:
        return {
            DisclosureLevel.SILENT: "say nothing about the position",
            DisclosureLevel.IDEA: IDEA_LEVEL_WORDS,
            DisclosureLevel.FOCUS: "the idea, the target and the piece; not the move",
            DisclosureLevel.MOVE: "the move and why it works may be named",
            DisclosureLevel.LINE: "everything, the line included",
        }[level]
