"""The game as the harness the model reads: one user message, in sections.

`# Setting` says whom the coach coaches. `# Game so far` is the record: one line per move
with the engine's verdict on it -- the label, the cost, the move the engine preferred and
where its line leads -- and under each move, indented, what was said after it (the
player's words, the coach's own replies) and any event (a takeback, a re-read of the board,
a dialog, the end of the game), the way a game record carries its annotations. Once the
model has seen a move's line it is frozen and never edited; facts that arrive later go in
a note of their own. `# Position now` is the last ply in full at the disclosure level in
force for the position the player faces: the board, the last move, the standing, the
engine's lines (withheld on the player's own turn below MOVE), the facts in the third
person per side, the disclosure line, the complete inventory of what stands on the board,
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

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

import chess

from sidecoach.analysis.engine import san_line
from sidecoach.coach.disclosure import PositionFacts, visible_facts
from sidecoach.coach.effects import effects_text
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
    "their moves the facts name -- with the principle behind it; never the player's move, "
    "the square it goes to, or which of their own pieces moves"
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
                "hint, give one at the level allowed, never the move itself unless asked "
                "for it:\n" + position_text(chess.Board(event.start_fen))
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
        # The legal moves name every move, the mating one with its mark: for the position
        # the player faces they are shown only once moves may be named.
        user_to_move = board_after.turn == user_colour
        lines: list[str] = [
            position_text(board_after, legal=not (user_to_move and level < DisclosureLevel.MOVE)),
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
        if user_to_move and level < DisclosureLevel.MOVE:
            out.append(("Disclosure", f"Level {level.name.lower()}: {self._level_words(level)}."))
        # The inventory lists the checks and captures on offer, the answer among them: on
        # the player's turn it comes with the move (at the focus rung it named Qxf7#).
        if not (user_to_move and level < DisclosureLevel.MOVE):
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
        material = material_text(board_after, user_colour)
        recapture = self._expected_recapture(record, board_before, board_after)
        if recapture is not None:
            # A capture the engine expects to be answered: "up a knight" after Nxe6 was
            # read as a knight won, and the coach said so, with dxe6 to come.
            san, then = recapture
            material += (
                f" before the expected recapture {san}, then {material_text(then, user_colour)}"
            )
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
                parts.append(f"the engine preferred {better}" + (f" -> {leads}" if leads else ""))
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

    @staticmethod
    def _expected_recapture(
        record: PlyRecord, board_before: chess.Board, board_after: chess.Board
    ) -> tuple[str, chess.Board] | None:
        """When the move was a capture and the engine's reply takes back on the same
        square: that reply in SAN, and the board after it."""
        analysis = record.analysis
        if analysis is None or not analysis.lines or not analysis.lines[0].moves:
            return None
        try:
            played = chess.Move.from_uci(record.move.uci)
            reply = chess.Move.from_uci(analysis.lines[0].moves[0])
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
        lines. Withheld below the candidate level on the player's turn -- what may be
        said is in the facts -- and the standing is a section of its own."""
        analysis = record.analysis
        if analysis is None or not analysis.lines:
            return "", ["not in yet."]
        to_move = board.turn
        user_to_move = to_move == user_colour
        if user_to_move and level < DisclosureLevel.MOVE:
            return "", ["withheld at this disclosure level."]
        whose = "the player's" if user_to_move else "the opponent's"
        top = analysis.lines[0].expected_score
        limit = LINE_PLIES if (not user_to_move or level >= DisclosureLevel.LINE) else 1
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
        return f"({whose} move; scores from the player's side)", out

    def _standing_text(self, top: Line, to_move: chess.Color, user_colour: chess.Color) -> str:
        """Who stands better, in words, with the engine's number named by side: "The player
        is better (Stockfish: White ahead by about 1.2 pawns)." A bare 0-1 number was read
        as a pawn score, or not at all; the words are what a coach says, and the side named
        leaves no sign to guess."""
        score = top.expected_score if to_move == user_colour else 1.0 - top.expected_score
        words = standing_words(score, third_person=True)
        text = f"{words[:1].upper()}{words[1:]} (Stockfish: "
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
        "threats": "what the other side threatens, if the side to move passed",
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

    def faced_text(self, facts: PositionFacts, board: chess.Board) -> str:
        """The facts of the position the player faced before their move, all of them (it is
        past): the lesson after the move is taught from them, and the record keeps only one
        line for that position."""
        visible = visible_facts(facts, DisclosureLevel.LINE)
        # The hint's questions were for finding the move; the lesson is on the move made.
        for key in ("question", "nudge"):
            visible.pop(key, None)
        return "\n".join(self._fact_lines(visible, board, "player", past=True))

    def _fact_lines(
        self, visible: dict, board: chess.Board, side: str, *, past: bool = False
    ) -> list[str]:
        """The sheet's lines for `visible`, in reading order, in the third person for `side`;
        `past` for a position the player has moved from."""
        names = self._FACT_NAMES
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
                # A threat is the opponent's move if the side to move passed; what it does
                # is computed on that position, so the coach can say it in words.
                out.append(
                    f"{name}: " + ", ".join(self._threat_words(board, str(v)) for v in value)
                )
            elif isinstance(value, list):
                out.append(f"{name}: " + ", ".join(str(v) for v in value))
            elif key in self._TO_THE_PLAYER:
                out.append(f"{name}: {value}")
            else:
                out.append(f"{name}: {third_person(str(value), side=side)}")
        return out

    def _tally_text(self, through: PlyRecord) -> str:
        """The player's moves judged so far, as of `through`: the tally reads as the game
        went, and the last move's message holds the whole of it."""
        counts: dict[str, int] = {}
        lost: list[str] = []
        n = 0
        for record in self.judged_records():
            if self.records.index(record) > self.records.index(through):
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
                lost.append(f"{self._numbered(record.move)} ({label})")
        if not n:
            return ""
        tally = ", ".join(f"{counts[k]} {k}" for k in LABEL_ORDER if k in counts)
        text = f"The player's moves so far ({n} judged): {tally}."
        if lost:
            text += " Ground lost at: " + "; ".join(lost[-6:]) + "."
        # The opening the game was in, as the book last named it: it was shown only at
        # the ply where the name changed, and asked later the coach named one from memory.
        opening = self._opening_through(through)
        if opening:
            text += f" Opening: {opening}."
        return text

    def _opening_through(self, through: PlyRecord) -> str | None:
        """The last opening name the book gave, up to `through`."""
        for record in reversed(self.records[: self.records.index(through) + 1]):
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
