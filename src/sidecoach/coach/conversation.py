"""The game as one conversation with the model.

Every move on the board -- the user's and the opponent's -- is a `user` message, built from
the live game state and written from the user's side. The coach is an `assistant` turn only
when it was prompted. Nothing is ever edited once the model has seen it: a takeback, a
re-read of the board, a dialog over it, the end of the game, each is a message of its own
that says what changed, so the position never jumps silently. Asked how the game went, the
model has the game in front of it, move by move, with the engine's word on each.

A move message carries, in order: the board state (the moves so far, the position, whose
turn), the move and who made it, the engine's verdict on it and what it did to the game, the
engine's lines -- the best three, the next three, and the played move with the replies it
allows -- and everything else that grounds a reply: the motifs with their squares, the
strategic features, the opponent's threats, the opening, the phase, the clock, and a running
tally of the user's moves by label. What the message says about the position the user now
faces obeys the disclosure level in force when it is rendered; a message is frozen the
first time it is sent in its final form, and facts that arrive later go in a message of
their own.

The encoding spends tokens where the model reasons. The current message carries the moves so
far and the last two plies carry the board diagram; an older ply's message says the same
position as a FEN, writes its engine lines without move numbers, and does not repeat in its
facts the candidates and line that its lines block already gives. Nothing is dropped: every
verdict, every line, every fact, every tally is there for every ply. Measured on a 165-ply
game the full form ran to about 130k tokens, past what a retry could fit in the model's
context; this form is well under half, and a ply's message takes it once it is older than
the last two, so the server's cached prefix stays valid.

The `Coach` owns one of these per game and asks it for the messages of every request.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from sidecoach.analysis.engine import san_line
from sidecoach.coach.disclosure import PositionFacts, visible_facts
from sidecoach.coach.facts import ascii_board, material_text
from sidecoach.events import (
    Analysis,
    DisclosureLevel,
    GameEnded,
    GameStarted,
    MoveAssessment,
    MovePlayed,
    Resync,
    Side,
    Takeback,
)
from sidecoach.llm.base import ChatMessage

# The latest plies whose messages carry the full encoding: the board diagram, numbered
# lines, the candidates and line in the facts. Older plies say the same with fewer tokens.
RECENT_PLIES = 2
BEST_LINES = 3
WEAKER_LINES = 3
LINE_PLIES = 8
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


_THEIRS = (
    ("your opponent's", "your"),
    ("your opponent", "you"),
    ("You are", "They are"),
    ("you are", "they are"),
    ("You have", "They have"),
    ("you have", "they have"),
    ("Your ", "Their "),
    ("your ", "their "),
    ("yours", "theirs"),
    (" you ", " they "),
    (" you.", " them."),
    (" you,", " them,"),
)


def _theirs(text: str) -> str:
    """A fact written for the side to move, reworded for the opponent: 'your king is
    still in the centre' becomes 'their king is still in the centre'."""
    for old, new in _THEIRS:
        text = text.replace(old, new)
    return text


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


@dataclass
class Conversation:
    user_side: Side = Side.WHITE
    level_text: str = "around 1200"
    entries: list[Entry] = field(default_factory=list)
    records: list[PlyRecord] = field(default_factory=list)
    start_fen: str = chess.STARTING_FEN
    # The disclosure level to render the position the user faces with, set by the coach
    # before each request from the current position's disclosure state.
    level: DisclosureLevel = DisclosureLevel.QUESTION
    _drained: int = 0

    # ---- the game's events, in the order they happen -----------------------------------

    def new_game(
        self,
        event: GameStarted,
        *,
        opponent: str = "a bot",
        control: str = "",
    ) -> None:
        self.user_side = event.user_side
        self.start_fen = event.start_fen
        self.entries = []
        self.records = []
        self._drained = 0
        side = "White" if event.user_side == Side.WHITE else "Black"
        text = f"New game. You coach the player, who is {side}, against {opponent}."
        if control:
            text += f" Time control: {control}."
        if event.start_fen != chess.STARTING_FEN:
            text += "\nThe game is joined in progress; the position is:\n" + ascii_board(
                chess.Board(event.start_fen)
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
            f"{self._last_active_name()}, {self._turn_text(board)}:\n{ascii_board(board)}\n"
            "Play continues from here; the next message is the replacement move."
        )
        self.entries.append(Entry("user", "takeback", text))

    def resync(self, event: Resync, board: chess.Board) -> None:
        """The board was re-read: the position jumped and some moves may have gone unseen."""
        for record in self.records:
            record.superseded = True  # the move list restarts here
        text = (
            f"The board was re-read ({event.reason}). Moves may have gone unseen. The position "
            f"is now, {self._turn_text(board)}:\n{ascii_board(board)}\n"
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

    def user_said(self, text: str, verdicts: list[str] | None = None) -> None:
        body = f'The player says: "{text}"'
        if verdicts:
            body += "\nTheir claims, checked against the board:\n- " + "\n- ".join(verdicts)
        self.entries.append(Entry("user", "ask", body))

    def coach_said(self, text: str) -> None:
        self.entries.append(Entry("assistant", "coach", text))

    # ---- rendering ---------------------------------------------------------------------

    def messages(self) -> list[ChatMessage]:
        """The conversation as the model sees it. Ply messages are frozen as they go out;
        the last ply message stays live until the next move, since the disclosure level
        for the position the user faces can rise while they think."""
        out: list[ChatMessage] = []
        ply_indexes = [i for i, e in enumerate(self.entries) if e.kind == "ply"]
        last_ply = ply_indexes[-1] if ply_indexes else None
        # The position the user faces and the one they moved from carry the diagram; an
        # older ply carries its FEN -- the same position, a quarter of the tokens.
        recent = set(ply_indexes[-RECENT_PLIES:])
        for i, entry in enumerate(self.entries):
            if entry.kind != "ply":
                out.append({"role": entry.role, "content": entry.text})  # type: ignore[typeddict-item]
                continue
            if entry.frozen is not None and i not in recent:
                text = entry.frozen
            else:
                assert entry.record is not None
                # History is at full disclosure: a position already played through hides
                # nothing. Only the position the user faces now is rendered at the level.
                level = self.level if i == last_ply else DisclosureLevel.LINE
                text = self.render_ply(
                    entry.record, level, current=i == last_ply, recent=i in recent
                )
                if i not in recent:
                    entry.frozen = text  # its final form: frozen from here on
            out.append({"role": "user", "content": text})
        return out

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
        current: bool = True,
        recent: bool = True,
    ) -> str:
        """The message for one ply. `current` is the position the user faces (it alone
        carries the moves so far); `recent` is one of the last two plies (they carry the
        board diagram; older plies carry the FEN, the lines without move numbers, and
        no copy of the candidates and line the lines block already gives)."""
        level = self.level if level is None else level
        move = record.move
        board_after = chess.Board(move.fen_after)
        board_before = chess.Board(move.fen_before)
        user_colour = chess.WHITE if self.user_side == Side.WHITE else chess.BLACK
        who = "you" if move.by_user else "your opponent"
        lines: list[str] = []
        # 1. Board state. The moves so far are written on the current message only: every
        # earlier message carries its own move, so the sequence is there in full, and the
        # line repeated on each of them was the one part of a message that grew with the
        # game (a fifth of a 165-ply game's tokens, all of it duplicate).
        if current:
            pgn = self._pgn_through(record)
            lines.append(f"Moves so far: {pgn}" if pgn else "Moves so far: (none)")
        if recent:
            lines.append(ascii_board(board_after))
        else:
            lines.append(f"Position (FEN): {board_after.fen()}")
        lines.append(self._turn_text(board_after).capitalize() + ".")
        # 2. The move.
        took = f", {move.think_ms / 1000:.1f} s" if move.think_ms is not None else ""
        clock = ""
        if record.white_ms is not None or record.black_ms is not None:
            clock = (
                f" Clock: white {self._clock(record.white_ms)}, "
                f"black {self._clock(record.black_ms)}."
            )
        lines.append(f"Move: {self._numbered(move)} by {who}{took}.{clock}")
        # 3. The engine's verdict on the move.
        lines.append(self._verdict_text(record, board_before, user_colour))
        # 4. The lines for the position now, subject to disclosure when it is the user's move.
        lines.extend(self._lines_text(record, board_after, user_colour, level, numbered=recent))
        # 5. Everything else.
        lines.extend(
            self._grounding_text(record, board_after, user_colour, level, with_lines=recent)
        )
        if move.by_user:
            lines.append(self._tally_text(record))
        return "\n".join(line for line in lines if line)

    # ---- pieces of a ply message -------------------------------------------------------

    def _verdict_text(
        self, record: PlyRecord, board_before: chess.Board, user_colour: chess.Color
    ) -> str:
        move = record.move
        assessment = record.assessment
        material = material_text(chess.Board(move.fen_after), user_colour)
        if assessment is None:
            return f"Engine verdict on {move.san}: not in yet. Material: {material}."
        label = assessment.label.value if assessment.label else assessment.severity.value
        loss = assessment.expected_loss
        parts = [f"Engine verdict on {move.san}: {label}"]
        if loss >= 0.005:
            parts.append(f"it cost {loss:.2f} of expected score")
        else:
            parts.append("it cost nothing")
        if assessment.best and loss >= 0.005:
            better = san_line(board_before, assessment.best, limit=4)
            if better and not better.endswith(move.san):
                parts.append(f"the engine preferred {better}")
        text = "; ".join(parts) + f". Material: {material}."
        tags = [t for t in assessment.tags if t.name not in ("quietMove", "capture", "check")]
        if tags:
            named = ", ".join(
                f"{t.name}" + (f" ({'/'.join(t.squares)})" if t.squares else "") for t in tags[:4]
            )
            text += f" Motifs in the move: {named}."
        return text

    def _lines_text(
        self,
        record: PlyRecord,
        board: chess.Board,
        user_colour: chess.Color,
        level: DisclosureLevel,
        *,
        numbered: bool = True,
    ) -> list[str]:
        analysis = record.analysis
        if analysis is None or not analysis.lines:
            return ["Engine lines for the position now: not in yet."]
        to_move = board.turn
        user_to_move = to_move == user_colour
        if user_to_move and level < DisclosureLevel.CANDIDATE:
            # The position the user faces: the lines are withheld at this level; what may be
            # said is in the grounding facts below.
            return [
                f"Engine lines for the position now: withheld (disclosure level "
                f"{level.name.lower()}: {self._level_words(level)})."
            ]
        out: list[str] = []
        whose = "your" if user_to_move else "your opponent's"
        top = analysis.lines[0].expected_score
        limit = LINE_PLIES if (not user_to_move or level >= DisclosureLevel.LINE) else 1
        ranked = analysis.lines
        best = ranked[:BEST_LINES]
        weaker = ranked[BEST_LINES : BEST_LINES + WEAKER_LINES]
        out.append(f"Engine lines for the position now ({whose} move; scores from your side):")
        for i, line in enumerate(best, start=1):
            out.append(
                f"  {i}. {san_line(board, line.moves, limit=limit, numbered=numbered)}  "
                f"{self._score(line.expected_score, to_move, user_colour)}"
            )
        if weaker:
            out.append("  weaker:")
            for i, line in enumerate(weaker, start=len(best) + 1):
                drop = top - line.expected_score
                out.append(
                    f"  {i}. {san_line(board, line.moves, limit=limit, numbered=numbered)}  "
                    f"{self._score(line.expected_score, to_move, user_colour)} "
                    f"({drop:.2f} behind the best)"
                )
        return out

    def _grounding_text(
        self,
        record: PlyRecord,
        board: chess.Board,
        user_colour: chess.Color,
        level: DisclosureLevel,
        *,
        with_lines: bool = True,
    ) -> list[str]:
        facts = record.facts
        if facts is None:
            return []
        user_to_move = board.turn == user_colour
        level = level if user_to_move else DisclosureLevel.LINE
        visible = visible_facts(facts, level)
        if not with_lines and record.analysis is not None and record.analysis.lines:
            # The candidates and the line are the lines block's first moves and first line;
            # an older ply's message does not write them twice.
            visible = {k: v for k, v in visible.items() if k not in ("candidates", "line")}
        out: list[str] = []
        shown: list[str] = []
        keys = (
            "phase",
            "decisive",
            "only_move",
            "nature",
            "theme",
            "plan",
            "region",
            "piece",
            "tactic",
            "threats",
            "ideas",
            "focus",
        )
        for key in keys:
            value = visible.get(key)
            if value in (None, "", [], False):
                continue
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            text = str(value)
            if not user_to_move:
                # The facts are the side to move's; said of the opponent, in their words.
                text = _theirs(text)
            shown.append(f"{key.replace('_', ' ')}: {text}")
        if record.analysis is not None and record.analysis.opening:
            shown.insert(0, f"opening: {record.analysis.opening}")
        if shown:
            owner = (
                "for the position you face"
                if user_to_move
                else "for the position your opponent faces (their plan, their ideas)"
            )
            out.append(f"Facts {owner}: " + "; ".join(shown) + ".")
        if user_to_move and level < DisclosureLevel.CANDIDATE:
            out.append(f"Disclosure now: {level.name.lower()} -- {self._level_words(level)}.")
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
        text = f"Your moves so far ({n} judged): {tally}."
        if lost:
            text += " Ground lost at: " + "; ".join(lost[-6:]) + "."
        return text

    # ---- helpers -----------------------------------------------------------------------

    def active_records(self) -> list[PlyRecord]:
        """The moves of the game as it stands: not taken back, since the last re-read."""
        return [r for r in self.records if not r.withdrawn and not r.superseded]

    def judged_records(self) -> list[PlyRecord]:
        """Every move that was played and judged, across re-reads, for the tally."""
        return [r for r in self.records if not r.withdrawn]

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
                        self._lines_text(record, after, user_colour, DisclosureLevel.LINE)
                    )
                    text = f"Engine note on the position after {name}:\n{shown}"
                self.entries.append(Entry("user", "note", text))
                return

    def _pgn_through(self, record: PlyRecord) -> str:
        board = chess.Board(self.start_fen)
        parts: list[str] = []
        for r in self.active_records():
            move = chess.Move.from_uci(r.move.uci)
            if move not in board.legal_moves:
                break
            if board.turn == chess.WHITE:
                parts.append(f"{board.fullmove_number}.")
            elif not parts:
                parts.append(f"{board.fullmove_number}...")
            parts.append(board.san(move))
            board.push(move)
            if r is record:
                break
        return " ".join(parts)

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
            DisclosureLevel.QUESTION: "an orienting question only; no theme, square, piece or move",
            DisclosureLevel.THEME: "the theme may be named; no square, piece or move",
            DisclosureLevel.REGION: "the theme and the region; no piece or move",
            DisclosureLevel.PIECE: "the theme, the region and the piece; no move",
            DisclosureLevel.CANDIDATE: "the candidate moves may be named",
            DisclosureLevel.LINE: "everything, the line included",
        }[level]
