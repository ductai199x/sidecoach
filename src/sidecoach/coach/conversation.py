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

import re
from dataclasses import dataclass, field

import chess

from sidecoach.analysis.engine import san_line
from sidecoach.coach.disclosure import PositionFacts, visible_facts
from sidecoach.coach.effects import effects_text
from sidecoach.coach.facts import material_text, motif_words, position_text, standing_words
from sidecoach.coach.inventory import inventory_text
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


# Whole words, in this order: the opponent's things become the player's first, then the
# side to move's become theirs. "you" is "them" at the end of a clause, "they" elsewhere.
_THEIRS = (
    (re.compile(r"\bYour opponent's\b"), "Your"),
    (re.compile(r"\bYour opponent\b"), "You"),
    (re.compile(r"\byour opponent's\b"), "your"),
    (re.compile(r"\byour opponent\b"), "you"),
    (re.compile(r"\bYou are\b"), "They are"),
    (re.compile(r"\byou are\b"), "they are"),
    (re.compile(r"\bYou have\b"), "They have"),
    (re.compile(r"\byou have\b"), "they have"),
    (re.compile(r"\bYours\b"), "Theirs"),
    (re.compile(r"\byours\b"), "theirs"),
    (re.compile(r"\bYour\b"), "Their"),
    (re.compile(r"\byour\b"), "their"),
    (re.compile(r"\bYou\b(?=\s*(?:[.,;:!?)]|$))"), "Them"),
    (re.compile(r"\byou\b(?=\s*(?:[.,;:!?)]|$))"), "them"),
    (re.compile(r"\bYou\b"), "They"),
    (re.compile(r"\byou\b"), "they"),
)


def _theirs(text: str) -> str:
    """A fact written for the side to move, reworded for the opponent: 'your king is
    still in the centre' becomes 'their king is still in the centre'. (A fact that began
    with "you still have ..." once kept its "you": the old table matched " you " with a
    space on each side, and a value's first word has none.)"""
    for pattern, new in _THEIRS:
        text = pattern.sub(new, text)
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
        text = f"You coach {side}."
        if control:
            text += f" Time control: {control}."
        if event.start_fen != chess.STARTING_FEN:
            text += "\nThe game is joined in progress; the position is:\n" + position_text(
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

    def user_said(self, text: str, verdicts: list[str] | None = None) -> None:
        body = f'"{text}"'
        if verdicts:
            body += " -- their claims, checked against the board: " + "; ".join(verdicts)
        self.entries.append(Entry("user", "ask", body))

    def coach_said(self, text: str) -> None:
        self.entries.append(Entry("assistant", "coach", text))

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
        for entry in self.entries:
            if entry.kind == "game":
                continue
            if entry.kind == "ply":
                assert entry.record is not None
                if entry is last:
                    sink = now
                    sink.append(
                        self.render_ply(entry.record, self.level, current=True, recent=True)
                    )
                    continue
                if entry.frozen is None:
                    entry.frozen = self.render_ply(
                        entry.record, DisclosureLevel.LINE, current=False, recent=False
                    )
                sink.append(entry.frozen)
            elif entry.kind == "ask":
                sink.append(f"    PLAYER said: {entry.text}")
            elif entry.kind == "coach":
                sink.append(f'    YOU (the coach) said: "{entry.text}"')
            else:  # takeback, resync, covered, ended, note
                sink.append(f"    EVENT: {entry.text}")
        sections = [f"# Setting\n{header}"]
        if game:
            sections.append("# Game so far\n" + "\n".join(game))
        if now:
            said = [x for x in now[1:]]
            sections.append(
                "# Position now\n"
                + now[0]
                + ("\n\n## Said here\n" + "\n".join(said) if said else "")
            )
        return "\n\n".join(x for x in sections if x)

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
        """The message for one ply. The current ply -- the position the user faces, or the
        one they just moved into -- carries everything: the moves so far, the position in
        full, the verdict, the standing, the lines, the facts, what stands on the board,
        the tally. The ply before it carries its position and verdict. Every older ply is
        one line: the move and its verdict -- the record, at a fortieth of the cost."""
        level = self.level if level is None else level
        move = record.move
        board_after = chess.Board(move.fen_after)
        board_before = chess.Board(move.fen_before)
        user_colour = chess.WHITE if self.user_side == Side.WHITE else chess.BLACK
        who = "the player" if move.by_user else "the opponent"
        if not recent:
            return f"{self._numbered(move)} ({who}): {self._verdict_line(record, board_before)}"
        lines: list[str] = []
        if current:
            # The legal moves name every move, the mating one with its mark: for the
            # position the player faces they are shown only once moves may be named.
            user_to_move = board_after.turn == user_colour
            lines.append(
                position_text(
                    board_after, legal=not (user_to_move and level < DisclosureLevel.CANDIDATE)
                )
            )
        else:
            lines.append(f"FEN: {board_after.fen()}")
        lines.append(self._turn_text(board_after).capitalize() + ".")
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
        if not current:
            return "\n".join(line for line in lines if line)
        # The current ply, in labelled sections.
        blocks: list[str] = [
            "## Board\n" + "\n".join(lines[:2]),
            "## Last move\n" + lines[2],
        ]
        engine = self._lines_text(record, board_after, user_colour, level, numbered=True)
        standing = next((x for x in engine if x.startswith("Standing: ")), None)
        if standing:
            blocks.append("## Standing\n" + standing.removeprefix("Standing: "))
        rest = [x for x in engine if not x.startswith("Standing: ")]
        if rest:
            head, body = rest[0], rest[1:]
            head = head.removeprefix("Engine lines for the position now").strip(": ")
            if head.startswith("withheld"):
                blocks.append("## Engine lines\n" + head)
            else:
                blocks.append(f"## Engine lines {head}".rstrip(":") + "\n" + "\n".join(body))
        for fact in self._grounding_text(record, board_after, user_colour, level, with_lines=True):
            if fact.startswith("Facts ("):
                owner, _, body = fact.removeprefix("Facts ").partition(": ")
                blocks.append(f"## Facts {owner}\n" + third_person(body))
            else:
                blocks.append("## Disclosure\n" + fact)
        user_to_move = board_after.turn == user_colour
        if not (user_to_move and level < DisclosureLevel.PIECE):
            inventory = inventory_text(board_after, user_colour)
            head, _, body = inventory.partition(": ")
            blocks.append(
                "## On the board (complete lists)\n" + (body.replace(" | ", "\n") if body else head)
            )
        tally = self._tally_text(record)
        if tally:
            blocks.append("## The player's record\n" + tally)
        return "\n\n".join(blocks)

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
        material = material_text(chess.Board(move.fen_after), user_colour)
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
            # The engine's line, when its first move is not the one played. (Comparing
            # the line's last move to the one played dropped the line whenever a four-ply
            # line happened to end on the same move: castling, most often.)
            better = san_line(board_before, assessment.best, limit=4)
            if better:
                parts.append(f"the engine preferred {better}")
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
        tags = [t for t in assessment.tags if t.name not in ("quietMove", "capture", "check")]
        if tags:
            named = ", ".join(
                motif_words(t.name) + (f" ({'/'.join(t.squares)})" if t.squares else "")
                for t in tags[:4]
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
            # said is in the grounding facts below. Who stands better is never withheld.
            return [
                self._standing_text(analysis.lines[0], to_move, user_colour),
                f"Engine lines for the position now: withheld (disclosure level "
                f"{level.name.lower()}: {self._level_words(level)}).",
            ]
        out: list[str] = []
        whose = "the player's" if user_to_move else "the opponent's"
        top = analysis.lines[0].expected_score
        limit = LINE_PLIES if (not user_to_move or level >= DisclosureLevel.LINE) else 1
        ranked = analysis.lines
        best = ranked[:BEST_LINES]
        weaker = ranked[BEST_LINES : BEST_LINES + WEAKER_LINES]
        out.append(self._standing_text(analysis.lines[0], to_move, user_colour))
        out.append(
            f"Engine lines for the position now ({whose} move; scores from the player's side):"
        )
        for i, line in enumerate(best, start=1):
            out.append(
                f"  {i}. {san_line(board, line.moves, limit=limit, numbered=numbered)}  "
                f"{self._score(line.expected_score, to_move, user_colour)}"
                + self._does(board, line.moves)
                + (self._leads(board, line.moves[:limit], user_colour) if numbered else "")
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

    def _standing_text(self, top: Line, to_move: chess.Color, user_colour: chess.Color) -> str:
        """Who stands better, in words, with the engine's number named by side: "Standing:
        you are better (Stockfish: White +1.2 pawns; your winning chances about 66%)". A
        bare 0-1 number was read as a pawn score, or not at all; the words are what a
        coach says, and the side named leaves no sign to guess."""
        score = top.expected_score if to_move == user_colour else 1.0 - top.expected_score
        text = f"Standing: {standing_words(score, third_person=True)} (Stockfish: "
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
            if key == "threats" and isinstance(value, list):
                # A threat is the opponent's move if the side to move passed; what it does
                # is computed on that position, so the coach can say it in words.
                value = [self._threat_words(board, str(v)) for v in value]
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            text = str(value)
            if not user_to_move:
                # The facts are the side to move's; said of the opponent, in their words.
                text = _theirs(text)
            shown.append(f"{key.replace('_', ' ')}: {text}")
        if record.analysis is not None and record.analysis.opening:
            previous = self._record_before(record)
            before = previous.analysis.opening if previous and previous.analysis else None
            if before != record.analysis.opening:
                shown.insert(0, f"opening: {record.analysis.opening}")
        if shown:
            owner = (
                "the player to move"
                if user_to_move
                else "the opponent to move: the opponent's plan, the opponent's ideas"
            )
            out.append(f"Facts ({owner}): " + "; ".join(shown) + ".")
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
        text = f"The player's moves so far ({n} judged): {tally}."
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
