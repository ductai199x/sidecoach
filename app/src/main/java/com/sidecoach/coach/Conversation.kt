package com.sidecoach.coach

import com.sidecoach.analysis.sanLine
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.events.Analysis
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.Line
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.Resync
import com.sidecoach.events.Severity
import com.sidecoach.events.Takeback
import com.sidecoach.llm.ChatMessage
import com.sidecoach.vision.Pieces
import java.math.BigDecimal
import java.math.RoundingMode
import java.util.Locale

/**
 * The game as the harness the model reads: one user message, in sections.
 * `coach/conversation.py`.
 *
 * `# Setting` says whom the coach coaches. `# Game so far` is the record: one line per move
 * with the engine's verdict on it -- the label, the cost, the move the engine preferred and
 * where its line leads -- and under each move, indented, what was said after it (the
 * player's words, the coach's own replies) and any event (a takeback, a re-read of the
 * board, a dialog, the end of the game), the way a game record carries its annotations.
 * Once the model has seen a move's line it is frozen and never edited; facts that arrive
 * later go in a note of their own. `# Position now` is the last ply in full at the
 * disclosure level in force for the position the player faces: the board, the last move,
 * the standing, the engine's lines, the facts in the third person per side (below MOVE on the
 * player's turn, those past the rung set apart, for the model to teach from), the disclosure
 * line -- the model sees everything and the reply is checked for the answer ([leakCheck]) --
 * the complete inventory of what stands on the board, the player's record so far, and what was
 * already said here. The coach appends its task and answers as the assistant.
 *
 * Two levers keep a long game's harness small. The dialogue older than the last N plies can
 * be folded by a summariser into one "Earlier dialogue, summarised" line ([foldable],
 * [fold]; the coach runs it in the background). With [Conversation.compactRecord], the
 * coach's routine nods after sound moves are written out only while recent, and sound moves
 * with nothing said after them run together as a move list.
 *
 * The coach owns one of these per game and asks it for the messages of every request.
 */

/**
 * What the first hint may say, as the disclosure line and the directive state it: the key fact
 * is named on the board (Tai, 2026-09-24), so "no square" -- the rule before the first hint was
 * a lesson -- told the model to say less than the template.
 */
const val IDEA_LEVEL_WORDS =
    "the idea: what is at stake and the key fact on the board -- the pieces, squares and " +
        "their moves the facts name, the target's square too, even where the move goes -- with " +
        "the principle behind it; never the player's move itself, or which of their own pieces " +
        "makes it"

/**
 * Below the move level the model sees everything -- the engine lines, the legal moves, the facts
 * past the rung -- and says only what the rung allows; the reply is checked for the answer before
 * it is spoken ([leakCheck]).
 */
const val BEYOND_TITLE =
    "Past this level (for you to teach from; do not say the move, its square or the piece " +
        "that makes it)"
const val SEE_ALL_WORDS =
    "You see everything, the answer included, so the hint can be precise and true: say what " +
        "this level allows, built from what you know."

/** One move as it was seen, with the engine's word on it as it arrives. */
class PlyRecord(val move: MovePlayed) {
    var assessment: MoveAssessment? = null

    /**
     * The analysis of the position the move led to (the side to move's lines), and the
     * facts built from it for the user's side.
     */
    var analysis: Analysis? = null
    var facts: PositionFacts? = null
    var whiteMs: Int? = null
    var blackMs: Int? = null

    /** Taken back: out of the move list, still in history. */
    var withdrawn: Boolean = false

    /** Before a re-read of the board: the move list restarted. */
    var superseded: Boolean = false
}

private val PLACED_PIECE = Regex("\\b(king|queen|rook|bishop|knight|pawn) on ([a-h][1-8])\\b")
private val PLACED_PIECE_TYPES = mapOf(
    "king" to KING,
    "queen" to QUEEN,
    "rook" to ROOK,
    "bishop" to BISHOP,
    "knight" to KNIGHT,
    "pawn" to PAWN,
)

/**
 * Whether [text] names a piece on a square that no longer holds that piece on [board].
 * `conversation._gone`.
 */
private fun gone(text: String, board: Board): Boolean {
    for (match in PLACED_PIECE.findAll(text.lowercase())) {
        val (word, square) = match.destructured
        if (board.pieceTypeAt(parseSquare(square)) != PLACED_PIECE_TYPES.getValue(word)) return true
    }
    return false
}

/**
 * The threat the better move's reason says it stops, when the move played stopped it too: on
 * the board right after it, the threat can no longer be played. `conversation._also_stopped`.
 */
internal fun alsoStopped(facts: PositionFacts, played: Board?): String? {
    if (played == null || facts.threats.isEmpty() || facts.why.isNullOrEmpty()) return null
    val threat = facts.threats[0]
    if ("stops $threat" !in facts.why) return null
    // `parse_san` raising: illegal, ambiguous or no move at all.
    if (played.parseSanFully(threat).second != null) return threat
    return null
}

/**
 * One message of the conversation. A ply entry renders itself from its record until it is
 * frozen; every other kind is plain text.
 *
 * Identity is what counts -- the render marks stale nods and the last ply by reference --
 * so this is a plain class, not a data class.
 */
class Entry(
    val role: String,
    /** "game", "ply", "takeback", "resync", "covered", "ended", "ask", "coach", "note". */
    val kind: String,
    val text: String = "",
    val record: PlyRecord? = null,
    /** A routine word after a sound move (the POST_MOVE nod). */
    val nod: Boolean = false,
) {
    var frozen: String? = null

    /** Dialogue folded into the summary: no longer written out. */
    var folded: Boolean = false
}

class Conversation {
    var userSide: Pieces.Side = Pieces.Side.WHITE
    var levelText: String = "around 1200"
    val entries: MutableList<Entry> = ArrayList()
    val records: MutableList<PlyRecord> = ArrayList()
    var startFen: String = STARTING_FEN

    /**
     * The disclosure level to render the position the user faces with, set by the coach
     * before each request from the current position's disclosure state.
     */
    var level: DisclosureLevel = DisclosureLevel.IDEA

    /** The earlier dialogue, summarised; empty until the first fold. */
    var summary: String = ""

    /**
     * The record compacted: the coach's routine nods after sound moves are written out only
     * for the last [NOD_PLIES] plies (the verdict is on the move's line), and sound moves
     * with nothing said after them run together as a move list.
     */
    var compactRecord: Boolean = false

    private var drained: Int = 0

    private val userColour: Boolean
        get() = if (userSide == Pieces.Side.WHITE) WHITE else BLACK

    // ---- the game's events, in the order they happen -----------------------------------

    /**
     * The setting of a new game. [profile] is what the player model says about this player
     * from earlier games -- the biggest cost, the last session -- in sentences the profile
     * stage computed, so the coach has continuity without inventing any.
     */
    fun newGame(
        event: GameStarted,
        opponent: String = "a bot",
        control: String = "",
        profile: List<String> = emptyList(),
    ) {
        userSide = event.userSide
        startFen = event.startFen
        entries.clear()
        records.clear()
        drained = 0
        val side = if (event.userSide == Pieces.Side.WHITE) "White" else "Black"
        var text = "You coach $side."
        if (control.isNotEmpty()) text += " Time control: $control."
        if (event.kind == "puzzle") {
            text += "\nThis is a puzzle the player is solving on the panel's board, $side to " +
                "move and to find the best line. Say nothing unasked; when they ask for a " +
                "hint, give one at the level allowed -- the move itself only at the third hint " +
                "or when asked for it:\n" + positionText(Board(event.startFen))
        } else if (event.kind != "game") {
            text += "\nThis is a position, not a game from the start: a puzzle, an analysis " +
                "board, or a game joined from the middle. There is no earlier history to " +
                "speak of. The player is the side at the bottom and, on a puzzle, to move; " +
                "coach the position as it stands:\n" + positionText(Board(event.startFen))
        } else if (event.startFen != STARTING_FEN) {
            text += "\nThe game is joined in progress; the position is:\n" +
                positionText(Board(event.startFen))
        }
        if (profile.isNotEmpty()) {
            text += "\nWhat earlier games say about this player (computed, not opinion): " +
                profile.joinToString(" ")
        }
        entries.add(Entry("user", "game", text))
    }

    fun move(event: MovePlayed): PlyRecord {
        val record = PlyRecord(event)
        records.add(record)
        entries.add(Entry("user", "ply", record = record))
        return record
    }

    fun assess(event: MoveAssessment): PlyRecord? {
        val record = recordAt(event.ply)
        if (record == null || record.move.uci != event.uci) return null
        record.assessment = event
        noteIfFrozen(record, "assessment")
        return record
    }

    fun analysed(event: Analysis, facts: PositionFacts): PlyRecord? {
        val record = recordAt(event.ply)
        if (record == null ||
            record.move.fenAfter.split(" ").take(3) != event.fen.split(" ").take(3)
        ) {
            return null
        }
        record.analysis = event
        record.facts = facts
        noteIfFrozen(record, "analysis")
        return record
    }

    fun clock(whiteMs: Int?, blackMs: Int?) {
        if (records.isNotEmpty()) {
            val last = records.last()
            if (last.whiteMs == null && last.blackMs == null) {
                last.whiteMs = whiteMs
                last.blackMs = blackMs
            }
        }
    }

    /** The last `plies` moves are withdrawn; they stay in history, marked so. */
    fun takeback(event: Takeback) {
        val active = activeRecords()
        val withdrawn = if (event.plies != 0) active.takeLast(event.plies) else emptyList()
        for (record in withdrawn) record.withdrawn = true
        val board = Board(event.fen)
        val names = withdrawn.joinToString(", ") { numbered(it.move) }.ifEmpty { "the last move" }
        val text = "Takeback: $names withdrawn. The game is back at the position after " +
            "${lastActiveName()}, ${turnText(board)}:\n${positionText(board)}\n" +
            "Play continues from here; the next message is the replacement move."
        entries.add(Entry("user", "takeback", text))
    }

    /** The board was re-read: the position jumped and some moves may have gone unseen. */
    fun resync(event: Resync, board: Board) {
        for (record in records) record.superseded = true // the move list restarts here
        val text = "The board was re-read (${event.reason}). Moves may have gone unseen. The " +
            "position is now, ${turnText(board)}:\n${positionText(board)}\n" +
            "The move numbering restarts from this position."
        entries.add(Entry("user", "resync", text))
        startFen = board.fen()
    }

    fun covered() {
        entries.add(
            Entry(
                "user",
                "covered",
                "The board is covered by a dialog. The game may be over or interrupted; wait.",
            )
        )
    }

    fun ended(event: GameEnded) {
        val how = if (!event.reason.isNullOrEmpty()) " (${event.reason})" else ""
        entries.add(Entry("user", "ended", "Game over: ${event.result}$how."))
    }

    /** Something that happened, in a sentence, for the record. */
    fun note(text: String) {
        entries.add(Entry("user", "note", text))
    }

    /**
     * The review of the game once it is over: the tally, the accuracy, where it turned and
     * why, from the engine's verdicts.
     */
    fun review(lines: List<String>) {
        entries.add(Entry("user", "review", "The review of the game: " + lines.joinToString(" ")))
    }

    fun userSaid(text: String, verdicts: List<String>? = null) {
        var body = "\"$text\""
        if (!verdicts.isNullOrEmpty()) {
            body += " -- their claims, checked against the board: " + verdicts.joinToString("; ")
        }
        entries.add(Entry("user", "ask", body))
    }

    fun coachSaid(text: String, nod: Boolean = false) {
        entries.add(Entry("assistant", "coach", text, nod = nod))
    }

    // ---- rendering ---------------------------------------------------------------------

    /**
     * The harness output: one user message, in sections. The coach appends its task to it
     * and answers as the assistant.
     */
    fun messages(): List<ChatMessage> = listOf(ChatMessage("user", render()))

    /**
     * The sections: the setting; the game so far -- one line per move with its verdict, and
     * under each move, indented, what was said after it (the player's words, your reply)
     * and any event, the way a game record carries its annotations; the position now in
     * full (the last ply, at the disclosure level in force), with what was said after it
     * under it. Nothing is written twice: the move list is the record.
     */
    fun render(): String {
        val header = entries.firstOrNull { it.kind == "game" }?.text ?: ""
        val plyEntries = entries.filter { it.kind == "ply" }
        val last = plyEntries.lastOrNull()
        var game = ArrayList<String>()
        val now = ArrayList<String>()
        var sink = game
        if (summary.isNotEmpty()) game.add("Earlier dialogue, summarised: $summary")
        val staleNods = if (compactRecord) staleNods() else emptySet()
        for (entry in entries) {
            if (entry.kind == "game" || entry.folded || entry in staleNods) continue
            if (entry.kind == "ply") {
                val record = entry.record!!
                if (entry === last) {
                    sink = now
                    sink.add(renderPly(record, level))
                    continue
                }
                if (entry.frozen == null) entry.frozen = renderPly(record, full = false)
                sink.add(entry.frozen!!)
            } else if (entry.kind == "ask") {
                sink.add("    PLAYER said: ${entry.text}")
            } else if (entry.kind == "coach") {
                sink.add("    YOU (the coach) said: \"${entry.text}\"")
            } else { // takeback, resync, covered, ended, note
                sink.add("    EVENT: ${entry.text}")
            }
        }
        val sections = ArrayList<String>()
        sections.add("# Setting\n$header")
        if (game.isNotEmpty()) {
            if (compactRecord) game = ArrayList(compact(game))
            sections.add("# Game so far\n" + game.joinToString("\n"))
        }
        if (now.isNotEmpty()) {
            val said = now.drop(1)
            sections.add(
                "# Position now\n" + now[0] +
                    (if (said.isNotEmpty()) "\n\n## Said here\n" + said.joinToString("\n") else "")
            )
        }
        return sections.filter { it.isNotEmpty() }.joinToString("\n\n")
    }

    /** The nods older than the last [NOD_PLIES] plies: left out of the record. */
    private fun staleNods(): Set<Entry> {
        val plyEntries = entries.filter { it.kind == "ply" }
        if (plyEntries.size <= NOD_PLIES) return emptySet()
        val cutoff = plyEntries[plyEntries.size - NOD_PLIES]
        val stale = java.util.Collections.newSetFromMap(java.util.IdentityHashMap<Entry, Boolean>())
        for (entry in entries) {
            if (entry === cutoff) break
            if (entry.kind == "coach" && entry.nod) stale.add(entry)
        }
        return stale
    }

    /**
     * The record's lines with the quiet plies run together: a ply line followed by no
     * annotation, whose verdict is a bare label, joins a move list; every other line
     * stands. "12. Bxf6 (the player): best." x 40 was most of the record.
     */
    private fun compact(lines: List<String>): List<String> {
        val out = ArrayList<String>()
        val run = ArrayList<String>()

        fun flush() {
            if (run.isNotEmpty()) {
                out.add("Moves: " + run.joinToString(" "))
                run.clear()
            }
        }

        val n = lines.size
        for (i in 0 until n) {
            val line = lines[i]
            val isPly = line.isNotEmpty() && line[0].isDigit()
            val annotated = i + 1 < n && lines[i + 1].startsWith("    ")
            val match = if (isPly && !annotated) BARE_VERDICT.find(line) else null
            if (match != null) {
                val numbered = match.groupValues[1]
                val san = numbered.split(" ", limit = 2)[1]
                if ("..." in numbered && run.isNotEmpty() && !run.last().endsWith(".")) {
                    run.add(san) // Black's reply right after White's move: bare
                } else {
                    run.add(numbered)
                }
                continue
            }
            flush()
            out.add(line)
        }
        flush()
        if (out.any { it.startsWith("Moves: ") }) {
            out.add(
                0,
                "(A move in a 'Moves:' list was sound -- best, excellent, great, good or book " +
                    "-- and nothing was said after it.)",
            )
        }
        return out
    }

    // ---- dialogue compaction -------------------------------------------------------------

    /**
     * The dialogue entries older than the last [keepPlies] plies that are not yet in the
     * summary, each with the move it followed.
     */
    fun foldable(keepPlies: Int): List<Pair<String, Entry>> {
        val plyEntries = entries.filter { it.kind == "ply" }
        if (plyEntries.size <= keepPlies) return emptyList()
        val cutoff = plyEntries[plyEntries.size - keepPlies]
        val out = ArrayList<Pair<String, Entry>>()
        var after = "the start"
        for (entry in entries) {
            if (entry === cutoff) break
            if (entry.kind == "ply" && entry.record != null) {
                after = numbered(entry.record.move)
            } else if ((entry.kind == "ask" || entry.kind == "coach") && !entry.folded) {
                if (compactRecord && entry.nod) {
                    entry.folded = true // a nod is not summarised: the verdict is on its line
                    continue
                }
                out.add(after to entry)
            }
        }
        return out
    }

    /**
     * Replace [toFold] in the record by [text] (the whole earlier dialogue, as the
     * summariser rewrote it).
     */
    fun fold(toFold: List<Entry>, text: String) {
        for (entry in toFold) entry.folded = true
        summary = text.trim()
    }

    /** Entries added since the last call, for the panel's conversation view. */
    fun drainNew(): List<Entry> {
        val new = entries.drop(drained)
        drained = entries.size
        return new
    }

    /**
     * One ply of the harness. In [full], the position now: the board, the last move with
     * its verdict, the standing, the lines, the facts, what stands on the board, the
     * player's record, in titled sections at [wanted]. Otherwise the record's one line for
     * it: the move and its verdict.
     */
    fun renderPly(
        record: PlyRecord,
        wanted: DisclosureLevel? = null,
        full: Boolean = true,
    ): String {
        val level = wanted ?: this.level
        val move = record.move
        val boardAfter = Board(move.fenAfter)
        val boardBefore = Board(move.fenBefore)
        val colour = userColour
        val who = if (move.byUser) "the player" else "the opponent"
        if (!full) {
            return "${numbered(move)} ($who): ${verdictLine(record, boardBefore)}"
        }
        // The legal moves name every move, the mating one with its mark: shown at every level,
        // the answer among them, since the model sees everything and what it says is checked
        // (Tai, 2026-09-26: "The model need to see everything including engine lines at hint").
        val lines = ArrayList<String>()
        lines.add(positionText(boardAfter, legal = true))
        val turn = turnText(boardAfter)
        // `str.capitalize()`: the first letter up, the rest down.
        lines.add(turn.take(1).uppercase() + turn.drop(1).lowercase() + ".")
        var clock = ""
        if (record.whiteMs != null || record.blackMs != null) {
            clock = " Clock: white ${clockText(record.whiteMs)}, " +
                "black ${clockText(record.blackMs)}."
        }
        val took = if (move.thinkMs != null && clock.isNotEmpty()) {
            " (" + format1(move.thinkMs / 1000.0) + " s)"
        } else {
            ""
        }
        lines.add(
            "${numbered(move)} by $who$took: " +
                Person.thirdPerson(verdictText(record, boardBefore, colour)) + clock
        )
        val sections = ArrayList<Pair<String, String>>()
        sections.add("Board" to lines.take(2).joinToString("\n"))
        sections.add("Last move" to lines[2])
        sections.addAll(positionSections(record, boardAfter, colour, level))
        return sections.joinToString("\n\n") { (title, body) -> "## $title\n$body" }
    }

    /**
     * The position the side to move faces, as (title, body) sections: the standing, the
     * engine lines, the facts, the disclosure level, what stands on the board, the player's
     * record.
     */
    private fun positionSections(
        record: PlyRecord,
        board: Board,
        colour: Boolean,
        level: DisclosureLevel,
    ): List<Pair<String, String>> {
        val userToMove = board.turn == colour
        val out = ArrayList<Pair<String, String>>()
        val analysis = record.analysis
        if (analysis != null && analysis.lines.isNotEmpty()) {
            out.add("Standing" to standingText(analysis.lines[0], board.turn, colour))
        }
        val (head, body) = linesBlock(record, board, colour, level)
        out.add("Engine lines $head".trimEnd() to body.joinToString("\n"))
        val facts = factsLines(record, board, colour, level)
        // Over the board, the game is over: a first hint for the mated position ("their queen on
        // b7 gives check") is noise beside the review.
        if (facts.isNotEmpty() && !board.isGameOver()) {
            out.add("Facts (${factsOwner(userToMove)})" to facts.joinToString("\n"))
        }
        val hiding = userToMove && level.value < DisclosureLevel.MOVE.value && !board.isGameOver()
        if (hiding) {
            // The facts past the rung -- the piece, the target, the move and why it works -- are
            // the model's to teach from, set apart so the rung's own stay plain (a hint built on
            // the heuristic sheet alone could only reword it).
            val beyond = beyondLines(record, board, level)
            if (beyond.isNotEmpty()) out.add(BEYOND_TITLE to beyond.joinToString("\n"))
            out.add(
                "Disclosure" to "Level ${level.name.lowercase()}: ${levelWords(level)}. $SEE_ALL_WORDS"
            )
        }
        // The inventory lists the checks and captures on offer, the answer among them.
        out.add("On the board (complete lists)" to inventoryLines(board, colour).joinToString("\n"))
        val tally = tallyText(record)
        if (tally.isNotEmpty()) out.add("The player's record" to tally)
        return out
    }

    /**
     * The record's line for an older ply: label, cost, the engine's preferred move and what
     * it would have done.
     */
    private fun verdictLine(record: PlyRecord, boardBefore: Board): String {
        val a = record.assessment ?: return "verdict not in."
        val label = a.label?.name?.lowercase() ?: a.severity.name.lowercase()
        if (a.expectedLoss < 0.005) return "$label."
        var text = "$label, cost " + format2(a.expectedLoss)
        if (a.best.isNotEmpty() && a.best[0] != record.move.uci) {
            val better = Move.fromUciOrNull(a.best[0])
            if (better != null) {
                text += "; engine preferred ${boardBefore.san(better)}"
                val would = effectsText(boardBefore, better, hazards = false)
                if (would.isNotEmpty()) text += " ($would)"
            }
        }
        return "$text."
    }

    // ---- pieces of a ply message -------------------------------------------------------

    private fun verdictText(record: PlyRecord, boardBefore: Board, colour: Boolean): String {
        val move = record.move
        val assessment = record.assessment
        val boardAfter = Board(move.fenAfter)
        val (now, san, then) = materialWords(record, boardBefore, boardAfter, colour)
        var material = now
        if (san != null) {
            // A capture the engine expects to be answered: "up a knight" after Nxe6 was
            // read as a knight won, and the coach said so, with dxe6 to come.
            material += " before the expected recapture $san, then $then"
        }
        if (assessment == null) return "verdict not in yet. Material: $material."
        val label = assessment.label?.name?.lowercase() ?: assessment.severity.name.lowercase()
        val loss = assessment.expectedLoss
        val parts = ArrayList<String>()
        parts.add(label)
        if (loss >= 0.005) {
            parts.add("it cost " + format2(loss) + " of expected score")
        } else {
            parts.add("it cost nothing")
        }
        if (assessment.best.isNotEmpty() && loss >= 0.005 && assessment.best[0] != move.uci) {
            // The engine's line, when its first move is not the one played, and where it
            // leads: the reason the move was better, computed, so a "why was that bad?" has
            // an answer when the better move's own effects are nothing to say. (Comparing
            // the line's last move to the one played dropped the line whenever a four-ply
            // line happened to end on the same move: castling, most often.)
            val better = sanLine(boardBefore, assessment.best, limit = 4)
            if (better.isNotEmpty()) {
                val leads = lineOutcome(boardBefore, assessment.best.take(LINE_PLIES), colour)
                val notes = lineNotes(boardBefore, assessment.best, limit = 4, player = colour)
                parts.add(
                    "the engine preferred $better" +
                        (if (leads.isNotEmpty()) " -> $leads" else "") +
                        (if (notes.isNotEmpty()) " (move by move: $notes)" else "")
                )
            }
        }
        var text = parts.joinToString("; ") + ". Material: $material."
        // What the move did on the board, and what the better move would have done: the
        // mechanism, computed, so the coach has a true reason to give instead of one it
        // makes up.
        val played = boardBefore.parseSan(move.san)
        // A move the engine judged sound has its hazards resolved by the tactics.
        val did = if (played != null) {
            effectsText(boardBefore, played, hazards = assessment.severity != Severity.NONE)
        } else {
            ""
        }
        if (did.isNotEmpty()) text += " What ${move.san} does: $did."
        if (assessment.best.isNotEmpty() && loss >= 0.005 && assessment.best[0] != move.uci) {
            val betterMove = Move.fromUciOrNull(assessment.best[0])
            val would = if (betterMove != null) {
                effectsText(boardBefore, betterMove, hazards = false)
            } else {
                ""
            }
            val betterSan = if (betterMove != null) boardBefore.san(betterMove) else ""
            if (would.isNotEmpty()) text += " What $betterSan would do: $would."
        }
        val tags = assessment.tags.filter { it.name in MOVE_MOTIFS }
        if (tags.isNotEmpty()) {
            // By name only: the squares a tag carries are the tagger's evidence, and named
            // in the harness they were narrated as "the f7/f6/f4 piece sequence". The words
            // are the mover's ("your opponent's exposed king"), so they are set in the third
            // person for the side that moved.
            val mover = if (move.byUser) "player" else "opponent"
            val named = tags.take(4)
                .map { Person.thirdPerson(motifWords(it.name), side = mover) }
                .toCollection(LinkedHashSet())
                .joinToString(", ")
            text += " Motifs in the move: $named."
        }
        return text
    }

    /**
     * The material after the move, to the player; and when the engine expects the capture taken
     * back, the recapture (SAN) and the material after it. `conversation._material_words`.
     */
    private fun materialWords(
        record: PlyRecord,
        boardBefore: Board,
        boardAfter: Board,
        colour: Boolean,
    ): Triple<String, String?, String?> {
        val material = materialText(boardAfter, colour)
        val recapture = expectedRecapture(record, boardBefore, boardAfter) ?: return Triple(material, null, null)
        val (san, then) = recapture
        return Triple(material, san, materialText(then, colour))
    }

    /**
     * The material as the game stands, as the record's last move gives it: now, and the expected
     * recapture with the material after it ([materialWords]). Null before the first move.
     */
    fun materialNow(): Triple<String, String?, String?>? {
        val active = activeRecords()
        if (active.isEmpty()) return null
        val record = active.last()
        return materialWords(record, Board(record.move.fenBefore), Board(record.move.fenAfter), userColour)
    }

    /**
     * When the move was a capture and the engine's reply takes back on the same square:
     * that reply in SAN, and the board after it.
     */
    internal fun expectedRecapture(
        record: PlyRecord,
        boardBefore: Board,
        boardAfter: Board,
    ): Pair<String, Board>? {
        val analysis = record.analysis
        val best = record.assessment?.best ?: emptyList()
        val expected = if (analysis != null && analysis.lines.isNotEmpty() && analysis.lines[0].moves.isNotEmpty()) {
            analysis.lines[0].moves[0]
        } else if (best.size > 1 && best[0] == record.move.uci) {
            // The position's own analysis not in yet: the engine's line for the move played
            // names the reply. Without it "up a queen for a pawn" was said of a queen about to
            // be taken back, and the model called the queen trade a queen won (the dev audit,
            // 2026-09-26).
            best[1]
        } else {
            return null
        }
        val played = Move.fromUciOrNull(record.move.uci) ?: return null
        val reply = Move.fromUciOrNull(expected) ?: return null
        if (!boardBefore.isCapture(played) || !boardAfter.legalMoves().contains(reply)) return null
        if (reply.to != played.to || !boardAfter.isCapture(reply)) return null
        val then = boardAfter.copy(withStack = false)
        then.push(reply)
        return boardAfter.san(reply) to then
    }

    /**
     * The engine lines section: its title's tail (whose move, whose scores) and its lines.
     * Withheld below the candidate level on the player's turn -- what may be said is in the
     * facts -- and the standing is a section of its own.
     */
    private fun linesBlock(
        record: PlyRecord,
        board: Board,
        colour: Boolean,
        level: DisclosureLevel,
    ): Pair<String, List<String>> {
        val analysis = record.analysis
        if (analysis == null || analysis.lines.isEmpty()) return "" to listOf("not in yet.")
        val toMove = board.turn
        val userToMove = toMove == colour
        val whose = if (userToMove) "the player's" else "the opponent's"
        val answer = if (userToMove && level.value < DisclosureLevel.MOVE.value) {
            "; the best first move is the answer they are finding: teach from it, never name it"
        } else {
            ""
        }
        val top = analysis.lines[0].expectedScore
        val limit = LINE_PLIES
        val best = analysis.lines.take(BEST_LINES)
        val weaker = analysis.lines.drop(BEST_LINES).take(WEAKER_LINES)
        val out = ArrayList<String>()
        for ((index, line) in best.withIndex()) {
            out.add(
                "  ${index + 1}. ${sanLine(board, line.moves, limit = limit)}  " +
                    score(line.expectedScore, toMove, colour) +
                    does(board, line.moves) +
                    leads(board, line.moves.take(limit), colour)
            )
            if (index == 0) {
                // The best line move by move, computed: what a line may be explained by.
                val notes = lineNotes(board, line.moves, limit = limit, player = colour)
                if (notes.isNotEmpty()) out.add("     move by move: $notes")
            }
        }
        if (weaker.isNotEmpty()) {
            // The scores are from the player's side, so on the opponent's turn a weaker line
            // for them shows a higher number: the title says whose.
            out.add("  weaker for ${whose.removeSuffix("'s")}:")
            for ((index, line) in weaker.withIndex()) {
                val drop = top - line.expectedScore
                val gap = if (drop >= 0.005) {
                    format2(drop) + " behind the best"
                } else {
                    "level with the best"
                }
                out.add(
                    "  ${best.size + index + 1}. ${sanLine(board, line.moves, limit = limit)}  " +
                        score(line.expectedScore, toMove, colour) + " ($gap)"
                )
            }
        }
        return "($whose move; scores from the player's side$answer)" to out
    }

    /**
     * Who stands better, in words, with the engine's number named by side: "The player is
     * better (Stockfish: White ahead by about 1.2 pawns)." A bare 0-1 number was read as a
     * pawn score, or not at all; the words are what a coach says, and the side named leaves
     * no sign to guess.
     */
    private fun standingText(top: Line, toMove: Boolean, colour: Boolean): String {
        val score = if (toMove == colour) top.expectedScore else 1.0 - top.expectedScore
        val words = standingWords(score, thirdPerson = true)
        var text = words.take(1).uppercase() + words.drop(1) + " (by the numbers: "
        val mate = top.mate
        val cp = top.scoreCp
        if (mate != null) {
            val n = kotlin.math.abs(mate)
            val winner = if (mate > 0) toMove else !toMove
            text += "mate in $n for " + (if (winner == WHITE) "White" else "Black")
        } else if (cp != null) {
            val cpWhite = if (toMove == WHITE) cp else -cp
            if (kotlin.math.abs(cpWhite) < 15) {
                text += "level, 0.0 pawns"
            } else {
                val ahead = if (cpWhite > 0) "White" else "Black"
                text += "$ahead ahead by about " + format1(kotlin.math.abs(cpWhite) / 100.0) +
                    " pawns"
            }
        } else {
            text += "no number"
        }
        return "$text)."
    }


    /** Where the line leads, for the lines block: "-> you win a pawn; the e-file opens". */
    private fun leads(board: Board, moves: List<String>, colour: Boolean): String {
        val text = lineOutcome(board, moves, colour)
        return if (text.isNotEmpty()) "  -> $text" else ""
    }

    /** What a line's first move does on the board, for the lines block. */
    private fun does(board: Board, moves: List<String>): String {
        if (moves.isEmpty()) return ""
        val move = Move.fromUciOrNull(moves[0]) ?: return ""
        val text = effectsText(board, move, hazards = false)
        return if (text.isNotEmpty()) "  ($text)" else ""
    }

    /**
     * The facts of the position, one per line, in the third person for the side to move:
     * the opponent's facts are the opponent's plan and ideas, and say so.
     */
    private fun factsLines(
        record: PlyRecord,
        board: Board,
        colour: Boolean,
        wanted: DisclosureLevel,
    ): List<String> {
        val facts = record.facts ?: return emptyList()
        val userToMove = board.turn == colour
        val level = if (userToMove) wanted else DisclosureLevel.LINE
        val visible = visibleFacts(facts, level)
        if (!userToMove) {
            // The hints are for the side to move: on the opponent's turn the first hint and the
            // two questions are the opponent's, and labelled "to the player" the model asked the
            // player the opponent's question (parity pass 2, 2026-09-25).
            for (key in listOf("situation", "question", "nudge")) visible.remove(key)
        }
        val side = if (userToMove) "player" else "opponent"
        val out = ArrayList<String>()
        val opening = record.analysis?.opening
        if (!opening.isNullOrEmpty()) {
            val before = recordBefore(record)?.analysis?.opening
            if (before != opening) out.add("opening: $opening")
        }
        out.addAll(factLines(visible, board, side))
        return out
    }

    /**
     * The facts past [level] on the player's turn, one per line: the piece, the target, the move
     * and why it works -- the model's to teach from, not to say. `conversation._beyond_lines`.
     */
    private fun beyondLines(record: PlyRecord, board: Board, level: DisclosureLevel): List<String> {
        val facts = record.facts ?: return emptyList()
        val shown = visibleFacts(facts, level)
        val every = visibleFacts(facts, DisclosureLevel.LINE)
        val beyond = LinkedHashMap<String, Any?>()
        for ((key, value) in every) if (key !in shown) beyond[key] = value
        return factLines(beyond, board, "player")
    }

    /**
     * The facts of the position the player faced before their move, all of them (it is past):
     * the lesson after the move is taught from them, and the record keeps only one line for
     * that position. With [after], the board once the move (and the reply the engine expects)
     * is played: an idea about a piece that no longer stands there is left out -- "the
     * opponent's backward d-pawn on d6 is a lasting target" beside Qxc5 dxc5 was said as
     * "preserving your lasting target on d6" (the test audit, 2026-09-26). With [played], the
     * board right after the move: a better move that "stops" their threat, where the move played
     * stopped it too, says so -- "Qc4+ ... stops gxh1=Q+" beside 19.Qxg2, which took the pawn,
     * was taught as "the priority was to stop the promotion" (the dev audit, 2026-09-26).
     */
    fun facedText(facts: PositionFacts, board: Board, after: Board? = null, played: Board? = null): String {
        val visible = visibleFacts(facts, DisclosureLevel.LINE)
        // The hint's questions were for finding the move; the lesson is on the move made.
        for (key in listOf("question", "nudge")) visible.remove(key)
        val ideas = visible["ideas"] as? List<*>
        if (after != null && !ideas.isNullOrEmpty()) {
            visible["ideas"] = ideas.filter { !gone(it.toString(), after) }
        }
        val lines = ArrayList(factLines(visible, board, "player", past = true))
        val stopped = alsoStopped(facts, played)
        if (!stopped.isNullOrEmpty()) lines.add("the move played also stops $stopped")
        return (listOf(positionText(board, legal = false)) + lines).joinToString("\n")
    }

    /**
     * The sheet's lines for [visible], in reading order, in the third person for [side]; [past]
     * for a position the player has moved from.
     */
    private fun factLines(
        visible: Map<String, Any?>,
        board: Board,
        side: String,
        past: Boolean = false,
    ): List<String> {
        var names = FACT_NAMES
        if (side == "opponent") {
            // Their best move now, not the player's: "axb4 meets the player's idea" was taken
            // into the player's better line (the dev audit, 2026-09-26).
            names = names + ("why" to "the opponent's best move now and what it does")
        }
        if (past) {
            names = names + ("situation" to "the first hint for that position (to the player, as it stood)")
        }
        val out = ArrayList<String>()
        for (key in FACT_KEYS) {
            val value = visible[key]
            if (value == null || value == "" || value == false ||
                (value is List<*> && value.isEmpty())
            ) {
                continue
            }
            val name = names[key] ?: key.replace("_", " ")
            if (value == true) {
                out.add("$name: yes")
            } else if (key == "ideas") {
                out.add("ideas:")
                for (v in value as List<*>) {
                    out.add("  - " + Person.thirdPerson(v.toString(), side = side))
                }
            } else if (key == "threats") {
                // A threat is the other side's move if the side to move passed; what it does is
                // computed on that position. Named with its owner and moment: "what the other
                // side threatens" was read as the opponent's where it was the player's ("your
                // opponent's queen ... e5" of the player's own e5), and a threat as something a
                // move stopped ("Rae1 before they can use Re3": the test audit).
                val (owner, mover) = if (side == "player") {
                    "the opponent's" to "the player"
                } else {
                    "the player's" to "the opponent"
                }
                out.add(
                    "$owner threats, if $mover passed: " + (value as List<*>).joinToString(", ") {
                        "$owner ${threatWords(board, it.toString())}"
                    }
                )
            } else if (value is List<*>) {
                out.add("$name: " + value.joinToString(", ") { it.toString() })
            } else if (key in TO_THE_PLAYER) {
                out.add("$name: $value")
            } else if (key == "target" && "free square" in value.toString()) {
                // The king's free squares as it stands: taken as what a move left them ("Qb4
                // leaves only e4 and e5" where Qb4 took e4 away: the test audit, 2026-09-26).
                val said = Person.thirdPerson(value.toString(), side = side)
                out.add("$name: $said (as it stands, before any move)")
            } else {
                out.add("$name: " + Person.thirdPerson(value.toString(), side = side))
            }
        }
        return out
    }

    /**
     * The player's moves judged so far, as of [through]: the tally reads as the game went,
     * and the last move's message holds the whole of it.
     */
    private fun tallyText(through: PlyRecord): String {
        val counts = LinkedHashMap<String, Int>()
        val lost = ArrayList<Pair<Double, String>>()
        var n = 0
        val throughAt = indexOf(through)
        for (record in judgedRecords()) {
            if (indexOf(record) > throughAt) break
            val assessment = record.assessment
            if (!record.move.byUser || assessment == null) continue
            n += 1
            val label = assessment.label?.name?.lowercase() ?: assessment.severity.name.lowercase()
            counts[label] = (counts[label] ?: 0) + 1
            if (assessment.severity.name.lowercase() != "none") {
                val cost = assessment.expectedLoss
                lost.add(cost to "${numbered(record.move)} ($label)")
            }
        }
        if (n == 0) return ""
        val tally = LABEL_ORDER.filter { it in counts }.joinToString(", ") { "${counts[it]} $it" }
        var text = "The player's moves so far ($n judged): $tally."
        if (lost.isNotEmpty()) {
            // The costliest three, then the latest: "how did I do?" is answered from where the
            // game turned, and the last six alone named 39.Rb7 and 52.Kh3 over the blunders at
            // moves 19 and 24 (round 2). Python's sort is stable: equal costs keep their order.
            val costliest = lost.sortedBy { -it.first }.take(3).map { it.second }
            val latest = lost.takeLast(3).map { it.second }.filter { it !in costliest }
            text += " Ground lost most at: " + costliest.joinToString("; ") + "."
            if (latest.isNotEmpty()) text += " Lately: " + latest.joinToString("; ") + "."
        }
        // The opening the game was in, as the book last named it: it was shown only at the
        // ply where the name changed, and asked later the coach named one from memory.
        val opening = openingThrough(through)
        if (opening != null) text += " Opening: $opening."
        return text
    }

    /** The last opening name the book gave, up to [through]. */
    private fun openingThrough(through: PlyRecord): String? {
        for (record in records.take(indexOf(through) + 1).asReversed()) {
            if (record.withdrawn || record.analysis == null) continue
            val opening = record.analysis?.opening
            if (!opening.isNullOrEmpty()) return opening
        }
        return null
    }

    // ---- helpers -----------------------------------------------------------------------

    /** The moves of the game as it stands: not taken back, since the last re-read. */
    fun activeRecords(): List<PlyRecord> = records.filter { !it.withdrawn && !it.superseded }

    /** Every move that was played and judged, across re-reads, for the tally. */
    fun judgedRecords(): List<PlyRecord> = records.filter { !it.withdrawn }

    /**
     * The player's moves before [beforePly] played in a position with the very same idea -- the
     * pattern or the plan its facts name, [key] -- the latest [limit] of them, one line each: the
     * move, how it went, and what the coach said after it. Book moves are theory, not a pattern
     * found. The only earlier moments a lesson may tie itself to: told to find its own in the
     * record, the model misremembered them (quality rounds 2 and 3, 2026-09-25).
     * `conversation.same_idea`.
     */
    fun sameIdea(key: String, beforePly: Int, limit: Int = 2): List<String> {
        val active = activeRecords()
        val lines = ArrayList<String>()
        for ((i, record) in active.withIndex()) {
            val move = record.move
            if (i == 0 || move.ply >= beforePly || !move.byUser) continue
            val faced = active[i - 1]
            val facedFacts = faced.facts
            if (faced.move.ply != move.ply - 1 || facedFacts == null) continue
            if ((facedFacts.theme?.ifEmpty { null } ?: facedFacts.plan) != key) continue
            val a = record.assessment ?: continue
            if (a.label != null && a.label.name.lowercase() == "book") continue
            val label = a.label?.name?.lowercase() ?: a.severity.name.lowercase()
            var better = ""
            if (a.best.isNotEmpty() && a.best[0] != move.uci) {
                val m = Move.fromUciOrNull(a.best[0])
                better = if (m != null) Board(move.fenBefore).san(m) else ""
            }
            val how = if (better.isEmpty()) {
                "$label -- the player found it"
            } else if (a.severity == Severity.NONE) {
                "$label; $better was better"
            } else {
                val rated = (if (label[0] in "aeiou") "an " else "a ") + label
                "$rated; $better was better"
            }
            var line = "- At ${move.san} (the player's move): $how."
            val said = saidAfter(record)
            if (said.isNotEmpty()) line += "\n    YOU (the coach) said: \"$said\""
            lines.add(line)
        }
        return lines.takeLast(limit)
    }

    /** The coach's first words after [record]'s move, before the next move: "" if none. */
    private fun saidAfter(record: PlyRecord): String {
        var after = false
        for (entry in entries) {
            if (entry.kind == "ply") {
                if (after) return ""
                after = entry.record === record
            } else if (after && entry.kind == "ask") {
                return ""
            } else if (after && entry.kind == "coach" && entry.text.isNotEmpty()) {
                return entry.text
            }
        }
        return ""
    }

    /**
     * The moves of the game before [ply] (SAN) -- those played and those the verdicts preferred
     * -- that name a piece, a capture or castling: a pawn's step reads the same as a square, and
     * a reply names squares for what stands on them. `conversation.moves_before`.
     */
    fun movesBefore(ply: Int): Set<String> {
        val out = HashSet<String>()
        for (record in activeRecords()) {
            if (record.move.ply >= ply) continue
            out.add(record.move.san)
            val a = record.assessment
            if (a != null && a.best.isNotEmpty()) {
                val m = Move.fromUciOrNull(a.best[0])
                if (m != null) out.add(Board(record.move.fenBefore).san(m))
            }
        }
        return out.filter { !PAWN_STEP.matches(it) }.toSet()
    }

    /** `list.index`, by identity: two records of the same move would otherwise tie. */
    private fun indexOf(record: PlyRecord): Int = records.indexOfFirst { it === record }

    private fun recordBefore(record: PlyRecord): PlyRecord? {
        val active = activeRecords()
        val i = active.indexOfFirst { it === record }
        return if (i > 0) active[i - 1] else null
    }

    private fun recordAt(ply: Int): PlyRecord? {
        for (record in records.asReversed()) {
            if (!record.withdrawn && !record.superseded && record.move.ply == ply) return record
        }
        return null
    }

    /** A fact that lands after its ply message went out goes in a note of its own. */
    private fun noteIfFrozen(record: PlyRecord, what: String) {
        for (entry in entries) {
            if (entry.record === record && entry.frozen != null) {
                val boardBefore = Board(record.move.fenBefore)
                val colour = userColour
                val name = numbered(record.move)
                val text = if (what == "assessment") {
                    "Engine note on $name: " + verdictText(record, boardBefore, colour)
                } else {
                    val after = Board(record.move.fenAfter)
                    val shown = positionSections(record, after, colour, DisclosureLevel.LINE)
                        .filter { it.first == "Standing" || it.first == "Engine lines" }
                        .joinToString("\n") { (title, body) -> "$title: $body" }
                    "Engine note on the position after $name:\n$shown"
                }
                entries.add(Entry("user", "note", text))
                return
            }
        }
    }

    /**
     * The move with its number, from the position it was played in (not the ply parity:
     * after a re-read the numbering restarts wherever the board stood).
     */
    fun numbered(move: MovePlayed): String {
        val before = Board(move.fenBefore)
        val number = before.fullmoveNumber
        return if (before.turn == WHITE) "$number. ${move.san}" else "$number... ${move.san}"
    }

    private fun lastActiveName(): String {
        val active = activeRecords()
        return if (active.isNotEmpty()) numbered(active.last().move) else "the start"
    }

    private fun turnText(board: Board): String {
        if (board.isGameOver()) return "the game is over (${board.result()})"
        return (if (board.turn == WHITE) "White" else "Black") + " to move"
    }

    private fun clockText(ms: Int?): String {
        if (ms == null) return "?"
        val seconds = ms / 1000
        return String.format(Locale.ROOT, "%d:%02d", seconds / 60, seconds % 60)
    }

    /** An expected score from the mover's view, shown from the user's side. */
    private fun score(expected: Double, toMove: Boolean, colour: Boolean): String =
        format2(if (toMove == colour) expected else 1.0 - expected)

    private fun levelWords(level: DisclosureLevel): String = when (level) {
        DisclosureLevel.SILENT -> "say nothing about the position"
        DisclosureLevel.IDEA -> IDEA_LEVEL_WORDS
        DisclosureLevel.FOCUS -> "the idea, the target and the piece; not the move"
        DisclosureLevel.MOVE -> "the move and why it works may be named"
        DisclosureLevel.LINE -> "everything, the line included"
    }

    private fun factsOwner(userToMove: Boolean): String = if (userToMove) {
        "the player to move"
    } else {
        "the opponent to move; the opponent's plan and ideas"
    }

    companion object {
    /** The engine lines the harness shows: the best three, then three weaker ones. */
    const val BEST_LINES = 3
    const val WEAKER_LINES = 3
    const val LINE_PLIES = 8

    /**
     * With the compact record, a routine nod is written out for this many plies, so the coach
     * still sees its own last words and varies them; older ones go.
     */
    const val NOD_PLIES = 6

    /**
     * A record line whose verdict is a bare label: "12. Bxf6 (the player): best." Such a move
     * joins the compact record's move list; one with a cost, a preferred move or a note stands.
     */
    private val BARE_VERDICT = Regex(
        "^(\\d+\\.(?:\\.\\.)? \\S+) \\((?:the player|the opponent)\\): " +
            "(?:best|excellent|great|brilliant|good|book)\\.\$"
    )

    /**
     * The motifs worth naming on a move. The tagger also emits bookkeeping tags (quietMove,
     * defensiveMove, capture, check) and geometric ones that hold on most quiet moves
     * (interference, clearance); named in the harness they had the coach narrating "a6 cuts the
     * h5/f3/d1 line" on book moves.
     */
    val MOVE_MOTIFS = setOf(
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
    )

    /**
     * A threat with what it does: "Nd3+ (forks the queen on d2 and the rook on f1)".
     * `Conversation._threat_words`, a static method there too.
     */
    fun threatWords(board: Board, san: String): String {
        if (board.isCheck()) return san
        val passed = board.copy(withStack = false)
        passed.push(Move.NULL)
        val move = passed.parseSan(san) ?: return san
        val text = effectsText(passed, move, hazards = false)
        return if (text.isNotEmpty()) "$san ($text)" else san
    }

    /** A pawn's step, which reads the same as a square ([movesBefore]). */
    private val PAWN_STEP = Regex("[a-h][1-8](?:=[QRBN])?[+#]?")

    val LABEL_ORDER = listOf(
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

        /**
         * The facts in the order a coach reads them: what kind of moment, the goal and the
         * pattern, where to look, then the move.
         */
        val FACT_KEYS = listOf(
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

        /** What a fact is called on the sheet, where its key would not say it. */
        val FACT_NAMES = mapOf(
            // The template's first hint whole: its stakes, key fact, principle and trap, several
            // of which exist nowhere else on the sheet. Without it the model was asked for "the
            // stakes" and "the trap" and never given them (parity pass, 2026-09-25).
            "situation" to "the first hint in the board's own words, all of it true (to the player, as it " +
                "stands): the least a hint says",
            "theme" to "pattern",
            "mechanism" to "how it works",
            "question" to "a question that sets them looking (to the player, as it stands)",
            "nudge" to "a question from the target to the move (to the player, as it stands)",
            "why" to "the move and what it does",
            // Whose: the other side's moves, were the side to move to pass.
            // Named per side in [factLines]: whose threats, and at what moment.
            "threats" to "threats",
        )

        /** The facts addressed to the player, kept in the second person. */
        val TO_THE_PLAYER = listOf("question", "nudge", "situation")

        /**
         * `f"{value:.2f}"`: the exact binary value, rounded half to even.
         *
         * Java's `%.2f` rounds half up, so an exactly representable tie -- 1.25 pawns from
         * a score of 125 centipawns, 1.25 seconds from 1250 ms -- would read differently.
         */
        private fun formatted(value: Double, places: Int): String =
            BigDecimal(value).setScale(places, RoundingMode.HALF_EVEN).toPlainString()

        private fun format1(value: Double): String = formatted(value, 1)

        private fun format2(value: Double): String = formatted(value, 2)
    }
}
