package com.sidecoach.ui

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.squareName
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.profile.Puzzle
import com.sidecoach.vision.Pieces

/**
 * A puzzle solved on the panel's own board (Tai, 2026-09-20: "a doing-puzzle mode where you do
 * the puzzle, moving the pieces, right in the panel, with hint/help by clicking a chip; no auto
 * hint or anything from the coach"). `sidecoach/ui/puzzle.py`.
 *
 * The board is interactive: the solver's position from their side, a click on a piece and a
 * click on a square (the legal targets shown), the line's replies played back, a wrong move
 * shown for a moment and taken back as Lichess does, the solution on Give up. The coach is told
 * the position as a game of kind "puzzle" -- the engine measures it, so Hint, Best move and
 * Line answer -- and says nothing unasked. The outcome, solved through the last move or failed
 * at the first wrong one, goes to the profile beside the puzzle's motif (profile/Puzzles) and
 * moves the puzzle level.
 *
 * The session is pure state over the board, driven by its room on the room's loop thread; the
 * room reads it for the page.
 */
class PuzzleSession(val puzzle: Puzzle, val motif: String?) {
    val gameId = "puzzle-${puzzle.id}"
    val board = Board(puzzle.fen)
    val solver: Boolean
    val side: Pieces.Side
    val startFen: String

    /** The next move of the line, the solver's. */
    var index = 1
        private set

    /** Plies played on the board since the start. */
    var ply = 0
        private set
    val sans = ArrayList<String>()

    /**
     * The positions the page steps through: the start, then one per ply, each with the move
     * that made it, the king in check, and whether the solver played it.
     */
    val positions = ArrayList<Map<String, Any?>>()

    /** A wrong move shown, to be taken back. */
    var wrong: String? = null
        private set

    /**
     * Counts every change, so the page can tell an older answer from a newer one: a poll that
     * left before a move and came back after it showed the move taken back.
     */
    var version = 0
        private set
    var misses = 0
        private set

    /** playing | solved | given_up */
    var status = "playing"
        private set
    var message = "Your move."
        private set

    /** The outcome given to the profile (the room sets it). */
    var recorded = false

    init {
        // The opponent's move: the position shown. `push_uci` refuses a move that is not legal.
        val first = Move.fromUci(puzzle.moves[0])
        require(board.isLegal(first)) { "illegal uci: '${puzzle.moves[0]}' in ${puzzle.fen}" }
        board.push(first)
        solver = board.turn
        side = Pieces.Side.of(solver)
        startFen = board.fen()
        positions.add(position(null, byUser = false))
    }

    // ---- what the page shows -------------------------------------------------------------

    val done: Boolean get() = status != "playing"

    val failed: Boolean get() = misses > 0 || status == "given_up"

    fun state(): LinkedHashMap<String, Any?> {
        val yourMove = !done && board.turn == solver && wrong == null
        return linkedMapOf(
            "id" to puzzle.id,
            "url" to puzzle.url,
            "rating" to puzzle.rating,
            "themes" to puzzle.themes.toList(),
            "motif" to motif,
            "version" to version,
            "positions" to positions.toList(),
            "bottom" to side.name.lowercase(),
            "your_move" to yourMove,
            "legal" to if (yourMove) legal() else emptyMap<String, List<String>>(),
            "status" to status,
            "message" to message,
            "moves" to sans.toList(),
            "wrong" to wrong,
            "misses" to misses,
            "remaining" to maxOf(0, (puzzle.moves.size - index + 1) / 2),
        )
    }

    /**
     * Where each of the solver's pieces may go, for the page's dots: sent with the state, so a
     * piece picked up shows its squares at once.
     */
    fun legal(): LinkedHashMap<String, List<String>> {
        val out = LinkedHashMap<String, java.util.TreeSet<String>>()
        for (move in board.legalMoves()) {
            out.getOrPut(squareName(move.from)) { java.util.TreeSet() }.add(squareName(move.to))
        }
        return out.entries.associateTo(LinkedHashMap()) { it.key to it.value.toList() }
    }

    // ---- the moves -----------------------------------------------------------------------

    fun started(): GameStarted =
        GameStarted(gameId = gameId, userSide = side, startFen = startFen, kind = "puzzle")

    /**
     * The solver's move: the line's next move, or any mate where the line mates, is right and
     * brings the reply; a wrong one is shown and taken back by [retract]; anything else (not
     * their turn, not legal) is ignored. Returns the events of the moves that stand.
     */
    fun move(uci: String): List<MovePlayed> {
        if (done || wrong != null || board.turn != solver) return emptyList()
        var move = Move.fromUciOrNull(uci) ?: return emptyList()
        if (!board.isLegal(move)) {
            // A pawn reaching the last rank without a promotion piece: a queen.
            val promoted = Move(move.from, move.to, QUEEN)
            if (move.promotion == 0 && board.isLegal(promoted)) {
                move = promoted
            } else {
                return emptyList()
            }
        }
        val expected = if (index < puzzle.moves.size) puzzle.moves[index] else null
        version += 1
        if (move.uci() != expected && !mates(move)) {
            wrong = move.uci()
            misses += 1
            message = "Not that one. Have another look."
            return emptyList()
        }
        val events = arrayListOf(play(move, byUser = true))
        index += 1
        if (index >= puzzle.moves.size || board.isGameOver()) {
            status = "solved"
            message = if (misses == 0) "Solved." else "Solved, after a miss."
            return events
        }
        val reply = Move.fromUci(puzzle.moves[index])
        events.add(play(reply, byUser = false))
        index += 1
        if (index >= puzzle.moves.size) {
            status = "solved"
            message = "Solved."
        } else {
            message = "Your move."
        }
        return events
    }

    /** The wrong move shown is taken back; the position stands as it was. */
    fun retract() {
        version += 1
        wrong = null
        if (!done) message = "Your move."
    }

    /** The rest of the line, played out on the board. */
    fun giveUp(): List<MovePlayed> {
        if (done) return emptyList()
        version += 1
        wrong = null
        val events = ArrayList<MovePlayed>()
        while (index < puzzle.moves.size) {
            val move = Move.fromUci(puzzle.moves[index])
            if (!board.isLegal(move)) break
            events.add(play(move, byUser = board.turn == solver))
            index += 1
        }
        status = "given_up"
        message = "The solution, played out."
        return events
    }

    /** A position the page steps through: the placement, the move that made it, the king in check, whose. */
    private fun position(move: Move?, byUser: Boolean): Map<String, Any?> {
        val king = board.king(board.turn)
        val checked = board.isCheck() && king != null
        return linkedMapOf(
            "fen" to board.fen(),
            "uci" to move?.uci(),
            "check" to if (checked) squareName(king) else null,
            "by_user" to byUser,
        )
    }

    /** Lichess takes any mate where the line's move mates. */
    internal fun mates(move: Move): Boolean {
        val expected = if (index < puzzle.moves.size) puzzle.moves[index] else return false
        var trial = board.copy(withStack = false)
        val line = Move.fromUciOrNull(expected) ?: return false
        if (!trial.isLegal(line)) return false
        trial.push(line)
        if (!trial.isCheckmate()) return false
        trial = board.copy(withStack = false)
        trial.push(move)
        return trial.isCheckmate()
    }

    private fun play(move: Move, byUser: Boolean): MovePlayed {
        val before = board.fen()
        val san = board.san(move)
        board.push(move)
        ply += 1
        sans.add(san)
        positions.add(position(move, byUser = byUser))
        return MovePlayed(
            gameId = gameId,
            ply = ply,
            san = san,
            uci = move.uci(),
            fenBefore = before,
            fenAfter = board.fen(),
            byUser = byUser,
        )
    }
}
