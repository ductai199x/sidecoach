package com.sidecoach.ui

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.parseSquare
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
    var lastMove: String? = null
        private set

    /** A wrong move shown, to be taken back. */
    var wrong: String? = null
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
    }

    // ---- what the page shows -------------------------------------------------------------

    val done: Boolean get() = status != "playing"

    val failed: Boolean get() = misses > 0 || status == "given_up"

    fun state(): LinkedHashMap<String, Any?> = linkedMapOf(
        "id" to puzzle.id,
        "url" to puzzle.url,
        "rating" to puzzle.rating,
        "themes" to puzzle.themes.toList(),
        "motif" to motif,
        "fen" to board.fen(),
        "bottom" to side.name.lowercase(),
        "turn" to if (board.turn) "white" else "black",
        "your_move" to (!done && board.turn == solver && wrong == null),
        "check" to if (board.isCheck()) board.king(board.turn)?.let { squareName(it) } else null,
        "status" to status,
        "message" to message,
        "moves" to sans.toList(),
        "last_move" to lastMove,
        "wrong" to wrong,
        "misses" to misses,
        "remaining" to maxOf(0, (puzzle.moves.size - index + 1) / 2),
    )

    /**
     * Where the piece on [square] may go, for the page's dots; empty unless it is the solver's
     * turn and their piece.
     */
    fun targets(square: String): List<String> {
        if (done || wrong != null || board.turn != solver) return emptyList()
        val sq = squareOf(square) ?: return emptyList()
        val piece = board.pieceAt(sq)
        if (piece == null || piece.colour != solver) return emptyList()
        return board.legalMoves().filter { it.from == sq }.map { squareName(it.to) }.toSortedSet().toList()
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
        wrong = null
        if (!done) message = "Your move."
    }

    /** The rest of the line, played out on the board. */
    fun giveUp(): List<MovePlayed> {
        if (done) return emptyList()
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
        lastMove = move.uci()
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

    private companion object {
        /** `chess.parse_square`: a square's name, or null for anything else. */
        fun squareOf(name: String): Int? =
            if (name.length == 2 && name[0] in 'a'..'h' && name[1] in '1'..'8') parseSquare(name) else null
    }
}
