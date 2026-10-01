package com.sidecoach.profile

import com.sidecoach.analysis.hangingPawns
import com.sidecoach.analysis.isolatedDPawn
import com.sidecoach.analysis.openFiles
import com.sidecoach.analysis.outposts
import com.sidecoach.analysis.pawnWeaknesses
import com.sidecoach.analysis.phaseOf
import com.sidecoach.analysis.positionScore
import com.sidecoach.analysis.spacePlan
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareRank
import com.sidecoach.events.Analysis
import com.sidecoach.events.MotifTag
import com.sidecoach.events.ReviewJson
import com.sidecoach.events.ReviewedGame
import com.sidecoach.profile.GameSummaries.round
import java.util.Locale

/**
 * What a reviewed game says for Stats (docs/stats-plan.md): each move's facts and the game's,
 * read off the review once and kept, so the Stats window adds them up at once.
 * `sidecoach/profile/stat_facts.py`.
 *
 * The definitions are ours, as chess.com publishes none, and each is shown on the page:
 *
 * - A **tactic chance**: the mover's position had one move clearly best (the review's decisive
 *   call: the best line beats the next by a tenth of a game or more), its line a tactic (the
 *   tagger's motif on it, dealt within the mover's first two moves), and leaves the mover at
 *   least level; a mate counts only when it is in three or fewer, so no mate in 18 is "missed".
 *   Found when the move played cost under 0.02 of a game (Lichess's tolerance for "the same
 *   move").
 * - **Allowed**: the chance the move handed the opponent, by the same rule, in the position after.
 * - **Kinds**, chess.com's nine: checkmate, fork, pin, discovery, trapped piece, removing the
 *   defender, free piece, sacrifice, other.
 * - **Structure** of the position the move was played from: open (a pawn or none on the d- and
 *   e-files), closed (a d- or e-pawn blocked head on by an enemy pawn), else semi-open.
 * - **Themes** of strategy there, from the positional layer (analysis/Positional.kt): *space*
 *   when the mover has more of it, *activity* when a rook or queen of theirs stands on an open
 *   file or a minor piece on an outpost, *structure* when either side has a weak pawn (isolated,
 *   backward, doubled, hanging, the isolated queen's pawn).
 * - **Endgames** where the review's phases put them ([phaseOf]), by what is left besides kings
 *   and pawns: pawn, rook, queen, other; entered better, equal or worse by the mover's expected
 *   score then (0.6 and 0.4).
 */
object StatFacts {
    val KINDS = listOf(
        "checkmate",
        "fork",
        "pin",
        "discovery",
        "trapped piece",
        "removing the defender",
        "free piece",
        "sacrifice",
        "other",
    )
    private val KIND_OF = mapOf(
        "fork" to "fork",
        "pin" to "pin",
        "absolutePin" to "pin",
        "discoveredAttack" to "discovery",
        "doubleCheck" to "discovery",
        "trappedPiece" to "trapped piece",
        "capturingDefender" to "removing the defender",
        "hangingPiece" to "free piece",
        "sacrifice" to "sacrifice",
    )
    const val FOUND_LOSS = 0.02
    const val MATE_FINDABLE = 3
    const val BLOW_PLIES = 3 // the line's first plies: the mover's first two moves are its first and third

    // The tactics: the tagger's (`Causes.TACTICAL`), and a double check, a discovery at its sharpest.
    val TACTICS: Set<String> = Causes.TACTICAL + "doubleCheck"
    const val BETTER = 0.6
    const val WORSE = 0.4
    val ENDGAMES = listOf("pawn", "rook", "queen", "other")
    val THEMES = listOf("space", "activity", "structure")
    val STRUCTURES = listOf("open", "semi-open", "closed")

    /**
     * A tactic's kind from the motifs of its line: a mate first, then the named patterns, a free
     * piece only when nothing more precise is there. `kind_of`.
     */
    fun kindOf(tags: List<MotifTag>): String? {
        val names = tags.map { it.name }.filter { it in TACTICS }.toSet()
        if (names.isEmpty()) return null
        if (names.any { it == "mate" || it.startsWith("mateIn") || it.endsWith("Mate") }) return "checkmate"
        for (name in listOf("fork", "pin", "absolutePin", "discoveredAttack", "doubleCheck", "trappedPiece")) {
            if (name in names) return KIND_OF.getValue(name)
        }
        for (name in listOf("capturingDefender", "hangingPiece", "sacrifice")) {
            if (name in names) return KIND_OF.getValue(name)
        }
        return "other"
    }

    /** The tactic the side to move had in the analysed position, by kind, or null. `chance_in`. */
    fun chanceIn(analysis: Analysis): String? {
        if (!analysis.decisive || analysis.lines.isEmpty() || analysis.lines[0].moves.isEmpty()) return null
        val best = analysis.lines[0]
        val board = Board(analysis.fen)
        if (positionScore(board, analysis.lines) < 0.5) return null
        val mover = best.moves.take(BLOW_PLIES).filterIndexed { i, _ -> i % 2 == 0 }
        val tags = analysis.tags.filter { it.move == null || it.move in mover }
        val kind = kindOf(tags)
        val mate = best.mate
        if (kind == "checkmate" && (mate == null || !(0 < mate && mate <= MATE_FINDABLE))) return null
        return kind
    }

    /** Open, semi-open or closed, from the centre pawns. `structure_of`. */
    fun structureOf(board: Board): String {
        val pawns = board.pieces(PAWN, WHITE) + board.pieces(PAWN, BLACK)
        val centre = pawns.filter { squareFile(it) in 3..4 }
        if (centre.size <= 1) return "open"
        for (square in board.pieces(PAWN, WHITE)) {
            if (squareFile(square) in 3..4 && squareRank(square) < 7) {
                val ahead = board.pieceAt(square + 8)
                if (ahead != null && ahead.pieceType == PAWN && ahead.colour == BLACK) return "closed"
            }
        }
        return "semi-open"
    }

    /** The strategy themes of the position for the side to move ([THEMES]). `themes_of`. */
    fun themesOf(board: Board): List<String> {
        val us = board.turn
        val out = ArrayList<String>()
        if (spacePlan(board, us).any { it.kind == "spaceAdvantage" }) out.add("space")
        if (openFiles(board, us).isNotEmpty() || outposts(board, us).isNotEmpty()) out.add("activity")
        if (pawnWeaknesses(board, us).isNotEmpty() ||
            hangingPawns(board, us).isNotEmpty() ||
            hangingPawns(board, !us).isNotEmpty() ||
            isolatedDPawn(board, us).isNotEmpty()
        ) {
            out.add("structure")
        }
        return out
    }

    /** An endgame's kind by what is left besides kings and pawns. `endgame_of`. */
    fun endgameOf(board: Board): String {
        val left = board.pieceMap().values.map { it.pieceType }.toSet() - setOf(KING, PAWN)
        if (left.isEmpty()) return "pawn"
        if (left == setOf(ROOK)) return "rook"
        if (left == setOf(QUEEN)) return "queen"
        return "other"
    }

    /** `standing`. */
    fun standing(score: Double): String = if (score >= BETTER) "better" else if (score <= WORSE) "worse" else "equal"

    /**
     * Whether a side was a computer: chess.com's bot and coach games, Lichess's BOT title, a name
     * marked as a bot. `bot_game`.
     */
    fun botGame(headers: Map<String, String>): Boolean {
        val event = (headers["Event"] ?: "").lowercase(Locale.ROOT)
        if (listOf("vs bot", "vs coach", "vs computer").any { it in event }) return true
        for (colour in listOf("White", "Black")) {
            val name = headers[colour] ?: ""
            if ((headers["${colour}Title"] ?: "").uppercase(Locale.ROOT) == "BOT" || name.uppercase(Locale.ROOT).endsWith("-BOT")) {
                return true
            }
            if (name.startsWith("Coach-")) return true
        }
        return false
    }

    /** A reviewed game's facts: the game's, and every move's. `facts_of`. */
    fun factsOf(game: ReviewedGame, row: LibraryRow): Pair<GameFact, List<MoveFact>> {
        val moves = ArrayList<MoveFact>()
        var before = game.start
        var inBook = game.start.inBook
        for (move in game.moves) {
            val board = Board(move.fenBefore)
            val score = positionScore(board, before.lines)
            val phase = phaseOf(board, inBook = inBook)
            val chance = chanceIn(before)
            val label = move.assessment.label
            moves.add(
                MoveFact(
                    ply = move.ply,
                    side = move.side.value,
                    fen = move.fenBefore,
                    uci = move.assessment.uci,
                    san = move.san,
                    phase = phase,
                    loss = round(move.assessment.expectedLoss, 4),
                    label = label?.let { ReviewJson.labelName(it) },
                    accuracy = move.assessment.accuracy,
                    score = round(score, 4),
                    chance = chance,
                    found = chance != null && move.assessment.expectedLoss < FOUND_LOSS,
                    allowed = chanceIn(move.after),
                    structure = structureOf(board),
                    themes = themesOf(board).joinToString(","),
                    endgame = if (phase == "endgame") endgameOf(board) else null,
                    spentMs = move.spentMs,
                    clockMs = move.clockMs,
                ),
            )
            before = move.after
            inBook = move.after.inBook
        }
        val opening = game.moves.reversed().firstOrNull { !it.after.opening.isNullOrEmpty() }?.after?.opening
        val headers = game.headers
        val fact = GameFact(
            key = row.key,
            source = row.source,
            timeClass = row.timeClass,
            playedAt = row.playedAt,
            userSide = row.userSide,
            bot = botGame(headers),
            result = headers["Result"]?.takeIf { it.isNotEmpty() } ?: row.result,
            opening = opening ?: game.start.opening,
            termination = headers["Termination"],
        )
        return fact to moves
    }

    /** A reviewed game's facts kept for Stats, in place of any from an older review. `keep_stats`. */
    fun keepStats(store: Store, strength: String, reviewed: ReviewedGame, row: LibraryRow) {
        val (game, moves) = factsOf(reviewed, row)
        store.saveStats(strength, game, moves)
    }
}

/** One move's facts for Stats. `MoveFact`. */
data class MoveFact(
    val ply: Int,
    val side: String,
    val fen: String, // the position the move was played from
    val uci: String,
    val san: String,
    val phase: String,
    val loss: Double,
    val label: String?,
    val accuracy: Double?,
    val score: Double, // the mover's expected score before the move
    val chance: String?, // the tactic the mover had, by kind
    val found: Boolean,
    val allowed: String?, // the tactic the move handed the opponent, by kind
    val structure: String,
    val themes: String, // the strategy themes there, comma-separated
    val endgame: String?,
    val spentMs: Int?,
    val clockMs: Int?,
)

/** A game's facts for Stats. `GameFact`. */
data class GameFact(
    val key: String,
    val source: String,
    val timeClass: String?,
    val playedAt: Double?,
    val userSide: String?,
    val bot: Boolean,
    val result: String?,
    val opening: String?,
    val termination: String?,
)
