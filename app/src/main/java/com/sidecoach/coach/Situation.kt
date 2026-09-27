package com.sidecoach.coach

import com.sidecoach.analysis.Exchange
import com.sidecoach.analysis.PositionalFact
import com.sidecoach.analysis.atHome
import com.sidecoach.analysis.isPassed
import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_FILES
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.popcount
import com.sidecoach.chess.square
import com.sidecoach.chess.squareDistance
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank
import com.sidecoach.chess.squaresOf
import com.sidecoach.events.Analysis
import com.sidecoach.events.Line

/**
 * The first hint: what is at stake, and the fact on the board that matters. A port of
 * `sidecoach/coach/situation.py`.
 *
 * Tai, 2026-09-24, after reading Capablanca-Marshall move by move: the first hint was a
 * principle and a question ("no tactic here -- this is about fighting for the centre: what
 * would make the centre yours?"), and in a middlegame there are a million ways to fight for the
 * centre while only a few moves keep the game. The first hint is now a short lesson on the
 * position:
 *
 * - the stakes, from the engine's lines: you are in check; this is opening theory; only one
 *   move holds (and what even the next best one costs); only two or three keep your edge;
 *   nothing is urgent and several moves are fine; the game is won, or lost;
 * - the key fact, named on the board: their king's squares and your checks, their threat, the
 *   target, the plan and the pieces it needs, the concrete thing the best move is about;
 * - the trap, when there is one: the capture that looks free and is not.
 *
 * It names pieces, squares and the opponent's moves -- never the player's move, nor the square
 * it goes to unless an enemy piece stands there (the capture's target is the lesson). A bare
 * principle is never the hint; the principle comes only after the fact it explains.
 */
object Situation {

    /** Within this of the best move's expected score, a move is as good. */
    const val GOOD_MARGIN = 0.04

    /** Past the one good move, a drop this big: only one move holds. */
    const val ONLY_DROP = 0.15

    /**
     * Past two or three good moves, a drop this big: only they keep it (14.Qf3 in
     * Capablanca-Marshall: d4 and Qf3 at 0.60, the rest at 0.50 and below).
     */
    const val FEW_DROP = 0.08

    /** This many good moves: nothing is urgent. */
    const val MANY = 4
    const val WON = 0.8
    const val LOST = 0.2

    /** The good moves named when several are fine, or in opening theory. */
    const val CHOICES = 3

    private val NAMES = mapOf(
        PAWN to "pawn",
        KNIGHT to "knight",
        BISHOP to "bishop",
        ROOK to "rook",
        QUEEN to "queen",
        KING to "king",
    )
    /**
     * Counts in words to twenty: a digit in the first hint is a figure the coach may not quote
     * ("you have 10 checks": the pre-spend audit, 2026-09-25).
     */
    private val NUMBER = mapOf(
        2 to "two", 3 to "three", 4 to "four", 5 to "five", 6 to "six", 7 to "seven", 8 to "eight",
        9 to "nine", 10 to "ten", 11 to "eleven", 12 to "twelve", 13 to "thirteen", 14 to "fourteen",
        15 to "fifteen", 16 to "sixteen", 17 to "seventeen", 18 to "eighteen", 19 to "nineteen",
        20 to "twenty",
    )

    /** `re.compile(r"[a-h](x[a-h])?[1-8]").match`: at the start of the text. */
    private val PAWN_MOVE = Regex("^[a-h](x[a-h])?[1-8]")

    /** `idea.VALUES`: pawns' worth, the king not counted. */
    private val VALUES = mapOf(PAWN to 1, KNIGHT to 3, BISHOP to 3, ROOK to 5, QUEEN to 9)

    /**
     * The second hint's question for the best move's own idea, when the idea reader has none:
     * from the fact the first hint named to the move, never the move itself.
     */
    val POSITIONAL_NUDGE: Map<String, String> = linkedMapOf(
        "castle" to "Where is your king safest, and how does it get there in one move?",
        "rescue" to "Where is it safe -- and where does it also do something?",
        "prophylaxis" to "Which move takes that idea away without losing time?",
        "develop" to "Which of them comes out first, and to which square?",
        "activateKing" to "Which way should your king walk?",
        "outpostMove" to "Which square is it, and which knight gets there?",
        "takeFile" to "Which rook goes to it?",
        "rookBehindPasser" to "Which square is behind it?",
        "break" to "Is it time to strike, or does it need one more move of support?",
        "centre" to "Which pawn move lets it out?",
        "centralise" to "Which pawn move lets it out?",
        "space" to "Which pawn move lets it out?",
        "activate" to "Where would it do more?",
        "pushPasser" to "How far can it go safely right now?",
        "passerPawnEnding" to "How far can it go safely right now?",
        "pawnStorm" to "Which pawn goes first?",
        "simplify" to "Which trade is it?",
    )

    /**
     * Questions that fit any position with something to win, and so narrow nothing once the
     * first hint has named the target: never said after it.
     */
    val ANY_POSITION: Set<String> = setOf(
        "Which of their pieces is loose or attacked more times than it is defended?",
        "Which of your pieces has no job right now?",
        "Go through your checks and captures: which leaves them the fewest good replies?",
    )

    /**
     * The second hint's question for the best move's own idea ([POSITIONAL_NUDGE]), said of
     * what the first hint named: one piece at home is "it", not "which of them"; space taken is
     * not a piece let out.
     */
    fun positionalNudge(board: Board, kind: String, key: String): String? {
        if (kind == "develop" && undeveloped(board, board.turn).size == 1) return "Which square does it belong on?"
        if (kind == "space" && "shut in" !in key) return "Which pawn goes forward, and how far?"
        return POSITIONAL_NUDGE[kind]
    }

    private val CENTRE = listOf(square(3, 3), square(4, 3), square(3, 4), square(4, 4))  // d4, e4, d5, e5

    /**
     * What a pawn move in the centre does, named on the board, for the second hint and the
     * third: the central squares it now covers, or the file a pawn capture opens. "Fighting for
     * the centre" said nothing (Tai, 2026-09-24); "f4 covers e5" says what f4 is for.
     */
    fun centreMove(board: Board, move: Move): Pair<String, String>? {
        if (board.pieceTypeAt(move.from) != PAWN) return null
        val us = board.turn
        val san = board.san(move)
        val after = board.copy(withStack = false)
        after.push(move)
        val before = board.attacksMask(move.from)
        var covered = CENTRE.filter { after.attacksMask(move.to) and BB_SQUARES[it] != BB_EMPTY }
        covered = covered.filter { before and BB_SQUARES[it] == BB_EMPTY }
        if (covered.isNotEmpty()) {
            val where = listed(covered.map { sq(it) })
            return "Which pawn move covers $where?" to "$san: the pawn now covers $where"
        }
        if (board.isCapture(move)) {
            val file = squareFile(move.from)
            if (board.piecesMask(PAWN, us) and BB_FILES[file] and BB_SQUARES[move.from].inv() == BB_EMPTY) {
                val letter = "abcdefgh"[file]
                return "Which capture opens a file for your rooks?" to
                    "$san: the capture opens the $letter-file for your rooks"
            }
        }
        return null
    }

    /**
     * A piece brought to a better square, counted: "Bf6: your bishop goes from four squares to
     * nine". "The move is Bf6: centralising a piece" said nothing (Karpov-Ribli).
     */
    fun reachWhy(board: Board, move: Move): String? {
        val piece = board.pieceAt(move.from) ?: return null
        if (piece.pieceType == PAWN || piece.pieceType == KING) return null
        val us = board.turn
        val own = if (us) 1 else 0
        val before = popcount(board.attacksMask(move.from) and board.occupiedCo[own].inv())
        val after = board.copy(withStack = false)
        after.push(move)
        val later = popcount(after.attacksMask(move.to) and after.occupiedCo[own].inv())
        if (later <= before) return null
        val was = if (before == 1) "one square" else "${IdeaWords.inWords(before)} squares"
        val name = NAMES.getValue(piece.pieceType)
        return "${board.san(move)}: your $name goes from $was to ${IdeaWords.inWords(later)}"
    }

    private fun wing(sq: Int): String {
        val file = squareFile(sq)
        if (file <= 2) return "on the queenside"
        if (file >= 5) return "on the kingside"
        return "in the centre"
    }

    /**
     * The first hint said as a coach speaks (Tai, 2026-09-25: "make it sound a bit better and more
     * natural"): the contractions of speech. A verb that ends its clause stays whole ("where it
     * is.", never "where it's."), so "is" and "are" contract only before a word.
     */
    private val CONTRACTIONS: List<Pair<Regex, String>> = listOf(
        Regex("\\b([Cc])annot\\b") to "\$1an't",
        Regex("\\b([Ww])ill not\\b") to "\$1on't",
        Regex(
            "\\b(is|are|was|were|does|do|did|has|have|had|could|would|should) not\\b",
            RegexOption.IGNORE_CASE,
        ) to "\$1n't",
        Regex("\\b([Ii]t|[Tt]here|[Tt]hat|[Ww]hat) is (?=\\w)") to "\$1's ",
        Regex("\\b([Yy]ou|[Tt]hey|[Ww]e) are (?=\\w)") to "\$1're ",
        Regex("\\bI am\\b") to "I'm",
    )

    /**
     * `text` with the contractions a coach speaks with: "it's a trap", "can't take back", "you're
     * lost".
     */
    fun spoken(text: String): String {
        var said = text
        for ((pattern, short) in CONTRACTIONS) said = pattern.replace(said, short)
        return said
    }

    /**
     * The most sentences a first hint says. Past it, the details of the pattern go first, the
     * least telling first: a second pattern in the line, the quiet first move, the in-between
     * move, which move the blow comes on. The stakes, the prize, the key fact, the pattern and
     * why it works, a sacrifice on the way, and the question stay (the first hint ran to seven,
     * eight, ten sentences when every detail stacked up: the real-game review, 2026-09-26).
     */
    const val FIRST_HINT_SENTENCES = 5
    private val DETAILS = listOf(
        "There is (?:a|an) [^.?!]* in it too\\.",
        "It starts with a quiet move: no check and no capture, but a threat they cannot meet\\.",
        "On the way comes an in-between move: a check before taking back\\.",
        "It comes on your \\w+ move(?:, once [^.?!]*)?\\.",
    ).map { Regex(it) }
    private val HINT_SENTENCE_END = Regex("(?<=[.?!])\\s+(?=[A-Z])")
    private val SPACES = Regex("\\s{2,}")

    /**
     * [text] with the pattern's details dropped, least telling first, until it says at most
     * [budget] sentences -- or none is left to drop. `situation.within_budget`.
     */
    fun withinBudget(text: String, budget: Int = FIRST_HINT_SENTENCES): String {
        var out = text
        for (detail in DETAILS) {
            if (out.trim().split(HINT_SENTENCE_END).size <= budget) break
            out = SPACES.replace(detail.replace(out, ""), " ").trim()
        }
        return out
    }

    /**
     * `text` as a sentence: a capital first (never on a pawn's move: "hxg4 looks..."), a stop
     * last.
     */
    fun sentence(raw: String): String {
        var text = raw.trim()
        if (text.isEmpty()) return ""
        if (!PAWN_MOVE.containsMatchIn(text)) text = text[0].uppercaseChar() + text.substring(1)
        return if (text.last() in ".!?") text else "$text."
    }

    /**
     * How much rides on the move: its kind, how many moves are as good as the best, the best
     * move's expected score, and the first line past the good ones.
     */
    data class Stakes(
        /** "check", "book", "won", "lost", "only", "few", "many", or "open". */
        val kind: String,
        val good: Int,
        val top: Double,
        val nextLine: Line?,
        val legal: Int,
        val choices: List<String> = emptyList(),
    )

    private fun sq(square: Int): String = squareName(square)

    private fun named(board: Board, square: Int): String {
        val piece = board.pieceAt(square) ?: return sq(square)
        return "${NAMES.getValue(piece.pieceType)} on ${sq(square)}"
    }

    private fun listed(items: List<String>): String {
        if (items.size == 1) return items[0]
        return items.dropLast(1).joinToString(", ") + " and ${items.last()}"
    }

    /** The stakes of the position, from the engine's lines. */
    fun readStakes(board: Board, analysis: Analysis): Stakes? {
        val lines = analysis.lines.filter { it.moves.isNotEmpty() }
        if (lines.isEmpty()) return null
        val legal = board.legalMoves().size
        val top = lines[0].expectedScore
        val topMate = lines[0].mate
        val goodLines = if (topMate != null && topMate > 0) {
            lines.filter { it.mate != null && it.mate > 0 }
        } else {
            lines.filter { top - it.expectedScore <= GOOD_MARGIN }
        }
        val rest = lines.filter { it !in goodLines }
        val nextLine = rest.firstOrNull()
        val good = goodLines.size
        val choices = ArrayList<String>()
        for (line in goodLines.take(CHOICES)) {
            val move = Move.fromUciOrNull(line.moves[0]) ?: continue
            if (board.isLegal(move)) choices.add(board.san(move))
        }
        val drop = if (nextLine != null) top - nextLine.expectedScore else 0.0
        val kind = when {
            board.isCheck() -> "check"
            analysis.inBook -> "book"
            top >= WON -> "won"
            top < LOST -> "lost"
            good == 1 && nextLine != null && drop >= ONLY_DROP -> "only"
            good <= 3 && nextLine != null && drop >= FEW_DROP -> "few"
            good >= MANY || (good == lines.size && legal > lines.size) -> "many"
            else -> "open"
        }
        return Stakes(kind, good, top, nextLine, legal, choices)
    }

    /**
     * What the line's first move hands the opponent, read from the position it leaves by the
     * same idea reader: "a knight", "your queen", "mate", "a winning attack".
     */
    private fun whatItGives(board: Board, line: Line): String? {
        val first = line.moves.getOrNull(0)?.let { Move.fromUciOrNull(it) } ?: return null
        if (!board.isLegal(first) || line.moves.size < 2) return null
        // With the board's moves: a capture of the player's is the start of an exchange, and the
        // piece taken back for it is not "given" ([IdeaReader.exchangeOwed]).
        val after = board.copy()
        after.push(first)
        val theirs = Analysis(
            gameId = "",
            fen = after.fen(),
            ply = 0,
            lines = listOf(
                Line(
                    moves = line.moves.drop(1),
                    depth = line.depth,
                    expectedScore = 1.0 - line.expectedScore,
                    mate = if (line.mate != null) -line.mate else null,
                ),
            ),
            decisive = false,
            onlyMove = false,
        )
        val idea = IdeaReader.read(after, theirs)
        if (idea.goal == "mate") return "mate"
        if (idea.goal == "win" && !idea.won.isNullOrEmpty() && !idea.level && !idea.back) return IdeaWords.yours(idea.won)
        if (idea.goal == "attack" && idea.winning) return "a winning attack"
        if (idea.goal == "promote") return "a new queen"
        if (idea.goal == "convert") return "a won ending"
        return null
    }

    /**
     * Whether the player has the pieces `given` names as lost: a knight promoted in the line and
     * taken is not "a knight" the player has to lose (Lichess UkpFL, a rook ending).
     */
    private fun have(board: Board, given: String): Boolean {
        val lost = given.split(" for ", limit = 2)[0]
        for (pieceType in intArrayOf(KNIGHT, BISHOP, ROOK, QUEEN)) {
            if (NAMES.getValue(pieceType) in lost && board.piecesMask(pieceType, board.turn) == BB_EMPTY) return false
        }
        return true
    }

    /** What the best move keeps, by its expected score: "your edge", "the balance". */
    fun edgeWords(top: Double): String = edge(top)

    private fun edge(top: Double): String {
        if (top >= 0.6) return "your edge"
        if (top >= 0.4) return "the balance"
        return "you in the game"
    }

    /**
     * What even the next best move costs: grounded in its line when the reader sees it, else in
     * the score.
     */
    private fun cost(board: Board, stakes: Stakes): String {
        val line = stakes.nextLine ?: return ""
        val given = whatItGives(board, line)
        if (given == "mate") return "even the next best move gets you mated"
        if (given == "a winning attack") return "even the next best move hands them a winning attack"
        if (given == "a new queen") return "even the next best move lets them queen"
        if (given == "a won ending") return "even the next best move hands them a won ending"
        if (given != null && have(board, given)) return "even the next best move loses $given"
        val rest = line.expectedScore
        if (rest < 0.3) return "anything else loses"
        if (rest < 0.45) return "anything else leaves you worse"
        if (stakes.top >= 0.6 && rest < 0.55) return "anything else gives away your edge"
        return "anything else gives back much of what you have"
    }

    /** The stakes as the first hint says them. */
    fun stakesWords(board: Board, stakes: Stakes, opening: String?): String? {
        val kind = stakes.kind
        if (kind == "check") {
            if (stakes.legal == 1) {
                // And whether it saves anything: "only one legal move" of a mated king was read
                // as "only one move holds; otherwise you are mated" (the test audit, 2026-09-26).
                if (stakes.top < LOST) {
                    return "You're in check, and there is only one legal move -- it does not save the game"
                }
                return "You're in check, and there is only one legal move"
            }
            if (stakes.good == 1 && stakes.nextLine != null) {
                if (stakes.top >= WON) {
                    return "You're in check but winning, and only one way out keeps it that way: " +
                        cost(board, stakes)
                }
                return "You're in check, and only one way out holds: ${cost(board, stakes)}"
            }
            if (stakes.top >= WON) return "You're in check but winning: get out of it without giving anything back"
            if (stakes.good >= 2) return "You're in check, and more than one way out is fine"
            return "You're in check: choose the way out with care"
        }
        if (kind == "book") {
            val moves = if (stakes.choices.isNotEmpty()) listed(stakes.choices) else null
            val name = if (!opening.isNullOrEmpty()) " -- the $opening" else ""
            if (moves != null) return "This is opening theory$name; the main moves here are $moves"
            return "This is opening theory$name"
        }
        if (kind == "won") {
            val line = stakes.nextLine
            if (stakes.good == 1 && line != null && stakes.top - line.expectedScore >= ONLY_DROP) {
                return "You're winning, but only one move keeps it that way: ${cost(board, stakes)}"
            }
            return null
        }
        if (kind == "lost") return "You're losing here; the best move makes them work for it"
        if (kind == "only") return "Only one move holds here: ${cost(board, stakes)}"
        if (kind == "few") {
            if (stakes.good == 1) return "Only one move keeps ${edge(stakes.top)} here: ${cost(board, stakes)}"
            val n = NUMBER[stakes.good] ?: stakes.good.toString()
            return "Only $n moves keep ${edge(stakes.top)} here: ${cost(board, stakes)}"
        }
        if (kind == "many") {
            val moves = if (stakes.choices.isNotEmpty()) listed(stakes.choices) else null
            if (moves != null) return "Nothing is urgent: several moves keep ${edge(stakes.top)}, $moves among them"
            return "Nothing is urgent: several moves keep ${edge(stakes.top)}"
        }
        return null  // a few moves close together: nothing worth saying about the stakes
    }

    /**
     * The traps in the order the first hint wants them: the one that takes what the best line
     * wins later ("not yet"), then those that walk into mate, then the worst first. Topalov-
     * Kramnik, 36.Qxg3?? "allows mate in two" was not the first trap found.
     */
    fun rankedTraps(traps: List<Line>, target: Int?): List<Line> {
        fun square(trap: Line): Int? = trap.moves.getOrNull(0)?.let { Move.fromUciOrNull(it) }?.to
        // `sorted(traps, key=(on target, mated, expected score))`: stable, as Python's sort is.
        return traps.sortedWith(
            compareBy<Line>(
                { if (target != null && square(it) == target) 0 else 1 },
                { if (it.mate != null && it.mate < 0) 0 else 1 },
                { it.expectedScore },
            ),
        )
    }

    /** How far a trap's line is followed to see what it nets. */
    const val TRAP_NET_PLIES = 8

    /** A better game, by expected score. */
    const val EDGE = 0.6

    /** Below the edge by this much after the capture: the edge is gone. */
    const val TRAP_EDGE_DROP = 0.05

    /** The capture costs this much ... */
    const val BALANCE_DROP = 0.2

    /** ... and leaves the player below this: clearly worse. */
    const val WORSE = 0.4

    /**
     * The capture that looks free and is not: "careful, though: hxg4 looks like it wins a knight,
     * but after hxg4 Qh4, they get a winning attack". When it takes what the best line wins later
     * (`target`), the two are one lesson: "their rook on a8 can be won, but not at once: Qxa8 right
     * away runs into Nxh3+, and they have mate" (16.Re2 in Capablanca-Marshall; [inOrder] when the
     * hint has named the target already). `top` is the best move's expected score: a game already
     * lost is not lost by the capture.
     */
    fun trapWords(
        board: Board,
        trap: Line,
        target: Int? = null,
        top: Double? = null,
        best: Move? = null,
    ): String? {
        val capture = trap.moves.getOrNull(0)?.let { Move.fromUciOrNull(it) } ?: return null
        val taken = board.pieceAt(capture.to)
        if (!board.isLegal(capture) || taken == null || trap.moves.size < 2) return null
        val san = board.san(capture)
        val after = board.copy(withStack = false)
        after.push(capture)
        val reply = Move.fromUciOrNull(trap.moves[1]) ?: return null
        if (!after.isLegal(reply)) return null
        val replySan = after.san(reply)
        // Their reply, unless it would say where the solution goes: a reply onto the first move's
        // square, empty now, or written as the first move is ("after Rxh6 e5" where the solution
        // is e5; "after Qxf7 Rxf7" where it is Rxf7). The trap is said without it.
        val shows = best != null && (
            replySan == board.san(best) ||
                (reply.to == best.to && board.pieceAt(best.to) == null)
            )
        val given = whatItGives(board, trap)
        // A capture that comes out ahead or even in material, with no mate, attack or lost game
        // behind it, is a lesser move, not a trap (19...Bxf2 Qxf2 won the exchange and 21.Qxf2
        // Qxc1 traded evenly in Capablanca-Marshall).
        val ahead = net(board, trap.moves) >= 0
        val tail = when {
            given == "mate" -> "they have mate"
            given == "a winning attack" -> "they get a winning attack"
            given != null && !ahead -> "they win $given"
            trap.expectedScore < 0.3 && (top == null || top >= 0.3) -> "you are lost"
            checks(after, trap.moves.drop(1)) >= 2 -> "their checks begin, and your king has no cover"
            !ahead -> "it costs you more than it wins"
            // The balance lost without material: the wrong recapture opening a file (23.exd5?
            // Re8 in Polugaevsky-Nezhmetdinov).
            top != null && top - trap.expectedScore >= BALANCE_DROP && trap.expectedScore < WORSE ->
                "you are left clearly worse"
            // Even in material but no longer better: "Not 17 Qxf2 Bg3 ... with an unclear
            // position" (Capablanca-Marshall).
            top != null && top >= EDGE && trap.expectedScore < EDGE - TRAP_EDGE_DROP -> "your edge is gone"
            else -> return null
        }
        val what = NAMES.getValue(taken.pieceType)
        if (best != null && best != capture && best.to == capture.to) {
            // The same piece, taken with the wrong one of yours.
            val afterIt = if (shows) san else "$san $replySan"
            return "$RIGHT_PIECE, though: $san looks natural, but after $afterIt, $tail"
        }
        if (target != null && capture.to == target) {
            val runs = if (shows) "a reply" else replySan
            return "their ${named(board, target)} can be won$NOT_YET$san right away runs into " +
                "$runs, and $tail"
        }
        val afterIt = if (shows) san else "$san $replySan"
        return "careful, though: $san looks like it wins a $what, but after $afterIt, $tail"
    }

    /**
     * The trap that takes the target too soon, as [trapWords] says it: the target named, then the
     * order ("their rook on b8 can be won, but not at once: Qxb8 right away runs into ...").
     */
    const val NOT_YET = ", but not at once: "
    const val RIGHT_PIECE = "take back with the right piece"

    /**
     * A trap that takes the target too soon, where the hint has named the target already: the
     * order alone ("careful with the order, though: Qxb8 right away runs into Qd2+").
     */
    fun inOrder(trap: String): String {
        if (NOT_YET !in trap) return trap
        return "careful with the order, though: " + trap.substringAfter(NOT_YET)
    }

    /** A trap said first, with nothing before it to turn from: "careful: ...". */
    fun firstSaid(trap: String): String = trap.replaceFirst(", though: ", ": ")

    /**
     * What the line nets the side to move in material, in pawns, as it stands after their last
     * reply within [TRAP_NET_PLIES].
     */
    private fun net(board: Board, moves: List<String>): Int {
        val us = board.turn
        val b = board.copy(withStack = false)

        fun balance(): Int = b.pieceMap().values.sumOf { p ->
            (VALUES[p.pieceType] ?: 0) * (if (p.colour == us) 1 else -1)
        }

        val start = balance()
        var net = 0
        for ((j, uci) in moves.take(TRAP_NET_PLIES).withIndex()) {
            val move = Move.fromUciOrNull(uci) ?: break
            if (!b.isLegal(move)) break
            b.push(move)
            if (j % 2 == 1) net = balance() - start
        }
        return net
    }

    /** How many checks the side to move gives in its first four moves of `moves`. */
    private fun checks(board: Board, moves: List<String>): Int {
        val b = board.copy(withStack = false)
        var given = 0
        for ((j, uci) in moves.take(8).withIndex()) {
            val move = Move.fromUciOrNull(uci) ?: break
            if (!b.isLegal(move)) break
            if (j % 2 == 0 && b.givesCheck(move)) given += 1
            b.push(move)
        }
        return given
    }

    /** The player's knights and bishops still on their home squares. */
    fun undeveloped(board: Board, us: Boolean): List<Int> = atHome(board, us)

    /**
     * A pawn move that lets a piece still at home out: the piece's square and the pawn's. "Your
     * bishop on c1 is shut in behind the pawn on d2" -- what 15.d4 was for in Capablanca-Marshall
     * ("development counts for more than material").
     */
    fun shutIn(board: Board, move: Move): Pair<Int, Int>? {
        val us = board.turn
        if (board.pieceTypeAt(move.from) != PAWN) return null
        val after = board.copy(withStack = false)
        after.push(move)
        val own = if (us) 1 else 0
        // Bishops only: a queen's diagonal opened is not what a pawn move is for (c4 "lets your
        // queen out" said nothing about the English).
        for (s in undeveloped(board, us)) {
            if (board.pieceTypeAt(s) != BISHOP) continue
            val before = popcount(board.attacksMask(s) and board.occupiedCo[own].inv())
            val later = popcount(after.attacksMask(s) and after.occupiedCo[own].inv())
            if (later - before >= 2) return s to move.from
        }
        return null
    }

    /** The cheapest enemy piece that can take on `square` legally. */
    private fun attacker(board: Board, square: Int): Int? {
        val piece = board.pieceAt(square) ?: return null
        val takers = Exchange.capturers(board, !piece.colour, square)
        if (takers.isEmpty()) return null
        return takers.minByOrNull { Exchange.VALUE[board.pieceTypeAt(it)] }
    }

    /**
     * The concrete situation behind the best move's own idea, named on the board -- the
     * position, not the move: "your king is still on e1", "your bishop on c1 is shut in".
     */
    fun positionalKey(board: Board, best: Move, kind: String, threat: Move?, phase: String): String? {
        val us = board.turn
        val piece = board.pieceAt(best.from) ?: return null
        val name = NAMES.getValue(piece.pieceType)
        val frm = sq(best.from)
        if (kind == "castle") {
            val king = board.king(us)
            return if (king != null) "Your king is still in the centre on ${sq(king)}" else null
        }
        if (kind == "rescue") {
            val attacker = attacker(board, best.from) ?: return null
            return "Your $name on $frm is attacked by their ${named(board, attacker)}"
        }
        if (kind == "prophylaxis" && threat != null) {
            val passed = board.copy(withStack = false)
            passed.push(Move.NULL)
            if (!passed.isLegal(threat)) return null
            if (threat.to == best.to) {
                // Your move takes the very square theirs wants: said by where their piece stands
                // ("their idea is b5" named the answer, b5, in a pawn ending).
                val mover = passed.pieceAt(threat.from) ?: return null
                val wants = if (mover.pieceType == PAWN) "wants to come forward" else "has a square in mind"
                return "Their ${named(passed, threat.from)} $wants: it is worth stopping before it comes"
            }
            return "Their idea is ${passed.san(threat)}: it is worth stopping before it comes"
        }
        if (kind == "develop") {
            val home = undeveloped(board, us)
            if (home.isEmpty()) return null
            val pieces = listed(home.map { named(board, it) })
            val verb = if (home.size == 1) "has" else "have"
            return "Your $pieces $verb not moved yet"
        }
        if (kind == "activateKing") return "This is an ending: your king on $frm belongs in the fight, not on the side"
        if (kind == "outpostMove") {
            return "There is a square in their half that no pawn of theirs can ever attack, and your " +
                "knight on $frm can get there"
        }
        if (kind == "takeFile") {
            val file = "abcdefgh"[squareFile(best.to)]
            return "The $file-file has no pawns of yours on it, and none of your rooks is there yet"
        }
        if ((kind == "pushPasser" || kind == "passerPawnEnding") && piece.pieceType == PAWN) {
            val rank = squareRank(best.from)
            val toGo = if (us == WHITE) 7 - rank else rank
            val away = "${IdeaWords.inWords(toGo)} square${if (toGo != 1) "s" else ""}"
            return "Your passed pawn on $frm is $away from queening"
        }
        if (kind == "rookBehindPasser") return "Your rook on $frm belongs behind a passed pawn, where it pushes it on"
        if (kind == "break") {
            val after = board.copy(withStack = false)
            after.push(best)
            val hit = squaresOf(after.attacksMask(best.to)).filter {
                after.pieceTypeAt(it) == PAWN && after.colourAt(it) != us
            }
            if (hit.isNotEmpty()) return "Your pawn on $frm can strike at their pawn on ${sq(hit[0])}"
            return null
        }
        if (kind == "centre" || kind == "space" || kind == "centralise" || kind == "activate") {
            val opened = shutIn(board, best)
            if (opened != null) {
                val (home, pawn) = opened
                return "Your ${named(board, home)} is still shut in behind the pawn on ${sq(pawn)}"
            }
            if (kind == "activate") {
                val own = if (us) 1 else 0
                val reach = popcount(board.attacksMask(best.from) and board.occupiedCo[own].inv())
                return "Your $name on $frm is your least active piece: it reaches only $reach squares"
            }
            if (kind == "space" && piece.pieceType == PAWN && phase != "endgame") {
                // Where, not which pawn: 14.Qe1 in Johner-Nimzowitsch, a4 "White's last chance to
                // make a fight of it".
                return "You can take space ${wing(best.from)}"
            }
            return null
        }
        if (kind == "pawnStorm") {
            val king = board.king(!us) ?: return null
            return "Their king on ${sq(king)} is on the other wing from yours: your pawns can roll at it"
        }
        if (kind == "simplify") return "You are ahead in material: every trade brings the win closer"
        return null
    }

    /** The line's plan, named on the board, as a lesson. */
    fun planKey(board: Board, plan: LinePlan): String? {
        val pieces = Plans.pieces(plan, board)
        val sq = plan.square?.let { sq(it) }
        if (plan.kind == "attack" && sq != null) {
            val who = if (!pieces.isNullOrEmpty()) ": $pieces can join in" else ""
            return "The plan is an attack on their king on $sq$who"
        }
        if (plan.kind == "storm" && sq != null) {
            return "The plan is a pawn storm on their king on $sq: with the kings on opposite " +
                "wings, it costs your own king nothing"
        }
        if (plan.kind == "double" && !plan.where.isNullOrEmpty()) {
            val detail = if (plan.detail.isNullOrEmpty()) "open" else plan.detail
            return "The $detail ${plan.where}-file is the way in: two of your heavy pieces belong on it"
        }
        if (plan.kind == "outpost" && sq != null) {
            val who = if (!pieces.isNullOrEmpty()) pieces else "a knight"
            return "$sq is a square no pawn of theirs can ever attack, and $who can reach it"
        }
        if (plan.kind == "passer" && sq != null) {
            return "Your passed pawn on $sq is the point: push it, and it ties their pieces down"
        }
        if (plan.kind == "break" && sq != null) return "The pawn break on $sq is the plan"
        if (plan.kind == "wing" && !plan.where.isNullOrEmpty()) {
            return "Your pawns on the ${plan.where} can take space, away from both kings"
        }
        return null
    }

    /** The target of the threat the best move makes, named: the lesson is to hit it. */
    fun threatKey(board: Board, threat: MoveThreat): String? {
        val target = board.pieceAt(threat.square)
        if (threat.kind == "mate") {
            return "Their king on ${sq(threat.square)} is short of squares: one quiet move can set up mate"
        }
        if (target == null || target.colour == board.turn) return null
        if (threat.kind == "trade") return "Their ${named(board, threat.square)} is worth trading off"
        return "Their ${named(board, threat.square)} can be hit"
    }

    /**
     * Their threat, named: the move they threaten and what it wins, or what it prepares. When
     * their move goes to the square the best move takes first, it is said by their piece: "they
     * are preparing e6" named the answer, e6 (Lichess vgdbX).
     */
    fun defendKey(board: Board, idea: Idea, best: Move? = null): String? {
        val threat = idea.threat ?: return null
        val passed = board.copy(withStack = false)
        passed.push(Move.NULL)
        if (!passed.isLegal(threat)) return null
        val san = passed.san(threat)
        if (best != null && threat.to == best.to) {
            val mover = named(passed, threat.from)
            if (idea.threatened == "mate") return "Their $mover is one move from a mating threat"
            return "Their $mover is one move from winning material"
        }
        if (idea.prepared && idea.follow != null) {
            val then = passed.copy(withStack = false)
            then.push(threat)
            then.push(Move.NULL)
            if (!then.isLegal(idea.follow)) return null
            val follow = then.san(idea.follow).trimEnd('#', '+')
            if (idea.threatened == "mate") return "They are preparing $san, and then $follow would be mate"
            val aim = board.pieceAt(idea.follow.to)
            val what = if (aim != null) "your ${named(board, idea.follow.to)}" else "material"
            return "They are preparing $san, and then $follow would win $what"
        }
        if (idea.threatened == "mate") return "They threaten ${san.trimEnd('#')} mate"
        val taken = board.pieceAt(threat.to)
        if (taken != null && taken.colour == board.turn) {
            return "They threaten $san, winning your ${named(board, threat.to)}"
        }
        return "They threaten $san"
    }

    /** Features that say a piece stands well where it is: not the point of a move that takes it away. */
    private val ASSETS = setOf("openFile", "halfOpenFile", "outpost")

    /**
     * A static feature of the board that the best move is about: one whose squares the move
     * itself touches. A square only a later move of the line touches sent the player to that
     * move first (15.Bd2 in Johner-Nimzowitsch, beside a4).
     */
    fun relevantFeature(board: Board, factsList: List<PositionalFact>, pv: List<Move>): String? {
        val touched = HashSet<String>()
        val left = HashSet<String>()
        for (move in pv.take(1)) {
            touched.add(sq(move.from))
            touched.add(sq(move.to))
            left.add(sq(move.from))
        }
        for (fact in factsList) {
            if (fact.kind in ASSETS && fact.squares.any { it in left }) continue
            if (fact.squares.any { it in touched }) return fact.text
        }
        return null
    }

    /**
     * Their king and its room, counted, and your checks: "their king on f1 has three free
     * squares, and you have two checks". The squares themselves come with the second hint: one
     * of them can be where your check lands (e2+ with e2 among the king's squares).
     */
    fun mateKey(board: Board, idea: Idea): String? {
        val target = idea.target ?: return null
        val room = idea.flight.size
        val squares = if (room == 0) {
            "no free square"
        } else if (room == 1) {
            "one free square"
        } else {
            "${NUMBER[room] ?: room.toString()} free squares"
        }
        val checks = idea.checks
        val said = if (checks > 1) {
            ", and you have ${NUMBER[checks] ?: checks.toString()} checks"
        } else if (checks == 1) {
            ", and you have one check"
        } else {
            ""
        }
        return "Their king on ${sq(target)} has $squares$said"
    }

    /**
     * What the way out of check does, at the move rung: takes the checker, blocks, or steps away
     * ("The move is Re4." said nothing). A king step says what sets its square apart only when
     * something does: every legal king move lands where nothing attacks it, so "a square nothing
     * of theirs can reach" read as a reason no other king move had (dev audit, 2026-09-26).
     */
    fun evasionWhy(board: Board, best: Move): String? {
        if (!board.isCheck()) return null
        val san = board.san(best)
        val checkers = board.checkersMask()
        if (checkers and BB_SQUARES[best.to] != BB_EMPTY) return "$san takes the piece giving check"
        if (board.pieceTypeAt(best.from) == KING) {
            val others = board.legalMoves().filter { it != best && board.pieceTypeAt(it.from) == KING }
            if (others.isNotEmpty() &&
                checksAfter(board, best) == 0 &&
                others.all { checksAfter(board, it) != 0 }
            ) {
                return "$san steps out of check to the one square where no check follows"
            }
            return "$san steps out of check"
        }
        return "$san blocks the check"
    }

    /** The checks the opponent has once [move] is played. `situation._checks_after`. */
    private fun checksAfter(board: Board, move: Move): Int {
        val after = board.copy(withStack = false)
        after.push(move)
        return after.legalMoves().count { after.givesCheck(it) }
    }

    /** The square of the enemy piece the player's move at `ply` takes, as it stands now. */
    private fun captured(board: Board, line: List<Move>, ply: Int): Int? {
        val b = board.copy(withStack = false)
        for ((j, move) in line.withIndex()) {
            if (!b.isLegal(move)) return null
            if (j == ply) {
                if (!b.isCapture(move)) return null
                // Followed back to where it stands now, through its own moves; an en passant
                // capture takes the pawn beside the square it lands on.
                var at = move.to
                if (b.isEnPassant(move)) at = square(squareFile(move.to), squareRank(move.from))
                for (k in j - 1 downTo 0) {
                    if (k % 2 == 1 && line[k].to == at) at = line[k].from
                }
                val piece = board.pieceAt(at)
                return if (piece != null && piece.colour != board.turn) at else null
            }
            b.push(move)
        }
        return null
    }

    /** What a win without a named pattern takes: "their pawn on d5 is the target". */
    fun winKey(board: Board, idea: Idea): String? {
        val line = idea.line
        val plies = listOf(idea.gainPly) + (0 until minOf(line.size, 8) step 2).toList()
        for (ply in plies) {
            val s = captured(board, line, ply) ?: continue
            return "Their ${named(board, s)} is the target"
        }
        return null
    }

    /** The trade that leaves the won ending: the first of theirs the player's line takes. */
    fun convertKey(board: Board, idea: Idea): String? {
        val line = idea.line
        for (ply in 0 until minOf(line.size, 8) step 2) {
            val s = captured(board, line, ply) ?: continue
            return "Their ${named(board, s)} is the piece to trade off"
        }
        return null
    }

    /** An ending the king decides: the kings face to face, or their pawns in its reach. */
    fun kingKey(board: Board, idea: Idea): String? {
        val us = board.turn
        val ours = board.king(us) ?: return null
        val theirs = board.king(!us) ?: return null
        if (idea.opposition) return "The kings stand close: your king on ${sq(ours)}, theirs on ${sq(theirs)}"
        val pawns = board.pieces(PAWN, !us).sortedBy { squareDistance(ours, it) }
        if (pawns.isNotEmpty()) {
            val near = listed(pawns.take(2).map { sq(it) })
            return "Your king on ${sq(ours)} is your best piece here: their pawns on $near are in its reach"
        }
        return "Your king on ${sq(ours)} is your best piece here"
    }

    /** The piece giving check -- and, when it is what the line wins (`prize`), said so once. */
    fun checkKey(board: Board, prize: Int? = null): String? {
        val checkers = squaresOf(board.checkersMask())
        if (checkers.size == 1) {
            if (checkers[0] == prize) return "Their ${named(board, checkers[0])} gives check, and it is what you win"
            return "Their ${named(board, checkers[0])} gives check"
        }
        if (checkers.size == 2) {
            val first = named(board, checkers[0])
            val second = named(board, checkers[1])
            return "Their $first and their $second both give check: only a king move gets out"
        }
        return null
    }

    /**
     * What they would do if you let them: the prophylactic question, answered on the board.
     * Where every other move loses (`sharp`), whatever it is; else only a move that changes
     * something -- a capture, a check, a pawn move: "if you let them, their best is Bd7" said
     * nothing (Johner-Nimzowitsch, eight times).
     */
    fun theirIdeaKey(board: Board, threat: Move?, sharp: Boolean = false): String? {
        if (threat == null || board.isCheck()) return null
        val passed = board.copy(withStack = false)
        passed.push(Move.NULL)
        if (!passed.isLegal(threat)) return null
        if (!(sharp ||
                passed.isCapture(threat) ||
                passed.givesCheck(threat) ||
                passed.pieceTypeAt(threat.from) == PAWN)
        ) {
            return null
        }
        return "If you let them, their best is ${passed.san(threat)}"
    }

    /**
     * A pawn ending's king move with the kings close: the standoff, named (Lichess R8gGn, ...Ke6
     * taking the opposition).
     */
    fun kingsKey(board: Board, best: Move): String? {
        if (board.pieceMap().values.any { it.pieceType != KING && it.pieceType != PAWN }) return null
        if (board.pieceTypeAt(best.from) != KING) return null
        val ours = board.king(board.turn) ?: return null
        val theirs = board.king(!board.turn) ?: return null
        if (squareDistance(ours, theirs) > 3) return null
        return "The kings stand close, yours on ${sq(ours)} and theirs on ${sq(theirs)}: whichever " +
            "has to give way loses ground"
    }

    /**
     * Their passed pawn near queening, when the best move takes it: what makes the capture the
     * point (Lichess UkpFL, Rxe3+ against a pawn two squares from e1).
     */
    fun passerKey(board: Board, best: Move): String? {
        val s = best.to
        if (!board.isCapture(best) || board.pieceTypeAt(s) != PAWN) return null
        if (board.colourAt(s) == board.turn || !isPassed(board, s)) return null
        val rank = squareRank(s)
        val toGo = if (board.turn == WHITE) rank else 7 - rank
        if (toGo > 3) return null
        val squares = if (toGo == 1) "one square" else "${NUMBER[toGo] ?: toGo.toString()} squares"
        return "Their passed pawn on ${sq(s)} is $squares from queening"
    }
}
