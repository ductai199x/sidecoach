package com.sidecoach.coach

import com.sidecoach.analysis.Exchange
import com.sidecoach.analysis.Motifs
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.chess.squareDistance
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareRank
import com.sidecoach.events.Analysis
import com.sidecoach.events.MotifTag
import kotlin.math.abs

/**
 * What the best line is for: the goal it reaches, the blow that reaches it, and its target. A
 * port of `sidecoach/coach/idea.py`.
 *
 * Stockfish gives the best line and its score; the tagger reads what each move does. The coach
 * named the idea from the position's static features and the line's first move alone, so a
 * fork on the player's second move was never named -- a static pin elsewhere on the board
 * took the headline -- and the first word on a mate in two was "what is your opponent's most
 * dangerous idea?" (Tai's puzzles, 2026-09-23). This reads the line through the player's own
 * moves: the goal (a mate in N, the material won, a promotion, a threat to meet), the tactic
 * that delivers it on whichever move of the line it comes, how the line starts (a check, a
 * trade, a sacrifice), and the target (their king, or what falls).
 *
 * No square here is named by the coach before the disclosure level allows it: [factsFromAnalysis]
 * turns this into words, each at its level ([IdeaWords]).
 */

/** Pawns' worth by piece type; the king is not counted (python's `VALUES.get(kind, 0)`). */
private val IDEA_VALUES = intArrayOf(0, 1, 3, 3, 5, 9, 0)
private val IDEA_COUNTED = intArrayOf(QUEEN, ROOK, BISHOP, KNIGHT, PAWN)

/**
 * The point of the best line, from the side to move's view: `idea.Idea`.
 *
 * Squares are board indices (python-chess's `chess.Square`); [blow]'s squares are as they
 * stand on [blowBoard].
 */
data class Idea(
    /** "mate", "win", "promote", "attack", "combination", "draw", "convert", "king", "breakthrough", "defend", "plan". */
    val goal: String,
    val first: Move?,
    val line: List<Move> = emptyList(),
    val mateIn: Int? = null,
    val gain: Int = 0,
    /** "their queen", "a rook", "the exchange", "a knight and a pawn". */
    val won: String? = null,
    val blow: MotifTag? = null,
    val blowMove: Move? = null,
    val blowBoard: Board? = null,
    /** Which of the player's moves carries the blow: 1 is the first. */
    val blowAt: Int = 1,
    /**
     * "win": the ply of the player's move that first wins it (0 is the first move): O-O and then
     * Nxd4 wins the pawn with the second move, not the first (Tai's game, 2026-09-24).
     */
    val gainPly: Int = 0,
    /** PREPARATION tags of the moves before the blow. */
    val preparation: List<String> = emptyList(),
    /** "check", "capture", "sacrifice", "promotion", "quiet". */
    val startsWith: String = "quiet",
    /** Their king (mate), or what falls, on the board now. */
    val target: Int? = null,
    /** "defend": the opponent's move if the player passed. */
    val threat: Move? = null,
    /** "mate", or the piece the threat wins: "your rook". */
    val threatened: String? = null,
    /**
     * "defend": the threat is a quiet move that would threaten it next, [follow] the move that
     * would then mate or take (Rh1, then Qh8#: the master games, 2026-09-24).
     */
    val prepared: Boolean = false,
    val follow: Move? = null,
    /** "mate": their king's free squares now. */
    val flight: List<Int> = emptyList(),
    /** How many checks the side to move has. */
    val checks: Int = 0,
    /** The line only holds the balance: what it wins, it wins back. */
    val level: Boolean = false,
    /** The player is behind in material now. */
    val down: Boolean = false,
    /** "draw": "perpetual", "stalemate" or "repetition". */
    val draw: String? = null,
    /**
     * A won endgame's technique: "convert" the ending traded into ("pawn", "rook", ...); "king"
     * whether the king move takes the opposition; "breakthrough" the pawn that runs.
     */
    val ending: String? = null,
    val opposition: Boolean = false,
    val runner: Int? = null,
    /**
     * Why it works, beyond the pattern: a piece of theirs that could take the striking piece
     * but is pinned (the "pin" of Lichess's fork puzzles), where it stands when the blow lands;
     * and whether the line holds an in-between move (a check before taking back, Lichess
     * "intermezzo").
     */
    val pinnedGuard: Int? = null,
    val intermezzo: Boolean = false,
    /**
     * A second tactic the line also cashes in, besides the blow: the fork that is also a
     * discovered attack (Lichess tags both; the coach named one).
     */
    val also: String? = null,
    /**
     * The player's sacrifices on the way to the point: (ply, what is given), "queen", "rook",
     * "exchange", "bishop", "knight", "pawn" -- material the reply keeps.
     */
    val sacrifices: List<Pair<Int, String>> = emptyList(),
    /** The engine calls the position won. */
    val winning: Boolean = false,
    /** The best line's expected score, for the player. */
    val score: Double? = null,
    /**
     * "win": what the line takes is mostly what the exchange in progress took from the player
     * ([IdeaReader.exchangeOwed]) -- won back, not won: a knight taken back after they took one
     * is a trade (Tai's puzzles, 2026-09-26: "you can win a knight" for 3.Nxh5 Rxh5).
     */
    val back: Boolean = false,
)

object IdeaReader {

    /**
     * Won and level, by the engine's expected score. `Analysis.decisive` is not "won": it marks
     * a critical moment, where only one move holds -- an equality puzzle is exactly that, and
     * reading it as won said "you can win their queen" and "a winning attack" of positions the
     * engine scored level (Lichess equality puzzles, 2026-09-23).
     */
    const val WON = 0.8
    const val LEVEL = 0.62

    /**
     * A line that nets this much (pawns' worth) within its first HORIZON plies wins material; a
     * pawn is a win too when the position is won (a pawn up into a won ending is the point).
     */
    const val WIN_GAIN = 2
    const val HORIZON = 16

    /**
     * Below this expected score, material the opponent hands over is theirs to give, not the
     * player's win ([firstGift]).
     */
    const val GIFT_SCORE = 0.3

    /**
     * A won position whose line checks this often in the player's first four moves, winning no
     * material in sight, is an attack on the king: "no tactic here" was said of a rook
     * sacrifice that opened one (Lichess 0199s, 2026-09-23).
     */
    const val ATTACK_CHECKS = 2

    /** The furthest of the player's moves a named pattern may land on and still be the idea. */
    const val FAR_BLOW = 3

    /** The named mating patterns, which say more than "a mating attack". */
    val MATE_PATTERNS: List<String> = listOf(
        "smotheredMate",
        "backRankMate",
        "anastasiaMate",
        "arabianMate",
        "hookMate",
        "bodenMate",
        "doubleBishopMate",
        "dovetailMate",
        "vukovicMate",
        "killBoxMate",
        "swallowstailMate",
        "epauletteMate",
        "triangleMate",
        "blindSwineMate",
        "cornerMate",
        "operaMate",
        "pillsburysMate",
        "morphysMate",
        "balestraMate",
    )

    /** The blow, the most telling first: what a coach names as the point of the line. */
    val BLOW_RANK: List<String> = MATE_PATTERNS + listOf(
        "doubleCheck",
        "fork",
        "skewer",
        "discoveredAttack",
        "absolutePin",
        "pin",
        "trappedPiece",
        "capturingDefender",
        "deflection",
        "attraction",
        "hangingPiece",
        "promotion",
        "mate",
    )
    private val BLOW_INDEX: Map<String, Int> = BLOW_RANK.withIndex().associate { (i, name) -> name to i }

    /**
     * What a mate is named by: its pattern, or two checks at once. A pin or a skewer that
     * happens to be on the board as the mate lands is not its point ("It's a pin" was said of
     * Qh7+ Kf6 Qxh6#, Lichess 01xbu). The tactics aimed at a piece, the blow only when the line
     * then takes that piece, are the tagger's ([Motifs.TARGETED], [Motifs.paysOff]).
     */
    val MATE_BLOWS: List<String> = MATE_PATTERNS + listOf("doubleCheck")

    /** What can clear a pawn's way or hold the defence off, for a promotion. */
    val PROMOTION_BLOWS: List<String> = listOf(
        "discoveredAttack",
        "deflection",
        "capturingDefender",
        "absolutePin",
        "pin",
        "attraction",
        "skewer",
        "fork",
        "doubleCheck",
    )

    /** The tactics that make a pawn won in a decided position the point of the line. */
    val PRIMARY: List<String> = listOf(
        "doubleCheck",
        "fork",
        "skewer",
        "discoveredAttack",
        "absolutePin",
        "pin",
        "trappedPiece",
    )

    /** What a move before the blow does to prepare it. */
    val PREPARATION: List<String> = listOf("sacrifice", "deflection", "attraction", "capturingDefender", "clearance")

    /** The tactics worth naming beside the blow: the ones a player learns as patterns. */
    val ALSO: List<String> = listOf("fork", "skewer", "discoveredAttack", "doubleCheck", "pin", "absolutePin", "trappedPiece")

    private val NAMES = mapOf(
        QUEEN to "queen",
        ROOK to "rook",
        BISHOP to "bishop",
        KNIGHT to "knight",
        PAWN to "pawn",
    )
    private val PLURALS = mapOf(
        "queen" to "queens",
        "rook" to "rooks",
        "bishop" to "bishops",
        "knight" to "knights",
        "pawn" to "pawns",
    )
    private val NUMBERS = mapOf(1 to "a", 2 to "two", 3 to "three", 4 to "four")

    private fun value(pieceType: Int): Int = IDEA_VALUES[pieceType]

    /** The boards along `pv` from `board`: `boards[k]` is the position before `pv[k]`. */
    private fun boardsOf(board: Board, pv: List<Move>): List<Board> {
        val boards = ArrayList<Board>(pv.size + 1)
        boards.add(board.copy(withStack = false))
        for (move in pv) {
            val after = boards.last().copy(withStack = false)
            after.push(move)
            boards.add(after)
        }
        return boards
    }

    /**
     * The exchange the last move belongs to, from the side to move's view: the run of captures
     * that ends with the opponent's last move, counted per kind of piece -- how many more of the
     * side to move's it took than of the opponent's. Returned only when that leaves the side to
     * move owed material (they took last; it is the player's to take back): taking it back is
     * winning back, not winning. Empty when the last move took nothing or the board does not
     * know its moves (a position set up from a FEN). `idea.exchange_owed`.
     */
    fun exchangeOwed(board: Board): Map<Int, Int> {
        if (board.moveStack.isEmpty()) return emptyMap()
        val us = board.turn
        val before = board.copy()
        val net = LinkedHashMap<Int, Int>()
        while (before.moveStack.isNotEmpty()) {
            val move = before.pop()
            if (!before.isCapture(move)) break
            val victim: Int
            if (before.isEnPassant(move)) {
                victim = PAWN
            } else {
                victim = before.pieceTypeAt(move.to)
                if (victim == 0) break
            }
            // Popped, the board has the capturer to move: the opponent took one of the player's.
            net[victim] = (net[victim] ?: 0) + (if (before.turn != us) 1 else -1)
        }
        val owed = net.filterValues { it != 0 }
        return if (worth(owed) > 0) owed else emptyMap()
    }

    /** What a count of pieces by kind is worth, in pawns. `idea.worth`. */
    fun worth(pieces: Map<Int, Int>): Int = pieces.entries.sumOf { (kind, n) -> value(kind) * n }

    /**
     * The idea of `analysis`'s best line from `board` (the position it measured), with the
     * sacrifices the player makes on the way to its point. [owed] is what the exchange in
     * progress took from the player ([exchangeOwed], read from the board's own moves when not
     * given): material the line takes up to that is won back, not won.
     */
    fun read(board: Board, analysis: Analysis, owed: Map<Int, Int>? = null): Idea {
        val idea = readLine(board, analysis, owed ?: exchangeOwed(board))
        if (idea.first == null) return idea
        val pv = idea.line
        val boards = boardsOf(board, pv)
        var upto: Int
        if (idea.goal == "mate" && idea.mateIn != null && idea.mateIn != 0) {
            upto = 2 * idea.mateIn - 1
        } else if (idea.goal == "win" || idea.goal == "combination") {
            upto = if (idea.blow != null) 2 * idea.blowAt else HORIZON
            val (gain, at) = lineGain(boards, pv, minOf(pv.size, HORIZON))
            upto = maxOf(upto, if (gain != 0) firstReaching(boards, pv, at, minOf(gain, WIN_GAIN)) else 0)
        } else if (idea.goal == "attack" || idea.goal == "promote") {
            upto = 8
        } else {
            upto = 2
        }
        val top = analysis.lines[0]
        val mated = top.mate != null && top.mate > 0
        val winning = mated || top.expectedScore >= WON
        return idea.copy(
            pinnedGuard = pinnedGuard(idea, boards, pv),
            also = alsoTactic(idea, boards, pv),
            intermezzo = intermezzo(boards, pv, minOf(pv.size, 10)),
            sacrifices = sacrifices(boards, pv, upto),
            winning = winning,
            level = top.expectedScore < LEVEL && !mated,
            down = balance(board, board.turn) < 0,
            score = top.expectedScore,
        )
    }

    /** `idea._read`: the goal, the blow and the target, before the sacrifices are counted. */
    private fun readLine(board: Board, analysis: Analysis, owed: Map<Int, Int>): Idea {
        val pv = if (analysis.lines.isNotEmpty()) legalPrefix(board, analysis.lines[0].moves) else emptyList()
        val us = board.turn
        val checks = board.legalMoves().count { board.givesCheck(it) }
        if (pv.isEmpty()) return Idea(goal = "plan", first = null, checks = checks)
        val boards = boardsOf(board, pv)
        val first = pv[0]
        val startsWith = start(board, boards, pv)
        val mate = analysis.lines[0].mate
        if (mate != null && mate > 0) {
            val end = minOf(pv.size, 2 * mate - 1)
            val (blow, blowAt, preparation) = findBlow(boards, pv, end, mate, MATE_BLOWS)
            val king = board.king(!us)
            return Idea(
                goal = "mate",
                first = first,
                line = pv,
                mateIn = mate,
                blow = blow?.second,
                blowMove = if (blow != null) pv[blow.first] else null,
                blowBoard = if (blow != null) boards[blow.first] else null,
                blowAt = blowAt,
                preparation = preparation,
                startsWith = startsWith,
                target = king,
                flight = flightOf(board, !us),
                checks = checks,
            )
        }
        val second = if (analysis.lines.size > 1) analysis.lines[1].expectedScore else null
        val draw = drawn(boards, pv, analysis.lines[0].expectedScore, second)
        if (draw != null) {
            return Idea(
                goal = "draw",
                first = first,
                line = pv,
                startsWith = startsWith,
                target = board.king(!us),
                checks = checks,
                draw = draw,
            )
        }
        val horizon = minOf(pv.size, HORIZON)
        // The player queening: the material before it is what a win wins; after it, what the
        // new queen does, and the pawn is the point (a race won with Kxb8, Lichess 02YZr).
        var queened: Int? = null
        for (j in 0 until horizon) {
            if (j % 2 == 0 && pv[j].promotion != 0) {
                queened = j + 1
                break
            }
        }
        var reach = if (queened != null && queened != 0) queened - 1 else horizon
        // In a position the engine calls lost, what they give up on the way is given for their
        // attack, not won: counted only up to their first piece left to be taken ("you can win
        // back a knight and a pawn" at 29.exf4 in Johner-Nimzowitsch, the knight Ng4+ given to
        // open the king).
        if (analysis.lines[0].expectedScore < GIFT_SCORE) reach = minOf(reach, firstGift(boards, pv, reach))
        var (gain, at) = lineGain(boards, pv, reach)
        val peak = at
        // A pawn of the player's reaching the seventh, or queening, anywhere in the line.
        val seventh = if (us == WHITE) 6 else 1
        // A pawn of the player's queening in the line; one reaching the seventh only in an
        // ending, where it is on its way: in a middlegame a pawn on the seventh is a capture on
        // f7 in an attack ("you can push a pawn through to promotion" was said of exf7 in
        // Gligoric-Fischer, move 17).
        val ending = piecesLeft(board) <= 4
        var queens = false
        for (j in 0 until horizon) {
            if (j % 2 != 0) continue
            val move = pv[j]
            if (move.promotion != 0 ||
                (ending && boards[j].pieceTypeAt(move.from) == PAWN && squareRank(move.to) == seventh)
            ) {
                queens = true
                break
            }
        }
        // A pawn is a win when the game is better for it; two pawns' worth always.
        val betterForIt = analysis.lines[0].expectedScore >= LEVEL
        // What they just took counts against the line ([exchangeOwed]): what it nets decides
        // whether it wins, by the same rule as any win; a line that nets less only wins back.
        val net = gain - worth(owed)
        var back = owed.isNotEmpty() &&
            gain > 0 &&
            !(net >= WIN_GAIN || (net >= 1 && betterForIt && !queens))
        // The blow is what wins it: looked for up to where the line first wins that much, not
        // where it has picked up the most (a fork five moves after the queen fell was named). A
        // net win is first won where the net is reached, past taking back what they took.
        val winsAt = if (owed.isNotEmpty() && !back) worth(owed) + minOf(net, WIN_GAIN) else minOf(gain, WIN_GAIN)
        var firstAt = if (gain > 0) firstReaching(boards, pv, at, winsAt) else 0
        if (owed.isNotEmpty() && !back && firstAt > 2 * FAR_BLOW) {
            // A net win first reached past the player's third move is the engine's line talking,
            // not the point: taking back is ("you can win a queen for a rook and a bishop" of a
            // queen given up on the fourteenth ply after 1...Nxe3 2.fxe3, Lichess 00LZf).
            back = true
            firstAt = firstReaching(boards, pv, at, minOf(gain, WIN_GAIN))
        }
        at = if (firstAt != 0) firstAt else at
        val gained = gainedKinds(board, if (at != 0) lineEnd(boards, pv, at) else boards.last(), us)
        var (blow, blowAt, preparation) = findBlow(
            boards, pv, if (at != 0) at else minOf(pv.size, 6), null, gained = gained,
        )
        if (blow != null && blowAt > FAR_BLOW) {
            // A loose rook picked up on the fifth move is not the idea a player can see from
            // here; the start of the line is (Tai's game g06, 2026-09-24).
            blow = null
            blowAt = 1
            preparation = emptyList()
        }
        val winning = analysis.lines[0].expectedScore >= WON
        // A win the score denies: the engine calls the game level, the player is not behind,
        // and the material comes only down the line -- "you can win a knight and two pawns for
        // a bishop" was said at move 6 of the Ruy Lopez (Capablanca-Marshall, 2026-09-24).
        val firstWin = if (gain > 0) maxOf(0, if ((at - 1) % 2 == 0) at - 1 else at - 2) else 0
        val topMate = analysis.lines[0].mate
        val phantom = !betterForIt &&
            balance(board, us) + worth(owed) >= 0 &&
            firstWin > 0 &&
            !(topMate != null && topMate > 0)
        if (!phantom && (gain >= WIN_GAIN || (gain >= 1 && betterForIt && !queens))) {
            // A pawn is a win too when the position turns on it and a real tactic takes it: a
            // discovered check that nets a pawn into a won ending is still the point. A line
            // that stops on the player's capture: what takes back is counted as given, as
            // `lineGain` counts it ("you win the exchange", not "a rook", after Nxe1).
            var (won, target) = wonBy(board, lineEnd(boards, pv, at), us, pv.subList(0, at), boards)
            if (back) {
                // Won back: the most valuable piece the line takes within the player's first
                // FAR_BLOW moves ("their queen" where they took the queen and the line takes a
                // bishop and then theirs); never one far down the line (00LZf's queen at ply 14).
                val near = minOf(peak, 2 * FAR_BLOW, pv.size)
                won = topTaken(board, lineEnd(boards, pv, near), us) ?: won
            } else if (owed.isNotEmpty()) {
                // A net win, named net: where they took a bishop and the line takes a knight and
                // two pawns, the win is two pawns (Tai, 2026-09-26: "it's not winning shit").
                val (netted, where) = wonBy(board, lineEnd(boards, pv, at), us, pv.subList(0, at), boards, owed)
                if (netted != null) {
                    won = netted
                    target = where
                }
            }
            // The player's move that first wins it: the last of theirs within `at` plies.
            val gainPly = maxOf(0, if ((at - 1) % 2 == 0) at - 1 else at - 2)
            return Idea(
                goal = "win",
                first = first,
                line = pv,
                gain = gain,
                gainPly = gainPly,
                won = won,
                back = back,
                blow = blow?.second,
                blowMove = if (blow != null) pv[blow.first] else null,
                blowBoard = if (blow != null) boards[blow.first] else null,
                blowAt = blowAt,
                preparation = preparation,
                startsWith = startsWith,
                target = target,
                checks = checks,
            )
        }
        if (queens) {
            // The pawn queening is the point, anywhere in the line, or the first move takes it
            // to the seventh: what holds the defence off is the blow, when there is one -- a
            // pin, a deflection; never a discovered attack on some pawn elsewhere.
            val upto = queened ?: minOf(pv.size, 6)
            val promoted = findBlow(boards, pv, upto, null, PROMOTION_BLOWS)
            blow = promoted.first
            blowAt = promoted.second
            preparation = promoted.third
            return Idea(
                goal = "promote",
                first = first,
                line = pv,
                blow = blow?.second,
                blowMove = if (blow != null) pv[blow.first] else null,
                blowBoard = if (blow != null) boards[blow.first] else null,
                blowAt = blowAt,
                preparation = preparation,
                startsWith = startsWith,
                checks = checks,
            )
        }
        var checksGiven = 0
        for (j in 0 until minOf(pv.size, 8) step 2) if (boards[j].givesCheck(pv[j])) checksGiven += 1
        // Won, or clearly better with material given for the checks: a sacrificial attack at
        // 0.78 was "which of your pieces has no job right now?" (Lichess PZm0M).
        val better = analysis.lines[0].expectedScore >= 0.65
        val theirKing = board.king(!us)
        val nearKing = theirKing != null &&
            (0 until minOf(pv.size, 8) step 2).any { squareDistance(pv[it].to, theirKing) <= 2 }
        val opened = better && sacrifices(boards, pv, 6).isNotEmpty() && (checksGiven >= 1 || nearKing)
        if ((winning && checksGiven >= ATTACK_CHECKS) || opened) {
            // The king is the target: what the checks open up is the point.
            return Idea(
                goal = "attack",
                first = first,
                line = pv,
                blow = blow?.second,
                blowMove = if (blow != null) pv[blow.first] else null,
                blowBoard = if (blow != null) boards[blow.first] else null,
                blowAt = blowAt,
                preparation = preparation,
                startsWith = startsWith,
                target = board.king(!us),
                flight = flightOf(board, !us),
                checks = checks,
            )
        }
        // A won position with nothing to win in sight is won by technique, never "no tactic
        // here -- gain space on the flank" in a king-and-pawn ending (Lichess pawnEndgame,
        // zugzwang and quietMove puzzles, 2026-09-23): the trade into a won ending, the king
        // leading, a pawn breakthrough.
        if (winning) {
            val technique = technique(board, boards, pv)
            if (technique != null) return technique.copy(startsWith = startsWith, checks = checks)
        }
        // Material given and taken back by a tactic, the position better for it: a combination,
        // never "no tactic here" (an exchange sacrifice and a skewer that wins the rook back was
        // read as a rook behind a passed pawn, Lichess 03gJT).
        val given = sacrifices(boards, pv, minOf(pv.size, 8))
        val primary = blow != null && blow.second.name in PRIMARY
        if (given.isNotEmpty() && primary && analysis.lines[0].expectedScore >= 0.6) {
            return Idea(
                goal = "combination",
                first = first,
                line = pv,
                gain = gain,
                blow = blow?.second,
                blowMove = if (blow != null) pv[blow.first] else null,
                blowBoard = if (blow != null) boards[blow.first] else null,
                blowAt = blowAt,
                preparation = preparation,
                startsWith = startsWith,
                checks = checks,
            )
        }
        val (threat, threatened, follow) = threatOf(board, analysis.threats, first)
        if (threat != null) {
            return Idea(
                goal = "defend",
                first = first,
                line = pv,
                startsWith = startsWith,
                threat = threat,
                threatened = threatened,
                prepared = follow != null,
                follow = follow,
                checks = checks,
            )
        }
        return Idea(goal = "plan", first = first, line = pv, startsWith = startsWith, checks = checks)
    }

    /**
     * The best move lets their direct threat come: it gives check (the attack goes on, and the
     * capture waits), or after it the capture is still there and still wins. Black's 14...Bh2+
     * in Capablanca-Marshall, with hxg4 hanging over the knight, was "your opponent threatens to
     * win material: meet it first".
     */
    private fun ignores(board: Board, best: Move, threat: Move): Boolean {
        if (board.givesCheck(best)) return true
        val after = board.copy(withStack = false)
        after.push(best)
        return after.isLegal(threat) && Exchange.exchange(after, threat) > 0
    }

    /** An ending: four pieces or fewer left besides kings and pawns. `idea.ending_of`. */
    fun endingOf(board: Board): Boolean = piecesLeft(board) <= 4

    /** Pieces other than kings and pawns, both sides. */
    private fun piecesLeft(board: Board): Int =
        board.pieceMap().values.count { it.pieceType != KING && it.pieceType != PAWN }

    /**
     * The kind of ending on `board`: "pawn", "rook", "queen", "bishop", "knight", or
     * "minor-piece" / "mixed".
     */
    private fun endingName(board: Board): String {
        val kinds = board.pieceMap().values.map { it.pieceType }.toSet() - setOf(KING, PAWN)
        if (kinds.isEmpty()) return "pawn"
        if (kinds.size == 1) return NAMES.getValue(kinds.first())
        if (kinds.all { it == BISHOP || it == KNIGHT }) return "minor-piece"
        return "mixed"
    }

    /**
     * How a won position with no material in sight is won: trading into a won ending (the
     * player's first moves take pieces off, material stays), the king leading in an ending
     * (with the opposition when the kings stand face to face, one square apart, and they must
     * move), or a pawn breakthrough in a pawn ending (a pawn given so another runs to the
     * seventh). Null when the line is none of these.
     */
    private fun technique(board: Board, boards: List<Board>, pv: List<Move>): Idea? {
        val us = board.turn
        val first = pv[0]
        val startPieces = piecesLeft(board)
        val later = boards[minOf(boards.size - 1, 4)]
        val levelMaterial = balance(later, us) >= balance(board, us) - 1
        // Pieces coming off in the first two moves each: Rh2+ Kd3 Rxc2 Kxc2 trades too.
        val trades = (0 until minOf(pv.size, 4)).any { boards[it].isCapture(pv[it]) }
        if (startPieces != 0 && piecesLeft(later) < startPieces && levelMaterial && trades) {
            return Idea(goal = "convert", first = first, line = pv, ending = endingName(later))
        }
        val endgame = startPieces <= 2 &&
            (board.piecesMask(QUEEN, WHITE) or board.piecesMask(QUEEN, !WHITE)) == BB_EMPTY
        val mover = board.pieceTypeAt(first.from)
        if (endgame && mover == KING) {
            val after = boards[1]
            val mine = after.king(us)
            val theirs = after.king(!us)
            var opposition = false
            if (mine != null && theirs != null) {
                val df = abs(squareFile(mine) - squareFile(theirs))
                val dr = abs(squareRank(mine) - squareRank(theirs))
                opposition = (df == 0 && dr == 2) || (dr == 0 && df == 2)
            }
            return Idea(goal = "king", first = first, line = pv, opposition = opposition)
        }
        if (startPieces == 0 && mover == PAWN) {
            val seventh = if (us == WHITE) 6 else 1
            for (j in 2 until minOf(pv.size, HORIZON) step 2) {
                val move = pv[j]
                if (boards[j].pieceTypeAt(move.from) == PAWN &&
                    (squareRank(move.to) == seventh || move.promotion != 0)
                ) {
                    val given = if (pv.size > 1) boards[1].isCapture(pv[1]) else false
                    if (given && pv[1].to == first.to) {
                        return Idea(goal = "breakthrough", first = first, line = pv, runner = move.from)
                    }
                    break
                }
            }
        }
        return null
    }

    /**
     * A second tactic of a different kind that the line also cashes in, on the player's moves
     * up to the blow: named beside it, never instead of it.
     */
    private fun alsoTactic(idea: Idea, boards: List<Board>, pv: List<Move>): String? {
        val blow = idea.blow ?: return null
        if (idea.goal !in listOf("win", "promote", "combination", "attack")) return null
        val upto = 2 * (idea.blowAt - 1)
        val main = if (blow.name == "absolutePin") "pin" else blow.name
        for (i in 0..upto step 2) {
            if (i >= pv.size) break
            for (tag in Motifs.tagsForMove(boards[i], pv[i], pv.subList(i + 1, pv.size), null)) {
                val name = if (tag.name == "absolutePin") "pin" else tag.name
                if (name in ALSO && name != main && Motifs.paysOff(boards, pv, i, tag)) return tag.name
            }
        }
        return null
    }

    /**
     * A piece of theirs that bears on the square the blow (or the first move) lands on but
     * cannot take, being pinned to its own king -- and whose capture would otherwise have won:
     * the reason the knight on f3 can fork with a pawn beside it.
     */
    private fun pinnedGuard(idea: Idea, boards: List<Board>, pv: List<Move>): Int? {
        val before: Board
        val move: Move
        if (idea.blowMove != null && idea.blowBoard != null) {
            before = idea.blowBoard
            move = idea.blowMove
        } else {
            before = boards[0]
            move = pv[0]
        }
        val after = before.copy(withStack = false)
        after.push(move)
        val mine = after.pieceAt(move.to) ?: return null
        for (guard in after.attackers(!mine.colour, move.to)) {
            val kind = after.pieceTypeAt(guard)
            if (kind == 0 || kind == KING ||
                after.pinMask(!mine.colour, guard) and BB_SQUARES[move.to] != BB_EMPTY
            ) {
                continue
            }
            val cheaper = value(kind) < value(mine.pieceType)
            if (cheaper || after.attackers(mine.colour, move.to).isEmpty()) return guard
        }
        return null
    }

    /**
     * The player, after one of theirs is taken, first plays a check elsewhere and only then
     * takes back on that square: an in-between move.
     */
    private fun intermezzo(boards: List<Board>, pv: List<Move>, upto: Int): Boolean {
        val us = boards[0].turn
        for (j in 2 until upto - 2 step 2) {
            val taken = pv[j - 1]
            val victim = boards[j - 1].pieceAt(taken.to)
            if (victim == null || victim.colour != us) continue
            val move = pv[j]
            if (move.to == taken.to || !boards[j].givesCheck(move)) continue
            if (pv[j + 2].to == taken.to && boards[j + 2].isCapture(pv[j + 2])) return true
        }
        return false
    }

    /**
     * The side to move on `board` can legally take on `square`: a recapture that exists, not
     * one a pinned piece or a king walking into a guarded square would make.
     */
    private fun retaken(board: Board, square: Int): Boolean = board.legalMoves().any { it.to == square }

    /** How much worse the next-best move must be for a draw to be the point: the draw saves the game. */
    const val RESCUE = 0.15

    /**
     * How the line saves the game, when it is a draw the player forces: stalemate on the
     * board, check after check with the game level (a perpetual), or a position that comes
     * round again. Null when the line is not that -- or when the draw saves nothing: the
     * next-best move (`second`, its expected score) does as well, and a level line that goes
     * round in a circle is the engine marking time, not a rescue ("you can hold the game" was
     * said at 80 of the master games' 93 such moves, all level whatever was played,
     * 2026-09-24).
     */
    internal fun drawn(boards: List<Board>, pv: List<Move>, score: Double, second: Double? = null): String? {
        if (!(score >= 0.35 && score <= 0.65)) return null
        if (second != null && second > score - RESCUE) return null
        if (boards.drop(1).any { it.isStalemate() }) return "stalemate"
        val ours = (0 until minOf(pv.size, 10) step 2).toList()
        if (ours.size >= 3 && ours.all { boards[it].givesCheck(pv[it]) }) return "perpetual"
        val seen = HashSet<String>()
        for (b in boards) {
            val key = b.epd()
            if (key in seen) return "repetition"
            seen.add(key)
        }
        return null
    }

    /**
     * The player's sacrifices in `pv[:upto]`, and what each gives. Counted against the material
     * the line starts with: after the reply the player is two points or more down, the player's
     * next move does not make it all back, and it goes deeper than the sacrifice before. So a
     * bishop for two pawns is one, and a rook given in two steps (Rxf7+ Rxf7 Rxf7+ Kxf7, the
     * first rook back at once) is one; a rook given for a queen taken the next move, or a pawn
     * the bishop takes and a pawn takes back, is not (both were said to "give up material",
     * 2026-09-23).
     */
    fun sacrifices(boards: List<Board>, pv: List<Move>, upto: Int): List<Pair<Int, String>> {
        val us = boards[0].turn
        val start = balance(boards[0], us)
        var lowest = start - 1  // a new sacrifice must go below this
        val out = ArrayList<Pair<Int, String>>()
        for (i in 0 until minOf(upto, pv.size) step 2) {
            if (i + 2 >= boards.size) break
            val afterReply = balance(boards[i + 2], us)
            if (afterReply > start - 2 || afterReply >= lowest) continue
            if (i + 3 < boards.size && balance(boards[i + 3], us) >= start) continue  // a trade
            val reply = pv[i + 1]
            val given = boards[i + 1].pieceAt(reply.to)
            if (given == null || given.colour != us) continue
            lowest = afterReply
            val took = if (boards[i].isCapture(pv[i])) boards[i].pieceTypeAt(pv[i].to) else 0
            if (given.pieceType == ROOK && (took == BISHOP || took == KNIGHT)) {
                out.add(i to "exchange")
            } else {
                out.add(i to NAMES.getValue(given.pieceType))
            }
        }
        return out
    }

    /**
     * The plies of the line before the player takes a piece the opponent just put where it
     * could be taken, worth more than what it took on the way there; `reach` if none.
     */
    private fun firstGift(boards: List<Board>, pv: List<Move>, reach: Int): Int {
        for (k in 1 until minOf(reach, pv.size - 1) step 2) {
            val move = pv[k]
            val answer = pv[k + 1]
            if (answer.to != move.to || !boards[k + 1].isCapture(answer)) continue
            val piece = boards[k].pieceTypeAt(move.from)
            val took = if (boards[k].isCapture(move)) boards[k].pieceTypeAt(move.to) else 0
            if (value(piece) > value(took)) return k + 1
        }
        return reach
    }

    /**
     * What the line wins in its first `horizon` plies, in pawns, and the ply it stands at: the
     * best the player's material stands after their opponent's replies, a queen of their own
     * making not counted (making one is a promotion, not a win of "their queen"). A line that
     * stops on the player's capture counts it less what can take back: "dxe4+ Ke3 Rxd4" is a
     * pawn, not a rook, with Rxd4 to come; a puzzle's solution stops the same way.
     */
    fun lineGain(boards: List<Board>, pv: List<Move>, horizon: Int): Pair<Int, Int> {
        val board = boards[0]
        val us = board.turn
        val base = balance(board, us)
        val made = promoted(boards, pv, us)
        var gain = 0
        var at = 0
        for (k in 2..horizon step 2) {
            val g = balance(boards[k], us) - base - made[k]
            if (g > gain) {
                gain = g
                at = k
            }
        }
        if (horizon == pv.size && horizon % 2 == 1) {
            var g = balance(boards[horizon], us) - base - made[horizon]
            val last = pv[horizon - 1]
            if (retaken(boards[horizon], last.to)) {
                val kind = boards[horizon].pieceTypeAt(last.to)
                g -= value(if (kind == 0) PAWN else kind)
            }
            if (g > gain) {
                gain = g
                at = horizon
            }
        }
        return gain to at
    }

    private fun legalPrefix(board: Board, moves: List<String>): List<Move> {
        val out = ArrayList<Move>()
        val at = board.copy(withStack = false)
        for (uci in moves) {
            val move = Move.fromUciOrNull(uci) ?: break
            if (!at.isLegal(move)) break
            out.add(move)
            at.push(move)
        }
        return out
    }

    private fun balance(board: Board, us: Boolean): Int {
        var total = 0
        for (piece in board.pieceMap().values) {
            val v = value(piece.pieceType)
            total += if (piece.colour == us) v else -v
        }
        return total
    }

    /**
     * How the line starts: with a check, a promotion, a sacrifice (material given that the
     * reply keeps), a capture, or quietly.
     */
    private fun start(board: Board, boards: List<Board>, pv: List<Move>): String {
        val first = pv[0]
        val kind = when {
            board.givesCheck(first) -> "check"
            first.promotion != 0 -> "promotion"
            board.isCapture(first) -> "capture"
            else -> "quiet"
        }
        if (sacrifices(boards, pv, 1).isNotEmpty()) return "sacrifice"
        return kind
    }

    /**
     * The most telling tactic of the player's moves in `pv[:upto]`, the later one of two alike
     * (the point, not the preparation); which of the player's moves carries it; and what the
     * moves before it do to prepare it. [gained]: the kinds of piece the line nets; a tactic
     * whose own move takes one of them, worth more than what it aims at, is not the point
     * ([captureIsThePoint]).
     */
    private fun findBlow(
        boards: List<Board>,
        pv: List<Move>,
        upto: Int,
        mate: Int?,
        allowed: List<String>? = null,
        gained: Set<Int> = emptySet(),
    ): Triple<Pair<Int, MotifTag>?, Int, List<String>> {
        var bestRank = 0
        var bestNegPly = 0
        var bestPly = -1
        var bestTag: MotifTag? = null
        val tagged = HashMap<Int, List<MotifTag>>()
        for (i in 0 until maxOf(upto, 1) step 2) {
            if (i >= pv.size) break
            val mateIn = if (mate != null) mate - i / 2 else null
            val tags = Motifs.tagsForMove(boards[i], pv[i], pv.subList(i + 1, pv.size), mateIn)
            tagged[i] = tags
            for (tag in tags) {
                val rank = BLOW_INDEX[tag.name] ?: continue
                if (allowed != null && tag.name !in allowed) continue
                val targeted = tag.name in Motifs.TARGETED
                if (mate == null && targeted && !Motifs.paysOff(boards, pv, i, tag)) continue
                if (mate == null && targeted && captureIsThePoint(boards[i], pv[i], tag, gained)) continue
                if (mate == null && tag.name in PREPARATION && i + 2 >= upto) {
                    // A decoy or a deflection prepares what a later move cashes in: on the last
                    // of the player's moves before the win it explains what comes after it (the
                    // rook that takes the queen "lures" the king that takes it back).
                    continue
                }
                // key = (rank, -i): the lower rank first; of the same rank, the later move.
                if (bestTag == null || rank < bestRank || (rank == bestRank && -i < bestNegPly)) {
                    bestRank = rank
                    bestNegPly = -i
                    bestPly = i
                    bestTag = tag
                }
            }
        }
        val tag = bestTag ?: return Triple(null, 1, emptyList())
        val ply = bestPly
        val preparation = ArrayList<String>()
        for (i in 0..ply step 2) {
            for (other in tagged[i] ?: emptyList()) {
                if (other.name in PREPARATION && other.name !in preparation) preparation.add(other.name)
            }
        }
        return Triple(ply to tag, ply / 2 + 1, preparation)
    }

    /**
     * Whether the move takes more than the tactic it carries aims at, and what it takes is what
     * the line wins ([gained], the kinds it nets): a capture of a queen that also forks a bishop
     * and a pawn wins the queen by taking it, not by the fork ("That's a fork" beside a queen won
     * on c2 whose targets were c1, h2 and a4: 35...Nxf5 in Johner-Nimzowitsch). A queen taken
     * back is no win, and the pin that nets the rook after it stays the point. Kings are not
     * counted: a check is no material target. `idea._capture_is_the_point`.
     */
    internal fun captureIsThePoint(board: Board, move: Move, tag: MotifTag, gained: Set<Int>): Boolean {
        if (!board.isCapture(move)) return false
        val taken = board.pieceAt(move.to)?.pieceType ?: PAWN // en passant: a pawn
        if (taken !in gained) return false
        val after = board.copy(withStack = false)
        after.push(move)
        var aims = 0
        for (name in tag.squares) {
            val square = parseSquare(name)
            if (square == move.from || square == move.to) continue
            val piece = after.pieceAt(square) ?: continue
            if (piece.colour != after.turn || piece.pieceType == KING) continue
            aims = maxOf(aims, value(piece.pieceType))
        }
        return value(taken) > aims
    }

    /**
     * The kinds of piece the line nets from [board] to [end]: taken from them more than they took
     * of the player's. `idea._gained_kinds`.
     */
    private fun gainedKinds(board: Board, end: Board, us: Boolean): Set<Int> {
        val them = !us
        return IDEA_COUNTED.filter { kind ->
            (board.pieces(kind, them).size - end.pieces(kind, them).size) -
                (board.pieces(kind, us).size - end.pieces(kind, us).size) > 0
        }.toSet()
    }

    /** The first ply, to `upto`, at which the line has won `target` or more. */
    private fun firstReaching(boards: List<Board>, pv: List<Move>, upto: Int, target: Int): Int {
        for (k in 2..upto) {
            if (k % 2 != 0 && k != upto) continue
            if (lineGain(boards, pv, k).first >= target) return k
        }
        return 0
    }

    /**
     * Per ply, what the player's new pieces add to the balance beyond the pawns they were: a
     * queen made and still standing counts eight; one taken back, nothing.
     */
    private fun promoted(boards: List<Board>, pv: List<Move>, us: Boolean): List<Int> {
        val out = arrayListOf(0)
        val alive = LinkedHashMap<Int, Int>()
        for ((j, move) in pv.withIndex()) {
            val mover = boards[j].turn
            if (mover == us) {
                val extra = alive.remove(move.from)
                if (move.promotion != 0) {
                    alive[move.to] = value(move.promotion) - 1
                } else if (extra != null) {
                    alive[move.to] = extra
                }
            } else {
                alive.remove(move.to)
            }
            out.add(alive.values.sum())
        }
        return out
    }

    /**
     * What the line wins, in words ("their queen", "the exchange", "a rook for a knight"), and
     * where the most valuable piece it wins stands now; with [owed], net of what the exchange in
     * progress took from the player.
     */
    private fun wonBy(
        board: Board,
        end: Board,
        us: Boolean,
        played: List<Move>,
        boards: List<Board>,
        owed: Map<Int, Int>? = null,
    ): Pair<String?, Int?> {
        val them = !us
        val nets = LinkedHashMap<Int, Int>()
        for (kind in IDEA_COUNTED) {
            nets[kind] = (board.pieces(kind, them).size - end.pieces(kind, them).size) -
                (board.pieces(kind, us).size - end.pieces(kind, us).size) -
                (owed?.get(kind) ?: 0)
        }
        if (!owed.isNullOrEmpty() && nets.getValue(KNIGHT) * nets.getValue(BISHOP) < 0) {
            // Net of what they took, a knight for a bishop is a trade: "two pawns", not "a knight
            // and two pawns for a bishop".
            val both = minOf(abs(nets.getValue(KNIGHT)), abs(nets.getValue(BISHOP)))
            for (kind in intArrayOf(KNIGHT, BISHOP)) {
                nets[kind] = nets.getValue(kind) - if (nets.getValue(kind) > 0) both else -both
            }
        }
        val gained = LinkedHashMap<Int, Int>()
        val given = LinkedHashMap<Int, Int>()
        for ((kind, n) in nets) {
            if (n > 0) {
                gained[kind] = n
            } else if (n < 0) {
                given[kind] = -n
            }
        }
        if (gained.isEmpty()) return null to null
        // max() over the dict: the first of the most valuable, in insertion order.
        var top = gained.keys.first()
        for (kind in gained.keys) if (value(kind) > value(top)) top = kind
        // Where the most valuable piece won stands now: the first of its kind the player takes.
        var target: Int? = null
        for ((i, move) in played.withIndex()) {
            if (i % 2 != 0) continue
            val taken = boards[i].pieceAt(move.to)
            if (taken != null && taken.colour == them && taken.pieceType == top) {
                target = origin(board, boards, i, move.to)
                break
            }
        }
        return wonWords(gained, given) to target
    }

    /**
     * The board `at` plies into the line; a line that stops on the player's capture loses the
     * capturing piece where it can be taken back, as [lineGain] counts it ("you win the
     * exchange", not "a rook", after Nxe1).
     */
    private fun lineEnd(boards: List<Board>, pv: List<Move>, at: Int): Board {
        var end = boards[at]
        if (at == pv.size && at % 2 == 1 && retaken(end, pv[at - 1].to)) {
            end = end.copy(withStack = false)
            end.removePieceAt(pv[at - 1].to)
        }
        return end
    }

    /**
     * The most valuable kind of the opponent's pieces the line takes, in words: "their queen",
     * "a rook", "a knight".
     */
    private fun topTaken(board: Board, end: Board, us: Boolean): String? {
        val them = !us
        val taken = IDEA_COUNTED.filter { end.pieces(it, them).size < board.pieces(it, them).size }
        if (taken.isEmpty()) return null
        // max() over the list: the first of the most valuable, in `_COUNTED` order.
        var top = taken.first()
        for (kind in taken) if (value(kind) > value(top)) top = kind
        return if (top == QUEEN) "their queen" else "a ${NAMES.getValue(top)}"
    }

    /** Where the piece standing on `square` at `ply` stood at the start, followed back. */
    fun origin(board: Board, boards: List<Board>, ply: Int, square: Int): Int? {
        var at = square
        for (i in ply - 1 downTo 0) {
            val before = boards[i]
            val after = boards[i + 1]
            val moved = (0 until 64).filter { sq ->
                before.pieceAt(sq) != after.pieceAt(sq) && after.pieceAt(sq) == null
            }
            if (after.pieceAt(at) != null && before.pieceAt(at) != after.pieceAt(at)) {
                // The piece arrived on `at` at this ply: it came from the square it left.
                val arrived = after.pieceAt(at)
                var came = moved.filter { before.pieceAt(it) == arrived }
                if (came.isEmpty() && arrived != null) {
                    // A promotion: the new piece came from the pawn that left ("their rook on d1"
                    // was named for the pawn on d2 that underpromotes in the line).
                    came = moved.filter { sq ->
                        val p = before.pieceAt(sq)
                        p != null && p.colour == arrived.colour && p.pieceType == PAWN
                    }
                }
                if (came.isNotEmpty()) at = came[0]
            }
        }
        return if (board.pieceAt(at) != null) at else null
    }

    private fun piecesText(counts: Map<Int, Int>): String {
        val parts = ArrayList<String>()
        for (kind in IDEA_COUNTED) {
            val n = counts[kind] ?: 0
            if (n == 1) {
                parts.add("a ${NAMES.getValue(kind)}")
            } else if (n > 1) {
                parts.add("${NUMBERS[n] ?: n.toString()} ${PLURALS.getValue(NAMES.getValue(kind))}")
            }
        }
        return if (parts.size == 1) parts[0] else parts.dropLast(1).joinToString(", ") + " and ${parts.last()}"
    }

    private fun wonWords(gained: Map<Int, Int>, given: Map<Int, Int>): String {
        if (given.isNotEmpty()) {
            val onlyRook = gained.size == 1 && gained[ROOK] == 1
            if (onlyRook && given.values.sum() == 1 && given.keys.all { it == KNIGHT || it == BISHOP }) {
                return "the exchange"
            }
            return "${piecesText(gained)} for ${piecesText(given)}"
        }
        if (gained.size == 1 && gained[QUEEN] == 1) return "their queen"
        return piecesText(gained)
    }

    /**
     * The squares `colour`'s king could step to, were it their move: not held by their own
     * pieces, not attacked by the other side.
     */
    private fun flightOf(board: Board, colour: Boolean): List<Int> {
        val king = board.king(colour) ?: return emptyList()
        val view = board.copy(withStack = false)
        view.removePieceAt(king)  // a square behind the king on a checking line is no refuge
        val out = ArrayList<Int>()
        for (square in 0 until 64) {
            if (squareDistance(square, king) != 1) continue
            val piece = board.pieceAt(square)
            if (piece != null && piece.colour == colour) continue
            if (view.isAttackedBy(!colour, square)) continue
            out.add(square)
        }
        return out
    }

    /**
     * A prepared threat is read only when what it prepares is mate or at least a minor piece:
     * every move attacks something, and "your opponent is preparing to win a pawn" would be
     * said of half the moves of a quiet game.
     */
    const val PREPARED_MIN = 3

    /**
     * The opponent's threat, if the player passed, when it mates or wins material; what it
     * threatens ("mate", or "your rook"); and, when the threat is a quiet move that prepares it
     * (their best reply to a pass threatens mate or a piece next), the move that would follow.
     * A prepared threat counts only when the best move takes it away: where the best move lets
     * it come and strikes first (g4 while ...b4 still wins the knight), the point is the
     * counter, and "stop it before it comes" would be false (the master games' notes,
     * 2026-09-24).
     */
    private fun threatOf(board: Board, threats: List<String>, best: Move? = null): Triple<Move?, String?, Move?> {
        if (threats.isEmpty() || board.isCheck()) return Triple(null, null, null)
        val passed = board.copy(withStack = false)
        passed.push(Move.NULL)
        for (uci in threats) {
            val move = Move.fromUciOrNull(uci) ?: continue
            if (!passed.isLegal(move)) continue
            val after = passed.copy(withStack = false)
            after.push(move)
            if (after.isCheckmate()) return Triple(move, "mate", null)
            val taken = passed.pieceAt(move.to)
            // A capture that wins material by the exchange, legal recaptures and all: a knight
            // left to a rook with only a pinned piece to take back is a threat (2026-09-23).
            if (taken != null && taken.colour == board.turn && Exchange.exchange(passed, move) > 0) {
                if (best != null && board.isLegal(best) && ignores(board, best, move)) {
                    continue  // the best move lets it come: the point is elsewhere
                }
                return Triple(move, "your ${NAMES.getValue(taken.pieceType)}", null)
            }
        }
        // The engine's first reply to a pass, when quiet: what it would threaten next.
        for (uci in threats.take(1)) {
            val move = Move.fromUciOrNull(uci) ?: continue
            if (!passed.isLegal(move) || passed.isCapture(move)) continue
            val prepared = Plans.threat(passed, move)
            if (prepared == null || prepared.kind == "trade") continue
            if (best != null && board.isLegal(best)) {
                val answered = board.copy(withStack = false)
                answered.push(best)
                if (answered.isLegal(move)) {
                    // The same threat, on the same target: after Nd4 the rook's Rd1 goes for the
                    // knight, not the queen it went for, and the queen's threat is met.
                    val still = Plans.threat(answered, move)
                    if (still != null &&
                        still.kind == prepared.kind &&
                        (still.kind == "mate" || still.square == prepared.square)
                    ) {
                        continue
                    }
                }
            }
            if (prepared.kind == "mate") return Triple(move, "mate", prepared.follow)
            val target = passed.pieceAt(prepared.square)
            if (target != null && prepared.gain >= PREPARED_MIN) {
                return Triple(move, "your ${NAMES.getValue(target.pieceType)}", prepared.follow)
            }
        }
        return Triple(null, null, null)
    }
}
