package com.sidecoach.coach

import com.sidecoach.analysis.Exchange
import com.sidecoach.analysis.isPassed
import com.sidecoach.chess.BB_ALL
import com.sidecoach.chess.BB_EMPTY
import com.sidecoach.chess.BB_SQUARES
import com.sidecoach.chess.BISHOP
import com.sidecoach.chess.BLACK
import com.sidecoach.chess.Board
import com.sidecoach.chess.KING
import com.sidecoach.chess.KNIGHT
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.ROOK
import com.sidecoach.chess.SanFailure
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.parseSquare
import com.sidecoach.chess.square
import com.sidecoach.chess.squareFile
import com.sidecoach.chess.squareName
import com.sidecoach.chess.squareRank
import com.sidecoach.chess.squaresOf
import com.sidecoach.events.Line
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.LLMProvider
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import org.json.JSONTokener

/**
 * Claim verification, both directions: `sidecoach/coach/claims.py`. DESIGN.md section 5.5.
 *
 * Direction one: what the user says in their own defence ("my knight is attacking e5 and the
 * pawn is hanging") is turned into structured claims by the LLM and every checkable one is
 * tested against the board before the coach agrees with anything. Only the extraction uses
 * the LLM. Resolving "my knight" to a square, counting attackers and defenders, trying the
 * null move and comparing with the engine's candidates all happen here, on the board, where
 * nothing can be made up. Attackers and defenders are counted by legal moves ([Exchange]): a
 * piece pinned to its king takes nothing and takes nothing back, and a king takes back only
 * where it may.
 *
 * Direction two: the coach's own draft is checked before it is shown. Every piece it places
 * on a square must be there, and every move it names must be playable, by the side to move
 * or as the opponent's threat after a pass.
 *
 * Every reason is one sentence a coach could say, naming squares and pieces as they stand.
 */

enum class ClaimKind(val slug: String) {
    ATTACKS("attacks"), // piece X attacks square/piece Y
    DEFENDS("defends"), // piece X defends square/piece Y
    THREATENS("threatens"), // the opponent threatens Y (or "I threaten Y next move")
    HANGING("hanging"), // piece on Y is undefended / can be taken for free
    SAFE("safe"), // piece/square Y is safe / cannot be taken
    BEST("best"), // move M is the best move / a good move
    PLAN("plan"), // a strategic intention; not checkable, prompts a follow-up
    OTHER("other"), // anything else; not checkable
    ;

    companion object {
        fun of(slug: String): ClaimKind? = entries.firstOrNull { it.slug == slug }
    }
}

/** "true", "false" or "unclear": `claims.Result`. */
enum class ClaimResult(val slug: String) {
    TRUE("true"),
    FALSE("false"),
    UNCLEAR("unclear"),
}

/**
 * One checkable thing the user said, in their own words.
 *
 * [subject] and [target] are kept as the user named them: a square ("f3") or a piece
 * description ("my knight", "the pawn on e5"). [move] is SAN: the move judged for BEST,
 * the threatened move for THREATENS when one was named, and for the board claims the move
 * the user is weighing when the claim is about the position after it ("Nf3 defends d4").
 */
data class Claim(
    val kind: ClaimKind,
    val subject: String?,
    val target: String?,
    val move: String?,
    val text: String,
)

data class Verdict(val claim: Claim, val result: ClaimResult, val reason: String)

data class DraftProblem(
    val text: String, // the offending phrase
    val reason: String, // what the board says instead
)

/** What [verifyClaims] needs from an engine; `analysis.Engine` satisfies it. */
fun interface ClaimEngine {
    fun analyse(board: Board, depth: Int?): List<Line>
}

// --- Direction one: extraction --------------------------------------------------------------

const val EXTRACT_SYSTEM =
    "You turn what a chess student says about the position in front of them into a JSON " +
        "array of claims, so a coach can check each one against the board before agreeing.\n" +
        "\n" +
        "Reply with the JSON array only: no prose, no code fences, no keys other than these. " +
        "Each element is an object:\n" +
        "- \"kind\": one of \"attacks\", \"defends\", \"threatens\", \"hanging\", \"safe\", " +
        "\"best\", \"plan\", \"other\"\n" +
        "- \"subject\": the actor, in the student's words (\"my knight\", \"the bishop\", " +
        "\"f3\", \"I\", \"they\"); null when the kind has no actor\n" +
        "- \"target\": what it acts on, in the student's words (\"e5\", \"the pawn\", \"their " +
        "queen\", \"my king\"); null when there is none\n" +
        "- \"move\": a move in standard algebraic notation, or null\n" +
        "- \"text\": the clause the claim came from, copied verbatim\n" +
        "\n" +
        "Kinds, with where the words go:\n" +
        "- attacks: a piece attacks a square or a piece (\"my knight hits e5\"): subject the " +
        "piece, target the square or piece\n" +
        "- defends: a piece defends a square or a piece (\"the bishop covers d4\"): same as " +
        "attacks\n" +
        "- threatens: the opponent threatens something (\"they are threatening my rook\") or the " +
        "student does (\"I threaten mate on f7\"): subject \"they\" or \"I\", target the thing " +
        "threatened, move the threatened move if one was named\n" +
        "- hanging: a piece is undefended or can be taken for free (\"the pawn is hanging\"): " +
        "subject null, target the piece\n" +
        "- safe: a piece or square cannot be taken (\"my knight is safe there\"): subject null, " +
        "target the piece or square\n" +
        "- best: a move is the best move or a good move (\"Nf3 is best\", \"I want to play " +
        "Bb5\"): subject null, target null, move the move\n" +
        "- plan: a strategic intention that no single move settles (\"push the h-pawn\"): " +
        "subject and target null\n" +
        "- other: anything else about the game: subject and target null\n" +
        "\n" +
        "For attacks, defends, hanging and safe, \"move\" is the move the student is weighing " +
        "when the claim is about the position after that move (\"Nf3 defends d4\"), else null.\n" +
        "\n" +
        "Keep the student's own words; do not turn \"my knight\" into a square, and do not check " +
        "the claims yourself. One sentence can hold several claims. The student is the side to " +
        "move: \"my\" and \"I\" are the student, \"your\" and \"their\" the opponent. Return [] " +
        "when nothing said is about the position. Decide quickly; the closest kind is good " +
        "enough.\n"

private val FENCE = Regex("```[A-Za-z]*\\s*(.*?)```", RegexOption.DOT_MATCHES_ALL)
private val THINK = Regex("<think>.*?</think>", RegexOption.DOT_MATCHES_ALL)

/**
 * The claims in [text], as the LLM reads them. Never raises; a failed call yields none.
 *
 * Measured 2026-09-15 on the local Qwen 3.8 27B: with thinking off the three claims in a
 * typical sentence come back in about 5 s; with thinking on the model reasons for some
 * 3000 tokens first, about 50 s, and a smaller budget is eaten whole by the reasoning and
 * returns no text at all. When that happens the call is retried once with thinking off,
 * so a runaway costs time but never the claims.
 */
fun extractClaims(
    text: String,
    board: Board,
    provider: LLMProvider,
    thinking: Boolean = true,
    maxTokens: Int = 4000,
): List<Claim> {
    if (text.isBlank()) return emptyList()
    val side = if (board.turn == WHITE) "white" else "black"
    val user = "Side to move: $side (the student).\nFEN: ${board.fen()}\n\nStudent: ${text.trim()}"
    val messages = listOf(ChatMessage("user", user))
    val reply = try {
        var completion = provider.chat(EXTRACT_SYSTEM, messages, maxTokens, 0.1, thinking)
        if (thinking && completion.text.isBlank()) {
            completion = provider.chat(EXTRACT_SYSTEM, messages, maxTokens, 0.1, false)
        }
        completion.text
    } catch (_: Exception) {
        return emptyList()
    }
    return parseClaims(reply)
}

/**
 * Claims from a model reply: a JSON array, fenced or not, or a single object.
 *
 * Entries with an unknown kind are dropped. Anything unreadable yields nothing.
 */
fun parseClaims(text: String): List<Claim> {
    var cleaned = THINK.replace(text, "").trim()
    val fenced = FENCE.find(cleaned)
    if (fenced != null) cleaned = fenced.groupValues[1].trim()
    var data = loadJson(cleaned)
    if (data is JSONObject) {
        val inner = data.optJSONArray("claims")
        data = inner ?: JSONArray().put(data)
    }
    if (data !is JSONArray) return emptyList()
    val claims = ArrayList<Claim>()
    for (i in 0 until data.length()) {
        val claim = claimFrom(data.opt(i))
        if (claim != null) claims.add(claim)
    }
    return claims
}

private fun loadJson(text: String): Any? {
    strictJson(text)?.let { return it }
    for ((opener, closer) in listOf('[' to ']', '{' to '}')) {
        val start = text.indexOf(opener)
        val end = text.lastIndexOf(closer)
        if (start in 0 until end) {
            strictJson(text.substring(start, end + 1))?.let { return it }
        }
    }
    return null
}

/** `json.loads`: one document and nothing after it, where JSONTokener alone would allow more. */
private fun strictJson(text: String): Any? = try {
    val tokener = JSONTokener(text)
    val value = tokener.nextValue()
    if (tokener.nextClean() != 0.toChar()) null else value
} catch (_: JSONException) {
    null
}

private fun claimFrom(item: Any?): Claim? {
    if (item !is JSONObject) return null
    val kindRaw = item.opt("kind")
    if (kindRaw !is String) return null
    val kind = ClaimKind.of(kindRaw.trim().lowercase()) ?: return null
    return Claim(
        kind = kind,
        subject = textOrNull(item.opt("subject")),
        target = textOrNull(item.opt("target")),
        move = textOrNull(item.opt("move")),
        text = falsyToEmpty(item.opt("text")).trim(),
    )
}

private fun textOrNull(value: Any?): String? {
    if (value !is String) return null
    val stripped = value.trim()
    if (stripped.isEmpty() || stripped.lowercase() in listOf("null", "none")) return null
    return stripped
}

/** `str(value or "")`: a missing, null, false or zero field is the empty string. */
private fun falsyToEmpty(value: Any?): String {
    if (value == null || value === JSONObject.NULL || value == false) return ""
    if (value is Number && value.toDouble() == 0.0) return ""
    return value.toString()
}

// --- Direction one: verification ------------------------------------------------------------

private val SQUARE_IN_WORDS = Regex("(?<![a-z0-9])([a-h][1-8])(?![a-z0-9])")
private val BARE_SQUARE = Regex("[a-h][1-8]")
private val BARE_SQUARE_AT_START = Regex("^[a-h][1-8]")
private val WORDS = Regex("[a-z']+")
private val PIECE_WORDS = mapOf(
    "pawn" to PAWN,
    "knight" to KNIGHT,
    "horse" to KNIGHT,
    "bishop" to BISHOP,
    "rook" to ROOK,
    "queen" to QUEEN,
    "king" to KING,
)

// The user speaks from their own side: "my" is the side to move, "your" and "their" the
// opponent. "white" and "black" are absolute.
private val OURS = setOf("my", "mine", "i", "me", "our", "ours", "us")
private val THEIRS = setOf(
    "your", "yours", "their", "theirs", "they", "them", "opponent", "opponent's", "his", "her",
)

// The tagger's own piece values, so a fork and a skewer are judged the way it judges them.
private val VALUE = intArrayOf(0, 1, 3, 3, 5, 9, 100)

private val TRAILING_MARKS = Regex("[+#!?]+\$")

/**
 * A verdict for every claim, checked on [board], whose side to move is the user.
 *
 * [facts] supplies the engine's candidates and threat line when they belong to this
 * position; otherwise BEST falls back to [engine], and to "unclear" without one.
 */
fun verifyClaims(
    claims: List<Claim>,
    board: Board,
    facts: PositionFacts? = null,
    engine: ClaimEngine? = null,
    depth: Int = 12,
): List<Verdict> {
    val useful = if (facts != null && !samePosition(facts.fen, board)) null else facts
    val verdicts = ArrayList<Verdict>(claims.size)
    for (claim in claims) {
        val (result, reason) = verify(claim, board, useful, engine, depth)
        verdicts.add(Verdict(claim = claim, result = result, reason = reason))
    }
    return verdicts
}

private fun samePosition(fen: String, board: Board): Boolean =
    fen.split(" ").take(3) == board.fen().split(" ").take(3)

/** The board a claim is checked on, and who the user is on it. */
private class Pos(
    val board: Board,
    val us: Boolean,
    /** A hypothetical move already pushed on [board]. */
    val played: Move? = null,
    val playedSan: String? = null,
) {
    val them: Boolean get() = !us

    fun side(colour: Boolean): String = if (colour == us) "your" else "their"

    fun sides(colour: Boolean): String = if (colour == us) "yours" else "theirs"

    /** "your knight on f3", or just "e5" for an empty square. */
    fun name(square: Int): String {
        val piece = board.pieceAt(square) ?: return squareName(square)
        return "${side(piece.colour)} ${COACH_PIECE_NAMES[piece.pieceType]} on ${squareName(square)}"
    }

    /** "the f3 knight", or the square when empty. */
    fun short(square: Int): String {
        val piece = board.pieceAt(square) ?: return squareName(square)
        return "the ${squareName(square)} ${COACH_PIECE_NAMES[piece.pieceType]}"
    }

    fun listed(squares: Iterable<Int>): String = join(squares.sorted().map { short(it) })

    /** "your knights on b1 and f3", for a description that fit several pieces. */
    fun group(squares: List<Int>): String {
        val pieces = squares.map { board.pieceAt(it) }
        val first = pieces[0] ?: return "those pieces"
        if (pieces.any { it == null || it != first }) return "those pieces"
        val where = join(squares.sorted().map { squareName(it) })
        return "${side(first.colour)} ${COACH_PIECE_NAMES[first.pieceType]}s on $where"
    }
}

private fun join(items: List<String>): String {
    if (items.isEmpty()) return "nothing"
    if (items.size == 1) return items[0]
    return items.dropLast(1).joinToString(", ") + " and ${items.last()}"
}

private fun times(count: Int): String = when (count) {
    0 -> "not at all"
    1 -> "once"
    2 -> "twice"
    3 -> "three times"
    else -> "$count times"
}

/**
 * The squares a description could mean: "f3", "the pawn on e5", "my knight", "they".
 *
 * A square in the words settles it. Otherwise a colour word picks the side ("my" is [us]),
 * [prefer] breaks the tie when none was said, and the piece word picks the pieces; a side
 * with no piece word ("I", "they") means every piece of that side. Several squares mean
 * the description was ambiguous and the caller tries each.
 */
private fun resolve(desc: String?, board: Board, us: Boolean, prefer: Boolean?): List<Int> {
    if (desc.isNullOrEmpty()) return emptyList()
    val lowered = desc.lowercase()
    val square = SQUARE_IN_WORDS.find(lowered)
    if (square != null) return listOf(parseSquare(square.groupValues[1]))
    var colour: Boolean? = null
    var pieceType = 0
    for (match in WORDS.findAll(lowered)) {
        val word = match.value
        when {
            word in OURS -> colour = us
            word in THEIRS -> colour = !us
            word == "white" || word == "white's" -> colour = WHITE
            word == "black" || word == "black's" -> colour = BLACK
            else -> pieceType = PIECE_WORDS[word.trimEnd('s')] ?: pieceType
        }
    }
    if (colour == null && pieceType == 0) return emptyList()

    fun of(side: Boolean): List<Int> =
        if (pieceType == 0) squaresOf(board.occupiedCo[if (side) 1 else 0])
        else board.pieces(pieceType, side)

    if (colour != null) return of(colour).sorted()
    if (prefer == null) return (of(WHITE) + of(BLACK)).sorted()
    return (of(prefer).ifEmpty { of(!prefer) }).sorted()
}

private fun saidOurs(desc: String?): Boolean =
    !desc.isNullOrEmpty() && WORDS.findAll(desc.lowercase()).any { it.value in OURS }

/** A legal move from SAN or UCI, or null with why: "ambiguous", "illegal", "unreadable". */
private fun parseMove(board: Board, text: String): Pair<Move?, String> {
    val cleaned = TRAILING_MARKS.replace(text.trim(), "")
    val (move, failure) = board.parseSanFully(cleaned)
    if (move != null) return move to ""
    when (failure) {
        SanFailure.AMBIGUOUS -> return null to "ambiguous"
        SanFailure.ILLEGAL -> return null to "illegal"
        else -> Unit
    }
    val uci = Move.fromUciOrNull(cleaned.lowercase()) ?: return null to "unreadable"
    return if (board.legalMoves().contains(uci)) uci to "" else null to "illegal"
}

/** The board to check a board claim on: after the move the user is weighing, if any. */
private fun position(claim: Claim, board: Board): Pos {
    val us = board.turn
    var move: Move? = null
    if (!claim.move.isNullOrEmpty()) move = parseMove(board, claim.move).first
    if (move == null && !claim.subject.isNullOrEmpty() &&
        !BARE_SQUARE.matches(claim.subject.trim())
    ) {
        move = parseMove(board, claim.subject).first
    }
    if (move == null) return Pos(board, us)
    val after = board.copy(withStack = false)
    val san = board.san(move)
    after.push(move)
    return Pos(after, us, move, san)
}

private fun verify(
    claim: Claim,
    board: Board,
    facts: PositionFacts?,
    engine: ClaimEngine?,
    depth: Int,
): Pair<ClaimResult, String> = when (claim.kind) {
    ClaimKind.ATTACKS -> checkAttacks(position(claim, board), claim)
    ClaimKind.DEFENDS -> checkDefends(position(claim, board), claim)
    ClaimKind.HANGING -> checkHanging(position(claim, board), claim)
    ClaimKind.SAFE -> checkSafe(position(claim, board), claim)
    ClaimKind.THREATENS -> checkThreatens(Pos(board, board.turn), claim, facts)
    ClaimKind.BEST -> checkBest(board, claim, facts, engine, depth)
    ClaimKind.PLAN -> ClaimResult.UNCLEAR to
        ("That is a plan rather than something I can check on the board; what do you " +
            "expect it to achieve, and which move starts it?")
    ClaimKind.OTHER -> ClaimResult.UNCLEAR to
        "I cannot check that against the board; what do you expect to happen, concretely?"
}

/** `claim.target or claim.subject`: an empty target falls through, as in Python. */
private fun targetOrSubject(claim: Claim): String? =
    if (claim.target.isNullOrEmpty()) claim.subject else claim.target

/** Where the actor stands; the piece just moved when the subject was the move itself. */
private fun subjects(pos: Pos, claim: Claim, prefer: Boolean): List<Int> {
    val found = resolve(claim.subject, pos.board, pos.us, prefer)
    if (found.isEmpty() && pos.played != null) return listOf(pos.played.to)
    return found
}

private fun unclearSubject(claim: Claim): Pair<ClaimResult, String> {
    if (!claim.subject.isNullOrEmpty()) {
        return ClaimResult.UNCLEAR to "I cannot tell which piece you mean by '${claim.subject}'."
    }
    return ClaimResult.UNCLEAR to "Which piece do you mean?"
}

private fun unclearTarget(claim: Claim): Pair<ClaimResult, String> {
    if (!claim.target.isNullOrEmpty()) {
        return ClaimResult.UNCLEAR to "I cannot tell what you mean by '${claim.target}'."
    }
    return ClaimResult.UNCLEAR to "Which square or piece do you mean?"
}

/** Prefix a reason with the hypothetical move it was checked after, if there was one. */
private fun afterMove(pos: Pos, reason: String): String {
    if (pos.playedSan == null) return reason
    return "after ${pos.playedSan}, $reason"
}

private fun Board.attacksSquare(from: Int, to: Int): Boolean =
    attacksMask(from) and BB_SQUARES[to] != BB_EMPTY

/** `board` with `colour` to move: the same position, handed over by a pass if need be. */
private fun toMove(board: Board, colour: Boolean): Board {
    if (board.turn == colour) return board
    val view = board.copy(withStack = false)
    view.push(Move.NULL)
    return view
}

/** The pieces that can legally take back after `capture`, a move of the side to move. */
private fun takesBack(board: Board, capture: Move): List<Int> {
    val after = board.copy(withStack = false)
    after.push(capture)
    val onto = sortedSetOf<Int>()
    after.generateLegalMoves(BB_ALL, BB_SQUARES[capture.to]) { onto.add(it.from) }
    return onto.toList()
}

/**
 * Each enemy piece that can legally take the piece on `square`, with the pieces that can legally
 * take back once it has. Read after the capture itself, so the e8 king takes back after Bxf7+
 * but not after Qxf7 with the bishop still on c4.
 */
private fun capturesOf(board: Board, square: Int): Map<Int, List<Int>> {
    val piece = board.pieceAt(square) ?: return emptyMap()
    val view = toMove(board, !piece.colour)
    val found = java.util.TreeMap<Int, java.util.TreeSet<Int>>()
    for (capture in Exchange.capturesOn(board, !piece.colour, square)) {
        found.getOrPut(capture.from) { java.util.TreeSet() }.addAll(takesBack(view, capture))
    }
    val out = LinkedHashMap<Int, List<Int>>()
    for ((attacker, backs) in found) out[attacker] = backs.toList()
    return out
}

/**
 * The pieces that can legally take back for the piece on `square`: after any capture of it, or,
 * when nothing can take it, were something to ([Exchange.defenders]).
 */
private fun guards(board: Board, square: Int): List<Int> {
    val piece = board.pieceAt(square) ?: return emptyList()
    val captures = capturesOf(board, square)
    if (captures.isEmpty()) return Exchange.defenders(board, piece.colour, square)
    return captures.values.flatten().toSortedSet().toList()
}

/**
 * The pieces of `colour` that reach `square` but are not among `active`, the ones that can
 * legally move there.
 */
private fun idle(board: Board, colour: Boolean, square: Int, active: Iterable<Int>): List<Int> {
    val moving = active.toSet()
    return board.attackers(colour, square).filter { it !in moving }
}

/**
 * "is pinned and cannot take it": why pieces of `colour` that reach a square cannot move there,
 * said as a pin when every one of them is pinned to its king.
 */
private fun cannot(board: Board, colour: Boolean, squares: List<Int>, action: String): String {
    if (squares.isNotEmpty() && squares.all { board.isPinned(colour, it) }) {
        return "${if (squares.size == 1) "is" else "are"} pinned and cannot $action"
    }
    return "cannot $action"
}

/**
 * "the d8 queen behind the d7 rook joins in": a piece of `colour` lined up behind one of
 * `fronts` that reaches `square` once the front one has taken there, an x-ray.
 */
private fun joinsIn(pos: Pos, colour: Boolean, square: Int, fronts: Iterable<Int>): String? {
    val board = pos.board
    val reach = board.attackers(colour, square)
    for (front in fronts) {
        val probe = board.copy(withStack = false)
        probe.removePieceAt(front)
        for (backer in probe.attackers(colour, square)) {
            if (backer !in reach) return "${pos.short(backer)} behind ${pos.short(front)} joins in"
        }
    }
    return null
}

private fun checkAttacks(pos: Pos, claim: Claim): Pair<ClaimResult, String> {
    val board = pos.board
    val subjects = subjects(pos, claim, pos.us)
    if (subjects.isEmpty()) return unclearSubject(claim)
    val targets = resolve(claim.target, board, pos.us, pos.them)
    if (targets.isEmpty()) return unclearTarget(claim)
    val occupied = subjects.filter { board.pieceAt(it) != null }
    if (occupied.isEmpty()) {
        return ClaimResult.FALSE to "There is nothing on ${squareName(subjects[0])}."
    }
    for (subject in occupied) {
        for (target in targets) {
            if (!board.attacksSquare(subject, target)) continue
            var reason = "${pos.name(subject)} attacks ${pos.name(target)}"
            val hit = board.pieceAt(target)
            if (hit != null && hit.colour != board.colourAt(subject) && hit.pieceType != KING) {
                // Attacking is the geometry; taking is a legal move. A pinned piece still
                // attacks, but it cannot take, and then who defends does not arise.
                val mover = !hit.colour
                val own = Exchange.capturesOn(board, mover, target).filter { it.from == subject }
                if (own.isEmpty()) {
                    reason += ", but it ${cannot(board, mover, listOf(subject), "take it")}"
                } else {
                    val guards = takesBack(toMove(board, mover), own[0])
                    val idle = idle(board, hit.colour, target, guards)
                    reason += if (guards.isNotEmpty()) {
                        ", though it is defended by ${pos.listed(guards)}"
                    } else if (idle.isNotEmpty()) {
                        ", and ${pos.listed(idle)} ${cannot(board, hit.colour, idle, "take back")}"
                    } else {
                        ", and nothing defends it"
                    }
                }
            }
            return ClaimResult.TRUE to sentence(afterMove(pos, reason))
        }
    }
    val wanted = join(targets.take(3).map { pos.name(it) })
    val reason = if (occupied.size == 1) {
        val subject = occupied[0]
        val hits = board.attacks(subject).filter {
            val colour = board.colourAt(it)
            colour != null && colour != board.colourAt(subject)
        }
        val what = if (hits.isNotEmpty()) {
            "it attacks ${pos.listed(hits)}"
        } else {
            "it attacks none of their pieces"
        }
        "${pos.name(subject)} does not attack $wanted; $what"
    } else {
        "none of ${pos.group(occupied)} attacks $wanted"
    }
    val ambiguous = occupied.size > 1 || targets.size > 1
    return (if (ambiguous) ClaimResult.UNCLEAR else ClaimResult.FALSE) to sentence(afterMove(pos, reason))
}

private fun checkDefends(pos: Pos, claim: Claim): Pair<ClaimResult, String> {
    val board = pos.board
    val subjects = subjects(pos, claim, pos.us)
    if (subjects.isEmpty()) return unclearSubject(claim)
    val targets = resolve(claim.target, board, pos.us, pos.us)
    if (targets.isEmpty()) return unclearTarget(claim)
    val occupied = subjects.filter { board.pieceAt(it) != null }
    if (occupied.isEmpty()) {
        return ClaimResult.FALSE to "There is nothing on ${squareName(subjects[0])}."
    }
    for (subject in occupied) {
        for (target in targets) {
            if (!board.attacksSquare(subject, target)) continue
            val held = board.pieceAt(target)
            if (held == null) {
                val reason =
                    "${pos.name(subject)} covers ${squareName(target)}, which is empty"
                return ClaimResult.TRUE to sentence(afterMove(pos, reason))
            }
            if (held.colour == board.colourAt(subject)) {
                // Defending is taking back: a piece pinned to its king reaches the square and
                // still cannot.
                val guards = guards(board, target)
                if (subject !in guards) {
                    val square = squareName(target)
                    val cannot = cannot(board, held.colour, listOf(subject), "take back on $square")
                    return ClaimResult.FALSE to sentence(afterMove(pos, "${pos.name(subject)} $cannot"))
                }
                val others = guards.filter { it != subject }
                var reason = "${pos.name(subject)} defends ${pos.name(target)}"
                if (others.isNotEmpty()) reason += ", together with ${pos.listed(others)}"
                return ClaimResult.TRUE to sentence(afterMove(pos, reason))
            }
            val reason = "${pos.name(target)} is not yours to defend; " +
                "${pos.name(subject)} attacks it"
            return ClaimResult.FALSE to sentence(afterMove(pos, reason))
        }
    }
    val wanted = join(targets.take(3).map { pos.name(it) })
    val reason = if (occupied.size == 1) {
        "${pos.name(occupied[0])} does not reach $wanted"
    } else {
        "none of ${pos.group(occupied)} reaches $wanted"
    }
    val ambiguous = occupied.size > 1 || targets.size > 1
    return (if (ambiguous) ClaimResult.UNCLEAR else ClaimResult.FALSE) to sentence(afterMove(pos, reason))
}

private fun checkHanging(pos: Pos, claim: Claim): Pair<ClaimResult, String> {
    val board = pos.board
    val targets = resolve(targetOrSubject(claim), board, pos.us, null)
    if (targets.isEmpty()) return unclearTarget(claim)
    var fallback: String? = null
    for (target in targets) {
        val piece = board.pieceAt(target)
        if (piece == null) {
            if (fallback == null) fallback = "there is nothing on ${squareName(target)}"
            continue
        }
        if (piece.pieceType == KING) {
            if (fallback == null) {
                fallback = "${pos.name(target)} cannot hang; it can only be in check"
            }
            continue
        }
        val captures = capturesOf(board, target)
        val attackers = captures.keys.toList()
        if (Exchange.hanging(board, target)) {
            val guards = guards(board, target)
            val idle = idle(board, piece.colour, target, guards)
            var reason = "${pos.name(target)} is attacked by ${pos.listed(attackers)}"
            if (guards.isNotEmpty()) {
                // Taken back after one capture, but not after another.
                val free = captures.entries.firstOrNull { it.value.isEmpty() }?.key ?: attackers[0]
                reason += ", and after ${pos.short(free)} takes, nothing can take back"
            } else if (idle.isNotEmpty()) {
                reason += ", and ${pos.listed(idle)} ${cannot(board, piece.colour, idle, "take back")}"
            } else {
                reason += " and nothing defends it"
            }
            return ClaimResult.TRUE to sentence(afterMove(pos, reason))
        }
        var reason: String
        if (attackers.isEmpty()) {
            val idle = idle(board, !piece.colour, target, emptyList())
            reason = if (idle.isNotEmpty()) {
                val only = pos.listed(idle)
                val cannot = cannot(board, !piece.colour, idle, "take it")
                "${pos.name(target)} is attacked only by $only, which $cannot"
            } else {
                "${pos.name(target)} is not attacked at all"
            }
        } else {
            reason = "${pos.name(target)} is attacked by ${pos.listed(attackers)} but " +
                "defended by ${pos.listed(guards(board, target))}, so it is not hanging outright"
            val where = squareName(target)
            val cheapest = attackers.minByOrNull { VALUE[board.pieceTypeAt(it)] }!!
            if (VALUE[board.pieceTypeAt(cheapest)] < VALUE[piece.pieceType]) {
                reason += ", though ${pos.short(cheapest)} taking a " +
                    "${COACH_PIECE_NAMES[piece.pieceType]} still wins material"
            } else if (Exchange.canBeWon(board, target)) {
                reason += ", though the exchange on $where still wins it"
                val joined = joinsIn(pos, !piece.colour, target, attackers)
                if (joined != null) reason += ": $joined"
            } else {
                reason += ", and the exchange on $where does not win it"
            }
        }
        // An attacked piece is the interesting one to explain when several fit the words.
        if (fallback == null || attackers.isNotEmpty()) {
            fallback = reason
            if (attackers.isNotEmpty()) break
        }
    }
    var reason = fallback!!
    if (targets.size > 1) reason = "none of ${pos.group(targets)} is hanging: $reason"
    return (if (targets.size > 1) ClaimResult.UNCLEAR else ClaimResult.FALSE) to
        sentence(afterMove(pos, reason))
}

private fun checkSafe(pos: Pos, claim: Claim): Pair<ClaimResult, String> {
    val board = pos.board
    val targets = resolve(targetOrSubject(claim), board, pos.us, pos.us)
    if (targets.isEmpty()) return unclearTarget(claim)
    val checked = targets.map { safety(pos, it) }
    val held = checked.filter { it.first }.map { it.second }
    val failed = checked.filter { !it.first }.map { it.second }
    if (held.isNotEmpty()) {
        // "Safe" is a universal claim, so a piece the words also fit that is not safe is
        // worth hearing about even when the claim holds for another.
        val reason = held[0] + (if (failed.isNotEmpty()) ", though ${failed[0]}" else "")
        return ClaimResult.TRUE to sentence(afterMove(pos, reason))
    }
    var reason = failed[0]
    if (targets.size > 1) reason = "none of ${pos.group(targets)} is safe: $reason"
    return (if (targets.size > 1) ClaimResult.UNCLEAR else ClaimResult.FALSE) to
        sentence(afterMove(pos, reason))
}

/**
 * Whether the piece on `target` can be taken for profit, or a piece of yours on the empty square
 * taken at all, with the counts. Only legal captures count, and the verdict is the exchange
 * played out by legal moves ([Exchange.canBeWon]), x-rays included; a king's safety is whether
 * it is in check.
 */
private fun safety(pos: Pos, target: Int): Pair<Boolean, String> {
    val board = pos.board
    val square = squareName(target)
    val piece = board.pieceAt(target)
    if (piece == null) {
        // Who of theirs could take a piece of yours standing there.
        val takers = Exchange.defenders(board, pos.them, target)
        if (takers.isNotEmpty()) return false to "$square is attacked by ${pos.listed(takers)}"
        val idle = idle(board, pos.them, target, emptyList())
        if (idle.isNotEmpty()) {
            val cannot = cannot(board, pos.them, idle, "take there")
            return true to "$square is attacked only by ${pos.listed(idle)}, which $cannot"
        }
        return true to "$square is not attacked by any of their pieces"
    }
    if (piece.pieceType == KING) {
        val checkers = board.attackers(!piece.colour, target)
        if (checkers.isEmpty()) return true to "${pos.name(target)} is not in check"
        return false to "${pos.name(target)} is in check from ${pos.listed(checkers)}"
    }
    val attackers = capturesOf(board, target).keys.toList()
    if (attackers.isEmpty()) {
        val idle = idle(board, !piece.colour, target, emptyList())
        if (idle.isNotEmpty()) {
            val only = pos.listed(idle)
            val cannot = cannot(board, !piece.colour, idle, "take it")
            return true to "${pos.name(target)} is attacked only by $only, which $cannot"
        }
        return true to "${pos.name(target)} is not attacked"
    }
    val guards = guards(board, target)
    var counts = "${pos.name(target)} is attacked ${times(attackers.size)} " +
        "(${pos.listed(attackers)}) and defended ${times(guards.size)}"
    if (guards.isNotEmpty()) counts += " (${pos.listed(guards)})"
    if (!Exchange.canBeWon(board, target)) return true to "$counts, so it holds"
    val cheapest = attackers.minByOrNull { VALUE[board.pieceTypeAt(it)] }!!
    var reason: String
    if (VALUE[board.pieceTypeAt(cheapest)] < VALUE[piece.pieceType]) {
        reason = "$counts, but ${pos.short(cheapest)} is worth less than a " +
            "${COACH_PIECE_NAMES[piece.pieceType]}, so it can still be taken profitably"
    } else if (guards.size < attackers.size) {
        reason = "$counts, so it can be taken"
    } else {
        // As many defenders, and still lost: a piece lined up behind an attacker joins in.
        reason = "$counts, but the exchange on $square still wins it"
        val joined = joinsIn(pos, !piece.colour, target, attackers)
        if (joined != null) reason += ": $joined"
    }
    val idle = idle(board, piece.colour, target, guards)
    if (idle.isNotEmpty()) {
        reason += "; ${pos.listed(idle)} ${cannot(board, piece.colour, idle, "take back")}"
    }
    return false to reason
}

private fun checkThreatens(
    pos: Pos,
    claim: Claim,
    facts: PositionFacts?,
): Pair<ClaimResult, String> {
    val board = pos.board
    val threatener = if (saidOurs(claim.subject)) pos.us else pos.them
    val victim = !threatener
    var position = board
    if (position.turn != threatener) {
        if (position.isCheck()) {
            return ClaimResult.UNCLEAR to sentence(
                "${pos.side(position.turn)} king is in check, so the first question is how " +
                    "to meet that"
            )
        }
        position = board.copy(withStack = false)
        position.push(Move.NULL)
    }
    val lead = if (threatener == pos.them) "if you pass, " else ""

    if (!claim.move.isNullOrEmpty()) {
        val (move, why) = parseMove(position, claim.move)
        if (move == null) {
            if (why == "unreadable") {
                return ClaimResult.UNCLEAR to "I cannot read '${claim.move}' as a move."
            }
            val whose = if (threatener == pos.us) "you" else "they"
            return ClaimResult.FALSE to
                sentence("${claim.move} is not a move $whose can play next")
        }
        return ClaimResult.TRUE to sentence(lead + describeThreat(pos, position, move))
    }

    val targets = resolve(claim.target, board, pos.us, victim)
    if (targets.isEmpty()) return unclearTarget(claim)

    val threatLines =
        if (threatener == pos.them) engineThreats(position, facts) else emptyList()
    var fallback: String? = null
    for (target in targets) {
        val piece = board.pieceAt(target)
        if (piece == null) {
            if (fallback == null) {
                fallback = "${squareName(target)} is empty; what do you expect to happen there?"
            }
            continue
        }
        if (piece.colour == threatener) {
            if (fallback == null) {
                fallback = "${pos.name(target)} is ${pos.sides(threatener)}, not a target"
            }
            continue
        }
        for (move in threatLines) {
            if (move.to == target || (piece.pieceType == KING && position.givesCheck(move))) {
                val reason = "the engine's main threat is ${position.san(move)}: " +
                    describeThreat(pos, position, move)
                return ClaimResult.TRUE to sentence(lead + reason)
            }
        }
        if (piece.pieceType == KING) {
            val checks = position.legalMoves().filter { position.givesCheck(it) }
            if (checks.isNotEmpty()) {
                return ClaimResult.TRUE to sentence(lead + describeThreat(pos, position, checks[0]))
            }
            if (fallback == null) {
                fallback = "no ${pos.side(threatener)} piece can give check next move"
            }
            continue
        }
        val captures = position.legalMoves().filter { it.to == target }
        if (captures.isNotEmpty()) {
            return ClaimResult.TRUE to sentence(lead + describeThreat(pos, position, captures[0]))
        }
        var reason = "nothing of ${pos.sides(threatener)} can take ${pos.name(target)} next move"
        if (threatLines.isNotEmpty()) {
            reason += "; the engine's threat is ${position.san(threatLines[0])}"
        }
        if (fallback == null) fallback = reason
    }
    var reason = fallback!!
    if (targets.size > 1) reason = "none of ${pos.group(targets)} is threatened: $reason"
    return (if (targets.size > 1) ClaimResult.UNCLEAR else ClaimResult.FALSE) to sentence(reason)
}

/** The engine's threat line as moves in `position`, which is the board after a pass. */
private fun engineThreats(position: Board, facts: PositionFacts?): List<Move> {
    if (facts == null || facts.threats.isEmpty()) return emptyList()
    val moves = ArrayList<Move>()
    for (san in facts.threats) {
        val move = parseMove(position, san).first
        if (move != null) moves.add(move)
    }
    return moves
}

/**
 * "their bishop on c4 takes your pawn on f7, and your king on e8 can take back".
 *
 * Who can take back is read by legal moves after the capture itself: after Qxf7# nothing can,
 * and after Bxf7+ with the queen still on h5 the e8 king cannot either.
 */
private fun describeThreat(pos: Pos, position: Board, move: Move): String {
    val actor = pos.name(move.from)
    val victim = position.pieceAt(move.to)
    val san = position.san(move)
    if (victim == null) {
        var what = "$actor can play $san"
        if (position.givesCheck(move)) what += ", with check"
        return what
    }
    val what = "$actor takes ${pos.name(move.to)} with $san"
    val recapturers = takesBack(position, move)
    if (recapturers.isNotEmpty()) return "$what, and ${pos.listed(recapturers)} can take back"
    if (san.endsWith("#")) return "$what, and it is checkmate"
    return "$what for free"
}

private fun checkBest(
    board: Board,
    claim: Claim,
    facts: PositionFacts?,
    engine: ClaimEngine?,
    depth: Int,
): Pair<ClaimResult, String> {
    val san = if (claim.move.isNullOrEmpty()) claim.subject else claim.move
    if (san.isNullOrEmpty()) return ClaimResult.UNCLEAR to "Which move do you mean?"
    val (move, why) = parseMove(board, san)
    if (move == null) {
        if (why == "ambiguous") {
            return ClaimResult.UNCLEAR to "'$san' could be more than one move here; which piece?"
        }
        if (why == "illegal") {
            return ClaimResult.FALSE to "$san is not a legal move in this position."
        }
        return ClaimResult.UNCLEAR to "I cannot read '$san' as a move."
    }
    val said = board.san(move)

    val candidates = candidates(board, facts, engine, depth)
        ?: return ClaimResult.UNCLEAR to
            "I have no engine view of this position yet, so I cannot rate $said."
    if (candidates.isEmpty()) {
        return ClaimResult.UNCLEAR to
            "The engine has nothing to say about this position, so $said is unrated."
    }
    if (move in candidates) {
        val shown = join(candidates.map { board.san(it) })
        if (candidates.size == 1) return ClaimResult.TRUE to "$said is the engine's best move here."
        return ClaimResult.TRUE to "$said is among the engine's best moves here ($shown)."
    }
    val preferred = join(candidates.map { board.san(it) })
    return ClaimResult.FALSE to
        "$said is not among the engine's best moves here; it prefers $preferred."
}

/** The near-equal best moves, from the facts when they have them, else the engine. */
private fun candidates(
    board: Board,
    facts: PositionFacts?,
    engine: ClaimEngine?,
    depth: Int,
): List<Move>? {
    if (facts != null && facts.candidates.isNotEmpty()) {
        val moves = ArrayList<Move>()
        for (san in facts.candidates) {
            val move = parseMove(board, san).first
            if (move != null) moves.add(move)
        }
        if (moves.isNotEmpty()) return moves
    }
    if (engine == null) return null
    val lines = engine.analyse(board, depth).filter { it.moves.isNotEmpty() }
    if (lines.isEmpty()) return emptyList()
    val top = lines.maxOf { it.expectedScore }
    val moves = ArrayList<Move>()
    for (line in lines) {
        if (top - line.expectedScore > CANDIDATE_MARGIN) continue
        val move = parseMove(board, line.moves[0]).first
        if (move != null && move !in moves) moves.add(move)
    }
    return moves
}

/** Capitalised and full-stopped, except that a square name stays as it is read. */
private fun sentence(reason: String): String {
    val text = reason.trim()
    if (text.isEmpty()) return text
    val cased = if (!BARE_SQUARE_AT_START.containsMatchIn(text)) {
        text[0].uppercaseChar() + text.substring(1)
    } else {
        text
    }
    return if (cased.last() in ".?!") cased else "$cased."
}

// --- Direction two: the coach's draft --------------------------------------------------------

private const val COLOUR_WORDS = "your|their|the opponent's|opponent's|white's|black's|white|black"
private const val PIECE_WORD = "king|queen|rook|bishop|knight|pawn"
private val PIECE_ON = Regex(
    "\\b(?:(?<colour>$COLOUR_WORDS)\\s+)?(?<piece>$PIECE_WORD)s?\\s+(?:on|at)\\s+" +
        "(?<square>[a-h][1-8])\\b",
    RegexOption.IGNORE_CASE,
)
private val SQUARE_PIECE = Regex(
    "\\b(?:(?<colour>$COLOUR_WORDS)\\s+)?(?<square>[a-h][1-8])[- ](?<piece>$PIECE_WORD)\\b",
    RegexOption.IGNORE_CASE,
)

// A token is a bare square ("f7") or a SAN move ending in its destination square ("Ng5",
// "Nxf7+", "exd5", "R1e1", "e8=Q"); the same idea as `disclosure.squaresIn`. Group 1 is
// the destination square. Splitting on everything else gives token boundaries for free, so
// "area1" or "Be careful" never match.
private val SAN_TOKEN = Regex("[KQRBN]?[a-h]?[1-8]?x?([a-h][1-8])(?:=[QRBN])?[+#]?[!?]*")
private val TOKEN_SPLIT = Regex("[^A-Za-z0-9+#=!?]+")
private val CASTLE = Regex("(?<![A-Za-z0-9-])(O-O-O|O-O|0-0-0|0-0)[+#]?(?![A-Za-z0-9-])")

// A tactical adjective claimed of a piece on a square: "the bishop on b3 is pinned", "your
// rook on d1 is hanging". The coach must not assert these unless the board bears them out.
private val TACTIC = Regex(
    "(?:(?<colour>$COLOUR_WORDS)\\s+)?(?<piece>$PIECE_WORD)s?\\s+(?:on|at)\\s+" +
        "(?<square>[a-h][1-8])\\s+" +
        "(?:(?:is|was|are|were)\\s+(?:now\\s+)?|,\\s*(?:which\\s+)?(?:is|was)\\s+|now\\s+)?" +
        "(?<state>pinned|hanging|undefended|unprotected|loose|en\\s*prise|trapped|safe|" +
        "forks?|forking|skewers?|skewering)",
    RegexOption.IGNORE_CASE,
)

private val SLIDERS = intArrayOf(BISHOP, ROOK, QUEEN)

/**
 * The friendly piece a piece on `square` shields, or null if it shields nothing.
 *
 * A pin, absolute or relative, by the inventory's rule ([pinTarget]), so a pin the coach was
 * told about is not called false: an enemy slider that can legally take the piece on `square`
 * attacks it along a line, and the next piece past it on that line is a friendly piece worth
 * more that the slider would win once the front one moved -- the king, or a piece the capture
 * wins (worth more than the slider, or undefended). A knight in front of a defended rook pins
 * nothing.
 */
private fun pinLine(board: Board, square: Int): Int? {
    val piece = board.pieceAt(square) ?: return null
    if (piece.pieceType == KING) return null
    for (attacker in Exchange.capturers(board, !piece.colour, square)) {
        if (board.pieceTypeAt(attacker) !in SLIDERS) continue
        val behind = beyond(board, attacker, square)
        val occupant = if (behind != null) board.pieceAt(behind) else null
        if (behind != null &&
            occupant != null &&
            occupant.colour == piece.colour &&
            VALUE[occupant.pieceType] > VALUE[piece.pieceType] &&
            wins(board, attacker, behind, without = square)
        ) {
            return behind
        }
    }
    return null
}

private fun sign(n: Int): Int = if (n > 0) 1 else if (n < 0) -1 else 0

/** The first occupied square past `through` on the line from `frm`, or null. */
private fun beyond(board: Board, frm: Int, through: Int): Int? {
    val df = sign(squareFile(through) - squareFile(frm))
    val dr = sign(squareRank(through) - squareRank(frm))
    if (df == 0 && dr == 0) return null
    var file = squareFile(through) + df
    var rank = squareRank(through) + dr
    while (file in 0..7 && rank in 0..7) {
        val sq = square(file, rank)
        if (board.pieceAt(sq) != null) return sq
        file += df
        rank += dr
    }
    return null
}

/**
 * Whether the piece on `frm` wins the piece on `target` by taking it, with the piece on
 * `without` stepped off the line: the target is the king, or the capture is legal and nets
 * material once the recaptures are played out ([Exchange.exchange]).
 */
private fun wins(board: Board, frm: Int, target: Int, without: Int? = null): Boolean {
    if (board.pieceTypeAt(target) == KING) return true
    val colour = board.colourAt(frm) ?: return false
    var probe = board.copy(withStack = false)
    if (without != null) probe.removePieceAt(without)
    probe = toMove(probe, colour)
    var capture: Move? = null
    probe.generateLegalMoves(BB_SQUARES[frm], BB_SQUARES[target]) { if (capture == null) capture = it }
    val found = capture ?: return false
    return Exchange.exchange(probe, found) > 0
}

/**
 * Enemy pieces a fork on `square` would win, by the inventory's rule ([forksInPlace]): the king
 * it checks, or a piece it can legally take that can be won -- worth more than the forker, or
 * not defended well enough. A pinned knight takes nothing, so it forks nothing.
 */
private fun forkTargets(board: Board, square: Int): List<Int> {
    val piece = board.pieceAt(square) ?: return emptyList()
    val targets = ArrayList<Int>()
    for (target in board.attacks(square)) {
        val other = board.pieceAt(target)
        if (other == null || other.colour == piece.colour) continue
        if (other.pieceType == KING ||
            (square in Exchange.capturers(board, piece.colour, target) && Exchange.canBeWon(board, target))
        ) {
            targets.add(target)
        }
    }
    return targets
}

/**
 * Whether the slider on `square` skewers, by the inventory's rule ([skewerTarget]): it attacks
 * an enemy piece that must step aside -- the king, or one it can legally take and win (worth
 * more than the slider, or undefended) -- and past it on the same line stands a less valuable
 * enemy piece the slider then wins.
 */
private fun skewers(board: Board, square: Int): Boolean {
    val piece = board.pieceAt(square) ?: return false
    if (piece.pieceType !in SLIDERS) return false
    for (front in board.attacks(square)) {
        val target = board.pieceAt(front)
        if (target == null || target.colour == piece.colour || !wins(board, square, front)) continue
        val behind = beyond(board, square, front)
        val occupant = if (behind != null) board.pieceAt(behind) else null
        if (behind != null &&
            occupant != null &&
            occupant.colour != piece.colour &&
            VALUE[occupant.pieceType] < VALUE[target.pieceType] &&
            wins(board, square, behind, without = front)
        ) {
            return true
        }
    }
    return false
}

/**
 * Whether the board bears out a tactic claimed of a piece; null when it does or might.
 *
 * Only plainly false claims are faulted: a pin that is not there (to any piece, not only the
 * king), a piece called loose that is defended, a piece called safe that can be won, a fork or
 * skewer the piece does not make. Attack and defence are legal moves: a piece pinned to its
 * king neither takes nor takes back, and a king takes back only where it may.
 */
private fun tacticProblem(board: Board, squareName: String, rawState: String): String? {
    val square = parseSquare(squareName.lowercase())
    val piece = board.pieceAt(square) ?: return "$squareName is empty"
    val name = COACH_PIECE_NAMES[piece.pieceType]
    val state = rawState.lowercase().replace(" ", "").trimEnd('s')
    when {
        state == "pinned" ->
            if (pinLine(board, square) == null) return "the $name on $squareName is not pinned"
        state == "hanging" || state == "enprise" -> {
            if (Exchange.capturers(board, !piece.colour, square).isEmpty()) {
                if (board.attackers(!piece.colour, square).isNotEmpty()) {
                    return "nothing can take the $name on $squareName"
                }
                return "nothing attacks the $name on $squareName"
            }
            if (!Exchange.hanging(board, square)) return "the $name on $squareName is defended"
        }
        state == "undefended" || state == "unprotected" || state == "loose" ->
            // A piece with no defender is undefended whether or not anything attacks it yet.
            if (guards(board, square).isNotEmpty()) return "the $name on $squareName is defended"
        state == "safe" -> {
            if (Exchange.hanging(board, square)) {
                return "the $name on $squareName is attacked and undefended"
            }
            if (Exchange.canBeWon(board, square)) {
                return "the $name on $squareName can be taken for profit"
            }
        }
        state == "trapped" && board.attackers(!piece.colour, square).isEmpty() ->
            return "nothing is attacking the $name on $squareName"
        state == "fork" || state == "forking" -> {
            if (forkTargets(board, square).size < 2) {
                if (board.isPinned(piece.colour, square)) {
                    return "the $name on $squareName is pinned, so it forks nothing"
                }
                return "the $name on $squareName does not fork two pieces"
            }
            if (Exchange.canBeWon(board, square)) {
                return "the $name on $squareName can be taken for profit, so it forks nothing"
            }
        }
        (state == "skewer" || state == "skewering") && !skewers(board, square) ->
            return "the $name on $squareName does not skewer anything"
    }
    return null
}

/**
 * Everything concrete in a coach draft that the board contradicts. Empty means clean.
 *
 * A "<piece> on <square>" phrase must match the board, colour word included ("your" is [us],
 * the player being coached, the side to move when not given: on the opponent's turn "your
 * knight on f3" is the player's, and read as the side to move's it was flagged false -- the
 * code review, 2026-09-26). A tactic claimed of such a piece (pinned, hanging, safe, a fork, a
 * skewer) must hold on the board, or on one of the [after] positions, which are the boards
 * along the best line: a fork the best move creates is real even though the current board
 * does not show it yet. A move must be playable on one of the frames: legal now, legal
 * for the opponent after a pass (a coach may name a threat), or legal further along the
 * line the coach was given. Bare squares are never flagged; a coach may point at one.
 */
fun checkDraft(
    text: String,
    board: Board,
    after: List<Board> = emptyList(),
    us: Boolean? = null,
): List<DraftProblem> {
    val problems = ArrayList<DraftProblem>()
    val seen = HashSet<String>()
    val frames = listOf(board) + after
    val us = us ?: board.turn

    fun add(phrase: String, reason: String) {
        if (seen.add(phrase)) problems.add(DraftProblem(text = phrase, reason = reason))
    }

    for (pattern in listOf(PIECE_ON, SQUARE_PIECE)) {
        for (match in pattern.findAll(text)) {
            // A placement is wrong only when it is wrong on every frame: a piece the best
            // move puts on a square is really there in the line the coach explains.
            val reasons = frames.map {
                placementProblem(
                    it,
                    match.groups["colour"]?.value,
                    match.groups["piece"]!!.value,
                    match.groups["square"]!!.value,
                    us,
                )
            }
            if (reasons.all { it != null }) add(match.value, reasons[0] ?: "")
        }
    }

    for (match in TACTIC.findAll(text)) {
        val square = match.groups["square"]!!.value
        val state = match.groups["state"]!!.value
        val present = frames.map {
            placementProblem(it, null, match.groups["piece"]!!.value, square, us)
        }
        // The piece is on that square on no frame; the placement loop has it.
        if (present.all { it != null }) continue
        val reasons = frames.map { tacticProblem(it, square, state) }
        if (reasons.any { it == null }) continue
        val clean = reasons.firstOrNull { !it.isNullOrEmpty() }
        if (!clean.isNullOrEmpty()) add(match.value, clean)
    }

    for (token in TOKEN_SPLIT.split(text)) {
        val match = SAN_TOKEN.matchEntire(token) ?: continue
        val core = TRAILING_MARKS.replace(token, "")
        if (core == match.groupValues[1]) continue // a bare square
        val reasons = frames.map { moveProblem(it, core) }
        if (reasons.all { it != null }) add(token, reasons[0] ?: "")
    }
    for (match in CASTLE.findAll(text)) {
        val san = match.groupValues[1].replace("0", "O")
        val reasons = frames.map { moveProblem(it, san) }
        if (reasons.all { it != null }) add(match.value, reasons[0] ?: "")
    }
    for ((phrase, reason) in strategicProblems(text, board, us)) add(phrase, reason)
    for ((phrase, reason) in lineProblems(text, frames, us)) add(phrase, reason)
    return problems
}

// What a move is said to force: the king away, or a piece onto a square. "The check forced the
// King away" where a block was legal, "the check on d6 forces your rook onto f7" where other
// moves were: 2 of the 5 false claims the bench's labellers found (2026-09-25).
private val FORCE_KING = Regex(
    "\\bforc(?:e|es|ed|ing)\\s+(?<colour>your|their|the|his|white's|black's)\\s+king\\s+" +
        "(?:away|back|out|off|to\\s+move|to\\s+step\\s+aside|to\\s+(?<square>[a-h][1-8]))\\b" +
        "|\\b(?<colour2>your|their)\\s+king\\s+(?:has\\s+to|must|is\\s+forced\\s+to)\\s+move\\b",
    RegexOption.IGNORE_CASE,
)
private val FORCE_ONTO = Regex(
    "\\bforc(?:e|es|ed|ing)\\s+(?<colour>your|their|the|his|white's|black's)\\s+" +
        "(?<piece>queen|rook|bishop|knight|pawn)\\s+(?:back\\s+)?(?:onto|to|into)\\s+" +
        "(?<square>[a-h][1-8])\\b",
    RegexOption.IGNORE_CASE,
)

// Files said to be kept open, or to be open, more than one at a time: "Rad8 keeps the b-file
// and e-file open", "the b- and e-files are open". One file alone is [OPEN_FILE]'s.
private val FILES_OPEN = Regex(
    "\\bkeep(?:s|ing)?\\s+(?<body>(?:the\\s+)?[a-h]-(?:file)?(?:\\s*(?:,|and)\\s*(?:the\\s+)?" +
        "[a-h]-(?:file)?)*s?)\\s+open\\b" +
        "|(?<body2>\\b[a-h]-(?:file)?(?:\\s*(?:,|and)\\s*(?:the\\s+)?[a-h]-(?:file)?)+s?)\\s+" +
        "(?:are|stay|remain)\\s+open\\b",
    RegexOption.IGNORE_CASE,
)
private val SAN_CHECK = Regex(
    "\\b(?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8]|[a-h](?:x[a-h])?[1-8])(?:=[QRBN])?\\+",
)
private val SENTENCE_BREAK = Regex("(?<=[.!?])\\s+")
private val FILE_LETTER = Regex("\\b([a-h])-")

/** The side a colour word names: "your" is [us]; "the" and "his" either side (null). */
private fun sideOf(word: String?, us: Boolean): Boolean? {
    val said = (word ?: "").lowercase()
    return when {
        said == "your" -> us
        said == "their" -> !us
        said.startsWith("white") -> WHITE
        said.startsWith("black") -> BLACK
        else -> null
    }
}

/**
 * The frames where [side] (either, when null) is in check: those given, and those after a check
 * the sentence names ("Rh4+"), played where it is legal. `claims._checked_frames`.
 */
private fun checkedFrames(text: String, frames: List<Board>, side: Boolean?): List<Board> {
    val out = frames.filter { it.isCheck() && (side == null || it.turn == side) }.toMutableList()
    for (token in SAN_CHECK.findAll(text).map { it.value }) {
        for (frame in frames) {
            val move = frame.parseSan(token) ?: continue
            val after = frame.copy(withStack = false)
            after.push(move)
            if (after.isCheck() && (side == null || after.turn == side)) out.add(after)
        }
    }
    return out
}

/**
 * What a line is said to force, or to keep open, that none of the frames bears out. A claim no
 * frame applies to -- no check on the board or the line for that side -- is left alone: it is
 * about some other moment of the game. `claims._line_problems`.
 */
private fun lineProblems(text: String, frames: List<Board>, us: Boolean): List<Pair<String, String>> {
    val problems = ArrayList<Pair<String, String>>()
    for (sentenceText in text.split(SENTENCE_BREAK)) {
        for (match in FORCE_KING.findAll(sentenceText)) {
            val side = sideOf(match.groups["colour"]?.value ?: match.groups["colour2"]?.value, us)
            val checked = checkedFrames(sentenceText, frames, side)
            if (checked.isEmpty()) continue
            if (checked.any { f -> f.legalMoves().all { f.pieceTypeAt(it.from) == KING } }) continue
            problems.add(
                match.value to
                    "the check does not force the king to move: it can be blocked " +
                    "or the checking piece taken",
            )
        }
        for (match in FORCE_ONTO.findAll(sentenceText)) {
            val side = sideOf(match.groups["colour"]?.value, us)
            val piece = match.groups["piece"]!!.value.lowercase()
            val squareText = match.groups["square"]!!.value.lowercase()
            val kind = PIECE_WORDS[piece]
            val square = parseSquare(squareText)
            fun lands(f: Board, m: Move): Boolean = m.to == square && f.pieceTypeAt(m.from) == kind
            val candidates = (frames + checkedFrames(sentenceText, frames, side)).filter { f ->
                (side == null || f.turn == side) && f.legalMoves().any { lands(f, it) }
            }
            if (candidates.isEmpty()) continue
            if (candidates.any { f -> f.legalMoves().all { lands(f, it) } }) continue
            problems.add(
                match.value to "nothing forces the $piece to $squareText: other moves are legal there",
            )
        }
    }
    for (match in FILES_OPEN.findAll(text)) {
        val body = match.groups["body"]?.value ?: match.groups["body2"]?.value ?: ""
        for (letter in FILE_LETTER.findAll(body.lowercase()).map { it.groupValues[1] }.distinct()) {
            val index = letter[0] - 'a'
            if (frames.none { f -> fileSquares(index).all { f.pieceTypeAt(it) != PAWN } }) {
                problems.add(match.value to "the $letter-file is not open; a pawn stands on it")
            }
        }
    }
    return problems
}

// A file called open or half-open, in either word order ("the open e-file", "the e-file is
// open"). A pawn called passed, by square or by file, in either order.
private val OPEN_FILE = Regex(
    "\\b(?:(?<k1>open|half[-\\s]?open)\\s+(?<f1>[a-h])[-\\s]?file" +
        "|(?<f2>[a-h])[-\\s]?file\\s+(?:is|are|remains?)\\s+(?:now\\s+)?" +
        "(?<k2>open|half[-\\s]?open))\\b",
    RegexOption.IGNORE_CASE,
)
private val PASSED = Regex(
    "\\bpassed\\s+pawn\\s+on\\s+(?<sq1>[a-h][1-8])" +
        "|\\bpawn\\s+on\\s+(?<sq2>[a-h][1-8])\\s+(?:is|are)\\s+(?:a\\s+|now\\s+)*passed" +
        "|\\bpassed\\s+(?<file1>[a-h])[-\\s]?pawn" +
        "|\\b(?<file2>[a-h])[-\\s]?pawn\\s+(?:is|are)\\s+(?:a\\s+|now\\s+)*passed",
    RegexOption.IGNORE_CASE,
)

private fun fileSquares(fileIndex: Int): List<Int> = (0..7).map { square(fileIndex, it) }

/** Whose half-open file the sentence says: theirs, or the player's. */
private val THEIRS_SAID = Regex("\\bopponent|\\btheir\\b|\\bfor them\\b")
private val YOURS_SAID = Regex("\\byour?\\b|\\bfor you\\b")

/**
 * Concrete strategic claims the board contradicts: a file said open that has pawns, a pawn
 * said passed that an enemy pawn still holds up. Only clear, checkable contradictions are
 * flagged; a fuzzy claim (space, king safety, activity) is left alone.
 */
private fun strategicProblems(text: String, board: Board, us: Boolean? = null): List<Pair<String, String>> {
    val us = us ?: board.turn
    val problems = ArrayList<Pair<String, String>>()
    for (match in OPEN_FILE.findAll(text)) {
        val kind = (match.groups["k1"]?.value ?: match.groups["k2"]!!.value)
            .lowercase().replace(" ", "").replace("-", "")
        val letter = (match.groups["f1"]?.value ?: match.groups["f2"]!!.value).lowercase()
        val pawns = fileSquares(letter[0] - 'a').map { board.pieceAt(it) }
        val ours = pawns.any { it != null && it.pieceType == PAWN && it.colour == us }
        val theirs = pawns.any { it != null && it.pieceType == PAWN && it.colour != us }
        if (kind == "open" && (ours || theirs)) {
            problems.add(match.value to "the $letter-file is not open; a pawn stands on it")
        } else if (kind == "halfopen") {
            // Whose file it is, as the sentence says: "for you" or "your" is the player's, "their"
            // or "the opponent's" theirs ("your opponent takes the half-open f-file": round 2
            // refused two good replies read as the player's); unsaid, either side's.
            val window = text.substring(
                maxOf(0, match.range.first - 60),
                minOf(text.length, match.range.last + 1 + 20),
            ).lowercase()
            val (wrong, whose) = if (THEIRS_SAID.containsMatchIn(window)) {
                theirs to "for them; they have a pawn there"
            } else if (YOURS_SAID.containsMatchIn(window)) {
                ours to "for you; you have a pawn there"
            } else {
                (ours && theirs) to "for either side: both have a pawn there"
            }
            if (wrong) problems.add(match.value to "the $letter-file is not half-open $whose")
        }
    }
    for (match in PASSED.findAll(text)) {
        val squareText = match.groups["sq1"]?.value ?: match.groups["sq2"]?.value
        val letter = match.groups["file1"]?.value ?: match.groups["file2"]?.value
        if (squareText != null) {
            val square = parseSquare(squareText.lowercase())
            val piece = board.pieceAt(square)
            if (piece != null && piece.pieceType == PAWN && !isPassed(board, square)) {
                problems.add(
                    match.value to "the pawn on ${squareText.lowercase()} is not passed"
                )
            }
        } else if (letter != null) {
            val ours = fileSquares(letter.lowercase()[0] - 'a').filter {
                board.pieceTypeAt(it) == PAWN && board.colourAt(it) == us
            }
            if (ours.isNotEmpty() && ours.none { isPassed(board, it) }) {
                problems.add(
                    match.value to "your ${letter.lowercase()}-pawn is not a passed pawn"
                )
            }
        }
    }
    return problems
}

/**
 * Whether a "<colour> <piece> on <square>" claim holds. [us] fixes whose "your" is.
 *
 * Across the current board and the board after the best move, whose turn it is flips, but
 * "your" always means the player being coached, so [us] is passed rather than read from
 * the frame.
 */
private fun placementProblem(
    board: Board,
    colourWord: String?,
    pieceWord: String,
    squareText: String,
    us: Boolean? = null,
): String? {
    val ours = us ?: board.turn
    val square = parseSquare(squareText.lowercase())
    val piece = board.pieceAt(square)
    val wanted = PIECE_WORDS[pieceWord.lowercase()]!!
    if (piece == null) return "$squareText is empty"
    if (piece.pieceType != wanted) {
        val whose = if (piece.colour == ours) "your" else "their"
        return "$squareText holds $whose ${COACH_PIECE_NAMES[piece.pieceType]}, " +
            "not a ${pieceWord.lowercase()}"
    }
    if (colourWord == null) return null
    val word = colourWord.lowercase()
    val expected = when {
        word == "your" -> ours
        word.startsWith("white") -> WHITE
        word.startsWith("black") -> BLACK
        else -> !ours
    }
    if (piece.colour == expected) return null
    if (word.startsWith("white") || word.startsWith("black")) {
        val actual = if (piece.colour == WHITE) "white's" else "black's"
        return "the ${pieceWord.lowercase()} on $squareText is $actual, not $word"
    }
    if (piece.colour == ours) {
        return "the ${pieceWord.lowercase()} on $squareText is yours, not your opponent's"
    }
    return "the ${pieceWord.lowercase()} on $squareText is your opponent's, not yours"
}

private fun moveProblem(board: Board, san: String): String? {
    val (move, why) = parseMove(board, san)
    if (move != null) return null
    if (why == "ambiguous") return "$san could be more than one move here"
    if (!board.isCheck()) {
        val passed = board.copy(withStack = false)
        passed.push(Move.NULL)
        val threat = parseMove(passed, san).first
        if (threat != null) return null
    }
    return "$san cannot be played here by either side"
}
