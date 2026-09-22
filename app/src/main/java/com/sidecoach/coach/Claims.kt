package com.sidecoach.coach

import com.sidecoach.analysis.isPassed
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
import com.sidecoach.chess.squareDistance
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
 * nothing can be made up.
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
            if (hit != null && hit.colour != board.colourAt(subject)) {
                val defenders = board.attackers(hit.colour, target)
                reason += if (defenders.isNotEmpty()) {
                    ", though it is defended by ${pos.listed(defenders)}"
                } else {
                    ", and nothing defends it"
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
                val others = board.attackers(held.colour, target).filter { it != subject }
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
        val attackers = board.attackers(!piece.colour, target)
        val defenders = board.attackers(piece.colour, target)
        if (attackers.isNotEmpty() && defenders.isEmpty()) {
            val reason = "${pos.name(target)} is attacked by ${pos.listed(attackers)} " +
                "and nothing defends it"
            return ClaimResult.TRUE to sentence(afterMove(pos, reason))
        }
        var reason: String
        if (attackers.isEmpty()) {
            reason = "${pos.name(target)} is not attacked at all"
        } else {
            reason = "${pos.name(target)} is attacked by ${pos.listed(attackers)} but " +
                "defended by ${pos.listed(defenders)}, so it is not hanging outright"
            val cheapest = attackers.minByOrNull { VALUE[board.pieceTypeAt(it)] }!!
            reason += if (VALUE[board.pieceTypeAt(cheapest)] < VALUE[piece.pieceType]) {
                ", though ${pos.short(cheapest)} taking a " +
                    "${COACH_PIECE_NAMES[piece.pieceType]} still wins material"
            } else {
                "; I have not counted the whole exchange"
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

/** Whether the piece on `target` (or the empty square) can be taken, with the counts. */
private fun safety(pos: Pos, target: Int): Pair<Boolean, String> {
    val board = pos.board
    val square = squareName(target)
    val piece = board.pieceAt(target)
    if (piece == null) {
        val attackers = board.attackers(pos.them, target)
        if (attackers.isEmpty()) return true to "$square is not attacked by any of their pieces"
        return false to "$square is attacked by ${pos.listed(attackers)}"
    }
    val attackers = board.attackers(!piece.colour, target)
    val defenders = board.attackers(piece.colour, target)
    if (piece.pieceType == KING) {
        if (attackers.isEmpty()) return true to "${pos.name(target)} is not in check"
        return false to "${pos.name(target)} is in check from ${pos.listed(attackers)}"
    }
    if (attackers.isEmpty()) return true to "${pos.name(target)} is not attacked"
    var counts = "${pos.name(target)} is attacked ${times(attackers.size)} " +
        "(${pos.listed(attackers)}) and defended ${times(defenders.size)}"
    if (defenders.isNotEmpty()) counts += " (${pos.listed(defenders)})"
    if (defenders.size < attackers.size) return false to "$counts, so it can be taken"
    val cheapest = attackers.minByOrNull { VALUE[board.pieceTypeAt(it)] }!!
    if (VALUE[board.pieceTypeAt(cheapest)] < VALUE[piece.pieceType]) {
        return false to (
            "$counts, but ${pos.short(cheapest)} is worth less than a " +
                "${COACH_PIECE_NAMES[piece.pieceType]}, so it can still be taken profitably"
            )
    }
    return true to "$counts, so it holds"
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

/** "their bishop on c4 takes your pawn on f7, and your king on e8 can take back". */
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
    val recapturers = position.attackers(victim.colour, move.to)
    if (recapturers.isNotEmpty()) return "$what, and ${pos.listed(recapturers)} can take back"
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

/**
 * The friendly piece a piece on `square` shields, or null if it shields nothing.
 *
 * A pin, absolute or relative: an enemy slider attacks `square` along a line, and the next
 * piece past it on that line is a friendly piece worth more (the king counts as most). This
 * is the tagger's own test, so a true pin to the queen is not called false.
 */
private fun pinLine(board: Board, square: Int): Int? {
    val piece = board.pieceAt(square) ?: return null
    for (attacker in board.attackers(!piece.colour, square)) {
        val slider = board.pieceAt(attacker) ?: continue
        if (slider.pieceType != BISHOP && slider.pieceType != ROOK && slider.pieceType != QUEEN) {
            continue
        }
        if (squareDistance(attacker, square) == 0) continue
        val df = sign(squareFile(square) - squareFile(attacker))
        val dr = sign(squareRank(square) - squareRank(attacker))
        var file = squareFile(square) + df
        var rank = squareRank(square) + dr
        while (file in 0..7 && rank in 0..7) {
            val behind = square(file, rank)
            val occupant = board.pieceAt(behind)
            if (occupant != null) {
                if (occupant.colour == piece.colour &&
                    VALUE[occupant.pieceType] > VALUE[piece.pieceType]
                ) {
                    return behind
                }
                break
            }
            file += df
            rank += dr
        }
    }
    return null
}

private fun sign(n: Int): Int = if (n > 0) 1 else if (n < 0) -1 else 0

/** Enemy pieces a fork on `square` would win: worth as much as the forker, or the king. */
private fun forkTargets(board: Board, square: Int): List<Int> {
    val piece = board.pieceAt(square) ?: return emptyList()
    val ownValue = VALUE[piece.pieceType]
    val targets = ArrayList<Int>()
    for (target in board.attacks(square)) {
        val other = board.pieceAt(target)
        if (other == null || other.colour == piece.colour) continue
        if (other.pieceType == KING || VALUE[other.pieceType] >= ownValue) {
            targets.add(target)
        } else if (board.attackers(other.colour, target).isEmpty()) {
            targets.add(target) // an undefended lesser piece is still worth the fork
        }
    }
    return targets
}

/**
 * Whether the board bears out a tactic claimed of a piece; null when it does or might.
 *
 * Only plainly false claims are faulted: a pin that is not there (to any piece, not only
 * the king), a piece called loose that is defended, a piece called safe that hangs, a fork
 * or skewer the piece does not make. The exchange is not evaluated, so "safe" is faulted
 * only when the piece is attacked and not defended at all.
 */
private fun tacticProblem(board: Board, squareName: String, rawState: String): String? {
    val square = parseSquare(squareName.lowercase())
    val piece = board.pieceAt(square) ?: return "$squareName is empty"
    val name = COACH_PIECE_NAMES[piece.pieceType]
    val state = rawState.lowercase().replace(" ", "").trimEnd('s')
    val attackers = board.attackers(!piece.colour, square)
    val defenders = board.attackers(piece.colour, square)
    when {
        state == "pinned" ->
            if (pinLine(board, square) == null) return "the $name on $squareName is not pinned"
        state == "hanging" || state == "enprise" -> {
            if (attackers.isEmpty()) return "nothing attacks the $name on $squareName"
            if (defenders.isNotEmpty()) return "the $name on $squareName is defended"
        }
        state == "undefended" || state == "unprotected" || state == "loose" ->
            // A piece with no defender is undefended whether or not anything attacks it yet.
            if (defenders.isNotEmpty()) return "the $name on $squareName is defended"
        state == "safe" ->
            if (attackers.isNotEmpty() && defenders.isEmpty()) {
                return "the $name on $squareName is attacked and undefended"
            }
        state == "trapped" && attackers.isEmpty() ->
            return "nothing is attacking the $name on $squareName"
        state == "fork" || state == "forking" ->
            if (forkTargets(board, square).size < 2) {
                return "the $name on $squareName does not fork two pieces"
            }
        state == "skewer" || state == "skewering" -> {
            // A skewer is a pin the other way round: the enemy piece behind is worth less.
            val hits = board.attacks(square).filter {
                behindIsEnemy(board, square, it, piece.colour)
            }
            if (hits.isEmpty()) return "the $name on $squareName does not skewer anything"
        }
    }
    return null
}

/** Whether, past an enemy piece on `through`, the same line holds another enemy piece. */
private fun behindIsEnemy(board: Board, frm: Int, through: Int, mover: Boolean): Boolean {
    val occupant = board.pieceAt(through)
    if (occupant == null || occupant.colour == mover) return false
    val df = sign(squareFile(through) - squareFile(frm))
    val dr = sign(squareRank(through) - squareRank(frm))
    if (df == 0 && dr == 0) return false
    var file = squareFile(through) + df
    var rank = squareRank(through) + dr
    while (file in 0..7 && rank in 0..7) {
        val behind = board.pieceAt(square(file, rank))
        if (behind != null) return behind.colour != mover
        file += df
        rank += dr
    }
    return false
}

/**
 * Everything concrete in a coach draft that the board contradicts. Empty means clean.
 *
 * A "<piece> on <square>" phrase must match the board, colour word included ("your" is
 * the side to move). A tactic claimed of such a piece (pinned, hanging, safe, a fork, a
 * skewer) must hold on the board, or on one of the [after] positions, which are the boards
 * along the best line: a fork the best move creates is real even though the current board
 * does not show it yet. A move must be playable on one of the frames: legal now, legal
 * for the opponent after a pass (a coach may name a threat), or legal further along the
 * line the coach was given. Bare squares are never flagged; a coach may point at one.
 */
fun checkDraft(text: String, board: Board, after: List<Board> = emptyList()): List<DraftProblem> {
    val problems = ArrayList<DraftProblem>()
    val seen = HashSet<String>()
    val frames = listOf(board) + after
    val us = board.turn

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
    for ((phrase, reason) in strategicProblems(text, board)) add(phrase, reason)
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

/**
 * Concrete strategic claims the board contradicts: a file said open that has pawns, a pawn
 * said passed that an enemy pawn still holds up. Only clear, checkable contradictions are
 * flagged; a fuzzy claim (space, king safety, activity) is left alone.
 */
private fun strategicProblems(text: String, board: Board): List<Pair<String, String>> {
    val us = board.turn
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
        } else if (kind == "halfopen" && ours) {
            problems.add(
                match.value to "the $letter-file is not half-open for you; you have a pawn there"
            )
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
