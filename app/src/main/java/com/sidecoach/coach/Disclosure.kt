package com.sidecoach.coach

import com.sidecoach.events.DisclosureLevel

/**
 * Disclosure state machine and fact-bundle filtering: `sidecoach/coach/disclosure.py`.
 *
 * Hiding is by construction: the LLM receives only the facts allowed at the current level, so
 * it cannot repeat what it never saw. The position itself (FEN) is always visible because the
 * coach needs it to ask sensible questions; [leakCheck] is the second line of defence over the
 * coach's draft text for anything it might work out on its own.
 *
 * Levels, from DESIGN.md section 5.5:
 * ```
 * 0 SILENT     nothing yet
 * 1 QUESTION   an orienting question
 * 2 THEME      the theme category
 * 3 REGION     a board region; autonomous ceiling
 * 4 PIECE      the piece carrying the idea; only on a hint request
 * 5 CANDIDATE  the move or near-equal set; user named it, played it, or asked
 * 6 LINE       the full line; user asked for it, or after the move
 * ```
 */

val AUTONOMOUS_CEILING = DisclosureLevel.REGION
val HINT_CEILING = DisclosureLevel.PIECE

enum class Unlock(val slug: String) {
    STUCK("stuck"),
    DECISIVE("decisive"),
    HINT("hint"),
    ASK_MOVE("ask_move"),
    ASK_LINE("ask_line"),
    USER_NAMED_MOVE("user_named_move"),
    MOVE_PLAYED("move_played"),
}

/** Everything analysis knows about one position, before filtering. */
data class PositionFacts(
    val fen: String,
    val sideToMove: String,
    /**
     * The side the coach speaks to, when the position is not that side's to move (the
     * opponent is thinking); null means the side to move.
     */
    val userSide: String? = null,
    val phase: String,
    val material: String? = null,
    val lastMove: String? = null,
    val decisive: Boolean = false,
    val onlyMove: Boolean = false,
    /**
     * Whether the best idea is forcing -- a check, a capture, a promotion -- or quiet. Given
     * from the question level: "have you looked at every check and capture?" is the first
     * thing a coach asks a stuck player, and it points at no square, piece or move.
     */
    val nature: String? = null,
    val theme: String? = null,
    val region: String? = null,
    val piece: String? = null,
    /**
     * The squares that carry the theme: the pinner, the pinned piece and what it shields; the
     * forking piece and its targets. Given only at the piece level and above.
     */
    val focus: List<String> = emptyList(),
    /**
     * The theme as a finished sentence, grounded on the board. Given at the piece level and
     * above, like focus.
     */
    val tactic: String? = null,
    /**
     * The strategic idea behind the best move, as a square-free theme. Given from the theme
     * level, since it orients without naming the move.
     */
    val plan: String? = null,
    /** The most instructive static features of the position, in the coach's words. */
    val ideas: List<String> = emptyList(),
    val candidates: List<String> = emptyList(),
    val threats: List<String> = emptyList(),
    val line: List<String> = emptyList(),
    val userClaims: List<Map<String, Any?>> = emptyList(),
) {
    /** The fields by the names the Python bundle uses, in declaration order. */
    fun asMap(): LinkedHashMap<String, Any?> {
        val out = LinkedHashMap<String, Any?>()
        out["fen"] = fen
        out["side_to_move"] = sideToMove
        out["user_side"] = userSide
        out["phase"] = phase
        out["material"] = material
        out["last_move"] = lastMove
        out["decisive"] = decisive
        out["only_move"] = onlyMove
        out["nature"] = nature
        out["theme"] = theme
        out["region"] = region
        out["piece"] = piece
        out["focus"] = focus
        out["tactic"] = tactic
        out["plan"] = plan
        out["ideas"] = ideas
        out["candidates"] = candidates
        out["threats"] = threats
        out["line"] = line
        out["user_claims"] = userClaims
        return out
    }
}

private val ALWAYS = listOf(
    "fen", "side_to_move", "user_side", "phase", "material", "last_move", "decisive",
    "only_move", "user_claims",
)

private val BY_LEVEL: Map<DisclosureLevel, List<String>> = mapOf(
    DisclosureLevel.SILENT to emptyList(),
    DisclosureLevel.QUESTION to listOf("nature"),
    DisclosureLevel.THEME to listOf("nature", "theme", "plan"),
    DisclosureLevel.REGION to listOf("nature", "theme", "plan", "ideas", "region"),
    DisclosureLevel.PIECE to listOf(
        "nature", "theme", "plan", "ideas", "region", "piece", "focus", "tactic",
    ),
    DisclosureLevel.CANDIDATE to listOf(
        "nature", "theme", "plan", "ideas", "region", "piece", "focus", "tactic",
        "candidates", "threats",
    ),
    DisclosureLevel.LINE to listOf(
        "nature", "theme", "plan", "ideas", "region", "piece", "focus", "tactic",
        "candidates", "threats", "line",
    ),
)

/** The fact bundle the LLM may see at `level`. */
fun visibleFacts(facts: PositionFacts, level: DisclosureLevel): LinkedHashMap<String, Any?> {
    val allowed = ALWAYS.toSet() + BY_LEVEL.getValue(level).toSet()
    val out = LinkedHashMap<String, Any?>()
    for ((key, value) in facts.asMap()) if (key in allowed) out[key] = value
    return out
}

/** Per-position disclosure state. Create a fresh one for every new position. */
class Disclosure(var level: DisclosureLevel = DisclosureLevel.SILENT) {

    val history = ArrayList<Pair<Unlock, DisclosureLevel>>()

    /** Apply an unlock. Returns the new level, or null when nothing changed. */
    fun apply(unlock: Unlock): DisclosureLevel? {
        val before = level
        when (unlock) {
            Unlock.STUCK ->
                // A player stuck for a while gets the theme at once, the region next.
                if (level.value < AUTONOMOUS_CEILING.value) {
                    level = DisclosureLevel.of(maxOf(level.value + 1, DisclosureLevel.THEME.value))
                }
            Unlock.DECISIVE ->
                if (level.value < AUTONOMOUS_CEILING.value) {
                    level = DisclosureLevel.of(level.value + 1)
                }
            Unlock.HINT ->
                if (level.value < HINT_CEILING.value) {
                    level = DisclosureLevel.of(level.value + 1)
                }
            Unlock.ASK_MOVE, Unlock.USER_NAMED_MOVE ->
                level = DisclosureLevel.of(maxOf(level.value, DisclosureLevel.CANDIDATE.value))
            Unlock.ASK_LINE, Unlock.MOVE_PLAYED ->
                level = DisclosureLevel.LINE
        }
        if (level == before) return null
        history.add(unlock to level)
        return level
    }
}

private const val PIECE_RE = "(?:king|queen|rook|bishop|knight|pawn)"

/**
 * Where a piece word is never a hint at the piece to move: an opening's name, the material
 * balance, a check or capture said of the move just played.
 */
private val PIECE_WORD_CONTEXTS = Regex(
    "(?i)" +
        "(?:king's|queen's|bishop's|knight's|rook's) " +
        "(?:indian|gambit|pawn|opening|attack|defen[cs]e)" +
        "|(?:up|down|ahead|behind|level|even|for|against|exchange)(?: a| an| two| three| the)? " +
        "(?:$PIECE_RE|piece)s?(?: (?:and|for) (?:a |an |two |three |the )?(?:$PIECE_RE|piece)s?)*" +
        "|(?:checks?|attacks?|takes|took|captures?d?|removes?d?|hits?) (?:your|their|the|my) " +
        "$PIECE_RE on [a-h][1-8]" +
        "|(?:your|their|my|the) $PIECE_RE on [a-h][1-8] (?:is|was|remains|stays) (?:in )?" +
        "(?:check|attacked|under attack|hanging|loose|defended|undefended|pinned)",
)

/**
 * A token is a bare square ("f7") or a SAN move ending in its destination square ("Ng5",
 * "Nxf7+", "exd5", "R1e1", "e8=Q"). Group 1 is the destination square.
 */
private val SQUARE_TOKEN = Regex("[KQRBN]?[a-h]?[1-8]?x?([a-h][1-8])(?:=[QRBN])?[+#]?[!?]*")
private val TOKEN_SPLIT = Regex("[^A-Za-z0-9+#=!?]+")
private val TRAILING_MARKS = Regex("[+#!?]+$")
private val WORD_RE = Regex("[a-z0-9=+#-]+")

/** Destination squares of every bare square or SAN token in `text`. */
fun squaresIn(text: String): Set<String> {
    val found = LinkedHashSet<String>()
    for (token in TOKEN_SPLIT.split(text)) {
        val match = SQUARE_TOKEN.matchEntire(token) ?: continue
        found.add(match.groupValues[1])
    }
    return found
}

/**
 * Withheld items that `text` mentions. An empty list means the draft is clean.
 *
 * Below CANDIDATE the coach may not name any move at all, and may mention only squares that
 * occur in visible facts other than the FEN, such as a region anchor. The model will
 * volunteer moves on its own even when told not to, so this is not limited to the withheld
 * facts. From CANDIDATE up only the withheld line is checked. Below PIECE the piece word from
 * the facts is checked as well.
 *
 * `said` is what the user asked this turn: a square, piece word or move they named is theirs
 * already, so confirming it back is not a leak and never faulted.
 */
fun leakCheck(
    text: String,
    facts: PositionFacts,
    level: DisclosureLevel,
    said: String? = null,
): List<String> {
    val visible = visibleFacts(facts, level)
    val allowedSquares = HashSet<String>()
    for ((key, value) in visible) {
        if (key == "fen") continue
        if (value == null) continue
        if (value is Boolean && !value) continue
        if (value is String && value.isEmpty()) continue
        if (value is List<*> && value.isEmpty()) continue
        val parts = if (value is List<*>) value else listOf(value)
        allowedSquares.addAll(squaresIn(parts.joinToString(" ") { it.toString() }))
    }
    val saidSquares = if (said != null) squaresIn(said) else emptySet()
    // A move said with its check or mate mark ("Qh4+") is the same move said without it.
    val saidWords = HashSet<String>()
    if (said != null) {
        for (match in WORD_RE.findAll(said.lowercase())) {
            saidWords.add(match.value)
            saidWords.add(match.value.trimEnd('+', '#'))
        }
    }
    allowedSquares.addAll(saidSquares)

    val leaks = ArrayList<String>()
    fun add(item: String) {
        if (item !in leaks) leaks.add(item)
    }

    if (level.value < DisclosureLevel.CANDIDATE.value) {
        for (token in TOKEN_SPLIT.split(text)) {
            val match = SQUARE_TOKEN.matchEntire(token) ?: continue
            val square = match.groupValues[1]
            val core = TRAILING_MARKS.replace(token, "")
            if (core != square) {
                if (core.lowercase() !in saidWords) add("move $core")
            } else if (square !in allowedSquares) {
                add("square $square")
            }
        }
    } else if (level.value < DisclosureLevel.LINE.value) {
        val shown = facts.candidates.toSet() + facts.threats.toSet()
        val withheldMoves = facts.line.filter { it !in shown }
        val withheldSquares = squaresIn(withheldMoves.joinToString(" ")) - allowedSquares
        for (square in (squaresIn(text) intersect withheldSquares).sorted()) add("square $square")
        for (move in withheldMoves) {
            if (Regex("(?<![A-Za-z])${Regex.escape(move)}(?![A-Za-z0-9])").containsMatchIn(text)) {
                add("move $move")
            }
        }
    }

    if (level.value < DisclosureLevel.PIECE.value && !facts.piece.isNullOrEmpty()) {
        val pieceWord = facts.piece.lowercase().split(" ").last()
        // The word is a hint only where it could point at the piece to move: not in an
        // opening's name, not in the material balance, not said of a move already played.
        val scrubbed = PIECE_WORD_CONTEXTS.replace(text.lowercase(), " ")
        if (pieceWord !in saidWords &&
            Regex("\\b${Regex.escape(pieceWord)}\\b").containsMatchIn(scrubbed)
        ) {
            add("piece $pieceWord")
        }
    }
    return leaks
}
