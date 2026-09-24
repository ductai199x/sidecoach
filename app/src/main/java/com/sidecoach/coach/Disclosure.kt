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
 * Levels, from DESIGN.md section 5.5 -- two hints, then the answer (Tai, 2026-09-23: four
 * rungs before the move left each saying a sliver, and the slivers came out generic):
 * ```
 * 0 SILENT  nothing yet
 * 1 IDEA    the situation, as a short lesson on the position: what is at stake (only one
 *           move holds, a few keep the edge, nothing is urgent), the key fact named on the
 *           board (their threat, the target, the plan and its pieces), the principle behind
 *           it, the trap -- pieces, squares and their moves, never the player's move
 *           (Tai, 2026-09-24: a square-free first hint could only say "fight for the
 *           centre", and in a middlegame that teaches nothing)
 * 2 FOCUS   the piece that does it and the question that leads to the move; the
 *           autonomous ceiling
 * 3 MOVE    the move, with a correct explanation; the third hint, or asked, named or played
 * 4 LINE    the full line; asked for, or after the move
 * ```
 */

/**
 * Unprompted, the coach gives the idea and, if the player is still stuck, the focus; never
 * the move (Tai, 2026-09-23). Asked for hints, the third is the move.
 */
val AUTONOMOUS_CEILING = DisclosureLevel.FOCUS
val HINT_CEILING = DisclosureLevel.MOVE

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
    /**
     * What the best line is for, from the side to move: "you can force mate in two", "you can
     * win their queen", "your opponent has a threat to meet first". The first thing a puzzle
     * book says and the one the coach never did: without it the first word on a mate in one
     * was "what is your opponent's most dangerous idea?" (Tai, 2026-09-23). Given at the idea
     * level; it names no square, piece or move.
     */
    val goal: String? = null,
    val theme: String? = null,
    /**
     * The pattern at work and why it works here, square-free: "a sacrifice first drags their
     * king out, then a rook and a knight mate it on the edge". The idea level.
     */
    val mechanism: String? = null,
    val region: String? = null,
    /**
     * The target on the board, grounded: "their king on h8, with no free square", "their king
     * on c1 and their rook on e1: one check can hit both". The focus level.
     */
    val target: String? = null,
    val piece: String? = null,
    /** Why the move works, played out: "after Kxh7, Rh5 is mate". The move level. */
    val why: String? = null,
    /**
     * The question that sets the player looking the right way for this goal and pattern:
     * "Which two of their pieces could one of yours hit at once, and from where?". The idea
     * rung's question, for the templates and as the model's starting point.
     */
    val question: String? = null,
    /** The focus rung's question, from the target to the move: "Which square hits both?". */
    val nudge: String? = null,
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
    /**
     * The first hint's parts ([Situation]), named on the board: what is at stake ("only one
     * move holds here: even the next best move loses a knight"), the key fact ("they threaten
     * Nxe5, winning your pawn on e5"), the trap ("hxg4 looks like it wins a knight, but it is a
     * trap: ..."), the good moves when several are fine or the position is opening theory, and
     * the whole of it as the first hint says it.
     */
    val stakes: String? = null,
    val key: String? = null,
    val trap: String? = null,
    val choices: List<String> = emptyList(),
    val situation: String? = null,
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
        out["goal"] = goal
        out["theme"] = theme
        out["mechanism"] = mechanism
        out["region"] = region
        out["target"] = target
        out["piece"] = piece
        out["why"] = why
        out["question"] = question
        out["nudge"] = nudge
        out["focus"] = focus
        out["tactic"] = tactic
        out["plan"] = plan
        out["ideas"] = ideas
        out["candidates"] = candidates
        out["threats"] = threats
        out["line"] = line
        out["user_claims"] = userClaims
        out["stakes"] = stakes
        out["key"] = key
        out["trap"] = trap
        out["choices"] = choices
        out["situation"] = situation
        return out
    }
}

private val ALWAYS = listOf(
    "fen", "side_to_move", "user_side", "phase", "material", "last_move", "decisive",
    "only_move", "user_claims",
)

private val IDEA_KEYS = listOf(
    "nature",
    "goal",
    "theme",
    "mechanism",
    "question",
    "plan",
    "stakes",
    "key",
    "trap",
    "choices",
    "situation",
)
private val FOCUS_KEYS = IDEA_KEYS + listOf("target", "piece", "nudge", "region", "ideas")
private val MOVE_KEYS = FOCUS_KEYS + listOf("why", "tactic", "focus", "candidates", "threats")
private val BY_LEVEL: Map<DisclosureLevel, List<String>> = mapOf(
    DisclosureLevel.SILENT to emptyList(),
    DisclosureLevel.IDEA to IDEA_KEYS,
    DisclosureLevel.FOCUS to FOCUS_KEYS,
    DisclosureLevel.MOVE to MOVE_KEYS,
    DisclosureLevel.LINE to MOVE_KEYS + listOf("line"),
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
                // A player stuck for a while gets the idea at once, the focus next.
                if (level.value < AUTONOMOUS_CEILING.value) {
                    level = DisclosureLevel.of(level.value + 1)
                }
            Unlock.DECISIVE ->
                // The critical moment's first word is the idea.
                if (level.value < DisclosureLevel.IDEA.value) {
                    level = DisclosureLevel.IDEA
                }
            Unlock.HINT ->
                // Two hints, then the move.
                if (level.value < HINT_CEILING.value) {
                    level = DisclosureLevel.of(level.value + 1)
                }
            Unlock.ASK_MOVE, Unlock.USER_NAMED_MOVE ->
                level = DisclosureLevel.of(maxOf(level.value, DisclosureLevel.MOVE.value))
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
        "|(?:their|opponent's|the opponent's|enemy) ${PIECE_RE}s?" +
        "|(?:wins?|winning|won|trap|traps|trapped|lose|loses|losing) (?:a |an |the |their |two )?" +
        "${PIECE_RE}s?" +
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
 * Below MOVE the coach may not name any move at all, and may mention only squares that occur
 * in visible facts other than the FEN, such as the target's. The model will volunteer moves
 * on its own even when told not to, so this is not limited to the withheld facts. From MOVE
 * up only the withheld line is checked. Below FOCUS the word for the player's own piece is
 * checked as well -- their pieces, and what a line wins, are not it.
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
    // The moves the visible facts name -- their threat, the trap, the moves that are all as good
    // -- are said in the open; never the one the player is to find.
    val allowedMoves = HashSet<String>()
    for ((key, value) in visible) {
        if (key == "fen") continue
        if (value == null) continue
        if (value is Boolean && !value) continue
        if (value is String && value.isEmpty()) continue
        if (value is List<*> && value.isEmpty()) continue
        val parts = if (value is List<*>) value else listOf(value)
        for (token in TOKEN_SPLIT.split(parts.joinToString(" ") { it.toString() })) {
            if (SQUARE_TOKEN.matches(token)) allowedMoves.add(TRAILING_MARKS.replace(token, ""))
        }
    }

    val leaks = ArrayList<String>()
    fun add(item: String) {
        if (item !in leaks) leaks.add(item)
    }

    if (level.value < DisclosureLevel.MOVE.value) {
        for (token in TOKEN_SPLIT.split(text)) {
            val match = SQUARE_TOKEN.matchEntire(token) ?: continue
            val square = match.groupValues[1]
            val core = TRAILING_MARKS.replace(token, "")
            if (core != square) {
                if (core.lowercase() !in saidWords && core !in allowedMoves) add("move $core")
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

    if (level.value < DisclosureLevel.FOCUS.value && !facts.piece.isNullOrEmpty()) {
        // The words for the player's own pieces ("your queen on h5", "your rook first, then
        // your knight") are a hint only where they could point at the piece to move: not in an
        // opening's name ("Queen's Gambit"), not in the material balance ("down a pawn"), not
        // in what a line wins or a fact the level shows ("you can win a queen and a pawn for
        // a bishop"), not said of a move already played -- the caller's `said` carries those.
        var scrubbed = text.lowercase()
        // The whole first hint first: once a part of it is taken out, the whole no longer
        // matches, and its words were read as the coach's own.
        for (key in listOf(
            "situation",
            "goal",
            "theme",
            "mechanism",
            "question",
            "plan",
            "nature",
            "stakes",
            "key",
            "trap",
        )) {
            val value = visible[key]
            if (value is String && value.isNotEmpty()) scrubbed = scrubbed.replace(value.lowercase(), " ")
        }
        scrubbed = PIECE_WORD_CONTEXTS.replace(scrubbed, " ")
        for (pieceWord in LinkedHashSet(PIECE_WORD.findAll(facts.piece.lowercase()).map { it.value }.toList())) {
            // "The king behind it", "their king runs out of squares": the word is in every
            // explanation of a pin or a mate, and says nothing about which piece to move.
            if (pieceWord in saidWords || pieceWord == "king") continue
            if (Regex("\\b${Regex.escape(pieceWord)}\\b").containsMatchIn(scrubbed)) {
                add("piece $pieceWord")
            }
        }
    }
    return leaks
}

/** `re.findall(_PIECE, ...)`: every piece word, wherever it stands in the text. */
private val PIECE_WORD = Regex(PIECE_RE)
