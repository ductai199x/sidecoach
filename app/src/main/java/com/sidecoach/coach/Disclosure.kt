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
    HINT("hint"),
    ASK_MOVE("ask_move"),
    ASK_LINE("ask_line"),
    USER_NAMED_MOVE("user_named_move"),
    MOVE_PLAYED("move_played"),

    /**
     * What the AI coach's reply to a question actually said ([saidLevel]): the ladder follows
     * the words, not a guess at what the question meant. `Unlock.REPLY`.
     */
    REPLY("reply"),
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
    /**
     * The moves about as good as the best (graded Best or Excellent), and those graded Good, a
     * little behind them ([gradedMoves]): a good move the player names is said to be good, never
     * refused as bad.
     */
    val candidates: List<String> = emptyList(),
    val good: List<String> = emptyList(),
    val threats: List<String> = emptyList(),
    val line: List<String> = emptyList(),
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
    /**
     * The principle's sentence as [situation] says it ("A move that attacks something they must
     * then defend keeps you a step ahead: ..."), so the coach can leave it out while it cools
     * down ([RULE_COOLDOWN]): it came 218 times in the master games' hints, 59 of them again in
     * the same game. Not on the model's sheet: [situation] carries it.
     */
    val principle: String? = null,
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
        out["good"] = good
        out["threats"] = threats
        out["line"] = line
        out["stakes"] = stakes
        out["key"] = key
        out["trap"] = trap
        out["choices"] = choices
        out["situation"] = situation
        out["principle"] = principle
        return out
    }
}

private val ALWAYS = listOf(
    "fen", "side_to_move", "user_side", "phase", "material", "last_move", "decisive",
    "only_move",
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
    // Their threat is the first hint's key fact as often as not ("they threaten Nxe5, winning
    // your pawn"): withheld to the move level, "what is their threat?" got "nothing" (the runtime
    // audit, 2026-09-25).
    "threats",
)
private val FOCUS_KEYS = IDEA_KEYS + listOf("target", "piece", "nudge", "region", "ideas")
private val MOVE_KEYS = FOCUS_KEYS + listOf("why", "tactic", "focus", "candidates")
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
            Unlock.HINT ->
                // Two hints, then the move.
                if (level.value < HINT_CEILING.value) {
                    level = DisclosureLevel.of(level.value + 1)
                }
            Unlock.ASK_MOVE, Unlock.USER_NAMED_MOVE ->
                level = DisclosureLevel.of(maxOf(level.value, DisclosureLevel.MOVE.value))
            Unlock.ASK_LINE, Unlock.MOVE_PLAYED ->
                level = DisclosureLevel.LINE
            Unlock.REPLY -> Unit // carries no level: [said] does
        }
        if (level == before) return null
        history.add(unlock to level)
        return level
    }

    /**
     * A reply to the player's question said up to [said] ([saidLevel]): the ladder climbs to it
     * and never falls. Returns the new level, or null when nothing climbed. `Disclosure.said`.
     */
    fun said(said: DisclosureLevel): DisclosureLevel? {
        if (said.value <= level.value) return null
        level = said
        history.add(Unlock.REPLY to said)
        return said
    }

    /**
     * The highest level said here before the player moved: every unlock but the move played,
     * which opens a past position to its line. `Disclosure.hinted`.
     */
    val hinted: DisclosureLevel
        get() = history.filter { it.first != Unlock.MOVE_PLAYED }.maxOfOrNull { it.second.value }
            ?.let { DisclosureLevel.of(it) } ?: DisclosureLevel.SILENT

    /**
     * Start from the hints already given for the combination this position continues
     * ([CoachSession.ladder]): each level is said once, and none below the highest (Tai,
     * 2026-09-29: the stuck hint's idea at a puzzle's start came again as the question after
     * the move was found and the reply left the engine's line). `Disclosure.carry`.
     */
    fun carry(carried: DisclosureLevel) {
        if (carried.value > this.carried.value) this.carried = carried
        if (carried.value > level.value) level = carried
    }

    /** What the combination had said before the player came here ([carry]). */
    var carried: DisclosureLevel = DisclosureLevel.SILENT
        private set

    /**
     * The move played from here is taken back: the position is the player's to move again, at the
     * level of the hints it had had. The line the move opened ([Unlock.MOVE_PLAYED]) belonged to
     * the move withdrawn; what was said before it stays said, and nothing lower is said after it
     * (Tai, 2026-09-29). `Disclosure.withdraw_move`.
     */
    fun withdrawMove() {
        history.removeAll { it.first == Unlock.MOVE_PLAYED }
        level = DisclosureLevel.of(maxOf(hinted.value, carried.value))
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
        "|(?:check(?:s|ing|ed)?|attack(?:s|ing|ed)?|takes|taking|took|captur(?:es?|ing|ed)" +
        "|remov(?:es?|ing|ed)|hit(?:s|ting)?|threaten(?:s|ing|ed)?|win(?:s|ning)?|won" +
        "|pin(?:s|ning|ned)?|fork(?:s|ing|ed)?|skewer(?:s|ing|ed)?|go(?:es|ing)? after) " +
        "(?:your|their|the|my) $PIECE_RE on [a-h][1-8]" +
        // What their threat goes after, said without its square: "g5 attacks your queen" is the
        // threat the sheet gives, not the move to find -- never "take your queen" or "attack with
        // your queen" (the smoke run, 2026-09-25).
        "|(?:attack(?:s|ing|ed)?|threaten(?:s|ing|ed)?|hit(?:s|ting)?|pin(?:s|ning|ned)?" +
        "|fork(?:s|ing|ed)?|skewer(?:s|ing|ed)?|go(?:es|ing)? after) (?:your|the|my) " +
        "${PIECE_RE}s?\\b(?! to\\b)" +
        "|(?:your|their|my|the) $PIECE_RE on [a-h][1-8] (?:is|was|remains|stays) (?:in )?" +
        "(?:check|attacked|under attack|hanging|loose|defended|undefended|pinned)",
)

/** A move with its number in front: the game's record when the game played it. */
private val NUMBERED_MOVE = Regex(
    "\\b\\d+\\.(?:\\.\\.)?\\s*(?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?"
)

private val SPACES = Regex("\\s+")

/** A numbered move in one spelling: "17. Rh1+" and "17.Rh1" are "17.Rh1". `disclosure.numbered_key`. */
fun numberedKey(text: String): String = SPACES.replace(text, "").trimEnd('+', '#')

/*
 * A token is a bare square ("f7") or a SAN move ending in its destination square ("Ng5",
 * "Nxf7+", "exd5", "R1e1", "e8=Q"): the desktop's `_SQUARE_TOKEN`,
 * `[KQRBN]?[a-h]?[1-8]?x?([a-h][1-8])(?:=[QRBN])?[+#]?[!?]*`, group 1 the destination square,
 * over the words `_TOKEN_SPLIT` (`[^A-Za-z0-9+#=!?]+`) leaves. Both are read by hand below
 * ([tokensOf], [destinationOf]).
 */
private val WORD_RE = Regex("[a-z0-9=+#-]+")

/** Destination squares of every bare square or SAN token in `text`. */
fun squaresIn(text: String): Set<String> {
    val found = LinkedHashSet<String>()
    for (token in tokensOf(text)) {
        val square = destinationOf(token) ?: continue
        found.add(square)
    }
    return found
}

/**
 * The words `_TOKEN_SPLIT` leaves of [text] that are not empty: each run of letters, digits and
 * `+#=!?`. Read by hand, as [destinationOf] and [trailingMarksOff] are: on Android every match of
 * a [Regex] opens a native matcher and registers it for cleaning, and these three ran once a
 * token, tens of tokens a reply -- the parity suite's 67,000 leak checks spent most of three
 * minutes there. An empty token never names a square, so leaving them out changes nothing.
 */
private fun tokensOf(text: String): List<String> {
    val out = ArrayList<String>()
    var start = -1
    for (i in 0..text.length) {
        val inToken = i < text.length && text[i].let {
            it in 'A'..'Z' || it in 'a'..'z' || it in '0'..'9' || it == '+' || it == '#' || it == '=' || it == '!' || it == '?'
        }
        if (inToken) {
            if (start < 0) start = i
        } else if (start >= 0) {
            out.add(text.substring(start, i))
            start = -1
        }
    }
    return out
}

/**
 * `_SQUARE_TOKEN.fullmatch(token).group(1)`: the destination square of a bare square or
 * SAN token, or null. Read from the end, which is the regex's one way to match: what may follow
 * the square (`=Q`, `+`, `!?`) never holds a file or a rank, and what may precede it
 * (`[KQRBN]?[a-h]?[1-8]?x?`) is four distinct classes in order.
 */
private fun destinationOf(token: String): String? {
    var end = token.length
    while (end > 0 && (token[end - 1] == '!' || token[end - 1] == '?')) end--
    if (end > 0 && (token[end - 1] == '+' || token[end - 1] == '#')) end--
    if (end >= 2 && token[end - 2] == '=' && token[end - 1] in "QRBN") end -= 2
    if (end < 2 || token[end - 2] !in 'a'..'h' || token[end - 1] !in '1'..'8') return null
    val stop = end - 2
    var i = 0
    if (i < stop && token[i] in "KQRBN") i++
    if (i < stop && token[i] in 'a'..'h') i++
    if (i < stop && token[i] in '1'..'8') i++
    if (i < stop && token[i] == 'x') i++
    return if (i == stop) token.substring(stop, end) else null
}

/**
 * `TRAILING_MARKS.replace(text, "")` (`[+#!?]+$`) read by hand: the check, mate and comment marks
 * at the end, or before a line break that ends the text, where the regex's `$` stops as well
 * (ICU's line breaks, which Android's regex is).
 */
private fun trailingMarksOff(text: String): String {
    val lineBreak = when {
        text.endsWith("\r\n") -> 2
        text.isNotEmpty() && text.last() in LINE_BREAKS -> 1
        else -> 0
    }
    val body = text.length - lineBreak
    var end = body
    while (end > 0 && text[end - 1].let { it == '+' || it == '#' || it == '!' || it == '?' }) end--
    return if (end == body) text else text.substring(0, end) + text.substring(body)
}

private const val LINE_BREAKS = "\n\u000b\u000c\r\u0085\u2028\u2029"

/**
 * Whether [text] holds a digit followed by a full stop, which every [NUMBERED_MOVE] begins with
 * (`\d` is any decimal digit, as ICU reads it): without one the record has nothing to take out,
 * and the regex is not run -- a coach's reply rarely numbers its moves.
 */
private fun mayNumberMoves(text: String): Boolean {
    var i = 0
    while (i < text.length) {
        val point = text.codePointAt(i)
        val next = i + Character.charCount(point)
        if (Character.isDigit(point) && next < text.length && text[next] == '.') return true
        i = next
    }
    return false
}

/**
 * The patterns made of a piece word, a square or a move, compiled once (each compile is a native
 * regex on Android): there are a few hundred of them, and a reply's check compiled several.
 */
private val compiled = object : LinkedHashMap<String, Regex>(256, 0.75f, true) {
    override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, Regex>?): Boolean = size > 512
}

private fun regexOf(pattern: String): Regex = synchronized(compiled) { compiled.getOrPut(pattern) { Regex(pattern) } }

/**
 * Withheld items that `text` mentions. An empty list means the draft is clean.
 *
 * Below MOVE only the answer is withheld (see below); from MOVE up only the withheld line.
 * [record] is the game's moves as numbered in the record ("17. Rh1"). Below FOCUS the word for
 * the player's own piece is checked as well -- their pieces, and what a line wins, are not it.
 *
 * `said` is what the user asked this turn: a square, piece word or move they named is theirs
 * already, so confirming it back is not a leak and never faulted. [shown] is what the sheet
 * shows beyond the facts' own words -- what the threats do -- whose pieces on their squares are
 * facts about the board, like the facts'.
 *
 * Below MOVE only the answer is withheld (Tai, 2026-09-25: "leaking is not a criminal offense
 * ... it's useful to point out in precise manner the plan (yours or opponent's) or the positional
 * advantage or disadvantage ... clamp down egregious offense but be smart about it"): the moves
 * the player is to find -- the good moves, the player's moves in the best line -- and the squares
 * they go to, unless the visible facts name them. Any other move or square -- a plan, a weakness,
 * their threat, a line the facts give -- may be said. The moves of the game are no exemption: the
 * answer is withheld even when it was played before (by mid-game that exemption covered most of
 * the board: the runtime audit).
 */
fun leakCheck(
    text: String,
    facts: PositionFacts,
    level: DisclosureLevel,
    said: String? = null,
    /** The game's moves as numbered in the record ("17. Rh1"). */
    record: Collection<String> = emptyList(),
    /**
     * What the sheet shows beyond the facts' own words -- what the threats do -- whose pieces on
     * their squares are facts about the board, like the facts'.
     */
    shown: String = "",
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
    // And the squares the sheet shows beyond the facts' words: a threat "attacks the knight on e7"
    // may be said though e7 is where the answer goes (the benchmark's draw, 2026-09-25).
    allowedSquares.addAll(squaresIn(shown))
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
        for (token in tokensOf(parts.joinToString(" ") { it.toString() })) {
            if (destinationOf(token) != null) allowedMoves.add(trailingMarksOff(token))
        }
    }

    val leaks = ArrayList<String>()
    fun add(item: String) {
        if (item !in leaks) leaks.add(item)
    }

    // A move of the game said with its number ("17. Rh1", "4... Nf6") is the record, not a
    // suggestion: the review names the moves where ground was lost, and one may be the answer
    // again. Only a numbered move the game actually played -- "5. cxd5 wins it" of the move to
    // find was let through by any number at all (the second audit).
    val played = record.map { numberedKey(it) }.toSet()
    val movesText = if (!mayNumberMoves(text)) {
        text
    } else {
        NUMBERED_MOVE.replace(text) { m -> if (numberedKey(m.value) in played) " " else m.value }
    }
    if (level.value < DisclosureLevel.MOVE.value) {
        val answer = (facts.candidates + facts.line.filterIndexed { i, _ -> i % 2 == 0 })
            .filter { it.isNotEmpty() }
            .map { trailingMarksOff(it) }
            .toSet()
        val answerSquares = squaresIn(answer.joinToString(" ")) - allowedSquares
        for (token in tokensOf(movesText)) {
            val square = destinationOf(token) ?: continue
            val core = trailingMarksOff(token)
            if (core != square) {
                val known = core.lowercase() in saidWords || core in allowedMoves
                if (core in answer && !known) add("move $core")
            } else if (square in answerSquares && square !in saidSquares) {
                add("square $square")
            }
        }
    } else if (level.value < DisclosureLevel.LINE.value) {
        val visibleMoves = (facts.candidates + facts.threats).map { trailingMarksOff(it) }.toSet()
        // The move's why names its continuation ("after Qf6, Bg2 ..."): what the level shows may
        // be said -- compared without check marks, as `allowedMoves` holds them ("Rc6+" was
        // withheld while the sheet gave it: the pre-spend audit, 2026-09-25).
        val withheldMoves = facts.line.filter {
            trailingMarksOff(it) !in visibleMoves && trailingMarksOff(it) !in allowedMoves
        }
        val withheldSquares = squaresIn(withheldMoves.joinToString(" ")) - allowedSquares
        for (square in (squaresIn(movesText) intersect withheldSquares).sorted()) add("square $square")
        for (move in withheldMoves) {
            if (regexOf("(?<![A-Za-z])${Regex.escape(move)}(?![A-Za-z0-9])").containsMatchIn(movesText)) {
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
            "material",
        )) {
            val value = visible[key]
            if (value is String && value.isNotEmpty()) scrubbed = scrubbed.replace(value.lowercase(), " ")
        }
        // A piece the visible facts name on its square, said in other words ("their pawn on b5 is
        // about to take your bishop on a4" for "winning your bishop on a4"), is the fact, not a
        // hint at the piece to move.
        val onSheet = (visible.values.filterIsInstance<String>() + shown).joinToString(" ").lowercase()
        for (found in PIECE_ON_SQUARE.findAll(onSheet).map { it.groupValues[1] to it.groupValues[2] }.toSet()) {
            val (pieceWord, square) = found
            scrubbed = regexOf("\\b${pieceWord}s? on $square\\b").replace(scrubbed, " ")
        }
        scrubbed = PIECE_WORD_CONTEXTS.replace(scrubbed, " ")
        // The piece to move is named when it is said on its own square, or as "your <piece>" with
        // no square; another piece of that kind on another square ("attacks the pawn on d2" when
        // the pawn to move is on e4) is a fact about the board, not the hint. Without a square
        // on the sheet, any mention counts, as before.
        val pieceSquares = squaresIn(facts.piece)
        // The file of a pawn to move: "the c-pawn" names the pawn on c4.
        val pieceFiles = pieceSquares.map { it[0] }.toSet()
        // The piece that does it, said as the one that takes or goes ("take it with the queen",
        // "capture with a pawn", "the c-pawn should take it": the second audit) -- not every
        // mention of its kind: "the pawn on c3", "your pawn on h2" are the board (round 2 refused
        // four good replies for them).
        for (pieceWord in LinkedHashSet(PIECE_WORD.findAll(facts.piece.lowercase()).map { it.value }.toList())) {
            // "The king behind it", "their king runs out of squares": the word is in every
            // explanation of a pin or a mate, and says nothing about which piece to move.
            if (pieceWord in saidWords || pieceWord == "king") continue
            // The sheet itself names the player's piece of this kind at this level ("the first
            // move takes your bishop out of danger"): said in other words, it is no leak -- a
            // paraphrase of the first hint was refused for it, and the bare template spoke (the
            // benchmark's smoke run, 2026-09-25; Tai: "a vague coach is worse than no coach at
            // all").
            if (regexOf("\\byour ${pieceWord}s?\\b").containsMatchIn(onSheet)) continue
            val mention = regexOf(
                "\\b(your |my |with (?:the |a |your )?|(?:([a-h])-))?${Regex.escape(pieceWord)}s?\\b" +
                    "(?: (?:on|from) ([a-h][1-8]))?"
            )
            for (found in mention.findAll(scrubbed)) {
                val prefix = found.groups[1]?.value
                val file = found.groups[2]?.value
                val square = found.groups[3]?.value
                val named = if (square != null) {
                    square in pieceSquares
                } else if (file != null) {
                    file[0] in pieceFiles || pieceFiles.isEmpty()
                } else {
                    prefix != null || pieceSquares.isEmpty()
                }
                if (named) {
                    add("piece $pieceWord")
                    break
                }
            }
        }
    }
    return leaks
}

/** `re.findall(_PIECE, ...)`: every piece word, wherever it stands in the text. */
private val PIECE_WORD = Regex(PIECE_RE)

/** `re.findall(rf"({_PIECE})s? on ([a-h][1-8])", ...)`: a piece word and its square. */
private val PIECE_ON_SQUARE = Regex("($PIECE_RE)s? on ([a-h][1-8])")

/** A piece named by whose it is: "their king", "your queen", "the opponent's rook". `_OWNED_PIECE`. */
private val OWNED_PIECE = Regex("(?i)\\b(your|their|the opponent's|opponent's)\\s+($PIECE_RE)s?\\b")
private val MATE_NAMED = Regex("(?i)\\b(?:check)?mat(?:e|es|ed|ing)\\b")
private val PROMOTION_NAMED = Regex("(?i)\\bpromot(?:e|es|ed|ing|ion)\\b")
private val CASTLE_NAMED = Regex("(?<![-A-Za-z0-9])O-O(?:-O)?(?![-A-Za-z0-9])")

/**
 * What [text] names besides squares and moves: the pieces it names by whose they are ("their
 * king"), a mate, a promotion. `disclosure._things_named`.
 */
private fun thingsNamed(text: String): Set<String> {
    val found = LinkedHashSet<String>()
    for (m in OWNED_PIECE.findAll(text)) {
        val owner = if (m.groupValues[1].lowercase() == "your") "your" else "their"
        found.add("$owner ${m.groupValues[2].lowercase()}")
    }
    if (MATE_NAMED.containsMatchIn(text)) found.add("mate")
    if (PROMOTION_NAMED.containsMatchIn(text)) found.add("promotion")
    return found
}

/**
 * The moves [text] names in SAN (a piece move, a capture, a promotion, castling), not the bare
 * squares. `disclosure._moves_in`.
 */
private fun movesIn(text: String): List<String> {
    val out = CASTLE_NAMED.findAll(text).map { it.value }.toMutableList()
    for (token in tokensOf(text)) {
        val square = destinationOf(token) ?: continue
        if (trailingMarksOff(token) != square) out.add(token)
    }
    return out
}

/**
 * Whether [text] names the move [move]: its letters as a whole move, not inside another ("e4" is
 * not named by "Nxe4" or "Qe4", "O-O" not by "O-O-O"), its check, mate and comment marks
 * aside. `disclosure._names_move`. The coach's [names] reads a bare square as a square, and a
 * pawn move like "e4" was rated a MOVE whenever a reply named a piece going to e4.
 */
private fun namesMove(text: String, move: String): Boolean {
    val core = trailingMarksOff(move)
    if (core.isEmpty()) return false
    val pattern = if (core.startsWith("O-O")) {
        "(?<![-A-Za-z0-9])" + Regex.escape(core) + "(?!-O)(?![A-Za-z0-9])"
    } else {
        "(?<![A-Za-z])" + Regex.escape(core) + "[+#]?(?![A-Za-z0-9])"
    }
    return Regex(pattern).containsMatchIn(text)
}

/**
 * The rung a reply of the AI coach carried, from what it names: the first two moves of the
 * engine's line (LINE); a move the sheet lists as a candidate (MOVE); a square of the target, the
 * piece or the squares that carry the theme, or the piece that does it (FOCUS); a square, a move,
 * a piece, a mate or a promotion that the idea's facts name -- the goal, the theme, the key fact,
 * the stakes, the trap, the threats, the target's piece (IDEA); else SILENT -- nothing of the
 * ladder was said. A reply that only answers "am I winning?" names none of it, and the ladder does
 * not move (Tai, 2026-10-02: no word-overlap guess, no floor).
 *
 * The ladder of hints (each rung said once, none below one already said) follows this, not a
 * guess from the question's words: with the AI on, the model reads the question and decides what
 * it answers (Tai, 2026-10-02). `disclosure.said_level` (its `_names_move` is [namesMove]).
 */
fun saidLevel(text: String, facts: PositionFacts): DisclosureLevel {
    val movesText = if (mayNumberMoves(text)) NUMBERED_MOVE.replace(text, " ") else text
    val line = facts.line.filter { it.isNotEmpty() }
    if (line.size >= 2 && namesMove(movesText, line[0]) && namesMove(movesText, line[1])) {
        return DisclosureLevel.LINE
    }
    val answers = facts.candidates + line.take(1)
    if (answers.any { namesMove(movesText, it) }) return DisclosureLevel.MOVE
    val squares = squaresIn(text)
    val named = thingsNamed(text)
    val focus = listOfNotNull(facts.target, facts.piece, *facts.focus.toTypedArray())
        .filter { it.isNotEmpty() }.joinToString(" ")
    val focusSquares = squaresIn(focus)
    val pieceNamed = !facts.piece.isNullOrEmpty() && named.any { it in thingsNamed(facts.piece) }
    if (squares.any { it in focusSquares } || pieceNamed) return DisclosureLevel.FOCUS
    val idea = listOfNotNull(
        facts.goal,
        facts.theme,
        facts.mechanism,
        facts.key,
        facts.stakes,
        facts.situation,
        facts.trap,
        facts.target,
    ).filter { it.isNotEmpty() }.joinToString(" ")
    val ideaSquares = squaresIn(idea)
    val ideaNamed = thingsNamed(idea)
    if (
        squares.any { it in ideaSquares } ||
        named.any { it in ideaNamed } ||
        (facts.threats + movesIn(idea)).any { namesMove(movesText, it) }
    ) {
        return DisclosureLevel.IDEA
    }
    return DisclosureLevel.SILENT
}
