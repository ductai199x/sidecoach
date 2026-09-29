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
        if (carried.value > level.value) level = carried
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
        for (token in TOKEN_SPLIT.split(parts.joinToString(" ") { it.toString() })) {
            if (SQUARE_TOKEN.matches(token)) allowedMoves.add(TRAILING_MARKS.replace(token, ""))
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
    val movesText = NUMBERED_MOVE.replace(text) { m -> if (numberedKey(m.value) in played) " " else m.value }
    if (level.value < DisclosureLevel.MOVE.value) {
        val answer = (facts.candidates + facts.line.filterIndexed { i, _ -> i % 2 == 0 })
            .filter { it.isNotEmpty() }
            .map { TRAILING_MARKS.replace(it, "") }
            .toSet()
        val answerSquares = squaresIn(answer.joinToString(" ")) - allowedSquares
        for (token in TOKEN_SPLIT.split(movesText)) {
            val match = SQUARE_TOKEN.matchEntire(token) ?: continue
            val square = match.groupValues[1]
            val core = TRAILING_MARKS.replace(token, "")
            if (core != square) {
                val known = core.lowercase() in saidWords || core in allowedMoves
                if (core in answer && !known) add("move $core")
            } else if (square in answerSquares && square !in saidSquares) {
                add("square $square")
            }
        }
    } else if (level.value < DisclosureLevel.LINE.value) {
        val visibleMoves = (facts.candidates + facts.threats).map { TRAILING_MARKS.replace(it, "") }.toSet()
        // The move's why names its continuation ("after Qf6, Bg2 ..."): what the level shows may
        // be said -- compared without check marks, as `allowedMoves` holds them ("Rc6+" was
        // withheld while the sheet gave it: the pre-spend audit, 2026-09-25).
        val withheldMoves = facts.line.filter {
            TRAILING_MARKS.replace(it, "") !in visibleMoves && TRAILING_MARKS.replace(it, "") !in allowedMoves
        }
        val withheldSquares = squaresIn(withheldMoves.joinToString(" ")) - allowedSquares
        for (square in (squaresIn(movesText) intersect withheldSquares).sorted()) add("square $square")
        for (move in withheldMoves) {
            if (Regex("(?<![A-Za-z])${Regex.escape(move)}(?![A-Za-z0-9])").containsMatchIn(movesText)) {
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
            scrubbed = Regex("\\b${pieceWord}s? on $square\\b").replace(scrubbed, " ")
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
            if (Regex("\\byour ${pieceWord}s?\\b").containsMatchIn(onSheet)) continue
            val mention = Regex(
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
