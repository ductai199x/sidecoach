package com.sidecoach.coach

import com.sidecoach.analysis.positionScore
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.squareName
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.ReviewJson
import com.sidecoach.events.ReviewedGame
import com.sidecoach.llm.CHARS_PER_TOKEN
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.Completion
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.llm.OpenAICompatProvider
import com.sidecoach.llm.Prices
import com.sidecoach.llm.Usage
import com.sidecoach.llm.modelOf
import com.sidecoach.voice.Py
import java.util.concurrent.ExecutorCompletionService
import java.util.concurrent.Executors

/**
 * The model's review of a game's key moves (docs/game-review-plan.md, decision 3):
 * `coach/review_words.py`.
 *
 * Asked for only when the player presses Game Review's button, about a cent a game, never ahead
 * of time and never for the games the background review goes through. One call a key move, all
 * at once: told every move together, the model lent one move's facts to another and one answer it
 * could not read lost the whole game (luna, 2026-09-30). In: the move with the coach's own fact
 * bundle for the position it was played from (what its answers are grounded in) and the review's
 * measures -- who played it, what it cost, the reply that punishes it, the better move and its
 * line. Out: a paragraph on why it mattered and how the position works, not what happened, which
 * the page already shows; for a bad move, a second on the better move; and a lesson in a
 * sentence or two.
 *
 * The model names the sides only by colour, and the student's words are made from its
 * paragraphs ([toStudent]: White and Black become "you" and "your opponent"). Asked to say "you"
 * itself, it spoke to whoever played the move -- about one shown paragraph in four, the
 * student's own move given to their opponent among them (luna at medium effort, 2026-09-30) --
 * and colours it does not turn round. The words are shown as the model wrote them: only a false
 * claim about the board refuses a paragraph, as in the live coach -- a piece the board puts
 * elsewhere or with the other side ([checkDraft]), a move given to the side that does not play
 * it. No rule over the words themselves drops or holds one back (Tai, 2026-10-02): a "why" that
 * says "you", a lesson that names a square, a "why" that names the better move are shown as
 * written.
 *
 * The regexes are Python's, read as `re` reads them: `pattern.search(text, pos, endpos)` is a
 * search from `pos` in the text cut at `endpos` (lookbehind sees before `pos`, nothing is seen
 * past `endpos`), and MULTILINE's `^` and `$` are spelled out as "after a newline or at the
 * start" and "before a newline or at the end", where Java's also stop at \r and \u2028.
 */
object ReviewWords {
    /** The key moves, dearest first when there are more than [MOST]: the review's key kinds. */
    val PRIORITY = listOf("blunder", "miss", "brilliant", "great", "mistake", "inaccuracy")
    val BAD = setOf("inaccuracy", "mistake", "miss", "blunder")
    const val MOST = 16

    /**
     * A move's answer's own tokens (a provider that thinks adds its headroom). Room for full
     * paragraphs while the words are made right (Tai, 2026-09-30: "let's first get it right
     * before get it concise"); to come down to about a cent a game at gpt-5.6-luna's prices after.
     */
    const val ANSWER_TOKENS = 1500

    /**
     * A move's answer as it comes, reasoning included: gpt-5.6-luna at medium effort wrote 764
     * tokens a move, 482 of them reasoning, over 55 key moves of Tai's games (2026-09-30).
     */
    const val TYPICAL_ANSWER = 770

    /**
     * The coach's facts a review needs; the rest serve a hint in play (the question to ask, the
     * nudge), repeat others (the choices, the situation) or the board itself (the FEN).
     */
    val DROP = setOf("fen", "user_side", "question", "nudge", "choices", "situation", "focus", "last_move")
    const val LINE_PLIES = 4

    val SYSTEM = """
        You are a chess coach going over a finished game with your student, one key move at a
        time.

        For the key move below, explain WHY it mattered and HOW the position works, as a strong coach
        would at the board: the plans of both sides, the pieces and squares that count, what the move
        changed, and what each side can do next because of it. The student can already see the move,
        its label and the evaluation, so never just say what happened: explain it. Take the room it
        needs.

        Ground every concrete statement in the facts given. Name only the pieces and squares the facts
        name, and only moves that appear in them. Never invent a variation. Speak as the coach who sees
        the board: never mention the facts, "the line given" or "the stated" anything.

        Name the two sides only as White and Black: "White's bishop on c1", "Black can take on e5",
        "White's winning chances fall". Never write "you", "your", "the opponent", "he", "they" or "its"
        for a side: the page turns White and Black into "you" and "your opponent" for the student, which
        it can do only when every side is named by its colour.

        Write these parts:
        "why": a full paragraph. For an inaccuracy, mistake, miss or blunder: what went wrong and how the
        other side can punish it (the best reply given may be named). When the move is the student's own,
        or the student's side is unknown, never give the better move, in notation or by naming its piece
        and its square: the student will try to find it first. For a brilliant or great move: what makes
        it work, and why it is hard to see.
        "better", for an inaccuracy, mistake, miss or blunder only: a paragraph that names the better
        move given outright and walks through its line: the idea behind it and how the line works
        (never the best reply, which is the other side's). The page shows "better" only once the
        student has found the move or asked for it.
        "lesson": a sentence or two of advice to take into the next games. Here alone speak to the
        student as "you", of "your opponent" in general; no colours, squares or moves.

        No headings inside a part, no lists, no move numbers, and do not open with the move or its label.

        Answer in this form and nothing else:
        ### why
        (the paragraph)
        ### better
        (the paragraph)
        ### lesson
        (the advice)
    """.trimIndent()

    /**
     * The student's words by ply ("why", "lesson", and for a bad move "better"); the plies with a
     * part refused for a false claim about the board, and why; the model's answers by ply (in
     * colours, for reading by hand); what the calls cost, and each call's measures.
     * `review_words.ReviewWords`.
     */
    class Words(
        val texts: LinkedHashMap<Int, Map<String, String>> = LinkedHashMap(),
        val dropped: LinkedHashMap<Int, String> = LinkedHashMap(),
        val answers: LinkedHashMap<Int, String> = LinkedHashMap(),
        val model: String = "",
        var cost: Double? = null,
        val calls: MutableList<Map<String, Any?>> = ArrayList(),
        // The plies whose call failed: not done, to ask again (Try again), never dropped for good.
        val failed: LinkedHashSet<Int> = LinkedHashSet(),
    )

    /**
     * The calls asked at once: every key move at once set a provider's rate limit off, and the
     * moves whose calls failed were left without words (the review of 2026-10-01). `PARALLEL`.
     */
    @Volatile var PARALLEL = 4

    /**
     * The plies of the game's key moves, in order: [MOST] of them at most, the dearest kinds
     * first and then the costliest. `key_plies`.
     */
    fun keyPlies(game: ReviewedGame): List<Int> {
        // (kind, -loss, ply), as Python's tuples sort: the loss by value, so -0.0 and 0.0 tie.
        val keyed = game.moves.mapNotNull { m ->
            val label = m.assessment.label?.let { ReviewJson.labelName(it) }
            if (label != null && label in PRIORITY) Triple(PRIORITY.indexOf(label), -m.assessment.expectedLoss, m.ply) else null
        }
        val order = Comparator<Triple<Int, Double, Int>> { a, b ->
            when {
                a.first != b.first -> a.first.compareTo(b.first)
                a.second < b.second -> -1
                a.second > b.second -> 1
                else -> a.third.compareTo(b.third)
            }
        }
        return keyed.sortedWith(order).take(MOST).map { it.third }.sorted()
    }

    /**
     * The coach's facts speak to the side to move ("your bishop", "if you let them"); a review
     * speaks of both sides' moves, so the facts go to the model in colours. "Among them" is of
     * moves ("several moves keep the balance, Bf5 and Bd6 among them").
     */
    private val PRONOUNS = Regex(
        "(?<!among )(?<!of )(?<!between )\\b(yours|your|you|their|theirs|them|they)\\b",
        RegexOption.IGNORE_CASE,
    )

    /**
     * [value] (a fact, or a list or map of them) with the side to move's pronouns turned into
     * its colour and the other side's into theirs: "your bishop" -> "White's bishop". `in_colours`.
     */
    fun inColours(value: Any?, mover: String): Any? {
        val other = if (mover == "White") "Black" else "White"
        val names = mapOf(
            "your" to "$mover's", "yours" to "$mover's", "you" to mover,
            "their" to "$other's", "theirs" to "$other's", "them" to other, "they" to other,
        )
        return when (value) {
            is String -> PRONOUNS.replace(value) { names.getValue(it.groupValues[1].lowercase()) }
            is List<*> -> value.map { inColours(it, mover) }
            is Map<*, *> -> LinkedHashMap<Any?, Any?>().apply { for ((k, v) in value) put(k, inColours(v, mover)) }
            else -> value
        }
    }

    /** What a move does, in colours: "Black's knight from a5 to c4, taking White's pawn". `_moved`. */
    private fun moved(board: Board, uci: String): String {
        val move = Move.fromUci(uci)
        val mover = if (board.turn == WHITE) "White" else "Black"
        val other = if (mover == "White") "Black" else "White"
        if (board.isCastling(move)) {
            return "$mover castles ${if (board.isKingsideCastling(move)) "kingside" else "queenside"}"
        }
        val piece = board.pieceTypeAt(move.from)
        var said = "$mover's ${COACH_PIECE_NAMES[if (piece == 0) PAWN else piece]} from " +
            "${squareName(move.from)} to ${squareName(move.to)}"
        if (board.isCapture(move)) {
            val taken = if (board.isEnPassant(move)) PAWN else board.pieceTypeAt(move.to)
            said += ", taking $other's ${COACH_PIECE_NAMES[if (taken == 0) PAWN else taken]}"
        }
        if (move.promotion != 0) said += ", promoting to a ${COACH_PIECE_NAMES[move.promotion]}"
        return said
    }

    /** A fact Python's `value not in (None, [], "", False)` leaves out. */
    private fun empty(value: Any?): Boolean =
        value == null || value == false || value == "" || (value is List<*> && value.isEmpty())

    /**
     * What the model is told of one key move, in colours: who played it, what it did, its kind,
     * the winning chances before and after, the reply that punishes it, the better move and its
     * line, its motifs, and the coach's facts for the position it was played from. `move_facts`.
     */
    fun moveFacts(game: ReviewedGame, ply: Int, user: String?): LinkedHashMap<String, Any?> {
        val i = ply - 1
        val move = game.moves[i]
        val before = if (i != 0) game.moves[i - 1].after else game.start
        val board = Board(move.fenBefore)
        val previous = if (i != 0) game.moves[i - 1].assessment.uci else null
        val facts = factsFromAnalysis(board, before, lastMove = previous)
        val mover = capitalized(move.side.value)
        val other = if (mover == "White") "Black" else "White"
        // The facts whose owner goes without saying to the side to move, named: the threats are
        // the other side's (luna gave White's Qa4+ to Black), the candidates and the line the mover's.
        val owned = mapOf(
            "threats" to "$other's threats",
            "candidates" to "$mover's candidates",
            "line" to "$mover's best line",
        )
        val visible = LinkedHashMap<String, Any?>()
        for ((key, value) in visibleFacts(facts, DisclosureLevel.LINE)) {
            if (key in DROP || empty(value)) continue
            visible[owned[key] ?: key] = if (key == "line") (value as List<*>).take(LINE_PLIES) else value
        }
        val position = inColours(visible, mover)
        val after = board.copy(withStack = false)
        after.push(after.parseUci(move.assessment.uci))
        val chances = positionScore(board, before.lines)
        val left = 1.0 - positionScore(after, move.after.lines) // the mover's, after
        val lines = move.after.lines
        val reply = if (lines.isNotEmpty() && lines[0].moves.isNotEmpty()) lines[0].moves[0] else null
        val kind = move.assessment.label?.let { ReviewJson.labelName(it) }
        val out = LinkedHashMap<String, Any?>()
        out["ply"] = ply
        out["move"] = move.san
        out["played by"] = if (move.side.value == user) "$mover (the student)" else mover
        out["what it does"] = moved(board, move.assessment.uci)
        out["kind"] = kind
        out["$mover's winning chances"] = "${pyRound(100 * chances)}% before, ${pyRound(100 * left)}% after"
        out["motifs"] = move.assessment.tags.map { it.name }.toSortedSet().toList()
        out["the position before it"] = position
        if (kind != null && kind in BAD) {
            if (!reply.isNullOrEmpty()) out["$other's best reply"] = after.san(Move.fromUci(reply))
            // The better move outright, and its line: read out of the facts, it was mistaken for
            // the reply (luna offered Black's answer as White's better move).
            if (move.assessment.best.isNotEmpty()) {
                val line = board.copy(withStack = false)
                val better = ArrayList<String>()
                for (uci in move.assessment.best.take(LINE_PLIES)) {
                    better.add(line.san(Move.fromUci(uci)))
                    line.push(line.parseUci(uci))
                }
                out["$mover's better move"] = better[0]
                out["its line"] = better
            }
        }
        return out
    }

    /** A call's message: the game in a line, then the key move's facts. `prompt`. */
    fun prompt(game: ReviewedGame, ply: Int, user: String?, opening: String?): String {
        val head = LinkedHashMap<String, Any?>()
        head["the student plays"] = if (!user.isNullOrEmpty()) capitalized(user) else "unknown"
        head["opening"] = opening
        head["result"] = if (game.headers.isNotEmpty()) game.headers["Result"] else null
        val told = LinkedHashMap<String, Any?>()
        told["game"] = head
        told["key move"] = moveFacts(game, ply, user)
        return pyJsonDumps(told, ensureAscii = false)
    }

    // "### why": each part's head; the part runs to the next. Plain text, not JSON: a quote left
    // unescaped in one paragraph lost a whole answer (luna, 2026-09-30). `^` and `$` as
    // re.MULTILINE has them: at a newline only.
    val PARTS = listOf("why", "better", "lesson")
    private val HEAD = Regex(
        "(?<![^\\n])[ \\t]*#{1,6}[ \\t]*\"?(${PARTS.joinToString("|")})\\b[^\\n]*(?![^\\n])",
        RegexOption.IGNORE_CASE,
    )

    /** The parts an answer holds, each one paragraph; none without its "why". `parse`. */
    fun parse(text: String): Map<String, String> {
        val heads = HEAD.findAll(text).toList()
        val ends = heads.drop(1).map { it.range.first } + text.length
        val out = LinkedHashMap<String, String>()
        for ((head, end) in heads.zip(ends)) {
            val paragraph = Py.split(text.substring(head.range.last + 1, end).replace("```", " ")).joinToString(" ")
            if (paragraph.isNotEmpty()) out[head.groupValues[1].lowercase()] = paragraph
        }
        return if ("why" in out) out else emptyMap()
    }

    // --- The student's words -------------------------------------------------------------------

    private val POSSESSIVE = Regex("\\b(?<side>[Ww]hite|[Bb]lack)'s\\b")

    // A side's colour before its pieces: "the Black king", "White pawns"; after "the", before what
    // else is a side's: "the White position". Not "White forces": that is a verb.
    private const val PIECES = "(?:king|queen|rook|bishop|knight|pawn|piece)s?"
    private val ADJECTIVE = Regex(
        "\\b(?:[Tt]he\\s+(?<a>[Ww]hite|[Bb]lack)\\s+(?=(?:$PIECES|army|forces|camp|side|position|" +
            "structure|majority|pair|player)\\b)|(?<b>White|Black)\\s+(?=$PIECES\\b))",
    )

    // A side on its own, and the verb it governs as a subject ("Black also has"), which agrees
    // with "you" once it is the student's.
    private const val ADVERBS =
        "also|still|then|now|only|not only|not|already|just|instead|never|often|always|even|soon|" +
            "again|thus|therefore|perhaps|sometimes|rather|no longer|first|next|later|[a-z]+ly"
    private val BARE = Regex(
        "\\b(?<side>White|Black)\\b(?<adverbs>(?:\\s+(?:$ADVERBS)\\b)*)" +
            "(?:\\s+(?<verb>[a-z]+(?:'t)?)\\b)?",
    )

    // The words after which a side starts a clause, and so governs the verb that follows it; and
    // the prepositions after which it is governed.
    private val CLAUSE = Regex(
        "(?:^|[.!?;:,(\\n\u2014\u2013]|\\b(?:and|but|so|because|while|when|whenever|if|after|before|" +
            "until|since|that|as|although|though|whereas|unless|once|where|then|now|here|instead|" +
            "meanwhile|otherwise|yet|or|nor|which|meaning))\\s*\$",
        RegexOption.IGNORE_CASE,
    )
    private val GOVERNED = Regex(
        "\\b(?:for|to|against|of|with|from|by|on|at|into|onto|over|like|than|between|among|" +
            "toward|towards|upon|about|without)\\s*\$",
        RegexOption.IGNORE_CASE,
    )
    private val AGREES = mapOf(
        "is" to "are",
        "was" to "were",
        "has" to "have",
        "does" to "do",
        "goes" to "go",
        "isn't" to "aren't",
        "wasn't" to "weren't",
        "hasn't" to "haven't",
        "doesn't" to "don't",
    )

    // The same for the opponent the other way: luna writes a side's verbs plural at times ("White
    // are down a queen"), as British English may; "your opponent" takes them singular.
    private val SINGULAR = mapOf(
        "are" to "is",
        "were" to "was",
        "have" to "has",
        "do" to "does",
        "aren't" to "isn't",
        "weren't" to "wasn't",
        "haven't" to "hasn't",
        "don't" to "doesn't",
    )

    // Verbs a review says of a side, as "White" governs them: known for verbs wherever they stand,
    // where a word in -s after a side could be a noun ("gives Black chances").
    private const val VERB_WORDS =
        "accepts|achieves|activates|addresses|advances|allows|answers|attacks|avoids|becomes|" +
            "begins|blocks|breaks|brings|builds|calculates|captures|castles|centralises|centralizes|" +
            "challenges|chases|chooses|clears|closes|collects|comes|commits|completes|concedes|" +
            "connects|consolidates|continues|controls|converts|covers|creates|defends|delays|" +
            "determines|develops|dictates|dominates|doubles|drives|eliminates|ends|ensures|enters|" +
            "escapes|establishes|exchanges|executes|exploits|exposes|eyes|faces|fails|finds|finishes|" +
            "fixes|follows|forces|frees|gains|gets|gives|goes|grabs|guards|handles|hangs|has|helps|" +
            "hits|holds|ignores|improves|increases|initiates|intends|invites|joins|keeps|lacks|lands|" +
            "launches|leads|leaves|lets|lifts|limits|locks|loses|maintains|makes|manages|marches|" +
            "meets|misses|mobilises|mobilizes|moves|needs|neglects|neutralises|neutralizes|obtains|" +
            "occupies|offers|opens|overlooks|pays|permits|picks|pins|places|plays|prepares|preserves|" +
            "presses|prevents|proceeds|produces|promotes|protects|provides|pulls|punishes|pursues|" +
            "pushes|puts|reaches|reacts|realises|realizes|recaptures|recognises|recognizes|recovers|" +
            "reduces|regains|regroups|reinforces|relieves|relocates|remains|removes|replies|" +
            "repositions|reroutes|resigns|resolves|responds|restricts|retains|retreats|returns|" +
            "reveals|risks|runs|sacrifices|saves|secures|seeks|sees|seizes|sends|sets|settles|shifts|" +
            "simplifies|slows|solidifies|solves|spends|stabilises|stabilizes|stands|starts|stays|" +
            "steps|stops|strengthens|strikes|supports|survives|swaps|switches|takes|targets|" +
            "threatens|tightens|trades|transfers|transforms|traps|tries|turns|undermines|unpins|uses|" +
            "vacates|waits|wants|wastes|weakens|wins|withdraws"
    private val VERBS = Regex(VERB_WORDS)

    // A plural the model's "their" may be of, between the side and it: "Black's rooks and their
    // files".
    private val PLURAL = Regex(
        "\\b(?:pieces|pawns|rooks|knights|bishops|queens|squares|files|ranks|diagonals)\\b" +
            "|\\b(?:king|queen|rook|bishop|knight|pawn)\\s+and\\s+(?:\\w+\\s+)?(?:king|queen|rook|bishop|" +
            "knight|pawn)\\b",
    )
    private val SIDE_OR_THEIRS = Regex(
        "\\b(?<side>White|Black)(?<owns>'s)?\\b|(?<!of )(?<!among )(?<!between )" +
            "\\b(?<pronoun>their|they|them)\\b|(?<end>[.!?])",
        RegexOption.IGNORE_CASE,
    )

    // A further verb of the same subject: "you block the file and leave the queen", "you improve
    // the rook, wait for the reply, and then use it".
    private val JOINED = Regex(
        "(?:,\\s*|\\b(?:and|or|but)\\s+)(?:(?:and|or|but)\\s+)?(?:(?:$ADVERBS)\\s+)*" +
            "(?<verb>[a-z]+)\\b(?:\\s+(?<next>[a-z]+))?",
    )

    // The words a verb's object opens with: "and stabilizes your position".
    private val OBJECT = Regex(
        "the|a|an|your|their|its|his|her|this|that|these|those|every|any|no|more|both|each|all",
    )

    // What ends a subject's verbs: another subject ("then the knight reaches"), a colon or dash.
    private val NEW_SUBJECT = Regex(
        "\\b(?:your opponent|it|he|she|which|who|that|this|these|those|they)\\b|[;:(\u2014\u2013]" +
            "|\\b(?:the|a|an)\\s+[a-z]+\\s+(?:$VERB_WORDS)\\b",
    )
    private val SENTENCE = Regex("[^.!?]+[.!?]?")
    private val YOU = Regex("\\byou\\b", RegexOption.IGNORE_CASE)
    private val YOU_WHO = Regex("\\byou, who (is|was|has)\\b")
    private val SENTENCE_GAP = Regex("(?<=[.!?])(\\s+)")
    private val YOUR_OPPONENT = Regex("\\byour opponent\\b", RegexOption.IGNORE_CASE)
    private val YOUR_OPPONENTS = Regex("\\byour opponent's\\b")

    /** A verb as "you" says it: "has" -> "have", "threatens" -> "threaten", "pushes" -> "push". `_base`. */
    private fun base(verb: String): String {
        AGREES[verb]?.let { return it }
        if (verb.length < 4 || !verb.endsWith("s") || verb.endsWith("ss") || verb.endsWith("us") || verb.endsWith("is")) {
            return verb
        }
        if (verb.endsWith("ies") && verb.length > 4) return verb.dropLast(3) + "y"
        if (listOf("ches", "shes", "sses", "xes", "zzes").any { verb.endsWith(it) }) return verb.dropLast(2)
        return verb.dropLast(1)
    }

    /** A verb as "your opponent" says it, from its plain form: "need" -> "needs". `_third`. */
    private fun third(verb: String): String {
        if (listOf("s", "sh", "ch", "x", "z").any { verb.endsWith(it) }) return verb + "es"
        // `verb[-2:-1] not in "aeiou"`: the letter before the y; "" (a lone "y") is in any string.
        val before = if (verb.length >= 2) verb.substring(verb.length - 2, verb.length - 1) else ""
        if (verb.endsWith("y") && !"aeiou".contains(before)) return verb.dropLast(1) + "ies"
        return verb + "s"
    }

    private fun opensSentence(text: String, start: Int): Boolean {
        val before = text.substring(0, start).trimEnd { Py.isSpace(it) }
        return before.isEmpty() || before.last() in ".!?\n"
    }

    private fun cased(words: String, text: String, start: Int): String =
        if (opensSentence(text, start)) words[0].uppercaseChar() + words.substring(1) else words

    /**
     * The model's "their", "they", "them" of the student's side, in its colour. A pronoun is of
     * the side its sentence named on its own last (a subject, most often), else of the side it
     * named as an owner last, else of the side named last before it -- unless a plural came
     * between ("Black's rooks and their files"). `_owned`.
     */
    private fun owned(text: String, ours: String): String {
        var alone: Pair<String, Int>? = null // this sentence's, and where each was named
        var owner: Pair<String, Int>? = null
        var last: Pair<String, Int>? = null
        val out = StringBuilder()
        var at = 0
        for (found in SIDE_OR_THEIRS.findAll(text)) {
            if (found.groups["end"] != null) {
                alone = null
                owner = null
                continue
            }
            val side = found.groups["side"]
            if (side != null) {
                val named = capitalized(side.value) to found.range.last + 1
                last = named
                if (found.groups["owns"] != null) owner = named else alone = named
                continue
            }
            val said = alone ?: owner ?: last
            val start = found.range.first
            // `_PLURAL.search(text, side[1], found.start())`: between the side and the pronoun.
            if (said == null || said.first != ours || PLURAL.find(text.substring(0, start), said.second) != null) continue
            val word = found.groups["pronoun"]!!.value.lowercase()
            out.append(text, at, start).append(if (word == "their") "$ours's" else ours)
            at = found.range.last + 1
        }
        return out.append(text, at, text.length).toString()
    }

    /**
     * The verbs joined to the student's own ("you block the file and leaves") agreeing too, in the
     * same sentence and before any other subject. `_joined`.
     */
    private fun joined(text: String): String {
        val out = StringBuilder()
        var at = 0
        for (sentence in SENTENCE.findAll(text)) {
            val body = sentence.value
            val begin = sentence.range.first
            for (subject in YOU.findAll(body)) {
                if (CLAUSE.find(body.substring(0, subject.range.first)) == null) {
                    continue // "allows you to consolidate and opens": the subject is another's
                }
                val from = subject.range.last + 1
                val cut = NEW_SUBJECT.find(body, from)
                val stop = cut?.range?.first ?: body.length
                // `_JOINED.finditer(body, subject.end(), stop)`: the body cut at the next subject.
                for (further in JOINED.findAll(body.substring(0, stop), from)) {
                    val group = further.groups["verb"]!!
                    val verb = group.value
                    val i = begin + group.range.first
                    val named = VERBS.matches(verb) ||
                        (verb.endsWith("s") && OBJECT.matches(further.groups["next"]?.value ?: ""))
                    if (named && i >= at) {
                        out.append(text, at, i).append(base(verb))
                        at = i + verb.length
                    }
                }
            }
        }
        return out.append(text, at, text.length).toString()
    }

    /**
     * A paragraph written in colours as the student reads it: their side "you" (its verbs
     * agreeing), the other "your opponent"; unchanged when their side is not known. `to_student`.
     */
    fun toStudent(paragraph: String, user: String?): String {
        if (user == null) return paragraph
        val ours = capitalized(user)

        // Each step reads its own input as Python's `found.string`.
        val coloured = owned(straightQuotes(paragraph), ours)
        val possessed = POSSESSIVE.replace(coloured) { found ->
            val mine = capitalized(found.groups["side"]!!.value) == ours
            cased(if (mine) "your" else "your opponent's", coloured, found.range.first)
        }
        val adjectived = ADJECTIVE.replace(possessed) { found ->
            val side = capitalized((found.groups["a"] ?: found.groups["b"])!!.value)
            val words = if (side == ours) "your " else "your opponent's "
            cased(words, possessed, found.range.first)
        }
        val bared = BARE.replace(adjectived) { found ->
            val start = found.range.first
            val tail = found.groups["adverbs"]?.value ?: ""
            var verb = found.groups["verb"]?.value
            val before = adjectived.substring(0, start)
            val words: String
            if (found.groups["side"]!!.value != ours) {
                words = "your opponent"
                if (verb != null && GOVERNED.find(before) == null) {
                    val singular = SINGULAR[verb]
                    if (singular != null) {
                        verb = singular
                    } else if (CLAUSE.find(before) != null && !VERBS.matches(verb)) {
                        verb = if (VERBS.matches(third(verb))) third(verb) else verb
                    }
                }
            } else {
                words = "you"
                val known = verb in AGREES || VERBS.matches(verb ?: "")
                if (verb != null && GOVERNED.find(before) == null) {
                    verb = if (known || CLAUSE.find(before) != null) base(verb) else verb
                }
            }
            cased(words, adjectived, start) + tail + (if (verb != null) " $verb" else "")
        }
        val who = YOU_WHO.replace(bared) { "you, who ${AGREES.getValue(it.groupValues[1])}" }
        return varied(joined(who))
    }

    /**
     * "your opponent's" said again in a sentence as "their": "your opponent needs to play where
     * their pawns point" (in the student's words "their" can be no one else's). `_varied`.
     */
    private fun varied(text: String): String {
        // `re.split(r"(?<=[.!?])(\s+)", text)`: the sentences, and the gaps kept between them.
        fun one(sentence: String): String {
            val first = YOUR_OPPONENT.find(sentence) ?: return sentence
            val end = first.range.last + 1
            return sentence.substring(0, end) + YOUR_OPPONENTS.replace(sentence.substring(end), "their")
        }
        val out = StringBuilder()
        var at = 0
        for (gap in SENTENCE_GAP.findAll(text)) {
            out.append(one(text.substring(at, gap.range.first))).append(gap.value)
            at = gap.range.last + 1
        }
        return out.append(one(text.substring(at))).toString()
    }

    // --- The checks: false claims about the board ----------------------------------------------

    // A move given to a side: "White's Nf3", "Black played Nc6", "White could have played Bc4".
    private const val GIVEN =
        "(?:'s|\\s+(?:played|plays|chose|chooses|went\\s+for|goes\\s+for|" +
            "(?:can|could|should|must|would|will)(?:\\s+have)?\\s+play(?:ed)?|" +
            "(?:answer|answers|answered|reply|replies|replied|respond|responds|responded)\\s+with))"

    /**
     * The positions a paragraph may speak of: after the move, along the reply's line, and along
     * the best line from before it. `_frames`.
     */
    private fun frames(game: ReviewedGame, ply: Int): List<Board> {
        val move = game.moves[ply - 1]
        val before = Board(move.fenBefore)
        val frames = ArrayList<Board>()
        val after = before.copy(withStack = false)
        after.push(after.parseUci(move.assessment.uci))
        frames.add(after.copy(withStack = false))
        if (move.after.lines.isNotEmpty()) {
            for (uci in move.after.lines[0].moves.take(LINE_PLIES)) {
                after.push(after.parseUci(uci))
                frames.add(after.copy(withStack = false))
            }
        }
        val along = before.copy(withStack = false)
        for (uci in move.assessment.best.take(LINE_PLIES)) {
            along.push(along.parseUci(uci))
            frames.add(along.copy(withStack = false))
        }
        return frames
    }

    /**
     * Who plays each move a paragraph on move [ply] may name, by colour: the move and the reply's
     * line after it, the better move's line; a move both sides play there is left out. `_players`.
     */
    private fun players(game: ReviewedGame, ply: Int): Map<String, String> {
        val move = game.moves[ply - 1]
        val start = Board(move.fenBefore)
        val reply = if (move.after.lines.isNotEmpty()) move.after.lines[0].moves.take(LINE_PLIES) else emptyList()
        val seen = LinkedHashMap<String, MutableSet<String>>()
        for (line in listOf(listOf(move.assessment.uci) + reply, move.assessment.best.take(LINE_PLIES))) {
            val board = start.copy(withStack = false)
            for (uci in line) {
                val played = Move.fromUci(uci)
                val side = if (board.turn == WHITE) "White" else "Black"
                seen.getOrPut(board.san(played).trimEnd('+', '#')) { LinkedHashSet() }.add(side)
                board.push(played)
            }
        }
        val out = LinkedHashMap<String, String>()
        for ((san, sides) in seen) if (sides.size == 1) out[san] = sides.single()
        return out
    }

    /**
     * A move the paragraph gives to the side that does not play it ("Black played Nf3" of White's
     * Nf3), in words; empty when none. `wrong_side`.
     */
    fun wrongSide(paragraph: String, game: ReviewedGame, ply: Int): String {
        val text = straightQuotes(paragraph)
        for ((san, side) in players(game, ply)) {
            val found = Regex("\\b(White|Black)$GIVEN\\s+${escaped(san)}(?![\\w])").find(text)
            if (found != null && found.groupValues[1] != side) return "${found.value}: $san is $side's"
        }
        return ""
    }

    /**
     * The false claims about the board a "why" or "better" on move [ply] makes, in words; empty
     * when none: a piece the board puts elsewhere or with the other side, a move given to the side
     * that does not play it. A "your" the model wrote anyway is the student's, as the student
     * reads it (the mover's when their side is not known). The lesson, general advice, makes no
     * claim about the board. `problems`.
     */
    fun problems(part: String, paragraph: String, game: ReviewedGame, ply: Int, user: String? = null): String {
        if (part == "lesson") return ""
        val board = Board(game.moves[ply - 1].fenBefore)
        val us = user?.let { it == "white" }
        val found = checkDraft(paragraph, board, after = frames(game, ply), us = us)
        if (found.isNotEmpty()) return found.joinToString("; ") { "${it.text}: ${it.reason}" }
        return wrongSide(paragraph, game, ply)
    }

    /**
     * A move's parts with no false claim about the board, in the student's words, each on its
     * own; why a part was refused (empty when none). `checked`.
     */
    fun checked(parts: Map<String, String>, game: ReviewedGame, ply: Int, user: String?): Pair<Map<String, String>, String> {
        val label = game.moves[ply - 1].assessment.label
        // "better" is a bad move's alone: a good move was the better move.
        val wanted = if (label != null && ReviewJson.labelName(label) in BAD) PARTS else listOf("why", "lesson")
        val kept = LinkedHashMap<String, String>()
        val reasons = ArrayList<String>()
        for (part in wanted) {
            val paragraph = parts[part] ?: continue
            val problem = problems(part, paragraph, game, ply, user)
            if (problem.isNotEmpty()) {
                reasons.add("$part: $problem")
                if (part == "why") {
                    // Wrong about the board or a side: the model misread the move, so nothing it
                    // said of it stands.
                    return Pair(emptyMap(), reasons.joinToString("; "))
                }
                continue
            }
            kept[part] = toStudent(paragraph, user)
        }
        return Pair(kept, reasons.joinToString("; "))
    }

    /**
     * The model's review of the game's key moves, checked: a call a move, all at once. Without
     * thinking unless asked: a small model reasons worse about chess, not better (Tai,
     * 2026-09-30). A call that fails leaves its move in `failed`, to ask again; when every one
     * fails (and none was answered before), the first failure is thrown. [answered]: the answers already paid for (words the
     * app was closed in the middle of), not asked again; [onAnswer] (ply, text, stats) is handed
     * each new answer the moment it arrives, to keep it. The cost is of the new calls. `write`.
     */
    fun write(
        game: ReviewedGame,
        provider: LLMProvider,
        user: String?,
        opening: String?,
        prices: Prices,
        thinking: Boolean = false,
        answered: Map<Int, String>? = null,
        onAnswer: ((Int, String, Map<String, Any?>) -> Unit)? = null,
        stop: (() -> Boolean)? = null,
    ): Words {
        val paid = LinkedHashMap(answered ?: emptyMap())
        val plies = keyPlies(game)
        // `getattr(provider, "model", "")`.
        val words = Words(model = modelOf(provider) ?: "")
        if (plies.isEmpty()) return words
        val todo = plies.filter { it !in paid }
        // A model that reasons by its effort (OpenAI's) reasons unless told not to:
        // `getattr(provider, "with_effort", None)`, which the two providers have.
        val asked = if (thinking) {
            provider
        } else {
            when (provider) {
                is OpenAICompatProvider -> provider.withEffort("none")
                is OfflineProvider -> provider.withEffort("none")
                else -> provider
            }
        }

        // [stop] true asks no more (the game removed meanwhile): the calls not made fail.
        fun ask(ply: Int): Any = if (stop != null && stop()) IllegalStateException("stopped") else try {
            asked.chat(
                SYSTEM,
                listOf(ChatMessage("user", prompt(game, ply, user, opening))),
                maxTokens = ANSWER_TOKENS,
                temperature = 0.4,
                thinking = thinking,
            )
        } catch (failed: Exception) { // one move's call; the rest stand
            failed
        }

        val replies = LinkedHashMap<Int, Any>()
        if (todo.isNotEmpty()) {
            // `as_completed`: each answer handled here, on the caller's thread, as it comes.
            val pool = Executors.newFixedThreadPool(minOf(PARALLEL, todo.size)) { task ->
                Thread(task, "sidecoach-words").apply { isDaemon = true }
            }
            try {
                val service = ExecutorCompletionService<Pair<Int, Any>>(pool)
                for (ply in todo) service.submit { ply to ask(ply) }
                repeat(todo.size) {
                    val (ply, reply) = service.take().get()
                    replies[ply] = reply
                    if (onAnswer != null && reply is Completion) {
                        try {
                            onAnswer(ply, reply.text, reply.stats)
                        } catch (_: Exception) { // not kept only costs a resume
                        }
                    }
                }
            } finally {
                pool.shutdown()
            }
        }
        val failures = replies.values.filterIsInstance<Exception>()
        if (todo.isNotEmpty() && failures.size == todo.size && paid.isEmpty()) throw failures[0]
        val spent = Usage()
        for (ply in plies) {
            val reply = replies[ply]
            if (reply is Exception) {
                words.dropped[ply] = "the call failed: ${reply.message ?: ""}"
                words.failed.add(ply)
                continue
            }
            val text = if (reply is Completion) {
                spent.record(reply.stats)
                words.calls.add(reply.stats)
                reply.text
            } else {
                paid.getValue(ply)
            }
            words.answers[ply] = text
            val (kept, dropped) = checked(parse(text), game, ply, user)
            if (kept.isNotEmpty()) words.texts[ply] = kept
            if (dropped.isNotEmpty()) words.dropped[ply] = dropped
        }
        words.cost = spent.cost(prices)
        return words
    }

    /**
     * What the calls come to at [prices] (dollars), as the button says it: the prompts as sent,
     * and a typical answer a move ([TYPICAL_ANSWER]); the moves [answered] already not asked.
     * `estimate`. The characters counted as Python's `len` counts them, by code point.
     */
    fun estimate(
        game: ReviewedGame,
        user: String?,
        opening: String?,
        prices: Prices,
        answered: Set<Int> = emptySet(),
    ): Double? {
        if (!prices.known) return null
        val plies = keyPlies(game).filter { it !in answered }
        var sent = 0L
        for (ply in plies) sent += chars(SYSTEM) + chars(prompt(game, ply, user, opening))
        val answers = plies.size * TYPICAL_ANSWER
        return sent / CHARS_PER_TOKEN / 1e6 * prices.prompt!! + answers / 1e6 * prices.completion!!
    }

    private fun chars(text: String): Int = text.codePointCount(0, text.length)

    /** `str.capitalize()`: "white" -> "White". */
    private fun capitalized(word: String): String =
        if (word.isEmpty()) word else word.substring(0, 1).uppercase() + word.substring(1).lowercase()

    /** `round(x)`: the nearest integer, a half to the even one. */
    private fun pyRound(value: Double): Long = Math.rint(value).toLong()

    /**
     * `re.escape` of a move: its regex metacharacters escaped, the rest as it is ("=" and "-"
     * are literal outside a set in Java's and ICU's syntax alike).
     */
    internal fun escaped(text: String): String = buildString {
        for (c in text) {
            if (c in "\\^\$.|?*+()[]{}") append('\\')
            append(c)
        }
    }
}
