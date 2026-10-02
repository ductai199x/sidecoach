package com.sidecoach.coach

import com.sidecoach.analysis.positionKey
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.chess.WHITE
import com.sidecoach.events.Analysis
import com.sidecoach.events.Annotation
import com.sidecoach.events.ClockObservation
import com.sidecoach.events.CoachMessage
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameReview
import com.sidecoach.events.GameStarted
import com.sidecoach.events.insightWords
import com.sidecoach.events.Line
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.ProfileEvidence
import com.sidecoach.events.ProfileSnapshot
import com.sidecoach.events.PuzzleOutcome
import com.sidecoach.events.Recommendation
import com.sidecoach.events.Resync
import com.sidecoach.events.ScreenState
import com.sidecoach.events.ScreenStateChanged
import com.sidecoach.events.Severity
import com.sidecoach.events.Takeback
import com.sidecoach.events.Trigger
import com.sidecoach.llm.ChatMessage
import com.sidecoach.llm.LLMProvider
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.profile.Causes
import com.sidecoach.vision.Pieces
import org.json.JSONArray
import org.json.JSONObject

/**
 * The second half of `coach/coach.py`: what the user's words mean for disclosure, and the
 * Coach itself -- the events in, the instruction and template for each moment, the messages
 * out. One Kotlin file per class keeps them readable; the package is the module.
 */

// What the user's words mean for disclosure. Deterministic on purpose: the model never
// decides how much may be said.

/**
 * A hint, or a question about the plan ("what should I be aiming for?", "what's the idea
 * here?"): both earn the next level, where the theme and the plan may be named. Answered at
 * the question level, a question about the plan got a question back.
 */
private val HINT = Regex(
    "\\b(hint|clue|nudge|help me|help|stuck|where should i look|" +
        "what (?:should|do|can|could) i (?:be )?(?:aim|go|look|play|work|strive)(?:ing)? for|" +
        "what(?:'s| is) (?:the|my) (?:plan|idea|goal|aim)|what to (?:aim|go) for|" +
        "what am i (?:aiming|going|playing) for)\\b",
    RegexOption.IGNORE_CASE,
)

/** An overall verdict on the game: "how did I do", "how was my game", "overall", "summary". */
internal val ASK_REVIEW = Regex(
    "\\b(?:how (?:did|have) i (?:do|play|done|played|playing|doing)|how am i playing|" +
        "how was (?:my|the) game|" +
        "overall|summar(?:y|ise|ize)|review (?:my|the) game|rate my (?:game|play)|" +
        "where did i (?:lose|go wrong)|what (?:do|did) i keep (?:missing|doing)|" +
        "what (?:did|have) i (?:do|done) wrong|my (?:mistakes|worst moves?)|" +
        "what should i work on)\\b",
    RegexOption.IGNORE_CASE,
)

/**
 * A question about the player over their games, not about this position: the record answers
 * it, the coach writing its own queries. "This game" and "the last move" stay with the
 * game's own paths above.
 */
private val ASK_PROFILE = Regex(
    "\\b(?:my (?:games|history|record|profile|stats|statistics|weakness(?:es)?|" +
        "strengths?|habits?|pattern|trend|progress|results?|last (?:few |\\d+ )?games)|" +
        "(?:over|in) (?:the )?(?:last|past|previous) (?:few |\\d+ )?" +
        "(?:games|days|weeks|months|sessions)|" +
        "(?:all|previous|earlier|other|past|recent) (?:games|sessions)|last session|" +
        "(?:across|between|over) (?:games|sessions)|how many games|" +
        "(?:do|am|have) i (?:always|usually|often|tend to|keep|generally|typically|normally)|" +
        "why (?:do|am) i (?:always|usually|often|keep|so|bad|weak|losing|blundering)|" +
        "(?:am|are) i (?:getting )?(?:better|worse|improving)|" +
        "(?:what|which) (?:openings?|tactics?|motifs?|phase) (?:do|am|have) i|" +
        "(?:my|the) (?:biggest|worst|most common|main|recurring) " +
        "(?:weakness|problem|mistake|cost|leak)s?|" +
        "what should i (?:study|practise|practice|train|focus on|" +
        "work on (?:in general|next|overall)))\\b",
    RegexOption.IGNORE_CASE,
)

/** Questions about facts that are never withheld: the material, the opening, who is better. */
private val ASK_MATERIAL = Regex(
    "\\b(?:material|(?:how many|what) (?:pieces|pawns)|(?:am|are) i up|who'?s up|up material|" +
        "down material|piece count)\\b",
    RegexOption.IGNORE_CASE,
)
private val ASK_OPENING = Regex("\\b(?:opening|theory|book move)\\b", RegexOption.IGNORE_CASE)

/**
 * A question about the move the player already played: its verdict is in the record and is
 * theirs in full -- that position is past, and the disclosure level binds only the one now.
 */
private val ASK_LAST_MOVE = Regex(
    "\\b(?:why was (?:that|it|my (?:last )?move) (?:bad|wrong|a mistake|a blunder|an inaccuracy)|" +
        "what was better|what should i have (?:played|done)|what did i miss|what went wrong|" +
        "was (?:that|it|my (?:last )?move) (?:a )?(?:mistake|blunder|bad|wrong)|" +
        "what'?s wrong with (?:that|it|my (?:last )?move)|why (?:is|was) (?:that|it) (?:so )?bad|" +
        "what did i do wrong|did i (?:just )?(?:blunder|mess up))\\b",
    RegexOption.IGNORE_CASE,
)

/**
 * A question about the opponent's last move: what it does, why it was played, whether it was
 * good. Its verdict and effects are in the record ("Last move") and are the player's to hear
 * in full; answered as a generic ask, the coach gave a question back and named neither the
 * move nor its verdict in most of them.
 */
private val ASK_OPPONENT_MOVE = Regex(
    "\\b(?:why did (?:he|she|they) (?:play|do) that|what does that (?:move )?do|" +
        "what(?:'s| is) the (?:point|idea) of that(?: move)?|" +
        "was that a (?:blunder|mistake|good move)|" +
        "did (?:he|she|they) (?:just )?(?:mess up|blunder)|" +
        "what (?:is|was) (?:he|she|they) (?:going for|up to|doing)|" +
        "why (?:that|this) move)\\b",
    RegexOption.IGNORE_CASE,
)
private val ASK_EVAL = Regex(
    "\\b(?:am i (?:winning|losing|better|worse|ok|okay|in trouble)|who'?s (?:better|winning)|" +
        "how am i doing|is this (?:position )?good for me|" +
        "what(?:'?s| is) the (?:eval|evaluation|score)|how bad is it|am i lost)\\b",
    RegexOption.IGNORE_CASE,
)

/** A review line's habit, left out for a strong player ([Coach.onReview]). */
private val HABIT_SAID = Regex(" The habit that fixes it: .*$")

/** What a move cost, asked in so many words: only then is the figure said. */
private val ASK_COST = Regex(
    "\\b(?:how much (?:did|does) (?:it|that|this|my move)\\s*(?:\\w+\\s*)?cost|what did (?:it|that) " +
        "cost|how (?:bad|costly) was (?:it|that|my move)|how much (?:did|have) i lose)\\b",
    RegexOption.IGNORE_CASE,
)

/** A bare request for a hint, with nothing claimed: "hint", "give me a hint", "a clue". */
internal val ASK_HINT = Regex(
    "^\\W*(?:(?:give me |i need |can i have |i want |any |another |one more )?(?:a |an |some )?" +
        "(?:hint|clue|nudge|help|tip)s?(?: please)?|i'?m stuck|i am stuck|where (?:should|do) i look" +
        "|i don'?t know what to (?:do|play))\\W*\$",
    RegexOption.IGNORE_CASE,
)

/** The player's own move, named as theirs: asked after the opponent has replied too. */
internal val MY_MOVE = Regex(
    "\\b(?:what should i have (?:played|done)|what did i (?:miss|do wrong)|did i (?:just )?" +
        "(?:blunder|mess up)|my (?:last )?move|was i wrong|should i have)\\b",
    RegexOption.IGNORE_CASE,
)

/** A question about the opponent's threat. */
internal val ASK_THREAT = Regex(
    "\\b(?:what(?:'s|s| is| are) (?:their|the|his|her|my opponent's|the opponent's) threats?" +
        "|what (?:do|does) (?:they|he|she|my opponent|the opponent) threaten" +
        "|what (?:is|are|'s|'re) (?:they|he|she|my opponent|the opponent) (?:threatening|up to)" +
        "|is (?:there|anything) (?:a )?threat(?:ened)?|am i (?:in danger|threatened)" +
        "|what (?:do|does) (?:they|he|she|my opponent|the opponent) want" +
        "|is (?:anything|something) (?:hanging|loose|en prise)" +
        "|what (?:are|is) (?:they|he|she|my opponent|the opponent) going for)\\b",
    RegexOption.IGNORE_CASE,
)

/** A question about what to aim for: the first hint answers it, and its template is the floor. */
private val ASK_AIM = Regex(
    "\\bwhat (?:should|do|can|could) i (?:be )?(?:aim|go|look|play|work|strive)(?:ing)? for\\b" +
        "|\\bwhat(?:'s| is) (?:the|my) (?:plan|idea|goal|aim)\\b|\\bwhat am i (?:aiming|going|playing) " +
        "for\\b|\\bwhat to (?:aim|go) for\\b|\\bhow (?:do|can|should) i (?:make progress|improve|" +
        "continue|proceed)\\b",
    RegexOption.IGNORE_CASE,
)

/**
 * Words that can carry a claim about the board worth checking: a move, a square, a piece. A
 * bare "hint" or "what should I play" carries none, and skipping the check is what makes the
 * buttons answer at once.
 */
private val CLAIM_WORDS = Regex(
    "\\b(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?|[a-h][1-8]|" +
        "pawn|knight|bishop|rook|queen|king|castl\\w*|check|mate|fork|pin|skewer|hang\\w*|" +
        "threat\\w*|attack\\w*|capture\\w*|take\\w*|trade\\w*|sac\\w*|win\\w*|los\\w*|" +
        "safe|blunder|mistake|good|bad|better|best|worse|worst)\\b",
    RegexOption.IGNORE_CASE,
)

private val QUESTION_START = Regex(
    "^(?:what|what's|whats|where|why|how|which|who|whose|is|isn't|are|can|could|should|" +
        "would|will|does|do|did|was|were|am)\\b",
    RegexOption.IGNORE_CASE,
)

private val SENTENCE = Regex("[^.!?]+[.!?]*")

/**
 * The sentences of the player's words that ask something -- a "?", or a question's first word,
 * as dictation leaves it -- or "" when none does.
 */
fun questionPart(text: String): String {
    val sentences = SENTENCE.findAll(text.replace("...", "…")).map { it.value }
    val asked = sentences.map { it.trim() }
        .filter { it.endsWith("?") || QUESTION_START.containsMatchIn(it) }
    return asked.joinToString(" ").replace("…", "...")
}

/** The opening asked for by name ("what opening is this?"), not judged ("how is my opening?"). */
private val OPENING_FACT = Regex(
    "\\b(?:what|which)(?:'s| is| was)? (?:the |this |my )?opening\\b|" +
        "\\bis (?:this|it|that) (?:still )?(?:theory|book)\\b|\\bname of (?:the |this )?opening\\b",
    RegexOption.IGNORE_CASE,
)
private val FACT_OTHER_WORDS = Regex(
    "\\b(?:why|how come|what should|what do i|explain|plan)\\b",
    RegexOption.IGNORE_CASE,
)

/**
 * Whether the words ask only for a fact the record holds whole -- the material, the opening, who
 * stands better -- which the template says exactly. The model added nothing there (the locked
 * test set, 2026-09-26: excellence 1.00 for the templates on all three) and once gave the
 * material after a recapture as the material now; answered by the template, no call is made.
 * `coach.fact_question`.
 */
fun factQuestion(text: String): Boolean {
    if (!(ASK_MATERIAL.containsMatchIn(text) || OPENING_FACT.containsMatchIn(text) ||
            ASK_EVAL.containsMatchIn(text))
    ) {
        return false
    }
    val others = ASK_HINT.matchesAt(text, 0) ||
        ASK_AIM.containsMatchIn(text) ||
        ASK_THREAT.containsMatchIn(text) ||
        ASK_MOVE.containsMatchIn(text) ||
        ASK_LINE.containsMatchIn(text) ||
        ASK_LAST_MOVE.containsMatchIn(text) ||
        ASK_OPPONENT_MOVE.containsMatchIn(text) ||
        ASK_REVIEW.containsMatchIn(text) ||
        ASK_COST.containsMatchIn(text) ||
        FACT_OTHER_WORDS.containsMatchIn(text) ||
        SAN.containsMatchIn(text)
    // Only the words beyond the question itself can carry a claim: "winning" in "am I winning?"
    // is the question.
    var rest = text
    for (pattern in listOf(ASK_MATERIAL, OPENING_FACT, ASK_EVAL)) rest = pattern.replace(rest, " ")
    return !others && !mayCarryClaims(rest)
}

/**
 * Whether the user's words could state something about the board to check. "What is he
 * threatening?" states nothing: it is answered from the facts' threats, not examined as a named
 * line (the examination read it back, question and move numbers, as the answer).
 */
fun mayCarryClaims(text: String): Boolean {
    if (ASK_HINT.matchesAt(text, 0) || ASK_REVIEW.containsMatchIn(text) || ASK_THREAT.containsMatchIn(text)) {
        return false
    }
    return CLAIM_WORDS.containsMatchIn(text)
}

/**
 * The move asked for, in words that ask for it: "which move" and "the answer" said of a move
 * already played -- "which move did they just play?", "what was the answer to their last
 * threat?" -- opened the move level, and the answer was given (the coach review, 2026-10-01).
 * `coach._ASK_MOVE`.
 */
private val ASK_MOVE = Regex(
    "\\b((?<!\\bwas the )(?<!\\bwas that the )(?<!\\bwas it the )best move|" +
        "what (?:should|would) i play|what would you play|" +
        "which move (?:should i (?:play|make)|is (?:best|better)|would you (?:play|make|suggest)|" +
        "do you (?:play|suggest|recommend))|" +
        "just tell me|what'?s the move|(?:tell|show|give) me the move|" +
        "(?:(?:tell|give|show) me|what(?:'s| is)) the answer)\\b",
    RegexOption.IGNORE_CASE,
)

/**
 * The moves besides the best one asked for as such: "any other good moves?", "what else works?",
 * "which moves are fine here?" -- the player asks for moves, and gets all that are fine, named
 * ([alternativesAnswer]). `coach._ASK_OTHER_MOVES`.
 */
internal val ASK_OTHER_MOVES = Regex(
    "\\bwhat else (?:works|is good|is fine|is playable|can i play|could i play|do i have)\\b" +
        "|\\bwhich moves (?:are|would be|work|hold)\\b" +
        "|\\b(?:other|another|more|alternative) (?:good |decent |fine |playable |strong |" +
        "reasonable )?(?:moves?|options?|candidates?|alternatives?)\\b" +
        "|\\b(?:best|good|fine|decent|playable|candidate|top) moves\\b|\\balternatives?\\b",
    RegexOption.IGNORE_CASE,
)

/**
 * Another move asked about as good as the one they have in mind or the one named: "is Nf3 also
 * good?", "does d4 work too?". Opens nothing by itself. `coach._ASK_ALSO`.
 */
internal val ASK_ALSO = Regex(
    "\\b(?:also|as well|too)\\b[^.?!]{0,25}\\b(?:good|fine|ok|okay|playable|works?|strong)\\b" +
        "|\\b(?:good|fine|ok|okay|playable|works?)\\b[^.?!]{0,15}\\b(?:also|as well|too)\\b",
    RegexOption.IGNORE_CASE,
)

/** "I am not stuck, just thinking" is no request for a hint. `coach._NOT_STUCK`. */
private val NOT_STUCK = Regex("(?:\\bnot|n't|\\bnever)\\s+(?:\\w+\\s+)?stuck\\b", RegexOption.IGNORE_CASE)
/**
 * How the position goes on is the line: "how would this position continues?" and "ok and after
 * that?" after a puzzle's last move unlocked nothing (Tai's puzzle, 2026-09-28). "After that"
 * only as a question: in a statement ("and after that I lost my queen") it asks for nothing.
 * A line asked for as one -- "the line", "some lines", "the main variation" -- not the word: "are
 * there any open lines for my rooks?" opened the line (the coach review, 2026-10-01).
 */
private val ASK_LINE = Regex(
    "\\b(?:the|a|some|main|best|engine'?s?)\\s+(?:(?:main|best|whole|full|engine'?s?|critical|" +
        "forcing|winning)\\s+)?(?:lines?|variations?|continuations?)\\b" +
        "|\\b(?:what happens (?:after|next)|what follows)\\b" +
        "|\\b(?:and )?then what\\W*\$|\\bwhat(?:'s| is| comes) next\\W*\$" +
        "|\\bafter that\\s*\\?" +
        "|\\bhow (?:would|will|does|do|could|can) (?:it|this|that|the game|(?:this|the) position|" +
        "things) (?:continue|continues|go on|goes on|play out|proceed)\\b",
    RegexOption.IGNORE_CASE,
)

/** Left or right as a direction ("to the right", "on the left side"), not "is that right?". `coach._SIDEWAYS`. */
internal val SIDEWAYS = Regex(
    "\\b(?:to|on|toward|towards) (?:the|my|his|her|their) (?:left|right)\\b" +
        "|\\b(?:left|right)(?:wards?|-hand| side| wing)\\b",
    RegexOption.IGNORE_CASE,
)

/** A square named in the player's words: `re.search(r"\b[a-h][1-8]\b", text)` in `Coach.ask`. */
private val NAMED_SQUARE = Regex("\\b[a-h][1-8]\\b")

/** The unlock the user's words earn, if any. */
fun unlockFor(text: String, board: Board, candidates: List<String>): Unlock? {
    if (ASK_LINE.containsMatchIn(text)) return Unlock.ASK_LINE
    if (ASK_MOVE.containsMatchIn(text) || ASK_OTHER_MOVES.containsMatchIn(text)) return Unlock.ASK_MOVE
    if (HINT.containsMatchIn(NOT_STUCK.replace(text, " "))) return Unlock.HINT
    for (token in SAN.findAll(text).map { it.groupValues[1] }) {
        val move = board.parseSan(token) ?: continue
        if (board.san(move) in candidates) return Unlock.USER_NAMED_MOVE
    }
    return null
}

/**
 * A label as said of a move: "a mistake", "an inaccuracy", "a miss"; the good ones as they are
 * ("best", "excellent"), and theory as "a book move". `coach._rated`.
 */
internal fun rated(label: String): String {
    if (label in listOf("inaccuracy", "mistake", "miss", "blunder")) {
        return (if (label[0] in "aeiou") "an " else "a ") + label
    }
    return if (label == "book") "a book move" else label
}

/** A SAN move described in words, or the SAN itself if the board cannot play it. */
internal fun inWords(board: Board, san: String): String {
    val move = board.parseSan(san) ?: return san
    return describeMove(board, move)
}

/** What the record's question-answerer hands back; the profile stage owns it. */
/**
 * What `profile.ask.answer` gives back: the answer in prose, the queries it ran (Ask.Exchange),
 * how many model calls it took, and whether the model never wrote prose (the text is then the
 * coach's own fallback from the last rows).
 */
class ProfileAnswer(
    val text: String,
    val exchanges: List<Any?> = emptyList(),
    val calls: Int = 0,
    val failed: Boolean = false,
)

/** What `claims.verifyClaims` gives the coach: one checked sentence per claim. */
fun interface ClaimsCheck {
    fun check(text: String, board: Board, facts: PositionFacts?): List<String>
}

/** The coach in a game: takes events and the user's words, hands back messages. */
class Coach(
    provider: LLMProvider,
    policy: InterjectionPolicy? = null,
    draftCheck: DraftCheck? = null,
    var claims: ClaimsCheck? = null,
    verifier: LLMProvider? = null,
    var engine: EngineLike? = null,
    /**
     * Dialogue compaction: every this-many plies the dialogue older than the last
     * this-many plies is folded into a summary by [summariser] (the coach's own model when
     * null). 0 is off.
     */
    var summariseEvery: Int = 0,
    summariser: LLMProvider? = null,
    compactRecord: Boolean = false,
    routineProvider: LLMProvider? = null,
    /** One reply instead of a verbalized distribution. */
    var verbalizedSampling: Boolean = true,
    /**
     * The word on a sound move written from the board ([verdict]) instead of asked of the
     * model: free, immediate, true by construction (Tai, 2026-09-20).
     */
    var localVerdicts: Boolean = true,
    persona: String? = null,
    rating: Pair<Int, Int>? = null,
    /**
     * The lesson on a mistake, held until the position the move left is analysed (or
     * [LESSON_WAIT_S] passes, or the game moves on): the app waits, the tests do not.
     */
    var lessonWaitS: Double = 0.0,
    /** How a bag is shuffled when refilled ([Bags]); injected so a test can fix the draw. */
    shuffle: (MutableList<String>) -> Unit = { it.shuffle() },
    sample: (List<Pair<Double, String>>) -> String = ::sampleReply,
) {
    val policy: InterjectionPolicy = policy ?: InterjectionPolicy()
    val speaker = Speaker(
        provider,
        draftCheck = draftCheck,
        verifier = verifier,
        routineProvider = routineProvider,
        sample = sample,
    )

    /**
     * The panel's voice, when it wants replies as they are written: a factory giving, for a
     * trigger and a ply, the sink the reply's sentences go to, or null.
     */
    var streamSink: ((Trigger, Int?) -> ((String) -> Boolean)?)? = null

    var summariser: LLMProvider = summariser ?: provider

    /** The system message, and the templates' register for the player: set by [reword]. */
    var system: String = ""

    /** A player below 1400 hears no term of art in the templates ([JARGON]). */
    var plainWords: Boolean = false

    /**
     * A player of 1800 and up hears no textbook rule: the strong simulated students listed about
     * 4.5 a session they already knew ("rooks belong on open files", 75 times).
     */
    var strongPlayer: Boolean = false

    init {
        reword(persona, rating)
    }

    /**
     * Run the claims check on the caller's thread (replay and evaluation) instead of a
     * worker: live it is threaded so a button press never stalls the pipeline.
     */
    var syncClaims = false

    /**
     * How a question's work is started off the caller's thread: a daemon thread, named. A test
     * hands its own, and runs the work when it chooses. `Coach.spawn`.
     */
    var spawn: (work: () -> Unit, name: String) -> Unit = { work, name ->
        Thread({ work() }, name).apply { isDaemon = true }.start()
    }

    /**
     * The same for the dialogue summary: live it folds in the background, a few seconds
     * after the move; in evaluation it folds before the next moment, so runs repeat.
     */
    var syncSummaries = false

    var gameId: String? = null
    var userSide: Pieces.Side = Pieces.Side.WHITE

    /** The game is over: everything may be said. */
    var finished = false
    var board: Board = Board()
    var positions: MutableMap<Int, CoachPosition> = LinkedHashMap()
    var moves: MutableMap<Int, MovePlayed> = LinkedHashMap()

    /** When each move of [moves] arrived (the loop's clock). `Coach._moved_at`. */
    private var movedAt: MutableMap<Int, Double> = LinkedHashMap()

    /** The game as the model sees it: one message per move, the coach's turns between. */
    val conversation = Conversation().apply { this.compactRecord = compactRecord }

    val spoken = ArrayList<Spoken>()
    private var current: CoachPosition? = null

    /** The held lesson ([lessonWaitS]): when it was held, the position, the verdict, the move, elicit. */
    private class HeldLesson(
        val ts: Double,
        val position: CoachPosition,
        val assessment: MoveAssessment,
        val played: MovePlayed,
        val elicit: Boolean,
    )

    private var heldLesson: HeldLesson? = null
    private var clockPeakMs: Int? = null

    /** The conversation is read from the caller's thread and from a claims check. */
    /** The coach's own lock; the panel reads the conversation under it. */
    internal val lock = Any()

    /** Every wording drawn like a marble from a bag, each once before any comes again. */
    internal val bags = Bags(shuffle)

    /** The kinds of lesson the last sound moves' words taught ([lesson]), [TAUGHT_MEMORY] at most. */
    private val taught = ArrayDeque<String>()

    /**
     * The coach's messages so far this session, and the rules of thumb said, each with the count
     * it was said at: kept across games and puzzles ([RULE_COOLDOWN]).
     */
    internal var messages = 0
    private val rules = HashMap<String, Int>()
    private var lastRule: Int? = null

    /**
     * What the game is (GameStarted.kind): a game from the start, or a position -- a puzzle,
     * an analysis board -- where the mating move gets no word and the end no summary (Tai,
     * 2026-09-20: a solved puzzle is left, not summed up); every other move gets its word or
     * its lesson as in a game.
     */
    var kind = "game"

    /** The user's last few think times, for the game's pace (see [FAST_PACE_S]). */
    private val thinkS = ArrayDeque<Double>()

    /**
     * The player model as the profile stage last reported it, and the review of the game
     * just ended, for the coach's summary.
     */
    var profile: ProfileSnapshot? = null
    internal var reviewLines: List<String> = emptyList()

    /**
     * The causes of the session's mistakes, from the player model: the pattern is said when
     * one comes back ([onEvidence]). Counted in the order first seen, as a Counter keeps them.
     */
    private val causesSeen = LinkedHashMap<String, Int>()

    /** The causes whose pattern has been said: once a session. */
    private val patternTold = HashSet<String>()

    /**
     * The record's question-answerer, set by whoever owns the store; the answers it gave,
     * with their queries, for the panel's log.
     */
    var profileAsk: ((String) -> ProfileAnswer)? = null
    val profileAnswers = ArrayList<Pair<String, Any?>>()

    /**
     * The prompts that failed their check before going to the model ([harnessProblem]), for the
     * panel's log: each one is a bug in how the prompt is built. `Coach.harness_problems`.
     */
    val harnessProblems: MutableList<String> = java.util.Collections.synchronizedList(ArrayList())

    /** The player's mistakes this game, for [ELICIT_EVERY]. */
    internal var mistakes = 0

    /** When the coach last asked what the player saw, and whether they answered. */
    private var elicitAsked: Double? = null
    private var elicitAnswered = false

    /**
     * A question asked before the engine had looked at the position on the board: it is
     * answered the moment the analysis lands (see [ask]).
     */
    private var pendingAsks = ArrayList<Pair<String, Double>>()

    /**
     * The moment of the game: a new game, a takeback and a re-read each start another. A reply
     * asked for in an earlier one and landing in this one is not a turn of this one: recorded
     * where it landed, the lesson on a withdrawn move stood under "Said here" of the position
     * taken back to, and a reply about the last puzzle under the next one's, where they made the
     * player's next words a follow-up or the answer to their question (the coach review,
     * 2026-10-01). `Coach._epoch`.
     */
    internal var epoch = 0
        private set

    // Lifecycle

    fun start() = speaker.start()

    fun close() = speaker.close()

    /**
     * The screen is no longer read (the panel's Stop; the coach outlives it): a lesson held for
     * its analysis is said now, as at a game's end, and nothing more is said unasked until the
     * board is read again -- the next Start's new game or re-read. Kept on the position it last
     * read, the coach gave stuck hints about the last session's board on the next Start's first
     * ticks (the game-state review, 2026-10-01). Asked before then, it answers from the last
     * position it read. `Coach.pause`.
     */
    fun pause(ts: Double) {
        releaseLesson(ts)
        current = null
        policy.pause()
    }

    val userColour: Boolean get() = userSide.color

    val userToMove: Boolean get() = board.turn == userColour

    // Events in

    fun onEvent(event: Event, ts: Double) {
        when (event) {
            is GameStarted -> newGame(event, ts)
            is MovePlayed -> onMove(event, ts)
            is Analysis -> onAnalysis(event, ts)
            is MoveAssessment -> onAssessment(event, ts)
            is ClockObservation -> onClock(event, ts)
            is Takeback -> {
                heldLesson = null // the move it was on is gone
                onTakeback(event, ts)
            }
            is Resync -> {
                heldLesson = null
                onResync(event, ts)
            }
            is GameEnded -> {
                releaseLesson(ts)
                current = null
                finished = true
                synchronized(lock) { conversation.ended(event) }
                // The questions held for the engine's look, answered now: nothing is withheld from
                // a game that is over, and left waiting they were never answered.
                val asks = pendingAsks
                pendingAsks = ArrayList()
                for ((held, _) in asks) ask(held, ts)
            }

            is Recommendation -> onRecommendation(event, ts)
            is PuzzleOutcome -> onPuzzleOutcome(event, ts)
            is GameReview -> onReview(event, ts)
            is ProfileEvidence -> onEvidence(event, ts)
            is ProfileSnapshot -> profile = event
            is ScreenStateChanged -> {
                val covered = event.state == ScreenState.COVERED
                if (covered && !policy.covered) synchronized(lock) { conversation.covered() }
                policy.covered = covered
            }

            else -> Unit
        }
    }

    /**
     * Nothing unasked -- no word on a move, no nudge, no hint on its own -- the policy's
     * quiet mode, turned on while a puzzle is worked on the panel's board (Tai: "no auto
     * hint or anything from the coach"). A question of the player's is answered as ever.
     */
    var quiet: Boolean
        get() = policy.settings.quiet
        set(value) {
            policy.settings = policy.settings.copy(quiet = value)
        }

    /**
     * Time passing: a held lesson once its wait is up; a hint if the user has been stuck long
     * enough.
     */
    fun tick(ts: Double) {
        val held = heldLesson
        if (held != null && ts - held.ts >= lessonWaitS) releaseLesson(ts)
        val trigger = policy.tick(ts) ?: return
        val position = current ?: return
        if (position.facts == null) return
        if (position.board.isGameOver()) return // nothing to be stuck on: the game is over on the board
        // A hint unasked only ever climbs: at the top of its ladder it is not said again.
        position.disclosure.carry(ladder(position))
        val level = position.disclosure.apply(Unlock.STUCK)
        if (level == null) {
            policy.exhausted(position.ply)
            return
        }
        policy.said(position.ply, ts)
        speak(trigger, position, level, ts)
    }

    /**
     * The user said something. Always answered, at the level their words earn.
     *
     * Returns at once. A canned ask -- a hint, the move, the line -- gets the template for
     * its level immediately, with its marks, so the answer is on the board before the model
     * has started; the model's fuller reply follows. Words that may carry a claim about the
     * board are checked first, on a worker thread: the check is a model call of its own, and
     * run on the caller's thread it stalled the whole pipeline for seconds (the first live
     * games: every button press froze the vision until it returned).
     */
    fun ask(text: String, ts: Double) {
        // The moment it was asked in: the reply is that moment's, whatever the game does while a
        // worker checks the words and the model writes (a new game starting then filed the
        // answer under the next game).
        val moment = epoch to gameId
        if (profileAsk != null && ASK_PROFILE.containsMatchIn(text)) {
            askProfile(text, ts, moment)
            return
        }
        val elicitAt = elicitAsked
        if (elicitAt != null && ts - elicitAt <= ELICIT_ANSWER_S) elicitAnswered = true
        var position = current
        if (position != null && position.facts == null &&
            (userToMove || waitsForTheBoard(text)) && !finished &&
            !position.board.isGameOver() // a finished board has nothing to read
        ) {
            // The engine is still looking at the position on the board: said so, and answered
            // the moment its analysis lands -- never from an older position's (Tai, 2026-09-24:
            // a hint pressed as a puzzle appeared got "I have not seen enough of this game yet
            // to say. Give me a move or two.").
            if (pendingAsks.isEmpty()) {
                queuePlain(
                    Trigger.USER_ASK,
                    "One moment -- still looking at this position.",
                    ts,
                    turn = false,
                )
            }
            pendingAsks.add(text to ts) // each is answered, in the order asked
            return
        }
        if (position == null || position.facts == null) position = latestWithFacts()
        if (position == null || position.facts == null) {
            queuePlain(
                Trigger.USER_ASK,
                "I have not seen a position to talk about yet.",
                ts,
            )
            return
        }
        val unlock = unlockFor(text, position.board, position.facts!!.candidates)
        if (position === current) {
            // A hint asked for climbs from what the combination has had already -- in the
            // position the player faces only: climbed in the one they had moved from, a hint
            // asked before the new one was read made the combination's next hint skip a rung
            // (the coach review, 2026-10-01).
            position.disclosure.carry(ladder(position))
            if (unlock != null) position.disclosure.apply(unlock)
            // Answered here, the stuck spell starts again ([InterjectionPolicy.asked]).
            policy.asked(ts)
        }
        var level = maxOf(position.disclosure.level, DisclosureLevel.IDEA)
        if (finished || position.board.isGameOver()) {
            // Nothing is withheld from a game that is over.
            level = DisclosureLevel.LINE
        }
        synchronized(lock) { conversation.userSaid(text) }
        val own = askedOwnMove(text)
        if (own != null) {
            // About the move they already played: spoken from the position it was played from,
            // with everything shown, as the lesson after a mistake is. That position is past;
            // the level in force binds only the one they face now.
            val before = positions[own.move.ply - 1]
            if (before?.facts != null) {
                position = before
                level = DisclosureLevel.LINE
            }
        }
        val where = position
        val facts = where.facts!!
        // With a model, every question is the model's to answer, the buttons' included: its
        // answer starts within a second or two (Tai, 2026-09-28: "I'm using AI coach so I better
        // be able to ask whatever I want and get answers for them ... it's already almost
        // instant"). Asked "what should I play against this opening? give me a line", the
        // template for the opening's name was all he got. The templates are the offline coach's,
        // and the model's fallback when a reply gets something wrong.
        val offline = this.offline
        // Offline, the answer at once: the template for a bare "hint", the move or the line.
        // The move or the line asked for, or asked about where it is already open: "what
        // happens after the best move?" at the line level is the line, though it unlocks
        // nothing (the harness check of the test audit, 2026-09-26).
        val exact = unlock == Unlock.ASK_MOVE || unlock == Unlock.ASK_LINE || (
            (level.value >= DisclosureLevel.LINE.value && ASK_LINE.containsMatchIn(text)) ||
                (level.value >= DisclosureLevel.MOVE.value && ASK_MOVE.containsMatchIn(text))
            )
        // A claim to check needs something named: "what happens after the best move?" names
        // no move or square, whatever its words.
        val named = mayCarryClaims(text) && (
            SAN.containsMatchIn(text) || NAMED_SQUARE.containsMatchIn(text)
            )
        val canned = offline && (
            (exact && !named) || (
                (
                    ASK_HINT.matchesAt(text, 0) ||
                        // A question about the plan: the first hint is the lesson it asks for (the
                        // stakes, the key fact, the principle), as for a bare hint.
                        ASK_AIM.containsMatchIn(text)
                    ) && !mayCarryClaims(text)
                )
            )
        if (canned) {
            val answer = instruction(
                Trigger.USER_ASK, where, level, text, emptyList(), null, null
            ).second
            takeLeads(where, answer)
            queuePlain(
                Trigger.USER_ASK,
                answer,
                ts,
                ply = where.ply,
                level = level,
                annotations = annotationsFor(facts, level, where.board, trigger = Trigger.USER_ASK),
                fen = where.board.fen(),
                now = true,
                recordNow = true,
            )
            return
        }
        if (offline && factQuestion(text)) {
            // The material, the opening, the standing: said by the template, exactly.
            val answer = instruction(
                Trigger.USER_ASK, where, level, text, emptyList(), null, null
            ).second
            queuePlain(
                Trigger.USER_ASK,
                answer,
                ts,
                ply = where.ply,
                level = level,
                fen = where.board.fen(),
                now = true,
                recordNow = true,
            )
            return
        }

        val asked = questionPart(text)
        var shownExamination = false

        /**
         * The engine on what they asked about: the model's to answer from, given with the
         * request. Offline its plain-words form is the answer, said as the coach's turn (correct
         * by construction): only what a question names -- named in a statement ("I see the fork
         * now: Nf6 attacks the queen on e8 and the rook on d7") or before the question ("my rook
         * on e1 stays active. Why is 12...h5 their best reply?"), the pieces and squares came
         * back as the whole answer and the question went unanswered (the simulated students,
         * 2026-09-26).
         */
        fun examined(): List<String> {
            val found = namedAnalysis(asked.ifEmpty { text }, where, level)
            if (!offline) return found // the model answers, from these
            // A move named in their words that is no legal move is the model's to
            // know, not the player's to hear: "after pass, Kd1 is not a legal move" was read
            // out as the answer to "How does Qc7+ prevent Re8?" (the students).
            var said = found.filter { "is not a legal move" !in it }
            // A move they propose in a statement ("I'll play Qd1, blocking the check") is
            // examined aloud as a question about it would be: that is the answer. Without it the
            // templates said the hint that had asked for it again, word for word, three times
            // (the students' round 5); pieces and squares named in passing stay unsaid.
            if (asked.isEmpty()) said = said.filter { "the expected score is" in it }
            if (said.isNotEmpty()) {
                shownExamination = true
                queuePlain(
                    Trigger.USER_ASK,
                    said.joinToString(" ") { spoken(it) },
                    ts,
                    ply = where.ply,
                    level = level,
                    fen = where.board.fen(),
                    now = true,
                    recordNow = true,
                    moment = moment,
                )
            }
            return found
        }

        val checker = claims
        // With the engine at the model's call its claims are the model's to check, with the whole
        // conversation in view: checked beforehand against the board as it stands, "Rxf1+ check
        // the white king" -- the move just played -- came back "false, Rxf1+ is not a move you can
        // play next" (Tai's puzzle, 2026-09-28).
        val tools = engine != null && !offline
        if (checker == null || tools || !mayCarryClaims(text)) {
            fun answered() {
                val verdicts = if (mayCarryClaims(text)) examined() else emptyList()
                if (!shownExamination) { // offline, the engine's word on it was the answer
                    speak(
                        Trigger.USER_ASK, where, level, ts, userText = text, verdicts = verdicts,
                        moment = moment,
                    )
                }
            }
            if (tools && !syncClaims) {
                // The engine's searches -- the moves named, the line's end -- take up to seconds
                // each: on the caller's thread they stalled the pipeline (the deep prompt review,
                // 2026-10-01).
                spawn({ guarded("question", ts, moment) { answered() }.run() }, "sidecoach-ask")
            } else {
                answered()
            }
            return
        }

        fun checked() {
            val verdicts = try {
                checker.check(text, where.board, where.facts)
            } catch (_: Exception) {
                emptyList()
            }
            val found = examined()
            if (shownExamination) return // offline, the engine's word on it was the answer
            speak(
                Trigger.USER_ASK,
                where,
                level,
                ts,
                userText = text,
                verdicts = verdicts + found,
                moment = moment,
            )
        }

        if (syncClaims) {
            checked() // replay and evaluation: the request must exist when ask() returns
        } else {
            spawn({ guarded("claims check", ts, moment) { checked() }.run() }, "sidecoach-claims")
        }
    }

    /**
     * A question's work for a worker thread, its failure counted and logged as a failed call is,
     * and the question answered with that rather than left with nothing. `Coach._guarded`.
     */
    private fun guarded(what: String, ts: Double, moment: Pair<Int, String?>, work: () -> Unit): Runnable =
        Runnable {
            try {
                work()
            } catch (exc: Exception) {
                speaker.noteFailure(what, exc)
                queuePlain(
                    Trigger.USER_ASK,
                    "I could not work that out just now; ask me again in a moment.",
                    ts,
                    recordNow = true,
                    moment = moment,
                )
            }
        }

    /**
     * A question about the player over their games: answered from the record by the
     * profile's own model loop, off this thread (several model calls), and spoken as the
     * coach's turn when it lands. The queries it ran are kept for the log.
     */
    private fun askProfile(text: String, ts: Double, moment: Pair<Int, String?>) {
        val ask = profileAsk!!
        synchronized(lock) { conversation.userSaid(text) }

        fun run() {
            val answer = try {
                ask(text)
            } catch (exc: Exception) {
                // The record must not take the coach down.
                queuePlain(
                    Trigger.USER_ASK,
                    "I could not read the record just now; ask me again in a moment.",
                    ts,
                    recordNow = true,
                    moment = moment,
                )
                profileAnswers.add("error" to exc.toString())
                return
            }
            synchronized(lock) {
                val n = answer.exchanges.size
                conversation.note(
                    "The coach looked the question up in the record ($n " +
                        (if (n == 1) "query" else "queries") + ")."
                )
            }
            profileAnswers.add("answer" to answer)
            queuePlain(Trigger.USER_ASK, answer.text, ts, recordNow = true, moment = moment)
        }

        spawn({ guarded("record question", ts, moment) { run() }.run() }, "profile-ask")
    }

    fun hint(ts: Double) = ask("hint", ts)

    /**
     * On the opponent's move, the position on the board not yet read: whether the words wait for
     * it. Their threat is read from it: answered from the position before the player's move,
     * "they threaten Qxf2#" was said of the threat the move had just stopped. So is anything but
     * the player's last move once its verdict is in and the position it was played from is read:
     * answered from an older position, the coach said "Give me a moment; I am still reading this
     * position." with nothing waiting (the coach review, 2026-10-01).
     * `Coach._waits_for_the_board`.
     */
    private fun waitsForTheBoard(text: String): Boolean {
        if (ASK_THREAT.containsMatchIn(text)) return true
        val own = lastOwnMove() ?: return true
        val faced = positions[own.move.ply - 1]
        return faced?.facts == null
    }

    /**
     * The question held for the engine's look, answered at the position the player faces now
     * once the coach has read it: wherever the board went meanwhile -- a takeback, a re-read, the
     * next game -- "One moment" was a promise (the coach review, 2026-10-01).
     * `Coach._answer_pending`.
     */
    private fun answerPending(ts: Double) {
        if (pendingAsks.isEmpty() || current?.facts == null) return
        val asks = pendingAsks
        pendingAsks = ArrayList()
        for ((held, _) in asks) ask(held, ts)
    }

    fun poll(): List<CoachMessage> {
        val out = ArrayList<CoachMessage>()
        for (one in speaker.poll()) {
            if (ofAMomentGone(one)) {
                // The lesson or the word on a move the board no longer has -- taken back, the
                // game over, the board re-read: never shown or spoken, and the question it was to
                // end with was never asked.
                withdrawn.addLast(one)
                while (withdrawn.size > SPOKEN_PROMPTS_KEPT) withdrawn.removeFirst()
                if (one.elicit) {
                    elicitAsked = null
                    elicitAnswered = false
                }
                policy.lastReason =
                    "not said: the ${one.message.trigger.slug} on ply ${one.message.ply} was " +
                        "about a position the board no longer has"
                continue
            }
            spoken.add(one)
            lighten()
            if (one.message.text.isEmpty()) {
                // A failed model reply with nothing to fall back on: kept for the log (its
                // calls say why), said to no one.
                continue
            }
            if (!one.recorded && one.epoch == epoch) {
                synchronized(lock) {
                    conversation.coachSaid(
                        one.message.text,
                        nod = one.message.trigger == Trigger.POST_MOVE,
                    )
                }
            }
            messages += 1
            out.add(one.message)
        }
        return out
    }

    // Internals

    /**
     * What was dropped unsaid for its moment being gone ([ofAMomentGone]), for the log.
     * `Coach.withdrawn`.
     */
    val withdrawn = ArrayDeque<Spoken>()

    /**
     * Whether a lesson or a word on a move was written for a moment that is over: a takeback, a
     * re-read or a new game came after the request. Only what is about a position by its ply: an
     * answer to the player's question, the review and the practice are theirs in whatever moment
     * they land. `Coach._of_a_moment_gone`.
     */
    private fun ofAMomentGone(one: Spoken): Boolean =
        one.epoch != epoch &&
            (one.message.trigger == Trigger.POST_MOVE || one.message.trigger == Trigger.POST_MISTAKE) &&
            one.message.ply != null

    /**
     * What the log keeps of the replies said long ago: their words and checks, not the prompts
     * (each the whole game's harness, up to twice a reply, for as long as the session ran).
     * `Coach._lighten`.
     */
    private fun lighten() {
        val old = spoken.size - SPOKEN_PROMPTS_KEPT - 1
        if (old >= 0) spoken[old] = spoken[old].withoutPrompts()
    }

    /**
     * The system message and the templates' register for the player: set at the start and
     * again when the persona or the rating is saved (the register was kept until the coach was
     * rebuilt: the parity audit, 2026-09-26). `Coach.reword`.
     */
    fun reword(persona: String?, rating: Pair<Int, Int>?) {
        system = systemMessage(persona = persona, rating = rating)
        // A player below 1400 hears no term of art in the templates ([JARGON]).
        plainWords = rating != null && rating.second < PLAIN_BELOW
        // A player of 1800 and up hears no textbook rule: the strong simulated students listed
        // about 4.5 a session they already knew ("rooks belong on open files", 75 times).
        strongPlayer = rating != null && rating.first >= STRONG_FROM
    }

    private fun newGame(event: GameStarted, ts: Double) {
        epoch += 1
        // A lesson held for its analysis is the last game's: kept, the next tick said it in this
        // one, over this one's first position (the coach review, 2026-10-01).
        heldLesson = null
        gameId = event.gameId
        userSide = event.userSide
        kind = event.kind
        board = Board(event.startFen)
        positions = linkedMapOf(0 to CoachPosition(ply = 0, board = board.copy()))
        moves = LinkedHashMap()
        movedAt = LinkedHashMap()
        synchronized(lock) { conversation.newGame(event, profile = profileLines()) }
        thinkS.clear()
        mistakes = 0
        // The review is the game's own: a new game answered "how did I do overall?" with the
        // last game's sentences when the model failed (the parity audit, 2026-09-26).
        reviewLines = emptyList()
        elicitAsked = null
        elicitAnswered = false
        // A question held for the engine's look stays: it is answered at this game's first
        // position once read ([answerPending]), as a hint pressed as the next puzzle appears.
        finished = false
        current = positions[0]
        policy.restart()
        policy.newPosition(0, ts, userToMove = userToMove)
        clockPeakMs = null
        policy.setRemaining(null)
        policy.timeControl = TimeControl.UNTIMED
    }

    /**
     * What the player model says, for the setting of a new game: the insights, and the last
     * session in a sentence.
     */
    private fun profileLines(): List<String> {
        val snapshot = profile ?: return emptyList()
        val lines = ArrayList(snapshot.insights)
        val session = snapshot.lastSession
        if (!session.isNullOrEmpty()) lines.add("Last session: $session")
        return lines
    }

    /**
     * The game is over and reviewed: the review joins the conversation and the coach sums
     * the game up from it -- through the model when a position with facts is there to check
     * the reply against, else the review's own sentences.
     */
    private fun onReview(event: GameReview, ts: Double) {
        if (event.gameId != gameId) return
        var lines = event.lines.toList()
        if (strongPlayer) {
            // The pattern and its count, not the textbook habit: a 2000 player was told "bring
            // your King to the middle and push your passed pawns" (the AI's sessions).
            lines = lines.map { it.replace(HABIT_SAID, "") }
        }
        synchronized(lock) { conversation.review(lines) }
        val position = latestWithFacts()
        if (position == null || lines.isEmpty()) {
            queuePlain(Trigger.REVIEW, lines.take(4).joinToString(" "), ts, recordNow = true)
            return
        }
        reviewLines = lines
        speak(Trigger.REVIEW, position, DisclosureLevel.LINE, ts)
    }

    /**
     * A mistake of the player's placed by the player model. The third time one cause is behind
     * the session's mistakes, the pattern is said once, with the habit that answers it, right
     * after the lesson on the move; the game's review sums it up after that. Each lesson had
     * taught its own moment and a principle of its own, and the threat-blind student's notebook
     * held five unrelated rules, none of them the habit (the simulated students, 2026-09-26);
     * said the second time and then in every game, "this keeps happening -- 5 times now" came on
     * top of a lesson (Tai's phone, 2026-09-27). A player of 1800 and up hears the pattern
     * without the textbook habit. Said from the model's cause, no model call.
     */
    /**
     * Nothing is said unasked: the policy's quiet mode, or bullet, where the pattern and the
     * practice said after the game's moves came regardless. `Coach._unprompted_off`.
     */
    private fun unpromptedOff(): Boolean = quiet || policy.timeControl == TimeControl.BULLET

    private fun onEvidence(event: ProfileEvidence, ts: Double) {
        val cause = event.cause
        if (event.gameId != gameId || cause == null || !Causes.HABIT.containsKey(cause)) return
        val n = (causesSeen[cause] ?: 0) + 1
        causesSeen[cause] = n
        if (n < 3 || cause in patternTold) return
        if (unpromptedOff()) return // nothing unasked: it comes the next time the cause does
        patternTold.add(cause)
        val (words, habit) = Causes.HABIT.getValue(cause)
        var text = "That is three times now: $words."
        if (!strongPlayer) text += " The habit to build: $habit."
        if (!ruleOk(null)) return // a rule of thumb just said: the review carries the pattern
        ruleSaid("habit:$cause")
        synchronized(lock) { conversation.note("The pattern, told to the player: $text") }
        queuePlain(Trigger.POST_MISTAKE, text, ts, recordNow = true)
    }

    /**
     * The practice the profile picked after the review: a line naming the motif and the
     * puzzles, on the card; no model call, the card carries the links.
     */
    private fun onRecommendation(event: Recommendation, ts: Double) {
        if (event.gameId != gameId || event.puzzles.isEmpty() || unpromptedOff()) return
        val n = event.puzzles.size
        val text = "For practice: $n ${event.motifWords} puzzle${if (n != 1) "s" else ""} around " +
            "${event.level} ${if (n != 1) "are" else "is"} on the review card -- ${event.reason}."
        synchronized(lock) { conversation.note("Practice recommended: $text") }
        queuePlain(Trigger.REVIEW, text, ts, recordNow = true)
    }

    /**
     * A puzzle from the store worked through on the screen: a word on a fail (the site says
     * nothing useful), nothing on a solve (the site does).
     */
    private fun onPuzzleOutcome(event: PuzzleOutcome, ts: Double) {
        synchronized(lock) {
            conversation.note(
                "Puzzle ${event.puzzleId} ${if (event.solved) "solved" else "failed"}; " +
                    "puzzle level now ${event.level}."
            )
        }
        if (event.solved || quiet) return // on the panel's own board the bar says it
        val motif = if (!event.motif.isNullOrEmpty()) {
            " That one was a ${insightWords(event.motif)} puzzle."
        } else {
            ""
        }
        queuePlain(
            Trigger.POST_MISTAKE,
            "Not that one -- have another look before the site shows you.$motif",
            ts,
            recordNow = true,
        )
    }

    /**
     * The screen's clock, when one is drawn. It scales the interjection cadence: the
     * stuck-hint threshold shrinks as the user's clock runs down, and the time control
     * decides which unprompted triggers fire at all (rapid gets everything, blitz only brief
     * feedback, bullet nothing). A missing, hidden, or unreadable clock never reaches here,
     * so the policy simply keeps its untimed defaults.
     */
    private fun onClock(event: ClockObservation, ts: Double) {
        val userMs = if (userSide == Pieces.Side.WHITE) event.whiteMs else event.blackMs
        if (userMs != null) policy.setRemaining(userMs)
        synchronized(lock) { conversation.clock(event.whiteMs, event.blackMs) }
        val seen = listOfNotNull(event.whiteMs, event.blackMs)
        if (seen.isEmpty()) return
        val peak = seen.max()
        clockPeakMs = clockPeakMs?.let { maxOf(it, peak) } ?: peak
        // The clock starts at its highest, so the peak seen approximates the initial time and
        // thus the control's band; increment slack does not cross the coarse bands.
        policy.timeControl = timeControlOf(clockPeakMs)
    }

    /**
     * The user's recent think time in this game, in seconds: the median of its last few
     * moves, or null before there are two to go by.
     */
    val paceS: Double?
        get() {
            if (thinkS.size < 2) return null
            val sorted = thinkS.sorted()
            val n = sorted.size
            return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }

    /**
     * Whether the game is being played faster than the coach can keep up with a model and a
     * voice: templates and text until the player slows down.
     */
    val fastPace: Boolean
        get() {
            val pace = paceS
            return pace != null && pace < FAST_PACE_S
        }

    /**
     * The pace is fast and the player has not slowed down in this position: once they have
     * been thinking longer than the fast pace, a hint or a lesson may take the model's time
     * and the voice again.
     */
    fun fastNow(ts: Double): Boolean {
        if (!fastPace) return false
        val since = policy.positionSince
        return !(policy.userToMove && since != null && ts - since >= FAST_PACE_S)
    }

    /** The held lesson on a mistake, spoken now. `coach._release_lesson`. */
    private fun releaseLesson(ts: Double) {
        val held = heldLesson ?: return
        heldLesson = null
        speak(
            Trigger.POST_MISTAKE,
            held.position,
            DisclosureLevel.LINE,
            ts,
            assessment = held.assessment,
            played = held.played,
            elicit = held.elicit,
        )
    }

    private fun onMove(event: MovePlayed, ts: Double) {
        // The game moved on: a lesson still waiting on its analysis goes now, before the new
        // move is in the record.
        releaseLesson(ts)
        if (event.byUser && event.thinkMs != null) {
            thinkS.addLast(event.thinkMs / 1000.0)
            while (thinkS.size > PACE_MOVES) thinkS.removeFirst()
        }
        if (event.byUser) pendingAsks = ArrayList() // they played on: the questions are answered by the move
        board = boardAfter(board, event)
        moves[event.ply] = event
        movedAt[event.ply] = ts
        synchronized(lock) { conversation.move(event) }
        val position =
            CoachPosition(ply = event.ply, board = board.copy(), lastMove = event.san)
        positions[event.ply] = position
        // Kept for a while: a verdict lands a few plies late when the engine is behind
        // (2026-09-22: at six, the verdicts of plies 82-126 found no position and the coach
        // said nothing, not even in the log).
        for (old in positions.keys.filter { it < event.ply - POSITIONS_KEPT }) positions.remove(old)
        current = position
        policy.newPosition(event.ply, ts, userToMove = userToMove)
        if (summariseEvery != 0 && event.ply % summariseEvery == 0) {
            if (syncSummaries) {
                runSummary()
            } else {
                Thread({ runSummary() }, "sidecoach-summary").apply { isDaemon = true }.start()
            }
        }
    }

    /**
     * Fold the earlier dialogue into the summary. The model call runs outside the lock
     * (seconds); the fold itself takes it.
     */
    private fun runSummary() {
        try {
            summarise(conversation, summariser, keep = summariseEvery, lock = lock)
        } catch (exc: Exception) {
            speaker.noteFailure("summary", exc)
        }
    }

    /**
     * Moves withdrawn: the conversation says so, and the game goes on from the position the
     * board is back at; the replacement moves arrive as ordinary moves.
     */
    private fun onTakeback(event: Takeback, ts: Double) {
        epoch += 1
        synchronized(lock) { conversation.takeback(event) }
        board = boardBack(board, event.fen)
        val active = conversation.activeRecords()
        val ply = if (active.isNotEmpty()) active.last().move.ply else 0
        for (gone in positions.keys.filter { it > ply }) positions.remove(gone)
        // The withdrawn moves go too: a late word on one was taken for the move that replaced it.
        for (gone in moves.keys.filter { it > ply }) moves.remove(gone)
        for (gone in movedAt.keys.filter { it > ply }) movedAt.remove(gone)
        var position = positions[ply]
        if (position == null) {
            position = CoachPosition(ply = ply, board = board.copy())
            positions[ply] = position
        }
        // The move played from here is gone, and with it the line its playing opened: the
        // position is the player's again at the hints it had had, stuck hints and all.
        position.disclosure.withdrawMove()
        current = position
        policy.withdraw(ply)
        policy.newPosition(ply, ts, userToMove = userToMove)
        // A question held for the engine's look is answered here, read already or once read.
        answerPending(ts)
    }

    /** The board was re-read: the position jumped; the move numbering restarts. */
    private fun onResync(event: Resync, ts: Double) {
        epoch += 1
        board = Board(event.fen)
        synchronized(lock) { conversation.resync(event, board) }
        positions = linkedMapOf(0 to CoachPosition(ply = 0, board = board.copy()))
        moves = LinkedHashMap()
        movedAt = LinkedHashMap()
        current = positions[0]
        // The pace is the position's play, as a game's own: the next puzzle in the same
        // orientation comes as a re-read, and the last one's quick finish made its first find
        // "fast", only shown (the game-state review, 2026-10-01; [FAST_PACE_S]).
        thinkS.clear()
        // A question held for the engine's look stays, answered once this position is read.
        policy.restart()
        policy.newPosition(0, ts, userToMove = userToMove)
    }

    private fun onAnalysis(event: Analysis, ts: Double) {
        if (event.gameId != gameId) return // the engine's late word on the game before this one
        val position = positions[event.ply] ?: return
        if (position.board.fen() != event.fen) return
        val had = position.analysis
        val full = had != null && !had.provisional
        if (event.provisional && full) return // the full depth is in already
        position.analysis = event
        val leads = LinkedHashMap<String, String>()
        val peek = { key: String, options: List<String> ->
            val wording = bags.peek(key, options)
            leads[key] = wording
            wording
        }
        position.facts = factsFromAnalysis(
            position.board, event, lastMove = position.lastMove, user = userColour, pick = peek
        )
        position.leads = leads
        // The record shows what the coach answers from, the early depth until the full one
        // replaces it: kept back until the full depth, a question answered from the early one
        // was asked of the model with "Engine lines: not in yet." and no facts (the coach review,
        // 2026-10-01).
        synchronized(lock) { conversation.analysed(event, position.facts!!) }
        if (event.provisional) {
            // Measured early, the full depth to follow: what the player asks is answered from it
            // now; the coach's own words -- a lesson -- wait for the full depth.
            if (position === current) answerPending(ts)
            return
        }
        if (heldLesson?.played?.ply == event.ply) releaseLesson(ts) // what the move allowed is known now
        if (position === current) answerPending(ts)
        // Nothing is said of its own at a decisive moment: while the player thinks, the first
        // word unasked is the stuck hint (Tai, 2026-10-01: nothing "until the user is stuck or
        // actively request a hint or after they have made a move").
    }

    /**
     * The highest hint the combination this position continues has had: back through the
     * player's positions while the move they played there was the engine's best and the position
     * goes on with it ([sameCombination]), the levels said before they moved (Tai, 2026-09-29:
     * "Each level can only be said once and if the level higher was said then the lower one
     * can't be said"). A move off the best line starts a new one; a puzzle's wrong moves are
     * taken back. Any run of best moves is not one combination: the hint said at the game's
     * first move kept the next fifty from a player who was stuck. `Coach._ladder`.
     */
    private fun ladder(position: CoachPosition): DisclosureLevel {
        var level = DisclosureLevel.SILENT
        var current = position
        while (true) {
            val ply = current.ply
            val own = moves[ply - 1]
            val before = positions[ply - 2]
            if (own == null || !own.byUser || before == null || !wasBest(before, own) ||
                !sameCombination(before, own, current)
            ) {
                return level
            }
            if (before.disclosure.hinted.value > level.value) level = before.disclosure.hinted
            current = before
        }
    }

    /**
     * Whether [after], where the player stands again, goes on with the combination they moved on
     * from at [before] by [own]: a forcing sequence the engine's line there continues -- their
     * move, the reply it expected, and the next move of the line the best now -- or one plan or
     * theme the facts name at both. `Coach._same_combination`.
     */
    private fun sameCombination(before: CoachPosition, own: MovePlayed, after: CoachPosition): Boolean {
        val was = before.facts ?: return false
        val now = after.facts ?: return false
        for ((first, second) in listOf(was.theme to now.theme, was.plan to now.plan)) {
            if (!first.isNullOrEmpty() && first == second) return true
        }
        val line = was.line.map { it.trimEnd('+', '#') }
        return was.nature != null && was.nature.startsWith("forcing") &&
            line.size >= 3 &&
            line[0] == own.san.trimEnd('+', '#') &&
            (after.lastMove ?: "").trimEnd('+', '#') == line[1] &&
            now.candidates.isNotEmpty() &&
            now.candidates[0].trimEnd('+', '#') == line[2]
    }

    /** Whether [move] was the engine's best from [position]. `Coach._was_best`. */
    private fun wasBest(position: CoachPosition, move: MovePlayed): Boolean {
        val analysis = position.analysis
        if (analysis != null && analysis.lines.isNotEmpty() && analysis.lines[0].moves.isNotEmpty()) {
            return analysis.lines[0].moves[0] == move.uci
        }
        val facts = position.facts ?: return false
        return facts.candidates.isNotEmpty() && facts.candidates[0] == move.san
    }

    private fun onAssessment(event: MoveAssessment, ts: Double) {
        if (event.gameId != gameId) return // the engine's late word on the game before this one
        val played = moves[event.ply]
        // The verdict on a move taken back since, not on the one played there now.
        if (played != null && played.uci != event.uci) return
        synchronized(lock) { conversation.assess(event) }
        val move = played ?: return
        if (!move.byUser) return
        val before = positions[event.ply - 1]
        if (before == null || before.facts == null) {
            // Nothing to speak from: the position is gone, or its analysis never came. Said
            // in the log, never silently.
            val why = if (before == null) "forgotten" else "not analysed"
            policy.lastReason = "no word on ply ${event.ply}: the position before it was $why"
            return
        }
        val trigger = policy.onUserMoveAssessed(event) ?: return
        var elicit = false
        if (trigger == Trigger.POST_MISTAKE) {
            mistakes += 1
            elicit = elicitDue(ts)
        }
        before.disclosure.apply(Unlock.MOVE_PLAYED)
        val after = positions[event.ply]
        if (kind != "game" && after != null && after.board.isCheckmate()) {
            return // the puzzle is solved with a mate: the site says so, and moves on
        }
        // Every move of the player's gets a word from the coach. A lecture is only for a move
        // that cost something. Theory, the engine's own move and an excellent one get a
        // sentence written from the board (a word of praise, and what the move teaches when
        // the board shows it), no model call; in blitz too. A great or brilliant move gets
        // the model's delight at its full effort; a good move the model's word on the better
        // one; the board's sentence stands in if the reply fails.
        if (trigger == Trigger.POST_MOVE && event.severity == Severity.NONE) {
            val facts = before.facts!!
            val best = facts.candidates.firstOrNull()
            val routine = best == null || best == move.san || isLocal(event.label)
            val blitz = policy.timeControl == TimeControl.BLITZ
            // Great and brilliant moves get the model's delight (Tai, 2026-09-26 18:57: "They
            // need to go through the api"), though its delight carried a false claim in 18% of
            // them on the locked test set: the draft checks stand guard over it.
            val local = routine && !isExciting(event.label) && (localVerdicts || blitz)
            if (!local) {
                speak(
                    trigger,
                    before,
                    DisclosureLevel.LINE,
                    ts,
                    assessment = event,
                    played = move,
                    candidates = if (routine) VS_ROUTINE_CANDIDATES else VS_CANDIDATES,
                    routine = routine && !isExciting(event.label),
                )
                return
            }
            val label = event.label?.name?.lowercase()
            val text = if (localVerdicts) {
                val (written, kind) = verdict(
                    before.board,
                    Move.fromUci(move.uci),
                    label,
                    event.tags,
                    phase = facts.phase,
                    threats = facts.threats,
                    taught = verdictTaught(),
                    bags = bags,
                    plain = plainWords,
                    strong = strongPlayer,
                )
                taughtByVerdict(kind)
                written
            } else {
                verdictWord(label)
            }
            val marks = annotationsFor(
                before.facts!!,
                DisclosureLevel.LINE,
                before.board,
                trigger = trigger,
                playedSan = move.san,
                severity = event.severity.name.lowercase(),
            )
            queuePlain(
                trigger,
                text,
                ts,
                ply = before.ply,
                level = DisclosureLevel.LINE,
                annotations = marks,
                fen = before.board.fen(),
            )
            return
        }
        if (trigger == Trigger.POST_MISTAKE && lessonWaitS > 0 && (after == null || after.analysis == null)) {
            heldLesson = HeldLesson(ts, before, event, move, elicit)
            return
        }
        speak(
            trigger,
            before,
            DisclosureLevel.LINE,
            ts,
            assessment = event,
            played = move,
            elicit = elicit,
        )
    }

    /**
     * Whether this mistake's lesson ends by asking what the player saw: every
     * [ELICIT_EVERY]-th mistake, and never again in a game once one such question went
     * unanswered -- the player is not talking back, and the question is noise.
     */
    internal fun elicitDue(ts: Double): Boolean {
        if (mistakes % ELICIT_EVERY != 0) return false
        if (elicitAsked != null && !elicitAnswered) return false
        elicitAsked = ts
        elicitAnswered = false
        return true
    }

    /**
     * The word of praise on a fine move, by its label, from the bag the local verdicts draw from
     * too: a game of "Good. Good. Good." wears thin.
     */
    internal fun verdictWord(label: String?): String {
        val openers = OPENERS[label ?: ""] ?: OPENERS.getValue("good")
        return bags.draw("praise:${if (label.isNullOrEmpty()) "good" else label}", openers) + "."
    }

    /**
     * A new model mid-game (the settings saved): the next call goes to it; a call in flight
     * finishes on the old one. The conversation is untouched.
     */
    fun setProviders(
        provider: LLMProvider,
        routineProvider: LLMProvider? = null,
        verifier: LLMProvider? = null,
        summariser: LLMProvider? = null,
        claims: ClaimsCheck? = null,
    ) {
        synchronized(lock) {
            speaker.replaceProviders(provider, routineProvider, verifier)
            this.summariser = summariser ?: provider
            this.claims = claims
        }
    }

    /**
     * What the coach has said in this game, as one text: a reason is not given twice in a game.
     * Remembered over six plies, "your pawn on h2 can strike at their pawn on g5" was said at
     * move 7 and again at 11 (Tai's phone, 2026-09-27); over none, 27% of the lessons came again
     * within three moves (Nova bot games, 2026-09-25). The entries are copied under the coach's
     * lock: a summary folds them on its own thread (Python's GIL makes the desktop's iteration of
     * them whole).
     */
    private fun recentlySaid(): String {
        val entries = synchronized(lock) { conversation.entries.toList() }
        return entries.filter { it.kind == "coach" }.joinToString(" ") { it.text }
    }

    /**
     * The first hint's wordings taken from their bags once the hint is said: this turn's template
     * is the position's first hint, or holds it.
     */
    private fun takeLeads(position: CoachPosition, template: String) {
        val facts = position.facts
        val situation = facts?.situation
        val principle = facts?.principle
        if (!situation.isNullOrEmpty() &&
            (situation in template || (!principle.isNullOrEmpty() && without(situation, principle) in template))
        ) {
            for ((key, wording) in position.leads) bags.take(key, wording)
            position.leads = emptyMap()
        }
    }

    /**
     * A hint's template with its principle ([PositionFacts.principle]) left out while it cools
     * down, and the word to the model that goes with it; counted as said when it stays in.
     * `Coach._cool_hint`.
     */
    internal fun coolHint(facts: PositionFacts, text: String): Pair<String, String> {
        val principle = facts.principle
        if (principle.isNullOrEmpty() || principle !in text) return text to ""
        if (ruleOk(principle)) {
            ruleSaid(principle)
            return text to ""
        }
        val note = " $RULE_HELD_MARK: leave out \"$principle\" -- the player heard it lately."
        return without(text, principle) to note
    }

    /**
     * [lesson], opening with [opener], written with its rule when one may be said, less what was
     * said lately ([fresh]), and whether the rule was held back: said within [RULE_COOLDOWN]
     * messages, or another within [RULE_SPACING]. The rule is the template's ([rule], [ruleOf])
     * or, for the model -- asked for "a pattern to remember" on every lesson, it stated one every
     * time -- the pattern it teaches ([facts]' theme or plan). A held rule is left out, and the
     * model is told ([RULE_HELD]). The rule is counted as said only when the lesson still says it
     * -- or, with a model, when the model is asked for it. [asked]: the player asked about the
     * move, and hears the reason whole, what the lesson after it said or not: trimmed of it, "why
     * was that bad?" was answered "f3 was better." and nothing more, 61 times in 95 (the
     * desktop's offline sweep of bot games, 2026-09-27). `Coach._fresh_lesson`.
     */
    private fun freshLesson(
        opener: String,
        rule: String?,
        facts: PositionFacts? = null,
        asked: Boolean = false,
        lesson: (Boolean) -> String,
    ): Pair<String, Boolean> {
        // `rule or (facts.theme if facts is not None else None) or None`
        val key = rule?.ifEmpty { null } ?: facts?.theme?.ifEmpty { null }
        val held = key == null || !ruleOk(key)
        var text = lesson(!held)
        if (!asked) text = fresh(text, recentlySaid(), opener)
        if (!held && rule != null && said(rule, text)) {
            ruleSaid(rule)
        } else if (!held && key != null && !offline) {
            ruleSaid(key)
        }
        return text to held
    }

    /** No model: every word is the templates' (llm.provider = none). `Coach._offline`. */
    private val offline: Boolean get() = speaker.provider is OfflineProvider

    /**
     * Whether a rule of thumb may be said now: none said in the last [RULE_SPACING] messages,
     * and this one (by its words, or a sound move's lesson kind) not in the last [RULE_COOLDOWN].
     * [rule] null asks only the first. `Coach._rule_ok`.
     */
    internal fun ruleOk(rule: String?): Boolean {
        val last = lastRule
        if (last != null && messages - last < RULE_SPACING) return false
        val said = if (rule != null) rules[rule] else null
        return said == null || messages - said >= RULE_COOLDOWN
    }

    /**
     * The rules of thumb said, with the message each was last said at: for the tests, which on the
     * desktop spy on `_rule_said`.
     */
    internal fun rulesSaid(): Map<String, Int> = HashMap(rules)

    private fun ruleSaid(rule: String) {
        rules[rule] = messages
        lastRule = messages
    }

    /**
     * A word on a sound move held to the rule budget: its pattern to keep asked for only while a
     * rule may be said ([ruleOk]), and then counted as said, under [key] -- with a model, being
     * asked for it is the saying (as in [freshLesson]). `Coach._budgeted`.
     */
    private fun budgeted(instruction: String, key: String): String {
        if (!ruleOk(null)) return holdRule(instruction)
        if (!offline) ruleSaid(key)
        return instruction
    }

    /**
     * What a sound move's word may not teach now: the kinds of the last few lessons
     * ([TAUGHT_MEMORY]), and every principle that may not be said ([ruleOk]).
     * `Coach._verdict_taught`.
     */
    internal fun verdictTaught(): List<String> = taught.toList() + PRINCIPLES.keys.filter { !ruleOk(it) }

    /**
     * A sound move's word taught [kind]: remembered for the next words, and counted as a rule
     * when it is a principle. `Coach._taught_by_verdict`.
     */
    internal fun taughtByVerdict(kind: String?) {
        if (kind == null) return
        taught.addLast(kind)
        if (taught.size > TAUGHT_MEMORY) taught.removeFirst()
        if (kind in PRINCIPLES) ruleSaid(kind)
    }

    /** The opening the game was in: the last name the book gave. */
    private fun openingSoFar(): String? {
        synchronized(lock) { return conversation.opening() }
    }

    /**
     * The engine's lines the record shows for [board], when it is one of the game's positions
     * still kept; none otherwise.
     */
    private fun sheetLines(board: Board?): List<Line> {
        if (board == null) return emptyList()
        val key = positionKey(board)
        synchronized(lock) {
            for (position in positions.values) {
                val analysis = position.analysis ?: continue
                if (positionKey(position.board) == key) return analysis.lines.toList()
            }
        }
        return emptyList()
    }

    /**
     * The engine's word on the moves, pieces and squares the player's words name; one line per
     * probe, none when nothing was named. A piece or square named needs no engine; a line does.
     * What else the words mean is the model's to ask the engine about ([EngineTools]).
     */
    private fun namedAnalysis(
        text: String,
        position: CoachPosition,
        level: DisclosureLevel,
    ): List<String> = try {
        analyseNamed(
            text,
            position.board,
            engine,
            userColour = userColour,
            level = level,
            // Pieces and squares as the board stands now, whatever position the question is
            // about.
            piecesBoard = board,
            // The lines the sheet already shows: a move they start is read from them.
            known = position.analysis?.lines ?: emptyList(),
        )
    } catch (exc: Exception) {
        speaker.noteFailure("named analysis", exc)
        emptyList()
    }

    /**
     * What the coach has said since the position now was reached: the last move, or a takeback or
     * a re-read of the board (what was said of a withdrawn move is not). `_said_since_move`.
     */
    private fun saidSinceMove(): List<String> {
        val entries = synchronized(lock) { conversation.entries.toList() }
        val out = ArrayList<String>()
        for (entry in entries.asReversed()) {
            if (entry.kind == "ply" || entry.kind == "takeback" || entry.kind == "resync") break
            if (entry.kind == "coach") out.add(entry.text)
        }
        return out
    }

    /**
     * Whether the player's words follow up what the coach has said since the last move, not an
     * answer to its question: the verdict and the better move were told already, and a
     * follow-up gets only what it asks.
     */
    private fun followingUp(said: String): Boolean =
        !answeringOurQuestion(said) && saidSinceMove().isNotEmpty()

    /**
     * Whether the player's words answer the question the coach just asked: its last turn ended
     * with one, nothing has happened since, and their words ask nothing.
     */
    internal fun answeringOurQuestion(said: String): Boolean {
        if (questionPart(said).isNotEmpty()) return false
        var entries = synchronized(lock) { conversation.entries.toList() }
        // Their words themselves, when ask() has recorded them already, are not the thread.
        if (entries.isNotEmpty() && entries.last().kind == "ask") entries = entries.dropLast(1)
        for (entry in entries.asReversed()) {
            if (entry.kind == "coach") return entry.text.trimEnd().endsWith("?")
            if (entry.kind != "note") return false
        }
        return false
    }

    /**
     * The player's own move, when it is the last move played and has its verdict. Once the
     * opponent has replied, "that move" is theirs -- unless the words say it is the player's
     * ([mine]): "what should I have played?", "did I blunder?".
     */
    private fun lastOwnMove(mine: Boolean = false): PlyRecord? {
        synchronized(lock) {
            val records = conversation.activeRecords()
            if (mine) {
                val own = records.lastOrNull { it.move.byUser }
                return if (own != null && own.assessment != null) own else null
            }
            val last = records.lastOrNull() ?: return null
            if (last.move.byUser && last.assessment != null) return last
        }
        return null
    }

    /**
     * The player's move a question is about, when it asks about their own last move: against a
     * bot the reply comes at once, and "what should I have played?" then found the opponent's
     * move last and got a hint for the next one (the runtime audit). `coach._asked_own_move`.
     *
     * Offline, words said while the opponent is to move -- the player's own move the last one
     * played -- are about that move too, unless they ask about the opponent's threat or move, the
     * material, the opening, the standing or the game: a simulated student's "I didn't even
     * notice they were aiming for my rook" after 9.Nf3? was answered for the side to move, "You
     * can win a knight. Their knight on f3 can simply be taken" -- the student's own knight
     * (2026-09-26). With a model that guess is the model's to make, from the record: made here,
     * "how would this position continue?" after a puzzle's last move was answered four times
     * running as a question about that move, from the position before it ("Best, and it cost
     * nothing: Rxf1+ ...", Tai's puzzle, 2026-09-28).
     */
    internal fun askedOwnMove(text: String): PlyRecord? {
        if (ASK_LAST_MOVE.containsMatchIn(text) || ASK_COST.containsMatchIn(text)) {
            // After a reply that came straight away "that move" is theirs too: they never faced
            // the position between (the opponent's move is last on the board, and the model took
            // it for the one asked about).
            return lastOwnMove(mine = MY_MOVE.containsMatchIn(text) || repliedAtOnce())
        }
        if (finished || userToMove || !offline) return null
        val elsewhere = listOf(
            ASK_THREAT, ASK_OPPONENT_MOVE, ASK_LINE, ASK_MATERIAL, ASK_OPENING, ASK_EVAL, ASK_REVIEW,
        )
        if (elsewhere.any { it.containsMatchIn(text) }) return null
        return lastOwnMove()
    }

    /**
     * Whether the last move on the board is the opponent's and came straight after the player's
     * own ([INSTANT_REPLY_S] by the loop's clock): a bot's. `Coach._replied_at_once`.
     */
    internal fun repliedAtOnce(): Boolean {
        if (moves.isEmpty()) return false
        val last = moves.keys.max()
        val reply = moves[last]
        val own = moves[last - 1]
        if (reply == null || own == null || reply.byUser || !own.byUser) return false
        val sent = movedAt[own.ply]
        val answered = movedAt[reply.ply]
        return sent != null && answered != null && answered - sent >= 0 && answered - sent <= INSTANT_REPLY_S
    }

    /** The opponent's move, when it is the last move played and has its verdict. */
    private fun lastOpponentMove(): PlyRecord? {
        synchronized(lock) {
            val records = conversation.activeRecords()
            val last = records.lastOrNull() ?: return null
            if (!last.move.byUser && last.assessment != null) return last
        }
        return null
    }

    /**
     * Who stands better, from the engine's score for the position, in plain words, and the
     * material ([material], said whole by [materialWords]; else the facts').
     */
    private fun standingTemplate(position: CoachPosition, material: String? = null): String {
        val analysis = position.analysis
        if (analysis == null || analysis.lines.isEmpty()) return "I have no evaluation for this yet."
        var score = analysis.lines[0].expectedScore
        if (position.board.turn != userColour) score = 1.0 - score
        // The Standing line's own words: its bands and this template's differed (0.8 and 0.85
        // for winning), and a reply that said what the line said was faulted.
        val words = standingWords(score)
        var said = material
        if (said.isNullOrEmpty() && position.facts != null) said = position.facts!!.material
        return "Right now $words" + (if (!said.isNullOrEmpty()) "; material: $said" else "") + "."
    }

    /**
     * The material as the game stands, to the player: now, and after the recapture the engine
     * expects when the last move was a capture -- named, unless it is the player's own move
     * before the move rung. The template said only the first half and the model only the second
     * ("you are up a bishop and a pawn for a knight", the recapture still to come: quality round
     * 3, 2026-09-25). `coach._material_words`.
     */
    private fun materialWords(level: DisclosureLevel, facts: PositionFacts): String? {
        val found = synchronized(lock) { conversation.materialNow() } ?: return facts.material
        val (now, recapture, thenWords) = found
        if (recapture == null || thenWords == null) return now
        val then = if (thenWords.startsWith("level")) "it is $thenWords" else thenWords
        if (userToMove && level.value < DisclosureLevel.MOVE.value) {
            // Both halves, the recapture unnamed: which piece takes back is the question, and
            // "once you take back" does not answer it. Said as the material now alone, the answer
            // to "am I up material?" was the vaguer and the more misleading (Tai, 2026-09-25: "a
            // vague coach is worse than no coach at all").
            return "$now -- and once you take back, $then"
        }
        val whose = if (userToMove) "your" else "their"
        return "$now -- and after $whose recapture $recapture, $then"
    }

    /**
     * The record in a sentence, for when the model's review fails the checks: the tally and
     * where the most ground was lost, from the same records the messages carry; once the
     * game is over, the review's own sentences (a lost game's "how did I do" once named
     * three late inaccuracies and not the blunder, offline).
     */
    private fun reviewTemplate(): String {
        if (reviewLines.isNotEmpty()) return reviewLines.take(4).joinToString(" ")
        val counts = LinkedHashMap<String, Int>()
        val lost = ArrayList<Triple<Double, Int, String>>()
        val judged: List<PlyRecord>
        synchronized(lock) {
            judged = conversation.judgedRecords()
                .filter { it.move.byUser && it.assessment != null }
            if (judged.isEmpty()) return "I have not seen enough of this game yet to say."
            for (r in judged) {
                val a = r.assessment!!
                val label = a.label?.name?.lowercase() ?: a.severity.name.lowercase()
                counts[label] = (counts[label] ?: 0) + 1
                if (a.severity.name.lowercase() != "none") {
                    lost.add(
                        Triple(a.expectedLoss, r.move.ply, "${conversation.numbered(r.move)} ($label)")
                    )
                }
            }
        }
        val strong = listOf("brilliant", "great", "best", "excellent")
        val good = strong.sumOf { counts[it] ?: 0 }
        // The count and the words name the same labels: "best or excellent" over a count that
        // held the great and the brilliant moves too said less than it counted.
        val named = strong.filter { (counts[it] ?: 0) > 0 }.ifEmpty { listOf("best", "excellent") }
        val labels = if (named.size == 1) named[0] else "${named.dropLast(1).joinToString(", ")} or ${named.last()}"
        var text = "$good of your ${judged.size} moves were $labels."
        // What they keep missing, from the session's causes, before the game has a review.
        val kept = causesSeen.entries
            .filter { (cause, n) -> n >= 2 && Causes.HABIT.containsKey(cause) }
            .map { (cause, n) -> n to cause }
        if (kept.isNotEmpty()) {
            // Python's max over (n, cause): the count, then the cause's name.
            val (n, cause) = kept.maxWith(compareBy<Pair<Int, String>>({ it.first }, { it.second }))
            val (words, habit) = Causes.HABIT.getValue(cause)
            text += " What you keep missing: $words, $n times."
            if (!strongPlayer) text += " The habit that fixes it: $habit."
        }
        if (lost.isNotEmpty()) {
            // The three that cost the most, told in the order they were played.
            val worst = lost.sortedByDescending { it.first }.take(3).sortedBy { it.second }
            text += " You lost the most ground at " + worst.joinToString(", ") { it.third } + "."
        } else {
            text += " Nothing cost you real ground."
        }
        return text
    }

    private fun latestWithFacts(): CoachPosition? {
        for (ply in positions.keys.sortedDescending()) {
            val position = positions[ply]
            if (position?.facts != null) return position
        }
        return null
    }

    /**
     * The directive and the template for a question about the moves besides the best: every move
     * about level with it, named, and the clearly worse ones with what goes wrong with them, from
     * the engine's lines ([worseMoves]). Asked whether a move is also good, the answer is the same
     * set and the move they name among it or not. `Coach._alternatives_answer`.
     */
    private fun alternativesAnswer(position: CoachPosition, facts: PositionFacts, said: String): Pair<String, String> {
        val fine = facts.candidates
        val analysis = position.analysis
        val worse = if (analysis != null) worseMoves(position.board, analysis, userColour) else emptyList()
        val also = ASK_ALSO.containsMatchIn(said) && !ASK_OTHER_MOVES.containsMatchIn(said)
        val levelPart = if (fine.size > 1) {
            "The moves about level with the best, all fine: ${fine.joinToString(", ")}. Say they " +
                "are about equally good, and name every one."
        } else {
            "One move stands out: ${fine[0]}. Say the others are clearly worse."
        }
        val worsePart = if (worse.isNotEmpty()) {
            " Clearly worse, and what goes wrong with them, true of the board: ${worseText(worse)}. " +
                "Say which, and why, in a few words."
        } else {
            ""
        }
        val askPart = if (also) {
            "They ask whether another move is also good: say yes plainly when the move they name " +
                "is among the fine ones, and what it does; when it is not, say it is weaker than " +
                "the best, and what goes wrong with it when the sheet says."
        } else {
            "They ask about the good moves here, not only the best one."
        }
        val directive = "$askPart $ALTERNATIVES_READ $levelPart$worsePart Say it in words. " +
            "$NO_FIGURES Two or three sentences, no question."
        var text = fineSaid(fine)
        val bare = fine.map { it.trimEnd('+', '#') }
        val named = SAN.findAll(said).map { it.groupValues[1] }.filter { it.trimEnd('+', '#') in bare }.toList()
        if (also && named.isNotEmpty()) {
            text = if (named[0].trimEnd('+', '#') == bare[0]) {
                "${named[0]} is the engine's top choice."
            } else {
                "Yes, ${named[0]} is fine too -- about as good as ${fine[0]}."
            }
            if (fine.size > 1) text += " " + fineSaid(fine)
        }
        return directive to "$text ${worseSaid(worse)}".trim()
    }

    private fun speak(
        trigger: Trigger,
        position: CoachPosition,
        level: DisclosureLevel,
        ts: Double,
        userText: String? = null,
        verdicts: List<String> = emptyList(),
        assessment: MoveAssessment? = null,
        played: MovePlayed? = null,
        candidates: Int? = null,
        routine: Boolean = false,
        elicit: Boolean = false,
        moment: Pair<Int, String?>? = null,
    ) {
        val facts = position.facts!!
        val (momentEpoch, momentGame) = moment ?: (epoch to gameId)
        var (instructionText, fallback) = instruction(
            trigger, position, level, userText, verdicts, assessment, played, elicit = elicit,
        )
        takeLeads(position, fallback)
        val mustName = mustName(trigger, position, level, fallback, assessment, played, userText)
        if (trigger == Trigger.USER_ASK && fallback.isNotEmpty() && fallback in saidSinceMove()) {
            // Nothing the coach said since the last move is said again word for word: a
            // student's answer to the hint got the same hint back, three times (round 5).
            fallback = ""
        }
        // A reply about a position already moved from -- the lesson on a move, an answer about it
        // -- is rendered at the level of the position on the board now, and must not give away
        // the answer there (a verdict can land after the opponent has replied).
        val now = current
        val past = now != null && now !== position && now.ply > position.ply
        val renderLevel = if (past && now != null) now.disclosure.level else level
        var nowAnswer: List<String> = emptyList()
        val nowFacts = now?.facts
        if (past && now != null && nowFacts != null && now.board.turn == userColour &&
            now.disclosure.level.value < DisclosureLevel.MOVE.value
        ) {
            val spokenOf = HashSet<String>()
            spokenOf.add(played?.san ?: "")
            spokenOf.addAll(playedMoves())
            if (assessment != null && assessment.best.isNotEmpty()) {
                val move = Move.fromUciOrNull(assessment.best[0])
                if (move != null) spokenOf.add(position.board.san(move))
            }
            val answer = nowFacts.candidates.take(1) + nowFacts.line
            nowAnswer = answer.distinct().filter { it !in spokenOf }
        }
        // The player faces a position again only once it is their turn: on the opponent's, what
        // the move allowed is the lesson, and "nothing about the position" forbade it.
        val facing = past && now != null && now.board.turn == userColour
        // Asked on the opponent's move, there is no move of the player's to keep back.
        val theirs = trigger == Trigger.USER_ASK && !past && !finished &&
            position.board.turn != userColour
        val line = levelLine(
            level, position, if (facing) renderLevel else null, past = past, theirs = theirs,
        )
        // A question is answered as asked: its words head the Task. Found only under "Said here",
        // far above, "whose bishop? ... he will just move away to the right, no?" got the line
        // again (Tai's puzzle, 2026-09-28).
        var asked = if (!userText.isNullOrEmpty()) "Their words, which you answer: \"$userText\"\n\n" else ""
        if (!userText.isNullOrEmpty() && SIDEWAYS.containsMatchIn(userText)) {
            // Said once in the Setting, far above, left and right were still read from White's
            // side: "from h1 there is no square to the right" (Tai's puzzle, both models). Told
            // so, the model still wrote "the only legal reply is Kg2, toward the a-file" as no
            // move to the right: the moves in question are given their way.
            val side = if (userSide == Pieces.Side.WHITE) "White" else "Black"
            asked += "They say left and right from $side's side: ${hands(userSide)}."
            val seen = SAN.findAll(userText).map { ways(position.board, listOf(it.groupValues[1]), userSide) }
                .toMutableList()
            position.facts?.let { seen.add(ways(position.board, it.line.take(2), userSide)) }
            val shown = seen.distinct().filter { it.isNotEmpty() }
            if (shown.isNotEmpty()) asked += " Seen so: ${shown.joinToString("; ")}."
            asked += "\n\n"
        }
        var directive = "$line$asked$instructionText"
        var faced = ""
        var frames: List<Board> = emptyList()
        var moved = if (trigger == Trigger.POST_MOVE || trigger == Trigger.POST_MISTAKE) played else null
        var facedAt = position // the position `moved` was played from
        if (moved == null && trigger == Trigger.USER_ASK) {
            val own = askedOwnMove(userText ?: "")
            val last = lastOwnMove()
            if (own != null && own.move.ply == position.ply + 1) {
                moved = own.move
            } else if (own == null && last != null &&
                (last.move.ply == position.ply || last.move.ply == position.ply + 1)
            ) {
                // The player's move is the last one played (its new position read, or not yet):
                // its sheet goes with any question, and the model reads which position the
                // words are about -- "why does the bishop check not work?" is about the move,
                // "how would this position continue?" is not (Tai's puzzle, 2026-09-28).
                val prior = if (last.move.ply == position.ply + 1) position else positions[last.move.ply - 1]
                if (prior?.facts != null) {
                    moved = last.move
                    facedAt = prior
                }
            }
        }
        if (moved != null) {
            // The lesson after a move is taught from the position the player faced, which the
            // record keeps as one line: its sheet goes with the request (parity pass,
            // 2026-09-25).
            val after = ArrayList(framesAfter(facedAt, moved).take(2))
            if (after.size == 1 && assessment != null && assessment.best.size > 1 &&
                assessment.best[0] == moved.uci
            ) {
                // The new position not yet read: the engine's own line names the reply.
                val reply = Move.fromUciOrNull(assessment.best[1])
                if (reply != null && after[0].isLegal(reply)) {
                    val replied = after[0].copy(withStack = false)
                    replied.push(reply)
                    after.add(replied)
                }
            }
            val sheet = synchronized(lock) {
                conversation.facedText(
                    facedAt.facts!!,
                    facedAt.board,
                    after.lastOrNull(),
                    after.firstOrNull(),
                )
            }
            if (sheet.isNotEmpty()) faced = "\n\n# The position they faced, before ${moved.san}\n$sheet"
            frames = framesAfter(facedAt, moved)
        }
        frames = frames + analysisFrames(position)
        // The earlier moments of the same idea, for a directive a link belongs in: found in the
        // record by the coach and listed with how each went and what was said, and every other
        // earlier move of the game held back from the reply ([unlistedLinks]).
        var earlier = ""
        var before: Set<String>? = null
        if (TIE_TO_THE_GAME in directive) {
            val key = position.facts!!.theme?.ifEmpty { null } ?: position.facts!!.plan
            val taught = moved?.ply ?: (position.ply + 1)
            val links: List<String>
            synchronized(lock) {
                links = if (!key.isNullOrEmpty()) conversation.sameIdea(key, taught) else emptyList()
                before = conversation.movesBefore(position.ply)
            }
            if (links.isNotEmpty()) {
                earlier = "\n\n# Earlier in this game, the same idea: $key\n" + links.joinToString("\n")
            }
            directive = directive.replace(TIE_TO_THE_GAME, if (links.isNotEmpty()) TIE_LISTED else NO_EARLIER)
        }
        // And the board before the last move played: an answer about it says what it took
        // ("15...Qxe7 took your knight on e7"), which the board after it no longer shows.
        val records = synchronized(lock) { conversation.activeRecords() }
        var lastBefore: Board? = null
        if (records.isNotEmpty()) {
            // `with suppress(ValueError)`: a record's FEN the board cannot read adds no frame.
            lastBefore = runCatching { Board(records.last().move.fenBefore) }.getOrNull()
            lastBefore?.let { frames = frames + it }
        }
        // A question of the player's: the model may ask the engine what the messages do not show,
        // from the board now or from before the last move ([EngineTools]).
        val namedEngine = engine
        val tools = if (trigger == Trigger.USER_ASK && namedEngine != null && !offline) {
            EngineTools(
                this.board,
                lastBefore,
                namedEngine,
                userColour,
                // The lines the record shows for each: a move on one is read from it.
                knownNow = sheetLines(this.board),
                knownBefore = sheetLines(lastBefore),
            )
        } else {
            null
        }
        if (tools != null) {
            val line = position.facts?.line.orEmpty()
            if (LINE_GIVEN in directive && line.isNotEmpty()) {
                // What comes after the line's end, asked of the engine here: left to the model,
                // Luna at no effort called the tool one time in two and gave the line again.
                val answer = tools.run("analyse", JSONObject().put("moves", JSONArray(line)))
                directive += "\n\nThe engine, asked about the line's end:\n$answer"
            }
            directive += "\n\n$TOOL_WORDS"
        }
        val messages: List<ChatMessage>
        val problem: String?
        synchronized(lock) {
            conversation.level = renderLevel
            val harness = conversation.render()
            messages = listOf(ChatMessage("user", "$harness$faced$earlier\n\n# Task\n$directive"))
            // What the coach holds for the position on the board, which the position now must
            // show (the early depth was answered from but never shown: the coach review).
            val held = current?.takeIf { it.board.fen() == this.board.fen() }
            problem = harnessProblem(
                harness,
                this.board,
                userSide,
                lines = held?.analysis?.lines?.isNotEmpty() == true,
                facts = held?.facts != null,
            )
        }
        if (problem != null) harnessProblems.add("${trigger.slug} at ply ${position.ply}: $problem")
        var unlisted: Set<String>? = null
        val earlierMoves = before
        if (earlierMoves != null) {
            var shown = GAME_SO_FAR.replace(messages[0].content, "")
            for (constant in listOf(NO_FIGURES, NOT_THE_MOVE, NOT_THE_MOVE_HELD, TIE_LISTED, NO_EARLIER)) {
                shown = shown.replace(constant, " ")
            }
            val sanctioned = HashSet<String>()
            for (m in SAN.findAll(shown)) sanctioned.add(m.groupValues[1].trimEnd('+', '#'))
            for (m in facts.candidates + facts.line + facts.threats) sanctioned.add(m.trimEnd('+', '#'))
            unlisted = earlierMoves.filter { it.trimEnd('+', '#') !in sanctioned }.toSet()
        }
        val request = CoachRequest(
            trigger = trigger,
            level = level,
            system = system,
            messages = messages,
            facts = facts,
            board = position.board,
            fallback = fallback,
            ply = position.ply,
            gameId = momentGame ?: "",
            userText = userText,
            disclosed = verdicts.filter { it.startsWith("Engine on ") }.joinToString("\n"),
            annotations = annotationsFor(
                facts,
                level,
                position.board,
                trigger = trigger,
                playedSan = played?.san,
                severity = assessment?.severity?.name?.lowercase(),
            ),
            candidates = if (verbalizedSampling) {
                candidates ?: (if (trigger == Trigger.USER_ASK) VS_ASK_CANDIDATES else VS_CANDIDATES)
            } else {
                1
            },
            played = recordMoves(),
            playedSan = played?.san,
            assessment = assessment,
            streamTo = streamSink?.invoke(trigger, position.ply),
            routine = routine,
            // A fast game gets the template, not a model reply; what the player asks, and the
            // review of a game that is over, always get the model.
            templateOnly = fastNow(ts) &&
                trigger != Trigger.USER_ASK && trigger != Trigger.REVIEW,
            mustName = mustName,
            frames = frames,
            nowAnswer = nowAnswer,
            record = numberedMoves(),
            unlisted = unlisted,
            tools = tools,
            epoch = momentEpoch,
            elicit = elicit,
        )
        speaker.submit(request)
    }

    /** The moves played in the game so far as the record numbers them: "17. Rh1". */
    private fun numberedMoves(): Set<String> = synchronized(lock) {
        conversation.judgedRecords().map { conversation.numbered(it.move) }.toSet()
    }

    /**
     * What a model reply must name, as its template does: the better move, after a move that was
     * not the best; for a hint, by its rung -- one of the key fact's squares, the target's or the
     * piece's, the move; their threat, asked for it. Never a move the sheet lists as one of
     * several good ones: the directive forbids the player's moves, and requiring them rejected
     * every obedient reply (the runtime audit, 2026-09-25). Nothing for a word of praise, another
     * question, or a template with none. `coach._must_name`.
     */
    private fun mustName(
        trigger: Trigger,
        position: CoachPosition,
        level: DisclosureLevel,
        template: String,
        assessment: MoveAssessment?,
        played: MovePlayed?,
        userText: String?,
    ): List<String> {
        if (template.isEmpty()) return emptyList()
        val said = userText ?: ""
        val facts = position.facts
        var trigger = trigger
        var assessment = assessment
        var played = played
        val own = if (trigger == Trigger.USER_ASK) askedOwnMove(said) else null
        if (own != null) {
            // "Why was that bad?": the lesson on their last move, as after it. Not on a
            // follow-up, which the directive tells not to repeat the better move: requiring it
            // refused every obedient reply, and the template said the verdict again (the prompt
            // audit, 2026-09-28).
            if (own.move.ply != position.ply + 1 || followingUp(said)) return emptyList()
            trigger = Trigger.POST_MISTAKE
            assessment = own.assessment
            played = own.move
        }
        if (trigger == Trigger.POST_MOVE || trigger == Trigger.POST_MISTAKE) {
            if (assessment == null || played == null || assessment.best.isEmpty()) return emptyList()
            val move = Move.fromUciOrNull(assessment.best[0]) ?: return emptyList()
            val best = position.board.san(move)
            return if (best != played.san && names(template, best)) listOf(best) else emptyList()
        }
        if (facts == null) return emptyList()
        if (trigger == Trigger.USER_ASK && ASK_THREAT.containsMatchIn(said)) {
            // On the opponent's turn their threat is their best move now (theirMoveNow).
            val theirs = if (position.board.turn != userColour) facts.candidates.take(1) else facts.threats
            return theirs.filter { names(template, it) }
        }
        val hint = trigger == Trigger.DECISIVE_QUESTION || trigger == Trigger.STUCK || (
            trigger == Trigger.USER_ASK && (ASK_HINT.matchesAt(said, 0) || ASK_AIM.containsMatchIn(said))
            )
        if (!hint) return emptyList()
        var level = level
        if (trigger == Trigger.STUCK) {
            level = minOf(level, DisclosureLevel.FOCUS) // the rung its template is on
        }
        if (level.value >= DisclosureLevel.MOVE.value) {
            val best = facts.candidates.firstOrNull() ?: ""
            return if (best.isNotEmpty() && names(template, best)) listOf(best) else emptyList()
        }
        val choices = squaresIn(facts.choices.joinToString(" "))
        val source = if (level.value >= DisclosureLevel.FOCUS.value) {
            listOfNotNull(facts.target, facts.piece).filter { it.isNotEmpty() }.joinToString(" ")
        } else {
            facts.key ?: ""
        }
        val named = squaresIn(template) - choices
        val squares = (squaresIn(source) intersect named).ifEmpty { named }
        return squares.sorted()
    }

    /** The moves played in the game so far, as SAN. `coach._played_moves`. */
    private fun playedMoves(): Set<String> = synchronized(lock) {
        conversation.judgedRecords().map { it.move.san }.toSet()
    }

    /**
     * Boards along every line the facts were read from -- the engine's lines, the traps, the
     * recaptures the best line declines, what a quiet best move threatens, the opponent's threats
     * after a pass -- so a sentence the template itself says ("after Rxa5, Kxf6 ...") is not
     * faulted as unplayable when the model says it too (the eval audit, 2026-09-25). The weaker
     * lines only to four plies. `coach._analysis_frames`.
     */
    internal fun analysisFrames(position: CoachPosition): List<Board> {
        val analysis = position.analysis ?: return emptyList()
        val root = position.board
        val frames = ArrayList<Board>()

        fun walk(start: Board, ucis: List<String>, limit: Int) {
            var board = start
            for (uci in ucis.take(limit)) {
                val move = Move.fromUciOrNull(uci) ?: return
                if (!board.isLegal(move)) return
                board = board.copy(withStack = false)
                board.push(move)
                frames.add(board)
            }
        }

        for ((i, line) in analysis.lines.withIndex()) walk(root, line.moves, if (i == 0) 8 else 4)
        for (line in analysis.traps + analysis.defences) walk(root, line.moves, 8)
        if (!root.isCheck()) {
            val passed = root.copy(withStack = false)
            passed.push(Move.NULL)
            for (san in analysis.threats) {
                val after = passed.copy(withStack = false)
                val threat = after.parseSan(san) ?: continue
                after.push(threat)
                frames.add(after)
            }
            if (analysis.threatens.isNotEmpty() && analysis.lines.isNotEmpty() &&
                analysis.lines[0].moves.isNotEmpty()
            ) {
                val best = Move.fromUciOrNull(analysis.lines[0].moves[0])
                if (best != null && root.isLegal(best)) {
                    val quiet = root.copy(withStack = false)
                    quiet.push(best)
                    if (!quiet.isCheck()) {
                        quiet.push(Move.NULL)
                        for (line in analysis.threatens) {
                            walk(quiet, line.moves, 8)
                            walk(root, line.moves, 8)
                        }
                    }
                }
            }
        }
        return frames
    }

    /**
     * The board after the move just played, and along the engine's line from there: the board
     * the model is shown, so a true sentence about it is not faulted against the position before
     * the move. `coach._frames_after`.
     */
    private fun framesAfter(position: CoachPosition, played: MovePlayed): List<Board> {
        val move = Move.fromUciOrNull(played.uci) ?: return emptyList()
        if (!position.board.isLegal(move)) return emptyList()
        var board = position.board.copy(withStack = false)
        board.push(move)
        val frames = arrayListOf(board)
        val left = positions[played.ply]
        val analysis = left?.analysis
        if (analysis != null && analysis.lines.isNotEmpty()) {
            for (uci in analysis.lines[0].moves.take(8)) {
                val step = Move.fromUciOrNull(uci) ?: break
                if (!board.isLegal(step)) break
                board = board.copy(withStack = false)
                board.push(step)
                frames.add(board)
            }
        }
        return frames
    }

    /**
     * The moves the messages name about the game so far: every move played, and the move
     * each verdict said the engine preferred.
     */
    private fun recordMoves(): Set<String> {
        val out = HashSet<String>()
        synchronized(lock) {
            for (r in conversation.judgedRecords()) {
                out.add(r.move.san)
                val best = r.assessment?.best
                if (!best.isNullOrEmpty()) {
                    val move = Move.fromUciOrNull(best[0])
                    // `with suppress(ValueError)`, as the desktop reads it.
                    if (move != null) runCatching { Board(r.move.fenBefore).san(move) }.getOrNull()?.let { out.add(it) }
                }
            }
        }
        return out
    }

    /**
     * What may be said about the position the player faces, stated on the directive, the one
     * place it is stated; with [nowLevel], the reply is about a position already moved from, and
     * the one on the board has its own level. Below the move, why the model sees the answer all
     * the same: said as "never" in five places, it gave the vague hint Tai ranks below one that
     * says a little too much (the prompt audit, 2026-09-28). [theirs]: the position is the
     * opponent's to move, and told to keep back "the player's move" there the model gave the
     * player's hint on the opponent's position (the coach review, 2026-10-01).
     */
    private fun levelLine(
        level: DisclosureLevel,
        position: CoachPosition,
        nowLevel: DisclosureLevel? = null,
        past: Boolean = false,
        theirs: Boolean = false,
    ): String {
        if (position.board.isGameOver()) return ""
        if (theirs) {
            return "It is the opponent's move: what they threaten -- their best move and what it " +
                "does -- may be said.\n\n"
        }
        fun words(of: DisclosureLevel): String = when (of) {
            DisclosureLevel.SILENT -> "nothing about the position"
            DisclosureLevel.IDEA -> IDEA_LEVEL_WORDS
            DisclosureLevel.FOCUS -> "the idea, the target and the piece; not the move"
            DisclosureLevel.MOVE -> "the move and why it works; not the whole line"
            DisclosureLevel.LINE -> "everything, the line included"
        }
        if (nowLevel != null) {
            return "The position this is about has been moved from: everything about it may be " +
                "said. Disclosure for the position the player faces now: ${words(nowLevel)}.\n\n"
        }
        if (past) {
            return "The position this is about has been moved from: everything about it may be " +
                "said, and what the opponent can do now.\n\n"
        }
        val why = if (level.value >= DisclosureLevel.IDEA.value && level.value < DisclosureLevel.MOVE.value) {
            " $SEE_ALL_WORDS"
        } else {
            ""
        }
        return "Disclosure for the position the player faces: ${words(level)}.$why\n\n"
    }

    /**
     * A message that needs no model; it queues behind whatever the model is writing, unless
     * [now], when it goes out at once. With [recordNow] it is a turn of the conversation from
     * this moment, so a request built next already shows it. Without [turn] it is never one:
     * "One moment -- still looking at this position." recorded as the coach's turn made the
     * question it held, answered when the analysis landed, a follow-up -- "You have already told
     * them the verdict ... do not repeat either" (the coach review, 2026-10-01).
     */
    private fun queuePlain(
        trigger: Trigger,
        text: String,
        ts: Double,
        ply: Int? = null,
        level: DisclosureLevel = DisclosureLevel.SILENT,
        annotations: List<Annotation>? = null,
        fen: String? = null,
        now: Boolean = false,
        recordNow: Boolean = false,
        turn: Boolean = true,
        moment: Pair<Int, String?>? = null,
    ) {
        val (momentEpoch, momentGame) = moment ?: (epoch to gameId)
        val message = CoachMessage(
            text = text,
            level = level,
            trigger = trigger,
            gameId = momentGame,
            ply = ply,
            fen = fen,
            annotations = annotations ?: emptyList(),
        )
        // Asked for in a moment gone by, it is no turn of this one ([poll]).
        if (recordNow && momentEpoch == epoch) synchronized(lock) { conversation.coachSaid(text) }
        speaker.submitReady(
            Spoken(
                message = message,
                level = level,
                problems = emptyList(),
                attempts = 0,
                recorded = recordNow || !turn,
                epoch = momentEpoch,
            ),
            now = now,
        )
    }

    /**
     * The turn's instruction to the model, and the template used if it fails. [elicit] ends a
     * mistake's lesson by asking what the player saw (see [elicitDue]).
     */
    internal fun instruction(
        trigger: Trigger,
        position: CoachPosition,
        level: DisclosureLevel,
        userText: String?,
        verdicts: List<String>,
        assessment: MoveAssessment?,
        played: MovePlayed?,
        elicit: Boolean = false,
    ): Pair<String, String> {
        val facts = position.facts!!

        if (trigger == Trigger.REVIEW) {
            return (
                "The game is over. Sum it up from the review in the record above and the " +
                    "verdicts on the moves above, and from nothing else: the result, what went " +
                    "well, and the one thing to work on. When the review says what they kept " +
                    "missing, that is the thing: the pattern in plain words and the habit that " +
                    "fixes it, as the review gives it, with the moment it cost most. Otherwise the " +
                    "moment the game turned, the better move there, and why it happened, in plain " +
                    "words. Three or four sentences, no question. Do not infer the course of the " +
                    "game from the position on the board."
                ) to reviewLines.take(4).joinToString(" ")
        }
        if (trigger == Trigger.DECISIVE_QUESTION) {
            // Never fired live since 2026-10-01 (nothing unasked before the stuck limit); the
            // eval's recorded requests of it still replay (tools/eval/judge.py).
            val (template, note) = coolHint(facts, ideaTemplate(facts))
            return "$IDEA_DIRECTIVE $NO_FIGURES$note" to template
        }
        if (trigger == Trigger.STUCK) {
            // The hints unasked stop at the focus: the move is only for when they ask. The template
            // on the level a question had opened named the move while the model was told not to,
            // and every reply that obeyed went for the template (Tai's game, 2026-09-28: "d5: your
            // bishop on c8 is still shut in", twice).
            val rung = minOf(level, DisclosureLevel.FOCUS)
            val directive = if (rung.value >= DisclosureLevel.FOCUS.value) FOCUS_DIRECTIVE else IDEA_DIRECTIVE
            val (template, note) = coolHint(facts, hintTemplate(facts, rung))
            return (
                "The player has been thinking for a while and is stuck. " +
                    directive +
                    " Take a different angle from anything already said about this position. " +
                    NO_FIGURES +
                    note
                ) to template
        }
        if ((trigger == Trigger.POST_MOVE || trigger == Trigger.POST_MISTAKE) &&
            assessment != null && played != null
        ) {
            // The better move is the engine's own from the position the player faced
            // (assessment.best, UCI); the fact sheet's first candidate stands in when the
            // assessment has none. Without a model this sentence is the whole lesson, so it
            // must never name the move played as the better one.
            var best = facts.candidates.firstOrNull() ?: "another move"
            if (assessment.best.isNotEmpty()) {
                val move = Move.fromUciOrNull(assessment.best[0])
                if (move != null) best = position.board.san(move)
            }
            if (best == played.san && trigger == Trigger.POST_MISTAKE) best = "another move"
            val loss = assessment.expectedLoss
            val label = assessment.label?.name?.lowercase() ?: assessment.severity.name.lowercase()
            val betterWords =
                if (facts.candidates.isNotEmpty()) inWords(position.board, best) else best
            val routine = facts.candidates.isEmpty() || best == played.san ||
                isLocal(assessment.label)
            var fallback: String
            var instructionText: String
            if (trigger == Trigger.POST_MISTAKE) {
                // The better move played out, and the pattern to keep: never "the idea was a
                // mating attack" and nothing more.
                // What the move let them do, from the position it left, when that has been
                // analysed: the grounded half of "what their move allowed".
                // Late -- the opponent has replied and the player is thinking again -- what the
                // move allowed is the threat on the board now, and a question that sends them to
                // the board is a nudge while they think: "one question that sends them to
                // find..." asked what the move allowed (the deep prompt review, 2026-10-01). The
                // lesson is then the better move and what theirs missed.
                val now = current
                val facing = now != null && now.ply > position.ply &&
                    now.board.turn == userColour && !now.board.isGameOver()
                val left = positions[played.ply]
                val leftAnalysis = left?.analysis
                val allowed = if (left != null && leftAnalysis != null && !facing) {
                    IdeaWords.allowed(left.board, leftAnalysis, played.san)
                } else {
                    null
                }
                val kind = costKind(assessment.severity, label)
                val opener = bags.draw("cost:$kind", COST_BY_SEVERITY.getValue(kind)).replace("{best}", best)
                val rule = ruleOf(facts, best, board = position.board, played = played.uci)
                val (lessonText, held) = freshLesson(opener, rule, facts) { withRule ->
                    lessonTemplate(
                        facts,
                        best,
                        opener,
                        board = position.board,
                        played = played.uci,
                        allowed = allowed,
                        withRule = withRule,
                        strong = strongPlayer,
                    )
                }
                fallback = lessonText
                val given = if (!held && !strongPlayer) rule else null
                var data = lessonData(position.board, best, played.uci, given)
                if (!allowed.isNullOrEmpty()) {
                    data += " What ${played.san} allowed, from the engine's line: $allowed."
                }
                // A question to close on is offered, not asked for: required, it was the one a
                // lesson had room for once the pattern was said, and often a quiz on what the
                // reply had just told them.
                instructionText =
                    "The player just played ${played.san}, which was ${rated(label)}: it cost " +
                        "${percent(loss)} of the game. The better move was $best ($betterWords). " +
                        "Teach the difference so they spot it next time: why the better move works " +
                        (if (facing) "and what their move missed, " else "and what their move missed or allowed, ") +
                        "from the sheet of the position they faced -- its goal, its pattern, the " +
                        "tactic when it names one -- as a pattern to remember. " +
                        (
                            if (facing) {
                                "Three sentences at most, no question: they are thinking about their " +
                                    "next move now, and what the opponent threatens is theirs to find. "
                            } else {
                                "Three sentences at most, then, if you like, one question that " +
                                    "sends them to find on the board something your reply has not said. "
                            }
                            ) +
                        "$TIE_TO_THE_GAME $NOT_THE_MOVE $NO_FIGURES$data"
                if (elicit) {
                    // Sparingly: what they saw at that moment, for the profile's record
                    // (causes.from_words reads "I thought it would..." as a line misjudged,
                    // "I didn't see..." as a threat missed). Asked as what they expected, in
                    // the coach's words: told to ask whether they "missed the idea altogether
                    // or misjudged the line", the model said exactly that, every time.
                    fallback += " What were you expecting that move to do?"
                    instructionText =
                        "The player just played ${played.san}, which was ${rated(label)}: it cost " +
                            "${percent(loss)} of the game. The better move was $best " +
                            "($betterWords). Teach the difference so they spot it next time: why " +
                            "the better move works, from the sheet of the position they faced -- its " +
                            "goal, its pattern, the tactic when it names one -- as a pattern to " +
                            "remember, in three sentences at most, then -- instead of a check " +
                            "question -- ask, in one short question of your own, what they " +
                            "expected their move to do. $NOT_THE_MOVE $NO_FIGURES" +
                            lessonData(position.board, best, played.uci, given)
                }
                if (held) instructionText = holdRule(instructionText)
            } else if (isExciting(assessment.label)) {
                // Rare and worth it: the board's own sentence stands in, the model says what
                // makes it the move, with some delight.
                val (written, kind) = verdict(
                    position.board,
                    Move.fromUci(played.uci),
                    label,
                    assessment.tags,
                    phase = facts.phase,
                    threats = facts.threats,
                    taught = verdictTaught(),
                    bags = bags,
                    plain = plainWords,
                    strong = strongPlayer,
                )
                fallback = written
                // Said by the model or, when its reply fails, by this sentence: either way the
                // idea is taught ("the king is a fighting piece" after a great Kd6, then again
                // after Kd5: the offline sweep, 2026-09-27).
                taughtByVerdict(kind)
                val meaning = when (label) {
                    "brilliant" -> "a sound sacrifice"
                    else -> greatMeaning(facts)
                }
                // Only the reason that holds: offered "the sacrifice that works" beside a great
                // move, the model called 11...Nxf3+ gxf3, a knight for a knight, "the knight
                // sacrifice" (the dev audit, 2026-09-26).
                val reason = if (label == "brilliant") {
                    "the sacrifice that works"
                } else {
                    meaning.split(":", limit = 2)[0]
                }
                instructionText =
                    "The player just played ${played.san}: $label -- $meaning. Show some " +
                        "delight, in your own voice, in one or two sentences: name what makes it " +
                        "the move -- $reason, and the idea it serves -- from the sections above, " +
                        "nothing invented. No question, no advice, and not the same words as your " +
                        "last such sentence. " +
                        "$NOT_THE_MOVE $NO_FIGURES"
                // The pattern to keep is a rule of thumb like any other, on the one budget: two
                // great moves running got "Keep spotting this pattern ..." and "keep recognizing
                // this pattern ..." back to back (Tai's puzzle, 2026-09-28).
                instructionText = budgeted(instructionText, "verdict:$kind")
            } else if (routine) {
                fallback = verdictWord(label)
                val meaning = when (label) {
                    "book" -> "opening theory"
                    "brilliant" -> "a sound sacrifice"
                    "great" -> greatMeaning(facts)
                    "best" -> "the engine's own choice"
                    "excellent" -> "as strong as the best move"
                    "good" -> "a sound move, a little behind the best"
                    else -> "a sound move"
                }
                instructionText =
                    "The player just played ${played.san}: $label -- $meaning. One short " +
                        "sentence, fifteen words at most, in your own voice: a word of praise, and " +
                        "the idea behind the move as a lesson to keep when a message gives one -- " +
                        "the plan, the tactic -- and only when it is what the move is for. No " +
                        "question, no lecture, no advice, no guess at what it allows or risks, and " +
                        "not the same words as your last such sentence. " +
                        "$NOT_THE_MOVE $NO_FIGURES"
                instructionText = budgeted(instructionText, "verdict:$label")
            } else if (assessment.severity == Severity.MISTAKE || assessment.severity == Severity.BLUNDER ||
                label == "miss"
            ) {
                // A mistake in blitz: the policy keeps it to the word after the move (no
                // lesson), but the word says what the move was -- never "fine".
                val kind = costKind(assessment.severity, label)
                fallback = bags.draw("cost:$kind", COST_BY_SEVERITY.getValue(kind)).replace("{best}", best)
                fallback += "."
                instructionText =
                    "The player just played ${played.san}, which was ${rated(label)}: $best " +
                        "($betterWords) was better. It is a fast game: one or two sentences, no " +
                        "lesson, no question. $NOT_THE_MOVE $NO_FIGURES"
            } else {
                // The better move and what it does, when the sheet has it, is the lesson. An
                // inaccuracy cost something, and is said so: given a sound move's word, one was
                // "Fine, but Qxf7# was a little tighter", the model told "an inaccuracy: sound, a
                // little behind" (the coach review, 2026-10-01).
                val slip = assessment.severity == Severity.INACCURACY || label == "inaccuracy"
                val opener = (
                    if (slip) {
                        bags.draw("cost:inaccuracy", COST_BY_SEVERITY.getValue("inaccuracy"))
                    } else {
                        bags.draw("fine", FINE_TIGHTER)
                    }
                    ).replace("{best}", best)
                val (lessonText, held) = freshLesson(
                    opener,
                    null, // a fine move: the better move and what it does, no rule of thumb
                ) { withRule ->
                    lessonTemplate(
                        facts,
                        best,
                        opener,
                        board = position.board,
                        played = played.uci,
                        withRule = withRule,
                        strong = strongPlayer,
                    )
                }
                fallback = lessonText
                // How it rated goes in passing: asked for first, it opened half these replies
                // with "Fine", "Playable" or "That works, though" (the prompt audit, 2026-09-28).
                val ratedAs = if (slip) {
                    "it cost a little, and $best ($betterWords) was better"
                } else {
                    "sound, a little behind $best ($betterWords)"
                }
                instructionText =
                    "The player just played ${played.san}, which was ${rated(label)}: $ratedAs. " +
                        "One or two sentences: the better move, what it does, and the idea behind it " +
                        "as a lesson to keep, with how theirs rated in passing. No lecture, no " +
                        "question. " +
                        "$TIE_TO_THE_GAME $NOT_THE_MOVE $NO_FIGURES" +
                        lessonData(position.board, best, played.uci, null)
                if (held) instructionText = holdRule(instructionText)
            }
            return instructionText to fallback
        }
        // An explicit ask. The player's words are already the last message of the
        // conversation; the directive says only how to answer them.
        val said = userText?.ifEmpty { null } ?: "hint" // `user_text or "hint"`
        var checks = ""
        val claimsChecked = verdicts.filter { !it.startsWith("Engine on ") }
        val engineWords = verdicts.filter { it.startsWith("Engine on ") }
        if (claimsChecked.isNotEmpty()) {
            checks = "Their claims, checked against the board:\n- " +
                claimsChecked.joinToString("\n- ") + "\n\n"
        }
        if (engineWords.isNotEmpty()) {
            // The engine on what they asked about: theirs to hear about in full.
            checks += engineWords.joinToString("\n") + "\n\n"
        }
        // The rung's template: two hints, then the answer; the line when asked for it. The hints
        // speak to the side to move: on the opponent's turn a failed reply falls back to what they
        // will play, never their hint in the player's voice.
        var fallback = hintTemplate(facts, level, position.board)
        if (position.board.turn != userColour && level.value < DisclosureLevel.LINE.value) {
            fallback = theirMoveNow(position.board, facts)
        }
        // The position answered about is not the one on the board (its analysis has not
        // landed): say so, or the template speaks of a past position as the present -- "the
        // line is Bb3 ..." right after Bb3 was played (the offline pass, 2026-09-21).
        val nowAt = current
        if (nowAt != null && position.ply != nowAt.ply) {
            val own = lastOwnMove()
            if (own != null && own.move.ply == nowAt.ply && position.ply == own.move.ply - 1) {
                val san = own.move.san
                val rest = if (facts.line.isNotEmpty() && facts.line[0] == san) {
                    facts.line.drop(1)
                } else {
                    facts.line
                }
                fallback = when {
                    level.value >= DisclosureLevel.LINE.value && rest.isNotEmpty() ->
                        "It's their move. After your move, the line runs ${rest.joinToString(" ")}."

                    level.value >= DisclosureLevel.MOVE.value && facts.candidates.isNotEmpty() -> {
                        val choice = facts.candidates[0]
                        "It's their move now. " + if (choice == san) {
                            "Yours was the engine's choice too."
                        } else {
                            "The engine's choice there was $choice."
                        }
                    }

                    else -> "It's their move. See what they do about yours, then ask again."
                }
            } else {
                fallback = "Give me a moment; I am still reading this position."
            }
        }
        // Asked about a move, square or piece, a refused reply falls back to the engine's word on
        // it, as the offline coach says it: "is Bxf7+ good?" got the mate-in-one hint (the deep
        // prompt review, 2026-10-01). A branch with its own answer replaces it below. Moves only:
        // a piece named in passing is not read out (the offline coach's rule).
        val scored = engineWords.filter { "the expected score is" in it }
        if (scored.isNotEmpty()) fallback = scored.joinToString(" ") { spoken(it) }
        // A question about a fact that is never withheld -- the material, the opening, who
        // stands better -- falls back to that fact, not to the hint for the position.
        val material = if (ASK_MATERIAL.containsMatchIn(said) || ASK_EVAL.containsMatchIn(said)) {
            materialWords(level, facts)
        } else {
            null
        }
        if (!material.isNullOrEmpty() && ASK_MATERIAL.containsMatchIn(said)) {
            fallback = "Material: $material."
        } else if (OPENING_FACT.containsMatchIn(said)) {
            // The opening asked for by name: "what should I play against this opening? give me a
            // line" was answered with its name (Tai's game, 2026-09-28).
            val opening = openingSoFar()
            fallback = if (opening != null) {
                "The opening was the $opening."
            } else {
                "We left the opening book early."
            }
        } else if (ASK_EVAL.containsMatchIn(said)) {
            fallback = standingTemplate(position, material)
        }
        // Their words answer the question the coach just asked: the thread is closed, not
        // carried on with another question. Every answer had earned a new question and none was
        // resolved -- the students' "what Nc4 was expected to do was asked but not answered"
        // (2026-09-26).
        val answering = answeringOurQuestion(said)
        val close = " They are answering the question you just asked, your last turn above: say first, " +
            "plainly and from the facts, whether they have it right; if they do not, give the " +
            "answer yourself in a sentence. Ask nothing more -- this closes that question."
        // After the coach has already spoken since the move, a follow-up gets only what it asks:
        // the verdict and the better move came back in every reply ("e6 was a miss costing 41% of
        // the game; Nxc3 was better" twice running), and "I understand: I should play Nxc3" got
        // the lesson again and "Which White piece is still at home on g1?" (the AI's sessions,
        // 2026-09-26).
        val followup = followingUp(said)
        val asks = questionPart(said).isNotEmpty()
        val follow = " You have already told them the verdict on it and the better move, in your turns " +
            "above: do not repeat either." +
            if (asks) {
                " Answer only what they ask now -- concretely, from the facts, in one to three " +
                    "sentences -- and ask nothing back."
            } else {
                " They ask nothing: reply in one short sentence -- say whether what they " +
                    "say is right, or put it right -- and ask nothing."
            }
        var directive: String
        val lastOpponent = lastOpponentMove()
        val lastOwn = askedOwnMove(said)
        if (ASK_REVIEW.containsMatchIn(said)) {
            // An overall verdict comes from the tally in "Game so far", never from an
            // impression of the last position: the first review told a player a queen up that
            // they had been a queen down.
            directive =
                "They asked how they did overall. Answer from the engine's verdicts on their " +
                    "moves in the record above -- the labels, the tally under their last move, " +
                    "where they lost ground, how the material went -- and from nothing " +
                    "else: do not infer the course of the game from the position on the board. " +
                    "Two or three sentences: what went well -- how many of their moves were strong, " +
                    "and their slips by kind, in the tally's numbers -- and the one thing to work " +
                    "on: when the review, or the pattern already told them, says what they keep " +
                    "missing, that pattern in plain words and the habit that fixes it, as given; " +
                    "otherwise the move the " +
                    "tally lists first where ground was lost most, the better move there, and why, " +
                    "in plain words. If no move has a verdict yet, say you have not seen enough to " +
                    "say."
            // A review has a record to fall back on, not a line.
            fallback = reviewTemplate()
        } else if (ASK_HINT.matchesAt(said, 0) || ASK_AIM.containsMatchIn(said)) {
            // A bare "hint" carries no claim to be honest about; treated as one, the model
            // answered its own last message as if the player had said it. Two hints, then the
            // answer: the rung decides what the hint may give. A question about what to aim
            // for is answered by the same hint: the generic answer ("be honest about their own
            // claim") gave the model no lesson to give (parity pass 2, 2026-09-25).
            var aim = if (ASK_HINT.matchesAt(said, 0)) "" else "They asked what to aim for: answer with "
            aim += if (aim.isNotEmpty()) "the hint at this rung. " else ""
            directive = when {
                // On the opponent's move there is no move of the player's to find: asked for
                // "the first hint" there, the model gave the player's hint on the opponent's
                // position (the coach review, 2026-10-01). Their threat is the hint, as
                // [theirMoveNow] says it without a model.
                position.board.turn != userColour && !position.board.isGameOver() ->
                    "They asked ${if (ASK_HINT.matchesAt(said, 0)) "for a hint" else "what to aim for"}" +
                        ", and it is the opponent's move, not theirs: there is no move of theirs to " +
                        "find yet. Say what the opponent threatens now -- their best move and what " +
                        "it does, from the engine lines (the opponent's move) -- and what the " +
                        "player should be ready for once it is played. One or two sentences, no " +
                        "question."
                level.value >= DisclosureLevel.MOVE.value -> {
                    val bestSan = facts.candidates.firstOrNull() ?: ""
                    val given = bestSan.isNotEmpty() &&
                        saidSinceMove().any { bestSan.trimEnd('+', '#') in it }
                    // Asked again once the move was given, "The next hint is the answer" said it
                    // was still to come; said again, it is with why it beats what is clearly
                    // worse, as the board says it -- not a restatement (`alternatives`).
                    val worse = if (given && position.analysis != null) {
                        worseMoves(position.board, position.analysis!!, userColour)
                    } else {
                        emptyList()
                    }
                    if (worse.isNotEmpty()) {
                        fallback = "$fallback ${worseSaid(worse)}".trim()
                        aim + MOVE_GIVEN_WHY.format(bestSan, worseText(worse))
                    } else {
                        aim + if (given) MOVE_GIVEN.replace("{move}", bestSan) else MOVE_DIRECTIVE
                    }
                }
                level.value >= DisclosureLevel.FOCUS.value -> {
                    val bestSan = facts.candidates.firstOrNull() ?: ""
                    // "The piece to use and what it goes after", the piece being the king, invited
                    // "your king goes after their queen" (the deep prompt review).
                    aim.ifEmpty { "They asked for another hint. " } + FOCUS_DIRECTIVE +
                        if (bestSan.startsWith("O-O")) " $FOCUS_KING" else ""
                }
                else -> aim.ifEmpty { "They asked for a hint. " } + IDEA_DIRECTIVE
            } + " $NO_FIGURES Do not comment on anything said earlier."
        } else if (position.board.turn != userColour && !position.board.isGameOver() &&
            (ASK_LINE.containsMatchIn(said) || ASK_MOVE.containsMatchIn(said))
        ) {
            // On the opponent's move the engine's line and best move are the OPPONENT's: "what
            // should I play?" there told the model to give the player the opponent's best move
            // as theirs (the deep prompt review, 2026-10-01). Answered like a hint there.
            fallback = theirMoveNow(position.board, facts)
            val asked = if (ASK_LINE.containsMatchIn(said)) "to see the line" else "what to play"
            directive =
                "They asked $asked, and it is the opponent's move, not theirs: there is no " +
                    "move of theirs to play yet. Say so, then what the opponent threatens now -- " +
                    "their best move and what it does, from the engine lines (the opponent's " +
                    "move) -- and what the player should be ready for once it is played. One or two " +
                    "sentences, no question. $NO_FIGURES"
        } else if (level.value >= DisclosureLevel.LINE.value &&
            ASK_LINE.containsMatchIn(said) && facts.line.isNotEmpty()
        ) {
            // The line move by move, computed: the only reasons a move of it may be given (4 of
            // the 5 false claims the bench's labellers found were in these answers).
            val notes = lineNotes(position.board, facts.line, player = userColour)
            val byMove = if (notes.isNotEmpty()) {
                " The line move by move, as the board gives it: $notes. Explain it only from " +
                    "these notes: a move 'forces' something only where a note says it is the " +
                    "only legal move or that only a king move gets out of check."
            } else {
                ""
            }
            // Given once already, the line asked for again is what comes after it: "ok and after
            // that?" got the same line back (Tai's puzzle, 2026-09-28). The engine's word on its
            // end is put before the model ([speak]).
            val lineStart = facts.line.take(3).map { it.trimEnd('+', '#') }
            val given = saidSinceMove().any { said -> lineStart.all { it in said } }
            directive =
                "They asked to see the line: give it -- ${facts.line.joinToString(" ")} -- " +
                    "plainly, then one short sentence on the idea behind it." + byMove +
                    " Do not answer with a question." + if (given) " $LINE_GIVEN" else ""
        } else if (level.value >= DisclosureLevel.MOVE.value && facts.candidates.isNotEmpty() &&
            position.board.turn == userColour &&
            (
                ASK_OTHER_MOVES.containsMatchIn(said) ||
                    (ASK_ALSO.containsMatchIn(said) && !ASK_LAST_MOVE.containsMatchIn(said))
                )
        ) {
            // The moves besides the best: all that are about as good, named, and the clearly
            // worse ones with what goes wrong ([alternativesAnswer]).
            val (asked, alternatives) = alternativesAnswer(position, facts, said)
            directive = asked
            // A move they name, rated by the engine, is that word's to say.
            if (engineWords.none { "the expected score is" in it }) fallback = alternatives
        } else if (level.value >= DisclosureLevel.MOVE.value &&
            ASK_MOVE.containsMatchIn(said) && facts.candidates.isNotEmpty()
        ) {
            var asked =
                "They asked what to play: tell them plainly it is ${facts.candidates[0]}, then " +
                    "why it works, from the fact sheet's why and pattern, correctly. Do not answer " +
                    "with a question this time."
            if (facts.candidates.size > 1) {
                // Not only line 1: others are as good, and the best is not one move.
                asked += " ${facts.candidates.size} moves are about equally good here " +
                    "(${facts.candidates.joinToString(", ")}): say so, and name them, rather " +
                    "than naming only the first."
                val rest = facts.candidates.drop(1)
                fallback = (
                    "$fallback " + (if (rest.size > 1) rest.dropLast(1).joinToString(", ") + " and " else "") +
                        "${rest.last()} ${if (rest.size > 1) "are" else "is"} about as good."
                    ).trim()
            }
            directive = asked
        } else if (ASK_OPPONENT_MOVE.containsMatchIn(said) && lastOpponent != null) {
            val theirs = lastOpponent.assessment!!
            val san = lastOpponent.move.san
            val label = theirs.label?.name?.lowercase() ?: theirs.severity.name.lowercase()
            fallback = "Their $san was ${rated(label)}."
            directive =
                "They ask about the opponent's last move, $san. It is in the record under " +
                    "'Last move' with its verdict ($label) and what it does; all of that is " +
                    "theirs to hear. Name the move, say how it rated and what it does, in " +
                    "plain words, and what they threaten now when the facts list a threat (at " +
                    "most two), two or three sentences; then, if you like, one question about " +
                    "what it asks of them. Never answer about the player's own move instead. " +
                    NO_FIGURES
        } else if (lastOwn != null) {
            // Asked about their own last move. Its verdict was given in its move message and
            // the facts of the position it was played from in the message before: all of it
            // may be said outright. Without this the model took "the better move is withheld"
            // for the position now to cover the past one and answered with a hint instead.
            val own = lastOwn.assessment!!
            val san = lastOwn.move.san
            val label = own.label?.name?.lowercase() ?: own.severity.name.lowercase()
            val loss = own.expectedLoss
            var best = ""
            if (own.best.isNotEmpty()) {
                val move = Move.fromUciOrNull(own.best[0])
                best = try {
                    if (move != null) Board(lastOwn.move.fenBefore).san(move) else ""
                } catch (_: Exception) {
                    "" // `except ValueError`
                }
            }
            if (best == san) best = ""
            // The request's facts are those of the position the move was played from when the
            // coach has them (ask() chose that position); the idea is theirs. The theme or the
            // plan, never a static feature: "the idea was the isolated d-pawn ..." became the
            // reason a bishop retreat was an inaccuracy.
            val idea = facts.theme?.ifEmpty { null } ?: facts.plan ?: "" // `facts.theme or facts.plan or ""`
            if (own.severity == Severity.NONE && best.isNotEmpty() && loss >= 0.005) {
                // Sound but not the best: said as the word after it said it ("Fine, but a3 was a
                // little tighter"), never "it cost nothing" (the runtime audit).
                val opener = bags.draw("fine", FINE_TIGHTER).replace("{best}", best)
                // A fine move: the better move and what it does, no rule of thumb; asked, said
                // whole.
                fallback = freshLesson(opener, null, asked = true) { withRule ->
                    lessonTemplate(
                        facts,
                        best,
                        opener,
                        played = lastOwn.move.uci,
                        withRule = withRule,
                        strong = strongPlayer,
                    )
                }.first
                fallback = withLine(fallback, opener, best, facts)
                // What they ask is the answer; the verdict is its ground, said once: told "say so
                // plainly" on every question about the move, the verdict came back four times
                // running (Tai's puzzle, 2026-09-28).
                directive =
                    "They ask about the move they already played, $san. It was $label: sound, " +
                        "a little behind $best. Answer what they ask" +
                        (
                            if (followup) {
                                "; your turns above already gave the verdict and the better move, so " +
                                    "repeat them only where that is what they ask."
                            } else {
                                "; where they ask how the move was, say so plainly -- it was fine -- " +
                                    "then the better move and the idea behind it."
                            }
                            ) +
                        " Two sentences at most, no question. $NO_FIGURES"
            } else if (own.severity == Severity.NONE) {
                fallback = verdictWord(label)
                directive =
                    "They ask about the move they already played, $san: its verdict was " +
                        "$label, and it cost nothing. Answer what they ask, from the record and the " +
                        "sheet of the position it was played from, and invent no fault" +
                        (
                            if (followup) {
                                "; your turns above already gave the verdict, so repeat it only where " +
                                    "that is what they ask."
                            } else {
                                "; where they ask how the move was, say so plainly."
                            }
                            ) +
                        " One or two sentences, no question. $NO_FIGURES"
            } else {
                // What it cost only when they ask: "It cost you 22% of the game." opened every
                // answer, and Tai had said the figure is for when he asks (2026-09-20).
                val askedCost = ASK_COST.containsMatchIn(said)
                fallback = if (askedCost) "It cost you ${percent(loss)} of the game." else ""
                // What the move let them do, from the position it left, as the lesson after the
                // move has it: without it "why was that bad?" named the better move and a rule,
                // and no piece, square or move of the reason (quality round 3).
                val left = positions[lastOwn.move.ply]
                val leftAnalysis = left?.analysis
                val allowed = if (left != null && leftAnalysis != null) {
                    IdeaWords.allowed(left.board, leftAnalysis, san)
                } else {
                    null
                }
                val faced = position.ply == lastOwn.move.ply - 1
                val board = if (faced) position.board else null
                var held = false
                var data = if (!allowed.isNullOrEmpty()) " What $san allowed, from the engine's line: $allowed." else ""
                if (best.isNotEmpty()) {
                    // The better move played out and the pattern to keep, as the lesson says it
                    // -- not "the idea was a pin" and nothing more -- and said whole, what the
                    // lesson after the move said or not: they asked.
                    val opener = "$best was better"
                    val rule = ruleOf(facts, best, board = board, played = lastOwn.move.uci)
                    val (written, wasHeld) = freshLesson(opener, rule, facts, asked = true) { withRule ->
                        lessonTemplate(
                            facts,
                            best,
                            opener,
                            board = board,
                            played = lastOwn.move.uci,
                            allowed = allowed,
                            withRule = withRule,
                            strong = strongPlayer,
                        )
                    }
                    held = wasHeld
                    val lesson = withLine(written, opener, best, facts)
                    fallback = (if (fallback.isNotEmpty()) "$fallback " else "") + lesson
                    val given = if (!held && !strongPlayer) rule else null
                    data += lessonData(board, best, lastOwn.move.uci, given)
                } else if (!allowed.isNullOrEmpty()) {
                    fallback = (if (fallback.isNotEmpty()) "$fallback " else "") +
                        templateSentence(allowed[0].uppercaseChar() + allowed.substring(1))
                }
                if (fallback.isEmpty()) fallback = "That was ${rated(label)}."
                val reason = if (best.isNotEmpty()) "what $best does, and " else ""
                val cost = if (askedCost) " -- it cost ${percent(loss)} of the game: they asked, so say it" else ""
                // A question of the coach's own after the answer is offered, not asked for: they
                // asked theirs, and required, one came after every answer. On a follow-up the
                // verdict is not said again: "say that plainly" beside "do not repeat either" asked
                // for both (the prompt audit, 2026-09-28).
                val tell = if (followup && !answering) "Say why" else "Say that plainly, and say why"
                directive =
                    "They ask about the move they already played, $san, from a position that " +
                        "is past, so all of it may be said: it was ${rated(label)}$cost" +
                        (if (best.isNotEmpty()) ", $best was better" else "") +
                        (if (idea.isNotEmpty()) ", and the idea was $idea" else "") +
                        ". $tell, concretely: ${reason}what $san missed or allowed, with the " +
                        "pieces, squares or moves from the sheet of the position they faced (its " +
                        "pattern, and its tactic when it names one)" +
                        (if (held) " -- ${RULE_HELD_MARK.lowercase()}: they heard one lately." else "") +
                        (if (held) "" else " -- and the pattern to keep.") +
                        data +
                        (
                            if (answering) {
                                "$close Three sentences at most."
                            } else if (followup) {
                                follow
                            } else {
                                " Three sentences at most, then, if you like, one question that " +
                                    "sends them to find on the board something your reply has not said."
                            }
                            ) +
                        (if (askedCost) " Name moves bare, never with a move number in front." else " $NO_FIGURES") +
                        " $TIE_TO_THE_GAME"
            }
            val reply = moves[lastOwn.move.ply + 1]
            if (reply != null && !reply.byUser && repliedAtOnce()) {
                // The opponent's move is last on the board; theirs is the one they ask about.
                directive += " By \"that\" they mean $san, their own move: ${reply.san}, which the " +
                    "opponent played straight after it, is not what they ask about."
            }
            if (followup && !asks) {
                fallback = "" // a nod needs no template: nothing, rather than the verdict again
            }
        } else if (ASK_MATERIAL.containsMatchIn(said) || OPENING_FACT.containsMatchIn(said) ||
            ASK_EVAL.containsMatchIn(said)
        ) {
            // A plain question about the game as it stands: answered plainly, no question back
            // (the generic answer's "point them ... with a question" asked one).
            var plainly =
                "They asked a plain question about the game as it stands. Answer it from the " +
                    "message -- the material, the opening, or who stands better, as it gives them " +
                    "-- in one or two sentences, and no question."
            if (ASK_EVAL.containsMatchIn(said)) {
                plainly += " Who stands better, in the words of the Standing line, and the material " +
                    "as it stands."
            }
            if (OPENING_FACT.containsMatchIn(said)) plainly += " The opening by the name the record gives it."
            if (!material.isNullOrEmpty()) {
                plainly += " The material is this, and your answer gives all of it, in these terms: " +
                    "$material. Never count the pieces yourself."
            }
            directive = plainly
        } else if (ASK_THREAT.containsMatchIn(said) && position.board.turn != userColour) {
            // The opponent is to move: their threat is their best move now. The sheet's threats
            // are the side NOT to move's -- the player's own -- and answered from them, "what are
            // they threatening?" got the player's threat (the code review, 2026-09-26).
            fallback = theirMoveNow(position.board, facts)
            directive =
                "They ask what the opponent threatens, and it is the opponent's move: say what " +
                    "the opponent's best move now is and what it does, from the engine lines (the " +
                    "opponent's move) -- not 'the player's threats', which are the player's own. " +
                    "One or two sentences, no question. " + NO_FIGURES
        } else if (ASK_THREAT.containsMatchIn(said) && position.board.isCheck()) {
            // In check there is no "if you passed": the sheet lists no threat, and "nothing is
            // threatened" was said with their queen checking and hitting a pawn (dev audit,
            // 2026-09-26). The check is the threat, and what else the checker attacks.
            fallback = inCheckThreat(position.board)
            directive =
                "They ask what the opponent threatens, and they are in check: the check is " +
                    "the threat. Say it as the board gives it: $fallback One or two sentences, " +
                    "no question. " + NO_FIGURES
        } else if (ASK_THREAT.containsMatchIn(said)) {
            val threats = facts.threats.take(2)
            fallback = if (threats.isNotEmpty()) {
                val words = threats.map { Conversation.threatWords(position.board, it) }
                "They threaten " + words.joinToString(" and ") + "."
            } else {
                "Nothing is threatened right now: no move of theirs wins anything."
            }
            directive =
                "They ask what the opponent threatens. Say it as the facts give it -- each threat " +
                    "they list, at most two: the move and what it does, from 'the opponent's " +
                    "threats' -- or, when the facts list " +
                    "none, say plainly that nothing is threatened. One or two sentences, no question. " +
                    NO_FIGURES
        } else {
            // What is kept back is the level line's to say: "never reveal that unless the sheet
            // lists it" stood here after the sheet had come to list everything (the prompt audit,
            // 2026-09-28). A claim is theirs only when they make one: told to be honest about
            // "their own claim", Luna answered a plain question with "You are right that g1 is
            // controlled by your King" (Tai's puzzle, 2026-09-29).
            directive =
                "Answer them directly. What they state or take for granted, take as they meant " +
                    "it: say what in it is right as plainly as what is wrong -- and credit them with " +
                    "nothing they did not say: a question states nothing to agree with. When they " +
                    "propose a move, or name a square or piece, say plainly whether it holds up and what it " +
                    "does or allows, from the engine's word on it where given above -- theirs to " +
                    "hear in full: the score after it, how far behind the best, the reply, what the " +
                    "move does; " +
                    (
                        if (position.board.turn == userColour) {
                            "when the engine does not rate their move, say it is not the strongest and, " +
                                "in general terms, why. "
                        } else {
                            // On the opponent's move a move they name is not examined
                            // ([analyseNamed]): "not rated" there means nothing (the deep prompt
                            // review, 2026-10-01).
                            "on the opponent's move no move they name is rated here: rate one only " +
                                "from the engine's word on it. "
                        }
                        ) +
                    "Of the position they face, keep back only " +
                    "what the level line does, and point them toward the right idea" +
                    (if (answering || (followup && !asks)) " in words." else " -- with a question, if that serves.") +
                    " What you already said in this position is under 'Said here': do not say " +
                    "it again in other words; give them something it did not." +
                    when {
                        answering -> close
                        followup && !asks ->
                            " They ask nothing: reply in one short sentence -- say whether what " +
                                "they say is right, or put it right -- and ask nothing."
                        else -> ""
                    }
            if (followup && !asks && engineWords.isEmpty()) {
                fallback = "" // a nod needs no template: nothing, rather than a hint again
            }
        }
        val (cooled, note) = coolHint(facts, fallback)
        return "$checks$directive$note" to cooled
    }
}
