package com.sidecoach.coach

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
private val ASK_REVIEW = Regex(
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
        "how am i doing|is this (?:position )?good for me|what'?s the (?:eval|evaluation|score)|" +
        "how bad is it|am i lost)\\b",
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
private val ASK_HINT = Regex(
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

private val ASK_MOVE = Regex(
    "\\b(best move|what (should|do|would) i (play|do)|what would you play|which move|" +
        "just tell me|what'?s the move|tell me the move|the answer)\\b",
    RegexOption.IGNORE_CASE,
)
private val ASK_LINE = Regex(
    "\\b(the line|variation|continuation|what happens (after|next)|show me the line|" +
        "and then what|what follows)\\b",
    RegexOption.IGNORE_CASE,
)

/** The unlock the user's words earn, if any. */
fun unlockFor(text: String, board: Board, candidates: List<String>): Unlock? {
    if (ASK_LINE.containsMatchIn(text)) return Unlock.ASK_LINE
    if (ASK_MOVE.containsMatchIn(text)) return Unlock.ASK_MOVE
    if (HINT.containsMatchIn(text)) return Unlock.HINT
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
    var system: String = systemMessage(persona = persona, rating = rating)

    /** A player below 1400 hears no term of art in the templates ([JARGON]). */
    var plainWords: Boolean = rating != null && rating.second < PLAIN_BELOW

    /**
     * A player of 1800 and up hears no textbook rule: the strong simulated students listed about
     * 4.5 a session they already knew ("rooks belong on open files", 75 times).
     */
    var strongPlayer: Boolean = rating != null && rating.first >= STRONG_FROM

    /**
     * Run the claims check on the caller's thread (replay and evaluation) instead of a
     * worker: live it is threaded so a button press never stalls the pipeline.
     */
    var syncClaims = false

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
    private var reviewLines: List<String> = emptyList()

    /**
     * The causes of the session's mistakes, from the player model: the pattern is said when
     * one comes back ([onEvidence]). Counted in the order first seen, as a Counter keeps them.
     */
    private val causesSeen = LinkedHashMap<String, Int>()

    /** The game each cause's pattern was last said in: once a game after the first. */
    private val patternTold = HashMap<String, String?>()

    /**
     * The record's question-answerer, set by whoever owns the store; the answers it gave,
     * with their queries, for the panel's log.
     */
    var profileAsk: ((String) -> ProfileAnswer)? = null
    val profileAnswers = ArrayList<Pair<String, Any?>>()

    /** The player's mistakes this game, for [ELICIT_EVERY]. */
    private var mistakes = 0

    /** When the coach last asked what the player saw, and whether they answered. */
    private var elicitAsked: Double? = null
    private var elicitAnswered = false

    /**
     * A question asked before the engine had looked at the position on the board: it is
     * answered the moment the analysis lands (see [ask]).
     */
    private var pendingAsk: Pair<String, Double>? = null

    // Lifecycle

    fun start() = speaker.start()

    fun close() = speaker.close()

    val userColour: Boolean get() = if (userSide == Pieces.Side.WHITE) WHITE else !WHITE

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
                conversation.ended(event)
            }

            is Recommendation -> onRecommendation(event, ts)
            is PuzzleOutcome -> onPuzzleOutcome(event, ts)
            is GameReview -> onReview(event, ts)
            is ProfileEvidence -> onEvidence(event, ts)
            is ProfileSnapshot -> profile = event
            is ScreenStateChanged -> {
                val covered = event.state == ScreenState.COVERED
                if (covered && !policy.covered) conversation.covered()
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
        val level = position.disclosure.apply(Unlock.STUCK) ?: position.disclosure.level
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
        if (profileAsk != null && ASK_PROFILE.containsMatchIn(text)) {
            askProfile(text, ts)
            return
        }
        val elicitAt = elicitAsked
        if (elicitAt != null && ts - elicitAt <= ELICIT_ANSWER_S) elicitAnswered = true
        var position = current
        if (position != null && position.facts == null && userToMove && !finished) {
            // The engine is still looking at the position on the board: said so, and answered
            // the moment its analysis lands -- never from an older position's (Tai, 2026-09-24:
            // a hint pressed as a puzzle appeared got "I have not seen enough of this game yet
            // to say. Give me a move or two.").
            if (pendingAsk == null) {
                queuePlain(Trigger.USER_ASK, "One moment -- still looking at this position.", ts)
            }
            pendingAsk = text to ts
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
        if (unlock != null) position.disclosure.apply(unlock)
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
        // The quick answer, shown before the model has started: the template for a bare
        // "hint", the move or the line. A question about the plan earns the hint's level but
        // not its template -- "What is your opponent's most dangerous idea?" as the first of
        // two answers to "what should I be aiming for?" was noise.
        val canned = (
            unlock == Unlock.ASK_MOVE || unlock == Unlock.ASK_LINE ||
                ASK_HINT.matchesAt(text, 0) ||
                // The first hint is now the lesson a question about the plan asks for (the
                // stakes, the key fact, the principle), no longer "what is your opponent's most
                // dangerous idea?": it goes out at once, as for a bare hint.
                ASK_AIM.containsMatchIn(text)
            ) && !mayCarryClaims(text)
        if (canned) {
            val quick = instruction(
                Trigger.USER_ASK, where, level, text, emptyList(), null, null
            ).second
            takeLeads(where, quick)
            queuePlain(
                Trigger.USER_ASK,
                quick,
                ts,
                ply = where.ply,
                level = level,
                annotations = annotationsFor(facts, level, where.board, trigger = Trigger.USER_ASK),
                fen = where.board.fen(),
                now = true,
                recordNow = true,
            )
        }

        val asked = questionPart(text)
        var shownExamination = false

        /**
         * The engine on what they asked about; its plain-words form goes out at once as the
         * coach's turn (correct by construction), and the model builds on it. Only what a
         * question names is said: named in a statement ("I see the fork now: Nf6 attacks the
         * queen on e8 and the rook on d7") or before the question ("my rook on e1 stays active.
         * Why is 12...h5 their best reply?"), the pieces and squares came back as the whole
         * answer -- where each can go, what each attacks -- and the question went unanswered
         * (the simulated students, 2026-09-26). Named in a statement they are still examined,
         * for the model to check it by.
         */
        fun examined(): List<String> {
            val found = namedAnalysis(asked.ifEmpty { text }, where, level)
            // A probe the planner made of their words that is no legal move is the model's to
            // know, not the player's to hear: "after pass, Kd1 is not a legal move" was read
            // out as the answer to "How does Qc7+ prevent Re8?" (the students).
            var said = found.filter { "is not a legal move" !in it }
            // A move they propose in a statement ("I'll play Qd1, blocking the check") is
            // examined aloud as a question about it would be: that is the answer. Without it the
            // templates said the hint that had asked for it again, word for word, three times
            // (the students' round 5); pieces and squares named in passing stay unsaid. With a
            // model to answer in words, a piece's or a square's inventory said first is noise
            // ("Your king on g8 can go to f7, g7, h8; ..." before the AI's own good answer to
            // "why doesn't the check prevent that capture?", 2026-09-26): only a move's
            // examination goes out. Offline it is the answer, and stays.
            val offline = speaker.provider is OfflineProvider
            if (asked.isEmpty() || !offline) said = said.filter { "the expected score is" in it }
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
                )
            }
            return found
        }

        val checker = claims
        if (checker == null || !mayCarryClaims(text)) {
            var verdicts: List<String> = emptyList()
            if (mayCarryClaims(text)) verdicts = examined()
            speak(
                Trigger.USER_ASK,
                where,
                level,
                ts,
                userText = text,
                verdicts = verdicts,
                quick = canned || shownExamination,
            )
            return
        }

        fun checked() {
            var verdicts = try {
                checker.check(text, where.board, where.facts)
            } catch (_: Exception) {
                emptyList()
            }
            // The engine on what they asked about, examined now and spoken first.
            val found = examined()
            verdicts = verdicts + found
            speak(
                Trigger.USER_ASK,
                where,
                level,
                ts,
                userText = text,
                verdicts = verdicts,
                quick = shownExamination,
            )
        }

        if (syncClaims) {
            checked() // replay and evaluation: the request must exist when ask() returns
        } else {
            Thread({ checked() }, "sidecoach-claims").apply { isDaemon = true }.start()
        }
    }

    /**
     * A question about the player over their games: answered from the record by the
     * profile's own model loop, off this thread (several model calls), and spoken as the
     * coach's turn when it lands. The queries it ran are kept for the log.
     */
    private fun askProfile(text: String, ts: Double) {
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
            queuePlain(Trigger.USER_ASK, answer.text, ts, recordNow = true)
        }

        Thread({ run() }, "profile-ask").apply { isDaemon = true }.start()
    }

    fun hint(ts: Double) = ask("hint", ts)

    fun askMove(ts: Double) = ask("what should I play", ts)

    fun askLine(ts: Double) = ask("show me the line", ts)

    fun poll(): List<CoachMessage> {
        val out = ArrayList<CoachMessage>()
        for (one in speaker.poll()) {
            spoken.add(one)
            if (one.message.text.isEmpty()) {
                // A failed model reply with nothing to fall back on: kept for the log (its
                // calls say why), said to no one.
                continue
            }
            if (!one.recorded) {
                synchronized(lock) {
                    conversation.coachSaid(
                        one.message.text,
                        nod = one.message.trigger == Trigger.POST_MOVE,
                    )
                }
            }
            out.add(one.message)
        }
        return out
    }

    // Internals

    private fun newGame(event: GameStarted, ts: Double) {
        gameId = event.gameId
        userSide = event.userSide
        kind = event.kind
        board = Board(event.startFen)
        positions = linkedMapOf(0 to CoachPosition(ply = 0, board = board.copy()))
        moves = LinkedHashMap()
        conversation.newGame(event, profile = profileLines())
        mistakes = 0
        elicitAsked = null
        elicitAnswered = false
        pendingAsk = null
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
     * A mistake of the player's placed by the player model. When its cause is behind an earlier
     * mistake of the session's too, the pattern is said, with the habit that answers it, right
     * after the lesson on the move -- the second time, and after that at most once in each game:
     * said every second time it came to "10 times now" in one session (the AI's sessions,
     * 2026-09-26). Each lesson had taught its own moment and a principle of its own, and the
     * threat-blind student's notebook held five unrelated rules, none of them the habit. A player
     * of 1800 and up hears the pattern without the textbook habit. Said from the model's cause,
     * no model call.
     */
    private fun onEvidence(event: ProfileEvidence, ts: Double) {
        val cause = event.cause
        if (event.gameId != gameId || cause == null || !Causes.HABIT.containsKey(cause)) return
        val n = (causesSeen[cause] ?: 0) + 1
        causesSeen[cause] = n
        if (n < 2 || (patternTold.containsKey(cause) && patternTold[cause] == gameId)) return
        patternTold[cause] = gameId
        val (words, habit) = Causes.HABIT.getValue(cause)
        val times = when (n) {
            2 -> "twice"
            3 -> "three times"
            else -> "$n times"
        }
        val text = if (strongPlayer) {
            bags.draw(
                "pattern-strong:$cause",
                listOf(
                    "That is $times now: $words.",
                    "Notice the pattern -- $words, $times now.",
                ),
            )
        } else {
            bags.draw(
                "pattern:$cause",
                listOf(
                    "That is $times now: $words. The habit to build: $habit.",
                    "This keeps happening -- $words, $times now. The habit that fixes it: $habit.",
                    "Notice the pattern: $words, $times now. So, $habit.",
                ),
            )
        }
        synchronized(lock) { conversation.note("The pattern, told to the player: $text") }
        queuePlain(Trigger.POST_MISTAKE, text, ts, recordNow = true)
    }

    /**
     * The practice the profile picked after the review: a line naming the motif and the
     * puzzles, on the card; no model call, the card carries the links.
     */
    private fun onRecommendation(event: Recommendation, ts: Double) {
        if (event.gameId != gameId || event.puzzles.isEmpty()) return
        val n = event.puzzles.size
        val text = "For practice: $n ${event.motifWords} puzzle${if (n != 1) "s" else ""} around " +
            "${event.level} are on the review card -- ${event.reason}."
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
        conversation.clock(event.whiteMs, event.blackMs)
        val seen = listOfNotNull(event.whiteMs, event.blackMs)
        if (seen.isEmpty()) return
        val peak = seen.max()
        clockPeakMs = clockPeakMs?.let { maxOf(it, peak) } ?: peak
        // The clock starts at its highest, so the peak seen approximates the initial time and
        // thus the control's band; increment slack does not cross the coarse bands.
        policy.timeControl = timeControlOf(clockPeakMs)
    }

    /**
     * The user's recent think time, in seconds: the median of the last few moves, or null
     * before there are two to go by.
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
        if (event.byUser) pendingAsk = null // they played on: the question is answered by the move
        board = boardAfter(board, event)
        moves[event.ply] = event
        conversation.move(event)
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
            speaker.lastError = "summary: ${exc::class.java.simpleName}: ${exc.message}"
        }
    }

    /**
     * Moves withdrawn: the conversation says so, and the game goes on from the position the
     * board is back at; the replacement moves arrive as ordinary moves.
     */
    private fun onTakeback(event: Takeback, ts: Double) {
        conversation.takeback(event)
        board = boardBack(board, event.fen)
        val active = conversation.activeRecords()
        val ply = if (active.isNotEmpty()) active.last().move.ply else 0
        for (gone in positions.keys.filter { it > ply }) positions.remove(gone)
        // The withdrawn moves go too: a late word on one was taken for the move that replaced it.
        for (gone in moves.keys.filter { it > ply }) moves.remove(gone)
        var position = positions[ply]
        if (position == null) {
            position = CoachPosition(ply = ply, board = board.copy())
            positions[ply] = position
        }
        current = position
        policy.newPosition(ply, ts, userToMove = userToMove)
    }

    /** The board was re-read: the position jumped; the move numbering restarts. */
    private fun onResync(event: Resync, ts: Double) {
        board = Board(event.fen)
        conversation.resync(event, board)
        positions = linkedMapOf(0 to CoachPosition(ply = 0, board = board.copy()))
        moves = LinkedHashMap()
        current = positions[0]
        pendingAsk = null
        policy.restart()
        policy.newPosition(0, ts, userToMove = userToMove)
    }

    private fun onAnalysis(event: Analysis, ts: Double) {
        if (event.gameId != gameId) return // the engine's late word on the game before this one
        val position = positions[event.ply] ?: return
        if (position.board.fen() != event.fen) return
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
        conversation.analysed(event, position.facts!!)
        if (heldLesson?.played?.ply == event.ply) releaseLesson(ts) // what the move allowed is known now
        val pending = pendingAsk
        if (position === current && pending != null) {
            val (text, _) = pending
            pendingAsk = null
            ask(text, ts)
            return
        }
        if (position === current && userToMove) {
            val decisive = event.decisive &&
                !obviousRecapture(position, event) &&
                !carryingOut(position)
            val trigger = policy.onPositionAnalysed(event.ply, decisive = decisive)
            if (trigger != null) {
                val level = position.disclosure.apply(Unlock.DECISIVE) ?: position.disclosure.level
                policy.said(position.ply, ts)
                speak(trigger, position, level, ts)
            }
        }
    }

    /**
     * The player's last move and the reply to it are the first two moves of the engine's line
     * from a decisive position: they found the idea and are playing it out, so a question about
     * the next step asks what they already know (Tai's desktop, 2026-09-24: after Qd1+ Rxd1
     * Rxd1+ Qe1, "You can mate in one move ... Which check leaves their king no square at
     * all?", heard as the puzzle's hint). An opponent who leaves the line keeps the question.
     */
    private fun carryingOut(position: CoachPosition): Boolean {
        val own = moves[position.ply - 1]
        val reply = moves[position.ply]
        val before = positions[position.ply - 2]
        if (own == null || reply == null || !own.byUser || before == null) return false
        val analysis = before.analysis
        if (analysis == null || !analysis.decisive || analysis.lines.isEmpty()) return false
        return analysis.lines[0].moves.take(2) == listOf(own.uci, reply.uci)
    }

    /**
     * The opponent just took something and the engine's move takes back on the same square:
     * the lines diverge because everything else drops a piece, not because there is anything
     * to find, so the decisive question stays unasked (a book recapture drew "This is a
     * moment to slow down", offline, 2026-09-21). A recapture that is not the best move -- a
     * zwischenzug on -- keeps the question.
     */
    private fun obviousRecapture(position: CoachPosition, event: Analysis): Boolean {
        val last = moves[position.ply] ?: return false
        val before = positions[position.ply - 1] ?: return false
        if (event.lines.isEmpty() || event.lines[0].moves.isEmpty()) return false
        val taken = Move.fromUciOrNull(last.uci) ?: return false
        val best = Move.fromUciOrNull(event.lines[0].moves[0]) ?: return false
        return before.board.isCapture(taken) &&
            best.to == taken.to &&
            position.board.isCapture(best)
    }

    private fun onAssessment(event: MoveAssessment, ts: Double) {
        if (event.gameId != gameId) return // the engine's late word on the game before this one
        val played = moves[event.ply]
        // The verdict on a move taken back since, not on the one played there now.
        if (played != null && played.uci != event.uci) return
        conversation.assess(event)
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
                    taught = taught.toList(),
                    bags = bags,
                    plain = plainWords,
                    strong = strongPlayer,
                )
                if (kind != null) {
                    taught.addLast(kind)
                    if (taught.size > TAUGHT_MEMORY) taught.removeFirst()
                }
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
    private fun elicitDue(ts: Double): Boolean {
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
            speaker.provider = provider
            speaker.routineProvider = routineProvider ?: provider
            speaker.verifier = verifier
            this.summariser = summariser ?: provider
            this.claims = claims
        }
    }

    /**
     * What the coach has said over the last [RECENT_PLIES] plies, as one text. The entries are
     * copied under the coach's lock: a summary folds them on its own thread (Python's GIL makes
     * the desktop's `list(...)` of them whole).
     */
    private fun recentlySaid(): String {
        val texts = ArrayList<String>()
        var plies = 0
        val entries = synchronized(lock) { conversation.entries.toList() }
        for (entry in entries.asReversed()) {
            if (entry.kind == "ply") {
                plies += 1
                if (plies > RECENT_PLIES) break
            } else if (entry.kind == "coach") {
                texts.add(entry.text)
            }
        }
        return texts.joinToString(" ")
    }

    /**
     * The first hint's wordings taken from their bags once the hint is said: this turn's template
     * is the position's first hint, or holds it.
     */
    private fun takeLeads(position: CoachPosition, template: String) {
        val situation = position.facts?.situation
        if (!situation.isNullOrEmpty() && situation in template) {
            for ((key, wording) in position.leads) bags.take(key, wording)
            position.leads = emptyMap()
        }
    }

    /**
     * [lesson], opening with [opener], written with the next rule-to-keep lead from its bag, less
     * what was said lately ([fresh]). The lead is taken from the bag only when the lesson still
     * says it: one with no rule to keep, or whose rule was said lately, leaves it for the next.
     */
    private fun freshLesson(opener: String, lesson: (String) -> String): String {
        val lead = bags.peek("keep", KEEP_LEADS)
        val text = fresh(lesson(lead), recentlySaid(), opener)
        if ("$lead:" in text.lowercase()) bags.take("keep", lead)
        return text
    }

    /** The opening the game was in: the last name the book gave. */
    private fun openingSoFar(): String? {
        synchronized(lock) {
            for (record in conversation.records.asReversed()) {
                val opening = record.analysis?.opening
                if (!opening.isNullOrEmpty()) return opening
            }
        }
        return null
    }

    /**
     * The engine's word on what the player asked about: the model decides what to examine,
     * the engine examines it; one line per probe, none when nothing was named. A piece or
     * square named needs no engine; a line does.
     */
    private fun namedAnalysis(
        text: String,
        position: CoachPosition,
        level: DisclosureLevel,
    ): List<String> = try {
        // Without an engine only pieces and squares can be examined, and the words name
        // those well enough: no planner call.
        val planned = if (engine != null) {
            probesFromPlan(planProbeJson(text, position.board), text, position.board)
        } else {
            null
        }
        analyseNamed(
            text,
            position.board,
            engine,
            userColour = userColour,
            level = level,
            plannedProbes = planned,
            // Pieces and squares as the board stands now, whatever position the question is
            // about.
            piecesBoard = board,
        )
    } catch (exc: Exception) {
        speaker.lastError = "named analysis: ${exc::class.java.simpleName}: ${exc.message}"
        emptyList()
    }

    /** What the coach has said since the last move was played. */
    private fun saidSinceMove(): List<String> {
        val entries = synchronized(lock) { conversation.entries.toList() }
        val out = ArrayList<String>()
        for (entry in entries.asReversed()) {
            if (entry.kind == "ply") break
            if (entry.kind == "coach") out.add(entry.text)
        }
        return out
    }

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
     * The planner's reply for [probesFromPlan]. The desktop's `plan_probes` makes this call
     * inside `analyse_named`; here the provider belongs to the caller, so the call is here.
     */
    private fun planProbeJson(text: String, board: Board): String? {
        val side = if (board.turn == WHITE) "White" else "Black"
        val player = if (userColour == WHITE) "White" else "Black"
        val legal = board.legalMoves().map { board.san(it) }.sorted().joinToString(" ")
        val content = "Position (FEN): ${board.fen()}\n$side to move. The player is $player.\n" +
            "Legal moves: $legal\n\nThe player says: \"${text.trim()}\"\n\nJSON only."
        return try {
            // Parsing their words into moves is no lesson: the routine effort, not the coach's
            // (at medium it was a second model call's worth of waiting).
            speaker.routineProvider.chat(
                PLAN_SYSTEM,
                listOf(ChatMessage("user", content)),
                maxTokens = 400,
                temperature = 0.0,
            ).text
        } catch (_: Exception) {
            null
        }
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
     */
    private fun askedOwnMove(text: String): PlyRecord? {
        if (!ASK_LAST_MOVE.containsMatchIn(text)) return null
        return lastOwnMove(mine = MY_MOVE.containsMatchIn(text))
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
        if (userToMove && level.value < DisclosureLevel.MOVE.value) return now
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
        val good = listOf("brilliant", "great", "best", "excellent").sumOf { counts[it] ?: 0 }
        var text = "$good of your ${judged.size} moves were best or excellent."
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

    private fun speak(
        trigger: Trigger,
        position: CoachPosition,
        level: DisclosureLevel,
        ts: Double,
        userText: String? = null,
        verdicts: List<String> = emptyList(),
        assessment: MoveAssessment? = null,
        played: MovePlayed? = null,
        quick: Boolean = false,
        candidates: Int? = null,
        routine: Boolean = false,
        elicit: Boolean = false,
    ) {
        val facts = position.facts!!
        var (instructionText, fallback) = instruction(
            trigger, position, level, userText, verdicts, assessment, played,
            quick = quick, elicit = elicit,
        )
        takeLeads(position, fallback)
        var mustName = mustName(trigger, position, level, fallback, assessment, played, userText)
        if (quick) {
            // The template is already out as the quick answer; a model reply that fails the
            // checks adds nothing, rather than the same line a second time.
            fallback = ""
            mustName = emptyList()
        } else if (trigger == Trigger.USER_ASK && fallback.isNotEmpty() && fallback in saidSinceMove()) {
            // Nor is anything the coach said since the last move said again word for word: a
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
        val line = levelLine(level, position, if (facing) renderLevel else null, past = past)
        var directive = "$line$instructionText"
        var faced = ""
        var frames: List<Board> = emptyList()
        var moved = if (trigger == Trigger.POST_MOVE || trigger == Trigger.POST_MISTAKE) played else null
        if (moved == null && trigger == Trigger.USER_ASK) {
            val own = askedOwnMove(userText ?: "")
            if (own != null && own.move.ply == position.ply + 1) moved = own.move
        }
        if (moved != null) {
            // The lesson after a move is taught from the position the player faced, which the
            // record keeps as one line: its sheet goes with the request, and the template's
            // lesson as the floor (parity pass, 2026-09-25).
            val sheet = synchronized(lock) { conversation.facedText(position.facts!!, position.board) }
            if (sheet.isNotEmpty()) faced = "\n\n# The position they faced, before ${moved.san}\n$sheet"
            if (mustName.isNotEmpty() && fallback.isNotEmpty()) {
                directive += "\n\nThe lesson in the board's own words, all of it true -- your reply says at " +
                    "least this, in your own words, and may say more: $fallback"
            }
            frames = framesAfter(position, moved)
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
        if (records.isNotEmpty()) frames = frames + Board(records.last().move.fenBefore)
        val messages: List<ChatMessage>
        synchronized(lock) {
            conversation.level = renderLevel
            val harness = conversation.render()
            messages = listOf(ChatMessage("user", "$harness$faced$earlier\n\n# Task\n$directive"))
        }
        var unlisted: Set<String>? = null
        val earlierMoves = before
        if (earlierMoves != null) {
            var shown = GAME_SO_FAR.replace(messages[0].content, "")
            for (constant in listOf(NO_NUMBERS, NOT_THE_MOVE, TIE_LISTED, NO_EARLIER)) {
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
            gameId = gameId ?: "",
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
            // "Why was that bad?": the lesson on their last move, as after it.
            if (own.move.ply != position.ply + 1) return emptyList()
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
            return facts.threats.filter { names(template, it) }
        }
        val hint = trigger == Trigger.DECISIVE_QUESTION || trigger == Trigger.STUCK || (
            trigger == Trigger.USER_ASK && (ASK_HINT.matchesAt(said, 0) || ASK_AIM.containsMatchIn(said))
            )
        if (!hint) return emptyList()
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
                    if (move != null) out.add(Board(r.move.fenBefore).san(move))
                }
            }
        }
        return out
    }

    /**
     * What may be said about the position the player faces, stated on the directive; with
     * [nowLevel], the reply is about a position already moved from, and the one on the board has
     * its own level.
     */
    private fun levelLine(
        level: DisclosureLevel,
        position: CoachPosition,
        nowLevel: DisclosureLevel? = null,
        past: Boolean = false,
    ): String {
        if (position.board.isGameOver()) return ""
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
        return "Disclosure for the position the player faces: ${words(level)}.\n\n"
    }

    /**
     * A message that needs no model; it queues behind whatever the model is writing, unless
     * [now], when it goes out at once. With [recordNow] it is a turn of the conversation from
     * this moment, so a request built next already shows it.
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
    ) {
        val message = CoachMessage(
            text = text,
            level = level,
            trigger = trigger,
            gameId = gameId,
            ply = ply,
            fen = fen,
            annotations = annotations ?: emptyList(),
        )
        if (recordNow) synchronized(lock) { conversation.coachSaid(text) }
        speaker.submitReady(
            Spoken(
                message = message,
                level = level,
                problems = emptyList(),
                attempts = 0,
                recorded = recordNow,
            ),
            now = now,
        )
    }

    /**
     * The turn's instruction to the model, and the template used if it fails.
     *
     * [quick] means the template has already been shown as an immediate answer, so the model
     * is asked to build on it rather than repeat it. [elicit] ends a mistake's lesson by
     * asking what the player saw (see [elicitDue]).
     */
    internal fun instruction(
        trigger: Trigger,
        position: CoachPosition,
        level: DisclosureLevel,
        userText: String?,
        verdicts: List<String>,
        assessment: MoveAssessment?,
        played: MovePlayed?,
        quick: Boolean = false,
        elicit: Boolean = false,
    ): Pair<String, String> {
        val facts = position.facts!!

        if (trigger == Trigger.REVIEW) {
            return (
                "The game is over. Sum it up from the review in the last message and the " +
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
            return "${ideaDirective(facts)} $NO_FIGURES" to ideaTemplate(facts)
        }
        if (trigger == Trigger.STUCK) {
            val directive =
                if (level.value >= DisclosureLevel.FOCUS.value) FOCUS_DIRECTIVE else ideaDirective(facts)
            return (
                "The player has been thinking for a while and is stuck. " +
                    directive +
                    " Take a different angle from anything already said about this position. " +
                    NO_FIGURES
                ) to hintTemplate(facts, level)
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
                val left = positions[played.ply]
                val leftAnalysis = left?.analysis
                val allowed = if (left != null && leftAnalysis != null) {
                    IdeaWords.allowed(left.board, leftAnalysis, played.san)
                } else {
                    null
                }
                val opener = bags.draw("cost", COST_YOU).replace("{best}", best)
                fallback = freshLesson(opener) { lead ->
                    lessonTemplate(
                        facts,
                        best,
                        opener,
                        board = position.board,
                        played = played.uci,
                        allowed = allowed,
                        keepLead = lead,
                        strong = strongPlayer,
                    )
                }
                instructionText =
                    "The player just played ${played.san}, which was ${rated(label)}: it cost " +
                        "${percent(loss)} of the game. The better move was $best ($betterWords). " +
                        "Teach the difference from the fact sheet of the position they faced -- " +
                        "its goal, its pattern and the tactic when the sheet names one, why the " +
                        "better move works, and what their move missed or allowed -- as a pattern " +
                        "to remember, in three sentences at " +
                        "most, then ask one question that makes them find on the board something " +
                        "your reply has not said. " +
                        "$TIE_TO_THE_GAME $NOT_THE_MOVE $NO_NUMBERS"
                if (!allowed.isNullOrEmpty()) {
                    instructionText += " What ${played.san} allowed, from the engine's line: $allowed."
                }
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
                            "($betterWords). Teach the difference from the fact sheet of the " +
                            "position they faced -- its goal, its pattern, why the better move " +
                            "works -- as a pattern to remember, in three sentences at most, then " +
                            "-- instead of a check question -- ask, in one short question of your " +
                            "own, what they expected their move to do. $NOT_THE_MOVE $NO_NUMBERS"
                }
            } else if (isExciting(assessment.label)) {
                // Rare and worth it: the board's own sentence stands in, the model says what
                // makes it the move, with some delight.
                fallback = verdict(
                    position.board,
                    Move.fromUci(played.uci),
                    label,
                    assessment.tags,
                    phase = facts.phase,
                    threats = facts.threats,
                    taught = taught.toList(),
                    bags = bags,
                    plain = plainWords,
                    strong = strongPlayer,
                ).first
                val meaning = when (label) {
                    "brilliant" -> "a sound sacrifice"
                    else -> "the one move that keeps the advantage, or the chance taken"
                }
                instructionText =
                    "The player just played ${played.san}: $label -- $meaning. Show some " +
                        "delight, in your own voice, in one or two sentences: name what makes it " +
                        "the move -- the chance taken, the only move, the sacrifice that works, " +
                        "the idea it serves -- from the messages above, nothing invented. No " +
                        "question, no advice, and not the same words as your last such sentence. " +
                        "$NOT_THE_MOVE $NO_NUMBERS"
            } else if (routine) {
                fallback = verdictWord(label)
                val meaning = when (label) {
                    "book" -> "opening theory"
                    "brilliant" -> "a sound sacrifice"
                    "great" -> "the one move that keeps the advantage"
                    "best" -> "the engine's own choice"
                    "excellent" -> "as strong as the best move"
                    "good" -> "a sound move, a little behind the best"
                    else -> "a sound move"
                }
                instructionText =
                    "The player just played ${played.san}: $label -- $meaning. One short " +
                        "sentence, fifteen words at most, in your own voice: a word of praise, and " +
                        "the idea behind the move as a lesson to keep when a message gives one -- " +
                        "the plan, the tactic. No question, no lecture, no advice, no guess at what " +
                        "it allows or risks, and not the same words as your last such sentence. " +
                        "$NOT_THE_MOVE $NO_NUMBERS"
            } else if (assessment.severity == Severity.MISTAKE || assessment.severity == Severity.BLUNDER ||
                label == "miss"
            ) {
                // A mistake in blitz: the policy keeps it to the word after the move (no
                // lesson), but the word says what the move was -- never "fine".
                fallback = bags.draw("cost", COST_YOU).replace("{best}", best) + "."
                instructionText =
                    "The player just played ${played.san}, which was ${rated(label)}: $best " +
                        "($betterWords) was better. It is a fast game: one or two sentences, no " +
                        "lesson, no question. $NOT_THE_MOVE $NO_NUMBERS"
            } else {
                // The better move and what it does, when the sheet has it, is the lesson.
                val opener = bags.draw("fine", FINE_TIGHTER).replace("{best}", best)
                fallback = freshLesson(opener) { lead ->
                    lessonTemplate(
                        facts,
                        best,
                        opener,
                        board = position.board,
                        played = played.uci,
                        keepLead = lead,
                        strong = strongPlayer,
                    )
                }
                instructionText =
                    "The player just played ${played.san}, which was ${rated(label)}: " +
                        "playable, not the strongest -- $best ($betterWords) was better, by a " +
                        "little. One or two sentences: how it rated, in plain words, then the " +
                        "better move, what it does, and the idea behind it as a lesson to keep. No " +
                        "lecture, no question. " +
                        "$TIE_TO_THE_GAME $NOT_THE_MOVE $NO_NUMBERS"
            }
            return instructionText to fallback
        }
        // An explicit ask. The player's words are already the last message of the
        // conversation; the directive says only how to answer them.
        val said = userText ?: "hint"
        var checks = ""
        val claimsChecked = verdicts.filter { !it.startsWith("Engine on ") }
        val engineWords = verdicts.filter { it.startsWith("Engine on ") }
        if (claimsChecked.isNotEmpty()) {
            checks = "Their claims, checked against the board:\n- " +
                claimsChecked.joinToString("\n- ") + "\n\n"
        }
        if (engineWords.isNotEmpty()) {
            // The engine on what they asked about: theirs to hear about in full, and already
            // said out loud as the coach's last turn when `quick`.
            checks += engineWords.joinToString("\n") + "\n\n"
        }
        // The rung's template: two hints, then the answer; the line when asked for it.
        var fallback = hintTemplate(facts, level)
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
        // A question about a fact that is never withheld -- the material, the opening, who
        // stands better -- falls back to that fact, not to the hint for the position.
        val material = if (ASK_MATERIAL.containsMatchIn(said) || ASK_EVAL.containsMatchIn(said)) {
            materialWords(level, facts)
        } else {
            null
        }
        if (!material.isNullOrEmpty() && ASK_MATERIAL.containsMatchIn(said)) {
            fallback = "Material: $material."
        } else if (ASK_OPENING.containsMatchIn(said)) {
            val opening = openingSoFar()
            fallback = if (opening != null) {
                "The opening was the $opening."
            } else {
                "We left the opening book early."
            }
        } else if (ASK_EVAL.containsMatchIn(said)) {
            fallback = standingTemplate(position, material)
        }
        val shown = " Your last turn above is the quick answer they have already been shown; " +
            "build on it, do not repeat it."
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
        val followup = !answering && saidSinceMove().isNotEmpty()
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
        val directive: String
        val lastOpponent = lastOpponentMove()
        val lastOwn = askedOwnMove(said)
        if (ASK_REVIEW.containsMatchIn(said)) {
            // An overall verdict comes from the tally in "Game so far", never from an
            // impression of the last position: the first review told a player a queen up that
            // they had been a queen down.
            directive =
                "They asked how they did overall. Answer from the engine's verdicts on their " +
                    "moves in the messages above -- the labels, the tally in their last move's " +
                    "message, where they lost ground, how the material went -- and from nothing " +
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
            val hinted = when {
                level.value >= DisclosureLevel.MOVE.value -> aim + MOVE_DIRECTIVE
                level.value >= DisclosureLevel.FOCUS.value ->
                    aim.ifEmpty { "They asked for another hint. " } + FOCUS_DIRECTIVE
                else -> aim.ifEmpty { "They asked for a hint. " } + ideaDirective(facts)
            } + " " + NO_FIGURES
            directive = if (quick && level.value < DisclosureLevel.MOVE.value) {
                // The hint is out already, in the template's words: the model adds to it, in a
                // sentence or two -- not the whole hint again, which the voice talked over the
                // template with (the runtime audit, 2026-09-25).
                "They asked for a hint and have just been shown it -- your last turn above. " +
                    "In one or two sentences, build on it: a sharper angle on the same key fact " +
                    "-- a piece or square of it, and what it does, threatens or allows, with the " +
                    "principle behind it in a clause -- then the one question that sets them " +
                    "looking. Do not repeat it, and give nothing the next hint would. Do not " +
                    "comment on anything said earlier. " + NO_FIGURES
            } else {
                hinted + " Do not comment on anything said earlier." + (if (quick) shown else "")
            }
        } else if (level.value >= DisclosureLevel.LINE.value &&
            ASK_LINE.containsMatchIn(said) && facts.line.isNotEmpty()
        ) {
            directive = if (quick) {
                "They asked to see the line and have just been shown it: " +
                    "${facts.line.joinToString(" ")}. Give one or two short sentences on the " +
                    "idea behind it. Do not answer with a question."
            } else {
                "They asked to see the line: give it -- ${facts.line.joinToString(" ")} -- " +
                    "plainly, then one short sentence on the idea behind it. Do not answer " +
                    "with a question."
            }
        } else if (level.value >= DisclosureLevel.MOVE.value &&
            ASK_MOVE.containsMatchIn(said) && facts.candidates.isNotEmpty()
        ) {
            directive = if (quick) {
                "They asked what to play and have just been shown that it is " +
                    "${facts.candidates[0]}, with what it does. Build on it in one or two " +
                    "sentences: why it works, from the fact sheet's why and pattern, and the " +
                    "idea to keep. Do not answer with a question this time."
            } else {
                "They asked what to play: tell them plainly it is ${facts.candidates[0]}, " +
                    "then why it works, from the fact sheet's why and pattern, correctly. Do " +
                    "not answer with a question this time."
            }
        } else if (engineWords.isNotEmpty() && quick) {
            // The examination is out already; the coach adds the lesson in it, not the
            // numbers again.
            directive =
                "They asked about a move or line and have just been shown the engine's word " +
                    "on it -- your last turn above. Build on it in one or two sentences: what " +
                    "their move does or allows, and what that means for their idea, in plain " +
                    "chess words, saying only what that analysis and the messages state." +
                    (
                        if (followup) {
                            " You have already told them the verdict on their move and the better one: " +
                                "do not repeat either, and ask nothing."
                        } else {
                            " Then one question that points them at the idea."
                        }
                        ) +
                    " Name no move the messages do not give. " +
                    NO_FIGURES
            fallback = ""
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
                best = if (move != null) Board(lastOwn.move.fenBefore).san(move) else ""
            }
            if (best == san) best = ""
            // The request's facts are those of the position the move was played from when the
            // coach has them (ask() chose that position); the idea is theirs. The theme or the
            // plan, never a static feature: "the idea was the isolated d-pawn ..." became the
            // reason a bishop retreat was an inaccuracy.
            val idea = facts.theme ?: facts.plan ?: ""
            if (own.severity == Severity.NONE && best.isNotEmpty() && loss >= 0.005) {
                // Sound but not the best: said as the word after it said it ("Fine, but a3 was a
                // little tighter"), never "it cost nothing" (the runtime audit).
                val opener = bags.draw("fine", FINE_TIGHTER).replace("{best}", best)
                fallback = freshLesson(opener) { lead ->
                    lessonTemplate(
                        facts,
                        best,
                        opener,
                        played = lastOwn.move.uci,
                        keepLead = lead,
                        strong = strongPlayer,
                    )
                }
                directive =
                    "They ask about the move they already played, $san. It was $label: sound, " +
                        "a little behind $best. Say so plainly -- it was fine, then the better move " +
                        "and the idea behind it. Two sentences at most, no question."
            } else if (own.severity == Severity.NONE) {
                fallback = verdictWord(label)
                directive =
                    "They ask about the move they already played, $san. It was not bad: the " +
                        "verdict was $label, and it cost nothing. Tell them so plainly, in one " +
                        "or two sentences, and do not invent a fault. No question."
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
                if (best.isNotEmpty()) {
                    // The better move played out and the pattern to keep, as the lesson says it
                    // -- not "the idea was a pin" and nothing more.
                    // Asked about the move just taught, the lesson is not said twice.
                    val opener = "$best was better"
                    fallback = (if (fallback.isNotEmpty()) "$fallback " else "") + freshLesson(opener) { lead ->
                        lessonTemplate(
                            facts,
                            best,
                            opener,
                            board = if (faced) position.board else null,
                            played = lastOwn.move.uci,
                            allowed = allowed,
                            keepLead = lead,
                            strong = strongPlayer,
                        )
                    }
                } else if (!allowed.isNullOrEmpty()) {
                    fallback = (if (fallback.isNotEmpty()) "$fallback " else "") +
                        templateSentence(allowed[0].uppercaseChar() + allowed.substring(1))
                }
                if (fallback.isEmpty()) fallback = "That was ${rated(label)}."
                val reason = if (best.isNotEmpty()) "what $best does, and " else ""
                val cost = if (askedCost) {
                    " that cost ${percent(loss)} of the game -- they asked, so say it"
                } else {
                    " (what it cost is not theirs to hear unless they ask: no figures)"
                }
                directive =
                    "They ask about the move they already played, $san. Its verdict is in its " +
                        "move message and is theirs in full: it was ${rated(label)}$cost" +
                        (if (best.isNotEmpty()) ", $best was better" else "") +
                        (if (idea.isNotEmpty()) ", and the idea was $idea" else "") +
                        ". Say that plainly -- the position it was played from is past, so " +
                        "nothing about it is withheld; the disclosure level binds only the " +
                        "position they face now -- and say why, concretely: ${reason}what $san " +
                        "missed or allowed, naming the pieces, squares or moves, from the sheet of " +
                        "the position they faced (its pattern, and its tactic when it names one) -- " +
                        "and the pattern to keep." +
                        (if (!allowed.isNullOrEmpty()) " What $san allowed, from the engine's line: $allowed." else "") +
                        (
                            if (answering) {
                                "$close Three sentences at most."
                            } else if (followup) {
                                follow
                            } else {
                                " Three sentences at most, then one question that makes them find " +
                                    "on the board something your reply has not said."
                            }
                            ) +
                        " Name moves bare, never with a move number in front. " +
                        TIE_TO_THE_GAME
                if (followup && !asks) {
                    fallback = "" // a nod needs no template: nothing, rather than the lesson again
                }
            }
        } else if (ASK_MATERIAL.containsMatchIn(said) || ASK_OPENING.containsMatchIn(said) ||
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
            if (ASK_OPENING.containsMatchIn(said)) plainly += " The opening by the name the record gives it."
            if (!material.isNullOrEmpty()) {
                plainly += " The material is this, and your answer gives all of it, in these terms: " +
                    "$material. Never count the pieces yourself."
            }
            directive = plainly
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
                    "they list, at most two: the move and what it does, from 'what the other side " +
                    "threatens' -- or, when the facts list " +
                    "none, say plainly that nothing is threatened. One or two sentences, no question. " +
                    NO_FIGURES
        } else {
            directive =
                "Answer them directly and be honest about their own claim. If they proposed a " +
                    "move, or named a square or piece, you may say it back and tell them " +
                    "plainly whether it holds up, and what it does or allows, as the engine's word " +
                    "on it gives it -- when the engine does not rate their move, " +
                    "say it is not the strongest and, in general terms, why. When the engine's " +
                    "word on the move or line they named is given above, that analysis is " +
                    "theirs to hear in full: the score after it, how far behind the best, the " +
                    "reply, what the move does. Only the engine's better move or line is " +
                    "withheld: never reveal that unless the sheet lists it, but do point them " +
                    "toward the right idea" +
                    (if (answering || (followup && !asks)) " in words." else " with a question.") +
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
        return "$checks$directive" to fallback
    }
}
