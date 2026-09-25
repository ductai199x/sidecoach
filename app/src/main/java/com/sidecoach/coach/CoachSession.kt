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
    "\\b(?:how (?:did|have|am) i (?:do|play|done|played|playing|doing)|how was (?:my|the) game|" +
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

/** A bare request for a hint, with nothing claimed: "hint", "give me a hint", "a clue". */
private val ASK_HINT = Regex(
    "^\\W*(?:(?:give me |i need |can i have |i want |any |another |one more )?(?:a |an |some )?" +
        "(?:hint|clue|nudge|help|tip)s?(?: please)?)\\W*\$",
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

/** Whether the user's words could state something about the board to check. */
fun mayCarryClaims(text: String): Boolean {
    if (ASK_HINT.matchesAt(text, 0) || ASK_REVIEW.containsMatchIn(text)) return false
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
            is Takeback -> onTakeback(event, ts)
            is Resync -> onResync(event, ts)
            is GameEnded -> {
                current = null
                finished = true
                conversation.ended(event)
            }

            is Recommendation -> onRecommendation(event, ts)
            is PuzzleOutcome -> onPuzzleOutcome(event, ts)
            is GameReview -> onReview(event, ts)
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

    /** Time passing: a hint if the user has been stuck long enough. */
    fun tick(ts: Double) {
        val trigger = policy.tick(ts) ?: return
        val position = current ?: return
        if (position.facts == null) return
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
        val asked = elicitAsked
        if (asked != null && ts - asked <= ELICIT_ANSWER_S) elicitAnswered = true
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
        if (ASK_LAST_MOVE.containsMatchIn(text)) {
            val own = lastOwnMove()
            if (own != null) {
                // About the move they already played: spoken from the position it was played
                // from, with everything shown, as the lesson after a mistake is. That
                // position is past; the level in force binds only the one they face now.
                val before = positions[own.move.ply - 1]
                if (before?.facts != null) {
                    position = before
                    level = DisclosureLevel.LINE
                }
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
                ASK_HINT.matchesAt(text, 0)
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

        /**
         * The engine on what they asked about; its plain-words form goes out at once as the
         * coach's turn (correct by construction), and the model builds on it.
         */
        fun examined(): List<String> {
            val found = namedAnalysis(text, where, level)
            if (found.isNotEmpty()) {
                queuePlain(
                    Trigger.USER_ASK,
                    found.joinToString(" ") { spoken(it) },
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
                quick = canned || verdicts.isNotEmpty(),
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
                quick = found.isNotEmpty(),
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
        synchronized(lock) { conversation.review(event.lines) }
        val position = latestWithFacts()
        if (position == null || event.lines.isEmpty()) {
            queuePlain(Trigger.REVIEW, event.lines.take(4).joinToString(" "), ts, recordNow = true)
            return
        }
        reviewLines = event.lines.toList()
        speak(Trigger.REVIEW, position, DisclosureLevel.LINE, ts)
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

    private fun onMove(event: MovePlayed, ts: Double) {
        if (event.byUser && event.thinkMs != null) {
            thinkS.addLast(event.thinkMs / 1000.0)
            while (thinkS.size > PACE_MOVES) thinkS.removeFirst()
        }
        if (event.byUser) pendingAsk = null // they played on: the question is answered by the move
        board = Board(event.fenAfter)
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
        board = Board(event.fen)
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
        )
    } catch (exc: Exception) {
        speaker.lastError = "named analysis: ${exc::class.java.simpleName}: ${exc.message}"
        emptyList()
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
            speaker.provider.chat(
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
     * opponent has replied, "that move" is theirs: the generic ask handles it.
     */
    private fun lastOwnMove(): PlyRecord? {
        synchronized(lock) {
            val records = conversation.activeRecords()
            val last = records.lastOrNull() ?: return null
            if (last.move.byUser && last.assessment != null) return last
        }
        return null
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

    /** Who stands better, from the engine's score for the position, in plain words. */
    private fun standingTemplate(position: CoachPosition): String {
        val analysis = position.analysis
        if (analysis == null || analysis.lines.isEmpty()) return "I have no evaluation for this yet."
        var score = analysis.lines[0].expectedScore
        if (position.board.turn != userColour) score = 1.0 - score
        val words = when {
            score >= 0.85 -> "you are winning"
            score >= 0.62 -> "you are better"
            score > 0.38 -> "it is about equal"
            score > 0.15 -> "you are worse"
            else -> "you are losing"
        }
        val material = position.facts?.material
        val tail = if (!material.isNullOrEmpty()) "; material: $material" else ""
        return "Right now $words$tail."
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
        if (quick) {
            // The template is already out as the quick answer; a model reply that fails the
            // checks adds nothing, rather than the same line a second time.
            fallback = ""
        }
        val directive = "${levelLine(level, position)}$instructionText"
        val messages: List<ChatMessage>
        synchronized(lock) {
            conversation.level = level
            val harness = conversation.render()
            messages = listOf(ChatMessage("user", "$harness\n\n# Task\n$directive"))
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
        )
        speaker.submit(request)
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

    /** What may be said about the position the player faces, stated on the directive. */
    private fun levelLine(level: DisclosureLevel, position: CoachPosition): String {
        if (position.board.isGameOver()) return ""
        val words = when (level) {
            DisclosureLevel.SILENT -> "nothing about the position"
            DisclosureLevel.IDEA ->
                "the idea: the goal, the pattern and why it works; no square, no move, and " +
                    "not which of the player's own pieces does it"
            DisclosureLevel.FOCUS -> "the idea, the target and the piece; not the move"
            DisclosureLevel.MOVE -> "the move and why it works; not the whole line"
            DisclosureLevel.LINE -> "everything, the line included"
        }
        return "Disclosure for the position the player faces: $words.\n\n"
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
                    "well, and the one thing to work on -- the moment the game turned, the " +
                    "better move there, and why it happened, in plain words. Three or four " +
                    "sentences, no question. Do not infer the course of the game from the " +
                    "position on the board."
                ) to reviewLines.take(4).joinToString(" ")
        }
        if (trigger == Trigger.DECISIVE_QUESTION) {
            return IDEA_DIRECTIVE to ideaTemplate(facts)
        }
        if (trigger == Trigger.STUCK) {
            val directive =
                if (level.value >= DisclosureLevel.FOCUS.value) FOCUS_DIRECTIVE else IDEA_DIRECTIVE
            return (
                "The player has been thinking for a while and is stuck. " +
                    directive +
                    " Take a different angle from anything already said about this position."
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
                    )
                }
                instructionText =
                    "The player just played ${played.san}, which was a $label: it cost " +
                        "${percent(loss)} of the game. The better move was $best ($betterWords). " +
                        "Teach the difference from the fact sheet of the position they faced -- " +
                        "its goal, its pattern, why the better move works, and what their move " +
                        "missed or allowed -- as a pattern to remember, in three sentences at " +
                        "most, then ask one question that checks they see it. " +
                        "$NOT_THE_MOVE $NO_NUMBERS"
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
                        "The player just played ${played.san}, which was a $label: it cost " +
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
                    "The player just played ${played.san}, which was a $label: $best " +
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
                    )
                }
                instructionText =
                    "The player just played ${played.san}, which was fine but not the " +
                        "strongest: $best ($betterWords) was better, by a little. One or two " +
                        "sentences: that it was fine, then the better move and the idea behind it " +
                        "as a lesson to keep. No lecture, no question. $NOT_THE_MOVE $NO_NUMBERS"
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
        if (ASK_MATERIAL.containsMatchIn(said) && !facts.material.isNullOrEmpty()) {
            fallback = "Material: ${facts.material}."
        } else if (ASK_OPENING.containsMatchIn(said)) {
            val opening = openingSoFar()
            fallback = if (opening != null) {
                "The opening was the $opening."
            } else {
                "We left the opening book early."
            }
        } else if (ASK_EVAL.containsMatchIn(said)) {
            fallback = standingTemplate(position)
        }
        val shown = " Your last turn above is the quick answer they have already been shown; " +
            "build on it, do not repeat it."
        val directive: String
        val lastOpponent = lastOpponentMove()
        val lastOwn = lastOwnMove()
        if (ASK_REVIEW.containsMatchIn(said)) {
            // An overall verdict comes from the tally in "Game so far", never from an
            // impression of the last position: the first review told a player a queen up that
            // they had been a queen down.
            directive =
                "They asked how they did overall. Answer from the engine's verdicts on their " +
                    "moves in the messages above -- the labels, the tally in their last move's " +
                    "message, where they lost ground, how the material went -- and from nothing " +
                    "else: do not infer the course of the game from the position on the board. " +
                    "Two or three sentences: what went well, the one thing to work on, in plain " +
                    "words. If no move has a verdict yet, say you have not seen enough to say."
            // A review has a record to fall back on, not a line.
            fallback = reviewTemplate()
        } else if (ASK_HINT.matchesAt(said, 0)) {
            // A bare "hint" carries no claim to be honest about; treated as one, the model
            // answered its own last message as if the player had said it. Two hints, then the
            // answer: the rung decides what the hint may give.
            directive = when {
                level.value >= DisclosureLevel.MOVE.value -> MOVE_DIRECTIVE
                level.value >= DisclosureLevel.FOCUS.value ->
                    "They asked for another hint. $FOCUS_DIRECTIVE"
                else -> "They asked for a hint. $IDEA_DIRECTIVE"
            } + " Do not comment on anything said earlier." + (if (quick) shown else "")
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
                    "it means for their idea, in plain chess words, saying only what that " +
                    "analysis and the messages state; do not repeat its numbers. Then one " +
                    "question that points them at the idea. Name no move the messages do not " +
                    "give."
            fallback = ""
        } else if (ASK_OPPONENT_MOVE.containsMatchIn(said) && lastOpponent != null) {
            val theirs = lastOpponent.assessment!!
            val san = lastOpponent.move.san
            val label = theirs.label?.name?.lowercase() ?: theirs.severity.name.lowercase()
            fallback = "Their $san was $label."
            directive =
                "They ask about the opponent's last move, $san. It is in the record under " +
                    "'Last move' with its verdict ($label) and what it does; all of that is " +
                    "theirs to hear. Name the move, say how it rated and what it does, in " +
                    "plain words, two or three sentences; then, if you like, one question " +
                    "about what it asks of them. Never answer about the player's own move " +
                    "instead."
        } else if (ASK_LAST_MOVE.containsMatchIn(said) && lastOwn != null) {
            // Asked about their own last move. Its verdict was given in its move message and
            // the facts of the position it was played from in the message before: all of it
            // may be said outright. Without this the model took "the better move is withheld"
            // for the position now to cover the past one and answered with a hint instead.
            val own = lastOwn.assessment!!
            val san = lastOwn.move.san
            val label = own.label?.name?.lowercase() ?: own.severity.name.lowercase()
            val loss = own.expectedLoss
            var best = ""
            if (own.best.isNotEmpty() && own.severity != Severity.NONE) {
                val move = Move.fromUciOrNull(own.best[0])
                best = if (move != null) Board(lastOwn.move.fenBefore).san(move) else ""
            }
            // The request's facts are those of the position the move was played from when the
            // coach has them (ask() chose that position); the idea is theirs. The theme or the
            // plan, never a static feature: "the idea was the isolated d-pawn ..." became the
            // reason a bishop retreat was an inaccuracy.
            val idea = facts.theme ?: facts.plan ?: ""
            if (own.severity == Severity.NONE) {
                fallback = verdictWord(label)
                directive =
                    "They ask about the move they already played, $san. It was not bad: the " +
                        "verdict was $label, and it cost nothing. Tell them so plainly, in one " +
                        "or two sentences, and do not invent a fault. No question."
            } else {
                fallback = "It cost you ${percent(loss)} of the game."
                if (best.isNotEmpty()) {
                    // The better move played out and the pattern to keep, as the lesson says it
                    // -- not "the idea was a pin" and nothing more.
                    // Asked about the move just taught, the lesson is not said twice.
                    val opener = "$best was better"
                    fallback += " " + freshLesson(opener) { lead ->
                        lessonTemplate(facts, best, opener, played = lastOwn.move.uci, keepLead = lead)
                    }
                }
                directive =
                    "They ask about the move they already played, $san. Its verdict is in its " +
                        "move message and is theirs in full: it was a $label that cost " +
                        "${percent(loss)} of the game" +
                        (if (best.isNotEmpty()) ", the engine preferred $best" else "") +
                        (if (idea.isNotEmpty()) ", and the idea was $idea" else "") +
                        ". Say that plainly -- the position it was played from is past, so " +
                        "nothing about it is withheld; the disclosure level binds only the " +
                        "position they face now. Three sentences at most, then one question " +
                        "that checks they see the pattern."
            }
        } else {
            directive =
                "Answer them directly and be honest about their own claim. If they proposed a " +
                    "move, or named a square or piece, you may say it back and tell them " +
                    "plainly whether it holds up -- when the engine does not rate their move, " +
                    "say it is not the strongest and, in general terms, why. When the engine's " +
                    "word on the move or line they named is given above, that analysis is " +
                    "theirs to hear in full: the score after it, how far behind the best, the " +
                    "reply, what the move does. Only the engine's better move or line is " +
                    "withheld: never reveal that unless the sheet lists it, but do point them " +
                    "toward the right idea with a question. What you already said in this " +
                    "position is under 'Said here': do not say it again in other words; give " +
                    "them something it did not."
        }
        return "$checks$directive" to fallback
    }
}
