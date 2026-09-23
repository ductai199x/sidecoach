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
import com.sidecoach.events.Label
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
import com.sidecoach.llm.OfflineProvider
import com.sidecoach.llm.StreamingProvider
import com.sidecoach.vision.Pieces
import java.math.BigDecimal
import java.math.RoundingMode
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.Executors
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import kotlin.random.Random

/**
 * The coach: what to say, when, from which facts, checked before it is shown.
 * `coach/coach.py`.
 *
 * Three things keep it honest. The disclosure level decides which facts reach the model at
 * all, so it cannot repeat what it never saw. The level only ever moves through the
 * deterministic unlocks in [Disclosure], driven by what the user did or asked, never by the
 * model's own judgement. And every draft is checked twice before it is shown: for leaks
 * past the current level, and for claims about the board that the board contradicts. A
 * draft that fails is regenerated once with the problem named, and then replaced by a
 * template, so the user never sees a leak and never sees nothing.
 *
 * The model is slow, a second or three, so it runs on a worker thread; the capture loop
 * hands in events and polls for messages, exactly as it does with the engine.
 */

const val MAX_TOKENS = 220

/**
 * After every this-many mistakes the lesson ends by asking what the player saw: a
 * self-report the profile keeps beside the mistake (DESIGN 5.6, "sparingly"). One that goes
 * unanswered is the last of the game: a player with the mic off heard "did you miss your
 * opponent's idea altogether, or see it and misjudge the line?" seven times in one game
 * (2026-09-19), the model echoing the directive's words each time.
 */
const val ELICIT_EVERY = 3

/**
 * The player's words within this long of the question count as its answer (the profiler's
 * SELF_REPORT_S, which keeps those words beside the mistake).
 */
const val ELICIT_ANSWER_S = 60.0

/**
 * The word on a move by its label (Tai, 2026-09-20). LOCAL: written from the board, no
 * model -- theory, the engine's own move, and an excellent one (as strong as the best).
 * EXCITING: great and brilliant, the model at its full effort with some delight, since they
 * are rare and worth a real sentence. A good move was a little behind the best, and the
 * model names the better one.
 */
val LOCAL = setOf("book", "best", "excellent")
val EXCITING = setOf("great", "brilliant")

internal fun isLocal(label: Label?): Boolean = label != null && label.name.lowercase() in LOCAL

internal fun isExciting(label: Label?): Boolean =
    label != null && label.name.lowercase() in EXCITING

/**
 * How many of the last lessons a sound move's word may not teach again ([lesson]): "develop
 * first" after every developing move of an opening is a lecture, not a lesson.
 */
const val TAUGHT_MEMORY = 3

/**
 * The praise, by the move's label, in the vocabulary chess.com uses ([OPENERS], with the
 * lesson the board gives): said alone when there is no lesson to add.
 */
val VERDICTS: Map<String, List<String>> =
    OPENERS.mapValues { (_, openers) -> openers.map { "$it." } }

/**
 * The templates said without a model, where the same sentence four times in a game wears
 * thin (a master-game pass, offline, 2026-09-21): a choice each time, never the last one
 * said. The orienting question at a critical moment names nothing on the board.
 */
val DECISIVE_QUESTIONS = listOf(
    "This is a moment to slow down. What is your opponent's most dangerous idea here, and " +
        "which of your pieces is doing the least?",
    "Take your time here: it matters. What does your opponent want to do next, and can " +
        "you stop it and improve at the same time?",
    "A critical moment. Before you move, what are their checks, captures and threats?",
    "Slow down. If you passed, what would your opponent play -- and does your move answer it?",
    "This one counts. Which of your pieces would you most like to improve, and what are they " +
        "threatening right now?",
)

/** The hint when the sheet gives nothing to point at: what to take stock of. */
val STUCK_GENERIC = listOf(
    "Take stock: what changed with the last move, and what does it allow?",
    "Start from their last move: what does it attack, and what did it leave behind?",
    "List the forcing moves first -- the checks and captures -- then the quiet ones.",
    "Which of your pieces is worst placed? A move that fixes it is rarely wrong.",
)

/**
 * The theme-level hint, "{theme}" the idea the sheet names (a noun phrase, "a pin", or a
 * plan, "finishing your development"; the frames read with either).
 */
val THEME_HINTS = listOf(
    "The idea here is {theme}. What would make it count?",
    "Think about {theme}. What is the first step?",
    "Something to build on: {theme}. Which move serves it?",
)

/** The one fact the question level allows: whether the answer is forcing or quiet. */
val FORCING_HINTS = listOf(
    "Something forcing is on: is there a check or a capture worth calculating first?",
    "There is a forcing move here. Go through your checks and captures, in order.",
    "Start with the forcing moves: which check or capture changes the most?",
)
val QUIET_HINTS = listOf(
    "Nothing forcing is on. Which of your pieces is doing the least, and where would it do more?",
    "A quiet move is wanted. Which piece would you most like to improve?",
    "No tactic to find here. What would your opponent least like you to do?",
)

/** A question at the question level, where nothing of the position may be named. */
/**
 * The word on a move a little behind the best, and the start of the lesson on one that cost
 * something; "{best}" is the better move. Without the move played in them, the same better
 * move twice running gave the same sentence twice (the offline sweep, 2026-09-23).
 */
val FINE_TIGHTER = listOf(
    "Fine, but {best} was a little tighter",
    "That works, though {best} was a touch better",
    "Playable -- {best} was a little more precise",
)
val COST_YOU = listOf(
    "That one cost you: {best} was better",
    "That cost you something: {best} was the move",
    "Not that one -- {best} was better",
)

val ASK_GENERIC = listOf(
    "What is your opponent's most dangerous idea here?",
    "Before anything else: what does your opponent want to play next?",
    "Which of your pieces is doing the least right now?",
    "What did their last move change -- what does it attack, and what did it leave?",
)

/**
 * Verbalized sampling: how many candidate replies to ask for, and the room and warmth to
 * give the distribution. Warmer than a single reply on purpose, since a check filters them.
 */
const val VS_CANDIDATES = 4

/**
 * An answer to the player's own question is wanted fast, and it is checked like any other;
 * two candidates halve the wait and still leave a spare for the checks to reject.
 */
const val VS_ASK_CANDIDATES = 2

/** A word on a fine move is one short sentence; two candidates keep the call short. */
const val VS_ROUTINE_CANDIDATES = 2
const val VS_MAX_TOKENS = 2048

/**
 * Cold. The candidates are already a distribution the model verbalizes (their p= marks);
 * hot sampling on top of that is where the invented sentence came from.
 */
const val VS_TEMPERATURE = 0.1

private val NUMBER_WORDS = mapOf(2 to "TWO", 3 to "THREE", 4 to "FOUR", 5 to "FIVE")

/**
 * The unprompted remarks carry no figures: the record says what a move cost in expected
 * score and the model was quoting it in every note ("costing 0.03"); a player who wants the
 * number asks for it (Tai, 2026-09-19).
 */
const val NO_NUMBERS =
    "Do not quote the cost or any number: say how the move rates in words, and keep the " +
        "figures for when the player asks. Name moves bare, as h3 or Bd5, never with a move " +
        "number in front."

/**
 * The player has just played the move and sees it on the board: naming it, or saying what it
 * plainly did ("Nxd5 -- the knight taking the pawn on d5"), tells them nothing (Tai,
 * 2026-09-23: "we want the coach to be instructive, always find a moment to teach something,
 * not just a dumb bot").
 */
const val NOT_THE_MOVE =
    "Do not open with the move they played, and do not say what it plainly did on the board " +
        "-- what it took, the check, the castling, the promotion: they just played it and see it. " +
        "Say what it means instead: the idea behind it, the pattern to keep, the better move and " +
        "why."

fun vsDirective(candidates: Int = VS_CANDIDATES): String {
    val count = NUMBER_WORDS[candidates] ?: candidates.toString()
    return "\n\nInstead of one reply, give $count different ways you could say this, varied in " +
        "wording and in the angle each takes, sampled from your full range rather than the " +
        "single most obvious phrasing. Put each on its own line as: [p=0.30] then the reply, " +
        "the replies in a random order, not the likeliest first. " +
        "Each reply obeys every rule above and keeps to the length asked, on that one line. " +
        "The p values are how likely you would be to choose each; they need not sum to one."
}

val VS_DIRECTIVE = vsDirective()

private val VS_MARKER = Regex("\\[\\s*p?\\s*=?\\s*([0-9]*\\.?[0-9]+)\\s*\\]\\s*", RegexOption.IGNORE_CASE)

/** A marker fragment left in a candidate's text: a whole one, or one cut off mid-way. */
private val STRAY_MARKER = Regex("\\s*\\[\\s*p\\s*=\\s*[0-9.]*\\]?\\s*")
private val THINK_BLOCK = Regex("<think>.*?</think>", RegexOption.DOT_MATCHES_ALL)

/**
 * Where a reply's last complete sentence ends: terminal punctuation, then any closing quote
 * or bracket. What follows the last of these is a sentence the model never finished. A move
 * number's stop ("after 6. d3", "12. Bxc6+") ends no sentence: a bare number of up to three
 * digits, at the start or after a space, before the stop.
 */
private val SENTENCE_END = Regex(
    "(?<!(?<![\\w.])\\d)(?<!(?<![\\w.])\\d\\d)(?<!(?<![\\w.])\\d\\d\\d)" +
        "[.!?\u2026][\"'\u201d\u2019)\\]]*(?=\\s|\$)"
)

private val PIECE_WORD = mapOf(
    'K' to "king", 'Q' to "queen", 'R' to "rook", 'B' to "bishop", 'N' to "knight",
)

/**
 * A reply may run this many sentences past the number its directive gives before it is
 * faulted: the length is reasonable, not strict.
 */
const val LENGTH_SLACK = 2

private val WORD_NUMBERS =
    mapOf("one" to 1, "two" to 2, "three" to 3, "four" to 4, "five" to 5, "six" to 6)

private val SAYS_BEST = Regex(
    "\\b(?:the |your )?(?:best|strongest|top|engine's) (?:move|choice)\\b|\\bwas best\\b|" +
        "\\bis best\\b",
    RegexOption.IGNORE_CASE,
)
private val SAYS_SLIP =
    Regex("\\b(?:blunder|mistake|inaccuracy|a miss)\\b", RegexOption.IGNORE_CASE)

/**
 * A figure after "cost": a share of the game (0.10, 10%), never a count of moves or pawns
 * ("lost the pawn on e4 2 moves ago" gave a cost of 2).
 */
private val SAYS_COST = Regex(
    "\\b(?:cost|costing|costs|lost|losing)\\b[^.;]{0,30}?(?<![A-Za-z0-9])(\\d+(?:\\.\\d+)?)" +
        "(?![.\\d]*[A-Za-z])(?!\\s*(?:moves?|plies|pawns?|pieces?|tempi|tempo|points?)\\b)" +
        "\\s*(%|percent)?",
    RegexOption.IGNORE_CASE,
)

private val LIMIT_WORDS = Regex(
    "\\b(?:(?:at most|no more than) (one|two|three|four|five|six) sentences?" +
        "|(one|two|three|four|five|six)(?: or (one|two|three|four|five|six))? (?:short )?sentences?" +
        "|(one|two|three|four|five|six) sentences? at most" +
        "|(?:fifteen|twenty) words at most)\\b",
    RegexOption.IGNORE_CASE,
)

private val JSON_OBJECT = Regex("\\{.*\\}", RegexOption.DOT_MATCHES_ALL)

internal val SAN = Regex("\\b(O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?)\\b")

/**
 * Faults in what a reply says of the move just assessed: a "best" that was not, a slip word
 * for a sound move, a cost that is not the recorded one.
 */
internal fun verdictProblems(text: String, request: CoachRequest): List<String> {
    val a = request.assessment ?: return emptyList()
    val played = request.playedSan
    if (played.isNullOrEmpty()) return emptyList()
    // Only when the reply speaks of that move, and of no other move by name.
    val others = SAN.findAll(text).map { it.groupValues[1] }
        .filter { it.trimEnd('+', '#') != played.trimEnd('+', '#') }
        .toList()
    val names = Regex(
        "(?<![A-Za-z])" + Regex.escape(played.trimEnd('+', '#')) + "[+#]?(?![A-Za-z0-9])"
    )
    if (others.isNotEmpty() || !names.containsMatchIn(text)) return emptyList()
    val label = a.label?.name?.lowercase() ?: a.severity.name.lowercase()
    val problems = ArrayList<String>()
    if (SAYS_BEST.containsMatchIn(text) &&
        label != "best" && label != "excellent" &&
        !Regex(
            "\\bnot (?:the )?(?:best|strongest)\\b|\\bwas better\\b|\\bwas stronger\\b",
            RegexOption.IGNORE_CASE,
        ).containsMatchIn(text)
    ) {
        problems.add("called $played the best move; its verdict is $label")
    }
    if (SAYS_SLIP.containsMatchIn(text) &&
        a.severity == Severity.NONE &&
        !Regex(
            "\\bnot (?:a )?(?:blunder|mistake)\\b|\\bno (?:blunder|mistake)\\b",
            RegexOption.IGNORE_CASE,
        ).containsMatchIn(text)
    ) {
        problems.add("called $played a slip; its verdict is $label")
    }
    for (found in SAYS_COST.findAll(text)) {
        var value = found.groupValues[1].toDouble()
        if (found.groupValues[2].isNotEmpty() || value > 1.0) value /= 100.0
        if (kotlin.math.abs(value - a.expectedLoss) > maxOf(0.02, 0.35 * a.expectedLoss)) {
            problems.add(
                "gave the cost of $played as ${found.groupValues[1]}; it was " +
                    formatted(a.expectedLoss, 2)
            )
        }
    }
    return problems
}

/**
 * The length the directive states, in sentences, and whether it asks for a closing question
 * (true), forbids one (false) or says nothing (null).
 */
fun directiveLimits(directive: String): Pair<Int?, Boolean?> {
    var limit: Int? = null
    for (m in LIMIT_WORDS.findAll(directive)) {
        val n = if (m.value.lowercase().endsWith("words at most")) {
            1
        } else {
            m.groupValues.drop(1).filter { it.isNotEmpty() }
                .maxOf { WORD_NUMBERS.getValue(it.lowercase()) }
        }
        limit = if (limit == null) n else maxOf(limit, n)
    }
    val low = directive.lowercase()
    var question: Boolean? = null
    if (Regex("\\bask one (?:orienting )?question\\b").containsMatchIn(low)) question = true
    if (Regex("\\bthen(?:, if you like,)? one question\\b").containsMatchIn(low)) {
        // A question asked for on top of the sentences: one more is allowed.
        question = true
        limit = limit?.plus(1)
    } else if (Regex("\\bending with (?:one |a )?question\\b").containsMatchIn(low)) {
        question = true
    }
    if (Regex("\\bno question\\b|\\bdo not answer with a question\\b").containsMatchIn(low)) {
        question = false
    }
    return limit to question
}

/**
 * The first [limit] complete sentences of [text]; the closing question kept in place of the
 * last one when the directive asks for a question and the text ends with one.
 */
fun trimToLimit(text: String, limit: Int, wantsQuestion: Boolean?): String {
    val ends = SENTENCE_END.findAll(text).map { it.range.last + 1 }.toList()
    if (ends.size <= limit) return text
    val starts = listOf(0) + ends.dropLast(1)
    val sentences = starts.zip(ends).map { (a, b) -> text.substring(a, b).trim() }
    val kept = if (wantsQuestion == true && sentences.last().endsWith("?") && limit >= 2) {
        sentences.take(limit - 1) + sentences.last()
    } else {
        sentences.take(limit)
    }
    return kept.joinToString(" ")
}

internal fun sentenceTotal(text: String): Int = SENTENCE_END.findAll(text).count()

/** Append the verbalized-sampling directive to the final user turn; none for one reply. */
internal fun vsWrap(messages: List<ChatMessage>, candidates: Int = VS_CANDIDATES): List<ChatMessage> {
    if (messages.isEmpty() || candidates <= 1) return messages
    val wrapped = messages.toMutableList()
    val last = wrapped.last()
    wrapped[wrapped.size - 1] = ChatMessage(last.role, last.content + vsDirective(candidates))
    return wrapped
}

/**
 * The text up to the end of its last complete sentence; empty if it has none.
 *
 * A reply the token budget cut off ends mid-sentence -- "...is spot-on. Given" -- and in the
 * first live game a one-word stump like "Given" was the candidate that passed every check
 * and got spoken. A stump is not a reply, so it is trimmed away or, alone, dropped.
 */
fun completeSentences(text: String): String {
    val ends = SENTENCE_END.findAll(text).toList()
    return if (ends.isNotEmpty()) text.substring(0, ends.last().range.last + 1).trim() else ""
}

/**
 * The verbalized replies as (probability, text). A reply with no markers is one candidate.
 *
 * The text of each candidate runs from its marker to the next, so a reply that wraps onto
 * another line is kept whole. A model that ignores the format still yields one usable reply.
 * Every candidate is cut back to its last complete sentence.
 */
internal fun parseCandidates(raw: String): List<Pair<Double, String>> {
    var cleaned = THINK_BLOCK.replace(raw, "").trim()
    val markers = VS_MARKER.findAll(cleaned).toList()
    if (markers.isEmpty()) {
        cleaned = completeSentences(STRAY_MARKER.replace(cleaned, "").trim())
        return if (cleaned.isNotEmpty()) listOf(1.0 to cleaned) else emptyList()
    }
    val out = ArrayList<Pair<Double, String>>()
    for ((index, marker) in markers.withIndex()) {
        val start = marker.range.last + 1
        val end = if (index + 1 < markers.size) markers[index + 1].range.first else cleaned.length
        // A reply cut off by the token budget can end in a half-written next marker
        // ("...move? [p=0.2" with no closing bracket); no fragment may reach the user.
        val text = completeSentences(
            STRAY_MARKER.replace(cleaned.substring(start, end), "").trim()
        )
        if (text.isEmpty()) continue
        val prob = marker.groupValues[1].toDoubleOrNull() ?: 0.0
        out.add(prob to text)
    }
    return out
}

/** The first JSON object in a model's answer, or null. */
internal fun jsonObject(text: String): org.json.JSONObject? {
    val m = JSON_OBJECT.find(text) ?: return null
    return runCatching { org.json.JSONObject(m.value) }.getOrNull()
}

/** Sample one reply, weighted by its stated likelihood; ties and zeros still get a chance. */
internal fun sampleReply(passed: List<Pair<Double, String>>): String {
    val weights = passed.map { maxOf(it.first, 0.01) }
    val total = weights.sum()
    var mark = Random.nextDouble() * total
    for ((index, weight) in weights.withIndex()) {
        mark -= weight
        if (mark <= 0.0) return passed[index].second
    }
    return passed.last().second
}

internal fun formatted(value: Double, places: Int): String =
    BigDecimal(value).setScale(places, RoundingMode.HALF_EVEN).toPlainString()

/** `f"{value:.0%}"`: the float scaled by a hundred, rounded half to even, with the sign. */
internal fun percent(value: Double): String =
    BigDecimal(value * 100.0).setScale(0, RoundingMode.HALF_EVEN).toPlainString() + "%"

/**
 * Reads the first verbalized candidate off the model's stream and hands it to a sink
 * sentence by sentence. Each new complete sentence is checked (the leak and board checks)
 * before it goes; a fault stops the stream with what was said so far kept. The stream is
 * stopped once the candidate is complete -- the second marker seen -- so the other
 * candidates are never written. A reply with no marker at all is one candidate.
 */
class FirstCandidate(
    private val sink: (String) -> Boolean,
    private val check: (String) -> List<String>,
) {
    private var buffer = ""

    /** What the sink has been given. */
    var spoken = ""
        private set
    var problems: List<String> = emptyList()
        private set
    private var done = false

    /** The sink refused the first sentence: the voice is not wanted. */
    var declined = false
        private set

    /** The first candidate's text so far, and whether it is complete. */
    private fun candidate(): Pair<String, Boolean> {
        val cleaned = THINK_BLOCK.replace(buffer, "")
        val markers = VS_MARKER.findAll(cleaned).toList()
        if (markers.isEmpty()) {
            // No marker yet: the model may be writing one, or writing a bare reply.
            if (cleaned.trimStart().startsWith("[") || cleaned.length < 12) return "" to false
            return cleaned to false
        }
        val start = markers[0].range.last + 1
        if (markers.size > 1) return cleaned.substring(start, markers[1].range.first) to true
        return cleaned.substring(start) to false
    }

    /**
     * Hand the sink the complete sentences of [text] not yet given. False to stop.
     *
     * A sentence has ended only once the space after its full stop has arrived: at the
     * buffer's end "costing 0." is the start of "costing 0.02", and a cost read as 0 failed
     * the check and stopped the stream. A chunk without letters ("6.", a move number the
     * model put first) waits for the sentence it introduces.
     */
    private fun offer(text: String, final: Boolean = false): Boolean {
        val cleaned = STRAY_MARKER.replace(text, "").trim()
        val ends = SENTENCE_END.findAll(cleaned)
            .filter { final || it.range.last + 1 < cleaned.length }.toList()
        val complete = if (ends.isNotEmpty()) {
            cleaned.substring(0, ends.last().range.last + 1).trim()
        } else {
            ""
        }
        if (complete.length <= spoken.length) return true
        val fresh = complete.substring(spoken.length).trim()
        if (fresh.count { it.isLetter() } < 2) {
            return true // a number and a stop: the sentence it begins is still coming
        }
        var faults = if (fresh.isNotEmpty()) check(fresh) else emptyList()
        faults = faults.filter { !it.startsWith("too long") && !it.contains("question") }
        if (faults.isNotEmpty()) {
            problems = faults
            return false
        }
        if (!sink(fresh)) {
            declined = true
            return false
        }
        spoken = complete
        return true
    }

    fun feed(delta: String): Boolean {
        if (done) return false
        buffer += delta
        val (text, complete) = candidate()
        // A complete candidate (its successor's marker seen) ends on its last stop, with
        // nothing after: that stop counts.
        if (!offer(text, final = complete)) {
            done = true
            return false
        }
        if (complete) {
            done = true
            return false
        }
        return true
    }

    /** The stream ended: whatever complete sentences remain go out. */
    fun finish() {
        if (done) return
        done = true
        val (text, _) = candidate()
        if (text.isNotEmpty()) offer(text, final = true)
    }
}

/**
 * One call to the model: the turn it was given (the last user message: memory, fact sheet,
 * instruction), what it wrote, what the server said about the call (finish reason, tokens,
 * seconds), and how the candidates fared. The debug log's answer to "why did the coach say
 * that?".
 */
class ModelCall(
    val raw: String,
    val stats: Map<String, Any?>,
    val candidates: Int,
    val passed: Int,
    val prompt: String = "",
)

/** One coach message, with what it was checked against, for the log. */
class Spoken(
    val message: CoachMessage,
    val level: DisclosureLevel,
    val problems: List<String>,
    val attempts: Int,
    val calls: List<ModelCall> = emptyList(),
    /**
     * Already a turn of the conversation (a quick answer recorded before the model was
     * asked to build on it), so polling it must not record it again.
     */
    val recorded: Boolean = false,
    /**
     * The reply went out through the request's sink as it was written: the voice has had
     * it, sentence by sentence, and nothing is to be spoken again.
     */
    val streamed: Boolean = false,
    /**
     * What the verifier found, when one runs: per candidate the statements the messages do
     * not support, and whether the reply spoken was a repair of the best candidate.
     */
    val verified: List<List<String>> = emptyList(),
    val repaired: Boolean = false,
)

class CoachPosition(
    val ply: Int,
    val board: Board,
    var analysis: Analysis? = null,
    var facts: PositionFacts? = null,
    val disclosure: Disclosure = Disclosure(),
    val lastMove: String? = null,
)

/**
 * The game's pace: the user's last few think times. Under [FAST_PACE_S] a move, the game is
 * fast -- a model reply (two to five seconds) would land after the next move, and a spoken
 * sentence (three to five) would be talked over the next two. So a fast game gets the
 * coach's templates, written from the board at once, and gets them as text; the voice
 * resumes when the player slows down and for what they ask (Tai, 2026-09-22: "the coach has
 * to be able to keep up with the game").
 */
const val FAST_PACE_S = 6.0
const val PACE_MOVES = 4

/** How many plies back the coach keeps a position and its facts. */
const val POSITIONS_KEPT = 40

class CoachRequest(
    val trigger: Trigger,
    val level: DisclosureLevel,
    val system: String,
    var messages: List<ChatMessage>,
    val facts: PositionFacts,
    val board: Board,
    val fallback: String,
    val ply: Int,
    val gameId: String,
    /**
     * What the user said this turn, if anything. A square, piece or move the user named is
     * not a leak for the coach to repeat back: confirming it reveals nothing they did not
     * say.
     */
    val userText: String? = null,
    /**
     * What the directive itself disclosed this turn -- the engine's word on the move or
     * line the player asked about -- which the coach may therefore repeat.
     */
    val disclosed: String = "",
    /** Board marks built from the same filtered facts the model sees. */
    val annotations: List<Annotation> = emptyList(),
    /** How many replies to ask the model for; fewer for an answer the player is waiting on. */
    val candidates: Int = VS_CANDIDATES,
    /**
     * The moves of the record (SAN): those played so far, and those the verdicts named as
     * better. A reply about the game as it went may name either; the board checker must not
     * fault "Rg6" from move 30 as unplayable now, nor the leak check count it.
     */
    val played: Set<String> = emptySet(),
    /**
     * The move just assessed and its assessment, when the turn is about it: what the reply
     * says of its verdict and cost is checked against them.
     */
    val playedSan: String? = null,
    val assessment: MoveAssessment? = null,
    /**
     * Where to send the reply sentence by sentence as the model writes it (the panel's
     * voice), or null for a reply that goes out whole once checked. The sink returns false
     * to stop the stream: the board has moved on, or the voice is not wanted.
     */
    val streamTo: ((String) -> Boolean)? = null,
    /**
     * A mundane reply -- a word on a fine move -- which the Speaker's routine model writes
     * (the same model at less reasoning, by default none: the first sound comes in under a
     * second instead of three).
     */
    val routine: Boolean = false,
    /**
     * A fast game: the fallback -- the template written from the board -- is the reply, and
     * the model is not called (it could not keep up).
     */
    val templateOnly: Boolean = false,
)

/**
 * The boards along the engine's line, so what the line does reads as real.
 *
 * A fork or pin the recommended move creates is not on the board in front of the player; the
 * draft checker is handed the frames after each move of the line so it does not fault the
 * coach for naming a piece on the square a move will put it on, or -- when asked for the
 * line and reciting it -- for a move that is only playable further along it.
 */
internal fun afterBest(request: CoachRequest): List<Board> {
    val line = request.facts.line.ifEmpty { request.facts.candidates.take(1) }
    val frames = ArrayList<Board>()
    var board = request.board.copy(withStack = false)
    for (san in line) {
        val move = board.parseSan(san) ?: break
        board = board.copy(withStack = false)
        board.push(move)
        frames.add(board)
    }
    // And along what the directive disclosed: the line the player asked about, examined.
    if (request.disclosed.isNotEmpty()) {
        frames.addAll(disclosedFrames(request.board, request.disclosed.split("\n")))
    }
    return frames
}

const val VERIFY_SYSTEM =
    "You check a chess coach's reply against the messages the coach was given: the game so " +
        "far, the engine's verdicts (the label, the cost, the move the engine preferred, what " +
        "the moves do), the lines, the facts (the nature of the position, the theme, the plan, " +
        "the threats, the ideas), the tally, and the directive. A statement is supported when a " +
        "message states it, or restates it in other words, or it follows directly from the move " +
        "itself (a capture takes the piece on that square, a check makes the king answer, a move " +
        "lands where it lands), or it is a plain inference the messages make obvious, or " +
        "ordinary chess usage of a stated fact (a rook on an open file 'active', a king on e1 " +
        "'in the centre'). Approximate or hedged wording of a supported fact is supported. " +
        "Unsupported is only what is specific and absent: what a piece attacks, defends, pins " +
        "or threatens when no message says so; a plan, weakness, structure or evaluation no " +
        "message gives; a reason for a move no message gives; a move, win or loss that did not " +
        "happen; a habit the record does not show. A question is not a statement. Quote each " +
        "unsupported statement in the reply's own words; when in doubt, it is supported. " +
        "Answer in JSON only: {\"unsupported\": [str]}"

const val REPAIR_SYSTEM =
    "You edit a chess coach's reply. Remove or reword only the statements listed as " +
        "unsupported, so that what remains is supported by the messages the coach was given. " +
        "Change nothing else: every other sentence stays exactly as it is -- in particular keep " +
        "every fact about the move that the messages give (its label, what it cost, the move " +
        "the engine preferred, the idea) -- the form stays (a closing question stays a closing " +
        "question), the voice stays; add nothing. Answer in JSON only: {\"reply\": str}"

/** What the draft checker does: `claims.checkDraft` over the board and the line's frames. */
fun interface DraftCheck {
    fun check(text: String, board: Board, after: List<Board>): List<String>
}

/**
 * Runs the model on a worker thread and checks what it says.
 *
 * A draft with a leak or a false claim about the board is sent back once with the problem
 * named; a second failure is replaced by the template, and the failure counted. With a
 * [verifier], every candidate that passes those checks is also read against the messages by
 * a second model, the clean ones are sampled from, and when none is clean the likeliest is
 * repaired once: test-time verification, on top of the demonstrations in C.
 */
class Speaker(
    var provider: LLMProvider,
    val draftCheck: DraftCheck? = null,
    var verifier: LLMProvider? = null,
    routineProvider: LLMProvider? = null,
    /** Which of the passing candidates is spoken; injected so a test can fix the draw. */
    val sample: (List<Pair<Double, String>>) -> String = ::sampleReply,
) {
    /**
     * The model for a mundane reply (a word on a fine move), where nothing is at stake and
     * the first sound matters: the same model at less reasoning (Tai: low at most; none is
     * what makes it fast, low reasons as long as medium). Everything else -- a mistake, a
     * decisive moment, a nudge, a question of the player's -- keeps [provider] at its effort.
     */
    var routineProvider: LLMProvider = routineProvider ?: provider

    var failures = 0
        internal set
    var fallbacks = 0
        internal set

    /** Unprompted requests an ask overtook before they started. */
    var dropped = 0
        internal set
    var lastError: String? = null

    private val requests = LinkedBlockingQueue<Any>()
    private val results = ConcurrentLinkedQueue<Spoken>()
    private var thread: Thread? = null

    @Volatile
    private var busy = false

    private object Stop

    val pending: Int get() = requests.size + (if (busy) 1 else 0)

    fun start() {
        if (thread == null) {
            thread = Thread({ run() }, "sidecoach-coach").apply {
                isDaemon = true
                start()
            }
        }
    }

    fun close(timeoutMs: Long = 10000) {
        val running = thread
        thread = null
        if (running != null) {
            requests.put(Stop)
            running.join(timeoutMs)
        }
    }

    /**
     * Queue a request. An ask goes ahead of any unprompted request still waiting: the player
     * is waiting for this one, and a post-move remark or a nudge that has not started yet
     * would only be stale by the time the answer came out.
     */
    fun submit(request: CoachRequest) {
        if (request.trigger == Trigger.USER_ASK) {
            val kept = ArrayList<Any>()
            while (true) {
                val item = requests.poll() ?: break
                if (item is CoachRequest && item.trigger != Trigger.USER_ASK) {
                    dropped += 1
                    continue
                }
                kept.add(item)
            }
            for (item in kept) requests.put(item)
        }
        requests.put(request)
    }

    /**
     * A message that is already final; it goes out through the same queue, in order -- or,
     * with [now], straight to the results, ahead of whatever the model is writing: the quick
     * answer to a button press must not wait behind a post-move remark.
     */
    fun submitReady(spoken: Spoken, now: Boolean = false) {
        if (now) results.add(spoken) else requests.put(spoken)
    }

    /** Answer every queued request on the calling thread: for tests and for replay. */
    fun runPending() {
        while (true) {
            val request = requests.poll() ?: return
            when (request) {
                is Spoken -> results.add(request)
                is CoachRequest -> results.add(speakNow(request))
                else -> return
            }
        }
    }

    fun poll(): List<Spoken> {
        val out = ArrayList<Spoken>()
        while (true) {
            out.add(results.poll() ?: return out)
        }
    }

    /**
     * Verbalized sampling: ask for several replies, keep a checked one, vary the voice.
     *
     * A single reply collapses onto the one most typical phrasing, which is why the coach
     * repeated itself. Asking the model to verbalize a small distribution of replies (arxiv
     * 2510.01171) recovers its range; the leak and board checks then filter that set, and a
     * reply is sampled by its own stated likelihood from those that pass. Because several
     * are offered at once, a clean one is usually among them, so the template is rarely
     * needed.
     */
    fun speakNow(request: CoachRequest): Spoken {
        var messages = vsWrap(request.messages, request.candidates)
        var problems: List<String> = emptyList()
        var text = ""
        var attempts = 0
        val calls = ArrayList<ModelCall>()
        var verified: List<List<String>> = emptyList()
        var repaired = false
        var streamed = false
        val provider = if (request.routine) routineProvider else this.provider
        if (provider is OfflineProvider || request.templateOnly) {
            // No model (llm.provider none), or no time for one (a fast game): the request's
            // fallback is the reply -- the verdict from the board, the lesson or hint
            // template, the review's own sentences -- or silence when there is none. Free
            // and offline.
            fallbacks += 1
            text = request.fallback
            return Spoken(
                message = CoachMessage(
                    text = text,
                    level = request.level,
                    trigger = request.trigger,
                    gameId = request.gameId,
                    ply = request.ply,
                    fen = request.board.fen(),
                    annotations = request.annotations + mentionsIn(text, request.board),
                ),
                level = request.level,
                problems = emptyList(),
                attempts = 0,
            )
        }
        while (attempts < 2) {
            attempts += 1
            var stats: MutableMap<String, Any?> = LinkedHashMap()
            var reader: FirstCandidate? = null
            val streaming = provider as? StreamingProvider
            if (request.streamTo != null && attempts == 1 && verifier == null && streaming != null) {
                // The first candidate goes to the voice sentence by sentence as it is
                // written, each sentence checked for leaks before it is said; the stream is
                // closed once that candidate is complete, the others unwritten.
                reader = FirstCandidate(request.streamTo) { t -> problemsOf(t, request) }
            }
            var raw: String
            try {
                val completion = if (reader != null && streaming != null) {
                    streaming.chatStream(
                        request.system,
                        messages,
                        maxTokens = VS_MAX_TOKENS,
                        temperature = VS_TEMPERATURE,
                        onDelta = reader::feed,
                    )
                } else {
                    provider.chat(
                        request.system,
                        messages,
                        maxTokens = VS_MAX_TOKENS,
                        temperature = VS_TEMPERATURE,
                    )
                }
                raw = completion.text
                stats = LinkedHashMap(completion.stats)
            } catch (exc: Exception) {
                failures += 1
                lastError = "${exc::class.java.simpleName}: ${exc.message}"
                raw = ""
            }
            if (reader != null) {
                reader.finish()
                if (reader.spoken.isNotEmpty()) {
                    // Said already, in part at least: the record keeps what was said and
                    // what, if anything, was wrong with the rest; no template follows a
                    // voice that has spoken.
                    text = reader.spoken
                    problems = reader.problems
                    streamed = true
                    val given = messages.takeLast(2).joinToString("\n\n---\n\n") { it.content }
                    calls.add(ModelCall(raw, stats, 1, if (problems.isEmpty()) 1 else 0, given))
                    break
                }
                // Nothing said: the first candidate failed its first sentence, or the voice
                // declined. The reply is read whole from here, as without a sink; what
                // stopped the stream goes in the call's stats for the log.
                stats["stream_problems"] = reader.problems.ifEmpty {
                    if (reader.declined) listOf("the voice declined") else listOf("the stream ended early")
                }
            }
            val scored = parseCandidates(raw).map { (prob, candidate) ->
                Triple(prob, candidate, problemsOf(candidate, request))
            }
            var passed = scored.filter { it.third.isEmpty() }.map { it.first to it.second }
            if (passed.isEmpty() && scored.isNotEmpty() &&
                scored.all { (_, _, faults) ->
                    faults.isNotEmpty() && faults.all { it.startsWith("too long") }
                }
            ) {
                // Every candidate clean but for its length: the likeliest, cut to the limit,
                // instead of a second call or the template.
                val directive = request.messages.last().content.substringAfterLast("# Task")
                val (limit, wantsQuestion) = directiveLimits(directive)
                val best = scored.maxByOrNull { it.first }!!
                val trimmed = trimToLimit(best.second, (limit ?: 2) + LENGTH_SLACK, wantsQuestion)
                if (problemsOf(trimmed, request).isEmpty()) passed = listOf(best.first to trimmed)
            }
            val given = messages.takeLast(2).joinToString("\n\n---\n\n") { it.content }
            calls.add(ModelCall(raw, stats, scored.size, passed.size, given))
            if (passed.isNotEmpty() && verifier != null) {
                val outcome = verify(passed, request)
                passed = outcome.first
                verified = outcome.second
                repaired = outcome.third
            }
            if (passed.isNotEmpty()) {
                text = sample(passed)
                problems = emptyList()
                break
            }
            problems = if (scored.isNotEmpty()) scored[0].third else listOf("empty reply")
            if (reader != null && reader.declined) {
                // The voice declined the reply (the board had moved on, or the voice was
                // busy): a second call about that moment would be paid for nothing. What the
                // stream gave is not clean, so the template stands.
                break
            }
            // The second try carries the complaint inside the directive: appending the
            // failed attempt and the complaint as two more messages doubled the prompt on a
            // long game (131k tokens became 276k, past the model's context).
            val rejected = problems.joinToString("; ")
            val shown = raw.trim().take(600).ifEmpty { "(no reply)" }
            val last = messages.last()
            val retry = ChatMessage(
                last.role,
                last.content +
                    "\n\nYour first attempt was:\n$shown\n\nThose replies are not allowed: " +
                    "$rejected. Give " +
                    (
                        if (request.candidates <= 1) {
                            "a fresh reply that avoids those"
                        } else {
                            "${NUMBER_WORDS[request.candidates] ?: request.candidates} " +
                                "fresh, varied replies that avoid those"
                        }
                        ) +
                    ", keeping to the messages above."
            )
            messages = messages.dropLast(1) + retry
        }
        if (problems.isNotEmpty() && !streamed) {
            fallbacks += 1
            text = request.fallback
        }
        val message = CoachMessage(
            text = text,
            level = request.level,
            trigger = request.trigger,
            gameId = request.gameId,
            ply = request.ply,
            fen = request.board.fen(),
            annotations = request.annotations + mentionsIn(text, request.board),
        )
        return Spoken(
            message = message,
            level = request.level,
            problems = problems,
            attempts = attempts,
            calls = calls,
            verified = verified,
            repaired = repaired,
            streamed = streamed,
        )
    }

    /**
     * The candidates read against the messages by the verifier: those with nothing
     * unsupported stay; when none does, the likeliest is repaired once and kept if the
     * repair passes the deterministic checks. Returns the candidates to sample from, what
     * the verifier found per candidate, and whether a repair was used.
     */
    private fun verify(
        passed: List<Pair<Double, String>>,
        request: CoachRequest,
    ): Triple<List<Pair<Double, String>>, List<List<String>>, Boolean> {
        val judge = verifier!!
        // The latest messages: the position the player faces, the one they moved from, the
        // dialogue since, the directive -- the facts a reply may draw on are in these.
        val context = request.messages.takeLast(5).joinToString("\n\n") { it.content }

        fun check(candidate: String): List<String> {
            val content = "The messages the coach was given (the latest):\n\n" + context +
                "\n\nThe coach's reply:\n\"\"\"\n" + candidate + "\n\"\"\"\n\nJSON only."
            val completion = try {
                judge.chat(
                    VERIFY_SYSTEM,
                    listOf(ChatMessage("user", content)),
                    maxTokens = 400,
                    temperature = 0.0,
                )
            } catch (_: Exception) {
                return emptyList() // the verifier failed: the candidate is not held against it
            }
            val found = jsonObject(completion.text)
            val items = found?.optJSONArray("unsupported") ?: return emptyList()
            return (0 until items.length()).map { items.opt(it).toString() }
                .filter { it.trim().isNotEmpty() }
        }

        val pool = Executors.newFixedThreadPool(maxOf(1, passed.size))
        val verified: List<List<String>> = try {
            pool.invokeAll(passed.map { (_, candidate) -> java.util.concurrent.Callable { check(candidate) } })
                .map { it.get() }
        } finally {
            pool.shutdown()
            pool.awaitTermination(1, TimeUnit.MINUTES)
        }
        val clean = passed.zip(verified).filter { it.second.isEmpty() }.map { it.first }
        if (clean.isNotEmpty()) return Triple(clean, verified, false)
        val pick = passed.zip(verified).maxByOrNull { it.first.first }!!.first
        val (prob, best) = pick
        var faults = verified[passed.indexOfFirst { it.second == best }]
        // Two tries at the repair: the second is also told what the deterministic checks
        // found wrong with the first.
        var note = ""
        repeat(2) {
            val content = "The messages the coach was given (the latest):\n\n" + context +
                "\n\nThe reply:\n\"\"\"\n" + best + "\n\"\"\"\n\nUnsupported statements: " +
                faults.joinToString("; ") + note + "\n\nJSON only."
            val completion = try {
                judge.chat(
                    REPAIR_SYSTEM,
                    listOf(ChatMessage("user", content)),
                    maxTokens = 500,
                    temperature = 0.0,
                )
            } catch (_: Exception) {
                return Triple(emptyList(), verified, false)
            }
            val found = jsonObject(completion.text)
            val mended = (found?.opt("reply")?.toString() ?: "").trim()
            if (mended.isEmpty()) return Triple(emptyList(), verified, false)
            val faultsNow = problemsOf(mended, request)
            if (faultsNow.isEmpty()) return Triple(listOf(prob to mended), verified, true)
            note = "\n\nYour previous edit was rejected for: " + faultsNow.joinToString("; ") + "."
        }
        return Triple(emptyList(), verified, false)
    }

    internal fun problemsOf(text: String, request: CoachRequest): List<String> {
        // What the user said this turn, what the directive itself disclosed (the engine on
        // what they asked about) and the moves of the game so far are not leaks: a move
        // already played may be discussed in full, whatever the level in force. The piece a
        // played move moved may be named too: "your opponent's rook took on c6" about a move
        // of the record is no hint, but the word "rook" was a leak whenever the withheld
        // piece happened to be a rook as well.
        val moved = request.played.filter { it.isNotEmpty() && it[0] != 'O' }
            .map { PIECE_WORD[it[0]] ?: "pawn" }
            .toSortedSet()
        val said = listOfNotNull(
            request.userText,
            request.disclosed.ifEmpty { null },
        ).plus(request.played.sorted()).plus(moved).filter { it.isNotEmpty() }.joinToString(" ")
        val found = ArrayList<String>()
        for (item in leakCheck(text, request.facts, request.level, said = said.ifEmpty { null })) {
            found.add("withheld $item")
        }
        // The length the directive states: a reply over it is a fault of its own, so a
        // shorter candidate wins the draw, or the likeliest is cut to size.
        val directive = if (request.messages.isNotEmpty()) {
            request.messages.last().content.substringAfterLast("# Task")
        } else {
            ""
        }
        val (limit, wantsQuestion) = directiveLimits(directive)
        if (limit != null && sentenceTotal(text) > limit + LENGTH_SLACK) {
            found.add("too long: ${sentenceTotal(text)} sentences, $limit asked for")
        }
        // The verdict word and the cost the reply gives for the move just played must be the
        // assessment's own: "best" for an excellent move, "cost 0.10" for a 0.05 loss, were
        // the judge's commonest remaining faults, and the coach holds the facts.
        found.addAll(verdictProblems(text, request))
        // The closing question the directive asks for: without this check the length rule
        // let a shorter candidate without one win the draw.
        if (wantsQuestion == true && !text.trimEnd().endsWith("?")) found.add("no closing question")
        if (wantsQuestion == false && text.trimEnd().endsWith("?")) {
            found.add("a question where the task allows none")
        }
        if (draftCheck != null) {
            for (problem in draftCheck.check(text, request.board, afterBest(request))) {
                val san = problem.substringBefore(":").trim()
                if (san in request.played) continue // a move of the game's history, named as such
                found.add(problem)
            }
        }
        return found
    }

    private fun run() {
        while (true) {
            val request = requests.take()
            if (request === Stop) return
            busy = true
            try {
                when (request) {
                    is Spoken -> results.add(request)
                    is CoachRequest -> results.add(speakNow(request))
                }
            } catch (exc: Exception) {
                failures += 1
                lastError = "${exc::class.java.simpleName}: ${exc.message}"
            } finally {
                busy = false
            }
        }
    }
}
