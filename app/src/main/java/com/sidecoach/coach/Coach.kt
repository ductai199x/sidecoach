package com.sidecoach.coach

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.STARTING_FEN
import com.sidecoach.chess.WHITE
import com.sidecoach.chess.squareName
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

/** The top of the rating band below which the templates say no term of art ([JARGON]). */
const val PLAIN_BELOW = 1400

/** The bottom of the rating band from which the templates add no textbook rule to a lesson. */
const val STRONG_FROM = 1800

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

/*
 * The templates said without a model are the coach on a phone with no model, and the floor the
 * model must rise above (Tai, 2026-09-23: "the template coach must be good already"). Each rung
 * says something true about this position that helps to find the answer: the idea (the goal,
 * the pattern and why it works), the focus (the target and the piece), the move (played out,
 * correct by construction). Nothing here could be said of any position: the stock questions it
 * replaces -- "what is your opponent's most dangerous idea?", "look at the whole board" -- were
 * said on a mate in one.
 */

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

/**
 * How the rule to keep is introduced, rotated like the openers (Tai, 2026-09-25: "there's a lot
 * of 'keep this one' ... it could use some variety"). The first is the templates' own.
 */
val KEEP_LEADS = listOf(
    "keep this one",
    "the lesson to keep",
    "worth remembering",
    "a rule to take with you",
    "the thing to remember",
)

/**
 * The pattern to keep, said after a mistake that missed it: the rule of thumb, not the
 * position. Keyed by the pattern's name as the fact sheet gives it.
 */
val PATTERN_LESSONS: Map<String, String> = linkedMapOf(
    "a fork" to "when two of their pieces can be hit from one square, look for the move that " +
        "gets there",
    "a skewer" to "a check or an attack on a line with something bigger behind it wins what is " +
        "behind",
    "a pin" to "a pinned piece cannot run, so hit it again",
    "a discovered attack" to "when one of your pieces blocks another's line, moving it with a " +
        "threat makes two threats at once",
    "a double check" to "against a double check the king must move, whatever else is attacked",
    "a trapped piece" to "a piece with no safe square is lost once it is attacked",
    "removing the defender" to "take the guard first, and what it guarded falls",
    "a deflection" to "a piece guarding two things guards neither once one is attacked",
    "a decoy" to "a sacrifice can drag a piece onto the square where it gets hit",
    "a loose piece" to "before every move, look at what they have left undefended",
    "a mating attack" to "when their king is short of squares, look at every check, the " +
        "sacrifices too",
)
private val MATES = listOf("mate", "Boden's")

/** Before the facts of the position have landed: an honest word, not a stock question. */
const val STILL_READING = "Give me a moment: I am still reading this position."

/** `re.match`: anchored at the start only. */
private val PAWN_MOVE = Regex("^[a-h](x[a-h])?[1-8]")

/** `text` as a sentence: a capital first, a stop last. */
internal fun templateSentence(raw: String): String {
    var text = raw.trim()
    if (text.isEmpty()) return ""
    if (!PAWN_MOVE.containsMatchIn(text)) {  // "f7+ forces mate", never "F7+"
        text = text[0].uppercaseChar() + text.substring(1)
    }
    return if (text.last() in ".!?") text else "$text."
}

/**
 * The first hint, without a model: the situation as a short lesson on the position -- what is
 * at stake, the key fact named on the board, the principle behind it, the trap ([Situation]).
 * Never the move.
 */
fun ideaTemplate(facts: PositionFacts): String {
    if (!facts.situation.isNullOrEmpty()) return facts.situation
    val parts = ArrayList<String>()
    if (!facts.goal.isNullOrEmpty()) parts.add(templateSentence(facts.goal))
    if (!facts.question.isNullOrEmpty()) parts.add(facts.question)
    return if (parts.isNotEmpty()) parts.joinToString(" ") else STILL_READING
}

/**
 * The second hint, without a model: the piece that does it, and the question that leads from
 * there to the move. The target was the first hint's.
 */
fun focusTemplate(facts: PositionFacts): String {
    val parts = ArrayList<String>()
    if (!facts.target.isNullOrEmpty() &&
        facts.target.startsWith("their king on") &&
        !facts.goal.isNullOrEmpty() &&
        ("mate" in facts.goal || "attack" in facts.goal)
    ) {
        // A mate's or an attack's squares, held back from the first hint: one may be where the
        // check lands.
        parts.add(templateSentence("look at ${facts.target}"))
    }
    if (!facts.piece.isNullOrEmpty()) parts.add(templateSentence("use ${facts.piece}"))
    if (!facts.nudge.isNullOrEmpty()) {
        parts.add(facts.nudge)
    } else if (!facts.question.isNullOrEmpty() && parts.isNotEmpty() && facts.question !in Situation.ANY_POSITION) {
        // Questions that fit any position narrow nothing: never the second hint's.
        parts.add(facts.question)
    }
    return if (parts.isNotEmpty()) parts.joinToString(" ") else ideaTemplate(facts)
}

/** The move rung, without a model: the move, played out and explained. */
fun moveTemplate(facts: PositionFacts): String {
    if (!facts.why.isNullOrEmpty()) return templateSentence(facts.why)
    if (facts.candidates.isEmpty()) return "I have no move for this yet."
    if (!facts.plan.isNullOrEmpty()) return templateSentence("the move is ${facts.candidates[0]}: ${facts.plan}")
    return "The move is ${facts.candidates[0]}."
}

fun lineTemplate(facts: PositionFacts): String {
    if (facts.line.isEmpty()) return "I have no line for this yet."
    val text = "The line is ${facts.line.joinToString(" ")}."
    return if (!facts.why.isNullOrEmpty()) "$text ${templateSentence(facts.why)}" else text
}

/** The template for a hint at `level`: two hints, then the answer. */
fun hintTemplate(facts: PositionFacts, level: DisclosureLevel): String {
    if (level.value >= DisclosureLevel.LINE.value) return lineTemplate(facts)
    if (level.value >= DisclosureLevel.MOVE.value) return moveTemplate(facts)
    if (level.value >= DisclosureLevel.FOCUS.value) return focusTemplate(facts)
    return ideaTemplate(facts)
}

/** The verbs a move's "why" goes on with after its move: said of the move, they take "it". */
private val WHY_VERBS = listOf(
    "is ",
    "forces ",
    "stops ",
    "starts ",
    "holds ",
    "takes ",
    "goes ",
    "meets ",
    "puts ",
    "threatens ",
    "doubles ",
    "brings ",
    "rolls ",
    "pushes ",
    "gains ",
)

/** A "why" that names its move a sacrifice first: "a queen sacrifice, and after Qxf7, ...". */
private val SACRIFICE = Regex("^(an? [a-z]+ sacrifice), (.+)", RegexOption.DOT_MATCHES_ALL)

/**
 * When the player made the right capture with the other piece: "take back with the rook on e1,
 * not the one on a1" (13.Raxd1 for Rexd1 was told "you win their queen", which the player had
 * just done, Tai's game, 2026-09-24). `coach._recapture`.
 */
private fun recapture(board: Board?, best: String, played: String?): String? {
    if (board == null || played == null) return null
    val good = board.parseSan(best) ?: return null
    val done = Move.fromUciOrNull(played) ?: return null
    if (good.to != done.to ||
        good.from == done.from ||
        !board.isCapture(good) ||
        !board.isCapture(done)
    ) {
        return null
    }
    val piece = board.pieceAt(good.from) ?: return null
    val other = board.pieceAt(done.from) ?: return null
    val mine = COACH_PIECE_NAMES[piece.pieceType]
    val theirs = if (other.pieceType == piece.pieceType) "the one" else "the ${COACH_PIECE_NAMES[other.pieceType]}"
    return "take with the $mine on ${squareName(good.from)}, not $theirs on ${squareName(done.from)}"
}

/**
 * After a move that cost something: the better move, what it does, what the move made allowed,
 * and the pattern to keep. `opener` is the rotated "That one cost you: {best} was better";
 * `board` and `played` (UCI) are the position the player faced and the move made, `allowed`
 * what it let the opponent do ([IdeaWords.allowed]), when known, and `keepLead` how the rule to
 * keep is introduced ([KEEP_LEADS], rotated by the coach). `strong` (a player of 1800 and up)
 * leaves the rule out: the concrete point is the lesson.
 */
fun lessonTemplate(
    facts: PositionFacts,
    best: String,
    opener: String,
    board: Board? = null,
    played: String? = null,
    allowed: String? = null,
    keepLead: String = KEEP_LEADS[0],
    strong: Boolean = false,
): String {
    var text = opener
    val why = facts.why ?: ""
    val otherPiece = recapture(board, best, played)
    if (otherPiece != null) {
        text += " -- $otherPiece"
        if (!allowed.isNullOrEmpty()) text = templateSentence(text) + " " + allowed
        if (strong) return templateSentence(text)
        return templateSentence(text) + " " +
            templateSentence("$keepLead: when two pieces can take, ask which you want to keep where it is")
    }
    if (why.startsWith(best)) {
        var rest = why.substring(best.length).trimStart(' ', ',', ':', ';')
        if (rest.startsWith("and ") && !rest.startsWith("and after")) {
            rest = why  // "Rfc8 and Qc6 double your rooks": the plan, said whole
        } else {
            rest = if (rest.startsWith("and ")) rest.substring(4) else rest
            val sacrifice = SACRIFICE.find(rest)
            if (sacrifice != null) {
                // "axb3, a knight sacrifice, stops bxc2": "-- it is a knight sacrifice, and it
                // stops bxc2" (the master games, Sokolov-Dreev, 2026-09-24).
                val named = sacrifice.groupValues[1]
                var tail = sacrifice.groupValues[2]
                if (WHY_VERBS.any { tail.startsWith(it) }) tail = "it $tail"
                rest = "it is $named, " + if (tail.startsWith("and ")) tail else "and $tail"
            } else if (WHY_VERBS.any { rest.startsWith(it) }) {
                rest = "it $rest"  // "Nf2# was better -- it is mate", "-- it stops Nxd3"
            }
        }
        if (rest.isNotEmpty() && stale(rest, played)) rest = ""
        if (rest.isNotEmpty()) text += " -- $rest"
    }
    text = templateSentence(text)
    if (!allowed.isNullOrEmpty()) text += " " + templateSentence(allowed)
    val theme = facts.theme ?: ""
    var keep = PATTERN_LESSONS[theme]
    if (keep == null && MATES.any { it in theme }) keep = PATTERN_LESSONS.getValue("a mating attack")
    if (keep == null && !facts.plan.isNullOrEmpty() && !facts.mechanism.isNullOrEmpty()) {
        keep = facts.mechanism.split(": ", limit = 2).last()
    }
    if (!keep.isNullOrEmpty() && !strong && !said(keep, text)) text += " " + templateSentence("$keepLead: $keep")
    return text
}

/**
 * `coach._stale`: whether a lesson's reason, read off the position the player faced, names a
 * piece on a square their move then left or landed on -- the position before it, not the one
 * now: "d4 was a little more precise -- your bishop on c1 is still shut in behind the pawn on d2"
 * after 4.d3, which had just let it out (Nova bot games, 2026-09-25). A line ("after Qg2, Rdg1,
 * the queen on g2 is trapped") is about the moves to come, and stands.
 */
internal fun stale(reason: String, played: String?): Boolean {
    if (played == null || reason.startsWith("after ")) return false
    val move = Move.fromUciOrNull(played) ?: return false
    val squares = listOf(squareName(move.from), squareName(move.to))
    return squares.any { sq -> Regex("\\bon $sq\\b").containsMatchIn(reason) }
}

/**
 * A lesson the player heard over this many plies (their last three moves) is not said again: in
 * amateur games "fine, but not the best" comes move after move, and the same "keep this one"
 * came with it -- 27% of them within three moves in twenty Nova bot games (2026-09-25).
 */
const val RECENT_PLIES = 6

/** `fresh`'s own split: `re.split(r"(?<=[.?!])\\s+", ...)`. */
private val LESSON_SENTENCES = Regex("(?<=[.?!])\\s+")

/**
 * `coach.fresh`: [text] without what the coach said lately ([recent], lower case): the "Keep
 * this one" sentence, and the reason after " -- " in the first sentence. The verdict and the
 * better move always stay: that is news. [opener] is what the first sentence opens with, the
 * verdict and the better move; the reason is what follows it.
 */
fun fresh(text: String, recent: String, opener: String? = null): String {
    val said = recent.lowercase()
    val sentences = text.trim().split(LESSON_SENTENCES)
    if (sentences.isEmpty()) return text
    var first = sentences[0]
    val rest = sentences.drop(1)
    var head = first
    var reason = ""
    if (opener != null) {
        // An opener may carry its own " -- " ("Not that one -- Ng3 was better"): with no reason
        // after it, the better move was taken for one and dropped when the same move had just
        // been said (Nova bot games, 2026-09-25).
        if (first.lowercase().startsWith(opener.lowercase() + " -- ")) {
            head = first.substring(0, opener.length)
            reason = first.substring(opener.length + 4)
        }
    } else if (" -- " in first) {
        // The last clause ("Playable -- d4 was a little more precise -- your bishop on c1 is
        // still shut in ...").
        head = first.substringBeforeLast(" -- ")
        reason = first.substringAfterLast(" -- ")
    }
    if (reason.isNotEmpty() && reason.trimEnd('.').lowercase() in said) first = head.trimEnd('.') + "."
    val kept = rest.filter { s -> val rule = kept(s); rule == null || rule !in said }
    return (listOf(first) + kept).joinToString(" ")
}

/** `coach._kept`: the rule a "keep this one" sentence gives, in lower case, whichever lead it has. */
private fun kept(sentence: String): String? {
    val low = sentence.lowercase()
    for (lead in KEEP_LEADS) {
        if (low.startsWith("$lead:")) return low.substring(lead.length + 1).trim().trimEnd('.')
    }
    return null
}

private val WORDS = Regex("[a-z']+")

/** Whether `text` already says `keep`: four of its words in a row. `coach._said`. */
fun said(keep: String, text: String): Boolean {
    val words = WORDS.findAll(keep.lowercase()).map { it.value }.toList()
    val said = WORDS.findAll(text.lowercase()).joinToString(" ") { it.value }
    return (0 until words.size - 3).any { j -> words.subList(j, j + 4).joinToString(" ") in said }
}

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
        "figures for when the player asks. Name moves bare, never with a move number in front."

/**
 * The player has just played the move and sees it on the board: naming it, or saying what it
 * plainly did ("Nxd5 -- the knight taking the pawn on d5"), tells them nothing (Tai,
 * 2026-09-23: "we want the coach to be instructive, always find a moment to teach something,
 * not just a dumb bot").
 */
const val NOT_THE_MOVE =
    "Do not open with the move they played, and do not say what it plainly did on the board " +
        "-- what it took, the check, the castling, the promotion: they just played it and see it. " +
        "Say what it means instead: the idea behind it and the pattern to keep -- and, when there " +
        "was a better move, that move and why."

/**
 * A hint, and an answer that is not about a figure: the code faults one ([Speaker.problemsOf]),
 * so the directive says so -- the stuck hint was faulted for "ahead by about 1.2 pawns" and
 * never told.
 */
const val NO_FIGURES = "Quote no figure -- no percentage, no score, no move number -- and name moves bare."

/*
 * What the model is asked to do on each rung before the player moves. The template for the
 * rung is the floor; the model is there to rise above it (Tai, 2026-09-23: "AI is supposed to
 * elevate good to great"): the same facts, in its own words, sharper for this position, and
 * tied to the player's game where that helps.
 */
const val IDEA_DIRECTIVE =
    "Give the first hint as a short lesson on this position, from the fact sheet: what is at " +
        "stake (its stakes: how many moves hold, and what even the next best one costs -- or its " +
        "goal, where the sheet gives no stakes, or its nature -- whether there is a tactic to find " +
        "-- where it gives neither), then the key fact on the board, named as the " +
        "sheet names it -- the pieces, the squares, their threat, the target, the plan, the " +
        "pattern -- and what it does, threatens or allows, with the principle behind it in a " +
        "clause, and the trap if the sheet gives one, so they know what to look for; then, if " +
        "you like, one question that sets them looking. Precise, never a principle on its own: " +
        "every sentence says " +
        "something about this board. Never name the player's move, and not which of their own " +
        "pieces makes it -- that is the next hint -- unless the key fact names that piece on its " +
        "square; the target's square, as the sheet names it, " +
        "may be said even where the move goes, and the moves the sheet lists as all as good, " +
        "when several are, may be named. Two or three sentences."

/**
 * The game so far, used: an earlier moment of the same idea is the coach's best material, and the
 * excellence rubric found it used in half the lessons that had one (2026-09-25). The coach finds
 * those moments itself ([Conversation.sameIdea]) and lists them: told to find its own in the
 * record, the model misremembered one ("as after 9...a6, when Bc4 saved your bishop": neither
 * happened) and copied the directive's own example move as a link (quality rounds 2 and 3,
 * 2026-09-25). TIE_TO_THE_GAME marks the directives a link belongs in; the coach's speak puts the
 * rule for the moments listed, or for none, in its place.
 */
const val TIE_TO_THE_GAME = "Tie this lesson to the game only through the earlier moments listed."
const val TIE_LISTED =
    "Under 'Earlier in this game' are the player's earlier moves played with the very same " +
        "idea as this one. Tie this lesson to the one that fits best, in a clause -- naming its " +
        "move bare, as the list does -- and say of it only what the list says. Speak of no other " +
        "earlier moment."
const val NO_EARLIER = "Speak of no earlier moment of the game: none had the same idea."

private val LENGTH_WORDS = mapOf(4 to "Four", 5 to "Five", 6 to "Six")

/**
 * [IDEA_DIRECTIVE], its length the first hint's own when that runs past three sentences (to six
 * at most): "say at least this" and "two or three sentences" asked for both, and a faithful reply
 * was cut or sent back as too long (the eval audit, 2026-09-25). `coach.idea_directive`.
 */
fun ideaDirective(facts: PositionFacts?): String {
    val n = if (facts != null && !facts.situation.isNullOrEmpty()) sentenceTotal(facts.situation) else 0
    if (n <= 3) return IDEA_DIRECTIVE
    return IDEA_DIRECTIVE.replace(
        "Two or three sentences.", "${LENGTH_WORDS[n] ?: "Six"} sentences at most."
    )
}

const val FOCUS_DIRECTIVE =
    "Name the piece to use and what it goes after -- the target -- as the fact sheet names " +
        "them, with the principle behind it in a clause, and ask the one question that leads from " +
        "there to the move. Do not name the move. One or two sentences."
const val MOVE_DIRECTIVE =
    "The next hint is the answer: give the move and explain it correctly from the fact " +
        "sheet's why -- what it does and what it leads to -- and the pattern to keep. One or two " +
        "sentences, no question."

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

private val ILLEGAL_CLAIM = Regex(
    "\\b(O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]? (?:is|was) " +
        "(?:not (?:a )?legal|illegal)",
)
private val CLAUSE_END = Regex("(?<=[.!?;])\\s+")

/**
 * A move the reply calls illegal that is legal for the side to move on the board it is about.
 * The AI said "O-O is not legal here because your opponent's bishop is still on e2", "Qxf7+ is
 * illegal here" and "Kf7 is illegal because f7 is controlled by the King on e8" of legal moves
 * (its own sessions, checked against the boards, 2026-09-26). A claim inside a line ("after
 * 22...Bxg5, 23.Bxg5 is illegal") is about a later position and is left alone, as is a move of
 * the side not to move. `coach.false_illegal`.
 */
fun falseIllegal(text: String, board: Board): List<String> {
    val out = ArrayList<String>()
    for (sentence in text.split(CLAUSE_END)) {
        for (found in ILLEGAL_CLAIM.findAll(sentence)) {
            if ("after" in sentence.substring(0, found.range.first).lowercase()) continue
            val san = found.groupValues[1]
            if (board.parseSan(san) == null) continue
            out.add("$san: called illegal, but it is a legal move here")
        }
    }
    return out
}

/** A reply's clauses: a sentence, or the halves of one joined by " -- " or ", but / though". */
private val CLAUSE = Regex("(?<=[.;:!?])\\s+|\\s+--\\s+|,\\s+(?:but|though|while)\\s+")

/** A clause about the move just played without naming it: "That was ...", "It cost ...". */
private val ABOUT_IT = Regex("^\\W*(?:that|it|this|your (?:last )?move)\\b", RegexOption.IGNORE_CASE)

/** `re.split`: the pieces between the matches, empty ones kept. */
private fun splitClauses(text: String): List<String> = text.split(CLAUSE)

/**
 * Faults in what a reply says of the move just assessed: a "best" that was not, a slip word
 * for a sound move, a cost that is not the recorded one -- clause by clause, where the clause is
 * about that move (it names it, or opens with "that" or "it"). Every lesson names the better
 * move too, and skipping any reply that did let "That was a blunder: a3 was better" through for a
 * good move (the runtime audit, 2026-09-25).
 */
internal fun verdictProblems(text: String, request: CoachRequest): List<String> {
    val a = request.assessment ?: return emptyList()
    val played = request.playedSan
    if (played.isNullOrEmpty()) return emptyList()
    val core = played.trimEnd('+', '#')
    val namesIt = Regex("(?<![A-Za-z])" + Regex.escape(core) + "[+#]?(?![A-Za-z0-9])")
    val label = a.label?.name?.lowercase() ?: a.severity.name.lowercase()
    // A move that cost nothing is as good as the best: "the best move on the board" for a great
    // move the engine chose itself is true.
    val bestOk = label == "best" || label == "excellent" || a.expectedLoss < 0.005
    val problems = ArrayList<String>()
    for (clause in splitClauses(text)) {
        if (!(namesIt.containsMatchIn(clause) || ABOUT_IT.containsMatchIn(clause))) continue
        val others = SAN.findAll(clause).map { it.groupValues[1] }
            .filter { it.trimEnd('+', '#') != core }
            .toList()
        if (others.isEmpty() && !bestOk && SAYS_BEST.containsMatchIn(clause) &&
            !Regex(
                "\\bnot (?:the )?(?:best|strongest)\\b|\\bwas better\\b|\\bwas stronger\\b",
                RegexOption.IGNORE_CASE,
            ).containsMatchIn(clause)
        ) {
            problems.add("called $played the best move; its verdict is $label")
        }
        if (a.severity == Severity.NONE && SAYS_SLIP.containsMatchIn(clause) &&
            !Regex(
                "\\bnot (?:a )?(?:blunder|mistake)\\b|\\bno (?:blunder|mistake)\\b",
                RegexOption.IGNORE_CASE,
            ).containsMatchIn(clause)
        ) {
            problems.add("called $played a slip; its verdict is $label")
        }
        for (found in SAYS_COST.findAll(clause)) {
            var value = found.groupValues[1].toDouble()
            if (found.groupValues[2].isNotEmpty() || value > 1.0) value /= 100.0
            if (kotlin.math.abs(value - a.expectedLoss) > maxOf(0.02, 0.35 * a.expectedLoss)) {
                problems.add(
                    "gave the cost of $played as ${found.groupValues[1]}; it was " +
                        formatted(a.expectedLoss, 2)
                )
            }
        }
    }
    return problems.distinct()
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
    if (Regex("\\bthen, if you like, one question\\b").containsMatchIn(low)) {
        // A question offered on top of the sentences: one more is allowed, none required.
        limit = limit?.plus(1)
    } else if (Regex("\\bthen one question\\b").containsMatchIn(low)) {
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
 * Markdown the model writes around words ("**Rc1**", "`e4`", "__now__") and at the start of a
 * line ("# ", "- ", "* "): the reply is spoken and shown as it is, and "?**" at its end failed the
 * closing-question check (the parity pilot, 2026-09-25).
 */
private val EMPHASIS = Regex("\\*\\*|__|`")
private val LINE_MARK = Regex("^\\s*(?:#{1,6}\\s+|[-*]\\s+(?=\\S))", RegexOption.MULTILINE)

/** [text] without markdown: the emphasis marks and the line marks go, the words stay. `coach.plain`. */
fun plain(text: String): String = LINE_MARK.replace(EMPHASIS.replace(text, ""), "")

/**
 * The verbalized replies as (probability, text). A reply with no markers is one candidate.
 *
 * The text of each candidate runs from its marker to the next, so a reply that wraps onto
 * another line is kept whole. A model that ignores the format still yields one usable reply.
 * Every candidate is cut back to its last complete sentence.
 */
internal fun parseCandidates(raw: String): List<Pair<Double, String>> {
    var cleaned = plain(THINK_BLOCK.replace(raw, "")).trim()
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
    /**
     * What the whole reply must name (held until it comes), and the task's length (the stream is
     * closed there): with the voice on, the reply was never checked whole, and "Mate is close."
     * and an eight-sentence homily both went out (the runtime audit).
     */
    private val mustName: List<String> = emptyList(),
    private val limit: Int? = null,
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
        val cleaned = plain(THINK_BLOCK.replace(buffer, ""))
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
        var ends = SENTENCE_END.findAll(cleaned)
            .filter { final || it.range.last + 1 < cleaned.length }.toList()
        if (limit != null) {
            // Cut at the task's length before anything is said, however large the chunk.
            ends = ends.take(limit + LENGTH_SLACK)
        }
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
        if (mustName.isNotEmpty() && spoken.isEmpty()) {
            // Nothing goes out before what the reply must name: held until it comes, and a reply
            // that ends without it is not said at all -- it is read whole instead.
            val leftOut = omitted(complete, mustName)
            if (leftOut != null) {
                if (final) {
                    problems = listOf(leftOut)
                    return false
                }
                return true
            }
        }
        var faults = if (fresh.isNotEmpty()) check(fresh) else emptyList()
        // Sentence by sentence, only what a sentence can get wrong: the length, the closing
        // question and what the whole must name are the whole reply's.
        faults = faults.filter {
            !it.startsWith("too long") && !it.startsWith("left out") && it != "no closing question" &&
                // Only the reply's first sentence can open it with the move.
                !(spoken.isNotEmpty() && it.startsWith("opens with"))
        }
        if (faults.isNotEmpty()) {
            problems = faults
            return false
        }
        if (!sink(fresh)) {
            declined = true
            return false
        }
        spoken = complete
        // As long as the task allows: the rest is not said.
        return limit == null || sentenceTotal(spoken) < limit + LENGTH_SLACK
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
) {
    /**
     * The first hint's wordings, peeked from the coach's bags as the facts were written and taken
     * from them when the hint is said ([Bags]).
     */
    var leads: Map<String, String> = emptyMap()
}

/**
 * The board after [event], carrying the moves that led to it: the idea reader reads an exchange
 * in progress from them ([IdeaReader.exchangeOwed]), and a board rebuilt from the FEN alone read
 * 20...Nxh5 Qxh5 as "you can win a knight" (Tai's puzzles, 2026-09-26). The move goes onto
 * [board] when that is where it was played from, else onto the position it was played from; a
 * board whose FEN is not the event's own (the analysis is matched on it) is the event's FEN,
 * without moves. `coach.board_after`.
 */
fun boardAfter(board: Board, event: MovePlayed): Board {
    for (start in listOf(board, Board(event.fenBefore))) {
        if (start.fen() != event.fenBefore) continue
        val move = Move.fromUciOrNull(event.uci) ?: break
        if (!start.isLegal(move)) break
        val after = start.copy()
        after.push(move)
        if (after.fen() == event.fenAfter) return after
        break
    }
    return Board(event.fenAfter)
}

/** The board taken back to [fen], keeping the moves before it when they reach it. `coach.board_back`. */
fun boardBack(board: Board, fen: String): Board {
    val back = board.copy()
    while (back.moveStack.isNotEmpty() && back.fen() != fen) back.pop()
    return if (back.fen() == fen) back else Board(fen)
}

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

/**
 * How long the lesson on a mistake waits for the analysis of the position the move left: the
 * verdict comes first, from the best line alone, and the analysis about a second after it --
 * without it the lesson had no "what your move allowed" (the parity audit, 2026-09-25).
 */
const val LESSON_WAIT_S = 1.5

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
    /**
     * What a reply must name, one of: the better move the lesson's template names, or a square
     * the hint's template names. The template is the floor; a true reply that says less than it
     * was a loss the checks let through (parity pass, 2026-09-25).
     */
    val mustName: List<String> = emptyList(),
    /**
     * More boards a statement may be true on: after the move the player just made, and along the
     * engine's line from there -- the board the model is shown for a lesson after a move.
     */
    val frames: List<Board> = emptyList(),
    /**
     * When the reply is about a position already moved from and the player is to move again: the
     * answer there (its best move and line), which the reply must not give away -- a late
     * verdict's lesson said "and now Qxf7# is mate" (the runtime audit, 2026-09-25).
     */
    val nowAnswer: List<String> = emptyList(),
    /** The game's moves as the record numbers them ("17. Rh1"): said so, they are history. */
    val record: Set<String> = emptySet(),
    /**
     * For a directive a link belongs in ([TIE_TO_THE_GAME]): the earlier moves of the game the
     * reply may not name -- all but those the list of earlier moments and the sheets give. Null
     * when the directive has no place for a link.
     */
    val unlisted: Set<String>? = null,
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
    return frames + request.frames
}

private val SQUARE_ITEM = Regex("[a-h][1-8]")

/** Whether [text] names [item]: a square, or a move in SAN (with or without its mark). `coach._names`. */
internal fun names(text: String, item: String): Boolean {
    if (SQUARE_ITEM.matches(item)) return item in squaresIn(text)
    val core = item.trimEnd('+', '#')
    if (core.startsWith("O-O")) {
        // Castling: "O-O" is not named by "O-O-O" (the recheck of round 3's replies).
        val pattern = "(?<![-A-Za-z0-9])" + Regex.escape(core) + "(?!-O)[+#]?(?![A-Za-z0-9])"
        return Regex(pattern).containsMatchIn(text)
    }
    return Regex("(?<![A-Za-z])" + Regex.escape(core) + "[+#]?(?![A-Za-z0-9])").containsMatchIn(text)
}

private val ENGINE_NAMED = Regex("\\b(?:stockfish|the engine|engine's|an engine)\\b", RegexOption.IGNORE_CASE)
private val ENGINE_ASKED = Regex("stockfish|engine", RegexOption.IGNORE_CASE)

private val ORDINAL = Regex("\\b\\d+(?:st|nd|rd|th)\\b|\\b[A-E]\\d\\d\\b")
private val DIGIT = Regex("\\d")

/**
 * Whether [text] quotes a figure: any digit outside a move, a square or an ordinal ("the 7th
 * rank") -- "ahead by about 1.2 pawns" went through a check for percentages alone.
 * `coach.figure_said`.
 */
fun figureSaid(text: String): Boolean {
    var rest = SAN.replace(text, " ")
    rest = ORDINAL.replace(rest, " ")
    return DIGIT.containsMatchIn(rest)
}

private val PLAIN_PIECE = mapOf(
    com.sidecoach.chess.PAWN to "pawn",
    com.sidecoach.chess.KNIGHT to "knight",
    com.sidecoach.chess.BISHOP to "bishop",
    com.sidecoach.chess.ROOK to "rook",
    com.sidecoach.chess.QUEEN to "queen",
)
private val SAYS_CHECK = Regex("\\b(?:with|gives?|giving) (?:a )?check\\b")

/**
 * What a word after the player's move says that they see on the board or did not ask for (Tai,
 * 2026-09-23 and 2026-09-19): opening with the move just played, what it plainly did -- the
 * capture, the check, the castling, the promotion -- unless the better move does the same, and a
 * figure in an unprompted remark. The templates never do; the model could, and nothing checked
 * it (the eval audit, 2026-09-25). `coach.restated`.
 */
internal fun restated(text: String, request: CoachRequest): List<String> {
    val playedSan = request.playedSan
    if ((request.trigger != Trigger.POST_MOVE && request.trigger != Trigger.POST_MISTAKE) ||
        playedSan.isNullOrEmpty()
    ) {
        return emptyList()
    }
    val played = playedSan.trimEnd('+', '#')
    val faults = ArrayList<String>()
    if (Regex("^\\W*" + Regex.escape(played) + "(?![A-Za-z0-9])").containsMatchIn(text)) {
        faults.add("opens with $played, the move just played")
    }
    if (figureSaid(text)) {
        faults.add("a figure in a remark nobody asked for: say it in words")
    }
    val board = request.board
    val move = board.parseSan(playedSan) ?: return faults
    val assessment = request.assessment
    val better = if (assessment != null && assessment.best.isNotEmpty()) {
        Move.fromUciOrNull(assessment.best[0])
    } else {
        null
    }
    val low = text.lowercase()
    val victim = board.pieceAt(move.to)
    if (victim != null && victim.pieceType in PLAIN_PIECE) {
        val square = squareName(move.to)
        val name = PLAIN_PIECE.getValue(victim.pieceType)
        val verb = "\\b(?:takes|taking|took|take|captures?|capturing|grabs?|grabbing|wins?|winning|won)"
        val took = "$verb\\s+(?:the|their|a|that)\\s+(?:$name\\s+on\\s+$square|$square\\s+$name)\\b" +
            "|\\b(?:takes|taking|took|take)\\s+on\\s+$square\\b"
        if (Regex(took).containsMatchIn(low) && !(better != null && better.to == move.to)) {
            faults.add("says what $played plainly did: the capture on $square")
        }
    }
    // What the better move does as well is the lesson, not a restatement.
    val legal = better != null && board.isLegal(better)
    val betterCastles = legal && board.isCastling(better!!)
    val betterChecks = legal && board.givesCheck(better!!)
    val betterPromotes = better != null && better.promotion != 0
    if (board.isCastling(move) && "castl" in low && !betterCastles) {
        faults.add("says what $played plainly did: castling")
    }
    if (board.givesCheck(move) && SAYS_CHECK.containsMatchIn(low) && !betterChecks) {
        faults.add("says what $played plainly did: the check")
    }
    if (move.promotion != 0 && "promot" in low && !betterPromotes) {
        faults.add("says what $played plainly did: the promotion")
    }
    return faults
}

/**
 * The game's history in a request, up to the next section: the moves that stand only there are
 * the earlier moments ([unlistedLinks]).
 */
internal val GAME_SO_FAR = Regex("# Game so far\\n.*?(?=\\n# )", RegexOption.DOT_MATCHES_ALL)

/** A move with its number in front: "after 9...a6", "12. Qc2". */
private val MOVE_NUMBER = Regex("\\b\\d+\\s*\\.(?:\\s*\\.\\.)?\\s*(?:O-O|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8])")

/**
 * What a lesson says of an earlier moment the coach did not list: a move of the game before the
 * position taught that neither the list of earlier moments nor the sheets give, and any move with
 * its number -- "the same safety lesson as after 9...a6, when Bc4 saved your bishop" named a
 * moment that never happened (quality round 3, 2026-09-25). `coach.unlisted_links`.
 */
fun unlistedLinks(text: String, unlisted: Set<String>): List<String> {
    val faults = unlisted.sorted().filter { names(text, it) }
        .map { "names $it, an earlier moment not listed: tie only to the moments listed" }
        .toMutableList()
    if (MOVE_NUMBER.containsMatchIn(text)) faults.add("a move with its number: name moves bare")
    return faults
}

/**
 * The fault when [text] names none of [mustName]; null when it names one, or there is nothing
 * it must name. `coach.omitted`.
 */
fun omitted(text: String, mustName: List<String>): String? {
    if (mustName.isEmpty() || mustName.any { names(text, it) }) return null
    if (mustName.size == 1) return "left out ${mustName[0]}: the reply must name it"
    return "left out the key fact: the reply must name one of ${mustName.joinToString(", ")}"
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
                val (limit, _) = directiveLimits(request.messages.last().content.substringAfterLast("# Task"))
                reader = FirstCandidate(
                    request.streamTo,
                    { t -> problemsOf(t, request) },
                    mustName = request.mustName,
                    limit = limit,
                )
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
        // What the user said and what the directive disclosed are theirs already. The moves of
        // the game are no exemption: only the answer is withheld ([leakCheck]), and exempting
        // the record's moves, squares and pieces had let the answer through.
        val said = listOfNotNull(request.userText, request.disclosed)
            .filter { it.isNotEmpty() }.joinToString(" ")
        // What the sheet says the threats do ("Ne6 (uncovers an attack on the pawn on d2)") is
        // shown to the model: a piece named there is the threat, not the hint.
        val threatWords = request.facts.threats.joinToString(" ") { Conversation.threatWords(request.board, it) }
        val found = ArrayList<String>()
        for (item in leakCheck(
            text,
            request.facts,
            request.level,
            said = said.ifEmpty { null },
            record = request.record,
            shown = threatWords,
        )) {
            found.add("withheld $item")
        }
        for (move in request.nowAnswer) {
            if (names(text, move)) found.add("gives away $move in the position they face now")
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
        val leftOut = omitted(text, request.mustName)
        if (leftOut != null) found.add(leftOut)
        found.addAll(restated(text, request))
        val unlisted = request.unlisted
        if (unlisted != null) found.addAll(unlistedLinks(text, unlisted))
        // A figure where none was asked for: an unprompted remark, and an answer whose directive
        // keeps the figures for when the player asks -- "46...Rh5 was excellent, costing only
        // 0.01" (quality round 3, 2026-09-25).
        val unasked = request.trigger == Trigger.DECISIVE_QUESTION || request.trigger == Trigger.STUCK || (
            request.trigger == Trigger.USER_ASK && (NO_NUMBERS in directive || NO_FIGURES in directive)
            )
        if (unasked && figureSaid(text)) {
            found.add("a figure in a remark nobody asked for: say it in words")
        }
        // RULES 7: the coach speaks as the one who saw the game -- "Stockfish gives White about
        // 0.8 pawns" came in seven of the pilot's answers.
        val askedEngine = ENGINE_ASKED.containsMatchIn(request.userText ?: "")
        if (ENGINE_NAMED.containsMatchIn(text) && !askedEngine) {
            found.add("named the engine: say it as the coach who saw the game")
        }
        if (draftCheck != null) {
            for (problem in draftCheck.check(text, request.board, afterBest(request))) {
                val san = problem.substringBefore(":").trim()
                if (san in request.played) continue // a move of the game's history, named as such
                found.add(problem)
            }
        }
        found.addAll(falseIllegal(text, request.board))
        return found.distinct()
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
