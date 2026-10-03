package com.sidecoach.voice

/**
 * The coach's voice follows the player's pace ("Follow my pace", `voice.pace = "follow"`):
 * `sidecoach/voice/pace.py`, line for line. What the player hears, tuned by the desktop's
 * `tools/voice_sim.py` over puzzles and games played slowly and fast, the move found and missed
 * (Tai, 2026-10-03):
 *
 * - slow play: every word, whole;
 * - a fast line: nothing during it; once the line is done (no move for [LINE_END_S], or the next
 *   puzzle is up), one sentence on the move that took the thought, if it is
 *   [LINE_WORD_FRESH_S] young;
 * - a mistake: its lesson, even after one more move of the player's (a premove), its first
 *   sentence only once they have moved on, never after a retry of that move, never older than
 *   [LESSON_FRESH_S];
 * - the player's question: the answer, whole, whatever they play meanwhile; the game's review too;
 * - a hint: only while its position is on the board;
 * - a short word (a template) at a fast pace: said when the move took [SHORT_OK_S] or more;
 * - no sentence cut by a move, but a hint whose position is gone and a lesson whose move is
 *   taken back; nothing about a move older than its freshness.
 *
 * Pure logic on an injected clock (seconds); thread-safe: every call takes the plan's lock.
 */
object Pace {
    const val QUICK_S = 3.0
    const val LINE_END_S = 2.5
    /** The model's reply comes 4-5 s after the move; while the player has not moved since, it is
     * still the news (the desktop's bench, 2026-10-03). */
    const val PRAISE_FRESH_S = 9.0
    const val LINE_WORD_FRESH_S = 9.0
    const val LESSON_FRESH_S = 15.0
    const val LESSON_MOVES = 1
    const val REVIEW_FRESH_S = 30.0
    const val SHORT_OK_S = 4.0
    const val FAST_PACE_S = 6.0
    const val PACE_MOVES = 4

    val PRIORITY = mapOf("answer" to 0, "review" to 1, "lesson" to 2, "praise" to 3, "short" to 3, "hint" to 4)

    const val SAY = "say"
    const val HOLD = "hold"
    const val TEXT = "text"

    /** The plan's kind of a coach message: by its trigger, and for a word on a move whether the
     * model wrote it (streamed) or a template did. `kind_of`. */
    fun kindOf(trigger: String, streamed: Boolean): String = when (trigger) {
        "user_ask" -> "answer"
        "review" -> "review"
        "post_mistake" -> "lesson"
        "stuck", "decisive_question" -> "hint"
        else -> if (streamed) "praise" else "short"
    }

    /** Python's `statistics.median`: the middle value, or the mean of the two middle ones. */
    internal fun median(values: List<Double>): Double {
        val sorted = values.sorted()
        val n = sorted.size
        return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/** One sentence of a coach message, as the plan hands it to the voice. `pace.Line`. */
data class PaceLine(
    val key: Int,
    val kind: String,
    val index: Int,
    val text: String,
    /** The line's word, made while it was held: it plays at once. */
    val prepared: Boolean = false,
)

/** The voice's plan for one session: what is said, held, or left to the text. `pace.PacePlan`. */
class PacePlan(private val clock: () -> Double) {
    private data class Message(
        val key: Int,
        val kind: String,
        val trigger: String,
        /** (epoch, ply) of the player's move (a hint: its position). */
        val about: Pair<Int, Int>?,
        /** When its move was played (a hint, a question: when it was asked). */
        val askedAt: Double,
        val model: Boolean,
        /** The player's moves counted when its move was played: the moves since are counted,
         * never compared by time. */
        val seq: Int,
    )

    private val lock = Any()
    private var epoch = 0
    private var ply = 0
    private var count = 0
    private val moveAt = HashMap<Pair<Int, Int>, Pair<Double, Int>>()
    /** The player's moves taken back, by their number ([count]), never by their square: a retry
     * is played at the same ply, and its word was taken for the withdrawn move's and never said
     * (Tai's Pixel, 2026-10-03: "Just what the position wanted." on a retry). */
    private val withdrawn = HashSet<Int>()
    private var positionSince = clock()
    private var thinks = ArrayList<Double>()
    private var quick = false
    private var lastThink = 0.0
    private var lastMoveAt = -1e9
    private val messages = HashMap<Int, Message>()
    private var queue = ArrayList<PaceLine>()
    private val held = ArrayList<PaceLine>()
    private val released = HashSet<Int>()
    private val speaking = HashSet<Int>()
    private val dead = HashSet<Int>()
    private var fed = ArrayList<PaceLine>()

    /** A word on a move of this board was given to the voice. */
    private var boardHeard = false

    // ---- the board -------------------------------------------------------------------------

    /** The player played the move at [ply] after [thinkS] of thought (null: not read). Whether
     * the voice must stop now: a hint whose position is gone. `player_moved`. */
    fun playerMoved(ply: Int, thinkS: Double?): Boolean = synchronized(lock) {
        val now = clock()
        val think = thinkS ?: (now - positionSince)
        val recent = (thinks + think).takeLast(Pace.PACE_MOVES)
        val fastPace = recent.size >= 2 && Pace.median(recent) < Pace.FAST_PACE_S
        // A move after real thought is slow whatever the pace was.
        quick = think < Pace.QUICK_S || (fastPace && think < Pace.FAST_PACE_S)
        lastThink = think
        thinks.add(think)
        lastMoveAt = now
        count += 1
        moveAt[epoch to ply] = now to count
        this.ply = ply
        positionSince = now
        afterChange(now, setOf("hint"))
    }

    fun opponentMoved(ply: Int) = synchronized(lock) {
        this.ply = ply
        positionSince = clock()
    }

    /** A takeback to [toPly]: the player's moves after it are withdrawn, and a lesson on one of
     * them is never said again. Whether the voice must stop now. `taken_back`. */
    fun takenBack(toPly: Int): Boolean = synchronized(lock) {
        val now = clock()
        for ((at, played) in moveAt) if (at.first == epoch && at.second > toPly) withdrawn.add(played.second)
        ply = toPly
        positionSince = now
        afterChange(now, setOf("hint"), stopWithdrawn = true)
    }

    /** A new game or puzzle, or the board re-read: the fast line's word is said now, if still
     * young. Whether the voice must stop now. `new_position`. */
    fun newPosition(): Boolean = synchronized(lock) {
        val now = clock()
        epoch += 1
        ply = 0
        thinks = ArrayList()
        positionSince = now
        release(now)
        boardHeard = false
        afterChange(now, setOf("hint"))
    }

    private fun afterChange(now: Double, stopKinds: Set<String>, stopWithdrawn: Boolean = false): Boolean {
        queue = ArrayList(queue.filter { keep(it, now) })
        val playing = fed.firstOrNull() ?: return false
        val msg = messages[playing.key] ?: return false
        if (msg.kind in stopKinds) return msg.kind != "hint" || !hintFresh(msg)
        return stopWithdrawn && withdrawnWord(msg)
    }

    /** A word on a move the player took back (a hint and a question are on no move). */
    private fun withdrawnWord(msg: Message): Boolean =
        msg.about != null && msg.kind != "hint" && msg.kind != "answer" && msg.seq in withdrawn

    // ---- the coach's messages ---------------------------------------------------------------

    /** A sentence of a coach message is ready: [Pace.SAY], [Pace.HOLD] or [Pace.TEXT]. [ply] is
     * the message's position: for a word on a move, the position it was played from. `offer`. */
    fun offer(key: Int, trigger: String, ply: Int?, index: Int, text: String, streamed: Boolean): String =
        synchronized(lock) {
            val now = clock()
            val msg = messages[key] ?: message(key, trigger, ply, streamed, now)
            if (key in dead) return@synchronized Pace.TEXT
            if (msg.kind == "praise" || msg.kind == "short") return@synchronized offerWord(msg, index, text, now)
            if (!sentenceFresh(msg, index, now)) {
                if (index == 0) dead.add(key)
                return@synchronized Pace.TEXT
            }
            enqueue(PaceLine(key, msg.kind, index, text))
            Pace.SAY
        }

    private fun message(key: Int, trigger: String, ply: Int?, streamed: Boolean, now: Double): Message {
        val kind = Pace.kindOf(trigger, streamed)
        var about: Pair<Int, Int>? = null
        var askedAt = now
        var seq = count
        if ((kind == "praise" || kind == "short" || kind == "lesson") && ply != null) {
            about = epoch to ply + 1
            val at = moveAt[about]
            if (at != null) {
                askedAt = at.first
                seq = at.second
            }
        } else if (kind == "hint" && ply != null) {
            about = epoch to ply
        }
        val msg = Message(key, kind, trigger, about, askedAt, streamed, seq)
        messages[key] = msg
        if (messages.size > 400) messages.keys.sorted().take(200).forEach { messages.remove(it) }
        return msg
    }

    private fun offerWord(msg: Message, index: Int, text: String, now: Double): String {
        val key = msg.key
        if (key in released || held.any { it.key == key }) return Pace.TEXT
        if (key in speaking) {
            if (sentenceFresh(msg, index, now)) {
                enqueue(PaceLine(key, msg.kind, index, text))
                return Pace.SAY
            }
            return Pace.TEXT
        }
        val latest = movesSince(msg) == 0 && msg.about != null && msg.about.first == epoch
        val shortOk = msg.kind == "short" && lastThink >= Pace.SHORT_OK_S
        if (latest && (!quick || shortOk) && fresh(msg, now)) {
            speaking.add(key)
            enqueue(PaceLine(key, msg.kind, index, text))
            return Pace.SAY
        }
        val holding = held.firstOrNull()
        if (index == 0 && msg.model && (holding == null || holding.kind == "short")) {
            // A fast line, or the player already moved on: the model's word on the line's first
            // find is held for the end of the line.
            if (holding != null) dead.add(holding.key)
            held.clear()
            held.add(PaceLine(key, msg.kind, 0, text, prepared = true))
            return Pace.HOLD
        }
        if (index == 0 && !msg.model && (holding == null || holding.kind == "short") &&
            now - msg.askedAt <= Pace.PRAISE_FRESH_S && !boardHeard
        ) {
            // With no word of the model's, the latest template word, said when the next puzzle
            // comes up: every "Spot on." of a quickly solved puzzle went to the text (Tai's Pixel,
            // 2026-10-03). Never at a pause in a game, never when a word on this board was said.
            if (holding != null) dead.add(holding.key)
            held.clear()
            held.add(PaceLine(key, msg.kind, 0, text, prepared = true))
            return Pace.HOLD
        }
        dead.add(key)
        return Pace.TEXT
    }

    private fun enqueue(line: PaceLine) {
        if (line.kind == "praise" || line.kind == "short" || line.kind == "lesson") {
            boardHeard = true
            val holding = held.firstOrNull()
            if (holding != null && holding.kind == "short") {
                // A template word held for the next puzzle: this board has its word now.
                dead.add(holding.key)
                held.clear()
            }
        }
        queue.add(line)
        queue.sortWith(compareBy({ Pace.PRIORITY.getValue(it.kind) }, { it.key }, { it.index }))
    }

    // ---- freshness ---------------------------------------------------------------------------

    private fun movesSince(msg: Message): Int = count - msg.seq

    private fun hintFresh(msg: Message): Boolean = msg.about == (epoch to ply)

    private fun fresh(msg: Message, now: Double): Boolean {
        if (msg.key in dead) return false
        val age = now - msg.askedAt
        if (msg.key in released) return age <= Pace.LINE_WORD_FRESH_S + 6.0
        return when (msg.kind) {
            "answer" -> true
            "review" -> age <= Pace.REVIEW_FRESH_S
            "hint" -> hintFresh(msg)
            "lesson" -> !withdrawnWord(msg) && movesSince(msg) <= Pace.LESSON_MOVES && age <= Pace.LESSON_FRESH_S
            else -> when {
                msg.about == null || msg.about.first != epoch -> false
                withdrawnWord(msg) -> false
                msg.key in speaking -> movesSince(msg) == 0
                else -> movesSince(msg) == 0 && age <= Pace.PRAISE_FRESH_S
            }
        }
    }

    private fun sentenceFresh(msg: Message, index: Int, now: Double): Boolean {
        if (!fresh(msg, now)) return false
        val movedOn = movesSince(msg) > 0 || (msg.about != null && msg.about.first != epoch)
        return !(index > 0 && msg.kind != "answer" && msg.kind != "review" && movedOn)
    }

    private fun keep(line: PaceLine, now: Double): Boolean {
        val msg = messages[line.key] ?: return false
        return sentenceFresh(msg, line.index, now)
    }

    // ---- the voice ---------------------------------------------------------------------------

    /** The fast line's word is released once the line is done. `tick`. */
    fun tick() = synchronized(lock) {
        val now = clock()
        // A template word waits for the next puzzle ([newPosition]), never a pause.
        if (held.isNotEmpty() && held[0].kind != "short" && now - lastMoveAt >= Pace.LINE_END_S) release(now)
    }

    private fun release(now: Double) {
        if (held.isEmpty()) return
        val line = held[0]
        held.clear()
        val msg = messages[line.key]
        if (msg != null && now - msg.askedAt <= Pace.LINE_WORD_FRESH_S) {
            released.add(line.key)
            enqueue(line)
        } else {
            dead.add(line.key)
        }
    }

    /** The fast line's word being held, for the voice to make ahead. `held_text`. */
    fun heldText(): String? = synchronized(lock) { held.firstOrNull()?.text }

    /** The next sentence for the voice, still fresh now; null when nothing waits. `next_line`. */
    fun nextLine(): PaceLine? = synchronized(lock) {
        val now = clock()
        while (queue.isNotEmpty()) {
            val line = queue.removeAt(0)
            if (keep(line, now)) {
                fed.add(line)
                return@synchronized line
            }
        }
        null
    }

    /** What the voice holds now: [busy], and [waiting] lines behind the one playing (null: the
     * voice cannot say). `voice_state`. */
    fun voiceState(busy: Boolean, waiting: Int?) = synchronized(lock) {
        if (!busy) {
            fed = ArrayList()
        } else if (waiting != null) {
            val keep = waiting + 1
            if (keep < fed.size) fed = ArrayList(fed.takeLast(keep))
        }
    }

    /** Whether a line handed to the voice and waiting behind the one playing is no longer the
     * news. `waiting_stale`. */
    fun waitingStale(): Boolean = synchronized(lock) {
        val now = clock()
        if (fed.size < 2) return@synchronized false
        if (fed.drop(1).any { !keep(it, now) }) {
            fed = ArrayList(fed.take(1))
            true
        } else {
            false
        }
    }

    /** The voice was stopped. `stopped`. */
    fun stopped() = synchronized(lock) { fed = ArrayList() }

    /** Muted, or the session over: nothing waits, nothing is held. `clear`. */
    fun clear() = synchronized(lock) {
        queue = ArrayList()
        held.clear()
        fed = ArrayList()
    }
}
