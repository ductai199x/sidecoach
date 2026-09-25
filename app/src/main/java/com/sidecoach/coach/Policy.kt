package com.sidecoach.coach

import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.Severity
import com.sidecoach.events.Trigger

/**
 * When the coach speaks without being asked: `sidecoach/coach/policy.py`.
 *
 * DESIGN.md section 5.5, in priority order: an explicit ask is always answered; a lesson
 * after a mistake; brief feedback after every move the user plays; an orienting question at a
 * decisive moment; a hint once the user has been stuck for a while. Higher numbers yield to
 * lower ones when they collide. While the user is thinking, at most two unprompted messages per
 * position -- the idea, then, if they are still stuck, the focus -- each after its own wait, so
 * the coach never nags and never gives the move unasked. A quiet toggle silences everything but
 * explicit asks.
 *
 * Time control scales all of it: rapid and slower get every trigger, with the stuck threshold
 * shrinking as the clock runs down; blitz gets the brief post-move feedback only; bullet gets
 * nothing during the game. Untimed games and analysis boards get everything.
 */

enum class TimeControl(val slug: String) {
    UNTIMED("untimed"),
    RAPID("rapid"),
    BLITZ("blitz"),
    BULLET("bullet"),
}

/**
 * Chess.com's bands: ten minutes and up is rapid, three to ten blitz, under three bullet. The
 * estimated game length counts forty moves of increment, as the sites do.
 */
fun timeControlOf(initialMs: Int?, incrementMs: Int = 0): TimeControl {
    if (initialMs == null) return TimeControl.UNTIMED
    val estimated = initialMs + 40 * incrementMs
    if (estimated >= 10 * 60_000) return TimeControl.RAPID
    if (estimated >= 3 * 60_000) return TimeControl.BLITZ
    return TimeControl.BULLET
}

/**
 * In a run of decisive positions (a won endgame: every move is "the" move), the decisive
 * question is asked at most once in this many plies.
 */
const val DECISIVE_EVERY = 8

/**
 * Unprompted messages in one position while the user thinks: the idea, then the focus (Tai,
 * 2026-09-23). The move itself only when asked.
 */
const val UNPROMPTED_PER_POSITION = 2

data class PolicySettings(
    val stuckSeconds: Double = 45.0,
    val quiet: Boolean = false,
    /**
     * The stuck threshold never drops below this, even on a low clock. Twenty seconds on a
     * move is thinking, not being stuck; at eight, the first live game got a nudge on nearly
     * every endgame move, each one spoken over the next.
     */
    val stuckFloorSeconds: Double = 20.0,
    /** With a clock, a hint comes after this share of the remaining time, at most. */
    val stuckShareOfRemaining: Double = 0.08,
    /**
     * Under this much clock the user is in a scramble: a two-sentence question takes longer
     * to hear than they have to think, so no stuck hints at all.
     */
    val stuckScrambleSeconds: Double = 30.0,
)

/** Decides which unprompted triggers fire, and remembers what it already said. */
class InterjectionPolicy(
    var settings: PolicySettings = PolicySettings(),
    var timeControl: TimeControl = TimeControl.UNTIMED,
) {

    // The position the user is thinking in: its ply, when it appeared, how many times the coach
    // has spoken unprompted while they think in it, and when it last did.
    private var plyValue: Int? = null
    private var since: Double? = null
    private val spoken = HashMap<Int, Int>()
    private var saidTs: Double? = null
    private var userToMoveValue = false
    private var remainingMs: Int? = null

    /** The board is under a dialog: the game may be over, and no nudge makes sense. */
    var covered: Boolean = false

    /**
     * Why the last unprompted trigger did not fire, in a few words, for the debug log -- the
     * answer to "why is the coach quiet?". Null once a trigger fires.
     */
    var lastReason: String? = null

    /** The last ply the decisive question was asked at, while the positions stay decisive. */
    private var lastDecisivePly: Int? = null

    // What the board is doing

    /** The board reached `ply`; the user is (or is not) the one to move. */
    fun newPosition(ply: Int, ts: Double, userToMove: Boolean) {
        if (ply != plyValue) saidTs = null
        plyValue = ply
        since = ts
        userToMoveValue = userToMove
    }

    /**
     * A new game, or a position adopted: move numbers start again, so the decisive stretch
     * pointed at in the last one says nothing about this one. A puzzle's first position was
     * skipped as part of the previous puzzle's stretch, which ended at a higher ply (Tai's
     * desktop, 2026-09-24: the mate in three was never asked, only hinted 45 s later).
     */
    fun restart() {
        lastDecisivePly = null
    }

    /** The user's clock, when the screen shows one. */
    fun setRemaining(remaining: Int?) {
        remainingMs = remaining
    }

    val ply: Int? get() = plyValue

    val userToMove: Boolean get() = userToMoveValue

    /** When the position the user is thinking in appeared (the loop's clock). */
    val positionSince: Double? get() = since

    // The triggers

    /** After the user's move is measured: a lesson if it was a mistake, else a word. */
    fun onUserMoveAssessed(assessment: MoveAssessment): Trigger? {
        if (settings.quiet) return silent("quiet mode")
        if (timeControl == TimeControl.BULLET) return silent("bullet: no unprompted coaching")
        val serious = assessment.severity == Severity.MISTAKE ||
            assessment.severity == Severity.BLUNDER ||
            (assessment.label != null && assessment.label.name.lowercase() == "miss")
        lastReason = null
        if (serious && timeControl != TimeControl.BLITZ) return Trigger.POST_MISTAKE
        return Trigger.POST_MOVE
    }

    /**
     * The engine has looked at the position the user is thinking in. The decisive question is
     * asked when a position turns decisive, not on every move of a won endgame: once a stretch
     * of decisive positions has been pointed at, the next question waits for a quiet position
     * in between, or [DECISIVE_EVERY] plies.
     */
    fun onPositionAnalysed(ply: Int, decisive: Boolean): Trigger? {
        if (!maySpeakNow(ply)) return null
        if ((spoken[ply] ?: 0) != 0) return silent("already spoke in this position")
        if (!decisive) {
            lastDecisivePly = null
            return silent("position not decisive")
        }
        val last = lastDecisivePly
        if (last != null && ply - last < DECISIVE_EVERY) {
            return silent("decisive stretch already pointed at (ply $last)")
        }
        lastDecisivePly = ply
        lastReason = null
        return Trigger.DECISIVE_QUESTION
    }

    /**
     * Time passing while the user thinks; a hint once they have been stuck long enough, counted
     * from the position or from the last thing said in it.
     */
    fun tick(ts: Double): Trigger? {
        val ply = plyValue
        val began = since
        if (ply == null || began == null) return silent("no position yet")
        if (!maySpeakNow(ply)) return null
        if (inScramble) return silent("time scramble: no stuck hints")
        val threshold = stuckThreshold()
        val said = saidTs
        val from = if (said != null) maxOf(began, said) else began
        if (ts - from < threshold) {
            return silent("waiting for a stuck hint at ${"%.0f".format(threshold)}s")
        }
        lastReason = null
        return Trigger.STUCK
    }

    /** Whether the user's clock is down to the last seconds. */
    val inScramble: Boolean
        get() {
            val remaining = remainingMs ?: return false
            return remaining < settings.stuckScrambleSeconds * 1000
        }

    private fun silent(reason: String): Trigger? {
        lastReason = reason
        return null
    }

    /**
     * One unprompted message went out for the position at `ply`, at `ts`: the next one waits a
     * full stuck spell from it, and none comes after the second.
     */
    fun said(ply: Int, ts: Double? = null) {
        spoken[ply] = (spoken[ply] ?: 0) + 1
        if (ply == plyValue && ts != null) saidTs = ts
    }

    /** Seconds on the move before a hint, shorter as the clock runs down. */
    fun stuckThreshold(): Double {
        var threshold = settings.stuckSeconds
        val remaining = remainingMs
        if (remaining != null) {
            val share = remaining / 1000.0 * settings.stuckShareOfRemaining
            threshold = minOf(threshold, maxOf(settings.stuckFloorSeconds, share))
        }
        return threshold
    }

    private fun maySpeakNow(ply: Int): Boolean {
        val reason = when {
            settings.quiet -> "quiet mode"
            covered -> "the board is covered"
            !userToMoveValue -> "opponent to move"
            ply != plyValue -> "position moved on"
            timeControl == TimeControl.BLITZ || timeControl == TimeControl.BULLET ->
                "${timeControl.slug}: no unprompted hints"
            (spoken[ply] ?: 0) >= UNPROMPTED_PER_POSITION ->
                "gave the idea and the focus here; the move only when asked"
            else -> null
        }
        if (reason != null) {
            lastReason = reason
            return false
        }
        return true
    }
}

/**
 * Lower is more important; when two triggers want the same moment, the coach speaks to the
 * lower one and folds the other in or drops it.
 */
val PRIORITY: Map<Trigger, Int> = mapOf(
    Trigger.USER_ASK to 1,
    Trigger.POST_MISTAKE to 2,
    Trigger.POST_MOVE to 3,
    Trigger.DECISIVE_QUESTION to 4,
    Trigger.STUCK to 5,
    Trigger.REVIEW to 6,
)

fun inPriority(triggers: List<Trigger>): List<Trigger> =
    triggers.distinct().sortedBy { PRIORITY.getValue(it) }
