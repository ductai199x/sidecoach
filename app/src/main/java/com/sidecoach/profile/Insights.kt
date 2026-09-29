package com.sidecoach.profile

import com.sidecoach.events.ProfileCell
import com.sidecoach.events.insightWords
import com.sidecoach.voice.Py
import java.math.BigDecimal
import java.math.RoundingMode

/**
 * What the cells add up to, in words (DESIGN section 5.6): the biggest cost, whether it is a
 * gap or a habit, the motif missed most often, the phase that costs most.
 * `sidecoach/profile/insights.py`.
 *
 * Everything is computed from counts and the expected score they cost, and an insight needs a
 * few mistakes behind it, so a single event never makes a weakness. The words are for the
 * coach's continuity and the panel; the numbers stay in the cells.
 */
object Insights {

    /** A cause needs this many mistakes before it is named; a motif this many occasions. */
    const val MIN_ERRORS = 3
    const val MIN_CHANCES = 4

    /** A cause whose cost sits this much in one kind of position is a gap; spread out, a habit. */
    const val GAP_SHARE = 0.6

    /** A tagger name in words: hangingPiece -> hanging piece, mateIn2 -> mate in 2. */
    fun motifWords(name: String?): String = insightWords(name)

    fun asCells(rows: List<CellRow>): List<ProfileCell> = rows.map {
        ProfileCell(
            cause = it.cause,
            motif = it.motif,
            phase = it.phase,
            errors = it.errors,
            chances = it.chances,
            expectedLoss = round(it.expectedLoss, 3),
        )
    }

    /** The player model in a few sentences, or nothing while the counts are small. */
    fun insights(rows: List<CellRow>, games: Int): List<String> {
        val out = ArrayList<String>()
        val byCause = LinkedHashMap<String, MutableList<CellRow>>()
        for (r in rows) {
            if (r.cause != null && r.errors > 0) byCause.getOrPut(r.cause) { ArrayList() }.add(r)
        }
        if (byCause.isNotEmpty()) {
            val (cause, cells) = byCause.entries.maxByOrNull { e -> Py.sum(e.value.map { it.expectedLoss }) }!!.toPair()
            val errors = cells.sumOf { it.errors }
            val loss = Py.sum(cells.map { it.expectedLoss })
            if (errors >= MIN_ERRORS) {
                val over = if (games > 1) " over $games games" else ""
                out.add("Your biggest cost$over is $cause: $errors mistakes, ${fixed(loss, 2)} expected score.")
                val top = cells.maxByOrNull { it.expectedLoss }!!
                if (loss > 0 && top.expectedLoss / loss >= GAP_SHARE && (top.motif != null || top.phase != null)) {
                    val what = if (top.motif != null) motifWords(top.motif) else "positions"
                    val where = if (top.phase != null) " in the ${top.phase}" else ""
                    out.add("That looks like a gap: it shows up as $what$where.")
                } else {
                    val kinds = cells.map { it.motif to it.phase }.toSet().size
                    out.add("That looks like a habit: it shows up across $kinds kinds of position.")
                }
            }
        }
        // The motif missed most often, as a rate over the occasions it was on the board.
        motifRates(rows).maxByOrNull { it.rate }?.let { (motif, rate, missed, seen) ->
            out.add(
                "You miss a ${motifWords(motif)} about ${percent(rate)} of the time one is on the board" +
                    " ($missed of $seen).",
            )
        }
        // The phase that costs most per move.
        val phaseMoves = LinkedHashMap<String, Int>()
        val phaseLoss = HashMap<String, Double>()
        for (r in rows) {
            val phase = r.phase ?: continue
            if (r.cause == null && r.motif == null) phaseMoves[phase] = (phaseMoves[phase] ?: 0) + r.chances
            phaseLoss[phase] = (phaseLoss[phase] ?: 0.0) + r.expectedLoss
        }
        val busy = phaseMoves.keys.filter { (phaseMoves[it] ?: 0) >= 10 }
        if (busy.size >= 2) {
            val worst = busy.maxByOrNull { (phaseLoss[it] ?: 0.0) / phaseMoves[it]!! }!!
            val perMove = (phaseLoss[worst] ?: 0.0) / phaseMoves[worst]!!
            if (perMove > 0) out.add("The $worst costs you most per move (${fixed(perMove, 3)} expected score).")
        }
        return out
    }

    internal data class Rated(val motif: String, val rate: Double, val missed: Int, val seen: Int)

    /**
     * Each motif missed at least twice over enough of its chances, with the rate it is missed,
     * smoothed as (misses + 1) / (chances + 2), in the order its chances came. A cell with a
     * cause counts misses, one without counts chances. `insights.motif_rates`.
     */
    internal fun motifRates(rows: List<CellRow>): List<Rated> {
        val chances = LinkedHashMap<String, Int>()
        val misses = HashMap<String, Int>()
        for (r in rows) {
            val motif = r.motif ?: continue
            if (r.cause == null) chances[motif] = (chances[motif] ?: 0) + r.chances
            else misses[motif] = (misses[motif] ?: 0) + r.errors
        }
        return chances.keys
            .filter { (chances[it] ?: 0) >= MIN_CHANCES && (misses[it] ?: 0) >= 2 }
            .map { m ->
                val missed = misses[m] ?: 0
                val seen = chances[m] ?: 0
                Rated(m, (missed + 1).toDouble() / (seen + 2), missed, seen)
            }
    }

    /** Python's `round(x, n)`: the exact value, half to even. */
    fun round(value: Double, places: Int): Double =
        BigDecimal(value).setScale(places, RoundingMode.HALF_EVEN).toDouble()

    /**
     * Python's `f"{x:.nf}"`. A negative that rounds to zero keeps its sign ("-0.000"), as
     * Python writes it; a BigDecimal has no negative zero.
     */
    fun fixed(value: Double, places: Int): String {
        val text = BigDecimal(value).setScale(places, RoundingMode.HALF_EVEN).toPlainString()
        return if ((value < 0 || 1.0 / value < 0) && !text.startsWith("-")) "-$text" else text
    }

    /** Python's `f"{x:.0%}"`: times a hundred as a float, then no decimals. */
    fun percent(value: Double): String =
        BigDecimal(value * 100.0).setScale(0, RoundingMode.HALF_EVEN).toPlainString() + "%"
}
