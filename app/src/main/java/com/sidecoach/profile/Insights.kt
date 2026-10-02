package com.sidecoach.profile

import com.sidecoach.events.ProfileCell
import com.sidecoach.events.insightWords
import java.math.BigDecimal
import java.math.RoundingMode

/**
 * What the player's record adds up to, in words (DESIGN section 5.6): the biggest cost (the area
 * Stats' "Work on this" puts first), the motif missed most often, the phase that costs most.
 * `sidecoach/profile/insights.py`.
 *
 * Everything is computed from counts and the expected score they cost, and an insight needs a
 * few mistakes behind it, so a single event never makes a weakness. The words are for the
 * coach's continuity and the panel; the numbers stay in the cells.
 */
object Insights {

    /**
     * A motif is named after this many mistakes of it and this many occasions, from this many
     * games: three mistakes in one game were "a habit" (the review of 2026-10-01).
     */
    const val MIN_ERRORS = 3
    const val MIN_CHANCES = 4
    const val MIN_GAMES = 3

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

    /**
     * The player model in a few sentences, or nothing while the counts are small. [cost] is the
     * area that costs the player most ([Stats.biggestCost]): the one rule Stats' "Work on this"
     * uses, so the coach and the page name the same thing. The cells' causes (a mistake's why)
     * were a second taxonomy beside Stats' areas and no longer name the biggest cost; the
     * sentences that said a gap or a habit were about that cause and go with it (Tai,
     * 2026-10-02: one taxonomy). `insights`.
     */
    fun insights(rows: List<CellRow>, games: Int, cost: org.json.JSONObject? = null): List<String> {
        val out = ArrayList<String>()
        if (cost != null) {
            val area = cost.getString("area")
            val name = Stats.AREA_NAMES[area] ?: area
            out.add("Your biggest cost over ${cost.getInt("games")} games is $name: ${cost.getInt("moves")} moves.")
        }
        // The motif missed most often, as a rate over the occasions it was on the board.
        motifRates(rows).maxByOrNull { it.rate }?.let { (motif, _, missed, seen) ->
            out.add("You missed a ${motifWords(motif)} $missed of the $seen times you had one.")
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
            if ((phaseLoss[worst] ?: 0.0) > 0) out.add("The $worst costs you most per move.")
        }
        return out
    }

    internal data class Rated(val motif: String, val rate: Double, val missed: Int, val seen: Int)

    /**
     * Each motif missed often enough, over enough of its chances and games, with the rate it is
     * missed, smoothed as (misses + 1) / (chances + 2) to rank them, in the order its chances
     * came. The cells without a cause count a motif's chances and, in their errors, the chances
     * missed; a mistake's own motif may be a tactic it allowed, never a chance the player had.
     * `insights.motif_rates`.
     */
    internal fun motifRates(rows: List<CellRow>): List<Rated> {
        val chances = LinkedHashMap<String, Int>()
        val misses = HashMap<String, Int>()
        val games = HashMap<String, MutableSet<String>>()
        for (r in rows) {
            val motif = r.motif ?: continue
            if (r.cause != null) continue
            chances[motif] = (chances[motif] ?: 0) + r.chances
            misses[motif] = (misses[motif] ?: 0) + r.errors
            games.getOrPut(motif) { HashSet() }.addAll(r.games)
        }
        return chances.keys
            .filter {
                (chances[it] ?: 0) >= MIN_CHANCES && (misses[it] ?: 0) >= MIN_ERRORS &&
                    (games[it]?.size ?: 0) >= MIN_GAMES
            }
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
