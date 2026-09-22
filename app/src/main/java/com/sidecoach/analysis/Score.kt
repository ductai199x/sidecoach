package com.sidecoach.analysis

import kotlin.math.exp

/**
 * Expected score: `sidecoach/analysis/score.py`, the part the engine needs.
 *
 * Everything is in expected score, never centipawns: 0.5 is a dead draw, 1.0 is a win for
 * the side to move. A 50-centipawn slip at +8 costs nothing and the same slip at level costs
 * the game, so a threshold in centipawns says nothing about how much a move mattered.
 */
object Score {

    const val DRAW = 0.5

    /**
     * Lichess's curve from centipawns to winning chances, fitted to games between people.
     * Stockfish's own win-draw-loss numbers answer a different question -- how often
     * Stockfish beats Stockfish from here, which is 100% from a knight up -- and say nothing
     * about the rest of a club game. A coach needs the human curve.
     */
    const val HUMAN_K = 0.00368208
    const val CP_CAP = 1000

    /** Expected score for the side to move, on the curve fitted to human games. */
    fun humanExpectedScore(scoreCp: Int?, mate: Int?): Double {
        if (mate != null) return if (mate > 0) 1.0 else 0.0
        if (scoreCp == null) return DRAW
        val capped = scoreCp.coerceIn(-CP_CAP, CP_CAP)
        return 1.0 / (1.0 + exp(-HUMAN_K * capped))
    }

    /** Expected score from the engine's own win-draw-loss numbers: (W + D/2) / 1000. */
    fun engineExpectedScore(wdl: Triple<Int, Int, Int>?): Double {
        if (wdl == null) return DRAW
        val total = wdl.first + wdl.second + wdl.third
        if (total == 0) return DRAW
        return (wdl.first + wdl.second / 2.0) / total
    }

    /** Expected score under the named model: "human" (the default) or "engine". */
    fun expectedScore(model: String, scoreCp: Int?, mate: Int?, wdl: Triple<Int, Int, Int>?): Double =
        if (model == "engine") engineExpectedScore(wdl) else humanExpectedScore(scoreCp, mate)
}
