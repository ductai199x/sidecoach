package com.sidecoach.profile

import com.sidecoach.coach.fixture
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * How a game ended, as the desktop's `termination_kind` read it for the same header and result
 * (stats.json's `units.termination_kind_result`). A flag fall against a side that cannot mate --
 * chess.com's "Game drawn by timeout vs insufficient material", Lichess's "Time forfeit" drawn --
 * was filed as a time ending in Stats (the audit of 2026-10-07;
 * `test_a_timeout_against_insufficient_material_is_counted_as_the_draw_it_is`). The device test
 * `StatsParityTest` replays the same table with the rest of Stats.
 */
class TerminationKindTest {
    @Test
    fun aTimeoutAgainstInsufficientMaterialIsADraw() {
        val cases = fixture("profile/stats.json", setOf("units"))
            .getJSONObject("units").getJSONArray("termination_kind_result")
        val failures = mutableListOf<String>()
        for (i in 0 until cases.length()) {
            val c = cases.getJSONArray(i)
            val text = if (c.isNull(0)) null else c.getString(0)
            val result = if (c.isNull(1)) null else c.getString(1)
            val got = Stats.terminationKind(text, result)
            if (got != c.getString(2)) failures.add("'$text' $result: kotlin $got, python ${c.getString(2)}")
        }
        assertEquals(emptyList<String>(), failures)
        assertEquals("insufficient material", Stats.terminationKind("Game drawn by timeout vs insufficient material"))
    }
}
