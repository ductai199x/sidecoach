package com.sidecoach.state

import com.sidecoach.chess.Move
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Moves that slide (desktop 8fadeda, `tests/test_slides.py`; Tai's chess.com puzzles,
 * 2026-09-24/25: "I played Rxc1+ but it misunderstood as Rc7"): a piece that stops partway along
 * its line and goes on the same way is one move in flight ([inFlight]). The two puzzles played
 * sliding at the phone's frame rate are pipeline fixtures (`slide-rook-green`,
 * `slide-queen-green` in [PipelineTest]).
 */
class SlidesTest {

    private fun m(vararg uci: String): List<Move> = uci.map { Move.fromUci(it) }

    @Test
    fun aPieceThatStopsPartwayAndGoesOnIsInFlight() {
        assertTrue(inFlight(m("c8c7", "h1f1", "c7c1")))  // the rook on its way to c1
        assertTrue(inFlight(m("d8d7", "a5c3", "c3e1")))  // the reply's queen, a5 to e1
        assertFalse(inFlight(m("c8c7", "h1f1", "c7c8")))  // back where it came from
        assertFalse(inFlight(m("g1f3", "e7e5", "f3g1")))
        assertFalse(inFlight(m("c8c7", "h1f1", "c7a7")))  // on, but another way
        assertFalse(inFlight(m("e2e4", "e7e5", "g1f3")))
    }
}
