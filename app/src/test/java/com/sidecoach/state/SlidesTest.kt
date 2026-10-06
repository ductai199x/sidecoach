package com.sidecoach.state

import com.sidecoach.chess.Move
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * Moves that slide (desktop 8fadeda, `tests/test_slides.py`; Tai's chess.com puzzles,
 * 2026-09-24/25: "I played Rxc1+ but it misunderstood as Rc7"): a piece that stops partway along
 * its line and goes on the same way is one move in flight ([inFlight]), which names the stop.
 * The two puzzles played sliding at the phone's frame rate are pipeline fixtures
 * (`slide-rook-green`, `slide-queen-green` in [PipelineTest]).
 */
class SlidesTest {

    private fun m(vararg uci: String): List<Move> = uci.map { Move.fromUci(it) }

    @Test
    fun aPieceThatStopsPartwayAndGoesOnIsInFlight() {
        assertEquals(Move.fromUci("c8c7"), inFlight(m("c8c7", "h1f1", "c7c1")))  // the rook on its way to c1
        assertEquals(Move.fromUci("a5c3"), inFlight(m("d8d7", "a5c3", "c3e1")))  // the reply's queen, a5 to e1
        assertNull(inFlight(m("c8c7", "h1f1", "c7c8")))  // back where it came from
        assertNull(inFlight(m("g1f3", "e7e5", "f3g1")))
        assertNull(inFlight(m("c8c7", "h1f1", "c7a7")))  // on, but another way
        assertNull(inFlight(m("e2e4", "e7e5", "g1f3")))
    }
}
