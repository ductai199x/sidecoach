package com.sidecoach.profile

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The player's puzzle rating as the profile keeps it (Puzzles.perfJson / parsePerf; the desktop's
 * `perf_json` / `parse_perf`, `test_the_puzzle_rating_starts_fresh_and_moves_as_lichess_moves_it`):
 * Lichess's own fields, a new player's 1500 / 500 / 0.09 when there is none, and anything saved
 * there that is not a rating read as a new player's.
 */
class PuzzlePerfTest {
    @Test
    fun aSavedRatingReadsBackAndAnythingElseIsANewPlayer() {
        val perf = Glicko.Perf(Glicko.Rating(1612.375, 87.5, 0.0899), nb = 12, latest = 1_790_000_000_123L)
        assertEquals(perf, Puzzles.parsePerf(Puzzles.perfJson(perf)))
        // Saved as the desktop saves it (`json.dumps`), byte for byte: one profile, either platform.
        assertEquals("""{"r": 1612.375, "d": 87.5, "v": 0.0899, "nb": 12, "la": 1790000000123}""", Puzzles.perfJson(perf))
        assertEquals("""{"r": 1500.0, "d": 500.0, "v": 0.09, "nb": 0, "la": null}""", Puzzles.perfJson(Glicko.Perf()))
        assertEquals(Glicko.Perf(), Puzzles.parsePerf(null))
        assertEquals(Glicko.Perf(), Puzzles.parsePerf(""))
        assertEquals(Glicko.Perf(), Puzzles.parsePerf("not json"))
        assertEquals(Glicko.Perf(), Puzzles.parsePerf("""{"r": 1500.0}"""))
        assertEquals("not a sane rating", Glicko.Perf(), Puzzles.parsePerf("""{"r": 9000.0, "d": 60.0, "v": 0.06, "nb": 3, "la": 5}"""))
        // As the desktop writes it (json.dumps), never yet rated.
        assertEquals(
            Glicko.Perf(Glicko.Rating(1500.0, 500.0, 0.09), nb = 0, latest = null),
            Puzzles.parsePerf("""{"r": 1500.0, "d": 500.0, "v": 0.09, "nb": 0, "la": null}"""),
        )
        assertEquals(1500, Glicko.Perf().glicko.intRating)
    }
}
