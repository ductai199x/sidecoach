package com.sidecoach.ui

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Stats' "Puzzles on it" (`server._remember_drills`, tests/test_puzzle_mode.py
 * `test_a_puzzle_stats_offers_for_an_area_opens_with_its_motif`): each puzzle offered for an
 * area is remembered with the area's motif, so the room it opens in records the attempt beside
 * it and the drill counts. An area with no motif remembers nothing.
 */
class StatsDrillsTest {
    @Test
    fun aPuzzleOfferedForAnAreaIsRememberedWithItsMotif() {
        val puzzle = { id: String -> JSONObject().put("id", id).put("rating", 1500) }
        val work = JSONArray()
            .put(JSONObject().put("area", "quiet").put("motif", JSONObject.NULL).put("puzzles", JSONArray()))
            .put(JSONObject().put("area", "tactic.fork").put("motif", "fork").put("puzzles", JSONArray().put(puzzle("000Pw")).put(puzzle("00Y5Q"))))
            .put(JSONObject().put("area", "opening").put("motif", "opening").put("puzzles", JSONArray().put(puzzle("12Exn"))))
        val motifs = HashMap<String, String>()
        motifs["old01"] = "pin" // a recommendation's, kept
        val lists = HashMap<String, List<String>>()
        StatsApi.rememberDrills(work, motifs, lists)
        assertEquals(mapOf("old01" to "pin", "000Pw" to "fork", "00Y5Q" to "fork", "12Exn" to "opening"), motifs)
        // And each with its area's list, for Next once the puzzle has ended
        // (`test_a_puzzle_has_a_next_once_it_ends_however_it_was_opened`).
        val fork = listOf("000Pw", "00Y5Q")
        assertEquals(mapOf("000Pw" to fork, "00Y5Q" to fork, "12Exn" to listOf("12Exn")), lists)
    }

    /**
     * Where Next goes once a puzzle has ended (`puzzle_room.following`,
     * `test_next_goes_down_its_list_then_the_recommendation_then_a_fresh_puzzle`): the ones after
     * it in the list it was offered in, not tried yet; else the session's recommended ones not
     * tried yet; else a fresh one.
     */
    @Test
    fun nextGoesDownItsListThenTheRecommendationThenAFreshPuzzle() {
        val fresh = { listOf("new") }
        val bTried = { id: String -> id == "b" }
        assertEquals("b was tried: passed", listOf("c"), PuzzleRooms.following("a", listOf("a", "b", "c"), emptyList(), bTried, fresh))
        assertEquals(listOf("p1", "p2"), PuzzleRooms.following("c", listOf("a", "b", "c"), listOf("p1", "p2"), bTried, fresh))
        assertEquals(listOf("p2", "p3"), PuzzleRooms.following("p1", null, listOf("p1", "p2", "p3"), bTried, fresh))
        assertEquals(listOf("p2"), PuzzleRooms.following("p3", null, listOf("p1", "p2", "p3"), { it == "p1" }, fresh))
        assertEquals(listOf("new"), PuzzleRooms.following("x", null, emptyList(), bTried, fresh))
        assertEquals("nothing to offer", emptyList<String>(), PuzzleRooms.following("x", listOf("x"), listOf("x"), bTried) { emptyList() })
    }
}
