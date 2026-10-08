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
        StatsApi.rememberDrills(work, motifs)
        assertEquals(mapOf("old01" to "pin", "000Pw" to "fork", "00Y5Q" to "fork", "12Exn" to "opening"), motifs)
    }
}
