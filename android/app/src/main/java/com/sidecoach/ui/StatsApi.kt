package com.sidecoach.ui

import com.sidecoach.profile.Library
import com.sidecoach.profile.Puzzles
import com.sidecoach.profile.Stats
import com.sidecoach.profile.Store
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Stats, the player's games added up (`server.stats_now`, `server.stats_positions` and their
 * helpers `_filter` and `_drills`; profile/Stats.kt, docs/stats.md): every section's
 * numbers for the games a filter takes, with puzzles for each area to work on, and the positions
 * behind a figure.
 *
 * The phone reviews a game when the player asks (Tai, 2026-09-30: no background review, as
 * chess.com has none), so the numbers come without a background reviewer's progress: `backlog`
 * is null, as on a desktop panel started without one.
 */
class StatsApi(private val files: File, private val userNames: () -> List<String> = { emptyList() }) {

    private fun <T> withStore(use: (Store) -> T): T {
        val store = Store(File(files, "profile.sqlite"))
        try {
            return use(store)
        } finally {
            store.close()
        }
    }

    /**
     * `_filter`: the page's query as the filter -- `time_class` (empty for all), `days` (0 for
     * all time), `opponents` (all, people, bots).
     */
    private fun filter(query: Map<String, String>): Stats.Filter = Stats.Filter(
        timeClass = query["time_class"]?.takeIf { it.isNotEmpty() },
        days = (query["days"]?.toIntOrNull() ?: 0).takeIf { it != 0 },
        opponents = query["opponents"] ?: "all",
    )

    /** `GET /api/stats`: the numbers for the query's filter, each area to work on with its puzzles. `stats_now`. */
    fun stats(query: Map<String, String>, now: Double = System.currentTimeMillis() / 1000.0): JSONObject {
        val numbers = withStore { store ->
            // A review whose Stats step a kill cut short counts too (no background review here
            // to catch it up: the review of 2026-10-01).
            Library.refresh(store, userNames())
            ReviewRoom.keepMissingStats(store)
            Stats.stats(store, filter(query), now).also {
                drills(it.getJSONArray("work"), files, store)
                rememberDrills(it.getJSONArray("work"), PanelEngine.puzzleMotifs)
            }
        }
        return JSONObject().put("ok", true).put("stats", numbers).put("backlog", JSONObject.NULL)
    }

    /**
     * `GET /api/stats/positions`: the player's moves behind a figure, the latest first -- `what`
     * the figure's selector, `arg` its kind. `stats_positions`.
     */
    fun positions(query: Map<String, String>, now: Double = System.currentTimeMillis() / 1000.0): JSONObject {
        val found = withStore {
            ReviewRoom.keepMissingStats(it)
            Stats.positions(it, filter(query), query["what"] ?: "", query["arg"] ?: "", now)
        }
        return JSONObject().put("ok", true).put("positions", found)
    }

    companion object {
        /**
         * Each area Stats says to work on gets puzzles of its motif at the player's puzzle level,
         * in place (none where the area has no motif, or there are no puzzles). `_drills`.
         */
        fun drills(work: JSONArray, files: File, store: Store) {
            val areas = (0 until work.length()).map { work.getJSONObject(it) }
            fun motif(w: JSONObject): String? = if (w.isNull("motif")) null else w.getString("motif").takeIf { it.isNotEmpty() }
            val puzzles = if (areas.any { motif(it) != null }) Puzzles.openStore(files) else null
            val level = Puzzles.puzzleLevel(store)
            try {
                for (w in areas) {
                    val name = motif(w)
                    val picked = if (puzzles != null && name != null) puzzles.pick(name, level) else emptyList()
                    w.put("puzzles", JSONArray(picked.map { JSONObject().put("id", it.id).put("rating", it.rating) }))
                }
            } finally {
                puzzles?.close()
            }
        }

        /**
         * Each puzzle Stats offers for an area ("Puzzles on it") remembered with the area's
         * motif, as the session's recommendations are: the room it opens in records the attempt
         * beside that motif, so the drill counts for it. `_remember_drills`.
         */
        fun rememberDrills(work: JSONArray, motifs: MutableMap<String, String>) {
            for (i in 0 until work.length()) {
                val w = work.getJSONObject(i)
                val motif = if (w.isNull("motif")) "" else w.optString("motif")
                if (motif.isEmpty()) continue
                val puzzles = w.optJSONArray("puzzles") ?: continue
                for (j in 0 until puzzles.length()) motifs[puzzles.getJSONObject(j).getString("id")] = motif
            }
        }
    }
}
