package com.sidecoach.ui

import com.sidecoach.profile.Puzzles
import com.sidecoach.profile.Stats
import com.sidecoach.profile.Store
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Stats, the player's games added up (`server.stats_now`, `server.stats_positions` and their
 * helpers `_filter` and `_drills`; profile/Stats.kt, docs/stats-plan.md): every section's
 * numbers for the games a filter takes, with puzzles for each area to work on, and the positions
 * behind a figure.
 *
 * The phone reviews a game when the player asks (Tai, 2026-09-30: no background review, as
 * chess.com has none), so the numbers come without a background reviewer's progress: `backlog`
 * is null, as on a desktop panel started without one.
 */
class StatsApi(private val files: File) {

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
            Stats.stats(store, filter(query), now).also { drills(it.getJSONArray("work"), files, store) }
        }
        return JSONObject().put("ok", true).put("stats", numbers).put("backlog", JSONObject.NULL)
    }

    /**
     * `GET /api/stats/positions`: the player's moves behind a figure, the latest first -- `what`
     * the figure's selector, `arg` its kind. `stats_positions`.
     */
    fun positions(query: Map<String, String>, now: Double = System.currentTimeMillis() / 1000.0): JSONObject {
        val found = withStore { Stats.positions(it, filter(query), query["what"] ?: "", query["arg"] ?: "", now) }
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
    }
}
