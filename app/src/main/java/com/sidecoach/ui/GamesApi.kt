package com.sidecoach.ui

import com.sidecoach.profile.Library
import com.sidecoach.profile.LibraryRow
import com.sidecoach.profile.Store
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException

/**
 * Your games (`server._library`, `server._import`, the DELETE route): the games Game Review
 * can open, imported and watched, and bringing more in -- a pasted PGN, the player's recent
 * games on chess.com or Lichess by the usernames in the settings.
 *
 * The phone reviews a game when the player asks (Tai, 2026-09-30: no background review, as
 * chess.com has none), so the list has no background reviewer's progress: `backlog` is null,
 * as on a desktop panel started without one.
 */
class GamesApi(private val files: File, private val settings: SettingsStore) {

    private fun <T> withStore(use: (Store) -> T): T {
        val store = Store(File(files, "profile.sqlite"))
        try {
            return use(store)
        } finally {
            store.close()
        }
    }

    private fun users(): Pair<String, String> {
        val profile = settings.section("profile")
        return profile.optString("chesscom_user", "") to profile.optString("lichess_user", "")
    }

    /**
     * The games Game Review can open, the latest played first -- imported and watched, a
     * watched game its import stands for once, as the import -- each with its review's
     * accuracies when it has one, and the player's usernames. `_library`.
     */
    fun list(): JSONObject {
        val games = JSONArray()
        withStore { store ->
            val twins = Library.reconcile(store)
            ReviewRoom.carryTwins(store, twins) // a pair's review goes with the import that stands for it
            val watched = Library.fromWatched(store).filter { it.key !in twins }
            val rows = (store.library() + watched).sortedByDescending { it.playedAt?.takeIf { p -> p != 0.0 } ?: it.addedAt }
            for (row in rows) {
                val review = store.latestReview(row.key)
                val summary = review?.summary?.takeIf { it.isNotEmpty() }?.let { JSONObject(it) }
                games.put(
                    rowJson(row).apply {
                        put("pgn", JSONObject.NULL) // the page lists games; a review reads its own PGN
                        put("reviewed", review != null)
                        put("accuracy", summary?.let { s -> sides { s.optJSONObject(it)?.opt("accuracy") } } ?: JSONObject.NULL)
                        put(
                            "performance",
                            summary?.let { s -> sides { s.optJSONObject(it)?.optJSONObject("performance")?.opt("rating") } }
                                ?: JSONObject.NULL,
                        )
                    },
                )
            }
        }
        val (chesscom, lichess) = users()
        return JSONObject()
            .put("games", games)
            .put("users", JSONObject().put("chesscom", chesscom).put("lichess", lichess))
            .put("backlog", JSONObject.NULL)
    }

    private fun sides(value: (String) -> Any?): JSONObject =
        JSONObject().put("white", value("white") ?: JSONObject.NULL).put("black", value("black") ?: JSONObject.NULL)

    /**
     * Games into the library: a pasted PGN, or the player's recent games on chess.com or
     * Lichess (their usernames from the settings). What came in says how many are new and how
     * many were there already; a failure says what to do, and names the setting when the
     * username is at fault (`field`: the page links to it). `_import`.
     */
    fun import(body: JSONObject): JSONObject {
        val source = body.getString("source")
        val (chesscomUser, lichessUser) = users()
        val names = listOf(chesscomUser, lichessUser).filter { it.isNotEmpty() }
        val site = mapOf("chesscom" to "chess.com", "lichess" to "Lichess")[source] ?: ""
        val field = if (site.isNotEmpty()) "profile.${source}_user" else null
        fun failed(error: String, field: String? = null) =
            JSONObject().put("ok", false).put("error", error).apply { if (field != null) put("field", field) }
        val rows: List<LibraryRow> = try {
            when (source) {
                "pgn" -> Library.fromPgn(body.optString("text", ""), names)
                "chesscom" -> {
                    if (chesscomUser.isEmpty()) return failed("Add your chess.com username first.", field)
                    Library.fromChesscom(chesscomUser, months = maxOf(1, body.optInt("months", 3)))
                }
                "lichess" -> {
                    if (lichessUser.isEmpty()) return failed("Add your Lichess username first.", field)
                    Library.fromLichess(lichessUser)
                }
                else -> return failed("no source $source")
            }
        } catch (exc: Library.NoPlayer) {
            return failed("${exc.message} Check the spelling.", field)
        } catch (_: IOException) {
            return failed("Could not reach $site. Check the connection and try again.")
        } catch (exc: Exception) { // anything else is said, never raised
            return failed("Could not bring the games in: ${exc.message ?: exc}")
        }
        // A site's games leave out the ones the player removed; a game pasted by hand comes back.
        val (kept, out, added) = withStore { store ->
            val out = if (source == "pgn") {
                store.unremove(rows.map { it.key })
                emptySet()
            } else {
                Library.leftOut(store, rows)
            }
            val kept = rows.filter { it.key !in out }
            val new = kept.filter { store.libraryGame(it.key) == null }
            for (row in kept) store.addLibraryGame(row)
            Library.reconcile(store) // a game Sidecoach watched, now imported: one game
            Triple(kept, out, new.size)
        }
        return JSONObject()
            .put("ok", true)
            .put("added", added)
            .put("already", kept.size - added)
            .put("left_out", out.size)
            .put("keys", JSONArray(kept.map { it.key }))
    }

    /** A game out of the player's record, with the game it is paired with; Undo puts it back. */
    fun delete(key: String): JSONObject {
        val removed = withStore { it.removeGame(key) }
        return JSONObject().put("ok", true).put("removed", JSONArray(removed))
    }

    /** A game removed moments ago, put back. `games_restore`. */
    fun restore(key: String): JSONObject {
        val restored = withStore { it.restoreGame(key) }
        if (restored.isEmpty()) {
            return JSONObject().put("ok", false).put("error", "That game was removed too long ago to put back.")
        }
        return JSONObject().put("ok", true).put("restored", JSONArray(restored))
    }

    companion object {
        /** A row as the page reads it: `row.__dict__`, its fields in their order. */
        fun rowJson(row: LibraryRow): JSONObject = JSONObject()
            .put("key", row.key)
            .put("source", row.source)
            .put("added_at", row.addedAt)
            .put("pgn", row.pgn)
            .put("played_at", row.playedAt ?: JSONObject.NULL)
            .put("white", row.white ?: JSONObject.NULL)
            .put("black", row.black ?: JSONObject.NULL)
            .put("white_elo", row.whiteElo ?: JSONObject.NULL)
            .put("black_elo", row.blackElo ?: JSONObject.NULL)
            .put("result", row.result ?: JSONObject.NULL)
            .put("time_class", row.timeClass ?: JSONObject.NULL)
            .put("user_side", row.userSide ?: JSONObject.NULL)
    }
}
