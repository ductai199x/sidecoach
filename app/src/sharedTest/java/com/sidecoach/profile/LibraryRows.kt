package com.sidecoach.profile

import org.json.JSONObject

/**
 * A library row as the desktop's fixtures write it (`profile/library.json`, `ui/analysis.json`):
 * read by LibraryParityTest on the device and ReviewAnalysisParityTest on the host.
 */
object LibraryRows {
    private fun optString(o: JSONObject, name: String): String? = if (o.isNull(name)) null else o.getString(name)
    private fun optDouble(o: JSONObject, name: String): Double? = if (o.isNull(name)) null else o.getDouble(name)
    private fun optInt(o: JSONObject, name: String): Int? = if (o.isNull(name)) null else o.getInt(name)

    fun libraryRow(o: JSONObject) = LibraryRow(
        key = o.getString("key"), source = o.getString("source"), addedAt = o.getDouble("added_at"),
        pgn = o.getString("pgn"), playedAt = optDouble(o, "played_at"), white = optString(o, "white"),
        black = optString(o, "black"), whiteElo = optInt(o, "white_elo"), blackElo = optInt(o, "black_elo"),
        result = optString(o, "result"), timeClass = optString(o, "time_class"), userSide = optString(o, "user_side"),
    )
}
