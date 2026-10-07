package com.sidecoach

import android.database.sqlite.SQLiteDatabase
import com.sidecoach.profile.PuzzleStore
import java.io.File

/**
 * Housekeeping over the app's files: `sidecoach/maintenance.py`. The profile emptied in place,
 * and the caches removed, each said in words for the log.
 */
object Maintenance {

    /**
     * Empty the profile database in place (every table), so a store open on it -- the running
     * session's -- stays valid and simply finds nothing. Returns the path.
     */
    fun resetProfile(files: File): File {
        val path = File(files, "profile.sqlite")
        if (!path.exists()) return path
        val db = SQLiteDatabase.openDatabase(path.path, null, SQLiteDatabase.OPEN_READWRITE)
        try {
            val tables = ArrayList<String>()
            db.rawQuery(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
                null,
            ).use { while (it.moveToNext()) tables.add(it.getString(0)) }
            for (table in tables) db.execSQL("DELETE FROM \"$table\"") // names from the schema
            runCatching { db.execSQL("VACUUM") } // another connection busy
        } finally {
            db.close()
        }
        return path
    }

    /**
     * Remove the caches in the app's files; returns what was done, in words.
     *
     * The theme files are deleted. The puzzle store, when a running session holds it open
     * ([puzzles]), is emptied and filled again from the CSVs beside it rather than deleted
     * under that session; otherwise the file goes, with its journal, and the next Start
     * rebuilds it.
     */
    fun clearCaches(files: File, puzzles: PuzzleStore? = null): List<String> {
        val done = ArrayList<String>()
        val themes = File(files, "themes")
            .listFiles { f: File -> f.name.startsWith("theme-") && f.name.endsWith(".bin") }
            ?.sortedBy { it.name }
            .orEmpty()
        for (path in themes) {
            if (runCatching { path.delete() }.getOrDefault(false)) done.add("removed ${path.name}")
        }
        if (puzzles != null) {
            puzzles.clear()
            var added = 0
            val csvs = File(files, "puzzles").listFiles { f: File -> f.name.endsWith(".csv") }
                ?.sortedBy { it.name }
                .orEmpty()
            for (csv in csvs) added += puzzles.importCsv(csv)
            done.add("puzzle store rebuilt: $added puzzles")
        } else {
            for (name in listOf("puzzles.sqlite", "puzzles.sqlite-journal")) {
                val path = File(files, name)
                if (path.exists() && runCatching { path.delete() }.getOrDefault(false)) done.add("removed $name")
            }
        }
        return done
    }
}
