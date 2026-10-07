package com.sidecoach.capture

import android.app.Activity
import android.app.AlertDialog
import android.content.Context
import android.content.pm.PackageManager

/**
 * The chess apps on the phone that can be told when their screen is recorded, named before a
 * session starts (Tai, 2026-09-24: "detect if the app it's reading the board from have the
 * DETECT_SCREEN_RECORDING permission and warn the user about it with a confirmation popup").
 *
 * Since Android 15 an app that declares `DETECT_SCREEN_RECORDING` may be told when one of its
 * windows is in a recording -- not by which app, and nothing of what the recording shows. A
 * capture is not told which app it shows, so the check is over the chess apps installed (the
 * manifest's `<queries>` names them: an app sees only the packages it names). The answer is
 * kept per app and version: a new version is where the permission would come from.
 */
object RecordingNotice {
    const val PERMISSION = "android.permission.DETECT_SCREEN_RECORDING"
    private const val PREFS = "recording_notice"

    /** The chess apps looked at, and the names they go by. */
    val APPS: Map<String, String> = linkedMapOf(
        "com.chess" to "Chess.com",
        "org.lichess.mobileapp" to "Lichess",
        "org.lichess.mobileV2" to "Lichess",
        "com.chesskid" to "ChessKid",
    )

    data class Watcher(val pkg: String, val name: String, val version: Long)

    /** What an app declares -- its permissions and its version -- or null when it is not there. */
    fun interface Lookup {
        fun find(pkg: String): Pair<List<String>, Long>?
    }

    fun installed(context: Context): Lookup = Lookup { pkg ->
        runCatching {
            val info = context.packageManager.getPackageInfo(
                pkg,
                PackageManager.PackageInfoFlags.of(PackageManager.GET_PERMISSIONS.toLong()),
            )
            (info.requestedPermissions?.toList() ?: emptyList()) to info.longVersionCode
        }.getOrNull()
    }

    /** The chess apps installed that declare the permission. */
    fun watchers(lookup: Lookup): List<Watcher> = APPS.mapNotNull { (pkg, name) ->
        val (declared, version) = lookup.find(pkg) ?: return@mapNotNull null
        if (PERMISSION in declared) Watcher(pkg, name, version) else null
    }

    /** The chess apps installed and whether each can be told, for the log: the check leaves no
     * other trace when none can. */
    fun survey(lookup: Lookup): String = APPS.mapNotNull { (pkg, name) ->
        val (declared, version) = lookup.find(pkg) ?: return@mapNotNull null
        "$name ($pkg $version) ${if (PERMISSION in declared) "can" else "cannot"} be told"
    }.joinToString("; ").ifEmpty { "none of the chess apps it knows is installed" }

    /** Those not yet agreed to at the version installed now. */
    fun unanswered(context: Context, lookup: Lookup = installed(context)): List<Watcher> {
        val seen = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return watchers(lookup).filter { seen.getLong(it.pkg, -1L) != it.version }
    }

    fun agree(context: Context, apps: List<Watcher>) {
        val edit = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit()
        for (app in apps) edit.putLong(app.pkg, app.version)
        edit.apply()
    }

    /** "Chess.com", "Chess.com and Lichess", "Chess.com, Lichess and ChessKid". */
    fun names(apps: List<Watcher>): String {
        val names = apps.map { it.name }.distinct()
        return if (names.size <= 1) names.firstOrNull().orEmpty()
        else names.dropLast(1).joinToString(", ") + " and " + names.last()
    }

    /** The question's title and message. */
    fun words(apps: List<Watcher>): Pair<String, String> {
        val who = names(apps)
        val one = apps.map { it.name }.distinct().size == 1
        val title = if (one) "$who can tell it is being recorded" else "$who can tell they are being recorded"
        val message = (if (one) "$who has asked Android to be told when its screen is recorded. " +
            "If you open it while Sidecoach watches the screen, it can be told that a recording is running."
        else "$who have asked Android to be told when their screen is recorded. " +
            "If you open one of them while Sidecoach watches the screen, it can be told that a recording is running.") +
            " It is not told which app is recording, and it gets nothing of what is recorded.\n\n" +
            "Sidecoach asks again only when ${if (one) "it" else "one of them"} updates."
        return title to message
    }

    /** The question, before the screen is shared; Continue keeps the answer for these versions. */
    fun ask(activity: Activity, apps: List<Watcher>, proceed: () -> Unit, cancelled: () -> Unit): AlertDialog {
        val (title, message) = words(apps)
        return AlertDialog.Builder(activity)
            .setTitle(title)
            .setMessage(message)
            .setPositiveButton("Continue") { _, _ ->
                agree(activity, apps)
                proceed()
            }
            .setNegativeButton("Cancel") { _, _ -> cancelled() }
            .setOnCancelListener { cancelled() }
            .show()
    }
}
