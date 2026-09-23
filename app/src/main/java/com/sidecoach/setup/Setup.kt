package com.sidecoach.setup

import android.content.Context
import com.sidecoach.analysis.EngineAssets
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * First-run setup: what the app needs beyond its code, fetched into its files.
 * `sidecoach/setup/__init__.py`.
 *
 * The APK carries the code and the engine; at first run the app fetches, with one progress
 * bar for the lot, what the desktop fetches and the phone can use: the engine's network (the
 * phone's half of the desktop's Stockfish item -- the engine ships in the APK, because it has
 * to be executed from there, and the 98 MB network would be in it once per ABI) and the
 * opening tables. Each item knows whether it is present, how big it is, and how to fetch
 * itself; [Setup] runs the missing ones on a thread and reports through `/api/setup/status`.
 */

/** Bytes done, and the total when it is known. */
typealias Progress = (done: Long, total: Long?) -> Unit

const val USER_AGENT = "sidecoach-setup"
const val OPENINGS_BASE = "https://raw.githubusercontent.com/lichess-org/chess-openings/master"
val OPENINGS_VOLUMES = listOf("a", "b", "c", "d", "e")

/**
 * Fetch [url] to [target] through a temporary file, resuming a partial one, with progress
 * callbacks; verified against [expectSha256] when given. `setup.download`.
 */
fun download(url: String, target: File, progress: Progress? = null, expectSha256: String? = null): File {
    target.parentFile?.mkdirs()
    val part = File(target.path + ".part")
    var done = if (part.exists()) part.length() else 0L
    val connection = (URL(url).openConnection() as HttpURLConnection).apply {
        setRequestProperty("User-Agent", USER_AGENT)
        if (done > 0) setRequestProperty("Range", "bytes=$done-")
        connectTimeout = 60_000
        readTimeout = 60_000
        instanceFollowRedirects = true
    }
    try {
        val status = connection.responseCode
        if (status >= 400) throw IOException("HTTP $status for $url")
        if (done > 0 && status != 206) done = 0  // the server did not resume; start over
        val length = connection.contentLengthLong
        val total = if (length >= 0) length + done else null
        connection.inputStream.use { input ->
            FileOutputStream(part, done > 0).use { output ->
                val chunk = ByteArray(1 shl 20)
                while (true) {
                    val n = input.read(chunk)
                    if (n < 0) break
                    output.write(chunk, 0, n)
                    done += n
                    progress?.invoke(done, total)
                }
            }
        }
    } finally {
        connection.disconnect()
    }
    if (expectSha256 != null) {
        val digest = MessageDigest.getInstance("SHA-256")
        part.inputStream().use { input ->
            val chunk = ByteArray(1 shl 20)
            while (true) {
                val n = input.read(chunk)
                if (n < 0) break
                digest.update(chunk, 0, n)
            }
        }
        val hex = digest.digest().joinToString("") { "%02x".format(it) }
        if (hex != expectSha256) {
            part.delete()
            throw IOException("${target.name}: checksum mismatch")
        }
    }
    if (!part.renameTo(target)) throw IOException("could not move ${part.name} into place")
    return target
}

/** One thing the app needs: how to tell it is there, how big it is, how to get it. */
class Item(
    val id: String,
    val title: String,
    /** Bytes, an estimate for the progress bar. */
    val size: Long,
    val present: () -> Boolean,
    val fetch: (Progress) -> Unit,
    val optional: Boolean = false,
    val detail: () -> String = { "" },
)

/** What this phone needs, in the order they are fetched. `setup.items`. */
fun items(context: Context): List<Item> {
    val files = context.filesDir
    val out = ArrayList<Item>()

    out.add(
        Item(
            "net",
            "The engine's network",
            EngineAssets.NET_BYTES,
            present = { EngineAssets.found(context) != null },
            fetch = { progress ->
                val target = download(EngineAssets.NET_URL, File(files, EngineAssets.NET), progress)
                if (target.length() < 1_000_000) {
                    target.delete()
                    throw IOException("only ${target.length()} bytes came")
                }
            },
            detail = {
                EngineAssets.found(context)?.let { "${it.length() / 1_000_000} MB, on the phone" }
                    ?: "not here yet; the game is still watched and followed, only unassessed"
            },
        ),
    )

    val openingsDir = File(files, "openings")
    out.add(
        Item(
            "openings",
            "The opening names",
            500_000,
            present = { OPENINGS_VOLUMES.all { File(openingsDir, "$it.tsv").exists() } },
            fetch = { progress ->
                for ((i, v) in OPENINGS_VOLUMES.withIndex()) {
                    val target = File(openingsDir, "$v.tsv")
                    if (!target.exists()) download("$OPENINGS_BASE/$v.tsv", target)
                    progress((i + 1) * 100_000L, null)
                }
            },
        ),
    )
    return out
}

/** The setup's state and the thread that runs it, one per app. `setup.Setup`. */
class Setup(private val context: Context) {

    private val lock = Any()
    private var thread: Thread? = null
    private var state = JSONObject()
    @Volatile private var cancelled = false

    /** Every item with whether it is present, and the run in progress if any. */
    fun status(): JSONObject {
        val rows = JSONArray()
        var needed = 0
        for (item in items(context)) {
            runCatching {
                val present = item.present()
                rows.put(
                    JSONObject().apply {
                        put("id", item.id)
                        put("title", item.title)
                        put("size", item.size)
                        put("present", present)
                        put("optional", item.optional)
                        put("detail", item.detail())
                    },
                )
                if (!present && !item.optional) needed++
            }
        }
        val run = synchronized(lock) { JSONObject(state.toString()) }
        return JSONObject().apply {
            put("items", rows)
            put("ready", needed == 0)
            put("run", run)
            put("data_dir", context.filesDir.absolutePath)
        }
    }

    /**
     * Fetch the items named (default: every missing one that is not optional) on a thread;
     * false when a run is already going.
     */
    fun start(ids: List<String>? = null): Boolean {
        synchronized(lock) {
            if (thread?.isAlive == true) return false
            cancelled = false
            state = JSONObject().apply {
                put("running", true)
                put("item", JSONObject.NULL)
                put("done", 0)
                put("total", 0)
                put("errors", JSONObject())
                put("finished", JSONArray())
            }
        }
        val todo = items(context).filter {
            (ids == null && !it.optional && !it.present()) || (ids != null && it.id in ids && !it.present())
        }
        val total = todo.sumOf { it.size }
        synchronized(lock) {
            state.put("total", total)
            state.put("queue", JSONArray(todo.map { it.id }))
        }
        val worker = Thread({
            var base = 0L
            for (item in todo) {
                if (cancelled) break
                synchronized(lock) { state.put("item", item.id) }
                val start = base
                try {
                    item.fetch { done, _ ->
                        synchronized(lock) { state.put("done", start + minOf(done, item.size)) }
                    }
                    synchronized(lock) { state.getJSONArray("finished").put(item.id) }
                } catch (exc: Exception) {
                    // Reported; the next item goes on.
                    synchronized(lock) {
                        state.getJSONObject("errors").put(item.id, "${exc::class.java.simpleName}: ${exc.message}")
                    }
                }
                base += item.size
                synchronized(lock) { state.put("done", base) }
            }
            synchronized(lock) {
                state.put("running", false)
                state.put("item", JSONObject.NULL)
            }
        }, "setup")
        worker.isDaemon = true
        synchronized(lock) { thread = worker }
        worker.start()
        return true
    }

    /** Stop after the item being fetched. */
    fun cancel() {
        cancelled = true
    }
}
