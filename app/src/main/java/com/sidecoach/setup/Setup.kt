package com.sidecoach.setup

import android.content.Context
import com.github.luben.zstd.ZstdInputStream
import com.sidecoach.analysis.EngineAssets
import com.sidecoach.profile.Puzzles
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedInputStream
import java.io.BufferedReader
import java.io.File
import java.io.FilterInputStream
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader
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
 * to be executed from there, and the 98 MB network would be in it once per ABI), the voice,
 * the opening tables, speech recognition, and, when asked for, a sample of the Lichess puzzles
 * for practice. Each item knows whether it is present, how big it is, and how to fetch
 * itself; [Setup] runs the missing ones on a thread and reports through `/api/setup/status`.
 */

/** Bytes done, and the total when it is known. */
typealias Progress = (done: Long, total: Long?) -> Unit

const val USER_AGENT = "sidecoach-setup"

/**
 * The voice's files, name -> (url, bytes): `setup.KOKORO_FILES`, with the int8 model where the
 * desktop takes the full-precision one (Tai, 2026-09-23: int8 on the phone, 92 MB for 326).
 */
val KOKORO_FILES: Map<String, Pair<String, Long>> = linkedMapOf(
    "kokoro-v1.0.int8.onnx" to (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx" to
            92_361_271L
        ),
    "voices-v1.0.bin" to (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin" to
            28_214_398L
        ),
)
/**
 * Speech recognition's files, folder -> name -> (url, bytes): what `onnx_asr.load_model` and
 * `load_vad("silero")` fetch through the hub for the desktop, with Parakeet's int8 graphs where
 * the desktop takes the full-precision ones (Tai, 2026-09-23: 670 MB for 2.5 GB).
 */
private const val HF = "https://huggingface.co"
val STT_FILES: Map<String, Map<String, Pair<String, Long>>> = linkedMapOf(
    "models/parakeet" to linkedMapOf(
        "encoder-model.int8.onnx" to ("$HF/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/encoder-model.int8.onnx" to 652_183_999L),
        "decoder_joint-model.int8.onnx" to ("$HF/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/decoder_joint-model.int8.onnx" to 18_202_004L),
        "vocab.txt" to ("$HF/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/vocab.txt" to 93_939L),
        "config.json" to ("$HF/istupakov/parakeet-tdt-0.6b-v3-onnx/resolve/main/config.json" to 97L),
    ),
    "models/silero" to linkedMapOf(
        "silero_vad.onnx" to ("$HF/istupakov/silero-vad-onnx/resolve/main/silero_vad.onnx" to 2_327_524L),
    ),
)
const val OPENINGS_BASE = "https://raw.githubusercontent.com/lichess-org/chess-openings/master"
val OPENINGS_VOLUMES = listOf("a", "b", "c", "d", "e")
const val PUZZLES_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
val PUZZLE_THEMES = listOf(
    "fork", "pin", "skewer", "discoveredAttack", "hangingPiece", "trappedPiece",
    "deflection", "attraction", "interference", "clearance", "exposedKing", "sacrifice",
    "advancedPawn", "promotion", "xRayAttack", "capturingDefender", "doubleCheck",
    "backRankMate", "smotheredMate", "anastasiaMate", "arabianMate", "bodenMate",
    "hookMate", "dovetailMate", "mateIn1", "mateIn2", "mateIn3",
)
const val PUZZLES_PER_THEME = 200
const val PUZZLES_MAX_SCAN = 600_000

/** The sample's size for the progress bar: about what the scan downloads. */
const val PUZZLES_BYTES = 30_000_000L

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

    val kokoroDir = File(files, "models/kokoro")
    out.add(
        Item(
            "kokoro",
            "The voice (Kokoro)",
            KOKORO_FILES.values.sumOf { it.second },
            present = { KOKORO_FILES.keys.all { File(kokoroDir, it).exists() } },
            fetch = { progress ->
                var base = 0L
                for ((name, entry) in KOKORO_FILES) {
                    val (url, size) = entry
                    if (File(kokoroDir, name).exists()) {
                        base += size
                        continue
                    }
                    val at = base
                    download(url, File(kokoroDir, name), { done, _ -> progress(at + done, null) })
                    base += size
                }
            },
        ),
    )

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

    out.add(
        Item(
            "stt",
            "Speech recognition (Parakeet) and the voice gate (Silero)",
            STT_FILES.values.sumOf { folder -> folder.values.sumOf { it.second } },
            present = {
                STT_FILES.all { (folder, names) -> names.keys.all { File(files, "$folder/$it").exists() } }
            },
            fetch = { progress ->
                var base = 0L
                for ((folder, names) in STT_FILES) {
                    for ((name, entry) in names) {
                        val (url, size) = entry
                        val target = File(files, "$folder/$name")
                        if (!target.exists()) {
                            val at = base
                            download(url, target, { done, _ -> progress(at + done, null) })
                        }
                        base += size
                    }
                }
            },
        ),
    )

    val puzzlesCsv = File(files, "puzzles/sample.csv")
    out.add(
        Item(
            "puzzles",
            "A sample of the Lichess puzzles",
            PUZZLES_BYTES,
            present = { puzzlesCsv.exists() },
            fetch = { progress -> fetchPuzzles(puzzlesCsv, progress) },
            optional = true,
        ),
    )
    return out
}

/**
 * A few puzzles of each theme the tagger names, from the start of the Lichess puzzle file:
 * streamed and decompressed as it comes, never stored whole, and the scan stops once every
 * theme has its share or at [maxScan] rows. `setup.items`' `puzzles_fetch`; the source, the
 * themes and the limits are parameters so a test can serve a small file of its own.
 */
fun fetchPuzzles(
    target: File,
    progress: Progress,
    source: () -> InputStream = ::lichessPuzzles,
    themes: List<String> = PUZZLE_THEMES,
    perTheme: Int = PUZZLES_PER_THEME,
    maxScan: Int = PUZZLES_MAX_SCAN,
) {
    val kept = LinkedHashMap<String, ArrayList<List<String>>>()
    for (theme in themes) kept[theme] = ArrayList()
    val seen = HashSet<String>()
    var scanned = 0
    val header: List<String>
    ZstdInputStream(BufferedInputStream(source(), 1 shl 16)).use { zin ->
        val rows = BufferedReader(InputStreamReader(zin, Charsets.UTF_8), 1 shl 16)
        header = csvRow(rows.readLine() ?: throw IOException("the puzzle file is empty"))
        while (true) {
            val line = rows.readLine() ?: break
            val row = csvRow(line)
            scanned += 1
            if (scanned % 20_000 == 0) {
                // In the item's own units (bytes), as the run's bar adds them up.
                progress(scanned.toLong() * PUZZLES_BYTES / maxScan, PUZZLES_BYTES)
            }
            val tagged = row[7].split(' ').filter { it.isNotEmpty() }.toHashSet()
            val wanted = themes.filter { it in tagged && kept.getValue(it).size < perTheme }
            if (wanted.isNotEmpty() && row[0] !in seen) {
                seen.add(row[0])
                for (theme in wanted) kept.getValue(theme).add(row)
            }
            if (themes.all { kept.getValue(it).size >= perTheme } || scanned >= maxScan) break
        }
    }
    val unique = LinkedHashMap<String, List<String>>()
    for (rows in kept.values) for (row in rows) unique[row[0]] = row
    target.parentFile?.mkdirs()
    target.bufferedWriter(Charsets.UTF_8).use { out ->
        out.write(csvLine(header))
        for (row in unique.values) out.write(csvLine(row))
    }
}

/** Lichess's puzzle file, as it comes; closing it ends the connection. */
fun lichessPuzzles(): InputStream {
    val connection = (URL(PUZZLES_URL).openConnection() as HttpURLConnection).apply {
        setRequestProperty("User-Agent", USER_AGENT)
        connectTimeout = 60_000
        readTimeout = 60_000
        instanceFollowRedirects = true
    }
    val status = connection.responseCode
    if (status >= 400) {
        connection.disconnect()
        throw IOException("HTTP $status for $PUZZLES_URL")
    }
    return object : FilterInputStream(connection.inputStream) {
        override fun close() {
            try {
                super.close()
            } finally {
                connection.disconnect()
            }
        }
    }
}

/**
 * One line of the puzzle file as `csv.reader` reads it. The file quotes nothing, so a plain
 * split does -- a line with a quote in it goes through the whole reader.
 */
private fun csvRow(line: String): List<String> =
    if ('"' in line) Puzzles.csvRows(line).firstOrNull() ?: emptyList() else line.split(',')

/** A row as `csv.writer` writes it: minimal quoting, and \r\n after it. */
private fun csvLine(row: List<String>): String = row.joinToString(",", postfix = "\r\n") { field ->
    if (field.any { it == ',' || it == '"' || it == '\r' || it == '\n' }) {
        "\"" + field.replace("\"", "\"\"") + "\""
    } else {
        field
    }
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
