package com.sidecoach.ui

import android.content.Context
import android.os.StatFs
import com.sidecoach.analysis.EngineAssets
import org.json.JSONArray
import org.json.JSONObject
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

/**
 * First-run setup: what the app needs beyond its code, as the panel's setup page asks for it
 * (the `/api/setup` routes in `sidecoach/ui/server.py`, rendered by `desktop/src/routes/setup`).
 *
 * On a computer that is Stockfish, the opening book, the puzzle sample and, if the machine
 * can use it, the CUDA pack. On the phone it is one item: the engine's network. The engine
 * itself ships in the APK (it has to, to be executed), and the network is the 98 MB that
 * would otherwise be in it once per ABI -- the same bargain as the desktop's, the other way
 * round.
 *
 * The shape of the answer is the page's, so the same progress bar, the same speed and the
 * same "N MB left" work without a line of frontend change.
 */
class SetupApi(private val context: Context) {

    private val pool = Executors.newSingleThreadExecutor()
    private val running = AtomicBoolean(false)

    @Volatile
    private var error: String? = null

    fun status(): JSONObject {
        val found = EngineAssets.found(context)
        val state = EngineAssets.net.value
        val detail = when (state) {
            is EngineAssets.State.Ready -> "${state.file.length() / 1_000_000} MB, on the phone"
            is EngineAssets.State.Fetching -> "fetching"
            is EngineAssets.State.Failed -> state.why
            else -> "not here yet; the game is still watched and followed, only unassessed"
        }
        val item = JSONObject().apply {
            put("id", "net")
            put("title", "The engine's network")
            put("size", EngineAssets.NET_BYTES)
            put("present", found != null)
            put("optional", false)
            put("detail", detail)
        }
        val run = JSONObject().apply {
            put("running", running.get())
            put("item", if (running.get()) "net" else JSONObject.NULL)
            val fetching = state as? EngineAssets.State.Fetching
            put("done", fetching?.bytes ?: 0L)
            put("total", fetching?.total ?: EngineAssets.NET_BYTES)
            put("errors", JSONObject().apply { error?.let { put("net", it) } })
            put("finished", JSONArray().apply { if (found != null) put("net") })
            put("queue", JSONArray())
        }
        return JSONObject().apply {
            put("items", JSONArray().put(item))
            put("ready", found != null)
            put("run", run)
            put("data_dir", context.filesDir.absolutePath)
            put("free_bytes", freeBytes())
            // No GPU pack on a phone; the page shows the section only when it is told of one.
            put("gpu", JSONObject.NULL)
        }
    }

    fun start(): JSONObject {
        if (!running.compareAndSet(false, true)) return JSONObject().put("ok", false)
        error = null
        PanelEngine.emit("sys", "info", "fetching the engine's network (98 MB)")
        pool.execute {
            try {
                val file = EngineAssets.fetch(context)
                if (file == null) {
                    error = "the download did not finish"
                    PanelEngine.emit("sys", "warn", "the engine's network did not download")
                } else {
                    PanelEngine.emit(
                        "sys",
                        "info",
                        "the engine's network is ready (${file.length() / 1_000_000} MB)",
                    )
                }
            } catch (exc: Exception) {
                error = exc.toString()
                PanelEngine.emit("sys", "warn", "the engine's network did not download: $exc")
            } finally {
                running.set(false)
            }
        }
        return JSONObject().put("ok", true)
    }

    fun cancel(): JSONObject = JSONObject().put("ok", false)

    private fun freeBytes(): Long = try {
        val stat = StatFs(context.filesDir.absolutePath)
        stat.availableBlocksLong * stat.blockSizeLong
    } catch (_: Exception) {
        -1L
    }
}
