package com.sidecoach.analysis

import android.content.Context
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

/**
 * The engine's network, fetched once and kept.
 *
 * The desktop downloads Stockfish itself on first run rather than shipping it; the phone
 * ships the engine -- 1.2 MB, because it has to be executed from the APK's library directory
 * -- and downloads the network, which is 98 MB and would otherwise be in the APK once per
 * ABI. Same rule, opposite halves.
 *
 * It is a data file, so it lives in the app's own directory and needs no permission to read.
 * A download that fails leaves nothing behind and the app works without an engine: the game
 * is still watched and followed, it is only unassessed.
 */
object EngineAssets {

    /** The network `src/evaluate.h` names for the Stockfish the app is built from. */
    const val NET = "nn-1a298aa575a0.nnue"
    const val NET_URL = "https://tests.stockfishchess.org/api/nn/$NET"
    const val NET_BYTES = 98_511_183L

    sealed class State {
        object Missing : State()
        data class Fetching(val bytes: Long, val total: Long) : State()
        data class Ready(val file: File) : State()
        data class Failed(val why: String) : State()
    }

    private val state = MutableStateFlow<State>(State.Missing)
    val net: StateFlow<State> = state.asStateFlow()

    /** The network already on disk, or null. */
    fun found(context: Context): File? {
        val file = File(context.filesDir, NET)
        if (file.exists() && file.length() > 1_000_000) {
            state.value = State.Ready(file)
            return file
        }
        // A network put there by hand for development may have another name.
        val other = context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull()
        if (other != null) state.value = State.Ready(other)
        return other
    }

    /**
     * Fetch the network unless it is already there. Blocks, so it runs on a worker; the
     * state flow carries the progress for the screen.
     */
    fun fetch(context: Context): File? {
        found(context)?.let { return it }
        val target = File(context.filesDir, NET)
        val partial = File(context.filesDir, "$NET.part")
        state.value = State.Fetching(0, NET_BYTES)
        try {
            val connection = (URL(NET_URL).openConnection() as HttpURLConnection).apply {
                // The net server turns away a request with no User-Agent.
                setRequestProperty("User-Agent", "sidecoach")
                connectTimeout = 30_000
                readTimeout = 60_000
                instanceFollowRedirects = true
            }
            val total = connection.contentLengthLong.takeIf { it > 0 } ?: NET_BYTES
            connection.inputStream.use { input ->
                partial.outputStream().use { output ->
                    val buffer = ByteArray(1 shl 16)
                    var read = 0L
                    while (true) {
                        val n = input.read(buffer)
                        if (n < 0) break
                        output.write(buffer, 0, n)
                        read += n
                        state.value = State.Fetching(read, total)
                    }
                }
            }
            if (partial.length() < 1_000_000) throw IllegalStateException("only ${partial.length()} bytes")
            partial.renameTo(target)
            Log.i(TAG, "network ready: ${target.length() / 1_000_000} MB")
            state.value = State.Ready(target)
            return target
        } catch (exc: Exception) {
            runCatching { partial.delete() }
            Log.w(TAG, "the network could not be fetched: $exc")
            state.value = State.Failed(exc.message ?: exc::class.java.simpleName)
            return null
        }
    }

    private const val TAG = "sidecoach.engine"
}
