package com.sidecoach.analysis

import android.content.Context
import java.io.File

/**
 * The engine's network, fetched once and kept.
 *
 * The desktop downloads Stockfish itself on first run rather than shipping it; the phone
 * ships the engine -- 1.2 MB, because it has to be executed from the APK's library directory
 * -- and downloads the network, which is 98 MB and would otherwise be in the APK once per
 * ABI. Same rule, opposite halves.
 *
 * It is a data file, so it lives in the app's own directory and needs no permission to read.
 * Setup fetches it (`setup/Setup.kt`); until then the app works without an engine: the game
 * is still watched and followed, it is only unassessed.
 */
object EngineAssets {

    /** The network `src/evaluate.h` names for the Stockfish the app is built from. */
    const val NET = "nn-1a298aa575a0.nnue"
    const val NET_URL = "https://tests.stockfishchess.org/api/nn/$NET"
    const val NET_BYTES = 98_511_183L

    /** The network already on disk, or null. */
    fun found(context: Context): File? {
        val file = File(context.filesDir, NET)
        if (file.exists() && file.length() > 1_000_000) return file
        // A network put there by hand for development may have another name.
        return context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull()
    }
}
