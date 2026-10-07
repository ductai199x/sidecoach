package com.sidecoach.voice

import android.util.Log

/**
 * The voice's logger: `logging.getLogger("sidecoach.voice")`, which every module of
 * `sidecoach/voice/` logs to. A warning or an error goes to the logcat and to the [handler], which
 * the panel installs as the desktop's `PanelEngine.__init__` adds its `_VoiceLogHandler`: a voice
 * that failed to load, or a line that failed, is said in the panel's log, not only in a console
 * the person never sees (desktop 41dc71d; the parity audit, 2026-09-26).
 */
object VoiceLog {
    const val TAG = "sidecoach.voice"

    /**
     * What a warning or an error is handed to, with its level ("warn" or "error") and message:
     * `_VoiceLogHandler.emit`. Null until the panel installs it.
     */
    @Volatile var handler: ((String, String) -> Unit)? = null

    /** `log.warning(...)`. */
    fun warn(message: String) {
        Log.w(TAG, message)
        forward("warn", message)
    }

    /** `log.error(...)`. */
    fun error(message: String) {
        Log.e(TAG, message)
        forward("error", message)
    }

    /** A handler that fails never raises out of the log (`logging.Handler.handleError`). */
    private fun forward(level: String, message: String) {
        runCatching { handler?.invoke(level, message) }
    }

    /** `str(exc)`, as `%s` writes an exception: its message, or the exception when it has none. */
    fun said(exc: Throwable): String = exc.message ?: exc.toString()
}
