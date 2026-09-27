package com.sidecoach.ui

import android.content.Context
import android.os.StatFs
import com.sidecoach.setup.Setup
import org.json.JSONObject

/**
 * The `/api/setup` routes of `sidecoach/ui/server.py`, over [Setup]: the status with the free
 * space beside it, a start with the items the page chose, a cancel. The page is the desktop's
 * own (`desktop/src/routes/setup`), so the shapes are the desktop's.
 *
 * One for the app ([shared]), as the desktop's server makes one `Setup` for its life: the
 * activity is rebuilt when the phone turns, and a second one lost the status of a download
 * under way and let a second Start race the first on the same `.part` file.
 */
class SetupApi(private val context: Context) {

    private val setup = Setup(context)

    fun status(): JSONObject = setup.status().apply {
        // No GPU pack on a phone; the page shows the section only when it is told of one.
        put("gpu", JSONObject.NULL)
        put("free_bytes", freeBytes() ?: JSONObject.NULL)
    }

    /** `{"ids": [...]}` the items the page chose; none named, every missing one. */
    fun start(body: JSONObject): JSONObject {
        val chosen = body.optJSONArray("ids")?.let { ids -> (0 until ids.length()).map { ids.getString(it) } }
        return JSONObject().put("ok", setup.start(chosen))
    }

    fun cancel(): JSONObject {
        setup.cancel()
        return JSONObject().put("ok", true)
    }

    /** The free space where the app keeps its files, or null when it cannot be told: `setup.free_space`. */
    private fun freeBytes(): Long? = try {
        val stat = StatFs(context.filesDir.absolutePath)
        stat.availableBlocksLong * stat.blockSizeLong
    } catch (_: Exception) {
        null
    }

    companion object {
        @Volatile private var shared: SetupApi? = null

        /** The app's setup, made on the first call and kept for the process. */
        fun shared(context: Context): SetupApi = shared ?: synchronized(this) {
            shared ?: SetupApi(context.applicationContext).also { shared = it }
        }
    }
}
