package com.sidecoach.ui

import android.content.Context
import android.os.StatFs
import com.sidecoach.setup.Setup
import org.json.JSONObject

/**
 * The `/api/setup` routes of `sidecoach/ui/server.py`, over [Setup]: the status with the free
 * space beside it, a start with the items the page chose, a cancel. The page is the desktop's
 * own (`desktop/src/routes/setup`), so the shapes are the desktop's.
 */
class SetupApi(private val context: Context) {

    private val setup = Setup(context)

    fun status(): JSONObject = setup.status().apply {
        // No GPU pack on a phone; the page shows the section only when it is told of one.
        put("gpu", JSONObject.NULL)
        put("free_bytes", freeBytes())
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

    private fun freeBytes(): Long = try {
        val stat = StatFs(context.filesDir.absolutePath)
        stat.availableBlocksLong * stat.blockSizeLong
    } catch (_: Exception) {
        -1L
    }
}
