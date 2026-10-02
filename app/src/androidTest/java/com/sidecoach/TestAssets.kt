package com.sidecoach

import android.util.Log
import androidx.test.platform.app.InstrumentationRegistry
import java.io.InputStream

/**
 * The test fixtures (androidTest/assets) and the app's own assets (main/assets), as the device's
 * tests read them: through the test APK's and the app's AssetManagers. The host's unit tests
 * have the same object over the source tree (test's TestAssets.kt).
 */
object TestAssets {
    /** A fixture of the test APK (`androidTest/assets`). */
    fun open(path: String): InputStream = InstrumentationRegistry.getInstrumentation().context.assets.open(path)

    fun text(path: String): String = open(path).bufferedReader(Charsets.UTF_8).use { it.readText() }

    /** An asset of the app itself (`main/assets`). */
    fun openApp(path: String): InputStream =
        InstrumentationRegistry.getInstrumentation().targetContext.assets.open(path)
}

/** A test's own report line (counts, times), to logcat on the device: standard output on the host. */
object TestLog {
    fun i(tag: String, message: String) {
        Log.i(tag, message)
    }
}
