package com.sidecoach.ui

import android.app.Instrumentation
import android.os.ParcelFileDescriptor
import android.os.SystemClock

/** The phone's home screen, for the caption tests: the box shows over other apps. */
internal object Launcher {
    /**
     * The home screen in front: Home pressed, and back once the launcher's window is the active
     * one (a test before may have left Sidecoach's panel up). A state waited for, not a time: it
     * used to be a second and a half's sleep, whether Home had anything to do or not.
     */
    fun inFront(instrumentation: Instrumentation) {
        val automation = instrumentation.uiAutomation
        // The launcher as the shell resolves Home: the app itself may not see it (Android 11's
        // package visibility answers with the settings' fallback instead).
        val home = ParcelFileDescriptor.AutoCloseInputStream(
            automation.executeShellCommand("pm resolve-activity --brief -c android.intent.category.HOME -a android.intent.action.MAIN"),
        ).bufferedReader().use { it.readText() }.lines().last { it.isNotBlank() }.substringBefore('/').trim()
        automation.executeShellCommand("input keyevent KEYCODE_HOME").close()
        val until = SystemClock.uptimeMillis() + 20_000 // a safety net only
        while (SystemClock.uptimeMillis() < until) {
            val front = automation.rootInActiveWindow?.packageName?.toString()
            if (front != null && front == home) return
            SystemClock.sleep(20)
        }
        throw AssertionError("the home screen ($home) never came to the front")
    }
}
