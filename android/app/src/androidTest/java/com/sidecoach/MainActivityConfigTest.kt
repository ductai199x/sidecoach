package com.sidecoach

import android.content.ComponentName
import android.content.pm.ActivityInfo
import android.content.pm.PackageManager
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The panel's activity as the system sees it (setup findings 6 and 12): one instance (the
 * session notification opened a second one, with a second server on another port), and not
 * rebuilt by a turn of the phone, dark mode or the font size (the page and its edits were lost).
 */
@RunWith(AndroidJUnit4::class)
class MainActivityConfigTest {

    @Test
    fun onePanelThatOutlivesConfigurationChanges() {
        val app = InstrumentationRegistry.getInstrumentation().targetContext
        val info = app.packageManager.getActivityInfo(
            ComponentName(app, MainActivity::class.java),
            PackageManager.ComponentInfoFlags.of(0),
        )
        assertEquals(ActivityInfo.LAUNCH_SINGLE_TASK, info.launchMode)
        val wanted = ActivityInfo.CONFIG_ORIENTATION or ActivityInfo.CONFIG_SCREEN_SIZE or
            ActivityInfo.CONFIG_SMALLEST_SCREEN_SIZE or ActivityInfo.CONFIG_SCREEN_LAYOUT or
            ActivityInfo.CONFIG_UI_MODE or ActivityInfo.CONFIG_FONT_SCALE or ActivityInfo.CONFIG_DENSITY
        assertTrue("configChanges ${info.configChanges}", info.configChanges and wanted == wanted)
    }

    /**
     * Tai, 2026-10-02: the test board (debug.BoardActivity) is the debug build's alone. In the
     * phone build, the one he runs and these tests run against, no activity but the panel is
     * there for another app to start.
     */
    @Test
    fun theTestBoardIsNotInThePhoneBuild() {
        val app = InstrumentationRegistry.getInstrumentation().targetContext
        val info = app.packageManager.getPackageInfo(
            app.packageName,
            PackageManager.PackageInfoFlags.of(PackageManager.GET_ACTIVITIES.toLong()),
        )
        val activities = info.activities.orEmpty().map { it.name }
        assertTrue("$activities", activities.none { it.endsWith(".debug.BoardActivity") })
        assertEquals(listOf(MainActivity::class.java.name), activities.filter { name ->
            info.activities.orEmpty().first { it.name == name }.exported
        })
    }
}
