package com.sidecoach.setup

import android.content.ComponentName
import android.content.pm.PackageManager
import android.content.pm.ServiceInfo
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.CountDownLatch
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * Tai, 2026-10-02: the setup's downloads ran on a plain thread, and leaving the app mid-download
 * could stop them. They run in a foreground service now, its notification shown only while a
 * download runs (Android's requirement); progress and Cancel as before. What the notification
 * shows follows the run's state ([SetupNotice]); the run tells it each change ([Setup.watcher]);
 * the service is declared for data sync and is not another app's to start.
 */
@RunWith(AndroidJUnit4::class)
class SetupServiceTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    private fun run(running: Boolean, item: String?, done: Long, total: Long) = JSONObject().apply {
        put("running", running)
        put("item", item ?: JSONObject.NULL)
        put("done", done)
        put("total", total)
        put("titles", JSONObject().put("a", "The voice").put("b", "Speech recognition"))
        put("errors", JSONObject())
        put("finished", JSONArray())
    }

    @Test
    fun theNoticeShowsWhileARunGoesAndOnlyWhatChanges() {
        val shown = ArrayList<SetupNotice.Shown?>()
        val notice = SetupNotice { shown.add(it) }
        notice.on(run(false, null, 0, 0)) // no run: nothing shown, nothing said
        assertEquals(emptyList<SetupNotice.Shown?>(), shown)
        notice.on(run(true, null, 0, 1000))
        notice.on(run(true, "a", 0, 1000))
        notice.on(run(true, "a", 4, 1000)) // still 0 %: not shown again
        notice.on(run(true, "a", 10, 1000))
        notice.on(run(true, "a", 19, 1000))
        notice.on(run(true, "b", 500, 1000))
        notice.on(run(false, null, 500, 1000)) // over (done, cancelled or failed): gone
        notice.on(run(false, null, 500, 1000))
        assertEquals(
            listOf(
                SetupNotice.Shown("Getting ready", 0),
                SetupNotice.Shown("The voice", 0),
                SetupNotice.Shown("The voice", 1),
                SetupNotice.Shown("Speech recognition", 50),
                null,
            ),
            shown,
        )
        // A run of unknown size: shown without a figure.
        shown.clear()
        notice.on(run(true, "a", 5, 0))
        assertEquals(listOf(SetupNotice.Shown("The voice", null)), shown)
    }

    @Test
    fun aRunTellsItsWatcherFromStartToEndCancelToo() {
        val told = LinkedBlockingQueue<SetupNotice.Shown?>()
        val midway = CountDownLatch(1)
        val go = CountDownLatch(1)
        val setup = Setup(app, itemsOf = {
            listOf(
                Item("a", "The voice", 100, { false }, { progress ->
                    progress(50, 100)
                    midway.countDown()
                    assertTrue(go.await(60, TimeUnit.SECONDS))
                    progress(60, 100) // after the Cancel: stops here
                }),
                Item("b", "Speech recognition", 100, { false }, { progress -> progress(1, 100) }),
            )
        })
        val notice = SetupNotice { told.put(it ?: GONE) }
        setup.watcher = notice::on
        assertTrue(setup.start())
        assertTrue(midway.await(60, TimeUnit.SECONDS))
        // Shown while the run goes: getting ready, the first item, its half.
        assertEquals(SetupNotice.Shown("Getting ready", 0), told.poll(60, TimeUnit.SECONDS))
        assertEquals(SetupNotice.Shown("The voice", 0), told.poll(60, TimeUnit.SECONDS))
        assertEquals(SetupNotice.Shown("The voice", 25), told.poll(60, TimeUnit.SECONDS))
        setup.cancel()
        go.countDown()
        // The Cancel ends the run, and the notice with it.
        assertEquals(GONE, told.poll(60, TimeUnit.SECONDS))
        assertFalse(setup.runState().getBoolean("running"))
    }

    @Test
    fun theServiceIsTheAppsOwnForDataSync() {
        val info = app.packageManager.getServiceInfo(
            ComponentName(app, SetupService::class.java),
            PackageManager.ComponentInfoFlags.of(0),
        )
        assertFalse("another app could start it", info.exported)
        assertEquals(ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC, info.foregroundServiceType)
        val asked = app.packageManager.getPackageInfo(
            app.packageName,
            PackageManager.PackageInfoFlags.of(PackageManager.GET_PERMISSIONS.toLong()),
        ).requestedPermissions.orEmpty().toList()
        assertTrue(asked.toString(), "android.permission.FOREGROUND_SERVICE_DATA_SYNC" in asked)
    }

    private companion object {
        /** The notice gone, in a queue that takes no null. */
        val GONE = SetupNotice.Shown("gone", -1)
    }
}
