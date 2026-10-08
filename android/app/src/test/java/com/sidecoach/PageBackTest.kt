package com.sidecoach

import androidx.activity.OnBackPressedDispatcher
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * Back on the panel ([PageBack]): the page's own history first, out of the app only when there
 * is nowhere left to go -- every time, not only until the app was first left (the audit of
 * 2026-10-07: after one exit by Back and a return, Back left the app from the settings).
 */
class PageBackTest {

    /** The page's history, as the WebView keeps it: how many pages back there are. */
    private var history = 0
    private var wentBack = 0
    private var left = 0

    private fun dispatcher(): OnBackPressedDispatcher {
        // The fallback is the activity's own Back: out of the app.
        lateinit var dispatcher: OnBackPressedDispatcher
        dispatcher = OnBackPressedDispatcher { left++ }
        dispatcher.addCallback(
            PageBack(
                canGoBack = { history > 0 },
                goBack = { history--; wentBack++ },
                dispatcher = { dispatcher },
            ),
        )
        return dispatcher
    }

    @Test
    fun backWalksThePageFirstAndLeavesOnlyWithNowhereToGo() {
        val back = dispatcher()
        history = 1 // the settings, opened from the panel
        back.onBackPressed()
        assertEquals("back to the panel", 1, wentBack)
        assertEquals(0, left)
        back.onBackPressed()
        assertEquals("nowhere left: out of the app", 1, left)
    }

    @Test
    fun afterLeavingOnceBackStillWalksThePageFirst() {
        val back = dispatcher()
        back.onBackPressed() // on the panel: out of the app (Android keeps the activity)
        assertEquals(1, left)
        // Back in the app, the settings opened again: Back returns to the panel, as the first
        // time, and does not leave the app.
        history = 1
        back.onBackPressed()
        assertEquals("back to the panel, not out of the app", 1, wentBack)
        assertEquals(1, left)
        back.onBackPressed()
        assertEquals("then out of the app", 2, left)
    }
}
