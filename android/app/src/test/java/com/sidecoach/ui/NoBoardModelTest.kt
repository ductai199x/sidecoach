package com.sidecoach.ui

import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * With no board-reading model the status line says so, and to reinstall (`panel.NO_BOARD_MODEL`,
 * word for word); with the model it is the usual "waiting for a board". The phone said "waiting
 * for a board" for ever (the audit of 2026-10-07). The session itself: PanelEngineTest's
 * theBoardReadingModelMissingIsSaidAndOnlyThen, on a device.
 */
class NoBoardModelTest {

    @Test
    fun theStatusSaysTheModelIsMissingOnlyWhenItIs() {
        assertEquals(
            "The board-reading model is missing or could not load: reinstall Sidecoach.",
            waitingStatus(noBoardModel = true),
        )
        assertEquals(NO_BOARD_MODEL, waitingStatus(noBoardModel = true))
        assertEquals("waiting for a board", waitingStatus(noBoardModel = false))
    }
}
