package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The captions' box going through words longer than it shows ([CaptionPace], voice finding 18;
 * Tai, 2026-10-02: never cut a line off, scroll through it at a reading pace, then the next line).
 * The desktop's `desktop/tests/captions_pace.test.mjs`. The time is given by the test and the
 * layout is a fixed number of characters a row: nothing waits, nothing is measured.
 */
@RunWith(AndroidJUnit4::class)
class CaptionPaceTest {
    private val perRow = 40

    private fun rows(length: Int) = object : CaptionRows {
        override val count = maxOf(1, (length + perRow - 1) / perRow)
        override fun rowOf(offset: Int) = offset / perRow
    }

    /** The time the reader takes over [chars] characters, in ms. */
    private fun reading(chars: Int): Long = (chars * 1000 / READ_CPS).toLong()

    @Test
    fun aLineLongerThanTheBoxIsShownFromItsStartRowByRowAtTheReadingPace() {
        val pace = CaptionPace()
        val length = 5 * perRow // five rows in a box of two
        pace.from(0, 0)
        // The start, not the end (the box showed rows 4 and 5 only).
        assertEquals(0, pace.top(rows(length), 2, length))
        pace.advance(reading(perRow) - 1, length)
        assertEquals("the first row is still being read", 0, pace.top(rows(length), 2, length))
        pace.advance(reading(perRow) + 1, length)
        assertEquals("one row on, not a jump", 1, pace.top(rows(length), 2, length))
        pace.advance(reading(2 * perRow) + 1, length)
        assertEquals(2, pace.top(rows(length), 2, length))
        assertFalse(pace.done(length))
        pace.advance(reading(length) + 1, length)
        // The end: the last two rows, as the box always ended.
        assertEquals(3, pace.top(rows(length), 2, length))
        assertTrue(pace.done(length))
        pace.advance(reading(10 * length), length)
        assertEquals(3, pace.top(rows(length), 2, length))
    }

    @Test
    fun theNextLineComesAfterTheOneBeingReadAndFromWhenItCameOnceAllIsRead() {
        val pace = CaptionPace()
        var length = 3 * perRow - 1 // three rows; the newline below ends the third
        pace.from(0, 0)
        // A line comes while the first is being read: the reading goes on where it was.
        pace.advance(reading(perRow / 2), length)
        length += 1 + 2 * perRow - 1
        assertEquals("still on the first line", 0, pace.top(rows(length), 2, length))
        pace.advance(reading(3 * perRow) + 1, length)
        assertEquals("the new line's first row", 3, pace.top(rows(length), 2, length))
        // All read; a line coming a minute later is read from when it comes, not skipped.
        val caughtUp = reading(length) + 1
        pace.advance(caughtUp, length)
        assertTrue(pace.done(length))
        val later = caughtUp + 60_000
        pace.advance(later, length)
        length += 1 + 4 * perRow - 1
        pace.advance(later, length)
        assertEquals("the last row read, then the new line", 4, pace.top(rows(length), 2, length))
        pace.advance(later + reading(2) + 1, length) // the newline, then the first character
        assertEquals(5, pace.top(rows(length), 2, length))
    }

    @Test
    fun theOldestWordsCutKeepTheReadersPlace() {
        val pace = CaptionPace()
        pace.from(1190, 0)
        pace.shift(52)
        assertEquals(1138.0, pace.read, 0.0)
        pace.shift(5000)
        assertEquals(0.0, pace.read, 0.0)
    }
}
