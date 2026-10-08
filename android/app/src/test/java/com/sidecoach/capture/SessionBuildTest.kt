package com.sidecoach.capture

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertSame
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * A session whose stages cannot be built ([CaptureService.builtOrSaid]): the failure is said
 * and the session stopped, and nothing is thrown on the vision thread, where it took the whole
 * app down (the audit of 2026-10-07: "session build failure crashes").
 */
class SessionBuildTest {

    @Test
    fun aFailedBuildIsSaidNotThrown() {
        val said = ArrayList<Throwable>()
        val boom = IllegalStateException("no engine")
        val built: String? = CaptureService.builtOrSaid(build = { throw boom }, failed = { said += it })
        assertNull(built)
        assertEquals(1, said.size)
        assertSame(boom, said.single())
    }

    @Test
    fun aNativeLibraryMissingIsSaidToo() {
        // The phone's own way to fail at a build: a library that does not load (0.1.80).
        val said = ArrayList<Throwable>()
        val built: String? = CaptureService.builtOrSaid(
            build = { throw UnsatisfiedLinkError("libsidecoach_espeak.so") },
            failed = { said += it },
        )
        assertNull(built)
        assertTrue(said.single() is UnsatisfiedLinkError)
    }

    @Test
    fun aBuiltSessionIsKeptAndNothingIsSaid() {
        val said = ArrayList<Throwable>()
        assertEquals("watcher", CaptureService.builtOrSaid(build = { "watcher" }, failed = { said += it }))
        assertTrue(said.isEmpty())
    }
}
