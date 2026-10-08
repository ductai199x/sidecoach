package com.sidecoach.voice

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The gate asked for before Setup fetched its file loads once the file is there, without a
 * restart, as the recogniser does; any other failure stands. The desktop's
 * `test_voice_mode_comes_once_setup_has_the_silero_file` (audit 2026-10-07, S1).
 */
@RunWith(AndroidJUnit4::class)
class SileroGateTest {

    private val app = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun theGateLoadsOnceSetupHasItsFile() {
        val pushed = File(app.filesDir, "models/silero/${SileroGate.FILE}")
        assumeTrue("no Silero in ${pushed.parent} (tools/test.ps1 pushes it)", pushed.exists())
        val dir = File(app.cacheDir, "silero-gate-test").apply {
            deleteRecursively()
            mkdirs()
        }
        val file = File(dir, SileroGate.FILE)
        val gate = SileroGate(file)
        assertFalse("no file yet", gate.load())
        pushed.copyTo(file) // Setup fetched it
        assertTrue("the file came: loaded without a restart", gate.load())
        assertEquals(true to "silero vad, cpu", gate.healthcheck())
        gate.close()

        val broken = File(dir, "broken.onnx").apply { writeText("not a model") }
        val bad = SileroGate(broken)
        assertFalse(bad.load())
        broken.delete()
        pushed.copyTo(broken)
        assertFalse("a failure that is not a missing file stands", bad.load())
        dir.deleteRecursively()
    }
}
