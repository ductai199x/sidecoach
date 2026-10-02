package com.sidecoach.voice

import android.media.AudioTrack
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The voice's output following the phone's (Tai, 2026-10-02: the voice follows the output device
 * without a restart; the desktop's `test_the_voice_follows_the_systems_output_mid_line`). Android
 * moves a track that names no device itself; one it could not move answers writes with
 * ERROR_DEAD_OBJECT, and the rest of the line goes on a track opened afresh. Fake outputs: nothing
 * plays.
 */
@RunWith(AndroidJUnit4::class)
class VoiceOutputTest {

    /** An output that takes [alive] writes, then answers every one with ERROR_DEAD_OBJECT. */
    private class Output(private var alive: Int = Int.MAX_VALUE) : PcmSink {
        var taken = 0
        override fun write(samples: FloatArray, at: Int, n: Int): Int {
            if (alive <= 0) return AudioTrack.ERROR_DEAD_OBJECT
            alive -= 1
            taken += n
            return n
        }
    }

    @Test
    fun aLineWhoseOutputDiesMidWayGoesOnANewOne() {
        val piece = FloatArray(KokoroTts.WRITE_SLICE * 5 + 7)
        val old = Output(alive = 2) // the route changed after two slices, and the track was lost
        val new = Output()
        var opened = 0
        val wrote = KokoroTts.writePiece(piece, { old }, { opened += 1; new }, { false })
        assertEquals(piece.size, wrote)
        assertEquals(1, opened)
        assertEquals(KokoroTts.WRITE_SLICE * 2, old.taken)
        assertEquals(piece.size - KokoroTts.WRITE_SLICE * 2, new.taken)
    }

    @Test
    fun anOutputThatStaysDeadIsOpenedOnceAPieceAndAStopStillEndsIt() {
        val piece = FloatArray(KokoroTts.WRITE_SLICE * 3)
        var opened = 0
        assertEquals(0, KokoroTts.writePiece(piece, { Output(0) }, { opened += 1; Output(0) }, { false }))
        assertEquals("one new output a piece, never a loop", 1, opened)
        var slices = 0
        val live = Output()
        assertEquals(KokoroTts.WRITE_SLICE, KokoroTts.writePiece(piece, { live }, { Output() }, { slices++ >= 1 }))
    }
}
