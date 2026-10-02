package com.sidecoach.capture

import android.graphics.Bitmap
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertSame
import org.junit.Test
import org.junit.runner.RunWith
import java.nio.ByteBuffer

/**
 * A plane whose last row ends at the image's width, short of the row stride: the bitmap copy
 * threw on every frame on such devices (round 2, finding 11). [CaptureService.fullRows] gives
 * the copy a whole block, the image's pixels where they were.
 */
@RunWith(AndroidJUnit4::class)
class FullRowsTest {

    @Test
    fun aPlaneWithAShortLastRowIsCopiedWhole() {
        val width = 5
        val stride = 8 * 4 // eight pixels a row, five of them the image's
        val height = 3
        val short = ByteBuffer.allocateDirect(stride * (height - 1) + width * 4)
        for (i in 0 until short.capacity()) short.put(i, byteAt(i))
        val bitmap = Bitmap.createBitmap(stride / 4, height, Bitmap.Config.ARGB_8888)
        val whole = CaptureService.fullRows(short, stride, height, null)
        bitmap.copyPixelsFromBuffer(whole) // threw with the short plane itself
        val back = ByteBuffer.allocate(stride * height)
        bitmap.copyPixelsToBuffer(back)
        for (y in 0 until height) {
            for (x in 0 until width * 4) {
                val at = y * stride + x
                assertEquals("byte $x of row $y", byteAt(at), back.get(at))
            }
        }
        // A scratch large enough is reused; a whole plane is passed through untouched.
        short.rewind()
        assertSame(whole, CaptureService.fullRows(short, stride, height, whole))
        val full = ByteBuffer.allocateDirect(stride * height)
        assertSame(full, CaptureService.fullRows(full, stride, height, null))
    }

    /** A pixel's bytes: opaque (no premultiplying to change them), the colours a pattern. */
    private fun byteAt(at: Int): Byte = if (at % 4 == 3) 0xFF.toByte() else (at % 251).toByte()
}
