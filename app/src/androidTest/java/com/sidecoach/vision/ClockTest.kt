package com.sidecoach.vision

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.state.clockObservation
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc

/**
 * The two chess.com clocks beside the board, held to the desktop's reading.
 *
 * The frames are the desktop's own synthetic renderer's, drawn with the same glyph templates
 * the reader classifies against and written out as PNGs, so both sides read the same pixels:
 * ten single frames over several cell sizes, polarities, page colours and box colours
 * (including two the desktop itself reads poorly, which the port must read just as poorly),
 * a sequence of eight where the running clock is decided from the digits falling and the
 * icon's hand sweeping and then goes stale, and a board with no clocks at all.
 *
 * Every box is compared in full -- where it was found, its background and text levels, each
 * glyph with its correlation and margin, the icon's rectangle -- so a failure says which
 * step of the reading drifted, not merely that the time came out wrong.
 */
@RunWith(AndroidJUnit4::class)
class ClockTest {

    private val failures = ArrayList<String>()

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine != theirs && failures.size < 10) {
            failures.add("$what:\n  kotlin: $mine\n  python: $theirs")
        }
    }

    private fun close(what: String, mine: Double, theirs: Double, tolerance: Double = 1e-6) {
        if (kotlin.math.abs(mine - theirs) > tolerance && failures.size < 10) {
            failures.add("$what:\n  kotlin: $mine\n  python: $theirs")
        }
    }

    @Test
    fun everyClockReadingMatchesThePython() {
        assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        val context = InstrumentationRegistry.getInstrumentation().context
        val app = InstrumentationRegistry.getInstrumentation().targetContext
        val templates = GlyphTemplates.load(app.assets)
        assertTrue("the shipped clock templates did not load", templates != null)
        val text = context.assets.open("chess/clock.json").bufferedReader().use { it.readText() }
        val root = JSONObject(text)
        val began = System.currentTimeMillis()

        val parses = root.getJSONArray("parse")
        for (i in 0 until parses.length()) {
            val one = parses.getJSONObject(i)
            val words = one.getString("text")
            check(
                "parseClock '$words'",
                parseClock(words),
                if (one.isNull("ms")) null else one.getInt("ms"),
            )
        }

        val cases = root.getJSONArray("cases")
        for (i in 0 until cases.length()) {
            val one = cases.getJSONObject(i)
            val label = one.getString("label")
            val frame = load(one.getString("frame"))
            try {
                val reader = ClockReader(templates)
                val reading = reader.read(frame, boardOf(one.getJSONObject("board")), one.getDouble("ts"))
                checkReading(label, reading, one.opt("reading"))
                val boxes = one.getJSONObject("boxes")
                checkBox("$label top box", reader.last[Which.TOP], boxes.opt("top"))
                checkBox("$label bottom box", reader.last[Which.BOTTOM], boxes.opt("bottom"))
                checkObservation(
                    "$label white at the bottom",
                    clockObservation(reading, Pieces.Side.WHITE),
                    one.opt("observation_white_bottom"),
                )
                checkObservation(
                    "$label black at the bottom",
                    clockObservation(reading, Pieces.Side.BLACK),
                    one.opt("observation_black_bottom"),
                )
            } finally {
                frame.release()
            }
        }

        // One reader across the sequence: the running clock is a fact about the frames
        // before this one.
        val reader = ClockReader(templates)
        val sequence = root.getJSONArray("sequence")
        for (i in 0 until sequence.length()) {
            val one = sequence.getJSONObject(i)
            val frame = load(one.getString("frame"))
            try {
                val reading = reader.read(frame, boardOf(one.getJSONObject("board")), one.getDouble("ts"))
                checkReading("sequence #$i", reading, one.opt("reading"))
                checkObservation(
                    "sequence #$i white at the bottom",
                    clockObservation(reading, Pieces.Side.WHITE),
                    one.opt("observation_white_bottom"),
                )
            } finally {
                frame.release()
            }
        }

        val bare = root.getJSONObject("bare")
        val bareFrame = load(bare.getString("frame"))
        try {
            checkReading(
                "a board with no clocks",
                ClockReader(templates).read(bareFrame, boardOf(bare.getJSONObject("board")), 1.0),
                bare.opt("reading"),
            )
        } finally {
            bareFrame.release()
        }

        val took = System.currentTimeMillis() - began
        Log.i(TAG, "${cases.length()} frames, ${sequence.length()} in sequence, $took ms")
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    private fun checkReading(what: String, mine: ClockReading?, theirs: Any?) {
        if (theirs == null || theirs === JSONObject.NULL) {
            check("$what read nothing", mine, null)
            return
        }
        val want = theirs as JSONObject
        if (mine == null) {
            check("$what read nothing", "null", want.toString())
            return
        }
        check("$what top", mine.topMs, if (want.isNull("top_ms")) null else want.getInt("top_ms"))
        check(
            "$what bottom",
            mine.bottomMs,
            if (want.isNull("bottom_ms")) null else want.getInt("bottom_ms"),
        )
        check(
            "$what running",
            mine.running?.slug,
            if (want.isNull("running")) null else want.getString("running"),
        )
        check(
            "$what light",
            mine.light?.slug,
            if (want.isNull("light")) null else want.getString("light"),
        )
        close("$what confidence", mine.confidence, want.getDouble("confidence"))
    }

    private fun checkBox(what: String, mine: BoxReading?, theirs: Any?) {
        if (theirs == null || theirs === JSONObject.NULL) {
            check("$what found nothing", mine?.text, null)
            return
        }
        val want = theirs as JSONObject
        if (mine == null) {
            check("$what found nothing", "null", want.getString("text"))
            return
        }
        check("$what text", mine.text, want.getString("text"))
        check("$what ms", mine.ms, if (want.isNull("ms")) null else want.getInt("ms"))
        close("$what confidence", mine.confidence, want.getDouble("confidence"))
        val box = want.getJSONObject("box")
        check("$what top", mine.box.top, box.getInt("top"))
        check("$what bottom", mine.box.bottom, box.getInt("bottom"))
        check("$what left", mine.box.left, box.getInt("left"))
        check("$what right", mine.box.right, box.getInt("right"))
        close("$what background", mine.box.background, box.getDouble("background"))
        close("$what text level", mine.box.text, box.getDouble("text"))
        check("$what light", mine.box.light, box.getBoolean("light"))
        val glyphs = want.getJSONArray("glyphs")
        check("$what glyph count", mine.glyphs.size, glyphs.length())
        for (k in mine.glyphs.indices) {
            if (k >= glyphs.length()) break
            val one = glyphs.getJSONArray(k)
            check("$what glyph $k", mine.glyphs[k].first, one.getString(0))
            close("$what glyph $k score", mine.glyphs[k].second, one.getDouble(1), 1e-5)
            close("$what glyph $k margin", mine.glyphs[k].third, one.getDouble(2), 1e-5)
        }
        val icon = want.opt("icon")
        if (icon == null || icon === JSONObject.NULL) {
            check("$what icon", mine.icon, null)
        } else {
            val want4 = icon as JSONArray
            check(
                "$what icon",
                listOf(mine.icon?.x, mine.icon?.y, mine.icon?.w, mine.icon?.h),
                listOf(want4.getInt(0), want4.getInt(1), want4.getInt(2), want4.getInt(3)),
            )
        }
    }

    private fun checkObservation(what: String, mine: Any?, theirs: Any?) {
        val observation = mine as? com.sidecoach.events.ClockObservation
        if (theirs == null || theirs === JSONObject.NULL) {
            check("$what observed nothing", observation?.let { "some" }, null)
            return
        }
        val want = theirs as JSONObject
        if (observation == null) {
            check("$what observed nothing", "null", want.toString())
            return
        }
        check(
            "$what white",
            observation.whiteMs,
            if (want.isNull("white_ms")) null else want.getInt("white_ms"),
        )
        check(
            "$what black",
            observation.blackMs,
            if (want.isNull("black_ms")) null else want.getInt("black_ms"),
        )
        check(
            "$what running",
            observation.running?.name?.lowercase(),
            if (want.isNull("running")) null else want.getString("running"),
        )
    }

    private fun boardOf(o: JSONObject): BoardCandidate = BoardCandidate(
        region = Region(o.getInt("x"), o.getInt("y"), o.getInt("w"), o.getInt("h")),
        cellPx = o.getDouble("cell"),
        score = 1.0,
        contrast = 60.0,
        fit = 1.0,
    )

    private fun load(file: String): Mat {
        val context = InstrumentationRegistry.getInstrumentation().context
        val bitmap = context.assets.open("clocks/$file").use {
            android.graphics.BitmapFactory.decodeStream(it)
        }
        val rgba = Mat()
        Utils.bitmapToMat(bitmap, rgba)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        bitmap.recycle()
        return bgr
    }

    companion object {
        private const val TAG = "sidecoach.vision"
    }
}
