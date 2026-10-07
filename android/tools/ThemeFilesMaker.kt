package com.sidecoach.vision

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import java.io.File
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.android.OpenCVLoader
import org.opencv.core.Core

/**
 * Writes the theme files `ThemeMigrationTest` reads: one learned set, written by the writer of
 * the version it is made in. Not a test of this checkout, and not compiled with it: copy it into
 * `app/src/androidTest/java/com/sidecoach/vision/` of a checkout of the version wanted (`git
 * archive <commit> | tar -x` into a scratch folder, with this checkout's `local.properties`),
 * build that checkout's two APKs, install them on the emulator and run this class alone:
 *
 *     adb -s emulator-5554 shell am instrument -w -e class com.sidecoach.vision.ThemeFilesMaker \
 *         com.sidecoach.test/androidx.test.runner.AndroidJUnitRunner
 *     adb -s emulator-5554 pull /sdcard/Android/data/com.sidecoach/files/theme-v<N>.bin
 *     adb -s emulator-5554 pull /sdcard/Android/data/com.sidecoach/files/theme-v<N>.json
 *
 * and put both files in `app/src/androidTest/assets/vision/themes/`. The versions so far came
 * from 2857f14 (App 0.1.18: version 4), 094a093 (5) and bc23731 (App 0.1.19: 6).
 *
 * The set is learned from the boards fixture `start-green`, OpenCV's generator seeded first,
 * so its k-means palette is the same in every version. Beside the file, what that version
 * made of it: the landing crops' pixel sums by the key that version kept them under, and what
 * it read on `italian-green`.
 */
@RunWith(AndroidJUnit4::class)
class ThemeFilesMaker {

    @Test
    fun writeTheLearnedSetAtThisVersion() {
        assertTrue("OpenCV did not load", OpenCVLoader.initLocal())
        val context = InstrumentationRegistry.getInstrumentation().targetContext
        val fixtures = Fixtures.all().associateBy { it.name }
        val start = fixtures.getValue("start-green")
        val italian = fixtures.getValue("italian-green")
        val geometry = BoardGeometry(Region(start.x, start.y, start.side, start.side), start.cell.toDouble())
        val directory = File(context.cacheDir, "theme-files-maker").apply { deleteRecursively(); mkdirs() }
        val out = context.getExternalFilesDir(null)!!
        val version = ThemeCache.CACHE_VERSION
        CellEmbedder.load(context).use { embedder ->
            val startFrame = Fixtures.mat(start)
            Core.setRNGSeed(SEED)
            val model = ThemeModel.fromStartPosition(startFrame, geometry, true, embedder)!!
            startFrame.release()
            val key = ThemeStore(directory).save(model)!!
            File(directory, "theme-$key.bin").copyTo(File(out, "theme-v$version.bin"), overwrite = true)
            val board = Fixtures.mat(italian)
            val reads = model.classifyFrame(board, geometry, embedder).map { it.symbol ?: "" }
            board.release()
            val crops = JSONObject()
            for ((kind, crop) in model.landingCrops) {
                crops.put(kind.toString(), crop.pixels.sumOf { (it.toInt() and 0xFF).toLong() })
            }
            val made = JSONObject()
                .put("version", version)
                .put("key", key)
                .put("seed", SEED)
                .put("light_bg", JSONArray(model.lightBg.toList()))
                .put("dark_bg", JSONArray(model.darkBg.toList()))
                .put("landing_px", model.landingPx)
                .put("crops", crops)
                .put("reads", JSONArray(reads))
            File(out, "theme-v$version.json").writeText(made.toString(1) + "\n")
        }
        directory.deleteRecursively()
    }

    private companion object {
        const val SEED = 20260927
    }
}
