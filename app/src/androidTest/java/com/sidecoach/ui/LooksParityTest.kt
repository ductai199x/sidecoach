package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.analysis.GameReviewParityTest
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Base64

/**
 * How the boards look, as the desktop's (ui/Looks.kt): `tools/make_looks_fixtures.py` records
 * `looks.py`'s catalogue, the look for every set and theme, which piece a file's name says, an
 * image's type from its bytes, and a player's own pieces saved over three rounds, read back and
 * cleared -- all replayed here in a folder of the test's own.
 */
@RunWith(AndroidJUnit4::class)
class LooksParityTest {
    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    @Test
    fun theLookIsTheDesktops() {
        val fixture = JSONObject(instrumentation.context.assets.open("ui/looks.json").bufferedReader().use { it.readText() })
        val files = File(instrumentation.targetContext.cacheDir, "looks-test").apply {
            deleteRecursively()
            mkdirs()
        }
        val failures = ArrayList<String>()
        GameReviewParityTest.same("catalogue", Looks.catalogue(files), fixture.getJSONObject("catalogue"), failures)
        val looks = fixture.getJSONArray("looks")
        for (i in 0 until looks.length()) {
            val one = looks.getJSONObject(i)
            val pieces = one.getString("pieces")
            val theme = one.getString("theme")
            GameReviewParityTest.same("look $pieces $theme", Looks.look(pieces, theme, files), one.getJSONObject("look"), failures)
        }
        val codes = fixture.getJSONArray("codes")
        for (i in 0 until codes.length()) {
            val pair = codes.getJSONArray(i)
            val want = if (pair.isNull(1)) null else pair.getString(1)
            val got = Looks.pieceCode(pair.getString(0))
            if (got != want) failures.add("piece code of '${pair.getString(0)}': mine $got, python $want")
        }
        val sniff = fixture.getJSONArray("sniff")
        for (i in 0 until sniff.length()) {
            val one = sniff.getJSONArray(i)
            val want = if (one.isNull(2)) null else one.getString(2)
            val got = Looks.sniff(Base64.getDecoder().decode(one.getString(1)))
            if (got != want) failures.add("sniff ${one.getString(0)}: mine $got, python $want")
        }
        val saves = fixture.getJSONArray("saves")
        for (i in 0 until saves.length()) {
            val one = saves.getJSONObject(i)
            GameReviewParityTest.same("save $i", Looks.saveCustom(files, one.getJSONArray("files")), one.getJSONObject("result"), failures)
            GameReviewParityTest.same("save $i images", JSONObject(Looks.customImages(files) as Map<*, *>), one.getJSONObject("images"), failures)
            GameReviewParityTest.same("save $i look", Looks.look("custom", "green", files), one.getJSONObject("look"), failures)
            GameReviewParityTest.same("save $i catalogue", Looks.catalogue(files), one.getJSONObject("catalogue"), failures)
        }
        Looks.clearCustom(files)
        GameReviewParityTest.same("cleared", Looks.catalogue(files), fixture.getJSONObject("cleared"), failures)
        assertTrue("${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }
}
