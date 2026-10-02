package com.sidecoach.ui

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
import com.sidecoach.analysis.GameReviewParity
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.Base64

/**
 * How the boards look, as the desktop's (ui/Looks.kt): `tools/make_looks_fixtures.py` records
 * `looks.py`'s catalogue, the look for every set and theme, which piece a file's name says, an
 * image's type from its bytes, and a player's own sets step by step -- the one set from before
 * taken in, sets made, added to, renamed, deleted, put back and gone past their Undo -- all
 * replayed here in a folder of the test's own.
 */
@Lane(Lane.Kind.FREE)
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
        GameReviewParity.same("catalogue", Looks.catalogue(files), fixture.getJSONObject("catalogue"), failures)
        val looks = fixture.getJSONArray("looks")
        for (i in 0 until looks.length()) {
            val one = looks.getJSONObject(i)
            val pieces = one.getString("pieces")
            val theme = one.getString("theme")
            GameReviewParity.same("look $pieces $theme", Looks.look(pieces, theme, files), one.getJSONObject("look"), failures)
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
        val steps = fixture.getJSONArray("steps")
        for (i in 0 until steps.length()) {
            val one = steps.getJSONObject(i)
            val op = one.getString("op")
            val id = one.optString("id")
            val what = "step $i $op $id"
            val got: Any? = when (op) {
                "legacy" -> {
                    val old = File(File(files, "pieces"), "custom").apply { mkdirs() }
                    val given = one.getJSONArray("files")
                    for (k in 0 until given.length()) {
                        val f = given.getJSONObject(k)
                        File(old, f.getString("name")).writeBytes(Base64.getDecoder().decode(f.getString("data")))
                    }
                    null
                }
                "expire" -> {
                    val aside = File(File(Looks.ownRoot(files), Looks.REMOVED), id)
                    aside.setLastModified(aside.lastModified() - (Looks.UNDO_S + 5) * 1000L)
                    null
                }
                "sets" -> JSONArray(
                    Looks.ownSets(files).map { JSONObject().put("id", it.id).put("name", it.name).put("order", it.order) },
                )
                "look" -> Looks.look(one.getString("pieces"), one.getString("theme"), files)
                "catalogue" -> Looks.catalogue(files)
                "create" -> Looks.createOwn(files, one.getString("name"), one.getJSONArray("files"))
                "add" -> Looks.addImages(files, id, one.getJSONArray("files"))
                "rename" -> Looks.renameOwn(files, id, one.getString("name"))
                "delete" -> Looks.deleteOwn(files, id)
                "restore" -> Looks.restoreOwn(files, id)
                else -> {
                    failures.add("$what: no such step")
                    null
                }
            }
            if (got is JSONObject) GameReviewParity.same(what, got, one.getJSONObject("result"), failures)
            if (got is JSONArray) GameReviewParity.same(what, got, one.getJSONArray("result"), failures)
        }
        assertTrue("${failures.size} failures:\n" + failures.take(30).joinToString("\n"), failures.isEmpty())
    }
}
