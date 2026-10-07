package com.sidecoach.analysis

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
import com.sidecoach.events.Label
import com.sidecoach.events.ReviewJson
import com.sidecoach.profile.Library
import com.sidecoach.profile.Store
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * A game reviewed on the phone with real engines side by side (analysis/GameReview.kt), stopped
 * part way and then resumed from the positions it kept (Store.addProgress), as a review the page
 * left or the app closed in goes on from where it was: the second run searches only what the
 * first did not, and the review is whole. Skipped without the engine's network
 * (tools/push_net.ps1).
 */
@Lane(Lane.Kind.FREE)
@RunWith(AndroidJUnit4::class)
class GameReviewLiveTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    private val pgn = "[White \"Someone\"]\n[Black \"nosegazm\"]\n[Result \"1-0\"]\n[TimeControl \"180\"]\n\n" +
        "1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0\n"

    private fun engines(n: Int, multipv: Int): List<Engine>? {
        val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
        val net = context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull()
        if (!binary.exists() || net == null) return null
        return List(n) { Engine(binary, net, threads = GameReview.THREADS, hashMb = 64, multipv = multipv).also { it.open() } }
    }

    @Test
    fun aReviewStoppedPartWayGoesOnFromWhereItWas() {
        val strength = GameReview.STRENGTHS.getValue("fast")
        // The engine and its network there: looked for, not started (each start loads the
        // network; the runs below start their own).
        val binary = Engine.binaryIn(context.applicationInfo.nativeLibraryDir)
        val net = context.filesDir.listFiles { f: File -> f.name.endsWith(".nnue") }?.firstOrNull()
        assumeTrue("the engine or its network is missing", binary.exists() && net != null)
        val file = File(context.cacheDir, "review-live.sqlite").apply { delete() }
        val store = Store(file)
        val row = Library.fromPgn(pgn, listOf("nosegazm")).single()
        val game = Library.gamesIn(row.pgn).single()
        val boards = GameReview.positions(game).size
        fun keep(m: Measured) =
            store.addProgress(row.key, strength.name, strength.depth, strength.multipv, m.fen, ReviewJson.measured(m).toString())

        // The first run, stopped once three positions are measured.
        val stop = AtomicBoolean(false)
        var engines = engines(GameReview.workersFor(), strength.multipv)!!
        val first = try {
            GameReview.reviewGame(
                game, engines, strength.depth, strength.multipv, Thresholds(), tagger = Motifs,
                strength = strength.name, gameId = row.key, stop = stop, onMeasured = ::keep,
                onProgress = { done, _ -> if (done >= 3) stop.set(true) },
            )
        } finally {
            engines.forEach { it.close() }
        }
        assertTrue("stopped before the walk", first.moves.isEmpty())
        val kept = store.progress(row.key, strength.name, strength.depth, strength.multipv)
        assertTrue("kept ${kept.size} of $boards positions", kept.size in 3 until boards)

        // The second run: the kept positions known, the rest searched, the review whole.
        val known = kept.map { ReviewJson.measured(JSONObject(it)) }.associateBy { it.fen }
        val searched = ArrayList<String>()
        var firstReport: Int? = null
        engines = engines(GameReview.workersFor(), strength.multipv)!!
        val second = try {
            GameReview.reviewGame(
                game, engines, strength.depth, strength.multipv, Thresholds(), tagger = Motifs,
                strength = strength.name, gameId = row.key, known = known,
                onMeasured = { synchronized(searched) { searched.add(it.fen) } },
                onProgress = { done, _ -> if (firstReport == null) firstReport = done },
            )
        } finally {
            engines.forEach { it.close() }
        }
        Log.i(TAG, "resumed: ${known.size} known, ${searched.size} searched, ${GameReview.workersFor()} engines")
        store.close()
        assertEquals("the kept positions counted at once", known.size, firstReport)
        assertTrue("nothing kept was searched again", searched.none { it in known })
        assertEquals("every position measured", boards, known.size + searched.size)
        assertEquals("every move reviewed", 7, second.moves.size)
        assertEquals("the mate is the best move", Label.BEST, second.moves.last().assessment.label)
        assertTrue("every move labelled", second.moves.all { it.assessment.label != null })
    }

    companion object {
        private const val TAG = "sidecoach.test"
    }
}
