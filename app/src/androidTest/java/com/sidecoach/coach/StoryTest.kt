package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.parseSquare
import com.sidecoach.events.Analysis
import com.sidecoach.events.Line
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The hints tell one story ([Story], desktop 10fdd21/a7d4ada): `tests/test_story.py`, case for
 * case -- six Lichess puzzles read by hand, with the engine's lines as the app's settings saw
 * them. Each asserts what the test asserts, and that the three hints are the desktop's word for
 * word, as recorded with the desktop's story checks (`tools/puzzles/story_checks.py`, desktop
 * tooling), which all hold on each. Then the traps made by hand: their ranking ([Situation.
 * rankedTraps]: on the target, into mate, the worst) and their words with and without the best
 * move -- the wrong piece to take back with, the balance lost.
 */
@RunWith(AndroidJUnit4::class)
class StoryTest {

    private val data: JSONObject by lazy {
        val context = InstrumentationRegistry.getInstrumentation().context
        JSONObject(context.assets.open("chess/coach.json").bufferedReader().use { it.readText() })
    }

    private fun strings(a: JSONArray): List<String> = (0 until a.length()).map { a.getString(it) }

    private fun lines(a: JSONArray): List<Line> = (0 until a.length()).map { k ->
        val l = a.getJSONObject(k)
        Line(
            moves = strings(l.getJSONArray("moves")),
            depth = l.getInt("depth"),
            expectedScore = l.getDouble("expected_score"),
            scoreCp = if (l.isNull("score_cp")) null else l.getInt("score_cp"),
            mate = if (l.isNull("mate")) null else l.getInt("mate"),
        )
    }

    /** The hints the app gives for a puzzle of `tests/test_story.py`, and the desktop's. */
    private fun hints(name: String): Pair<Map<String, String>, JSONObject> {
        val stories = data.getJSONArray("stories")
        for (i in 0 until stories.length()) {
            val story = stories.getJSONObject(i)
            if (!story.getString("name").startsWith(name)) continue
            val fen = story.getString("fen")
            val board = Board(fen)
            val o = story.getJSONObject("analysis")
            val analysis = Analysis(
                gameId = "",
                fen = fen,
                ply = 0,
                lines = lines(o.getJSONArray("lines")),
                decisive = false,
                onlyMove = false,
                threats = strings(o.getJSONArray("threats")),
            )
            val facts = factsFromAnalysis(board, analysis, user = board.turn)
            val texts = mapOf(
                "idea" to ideaTemplate(facts),
                "focus" to focusTemplate(facts),
                "move" to moveTemplate(facts),
            )
            val theirs = story.getJSONObject("texts")
            for (rung in listOf("idea", "focus", "move")) {
                assertEquals("$name $rung, as the desktop says it", theirs.getString(rung), texts[rung])
            }
            assertTrue("$name: every story check holds on the desktop ${story.getJSONObject("checks")}", story.getBoolean("told"))
            return texts to story
        }
        throw AssertionError("no story test $name in the coach fixture")
    }

    @Test
    fun aQueenThatSimplyHangsIsSaidPlainly() {
        // gNzJg: a pawn takes the queen -- no discovered attack about it.
        val (texts, _) = hints("gNzJg")
        assertTrue(texts.getValue("idea"), "Their queen on g4 can simply be taken: nothing defends it" in texts.getValue("idea"))
        assertTrue("discovered" !in texts.getValue("idea"))
        assertTrue(texts.getValue("move"), texts.getValue("move").startsWith("fxg4 takes their queen"))
    }

    @Test
    fun aFreePieceFirstThenWhatTheLineWinsAfterIt() {
        // DWvS7: Qxe8+ takes a free rook, and the line wins more after it.
        val (texts, _) = hints("DWvS7")
        assertTrue(texts.getValue("idea"), "Their rook on e8 can simply be taken" in texts.getValue("idea"))
    }

    @Test
    fun inCheckTheCheckerComesFirst() {
        // EcCX4: their queen gives check, and taking it is the point.
        val (texts, _) = hints("EcCX4")
        assertTrue(texts.getValue("idea"), "Their queen on c5 gives check" in texts.getValue("idea"))
        assertTrue(texts.getValue("idea"), "The first move takes their queen on c5" in texts.getValue("idea"))
    }

    @Test
    fun thePawnTheWinRestsOnIsNamed() {
        // uSAIW: f7, and the rook must give itself up for the pawn.
        val (texts, _) = hints("uSAIW")
        assertTrue(texts.getValue("idea"), "Your pawn on f6 is two squares from queening" in texts.getValue("idea"))
    }

    @Test
    fun aWinWithNoPatternSaysHow() {
        // jlCZP, a zugzwang: never "their pawn on h5 is the target" and "you win two pawns" alone.
        val (texts, _) = hints("jlCZP")
        assertTrue(texts.getValue("idea"), "The first move stops Ke5" in texts.getValue("idea"))
        assertTrue(texts.getValue("move"), texts.getValue("move").startsWith("Kf4") && "and after" in texts.getValue("move"))
    }

    @Test
    fun whereOnlyOneMoveHoldsTheMoveSaysWhatItDoes() {
        // TzhLp: "The move is gxh5." said nothing.
        val (texts, _) = hints("TzhLp")
        assertTrue(texts.getValue("move"), "takes their pawn on h5" in texts.getValue("move"))
        assertTrue("?" in texts.getValue("focus"))
    }

    @Test
    fun trapsAreRankedAndSaidAsTheDesktopSaysThem() {
        val tests = data.getJSONArray("trap_tests")
        var wrongPiece = 0
        for (i in 0 until tests.length()) {
            val t = tests.getJSONObject(i)
            val board = Board(t.getString("fen"))
            val best = Move.fromUci(t.getString("best"))
            val target = if (t.isNull("target")) null else parseSquare(t.getString("target"))
            val top = t.getDouble("top")
            val ranked = Situation.rankedTraps(lines(t.getJSONArray("traps")), target)
            val wanted = t.getJSONArray("ranked")
            assertEquals(
                "${t.getString("name")} ranking",
                (0 until wanted.length()).map { strings(wanted.getJSONArray(it)) },
                ranked.map { it.moves },
            )
            val words = t.getJSONArray("words")
            val bare = t.getJSONArray("words_without_best")
            for ((k, trap) in ranked.withIndex()) {
                val said = Situation.trapWords(board, trap, target, top, best)
                assertEquals("${t.getString("name")} trap $k", if (words.isNull(k)) null else words.getString(k), said)
                assertEquals(
                    "${t.getString("name")} trap $k without the best move",
                    if (bare.isNull(k)) null else bare.getString(k),
                    Situation.trapWords(board, trap, target, top),
                )
                if (said != null && said.startsWith("Take it with the right piece")) wrongPiece += 1
            }
        }
        assertTrue("the wrong piece to take back with is said", wrongPiece > 0)
    }
}
