package com.sidecoach.coach

import androidx.test.ext.junit.runners.AndroidJUnit4
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
 * tooling), which all hold on each; desktop a1f7f81's two, why the obvious defence fails
 * ([Story.defence]), with the declined recaptures and the traps the test gives; and desktop
 * 29ae423's three, a quiet move told by what it threatens ([Story.aimHow]). Then the traps
 * made by hand: their ranking ([Situation.rankedTraps]: on the target, into mate, the worst) and
 * their words with and without the best move -- the wrong piece to take back with, the balance
 * lost.
 */
@RunWith(AndroidJUnit4::class)
class StoryTest {

    private val data: JSONObject get() = stories

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
                decisive = o.getBoolean("decisive"),
                onlyMove = o.getBoolean("only_move"),
                threats = strings(o.getJSONArray("threats")),
                traps = lines(o.getJSONArray("traps")),
                defences = lines(o.getJSONArray("defences")),
                threatens = lines(o.getJSONArray("threatens")),
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
        assertTrue(texts.getValue("idea"), "the first move takes their queen on c5" in texts.getValue("idea"))
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
        assertTrue(
            texts.getValue("idea"),
            "Their pawn on h5 is the target, and the first move stops Ke5." in texts.getValue("idea"),
        )
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
    fun whyTheObviousRecaptureFailsIsTheLesson() {
        // Tai's chess.com puzzle, 2026-09-24: 1...Qxg3 and White does not take back -- 2.fxg3
        // Bc5+ loses more. "You can win their queen" alone read as a lie beside fxg3.
        val (texts, _) = hints("Qxg3")
        val idea = texts.getValue("idea")
        // Said as a coach speaks (Tai, 2026-09-25): a tactic to find, the target and the defender
        // that cannot take back as one sentence, the pattern said of it.
        assertTrue(idea, "tactic that wins their queen." in idea)
        assertTrue(
            idea,
            "Their queen on g3 is the target, and their pawn on f2 can't take back there: it also " +
                "shields their king. That's an overloaded defender: one piece with two jobs can't do " +
                "both." in idea,
        )
        assertTrue(texts.getValue("focus"), "Which of their pieces has two jobs?" in texts.getValue("focus"))
        assertTrue(texts.getValue("move"), "if fxg3, Bc5+" in texts.getValue("move"))
    }

    @Test
    fun aDefenderWithTwoJobsIsNotALoosePiece() {
        // Tai's chess.com puzzle, 2026-09-24: 2.exf7+ Kh8 3.Qxb8 wins the rook because the rook
        // on f8 must stop f8=Q; "a loose piece: nothing defends it" was the wrong lesson, and
        // 2.Qxb8? at once (Qd2+) was the natural mistake, unwarned.
        val (texts, _) = hints("Qxb8")
        val idea = texts.getValue("idea")
        assertTrue(
            idea,
            "Your pawn on e6 is two squares from queening, and their rook on f8 has to stop it, so it " +
                "can't also take back on b8." in idea,
        )
        assertTrue(idea, "loose" !in idea && "no defender" !in idea)
        assertTrue(idea, "Their rook on b8 can be won, but not at once: Qxb8 right away runs into Qd2+" in idea)
        assertTrue(texts.getValue("move"), "if Rxb8, f8=Q+" in texts.getValue("move"))
    }

    @Test
    fun aQuietMoveIsToldByWhatItThreatens() {
        // The 28 wins of the puzzle sample whose first hint was "their X is the target" and no
        // more (2026-09-25): a quiet first move's point is its threat, asked of the engine
        // ([com.sidecoach.analysis.aimLines]). Lichess 06VoP: 32.Qg4, threatening Qxg7#.
        val (texts, _) = hints("06VoP")
        assertTrue(texts.getValue("idea"), "The first move is quiet, but it threatens mate." in texts.getValue("idea"))
        assertTrue(texts.getValue("focus"), "Which quiet move sets up a mate?" in texts.getValue("focus"))
    }

    @Test
    fun aQuietMoveThatThreatensAPieceNamesItWhereItStands() {
        // Lichess 05cb6: in check, 35.Kd2 steps out and threatens Rxb1, their queen.
        val (texts, _) = hints("05cb6")
        assertTrue(
            texts.getValue("idea"),
            "The first move is quiet, but it threatens to take their queen on b1." in texts.getValue("idea"),
        )
    }

    @Test
    fun aQuietPawnMoveThreatensABreak() {
        // Lichess 00pWF, a king and pawn ending: 53...a5, threatening a4 and a passed pawn.
        val (texts, _) = hints("00pWF")
        assertTrue(texts.getValue("idea"), "The first move is quiet, but it threatens a pawn break." in texts.getValue("idea"))
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
                if (said != null && said.startsWith(Situation.RIGHT_PIECE)) wrongPiece += 1
            }
        }
        assertTrue("the wrong piece to take back with is said", wrongPiece > 0)
    }

    @Test
    fun aPieceWithNoSafeSquareIsTaughtAsTrapped() {
        // Desktop 6bea675 (`tests/test_story.py`): Tai's chess.com puzzle, Bg4 against a queen on
        // h5 with nowhere to go; the hint said only "the first move attacks their queen on h5".
        val fen = "r1bq1k1r/1p2bp2/p2p3p/2p1n2Q/P3P3/1BN5/1PP2PPP/R1B2RK1 b - - 1 15"
        val board = Board(fen)
        val analysis = Analysis(
            gameId = "",
            fen = fen,
            ply = 0,
            lines = listOf(
                Line(moves = listOf("c8g4", "b3f7", "g4h5", "f7h5", "b7b5", "c3d5", "b5b4", "f2f4"), depth = 18, expectedScore = 0.797),
                Line(moves = listOf("c5c4", "h2h3", "c4b3", "c2b3", "e7e6", "a1a3", "a8c8", "c1e3"), depth = 18, expectedScore = 0.63),
            ),
            decisive = false,
            onlyMove = false,
        )
        val facts = factsFromAnalysis(board, analysis, user = board.turn)
        assertEquals(
            "This position has a tactic that wins a queen for a bishop and a pawn. Their queen on h5 is the " +
                "target. The first move attacks it, and it has no safe square to go to. That's a trapped piece: " +
                "one of their pieces has no safe square left, so an attack on it wins it.",
            ideaTemplate(facts),
        )
        assertEquals("Use your bishop on c8. Which of their pieces has no safe square left?", focusTemplate(facts))
        assertEquals("Bg4, and after Bxf7, Bxh5; you win a queen for a bishop and a pawn.", moveTemplate(facts))
    }

    private companion object {
        /**
         * The stories and the trap tests of coach.json, read once for the class: its 29 MB of
         * positions are skipped (each test read the whole file into a tree, thirteen a run).
         */
        val stories: JSONObject by lazy { fixture("chess/coach.json", setOf("stories", "trap_tests")) }
    }

    /**
     * `test_covers_only_when_the_moved_piece_guards_it` and
     * `test_attacks_only_what_the_moved_piece_can_take` (the facts audit, 2026-10-01).
     */
    @Test
    fun coversAndAttacksOnlyWhatTheMovedPieceDoes() {
        var board = Board("2r4r/4kpp1/p3p2p/7n/Bq3P2/2N2Q1P/1PP3P1/4R1K1 w - - 2 28")
        var said = Story.effects(board, board.parseSan("Ne4")!!, null).map { it.said }
        assertTrue(said.toString(), "makes your pawn on f4 safe" in said && said.none { it.startsWith("covers") })
        said = Story.effects(board, board.parseSan("g3")!!, null).map { it.said }
        assertTrue(said.toString(), "covers your pawn on f4" in said)
        // Kf7 next to a rook on e8 that the queen on e5 guards: the king cannot take it.
        board = Board("1r2R1k1/p5p1/2p4p/2PpQ3/2nP4/B3PP1P/P5PK/4q3 b - - 10 34")
        said = Story.effects(board, board.parseSan("Kf7")!!, null).map { it.said }
        assertTrue(said.toString(), said.none { it.startsWith("attacks") })
    }
}
