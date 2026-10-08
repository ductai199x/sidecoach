package com.sidecoach.profile

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.Lane
import com.sidecoach.chess.Board
import com.sidecoach.chess.WHITE
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.PuzzleOutcome
import com.sidecoach.vision.Pieces
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

/**
 * The puzzle store, `tests/test_puzzles.py`'s cases the profile fixture does not replay: the
 * import, the pick by motif and level (the window, the excluded, the due first, the mates, a
 * theme with none), the store opened from the CSV beside it once, an attempt solved, failed and
 * decided once, a failed puzzle on the screen stepping the level back. And the CSV read as
 * Python's `csv` module reads it (parity pv-06), from the profile fixture.
 */
@Lane(Lane.Kind.FREE)
@RunWith(AndroidJUnit4::class)
class PuzzlesTest {

    private val instrumentation = InstrumentationRegistry.getInstrumentation()

    private fun dir(name: String): File =
        File(instrumentation.targetContext.cacheDir, "puzzles-$name").apply { deleteRecursively(); mkdirs() }

    private fun csv(into: File): File =
        File(into, "sample.csv").apply { writeText(HEADER + "\n" + ROWS.joinToString("\n") + "\n") }

    @Test
    fun theStoreImportsTheCsvAndPicksByMotifAndLevel() {
        val store = PuzzleStore(null)
        val file = csv(dir("import"))
        assertEquals(6, store.importCsv(file))
        assertEquals("the same puzzles again add nothing", 0, store.importCsv(file))
        val fork = store.get("fork1")!!
        assertEquals(listOf("f3e5"), fork.solverMoves)
        assertTrue(fork.url.endsWith("/fork1"))
        // The solver's position is after the opponent's first move: what the screen shows.
        val board = Board(fork.fen)
        board.push(com.sidecoach.chess.Move.fromUci("g8f6"))
        assertEquals(Puzzles.positionKey(board.fen()), fork.position())
        assertEquals(listOf("fork1"), store.atPosition(fork.position()).map { it.id })
        // The motif's themes within the window of the level; the excluded out; the due first.
        assertEquals("000Pw is 1547, hi999 is 1900", setOf("000rO", "fork1"), store.pick("fork", 1100, count = 3).map { it.id }.toSet())
        assertEquals(listOf("000rO"), store.pick("fork", 1100, exclude = setOf("fork1")).map { it.id })
        assertEquals("fork1", store.pick("fork", 1100, first = listOf("fork1"), count = 1)[0].id)
        assertEquals("mate means mateIn1/2/3", listOf("mate1"), store.pick("mate", 1000).map { it.id })
        assertEquals(emptyList<Puzzle>(), store.pick("skewer", 1100))
        store.close()
    }

    @Test
    fun theStoreIsFilledFromTheCsvBesideItOnce() {
        val files = dir("open")
        assertNull("no puzzles anywhere", Puzzles.openStore(files))
        csv(File(files, "puzzles").apply { mkdirs() })
        val store = Puzzles.openStore(files)!!
        assertEquals(6, store.count())
        store.close()
        val again = Puzzles.openStore(files)!!
        assertEquals(6, again.count())
        assertEquals(File(files, "puzzles.sqlite"), again.path)
        again.close()
    }

    @Test
    fun anAttemptIsSolvedByTheLineFailedByAWrongMoveAndDecidedOnce() {
        val puzzle = Puzzle(
            "000Pw", "6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - - 2 37",
            listOf("e4d2", "d4e2", "g1f1", "e2c3"), 1547, 76, 629, listOf("fork"),
        )
        val a = Attempt(puzzle, "fork")
        assertNull("one of two solver moves: not yet", a.move("d4e2", byUser = true))
        assertNull("the site's reply", a.move("g1f1", byUser = false))
        assertEquals("solved", a.move("e2c3", byUser = true))
        val b = Attempt(puzzle, "fork")
        assertEquals("failed", b.move("d4c2", byUser = true))
        assertNull("decided once", b.move("d4e2", byUser = true))
        // Any mate counts where the line mates: Black's king steps to the corner and White mates
        // on the back rank (desktop 4e37a6f: the line had White moving twice, g1h1 a1a8).
        val mate = Puzzle("m", "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1", listOf("g8h8", "a1a8"), 900, 1, 1, listOf("mateIn1"))
        assertEquals("solved", Attempt(mate, "mate").move("a1a8", byUser = true))
        // Another mating move than the line's is played through the line and read on the board
        // (`alsoMates`, which the line's own move never reaches); a move that does not mate fails
        // (the parity audit, 2026-09-29).
        val two = Puzzle("m2", "6k1/5ppp/8/8/8/8/5PPP/RR4K1 b - - 0 1", listOf("g8h8", "a1a8"), 900, 1, 1, listOf("mateIn1"))
        assertEquals("solved", Attempt(two, "mate").move("b1b8", byUser = true))
        assertEquals("failed", Attempt(two, "mate").move("b1b7", byUser = true))
    }

    @Test
    fun aWrongMoveOnAPuzzleOnTheScreenFailsItAndTheLevelStepsBack() {
        val puzzles = PuzzleStore(null)
        puzzles.importCsv(csv(dir("screen")))
        val store = Store(null)
        val profiler = Profiler(store, sessionId = "s", ts = 0.0, puzzles = puzzles)
        val pin = puzzles.get("pin01")!!
        val board = Board(pin.fen)
        board.push(com.sidecoach.chess.Move.fromUci("d7d6"))
        // The position as a cold read gives it: placement and side, no rights or counters.
        val shown = "${board.boardFen()} ${if (board.turn == WHITE) "w" else "b"} - - 0 1"
        profiler.onEvent(GameStarted("q", Pieces.Side.WHITE, shown, kind = "position"), 3.0)
        val out = profiler.onEvent(played("q", shown, "Nc3"), 4.0)
        val outcome = out.filterIsInstance<PuzzleOutcome>().single()
        assertTrue(!outcome.solved && outcome.puzzleId == "pin01")
        assertEquals(Puzzles.LEVEL_START - Puzzles.LEVEL_STEP, outcome.level)
        assertNull("not one the profile recommended", outcome.motif)
        // A position that is no puzzle of the store is left alone.
        val start = com.sidecoach.chess.STARTING_FEN
        assertEquals(emptyList<Any>(), profiler.onEvent(GameStarted("r", Pieces.Side.WHITE, start, kind = "position"), 5.0))
        assertEquals(emptyList<Any>(), profiler.onEvent(played("r", start, "e4"), 6.0))
        store.close()
        puzzles.close()
    }

    private fun played(gameId: String, fen: String, san: String): MovePlayed {
        val board = Board(fen)
        val move = board.parseSan(san)!!
        val after = board.copy()
        after.push(move)
        return MovePlayed(gameId, 1, san, move.uci(), board.fen(), after.fen(), byUser = true)
    }

    /**
     * The CSV as Python's `csv.reader` reads it (every kind of line end, quotes, blank lines, a
     * record the text ends in), and a puzzle file with all of that as `import_csv` keeps it.
     */
    @Test
    fun theCsvIsReadAsPythonReadsIt() {
        val data = instrumentation.context.assets.open("profile/profile.json").bufferedReader().use { JSONObject(it.readText()) }
        val texts = data.getJSONArray("csv_texts")
        for (i in 0 until texts.length()) {
            val case = texts.getJSONObject(i)
            val want = case.getJSONArray("records").let { r -> (0 until r.length()).map { strings(r.getJSONArray(it)) } }
            assertEquals(JSONObject.quote(case.getString("text")), want, Puzzles.csvRows(case.getString("text")))
        }
        val tricky = data.getJSONObject("csv_import")
        val file = File(dir("tricky"), "tricky.csv").apply { writeBytes(tricky.getString("text").toByteArray(Charsets.UTF_8)) }
        val store = PuzzleStore(null)
        assertEquals("added", tricky.getInt("added"), store.importCsv(file))
        assertEquals(tricky.getInt("added"), store.count())
        val rows = tricky.getJSONArray("rows")
        for (i in 0 until rows.length()) {
            val row = rows.getJSONArray(i)
            val puzzle = store.get(row.getString(0))
            assertNotNull(row.getString(0), puzzle)
            puzzle!!
            assertEquals(row.getString(1), puzzle.fen)
            assertEquals(Py.split(row.getString(2)), puzzle.moves)
            assertEquals(row.getInt(3), puzzle.rating)
            assertEquals(row.getInt(4), puzzle.popularity)
            assertEquals(row.getInt(5), puzzle.plays)
            assertEquals(Py.split(row.getString(6)), puzzle.themes)
            assertEquals(row.getString(7), puzzle.position())
        }
        store.close()
    }

    /**
     * `test_a_row_too_short_for_a_rating_or_moves_is_skipped` (desktop 1e2e10f): a short row's
     * missing fields read as None (csv.DictReader), and int(None) and None.split() crashed the
     * desktop's import; both skip it now, as the phone did.
     */
    @Test
    fun aRowTooShortForARatingOrMovesIsSkipped() {
        val file = File(dir("short"), "short.csv")
        file.writeText(listOf(HEADER, ROWS[0], "short1,8/8/8/8/8/8/8/K6k w - - 0 1", "short2").joinToString("\n") + "\n")
        val store = PuzzleStore(null)
        assertEquals(1, store.importCsv(file))
        store.close()
    }

    private fun strings(a: JSONArray): List<String> = (0 until a.length()).map { a.getString(it) }

    /**
     * The player's rating is computed against the puzzle's own rating and deviation, as Lichess's
     * PuzzleFinisher does: the import keeps the CSV's RatingDeviation, 75 where a row has none or
     * not a number (`test_a_puzzle_keeps_the_rating_deviation_lichess_gives_it`).
     */
    @Test
    fun aPuzzleKeepsTheRatingDeviationLichessGivesIt() {
        val store = PuzzleStore(null)
        store.importCsv(csv(dir("deviation")))
        assertEquals(listOf(76, 82, 70), listOf("000Pw", "000rO", "mate1").map { store.get(it)!!.deviation })
        val odd = ROWS[2].replace(",1000,80,", ",1000,,").replaceFirst("fork1", "none1")
        val bad = ROWS[3].replace(",1200,80,", ",1200,x,").replaceFirst("pin01", "bad01")
        val file = File(dir("odd"), "odd.csv").apply { writeText(listOf(HEADER, odd, bad).joinToString("\n") + "\n") }
        assertEquals(2, store.importCsv(file))
        assertEquals(75, store.get("none1")!!.deviation)
        assertEquals(75, store.get("bad01")!!.deviation)
        assertEquals(75, Puzzle("x", com.sidecoach.chess.STARTING_FEN, listOf("e2e4", "e7e5"), 1000, 0, 0, emptyList()).deviation)
        store.close()
    }

    /** A store from before the puzzles kept their deviation, as an older app left it. */
    private fun oldStore(files: File, ids: List<String>) {
        val db = android.database.sqlite.SQLiteDatabase.openOrCreateDatabase(File(files, "puzzles.sqlite"), null)
        db.execSQL(
            "CREATE TABLE puzzles (id TEXT PRIMARY KEY, fen TEXT NOT NULL, moves TEXT NOT NULL, rating INTEGER NOT NULL," +
                " popularity INTEGER NOT NULL DEFAULT 0, plays INTEGER NOT NULL DEFAULT 0, themes TEXT NOT NULL," +
                " position TEXT NOT NULL)",
        )
        for (id in ids) {
            db.execSQL(
                "INSERT INTO puzzles VALUES (?, ?, 'g8f6 f3e5 c6e5', 1000, 80, 500, 'fork', 'k')",
                arrayOf(id, "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2"),
            )
        }
        db.close()
    }

    /** `test_a_store_from_before_the_deviations_gets_them_from_its_csv_once`. */
    @Test
    fun aStoreFromBeforeTheDeviationsGetsThemFromItsCsvOnce() {
        val files = dir("old")
        csv(File(files, "puzzles").apply { mkdirs() })
        oldStore(files, listOf("fork1", "gone1"))
        val store = Puzzles.openStore(files)!!
        assertEquals(1, store.version())
        assertEquals("the puzzles stay; only their deviations are filled in", 2, store.count())
        assertEquals("from the CSV", 80, store.get("fork1")!!.deviation)
        assertEquals("none in the CSV: the fallback", 75, store.get("gone1")!!.deviation)
        store.close()
        // Once: a CSV changed afterwards is not read again.
        File(File(files, "puzzles"), "sample.csv").writeText(
            HEADER + "\n" + ROWS.joinToString("\n") { it.replace(",1000,80,", ",1000,99,") } + "\n",
        )
        val again = Puzzles.openStore(files)!!
        assertEquals(80, again.get("fork1")!!.deviation)
        again.close()
    }

    /** `test_without_its_csv_an_old_store_keeps_the_fallback_until_the_csv_comes`. */
    @Test
    fun withoutItsCsvAnOldStoreKeepsTheFallbackUntilTheCsvComes() {
        val files = dir("nocsv")
        oldStore(files, listOf("fork1"))
        val store = Puzzles.openStore(files)!!
        assertEquals(0, store.version())
        assertEquals(75, store.get("fork1")!!.deviation)
        store.close()
        csv(File(files, "puzzles").apply { mkdirs() })
        val again = Puzzles.openStore(files)!!
        assertEquals(1, again.version())
        assertEquals(80, again.get("fork1")!!.deviation)
        again.close()
    }

    private companion object {
        val Py = com.sidecoach.voice.Py

        const val HEADER = "PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags"

        /** The desktop test's rows, in the Lichess database's format: the opponent's move first. */
        val ROWS = listOf(
            "000Pw,6k1/5p1p/4p3/4q3/3nN3/2Q3P1/PP3P1P/6K1 w - - 2 37,e4d2 d4e2 g1f1 e2c3,1547,76,92,629," +
                "crushing endgame fork short,https://lichess.org/au2lCK5o#73,",
            "000rO,3R4/8/K7/pB2b3/1p6/1P2k3/3p4/8 w - - 4 58,a6a5 e5c7 a5b4 c7d8,1110,82,85,74," +
                "crushing endgame fork master short,https://lichess.org/tzeeBEc2#115,",
            "fork1,r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,g8f6 f3e5 c6e5," +
                "1000,80,90,500,fork opening short,https://lichess.org/x#3,",
            "pin01,rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2,d7d6 f1b5 c7c6," +
                "1200,80,90,400,pin opening short,https://lichess.org/y#3,",
            // Black's king steps to the corner; White mates on the back rank (desktop 4e37a6f).
            "mate1,6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1,g8h8 a1a8,900,70,95,900," +
                "mateIn1 backRankMate endgame oneMove,https://lichess.org/z#1,",
            "hi999,6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1,g8h8 a1a8,1900,70,95,900,fork," +
                "https://lichess.org/w#1,",
        )
    }
}
