package com.sidecoach.state

import com.sidecoach.TestAssets
import com.sidecoach.TestLog
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.Piece
import com.sidecoach.events.Event
import com.sidecoach.events.GameEnded
import com.sidecoach.events.GameStarted
import com.sidecoach.events.MovePlayed
import com.sidecoach.events.PositionChanged
import com.sidecoach.events.Resync
import com.sidecoach.events.Takeback
import com.sidecoach.events.Unexplained
import com.sidecoach.vision.Pieces
import com.sidecoach.vision.Pieces.CellReading
import com.sidecoach.vision.Pieces.Kind
import com.sidecoach.vision.Pieces.Side
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The desktop's own tests of the reconciler, done to the port op for op.
 *
 * `tools/make_reconcile_fixtures.py` runs `tests/test_reconcile.py`, `test_premoves.py`, the
 * premove sweep's seeds (`test_premove_sweep.py`, two more that put a user's piece on a red
 * square), a few seeds of the stress sweep (`test_stress.py`: an arrow over a square a move
 * touches, a king under its own decoration; the parity audit, 2026-09-29) and the pipeline
 * tests whose games go through a reconciler, as they stand, with every
 * reconciler they build writing down what is done to it: built with which tunables, each frame
 * of readings and the events it answered, every call from outside, every field a test set on
 * it -- and after each, what it said of itself. Here each op is done to the ported reconciler
 * and its answer must be the same: the events, line for line, and the state (the position,
 * the record, browsing, the turn left open, a story pending, a drag, the frames unexplained,
 * whether the record agrees with the screen, which game, the cells remembered).
 *
 * A run stops at its first difference: what follows it is the difference again.
 */
class ReconcileOpsTest {

    @Test
    fun theDesktopsTestsOfTheReconcilerGoTheSameWayHere() {
        val text = TestAssets.text("chess/reconcile_ops.json")
        val root = JSONObject(text)
        val table = root.getJSONArray("readings")
        val runs = root.getJSONArray("runs")
        val failures = ArrayList<String>()
        var opsDone = 0
        val began = System.currentTimeMillis()
        val took = ArrayList<Pair<Long, String>>()
        for (r in 0 until runs.length()) {
            val run = runs.getJSONObject(r)
            val name = run.getString("name")
            val runBegan = System.currentTimeMillis()
            val ops = run.getJSONArray("ops")
            val recs = HashMap<Int, Reconciler>()
            val games = HashMap<String, Int>()
            val expected = HashMap<Int, MutableMap<String, Any?>>()
            for (o in 0 until ops.length()) {
                val op = ops.getJSONObject(o)
                val which = op.getInt("r")
                val kind = op.getString("op")
                val problem = try {
                    doOp(op, kind, which, recs, table)
                } catch (e: Exception) {
                    "threw ${e::class.java.simpleName}: ${e.message}"
                }
                opsDone += 1
                val rec = recs[which]
                val state = expected.getOrPut(which) { HashMap() }
                val delta = op.getJSONObject("state")
                for (key in delta.keys()) state[key] = plain(delta.get(key))
                val differs = problem ?: rec?.let { stateDiff(state, snapshot(it, games)) }
                if (differs != null) {
                    failures.add("$name op $o ($kind on reconciler $which): $differs")
                    break
                }
            }
            took.add(System.currentTimeMillis() - runBegan to name)
            if (failures.size >= 10) break
        }
        TestLog.i(
            "sidecoach.state",
            "the desktop's reconciler tests: ${runs.length()} tests, $opsDone ops, " +
                "${System.currentTimeMillis() - began} ms; the slowest: " +
                took.sortedByDescending { it.first }.take(5).joinToString { "${it.second} ${it.first} ms" },
        )
        assertTrue(failures.joinToString("\n"), failures.isEmpty())
    }

    /** `test_colour_only_reading_means_occupied_not_a_colour`, which builds no reconciler. */
    @Test
    fun aReadingOfAColourOnlySaysOccupiedNotWhichColour() {
        val reading = CellReading(Kind.PIECE, null, Side.WHITE, 0.5, 0.5, 240.0, false)
        assertTrue(cellAgrees(reading, Piece.fromSymbol("q")))
        assertTrue(cellAgrees(reading, Piece.fromSymbol("Q")))
        assertFalse(cellAgrees(reading, null))
    }

    /** Does one op; a description of what the port answered differently, or null. */
    private fun doOp(
        op: JSONObject,
        kind: String,
        which: Int,
        recs: HashMap<Int, Reconciler>,
        table: JSONArray,
    ): String? {
        if (kind == "new") {
            recs[which] = build(op.getJSONObject("args"))
            return null
        }
        val rec = recs.getValue(which)
        val events: List<Event>? = when (kind) {
            "update" -> rec.update(readings(table.getString(op.getInt("readings"))), op.getDouble("ts"))
            "board_unseen" -> { rec.boardUnseen(); null }
            "start" -> rec.start()
            "start_from" -> rec.startFrom(
                Board(op.getString("fen")),
                if (op.isNull("ts")) null else op.getDouble("ts"),
                ints(op.getJSONArray("highlighted")).toSet(),
                op.optJSONObject("before")?.let {
                    Board(it.getString("fen")) to ints(it.getJSONArray("highlighted")).toSet()
                },
            )
            "turn_from_clock" -> rec.turnFromClock(op.getBoolean("colour"))
            "end" -> rec.end(op.getString("result"), op.getString("reason"))
            "commit_pending_terminal" -> rec.commitPendingTerminal()
            "reorient" -> rec.reorient(side(op.getString("bottom")))
            "commit" -> rec.commit(Move.fromUci(op.getString("uci")))
            "queue_path" -> {
                val path = rec.queuePath(
                    op.getInt("origin"),
                    op.getInt("target"),
                    op.getString("symbol"),
                    ints(op.getJSONArray("tinted")).toSet(),
                    if (op.isNull("board")) null else Board(op.getString("board")),
                )?.map { it.uci() }
                val wanted = if (op.isNull("path")) null else strings(op.getJSONArray("path"))
                return if (path != wanted) "queue path $path, python's $wanted" else null
            }
            "observe" -> {
                val seen = rec.observe(readings(table.getString(op.getInt("readings")))).keys.sorted()
                val wanted = ints(op.getJSONArray("seen"))
                return if (seen != wanted) "observed $seen, python's $wanted" else null
            }
            "set" -> { set(rec, op.getString("name"), op.get("value")); null }
            else -> return "an op this test does not know"
        }
        if (op.has("events")) {
            val mine = events.orEmpty().map { line(it) }
            val theirs = strings(op.getJSONArray("events"))
            if (mine != theirs) {
                return "events\n  kotlin: ${mine.joinToString(" | ")}\n  python: ${theirs.joinToString(" | ")}"
            }
        }
        return null
    }

    private fun build(args: JSONObject) = Reconciler(
        bottom = side(args.getString("bottom")),
        userSide = side(args.getString("user_side")),
        settleFrames = args.getInt("settle_frames"),
        maxPlies = args.getInt("max_plies"),
        searchBudget = args.getInt("search_budget"),
        staleAfter = args.getInt("stale_after"),
        resyncCells = args.getInt("resync_cells"),
        resyncAfter = args.getInt("resync_after"),
        resyncHoldFrames = args.getInt("resync_hold_frames"),
        resyncMinFrames = args.getInt("resync_min_frames"),
        resyncHoldS = args.getDouble("resync_hold_s"),
        resyncJumpCells = args.getInt("resync_jump_cells"),
        maxStaleSubsetCells = args.getInt("max_stale_subset_cells"),
        pendingMaxAge = args.getInt("pending_max_age"),
        maxChainPlies = args.getInt("max_chain_plies"),
        maxQueueSteps = args.getInt("max_queue_steps"),
        resyncMaxSoft = args.getInt("resync_max_soft"),
        memoryFrames = args.getInt("memory_frames"),
        memoryGapTicks = args.getInt("memory_gap_ticks"),
        minReadableCells = args.getInt("min_readable_cells"),
    )

    /** A field a desktop test (or the pipeline) set on the reconciler from outside. */
    private fun set(rec: Reconciler, name: String, value: Any) {
        when (name) {
            "board" -> {
                val given = value as JSONObject
                val board = Board(given.getString("fen"))
                for (uci in strings(given.getJSONArray("moves"))) board.push(Move.fromUci(uci))
                rec.board = board
            }
            "_base_fen" -> rec.baseFenValue = value as String
            "_started" -> rec.startedFlag = value as Boolean
            "_finished" -> rec.finishedFlag = value as Boolean
            "bottom" -> rec.bottom = side(value as String)
            "user_side" -> rec.userSide = side(value as String)
            "settle_frames" -> rec.settleFrames = value as Int
            "max_plies" -> rec.maxPlies = value as Int
            "search_budget" -> rec.searchBudget = value as Int
            "stale_after" -> rec.staleAfter = value as Int
            "resync_cells" -> rec.resyncCells = value as Int
            "resync_after" -> rec.resyncAfter = value as Int
            "resync_hold_frames" -> rec.resyncHoldFrames = value as Int
            "resync_min_frames" -> rec.resyncMinFrames = value as Int
            "resync_hold_s" -> rec.resyncHoldS = (value as Number).toDouble()
            "resync_jump_cells" -> rec.resyncJumpCells = value as Int
            "max_stale_subset_cells" -> rec.maxStaleSubsetCells = value as Int
            "pending_max_age" -> rec.pendingMaxAge = value as Int
            "max_chain_plies" -> rec.maxChainPlies = value as Int
            "max_queue_steps" -> rec.maxQueueSteps = value as Int
            "resync_max_soft" -> rec.resyncMaxSoft = value as Int
            "memory_frames" -> rec.memoryFrames = value as Int
            "memory_gap_ticks" -> rec.memoryGapTicks = value as Int
            "min_readable_cells" -> rec.minReadableCells = value as Int
            "unexplained_note_frames" -> rec.unexplainedNoteFrames = value as Int
            else -> throw IllegalArgumentException("a field this test cannot set: $name")
        }
    }

    /** What the reconciler says of itself, keyed as the fixture keys it. */
    private fun snapshot(rec: Reconciler, games: HashMap<String, Int>): Map<String, Any?> = mapOf(
        "fen" to rec.board.fen(),
        "base" to rec.baseFen,
        "pgn" to rec.pgn(),
        "viewing" to rec.viewing,
        "turn_unknown" to rec.turnUnknown,
        "started" to rec.started,
        "finished" to rec.finished,
        "result" to rec.result,
        "kind" to rec.kind,
        "pending" to rec.pendingStory,
        "blocked" to rec.blocked,
        "unexplained" to rec.unexplainedFor,
        "agrees" to rec.agrees,
        "game" to games.getOrPut(rec.gameId) { games.size },
        "last_seen" to rec.lastSeen.keys.sorted(),
    )

    private fun stateDiff(wanted: Map<String, Any?>, mine: Map<String, Any?>): String? {
        val off = wanted.keys.filter { wanted[it] != mine[it] }
        if (off.isEmpty()) return null
        return off.joinToString("; ") { "$it is ${mine[it]}, python's ${wanted[it]}" }
    }

    /** A JSON value as the snapshot holds it: null, numbers as Int, lists of ints. */
    private fun plain(value: Any?): Any? = when (value) {
        JSONObject.NULL, null -> null
        is JSONArray -> ints(value)
        is Long -> value.toInt()
        else -> value
    }

    private fun ints(array: JSONArray): List<Int> = List(array.length()) { array.getInt(it) }

    private fun strings(array: JSONArray): List<String> = List(array.length()) { array.getString(it) }

    private fun side(value: String): Side = if (value == "white") Side.WHITE else Side.BLACK

    /**
     * A frame of readings from its code: six characters a cell (`cell_code` in the generator):
     * kind, symbol, colour, flags from 'A' (settled 1, hover 2, covered 4, drawn 8,
     * highlighted 16), tint, guess.
     */
    private fun readings(code: String): Readings = MutableList(64) { i ->
        val c = code.substring(6 * i, 6 * i + 6)
        val flags = c[3] - 'A'
        CellReading(
            kind = when (c[0]) {
                'e' -> Kind.EMPTY
                'p' -> Kind.PIECE
                else -> Kind.UNKNOWN
            },
            symbol = if (c[1] == '-') null else c[1].toString(),
            colour = when (c[2]) {
                'w' -> Side.WHITE
                'b' -> Side.BLACK
                else -> null
            },
            settled = flags and 1 != 0,
            hover = flags and 2 != 0,
            covered = flags and 4 != 0,
            drawn = flags and 8 != 0,
            highlighted = flags and 16 != 0,
            tint = when (c[4]) {
                'y' -> "yellow"
                'r' -> "red"
                'o' -> "other"
                else -> null
            },
            guess = if (c[5] == '-') null else c[5].toString(),
        )
    }

    /** The line `make_reconcile_fixtures.py` writes; the game id is random, so it is out. */
    private fun line(event: Event): String = when (event) {
        is GameStarted -> "GameStarted ${event.kind} ${sideName(event.userSide)} ${event.startFen}"
        is PositionChanged -> "PositionChanged ${event.ply} ${event.fen}"
        is MovePlayed ->
            "MovePlayed ${event.ply} ${event.san} ${event.uci} " +
                "${if (event.byUser) "user" else "them"} ${event.thinkMs ?: "None"}"
        is Takeback -> "Takeback ${event.plies} ${event.fen}"
        is Resync -> "Resync ${event.reason} | ${event.fen}"
        is Unexplained -> "Unexplained ${event.reason} | ${event.fen}" + if (event.told) " | told" else ""
        is GameEnded -> "GameEnded ${event.result} ${event.reason} | ${event.pgn} | ${event.finalFen}"
        else -> event::class.java.simpleName
    }

    private fun sideName(side: Pieces.Side): String = if (side == Side.WHITE) "white" else "black"
}
