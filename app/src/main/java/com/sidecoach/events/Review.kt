package com.sidecoach.events

import org.json.JSONArray
import org.json.JSONObject

/**
 * A game reviewed whole and summed up: `events.ReviewedMove`, `ReviewedGame`, `PhaseGrade`,
 * `PerformanceRating`, `SideSummary`, `KeyMoment`, `GameSummary` -- and their JSON, as pydantic
 * writes and reads it (`model_dump_json`, `model_validate_json`): the review is kept in the
 * store in it, and the page reads the summary in it.
 */

/** `events.Side`. */
enum class Side(val value: String) {
    WHITE("white"),
    BLACK("black"),
    ;

    val other: Side get() = if (this == WHITE) BLACK else WHITE

    companion object {
        fun of(value: String): Side = entries.first { it.value == value }

        /** `Side.of(color)`: White for true. */
        fun of(white: Boolean): Side = if (white) WHITE else BLACK
    }
}

/**
 * One move of a game reviewed whole (`analysis/GameReview.kt`): who played it from where, what
 * it cost, and what the position it led to holds. The mover's clock after the move and how long
 * it took, in ms, from the PGN's clock comments; null without them.
 */
class ReviewedMove(
    val ply: Int,
    val san: String,
    val side: Side,
    val fenBefore: String,
    val assessment: MoveAssessment,
    val after: Analysis,
    val clockMs: Int? = null,
    val spentMs: Int? = null,
)

/** A game reviewed whole: its headers, what its first position holds, and every move. */
class ReviewedGame(
    val headers: Map<String, String> = emptyMap(),
    val start: Analysis,
    val moves: List<ReviewedMove> = emptyList(),
    val strength: String,
    val depth: Int,
    val seconds: Double = 0.0,
)

/** How one side played one phase: its accuracy there and the grade drawn from it. */
data class PhaseGrade(val phase: String, val moves: Int, val accuracy: Double, val grade: Label)

/** The rating a side played a game at, from its moves alone, with the middle 80% of where it may be. */
data class PerformanceRating(val rating: Int, val low: Int, val high: Int)

/** One side of a reviewed game in numbers: the summary screen's column. */
data class SideSummary(
    val side: Side,
    val name: String? = null,
    val rating: Int? = null,
    val accuracy: Double? = null,
    val counts: Map<String, Int> = emptyMap(),
    val phases: List<PhaseGrade> = emptyList(),
    val performance: PerformanceRating? = null,
)

/** A move the review stops at: the last book move, or a move with a label worth a word of its own. */
data class KeyMoment(val ply: Int, val side: Side, val kind: String)

/**
 * A reviewed game summed up: each side's numbers, the moves the review stops at, and White's
 * expected score after every ply (the graph; the start position first).
 */
data class GameSummary(
    val white: SideSummary,
    val black: SideSummary,
    val opening: String? = null,
    val lastBookPly: Int? = null,
    val keyMoments: List<KeyMoment> = emptyList(),
    val graph: List<Double> = emptyList(),
    val shape: String? = null,
)

/** The JSON of the events above and of those they hold, in pydantic's field names and order. */
object ReviewJson {

    private fun nul(value: Any?): Any = value ?: JSONObject.NULL

    private fun strings(values: List<String>) = JSONArray().apply { values.forEach { put(it) } }

    private fun JSONObject.str(name: String): String? = if (isNull(name)) null else getString(name)
    private fun JSONObject.int(name: String): Int? = if (isNull(name)) null else getInt(name)
    private fun JSONObject.dbl(name: String): Double? = if (isNull(name)) null else getDouble(name)

    private fun stringList(array: JSONArray?): List<String> =
        if (array == null) emptyList() else (0 until array.length()).map { array.getString(it) }

    private fun <T> list(array: JSONArray?, read: (JSONObject) -> T): List<T> =
        if (array == null) emptyList() else (0 until array.length()).map { read(array.getJSONObject(it)) }

    fun label(value: String): Label = Label.valueOf(value.uppercase())

    fun labelName(label: Label): String = label.name.lowercase()

    // ---- Line, MotifTag ---------------------------------------------------------------------

    fun line(l: Line): JSONObject = JSONObject()
        .put("moves", strings(l.moves))
        .put("depth", l.depth)
        .put("expected_score", l.expectedScore)
        .put("score_cp", nul(l.scoreCp))
        .put("mate", nul(l.mate))
        .put("wdl", l.wdl?.let { JSONArray().put(it.first).put(it.second).put(it.third) } ?: JSONObject.NULL)

    fun line(o: JSONObject): Line = Line(
        moves = stringList(o.getJSONArray("moves")),
        depth = o.getInt("depth"),
        expectedScore = o.getDouble("expected_score"),
        scoreCp = o.int("score_cp"),
        mate = o.int("mate"),
        wdl = if (o.isNull("wdl")) null else o.getJSONArray("wdl").let { Triple(it.getInt(0), it.getInt(1), it.getInt(2)) },
    )

    fun tag(t: MotifTag): JSONObject =
        JSONObject().put("name", t.name).put("squares", strings(t.squares)).put("move", nul(t.move))

    fun tag(o: JSONObject): MotifTag =
        MotifTag(o.getString("name"), stringList(o.optJSONArray("squares")), o.str("move"))

    // ---- Analysis, MoveAssessment -----------------------------------------------------------

    fun analysis(a: Analysis): JSONObject = JSONObject()
        .put("ts", a.ts)
        .put("source_frame_id", nul(a.sourceFrameId))
        .put("game_id", a.gameId)
        .put("fen", a.fen)
        .put("ply", a.ply)
        .put("lines", JSONArray(a.lines.map { line(it) }))
        .put("decisive", a.decisive)
        .put("only_move", a.onlyMove)
        .put("threats", strings(a.threats))
        .put("traps", JSONArray(a.traps.map { line(it) }))
        .put("defences", JSONArray(a.defences.map { line(it) }))
        .put("threatens", JSONArray(a.threatens.map { line(it) }))
        .put("tags", JSONArray(a.tags.map { tag(it) }))
        .put("opening", nul(a.opening))
        .put("in_book", a.inBook)
        .put("provisional", a.provisional)

    fun analysis(o: JSONObject): Analysis = Analysis(
        gameId = o.getString("game_id"),
        fen = o.getString("fen"),
        ply = o.getInt("ply"),
        lines = list(o.getJSONArray("lines")) { line(it) },
        decisive = o.getBoolean("decisive"),
        onlyMove = o.getBoolean("only_move"),
        threats = stringList(o.optJSONArray("threats")),
        traps = list(o.optJSONArray("traps")) { line(it) },
        defences = list(o.optJSONArray("defences")) { line(it) },
        threatens = list(o.optJSONArray("threatens")) { line(it) },
        tags = list(o.optJSONArray("tags")) { tag(it) },
        opening = o.str("opening"),
        inBook = o.optBoolean("in_book", false),
        provisional = o.optBoolean("provisional", false),
    ).also {
        it.ts = o.optDouble("ts", 0.0)
        it.sourceFrameId = o.int("source_frame_id")
    }

    fun assessment(m: MoveAssessment): JSONObject = JSONObject()
        .put("ts", m.ts)
        .put("source_frame_id", nul(m.sourceFrameId))
        .put("game_id", m.gameId)
        .put("ply", m.ply)
        .put("uci", m.uci)
        .put("expected_loss", m.expectedLoss)
        .put("severity", m.severity.name.lowercase())
        .put("best", strings(m.best))
        .put("tags", JSONArray(m.tags.map { tag(it) }))
        .put("depth", m.depth)
        .put("label", m.label?.let { labelName(it) } ?: JSONObject.NULL)
        .put("accuracy", nul(m.accuracy))

    fun assessment(o: JSONObject): MoveAssessment = MoveAssessment(
        gameId = o.getString("game_id"),
        ply = o.getInt("ply"),
        uci = o.getString("uci"),
        expectedLoss = o.getDouble("expected_loss"),
        severity = Severity.valueOf(o.getString("severity").uppercase()),
        best = stringList(o.getJSONArray("best")),
        tags = list(o.optJSONArray("tags")) { tag(it) },
        depth = o.optInt("depth", 0),
        label = o.str("label")?.let { label(it) },
        accuracy = o.dbl("accuracy"),
    ).also {
        it.ts = o.optDouble("ts", 0.0)
        it.sourceFrameId = o.int("source_frame_id")
    }

    // ---- Measured (game_review.measured_json) ------------------------------------------------

    fun measured(m: com.sidecoach.analysis.Measured): JSONObject = JSONObject()
        .put("fen", m.fen)
        .put("lines", JSONArray(m.lines.map { line(it) }))
        .put("score", m.score)
        .put("decisive", m.decisive)
        .put("only_move", m.onlyMove)
        .put("threats", strings(m.threats))
        .put("traps", JSONArray(m.traps.map { line(it) }))
        .put("defences", JSONArray(m.defences.map { line(it) }))
        .put("threatens", JSONArray(m.threatens.map { line(it) }))

    fun measured(o: JSONObject): com.sidecoach.analysis.Measured = com.sidecoach.analysis.Measured(
        fen = o.getString("fen"),
        lines = list(o.getJSONArray("lines")) { line(it) },
        score = o.getDouble("score"),
        decisive = o.getBoolean("decisive"),
        onlyMove = o.getBoolean("only_move"),
        threats = stringList(o.optJSONArray("threats")),
        traps = list(o.optJSONArray("traps")) { line(it) },
        defences = list(o.optJSONArray("defences")) { line(it) },
        threatens = list(o.optJSONArray("threatens")) { line(it) },
    )

    // ---- ReviewedGame -----------------------------------------------------------------------

    fun move(m: ReviewedMove): JSONObject = JSONObject()
        .put("ply", m.ply)
        .put("san", m.san)
        .put("side", m.side.value)
        .put("fen_before", m.fenBefore)
        .put("assessment", assessment(m.assessment))
        .put("after", analysis(m.after))
        .put("clock_ms", nul(m.clockMs))
        .put("spent_ms", nul(m.spentMs))

    fun move(o: JSONObject): ReviewedMove = ReviewedMove(
        ply = o.getInt("ply"),
        san = o.getString("san"),
        side = Side.of(o.getString("side")),
        fenBefore = o.getString("fen_before"),
        assessment = assessment(o.getJSONObject("assessment")),
        after = analysis(o.getJSONObject("after")),
        clockMs = o.int("clock_ms"),
        spentMs = o.int("spent_ms"),
    )

    fun reviewed(g: ReviewedGame): JSONObject = JSONObject()
        .put("headers", JSONObject().apply { for ((k, v) in g.headers) put(k, v) })
        .put("start", analysis(g.start))
        .put("moves", JSONArray(g.moves.map { move(it) }))
        .put("strength", g.strength)
        .put("depth", g.depth)
        .put("seconds", g.seconds)

    fun reviewed(o: JSONObject): ReviewedGame = ReviewedGame(
        headers = o.optJSONObject("headers")?.let { h -> h.keys().asSequence().associateWithTo(LinkedHashMap()) { h.getString(it) } }
            ?: emptyMap(),
        start = analysis(o.getJSONObject("start")),
        moves = list(o.optJSONArray("moves")) { move(it) },
        strength = o.getString("strength"),
        depth = o.getInt("depth"),
        seconds = o.optDouble("seconds", 0.0),
    )

    fun reviewed(text: String): ReviewedGame = reviewed(JSONObject(text))

    // ---- GameSummary ------------------------------------------------------------------------

    fun summary(s: GameSummary): JSONObject = JSONObject()
        .put("white", side(s.white))
        .put("black", side(s.black))
        .put("opening", nul(s.opening))
        .put("last_book_ply", nul(s.lastBookPly))
        .put(
            "key_moments",
            JSONArray(s.keyMoments.map { JSONObject().put("ply", it.ply).put("side", it.side.value).put("kind", it.kind) }),
        )
        .put("graph", JSONArray().apply { s.graph.forEach { put(it) } })
        .put("shape", nul(s.shape))

    private fun side(s: SideSummary): JSONObject = JSONObject()
        .put("side", s.side.value)
        .put("name", nul(s.name))
        .put("rating", nul(s.rating))
        .put("accuracy", nul(s.accuracy))
        .put("counts", JSONObject().apply { for ((k, v) in s.counts) put(k, v) })
        .put(
            "phases",
            JSONArray(
                s.phases.map {
                    JSONObject().put("phase", it.phase).put("moves", it.moves).put("accuracy", it.accuracy)
                        .put("grade", labelName(it.grade))
                },
            ),
        )
        .put(
            "performance",
            s.performance?.let { JSONObject().put("rating", it.rating).put("low", it.low).put("high", it.high) }
                ?: JSONObject.NULL,
        )

    /** `GameSummary.model_validate_json(text)`; "{}" fails, as pydantic's does (white and black are required). */
    fun summary(o: JSONObject): GameSummary = GameSummary(
        white = side(o.getJSONObject("white")),
        black = side(o.getJSONObject("black")),
        opening = o.str("opening"),
        lastBookPly = o.int("last_book_ply"),
        keyMoments = list(o.optJSONArray("key_moments")) {
            KeyMoment(it.getInt("ply"), Side.of(it.getString("side")), it.getString("kind"))
        },
        graph = o.optJSONArray("graph")?.let { g -> (0 until g.length()).map { g.getDouble(it) } } ?: emptyList(),
        shape = o.str("shape"),
    )

    fun summary(text: String): GameSummary = summary(JSONObject(text))

    private fun side(o: JSONObject): SideSummary = SideSummary(
        side = Side.of(o.getString("side")),
        name = o.str("name"),
        rating = o.int("rating"),
        accuracy = o.dbl("accuracy"),
        counts = o.optJSONObject("counts")?.let { c -> c.keys().asSequence().associateWithTo(LinkedHashMap()) { c.getInt(it) } }
            ?: emptyMap(),
        phases = list(o.optJSONArray("phases")) {
            PhaseGrade(it.getString("phase"), it.getInt("moves"), it.getDouble("accuracy"), label(it.getString("grade")))
        },
        performance = if (o.isNull("performance")) null else o.getJSONObject("performance").let {
            PerformanceRating(it.getInt("rating"), it.getInt("low"), it.getInt("high"))
        },
    )
}
