package com.sidecoach.coach

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import com.sidecoach.analysis.phaseOf
import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.squareName
import com.sidecoach.events.Analysis
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.Label
import com.sidecoach.events.Line
import com.sidecoach.events.MotifTag
import com.sidecoach.events.MoveAssessment
import com.sidecoach.events.Severity
import com.sidecoach.events.Trigger
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs

/**
 * The coach's primitives held to the desktop's, sentence for sentence and mark for mark.
 *
 * Everything under `coach/` that is a pure function of the board and the analysis: who "you"
 * is, the disclosure filter and the leak check, the fact bundle, the idea of the best line
 * and its words, the offline coach's templates, what a move does, what stands on the board,
 * where a line leads, the word on a sound move, when the coach may speak, and the marks that
 * go with a message. The coach reads these out, so a port that differs anywhere says
 * something different. Every Analysis is the one the desktop was fed, read from the fixture.
 */
@RunWith(AndroidJUnit4::class)
class CoachPrimitivesTest {

    /** coach.json less its positions, which [forEachIn] streams one at a time. */
    private fun fixtures(): JSONObject =
        fixture(COACH, setOf("persons", "unlocks", "controls", "policy", "standings"))

    private val failures = ArrayList<String>()
    private var mismatches = 0

    /** Safe from the replay's several threads. */
    @Synchronized
    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine != theirs) {
            mismatches += 1
            if (failures.size < 8) failures.add("$what:\n  kotlin: $mine\n  python: $theirs")
        }
    }

    @Test
    fun everyPrimitiveMatchesThePython() {
        val data = fixtures()
        val began = System.currentTimeMillis()

        // Who "you" is.
        val persons = data.getJSONArray("persons")
        for (i in 0 until persons.length()) {
            val o = persons.getJSONObject(i)
            check(
                "thirdPerson(${o.getString("text")}, ${o.getString("side")})",
                Person.thirdPerson(o.getString("text"), o.getString("side")),
                o.getString("out"),
            )
        }

        // The disclosure state machine.
        val unlocks = data.getJSONArray("unlocks")
        for (i in 0 until unlocks.length()) {
            val o = unlocks.getJSONObject(i)
            val state = Disclosure(DisclosureLevel.of(o.getInt("start")))
            val result = state.apply(Unlock.entries.first { it.slug == o.getString("unlock") })
            check(
                "unlock ${o.getString("unlock")} from ${o.getInt("start")}",
                (result?.value ?: -1) to state.level.value,
                (if (o.isNull("result")) -1 else o.getInt("result")) to o.getInt("level"),
            )
        }

        // The time-control bands and the trigger priority.
        val controls = data.getJSONArray("controls")
        for (i in 0 until controls.length()) {
            val o = controls.getJSONObject(i)
            val initial = if (o.isNull("initial")) null else o.getInt("initial")
            check(
                "timeControlOf($initial, ${o.getInt("increment")})",
                timeControlOf(initial, o.getInt("increment")).slug,
                o.getString("control"),
            )
        }
        checkPolicy(data.getJSONArray("policy"))

        // The standing in words, at the bands' edges.
        val standings = data.getJSONArray("standings")
        for (i in 0 until standings.length()) {
            val o = standings.getJSONObject(i)
            val score = o.getDouble("score")
            check("standingWords($score)", standingWords(score), o.getString("you"))
            check("standingWords($score, third)", standingWords(score, thirdPerson = true), o.getString("third"))
        }

        // The positions.
        // Every position on its own, side by side on the cores; the counts kept safe from them.
        val positions = AtomicInteger()
        val deepCases = AtomicInteger()
        val trapped = AtomicInteger()
        val goals = ConcurrentHashMap<String, Int>()
        forEachInParallel(COACH, "cases") { case ->
            positions.incrementAndGet()
            val board = boardOf(case)
            val us = board.turn
            val analysis = analysisFrom(case.getString("fen"), case.getJSONObject("analysis"))
            val facts = factsFromAnalysis(board, analysis, lastMove = "e4", user = us)
            checkFacts(case.getString("fen"), facts, case.getJSONObject("facts"))

            val idea = IdeaReader.read(board, analysis)
            goals.merge(idea.goal, 1, Int::plus)
            checkIdea(case.getString("fen"), idea, case.getJSONObject("idea"))
            checkWords(case.getString("fen"), idea, board, case.getJSONObject("words"))
            checkTemplates(case.getString("fen"), facts, board, case.getJSONObject("templates"))

            check("${case.getString("fen")} phase", phaseOf(board), case.getString("phase"))
            check(
                "${case.getString("fen")} allowed",
                IdeaWords.allowed(board, analysis, "e4"),
                if (case.isNull("allowed")) null else case.getString("allowed"),
            )
            // The traps ranked as the first hint wants them, each said with the best move.
            if (!case.isNull("trap_reads")) {
                val reads = case.getJSONObject("trap_reads")
                val ranked = Situation.rankedTraps(analysis.traps, idea.target)
                val wanted = reads.getJSONArray("ranked")
                check(
                    "${case.getString("fen")} ranked traps",
                    ranked.map { it.moves },
                    (0 until wanted.length()).map { strings(wanted.getJSONArray(it)) },
                )
                val best = analysis.lines.firstOrNull()?.moves?.firstOrNull()?.let { Move.fromUciOrNull(it) }
                val top = analysis.lines.firstOrNull()?.expectedScore
                val words = reads.getJSONArray("words")
                check(
                    "${case.getString("fen")} trap words",
                    ranked.map { Situation.trapWords(board, it, idea.target, top, best) },
                    (0 until words.length()).map { if (words.isNull(it)) null else words.getString(it) },
                )
                trapped.incrementAndGet()
            }
            check("material us", materialText(board, us), case.getString("material_us"))
            check("material them", materialText(board, !us), case.getString("material_them"))
            check("position text", positionText(board), case.getString("position_text"))
            check("ascii", asciiBoard(board), case.getString("ascii"))
            check("inventory lines", inventoryLines(board, us), strings(case.getJSONArray("inventory_lines")))
            check(
                "line outcome",
                if (analysis.lines.isNotEmpty()) lineOutcome(board, analysis.lines[0].moves, us) else null,
                if (case.isNull("line_outcome")) null else case.getString("line_outcome"),
            )
            check(
                "line notes",
                if (analysis.lines.isNotEmpty()) lineNotes(board, analysis.lines[0].moves) else null,
                if (case.isNull("line_notes")) null else case.getString("line_notes"),
            )
            check(
                "line notes with the player",
                if (analysis.lines.isNotEmpty()) lineNotes(board, analysis.lines[0].moves, player = us) else null,
                if (case.isNull("line_notes_player")) null else case.getString("line_notes_player"),
            )

            val effects = case.getJSONArray("effects")
            for (k in 0 until effects.length()) {
                val o = effects.getJSONObject(k)
                val move = Move.fromUci(o.getString("move"))
                check("describeMove ${o.getString("move")}", describeMove(board, move), o.getString("described"))
                check(
                    "effects ${o.getString("move")}",
                    describeEffects(board, move, hazards = true),
                    strings(o.getJSONArray("hazards")),
                )
                check(
                    "effects no hazards ${o.getString("move")}",
                    describeEffects(board, move, hazards = false),
                    strings(o.getJSONArray("no_hazards")),
                )
                check("effects text ${o.getString("move")}", effectsText(board, move), o.getString("text"))
            }

            val leaks = case.getJSONArray("leaks")
            if (case.getJSONArray("verdicts").length() > 0) deepCases.incrementAndGet()
            for (k in 0 until leaks.length()) {
                val o = leaks.getJSONObject(k)
                val said = if (o.isNull("said")) null else o.getString("said")
                check(
                    "leak \"${o.getString("text")}\" at ${o.getInt("level")}",
                    leakCheck(o.getString("text"), facts, DisclosureLevel.of(o.getInt("level")), said),
                    strings(o.getJSONArray("leaks")),
                )
            }

            val marks = case.getJSONArray("marks")
            for (k in 0 until marks.length()) {
                val o = marks.getJSONObject(k)
                val played = if (o.isNull("played")) null else o.getString("played")
                val mine = annotationsFor(
                    facts,
                    DisclosureLevel.of(o.getInt("level")),
                    board,
                    Trigger.entries.first { it.slug == o.getString("trigger") },
                    played,
                    if (o.isNull("severity")) null else o.getString("severity"),
                )
                check(
                    "marks at ${o.getInt("level")} ${o.getString("trigger")}",
                    mine.map { showMark(it) },
                    annotations(o.getJSONArray("marks")),
                )
            }

            val verdicts = case.getJSONArray("verdicts")
            for (k in 0 until verdicts.length()) {
                val o = verdicts.getJSONObject(k)
                val move = Move.fromUci(o.getString("move"))
                val label = if (o.isNull("label")) null else o.getString("label")
                val last = if (o.isNull("last")) null else o.getString("last")
                val taught = strings(o.getJSONArray("taught"))
                check(
                    "praise ${o.getString("move")} $label",
                    praiseFor(label, last),
                    strings(o.getJSONArray("choices")),
                )
                val lesson = lesson(board, move, analysis.tags, facts.phase, facts.threats)
                check(
                    "lesson ${o.getString("move")}",
                    lesson?.let { listOf(it.first, it.second) },
                    if (o.isNull("lesson")) {
                        null
                    } else {
                        o.getJSONArray("lesson").let { listOf(it.getString(0), strings(it.getJSONArray(1))) }
                    },
                )
                check(
                    "verdict ${o.getString("move")} $label",
                    verdict(board, move, label, analysis.tags, facts.phase, facts.threats, taught, last, pick = { it[0] }),
                    o.getString("sentence") to if (o.isNull("kind")) null else o.getString("kind"),
                )
                // From 1800, below 1400 and from the live coach's bags, the shuffle fixed (TG-12),
                // written where they say something: `strong` where it differs from the sentence,
                // the plain words drawn last beside the same draw without them where the two
                // differ (the first wordings are all plain), the bags for a best and a good move.
                val sentence = o.getString("sentence") to if (o.isNull("kind")) null else o.getString("kind")
                fun pair(key: String) = o.getJSONArray(key).let { it.getString(0) to if (it.isNull(1)) null else it.getString(1) }
                check(
                    "verdict ${o.getString("move")} $label strong",
                    verdict(board, move, label, analysis.tags, facts.phase, facts.threats, taught, last, pick = { it[0] }, strong = true),
                    if (o.has("strong")) pair("strong") else sentence,
                )
                val drawn = verdict(board, move, label, analysis.tags, facts.phase, facts.threats, taught, last, pick = { it.last() })
                val plain = verdict(board, move, label, analysis.tags, facts.phase, facts.threats, taught, last, pick = { it.last() }, plain = true)
                if (o.has("plain")) {
                    check("verdict ${o.getString("move")} $label drawn last", drawn, pair("last_drawn"))
                    check("verdict ${o.getString("move")} $label plain", plain, pair("plain"))
                } else {
                    check("verdict ${o.getString("move")} $label plain as drawn", plain, drawn)
                }
                if (o.has("bagged")) {
                    check(
                        "verdict ${o.getString("move")} $label bagged",
                        verdict(board, move, label, analysis.tags, facts.phase, facts.threats, taught, last, bags = Bags { }),
                        pair("bagged"),
                    )
                }
            }

            val mentions = case.getJSONArray("mentions")
            for (k in 0 until mentions.length()) {
                val o = mentions.getJSONObject(k)
                check(
                    "mentions \"${o.getString("text")}\"",
                    mentionsIn(o.getString("text"), board).map { showMark(it) },
                    annotations(o.getJSONArray("marks")),
                )
            }
        }
        val took = System.currentTimeMillis() - began
        Log.i(
            TAG,
            "$positions positions ($deepCases with the whole sweep, $trapped with traps read), " +
                "${persons.length()} rewrites, goals $goals, ${took} ms",
        )
        assertTrue("$mismatches mismatches\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    /**
     * Every call the desktop's knowledge tests make, replayed with the desktop's answers
     * (ck-05, ck-06, ck-07, TG-08): `test_effects_legal.py` (a pinned piece or a king that
     * cannot take attacks nothing, attacked and defended by legal captures, a defence that
     * saves, a fork that wins from a square it keeps, a pin broken at a profit, a check
     * answered), `test_facts.py` (the theme, region, piece and candidates, a line the board
     * cannot play, near-equal candidates, a quiet developing move, regions, phase and material
     * -- once `test_coach_policy.py`'s -- the threat that becomes prophylaxis, a move and a
     * tactic in words, a fork the best move creates, only enemy targets, forcing or quiet, a
     * quiet move whose line wins) and `test_coach_annotations.py` (the marks by level, a played
     * move's colour, the mentions).
     */
    @Test
    fun theDesktopsKnowledgeTestsMatch() {
        val tests = fixture("chess/knowledge.json").getJSONObject("tests")
        assertTrue("the desktop's knowledge tests ran", tests.getJSONArray("ran").length() >= 30)
        val calls = tests.getJSONArray("calls")
        for (i in 0 until calls.length()) {
            val call = calls.getJSONObject(i)
            val fn = call.getString("fn")
            val a = call.getJSONObject("args")
            val board = if (a.has("fen")) Board(a.getString("fen")) else Board()
            fun colour(key: String): Boolean? = if (a.isNull(key)) null else a.getBoolean(key)
            val out = call.get("out")
            val (mine, theirs) = when (fn) {
                "facts_from_analysis" -> {
                    val o = a.getJSONObject("analysis")
                    val facts = factsFromAnalysis(
                        board, analysisFrom(o.getString("fen"), o),
                        lastMove = if (a.isNull("last_move")) null else a.getString("last_move"),
                        user = colour("user"),
                    )
                    facts.asMap() to factsMap(out as JSONObject)
                }
                "visible_facts" ->
                    visibleFacts(factsFrom(a.getJSONObject("facts")), DisclosureLevel.of(a.getInt("level"))) to
                        factsMap(out as JSONObject)
                "describe_move" -> describeMove(board, Move.fromUci(a.getString("move"))) to out
                "describe_tactic" -> describeTactic(
                    board, a.getString("name"), strings(a.getJSONArray("squares")),
                    if (a.isNull("best")) null else Move.fromUci(a.getString("best")),
                ) to (if (out == JSONObject.NULL) null else out)
                "describe_effects" -> describeEffects(
                    board, Move.fromUci(a.getString("move")), limit = a.getInt("limit"), hazards = a.getBoolean("hazards"),
                ) to strings(out as JSONArray)
                "line_notes" -> lineNotes(board, strings(a.getJSONArray("moves")), limit = a.getInt("limit"), player = colour("player")) to out
                "line_spoken" -> lineSpoken(board, strings(a.getJSONArray("moves")), limit = a.getInt("limit")) to out
                "region_of" -> regionOf(strings(a.getJSONArray("squares"))) to (if (out == JSONObject.NULL) null else out)
                "phase_of" -> phaseOf(board) to out
                "material_text" -> materialText(board, colour("perspective")) to out
                "annotations_for" -> annotationsFor(
                    factsFrom(a.getJSONObject("facts")), DisclosureLevel.of(a.getInt("level")), board,
                    Trigger.entries.first { it.slug == a.getString("trigger") },
                    if (a.isNull("played_san")) null else a.getString("played_san"),
                    if (a.isNull("severity")) null else a.getString("severity"),
                ).map { showMark(it) } to annotations(out as JSONArray)
                "mentions_in" -> mentionsIn(a.getString("text"), board).map { showMark(it) } to annotations(out as JSONArray)
                else -> throw IllegalArgumentException("no replay for $fn")
            }
            check("${call.getString("test")} $fn ${a.toString().take(160)}", mine, theirs)
        }
        assertTrue("$mismatches mismatches\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    /** A facts dump as the Kotlin's [PositionFacts.asMap] has it: lists, booleans, strings, nulls. */
    private fun factsMap(o: JSONObject): Map<String, Any?> = o.keys().asSequence().associateWith { key -> plainJson(o.get(key)) }

    private fun plainJson(value: Any?): Any? = when (value) {
        JSONObject.NULL, null -> null
        is JSONArray -> (0 until value.length()).map { plainJson(value.get(it)) }
        is JSONObject -> value.keys().asSequence().associateWith { plainJson(value.get(it)) }
        else -> value
    }

    private fun tagsFrom(a: JSONArray): List<MotifTag> = List(a.length()) { k ->
        val t = a.getJSONObject(k)
        MotifTag(
            name = t.getString("name"),
            squares = strings(t.getJSONArray("squares")),
            move = if (t.isNull("move")) null else t.getString("move"),
        )
    }

    private fun nullable(o: JSONObject, key: String): String? = if (o.isNull(key)) null else o.getString(key)

    /** PositionFacts from the desktop's `model_dump()`. */
    private fun factsFrom(o: JSONObject) = PositionFacts(
        fen = o.getString("fen"),
        sideToMove = o.getString("side_to_move"),
        userSide = nullable(o, "user_side"),
        phase = o.getString("phase"),
        material = nullable(o, "material"),
        lastMove = nullable(o, "last_move"),
        decisive = o.getBoolean("decisive"),
        onlyMove = o.getBoolean("only_move"),
        nature = nullable(o, "nature"),
        goal = nullable(o, "goal"),
        theme = nullable(o, "theme"),
        mechanism = nullable(o, "mechanism"),
        region = nullable(o, "region"),
        target = nullable(o, "target"),
        piece = nullable(o, "piece"),
        why = nullable(o, "why"),
        question = nullable(o, "question"),
        nudge = nullable(o, "nudge"),
        focus = strings(o.getJSONArray("focus")),
        tactic = nullable(o, "tactic"),
        plan = nullable(o, "plan"),
        ideas = strings(o.getJSONArray("ideas")),
        candidates = strings(o.getJSONArray("candidates")),
        good = o.optJSONArray("good")?.let { strings(it) } ?: emptyList(),
        threats = strings(o.getJSONArray("threats")),
        line = strings(o.getJSONArray("line")),
        stakes = nullable(o, "stakes"),
        key = nullable(o, "key"),
        trap = nullable(o, "trap"),
        choices = strings(o.getJSONArray("choices")),
        situation = nullable(o, "situation"),
        principle = nullable(o, "principle"),
    )

    /** Every field of the idea, squares by name and moves in UCI, as the generator wrote it. */
    private fun checkIdea(fen: String, idea: Idea, want: JSONObject) {
        fun sq(square: Int?): String? = square?.let { squareName(it) }
        fun opt(key: String): String? = if (want.isNull(key)) null else want.getString(key)
        fun optInt(key: String): Int? = if (want.isNull(key)) null else want.getInt(key)
        val mine = linkedMapOf<String, Any?>(
            "goal" to idea.goal,
            "first" to idea.first?.uci(),
            "line" to idea.line.map { it.uci() },
            "mate_in" to idea.mateIn,
            "gain" to idea.gain,
            "won" to idea.won,
            "blow" to idea.blow?.let { showTag(it.name, it.squares, it.move) },
            "blow_move" to idea.blowMove?.uci(),
            "blow_board" to idea.blowBoard?.fen(),
            "blow_at" to idea.blowAt,
            "gain_ply" to idea.gainPly,
            "preparation" to idea.preparation,
            "starts_with" to idea.startsWith,
            "target" to sq(idea.target),
            "threat" to idea.threat?.uci(),
            "threatened" to idea.threatened,
            "prepared" to idea.prepared,
            "follow" to idea.follow?.uci(),
            "flight" to idea.flight.map { squareName(it) },
            "checks" to idea.checks,
            "level" to idea.level,
            "down" to idea.down,
            "draw" to idea.draw,
            "ending" to idea.ending,
            "opposition" to idea.opposition,
            "runner" to sq(idea.runner),
            "pinned_guard" to sq(idea.pinnedGuard),
            "intermezzo" to idea.intermezzo,
            "also" to idea.also,
            "sacrifices" to idea.sacrifices.map { "${it.first}:${it.second}" },
            "winning" to idea.winning,
            "score" to idea.score,
            "back" to idea.back,
        )
        val blow = if (want.isNull("blow")) {
            null
        } else {
            want.getJSONObject("blow").let { b ->
                showTag(b.getString("name"), strings(b.getJSONArray("squares")), if (b.isNull("move")) null else b.getString("move"))
            }
        }
        val theirs = linkedMapOf<String, Any?>(
            "goal" to want.getString("goal"),
            "first" to opt("first"),
            "line" to strings(want.getJSONArray("line")),
            "mate_in" to optInt("mate_in"),
            "gain" to want.getInt("gain"),
            "won" to opt("won"),
            "blow" to blow,
            "blow_move" to opt("blow_move"),
            "blow_board" to opt("blow_board"),
            "blow_at" to want.getInt("blow_at"),
            "gain_ply" to want.getInt("gain_ply"),
            "preparation" to strings(want.getJSONArray("preparation")),
            "starts_with" to want.getString("starts_with"),
            "target" to opt("target"),
            "threat" to opt("threat"),
            "threatened" to opt("threatened"),
            "prepared" to want.getBoolean("prepared"),
            "follow" to opt("follow"),
            "flight" to strings(want.getJSONArray("flight")),
            "checks" to want.getInt("checks"),
            "level" to want.getBoolean("level"),
            "down" to want.getBoolean("down"),
            "draw" to opt("draw"),
            "ending" to opt("ending"),
            "opposition" to want.getBoolean("opposition"),
            "runner" to opt("runner"),
            "pinned_guard" to opt("pinned_guard"),
            "intermezzo" to want.getBoolean("intermezzo"),
            "also" to opt("also"),
            "sacrifices" to want.getJSONArray("sacrifices").let { a ->
                List(a.length()) { k -> a.getJSONArray(k).let { "${it.getInt(0)}:${it.getString(1)}" } }
            },
            "winning" to want.getBoolean("winning"),
            "score" to if (want.isNull("score")) null else want.getDouble("score"),
            "back" to want.getBoolean("back"),
        )
        for ((key, value) in mine) check("$fen idea.$key", value, theirs[key])
    }

    /** The idea in the coach's words, straight from [IdeaWords]. */
    private fun checkWords(fen: String, idea: Idea, board: Board, want: JSONObject) {
        fun opt(key: String): String? = if (want.isNull(key)) null else want.getString(key)
        check("$fen words.pattern", IdeaWords.pattern(idea), opt("pattern"))
        check("$fen words.goal", IdeaWords.goal(idea), opt("goal"))
        check("$fen words.mechanism", IdeaWords.mechanism(idea), opt("mechanism"))
        check("$fen words.question", IdeaWords.question(idea, null), opt("question"))
        check("$fen words.nudge", IdeaWords.nudge(idea), opt("nudge"))
        check("$fen words.target", IdeaWords.target(idea, board), opt("target"))
        check("$fen words.piece", IdeaWords.piece(idea, board), opt("piece"))
        check("$fen words.why", IdeaWords.why(idea, board, IdeaWords.Describe(::describeTactic)), opt("why"))
        // With the board, as the facts ask them (`clears_path`; the parity audit ck-09).
        // Written only where it differs from the boardless word.
        check(
            "$fen words.question with the board",
            IdeaWords.question(idea, null, board),
            if (want.has("question_board")) opt("question_board") else opt("question"),
        )
        check(
            "$fen words.nudge with the board",
            IdeaWords.nudge(idea, board),
            if (want.has("nudge_board")) opt("nudge_board") else opt("nudge"),
        )
    }

    /** The offline coach's words at each rung, the lessons, and each rung leak-checked. */
    private fun checkTemplates(fen: String, facts: PositionFacts, board: Board, want: JSONObject) {
        check("$fen ideaTemplate", ideaTemplate(facts), want.getString("idea"))
        check("$fen focusTemplate", focusTemplate(facts), want.getString("focus"))
        check("$fen moveTemplate", moveTemplate(facts), want.getString("move"))
        check("$fen lineTemplate", lineTemplate(facts), want.getString("line"))
        check("$fen lineTemplate with the board", lineTemplate(facts, board), want.getString("line_board"))
        val hints = want.getJSONObject("hints")
        for (key in hints.keys()) {
            check("$fen hintTemplate at $key", hintTemplate(facts, DisclosureLevel.of(key.toInt())), hints.getString(key))
        }
        val hintsBoard = want.getJSONObject("hints_board")
        for (key in hintsBoard.keys()) {
            check(
                "$fen hintTemplate at $key with the board",
                hintTemplate(facts, DisclosureLevel.of(key.toInt()), board),
                hintsBoard.getString(key),
            )
        }
        val best = want.getString("best")
        val lessons = want.getJSONArray("lessons")
        for (k in 0 until lessons.length()) {
            val o = lessons.getJSONObject(k)
            check("$fen lessonTemplate ${o.getString("opener")}", lessonTemplate(facts, best, o.getString("opener")), o.getString("text"))
        }
        // The lesson after the move played, with the position it was played from.
        val played = want.getJSONArray("lessons_played")
        for (k in 0 until played.length()) {
            val o = played.getJSONObject(k)
            val allowed = if (o.isNull("allowed")) null else o.getString("allowed")
            check(
                "$fen lessonTemplate after ${o.getString("played")} allowing $allowed",
                lessonTemplate(
                    facts, best, o.getString("opener"), board = board, played = o.getString("played"), allowed = allowed,
                ),
                o.getString("text"),
            )
        }
        val leaks = want.getJSONArray("leaks")
        for (k in 0 until leaks.length()) {
            val o = leaks.getJSONObject(k)
            val rung = DisclosureLevel.of(o.getInt("rung"))
            val level = DisclosureLevel.of(o.getInt("level"))
            check(
                "$fen leaks of the ${rung.name} template at ${level.name}",
                leakCheck(hintTemplate(facts, rung), facts, level),
                strings(o.getJSONArray("leaks")),
            )
        }
    }

    private fun showTag(name: String, squares: List<String>, move: String?): String =
        "$name ${squares.joinToString(",")} ${move ?: "-"}"

    private fun checkPolicy(runs: JSONArray) {
        for (i in 0 until runs.length()) {
            val run = runs.getJSONObject(i)
            val control = TimeControl.entries.first { it.slug == run.getString("control") }
            val policy = InterjectionPolicy(
                PolicySettings(quiet = run.getBoolean("quiet")), control,
            )
            val steps = run.getJSONArray("steps")
            val seen = ArrayList<Map<String, Any?>>()

            fun note(what: String, trigger: Trigger?) {
                seen.add(
                    mapOf(
                        "step" to what,
                        "trigger" to trigger?.slug,
                        "reason" to policy.lastReason,
                        "threshold" to policy.stuckThreshold(),
                        "scramble" to policy.inScramble,
                        "stuck_from" to policy.stuckFrom,
                    ),
                )
            }

            fun assessment(severity: Severity, label: Label?) = MoveAssessment(
                gameId = "g", ply = 1, uci = "e2e4", expectedLoss = 0.2,
                severity = severity, best = emptyList(), label = label,
            )

            policy.newPosition(2, 0.0, userToMove = true)
            note("new position", null)
            note("tick 10", policy.tick(10.0))
            note("tick 60", policy.tick(60.0))
            policy.said(2, 60.0)
            note("tick soon after said", policy.tick(80.0))
            note("tick after said", policy.tick(120.0))
            policy.said(2, 120.0)
            note("tick after two", policy.tick(300.0))
            policy.newPosition(2, 300.0, userToMove = true)
            note("same ply again", policy.tick(400.0))
            policy.newPosition(4, 100.0, userToMove = true)
            policy.said(4)
            note("tick after said without a time", policy.tick(150.0))
            policy.said(3, 150.0)
            note("said of another ply", policy.tick(151.0))
            policy.newPosition(12, 200.0, userToMove = true)
            policy.setRemaining(20_000)
            note("tick in scramble", policy.tick(400.0))
            policy.setRemaining(600_000)
            note("tick with clock", policy.tick(500.0))
            policy.newPosition(14, 500.0, userToMove = false)
            note("opponent to move", policy.tick(600.0))
            policy.newPosition(16, 600.0, userToMove = true)
            policy.covered = true
            note("covered", policy.tick(700.0))
            policy.covered = false
            note("blunder", policy.onUserMoveAssessed(assessment(Severity.BLUNDER, Label.BLUNDER)))
            note("fine move", policy.onUserMoveAssessed(assessment(Severity.NONE, Label.BEST)))
            note("miss", policy.onUserMoveAssessed(assessment(Severity.NONE, Label.MISS)))
            // The branches the script had never run (the parity audit, 2026-09-29): the clock's
            // share between the floor and the setting (five minutes gives 24 s, 100 s still
            // hints at the floor).
            policy.newPosition(22, 900.0, userToMove = true)
            policy.setRemaining(300_000)
            note("five minutes, before the share", policy.tick(920.0))
            note("five minutes, past the share", policy.tick(925.0))
            policy.setRemaining(100_000)
            policy.newPosition(24, 1000.0, userToMove = true)
            note("100 s, at the floor", policy.tick(1021.0))

            check("${run.getString("control")} steps", seen.size, steps.length())
            for (k in 0 until minOf(steps.length(), seen.size)) {
                val want = steps.getJSONObject(k)
                val got = seen[k]
                val label = "${run.getString("control")} quiet=${run.getBoolean("quiet")} " +
                    "step ${want.getString("step")}"
                check("$label trigger", got["trigger"], if (want.isNull("trigger")) null else want.getString("trigger"))
                check("$label reason", got["reason"], if (want.isNull("reason")) null else want.getString("reason"))
                val threshold = got["threshold"] as Double
                if (abs(threshold - want.getDouble("threshold")) > 1e-9) {
                    check("$label threshold", threshold, want.getDouble("threshold"))
                }
                check("$label scramble", got["scramble"], want.getBoolean("scramble"))
                check(
                    "$label stuck from",
                    got["stuck_from"],
                    if (want.isNull("stuck_from")) null else want.getDouble("stuck_from"),
                )
            }
        }
    }

    private fun checkFacts(fen: String, mine: PositionFacts, theirs: JSONObject) {
        val map = mine.asMap()
        for ((key, value) in map) {
            val want: Any? = when {
                theirs.isNull(key) -> null
                value is List<*> -> strings(theirs.getJSONArray(key))
                value is Boolean -> theirs.getBoolean(key)
                else -> theirs.getString(key)
            }
            check("$fen facts.$key", value, want)
        }
    }

    /** The Analysis the desktop was fed, exactly: its lines, flags, threats and tags. */
    private fun linesFrom(a: JSONArray): List<Line> = List(a.length()) { k ->
        val l = a.getJSONObject(k)
        Line(
            moves = strings(l.getJSONArray("moves")),
            depth = l.getInt("depth"),
            expectedScore = l.getDouble("expected_score"),
            scoreCp = if (l.isNull("score_cp")) null else l.getInt("score_cp"),
            mate = if (l.isNull("mate")) null else l.getInt("mate"),
        )
    }

    private fun analysisFrom(fen: String, o: JSONObject): Analysis {
        val lines = linesFrom(o.getJSONArray("lines"))
        val tags = o.getJSONArray("tags").let { a ->
            List(a.length()) { k ->
                val t = a.getJSONObject(k)
                MotifTag(
                    name = t.getString("name"),
                    squares = strings(t.getJSONArray("squares")),
                    move = if (t.isNull("move")) null else t.getString("move"),
                )
            }
        }
        return Analysis(
            gameId = "g",
            fen = fen,
            ply = 0,
            lines = lines,
            decisive = o.getBoolean("decisive"),
            onlyMove = o.getBoolean("only_move"),
            threats = strings(o.getJSONArray("threats")),
            traps = linesFrom(o.getJSONArray("traps")),
            defences = linesFrom(o.getJSONArray("defences")),
            threatens = linesFrom(o.getJSONArray("threatens")),
            tags = tags,
            opening = if (o.isNull("opening")) null else o.getString("opening"),
            inBook = o.getBoolean("in_book"),
        )
    }

    /**
     * The case's board, with the moves that led to it when the desktop's board had them: the
     * story reads their last move ("taken back", "win back").
     */
    private fun boardOf(case: JSONObject): Board {
        if (case.isNull("history")) return Board(case.getString("fen"))
        val history = case.getJSONObject("history")
        val board = Board(history.getString("root"))
        val moves = history.getJSONArray("moves")
        for (k in 0 until moves.length()) board.push(Move.fromUci(moves.getString(k)))
        check("${case.getString("fen")} replayed", board.fen(), case.getString("fen"))
        return board
    }

    private fun strings(array: JSONArray): List<String> =
        (0 until array.length()).map { array.getString(it) }

    private fun annotations(array: JSONArray): List<String> =
        (0 until array.length()).map { i ->
            val o = array.getJSONObject(i)
            val squares = o.getJSONArray("squares").let { a -> List(a.length()) { a.getString(it) } }
            mark(
                o.getString("kind"), squares, o.getString("color"), o.getDouble("opacity"),
                if (o.isNull("label")) null else o.getString("label"),
            )
        }

    private fun showMark(a: com.sidecoach.events.Annotation): String =
        mark(a.kind, a.squares, a.color, a.opacity, a.label)

    private fun mark(
        kind: String,
        squares: List<String>,
        color: String,
        opacity: Double,
        label: String?,
    ): String = "$kind ${squares.joinToString(",")} $color ${"%.4f".format(opacity)} ${label ?: "-"}"

    companion object {
        private const val TAG = "sidecoach.coach"
        private const val COACH = "chess/coach.json"
    }

    /**
     * `test_the_words_hold_for_the_board_the_move_leaves` (the facts audit, 2026-10-01): the file
     * as the capture leaves it, "defends" only of the moved piece, and no fork that one answer
     * to the check undoes.
     */
    @Test
    fun theWordsHoldForTheBoardTheMoveLeaves() {
        var board = Board("8/6bk/6p1/3QP1Bp/3N4/Pp6/1q4PP/2r3RK b - - 0 40")
        var said = describeEffects(board, board.parseSan("Qxa3")!!, limit = 9)
        assertTrue(said.toString(), "takes the open a-file" in said)
        // Kc1 steps off the d-file: the knight on d2, no longer pinned, guards the queen on b3.
        board = Board("2kr1b1r/p4ppp/1P6/3q4/8/1Q6/1P1N4/R2K4 w - - 7 28")
        said = describeEffects(board, board.parseSan("Kc1")!!, limit = 9)
        assertTrue(said.toString(), said.none { it.startsWith("defends") })
        // Qa4+ "forks" the king and the knight on b4, but Nc6 blocks and saves the knight.
        board = Board("r1bqk2r/ppp2ppp/3p4/8/1nBPP1n1/5N2/PP3PPP/R1BQR1K1 w kq - 0 12")
        said = describeEffects(board, board.parseSan("Qa4+")!!, limit = 9)
        assertTrue(said.toString(), said.none { it.startsWith("forks") })
    }
}
