package com.sidecoach.coach

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.sidecoach.chess.Board
import com.sidecoach.events.Line
import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * What the player claims and what the coach drafts, held to the desktop's answer.
 *
 * `coach/claims.py` runs in both directions and only the extraction uses a model, so all
 * three public entry points are compared exactly: `parse_claims` on recorded model replies,
 * `verify_claims` on every claim of every position (bare, with the facts, with an engine,
 * and with facts that belong to another position), and `check_draft` on the coach's own
 * sentences, with and without the line it was given. The engine is the recorded one the
 * generator used -- a deterministic stub whose lines are in the fixture.
 */
@RunWith(AndroidJUnit4::class)
class ClaimsTest {

    private val failures = ArrayList<String>()
    private var mismatches = 0

    private fun check(what: String, mine: Any?, theirs: Any?) {
        if (mine != theirs) {
            mismatches += 1
            if (failures.size < 8) failures.add("$what:\n  kotlin: $mine\n  python: $theirs")
        }
    }

    @Test
    fun everyClaimAndDraftMatchesThePython() {
        val context = InstrumentationRegistry.getInstrumentation().context
        val text = context.assets.open("chess/claims.json").bufferedReader().use { it.readText() }
        val root = JSONObject(text)

        // The prompt is sent to a model, so a drifted copy would not fail anywhere else.
        check("extraction system prompt", EXTRACT_SYSTEM, root.getString("system"))

        val replies = root.getJSONArray("replies")
        for (i in 0 until replies.length()) {
            val reply = replies.getJSONObject(i)
            val words = reply.getString("text")
            check(
                "parse_claims of ${words.take(40)}",
                parseClaims(words).map { showClaim(it) },
                claimStrings(reply.getJSONArray("claims")),
            )
        }

        val cases = root.getJSONArray("cases")
        var verified = 0
        var drafted = 0
        val began = System.currentTimeMillis()
        for (i in 0 until cases.length()) {
            val case = cases.getJSONObject(i)
            val fen = case.getString("fen")
            val board = Board(fen)
            val engine = recorded(case.getJSONObject("engine"))
            val facts = factsFrom(case.getJSONObject("facts"))
            val stale = factsFrom(case.getJSONObject("stale_facts"))
            val claims = claimList(case.getJSONArray("claims"))
            verified += claims.size * 4

            check(
                "$fen bare",
                verifyClaims(claims, board).map { showVerdict(it) },
                verdictStrings(case.getJSONArray("bare")),
            )
            check(
                "$fen with facts",
                verifyClaims(claims, board, facts = facts).map { showVerdict(it) },
                verdictStrings(case.getJSONArray("with_facts")),
            )
            check(
                "$fen with engine",
                verifyClaims(claims, board, engine = engine, depth = 14)
                    .map { showVerdict(it) },
                verdictStrings(case.getJSONArray("with_engine")),
            )
            check(
                "$fen with stale facts",
                verifyClaims(claims, board, facts = stale, engine = engine, depth = 14)
                    .map { showVerdict(it) },
                verdictStrings(case.getJSONArray("with_stale")),
            )

            val after = case.getJSONArray("after").let { list ->
                (0 until list.length()).map { Board(list.getString(it)) }
            }
            val drafts = case.getJSONArray("drafts")
            for (k in 0 until drafts.length()) {
                val draft = drafts.getJSONObject(k)
                val words = draft.getString("text")
                drafted += 2
                check(
                    "$fen draft \"$words\"",
                    checkDraft(words, board).map { "${it.text}|${it.reason}" },
                    problemStrings(draft.getJSONArray("now")),
                )
                check(
                    "$fen draft along the line \"$words\"",
                    checkDraft(words, board, after).map { "${it.text}|${it.reason}" },
                    problemStrings(draft.getJSONArray("line")),
                )
            }
        }
        val took = System.currentTimeMillis() - began
        Log.i(TAG, "${cases.length()} positions, $verified verdicts, $drafted drafts, $took ms")
        assertTrue("$mismatches mismatches\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    /** Round 2: "your opponent's best is Qf7, taking the half-open f-file" is theirs (desktop 47138dd). */
    @Test
    fun aHalfOpenFileIsReadAsWhoseTheSentenceSays() {
        val board = Board("4k3/8/8/8/8/8/5P2/4K3 w - - 0 1") // a white pawn on f2 only
        assertTrue(checkDraft("Your opponent's rook takes the half-open f-file.", board).isEmpty())
        assertTrue(checkDraft("The f-file is half-open for you.", board).isNotEmpty())
        assertTrue(checkDraft("The half-open f-file is theirs to use.", board).isEmpty())
    }

    /**
     * On the opponent's turn "your" is still the player being coached: read as the side to move,
     * "your knight on f3" was flagged false and a true draft thrown away (desktop 2ec1747).
     */
    @Test
    fun yourIsThePlayerOnTheOpponentsTurn() {
        // test_claims' ITALIAN with Black, the opponent, to move; the player is White.
        val board = Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4")
        assertTrue(checkDraft("Your knight on f3 eyes e5.", board, us = com.sidecoach.chess.WHITE).isEmpty())
        val problems = checkDraft("Your knight on c6 defends e5.", board, us = com.sidecoach.chess.WHITE)
        assertTrue(problems.toString(), problems.isNotEmpty() && problems[0].reason == "the knight on c6 is your opponent's, not yours")
        assertTrue(checkDraft("Their knight on c6 defends e5.", board, us = com.sidecoach.chess.WHITE).isEmpty())
    }

    /**
     * The bench's gold false claims, from explanations of lines (2026-09-25): a rook "forced onto
     * f7" where other moves were legal, files "kept open" with pawns on them. A claim no frame
     * applies to is left alone (desktop 00894e0).
     */
    @Test
    fun whatALineForcesOrKeepsOpenIsChecked() {
        val white = com.sidecoach.chess.WHITE
        val board = Board("8/p1q1k2K/2p1p3/3p4/3P1R1P/6r1/1r6/8 w - - 0 49")
        val after = ArrayList<Board>()
        val line = board.copy()
        for (san in listOf("Rf1", "Qd6")) {
            line.push(line.parseSan(san)!!)
            after.add(line.copy())
        }
        val text = "Black then uses the check on d6 to force your rook onto f7."
        var problems = checkDraft(text, board, after = after, us = white)
        assertEquals(listOf("force your rook onto f7"), problems.map { it.text })
        val rad8 = Board("r6r/pQ3pkp/4p1p1/P2BPb2/3b4/8/1P3PPP/R4RK1 b - - 0 21")
        problems = checkDraft("Rad8 keeps the b-file and e-file open.", rad8)
        assertEquals(listOf("the b-file is not open; a pawn stands on it"), problems.map { it.reason })
        val lone = Board("8/8/8/8/8/5k2/8/R3K3 w - - 0 1")
        assertEquals(emptyList<DraftProblem>(), checkDraft("Ra3+ forces their king to move.", lone, us = white))
        val blocked = Board("8/8/8/8/5n2/5k2/8/R3K3 w - - 0 1")
        problems = checkDraft("Ra3+ forces their king to move.", blocked, us = white)
        assertTrue(problems.toString(), problems.isNotEmpty() && "can be blocked" in problems[0].reason)
        // No check anywhere: about another moment, left alone.
        val quiet = Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        assertEquals(emptyList<DraftProblem>(), checkDraft("The check forced their king away.", quiet, us = white))
        // "The g-file has no pawns" of a half-open file (the pilot, 2026-09-26; desktop cc06e83).
        val g6 = Board("rn2k3/pp2bp2/2p3p1/4Q2p/P1P5/1P3BK1/7R/RN4q1 w q - 7 33")
        problems = checkDraft("The g-file has no pawns, so use it.", g6, us = white)
        assertEquals(listOf("the g-file has a pawn on it"), problems.map { it.reason })
        assertEquals(emptyList<DraftProblem>(), checkDraft("The d-file has no pawns, so use it.", g6, us = white))
        // "No pawns of yours on it" is a half-open file: the template says it, and it is true
        // (desktop ba861d6).
        val half = "Rg2 was better -- the g-file has no pawns of yours on it."
        assertEquals(emptyList<DraftProblem>(), checkDraft(half, g6, us = white))
    }

    /**
     * The locked test set's audit (2026-09-26): false claims the checker now catches, each beside
     * the true sentence it must not fault (desktop 4365dfa).
     */
    @Test
    fun theTestAuditsFalseClaimsAreCaughtAndTheirTrueTwinsPass() {
        val white = com.sidecoach.chess.WHITE
        val black = com.sidecoach.chess.BLACK
        // "Qf2+ drives the King to h3": Kh1 was legal too; judged on Qf2+'s own frame.
        val box = Board("6k1/5pp1/p2p4/1p3Pb1/2p1r1q1/P1P1p3/1P5K/4R3 b - - 1 48")
        val after = ArrayList<Board>()
        val line = box.copy()
        for (san in listOf("Qh4+", "Kg2", "Qf2+", "Kh3", "Rh4#")) {
            line.push(line.parseSan(san)!!)
            after.add(line.copy())
        }
        var problems = checkDraft("Qf2+ drives the King to h3.", box, after = after, us = black)
        assertTrue(problems.toString(), problems.isNotEmpty() && "nothing forces the king to h3" in problems[0].reason)
        // "Kh8 is the only king move" where four were legal; "the only move that keeps your edge"
        // is the stakes, and is not checked.
        val q = Board("8/6k1/8/8/3Q4/8/5PPP/6K1 b - - 0 1") // Qd4+ given: the king has five
        problems = checkDraft("After the check, Kh7 is the only king move.", q, us = white)
        assertTrue(
            problems.toString(),
            problems.isNotEmpty() && problems[0].reason == "Kh7 is not the only king move: others are legal",
        )
        assertEquals(emptyList<DraftProblem>(), checkDraft("Kh7 was the only move that keeps it level.", q, us = white))
        // "Ne4 ... but it allows exf6" where exf6 cannot follow Ne4; across clauses, not judged.
        val n = Board("r1bqkb1r/pppp1ppp/2n2n2/4P3/2B5/8/PPPP1PPP/RNBQK1NR b KQkq - 0 4")
        problems = checkDraft("Ne4 attacks f2, but it allows exf6.", n, us = black)
        assertTrue(problems.toString(), problems.isNotEmpty() && "exf6 cannot be played after Ne4" in problems[0].reason)
        val fine = "After Nd5, Bxd5 wins a pawn, while your move allowed exf6."
        assertEquals(emptyList<DraftProblem>(), checkDraft(fine, n, us = black))
        // "Your queen checks" with no queen; "it took your bishop" is the story, not checked.
        val none = Board("k7/8/8/8/8/8/8/K5R1 b - - 0 1")
        problems = checkDraft("Your queen checks the king.", none, us = black)
        assertTrue(problems.toString(), problems.isNotEmpty() && problems[0].reason == "you have no queen")
        assertEquals(emptyList<DraftProblem>(), checkDraft("That exchange took your bishop.", none, us = black))
    }

    /** The stub the generator used: the same lines for the same position, read from the file. */
    private fun recorded(seen: JSONObject): ClaimEngine = ClaimEngine { board, _ ->
        val entry = seen.optJSONArray(board.fen()) ?: JSONArray()
        (0 until entry.length()).map { i ->
            val one = entry.getJSONObject(i)
            Line(
                moves = one.getJSONArray("moves").let { a -> List(a.length()) { a.getString(it) } },
                depth = one.getInt("depth"),
                expectedScore = one.getDouble("expected_score"),
                scoreCp = if (one.isNull("score_cp")) null else one.getInt("score_cp"),
            )
        }
    }

    private fun factsFrom(o: JSONObject): PositionFacts = PositionFacts(
        fen = o.getString("fen"),
        sideToMove = o.getString("side_to_move"),
        phase = o.getString("phase"),
        candidates = strings(o.getJSONArray("candidates")),
        threats = strings(o.getJSONArray("threats")),
    )

    private fun claimList(array: JSONArray): List<Claim> = (0 until array.length()).map { i ->
        val o = array.getJSONObject(i)
        Claim(
            kind = ClaimKind.of(o.getString("kind"))!!,
            subject = if (o.isNull("subject")) null else o.getString("subject"),
            target = if (o.isNull("target")) null else o.getString("target"),
            move = if (o.isNull("move")) null else o.getString("move"),
            text = o.getString("text"),
        )
    }

    private fun showClaim(claim: Claim): String =
        "${claim.kind.slug}|${claim.subject}|${claim.target}|${claim.move}|${claim.text}"

    private fun claimStrings(array: JSONArray): List<String> = (0 until array.length()).map { i ->
        val o = array.getJSONObject(i)
        "${o.getString("kind")}|${orNull(o, "subject")}|${orNull(o, "target")}|" +
            "${orNull(o, "move")}|${o.getString("text")}"
    }

    private fun orNull(o: JSONObject, key: String): String? =
        if (o.isNull(key)) null else o.getString(key)

    private fun showVerdict(verdict: Verdict): String =
        "${verdict.result.slug}|${verdict.reason}"

    private fun verdictStrings(array: JSONArray): List<String> = (0 until array.length()).map { i ->
        val o = array.getJSONObject(i)
        "${o.getString("result")}|${o.getString("reason")}"
    }

    private fun problemStrings(array: JSONArray): List<String> = (0 until array.length()).map { i ->
        val o = array.getJSONObject(i)
        "${o.getString("text")}|${o.getString("reason")}"
    }

    private fun strings(array: JSONArray): List<String> =
        (0 until array.length()).map { array.getString(it) }

    companion object {
        private const val TAG = "sidecoach.coach"
    }
}
