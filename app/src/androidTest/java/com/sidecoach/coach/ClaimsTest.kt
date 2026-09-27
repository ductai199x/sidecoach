package com.sidecoach.coach

import android.util.Log
import androidx.test.ext.junit.runners.AndroidJUnit4
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
        val root = fixture("chess/claims.json")

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

    /**
     * Every call the desktop's own `tests/test_claims.py` makes, replayed with the desktop's
     * answers (TG-02, TG-03): its positions, claims and drafts -- a cheaper attacker, a threat
     * after the pass, who can take back, a pinned piece, a fabricated pin, a fork the best move
     * makes, a false open file or passed pawn, what a tactic must win -- and its extraction
     * requests, the retry without thinking and the failures. And a checked claim as the model
     * is shown it, the text in Python's repr (the parity audit cm-01).
     */
    @Test
    fun theDesktopsClaimTestsAndTheCheckedLinesMatch() {
        val root = fixture("chess/claims.json")
        val tests = root.getJSONObject("tests")
        assertTrue("the desktop's claim tests ran", tests.getJSONArray("ran").length() >= 60)

        val verify = tests.getJSONArray("verify")
        for (i in 0 until verify.length()) {
            val one = verify.getJSONObject(i)
            val board = Board(one.getString("fen"))
            val facts = if (one.isNull("facts")) null else factsFrom(one.getJSONObject("facts"))
            val engine = if (one.isNull("engine")) null else recorded(one.getJSONObject("engine"))
            check(
                "${one.getString("test")} verify on ${one.getString("fen")}",
                verifyClaims(claimList(one.getJSONArray("claims")), board, facts, engine, one.getInt("depth"))
                    .map { showVerdict(it) },
                verdictStrings(one.getJSONArray("verdicts")),
            )
        }
        val drafts = tests.getJSONArray("drafts")
        for (i in 0 until drafts.length()) {
            val one = drafts.getJSONObject(i)
            val after = strings(one.getJSONArray("after")).map { Board(it) }
            val us = if (one.isNull("us")) null else one.getBoolean("us")
            check(
                "${one.getString("test")} draft \"${one.getString("text")}\"",
                checkDraft(one.getString("text"), Board(one.getString("fen")), after, us)
                    .map { "${it.text}|${it.reason}" },
                problemStrings(one.getJSONArray("problems")),
            )
        }
        val extract = tests.getJSONArray("extract")
        for (i in 0 until extract.length()) {
            val one = extract.getJSONObject(i)
            val calls = one.getJSONArray("calls")
            val provider = Replaying(calls)
            val found = extractClaims(
                one.getString("text"),
                Board(one.getString("fen")),
                provider,
                thinking = one.getBoolean("thinking"),
                maxTokens = one.getInt("max_tokens"),
            )
            val label = "${one.getString("test")} extract \"${one.getString("text").take(30)}\""
            check("$label calls", provider.asked, (0 until calls.length()).map { k ->
                val call = calls.getJSONObject(k)
                val messages = call.getJSONArray("messages")
                listOf(
                    call.getString("system"),
                    (0 until messages.length()).map { m ->
                        messages.getJSONObject(m).let { "${it.getString("role")}|${it.getString("content")}" }
                    },
                    call.getInt("max_tokens"),
                    call.getDouble("temperature"),
                    call.getBoolean("thinking"),
                ).toString()
            })
            check("$label claims", found.map { showClaim(it) }, claimStrings(one.getJSONArray("claims")))
        }
        val lines = root.getJSONArray("lines")
        for (i in 0 until lines.length()) {
            val one = lines.getJSONObject(i)
            val verdict = verifyClaims(claimList(JSONArray().put(one.getJSONObject("claim"))), Board()).single()
            check("the checked line", CoachRunner.claimLine(verdict), one.getString("line"))
        }
        assertTrue("$mismatches mismatches\n" + failures.joinToString("\n"), failures.isEmpty())
    }

    /** A model that answers as the desktop's did, call by call, raising where it raised. */
    private class Replaying(private val calls: JSONArray) : com.sidecoach.llm.LLMProvider {
        override val name = "replaying"
        val asked = ArrayList<String>()

        override fun healthcheck(): Pair<Boolean, String> = true to "replaying"

        override fun chat(
            system: String,
            messages: List<com.sidecoach.llm.ChatMessage>,
            maxTokens: Int,
            temperature: Double,
            thinking: Boolean,
        ): com.sidecoach.llm.Completion {
            val call = calls.getJSONObject(asked.size)
            asked.add(listOf(system, messages.map { "${it.role}|${it.content}" }, maxTokens, temperature, thinking).toString())
            if (call.has("raises")) throw RuntimeException(call.getString("raises"))
            return com.sidecoach.llm.Completion(text = call.getString("reply"))
        }
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
