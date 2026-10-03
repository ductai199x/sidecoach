package com.sidecoach.coach

import com.sidecoach.profile.Causes
import com.sidecoach.profile.Review
import java.io.File
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The coach teaches gently, offline and with a model (Tai, 2026-10-02: "make all messages a
 * welcoming experience where you learn"; "That is three times now" and "That one hurts" read as
 * scolding). The offline coach's fixed wordings hold no tally of failures and no blame; the AI
 * coach is told the same in its rules. `tests/test_coach_tone.py`.
 */
class CoachToneTest {

    /** What reads as scolding: counting the times something went wrong, "again", a verdict on
     * the player rather than a lesson. */
    private val scolding = Regex(
        "\\b(?:times now|again|keep missing|kept missing|that one hurts|not that one|" +
            "careless|you always|you never)\\b",
        RegexOption.IGNORE_CASE,
    )

    /** Every fixed wording the offline coach says about a mistake or a pattern. */
    private fun offlineWordings(): List<String> =
        FINE_TIGHTER + COST_BY_SEVERITY.values.flatten() +
            Causes.HABIT.values.flatMap { (skill, habit) -> listOf(skill, habit) }

    @Test
    fun noOfflineWordingScolds() {
        for (wording in offlineWordings()) assertFalse(wording, scolding.containsMatchIn(wording))
    }

    @Test
    fun aPatternIsASkillToWorkOnNeverACount() {
        val line = Review.patternLine(mapOf(Causes.MISSED_THREAT to 3), mapOf(Causes.MISSED_THREAT to 0.5))
        val (skill, habit) = Causes.HABIT.getValue(Causes.MISSED_THREAT)
        assertEquals("One thing to work on: $skill. A habit that helps: $habit.", line)
        assertFalse("no count of the mistakes: $line", Regex("\\d").containsMatchIn(line!!))
    }

    @Test
    fun theAiCoachIsToldToTeachGentlyInEveryPersona() {
        for (persona in PERSONAS.keys) {
            val system = systemMessage(persona = persona)
            assertTrue(persona, "Teach it gently and warmly" in system)
            assertTrue(persona, "never count how often something went wrong" in system)
            assertTrue(persona, "vary how you say it" in system)
            assertTrue(persona, "Never a question about them" in system)
            assertFalse("no teasing about a blunder: $persona", "tease" in system)
        }
    }

    /**
     * The corrections the model learns from (the shipped `coach/corrections.md`) teach by example:
     * none of them may show it a tally or a scolding.
     */
    @Test
    fun theAiCoachsWorkedExamplesHoldNoScolding() {
        val corrections = File("src/main/assets/coach/corrections.md")
        assertTrue(corrections.absolutePath, corrections.isFile)
        assertFalse(scolding.containsMatchIn(corrections.readText()))
    }
}
