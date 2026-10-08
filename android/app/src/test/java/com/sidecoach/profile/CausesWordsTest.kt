package com.sidecoach.profile

import com.sidecoach.coach.fixture
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The cause a player's own words name, as the desktop's `causes.from_words` gave it for the same
 * words (profile.json's `cause_tables.from_words`: tests/test_profile.py's CLOCK_SAID and
 * CLOCK_NOT_SAID among them). "This time I missed the fork" filed the mistake as time pressure
 * (the audit, 2026-10-07; `test_only_words_about_the_clock_say_time_pressure`). The device test
 * `ProfileTest` replays the same table with the rest of the causes.
 */
class CausesWordsTest {
    @Test
    fun onlyWordsAboutTheClockSayTimePressure() {
        val words = fixture("profile/profile.json", setOf("cause_tables"))
            .getJSONObject("cause_tables").getJSONArray("from_words")
        val failures = mutableListOf<String>()
        for (i in 0 until words.length()) {
            val o = words.getJSONObject(i)
            val want = if (o.isNull("cause")) null else o.getString("cause")
            val got = Causes.fromWords(o.getString("text"))
            if (got != want) failures.add("'${o.getString("text")}': kotlin $got, python $want")
        }
        assertEquals(emptyList<String>(), failures)
        assertEquals(Causes.MISSED_THREAT, Causes.fromWords("this time I missed the fork"))
    }
}
