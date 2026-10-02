package com.sidecoach.coach

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/** The coach's wordings drawn like marbles from a bag ([Bags]): the desktop's `tests/test_bag.py`. */
class BagsTest {

    /** `Bags(random.Random(seed))`: one seeded generator for the bags' shuffles. */
    private fun bags(seed: Long): Bags {
        val rng = java.util.Random(seed)
        return Bags { it.shuffle(rng) }
    }

    @Test
    fun everyWordingOnceBeforeAnyComesAgainAndNeverTwiceRunning() {
        for (seed in 0L until 50L) {
            val bags = bags(seed)
            val said = List(WORDINGS.size * 8) { bags.draw("keep", WORDINGS) }
            for (start in said.indices step WORDINGS.size) {
                assertEquals("seed $seed", WORDINGS.sorted(), said.subList(start, start + WORDINGS.size).sorted())
            }
            // Across a refill too: the new bag never opens with the wording just said.
            assertTrue("seed $seed: $said", said.zipWithNext().all { (a, b) -> a != b })
        }
    }

    @Test
    fun eachKeyIsItsOwnBagAndAListOfOneIsSaidAgain() {
        val bags = bags(1)
        val first = List(2) { bags.draw("cost", listOf("a", "b", "c")) }
        repeat(3) { bags.draw("fine", listOf("a", "b", "c")) }
        assertEquals(listOf("a", "b", "c"), (first + bags.draw("cost", listOf("a", "b", "c"))).sorted())
        assertEquals(List(3) { "only" }, List(3) { bags.draw("one", listOf("only")) })
        assertThrows(IllegalArgumentException::class.java) { bags.draw("none", emptyList()) }
    }

    @Test
    fun aWordingNoLongerOfferedLeavesTheBag() {
        val bags = bags(2)
        bags.draw("praise", listOf("a", "b", "c", "d"))
        val said = List(6) { bags.draw("praise", listOf("a", "b")) }
        assertEquals(setOf("a", "b"), said.toSet())
    }

    /**
     * The first hint is written for every position and said in few: written, it only peeks; said,
     * it takes.
     */
    @Test
    fun aPeekIsTheNextDrawAndUsesNothingUpUntilTaken() {
        val bags = bags(3)
        val peeked = List(4) { bags.peek("lead", WORDINGS) }
        assertEquals(1, peeked.toSet().size)
        bags.take("lead", peeked[0])
        val rest = List(WORDINGS.size - 1) { bags.draw("lead", WORDINGS) }
        assertEquals(WORDINGS.sorted(), (listOf(peeked[0]) + rest).sorted())
        assertNotEquals(rest.last(), bags.peek("lead", WORDINGS))
    }

    private companion object {
        val WORDINGS = listOf("keep this one", "the lesson to keep", "worth remembering", "a rule", "remember")
    }
}
