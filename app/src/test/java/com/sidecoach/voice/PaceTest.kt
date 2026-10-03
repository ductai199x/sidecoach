package com.sidecoach.voice

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The voice follows the player's pace ([PacePlan]): what is said, held to a fast line's end, or
 * left to the text, in the sessions the desktop's `tools/voice_sim.py` tuned it on. The desktop's
 * `tests/test_voice_pace.py`, case for case, on a clock the test moves.
 */
class PaceTest {
    private class Clock(var now: Double = 1000.0) {
        fun advance(s: Double) {
            now += s
        }
    }

    private fun planAt(): Pair<PacePlan, Clock> {
        val clock = Clock()
        val plan = PacePlan { clock.now }
        plan.newPosition()
        return plan to clock
    }

    private fun said(plan: PacePlan): List<String> {
        val out = ArrayList<String>()
        while (true) out.add(plan.nextLine()?.text ?: break)
        return out
    }

    @Test
    fun slowPlayHearsEveryWordWhole() {
        val (plan, clock) = planAt()
        clock.advance(20.0)
        plan.playerMoved(1, 20.0)
        clock.advance(1.6)
        assertEquals(Pace.SAY, plan.offer(1, "post_move", 0, 0, "Excellent: the only move.", streamed = true))
        assertEquals(Pace.SAY, plan.offer(1, "post_move", 0, 1, "It pulls the defender away.", streamed = true))
        assertEquals(listOf("Excellent: the only move.", "It pulls the defender away."), said(plan))
    }

    @Test
    fun aFastLineIsQuietThenOneWordOnTheFindAtItsEnd() {
        val (plan, clock) = planAt()
        clock.advance(30.0)
        plan.playerMoved(1, 30.0)
        clock.advance(1.5)
        plan.opponentMoved(2)
        plan.playerMoved(3, 1.5)
        clock.advance(0.1)
        assertEquals(Pace.HOLD, plan.offer(1, "post_move", 0, 0, "Excellent: the only move.", streamed = true))
        assertEquals(Pace.TEXT, plan.offer(1, "post_move", 0, 1, "It pulls the defender.", streamed = true))
        assertEquals(Pace.TEXT, plan.offer(2, "post_move", 2, 0, "Exactly.", streamed = false))
        assertEquals("Excellent: the only move.", plan.heldText())
        plan.tick()
        assertEquals("not while the line is played", emptyList<String>(), said(plan))
        clock.advance(Pace.LINE_END_S)
        plan.tick()
        val line = plan.nextLine()
        assertNotNull(line)
        assertEquals("Excellent: the only move.", line!!.text)
        assertTrue(line.prepared)
        assertEquals(emptyList<String>(), said(plan))
    }

    @Test
    fun theLineWordIsSaidAtOnceWhenTheNextPuzzleComes() {
        val (plan, clock) = planAt()
        clock.advance(30.0)
        plan.playerMoved(1, 30.0)
        clock.advance(1.0)
        plan.playerMoved(3, 1.0)
        clock.advance(0.6)
        assertEquals(Pace.HOLD, plan.offer(1, "post_move", 0, 0, "Well found.", streamed = true))
        clock.advance(1.0)
        plan.newPosition()
        assertEquals(listOf("Well found."), said(plan))
    }

    @Test
    fun aLineWordTooOldIsDropped() {
        val (plan, clock) = planAt()
        clock.advance(30.0)
        plan.playerMoved(1, 30.0)
        clock.advance(1.0)
        plan.playerMoved(3, 1.0)
        clock.advance(0.5)
        assertEquals(Pace.HOLD, plan.offer(1, "post_move", 0, 0, "Well found.", streamed = true))
        for (ply in listOf(5, 7, 9, 11, 13, 15)) {
            clock.advance(1.5)
            plan.playerMoved(ply, 1.5)
        }
        clock.advance(Pace.LINE_END_S)
        plan.tick()
        assertEquals(emptyList<String>(), said(plan))
    }

    @Test
    fun aLessonIsSaidAfterAPremoveItsFirstSentenceOnly() {
        val (plan, clock) = planAt()
        clock.advance(25.0)
        plan.playerMoved(1, 25.0)
        clock.advance(0.8)
        plan.playerMoved(3, 0.8)
        clock.advance(1.1)
        assertEquals(Pace.SAY, plan.offer(1, "post_mistake", 0, 0, "exf5 was the move.", streamed = true))
        assertEquals(Pace.TEXT, plan.offer(1, "post_mistake", 0, 1, "It wins the queen.", streamed = true))
        assertEquals(listOf("exf5 was the move."), said(plan))
    }

    @Test
    fun aLessonOnAMoveTakenBackIsNeverSaid() {
        val (plan, clock) = planAt()
        clock.advance(40.0)
        plan.playerMoved(1, 40.0)
        clock.advance(1.9)
        assertEquals(Pace.SAY, plan.offer(1, "post_mistake", 0, 0, "Rg8 was the move.", streamed = true))
        assertNotNull(plan.nextLine())
        plan.voiceState(busy = true, waiting = 0)
        clock.advance(3.0)
        assertTrue("the lesson playing is cut: it names the answer", plan.takenBack(0))
        plan.stopped()
        assertEquals(Pace.TEXT, plan.offer(1, "post_mistake", 0, 1, "It mates.", streamed = true))
    }

    @Test
    fun theAnswerToAQuestionIsSaidWholeWhateverIsPlayed() {
        val (plan, clock) = planAt()
        clock.advance(30.0)
        assertEquals(Pace.SAY, plan.offer(1, "user_ask", 0, 0, "Play Qa8+.", streamed = true))
        assertNotNull(plan.nextLine())
        plan.voiceState(busy = true, waiting = 0)
        clock.advance(5.0)
        assertFalse("the answer is not cut", plan.playerMoved(1, 35.0))
        assertEquals(Pace.SAY, plan.offer(1, "user_ask", 0, 1, "It deflects the rook.", streamed = true))
        assertEquals(listOf("It deflects the rook."), said(plan))
    }

    @Test
    fun aHintIsSaidOnlyWhileItsPositionIsOnTheBoard() {
        val (plan, clock) = planAt()
        clock.advance(45.0)
        assertEquals(Pace.SAY, plan.offer(1, "stuck", 0, 0, "Only one move holds.", streamed = true))
        assertNotNull(plan.nextLine())
        plan.voiceState(busy = true, waiting = 0)
        clock.advance(1.0)
        assertTrue("a hint whose position is gone stops", plan.playerMoved(1, 46.0))
        plan.stopped()
        assertEquals(Pace.TEXT, plan.offer(1, "stuck", 0, 1, "Which check?", streamed = true))
        assertEquals(Pace.TEXT, plan.offer(2, "stuck", 0, 0, "Look at f7.", streamed = true))
    }

    @Test
    fun blitzSaysTheLessonsAndLeavesThePraiseToTheText() {
        val (plan, clock) = planAt()
        var ply = 1
        repeat(3) {
            clock.advance(3.0)
            plan.playerMoved(ply, 2.8)
            ply += 2
        }
        clock.advance(0.15)
        assertEquals(Pace.TEXT, plan.offer(1, "post_move", ply - 3, 0, "Exactly.", streamed = false))
        clock.advance(2.8)
        plan.playerMoved(ply, 2.8)
        clock.advance(0.15)
        assertEquals(Pace.SAY, plan.offer(2, "post_mistake", ply - 1, 0, "Rg8 was stronger.", streamed = false))
    }

    @Test
    fun aModeratePaceHearsItsShortWords() {
        val (plan, clock) = planAt()
        var ply = 1
        for (think in listOf(5.0, 4.5, 5.5)) {
            clock.advance(think)
            plan.playerMoved(ply, think)
            ply += 2
        }
        clock.advance(0.15)
        assertEquals(Pace.SAY, plan.offer(1, "post_move", ply - 3, 0, "Exactly.", streamed = false))
    }

    @Test
    fun aMoveAfterLongThoughtIsSlowWhateverThePaceWas() {
        val (plan, clock) = planAt()
        for (ply in listOf(1, 3, 5, 7)) {
            clock.advance(1.2)
            plan.playerMoved(ply, 1.2)
        }
        clock.advance(22.0)
        plan.playerMoved(9, 22.0)
        clock.advance(1.6)
        assertEquals(Pace.SAY, plan.offer(1, "post_move", 8, 0, "Good square.", streamed = true))
    }

    @Test
    fun noWordIsSaidLate() {
        val (plan, clock) = planAt()
        clock.advance(20.0)
        plan.playerMoved(1, 20.0)
        clock.advance(Pace.PRAISE_FRESH_S + 0.5)
        assertEquals(Pace.TEXT, plan.offer(1, "post_move", 0, 0, "Exactly.", streamed = false))
        clock.advance(20.0)
        plan.playerMoved(3, 20.0)
        clock.advance(Pace.LESSON_FRESH_S + 1)
        assertEquals(Pace.TEXT, plan.offer(2, "post_mistake", 2, 0, "Late lesson.", streamed = true))
    }

    @Test
    fun whatWaitsIsJudgedAgainWhenThePlayerMoves() {
        val (plan, clock) = planAt()
        clock.advance(20.0)
        plan.playerMoved(1, 20.0)
        clock.advance(1.6)
        plan.offer(1, "post_move", 0, 0, "One.", streamed = true)
        plan.offer(1, "post_move", 0, 1, "Two.", streamed = true)
        assertNotNull(plan.nextLine())
        assertNotNull(plan.nextLine())
        plan.voiceState(busy = true, waiting = 1)
        clock.advance(4.0)
        assertFalse("the sentence playing finishes", plan.playerMoved(3, 4.0))
        assertTrue("the one waiting is dropped", plan.waitingStale())
        assertNull(plan.nextLine())
    }

    @Test
    fun theMedianIsPythons() {
        assertEquals(2.5, Pace.median(listOf(1.0, 4.0, 2.0, 3.0)), 0.0)
        assertEquals(2.0, Pace.median(listOf(3.0, 1.0, 2.0)), 0.0)
    }
}
