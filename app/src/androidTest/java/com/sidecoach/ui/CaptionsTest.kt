package com.sidecoach.ui

import android.graphics.Rect
import android.graphics.Typeface
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

/**
 * The captions' two pieces of arithmetic: how much of a line is out at a moment of its speech
 * (whole words, at the voice's pace), and where a box dropped over the board goes instead.
 */
@RunWith(AndroidJUnit4::class)
class CaptionsTest {

    @Test
    fun theWordsComeOutWholeAtThePaceOfTheVoice() {
        val line = "Spot on -- rooks belong on open files."
        assertEquals(0, Captions.revealed(line, 0, 2000))
        // A quarter of the way through, the word under way is shown whole.
        val quarter = Captions.revealed(line, 500, 2000)
        assertTrue(quarter in 1 until line.length)
        assertTrue("cut mid-word: '${line.substring(0, quarter)}'", quarter == line.length || line[quarter] == ' ')
        // Later never shows less.
        var last = 0
        for (t in 0L..2000L step 50) {
            val n = Captions.revealed(line, t, 2000)
            assertTrue(n >= last)
            last = n
        }
        assertEquals(line.length, Captions.revealed(line, 2000, 2000))
        // A line not spoken is out at once.
        assertEquals(line.length, Captions.revealed(line, 0, 0))
    }

    /** Tai, 2026-09-29: "why are the pieces and their names and square names and SAN
     * annotations not highlighted". */
    @Test
    fun movesSquaresAndPiecesAreSetOffAndTheProseIsNot() {
        val text = "Play Be3+. Your bishop from f4 checks the King; G4 was a miss, and a plan is O-O."
        val styled = Captions.styled(text, 0xEB1F1F1F.toInt()) as Spanned
        fun marks(word: String): Pair<Boolean, Int?> {
            val at = text.indexOf(word)
            val bold = styled.getSpans(at, at + word.length, StyleSpan::class.java).any { it.style == Typeface.BOLD }
            val colour = styled.getSpans(at, at + word.length, ForegroundColorSpan::class.java).firstOrNull()?.foregroundColor
            return bold to colour
        }
        val gold = 0xFFFFD479.toInt()
        for (move in listOf("Be3+", "f4", "G4", "O-O")) assertEquals(move, true to gold, marks(move))
        for (piece in listOf("bishop", "King")) assertEquals(piece, true to null, marks(piece))
        for (word in listOf("Play", "checks", "plan")) assertEquals(word, false to null, marks(word))
        // " a plan": the article is no square.
        assertEquals(false to null, marks(" a "))
        // On a light box the moves are a deep amber, not gold.
        val light = Captions.styled("Nf3", 0xFFF2F2F2.toInt()) as Spanned
        assertEquals(0xFF8A5300.toInt(), light.getSpans(0, 3, ForegroundColorSpan::class.java).single().foregroundColor)
    }

    @Test
    fun aBoxDroppedOnTheBoardGoesAboveOrBelowIt() {
        val screen = Rect(0, 0, 1080, 2400)
        val board = Rect(0, 600, 1080, 1680)
        val (w, h) = 900 to 200
        // Clear of the board: where it was dropped.
        assertEquals(90 to 2000, Captions.clear(90, 2000, w, h, screen, board, 16))
        // Over the top half of the board: above it.
        val (_, up) = Captions.clear(90, 700, w, h, screen, board, 16)
        assertEquals(600 - 16 - h, up)
        // Over the bottom half: below it.
        val (_, down) = Captions.clear(90, 1500, w, h, screen, board, 16)
        assertEquals(1680 + 16, down)
        for ((x, y) in listOf(90 to 700, 90 to 1500, 0 to 1000)) {
            val (cx, cy) = Captions.clear(x, y, w, h, screen, board, 16)
            assertFalse(Rect.intersects(Rect(cx, cy, cx + w, cy + h), board))
        }
        // Off the screen: back on it.
        assertEquals(1080 - w to 0, Captions.clear(5000, -300, w, h, screen, null, 16))
        // No room above: below.
        val tall = Rect(0, 100, 1080, 1680)
        assertEquals(1680 + 16, Captions.clear(90, 150, w, h, screen, tall, 16).second)
    }
}
