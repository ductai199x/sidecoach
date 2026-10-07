package com.sidecoach.vision

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * What sits over the board, told apart by geometry: ported from `pieces.py`'s cover half.
 *
 * A dialog, a tooltip or the promotion picker hides cells; a highlight or a mark tints
 * them and leaves them readable. The difference is the neighbours: only an opaque overlay
 * makes two neighbouring squares the same colour, and only a block of cells fits one
 * translucent tint. Nothing here looks at hue.
 */
object Covers {

    /**
     * Cells whose background matches a 4-neighbour's, both away from their square colours.
     *
     * A translucent tint keeps light and dark squares apart under it; only an opaque
     * overlay makes two neighbouring squares the same colour. That marks a dialog, a
     * tooltip or the promotion picker as foreign whatever colour it has, while a badge or
     * a highlight on a single cell is never mistaken for one.
     */
    fun opaqueCover(backgrounds: Array<DoubleArray>, shifted: BooleanArray): BooleanArray {
        val covered = BooleanArray(64)
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                if (!shifted[r * 8 + c]) continue
                for ((rr, cc) in listOf(r - 1 to c, r + 1 to c, r to c - 1, r to c + 1)) {
                    if (rr !in 0..7 || cc !in 0..7 || !shifted[rr * 8 + cc]) continue
                    if (distance(backgrounds[r * 8 + c], backgrounds[rr * 8 + cc]) <= Pieces.COVER_TOLERANCE) {
                        covered[r * 8 + c] = true
                        break
                    }
                }
            }
        }
        return covered
    }

    /**
     * Whether two neighbouring cells sit under one translucent overlay.
     *
     * Under an overlay of colour T and opacity a, a cell of colour X shows (1 - a) X + a T,
     * so the difference between two neighbours shrinks to (1 - a) of what the theme gives
     * them. The opacity read from each colour channel must agree, be strong enough to hide
     * a piece, and imply an overlay colour that exists.
     */
    fun sameTint(x: DoubleArray, y: DoubleArray, ex: DoubleArray, ey: DoubleArray): Boolean {
        val alphas = ArrayList<Double>(3)
        for (c in 0 until 3) {
            val gap = ex[c] - ey[c]
            if (abs(gap) < 8.0) continue
            alphas.add(1.0 - (x[c] - y[c]) / gap)
        }
        if (alphas.size < 2) return false
        if (alphas.max() - alphas.min() > Pieces.TINT_ALPHA_SPREAD) return false
        var alpha = alphas.average()
        if (alpha < Pieces.TINT_MIN_ALPHA || alpha > 1.05) return false
        alpha = minOf(alpha, 1.0)
        for (c in 0 until 3) {
            val tint = (x[c] - (1.0 - alpha) * ex[c]) / alpha
            if (tint < -15.0 || tint > 270.0) return false
        }
        return true
    }

    /**
     * The groups of marked [cells], neighbours joined when [joined] says so (two cell indices),
     * that are shaped like a dialog: at least [Pieces.TINT_MIN_CELLS], two cells or more each
     * way, filling [fill] of their bounding box (`pieces._blocks`). A file of right-click marks
     * or a queued premove's squares make a line, and a highlight or two a small group; neither
     * is a dialog.
     */
    private fun blocks(cells: BooleanArray, joined: (Int, Int) -> Boolean, fill: Double): List<List<Int>> {
        val parent = IntArray(64) { it }
        fun find(start: Int): Int {
            var i = start
            while (parent[i] != i) {
                parent[i] = parent[parent[i]]
                i = parent[i]
            }
            return i
        }
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                if (!cells[r * 8 + c]) continue
                for ((rr, cc) in listOf(r to c + 1, r + 1 to c)) {
                    if (rr < 8 && cc < 8 && cells[rr * 8 + cc] && joined(r * 8 + c, rr * 8 + cc)) {
                        parent[find(r * 8 + c)] = find(rr * 8 + cc)
                    }
                }
            }
        }
        val groups = LinkedHashMap<Int, MutableList<Int>>()
        for (i in 0 until 64) if (cells[i]) groups.getOrPut(find(i)) { ArrayList() }.add(i)
        return groups.values.filter { members ->
            val rows = members.map { it / 8 }
            val cols = members.map { it % 8 }
            val height = rows.max() - rows.min() + 1
            val width = cols.max() - cols.min() + 1
            members.size >= Pieces.TINT_MIN_CELLS && height >= 2 && width >= 2 &&
                members.size >= fill * height * width
        }
    }

    /**
     * Cells in a dialog-shaped block ([blocks]) that share one translucent overlay: a dialog
     * fades in over a few frames, and while it is half transparent the pieces under it are
     * unreadable but the squares still differ, so the opaque test does not fire. A move
     * highlight tints one to three cells and a right-click mark a few, and stay below the
     * block size.
     */
    fun translucentCover(
        backgrounds: Array<DoubleArray>,
        expected: Array<DoubleArray>,
        shifted: BooleanArray,
    ): BooleanArray {
        val covered = BooleanArray(64)
        val tinted = { a: Int, b: Int -> sameTint(backgrounds[a], backgrounds[b], expected[a], expected[b]) }
        for (block in blocks(shifted, tinted, Pieces.TINT_BLOCK_FILL)) for (i in block) covered[i] = true
        // A highlighted square, or one with the dialog's text over its ring, does not fit
        // the tint, yet it lies under the same overlay when its neighbours do.
        repeat(2) {
            for (r in 0 until 8) {
                for (c in 0 until 8) {
                    if (covered[r * 8 + c] || !shifted[r * 8 + c]) continue
                    var around = 0
                    for ((rr, cc) in listOf(r - 1 to c, r + 1 to c, r to c - 1, r to c + 1)) {
                        if (rr in 0..7 && cc in 0..7 && covered[rr * 8 + cc]) around++
                    }
                    if (around >= 2) covered[r * 8 + c] = true
                }
            }
        }
        return covered
    }

    /**
     * The covered cells plus the body of each dialog: cells off their square colour that
     * connect (4-neighbours) to a covered block and, with it, make a dialog-shaped block
     * ([blocks]). A highlight or two beside a tooltip does not fill a rectangle with it; a
     * fading dialog's buttons and text do.
     */
    fun coverBody(covered: BooleanArray, shifts: DoubleArray): BooleanArray {
        val candidate = BooleanArray(64) { covered[it] || shifts[it] > Pieces.COVER_BODY_SHIFT }
        val out = covered.copyOf()
        for (block in blocks(candidate, { _, _ -> true }, Pieces.COVER_BODY_FILL)) {
            if (block.any { covered[it] }) for (i in block) out[i] = true
        }
        return out
    }

    /**
     * Whether an overlay of [colour] reaches into this cell from a covered neighbour: its
     * part of the cell is a rectangle anchored on a side or a corner. A piece is neither,
     * so a white piece next to a white dialog keeps reading as a piece.
     */
    fun overlayReaches(sil: Sil, colour: DoubleArray, size: Int = Squares.CELL): Boolean {
        val near = BooleanArray(size * size)
        var any = false
        for (i in near.indices) {
            if (!sil.mask[i]) continue
            val b = sil.inner[i * 3].toDouble()
            val g = sil.inner[i * 3 + 1].toDouble()
            val r = sil.inner[i * 3 + 2].toDouble()
            if (distance(doubleArrayOf(b, g, r), colour) <= Pieces.BG_TOLERANCE) {
                near[i] = true
                any = true
            }
        }
        if (!any) return false
        // A whole side of the inner patch.
        val sides = listOf(
            (0 until size).map { near[it] },                       // top row
            (0 until size).map { near[(size - 1) * size + it] },   // bottom row
            (0 until size).map { near[it * size] },                // left column
            (0 until size).map { near[it * size + size - 1] },     // right column
        )
        if (sides.any { side -> side.count { it }.toDouble() / size >= Pieces.OVERLAY_SIDE_FILL }) return true
        var on = 0
        for (v in near) if (v) on++
        if (on.toDouble() / (size * size) < Pieces.EMPTY_MAX_FRACTION) return false
        var minY = size; var maxY = -1; var minX = size; var maxX = -1
        for (i in near.indices) {
            if (!near[i]) continue
            val y = i / size
            val x = i % size
            if (y < minY) minY = y
            if (y > maxY) maxY = y
            if (x < minX) minX = x
            if (x > maxX) maxX = x
        }
        val boxArea = (maxY - minY + 1) * (maxX - minX + 1)
        if (on.toDouble() / boxArea < Pieces.OVERLAY_BLOCK_FILL) return false
        // A solid three-by-three block holding a corner.
        fun corner(y0: Int, x0: Int): Boolean {
            for (y in y0 until y0 + 3) for (x in x0 until x0 + 3) if (!near[y * size + x]) return false
            return true
        }
        return corner(0, 0) || corner(0, size - 3) || corner(size - 3, 0) || corner(size - 3, size - 3)
    }

    /** Whether the marked cells hold a 2x2 block, as a dialog does and a strip does not. */
    fun hasBlock(cells: BooleanArray): Boolean {
        for (r in 0 until 7) {
            for (c in 0 until 7) {
                if (cells[r * 8 + c] && cells[(r + 1) * 8 + c] &&
                    cells[r * 8 + c + 1] && cells[(r + 1) * 8 + c + 1]
                ) {
                    return true
                }
            }
        }
        return false
    }

    private fun distance(a: DoubleArray, b: DoubleArray): Double {
        var sum = 0.0
        for (i in 0 until 3) {
            val d = a[i] - b[i]
            sum += d * d
        }
        return sqrt(sum)
    }
}
