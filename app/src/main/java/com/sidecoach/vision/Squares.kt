package com.sidecoach.vision

import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Cell geometry for a locked board, ported from `sidecoach/vision/squares.py`.
 *
 * Rows and columns are screen order: row 0 is the top of the board on screen, column 0 the
 * left. Which chess square a cell holds depends on which side sits at the bottom, and that
 * is detected per game, never assumed.
 */
object Squares {

    const val CELL = 48

    /** The crop the embedder reads, twice the working cell, as the desktop cuts it. */
    const val EMBED_CROP = 2 * CELL

    /**
     * An edge this close to a pixel boundary is taken as on it: a lock measured a hair off
     * must not drop a whole pixel of every cell.
     */
    private const val EDGE_SLACK = 0.1

    /** Top-left cell of a full board is a light square in either orientation. */
    fun isLight(row: Int, col: Int): Boolean = (row + col) % 2 == 0

    /** 0..63 in python-chess's order (a1 = 0), for the side at the bottom of the screen. */
    fun squareAt(row: Int, col: Int, whiteAtBottom: Boolean): Int =
        if (whiteAtBottom) (7 - row) * 8 + col else row * 8 + (7 - col)

    fun rowCol(square: Int, whiteAtBottom: Boolean): Pair<Int, Int> {
        val file = square % 8
        val rank = square / 8
        return if (whiteAtBottom) (7 - rank) to file else rank to (7 - file)
    }
}

/** A board locked in a frame: where it is, and how to cut its cells. */
data class BoardGeometry(val region: Region, val cellPx: Double) {

    /**
     * The pixels that lie wholly inside the cell. A cell's true edges fall between pixels
     * and a pixel straddling one is a blend of the two square colours; taking only whole
     * pixels costs at most a pixel a side and keeps the blends out (on the desktop, letting
     * them in wore a permanent hover outline on one cell and no move through it settled).
     */
    fun cellBox(row: Int, col: Int): Rect {
        val x0 = ceil(region.x + col * cellPx - Squares.EDGE_SLACK_PUBLIC).toInt()
        val y0 = ceil(region.y + row * cellPx - Squares.EDGE_SLACK_PUBLIC).toInt()
        val x1 = floor(region.x + (col + 1) * cellPx + Squares.EDGE_SLACK_PUBLIC).toInt()
        val y1 = floor(region.y + (row + 1) * cellPx + Squares.EDGE_SLACK_PUBLIC).toInt()
        return Rect(x0, y0, x1 - x0, y1 - y0)
    }

    /**
     * The cell resized to [size] square, BGR, as the embedder wants it. INTER_AREA averages
     * when shrinking but degenerates to nearest-neighbour when enlarging, so a small cell
     * blown up is interpolated instead.
     */
    fun crop(frame: Mat, row: Int, col: Int, size: Int = Squares.EMBED_CROP): Mat {
        val box = cellBox(row, col)
        val x0 = max(0, box.x)
        val y0 = max(0, box.y)
        val x1 = min(frame.cols(), box.x + box.width)
        val y1 = min(frame.rows(), box.y + box.height)
        if (x1 - x0 < 2 || y1 - y0 < 2) return Mat.zeros(size, size, frame.type())
        val patch = frame.submat(y0, y1, x0, x1)
        val out = Mat()
        val shrinking = (x1 - x0) >= size && (y1 - y0) >= size
        Imgproc.resize(
            patch,
            out,
            Size(size.toDouble(), size.toDouble()),
            0.0,
            0.0,
            if (shrinking) Imgproc.INTER_AREA else Imgproc.INTER_LINEAR,
        )
        return out
    }
}

/** Kotlin cannot see a private const from another file; the value is the Python's. */
private const val EDGE_SLACK = 0.1

internal val Squares.EDGE_SLACK_PUBLIC: Double
    get() = EDGE_SLACK
