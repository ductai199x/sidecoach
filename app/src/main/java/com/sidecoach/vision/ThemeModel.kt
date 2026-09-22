package com.sidecoach.vision

import org.opencv.core.Mat

/**
 * What one board's pieces look like, learned from the start position, and the reading of a
 * later frame against it. The embedding half of `sidecoach/vision/pieces.py`.
 *
 * The desktop learns the theme when it sees the start position: all 64 cells are embedded
 * and labelled from the known layout, and every later cell is named by the nearest of those
 * 64 in embedding space. Nothing is trained and no piece set is known in advance, which is
 * why a theme nobody has seen reads on the first move.
 */
class ThemeModel(
    val vectors: Array<FloatArray>,
    val labels: List<String>,
    val whiteAtBottom: Boolean,
) {

    /** What one cell holds: a piece symbol in FEN case, "empty", or null when unsure. */
    data class Reading(val label: String?, val similarity: Float, val margin: Float) {
        val occupied: Boolean get() = label != null && label != "empty"
    }

    /**
     * The cell named by the nearest exemplar. Below [CellEmbedder.MIN_SIMILARITY] nothing
     * on the start position looks like this and the cell is occupied by something of no
     * sure kind; an "empty" that wins by less than the margin is no answer either.
     */
    fun read(vector: FloatArray): Reading {
        var best = -1
        var bestSim = Float.NEGATIVE_INFINITY
        val sims = FloatArray(vectors.size)
        for (i in vectors.indices) {
            var dot = 0f
            val row = vectors[i]
            for (k in row.indices) dot += row[k] * vector[k]
            sims[i] = dot
            if (dot > bestSim) { bestSim = dot; best = i }
        }
        if (best < 0) return Reading(null, 0f, 0f)
        val label = labels[best]
        var other = -1f
        for (i in vectors.indices) if (labels[i] != label && sims[i] > other) other = sims[i]
        val margin = bestSim - other
        if (bestSim < CellEmbedder.MIN_SIMILARITY) return Reading(null, bestSim, margin)
        if (label == "empty" && margin < CellEmbedder.MIN_MARGIN) return Reading(null, bestSim, margin)
        return Reading(label, bestSim, margin)
    }

    /** The 64 cells of a frame, read in screen order (row 0 at the top). */
    fun readBoard(frame: Mat, geometry: BoardGeometry, embedder: CellEmbedder): List<Reading> {
        val crops = ArrayList<Mat>(64)
        for (r in 0 until 8) for (c in 0 until 8) crops.add(geometry.crop(frame, r, c))
        val vectors = embedder.embed(crops)
        crops.forEach { it.release() }
        return vectors.map { read(it) }
    }

    /** The position as FEN placement, from a board read in screen order. */
    fun placement(readings: List<Reading>): String {
        val squares = arrayOfNulls<String>(64)
        for (r in 0 until 8) {
            for (c in 0 until 8) {
                val reading = readings[r * 8 + c]
                val symbol = reading.label
                if (symbol != null && symbol != "empty") {
                    squares[Squares.squareAt(r, c, whiteAtBottom)] = symbol
                }
            }
        }
        val out = StringBuilder()
        for (rank in 7 downTo 0) {
            var empty = 0
            for (file in 0 until 8) {
                val piece = squares[rank * 8 + file]
                if (piece == null) {
                    empty++
                } else {
                    if (empty > 0) { out.append(empty); empty = 0 }
                    out.append(piece)
                }
            }
            if (empty > 0) out.append(empty)
            if (rank > 0) out.append('/')
        }
        return out.toString()
    }

    companion object {
        /** 8x8 foreground fractions, the cheap occupancy signal (`pieces.occupancy`). */
        fun occupancy(frame: Mat, geometry: BoardGeometry): List<Sil> {
            val out = ArrayList<Sil>(64)
            for (r in 0 until 8) {
                for (c in 0 until 8) {
                    val crop = geometry.crop(frame, r, c, 2 * Squares.CELL)
                    out.add(SilhouetteReader.of(crop))
                    crop.release()
                }
            }
            return out
        }

        /**
         * Rows 0, 1, 6 and 7 occupied and rows 2 to 5 empty, allowing one stray cell
         * (`pieces._start_shaped`): one move away from the start already differs in two,
         * so the allowance is a single cell for a badge, a hover outline or a cursor.
         */
        fun startShaped(cells: List<Sil>): Boolean {
            var wrong = 0
            for (r in 0 until 8) {
                for (c in 0 until 8) {
                    val occupied = cells[r * 8 + c].fraction >= Pieces.PIECE_MIN_FRACTION
                    if (occupied != (r <= 1 || r >= 6)) wrong++
                }
            }
            return wrong <= 1
        }

        /**
         * Which colour sits at the bottom of a start position: the brighter pieces are
         * white (`pieces.orientation_from_start`).
         */
        fun orientationFromStart(cells: List<Sil>): Boolean {
            val top = ArrayList<Double>(16)
            val bottom = ArrayList<Double>(16)
            for (c in 0 until 8) {
                top.add(cells[c].luminance)
                top.add(cells[8 + c].luminance)
                bottom.add(cells[48 + c].luminance)
                bottom.add(cells[56 + c].luminance)
            }
            return median(bottom) > median(top)
        }

        private fun median(values: List<Double>): Double {
            val sorted = values.sorted()
            val n = sorted.size
            return if (n % 2 == 1) sorted[n / 2] else (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
        }

        /** The start position in screen order for the side at the bottom: 64 labels. */
        fun startLabels(whiteAtBottom: Boolean): List<String> {
            val back = listOf("r", "n", "b", "q", "k", "b", "n", "r")
            val labels = ArrayList<String>(64)
            for (r in 0 until 8) {
                for (c in 0 until 8) {
                    val square = Squares.squareAt(r, c, whiteAtBottom)
                    val rank = square / 8
                    val file = square % 8
                    labels.add(
                        when (rank) {
                            0 -> back[file].uppercase()
                            1 -> "P"
                            6 -> "p"
                            7 -> back[file]
                            else -> "empty"
                        },
                    )
                }
            }
            return labels
        }

        /**
         * Learn a theme from a frame showing the start position on a locked board.
         *
         * The caller has already seen the start position's shape (`Silhouette.startShaped`,
         * the desktop's `looks_like_start_position`). Returns null when the frame does not
         * read as a start position: a cell hidden under a cursor or a badge, or erased,
         * sits nearer an empty square than any piece of its own colour, and would be that
         * piece's exemplar for the whole game (`pieces._start_reads_as_start`).
         */
        fun learn(
            frame: Mat,
            geometry: BoardGeometry,
            embedder: CellEmbedder,
            whiteAtBottom: Boolean,
        ): ThemeModel? {
            val crops = ArrayList<Mat>(64)
            for (r in 0 until 8) for (c in 0 until 8) crops.add(geometry.crop(frame, r, c))
            val vectors = embedder.embed(crops)
            crops.forEach { it.release() }
            val labels = startLabels(whiteAtBottom)
            if (!startReadsAsStart(vectors, labels)) return null
            return ThemeModel(vectors, labels, whiteAtBottom)
        }

        /** `pieces._start_reads_as_start`, cosine similarities over unit vectors. */
        private fun startReadsAsStart(vectors: Array<FloatArray>, labels: List<String>): Boolean {
            val empties = labels.indices.filter { labels[it] == "empty" }
            if (empties.isEmpty()) return true
            for (i in labels.indices) {
                val label = labels[i]
                if (label == "empty") continue
                val own = labels.indices.filter {
                    it != i && labels[it] != "empty" &&
                        labels[it][0].isUpperCase() == label[0].isUpperCase()
                }
                if (own.isEmpty()) continue
                var nearestEmpty = Float.NEGATIVE_INFINITY
                for (j in empties) nearestEmpty = maxOf(nearestEmpty, dot(vectors[i], vectors[j]))
                var nearestOwn = Float.NEGATIVE_INFINITY
                for (j in own) nearestOwn = maxOf(nearestOwn, dot(vectors[i], vectors[j]))
                if (nearestEmpty > nearestOwn) {
                    android.util.Log.d(
                        "sidecoach.vision",
                        "start does not read as start: cell $i ($label) is nearer an empty square " +
                            "($nearestEmpty) than any ${if (label[0].isUpperCase()) "white" else "black"} " +
                            "piece ($nearestOwn)",
                    )
                    return false
                }
            }
            return true
        }

        private fun dot(a: FloatArray, b: FloatArray): Float {
            var sum = 0f
            for (k in a.indices) sum += a[k] * b[k]
            return sum
        }
    }
}
