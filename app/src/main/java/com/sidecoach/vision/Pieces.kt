package com.sidecoach.vision

/**
 * Per-theme piece recognition bootstrapped from the start position: the constants and the
 * types of `sidecoach/vision/pieces.py`, ported line for line.
 *
 * Every game starts from a known position, which hands us labelled examples of all twelve
 * piece kinds on whatever theme is on screen. Recognition then works on silhouettes:
 *
 * 1. Each cell's own background colour is the median of a ring just inside its border, so
 *    highlights need no special handling and the thin hover outline chess.com draws around
 *    the square under the cursor is skipped. A piece rarely reaches that ring.
 * 2. Foreground is every pixel far from that background, with enclosed pockets filled (a
 *    white piece on a light square differs from the square mostly by its outline) and a
 *    small opening applied so labels and anti-aliasing do not count.
 * 3. The embedding matcher names the cell: the 64 cells of the start position embedded by
 *    a frozen pretrained net, a cell named by its nearest one. No threshold on the
 *    foreground at all, so a shaded set and thin detail read like any other.
 * 4. What sits on top of the board is told apart by geometry, never by colour: a tint
 *    shifts the ring and the deeper corners the same way; a hover outline or an edge
 *    shifts the ring while the corners stay clean; a cover is an overlay confirmed by the
 *    neighbours; a drawing is saturated colour off every line between the square's colour
 *    and the set's.
 *
 * The desktop's other matchers (the silhouette overlap, and the cookie cutter behind
 * `MATCHER`) are not ported: the embedding has been the default since 2026-09-19 and the
 * others exist there as the record of what the thresholds could do.
 */
object Pieces {

    const val INNER = 0.08
    const val RING_SKIP = 0.05
    const val RING_BAND = 0.07

    /**
     * The four corner blocks between these insets sample the background deeper than the
     * ring: a piece is round-ish and covers at most one or two of them partially, a badge
     * one. Deeper is not an option: on the recorded games rooks and bishops reach blocks
     * at 12% and beyond.
     */
    val CORNER_INSET = 0.10 to 0.20

    const val BG_TOLERANCE = 25.0

    /**
     * Once the set's fills are known, a piece whose fill has little contrast against its
     * square is read again with a tolerance of half that contrast, within these bounds: a
     * dark shaded set on a dark board, a greyish white on a pale one, had half of every
     * piece within 25 of the square and read as two shapes. Only that piece's own fill
     * counts: a lower tolerance for every piece let a label's anti-aliasing join a rook.
     */
    const val BG_TOLERANCE_MIN = 8.0
    const val BG_CONTRAST_SHARE = 0.5
    const val POCKET_TOLERANCE = 8.0
    const val HIGHLIGHT_TOLERANCE = 30.0

    /**
     * A tint shifts the ring and the corners the same way; an outline or an edge does not.
     * Each corner pixel is read on its own: clean (at the square colour), shifted like the
     * ring, or something else (a piece, a badge, a label), and a cell is foreign when a
     * quarter of the corner pixels that are one or the other are clean, or when too few
     * are either.
     */
    const val TINT_MIN_COSINE = 0.9
    const val TINT_MIN_CORNER_SHIFT = HIGHLIGHT_TOLERANCE / 2
    const val CORNER_CLEAN_MAX = 0.25
    const val CORNER_MIN_KNOWN = 0.2

    /**
     * Something drawn over a square -- an arrow, a mark's edge -- is saturated colour far
     * from the square's own colour and from every colour of the piece set. The smoothed
     * edge of a dark piece on a bright highlight is chromatic too, but it lies on the line
     * from the square's colour to the piece's; a drawing is off every such line by more
     * than [OVERLAY_OFF_LINE]. A cell with more than [OVERLAY_MAX] of its inner pixels
     * like that cannot be read, so no move commits through it.
     */
    const val OVERLAY_DISTANCE = 60.0
    const val OVERLAY_CHROMA = 70.0
    const val OVERLAY_OFF_LINE = 40.0
    const val OVERLAY_MAX = 0.03

    /**
     * The move-classification badge sits on the top-right corner; a quarter-disc of this
     * radius (as a share of the cell) around that corner is not measured for drawings.
     */
    const val OVERLAY_BADGE_RADIUS = 0.4

    /** An outline colour is read from a piece only when this many pixels lie outside its tone. */
    const val EDGE_MIN_PIXELS = 8

    const val EMPTY_MAX_FRACTION = 0.04

    /**
     * Chess.com's legal-move hint dots cover about a tenth of a cell; the smallest piece,
     * a pawn, covers a third. Anything in between is reported as unknown, which carries no
     * evidence.
     */
    const val PIECE_MIN_FRACTION = 0.16

    const val MIN_IOU = 0.35
    const val MIN_MARGIN = 0.03
    const val BRIGHTNESS_WEIGHT = 1.5

    /**
     * A piece's side is read from its brightness first: when it lies within one side's
     * exemplars' range of brightness, widened by this much, and not the other's, the shape
     * is matched among that side's pieces alone.
     */
    const val SIDE_MARGIN = 0.05

    /** Of the patch: a component within one corner's box is no part of a piece. */
    const val CORNER_BOX = 0.3

    /** A tone mask smaller than this share of the full mask, or of the patch, is no tone. */
    const val TONE_MIN_SHARE = 0.2
    const val TONE_MIN_AREA = 0.08

    /**
     * The embedder sees every cell at this resolution, whatever the board's size: the crop
     * brought down to CANONICAL_PX a side and back ([canonical]). A board drawn larger carries
     * more detail, and the embeddings of a board resized away from the one its set was learned
     * at drifted (bishops read as knights after a board was enlarged half again: the parity
     * audit's sweep, 2026-09-26); at one resolution every size looks alike. 40 px, from the
     * desktop's scale benchmark (44 sets, learned at one of 32-128 px and read at another):
     * identity errors 161 -> 13, where 24 px lost the detail that tells some sets' pieces apart
     * (a pirouetti queen read as a bishop). Landing keeps the full detail: brought down too, a
     * piece in flight read as landed.
     */
    const val CANONICAL_PX = 40

    const val SETTLED_TOLERANCE = 0.08  // of a cell; a dragged piece follows the cursor
    const val SETTLED_FLOOR_PX = 3.0    // source pixels: the lock is only good to about one

    /**
     * The embedding matcher lets the brightness choose a piece's side only when its own
     * nearest white and nearest black piece lie within this of each other.
     */
    const val COLOUR_TIE = 0.05

    /**
     * Where a piece sits, under the embedding matcher (`pieces.LANDING`): "phase", the phase
     * correlation of the cell against the named piece's start-position crop, which measures the
     * shift itself (no silhouette: at 0.03, 0.06, 0.1 and 0.2 of a cell off it read 0.031,
     * 0.062, 0.098 and 0.199 over eight sets, three sizes, both square colours, highlighted or
     * not, and 0.033 at rest with a pixel of lock jitter; 0.9% of pieces at rest and 0.1% of
     * pieces a tenth off wrongly called at the 0.08 tolerance; unmoved by a badge, the check
     * glow, a hint ring, an arrow, a tint or a label); "centroid", the silhouette's centroid
     * against the exemplar's (the overlap's rule). Both need the hover outline absent. The
     * desktop reads it from `SIDECOACH_LANDING`; the phone has no environment to read, so it is
     * a property a bench may set ([com.sidecoach.vision.LandingCostTest]).
     */
    @Volatile
    var LANDING: String = "phase"

    /** The fill read under the exemplar's mask (`_fill_side`). */
    const val FILL_SIDE_ERODE = 2
    const val FILL_SIDE_MIN_PIXELS = 30
    const val FILL_SIDE_RATIO = 1.5

    /** A white outline over a light square differs from it by only about 50 levels. */
    const val HOVER_TOLERANCE = 30.0

    /** The outer band an outline starts in, as a share of the cell. */
    const val HOVER_BAND = 0.05
    const val HOVER_LINE_FILL = 0.9
    const val HOVER_EDGE = 0.04
    const val HOVER_CORNER = 0.2
    const val HOVER_STEP = 10.0

    const val COVER_TOLERANCE = 12.0
    const val TINT_MIN_ALPHA = 0.2
    const val TINT_ALPHA_SPREAD = 0.15
    const val TINT_MIN_CELLS = 6
    const val TINT_BLOCK_FILL = 0.8
    const val THEME_MIN_TINTED = 32
    const val OVERLAY_SIDE_FILL = 0.9
    const val OVERLAY_BLOCK_FILL = 0.85
    const val COVER_BODY_SHIFT = 20.0
    const val COVER_BODY_FILL = 0.8

    const val EXEMPLAR_MAX_FRACTION = 0.9
    const val EXEMPLAR_MIN_TONE = 0.08
    const val EXEMPLAR_MAX_SIDEWAYS = 0.12  // of a cell
    const val EXEMPLAR_MAX_VERTICAL = 0.28  // of a cell
    const val PAIR_MIN_IOU = 0.8
    const val PAIR_MAX_CENTROID = 0.06  // of a cell

    /**
     * The landing tolerance in inner-patch units: a fraction of a cell, but never less
     * than a few source pixels, so a small cell does not turn lock jitter into a piece in
     * flight.
     */
    fun settleTolerance(cellPx: Double?): Double {
        if (cellPx == null) return SETTLED_TOLERANCE
        val innerPx = cellPx * (1.0 - 2.0 * INNER)
        if (innerPx <= 0) return SETTLED_TOLERANCE
        return maxOf(SETTLED_TOLERANCE, SETTLED_FLOOR_PX / innerPx)
    }

    /**
     * A cell's crop as the embedder sees it: brought down to [CANONICAL_PX] a side
     * (`INTER_AREA`) and back (`INTER_LINEAR`), so a board of any size shows the embedder the
     * same detail. Only the embedding goes through it; silhouettes, outlines, tints and landing
     * keep the full-detail crop. `pieces.canonical`. The caller releases what it returns.
     */
    fun canonical(crop: org.opencv.core.Mat): org.opencv.core.Mat {
        val size = crop.rows().toDouble()
        val small = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.resize(
            crop, small, org.opencv.core.Size(CANONICAL_PX.toDouble(), CANONICAL_PX.toDouble()),
            0.0, 0.0, org.opencv.imgproc.Imgproc.INTER_AREA,
        )
        val back = org.opencv.core.Mat()
        org.opencv.imgproc.Imgproc.resize(
            small, back, org.opencv.core.Size(size, size), 0.0, 0.0, org.opencv.imgproc.Imgproc.INTER_LINEAR,
        )
        small.release()
        return back
    }

    /** What a cell holds, as `pieces.Kind`. */
    enum class Kind { EMPTY, PIECE, UNKNOWN }

    /** Whether a shifted background is a readable tint, as `pieces.TintKind`. */
    enum class TintKind { NONE, TINT, FOREIGN }

    /** One cell, read: `pieces.CellReading`. */
    data class CellReading(
        val kind: Kind,
        val symbol: String? = null,
        val colour: Side? = null,
        val confidence: Double = 0.0,
        val fgFraction: Double = 0.0,
        val luminance: Double = 0.0,
        val highlighted: Boolean = false,
        val background: DoubleArray = doubleArrayOf(0.0, 0.0, 0.0),
        val brightness: Double = 0.0,
        /**
         * Whether the piece sits where a settled piece of its kind sits. The default is
         * true so a reading built by hand counts as landed; every branch of the classifier
         * decides it explicitly.
         */
        val settled: Boolean = true,
        val hover: Boolean = false,
        val offset: Double = 0.0,
        /** Best symbol even when not confident enough to assert it. */
        val guess: String? = null,
        /** An opaque overlay confirmed by the neighbours hides this cell. */
        val covered: Boolean = false,
        /** The hue of a translucent tint: "yellow", "red", "other", or null. */
        val tint: String? = null,
        /** Something drawn over the square: no move commits through it. */
        val drawn: Boolean = false,
    ) {
        /**
         * Equal field by field, as the desktop's frozen dataclass is: the background too, by
         * its values (a data class would compare the array by identity).
         */
        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is CellReading) return false
            return kind == other.kind && symbol == other.symbol && colour == other.colour &&
                confidence == other.confidence && fgFraction == other.fgFraction &&
                luminance == other.luminance && highlighted == other.highlighted &&
                background.contentEquals(other.background) && brightness == other.brightness &&
                settled == other.settled && hover == other.hover && offset == other.offset &&
                guess == other.guess && covered == other.covered && tint == other.tint &&
                drawn == other.drawn
        }

        override fun hashCode(): Int {
            var h = kind.hashCode()
            h = 31 * h + (symbol?.hashCode() ?: 0)
            h = 31 * h + (colour?.hashCode() ?: 0)
            h = 31 * h + confidence.hashCode()
            h = 31 * h + fgFraction.hashCode()
            h = 31 * h + luminance.hashCode()
            h = 31 * h + highlighted.hashCode()
            h = 31 * h + background.contentHashCode()
            h = 31 * h + brightness.hashCode()
            h = 31 * h + settled.hashCode()
            h = 31 * h + hover.hashCode()
            h = 31 * h + offset.hashCode()
            h = 31 * h + (guess?.hashCode() ?: 0)
            h = 31 * h + covered.hashCode()
            h = 31 * h + (tint?.hashCode() ?: 0)
            h = 31 * h + drawn.hashCode()
            return h
        }
    }

    /**
     * `pieces.mark_on_empty`: a little drawn on an empty square and nothing more -- a
     * legal-move dot, a label -- less than the smallest piece covers, no arrow over it, no drag
     * outline on it, nothing hiding it. It is no evidence while pieces move (a dot is read as
     * unknown for that: see [ThemeModel.classify]), but a board read on the player's word
     * counts it as the empty square it is: a piece left selected put dots on five squares, and
     * "Re-read the screen" failed every time (Tai's desktop, 2026-09-24, dump
     * 20260924-165745).
     */
    fun markOnEmpty(reading: CellReading): Boolean =
        reading.kind == Kind.UNKNOWN &&
            !reading.drawn &&
            !reading.hover &&
            !reading.covered &&
            reading.fgFraction < PIECE_MIN_FRACTION

    /** Which side a piece belongs to, as `sidecoach.events.Side`. */
    enum class Side { WHITE, BLACK }
}
