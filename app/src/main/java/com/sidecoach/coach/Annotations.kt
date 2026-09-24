package com.sidecoach.coach

import com.sidecoach.chess.Board
import com.sidecoach.chess.Move
import com.sidecoach.chess.squareName
import com.sidecoach.events.Annotation
import com.sidecoach.events.DisclosureLevel
import com.sidecoach.events.Trigger

/**
 * Board marks that go with a coach message: `sidecoach/coach/annotations.py`.
 *
 * The marks come from the same disclosure-filtered fact bundle the message was written from,
 * never from the prose alone, so an arrow cannot give away more than the words do: at the idea
 * level there is nothing to draw; at the focus level a tint on the region, the target and a
 * ring on the piece; at the move level the focus squares and an arrow for the move; at the line
 * level the line. A message about a move already played gets the played move coloured by how
 * it scored and a green arrow for the better one -- two arrows, no more.
 *
 * A second, smaller source is the finished text: the squares and moves the coach actually
 * named. Those passed the leak check, so drawing them reveals nothing new.
 */

/**
 * The played move's arrow takes the colour of how the move scored; a move that scored fine is
 * "played" (neutral) rather than green, so green always means "the better move".
 */
private val SEVERITY_COLOR = mapOf(
    "none" to "played",
    "inaccuracy" to "warn",
    "mistake" to "bad",
    "blunder" to "bad",
)

/** How far into a line to draw, and how the continuation fades. */
private const val LINE_PLIES = 4
private val LINE_OPACITY = doubleArrayOf(0.9, 0.55, 0.4, 0.3)

/** A SAN token in prose (a piece move, a pawn move, a capture, a promotion, castling). */
private val SAN_TOKEN = Regex(
    "(?<![A-Za-z0-9])((?:O-O(?:-O)?|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?)(?![A-Za-z0-9])",
)

/** The squares a region name covers, for a light tint: "the kingside, around f7". */
private fun regionSquares(region: String): List<String> {
    val name = region.lowercase()
    val files = when {
        "kingside" in name -> "efgh"
        "queenside" in name -> "abcd"
        "centre" in name || "center" in name -> return listOf("d4", "d5", "e4", "e5")
        else -> return emptyList()
    }
    return files.flatMap { f -> "12345678".map { r -> "$f$r" } }
}

/** A SAN token read on `board`, or null when it is not a legal move there. */
fun parseSan(board: Board, san: String): Move? = board.parseSan(san)

private fun arrow(board: Board, san: String, color: String, opacity: Double = 1.0): Annotation? {
    val move = parseSan(board, san) ?: return null
    return Annotation(
        kind = "arrow",
        squares = listOf(squareName(move.from), squareName(move.to)),
        color = color,
        opacity = opacity,
        label = san,
    )
}

/** Arrows for the first plies of a line, fading, each on the board it is played from. */
private fun lineArrows(board: Board, sans: List<String>, color: String): List<Annotation> {
    val out = ArrayList<Annotation>()
    val work = board.copy(withStack = false)
    for ((i, san) in sans.take(LINE_PLIES).withIndex()) {
        val move = parseSan(work, san) ?: break
        out.add(
            Annotation(
                kind = "arrow",
                squares = listOf(squareName(move.from), squareName(move.to)),
                color = color,
                opacity = LINE_OPACITY[minOf(i, LINE_OPACITY.size - 1)],
                label = san,
            ),
        )
        work.push(move)
    }
    return out
}

/**
 * The marks for a message written from `facts` at `level`, on `board` (the position the facts
 * describe: the one a played move was played from, or the live one for a hint).
 */
fun annotationsFor(
    facts: PositionFacts,
    level: DisclosureLevel,
    board: Board,
    trigger: Trigger,
    playedSan: String? = null,
    severity: String? = null,
): List<Annotation> {
    val visible = visibleFacts(facts, level)
    val marks = ArrayList<Annotation>()
    val seen = HashSet<Pair<String, List<String>>>()

    fun add(mark: Annotation?) {
        if (mark == null) return
        val key = mark.kind to mark.squares
        if (!seen.add(key)) return
        marks.add(mark)
    }

    val aboutPlayed =
        (trigger == Trigger.POST_MOVE || trigger == Trigger.POST_MISTAKE) && !playedSan.isNullOrEmpty()
    if (aboutPlayed) {
        // Two arrows and no more: what was played, and what was better.
        add(arrow(board, playedSan!!, SEVERITY_COLOR[severity ?: "none"] ?: "good"))
        @Suppress("UNCHECKED_CAST")
        val candidates = visible["candidates"] as? List<String> ?: emptyList()
        val better = candidates.firstOrNull()
        if (better != null && better != playedSan) add(arrow(board, better, "good"))
        return marks
    }

    // A hint about the live position: exactly what the level lets the words say. The target
    // first: a square the region also tints is marked as the target.
    val target = visible["target"] as? String
    if (!target.isNullOrEmpty()) {
        for (square in squaresIn(target).sorted()) {
            add(Annotation(kind = "square", squares = listOf(square), color = "focus"))
        }
    }
    val region = visible["region"] as? String
    if (!region.isNullOrEmpty()) {
        for (square in regionSquares(region)) {
            add(Annotation(kind = "square", squares = listOf(square), color = "region", opacity = 0.35))
        }
        for (square in squaresIn(region).sorted()) {
            add(Annotation(kind = "square", squares = listOf(square), color = "focus", opacity = 0.6))
        }
    }
    @Suppress("UNCHECKED_CAST")
    for (square in (visible["focus"] as? List<String> ?: emptyList())) {
        add(Annotation(kind = "square", squares = listOf(square), color = "focus"))
    }
    val piece = visible["piece"] as? String
    if (!piece.isNullOrEmpty()) {
        for (square in squaresIn(piece).sorted()) {
            add(Annotation(kind = "ring", squares = listOf(square), color = "focus"))
        }
    }
    @Suppress("UNCHECKED_CAST")
    for (text in (visible["ideas"] as? List<String> ?: emptyList())) {
        // Sorted, as the desktop now sorts: its `squares_in` returns a set of strings and
        // CPython randomises string hashing per process.
        for (square in squaresIn(text).sorted()) {
            add(Annotation(kind = "square", squares = listOf(square), color = "mention", opacity = 0.6))
        }
    }
    @Suppress("UNCHECKED_CAST")
    for (san in (visible["threats"] as? List<String> ?: emptyList()).take(2)) {
        // A threat is the opponent's move, read after a null move.
        add(arrow(nullMoved(board), san, "threat", 0.8))
    }
    @Suppress("UNCHECKED_CAST")
    val line = visible["line"] as? List<String>
    @Suppress("UNCHECKED_CAST")
    val candidates = visible["candidates"] as? List<String>
    if (visible.containsKey("line") && !line.isNullOrEmpty()) {
        for (mark in lineArrows(board, line, "line")) add(mark)
    } else if (visible.containsKey("candidates") && !candidates.isNullOrEmpty()) {
        add(arrow(board, candidates[0], "good"))
    }
    return marks
}

/**
 * Marks for the squares and moves the finished message names, so the sentence and the board
 * point at the same things. Moves are read on the message's board; a token that is not a legal
 * move there is treated as a bare square.
 */
fun mentionsIn(text: String, board: Board): List<Annotation> {
    val marks = ArrayList<Annotation>()
    val seen = HashSet<String>()
    val landed = HashSet<String>()  // squares an arrow already points at: no highlight on top
    for (match in SAN_TOKEN.findAll(text)) {
        val san = match.groupValues[1]
        val move = parseSan(board, san) ?: continue
        val key = "m:" + move.uci()
        if (key in seen) continue
        seen.add(key)
        val to = squareName(move.to)
        landed.add(to)
        marks.add(
            Annotation(
                kind = "arrow",
                squares = listOf(squareName(move.from), to),
                color = "mention",
                opacity = 0.7,
                label = san,
            ),
        )
    }
    for (square in (squaresIn(text) - landed).sorted()) {
        marks.add(Annotation(kind = "square", squares = listOf(square), color = "mention", opacity = 0.5))
    }
    return marks
}
