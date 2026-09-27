package com.sidecoach.voice

/**
 * Chess notation turned into words before it is spoken. `sidecoach/voice/notation.py`.
 *
 * A voice reads "Nf3" as "enn eff three" and "O-O" as "oh oh"; the coach's prose is full of such
 * tokens. The standard algebraic notation embedded in a line -- piece moves, captures, castling,
 * checks, promotions, en passant, and bare squares -- becomes its spoken form, and the prose
 * around it is left alone. Only tokens shaped like notation are matched, so an ordinary word or
 * a number in the sentence is never mangled.
 */
object Notation {
    private val PIECE = mapOf("K" to "king", "Q" to "queen", "R" to "rook", "B" to "bishop", "N" to "knight")
    private val RANK = mapOf(
        "1" to "one", "2" to "two", "3" to "three", "4" to "four",
        "5" to "five", "6" to "six", "7" to "seven", "8" to "eight",
    )

    // A SAN move: an optional piece, an optional from-file and/or from-rank for disambiguation,
    // an optional capture, the destination square, an optional promotion, and an optional check
    // or mate mark. Castling and game results are handled separately. Word-bounded so it matches
    // a token, not a fragment of a longer word.
    //
    // Python's `\b`, written out: every token here begins and ends with what Python counts a word
    // character -- a letter, a number of any kind, the underscore -- so the bound is "no word
    // character beside it". ICU's `\b` does not count "½" one, and never matched "½-½" after a
    // space (the parity audit, 2026-09-26).
    private const val NOT_AFTER_WORD = """(?<![\p{L}\p{N}_])"""
    private const val NOT_BEFORE_WORD = """(?![\p{L}\p{N}_])"""
    private val MOVE = Regex(NOT_AFTER_WORD + """([KQRBN])?([a-h])?([1-8])?(x)?([a-h])([1-8])(=[QRBN])?([+#])?(?![A-Za-z0-9])""")
    private val CASTLE = Regex(NOT_AFTER_WORD + """(O-O-O|O-O)([+#])?(?![A-Za-z0-9])""")
    private val RESULT = Regex(NOT_AFTER_WORD + """(1-0|0-1|1/2-1/2|½-½)""" + NOT_BEFORE_WORD)

    private fun square(file: String, rank: String): String = "$file ${RANK.getValue(rank)}"

    private fun mark(mark: String?): String = when (mark) {
        "+" -> " check"
        "#" -> " checkmate"
        else -> ""
    }

    private fun castle(match: MatchResult): String {
        val side = if (match.groupValues[1] == "O-O-O") "queenside" else "kingside"
        return "castles $side${mark(match.groups[2]?.value)}"
    }

    private fun move(match: MatchResult): String {
        val piece = match.groups[1]?.value
        val fromFile = match.groups[2]?.value
        val fromRank = match.groups[3]?.value
        val capture = match.groups[4]?.value
        val toFile = match.groupValues[5]
        val toRank = match.groupValues[6]
        val promo = match.groups[7]?.value
        val check = match.groups[8]?.value
        // A bare square with none of a piece, capture or promotion is a pawn move (or a square
        // the prose just names); anything with only a destination and nothing else is still SAN.
        val words = ArrayList<String>()
        if (!piece.isNullOrEmpty()) words.add(PIECE.getValue(piece))
        if (!fromFile.isNullOrEmpty()) words.add(fromFile)
        if (!fromRank.isNullOrEmpty()) words.add(RANK.getValue(fromRank))
        if (!capture.isNullOrEmpty()) words.add("takes")
        words.add(square(toFile, toRank))
        var text = words.joinToString(" ")
        if (!promo.isNullOrEmpty()) text += " promotes to ${PIECE.getValue(promo.substring(1, 2))}"
        return text + mark(check)
    }

    private fun result(match: MatchResult): String = when (match.groupValues[1]) {
        "1-0" -> "white wins"
        "0-1" -> "black wins"
        else -> "draw"
    }

    /**
     * [text] with every chess-notation token replaced by how it should be read aloud. Results
     * and castling are recognised before the general move pattern, which would otherwise
     * swallow part of them; everything the patterns do not match is left as it is.
     */
    fun speak(text: String): String {
        var out = RESULT.replace(text) { result(it) }
        out = CASTLE.replace(out) { castle(it) }
        return MOVE.replace(out) { move(it) }
    }
}
