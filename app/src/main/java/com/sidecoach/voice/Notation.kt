package com.sidecoach.voice

/**
 * Chess notation turned into words before it is spoken. `sidecoach/voice/notation.py`.
 *
 * A voice reads "Nf3" as "enn eff three" and "O-O" as "oh oh"; the coach's prose is full of such
 * tokens. The standard algebraic notation embedded in a line -- piece moves, captures, castling,
 * checks, promotions, en passant, and bare squares -- becomes its spoken form, and the prose
 * around it is left alone. Only tokens shaped like notation are matched, so an ordinary word or
 * a number in the sentence is never mangled.
 *
 * Two things the voice needed on top (Tai, 2026-09-29: "dxe5 pronounced like d-ai-ex-ee-five ...
 * anything involving the a-file would be pronounce 'er'"). A square or a pawn move written with a
 * capital, as the model starts a sentence ("G4 was a miss", "Dxe5 was better"), matched no move
 * and was spelled out letter by letter; it is read as the square or the pawn move it is. And the
 * file letter a on its own is the article to the voice's phonemizer (espeak: "uh"), so "rook a
 * four" was heard as "rook of fours": the voice is given "eh" instead, which the phonemizer reads
 * as the letter. That one is for the voice only (`voice = true`): the speech correction matches
 * what a person says against these words, and a person says "a four".
 */
object Notation {
    private val PIECE = mapOf("K" to "king", "Q" to "queen", "R" to "rook", "B" to "bishop", "N" to "knight", "P" to "pawn")
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
    private val MOVE = Regex(NOT_AFTER_WORD + """([KQRBNP])?([a-h])?([1-8])?(x)?([a-h])([1-8])(=?[QRBN])?([+#])?(?![A-Za-z0-9])""")
    // A square or a pawn move with a capital file letter, as a sentence's first word has it:
    // "G4", "Dxe5", "Exd8=Q+". A capital B before a capture is a bishop's ("Bxc6" is SAN); straight
    // before a rank it is the square. The whole token is checked, so an opening's code ("A40") is
    // not one.
    private val CAPITAL = Regex(NOT_AFTER_WORD + """([A-H])(?=(x[a-h])?[1-8](?:=?[QRBN])?[+#]?(?![A-Za-z0-9]))""")

    // The file letter a as the voice reads it (see the note above).
    private val VOICED = mapOf("a" to "eh")
    // Castling with the letter O, or with zeros as many write it ("0-0" was read "zero dash zero").
    private val CASTLE = Regex(NOT_AFTER_WORD + """(O-O-O|O-O|0-0-0|0-0)([+#])?(?![A-Za-z0-9-])""")
    private val RESULT = Regex(NOT_AFTER_WORD + """(1-0|0-1|1/2-1/2|½-½)""" + NOT_BEFORE_WORD)

    private fun file(letter: String, voice: Boolean): String = if (voice) VOICED[letter] ?: letter else letter

    private fun square(file: String, rank: String, voice: Boolean): String = "${file(file, voice)} ${RANK.getValue(rank)}"

    private fun lower(match: MatchResult): String {
        val letter = match.groupValues[1]
        return if (letter == "B" && match.groups[2] != null) letter else letter.lowercase()
    }

    private fun mark(mark: String?): String = when (mark) {
        "+" -> " check"
        "#" -> " checkmate"
        else -> ""
    }

    private fun castle(match: MatchResult): String {
        val side = if (match.groupValues[1].length == 5) "queenside" else "kingside"
        return "castles $side${mark(match.groups[2]?.value)}"
    }

    private fun move(match: MatchResult, voice: Boolean): String {
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
        if (!fromFile.isNullOrEmpty()) words.add(file(fromFile, voice))
        if (!fromRank.isNullOrEmpty()) words.add(RANK.getValue(fromRank))
        if (!capture.isNullOrEmpty()) words.add("takes")
        words.add(square(toFile, toRank, voice))
        var text = words.joinToString(" ")
        if (!promo.isNullOrEmpty()) text += " promotes to ${PIECE.getValue(promo.takeLast(1))}"
        return text + mark(check)
    }

    private fun result(match: MatchResult): String = when (match.groupValues[1]) {
        "1-0" -> "white wins"
        "0-1" -> "black wins"
        else -> "draw"
    }

    /**
     * [text] with every chess-notation token replaced by how it should be read aloud; [voice],
     * as the voice is given it (the file letter a as "eh"). Results and castling are recognised
     * before the general move pattern, which would otherwise swallow part of them, and a capital
     * file letter is made the letter it is before it; everything the patterns do not match is
     * left as it is.
     */
    fun speak(text: String, voice: Boolean = false): String {
        var out = RESULT.replace(text) { result(it) }
        out = CASTLE.replace(out) { castle(it) }
        out = CAPITAL.replace(out) { lower(it) }
        return MOVE.replace(out) { move(it, voice) }
    }
}
