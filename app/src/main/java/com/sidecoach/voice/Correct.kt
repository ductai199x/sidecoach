package com.sidecoach.voice

import com.sidecoach.chess.Board
import com.sidecoach.chess.PAWN
import com.sidecoach.chess.QUEEN
import com.sidecoach.chess.squareName

/**
 * Chess-aware correction of a raw transcript, against the moves that are legal right now.
 * `sidecoach/voice/correct.py`.
 *
 * A speech recogniser hears English, not chess: "knight" comes back as "night" or "9", "e takes
 * d five" as "E-Tix D5", "the a-pawn" as "the upon". The fixes are of two kinds. First, a table
 * of known confusions -- homophones and glued tokens -- that is safe in any chess sentence.
 * Second, and the real one: every legal move on the board has a handful of spoken forms
 * ("knight f three", "knight to f three", "bishop takes c six", "castle kingside"), and a span of
 * the transcript that matches one of them closely is rewritten as that move's SAN. The board is
 * the ground truth, so the correction can only ever produce a move that exists; the rest of the
 * sentence is left as the user said it, for the coach and its claim checker.
 */
object Correct {
    private val WORDS = linkedMapOf(
        "1" to "one", "2" to "two", "3" to "three", "4" to "four",
        "5" to "five", "6" to "six", "7" to "seven", "8" to "eight",
    )

    /** What a number sounds like after a file letter ("e for" is e4, "knight f tree" is f3). */
    private val NUMBER_LIKE = linkedMapOf(
        "for" to "four", "fore" to "four", "to" to "two", "too" to "two", "tree" to "three",
        "free" to "three", "won" to "one", "ate" to "eight", "sex" to "six", "sevin" to "seven",
    )

    /** What a file letter sounds like before a number ("see six" is c6). */
    private val LETTER_LIKE = linkedMapOf(
        "eh" to "a", "ay" to "a", "bee" to "b", "be" to "b", "see" to "c", "sea" to "c",
        "dee" to "d", "gee" to "g", "aitch" to "h", "age" to "h",
    )

    private val ANY_NUMBER = (WORDS.values + WORDS.keys + NUMBER_LIKE.keys).joinToString("|")

    private val I = setOf(RegexOption.IGNORE_CASE)

    /** Known confusions, applied before matching. Each is safe in any chess sentence. */
    private val CONFUSIONS: List<Pair<Regex, String>> = listOf(
        Regex("""\b(night|nite|nights)\b""", I) to "knight",
        Regex("""\b(nine|9)\b(?=\s+(?:on|to|from|takes|[a-h]\b))""", I) to "knight",
        Regex("""\bwith the (?:nine|9)\b""", I) to "with the knight",
        Regex("""\b(rock|rocks)\b(?=\s+(?:on|to|[a-h]\b|takes))""", I) to "rook",
        Regex("""\b(?:rooka|roka) ?([1-8])\b""", I) to "rook a$1", // "Roka 1" is rook + a1
        Regex("""\b(?:rook|rock) ?([a-h]) ?([1-8])\b""", I) to "rook $1$2",
        Regex("""\bthe (?:upon|appawn|a-pawn|apon)\b""", I) to "the a pawn",
        Regex("""\bczech\b""", I) to "check",
        Regex("""\bcheque\b""", I) to "check",
        Regex("""\bking'?s? ?side\b""", I) to "kingside",
        Regex("""\bqueen'?s? ?side\b""", I) to "queenside",
        Regex("""\bcastle (?:short|shorts)\b""", I) to "castle kingside",
        Regex("""\bcastle long\b""", I) to "castle queenside",
        Regex("""\b([a-h])[- ]?(?:takes|tix|tex|tics|tx|x)\b""", I) to "$1 takes",
        Regex("""\b([a-h])x([a-h])\s*($ANY_NUMBER)\b""", I) to "$1 takes $2 $3",
        Regex("""\bcaptures\b""", I) to "takes",
    )

    private val NUMBER_AFTER_FILE = Regex("""\b([a-h]) ?(${NUMBER_LIKE.keys.joinToString("|")})\b""")
    private val LETTER_BEFORE_NUMBER = Regex("""\b(${LETTER_LIKE.keys.joinToString("|")}) (${WORDS.values.joinToString("|")})\b""")
    private val SQUARE = Regex("""\b([a-h]) ?([1-8])\b""")
    private val NOT_WORD = Regex("""[^\w\s'-]""")
    private val SPACES = Regex("""\s+""")

    /** Lowercase, punctuation stripped, confusions fixed, squares as 'e four'. */
    fun normalise(text: String): String {
        var t = text.lowercase()
        t = NOT_WORD.replace(t, " ")
        t = t.replace("-", " ")
        for ((pattern, repl) in CONFUSIONS) t = pattern.replace(t, repl)
        // A file letter followed by something that sounds like a number is a square.
        t = NUMBER_AFTER_FILE.replace(t) { m -> "${m.groupValues[1]} ${NUMBER_LIKE.getValue(m.groupValues[2])}" }
        t = LETTER_BEFORE_NUMBER.replace(t) { m -> "${LETTER_LIKE.getValue(m.groupValues[1])} ${m.groupValues[2]}" }
        t = SQUARE.replace(t) { m -> "${m.groupValues[1]} ${WORDS.getValue(m.groupValues[2])}" }
        return Py.strip(SPACES.replace(t, " "))
    }

    private val PIECE_NAMES = listOf("pawn", "knight", "bishop", "rook", "queen", "king")

    private fun pieceName(type: Int): String = PIECE_NAMES[type - 1]

    /** Every way a legal move might be said -> its SAN. `spoken_forms`. */
    fun spokenForms(board: Board): LinkedHashMap<String, String> {
        val forms = LinkedHashMap<String, String>()
        for (move in board.legalMoves()) {
            val san = board.san(move)
            val base = Notation.speak(san) // "knight f three", "bishop takes c six", "castles kingside"
            val variants = LinkedHashSet<String>()
            variants.add(base)
            val stem = base.removeSuffix(" check").removeSuffix(" checkmate")
            variants.add(stem)
            if (san.startsWith("O-O")) {
                val side = if (san.startsWith("O-O-O")) "queenside" else "kingside"
                variants.addAll(listOf("castle $side", "castles $side", "castling $side", side))
            } else {
                val type = board.pieceTypeAt(move.from).let { if (it == 0) PAWN else it }
                val piece = pieceName(type)
                val to = squareName(move.to)
                val dest = "${to[0]} ${WORDS.getValue(to.substring(1))}"
                if (piece == "pawn") {
                    variants.addAll(listOf("pawn $dest", "pawn to $dest", dest))
                    if (board.isCapture(move)) variants.add("pawn takes $dest")
                } else {
                    variants.addAll(listOf("$piece to $dest", "$piece $dest"))
                    if (board.isCapture(move)) variants.addAll(listOf("$piece takes $dest", "$piece takes on $dest"))
                }
                if (board.isCapture(move) && piece != "pawn") {
                    val victim = board.pieceTypeAt(move.to)
                    if (victim != 0) {
                        variants.add("$piece takes ${pieceName(victim)}")
                        variants.add("$piece takes the ${pieceName(victim)}")
                    }
                }
            }
            val promotion = move.promotion
            if (promotion != 0) {
                // What the pawn becomes is said after the square too: "e eight queen", "e eight
                // equals queen", as well as the notation's "e eight promotes to queen".
                val name = pieceName(promotion)
                val head = stem.split(" promotes to ")[0]
                variants.addAll(listOf("$head $name", "$head equals $name"))
            }
            for (v in variants) {
                // A promotion that does not say to what is a queen's: the forms that name no
                // piece belong to it alone. The knight's, generated last, used to take them all,
                // and "e8 queen" came out as e8=N+ (2026-09-23).
                if (promotion != 0 && promotion != QUEEN && pieceName(promotion) !in v) continue
                forms[v] = san
            }
        }
        return forms
    }

    /** A corrected transcript: the sentence with spoken moves as SAN, the moves found, the raw text. */
    data class Correction(val text: String, val moves: List<String> = emptyList(), val raw: String = "")

    private const val MIN_RATIO = 0.86

    /**
     * Fix known confusions and rewrite any spoken legal move as its SAN. Without a board only
     * the confusion table applies. Matching is greedy over the longest spans first; a span
     * needs a square (or a castling word) in it before it is compared, so ordinary sentences
     * are never touched.
     */
    fun correct(text: String, board: Board?): Correction {
        val norm = normalise(text)
        if (board == null) return Correction(text = norm, raw = text)
        val forms = spokenForms(board)
        val tokens = Py.split(norm)
        val out = ArrayList<String>()
        val moves = ArrayList<String>()
        var i = 0
        while (i < tokens.size) {
            var best: Triple<Double, Int, String>? = null // (ratio, span length, san)
            for (n in minOf(5, tokens.size - i) downTo 1) {
                val span = tokens.subList(i, i + n)
                if (!plausible(span)) continue
                if (n == 2 && i > 0 && tokens[i - 1] == "on" && span[0].length == 1) {
                    continue // "the pawn on d five" is a place, not a move there
                }
                val phrase = span.joinToString(" ")
                val exact = forms[phrase]
                if (exact != null) {
                    best = Triple(1.0, n, exact)
                    break
                }
                val spanKeys = keys(span)
                for ((form, san) in forms) {
                    val ratio = SequenceMatcher(phrase, form).ratio()
                    // Close is not enough: the piece and the square the person named must be the
                    // form's. Fuzz may mend how the words around them were heard, never swap them
                    // for another legal move -- "knight to f three" became Kf3 when only the king
                    // could go there (Tai, 2026-09-23: what matches no legal move stays as words,
                    // for the coach to read).
                    if (ratio >= MIN_RATIO && (best == null || ratio > best.first) && keys(form.split(" ")) == spanKeys) {
                        best = Triple(ratio, n, san)
                    }
                }
            }
            if (best != null && best.first >= MIN_RATIO) {
                out.add(best.third)
                moves.add(best.third)
                i += best.second
            } else {
                out.add(tokens[i])
                i += 1
            }
        }
        return Correction(text = out.joinToString(" "), moves = moves, raw = text)
    }

    private val SQUARE_WORDS = WORDS.values.toSet()
    private val CASTLE_WORDS = setOf("castle", "castles", "castling", "kingside", "queenside")

    /**
     * The only words a spoken move is made of. A span with anything else in it is not compared,
     * so "play knight to f three here" matches on "knight to f three" alone.
     */
    private val MOVE_VOCAB: Set<String> = SQUARE_WORDS + CASTLE_WORDS + "abcdefgh".map { it.toString() } + setOf(
        "knight", "bishop", "rook", "queen", "king", "pawn", "takes", "to", "on", "the", "x",
        "check", "checkmate", "promotes", "promote", "equals",
    )

    /**
     * The words that say which move: the piece, the square, the side castled to, what a pawn
     * becomes -- and "on", which makes a square a place ("the pawn on e five") rather than where
     * a move goes. A fuzzy match must keep all of them as they were said.
     */
    private val KEY_WORDS: Set<String> = SQUARE_WORDS + "abcdefgh".map { it.toString() } +
        setOf("knight", "bishop", "rook", "queen", "king", "pawn", "kingside", "queenside", "on")

    private fun keys(words: List<String>): List<String> = words.filter { it in KEY_WORDS }

    /** A span worth comparing to the move forms: move words only, naming a square or castling. */
    private fun plausible(span: List<String>): Boolean {
        if (!span.all { it in MOVE_VOCAB }) return false
        if (span.any { it in CASTLE_WORDS }) return true
        for ((a, b) in span.zip(span.drop(1))) {
            if (a.length == 1 && a in "abcdefgh" && b in SQUARE_WORDS) return true
        }
        return false
    }
}

/**
 * `difflib.SequenceMatcher(None, a, b)` for [ratio]: the matching blocks found as Python finds
 * them (the longest match first, then recursively either side), so the ratio is the same to the
 * last bit. No junk; strings this short never trip the popular-element heuristic (200+).
 */
class SequenceMatcher(private val a: String, private val b: String) {
    private val b2j: Map<Char, List<Int>> = run {
        val out = LinkedHashMap<Char, MutableList<Int>>()
        for ((i, c) in b.withIndex()) out.getOrPut(c) { ArrayList() }.add(i)
        if (b.length >= 200) {
            // `autojunk`: an element in more than 1% of a long b is "popular" and not indexed.
            val ntest = b.length / 100 + 1
            out.entries.removeAll { it.value.size > ntest }
        }
        out
    }

    private data class Match(val i: Int, val j: Int, val size: Int)

    private fun findLongestMatch(alo: Int, ahi: Int, blo: Int, bhi: Int): Match {
        var besti = alo
        var bestj = blo
        var bestsize = 0
        var j2len = HashMap<Int, Int>()
        for (i in alo until ahi) {
            val newj2len = HashMap<Int, Int>()
            for (j in b2j[a[i]] ?: emptyList()) {
                if (j < blo) continue
                if (j >= bhi) break
                val k = (j2len[j - 1] ?: 0) + 1
                newj2len[j] = k
                if (k > bestsize) {
                    besti = i - k + 1
                    bestj = j - k + 1
                    bestsize = k
                }
            }
            j2len = newj2len
        }
        // With no junk, the extensions only take in equal elements the index left out.
        while (besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1]) {
            besti -= 1
            bestj -= 1
            bestsize += 1
        }
        while (besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] == b[bestj + bestsize]) {
            bestsize += 1
        }
        return Match(besti, bestj, bestsize)
    }

    /** The number of matched characters: `sum(size for ... in get_matching_blocks())`. */
    private fun matches(): Int {
        val queue = ArrayDeque<IntArray>()
        queue.addLast(intArrayOf(0, a.length, 0, b.length))
        var total = 0
        while (queue.isNotEmpty()) {
            val (alo, ahi, blo, bhi) = queue.removeLast()
            val (i, j, k) = findLongestMatch(alo, ahi, blo, bhi)
            if (k != 0) {
                total += k
                if (alo < i && blo < j) queue.addLast(intArrayOf(alo, i, blo, j))
                if (i + k < ahi && j + k < bhi) queue.addLast(intArrayOf(i + k, ahi, j + k, bhi))
            }
        }
        return total
    }

    fun ratio(): Double {
        val length = a.length + b.length
        return if (length == 0) 1.0 else 2.0 * matches() / length
    }
}
