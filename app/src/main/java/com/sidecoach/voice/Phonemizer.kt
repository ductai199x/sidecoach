package com.sidecoach.voice

import android.content.Context

/**
 * `phonemizer.phonemize(text, language, preserve_punctuation=True, with_stress=True)` with the
 * espeak backend and every other argument at its default -- the one call kokoro-onnx makes
 * (`Tokenizer.phonemize`). phonemizer 3.4.0 as the desktop has it: the punctuation taken out
 * and kept (`Punctuation.preserve`), each piece phonemized by espeak-ng and tidied
 * (`EspeakBackend._postprocess_line`), the punctuation put back (`Punctuation.restore`).
 *
 * The defaults that are not spelled out: the word separator ' ' and no phone separator
 * (`default_separator`), strip False, the language switch flags kept and word-count
 * mismatches ignored, one job. Lines split and join on '\n', as Python's `os.linesep` does
 * on the phone's Linux.
 */
object Phonemizer {

    /** The word separator of `default_separator`; its phone separator is empty. */
    private const val WORD = " "

    fun phonemize(context: Context, text: String, language: String = "en-us"): String {
        // `_phonemize`: the text as lines, each stripped of line breaks, the empty ones out.
        val lines = text.trim('\n').split("\n").map { it.trim('\n') }.filter { Py.strip(it).isNotEmpty() }
        if (lines.isEmpty()) return ""
        val phonemized = synchronized(Espeak.lock) {
            Espeak.ensure(context)
            Espeak.setVoice(language)
            backend(lines)
        }
        return phonemized.joinToString("\n")
    }

    /** `BaseBackend.phonemize` for the espeak backend with punctuation preserved. */
    private fun backend(text: List<String>): List<String> {
        val (chunks, marks) = Punctuation.preserve(text)
        val phonemized = chunks.map { postprocessLine(Espeak.textToPhonemes(it)) }
        return Punctuation.restore(phonemized, marks, WORD, strip = false)
    }

    /**
     * `EspeakBackend._postprocess_line` with stress kept, no tie, strip False: espeak's lines
     * merged into one, its stray separators fixed, and each word ended with the word
     * separator once the phone separators ('_') are gone.
     */
    private fun postprocessLine(raw: String): String {
        var line = Py.strip(raw).replace("\n", " ").replace("  ", " ")
        line = MANY_UNDERSCORES.replace(line, "_")
        line = UNDERSCORE_SPACE.replace(line, " ")
        // `KeepFlags.process`: a language switch stays as it is.
        if (line.isEmpty()) return ""
        val out = StringBuilder()
        for (word in line.split(" ")) {
            var w = Py.strip(word)
            w += "_"
            // `_process_tie` with no tie: the phone separators become the (empty) phone separator.
            w = w.replace("_", "")
            out.append(w).append(WORD)
        }
        return out.toString()
    }

    private val MANY_UNDERSCORES = Regex("_+")
    private val UNDERSCORE_SPACE = Regex("_ ")

    /**
     * `phonemizer.punctuation.Punctuation` with the default marks: `preserve` cuts each line at
     * its punctuation and remembers where each mark stood, `restore` puts them back between
     * the phonemized pieces.
     */
    object Punctuation {
        private const val DEFAULT_MARKS = ";:,.!?¡¿—…\"«»“”(){}[]"
        private const val DECIMAL_SEPARATORS = ",."

        /** A mark kept for restoration: its line, the text of it, and B / E / I / A. */
        data class Mark(val index: Int, val mark: String, val position: Char)

        /**
         * Zero or more spaces, one or more marks, zero or more spaces -- a mark that doubles as
         * a decimal separator only when it is not between two digits.
         */
        private val marksRe: Regex = run {
            val marks = DEFAULT_MARKS
            val decimals = marks.filter { it in DECIMAL_SEPARATORS }
            val others = marks.filter { it !in DECIMAL_SEPARATORS }
            val alternatives = ArrayList<String>()
            if (others.isNotEmpty()) alternatives.add("[${escape(others)}]")
            if (decimals.isNotEmpty()) {
                val escaped = escape(decimals)
                alternatives.add("(?<![0-9])[$escaped]")
                alternatives.add("[$escaped](?![0-9])")
            }
            Regex("""(\s*(?:${alternatives.joinToString("|")})+\s*)+""")
        }

        /** Every mark backslashed for a character class: none of them is a letter or digit. */
        private fun escape(marks: String): String = marks.map { "\\" + it }.joinToString("")

        fun preserve(text: List<String>): Pair<List<String>, List<Mark>> {
            val preservedText = ArrayList<String>()
            val preservedMarks = ArrayList<Mark>()
            for ((num, line) in text.withIndex()) {
                val (pieces, marks) = preserveLine(line, num)
                preservedText += pieces
                preservedMarks += marks
            }
            return preservedText.filter { it.isNotEmpty() } to preservedMarks
        }

        private fun preserveLine(original: String, num: Int): Pair<List<String>, List<Mark>> {
            var line = original
            val matches = marksRe.findAll(line).toList()
            if (matches.isEmpty()) return listOf(line) to emptyList()

            // The line is made only of punctuation marks.
            if (matches.size == 1 && matches[0].value == line) return emptyList<String>() to listOf(Mark(num, line, 'A'))

            // Where each mark stands: at the beginning (B), the end (E), in the middle (I).
            val marks = ArrayList<Mark>()
            for ((i, match) in matches.withIndex()) {
                var position = 'I'
                if (i == 0 && line.startsWith(match.value)) {
                    position = 'B'
                } else if (i == matches.size - 1 && line.endsWith(match.value)) {
                    position = 'E'
                }
                marks.add(Mark(num, match.value, position))
            }

            // The line cut at each mark in turn -- at the mark's text's first occurrence in
            // what is left, as `str.split(mark)` does.
            val preservedLine = ArrayList<String>()
            for (mark in marks) {
                val at = line.indexOf(mark.mark)
                val prefix: String
                val suffix: String
                if (at < 0) {
                    prefix = line
                    suffix = ""
                } else {
                    prefix = line.substring(0, at)
                    suffix = line.substring(at + mark.mark.length)
                }
                preservedLine.add(prefix)
                line = suffix
            }
            return (preservedLine + line) to marks
        }

        fun restore(phonemized: List<String>, preserved: List<Mark>, word: String, strip: Boolean): List<String> {
            var text = phonemized.toMutableList()
            var marks = preserved.toList()
            val punctuated = ArrayList<String>()
            var pos = 0

            while (text.isNotEmpty() || marks.isNotEmpty()) {
                if (marks.isEmpty()) {
                    for (one in text) {
                        var line = one
                        // If strip is False, the final word ends with a word separator.
                        if (!strip && word.isNotEmpty() && !line.endsWith(word)) line += word
                        punctuated.add(line)
                    }
                    text = ArrayList()
                } else if (text.isEmpty()) {
                    // Nothing has been phonemized: the marks alone, internal spaces as words.
                    punctuated.add(marks.joinToString("") { it.mark }.replace(" ", word))
                    marks = emptyList()
                } else {
                    val current = marks[0]
                    if (current.index == pos) {
                        marks = marks.drop(1)
                        val mark = current.mark.replace(" ", word)

                        // The last word separator of the current word comes off.
                        if (word.isNotEmpty() && text[0].endsWith(word)) text[0] = text[0].dropLast(word.length)

                        when (current.position) {
                            'B' -> text[0] = mark + text[0]
                            'E' -> {
                                punctuated.add(text[0] + mark + if (strip || mark.endsWith(word)) "" else word)
                                text = text.drop(1).toMutableList()
                                pos += 1
                            }
                            'A' -> {
                                punctuated.add(mark + if (strip || mark.endsWith(word)) "" else word)
                                pos += 1
                            }
                            else -> {
                                // 'I'
                                if (text.size == 1) {
                                    // The final part of an intermediate mark was not phonemized.
                                    text[0] = text[0] + mark
                                } else {
                                    val firstWord = text[0]
                                    text = text.drop(1).toMutableList()
                                    text[0] = firstWord + mark + text[0]
                                }
                            }
                        }
                    } else {
                        punctuated.add(text[0])
                        text = text.drop(1).toMutableList()
                        pos += 1
                    }
                }
            }
            return punctuated
        }
    }
}

/** Python's string whitespace, for the places a port must strip and split as `str` does. */
object Py {
    fun isSpace(c: Char): Boolean = when (c) {
        ' ', '\t', '\n', '\u000B', '\u000C', '\r', '\u001C', '\u001D', '\u001E', '\u001F',
        '\u0085', ' ', ' ', ' ', ' ', ' ', ' ', '　' -> true
        else -> c in ' '..' '
    }

    /** `str.strip()`. */
    fun strip(s: String): String = s.trim { isSpace(it) }

    /** `str.split()`: runs of whitespace, no empty pieces. */
    fun split(s: String): List<String> {
        val out = ArrayList<String>()
        val piece = StringBuilder()
        for (c in s) {
            if (isSpace(c)) {
                if (piece.isNotEmpty()) {
                    out.add(piece.toString())
                    piece.setLength(0)
                }
            } else {
                piece.append(c)
            }
        }
        if (piece.isNotEmpty()) out.add(piece.toString())
        return out
    }

    /**
     * A string as Python's `repr` writes it (`%r`, `{x!r}`): single quotes unless the text holds
     * one and no double; backslash, the quote, tab, newline and return escaped; any other
     * character `str.isprintable` refuses as `\xNN`, `\uNNNN` or `\UNNNNNNNN`.
     */
    fun repr(text: String): String {
        val quote = if ('\'' in text && '"' !in text) '"' else '\''
        val out = StringBuilder().append(quote)
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            i += Character.charCount(cp)
            when {
                cp == '\\'.code -> out.append("\\\\")
                cp == quote.code -> out.append('\\').append(quote)
                cp == '\t'.code -> out.append("\\t")
                cp == '\n'.code -> out.append("\\n")
                cp == '\r'.code -> out.append("\\r")
                cp < 0x20 || cp == 0x7f -> out.append(String.format(java.util.Locale.ROOT, "\\x%02x", cp))
                cp < 0x7f || printable(cp) -> out.appendCodePoint(cp)
                cp <= 0xff -> out.append(String.format(java.util.Locale.ROOT, "\\x%02x", cp))
                cp <= 0xffff -> out.append(String.format(java.util.Locale.ROOT, "\\u%04x", cp))
                else -> out.append(String.format(java.util.Locale.ROOT, "\\U%08x", cp))
            }
        }
        return out.append(quote).toString()
    }

    /**
     * `bytes` as Python's `str`/`repr` writes them: `b'...'`, quoted as a string is; backslash,
     * the quote, tab, newline and return escaped, and every byte outside printable ASCII as `\xNN`.
     */
    fun bytesRepr(data: ByteArray): String {
        val hasSingle = data.any { it == '\''.code.toByte() }
        val hasDouble = data.any { it == '"'.code.toByte() }
        val quote = if (hasSingle && !hasDouble) '"' else '\''
        val out = StringBuilder("b").append(quote)
        for (byte in data) {
            val b = byte.toInt() and 0xff
            when {
                b == '\\'.code -> out.append("\\\\")
                b == quote.code -> out.append('\\').append(quote)
                b == '\t'.code -> out.append("\\t")
                b == '\n'.code -> out.append("\\n")
                b == '\r'.code -> out.append("\\r")
                b < 0x20 || b >= 0x7f -> out.append(String.format(java.util.Locale.ROOT, "\\x%02x", b))
                else -> out.append(b.toChar())
            }
        }
        return out.append(quote).toString()
    }

    /** `str.isprintable` for one character: not "Other" (Cc, Cf, Cs, Co, Cn) nor a separator but space. */
    private fun printable(cp: Int): Boolean = when (Character.getType(cp).toByte()) {
        Character.CONTROL, Character.FORMAT, Character.SURROGATE, Character.PRIVATE_USE,
        Character.UNASSIGNED, Character.LINE_SEPARATOR, Character.PARAGRAPH_SEPARATOR,
        -> false
        Character.SPACE_SEPARATOR -> cp == ' '.code
        else -> true
    }
}
