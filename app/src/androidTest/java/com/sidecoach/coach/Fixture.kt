package com.sidecoach.coach

import androidx.test.platform.app.InstrumentationRegistry
import org.json.JSONArray
import org.json.JSONObject
import java.io.Reader

/**
 * A parity fixture of the test APK (`androidTest/assets`), read as it streams: the JSON tree is
 * built from the reader, never from the whole text held as a String. coach.json is 29 MB -- 58 MB
 * as a String, more while it is read -- and the full suite's heap is 192 MB: read whole, twice,
 * it ran out of memory (the parity pass, 2026-09-26). The values are org.json's, as
 * `JSONObject(text)` gives them: an int an Integer (a Long past it), a fraction a Double.
 */
internal fun fixture(name: String): JSONObject {
    val context = InstrumentationRegistry.getInstrumentation().context
    return context.assets.open(name).bufferedReader(Charsets.UTF_8).use { StreamedJson(it).document() }
}

private class StreamedJson(private val reader: Reader) {
    private var ahead = -2

    private fun peek(): Int {
        if (ahead == -2) ahead = reader.read()
        return ahead
    }

    private fun next(): Int {
        val c = peek()
        ahead = -2
        return c
    }

    private fun skip() {
        while (peek() == ' '.code || peek() == '\n'.code || peek() == '\r'.code || peek() == '\t'.code) next()
    }

    private fun expect(c: Char) {
        val got = next()
        check(got == c.code) { "expected '$c', got '${if (got < 0) "end" else got.toChar()}'" }
    }

    fun document(): JSONObject {
        val value = value()
        skip()
        check(peek() == -1) { "trailing text after the document" }
        return value as JSONObject
    }

    private fun value(): Any {
        skip()
        return when (val c = peek()) {
            '{'.code -> obj()
            '['.code -> array()
            '"'.code -> string()
            't'.code -> word("true", true)
            'f'.code -> word("false", false)
            'n'.code -> word("null", JSONObject.NULL)
            else -> if (c == '-'.code || c in '0'.code..'9'.code) number() else error("unexpected '${c.toChar()}'")
        }
    }

    private fun word(text: String, value: Any): Any {
        for (ch in text) expect(ch)
        return value
    }

    private fun obj(): JSONObject {
        expect('{')
        val out = JSONObject()
        skip()
        if (peek() == '}'.code) {
            next()
            return out
        }
        while (true) {
            skip()
            val key = string()
            skip()
            expect(':')
            out.put(key, value())
            skip()
            when (next()) {
                ','.code -> continue
                '}'.code -> return out
                else -> error("expected ',' or '}'")
            }
        }
    }

    private fun array(): JSONArray {
        expect('[')
        val out = JSONArray()
        skip()
        if (peek() == ']'.code) {
            next()
            return out
        }
        while (true) {
            out.put(value())
            skip()
            when (next()) {
                ','.code -> continue
                ']'.code -> return out
                else -> error("expected ',' or ']'")
            }
        }
    }

    private fun string(): String {
        expect('"')
        val out = StringBuilder()
        while (true) {
            val c = next()
            check(c >= 0) { "unterminated string" }
            when (c) {
                '"'.code -> return out.toString()
                '\\'.code -> when (val e = next()) {
                    'u'.code -> {
                        val hex = CharArray(4) { next().toChar() }
                        out.append(String(hex).toInt(16).toChar())
                    }
                    'b'.code -> out.append('\b')
                    'f'.code -> out.append('\u000c')
                    'n'.code -> out.append('\n')
                    'r'.code -> out.append('\r')
                    't'.code -> out.append('\t')
                    else -> out.append(e.toChar()) // " \ /
                }
                else -> out.append(c.toChar())
            }
        }
    }

    private fun number(): Any {
        val text = StringBuilder()
        while (peek() == '-'.code || peek() == '+'.code || peek() == '.'.code || peek() == 'e'.code ||
            peek() == 'E'.code || peek() in '0'.code..'9'.code
        ) {
            text.append(next().toChar())
        }
        val literal = text.toString()
        if ('.' in literal || 'e' in literal || 'E' in literal) return literal.toDouble()
        val long = literal.toLongOrNull() ?: return literal.toDouble()
        return if (long in Int.MIN_VALUE..Int.MAX_VALUE) long.toInt() else long
    }
}
