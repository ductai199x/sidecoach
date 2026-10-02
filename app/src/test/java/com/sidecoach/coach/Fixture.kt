package com.sidecoach.coach

import com.sidecoach.TestAssets
import com.sidecoach.Workers
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
internal fun fixture(name: String): JSONObject = open(name) { it.document(null) }

/**
 * The fixture's top-level object with only [keys] in it: every other value is skipped as it
 * streams past, never built. StoryTest wants two small lists of coach.json and read all 29 MB of
 * it into a tree once a test, thirteen times a run.
 */
internal fun fixture(name: String, keys: Set<String>): JSONObject = open(name) { it.document(keys) }

/**
 * Each element of the top-level array [key], built one at a time and let go: an array of
 * thousands of positions never in memory at once. The heap held the whole of coach.json's while
 * its 1,465 positions were checked, and spent the run collecting at 187 of its 192 MB.
 */
internal fun forEachIn(name: String, key: String, each: (JSONObject) -> Unit) {
    open(name) { it.stream(key, each) }
}

/**
 * [forEachIn], the elements checked side by side on every core as they stream ([Workers]): for a
 * replay whose cases are independent and whose checks are safe from several threads.
 */
internal fun forEachInParallel(name: String, key: String, each: (JSONObject) -> Unit) {
    Workers().use { workers -> forEachIn(name, key) { element -> workers.submit { each(element) } } }
}

private fun <T> open(name: String, read: (StreamedJson) -> T): T {
    return TestAssets.open(name).reader(Charsets.UTF_8).use { read(StreamedJson(it)) }
}

private class StreamedJson(private val reader: Reader) {
    // Read in blocks into a buffer of its own: a call into the reader for every character was
    // most of the time a 29 MB fixture took.
    private val buffer = CharArray(64 * 1024)
    private var length = 0
    private var at = 0

    private fun fill(): Boolean {
        if (at < length) return true
        length = reader.read(buffer, 0, buffer.size)
        at = 0
        return length > 0
    }

    private fun peek(): Int = if (fill()) buffer[at].code else -1

    private fun next(): Int = if (fill()) buffer[at++].code else -1

    private fun skip() {
        while (fill()) {
            val c = buffer[at]
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') at++ else return
        }
    }

    private fun expect(c: Char) {
        val got = next()
        check(got == c.code) { "expected '$c', got '${if (got < 0) "end" else got.toChar()}'" }
    }

    /** The top-level object, with only [keys] when they are given. */
    fun document(keys: Set<String>?): JSONObject {
        val value = if (keys == null) value() else topLevel(keys, null, null)
        skip()
        check(peek() == -1) { "trailing text after the document" }
        return value as JSONObject
    }

    /** Each element of the top-level array [key], one at a time; everything else skipped. */
    fun stream(key: String, each: (JSONObject) -> Unit) {
        topLevel(emptySet(), key, each)
        skip()
        check(peek() == -1) { "trailing text after the document" }
    }

    private fun topLevel(keys: Set<String>, streamed: String?, each: ((JSONObject) -> Unit)?): JSONObject {
        skip()
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
            skip()
            when {
                key == streamed && each != null -> elements(each)
                key in keys -> out.put(key, value())
                else -> skipValue()
            }
            skip()
            when (next()) {
                ','.code -> continue
                '}'.code -> return out
                else -> error("expected ',' or '}'")
            }
        }
    }

    private fun elements(each: (JSONObject) -> Unit) {
        expect('[')
        skip()
        if (peek() == ']'.code) {
            next()
            return
        }
        while (true) {
            each(value() as JSONObject)
            skip()
            when (next()) {
                ','.code -> continue
                ']'.code -> return
                else -> error("expected ',' or ']'")
            }
        }
    }

    /** Past one value, building nothing: brackets counted, strings stepped over whole. */
    private fun skipValue() {
        skip()
        var depth = 0
        while (true) {
            val c = next()
            check(c >= 0) { "the document ended inside a value" }
            when (c) {
                '"'.code -> skipString()
                '{'.code, '['.code -> depth++
                '}'.code, ']'.code -> {
                    depth--
                    check(depth >= 0) { "unbalanced brackets" }
                }
            }
            if (depth == 0) {
                // A scalar ends where the next separator begins; a bracket or a string has ended.
                if (c == '}'.code || c == ']'.code || c == '"'.code) return
                while (true) {
                    val d = peek()
                    if (d < 0 || d == ','.code || d == '}'.code || d == ']'.code || d == ' '.code ||
                        d == '\n'.code || d == '\r'.code || d == '\t'.code
                    ) {
                        return
                    }
                    next()
                }
            }
        }
    }

    /** Past the rest of a string whose opening quote was read. */
    private fun skipString() {
        while (true) {
            val c = next()
            check(c >= 0) { "unterminated string" }
            if (c == '\\'.code) next() else if (c == '"'.code) return
        }
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
            check(fill()) { "unterminated string" }
            // The run up to the next quote or escape, appended whole.
            var end = at
            while (end < length && buffer[end] != '"' && buffer[end] != '\\') end++
            out.append(buffer, at, end - at)
            at = end
            if (at == length) continue
            val c = next()
            if (c == '"'.code) return out.toString()
            when (val e = next()) {
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
