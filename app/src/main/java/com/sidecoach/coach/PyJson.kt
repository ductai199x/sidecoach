package com.sidecoach.coach

import org.json.JSONArray
import org.json.JSONObject
import java.math.BigDecimal
import java.math.BigInteger

/**
 * What the desktop's Python does with a model's JSON, for the ports of the code that reads it:
 * `json.loads` -- strict, where org.json accepts unquoted keys, single quotes, comments and
 * trailing text, and refuses NaN in an object -- and `str`, truthiness and iteration of the
 * values it gives. A model's reply is read the same way on both: a JSON null label is Python's
 * `None`, falsy, not the string "null" (the parity audit cc-05, 2026-09-26).
 *
 * The values are Python's, in Kotlin: a dict is a `Map<String, Any?>` in its keys' order, a
 * list a `List<Any?>`, a str a String, an int a Long (a BigInteger past it), a float a Double,
 * True and False Booleans, None null.
 */

/** Thrown where Python's `json.loads` raises `JSONDecodeError`. */
class PyJsonError(message: String) : Exception(message)

/**
 * `json.loads(text)`: one JSON document, whitespace around it, nothing else; `NaN`, `Infinity`
 * and `-Infinity` as Python reads them; a key given twice keeps its first place and its last
 * value, as a dict does.
 */
fun pyJsonLoads(text: String): Any? {
    val parser = PyJsonParser(text)
    parser.skip()
    val value = parser.value()
    parser.skip()
    if (!parser.atEnd()) throw PyJsonError("Extra data")
    return value
}

/** `json.loads(text)`, or [fallback] where it raises: `except ValueError`. */
fun pyJsonOr(text: String, fallback: Any?): Any? = try {
    pyJsonLoads(text)
} catch (_: PyJsonError) {
    fallback
}

/**
 * `json.dumps(value)`: `", "` and `": "` between the parts, a dict's keys in their order, every
 * character outside ' '..'~' escaped (`ensure_ascii`), a float as its repr, NaN and the
 * infinities as Python writes them. Python's values here, and org.json's: the arguments of a
 * model's call go back to it as the desktop writes them (`openai_compat.chat_turns`).
 */
fun pyJsonDumps(value: Any?): String = StringBuilder().also { pyDump(value, it) }.toString()

private fun pyDump(value: Any?, out: StringBuilder) {
    when {
        value == null || value == JSONObject.NULL -> out.append("null")
        value is Boolean -> out.append(if (value) "true" else "false")
        value is String -> pyDumpString(value, out)
        value is Double || value is Float -> {
            val number = (value as Number).toDouble()
            out.append(
                when {
                    number.isNaN() -> "NaN"
                    number == Double.POSITIVE_INFINITY -> "Infinity"
                    number == Double.NEGATIVE_INFINITY -> "-Infinity"
                    else -> pyFloatRepr(number)
                },
            )
        }
        value is Number -> out.append(value.toString()) // an int
        value is Map<*, *> -> {
            out.append('{')
            for ((index, entry) in value.entries.withIndex()) {
                if (index > 0) out.append(", ")
                pyDumpString(entry.key.toString(), out)
                out.append(": ")
                pyDump(entry.value, out)
            }
            out.append('}')
        }
        value is List<*> -> {
            out.append('[')
            for ((index, one) in value.withIndex()) {
                if (index > 0) out.append(", ")
                pyDump(one, out)
            }
            out.append(']')
        }
        value is JSONObject -> {
            out.append('{')
            for ((index, key) in value.keys().asSequence().withIndex()) {
                if (index > 0) out.append(", ")
                pyDumpString(key, out)
                out.append(": ")
                pyDump(value.opt(key), out)
            }
            out.append('}')
        }
        value is JSONArray -> {
            out.append('[')
            for (index in 0 until value.length()) {
                if (index > 0) out.append(", ")
                pyDump(value.opt(index), out)
            }
            out.append(']')
        }
        else -> pyDumpString(value.toString(), out)
    }
}

private fun pyDumpString(text: String, out: StringBuilder) {
    out.append('"')
    for (char in text) {
        when (char) {
            '"' -> out.append("\\\"")
            '\\' -> out.append("\\\\")
            '\n' -> out.append("\\n")
            '\r' -> out.append("\\r")
            '\t' -> out.append("\\t")
            '\b' -> out.append("\\b")
            '\u000C' -> out.append("\\f")
            in ' '..'~' -> out.append(char)
            else -> out.append("\\u").append(hex(char.code, 4))
        }
    }
    out.append('"')
}

/** Python's truthiness of a JSON value: None, False, 0, 0.0, "", [] and {} are false. */
fun pyTruthy(value: Any?): Boolean = when (value) {
    null -> false
    is Boolean -> value
    is Double -> value != 0.0 // NaN is true, as in Python
    is Long -> value != 0L
    is BigInteger -> value.signum() != 0
    is String -> value.isNotEmpty()
    is List<*> -> value.isNotEmpty()
    is Map<*, *> -> value.isNotEmpty()
    else -> true
}

/** `str(value)` of a JSON value, as Python writes it. */
fun pyStr(value: Any?): String = if (value is String) value else pyReprValue(value)

/**
 * `for x in value`: a list's items, a string's characters, a dict's keys; anything else is
 * not iterable and raises, as Python's TypeError does.
 */
fun pyIter(value: Any?): List<Any?> = when (value) {
    is List<*> -> value
    is String -> value.codePoints().toArray().map { String(Character.toChars(it)) }
    is Map<*, *> -> value.keys.toList()
    else -> throw IllegalArgumentException("'${pyTypeName(value)}' object is not iterable")
}

private fun pyTypeName(value: Any?): String = when (value) {
    null -> "NoneType"
    is Boolean -> "bool"
    is Double -> "float"
    is Long, is BigInteger -> "int"
    else -> value.javaClass.simpleName
}

/**
 * `repr(s)` of a str, as CPython writes it: single quotes unless the text has a single quote
 * and no double one; the backslash, the chosen quote, \t \n \r escaped; other characters that
 * are not printable as \xhh, \uhhhh or \Uhhhhhhhh.
 */
fun pyRepr(text: String): String {
    val quote = if ('\'' in text && '"' !in text) '"' else '\''
    val out = StringBuilder().append(quote)
    var i = 0
    while (i < text.length) {
        val cp = text.codePointAt(i)
        i += Character.charCount(cp)
        when {
            cp == quote.code || cp == '\\'.code -> out.append('\\').appendCodePoint(cp)
            cp == '\t'.code -> out.append("\\t")
            cp == '\n'.code -> out.append("\\n")
            cp == '\r'.code -> out.append("\\r")
            cp < 0x20 || cp == 0x7f -> out.append("\\x").append(hex(cp, 2))
            cp < 0x7f -> out.appendCodePoint(cp)
            printable(cp) -> out.appendCodePoint(cp)
            cp <= 0xff -> out.append("\\x").append(hex(cp, 2))
            cp <= 0xffff -> out.append("\\u").append(hex(cp, 4))
            else -> out.append("\\U").append(hex(cp, 8))
        }
    }
    return out.append(quote).toString()
}

private fun hex(value: Int, width: Int): String = Integer.toHexString(value).padStart(width, '0')

/**
 * `str.isprintable` of one character: not in the categories Other (Cc, Cf, Cs, Co, Cn) nor
 * Separator (Zl, Zp, Zs), the space excepted.
 */
private fun printable(cp: Int): Boolean {
    if (cp == ' '.code) return true
    return when (Character.getType(cp).toByte()) {
        Character.CONTROL, Character.FORMAT, Character.SURROGATE, Character.PRIVATE_USE,
        Character.UNASSIGNED, Character.LINE_SEPARATOR, Character.PARAGRAPH_SEPARATOR,
        Character.SPACE_SEPARATOR,
        -> false
        else -> true
    }
}

/** `repr` of a JSON value: None, True, 1, 1.5, 'text', ['a', 1], {'k': None}. */
private fun pyReprValue(value: Any?): String = when (value) {
    null -> "None"
    is Boolean -> if (value) "True" else "False"
    is Double -> pyFloatRepr(value)
    is String -> pyRepr(value)
    is List<*> -> value.joinToString(", ", "[", "]") { pyReprValue(it) }
    is Map<*, *> -> value.entries.joinToString(", ", "{", "}") {
        "${pyReprValue(it.key)}: ${pyReprValue(it.value)}"
    }
    else -> value.toString() // an int
}

/** `repr(x)` of a float: the shortest digits, fixed from 1e-4 up to 1e16, else 1e+16 style. */
private fun pyFloatRepr(value: Double): String {
    if (value.isNaN()) return "nan"
    if (value.isInfinite()) return if (value > 0) "inf" else "-inf"
    if (value == 0.0) return if (1.0 / value < 0) "-0.0" else "0.0"
    val exact = BigDecimal(value.toString()).stripTrailingZeros()
    val digits = exact.unscaledValue().abs().toString()
    val exponent = digits.length - 1 - exact.scale() // the power of ten of the first digit
    val sign = if (value < 0) "-" else ""
    if (exponent < -4 || exponent >= 16) {
        val mantissa = if (digits.length == 1) digits else digits[0] + "." + digits.substring(1)
        val power = if (exponent < 0) "-" + (-exponent).toString().padStart(2, '0') else "+" + exponent.toString().padStart(2, '0')
        return "$sign${mantissa}e$power"
    }
    val plain = exact.abs().toPlainString()
    return sign + if ('.' in plain) plain else "$plain.0"
}

/** Python's JSON decoder: RFC 8259, plus NaN and the infinities; no raw control character in a string. */
private class PyJsonParser(private val s: String) {
    private var i = 0

    fun atEnd(): Boolean = i == s.length

    fun skip() {
        while (i < s.length && s[i] in " \t\n\r") i++
    }

    private fun fail(): Nothing = throw PyJsonError("Expecting value at char $i")

    fun value(): Any? {
        if (i >= s.length) fail()
        return when (s[i]) {
            '{' -> obj()
            '[' -> array()
            '"' -> string()
            't' -> word("true", true)
            'f' -> word("false", false)
            'n' -> word("null", null)
            'N' -> word("NaN", Double.NaN)
            'I' -> word("Infinity", Double.POSITIVE_INFINITY)
            '-' -> if (s.startsWith("-Infinity", i)) word("-Infinity", Double.NEGATIVE_INFINITY) else number()
            else -> number()
        }
    }

    private fun word(text: String, value: Any?): Any? {
        if (!s.startsWith(text, i)) fail()
        i += text.length
        return value
    }

    private fun obj(): Map<String, Any?> {
        i++ // {
        val out = LinkedHashMap<String, Any?>()
        skip()
        if (i < s.length && s[i] == '}') {
            i++
            return out
        }
        while (true) {
            skip()
            if (i >= s.length || s[i] != '"') fail()
            val key = string()
            skip()
            if (i >= s.length || s[i] != ':') fail()
            i++
            skip()
            out[key] = value()
            skip()
            if (i >= s.length) fail()
            when (s[i]) {
                ',' -> i++
                '}' -> {
                    i++
                    return out
                }
                else -> fail()
            }
        }
    }

    private fun array(): List<Any?> {
        i++ // [
        val out = ArrayList<Any?>()
        skip()
        if (i < s.length && s[i] == ']') {
            i++
            return out
        }
        while (true) {
            skip()
            out.add(value())
            skip()
            if (i >= s.length) fail()
            when (s[i]) {
                ',' -> i++
                ']' -> {
                    i++
                    return out
                }
                else -> fail()
            }
        }
    }

    private fun string(): String {
        i++ // the opening quote
        val out = StringBuilder()
        while (i < s.length) {
            val c = s[i]
            when {
                c == '"' -> {
                    i++
                    return out.toString()
                }
                c == '\\' -> {
                    if (i + 1 >= s.length) fail()
                    when (val e = s[i + 1]) {
                        'u' -> {
                            if (i + 6 > s.length) fail()
                            val digits = s.substring(i + 2, i + 6)
                            if (!digits.all { it in "0123456789abcdefABCDEF" }) fail()
                            out.append(digits.toInt(16).toChar())
                            i += 6
                        }
                        '"', '\\', '/' -> { out.append(e); i += 2 }
                        'b' -> { out.append('\b'); i += 2 }
                        'f' -> { out.append('\u000c'); i += 2 }
                        'n' -> { out.append('\n'); i += 2 }
                        'r' -> { out.append('\r'); i += 2 }
                        't' -> { out.append('\t'); i += 2 }
                        else -> fail()
                    }
                }
                c.code < 0x20 -> fail()
                else -> {
                    out.append(c)
                    i++
                }
            }
        }
        fail()
    }

    /** `-?(?:0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?`: an int without a fraction or exponent. */
    private fun number(): Any {
        val start = i
        if (i < s.length && s[i] == '-') i++
        if (i >= s.length) fail()
        if (s[i] == '0') {
            i++
        } else if (s[i] in '1'..'9') {
            while (i < s.length && s[i] in '0'..'9') i++
        } else {
            fail()
        }
        var float = false
        if (i + 1 < s.length && s[i] == '.' && s[i + 1] in '0'..'9') {
            float = true
            i++
            while (i < s.length && s[i] in '0'..'9') i++
        }
        if (i < s.length && (s[i] == 'e' || s[i] == 'E')) {
            var j = i + 1
            if (j < s.length && (s[j] == '+' || s[j] == '-')) j++
            if (j < s.length && s[j] in '0'..'9') {
                while (j < s.length && s[j] in '0'..'9') j++
                float = true
                i = j
            }
        }
        val text = s.substring(start, i)
        if (float) return text.toDouble()
        return text.toLongOrNull() ?: BigInteger(text)
    }
}
