package com.sidecoach.vision

/**
 * numpy's arithmetic where the vision depends on its exact numbers: the indices an even
 * thinning keeps, and half precision.
 */

/**
 * `np.linspace(0, n - 1, count).astype(int)`: [count] indices spread evenly over [n] items,
 * the first and the last among them. numpy multiplies each position by the step and sets the
 * last to the stop exactly; computing the last one by the step too can land a hair under it,
 * which truncation turns into the one before (the parity audit, 2026-09-26).
 */
internal fun linspaceIndices(n: Int, count: Int): IntArray {
    if (count <= 0) return IntArray(0)
    if (count == 1) return intArrayOf(0)
    val stop = (n - 1).toDouble()
    val step = stop / (count - 1)
    return IntArray(count) { if (it == count - 1) n - 1 else (it.toDouble() * step).toInt() }
}

/**
 * The bits of the half-precision number nearest [value], ties to even: numpy's
 * `astype(np.float16)`. Too large a value becomes infinity and a NaN stays one.
 */
internal fun halfBits(value: Float): Short {
    val bits = java.lang.Float.floatToRawIntBits(value)
    val sign = (bits ushr 16) and 0x8000
    val magnitude = bits and 0x7fffffff
    if (magnitude >= 0x7f800000) {
        // Infinity, or a NaN that keeps a mantissa bit.
        return (sign or 0x7c00 or (if (magnitude > 0x7f800000) 0x200 else 0)).toShort()
    }
    if (magnitude >= 0x477ff000) return (sign or 0x7c00).toShort() // 65520 and over: infinity
    if (magnitude < 0x38800000) {
        // Below the smallest normal half (2^-14): a subnormal half, in steps of 2^-24.
        val scaled = Math.rint(Math.abs(value).toDouble() * 16777216.0).toInt()
        return (sign or scaled).toShort()
    }
    // A normal half: thirteen mantissa bits dropped, rounding to nearest, ties to even; a
    // carry out of the mantissa moves the exponent, as it should.
    val lsb = (magnitude ushr 13) and 1
    val rounded = (magnitude + 0xfff + lsb) ushr 13
    return (sign or (rounded - ((127 - 15) shl 10))).toShort()
}

/** The float a half-precision number stands for, exactly. */
internal fun halfToFloat(half: Short): Float {
    val h = half.toInt() and 0xffff
    val sign = (h and 0x8000) shl 16
    val exponent = (h ushr 10) and 0x1f
    val mantissa = h and 0x3ff
    val bits = when {
        exponent == 0 && mantissa == 0 -> sign
        exponent == 0 -> {
            // A subnormal: mantissa times 2^-24, exact in a float.
            val magnitude = mantissa * (1.0f / 16777216.0f)
            return if (sign != 0) -magnitude else magnitude
        }
        exponent == 0x1f -> sign or 0x7f800000 or (mantissa shl 13)
        else -> sign or ((exponent + (127 - 15)) shl 23) or (mantissa shl 13)
    }
    return java.lang.Float.intBitsToFloat(bits)
}

/** [value] held at half precision, as numpy's `astype(np.float16)` holds it, read back. */
internal fun halfPrecision(value: Float): Float = halfToFloat(halfBits(value))
