package com.sidecoach.vision

import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * The few numpy operations the board detector's comb relies on, computed the way numpy
 * computes them on the desktop, to the last bit: the comb's placement is an argmax over
 * sums of these, and a tie numpy breaks one way is broken the same way here only when both
 * sides add in the same order.
 */
internal object Numpy {

    /**
     * `np.arange(start, stop, step)`: `ceil((stop - start) / step)` values, the first
     * `start`, the second `start + step`, and the i-th `start + i * delta` for the step as
     * the first two differ (`delta = (start + step) - start`), as numpy fills them.
     */
    fun arange(start: Double, stop: Double, step: Double): DoubleArray {
        if (step <= 0.0) return DoubleArray(0)
        val n = max(0.0, ceil((stop - start) / step)).toInt()
        if (n == 0) return DoubleArray(0)
        val out = DoubleArray(n)
        out[0] = start
        if (n > 1) out[1] = start + step
        val delta = out.getOrElse(1) { start } - start
        for (i in 2 until n) out[i] = start + i * delta
        return out
    }

    /**
     * `np.interp(x, np.arange(len(profile)), profile, left=0.0, right=0.0)`: nothing before
     * the first entry or past the last, and between two entries numpy's own formula,
     * `slope * (x - j) + profile[j]`.
     */
    fun interp(profile: DoubleArray, x: Double): Double {
        val last = profile.size - 1
        if (x < 0.0 || x > last) return 0.0
        if (x == last.toDouble()) return profile[last]
        val j = x.toInt()
        val slope = profile[j + 1] - profile[j]
        return slope * (x - j) + profile[j]
    }

    /**
     * numpy's sum of a float64 array: in order below eight values, eight running sums up to
     * 128 (added in pairs at the end, the rest one by one), and the halves apart above that.
     */
    fun sum(values: DoubleArray, from: Int = 0, n: Int = values.size - from): Double {
        if (n < 8) {
            var total = 0.0
            for (i in from until from + n) total += values[i]
            return total
        }
        if (n <= 128) {
            val r = DoubleArray(8) { values[from + it] }
            var i = 8
            while (i < n - n % 8) {
                for (j in 0 until 8) r[j] += values[from + i + j]
                i += 8
            }
            var total = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
            while (i < n) {
                total += values[from + i]
                i++
            }
            return total
        }
        var half = n / 2
        half -= half % 8
        return sum(values, from, half) + sum(values, from + half, n - half)
    }

    /** `np.bartlett(m)`: the triangle window, 0 at both ends and 1 in the middle. */
    fun bartlett(m: Int): DoubleArray = DoubleArray(m) {
        val n = (1 - m + 2 * it).toDouble()
        if (n <= 0.0) 1.0 + n / (m - 1) else 1.0 - n / (m - 1)
    }

    /**
     * `np.convolve(a, kernel, mode="same")` for a symmetric kernel shorter than `a`.
     *
     * numpy sums each output's products in order, but two ways: where the kernel lies wholly
     * on `a` and has at most eleven taps, plainly (its `small_correlate`); everywhere else
     * through BLAS's dot, which on the desktop fuses each multiply with its add. The fused
     * order is exact here up to fifteen taps; from sixteen products BLAS sums in blocks whose
     * order is its processor's own, and this sums them in order (a kernel that long comes
     * only from cells over 200 px).
     */
    fun convolveSame(a: DoubleArray, kernel: DoubleArray): DoubleArray {
        val n = a.size
        val taps = kernel.size
        val left = taps / 2
        val out = DoubleArray(n)
        for (i in 0 until n) {
            val first = max(0, left - i)
            val last = min(taps, n + left - i)
            val whole = first == 0 && last == taps
            var total = 0.0
            if (whole && taps <= 11) {
                for (j in first until last) total += a[i - left + j] * kernel[j]
            } else {
                for (j in first until last) total = Math.fma(a[i - left + j], kernel[j], total)
            }
            out[i] = total
        }
        return out
    }

    /**
     * The indices a Python slice `[start:stop]` takes of a sequence of [n]: a negative bound
     * counts from the end, and both are clipped to it.
     */
    fun slice(start: Int, stop: Int, n: Int): IntRange {
        val lo = if (start < 0) max(0, start + n) else min(start, n)
        val hi = if (stop < 0) max(0, stop + n) else min(stop, n)
        return lo until hi
    }
}
