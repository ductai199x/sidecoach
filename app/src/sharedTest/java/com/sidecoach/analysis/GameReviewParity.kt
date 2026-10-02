package com.sidecoach.analysis

import org.json.JSONArray
import org.json.JSONObject

/**
 * The JSON comparison of the review parity tests (GameReviewParityTest and ReviewAnalysisParityTest
 * on the host, LooksParityTest on the device): shared by both kinds of test.
 */
object GameReviewParity {
    private val SKIPPED = setOf("ts", "seconds")

    /** The two JSON values the same, numbers by value, `ts` and `seconds` left out. */
    fun same(path: String, mine: Any?, theirs: Any?, failures: MutableList<String>) {
        if (failures.size >= 40) return
        val a = if (mine == JSONObject.NULL) null else mine
        val b = if (theirs == JSONObject.NULL) null else theirs
        when {
            a is JSONObject && b is JSONObject -> {
                val keys = (a.keys().asSequence() + b.keys().asSequence()).toSortedSet() - SKIPPED
                for (k in keys) {
                    if (!a.has(k) || !b.has(k)) {
                        failures.add("$path.$k: ${if (a.has(k)) "only mine" else "only python's"}")
                    } else {
                        same("$path.$k", a.get(k), b.get(k), failures)
                    }
                }
            }
            a is JSONArray && b is JSONArray -> {
                if (a.length() != b.length()) failures.add("$path: ${a.length()} items, python ${b.length()}\n  mine:   $a\n  python: $b")
                else for (i in 0 until a.length()) same("$path[$i]", a.get(i), b.get(i), failures)
            }
            a is Number && b is Number -> if (a.toDouble() != b.toDouble()) failures.add("$path: mine $a, python $b")
            else -> if (a != b) failures.add("$path: mine $a, python $b")
        }
    }
}
