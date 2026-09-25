package com.sidecoach.coach

/**
 * The coach's wordings, drawn like marbles from a bag: every wording of a template once before
 * any comes again, and a fresh bag never opens with the one just said (Tai, 2026-09-25: "are you
 * doing taking out a marbles in a bag without replacement? probably should do that for anything
 * that has multiple templates -- it's a small thing but important UX"). "Random, but never the
 * last one" still said A, B, A, B while C, D and E waited. A port of `sidecoach/coach/bag.py`.
 *
 * One bag per key: a template ("cost", "keep"), a label's praise ("praise:good"), a lesson's
 * clauses ("clause:develop"), the first hint's lead ("lead:tactic"). A wording no longer among
 * the options is dropped from its bag; an empty bag is refilled and shuffled.
 *
 * A wording written ahead of time -- the first hint is in every position's facts, and said only
 * in the few where the player asks -- is [peek]ed when it is written and [take]n when it is said,
 * so the hints never said do not use up the bag.
 *
 * The bags of one coach (a game's worth of wordings, or a session's). [shuffle] is the desktop's
 * `rng.shuffle`, in place, injected so a test can fix the order as the desktop's `rng` is.
 */
class Bags(private val shuffle: (MutableList<String>) -> Unit = { it.shuffle() }) {

    private val bags = HashMap<String, MutableList<String>>()
    private val last = HashMap<String, String>()

    /** One of [options], from [key]'s bag, said now. */
    fun draw(key: String, options: List<String>): String {
        val chosen = peek(key, options)
        take(key, chosen)
        return chosen
    }

    /** The wording [key]'s next draw gives, left in the bag. */
    fun peek(key: String, options: List<String>): String {
        val wordings = options.distinct()
        require(wordings.isNotEmpty()) { "no wordings for '$key'" }
        if (wordings.size == 1) return wordings[0]
        var bag = (bags[key] ?: emptyList<String>()).filter { it in wordings }.toMutableList()
        if (bag.isEmpty()) {
            bag = wordings.toMutableList()
            shuffle(bag)
            // Drawn from the end: a new bag never opens with the wording just said.
            if (bag.last() == last[key]) {
                val first = bag[0]
                bag[0] = bag[bag.size - 1]
                bag[bag.size - 1] = first
            }
        }
        bags[key] = bag
        return bag.last()
    }

    /** [wording] said: out of [key]'s bag until the bag is refilled. */
    fun take(key: String, wording: String) {
        bags[key]?.remove(wording)
        last[key] = wording
    }

    /** The wordings still in [key]'s bag: what the desktop's tests read of `_bags`. */
    internal fun left(key: String): List<String> = bags[key]?.toList() ?: emptyList()
}
