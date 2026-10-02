package com.sidecoach.coach

/**
 * Who "you" is: a port of `sidecoach/coach/person.py`.
 *
 * The harness is written for the coach's eyes, so in it "you" is the coach, the person being
 * coached is "the player" and the other side is "the opponent". The facts and ideas are first
 * written for the side to move as "you"/"your" (that is how they read when the coach says
 * them aloud), so the harness rewrites them into the third person before it shows them: for
 * the player's turn "you" is the player; for the opponent's turn "you" is the opponent and
 * "your opponent" is the player. The coach's own speech is not touched.
 *
 * The rewrite is one pass over the text. A table applied pattern by pattern turned "your
 * opponent's" into "your" and then, on the next pattern, into "their" -- the opponent's pawn
 * became the player's.
 */
object Person {

    /**
     * One alternation, longest forms first: "your opponent's" before "your", "you are" before
     * a bare "you". A subject ("you"/"they") takes its verb along so the verb can agree.
     */
    private val PRONOUNS = Regex(
        "(?i)\\b(?:" +
            "(?<otherPoss>your opponent's|theirs|their)" +
            "|(?<other>your opponent|them)" +
            "|(?<ownPoss>yours|your)" +
            "|(?<subject>you|they)" +
            "(?: (?<adverb>still|now|just|already|also|only|never|no longer))?" +
            "(?: (?<verb>[a-z]+))?" +
            ")\\b",
    )

    private val MODALS = setOf(
        "can", "cannot", "could", "would", "should", "must", "may", "might", "will", "did", "had",
    )

    /** Words that follow a subject without being its verb: "you, the player" or "they and". */
    private val NOT_VERBS = setOf(
        "the", "a", "an", "it", "them", "him", "her", "and", "or", "but", "to", "of", "on",
        "in", "at", "by", "for", "with", "from", "up", "down", "back", "both", "each", "all",
        // Words after "you" that are never its verb: "it gives you a pawn instead", "costs you
        // twice" became "the player insteads", "the player twices" (the deep prompt review,
        // 2026-10-01).
        "instead", "twice", "better", "worse", "more", "less", "too", "again", "first", "here",
        "there", "nothing", "something", "enough", "time", "tempo", "material", "trouble",
        "much", "little", "plenty", "room", "options", "chances", "any", "some", "no", "one",
        "two", "three", "extra", "free", "away", "out", "over", "into", "onto", "off",
    )

    /**
     * The word before "you" that makes it an object: a preposition, or a verb taking one --
     * "it attacks you twice", "this leaves you a pawn down".
     */
    private val OBJECT_AFTER = setOf(
        "to", "for", "with", "against", "than", "from", "at", "on", "by", "of", "into",
        "onto", "toward", "towards", "behind", "past", "around", "near", "beside", "like",
        "give", "gives", "leave", "leaves", "let", "lets", "cost", "costs", "get", "gets",
        "make", "makes", "offer", "offers", "hand", "hands", "show", "shows", "tell", "tells",
        "ask", "asks", "attack", "attacks", "hit", "hits", "check", "checks", "pin", "pins",
        "fork", "forks", "threaten", "threatens", "help", "helps", "allow", "allows",
        "force", "forces", "beat", "beats", "hurt", "hurts", "trap", "traps", "keep", "keeps",
        "put", "puts", "send", "sends", "stop", "stops", "deny", "denies", "answer", "answers",
    )
    private val WORD_BEFORE = Regex("([A-Za-z']+)[^A-Za-z']*$")

    /** Verbs ending in -ed that are present tense, not past. */
    private val PRESENT_ED = setOf("need", "succeed", "proceed", "exceed", "feed", "bleed", "speed", "shed")

    /** The verb of "you <verb>" as it reads after "the player": have -> has, win -> wins. */
    private fun conjugate(verb: String): String {
        val low = verb.lowercase()
        if (low in MODALS || low == "is" || low == "was" || low == "has") return verb
        if (low == "are") return "is"
        if (low == "were") return "was"
        if (low == "have") return "has"
        if (low == "do") return "does"
        if (low == "don't") return "doesn't"
        if (low.endsWith("ed") && low !in PRESENT_ED) return verb  // past tense stays
        if (low.endsWith("s") || low.endsWith("sh") || low.endsWith("ch") ||
            low.endsWith("x") || low.endsWith("z") || low.endsWith("o")
        ) {
            return verb + "es"
        }
        if (low.endsWith("y") && low.length > 1 && low[low.length - 2] !in "aeiou") {
            return verb.dropLast(1) + "ies"
        }
        return verb + "s"
    }

    /**
     * A fact written for the side to move as "you", reworded for the harness. `side` is who
     * "you" is: "player" (the default) or "opponent".
     *
     * "your king is still in the centre" -> "the player's king is still in the centre"; "you
     * win a pawn" -> "the player wins a pawn"; with side="opponent", "your opponent's isolated
     * pawn" -> "the player's isolated pawn".
     */
    fun thirdPerson(text: String, side: String = "player"): String {
        val own = if (side == "player") "the player" else "the opponent"
        val other = if (side == "player") "the opponent" else "the player"

        val out = PRONOUNS.replace(text) { match ->
            val otherPoss = match.groups["otherPoss"]?.value
            if (!otherPoss.isNullOrEmpty()) return@replace "$other's"
            val otherWord = match.groups["other"]?.value
            if (!otherWord.isNullOrEmpty()) return@replace other
            val ownPoss = match.groups["ownPoss"]?.value
            if (!ownPoss.isNullOrEmpty()) return@replace "$own's"
            val subjectWord = match.groups["subject"]!!.value
            val subject = if (subjectWord.lowercase() == "you") own else other
            val adverb = match.groups["adverb"]?.value
            val verb = match.groups["verb"]?.value
            val built = StringBuilder(subject)
            if (!adverb.isNullOrEmpty()) built.append(" ").append(adverb)
            val before = WORD_BEFORE.find(text.substring(0, match.range.first))
            val isObject = before != null && before.groupValues[1].lowercase() in OBJECT_AFTER
            if (!verb.isNullOrEmpty()) {
                val kept = verb.lowercase() in NOT_VERBS || isObject
                built.append(" ").append(if (kept) verb else conjugate(verb))
            }
            built.toString()
        }
        // A text that began with a capital ("Your king ...") keeps one ("The player's king ...").
        return if (text.isNotEmpty() && text[0].isUpperCase()) {
            out.substring(0, 1).uppercase() + out.substring(1)
        } else {
            out
        }
    }
}
