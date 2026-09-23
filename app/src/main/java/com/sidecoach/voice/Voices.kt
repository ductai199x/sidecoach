package com.sidecoach.voice

import java.io.File
import java.util.zip.ZipFile

/**
 * The voices a person can pick: Kokoro's English ones, American and British, female and male,
 * and the label each is shown under ("American Female, Sarah"). `sidecoach/voice/voices.py`.
 *
 * The list comes from the voices file when it is there (every English voice the model ships),
 * else from the names Kokoro v1.0 is known to carry, so the picker is never empty before the
 * first download. Nothing else is offered: the coach speaks English.
 */
object Voices {
    /** Kokoro v1.0's English voices, in the order the picker shows them. */
    val ENGLISH_VOICES = listOf(
        "af_heart", "af_sarah", "af_bella", "af_nicole", "af_nova", "af_sky", "af_alloy",
        "af_aoede", "af_jessica", "af_kore", "af_river",
        "am_adam", "am_michael", "am_echo", "am_liam", "am_eric", "am_fenrir", "am_onyx",
        "am_puck", "am_santa",
        "bf_emma", "bf_isabella", "bf_alice", "bf_lily",
        "bm_george", "bm_lewis", "bm_daniel", "bm_fable",
    )
    private val ACCENT = mapOf('a' to "American", 'b' to "British")
    private val SEX = mapOf('f' to "Female", 'm' to "Male")

    /** `af_`, `am_`, `bf_`, `bm_`: American or British, female or male. */
    fun isEnglish(name: String): Boolean =
        name.length > 3 && name[0] in ACCENT && name[1] in SEX && name[2] == '_'

    /** "af_sarah" -> "American Female, Sarah"; a name outside the scheme stays as it is. */
    fun label(name: String): String {
        if (!isEnglish(name)) return name
        val rest = name.substring(3)
        // str.capitalize(): the first character upper, the rest lower.
        val capitalised = if (rest.isEmpty()) rest else rest.substring(0, 1).uppercase() + rest.substring(1).lowercase()
        return "${ACCENT.getValue(name[0])} ${SEX.getValue(name[1])}, $capitalised"
    }

    /**
     * The English voices to offer: the file's, in the picker's order with any the list does not
     * know appended, else the known list.
     */
    fun english(voicesPath: File?): List<String> {
        val known = ENGLISH_VOICES
        if (voicesPath == null || !voicesPath.exists()) return known
        val present = try {
            names(voicesPath).filter { isEnglish(it) }.toSet()
        } catch (_: Exception) { // a file that will not open costs nothing but the list
            return known
        }
        if (present.isEmpty()) return known
        return known.filter { it in present } + (present - known.toSet()).sorted()
    }

    /** The voices in a Kokoro voices file (an .npz: one .npy per voice), `NpzFile.files`. */
    fun names(voicesPath: File): List<String> = ZipFile(voicesPath).use { zip ->
        zip.entries().asSequence().map { it.name }.map { if (it.endsWith(".npy")) it.dropLast(4) else it }.toList()
    }
}
