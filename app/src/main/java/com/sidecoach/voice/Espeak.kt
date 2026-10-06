package com.sidecoach.voice

import android.content.Context
import java.io.File

/**
 * espeak-ng, as phonemizer drives it (`phonemizer.backend.espeak.wrapper.EspeakWrapper` and
 * `api.EspeakAPI`): the library built from 1.52.0 with the NDK (`app/src/main/cpp`), the
 * desktop's compiled English data from the app's assets, one voice chosen by its language
 * code, and text turned into IPA clause by clause.
 *
 * espeak-ng keeps its state in globals, so every call goes through [lock] -- kokoro-onnx's
 * `_espeak_lock` -- and the library is initialised once for the process.
 */
object Espeak {

    /** Phonemizer holds its espeak calls under one lock; so do we. */
    val lock = Any()

    private var initialised = false
    private var voiceCode: String? = null

    @JvmStatic private external fun initialize(dataPath: ByteArray): Int
    @JvmStatic private external fun info(): ByteArray
    @JvmStatic private external fun listVoices(): Array<ByteArray>
    @JvmStatic private external fun setVoiceByName(name: ByteArray): Int
    @JvmStatic private external fun currentVoiceName(): ByteArray
    @JvmStatic private external fun textToPhonemes(text: ByteArray, textMode: Int, phonemesMode: Int): ByteArray

    /** A voice as phonemizer lists it (`EspeakVoice`). */
    data class Voice(val name: String, val language: String, val identifier: String)

    /**
     * Load the library and point it at the data, copied out of the assets the first time and
     * again after each install. `EspeakAPI.__init__`: espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS,
     * 0, data_path, 0). Call under [lock].
     */
    fun ensure(context: Context) {
        if (initialised) return
        try {
            System.loadLibrary("sidecoach_espeak")
        } catch (missing: UnsatisfiedLinkError) {
            // An Error, which the voice's threads do not catch: the app shut down after its
            // splash on every launch (0.1.80 and 0.1.81, built without the library). As an
            // exception it is the voice failing, which the voice already reports and survives.
            throw IllegalStateException("the espeak library is missing: ${missing.message}", missing)
        }
        val data = dataPath(context)
        if (initialize(data.path.toByteArray(Charsets.UTF_8)) <= 0) {
            throw IllegalStateException("failed to initialize espeak shared library")
        }
        initialised = true
    }

    /** The version as a tuple, as `EspeakWrapper.version` reads it from espeak_Info. */
    fun version(): List<Int> {
        val text = String(info(), Charsets.UTF_8).trim().split(" ")[0].replace("-dev", "")
        return text.split(".").map { it.toInt() }
    }

    /** `EspeakWrapper.available_voices()`: every voice espeak lists, in its order. */
    fun availableVoices(): List<Voice> {
        val raw = listVoices()
        return (raw.indices step 3).map { i ->
            Voice(
                name = String(raw[i], Charsets.UTF_8).replace("_", " "),
                language = String(raw[i + 1], Charsets.UTF_8),
                identifier = String(raw[i + 2], Charsets.UTF_8),
            )
        }
    }

    /**
     * `EspeakWrapper.set_voice(voice_code)` for an espeak (not mbrola) voice: the first voice
     * listed under the language code, the mbrola variants skipped. Call under [lock].
     */
    fun setVoice(code: String) {
        if (voiceCode == code) return
        val available = LinkedHashMap<String, String>()
        for (voice in availableVoices()) {
            if (voice.identifier.replace("\\", "/").startsWith("mb/")) continue
            if (voice.language !in available) available[voice.language] = voice.identifier
        }
        val name = available[code] ?: throw IllegalArgumentException("invalid voice code \"$code\"")
        if (setVoiceByName(name.toByteArray(Charsets.UTF_8)) != 0) {
            throw IllegalStateException("failed to load voice \"$code\"")
        }
        if (currentVoiceName().isEmpty()) throw IllegalStateException("failed to load voice \"$code\"")
        voiceCode = code
    }

    /**
     * `EspeakWrapper.text_to_phonemes(text, tie=False)`: IPA with '_' between phonemes and ' '
     * between words, the clauses joined by spaces. Call under [lock], after [setVoice].
     */
    fun textToPhonemes(text: String): String {
        // UTF-8 in (text mode 1); IPA out with '_' separating phonemes: ord("_") << 8 | 0x02.
        val mode = ('_'.code shl 8) or 0x02
        return String(textToPhonemes(text.toByteArray(Charsets.UTF_8), 1, mode), Charsets.UTF_8)
    }

    /**
     * The data directory: the assets' `espeak-ng-data` copied into the app's files, again
     * whenever the app was installed or updated since (its data may have changed with it).
     */
    private fun dataPath(context: Context): File {
        val target = File(context.filesDir, "espeak-ng-data")
        val stamp = File(target, ".installed")
        val installed = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime.toString()
        if (stamp.exists() && stamp.readText() == installed) return target
        target.deleteRecursively()
        copyAssets(context, "espeak-ng-data", target)
        stamp.writeText(installed)
        return target
    }

    private fun copyAssets(context: Context, from: String, to: File) {
        val children = context.assets.list(from).orEmpty()
        if (children.isEmpty()) {
            to.parentFile?.mkdirs()
            context.assets.open(from).use { input -> to.outputStream().use { input.copyTo(it) } }
            return
        }
        to.mkdirs()
        for (child in children) copyAssets(context, "$from/$child", File(to, child))
    }
}
