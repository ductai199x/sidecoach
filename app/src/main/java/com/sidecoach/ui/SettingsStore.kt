package com.sidecoach.ui

import android.content.Context
import com.sidecoach.analysis.Thresholds
import com.sidecoach.coach.PolicySettings
import com.sidecoach.llm.LLMSettings
import org.json.JSONArray
import org.json.JSONObject

/**
 * The settings, as the panel's settings page reads and writes them.
 *
 * On a computer they are a TOML file; on a phone there is no file to edit by hand, so they
 * are one JSON document in the app's own storage, over the defaults the desktop's
 * `config.py` gives (shipped by `tools/sync_settings.py` with the schema the page renders
 * from). What the page gets is therefore the same shape it gets on a computer: the values,
 * the schema, where they live, and which keys are set.
 *
 * The API key goes in a store of its own, never in the settings document -- the same rule as
 * the desktop's, where the key is in the environment or the credential store and the TOML
 * holds only the name of the variable. The app's storage is private to it and the manifest
 * turns backups off, so nothing carries the key off the phone.
 */
/**
 * `config.EngineSettings`, the fields the phone's engine is built from. The defaults are the
 * phone's own (tools/sync_settings.py writes them into the defaults asset): two threads and
 * 64 MB where the desktop has four and 256, for the cores and the heat a phone can spare.
 */
data class EngineSettings(
    val multipv: Int = 6,
    val threads: Int = 2,
    val hashMb: Int = 64,
    val depth: Int = 18,
    val pvPlies: Int = 12,
    val scoreModel: String = "human",
    val maxSeconds: Double? = 5.0,
)

class SettingsStore(private val context: Context) {

    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val keys = context.getSharedPreferences("keys", Context.MODE_PRIVATE)

    private val defaults: JSONObject by lazy {
        JSONObject(read("settings/defaults.json") ?: "{}")
    }

    val schema: String by lazy { read("settings/schema.json") ?: "{}" }

    val keyEnvs: JSONObject by lazy { JSONObject(read("settings/key_envs.json") ?: "{}") }

    private fun read(asset: String): String? = try {
        context.assets.open(asset).bufferedReader(Charsets.UTF_8).use { it.readText() }
    } catch (_: Exception) {
        null
    }

    /** The settings the phone starts from, every field in the settings' own order. */
    fun defaults(): JSONObject = JSONObject(defaults.toString())

    /** The settings as they stand: the defaults, with whatever was saved over them. */
    fun values(): JSONObject {
        val out = JSONObject(defaults.toString())
        val saved = prefs.getString("json", null) ?: return withPaths(out)
        val over = runCatching { JSONObject(saved) }.getOrNull() ?: return withPaths(out)
        merge(out, over)
        return withPaths(out)
    }

    /** Where the phone keeps what it keeps; the page shows these and cannot change them. */
    private fun withPaths(out: JSONObject): JSONObject {
        out.put("data_dir", context.filesDir.absolutePath)
        out.put("tools_dir", context.applicationInfo.nativeLibraryDir)
        return out
    }

    /** One section of the settings, or an empty object. */
    fun section(name: String): JSONObject = values().optJSONObject(name) ?: JSONObject()

    fun save(values: JSONObject) {
        prefs.edit().putString("json", values.toString()).apply()
    }

    fun secretNames(): List<String> {
        val names = linkedSetOf("OPENAI_API_KEY", "DASHSCOPE_API_KEY")
        val custom = section("llm").optString("api_key_env")
        if (custom.isNotEmpty()) names.add(custom)
        return names.toList()
    }

    fun secretSet(name: String): Boolean = !keys.getString(name, null).isNullOrEmpty()

    fun secret(name: String): String? = keys.getString(name, null)

    fun setSecret(name: String, value: String?) {
        val editor = keys.edit()
        if (value.isNullOrEmpty()) editor.remove(name) else editor.putString(name, value)
        editor.apply()
    }

    /** The `[llm]` section as the factory wants it. */
    fun llm(): LLMSettings {
        val o = section("llm")
        return LLMSettings(
            provider = o.optString("provider", "none"),
            model = nullable(o, "model"),
            baseUrl = nullable(o, "base_url"),
            apiKeyEnv = nullable(o, "api_key_env"),
            thinkingControl = nullable(o, "thinking_control"),
            reasoningEffort = nullable(o, "reasoning_effort"),
            routineEffort = nullable(o, "routine_effort"),
            pricePromptPerM = number(o, "price_prompt_per_m"),
            priceCachedPerM = number(o, "price_cached_per_m"),
            priceCompletionPerM = number(o, "price_completion_per_m"),
            timeoutS = o.optDouble("timeout_s", 60.0),
            extraBody = plain(o.optJSONObject("extra_body")) as? Map<String, Any?> ?: emptyMap(),
        )
    }

    /** JSON as the provider sends it: maps, lists and values (`llm.extra_body`). */
    private fun plain(value: Any?): Any? = when (value) {
        null, JSONObject.NULL -> null
        is JSONObject -> value.keys().asSequence().associateWith { plain(value.opt(it)) }
        is JSONArray -> (0 until value.length()).map { plain(value.opt(it)) }
        else -> value
    }

    /** How often the screen is read, from `capture.fps`; the service takes it at Start. */
    fun captureFps(): Int =
        section("capture").optInt("fps", com.sidecoach.capture.CaptureService.TARGET_FPS).coerceIn(1, 30)

    /** The `[coach]` section the interjection policy takes. */
    fun policy(): PolicySettings {
        val o = section("coach")
        val base = PolicySettings()
        return base.copy(
            stuckSeconds = o.optDouble("stuck_seconds", base.stuckSeconds),
            quiet = o.optBoolean("quiet", base.quiet),
        )
    }

    /** The `engine` section, as `_make_analyser` reads it. */
    fun engine(): EngineSettings {
        val o = section("engine")
        val base = EngineSettings()
        return EngineSettings(
            multipv = o.optInt("multipv", base.multipv),
            threads = o.optInt("threads", base.threads),
            hashMb = o.optInt("hash_mb", base.hashMb),
            depth = o.optInt("depth", base.depth),
            pvPlies = o.optInt("pv_plies", base.pvPlies),
            scoreModel = o.optString("score_model", base.scoreModel),
            maxSeconds = if (o.has("max_seconds") && o.isNull("max_seconds")) {
                null
            } else {
                number(o, "max_seconds") ?: base.maxSeconds
            },
        )
    }

    /** `Thresholds.from_settings(settings.coach)`. */
    fun thresholds(): Thresholds {
        val o = section("coach")
        val base = Thresholds()
        return Thresholds(
            inaccuracy = number(o, "inaccuracy") ?: base.inaccuracy,
            mistake = number(o, "mistake") ?: base.mistake,
            blunder = number(o, "blunder") ?: base.blunder,
            decisiveGap = number(o, "decisive_gap") ?: base.decisiveGap,
            onlyMoveGap = number(o, "only_move_gap") ?: base.onlyMoveGap,
            excellent = number(o, "excellent") ?: base.excellent,
            decided = number(o, "decided") ?: base.decided,
        )
    }

    /** What the settings page's `listen` holds: off, ptt or vad. */
    fun listen(): String = section("voice").optString("listen", "off")

    /** The voice the coach speaks in, how fast, in which language: `settings.voice`. */
    fun voice(): String = section("voice").optString("voice", "af_sarah")

    /** The coach's words over other apps ([Captions]); the phone's own setting. */
    fun captions(): Boolean = section("voice").optBoolean("captions", false)

    /** Whether the captions' box stays on screen between lines ([Captions.keep]). */
    fun captionsKeep(): Boolean = section("voice").optBoolean("captions_keep", false)

    /** How the captions' box looks; a colour that does not read keeps the default. */
    fun captionStyle(): Captions.Style {
        val voice = section("voice")
        val plain = Captions.Style()
        fun colour(name: String, fallback: Int): Int =
            runCatching { android.graphics.Color.parseColor(voice.optString(name)) }.getOrDefault(fallback)
        return Captions.Style(
            text = colour("captions_text_color", plain.text),
            background = colour("captions_background", plain.background),
            opacityPercent = voice.optInt("captions_opacity", plain.opacityPercent).coerceIn(20, 100),
            sizeSp = voice.optInt("captions_text_size", plain.sizeSp.toInt()).coerceIn(12, 32).toFloat(),
        )
    }

    fun speed(): Float = (number(section("voice"), "speed") ?: 1.0).toFloat()

    fun lang(): String = section("voice").optString("lang", "en-us")

    fun persona(): String = section("coach").optString("persona", "warm_direct")

    fun rating(): Pair<Int, Int> {
        val o = section("profile")
        return o.optInt("rating_low", 1000) to o.optInt("rating_high", 1400)
    }

    fun summariseEvery(): Int = section("coach").optInt("summarise_every", 0)

    fun compactRecord(): Boolean = section("coach").optBoolean("compact_record", false)

    fun verbalizedSampling(): Boolean = section("coach").optBoolean("verbalized_sampling", false)

    fun localVerdicts(): Boolean = section("coach").optBoolean("local_verdicts", true)

    fun verify(): String = section("coach").optString("verify", "")

    /** The model the second opinion uses, when a provider is named: `coach.verify_model`. */
    fun verifyModel(): String? = nullable(section("coach"), "verify_model")

    private fun nullable(o: JSONObject, key: String): String? =
        if (o.isNull(key)) null else o.optString(key).ifEmpty { null }

    private fun number(o: JSONObject, key: String): Double? =
        if (o.isNull(key)) null else o.optDouble(key).takeIf { !it.isNaN() }

    /** The saved values over the defaults, section by section. */
    private fun merge(into: JSONObject, from: JSONObject) {
        for (key in from.keys()) {
            val value = from.opt(key)
            val existing = into.opt(key)
            if (value is JSONObject && existing is JSONObject) {
                merge(existing, value)
            } else {
                into.put(key, value)
            }
        }
    }
}
