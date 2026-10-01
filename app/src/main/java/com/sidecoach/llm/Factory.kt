package com.sidecoach.llm

/**
 * Build the configured LLM provider: `llm/factory.py`.
 *
 * Any OpenAI-compatible endpoint works: a preset name fills in the endpoint, the model, the
 * key's environment variable and how reasoning is controlled; any other name is a custom
 * provider and needs `baseUrl` and `model` in the settings. Secrets live outside the
 * settings file, never in it.
 *
 * The desktop reads the key from `os.environ`; a phone has no environment to read, so the
 * lookup is a parameter -- the app passes the one that reads its credential store, and a
 * test passes its own.
 */

/** `config.LLMSettings`: the `[llm]` block of the desktop's settings. */
data class LLMSettings(
    /** Who the coach thinks with: openai | local | dashscope | custom | none. */
    val provider: String = "openai",
    /** The model's name; null uses the provider's default. */
    val model: String? = null,
    /** Where the model is served; null uses the provider's default. */
    val baseUrl: String? = null,
    /** The name the key is stored under; null uses the provider's default. */
    val apiKeyEnv: String? = null,
    /** How a preset switches reasoning on the wire; null uses the preset's own. */
    val thinkingControl: String? = null,
    /** How long the model may think before it answers. */
    val reasoningEffort: String? = null,
    /** A word on a fine move needs no deep thought; lessons and answers keep the effort above. */
    val routineEffort: String? = "none",
    val pricePromptPerM: Double? = null,
    val priceCachedPerM: Double? = null,
    val priceCompletionPerM: Double? = null,
    /** How long the coach waits for the model each time it asks, before it speaks from the board instead. */
    val timeoutS: Double = 60.0,
    /** Extra fields sent with every request, for a service that needs something unusual. */
    val extraBody: Map<String, Any?> = emptyMap(),
)

data class ProviderDefaults(
    val baseUrl: String,
    val model: String,
    val apiKeyEnv: String?,
    val thinkingControl: ThinkingControl,
    val reasoningEffort: String? = null,
    /** The API spoken ([Wire]): chat completions, or OpenAI's responses. */
    val wire: Wire = Wire.CHAT,
    /** A name that overrides baseUrl: a hosted provider's endpoint can be per account. */
    val baseUrlEnv: String? = null,
)

val PROVIDER_DEFAULTS: Map<String, ProviderDefaults> = mapOf(
    // LM Studio on a computer on the network. No key. Reasoning suppressed by prefill.
    "local" to ProviderDefaults(
        baseUrl = "http://localhost:1234/v1",
        model = "huihui-qwen3.8-27b-abliterated@q8_k_l",
        apiKeyEnv = null,
        thinkingControl = ThinkingControl.PREFILL,
    ),
    // Alibaba Cloud Model Studio, US region. qwen3.8-flash is the faster, cheaper option.
    "dashscope" to ProviderDefaults(
        baseUrl = "https://dashscope-us.aliyuncs.com/compatible-mode/v1",
        model = "qwen3.8-max",
        apiKeyEnv = "DASHSCOPE_API_KEY",
        thinkingControl = ThinkingControl.PARAM,
        baseUrlEnv = "DASHSCOPE_BASE_URL",
    ),
    // OpenAI's reasoning models: the effort is the model's setting, not the call's. This is
    // what the coach was evaluated on (docs/eval-results.md).
    "openai" to ProviderDefaults(
        baseUrl = "https://api.openai.com/v1",
        model = "gpt-5.6-luna",
        apiKeyEnv = "OPENAI_API_KEY",
        thinkingControl = ThinkingControl.EFFORT,
        reasoningEffort = "medium",
        // Its reasoning models take function tools with an effort only on /v1/responses.
        wire = Wire.RESPONSES,
        baseUrlEnv = "OPENAI_BASE_URL",
    ),
)

/** The controls as python prints the tuple, so the refusal reads the same. */
private val CONTROLS_SHOWN = THINKING_CONTROLS.joinToString(", ", "(", ")") { "'" + it + "'" }

/** Where a key or an endpoint override is read from: the app's store, or the environment. */
fun interface KeyStore {
    fun get(name: String): String?
}

val ENVIRONMENT = KeyStore { name -> System.getenv(name) }

/**
 * The name the key is read under: the settings', else the preset's; a custom service keeps its
 * key under CUSTOM_API_KEY unless the settings name another ("": none). `factory._key_env`.
 */
private fun keyEnvOf(settings: LLMSettings, defaults: ProviderDefaults?): String? = when {
    defaults == null -> settings.apiKeyEnv ?: "CUSTOM_API_KEY"
    settings.apiKeyEnv.isNullOrEmpty() -> defaults.apiKeyEnv
    else -> settings.apiKeyEnv
}

/**
 * The key a provider made from [settings] now would carry, or null: for a model kept and made
 * again when the key changes (Game Review's kept the old one until a restart, the settings
 * review 2026-10-01). `factory.api_key_of`.
 */
fun apiKeyOf(settings: LLMSettings, keys: KeyStore = ENVIRONMENT): String? {
    if (settings.provider == "none") return null
    val name = keyEnvOf(settings, PROVIDER_DEFAULTS[settings.provider])
    return if (!name.isNullOrEmpty()) keys.get(name) else null
}

/**
 * The coach's model: a preset, a custom endpoint, or none at all ([OfflineProvider]), where
 * the coach speaks from the board and its templates.
 */
fun makeProvider(
    settings: LLMSettings,
    timeoutS: Double? = null,
    keys: KeyStore = ENVIRONMENT,
    transport: Transport = UrlTransport(),
): LLMProvider {
    if (settings.provider == "none") return OfflineProvider()
    var defaults = PROVIDER_DEFAULTS[settings.provider]
    if (defaults == null) {
        // A custom endpoint: everything comes from the settings.
        if (settings.baseUrl.isNullOrEmpty() || settings.model.isNullOrEmpty()) {
            val known = PROVIDER_DEFAULTS.keys.sorted().joinToString(", ")
            throw IllegalArgumentException(
                "unknown llm.provider '${settings.provider}': use one of $known, or set " +
                    "llm.base_url and llm.model for a custom OpenAI-compatible endpoint"
            )
        }
        defaults = ProviderDefaults(
            baseUrl = settings.baseUrl,
            model = settings.model,
            // The app stores a custom service's key under CUSTOM_API_KEY; a service that
            // takes no key at all is the empty string.
            apiKeyEnv = settings.apiKeyEnv ?: "CUSTOM_API_KEY",
            thinkingControl = ThinkingControl.NONE,
        )
    }

    val keyEnv = keyEnvOf(settings, defaults)
    val apiKey = if (!keyEnv.isNullOrEmpty()) keys.get(keyEnv) else null
    if (!keyEnv.isNullOrEmpty() && apiKey.isNullOrEmpty()) {
        throw IllegalStateException("$keyEnv is not set: paste the key in Settings.")
    }

    val wanted = if (settings.thinkingControl.isNullOrEmpty()) {
        defaults.thinkingControl
    } else {
        ThinkingControl.of(settings.thinkingControl) ?: throw IllegalArgumentException(
            "llm.thinking_control must be one of $CONTROLS_SHOWN, " +
                "got '${settings.thinkingControl}'"
        )
    }
    val effort = if (settings.reasoningEffort.isNullOrEmpty()) {
        defaults.reasoningEffort
    } else {
        settings.reasoningEffort
    }

    val envUrl = defaults.baseUrlEnv?.let { keys.get(it) }
    return OpenAICompatProvider(
        settings.provider,
        settings.baseUrl?.ifEmpty { null } ?: envUrl?.ifEmpty { null } ?: defaults.baseUrl,
        settings.model?.ifEmpty { null } ?: defaults.model,
        apiKey = apiKey,
        thinkingControl = wanted,
        reasoningEffort = effort,
        wire = defaults.wire,
        timeoutS = timeoutS ?: settings.timeoutS,
        extraBody = settings.extraBody,
        transport = transport,
    )
}

/**
 * A provider from a short spec, for command lines: `local`, `dashscope`, `flash`
 * (dashscope's qwen3.8-flash), `openai`, `openai:<model>`, `openai:<model>@<effort>`, or
 * `<provider>:<model>` for any preset.
 */
fun providerFromSpec(
    spec: String,
    timeoutS: Double? = null,
    keys: KeyStore = ENVIRONMENT,
    transport: Transport = UrlTransport(),
): LLMProvider {
    if (spec == "flash") {
        return makeProvider(
            LLMSettings(provider = "dashscope", model = "qwen3.8-flash"),
            timeoutS,
            keys,
            transport,
        )
    }
    val colon = spec.indexOf(':')
    val provider = if (colon < 0) spec else spec.substring(0, colon)
    val rest = if (colon < 0) "" else spec.substring(colon + 1)
    val at = rest.indexOf('@')
    val model = if (at < 0) rest else rest.substring(0, at)
    val effort = if (at < 0) "" else rest.substring(at + 1)
    return makeProvider(
        LLMSettings(
            provider = provider,
            model = model.ifEmpty { null },
            reasoningEffort = effort.ifEmpty { null },
        ),
        timeoutS,
        keys,
        transport,
    )
}
