package com.sidecoach.ui

import com.sidecoach.voice.Py
import com.sidecoach.voice.Voices
import org.json.JSONArray
import org.json.JSONObject

/**
 * The settings over HTTP, for the panel's settings page. `sidecoach/ui/settings_api.py`.
 *
 * GET gives the settings, the JSON schema the page renders from (every field's description,
 * its choices, and whether a change applies now, at the next Start or after a restart),
 * where they live, which keys are set, and the personas. PUT validates what the page sends
 * against the schema ([SettingsCheck], as pydantic's `Settings.model_validate` does on the
 * desktop), saves it, stores the keys in the store of their own, and answers with what took
 * effect now, what waits for the next Start and what needs a restart -- or, when something
 * does not fit, with the errors by field and nothing saved.
 *
 * The voices are the phone's own (`Voices.english`, the list the panel's drawer shows). The
 * schema is the phone's (`tools/sync_settings.py`): a field the phone does not honour is not
 * in it, so the page never shows one that does nothing (Tai, 2026-09-24: the Voice setting
 * listed one voice, and the page showed the desktop's devices, folders and GPU switches).
 */
class SettingsApi(val store: SettingsStore) {

    companion object {
        /**
         * One write of the settings at a time, each whole -- read, merge, save, apply: a
         * Settings save, a switch on the panel kept for the next session (muted, how the coach
         * listens), the captions turned off from their own box. Each read the settings, laid its
         * change over them and saved, and the later save put back what the earlier one changed
         * (the settings review, 2026-10-01: the desktop lost 81 usernames in 150 rounds). A JVM
         * monitor is re-entrant: a panel switch holds it around its own change and the
         * remembering. `settings_api.saving`.
         */
        val SAVING = Any()
    }

    fun describe(): JSONObject = JSONObject().apply {
        put("settings", store.values())
        put("schema", JSONObject(store.schema))
        put("path", store.values().optString("data_dir"))
        put(
            "secrets",
            JSONObject().apply {
                for (name in store.secretNames()) put(name, store.secretSet(name))
            },
        )
        // The keys are in the app's own storage, which nothing else on the phone can read.
        put("keyring", true)
        put("looks", Looks.catalogue(store.files))
        put("key_envs", store.keyEnvs)
        put("devices", JSONObject().apply { put("inputs", JSONArray()); put("outputs", JSONArray()) })
        put("voices", JSONArray(Voices.english(PanelVoice.tts?.voicesPath)))
        put(
            "personas",
            JSONObject().apply {
                for ((name, pair) in com.sidecoach.coach.PERSONAS) put(name, pair.first)
            },
        )
    }

    fun update(body: JSONObject): JSONObject {
        val given = body.optJSONObject("settings") ?: return JSONObject().apply {
            put("ok", false)
            put("errors", JSONArray().put(JSONObject().put("msg", "no settings in the request")))
        }
        val stored = JSONObject()
        val (after, changed) = synchronized(SAVING) {
            val before = store.values()
            // What changed, laid over the settings as they are now (`settings_api.update`'s
            // `_merged`): the page sends only what changed, and laid over the defaults that reset
            // every other setting on each save (Tai's phone, 2026-10-01: a save of the captions
            // wiped all of his settings, and each caption switch saved alone reset the other).
            val checked = SettingsCheck.validate(JSONObject(store.schema), before, given)
            if (checked.errors.length() > 0) {
                return JSONObject().put("ok", false).put("errors", checked.errors)
            }
            val wanted = checked.values
            // The phone owns these; the page shows them and cannot move them.
            wanted.put("data_dir", before.optString("data_dir"))
            wanted.put("tools_dir", before.optString("tools_dir"))
            store.save(wanted)
            val after = store.values()
            val changed = changed(before, after)

            val secrets = body.optJSONObject("secrets")
            if (secrets != null) {
                val names = store.secretNames()
                for (name in secrets.keys()) {
                    if (name !in names) continue
                    val value = if (secrets.isNull(name)) null else secrets.optString(name)
                    store.setSecret(name, value)
                    stored.put(name, true)
                }
            }

            // What the kept coach and engine can change, they do now; a "now" field with nothing
            // built to change is simply what the next build reads (`settings_api.update`).
            PanelWarm.applySettings(store, changed)
            PanelEngine.settingsTouched()
            after to changed
        }
        val applies = applies(JSONObject(store.schema))
        val now = changed.filter { applies[it] == "now" }
        val later = changed.filter { applies[it] == "next" }
        val restart = changed.filter { applies[it] == "restart" }
        val keys = stored.keys().asSequence().toList()

        PanelEngine.emit(
            "sys",
            "info",
            "settings saved" +
                (if (now.isNotEmpty()) "; applied now: " + words(now) else "") +
                (if (later.isNotEmpty()) "; at the next Start: " + words(later) else "") +
                (if (restart.isNotEmpty()) "; after a restart: " + words(restart) else "") +
                (if (keys.isNotEmpty()) "; keys: " + keys.joinToString(", ") else "") +
                (if (now.isEmpty() && later.isEmpty() && restart.isEmpty() && keys.isEmpty()) "; nothing changed" else ""),
        )
        // The model is rebuilt for a change of its own fields or of a key: a key saved alone
        // changed no field, and the coach kept the model it was built with -- offline for want of
        // the key, or the old one -- until a restart (the settings review, 2026-10-01). Outside
        // the lock: the model is asked whether it answers, a minute for a server that does not,
        // and a mute pressed meanwhile does not wait for that; it builds from the settings as
        // they are when it runs, so of two saves the later one's model is the one kept.
        val rebuild = modelFields(changed) + keys
        if (rebuild.isNotEmpty()) PanelWarm.reprovision(store, rebuild)
        return JSONObject().apply {
            put("ok", true)
            put("path", after.optString("data_dir"))
            put("changed", JSONArray(changed))
            put("now", JSONArray(now))
            put("next", JSONArray(later))
            put("restart", JSONArray(restart))
            put("secrets", stored)
        }
    }

    /**
     * A choice made on the panel itself -- muted, how the coach listens -- kept in the settings,
     * so the next session starts as this one was left (Tai, 2026-10-01: "mute setting isn't
     * remembered between sessions"). The panel applied it already; nothing is applied again.
     * `settings_api.remember`.
     */
    fun remember(change: JSONObject) {
        synchronized(SAVING) {
            val before = store.values()
            val checked = SettingsCheck.validate(JSONObject(store.schema), before, change)
            if (checked.errors.length() > 0) return
            val wanted = checked.values
            wanted.put("data_dir", before.optString("data_dir"))
            wanted.put("tools_dir", before.optString("tools_dir"))
            store.save(wanted)
            PanelEngine.settingsTouched()
        }
    }

    private fun words(names: List<String>): String =
        names.joinToString(", ") { it.substringAfterLast('.').replace("_", " ") }

    /** The changed fields the coach's model is built from. `settings_api._model_fields`. */
    private fun modelFields(changed: List<String>): List<String> = changed.filter {
        (it.startsWith("llm.") && !it.startsWith("llm.price_")) || it == "coach.verify" || it == "coach.verify_model"
    }

    /** "section.field" -> now | next | restart, from the schema's own marks. */
    private fun applies(schema: JSONObject): Map<String, String> {
        val out = HashMap<String, String>()
        val defs = schema.optJSONObject("\$defs") ?: JSONObject()
        val properties = schema.optJSONObject("properties") ?: JSONObject()
        for (name in properties.keys()) {
            val prop = properties.optJSONObject(name) ?: continue
            val ref = prop.optString("\$ref")
            if (ref.isNotEmpty()) {
                val model = defs.optJSONObject(ref.substringAfterLast('/')) ?: continue
                val fields = model.optJSONObject("properties") ?: continue
                for (field in fields.keys()) {
                    val one = fields.optJSONObject(field) ?: continue
                    out["$name.$field"] = one.optString("applies").ifEmpty { "next" }
                }
            } else {
                out[name] = prop.optString("applies").ifEmpty { "next" }
            }
        }
        return out
    }

    private fun changed(old: JSONObject, new: JSONObject, prefix: String = ""): List<String> {
        val out = ArrayList<String>()
        val names = linkedSetOf<String>()
        for (key in old.keys()) names.add(key)
        for (key in new.keys()) names.add(key)
        for (key in names) {
            val a = old.opt(key)
            val b = new.opt(key)
            if (a is JSONObject && b is JSONObject) {
                out.addAll(changed(a, b, "$prefix$key."))
            } else if (a?.toString() != b?.toString()) {
                out.add("$prefix$key")
            }
        }
        return out.sorted()
    }
}

/**
 * What pydantic's `Settings.model_validate` makes of a save, against the phone's schema: each
 * field of the page coerced as pydantic's lax mode coerces it (a number in a string, a boolean
 * written as "yes") and held to its bounds, or an error in pydantic's own shape and words --
 * `{"type", "loc", "msg", "input", "ctx"}`, which the page shows by field. The fields are
 * checked in the settings' order, so the errors come in pydantic's. A field the phone's page
 * does not show is kept as it is; one the settings do not have is dropped, as pydantic
 * ignores it. Like pydantic, the choices a field lists are not enforced: they are the page's.
 */
object SettingsCheck {

    class Result(val values: JSONObject, val errors: JSONArray)

    /** What one field came to: its value, or the error that refused it. */
    private class Checked(val value: Any?, val error: JSONObject? = null)

    /** [given] checked and laid over [under] (the settings as they are): what it leaves out stays. */
    fun validate(schema: JSONObject, under: JSONObject, given: JSONObject): Result {
        val defs = schema.optJSONObject("\$defs") ?: JSONObject()
        val properties = schema.optJSONObject("properties") ?: JSONObject()
        val values = JSONObject()
        val errors = JSONArray()
        for (name in under.keys()) {
            val base = under.opt(name)
            if (!given.has(name)) {
                values.put(name, copy(base))
                continue
            }
            val value = given.opt(name)
            val prop = properties.optJSONObject(name)
            val ref = prop?.optString("\$ref").orEmpty()
            if (ref.isNotEmpty() && base is JSONObject) {
                val model = ref.substringAfterLast('/')
                if (value !is JSONObject) {
                    errors.put(
                        error(
                            "model_type",
                            listOf(name),
                            "Input should be a valid dictionary or instance of $model",
                            value,
                            JSONObject().put("class_name", model),
                        ),
                    )
                    continue
                }
                val fields = defs.optJSONObject(model)?.optJSONObject("properties") ?: JSONObject()
                val section = JSONObject()
                for (field in base.keys()) {
                    if (!value.has(field)) {
                        section.put(field, copy(base.opt(field)))
                        continue
                    }
                    val fprop = fields.optJSONObject(field)
                    if (fprop == null) {
                        // Not on the phone's page: kept as it came.
                        section.put(field, value.opt(field))
                        continue
                    }
                    val loc = listOf(name, field)
                    val checked = check(fprop, value.opt(field), loc)
                    val refused = checked.error ?: known(loc, checked.value)
                    if (refused != null) errors.put(refused) else section.put(field, checked.value)
                }
                values.put(name, section)
            } else if (prop != null) {
                val checked = check(prop, value, listOf(name))
                if (checked.error != null) errors.put(checked.error) else values.put(name, checked.value)
            } else {
                values.put(name, value)
            }
        }
        return Result(values, errors)
    }

    /**
     * The fields pydantic checks with a validator of their own: the board's pieces, a shipped
     * set or one of the player's own (`BoardSettings._known_pieces`), and its theme
     * (`_known_theme`). The page offers only those; a name it does not is refused as there.
     */
    private fun known(loc: List<String>, value: Any?): JSONObject? {
        if (loc.size != 2 || loc[0] != "board" || value !is String) return null
        val wrong = when (loc[1]) {
            "pieces" -> if (value !in Looks.PIECE_SETS && Looks.ownId(value) == null) "no piece set named ${repr(value)}" else null
            "theme" -> if (value !in Looks.BOARD_THEMES) "no board theme named ${repr(value)}" else null
            else -> null
        } ?: return null
        return error("value_error", loc, "Value error, $wrong", value, JSONObject().put("error", JSONObject()))
    }

    /** Python's repr of a string: single quotes, double ones when it holds a single and no double. */
    private fun repr(s: String): String {
        val quote = if ('\'' in s && '"' !in s) '"' else '\''
        val body = s.replace("\\", "\\\\").replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
        val inner = if (quote == '\'') body.replace("'", "\\'") else body
        return "$quote$inner$quote"
    }

    private fun copy(value: Any?): Any? = when (value) {
        is JSONObject -> JSONObject(value.toString())
        is JSONArray -> JSONArray(value.toString())
        else -> value
    }

    private fun error(type: String, loc: List<String>, msg: String, input: Any?, ctx: JSONObject? = null): JSONObject =
        JSONObject().apply {
            put("type", type)
            put("loc", JSONArray(loc))
            put("msg", msg)
            put("input", input ?: JSONObject.NULL)
            if (ctx != null) put("ctx", ctx)
        }

    /** One field: its type (with `null` when the schema allows it), then its bounds. */
    private fun check(prop: JSONObject, value: Any?, loc: List<String>): Checked {
        val anyOf = prop.optJSONArray("anyOf")
        var inner = prop
        if (anyOf != null) {
            val options = (0 until anyOf.length()).mapNotNull { anyOf.optJSONObject(it) }
            val nullable = options.any { it.optString("type") == "null" }
            if (nullable && (value == null || value == JSONObject.NULL)) return Checked(JSONObject.NULL)
            inner = options.firstOrNull { it.optString("type") != "null" } ?: return Checked(value)
        }
        val coerced = when (inner.optString("type")) {
            "integer" -> integer(value, loc)
            "number" -> number(value, loc)
            "boolean" -> boolean(value, loc)
            "string" -> if (value is String) Checked(value) else Checked(null, error("string_type", loc, "Input should be a valid string", value))
            "object" -> if (value is JSONObject) Checked(value) else Checked(null, error("dict_type", loc, "Input should be a valid dictionary", value))
            else -> Checked(value)
        }
        if (coerced.error != null) return coerced
        val number = coerced.value as? Number ?: return coerced
        return bounds(inner, number, value, loc) ?: coerced
    }

    /** pydantic-core's order: at most, below, at least, above. */
    private fun bounds(prop: JSONObject, number: Number, input: Any?, loc: List<String>): Checked? {
        val x = number.toDouble()
        fun failed(key: String, type: String, words: String): Checked? {
            val bound = prop.opt(key) as? Number ?: return null
            val ok = when (key) {
                "maximum" -> x <= bound.toDouble()
                "exclusiveMaximum" -> x < bound.toDouble()
                "minimum" -> x >= bound.toDouble()
                else -> x > bound.toDouble()
            }
            if (ok) return null
            val name = when (key) {
                "maximum" -> "le"
                "exclusiveMaximum" -> "lt"
                "minimum" -> "ge"
                else -> "gt"
            }
            return Checked(null, error(type, loc, "Input should be $words ${shown(bound)}", input, JSONObject().put(name, bound)))
        }
        return failed("maximum", "less_than_equal", "less than or equal to")
            ?: failed("exclusiveMaximum", "less_than", "less than")
            ?: failed("minimum", "greater_than_equal", "greater than or equal to")
            ?: failed("exclusiveMinimum", "greater_than", "greater than")
    }

    /** A bound as pydantic-core writes it in a message: 60.0 as "60", 0.5 as "0.5". */
    private fun shown(bound: Number): String {
        val x = bound.toDouble()
        return if (x == Math.floor(x) && !x.isInfinite() && Math.abs(x) < 1e15) x.toLong().toString() else x.toString()
    }

    /** A whole number in a string: digits, underscores between them, a sign; ".0" allowed. */
    private val INT_TEXT = Regex("[+-]?\\d+(_\\d+)*(\\.0+)?")

    /** Python's float syntax: digits with underscores, a point, an exponent; inf and nan. */
    private val FLOAT_TEXT = Regex(
        "[+-]?((\\d+(_\\d+)*(\\.(\\d+(_\\d+)*)?)?|\\.\\d+(_\\d+)*)([eE][+-]?\\d+(_\\d+)*)?|inf|infinity|nan)",
        RegexOption.IGNORE_CASE,
    )

    private fun integer(value: Any?, loc: List<String>): Checked = when (value) {
        is Boolean -> Checked(if (value) 1 else 0)
        is Int, is Long -> Checked(value)
        is Number -> {
            val x = value.toDouble()
            when {
                x.isNaN() || x.isInfinite() -> Checked(null, error("finite_number", loc, "Input should be a finite number", value))
                x != Math.floor(x) -> Checked(
                    null,
                    error("int_from_float", loc, "Input should be a valid integer, got a number with a fractional part", value),
                )
                else -> Checked(x.toLong())
            }
        }
        is String -> {
            val text = Py.strip(value)
            if (INT_TEXT.matches(text) && text.all { it.code < 128 }) {
                Checked(text.substringBefore('.').replace("_", "").toLong())
            } else {
                Checked(null, error("int_parsing", loc, "Input should be a valid integer, unable to parse string as an integer", value))
            }
        }
        else -> Checked(null, error("int_type", loc, "Input should be a valid integer", value))
    }

    private fun number(value: Any?, loc: List<String>): Checked = when (value) {
        is Boolean -> Checked(if (value) 1.0 else 0.0)
        is Number -> Checked(value.toDouble())
        is String -> {
            val text = Py.strip(value)
            if (FLOAT_TEXT.matches(text) && text.all { it.code < 128 }) {
                val plain = text.replace("_", "").lowercase()
                val sign = if (plain.startsWith("-")) -1.0 else 1.0
                val body = plain.trimStart('+', '-')
                Checked(
                    when (body) {
                        "inf", "infinity" -> sign * Double.POSITIVE_INFINITY
                        "nan" -> Double.NaN
                        else -> plain.toDouble()
                    },
                )
            } else {
                Checked(null, error("float_parsing", loc, "Input should be a valid number, unable to parse string as a number", value))
            }
        }
        else -> Checked(null, error("float_type", loc, "Input should be a valid number", value))
    }

    private val TRUE_TEXT = setOf("1", "on", "t", "true", "y", "yes")
    private val FALSE_TEXT = setOf("0", "off", "f", "false", "n", "no")

    private fun boolean(value: Any?, loc: List<String>): Checked {
        val parsing = error("bool_parsing", loc, "Input should be a valid boolean, unable to interpret input", value)
        return when (value) {
            is Boolean -> Checked(value)
            is Int, is Long -> when ((value as Number).toLong()) {
                0L -> Checked(false)
                1L -> Checked(true)
                else -> Checked(null, parsing)
            }
            is Number -> when (value.toDouble()) {
                0.0 -> Checked(false)
                1.0 -> Checked(true)
                else -> Checked(null, error("bool_type", loc, "Input should be a valid boolean", value))
            }
            is String -> when (value.lowercase()) {
                in TRUE_TEXT -> Checked(true)
                in FALSE_TEXT -> Checked(false)
                else -> Checked(null, parsing)
            }
            else -> Checked(null, error("bool_type", loc, "Input should be a valid boolean", value))
        }
    }
}
