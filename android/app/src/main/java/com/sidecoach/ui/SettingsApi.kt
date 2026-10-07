package com.sidecoach.ui

import com.sidecoach.profile.Library
import com.sidecoach.profile.Store
import com.sidecoach.voice.Py
import com.sidecoach.voice.Voices
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException

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
class SettingsApi(
    val store: SettingsStore,
    /** The player's record, whose games follow the usernames ([refreshLibrary]). */
    private val profile: File = File(store.files, "profile.sqlite"),
) {

    companion object {
        /** The usernames in the settings, for the games that follow them. */
        fun userNames(store: SettingsStore): List<String> {
            val profile = store.section("profile")
            return listOf(profile.optString("chesscom_user", ""), profile.optString("lichess_user", ""))
                .map { Py.strip(it) }.filter { it.isNotEmpty() }
        }

        /**
         * The games' fields that follow the usernames, as the settings read now
         * ([Library.refresh]): a game brought in before the username was set gets the player's
         * side. The desktop's background review refreshes at every look; the phone has none, so
         * it is done where the usernames can change -- a save that changes one, and the app's
         * start -- not only when Stats or Your games is opened: until then the coach's biggest
         * cost and Stats left such games out (the review of 2026-10-01).
         */
        fun refreshLibrary(store: SettingsStore, profile: File = File(store.files, "profile.sqlite")) {
            val record = Store(profile)
            try {
                Library.refresh(record, userNames(store))
            } finally {
                record.close()
            }
        }

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

        /**
         * Each Settings page's latest save applied, by its count, guarded by [SAVING]. The page
         * sends a save as it closes (the phone hides it at every app switch) beside the one on its
         * way; the two can land in either order, and the older one, landing last, put back what
         * the later one changed (the audit of 2026-10-02). The later save carries every change
         * since the last acknowledged, so the older one has nothing it lacks. `_applied`.
         */
        val APPLIED = HashMap<String, Long>()
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
        // The Settings page that sent it and its count of saves (`SettingsUpdate.page`, `seq`).
        val page = if (body.isNull("page")) null else body.optString("page", "").ifEmpty { null }
        val seq = if (body.isNull("seq") || !body.has("seq")) null else body.optLong("seq")
        val (after, changed) = synchronized(SAVING) {
            if (page != null && seq != null && seq <= (APPLIED[page] ?: 0L)) {
                // An older save of the page, landing after a later one: the later one had it all.
                return JSONObject().apply {
                    put("ok", true)
                    put("path", store.values().optString("data_dir"))
                    put("changed", JSONArray())
                    put("now", JSONArray())
                    put("next", JSONArray())
                    put("restart", JSONArray())
                    put("secrets", JSONObject())
                    put("superseded", true)
                }
            }
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
            // One service's model is not another's: kept, OpenAI was asked for the local
            // server's model (the API review, 2026-10-01). Each service keeps its own: the one
            // left is remembered, and the one switched to comes back with the model last chosen
            // for it, else empty, its default (Tai, 2026-10-02). `settings_api._validated`.
            val llmGiven = given.optJSONObject("llm")
            val llmWanted = wanted.optJSONObject("llm")
            val llmBefore = before.optJSONObject("llm") ?: JSONObject()
            val was = llmBefore.optString("provider")
            if (llmGiven != null && llmWanted != null && llmGiven.has("provider") &&
                llmGiven.optString("provider") != was
            ) {
                val models = JSONObject()
                val kept = llmBefore.optJSONObject("models") ?: JSONObject()
                for (name in kept.keys()) {
                    val model = kept.optString(name)
                    if (!kept.isNull(name) && model.isNotEmpty()) models.put(name, model)
                }
                val current = if (llmBefore.isNull("model")) "" else llmBefore.optString("model")
                if (current.isNotEmpty()) models.put(was, current) else models.remove(was)
                llmWanted.put("models", models)
                if (!llmGiven.has("model")) {
                    llmWanted.put("model", models.opt(llmGiven.optString("provider")) ?: JSONObject.NULL)
                }
            }
            dropHeld(wanted)
            // The phone owns these; the page shows them and cannot move them.
            wanted.put("data_dir", before.optString("data_dir"))
            wanted.put("tools_dir", before.optString("tools_dir"))
            try {
                store.save(wanted)
            } catch (e: IOException) {
                // Not kept: said to the page, never a crash (`settings_api.update`'s OSError).
                return JSONObject().put("ok", false).put("error", "The settings could not be saved: ${e.message}")
            }
            if (page != null && seq != null) APPLIED[page] = seq
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
        if ("profile.chesscom_user" in changed || "profile.lichess_user" in changed) {
            // The player's side of the games brought in before the username was set.
            try {
                refreshLibrary(store, profile)
            } catch (e: Exception) {
                PanelEngine.emit("sys", "warn", "your games could not follow the new username yet: $e")
            }
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
     * A field the check refuses keeps the value it had, and the log says so: it returned
     * without a word, and the change was not kept (the settings review, 2026-10-01); what else
     * the change carried is kept. `settings_api.remember`.
     */
    fun remember(change: JSONObject) {
        synchronized(SAVING) {
            val before = store.values()
            val checked = SettingsCheck.validate(JSONObject(store.schema), before, change)
            val wanted = checked.values
            for (i in 0 until checked.errors.length()) {
                val error = checked.errors.getJSONObject(i)
                val loc = error.getJSONArray("loc").let { a -> (0 until a.length()).map { a.getString(it) } }
                if (loc.size == 1) {
                    wanted.put(loc[0], before.opt(loc[0]))
                } else if (loc.size == 2) {
                    wanted.optJSONObject(loc[0])?.put(loc[1], before.optJSONObject(loc[0])?.opt(loc[1]))
                }
                PanelEngine.emit("sys", "warn", "${loc.joinToString(".")}: not kept for the next session (${error.optString("msg")})")
            }
            dropHeld(wanted)
            wanted.put("data_dir", before.optString("data_dir"))
            wanted.put("tools_dir", before.optString("tools_dir"))
            try {
                store.save(wanted)
            } catch (e: IOException) {
                // The switch has moved; only its keeping failed, and the log says so
                // (`settings_api.remember`'s OSError).
                PanelEngine.emit("sys", "warn", "${names(change)}: not kept for the next session (${e.message})")
                return
            }
            PanelEngine.settingsTouched()
        }
    }

    /**
     * A value the settings hold and refuse, which the change does not set, put back to its
     * default, and the log says so: left, it refused every later save and mute for a field
     * nobody had sent (the settings review, 2026-10-01). Only a field the change itself sets is
     * refused; those were checked before this. `settings_api._validated` with `drop_refused`.
     */
    private fun dropHeld(wanted: JSONObject) {
        val dropped = SettingsCheck.dropRefused(JSONObject(store.schema), store.defaults(), wanted)
        if (dropped.isNotEmpty()) {
            PanelEngine.emit("sys", "warn", "settings: ${dropped.joinToString(", ")} could not be kept, back at the default")
        }
    }

    /** The fields a change sets, `section.field`, as the log names them. */
    private fun names(change: JSONObject): String = change.keys().asSequence().flatMap { key ->
        val section = change.opt(key)
        if (section is JSONObject) section.keys().asSequence().map { "$key.$it" } else sequenceOf(key)
    }.joinToString(", ")

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

    /**
     * The fields that changed, `section.field`: a section is gone into, a field is not -- an
     * object field (llm.extra_body) is one field, as its mark is. Gone into, its keys came out
     * as names with no mark, and the change was told "nothing changed" (the settings review,
     * 2026-10-01). `settings_api._changed`.
     */
    private fun changed(old: JSONObject, new: JSONObject, prefix: String = ""): List<String> {
        val out = ArrayList<String>()
        val names = linkedSetOf<String>()
        for (key in old.keys()) names.add(key)
        for (key in new.keys()) names.add(key)
        for (key in names) {
            val a = old.opt(key)
            val b = new.opt(key)
            if (prefix.isEmpty() && a is JSONObject && b is JSONObject) {
                out.addAll(changed(a, b, "$key."))
            } else if (!same(a, b)) {
                out.add("$prefix$key")
            }
        }
        return out.sorted()
    }

    /**
     * Two JSON values equal as Python's == finds them: objects by their keys and values in any
     * order, lists in order, numbers by value (1 is 1.0, True is 1). Compared as text,
     * llm.extra_body re-sent with its keys in another order counted as changed and rebuilt the
     * coach's model for nothing (the review of 2026-10-01).
     */
    private fun same(a: Any?, b: Any?): Boolean {
        val x = if (a == JSONObject.NULL) null else a
        val y = if (b == JSONObject.NULL) null else b
        return when {
            x == null || y == null -> x == null && y == null
            x is JSONObject && y is JSONObject -> {
                val keys = x.keys().asSequence().toSet()
                keys == y.keys().asSequence().toSet() && keys.all { same(x.opt(it), y.opt(it)) }
            }
            x is JSONArray && y is JSONArray -> x.length() == y.length() && (0 until x.length()).all { same(x.opt(it), y.opt(it)) }
            (x is Number || x is Boolean) && (y is Number || y is Boolean) -> {
                if (x is Boolean && y is Boolean) return x == y
                fun value(v: Any): Double = if (v is Boolean) (if (v) 1.0 else 0.0) else (v as Number).toDouble()
                val bothWhole = (x is Int || x is Long || x is Boolean) && (y is Int || y is Long || y is Boolean)
                if (bothWhole) {
                    fun whole(v: Any): Long = if (v is Boolean) (if (v) 1L else 0L) else (v as Number).toLong()
                    whole(x) == whole(y)
                } else {
                    value(x) == value(y)
                }
            }
            else -> x == y
        }
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
                    if (refused != null) errors.put(refused) else section.put(field, trimmed(loc, checked.value))
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
     * Each field of [values] (a whole settings document) the settings refuse put back to its
     * value in [defaults], in place, as the desktop's `drop_refused` takes it out of the data so
     * it takes its default; the others stay. Checked again after each round, as
     * `config._read_settings` and `settings_api._validated` validate again; returns the dotted
     * names put back (`section.field`, or a top-level one).
     */
    fun dropRefused(schema: JSONObject, defaults: JSONObject, values: JSONObject): List<String> {
        val dropped = ArrayList<String>()
        repeat(values.length() + 1) { // each round puts at least one field back
            val errors = validate(schema, defaults, values).errors
            var took = false
            for (i in 0 until errors.length()) {
                val loc = errors.getJSONObject(i).getJSONArray("loc").let { a -> (0 until minOf(a.length(), 2)).map { a.get(it).toString() } }
                if (loc.isEmpty()) continue
                val holder = if (loc.size == 1) values else values.optJSONObject(loc[0]) ?: continue
                val fallback = if (loc.size == 1) defaults.opt(loc[0]) else defaults.optJSONObject(loc[0])?.opt(loc[1])
                if (holder.has(loc.last()) && holder.opt(loc.last())?.toString() == fallback?.toString()) continue // already the default
                holder.put(loc.last(), copy(fallback))
                dropped.add(loc.joinToString("."))
                took = true
            }
            if (!took) return dropped
        }
        return dropped
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

    /** The settings' usernames (`ProfileSettings._trimmed`). */
    val USERNAMES = setOf("chesscom_user", "lichess_user")

    /**
     * A username without the spaces around it: a space after a pasted name went into the address
     * the games are fetched from, and the site knew no such player (the settings review,
     * 2026-10-01). `ProfileSettings._trimmed`.
     */
    private fun trimmed(loc: List<String>, value: Any?): Any? =
        if (loc.size == 2 && loc[0] == "profile" && loc[1] in USERNAMES && value is String) Py.strip(value) else value

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
        // A JSON document cannot hold a number that is not finite: "inf" in a field with no
        // bound made the save throw, a 500 (the review of 2026-10-01). Refused as a field,
        // in pydantic's words for it, and before the bounds, as the desktop's settings
        // (allow_inf_nan off) refuse it.
        val x = number.toDouble()
        if (x.isNaN() || x.isInfinite()) return Checked(null, error("finite_number", loc, "Input should be a finite number", value))
        bounds(inner, number, value, loc)?.let { return it }
        return coerced
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
